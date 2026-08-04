"""The runtime safety-placement policy: keep safety off-GPU when its charge cannot fit beside sampling.

The policy generalises the whole-card safety-off lever to the ordinary case. Demotion prices a modeled worst
case (device total, largest learned sampling peak, proportional noise buffer, the safety charge);
re-promotion instead reads the chosen card's measured device-free between allocation peaks, so it stays
satisfiable under sustained load rather than waiting for a sampling-free window that never comes. It only ever
degrades the operator's placement (GPU to CPU) and back, never beyond the operator's grant, and its
pause/restore is hysteresis-gated so a card oscillating around the fit boundary does not flap the safety
process on and off the card. Placement is headroom-aware across cards, not a fixed device 0.
"""

from __future__ import annotations

import sys
import uuid
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.vram_arbiter import VramRequest
from horde_worker_regen.process_management.resources.vram_footprints import (
    SAFETY_PROCESS_BASELINE,
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
)
from horde_worker_regen.process_management.scheduling import inference_scheduler as sched_mod
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _SAFETY_GPU_LOAD_CHARGE_MB,
    _SAFETY_PLACEMENT_PAUSE_STREAK,
    _SAFETY_PLACEMENT_RESTORE_STREAK,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _placement_scheduler(monkeypatch: pytest.MonkeyPatch, *, safety_on_gpu: bool = True):  # noqa: ANN202
    """A single-GPU scheduler whose safety process is placement-managed, with a mocked lifecycle.

    The CPU-only guard is patched off so the policy is active regardless of the test host: on a real CPU-only
    install safety is always off-GPU already, so the policy would (correctly) be inert.
    """
    monkeypatch.setattr(sched_mod, "is_cpu_only_install", lambda: False)
    bridge_data = make_mock_bridge_data(safety_on_gpu=safety_on_gpu)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({}), bridge_data=bridge_data)
    lifecycle = Mock()
    lifecycle.is_safety_gpu_paused = False
    lifecycle.safety_pause_owner = None
    lifecycle.safety_placement_transition_pending = False
    lifecycle.pause_safety_on_gpu = Mock(return_value=True)
    lifecycle.restore_safety_on_gpu = Mock(return_value=True)
    scheduler._process_lifecycle = lifecycle
    return scheduler


async def _queue_safety_backlog(scheduler: object, depth: int) -> None:
    """Place ``depth`` completed jobs in the pending safety stage."""
    for _ in range(depth):
        job = Mock()
        job.id_ = uuid.uuid4()
        job.model = "stable_diffusion"
        job_info = Mock()
        job_info.sdk_api_job_info = job
        await scheduler._job_tracker.queue_for_safety(job_info)  # type: ignore[attr-defined]


class TestSafetyFitArithmetic:
    """The structural fit is arithmetic over the device total and the largest active sampling peak."""

    def test_charge_fits_on_a_roomy_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A large card holds the safety charge beside a moderate peak, bare and with margin."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=24000.0)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=8192.0)
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=False) is True
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=True) is True

    def test_tight_card_bare_fit_but_no_margin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On a tight card the charge bare-fits but fails the proportional restore margin (hysteresis band)."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=11500.0)
        # 16000 - 11500 - 800 (5% noise) - 3044 (the safety seed) = 656 >= 0 bare; a second 800 margin makes
        # it negative.
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=False) is True
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=True) is False

    def test_nothing_sampling_fits_trivially(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no active sampling peak the charge trivially fits (nothing to fit beside)."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=None)
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=True) is True

    def test_unknown_total_fits_trivially(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unreported device total leaves the charge fitting (missing-telemetry admits)."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=None)
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=False) is True


def _observe_safety_footprint(scheduler: object, device_footprint_mb: float) -> None:
    """Fold one measured safety device footprint into the scheduler's learned-footprint store."""
    store = LearnedFootprintStore()
    store.observe_peak(
        FootprintKey(
            model_baseline=SAFETY_PROCESS_BASELINE,
            resolution_bucket=None,
            platform=sys.platform,
            stage=FootprintStage.SAFETY,
        ),
        device_footprint_mb,
    )
    scheduler.set_footprint_store(store)  # type: ignore[attr-defined]


class TestLearnedSafetyPrice:
    """The safety charge is the static seed raised by any measured SAFETY watermark, and nothing else."""

    def test_cold_store_prices_the_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With nothing measured (and with no store at all) the price is exactly the documented seed."""
        scheduler = _placement_scheduler(monkeypatch)
        assert scheduler._safety_footprint_mb() == _SAFETY_GPU_LOAD_CHARGE_MB
        scheduler.set_footprint_store(LearnedFootprintStore())
        assert scheduler._safety_footprint_mb() == _SAFETY_GPU_LOAD_CHARGE_MB

    def test_measured_footprint_above_the_seed_raises_the_price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A safety process measured heavier than the seed is priced at what it actually costs."""
        scheduler = _placement_scheduler(monkeypatch)
        _observe_safety_footprint(scheduler, _SAFETY_GPU_LOAD_CHARGE_MB + 1500.0)
        assert scheduler._safety_footprint_mb() == _SAFETY_GPU_LOAD_CHARGE_MB + 1500.0

    def test_measured_footprint_below_the_seed_never_lowers_the_price(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The overlay is raise-only: a lighter measurement leaves the conservative seed standing."""
        scheduler = _placement_scheduler(monkeypatch)
        _observe_safety_footprint(scheduler, _SAFETY_GPU_LOAD_CHARGE_MB - 1500.0)
        assert scheduler._safety_footprint_mb() == _SAFETY_GPU_LOAD_CHARGE_MB

    def test_learned_price_evicts_safety_from_a_card_the_seed_fit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A measured footprint the seed under-stated turns a modeled fit into a modeled non-fit."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=11500.0)
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=False) is True

        # The bare fit had 656 MB of slack at the seed; a measurement 1 GB above it consumes that slack.
        _observe_safety_footprint(scheduler, _SAFETY_GPU_LOAD_CHARGE_MB + 1024.0)
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=False) is False


class TestOneSafetyPrice:
    """Admission, placement, and the streaming forecast charge the identical figure for a given store state."""

    @pytest.mark.parametrize("learned_footprint_mb", [None, _SAFETY_GPU_LOAD_CHARGE_MB + 2000.0])
    def test_forecast_and_arbiter_charge_the_same_figure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        learned_footprint_mb: float | None,
    ) -> None:
        """The forecast's safety term and the arbiter's SAFETY_LOAD delta agree, seeded and learned alike.

        Two prices for one process is how admission and placement come to disagree about whether safety can
        sit on the card, so the pin reads both surfaces rather than the accessor they share: the forecast
        term is recovered as the achievable-free difference between safety on-GPU and safety paused, and the
        arbiter term is the delta the SAFETY_LOAD request actually carries.
        """
        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    0,
                    model_name=None,
                    state=HordeProcessState.WAITING_FOR_JOB,
                    process_type=HordeProcessType.SAFETY,
                ),
            },
        )
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_map = process_map
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=24000.0)
        if learned_footprint_mb is not None:
            _observe_safety_footprint(scheduler, learned_footprint_mb)

        expected_mb = learned_footprint_mb if learned_footprint_mb is not None else _SAFETY_GPU_LOAD_CHARGE_MB

        job = make_job_pop_response("Deliberate")
        scheduler._process_lifecycle.is_safety_gpu_paused = False
        on_gpu = scheduler._forecast_streaming(job, "stable_diffusion_1")
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        paused = scheduler._forecast_streaming(job, "stable_diffusion_1")
        assert on_gpu.free_after_model_evict_mb is not None
        assert paused.free_after_model_evict_mb is not None
        forecast_charge_mb = paused.free_after_model_evict_mb - on_gpu.free_after_model_evict_mb

        requests: list[VramRequest] = []
        arbiter = Mock()
        arbiter.has_cycle = True
        arbiter.evaluate = Mock(side_effect=lambda request: (requests.append(request), Mock(admits=True))[1])
        scheduler._vram_arbiter = arbiter
        scheduler._arbiter_admits_safety_gpu_load(None)

        assert forecast_charge_mb == expected_mb
        assert [request.candidate_delta_mb for request in requests] == [expected_mb]


class TestPlacementHysteresis:
    """The pause/restore latch turns on and off only after runs of consecutive non-fitting / fitting cycles."""

    def test_pauses_only_after_consecutive_misses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Safety is not evicted on a single miss; it takes the configured run of consecutive misses."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False

        for _ in range(_SAFETY_PLACEMENT_PAUSE_STREAK - 1):
            scheduler._reconcile_runtime_safety_placement()
            scheduler._process_lifecycle.pause_safety_on_gpu.assert_not_called()

        scheduler._reconcile_runtime_safety_placement()
        scheduler._process_lifecycle.pause_safety_on_gpu.assert_called_once_with(
            owner=PauseOwner.RUNTIME_SAFETY_PLACEMENT,
        )
        assert scheduler._safety_placement_wants_off is True

    def test_restores_only_after_consecutive_measured_headroom_cycles(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A paused-off safety is restored only after the longer run of measured-device-free-headroom cycles.

        The modeled charge stays non-fitting throughout (sustained load), proving the re-promotion is driven by
        the measured device-free signal, not the modeled one that is unsatisfiable while jobs flow.
        """
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False
        scheduler._safety_restore_headroom_fits = lambda device_index: True
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._process_lifecycle.safety_pause_owner = PauseOwner.RUNTIME_SAFETY_PLACEMENT
        scheduler._safety_placement_wants_off = True

        for _ in range(_SAFETY_PLACEMENT_RESTORE_STREAK - 1):
            scheduler._reconcile_runtime_safety_placement()
            scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()

        scheduler._reconcile_runtime_safety_placement()
        scheduler._process_lifecycle.restore_safety_on_gpu.assert_called_once_with(
            owner=PauseOwner.RUNTIME_SAFETY_PLACEMENT,
        )
        assert scheduler._safety_placement_wants_off is False
        assert scheduler._safety_placement_promotions == 1

    def test_config_false_never_promotes_to_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ``safety_on_gpu`` off the policy is inert: it never restores safety to the GPU."""
        scheduler = _placement_scheduler(monkeypatch, safety_on_gpu=False)
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._safety_placement_wants_off = True
        scheduler._safety_restore_headroom_fits = lambda device_index: True

        for _ in range(_SAFETY_PLACEMENT_RESTORE_STREAK + 2):
            scheduler._reconcile_runtime_safety_placement()

        scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()
        scheduler._process_lifecycle.pause_safety_on_gpu.assert_not_called()
        assert scheduler._safety_placement_wants_off is False

    def test_reconciled_restore_withheld_while_placement_wants_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A residency release does not fight the placement latch back on-GPU."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._process_lifecycle.safety_pause_owner = PauseOwner.WHOLE_CARD
        scheduler._safety_placement_wants_off = True

        scheduler._reconcile_runtime_safety_placement()

        scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()

    def test_unready_placement_transition_does_not_chain_a_restore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A slow healthy off-GPU respawn reaches readiness before a contrary placement wish may replace it."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._process_lifecycle.safety_pause_owner = PauseOwner.RUNTIME_SAFETY_PLACEMENT
        scheduler._process_lifecycle.safety_placement_transition_pending = True
        scheduler._safety_placement_wants_off = True
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False
        scheduler._safety_restore_headroom_fits = lambda device_index: True

        for _ in range(_SAFETY_PLACEMENT_RESTORE_STREAK + 3):
            scheduler._reconcile_runtime_safety_placement()

        assert scheduler._safety_placement_wants_off is False
        scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()


class TestPlacementRequestOwnership:
    """Residency and reclaim file placement demand; only the reconciler invokes the lifecycle actuator."""

    def test_held_residency_is_applied_as_a_whole_card_owned_pause(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A held residency becomes one owner-attributed pause when the recurring reconciler observes it."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._runtime_config.bridge_data.whole_card_residency_safety_off_gpu = True
        scheduler._whole_card_ledger.record_grant(
            None,
            model="heavy-model",
            forecast=None,
            cooldown_until=100.0,
            now=0.0,
            refresh_established=True,
        )

        scheduler._reconcile_runtime_safety_placement()

        scheduler._process_lifecycle.pause_safety_on_gpu.assert_called_once_with(owner=PauseOwner.WHOLE_CARD)

    def test_reclaim_request_is_applied_without_advancing_fit_hysteresis(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The ladder's one-shot request is actuated by the reconciler and does not count as a fit miss."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._runtime_config.bridge_data.whole_card_residency_safety_off_gpu = True
        scheduler._safety_fits_beside_largest_sampling_peak = Mock(return_value=False)

        assert scheduler.safety_off_gpu(None) is True

        scheduler._process_lifecycle.pause_safety_on_gpu.assert_called_once_with(owner=PauseOwner.RECLAIM_LADDER)
        assert scheduler._safety_placement_miss_streak == 0


class TestResidencyRestoreRespectsPlacementWish:
    """Both owners of safety's placement must agree before safety goes back on the card.

    The scheduler reconciler consumes both the residency veto and the runtime placement latch before it may put
    safety back on the GPU. With a heavy job still pending, a restore the placement policy immediately
    re-demotes would cost two full safety process rebuilds per residency.
    """

    def _drained_residency_scheduler(self, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202
        """A scheduler holding a fully-drained whole-card residency on the safety card, ready to restore."""
        bridge_data = make_mock_bridge_data(
            safety_on_gpu=True,
            whole_card_residency_safety_off_gpu=True,
            whole_card_residency_cooldown_seconds=0,
        )
        monkeypatch.setattr(sched_mod, "is_cpu_only_install", lambda: False)
        scheduler = _make_inference_scheduler(process_map=ProcessMap({}), bridge_data=bridge_data)
        lifecycle = Mock()
        lifecycle.is_safety_gpu_paused = True
        lifecycle.safety_pause_owner = PauseOwner.WHOLE_CARD
        lifecycle.safety_placement_transition_pending = False
        lifecycle.pause_safety_on_gpu = Mock(return_value=True)
        lifecycle.restore_safety_on_gpu = Mock(return_value=True)
        lifecycle.post_process_lane_enabled = Mock(return_value=False)
        lifecycle.component_lane_enabled = Mock(return_value=False)
        lifecycle.vae_lane_enabled = Mock(return_value=False)
        scheduler._process_lifecycle = lifecycle
        scheduler._whole_card_ledger.record_grant(
            None,
            model="heavy-model",
            forecast=None,
            cooldown_until=0.0,
            now=0.0,
            refresh_established=True,
        )
        return scheduler

    def test_residency_end_does_not_restore_safety_the_placement_policy_wants_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A drained residency leaves safety off the card while the placement latch still holds it off."""
        scheduler = self._drained_residency_scheduler(monkeypatch)
        # Drive the latch on through the policy itself: the modeled charge does not fit beside the peak the
        # card is committed to, for the full run of cycles the hysteresis demands.
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False
        scheduler._safety_restore_headroom_fits = lambda device_index: False
        for _ in range(_SAFETY_PLACEMENT_PAUSE_STREAK):
            scheduler._reconcile_runtime_safety_placement()
        assert scheduler._safety_placement_wants_off is True, (
            "precondition: the placement policy holds safety off the card"
        )

        scheduler._restore_siblings_after_whole_card()
        scheduler._reconcile_runtime_safety_placement()

        assert scheduler._residency_state(None).model is None, (
            "precondition: the drained residency is released, so this is the restore path under test"
        )
        scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()

    def test_residency_end_restores_safety_when_the_placement_policy_has_no_wish(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no off-GPU wish outstanding, the drained residency puts safety back on its card."""
        scheduler = self._drained_residency_scheduler(monkeypatch)
        assert scheduler._safety_placement_wants_off is False

        scheduler._restore_siblings_after_whole_card()
        scheduler._reconcile_runtime_safety_placement()

        assert scheduler._residency_state(None).model is None
        scheduler._process_lifecycle.restore_safety_on_gpu.assert_called_once_with(owner=PauseOwner.WHOLE_CARD)


class TestDemoteThenMeasuredRepromote:
    """Demotion latches the policy off, and a later run of measured-headroom cycles re-promotes safety."""

    def test_demotion_latches_and_measured_headroom_repromotes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The full timeline: modeled non-fit demotes, measured device-free headroom promotes, counters move."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False
        scheduler._safety_restore_headroom_fits = lambda device_index: False

        for _ in range(_SAFETY_PLACEMENT_PAUSE_STREAK):
            scheduler._reconcile_runtime_safety_placement()

        assert scheduler._safety_placement_wants_off is True
        assert scheduler._safety_placement_demotions == 1
        scheduler._process_lifecycle.pause_safety_on_gpu.assert_called_once_with(
            owner=PauseOwner.RUNTIME_SAFETY_PLACEMENT,
        )

        # The pause has taken effect; the card now reports durable measured free between sampling peaks even
        # though the modeled peak (sustained load) still says it does not fit.
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._process_lifecycle.safety_pause_owner = PauseOwner.RUNTIME_SAFETY_PLACEMENT
        scheduler._safety_restore_headroom_fits = lambda device_index: True

        for _ in range(_SAFETY_PLACEMENT_RESTORE_STREAK - 1):
            scheduler._reconcile_runtime_safety_placement()
            scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()

        scheduler._reconcile_runtime_safety_placement()
        scheduler._process_lifecycle.restore_safety_on_gpu.assert_called_once_with(
            owner=PauseOwner.RUNTIME_SAFETY_PLACEMENT,
        )
        assert scheduler._safety_placement_wants_off is False
        assert scheduler._safety_placement_promotions == 1

    def test_transient_measured_headroom_does_not_flap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A single measured-headroom cycle inside a demoted run does not re-promote (hysteresis)."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._process_lifecycle.safety_pause_owner = PauseOwner.RUNTIME_SAFETY_PLACEMENT
        scheduler._safety_placement_wants_off = True
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False

        headroom_readings = iter([True, False, True, False, True, False])
        scheduler._safety_restore_headroom_fits = lambda device_index: next(headroom_readings)

        for _ in range(6):
            scheduler._reconcile_runtime_safety_placement()

        scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()
        assert scheduler._safety_placement_wants_off is True


class TestSafetyBacklogPriority:
    """A deep safety backlog makes safety placement more urgent, not less."""

    async def test_deep_backlog_allows_repromotion_when_headroom_is_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A paused safety process should return to GPU service while a backlog waits and headroom is proven."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._process_lifecycle.safety_pause_owner = PauseOwner.RUNTIME_SAFETY_PLACEMENT
        scheduler._safety_placement_wants_off = True
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False
        scheduler._safety_restore_headroom_fits = lambda device_index: True
        await _queue_safety_backlog(scheduler, depth=3)

        for _ in range(_SAFETY_PLACEMENT_RESTORE_STREAK):
            scheduler._reconcile_runtime_safety_placement()

        scheduler._process_lifecycle.restore_safety_on_gpu.assert_called_once_with(
            owner=PauseOwner.RUNTIME_SAFETY_PLACEMENT,
        )
        scheduler._process_lifecycle.pause_safety_on_gpu.assert_not_called()
        assert scheduler._safety_placement_wants_off is False

    async def test_deep_backlog_does_not_demote_safety_that_is_already_on_gpu(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A deep backlog protects an active GPU safety process from placement demotion."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False
        scheduler._safety_restore_headroom_fits = lambda device_index: False
        await _queue_safety_backlog(scheduler, depth=3)

        for _ in range(_SAFETY_PLACEMENT_PAUSE_STREAK):
            scheduler._reconcile_runtime_safety_placement()

        scheduler._process_lifecycle.pause_safety_on_gpu.assert_not_called()
        assert scheduler._safety_placement_wants_off is False


class TestHeadroomAwarePlacement:
    """The placement identity chooses the card with the most verified headroom, not a fixed device 0."""

    def _two_card_scheduler(self, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202
        monkeypatch.setattr(sched_mod, "is_cpu_only_install", lambda: False)
        bridge_data = make_mock_bridge_data(safety_on_gpu=True)
        card_runtimes = make_test_card_runtimes(device_indices=(0, 1), mask_kind="cuda")
        scheduler = _make_inference_scheduler(bridge_data=bridge_data, card_runtimes=card_runtimes)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=4500.0)
        return scheduler

    def test_chooses_card_with_more_measured_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With measured device-free reported per card, the roomier card wins."""
        scheduler = self._two_card_scheduler(monkeypatch)
        free_by_device = {0: 2000.0, 1: 6000.0}
        scheduler._process_map.get_free_vram_mb = Mock(
            side_effect=lambda *, device_index: free_by_device[device_index]
        )
        assert scheduler._choose_safety_gpu_card() == 1

    def test_falls_back_to_total_less_peak_without_measured_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without measured free, the choice is card total less the modeled sampling peak; the larger card wins."""
        scheduler = self._two_card_scheduler(monkeypatch)
        scheduler._process_map.get_free_vram_mb = Mock(return_value=None)
        total_by_device = {0: 8000.0, 1: 24000.0}
        scheduler._process_map.get_reported_total_vram_mb = Mock(
            side_effect=lambda *, device_index: total_by_device[device_index],
        )
        assert scheduler._choose_safety_gpu_card() == 1

    def test_reconcile_pushes_chosen_card_to_lifecycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each reconcile cycle pushes the chosen card to the lifecycle manager so spawn and restore agree."""
        scheduler = self._two_card_scheduler(monkeypatch)
        scheduler._choose_safety_gpu_card = Mock(return_value=1)
        scheduler._process_lifecycle.safety_gpu_card_index = Mock(return_value=None)

        scheduler._reconcile_runtime_safety_placement()

        scheduler._process_lifecycle.set_desired_safety_card.assert_called_with(1)
