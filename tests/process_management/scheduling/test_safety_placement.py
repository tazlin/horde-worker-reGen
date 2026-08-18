"""The runtime safety-placement policy: keep safety off-GPU only while its own card is really short of memory.

The policy generalises the whole-card safety-off lever to the ordinary case, and every term it reads is about
the card safety occupies (or the card it would land on while it is off). Demotion needs measured pressure there:
the device-free governor off HEALTHY, or measured free below the marginal step the heaviest peak that card is
committed to still has to make. The modeled non-fit is a forecast input and never a trigger on its own, because
on a small card that arithmetic can fail by less than the noise buffer while the card is comfortably serving.
Restoration needs the mirror-image forecast, measured free covering safety *beside* that same peak, so a restore
does not immediately re-trip.

Both sides are dwelt in seconds against what a flip actually costs (the measured safety readiness latency), not
in control cycles: the loop runs several times a second, so a cycle count lets a fraction of a second of reading
spend tens of seconds of safety unavailability. Evidence is frozen across an intentional rebuild and restarts
from scratch after it, so the respawn window cannot decide the flip that follows it. Placement is headroom-aware
across cards and is only re-chosen while a spawn could use it.
"""

from __future__ import annotations

import dataclasses
import sys
import time
import uuid
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
from horde_worker_regen.process_management.ipc.messages import HordeImageResult, HordeProcessState
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    SAFETY_READINESS_LATENCY_FLOOR_SECONDS,
    PauseOwner,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
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
    _SAFETY_PLACEMENT_RESTORE_DWELL_FACTOR,
    _SAFETY_RESTORE_PP_BACKLOG_DEPTH,
    _SAFETY_RESTORE_PP_BACKLOG_MAX_AGE_SECONDS,
    InferenceScheduler,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_READINESS_SECONDS = 30.0
"""What the stand-in lifecycle reports one safety placement flip costs, and hence the demotion dwell.

Held at the floor the real manager uses before it has timed a flip, so the rows read as the cold-start case an
operator's first eviction of a session actually runs under."""


class _TestClock:
    """A hand-advanced clock, so a dwell measured in seconds is exercised without sleeping."""

    def __init__(self) -> None:
        self.now = 10_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward."""
        self.now += seconds


@dataclass
class _PlacementHarness:
    """A scheduler whose safety placement is policy-managed, with the clock and lifecycle the rows drive."""

    scheduler: InferenceScheduler
    clock: _TestClock
    lifecycle: Mock

    @property
    def demotion_dwell_seconds(self) -> float:
        """The seconds of sustained pressure a demotion has to outlast."""
        return _READINESS_SECONDS

    @property
    def restore_dwell_seconds(self) -> float:
        """The seconds of sustained forecast headroom a restore has to earn."""
        return _READINESS_SECONDS * _SAFETY_PLACEMENT_RESTORE_DWELL_FACTOR

    def reconcile(self) -> None:
        """Run one placement reconciliation at the current clock."""
        self.scheduler._reconcile_runtime_safety_placement()

    def reconcile_over(self, seconds: float, *, step_seconds: float = 2.0) -> None:
        """Reconcile repeatedly across ``seconds`` of clock, the way the control loop samples a window.

        The first reconciliation happens before the clock moves at all, so a window of zero is still one
        observation: that is what makes "a single reading never actuates" expressible.
        """
        self.reconcile()
        elapsed = 0.0
        while elapsed < seconds:
            self.clock.advance(step_seconds)
            elapsed += step_seconds
            self.reconcile()

    def pause_and_settle(self, owner: PauseOwner) -> None:
        """Put the world in the state a completed pause leaves it in: safety off-GPU, owned by ``owner``."""
        self.lifecycle.is_safety_gpu_paused = True
        self.lifecycle.safety_pause_owner = owner


def _placement_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    safety_on_gpu: bool = True,
    card_runtimes: dict[int, CardRuntime] | None = None,
    device_free_mb: float | None = 24000.0,
) -> _PlacementHarness:
    """A single-GPU scheduler whose safety process is placement-managed, with a mocked lifecycle.

    The CPU-only guard is patched off so the policy is active regardless of the test host: on a real CPU-only
    install safety is always off-GPU already, so the policy would (correctly) be inert.
    """
    monkeypatch.setattr(sched_mod, "is_cpu_only_install", lambda: False)
    bridge_data = make_mock_bridge_data(safety_on_gpu=safety_on_gpu)
    # Production derives every card's effective config from the global one, so the plan's cards carry the
    # bridge data under test: safety_on_gpu is a per-card permission read off that config.
    if card_runtimes is not None:
        card_runtimes = {
            index: dataclasses.replace(card, config=bridge_data)  # pyrefly: ignore - a mock stands in
            for index, card in card_runtimes.items()
        }
    clock = _TestClock()
    scheduler = _make_inference_scheduler(
        process_map=ProcessMap({}),
        bridge_data=bridge_data,
        clock=clock,
        card_runtimes=card_runtimes,
        device_free_mb=device_free_mb,
    )
    lifecycle = Mock()
    lifecycle.is_safety_gpu_paused = False
    lifecycle.safety_pause_owner = None
    lifecycle.safety_placement_transition_pending = False

    def _mark_paused(*, owner: PauseOwner) -> bool:
        lifecycle.is_safety_gpu_paused = True
        lifecycle.safety_pause_owner = owner
        return True

    def _mark_restored(*, owner: PauseOwner) -> bool:
        lifecycle.is_safety_gpu_paused = False
        lifecycle.safety_pause_owner = None
        return True

    # The actuators move the placement state the way the real lifecycle manager does, so a row that keeps
    # reconciling after a flip sees the world the flip produced rather than one stuck before it.
    lifecycle.pause_safety_on_gpu = Mock(side_effect=_mark_paused)
    lifecycle.restore_safety_on_gpu = Mock(side_effect=_mark_restored)
    lifecycle.safety_readiness_latency_seconds = Mock(return_value=_READINESS_SECONDS)
    scheduler._process_lifecycle = lifecycle
    return _PlacementHarness(scheduler=scheduler, clock=clock, lifecycle=lifecycle)


def _pin_evidence(
    harness: _PlacementHarness,
    *,
    pressured: bool = False,
    headroom_fits: bool = False,
) -> None:
    """Pin both placement predicates, so a row states the state machine rather than the arithmetic.

    The arithmetic has its own rows (:class:`TestPlacementEvidence`); pinning it here is what lets a dwell row
    say exactly how long a given verdict was held for.
    """
    harness.scheduler._safety_placement_card_is_pressured = lambda device_index: pressured  # type: ignore[method-assign]
    harness.scheduler._safety_restore_headroom_fits = lambda device_index: headroom_fits  # type: ignore[method-assign]


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
    """The structural fit is arithmetic over the device total and the largest peak the card is committed to."""

    def test_charge_fits_on_a_roomy_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A large card holds the safety charge beside a moderate peak, bare and with margin."""
        harness = _placement_harness(monkeypatch)
        scheduler = harness.scheduler
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=24000.0)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=8192.0)
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=False) is True
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=True) is True

    def test_tight_card_bare_fit_but_no_margin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On a tight card the charge bare-fits but fails the proportional margin."""
        harness = _placement_harness(monkeypatch)
        scheduler = harness.scheduler
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=11500.0)
        # 16000 - 11500 - 800 (5% noise) - 3044 (the safety seed) = 656 >= 0 bare; a second 800 margin makes
        # it negative.
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=False) is True
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=True) is False

    def test_nothing_sampling_fits_trivially(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no active sampling peak the charge trivially fits (nothing to fit beside)."""
        harness = _placement_harness(monkeypatch)
        scheduler = harness.scheduler
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=None)
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=True) is True

    def test_unknown_total_fits_trivially(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unreported device total leaves the charge fitting (missing-telemetry admits)."""
        harness = _placement_harness(monkeypatch)
        harness.scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=None)
        assert harness.scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=False) is True


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
        scheduler = _placement_harness(monkeypatch).scheduler
        assert scheduler._safety_footprint_mb() == _SAFETY_GPU_LOAD_CHARGE_MB
        scheduler.set_footprint_store(LearnedFootprintStore())
        assert scheduler._safety_footprint_mb() == _SAFETY_GPU_LOAD_CHARGE_MB

    def test_measured_footprint_above_the_seed_raises_the_price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A safety process measured heavier than the seed is priced at what it actually costs."""
        scheduler = _placement_harness(monkeypatch).scheduler
        _observe_safety_footprint(scheduler, _SAFETY_GPU_LOAD_CHARGE_MB + 1500.0)
        assert scheduler._safety_footprint_mb() == _SAFETY_GPU_LOAD_CHARGE_MB + 1500.0

    def test_measured_footprint_below_the_seed_never_lowers_the_price(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The overlay is raise-only: a lighter measurement leaves the conservative seed standing."""
        scheduler = _placement_harness(monkeypatch).scheduler
        _observe_safety_footprint(scheduler, _SAFETY_GPU_LOAD_CHARGE_MB - 1500.0)
        assert scheduler._safety_footprint_mb() == _SAFETY_GPU_LOAD_CHARGE_MB

    def test_learned_price_raises_the_restore_requirement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A measured footprint the seed under-stated raises what a restore has to find on the card."""
        harness = _placement_harness(monkeypatch, device_free_mb=None)
        scheduler = harness.scheduler
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)
        scheduler._process_map.get_free_vram_mb = Mock(return_value=4000.0)
        scheduler._safety_placement_marginal_need_mb = Mock(return_value=0.0)
        assert scheduler._safety_restore_headroom_fits(None) is True

        # 4000 free covered 3044 + 800 with 156MB to spare; a measurement 1GB above the seed does not.
        _observe_safety_footprint(scheduler, _SAFETY_GPU_LOAD_CHARGE_MB + 1024.0)
        assert scheduler._safety_restore_headroom_fits(None) is False


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
        harness = _placement_harness(monkeypatch)
        scheduler = harness.scheduler
        scheduler._process_map = process_map
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=24000.0)
        if learned_footprint_mb is not None:
            _observe_safety_footprint(scheduler, learned_footprint_mb)

        expected_mb = learned_footprint_mb if learned_footprint_mb is not None else _SAFETY_GPU_LOAD_CHARGE_MB

        job = make_job_pop_response("Deliberate")
        harness.lifecycle.is_safety_gpu_paused = False
        on_gpu = scheduler._forecast_streaming(job, "stable_diffusion_1")
        harness.lifecycle.is_safety_gpu_paused = True
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


class TestPlacementEvidence:
    """What the policy reads about safety's own card: the marginal need, the pressure test, the forecast.

    Rewritten from the old modeled-non-fit contract. The predicate that used to demote (device total less the
    *whole* peak, the noise buffer and safety's footprint) is unsatisfiable on a small card by less than the
    noise buffer, so it was permanently armed while the card served its work; and the predicate that used to
    promote priced only safety's own context, so a restore onto a card already committed to a peak re-tripped
    it. Both are replaced by per-card measured evidence against the *marginal* step the committed peak still
    has to make.
    """

    def _card_scheduler(self, monkeypatch: pytest.MonkeyPatch, *, total_mb: float = 8107.0) -> InferenceScheduler:
        """A scheduler reporting one small card's total, with the device-truth ceiling out of the way."""
        harness = _placement_harness(monkeypatch, device_free_mb=None)
        harness.scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=total_mb)
        return harness.scheduler

    def test_marginal_need_nets_out_what_the_card_already_holds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A resident the peak samples through is already on the card, so only the step above it is needed."""
        scheduler = self._card_scheduler(monkeypatch)
        scheduler._process_map = ProcessMap({1: make_mock_process_info(1, model_name="heavy-model")})
        scheduler._process_map[1].process_reserved_mb = 3200
        scheduler.resolved_context_constant_mb = Mock(return_value=500.0)  # type: ignore[method-assign]
        scheduler._largest_active_sampling_peak = Mock(return_value=(4600.0, "heavy-model"))

        assert scheduler._safety_placement_marginal_need_mb(0) == pytest.approx(4600.0 - 3700.0)

    def test_marginal_need_is_the_whole_peak_when_nothing_holds_the_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no priced process holding the model, the card has to find the whole peak."""
        scheduler = self._card_scheduler(monkeypatch)
        scheduler._largest_active_sampling_peak = Mock(return_value=(4600.0, "heavy-model"))

        assert scheduler._safety_placement_marginal_need_mb(0) == pytest.approx(4600.0)

    def test_marginal_need_is_zero_without_a_committed_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A card committed to no sampling needs nothing for a peak it does not have."""
        scheduler = self._card_scheduler(monkeypatch)
        scheduler._largest_active_sampling_peak = Mock(return_value=None)

        assert scheduler._safety_placement_marginal_need_mb(0) == 0.0

    def test_modeled_non_fit_alone_is_not_pressure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 8GB card whose modeled margin fails by tens of megabytes is not under pressure.

        8107 - 4600 - 512 - 3044 is negative by 49MB, well inside the buffer, so the modeled fit says safety
        cannot be there and always will; the card is
        nonetheless holding the resident its peak samples through and reporting free above the marginal step,
        which is the whole distinction the policy now draws.
        """
        scheduler = self._card_scheduler(monkeypatch)
        scheduler._largest_active_sampling_peak = Mock(return_value=(4600.0, "heavy-model"))
        scheduler._process_map.get_free_vram_mb = Mock(return_value=1600.0)
        scheduler._resident_committed_mb_for_model = Mock(return_value=3700.0)

        assert scheduler._safety_fits_beside_largest_sampling_peak(0, require_margin=False) is False
        assert scheduler._safety_placement_card_is_pressured(0) is False

    def test_measured_free_below_the_marginal_need_is_pressure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Once the card cannot cover the step its committed peak still has to make, it is pressured."""
        scheduler = self._card_scheduler(monkeypatch)
        scheduler._largest_active_sampling_peak = Mock(return_value=(4600.0, "heavy-model"))
        scheduler._resident_committed_mb_for_model = Mock(return_value=3700.0)
        scheduler._process_map.get_free_vram_mb = Mock(return_value=800.0)

        assert scheduler._safety_placement_card_is_pressured(0) is True

    def test_governor_off_healthy_is_pressure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A card the device-free governor has taken off HEALTHY is pressured whatever the arithmetic says."""
        scheduler = self._card_scheduler(monkeypatch)
        scheduler._largest_active_sampling_peak = Mock(return_value=None)
        scheduler._process_map.get_free_vram_mb = Mock(return_value=6000.0)
        scheduler.set_governor_state(0, GovernorState.PRESSURE)

        assert scheduler._safety_placement_card_is_pressured(0) is True

    def test_a_card_full_of_weights_with_nothing_left_to_find_is_not_pressure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A low free reading beside work whose memory the card already holds is not safety's problem.

        The card can be almost entirely committed (one heavy resident kept warm between jobs) while every peak
        it is committed to is already covered by what is resident. Evicting safety there buys the card nothing it
        needs, and the eviction outlives the residency that caused the reading, because those weights are
        exactly what the card then has no room beside.
        """
        scheduler = self._card_scheduler(monkeypatch)
        scheduler._largest_active_sampling_peak = Mock(return_value=None)
        scheduler._process_map.get_free_vram_mb = Mock(return_value=0.0)

        assert scheduler._safety_placement_card_is_pressured(0) is False

    def test_missing_measured_free_is_not_pressure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a measured reading the policy does not demote (missing telemetry admits)."""
        scheduler = self._card_scheduler(monkeypatch)
        scheduler._largest_active_sampling_peak = Mock(return_value=(4600.0, "heavy-model"))
        scheduler._process_map.get_free_vram_mb = Mock(return_value=None)

        assert scheduler._safety_placement_card_is_pressured(0) is False

    def test_restore_forecast_requires_room_for_safety_beside_the_committed_peak(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Room for safety's context alone is not enough: the peak the card is committed to takes it back."""
        scheduler = self._card_scheduler(monkeypatch)
        scheduler._largest_active_sampling_peak = Mock(return_value=(4600.0, "heavy-model"))
        scheduler._resident_committed_mb_for_model = Mock(return_value=3700.0)
        # 3044 (safety) + 900 (marginal need) + 512 (noise floor) = 4456 required.
        scheduler._process_map.get_free_vram_mb = Mock(return_value=4300.0)
        assert scheduler._safety_restore_headroom_fits(0) is False

        scheduler._process_map.get_free_vram_mb = Mock(return_value=4600.0)
        assert scheduler._safety_restore_headroom_fits(0) is True

    def test_restore_forecast_refuses_an_unhealthy_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A card hovering at the paging cliff never readmits safety however much free it reports."""
        scheduler = self._card_scheduler(monkeypatch)
        scheduler._largest_active_sampling_peak = Mock(return_value=None)
        scheduler._process_map.get_free_vram_mb = Mock(return_value=7000.0)
        scheduler.set_governor_state(0, GovernorState.SATURATED)

        assert scheduler._safety_restore_headroom_fits(0) is False


def _pin_tracker_jobs(
    scheduler: InferenceScheduler,
    *,
    in_progress: tuple[object, ...] = (),
    pending_inference: tuple[object, ...] = (),
) -> None:
    """Pin what the tracker reports as running and queued, so a routing row states only the attribution."""
    tracker = Mock()
    tracker.jobs_in_progress = in_progress
    tracker.jobs_pending_inference = pending_inference
    scheduler._job_tracker = tracker


class TestPerCardCommittedPeak:
    """The peak the policy prices is the one its own card is committed to, not the worker's largest anywhere.

    The old worker-wide figure is why an eviction on one card was armed by a job that could only ever run on
    the other, and why merely queued work anywhere counted against safety's residency everywhere.
    """

    def _two_card_scheduler(self, monkeypatch: pytest.MonkeyPatch) -> InferenceScheduler:
        return _placement_harness(
            monkeypatch,
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1), mask_kind="cuda"),
        ).scheduler

    def test_in_progress_jobs_are_attributed_to_their_own_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job sampling on card 1 is not part of what card 0 is committed to."""
        scheduler = self._two_card_scheduler(monkeypatch)
        on_card_zero = make_job_pop_response("model-a")
        on_card_one = make_job_pop_response("model-b")
        _pin_tracker_jobs(scheduler, in_progress=(on_card_zero, on_card_one))
        jobs_by_card = {0: [on_card_zero], 1: [on_card_one]}
        scheduler._jobs_in_progress_on_card = lambda device_index: jobs_by_card[device_index]  # type: ignore[method-assign]

        assert scheduler._sampling_peak_jobs_for_card(0) == [on_card_zero]
        assert scheduler._sampling_peak_jobs_for_card(1) == [on_card_one]

    def test_queued_jobs_count_only_where_they_could_land(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A queued job only one card can serve is only that card's commitment."""
        scheduler = self._two_card_scheduler(monkeypatch)
        large_job = make_job_pop_response("model-a", width=1536, height=1536)
        _pin_tracker_jobs(scheduler, pending_inference=(large_job,))
        scheduler._jobs_in_progress_on_card = lambda device_index: []  # type: ignore[method-assign]
        scheduler._eligible_card_indices = lambda job: {1}  # type: ignore[method-assign]

        assert scheduler._sampling_peak_jobs_for_card(0) == []
        assert scheduler._sampling_peak_jobs_for_card(1) == [large_job]

    def test_single_gpu_keeps_the_whole_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On one card the per-card question has the worker-wide answer, so routing is a no-op."""
        scheduler = _placement_harness(monkeypatch).scheduler
        running = make_job_pop_response("model-a")
        queued = make_job_pop_response("model-b")
        _pin_tracker_jobs(scheduler, in_progress=(running,), pending_inference=(queued,))

        assert scheduler._sampling_peak_jobs_for_card(0) == [running, queued]
        assert scheduler._sampling_peak_jobs_for_card(None) == [running, queued]

    def test_heaviest_priced_job_carries_the_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Among the jobs a card is committed to, the heaviest learned peak and its model are what is priced."""
        scheduler = _placement_harness(monkeypatch).scheduler
        light = make_job_pop_response("light-model")
        heavy = make_job_pop_response("heavy-model")
        _pin_tracker_jobs(scheduler, in_progress=(light, heavy))
        peak_by_model = {"light-model": 1000.0, "heavy-model": 4600.0}
        monkeypatch.setattr(
            sched_mod,
            "predict_job_sampling_vram_mb",
            lambda job, baseline: peak_by_model[str(job.model)],
        )
        scheduler._learned_sampling_peak_mb = (  # type: ignore[method-assign]
            lambda job, baseline, *, static_seed_mb, stage: static_seed_mb
        )

        assert scheduler._largest_active_sampling_peak(0) == (4600.0, "heavy-model")


class TestDemotionDwell:
    """A demotion is justified by pressure that persisted on the order of what relieving it costs.

    Rewritten from ``TestPlacementHysteresis``: the old contract was a count of consecutive control cycles
    (two to evict, five to readmit), which is a fraction of a second of wall clock against a rebuild measured
    in tens of seconds, so a respawn window could decide the next flip. The band is now seconds.
    """

    def test_a_short_transient_never_evicts_safety(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pressure that clears well inside the dwell costs nothing: no flip is spent on it."""
        harness = _placement_harness(monkeypatch)
        _pin_evidence(harness, pressured=True)

        harness.reconcile_over(harness.demotion_dwell_seconds / 3.0)

        harness.lifecycle.pause_safety_on_gpu.assert_not_called()

    def test_sustained_pressure_past_the_dwell_evicts_safety_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pressure that outlasts the dwell earns exactly one eviction."""
        harness = _placement_harness(monkeypatch)
        _pin_evidence(harness, pressured=True)

        harness.reconcile_over(harness.demotion_dwell_seconds + 4.0)

        harness.lifecycle.pause_safety_on_gpu.assert_called_once_with(owner=PauseOwner.RUNTIME_SAFETY_PLACEMENT)
        assert harness.scheduler._safety_placement_demotions == 1

    def test_intermittent_pressure_restarts_the_dwell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A card that recovers between readings never accumulates a dwell, however long the run is."""
        harness = _placement_harness(monkeypatch)
        readings = iter([True, False] * 200)
        harness.scheduler._safety_placement_card_is_pressured = lambda device_index: next(readings)  # type: ignore[method-assign]
        harness.scheduler._safety_restore_headroom_fits = lambda device_index: False  # type: ignore[method-assign]

        harness.reconcile_over(harness.demotion_dwell_seconds * 4.0)

        harness.lifecycle.pause_safety_on_gpu.assert_not_called()

    def test_modeled_non_fit_alone_never_evicts_safety(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The permanently-armed modeled non-fit is not a trigger: a healthy card keeps its safety process.

        This is the incident in one row. With the modeled arithmetic failing every cycle for the whole run, a
        HEALTHY governor and measured free above the marginal need leave safety exactly where the operator put
        it.
        """
        harness = _placement_harness(monkeypatch)
        harness.scheduler._safety_fits_beside_largest_sampling_peak = (  # type: ignore[method-assign]
            lambda device_index, *, require_margin: False
        )
        _pin_evidence(harness, pressured=False)

        harness.reconcile_over(harness.demotion_dwell_seconds * 6.0)

        harness.lifecycle.pause_safety_on_gpu.assert_not_called()
        assert harness.scheduler._safety_placement_demotions == 0

    def test_config_false_leaves_the_policy_inert(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ``safety_on_gpu`` off the policy neither evicts nor promotes."""
        harness = _placement_harness(monkeypatch, safety_on_gpu=False)
        harness.pause_and_settle(PauseOwner.RUNTIME_SAFETY_PLACEMENT)
        _pin_evidence(harness, pressured=True, headroom_fits=True)

        harness.reconcile_over(harness.restore_dwell_seconds * 2.0)

        harness.lifecycle.restore_safety_on_gpu.assert_not_called()
        harness.lifecycle.pause_safety_on_gpu.assert_not_called()
        assert harness.scheduler._safety_placement_pressure_since is None
        assert harness.scheduler._safety_placement_headroom_since is None


class TestRestoreForecastDwell:
    """A restore needs the card to cover safety beside its committed peak, durably.

    Rewritten from the old measured-headroom streak, which priced safety's context alone and counted cycles.
    """

    def test_restore_waits_out_the_longer_dwell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Forecast headroom short of the restore dwell does not readmit safety; past it, it does."""
        harness = _placement_harness(monkeypatch)
        harness.pause_and_settle(PauseOwner.RUNTIME_SAFETY_PLACEMENT)
        _pin_evidence(harness, headroom_fits=True)

        harness.reconcile_over(harness.restore_dwell_seconds - 4.0)
        harness.lifecycle.restore_safety_on_gpu.assert_not_called()

        harness.reconcile_over(8.0)
        harness.lifecycle.restore_safety_on_gpu.assert_called_once_with(owner=PauseOwner.RUNTIME_SAFETY_PLACEMENT)
        assert harness.scheduler._safety_placement_promotions == 1
        assert harness.scheduler._safety_placement_headroom_since is None

    def test_the_restore_dwell_is_longer_than_the_demotion_dwell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Leave promptly, come back slowly: the asymmetry is what stops a readmit re-tripping the evict."""
        harness = _placement_harness(monkeypatch)
        assert harness.restore_dwell_seconds > harness.demotion_dwell_seconds
        assert harness.scheduler._safety_placement_restore_dwell_seconds() == pytest.approx(
            harness.restore_dwell_seconds,
        )
        assert harness.scheduler._safety_placement_dwell_seconds() == pytest.approx(
            harness.demotion_dwell_seconds,
        )

    def test_a_forecast_that_never_holds_never_restores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A card that cannot host safety beside its committed peak keeps the CPU placement, indefinitely."""
        harness = _placement_harness(monkeypatch)
        harness.pause_and_settle(PauseOwner.RUNTIME_SAFETY_PLACEMENT)
        _pin_evidence(harness, headroom_fits=False)

        harness.reconcile_over(harness.restore_dwell_seconds * 4.0)

        harness.lifecycle.restore_safety_on_gpu.assert_not_called()

    def test_intermittent_forecast_headroom_does_not_flap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A single passing reading inside a demoted run does not readmit safety."""
        harness = _placement_harness(monkeypatch)
        harness.pause_and_settle(PauseOwner.RUNTIME_SAFETY_PLACEMENT)
        readings = iter([True, False] * 200)
        harness.scheduler._safety_restore_headroom_fits = lambda device_index: next(readings)  # type: ignore[method-assign]
        harness.scheduler._safety_placement_card_is_pressured = lambda device_index: True  # type: ignore[method-assign]

        harness.reconcile_over(harness.restore_dwell_seconds * 4.0)

        harness.lifecycle.restore_safety_on_gpu.assert_not_called()

    def test_a_reclaim_pause_earns_the_same_forecast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ladder took the card for memory too, so its restore waits out the same forecast dwell.

        Rewritten from the fit-streak clause the reclaim restore used to carry: the streak is gone and the
        forecast dwell is what stands in its place, for the same reason it existed.
        """
        harness = _placement_harness(monkeypatch)
        harness.pause_and_settle(PauseOwner.RECLAIM_LADDER)
        _pin_evidence(harness, headroom_fits=True)

        harness.reconcile_over(harness.restore_dwell_seconds - 4.0)
        harness.lifecycle.restore_safety_on_gpu.assert_not_called()

        harness.reconcile_over(8.0)
        harness.lifecycle.restore_safety_on_gpu.assert_called_once_with(owner=PauseOwner.RECLAIM_LADDER)
        # The placement counters belong to the policy's own moves, not to the ladder's.
        assert harness.scheduler._safety_placement_promotions == 0

    def test_a_whole_card_pause_is_not_held_to_the_memory_forecast(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A residency's pause ends when its model drains, whatever the card then has room for.

        Rewritten from ``TestResidencyRestoreRespectsPlacementWish``, which expressed a related rule through the
        sticky off-GPU intent this change removes. The rule that survives is the liveness one: a card hosting one
        heavy resident has little measured free by construction, so holding its safety restore to a memory
        forecast would leave that worker running every safety check on the CPU for the session. The residency
        owns safety's placement while it holds the card, and the placement policy accrues no pressure beside it.
        """
        harness = _placement_harness(monkeypatch)
        harness.pause_and_settle(PauseOwner.WHOLE_CARD)
        _pin_evidence(harness, headroom_fits=False)

        harness.reconcile()

        harness.lifecycle.restore_safety_on_gpu.assert_called_once_with(owner=PauseOwner.WHOLE_CARD)

    def test_a_held_residency_stops_the_pressure_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A card a residency is deliberately filling does not also accrue a pressure-owned eviction.

        The defect this encodes: a whole-card residency fills the card by design, so the placement policy read
        it as pressure and took its own pause beside the residency's. That pause outlives the residency (its
        weights are what the card has no room beside), so the worker ends the session with safety on the CPU.
        """
        harness = _placement_harness(monkeypatch)
        harness.scheduler._runtime_config.bridge_data.whole_card_residency_safety_off_gpu = True
        harness.scheduler._whole_card_ledger.record_grant(
            None,
            model="heavy-model",
            forecast=None,
            cooldown_until=harness.clock() + 10_000.0,
            now=harness.clock(),
        )
        _pin_evidence(harness, pressured=True)

        harness.reconcile_over(harness.demotion_dwell_seconds * 3.0)

        assert harness.scheduler._safety_placement_pressure_since is None
        assert harness.scheduler._safety_placement_demotions == 0
        harness.lifecycle.pause_safety_on_gpu.assert_called_once_with(owner=PauseOwner.WHOLE_CARD)


class TestTransitionFreeze:
    """Evidence does not accrue across an intentional rebuild, and restarts from scratch after it.

    The defect this encodes: the streaks advanced while a placement rebuild was still unready, so the pause
    that had just been actuated was itself the window that earned the next decision. Pauses logged "after 0
    consecutive cycles" because the intent latched during the respawn.
    """

    def test_no_dwell_accrues_while_a_rebuild_is_pending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pressure held for many dwells' worth of a pending rebuild earns no eviction."""
        harness = _placement_harness(monkeypatch)
        harness.lifecycle.safety_placement_transition_pending = True
        _pin_evidence(harness, pressured=True)

        harness.reconcile_over(harness.demotion_dwell_seconds * 5.0)

        harness.lifecycle.pause_safety_on_gpu.assert_not_called()
        assert harness.scheduler._safety_placement_pressure_since is None

    def test_evidence_restarts_after_the_rebuild_clears(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The dwell is measured from the readiness edge, not from before the rebuild began."""
        harness = _placement_harness(monkeypatch)
        harness.lifecycle.safety_placement_transition_pending = True
        _pin_evidence(harness, pressured=True)
        harness.reconcile_over(harness.demotion_dwell_seconds * 3.0)

        harness.lifecycle.safety_placement_transition_pending = False
        harness.reconcile_over(harness.demotion_dwell_seconds / 3.0)
        harness.lifecycle.pause_safety_on_gpu.assert_not_called()

        harness.reconcile_over(harness.demotion_dwell_seconds)
        harness.lifecycle.pause_safety_on_gpu.assert_called_once_with(owner=PauseOwner.RUNTIME_SAFETY_PLACEMENT)

    def test_a_pending_rebuild_does_not_chain_a_restore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A slow healthy off-GPU respawn reaches readiness before a contrary wish may replace it."""
        harness = _placement_harness(monkeypatch)
        harness.pause_and_settle(PauseOwner.RUNTIME_SAFETY_PLACEMENT)
        harness.lifecycle.safety_placement_transition_pending = True
        _pin_evidence(harness, headroom_fits=True)

        harness.reconcile_over(harness.restore_dwell_seconds * 3.0)

        harness.lifecycle.restore_safety_on_gpu.assert_not_called()
        assert harness.scheduler._safety_placement_headroom_since is None

    def test_an_actuated_pause_discards_the_evidence_that_bought_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The next decision starts from nothing, so one pressure episode cannot buy two flips."""
        harness = _placement_harness(monkeypatch)
        _pin_evidence(harness, pressured=True)

        harness.reconcile_over(harness.demotion_dwell_seconds + 4.0)

        harness.lifecycle.pause_safety_on_gpu.assert_called_once()
        assert harness.scheduler._safety_placement_pressure_since is None


class TestPlacementRequestOwnership:
    """Residency and reclaim file placement demand; only the reconciler invokes the lifecycle actuator."""

    def test_held_residency_is_applied_as_a_whole_card_owned_pause(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A held residency becomes one owner-attributed pause when the recurring reconciler observes it."""
        harness = _placement_harness(monkeypatch)
        harness.scheduler._runtime_config.bridge_data.whole_card_residency_safety_off_gpu = True
        harness.scheduler._whole_card_ledger.record_grant(
            None,
            model="heavy-model",
            forecast=None,
            cooldown_until=100.0,
            now=0.0,
        )

        harness.reconcile()

        harness.lifecycle.pause_safety_on_gpu.assert_called_once_with(owner=PauseOwner.WHOLE_CARD)

    def test_reclaim_request_is_applied_without_advancing_the_pressure_dwell(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The ladder's one-shot request is actuated by the reconciler and starts no pressure clock."""
        harness = _placement_harness(monkeypatch)
        harness.scheduler._runtime_config.bridge_data.whole_card_residency_safety_off_gpu = True
        _pin_evidence(harness, pressured=True)

        assert harness.scheduler.safety_off_gpu(None) is True

        harness.lifecycle.pause_safety_on_gpu.assert_called_once_with(owner=PauseOwner.RECLAIM_LADDER)
        assert harness.scheduler._safety_placement_pressure_since is None


class TestPostProcessingRestoreBound:
    """Post-processing defers a safety restore on depth and age, never absolutely.

    A worker serving post-processed requests is rarely without some post-processing work in flight, so an
    absolute veto lets an arbitrarily shallow but unbroken trickle keep safety on the CPU for the whole run
    however much room the card has.
    """

    def _restorable_harness(self, monkeypatch: pytest.MonkeyPatch) -> _PlacementHarness:
        """A paused-off harness whose every restore gate but the post-processing one is satisfied."""
        harness = _placement_harness(monkeypatch)
        harness.pause_and_settle(PauseOwner.RUNTIME_SAFETY_PLACEMENT)
        _pin_evidence(harness, headroom_fits=True)
        return harness

    async def _queue_post_processing(self, scheduler: object, depth: int) -> list[HordeJobInfo]:
        """Place ``depth`` generated jobs in the pending post-processing stage; return them."""
        queued: list[HordeJobInfo] = []
        for _ in range(depth):
            job_info = HordeJobInfo(
                sdk_api_job_info=make_job_pop_response(post_processing=["RealESRGAN_x4plus"]),
                job_image_results=[HordeImageResult(image_bytes=b"raw-image")],
                state=GENERATION_STATE.ok,
                censored=False,
                time_popped=time.time(),
            )
            await scheduler._job_tracker.queue_for_post_processing(job_info)  # type: ignore[attr-defined]
            queued.append(job_info)
        return queued

    async def test_a_persistent_shallow_backlog_stops_deferring_after_the_bound(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A one-deep backlog that never drains defers only until it has aged past the bound."""
        harness = self._restorable_harness(monkeypatch)
        await self._queue_post_processing(harness.scheduler, 1)

        # Kept inside the backlog's own age bound, so what is under test is the defer and not the bound.
        harness.reconcile_over(harness.restore_dwell_seconds + 4.0)
        harness.lifecycle.restore_safety_on_gpu.assert_not_called()

        # Age the unbroken backlog past the bound, leaving it just as deep and every other gate as it was.
        harness.scheduler._safety_restore_pp_backlog_since = harness.clock() - (
            _SAFETY_RESTORE_PP_BACKLOG_MAX_AGE_SECONDS + 1.0
        )
        harness.reconcile()

        harness.lifecycle.restore_safety_on_gpu.assert_called_once_with(
            owner=PauseOwner.RUNTIME_SAFETY_PLACEMENT,
        )

    async def test_a_deep_backlog_defers_however_long_it_lasts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A lane under real load keeps the card, no matter how long it has been under load."""
        harness = self._restorable_harness(monkeypatch)
        await self._queue_post_processing(harness.scheduler, _SAFETY_RESTORE_PP_BACKLOG_DEPTH + 1)

        harness.reconcile_over(harness.restore_dwell_seconds * 2.0)
        harness.scheduler._safety_restore_pp_backlog_since = harness.clock() - (
            _SAFETY_RESTORE_PP_BACKLOG_MAX_AGE_SECONDS * 10.0
        )
        harness.reconcile()

        harness.lifecycle.restore_safety_on_gpu.assert_not_called()

    async def test_a_young_shallow_backlog_defers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ordinary tail of post-processing after a batch of jobs still holds the restore off."""
        harness = self._restorable_harness(monkeypatch)
        await self._queue_post_processing(harness.scheduler, 1)

        harness.reconcile_over(harness.restore_dwell_seconds + 4.0)

        harness.lifecycle.restore_safety_on_gpu.assert_not_called()

    async def test_a_drained_backlog_resets_the_age(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An emptying lane restarts the clock, so intermittent work never accumulates toward the bound."""
        harness = self._restorable_harness(monkeypatch)
        queued = await self._queue_post_processing(harness.scheduler, 1)
        harness.reconcile()
        aged = harness.clock() - (_SAFETY_RESTORE_PP_BACKLOG_MAX_AGE_SECONDS - 1.0)
        harness.scheduler._safety_restore_pp_backlog_since = aged

        await harness.scheduler._job_tracker.abandon_pending_post_processing(queued[0])
        harness.reconcile()
        assert harness.scheduler._safety_restore_pp_backlog_since is None

        await self._queue_post_processing(harness.scheduler, 1)
        harness.reconcile()

        assert harness.scheduler._safety_restore_pp_backlog_since is not None
        assert harness.scheduler._safety_restore_pp_backlog_since > aged


class TestSafetyBacklogPriority:
    """A deep safety backlog makes safety placement more urgent, not less."""

    async def test_deep_backlog_allows_repromotion_when_the_forecast_holds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A paused safety process returns to GPU service while a backlog waits and the forecast holds."""
        harness = _placement_harness(monkeypatch)
        harness.pause_and_settle(PauseOwner.RUNTIME_SAFETY_PLACEMENT)
        _pin_evidence(harness, headroom_fits=True)
        await _queue_safety_backlog(harness.scheduler, depth=3)

        harness.reconcile_over(harness.restore_dwell_seconds + 4.0)

        harness.lifecycle.restore_safety_on_gpu.assert_called_once_with(
            owner=PauseOwner.RUNTIME_SAFETY_PLACEMENT,
        )
        harness.lifecycle.pause_safety_on_gpu.assert_not_called()

    async def test_a_backlog_protects_safety_that_is_already_on_gpu(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Evicting safety while the worker is behind on safety checks stalls the stage it is behind on."""
        harness = _placement_harness(monkeypatch)
        _pin_evidence(harness, pressured=True)
        await _queue_safety_backlog(harness.scheduler, depth=3)

        harness.reconcile_over(harness.demotion_dwell_seconds * 3.0)

        harness.lifecycle.pause_safety_on_gpu.assert_not_called()
        assert harness.scheduler._safety_placement_pressure_since is None


class TestPerCardSafetyPermission:
    """``safety_on_gpu`` is a per-card permission to host, so the chooser picks only among granting cards."""

    def _harness(self, monkeypatch: pytest.MonkeyPatch, permissions: dict[int, bool]) -> _PlacementHarness:
        """A harness whose cards carry the given per-card safety permissions."""
        harness = _placement_harness(
            monkeypatch,
            card_runtimes=make_test_card_runtimes(
                device_indices=tuple(sorted(permissions)),
                mask_kind="cuda",
            ),
        )
        harness.scheduler._card_runtimes = {
            index: dataclasses.replace(
                card,
                config=make_mock_bridge_data(safety_on_gpu=permissions[index]),  # pyrefly: ignore - a mock
            )
            for index, card in harness.scheduler._card_runtimes.items()
        }
        harness.scheduler._largest_active_sampling_peak_mb = Mock(return_value=4500.0)
        return harness

    def test_the_roomiest_card_is_skipped_when_it_withholds_the_permission(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Headroom only decides among cards that permit safety; a card that forbids it is never chosen."""
        harness = self._harness(monkeypatch, {0: False, 1: True})
        free_by_device = {0: 6000.0, 1: 2000.0}
        harness.scheduler._process_map.get_free_vram_mb = Mock(
            side_effect=lambda *, device_index: free_by_device[device_index],
        )

        assert harness.scheduler._choose_safety_gpu_card() == 1

    def test_no_permitted_card_runs_safety_off_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With every card withholding the permission there is no card to place safety on, and no policy."""
        harness = self._harness(monkeypatch, {0: False, 1: False})

        assert harness.scheduler._choose_safety_gpu_card() is None
        assert harness.scheduler._runtime_safety_placement_enabled() is False

    def test_single_gpu_follows_the_global_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One card and no per-card delta answers exactly what the global flag says, on and off."""
        on = _placement_harness(monkeypatch, card_runtimes=make_test_card_runtimes(device_indices=(0,)))
        assert on.scheduler._runtime_safety_placement_enabled() is True
        assert on.scheduler._choose_safety_gpu_card() == 0

        off = _placement_harness(
            monkeypatch,
            safety_on_gpu=False,
            card_runtimes=make_test_card_runtimes(device_indices=(0,)),
        )
        assert off.scheduler._runtime_safety_placement_enabled() is False
        assert off.scheduler._choose_safety_gpu_card() is None

    def test_reconcile_asks_the_one_actuator_to_leave_an_unpermitted_card(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A card losing its permission is handled by the placement actuator, not a second path."""
        harness = self._harness(monkeypatch, {0: False, 1: False})

        harness.reconcile()

        harness.lifecycle.demote_safety_from_unpermitted_card.assert_called()
        harness.lifecycle.pause_safety_on_gpu.assert_not_called()


class TestHeadroomAwarePlacement:
    """The placement identity chooses the card with the most verified headroom, and only when a spawn can use it."""

    def _two_card_harness(self, monkeypatch: pytest.MonkeyPatch) -> _PlacementHarness:
        harness = _placement_harness(
            monkeypatch,
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1), mask_kind="cuda"),
        )
        harness.scheduler._largest_active_sampling_peak_mb = Mock(return_value=4500.0)
        return harness

    def test_chooses_card_with_more_measured_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With measured device-free reported per card, the roomier card wins."""
        harness = self._two_card_harness(monkeypatch)
        free_by_device = {0: 2000.0, 1: 6000.0}
        harness.scheduler._process_map.get_free_vram_mb = Mock(
            side_effect=lambda *, device_index: free_by_device[device_index],
        )
        assert harness.scheduler._choose_safety_gpu_card() == 1

    def test_falls_back_to_total_less_peak_without_measured_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without measured free, the choice is card total less that card's committed peak; the larger wins."""
        harness = self._two_card_harness(monkeypatch)
        harness.scheduler._process_map.get_free_vram_mb = Mock(return_value=None)
        total_by_device = {0: 8000.0, 1: 24000.0}
        harness.scheduler._process_map.get_reported_total_vram_mb = Mock(
            side_effect=lambda *, device_index: total_by_device[device_index],
        )
        assert harness.scheduler._choose_safety_gpu_card() == 1

    def test_reconcile_pushes_the_chosen_card_while_safety_is_off_gpu(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A spawn could use the choice, so it is pushed to the lifecycle manager."""
        harness = self._two_card_harness(monkeypatch)
        harness.scheduler._choose_safety_gpu_card = Mock(return_value=1)
        harness.lifecycle.safety_gpu_card_index = Mock(return_value=None)

        harness.reconcile()

        harness.lifecycle.set_desired_safety_card.assert_called_with(1)

    def test_the_desired_card_is_not_re_chosen_while_safety_is_resident(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A live safety process stays where it is pinned: re-choosing every cycle migrates it on any respawn.

        The defect this encodes: the choice was pushed unconditionally, so a crash rebuild or a residency
        restore landed safety on whichever card happened to read roomiest that cycle rather than back where it
        was, and a restore could migrate the process for reasons unrelated to the restore.
        """
        harness = self._two_card_harness(monkeypatch)
        harness.scheduler._choose_safety_gpu_card = Mock(return_value=1)
        harness.lifecycle.safety_gpu_card_index = Mock(return_value=0)

        harness.reconcile()

        harness.lifecycle.set_desired_safety_card.assert_not_called()

    def test_single_gpu_never_pushes_a_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One card keeps the historical fixed pin, so the spawn path is byte-identical."""
        harness = _placement_harness(monkeypatch)
        harness.lifecycle.safety_gpu_card_index = Mock(return_value=None)

        harness.reconcile()

        harness.lifecycle.set_desired_safety_card.assert_not_called()

    def test_residency_readiness_is_card_local_when_safety_lives_elsewhere(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A live safety process on card 1 is already clear of a residency on card 0."""
        harness = self._two_card_harness(monkeypatch)
        harness.scheduler._runtime_config.bridge_data.whole_card_residency_safety_off_gpu = True
        harness.lifecycle.is_safety_gpu_paused = False
        harness.lifecycle.safety_gpu_card_index = Mock(return_value=1)

        assert harness.scheduler._safety_clear_of_residency_card(0) is True
        assert harness.scheduler._safety_clear_of_residency_card(1) is False

    def test_paused_safety_is_clear_of_its_selected_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Once safety is off-GPU, the selected restore destination does not keep teardown open."""
        harness = self._two_card_harness(monkeypatch)
        harness.scheduler._runtime_config.bridge_data.whole_card_residency_safety_off_gpu = True
        harness.lifecycle.is_safety_gpu_paused = True
        harness.lifecycle.safety_gpu_card_index = Mock(return_value=None)
        harness.scheduler._choose_safety_gpu_card = Mock(return_value=1)

        assert harness.scheduler._safety_clear_of_residency_card(1) is True


class TestReadinessLatencyPricesTheDwell:
    """The dwell is what a flip costs on this host, floored for a cold start, never a cycle count."""

    def test_the_dwell_follows_the_lifecycle_measurement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A host whose safety process takes longer to come up demands proportionally longer evidence."""
        harness = _placement_harness(monkeypatch)
        harness.lifecycle.safety_readiness_latency_seconds = Mock(return_value=90.0)

        assert harness.scheduler._safety_placement_dwell_seconds() == pytest.approx(90.0)
        assert harness.scheduler._safety_placement_restore_dwell_seconds() == pytest.approx(
            90.0 * _SAFETY_PLACEMENT_RESTORE_DWELL_FACTOR,
        )

    def test_a_longer_measured_latency_holds_safety_through_a_transient(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pressure that would evict on a fast host does not on a slow one: the flip costs more there."""
        harness = _placement_harness(monkeypatch)
        harness.lifecycle.safety_readiness_latency_seconds = Mock(return_value=_READINESS_SECONDS * 4.0)
        _pin_evidence(harness, pressured=True)

        harness.reconcile_over(_READINESS_SECONDS + 4.0)

        harness.lifecycle.pause_safety_on_gpu.assert_not_called()

    def test_the_floor_is_never_below_a_cold_start(self) -> None:
        """The lifecycle's own floor is what an untimed host prices a flip at, so no flip is ever free."""
        assert SAFETY_READINESS_LATENCY_FLOOR_SECONDS > 0.0


class TestReclaimableIdleResidents:
    """Idle retained residents are room the card can produce on demand, not memory it lacks."""

    def _card_scheduler(self, monkeypatch: pytest.MonkeyPatch, *, total_mb: float = 8107.0) -> InferenceScheduler:
        harness = _placement_harness(monkeypatch, device_free_mb=None)
        harness.scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=total_mb)
        return harness.scheduler

    def _idle_retained_slot(self, scheduler: InferenceScheduler, *, reserved_mb: int) -> None:
        slot = make_mock_process_info(1, model_name="warm-model")
        slot.retained_resident_model = "warm-model"
        slot.process_reserved_mb = reserved_mb
        slot.last_process_state = HordeProcessState.WAITING_FOR_JOB
        scheduler._process_map = ProcessMap({1: slot})

    def test_an_idle_retained_resident_counts_as_room_against_pressure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A low free reading beside an idle retained resident is not pressure: the resident is evictable."""
        scheduler = self._card_scheduler(monkeypatch)
        self._idle_retained_slot(scheduler, reserved_mb=3200)
        scheduler._largest_active_sampling_peak = Mock(return_value=(4600.0, "heavy-model"))
        scheduler._resident_committed_mb_for_model = Mock(return_value=0.0)
        # 800 free alone is pressure against a 4600 need; 800 + 3200 reclaimable is not against 4600 + 512.
        scheduler._process_map.get_free_vram_mb = Mock(return_value=800.0)
        assert scheduler._safety_placement_card_is_pressured(0) is True

        scheduler._process_map.get_free_vram_mb = Mock(return_value=2000.0)
        assert scheduler._safety_placement_card_is_pressured(0) is False

    def test_a_busy_slot_is_not_reclaimable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Weights a sampling slot holds are in use, so they add no room."""
        scheduler = self._card_scheduler(monkeypatch)
        self._idle_retained_slot(scheduler, reserved_mb=3200)
        scheduler._process_map[1].last_process_state = HordeProcessState.INFERENCE_STARTING
        assert scheduler._idle_retained_resident_mb(0) == 0.0

    def test_a_retained_slot_without_a_reservation_reading_adds_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing telemetry never inflates the room."""
        scheduler = self._card_scheduler(monkeypatch)
        self._idle_retained_slot(scheduler, reserved_mb=3200)
        scheduler._process_map[1].process_reserved_mb = None
        assert scheduler._idle_retained_resident_mb(0) == 0.0

    def test_the_restore_forecast_counts_reclaimable_room(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A card retaining a resident between jobs can still earn its safety process back."""
        scheduler = self._card_scheduler(monkeypatch)
        self._idle_retained_slot(scheduler, reserved_mb=3200)
        scheduler._largest_active_sampling_peak = Mock(return_value=(4600.0, "heavy-model"))
        scheduler._resident_committed_mb_for_model = Mock(return_value=3700.0)
        # 3044 (safety) + 900 (marginal need) + 512 (noise floor) = 4456 required; 1400 free + 3200 reclaimable.
        scheduler._process_map.get_free_vram_mb = Mock(return_value=1400.0)
        assert scheduler._safety_restore_headroom_fits(0) is True

        scheduler._process_map[1].retained_resident_model = None
        assert scheduler._safety_restore_headroom_fits(0) is False
