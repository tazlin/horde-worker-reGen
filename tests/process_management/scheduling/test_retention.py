"""Tests for the retention ledger and the pure per-card retention arithmetic."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from hordelib.execution.component_cache import DEFAULT_APPROX_RAM_MB, ComponentSlotKind

from horde_worker_regen.process_management.ipc.messages import (
    HeldComponentSnapshot,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.resource_budget import (
    _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB,
    predict_job_footprint_mb,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.ledgers import retention as retention_module
from horde_worker_regen.process_management.scheduling.ledgers.retention import (
    RETENTION_EVICTION_CONFIRMATION_PASSES,
    RETENTION_PRESSURE_REVOKE_SECONDS,
    RETENTION_REPEAT_EVIDENCE_DISPATCHES,
    RETENTION_STALE_HOLD_SECONDS,
    PendingRetentionEviction,
    RetentionDenialReason,
    RetentionFit,
    RetentionLedger,
    idle_lane_component_charges_mb,
    idle_resident_reclaimable_mb,
    retained_resident_charges_mb,
    sibling_context_count,
    sibling_retained_resident_present,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_CARD_24GB_MB = 24 * 1024
_SDXL_MODEL = "sdxl_checkpoint"
_OTHER_SDXL_MODEL = "other_sdxl_checkpoint"
_IDLE_LANE_RESERVED_MB = 64
"""An idle lane's allocator reservation when its seated checkpoint sits in host RAM: cached blocks, no weights."""


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _slot(
    process_id: int,
    *,
    retained: str | None = None,
    device_index: int = 0,
    process_type: HordeProcessType = HordeProcessType.INFERENCE,
    busy: bool = False,
    reserved_mb: int | None = None,
) -> HordeProcessInfo:
    info = make_mock_process_info(process_id, model_name=retained, process_type=process_type)
    info.retained_resident_model = retained
    info.device_index = device_index
    info.process_reserved_mb = reserved_mb
    info.last_process_state = HordeProcessState.INFERENCE_STARTING if busy else HordeProcessState.WAITING_FOR_JOB
    return info


class TestRepeatEvidence:
    """The trailing dispatch window a grant is predicted from."""

    def test_no_history_is_no_evidence(self) -> None:
        """A slot that has run nothing predicts nothing."""
        ledger = RetentionLedger(_Clock())
        assert ledger.slot_has_repeat_evidence(1, "a") is False

    def test_a_model_in_the_window_is_evidence(self) -> None:
        """Any of the last N dispatches naming the model is evidence for it."""
        ledger = RetentionLedger(_Clock())
        for model in ["a", "b", "c"]:
            ledger.record_slot_dispatch(1, model)
        assert ledger.slot_has_repeat_evidence(1, "a") is True
        assert ledger.slot_has_repeat_evidence(1, "z") is False

    def test_the_window_forgets_beyond_its_depth(self) -> None:
        """A model pushed out of the window by newer dispatches stops counting."""
        ledger = RetentionLedger(_Clock())
        ledger.record_slot_dispatch(1, "old")
        for i in range(RETENTION_REPEAT_EVIDENCE_DISPATCHES):
            ledger.record_slot_dispatch(1, f"new{i}")
        assert ledger.slot_has_repeat_evidence(1, "old") is False

    def test_exclude_latest_asks_the_question_issuance_asked(self) -> None:
        """Skipping the newest dispatch keeps a live grant from being its own evidence."""
        ledger = RetentionLedger(_Clock())
        ledger.record_slot_dispatch(1, "b")
        ledger.record_slot_dispatch(1, "a")
        assert ledger.slot_has_repeat_evidence(1, "a") is True
        assert ledger.slot_has_repeat_evidence(1, "a", exclude_latest=True) is False

    def test_slots_are_independent(self) -> None:
        """Evidence is per slot, since retention only pays through a successor on the same slot."""
        ledger = RetentionLedger(_Clock())
        ledger.record_slot_dispatch(1, "a")
        assert ledger.slot_has_repeat_evidence(2, "a") is False

    def test_window_depth_reads_the_module_constant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The searched depth follows the module constant so a sweep can shrink it to zero."""
        monkeypatch.setattr(retention_module, "RETENTION_REPEAT_EVIDENCE_DISPATCHES", 0)
        ledger = RetentionLedger(_Clock())
        ledger.record_slot_dispatch(1, "a")
        assert ledger.slot_has_repeat_evidence(1, "a") is False


class TestHoldAges:
    """When a retention episode starts, ends, and counts as stale."""

    def test_stamp_starts_and_keeps_an_episode(self) -> None:
        """An unstamped hold is stamped once; a stamped one keeps its original start."""
        clock = _Clock()
        ledger = RetentionLedger(clock)
        slot = _slot(1, retained="a")
        ledger.stamp_hold_ages([slot])
        assert slot.retained_resident_since == clock.now
        clock.now += 10.0
        ledger.stamp_hold_ages([slot])
        assert slot.retained_resident_since == clock.now - 10.0

    def test_an_ended_episode_carries_no_stamp(self) -> None:
        """A slot that no longer retains anything has its stamp cleared."""
        ledger = RetentionLedger(_Clock())
        slot = _slot(1, retained=None)
        slot.retained_resident_since = 5.0
        ledger.stamp_hold_ages([slot])
        assert slot.retained_resident_since is None

    def test_staleness_needs_a_stamp_and_the_horizon(self) -> None:
        """An unstamped hold is never stale; a stamped one is stale at the horizon."""
        clock = _Clock()
        ledger = RetentionLedger(clock)
        slot = _slot(1, retained="a")
        assert ledger.hold_is_stale(slot) is False
        ledger.stamp_hold_ages([slot])
        clock.now += RETENTION_STALE_HOLD_SECONDS - 0.5
        assert ledger.hold_is_stale(slot) is False
        clock.now += 0.5
        assert ledger.hold_is_stale(slot) is True

    def test_a_reuse_ends_the_episode_and_is_tallied(self) -> None:
        """A dispatch onto the retained model counts a reuse and resets the hold age."""
        ledger = RetentionLedger(_Clock())
        slot = _slot(1, retained="a")
        slot.retained_resident_since = 3.0
        ledger.note_reuse_if_retained(slot, "b")
        assert ledger.reuses == 0 and slot.retained_resident_since == 3.0
        ledger.note_reuse_if_retained(slot, "a")
        assert ledger.reuses == 1 and slot.retained_resident_since is None


class TestPressureDebounce:
    """The revoke sweep waits for pressure to have held, and a HEALTHY commit resets it."""

    def test_pressure_must_hold_for_the_debounce(self) -> None:
        """Off-HEALTHY reports below the debounce do not trigger; reaching it does."""
        clock = _Clock()
        ledger = RetentionLedger(clock)
        assert ledger.pressure_sustained(0, healthy=False) is False
        clock.now += RETENTION_PRESSURE_REVOKE_SECONDS
        assert ledger.pressure_sustained(0, healthy=False) is True

    def test_a_healthy_commit_resets_the_run(self) -> None:
        """A dip that clears never accumulates toward the debounce."""
        clock = _Clock()
        ledger = RetentionLedger(clock)
        ledger.pressure_sustained(0, healthy=False)
        clock.now += RETENTION_PRESSURE_REVOKE_SECONDS
        assert ledger.pressure_sustained(0, healthy=True) is False
        assert ledger.pressure_sustained(0, healthy=False) is False


class TestPendingEvictions:
    """In-flight evictions hold a card until the child evidences the free, boundedly."""

    def _world(self) -> tuple[RetentionLedger, HordeProcessInfo, HordeModelMap]:
        ledger = RetentionLedger(_Clock())
        slot = _slot(1, retained="a", reserved_mb=4000)
        slot.loaded_horde_model_name = "a"
        model_map = HordeModelMap(root={})
        model_map.update_entry("a", process_id=1, load_state=ModelLoadState.LOADED_IN_VRAM)
        ledger.record_pending_eviction(
            1, PendingRetentionEviction("a", reserved_baseline_mb=4000.0, device_free_baseline_mb=2000.0)
        )
        return ledger, slot, model_map

    def test_pending_is_scoped_to_the_card(self) -> None:
        """A pending eviction on card 0 holds card 0 and the whole worker, never card 1."""
        ledger, slot, _ = self._world()
        assert ledger.eviction_pending({1: slot}, 0) is True
        assert ledger.eviction_pending({1: slot}, None) is True
        assert ledger.eviction_pending({1: slot}, 1) is False
        assert ledger.eviction_pending({}, None) is False

    def test_a_fallen_reservation_evidences_the_free(self) -> None:
        """The slot's reservation dropping below its baseline releases the hold."""
        ledger, slot, model_map = self._world()
        slot.process_reserved_mb = 3000
        ledger.prune_confirmed_evictions({1: slot}, model_map, measured_free_mb=lambda _p: None)
        assert ledger.eviction_pending({1: slot}, None) is False

    def test_a_risen_device_free_evidences_the_free(self) -> None:
        """The card's free reading rising above its baseline releases the hold."""
        ledger, slot, model_map = self._world()
        ledger.prune_confirmed_evictions({1: slot}, model_map, measured_free_mb=lambda _p: 2500.0)
        assert ledger.eviction_pending({1: slot}, None) is False

    def test_the_map_moving_the_weights_evidences_the_free(self) -> None:
        """The model map no longer placing the weights on the slot releases the hold."""
        ledger, slot, model_map = self._world()
        model_map.expire_entry("a")
        ledger.prune_confirmed_evictions({1: slot}, model_map, measured_free_mb=lambda _p: None)
        assert ledger.eviction_pending({1: slot}, None) is False

    def test_without_evidence_the_hold_is_bounded(self) -> None:
        """Absent every signal the record survives a bounded number of passes, then drops."""
        ledger, slot, model_map = self._world()
        for _ in range(RETENTION_EVICTION_CONFIRMATION_PASSES - 1):
            ledger.prune_confirmed_evictions({1: slot}, model_map, measured_free_mb=lambda _p: 2000.0)
            assert ledger.eviction_pending({1: slot}, None) is True
        ledger.prune_confirmed_evictions({1: slot}, model_map, measured_free_mb=lambda _p: 2000.0)
        assert ledger.eviction_pending({1: slot}, None) is False


class TestTallies:
    """The session counters partition retention outcomes."""

    def test_denials_are_bucketed_by_gate(self) -> None:
        """Each refusal lands under the gate that refused."""
        ledger = RetentionLedger(_Clock())
        ledger.note_denial(RetentionDenialReason.STATIC_FIT)
        ledger.note_denial(RetentionDenialReason.STATIC_FIT)
        ledger.note_denial(RetentionDenialReason.NO_REPEAT_EVIDENCE)
        assert dict(ledger.grant_denials) == {
            RetentionDenialReason.STATIC_FIT: 2,
            RetentionDenialReason.NO_REPEAT_EVIDENCE: 1,
        }

    def test_evicted_unused_counts_only_retaining_slots(self) -> None:
        """Giving back a slot that retained nothing is not an unused retention."""
        ledger = RetentionLedger(_Clock())
        ledger.note_evicted_unused(_slot(1, retained=None))
        ledger.note_evicted_unused(_slot(2, retained="a"))
        assert ledger.evicted_unused == 1


class TestCardArithmetic:
    """Pure charges read off the process set."""

    def test_sibling_retained_resident_excludes_the_target_and_other_cards(self) -> None:
        """Only another slot on the same card counts as a sibling retainer."""
        target = _slot(1, retained="a")
        assert sibling_retained_resident_present([target], target_id=1, device_index=None) is False
        other_card = _slot(2, retained="b", device_index=1)
        assert sibling_retained_resident_present([target, other_card], target_id=1, device_index=0) is False
        assert sibling_retained_resident_present([target, other_card], target_id=1, device_index=None) is True

    def test_idle_resident_reclaimable_mb_prices_reservations_of_idle_retainers(self) -> None:
        """Busy slots, unreserved slots and non-inference processes add nothing."""
        idle = _slot(1, retained="a", reserved_mb=3000)
        busy = _slot(2, retained="b", reserved_mb=3000, busy=True)
        unread = _slot(3, retained="c", reserved_mb=None)
        lane = _slot(4, retained="d", reserved_mb=3000, process_type=HordeProcessType.POST_PROCESS)
        assert idle_resident_reclaimable_mb([idle, busy, unread, lane], None) == 3000.0

    def test_a_model_left_loaded_without_a_grant_is_reclaimable_too(self) -> None:
        """The budget-off regime grants no retention, yet the weights stay on the card; eviction returns them."""
        ungranted = _slot(1, reserved_mb=2500)
        ungranted.loaded_horde_model_name = "a"
        empty = _slot(2, reserved_mb=2500)
        empty.loaded_horde_model_name = None
        assert idle_resident_reclaimable_mb([ungranted, empty], None) == 2500.0

    def test_idle_lane_component_charges_skip_target_and_busy_lanes(self) -> None:
        """Held components are charged for idle lanes other than the target."""
        target = _slot(1)
        target.held_components = [Mock(approx_ram_mb=500.0)]
        idle_lane = _slot(2, process_type=HordeProcessType.COMPONENT)
        idle_lane.held_components = [Mock(approx_ram_mb=700.0), Mock(approx_ram_mb=-5.0)]
        busy_lane = _slot(3, process_type=HordeProcessType.COMPONENT, busy=True)
        busy_lane.held_components = [Mock(approx_ram_mb=900.0)]
        assert idle_lane_component_charges_mb([target, idle_lane, busy_lane], target_id=1, device_index=None) == 700.0

    def test_a_lane_holding_components_in_ram_is_charged_its_reservation(self) -> None:
        """A cache entry in host RAM costs the card nothing, so the lane's small reservation is its charge."""
        lane = _slot(2, reserved_mb=_IDLE_LANE_RESERVED_MB)
        lane.held_components = [HeldComponentSnapshot(kind="checkpoint", identity="a", approx_ram_mb=7000.0)]
        assert idle_lane_component_charges_mb([lane], target_id=1, device_index=None) == _IDLE_LANE_RESERVED_MB

    def test_a_lane_with_no_reservation_reading_is_charged_the_cache_estimate(self) -> None:
        """Before a lane's first memory report the RAM estimate stands in, so nothing is under-charged."""
        lane = _slot(2, reserved_mb=None)
        lane.held_components = [
            HeldComponentSnapshot(kind="unet", identity="a", approx_ram_mb=5000.0),
            HeldComponentSnapshot(kind="clip", identity="a", approx_ram_mb=1600.0),
        ]
        assert idle_lane_component_charges_mb([lane], target_id=1, device_index=None) == 6600.0

    def test_a_lane_holding_components_on_the_device_is_charged_in_full(self) -> None:
        """A component lane's device-held weights are inside its reservation, which can exceed the RAM estimate."""
        lane = _slot(2, reserved_mb=7800, process_type=HordeProcessType.COMPONENT)
        lane.held_components = [HeldComponentSnapshot(kind="unet", identity="a", approx_ram_mb=7600.0)]
        assert idle_lane_component_charges_mb([lane], target_id=1, device_index=None) == 7800.0

    def test_a_retaining_lane_is_charged_once(self) -> None:
        """A retainer's reservation holds the retained weights, which the retained-resident term already charges."""
        retainer = _slot(2, retained="b", reserved_mb=7000)
        retainer.held_components = [HeldComponentSnapshot(kind="checkpoint", identity="b", approx_ram_mb=7000.0)]
        processes = [_slot(1), retainer]

        component_mb = idle_lane_component_charges_mb(processes, target_id=1, device_index=None)
        retained_mb = retained_resident_charges_mb(
            processes,
            target_id=1,
            dispatched_model="a",
            device_index=None,
            include_target_retained=False,
            footprint_mb=lambda _p, _model: 6800.0,
        )

        assert component_mb == 0.0
        assert retained_mb == 6800.0

    def test_sibling_context_count_follows_the_safety_placement(self) -> None:
        """The safety process holds a context only while it is permitted on the GPU."""
        processes = [
            _slot(1),
            _slot(2),
            _slot(3, process_type=HordeProcessType.POST_PROCESS),
            _slot(4, process_type=HordeProcessType.SAFETY),
            _slot(5, process_type=HordeProcessType.DOWNLOAD),
        ]
        assert sibling_context_count(processes, target_id=1, device_index=None, safety_on_gpu=False) == 2
        assert sibling_context_count(processes, target_id=1, device_index=None, safety_on_gpu=True) == 3

    def test_retained_resident_charges_skip_the_same_model_and_deny_on_unpriceable(self) -> None:
        """A same-model re-grant charges nothing; an unpriceable tenant returns None."""
        target = _slot(1, retained="a")
        sibling = _slot(2, retained="b")
        footprints = {"a": 1000.0, "b": 2000.0}

        def priced(_p: HordeProcessInfo, model: str) -> float | None:
            return footprints.get(model)

        kwargs = {"target_id": 1, "dispatched_model": "a", "device_index": None, "footprint_mb": priced}
        assert retained_resident_charges_mb([target, sibling], include_target_retained=True, **kwargs) == 2000.0
        target.retained_resident_model = "c"
        assert retained_resident_charges_mb([target, sibling], include_target_retained=False, **kwargs) == 2000.0
        footprints["c"] = 500.0
        assert retained_resident_charges_mb([target, sibling], include_target_retained=True, **kwargs) == 2500.0
        del footprints["b"]
        assert retained_resident_charges_mb([target, sibling], include_target_retained=True, **kwargs) is None


class TestRetentionFit:
    """The static fit de-stacks the operator reserve and treats an unknown peak as fitting."""

    def _fit(self, predicted_mb: float | None) -> RetentionFit:
        return RetentionFit(
            predicted_mb=predicted_mb,
            noise_mb=500.0,
            total_vram_mb=16000.0,
            static_charges_mb=3000.0,
            retained_resident_mb=4000.0,
            committed_reserve_mb=1000.0,
            foreign_floor_mb=2000.0,
        )

    def test_effective_available_nets_every_charge(self) -> None:
        """The card total less contexts, retained residents, commitments and the foreign floor is the room."""
        assert self._fit(1.0).effective_available_mb == 6000.0

    def test_granted_at_the_boundary_and_denied_past_it(self) -> None:
        """Peak plus noise at exactly the room fits; one MB more does not."""
        assert self._fit(5500.0).granted is True
        assert self._fit(5501.0).granted is False

    def test_an_unknown_peak_cannot_be_refused_statically(self) -> None:
        """A static gate has no figure to refuse on when the peak is unknown."""
        assert self._fit(None).granted is True

    def test_describe_names_the_retained_charge_and_floor_only_when_present(self) -> None:
        """The log line carries the retained and foreign-floor figures only when each is nonzero."""
        described = self._fit(1.0).describe()
        assert "retained residents 4000MB" in described
        assert "foreign floor 2000MB" in described
        bare = RetentionFit(1.0, 0.0, 16000.0, 0.0, 0.0, 0.0, 0.0)
        assert "retained residents" not in bare.describe()
        assert "foreign floor" not in bare.describe()

    def test_retained_residents_decide_only_when_returning_them_makes_room(self) -> None:
        """Denied as charged and granted with the retained charge returned; anything else is not theirs to fix."""
        # 6000MB room plus the 4000MB retained charge is 10000MB once they are gone; noise is 500MB.
        assert self._fit(5500.0).retained_residents_decide is False, "already fits"
        assert self._fit(5501.0).retained_residents_decide is True
        assert self._fit(9500.0).retained_residents_decide is True
        assert self._fit(9501.0).retained_residents_decide is False, "does not fit even without them"
        assert self._fit(None).retained_residents_decide is False


def _lane(
    process_id: int,
    *,
    model: str | None = _OTHER_SDXL_MODEL,
    state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB,
    process_type: HordeProcessType = HordeProcessType.INFERENCE,
    reserved_mb: int | None = None,
    retained: str | None = None,
) -> HordeProcessInfo:
    info = make_mock_process_info(process_id, model_name=model, state=state, process_type=process_type)
    info.total_vram_mb = _CARD_24GB_MB
    info.process_reserved_mb = reserved_mb
    info.retained_resident_model = retained
    return info


def _sdxl_scheduler(job_tracker: JobTracker, processes: list[HordeProcessInfo]) -> InferenceScheduler:
    """A budget-on scheduler over a 24 GB card whose SDXL models have repeat evidence on every lane."""
    models = (_SDXL_MODEL, _OTHER_SDXL_MODEL)
    reference = {
        model: make_mock_model_reference_record(model, baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl)
        for model in models
    }
    scheduler = _make_inference_scheduler(
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(enable_vram_budget=True, vram_reserve_mb=2048, ram_reserve_mb=4096),
        process_map=ProcessMap({p.process_id: p for p in processes}),
        model_metadata=make_test_model_metadata(reference),
    )
    # The seeded per-context figure stands in for a probe, which a sibling context needs before it can be priced.
    scheduler._overhead.set_marginal_overhead_mb(_SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB)
    for process_info in processes:
        for model in models:
            scheduler.retention.record_slot_dispatch(process_info.process_id, model)
    return scheduler


def _fix_peak(scheduler: InferenceScheduler, peak_box: list[float]) -> None:
    """Make the static fit price ``peak_box[0]`` as the job's peak, so a test places it against the room."""

    def fixed_peak(job, baseline, free_vram_mb, committed_reserve_mb=0.0, *, disaggregated=False):  # noqa: ANN001, ANN202
        return Mock(fits=True, predicted_mb=peak_box[0], reserve_mb=0.0)

    scheduler._vram_budget.check_job = fixed_peak  # type: ignore[method-assign]


def _sdxl_checkpoint_mb(scheduler: InferenceScheduler, job: ImageGenerateJobPopResponse) -> float:
    footprint_mb = predict_job_footprint_mb(job, scheduler._model_metadata.get_baseline(_SDXL_MODEL))
    assert footprint_mb is not None
    return footprint_mb


class TestIdleLaneChargeInTheFit:
    """The grant gate charges an idle lane what it holds on the card."""

    async def test_two_lanes_seated_in_ram_leave_room_for_an_sdxl_retention(self) -> None:
        """Four lanes and a post-processing lane on 24 GB, two lanes idle with SDXL checkpoints in RAM: a grant fits.

        Charged at the cache's RAM estimate, the two seated lanes alone would take more than half the card and
        deny a 1024x1024 SDXL retention, so each reload is paid again on the next same-model job.
        """
        job_tracker = JobTracker()
        job = make_job_pop_response(_SDXL_MODEL, width=1024, height=1024)
        await track_popped_job_async(job_tracker, job)
        target = _lane(1, model=_SDXL_MODEL)
        seated = [_lane(process_id, reserved_mb=_IDLE_LANE_RESERVED_MB) for process_id in (2, 3)]
        sampling = _lane(4, state=HordeProcessState.INFERENCE_STARTING, reserved_mb=_CARD_24GB_MB // 2)
        post_processing = _lane(5, model=None, process_type=HordeProcessType.POST_PROCESS)
        scheduler = _sdxl_scheduler(job_tracker, [target, *seated, sampling, post_processing])
        checkpoint_mb = DEFAULT_APPROX_RAM_MB[ComponentSlotKind.CHECKPOINT]
        for lane in seated:
            lane.held_components = [
                HeldComponentSnapshot(kind="checkpoint", identity=_OTHER_SDXL_MODEL, approx_ram_mb=checkpoint_mb),
            ]

        fit = scheduler._retention_fit(job, target=target, device_index=None, include_target_retained=True)

        assert isinstance(fit, RetentionFit) and fit.predicted_mb is not None
        ram_charged_room_mb = fit.effective_available_mb - 2 * (checkpoint_mb - _IDLE_LANE_RESERVED_MB)
        assert fit.predicted_mb + fit.noise_mb > ram_charged_room_mb, "premise: the RAM estimate denies the grant"
        assert scheduler._should_keep_model_resident(job, process_with_model=target, device_index=None) is True

    async def test_a_lane_holding_components_on_the_device_still_denies_a_grant_past_the_card(self) -> None:
        """A component lane's device-held weights stay charged, so two claims cannot jointly overflow the card.

        The idle lane's reservation is its cached UNet on the device. Uncharged, the grant reads as a fit and the
        next load packs the card past what it physically has.
        """
        job_tracker = JobTracker()
        job = make_job_pop_response(_SDXL_MODEL, width=1024, height=1024)
        await track_popped_job_async(job_tracker, job)
        target = _lane(1, model=_SDXL_MODEL)
        component_lane = _lane(2, model=None, process_type=HordeProcessType.COMPONENT)
        scheduler = _sdxl_scheduler(job_tracker, [target, component_lane])
        peak_box = [0.0]
        _fix_peak(scheduler, peak_box)
        empty_lane_fit = scheduler._retention_fit(job, target=target, device_index=None, include_target_retained=True)
        assert isinstance(empty_lane_fit, RetentionFit)
        device_held_mb = _sdxl_checkpoint_mb(scheduler, job)
        # The peak fits the card with the lane empty and does not fit once its device-held UNet is charged.
        peak_box[0] = empty_lane_fit.effective_available_mb - empty_lane_fit.noise_mb - device_held_mb / 2
        assert scheduler._should_keep_model_resident(job, process_with_model=target, device_index=None) is True

        component_lane.process_reserved_mb = int(device_held_mb)
        component_lane.held_components = [
            HeldComponentSnapshot(kind="unet", identity=_OTHER_SDXL_MODEL, approx_ram_mb=device_held_mb),
        ]

        assert scheduler._should_keep_model_resident(job, process_with_model=target, device_index=None) is False


class TestRetainedResidentDispatchHold:
    """The dispatch hold evicts a sibling's retained weights only when that eviction makes the load fit."""

    async def _world(
        self,
    ) -> tuple[InferenceScheduler, HordeProcessInfo, HordeProcessInfo, ImageGenerateJobPopResponse]:
        job_tracker = JobTracker()
        job = make_job_pop_response(_SDXL_MODEL, width=1024, height=1024)
        await track_popped_job_async(job_tracker, job)
        target = _lane(1, model=_SDXL_MODEL)
        retainer = _lane(2, retained=_OTHER_SDXL_MODEL, reserved_mb=_CARD_24GB_MB // 3)
        scheduler = _sdxl_scheduler(job_tracker, [target, retainer])
        scheduler.unload_idle_model = Mock(return_value=True)  # type: ignore[method-assign]
        return scheduler, target, retainer, job

    def _place_peak(
        self,
        scheduler: InferenceScheduler,
        target: HordeProcessInfo,
        job: ImageGenerateJobPopResponse,
        *,
        beyond_room_by_retained_share: float,
    ) -> None:
        peak_box = [0.0]
        _fix_peak(scheduler, peak_box)
        fit = scheduler._retention_fit(job, target=target, device_index=None, include_target_retained=False)
        assert isinstance(fit, RetentionFit) and fit.retained_resident_mb > 0
        peak_box[0] = (
            fit.effective_available_mb - fit.noise_mb + fit.retained_resident_mb * beyond_room_by_retained_share
        )

    async def test_holds_and_evicts_when_the_retained_residents_decide_the_fit(self) -> None:
        """Returning the retained weights makes room, so the dispatch waits for them and they are asked back."""
        scheduler, target, retainer, job = await self._world()
        self._place_peak(scheduler, target, job, beyond_room_by_retained_share=0.5)

        assert scheduler._retained_resident_dispatch_holds(job, target) is True
        unload = scheduler.unload_idle_model
        assert isinstance(unload, Mock)
        unload.assert_called_once()
        assert unload.call_args.args[0] == retainer.process_id

    async def test_leaves_the_retained_weights_when_eviction_cannot_make_the_fit(self) -> None:
        """The load does not fit even with the retained charge returned, so the weights stay and nothing waits."""
        scheduler, target, retainer, job = await self._world()
        self._place_peak(scheduler, target, job, beyond_room_by_retained_share=1.5)

        assert scheduler._retained_resident_dispatch_holds(job, target) is False
        unload = scheduler.unload_idle_model
        assert isinstance(unload, Mock)
        unload.assert_not_called()
        assert retainer.retained_resident_model == _OTHER_SDXL_MODEL


class TestWddmPagingRecord:
    """The paging verdict is recorded with a rising edge, a refreshed victim set, and a freshness bound."""

    def test_rising_edge_fires_once_per_episode(self) -> None:
        """Only the first active verdict of an episode is the edge; clearing and re-raising starts a new one."""
        ledger = RetentionLedger(_Clock())
        assert ledger.note_wddm_paging({1: 512.0}, active=True) is True
        assert ledger.note_wddm_paging({1: 600.0}, active=True) is False
        assert ledger.wddm_paging_victims_shared_mb_by_pid == {1: 600.0}, "a repeat refreshes the victims"
        assert ledger.note_wddm_paging({}, active=False) is False
        assert ledger.wddm_paging_active is False
        assert ledger.wddm_paging_victims_shared_mb_by_pid == {}
        assert ledger.note_wddm_paging({2: 100.0}, active=True) is True

    def test_victims_age_out_on_the_ledger_clock(self) -> None:
        """A verdict older than the freshness window yields no victims."""
        clock = _Clock()
        ledger = RetentionLedger(clock)
        ledger.note_wddm_paging({1: 512.0}, active=True)
        assert ledger.wddm_paging_victims(5.0) == {1: 512.0}
        clock.now += 5.0
        assert ledger.wddm_paging_victims(5.0) == {1: 512.0}
        clock.now += 0.1
        assert ledger.wddm_paging_victims(5.0) == {}
        assert ledger.wddm_paging_victims(100.0) == {1: 512.0}
