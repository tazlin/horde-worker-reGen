"""Tests for the retention ledger and the pure per-card retention arithmetic."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.scheduling import retention as retention_module
from horde_worker_regen.process_management.scheduling.retention import (
    RETENTION_EVICTION_CONFIRMATION_PASSES,
    RETENTION_PRESSURE_REVOKE_SECONDS,
    RETENTION_REPEAT_EVIDENCE_DISPATCHES,
    RETENTION_STALE_HOLD_SECONDS,
    PendingRetentionEviction,
    RetentionDenialReason,
    RetentionFit,
    RetentionLedger,
    idle_lane_component_charges_mb,
    idle_retained_resident_mb,
    retained_resident_charges_mb,
    sibling_context_count,
    sibling_retained_resident_present,
)
from tests.process_management.conftest import make_mock_process_info


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

    def test_idle_retained_resident_mb_prices_reservations_of_idle_retainers(self) -> None:
        """Busy slots, unreserved slots and non-inference processes add nothing."""
        idle = _slot(1, retained="a", reserved_mb=3000)
        busy = _slot(2, retained="b", reserved_mb=3000, busy=True)
        unread = _slot(3, retained="c", reserved_mb=None)
        lane = _slot(4, retained="d", reserved_mb=3000, process_type=HordeProcessType.POST_PROCESS)
        assert idle_retained_resident_mb([idle, busy, unread, lane], None) == 3000.0

    def test_idle_lane_component_charges_skip_target_and_busy_lanes(self) -> None:
        """Held components are charged for idle lanes other than the target."""
        target = _slot(1)
        target.held_components = [Mock(approx_ram_mb=500.0)]
        idle_lane = _slot(2, process_type=HordeProcessType.COMPONENT)
        idle_lane.held_components = [Mock(approx_ram_mb=700.0), Mock(approx_ram_mb=-5.0)]
        busy_lane = _slot(3, process_type=HordeProcessType.COMPONENT, busy=True)
        busy_lane.held_components = [Mock(approx_ram_mb=900.0)]
        assert idle_lane_component_charges_mb([target, idle_lane, busy_lane], target_id=1, device_index=None) == 700.0

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
        )

    def test_effective_available_nets_every_charge(self) -> None:
        """The card total less contexts, retained residents and commitments is what the peak must fit."""
        assert self._fit(1.0).effective_available_mb == 8000.0

    def test_granted_at_the_boundary_and_denied_past_it(self) -> None:
        """Peak plus noise at exactly the room fits; one MB more does not."""
        assert self._fit(7500.0).granted is True
        assert self._fit(7501.0).granted is False

    def test_an_unknown_peak_cannot_be_refused_statically(self) -> None:
        """A static gate has no figure to refuse on when the peak is unknown."""
        assert self._fit(None).granted is True

    def test_describe_names_the_retained_charge_only_when_present(self) -> None:
        """The log line carries the retained figure only when something is retained."""
        assert "retained residents 4000MB" in self._fit(1.0).describe()
        bare = RetentionFit(1.0, 0.0, 16000.0, 0.0, 0.0, 0.0)
        assert "retained residents" not in bare.describe()
