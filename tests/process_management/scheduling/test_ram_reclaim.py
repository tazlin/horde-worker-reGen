"""The pure RAM reclaim selections and the reclaim ledger, stated without a scheduler.

The scheduler's own tests cover the actuation these feed (cycling a slot, sending an unload); these rows pin the
selection rules and the ledger's clocks directly, so a change to either is caught where it lives.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.ram_reclaim import (
    CREEP_CONTAINMENT_RSS_BYTES,
    FRESH_INFERENCE_CHILD_BASELINE_MB,
    LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS,
    LANE_RAM_CONTAINMENT_RSS_BYTES,
    RAM_RECLAIM_CYCLE_GRACE_SECONDS,
    REUSE_CREDIT_RECONCILE_SETTLE_SECONDS,
    REUSE_CREDIT_RECONCILE_SLACK_MB,
    STALE_RAM_UNLOAD_CYCLE_MIN_INTERVAL_SECONDS,
    RamCycleReason,
    RamReclaimLedger,
    ReuseCreditKind,
    idle_lanes_over_ram_ceiling,
    select_ram_cycle_victim,
    staging_reuse_credit_mb,
    stale_ram_unload_replace_bytes,
)
from tests.process_management.conftest import make_mock_process_info, mark_ram_unload_settled

_MB = 1024 * 1024
_TOTAL_RAM_MB = 64 * 1024.0
_STALE_BYTES = stale_ram_unload_replace_bytes(_TOTAL_RAM_MB)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _stale_slot(process_id: int = 0, *, rss_bytes: int | None = None) -> HordeProcessInfo:
    """An idle, model-less slot whose post-unload reading still shows retained pages."""
    slot = make_mock_process_info(process_id, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    slot.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM
    slot.ram_usage_bytes = int(_STALE_BYTES) + _MB if rss_bytes is None else rss_bytes
    mark_ram_unload_settled(slot)
    return slot


def _select(*slots: HordeProcessInfo, protect: int | None = None, cycled_at: dict[int, float] | None = None):  # noqa: ANN202
    return select_ram_cycle_victim(
        slots,
        now=1_000.0,
        protect_process_id=protect,
        cycled_at=cycled_at or {},
        stale_replace_bytes=_STALE_BYTES,
    )


class TestStaleThreshold:
    """The stale-unload threshold sits above what a freshly spawned child can reach."""

    @pytest.mark.parametrize("total_ram_mb", [8 * 1024.0, 16 * 1024.0, 64 * 1024.0, 256 * 1024.0])
    def test_a_fresh_child_never_qualifies(self, total_ram_mb: float) -> None:
        """On any host size, a slot at the cold baseline is below the threshold."""
        assert stale_ram_unload_replace_bytes(total_ram_mb) > FRESH_INFERENCE_CHILD_BASELINE_MB * _MB

    def test_the_margin_scales_with_the_host_above_the_floor(self) -> None:
        """A large host's threshold is higher than a small host's; a small host still clears the floor."""
        small = stale_ram_unload_replace_bytes(8 * 1024.0)
        large = stale_ram_unload_replace_bytes(256 * 1024.0)
        assert large > small
        assert small > FRESH_INFERENCE_CHILD_BASELINE_MB * _MB


class TestStagingReuseCredit:
    """The credit is the idle target's resident RSS above a fresh child's baseline."""

    def test_idle_retaining_target(self) -> None:
        """An idle slot above baseline yields the excess."""
        slot = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        slot.ram_usage_bytes = int(8000.0 * _MB)
        assert staging_reuse_credit_mb(slot) == pytest.approx(8000.0 - FRESH_INFERENCE_CHILD_BASELINE_MB)

    def test_busy_and_fresh_targets_earn_nothing(self) -> None:
        """Pages in live use are not reusable, and a slot at baseline has none to reuse."""
        busy = make_mock_process_info(0, model_name="m", state=HordeProcessState.INFERENCE_PRIMED)
        busy.ram_usage_bytes = int(9000.0 * _MB)
        fresh = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        fresh.ram_usage_bytes = int((FRESH_INFERENCE_CHILD_BASELINE_MB - 1.0) * _MB)
        assert staging_reuse_credit_mb(busy) == 0.0
        assert staging_reuse_credit_mb(fresh) == 0.0


class TestSelectRamCycleVictim:
    """Creep victims are taken ahead of stale-unload victims, and each trigger has its own guards."""

    def test_stale_slot_is_selected_with_its_reason(self) -> None:
        """The intended victim: idle, model-less, unloaded, settled, above the threshold."""
        slot = _stale_slot()
        assert _select(slot) == (slot, RamCycleReason.STALE_UNLOAD)

    def test_creep_wins_over_stale(self) -> None:
        """A crept slot is contained before an ordinary retained-page reclaim, whatever the scan order."""
        stale = _stale_slot(0)
        crept = make_mock_process_info(1, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)
        crept.ram_usage_bytes = CREEP_CONTAINMENT_RSS_BYTES + _MB
        assert _select(stale, crept) == (crept, RamCycleReason.CREEP)
        assert _select(crept, stale) == (crept, RamCycleReason.CREEP)

    def test_protection_spares_the_stale_trigger_but_not_creep(self) -> None:
        """The head's staging target keeps its reusable pages, unless those pages are leak."""
        stale = _stale_slot(0)
        assert _select(stale, protect=0) is None
        stale.ram_usage_bytes = CREEP_CONTAINMENT_RSS_BYTES + _MB
        assert _select(stale, protect=0) == (stale, RamCycleReason.CREEP)

    def test_a_slot_routed_a_preload_is_off_limits_to_both_triggers(self) -> None:
        """Reaping a mid-stage slot would fault the head's load."""
        slot = _stale_slot(0, rss_bytes=CREEP_CONTAINMENT_RSS_BYTES + _MB)
        slot.last_control_flag = HordeControlFlag.PRELOAD_MODEL
        assert _select(slot) is None

    def test_busy_resident_or_unsettled_slots_are_not_stale(self) -> None:
        """Each stale precondition on its own withholds the selection."""
        busy = _stale_slot(0)
        busy.last_process_state = HordeProcessState.INFERENCE_STARTING
        resident = _stale_slot(1)
        resident.loaded_horde_model_name = "still-here"
        unsettled = _stale_slot(2)
        unsettled.report_sampled_at = None
        below = _stale_slot(3, rss_bytes=int(_STALE_BYTES) - _MB)
        assert _select(busy, resident, unsettled, below) is None

    def test_a_recently_cycled_slot_waits_out_the_interval(self) -> None:
        """One slot contributes at most one stale cycle per interval; another eligible slot is taken instead."""
        first = _stale_slot(0)
        second = _stale_slot(1)
        recent = {0: 1_000.0 - STALE_RAM_UNLOAD_CYCLE_MIN_INTERVAL_SECONDS + 1.0}
        assert _select(first, second, cycled_at=recent) == (second, RamCycleReason.STALE_UNLOAD)
        assert _select(first, cycled_at=recent) is None
        expired = {0: 1_000.0 - STALE_RAM_UNLOAD_CYCLE_MIN_INTERVAL_SECONDS}
        assert _select(first, cycled_at=expired) == (first, RamCycleReason.STALE_UNLOAD)

    def test_only_inference_slots_are_candidates(self) -> None:
        """A service lane above every ceiling is never cycled by this path."""
        lane = make_mock_process_info(0, model_name=None, process_type=HordeProcessType.COMPONENT)
        lane.ram_usage_bytes = CREEP_CONTAINMENT_RSS_BYTES + _MB
        assert _select(lane) is None


class TestIdleLanesOverCeiling:
    """Only idle, live service lanes above the ceiling and outside their throttle are selected."""

    def _lane(self, process_id: int, process_type: HordeProcessType, rss_bytes: int) -> HordeProcessInfo:
        lane = make_mock_process_info(process_id, model_name=None, process_type=process_type)
        lane.ram_usage_bytes = rss_bytes
        return lane

    def test_component_and_vae_lanes_above_the_ceiling(self) -> None:
        """Both lane kinds qualify; an inference slot and a lane under the ceiling do not."""
        component = self._lane(0, HordeProcessType.COMPONENT, LANE_RAM_CONTAINMENT_RSS_BYTES + 1)
        vae = self._lane(1, HordeProcessType.VAE_LANE, LANE_RAM_CONTAINMENT_RSS_BYTES + 1)
        under = self._lane(2, HordeProcessType.COMPONENT, LANE_RAM_CONTAINMENT_RSS_BYTES - 1)
        inference = self._lane(3, HordeProcessType.INFERENCE, LANE_RAM_CONTAINMENT_RSS_BYTES + 1)
        selected = idle_lanes_over_ram_ceiling([component, vae, under, inference], now=1_000.0, contained_at={})
        assert selected == [component, vae]

    def test_a_busy_lane_is_left_alone(self) -> None:
        """An unload waits for the lane to be idle."""
        lane = self._lane(0, HordeProcessType.COMPONENT, LANE_RAM_CONTAINMENT_RSS_BYTES + 1)
        lane.last_process_state = HordeProcessState.INFERENCE_STARTING
        assert idle_lanes_over_ram_ceiling([lane], now=1_000.0, contained_at={}) == []

    def test_the_throttle_holds_until_the_interval_elapses(self) -> None:
        """A lane asked to unload recently is skipped; after the interval it is asked again."""
        lane = self._lane(0, HordeProcessType.VAE_LANE, LANE_RAM_CONTAINMENT_RSS_BYTES + 1)
        recent = {0: 1_000.0 - LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS + 1.0}
        assert idle_lanes_over_ram_ceiling([lane], now=1_000.0, contained_at=recent) == []
        expired = {0: 1_000.0 - LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS}
        assert idle_lanes_over_ram_ceiling([lane], now=1_000.0, contained_at=expired) == [lane]


class TestRamReclaimLedger:
    """Reuse credits settle against measured growth, and the cycle grace is bounded by the clock."""

    def _credited(self, ledger: RamReclaimLedger, *, rss_mb: float, charge_mb: float) -> HordeProcessInfo:
        target = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        target.ram_usage_bytes = int(rss_mb * _MB)
        ledger.record_credit(target, model="m", effective_charge_mb=charge_mb, kind=ReuseCreditKind.PAGE_REUSE)
        return target

    def test_record_captures_the_admit_baseline(self) -> None:
        """The record holds the target's RSS at admit time and the clock's admission stamp."""
        clock = _Clock()
        ledger = RamReclaimLedger(clock)
        self._credited(ledger, rss_mb=4000.0, charge_mb=1000.0)
        record = ledger.pending_reuse_credits[0]
        assert record.rss_at_admit_mb == pytest.approx(4000.0)
        assert record.admitted_at == clock.now
        assert record.kind is ReuseCreditKind.PAGE_REUSE

    def test_a_credit_does_not_settle_before_the_grace(self) -> None:
        """Inside the settle grace the record stays pending and nothing is reported."""
        clock = _Clock()
        ledger = RamReclaimLedger(clock)
        target = self._credited(ledger, rss_mb=4000.0, charge_mb=1000.0)
        target.loaded_horde_model_name = "m"
        clock.now += REUSE_CREDIT_RECONCILE_SETTLE_SECONDS - 1.0
        assert ledger.settle_credits(ProcessMap({0: target})) == []
        assert 0 in ledger.pending_reuse_credits

    def test_growth_within_slack_settles_silently(self) -> None:
        """Growth at the charge plus slack retires the record without a discrepancy."""
        clock = _Clock()
        ledger = RamReclaimLedger(clock)
        target = self._credited(ledger, rss_mb=4000.0, charge_mb=1000.0)
        target.loaded_horde_model_name = "m"
        target.ram_usage_bytes = int((4000.0 + 1000.0 + REUSE_CREDIT_RECONCILE_SLACK_MB) * _MB)
        clock.now += REUSE_CREDIT_RECONCILE_SETTLE_SECONDS
        assert ledger.settle_credits(ProcessMap({0: target})) == []
        assert 0 not in ledger.pending_reuse_credits

    def test_growth_past_slack_is_reported_once(self) -> None:
        """A too-generous credit comes back as one discrepancy carrying the measured growth."""
        clock = _Clock()
        ledger = RamReclaimLedger(clock)
        target = self._credited(ledger, rss_mb=4000.0, charge_mb=1000.0)
        target.loaded_horde_model_name = "m"
        target.ram_usage_bytes = int((4000.0 + 1000.0 + REUSE_CREDIT_RECONCILE_SLACK_MB + 1.0) * _MB)
        clock.now += REUSE_CREDIT_RECONCILE_SETTLE_SECONDS
        discrepancies = ledger.settle_credits(ProcessMap({0: target}))
        assert len(discrepancies) == 1
        assert discrepancies[0].process_id == 0
        assert discrepancies[0].growth_mb == pytest.approx(1000.0 + REUSE_CREDIT_RECONCILE_SLACK_MB + 1.0)
        assert ledger.settle_credits(ProcessMap({0: target})) == []

    def test_a_vanished_slot_drops_its_record(self) -> None:
        """A record for a process no longer in the map is discarded, never reported."""
        ledger = RamReclaimLedger(_Clock())
        self._credited(ledger, rss_mb=4000.0, charge_mb=1000.0)
        assert ledger.settle_credits(ProcessMap({})) == []
        assert ledger.pending_reuse_credits == {}

    def test_void_drops_a_pending_credit(self) -> None:
        """A cycled slot's credit is void; voiding an unknown slot is a no-op."""
        ledger = RamReclaimLedger(_Clock())
        self._credited(ledger, rss_mb=4000.0, charge_mb=1000.0)
        ledger.void_credit(0)
        ledger.void_credit(7)
        assert ledger.pending_reuse_credits == {}

    def test_announce_admission_is_edge_triggered(self) -> None:
        """The same (target, model, charge) key announces once; a changed key announces again."""
        ledger = RamReclaimLedger(_Clock())
        assert ledger.announce_admission((0, "m", 1000)) is True
        assert ledger.announce_admission((0, "m", 1000)) is False
        assert ledger.announce_admission((0, "m", 1001)) is True

    def test_cycle_grace_is_bounded_by_the_clock(self) -> None:
        """No cycle means no grace; a cycle opens it for exactly the grace window."""
        clock = _Clock()
        ledger = RamReclaimLedger(clock)
        assert ledger.cycle_grace_active() is False
        ledger.note_cycle()
        assert ledger.cycle_grace_active() is True
        clock.now += RAM_RECLAIM_CYCLE_GRACE_SECONDS - 1.0
        assert ledger.cycle_grace_active() is True
        clock.now += 1.0
        assert ledger.cycle_grace_active() is False
