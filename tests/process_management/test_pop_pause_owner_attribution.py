"""Attribution of the shared self-throttle pop pause to the backstop that armed it.

Three independent subsystems arm the single pop-pause deadline (:attr:`WorkerState.self_throttle_paused`):
the resource/OOM-fault self-maintenance backstop, the host-RAM-pressure governor, and the safety
soft-pause. These tests pin the disentangled observability: each arming site stamps its own
:class:`PopPauseOwner` and a truthful reason, the pop-governor reading reports that owner's reason (not a
single hardcoded string), the action ledger records the arm and the lapse with the owner and its numeric
context, and an overlapping arm never shortens a longer standing deadline (the later deadline wins the
label). Behavior preservation: the effective gate, durations, and re-arm conditions are unchanged; only the
attribution is added.
"""

from __future__ import annotations

import time

from loguru import logger

from horde_worker_regen.process_management.config.worker_state import PopPauseOwner
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.resources.resource_budget import assess_ram_pressure
from horde_worker_regen.process_management.scheduling.governance import (
    HostMemorySnapshot,
    PausePops,
    decide_degrade_response,
)
from tests.process_management.conftest import make_testable_process_manager

_UNSERVABLE_MODEL = "AlbedoBase XL (SDXL)"
_TOTAL_RAM_MB = 32000.0
_CRITICAL_AVAILABLE_MB = 900.0
_NOW = 1_000_000.0


def _pressure_snapshot(
    *,
    available_mb: float = _CRITICAL_AVAILABLE_MB,
    pop_pause_active: bool = False,
    pop_pause_until: float = 0.0,
    now: float = _NOW,
) -> HostMemorySnapshot:
    """A host-memory snapshot under the RAM danger floor, varying only the standing-pause fields."""
    return HostMemorySnapshot(
        verdict=assess_ram_pressure(available_mb, _TOTAL_RAM_MB),
        now=now,
        pop_pause_active=pop_pause_active,
        pop_pause_until=pop_pause_until,
        pop_hold_margin_mb=4096.0,
        per_process_ceiling_mb=None,
        multi_gpu_routing_active=False,
        in_flight_job_count=0,
        loaded_worker_process_count=1,
        planned_worker_process_count=1,
        inference_slots=(),
        cards=(),
        draining_process_ids=frozenset(),
        shed_card_indices=frozenset(),
        restore_headroom_mb=0.0,
        per_context_ram_estimate_mb=4096.0,
        worker_shed_planned_process_count=None,
        worker_shed_process_count=0,
    )


def _ledger_events(manager: object, event_type: LedgerEventType) -> list[object]:
    """Every ledger event of ``event_type`` currently in the shared in-memory ring."""
    return [event for event in manager._action_ledger.recent(limit=100) if event.event_type == event_type]


def _self_throttle_reading(manager: object, now: float) -> object:
    """The pop-governor reading for the shared self-throttle pause this cycle."""
    readings = manager._collect_pop_governor_readings(now)
    matches = [reading for reading in readings if reading.name == "self_throttle_pause"]
    assert len(matches) == 1
    return matches[0]


class TestRamPressurePauseAttribution:
    """The RAM-pressure governor owns its arm: the reading names RAM and the ledger carries the numbers."""

    def test_decision_pause_carries_measured_free_and_floor(self) -> None:
        """``decide_degrade_response`` stamps the measured free and floor MB onto the pause action.

        These travel to the arm site so the ledger record can report the numbers the pause acted on.
        """
        verdict = assess_ram_pressure(_CRITICAL_AVAILABLE_MB, _TOTAL_RAM_MB)
        actions = decide_degrade_response(_pressure_snapshot())
        pauses = [action for action in actions if isinstance(action, PausePops)]
        assert len(pauses) == 1
        assert pauses[0].available_mb == verdict.available_mb
        assert pauses[0].floor_mb == verdict.floor_mb

    def test_execute_stamps_ram_owner_reading_and_ledger(self) -> None:
        """Executing a RAM PausePops arms the shared pause as RAM-owned, and the reading/ledger say so."""
        manager = make_testable_process_manager()
        now = time.time()
        verdict = assess_ram_pressure(_CRITICAL_AVAILABLE_MB, _TOTAL_RAM_MB)
        action = PausePops(
            until_time=now + 30.0,
            pause_seconds=30.0,
            reason=verdict.reason(),
            available_mb=verdict.available_mb,
            floor_mb=verdict.floor_mb,
        )

        manager._inference_scheduler._execute_governance_actions([action])

        assert manager._state.self_throttle_paused is True
        assert manager._state.self_throttle_paused_until == now + 30.0
        assert manager._state.self_throttle_pause_owner is PopPauseOwner.RAM_PRESSURE

        reading = _self_throttle_reading(manager, now)
        assert reading.active is True
        assert reading.reason is not None
        assert "ram" in reading.reason.lower()
        assert "oom" not in reading.reason.lower()
        assert "fault" not in reading.reason.lower()

        armed = _ledger_events(manager, LedgerEventType.POP_PAUSE_ARMED)
        assert len(armed) == 1
        detail = armed[0].detail
        assert detail["owner"] == PopPauseOwner.RAM_PRESSURE.value
        assert detail["available_ram_mb"] == round(verdict.available_mb, 1)
        assert detail["floor_ram_mb"] == round(verdict.floor_mb, 1)
        assert detail["duration_seconds"] == 30.0

    def test_ram_governor_does_not_emit_pause_when_a_longer_deadline_covers(self) -> None:
        """A standing pause that already covers now+30s suppresses a fresh RAM pause (no shortening).

        This is the guard that lets a long fault-throttle deadline survive a RAM-pressure reading: the RAM
        governor only arms when its own deadline would be the later one.
        """
        snapshot = _pressure_snapshot(pop_pause_active=True, pop_pause_until=_NOW + 300.0)
        actions = decide_degrade_response(snapshot)
        assert not any(isinstance(action, PausePops) for action in actions)


class TestFaultThrottleAttribution:
    """The resource/OOM-fault backstop owns its arm distinctly from RAM and safety."""

    def test_fault_throttle_stamps_owner_reading_and_ledger(self) -> None:
        """Crossing the fault threshold arms a fault-owned pause; the reading names faults, not RAM."""
        manager = make_testable_process_manager(
            self_maintenance_fault_threshold=3,
            self_maintenance_window_seconds=600,
            self_maintenance_cooldown_seconds=300,
        )
        for _ in range(3):
            manager._job_tracker._record_resource_fault(_UNSERVABLE_MODEL)

        manager._apply_self_maintenance_throttle()

        assert manager._state.self_throttle_paused is True
        assert manager._state.self_throttle_pause_owner is PopPauseOwner.FAULT_THROTTLE

        reading = _self_throttle_reading(manager, time.time())
        assert reading.reason is not None
        assert "fault" in reading.reason.lower()
        assert "ram" not in reading.reason.lower()

        armed = _ledger_events(manager, LedgerEventType.POP_PAUSE_ARMED)
        assert len(armed) == 1
        detail = armed[0].detail
        assert detail["owner"] == PopPauseOwner.FAULT_THROTTLE.value
        assert detail["recent_faults"] == 3
        assert detail["threshold"] == 3
        assert detail["duration_seconds"] == 300.0


class TestSafetySoftPauseAttribution:
    """The safety soft-pause owns its arm distinctly from RAM and fault-throttle."""

    def test_safety_soft_pause_stamps_owner_reading_and_ledger(self) -> None:
        """Engaging the safety soft-pause arms a safety-owned pause; the reading names safety."""
        manager = make_testable_process_manager()

        manager._recovery_coordinator.engage_safety_soft_pause("verdict never returned")

        assert manager._state.self_throttle_paused is True
        assert manager._state.self_throttle_pause_owner is PopPauseOwner.SAFETY

        reading = _self_throttle_reading(manager, time.time())
        assert reading.reason is not None
        assert "safety" in reading.reason.lower()
        assert "ram" not in reading.reason.lower()

        armed = _ledger_events(manager, LedgerEventType.POP_PAUSE_ARMED)
        assert len(armed) == 1
        assert armed[0].detail["owner"] == PopPauseOwner.SAFETY.value
        assert armed[0].detail["duration_seconds"] == 60.0


class TestOverlappingArmsLaterDeadlineWins:
    """When two backstops overlap, the later deadline holds the pause and owns the reported reason."""

    def test_ram_pressure_does_not_shorten_active_fault_throttle(self) -> None:
        """A RAM-pressure reading under a still-standing longer fault throttle leaves the fault pause intact.

        The fault throttle's 300s deadline covers the RAM governor's now+30s window, so no RAM pause is
        emitted, the effective deadline is unchanged, and the reported owner stays the fault throttle.
        """
        manager = make_testable_process_manager(
            self_maintenance_fault_threshold=1,
            self_maintenance_window_seconds=600,
            self_maintenance_cooldown_seconds=300,
        )
        manager._job_tracker._record_resource_fault(_UNSERVABLE_MODEL)
        manager._apply_self_maintenance_throttle()
        deadline_before = manager._state.self_throttle_paused_until
        assert manager._state.self_throttle_pause_owner is PopPauseOwner.FAULT_THROTTLE

        # A RAM-pressure reading whose would-be deadline (now+30s) is covered by the standing fault throttle.
        snapshot = _pressure_snapshot(
            now=time.time(),
            pop_pause_active=True,
            pop_pause_until=manager._state.self_throttle_paused_until,
        )
        actions = decide_degrade_response(snapshot)
        assert not any(isinstance(action, PausePops) for action in actions)

        # The suppressed pause leaves the fault throttle's deadline and ownership standing.
        assert manager._state.self_throttle_paused_until == deadline_before
        assert manager._state.self_throttle_pause_owner is PopPauseOwner.FAULT_THROTTLE
        reading = _self_throttle_reading(manager, time.time())
        assert reading.reason is not None
        assert "fault" in reading.reason.lower()

    def test_ram_pressure_supersedes_a_nearly_lapsed_fault_throttle(self) -> None:
        """When the RAM deadline is the later one, it takes ownership and the transition is logged."""
        manager = make_testable_process_manager(
            self_maintenance_fault_threshold=1,
            self_maintenance_window_seconds=600,
            self_maintenance_cooldown_seconds=300,
        )
        manager._job_tracker._record_resource_fault(_UNSERVABLE_MODEL)
        manager._apply_self_maintenance_throttle()

        # Fast-forward to the last few seconds of the fault throttle so a fresh RAM pause outlasts it.
        now = time.time()
        manager._state.self_throttle_paused_until = now + 5.0

        verdict = assess_ram_pressure(_CRITICAL_AVAILABLE_MB, _TOTAL_RAM_MB)
        ram_pause = PausePops(
            until_time=now + 30.0,
            pause_seconds=30.0,
            reason=verdict.reason(),
            available_mb=verdict.available_mb,
            floor_mb=verdict.floor_mb,
        )

        lines: list[str] = []
        sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="WARNING")
        try:
            manager._inference_scheduler._execute_governance_actions([ram_pause])
        finally:
            logger.remove(sink_id)

        assert manager._state.self_throttle_paused_until == now + 30.0
        assert manager._state.self_throttle_pause_owner is PopPauseOwner.RAM_PRESSURE
        assert any(PopPauseOwner.FAULT_THROTTLE.value in line for line in lines), (
            "the transition log should name the superseded owner"
        )


class TestLapseAttribution:
    """The unified lapse clears the shared pause and names the owner that had armed it."""

    def test_lapse_clears_and_records_owner(self) -> None:
        """Once the deadline elapses the pause clears, the owner resets, and the lapse is logged/ledgered."""
        manager = make_testable_process_manager()
        now = time.time()
        verdict = assess_ram_pressure(_CRITICAL_AVAILABLE_MB, _TOTAL_RAM_MB)
        manager._inference_scheduler._execute_governance_actions(
            [
                PausePops(
                    until_time=now + 30.0,
                    pause_seconds=30.0,
                    reason=verdict.reason(),
                    available_mb=verdict.available_mb,
                    floor_mb=verdict.floor_mb,
                ),
            ],
        )
        assert manager._state.self_throttle_pause_owner is PopPauseOwner.RAM_PRESSURE

        # Move the deadline into the past so the next maintenance tick lapses the pause.
        manager._state.self_throttle_paused_until = time.time() - 1.0

        lines: list[str] = []
        sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
        try:
            manager._apply_self_maintenance_throttle()
        finally:
            logger.remove(sink_id)

        assert manager._state.self_throttle_paused is False
        assert manager._state.self_throttle_pause_owner is None
        assert manager._state.self_throttle_pause_reason == ""

        lapsed = _ledger_events(manager, LedgerEventType.POP_PAUSE_LAPSED)
        assert len(lapsed) == 1
        assert lapsed[0].detail["owner"] == PopPauseOwner.RAM_PRESSURE.value
        assert any(PopPauseOwner.RAM_PRESSURE.value in line for line in lines), (
            "the resume log should name the owner that had armed the pause"
        )
