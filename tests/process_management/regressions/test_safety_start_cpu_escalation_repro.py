"""Reproduction: a deferred safety GPU start on a card that never drains leaves the worker without safety.

A safety start waits for card headroom (:meth:`ProcessLifecycleManager._defer_gpu_start`) and is retried only
when that headroom appears (:meth:`ProcessLifecycleManager.drain_pending_gpu_starts`). When the pressure on the
card comes from a tenant the reclaim ladder cannot evict, the card never reaches the threshold, so no safety
process ever starts: pops stay held at ``no_safety_process`` and every finished image waits on a check that
cannot run. A soft reset that rebuilt the pools while this was true also reported them recovered.

Contracts:

1. Escalation. A deferred safety start that outlives ``PENDING_GPU_START_NO_PROGRESS_SECONDS`` with no headroom
   progress on its card starts on the CPU through the existing off-GPU placement, owned by runtime safety
   placement, so the placement reconciler can return it to the card once the card shows durable room.
2. The pair. Headroom that returns inside the window starts safety on the GPU with no escalation, and drain
   progress inside the window postpones the escalation.
3. Recovery verdict. A soft-reset episode does not close, and ``pools recovered`` is not logged, while the safety
   start is deferred; it closes once safety is ready.

The scenario drives the per-tick pieces of the main loop directly (readiness observation, the pending-start
drain, the safety replacement state machine, the placement reconciler) so it can move the clock across the
no-progress window without the hung-process timers judging the fake children against real heartbeats.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen.process_management.config.worker_state import PopGate
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    PENDING_GPU_START_NO_PROGRESS_SECONDS,
    PENDING_GPU_START_PROGRESS_EPSILON_MB,
    SAFETY_PROCESS_ID,
    PauseOwner,
    ProcessLifecycleManager,
)
from horde_worker_regen.process_management.lifecycle.recovery_supervisor import RecoveryAction, RecoverySupervisor
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.resources.admission_identity import admission_margin_mb
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.resources.foreign_vram_floor import ManagedVramTenant
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager

_CARD_TOTAL_MB = 10240.0
"""A 10 GB card, the size of the one the managed text backend filled."""

_PRESSURED_FREE_MB = 650.0
"""Free VRAM on a card another tenant holds: far below any GPU start threshold, and it never moves."""

_ROOMY_FREE_MB = 9000.0
"""Free VRAM on the same card once its tenant has gone."""

_CYCLE_SECONDS = 10.0
"""World time one driven control cycle spans."""

_ESCALATION_CYCLE_BOUND = math.ceil(PENDING_GPU_START_NO_PROGRESS_SECONDS / _CYCLE_SECONDS) + 2
"""Cycles within which a stalled deferral must have produced a safety process."""


class _AdvanceableTime:
    """Stand-in for the ``time`` module whose wall and monotonic clocks the test advances together."""

    def __init__(self) -> None:
        self._offset = 0.0

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - forwards the real module's arbitrary members
        """Return the real ``time`` module's attribute of that name."""
        return getattr(time, name)

    def time(self) -> float:
        """Return the wall clock, shifted by however far the test has advanced it."""
        return time.time() + self._offset

    def monotonic(self) -> float:
        """Return the monotonic clock, shifted by however far the test has advanced it."""
        return time.monotonic() + self._offset

    def advance(self, seconds: float) -> None:
        """Move both clocks forward."""
        self._offset += seconds


@pytest.fixture
def fake_time(monkeypatch: pytest.MonkeyPatch) -> _AdvanceableTime:
    """Advanceable clocks for the lifecycle module, where deferral ages and readiness stamps are taken."""
    advanceable = _AdvanceableTime()
    monkeypatch.setattr("horde_worker_regen.process_management.lifecycle.process_lifecycle.time", advanceable)
    return advanceable


@pytest.fixture
def captured_messages() -> Iterator[list[str]]:
    """Every INFO-and-above parent log message emitted during the test."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    yield messages
    logger.remove(sink_id)


def _pressured_manager(fake_time: _AdvanceableTime) -> HordeWorkerProcessManager:
    """Build a single-card worker with safety permitted on-GPU and a card held at the pressure tier."""
    pm = make_testable_process_manager(safety_on_gpu=True, device_free_mb=_PRESSURED_FREE_MB)
    pm._last_device_total_mb_by_device[0] = _CARD_TOTAL_MB
    pm._governor_states_by_device[0] = GovernorState.PRESSURE
    pm._inference_scheduler._clock = fake_time.time
    pm._process_lifecycle._ctx.Pipe.return_value = (Mock(), Mock())  # pyrefly: ignore[missing-attribute]
    pm._process_lifecycle._new_process.return_value.pid = 4242  # pyrefly: ignore[missing-attribute]
    return pm


def _set_card(pm: HordeWorkerProcessManager, *, free_mb: float, governor_state: GovernorState) -> None:
    """Set the parent's device reading for card 0."""
    pm._last_device_free_mb_by_device[0] = free_mb
    pm._governor_states_by_device[0] = governor_state


def _defer_safety_through_a_supervised_rebuild(pm: HordeWorkerProcessManager) -> ProcessLifecycleManager:
    """Rebuild the safety pool the way a soft reset does, and leave its start deferred on the pressured card."""
    lifecycle = pm._process_lifecycle
    lifecycle.rebuild_safety_pool(reason="soft reset #1")
    lifecycle._replace_all_safety_process()
    assert lifecycle.has_pending_safety_starts() is True
    assert pm._process_map.num_safety_processes() == 0
    return lifecycle


def _control_cycle(pm: HordeWorkerProcessManager, fake_time: _AdvanceableTime) -> None:
    """Advance the clock one cycle and run the per-tick safety pieces of the control loop."""
    fake_time.advance(_CYCLE_SECONDS)
    lifecycle = pm._process_lifecycle
    lifecycle._observe_safety_pool_readiness()
    lifecycle.drain_pending_gpu_starts()
    lifecycle._replace_all_safety_process()
    pm._inference_scheduler._reconcile_runtime_safety_placement()


def _last_safety_start_was_cpu_only(lifecycle: ProcessLifecycleManager) -> bool:
    """Return the ``cpu_only`` argument the most recent child spawn was given."""
    spawn = lifecycle._new_process.call_args  # pyrefly: ignore[missing-attribute] - Mock stand-in for ctx.Process
    return bool(spawn.kwargs["args"][5])


def _mark_safety_ready(pm: HordeWorkerProcessManager) -> None:
    """Let the spawned safety child report readiness."""
    safety_process = pm._process_map.get_safety_process()
    assert safety_process is not None
    safety_process.last_process_state = HordeProcessState.WAITING_FOR_JOB


def _escalation_events(lifecycle: ProcessLifecycleManager) -> list[Any]:
    """Return the ledger's CPU escalation records."""
    return [
        event
        for event in lifecycle.action_ledger.recent(limit=500)
        if event.event_type is LedgerEventType.SAFETY_START_ESCALATED_TO_CPU
    ]


def _escalate_to_cpu_safety(pm: HordeWorkerProcessManager, fake_time: _AdvanceableTime) -> int:
    """Drive cycles on the pressured card until a safety process exists; return how many it took."""
    for cycle in range(1, _ESCALATION_CYCLE_BOUND + 1):
        _control_cycle(pm, fake_time)
        if pm._process_map.num_safety_processes() > 0:
            return cycle
    raise AssertionError(
        f"no safety process started within {_ESCALATION_CYCLE_BOUND} cycles on a card that never drains"
    )


class TestStalledSafetyStartEscalatesToCpu:
    """A deferred safety start on a card that never drains ends in a working CPU safety process."""

    def test_a_card_that_never_drains_gets_a_cpu_safety_process_and_pops_resume(
        self,
        fake_time: _AdvanceableTime,
        captured_messages: list[str],
    ) -> None:
        """The incident: the start is deferred, the card holds at pressure, and safety comes back on the CPU."""
        pm = _pressured_manager(fake_time)
        lifecycle = _defer_safety_through_a_supervised_rebuild(pm)
        coordinator = pm._recovery_coordinator
        pm._state.last_pop_gate = str(PopGate.NO_SAFETY_PROCESS)
        assert coordinator.safety_start_pending() is True

        cycles = _escalate_to_cpu_safety(pm, fake_time)

        assert cycles * _CYCLE_SECONDS >= PENDING_GPU_START_NO_PROGRESS_SECONDS
        assert _last_safety_start_was_cpu_only(lifecycle) is True
        assert lifecycle.has_pending_safety_starts() is False
        assert lifecycle.is_safety_gpu_paused is True
        assert lifecycle.safety_pause_owner is PauseOwner.RUNTIME_SAFETY_PLACEMENT
        assert lifecycle.safety_gpu_card_index() is None
        escalations = _escalation_events(lifecycle)
        assert len(escalations) == 1
        assert escalations[0].detail["device_index"] == 0
        warnings = [message for message in captured_messages if "Starting the safety process on the CPU" in message]
        assert len(warnings) == 1
        assert "device 0" in warnings[0]
        assert f"{_PRESSURED_FREE_MB:.0f}MB free" in warnings[0]

        _mark_safety_ready(pm)
        _control_cycle(pm, fake_time)

        # The pop gate's own predicate: a safety process can take a check, so intake is no longer held on it.
        assert pm._process_map.get_first_available_safety_process() is not None
        assert coordinator.is_safety_pool_ready() is True
        assert coordinator.safety_start_pending() is False
        assert lifecycle.safety_placement_transition_pending is False

        # The card still holds at pressure, so placement keeps safety on the CPU instead of handing it back.
        for _ in range(_ESCALATION_CYCLE_BOUND):
            _control_cycle(pm, fake_time)
        assert lifecycle.is_safety_gpu_paused is True
        assert lifecycle.safety_gpu_restore_count == 0
        assert pm._process_map.get_first_available_safety_process() is not None
        assert len(_escalation_events(lifecycle)) == 1

    def test_an_escalated_safety_process_returns_to_the_card_once_its_room_is_durable(
        self,
        fake_time: _AdvanceableTime,
    ) -> None:
        """The escalation is a placement pause, so the placement reconciler's restore brings safety back."""
        pm = _pressured_manager(fake_time)
        lifecycle = _defer_safety_through_a_supervised_rebuild(pm)
        _escalate_to_cpu_safety(pm, fake_time)
        _mark_safety_ready(pm)
        _control_cycle(pm, fake_time)
        assert lifecycle.is_safety_gpu_paused is True

        # The tenant leaves: a GPU child on the card reports the room, and the governor reads healthy.
        reporter = make_mock_process_info(1, model_name=None, process_type=HordeProcessType.INFERENCE)
        reporter.total_vram_mb = int(_CARD_TOTAL_MB)
        reporter.vram_usage_mb = int(_CARD_TOTAL_MB - _ROOMY_FREE_MB)
        pm._process_map[1] = reporter
        _set_card(pm, free_mb=_ROOMY_FREE_MB, governor_state=GovernorState.HEALTHY)

        restore_dwell = pm._inference_scheduler._safety_placement_restore_dwell_seconds()
        _control_cycle(pm, fake_time)
        assert lifecycle.is_safety_gpu_paused is True, "restore must wait for the headroom dwell"

        for _ in range(math.ceil(restore_dwell / _CYCLE_SECONDS) + 2):
            _control_cycle(pm, fake_time)
            if not lifecycle.is_safety_gpu_paused:
                break
        assert lifecycle.is_safety_gpu_paused is False
        assert lifecycle.safety_gpu_restore_count == 1


class TestHeadroomInsideTheWindowNeedsNoEscalation:
    """The pair: a card that drains, or is draining, keeps the ordinary GPU start."""

    def test_headroom_returning_inside_the_window_starts_safety_on_the_gpu(self, fake_time: _AdvanceableTime) -> None:
        """Room that appears before the window elapses starts safety on its card with no escalation."""
        pm = _pressured_manager(fake_time)
        lifecycle = _defer_safety_through_a_supervised_rebuild(pm)

        for _ in range(int(PENDING_GPU_START_NO_PROGRESS_SECONDS / _CYCLE_SECONDS / 2)):
            _control_cycle(pm, fake_time)
        assert pm._process_map.num_safety_processes() == 0

        _set_card(pm, free_mb=_ROOMY_FREE_MB, governor_state=GovernorState.HEALTHY)
        _control_cycle(pm, fake_time)

        assert pm._process_map.num_safety_processes() == 1
        assert _last_safety_start_was_cpu_only(lifecycle) is False
        assert lifecycle.is_safety_gpu_paused is False
        assert lifecycle.safety_gpu_pause_count == 0
        assert _escalation_events(lifecycle) == []

    def test_drain_progress_inside_the_window_postpones_the_escalation(self, fake_time: _AdvanceableTime) -> None:
        """A card whose free reading is climbing is still draining, so the start keeps waiting for the GPU."""
        pm = _pressured_manager(fake_time)
        lifecycle = _defer_safety_through_a_supervised_rebuild(pm)

        half_window_cycles = int(PENDING_GPU_START_NO_PROGRESS_SECONDS / _CYCLE_SECONDS / 2)
        for _ in range(half_window_cycles):
            _control_cycle(pm, fake_time)
        _set_card(
            pm,
            free_mb=_PRESSURED_FREE_MB + PENDING_GPU_START_PROGRESS_EPSILON_MB * 2,
            governor_state=GovernorState.PRESSURE,
        )
        for _ in range(half_window_cycles + 2):
            _control_cycle(pm, fake_time)

        assert pm._process_map.num_safety_processes() == 0
        assert lifecycle.has_pending_safety_starts() is True
        assert _escalation_events(lifecycle) == []

        cycles = _escalate_to_cpu_safety(pm, fake_time)
        assert cycles > 1
        assert _last_safety_start_was_cpu_only(lifecycle) is True


class _FakeClock:
    """A monotonic stand-in the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward."""
        self.now += seconds


class TestRecoveryVerdictWaitsOnSafety:
    """A soft-reset episode is not declared recovered while its safety process has not come back."""

    def test_the_episode_close_holds_while_a_safety_start_is_pending(self) -> None:
        """The supervisor's close rule: streak and progress are not enough without safety."""
        clock = _FakeClock()
        supervisor = RecoverySupervisor(wedge_grace_seconds=0.0, clean_streak_seconds=5.0, clock=clock)
        assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.SOFT_RESET

        supervisor.evaluate(is_wedged=False, pool_ready=True, made_progress=True, safety_start_pending=True)
        clock.advance(60.0)
        supervisor.evaluate(is_wedged=False, pool_ready=True, made_progress=True, safety_start_pending=True)
        assert supervisor.is_in_episode is True

        supervisor.evaluate(is_wedged=False, pool_ready=True, made_progress=True, safety_start_pending=False)
        assert supervisor.is_in_episode is False

    def test_pools_recovered_is_logged_only_once_safety_is_ready(
        self,
        fake_time: _AdvanceableTime,
        captured_messages: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A soft reset that re-defers safety is not reported recovered until the safety process is ready."""
        pm = _pressured_manager(fake_time)
        lifecycle = pm._process_lifecycle
        coordinator = pm._recovery_coordinator
        clock = _FakeClock()
        coordinator.recovery_supervisor = RecoverySupervisor(
            wedge_grace_seconds=0.0,
            clean_streak_seconds=5.0,
            clock=clock,
        )
        wedged = True
        monkeypatch.setattr(coordinator, "assess_wedge", lambda: wedged)
        monkeypatch.setattr(coordinator, "made_progress_since_episode", lambda: True)
        monkeypatch.setattr(coordinator, "constructive_remedy_available", lambda: False)

        coordinator.run_recovery_supervisor()
        lifecycle._replace_all_safety_process()
        assert coordinator.limp_by_active is True
        assert lifecycle.has_pending_safety_starts() is True
        # The popper stamps this gate on its next cycle, and nothing re-stamps it once safety is back unless a
        # pop cycle runs, so the verdict has to read past a stale stamp.
        pm._state.last_pop_gate = str(PopGate.NO_SAFETY_PROCESS)

        wedged = False
        for _ in range(3):
            clock.advance(10.0)
            coordinator.run_recovery_supervisor()
        assert coordinator.recovery_supervisor.is_in_episode is True
        assert coordinator.limp_by_active is True
        assert not any("pools recovered" in message for message in captured_messages)

        _escalate_to_cpu_safety(pm, fake_time)
        coordinator.run_recovery_supervisor()
        assert not any("pools recovered" in message for message in captured_messages), (
            "a safety process that is still booting is not a recovered pool"
        )

        _mark_safety_ready(pm)
        lifecycle._observe_safety_pool_readiness()
        coordinator.run_recovery_supervisor()
        assert coordinator.recovery_supervisor.is_in_episode is False
        assert coordinator.limp_by_active is False
        assert sum("pools recovered" in message for message in captured_messages) == 1
        assert pm._process_map.get_safety_process() is not None
        assert pm._process_map.get_safety_process().process_id == SAFETY_PROCESS_ID  # pyrefly: ignore[missing-attribute]


_STRUCTURAL_ROOM_MB = 677.0
"""The room a card can give back beside a tenant that holds nearly all of it."""


def _charge_managed_tenant(pm: HordeWorkerProcessManager, *, room_mb: float) -> float:
    """Make the managed text backend a tenant of card 0 that leaves ``room_mb`` achievable; return its footprint."""
    footprint_mb = _CARD_TOTAL_MB - admission_margin_mb(_CARD_TOTAL_MB) - room_mb
    tenant = ManagedVramTenant(device_index=0, footprint_mb=footprint_mb)
    pm._inference_scheduler.set_managed_tenant_provider(lambda: tenant)
    return footprint_mb


class TestStructuralShortfallEscalatesAtOnce:
    """A safety start the card can never make room for escalates without waiting out the no-progress window."""

    def test_a_card_whose_standing_floor_leaves_too_little_room_escalates_within_two_cycles(
        self,
        fake_time: _AdvanceableTime,
        captured_messages: list[str],
    ) -> None:
        """The incident shape: the managed backend leaves 677 MB against a larger start requirement."""
        pm = _pressured_manager(fake_time)
        _charge_managed_tenant(pm, room_mb=_STRUCTURAL_ROOM_MB)
        lifecycle = _defer_safety_through_a_supervised_rebuild(pm)
        required_mb = lifecycle._gpu_start_required_free_mb(0)
        assert required_mb > _STRUCTURAL_ROOM_MB

        cycles = 0
        while pm._process_map.num_safety_processes() == 0:
            assert cycles < 2, "a structural shortfall must escalate within two cycles"
            _control_cycle(pm, fake_time)
            cycles += 1

        assert _last_safety_start_was_cpu_only(lifecycle) is True
        assert lifecycle.has_pending_safety_starts() is False
        assert lifecycle.safety_pause_owner is PauseOwner.RUNTIME_SAFETY_PLACEMENT
        escalations = _escalation_events(lifecycle)
        assert len(escalations) == 1
        assert "structural shortfall" in escalations[0].reason
        assert escalations[0].detail["room_mb"] == pytest.approx(_STRUCTURAL_ROOM_MB, abs=0.1)
        warnings = [message for message in captured_messages if "structural shortfall" in message]
        assert any(
            f"needs {required_mb:.0f}MB" in message and f"at most {_STRUCTURAL_ROOM_MB:.0f}MB" in message
            for message in warnings
        )

    def test_a_drainable_shortfall_still_waits_for_the_window(self, fake_time: _AdvanceableTime) -> None:
        """A tenant that leaves room for the start keeps the ordinary wait: headroom may still come back."""
        pm = _pressured_manager(fake_time)
        _charge_managed_tenant(pm, room_mb=4000.0)
        lifecycle = _defer_safety_through_a_supervised_rebuild(pm)
        assert lifecycle._gpu_start_required_free_mb(0) < 4000.0

        for _ in range(5):
            _control_cycle(pm, fake_time)

        assert lifecycle.has_pending_safety_starts() is True
        assert pm._process_map.num_safety_processes() == 0
        assert _escalation_events(lifecycle) == []
