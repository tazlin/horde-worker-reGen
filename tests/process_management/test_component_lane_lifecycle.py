"""Tests for the component (text-encode service) lane's crash recovery in the lifecycle."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager


def _inject_dead_component_lane(process_manager: object, process_id: int) -> object:
    """Put a component lane into the map that has already exited, and return its info."""
    lifecycle = process_manager._process_lifecycle  # type: ignore[attr-defined]
    info = make_mock_process_info(process_id)
    info.process_type = HordeProcessType.COMPONENT
    info.end_intended = False
    info.mp_process = Mock()
    info.mp_process.is_alive.return_value = False
    info.mp_process.exitcode = 1
    lifecycle._process_map[process_id] = info
    return info


def test_crashed_component_lane_is_flagged_for_replacement() -> None:
    """A component lane found dead (not intentionally ended) is flagged for the replacement state machine."""
    process_manager = make_testable_process_manager()
    lifecycle = process_manager._process_lifecycle
    info = _inject_dead_component_lane(process_manager, 99)

    reaped = lifecycle._reap_if_crashed(info)

    assert reaped is True
    assert lifecycle._component_processes_should_be_replaced is True


def test_intentionally_ended_component_lane_is_not_reaped() -> None:
    """A lane the parent asked to end is left alone by crash recovery."""
    process_manager = make_testable_process_manager()
    lifecycle = process_manager._process_lifecycle
    info = _inject_dead_component_lane(process_manager, 99)
    info.end_intended = True

    assert lifecycle._reap_if_crashed(info) is False
    assert lifecycle._component_processes_should_be_replaced is False
