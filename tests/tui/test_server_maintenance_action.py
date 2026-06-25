"""The TUI's server-side maintenance toggle action chooses the right on/off request from the snapshot."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.tui.app import HordeWorkerTUI


def _stub_app(*, worker_details_maintenance: bool, intended: bool | None = None) -> Mock:
    """A stub standing in for the app instance, carrying just what the action reads."""
    stub = Mock()
    stub._supervisor.latest_snapshot = Mock(worker_details_maintenance=worker_details_maintenance)
    stub._supervisor.request_set_server_maintenance.return_value = True
    # Must be set explicitly; a Mock auto-attribute is truthy and would short-circuit the intent logic.
    stub._intended_server_maintenance = intended
    return stub


def test_toggle_enables_server_maintenance_when_not_in_maintenance() -> None:
    """When the horde does not have the worker in maintenance, the action requests turning it ON."""
    stub = _stub_app(worker_details_maintenance=False)
    HordeWorkerTUI.action_toggle_server_maintenance(stub)
    stub._supervisor.request_set_server_maintenance.assert_called_once_with(True)


def test_toggle_disables_server_maintenance_when_in_maintenance() -> None:
    """When the horde already has the worker in maintenance, the action requests turning it OFF."""
    stub = _stub_app(worker_details_maintenance=True)
    HordeWorkerTUI.action_toggle_server_maintenance(stub)
    stub._supervisor.request_set_server_maintenance.assert_called_once_with(False)


def test_rapid_double_press_sends_opposing_commands() -> None:
    """A second press before the advisory poll refreshes reverses the first, not duplicates it.

    Without optimistic intent tracking both presses read the same stale snapshot value and dispatch
    the same command twice (e.g. enable + enable). With it the second press should send the opposite.
    """
    stub = _stub_app(worker_details_maintenance=False)
    HordeWorkerTUI.action_toggle_server_maintenance(stub)   # first press: enable
    HordeWorkerTUI.action_toggle_server_maintenance(stub)   # second press: must disable, not enable again

    calls = stub._supervisor.request_set_server_maintenance.call_args_list
    assert len(calls) == 2
    assert calls[0].args[0] is True   # first press enabled maintenance
    assert calls[1].args[0] is False  # second press reversed it


def test_after_poll_confirms_state_next_press_reads_from_snapshot() -> None:
    """Once the advisory poll catches up, the intent is cleared and the snapshot is canonical again.

    Simulates: press "m" to enable (intent=True), advisory poll confirms maintenance=True (intent
    clears), then a third press should read the now-confirmed snapshot and send False.
    """
    stub = _stub_app(worker_details_maintenance=False)

    # First press: enable; intent is now True.
    HordeWorkerTUI.action_toggle_server_maintenance(stub)
    assert stub._intended_server_maintenance is True

    # Simulate the advisory poll confirming the state: update snapshot and clear intent the same way
    # _tick does (snapshot.worker_details_maintenance == _intended_server_maintenance).
    stub._supervisor.latest_snapshot = Mock(worker_details_maintenance=True)
    stub._intended_server_maintenance = None  # mirrors what _tick does on confirmation

    # Third press: intent is clear, reads snapshot (maintenance=True), should send False.
    HordeWorkerTUI.action_toggle_server_maintenance(stub)
    calls = stub._supervisor.request_set_server_maintenance.call_args_list
    assert calls[-1].args[0] is False
