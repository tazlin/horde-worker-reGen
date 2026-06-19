"""The TUI's server-side maintenance toggle action chooses the right on/off request from the snapshot."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.tui.app import HordeWorkerTUI


def _stub_app(*, worker_details_maintenance: bool) -> Mock:
    """A stub standing in for the app instance, carrying just what the action reads."""
    stub = Mock()
    stub._supervisor.latest_snapshot = Mock(worker_details_maintenance=worker_details_maintenance)
    stub._supervisor.request_set_server_maintenance.return_value = True
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
