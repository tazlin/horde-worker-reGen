"""The TUI's local pause/resume action uses supervisor_paused, not the aggregate maintenance_mode.

Keeping the two distinct prevents a permanent one-way latch: when the horde forces maintenance via a
pop response (last_pop_maintenance_mode=True), the aggregate maintenance_mode is True but RESUME only
clears supervisor_paused. Reading the aggregate here would make every local pause press send RESUME forever,
since last_pop_maintenance_mode is only cleared by a successful pop - not by a RESUME command.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.tui.app import HordeWorkerTUI


def _stub_app(*, supervisor_paused: bool, maintenance_mode: bool | None = None) -> Mock:
    """Stub that exposes the narrow supervisor_paused field alongside the aggregate."""
    stub = Mock()
    stub._supervisor.latest_snapshot = Mock(
        supervisor_paused=supervisor_paused,
        maintenance_mode=maintenance_mode if maintenance_mode is not None else supervisor_paused,
    )
    stub._supervisor.request_pause.return_value = True
    stub._supervisor.request_resume.return_value = True
    return stub


def test_f2_sends_pause_when_not_locally_paused_even_with_server_maintenance_active() -> None:
    """Local pause pauses locally even when aggregate maintenance_mode is True due to server-side state.

    The horde may force maintenance while supervisor_paused is False: the aggregate is True but local pause
    should still send PAUSE (the operator wants to add a local hold), not RESUME (which would clear
    nothing and re-send on every press).
    """
    stub = _stub_app(supervisor_paused=False, maintenance_mode=True)
    HordeWorkerTUI.action_toggle_pause(stub)
    stub._supervisor.request_pause.assert_called_once()
    stub._supervisor.request_resume.assert_not_called()


def test_f2_sends_resume_when_locally_paused() -> None:
    """Local pause resumes when the operator has locally paused the worker."""
    stub = _stub_app(supervisor_paused=True)
    HordeWorkerTUI.action_toggle_pause(stub)
    stub._supervisor.request_resume.assert_called_once()
    stub._supervisor.request_pause.assert_not_called()


def test_f2_sends_pause_when_snapshot_is_none() -> None:
    """Local pause falls through to PAUSE when there is no snapshot (worker not yet started)."""
    stub = Mock()
    stub._supervisor.latest_snapshot = None
    stub._supervisor.request_pause.return_value = True
    HordeWorkerTUI.action_toggle_pause(stub)
    stub._supervisor.request_pause.assert_called_once()
    stub._supervisor.request_resume.assert_not_called()
