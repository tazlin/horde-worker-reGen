"""Server-side (horde) maintenance toggle: API call, supervisor command, and the F2-resume gating.

The worker can put itself into, or out of, *server-side* maintenance on the horde (so the horde stops or
resumes sending it jobs), distinct from the local pop-pause. This is exposed as a dedicated supervisor
command and a TUI key. An operator resume (F2) additionally clears server maintenance only when the
``remove_maintenance_on_init`` config opts into it.
"""

from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import process_manager as process_manager_module
from horde_worker_regen.process_management.supervisor_channel import SupervisorCommand, SupervisorControlMessage

from .conftest import make_testable_process_manager


class TestSetMaintenanceApiCall:
    """``set_maintenance`` issues a ModifyWorkerRequest with the requested flag."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_builds_modify_request_with_flag(self, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """``set_maintenance(enabled)`` looks up the worker and modifies it with ``maintenance=enabled``."""
        captured: dict[str, object] = {}

        class FakeClient:
            def worker_details_by_name(self, worker_name: str) -> Mock:
                details = Mock()
                details.id_ = "worker-1"
                return details

            def worker_modify(self, request: object) -> None:
                captured["maintenance"] = request.maintenance  # type: ignore[attr-defined]

        monkeypatch.setattr(process_manager_module, "AIHordeAPISimpleClient", lambda: FakeClient())
        manager = make_testable_process_manager()

        manager.set_maintenance(enabled)

        assert captured["maintenance"] is enabled

    def test_remove_maintenance_clears_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``remove_maintenance`` is the ``set_maintenance(False)`` convenience wrapper."""
        manager = make_testable_process_manager()
        recorder = Mock()
        monkeypatch.setattr(manager, "set_maintenance", recorder)
        manager.remove_maintenance()
        recorder.assert_called_once_with(False)


class TestSupervisorMaintenanceCommand:
    """The SET_SERVER_MAINTENANCE supervisor command dispatches the API call off the control loop."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_command_dispatches_set_maintenance(self, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """The command runs ``_set_server_maintenance_safe(enabled)`` on a background thread."""
        manager = make_testable_process_manager()
        recorded: dict[str, object] = {}
        done = threading.Event()

        def fake_safe(value: bool) -> None:
            recorded["enabled"] = value
            done.set()

        monkeypatch.setattr(manager, "_set_server_maintenance_safe", fake_safe)
        manager._apply_supervisor_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_SERVER_MAINTENANCE,
                server_maintenance_enabled=enabled,
            ),
        )

        assert done.wait(timeout=2.0), "off-loop maintenance call was not made"
        assert recorded["enabled"] is enabled


class TestResumeClearsMaintenanceOnlyWhenConfigured:
    """F2-resume clears *server* maintenance only when ``remove_maintenance_on_init`` is set (per config)."""

    def test_resume_clears_when_flag_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the flag on, a resume also clears horde-side maintenance (off-loop)."""
        manager = make_testable_process_manager(remove_maintenance_on_init=True)
        manager._state.supervisor_paused = True
        recorded: list[bool] = []
        done = threading.Event()
        monkeypatch.setattr(
            manager,
            "_set_server_maintenance_safe",
            lambda value: (recorded.append(value), done.set()),
        )

        manager._apply_supervisor_command(SupervisorControlMessage(command=SupervisorCommand.RESUME))

        assert manager._state.supervisor_paused is False
        assert done.wait(timeout=2.0)
        assert recorded == [False]

    def test_resume_does_not_clear_when_flag_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the flag off (the default), a resume only lifts the local pause; server state is untouched."""
        manager = make_testable_process_manager(remove_maintenance_on_init=False)
        manager._state.supervisor_paused = True
        recorded: list[bool] = []
        monkeypatch.setattr(manager, "_set_server_maintenance_safe", lambda value: recorded.append(value))

        manager._apply_supervisor_command(SupervisorControlMessage(command=SupervisorCommand.RESUME))

        assert manager._state.supervisor_paused is False
        assert recorded == []
