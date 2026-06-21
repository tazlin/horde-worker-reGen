"""Server-side (horde) maintenance toggle: API call, supervisor command, and the F2-resume gating.

The worker can put itself into, or out of, *server-side* maintenance on the horde (so the horde stops or
resumes sending it jobs), distinct from the local pop-pause. This is exposed as a dedicated supervisor
command and a TUI key. An operator resume (F2) additionally clears server maintenance only when the
``remove_maintenance_on_init`` config opts into it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import process_manager as process_manager_module
from horde_worker_regen.process_management.supervisor_channel import SupervisorCommand, SupervisorControlMessage

from .conftest import make_testable_process_manager


class _InlineThread:
    """A ``threading.Thread`` stand-in that runs its target inline on ``start()``.

    The supervisor handler dispatches the (blocking) horde maintenance call off the control loop on a
    daemon thread; tests patch this in so that dispatch is synchronous and deterministic, with no real
    background threads to race the assertions.
    """

    def __init__(self, *, target: Callable[..., Any], args: tuple[Any, ...] = (), **_kwargs: Any) -> None:  # noqa: ANN401
        """Capture the target and positional args, ignoring thread-only kwargs (name/daemon)."""
        self._target = target
        self._args = args

    def start(self) -> None:
        """Run the captured target synchronously."""
        self._target(*self._args)


def _run_off_loop_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the supervisor handler's off-loop thread dispatch run inline for deterministic assertions."""
    monkeypatch.setattr(process_manager_module.threading, "Thread", _InlineThread)


class TestSetMaintenanceApiCall:
    """``set_maintenance`` issues a ModifyWorkerRequest with the requested flag."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_builds_modify_request_with_flag(self, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """``set_maintenance(enabled)`` looks up the worker and modifies it with ``maintenance=enabled``."""
        captured: dict[str, object] = {}

        class FakeClient:
            def workers_all_details(self, worker_name: str | None = None, *, api_key: str | None = None) -> list[Mock]:
                details = Mock()
                details.id_ = "worker-1"
                details.name = worker_name
                return [details]

            def worker_modify(self, request: object) -> None:
                captured["maintenance"] = request.maintenance  # type: ignore[attr-defined]

        monkeypatch.setattr(process_manager_module, "AIHordeAPISimpleClient", lambda: FakeClient())
        manager = make_testable_process_manager()

        manager.set_maintenance(enabled)

        assert captured["maintenance"] is enabled

    @pytest.mark.parametrize("enabled", [True, False])
    def test_unregistered_worker_is_a_quiet_no_op(self, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """A not-yet-registered name (empty list result) returns without modifying or raising.

        The horde registers a worker implicitly on its first pop, so a brand-new name is unknown until
        then; that must be treated as the normal first-run case, not an error.
        """
        modified = False

        class FakeClient:
            def workers_all_details(self, worker_name: str | None = None, *, api_key: str | None = None) -> list[Mock]:
                return []

            def worker_modify(self, request: object) -> None:
                nonlocal modified
                modified = True

        monkeypatch.setattr(process_manager_module, "AIHordeAPISimpleClient", lambda: FakeClient())
        manager = make_testable_process_manager()

        manager.set_maintenance(enabled)

        assert modified is False

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
        _run_off_loop_inline(monkeypatch)
        recorded: list[bool] = []
        monkeypatch.setattr(manager, "_set_server_maintenance_safe", lambda value: recorded.append(value))

        manager._apply_supervisor_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_SERVER_MAINTENANCE,
                server_maintenance_enabled=enabled,
            ),
        )

        assert recorded == [enabled]


class TestResumeClearsMaintenanceOnlyWhenConfigured:
    """F2-resume clears *server* maintenance only when ``remove_maintenance_on_init`` is set (per config)."""

    def test_resume_clears_when_flag_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the flag on, a resume also clears horde-side maintenance (off-loop)."""
        manager = make_testable_process_manager(remove_maintenance_on_init=True)
        manager._state.supervisor_paused = True
        _run_off_loop_inline(monkeypatch)
        recorded: list[bool] = []
        monkeypatch.setattr(manager, "_set_server_maintenance_safe", lambda value: recorded.append(value))

        manager._apply_supervisor_command(SupervisorControlMessage(command=SupervisorCommand.RESUME))

        assert manager._state.supervisor_paused is False
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
