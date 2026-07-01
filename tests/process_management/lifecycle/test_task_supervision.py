"""Tests for the main-loop task supervisor (``HordeWorkerProcessManager._handle_exception``).

A main-loop coroutine that ends unexpectedly, by raising or by returning early while the worker is
running, must trigger a graceful shutdown rather than leave the worker limping with a dead loop.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

import horde_worker_regen.server_capabilities as server_capabilities_module
import horde_worker_regen.update_check as update_check_module
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import make_testable_process_manager


class _FakeTask:
    """A stand-in for an ``asyncio.Task`` exposing only what ``_handle_exception`` reads."""

    def __init__(self, *, cancelled: bool = False, exception: BaseException | None = None) -> None:
        self._cancelled = cancelled
        self._exception = exception

    def cancelled(self) -> bool:
        return self._cancelled

    def exception(self) -> BaseException | None:
        return self._exception


def _pm_with_spied_backstop() -> HordeWorkerProcessManager:
    """A testable process manager whose force-kill backstop is replaced with a spy (no real thread)."""
    pm = make_testable_process_manager()
    pm._shutdown_manager.start_timed_shutdown = Mock()  # type: ignore[method-assign]
    return pm


def test_task_exception_initiates_graceful_shutdown() -> None:
    """An unhandled exception in a main-loop task initiates shutdown and arms the backstop."""
    pm = _pm_with_spied_backstop()
    pm._handle_exception(_FakeTask(exception=RuntimeError("boom")))  # type: ignore[arg-type]

    assert pm._state.shutting_down is True
    pm._shutdown_manager.start_timed_shutdown.assert_called_once()  # type: ignore[attr-defined]


def test_clean_return_while_running_initiates_shutdown() -> None:
    """A loop that simply returns while the worker is running is also treated as unexpected."""
    pm = _pm_with_spied_backstop()
    pm._handle_exception(_FakeTask(exception=None))  # type: ignore[arg-type]

    assert pm._state.shutting_down is True
    pm._shutdown_manager.start_timed_shutdown.assert_called_once()  # type: ignore[attr-defined]


def test_cancelled_task_is_ignored() -> None:
    """A cancelled task (normal during shutdown/gather-cancel) does not initiate shutdown."""
    pm = _pm_with_spied_backstop()
    pm._handle_exception(_FakeTask(cancelled=True))  # type: ignore[arg-type]

    assert pm._state.shutting_down is False
    pm._shutdown_manager.start_timed_shutdown.assert_not_called()  # type: ignore[attr-defined]


def test_exception_during_shutdown_does_not_retrigger() -> None:
    """Once already shutting down, a task ending does not re-arm the backstop."""
    pm = _pm_with_spied_backstop()
    pm._state.shutting_down = True

    pm._handle_exception(_FakeTask(exception=RuntimeError("boom")))  # type: ignore[arg-type]

    pm._shutdown_manager.start_timed_shutdown.assert_not_called()  # type: ignore[attr-defined]


async def test_disabled_update_check_loop_stays_alive_until_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A disabled update check must keep its main-loop task alive until shutdown, not return early.

    The update-check loop is one of the supervised main-loop tasks, and the supervisor treats any of them
    ending while the worker is live as fatal (see the tests above). So when update checks are disabled the
    loop must idle until shutdown rather than returning, which would otherwise shut the worker down at
    startup whenever checks are off (the test environment, or an operator's opt-out).
    """
    monkeypatch.setattr(update_check_module, "update_check_disabled", lambda: True)
    pm = _pm_with_spied_backstop()
    monkeypatch.setattr(pm, "_UPDATE_CHECK_SHUTDOWN_POLL_SECONDS", 0.01, raising=False)

    task = asyncio.create_task(pm._periodic_update_check_loop())
    await asyncio.sleep(0.05)  # several poll intervals
    assert not task.done(), "the disabled update-check loop must keep running, not return while the worker is live"

    pm._state.shut_down = True  # the loop's only legitimate exit: a shutdown signal
    await asyncio.wait_for(task, timeout=1.0)
    assert task.exception() is None


async def test_server_capabilities_loop_refreshes_off_hot_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capability probe runs on its own supervised loop, not inside a job pop loop.

    Moving the probe here is what keeps a slow or hung swagger fetch from stalling job popping. The loop
    must call the refresh and, like the other main-loop tasks, keep running until shutdown.
    """
    calls = 0

    async def _fake_refresh(*, force: bool = False) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(server_capabilities_module, "refresh_server_capabilities", _fake_refresh)
    pm = _pm_with_spied_backstop()
    monkeypatch.setattr(pm, "_SERVER_CAPABILITIES_POLL_SECONDS", 0.01, raising=False)

    task = asyncio.create_task(pm._periodic_server_capabilities_loop())
    await asyncio.sleep(0.05)  # several poll intervals
    assert not task.done(), "the capability loop must keep running while the worker is live"
    assert calls >= 1, "the loop must drive the server-capability refresh"

    pm._state.shut_down = True
    await asyncio.wait_for(task, timeout=1.0)
    assert task.exception() is None
