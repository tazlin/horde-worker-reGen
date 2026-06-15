"""Tests for the main-loop task supervisor (``HordeWorkerProcessManager._handle_exception``).

A main-loop coroutine that ends unexpectedly, by raising or by returning early while the worker is
running, must trigger a graceful shutdown rather than leave the worker limping with a dead loop.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

from .conftest import make_testable_process_manager


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
