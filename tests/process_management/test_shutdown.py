"""Tests for ShutdownManager."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.shutdown_manager import ShutdownManager
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import (
    make_mock_process_info,
    mark_job_in_progress_async,
    move_job_to_being_safety_checked_async,
    queue_job_for_safety_async,
    queue_job_for_submit_async,
    track_popped_job_async,
)


def _make_shutdown_manager(
    *,
    state: WorkerState | None = None,
    job_tracker: JobTracker | None = None,
    process_map: ProcessMap | None = None,
) -> ShutdownManager:
    """Build a ShutdownManager with mostly-mocked dependencies."""
    if state is None:
        state = WorkerState()
    if job_tracker is None:
        job_tracker = JobTracker()
    if process_map is None:
        process_map = ProcessMap({})

    process_lifecycle = Mock()
    process_lifecycle.recently_recovered = False

    return ShutdownManager(
        state=state,
        job_tracker=job_tracker,
        process_map=process_map,
        process_lifecycle=process_lifecycle,
    )


class TestIsTimeForShutdown:
    """Tests for is_time_for_shutdown."""

    def test_not_shutting_down_returns_false(self) -> None:
        """If we're not shutting down, is_time_for_shutdown should return False regardless of other conditions."""
        shutdown_manager = _make_shutdown_manager()
        assert shutdown_manager.is_time_for_shutdown() is False

    def test_recently_recovered_returns_false(self) -> None:
        """If we've recently recovered from a failure, we should delay shutdown to avoid a shutdown loop."""
        state = WorkerState(shutting_down=True)
        shutdown_manager = _make_shutdown_manager(state=state)
        shutdown_manager._process_lifecycle.recently_recovered = True

        assert shutdown_manager.is_time_for_shutdown() is False

    async def test_jobs_pending_submit_returns_false(self) -> None:
        """Jobs pending submit should prevent actually shutting down."""
        state = WorkerState(shutting_down=True)
        job_tracker = JobTracker()
        await queue_job_for_submit_async(job_tracker, Mock())

        shutdown_manager = _make_shutdown_manager(state=state, job_tracker=job_tracker)
        assert shutdown_manager.is_time_for_shutdown() is False

    async def test_jobs_being_safety_checked_returns_false(self) -> None:
        """Jobs being safety checked should prevent actually shutting down."""
        state = WorkerState(shutting_down=True)
        job_tracker = JobTracker()
        await move_job_to_being_safety_checked_async(job_tracker, Mock())

        shutdown_manager = _make_shutdown_manager(state=state, job_tracker=job_tracker)
        assert shutdown_manager.is_time_for_shutdown() is False

    async def test_jobs_pending_safety_check_returns_false(self) -> None:
        """Jobs pending safety check should prevent actually shutting down."""
        state = WorkerState(shutting_down=True)
        job_tracker = JobTracker()
        await queue_job_for_safety_async(job_tracker, Mock())

        shutdown_manager = _make_shutdown_manager(state=state, job_tracker=job_tracker)
        assert shutdown_manager.is_time_for_shutdown() is False

    async def test_jobs_in_progress_returns_false(self) -> None:
        """Jobs in progress should prevent actually shutting down."""
        state = WorkerState(shutting_down=True)
        job_tracker = JobTracker()
        await mark_job_in_progress_async(job_tracker, Mock())

        shutdown_manager = _make_shutdown_manager(state=state, job_tracker=job_tracker)
        assert shutdown_manager.is_time_for_shutdown() is False

    async def test_jobs_pending_inference_returns_false(self) -> None:
        """Jobs pending inference should prevent actually shutting down."""
        state = WorkerState(shutting_down=True)
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, Mock())

        shutdown_manager = _make_shutdown_manager(state=state, job_tracker=job_tracker)
        assert shutdown_manager.is_time_for_shutdown() is False

    def test_all_processes_ending_returns_true(self) -> None:
        """If all processes are ending, is_time_for_shutdown should return True."""
        state = WorkerState(shutting_down=True)
        process_info = make_mock_process_info(0, state=HordeProcessState.PROCESS_ENDING)
        process_map = ProcessMap({0: process_info})

        shutdown_manager = _make_shutdown_manager(state=state, process_map=process_map)
        assert shutdown_manager.is_time_for_shutdown() is True

    def test_no_processes_returns_true(self) -> None:
        """If there are no processes, is_time_for_shutdown should return True."""
        state = WorkerState(shutting_down=True)
        shutdown_manager = _make_shutdown_manager(state=state)
        assert shutdown_manager.is_time_for_shutdown() is True


class TestShutdown:
    """Tests for shutdown."""

    def test_shutdown_sets_flag(self) -> None:
        """Calling shutdown should set the shutting_down flag and record the time."""
        shutdown_manager = _make_shutdown_manager()
        assert shutdown_manager._state.shutting_down is False

        shutdown_manager.shutdown()
        assert shutdown_manager._state.shutting_down is True
        assert shutdown_manager._state.shutting_down_time > 0

    def test_double_shutdown_no_effect(self) -> None:
        """Calling shutdown twice should not change the shutting_down_time."""
        shutdown_manager = _make_shutdown_manager()
        shutdown_manager.shutdown()
        first_time = shutdown_manager._state.shutting_down_time

        shutdown_manager.shutdown()
        assert shutdown_manager._state.shutting_down_time == first_time


class TestAbort:
    """Tests for abort."""

    def test_abort_purges_jobs_and_shuts_down(self) -> None:
        """Calling abort should set shutting_down to True, and trigger relevant cleanup actions."""
        shutdown_manager = _make_shutdown_manager()
        shutdown_manager._job_tracker._purge_jobs = Mock()
        shutdown_manager._process_lifecycle._hard_kill_processes = Mock()

        shutdown_manager.abort()

        assert shutdown_manager._state.shutting_down is True
        shutdown_manager._job_tracker._purge_jobs.assert_called_once()
        shutdown_manager._process_lifecycle._hard_kill_processes.assert_called_once()


class TestSignalHandler:
    """Tests for signal_handler."""

    def test_first_signal_initiates_shutdown(self) -> None:
        """Receiving a SIGINT should set the shutting_down flag and record the time, and increment caught_sigints."""
        shutdown_manager = _make_shutdown_manager()
        shutdown_manager.signal_handler(2, None)

        assert shutdown_manager._state.shutting_down is True
        assert shutdown_manager._caught_sigints == 1

    def test_second_signal_increments_count(self) -> None:
        """Receiving a second SIGINT should increment the count but not have other side effects."""
        shutdown_manager = _make_shutdown_manager()
        shutdown_manager.signal_handler(2, None)
        shutdown_manager.signal_handler(2, None)

        assert shutdown_manager._caught_sigints == 2
