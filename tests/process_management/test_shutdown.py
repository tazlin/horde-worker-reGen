"""Tests for ShutdownManager."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.shutdown_manager import ShutdownManager
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import make_mock_process_info


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
        sm = _make_shutdown_manager()
        assert sm.is_time_for_shutdown() is False

    def test_recently_recovered_returns_false(self) -> None:
        state = WorkerState(shutting_down=True)
        sm = _make_shutdown_manager(state=state)
        sm._process_lifecycle.recently_recovered = True

        assert sm.is_time_for_shutdown() is False

    def test_jobs_pending_submit_returns_false(self) -> None:
        state = WorkerState(shutting_down=True)
        jt = JobTracker()
        jt.jobs_pending_submit.append(Mock())

        sm = _make_shutdown_manager(state=state, job_tracker=jt)
        assert sm.is_time_for_shutdown() is False

    def test_jobs_being_safety_checked_returns_false(self) -> None:
        state = WorkerState(shutting_down=True)
        jt = JobTracker()
        jt.jobs_being_safety_checked.append(Mock())

        sm = _make_shutdown_manager(state=state, job_tracker=jt)
        assert sm.is_time_for_shutdown() is False

    def test_jobs_pending_safety_check_returns_false(self) -> None:
        state = WorkerState(shutting_down=True)
        jt = JobTracker()
        jt.jobs_pending_safety_check.append(Mock())

        sm = _make_shutdown_manager(state=state, job_tracker=jt)
        assert sm.is_time_for_shutdown() is False

    def test_jobs_in_progress_returns_false(self) -> None:
        state = WorkerState(shutting_down=True)
        jt = JobTracker()
        jt.jobs_in_progress.append(Mock())

        sm = _make_shutdown_manager(state=state, job_tracker=jt)
        assert sm.is_time_for_shutdown() is False

    def test_jobs_pending_inference_returns_false(self) -> None:
        state = WorkerState(shutting_down=True)
        jt = JobTracker()
        jt.jobs_pending_inference.append(Mock())

        sm = _make_shutdown_manager(state=state, job_tracker=jt)
        assert sm.is_time_for_shutdown() is False

    def test_all_processes_ending_returns_true(self) -> None:
        state = WorkerState(shutting_down=True)
        proc = make_mock_process_info(0, state=HordeProcessState.PROCESS_ENDING)
        process_map = ProcessMap({0: proc})

        sm = _make_shutdown_manager(state=state, process_map=process_map)
        assert sm.is_time_for_shutdown() is True

    def test_no_processes_returns_true(self) -> None:
        state = WorkerState(shutting_down=True)
        sm = _make_shutdown_manager(state=state)
        assert sm.is_time_for_shutdown() is True


class TestShutdown:
    """Tests for shutdown."""

    def test_shutdown_sets_flag(self) -> None:
        sm = _make_shutdown_manager()
        assert sm._state.shutting_down is False

        sm.shutdown()
        assert sm._state.shutting_down is True
        assert sm._state.shutting_down_time > 0

    def test_double_shutdown_no_effect(self) -> None:
        sm = _make_shutdown_manager()
        sm.shutdown()
        first_time = sm._state.shutting_down_time

        sm.shutdown()
        assert sm._state.shutting_down_time == first_time


class TestAbort:
    """Tests for abort."""

    def test_abort_purges_jobs_and_shuts_down(self) -> None:
        sm = _make_shutdown_manager()
        sm._job_tracker._purge_jobs = Mock()
        sm._process_lifecycle._hard_kill_processes = Mock()

        sm.abort()

        assert sm._state.shutting_down is True
        sm._job_tracker._purge_jobs.assert_called_once()
        sm._process_lifecycle._hard_kill_processes.assert_called_once()


class TestSignalHandler:
    """Tests for signal_handler."""

    def test_first_signal_initiates_shutdown(self) -> None:
        sm = _make_shutdown_manager()
        sm.signal_handler(2, None)

        assert sm._state.shutting_down is True
        assert sm._caught_sigints == 1

    def test_second_signal_increments_count(self) -> None:
        sm = _make_shutdown_manager()
        sm.signal_handler(2, None)
        sm.signal_handler(2, None)

        assert sm._caught_sigints == 2
