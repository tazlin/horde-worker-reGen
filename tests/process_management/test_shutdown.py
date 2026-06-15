"""Tests for ShutdownManager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.shutdown_manager import (
    _SHUTDOWN_GRACE_BASE_SECONDS,
    _SHUTDOWN_GRACE_PER_JOB_SECONDS,
    MAX_SHUTDOWN_GRACE_SECONDS,
    ShutdownManager,
)
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
        shutdown_manager._process_lifecycle.recently_recovered = True  # pyrefly: ignore - we aren't testing the process lifecycle here, just that the shutdown manager respects this flag

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

    def test_abort_purges_jobs_and_shuts_down(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling abort should set shutting_down to True, and trigger relevant cleanup actions."""
        # abort() writes a .abort file to the CWD; isolate it so the test cannot
        # signal-shutdown a real worker running from the repo root.
        monkeypatch.chdir(tmp_path)
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


class TestComputeShutdownGrace:
    """The force-kill grace scales with outstanding work (all stages) and is hard-capped."""

    def test_empty_tracker_uses_base_grace(self) -> None:
        """With no jobs in flight, the grace is just the base."""
        shutdown_manager = _make_shutdown_manager()
        assert shutdown_manager._compute_shutdown_grace() == _SHUTDOWN_GRACE_BASE_SECONDS

    async def test_grace_scales_with_outstanding_jobs(self) -> None:
        """Each in-flight job (here, two pending inference) extends the grace."""
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, Mock())
        await track_popped_job_async(job_tracker, Mock())

        shutdown_manager = _make_shutdown_manager(job_tracker=job_tracker)
        expected = _SHUTDOWN_GRACE_BASE_SECONDS + (2 * _SHUTDOWN_GRACE_PER_JOB_SECONDS)
        assert shutdown_manager._compute_shutdown_grace() == pytest.approx(expected)

    async def test_grace_is_capped(self) -> None:
        """A very large backlog cannot push the grace past the hard ceiling."""
        job_tracker = JobTracker()
        for _ in range(100):
            await track_popped_job_async(job_tracker, Mock())

        shutdown_manager = _make_shutdown_manager(job_tracker=job_tracker)
        assert shutdown_manager._compute_shutdown_grace() == MAX_SHUTDOWN_GRACE_SECONDS


class TestFaultReportOutstandingJobs:
    """The last-resort fault-report moves un-submitted in-flight jobs to PENDING_SUBMIT (faulted)."""

    async def test_in_flight_jobs_are_faulted_for_resubmission(self) -> None:
        """Pending-inference jobs are moved to PENDING_SUBMIT (faulted) so the submitter can report them."""
        # shut_down=True so the report's drain poll exits immediately (no submitter runs in this test).
        state = WorkerState(shutting_down=True, shut_down=True)
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, Mock())
        await track_popped_job_async(job_tracker, Mock())
        assert len(job_tracker.jobs_pending_inference) == 2

        shutdown_manager = _make_shutdown_manager(state=state, job_tracker=job_tracker)
        shutdown_manager._fault_report_outstanding_jobs()

        assert len(job_tracker.jobs_pending_inference) == 0
        assert len(job_tracker.jobs_pending_submit) == 2

    def test_no_outstanding_jobs_is_a_noop(self) -> None:
        """With nothing in flight, the fault-report does nothing and never raises."""
        state = WorkerState(shutting_down=True, shut_down=True)
        shutdown_manager = _make_shutdown_manager(state=state)
        shutdown_manager._fault_report_outstanding_jobs()
        assert len(shutdown_manager._job_tracker.jobs_pending_submit) == 0


class TestStartTimedShutdownIdempotent:
    """The force-kill backstop thread is started at most once, however many callers request it."""

    def test_only_one_backstop_thread_is_started(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated requests to start the backstop create exactly one thread."""
        shutdown_manager = _make_shutdown_manager()
        created: list[object] = []

        class _FakeThread:
            def __init__(self, *args: object, **kwargs: object) -> None:
                created.append(self)

            def start(self) -> None:
                pass

        monkeypatch.setattr(
            "horde_worker_regen.process_management.shutdown_manager.threading.Thread",
            _FakeThread,
        )

        shutdown_manager.start_timed_shutdown()
        shutdown_manager.start_timed_shutdown()

        assert len(created) == 1
        assert shutdown_manager._timed_shutdown_started is True
