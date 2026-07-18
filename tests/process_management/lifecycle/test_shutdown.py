"""Tests for ShutdownManager."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeImageResult, HordeProcessState
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_submitter import JobSubmitter
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle import shutdown_manager as shutdown_manager_module
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.lifecycle.shutdown_manager import (
    _EMPTY_SHUTDOWN_GRACE_SECONDS,
    _SHUTDOWN_GRACE_BASE_SECONDS,
    _SHUTDOWN_GRACE_PER_JOB_SECONDS,
    MAX_SHUTDOWN_GRACE_SECONDS,
    ShutdownManager,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_test_api_sessions,
    make_test_model_metadata,
    make_test_runtime_config,
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

    def test_alchemy_forms_in_flight_returns_false(self) -> None:
        """Alchemy work should prevent shutdown while it drains."""
        state = WorkerState(shutting_down=True, alchemy_forms_in_flight=1)
        shutdown_manager = _make_shutdown_manager(state=state)
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

    def test_live_safety_process_blocks_shutdown_until_ending(self) -> None:
        """A safety process must be told to end before shutdown is considered complete."""
        state = WorkerState(shutting_down=True)
        safety = make_mock_process_info(
            0,
            model_name=None,
            state=HordeProcessState.PROCESS_STARTING,
            process_type=HordeProcessType.SAFETY,
        )
        process_map = ProcessMap({0: safety})
        shutdown_manager = _make_shutdown_manager(state=state, process_map=process_map)

        assert shutdown_manager.is_time_for_shutdown() is False

        process_map.on_process_ending(0)
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
        shutdown_manager.start_timed_shutdown = Mock()

        shutdown_manager.abort()

        assert shutdown_manager._state.shutting_down is True
        shutdown_manager._job_tracker._purge_jobs.assert_called_once()
        shutdown_manager._process_lifecycle._hard_kill_processes.assert_called_once()
        shutdown_manager.start_timed_shutdown.assert_called_once()


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

    def test_empty_pipeline_uses_short_grace(self) -> None:
        """With no accepted work in flight, the force-kill backstop uses the short empty-pipeline grace."""
        shutdown_manager = _make_shutdown_manager()
        assert shutdown_manager._compute_shutdown_grace() == _EMPTY_SHUTDOWN_GRACE_SECONDS

    def test_alchemy_work_uses_drain_grace(self) -> None:
        """Alchemy in flight is accepted work, so it must not take the empty-pipeline fast path."""
        state = WorkerState(alchemy_forms_in_flight=1)
        shutdown_manager = _make_shutdown_manager(state=state)
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


class TestShutdownDrainsUnsubmittableJob:
    """Integration: an un-submittable job must never be able to block shutdown forever.

    This reproduces the failure end to end: a job reaches PENDING_SUBMIT without a safety verdict
    (censored is None), is_time_for_shutdown() stays False because the submit queue is non-empty, and a
    naive submitter would spin on it forever. The submitter must punt the job so the queue drains and
    shutdown can complete.
    """

    async def test_poison_submit_job_is_drained_and_shutdown_completes(self) -> None:
        """A poison submit job blocks shutdown until the submitter punts it, then shutdown is ready."""
        state = WorkerState(shutting_down=True)
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion", r2_upload="https://example.com/upload")
        poison = HordeJobInfo(
            sdk_api_job_info=job,
            state=GENERATION_STATE.ok,
            time_popped=0.0,
            job_image_results=[HordeImageResult(image_bytes=b"data")],
            censored=None,
        )
        await track_popped_job_async(job_tracker, job, time_popped=0.0)
        await queue_job_for_submit_async(job_tracker, poison)

        shutdown_manager = _make_shutdown_manager(state=state, job_tracker=job_tracker)
        # The poison job keeps shutdown from completing: jobs_pending_submit > 0.
        assert shutdown_manager.is_time_for_shutdown() is False

        horde_session = AsyncMock()
        horde_session.submit_request = AsyncMock(return_value=Mock(reward=0.0))
        submitter = JobSubmitter(
            state=state,
            job_tracker=job_tracker,
            shutdown_manager=shutdown_manager,
            runtime_config=make_test_runtime_config(),
            api_sessions=make_test_api_sessions(
                horde_client_session=horde_session,
                aiohttp_session=AsyncMock(),
            ),
            model_metadata=make_test_model_metadata(),
        )

        await submitter.api_submit_job()

        assert len(job_tracker.jobs_pending_submit) == 0
        # With the queue drained and no live processes, shutdown can finally complete.
        assert shutdown_manager.is_time_for_shutdown() is True


class TestTimedShutdownBackstop:
    """The background shutdown backstop must terminate the worker process, not only its own thread."""

    def test_backstop_force_exits_after_killing_children(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A timed-out shutdown uses the process-exit lever so the supervisor can observe and restart it."""
        shutdown_manager = _make_shutdown_manager()
        shutdown_manager._process_lifecycle._hard_kill_processes = Mock()
        exit_codes: list[int] = []

        class _InlineThread:
            def __init__(self, target: Callable[[], None], *args: object, **kwargs: object) -> None:
                self._target = target

            def start(self) -> None:
                self._target()

        monkeypatch.setattr(shutdown_manager_module, "_EMPTY_SHUTDOWN_GRACE_SECONDS", 0.0)
        monkeypatch.setattr(shutdown_manager_module, "_force_exit_process", exit_codes.append)
        monkeypatch.setattr(shutdown_manager_module.threading, "Thread", _InlineThread)

        shutdown_manager.start_timed_shutdown()

        shutdown_manager._process_lifecycle._hard_kill_processes.assert_called_once()
        assert exit_codes == [1]


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 3.0) -> bool:
    """Poll ``predicate`` until it is true or the timeout elapses; returns the final value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestBackstopLifecycle:
    """The force-kill backstop must fire for a wedged shutdown yet stay cancellable by an embedder.

    A real thread is used (not the inline stub) so the cancel/grace timing is exercised as it runs in
    production. ``_force_exit_process`` is replaced with a recorder so an actual ``os._exit`` never takes
    the test process down, and the grace is shrunk so the wedged-path window is testable.
    """

    def test_wedged_shutdown_backstop_fires_within_grace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A graceful shutdown that never completes (shut_down stays False, no cancel) is force-exited."""
        shutdown_manager = _make_shutdown_manager()
        shutdown_manager._process_lifecycle._hard_kill_processes = Mock()
        exit_codes: list[int] = []
        monkeypatch.setattr(shutdown_manager_module, "_EMPTY_SHUTDOWN_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(shutdown_manager_module, "_force_exit_process", exit_codes.append)

        shutdown_manager.start_timed_shutdown()

        assert _wait_until(lambda: exit_codes == [1]), "wedged shutdown backstop did not force-exit"
        shutdown_manager._process_lifecycle._hard_kill_processes.assert_called_once()
        thread = shutdown_manager._backstop_thread
        assert thread is not None
        thread.join(timeout=3.0)
        assert not thread.is_alive()

    def test_cancel_suppresses_force_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cancelling an armed backstop wakes its thread and suppresses the force-exit, even past the grace."""
        shutdown_manager = _make_shutdown_manager()
        shutdown_manager._process_lifecycle._hard_kill_processes = Mock()
        exit_codes: list[int] = []
        # A grace long enough that the fire can only happen after the test has cancelled, so the outcome
        # is decided by the cancel and not by a race against the grace.
        monkeypatch.setattr(shutdown_manager_module, "_EMPTY_SHUTDOWN_GRACE_SECONDS", 5.0)
        monkeypatch.setattr(shutdown_manager_module, "_force_exit_process", exit_codes.append)

        shutdown_manager.start_timed_shutdown()
        shutdown_manager.cancel_timed_shutdown()

        # The join inside cancel_timed_shutdown has already reaped the thread; it can never fire now.
        assert shutdown_manager._backstop_thread is None
        assert exit_codes == []
        shutdown_manager._process_lifecycle._hard_kill_processes.assert_not_called()

    def test_cancel_before_arm_neutralizes_a_later_backstop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cancel that precedes arming keeps the later backstop from ever force-exiting.

        Models an embedder that has taken ownership of the interpreter's exit: any backstop the same
        manager arms afterwards must stay inert.
        """
        shutdown_manager = _make_shutdown_manager()
        shutdown_manager._process_lifecycle._hard_kill_processes = Mock()
        exit_codes: list[int] = []
        monkeypatch.setattr(shutdown_manager_module, "_EMPTY_SHUTDOWN_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(shutdown_manager_module, "_force_exit_process", exit_codes.append)

        shutdown_manager.cancel_timed_shutdown()
        shutdown_manager.start_timed_shutdown()

        thread = shutdown_manager._backstop_thread
        assert thread is not None
        thread.join(timeout=3.0)
        assert not thread.is_alive()
        assert exit_codes == []
        shutdown_manager._process_lifecycle._hard_kill_processes.assert_not_called()

    def test_leaked_backstop_cannot_kill_a_later_lifecycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A backstop armed by one lifecycle must not force-exit the shared process once cancelled.

        Reproduces the cross-run defect directly: two managers share one interpreter (as sequential
        ``run_harness`` calls do). The first arms a backstop whose ``shut_down`` never flips. Left alone it
        force-exits, which in-process would kill the *second* lifecycle. The embedder's cancel (the fix,
        which ``run_harness`` now performs on teardown) must neutralize it so only the intended run owns the
        exit.
        """
        exit_codes: list[int] = []
        monkeypatch.setattr(shutdown_manager_module, "_EMPTY_SHUTDOWN_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(shutdown_manager_module, "_force_exit_process", exit_codes.append)

        # Without the cancel, the leaked backstop reaches the force-exit lever (the behavior that, in one
        # interpreter, terminates the following lifecycle).
        uncancelled = _make_shutdown_manager()
        uncancelled._process_lifecycle._hard_kill_processes = Mock()
        uncancelled.start_timed_shutdown()
        assert _wait_until(lambda: exit_codes == [1]), "an uncancelled leaked backstop should still fire"

        exit_codes.clear()

        # With the cancel the embedder performs on teardown, the same wedged backstop never fires, so a
        # following lifecycle survives.
        cancelled = _make_shutdown_manager()
        cancelled._process_lifecycle._hard_kill_processes = Mock()
        cancelled.start_timed_shutdown()
        cancelled.cancel_timed_shutdown()
        # Give any (incorrectly) surviving thread more than a grace to prove it cannot fire late.
        time.sleep(0.2)
        assert exit_codes == []
        assert cancelled._backstop_thread is None

    def test_backstop_thread_is_a_named_daemon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The backstop runs as a named daemon so a stray one can neither block exit nor hide in a dump."""
        shutdown_manager = _make_shutdown_manager()
        monkeypatch.setattr(shutdown_manager_module, "_EMPTY_SHUTDOWN_GRACE_SECONDS", 5.0)
        monkeypatch.setattr(shutdown_manager_module, "_force_exit_process", Mock())

        shutdown_manager.start_timed_shutdown()
        try:
            thread = shutdown_manager._backstop_thread
            assert thread is not None
            assert thread.daemon is True
            assert thread.name == "shutdown-backstop"
            assert isinstance(thread, threading.Thread)
        finally:
            shutdown_manager.cancel_timed_shutdown()


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
            "horde_worker_regen.process_management.lifecycle.shutdown_manager.threading.Thread",
            _FakeThread,
        )

        shutdown_manager.start_timed_shutdown()
        shutdown_manager.start_timed_shutdown()

        assert len(created) == 1
        assert shutdown_manager._timed_shutdown_started is True
