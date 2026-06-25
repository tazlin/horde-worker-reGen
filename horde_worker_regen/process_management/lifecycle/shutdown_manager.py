"""Manages graceful and forceful shutdown of the worker."""

from __future__ import annotations

import sys
import threading
import time

from loguru import logger

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap

_SHUTDOWN_GRACE_BASE_SECONDS = 20.0
"""Minimum grace before the force-kill backstop fires, regardless of outstanding work."""

_SHUTDOWN_GRACE_PER_JOB_SECONDS = 40.0
"""Extra grace granted per outstanding job (any stage), so in-flight work can drain before a kill."""

MAX_SHUTDOWN_GRACE_SECONDS = 120.0
"""Hard ceiling on the drain grace: long enough for an in-flight inference + safety + submit, but
the backstop must never block forever. The TUI's stop timeout is kept above this (see
``tui.worker_launcher.GRACEFUL_STOP_TIMEOUT_SECONDS``)."""

_FAULT_REPORT_GRACE_SECONDS = 15.0
"""After faulting still-outstanding jobs, how long to let the submitter report those faults to the
API before the last-resort kill."""

_SHUTDOWN_POLL_INTERVAL_SECONDS = 0.25
"""Granularity of the backstop's wait loops, so a clean exit is detected promptly."""


class ShutdownManager:
    """Owns shutdown/abort/signal-handling logic and related state."""

    _state: WorkerState
    _job_tracker: JobTracker
    _process_map: ProcessMap
    _process_lifecycle: ProcessLifecycleManager
    _caught_sigints: int
    _timed_shutdown_started: bool

    def __init__(
        self,
        *,
        state: WorkerState,
        job_tracker: JobTracker,
        process_map: ProcessMap,
        process_lifecycle: ProcessLifecycleManager,
    ) -> None:
        """Initialize the manager with references to the components it needs to manage.

        Args:
            state (WorkerState): The worker's state object, containing all of the mutable flags
                relating to the worker's active state and lifecycle.
            job_tracker (JobTracker): The worker's JobTracker, which tracks all jobs in-flight
                and is responsible for managing their state transitions.
            process_map (ProcessMap): The worker's ProcessMap, which tracks all active processes and
                their states.
            process_lifecycle (ProcessLifecycleManager): The worker's ProcessLifecycleManager, which is responsible
                for launching, monitoring, and killing processes as needed.
        """
        self._state = state
        self._job_tracker = job_tracker
        self._process_map = process_map
        self._process_lifecycle = process_lifecycle
        self._caught_sigints = 0
        self._timed_shutdown_started = False

    def shutdown(self) -> None:
        """Initiate a graceful shutdown (idempotent)."""
        self._state.initiate_shutdown()

    def abort(self) -> None:
        """Exit as soon as possible, aborting all processes and jobs immediately."""
        with logger.catch(), open(".abort", "w") as f:
            f.write("")

        self._job_tracker._purge_jobs()
        self.shutdown()
        self._process_lifecycle._hard_kill_processes()
        self.start_timed_shutdown()

    def signal_handler(self, sig: int, frame: object) -> None:
        """Handle SIGINT and SIGTERM."""
        if self._caught_sigints >= 2:
            logger.warning("Caught SIGINT or SIGTERM three times, exiting immediately")
            self.start_timed_shutdown()
            sys.exit(1)

        self._caught_sigints += 1
        logger.warning("Shutting down after current jobs are finished...")
        self.shutdown()

    def _outstanding_job_count(self) -> int:
        """Count jobs still anywhere in the pipeline (queued, inferring, safety-checking, or pending submit).

        ``jobs_pending_inference`` already includes in-progress inference, so it is not double-counted.
        """
        tracker = self._job_tracker
        return (
            len(tracker.jobs_pending_inference)
            + len(tracker.jobs_pending_safety_check)
            + len(tracker.jobs_being_safety_checked)
            + len(tracker.jobs_pending_submit)
        )

    def _compute_shutdown_grace(self) -> float:
        """Grace before the force-kill backstop, scaled by outstanding work and hard-capped."""
        grace = _SHUTDOWN_GRACE_BASE_SECONDS + (_SHUTDOWN_GRACE_PER_JOB_SECONDS * self._outstanding_job_count())
        return min(grace, MAX_SHUTDOWN_GRACE_SECONDS)

    def _fault_report_outstanding_jobs(self) -> None:
        """Last resort: fault jobs that never reached submit and let the submitter report them.

        Runs only when the generous drain grace has already expired with work still outstanding, so
        those jobs would otherwise be killed silently and only reissued after the horde's own timeout.
        Faulting moves them to PENDING_SUBMIT; the (still-running) submitter loop then reports them as
        faulted so the horde reissues them immediately. Entirely best-effort: any failure here must
        never prevent the kill that follows.
        """
        try:
            # Snapshot to plain lists first; the properties return copies, so iterating them is safe
            # even though the event loop may still be mutating the tracker on another thread.
            jobs_to_fault = list(self._job_tracker.jobs_pending_inference)
            jobs_to_fault.extend(info.sdk_api_job_info for info in self._job_tracker.jobs_pending_safety_check)
            jobs_to_fault.extend(info.sdk_api_job_info for info in self._job_tracker.jobs_being_safety_checked)

            if not jobs_to_fault:
                return

            logger.warning(
                f"Shutdown grace expired with {len(jobs_to_fault)} job(s) still in flight; "
                "faulting them so the horde reissues them promptly.",
            )
            for job in jobs_to_fault:
                try:
                    # Shutting down: drain to a terminal fault so the horde reissues promptly; do not requeue.
                    self._job_tracker.handle_job_fault_now(job, retryable=False)
                except Exception as fault_error:
                    logger.error(f"Failed to fault outstanding job {job.id_ or '?'}: {fault_error}")

            report_deadline = time.monotonic() + _FAULT_REPORT_GRACE_SECONDS
            while time.monotonic() < report_deadline:
                if self._state.shut_down or len(self._job_tracker.jobs_pending_submit) == 0:
                    break
                time.sleep(_SHUTDOWN_POLL_INTERVAL_SECONDS)
        except Exception as report_error:
            logger.error(f"Best-effort fault-report during shutdown failed: {report_error}")

    def start_timed_shutdown(self) -> None:
        """Arm the background force-kill backstop (idempotent: only the first call starts the thread).

        The backstop thread force-kills all processes if the graceful drain does not complete within
        a grace period scaled to the outstanding work.
        """
        if self._timed_shutdown_started:
            return
        self._timed_shutdown_started = True

        grace = self._compute_shutdown_grace()

        def hard_shutdown() -> None:
            # Wait for the graceful path to finish, polling so a clean exit is detected promptly
            # instead of always burning the full grace.
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                if self._state.shut_down:
                    return
                time.sleep(_SHUTDOWN_POLL_INTERVAL_SECONDS)

            # Grace expired with the worker still up: report any stuck jobs, then force the kill.
            self._fault_report_outstanding_jobs()

            for process in self._process_map.values():
                try:
                    process.mp_process.kill()
                    process.mp_process.join(1)
                except Exception as e:
                    logger.error(f"Failed to kill process {process}: {e}")

            # Only force-exit if the graceful shutdown hasn't completed; a clean exit
            # should be left to the main thread (and embedders like the test harness).
            if not self._state.shut_down:
                sys.exit(1)

        threading.Thread(target=hard_shutdown).start()

    def is_time_for_shutdown(self) -> bool:
        """Return True if it is time to shut down."""
        if not self._state.shutting_down:
            return False

        if self._process_lifecycle.recently_recovered:
            return False

        if len(self._job_tracker.jobs_pending_submit) > 0:
            return False
        if (
            len(self._job_tracker.jobs_being_safety_checked) > 0
            or len(self._job_tracker.jobs_pending_safety_check) > 0
        ):
            return False
        if len(self._job_tracker.jobs_in_progress) > 0:
            return False
        if len(self._job_tracker.jobs_pending_inference) > 0:
            return False
        if self._state.alchemy_forms_in_flight > 0:
            return False

        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.SAFETY:
                continue
            if process_info.last_process_state not in (
                HordeProcessState.PROCESS_ENDING,
                HordeProcessState.PROCESS_ENDED,
            ):
                return False

        # If no inference processes exist at all (e.g. before any have started),
        # Python's all([]) returns True — this is intentional: with no processes
        # and no pending/in-progress jobs, we are ready to shut down.
        inference_processes = self._process_map.get_inference_processes()
        if all(
            inference_process.last_process_state == HordeProcessState.PROCESS_ENDING
            or inference_process.last_process_state == HordeProcessState.PROCESS_ENDED
            or inference_process.last_process_state == HordeProcessState.PROCESS_STARTING
            for inference_process in inference_processes
        ):
            return True

        any_process_alive = False

        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if (process_info.last_process_state == HordeProcessState.INFERENCE_STARTING) or (
                process_info.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
            ):
                any_process_alive = True
                continue

        return not any_process_alive
