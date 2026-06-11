"""Manages graceful and forceful shutdown of the worker."""

from __future__ import annotations

import sys
import threading
import time

from loguru import logger

from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState


class ShutdownManager:
    """Owns shutdown/abort/signal-handling logic and related state."""

    _state: WorkerState
    _job_tracker: JobTracker
    _process_map: ProcessMap
    _process_lifecycle: ProcessLifecycleManager
    _caught_sigints: int

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

    def start_timed_shutdown(self) -> None:
        """Launch a background thread that force-kills all processes after a grace period."""

        def hard_shutdown() -> None:
            time.sleep((len(self._job_tracker.jobs_pending_submit) * 4) + 2)

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
