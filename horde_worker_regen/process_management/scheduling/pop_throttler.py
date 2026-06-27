"""Pop throttling logic: controls how frequently and when job pops can occur."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

from horde_worker_regen.process_management.jobs.job_tracker import JobTracker

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData

CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS: int = 180
"""How long to pause job pops after too many consecutive failures."""


class PopThrottler:
    """Decide whether a job pop should be skipped this cycle.

    Encapsulates timing state, failure-backoff, frequency gating, and
    the megapixelstep wait logic that was previously spread across
    ``JobPopper.api_job_pop``.
    """

    _job_tracker: JobTracker

    _default_pop_frequency: float
    _error_pop_frequency: float
    _current_pop_frequency: float

    _last_pop_no_jobs_available_time: float
    _time_spent_no_jobs_available: float
    _max_time_spent_no_jobs_available: float

    def __init__(
        self,
        *,
        job_tracker: JobTracker,
        default_pop_frequency: float = 1.0,
        error_pop_frequency: float = 5.0,
    ) -> None:
        """Initialize with a job tracker and configurable pop frequencies."""
        self._job_tracker = job_tracker
        self._default_pop_frequency = default_pop_frequency
        self._error_pop_frequency = error_pop_frequency
        self._current_pop_frequency = default_pop_frequency

        self._last_pop_no_jobs_available_time = 0.0
        self._time_spent_no_jobs_available = 0.0
        self._max_time_spent_no_jobs_available = 0.0

    @property
    def current_pop_frequency(self) -> float:
        """Return the current pop frequency in seconds."""
        return self._current_pop_frequency

    @property
    def is_in_error_backoff(self) -> bool:
        """Whether pops are currently slowed due to a recent error.

        While True, fast/urgent pop-ahead must not bypass the frequency gate — the worker is
        deliberately backing off the API after a failure.
        """
        return self._current_pop_frequency > self._default_pop_frequency

    def is_pop_too_soon(self, last_pop_time: float) -> bool:
        """Return True if not enough time has elapsed since the last pop."""
        return (time.time() - last_pop_time) < self._current_pop_frequency

    def on_pop_success(self) -> None:
        """Reset frequency to default after a successful pop."""
        self._current_pop_frequency = self._default_pop_frequency

    def on_pop_error(self) -> None:
        """Slow down pop frequency after an error."""
        self._current_pop_frequency = self._error_pop_frequency

    def on_no_jobs_available(self, cur_time: float, *, queue_empty: bool) -> None:
        """Track idle time when no jobs are available."""
        if queue_empty:
            if self._last_pop_no_jobs_available_time == 0.0:
                self._last_pop_no_jobs_available_time = cur_time
            self._time_spent_no_jobs_available += cur_time - self._last_pop_no_jobs_available_time
            if self._time_spent_no_jobs_available > self._max_time_spent_no_jobs_available:
                self._max_time_spent_no_jobs_available = self._time_spent_no_jobs_available
            self._last_pop_no_jobs_available_time = cur_time

    def on_job_popped(self) -> None:
        """Reset idle tracking after a job is popped."""
        self._last_pop_no_jobs_available_time = 0.0

    def should_wait_for_megapixelsteps(self, bridge_data: reGenBridgeData) -> bool:
        """Return True if job pops should be paused to let large jobs drain.

        Manages the internal trigger state on ``self._job_tracker`` and logs
        the first time a wait is initiated.
        """
        if self._should_preserve_standby_job(bridge_data):
            self._job_tracker.reset_megapixelstep_trigger()
            return False

        if not self._job_tracker.should_wait_for_pending_megapixelsteps():
            self._job_tracker.reset_megapixelstep_trigger()
            return False

        seconds_to_wait = self._calculate_megapixelstep_wait(bridge_data)

        if not self._job_tracker._triggered_max_pending_megapixelsteps:
            self._job_tracker._triggered_max_pending_megapixelsteps = True
            self._job_tracker._triggered_max_pending_megapixelsteps_time = time.time()
            if seconds_to_wait > 2:
                logger.opt(ansi=True).info(
                    f"<fg #7dcea0><i>Pausing job pops for {round(seconds_to_wait, 2)} seconds "
                    "so some long running jobs can make some progress.</i></>",
                )
            logger.debug(
                "Paused job pops for pending megapixelsteps to decrease below "
                f"{self._job_tracker._max_pending_megapixelsteps}",
            )
            logger.debug(
                f"Pending megapixelsteps: {self._job_tracker.get_pending_megapixelsteps()} | "
                f"Max pending megapixelsteps: {self._job_tracker._max_pending_megapixelsteps} | "
                f"Scheduled to wait for {seconds_to_wait} seconds",
            )
            logger.debug(
                f"high_performance_mode: {bridge_data.high_performance_mode} | "
                f"moderate_performance_mode: {bridge_data.moderate_performance_mode}",
            )
            return True

        elapsed = time.time() - self._job_tracker._triggered_max_pending_megapixelsteps_time
        if elapsed <= seconds_to_wait:
            return True

        self._job_tracker.reset_megapixelstep_trigger()
        logger.debug(
            "Pending megapixelsteps decreased below "
            f"{self._job_tracker._max_pending_megapixelsteps}, continuing with job pops",
        )
        return False

    def _should_preserve_standby_job(self, bridge_data: reGenBridgeData) -> bool:
        """Return whether popping should continue to fill the first standby slot."""
        queue_size = bridge_data.queue_size
        if queue_size <= 0:
            return False

        standby_jobs = len(self._job_tracker.jobs_pending_inference) - len(self._job_tracker.jobs_in_progress)
        return standby_jobs < 1

    def _calculate_megapixelstep_wait(self, bridge_data: reGenBridgeData) -> float:
        """Calculate how many seconds to wait based on pending megapixelsteps and config."""
        pending = self._job_tracker.get_pending_megapixelsteps()

        if pending < 40:
            seconds_to_wait = pending * 0.5
        elif pending < 80:
            seconds_to_wait = pending * 0.7
        else:
            seconds_to_wait = pending * 0.8

        if bridge_data.max_threads > 1:
            seconds_to_wait *= 0.75

        if bridge_data.high_performance_mode:
            seconds_to_wait *= 0.2
            if seconds_to_wait < 35:
                seconds_to_wait = 1
        elif bridge_data.moderate_performance_mode:
            seconds_to_wait *= 0.4
            if seconds_to_wait < 20:
                seconds_to_wait = 1

        return seconds_to_wait
