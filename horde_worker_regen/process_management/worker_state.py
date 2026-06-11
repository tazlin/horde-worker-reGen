"""Mutable shared state container for cross-cutting worker flags.

All sub-components that need to read or write these flags receive a
reference to the same WorkerState instance, eliminating callback
lambdas and cross-component property writes.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque


@dataclasses.dataclass
class WorkerState:
    """Cross-cutting mutable flags shared by all process-management sub-components."""

    shutting_down: bool = False
    shutting_down_time: float = 0.0
    shut_down: bool = False

    last_job_pop_time: float = 0.0
    last_pop_no_jobs_available: bool = False
    last_pop_maintenance_mode: bool = False

    consecutive_failed_jobs: int = 0
    too_many_consecutive_failed_jobs: bool = False
    too_many_consecutive_failed_jobs_time: float = 0.0
    # Must match CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS in pop_throttler.py;
    # this field is used by the status reporter for display only.
    too_many_consecutive_failed_jobs_wait_time: float = 180

    kudos_generated_this_session: float = 0.0
    kudos_events: deque[tuple[float, float]] = dataclasses.field(default_factory=deque)

    def initiate_shutdown(self) -> None:
        """Mark the worker as shutting down (idempotent)."""
        if not self.shutting_down:
            self.shutting_down = True
            self.shutting_down_time = time.time()

    def last_pop_recently(self) -> bool:
        """Return True if a job was popped within the last 10 seconds."""
        return (time.time() - self.last_job_pop_time) < 10
