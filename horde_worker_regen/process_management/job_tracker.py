"""Tracks all job-related state: collections, locks, and job lifecycle methods."""

from __future__ import annotations

import contextlib
import time
from asyncio import Lock as Lock_Asyncio
from collections import deque
from collections.abc import AsyncIterator

from horde_sdk.ai_horde_api.apimodels import (
    GenMetadataEntry,
    ImageGenerateJobPopResponse,
)
from horde_sdk.ai_horde_api.fields import JobID
from loguru import logger

from horde_worker_regen.process_management.job_models import (
    HordeJobInfo,
    NextJobAndProcess,
)
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.utils.job_queue_analyzer import JobQueueAnalyzer


class JobTracker:
    """Owns all job collections, locks, and provides job lifecycle methods."""

    jobs_lookup: dict[ImageGenerateJobPopResponse, HordeJobInfo]
    jobs_in_progress: list[ImageGenerateJobPopResponse]
    job_faults: dict[JobID, list[GenMetadataEntry]]
    jobs_pending_safety_check: list[HordeJobInfo]
    jobs_being_safety_checked: list[HordeJobInfo]
    jobs_pending_submit: list[HordeJobInfo]
    jobs_pending_inference: deque[ImageGenerateJobPopResponse]
    job_pop_timestamps: dict[ImageGenerateJobPopResponse, float]

    lookup_lock: Lock_Asyncio
    completed_jobs_lock: Lock_Asyncio
    safety_check_lock: Lock_Asyncio
    pending_inference_lock: Lock_Asyncio
    pop_timestamps_lock: Lock_Asyncio

    _num_jobs_faulted: int
    total_num_completed_jobs: int
    _max_pending_megapixelsteps: int
    _triggered_max_pending_megapixelsteps: bool
    _triggered_max_pending_megapixelsteps_time: float
    _last_job_submitted_time: float
    _skipped_line_next_job_and_process: NextJobAndProcess | None

    def __init__(self) -> None:
        """Initialize all job collections, locks, and counters."""
        self.jobs_lookup = {}
        self.jobs_in_progress = []
        self.job_faults = {}
        self.jobs_pending_safety_check = []
        self.jobs_being_safety_checked = []
        self.jobs_pending_submit = []
        self.jobs_pending_inference = deque()
        self.job_pop_timestamps = {}

        self.lookup_lock = Lock_Asyncio()
        self.completed_jobs_lock = Lock_Asyncio()
        self.safety_check_lock = Lock_Asyncio()
        self.pending_inference_lock = Lock_Asyncio()
        self.pop_timestamps_lock = Lock_Asyncio()

        self._all_locks = (
            self.lookup_lock,
            self.pending_inference_lock,
            self.safety_check_lock,
            self.completed_jobs_lock,
        )
        self._all_locks_with_timestamps = (
            self.lookup_lock,
            self.pending_inference_lock,
            self.safety_check_lock,
            self.completed_jobs_lock,
            self.pop_timestamps_lock,
        )

        self._num_jobs_faulted = 0
        self.total_num_completed_jobs = 0
        self._max_pending_megapixelsteps = 25
        self._triggered_max_pending_megapixelsteps = False
        self._triggered_max_pending_megapixelsteps_time = 0.0
        self._last_job_submitted_time = time.time()
        self._skipped_line_next_job_and_process = None

    @contextlib.asynccontextmanager
    async def all_locks(self, *, include_timestamps: bool = False) -> AsyncIterator[None]:
        """Acquire all job-related locks."""
        locks = self._all_locks_with_timestamps if include_timestamps else self._all_locks
        async with contextlib.AsyncExitStack() as stack:
            for lock in locks:
                await stack.enter_async_context(lock)
            yield

    @property
    def num_jobs_total(self) -> int:
        """The total number of jobs across all live stages."""
        return (
            len(self.jobs_pending_inference)
            + len(self.jobs_in_progress)
            + len(self.jobs_pending_safety_check)
            + len(self.jobs_being_safety_checked)
            + len(self.jobs_pending_submit)
        )

    @property
    def current_queue_size(self) -> int:
        """The current number of jobs queued for inference."""
        return len(self.jobs_pending_inference)

    def handle_job_fault(
        self,
        faulted_job: ImageGenerateJobPopResponse,
        process_info: HordeProcessInfo | None = None,
        process_timeout: float = 0.0,
    ) -> None:
        """Mark a job as faulted and add it to the completed jobs list to report it faulted.

        Args:
            faulted_job: The job that faulted.
            process_info: The process that faulted the job.
            process_timeout: The configured process timeout value for time_to_generate.
        """
        job_info = self.jobs_lookup.get(faulted_job)

        if job_info is None:
            logger.error(f"Job {faulted_job.id_} not found in jobs_lookup")
        else:
            if faulted_job in self.jobs_pending_inference:
                self.jobs_pending_inference.remove(faulted_job)

            if (
                self._skipped_line_next_job_and_process is not None
                and faulted_job.model == self._skipped_line_next_job_and_process.next_job.model
            ):
                self._skipped_line_next_job_and_process = None

            job_info.fault_job()
            job_info.time_to_generate = process_timeout

            if process_info is not None:
                logger.error(f"Job {faulted_job.id_} faulted due to process {process_info.process_id} crashing")

            if faulted_job in self.jobs_in_progress:
                logger.debug(f"Removing job {faulted_job.id_} from jobs_in_progress")
                self.jobs_in_progress.remove(faulted_job)

            if faulted_job in self.jobs_pending_safety_check:
                logger.debug(f"Removing job {faulted_job.id_} from jobs_pending_safety_check")
                for horde_job_info in self.jobs_pending_safety_check:
                    if horde_job_info.sdk_api_job_info.id_ == faulted_job.id_:
                        self.jobs_pending_safety_check.remove(horde_job_info)
                        break

            if job_info not in self.jobs_pending_submit:
                self.jobs_pending_submit.append(job_info)
            else:
                logger.warning(f"Job {faulted_job.id_} already in completed_jobs")

    def _purge_jobs(self) -> None:
        """Clear all jobs immediately.

        Note: This is a last resort and should only be used when the worker is in a black hole and can't recover.
        """
        if len(self.jobs_pending_inference) > 0:
            self.jobs_pending_inference.clear()
            self._last_job_submitted_time = time.time()
            logger.error("Cleared jobs pending inference")

        if len(self.jobs_being_safety_checked) > 0:
            self.jobs_being_safety_checked.clear()
            logger.error("Cleared jobs being safety checked")

        if len(self.jobs_pending_safety_check) > 0:
            self.jobs_pending_safety_check.clear()
            logger.error("Cleared jobs pending safety check")

        if len(self.jobs_lookup) > 0:
            self.jobs_lookup.clear()
            logger.error("Cleared jobs lookup")

        if len(self.jobs_in_progress) > 0:
            self.jobs_in_progress.clear()
            logger.error("Cleared jobs in progress")

        if len(self.jobs_pending_submit) > 0:
            self.jobs_pending_submit.clear()
            logger.error("Cleared completed jobs")

        if self._skipped_line_next_job_and_process is not None:
            self._skipped_line_next_job_and_process = None
            logger.error("Cleared skipped line next job and process")

    def get_pending_megapixelsteps(self) -> int:
        """Return the number of megapixelsteps that are pending in the job deque."""
        return JobQueueAnalyzer.calculate_pending_megapixelsteps(
            self.jobs_pending_inference,
            len(self.jobs_pending_submit),
        )

    def should_wait_for_pending_megapixelsteps(self) -> bool:
        """Check if the number of megapixelsteps in the job deque is above the limit."""
        pending_megapixelsteps = self.get_pending_megapixelsteps()
        return JobQueueAnalyzer.should_wait_for_pending_megapixelsteps(
            pending_megapixelsteps,
            self._max_pending_megapixelsteps,
        )

    def set_performance_mode_thresholds(self, max_pending_megapixelsteps: int) -> None:
        """Set the max pending megapixelsteps threshold.

        Args:
            max_pending_megapixelsteps: The new threshold value.
        """
        self._max_pending_megapixelsteps = max_pending_megapixelsteps
