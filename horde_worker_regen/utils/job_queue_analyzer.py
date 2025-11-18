"""Job queue analysis utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from horde_worker_regen.utils.job_utils import get_single_job_effective_megapixelsteps

if TYPE_CHECKING:
    from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse


class JobQueueAnalyzer:
    """Analyzes job queues to determine pending workload."""

    @staticmethod
    def calculate_pending_megapixelsteps(
        jobs_pending_inference: Iterable[ImageGenerateJobPopResponse],
        jobs_pending_submit_count: int,
    ) -> int:
        """Calculate the total number of megapixelsteps pending in the job queues.

        Args:
            jobs_pending_inference: List of jobs pending inference.
            jobs_pending_submit_count: Number of jobs pending submission.

        Returns:
            Total pending megapixelsteps.
        """
        job_deque_megapixelsteps = 0

        # Calculate megapixelsteps for jobs pending inference
        for job in jobs_pending_inference:
            job_megapixelsteps = get_single_job_effective_megapixelsteps(job)
            job_deque_megapixelsteps += job_megapixelsteps

        # Add 4 megapixelsteps for each job pending submit
        job_deque_megapixelsteps += jobs_pending_submit_count * 4

        return job_deque_megapixelsteps

    @staticmethod
    def should_wait_for_pending_megapixelsteps(
        pending_megapixelsteps: int,
        max_pending_megapixelsteps: int,
    ) -> bool:
        """Check if we should wait due to pending megapixelsteps exceeding the limit.

        Args:
            pending_megapixelsteps: Current pending megapixelsteps.
            max_pending_megapixelsteps: Maximum allowed pending megapixelsteps.

        Returns:
            True if we should wait, False otherwise.
        """
        return pending_megapixelsteps > max_pending_megapixelsteps
