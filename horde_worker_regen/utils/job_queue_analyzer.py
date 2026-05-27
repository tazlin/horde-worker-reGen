"""Job queue analysis utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from horde_worker_regen.utils.job_utils import get_single_job_magnitude

if TYPE_CHECKING:
    from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse


class JobQueueAnalyzer:
    """Analyzes job queues to determine pending workload."""

    @staticmethod
    def calculate_pending_job_magnitude(
        jobs_pending_inference: Iterable[ImageGenerateJobPopResponse],
        jobs_pending_submit_count: int,
    ) -> int:
        """Calculate an approximate magnitude of pending jobs.

        This is based on the magnitude of individual jobs, as calculated by `get_single_job_magnitude`.
        This calculation is heavily approximate and should be used for estimation purposes only.


        Args:
            jobs_pending_inference: List of jobs pending inference.
            jobs_pending_submit_count: Number of jobs pending submission.

        Returns:
            Total pending megapixelsteps.
        """
        job_deque_megapixelsteps = 0

        for job in jobs_pending_inference:
            job_megapixelsteps = get_single_job_magnitude(job)
            job_deque_megapixelsteps += job_megapixelsteps

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
