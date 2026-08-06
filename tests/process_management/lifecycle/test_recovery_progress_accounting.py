"""Tests for what the recovery coordinator accepts as forward progress when closing a wedge episode.

The escalation ladder resets only on progress the failure mode cannot manufacture. An attempt counter
(inference starts) rises freely while a downstream stage is the thing that is stuck, so a backlog waiting
on that stage must not be credited as recovery.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from tests.process_management.conftest import (
    FakeClock,
    make_job_pop_response,
    make_test_recovery_coordinator,
    mark_job_in_progress_async,
    move_job_to_being_safety_checked_async,
    queue_job_for_safety_async,
    track_popped_job_async,
)


class TestSafetyBacklogIsNotProgress:
    """A safety-stage backlog is not disproved by more upstream inference starting."""

    async def test_inference_starts_do_not_credit_progress_while_safety_is_backlogged(self) -> None:
        """Starting more inference while safety checks pile up is not forward progress.

        A safety pool that cannot serve leaves accepted work parked after generation. Inference keeps
        starting behind it, so crediting the start counter would close the wedge episode on motion that
        the stalled stage itself produces, resetting the escalation instead of climbing it.
        """
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = make_test_recovery_coordinator(job_tracker=job_tracker, clock=clock)

        parked = await track_popped_job_async(job_tracker, make_job_pop_response())
        # The generated job parks waiting on safety, and a fresh job starts inference behind it.
        await queue_job_for_safety_async(job_tracker, _job_info_for(parked))
        coordinator._capture_progress_baseline()
        follower = await track_popped_job_async(job_tracker, make_job_pop_response())
        await mark_job_in_progress_async(job_tracker, follower)

        assert job_tracker.total_num_inference_starts > 0
        assert job_tracker.jobs_pending_safety_check
        assert coordinator.made_progress_since_episode() is False

    async def test_in_flight_safety_check_also_withholds_progress(self) -> None:
        """A safety check that has begun but never returns withholds credit just as a queued one does."""
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = make_test_recovery_coordinator(job_tracker=job_tracker, clock=clock)

        parked = await track_popped_job_async(job_tracker, make_job_pop_response())
        await move_job_to_being_safety_checked_async(job_tracker, _job_info_for(parked))
        coordinator._capture_progress_baseline()
        follower = await track_popped_job_async(job_tracker, make_job_pop_response())
        await mark_job_in_progress_async(job_tracker, follower)

        assert job_tracker.jobs_being_safety_checked
        assert coordinator.made_progress_since_episode() is False

    async def test_a_drained_safety_stage_restores_progress_credit(self) -> None:
        """With no safety backlog, an inference start is ordinary forward progress again.

        The control for the guard: withholding credit must be scoped to an actual safety backlog, or a
        healthy worker's wedge episode could never close.
        """
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = make_test_recovery_coordinator(job_tracker=job_tracker, clock=clock)

        coordinator._capture_progress_baseline()

        follower = await track_popped_job_async(job_tracker, make_job_pop_response())
        await mark_job_in_progress_async(job_tracker, follower)

        assert not job_tracker.jobs_pending_safety_check
        assert not job_tracker.jobs_being_safety_checked
        assert coordinator.made_progress_since_episode() is True


class TestFaultedWorkIsNotRecoveryProgress:
    """A terminal fault drains accounting but does not prove that the worker can serve accepted work."""

    async def test_terminal_fault_cannot_close_the_episode_that_produced_it(self) -> None:
        """Moving a pending job to faulted submission is not successful forward progress."""
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = make_test_recovery_coordinator(job_tracker=job_tracker, clock=clock)

        job = await track_popped_job_async(job_tracker, make_job_pop_response())
        coordinator._capture_progress_baseline()

        job_tracker.handle_job_fault_now(job, retryable=False)

        assert coordinator.episode_progress_baseline is not None
        assert job_tracker.total_num_completed_jobs > coordinator.episode_progress_baseline
        assert coordinator.made_progress_since_episode() is False


def _job_info_for(job: object) -> Mock:
    """Wrap a popped job in the job-info shape the safety queue expects."""
    job_info = Mock()
    job_info.sdk_api_job_info = job
    job_info.job_image_results = []
    return job_info
