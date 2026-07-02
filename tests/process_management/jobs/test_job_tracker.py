"""Tests for JobTracker."""

from __future__ import annotations

from unittest.mock import Mock

from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.ipc.messages import HordeImageResult
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from tests.process_management.conftest import (
    make_mock_job,
    mark_job_in_progress_async,
    move_job_to_being_safety_checked_async,
    queue_job_for_safety_async,
    queue_job_for_submit_async,
    track_popped_job_async,
)


def test_init_empty_collections(job_tracker: JobTracker) -> None:
    """Test that the constructor initializes all collections to empty."""
    assert len(job_tracker.jobs_lookup) == 0
    assert len(job_tracker.jobs_in_progress) == 0
    assert len(job_tracker.job_faults) == 0
    assert len(job_tracker.jobs_pending_safety_check) == 0
    assert len(job_tracker.jobs_being_safety_checked) == 0
    assert len(job_tracker.jobs_pending_submit) == 0
    assert len(job_tracker.jobs_pending_inference) == 0
    assert len(job_tracker.job_pop_timestamps) == 0
    assert job_tracker.num_jobs_faulted == 0
    assert job_tracker.total_num_completed_jobs == 0


def test_num_jobs_total_empty(job_tracker: JobTracker) -> None:
    """If there are no jobs in any collection, num_jobs_total should return 0."""
    assert job_tracker.num_jobs_total == 0


async def test_num_jobs_total_counts_all_stages(job_tracker: JobTracker) -> None:
    """num_jobs_total should return the total number of jobs across all collections."""
    pending = Mock()
    pending.id_ = "pending"
    await track_popped_job_async(job_tracker, pending)

    in_progress = Mock()
    in_progress.id_ = "in-progress"
    await mark_job_in_progress_async(job_tracker, in_progress)

    await queue_job_for_safety_async(job_tracker, Mock())
    await move_job_to_being_safety_checked_async(job_tracker, Mock())
    await queue_job_for_submit_async(job_tracker, Mock())

    assert job_tracker.num_jobs_total == 5


async def test_current_queue_size(job_tracker: JobTracker) -> None:
    """current_queue_size should return the number of jobs pending inference."""
    assert job_tracker.current_queue_size == 0
    job1 = Mock()
    job1.id_ = "job-1"
    job2 = Mock()
    job2.id_ = "job-2"
    await track_popped_job_async(job_tracker, job1)
    await track_popped_job_async(job_tracker, job2)
    assert job_tracker.current_queue_size == 2


async def test_handle_job_fault_removes_from_pending_inference(
    job_tracker: JobTracker,
    mock_job_pop_response: Mock,
) -> None:
    """When a job fault occurs, it should be removed from the pending inference list."""
    job_info = await job_tracker.record_popped_job(mock_job_pop_response)

    await job_tracker.handle_job_fault(faulted_job=mock_job_pop_response)

    assert mock_job_pop_response not in job_tracker.jobs_pending_inference
    assert job_info in job_tracker.jobs_pending_submit


async def test_handle_job_fault_removes_from_in_progress(
    job_tracker: JobTracker,
    mock_job_pop_response: Mock,
) -> None:
    """When a job fault occurs, it should be removed from the in-progress list."""
    await job_tracker.record_popped_job(mock_job_pop_response)
    await mark_job_in_progress_async(job_tracker, mock_job_pop_response)

    await job_tracker.handle_job_fault(faulted_job=mock_job_pop_response)

    assert mock_job_pop_response not in job_tracker.jobs_in_progress


async def test_handle_job_fault_removes_from_pending_safety_check(
    job_tracker: JobTracker,
    mock_job_pop_response: Mock,
) -> None:
    """When a job fault occurs, it should be removed from the pending safety check list."""
    job_info = await job_tracker.record_popped_job(mock_job_pop_response)
    await queue_job_for_safety_async(job_tracker, job_info)

    await job_tracker.handle_job_fault(faulted_job=mock_job_pop_response)

    assert len(job_tracker.jobs_pending_safety_check) == 0


async def test_handle_job_fault_unknown_job(job_tracker: JobTracker) -> None:
    """Job not in jobs_lookup should log error but not crash."""
    unknown_job = Mock()
    unknown_job.id_ = "unknown-id"
    await job_tracker.handle_job_fault(faulted_job=unknown_job)
    # Should not raise


async def test_handle_job_fault_no_double_add(
    job_tracker: JobTracker,
    mock_job_pop_response: Mock,
) -> None:
    """If a job fault occurs, the job should not be added multiple times to pending submit."""
    job_info = await job_tracker.record_popped_job(mock_job_pop_response)
    await queue_job_for_submit_async(job_tracker, job_info)

    await job_tracker.handle_job_fault(faulted_job=mock_job_pop_response)

    assert sum(1 for info in job_tracker.jobs_pending_submit if info is job_info) == 1


async def test_purge_jobs_clears_all(job_tracker: JobTracker) -> None:
    """_purge_jobs should clear all job collections."""
    pending = Mock()
    pending.id_ = "pending"
    await track_popped_job_async(job_tracker, pending)
    await move_job_to_being_safety_checked_async(job_tracker, Mock())
    await queue_job_for_safety_async(job_tracker, Mock())
    in_progress = Mock()
    in_progress.id_ = "in-progress"
    await mark_job_in_progress_async(job_tracker, in_progress)
    await queue_job_for_submit_async(job_tracker, Mock())

    job_tracker._purge_jobs()

    assert len(job_tracker.jobs_pending_inference) == 0
    assert len(job_tracker.jobs_being_safety_checked) == 0
    assert len(job_tracker.jobs_pending_safety_check) == 0
    assert len(job_tracker.jobs_lookup) == 0
    assert len(job_tracker.jobs_in_progress) == 0
    assert len(job_tracker.jobs_pending_submit) == 0


async def test_purge_jobs_updates_last_submitted_time(job_tracker: JobTracker) -> None:
    """After purging jobs, last_job_submitted_time should be updated to current time."""
    import time

    old_time = job_tracker.snapshot().last_job_submitted_time
    pending = Mock()
    pending.id_ = "pending"
    await track_popped_job_async(job_tracker, pending)
    time.sleep(0.01)

    job_tracker._purge_jobs()

    assert job_tracker.snapshot().last_job_submitted_time > old_time


def test_purge_jobs_empty_noop(job_tracker: JobTracker) -> None:
    """Calling _purge_jobs when there are no jobs should not raise an error."""
    job_tracker._purge_jobs()
    # Should not raise


def test_set_performance_mode_thresholds(job_tracker: JobTracker) -> None:
    """set_performance_mode_thresholds correctly updates the max pending megapixelsteps threshold."""
    job_tracker.set_performance_mode_thresholds(80)
    assert job_tracker._max_pending_megapixelsteps == 80


async def test_should_wait_for_pending_megapixelsteps(job_tracker: JobTracker) -> None:
    """should_wait_for_pending_megapixelsteps returns True when pending megapixelsteps exceed the threshold."""
    job_tracker.set_performance_mode_thresholds(10)
    assert not job_tracker.should_wait_for_pending_megapixelsteps()

    # Add a job that creates enough megapixelsteps to exceed threshold
    big_job = make_mock_job(width=1024, height=1024, ddim_steps=50)
    await track_popped_job_async(job_tracker, big_job)

    assert job_tracker.should_wait_for_pending_megapixelsteps()


async def test_snapshot_reflects_tracker_state(job_tracker: JobTracker) -> None:
    """A snapshot must reflect the live derived views at the moment it is taken."""
    job = await track_popped_job_async(job_tracker, make_mock_job())
    await job_tracker.increment_jobs_completed()

    snapshot = job_tracker.snapshot()

    assert snapshot.jobs_pending_inference == job_tracker.jobs_pending_inference
    assert snapshot.jobs_in_progress == ()
    assert snapshot.total_num_completed_jobs == 1
    assert job in snapshot.jobs_lookup
    assert job in snapshot.job_pop_timestamps


class TestJobStages:
    """Tests for the unified stage model: a job is in exactly one stage."""

    async def test_normal_lifecycle_stages(self, job_tracker: JobTracker) -> None:
        """A job should move through the stages of the happy path one at a time."""
        job = await track_popped_job_async(job_tracker, make_mock_job())
        assert job.id_ is not None
        assert job_tracker.get_stage(job.id_) is JobStage.PENDING_INFERENCE

        await job_tracker.mark_inference_started(job)
        assert job_tracker.get_stage(job.id_) is JobStage.INFERENCE_IN_PROGRESS
        # In-progress jobs remain visible in the pending-inference view
        assert job in job_tracker.jobs_pending_inference

        assert await job_tracker.release_in_progress(job)
        assert await job_tracker.drop_pending_inference_by_id(job.id_)
        assert job_tracker.get_stage(job.id_) is JobStage.DETACHED
        assert job not in job_tracker.jobs_pending_inference

        job_info = await job_tracker.get_job_info(job)
        assert job_info is not None
        await job_tracker.queue_for_safety(job_info)
        assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SAFETY_CHECK

        await job_tracker.begin_safety_check(job_info)
        assert job_tracker.get_stage(job.id_) is JobStage.SAFETY_CHECKING

        taken = await job_tracker.take_being_safety_checked(job.id_)
        assert taken is job_info
        assert job_tracker.get_stage(job.id_) is JobStage.DETACHED

        await job_tracker.queue_for_submit(job_info)
        assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT

        await job_tracker.finalize_submitted(job_info)
        assert job_tracker.get_stage(job.id_) is None
        assert job_tracker.num_jobs_total == 0

    async def test_job_is_never_in_two_stage_views(self, job_tracker: JobTracker) -> None:
        """No matter the call sequence, a job appears in at most one stage-specific view.

        jobs_pending_inference intentionally includes in-progress jobs, so it is
        checked against the downstream views only.
        """
        job = await track_popped_job_async(job_tracker, make_mock_job())
        job_info = await job_tracker.get_job_info(job)
        assert job_info is not None

        # Try to put the job in several stages by skipping intermediate calls
        await job_tracker.mark_inference_started(job)
        await job_tracker.queue_for_safety(job_info)
        await job_tracker.queue_for_submit(job_info)

        views = {
            "in_progress": job in job_tracker.jobs_in_progress,
            "pending_safety": job_info in job_tracker.jobs_pending_safety_check,
            "being_checked": job_info in job_tracker.jobs_being_safety_checked,
            "pending_submit": job_info in job_tracker.jobs_pending_submit,
        }
        assert sum(views.values()) == 1, f"Job is in multiple stage views: {views}"

    async def test_illegal_transition_is_refused(self, job_tracker: JobTracker) -> None:
        """A terminal-stage job cannot move back to an earlier stage."""
        job = await track_popped_job_async(job_tracker, make_mock_job())
        assert job.id_ is not None
        job_info = await job_tracker.get_job_info(job)
        assert job_info is not None

        await job_tracker.queue_for_submit(job_info)
        assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT

        # Attempting to begin a safety check on a pending-submit job must not move it
        await job_tracker.begin_safety_check(job_info)
        assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT

    async def test_duplicate_pop_replaces_tracked_entry(self, job_tracker: JobTracker) -> None:
        """Re-popping an already-tracked generation ID replaces the entry instead of duplicating it."""
        job = await track_popped_job_async(job_tracker, make_mock_job())
        await track_popped_job_async(job_tracker, job)

        assert job_tracker.num_jobs_total == 1
        assert len(job_tracker.jobs_pending_inference) == 1

    async def test_lookup_is_id_based_despite_object_rebuild(self, job_tracker: JobTracker) -> None:
        """A value-equal rebuilt response object must still resolve to the tracked job."""
        job = await track_popped_job_async(job_tracker, make_mock_job())

        rebuilt = job.model_copy(deep=True)
        assert rebuilt is not job

        assert await job_tracker.get_job_info(rebuilt) is not None
        assert await job_tracker.get_time_popped(rebuilt) is not None

    async def test_fault_during_safety_check_prevents_double_submit(self, job_tracker: JobTracker) -> None:
        """A job faulted while being safety checked cannot be taken (and re-queued) again."""
        job = await track_popped_job_async(job_tracker, make_mock_job())
        job_info = await job_tracker.get_job_info(job)
        assert job_info is not None

        await job_tracker.queue_for_safety(job_info)
        await job_tracker.begin_safety_check(job_info)

        await job_tracker.handle_job_fault(faulted_job=job)
        assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT  # pyrefly: ignore

        # The (late) safety result must not be able to re-queue the job for submit
        assert await job_tracker.take_being_safety_checked(job.id_) is None  # pyrefly: ignore
        assert len(job_tracker.jobs_pending_submit) == 1

    async def test_refused_safety_requeue_preserves_existing_result(self, job_tracker: JobTracker) -> None:
        """A stale/duplicate safety re-queue of a PENDING_SUBMIT job must not corrupt its stored result.

        Regression for a shutdown crash loop: queue_for_safety used to overwrite ``job_info`` *before*
        validating the transition, so a late inference result for an already-terminal job replaced its
        safety-checked result with a fresh, pre-safety one (censored=None) while the refused transition
        left the job in PENDING_SUBMIT. The resulting un-submittable "poison" job spun the submit loop
        forever and blocked shutdown.
        """
        job = await track_popped_job_async(job_tracker, make_mock_job())
        assert job.id_ is not None

        safety_checked = HordeJobInfo(
            sdk_api_job_info=job,
            state=GENERATION_STATE.ok,
            censored=False,
            time_popped=0.0,
            job_image_results=[HordeImageResult(image_bytes=b"data")],
        )
        await job_tracker.queue_for_submit(safety_checked)
        assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT

        # A late/duplicate inference result tries to send the already-terminal job back for safety.
        stale_pre_safety = HordeJobInfo(
            sdk_api_job_info=job,
            state=None,
            censored=None,
            time_popped=0.0,
            job_image_results=[HordeImageResult(image_bytes=b"data")],
        )
        await job_tracker.queue_for_safety(stale_pre_safety)

        # Stage is preserved AND the safety-checked result is untouched (not corrupted to censored=None).
        assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT
        head = job_tracker.jobs_pending_submit[0]
        assert head is safety_checked
        assert head.censored is False
