"""Tests for JobTracker."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.job_tracker import JobTracker, JobTrackerSnapshot, _JobTrackerState

from .conftest import (
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


def test_job_tracker_snapshot_fields_match_1_to_1() -> None:
    """Verify the fields are exactly equal between _JobTrackerState and JobTrackerSnapshot."""
    state_fields = {f.name for f in _JobTrackerState.__dataclass_fields__.values()}
    snapshot_fields = {f.name for f in JobTrackerSnapshot.__dataclass_fields__.values()}
    assert state_fields == snapshot_fields
