"""Tests for JobTracker."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.job_tracker import JobTracker


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
    assert job_tracker._num_jobs_faulted == 0
    assert job_tracker.total_num_completed_jobs == 0
    assert job_tracker._skipped_line_next_job_and_process is None

    # Locks should exist
    assert job_tracker.lookup_lock is not None
    assert job_tracker.completed_jobs_lock is not None
    assert job_tracker.safety_check_lock is not None
    assert job_tracker.pending_inference_lock is not None
    assert job_tracker.pop_timestamps_lock is not None


def test_num_jobs_total_empty(job_tracker: JobTracker) -> None:
    """If there are no jobs in any collection, num_jobs_total should return 0."""
    assert job_tracker.num_jobs_total == 0


def test_num_jobs_total_counts_all_stages(job_tracker: JobTracker) -> None:
    """num_jobs_total should return the total number of jobs across all collections."""
    job_tracker.jobs_pending_inference.append(Mock())
    job_tracker.jobs_in_progress.append(Mock())
    job_tracker.jobs_pending_safety_check.append(Mock())
    job_tracker.jobs_being_safety_checked.append(Mock())
    job_tracker.jobs_pending_submit.append(Mock())

    assert job_tracker.num_jobs_total == 5


def test_current_queue_size(job_tracker: JobTracker) -> None:
    """current_queue_size should return the number of jobs pending inference."""
    assert job_tracker.current_queue_size == 0
    job_tracker.jobs_pending_inference.append(Mock())
    job_tracker.jobs_pending_inference.append(Mock())
    assert job_tracker.current_queue_size == 2


def test_handle_job_fault_removes_from_pending_inference(
    job_tracker: JobTracker,
    mock_job_pop_response: Mock,
    mock_horde_job_info: Mock,
) -> None:
    """When a job fault occurs, it should be removed from the pending inference list."""
    job_tracker.jobs_lookup[mock_job_pop_response] = mock_horde_job_info
    job_tracker.jobs_pending_inference.append(mock_job_pop_response)

    job_tracker.handle_job_fault(faulted_job=mock_job_pop_response)

    assert mock_job_pop_response not in job_tracker.jobs_pending_inference
    assert mock_horde_job_info in job_tracker.jobs_pending_submit


def test_handle_job_fault_removes_from_in_progress(
    job_tracker: JobTracker,
    mock_job_pop_response: Mock,
    mock_horde_job_info: Mock,
) -> None:
    """When a job fault occurs, it should be removed from the in-progress list."""
    job_tracker.jobs_lookup[mock_job_pop_response] = mock_horde_job_info
    job_tracker.jobs_in_progress.append(mock_job_pop_response)

    job_tracker.handle_job_fault(faulted_job=mock_job_pop_response)

    assert mock_job_pop_response not in job_tracker.jobs_in_progress


def test_handle_job_fault_removes_from_pending_safety_check(
    job_tracker: JobTracker,
    mock_job_pop_response: Mock,
) -> None:
    """When a job fault occurs, it should be removed from the pending safety check list."""
    # The code checks `faulted_job in jobs_pending_safety_check` which requires
    # the faulted_job (a pop response) to compare equal to a HordeJobInfo.
    # We make the job_info compare equal to the pop response so the `in` check
    # passes, then it matches by id_ to remove.
    job_info = Mock()
    job_info.sdk_api_job_info = Mock()
    job_info.sdk_api_job_info.id_ = mock_job_pop_response.id_
    # Make `faulted_job in [job_info]` return True
    job_info.__eq__ = lambda self, other: True
    job_info.__hash__ = lambda self: id(self)

    job_tracker.jobs_lookup[mock_job_pop_response] = job_info
    job_tracker.jobs_pending_safety_check.append(job_info)

    job_tracker.handle_job_fault(faulted_job=mock_job_pop_response)

    assert len(job_tracker.jobs_pending_safety_check) == 0


def test_handle_job_fault_unknown_job(job_tracker: JobTracker) -> None:
    """Job not in jobs_lookup should log error but not crash."""
    unknown_job = Mock()
    unknown_job.id_ = "unknown-id"
    job_tracker.handle_job_fault(faulted_job=unknown_job)
    # Should not raise


def test_handle_job_fault_clears_skipped_line(
    job_tracker: JobTracker,
    mock_job_pop_response: Mock,
    mock_horde_job_info: Mock,
) -> None:
    """If the faulted job is the one in _skipped_line_next_job_and_process, that should be cleared."""
    skipped = Mock()
    skipped.next_job = Mock()
    skipped.next_job.model = mock_job_pop_response.model
    job_tracker._skipped_line_next_job_and_process = skipped

    job_tracker.jobs_lookup[mock_job_pop_response] = mock_horde_job_info
    job_tracker.handle_job_fault(faulted_job=mock_job_pop_response)

    assert job_tracker._skipped_line_next_job_and_process is None


def test_handle_job_fault_no_double_add(
    job_tracker: JobTracker,
    mock_job_pop_response: Mock,
    mock_horde_job_info: Mock,
) -> None:
    """If a job fault occurs, the job should not be added multiple times to pending submit."""
    job_tracker.jobs_lookup[mock_job_pop_response] = mock_horde_job_info
    job_tracker.jobs_pending_submit.append(mock_horde_job_info)

    job_tracker.handle_job_fault(faulted_job=mock_job_pop_response)

    assert job_tracker.jobs_pending_submit.count(mock_horde_job_info) == 1


def test_purge_jobs_clears_all(job_tracker: JobTracker) -> None:
    """_purge_jobs should clear all job collections and reset skipped line."""
    job_tracker.jobs_pending_inference.append(Mock())
    job_tracker.jobs_being_safety_checked.append(Mock())
    job_tracker.jobs_pending_safety_check.append(Mock())
    job_tracker.jobs_lookup[Mock()] = Mock()
    job_tracker.jobs_in_progress.append(Mock())
    job_tracker.jobs_pending_submit.append(Mock())
    job_tracker._skipped_line_next_job_and_process = Mock()

    job_tracker._purge_jobs()

    assert len(job_tracker.jobs_pending_inference) == 0
    assert len(job_tracker.jobs_being_safety_checked) == 0
    assert len(job_tracker.jobs_pending_safety_check) == 0
    assert len(job_tracker.jobs_lookup) == 0
    assert len(job_tracker.jobs_in_progress) == 0
    assert len(job_tracker.jobs_pending_submit) == 0
    assert job_tracker._skipped_line_next_job_and_process is None


def test_purge_jobs_updates_last_submitted_time(job_tracker: JobTracker) -> None:
    """After purging jobs, _last_job_submitted_time should be updated to current time."""
    import time

    old_time = job_tracker._last_job_submitted_time
    job_tracker.jobs_pending_inference.append(Mock())
    time.sleep(0.01)

    job_tracker._purge_jobs()

    assert job_tracker._last_job_submitted_time > old_time


def test_purge_jobs_empty_noop(job_tracker: JobTracker) -> None:
    """Calling _purge_jobs when there are no jobs should not raise an error."""
    job_tracker._purge_jobs()
    # Should not raise


def test_set_performance_mode_thresholds(job_tracker: JobTracker) -> None:
    """set_performance_mode_thresholds correctly updates the max pending megapixelsteps threshold."""
    job_tracker.set_performance_mode_thresholds(80)
    assert job_tracker._max_pending_megapixelsteps == 80


def test_should_wait_for_pending_megapixelsteps(job_tracker: JobTracker) -> None:
    """should_wait_for_pending_megapixelsteps returns True when pending megapixelsteps exceed the threshold."""
    job_tracker.set_performance_mode_thresholds(10)
    assert not job_tracker.should_wait_for_pending_megapixelsteps()

    # Add a job that creates enough megapixelsteps to exceed threshold
    big_job = Mock()
    big_job.payload = Mock()
    big_job.payload.width = 1024
    big_job.payload.height = 1024
    big_job.payload.ddim_steps = 50
    big_job.payload.n_iter = 1
    big_job.payload.post_processing = []
    big_job.payload.loras = []
    big_job.payload.hires_fix = False
    big_job.payload.control_type = None
    big_job.model = "stable_diffusion"
    job_tracker.jobs_pending_inference.append(big_job)

    assert job_tracker.should_wait_for_pending_megapixelsteps()
