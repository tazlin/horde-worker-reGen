"""Shared fixtures for process management tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.job_tracker import JobTracker


@pytest.fixture()
def mock_job_pop_response() -> Mock:
    """Create a mock ImageGenerateJobPopResponse."""
    job = Mock()
    job.id_ = "test-job-id-1234"
    job.ids = ["test-job-id-1234"]
    job.model = "stable_diffusion"
    job.payload = Mock()
    job.payload.width = 512
    job.payload.height = 512
    job.payload.ddim_steps = 30
    job.payload.n_iter = 1
    job.payload.post_processing = []
    job.payload.loras = []
    job.payload.hires_fix = False
    job.payload.control_type = None
    job.payload.seed = 42
    return job


@pytest.fixture()
def mock_horde_job_info(mock_job_pop_response: Mock) -> Mock:
    """Create a mock HordeJobInfo wrapping a pop response."""
    job_info = Mock()
    job_info.sdk_api_job_info = mock_job_pop_response
    job_info.state = None
    job_info.time_popped = 0.0
    job_info.time_to_generate = None
    job_info.censored = None
    job_info.job_image_results = None
    return job_info


@pytest.fixture()
def job_tracker() -> JobTracker:
    """Create a fresh JobTracker instance."""
    return JobTracker()


@pytest.fixture()
def mock_process_info() -> Mock:
    """Create a mock HordeProcessInfo."""
    process = Mock()
    process.process_id = 0
    process.loaded_horde_model_name = "stable_diffusion"
    process.last_process_state = Mock()
    process.process_type = Mock()
    return process
