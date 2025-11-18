"""Tests for job queue analyzer."""

from unittest.mock import Mock

from horde_worker_regen.utils.job_queue_analyzer import JobQueueAnalyzer


def create_mock_job(
    width: int = 512,
    height: int = 512,
    ddim_steps: int = 30,
    n_iter: int = 1,
) -> Mock:
    """Create a mock ImageGenerateJobPopResponse for testing.

    Args:
        width: Image width.
        height: Image height.
        ddim_steps: Number of steps.
        n_iter: Number of iterations.

    Returns:
        Mock job object.
    """
    job = Mock()
    job.payload = Mock()
    job.payload.width = width
    job.payload.height = height
    job.payload.ddim_steps = ddim_steps
    job.payload.n_iter = n_iter
    job.payload.post_processing = []
    job.payload.loras = []
    job.payload.hires_fix = False
    job.payload.control_type = None
    job.model = "stable_diffusion"
    return job


def test_calculate_pending_megapixelsteps_empty_queues() -> None:
    """Test calculating pending megapixelsteps with empty queues."""
    result = JobQueueAnalyzer.calculate_pending_megapixelsteps(
        jobs_pending_inference=[],
        jobs_pending_submit_count=0,
    )

    assert result == 0


def test_calculate_pending_megapixelsteps_single_job() -> None:
    """Test calculating pending megapixelsteps with a single job."""
    job = create_mock_job(width=512, height=512, ddim_steps=30, n_iter=1)

    result = JobQueueAnalyzer.calculate_pending_megapixelsteps(
        jobs_pending_inference=[job],
        jobs_pending_submit_count=0,
    )

    # 512x512 = 0.262144 megapixels, 30 steps = ~7.86 megapixelsteps
    # Rounded to 7
    assert result == 7


def test_calculate_pending_megapixelsteps_multiple_jobs() -> None:
    """Test calculating pending megapixelsteps with multiple jobs."""
    job1 = create_mock_job(width=512, height=512, ddim_steps=30, n_iter=1)
    job2 = create_mock_job(width=1024, height=1024, ddim_steps=50, n_iter=1)

    result = JobQueueAnalyzer.calculate_pending_megapixelsteps(
        jobs_pending_inference=[job1, job2],
        jobs_pending_submit_count=0,
    )

    # Job1: 7 MPS, Job2: 52 MPS
    assert result == 59


def test_calculate_pending_megapixelsteps_with_submit_queue() -> None:
    """Test calculating pending megapixelsteps with jobs pending submit."""
    job = create_mock_job(width=512, height=512, ddim_steps=30, n_iter=1)

    result = JobQueueAnalyzer.calculate_pending_megapixelsteps(
        jobs_pending_inference=[job],
        jobs_pending_submit_count=2,
    )

    # Job: 7 MPS, 2 pending submit: 2*4 = 8 MPS
    assert result == 15


def test_calculate_pending_megapixelsteps_only_submit_queue() -> None:
    """Test calculating pending megapixelsteps with only jobs pending submit."""
    result = JobQueueAnalyzer.calculate_pending_megapixelsteps(
        jobs_pending_inference=[],
        jobs_pending_submit_count=5,
    )

    # 5 pending submit: 5*4 = 20 MPS
    assert result == 20


def test_calculate_pending_megapixelsteps_large_job() -> None:
    """Test calculating pending megapixelsteps with a large resolution job."""
    job = create_mock_job(width=2048, height=2048, ddim_steps=100, n_iter=1)

    result = JobQueueAnalyzer.calculate_pending_megapixelsteps(
        jobs_pending_inference=[job],
        jobs_pending_submit_count=0,
    )

    # 2048x2048 = 4.194304 megapixels, 100 steps = ~419 megapixelsteps
    assert result == 419


def test_calculate_pending_megapixelsteps_batched_job() -> None:
    """Test calculating pending megapixelsteps with a batched job (n_iter > 1)."""
    job = create_mock_job(width=512, height=512, ddim_steps=30, n_iter=4)

    result = JobQueueAnalyzer.calculate_pending_megapixelsteps(
        jobs_pending_inference=[job],
        jobs_pending_submit_count=0,
    )

    # Single job with n_iter=4: batching_multiplier = 1 + ((4-1) * 0.2) = 1.6
    # 512*512*1.6*30/1M = ~12 MPS
    assert result == 12


def test_should_wait_for_pending_megapixelsteps_below_limit() -> None:
    """Test should_wait when pending is below limit."""
    result = JobQueueAnalyzer.should_wait_for_pending_megapixelsteps(
        pending_megapixelsteps=50,
        max_pending_megapixelsteps=100,
    )

    assert result is False


def test_should_wait_for_pending_megapixelsteps_at_limit() -> None:
    """Test should_wait when pending equals limit."""
    result = JobQueueAnalyzer.should_wait_for_pending_megapixelsteps(
        pending_megapixelsteps=100,
        max_pending_megapixelsteps=100,
    )

    assert result is False


def test_should_wait_for_pending_megapixelsteps_above_limit() -> None:
    """Test should_wait when pending exceeds limit."""
    result = JobQueueAnalyzer.should_wait_for_pending_megapixelsteps(
        pending_megapixelsteps=150,
        max_pending_megapixelsteps=100,
    )

    assert result is True


def test_should_wait_for_pending_megapixelsteps_slightly_above() -> None:
    """Test should_wait when pending slightly exceeds limit."""
    result = JobQueueAnalyzer.should_wait_for_pending_megapixelsteps(
        pending_megapixelsteps=101,
        max_pending_megapixelsteps=100,
    )

    assert result is True


def test_should_wait_for_pending_megapixelsteps_zero() -> None:
    """Test should_wait when pending is zero."""
    result = JobQueueAnalyzer.should_wait_for_pending_megapixelsteps(
        pending_megapixelsteps=0,
        max_pending_megapixelsteps=100,
    )

    assert result is False


def test_should_wait_for_pending_megapixelsteps_zero_limit() -> None:
    """Test should_wait with zero limit (edge case)."""
    result = JobQueueAnalyzer.should_wait_for_pending_megapixelsteps(
        pending_megapixelsteps=1,
        max_pending_megapixelsteps=0,
    )

    assert result is True


def test_calculate_pending_megapixelsteps_mixed_sizes() -> None:
    """Test calculating pending megapixelsteps with various job sizes."""
    jobs = [
        create_mock_job(width=512, height=512, ddim_steps=20, n_iter=1),
        create_mock_job(width=768, height=768, ddim_steps=30, n_iter=1),
        create_mock_job(width=1024, height=1024, ddim_steps=40, n_iter=1),
    ]

    result = JobQueueAnalyzer.calculate_pending_megapixelsteps(
        jobs_pending_inference=jobs,
        jobs_pending_submit_count=1,
    )

    # Job1: ~5 MPS, Job2: ~18 MPS, Job3: ~40 MPS, 1 pending submit: 4 MPS
    # Total: ~67 MPS
    assert result == 67
