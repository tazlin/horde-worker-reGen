"""Tests for job utility functions."""

from unittest.mock import MagicMock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.ai_horde_api.consts import KNOWN_UPSCALERS

from horde_worker_regen.consts import (
    KNOWN_CONTROLNET_WORKFLOWS,
    KNOWN_SLOW_MODELS_DIFFICULTIES,
    KNOWN_SLOW_WORKFLOWS,
)
from horde_worker_regen.utils.job_utils import get_single_job_effective_megapixelsteps


def create_mock_job(
    width: int = 512,
    height: int = 512,
    ddim_steps: int = 20,
    n_iter: int = 1,
    post_processing: list[str] | None = None,
    loras: list | None = None,
    hires_fix: bool = False,
    model: str = "stable_diffusion",
    workflow: str = "txt2img",
) -> ImageGenerateJobPopResponse:
    """Create a mock job for testing."""
    job = MagicMock(spec=ImageGenerateJobPopResponse)
    job.payload = MagicMock()
    job.payload.width = width
    job.payload.height = height
    job.payload.ddim_steps = ddim_steps
    job.payload.n_iter = n_iter
    job.payload.post_processing = post_processing or []
    job.payload.loras = loras
    job.payload.hires_fix = hires_fix
    job.payload.workflow = workflow
    job.model = model
    return job


def test_get_single_job_effective_megapixelsteps_basic() -> None:
    """Test basic megapixelsteps calculation."""
    job = create_mock_job(width=512, height=512, ddim_steps=20, n_iter=1)
    result = get_single_job_effective_megapixelsteps(job)

    # 512 * 512 * 20 = 5,242,880 pixels * steps
    # Divided by 1,000,000 = 5 megapixelsteps
    assert result == 5


def test_get_single_job_effective_megapixelsteps_with_batching() -> None:
    """Test megapixelsteps calculation with batching multiplier."""
    job = create_mock_job(width=512, height=512, ddim_steps=20, n_iter=3)
    result = get_single_job_effective_megapixelsteps(job)

    # Base: 512 * 512 * 20 = 5,242,880
    # With n_iter=3: batching_multiplier = 1 + ((3-1) * 0.2) = 1.4
    # 5,242,880 * 1.4 = 7,339,992 / 1,000,000 = 7 megapixelsteps
    assert result == 7


def test_get_single_job_effective_megapixelsteps_with_upscaler() -> None:
    """Test megapixelsteps calculation with upscaler."""
    # Get a known upscaler from the SDK
    upscaler_value = list(KNOWN_UPSCALERS)[0].value
    job = create_mock_job(
        width=512,
        height=512,
        ddim_steps=20,
        n_iter=1,
        post_processing=[upscaler_value],
    )
    result = get_single_job_effective_megapixelsteps(job)

    # Base: 512 * 512 * 20 = 5,242,880
    # Upscaling adjustment: 512 * 512 * 20 * 1 * 1 = 5,242,880
    # Total: 5,242,880 + 5,242,880 = 10,485,760 / 1,000,000 = 10 megapixelsteps
    assert result == 10


def test_get_single_job_effective_megapixelsteps_with_loras() -> None:
    """Test megapixelsteps calculation with LoRAs."""
    job = create_mock_job(width=512, height=512, ddim_steps=20, n_iter=1, loras=[{"name": "test_lora"}])
    result = get_single_job_effective_megapixelsteps(job)

    # Base: 512 * 512 * 20 = 5,242,880
    # LoRA adjustment: 4,000,000
    # Total: 5,242,880 + 4,000,000 = 9,242,880 / 1,000,000 = 9 megapixelsteps
    assert result == 9


def test_get_single_job_effective_megapixelsteps_with_hires_fix() -> None:
    """Test megapixelsteps calculation with hires fix."""
    job = create_mock_job(width=512, height=512, ddim_steps=20, n_iter=1, hires_fix=True)
    result = get_single_job_effective_megapixelsteps(job)

    # Base: 512 * 512 * 20 = 5,242,880
    # Hires fix: 512 * 512 * 20 = 5,242,880
    # Total: 5,242,880 + 5,242,880 = 10,485,760 / 1,000,000 = 10 megapixelsteps
    assert result == 10


def test_get_single_job_effective_megapixelsteps_with_slow_model() -> None:
    """Test megapixelsteps calculation with slow model multiplier."""
    # Get a known slow model if available
    if KNOWN_SLOW_MODELS_DIFFICULTIES:
        slow_model = list(KNOWN_SLOW_MODELS_DIFFICULTIES.keys())[0]
        multiplier = KNOWN_SLOW_MODELS_DIFFICULTIES[slow_model]

        job = create_mock_job(width=512, height=512, ddim_steps=20, n_iter=1, model=slow_model)
        result = get_single_job_effective_megapixelsteps(job)

        # Base: 512 * 512 * 20 = 5,242,880
        # Multiplied by model difficulty
        expected = int((5_242_880 * multiplier) / 1_000_000)
        assert result == expected
    else:
        pytest.skip("No slow models defined in KNOWN_SLOW_MODELS_DIFFICULTIES")


def test_get_single_job_effective_megapixelsteps_with_slow_workflow() -> None:
    """Test megapixelsteps calculation with slow workflow."""
    if KNOWN_SLOW_WORKFLOWS:
        slow_workflow = list(KNOWN_SLOW_WORKFLOWS.keys())[0]
        multiplier = KNOWN_SLOW_WORKFLOWS[slow_workflow]

        job = create_mock_job(width=512, height=512, ddim_steps=20, n_iter=1, workflow=slow_workflow)
        result = get_single_job_effective_megapixelsteps(job)

        # Base: 512 * 512 * 20 = 5,242,880
        # Multiplied by workflow difficulty
        expected = int((5_242_880 * multiplier) / 1_000_000)
        assert result == expected
    else:
        pytest.skip("No slow workflows defined in KNOWN_SLOW_WORKFLOWS")


def test_get_single_job_effective_megapixelsteps_with_controlnet_workflow() -> None:
    """Test megapixelsteps calculation with controlnet workflow."""
    if KNOWN_CONTROLNET_WORKFLOWS:
        controlnet_workflow = list(KNOWN_CONTROLNET_WORKFLOWS.keys())[0]

        job = create_mock_job(width=512, height=512, ddim_steps=20, n_iter=1, workflow=controlnet_workflow)
        result = get_single_job_effective_megapixelsteps(job)

        # Base: 512 * 512 * 20 = 5,242,880
        # Multiplied by 2 for controlnet
        expected = int((5_242_880 * 2) / 1_000_000)
        assert result == expected
    else:
        pytest.skip("No controlnet workflows defined in KNOWN_CONTROLNET_WORKFLOWS")


def test_get_single_job_effective_megapixelsteps_large_resolution() -> None:
    """Test megapixelsteps calculation with large resolution."""
    job = create_mock_job(width=1024, height=1024, ddim_steps=30, n_iter=1)
    result = get_single_job_effective_megapixelsteps(job)

    # 1024 * 1024 * 30 = 31,457,280 / 1,000,000 = 31 megapixelsteps
    assert result == 31


def test_get_single_job_effective_megapixelsteps_complex_job() -> None:
    """Test megapixelsteps calculation with multiple factors."""
    # Get an upscaler if available
    if KNOWN_UPSCALERS:
        upscaler_value = list(KNOWN_UPSCALERS)[0].value
    else:
        upscaler_value = None

    post_processing = [upscaler_value] if upscaler_value else []

    job = create_mock_job(
        width=768,
        height=768,
        ddim_steps=25,
        n_iter=2,
        post_processing=post_processing,
        loras=[{"name": "lora1"}],
        hires_fix=True,
    )
    result = get_single_job_effective_megapixelsteps(job)

    # This should be a higher value due to all the factors
    assert result > 20  # At least 20 megapixelsteps with all these factors
