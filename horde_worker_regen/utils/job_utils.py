"""Job processing utility functions."""

from __future__ import annotations

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.ai_horde_api.consts import KNOWN_UPSCALERS

from horde_worker_regen.consts import (
    KNOWN_CONTROLNET_WORKFLOWS,
    KNOWN_SLOW_MODELS_DIFFICULTIES,
    KNOWN_SLOW_WORKFLOWS,
)


def get_single_job_effective_megapixelsteps(job: ImageGenerateJobPopResponse) -> int:
    """Return the number of megapixelsteps for a single job.

    Args:
        job: The job to get the number of megapixelsteps for.

    Returns:
        The number of effective megapixelsteps for the job.
    """
    has_upscaler = any(pp in [u.value for u in KNOWN_UPSCALERS] for pp in job.payload.post_processing)
    upscaler_multiplier = 1 if has_upscaler else 0
    job_pixels = job.payload.width * job.payload.height

    # Each extra batched image increases our difficulty by 20%
    batching_multiplier = 1 + ((job.payload.n_iter - 1) * 0.2)

    lora_adjustment = 0
    if job.payload.loras is not None:
        lora_adjustment = 4 * 1_000_000 if len(job.payload.loras) > 0 else 0

    hires_fix_adjustment = 0

    if job.payload.hires_fix:
        hires_fix_adjustment = 512 * 512 * job.payload.ddim_steps

    # If upscaling was requested, due to it being serial, each extra image in the batch
    # Further increases our difficulty.
    # In this calculation we treat each upscaler as adding 20 steps per image
    upscaling_adjustment = job_pixels * 20 * upscaler_multiplier * job.payload.n_iter
    job_effective_pixel_steps = (
        (job_pixels * batching_multiplier * job.payload.ddim_steps)
        + upscaling_adjustment
        + lora_adjustment
        + hires_fix_adjustment
    )

    # Hard model difficulty is increased due to variations in the performance
    # of different architectures. This look up is a rough estimate based on a median case
    if job.model in KNOWN_SLOW_MODELS_DIFFICULTIES:
        job_effective_pixel_steps *= KNOWN_SLOW_MODELS_DIFFICULTIES[job.model]

    # We treat slow workflows add extra slowdowns (as they might perform many more steps of inference)
    if job.payload.workflow in KNOWN_SLOW_WORKFLOWS:
        job_effective_pixel_steps *= KNOWN_SLOW_WORKFLOWS[job.payload.workflow]

    # Some workflows by default require controlnets, but the user doesn't have to specify them.
    # In this case, we use this to know when we have SDXL workflows, as they can double the VRAM usage
    if job.payload.workflow in KNOWN_CONTROLNET_WORKFLOWS:
        job_effective_pixel_steps *= 2
    return int(job_effective_pixel_steps / 1_000_000)
