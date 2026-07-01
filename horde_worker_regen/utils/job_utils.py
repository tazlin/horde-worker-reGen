"""Job processing utility functions."""

from __future__ import annotations

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.generation_parameters import KNOWN_UPSCALERS

from horde_worker_regen.consts import (
    KNOWN_CONTROLNET_WORKFLOWS,
    KNOWN_SLOW_MODELS_DIFFICULTIES,
    KNOWN_SLOW_WORKFLOWS,
)


def get_single_job_magnitude(job: ImageGenerateJobPopResponse) -> int:
    """Return an approximate magnitude of a single job based on megapixelsteps and other factors.

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

    workflow_name = job.payload.workflow

    # We treat slow workflows add extra slowdowns (as they might perform many more steps of inference)
    slow_multiplier = KNOWN_SLOW_WORKFLOWS.get(workflow_name) if workflow_name else None
    if slow_multiplier:
        job_effective_pixel_steps *= slow_multiplier

    # Some workflows by default require controlnets, but the user doesn't have to specify them.
    # In this case, we use this to know when we have SDXL workflows, as they can double the VRAM usage
    controlnet_multiplier = KNOWN_CONTROLNET_WORKFLOWS.get(workflow_name) if workflow_name else None
    if controlnet_multiplier and not slow_multiplier:
        job_effective_pixel_steps *= controlnet_multiplier
    return int(job_effective_pixel_steps / 1_000_000)


def line_skip_candidate_emps_limit(
    *,
    high_performance_mode: bool,
    moderate_performance_mode: bool,
) -> int:
    """Return the eMPS ceiling a job must fall under to line-skip a slot blocked on aux downloads.

    A line-skip job fills an idle sibling process while another slot waits on an auxiliary-model
    download, so it must be quick enough that it does not become a long-running tenant of its own. The
    ceiling widens with the worker's performance mode, matching the job sizes those modes are provisioned
    to absorb.
    """
    if high_performance_mode:
        return 100
    if moderate_performance_mode:
        return 50
    return 25


_LINE_SKIP_POP_NOMINAL_STEPS = 30
"""Sampling-step count assumed when translating the line-skip eMPS ceiling into a pop ``max_power``.

Real jobs vary in step count, so this only biases the horde toward returning a skippable job; it is not a
guarantee. The scheduler re-checks each popped job's true eMPS against
:func:`line_skip_candidate_emps_limit` before allowing it to skip, so a returned job that is larger than
expected simply waits its turn in the queue instead of skipping."""


def line_skip_pop_max_power(
    *,
    high_performance_mode: bool,
    moderate_performance_mode: bool,
) -> int:
    """Return a ``max_power`` that biases a pop toward a job able to line-skip an aux-download-blocked slot.

    The horde honours ``max_pixels`` (``max_power * 8 * 64 * 64``) as a hard cap on the resolution of
    returned jobs. Translating the perf-mode eMPS ceiling into an approximate resolution ceiling (assuming
    a nominal step count) keeps the returned job small enough to slip past the blocked head and onto an
    idle sibling process.
    """
    emps_limit = line_skip_candidate_emps_limit(
        high_performance_mode=high_performance_mode,
        moderate_performance_mode=moderate_performance_mode,
    )
    max_pixels = emps_limit * 1_000_000 // _LINE_SKIP_POP_NOMINAL_STEPS
    return max(1, max_pixels // (8 * 64 * 64))
