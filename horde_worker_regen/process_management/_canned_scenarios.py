"""Predetermined job scenarios for dry-run and benchmark modes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopPayload,
    ImageGenerateJobPopResponse,
    ImageGenerateJobPopSkippedStatus,
)
from horde_sdk.ai_horde_api.fields import GenerationID

from horde_worker_regen.process_management._dummy_jobs import DUMMY_R2_UPLOAD_URL, dummy_job_factory


@dataclass
class BenchModelConfig:
    """Describes a model tier used for benchmarking."""

    model: str
    resolutions: list[tuple[int, int]]


class CannedJobSource:
    """Hands out predetermined jobs in place of real API pops.

    A source can either cycle forever (the historical dry-run behavior) or run
    through its job list once and then report exhaustion, which lets a harness
    run a bounded scenario to completion.
    """

    _jobs: list[ImageGenerateJobPopResponse]
    _cycle: bool
    _next_index: int

    def __init__(
        self,
        jobs: list[ImageGenerateJobPopResponse],
        *,
        cycle: bool = False,
    ) -> None:
        """Initialise the source.

        Args:
            jobs (list[ImageGenerateJobPopResponse]): The jobs to hand out, in order.
            cycle (bool, optional): If true, restart from the beginning when the list is \
                exhausted instead of stopping. Defaults to False.
        """
        self._jobs = list(jobs)
        self._cycle = cycle
        self._next_index = 0

    @property
    def exhausted(self) -> bool:
        """Whether all jobs have been handed out (always False for cycling sources)."""
        if self._cycle:
            return False
        return self._next_index >= len(self._jobs)

    @property
    def total_jobs(self) -> int | None:
        """The number of jobs this source will hand out, or None if it cycles forever."""
        if self._cycle:
            return None
        return len(self._jobs)

    def next_pop_response(self) -> ImageGenerateJobPopResponse:
        """Return the next canned job, or a no-job-available response once exhausted."""
        if len(self._jobs) == 0 or self.exhausted:
            return make_empty_pop_response()

        job = self._jobs[self._next_index % len(self._jobs)]
        self._next_index += 1
        return job


def make_empty_pop_response() -> ImageGenerateJobPopResponse:
    """Return a pop response indicating no jobs are available, as the live API would."""
    return ImageGenerateJobPopResponse(
        id=None,
        ids=[],
        skipped=ImageGenerateJobPopSkippedStatus(),
        payload=ImageGenerateJobPopPayload(),
    )


def make_simple_scenario(
    num_jobs: int,
    *,
    model_name: str = "Deliberate",
) -> list[ImageGenerateJobPopResponse]:
    """Create a scenario of identical txt2img jobs for the given model."""
    return [dummy_job_factory(model_name) for _ in range(num_jobs)]


def make_canned_job(
    model_name: str = "Deliberate",
    *,
    width: int = 512,
    height: int = 512,
    ddim_steps: int = 30,
    n_iter: int = 1,
) -> ImageGenerateJobPopResponse:
    """Create a single canned job with configurable size, steps, and batch amount.

    Batched jobs (``n_iter > 1``) get one generation ID and R2 upload slot per image,
    as the live API provides.
    """
    job = dummy_job_factory(model_name)
    data = job.model_dump(by_alias=True)

    ids = [GenerationID(root=uuid.uuid4()) for _ in range(n_iter)]
    data["id"] = ids[0]
    data["ids"] = ids
    data["r2_uploads"] = [DUMMY_R2_UPLOAD_URL] * n_iter

    data["payload"]["width"] = width
    data["payload"]["height"] = height
    data["payload"]["ddim_steps"] = ddim_steps
    data["payload"]["n_iter"] = n_iter

    return ImageGenerateJobPopResponse(**data)


def make_mixed_model_scenario(
    num_jobs: int,
    model_names: list[str],
) -> list[ImageGenerateJobPopResponse]:
    """Create a scenario that alternates between the given models round-robin.

    Forces the scheduler to preload, swap, and unload models between jobs.
    """
    if not model_names:
        raise ValueError("model_names must not be empty")
    return [make_canned_job(model_names[i % len(model_names)]) for i in range(num_jobs)]


def make_batch_scenario(
    num_jobs: int,
    batch_size: int,
    *,
    model_name: str = "Deliberate",
) -> list[ImageGenerateJobPopResponse]:
    """Create a scenario of batched jobs (multiple images per job)."""
    return [make_canned_job(model_name, n_iter=batch_size) for _ in range(num_jobs)]


def make_varied_size_scenario(
    num_jobs: int,
    *,
    model_name: str = "Deliberate",
) -> list[ImageGenerateJobPopResponse]:
    """Create a scenario mixing small and large jobs to exercise megapixelstep backpressure."""
    sizes = [(512, 512, 20), (1024, 1024, 50), (768, 768, 30)]
    return [
        make_canned_job(model_name, width=w, height=h, ddim_steps=steps)
        for w, h, steps in (sizes[i % len(sizes)] for i in range(num_jobs))
    ]


# ---------------------------------------------------------------------------
# Dry-run scenarios (no GPU needed)
# ---------------------------------------------------------------------------

SCENARIO_TRIVIAL: list[ImageGenerateJobPopResponse] = make_simple_scenario(1)

SCENARIO_BASIC: list[ImageGenerateJobPopResponse] = make_simple_scenario(5)

# ---------------------------------------------------------------------------
# Benchmark model tiers (real inference, well-known models)
# ---------------------------------------------------------------------------

BENCH_MODELS: dict[str, BenchModelConfig] = {
    "sd15": BenchModelConfig(model="Deliberate", resolutions=[(512, 512), (768, 768)]),
    "sdxl": BenchModelConfig(model="AlbedoBase XL (SDXL)", resolutions=[(1024, 1024)]),
    "flux": BenchModelConfig(model="FLUX.1 [schnell]", resolutions=[(1024, 1024)]),
}


def make_default_dry_run_source() -> CannedJobSource:
    """Return the default endlessly-cycling dry-run job source."""
    return CannedJobSource(SCENARIO_BASIC, cycle=True)
