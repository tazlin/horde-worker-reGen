"""Predetermined job scenarios for dry-run and benchmark modes."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management._dummy_jobs import dummy_job_factory


@dataclass
class BenchModelConfig:
    """Describes a model tier used for benchmarking."""

    model: str
    resolutions: list[tuple[int, int]]


# ---------------------------------------------------------------------------
# Dry-run scenarios (no GPU needed)
# ---------------------------------------------------------------------------

SCENARIO_TRIVIAL: list[ImageGenerateJobPopResponse] = [dummy_job_factory("Deliberate")]

SCENARIO_BASIC: list[ImageGenerateJobPopResponse] = [
    dummy_job_factory("Deliberate"),
    dummy_job_factory("Deliberate"),
    dummy_job_factory("Deliberate"),
    dummy_job_factory("Deliberate"),
    dummy_job_factory("Deliberate"),
]

# ---------------------------------------------------------------------------
# Benchmark model tiers (real inference, well-known models)
# ---------------------------------------------------------------------------

BENCH_MODELS: dict[str, BenchModelConfig] = {
    "sd15": BenchModelConfig(model="Deliberate", resolutions=[(512, 512), (768, 768)]),
    "sdxl": BenchModelConfig(model="AlbedoBase XL (SDXL)", resolutions=[(1024, 1024)]),
    "flux": BenchModelConfig(model="FLUX.1 [schnell]", resolutions=[(1024, 1024)]),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_dry_run_cycle = itertools.cycle(SCENARIO_BASIC)


def get_dry_run_job() -> ImageGenerateJobPopResponse:
    """Return the next canned job for dry-run API bypass."""
    return next(_dry_run_cycle)
