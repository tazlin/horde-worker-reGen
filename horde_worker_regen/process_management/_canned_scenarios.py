"""Predetermined job scenarios for dry-run and benchmark modes."""

from __future__ import annotations

import base64
import functools
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopPayload,
    ImageGenerateJobPopResponse,
    ImageGenerateJobPopSkippedStatus,
    LorasPayloadEntry,
    TIPayloadEntry,
)
from horde_sdk.ai_horde_api.fields import GenerationID

from horde_worker_regen.process_management._dummy_jobs import DUMMY_R2_UPLOAD_URL, dummy_job_factory
from horde_worker_regen.process_management.messages import AlchemyFormSpec


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


@functools.cache
def make_source_image_base64(width: int = 512, height: int = 512) -> str:
    """Return a synthetic source image (PNG base64) suitable for img2img/controlnet/alchemy.

    Uses PIL when available (a gradient with enough structure for annotators to find
    edges); falls back to a 1x1 dummy PNG in environments without PIL.
    """
    try:
        import io

        import PIL.Image
    except ImportError:
        from horde_worker_regen.process_management._dummy_images import make_dummy_png_base64

        return make_dummy_png_base64()

    image = PIL.Image.new("RGB", (width, height))
    pixels = image.load()
    assert pixels is not None
    for y in range(height):
        for x in range(width):
            # Diagonal bands give annotators (canny/hed/depth) real edges to detect.
            band = 255 if ((x + y) // 64) % 2 == 0 else 40
            pixels[x, y] = (band, (x * 255) // width, (y * 255) // height)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_canned_job(
    model_name: str = "Deliberate",
    *,
    width: int = 512,
    height: int = 512,
    ddim_steps: int = 30,
    n_iter: int = 1,
    loras: list[LorasPayloadEntry] | None = None,
    tis: list[TIPayloadEntry] | None = None,
    control_type: str | None = None,
    post_processing: list[str] | None = None,
    hires_fix: bool = False,
    source_image_base64: str | None = None,
    source_processing: str | None = None,
) -> ImageGenerateJobPopResponse:
    """Create a single canned job with configurable size, steps, batch amount, and features.

    Batched jobs (``n_iter > 1``) get one generation ID and R2 upload slot per image,
    as the live API provides. A controlnet job without an explicit source image gets a
    synthetic one automatically.
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
    data["payload"]["hires_fix"] = hires_fix

    if loras is not None:
        data["payload"]["loras"] = [entry.model_dump(by_alias=True) for entry in loras]
    if tis is not None:
        data["payload"]["tis"] = [entry.model_dump(by_alias=True) for entry in tis]
    if post_processing is not None:
        data["payload"]["post_processing"] = post_processing
    if control_type is not None:
        data["payload"]["control_type"] = control_type
        if source_image_base64 is None:
            source_image_base64 = make_source_image_base64()
    if source_image_base64 is not None:
        data["source_image"] = source_image_base64
        data["source_processing"] = source_processing if source_processing is not None else "img2img"
    elif source_processing is not None:
        data["source_processing"] = source_processing

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


def make_lora_scenario(
    num_jobs: int,
    lora_names: list[str],
    *,
    model_name: str = "Deliberate",
    is_version: bool = False,
) -> list[ImageGenerateJobPopResponse]:
    """Create jobs that each request the given loras (round-robin over the names).

    Names not already in the local cache trigger ad-hoc CivitAI downloads in real mode —
    that is the download-bandwidth dimension of the benchmark.
    """
    if not lora_names:
        raise ValueError("lora_names must not be empty")
    return [
        make_canned_job(
            model_name,
            loras=[LorasPayloadEntry(name=lora_names[i % len(lora_names)], is_version=is_version)],
        )
        for i in range(num_jobs)
    ]


def make_ti_scenario(
    num_jobs: int,
    ti_names: list[str],
    *,
    model_name: str = "Deliberate",
) -> list[ImageGenerateJobPopResponse]:
    """Create jobs that each request the given textual inversions (round-robin)."""
    if not ti_names:
        raise ValueError("ti_names must not be empty")
    return [
        make_canned_job(
            model_name,
            tis=[TIPayloadEntry(name=ti_names[i % len(ti_names)], inject_ti="prompt")],
        )
        for i in range(num_jobs)
    ]


def make_controlnet_scenario(
    num_jobs: int,
    *,
    model_name: str = "Deliberate",
    control_type: str = "canny",
) -> list[ImageGenerateJobPopResponse]:
    """Create controlnet jobs with a bundled synthetic source image."""
    return [make_canned_job(model_name, control_type=control_type) for _ in range(num_jobs)]


def make_post_processing_scenario(
    num_jobs: int,
    *,
    model_name: str = "Deliberate",
    post_processors: list[str] | None = None,
) -> list[ImageGenerateJobPopResponse]:
    """Create jobs with post-processors attached (default: one upscaler + one facefixer)."""
    if post_processors is None:
        post_processors = ["RealESRGAN_x4plus", "GFPGAN"]
    return [make_canned_job(model_name, post_processing=post_processors) for _ in range(num_jobs)]


def make_hires_fix_scenario(
    num_jobs: int,
    *,
    model_name: str = "Deliberate",
    width: int = 1024,
    height: int = 1024,
) -> list[ImageGenerateJobPopResponse]:
    """Create hires-fix jobs (two-pass sampling at the target resolution)."""
    return [make_canned_job(model_name, width=width, height=height, hires_fix=True) for _ in range(num_jobs)]


# ---------------------------------------------------------------------------
# Alchemy scenarios
# ---------------------------------------------------------------------------


class CannedAlchemySource:
    """Hands out predetermined alchemy forms in place of real API pops.

    The alchemy counterpart of :class:`CannedJobSource`; consumed by
    ``AlchemyCoordinator`` when the harness runs with the API faked out.
    """

    def __init__(self, forms: list[AlchemyFormSpec]) -> None:
        """Initialise the source with the forms to hand out, in order."""
        self._forms = list(forms)
        self._next_index = 0

    @property
    def exhausted(self) -> bool:
        """Whether all forms have been handed out."""
        return self._next_index >= len(self._forms)

    @property
    def total_forms(self) -> int:
        """The number of forms this source will hand out."""
        return len(self._forms)

    def next_form(self) -> AlchemyFormSpec | None:
        """Return the next canned form, or None once exhausted."""
        if self.exhausted:
            return None
        form = self._forms[self._next_index]
        self._next_index += 1
        return form


def make_alchemy_scenario(
    forms: list[str],
    num_forms: int,
    *,
    source_image_base64: str | None = None,
) -> list[AlchemyFormSpec]:
    """Create alchemy form specs cycling over the given form names (e.g. caption, RealESRGAN_x4plus)."""
    if not forms:
        raise ValueError("forms must not be empty")
    image = source_image_base64 if source_image_base64 is not None else make_source_image_base64()
    return [
        AlchemyFormSpec(
            form_id=str(uuid.uuid4()),
            form=forms[i % len(forms)],
            source_image_base64=image,
            r2_upload=DUMMY_R2_UPLOAD_URL,
        )
        for i in range(num_forms)
    ]


# ---------------------------------------------------------------------------
# Arrival-time control
# ---------------------------------------------------------------------------


@dataclass
class ArrivalSchedule:
    """Describes when canned jobs become available to pop, simulating queue structures."""

    kind: str = "all_at_once"
    """One of "all_at_once", "steady", or "bursts"."""
    rate_per_minute: float = 0.0
    """For "steady": how many jobs become available per minute."""
    burst_size: int = 0
    """For "bursts": how many jobs become available at once."""
    burst_interval_seconds: float = 0.0
    """For "bursts": how long between bursts."""

    def release_time_offset(self, job_index: int) -> float:
        """Return the seconds after start at which job *job_index* becomes available."""
        if self.kind == "steady" and self.rate_per_minute > 0:
            return job_index * (60.0 / self.rate_per_minute)
        if self.kind == "bursts" and self.burst_size > 0:
            return (job_index // self.burst_size) * self.burst_interval_seconds
        return 0.0


class TimedJobSource(CannedJobSource):
    """A canned job source that releases jobs on a schedule instead of all at once.

    Returns the no-job-available response until the schedule says the next job has
    "arrived" — the popper needs no changes since it already handles empty pops.
    """

    def __init__(
        self,
        jobs: list[ImageGenerateJobPopResponse],
        schedule: ArrivalSchedule,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialise the source.

        Args:
            jobs: The jobs to hand out, in order.
            schedule: When each job becomes available relative to the first pop attempt.
            clock: Override for ``time.monotonic`` (injectable for tests).
        """
        super().__init__(jobs, cycle=False)
        self._schedule = schedule
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._start_time: float | None = None

    def next_pop_response(self) -> ImageGenerateJobPopResponse:
        """Return the next job if its arrival time has passed, else a no-job response."""
        if self._start_time is None:
            self._start_time = self._clock()

        if self.exhausted or len(self._jobs) == 0:
            return make_empty_pop_response()

        elapsed = self._clock() - self._start_time
        if elapsed < self._schedule.release_time_offset(self._next_index):
            return make_empty_pop_response()

        return super().next_pop_response()


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
