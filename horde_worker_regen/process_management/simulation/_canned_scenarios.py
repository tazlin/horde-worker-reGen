"""Predetermined job scenarios for dry-run and benchmark modes."""

from __future__ import annotations

import base64
import functools
import random
import time
import uuid
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopPayload,
    ImageGenerateJobPopRequest,
    ImageGenerateJobPopResponse,
    ImageGenerateJobPopSkippedStatus,
    LorasPayloadEntry,
    TIPayloadEntry,
)
from horde_sdk.ai_horde_api.fields import GenerationID

from horde_worker_regen.consts import EXTENDED_CONTROL_TYPES, KNOWN_CONTROLNET_WORKFLOWS
from horde_worker_regen.process_management.ipc.messages import AlchemyFormSpec
from horde_worker_regen.process_management.simulation._dummy_images import make_dummy_source_png_bytes
from horde_worker_regen.process_management.simulation._dummy_jobs import DUMMY_R2_UPLOAD_URL, dummy_job_factory

_IMG2IMG_SOURCE_PROCESSING = frozenset({"img2img", "remix"})
"""Source-processing modes whose jobs carry a start image (and so drive the VAE-encode lane)."""


def _source_image_seed(seed: str | None) -> int:
    """Derive a stable integer image seed from a job's (string) generation seed.

    Numeric seeds (what the canned/gate builders mint) map straight through; any other string falls back
    to a CRC so the derivation stays deterministic across processes without depending on ``hash`` salting.
    """
    if seed is None:
        return 0
    try:
        return int(seed) & 0xFFFFFFFF
    except ValueError:
        return zlib.crc32(seed.encode("utf-8"))


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

    def next_pop_response(self, pop_request: ImageGenerateJobPopRequest | None = None) -> ImageGenerateJobPopResponse:
        """Return the next canned job, or a no-job-available response once exhausted.

        ``pop_request`` is the request the popper would have sent to the live API. A fixed-list source
        replays a scripted scenario verbatim, so it ignores the request's shaping; generating sources
        honor it so pop-side governors (LoRA share caps, feature withholding) shape simulated traffic
        exactly as they shape live traffic.
        """
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
def make_source_image_bytes(width: int = 512, height: int = 512) -> bytes:
    """Return a synthetic source image (PNG bytes) suitable for img2img/controlnet/alchemy.

    Uses PIL when available (a gradient with enough structure for annotators to find
    edges); falls back to a 1x1 dummy PNG in environments without PIL.
    """
    try:
        import io

        import PIL.Image
    except ImportError:
        from horde_worker_regen.process_management.simulation._dummy_images import make_dummy_png_bytes

        return make_dummy_png_bytes()

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
    return buffer.getvalue()


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
    return_control_map: bool = False,
    image_is_control: bool = False,
    workflow: str | None = None,
    post_processing: list[str] | None = None,
    hires_fix: bool = False,
    cfg_scale: float | None = None,
    source_image_base64: str | None = None,
    source_processing: str | None = None,
    seed: str | None = None,
    prompt: str | None = None,
) -> ImageGenerateJobPopResponse:
    """Create a single canned job with configurable size, steps, batch amount, and features.

    Batched jobs (``n_iter > 1``) get one generation ID and R2 upload slot per image,
    as the live API provides. A controlnet job, or a workflow job (e.g. ``qr_code``), without
    an explicit source image gets a synthetic one automatically, since both consume a control
    image.

    ``return_control_map`` requests the derived control map as the deliverable in place of a
    generation, and ``image_is_control`` marks the source image as already being a control map
    (which suppresses pre-annotation). Both mirror the like-named ``ImageGenerateJobPopPayload``
    fields the image-utilities routing predicate reads.

    ``seed`` and ``prompt`` override the factory's pinned defaults so a caller can build a fixed,
    deterministic job list for A/B soak parity; left None, the historical pins are preserved.
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
    if cfg_scale is not None:
        data["payload"]["cfg_scale"] = cfg_scale
    if seed is not None:
        data["payload"]["seed"] = seed
    if prompt is not None:
        data["payload"]["prompt"] = prompt

    if loras is not None:
        data["payload"]["loras"] = [entry.model_dump(by_alias=True) for entry in loras]
    if tis is not None:
        data["payload"]["tis"] = [entry.model_dump(by_alias=True) for entry in tis]
    if post_processing is not None:
        data["payload"]["post_processing"] = post_processing
    if control_type is not None:
        data["payload"]["control_type"] = control_type
        if source_image_base64 is None:
            source_image_base64 = base64.b64encode(make_source_image_bytes()).decode("utf-8")
    if return_control_map:
        data["payload"]["return_control_map"] = True
    if image_is_control:
        data["payload"]["image_is_control"] = True
    if workflow is not None:
        data["payload"]["workflow"] = workflow
        if source_image_base64 is None:
            source_image_base64 = base64.b64encode(make_source_image_bytes()).decode("utf-8")
    # A plain img2img/remix job carries no control or workflow, so it would otherwise ship without a
    # start image and degrade to txt2img (never exercising the VAE-encode lane); attach a deterministic
    # one at the job's own resolution so the encode cost is real. Content is seed-derived and byte-stable.
    if source_processing in _IMG2IMG_SOURCE_PROCESSING and source_image_base64 is None:
        source_image_base64 = base64.b64encode(
            make_dummy_source_png_bytes(width, height, _source_image_seed(seed)),
        ).decode("utf-8")
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

    Names not already in the local cache trigger ad-hoc CivitAI downloads in real mode;
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
    source_image_bytes: bytes | None = None,
) -> list[AlchemyFormSpec]:
    """Create alchemy form specs cycling over the given form names (e.g. caption, RealESRGAN_x4plus)."""
    if not forms:
        raise ValueError("forms must not be empty")
    image = source_image_bytes if source_image_bytes is not None else make_source_image_bytes()
    return [
        AlchemyFormSpec(
            form_id=str(uuid.uuid4()),
            form=forms[i % len(forms)],
            source_image_bytes=image,
            r2_upload=DUMMY_R2_UPLOAD_URL,
        )
        for i in range(num_forms)
    ]


# ---------------------------------------------------------------------------
# Sustained-load (soak) generation
# ---------------------------------------------------------------------------


@dataclass
class SoakImageTemplate:
    """A weighted image-job template for sustained-load (soak) generation.

    Unlike a fixed scenario list, a soak source mints a *fresh* job (new generation IDs)
    from one of these templates on every pop, so it can feed the worker indefinitely
    without the ID collisions that recycling a fixed list would cause.
    """

    model: str = "Deliberate"
    width: int = 512
    height: int = 512
    steps: int = 30
    n_iter: int = 1
    seed: str | None = None
    """Fixed seed every job minted from this template carries, or None to keep the factory's pinned default."""
    prompt: str | None = None
    """Fixed prompt every job minted from this template carries, or None to keep the factory's pinned default."""
    source_processing: str | None = None
    """Source-processing mode (e.g. ``img2img``) every minted job carries, or None to keep txt2img default."""
    control_type: str | None = None
    return_control_map: bool = False
    image_is_control: bool = False
    workflow: str | None = None
    post_processing: list[str] = field(default_factory=list)
    hires_fix: bool = False
    cfg_scale: float | None = None
    loras: list[LorasPayloadEntry] = field(default_factory=list)
    """LoRA references every job minted from this template carries.

    The references are fixed on the template, so every job it mints requests the same files: jobs sharing
    a template (or sharing a reference across templates) resolve to the same on-disk LoRA, which is what
    lets a soak exercise the cache-reuse and download-dedup paths that pop-time prefetch depends on."""
    tis: list[TIPayloadEntry] = field(default_factory=list)
    """Textual-inversion references every job minted from this template carries (symmetric to ``loras``)."""
    weight: float = 1.0
    """Relative likelihood of this template being chosen on each pop."""


class GeneratingJobSource(CannedJobSource):
    """A never-exhausting job source that mints fresh jobs from weighted templates.

    Used by the sustained-load validation phase: it keeps the worker saturated for the
    whole soak period, picking a template by weight and generating a new job (with unique
    generation IDs) on each pop. Deterministic for a given seed.
    """

    def __init__(self, templates: list[SoakImageTemplate], *, seed: int = 0) -> None:
        """Initialise with the weighted templates to generate from."""
        super().__init__([], cycle=False)
        if not templates:
            raise ValueError("GeneratingJobSource requires at least one template")
        self._templates = list(templates)
        self._weights = [max(template.weight, 0.0) for template in templates]
        if sum(self._weights) <= 0:
            self._weights = [1.0] * len(templates)
        self._rng = random.Random(seed)
        self._stopped = False

    def stop(self) -> None:
        """Stop generating new jobs so the worker can drain what it has already accepted."""
        self._stopped = True

    @property
    def exhausted(self) -> bool:
        """False while generating; True once stopped, so the popper drains and idles."""
        return self._stopped

    @property
    def total_jobs(self) -> int | None:
        """Unbounded; the soak watcher decides when to stop."""
        return None

    @staticmethod
    def _template_matches_pop_request(
        template: SoakImageTemplate,
        pop_request: ImageGenerateJobPopRequest,
    ) -> bool:
        """Return whether the simulated Horde may assign ``template`` for this pop request.

        The generating source can only enforce requirements represented by :class:`SoakImageTemplate`.
        Request properties requiring synthetic metadata or server-side knowledge (prompt blacklist, requester
        kudos/IP, NSFW intent, painting mode, and model-average step limits) remain outside simulation;
        fixed-list sources deliberately replay their scripts verbatim.
        """
        if template.model not in pop_request.models:
            return False
        if template.width * template.height > pop_request.max_pixels:
            return False
        if (template.loras or template.tis) and not pop_request.allow_lora:
            return False
        if template.post_processing and not pop_request.allow_post_processing:
            return False

        needs_source_image = template.control_type is not None or template.workflow is not None
        needs_controlnet = template.control_type is not None or template.workflow in KNOWN_CONTROLNET_WORKFLOWS
        if needs_controlnet and not pop_request.allow_controlnet:
            return False
        if template.control_type in EXTENDED_CONTROL_TYPES and not pop_request.allow_extended_controlnet:
            return False
        if template.workflow in KNOWN_CONTROLNET_WORKFLOWS and not pop_request.allow_sdxl_controlnet:
            return False

        # make_canned_job supplies a source image for ControlNet and every workflow template. The worker-side
        # eligibility contract treats every source-image job as img2img-class in addition to any feature gate.
        return not (needs_source_image and not pop_request.allow_img2img)

    def _eligible_templates(
        self,
        pop_request: ImageGenerateJobPopRequest | None,
    ) -> tuple[list[SoakImageTemplate], list[float]]:
        """Return templates and normalized draw weights eligible for ``pop_request``."""
        if pop_request is None:
            return self._templates, self._weights

        eligible = [
            (template, weight)
            for template, weight in zip(self._templates, self._weights, strict=True)
            if self._template_matches_pop_request(template, pop_request)
        ]
        templates = [template for template, _weight in eligible]
        weights = [weight for _template, weight in eligible]
        if templates and sum(weights) <= 0:
            weights = [1.0] * len(templates)
        return templates, weights

    def next_pop_response(self, pop_request: ImageGenerateJobPopRequest | None = None) -> ImageGenerateJobPopResponse:
        """Generate a fresh job from a weighted-random template (or no-job once stopped).

        The popper's request shaping is honored the way the live API honors it: ineligible templates are
        excluded and their weight redistributes over the eligible templates. A mix with nothing suitable
        reports no job available. This keeps pop-side governors (including LoRA intake caps, feature
        withholding, multi-GPU targeting, and the idle-fill model/size ladder) load-bearing in simulation.
        """
        if self._stopped:
            return make_empty_pop_response()
        templates, weights = self._eligible_templates(pop_request)
        if not templates:
            return make_empty_pop_response()
        template = self._rng.choices(templates, weights=weights, k=1)[0]
        return make_canned_job(
            template.model,
            width=template.width,
            height=template.height,
            ddim_steps=template.steps,
            n_iter=template.n_iter,
            control_type=template.control_type,
            return_control_map=template.return_control_map,
            image_is_control=template.image_is_control,
            workflow=template.workflow,
            post_processing=template.post_processing or None,
            hires_fix=template.hires_fix,
            cfg_scale=template.cfg_scale,
            loras=template.loras or None,
            tis=template.tis or None,
            source_processing=template.source_processing,
            seed=template.seed,
            prompt=template.prompt,
        )


SoakAlchemyForm = tuple[str, float] | tuple[str, float, str | None]
"""A weighted alchemy-form entry for the soak: ``(form, weight)`` or ``(form, weight, control_type)``.

The optional third element is the control type an ``annotation`` form carries (canny, depth, teed,
...); it is ``None`` (or omitted) for every other form, which needs no control type."""


class GeneratingAlchemySource(CannedAlchemySource):
    """A never-exhausting alchemy source that mints fresh forms from weighted form names.

    An ``annotation`` form entry carries the control type to annotate, so the minted spec routes to the
    image-utilities lane with the control type that lane needs (a control-type-less annotation form
    faults there). Every other form leaves the control type ``None``.
    """

    def __init__(self, form_weights: Sequence[SoakAlchemyForm], *, seed: int = 0) -> None:
        """Initialise with ``(form, weight)`` or ``(form, weight, control_type)`` entries to generate from."""
        super().__init__([])
        if not form_weights:
            raise ValueError("GeneratingAlchemySource requires at least one form")
        self._form_names = [entry[0] for entry in form_weights]
        self._form_control_types = [entry[2] if len(entry) > 2 else None for entry in form_weights]
        self._form_weights = [max(entry[1], 0.0) for entry in form_weights]
        if sum(self._form_weights) <= 0:
            self._form_weights = [1.0] * len(self._form_names)
        self._rng = random.Random(seed)
        self._source_image = make_source_image_bytes()
        self._stopped = False

    def stop(self) -> None:
        """Stop generating new forms so in-flight alchemy can drain."""
        self._stopped = True

    @property
    def exhausted(self) -> bool:
        """False while generating; True once stopped."""
        return self._stopped

    @property
    def total_forms(self) -> int:
        """Unbounded; the soak watcher decides when to stop."""
        return 0

    def next_form(self) -> AlchemyFormSpec | None:
        """Generate a fresh alchemy form from a weighted-random form name (or None once stopped)."""
        if self._stopped:
            return None
        index = self._rng.choices(range(len(self._form_names)), weights=self._form_weights, k=1)[0]
        return AlchemyFormSpec(
            form_id=str(uuid.uuid4()),
            form=self._form_names[index],
            source_image_bytes=self._source_image,
            r2_upload=DUMMY_R2_UPLOAD_URL,
            control_type=self._form_control_types[index],
        )


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
    "arrived"; the popper needs no changes since it already handles empty pops.
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

    def next_pop_response(self, pop_request: ImageGenerateJobPopRequest | None = None) -> ImageGenerateJobPopResponse:
        """Return the next job if its arrival time has passed, else a no-job response."""
        if self._start_time is None:
            self._start_time = self._clock()

        if self.exhausted or len(self._jobs) == 0:
            return make_empty_pop_response()

        elapsed = self._clock() - self._start_time
        if elapsed < self._schedule.release_time_offset(self._next_index):
            return make_empty_pop_response()

        return super().next_pop_response(pop_request)


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
