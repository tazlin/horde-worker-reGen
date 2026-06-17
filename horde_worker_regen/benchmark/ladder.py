"""The default ramp ladder: ordered benchmark levels from conservative to demanding.

Ordering principle: prove each model tier viable at a conservative configuration first
(stage A, which also establishes the tier's reference sampling rate), then ramp
concurrency (B), then features (C), then alchemy (D), then ad-hoc downloads (E).
A stage-A failure skips the tier's dependent levels; a failure within one axis stops
further rungs of that axis only.

The matrix is grounded in what the worker and hordelib can actually do:

- **Controlnet** (canny/hed/depth/openpose preprocessors) is SD1.5-only in hordelib; every control
  type maps to an SD1.5 weight. The SDXL "controlnet" capability is really the **QR-code workflow**
  (``KNOWN_CONTROLNET_WORKFLOWS``), gated by ``allow_sdxl_controlnet``, so it lives on its own
  ``qr_code`` axis that runs on both SD1.5 and SDXL rather than a mislabeled SDXL controlnet level.
- **Post-processing** is exercised across every known upscaler and face-fixer (drawn from the SDK
  enums, not hardcoded), at representative resolutions up to a VRAM-derived maximum.
- **Alchemy** runs on two independent lanes: the CLIP lane (caption/interrogation/NSFW, on the safety
  process) and the graph lane (upscalers/face-fixers/strip-background, on the inference processes),
  each its own axis, plus a concurrent-with-image-jobs rung.
- **flux/qwen** are very large (17-20 GB download, 13-16 GB VRAM); they are opt-in tiers and the
  controller warns and pre-flight-skips them when the machine cannot hold them. qwen is sourced from
  the beta/pending reference (see :data:`BETA_TIERS`).
"""

from __future__ import annotations

from typing import Protocol

from horde_sdk.generation_parameters.alchemy.consts import (
    KNOWN_CLIP_BLIP_TYPES,
    KNOWN_FACEFIXERS,
    KNOWN_MISC_POST_PROCESSORS,
    KNOWN_UPSCALERS,
)
from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.criteria import LevelCriteria
from horde_worker_regen.benchmark.enums import BenchAxis, BenchStage, BenchTier
from horde_worker_regen.benchmark.scenarios import (
    CannedAlchemyFormSpec,
    CannedImageJobSpec,
    ScenarioSpec,
)
from horde_worker_regen.benchmark.sizing import max_post_processing_resolution

_QWEN_BETA_MODEL_NAME = "Qwen-Image"
"""Best-guess name for the beta qwen checkpoint, sourced via the pending reference (see BETA_TIERS).

No qwen model is in the canonical AI-Horde image reference yet; this is the expected release name.
Confirm against the PRIMARY pending queue once published and update if it differs. A wrong/absent
name simply pre-flight-skips the qwen tier rather than failing the run."""

BENCH_TIER_MODELS: dict[BenchTier, str] = {
    BenchTier.SD15: "Deliberate",
    BenchTier.SDXL: "AlbedoBase XL (SDXL)",
    BenchTier.FLUX: "Flux.1-Schnell fp8 (Compact)",
    BenchTier.QWEN: _QWEN_BETA_MODEL_NAME,
}

BENCH_TIER_MODEL_POOLS: dict[BenchTier, list[str]] = {
    # Distinct same-tier checkpoints for the multi-model soak. A single-model soak cannot
    # saturate >2 inference jobs (the popper caps in-flight jobs at 2 per model), so the soak
    # needs a pool of distinct models — one per inference process — to actually exercise every
    # process and the cross-process coordination. pool[0] MUST equal BENCH_TIER_MODELS[tier]
    # (the baseline/single-model paths use that name). All names must exist in the AI-Horde
    # image model reference; the controller trims the pool to what fits in VRAM.
    BenchTier.SD15: [
        "Deliberate",
        "Dreamshaper",
        "ICBINP - I Can't Believe It's Not Photography",
        "Anything Diffusion",
    ],
    BenchTier.SDXL: ["AlbedoBase XL (SDXL)"],
    BenchTier.FLUX: ["Flux.1-Schnell fp8 (Compact)"],
    BenchTier.QWEN: [_QWEN_BETA_MODEL_NAME],
}

_TIER_BASELINES: dict[BenchTier, str] = {
    BenchTier.SD15: "stable_diffusion_1",
    BenchTier.SDXL: "stable_diffusion_xl",
    BenchTier.FLUX: "flux_1",
    BenchTier.QWEN: "qwen_image",
}

_TIER_RESOLUTIONS: dict[BenchTier, int] = {
    BenchTier.SD15: 512,
    BenchTier.SDXL: 1024,
    BenchTier.FLUX: 1024,
    BenchTier.QWEN: 1024,
}

HUGE_TIERS: frozenset[BenchTier] = frozenset({BenchTier.FLUX, BenchTier.QWEN})
"""Tiers whose models are very large (17-20 GB); the controller warns and pre-flight-skips them."""

BETA_TIERS: frozenset[BenchTier] = frozenset({BenchTier.QWEN})
"""Tiers sourced from the beta/pending reference; the worker env opts into beta categories for these."""

_CONTROLNET_SWEEP_TYPES: tuple[str, ...] = ("canny", "depth", "openpose")
"""A representative spread of SD1.5 controlnet preprocessors (distinct annotator families)."""

_QR_CODE_WORKFLOW = "qr_code"
"""The hordelib workflow name for the QR-code controlnet (the real SDXL controlnet capability)."""


def _image_post_processor_names() -> list[str]:
    """Return every image post-processor name (upscalers + face-fixers), minus backend-default sentinels."""
    upscalers = [member.value for member in KNOWN_UPSCALERS if member is not KNOWN_UPSCALERS.BACKEND_DEFAULT]
    facefixers = [member.value for member in KNOWN_FACEFIXERS if member is not KNOWN_FACEFIXERS.BACKEND_DEFAULT]
    return upscalers + facefixers


def _clip_alchemy_form_names() -> list[str]:
    """Return the CLIP-lane alchemy forms (caption, interrogation, NSFW) that run on the safety process."""
    return [member.value for member in KNOWN_CLIP_BLIP_TYPES]


def _graph_alchemy_form_names(*, include_strip_background: bool) -> list[str]:
    """Return the graph-lane alchemy forms (every upscaler + face-fixer, plus strip-background if available)."""
    forms = _image_post_processor_names()
    if include_strip_background:
        forms += [member.value for member in KNOWN_MISC_POST_PROCESSORS]
    return forms


def _strip_background_available() -> bool:
    """Return whether background removal (rembg) is installed, defaulting to False if undetectable.

    Mirrors ``alchemy_popper.expand_offered_forms``: a lean install omits strip-background, so offering
    it would only fault. Best-effort and import-safe so fake/CI ladders build without rembg.
    """
    try:
        from horde_worker_regen.capabilities import strip_background_available

        return strip_background_available()
    except Exception:  # noqa: BLE001 - undetectable capability defaults to "absent", never breaks build
        return False


class RampLevel(BaseModel):
    """One rung of the ramp ladder."""

    id: str
    stage: BenchStage
    tier: BenchTier
    axis: BenchAxis
    """What this level ramps; failures stop higher rungs of the same axis."""
    rung: int = 0
    """Position within the axis; higher rungs are skipped after a failure on the axis."""
    description: str
    scenario: ScenarioSpec
    bridge_data_overrides: dict[str, object] = Field(default_factory=dict)
    requires_network: bool = False
    timeout_seconds: float = 900.0
    criteria: LevelCriteria = Field(default_factory=LevelCriteria)
    baseline_hordelib: str = ""
    """The KNOWN_IMAGE_GENERATION_BASELINE value, for pre-flight burden estimates."""
    establishes_tier_baseline: bool = False
    """Stage-A levels record their observed it/s p50 as the tier reference."""


class LadderOptions(BaseModel):
    """Knobs for building the default ladder."""

    tiers: list[BenchTier] = Field(default_factory=lambda: [BenchTier.SD15, BenchTier.SDXL])
    """Which model tiers to attempt, in order. flux/qwen are opt-in (large download/VRAM)."""
    jobs_per_level: int = 4
    include_concurrency: bool = True
    include_features: bool = True
    include_alchemy: bool = True
    include_downloads: bool = False
    """Download levels need network + (for loras) a CivitAI-reachable connection."""
    download_lora_names: list[str] = Field(default_factory=lambda: ["GlowingRunesAI"])
    level_timeout_seconds: float = 900.0
    total_vram_mb: int | None = None
    """The machine's total VRAM, used to size the post-processing sweep's max resolution. None
    (fake/CI or undetected) falls back to a bounded multiple of the baseline's native resolution."""


def _tier_job(tier: BenchTier, **overrides: object) -> CannedImageJobSpec:
    resolution_override = overrides.pop("resolution", None)
    resolution = resolution_override if isinstance(resolution_override, int) else _TIER_RESOLUTIONS[tier]
    return CannedImageJobSpec(
        model=BENCH_TIER_MODELS[tier],
        width=resolution,
        height=resolution,
        **overrides,  # type: ignore[arg-type]
    )


def _its_advisory_criteria() -> LevelCriteria:
    """Criteria for levels whose raw it/s legitimately differs from the stage-A baseline.

    Batch (more images per step), extra-thread (per-job it/s drops as the GPU is shared),
    and feature levels do not have the same per-step work as the baseline, so the it/s
    comparison is reported as an advisory rather than gating the verdict; stability still
    decides pass/fail. See :class:`LevelCriteria.gate_its_against_baseline`.
    """
    return LevelCriteria(gate_its_against_baseline=False)


def _post_processing_resolutions(tier: BenchTier, total_vram_mb: int | None) -> list[int]:
    """Return the de-duplicated, ascending resolutions the post-processing sweep probes for *tier*.

    Always includes 512 and 1024 (the common request sizes) plus the VRAM-derived maximum (the real
    post-processing stress, since the cost scales with output megapixels). The max may equal 1024 on
    a small GPU, in which case the set collapses.
    """
    max_resolution = max_post_processing_resolution(
        baseline=_TIER_BASELINES[tier],
        total_vram_mb=total_vram_mb,
    )
    return sorted({512, 1024, max_resolution})


def build_default_ladder(options: LadderOptions | None = None) -> list[RampLevel]:  # noqa: C901 - one builder per stage reads better inline than fragmented across helpers
    """Build the ordered default ladder for the requested tiers and stages."""
    opts = options if options is not None else LadderOptions()
    levels: list[RampLevel] = []

    def add(
        *,
        stage: BenchStage,
        tier: BenchTier,
        axis: BenchAxis,
        rung: int,
        name: str,
        description: str,
        scenario: ScenarioSpec,
        bridge_data_overrides: dict[str, object] | None = None,
        requires_network: bool = False,
        establishes_tier_baseline: bool = False,
        criteria: LevelCriteria | None = None,
    ) -> None:
        levels.append(
            RampLevel(
                id=f"{stage}-{tier}-{name}",
                stage=stage,
                tier=tier,
                axis=axis,
                rung=rung,
                description=description,
                scenario=scenario,
                bridge_data_overrides=bridge_data_overrides or {},
                requires_network=requires_network,
                timeout_seconds=opts.level_timeout_seconds,
                baseline_hordelib=_TIER_BASELINES[tier],
                establishes_tier_baseline=establishes_tier_baseline,
                criteria=criteria or LevelCriteria(),
            ),
        )

    for tier in opts.tiers:
        if tier not in BENCH_TIER_MODELS:
            raise ValueError(f"Unknown tier {tier!r}; known: {sorted(BENCH_TIER_MODELS)}")

        _add_baseline_level(add, tier=tier, opts=opts)
        if opts.include_concurrency:
            _add_concurrency_levels(add, tier=tier, opts=opts)
        if opts.include_features:
            _add_feature_levels(add, tier=tier, opts=opts)

    if opts.include_alchemy and opts.tiers:
        _add_alchemy_levels(add, tier=opts.tiers[0], opts=opts)

    if opts.include_downloads and opts.tiers:
        _add_download_level(add, tier=opts.tiers[0], opts=opts)

    return levels


class _LevelAdder(Protocol):
    """The closure-bound level appender from :func:`build_default_ladder`, passed to each stage builder.

    Typing it as a protocol lets the per-stage builders stay self-contained and fully type-checked
    while sharing the one ``add`` closure that owns ``levels`` and the shared per-level defaults.
    """

    def __call__(
        self,
        *,
        stage: BenchStage,
        tier: BenchTier,
        axis: BenchAxis,
        rung: int,
        name: str,
        description: str,
        scenario: ScenarioSpec,
        bridge_data_overrides: dict[str, object] | None = None,
        requires_network: bool = False,
        establishes_tier_baseline: bool = False,
        criteria: LevelCriteria | None = None,
    ) -> None:
        """Append one configured :class:`RampLevel` to the ladder under construction."""
        ...


def _add_baseline_level(add: _LevelAdder, *, tier: BenchTier, opts: LadderOptions) -> None:
    """Add the stage-A tier baseline at the most conservative configuration."""
    add(
        stage=BenchStage.BASELINE,
        tier=tier,
        axis=BenchAxis.BASELINE,
        rung=0,
        name="baseline",
        description=f"{tier} baseline: threads=1 queue=1 batch=1 at native resolution",
        scenario=ScenarioSpec(
            name=f"{tier}-baseline",
            image_jobs=[_tier_job(tier, count=opts.jobs_per_level)],
        ),
        establishes_tier_baseline=True,
    )


def _add_concurrency_levels(add: _LevelAdder, *, tier: BenchTier, opts: LadderOptions) -> None:
    """Add the stage-B concurrency axes: queue depth, thread count, and batch size."""
    add(
        stage=BenchStage.CONCURRENCY,
        tier=tier,
        axis=BenchAxis.QUEUE_SIZE,
        rung=1,
        name="queue2",
        description=f"{tier}: queue_size 2 (preload overlaps inference)",
        scenario=ScenarioSpec(name=f"{tier}-q2", image_jobs=[_tier_job(tier, count=opts.jobs_per_level)]),
        bridge_data_overrides={"queue_size": 2},
    )
    add(
        stage=BenchStage.CONCURRENCY,
        tier=tier,
        axis=BenchAxis.THREADS,
        rung=1,
        name="threads2",
        description=f"{tier}: max_threads 2 (two concurrent inference jobs)",
        scenario=ScenarioSpec(name=f"{tier}-t2", image_jobs=[_tier_job(tier, count=opts.jobs_per_level)]),
        bridge_data_overrides={"max_threads": 2, "queue_size": 2},
        criteria=_its_advisory_criteria(),
    )
    for rung, batch in enumerate((2, 4), start=1):
        add(
            stage=BenchStage.CONCURRENCY,
            tier=tier,
            axis=BenchAxis.BATCH,
            rung=rung,
            name=f"batch{batch}",
            description=f"{tier}: n_iter {batch} (batched sampling)",
            scenario=ScenarioSpec(
                name=f"{tier}-b{batch}",
                image_jobs=[_tier_job(tier, count=max(2, opts.jobs_per_level // 2), n_iter=batch)],
            ),
            criteria=_its_advisory_criteria(),
        )


def _add_feature_levels(add: _LevelAdder, *, tier: BenchTier, opts: LadderOptions) -> None:
    """Add the stage-C feature axes: hires-fix, post-processing, controlnet, and the QR-code workflow."""
    half_jobs = max(2, opts.jobs_per_level // 2)

    add(
        stage=BenchStage.FEATURES,
        tier=tier,
        axis=BenchAxis.HIRES_FIX,
        rung=1,
        name="hires_fix",
        description=f"{tier}: hires_fix (second upscaled sampling pass)",
        scenario=ScenarioSpec(
            name=f"{tier}-hires",
            image_jobs=[
                _tier_job(
                    tier,
                    count=half_jobs,
                    resolution=_TIER_RESOLUTIONS[tier] * 2 if tier is BenchTier.SD15 else 1024,
                    hires_fix=True,
                ),
            ],
        ),
        criteria=_its_advisory_criteria(),
    )

    _add_post_processing_levels(add, tier=tier, opts=opts)

    # Classic controlnet (preprocessor-driven) is SD1.5-only in hordelib; SDXL+ map to no real weights.
    if tier is BenchTier.SD15:
        add(
            stage=BenchStage.FEATURES,
            tier=tier,
            axis=BenchAxis.CONTROLNET,
            rung=1,
            name="controlnet",
            description=f"{tier}: controlnet preprocessor sweep ({', '.join(_CONTROLNET_SWEEP_TYPES)})",
            scenario=ScenarioSpec(
                name=f"{tier}-controlnet",
                image_jobs=[
                    _tier_job(tier, count=1, control_type=control_type) for control_type in _CONTROLNET_SWEEP_TYPES
                ],
            ),
            bridge_data_overrides={"allow_controlnet": True},
            criteria=_its_advisory_criteria(),
        )

    # The QR-code workflow is the genuine SDXL controlnet capability, and also runs on SD1.5.
    if tier in (BenchTier.SD15, BenchTier.SDXL):
        qr_overrides: dict[str, object] = {"allow_controlnet": True}
        if tier is BenchTier.SDXL:
            qr_overrides["allow_sdxl_controlnet"] = True
        add(
            stage=BenchStage.FEATURES,
            tier=tier,
            axis=BenchAxis.QR_CODE,
            rung=1,
            name="qr_code",
            description=f"{tier}: qr_code controlnet workflow (the real SDXL controlnet capability)",
            scenario=ScenarioSpec(
                name=f"{tier}-qr",
                image_jobs=[_tier_job(tier, count=half_jobs, workflow=_QR_CODE_WORKFLOW)],
            ),
            bridge_data_overrides=qr_overrides,
            criteria=_its_advisory_criteria(),
        )


def _add_post_processing_levels(add: _LevelAdder, *, tier: BenchTier, opts: LadderOptions) -> None:
    """Add the post-processing axis: a full-coverage sweep, then a resolution-scaling probe.

    Rung 1 exercises *every* upscaler and face-fixer once at native resolution. Rung 2 probes the
    output-resolution scaling that dominates post-processing VRAM, at 512, 1024 and a VRAM-derived
    max; on an undersized GPU the heavy max-resolution rung pre-flight-skips on its own without
    losing the rung-1 coverage.
    """
    post_processors = _image_post_processor_names()
    add(
        stage=BenchStage.FEATURES,
        tier=tier,
        axis=BenchAxis.POST_PROCESSING,
        rung=1,
        name="pp-sweep",
        description=f"{tier}: post-processing sweep ({len(post_processors)} upscalers + face-fixers)",
        scenario=ScenarioSpec(
            name=f"{tier}-pp-sweep",
            image_jobs=[
                _tier_job(tier, count=1, post_processing=[post_processor]) for post_processor in post_processors
            ],
        ),
        bridge_data_overrides={"allow_post_processing": True},
        criteria=_its_advisory_criteria(),
    )

    default_pair = [KNOWN_UPSCALERS.RealESRGAN_x4plus.value, KNOWN_FACEFIXERS.GFPGAN.value]
    resolutions = _post_processing_resolutions(tier, opts.total_vram_mb)
    add(
        stage=BenchStage.FEATURES,
        tier=tier,
        axis=BenchAxis.POST_PROCESSING,
        rung=2,
        name="pp-resolutions",
        description=f"{tier}: post-processing at resolutions {resolutions} (output-megapixel scaling)",
        scenario=ScenarioSpec(
            name=f"{tier}-pp-res",
            image_jobs=[
                _tier_job(tier, count=1, resolution=resolution, post_processing=default_pair)
                for resolution in resolutions
            ],
        ),
        bridge_data_overrides={"allow_post_processing": True},
        criteria=_its_advisory_criteria(),
    )


def _add_alchemy_levels(add: _LevelAdder, *, tier: BenchTier, opts: LadderOptions) -> None:
    """Add the stage-D alchemy axes: the CLIP lane, the graph lane, and concurrent-with-image.

    The CLIP lane (caption/interrogation/NSFW, on the safety process) and the graph lane
    (upscalers/face-fixers/strip-background, on the inference processes) are independent capabilities,
    so each is its own axis; a failure in one never skips the other. The concurrent rung mixes both
    lanes against image jobs to prove (and gate) ``alchemy_allow_concurrent``.
    """
    tier_model = BENCH_TIER_MODELS[tier]
    no_fault_criteria = LevelCriteria(max_faulted_alchemy_forms=0)

    clip_forms = _clip_alchemy_form_names()
    add(
        stage=BenchStage.ALCHEMY,
        tier=tier,
        axis=BenchAxis.ALCHEMY_CLIP,
        rung=1,
        name="alchemy-clip",
        description=f"alchemy CLIP lane (safety process): {', '.join(clip_forms)}",
        scenario=ScenarioSpec(
            name="alchemy-clip",
            alchemy_forms=[CannedAlchemyFormSpec(form=form, count=1) for form in clip_forms],
        ),
        # No image jobs, so models_to_load must be set explicitly (the harness would otherwise derive
        # an empty list). caption requires the explicit BLIP opt-in to be offered.
        bridge_data_overrides={
            "alchemist": True,
            "alchemy_caption_enabled": True,
            "models_to_load": [tier_model],
        },
        criteria=no_fault_criteria,
    )

    graph_forms = _graph_alchemy_form_names(include_strip_background=_strip_background_available())
    add(
        stage=BenchStage.ALCHEMY,
        tier=tier,
        axis=BenchAxis.ALCHEMY_GRAPH,
        rung=1,
        name="alchemy-graph",
        description=f"alchemy graph lane (inference processes): {len(graph_forms)} post-processor forms",
        scenario=ScenarioSpec(
            name="alchemy-graph",
            alchemy_forms=[CannedAlchemyFormSpec(form=form, count=1) for form in graph_forms],
        ),
        bridge_data_overrides={"alchemist": True, "models_to_load": [tier_model]},
        criteria=no_fault_criteria,
    )

    add(
        stage=BenchStage.ALCHEMY,
        tier=tier,
        axis=BenchAxis.ALCHEMY_CONCURRENT,
        rung=1,
        name="alchemy-concurrent",
        description="image + alchemy concurrently on both lanes (headroom-gated)",
        scenario=ScenarioSpec(
            name="alchemy-concurrent",
            image_jobs=[_tier_job(tier, count=max(2, opts.jobs_per_level // 2))],
            alchemy_forms=[
                CannedAlchemyFormSpec(form=KNOWN_CLIP_BLIP_TYPES.caption.value, count=2),
                CannedAlchemyFormSpec(form=KNOWN_UPSCALERS.RealESRGAN_x4plus.value, count=2),
                CannedAlchemyFormSpec(form=KNOWN_FACEFIXERS.GFPGAN.value, count=1),
            ],
        ),
        bridge_data_overrides={
            "alchemist": True,
            "alchemy_caption_enabled": True,
            "alchemy_allow_concurrent": True,
            "alchemy_max_concurrency": 2,
            "queue_size": 2,
        },
        # Concurrent alchemy legitimately lowers image it/s; stability (no faults) decides the verdict.
        criteria=LevelCriteria(gate_its_against_baseline=False, max_faulted_alchemy_forms=0),
    )


def _add_download_level(add: _LevelAdder, *, tier: BenchTier, opts: LadderOptions) -> None:
    """Add the stage-E ad-hoc lora download level (measures download bandwidth)."""
    add(
        stage=BenchStage.DOWNLOADS,
        tier=tier,
        axis=BenchAxis.DOWNLOADS,
        rung=1,
        name="adhoc-lora",
        description="ad-hoc lora fetches from CivitAI (measures download bandwidth)",
        scenario=ScenarioSpec(
            name="adhoc-lora",
            image_jobs=[
                CannedImageJobSpec(
                    model=BENCH_TIER_MODELS[tier],
                    width=_TIER_RESOLUTIONS[tier],
                    height=_TIER_RESOLUTIONS[tier],
                    lora_names=opts.download_lora_names,
                    count=2,
                ),
            ],
        ),
        bridge_data_overrides={"allow_lora": True},
        requires_network=True,
    )


__all__ = [
    "BENCH_TIER_MODELS",
    "BENCH_TIER_MODEL_POOLS",
    "BETA_TIERS",
    "HUGE_TIERS",
    "LadderOptions",
    "RampLevel",
    "build_default_ladder",
]
