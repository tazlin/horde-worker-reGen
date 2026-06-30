"""Build the catalog of capability probes: one experiment per provable property, with its edges.

This is the capability-engine replacement for ``build_default_ladder`` plus ``build_validation_level``.
It produces :class:`~horde_worker_regen.benchmark.capabilities.probe.CapabilityProbe` objects whose
workloads and criteria are identical to the old ladder, but whose ordering is expressed declaratively
through each probe's ``requires`` edges (a topological DAG) rather than the old stage/axis/rung lattice
and imperative skip cascade. :func:`build_capability_catalog` returns the static probes
(conservative-to-demanding, every non-baseline probe requiring its tier baseline); the sustained-load
soak is built separately by :func:`build_sustained_probe`, because its scenario is derived from the
recommendation synthesized *after* the static probes run.

The per-tier model / baseline / resolution tables and the small scenario helpers are reused from the
ladder module for now; the source-of-truth move into this package happens when the ladder is deleted.
"""

from __future__ import annotations

from typing import Protocol

from horde_sdk.generation_parameters.alchemy.consts import (
    KNOWN_CLIP_BLIP_TYPES,
    KNOWN_FACEFIXERS,
    KNOWN_UPSCALERS,
)
from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityKind
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.criteria import LevelCriteria
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.ladder import (
    _CONTROLNET_SWEEP_TYPES,
    _QR_CODE_WORKFLOW,
    _TIER_BASELINES,
    _TIER_RESOLUTIONS,
    BENCH_TIER_MODELS,
    _clip_alchemy_form_names,
    _graph_alchemy_form_names,
    _image_post_processor_names,
    _its_advisory_criteria,
    _post_processing_resolutions,
    _strip_background_available,
    _tier_job,
)
from horde_worker_regen.benchmark.report import SuggestedBridgeData
from horde_worker_regen.benchmark.scenarios import CannedAlchemyFormSpec, CannedImageJobSpec, Scenario
from horde_worker_regen.benchmark.soak import build_soak_scenario

_SUSTAINED_DRAIN_TIMEOUT_SECONDS = 60.0
_SUSTAINED_START_MARGIN_SECONDS = 180.0
_DUTY_CYCLE_TARGET_PERCENT = 90.0


class CatalogOptions(BaseModel):
    """Knobs for building the capability catalog (the replacement for ``LadderOptions``)."""

    tiers: list[BenchTier] = Field(default_factory=lambda: [BenchTier.SD15, BenchTier.SDXL])
    """Which model tiers to probe, in order. flux/qwen/zimage are opt-in (large download/VRAM)."""
    jobs_per_level: int = 4
    include_concurrency: bool = True
    include_features: bool = True
    include_alchemy: bool = True
    include_downloads: bool = False
    """Download probes need network + (for loras) a CivitAI-reachable connection."""
    excluded_kinds: set[CapabilityKind] = Field(default_factory=set)
    """Individual capability kinds to drop, independent of the coarse stage toggles.

    A probe is built only if its stage is included *and* its kind is not excluded. BASELINE,
    LORA_DOWNLOAD, and SUSTAINED are governed by other flags (the tier set, ``include_downloads``,
    and the validate switch) rather than by per-kind selection."""
    download_lora_names: list[str] = Field(default_factory=lambda: ["GlowingRunesAI"])
    probe_timeout_seconds: float = 900.0
    total_vram_mb: int | None = None
    """The machine's total VRAM, used to size the post-processing sweep's max resolution. None
    (fake/CI or undetected) falls back to a bounded multiple of the baseline's native resolution."""


class _ProbeAdder(Protocol):
    """The closure-bound probe appender from :func:`build_capability_catalog`, passed to each builder.

    Typing it as a protocol lets the per-stage builders stay self-contained and fully type-checked
    while sharing the one ``add`` closure that owns ``probes`` and the shared per-probe defaults.
    """

    def __call__(
        self,
        *,
        capability: Capability,
        scenario: Scenario,
        requires: tuple[Capability, ...] = (),
        bridge_data_overrides: dict[str, object] | None = None,
        requires_network: bool = False,
        establishes_baseline: bool = False,
        criteria: LevelCriteria | None = None,
    ) -> None:
        """Append one configured :class:`CapabilityProbe` to the catalog under construction."""
        ...


def build_capability_catalog(options: CatalogOptions | None = None) -> list[CapabilityProbe]:
    """Build the static capability probes for the requested tiers and stages, in catalog order."""
    opts = options if options is not None else CatalogOptions()
    probes: list[CapabilityProbe] = []

    def add(
        *,
        capability: Capability,
        scenario: Scenario,
        requires: tuple[Capability, ...] = (),
        bridge_data_overrides: dict[str, object] | None = None,
        requires_network: bool = False,
        establishes_baseline: bool = False,
        criteria: LevelCriteria | None = None,
    ) -> None:
        # One chokepoint for per-kind deselection: every probe declares its kind via its capability, so
        # excluding a kind drops exactly its probes (and, since rungs of one kind share it, their higher
        # rungs too) without each stage builder needing to know about the exclusion.
        if capability.kind in opts.excluded_kinds:
            return
        probes.append(
            CapabilityProbe(
                capability=capability,
                scenario=scenario,
                requires=requires,
                bridge_data_overrides=bridge_data_overrides or {},
                requires_network=requires_network,
                timeout_seconds=opts.probe_timeout_seconds,
                baseline_hordelib=_TIER_BASELINES[capability.tier],
                establishes_baseline=establishes_baseline,
                criteria=criteria or LevelCriteria(),
            ),
        )

    for tier in opts.tiers:
        if tier not in BENCH_TIER_MODELS:
            raise ValueError(f"Unknown tier {tier!r}; known: {sorted(BENCH_TIER_MODELS)}")

        baseline = Capability(tier=tier, kind=CapabilityKind.BASELINE)
        _add_baseline_probe(add, tier=tier, baseline=baseline, opts=opts)
        if opts.include_concurrency:
            _add_concurrency_probes(add, tier=tier, baseline=baseline, opts=opts)
        if opts.include_features:
            _add_feature_probes(add, tier=tier, baseline=baseline, opts=opts)

    if opts.include_alchemy and opts.tiers:
        _add_alchemy_probes(add, tier=opts.tiers[0], opts=opts)

    if opts.include_downloads and opts.tiers:
        _add_download_probe(add, tier=opts.tiers[0], opts=opts)

    return probes


def _add_baseline_probe(add: _ProbeAdder, *, tier: BenchTier, baseline: Capability, opts: CatalogOptions) -> None:
    """Add the tier baseline at the most conservative configuration (establishes the it/s reference)."""
    add(
        capability=baseline,
        scenario=Scenario(
            name=f"{tier}-baseline",
            image_jobs=[_tier_job(tier, count=opts.jobs_per_level)],
        ),
        establishes_baseline=True,
    )


def _add_concurrency_probes(add: _ProbeAdder, *, tier: BenchTier, baseline: Capability, opts: CatalogOptions) -> None:
    """Add the concurrency probes: queue depth, thread count, and batch size (batch 4 builds on batch 2)."""
    # magnitude carries the proven value (queue depth / thread count) so the recommendation reads it
    # straight off the result, the same way BATCH carries its size; the runtime value still rides in
    # bridge_data_overrides, which is what the harness applies.
    add(
        capability=Capability(tier=tier, kind=CapabilityKind.QUEUE_SIZE, magnitude=2),
        scenario=Scenario(name=f"{tier}-q2", image_jobs=[_tier_job(tier, count=opts.jobs_per_level)]),
        requires=(baseline,),
        bridge_data_overrides={"queue_size": 2},
    )
    add(
        capability=Capability(tier=tier, kind=CapabilityKind.THREADS, magnitude=2),
        scenario=Scenario(name=f"{tier}-t2", image_jobs=[_tier_job(tier, count=opts.jobs_per_level)]),
        requires=(baseline,),
        bridge_data_overrides={"max_threads": 2, "queue_size": 2},
        criteria=_its_advisory_criteria(),
    )
    previous = baseline
    for batch in (2, 4):
        batch_capability = Capability(tier=tier, kind=CapabilityKind.BATCH, magnitude=batch)
        add(
            capability=batch_capability,
            scenario=Scenario(
                name=f"{tier}-b{batch}",
                image_jobs=[_tier_job(tier, count=max(2, opts.jobs_per_level // 2), n_iter=batch)],
            ),
            requires=(previous,),
            criteria=_its_advisory_criteria(),
        )
        previous = batch_capability


def _add_feature_probes(add: _ProbeAdder, *, tier: BenchTier, baseline: Capability, opts: CatalogOptions) -> None:
    """Add the feature probes: hires-fix, post-processing, controlnet, and the QR-code workflow."""
    half_jobs = max(2, opts.jobs_per_level // 2)

    # hires_fix is not supported by all model families (e.g. zimage declares it unsupported).
    if tier is not BenchTier.ZIMAGE:
        add(
            capability=Capability(tier=tier, kind=CapabilityKind.HIRES_FIX),
            scenario=Scenario(
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
            requires=(baseline,),
            criteria=_its_advisory_criteria(),
        )

    _add_post_processing_probes(add, tier=tier, baseline=baseline, opts=opts)

    # Classic controlnet (preprocessor-driven) is SD1.5-only in hordelib; SDXL+ map to no real weights.
    if tier is BenchTier.SD15:
        add(
            capability=Capability(tier=tier, kind=CapabilityKind.CONTROLNET),
            scenario=Scenario(
                name=f"{tier}-controlnet",
                image_jobs=[
                    _tier_job(tier, count=1, control_type=control_type) for control_type in _CONTROLNET_SWEEP_TYPES
                ],
            ),
            requires=(baseline,),
            bridge_data_overrides={"allow_controlnet": True},
            criteria=_its_advisory_criteria(),
        )

    # The QR-code workflow is the genuine SDXL controlnet capability, and also runs on SD1.5.
    if tier in (BenchTier.SD15, BenchTier.SDXL):
        qr_overrides: dict[str, object] = {"allow_controlnet": True}
        if tier is BenchTier.SDXL:
            qr_overrides["allow_sdxl_controlnet"] = True
        add(
            capability=Capability(tier=tier, kind=CapabilityKind.QR_CODE),
            scenario=Scenario(
                name=f"{tier}-qr",
                image_jobs=[_tier_job(tier, count=half_jobs, workflow=_QR_CODE_WORKFLOW)],
            ),
            requires=(baseline,),
            bridge_data_overrides=qr_overrides,
            criteria=_its_advisory_criteria(),
        )


def _add_post_processing_probes(
    add: _ProbeAdder, *, tier: BenchTier, baseline: Capability, opts: CatalogOptions
) -> None:
    """Add the post-processing sweep, then the resolution-scaling probe that builds on it.

    The sweep exercises every upscaler and face-fixer once at native resolution. The resolution probe
    then probes the output-resolution scaling that dominates post-processing VRAM, at 512, 1024 and a
    VRAM-derived max; on an undersized GPU the heavy resolution probe pre-flight-skips on its own
    without losing the sweep's coverage. The resolution probe requires the sweep (the old rung-2 edge).
    """
    post_processors = _image_post_processor_names()
    sweep = Capability(tier=tier, kind=CapabilityKind.POST_PROCESSING)
    add(
        capability=sweep,
        scenario=Scenario(
            name=f"{tier}-pp-sweep",
            image_jobs=[
                _tier_job(tier, count=1, post_processing=[post_processor]) for post_processor in post_processors
            ],
        ),
        requires=(baseline,),
        bridge_data_overrides={"allow_post_processing": True},
        criteria=_its_advisory_criteria(),
    )

    default_pair = [KNOWN_UPSCALERS.RealESRGAN_x4plus.value, KNOWN_FACEFIXERS.GFPGAN.value]
    resolutions = _post_processing_resolutions(tier, opts.total_vram_mb)
    add(
        # Magnitude is the maximum resolution probed, both to disambiguate from the sweep and to read
        # meaningfully in the slug (e.g. sd15-post_processing-2048).
        capability=Capability(tier=tier, kind=CapabilityKind.POST_PROCESSING, magnitude=max(resolutions)),
        scenario=Scenario(
            name=f"{tier}-pp-res",
            image_jobs=[
                _tier_job(tier, count=1, resolution=resolution, post_processing=default_pair)
                for resolution in resolutions
            ],
        ),
        requires=(sweep,),
        bridge_data_overrides={"allow_post_processing": True},
        criteria=_its_advisory_criteria(),
    )


def _add_alchemy_probes(add: _ProbeAdder, *, tier: BenchTier, opts: CatalogOptions) -> None:
    """Add the alchemy probes: the CLIP lane, the graph lane, and concurrent-with-image.

    The CLIP lane (caption/interrogation/NSFW, on the safety process) and the graph lane
    (upscalers/face-fixers/strip-background, on the inference processes) are independent capabilities;
    each requires only the tier baseline so a failure in one never skips the other. The concurrent
    probe mixes both lanes against image jobs to prove (and gate) ``alchemy_allow_concurrent``.
    """
    baseline = Capability(tier=tier, kind=CapabilityKind.BASELINE)
    tier_model = BENCH_TIER_MODELS[tier]
    no_fault_criteria = LevelCriteria(max_faulted_alchemy_forms=0)

    clip_forms = _clip_alchemy_form_names()
    add(
        capability=Capability(tier=tier, kind=CapabilityKind.ALCHEMY_CLIP),
        scenario=Scenario(
            name="alchemy-clip",
            alchemy_forms=[CannedAlchemyFormSpec(form=form, count=1) for form in clip_forms],
        ),
        requires=(baseline,),
        # No image jobs, so models_to_load must be set explicitly (the harness would otherwise derive an
        # empty list). caption requires the explicit BLIP opt-in to be offered.
        bridge_data_overrides={
            "alchemist": True,
            "alchemy_caption_enabled": True,
            "models_to_load": [tier_model],
        },
        criteria=no_fault_criteria,
    )

    graph_forms = _graph_alchemy_form_names(include_strip_background=_strip_background_available())
    add(
        capability=Capability(tier=tier, kind=CapabilityKind.ALCHEMY_GRAPH),
        scenario=Scenario(
            name="alchemy-graph",
            alchemy_forms=[CannedAlchemyFormSpec(form=form, count=1) for form in graph_forms],
        ),
        requires=(baseline,),
        bridge_data_overrides={"alchemist": True, "models_to_load": [tier_model]},
        criteria=no_fault_criteria,
    )

    add(
        capability=Capability(tier=tier, kind=CapabilityKind.ALCHEMY_CONCURRENT),
        scenario=Scenario(
            name="alchemy-concurrent",
            image_jobs=[_tier_job(tier, count=max(2, opts.jobs_per_level // 2))],
            alchemy_forms=[
                CannedAlchemyFormSpec(form=KNOWN_CLIP_BLIP_TYPES.caption.value, count=2),
                CannedAlchemyFormSpec(form=KNOWN_UPSCALERS.RealESRGAN_x4plus.value, count=2),
                CannedAlchemyFormSpec(form=KNOWN_FACEFIXERS.GFPGAN.value, count=1),
            ],
        ),
        requires=(baseline,),
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


def _add_download_probe(add: _ProbeAdder, *, tier: BenchTier, opts: CatalogOptions) -> None:
    """Add the ad-hoc lora download probe (measures download bandwidth, proves ``allow_lora``)."""
    baseline = Capability(tier=tier, kind=CapabilityKind.BASELINE)
    add(
        capability=Capability(tier=tier, kind=CapabilityKind.LORA_DOWNLOAD),
        scenario=Scenario(
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
        requires=(baseline,),
        bridge_data_overrides={"allow_lora": True},
        requires_network=True,
    )


def build_sustained_probe(
    suggested: SuggestedBridgeData,
    tier: BenchTier,
    *,
    soak_seconds: float,
    requires: tuple[Capability, ...] = (),
    min_its_retention: float = 0.85,
    min_completed_jobs: int = 4,
    strict_duty_cycle: bool = False,
    model_pool: list[str] | None = None,
    expect_vram_residency: bool = False,
) -> CapabilityProbe:
    """Build the sustained-load (soak) probe from the synthesized recommendation, for a tier.

    Unlike the static catalog probes, the soak's workload is derived from the recommendation the static
    run produced, so this is built by the executor after synthesis rather than at catalog time. It
    requires the tier baseline plus whatever feature capabilities the caller proved (so the soak only
    exercises configurations that held in isolation). GPU duty cycle is an advisory by default;
    ``strict_duty_cycle`` promotes it to a hard gate for a reference machine.
    """
    scenario = build_soak_scenario(suggested, tier, soak_seconds=soak_seconds, model_pool=model_pool)
    timeout_seconds = soak_seconds + _SUSTAINED_DRAIN_TIMEOUT_SECONDS + _SUSTAINED_START_MARGIN_SECONDS

    overrides = suggested.to_bridge_overrides()
    if model_pool and len(model_pool) > 1:
        overrides["models_to_load"] = list(model_pool)

    return CapabilityProbe(
        capability=Capability(tier=tier, kind=CapabilityKind.SUSTAINED),
        scenario=scenario,
        requires=requires,
        bridge_data_overrides=overrides,
        timeout_seconds=timeout_seconds,
        baseline_hordelib=_TIER_BASELINES[tier],
        criteria=LevelCriteria(
            gate_its_against_baseline=False,
            min_its_retention=min_its_retention,
            min_completed_jobs=min_completed_jobs,
            target_gpu_utilization_percent=_DUTY_CYCLE_TARGET_PERCENT,
            min_gpu_duty_cycle_percent=(_DUTY_CYCLE_TARGET_PERCENT if strict_duty_cycle else None),
            expect_vram_residency=expect_vram_residency,
        ),
    )


__all__ = [
    "CatalogOptions",
    "build_capability_catalog",
    "build_sustained_probe",
]
