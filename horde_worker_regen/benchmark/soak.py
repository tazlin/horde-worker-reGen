"""The post-ramp sustained-load validation (soak) phase.

The ramp proves each capability in isolation over a handful of jobs. This phase then takes
the *synthesized* recommendation and runs the worker under continuous, mixed traffic,
weighted toward the jobs that most stress the chosen configuration (max batch, the heaviest
enabled features, concurrent alchemy), for a fixed period. It catches problems a short
ramp level cannot: VRAM/RAM creep, thermal throttling, queue backpressure, or recoveries
that only accumulate over time.

The verdict rests on stability over the whole period (no faults, no process recoveries) plus
throughput retention: the sampling rate in the second half of the soak must hold up against
the first half (see ``LevelCriteria.min_its_retention``).
"""

from __future__ import annotations

import dataclasses

from horde_sdk.generation_parameters.alchemy.consts import (
    KNOWN_CLIP_BLIP_TYPES,
    KNOWN_FACEFIXERS,
    KNOWN_UPSCALERS,
)

from horde_worker_regen.benchmark.criteria import LevelCriteria
from horde_worker_regen.benchmark.enums import BenchAxis, BenchStage, BenchTier
from horde_worker_regen.benchmark.ladder import (
    _TIER_BASELINES,
    _TIER_RESOLUTIONS,
    BENCH_TIER_MODELS,
    RampLevel,
    tier_canned_job_overrides,
)
from horde_worker_regen.benchmark.report import SuggestedBridgeData
from horde_worker_regen.benchmark.scenarios import (
    CannedAlchemyFormSpec,
    CannedImageJobSpec,
    Scenario,
)

_SOAK_POST_PROCESSING = [KNOWN_UPSCALERS.RealESRGAN_x4plus.value, KNOWN_FACEFIXERS.GFPGAN.value]
"""A representative upscaler + face-fixer pair for the soak's post-processing profile."""


@dataclasses.dataclass(frozen=True)
class _SoakProfile:
    """One weighted job shape in the soak mix, replicated across every model in the pool."""

    weight: int
    n_iter: int = 1
    control_type: str | None = None
    workflow: str | None = None
    post_processing: list[str] = dataclasses.field(default_factory=list)


_SOAK_START_MARGIN_SECONDS = 180.0
"""Headroom over the soak period for model load at startup and the drain at the end."""


def build_soak_scenario(
    suggested: SuggestedBridgeData,
    tier: BenchTier,
    *,
    soak_seconds: float,
    model_pool: list[str] | None = None,
) -> Scenario:
    """Build a sustained, mixed workload weighted toward the jobs that max the chosen config.

    Each spec's ``count`` is used as a *relative weight* (the soak generates jobs continuously
    rather than expanding a fixed list), so the heavy specs dominate the stream.

    When ``model_pool`` holds more than one model, every job profile is replicated across the
    pool so the generated stream spreads over distinct models. That defeats the popper's
    2-per-model in-flight cap (which would otherwise throttle a 4-process soak to 2 concurrent
    jobs) and exercises the cross-process model placement the duty-cycle work targets. The
    relative weighting between profiles is preserved within each model. Defaults to the tier's
    single baseline model, preserving the original single-model soak.
    """
    models = model_pool if model_pool else [BENCH_TIER_MODELS[tier]]
    resolution = _TIER_RESOLUTIONS[tier]

    # One weighted profile per enabled job shape, replicated across every model in the pool.
    profiles: list[_SoakProfile] = [
        # A baseline single-image job keeps the mix realistic (not literally everything maxed).
        _SoakProfile(weight=1),
    ]
    if suggested.max_batch > 1:
        profiles.append(_SoakProfile(weight=3, n_iter=suggested.max_batch))
    # Controlnet stress is tier-specific: SD1.5 uses a preprocessor control type; the SDXL controlnet
    # capability is the qr_code workflow, so an SDXL soak must drive that, not a (non-existent) SDXL
    # canny controlnet.
    if suggested.allow_sdxl_controlnet and tier is BenchTier.SDXL:
        profiles.append(_SoakProfile(weight=3, workflow="qr_code"))
    elif suggested.allow_controlnet and tier is BenchTier.SD15:
        profiles.append(_SoakProfile(weight=3, control_type="canny"))
    if suggested.allow_post_processing:
        profiles.append(_SoakProfile(weight=2, post_processing=list(_SOAK_POST_PROCESSING)))

    fixed_job_kwargs = tier_canned_job_overrides(tier)
    image_jobs: list[CannedImageJobSpec] = [
        CannedImageJobSpec(
            model=model,
            width=resolution,
            height=resolution,
            count=profile.weight,
            n_iter=profile.n_iter,
            control_type=profile.control_type,
            workflow=profile.workflow,
            post_processing=list(profile.post_processing),
            **fixed_job_kwargs,  # type: ignore[arg-type]
        )
        for model in models
        for profile in profiles
    ]

    alchemy_forms: list[CannedAlchemyFormSpec] = []
    if suggested.alchemist:
        # Exercise both alchemy lanes in the soak: a CLIP form (safety process) and graph forms
        # (inference processes), so sustained-load alchemy is not silently single-lane.
        alchemy_forms = [
            CannedAlchemyFormSpec(form=KNOWN_CLIP_BLIP_TYPES.caption.value, count=1),
            CannedAlchemyFormSpec(form=KNOWN_UPSCALERS.RealESRGAN_x4plus.value, count=2),
            CannedAlchemyFormSpec(form=KNOWN_FACEFIXERS.GFPGAN.value, count=1),
        ]

    return Scenario(
        name=f"{tier}-soak",
        image_jobs=image_jobs,
        alchemy_forms=alchemy_forms,
        soak_seconds=soak_seconds,
    )


_DUTY_CYCLE_TARGET_PERCENT = 90.0
"""The GPU duty-cycle the soak drives toward; reported as an advisory, not a default pass/fail gate."""


def build_validation_level(
    suggested: SuggestedBridgeData,
    tier: BenchTier,
    *,
    soak_seconds: float,
    drain_timeout_seconds: float = 60.0,
    min_its_retention: float = 0.85,
    min_completed_jobs: int = 4,
    target_gpu_duty_cycle_percent: float = _DUTY_CYCLE_TARGET_PERCENT,
    strict_duty_cycle: bool = False,
    model_pool: list[str] | None = None,
    expect_vram_residency: bool = False,
) -> RampLevel:
    """Build the stage-V validation level that soaks the synthesized config for a tier.

    ``model_pool`` (when it holds more than one model) spreads the soak across distinct models
    so every inference process is exercised; the pool is also loaded by the worker via
    ``models_to_load``. ``expect_vram_residency`` turns on the residency-defeated advisory (set
    once the ``--highvram`` + worker-budget levers are enabled, not for the NORMAL_VRAM baseline).

    GPU duty cycle is reported against ``target_gpu_duty_cycle_percent`` as an advisory by default:
    a baseline soak legitimately misses the 90% north-star until the residency/overlap levers land,
    so the soak passes on stability and throughput retention and surfaces the duty-cycle shortfall
    with full attribution. ``strict_duty_cycle`` promotes the target to a hard pass/fail gate, for
    enforcing the number on a reference machine.
    """
    scenario = build_soak_scenario(suggested, tier, soak_seconds=soak_seconds, model_pool=model_pool)
    timeout_seconds = soak_seconds + drain_timeout_seconds + _SOAK_START_MARGIN_SECONDS

    overrides = suggested.to_bridge_overrides()
    if model_pool and len(model_pool) > 1:
        overrides["models_to_load"] = list(model_pool)

    pool_note = f" across {len(model_pool)} models" if model_pool and len(model_pool) > 1 else ""

    return RampLevel(
        id=f"V-{tier}-soak",
        stage=BenchStage.VALIDATION,
        tier=tier,
        axis=BenchAxis.VALIDATION,
        rung=1,
        description=(
            f"{tier} sustained-load validation: ~{soak_seconds:.0f}s of mixed, mostly-max-config "
            f"traffic{pool_note} against the recommended bridgeData"
        ),
        scenario=scenario,
        bridge_data_overrides=overrides,
        timeout_seconds=timeout_seconds,
        baseline_hordelib=_TIER_BASELINES[tier],
        criteria=LevelCriteria(
            gate_its_against_baseline=False,
            min_its_retention=min_its_retention,
            min_completed_jobs=min_completed_jobs,
            target_gpu_utilization_percent=target_gpu_duty_cycle_percent,
            min_gpu_duty_cycle_percent=(target_gpu_duty_cycle_percent if strict_duty_cycle else None),
            expect_vram_residency=expect_vram_residency,
        ),
    )


__all__ = ["build_soak_scenario", "build_validation_level"]
