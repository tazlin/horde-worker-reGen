"""The post-ramp sustained-load validation (soak) phase.

The ramp proves each capability in isolation over a handful of jobs. This phase then takes
the *synthesized* recommendation and runs the worker under continuous, mixed traffic —
weighted toward the jobs that most stress the chosen configuration (max batch, the heaviest
enabled features, concurrent alchemy) — for a fixed period. It catches problems a short
ramp level cannot: VRAM/RAM creep, thermal throttling, queue backpressure, or recoveries
that only accumulate over time.

The verdict rests on stability over the whole period (no faults, no process recoveries) plus
throughput retention: the sampling rate in the second half of the soak must hold up against
the first half (see ``LevelCriteria.min_its_retention``).
"""

from __future__ import annotations

from horde_worker_regen.benchmark.criteria import LevelCriteria
from horde_worker_regen.benchmark.ladder import (
    _TIER_BASELINES,
    _TIER_RESOLUTIONS,
    BENCH_TIER_MODELS,
    RampLevel,
)
from horde_worker_regen.benchmark.report import SuggestedBridgeData
from horde_worker_regen.benchmark.scenarios import (
    CannedAlchemyFormSpec,
    CannedImageJobSpec,
    ScenarioSpec,
)

_SOAK_START_MARGIN_SECONDS = 180.0
"""Headroom over the soak period for model load at startup and the drain at the end."""


def build_soak_scenario(
    suggested: SuggestedBridgeData,
    tier: str,
    *,
    soak_seconds: float,
    model_pool: list[str] | None = None,
) -> ScenarioSpec:
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

    # (extra kwargs, relative weight) for each enabled profile; replicated across every model.
    profiles: list[tuple[dict, int]] = [
        # A baseline single-image job keeps the mix realistic (not literally everything maxed).
        ({}, 1),
    ]
    if suggested.max_batch > 1:
        profiles.append(({"n_iter": suggested.max_batch}, 3))
    if suggested.allow_controlnet:
        profiles.append(({"control_type": "canny"}, 3))
    if suggested.allow_post_processing:
        profiles.append(({"post_processing": ["RealESRGAN_x4plus", "GFPGAN"]}, 2))

    image_jobs: list[CannedImageJobSpec] = [
        CannedImageJobSpec(model=model, width=resolution, height=resolution, count=weight, **kwargs)
        for model in models
        for kwargs, weight in profiles
    ]

    alchemy_forms: list[CannedAlchemyFormSpec] = []
    if suggested.alchemist:
        alchemy_forms = [
            CannedAlchemyFormSpec(form="caption", count=1),
            CannedAlchemyFormSpec(form="RealESRGAN_x4plus", count=2),
        ]

    return ScenarioSpec(
        name=f"{tier}-soak",
        image_jobs=image_jobs,
        alchemy_forms=alchemy_forms,
        soak_seconds=soak_seconds,
    )


def build_validation_level(
    suggested: SuggestedBridgeData,
    tier: str,
    *,
    soak_seconds: float,
    drain_timeout_seconds: float = 60.0,
    min_its_retention: float = 0.85,
    min_completed_jobs: int = 4,
    min_gpu_duty_cycle_percent: float = 90.0,
    model_pool: list[str] | None = None,
    expect_vram_residency: bool = False,
) -> RampLevel:
    """Build the stage-V validation level that soaks the synthesized config for a tier.

    ``model_pool`` (when it holds more than one model) spreads the soak across distinct models
    so every inference process is exercised; the pool is also loaded by the worker via
    ``models_to_load``. ``expect_vram_residency`` turns on the residency-defeated advisory (set
    once the ``--highvram`` + worker-budget levers are enabled, not for the NORMAL_VRAM baseline).
    """
    scenario = build_soak_scenario(suggested, tier, soak_seconds=soak_seconds, model_pool=model_pool)
    timeout_seconds = soak_seconds + drain_timeout_seconds + _SOAK_START_MARGIN_SECONDS

    overrides = suggested.to_bridge_overrides()
    if model_pool and len(model_pool) > 1:
        overrides["models_to_load"] = list(model_pool)

    pool_note = f" across {len(model_pool)} models" if model_pool and len(model_pool) > 1 else ""

    return RampLevel(
        id=f"V-{tier}-soak",
        stage="V",
        tier=tier,
        axis="validation",
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
            min_gpu_duty_cycle_percent=min_gpu_duty_cycle_percent,
            expect_vram_residency=expect_vram_residency,
        ),
    )
