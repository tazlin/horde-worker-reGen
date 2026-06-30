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

from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.ladder import (
    _TIER_RESOLUTIONS,
    BENCH_TIER_MODELS,
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


__all__ = ["build_soak_scenario"]
