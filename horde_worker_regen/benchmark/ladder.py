"""The default ramp ladder: ordered benchmark levels from conservative to demanding.

Ordering principle: prove each model tier viable at a conservative configuration first
(stage A, which also establishes the tier's reference sampling rate), then ramp
concurrency (B), then features (C), then alchemy (D), then ad-hoc downloads (E).
A stage-A failure skips the tier's dependent levels; a failure within one axis stops
further rungs of that axis only.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.criteria import LevelCriteria
from horde_worker_regen.benchmark.scenarios import (
    CannedAlchemyFormSpec,
    CannedImageJobSpec,
    ScenarioSpec,
)

BENCH_TIER_MODELS: dict[str, str] = {
    "sd15": "Deliberate",
    "sdxl": "AlbedoBase XL (SDXL)",
    "flux": "FLUX.1 [schnell]",
}

_TIER_BASELINES: dict[str, str] = {
    "sd15": "stable_diffusion_1",
    "sdxl": "stable_diffusion_xl",
    "flux": "flux_1",
}

_TIER_RESOLUTIONS: dict[str, int] = {"sd15": 512, "sdxl": 1024, "flux": 1024}


class RampLevel(BaseModel):
    """One rung of the ramp ladder."""

    id: str
    stage: str
    """A (tier baseline), B (concurrency), C (features), D (alchemy), E (downloads)."""
    tier: str
    axis: str
    """What this level ramps (e.g. "baseline", "queue_size", "lora"); failures stop the axis."""
    rung: int = 0
    """Position within the axis; higher rungs are skipped after a failure on the axis."""
    description: str
    scenario: ScenarioSpec
    bridge_data_overrides: dict = Field(default_factory=dict)
    requires_network: bool = False
    timeout_seconds: float = 900.0
    criteria: LevelCriteria = Field(default_factory=LevelCriteria)
    baseline_hordelib: str = ""
    """The KNOWN_IMAGE_GENERATION_BASELINE value, for pre-flight burden estimates."""
    establishes_tier_baseline: bool = False
    """Stage-A levels record their observed it/s p50 as the tier reference."""


class LadderOptions(BaseModel):
    """Knobs for building the default ladder."""

    tiers: list[str] = Field(default_factory=lambda: ["sd15", "sdxl"])
    """Which model tiers to attempt, in order. flux is opt-in (large download/VRAM)."""
    jobs_per_level: int = 4
    include_concurrency: bool = True
    include_features: bool = True
    include_alchemy: bool = True
    include_downloads: bool = False
    """Download levels need network + (for loras) a CivitAI-reachable connection."""
    download_lora_names: list[str] = Field(default_factory=lambda: ["GlowingRunesAI"])
    level_timeout_seconds: float = 900.0


def _tier_job(tier: str, **overrides: object) -> CannedImageJobSpec:
    resolution = int(overrides.pop("resolution", _TIER_RESOLUTIONS[tier]))
    return CannedImageJobSpec(
        model=BENCH_TIER_MODELS[tier],
        width=resolution,
        height=resolution,
        **overrides,  # type: ignore[arg-type]
    )


def build_default_ladder(options: LadderOptions | None = None) -> list[RampLevel]:
    """Build the ordered default ladder for the requested tiers and stages."""
    opts = options if options is not None else LadderOptions()
    levels: list[RampLevel] = []

    def add(
        *,
        stage: str,
        tier: str,
        axis: str,
        rung: int,
        name: str,
        description: str,
        scenario: ScenarioSpec,
        bridge_data_overrides: dict | None = None,
        requires_network: bool = False,
        establishes_tier_baseline: bool = False,
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
            ),
        )

    for tier in opts.tiers:
        if tier not in BENCH_TIER_MODELS:
            raise ValueError(f"Unknown tier {tier!r}; known: {sorted(BENCH_TIER_MODELS)}")

        # Stage A: tier baseline at the most conservative configuration.
        add(
            stage="A",
            tier=tier,
            axis="baseline",
            rung=0,
            name="baseline",
            description=f"{tier} baseline: threads=1 queue=1 batch=1 at native resolution",
            scenario=ScenarioSpec(
                name=f"{tier}-baseline",
                image_jobs=[_tier_job(tier, count=opts.jobs_per_level)],
            ),
            establishes_tier_baseline=True,
        )

        if opts.include_concurrency:
            add(
                stage="B",
                tier=tier,
                axis="queue_size",
                rung=1,
                name="queue2",
                description=f"{tier}: queue_size 2 (preload overlaps inference)",
                scenario=ScenarioSpec(name=f"{tier}-q2", image_jobs=[_tier_job(tier, count=opts.jobs_per_level)]),
                bridge_data_overrides={"queue_size": 2},
            )
            add(
                stage="B",
                tier=tier,
                axis="threads",
                rung=1,
                name="threads2",
                description=f"{tier}: max_threads 2 (two concurrent inference jobs)",
                scenario=ScenarioSpec(
                    name=f"{tier}-t2",
                    image_jobs=[_tier_job(tier, count=opts.jobs_per_level)],
                ),
                bridge_data_overrides={"max_threads": 2, "queue_size": 2},
            )
            for rung, batch in enumerate((2, 4), start=1):
                add(
                    stage="B",
                    tier=tier,
                    axis="batch",
                    rung=rung,
                    name=f"batch{batch}",
                    description=f"{tier}: n_iter {batch} (batched sampling)",
                    scenario=ScenarioSpec(
                        name=f"{tier}-b{batch}",
                        image_jobs=[_tier_job(tier, count=max(2, opts.jobs_per_level // 2), n_iter=batch)],
                    ),
                )

        if opts.include_features:
            feature_specs: list[tuple[str, ScenarioSpec, bool]] = [
                (
                    "hires_fix",
                    ScenarioSpec(
                        name=f"{tier}-hires",
                        image_jobs=[
                            _tier_job(
                                tier,
                                count=max(2, opts.jobs_per_level // 2),
                                resolution=_TIER_RESOLUTIONS[tier] * 2 if tier == "sd15" else 1024,
                                hires_fix=True,
                            ),
                        ],
                    ),
                    False,
                ),
                (
                    "post_processing",
                    ScenarioSpec(
                        name=f"{tier}-pp",
                        image_jobs=[
                            _tier_job(
                                tier,
                                count=max(2, opts.jobs_per_level // 2),
                                post_processing=["RealESRGAN_x4plus", "GFPGAN"],
                            ),
                        ],
                    ),
                    False,
                ),
            ]
            if tier in ("sd15", "sdxl"):
                feature_specs.append(
                    (
                        "controlnet",
                        ScenarioSpec(
                            name=f"{tier}-controlnet",
                            image_jobs=[
                                _tier_job(tier, count=max(2, opts.jobs_per_level // 2), control_type="canny"),
                            ],
                        ),
                        False,
                    ),
                )

            for axis, scenario, requires_network in feature_specs:
                overrides: dict = {}
                if axis == "post_processing":
                    overrides["allow_post_processing"] = True
                if axis == "controlnet":
                    overrides["allow_controlnet"] = True
                    if tier == "sdxl":
                        overrides["allow_sdxl_controlnet"] = True
                add(
                    stage="C",
                    tier=tier,
                    axis=axis,
                    rung=1,
                    name=axis,
                    description=f"{tier}: {axis}",
                    scenario=scenario,
                    bridge_data_overrides=overrides,
                    requires_network=requires_network,
                )

    if opts.include_alchemy and opts.tiers:
        first_tier = opts.tiers[0]
        add(
            stage="D",
            tier=first_tier,
            axis="alchemy",
            rung=1,
            name="alchemy-solo",
            description="alchemy only: caption + upscale forms, no image jobs",
            scenario=ScenarioSpec(
                name="alchemy-solo",
                alchemy_forms=[
                    CannedAlchemyFormSpec(form="caption", count=2),
                    CannedAlchemyFormSpec(form="RealESRGAN_x4plus", count=2),
                ],
            ),
            # No image jobs in the scenario, so the harness would otherwise derive an
            # empty models_to_load.
            bridge_data_overrides={"alchemist": True, "models_to_load": [BENCH_TIER_MODELS[first_tier]]},
        )
        add(
            stage="D",
            tier=first_tier,
            axis="alchemy",
            rung=2,
            name="alchemy-concurrent",
            description="image + alchemy concurrently (headroom-gated)",
            scenario=ScenarioSpec(
                name="alchemy-concurrent",
                image_jobs=[_tier_job(first_tier, count=max(2, opts.jobs_per_level // 2))],
                alchemy_forms=[
                    CannedAlchemyFormSpec(form="caption", count=2),
                    CannedAlchemyFormSpec(form="RealESRGAN_x4plus", count=1),
                ],
            ),
            bridge_data_overrides={
                "alchemist": True,
                "alchemy_allow_concurrent": True,
                "queue_size": 2,
            },
        )

    if opts.include_downloads and opts.tiers:
        first_tier = opts.tiers[0]
        add(
            stage="E",
            tier=first_tier,
            axis="downloads",
            rung=1,
            name="adhoc-lora",
            description="ad-hoc lora fetches from CivitAI (measures download bandwidth)",
            scenario=ScenarioSpec(
                name="adhoc-lora",
                image_jobs=[
                    CannedImageJobSpec(
                        model=BENCH_TIER_MODELS[first_tier],
                        width=_TIER_RESOLUTIONS[first_tier],
                        height=_TIER_RESOLUTIONS[first_tier],
                        lora_names=opts.download_lora_names,
                        count=2,
                    ),
                ],
            ),
            bridge_data_overrides={"allow_lora": True},
            requires_network=True,
        )

    return levels
