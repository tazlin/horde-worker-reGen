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
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

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

_LORA_STORM_SHARED_REFERENCES: tuple[str, ...] = (
    "lora_storm_shared_00",
    "lora_storm_shared_01",
    "lora_storm_shared_02",
)
"""A small pool of LoRA references reused across many soak jobs: the cache-hit path.

Because a soak template mints every job with the same references, jobs drawn from these templates
resolve to the same on-disk files, so after the first download each subsequent request is served from
the prefetch cache. Synthetic names (they need not exist on CivitAI): the plumbing they exercise is the
worker's dedup/reuse, and in a fake-mode soak no real download is attempted."""

_LORA_STORM_UNIQUE_REFERENCES: tuple[str, ...] = tuple(f"lora_storm_unique_{index:02d}" for index in range(8))
"""A larger pool of distinct LoRA references, one per template, spread across the mix: the download path.

Sustained pressure comes from the breadth of distinct references the stream cycles through, each forcing
a first-time resolution/download before it can be cached."""

_PRODUCTION_REPLAY_SHARED_REFERENCES: tuple[str, ...] = (
    "production_replay_shared_00",
    "production_replay_shared_01",
    "production_replay_shared_02",
)
"""The cache-hit LoRA pool for the ``production_replay`` mix, reused across its LoRA-bearing templates.

Synthetic like the storm's pools: meaningful only in a fake-mode run. A real-mode soak supplies resolvable
references so the modest LoRA share the fingerprint records is exercised against real files."""

_PRODUCTION_REPLAY_UNIQUE_REFERENCES: tuple[str, ...] = tuple(
    f"production_replay_unique_{index:02d}" for index in range(8)
)
"""The download-pressure LoRA pool for the ``production_replay`` mix, distinct references drawn once each.

Production LoRA traffic is a minority of pops, so this pool is used sparingly relative to the storm's; it
exists so the download path is present in the replay, not so it dominates it."""


def _validate_lora_reference_pools(mix_name: str, shared: Sequence[str], unique: Sequence[str]) -> None:
    """Reject reference pools too small for a LoRA soak mix's indexing (>=3 shared, >=8 unique).

    Shared by the LoRA-bearing soak builders so a real-mode operator who supplies partial pools fails fast
    with a clear message instead of building a mix that silently drops references or measures the failure
    path. The mix name is threaded into the message so the caller knows which builder rejected the pools.
    """
    if len(shared) < 3 or len(unique) < 8:
        raise ValueError(
            f"the {mix_name} mix needs at least 3 shared and 8 unique LoRA references "
            f"(got {len(shared)} shared, {len(unique)} unique)",
        )


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


def build_lora_storm_soak_scenario(
    *,
    soak_seconds: float,
    shared_lora_references: Sequence[str] = _LORA_STORM_SHARED_REFERENCES,
    unique_lora_references: Sequence[str] = _LORA_STORM_UNIQUE_REFERENCES,
) -> Scenario:
    """Build the ``lora_storm`` soak: sustained LoRA-heavy traffic across a light and a heavy base model.

    The default reference pools are synthetic and only meaningful for a fake-mode run (no real download is
    attempted). A real-mode soak must supply real, resolvable LoRA references (CivitAI names or version ids)
    through ``shared_lora_references`` (at least 3: the cache-hit pool) and ``unique_lora_references`` (at
    least 8: the download-pressure pool), or every prefetch would fail and the run would measure the failure
    path instead of the storm.

    The mix is designed to exercise, and let a run's result surface evaluate, three things at once:

    - **Download pressure off the inference lane.** A larger pool of unique LoRA references
      (:data:`_LORA_STORM_UNIQUE_REFERENCES`) keeps first-time resolutions flowing so the dedicated
      download process, not the inference or preload path, absorbs the fetch cost. A small pool of
      repeated references (:data:`_LORA_STORM_SHARED_REFERENCES`) interleaves cache-hit jobs, proving
      reuse short-circuits the download for an already-prepared reference.
    - **Gate liveness.** Roughly two thirds of pops carry one to four LoRAs; the remaining plain jobs are
      the control group. A job with unprepared auxiliaries holds no inference lane or VRAM reservation, so
      the plain jobs must keep flowing through the gate while LoRA jobs are still downloading. Both base
      models (a light SD1.5 checkpoint and a heavier SDXL one) carry LoRA and plain traffic so the storm
      spans warm and cold base loads.
    - **Backoff behaviour.** With prefetch working, no LoRA job should fault for a missing auxiliary, so the
      consecutive-failures pop pause must never arm. ``HarnessResult.consecutive_failed_jobs_pause_count``
      staying zero (alongside a zero ``num_jobs_faulted_with_loras``) is the scheduling-clean verdict.

    Each spec's ``count`` is a relative generation weight (the soak generates continuously rather than
    expanding a fixed list); a template always mints the same references, which is what makes shared
    references recur (cache hits) and unique references stay distinct (downloads).
    """
    _validate_lora_reference_pools("lora_storm", shared_lora_references, unique_lora_references)

    light_model = BENCH_TIER_MODELS[BenchTier.SD15]
    light_resolution = _TIER_RESOLUTIONS[BenchTier.SD15]
    heavy_model = BENCH_TIER_MODELS[BenchTier.SDXL]
    heavy_resolution = _TIER_RESOLUTIONS[BenchTier.SDXL]

    shared = tuple(shared_lora_references)
    unique = tuple(unique_lora_references)

    # (model, resolution, lora_names, weight). Cache-hit specs reuse the shared pool; download-pressure
    # specs each own distinct unique references; the trailing pair are the plain-job liveness control.
    profiles: list[tuple[str, int, list[str], int]] = [
        # Cache-hit path: shared references recur across the run, at 1-3 LoRAs per job.
        (light_model, light_resolution, [shared[0]], 10),
        (light_model, light_resolution, [shared[0], shared[1]], 8),
        (light_model, light_resolution, [shared[0], shared[1], shared[2]], 6),
        (heavy_model, heavy_resolution, [shared[2]], 6),
        # Download-pressure path: distinct references, spread over both models, up to 4 LoRAs per job.
        (light_model, light_resolution, [unique[0]], 3),
        (light_model, light_resolution, [unique[1], unique[2]], 3),
        (heavy_model, heavy_resolution, [unique[3], unique[4], unique[5]], 3),
        (heavy_model, heavy_resolution, [unique[6], unique[7], shared[0], shared[1]], 3),
        # Liveness control group: plain jobs that must keep flowing while LoRA jobs download.
        (light_model, light_resolution, [], 12),
        (heavy_model, heavy_resolution, [], 8),
    ]

    image_jobs = [
        CannedImageJobSpec(
            model=model,
            width=resolution,
            height=resolution,
            lora_names=list(lora_names),
            count=weight,
        )
        for model, resolution, lora_names, weight in profiles
    ]

    return Scenario(name="lora_storm", image_jobs=image_jobs, soak_seconds=soak_seconds)


PRODUCTION_REPLAY_SDXL_MODELS: tuple[str, ...] = (
    "WAI-NSFW-illustrious-SDXL",
    "CyberRealistic Pony",
    "AlbedoBase XL (SDXL)",
    "Juggernaut XL",
)
"""The SDXL-family checkpoints of the measured fingerprint, dominated by ``WAI-NSFW-illustrious-SDXL``."""

PRODUCTION_REPLAY_SD15_MODELS: tuple[str, ...] = (
    "Abyss OrangeMix",
    "stable_diffusion",
)
"""The SD1.5/other checkpoints of the measured fingerprint (the small-image band)."""

PRODUCTION_REPLAY_FLUX_MODELS: tuple[str, ...] = ("Flux.1-Schnell fp8 (Compact)",)
"""The Flux checkpoint of the fingerprint: a rare, heavy model load interleaved into the stream."""


@dataclasses.dataclass(frozen=True)
class _ReplayProfile:
    """One weighted job shape in the ``production_replay`` mix, minting identical jobs at its draw weight."""

    model: str
    width: int
    height: int
    steps: int
    weight: int
    n_iter: int = 1
    hires_fix: bool = False
    lora_names: tuple[str, ...] = ()
    ti_names: tuple[str, ...] = ()
    post_processing: tuple[str, ...] = ()


def build_production_replay_soak_scenario(
    *,
    soak_seconds: float,
    shared_lora_references: Sequence[str] = _PRODUCTION_REPLAY_SHARED_REFERENCES,
    unique_lora_references: Sequence[str] = _PRODUCTION_REPLAY_UNIQUE_REFERENCES,
) -> Scenario:
    """Build the ``production_replay`` soak: a cadence-faithful replay of measured production traffic.

    Where ``lora_storm`` is the adversarial download-pressure mix, this mix reproduces the shape of the
    real pop stream so an acceptance soak runs the worker under the traffic it will actually meet. The
    template weights are draw weights (the soak generates jobs continuously rather than expanding a fixed
    list), tuned so the pop-count distribution approximates a fingerprint measured over 4148 real pops
    across three weeks of one worker's logs:

    - **Family share by pop count**: SDXL-family ~71%, SD1.5/other ~26%, Flux ~3%. The four SDXL
      checkpoints carry the bulk, dominated by ``WAI-NSFW-illustrious-SDXL`` (~28% of all pops on its own);
      two SD1.5 checkpoints carry the small-image band; one Flux checkpoint carries the rare heavy load.
    - **Job size (effective megapixel-steps)**: the measured quantiles are p10=7, p50=31, p90=61, p99=88,
      max=156. The templates span that range: small SD1.5 jobs (512x512 / 512x768 at 20-25 steps), the
      SDXL bulk at ~1024x1024 and 25-30 steps, a heavy band (1216x832 or batched 1024x1024 at 30+ steps,
      some with hires_fix), and a rare pathological rung (~1% weight, 1024x1024 at 50 steps, batch 2,
      several LoRAs) matching the observed max.
    - **Batch distribution**: ~75% batch 1, ~20% batch 2, ~4% batch 3, ~1% batch 4, plus a rare batch 8
      placed on one small-image template so the wide-batch path is exercised without dwarfing the run.
    - **LoRAs on ~23% of pops**, biased toward the SDXL templates; LoRAs together with batch>=2 on ~10%.
    - **Post-processing on ~37% of pops**: ``RealESRGAN_x4plus`` on most (the common upscale), ``GFPGAN``
      on some (the face-fix pattern), spread across families.

    The LoRA-bearing templates draw from both pools the way the storm does: repeated references from the
    shared pool for cache hits, distinct references from the unique pool for download pressure. The default
    pools are synthetic and only meaningful for a fake-mode run; a real-mode soak must supply resolvable
    references (at least 3 shared, at least 8 unique) or the modest LoRA share would measure the failure
    path instead of production cadence. No arrival scheduling is imposed: cadence emerges from the job
    durations and the queue depth, as it does in production.
    """
    _validate_lora_reference_pools("production_replay", shared_lora_references, unique_lora_references)

    shared = tuple(shared_lora_references)
    unique = tuple(unique_lora_references)

    wai, cyber, albedo, juggernaut = PRODUCTION_REPLAY_SDXL_MODELS
    abyss, sd15 = PRODUCTION_REPLAY_SD15_MODELS
    (flux,) = PRODUCTION_REPLAY_FLUX_MODELS

    upscale = (KNOWN_UPSCALERS.RealESRGAN_x4plus.value,)
    facefix = (KNOWN_FACEFIXERS.GFPGAN.value,)

    # Weights are pop-count targets summing to 100 so each equals a percent of the stream. The comment on
    # each band names the fingerprint feature it reproduces; per-template comments name the rare rungs.
    profiles: list[_ReplayProfile] = [
        # SD1.5 / other small-image band: ~26% of pops, the p10-p50 job-size floor.
        _ReplayProfile(abyss, 512, 512, 20, weight=4),
        _ReplayProfile(abyss, 512, 768, 25, weight=4, post_processing=facefix),
        _ReplayProfile(sd15, 512, 512, 20, weight=6),
        _ReplayProfile(sd15, 512, 512, 20, weight=2, n_iter=8),  # rare wide batch on a small image
        _ReplayProfile(abyss, 512, 768, 25, weight=5, n_iter=2),
        _ReplayProfile(abyss, 512, 512, 22, weight=5, post_processing=upscale),
        # WAI band: the dominant checkpoint, ~28% of pops on its own.
        _ReplayProfile(wai, 1024, 1024, 25, weight=6),
        _ReplayProfile(wai, 1024, 1024, 30, weight=8, lora_names=(shared[0],)),
        _ReplayProfile(wai, 1024, 1024, 25, weight=6, post_processing=upscale),
        _ReplayProfile(wai, 1216, 832, 30, weight=4, n_iter=2, hires_fix=True, lora_names=(shared[1],)),
        _ReplayProfile(wai, 1024, 1024, 28, weight=4, post_processing=facefix),
        # CyberRealistic Pony band: SDXL bulk with a download-pressure LoRA template.
        _ReplayProfile(cyber, 1024, 1024, 28, weight=3),
        _ReplayProfile(cyber, 1024, 1024, 30, weight=5, lora_names=(unique[0],)),
        _ReplayProfile(cyber, 1024, 1024, 28, weight=5, post_processing=upscale),
        # AlbedoBase XL band: carries the rare TI path and the batch-3 rung.
        _ReplayProfile(albedo, 1024, 1024, 25, weight=6, post_processing=upscale),
        _ReplayProfile(albedo, 1024, 1024, 30, weight=2, ti_names=(shared[2],)),  # rare TI path
        _ReplayProfile(albedo, 1024, 1024, 28, weight=3, n_iter=3),  # batch-3 rung
        _ReplayProfile(albedo, 1024, 1024, 26, weight=2, post_processing=upscale),
        # Juggernaut XL band: the heavy hires band, the batch-4 rung, and the pathological max rung.
        _ReplayProfile(juggernaut, 1024, 1024, 28, weight=6),
        _ReplayProfile(juggernaut, 1216, 832, 32, weight=5, n_iter=2, hires_fix=True, post_processing=upscale),
        _ReplayProfile(juggernaut, 1024, 1024, 30, weight=4, n_iter=2, lora_names=(shared[2], unique[1])),
        _ReplayProfile(juggernaut, 1024, 1024, 28, weight=1, n_iter=4),  # batch-4 rung
        _ReplayProfile(  # pathological max rung: the observed p99-to-max job
            juggernaut,
            1024,
            1024,
            50,
            weight=1,
            n_iter=2,
            lora_names=(unique[2], unique[3], unique[4], shared[0]),
        ),
        # Flux band: ~3% of pops, the rare heavy model load interleaved into the stream (no LoRAs).
        _ReplayProfile(flux, 1024, 1024, 6, weight=3),
    ]

    image_jobs = [
        CannedImageJobSpec(
            model=profile.model,
            width=profile.width,
            height=profile.height,
            steps=profile.steps,
            n_iter=profile.n_iter,
            hires_fix=profile.hires_fix,
            lora_names=list(profile.lora_names),
            ti_names=list(profile.ti_names),
            post_processing=list(profile.post_processing),
            count=profile.weight,
        )
        for profile in profiles
    ]

    return Scenario(name="production_replay", image_jobs=image_jobs, soak_seconds=soak_seconds)


SoakMix = Literal["production_replay", "lora_storm"]
"""The named sustained-traffic mixes a standalone soak can run."""

SOAK_MIXES: tuple[SoakMix, ...] = ("production_replay", "lora_storm")
"""``SoakMix`` values in CLI choice order; the first is the default (production cadence)."""

OPERATOR_THROUGHPUT_BRIDGE_FIELDS: tuple[str, ...] = (
    "max_threads",
    "queue_size",
    "high_performance_mode",
    "moderate_performance_mode",
    "safety_on_gpu",
    "unload_models_from_vram_often",
    "legacy_comfy_vram_unload",
    "whole_card_exclusive_residency",
    "gpu_sampling_lease_enabled",
    "dedicated_post_processing",
    "enable_pipeline_disaggregation",
    "gpu_device_indices",
    "vram_to_leave_free",
    "ram_to_leave_free",
)
"""Config fields a standalone soak carries over from the operator's ``bridgeData.yaml``.

The set is the throughput-shaping half of the config: concurrency, residency and lane policy, plus the
device selection those are measured on. Everything else the harness owns, because it is derived from the
workload rather than chosen by the operator: ``models_to_load`` and ``max_power`` come from the mix (an
operator ``max_power`` below what a template needs would make the job ineligible and fault it at
dispatch), the ``allow_*`` capability flags come from the templates' features, and the api/dry-run keys
are the harness's own. Only keys the file actually sets are carried, so an unwritten field keeps its
worker default rather than being pinned to one.
"""

_DEFAULT_BRIDGE_DATA_PATH = Path("bridgeData.yaml")


def operator_throughput_overrides(config_path: Path = _DEFAULT_BRIDGE_DATA_PATH) -> dict[str, object]:
    """Read the throughput-relevant fields the operator set in ``bridgeData.yaml``.

    Parses the file directly rather than through ``BridgeDataLoader`` so nothing is defaulted, coerced or
    resolved on the way through: a soak arm has to be able to say which values it pinned, and a loader
    would hand back a full config in which an operator's choice is indistinguishable from a default. Keys
    absent from the file are absent from the result.

    A missing or unreadable config yields an empty mapping: the soak then runs on harness defaults, which
    is a valid (if less representative) measurement, so it must not be a hard failure.
    """
    if not config_path.is_file():
        return {}
    try:
        from ruamel.yaml import YAML

        data = YAML(typ="safe").load(config_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - an unreadable config degrades the run's fidelity, not its validity
        return {}
    if not isinstance(data, dict):
        return {}
    parsed: dict[str, Any] = data
    return {field: parsed[field] for field in OPERATOR_THROUGHPUT_BRIDGE_FIELDS if field in parsed}


def build_soak_mix_scenario(
    mix: SoakMix,
    *,
    soak_seconds: float,
    shared_lora_references: Sequence[str] | None = None,
    unique_lora_references: Sequence[str] | None = None,
    include_auxiliary_references: bool = True,
) -> Scenario:
    """Build one named soak mix, optionally with its LoRA/TI references stripped.

    ``include_auxiliary_references=False`` drops every template's LoRA and textual-inversion references
    while keeping the mix's model, resolution, batch and post-processing shape intact. Both classes go
    together because both resolve through the same CivitAI prefetch path, so leaving either in place with
    the builders' synthetic default names would measure the resolution-failure path in a real-mode run.
    The result is a slightly cheaper mix than production (no LoRA load, no download pressure), which is
    the caveat a before/after comparison run this way has to carry.
    """
    shared = shared_lora_references if shared_lora_references is not None else None
    unique = unique_lora_references if unique_lora_references is not None else None
    kwargs: dict[str, Any] = {"soak_seconds": soak_seconds}
    if shared is not None:
        kwargs["shared_lora_references"] = shared
    if unique is not None:
        kwargs["unique_lora_references"] = unique

    if mix == "lora_storm":
        scenario = build_lora_storm_soak_scenario(**kwargs)
    else:
        scenario = build_production_replay_soak_scenario(**kwargs)

    if include_auxiliary_references:
        return scenario
    return strip_auxiliary_references(scenario)


def strip_auxiliary_references(scenario: Scenario) -> Scenario:
    """Return *scenario* with every image template's LoRA and textual-inversion references removed."""
    return scenario.model_copy(
        update={
            "image_jobs": [spec.model_copy(update={"lora_names": [], "ti_names": []}) for spec in scenario.image_jobs],
        },
    )


__all__ = [
    "OPERATOR_THROUGHPUT_BRIDGE_FIELDS",
    "SOAK_MIXES",
    "SoakMix",
    "build_lora_storm_soak_scenario",
    "build_production_replay_soak_scenario",
    "build_soak_mix_scenario",
    "build_soak_scenario",
    "operator_throughput_overrides",
    "strip_auxiliary_references",
]
