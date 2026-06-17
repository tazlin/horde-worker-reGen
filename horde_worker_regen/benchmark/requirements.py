"""Per-level resource requirements and the machine-fit verdict, computed read-only from a level.

A benchmark level declares *what* work it offers (see :mod:`horde_worker_regen.benchmark.scenarios`).
This module derives *what that work needs* (VRAM, disk, downloads, a CivitAI token) without ever
touching the scenario's content, so the same numbers drive three surfaces that must agree:

- the ``horde-benchmark plan`` subcommand (a dry preview, no worker boot),
- the controller's runtime pre-flight (``_pre_flight_skip_reason``), and
- the TUI's plan pane.

Keeping the computation here (and out of the controller) is what guarantees the preview an operator
sees matches the skip decision the ramp actually makes. The benchmark stays apples-to-apples across
machines: requirements only decide *whether* a level runs, never *what* it runs.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.ladder import (
    BETA_TIERS,
    HUGE_TIERS,
    RampLevel,
)

if TYPE_CHECKING:
    from pathlib import Path

    from horde_worker_regen.benchmark.enums import BenchTier
    from horde_worker_regen.benchmark.report import MachineInfo

_CIVITAI_TOKEN_ENV_VARS = ("CIVIT_API_TOKEN", "AIWORKER_CIVITAI_API_TOKEN")
"""Env vars that carry a CivitAI token. ``load_env_vars`` populates ``CIVIT_API_TOKEN`` from the
config's ``civitai_api_token``; the Docker images use ``AIWORKER_CIVITAI_API_TOKEN``."""


class LevelRequirements(BaseModel):
    """What one benchmark level needs to run, derived read-only from its scenario."""

    level_id: str
    stage: str
    tier: str
    axis: str
    baseline: str
    estimated_vram_mb: int | None = None
    """Estimated peak VRAM for the level's heaviest job (hordelib burden), or None when unavailable."""
    min_disk_free_gb: float = 0.0
    estimated_download_bytes: int | None = None
    """Informational: the tier checkpoint's on-disk size for huge tiers (what a fresh fetch would cost)."""
    models_required: list[str] = Field(default_factory=list)
    models_missing: list[str] = Field(default_factory=list)
    """The subset of ``models_required`` confirmed absent on disk (indeterminate ones are omitted)."""
    requires_network: bool = False
    requires_civitai_key: bool = False
    """True when a job pulls loras/TIs, which are fetched from CivitAI and may need a token."""
    features: list[str] = Field(default_factory=list)
    """Human-readable feature tags exercised by the level (hires_fix, controlnet, post_processing, ...)."""


def civitai_token_available() -> bool:
    """Whether a CivitAI token is configured in this process's environment (best-effort)."""
    return any(os.getenv(var) for var in _CIVITAI_TOKEN_ENV_VARS)


def model_present_on_disk(model_name: str) -> bool | None:
    """Whether *model_name*'s files are all on disk, or None when it cannot be determined.

    Uses the parent-owned reference (the parent is the only process allowed to fetch references) and
    the existing :func:`is_model_present` existence check. Fails open (returns None) on any error.
    """
    try:
        from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

        from horde_worker_regen.model_download_plan import is_model_present
        from horde_worker_regen.reference_helper import ensure_model_reference_manager_initialized

        manager = ensure_model_reference_manager_initialized()
        records = manager.get_all_model_references().get(MODEL_REFERENCE_CATEGORY.image_generation) or {}
        return is_model_present(model_name, records)
    except Exception as e:  # noqa: BLE001 - presence is best-effort; fail open
        logger.debug(f"Could not determine on-disk presence of {model_name!r}: {e}")
        return None


def _estimate_vram_mb(level: RampLevel) -> int | None:
    """Estimate the level's heaviest-job VRAM via the hordelib burden registry, or None on error."""
    try:
        from hordelib.api import estimate_job_burden

        burden = estimate_job_burden(
            baseline=level.baseline_hordelib,
            width=max((job.width for job in level.scenario.image_jobs), default=512),
            height=max((job.height for job in level.scenario.image_jobs), default=512),
            batch=max((job.n_iter for job in level.scenario.image_jobs), default=1),
        )
        return burden.vram_mb
    except Exception as e:  # noqa: BLE001 - estimate is informational; never blocks
        logger.debug(f"Burden estimate unavailable for {level.id}: {e}")
        return None


def _tier_download_bytes(tier: BenchTier) -> int | None:
    """The tier checkpoint's declared download size (huge tiers only), or None when unavailable."""
    if tier not in HUGE_TIERS:
        return None
    try:
        from hordelib.api import estimate_job_burden

        from horde_worker_regen.benchmark.ladder import _TIER_BASELINES, _TIER_RESOLUTIONS

        resolution = _TIER_RESOLUTIONS[tier]
        burden = estimate_job_burden(baseline=_TIER_BASELINES[tier], width=resolution, height=resolution, batch=1)
        return burden.disk_bytes_needed
    except Exception as e:  # noqa: BLE001 - informational only
        logger.debug(f"Download-size estimate unavailable for {tier}: {e}")
        return None


def _level_features(level: RampLevel) -> list[str]:
    """Human-readable feature tags the level exercises, derived from its image jobs."""
    jobs = level.scenario.image_jobs
    features: list[str] = []
    if any(job.hires_fix for job in jobs):
        features.append("hires_fix")
    if any(job.control_type for job in jobs):
        features.append("controlnet")
    if any(job.workflow for job in jobs):
        features.append("qr_code")
    if any(job.post_processing for job in jobs):
        features.append("post_processing")
    if any(job.lora_names for job in jobs):
        features.append("loras")
    if any(job.ti_names for job in jobs):
        features.append("ti")
    if any(job.n_iter > 1 for job in jobs):
        features.append("batch")
    if level.scenario.alchemy_forms:
        features.append("alchemy")
    return features


def compute_level_requirements(
    level: RampLevel,
    *,
    present_resolver: Callable[[str], bool | None] | None = None,
) -> LevelRequirements:
    """Derive the read-only resource requirements of *level*.

    Args:
        level: The ladder level to inspect (never mutated).
        present_resolver: Returns whether a model's files are on disk (True/False/None=unknown);
            defaults to :func:`model_present_on_disk`. Injectable so the ``plan`` subcommand and tests
            can supply a cheap or fixed resolver instead of touching the reference manager.
    """
    resolver = present_resolver if present_resolver is not None else model_present_on_disk
    models_required = level.scenario.models_referenced()
    models_missing = [name for name in models_required if resolver(name) is False]
    requires_civitai_key = any(job.lora_names or job.ti_names for job in level.scenario.image_jobs)
    return LevelRequirements(
        level_id=level.id,
        stage=str(level.stage),
        tier=str(level.tier),
        axis=str(level.axis),
        baseline=level.baseline_hordelib,
        estimated_vram_mb=_estimate_vram_mb(level),
        min_disk_free_gb=level.criteria.min_disk_free_gb,
        estimated_download_bytes=_tier_download_bytes(level.tier),
        models_required=models_required,
        models_missing=models_missing,
        requires_network=level.requires_network,
        requires_civitai_key=requires_civitai_key,
        features=_level_features(level),
    )


def requirement_skip_reason(
    req: LevelRequirements,
    *,
    machine: MachineInfo,
    process_mode: str,
    cache_path: Path,
    civitai_available: bool,
    force: bool = False,
) -> str | None:
    """Return why *req* cannot run on this machine, or None to proceed.

    Covers only the per-level *resource* gates (disk, model presence, VRAM, CivitAI key); the dynamic
    ramp gates (failed-baseline/axis cascades, ``--skip-downloads``, the empty-weights-root guard) stay
    with the controller, which calls this after them. ``force`` bypasses the machine-fit and key gates
    (insufficient VRAM/disk, missing CivitAI key) but never the absent-checkpoint gate: there is simply
    nothing to run when the weights are not present.

    Resource gates apply only in ``real`` mode; ``fake``/``dry_run`` download and infer nothing.
    """
    if process_mode != "real":
        return None

    if not force:
        free_disk = shutil.disk_usage(cache_path).free
        if free_disk < req.min_disk_free_gb * 1024**3:
            return f"insufficient disk on {cache_path}: {free_disk / 1024**3:.1f} GB free"

    # A genuinely-absent huge/beta checkpoint is a hard skip even under --force: real-mode benchmarking
    # never downloads checkpoints, so there is nothing to run.
    if req.models_missing and _tier_is_huge(req.tier):
        beta_hint = (
            " (a beta model: set HORDE_MODEL_REFERENCE_PRIMARY_API_URL and await publication)"
            if _tier_is_beta(req.tier)
            else " (real-mode benchmarking does not download checkpoints)"
        )
        missing = ", ".join(repr(name) for name in req.models_missing)
        return f"{req.tier} model {missing} is not present on disk{beta_hint}"

    if (
        not force
        and req.estimated_vram_mb is not None
        and machine.total_vram_mb
        and req.estimated_vram_mb > machine.total_vram_mb
    ):
        return f"insufficient VRAM: estimated {req.estimated_vram_mb} MB needed, {machine.total_vram_mb} MB available"

    if not force and req.requires_civitai_key and not civitai_available:
        return (
            "requires a CivitAI API token for lora/TI downloads (set `civitai_api_token` in bridgeData.yaml "
            "or export CIVIT_API_TOKEN)"
        )

    return None


def _tier_is_huge(tier: str) -> bool:
    """Whether the (stringified) tier is one of the huge-download tiers."""
    return any(tier == str(huge) for huge in HUGE_TIERS)


def _tier_is_beta(tier: str) -> bool:
    """Whether the (stringified) tier is sourced from the beta/pending reference."""
    return any(tier == str(beta) for beta in BETA_TIERS)


__all__ = [
    "LevelRequirements",
    "civitai_token_available",
    "compute_level_requirements",
    "model_present_on_disk",
    "requirement_skip_reason",
]
