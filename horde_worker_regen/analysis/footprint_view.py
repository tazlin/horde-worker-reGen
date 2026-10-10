"""Offline view of a learned VRAM footprint store, in the figures the worker prices sampling from.

A support bundle carries the worker's ``vram_footprints.json``. Reading which sampling keys price high, and
against which resident weights, otherwise means loading the file by hand. This module loads it through
:class:`~horde_worker_regen.process_management.resources.vram_footprints.LearnedFootprintStore` and reads each
figure through the store's own estimate methods, so the view moves with the production statistic.

- :func:`learned_sampling_footprints`: one row per sampling key with its two pricing terms, the resident
  figures for its baseline and the activation watermark for its band.
- :func:`render_learned_sampling_footprints`: the terminal rendering.
- :func:`default_footprint_store_path`: the bundle's copy beside a bundle's ``stats/``, else the worker's store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from horde_worker_regen.app_state import default_app_state_dir
from horde_worker_regen.process_management.resources.resource_budget import platform_context_constant_mb
from horde_worker_regen.process_management.resources.vram_footprints import (
    FOOTPRINT_STORE_FILENAME,
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
)

_BUNDLE_CONFIG_DIRNAME = "config"
"""The support-bundle directory holding the copied footprint store, beside its ``stats/``."""

_SAMPLING_STAGES = (FootprintStage.SAMPLE, FootprintStage.SAMPLE_ISOLATED)
"""The stages ``pricing.learned_sampling_peak_mb`` prices a sampling job from."""

_LISTED_STAGES = (*_SAMPLING_STAGES, FootprintStage.POST_PROCESS)
"""The stages listed, which are the sampling stages and the post-processing stage chains are booked from."""


@dataclass
class ResidentFootprint:
    """Represents one checkpoint's learned resident figure, net of the context charge."""

    checkpoint: str
    resident_mb: float


@dataclass
class SamplingFootprintRow:
    """Represents one sampling key and the figures the worker prices it from.

    ``watermark_mb`` is the raise-only term (``estimate_mb`` with no seed); a job's seed can only raise it.
    ``measured_mb`` is ``measured_estimate_mb`` net of the context charge, the figure that lowers a job's
    price when it is the smaller one; None while the key's window is under-observed.
    """

    baseline: str
    resolution_bucket: str | None
    stage: str
    job_class: str
    """``all`` for the pooled key, else ``b<batch>`` with `` hires`` appended for a hires-fix pass. A chain's
    operations for a post-processing key."""
    platform: str
    observation_count: int
    window_size: int
    watermark_mb: float
    measured_mb: float | None
    activation_watermark_mb: float | None
    """The ``sample_activation`` watermark for the same baseline and band, or None when unobserved."""
    residents: list[ResidentFootprint] = field(default_factory=list)
    """Learned resident figures for every checkpoint of the same baseline, largest first."""


def default_footprint_store_path(stats_dir: Path | None) -> Path:
    """Return the footprint store a duty report reads when none is named.

    Args:
        stats_dir: The ``--stats`` directory, or None for the worker's own.

    Returns:
        The bundle's ``config/vram_footprints.json`` when ``stats_dir`` sits in a support bundle that has one,
        else the store in the worker's app-state directory.
    """
    if stats_dir is not None:
        bundle_copy = stats_dir.parent / _BUNDLE_CONFIG_DIRNAME / FOOTPRINT_STORE_FILENAME
        if bundle_copy.is_file():
            return bundle_copy
    return default_app_state_dir() / FOOTPRINT_STORE_FILENAME


def learned_sampling_footprints(path: Path) -> list[SamplingFootprintRow]:
    """Load the footprint store at ``path`` and return one row per sampling key, read-only.

    The store loads the file exactly as the worker does, so a key a worker would discard (another schema
    version, a malformed entry) is absent here too. Nothing is written back.

    Args:
        path: A ``vram_footprints.json`` file.

    Returns:
        Sampling rows sorted by baseline, band and stage. Empty for a missing or unreadable file.
    """
    store = LearnedFootprintStore(path=path)
    keys = [key for key in _persisted_keys(path) if store.get_observation(key) is not None]
    rows: list[SamplingFootprintRow] = []
    for key in keys:
        if key.stage not in _LISTED_STAGES:
            continue
        observation = store.get_observation(key)
        if observation is None:
            continue
        rows.append(
            SamplingFootprintRow(
                baseline=key.model_baseline,
                resolution_bucket=None if key.resolution_bucket is None else key.resolution_bucket.value,
                stage=key.stage.value,
                job_class=_job_class_label(key),
                platform=key.platform,
                observation_count=observation.observation_count,
                window_size=len(observation.recent_mb),
                watermark_mb=store.estimate_mb(key, static_seed_mb=0.0),
                measured_mb=store.measured_estimate_net_of_context_mb(key),
                activation_watermark_mb=_activation_watermark_mb(store, key),
                residents=_residents_for(store, keys, key),
            ),
        )
    rows.sort(key=lambda row: (row.baseline, row.resolution_bucket or "", row.stage, row.job_class))
    return rows


def _job_class_label(key: FootprintKey) -> str:
    if key.stage is FootprintStage.POST_PROCESS:
        return key.checkpoint or "?"
    if key.batch is None:
        return "all"
    return f"b{key.batch}{' hires' if key.hires_fix else ''}"


def render_learned_sampling_footprints(path: Path, rows: list[SamplingFootprintRow]) -> str:
    """Render the sampling rows for terminal output, figures only."""
    if not rows:
        return f"== Learned footprints ({path}) ==\n   no sampling keys"
    out = [
        f"== Learned footprints ({path}) ==",
        "   watermark: raise-only term; measured: margined window figure net of the context charge, "
        "which prices a job when lower; resident: net of the context charge",
    ]
    for row in rows:
        measured = (
            f"measured {row.measured_mb:.0f}MB"
            if row.measured_mb is not None
            else f"measured n/a ({row.window_size} in window)"
        )
        activation = "n/a" if row.activation_watermark_mb is None else f"{row.activation_watermark_mb:.0f}MB"
        out.append(
            f"   {row.baseline} {row.resolution_bucket or '-'} {row.stage} [{row.job_class}]: "
            f"watermark {row.watermark_mb:.0f}MB, "
            f"{measured} ({row.observation_count} obs); activation watermark {activation}; "
            f"resident {_render_residents(row.residents)}"
        )
    return "\n".join(out)


def _persisted_keys(path: Path) -> list[FootprintKey]:
    """Return the keys named in the file; the store holds no public key listing, the numbers come from it."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = raw.get("observations") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        return []
    keys: list[FootprintKey] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            keys.append(FootprintKey.model_validate(entry.get("key")))
        except ValueError:
            continue
    return keys


def _activation_watermark_mb(store: LearnedFootprintStore, key: FootprintKey) -> float | None:
    activation_key = key.model_copy(update={"stage": FootprintStage.SAMPLE_ACTIVATION})
    if store.get_observation(activation_key) is None:
        return None
    return store.estimate_mb(activation_key, static_seed_mb=0.0)


def _residents_for(
    store: LearnedFootprintStore,
    keys: list[FootprintKey],
    sampling_key: FootprintKey,
) -> list[ResidentFootprint]:
    """Return the resident figures for the sampling key's baseline, as ``learned_resident_footprint_mb`` nets them."""
    context_mb = platform_context_constant_mb(platform=sampling_key.platform)
    residents = [
        ResidentFootprint(
            checkpoint=key.checkpoint or "?",
            resident_mb=max(0.0, store.estimate_mb(key, static_seed_mb=0.0) - context_mb),
        )
        for key in keys
        if key.stage is FootprintStage.RESIDENT
        and key.model_baseline == sampling_key.model_baseline
        and key.platform == sampling_key.platform
    ]
    residents.sort(key=lambda resident: resident.resident_mb, reverse=True)
    return residents


def _render_residents(residents: list[ResidentFootprint]) -> str:
    if not residents:
        return "n/a"
    if len(residents) == 1:
        return f"{residents[0].resident_mb:.0f}MB ({residents[0].checkpoint})"
    return f"{residents[-1].resident_mb:.0f}-{residents[0].resident_mb:.0f}MB over {len(residents)} checkpoints"


__all__ = [
    "ResidentFootprint",
    "SamplingFootprintRow",
    "default_footprint_store_path",
    "learned_sampling_footprints",
    "render_learned_sampling_footprints",
]
