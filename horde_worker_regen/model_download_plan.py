"""Type-safe planning of what a model-loading config implies for disk.

Answers, without importing torch/ComfyUI, the questions the TUI, console and model picker need: which
configured models are already on disk, how much disk the configuration will consume, and whether the
target volume can hold it.

Design notes:
    - All canonical on-disk knowledge (where the weights root is, which folder a category uses, where each
      declared file actually sits including components routed to sibling folders, and whether the files are
      present) is delegated to :mod:`horde_model_reference.on_disk_layout`. There is no worker-local
      category/folder/size bridge to keep in step with hordelib any more; supporting a new model category
      is a horde_model_reference concern, not a worker one.
    - Sizes come from the record itself (:attr:`GenericModelRecord.declared_total_size_bytes`), which prefers
      summed per-file sizes and falls back to the declared aggregate.
    - Presence is an EXISTENCE-ONLY check (no SHA256 hashing), so it is fast but "unverified": a present
      file could still be corrupt. The worker's dedicated download process re-validates and remains the
      authority on integrity; this module is for fast, ahead-of-time guidance.
    - Multiple weights roots are supported so a deployment can spread files across disks. Presence searches
      ``[primary_root, *extra_roots]`` and the first existing copy wins; extra roots come from the
      ``extra_model_directories`` argument or, when that is omitted, the ``AIWORKER_EXTRA_MODEL_DIRECTORIES``
      environment variable (an ``os.pathsep``-separated list of weights-root directories).

The model records are the typed pydantic models from :mod:`horde_model_reference.model_reference_records`,
so attribute access is statically analyzable; no dynamic attribute lookup is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    file_paths_for,
    free_bytes_for,
    is_present,
    resolve_weights_root,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from horde_model_reference.model_reference_records import GenericModelRecord

__all__ = [
    "ENV_EXTRA_MODEL_DIRECTORIES",
    "DownloadPlan",
    "ModelDiskInfo",
    "compute_download_plan",
    "extra_model_roots_from_env",
    "free_model_bytes",
    "is_model_present",
]


ENV_EXTRA_MODEL_DIRECTORIES = "AIWORKER_EXTRA_MODEL_DIRECTORIES"
"""Env var naming extra model weights-root directories, ``os.pathsep``-separated, searched after the primary."""


def extra_model_roots_from_env() -> list[Path]:
    """Return the extra weights-root directories declared in :data:`ENV_EXTRA_MODEL_DIRECTORIES`."""
    raw = os.environ.get(ENV_EXTRA_MODEL_DIRECTORIES, "")
    return [Path(entry) for entry in raw.split(os.pathsep) if entry.strip()]


def _coerce_extra_roots(extra_model_directories: Sequence[str | os.PathLike[str]] | None) -> list[Path]:
    """Resolve the extra weights-roots: the explicit argument when given, else the environment."""
    if extra_model_directories is None:
        return extra_model_roots_from_env()
    return [Path(entry) for entry in extra_model_directories]


@dataclass(frozen=True)
class ModelDiskInfo:
    """The on-disk picture for a single configured model."""

    name: str
    category: MODEL_REFERENCE_CATEGORY | None
    """The model's category, or None when the record is missing or its type is unrecognised."""
    size_bytes: int | None
    """Declared size from the record, or None when the record carries no size metadata."""
    on_disk: bool
    """Whether every declared file for the model exists on disk (existence-only, not validated)."""
    target_path: str
    """Where the model's primary file lives (or would be downloaded to); empty when undeterminable."""


@dataclass(frozen=True)
class DownloadPlan:
    """The aggregate disk implications of a resolved model-loading config."""

    models: list[ModelDiskInfo]
    present_bytes: int
    to_download_bytes: int
    total_bytes: int
    free_disk_bytes: int | None
    """Free space on the model volume, or None when it could not be determined."""
    fits: bool
    """Whether the to-download bytes fit in the free space (True when free space is unknown)."""
    shortfall_bytes: int
    """How many bytes short the volume is (0 when it fits or free space is unknown)."""
    unknown_size_models: list[str]
    """Configured models with no size metadata; the byte totals are a lower bound when non-empty."""

    @property
    def num_present(self) -> int:
        """Return how many configured models are already on disk."""
        return sum(1 for model in self.models if model.on_disk)

    @property
    def num_to_download(self) -> int:
        """Return how many configured models are not yet on disk."""
        return sum(1 for model in self.models if not model.on_disk)

    @property
    def sizes_complete(self) -> bool:
        """Return whether every configured model contributed a known size to the totals."""
        return not self.unknown_size_models


def _disk_info_for(
    name: str,
    record: GenericModelRecord | None,
    root: Path,
    extra_roots: Sequence[Path],
) -> ModelDiskInfo:
    """Build the on-disk picture for one model name against already-resolved weights roots."""
    if record is None:
        return ModelDiskInfo(name=name, category=None, size_bytes=None, on_disk=False, target_path="")
    file_paths = file_paths_for(record, root, extra_roots=extra_roots)
    return ModelDiskInfo(
        name=name,
        category=record.category,
        size_bytes=record.declared_total_size_bytes,
        on_disk=is_present(record, root, extra_roots=extra_roots),
        target_path=str(file_paths[0]) if file_paths else "",
    )


def compute_download_plan(
    model_names: list[str],
    reference: Mapping[str, GenericModelRecord],
    *,
    cache_home: str | None = None,
    extra_model_directories: Sequence[str | os.PathLike[str]] | None = None,
) -> DownloadPlan:
    """Compute the disk implications of loading ``model_names`` against the model ``reference``.

    Args:
        model_names: The resolved model names the config will load.
        reference: The model reference (name -> record), as loaded by the worker or
            ``tui.model_catalog.load_image_models``. Any category is accepted; each record's own type
            determines its folder and whether it carries a size.
        cache_home: Override for the model-directory base (defaults to ``AIWORKER_CACHE_HOME``).
        extra_model_directories: Additional weights-root directories to search for already-present files.
            When omitted, :data:`ENV_EXTRA_MODEL_DIRECTORIES` is consulted. New downloads always target the
            primary root.

    Returns:
        A :class:`DownloadPlan` describing per-model presence/size and the aggregate fit.
    """
    root = resolve_weights_root(cache_home)
    extra_roots = _coerce_extra_roots(extra_model_directories)

    infos: list[ModelDiskInfo] = []
    present_bytes = 0
    to_download_bytes = 0
    unknown_size: list[str] = []

    for name in model_names:
        info = _disk_info_for(name, reference.get(name), root, extra_roots)
        infos.append(info)
        if info.size_bytes is None:
            unknown_size.append(name)
        if info.on_disk:
            present_bytes += info.size_bytes or 0
        else:
            to_download_bytes += info.size_bytes or 0

    free_disk_bytes = free_bytes_for(root)
    total_bytes = present_bytes + to_download_bytes
    fits = free_disk_bytes is None or to_download_bytes <= free_disk_bytes
    shortfall_bytes = 0 if fits or free_disk_bytes is None else (to_download_bytes - free_disk_bytes)

    return DownloadPlan(
        models=infos,
        present_bytes=present_bytes,
        to_download_bytes=to_download_bytes,
        total_bytes=total_bytes,
        free_disk_bytes=free_disk_bytes,
        fits=fits,
        shortfall_bytes=shortfall_bytes,
        unknown_size_models=unknown_size,
    )


def free_model_bytes(cache_home: str | None = None) -> int | None:
    """Return free bytes on the volume that holds the model directory, or None when undeterminable."""
    return free_bytes_for(resolve_weights_root(cache_home))


def is_model_present(
    name: str,
    reference: Mapping[str, GenericModelRecord],
    *,
    cache_home: str | None = None,
    extra_model_directories: Sequence[str | os.PathLike[str]] | None = None,
) -> bool:
    """Return whether a single model's declared files all exist on disk (existence-only)."""
    record = reference.get(name)
    if record is None:
        return False
    return is_present(
        record,
        resolve_weights_root(cache_home),
        extra_roots=_coerce_extra_roots(extra_model_directories),
    )
