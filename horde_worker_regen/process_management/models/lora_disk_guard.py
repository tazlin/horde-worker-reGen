"""Disk-space fail-safes for the on-demand LoRA cache.

The LoRA manager downloads CivitAI LoRAs on demand into a size-bounded cache. Its byte budget
(``max_lora_cache_size``) is the primary bound, but a mis-set budget, a volume shared with other
data, or a stale hordelib can still let the cache approach a full disk, where every weight write
fails with ENOSPC and can take co-located worker data down with it.

This module adds a disk-space floor on top of the byte budget, split across the two processes that
matter:

* In the inference subprocess (which owns the live :class:`LoraModelManager`),
  :func:`constrain_lora_cache_to_disk` runs before each LoRA-bearing job: it shrinks the effective
  ad-hoc budget to fit the free space and evicts least-recently-used ad-hoc LoRAs until the floor is
  clear, making room for the job's LoRAs. It relies only on the manager's stable public surface, so
  it works against the currently published hordelib as well as newer builds.

* In the main process (which has no manager), :func:`is_lora_disk_exhausted` decides whether to stop
  advertising LoRA support at all. It disables LoRAs only when even evicting every ad-hoc entry
  could not clear the floor (read from the persisted ``lora.json``), so a recoverable shortfall is
  left to the inference-side eviction rather than latching the worker out of LoRA work.

Deliberately torch-free and hordelib-free at import time so the main (orchestrator) process can use
it without dragging in the inference stack.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from loguru import logger

EnvLookup = Callable[[str], str | None]
"""A ``os.getenv``-style lookup, injected so either process resolves the same env-derived values."""

_BYTES_PER_MB = 1024 * 1024

DEFAULT_MIN_FREE_DISK_MB = 1024
"""Default free-space floor (MB) for the LoRA cache volume when ``AIWORKER_LORA_MIN_DISK_FREE_MB`` is unset.

Mirrors hordelib's own default so the worker and the manager agree when no floor is configured."""

DEFAULT_LORA_CACHE_SIZE_MB = 10 * 1024
"""Default ad-hoc LoRA budget (MB) when ``AIWORKER_LORA_CACHE_SIZE`` is unset, mirroring hordelib."""


def lora_disk_floor_mb_from_env(getenv: EnvLookup) -> float:
    """Return the configured free-space floor (MB), read from ``AIWORKER_LORA_MIN_DISK_FREE_MB``.

    Falls back to :data:`DEFAULT_MIN_FREE_DISK_MB` when unset or unparseable. *getenv* is injected
    (normally ``os.getenv``) so callers in either process resolve the same value the manager sees.
    """
    raw = getenv("AIWORKER_LORA_MIN_DISK_FREE_MB")
    if raw is None:
        return float(DEFAULT_MIN_FREE_DISK_MB)
    try:
        return max(0.0, float(raw))
    except (ValueError, TypeError):
        return float(DEFAULT_MIN_FREE_DISK_MB)


def configured_lora_budget_mb_from_env(getenv: EnvLookup) -> int:
    """Return the configured ad-hoc LoRA budget (MB) from ``AIWORKER_LORA_CACHE_SIZE``.

    Falls back to :data:`DEFAULT_LORA_CACHE_SIZE_MB` when unset or unparseable.
    """
    raw = getenv("AIWORKER_LORA_CACHE_SIZE")
    if raw is not None:
        try:
            return int(raw)
        except (ValueError, TypeError):
            pass
    return DEFAULT_LORA_CACHE_SIZE_MB


@runtime_checkable
class LoraCacheManager(Protocol):
    """The subset of the hordelib LoRA manager that the disk guard drives.

    Declared structurally so this module need not import hordelib (and thus torch). Every member
    here is part of the manager's stable public surface in the currently published build.
    """

    model_folder_path: Path
    """The directory the LoRA weight files are written to (used to sample the cache volume)."""
    max_adhoc_disk: int
    """The ad-hoc cache byte budget, in megabytes; lowered in place to constrain growth."""

    def calculate_adhoc_cache(self) -> float:
        """Return the current total size of ad-hoc cache entries, in megabytes."""
        ...

    def find_oldest_adhoc_entry(self) -> object | None:
        """Return the least-recently-used ad-hoc entry, or ``None`` if there are none left."""
        ...

    def delete_oldest(self) -> None:
        """Evict the least-recently-used ad-hoc entry (file and reference)."""
        ...

    def save_reference_to_disk(self) -> None:
        """Persist the in-memory reference after eviction."""
        ...


@dataclass(frozen=True)
class LoraDiskConstraintResult:
    """The outcome of one :func:`constrain_lora_cache_to_disk` pass."""

    free_mb_before: float | None
    """Free space on the cache volume before constraining, or ``None`` if it could not be sampled."""
    free_mb_after: float | None
    """Free space after eviction, or ``None`` if it could not be sampled."""
    floor_mb: float
    """The free-space floor that was enforced."""
    evicted_count: int
    """How many ad-hoc LoRA entries were evicted this pass."""
    budget_mb_before: int
    """The ad-hoc byte budget (MB) before constraining."""
    budget_mb_after: int
    """The ad-hoc byte budget (MB) after constraining."""
    solved: bool
    """Whether the floor is now clear (or the volume could not be sampled, so no action was needed)."""

    @property
    def acted(self) -> bool:
        """Whether the guard changed anything (evicted entries or moved the budget)."""
        return self.evicted_count > 0 or self.budget_mb_before != self.budget_mb_after


def free_mb(path: Path) -> float | None:
    """Return free space on *path*'s volume in megabytes, or ``None`` if it cannot be sampled."""
    try:
        return shutil.disk_usage(path).free / _BYTES_PER_MB
    except OSError as usage_error:
        logger.warning(f"LoRA disk guard: could not sample free space for {path}: {usage_error}")
        return None


def constrain_lora_cache_to_disk(
    manager: LoraCacheManager,
    *,
    floor_mb: float,
    configured_budget_mb: int,
) -> LoraDiskConstraintResult:
    """Shrink the ad-hoc budget to fit free space and evict ad-hoc LoRAs until the floor is clear.

    Runs before a LoRA-bearing job downloads its LoRAs, so eviction makes room for the incoming
    weights. When free space is healthy the (possibly previously shrunk) budget is restored to its
    configured value. Uses only the manager's stable public surface so it is safe against the
    published hordelib (whose own budget eviction may still be unit-buggy): correctness here rests on
    measured free space and least-recently-used eviction, not the manager's budget arithmetic.

    Args:
        manager: The live LoRA manager for this subprocess.
        floor_mb: Keep at least this many megabytes free on the cache volume.
        configured_budget_mb: The operator-configured ad-hoc budget (MB) to restore to when healthy.

    Returns:
        A :class:`LoraDiskConstraintResult` describing what was sampled, evicted, and left.
    """
    budget_before = manager.max_adhoc_disk
    free_before = free_mb(manager.model_folder_path)

    if floor_mb <= 0 or free_before is None:
        return LoraDiskConstraintResult(
            free_mb_before=free_before,
            free_mb_after=free_before,
            floor_mb=floor_mb,
            evicted_count=0,
            budget_mb_before=budget_before,
            budget_mb_after=budget_before,
            solved=True,
        )

    if free_before >= floor_mb:
        # Healthy: undo any earlier shrink so the cache can use its full configured budget again.
        manager.max_adhoc_disk = configured_budget_mb
        return LoraDiskConstraintResult(
            free_mb_before=free_before,
            free_mb_after=free_before,
            floor_mb=floor_mb,
            evicted_count=0,
            budget_mb_before=budget_before,
            budget_mb_after=configured_budget_mb,
            solved=True,
        )

    # Below the floor: cap the budget at what the volume can actually hold (current ad-hoc footprint
    # plus the negative headroom), so even a fixed hordelib won't immediately regrow into the wall.
    affordable = manager.calculate_adhoc_cache() + (free_before - floor_mb)
    new_budget = int(max(0.0, min(float(configured_budget_mb), affordable)))
    manager.max_adhoc_disk = new_budget

    evicted = 0
    free_after: float | None = free_before
    while free_after is not None and free_after < floor_mb:
        if manager.find_oldest_adhoc_entry() is None:
            break  # No ad-hoc entries left to reclaim; the shortfall is from other data.
        manager.delete_oldest()
        evicted += 1
        free_after = free_mb(manager.model_folder_path)

    if evicted:
        manager.save_reference_to_disk()

    return LoraDiskConstraintResult(
        free_mb_before=free_before,
        free_mb_after=free_after,
        floor_mb=floor_mb,
        evicted_count=evicted,
        budget_mb_before=budget_before,
        budget_mb_after=new_budget,
        solved=free_after is None or free_after >= floor_mb,
    )


def read_evictable_adhoc_mb(lora_reference_path: Path) -> float:
    """Return the total megabytes of ad-hoc LoRA versions recorded in the persisted ``lora.json``.

    This is the amount the inference-side guard could reclaim by eviction (default-set LoRAs are not
    evictable and are excluded). Best-effort: a missing or unreadable reference yields ``0.0``, which
    makes the caller treat a shortfall as unsolvable (the disk-safe default).
    """
    try:
        raw = json.loads(lora_reference_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    if not isinstance(raw, dict):
        return 0.0

    total = 0.0
    for record in raw.values():
        if not isinstance(record, dict):
            continue
        versions = record.get("versions")
        if not isinstance(versions, dict):
            continue
        for version in versions.values():
            if not isinstance(version, dict) or not version.get("adhoc"):
                continue
            size = version.get("size_mb")
            if isinstance(size, (int, float)):
                total += float(size)
    return total


def is_lora_disk_exhausted(*, free_mb_value: float | None, floor_mb: float, evictable_adhoc_mb: float) -> bool:
    """Return whether LoRA support should be suppressed because the cache disk cannot be cleared.

    Exhausted means free space is below the floor *and* evicting every ad-hoc LoRA still would not
    reach it, so the shortfall is structural (e.g. non-LoRA data) rather than something the
    inference-side eviction can recover. A recoverable shortfall returns ``False`` so LoRAs keep
    being offered and the next LoRA job's eviction makes room.
    """
    if free_mb_value is None or floor_mb <= 0:
        return False
    if free_mb_value >= floor_mb:
        return False
    return (free_mb_value + evictable_adhoc_mb) < floor_mb
