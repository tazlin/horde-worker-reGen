"""Age-out and size-cap purge for the worker's on-disk stats-export directory.

The stats exporter writes one size-rotated ``stats-v*.jsonl`` family per session under
``.horde_worker_regen/stats`` and, with autozip enabled, compresses inactive files to ``.jsonl.gz``.
Left unbounded, a worker run every day would accumulate those indefinitely. This module bounds the
directory the same way :mod:`horde_worker_regen.logging_purge` bounds ``logs/``: a single startup sweep
that ages files out, then trims oldest-first to a size budget.

It shares the audited delete guard in :mod:`horde_worker_regen._guarded_purge` (non-recursive, symlink
refusing, parent re-verifying, single-file unlink only) and supplies only the stats-file recognizer
(:func:`horde_worker_regen.stats_operations.is_worker_stats_file`). Deletion is therefore fail-closed:
only ``stats-v*.jsonl``/``.jsonl.gz`` files are ever eligible, so a foreign file an operator drops in the
stats directory, a leftover ``.tmp`` from an interrupted compression, or a nested folder is never touched.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from horde_worker_regen._guarded_purge import PurgeResult, guarded_purge_directory
from horde_worker_regen.stats_operations import default_stats_dir, is_worker_stats_file


def purge_stats_directory(
    stats_dir: Path | str | None = None,
    *,
    max_age_days: float,
    max_total_gb: float,
) -> PurgeResult:
    """Delete aged-out and over-budget stats files from *stats_dir*.

    Files last modified more than ``max_age_days`` ago are removed first; then, if the surviving files
    still total more than ``max_total_gb``, the oldest are removed until the directory fits. A limit of 0
    (or less) disables that stage. A missing directory is a no-op. Only recognized worker stats files are
    ever eligible for deletion (see
    :func:`horde_worker_regen.stats_operations.is_worker_stats_file`).

    Args:
        stats_dir: The worker's stats directory; defaults to ``.horde_worker_regen/stats``.
        max_age_days: Delete files older than this many days. 0 disables the age-out.
        max_total_gb: Cap the directory at this many gigabytes. 0 disables the size cap.

    Returns:
        A :class:`horde_worker_regen._guarded_purge.PurgeResult` summarising what was removed.
    """
    directory = Path(stats_dir) if stats_dir is not None else default_stats_dir()
    return guarded_purge_directory(
        directory,
        recognizer=is_worker_stats_file,
        max_age_days=max_age_days,
        max_total_gb=max_total_gb,
        label="Stats purge",
    )


def purge_worker_stats_safely(
    *,
    max_age_days: float,
    max_total_gb: float,
    stats_dir: Path | str | None = None,
) -> None:
    """Run :func:`purge_stats_directory`, swallowing any error so stats hygiene can never block startup."""
    try:
        purge_stats_directory(stats_dir, max_age_days=max_age_days, max_total_gb=max_total_gb)
    except Exception as purge_error:
        logger.warning(f"Stats purge failed ({type(purge_error).__name__}: {purge_error}); continuing startup.")
