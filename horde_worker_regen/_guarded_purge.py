"""A fail-closed, non-recursive age-out and size-cap sweep over a flat directory of recognized files.

Several of the worker's on-disk artifact families (log files, exported stats JSONL) accumulate one file
per run and need bounded retention: delete anything past an age limit, then, if the directory still
exceeds a size budget, delete oldest-first until it fits. The mechanics of *safely* deleting are
identical across families and are the delicate part, so they live here once rather than being copied per
family. A caller supplies only a filename ``recognizer`` describing which files in the directory are its
own; everything else is the shared, audited guard.

Deletion is deliberately narrow, because a startup process silently removing files must be provably
incapable of removing the wrong ones. The sweep can only ever delete a file that satisfies *all* of:

* It is a **direct child** of the given directory. Enumeration is non-recursive (``os.scandir`` lists one
  level only) and every deletion re-checks the parent, so nested folders an operator keeps under the
  directory are never inspected, descended, or removed.
* It is a **regular file**, evaluated with ``follow_symlinks=False``. Directories, symlinks, and special
  files are skipped. A symlink is judged as itself and never resolved, so the sweep can never reach
  through one to a target outside the directory.
* It is **positively recognized** by the caller-supplied ``recognizer``. This is *fail-closed*: a file the
  recognizer does not describe is left untouched, not guessed at.

There is no directory removal anywhere in this module: every deletion is a single-file ``unlink()``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

_BYTES_PER_GB = 1024 * 1024 * 1024
_SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class PurgeResult:
    """What a single :func:`guarded_purge_directory` sweep removed, for logging and tests."""

    deleted_files: int = 0
    deleted_bytes: int = 0
    aged_out: int = 0
    size_trimmed: int = 0
    remaining_bytes: int = 0


def _collect_purgeable_files(directory: Path, recognizer: Callable[[str], bool]) -> list[tuple[Path, int, float]]:
    """Return ``(path, size_bytes, mtime)`` for each top-level regular file *recognizer* accepts.

    Non-recursive and symlink-safe by construction: ``os.scandir`` yields only the direct children of
    *directory*, symlinks are skipped outright, and only regular files the recognizer positively accepts
    are returned. A subdirectory (or anything under it) is never descended into.
    """
    collected: list[tuple[Path, int, float]] = []
    with os.scandir(directory) as scan:
        for entry in scan:
            # Judge a symlink as itself and skip it; never resolve or delete through it.
            if entry.is_symlink():
                continue
            # Non-recursive: a subdirectory is skipped whole, never opened or descended.
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            # Fail-closed: only files the recognizer positively accepts are eligible.
            if not recognizer(entry.name):
                continue
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            collected.append((Path(entry.path), stat.st_size, stat.st_mtime))
    return collected


def _try_delete(path: Path, *, expected_parent: Path, label: str) -> bool:
    """Delete *path*, returning whether it went. Re-verifies the file before unlinking.

    The candidate must still be a direct child of *expected_parent* and still a regular, non-symlink
    file at the moment of deletion; anything else is refused. A held handle (Windows) or a race is
    swallowed so a single undeletable file never fails the whole sweep. *label* names the sweep in log
    lines (e.g. ``"Log purge"``).
    """
    if path.parent != expected_parent:
        logger.warning(f"{label} refused {path}: not a direct child of {expected_parent}; skipping.")
        return False
    if path.is_symlink() or not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError as delete_error:
        logger.debug(f"{label} could not delete {path}: {type(delete_error).__name__}: {delete_error}")
        return False


def guarded_purge_directory(
    directory: Path | str,
    *,
    recognizer: Callable[[str], bool],
    max_age_days: float,
    max_total_gb: float,
    label: str,
) -> PurgeResult:
    """Delete aged-out and over-budget recognized files from *directory*.

    Files last modified more than ``max_age_days`` ago are removed first; then, if the surviving files
    still total more than ``max_total_gb``, the oldest are removed until the directory fits. A limit of 0
    (or less) disables that stage. A missing directory is a no-op. Per-file delete errors are swallowed,
    so the sweep is best-effort and never raises for an individual file.

    Args:
        directory: The directory to sweep. Only its direct children are ever considered.
        recognizer: Predicate on a bare filename; only files it accepts are eligible for deletion.
        max_age_days: Delete files older than this many days. 0 disables the age-out.
        max_total_gb: Cap the directory at this many gigabytes. 0 disables the size cap.
        label: Short human-readable name for the sweep, used in log lines (e.g. ``"Log purge"``).

    Returns:
        A :class:`PurgeResult` summarising what was removed.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return PurgeResult()

    # One consistent, non-recursive, symlink-safe snapshot that both stages reason over. Every path here
    # is a top-level regular recognized file of *directory* (see _collect_purgeable_files); nothing else
    # is even a candidate, and every deletion below re-verifies the parent against this same directory.
    entries = _collect_purgeable_files(directory, recognizer)

    deleted_files = 0
    deleted_bytes = 0
    aged_out = 0

    survivors: list[tuple[Path, int, float]] = []
    if max_age_days > 0:
        cutoff = time.time() - max_age_days * _SECONDS_PER_DAY
        for entry in entries:
            path, size, mtime = entry
            if mtime < cutoff and _try_delete(path, expected_parent=directory, label=label):
                deleted_files += 1
                deleted_bytes += size
                aged_out += 1
            else:
                survivors.append(entry)
    else:
        survivors = list(entries)

    size_trimmed = 0
    remaining_bytes = sum(size for _, size, _ in survivors)
    if max_total_gb > 0:
        budget = int(max_total_gb * _BYTES_PER_GB)
        if remaining_bytes > budget:
            # Oldest-first, so the active (newest) files are the last candidates to be trimmed.
            for path, size, _mtime in sorted(survivors, key=lambda item: item[2]):
                if remaining_bytes <= budget:
                    break
                if _try_delete(path, expected_parent=directory, label=label):
                    deleted_files += 1
                    deleted_bytes += size
                    size_trimmed += 1
                    remaining_bytes -= size

    if deleted_files:
        logger.info(
            f"{label} removed {deleted_files} file(s), {deleted_bytes / _BYTES_PER_GB:.2f} GB, from "
            f"{directory} ({aged_out} aged out, {size_trimmed} over the size budget); "
            f"{remaining_bytes / _BYTES_PER_GB:.2f} GB remain.",
        )

    return PurgeResult(
        deleted_files=deleted_files,
        deleted_bytes=deleted_bytes,
        aged_out=aged_out,
        size_trimmed=size_trimmed,
        remaining_bytes=remaining_bytes,
    )
