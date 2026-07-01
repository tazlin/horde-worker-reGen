"""Age-out and size-cap purge for the worker's on-disk log directory.

hordelib's loguru sinks rotate and zip the busy logs (``bridge*.log``, ``trace*.log``) and keep a
bounded *count* of rotations, but that per-sink count leaves two gaps this module closes:

* Several log families the worker writes have no rotation or retention at all and accumulate one file
  per run: the ``stdout``/``stderr`` redirections, the pre-sink startup-crash backstops
  (``bridge_*_startup.log``), the supervised-console redirect (``bridge_main_console.log``), and the
  ``*.faulthandler`` hard-fault dumps.
* A count-based retention never ages files out (a rarely-run worker keeps its old zips indefinitely)
  and does not bound the *total* size of the directory across every family at once.

The worker runs a single sweep of the top level of the ``logs/`` directory once at startup: delete
anything older than the age limit, then, if the directory still exceeds the size budget, delete
oldest-first until it fits. Both limits are generous by default and either can be disabled independently
(a limit of 0). The currently-active sinks are always the newest files, so the oldest-first size trim
reaches them last and the age-out never touches them; an actively-written file that cannot be unlinked
(a held handle on Windows) is skipped rather than allowed to fail the sweep.

Deletion is deliberately narrow, because a startup process silently removing files must be provably
incapable of removing the wrong ones. The sweep can only ever delete a file that satisfies *all* of:

* It is a **direct child** of the given log directory. Enumeration is non-recursive (``os.scandir``
  lists one level only) and every deletion re-checks the parent, so nested folders an operator keeps
  under ``logs/`` (e.g. a ``logs/remote_support/`` tree) are never inspected, descended, or removed.
* It is a **regular file**, evaluated with ``follow_symlinks=False``. Directories, symlinks, and
  special files are skipped. A symlink is judged as itself and never resolved, so the sweep can never
  reach through one to a target outside ``logs/``.
* It is a **recognized worker log file** per :mod:`horde_worker_regen.log_file_registry`. This is
  *fail-closed*: a file the registry does not positively describe is left untouched, not guessed at.
  The registry is the single declared source of truth for the worker's log-file families, and a CI
  check keeps it in step with the sinks the code actually opens.

There is no directory removal anywhere in this module: every deletion is a single-file ``unlink()``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from horde_worker_regen.log_file_registry import is_worker_log_file

_BYTES_PER_GB = 1024 * 1024 * 1024
_SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class LogPurgeResult:
    """What a single :func:`purge_log_directory` sweep removed, for logging and tests."""

    deleted_files: int = 0
    deleted_bytes: int = 0
    aged_out: int = 0
    size_trimmed: int = 0
    remaining_bytes: int = 0


def _collect_purgeable_files(directory: Path) -> list[tuple[Path, int, float]]:
    """Return ``(path, size_bytes, mtime)`` for each top-level regular log file in *directory*.

    Non-recursive and symlink-safe by construction: ``os.scandir`` yields only the direct children of
    *directory*, symlinks are skipped outright, and only regular files the log-file registry recognizes
    as worker logs are returned. A subdirectory (or anything under it) is never descended into.
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
            # Fail-closed: only files the registry positively recognizes as worker logs are eligible.
            if not is_worker_log_file(entry.name):
                continue
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            collected.append((Path(entry.path), stat.st_size, stat.st_mtime))
    return collected


def _try_delete(path: Path, *, expected_parent: Path) -> bool:
    """Delete *path*, returning whether it went. Re-verifies the file before unlinking.

    The candidate must still be a direct child of *expected_parent* and still a regular, non-symlink
    file at the moment of deletion; anything else is refused. A held handle (Windows) or a race is
    swallowed so a single undeletable file never fails the whole sweep.
    """
    if path.parent != expected_parent:
        logger.warning(f"Log purge refused {path}: not a direct child of {expected_parent}; skipping.")
        return False
    if path.is_symlink() or not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError as delete_error:
        logger.debug(f"Log purge could not delete {path}: {type(delete_error).__name__}: {delete_error}")
        return False


def purge_log_directory(
    log_dir: Path | str = Path("logs"),
    *,
    max_age_days: float,
    max_total_gb: float,
) -> LogPurgeResult:
    """Delete aged-out and over-budget log files from *log_dir*.

    Files last modified more than ``max_age_days`` ago are removed first; then, if the surviving files
    still total more than ``max_total_gb``, the oldest are removed until the directory fits. A limit of 0
    (or less) disables that stage. A missing directory is a no-op. Per-file delete errors are swallowed,
    so the sweep is best-effort and never raises for an individual file.

    Args:
        log_dir: The worker's log directory (usually ``logs``).
        max_age_days: Delete files older than this many days. 0 disables the age-out.
        max_total_gb: Cap the directory at this many gigabytes. 0 disables the size cap.

    Returns:
        A :class:`LogPurgeResult` summarising what was removed.
    """
    directory = Path(log_dir)
    if not directory.is_dir():
        return LogPurgeResult()

    # One consistent, non-recursive, symlink-safe snapshot that both stages reason over. Every path here
    # is a top-level regular log file of *directory* (see _collect_purgeable_files); nothing else is even
    # a candidate, and every deletion below re-verifies the parent against this same directory.
    entries = _collect_purgeable_files(directory)

    deleted_files = 0
    deleted_bytes = 0
    aged_out = 0

    survivors: list[tuple[Path, int, float]] = []
    if max_age_days > 0:
        cutoff = time.time() - max_age_days * _SECONDS_PER_DAY
        for entry in entries:
            path, size, mtime = entry
            if mtime < cutoff and _try_delete(path, expected_parent=directory):
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
            # Oldest-first, so the active (newest) sinks are the last candidates to be trimmed.
            for path, size, _mtime in sorted(survivors, key=lambda item: item[2]):
                if remaining_bytes <= budget:
                    break
                if _try_delete(path, expected_parent=directory):
                    deleted_files += 1
                    deleted_bytes += size
                    size_trimmed += 1
                    remaining_bytes -= size

    if deleted_files:
        logger.info(
            f"Log purge removed {deleted_files} file(s), {deleted_bytes / _BYTES_PER_GB:.2f} GB, from "
            f"{directory} ({aged_out} aged out, {size_trimmed} over the size budget); "
            f"{remaining_bytes / _BYTES_PER_GB:.2f} GB remain.",
        )

    return LogPurgeResult(
        deleted_files=deleted_files,
        deleted_bytes=deleted_bytes,
        aged_out=aged_out,
        size_trimmed=size_trimmed,
        remaining_bytes=remaining_bytes,
    )


def purge_worker_logs_safely(
    *,
    max_age_days: float,
    max_total_gb: float,
    log_dir: Path | str = Path("logs"),
) -> None:
    """Run :func:`purge_log_directory`, swallowing any error so log hygiene can never block startup."""
    try:
        purge_log_directory(log_dir, max_age_days=max_age_days, max_total_gb=max_total_gb)
    except Exception as purge_error:
        logger.warning(f"Log purge failed ({type(purge_error).__name__}: {purge_error}); continuing startup.")
