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

The delete-safety mechanics (non-recursive enumeration, symlink refusal, parent re-verification,
fail-closed recognition, single-file unlink only) live in :mod:`horde_worker_regen._guarded_purge` and
are shared with the stats-file purge. This module only supplies the log-file recognizer: deletion is
limited to files :mod:`horde_worker_regen.log_file_registry` positively recognizes as worker logs, so an
unfamiliar file under ``logs/`` (or anything under a nested folder there) is never touched.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from horde_worker_regen._guarded_purge import (
    _BYTES_PER_GB,
    _SECONDS_PER_DAY,
    PurgeResult,
    guarded_purge_directory,
)
from horde_worker_regen.log_file_registry import is_worker_log_file

__all__ = [
    "_BYTES_PER_GB",
    "_SECONDS_PER_DAY",
    "LogPurgeResult",
    "purge_log_directory",
    "purge_worker_logs_safely",
]

# The log purge is one application of the shared guarded sweep; its result type is that primitive's.
LogPurgeResult = PurgeResult


def purge_log_directory(
    log_dir: Path | str = Path("logs"),
    *,
    max_age_days: float,
    max_total_gb: float,
) -> LogPurgeResult:
    """Delete aged-out and over-budget log files from *log_dir*.

    Files last modified more than ``max_age_days`` ago are removed first; then, if the surviving files
    still total more than ``max_total_gb``, the oldest are removed until the directory fits. A limit of 0
    (or less) disables that stage. A missing directory is a no-op. Only recognized worker log files are
    ever eligible for deletion (see :func:`horde_worker_regen.log_file_registry.is_worker_log_file`).

    Args:
        log_dir: The worker's log directory (usually ``logs``).
        max_age_days: Delete files older than this many days. 0 disables the age-out.
        max_total_gb: Cap the directory at this many gigabytes. 0 disables the size cap.

    Returns:
        A :class:`LogPurgeResult` summarising what was removed.
    """
    return guarded_purge_directory(
        log_dir,
        recognizer=is_worker_log_file,
        max_age_days=max_age_days,
        max_total_gb=max_total_gb,
        label="Log purge",
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
