"""File-descriptor limits and headroom: raise the soft ceiling, and measure per-process usage.

A process that leaks file descriptors climbs toward its per-process ``RLIMIT_NOFILE`` ceiling; once
there, every ``open()`` is refused with ``EMFILE`` (errno 24) and the process faults every job while
still heart-beating, so the silence watchdog cannot see it. The best-documented source in this stack is
PyTorch's default ``file_descriptor`` tensor-sharing strategy, which caches one descriptor per shared
tensor and so accumulates descriptors when many tensors cross a process boundary (see the PyTorch
multiprocessing notes); a raised ceiling is the documented counterpart to switching that strategy to
``file_system``.

Two defences live here, both POSIX-shaped. Windows has no ``RLIMIT_NOFILE`` and an effectively unbounded
handle count, so the raise is a no-op and the headroom query reports a count without a ceiling:

* :func:`raise_open_file_soft_limit` lifts the soft ceiling to the hard ceiling, so a slow leak takes far
  longer (in practice longer than a session) to exhaust. Child processes inherit the raised limit, so
  calling it once in the parent hardens the whole worker.
* :func:`open_descriptor_count` and :func:`descriptor_soft_limit` expose per-process headroom so a leak is
  visible as it grows rather than only after it has poisoned a slot.

Stdlib plus psutil only (no torch), so the torch-free orchestrator can import it.
"""

from __future__ import annotations

import sys

import psutil
from loguru import logger

# ``resource`` is POSIX-only. Guarding the import on ``sys.platform`` (rather than a try/except that
# rebinds it to None) lets the type checker treat it as the real module in the non-Windows branches, where
# every function that touches it has already returned on Windows.
if sys.platform != "win32":
    import resource

FD_HEADROOM_WARN_FRACTION = 0.8
"""Warn when a process's descriptor usage crosses this fraction of its soft ceiling.

Set below 1.0 so the warning lands while the slot is still serving, giving a real-time signal of a
descriptor climb before it reaches ``EMFILE`` and starts faulting every job."""


def raise_open_file_soft_limit() -> tuple[int, int] | None:
    """Raise this process's soft ``RLIMIT_NOFILE`` to its hard ceiling.

    Returns the ``(old_soft, new_soft)`` pair when the limit was raised, or None when there was nothing to
    do (already at the ceiling) or the platform has no such limit (Windows). Child processes spawned after
    this call inherit the raised soft limit, so one call in the parent covers the worker's whole process
    tree. Never raises: a platform that refuses the change is logged and left as-is.
    """
    if sys.platform == "win32":
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)  # pyrefly: ignore - POSIX-only, guarded above
    except (ValueError, OSError):
        return None
    if soft == hard:
        return None
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))  # pyrefly: ignore - POSIX-only, guarded above
    except (ValueError, OSError) as exc:
        logger.warning(f"Could not raise the open-file soft limit from {soft}: {exc}")
        return None
    logger.info(
        f"Raised open-file soft limit {soft} -> {hard} to harden inference processes against descriptor leaks",
    )
    return (soft, hard)


def open_descriptor_count(process: psutil.Process | None = None) -> int | None:
    """The number of open descriptors/handles held by ``process`` (defaulting to the caller), or None.

    Uses ``num_fds`` on POSIX and ``num_handles`` on Windows so the count is meaningful on both; returns
    None if the platform metric is unavailable or the process has gone (never raises).
    """
    proc = process if process is not None else psutil.Process()
    try:
        if sys.platform == "win32":
            return proc.num_handles()  # pyrefly: ignore - Windows-only psutil metric, guarded by sys.platform
        return proc.num_fds()  # pyrefly: ignore - POSIX-only psutil metric, guarded by sys.platform
    except (psutil.Error, OSError):
        return None


def descriptor_soft_limit() -> int | None:
    """This process's soft ``RLIMIT_NOFILE``, or None when there is no finite POSIX limit (e.g. Windows)."""
    if sys.platform == "win32":
        return None
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)  # pyrefly: ignore - POSIX-only, guarded above
    except (ValueError, OSError):
        return None
    if soft == resource.RLIM_INFINITY:  # pyrefly: ignore - POSIX-only, guarded above
        return None
    return soft


def descriptor_headroom_fraction(open_fds: int | None, soft_limit: int | None) -> float | None:
    """The fraction of the descriptor ceiling in use (``open_fds / soft_limit``), or None if unknowable.

    A value approaching 1.0 means the process is near ``EMFILE``. None when either input is missing or the
    limit is non-positive, so callers treat "cannot tell" distinctly from "plenty of headroom".
    """
    if open_fds is None or soft_limit is None or soft_limit <= 0:
        return None
    return open_fds / soft_limit
