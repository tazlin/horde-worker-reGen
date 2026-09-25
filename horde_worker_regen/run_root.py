"""Where a worker run keeps the files it leaves on disk.

A run writes three things next to itself: the ``.abort`` sentinel (dropped by an operator or by the
worker's own abort path, polled by every control loop tick), the ``logs/`` directory, and the
``.horde_worker_regen/`` state directory. All three used to be relative paths resolved against whatever
the working directory happened to be, so two runs sharing a directory coupled through them: one run's
abort stopped the other, and both wrote the same log files. This module gives them one root.

``HORDE_WORKER_RUN_ROOT`` names the root. Unset, it is the working directory, which is what a production
launch has always used, so nothing changes for an operator who does not set it. Spawned children inherit
the variable, so a parent and its children agree without any extra plumbing. The test suite points it
at a temporary directory per test.

The resolver announces itself once per process per distinct value: at INFO when the variable is set (with
what it was set to and where that resolved), at DEBUG when the working directory is in use, and with a
WARNING when the named directory is missing or not writable. A developer chasing a file that landed in
the wrong place, or a CI run whose variable was set to the wrong path, finds the answer at the top of
the log rather than by reading this module.

A benchmark run (``horde-benchmark``, the harness) shares the checkout's run root with any live worker but
watches a sentinel of its own, :data:`BENCHMARK_ABORT_SENTINEL_NAME`. The harness claims it for the
process with :func:`claim_abort_sentinel`, so the worker it embeds reads, writes and clears that file
instead of ``.abort``: one run's abort never stops the other, and a benchmark's startup cleanup never
deletes a live worker's sentinel.
"""

from __future__ import annotations

import enum
import os
from pathlib import Path

from loguru import logger

RUN_ROOT_ENV_VAR = "HORDE_WORKER_RUN_ROOT"
ABORT_SENTINEL_NAME = ".abort"
BENCHMARK_ABORT_SENTINEL_NAME = ".abort_benchmark"
LOGS_DIR_NAME = "logs"


class AbortSentinelKind(enum.StrEnum):
    """Which kind of run an abort sentinel belongs to."""

    WORKER = enum.auto()
    BENCHMARK = enum.auto()


_ABORT_SENTINEL_NAMES: dict[AbortSentinelKind, str] = {
    AbortSentinelKind.WORKER: ABORT_SENTINEL_NAME,
    AbortSentinelKind.BENCHMARK: BENCHMARK_ABORT_SENTINEL_NAME,
}

_claimed_abort_sentinel_kind: AbortSentinelKind = AbortSentinelKind.WORKER
"""The sentinel this process watches and writes; a worker unless a benchmark run has claimed its own."""

_announced: str | None = None
"""The last ``source:path`` announced in this process, so a repeat resolution stays silent."""


def run_root() -> Path:
    """The directory this run's sentinel, logs and state live under, absolute."""
    raw = os.environ.get(RUN_ROOT_ENV_VAR)
    root = Path(raw).expanduser().resolve() if raw else Path.cwd()
    _announce(root, raw)
    return root


def _announce(root: Path, raw: str | None) -> None:
    global _announced
    key = f"{raw!r}:{root}"
    if key == _announced:
        return
    _announced = key
    if raw is None:
        logger.debug(f"Run root is the working directory: {root}")
        return
    logger.info(f"Run root from {RUN_ROOT_ENV_VAR}={raw!r}: {root}")
    if not root.is_dir():
        logger.warning(
            f"{RUN_ROOT_ENV_VAR} names a directory that does not exist: {root}. It is created on first write; "
            "check the value if that is not where this run's logs and sentinel should go."
        )
    elif not os.access(root, os.W_OK):
        logger.warning(f"{RUN_ROOT_ENV_VAR} names a directory this process cannot write to: {root}")


def describe_run_root() -> str:
    """One line for a startup banner or a support bundle: the root and where it came from."""
    raw = os.environ.get(RUN_ROOT_ENV_VAR)
    root = run_root()
    source = f"{RUN_ROOT_ENV_VAR}={raw!r}" if raw else "working directory"
    return f"run root: {root} ({source})"


def abort_sentinel_name(kind: AbortSentinelKind) -> str:
    """Return the sentinel file name a run of ``kind`` watches."""
    return _ABORT_SENTINEL_NAMES[kind]


def claimed_abort_sentinel_kind() -> AbortSentinelKind:
    """Return the kind of sentinel this process watches and writes."""
    return _claimed_abort_sentinel_kind


def claim_abort_sentinel(kind: AbortSentinelKind) -> AbortSentinelKind:
    """Make this process watch and write the sentinel of ``kind``, returning the kind it replaced.

    The claim is process-wide because the worker's control loop and abort path resolve the sentinel through
    :func:`abort_sentinel_path` with no run context of their own. A caller that embeds a benchmark run in a
    longer-lived process restores the returned kind when the run ends.
    """
    global _claimed_abort_sentinel_kind
    previous = _claimed_abort_sentinel_kind
    _claimed_abort_sentinel_kind = kind
    return previous


def abort_sentinel_path(kind: AbortSentinelKind | None = None) -> Path:
    """Return the sentinel file whose presence aborts a run of ``kind``, by default this process's claim."""
    resolved_kind = _claimed_abort_sentinel_kind if kind is None else kind
    return run_root() / abort_sentinel_name(resolved_kind)


def logs_dir(*, create: bool = False) -> Path:
    """This run's ``logs/`` directory, created when asked so a sink can open a file in it at once."""
    path = run_root() / LOGS_DIR_NAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
