"""Segment an appended ``bridge.log`` into per-launch worker sessions with an end-reason verdict.

``bridge.log`` is appended across restarts, so one file holds many worker lifetimes back to back. Every
incident question ("what did *this* run do") starts by isolating the right lifetime. The worker's
process manager constructs exactly once per launch and logs a burst of ``__init__`` lines; that banner
is the session boundary, reusing the same contract :mod:`duty_log_report` already relies on.

Each session is then classified by *how it ended* (clean exit, gave up and aborted, operator shutdown,
or killed/crashed mid-run), which is the first thing you want to know and the signal the detectors key
the recovery story off of.

A ``horde-benchmark`` run logs the same orchestrator lines to ``bridge_harness.log``.
:func:`segment_bundle_sessions` segments that file on its own and lists its sessions after the worker's,
each tagged with the :class:`~horde_worker_regen.analysis.bundle.OrchestratorLogKind` it came from and
numbered within its own kind, so a worker session keeps the index it has without a harness log.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from datetime import datetime

from .bundle import LogBundle, OrchestratorLogKind
from .duty_log_report import (
    _EPOCH_BOUNDARY_FALLBACK_RE,
    _EPOCH_BOUNDARY_RE,
    _EPOCH_COLLAPSE_SECONDS,
    _IDENTITY_RE,
)
from .log_ingest import LogRecord
from .log_signatures import pattern_for

_VERSION_RE = re.compile(r"\(v(?P<version>[^)]+)\)")
_RECOVERIES_RE = re.compile(r"process_recoveries: (?P<count>\d+)")
_HORDELIB_IDENTITY_RE = pattern_for("hordelib_identity")

# The main process logs this exactly once, as its very first line, before any model loading or the
# process-manager init banner. It is the most accurate session boundary because it does not let a new
# process's slow startup (logger -> model reference -> __init__, tens of seconds) bleed onto the prior
# session. Subprocesses log a *different* "Logger finished setting up for process:" line, so this stays
# main-process-only. The __init__ banner remains the fallback for partial captures that begin mid-run.
_MAIN_STARTUP_RE = re.compile(r"Setting up logger for main process")

# End-of-life markers, most-specific first; the first match wins when classifying a session.
_ABANDON_SHIP_RE = re.compile(r"cannot restore a working process pool|abandoning ship")
_ABORT_FILE_RE = pattern_for("abort_sentinel_found")
_SUPERVISOR_SHUTDOWN_RE = re.compile(r"Supervisor requested shutdown")
# The control loop's last word before an unexpected exception. The graceful shutdown that follows also logs
# the clean-exit marker, so this must outrank it: the run ended because the parent faulted, not by choice.
_CONTROL_LOOP_CRASH_RE = pattern_for("control_loop_crash")
# This is emitted only after ``start_working`` returns through session persistence. The process manager's
# earlier "Shutting down process manager" line proves only that child teardown began/completed; a gathered
# sibling task can still hold the worker process open, so treating that line as final produced false clean exits.
_CLEAN_EXIT_RE = re.compile(r"Worker has finished working")


class SessionEndReason(enum.StrEnum):
    """How a worker session terminated, as read from its final log lines."""

    CLEAN_EXIT = "clean_exit"
    """Normal shutdown drained and exited (no recovery abort involved)."""
    GAVE_UP_ABORTED = "gave_up_aborted"
    """Save-our-ship abandoned ship: pools were unrecoverable and the worker self-terminated."""
    ABORTED = "aborted"
    """An ``.abort`` sentinel forced an immediate stop (not via the give-up path)."""
    SUPERVISOR_SHUTDOWN = "supervisor_shutdown"
    """An operator/TUI shutdown (e.g. Ctrl-Q) stopped the worker."""
    CONTROL_LOOP_CRASH = "control_loop_crash"
    """The parent's control loop raised an unexpected exception and the worker shut itself down."""
    KILLED_OR_CRASHED = "killed_or_crashed"
    """No exit marker before the next session began: the run was killed or died without draining."""
    STILL_RUNNING = "still_running"
    """The most recent session has no exit marker: still running, or the log was captured mid-run."""


@dataclass
class WorkerSession:
    """One worker lifetime within an appended log: its records, identity, and how it ended."""

    index: int
    records: list[LogRecord] = field(default_factory=list)
    version: str | None = None
    hordelib: str | None = None
    """The run's hordelib version and source (``7.9.0 (editable hordelib gc3202bd0.modified)``), if logged."""
    dreamer_name: str | None = None
    num_models: int | None = None
    max_threads: int | None = None
    peak_process_recoveries: int = 0
    end_reason: SessionEndReason = SessionEndReason.STILL_RUNNING
    start_is_lower_bound: bool = False
    """Whether the launch boundary is missing from the capture, making ``start_ts`` an upper bound on it.

    The session then began at or before ``start_ts``, and ``duration_seconds`` is a lower bound on the
    real span. Reporting either as exact understates how long a run had been going, which is the figure a
    wedge-age or duty judgement rests on.
    """
    start_truncated: bool = False
    """Whether the missing launch boundary is attributable to a bundle truncation note in the capture."""
    log_kind: OrchestratorLogKind = OrchestratorLogKind.WORKER
    """Which orchestrator log the session was segmented from: a worker launch or a benchmark run."""

    @property
    def is_benchmark_run(self) -> bool:
        """Whether the session is a ``horde-benchmark`` run read from ``bridge_harness.log``."""
        return self.log_kind is OrchestratorLogKind.HARNESS

    @property
    def start_ts(self) -> datetime | None:
        """Timestamp of the first record with one, or None."""
        return next((record.timestamp for record in self.records if record.timestamp is not None), None)

    @property
    def end_ts(self) -> datetime | None:
        """Timestamp of the last record with one, or None."""
        return next((record.timestamp for record in reversed(self.records) if record.timestamp is not None), None)

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock span of the session, or None if it cannot be determined."""
        start, end = self.start_ts, self.end_ts
        if start is None or end is None:
            return None
        return (end - start).total_seconds()


def _is_main_startup(record: LogRecord) -> bool:
    """Whether a record is the main process's first-line logger-setup marker (the preferred boundary)."""
    return bool(_MAIN_STARTUP_RE.search(record.message))


def _is_init_banner(record: LogRecord) -> bool:
    """Whether a record is the once-per-launch process-manager init banner (the fallback boundary)."""
    return bool(_EPOCH_BOUNDARY_RE.search(record.location) or _EPOCH_BOUNDARY_FALLBACK_RE.search(record.full_text))


def segment_bundle_sessions(bundle: LogBundle, *, active_only: bool = False) -> list[WorkerSession]:
    """Segment every orchestrator log in ``bundle`` into sessions: the worker's, then the benchmark runs.

    ``bridge.log`` and ``bridge_harness.log`` are segmented separately, so a benchmark run that overlaps a
    worker launch in time stays its own session instead of absorbing or splitting the worker's. Each kind
    is numbered from 0 on its own, so a worker session's index is the same with or without a harness log
    and a session is addressed by its ``log_kind`` and ``index`` together.

    Args:
        bundle: The grouped log files to read.
        active_only: Read only the live logs and skip their rotations.

    Returns:
        The worker sessions in log order, followed by the benchmark runs in log order.
    """
    worker_records = bundle.active_orchestrator_records() if active_only else bundle.orchestrator_records()
    harness_records = bundle.active_harness_records() if active_only else bundle.harness_records()
    worker_sessions = segment_sessions(worker_records)
    harness_sessions = segment_sessions(harness_records, log_kind=OrchestratorLogKind.HARNESS)
    return worker_sessions + harness_sessions


def segment_sessions(
    records: list[LogRecord],
    *,
    log_kind: OrchestratorLogKind = OrchestratorLogKind.WORKER,
) -> list[WorkerSession]:
    """Split one orchestrator log's records into sessions, one per launch.

    Prefers the main process's first-line logger-setup marker so a new launch's slow startup does not
    bleed onto the prior session; falls back to the process-manager init banner for captures that have
    no logger-setup line (older logs, a bundle that begins mid-run, or ``bridge_harness.log``, which
    does not carry the main-process marker). A 30s burst-collapse keeps the several init lines of one
    launch from each opening a session.

    A session opened without a boundary record (the capture starts mid-run, e.g. a size-trimmed log)
    carries ``start_is_lower_bound``, so its start and duration are reported as bounds rather than as
    facts the capture cannot support. Every session is tagged with ``log_kind``; records from two
    different orchestrator logs must not be passed together (see :func:`segment_bundle_sessions`).
    """
    use_startup_boundary = any(_is_main_startup(record) for record in records)
    is_boundary = _is_main_startup if use_startup_boundary else _is_init_banner

    sessions: list[WorkerSession] = []
    current: WorkerSession | None = None
    last_boundary_ts: datetime | None = None

    for record in records:
        if is_boundary(record):
            ts = record.timestamp
            within_burst = (
                current is not None
                and last_boundary_ts is not None
                and ts is not None
                and (ts - last_boundary_ts).total_seconds() <= _EPOCH_COLLAPSE_SECONDS
            )
            if current is not None and last_boundary_ts is None and _is_harness_preamble(current, ts):
                current.start_is_lower_bound = False
            elif not within_burst:
                current = WorkerSession(index=len(sessions), log_kind=log_kind)
                sessions.append(current)
            if ts is not None:
                last_boundary_ts = ts
        if current is None:
            # Records before the first banner (the file began mid-session): open an implicit session 0.
            # Its launch line is not in the capture, so its start is only an upper bound on the real one.
            current = WorkerSession(
                index=0,
                start_is_lower_bound=True,
                start_truncated=record.follows_truncation,
                log_kind=log_kind,
            )
            sessions.append(current)
        current.records.append(record)

    for index, session in enumerate(sessions):
        _populate_session(session, is_last=index == len(sessions) - 1)
    return sessions


def _is_harness_preamble(implicit_session: WorkerSession, boundary_ts: datetime | None) -> bool:
    """Whether the records ahead of a harness log's first init banner are that run's own startup lines.

    The harness arms its log sink at the start of each run, before it builds the process manager, so a
    few lines (model reference and provider setup) always precede the banner. Opening them as a separate
    mid-run session would report a benchmark run that never happened. A capture cut by truncation, or a
    preamble longer than the burst window, is still treated as the tail of an earlier run.
    """
    if implicit_session.log_kind is not OrchestratorLogKind.HARNESS or implicit_session.start_truncated:
        return False
    start = implicit_session.start_ts
    if start is None or boundary_ts is None:
        return False
    return (boundary_ts - start).total_seconds() <= _EPOCH_COLLAPSE_SECONDS


def _populate_session(session: WorkerSession, *, is_last: bool) -> None:
    """Fill in identity, peak recoveries, and the end-reason verdict from the session's records."""
    for record in session.records:
        identity = _IDENTITY_RE.search(record.message)
        if identity is not None:
            session.dreamer_name = identity.group("name").strip()
            session.num_models = int(identity.group("num_models"))
            session.max_threads = int(identity.group("max_threads"))
            version = _VERSION_RE.search(record.message)
            if version is not None:
                session.version = version.group("version")
        hordelib = _HORDELIB_IDENTITY_RE.search(record.message)
        if hordelib is not None:
            session.hordelib = f"{hordelib.group('version')} ({hordelib.group('source')})"
        recoveries = _RECOVERIES_RE.search(record.message)
        if recoveries is not None:
            session.peak_process_recoveries = max(session.peak_process_recoveries, int(recoveries.group("count")))

    session.end_reason = _classify_end_reason(session, is_last=is_last)


def _classify_end_reason(session: WorkerSession, *, is_last: bool) -> SessionEndReason:
    """Decide how the session ended from its terminal markers (and whether it is the last session)."""
    saw_clean_exit = False
    saw_abandon = False
    saw_abort = False
    saw_supervisor = False
    saw_control_loop_crash = False
    for record in session.records:
        text = record.full_text
        if _ABANDON_SHIP_RE.search(text):
            saw_abandon = True
        if _ABORT_FILE_RE.search(text):
            saw_abort = True
        if _SUPERVISOR_SHUTDOWN_RE.search(text):
            saw_supervisor = True
        if _CONTROL_LOOP_CRASH_RE.search(text):
            saw_control_loop_crash = True
        if _CLEAN_EXIT_RE.search(text):
            saw_clean_exit = True

    if saw_abandon:
        return SessionEndReason.GAVE_UP_ABORTED
    if saw_control_loop_crash:
        return SessionEndReason.CONTROL_LOOP_CRASH
    if saw_supervisor and saw_clean_exit:
        return SessionEndReason.SUPERVISOR_SHUTDOWN
    if saw_abort:
        return SessionEndReason.ABORTED
    if saw_clean_exit:
        return SessionEndReason.CLEAN_EXIT
    # No terminal marker: the last session is presumably still live; an earlier one was cut short.
    return SessionEndReason.STILL_RUNNING if is_last else SessionEndReason.KILLED_OR_CRASHED
