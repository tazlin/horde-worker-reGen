"""Segment an appended ``bridge.log`` into per-launch worker sessions with an end-reason verdict.

``bridge.log`` is appended across restarts, so one file holds many worker lifetimes back to back. Every
incident question ("what did *this* run do") starts by isolating the right lifetime. The worker's
process manager constructs exactly once per launch and logs a burst of ``__init__`` lines; that banner
is the session boundary, reusing the same contract :mod:`duty_log_report` already relies on.

Each session is then classified by *how it ended* (clean exit, gave up and aborted, operator shutdown,
or killed/crashed mid-run), which is the first thing you want to know and the signal the detectors key
the recovery story off of.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from datetime import datetime

from .duty_log_report import (
    _EPOCH_BOUNDARY_FALLBACK_RE,
    _EPOCH_BOUNDARY_RE,
    _EPOCH_COLLAPSE_SECONDS,
    _IDENTITY_RE,
)
from .log_ingest import LogRecord

_VERSION_RE = re.compile(r"\(v(?P<version>[^)]+)\)")
_RECOVERIES_RE = re.compile(r"process_recoveries: (?P<count>\d+)")

# The main process logs this exactly once, as its very first line, before any model loading or the
# process-manager init banner. It is the most accurate session boundary because it does not let a new
# process's slow startup (logger -> model reference -> __init__, tens of seconds) bleed onto the prior
# session. Subprocesses log a *different* "Logger finished setting up for process:" line, so this stays
# main-process-only. The __init__ banner remains the fallback for partial captures that begin mid-run.
_MAIN_STARTUP_RE = re.compile(r"Setting up logger for main process")

# End-of-life markers, most-specific first; the first match wins when classifying a session.
_ABANDON_SHIP_RE = re.compile(r"cannot restore a working process pool|abandoning ship")
_ABORT_FILE_RE = re.compile(r"Found \.abort file")
_SUPERVISOR_SHUTDOWN_RE = re.compile(r"Supervisor requested shutdown")
_CLEAN_EXIT_RE = re.compile(r"Worker has finished working|Shutting down process manager")


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
    dreamer_name: str | None = None
    num_models: int | None = None
    max_threads: int | None = None
    peak_process_recoveries: int = 0
    end_reason: SessionEndReason = SessionEndReason.STILL_RUNNING

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


def segment_sessions(records: list[LogRecord]) -> list[WorkerSession]:
    """Split orchestrator records into sessions, one per worker launch.

    Prefers the main process's first-line logger-setup marker so a new launch's slow startup does not
    bleed onto the prior session; falls back to the process-manager init banner for captures that have
    no logger-setup line (older logs, or a bundle that begins mid-run). A 30s burst-collapse keeps the
    several init lines of one launch from each opening a session.
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
            if not within_burst:
                current = WorkerSession(index=len(sessions))
                sessions.append(current)
            if ts is not None:
                last_boundary_ts = ts
        if current is None:
            # Records before the first banner (the file began mid-session): open an implicit session 0.
            current = WorkerSession(index=0)
            sessions.append(current)
        current.records.append(record)

    for index, session in enumerate(sessions):
        _populate_session(session, is_last=index == len(sessions) - 1)
    return sessions


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
    for record in session.records:
        text = record.full_text
        if _ABANDON_SHIP_RE.search(text):
            saw_abandon = True
        if _ABORT_FILE_RE.search(text):
            saw_abort = True
        if _SUPERVISOR_SHUTDOWN_RE.search(text):
            saw_supervisor = True
        if _CLEAN_EXIT_RE.search(text):
            saw_clean_exit = True

    if saw_abandon:
        return SessionEndReason.GAVE_UP_ABORTED
    if saw_supervisor and saw_clean_exit:
        return SessionEndReason.SUPERVISOR_SHUTDOWN
    if saw_abort:
        return SessionEndReason.ABORTED
    if saw_clean_exit:
        return SessionEndReason.CLEAN_EXIT
    # No terminal marker: the last session is presumably still live; an earlier one was cut short.
    return SessionEndReason.STILL_RUNNING if is_last else SessionEndReason.KILLED_OR_CRASHED
