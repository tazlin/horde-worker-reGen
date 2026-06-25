"""Stitch the orchestrator log to the subprocess that actually crashed, and to the ledger.

The single most time-consuming step of a worker post-mortem is crossing the process boundary: the
parent ``bridge.log`` says "process 1 exited unexpectedly (exitcode=1); recovering", and the *reason*
is a Python traceback in ``bridge_inference_1_startup.log`` written by a different process. The parent
knows the slot's ``os_pid`` and ``launch`` (in the recovery diagnostics line and the action ledger);
the child log is keyed only by slot id in its filename and is appended across every relaunch. So the
join is slot id plus a timestamp window: for a parent recovery at time T, the child's crash is the
startup-log record nearest T.

This module parses the recovery diagnostics, extracts the exception summary from a child traceback,
performs that join, and merges orchestrator + ledger + child-crash records into one time-ordered
timeline per session for the ``timeline`` / ``job`` views and the detectors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from horde_worker_regen.process_management.ipc.action_ledger import LedgerEvent

from .bundle import LogBundle
from .log_ingest import LogRecord
from .sessions import WorkerSession

# "Recovery diagnostics for process 1 (os_pid=29872, launch=16): reason='...'; last_state=...; exitcode=1; ..."
_RECOVERY_RE = re.compile(
    r"Recovery diagnostics for process (?P<pid>\d+) \(os_pid=(?P<os_pid>\d+), launch=(?P<launch>\d+)\): "
    r"reason='(?P<reason>[^']*)'; last_state=(?P<last_state>\w+); exitcode=(?P<exitcode>[^;]+);",
)
# The last "ExceptionClass: message" line of a Python traceback.
_EXCEPTION_RE = re.compile(r"^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt|Exit)): ?(?P<msg>.*)$")
# A generic fallback for assertion-style or bare exception last lines.
_EXCEPTION_FALLBACK_RE = re.compile(r"^(?P<exc>[A-Za-z_][\w.]*): (?P<msg>.+)$")
# The optional identity the child stamps into its startup-crash line (newer workers), for an exact join.
_CHILD_OS_PID_RE = re.compile(r"os_pid=(?P<os_pid>\d+)")

# How far apart a parent recovery and a child startup crash may be and still be considered the same event.
_CHILD_JOIN_WINDOW = timedelta(seconds=30)


@dataclass
class RecoveryDiagnostic:
    """One parsed parent-side recovery diagnostic, the join key into the child logs."""

    timestamp: datetime | None
    process_id: int
    os_pid: int | None
    launch: int | None
    reason: str
    last_state: str
    exitcode: str
    record: LogRecord


@dataclass
class ChildCrash:
    """A subprocess crash lifted from its startup-log traceback, joined to a parent recovery."""

    process_id: int
    timestamp: datetime | None
    exception: str
    """The one-line exception summary, e.g. ``AssertionError: Torch not compiled with CUDA enabled``."""
    record: LogRecord


@dataclass
class TimelineEntry:
    """One event in a merged, time-ordered session timeline."""

    timestamp: datetime | None
    source: str
    """``orchestrator``, ``ledger``, ``child``, or ``child_startup``."""
    process_id: int | None
    level: str
    text: str
    location: str
    evidence: tuple[Path, int] | None
    job_id: str | None = None


@dataclass
class SessionContext:
    """A session plus the cross-process correlation derived from its bundle."""

    session: WorkerSession
    bundle: LogBundle
    recoveries: list[RecoveryDiagnostic] = field(default_factory=list)
    ledger_events: list[LedgerEvent] = field(default_factory=list)


def extract_exception(text: str) -> str | None:
    """Return the one-line exception summary from a traceback block, or None if none is present.

    Scans bottom-up for the conventional ``ExceptionClass: message`` final line (e.g.
    ``AssertionError: Torch not compiled with CUDA enabled``), which is the actionable root cause.
    """
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        match = _EXCEPTION_RE.match(stripped) or _EXCEPTION_FALLBACK_RE.match(stripped)
        if match is not None:
            message = match.group("msg").strip()
            return f"{match.group('exc')}: {message}" if message else match.group("exc")
    return None


def parse_recoveries(records: list[LogRecord]) -> list[RecoveryDiagnostic]:
    """Extract recovery diagnostics from orchestrator records (the parent's account of each replacement)."""
    diagnostics: list[RecoveryDiagnostic] = []
    for record in records:
        match = _RECOVERY_RE.search(record.message)
        if match is None:
            continue
        diagnostics.append(
            RecoveryDiagnostic(
                timestamp=record.timestamp,
                process_id=int(match.group("pid")),
                os_pid=int(match.group("os_pid")),
                launch=int(match.group("launch")),
                reason=match.group("reason"),
                last_state=match.group("last_state"),
                exitcode=match.group("exitcode").strip(),
                record=record,
            ),
        )
    return diagnostics


def _within(ts: datetime | None, start: datetime | None, end: datetime | None) -> bool:
    """Whether ``ts`` falls within the (inclusive, open-ended if a bound is None) session window."""
    if ts is None:
        return False
    if start is not None and ts < start - timedelta(seconds=2):
        return False
    return not (end is not None and ts > end + timedelta(seconds=2))


def find_child_crash(
    bundle: LogBundle,
    process_id: int,
    around: datetime | None,
    *,
    os_pid: int | None = None,
) -> ChildCrash | None:
    """Find the subprocess crash for slot ``process_id`` that caused a recovery (its root cause).

    When the child stamped its ``os_pid`` into the startup-crash line (newer workers) and the parent's
    recovery ``os_pid`` is supplied, an exact identity match wins outright. Otherwise the join falls back
    to the record whose timestamp is closest to the parent recovery within :data:`_CHILD_JOIN_WINDOW`,
    which is all older logs allow. Returns the extracted exception summary either way.
    """
    best: ChildCrash | None = None
    best_delta: timedelta | None = None
    for record in bundle.startup_records(process_id):
        exception = extract_exception(record.full_text)
        if exception is None:
            continue
        if os_pid is not None:
            stamped = _CHILD_OS_PID_RE.search(record.message)
            if stamped is not None and int(stamped.group("os_pid")) == os_pid:
                return ChildCrash(process_id, record.timestamp, exception, record)
        if around is not None and record.timestamp is not None:
            delta = abs(record.timestamp - around)
            if delta > _CHILD_JOIN_WINDOW:
                continue
            if best_delta is None or delta < best_delta:
                best, best_delta = ChildCrash(process_id, record.timestamp, exception, record), delta
        elif best is None:
            best = ChildCrash(process_id, record.timestamp, exception, record)
    return best


def build_session_context(session: WorkerSession, bundle: LogBundle) -> SessionContext:
    """Parse the recoveries and ledger window for one session."""
    start, end = session.start_ts, session.end_ts
    ledger_events = [
        event for event in bundle.ledger_events() if _within(datetime.fromtimestamp(event.timestamp), start, end)
    ]
    return SessionContext(
        session=session,
        bundle=bundle,
        recoveries=parse_recoveries(session.records),
        ledger_events=ledger_events,
    )


def build_timeline(context: SessionContext, *, include_child_loop: bool = False) -> list[TimelineEntry]:
    """Merge orchestrator + ledger + child-crash (and optionally child-loop) records in time order."""
    session, bundle = context.session, context.bundle
    start, end = session.start_ts, session.end_ts
    # Child crashes can land just outside the parent's record window (a slot dies seconds before the
    # parent reaps it, and a session can be only a few parent lines wide), so widen the inclusion window
    # for child records by the join window. This keeps a crash that *caused* a recovery in this session.
    child_start = start - _CHILD_JOIN_WINDOW if start is not None else None
    child_end = end + _CHILD_JOIN_WINDOW if end is not None else None
    entries: list[TimelineEntry] = []

    for record in session.records:
        entries.append(
            TimelineEntry(
                timestamp=record.timestamp,
                source="orchestrator",
                process_id=None,
                level=record.level,
                text=record.message,
                location=record.location,
                evidence=(record.source_path, record.raw_lineno),
            ),
        )

    for event in context.ledger_events:
        detail = f" {event.reason}" if event.reason else ""
        entries.append(
            TimelineEntry(
                timestamp=datetime.fromtimestamp(event.timestamp),
                source="ledger",
                process_id=event.process_id,
                level="LEDGER",
                text=f"{event.event_type}{detail}",
                location=str(event.event_type),
                evidence=None,
                job_id=event.job_id,
            ),
        )

    for process_id in sorted(bundle.process_ids()):
        for record in bundle.startup_records(process_id):
            if not _within(record.timestamp, child_start, child_end):
                continue
            exception = extract_exception(record.full_text)
            entries.append(
                TimelineEntry(
                    timestamp=record.timestamp,
                    source="child_startup",
                    process_id=process_id,
                    level=record.level,
                    text=exception or record.message,
                    location=record.location,
                    evidence=(record.source_path, record.raw_lineno),
                ),
            )
        if include_child_loop:
            for record in bundle.child_records(process_id):
                if not _within(record.timestamp, child_start, child_end):
                    continue
                entries.append(
                    TimelineEntry(
                        timestamp=record.timestamp,
                        source="child",
                        process_id=process_id,
                        level=record.level,
                        text=record.message,
                        location=record.location,
                        evidence=(record.source_path, record.raw_lineno),
                    ),
                )

    entries.sort(key=lambda entry: entry.timestamp or datetime.min)
    return entries
