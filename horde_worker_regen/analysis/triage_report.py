"""Render sessions, timelines, and findings as human-readable text or machine-readable JSON.

Kept separate from the analysis so the same parsed data can be printed for an operator at the console
or emitted as JSON for CI / further tooling, without the producers knowing which.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .bundle import RotationStitch
from .correlate import TimelineEntry
from .detectors import Finding, Severity
from .sessions import WorkerSession


def _fmt_ts_ms(ts: datetime | None) -> str:
    """Format a timestamp as HH:MM:SS.mmm for the fine-grained timeline view."""
    return ts.strftime("%H:%M:%S.%f")[:-3] if ts is not None else "--:--:--.---"


def _fmt_ts(ts: datetime | None) -> str:
    """Format a timestamp as HH:MM:SS, or a placeholder when absent."""
    return ts.strftime("%H:%M:%S") if ts is not None else "--:--:--"


def _fmt_duration(seconds: float | None) -> str:
    """Format a duration compactly (e.g. ``16m1s``), or ``?`` when unknown."""
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes}m"


def _fmt_session_span(session: WorkerSession) -> tuple[str, str]:
    """Format a session's span and duration, marked as bounds when its launch line is not in the capture."""
    start = _fmt_ts(session.start_ts)
    duration = _fmt_duration(session.duration_seconds)
    if session.start_is_lower_bound:
        return f"<={start} -> {_fmt_ts(session.end_ts)}", f">={duration}"
    return f"{start} -> {_fmt_ts(session.end_ts)}", duration


def _start_bound_note(session: WorkerSession) -> str:
    """The line explaining why a session's start is a bound, or an empty string when it is exact."""
    if not session.start_is_lower_bound:
        return ""
    cause = "log truncated" if session.start_truncated else "capture begins mid-run"
    return (
        f"    started at or before {_fmt_ts(session.start_ts)} ({cause}): the span and duration above are lower bounds"
    )


def session_to_dict(session: WorkerSession) -> dict[str, object]:
    """A JSON-serializable summary of one session."""
    return {
        "index": session.index,
        "start": session.start_ts.isoformat() if session.start_ts else None,
        "start_is_lower_bound": session.start_is_lower_bound,
        "start_truncated": session.start_truncated,
        "end": session.end_ts.isoformat() if session.end_ts else None,
        "duration_seconds": session.duration_seconds,
        "version": session.version,
        "dreamer_name": session.dreamer_name,
        "num_models": session.num_models,
        "max_threads": session.max_threads,
        "peak_process_recoveries": session.peak_process_recoveries,
        "end_reason": str(session.end_reason),
        "num_records": len(session.records),
    }


def render_sessions(sessions: list[WorkerSession], *, root: Path, stitch: RotationStitch | None = None) -> str:
    """A compact per-session listing: span, version, end-reason, and peak recoveries.

    ``stitch`` names the rotated predecessors folded into the parse, so a span that covers more than the
    targeted file says which archives it came from.
    """
    if not sessions:
        return f"No worker sessions found in {root}."

    lines = [f"{len(sessions)} worker session(s) in {root}"]
    if stitch is not None:
        lines.append(f"Rotation: {stitch.describe()}")
    lines.append("")
    for session in sessions:
        span, duration = _fmt_session_span(session)
        version = session.version or "?"
        recoveries = session.peak_process_recoveries
        flag = "  <-- recovery storm" if recoveries >= 5 else ""
        lines.append(
            f"#{session.index}  {span}  ({duration})  v{version}  "
            f"{session.end_reason}  recoveries: {recoveries}{flag}",
        )
        models = session.num_models if session.num_models is not None else "?"
        threads = session.max_threads if session.max_threads is not None else "?"
        lines.append(f"    dreamer: {session.dreamer_name or '?'} | models: {models} | threads: {threads}")
        note = _start_bound_note(session)
        if note:
            lines.append(note)
    return "\n".join(lines)


_SOURCE_TAG = {
    "orchestrator": "orch ",
    "ledger": "ldgr ",
    "child_startup": "CRASH",
    "child": "chld ",
}


def timeline_entry_to_dict(entry: TimelineEntry) -> dict[str, object]:
    """A JSON-serializable timeline entry."""
    return {
        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
        "source": entry.source,
        "process_id": entry.process_id,
        "level": entry.level,
        "location": entry.location,
        "text": entry.text,
        "job_id": entry.job_id,
        "evidence": [str(entry.evidence[0]), entry.evidence[1]] if entry.evidence else None,
    }


def render_timeline(entries: list[TimelineEntry]) -> str:
    """A merged, time-ordered parent/child/ledger event stream, one entry per line."""
    if not entries:
        return "No timeline entries."
    lines = []
    for entry in entries:
        tag = _SOURCE_TAG.get(entry.source, entry.source[:5].ljust(5))
        slot = f"p{entry.process_id}" if entry.process_id is not None else "  "
        lines.append(f"{_fmt_ts_ms(entry.timestamp)}  {tag}  {slot:>3}  {entry.level:<8}  {entry.text}")
    return "\n".join(lines)


_SEVERITY_MARK = {Severity.CRITICAL: "[!!]", Severity.WARNING: "[! ]", Severity.INFO: "[i ]"}


def finding_to_dict(finding: Finding) -> dict[str, object]:
    """A JSON-serializable finding."""
    return {
        "id": finding.id,
        "severity": str(finding.severity),
        "title": finding.title,
        "verdict": finding.verdict,
        "remediation": finding.remediation,
        "evidence": finding.evidence,
        "see_also": finding.see_also,
    }


def render_findings(session: WorkerSession, findings: list[Finding]) -> str:
    """A per-session diagnosis block: each finding's verdict, evidence, and remediation."""
    span, duration = _fmt_session_span(session)
    header = f"=== Session #{session.index}  {span}  ({duration})  {session.end_reason} ==="
    note = _start_bound_note(session)
    if note:
        header = f"{header}\n{note.strip()}"
    if not findings:
        return header + "\n  (no findings)"
    blocks = [header]
    for finding in findings:
        mark = _SEVERITY_MARK.get(finding.severity, "[? ]")
        blocks.append(f"\n{mark} {finding.title}  ({finding.id})")
        blocks.append(f"    {finding.verdict}")
        for line in finding.evidence:
            blocks.append(f"      - {line}")
        if finding.remediation:
            blocks.append(f"    -> {finding.remediation}")
        if finding.see_also:
            blocks.append(f"    see also: {finding.see_also}")
    return "\n".join(blocks)
