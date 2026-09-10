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
from .job_lifecycle import JobLifecycleModel, JobRecord
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
        "see_also": finding.see_also.value if finding.see_also is not None else None,
        "reference_page": finding.reference_page,
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
        if finding.reference_page:
            blocks.append(f"    see: {finding.reference_page}")
    return "\n".join(blocks)


def _fmt_seconds(seconds: float | None) -> str:
    """Format a wait segment in seconds to one decimal, or a placeholder when it is unmeasurable."""
    return f"{seconds:.1f}" if seconds is not None else "-"


def _job_sort_key(job: JobRecord) -> datetime:
    """Order jobs by pop time, falling back to dispatch for a job whose pop is off the front of the log."""
    return job.popped_at or job.dispatched_at or datetime.min


def job_record_to_dict(job: JobRecord) -> dict[str, object]:
    """A JSON-serializable per-job lifecycle row."""
    return {
        "job_id": job.job_id,
        "model": job.model,
        "popped_at": job.popped_at.isoformat() if job.popped_at else None,
        "dispatched_at": job.dispatched_at.isoformat() if job.dispatched_at else None,
        "inference_finished_at": (job.inference_finished_at.isoformat() if job.inference_finished_at else None),
        "submitted_at": job.submitted_at.isoformat() if job.submitted_at else None,
        "process_id": job.process_id,
        "device_index": job.device_index,
        "batch": job.batch,
        "effective_megapixelsteps": job.effective_megapixelsteps,
        "pre_inference_seconds": job.pre_inference_seconds,
        "inference_seconds": job.inference_seconds,
        "post_inference_seconds": job.post_inference_seconds,
        "safety_seconds": job.safety_seconds,
        "faulted": job.faulted,
        "line_skipped": job.line_skipped,
    }


def job_lifecycle_to_dict(session: WorkerSession, model: JobLifecycleModel) -> dict[str, object]:
    """A JSON-serializable per-job table plus the session's wait percentiles and concurrency histogram."""
    profile = model.sampling_concurrency()
    census = model.status_census()
    return {
        "session": session_to_dict(session),
        "card_count": model.card_count,
        "intake_budget": model.intake_budget,
        "jobs": [job_record_to_dict(job) for job in sorted(model.jobs.values(), key=_job_sort_key)],
        "wait_segments": [
            {
                "name": segment.name,
                "count": segment.count,
                "median_seconds": segment.median,
                "p90_seconds": segment.p90,
            }
            for segment in model.wait_segments()
        ],
        "sampling_concurrency": {
            "observed_seconds": profile.observed_seconds,
            "mean_cards_busy": profile.mean_cards_busy,
            "seconds_by_card_count": {str(count): seconds for count, seconds in profile.seconds_by_card_count.items()},
        },
        "status_census": {
            "snapshots": census.snapshots,
            "median_pending": census.median_pending,
            "median_idle_lanes": census.median_idle_lanes,
            "median_seatable_pending": census.median_seatable_pending,
            "max_seatable_pending": census.max_seatable_pending,
            "head_seatable": census.head_states.seatable,
            "head_resident_but_blocked": census.head_states.resident_but_blocked,
            "head_cold": census.head_states.cold,
        },
        "line_skips": model.line_skip_census(),
        "model_movement": {
            "preloads": model.model_movement.preloads,
            "unloads": model.model_movement.unloads,
            "cleared_preloads": model.model_movement.cleared_preloads,
            "displaced_expiries": model.model_movement.displaced_expiries,
        },
        "lane_placement": {
            str(device_index): [
                {"process_id": placement.process_id, "role": str(placement.role)} for placement in lanes
            ]
            for device_index, lanes in model.card_occupancy().items()
        },
    }


_JOB_TABLE_HEADER = (
    f"{'job':<9}{'popped':<10}{'model':<28}{'proc':>5}{'card':>6}{'pop->disp':>11}{'generate':>10}{'->submit':>10}"
)


def render_job_lifecycle(session: WorkerSession, model: JobLifecycleModel, *, limit: int | None = None) -> str:
    """A per-job lifecycle table with the session's wait percentiles and sampling-concurrency histogram.

    The three wait columns are the whole point: a session whose time sits in ``pop->disp`` is a
    scheduling problem, in ``generate`` a GPU/config one, and in ``->submit`` a pipeline-balance one.
    """
    span, duration = _fmt_session_span(session)
    lines = [f"=== Session #{session.index}  {span}  ({duration})  {session.end_reason} ==="]
    if not model.jobs:
        return "\n".join([*lines, "  (no job lifecycle lines in this session)"])

    ordered = sorted(model.jobs.values(), key=_job_sort_key)
    shown = ordered if limit is None else ordered[:limit]
    lines.append("")
    lines.append(_JOB_TABLE_HEADER)
    for job in shown:
        model_name = (job.model or "?")[:27]
        card = str(job.device_index) if job.device_index is not None else "-"
        process = str(job.process_id) if job.process_id is not None else "-"
        lines.append(
            f"{job.job_id:<9}{_fmt_ts(job.popped_at):<10}{model_name:<28}{process:>5}{card:>6}"
            f"{_fmt_seconds(job.pre_inference_seconds):>11}"
            f"{_fmt_seconds(job.inference_seconds):>10}"
            f"{_fmt_seconds(job.post_inference_seconds):>10}",
        )
    if limit is not None and len(ordered) > limit:
        lines.append(f"... {len(ordered) - limit} more job(s) not shown (use --limit 0 for all)")

    lines.append("")
    lines.append(f"{len(ordered)} job(s); wait segments (seconds):")
    for segment in model.wait_segments():
        lines.append(
            f"  {segment.name:<15} n={segment.count:<5} median={_fmt_seconds(segment.median):>8} "
            f"p90={_fmt_seconds(segment.p90):>8}",
        )

    profile = model.sampling_concurrency()
    cards = model.card_count
    lines.append("")
    if cards is None:
        lines.append("Cards driven: unknown (no 'Driving N cards' line in this capture)")
    else:
        lines.append(f"Cards driven: {cards} | worker-wide intake budget: {model.intake_budget or '?'}")
    mean_busy = profile.mean_cards_busy
    mean_clause = f"{mean_busy:.2f}" if mean_busy is not None else "?"
    lines.append(f"Sampling concurrency (time-weighted mean cards busy: {mean_clause})")
    lines.append(f"  {profile.describe(top=(cards or 8) + 1)}")

    census = model.status_census()
    if census.snapshots:
        lines.append(
            f"Status prints: {census.snapshots} | median pending {census.median_pending:.0f} | "
            f"median idle lanes {census.median_idle_lanes:.0f} | "
            f"median seatable pending {census.median_seatable_pending:.0f}",
        )
    skips = model.line_skip_census()
    if skips:
        lines.append("Line skips: " + ", ".join(f"{reason} {count}" for reason, count in sorted(skips.items())))
    return "\n".join(lines)
