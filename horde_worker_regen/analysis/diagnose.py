"""The single first-class entry point for log triage: logs in, ranked findings out.

The ``horde-log diagnose`` CLI and the TUI Diagnostics tab both want the same thing -- segment a log
path into sessions and run every detector over them -- and both want the structured
:class:`~horde_worker_regen.analysis.detectors.Finding` objects, not printed text. Keeping that
orchestration here (rather than inside the argparse layer) lets either caller import it directly
without shelling out, and guarantees they cannot drift apart: the CLI renders what this returns, the
TUI renders the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bundle import LogBundle
from .correlate import build_session_context
from .detectors import Finding, run_detectors
from .sessions import WorkerSession, segment_sessions

DEFAULT_LOG_PATH = Path("logs")


@dataclass
class SessionDiagnosis:
    """One session paired with the findings the detectors produced for it (most-severe first)."""

    session: WorkerSession
    findings: list[Finding]


@dataclass
class SessionSummary:
    """The display-relevant fields of a session, without its (large) parsed record list.

    A :class:`WorkerSession` carries every parsed :class:`~horde_worker_regen.analysis.log_ingest.LogRecord`
    of the session, which is megabytes the UI never shows. This summary is what crosses a process
    boundary instead, so an off-process diagnosis returns kilobytes, not the whole parsed log.
    """

    index: int
    version: str | None
    end_reason: str
    num_models: int | None
    max_threads: int | None
    peak_process_recoveries: int


@dataclass
class SessionDiagnosisView:
    """A session summary plus its findings: the lightweight, picklable result the TUI consumes."""

    session: SessionSummary
    findings: list[Finding]


def diagnose(
    path: Path = DEFAULT_LOG_PATH,
    *,
    last: bool = False,
    session_index: int | None = None,
    recent: int | None = None,
    active_only: bool = False,
) -> list[SessionDiagnosis]:
    """Load the logs at ``path``, segment them into sessions, and run all detectors over each.

    Args:
        path: A logs directory, a single log file, or a ``.zip`` of logs.
        last: Restrict the result to only the most recent session.
        session_index: Restrict the result to only the session with this index. Takes precedence
            over ``last`` when both are given.
        recent: Restrict the result to the most recent ``recent`` sessions (lowest precedence; ignored
            when ``last`` or ``session_index`` is given). Useful for a quick, bounded pass that does
            not run detectors over a long restart history.
        active_only: Read only the live ``bridge.log`` and skip zipped rotations. Pairs with ``recent``
            for a fast pass (the recent sessions live in the active file); a full history needs the
            rotations, so leave this off for that.

    Returns:
        One :class:`SessionDiagnosis` per selected session, in session order. Empty if the path holds
        no recognizable sessions. Detectors never raise (see
        :func:`~horde_worker_regen.analysis.detectors.run_detectors`), so this is safe to call on
        partial or torn logs.
    """
    bundle = LogBundle.from_path(path)
    records = bundle.active_orchestrator_records() if active_only else bundle.orchestrator_records()
    sessions = segment_sessions(records)
    selected = select_sessions(sessions, last=last, session_index=session_index, recent=recent)
    return [
        SessionDiagnosis(session=session, findings=run_detectors(build_session_context(session, bundle)))
        for session in selected
    ]


def diagnose_views(
    path: Path = DEFAULT_LOG_PATH,
    *,
    last: bool = False,
    session_index: int | None = None,
    recent: int | None = None,
    active_only: bool = False,
) -> list[SessionDiagnosisView]:
    """:func:`diagnose`, but returning lightweight, picklable views (no parsed records).

    This is what a caller runs across a process boundary: the heavy parsed sessions stay (and are
    freed) in the worker process, and only the per-session summaries plus findings are returned.
    """
    return [
        SessionDiagnosisView(session=_summarize(result.session), findings=result.findings)
        for result in diagnose(path, last=last, session_index=session_index, recent=recent, active_only=active_only)
    ]


def diagnose_views_for(path: Path, recent: int | None, active_only: bool) -> list[SessionDiagnosisView]:
    """Positional, picklable entry point for running a view-diagnosis in a worker process."""
    return diagnose_views(path, recent=recent, active_only=active_only)


def _summarize(session: WorkerSession) -> SessionSummary:
    """Reduce a parsed session to its display fields (dropping its record list)."""
    return SessionSummary(
        index=session.index,
        version=session.version,
        end_reason=str(session.end_reason),
        num_models=session.num_models,
        max_threads=session.max_threads,
        peak_process_recoveries=session.peak_process_recoveries,
    )


def select_sessions(
    sessions: list[WorkerSession],
    *,
    last: bool,
    session_index: int | None,
    recent: int | None = None,
) -> list[WorkerSession]:
    """Apply ``--last`` / ``--session N`` / recent-N selection to a session list (shared CLI/TUI rule)."""
    if session_index is not None:
        return [session for session in sessions if session.index == session_index]
    if last and sessions:
        return sessions[-1:]
    if recent is not None and recent > 0:
        return sessions[-recent:]
    return sessions
