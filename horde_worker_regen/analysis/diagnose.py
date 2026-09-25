"""The single first-class entry point for log triage: logs in, ranked findings out.

The ``horde-log diagnose`` CLI and the TUI Diagnostics tab both want the same thing: segment a log
path into sessions and run every detector over them, returning the structured
:class:`~horde_worker_regen.analysis.detectors.Finding` objects, not printed text. Keeping that
orchestration here (rather than inside the argparse layer) lets either caller import it directly
without shelling out, and guarantees they cannot drift apart: the CLI renders what this returns, the
TUI renders the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bundle import LogBundle, OrchestratorLogKind
from .correlate import build_session_context
from .detectors import Finding, run_detectors
from .sessions import WorkerSession, segment_bundle_sessions

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
    """The session's number within its ``log_kind``: worker sessions and benchmark runs each count from 0."""
    version: str | None
    end_reason: str
    num_models: int | None
    max_threads: int | None
    peak_process_recoveries: int
    log_kind: OrchestratorLogKind = OrchestratorLogKind.WORKER
    """``worker`` for a ``bridge.log`` session, ``harness`` for a benchmark run from ``bridge_harness.log``."""


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
    last_benchmark: bool = False,
    benchmark_index: int | None = None,
    recent: int | None = None,
    active_only: bool = False,
) -> list[SessionDiagnosis]:
    """Load the logs at ``path``, segment them into sessions, and run all detectors over each.

    Worker sessions and benchmark runs are numbered separately (see
    :func:`~horde_worker_regen.analysis.sessions.segment_bundle_sessions`), and each kind has its own
    selectors; :func:`select_sessions` says how they combine.

    Args:
        path: A logs directory, a single log file, or a ``.zip`` of logs.
        last: Select the most recent worker session.
        session_index: Select worker session #N. Takes precedence over ``last`` when both are given.
        last_benchmark: Select the most recent benchmark run.
        benchmark_index: Select benchmark run #N. Takes precedence over ``last_benchmark``.
        recent: With no other selector, restrict the result to the most recent ``recent`` sessions of
            each kind. Useful for a quick, bounded pass that does not run detectors over a long restart
            history.
        active_only: Read only the live ``bridge.log`` and ``bridge_harness.log`` and skip zipped
            rotations. Pairs with ``recent`` for a fast pass (the recent sessions live in the active
            files); a full history needs the rotations, so leave this off for that.

    Returns:
        One :class:`SessionDiagnosis` per selected session: the worker sessions, then the benchmark
        runs, told apart by ``session.log_kind``. Empty if the path holds no recognizable sessions.
        Detectors never raise (see :func:`~horde_worker_regen.analysis.detectors.run_detectors`), so
        this is safe to call on partial or torn logs.
    """
    bundle = LogBundle.from_path(path)
    sessions = segment_bundle_sessions(bundle, active_only=active_only)
    selected = select_sessions(
        sessions,
        last=last,
        session_index=session_index,
        last_benchmark=last_benchmark,
        benchmark_index=benchmark_index,
        recent=recent,
    )
    return [
        SessionDiagnosis(session=session, findings=run_detectors(build_session_context(session, bundle)))
        for session in selected
    ]


def diagnose_views(
    path: Path = DEFAULT_LOG_PATH,
    *,
    last: bool = False,
    session_index: int | None = None,
    last_benchmark: bool = False,
    benchmark_index: int | None = None,
    recent: int | None = None,
    active_only: bool = False,
) -> list[SessionDiagnosisView]:
    """:func:`diagnose`, but returning lightweight, picklable views (no parsed records).

    This is what a caller runs across a process boundary: the heavy parsed sessions stay (and are
    freed) in the worker process, and only the per-session summaries plus findings are returned.
    """
    results = diagnose(
        path,
        last=last,
        session_index=session_index,
        last_benchmark=last_benchmark,
        benchmark_index=benchmark_index,
        recent=recent,
        active_only=active_only,
    )
    return [SessionDiagnosisView(session=_summarize(result.session), findings=result.findings) for result in results]


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
        log_kind=session.log_kind,
    )


def select_sessions(
    sessions: list[WorkerSession],
    *,
    last: bool,
    session_index: int | None,
    last_benchmark: bool = False,
    benchmark_index: int | None = None,
    recent: int | None = None,
) -> list[WorkerSession]:
    """Apply the session selectors to a session list (the shared CLI, bundle and TUI rule).

    ``--last`` and ``--session N`` address worker sessions only; ``--last-benchmark`` and ``--benchmark N``
    address benchmark runs only, because each kind is numbered on its own. Selectors of both kinds
    combine, worker sessions first. With no selector, every session is returned, or the most recent
    ``recent`` of each kind when ``recent`` is set.
    """
    worker_sessions = [session for session in sessions if not session.is_benchmark_run]
    benchmark_runs = [session for session in sessions if session.is_benchmark_run]
    selects_worker = last or session_index is not None
    selects_benchmark = last_benchmark or benchmark_index is not None
    if not selects_worker and not selects_benchmark:
        if recent is not None and recent > 0:
            return worker_sessions[-recent:] + benchmark_runs[-recent:]
        return worker_sessions + benchmark_runs
    selected: list[WorkerSession] = []
    if selects_worker:
        selected.extend(_select_within_kind(worker_sessions, last=last, index=session_index))
    if selects_benchmark:
        selected.extend(_select_within_kind(benchmark_runs, last=last_benchmark, index=benchmark_index))
    return selected


def _select_within_kind(sessions: list[WorkerSession], *, last: bool, index: int | None) -> list[WorkerSession]:
    """Pick session #``index`` of one kind, or its most recent session when ``last`` is set."""
    if index is not None:
        return [session for session in sessions if session.index == index]
    return sessions[-1:] if last else []
