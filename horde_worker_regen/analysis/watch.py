"""Incremental, alert-on-change watching of a live worker's logs.

Post-mortem ``diagnose`` answers "what went wrong after the fact"; ``watch`` answers "tell me the moment
it starts going wrong" while the worker runs. It re-diagnoses the most recent session on each pass and
emits an alert only for *newly appeared* warning/critical findings and for a rising recovery count, so a
quiet worker stays quiet and a storm announces itself as it builds.

The change-detection is factored into :func:`watch_pass` (pure, testable); the polling loop that calls
it lives in the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .bundle import LogBundle
from .correlate import build_session_context
from .detectors import Severity, run_detectors
from .sessions import segment_sessions


@dataclass
class WatchState:
    """What the watcher has already reported, so it only alerts on changes."""

    session_index: int | None = None
    seen_finding_ids: set[str] = field(default_factory=set)
    last_recovery_count: int = 0


def watch_pass(bundle: LogBundle, state: WatchState) -> tuple[list[str], WatchState]:
    """Diagnose the latest session and return alerts for what changed since ``state``.

    A new session resets the baseline (its findings and recovery count are reported fresh). Within a
    session, only findings not yet seen and an increased recovery count produce alerts.
    """
    sessions = segment_sessions(bundle.orchestrator_records())
    if not sessions:
        return [], state
    session = sessions[-1]

    new_session = session.index != state.session_index
    seen = set() if new_session else set(state.seen_finding_ids)
    baseline_recoveries = 0 if new_session else state.last_recovery_count

    alerts: list[str] = []
    if new_session:
        alerts.append(f"--- session #{session.index} started (v{session.version or '?'}) ---")

    for finding in run_detectors(build_session_context(session, bundle)):
        if finding.severity is Severity.INFO or finding.id in seen:
            continue
        seen.add(finding.id)
        stamp = datetime.now().strftime("%H:%M:%S")
        alerts.append(f"{stamp}  [{finding.severity}] {finding.title}: {finding.verdict}")

    if session.peak_process_recoveries > baseline_recoveries:
        alerts.append(f"process recoveries rose to {session.peak_process_recoveries}")

    return alerts, WatchState(
        session_index=session.index,
        seen_finding_ids=seen,
        last_recovery_count=max(baseline_recoveries, session.peak_process_recoveries),
    )
