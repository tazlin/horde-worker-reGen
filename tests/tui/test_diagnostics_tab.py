"""Lightweight tests for the Diagnostics tab: it renders findings (and the empty state) off-loop.

These drive the real widget through its analysis path but stub the off-process entry point and force an
in-process executor, so they exercise the tab's wiring (open -> run -> render, the session selector,
timing/staleness) without spawning a worker process or parsing real logs. The generic rendering means a
new detector needs no change here: the tab is asserted to render whatever findings the facade returns.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from rich.text import Text
from textual.widgets import Select, Static, TabbedContent

from horde_worker_regen.analysis.detectors import Finding, Severity
from horde_worker_regen.analysis.diagnose import SessionDiagnosisView, SessionSummary
from horde_worker_regen.app_state import AppStateStore
from horde_worker_regen.process_management.ipc.supervisor_channel import WorkerConfigSummary, WorkerStateSnapshot
from horde_worker_regen.tui.app import HordeWorkerTUI
from horde_worker_regen.tui.widgets.diagnostics import DiagnosticsView
from tests.tui._fake_supervisor import FakeSupervisor


def _diagnosis(index: int, *findings: Finding, version: str = "12.29.0") -> SessionDiagnosisView:
    """A lightweight session-diagnosis view (what the off-process pass returns) with the given findings."""
    session = SessionSummary(
        index=index,
        version=version,
        end_reason="gave_up_aborted",
        num_models=None,
        max_threads=None,
        peak_process_recoveries=0,
    )
    return SessionDiagnosisView(session=session, findings=list(findings))


def _finding(finding_id: str, severity: Severity) -> Finding:
    """A minimal finding with distinct, assertable text."""
    return Finding(
        id=finding_id,
        severity=severity,
        title=f"Title for {finding_id}",
        verdict=f"Verdict for {finding_id}",
        remediation=f"Fix {finding_id}",
    )


def _text(widget: Static) -> str:
    """The plain text a Static is currently rendering."""
    rendered = widget.render()
    return rendered.plain if isinstance(rendered, Text) else str(rendered)


def _make_app(tmp_path: Path, *, alive: bool = True) -> HordeWorkerTUI:
    """A TUI bound to a fake (non-spawning) supervisor with a minimal snapshot."""
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    supervisor = FakeSupervisor(alive=alive)
    supervisor.latest_snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Diag", worker_version="12.29.0"),
    )
    return HordeWorkerTUI(supervisor, config_path=Path("bridgeData.yaml"), app_state_store=store)


def _activate_diagnostics(app: HordeWorkerTUI) -> DiagnosticsView:
    """Open the Diagnostics tab, forcing in-process execution (no spawned worker in tests).

    Returning None from the executor makes the widget use the default thread executor, in-process, so
    the monkeypatched entry point (same process) is the function actually invoked. Opening the tab does
    *not* start an analysis (it is deferred to Run analysis), so the override timing is not delicate.
    """
    view = app.query_one(DiagnosticsView)
    view._analysis_executor = lambda: None  # type: ignore[method-assign]
    app.query_one("#main-tabs", TabbedContent).active = "tab-diagnostics"
    return view


async def _wait_for_analysis(pilot: object, view: DiagnosticsView) -> str:
    """Drive the loop until the analysis pass finishes, returning the terminal status text."""
    for _ in range(100):
        await pilot.pause()  # type: ignore[attr-defined]
        status = _text(view.query_one("#diag-status", Static))
        if not view._run_in_flight and ("finding(s)" in status or "No worker sessions" in status):
            return status
        await asyncio.sleep(0.02)
    raise AssertionError("diagnostics analysis did not complete")


async def _run_and_wait(pilot: object, view: DiagnosticsView) -> str:
    """Press Run analysis (at the current scope) and wait for the pass to finish."""
    view._run_selected_scope()
    return await _wait_for_analysis(pilot, view)


async def test_diagnostics_tab_renders_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening the tab runs the analysis off the loop and renders the latest session's findings generically."""
    results = [
        _diagnosis(0, _finding("oom", Severity.CRITICAL)),
        _diagnosis(1, _finding("forced_maintenance", Severity.CRITICAL), _finding("session_summary", Severity.INFO)),
    ]
    monkeypatch.setattr("horde_worker_regen.analysis.diagnose.diagnose_views_for", lambda *a: results)

    app = _make_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        view = _activate_diagnostics(app)
        status = await _run_and_wait(pilot, view)

        # The latest session (#1, two findings) is shown by default; the selector lists both sessions.
        assert "Session #1" in status and "2 finding(s)" in status
        assert view._diagnoses == results
        select = view.query_one("#diag-session", Select)
        assert select.disabled is False
        assert select.value == 1

        # Switching to the earlier session re-renders from the cache (no re-parse) without error.
        view._render_selected(0)
        assert "Session #0" in _text(view.query_one("#diag-status", Static))


async def test_diagnostics_tab_empty_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no sessions in the logs, the tab shows the no-logs guidance and disables the selector."""
    monkeypatch.setattr("horde_worker_regen.analysis.diagnose.diagnose_views_for", lambda *a: [])

    app = _make_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        view = _activate_diagnostics(app)
        status = await _run_and_wait(pilot, view)

        assert "No worker sessions" in status
        assert view.query_one("#diag-session", Select).disabled is True
        assert "No logs found" in _text(view.query_one("#diag-results", Static))


async def test_scope_change_defers_until_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing the scope does not start an analysis; Run analysis applies whatever scope is selected."""
    from horde_worker_regen.tui.widgets.diagnostics import _DEFAULT_SCOPE, _SCOPE_PARAMS

    calls: list[tuple[int | None, bool]] = []

    def _fake(_path: Path, recent: int | None, active_only: bool) -> list[SessionDiagnosisView]:
        calls.append((recent, active_only))
        return [_diagnosis(0, _finding("oom", Severity.CRITICAL))]

    monkeypatch.setattr("horde_worker_regen.analysis.diagnose.diagnose_views_for", _fake)

    app = _make_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        view = _activate_diagnostics(app)

        # Selecting a scope must NOT run anything; it only shows a "press Run analysis" pending hint.
        view.query_one("#diag-scope", Select).value = "all"
        await pilot.pause()
        assert calls == []
        assert "press run analysis" in _text(view.query_one("#diag-status", Static)).lower()

        # Run applies the selected scope (All logs -> full history).
        await _run_and_wait(pilot, view)
        assert calls == [_SCOPE_PARAMS["all"]] == [(None, False)]

        # Switching scope again still defers until the next Run.
        view.query_one("#diag-scope", Select).value = "current"
        await pilot.pause()
        assert calls == [(None, False)]  # unchanged: no new run on the dropdown change
        await _run_and_wait(pilot, view)
        assert calls[-1] == _SCOPE_PARAMS["current"] == (1, True)

    assert _DEFAULT_SCOPE in _SCOPE_PARAMS  # the default is a real scope key


async def test_diagnostics_runs_with_worker_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Triage reads logs off disk, so it works when the worker is not running (the common case)."""
    results = [_diagnosis(0, _finding("oom", Severity.CRITICAL))]
    monkeypatch.setattr("horde_worker_regen.analysis.diagnose.diagnose_views_for", lambda *a: results)

    app = _make_app(tmp_path, alive=False)  # worker stopped
    async with app.run_test(size=(120, 40)) as pilot:
        assert app._supervisor.is_alive() is False
        view = _activate_diagnostics(app)
        status = await _run_and_wait(pilot, view)
        assert "Session #0" in status and view._diagnoses == results


async def test_diagnostics_timing_shows_analysis_and_current_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a pass, the timing line dates the analysis, shows the current time, and its age."""
    results = [_diagnosis(0, _finding("oom", Severity.CRITICAL))]
    monkeypatch.setattr("horde_worker_regen.analysis.diagnose.diagnose_views_for", lambda *a: results)

    app = _make_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        view = _activate_diagnostics(app)
        await _run_and_wait(pilot, view)

        assert view._analyzed_at is not None
        timing = _text(view.query_one("#diag-timing", Static))
        assert "Analyzed" in timing and "now" in timing and "ago" in timing
        assert "STALE" not in timing


async def test_diagnostics_timing_flags_stale_after_five_minutes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An analysis older than five minutes flips the timing line to a clear stale warning."""
    from datetime import datetime, timedelta

    results = [_diagnosis(0, _finding("oom", Severity.CRITICAL))]
    monkeypatch.setattr("horde_worker_regen.analysis.diagnose.diagnose_views_for", lambda *a: results)

    app = _make_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        view = _activate_diagnostics(app)
        await _run_and_wait(pilot, view)

        # Backdate the analysis past the 5-minute threshold; the next clock tick repaints it as stale.
        view._analyzed_at = datetime.now() - timedelta(minutes=6)
        view._refresh_timing()
        timing = _text(view.query_one("#diag-timing", Static))
        assert "STALE" in timing and "old (>5 min)" in timing


async def test_diagnostics_tab_reruns_on_button(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Run analysis button triggers a fresh pass that picks up newly-appeared findings."""
    state: dict[str, list[SessionDiagnosisView]] = {"results": []}
    monkeypatch.setattr("horde_worker_regen.analysis.diagnose.diagnose_views_for", lambda *a: state["results"])

    app = _make_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        view = _activate_diagnostics(app)
        await _run_and_wait(pilot, view)
        assert view._diagnoses == []

        # A new incident appears in the logs; the operator re-runs and the tab reflects it.
        state["results"] = [_diagnosis(0, _finding("oom", Severity.CRITICAL))]
        status = await _run_and_wait(pilot, view)
        assert "Session #0" in status and len(view._diagnoses) == 1
