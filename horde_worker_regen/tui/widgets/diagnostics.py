"""The diagnostics view: run the ``horde-log`` triage detectors in-TUI and show ranked findings.

This is the in-TUI equivalent of ``horde-log diagnose``: it calls the same first-class facade
(:func:`horde_worker_regen.analysis.diagnose.diagnose`) and renders the structured
:class:`~horde_worker_regen.analysis.detectors.Finding` objects it returns. Findings are rendered
*generically* from their fields (severity, title, verdict, evidence, remediation) rather than per
incident class, so a newly-added detector appears here with no change to this widget -- the only
coupling to the analysis layer is the shape of ``Finding``.

The analysis parses an append-across-restarts log and runs every detector, which is CPU-bound and can
take seconds. On a thread the GIL would starve the TUI event loop, so each pass runs in a worker
*process* and returns only a lightweight, record-free view; the per-session results are cached so the
selector can switch between them instantly without re-parsing.
"""

from __future__ import annotations

import asyncio
import multiprocessing
from concurrent.futures import Executor, ProcessPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, Label, LoadingIndicator, Rule, Select, Static

from horde_worker_regen.analysis.detectors import Finding, Severity
from horde_worker_regen.tui.formatters import human_duration
from horde_worker_regen.tui.log_tailer import LOG_DIR

if TYPE_CHECKING:
    from horde_worker_regen.analysis.diagnose import SessionDiagnosisView

# A displayed analysis older than this is flagged stale: the worker may have logged new incidents since
# the parse, so the findings on screen can no longer be trusted to reflect the current logs.
_STALE_AFTER_SECONDS = 300

# The automatic pass on first tab open diagnoses only the most recent few sessions so it returns
# promptly (and the selector is not flooded with a long restart history).
_AUTO_RUN_RECENT_SESSIONS = 3

# The scope selector's choices, each mapping to the (recent-session-limit, active-log-only) the analysis
# runs with. "current"/"recent" read only the live bridge.log (fast); "all" decompresses every rotation.
_SCOPE_PARAMS: dict[str, tuple[int | None, bool]] = {
    "current": (1, True),
    "recent": (_AUTO_RUN_RECENT_SESSIONS, True),
    "all": (None, False),
}
_SCOPE_LABELS: dict[str, str] = {
    "current": "Current session",
    "recent": f"Last {_AUTO_RUN_RECENT_SESSIONS} sessions",
    "all": "All logs",
}
# The default scope on first open: a quick, bounded pass that never looks hung on a long history.
_DEFAULT_SCOPE = "recent"

# Presentation-only mapping of a finding's severity to a badge label and colour. Kept in the view
# (not the analysis layer) because it is a display choice; severities themselves come from the
# analysis ``Severity`` enum, so a new severity surfaces here as the fallback badge until styled.
_SEVERITY_BADGE: dict[Severity, tuple[str, str]] = {
    Severity.CRITICAL: ("CRITICAL", "bold white on red"),
    Severity.WARNING: ("WARNING", "black on yellow"),
    Severity.INFO: ("INFO", "black on cyan"),
}
_SEVERITY_BORDER: dict[Severity, str] = {
    Severity.CRITICAL: "red",
    Severity.WARNING: "yellow",
    Severity.INFO: "cyan",
}


class DiagnosticsView(Vertical):
    """Run the log-triage detectors over the worker's logs and show ranked, actionable findings."""

    DEFAULT_CSS = """
    DiagnosticsView #diag-controls {
        height: 3;
        padding: 0 1;
    }
    DiagnosticsView .diag-cap {
        width: auto;
        height: 3;
        content-align: left middle;
        margin-right: 1;
        color: $text-muted;
    }
    DiagnosticsView #diag-scope {
        width: 24;
    }
    DiagnosticsView #diag-controls-spacer {
        width: 1fr;
    }
    DiagnosticsView #diag-divider {
        width: 1;
        height: 3;
        margin: 0 1;
    }
    DiagnosticsView #diag-session {
        width: 36;
        margin-right: 1;
    }
    DiagnosticsView #diag-run {
        width: auto;
    }
    DiagnosticsView #diag-status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    DiagnosticsView #diag-timing {
        height: 1;
        padding: 0 1;
    }
    DiagnosticsView #diag-loading {
        height: 1;
        display: none;
    }
    DiagnosticsView #diag-results {
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        """Initialize the empty result cache and run guards."""
        super().__init__()
        self._diagnoses: list[SessionDiagnosisView] = []
        """Cached per-session detector results from the last pass; the selector switches between them."""
        self._run_in_flight = False
        """Guards against a second pass stacking on the first (off-loop parse can take seconds)."""
        self._pool: ProcessPoolExecutor | None = None
        """Lazily-created worker process the analysis runs in, so its CPU never starves the TUI loop."""
        self._updating_select = False
        """Set while repopulating the session selector so its Changed event is ignored."""
        self._analyzed_at: datetime | None = None
        """Wall-clock time the displayed analysis was computed; drives the age/staleness readout."""
        self._pending_recent: int | None = None
        """Session limit for the in-flight pass (set before the worker process starts; read when it does)."""
        self._pending_active_only = False
        """Whether the in-flight pass reads only the live log (set alongside ``_pending_recent``)."""
        self._inflight_scope: str = _DEFAULT_SCOPE
        """The scope key of the pass currently running (recorded so the result can be labelled by it)."""
        self._analyzed_scope: str | None = None
        """The scope key the displayed results were computed with; None until the first run completes."""

    def compose(self) -> ComposeResult:
        """Lay out the session selector, the run button, a status line, and the scrollable results."""
        with Horizontal(id="diag-controls"):
            yield Label("Scope", classes="diag-cap")
            yield Select(
                ((label, key) for key, label in _SCOPE_LABELS.items()),
                value=_DEFAULT_SCOPE,
                allow_blank=False,
                id="diag-scope",
            )
            yield Button("Run analysis", id="diag-run", variant="primary")
            yield Static(id="diag-controls-spacer")
            # The session selector and Run button form the right-hand cluster, set off from the scope
            # control by a divider so it reads as one group: pick a session to view, Run to (re)analyze.
            yield Rule(orientation="vertical", id="diag-divider")
            yield Label("Session", classes="diag-cap")
            yield Select((), prompt="run analysis to list sessions…", id="diag-session", allow_blank=True)
        yield Static(id="diag-status")
        yield Static(id="diag-timing")
        yield LoadingIndicator(id="diag-loading")
        with VerticalScroll():
            yield Static(id="diag-results")

    def on_mount(self) -> None:
        """Show the idle hint, and tick a clock so the analysis age (and staleness) stays current."""
        self._set_status("Choose a scope, then press Run analysis (works whether or not the worker is running).")
        self.query_one("#diag-results", Static).update(
            Text("No analysis yet; press Run analysis to triage the worker logs.", style="grey50"),
        )
        # A 1s tick keeps the "now" clock and the age/staleness readout live without re-parsing; it only
        # rewrites a one-line Static, so it is cheap even while the tab is in the background.
        self.set_interval(1.0, self._refresh_timing)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Run the analysis at the currently-selected scope when the operator asks (the only trigger)."""
        if event.button.id == "diag-run":
            self._run_selected_scope()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Mark a pending run when the scope changes; re-render from cache when the session changes.

        Changing the scope deliberately does *not* start an analysis -- the operator presses Run
        analysis when ready -- so a slow pass is never kicked off by merely browsing the dropdown.
        """
        if self._updating_select:
            return
        if event.select.id == "diag-scope":
            self._indicate_scope_pending(str(event.value))
            return
        # The session option values are indices (ints); the isinstance guard also covers the BLANK case.
        if event.select.id == "diag-session" and isinstance(event.value, int):
            self._render_selected(event.value)

    def _indicate_scope_pending(self, scope: str) -> None:
        """Show that the chosen scope is not applied until Run analysis is pressed."""
        label = _SCOPE_LABELS.get(scope, scope)
        if self._analyzed_scope is not None and scope != self._analyzed_scope:
            self._set_status(f"Scope changed to '{label}'; press Run analysis to apply it.", style="bold yellow")
        else:
            self._set_status(f"Scope: {label}. Press Run analysis to run.")

    def _run_selected_scope(self) -> None:
        """Start an analysis pass at the scope currently chosen in the scope selector."""
        scope = str(self.query_one("#diag-scope", Select).value)
        self._inflight_scope = scope
        recent, active_only = _SCOPE_PARAMS.get(scope, _SCOPE_PARAMS[_DEFAULT_SCOPE])
        self._start_analysis(recent=recent, active_only=active_only)

    def _start_analysis(self, *, recent: int | None, active_only: bool) -> None:
        """Launch a detector pass in a worker process, showing a clear "analyzing" state while it runs.

        The controls grey out *and* a loading indicator plus an explicit message appear, so a slow parse
        on a large log reads as "working", never as a locked-up tab.
        """
        if self._run_in_flight:
            return
        self._run_in_flight = True
        self._pending_recent = recent
        self._pending_active_only = active_only
        self.query_one("#diag-run", Button).disabled = True
        self.query_one("#diag-scope", Select).disabled = True
        self.query_one("#diag-session", Select).disabled = True
        self.query_one("#diag-loading", LoadingIndicator).display = True
        scope = self._scope_text(recent)
        self._set_status(f"Analyzing {scope}…")
        self.query_one("#diag-results", Static).update(
            Text(
                f"Analyzing {scope} in {LOG_DIR}/…\n"
                "This can take a few seconds on a large log. The worker does not need to be running.",
                style="bold",
            ),
        )
        self.run_worker(self._run_analysis(recent, active_only), exclusive=True, group="diagnostics")

    @staticmethod
    def _scope_text(recent: int | None) -> str:
        """Human description of how many sessions a pass covers."""
        if recent is None:
            return "all sessions"
        if recent == 1:
            return "the current session"
        return f"the most recent {recent} sessions"

    async def _run_analysis(self, recent: int | None, active_only: bool) -> None:
        """Run the detectors in a worker *process* and await the result without blocking the UI loop.

        The parse + detectors are CPU-bound pure Python: on a thread the GIL would starve the event
        loop and freeze the TUI, so the work runs in a separate process and only a lightweight
        (record-free) view is returned. ``active_only`` reads just the live log instead of decompressing
        every rotation. Imported lazily so other tabs never pull in the analysis package.
        """
        from horde_worker_regen.analysis.diagnose import diagnose_views_for

        loop = asyncio.get_running_loop()
        try:
            views = await loop.run_in_executor(
                self._analysis_executor(), diagnose_views_for, LOG_DIR, recent, active_only
            )
        except Exception as error:  # noqa: BLE001 - never let a triage failure crash the TUI
            self._on_analysis_error(error)
            return
        self._on_analysis_done(views)

    def _analysis_executor(self) -> Executor | None:
        """The executor the analysis runs in: a lazily-spawned worker process (overridable in tests).

        A single reused spawn-context process keeps the heavy parse out of the TUI process entirely;
        the first run pays the spawn/import cost once, then stays warm for subsequent runs.
        """
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        return self._pool

    def on_unmount(self) -> None:
        """Tear down the worker process when the view goes away."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _on_analysis_done(self, results: list[SessionDiagnosisView]) -> None:
        """UI-thread callback: cache results, repopulate the selector, and show the latest session."""
        self._run_in_flight = False
        self._analyzed_scope = self._inflight_scope
        self.query_one("#diag-run", Button).disabled = False
        self.query_one("#diag-scope", Select).disabled = False
        self.query_one("#diag-loading", LoadingIndicator).display = False
        self._diagnoses = results
        # Stamp when this parse completed (even when empty): the timing line dates the displayed analysis.
        self._analyzed_at = datetime.now()
        self._refresh_timing()
        if not results:
            self.query_one("#diag-session", Select).disabled = True
            self._set_status(f"No worker sessions found in {LOG_DIR}.")
            self.query_one("#diag-results", Static).update(
                Text(
                    f"No logs found in {LOG_DIR}/. Start the worker to generate logs, then run analysis.",
                    style="grey50",
                ),
            )
            return
        self._populate_session_select(results)
        latest_index = results[-1].session.index
        self._render_selected(latest_index)

    def _on_analysis_error(self, error: Exception) -> None:
        """UI-thread callback: report a triage failure and re-enable the controls."""
        self._run_in_flight = False
        self.query_one("#diag-run", Button).disabled = False
        self.query_one("#diag-scope", Select).disabled = False
        self.query_one("#diag-loading", LoadingIndicator).display = False
        self.query_one("#diag-session", Select).disabled = bool(not self._diagnoses)
        self._set_status(f"Analysis failed: {type(error).__name__}: {error}")
        self.notify(f"Diagnostics failed: {type(error).__name__}: {error}", severity="error")

    def _populate_session_select(self, results: list[SessionDiagnosisView]) -> None:
        """Repopulate the session selector (latest last, default-selected) from the diagnosed sessions."""
        select = self.query_one("#diag-session", Select)
        last_pos = len(results) - 1
        options = [
            (self._session_label(diagnosis, is_latest=pos == last_pos), diagnosis.session.index)
            for pos, diagnosis in enumerate(results)
        ]
        self._updating_select = True
        try:
            select.set_options(options)
            select.disabled = False
            select.value = results[-1].session.index
        finally:
            self._updating_select = False

    def _session_label(self, diagnosis: SessionDiagnosisView, *, is_latest: bool) -> str:
        """A one-line selector label: position, worker version, and how the session ended."""
        session = diagnosis.session
        position = "latest" if is_latest else f"#{session.index}"
        version = f"v{session.version}" if session.version else "v?"
        worst = self._worst_severity(diagnosis)
        flag = f"; {worst.value}" if worst is not None else ""
        return f"{position}  {version}  ({session.end_reason}){flag}"

    @staticmethod
    def _worst_severity(diagnosis: SessionDiagnosisView) -> Severity | None:
        """The most-severe finding severity in a session (findings are already sorted), or None."""
        return diagnosis.findings[0].severity if diagnosis.findings else None

    def _render_selected(self, session_index: int) -> None:
        """Render the findings for the session with ``session_index`` from the cache."""
        diagnosis = next((d for d in self._diagnoses if d.session.index == session_index), None)
        results = self.query_one("#diag-results", Static)
        if diagnosis is None:
            results.update(Text("Session no longer available; run analysis again.", style="grey50"))
            return
        session = diagnosis.session
        scope_label = _SCOPE_LABELS.get(self._analyzed_scope or "", "")
        self._set_status(
            f"Session #{session.index} ({session.end_reason}); {len(diagnosis.findings)} finding(s). "
            f"Scope: {scope_label}. Change scope or Run analysis to refresh.",
        )
        if not diagnosis.findings:
            results.update(Text("No findings; this session looks clean.", style="green"))
            return
        results.update(Group(*(self._render_finding(finding) for finding in diagnosis.findings)))

    def _render_finding(self, finding: Finding) -> Panel:
        """Render one finding as a severity-bordered panel with the diagnosis and the fix set apart.

        The three parts get distinct visual treatments so the *diagnosis* (what went wrong) is never
        confused with the *suggestion* (what to do): the verdict reads as plain body text, the evidence
        is a dim sub-block, and the remediation is a green, separately-headed "Suggested fix" block.
        """
        label, badge_style = _SEVERITY_BADGE.get(finding.severity, (finding.severity.value.upper(), "bold"))
        border = _SEVERITY_BORDER.get(finding.severity, "grey50")

        sections: list[Text] = [Text.assemble(("Diagnosis  ", "grey46 bold"), (finding.verdict, "default"))]
        if finding.evidence:
            evidence = Text("Evidence", style="grey46 bold")
            for line in finding.evidence:
                evidence.append(f"\n  • {line}", style="grey58")
            sections.append(evidence)
        if finding.remediation:
            fix = Text()
            fix.append("→ Suggested fix\n", style="bold green")
            fix.append(f"   {finding.remediation}", style="green")
            sections.append(fix)
        if finding.see_also:
            sections.append(Text(f"see also: {finding.see_also}", style="grey50 italic"))

        title = Text.assemble((f" {label} ", badge_style), ("  ", ""), (finding.title, "bold"))
        return Panel(Group(*self._blank_separated(sections)), title=title, title_align="left", border_style=border)

    @staticmethod
    def _blank_separated(sections: list[Text]) -> list[Text]:
        """Interleave a blank line between sections so the diagnosis/evidence/fix blocks read apart."""
        spaced: list[Text] = []
        for index, section in enumerate(sections):
            if index:
                spaced.append(Text(""))
            spaced.append(section)
        return spaced

    def _refresh_timing(self) -> None:
        """Update the one-line timing readout: when the analysis ran, the time now, and its age.

        Once the displayed analysis is older than :data:`_STALE_AFTER_SECONDS`, the line flips to a bold
        stale warning, because the worker may have logged new incidents the on-screen findings predate.
        """
        try:
            timing = self.query_one("#diag-timing", Static)
        except NoMatches:
            return
        if self._analyzed_at is None:
            timing.update("")
            return
        now = datetime.now()
        age = (now - self._analyzed_at).total_seconds()
        analyzed_text = self._analyzed_at.strftime("%Y-%m-%d %H:%M:%S")
        now_text = now.strftime("%H:%M:%S")
        if age >= _STALE_AFTER_SECONDS:
            line = Text()
            line.append(" STALE ", style="bold white on red")
            line.append(
                f"  analysis is {human_duration(age)} old (>5 min); press Run analysis to refresh.  ",
                style="bold yellow",
            )
            line.append(f"Analyzed {analyzed_text} · now {now_text}", style="yellow")
        else:
            line = Text(
                f"Analyzed {analyzed_text}  ·  now {now_text}  ·  {human_duration(age)} ago",
                style="grey50",
            )
        timing.update(line)

    def _set_status(self, message: str, *, style: str | None = None) -> None:
        """Update the one-line status above the results (styled when an attention cue is wanted)."""
        self.query_one("#diag-status", Static).update(Text(message, style=style) if style else message)
