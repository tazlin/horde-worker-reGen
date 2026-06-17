"""The benchmark view: launch a ramp, watch it live, and apply its recommended config.

Renders from the [`BenchmarkRunState`][horde_worker_regen.tui.benchmark_launcher.BenchmarkRunState] the
supervisor accumulates by tailing the run's progress file. The widget itself owns no process: it posts
``RunRequested`` / ``CancelRequested`` / ``ApplyConfigRequested`` messages and lets the app coordinate the
GPU-exclusive worker/benchmark hand-off.
"""

from __future__ import annotations

import contextlib

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Input, Label, Static, Switch

from horde_worker_regen import __version__
from horde_worker_regen.app_state import AppStateStore, BenchmarkAvailability, benchmark_status_summary
from horde_worker_regen.tui.benchmark_launcher import (
    BenchmarkOptions,
    BenchmarkRunState,
    BenchmarkSupervisorStatus,
    LevelState,
)

_OUTCOME_COLOURS: dict[str, str] = {
    "passed": "green",
    "failed": "red",
    "crashed": "red",
    "crashed_hang": "red",
    "skipped": "grey50",
}


class BenchmarkView(VerticalScroll):
    """Launch and monitor a benchmark ramp, then apply the recommended bridgeData."""

    DEFAULT_CSS = """
    BenchmarkView #benchmark-actions {
        height: 3;
        padding: 0 1;
    }
    BenchmarkView #benchmark-actions Button {
        margin-right: 1;
    }
    BenchmarkView #benchmark-options {
        height: 3;
        padding: 0 1;
    }
    BenchmarkView #benchmark-options Label {
        content-align: left middle;
        height: 3;
        padding: 0 1;
    }
    BenchmarkView #benchmark-options Input {
        width: 22;
    }
    BenchmarkView .benchmark-body {
        padding: 1 1;
    }
    """

    class RunRequested(Message):
        """Posted when the user asks to start a benchmark (the app supplies the process mode)."""

        def __init__(self, options: BenchmarkOptions) -> None:
            """Carry the user-chosen ramp options."""
            super().__init__()
            self.options = options

    class CancelRequested(Message):
        """Posted when the user asks to cancel the running benchmark."""

    class ApplyConfigRequested(Message):
        """Posted when the user accepts the suggested bridgeData."""

    class RestoreKnownGoodRequested(Message):
        """Posted when the user asks to restore the last known-good configuration."""

    def __init__(self, *, worker_mode: str) -> None:
        """Store the worker's process mode, which decides the benchmark's default process mode."""
        super().__init__()
        self._worker_mode = worker_mode
        self._app_state_summary: Text = Text("Loading benchmark status…", style="grey70")
        self._has_known_good = False

    def compose(self) -> ComposeResult:
        """Lay out the action bar, the options row, and the live body panel."""
        with Horizontal(id="benchmark-actions"):
            yield Button("Run benchmark", id="benchmark-run", variant="success")
            yield Button("Cancel", id="benchmark-cancel", variant="warning")
            yield Button("Apply suggested config", id="benchmark-apply", variant="primary")
            yield Button("Restore last-known-good", id="benchmark-restore", variant="default")
        with Horizontal(id="benchmark-options"):
            yield Label("Tiers")
            yield Input(value="sd15,sdxl", id="benchmark-tiers")
            yield Label("Soak (min)")
            yield Input(value="5", id="benchmark-soak", type="number")
            yield Label("Validate")
            yield Switch(value=True, id="benchmark-validate")
            yield Label("Downloads")
            yield Switch(value=False, id="benchmark-downloads")
            yield Label("Verbose")
            yield Switch(value=False, id="benchmark-verbose")
        yield Static(self._app_state_summary, id="benchmark-status", classes="benchmark-body")
        yield Static(id="benchmark-body", classes="benchmark-body")

    def on_mount(self) -> None:
        """Load the persisted benchmark status and render the initial idle view."""
        self.refresh_app_state_summary()
        self.update_view(BenchmarkRunState(), BenchmarkSupervisorStatus.IDLE)

    def refresh_app_state_summary(self) -> None:
        """Re-read durable app state and update the persisted-status line."""
        self._app_state_summary = self._build_app_state_summary()
        with contextlib.suppress(NoMatches):
            self.query_one("#benchmark-status", Static).update(self._app_state_summary)

    def update_view(self, run_state: BenchmarkRunState, status: BenchmarkSupervisorStatus) -> None:
        """Refresh the action buttons and the body panel from the supervisor's latest state."""
        self._update_buttons(status, run_state)
        self.query_one("#benchmark-body", Static).update(self._render_body(run_state, status))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Translate the action buttons into messages the app coordinates."""
        if event.button.id == "benchmark-run":
            self.post_message(self.RunRequested(self._collect_options()))
        elif event.button.id == "benchmark-cancel":
            self.post_message(self.CancelRequested())
        elif event.button.id == "benchmark-apply":
            self.post_message(self.ApplyConfigRequested())
        elif event.button.id == "benchmark-restore":
            self.post_message(self.RestoreKnownGoodRequested())

    def _collect_options(self) -> BenchmarkOptions:
        """Build ramp options from the option widgets (the app overrides the process mode)."""
        tiers_raw = self.query_one("#benchmark-tiers", Input).value
        tiers = [tier.strip() for tier in tiers_raw.split(",") if tier.strip()] or ["sd15", "sdxl"]
        try:
            soak_minutes = float(self.query_one("#benchmark-soak", Input).value or "5")
        except ValueError:
            soak_minutes = 5.0
        return BenchmarkOptions(
            tiers=tiers,
            process_mode=self._worker_mode,
            validate=self.query_one("#benchmark-validate", Switch).value,
            soak_minutes=soak_minutes,
            include_downloads=self.query_one("#benchmark-downloads", Switch).value,
            verbose=self.query_one("#benchmark-verbose", Switch).value,
        )

    def _update_buttons(self, status: BenchmarkSupervisorStatus, run_state: BenchmarkRunState) -> None:
        """Enable only the actions valid for the current status."""
        running = status is BenchmarkSupervisorStatus.RUNNING
        # PREPARING (worker being stopped, before the subprocess launches) is busy too: block a second
        # Run and the worker-restarting Restore, but there is no subprocess yet to Cancel.
        active = status in (BenchmarkSupervisorStatus.PREPARING, BenchmarkSupervisorStatus.RUNNING)
        has_suggestion = bool(run_state.suggested_bridge_data_yaml)
        self.query_one("#benchmark-run", Button).disabled = active
        self.query_one("#benchmark-cancel", Button).disabled = not running
        self.query_one("#benchmark-apply", Button).disabled = not (
            status is BenchmarkSupervisorStatus.FINISHED and has_suggestion
        )
        self.query_one("#benchmark-restore", Button).disabled = active or not self._has_known_good

    def _build_app_state_summary(self) -> Text:
        """Render the persisted last-benchmark status (and any known-good config) as a short summary."""
        state = AppStateStore().load()
        self._has_known_good = state.last_known_good_settings is not None
        availability = benchmark_status_summary(state, current_version=__version__)

        if availability is BenchmarkAvailability.NONE:
            summary = Text("No benchmark on record; run one to auto-tune this worker.", style="yellow")
        else:
            benchmark = state.last_benchmark
            assert benchmark is not None  # NONE is the only case with no record
            badge_style = "black on yellow" if availability is BenchmarkAvailability.STALE else "black on green"
            badge_text = " STALE (version changed) " if availability is BenchmarkAvailability.STALE else " CURRENT "
            summary = Text.assemble(
                Text(badge_text, style=badge_style),
                "  ",
                Text.from_markup(f"[grey62]last run[/] {benchmark.run_id}"),
                "   ",
                Text.from_markup(f"[grey62]passed[/] {benchmark.levels_passed}/{benchmark.levels_total}"),
                "   ",
                Text.from_markup(f"[grey62]version[/] {benchmark.worker_version}"),
            )

        known_good = state.last_known_good_settings
        if known_good is not None:
            summary.append("\n")
            summary.append_text(
                Text.from_markup(
                    f"[grey62]known-good[/] {known_good.source.value} on v{known_good.worker_version} "
                    "- restorable below",
                ),
            )
        return summary

    def _render_body(self, run_state: BenchmarkRunState, status: BenchmarkSupervisorStatus) -> RenderableType:
        """Render the headline, current-level card, per-level table, and (when done) the recommendation."""
        if status is BenchmarkSupervisorStatus.IDLE and not run_state.level_order:
            return self._render_idle_hint()

        # Before any level exists (worker stop, then the subprocess's import + hardware-probe window),
        # show the startup phase so the slow hand-off reads as motion rather than a frozen blank tab.
        if not run_state.level_order and run_state.startup_phase:
            return self._render_starting(run_state, status)

        sections: list[RenderableType] = [self._render_headline(run_state, status)]
        current = run_state.current_level_id
        if current is not None and current in run_state.levels:
            sections.append(self._render_current_level(run_state.levels[current]))
        if run_state.level_order:
            sections.append(self._render_level_table(run_state))
        if run_state.finished and run_state.suggested_bridge_data_yaml:
            sections.append(self._render_suggestion(run_state))
        return Group(*sections)

    @staticmethod
    def _render_idle_hint() -> Panel:
        """The pre-run message describing what a benchmark does and its GPU-exclusivity caveat."""
        body = Text.assemble(
            (
                "A benchmark ramps the worker through safe difficulty levels, suggests a tuned bridgeData, "
                "and flags robustness problems.\n\n",
                "grey70",
            ),
            ("Running it stops the worker (the benchmark needs the GPU). Press ", "grey70"),
            ("Run benchmark", "bold green"),
            (" to start.", "grey70"),
        )
        return Panel(body, title="Benchmark", title_align="left", border_style="cyan")

    @staticmethod
    def _render_starting(run_state: BenchmarkRunState, status: BenchmarkSupervisorStatus) -> Panel:
        """The pre-level startup card: shows the current phase during the worker-stop and import window."""
        run_id = run_state.run_id or "-"
        body = Text.assemble(
            (f" {status.value.upper()} ", "black on yellow"),
            "  ",
            (run_id, "bold"),
            "\n\n",
            (run_state.startup_phase or "Starting…", "yellow"),
            "\n\n",
            ("This can take a minute on a cold start (importing the inference stack and probing the GPU). ", "grey70"),
            ("Live detail is written to the run's ", "grey70"),
            ("console.log", "grey85"),
            (".", "grey70"),
        )
        return Panel(body, title="Benchmark starting", title_align="left", border_style="yellow")

    @staticmethod
    def _render_headline(run_state: BenchmarkRunState, status: BenchmarkSupervisorStatus) -> Panel:
        """A one-line summary of the run's identity, mode, and overall progress."""
        finished_levels = sum(1 for level in run_state.levels.values() if level.outcome is not None)
        total = run_state.num_levels or run_state.levels_total or len(run_state.level_order)
        gpu = run_state.gpu_name or "unknown GPU"
        status_colour = {
            BenchmarkSupervisorStatus.PREPARING: "yellow",
            BenchmarkSupervisorStatus.RUNNING: "yellow",
            BenchmarkSupervisorStatus.FINISHED: "green",
            BenchmarkSupervisorStatus.FAILED: "red",
            BenchmarkSupervisorStatus.CANCELLED: "grey50",
            BenchmarkSupervisorStatus.IDLE: "grey50",
        }.get(status, "white")
        body = Text.assemble(
            (f" {status.value.upper()} ", f"black on {status_colour}"),
            "  ",
            (f"{run_state.run_id or '-'}", "bold"),
            (f"   mode={run_state.process_mode or '-'}   {gpu}   ", "grey62"),
            (f"levels {finished_levels}/{total}", "bold cyan"),
        )
        return Panel(body, border_style=status_colour)

    @staticmethod
    def _render_current_level(level: LevelState) -> Panel:
        """A live metric card for the level currently running."""
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan")
        table.add_column()
        jobs = (
            f"{level.jobs_completed}/{level.jobs_expected}"
            if level.jobs_expected is not None
            else str(level.jobs_completed)
        )
        table.add_row("Level", f"{level.level_id} ({level.stage}/{level.tier}/{level.axis})")
        if level.phase:
            table.add_row("Status", Text(level.phase, style="yellow"))
        table.add_row("Jobs", jobs + (f"  ({level.jobs_faulted} faulted)" if level.jobs_faulted else ""))
        table.add_row("it/s", "-" if level.iterations_per_second is None else f"{level.iterations_per_second:.2f}")
        table.add_row("VRAM", "-" if level.vram_used_mb is None else f"{level.vram_used_mb} MB")
        table.add_row("GPU busy", "-" if level.gpu_busy_percent is None else f"{level.gpu_busy_percent:.0f}%")
        table.add_row("Elapsed", f"{level.elapsed_seconds:.0f}s")
        if level.num_process_recoveries:
            table.add_row("Restarts", Text(f"{level.num_process_recoveries} (!)", style="bold red"))
        return Panel(table, title="Current level", title_align="left", border_style="cyan")

    def _render_level_table(self, run_state: BenchmarkRunState) -> Panel:
        """A per-level verdict table built up as levels finish."""
        table = Table(expand=True)
        table.add_column("Level")
        table.add_column("Stage/Tier")
        table.add_column("Outcome")
        table.add_column("it/s p50", justify="right")
        table.add_column("Notes")
        for level in run_state.ordered_levels():
            outcome = level.outcome or ("running" if run_state.current_level_id == level.level_id else "pending")
            colour = _OUTCOME_COLOURS.get(level.outcome or "", "yellow")
            its = "-" if level.iterations_per_second is None else f"{level.iterations_per_second:.2f}"
            notes = "; ".join(level.reasons + level.advisories)
            table.add_row(
                level.level_id,
                f"{level.stage}/{level.tier}",
                Text(outcome, style=colour),
                its,
                notes,
            )
        return Panel(table, title="Levels", title_align="left", border_style="grey37")

    @staticmethod
    def _render_suggestion(run_state: BenchmarkRunState) -> Panel:
        """The recommended bridgeData plus the run totals and findings count."""
        findings = f"  ·  {run_state.num_findings} robustness findings" if run_state.num_findings else ""
        header = Text.from_markup(
            f"[bold]{run_state.levels_passed}/{run_state.levels_total}[/] levels passed{findings}\n"
            "Press [bold green]Apply suggested config[/] to write these into bridgeData.yaml.\n",
        )
        body = Group(header, Text(run_state.suggested_bridge_data_yaml, style="grey82"))
        return Panel(body, title="Suggested bridgeData", title_align="left", border_style="green")
