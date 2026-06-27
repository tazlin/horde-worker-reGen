"""Stats tab: worker-owned counters, rollups, trend detail, and JSONL export control."""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    PopGovernorsSnapshot,
    StatsRollupRow,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.formatters import format_percent, human_bytes, human_duration, short_baseline, shorten


class StatsView(Vertical):
    """Expanded statistics home for session counters, job rollups, and stats export."""

    class ExportToggled(Message):
        """Request that the worker enable or disable stats JSONL export."""

        def __init__(self, enabled: bool) -> None:
            """Store the requested export state."""
            super().__init__()
            self.enabled = enabled

    DEFAULT_CSS = """
    StatsView #stats-body {
        height: 1fr;
    }
    StatsView #stats-export-button {
        width: auto;
        margin-bottom: 1;
    }
    """

    def __init__(self) -> None:
        """Initialize the tab with no snapshot yet."""
        super().__init__()
        self._snapshot: WorkerStateSnapshot | None = None

    def compose(self) -> ComposeResult:
        """Lay out export control and the scrollable stats body."""
        yield Button("Enable JSONL export", id="stats-export-button")
        with VerticalScroll(id="stats-body"):
            yield Static(id="stats-headlines")
            yield Static(id="stats-governors")
            yield Static(id="stats-export")
            yield Static(id="stats-by-model")
            yield Static(id="stats-by-baseline")

    def update_snapshot(self, snapshot: WorkerStateSnapshot | None) -> None:
        """Refresh all stats panels from the latest worker snapshot."""
        self._snapshot = snapshot
        if snapshot is None:
            placeholder = Panel(
                Text("Waiting for worker snapshot.", style="grey62"), title="Stats", border_style="grey37"
            )
            self.query_one("#stats-headlines", Static).update(placeholder)
            return
        button = self.query_one("#stats-export-button", Button)
        button.label = "Disable JSONL export" if snapshot.stats_export.enabled else "Enable JSONL export"
        self.query_one("#stats-headlines", Static).update(self._render_headlines(snapshot))
        governors_static = self.query_one("#stats-governors", Static)
        has_governors = bool(snapshot.pop_governors.governors)
        governors_static.display = has_governors
        if has_governors:
            governors_static.update(self._render_governors(snapshot.pop_governors))
        self.query_one("#stats-export", Static).update(self._render_export(snapshot))
        self.query_one("#stats-by-model", Static).update(
            self._render_rollups("By model totals", snapshot.stats_model_rollups)
        )
        self.query_one("#stats-by-baseline", Static).update(
            self._render_rollups("By baseline totals", snapshot.stats_baseline_rollups),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Translate the export button into a typed request handled by the app."""
        if event.button.id != "stats-export-button" or self._snapshot is None:
            return
        self.post_message(self.ExportToggled(not self._snapshot.stats_export.enabled))

    @staticmethod
    def _render_headlines(snapshot: WorkerStateSnapshot) -> Panel:
        sample = snapshot.latest_stats_sample
        grid = Table.grid(padding=(0, 3))
        grid.add_column(style="bold cyan", no_wrap=True)
        grid.add_column(no_wrap=True)
        rows = [
            ("Jobs", f"{snapshot.num_jobs_submitted:,} submitted / {snapshot.num_jobs_faulted:,} faulted"),
            ("Kudos/hr", "-" if snapshot.kudos_per_hour is None else f"{snapshot.kudos_per_hour:,.0f}"),
            ("GPU duty", format_percent(snapshot.gpu_utilization_mean_percent)),
            ("Recoveries", f"{snapshot.num_process_recoveries:,}"),
            ("Slowdowns", f"{snapshot.num_job_slowdowns:,}"),
            ("No-work time", human_duration(snapshot.time_spent_no_jobs_available)),
            ("Pipeline", f"{snapshot.jobs_pending_inference} queued / {snapshot.jobs_in_progress} in progress"),
        ]
        if snapshot.config.alchemist:
            rows.append(
                (
                    "Alchemy",
                    f"{snapshot.alchemy_total_submitted:,} submitted / {snapshot.alchemy_total_faulted:,} faulted",
                ),
            )
        if sample is not None:
            rows.append(("Last sample", human_duration(max(0.0, snapshot.timestamp - sample.timestamp)) + " ago"))
        for label, value in rows:
            grid.add_row(label, value)
        return Panel(grid, title="Session stats", title_align="left", border_style="grey37", padding=(0, 1))

    @staticmethod
    def _render_export(snapshot: WorkerStateSnapshot) -> Panel:
        export = snapshot.stats_export
        lines: list[Text] = []
        state = "enabled" if export.enabled else "off"
        style = "green" if export.enabled else "grey62"
        lines.append(Text.assemble(("Export ", "grey50"), (state, f"bold {style}")))
        if export.active_file_path:
            lines.append(Text.assemble(("File ", "grey50"), (export.active_file_path, "grey70")))
        lines.append(Text.assemble(("Stats files ", "grey50"), (human_bytes(export.bytes_in_stats_files), "grey70")))
        if export.warning_over_50_mib:
            lines.append(
                Text("Stats JSONL files exceed 50 MiB; remove old files when you no longer need them.", style="yellow")
            )
        if export.last_write_error:
            lines.append(Text(f"Last write error: {export.last_write_error}", style="red"))
        return Panel(Group(*lines), title="JSONL export", title_align="left", border_style="grey37", padding=(0, 1))

    @staticmethod
    def _render_governors(governors: PopGovernorsSnapshot) -> Panel:
        """Render the per-governor session aggregates: how often each engaged and how long it held.

        Lets an operator compare how much of the session each pop/scheduling governor consumed (the share is
        the clearest signal of which condition is shaping throughput), with the currently-engaged ones marked.
        Ordered by the registry: active governors first, then by total time.
        """
        table = Table(expand=True, border_style="grey37", header_style="bold")
        table.add_column("Governor", no_wrap=True)
        table.add_column("State", no_wrap=True)
        table.add_column("Times", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("% session", justify="right")
        for governor in governors.governors:
            if governor.active:
                if governor.expected_remaining_seconds is not None:
                    state = Text(
                        f"active (~{human_duration(governor.expected_remaining_seconds)} left)", style="yellow"
                    )
                else:
                    state = Text(f"active ({human_duration(governor.current_spell_seconds)})", style="yellow")
            else:
                state = Text("idle", style="grey50")
            table.add_row(
                governor.label,
                state,
                f"{governor.triggers:,}",
                human_duration(governor.total_active_seconds),
                f"{governor.fraction_of_session * 100:.1f}%",
            )
        return Panel(table, title="Pop governors", title_align="left", border_style="grey37", padding=(0, 1))

    @staticmethod
    def _render_rollups(title: str, rows: list[StatsRollupRow]) -> Panel:
        table = Table(expand=True, border_style="grey37", header_style="bold")
        is_model_table = title == "By model totals"
        first = "Model" if is_model_table else "Baseline"
        table.add_column(first, no_wrap=True)
        if is_model_table:
            table.add_column("Baseline", no_wrap=True)
        table.add_column("Jobs", justify="right")
        table.add_column("Megapixelsteps", justify="right")
        table.add_column("Sampling", justify="right")
        table.add_column("E2E", justify="right")
        table.add_column("Batch>1", justify="right")
        if not rows:
            empty = ["no finalized image jobs yet"] + ([""] if is_model_table else []) + ["", "", "", "", ""]
            table.add_row(*empty)
        else:
            for row in rows:
                cells = [shorten(row.model, 32) if is_model_table else short_baseline(row.baseline)]
                if is_model_table:
                    cells.append(short_baseline(row.baseline))
                cells.extend(
                    [
                        f"{row.jobs:,}",
                        f"{row.megapixelsteps:,.1f}",
                        human_duration(row.sampling_seconds),
                        human_duration(row.e2e_seconds),
                        f"{row.batch_gt_one_jobs:,}",
                    ],
                )
                table.add_row(*cells)
        return Panel(table, title=title, title_align="left", border_style="grey37", padding=(0, 1))
