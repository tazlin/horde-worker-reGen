"""The insights view: actionable recommendations plus a recent-activity summary."""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from horde_worker_regen.process_management.supervisor_channel import WorkerStateSnapshot
from horde_worker_regen.tui.formatters import human_duration
from horde_worker_regen.tui.recommendations import analyze


class InsightsView(VerticalScroll):
    """Live recommendations plus a recent-activity rollup and a benchmark pointer."""

    def compose(self) -> ComposeResult:
        """Hold the recommendations, activity, and benchmark-hint panels."""
        yield Static(id="insights-recommendations")
        yield Static(id="insights-activity")
        yield Static(self._benchmark_hint(), id="insights-benchmark")

    def update_snapshot(self, snapshot: WorkerStateSnapshot) -> None:
        """Recompute recommendations and the activity summary from a snapshot."""
        self.query_one("#insights-recommendations", Static).update(self._render_recommendations(snapshot))
        self.query_one("#insights-activity", Static).update(self._render_activity(snapshot))

    def _render_recommendations(self, snapshot: WorkerStateSnapshot) -> Panel:
        """Render the recommendation list as a bordered panel."""
        rows = []
        for item in analyze(snapshot):
            badge = Text(f" {item.severity.label} ", style=item.severity.colour)
            rows.append(
                Group(
                    Text.assemble(badge, ("  ", ""), (item.title, "bold")),
                    Text(f"    {item.detail}", style="grey70"),
                ),
            )
        return Panel(Group(*rows), title="Recommendations", title_align="left", border_style="cyan")

    def _render_activity(self, snapshot: WorkerStateSnapshot) -> Panel:
        """Render a rollup of recent finished jobs."""
        jobs = snapshot.recent_jobs
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan")
        table.add_column()

        if not jobs:
            table.add_row("Recent jobs", "none yet")
        else:
            faulted = sum(1 for job in jobs if job.faulted)
            e2e_values = [job.e2e_seconds for job in jobs if job.e2e_seconds is not None]
            queue_values = [job.queue_wait_seconds for job in jobs if job.queue_wait_seconds is not None]
            table.add_row("Recent jobs", str(len(jobs)))
            table.add_row("Faulted", str(faulted))
            if e2e_values:
                table.add_row("Avg end-to-end", human_duration(sum(e2e_values) / len(e2e_values)))
            if queue_values:
                table.add_row("Avg queue wait", human_duration(sum(queue_values) / len(queue_values)))

        table.add_row("GPU busy fraction", self._fraction(snapshot.gpu_utilization_busy_fraction))
        return Panel(table, title="Recent activity", title_align="left", border_style="grey37")

    @staticmethod
    def _fraction(value: float | None) -> str:
        """Render a 0–1 fraction as a percentage."""
        return "-" if value is None else f"{value * 100:.0f}%"

    @staticmethod
    def _benchmark_hint() -> Panel:
        """A static pointer to the full benchmark sweep (the in-TUI Benchmark tab, or the CLI)."""
        body = Text.assemble(
            (
                "For an authoritative capability sweep (safe ramp levels, suggested bridgeData, and "
                "robustness findings), open the ",
                "grey70",
            ),
            ("Benchmark", "bold green"),
            (" tab (", "grey70"),
            ("F8", "bold green"),
            (") to run it live here, or from a shell:\n\n", "grey70"),
            ("    horde-benchmark\n", "bold green"),
        )
        return Panel(body, title="Deeper analysis", title_align="left", border_style="grey37")
