"""The insights view: actionable recommendations plus a recent-activity summary."""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from horde_worker_regen.process_management.ipc.supervisor_channel import ModelPoolSnapshot, WorkerStateSnapshot
from horde_worker_regen.tui.formatters import human_bytes, human_duration
from horde_worker_regen.tui.recommendations import analyze

_SEAT_SOURCE_GLYPHS: dict[str, tuple[str, str]] = {
    "MANUAL": ("M", "yellow"),
    "RANKER": ("R", "cyan"),
    "RESCUE": ("S", "magenta"),
}
"""Compact glyph plus style per seat source (``M``anual pin, ``R``anker fill, re``S``cue)."""

_BENCH_ROWS_SHOWN = 3
"""How many benched models the pool panel lists before summarizing the remainder."""


class InsightsView(VerticalScroll):
    """Live recommendations plus a recent-activity rollup and a benchmark pointer."""

    def compose(self) -> ComposeResult:
        """Hold the recommendations, activity, model-pool, and benchmark-hint panels."""
        yield Static(id="insights-recommendations")
        yield Static(id="insights-activity")
        yield Static(id="insights-model-pool")
        yield Static(self._benchmark_hint(), id="insights-benchmark")

    def update_snapshot(self, snapshot: WorkerStateSnapshot) -> None:
        """Recompute recommendations, the activity summary, and the model-pool panel from a snapshot."""
        self.query_one("#insights-recommendations", Static).update(self._render_recommendations(snapshot))
        self.query_one("#insights-activity", Static).update(self._render_activity(snapshot))
        self._update_model_pool(snapshot.model_pool)

    def _update_model_pool(self, pool: ModelPoolSnapshot | None) -> None:
        """Render the model-pool panel, showing a one-line off-state with guidance when the pool is disabled."""
        panel = self.query_one("#insights-model-pool", Static)
        panel.display = True
        if pool is None or not pool.enabled:
            panel.update(self._render_model_pool_disabled())
            return
        panel.update(self._render_model_pool(pool))

    @staticmethod
    def _render_model_pool_disabled() -> Panel:
        """Render the pool panel's off-state: a plain 'off' line plus what turning it on would trade.

        Kept visible (rather than hidden) so an operator who has never seen the pool learns it exists and can
        weigh the throughput-versus-variety choice from the same place the live seats would otherwise appear.
        """
        body = Text.assemble(
            ("Model pool: ", "grey70"),
            ("off", "bold grey62"),
            (
                "  ·  the worker serves your whole model list evenly, for the widest variety of jobs. Enabling "
                "the pool (or Max throughput mode) commits it to a small, ready seat set: fewer model swaps and "
                "higher kudos/hr, at the cost of serving less model variety.",
                "grey62",
            ),
        )
        return Panel(body, title="Model pool", title_align="left", border_style="grey37")

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

    def _render_model_pool(self, pool: ModelPoolSnapshot) -> Panel:
        """Render the fixed model pool's seats, a lane/demand status line, and a bench summary."""
        status_line = self._render_pool_status_line(pool)
        seats_table = self._render_pool_seats(pool)
        bench_line = self._render_pool_bench(pool)
        return Panel(
            Group(status_line, seats_table, bench_line),
            title="Model pool",
            title_align="left",
            border_style="green",
        )

    def _render_pool_status_line(self, pool: ModelPoolSnapshot) -> Text:
        """Render the one-line lane/demand/budget status for the pool panel."""
        lane = pool.current_lane or "-"
        demand = human_duration(pool.demand_age_seconds) if pool.demand_age_seconds is not None else "-"
        parts = [
            Text.assemble(("Lane ", "grey70"), (lane, "bold")),
            Text.assemble(("seats ", "grey70"), (str(pool.last_fixed_seat_count), "bold")),
            Text.assemble(("demand ", "grey70"), (demand, "bold")),
        ]
        if pool.download_budget_gb > 0:
            used = human_bytes(pool.download_bytes_charged)
            total = f"{pool.download_budget_gb:.1f} GB"
            parts.append(Text.assemble(("budget ", "grey70"), (f"{used} / {total}", "bold")))
        return Text("  ·  ").join(parts)

    def _render_pool_seats(self, pool: ModelPoolSnapshot) -> Table:
        """Render the pool's seats as a compact table (model, source, state, dwell, fulfilled age, empty, rescue).

        The fulfilled-age column is how long since each seat last served work: a fresh age is a seat earning
        its place, while a growing age alongside empty pops is what pushes a seat toward demotion.
        """
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left")
        table.add_column(justify="center")
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_column(justify="right")
        table.add_column(justify="right")
        table.add_column(justify="right")
        table.add_row(
            Text("Model", style="bold grey70"),
            Text("Src", style="bold grey70"),
            Text("State", style="bold grey70"),
            Text("Dwell", style="bold grey70"),
            Text("Fulfilled", style="bold grey70"),
            Text("Empty", style="bold grey70"),
            Text("Rescue", style="bold grey70"),
        )
        for seat in pool.seats:
            model = seat.model if seat.model is not None else "-"
            state = f"dl:{seat.pending_model}" if seat.pending_model is not None else str(seat.state)
            dwell = human_duration(seat.dwell_seconds) if seat.dwell_seconds is not None else "-"
            fulfilled = (
                human_duration(seat.last_fulfilled_age_seconds) if seat.last_fulfilled_age_seconds is not None else "-"
            )
            rescue = (
                human_duration(seat.rescue_expires_in_seconds) if seat.rescue_expires_in_seconds is not None else "-"
            )
            table.add_row(
                Text(model),
                self._source_glyph(seat.source),
                Text(state, style="grey70"),
                Text(dwell),
                Text(fulfilled),
                Text(str(seat.empty_pops)),
                Text(rescue),
            )
        return table

    def _render_pool_bench(self, pool: ModelPoolSnapshot) -> Text:
        """Render the top few benched models (name, reason, remaining cooldown) as one line."""
        if not pool.bench:
            return Text.assemble(("Bench ", "grey70"), ("empty", "grey50"))
        shown = pool.bench[:_BENCH_ROWS_SHOWN]
        entries = [f"{row.model} ({row.reason}, {human_duration(row.cooldown_remaining_seconds)})" for row in shown]
        remainder = len(pool.bench) - len(shown)
        if remainder > 0:
            entries.append(f"+{remainder} more")
        return Text.assemble(("Bench ", "grey70"), ("  |  ".join(entries), "grey50"))

    @staticmethod
    def _source_glyph(source: str | None) -> Text:
        """Render a seat source as a compact, colour-coded glyph (or a dash when the seat is empty)."""
        if source is None:
            return Text("-", style="grey50")
        glyph, style = _SEAT_SOURCE_GLYPHS.get(source, (source[:1].upper(), "grey70"))
        return Text(glyph, style=style)

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
            ("Benchmark tab "),
            ("to run it live here, or from a shell:\n\n", "grey70"),
            ("    horde-benchmark\n", "bold green"),
        )
        return Panel(body, title="Deeper analysis", title_align="left", border_style="grey37")
