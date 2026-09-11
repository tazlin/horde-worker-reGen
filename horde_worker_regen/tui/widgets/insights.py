"""The insights view: live findings plus a recent-activity summary.

The findings come from :func:`horde_worker_regen.tui.recommendations.analyze` over the latest worker
snapshot and are shown with the same card the Diagnostics tab uses for a log finding, so the two tabs
read alike. Snapshots arrive every few seconds; the cards are rebuilt only when the findings change, so
an open "Details" section stays open while the numbers behind it stand.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

from horde_worker_regen.analysis.finding_kinds import Finding
from horde_worker_regen.process_management.ipc.supervisor_channel import ModelPoolSnapshot, WorkerStateSnapshot
from horde_worker_regen.tui.formatters import human_bytes, human_duration
from horde_worker_regen.tui.recommendations import analyze
from horde_worker_regen.tui.widgets.finding_card import FindingCard

_SEAT_SOURCE_GLYPHS: dict[str, tuple[str, str]] = {
    "MANUAL": ("M", "yellow"),
    "RANKER": ("R", "cyan"),
    "RESCUE": ("S", "magenta"),
}
"""Compact glyph plus style per seat source (``M``anual pin, ``R``anker fill, re``S``cue)."""

_BENCH_ROWS_SHOWN = 3
"""How many benched models the pool panel lists before summarizing the remainder."""


class InsightsView(VerticalScroll):
    """Live findings plus a recent-activity rollup and a benchmark pointer."""

    DEFAULT_CSS = """
    InsightsView #insights-findings {
        height: auto;
        border: round $accent;
        border-title-color: $accent;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    InsightsView #insights-findings-message {
        height: auto;
    }
    InsightsView #insights-findings-cards {
        height: auto;
    }
    """

    def __init__(self) -> None:
        """Start with no findings shown and no card generation issued."""
        super().__init__()
        self._shown_findings: list[tuple[str, str, str, str]] = []
        self._cards_generation = 0

    def compose(self) -> ComposeResult:
        """Hold the findings, activity, model-pool, and benchmark-hint panels."""
        with Vertical(id="insights-findings") as findings:
            findings.border_title = "Findings"
            yield Static(
                Text("Waiting for the first worker snapshot.", style="grey50"), id="insights-findings-message"
            )
            yield Vertical(id="insights-findings-cards")
        yield Static(id="insights-activity")
        yield Static(id="insights-model-pool")
        yield Static(self._benchmark_hint(), id="insights-benchmark")

    def update_snapshot(self, snapshot: WorkerStateSnapshot) -> None:
        """Recompute the findings, the activity summary, and the model-pool panel from a snapshot."""
        self._update_findings(analyze(snapshot))
        self.query_one("#insights-activity", Static).update(self._render_activity(snapshot))
        self._update_model_pool(snapshot.model_pool)

    def _update_findings(self, findings: list[Finding]) -> None:
        """Show the findings as cards, rebuilding them only when what they say has changed."""
        shown = [(finding.id, finding.severity.value, finding.headline, finding.action) for finding in findings]
        if shown == self._shown_findings:
            return
        self._shown_findings = shown
        message = self.query_one("#insights-findings-message", Static)
        if not findings:
            message.update(Text("No issues found. The worker looks healthy.", style="green"))
        message.display = not findings
        # Removal is asynchronous in Textual, so the mount runs as the next callback on this widget's own
        # queue and awaits the removal first; a render superseded by a newer snapshot mounts nothing.
        self._cards_generation += 1
        self.call_next(self._mount_cards, findings, self._cards_generation)

    async def _mount_cards(self, findings: list[Finding], generation: int) -> None:
        """Replace the cards container's children with one card per finding, in order."""
        if generation != self._cards_generation:
            return
        cards = self.query_one("#insights-findings-cards", Vertical)
        await cards.remove_children()
        if generation != self._cards_generation:
            return
        await cards.mount_all(FindingCard(finding) for finding in findings)

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
                "the pool (or the demand-following preset) biases pops toward a small seat set. This can reduce "
                "model swaps when those seats are resident, at the cost of serving less model variety.",
                "grey62",
            ),
        )
        return Panel(body, title="Model pool", title_align="left", border_style="grey37")

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
            Text.assemble(
                ("resident ", "grey70"),
                (str(sum(seat.readiness == "RESIDENT" for seat in pool.seats)), "bold"),
                (f" / {sum(seat.model is not None for seat in pool.seats)} seated", "grey70"),
            ),
            Text.assemble(("last fixed offer ", "grey70"), (str(pool.last_fixed_seat_count), "bold")),
            Text.assemble(("demand ", "grey70"), (demand, "bold")),
        ]
        if pool.download_budget_gb > 0:
            used = human_bytes(pool.download_bytes_charged)
            total = f"{pool.download_budget_gb:.1f} GB"
            parts.append(Text.assemble(("download admission ", "grey70"), (f"{used} / {total}", "bold")))
        return Text("  ·  ").join(parts)

    def _render_pool_seats(self, pool: ModelPoolSnapshot) -> Table:
        """Render the pool's seats as a compact table with readiness and recent-match evidence.

        ``Readiness`` distinguishes a logical seat from a model currently loaded by a live inference process.
        ``Matched`` is the age of the last successful pop for the seat; its suffix says whether that match was
        resident or cold. It does not claim that the accepted job has completed.
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
            Text("Readiness", style="bold grey70"),
            Text("Dwell", style="bold grey70"),
            Text("Matched", style="bold grey70"),
            Text("Empty", style="bold grey70"),
            Text("Rescue", style="bold grey70"),
        )
        for seat in pool.seats:
            model = seat.model if seat.model is not None else "-"
            readiness = str(seat.readiness).lower()
            if seat.pending_model is not None:
                readiness = f"{readiness} · dl:{seat.pending_model}"
            dwell = human_duration(seat.dwell_seconds) if seat.dwell_seconds is not None else "-"
            matched = (
                human_duration(seat.last_fulfilled_age_seconds) if seat.last_fulfilled_age_seconds is not None else "-"
            )
            if seat.last_match_was_resident is not None and matched != "-":
                matched += " resident" if seat.last_match_was_resident else " cold"
            rescue = (
                human_duration(seat.rescue_expires_in_seconds) if seat.rescue_expires_in_seconds is not None else "-"
            )
            table.add_row(
                Text(model),
                self._source_glyph(seat.source),
                Text(readiness, style="green" if seat.readiness == "RESIDENT" else "grey70"),
                Text(dwell),
                Text(matched),
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
