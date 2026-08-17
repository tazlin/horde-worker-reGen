"""Stats tab: worker-owned counters, rollups, trend detail, and JSONL export control."""

from __future__ import annotations

import time

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    ModelPoolSnapshot,
    PopGovernorsSnapshot,
    StatsRollupRow,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.lifecycle.horde_process import WorkerCapability
from horde_worker_regen.process_management.scheduling.workload_flow import capability_for_alchemy_form
from horde_worker_regen.tui.formatters import (
    format_percent,
    human_bytes,
    human_duration,
    human_mb,
    short_baseline,
    shorten,
)


class StatsView(Vertical):
    """Expanded statistics home for session counters, job rollups, and stats export."""

    class ExportToggled(Message):
        """Request that the worker enable or disable stats JSONL export."""

        def __init__(self, enabled: bool) -> None:
            """Store the requested export state."""
            super().__init__()
            self.enabled = enabled

    DEFAULT_CSS = """
    StatsView #stats-maintenance-clock {
        height: 1;
        padding: 0 1;
    }
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
        self._maintenance_started_at: float | None = None

    def compose(self) -> ComposeResult:
        """Lay out export control and the scrollable stats body."""
        yield Static(id="stats-maintenance-clock")
        yield Button("Enable JSONL export", id="stats-export-button")
        with VerticalScroll(id="stats-body"):
            yield Static(id="stats-headlines")
            yield Static(id="stats-governors")
            yield Static(id="stats-model-pool")
            yield Static(id="stats-export")
            yield Static(id="stats-by-model")
            yield Static(id="stats-by-baseline")
            yield Static(id="stats-by-form")

    def on_mount(self) -> None:
        """Hide the maintenance clock banner until maintenance is active."""
        self.query_one("#stats-maintenance-clock").display = False
        self.set_interval(1.0, self._refresh_maintenance_clock)

    def update_maintenance_start(self, started_at: float | None) -> None:
        """Set or clear the maintenance start timestamp; the 1s timer handles rendering."""
        self._maintenance_started_at = started_at

    def _refresh_maintenance_clock(self) -> None:
        """Rerender the maintenance-duration banner every second."""
        static = self.query_one("#stats-maintenance-clock", Static)
        if self._maintenance_started_at is None:
            static.display = False
            return
        elapsed = int(time.monotonic() - self._maintenance_started_at)
        h, remainder = divmod(elapsed, 3600)
        m, s = divmod(remainder, 60)
        text = Text()
        text.append(" MAINTENANCE ", style="bold white on dark_orange")
        text.append(f"  In maintenance mode for {h:02d}:{m:02d}:{s:02d}", style="bold yellow")
        static.update(text)
        static.display = True

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
        pool = snapshot.model_pool
        pool_static = self.query_one("#stats-model-pool", Static)
        if pool is not None and pool.enabled:
            pool_static.display = True
            pool_static.update(self._render_model_pool(pool))
            seated_models = frozenset(seat.model for seat in pool.seats if seat.model is not None)
        else:
            pool_static.display = False
            seated_models = frozenset()
        self.query_one("#stats-export", Static).update(self._render_export(snapshot))
        self.query_one("#stats-by-model", Static).update(
            self._render_rollups("By model totals", snapshot.stats_model_rollups, seated_models=seated_models)
        )
        self.query_one("#stats-by-baseline", Static).update(
            self._render_rollups("By baseline totals", snapshot.stats_baseline_rollups),
        )
        # The by-form table is alchemy-specific; show it only for an alchemist worker (or once forms exist)
        # so a dreamer worker's Stats tab is unchanged.
        form_static = self.query_one("#stats-by-form", Static)
        form_static.display = snapshot.config.alchemist or bool(snapshot.stats_form_rollups)
        if form_static.display:
            form_static.update(self._render_form_rollups(snapshot.stats_form_rollups))

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
        # The headline figure is a reduction across the driven cards, so a multi-GPU worker gets each
        # card's own duty beside it rather than a number that describes neither card.
        duty_per_card = snapshot.gpu_utilization_mean_percent_per_card
        gpu_duty = format_percent(snapshot.gpu_utilization_mean_percent)
        if len(duty_per_card) > 1:
            breakdown = ", ".join(f"card {index} {value:.0f}%" for index, value in sorted(duty_per_card.items()))
            gpu_duty = f"{gpu_duty} ({breakdown})"
        rows = [
            ("Jobs", f"{snapshot.num_jobs_submitted:,} submitted / {snapshot.num_jobs_faulted:,} faulted"),
            ("Kudos/hr", "-" if snapshot.kudos_per_hour is None else f"{snapshot.kudos_per_hour:,.0f}"),
            ("GPU duty", gpu_duty),
            ("Recoveries", f"{snapshot.num_process_recoveries:,}"),
            ("Slowdowns", f"{snapshot.num_job_slowdowns:,}"),
            ("No-work time", human_duration(snapshot.time_spent_no_jobs_available)),
            ("Pipeline", f"{snapshot.jobs_pending_inference} queued / {snapshot.jobs_in_progress} in progress"),
        ]
        if snapshot.config.alchemist:
            detail = f"{snapshot.alchemy_total_submitted:,} submitted / {snapshot.alchemy_total_faulted:,} faulted"
            graph_forms = sum(
                row.jobs for row in snapshot.stats_form_rollups if StatsView._form_kind_label(row.model) == "graph"
            )
            clip_forms = sum(
                row.jobs for row in snapshot.stats_form_rollups if StatsView._form_kind_label(row.model) == "clip"
            )
            if graph_forms or clip_forms:
                detail += f"  ({graph_forms:,} graph / {clip_forms:,} clip)"
            rows.append(("Alchemy", detail))
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
    def _render_model_pool(pool: ModelPoolSnapshot) -> Panel:
        """Render model-pool pop matches, resident hits, seat readiness, and bench size.

        The fixed and free lanes are shown apart. Matches measure successful pop responses; resident hits are
        the subset whose model was already loaded by a live inference process. Neither is completion throughput.
        """
        grid = Table.grid(padding=(0, 3))
        grid.add_column(style="bold cyan", no_wrap=True)
        grid.add_column(no_wrap=True)
        active_seats = sum(1 for seat in pool.seats if seat.model is not None)
        resident_seats = sum(1 for seat in pool.seats if seat.readiness == "RESIDENT")
        grid.add_row("Seats", f"{resident_seats} resident / {active_seats} seated / {len(pool.seats)} total")
        grid.add_row("Bench", str(len(pool.bench)))
        grid.add_row("Last routed lane", pool.current_lane or "-")
        grid.add_row(
            "Fixed lane",
            StatsView._lane_pop_summary(pool.fixed_fulfilled, pool.fixed_pops, pool.fixed_resident_hits),
        )
        grid.add_row(
            "Free lane",
            StatsView._lane_pop_summary(pool.free_fulfilled, pool.free_pops, pool.free_resident_hits),
        )
        return Panel(grid, title="Model pool", title_align="left", border_style="grey37", padding=(0, 1))

    @staticmethod
    def _lane_pop_summary(matches: int, pops: int, resident_hits: int) -> str:
        """Format a lane's matches, pop rate, and resident subset, or a dash before any pop."""
        if pops <= 0:
            return "-"
        return f"{matches:,} / {pops:,} matched ({matches / pops * 100:.0f}%); {resident_hits:,} resident"

    @staticmethod
    def _render_rollups(
        title: str,
        rows: list[StatsRollupRow],
        *,
        seated_models: frozenset[str] = frozenset(),
    ) -> Panel:
        """Render a by-model or by-baseline totals table.

        On the by-model table, a model that currently holds a pool seat is marked with a ``◆`` suffix (a
        legend rides the panel subtitle) so an operator reads a seated model's jobs/megapixelsteps/latency in
        place, without a duplicate per-seat table.
        """
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
        any_seated_shown = False
        if not rows:
            empty = ["no finalized image jobs yet"] + ([""] if is_model_table else []) + ["", "", "", "", ""]
            table.add_row(*empty)
        else:
            for row in rows:
                seated = is_model_table and row.model is not None and row.model in seated_models
                any_seated_shown = any_seated_shown or seated
                first_cell: str | Text
                if not is_model_table:
                    first_cell = short_baseline(row.baseline)
                elif seated:
                    first_cell = Text.assemble((shorten(row.model, 32), ""), (" ◆", "cyan"))
                else:
                    first_cell = shorten(row.model, 32)
                cells: list[str | Text] = [first_cell]
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
        subtitle = "◆ holds a model-pool seat" if any_seated_shown else None
        return Panel(
            table,
            title=title,
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style="grey37",
            padding=(0, 1),
        )

    @staticmethod
    def _render_form_rollups(rows: list[StatsRollupRow]) -> Panel:
        """Render finalized alchemy forms by form: count, average and total pop->submit time.

        The alchemist analogue of the by-model table. The diffusion-specific columns (megapixelsteps,
        sampling, batch) do not apply to alchemy forms and are omitted; the per-form average is the headline
        an operator wants for sense of how long each form takes.
        """
        table = Table(expand=True, border_style="grey37", header_style="bold")
        table.add_column("Form", no_wrap=True)
        table.add_column("Kind", no_wrap=True)
        table.add_column("Forms", justify="right")
        table.add_column("Faulted", justify="right")
        table.add_column("Avg E2E", justify="right")
        table.add_column("Total E2E", justify="right")
        table.add_column("Peak VRAM", justify="right")
        if not rows:
            table.add_row("no finalized alchemy forms yet", "", "", "", "", "", "")
        else:
            for row in rows:
                average = row.e2e_seconds / row.jobs if row.jobs else 0.0
                faulted = Text(f"{row.faulted_jobs:,}", style="red" if row.faulted_jobs else "grey50")
                table.add_row(
                    shorten(row.model, 32),
                    StatsView._form_kind_label(row.model),
                    f"{row.jobs:,}",
                    faulted,
                    human_duration(average),
                    human_duration(row.e2e_seconds),
                    human_mb(row.vram_high_water_mb) if row.vram_high_water_mb else "-",
                )
        return Panel(table, title="By alchemy form totals", title_align="left", border_style="grey37", padding=(0, 1))

    @staticmethod
    def _form_kind_label(form: str | None) -> str:
        """Classify a form as graph (runs on an inference process) or CLIP (runs on the safety process)."""
        if form is None:
            return "-"
        return "graph" if capability_for_alchemy_form(form) is WorkerCapability.ALCHEMY_GRAPH else "clip"
