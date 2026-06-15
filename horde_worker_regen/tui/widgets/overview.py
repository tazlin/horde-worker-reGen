"""The overview screen: a live status-monitor hero, a health checklist, then metrics and processes."""

from __future__ import annotations

import time

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Static

from horde_worker_regen.process_management.supervisor_channel import WorkerStateSnapshot
from horde_worker_regen.tui.formatters import (
    format_its,
    format_percent,
    human_duration,
    human_mb,
    shorten,
)
from horde_worker_regen.tui.health import HealthReport, HealthStatus, WorkerPhase
from horde_worker_regen.tui.widgets.common import StatCard

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SERVING_PULSE = ("dark_green", "green", "green3", "bright_green", "green3", "green")

_STATIC_GLYPHS: dict[WorkerPhase, str] = {
    WorkerPhase.STOPPED: "■",
    WorkerPhase.CRASHED: "✗",
    WorkerPhase.DISCONNECTED: "✗",
    WorkerPhase.DEGRADED: "▲",
    WorkerPhase.PAUSED: "⏸",
    WorkerPhase.IDLE: "○",
    WorkerPhase.READY: "●",
}


class OverviewView(VerticalScroll):
    """A dashboard led by a living status hero and a health checklist."""

    def compose(self) -> ComposeResult:
        """Lay out the hero, health checklist, stat cards, and detail tables."""
        yield Static(id="overview-hero")
        yield Static(id="overview-health")
        with Horizontal(id="overview-cards"):
            yield StatCard("Jobs submitted", card_id="ov-submitted")
            yield StatCard("Jobs faulted", card_id="ov-faulted")
            yield StatCard("Queue (eMPS)", card_id="ov-queue")
            yield StatCard("GPU duty", card_id="ov-gpu")
            yield StatCard("Kudos / hr", card_id="ov-kudos")
            yield StatCard("Processes", card_id="ov-processes")
        yield Static(id="overview-worker")
        yield Static(id="overview-processes")

    def update_view(
        self,
        report: HealthReport,
        snapshot: WorkerStateSnapshot | None,
        *,
        frame: int,
    ) -> None:
        """Refresh the hero/health from the report and the metrics from the snapshot (if any)."""
        self.query_one("#overview-hero", Static).update(self._render_hero(report, snapshot, frame))
        self.query_one("#overview-health", Static).update(self._render_health(report))
        if snapshot is not None:
            self._update_cards(snapshot)
            self.query_one("#overview-worker", Static).update(self._render_worker_table(snapshot))
            self.query_one("#overview-processes", Static).update(self._render_process_table(snapshot))

    def _hero_glyph(self, report: HealthReport, frame: int) -> Text:
        """A status glyph that pulses/spins for in-progress or attention states."""
        if report.animated:
            if report.phase is WorkerPhase.SERVING:
                colour = _SERVING_PULSE[frame % len(_SERVING_PULSE)]
                return Text("●", style=f"bold {colour}")
            if report.phase is WorkerPhase.UNRESPONSIVE:
                return Text("▲", style="bold red" if frame % 2 == 0 else "red dim")
            return Text(_SPINNER[frame % len(_SPINNER)], style=f"bold {report.severity.colour}")
        glyph = _STATIC_GLYPHS.get(report.phase, "●")
        return Text(glyph, style=report.severity.colour)

    def _render_hero(self, report: HealthReport, snapshot: WorkerStateSnapshot | None, frame: int) -> Panel:
        """Render the headline status panel."""
        title = Text.assemble(
            self._hero_glyph(report, frame),
            ("  ", ""),
            (report.phase.value.upper(), f"bold {report.severity.colour}"),
            ("   ", ""),
            (report.headline, "bold"),
        )
        body: list[Text] = [Text(report.detail, style="grey70")]

        if snapshot is not None:
            body.append(self._activity_line(snapshot))
            for message in snapshot.api_messages[:3]:
                body.append(Text.assemble(("✉ ", "cyan"), (message, "italic cyan")))

        border = "red" if report.severity is HealthStatus.ERROR else report.severity.colour
        return Panel(Group(*body), title=title, title_align="left", border_style=border, padding=(0, 1))

    @staticmethod
    def _activity_line(snapshot: WorkerStateSnapshot) -> Text:
        """A heartbeat line conveying recent activity and freshness."""
        age = time.time() - snapshot.timestamp if snapshot.timestamp else None
        since_pop = (
            human_duration(snapshot.seconds_since_last_pop) + " ago"
            if snapshot.seconds_since_last_pop is not None
            else "never"
        )
        return Text.assemble(
            ("updated ", "grey50"),
            (f"{human_duration(age)} ago", "grey70"),
            ("  ·  last pop ", "grey50"),
            (since_pop, "grey70"),
            ("  ·  in progress ", "grey50"),
            (str(snapshot.jobs_in_progress), "grey70"),
            ("  ·  queued ", "grey50"),
            (str(snapshot.jobs_pending_inference), "grey70"),
        )

    def _render_health(self, report: HealthReport) -> Panel:
        """Render the health checklist."""
        table = Table.grid(padding=(0, 2))
        table.add_column(width=2)
        table.add_column(style="bold", no_wrap=True)
        table.add_column()
        if not report.checks:
            table.add_row("", Text("-", style="grey50"), "no checks while the worker is not running")
        for check in report.checks:
            table.add_row(
                Text(check.status.glyph, style=check.status.colour),
                Text(check.name, style=check.status.colour),
                Text(check.detail, style="grey70"),
            )
        return Panel(table, title="Health", title_align="left", border_style="grey37", padding=(0, 1))

    def _update_cards(self, snapshot: WorkerStateSnapshot) -> None:
        """Refresh the headline stat cards from a snapshot."""
        self.query_one("#ov-submitted", StatCard).update_value(str(snapshot.num_jobs_submitted))
        self.query_one("#ov-faulted", StatCard).update_value(str(snapshot.num_jobs_faulted))
        self.query_one("#ov-queue", StatCard).update_value(
            f"{snapshot.pending_megapixelsteps} / {snapshot.jobs_pending_inference} jobs",
        )
        self.query_one("#ov-gpu", StatCard).update_value(format_percent(snapshot.gpu_utilization_mean_percent))
        kudos = "-" if snapshot.kudos_per_hour is None else f"{snapshot.kudos_per_hour:,.0f}"
        self.query_one("#ov-kudos", StatCard).update_value(kudos)
        alive = sum(1 for process in snapshot.processes if process.is_alive)
        self.query_one("#ov-processes", StatCard).update_value(f"{alive} / {len(snapshot.processes)}")

    def _render_worker_table(self, snapshot: WorkerStateSnapshot) -> Table:
        """Build a key/value table of worker identity and configuration."""
        config = snapshot.config
        uptime = human_duration(time.time() - snapshot.session_start_time) if snapshot.session_start_time else "-"

        memory_mode = "normal"
        if config.very_high_memory_mode:
            memory_mode = "very high"
        elif config.high_memory_mode:
            memory_mode = "high"

        performance_mode = "normal"
        if config.high_performance_mode:
            performance_mode = "high"
        elif config.moderate_performance_mode:
            performance_mode = "moderate"
        elif config.extra_slow_worker:
            performance_mode = "extra slow"

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column()

        table.add_row("Dreamer", config.dreamer_name, "Version", f"v{config.worker_version}")
        table.add_row("Horde user", config.horde_username or "-", "Uptime", uptime)
        table.add_row("Models", str(config.num_models), "Custom models", "yes" if config.custom_models else "no")
        table.add_row("Threads", str(config.max_threads), "Queue size", str(config.queue_size))
        table.add_row("Max power", str(config.max_power), "Max batch", str(config.max_batch))
        table.add_row("Memory mode", memory_mode, "Performance", performance_mode)
        table.add_row(
            "Safety on GPU",
            "yes" if config.safety_on_gpu else "no",
            "Allows",
            self._allow_summary(snapshot),
        )
        return table

    @staticmethod
    def _allow_summary(snapshot: WorkerStateSnapshot) -> str:
        """Summarize which optional job features the worker accepts."""
        config = snapshot.config
        flags = []
        if config.allow_img2img:
            flags.append("img2img")
        if config.allow_lora:
            flags.append("lora")
        if config.allow_controlnet:
            flags.append("controlnet")
        if config.allow_post_processing:
            flags.append("post")
        return ", ".join(flags) if flags else "none"

    def _render_process_table(self, snapshot: WorkerStateSnapshot) -> Table:
        """Build a compact per-process summary table."""
        table = Table(
            title="Processes",
            title_style="bold",
            expand=True,
            border_style="grey37",
            header_style="bold",
        )
        table.add_column("ID", justify="right")
        table.add_column("Type")
        table.add_column("State")
        table.add_column("Model")
        table.add_column("Step", justify="right")
        table.add_column("it/s", justify="right")
        table.add_column("VRAM", justify="right")

        if not snapshot.processes:
            table.add_row("-", "-", "waiting for first snapshot", "-", "-", "-", "-")
            return table

        for process in snapshot.processes:
            step = (
                f"{process.last_current_step}/{process.last_total_steps}"
                if process.last_current_step is not None and process.last_total_steps
                else "-"
            )
            vram = (
                f"{human_mb(process.vram_usage_mb)} / {human_mb(process.total_vram_mb)}"
                if process.total_vram_mb
                else human_mb(process.vram_usage_mb)
            )
            table.add_row(
                str(process.process_id),
                process.process_type.title(),
                Text(process.last_process_state, style=self._state_style(process.last_process_state)),
                shorten(process.loaded_horde_model_name, 26),
                step,
                format_its(process.last_iterations_per_second),
                vram,
            )
        return table

    @staticmethod
    def _state_style(state: str) -> str:
        """Map a process state name to a display colour."""
        if state in ("INFERENCE_STARTING", "INFERENCE_POST_PROCESSING", "ALCHEMY_STARTING"):
            return "green"
        if state in ("INFERENCE_FAILED", "ALCHEMY_FAILED", "SAFETY_FAILED", "PROCESS_ENDED"):
            return "red"
        if state == "WAITING_FOR_JOB":
            return "grey62"
        return "yellow"
