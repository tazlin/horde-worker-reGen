"""The overview screen: a live status-monitor hero, a health checklist, then metrics and processes."""

from __future__ import annotations

import time
from collections import deque

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from horde_worker_regen.app_state import OverviewViewMode
from horde_worker_regen.process_management.supervisor_channel import (
    JobQueueEntry,
    ProcessSnapshot,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.formatters import (
    format_its,
    format_percent,
    human_duration,
    human_mb,
    label_state,
    mini_bar,
    shorten,
    sparkline,
)
from horde_worker_regen.tui.health import HealthReport, HealthStatus, WorkerPhase, summarize_skips

_TREND_HISTORY = 180
"""How many trend samples (GPU-duty / kudos-per-hour / job counts) the Trends region retains."""

_TREND_SAMPLE_INTERVAL = 1.0
"""Minimum wall-clock seconds between recorded trend samples, so the window spans minutes not frames."""

_TREND_SPARK_WIDTH = 40
"""Maximum number of recent samples drawn in a Trends sparkline, keeping the line terminal-friendly."""

_SAMPLING_STATES = frozenset({"INFERENCE_STARTING", "INFERENCE_POST_PROCESSING", "ALCHEMY_STARTING"})
"""States with a live sampling step/it-s; outside these the snapshot's step numbers are last-job residue."""

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

    def __init__(self) -> None:
        """Set up the view, including the client-side trend history for the Trends sparklines."""
        super().__init__()
        self._gpu_duty_history: deque[float] = deque(maxlen=_TREND_HISTORY)
        self._kudos_history: deque[float] = deque(maxlen=_TREND_HISTORY)
        self._jobs_history: deque[tuple[float, int]] = deque(maxlen=_TREND_HISTORY)
        self._last_trend_sample = 0.0

    def compose(self) -> ComposeResult:
        """Lay out the compact bar plus the hero, health, trends, pipeline, and detail tables.

        Only one set is visible at a time: ``update_view`` toggles each node's ``display`` from the
        active :class:`OverviewViewMode` (thin shows only the compact bar; the worker/alchemy/queue/
        recent statics appear only in details mode).
        """
        yield Static(id="overview-thin")
        yield Static(id="overview-hero")
        yield Static(id="overview-health")
        yield Static(id="overview-trends")
        yield Static(id="overview-pipeline")
        yield Static(id="overview-processes")
        yield Static(id="overview-worker")
        yield Static(id="overview-alchemy")
        yield Static(id="overview-queue")
        yield Static(id="overview-recent")

    _NORMAL_NODE_IDS = (
        "#overview-hero",
        "#overview-health",
        "#overview-trends",
        "#overview-pipeline",
        "#overview-processes",
    )
    """Statics shown in normal (and details) mode, hidden in thin mode."""

    _DETAIL_NODE_IDS = (
        "#overview-worker",
        "#overview-alchemy",
        "#overview-queue",
        "#overview-recent",
    )
    """Statics shown only in details mode (the demoted panels)."""

    def update_view(
        self,
        report: HealthReport,
        snapshot: WorkerStateSnapshot | None,
        *,
        frame: int,
        mode: OverviewViewMode = OverviewViewMode.NORMAL,
    ) -> None:
        """Refresh the visible regions for the active view ``mode`` from the report and snapshot."""
        thin = mode is OverviewViewMode.THIN
        detailed = mode is OverviewViewMode.DETAILS

        self.query_one("#overview-thin", Static).display = thin
        for node_id in self._NORMAL_NODE_IDS:
            self.query_one(node_id, Static).display = not thin
        for node_id in self._DETAIL_NODE_IDS:
            self.query_one(node_id, Static).display = detailed

        if snapshot is not None:
            self._maybe_record_trends(snapshot)

        if thin:
            self.query_one("#overview-thin", Static).update(self._render_compact_bar(report, snapshot, frame))
            return

        self.query_one("#overview-hero", Static).update(self._render_hero(report, snapshot, frame))
        self.query_one("#overview-health", Static).update(self._render_health(report))
        if snapshot is not None:
            self.query_one("#overview-trends", Static).update(self._render_trends(snapshot))
            self.query_one("#overview-pipeline", Static).update(self._render_pipeline_strip(snapshot))
            self.query_one("#overview-processes", Static).update(
                self._render_process_table(snapshot, detailed=detailed),
            )
            if detailed:
                self.query_one("#overview-worker", Static).update(self._render_worker_table(snapshot))
                self.query_one("#overview-alchemy", Static).update(self._render_alchemy_panel(snapshot))
                self.query_one("#overview-queue", Static).update(self._render_queue_table(snapshot))
                self.query_one("#overview-recent", Static).update(self._render_recent_jobs(snapshot))

    def _maybe_record_trends(self, snapshot: WorkerStateSnapshot) -> None:
        """Record a trend sample at most once per :data:`_TREND_SAMPLE_INTERVAL` of wall-clock time."""
        now = time.time()
        if now - self._last_trend_sample < _TREND_SAMPLE_INTERVAL:
            return
        self._last_trend_sample = now
        self._record_trends(snapshot)

    def _record_trends(self, snapshot: WorkerStateSnapshot) -> None:
        """Append one sample of GPU-duty, kudos/hr, and the cumulative job counter to the buffers."""
        if snapshot.gpu_utilization_mean_percent is not None:
            self._gpu_duty_history.append(snapshot.gpu_utilization_mean_percent)
        if snapshot.kudos_per_hour is not None:
            self._kudos_history.append(snapshot.kudos_per_hour)
        self._jobs_history.append((time.time(), snapshot.num_jobs_submitted))

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
            body.append(self._headline_metrics_line(snapshot))
            body.append(self._activity_line(snapshot))
            why_no_work = summarize_skips(snapshot.last_pop_skipped_reasons)
            if why_no_work:
                body.append(Text.assemble(("∅ why no work: ", "yellow"), (why_no_work, "italic yellow")))
            if snapshot.lora_pops_blocked_by_downloads:
                body.append(
                    Text(
                        "LoRA pops paused while background downloads are active.",
                        style="yellow",
                    )
                )
            for message in snapshot.api_messages[:3]:
                body.append(Text.assemble(("✉ ", "cyan"), (message, "italic cyan")))

        border = "red" if report.severity is HealthStatus.ERROR else report.severity.colour
        return Panel(Group(*body), title=title, title_align="left", border_style=border, padding=(0, 1))

    @staticmethod
    def _headline_metrics_line(snapshot: WorkerStateSnapshot) -> Text:
        """The session totals the dropped stat cards used to carry: submitted, kudos/hr, faulted."""
        kudos = "-" if snapshot.kudos_per_hour is None else f"{snapshot.kudos_per_hour:,.0f}"
        faulted_colour = "red" if snapshot.num_jobs_faulted else "grey70"
        return Text.assemble(
            (f"{snapshot.num_jobs_submitted:,}", "bold"),
            (" jobs submitted", "grey50"),
            ("   ·   ", "grey37"),
            (kudos, "bold cyan"),
            (" kudos/hr", "grey50"),
            ("   ·   ", "grey37"),
            (f"{snapshot.num_jobs_faulted:,}", faulted_colour),
            (" faulted", "grey50"),
        )

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
            if snapshot.lora_pops_blocked_by_downloads or config.effective_allow_lora is False:
                flags.append("lora paused")
            else:
                flags.append("lora")
        if config.allow_controlnet:
            flags.append("controlnet")
        if config.allow_post_processing:
            flags.append("post")
        return ", ".join(flags) if flags else "none"

    @staticmethod
    def _render_alchemy_panel(snapshot: WorkerStateSnapshot) -> Panel:
        """Render the alchemy configuration and runtime state panel."""
        config = snapshot.config

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column()

        if not config.alchemist:
            table.add_row("Status", Text("disabled", style="grey50"), "", "")
            return Panel(table, title="Alchemy", title_align="left", border_style="grey37", padding=(0, 1))

        mode_parts = []
        if config.alchemy_concurrent:
            mode_parts.append(Text("concurrent", style="green"))
            headroom_detail = f" (max {config.alchemy_max_concurrency}, {config.alchemy_vram_headroom_mb} MB headroom)"
            mode_parts.append(Text(headroom_detail, style="grey62"))
        else:
            mode_parts.append(Text("backfill only", style="yellow"))
        if config.alchemy_caption_enabled:
            mode_parts.append(Text(" · caption on", style="grey62"))

        mode_text = Text.assemble(*mode_parts)
        forms_text = ", ".join(config.alchemy_forms) if config.alchemy_forms else "-"

        total_active = (
            snapshot.alchemy_forms_pending + snapshot.alchemy_forms_in_flight + snapshot.alchemy_forms_awaiting_submit
        )
        active_colour = "green" if total_active > 0 else "grey62"
        faulted_colour = "red" if snapshot.alchemy_total_faulted else "grey62"
        runtime_text = Text.assemble(
            (str(snapshot.alchemy_forms_pending), active_colour),
            (" pending  ", "grey50"),
            (str(snapshot.alchemy_forms_in_flight), active_colour),
            (" in flight  ", "grey50"),
            (str(snapshot.alchemy_forms_awaiting_submit), "grey62"),
            (" submitting  ", "grey50"),
            (str(snapshot.alchemy_total_submitted), "grey70"),
            (" done  ", "grey50"),
            (str(snapshot.alchemy_total_faulted), faulted_colour),
            (" faulted", "grey50"),
        )

        table.add_row("Mode", mode_text, "Forms", forms_text)
        table.add_row("Runtime", runtime_text, "", "")

        border = "green" if total_active > 0 else ("yellow" if not config.alchemy_concurrent else "grey37")
        return Panel(table, title="Alchemy", title_align="left", border_style=border, padding=(0, 1))

    _PIPELINE_BAR_WIDTH = 8
    """Maximum block-bar width (chars) for one job-pipeline stage; bars scale to the busiest stage."""

    @staticmethod
    def _stage_segment(label: str, count: int, peak: int) -> Text:
        """Render one labelled pipeline stage: name, a count-proportional bar, and the count."""
        colour = "green" if count > 0 else "grey50"
        if peak > 0 and count > 0:
            width = max(1, round(count / peak * OverviewView._PIPELINE_BAR_WIDTH))
            bar = mini_bar(count / peak, width)
        else:
            bar = "·"
        return Text.assemble((f"{label} ", "bold"), (bar + " ", colour), (str(count), f"bold {colour}"))

    def _render_pipeline_strip(self, snapshot: WorkerStateSnapshot) -> Panel:
        """Render the job lifecycle as a labelled flow: what is queued, in-flight, and finishing.

        The first stages are live in-flight queues (they scale together against the busiest stage);
        the trailing "Submitted" is the session running total, shown plainly so a cumulative figure is
        not mistaken for a backlog.
        """
        queue = snapshot.jobs_pending_inference
        inference = snapshot.jobs_in_progress
        safety = snapshot.jobs_pending_safety_check + snapshot.jobs_being_safety_checked
        submit = snapshot.jobs_pending_submit
        peak = max(queue, inference, safety, submit, 1)

        arrow = Text(" ▶ ", style="grey50")
        flow = Text.assemble(
            self._stage_segment("Queue", queue, peak),
            arrow,
            self._stage_segment("Inference", inference, peak),
            arrow,
            self._stage_segment("Safety", safety, peak),
            arrow,
            self._stage_segment("Submit", submit, peak),
            ("    ", ""),
            (f"✓ {snapshot.num_jobs_submitted:,} submitted", "grey62"),
        )
        rows: list[Text] = [flow]

        if snapshot.config.alchemist:
            alch_peak = max(
                snapshot.alchemy_forms_pending,
                snapshot.alchemy_forms_in_flight,
                snapshot.alchemy_forms_awaiting_submit,
                1,
            )
            rows.append(
                Text.assemble(
                    self._stage_segment("Alchemy pending", snapshot.alchemy_forms_pending, alch_peak),
                    arrow,
                    self._stage_segment("active", snapshot.alchemy_forms_in_flight, alch_peak),
                    arrow,
                    self._stage_segment("submit", snapshot.alchemy_forms_awaiting_submit, alch_peak),
                    ("    ", ""),
                    (f"✓ {snapshot.alchemy_total_submitted:,} submitted", "grey62"),
                ),
            )

        border = "green" if (queue or inference or safety or submit) else "grey37"
        return Panel(Group(*rows), title="Job pipeline", title_align="left", border_style=border, padding=(0, 1))

    @staticmethod
    def _trend_arrow(series: list[float]) -> Text:
        """A direction marker comparing the latest sample to the window start, with a percent delta.

        A short or flat series reads as steady rather than asserting a trend the data cannot support.
        """
        if len(series) < 2 or series[0] == 0:
            return Text("→", style="grey50")
        change = (series[-1] - series[0]) / abs(series[0])
        if change > 0.02:
            return Text(f"▲ {abs(change) * 100:.0f}%", style="green")
        if change < -0.02:
            return Text(f"▼ {abs(change) * 100:.0f}%", style="red")
        return Text("→", style="grey50")

    def _jobs_per_hour(self) -> tuple[float | None, list[float]]:
        """Derive a jobs/hr rate and a per-sample jobs-completed series from the job-count history.

        Returns ``(rate, deltas)``: ``rate`` is None until two samples span a positive interval; the
        ``deltas`` series is the jobs finished between consecutive samples (the sparkline's signal).
        """
        samples = list(self._jobs_history)
        if len(samples) < 2:
            return None, []
        deltas = [float(max(0, b[1] - a[1])) for a, b in zip(samples, samples[1:], strict=False)]
        elapsed = samples[-1][0] - samples[0][0]
        completed = samples[-1][1] - samples[0][1]
        rate = (completed / elapsed * 3600.0) if elapsed > 0 else None
        return rate, deltas

    def _render_trends(self, snapshot: WorkerStateSnapshot) -> Panel:
        """Render recent kudos/hr, jobs/hr, and GPU-duty trends: a value, direction, and sparkline.

        Replaces the old momentum gauge, whose self-scaled sparklines carried neither a reference
        value nor a window. Here each row pairs a current figure with a direction marker against the
        window start, and the GPU row adds a duty bar so "how much of the time it is working" reads
        at a glance alongside the over-time shape.
        """
        kudos_series = list(self._kudos_history)[-_TREND_SPARK_WIDTH:]
        gpu_series = list(self._gpu_duty_history)[-_TREND_SPARK_WIDTH:]
        rate, jobs_deltas = self._jobs_per_hour()
        jobs_deltas = jobs_deltas[-_TREND_SPARK_WIDTH:]

        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="right", style="bold cyan", no_wrap=True)
        grid.add_column(justify="right", no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(style="grey50", no_wrap=True)

        kudos_now = "-" if snapshot.kudos_per_hour is None else f"{snapshot.kudos_per_hour:,.0f}"
        kudos_peak = f"peak {max(kudos_series):,.0f}" if kudos_series else ""
        grid.add_row(
            "Kudos/hr",
            kudos_now,
            self._trend_arrow(kudos_series),
            Text(sparkline(kudos_series) or "…", style="cyan"),
            kudos_peak,
        )

        jobs_now = "-" if rate is None else f"{rate:,.0f}"
        grid.add_row(
            "Jobs/hr",
            jobs_now,
            self._trend_arrow(jobs_deltas),
            Text(sparkline(jobs_deltas) or "…", style="green"),
            f"{snapshot.num_jobs_submitted:,} done",
        )

        busy_fraction = snapshot.gpu_utilization_busy_fraction
        if busy_fraction is None and snapshot.gpu_utilization_mean_percent is not None:
            busy_fraction = snapshot.gpu_utilization_mean_percent / 100.0
        duty_bar = Text(mini_bar(busy_fraction, 12), style="green") if busy_fraction is not None else Text("…")
        grid.add_row(
            "GPU duty",
            format_percent(snapshot.gpu_utilization_mean_percent),
            duty_bar,
            Text(sparkline(gpu_series) or "…", style="green"),
            "busy" if busy_fraction and busy_fraction > 0.5 else "idle",
        )

        window = self._trend_window_label()
        return Panel(
            grid,
            title="Trends",
            title_align="left",
            subtitle=Text(window, style="grey50"),
            subtitle_align="right",
            border_style="grey37",
            padding=(0, 1),
        )

    def _trend_window_label(self) -> str:
        """Describe the span the trend buffers currently cover (e.g. ``last 3m 20s``)."""
        samples = list(self._jobs_history)
        if len(samples) < 2:
            return "warming up"
        return f"last {human_duration(samples[-1][0] - samples[0][0])}"

    def _render_compact_bar(
        self,
        report: HealthReport,
        snapshot: WorkerStateSnapshot | None,
        frame: int,
    ) -> Panel:
        """Render the whole worker as one dense status line (the thin, tmux-style view).

        Reuses the hero glyph and the same report/snapshot the full dashboard draws from, so the bar
        always agrees with the larger views. With no snapshot yet it states the phase and headline.
        """
        sep = ("   ·   ", "grey37")
        parts: list[Text | tuple[str, str]] = [
            self._hero_glyph(report, frame),
            (" ", ""),
            (report.phase.value.upper(), f"bold {report.severity.colour}"),
        ]
        if snapshot is None:
            parts += [sep, (report.headline, "grey70")]
            return Panel(Text.assemble(*parts), border_style=report.severity.colour, padding=(0, 1))

        kudos = "-" if snapshot.kudos_per_hour is None else f"{snapshot.kudos_per_hour:,.0f}"
        busy_fraction = snapshot.gpu_utilization_busy_fraction
        if busy_fraction is None and snapshot.gpu_utilization_mean_percent is not None:
            busy_fraction = snapshot.gpu_utilization_mean_percent / 100.0
        alive = sum(1 for process in snapshot.processes if process.is_alive)
        safety = snapshot.jobs_pending_safety_check + snapshot.jobs_being_safety_checked
        age = time.time() - snapshot.timestamp if snapshot.timestamp else None

        line = Text.assemble(*parts)
        line.append_text(Text.assemble(sep, (f"{snapshot.num_jobs_submitted:,}", "bold"), (" done", "grey50")))
        line.append_text(Text.assemble(sep, (kudos, "bold cyan"), ("/h ", "grey50")))
        line.append_text(self._trend_arrow(list(self._kudos_history)[-_TREND_SPARK_WIDTH:]))
        gpu_pct = format_percent(snapshot.gpu_utilization_mean_percent)
        line.append_text(Text.assemble((" gpu ", "grey50"), (gpu_pct, "")))
        if busy_fraction is not None:
            line.append_text(Text(" " + mini_bar(busy_fraction, 8), style="green"))
        line.append_text(
            Text.assemble(
                sep,
                ("q", "grey50"),
                (str(snapshot.jobs_pending_inference), "cyan"),
                ("▸inf", "grey50"),
                (str(snapshot.jobs_in_progress), "green"),
                ("▸saf", "grey50"),
                (str(safety), "grey70"),
                ("▸sub", "grey50"),
                (str(snapshot.jobs_pending_submit), "cyan"),
            ),
        )
        total_procs = len(snapshot.processes)
        procs_colour = "green" if alive == total_procs else "yellow"
        line.append_text(Text.assemble(sep, ("procs ", "grey50"), (f"{alive}/{total_procs}", procs_colour)))
        line.append_text(Text.assemble(sep, ("⌚ ", "grey50"), (f"{human_duration(age)} ago", "grey70")))
        return Panel(line, border_style=report.severity.colour, padding=(0, 1))

    def _render_process_table(self, snapshot: WorkerStateSnapshot, *, detailed: bool = False) -> Table:
        """Build a compact per-process summary table with stable column widths.

        ``detailed`` appends the more technical columns (per-job steps, heartbeat age and type) that
        the F6 toggle reveals; the default view stays lean. Resolution/batch (Size) show either way.
        """
        table = Table(
            title="Processes",
            title_style="bold",
            expand=True,
            border_style="grey37",
            header_style="bold",
        )
        table.add_column("ID", justify="right", width=3)
        table.add_column("Type", width=9)
        table.add_column("State", width=14)
        table.add_column("Model", min_width=18, max_width=24, no_wrap=True)
        table.add_column("Features", min_width=12, no_wrap=True)
        table.add_column("Size", justify="right", width=12, no_wrap=True)
        table.add_column("Progress", width=16, no_wrap=True)
        table.add_column("it/s", justify="right", width=7)
        table.add_column("VRAM", justify="right", min_width=12)
        table.add_column("Done", justify="right", width=5)
        if detailed:
            table.add_column("Steps", justify="right", width=6)
            table.add_column("Heartbeat", justify="right", width=10)
            table.add_column("HB type", width=12, no_wrap=True)

        if not snapshot.processes:
            placeholder = ["-", "-", "waiting for first snapshot", "-", "-", "-", "-", "-", "-", "-"]
            if detailed:
                placeholder += ["-", "-", "-"]
            table.add_row(*placeholder)
            return table

        now = time.time()
        for process in snapshot.processes:
            vram = (
                f"{human_mb(process.vram_usage_mb)} / {human_mb(process.total_vram_mb)}"
                if process.total_vram_mb
                else human_mb(process.vram_usage_mb)
            )
            state_label = label_state(process.last_process_state)
            features_text = (
                ", ".join(process.current_job_features.as_tags())
                if process.current_job_features is not None and not process.current_job_features.is_empty()
                else "-"
            )
            cells: list[str | Text] = [
                str(process.process_id),
                process.process_type.title(),
                Text(state_label, style=self._state_style(process.last_process_state)),
                shorten(process.loaded_horde_model_name, 22),
                features_text,
                self._size_cell(process),
                self._progress_cell(process),
                format_its(process.last_iterations_per_second),
                vram,
                f"{process.num_jobs_completed:,}",
            ]
            if detailed:
                heartbeat_age = now - process.last_heartbeat_timestamp if process.last_heartbeat_timestamp else None
                cells += [
                    str(process.current_job_steps) if process.current_job_steps else "-",
                    self._heartbeat_cell(heartbeat_age, process.is_alive),
                    process.last_heartbeat_type.replace("_", " ").title() if process.is_busy else "-",
                ]
            table.add_row(*cells)
        return table

    @staticmethod
    def _size_cell(process: ProcessSnapshot) -> str:
        """Render the active job's resolution and batch (e.g. ``768×1024 ×2``), or a dash when idle."""
        if not (process.current_job_width and process.current_job_height):
            return "-"
        size = f"{process.current_job_width}×{process.current_job_height}"
        return f"{size} ×{process.batch_amount}" if process.batch_amount > 1 else size

    @staticmethod
    def _progress_cell(process: ProcessSnapshot) -> Text:
        """Render an inline sampling progress bar with step counts, or a dash when not sampling."""
        if process.last_process_state not in _SAMPLING_STATES or not process.last_total_steps:
            return Text("-", style="grey50")
        current = process.last_current_step or 0
        fraction = current / process.last_total_steps
        colour = "green" if fraction >= 0.999 else "cyan"
        return Text.assemble(
            (mini_bar(fraction, 8), colour),
            (f" {current}/{process.last_total_steps}", "grey62"),
        )

    @staticmethod
    def _heartbeat_cell(age: float | None, is_alive: bool) -> Text:
        """Render heartbeat freshness coloured by staleness (shared thresholds with the live view)."""
        if not is_alive:
            return Text("not alive", style="bold red")
        if age is None:
            return Text("-", style="grey62")
        colour = "green" if age < 5 else ("yellow" if age < 15 else "red")
        return Text(f"{age:.1f}s", style=colour)

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

    @staticmethod
    def _render_queue_table(snapshot: WorkerStateSnapshot) -> Panel:
        """Render a table of pending-inference jobs."""
        table = Table(
            title="",
            expand=True,
            border_style="grey37",
            header_style="bold",
            show_header=True,
        )
        table.add_column("Model", min_width=20, no_wrap=True)
        table.add_column("Features")
        table.add_column("Steps", justify="right", width=6)
        table.add_column("Size", justify="right", width=10)

        if not snapshot.pending_jobs:
            table.add_row(Text("queue empty", style="grey50"), "", "", "")
        else:
            for entry in snapshot.pending_jobs:
                features_text = ", ".join(entry.features.as_tags()) if entry.features is not None else "-"
                size = f"{entry.width}×{entry.height}" if entry.width and entry.height else "-"
                table.add_row(
                    shorten(entry.model, 28),
                    features_text,
                    str(entry.steps) if entry.steps else "-",
                    size,
                )

        lane = OverviewView._render_queue_lane(snapshot)
        body = Group(lane, Text(""), table) if lane is not None else table
        return Panel(body, title="Queue", title_align="left", border_style="grey37", padding=(0, 1))

    @staticmethod
    def _render_queue_lane(snapshot: WorkerStateSnapshot) -> Text | None:
        """Render upcoming jobs as cost-scaled blocks so a busy queue visibly "looks" busy.

        Each block's width scales with the job's relative cost (``width×height×steps``) so larger jobs
        read as heavier. Returns None when the queue is empty (the table already says so).
        """
        if not snapshot.pending_jobs:
            return None

        def cost(entry: JobQueueEntry) -> int:
            return (entry.width or 0) * (entry.height or 0) * (entry.steps or 1)

        peak = max((cost(entry) for entry in snapshot.pending_jobs), default=0)
        lane = Text.assemble(("Up next  ", "bold"))
        for entry in snapshot.pending_jobs:
            width = max(1, round(cost(entry) / peak * 6)) if peak > 0 else 1
            label = (
                f"{entry.width}²"
                if entry.width and entry.width == entry.height
                else (f"{entry.width}×{entry.height}" if entry.width and entry.height else "?")
            )
            lane.append("▮" * width, style="cyan")
            lane.append(f"{label} ", style="grey62")
        return lane

    @staticmethod
    def _render_recent_jobs(snapshot: WorkerStateSnapshot) -> Panel:
        """Render a table of recently completed jobs (newest first, last 8)."""
        table = Table(
            title="",
            expand=True,
            border_style="grey37",
            header_style="bold",
            show_header=True,
        )
        table.add_column("", width=2)
        table.add_column("Model / type", min_width=18, no_wrap=True)
        table.add_column("Features")
        table.add_column("Steps", justify="right", width=6)
        table.add_column("E2E", justify="right", width=8)

        recent = list(reversed(snapshot.recent_jobs[-8:]))

        if not recent:
            table.add_row("", Text("no completed jobs yet", style="grey50"), "", "", "")
        else:
            for job in recent:
                glyph = Text("✗", style="red") if job.faulted else Text("✓", style="green")

                if job.is_alchemy:
                    model_text = Text("alchemy", style="grey62")
                else:
                    model_text = Text(shorten(job.model_name, 24) if job.model_name else "?", style="")

                features_text = ", ".join(job.features.as_tags()) if job.features is not None else "-"
                e2e = human_duration(job.e2e_seconds) if job.e2e_seconds is not None else "-"
                table.add_row(
                    glyph,
                    model_text,
                    features_text,
                    str(job.steps) if job.steps else "-",
                    e2e,
                )

        return Panel(table, title="Recent jobs", title_align="left", border_style="grey37", padding=(0, 1))
