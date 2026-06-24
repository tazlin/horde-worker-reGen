"""The overview screen: a live status-monitor hero, a health checklist, then metrics and processes."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from horde_worker_regen.app_state import OverviewViewMode
from horde_worker_regen.process_management.feature_readiness import FeatureReadinessState
from horde_worker_regen.process_management.process_temperature import (
    ProcessTemperature,
    classify_process_temperature,
    temperature_phrase,
)
from horde_worker_regen.process_management.supervisor_channel import (
    CardSnapshot,
    FeatureReadinessSummary,
    JobQueueEntry,
    ProcessSnapshot,
    RecentJobRecord,
    WholeCardResidencyStatus,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.formatters import (
    format_its,
    format_percent,
    gpu_label,
    human_bytes,
    human_duration,
    human_mb,
    job_id_text,
    label_state,
    mini_bar,
    short_baseline,
    shorten,
    sparkline,
    temperature_colour,
)
from horde_worker_regen.tui.health import HealthReport, HealthStatus, WorkerPhase, summarize_skips
from horde_worker_regen.tui.responsive import (
    ColumnSpec,
    DensityTier,
    add_columns,
    intent_ceiling,
    placeholder_row,
    select_columns,
    shed_hint,
)
from horde_worker_regen.tui.widgets.downloads import summarize_download_activity

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

    DEFAULT_CSS = """
    OverviewView #overview-residency {
        width: auto;
    }
    OverviewView #overview-alchemy {
        width: auto;
    }
    """
    """The residency and alchemy boxes (non-expanding Panels) size to their content rather than the full
    screen width, since their content is never wide enough to justify spanning the whole row."""

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
        yield Static(id="overview-gpus")
        yield Static(id="overview-trends")
        yield Static(id="overview-pipeline")
        yield Static(id="overview-processes")
        yield Static(id="overview-residency")
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

        # The residency detail is details-only AND only when the feature applies, so the panel never
        # clutters the detailed view on hardware/configs that never engage whole-card residency.
        residency = snapshot.whole_card_residency if snapshot is not None else None
        show_residency = detailed and residency is not None and (residency.active or residency.possible)
        self.query_one("#overview-residency", Static).display = show_residency

        # The per-card strip rides the normal/details modes (hidden in thin, where the compact bar stands in)
        # and only appears once the worker reports per-card data, so an older worker's overview is unchanged.
        show_gpus = not thin and snapshot is not None and bool(snapshot.per_card)
        self.query_one("#overview-gpus", Static).display = show_gpus

        if snapshot is not None:
            self._maybe_record_trends(snapshot)

        if thin:
            self.query_one("#overview-thin", Static).update(self._render_compact_bar(report, snapshot, frame))
            return

        # The laid-out content width drives column shedding; it is 0 before the first layout pass, where
        # None disables shedding so the very first frame shows the full table rather than collapsing it.
        width = self.content_size.width or None

        self.query_one("#overview-hero", Static).update(self._render_hero(report, snapshot, frame))
        self.query_one("#overview-health", Static).update(
            self._render_health(report, snapshot.feature_readiness if snapshot is not None else None),
        )
        if snapshot is not None:
            if show_gpus:
                self.query_one("#overview-gpus", Static).update(self._render_gpus_strip(snapshot, detailed=detailed))
            self.query_one("#overview-trends", Static).update(self._render_trends(snapshot))
            self.query_one("#overview-pipeline", Static).update(self._render_pipeline_strip(snapshot))
            self.query_one("#overview-processes", Static).update(
                self._render_process_table(snapshot, detailed=detailed, available_width=width),
            )
            if detailed:
                self.query_one("#overview-worker", Static).update(self._render_worker_table(snapshot))
                self.query_one("#overview-alchemy", Static).update(self._render_alchemy_panel(snapshot))
                self.query_one("#overview-queue", Static).update(
                    self._render_queue_table(snapshot, available_width=width),
                )
                self.query_one("#overview-recent", Static).update(
                    self._render_recent_jobs(snapshot, available_width=width),
                )
                if show_residency:
                    self.query_one("#overview-residency", Static).update(
                        self._render_residency_panel(snapshot.whole_card_residency),
                    )

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
            memory_line = self._memory_line(snapshot)
            if memory_line is not None:
                body.append(memory_line)
            download_line = self._download_line(snapshot)
            if download_line is not None:
                body.append(download_line)
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
            if snapshot.whole_card_residency.active:
                body.append(self._residency_banner(snapshot.whole_card_residency))
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

    @staticmethod
    def _memory_line(snapshot: WorkerStateSnapshot) -> Text | None:
        """A system-RAM line: whole-machine in-use/total and the worker's per-role share.

        Returns None when no memory sample has arrived yet (older worker, or before the first report), so
        the hero simply omits the line rather than showing zeroes.
        """
        wire = snapshot.system_memory
        if wire is None or wire.total_bytes <= 0:
            return None
        from horde_worker_regen.process_management.system_memory import ROLE_LABELS

        summary = wire.to_summary()
        used_fraction = summary.used_fraction
        used_pct = f" ({used_fraction * 100:.0f}%)" if used_fraction is not None else ""

        line = Text.assemble(
            ("RAM ", "grey50"),
            (f"{human_bytes(summary.used_bytes)} / {human_bytes(summary.total_bytes)}", "grey70"),
            (used_pct, "grey50"),
            ("  ·  worker ", "grey50"),
            (human_bytes(summary.worker_total_bytes), "bold cyan"),
        )
        role_items = summary.nonzero_role_items()
        if role_items:
            parts = ", ".join(f"{ROLE_LABELS.get(role, role)} {human_bytes(value)}" for role, value in role_items)
            line.append(f" ({parts})", style="grey50")
        return line

    @staticmethod
    def _download_line(snapshot: WorkerStateSnapshot) -> Text | None:
        """A slim hero line for an in-flight background download: model, current-file %, and speed.

        Returns None when nothing is downloading, so the hero stays uncluttered on an idle worker.
        """
        activity = summarize_download_activity(snapshot)
        if activity is None:
            return None
        percent = f"{activity.percent:.0f}%" if activity.percent is not None else "?"
        speed = f"{human_bytes(activity.speed_bps)}/s" if activity.speed_bps else "-"
        marker, marker_style = ("⏸ downloading (paused)", "yellow") if activity.paused else ("⬇ downloading", "cyan")
        line = Text.assemble(
            (marker + "  ", marker_style),
            (shorten(activity.current_name, 28), "grey70"),
            ("  ", ""),
            (percent, "bold"),
        )
        if activity.total is not None:
            line.append("  ·  ", style="grey37")
            line.append(f"{activity.ready}/{activity.total} ready", style="grey50")
        line.append("  ·  ⇣ ", style="grey50")
        line.append(speed, style="grey70")
        return line

    @staticmethod
    def _residency_banner(residency: WholeCardResidencyStatus) -> Text:
        """A hero line explaining an active whole-card residency: what is happening, why, and by how much.

        Reassures the operator that the disappeared inference rows and cycled safety process are a
        deliberate response to a very heavy model, not a fault.
        """
        model = residency.model or "a heavy model"
        line = Text.assemble(
            ("♦ whole-card residency: ", "bold #f0beff"),
            (model, "bold #f0beff"),
            (" has sole use of the GPU", "#f0beff"),
        )
        line.append(
            f" — running {residency.processes_now}/{residency.processes_max} inference processes",
            style="grey70",
        )
        if residency.safety_paused:
            line.append(", safety off-GPU", style="grey70")
        if residency.weights_mb and residency.total_vram_mb:
            line.append(
                f"; needs ~{human_mb(residency.weights_mb)} of {human_mb(residency.total_vram_mb)} for weights",
                style="grey70",
            )
        if residency.phase == "establishing":
            line.append(" (establishing…)", style="italic yellow")
        else:
            line.append(" (intentional for very heavy models)", style="italic grey50")
        return line

    @staticmethod
    def _residency_caption(residency: WholeCardResidencyStatus) -> str | None:
        """A one-line caption for the process table naming what residency paused, or None when inactive.

        Answers "where did my process rows go?" right at the table whose rows the teardown removed.
        """
        if not residency.active:
            return None
        model = residency.model or "a heavy model"
        clauses: list[str] = []
        paused = max(0, residency.processes_max - residency.processes_now)
        if paused:
            plural = "process" if paused == 1 else "processes"
            clauses.append(f"{paused} idle inference {plural} paused")
        if residency.safety_paused:
            clauses.append("safety off-GPU")
        if not clauses:
            return f"Whole-card residency: {model} has sole use of the GPU (intentional)"
        return f"Whole-card residency: {' + '.join(clauses)} for {model} (intentional)"

    @staticmethod
    def _render_residency_panel(residency: WholeCardResidencyStatus) -> Panel:
        """Render the whole-card residency posture and, when active, the live forecast numbers.

        A details-only panel: the operationally-relevant config, plus while a residency is held the hard
        VRAM figures behind the decision (weights, the per-step reserve, the free achievable alone) that
        are a hair too technical for the normal view.
        """
        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="right", style="bold cyan", no_wrap=True)
        grid.add_column()
        grid.add_column(justify="right", style="bold cyan", no_wrap=True)
        grid.add_column()

        grid.add_row(
            "Enabled",
            "yes" if residency.enabled else "no",
            "Safety off-GPU",
            "yes" if residency.safety_off_gpu_enabled else "no",
        )
        overhead = human_mb(residency.per_process_overhead_mb) if residency.per_process_overhead_mb else "auto"
        grid.add_row("Cooldown", f"{residency.cooldown_seconds}s", "Per-process overhead", overhead)
        grid.add_row(
            "Total VRAM",
            human_mb(residency.total_vram_mb) if residency.total_vram_mb else "-",
            "",
            "",
        )

        if residency.active:
            grid.add_row(
                "Phase",
                Text(residency.phase or "-", style="#f0beff"),
                "Model",
                residency.model or "-",
            )
            grid.add_row(
                "Processes",
                f"{residency.processes_now} / {residency.processes_target} (of {residency.processes_max})",
                "Safety paused",
                "yes" if residency.safety_paused else "no",
            )
            grid.add_row("Weights", human_mb(residency.weights_mb), "Step reserve", human_mb(residency.reserve_mb))
            grid.add_row(
                "Free at load",
                human_mb(residency.free_now_mb),
                "Free if alone",
                human_mb(residency.free_if_alone_mb),
            )
            max_resident = str(residency.max_resident_processes) if residency.max_resident_processes else "-"
            cooldown_left = (
                human_duration(residency.cooldown_remaining_seconds)
                if residency.cooldown_remaining_seconds is not None
                else "-"
            )
            grid.add_row("Max co-resident", max_resident, "Restores in", cooldown_left)

        border = "#f0beff" if residency.active else "grey37"
        subtitle = (
            Text("armed; engages for very heavy models", style="grey50")
            if not residency.active and residency.possible
            else None
        )
        return Panel(
            grid,
            title="Whole-card residency",
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style=border,
            padding=(0, 1),
            expand=False,
        )

    def _render_health(self, report: HealthReport, feature_readiness: FeatureReadinessSummary | None = None) -> Panel:
        """Render the health checklist, with a compact feature-readiness line when any feature is engaged."""
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
        features_line = self._feature_readiness_line(feature_readiness)
        if features_line is not None:
            table.add_row(Text("⊟", style="grey62"), Text("Features", style="bold"), features_line)
        return Panel(table, title="Health", title_align="left", border_style="grey37", padding=(0, 1))

    _COMPACT_READINESS_STYLE: dict[FeatureReadinessState, str] = {
        FeatureReadinessState.OFFERED: "green",
        FeatureReadinessState.WAITING: "yellow",
        FeatureReadinessState.MISSING_DEPS: "red",
        FeatureReadinessState.DISABLED: "grey50",
    }

    _COMPACT_READINESS_VERB: dict[FeatureReadinessState, str] = {
        FeatureReadinessState.OFFERED: "offered",
        FeatureReadinessState.WAITING: "downloading",
        FeatureReadinessState.MISSING_DEPS: "no deps",
    }

    def _feature_readiness_line(self, summary: FeatureReadinessSummary | None) -> Text | None:
        """A one-line summary of the engaged gated features, or None when none are engaged.

        Purely-disabled features are omitted so an operator who uses none of them sees no noise; the full
        per-feature table lives on the Downloads tab.
        """
        if summary is None:
            return None
        shown = [feature for feature in summary.gated if feature.state is not FeatureReadinessState.DISABLED]
        if not shown:
            return None
        line = Text()
        for index, feature in enumerate(shown):
            if index:
                line.append("  ·  ", style="grey37")
            verb = self._COMPACT_READINESS_VERB.get(feature.state, feature.state.value)
            line.append(f"{feature.label} ", style="grey70")
            line.append(verb, style=self._COMPACT_READINESS_STYLE.get(feature.state, "grey62"))
        return line

    def _render_worker_table(self, snapshot: WorkerStateSnapshot) -> Table:
        """Build a key/value table of worker identity and configuration."""
        config = snapshot.config
        uptime = human_duration(time.time() - snapshot.session_start_time) if snapshot.session_start_time else "-"

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
        table.add_row("Performance", performance_mode, "Safety on GPU", "yes" if config.safety_on_gpu else "no")
        table.add_row(
            "Allows",
            self._allow_summary(snapshot),
            "",
            "",
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
            if snapshot.lora_pops_blocked_by_disk:
                flags.append("lora OFF (disk full)")
            elif snapshot.lora_pops_blocked_by_downloads or config.effective_allow_lora is False:
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

    @staticmethod
    def _gpus_strip_vram(card: CardSnapshot) -> Text:
        """A compact used-fraction bar plus free/total VRAM for one card, reddened under VRAM pressure."""
        if card.free_vram_mb is None or not card.total_vram_mb:
            return Text("VRAM ?", style="grey50")
        fraction = card.vram_headroom_fraction
        used_fraction = 1.0 - fraction if fraction is not None else 0.0
        style = "red" if card.is_vram_pressured else "green"
        text = Text()
        text.append(mini_bar(used_fraction, 6), style=style)
        text.append(f" {card.free_vram_mb / 1024:.1f}/{card.total_vram_mb / 1024:.1f}G", style="")
        return text

    def _render_gpus_strip(self, snapshot: WorkerStateSnapshot, *, detailed: bool) -> Panel:
        """Render the per-card strip: one compact row per card, with residency/fault detail in details mode.

        The single collapsed card on a single-GPU host is intentional (presentational consistency). In
        details mode each row also names the whole-card residency it holds and flags any models gone
        locally unservable on it, so a pressured or quarantining card stands out without leaving the tab.
        """
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold", no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(justify="right", no_wrap=True)
        grid.add_column(no_wrap=True)
        for card in snapshot.per_card:
            tail = Text(f"{card.busy_contexts} job{'s' if card.busy_contexts != 1 else ''}", style="green")
            if detailed:
                if card.residency_model:
                    tail.append_text(
                        Text(f"  ★ {shorten(card.residency_model, 14)} ({card.residency_phase})", style="#f0beff"),
                    )
                if card.unservable_models:
                    tail.append_text(Text(f"  ⚠ {len(card.unservable_models)} unservable", style="bold red"))
            grid.add_row(
                gpu_label(card.device_index, card.device_name, card.kind),
                self._gpus_strip_vram(card),
                f"{card.loaded_contexts}/{card.target_process_count} ctx",
                tail,
            )
        return Panel(grid, title="GPUs", title_align="left", border_style="grey37", padding=(0, 1))

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

        activity = summarize_download_activity(snapshot)
        if activity is not None:
            percent = f"{activity.percent:.0f}%" if activity.percent is not None else "?"
            speed = f"{human_bytes(activity.speed_bps)}/s" if activity.speed_bps else "-"
            marker, marker_style = ("⏸", "yellow") if activity.paused else ("⬇", "cyan")
            line.append_text(
                Text.assemble(sep, (f"{marker} ", marker_style), (percent, "bold"), (" ", ""), (speed, "grey62")),
            )

        line.append_text(Text.assemble(sep, ("⌚ ", "grey50"), (f"{human_duration(age)} ago", "grey70")))
        return Panel(line, border_style=report.severity.colour, padding=(0, 1))

    def _render_process_table(
        self,
        snapshot: WorkerStateSnapshot,
        *,
        detailed: bool = False,
        available_width: int | None = None,
    ) -> Table:
        """Build a per-process summary table whose columns shed to fit ``available_width``.

        Every row names the slot's active job by its colour-coded id and the loaded model's baseline, so
        one job can be followed across the dashboard (the same colour appears in the queue, recent-jobs
        and live views). The columns are tagged by :class:`DensityTier`: the essentials always show, the
        richer columns drop first on a narrow terminal, and the F6 details view's diagnostic columns (per-
        job steps, heartbeat age and type) appear only when both that intent and the width allow. With no
        width known (``None``) nothing sheds. When the width clamps below what was wanted, the caption
        says so, unless a whole-card residency note has already claimed it.
        """
        layout = select_columns(
            _PROCESS_COLUMNS,
            ceiling=intent_ceiling(detailed),
            available_width=available_width,
        )
        table = Table(
            title="Processes",
            title_style="bold",
            expand=True,
            border_style="grey37",
            header_style="bold",
        )
        add_columns(table, layout.columns)

        if not snapshot.processes:
            table.add_row(*placeholder_row(layout.columns, "State", "waiting for first snapshot"))
            return table

        now = time.time()
        residency = snapshot.whole_card_residency
        pending_models = frozenset(entry.model for entry in snapshot.pending_jobs if entry.model)
        # Group each card's slots together (then by slot id), so the GPU column reads as contiguous blocks on
        # a multi-GPU host; single-GPU is unaffected (every slot is device 0, leaving slot-id order intact).
        for process in sorted(snapshot.processes, key=lambda p: (p.device_index, p.process_id)):
            row = _ProcessRow(process=process, now=now, residency=residency, pending_models=pending_models)
            table.add_row(*[spec.render(row) for spec in layout.columns])

        caption = self._residency_caption(residency)
        if caption is not None:
            table.caption = caption
            table.caption_style = "italic #f0beff"
        elif (hint := shed_hint(layout)) is not None:
            table.caption = hint
            table.caption_style = "italic grey50"
        return table

    @staticmethod
    def _process_state_cell(row: _ProcessRow) -> Text:
        """The State cell: a temperature-led label (``Hot · sampling``), ★ for the whole-card slot.

        Folding the slot's temperature into the label is what makes a primed slot read as primed rather
        than a uniform ``Idle``: ``Next`` (a queued job will use this resident model), ``Warm`` (resident,
        nothing queued for it), ``Priming`` (loading), and ``Cold`` (empty) all render distinctly where the
        raw state would otherwise be ``WAITING_FOR_JOB`` for the first three.
        """
        process = row.process
        temperature = classify_process_temperature(
            state=process.last_process_state,
            loaded_model=process.loaded_horde_model_name,
            pending_models=row.pending_models,
        )
        if temperature == ProcessTemperature.DOWN:
            # Terminal/failed slots keep their plain state label and colour; temperature framing would only
            # obscure why the slot is gone.
            state = process.last_process_state
            cell = Text(label_state(state), style=OverviewView._state_style(state))
        else:
            phrase = temperature_phrase(temperature, process.last_process_state)
            cell = Text(
                f"{temperature.value.title()} · {phrase}",
                style=OverviewView._temperature_style(temperature),
            )
        residency = row.residency
        if residency.active and residency.model and process.loaded_horde_model_name == residency.model:
            cell.append(" ★", style="#f0beff")
        return cell

    @staticmethod
    def _temperature_style(temperature: ProcessTemperature) -> str:
        """Map a process temperature to a display colour (hot=active green, cold=dim grey)."""
        return temperature_colour(temperature)

    @staticmethod
    def _vram_cell(process: ProcessSnapshot) -> str:
        """The GPU VRAM cell: used / total when total is known, else just the used figure."""
        if process.total_vram_mb:
            return f"{human_mb(process.vram_usage_mb)} / {human_mb(process.total_vram_mb)}"
        return human_mb(process.vram_usage_mb)

    @staticmethod
    def _features_cell(process: ProcessSnapshot) -> str:
        """The Features cell: the active job's feature tags, or a dash when none apply."""
        features = process.current_job_features
        if features is not None and not features.is_empty():
            return ", ".join(features.as_tags())
        return "-"

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
    def _render_queue_table(snapshot: WorkerStateSnapshot, *, available_width: int | None = None) -> Panel:
        """Render a table of pending-inference jobs, shedding columns to fit ``available_width``."""
        layout = select_columns(_QUEUE_COLUMNS, ceiling=DensityTier.WIDE, available_width=available_width)
        table = Table(
            title="",
            expand=True,
            border_style="grey37",
            header_style="bold",
            show_header=True,
        )
        add_columns(table, layout.columns)

        if not snapshot.pending_jobs:
            table.add_row(*placeholder_row(layout.columns, "Model", "queue empty"))
        else:
            for entry in snapshot.pending_jobs:
                table.add_row(*[spec.render(entry) for spec in layout.columns])

        lane = OverviewView._render_queue_lane(snapshot)
        body = Group(lane, Text(""), table) if lane is not None else table
        subtitle = shed_hint(layout)
        return Panel(
            body,
            title="Queue",
            title_align="left",
            subtitle=Text(subtitle, style="grey50") if subtitle else None,
            subtitle_align="right",
            border_style="grey37",
            padding=(0, 1),
        )

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
    def _render_recent_jobs(snapshot: WorkerStateSnapshot, *, available_width: int | None = None) -> Panel:
        """Render a table of recently completed jobs (newest first, last 8), shedding to fit the width."""
        layout = select_columns(_RECENT_COLUMNS, ceiling=DensityTier.WIDE, available_width=available_width)
        table = Table(
            title="",
            expand=True,
            border_style="grey37",
            header_style="bold",
            show_header=True,
        )
        add_columns(table, layout.columns)

        recent = list(reversed(snapshot.recent_jobs[-8:]))
        if not recent:
            table.add_row(*placeholder_row(layout.columns, "Model / type", "no completed jobs yet"))
        else:
            for job in recent:
                table.add_row(*[spec.render(job) for spec in layout.columns])

        subtitle = shed_hint(layout)
        return Panel(
            table,
            title="Recent jobs",
            title_align="left",
            subtitle=Text(subtitle, style="grey50") if subtitle else None,
            subtitle_align="right",
            border_style="grey37",
            padding=(0, 1),
        )

    @staticmethod
    def _recent_model_cell(job: RecentJobRecord) -> Text:
        """The Model/type cell: ``alchemy`` for alchemy jobs, else the (shortened) model name."""
        if job.is_alchemy:
            return Text("alchemy", style="grey62")
        return Text(shorten(job.model_name, 24) if job.model_name else "?", style="")


@dataclass(frozen=True)
class _ProcessRow:
    """One process-table row paired with the render-time context its cells need.

    Bundling the snapshot-wide ``now`` and ``residency`` with the per-process snapshot lets every process
    column be a plain ``ColumnSpec[_ProcessRow]`` whose render takes a single argument.
    """

    process: ProcessSnapshot
    now: float
    residency: WholeCardResidencyStatus
    pending_models: frozenset[str] = frozenset()
    """Models named by queued (not-yet-running) jobs, so a primed slot they target reads as 'next'."""


def _heartbeat_age(row: _ProcessRow) -> float | None:
    """Seconds since the process last sent a heartbeat, or None when it never has."""
    timestamp = row.process.last_heartbeat_timestamp
    return row.now - timestamp if timestamp else None


_PROCESS_COLUMNS: list[ColumnSpec[_ProcessRow]] = [
    ColumnSpec("ID", DensityTier.ESSENTIAL, lambda r: str(r.process.process_id), justify="right", width=3),
    ColumnSpec("Type", DensityTier.ESSENTIAL, lambda r: r.process.process_type.title(), width=9),
    ColumnSpec("GPU", DensityTier.NORMAL, lambda r: str(r.process.device_index), justify="right", width=4),
    ColumnSpec("State", DensityTier.ESSENTIAL, OverviewView._process_state_cell, width=18, no_wrap=True),
    ColumnSpec("Job", DensityTier.ESSENTIAL, lambda r: job_id_text(r.process.current_job_id), width=8, no_wrap=True),
    ColumnSpec(
        "Progress",
        DensityTier.ESSENTIAL,
        lambda r: OverviewView._progress_cell(r.process),
        width=12,
        no_wrap=True,
    ),
    ColumnSpec(
        "Model",
        DensityTier.NORMAL,
        lambda r: shorten(r.process.loaded_horde_model_name, 22),
        min_width=14,
        max_width=22,
        no_wrap=True,
    ),
    ColumnSpec(
        "Baseline",
        DensityTier.NORMAL,
        lambda r: short_baseline(r.process.loaded_horde_model_baseline),
        width=8,
        no_wrap=True,
    ),
    ColumnSpec("Done", DensityTier.NORMAL, lambda r: f"{r.process.num_jobs_completed:,}", justify="right", width=5),
    ColumnSpec(
        "Features",
        DensityTier.WIDE,
        lambda r: OverviewView._features_cell(r.process),
        min_width=10,
        no_wrap=True,
    ),
    ColumnSpec(
        "Size",
        DensityTier.WIDE,
        lambda r: OverviewView._size_cell(r.process),
        justify="right",
        width=11,
        no_wrap=True,
    ),
    ColumnSpec(
        "it/s",
        DensityTier.WIDE,
        lambda r: format_its(r.process.last_iterations_per_second),
        justify="right",
        width=6,
    ),
    ColumnSpec(
        "GPU VRAM",
        DensityTier.WIDE,
        lambda r: OverviewView._vram_cell(r.process),
        justify="right",
        min_width=15,
        no_wrap=True,
    ),
    ColumnSpec(
        "Steps",
        DensityTier.DETAILS,
        lambda r: str(r.process.current_job_steps) if r.process.current_job_steps else "-",
        justify="right",
        width=6,
    ),
    ColumnSpec(
        "Heartbeat",
        DensityTier.DETAILS,
        lambda r: OverviewView._heartbeat_cell(_heartbeat_age(r), r.process.is_alive),
        justify="right",
        width=9,
    ),
    ColumnSpec(
        "HB type",
        DensityTier.DETAILS,
        lambda r: r.process.last_heartbeat_type.replace("_", " ").title() if r.process.is_busy else "-",
        width=11,
        no_wrap=True,
    ),
]
"""The process table's columns, tagged by the density tier at which each appears."""


def _entry_features(entry: JobQueueEntry) -> str:
    """Comma-joined feature tags for a queued job, or a dash when it carries none."""
    return ", ".join(entry.features.as_tags()) if entry.features is not None else "-"


def _entry_size(entry: JobQueueEntry) -> str:
    """A queued job's ``width×height``, or a dash when its dimensions are unknown."""
    return f"{entry.width}×{entry.height}" if entry.width and entry.height else "-"


_QUEUE_COLUMNS: list[ColumnSpec[JobQueueEntry]] = [
    ColumnSpec("Job", DensityTier.ESSENTIAL, lambda e: job_id_text(e.job_id), width=8, no_wrap=True),
    ColumnSpec("Model", DensityTier.ESSENTIAL, lambda e: shorten(e.model, 28), min_width=20, no_wrap=True),
    ColumnSpec("Baseline", DensityTier.NORMAL, lambda e: short_baseline(e.baseline), width=8, no_wrap=True),
    ColumnSpec("Size", DensityTier.NORMAL, _entry_size, justify="right", width=10),
    ColumnSpec("Features", DensityTier.WIDE, _entry_features, min_width=10),
    ColumnSpec("Steps", DensityTier.WIDE, lambda e: str(e.steps) if e.steps else "-", justify="right", width=6),
]
"""The pending-jobs queue table's columns, tagged by density tier."""


def _recent_features(job: RecentJobRecord) -> str:
    """Comma-joined feature tags for a completed job, or a dash when it carries none."""
    return ", ".join(job.features.as_tags()) if job.features is not None else "-"


def _recent_size(job: RecentJobRecord) -> str:
    """A completed job's ``width×height``, or a dash when its dimensions are unknown."""
    return f"{job.width}×{job.height}" if job.width and job.height else "-"


_RECENT_COLUMNS: list[ColumnSpec[RecentJobRecord]] = [
    ColumnSpec(
        "",
        DensityTier.ESSENTIAL,
        lambda j: Text("✗", style="red") if j.faulted else Text("✓", style="green"),
        width=2,
    ),
    ColumnSpec("Job", DensityTier.ESSENTIAL, lambda j: job_id_text(j.job_id), width=8, no_wrap=True),
    ColumnSpec("Model / type", DensityTier.ESSENTIAL, OverviewView._recent_model_cell, min_width=18, no_wrap=True),
    ColumnSpec("Baseline", DensityTier.NORMAL, lambda j: short_baseline(j.baseline), width=8, no_wrap=True),
    ColumnSpec("Size", DensityTier.NORMAL, _recent_size, justify="right", width=10),
    ColumnSpec(
        "E2E",
        DensityTier.NORMAL,
        lambda j: human_duration(j.e2e_seconds) if j.e2e_seconds is not None else "-",
        justify="right",
        width=8,
    ),
    ColumnSpec("Features", DensityTier.WIDE, _recent_features, min_width=10),
    ColumnSpec("Steps", DensityTier.WIDE, lambda j: str(j.steps) if j.steps else "-", justify="right", width=6),
    ColumnSpec(
        "Queue",
        DensityTier.WIDE,
        lambda j: human_duration(j.queue_wait_seconds) if j.queue_wait_seconds is not None else "-",
        justify="right",
        width=7,
    ),
    ColumnSpec(
        "Safety",
        DensityTier.WIDE,
        lambda j: human_duration(j.safety_seconds) if j.safety_seconds is not None else "-",
        justify="right",
        width=7,
    ),
]
"""The recent-jobs table's columns, tagged by density tier."""
