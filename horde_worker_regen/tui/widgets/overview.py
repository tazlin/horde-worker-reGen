"""The overview screen: a live status-monitor hero, a health checklist, then metrics and processes."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Static

from horde_worker_regen.app_state import OverviewTrendWindow, OverviewViewMode
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    RECENT_JOBS_IN_SNAPSHOT,
    CardSnapshot,
    FeatureReadinessSummary,
    JobQueueEntry,
    PopGovernorsSnapshot,
    ProcessSnapshot,
    RecentJobRecord,
    SchedulingGovernanceSnapshot,
    WholeCardResidencyStatus,
    WorkerStateSnapshot,
    WorkLedgerEntry,
    WorkLedgerStage,
)
from horde_worker_regen.process_management.lifecycle.process_temperature import (
    ProcessTemperature,
    classify_process_temperature,
    temperature_phrase,
)
from horde_worker_regen.process_management.models.feature_readiness import FeatureReadinessState
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind
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
from horde_worker_regen.tui.health import (
    HealthReport,
    HealthStatus,
    WorkerPhase,
    is_gpu_duty_low,
    summarize_skips,
)
from horde_worker_regen.tui.responsive import (
    ColumnSpec,
    DensityTier,
    add_columns,
    intent_ceiling,
    placeholder_row,
    select_columns,
    shed_hint,
)
from horde_worker_regen.tui.trends import (
    fixed_counter_deltas,
    fixed_float_buckets,
    fixed_ratio_deltas,
    trend_bounds,
)
from horde_worker_regen.tui.widgets.downloads import summarize_download_activity
from horde_worker_regen.tui.widgets.overview_layout import OVERVIEW_ELEMENTS
from horde_worker_regen.update_check import UpdateInfo, current_version

_TREND_HISTORY = 21600
"""How many one-second trend samples the Trends region retains (up to roughly six hours)."""

_TREND_SAMPLE_INTERVAL = 1.0
"""Minimum wall-clock seconds between recorded trend samples, so the window spans minutes not frames."""

_TREND_SPARK_WIDTH = 48
"""Maximum number of samples drawn in a Trends sparkline, keeping the line terminal-friendly."""

_TWO_COLUMN_MIN_WIDTH = 100
"""Minimum Overview width (columns) at which panels lay out two-up instead of a single stack."""

_THREE_COLUMN_MIN_WIDTH = 165
"""Minimum Overview width (columns) at which the widest panel regions spread across three columns."""

_SAMPLING_STATES = frozenset({"INFERENCE_STARTING", "POST_PROCESSING", "ALCHEMY_STARTING"})
"""States with a live sampling step/it-s; outside these the snapshot's step numbers are last-job residue."""

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SERVING_PULSE = ("dark_green", "green", "green3", "bright_green", "green3", "green")


def _join_styled(parts: list[tuple[str, str]], separator: str) -> list[tuple[str, str] | str]:
    """Return Rich Text.assemble fragments with a plain separator between styled parts."""
    fragments: list[tuple[str, str] | str] = []
    for index, part in enumerate(parts):
        if index:
            fragments.append(separator)
        fragments.append(part)
    return fragments


def _preload_decision_style(decision: str) -> str:
    """Style preload-admission decisions by whether they admitted, skipped, or deferred work."""
    if not decision:
        return "grey50"
    if decision in {"admit", "prestage", "terminal_admit", "already_loaded"}:
        return "green"
    if decision in {"next_job", "quarantined"}:
        return "grey62"
    return "yellow"


_STATIC_GLYPHS: dict[WorkerPhase, str] = {
    WorkerPhase.STOPPED: "■",
    WorkerPhase.CRASHED: "✗",
    WorkerPhase.DISCONNECTED: "✗",
    WorkerPhase.DEGRADED: "▲",
    WorkerPhase.MAINTENANCE: "⏸",
    WorkerPhase.PAUSED: "⏸",
    WorkerPhase.IDLE: "○",
    WorkerPhase.READY: "●",
}


class OverviewView(Vertical):
    """A dashboard led by a living status hero and a health checklist."""

    DEFAULT_CSS = """
    OverviewView #overview-body {
        height: 1fr;
    }
    OverviewView .overview-row,
    OverviewView .overview-grid {
        height: auto;
        width: 100%;
        layout: vertical;
    }
    OverviewView .overview-row Static,
    OverviewView .overview-grid Static {
        height: auto;
        width: 100%;
    }
    OverviewView #overview-workload-column {
        height: auto;
        width: 100%;
        layout: vertical;
    }
    OverviewView #overview-residency {
        width: auto;
    }
    OverviewView #overview-alchemy {
        width: auto;
    }
    OverviewView #overview-update-nag {
        height: auto;
        width: 100%;
    }
    OverviewView #overview-intent-row {
        layout: vertical;
        height: auto;
        width: 100%;
    }
    OverviewView #overview-intent-row Static {
        width: 100%;
        height: auto;
        margin: 0;
    }

    /* Two-column layout, applied whenever the width can host it (any density mode). */
    OverviewView.-cols-2 #overview-core-grid {
        layout: grid;
        grid-size: 2;
        grid-columns: 2fr 1fr;
        grid-rows: auto;
        grid-gutter: 0 1;
    }
    OverviewView.-cols-2 #overview-core-grid Static,
    OverviewView.-cols-2 #overview-workload-column {
        width: 100%;
        height: auto;
    }
    /* Trends carries a wide sparkline, so it takes the full grid width on its own row rather than being
       squeezed into a fractional column (which would scrunch the label/value/spark/peak layout). */
    OverviewView.-cols-2 #overview-trends {
        column-span: 2;
    }
    OverviewView.-cols-2 .overview-row {
        layout: horizontal;
        grid-gutter: 0 1;
    }
    OverviewView.-cols-2 .overview-row Static {
        width: 1fr;
        height: auto;
        margin: 0 1 0 0;
    }
    OverviewView.-cols-2 #overview-ops-row {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-rows: auto;
        grid-gutter: 1 1;
    }
    OverviewView.-cols-2 #overview-ops-row Static {
        width: 100%;
        height: auto;
        margin: 0;
    }
    OverviewView.-cols-2 #overview-worker {
        row-span: 2;
    }
    OverviewView.-cols-2 #overview-intent-row {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-rows: auto;
        grid-gutter: 1 1;
    }
    OverviewView.-cols-2 #overview-queue {
        column-span: 2;
    }

    /* Three-column layout, layered on top of -cols-2 on a roomy terminal: spread the compact ops-row
       panels (worker/alchemy/residency) three-up. The core grid stays two-up on purpose: Health wants a
       wide detail column and Trends spans full width, so a third core column would only cramp them. */
    OverviewView.-cols-3 #overview-ops-row {
        grid-size: 3;
        grid-columns: 1fr 1fr 1fr;
    }
    OverviewView.-cols-3 #overview-worker {
        row-span: 1;
    }
    """
    """Row containers stay vertical (single column) by default; ``-cols-2``/``-cols-3`` opt into horizontal
    adjacency once the terminal is wide enough (see ``_apply_overview_columns``). ``-cols-3`` is layered on
    top of ``-cols-2`` and widens only the compact ops-row; Health/Trends keep their two-column band."""

    def __init__(self) -> None:
        """Set up the view, including the client-side trend history for the Trends sparklines."""
        super().__init__()
        self._gpu_duty_history: deque[tuple[float, float]] = deque(maxlen=_TREND_HISTORY)
        # (timestamp, cumulative kudos, cumulative productive seconds): the windowed kudos/hr divides the
        # kudos earned across the window by the productive seconds earned across it, so idle/maintenance
        # time inside the window is excluded rather than dragging a raw-wall-clock rate down.
        self._kudos_history: deque[tuple[float, float, float]] = deque(maxlen=_TREND_HISTORY)
        self._jobs_history: deque[tuple[float, int]] = deque(maxlen=_TREND_HISTORY)
        self._forms_history: deque[tuple[float, int]] = deque(maxlen=_TREND_HISTORY)
        self._last_trend_sample = 0.0
        self._trend_window = OverviewTrendWindow.FIFTEEN_MINUTES
        self._trend_epoch = time.time()
        self._trend_notice: str | None = None
        self._trend_session_start: float | None = None
        self._update_info: UpdateInfo | None = None

    def set_trend_window(self, window: OverviewTrendWindow) -> None:
        """Set the rendered trend window without discarding the session sample buffers."""
        self._trend_window = window

    def trend_window(self) -> OverviewTrendWindow:
        """Return the current trend window."""
        return self._trend_window

    def soft_reset_trends(self, *, notice: str = "Trends reset; waiting for fresh samples.") -> None:
        """Reset the client-side trend view: clear the sample buffers and start a fresh epoch.

        Every buffer here is display-only (no scheduling, throttle, or dispatch path reads them), so clearing
        them cannot affect worker behavior. Backend session accounting (the console kudos line, the Stats-tab
        "this session" totals) is deliberately left running; this reset is purely the operator's trend view.
        """
        self._gpu_duty_history.clear()
        self._kudos_history.clear()
        self._jobs_history.clear()
        self._forms_history.clear()
        self._trend_epoch = time.time()
        self._trend_notice = notice

    def note_config_changed(self) -> None:
        """Mark trend output as stabilizing after a capacity/workload-affecting config change."""
        self.soft_reset_trends(notice="Config changed; trends may take time to restabilize.")

    def set_update_available(self, info: UpdateInfo) -> None:
        """Record that a newer release is available; the nag box shows on the next update_view call."""
        self._update_info = info

    def compose(self) -> ComposeResult:
        """Lay out the compact bar plus the hero, health, trends, pipeline, and detail tables.

        Only one set is visible at a time: ``update_view`` toggles each node's ``display`` from the
        active :class:`OverviewViewMode` (thin shows only the compact bar; the worker/alchemy/queue/
        recent statics appear only in details mode).
        """
        yield Static(id="overview-thin")
        yield Static(id="overview-hero")
        yield Static(id="overview-update-nag")
        with VerticalScroll(id="overview-body"):
            with Container(id="overview-core-grid", classes="overview-grid"):
                yield Static(id="overview-health")
                with Container(id="overview-workload-column"):
                    yield Static(id="overview-gpus")
                    yield Static(id="overview-pipeline")
                yield Static(id="overview-trends")
            with Container(id="overview-intent-row"):
                yield Static(id="overview-queue")
                yield Static(id="overview-intent")
                yield Static(id="overview-governance")
            yield Static(id="overview-work")
            yield Static(id="overview-processes")
            with Container(id="overview-ops-row", classes="overview-row"):
                yield Static(id="overview-worker")
                yield Static(id="overview-alchemy")
                yield Static(id="overview-residency")
            with Container(id="overview-history-row", classes="overview-row"):
                yield Static(id="overview-recent")

    _NORMAL_NODE_IDS = (
        "#overview-hero",
        "#overview-health",
        "#overview-trends",
        "#overview-pipeline",
        "#overview-work",
        "#overview-processes",
    )
    """Statics shown in normal (and details) mode, hidden in thin mode."""

    _DETAIL_NODE_IDS = ("#overview-recent",)
    """Statics shown only in details mode (the demoted panels)."""

    _NORMAL_ROW_IDS = ("#overview-core-grid", "#overview-body", "#overview-intent-row")
    """Row containers shown in normal/details mode, hidden in thin mode."""

    _DETAIL_ROW_IDS = ("#overview-history-row",)
    """Row containers shown only in details mode."""

    def on_mount(self) -> None:
        """Pick the column count for the initial size so the first paint is not misaligned."""
        self._apply_overview_columns()

    def on_resize(self) -> None:
        """Re-pick the two/three-column layout when the terminal is resized."""
        self._apply_overview_columns()

    def _apply_overview_columns(self) -> None:
        """Stamp ``-cols-2``/``-cols-3`` from the current width so panels lay out two- or three-up.

        The classes are additive: ``-cols-3`` layers on top of ``-cols-2`` (the base horizontal rules live
        under ``-cols-2``), so a three-column screen carries both. Below the two-column threshold neither is
        set and every region falls back to the single stacked column defined by the base CSS.
        """
        width = self.size.width
        self.set_class(width >= _TWO_COLUMN_MIN_WIDTH, "-cols-2")
        self.set_class(width >= _THREE_COLUMN_MIN_WIDTH, "-cols-3")

    def update_view(
        self,
        report: HealthReport,
        snapshot: WorkerStateSnapshot | None,
        *,
        frame: int,
        mode: OverviewViewMode = OverviewViewMode.NORMAL,
        trend_window: OverviewTrendWindow | None = None,
        show_recent_work_ledger_jobs: bool = True,
        hidden_keys: frozenset[str] = frozenset(),
        reveal_hidden: bool = False,
    ) -> None:
        """Refresh the visible regions for the active view ``mode`` from the report and snapshot.

        ``hidden_keys`` are the operator's hidden-element choices (registry keys); ``reveal_hidden``
        temporarily un-suppresses them for the quick reveal toggle. Hiding is layered on top of the mode
        and per-panel guards: it can only remove an element that would otherwise show, never force one on.
        """
        if trend_window is not None:
            self.set_trend_window(trend_window)
        thin = mode is OverviewViewMode.THIN
        detailed = mode is OverviewViewMode.DETAILS
        self._apply_overview_columns()
        # The laid-out content width drives column shedding. It is 0 before the first layout pass, where
        # None disables shedding so the first frame renders fully.
        width = self.content_size.width or None

        effective_hidden = frozenset() if reveal_hidden else hidden_keys
        hidden_nodes = {element.node_id for element in OVERVIEW_ELEMENTS if element.key in effective_hidden}

        def visible(node_id: str, base: bool) -> bool:
            """Final display for a registry node: its mode/guard base minus any operator hide."""
            return base and node_id not in hidden_nodes

        self.query_one("#overview-thin", Static).display = thin
        for row_id in self._NORMAL_ROW_IDS:
            self.query_one(row_id).display = not thin
        for row_id in self._DETAIL_ROW_IDS:
            self.query_one(row_id).display = detailed

        # Governance folds the pop governors and the scheduler-governance diagnostics into one grouping:
        # the pop half surfaces whenever a governor is engaged (so a teardown/cooldown/backpressure is never
        # a silent mystery, and in details mode also once one has a session history), while the scheduling
        # half is a details-mode diagnostic.
        residency = snapshot.whole_card_residency if snapshot is not None else None
        governors = snapshot.pop_governors if snapshot is not None else None
        scheduling = snapshot.scheduling_governance if snapshot is not None else None
        governors_have_history = detailed and governors is not None and bool(governors.governors)
        show_pop = governors is not None and (governors.any_active or governors_have_history)
        show_sched = detailed and scheduling is not None

        show = {
            "#overview-hero": visible("#overview-hero", not thin),
            "#overview-health": visible("#overview-health", not thin),
            "#overview-trends": visible("#overview-trends", not thin),
            "#overview-pipeline": visible("#overview-pipeline", not thin),
            "#overview-work": visible("#overview-work", not thin),
            "#overview-processes": visible("#overview-processes", not thin),
            "#overview-intent": visible("#overview-intent", not thin and snapshot is not None),
            "#overview-queue": visible("#overview-queue", detailed and snapshot is not None),
            "#overview-governance": visible("#overview-governance", not thin and (show_pop or show_sched)),
            "#overview-gpus": visible("#overview-gpus", not thin and snapshot is not None and bool(snapshot.per_card)),
            "#overview-worker": visible("#overview-worker", not thin and snapshot is not None),
            "#overview-alchemy": visible(
                "#overview-alchemy",
                not thin and snapshot is not None and self._show_alchemy_panel(snapshot),
            ),
            # Residency detail is details-only AND only when the feature applies, so it never clutters the
            # detailed view on hardware/configs that never engage whole-card residency.
            "#overview-residency": visible(
                "#overview-residency",
                detailed and residency is not None and (residency.active or residency.possible),
            ),
            "#overview-recent": visible("#overview-recent", detailed and snapshot is not None),
        }
        for node_id, is_shown in show.items():
            self.query_one(node_id, Static).display = is_shown
        # The ops-row container carries the worker/alchemy/residency panels; show it while any of them do.
        self.query_one("#overview-ops-row").display = (
            show["#overview-worker"] or show["#overview-alchemy"] or show["#overview-residency"]
        )

        nag = self.query_one("#overview-update-nag", Static)
        if self._update_info is not None:
            nag.display = True
            nag.update(self._render_update_nag(self._update_info))
        else:
            nag.display = False

        if snapshot is not None:
            self._maybe_record_trends(snapshot)

        if thin:
            self.query_one("#overview-thin", Static).update(self._render_compact_bar(report, snapshot, frame))
            return

        if show["#overview-hero"]:
            self.query_one("#overview-hero", Static).update(self._render_hero(report, snapshot, frame))
        if show["#overview-health"]:
            self.query_one("#overview-health", Static).update(
                self._render_health(
                    report,
                    snapshot.feature_readiness if snapshot is not None else None,
                    models_loaded=len(snapshot.active_models) if snapshot is not None else None,
                ),
            )
        if snapshot is None:
            return

        if show["#overview-intent"]:
            self.query_one("#overview-intent", Static).update(self._render_intent(snapshot, detailed=detailed))
        if show["#overview-governance"]:
            self.query_one("#overview-governance", Static).update(
                self._render_governance_combined(
                    governors if show_pop else None,
                    scheduling if show_sched else None,
                    detailed=detailed,
                ),
            )
        if show["#overview-queue"]:
            self.query_one("#overview-queue", Static).update(self._render_queue_table(snapshot, available_width=width))
        if show["#overview-gpus"]:
            self.query_one("#overview-gpus", Static).update(self._render_gpus_strip(snapshot, detailed=detailed))
        if show["#overview-trends"]:
            self.query_one("#overview-trends", Static).update(self._render_trends(snapshot))
        if show["#overview-pipeline"]:
            self.query_one("#overview-pipeline", Static).update(self._render_pipeline_strip(snapshot))
        if show["#overview-work"]:
            self.query_one("#overview-work", Static).update(
                self._render_work_ledger(
                    snapshot,
                    detailed=detailed,
                    available_width=width,
                    show_recent_jobs=show_recent_work_ledger_jobs,
                ),
            )
        if show["#overview-processes"]:
            self.query_one("#overview-processes", Static).update(
                self._render_process_table(snapshot, detailed=detailed, available_width=width),
            )
        if show["#overview-worker"]:
            self.query_one("#overview-worker", Static).update(self._render_worker_table(snapshot))
        if show["#overview-alchemy"]:
            self.query_one("#overview-alchemy", Static).update(self._render_alchemy_panel(snapshot))
        if show["#overview-recent"]:
            self.query_one("#overview-recent", Static).update(
                self._render_recent_jobs(snapshot, available_width=width)
            )
        if show["#overview-residency"] and residency is not None:
            self.query_one("#overview-residency", Static).update(self._render_residency_panel(residency))

    def _maybe_record_trends(self, snapshot: WorkerStateSnapshot) -> None:
        """Record a trend sample at most once per :data:`_TREND_SAMPLE_INTERVAL` of wall-clock time."""
        now = time.time()
        if now - self._last_trend_sample < _TREND_SAMPLE_INTERVAL:
            return
        self._last_trend_sample = now
        self._record_trends(snapshot)

    def _record_trends(self, snapshot: WorkerStateSnapshot) -> None:
        """Append one timestamped sample of GPU-duty, kudos/hr, and the cumulative job counter."""
        sample = snapshot.latest_stats_sample
        now = sample.timestamp if sample is not None else (snapshot.timestamp or time.time())
        self._trend_session_start = snapshot.session_start_time or self._trend_session_start
        gpu_duty = sample.gpu_duty_percent if sample is not None else snapshot.gpu_utilization_mean_percent
        kudos_total = sample.kudos_this_session if sample is not None else snapshot.kudos_this_session
        eligible_seconds = sample.eligible_seconds_total if sample is not None else snapshot.eligible_seconds_total
        jobs_submitted = sample.jobs_submitted if sample is not None else snapshot.num_jobs_submitted
        forms_submitted = sample.alchemy_total_submitted if sample is not None else snapshot.alchemy_total_submitted
        if gpu_duty is not None:
            self._gpu_duty_history.append((now, gpu_duty))
        if kudos_total is not None:
            self._kudos_history.append((now, kudos_total, eligible_seconds))
        self._jobs_history.append((now, jobs_submitted))
        self._forms_history.append((now, forms_submitted))

    def _windowed_float_series(self, samples: deque[tuple[float, float]]) -> list[float]:
        """Return fixed buckets spanning the active trend window and epoch."""
        return fixed_float_buckets(
            list(samples),
            self._trend_window,
            session_start=self._trend_session_start,
            epoch=self._trend_epoch,
            buckets=_TREND_SPARK_WIDTH,
        )

    def _windowed_job_samples(self) -> list[tuple[float, int]]:
        """Return job-counter samples in the active trend window and epoch."""
        start, end, _configured = trend_bounds(
            self._trend_window,
            session_start=self._trend_session_start,
            epoch=self._trend_epoch,
        )
        return [(timestamp, count) for timestamp, count in self._jobs_history if start <= timestamp <= end]

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

    @staticmethod
    def _render_update_nag(info: UpdateInfo) -> Panel:
        """Render the update-available nag panel shown at the top of the overview."""
        line = Text.assemble(
            ("v", "grey70"),
            (current_version(), "bold"),
            (" -> ", "grey50"),
            (f"v{info.latest_version}", "bold yellow"),
            ("   Run ", "grey70"),
            ("'update.cmd'", "bold cyan"),
            (" / ", "grey50"),
            ("'update.sh'", "bold cyan"),
            (" to update, or re-run the installer.", "grey70"),
        )
        return Panel(line, title="Update available", title_align="left", border_style="yellow", padding=(0, 1))

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
            if self._is_alchemist_only(snapshot):
                body.append(self._alchemist_only_identity_line(snapshot))
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
    def _alchemist_only_identity_line(snapshot: WorkerStateSnapshot) -> Text:
        """A prominent banner naming this worker as alchemist-only and (when known) its alchemist name."""
        name = snapshot.config.alchemist_name
        line = Text.assemble(("⚗ ALCHEMIST-ONLY WORKER", "bold magenta"))
        if name:
            line.append("  ·  ", style="grey37")
            line.append(name, style="bold")
        return line

    @classmethod
    def _headline_metrics_line(cls, snapshot: WorkerStateSnapshot) -> Text:
        """The session totals the dropped stat cards used to carry: submitted, kudos/hr, faulted.

        An alchemist-only worker counts alchemy forms instead of image jobs, so its headline reads in the
        units it actually serves.
        """
        kudos = "-" if snapshot.kudos_per_hour is None else f"{snapshot.kudos_per_hour:,.0f}"
        if cls._is_alchemist_only(snapshot):
            faulted_colour = "red" if snapshot.alchemy_total_faulted else "grey70"
            return Text.assemble(
                (f"{snapshot.alchemy_total_submitted:,}", "bold"),
                (" forms submitted", "grey50"),
                ("   ·   ", "grey37"),
                (kudos, "bold cyan"),
                (" kudos/hr", "grey50"),
                ("   ·   ", "grey37"),
                (f"{snapshot.alchemy_total_faulted:,}", faulted_colour),
                (" faulted", "grey50"),
            )
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

    @classmethod
    def _activity_line(cls, snapshot: WorkerStateSnapshot) -> Text:
        """A heartbeat line conveying recent activity and freshness.

        Shows the in-flight counts of the work actually being served: alchemy forms for an alchemist-only
        worker, image jobs otherwise. A mixed (dreamer + alchemist) worker appends a live alchemy count so
        active alchemy is visible alongside image work rather than buried in a side panel.
        """
        age = time.time() - snapshot.timestamp if snapshot.timestamp else None
        since_pop = (
            human_duration(snapshot.seconds_since_last_pop) + " ago"
            if snapshot.seconds_since_last_pop is not None
            else "never"
        )
        if cls._is_alchemist_only(snapshot):
            return Text.assemble(
                ("updated ", "grey50"),
                (f"{human_duration(age)} ago", "grey70"),
                ("  ·  last pop ", "grey50"),
                (since_pop, "grey70"),
                ("  ·  forms in flight ", "grey50"),
                (str(snapshot.alchemy_forms_in_flight), "grey70"),
                ("  ·  pending ", "grey50"),
                (str(snapshot.alchemy_forms_pending), "grey70"),
            )
        line = Text.assemble(
            ("updated ", "grey50"),
            (f"{human_duration(age)} ago", "grey70"),
            ("  ·  last pop ", "grey50"),
            (since_pop, "grey70"),
            ("  ·  in progress ", "grey50"),
            (str(snapshot.jobs_in_progress), "grey70"),
            ("  ·  queued ", "grey50"),
            (str(snapshot.jobs_pending_inference), "grey70"),
        )
        alchemy_live = snapshot.alchemy_forms_in_flight + snapshot.alchemy_forms_pending
        if snapshot.config.alchemist and alchemy_live > 0:
            line.append("  ·  alchemy ", style="grey50")
            line.append(str(alchemy_live), style="grey70")
        return line

    @staticmethod
    def _memory_line(snapshot: WorkerStateSnapshot) -> Text | None:
        """A system-RAM line: whole-machine in-use/total and the worker's total share.

        The per-role worker breakdown lives on the Live tab, which already carries per-process RAM; the
        hero keeps only the overall figures so the status view stays scannable. Returns None when no memory
        sample has arrived yet (older worker, or before the first report), so the hero simply omits the line.
        """
        wire = snapshot.system_memory
        if wire is None or wire.total_bytes <= 0:
            return None

        summary = wire.to_summary()
        used_fraction = summary.used_fraction
        used_pct = f" ({used_fraction * 100:.0f}%)" if used_fraction is not None else ""

        return Text.assemble(
            ("RAM ", "grey50"),
            (f"{human_bytes(summary.used_bytes)} / {human_bytes(summary.total_bytes)}", "grey70"),
            (used_pct, "grey50"),
            ("  ·  worker ", "grey50"),
            (human_bytes(summary.worker_total_bytes), "bold cyan"),
        )

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
            f"; running {residency.processes_now}/{residency.processes_max} inference processes",
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

    @staticmethod
    def _preload_decision_label(decision: str) -> str:
        """Humanize a preload-admission decision key for compact dashboard text."""
        if not decision:
            return "no decision yet"
        return decision.replace("_", " ").title()

    @staticmethod
    def _scheduling_grid(governance: SchedulingGovernanceSnapshot) -> tuple[Table, bool]:
        """Build the scheduler-governance diagnostics grid, plus whether anything is actively holding.

        Returns the grid and an ``active`` flag (RAM pressure, a pop hold, or a non-admitting preload
        decision) so a caller can colour the surrounding panel border.
        """
        ram = governance.ram
        preload = governance.preload
        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="right", style="bold cyan", no_wrap=True)
        grid.add_column(ratio=1)

        if not ram.measured:
            ram_text = Text("waiting for first governor tick", style="grey50")
        else:
            ram_style = "red" if ram.under_pressure else "yellow" if ram.pop_hold_active else "green"
            ram_text = Text(ram.reason or "RAM governor measured", style=ram_style)
        grid.add_row("RAM", ram_text)

        intake_parts: list[tuple[str, str]] = []
        intake_parts.append(
            ("pop hold on" if ram.pop_hold_active else "pop hold off", "yellow" if ram.pop_hold_active else "grey62")
        )
        if ram.pop_pause_active:
            pause = human_duration(ram.pop_pause_remaining_seconds)
            intake_parts.append((f"hard pause {pause}", "yellow"))
        grid.add_row("Intake", Text.assemble(*_join_styled(intake_parts, " · ")))

        reclaim_bits: list[str] = []
        if ram.draining_process_ids:
            reclaim_bits.append("draining p" + ", p".join(str(pid) for pid in ram.draining_process_ids))
        if ram.shed_card_indices:
            reclaim_bits.append("shed GPU " + ", ".join(str(index) for index in ram.shed_card_indices))
        grid.add_row("Reclaim", Text("; ".join(reclaim_bits) if reclaim_bits else "none active", style="grey62"))

        if ram.measured:
            restore = (
                f"{human_mb(ram.restore_headroom_mb)} headroom; "
                f"{human_mb(ram.per_context_ram_estimate_mb)} per context"
            )
            if ram.per_process_ceiling_mb is not None:
                restore += f"; {human_mb(ram.per_process_ceiling_mb)} process ceiling"
            grid.add_row("Restore", Text(restore, style="grey62"))

        decision = OverviewView._preload_decision_label(preload.decision)
        target = f"p{preload.process_id}" if preload.process_id is not None else "no target"
        model = shorten(preload.model, 34) if preload.model else "-"
        age = f" · {human_duration(time.time() - preload.timestamp)} ago" if preload.timestamp else ""
        preload_style = _preload_decision_style(preload.decision)
        grid.add_row("Preload", Text.assemble((decision, preload_style), (f" · {model} · {target}{age}", "grey70")))
        if preload.reason:
            grid.add_row("Gate", Text(preload.reason, style="grey62"))

        active = ram.under_pressure or ram.pop_hold_active or preload_style == "yellow"
        return grid, active

    @staticmethod
    def _render_governance_panel(governance: SchedulingGovernanceSnapshot) -> Panel:
        """Render the scheduler governance diagnostics as a standalone panel."""
        grid, active = OverviewView._scheduling_grid(governance)
        border = "yellow" if active else "grey37"
        return Panel(grid, title="Scheduling governance", title_align="left", border_style=border, padding=(0, 1))

    @staticmethod
    def _governors_grid(governors: PopGovernorsSnapshot, *, detailed: bool) -> Table:
        """Build the pop-governor grid: active governors first, then a dim session history in details mode.

        Active governors lead, each with its reason and either a live countdown (timed windows: a residency
        cooldown, the switch/re-entry windows, the consecutive-failure or self-throttle pause, the
        megapixelstep wait) or the elapsed spell duration (condition-based gates). In details mode, governors
        that have engaged earlier this session are appended dim with their trigger count and total time, so an
        operator can see how much each one has cost even once it has released.
        """
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=2)  # status dot
        grid.add_column(style="bold", no_wrap=True)  # label
        grid.add_column()  # reason / timing

        active = [g for g in governors.governors if g.active]
        history = [g for g in governors.governors if not g.active and g.triggers > 0]

        for governor in active:
            if governor.expected_remaining_seconds is not None:
                timing = Text(f"~{human_duration(governor.expected_remaining_seconds)} left", style="yellow")
            else:
                timing = Text(f"{human_duration(governor.current_spell_seconds)} so far", style="yellow")
            reason = f" {governor.reason}" if governor.reason else ""
            grid.add_row(Text("●", style="yellow"), governor.label, Text.assemble(reason.strip(), "  ", timing))

        if not active:
            grid.add_row(Text("·", style="grey50"), Text("none engaged", style="grey50"), "")

        if detailed and history:
            for governor in history:
                summary = f"{governor.triggers}x, {human_duration(governor.total_active_seconds)} total"
                pct = (
                    f" ({governor.fraction_of_session * 100:.0f}% of session)" if governor.fraction_of_session else ""
                )
                grid.add_row(
                    Text("○", style="grey50"),
                    Text(governor.label, style="grey50"),
                    Text(summary + pct, style="grey50"),
                )

        return grid

    @staticmethod
    def _render_governors_panel(governors: PopGovernorsSnapshot, *, detailed: bool) -> Panel:
        """Render the pop governors currently holding back or reshaping job pops as a standalone panel."""
        border = "yellow" if governors.any_active else "grey37"
        return Panel(
            OverviewView._governors_grid(governors, detailed=detailed),
            title="Pop governors",
            title_align="left",
            border_style=border,
            padding=(0, 1),
            expand=False,
        )

    @staticmethod
    def _render_governance_combined(
        governors: PopGovernorsSnapshot | None,
        scheduling: SchedulingGovernanceSnapshot | None,
        *,
        detailed: bool,
    ) -> Panel:
        """Fold the pop governors and scheduler governance into one panel with labelled sub-sections.

        Either sub-section is omitted when its caller passes None, so the panel shows only what is relevant:
        an active-governor summary in normal mode, both sub-sections in details mode. The active-governor
        count rides the title so an operator sees at a glance how many holds are engaged without expanding.
        """
        parts: list[RenderableType] = []
        active_governors = 0
        border = "grey37"

        if governors is not None:
            active_governors = sum(1 for governor in governors.governors if governor.active)
            if governors.any_active:
                border = "yellow"
            parts.append(Text("Pop governors", style="bold grey70"))
            parts.append(OverviewView._governors_grid(governors, detailed=detailed))

        if scheduling is not None:
            grid, sched_active = OverviewView._scheduling_grid(scheduling)
            if sched_active:
                border = "yellow"
            if parts:
                parts.append(Text(""))
            parts.append(Text("Scheduling", style="bold grey70"))
            parts.append(grid)

        title = "Governance" if not active_governors else f"Governance · {active_governors} active"
        return Panel(Group(*parts), title=title, title_align="left", border_style=border, padding=(0, 1))

    def _render_health(
        self,
        report: HealthReport,
        feature_readiness: FeatureReadinessSummary | None = None,
        *,
        models_loaded: int | None = None,
    ) -> Panel:
        """Render the health checklist, with a compact feature-readiness line when any feature is engaged.

        The loaded-model count rides the panel title as an at-a-glance figure rather than a checklist row,
        since it is a plain count with no pass/warn/fail character of its own.
        """
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
        title = "Health"
        if models_loaded is not None:
            title += f" · {models_loaded} models" if models_loaded else " · no models"
        return Panel(table, title=title, title_align="left", border_style="grey37", padding=(0, 1))

    _COMPACT_READINESS_STYLE: dict[FeatureReadinessState, str] = {
        FeatureReadinessState.OFFERED: "green",
        FeatureReadinessState.WAITING: "yellow",
        FeatureReadinessState.MISSING_DEPS: "red",
        FeatureReadinessState.FAILED: "red",
        FeatureReadinessState.DISABLED: "grey50",
    }

    _COMPACT_READINESS_VERB: dict[FeatureReadinessState, str] = {
        FeatureReadinessState.OFFERED: "offered",
        FeatureReadinessState.WAITING: "downloading",
        FeatureReadinessState.MISSING_DEPS: "no deps",
        FeatureReadinessState.FAILED: "failed",
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

        if self._is_alchemist_only(snapshot):
            table.add_row("Alchemist", config.alchemist_name or "-", "Version", f"v{config.worker_version}")
        else:
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

        # Context-sensitive: call out a CPU / alchemist-only install (image generation disabled). A GPU
        # install is the default and adds no row, keeping the usual dashboard unchanged.
        from horde_worker_regen.compute_mode import compute_mode_display_label

        compute_label = compute_mode_display_label()
        if compute_label is not None:
            table.add_row("Compute", compute_label, "", "")
        return table

    @staticmethod
    def _enabled_workloads(snapshot: WorkerStateSnapshot) -> frozenset[WorkloadKind]:
        """Reconstruct the typed served-workload set from the snapshot's string values.

        Unknown values (a workload a newer worker serves that this dashboard build predates) are skipped
        rather than raising, so a forward-version worker degrades to "the workloads we recognise".
        """
        result: set[WorkloadKind] = set()
        for value in snapshot.enabled_workloads:
            try:
                result.add(WorkloadKind(value))
            except ValueError:
                continue
        return frozenset(result)

    @classmethod
    def _primary_workload(cls, snapshot: WorkerStateSnapshot) -> WorkloadKind | None:
        """The workload the dashboard foregrounds.

        Image generation when served (the usual dreamer/mixed worker, whose existing layout is
        unchanged); otherwise the sole remaining workload (an alchemist-only worker foregrounds alchemy);
        None when nothing is served or the worker predates the served-workload signal.
        """
        workloads = cls._enabled_workloads(snapshot)
        if not workloads:
            return None
        if WorkloadKind.IMAGE_GENERATION in workloads:
            return WorkloadKind.IMAGE_GENERATION
        return next(iter(sorted(workloads, key=lambda workload: workload.value)))

    @classmethod
    def _is_alchemist_only(cls, snapshot: WorkerStateSnapshot) -> bool:
        """Whether this worker serves alchemy and nothing else (the alchemist-only reshape trigger)."""
        return cls._enabled_workloads(snapshot) == frozenset({WorkloadKind.ALCHEMY})

    @staticmethod
    def _show_alchemy_panel(snapshot: WorkerStateSnapshot) -> bool:
        """Return whether the Overview should show the alchemy panel outside thin mode."""
        return snapshot.config.alchemist or (
            snapshot.alchemy_forms_pending + snapshot.alchemy_forms_in_flight + snapshot.alchemy_forms_awaiting_submit
            > 0
        )

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
            if snapshot.post_processing_disabled:
                flags.append("post OFF")
            else:
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

    @staticmethod
    def _render_intent(snapshot: WorkerStateSnapshot, *, detailed: bool) -> Panel:
        """Render the Now / Next / Why orchestration strip."""
        intent = snapshot.orchestration_intent
        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="right", style="bold cyan", no_wrap=True)
        grid.add_column()
        grid.add_row("Now", Text(intent.summary, style="bold"))
        if intent.next_action:
            grid.add_row("Next", Text(intent.next_action, style="grey70"))
        hide_duplicate_why = bool(
            detailed and intent.why and intent.raw_gate and intent.why.strip() == intent.raw_gate.strip()
        )
        if intent.why and not hide_duplicate_why:
            grid.add_row("Why", Text(intent.why, style="yellow" if "blocked" in intent.why.lower() else "grey70"))
        if detailed and intent.raw_gate:
            grid.add_row("Gate", Text(intent.raw_gate, style="grey62"))
        target_parts = []
        if intent.target_job_id:
            target_parts.append(f"job {intent.target_job_id[:8]}")
        if intent.target_model:
            target_parts.append(shorten(intent.target_model, 32))
        if intent.target_process_id is not None:
            target_parts.append(f"proc {intent.target_process_id}")
        if detailed and intent.target_device_index is not None:
            target_parts.append(f"gpu {intent.target_device_index}")
        if target_parts:
            grid.add_row("Target", Text(" · ".join(target_parts), style="grey62"))
        return Panel(grid, title="Now / Next / Why", title_align="left", border_style="cyan", padding=(0, 1))

    @staticmethod
    def _work_stage_cell(entry: WorkLedgerEntry) -> Text:
        """Render a work-ledger stage with stable color semantics."""
        style = {
            WorkLedgerStage.QUEUED: "cyan",
            WorkLedgerStage.PREPARING: "yellow",
            WorkLedgerStage.INFERENCE: "green",
            WorkLedgerStage.POST_PROCESSING: "green3",
            WorkLedgerStage.SAFETY: "magenta",
            WorkLedgerStage.SUBMIT: "blue",
            WorkLedgerStage.COMPLETED: "grey70",
            WorkLedgerStage.FAULTED: "red",
        }.get(entry.stage, "grey62")
        return Text(entry.stage.value, style=style)

    @staticmethod
    def _work_model_cell(entry: WorkLedgerEntry) -> Text:
        """Render model plus baseline in one compact job-owned cell."""
        model = shorten(entry.model, 24) if entry.model else "-"
        baseline = short_baseline(entry.baseline)
        if baseline == "-":
            return Text(model)
        return Text.assemble((model, ""), (f" · {baseline}", "grey50"))

    @staticmethod
    def _work_progress_cell(entry: WorkLedgerEntry) -> Text:
        """Render active job progress, or timing for completed work."""
        if entry.progress_total:
            current = entry.progress_current or 0
            fraction = current / entry.progress_total
            return Text.assemble((mini_bar(fraction, 8), "green"), (f" {current}/{entry.progress_total}", "grey62"))
        if entry.e2e_seconds is not None:
            return Text(human_duration(entry.e2e_seconds), style="grey62")
        return Text("-", style="grey50")

    @staticmethod
    def _work_size_cell(entry: WorkLedgerEntry) -> str:
        """Render a job's resolution and steps."""
        size = f"{entry.width}×{entry.height}" if entry.width and entry.height else "-"
        if entry.steps:
            return f"{size} · {entry.steps}s"
        return size

    @staticmethod
    def _work_age_cell(entry: WorkLedgerEntry) -> str:
        """Render the active stage age or the recent job's end-to-end time."""
        if entry.age_seconds is not None:
            return human_duration(entry.age_seconds)
        if entry.e2e_seconds is not None:
            return human_duration(entry.e2e_seconds)
        return "-"

    @staticmethod
    def _work_features_cell(entry: WorkLedgerEntry) -> str:
        """Render compact feature tags for a work-ledger entry."""
        return ", ".join(entry.features.as_tags()) if entry.features is not None else "-"

    def _render_work_ledger(
        self,
        snapshot: WorkerStateSnapshot,
        *,
        detailed: bool,
        available_width: int | None = None,
        show_recent_jobs: bool = True,
    ) -> Panel:
        """Render active and, optionally, recent job-owned state separately from process-owned state."""
        layout = select_columns(
            _WORK_LEDGER_COLUMNS,
            ceiling=intent_ceiling(detailed),
            available_width=available_width,
        )
        table = Table(title="", expand=True, border_style="grey37", header_style="bold", show_header=True)
        add_columns(table, layout.columns)
        recent_stages = {WorkLedgerStage.COMPLETED, WorkLedgerStage.FAULTED}
        recent_entries = [entry for entry in snapshot.work_ledger if entry.stage in recent_stages]
        visible_entries = (
            snapshot.work_ledger
            if show_recent_jobs
            else [entry for entry in snapshot.work_ledger if entry.stage not in recent_stages]
        )
        if not visible_entries:
            placeholder = "no active or recent work" if show_recent_jobs else "no active work"
            table.add_row(*placeholder_row(layout.columns, "Stage", placeholder))
        else:
            for entry in visible_entries:
                table.add_row(*[spec.render(entry) for spec in layout.columns])
        subtitle = shed_hint(layout)
        body = table
        completed = sum(1 for entry in recent_entries if not entry.faulted)
        faulted = len(recent_entries) - completed
        if not show_recent_jobs and recent_entries:
            summary = f"{completed} job{'s' if completed != 1 else ''} completed recently"
            if faulted:
                summary += f"; {faulted} faulted"
            body = Group(table, Text(f"({summary})", style="grey62"))
        # Surface the active/finished tally in the title so the ledger's scale reads without scanning rows.
        active_count = sum(1 for entry in snapshot.work_ledger if entry.stage not in recent_stages)
        title = "Work ledger"
        if active_count or recent_entries:
            title = f"Work ledger · {active_count} active · {completed + faulted} done"
        return Panel(
            body,
            title=title,
            title_align="left",
            subtitle=Text(subtitle, style="grey50") if subtitle else None,
            subtitle_align="right",
            border_style="green" if visible_entries else "grey37",
            padding=(0, 1),
        )

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
        alchemist_only = self._is_alchemist_only(snapshot)

        queue = snapshot.jobs_pending_inference
        inference = snapshot.jobs_in_progress
        post_processing = snapshot.jobs_pending_post_processing + snapshot.jobs_being_post_processed
        safety = snapshot.jobs_pending_safety_check + snapshot.jobs_being_safety_checked
        submit = snapshot.jobs_pending_submit
        peak = max(queue, inference, post_processing, safety, submit, 1)

        arrow = Text(" ▶ ", style="grey50")
        rows: list[Text] = []

        # An alchemist-only worker pops no image jobs, so its image lifecycle row is permanently empty;
        # the alchemy flow becomes the primary (and only) pipeline content instead.
        if not alchemist_only:
            if snapshot.config.allow_post_processing or post_processing:
                rows.append(
                    Text.assemble(
                        self._stage_segment("Queue", queue, peak),
                        arrow,
                        self._stage_segment("Inference", inference, peak),
                        arrow,
                        self._stage_segment("Post-proc", post_processing, peak),
                        arrow,
                        self._stage_segment("Safety", safety, peak),
                        arrow,
                        self._stage_segment("Submit", submit, peak),
                        ("    ", ""),
                        (f"✓ {snapshot.num_jobs_submitted:,} submitted", "grey62"),
                    )
                )
            else:
                rows.append(
                    Text.assemble(
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
                )

        alchemy_active = (
            snapshot.alchemy_forms_pending + snapshot.alchemy_forms_in_flight + snapshot.alchemy_forms_awaiting_submit
        )
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

        if not rows:
            rows.append(Text("idle", style="grey50"))

        if alchemist_only:
            title = "Alchemy pipeline"
            border = "green" if alchemy_active else "grey37"
        else:
            title = "Job pipeline"
            border = "green" if (queue or inference or post_processing or safety or submit) else "grey37"
        return Panel(Group(*rows), title=title, title_align="left", border_style=border, padding=(0, 1))

    @staticmethod
    def _trend_arrow(series: list[float]) -> Text:
        """A direction marker from comparing averaged early and late non-empty segments.

        Bucketed series often contain zeroes where no sample fell in a time slice; those are *absent
        data*, not genuine zeroes, so the arrow skips them.  The series is split into halves and each
        half must contain at least two positive values, otherwise the arrow declines to assert a trend
        the data cannot support.
        """
        if len(series) < 4:
            return Text("→", style="grey50")
        mid = len(series) // 2
        head = [v for v in series[:mid] if v > 0]
        tail = [v for v in series[mid:] if v > 0]
        if len(head) < 2 or len(tail) < 2:
            return Text("→", style="grey50")
        head_avg = sum(head) / len(head)
        tail_avg = sum(tail) / len(tail)
        if head_avg < 0.01:
            return Text("→", style="grey50")
        change = (tail_avg - head_avg) / head_avg
        if change > 0.05:
            return Text(f"▲ {abs(change) * 100:.0f}%", style="green")
        if change < -0.05:
            return Text(f"▼ {abs(change) * 100:.0f}%", style="red")
        return Text("→", style="grey50")

    def _kudos_per_hour(self) -> tuple[float | None, list[float]]:
        """Derive windowed kudos/hr and per-bucket kudos deltas over productive seconds in the window.

        Unlike a sampled session average, this is a stable windowed rate (the kudos/hr analogue of jobs/hr):
        it divides the kudos earned across the window by the productive seconds earned across it, so it does
        not sawtooth on each submit and does not charge idle/maintenance time inside the window.
        """
        rate, deltas, _sampled_span = fixed_ratio_deltas(
            list(self._kudos_history),
            self._trend_window,
            session_start=self._trend_session_start,
            epoch=self._trend_epoch,
            buckets=_TREND_SPARK_WIDTH,
        )
        return rate, deltas

    def _jobs_per_hour(self) -> tuple[float | None, list[float]]:
        """Derive jobs/hr and fixed-window completion buckets from the cumulative job counter."""
        rate, deltas, _sampled_span = fixed_counter_deltas(
            list(self._jobs_history),
            self._trend_window,
            session_start=self._trend_session_start,
            epoch=self._trend_epoch,
            buckets=_TREND_SPARK_WIDTH,
        )
        return rate, deltas

    def _forms_per_hour(self) -> tuple[float | None, list[float]]:
        """Derive alchemy forms/hr and fixed-window buckets from the cumulative submitted-forms counter."""
        rate, deltas, _sampled_span = fixed_counter_deltas(
            list(self._forms_history),
            self._trend_window,
            session_start=self._trend_session_start,
            epoch=self._trend_epoch,
            buckets=_TREND_SPARK_WIDTH,
        )
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
        # Card count and mean duty ride the title so utilization reads at a glance, and a multi-GPU host is
        # obvious even before the rows are scanned.
        card_count = len(snapshot.per_card)
        title = f"GPUs · {card_count} card{'s' if card_count != 1 else ''}"
        if snapshot.gpu_utilization_mean_percent is not None:
            title += f" · {format_percent(snapshot.gpu_utilization_mean_percent)} duty"
        return Panel(grid, title=title, title_align="left", border_style="grey37", padding=(0, 1))

    def _render_trends(self, snapshot: WorkerStateSnapshot) -> Panel:
        """Render recent kudos/hr, jobs/hr, and GPU-duty trends: a value, direction, and sparkline.

        Replaces the old momentum gauge, whose self-scaled sparklines carried neither a reference
        value nor a window. Here each row pairs a current figure with a direction marker against the
        window start, and the GPU row adds a duty bar so "how much of the time it is working" reads
        at a glance alongside the over-time shape.
        """
        kudos_rate, kudos_deltas = self._kudos_per_hour()
        kudos_deltas = kudos_deltas[-_TREND_SPARK_WIDTH:]
        gpu_series = self._windowed_float_series(self._gpu_duty_history)[-_TREND_SPARK_WIDTH:]
        rate, jobs_deltas = self._jobs_per_hour()
        jobs_deltas = jobs_deltas[-_TREND_SPARK_WIDTH:]

        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="right", style="bold cyan", no_wrap=True)
        grid.add_column(justify="right", no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(style="grey50", no_wrap=True)

        kudos_now = "-" if kudos_rate is None else f"{kudos_rate:,.0f}"
        kudos_total = snapshot.kudos_this_session
        kudos_tail = f"{kudos_total:,.0f} kudos" if kudos_total else ""
        grid.add_row(
            "Kudos/hr",
            kudos_now,
            self._trend_arrow(kudos_deltas),
            Text(sparkline(kudos_deltas) or "…", style="cyan"),
            kudos_tail,
        )

        jobs_now = "-" if rate is None else f"{rate:,.0f}"
        grid.add_row(
            "Jobs/hr",
            jobs_now,
            self._trend_arrow(jobs_deltas),
            Text(sparkline(jobs_deltas) or "…", style="green"),
            f"{snapshot.num_jobs_submitted:,} done",
        )

        # An alchemist worker gets its own throughput row, the forms/hr analogue of jobs/hr.
        if snapshot.config.alchemist:
            forms_rate, forms_deltas = self._forms_per_hour()
            forms_deltas = forms_deltas[-_TREND_SPARK_WIDTH:]
            grid.add_row(
                "Forms/hr",
                "-" if forms_rate is None else f"{forms_rate:,.0f}",
                self._trend_arrow(forms_deltas),
                Text(sparkline(forms_deltas) or "…", style="magenta"),
                f"{snapshot.alchemy_total_submitted:,} done",
            )

        busy_fraction = snapshot.gpu_utilization_busy_fraction
        if busy_fraction is None and snapshot.gpu_utilization_mean_percent is not None:
            busy_fraction = snapshot.gpu_utilization_mean_percent / 100.0
        duty_bar = Text(mini_bar(busy_fraction, 12), style="green") if busy_fraction is not None else Text("…")
        # A near-idle GPU while a job is in flight is the one duty-cycle condition worth an alert. Rather
        # than a separate health row, it is flagged here where the duty figure already lives: the row and
        # the panel border turn orange and a "(!)" rides the value.
        duty_low = is_gpu_duty_low(snapshot)
        duty_style = "dark_orange" if duty_low else "bold cyan"
        duty_value: RenderableType = format_percent(snapshot.gpu_utilization_mean_percent)
        if duty_low:
            duty_value = Text.assemble(
                (format_percent(snapshot.gpu_utilization_mean_percent), "dark_orange"),
                (" (!)", "bold dark_orange"),
            )
        grid.add_row(
            Text("GPU duty", style=duty_style),
            duty_value,
            duty_bar,
            Text(sparkline(gpu_series) or "…", style="green"),
            "busy" if busy_fraction and busy_fraction > 0.5 else "idle",
        )

        window = self._trend_window_label()

        # Surface a transient notice (soft-reset, config change) and clear it once
        # enough data has accumulated that the trend arrows are informative again.
        notice: Text | None = None
        if self._trend_notice is not None:
            job_samples = self._windowed_job_samples()
            if len(job_samples) >= 4:
                self._trend_notice = None
            else:
                notice = Text(self._trend_notice, style="italic yellow")

        duty_warning = (
            Text("GPU near-idle while a job is running: check the loaded model and the logs.", style="dark_orange")
            if duty_low
            else None
        )

        parts: list[RenderableType] = [part for part in (notice, duty_warning, grid) if part is not None]
        body = Group(*parts)

        return Panel(
            body,
            title="Trends",
            title_align="left",
            subtitle=Text(window, style="grey50"),
            subtitle_align="right",
            border_style="dark_orange" if duty_low else "grey37",
            padding=(0, 1),
        )

    def _trend_window_label(self) -> str:
        """Describe the configured and actual span the trend buffers currently cover."""
        samples = self._windowed_job_samples()
        label = "All" if self._trend_window is OverviewTrendWindow.ALL else self._trend_window.value
        if len(samples) < 2:
            return f"{label} window · warming up"
        start, end, configured = trend_bounds(
            self._trend_window,
            session_start=self._trend_session_start,
            epoch=self._trend_epoch,
        )
        span = configured if configured is not None else max(end - start, 0.0)
        return f"{label} window · {human_duration(samples[-1][0] - samples[0][0])} sampled of {human_duration(span)}"

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
        line.append_text(self._trend_arrow(self._kudos_per_hour()[1][-_TREND_SPARK_WIDTH:]))
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
        # Fold the alive/hot tally into the title so process health reads at a glance without counting rows.
        total_procs = len(snapshot.processes)
        alive = sum(1 for process in snapshot.processes if process.is_alive)
        hot = sum(1 for process in snapshot.processes if process.is_busy)
        title = "Processes"
        if total_procs:
            title = f"Processes · {alive}/{total_procs} alive"
            if hot:
                title += f" · {hot} hot"
        table = Table(
            title=title,
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
        if state in ("INFERENCE_STARTING", "POST_PROCESSING", "ALCHEMY_STARTING"):
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
        # The pending count and its total megapixelstep cost ride the title, so queue depth and weight read
        # at a glance without expanding every row.
        pending = len(snapshot.pending_jobs)
        title = "Queue"
        if pending:
            title = f"Queue · {pending} pending"
            if snapshot.pending_megapixelsteps:
                title += f" · {snapshot.pending_megapixelsteps:,.0f} MP"
        return Panel(
            body,
            title=title,
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

    @classmethod
    def _render_recent_jobs(
        cls,
        snapshot: WorkerStateSnapshot,
        *,
        available_width: int | None = None,
    ) -> Panel:
        """Render a table of recently completed jobs, newest first, shedding to fit the width.

        A dreamer/mixed worker shows the most recent few (image work is frequent, so a short list stays
        current). An alchemist-only worker shows the full retained set: alchemy work is sparse, so a
        time-windowed "recent" view would often look empty, and an operator wants to see that work has
        been happening over the session.
        """
        layout = select_columns(_RECENT_COLUMNS, ceiling=DensityTier.WIDE, available_width=available_width)
        table = Table(
            title="",
            expand=True,
            border_style="grey37",
            header_style="bold",
            show_header=True,
        )
        add_columns(table, layout.columns)

        row_limit = RECENT_JOBS_IN_SNAPSHOT if cls._is_alchemist_only(snapshot) else 8
        recent = list(reversed(snapshot.recent_jobs[-row_limit:]))
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
        """The Model/type cell: the alchemy form (when known) for alchemy jobs, else the model name."""
        if job.is_alchemy:
            if job.model_name:
                return Text(f"⚗ {shorten(job.model_name, 22)}", style="grey62")
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


_WORK_LEDGER_COLUMNS: list[ColumnSpec[WorkLedgerEntry]] = [
    ColumnSpec("Stage", DensityTier.ESSENTIAL, OverviewView._work_stage_cell, width=9, no_wrap=True),
    ColumnSpec("Job", DensityTier.ESSENTIAL, lambda e: job_id_text(e.job_id), width=8, no_wrap=True),
    ColumnSpec("Order", DensityTier.NORMAL, lambda e: _queue_order_cell(e.queue_order), width=6, no_wrap=True),
    ColumnSpec("Model", DensityTier.ESSENTIAL, OverviewView._work_model_cell, min_width=18, no_wrap=True),
    ColumnSpec("Progress", DensityTier.ESSENTIAL, OverviewView._work_progress_cell, width=12, no_wrap=True),
    ColumnSpec("Intent", DensityTier.NORMAL, lambda e: shorten(e.intent, 28) if e.intent else "-", min_width=16),
    ColumnSpec(
        "Proc/GPU",
        DensityTier.NORMAL,
        lambda e: (
            "-"
            if e.process_id is None
            else f"{e.process_id}" + (f"/g{e.device_index}" if e.device_index is not None else "")
        ),
        width=8,
        no_wrap=True,
    ),
    ColumnSpec("Size", DensityTier.WIDE, OverviewView._work_size_cell, width=15, no_wrap=True),
    ColumnSpec("it/s", DensityTier.WIDE, lambda e: format_its(e.iterations_per_second), justify="right", width=6),
    ColumnSpec("Age", DensityTier.WIDE, OverviewView._work_age_cell, justify="right", width=8),
    ColumnSpec("Features", DensityTier.DETAILS, OverviewView._work_features_cell, min_width=10, no_wrap=True),
    ColumnSpec(
        "Reason",
        DensityTier.DETAILS,
        lambda e: shorten(e.raw_reason, 32) if e.raw_reason else "-",
        min_width=16,
    ),
]
"""The work-ledger columns, tagged by the density tier at which each appears."""

_PROCESS_COLUMNS: list[ColumnSpec[_ProcessRow]] = [
    ColumnSpec("ID", DensityTier.ESSENTIAL, lambda r: str(r.process.process_id), justify="right", width=3),
    ColumnSpec("Type", DensityTier.ESSENTIAL, lambda r: _process_type_label(r.process.process_type), width=9),
    ColumnSpec("State", DensityTier.ESSENTIAL, OverviewView._process_state_cell, width=18, no_wrap=True),
    ColumnSpec("GPU", DensityTier.NORMAL, lambda r: str(r.process.device_index), justify="right", width=4),
    ColumnSpec(
        "Resident model",
        DensityTier.NORMAL,
        lambda r: shorten(r.process.loaded_horde_model_name, 24),
        min_width=16,
        max_width=24,
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
        "GPU VRAM",
        DensityTier.WIDE,
        lambda r: OverviewView._vram_cell(r.process),
        justify="right",
        min_width=15,
        no_wrap=True,
    ),
    ColumnSpec(
        "RAM peak",
        DensityTier.WIDE,
        lambda r: human_mb(r.process.ram_used_high_water_mb) if r.process.ram_used_high_water_mb else "-",
        justify="right",
        width=9,
        no_wrap=True,
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


def _process_type_label(process_type: str) -> str:
    """Humanize process type enum names for compact tables."""
    return {
        "INFERENCE": "Inference",
        "SAFETY": "Safety",
        "POST_PROCESS": "Post-proc",
        "DOWNLOAD": "Download",
    }.get(process_type, process_type.replace("_", " ").title())


def _queue_order_cell(order: int | None) -> str:
    """Render a 1-based queue/pop order, or a dash for rows outside the image queue."""
    return f"#{order}" if order is not None else "-"


def _entry_features(entry: JobQueueEntry) -> str:
    """Comma-joined feature tags for a queued job, or a dash when it carries none."""
    return ", ".join(entry.features.as_tags()) if entry.features is not None else "-"


def _entry_size(entry: JobQueueEntry) -> str:
    """A queued job's ``width×height``, or a dash when its dimensions are unknown."""
    return f"{entry.width}×{entry.height}" if entry.width and entry.height else "-"


_QUEUE_COLUMNS: list[ColumnSpec[JobQueueEntry]] = [
    ColumnSpec("Job", DensityTier.ESSENTIAL, lambda e: job_id_text(e.job_id), width=8, no_wrap=True),
    ColumnSpec("Order", DensityTier.ESSENTIAL, lambda e: _queue_order_cell(e.queue_order), width=6, no_wrap=True),
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
