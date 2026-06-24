"""The Downloads screen: what is on disk, what the config implies, and live download progress.

Consumes the snapshot's ``download_plan`` (the config's disk implications, computed once and always
available when the model reference is loaded) and ``downloads`` (the live download-process status,
present only when background downloads are enabled). It answers, in plain language: which models are
already on disk, how much disk the configuration needs, whether the disk can hold it, and when/where/
why anything is downloading right now.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Static

from horde_worker_regen.app_state import OverviewViewMode
from horde_worker_regen.process_management.feature_readiness import FeatureReadinessState
from horde_worker_regen.process_management.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadPhase,
    DownloadPlanSummary,
    DownloadStatusSnapshot,
    FeatureReadinessSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.formatters import human_bytes, human_duration, shorten


def _host_label(item: CurrentDownloadStatus) -> str:
    """The source host for one download, for grouping/counting; a placeholder when it is not tracked."""
    return item.host or "?"

_PHASE_STYLE: dict[DownloadPhase, str] = {
    DownloadPhase.INITIALIZING: "yellow",
    DownloadPhase.SCANNING: "yellow",
    DownloadPhase.DOWNLOADING: "green",
    DownloadPhase.IDLE: "grey62",
    DownloadPhase.PAUSED: "yellow",
    DownloadPhase.ERROR: "red",
}

_PHASE_DETAIL: dict[DownloadPhase, str] = {
    DownloadPhase.INITIALIZING: "Fetching the model reference (a network call on first run)…",
    DownloadPhase.SCANNING: "Verifying which configured models are already on disk…",
    DownloadPhase.DOWNLOADING: "Downloading models in the background while the worker serves what it has.",
    DownloadPhase.IDLE: "All requested models are present; nothing queued.",
    DownloadPhase.PAUSED: "Downloads are paused; queued work is held until resumed.",
    DownloadPhase.ERROR: "The download subsystem hit an error (see below).",
}

_BAR_WIDTH = 32


@dataclass(frozen=True)
class DownloadActivity:
    """A compact, cross-view summary of live background-download progress.

    Lets headline surfaces (the Downloads tab label, the overview's slim line) show "what is fetching
    and how far along" without each reaching into the snapshot's download internals or re-deriving the
    ready/total fraction.
    """

    paused: bool
    """Whether the active download is currently held paused (vs actively transferring)."""
    ready: int | None
    """Configured models already on disk, or None when the count is not yet known."""
    total: int | None
    """Configured models in total (present + still to download), or None when not yet known."""
    current_name: str
    """The model whose file is downloading (or paused) right now."""
    percent: float | None
    """Completion of the current file (0-100), or None when its size is unknown."""
    speed_bps: float | None
    """Current transfer speed in bytes/sec, or None before a sample exists."""


def summarize_download_activity(snapshot: WorkerStateSnapshot | None) -> DownloadActivity | None:
    """Summarize the in-flight download, or None when nothing is downloading (or paused mid-download).

    Returns None unless the download process is actively working a file (phase DOWNLOADING or PAUSED with
    a current entry), so a caller can cheaply decide whether to surface a download indicator at all. The
    ready/total fraction reuses :meth:`DownloadsView._readiness` so the tab, overview and Downloads view
    can never disagree.
    """
    downloads = snapshot.downloads if snapshot is not None else None
    if downloads is None or downloads.current is None:
        return None
    if downloads.phase not in (DownloadPhase.DOWNLOADING, DownloadPhase.PAUSED):
        return None

    plan = snapshot.download_plan if snapshot is not None else None
    readiness = DownloadsView._readiness(plan, downloads) if plan is not None else None
    # Carry None (not a fabricated 0/0) when readiness is unknown, so a caller renders the activity
    # marker without a misleading count during the window before the plan is computed.
    ready, total = readiness if readiness is not None else (None, None)
    current = downloads.current
    return DownloadActivity(
        paused=downloads.paused,
        ready=ready,
        total=total,
        current_name=current.model_name,
        percent=current.percent,
        speed_bps=current.speed_bps,
    )


class DownloadsView(VerticalScroll):
    """A dashboard for model presence, the config's disk budget, and live download progress."""

    DEFAULT_CSS = """
    DownloadsView #downloads-controls {
        height: 3;
    }
    DownloadsView #downloads-controls Button {
        margin-right: 1;
    }
    DownloadsView #downloads-rate {
        width: 28;
        margin-right: 1;
    }
    """

    class PauseToggleRequested(Message):
        """Posted when the user toggles the download pause control."""

        def __init__(self, *, currently_paused: bool) -> None:
            """Carry whether downloads are currently paused, so the app picks pause vs resume."""
            super().__init__()
            self.currently_paused = currently_paused

    class RateLimitRequested(Message):
        """Posted when the user applies a download bandwidth cap (KB/s; 0 clears the cap)."""

        def __init__(self, kbps: int) -> None:
            """Carry the requested cap in KB/s."""
            super().__init__()
            self.kbps = kbps

    class DownloadsOnlyHoldRequested(Message):
        """Posted when the user asks to pre-fetch models without committing the GPU (download-only hold)."""

    class GoLiveRequested(Message):
        """Posted when the user asks to leave download-only mode and start serving jobs."""

    class DownloadPickerRequested(Message):
        """Posted when the user wants to choose which models to download (opens the picker modal)."""

    def __init__(self) -> None:
        """Track the last-seen paused state so the control row labels itself correctly."""
        super().__init__()
        self._paused = False

    def compose(self) -> ComposeResult:
        """Lay out the controls row, phase banner, disk-plan panel, current download, queue, failures."""
        with Horizontal(id="downloads-controls"):
            yield Button("Pause downloads", id="downloads-pause")
            yield Input(placeholder="rate limit KB/s (0 = off)", id="downloads-rate", type="integer")
            yield Button("Apply limit", id="downloads-rate-apply")
            yield Button("Download only", id="downloads-only-hold")
            yield Button("Choose models…", id="downloads-pick")
            yield Button("Go live", id="downloads-go-live")
        yield Static(id="downloads-banner")
        yield Static(id="downloads-plan")
        yield Static(id="downloads-readiness")
        yield Static(id="downloads-current")
        yield Static(id="downloads-queue")
        yield Static(id="downloads-failures")

    def update_view(
        self,
        snapshot: WorkerStateSnapshot | None,
        *,
        mode: OverviewViewMode = OverviewViewMode.NORMAL,
    ) -> None:
        """Refresh every panel from the latest snapshot (tolerant of missing download data).

        ``mode`` follows the shared F6 density contract. Thin collapses the page to "what is fetching
        right now plus whether it fits", hiding the queue and failure panels; normal and detailed both
        show the full plan, queue and failures (detailed never shows *less* than normal).
        """
        downloads = snapshot.downloads if snapshot is not None else None
        plan = snapshot.download_plan if snapshot is not None else None
        thin = mode is OverviewViewMode.THIN

        self.query_one("#downloads-queue", Static).display = not thin
        self.query_one("#downloads-failures", Static).display = not thin

        self._paused = downloads.paused if downloads is not None else False
        self.query_one("#downloads-pause", Button).label = "Resume downloads" if self._paused else "Pause downloads"
        self.query_one("#downloads-banner", Static).update(self._render_banner(downloads, plan))
        if thin:
            self.query_one("#downloads-plan", Static).update(self._render_plan_compact(plan))
        else:
            self.query_one("#downloads-plan", Static).update(self._render_plan(plan, downloads))
        readiness_widget = self.query_one("#downloads-readiness", Static)
        feature_readiness = snapshot.feature_readiness if snapshot is not None else None
        readiness_widget.display = not thin and feature_readiness is not None
        if feature_readiness is not None:
            readiness_widget.update(self._render_readiness(feature_readiness))
        self.query_one("#downloads-current", Static).update(self._render_current(downloads))
        self.query_one("#downloads-queue", Static).update(self._render_queue(downloads))
        self.query_one("#downloads-failures", Static).update(self._render_failures(downloads))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Translate the control buttons into messages the app forwards to the supervisor."""
        if event.button.id == "downloads-pause":
            self.post_message(self.PauseToggleRequested(currently_paused=self._paused))
        elif event.button.id == "downloads-rate-apply":
            self._post_rate_limit()
        elif event.button.id == "downloads-only-hold":
            self.post_message(self.DownloadsOnlyHoldRequested())
        elif event.button.id == "downloads-pick":
            self.post_message(self.DownloadPickerRequested())
        elif event.button.id == "downloads-go-live":
            self.post_message(self.GoLiveRequested())
        event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Apply the rate limit when Enter is pressed in the rate field."""
        if event.input.id == "downloads-rate":
            self._post_rate_limit()
            event.stop()

    def _post_rate_limit(self) -> None:
        """Parse the rate field and post a RateLimitRequested (ignoring invalid input)."""
        raw = self.query_one("#downloads-rate", Input).value.strip()
        try:
            kbps = max(int(raw), 0) if raw else 0
        except ValueError:
            return
        self.post_message(self.RateLimitRequested(kbps))

    def _render_banner(
        self,
        downloads: DownloadStatusSnapshot | None,
        plan: DownloadPlanSummary | None,
    ) -> Panel:
        """Render the headline phase banner, with the over-budget warning taking precedence."""
        if plan is not None and not plan.fits:
            title = Text("⚠  DISK OVER BUDGET", style="bold red")
            detail = Text(
                f"The configured models need {human_bytes(plan.to_download_bytes)} to download but only "
                f"{human_bytes(plan.free_disk_bytes)} is free (short by {human_bytes(plan.shortfall_bytes)}). "
                "Downloads will proceed until the disk fills, then stop with a disk-full error.",
                style="red",
            )
            return Panel(detail, title=title, title_align="left", border_style="red", padding=(0, 1))

        if downloads is None:
            detail = Text(
                "No download process is running. Models load at worker startup; this tab shows the "
                "configuration's disk plan below once the model reference is loaded.",
                style="grey70",
            )
            return Panel(detail, title="Downloads", title_align="left", border_style="grey37", padding=(0, 1))

        phase = downloads.phase
        colour = _PHASE_STYLE.get(phase, "grey62")
        title = Text.assemble((phase.value.upper(), f"bold {colour}"))
        lines: list[Text] = [Text(_PHASE_DETAIL.get(phase, ""), style="grey70")]
        if downloads.error_message:
            lines.append(Text(downloads.error_message, style="red"))
        control = self._control_line(downloads)
        if control is not None:
            lines.append(control)
        return Panel(Group(*lines), title=title, title_align="left", border_style=colour, padding=(0, 1))

    @staticmethod
    def _control_line(downloads: DownloadStatusSnapshot) -> Text | None:
        """Summarize the active pause state and bandwidth cap, when either is set."""
        parts: list[tuple[str, str]] = []
        if downloads.paused:
            parts.append(("paused", "yellow"))
        if downloads.rate_limit_kbps:
            parts.append((f"limit {downloads.rate_limit_kbps} KB/s", "grey70"))
        if not parts:
            return None
        text = Text("controls: ", style="grey50")
        for index, (label, style) in enumerate(parts):
            if index:
                text.append("  ·  ", style="grey50")
            text.append(label, style=style)
        return text

    @staticmethod
    def _readiness(plan: DownloadPlanSummary, downloads: DownloadStatusSnapshot | None) -> tuple[int, int] | None:
        """Live ``(ready, total)`` model count, or None when there is no download process to track.

        Single-sourced from the plan, which the worker recomputes live from the one on-disk presence
        authority (:mod:`horde_model_reference.on_disk_layout`): ``ready`` is exactly the number of
        configured models present on disk and ``total`` the configured count, so ``ready`` climbs as
        downloads land and is bounded to ``[num_present, total]`` by construction.

        The live download queue is deliberately NOT folded into this count. It derives presence by a
        different route (hordelib's sha256-gated availability), and mixing the two once let a queue that
        drifted larger than the plan's missing set drag an already-present worker's count toward 0 (the
        100/101 -> 0 report). The queue still drives the per-model downloading/queued/failed detail
        rendered elsewhere; it just no longer competes with the plan over the headline tally. ``downloads``
        is retained only as the gate for "is a download process even running?".
        """
        if downloads is None:
            return None
        total = plan.num_present + plan.num_to_download
        if total <= 0:
            return None
        return plan.num_present, total

    def _render_plan(self, plan: DownloadPlanSummary | None, downloads: DownloadStatusSnapshot | None) -> Panel:
        """Render the configuration's disk budget: present, to-download, total, free, and fit."""
        if plan is None:
            body: RenderableType = Text("Disk plan not available yet (model reference still loading).", style="grey50")
            return Panel(body, title="Disk plan", title_align="left", border_style="grey37", padding=(0, 1))

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_row("On disk", f"{plan.num_present} models · {human_bytes(plan.present_bytes)}")
        table.add_row("To download", f"{plan.num_to_download} models · {human_bytes(plan.to_download_bytes)}")
        readiness = self._readiness(plan, downloads)
        if readiness is not None:
            ready, total = readiness
            table.add_row(
                "Ready now",
                Text(f"{ready} of {total} models", style="green" if ready >= total else "yellow"),
            )
        table.add_row("Total", human_bytes(plan.total_bytes))
        table.add_row("Free disk", human_bytes(plan.free_disk_bytes))

        fit_text = (
            Text("fits on disk", style="green")
            if plan.fits
            else Text(f"OVER BUDGET by {human_bytes(plan.shortfall_bytes)}", style="bold red")
        )
        table.add_row("Status", fit_text)
        if not plan.sizes_complete:
            table.add_row(
                "Note",
                Text("some models lack size metadata; totals are a lower bound", style="yellow"),
            )

        return Panel(table, title="Disk plan", title_align="left", border_style="grey37", padding=(0, 1))

    def _render_plan_compact(self, plan: DownloadPlanSummary | None) -> Panel:
        """Render the disk plan as one dense line for the thin view: present, to-download, and fit."""
        if plan is None:
            body: RenderableType = Text("Disk plan not available yet (model reference still loading).", style="grey50")
            return Panel(body, title="Disk plan", title_align="left", border_style="grey37", padding=(0, 1))

        fit = (
            Text("fits on disk", style="green")
            if plan.fits
            else Text(f"OVER BUDGET by {human_bytes(plan.shortfall_bytes)}", style="bold red")
        )
        line = Text.assemble(
            (f"{plan.num_present}", "bold"),
            (" on disk", "grey50"),
            ("  ·  ", "grey37"),
            (f"{plan.num_to_download}", "bold"),
            (" to fetch ", "grey50"),
            (f"({human_bytes(plan.to_download_bytes)})", "grey62"),
            ("  ·  free ", "grey50"),
            (human_bytes(plan.free_disk_bytes), "grey62"),
            ("  ·  ", "grey37"),
        )
        line.append_text(fit)
        border = "green" if plan.fits else "red"
        return Panel(line, title="Disk plan", title_align="left", border_style=border, padding=(0, 1))

    _READINESS_STYLE: dict[FeatureReadinessState, tuple[str, str]] = {
        FeatureReadinessState.OFFERED: ("green", "offered"),
        FeatureReadinessState.WAITING: ("yellow", "downloading…"),
        FeatureReadinessState.MISSING_DEPS: ("red", "missing deps"),
        FeatureReadinessState.DISABLED: ("grey50", "off"),
    }

    def _render_readiness(self, summary: FeatureReadinessSummary) -> Panel:
        """Render which gated features the worker offers to the horde, and why anything is withheld.

        Each gated feature shows its offer state (offered / still downloading / missing deps / off) and a
        short reason; the informational rows below carry LoRA and safety, which keep their own gating. The
        table mirrors the worker's actual pop decision, so 'offered' here means the horde is being told the
        worker can serve that feature right now.
        """
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", no_wrap=True)
        table.add_column(no_wrap=True)
        table.add_column(style="grey62")
        # Details/status are wrapped in Text (not raw strings) so an install hint like
        # ``horde-worker-reGen[post-processing]`` is shown literally rather than parsed as console markup.
        for feature in summary.gated:
            style, label = self._READINESS_STYLE.get(feature.state, ("grey62", feature.state.value))
            table.add_row(feature.label, Text(label, style=style), Text(feature.detail))
        for info in summary.informational:
            table.add_row(
                Text(info.label, style="bold"),
                Text("ready" if info.ok else "waiting", style="green" if info.ok else "grey62"),
                Text(info.status),
            )
        return Panel(table, title="Feature readiness", title_align="left", border_style="grey37", padding=(0, 1))

    def _render_current(self, downloads: DownloadStatusSnapshot | None) -> Panel:
        """Render the in-progress downloads (one block per concurrent download) with bars, speed, and ETA.

        Several downloads can run at once (one per source host by default), so every entry in ``active`` is
        rendered. ``current`` is used as a single-entry fallback for older single-download snapshots.
        """
        active = list(downloads.active) if downloads is not None else []
        if not active and downloads is not None and downloads.current is not None:
            active = [downloads.current]
        if not active:
            body: RenderableType = Text("Nothing downloading right now.", style="grey50")
            return Panel(body, title="Downloading now", title_align="left", border_style="grey37", padding=(0, 1))

        rate_limit_kbps = downloads.rate_limit_kbps if downloads is not None else None
        renderables: list[RenderableType] = []
        for index, item in enumerate(active):
            if index:
                renderables.append(Text(""))
            renderables.extend(self._render_one_download(item, rate_limit_kbps))
        hosts = len({_host_label(item) for item in active})
        title = "Downloading now" if len(active) == 1 else f"Downloading now ({len(active)} across {hosts} hosts)"
        return Panel(Group(*renderables), title=title, title_align="left", border_style="green", padding=(0, 1))

    def _render_one_download(
        self,
        current: CurrentDownloadStatus,
        rate_limit_kbps: int | None,
    ) -> list[RenderableType]:
        """Render one download's header, target, and progress line (shared by the concurrent list)."""
        header = Text.assemble(
            (shorten(current.model_name, 40), "bold"),
            ("   ", ""),
            (current.feature, "cyan"),
        )
        where = Text.assemble(("→ ", "grey50"), (current.target_dir, "grey70"))

        bar = self._progress_bar(current.percent)
        sizes = f"{human_bytes(current.downloaded_bytes)} / {human_bytes(current.total_bytes)}"
        speed = f"{human_bytes(current.speed_bps)}/s" if current.speed_bps else "-"
        eta = human_duration(current.eta_seconds) if current.eta_seconds is not None else "-"
        progress = Text.assemble(
            (bar, "green"),
            ("  ", ""),
            (sizes, "grey70"),
            ("   ", ""),
            ("⇣ ", "grey50"),
            (speed, "grey70"),
            ("   ", ""),
            ("ETA ", "grey50"),
            (eta, "grey70"),
        )
        renderables: list[RenderableType] = [header, where, progress]
        if rate_limit_kbps and current.speed_bps is not None and current.speed_bps > rate_limit_kbps * 1024:
            renderables.append(
                Text(
                    "Speed shown above the set limit: the rolling average takes a moment to settle "
                    "after the limit is applied.",
                    style="grey50 italic",
                )
            )
        return renderables

    @staticmethod
    def _progress_bar(percent: float | None) -> str:
        """Render a fixed-width text progress bar; an indeterminate marker when the total is unknown."""
        if percent is None:
            return "[" + "?" * _BAR_WIDTH + "]"
        filled = int(round(percent / 100.0 * _BAR_WIDTH))
        return "[" + "█" * filled + "░" * (_BAR_WIDTH - filled) + f"] {percent:5.1f}%"

    def _render_queue(self, downloads: DownloadStatusSnapshot | None) -> Panel:
        """Render the queue of pending downloads, labelled with feature and size."""
        pending = downloads.pending if downloads is not None else []
        table = Table.grid(padding=(0, 2))
        table.add_column(style="grey70")
        table.add_column(style="cyan")
        table.add_column(justify="right", style="grey62")
        if not pending:
            table.add_row(Text("queue empty", style="grey50"), "", "")
        for item in pending:
            table.add_row(shorten(item.model_name, 40), item.feature, human_bytes(item.size_bytes))
        title = f"Queued ({len(pending)})"
        return Panel(table, title=title, title_align="left", border_style="grey37", padding=(0, 1))

    def _render_failures(self, downloads: DownloadStatusSnapshot | None) -> Panel:
        """Render any failed downloads with their feature and reason."""
        failures = downloads.failures if downloads is not None else []
        if not failures:
            body: RenderableType = Text("no failures", style="grey50")
            return Panel(body, title="Failures", title_align="left", border_style="grey37", padding=(0, 1))

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column(style="cyan")
        table.add_column(style="red")
        for failure in failures:
            table.add_row(shorten(failure.model_name, 40), failure.feature, failure.reason)
        title = f"Failures ({len(failures)})"
        return Panel(table, title=title, title_align="left", border_style="red", padding=(0, 1))
