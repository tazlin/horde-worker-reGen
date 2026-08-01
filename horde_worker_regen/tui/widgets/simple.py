"""Plain-language presentations of the core destinations for the Simple experience.

These render the same worker state the operator views render, in contributor vocabulary and with the
live feedback a first-time user needs to believe the worker is working: real per-job progress, an
accumulating trend, a feed of recent outcomes, and a liveness indicator that stops when the worker does.

Only Overview, Live, Downloads and Config get bespoke Simple presentations. The remaining destinations
keep their operator widgets at every level and gain only an explanatory line (see ``tab_intro``).
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static

from horde_worker_regen.tui.formatters import human_duration, mini_bar, sparkline
from horde_worker_regen.tui.health import HealthReport, HealthStatus, WorkerPhase

if TYPE_CHECKING:
    from horde_worker_regen.process_management.ipc.supervisor_channel import (
        ProcessSnapshot,
        WorkerStateSnapshot,
    )

_PHASE_WORDS: dict[WorkerPhase, str] = {
    WorkerPhase.STOPPED: "Not contributing yet",
    WorkerPhase.CRASHED: "The worker stopped unexpectedly",
    WorkerPhase.RESTARTING: "Restarting",
    WorkerPhase.INITIALIZING: "Starting up",
    WorkerPhase.WARMING_UP: "Getting models ready",
    WorkerPhase.SERVING: "Creating images for the community",
    WorkerPhase.READY: "Ready for requests",
    WorkerPhase.IDLE: "Waiting for the next request",
    WorkerPhase.MAINTENANCE: "Paused for maintenance",
    WorkerPhase.PAUSED: "Paused",
    WorkerPhase.SHUTTING_DOWN: "Finishing current work",
    WorkerPhase.DEGRADED: "Working, but something needs attention",
    WorkerPhase.DISCONNECTED: "Lost contact with the worker",
    WorkerPhase.UNRESPONSIVE: "The worker has stopped responding",
}

_WORKING_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
"""Frames for the working indicator; advanced by measured progress, never by wall-clock alone."""

_WORKING_FRAMES_ASCII = ("|", "/", "-", "\\")

_IDLE_FRAMES = ("·", "•", "·", " ")
"""A slower breathing indicator for a worker that is responsive but has no work in hand."""

_STALL_TICKS = 40
"""Consecutive frames with a job in hand and no sampling progress before the indicator says so."""

_TREND_SAMPLE_MIN_SECONDS = 2.0
"""Minimum wall-clock gap between retained trend samples, so a window spans minutes not frames."""

_TREND_MAX_SAMPLES = 120
"""Retained trend samples; at the sampling floor this is roughly the last four minutes."""

_TICKER_LINES = 6
"""Recent events kept in the Home ticker; the Activity destination shows a longer history."""


def tab_intro(text: str) -> Static:
    """Build the one-line plain-language framing shown above an operator widget in Simple."""
    return Static(text, classes="horde-tab-intro level-simple-only")


def titled(title: str, body: RenderableType) -> Table:
    """Stack a bold ``title`` above ``body`` in a single grid."""
    wrapper = Table.grid()
    wrapper.add_column()
    wrapper.add_row(Text(title, "bold"))
    wrapper.add_row(body)
    return wrapper


def job_progress_fraction(process: ProcessSnapshot) -> float | None:
    """Return 0..1 progress for a busy process, or None when it reports none.

    Prefers the reported percentage and falls back to the step counters. Returns None rather than
    guessing, so the caller shows an indeterminate state instead of inventing a number.
    """
    percent = process.last_heartbeat_percent_complete
    if percent is not None and 0 <= percent <= 100:
        return percent / 100.0
    current = process.last_current_step
    total = process.last_total_steps
    if current is not None and total:
        return max(0.0, min(1.0, current / total))
    return None


class LivenessIndicator:
    """Derives an honest working/idle/stalled indicator from the worker's own progress counters.

    The animation advances on observed change in worker state, never on the render loop, so a frozen
    worker visibly freezes. A progress signal the failure itself can satisfy proves nothing: a spinner
    driven by the dashboard's own timer keeps spinning cheerfully over a wedged worker.

    Two signals are tracked because they fail differently. ``heartbeats_inference_steps`` advances only
    when sampling actually progresses, so it is the truthful signal while a job is in hand. The snapshot
    timestamp advances whenever the worker's loop is alive at all, which is the right signal when there
    is no work to do. A job in hand with the step counter static is a stall.
    """

    def __init__(self) -> None:
        """Start with no observations."""
        self._phase = 0
        self._last_steps: int | None = None
        self._last_timestamp: float | None = None
        self._stalled_ticks = 0

    def update(self, snapshot: WorkerStateSnapshot | None, *, is_alive: bool) -> None:
        """Fold one frame of worker state into the indicator."""
        if snapshot is None or not is_alive:
            self._last_steps = None
            self._last_timestamp = None
            self._stalled_ticks = 0
            return
        steps = sum(process.heartbeats_inference_steps for process in snapshot.processes)
        working = snapshot.jobs_in_progress > 0
        progressed = self._last_steps is not None and steps != self._last_steps
        responded = self._last_timestamp is not None and snapshot.timestamp != self._last_timestamp
        if progressed or (not working and responded):
            self._phase += 1
            self._stalled_ticks = 0
        elif working and self._last_steps is not None:
            self._stalled_ticks += 1
        self._last_steps = steps
        self._last_timestamp = snapshot.timestamp

    @property
    def stalled(self) -> bool:
        """Whether a job has been in hand for a while with no sampling progress reported."""
        return self._stalled_ticks >= _STALL_TICKS

    def marker(self, snapshot: WorkerStateSnapshot | None, *, is_alive: bool, ascii_only: bool = False) -> Text:
        """Render the current indicator glyph."""
        if snapshot is None or not is_alive:
            return Text("○", "grey50")
        if self.stalled:
            return Text("!", "bold yellow")
        if snapshot.jobs_in_progress > 0:
            frames = _WORKING_FRAMES_ASCII if ascii_only else _WORKING_FRAMES
            return Text(frames[self._phase % len(frames)], "bold green")
        return Text(_IDLE_FRAMES[self._phase % len(_IDLE_FRAMES)], "cyan")


class SimpleHomeView(VerticalScroll):
    """Plain-language home: what the worker is doing, how it is going, and what needs attention."""

    class StartStopRequested(Message):
        """Posted when the contributor uses the primary start/stop action."""

    class SetupRequested(Message):
        """Posted when the contributor asks to finish first-time setup."""

    class NavigateRequested(Message):
        """Posted when a Home action links to another destination."""

        def __init__(self, destination: str) -> None:
            """Store the destination tab identifier."""
            super().__init__()
            self.destination = destination

    DEFAULT_CSS = """
    SimpleHomeView #simple-headlines {
        height: auto;
    }
    SimpleHomeView .simple-headline {
        width: 1fr;
        min-height: 5;
        margin-right: 1;
    }
    SimpleHomeView #simple-setup {
        border: tall $warning;
    }
    Screen.-narrow SimpleHomeView .simple-headline {
        margin-right: 0;
    }
    """

    def __init__(self) -> None:
        """Start with empty trend history and no observations."""
        super().__init__()
        self._liveness = LivenessIndicator()
        self._kudos_samples: deque[float] = deque(maxlen=_TREND_MAX_SAMPLES)
        self._completed_samples: deque[float] = deque(maxlen=_TREND_MAX_SAMPLES)
        self._last_sample_at = 0.0
        self._ticker: deque[str] = deque(maxlen=_TICKER_LINES)
        self._seen_job_ids: set[str] = set()
        self._setup_required = False

    def compose(self) -> ComposeResult:
        """Lay out the setup prompt, status hero, headline figures, live progress, and feed."""
        yield Static(id="simple-setup", classes="horde-card")
        yield Static(id="simple-status", classes="horde-hero")
        with Horizontal(id="simple-headlines"):
            yield Static(id="simple-requests", classes="horde-card simple-headline")
            yield Static(id="simple-kudos", classes="horde-card simple-headline")
        yield Static(id="simple-current", classes="horde-card")
        yield Static(id="simple-attention", classes="horde-card")
        yield Static(id="simple-ticker", classes="horde-card")
        with Horizontal(id="simple-actions", classes="horde-actions"):
            yield Button("Start contributing", id="simple-start-stop", variant="success")
            yield Button("Finish setup", id="simple-setup-action", variant="success")
            yield Button("See activity", id="simple-show-activity")
            yield Button("Models", id="simple-show-models")

    def on_mount(self) -> None:
        """Hide the conditional cards until a state update decides they are needed."""
        self.set_setup_required(False)
        self.query_one("#simple-attention", Static).display = False

    def set_setup_required(self, required: bool) -> None:
        """Switch between the finish-setup prompt and the everyday home."""
        self._setup_required = required
        self.query_one("#simple-setup", Static).display = required
        self.query_one("#simple-setup-action", Button).display = required
        self.query_one("#simple-start-stop", Button).display = not required
        if required:
            self.query_one("#simple-setup", Static).update(
                Text.assemble(
                    ("Finish setting up this worker\n", "bold"),
                    (
                        "This computer needs an AI Horde API key and a worker name before it can take "
                        "requests. The setup guide walks through both.",
                        "grey70",
                    ),
                ),
            )

    def update_view(
        self,
        report: HealthReport,
        snapshot: WorkerStateSnapshot | None,
        *,
        is_alive: bool,
        ascii_only: bool = False,
    ) -> None:
        """Refresh every card from one frame of worker state."""
        self._liveness.update(snapshot, is_alive=is_alive)
        self._record_trend(snapshot)
        self._record_events(snapshot)
        self._render_status(report, snapshot, is_alive=is_alive, ascii_only=ascii_only)
        self._render_headlines(snapshot)
        self._render_current(snapshot, is_alive=is_alive)
        self._render_attention(report)
        self._render_ticker()
        action = self.query_one("#simple-start-stop", Button)
        action.label = "Stop after current work" if is_alive else "Start contributing"
        action.variant = "warning" if is_alive else "success"

    def _record_trend(self, snapshot: WorkerStateSnapshot | None) -> None:
        """Retain a bounded trend history, sampled on wall-clock rather than every frame."""
        if snapshot is None:
            return
        now = time.time()
        if now - self._last_sample_at < _TREND_SAMPLE_MIN_SECONDS:
            return
        self._last_sample_at = now
        self._kudos_samples.append(float(snapshot.kudos_this_session or 0.0))
        self._completed_samples.append(float(snapshot.num_jobs_submitted))

    def _record_events(self, snapshot: WorkerStateSnapshot | None) -> None:
        """Append newly finished requests to the feed, newest last."""
        if snapshot is None:
            return
        for job in snapshot.recent_jobs:
            if job.job_id in self._seen_job_ids:
                continue
            self._seen_job_ids.add(job.job_id)
            elapsed = f" in {job.e2e_seconds:.0f}s" if job.e2e_seconds is not None else ""
            if job.faulted:
                self._ticker.append(f"Could not finish a request{elapsed}")
            elif job.is_alchemy:
                self._ticker.append(f"Finished an alchemy request{elapsed}")
            else:
                self._ticker.append(f"Finished an image request{elapsed}")

    def _render_status(
        self,
        report: HealthReport,
        snapshot: WorkerStateSnapshot | None,
        *,
        is_alive: bool,
        ascii_only: bool,
    ) -> None:
        """Render the headline phase with the liveness marker beside it."""
        marker = self._liveness.marker(snapshot, is_alive=is_alive, ascii_only=ascii_only)
        phase = _PHASE_WORDS.get(report.phase, report.headline)
        if self._liveness.stalled:
            phase = "Working, but no progress reported recently"
        style = "bold red" if report.severity >= HealthStatus.ERROR else "bold green"
        self.query_one("#simple-status", Static).update(
            Text.assemble(marker, ("  ", ""), (f"{phase}\n", style), (report.detail, "grey70")),
        )

    def _render_headlines(self, snapshot: WorkerStateSnapshot | None) -> None:
        """Render the two headline figures, each over its own accumulating trend."""
        completed = snapshot.num_jobs_submitted if snapshot is not None else 0
        kudos = (snapshot.kudos_this_session or 0.0) if snapshot is not None else 0.0
        self.query_one("#simple-requests", Static).update(
            Text.assemble(
                (f"{completed:,}\n", "bold cyan"),
                ("community requests completed\n", "grey70"),
                (sparkline(list(self._completed_samples)), "cyan"),
            ),
        )
        self.query_one("#simple-kudos", Static).update(
            Text.assemble(
                (f"{kudos:,.1f}\n", "bold magenta"),
                ("kudos earned this session\n", "grey70"),
                (sparkline(list(self._kudos_samples)), "magenta"),
            ),
        )

    def _render_current(self, snapshot: WorkerStateSnapshot | None, *, is_alive: bool) -> None:
        """Render a real progress bar per request in flight."""
        card = self.query_one("#simple-current", Static)
        if snapshot is None or not is_alive:
            card.update(titled("Right now", Text("Nothing running yet.", "grey70")))
            return
        busy = [process for process in snapshot.processes if process.is_busy and process.current_job_id]
        if not busy:
            card.update(titled("Right now", Text(self._idle_detail(snapshot), "grey70")))
            return
        table = Table.grid(padding=(0, 1))
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True)
        table.add_column(justify="right", no_wrap=True)
        for process in busy:
            fraction = job_progress_fraction(process)
            name = process.loaded_horde_model_name or "an image request"
            if fraction is None:
                table.add_row(Text("working", "green"), Text(name, "grey70"), Text(""))
                continue
            table.add_row(
                Text(mini_bar(fraction, 16), "green"),
                Text(name, "grey70"),
                Text(f"{fraction * 100:.0f}%", "bold"),
            )
        card.update(titled("Right now", table))

    @staticmethod
    def _idle_detail(snapshot: WorkerStateSnapshot) -> str:
        """Explain what the worker is doing when nothing is being sampled."""
        downloads = snapshot.downloads
        if downloads is not None and downloads.phase.value != "idle":
            return "Downloading the models needed to contribute."
        pending = snapshot.jobs_pending_inference
        if pending:
            return f"Preparing {pending} request{'s' if pending != 1 else ''}."
        return "Waiting for a community request."

    def _render_attention(self, report: HealthReport) -> None:
        """Show the first concerning check, if any."""
        concerning = [check for check in report.checks if check.status >= HealthStatus.WARN]
        card = self.query_one("#simple-attention", Static)
        if not concerning:
            card.display = False
            return
        finding = concerning[0]
        card.display = True
        card.update(Text.assemble(("Needs attention\n", "bold yellow"), (finding.detail, "grey70")))

    def _render_ticker(self) -> None:
        """Render the recent-events feed, newest last so it reads as accumulating."""
        card = self.query_one("#simple-ticker", Static)
        if not self._ticker:
            card.update(titled("Recent", Text("Nothing has finished yet.", "grey70")))
            return
        body = Text()
        for entry in self._ticker:
            body.append(f"{entry}\n", "grey70")
        card.update(titled("Recent", body))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the Home actions."""
        routes: dict[str, Message] = {
            "simple-start-stop": self.StartStopRequested(),
            "simple-setup-action": self.SetupRequested(),
            "simple-show-activity": self.NavigateRequested("tab-live"),
            "simple-show-models": self.NavigateRequested("tab-downloads"),
        }
        message = routes.get(event.button.id or "")
        if message is None:
            return
        event.stop()
        self.post_message(message)


class SimpleActivityView(VerticalScroll):
    """Recent contribution activity, without process or scheduler vocabulary."""

    def compose(self) -> ComposeResult:
        """Lay out the session summary and the request feed."""
        yield Static("Community activity", classes="horde-hero horde-card-title")
        yield Static(id="simple-activity-summary", classes="horde-card")
        yield Static(id="simple-activity-recent", classes="horde-card")

    def update_view(self, snapshot: WorkerStateSnapshot | None) -> None:
        """Refresh the totals and the recent request outcomes."""
        summary = self.query_one("#simple-activity-summary", Static)
        recent = self.query_one("#simple-activity-recent", Static)
        if snapshot is None:
            summary.update(Text("Activity appears once the worker starts.", "grey70"))
            recent.update(Text(""))
            return
        grid = Table.grid(padding=(0, 3))
        grid.add_column(style="bold cyan", justify="right")
        grid.add_column()
        grid.add_row(f"{snapshot.num_jobs_submitted:,}", "requests completed")
        grid.add_row(f"{snapshot.num_jobs_faulted:,}", "could not be completed")
        grid.add_row(f"{snapshot.kudos_this_session or 0:,.1f}", "kudos earned this session")
        if snapshot.kudos_per_hour is not None:
            grid.add_row(f"{snapshot.kudos_per_hour:,.0f}", "kudos per hour while working")
        if snapshot.session_start_time:
            uptime = time.time() - snapshot.session_start_time
            if uptime > 0:
                grid.add_row(human_duration(uptime), "contributing this session")
        summary.update(titled("This session", grid))

        table = Table(expand=True, box=None)
        table.add_column("Result", no_wrap=True)
        table.add_column("Request")
        table.add_column("Took", justify="right", no_wrap=True)
        for job in list(snapshot.recent_jobs)[-12:][::-1]:
            result = Text("Not completed", "yellow") if job.faulted else Text("Completed", "green")
            work = "Alchemy request" if job.is_alchemy else (job.model_name or "Image request")
            elapsed = f"{job.e2e_seconds:.1f}s" if job.e2e_seconds is not None else "-"
            table.add_row(result, work, elapsed)
        if not snapshot.recent_jobs:
            table.add_row(Text("Waiting", "grey62"), "No requests have finished yet", "-")
        recent.update(titled("Recent requests", table))


class SimpleModelStatusView(VerticalScroll):
    """Read-only model readiness and download progress.

    Deliberately offers no editing. Which models this worker runs is configured in one place, on the
    Config destination, and this view links there rather than duplicating the control; two editable
    surfaces over one setting is how they drift apart.
    """

    class ManageRequested(Message):
        """Posted when the contributor asks to change which models are configured."""

    def compose(self) -> ComposeResult:
        """Lay out the readiness summary, download progress, and the link to configuration."""
        yield Static("Models", classes="horde-hero horde-card-title")
        yield Static(id="simple-models-state", classes="horde-card")
        yield Static(id="simple-models-downloads", classes="horde-card")
        with Horizontal(classes="horde-actions"):
            yield Button("Change which models to run", id="simple-models-manage", variant="primary")

    def update_view(self, snapshot: WorkerStateSnapshot | None) -> None:
        """Refresh readiness and download progress."""
        state = self.query_one("#simple-models-state", Static)
        downloads = self.query_one("#simple-models-downloads", Static)
        if snapshot is None:
            state.update(Text("Model status appears once the worker starts.", "grey70"))
            downloads.update(Text(""))
            return
        state.update(titled("Model status", self._render_state(snapshot)))
        downloads.update(titled("Downloads", self._render_downloads(snapshot)))

    @staticmethod
    def _render_state(snapshot: WorkerStateSnapshot) -> RenderableType:
        """Summarise which configured models are loaded, on disk, fetching, or failed."""
        loaded = sorted({p.loaded_horde_model_name for p in snapshot.processes if p.loaded_horde_model_name})
        configured = list(snapshot.active_models)
        activity = snapshot.downloads
        on_disk = set(activity.present_model_names) if activity is not None else set()
        fetching = sorted({item.model_name for item in activity.active}) if activity is not None else []
        failed = sorted({failure.model_name for failure in activity.failures}) if activity is not None else []
        ready = sorted(name for name in configured if name in on_disk and name not in loaded)
        waiting = sorted(
            name for name in configured if name not in on_disk and name not in fetching and name not in failed
        )

        grid = Table.grid(padding=(0, 3))
        grid.add_column(style="bold", justify="right")
        grid.add_column()
        grid.add_row(Text(str(len(loaded)), "green"), "loaded and serving requests")
        grid.add_row(Text(str(len(ready)), "cyan"), "downloaded, ready to load")
        if fetching:
            grid.add_row(Text(str(len(fetching)), "yellow"), "downloading now")
        if waiting:
            grid.add_row(Text(str(len(waiting)), "grey62"), "queued to download")
        if failed:
            grid.add_row(Text(str(len(failed)), "red"), "could not be downloaded")

        body = Table.grid()
        body.add_column()
        body.add_row(grid)
        if loaded:
            names = Text()
            names.append("\nReady now: ", "bold")
            names.append(", ".join(loaded), "grey70")
            body.add_row(names)
        if failed:
            names = Text()
            names.append("\nFailed: ", "bold red")
            names.append(", ".join(failed), "grey70")
            body.add_row(names)
        return body

    @staticmethod
    def _render_downloads(snapshot: WorkerStateSnapshot) -> RenderableType:
        """Render live download progress, or say plainly that nothing is downloading."""
        activity = snapshot.downloads
        if activity is None or activity.phase.value == "idle" or not activity.active:
            return Text("Nothing is downloading right now.", "grey70")
        table = Table.grid(padding=(0, 1))
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True)
        table.add_column(justify="right", no_wrap=True)
        for item in activity.active:
            percent = item.percent
            if percent is None:
                table.add_row(Text("fetching", "yellow"), Text(item.model_name, "grey70"), Text(""))
                continue
            table.add_row(
                Text(mini_bar(percent / 100.0, 16), "yellow"),
                Text(item.model_name, "grey70"),
                Text(f"{percent:.0f}%", "bold"),
            )
        note = Text("\nModels download in the background; the worker keeps contributing meanwhile.", "grey70")
        wrapper = Table.grid()
        wrapper.add_column()
        wrapper.add_row(table)
        wrapper.add_row(note)
        return wrapper

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the link into configuration."""
        if event.button.id == "simple-models-manage":
            event.stop()
            self.post_message(self.ManageRequested())


__all__ = [
    "LivenessIndicator",
    "SimpleActivityView",
    "SimpleHomeView",
    "SimpleModelStatusView",
    "job_progress_fraction",
    "tab_intro",
    "titled",
]
