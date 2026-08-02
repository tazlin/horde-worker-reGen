"""Plain-language presentations of the core destinations for the Simple experience.

These render the same worker state the operator views render, in contributor vocabulary and with the
live feedback a first-time user needs to believe the worker is working: real per-job progress, an
accumulating trend, a feed of recent outcomes, and a liveness indicator that stops when the worker does.

Only Overview, Live, Downloads and Config get bespoke Simple presentations. The remaining destinations
keep their operator widgets at every level, framed by a ``TabPrimer`` that explains what the widget is
for and works through the figures it is currently showing.
"""

from __future__ import annotations

import contextlib
import enum
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Collapsible, Static

from horde_worker_regen.process_management.ipc.supervisor_channel import RECENT_JOBS_IN_SNAPSHOT, DownloadPhase
from horde_worker_regen.tui.formatters import format_percent, human_bytes, human_duration, mini_bar, sparkline
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

_DOWNLOAD_PHASE_DETAIL: dict[DownloadPhase, str] = {
    DownloadPhase.INITIALIZING: "Looking up which models are available.",
    DownloadPhase.SCANNING: "Checking which models are already on this computer.",
    DownloadPhase.DOWNLOADING: "Downloading the models needed to contribute.",
    DownloadPhase.PAUSED: "Model downloads are paused.",
    DownloadPhase.ERROR: "A model download could not be completed.",
}
"""Plain-language reading of each download phase; an absent phase means downloads are not the story."""

_IDLE_FRAMES = ("·", "•", "·", " ")
"""A slower breathing indicator for a worker that is responsive but has no work in hand."""

_SEEN_JOB_MEMORY = RECENT_JOBS_IN_SNAPSHOT * 10
"""Finished requests remembered so the feed does not repeat one.

A job can only be re-offered while it remains inside the snapshot's own window, so any bound above that
window suffices; the multiple leaves room for the window to widen. The bound exists because a session
can run for days, and an unbounded record would hold one identifier per request served.
"""

_TREND_SAMPLE_MIN_SECONDS = 2.0
"""Minimum wall-clock gap between retained trend samples, so a window spans minutes not frames."""

_TREND_MAX_SAMPLES = 120
"""Retained trend samples; at the sampling floor this is roughly the last four minutes."""

_TICKER_LINES = 6
"""Recent events kept in the Home ticker; the Activity destination shows a longer history."""


class SimpleDestination(enum.StrEnum):
    """Where a Simple action wants to send the contributor.

    Named by intent rather than by tab identifier so these views stay ignorant of the app's DOM: the
    dashboard owns the mapping onto tabs, and a tab that is renamed or re-identified does not silently
    break a link from here.
    """

    ACTIVITY = "activity"
    """Recent contribution activity."""
    MODELS = "models"
    """Model readiness and download progress."""


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
    """Derives an honest working/idle indicator from signals only the worker can advance.

    The animation advances on observed change in worker state, never on the render loop, so a frozen
    worker visibly freezes. A signal the failure itself can satisfy carries no information: a spinner
    driven by the dashboard's own timer keeps turning over a wedged worker.

    Which signal carries that proof depends on whether there is work in hand.

    While a job is in hand, only the child processes may advance the frame, through their sampling
    counter or their heartbeat timestamps. A supervisor whose own loop is healthy goes on stamping
    snapshots over a wedged child, so its snapshot timestamp is excluded in that state. With nothing in
    hand the snapshot timestamp becomes the correct signal: the worker is alive with nothing to do, and
    no child is reporting.

    Both child signals are read, because they answer different questions.
    ``heartbeats_inference_steps`` advances only while sampling and resets to zero on every other kind of
    heartbeat, so alone it reads a model load or a post-processing pass as a stall.
    ``last_heartbeat_timestamp`` advances on any heartbeat, which separates a busy non-sampling stage
    from an absent process.

    Whether a stall amounts to a fault is settled by
    [`derive`][horde_worker_regen.tui.health.derive], against tuned, download-aware thresholds the whole
    dashboard shares. This class supplies the animation and takes the verdict from the health report.
    """

    def __init__(self) -> None:
        """Start with no observations."""
        self._phase = 0
        self._last_steps: int | None = None
        self._last_child_heartbeat: float | None = None
        self._last_timestamp: float | None = None

    def update(self, snapshot: WorkerStateSnapshot | None, *, is_alive: bool) -> None:
        """Fold one frame of worker state into the indicator."""
        if snapshot is None or not is_alive:
            self._last_steps = None
            self._last_child_heartbeat = None
            self._last_timestamp = None
            return
        steps = sum(process.heartbeats_inference_steps for process in snapshot.processes)
        child_heartbeat = max((process.last_heartbeat_timestamp for process in snapshot.processes), default=0.0)
        working = snapshot.jobs_in_progress > 0
        sampled = self._last_steps is not None and steps != self._last_steps
        child_reported = self._last_child_heartbeat is not None and child_heartbeat != self._last_child_heartbeat
        supervisor_reported = self._last_timestamp is not None and snapshot.timestamp != self._last_timestamp
        advanced = (sampled or child_reported) if working else supervisor_reported
        if advanced:
            self._phase += 1
        self._last_steps = steps
        self._last_child_heartbeat = child_heartbeat
        self._last_timestamp = snapshot.timestamp

    def marker(
        self,
        snapshot: WorkerStateSnapshot | None,
        *,
        is_alive: bool,
        concerning: bool = False,
        ascii_only: bool = False,
    ) -> Text:
        """Render the current indicator glyph, deferring to the health report when it is ``concerning``."""
        if snapshot is None or not is_alive:
            return Text("○", "grey50")
        if concerning:
            return Text("!", "bold yellow")
        if snapshot.jobs_in_progress > 0:
            frames = _WORKING_FRAMES_ASCII if ascii_only else _WORKING_FRAMES
            return Text(frames[self._phase % len(frames)], "bold green")
        return Text(_IDLE_FRAMES[self._phase % len(_IDLE_FRAMES)], "cyan")


class PrimerReading(NamedTuple):
    """One figure on a technical destination, with the plain-language reading of it."""

    label: str
    """The name the operator widget below uses for this figure, kept verbatim so the two line up."""
    explain: str
    """What the figure means and what a contributor should conclude from it."""
    read: Callable[[WorkerStateSnapshot, HealthReport], str | None]
    """Extract the figure's current value, or None when the worker cannot report it yet."""


def _alive_processes(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str:
    """Count the background programs that are up, against the number configured."""
    alive = sum(1 for process in snapshot.processes if process.is_alive)
    return f"{alive} of {len(snapshot.processes)}"


def _busy_processes(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str:
    """Count the background programs holding a request right now."""
    return str(sum(1 for process in snapshot.processes if process.is_busy))


def _passing_checks(_snapshot: WorkerStateSnapshot, report: HealthReport) -> str:
    """Count the health checks currently at OK, against the total run."""
    passing = sum(1 for check in report.checks if check.status is HealthStatus.OK)
    return f"{passing} of {len(report.checks)}"


def _free_disk(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str | None:
    """Report the tightest of the tracked disks, since that is the one that runs out first."""
    if not snapshot.disk_free_bytes:
        return None
    return human_bytes(min(snapshot.disk_free_bytes.values()))


def _free_memory(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str | None:
    """Report system RAM still available to the machine as a whole."""
    memory = snapshot.system_memory
    return human_bytes(memory.available_bytes) if memory is not None else None


PRIMERS: dict[str, tuple[PrimerReading, ...]] = {
    "Stats": (
        PrimerReading(
            "Submitted",
            "Requests you finished and sent back to the horde. This is the number that earns kudos.",
            lambda snapshot, _report: f"{snapshot.num_jobs_submitted:,}",
        ),
        PrimerReading(
            "Faulted",
            "Requests that could not be completed. A handful over a long session is ordinary; a share "
            "that climbs alongside Submitted is worth chasing on the Logs tab.",
            lambda snapshot, _report: f"{snapshot.num_jobs_faulted:,}",
        ),
        PrimerReading(
            "Kudos/hr",
            "Kudos earned per hour of work actually held, rather than per hour this window has been "
            "open, so it does not sag while the horde has nothing to send you.",
            lambda snapshot, _report: (
                f"{snapshot.kudos_per_hour:,.0f}" if snapshot.kudos_per_hour is not None else None
            ),
        ),
        PrimerReading(
            "GPU busy",
            "Share of the time your graphics card was working. Low while requests are waiting points at "
            "a bottleneck somewhere else; low with nothing waiting just means a quiet horde.",
            lambda snapshot, _report: format_percent(snapshot.gpu_utilization_busy_fraction),
        ),
    ),
    "Control": (
        PrimerReading(
            "Programs up",
            "The worker splits its work across separate background programs. All of them being up is "
            "the healthy state.",
            _alive_processes,
        ),
        PrimerReading(
            "Working now",
            "How many of those programs hold a request at this moment. Zero while requests are queued "
            "usually means they are still getting a model ready.",
            _busy_processes,
        ),
        PrimerReading(
            "Restarts",
            "Times the worker noticed a program had stopped responding and replaced it. It does this "
            "unprompted, so a count that keeps climbing is the signal something deeper is wrong.",
            lambda snapshot, _report: str(snapshot.num_process_recoveries),
        ),
    ),
    "Logs": (
        PrimerReading(
            "Failures in a row",
            "Consecutive requests that could not be completed. This resets on the next success, so a "
            "climbing value is the useful thing to quote when you ask for help.",
            lambda snapshot, _report: str(snapshot.consecutive_failed_jobs),
        ),
        PrimerReading(
            "Backing off",
            "Whether the worker has slowed its requests to the horde after repeated errors. It clears "
            "itself once calls succeed again.",
            lambda snapshot, _report: "yes" if snapshot.in_error_backoff else "no",
        ),
        PrimerReading(
            "Notices from the horde",
            "Messages the horde sent this worker, about maintenance or your account. They appear in the "
            "log alongside the worker's own lines.",
            lambda snapshot, _report: str(len(snapshot.api_messages)),
        ),
    ),
    "Insights": (
        PrimerReading(
            "Waiting for work",
            "Time spent with the horde having nothing to send. This is demand, and no setting on this "
            "machine shortens it.",
            lambda snapshot, _report: human_duration(snapshot.time_spent_no_jobs_available),
        ),
        PrimerReading(
            "GPU busy",
            "Share of the time the card was working. Raising this is what raises kudos per hour, and "
            "the rows below name whatever is currently keeping it idle.",
            lambda snapshot, _report: format_percent(snapshot.gpu_utilization_busy_fraction),
        ),
        PrimerReading(
            "Queued to start",
            "Requests accepted but not yet started. A queue that never empties means work arrives "
            "faster than this machine finishes it.",
            lambda snapshot, _report: str(snapshot.jobs_pending_inference),
        ),
        PrimerReading(
            "Slowdowns",
            "Times a request ran slower than the horde was promised. A few are normal; many suggest "
            "this worker has taken on more than the hardware sustains.",
            lambda snapshot, _report: str(snapshot.num_job_slowdowns),
        ),
    ),
    "Diagnostics": (
        PrimerReading(
            "Checks passing",
            "Automated checks on this machine's setup. Anything not passing is described in full below, "
            "with what to do about it.",
            _passing_checks,
        ),
        PrimerReading(
            "Free disk",
            "Space left on the tightest of the drives the worker uses. Models are large, and a full "
            "disk stops downloads quietly rather than failing loudly.",
            _free_disk,
        ),
        PrimerReading(
            "Free memory",
            "System RAM still available. The worker holds models in memory as well as on the card, so "
            "this running low slows everything down.",
            _free_memory,
        ),
    ),
    "Benchmark": (
        PrimerReading(
            "Graphics cards",
            "Cards a benchmark would measure. It tries progressively harder work on each and stops "
            "where the hardware stops keeping up.",
            lambda snapshot, _report: str(len(snapshot.per_card) or 1),
        ),
        PrimerReading(
            "Models configured",
            "Models this worker offers. The suggestions a benchmark makes account for how many it has "
            "to keep ready at once.",
            lambda snapshot, _report: str(snapshot.config.num_models),
        ),
        PrimerReading(
            "Requests at once",
            "How many requests the worker takes in parallel today. This is the main setting a benchmark "
            "recommends changing.",
            lambda snapshot, _report: str(snapshot.config.max_threads),
        ),
    ),
}
"""Live worked examples for the destinations that keep their operator widget at every level.

Keyed by tab caption. Each reading names a figure the widget below already shows, so the explanation
and the table cannot end up describing different things.
"""


class TabPrimer(Vertical):
    """A plain-language framing of a technical destination, worked through its own current numbers.

    Shown only in Simple, above the operator widget it explains. The figures are read live rather than
    illustrated with invented ones, so the numbers named in the explanation are the numbers on screen
    underneath it.

    The framing line is always visible and the worked example folds away, because the example runs to
    more rows than an eighty-by-twenty-four terminal has to spare and would otherwise push the widget it
    explains off the bottom of the screen. It opens expanded, so it is read before it is dismissed.
    """

    _EXAMPLE_CLASS = "tab-primer-example"

    DEFAULT_CSS = """
    TabPrimer {
        height: auto;
    }
    TabPrimer Collapsible {
        border: none;
        padding: 0;
        margin: 0;
    }
    TabPrimer CollapsibleTitle {
        color: $text-muted;
    }
    """

    def __init__(self, intro: str, readings: Sequence[PrimerReading] = ()) -> None:
        """Frame a destination with ``intro`` and explain each of ``readings`` beneath it."""
        super().__init__(classes="horde-tab-intro level-simple-only")
        self._intro = intro
        self._readings = tuple(readings)

    def compose(self) -> ComposeResult:
        """Lay out the always-visible framing line, then the collapsible worked example."""
        yield Static(Text(self._intro, "grey70"))
        if not self._readings:
            return
        with Collapsible(title="Example: reading your numbers", collapsed=False):
            yield Static(classes=self._EXAMPLE_CLASS)

    def on_mount(self) -> None:
        """Render the example before any worker state has arrived."""
        self.update_view(None, None)

    def update_view(self, snapshot: WorkerStateSnapshot | None, report: HealthReport | None) -> None:
        """Refresh the worked figures from one frame of worker state."""
        if not self._readings:
            return
        with contextlib.suppress(NoMatches):
            self.query_one(f".{self._EXAMPLE_CLASS}", Static).update(self._build(snapshot, report))

    def _build(self, snapshot: WorkerStateSnapshot | None, report: HealthReport | None) -> RenderableType:
        """Build one named figure and its explanation per reading."""
        body = Table.grid()
        body.add_column()
        for index, reading in enumerate(self._readings):
            value: str | None = None
            if snapshot is not None and report is not None:
                value = reading.read(snapshot, report)
            if index:
                body.add_row(Text(""))
            figure = Table.grid(expand=True)
            figure.add_column(no_wrap=True)
            figure.add_column(justify="right", no_wrap=True)
            figure.add_row(Text(reading.label, "bold cyan"), Text(value or "not yet", "bold"))
            body.add_row(figure)
            body.add_row(Text(reading.explain, "grey62"))
        return body


class SimpleHomeView(VerticalScroll):
    """Plain-language home: what the worker is doing, how it is going, and what needs attention."""

    class StartStopRequested(Message):
        """Posted when the contributor uses the primary start/stop action."""

    class SetupRequested(Message):
        """Posted when the contributor asks to finish first-time setup."""

    class NavigateRequested(Message):
        """Posted when a Home action links to another destination."""

        def __init__(self, destination: SimpleDestination) -> None:
            """Store the requested destination."""
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
        # Insertion-ordered so the oldest identifier can be evicted; a plain set would grow for the life
        # of a session that runs for days.
        self._seen_job_ids: dict[str, None] = {}
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
            yield Button("Activity", id="simple-show-activity")
            yield Button("Model downloads", id="simple-show-models")

    def on_mount(self) -> None:
        """Hide the conditional cards until a state update decides they are needed."""
        self.set_setup_required(False, force=True)
        self.query_one("#simple-attention", Static).display = False

    def set_setup_required(self, required: bool, *, force: bool = False) -> None:
        """Switch between the finish-setup prompt and the everyday home.

        Returns early when nothing changed: the render loop calls this every frame, and re-rendering
        unchanged prompt text costs a repaint for no visible difference.
        """
        if required == self._setup_required and not force:
            return
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
        action.label = "Stop contributing" if is_alive else "Start contributing"
        action.variant = "warning" if is_alive else "success"

    def _record_trend(self, snapshot: WorkerStateSnapshot | None) -> None:
        """Retain a bounded trend history, sampled on wall-clock rather than every frame."""
        if snapshot is None:
            return
        # Monotonic: this measures an elapsed gap, and a wall-clock step would either stall the trend or
        # flood it with samples.
        now = time.monotonic()
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
            self._seen_job_ids[job.job_id] = None
            while len(self._seen_job_ids) > _SEEN_JOB_MEMORY:
                del self._seen_job_ids[next(iter(self._seen_job_ids))]
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
        concerning = report.severity >= HealthStatus.WARN
        marker = self._liveness.marker(
            snapshot,
            is_alive=is_alive,
            concerning=concerning,
            ascii_only=ascii_only,
        )
        phase = _PHASE_WORDS.get(report.phase, report.headline)
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
        """Explain what the worker is doing when nothing is being sampled.

        Reads the download phase rather than merely testing it against idle, so a paused or failed
        download is never described as one in progress.
        """
        downloads = snapshot.downloads
        if downloads is not None:
            detail = _DOWNLOAD_PHASE_DETAIL.get(downloads.phase)
            if detail is not None:
                return detail
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
            "simple-show-activity": self.NavigateRequested(SimpleDestination.ACTIVITY),
            "simple-show-models": self.NavigateRequested(SimpleDestination.MODELS),
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
        yield Static("Models and downloads", classes="horde-hero horde-card-title")
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
        loaded = sorted(
            {process.loaded_horde_model_name for process in snapshot.processes if process.loaded_horde_model_name},
        )
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
        if activity is None or activity.phase is DownloadPhase.IDLE or not activity.active:
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
    "PRIMERS",
    "PrimerReading",
    "SimpleDestination",
    "SimpleActivityView",
    "SimpleHomeView",
    "SimpleModelStatusView",
    "job_progress_fraction",
    "TabPrimer",
    "titled",
]
