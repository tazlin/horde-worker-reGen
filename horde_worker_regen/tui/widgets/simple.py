"""Plain-language presentations of the core destinations for the Simple experience.

These render the same worker state the operator views render, in contributor vocabulary and with the
live feedback a first-time user needs to believe the worker is working: real per-job progress, a trend
that charts recent pace rather than a running total, a feed of recent outcomes, and a liveness indicator
that stops when the worker does.

Only Overview, Live and Downloads get bespoke Simple presentations; Config keeps its editor, with pages
and fields scoped to the level. The remaining destinations keep their operator widgets at every level,
framed by a ``TabPrimer`` that says what the widget is for and, when the live figures warrant it, what
they currently mean.
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

from horde_worker_regen.app_state import OverviewTrendWindow
from horde_worker_regen.process_management.ipc.supervisor_channel import RECENT_JOBS_IN_SNAPSHOT, DownloadPhase
from horde_worker_regen.tui.formatters import (
    human_duration,
    is_low_fidelity,
    mini_bar,
    shorten,
    sparkline,
)
from horde_worker_regen.tui.health import HealthReport, HealthStatus, WorkerPhase
from horde_worker_regen.tui.trends import fixed_counter_deltas, fixed_ratio_deltas

if TYPE_CHECKING:
    from horde_worker_regen.process_management.ipc.supervisor_channel import (
        ProcessSnapshot,
        WorkerConfigSummary,
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

_TREND_MAX_SAMPLES = 480
"""Retained trend samples; at the sampling floor this covers the charted window with room to spare."""

_TREND_WINDOW = OverviewTrendWindow.FIFTEEN_MINUTES
"""The span the Home trends chart.

Fixed rather than selectable: Home answers "is it going now", and a window long enough to average a
stall away would defeat that, while the operator Overview keeps the selectable windows.
"""

_TREND_BUCKETS = 24
"""Points in a Home sparkline; narrow enough that both headline cards fit an 80-column terminal."""

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


def _identity_words(config: WorkerConfigSummary, *, uptime: float) -> str:
    """Name this worker, the build it is running, and who it is contributing for."""
    names = [name for name in (config.dreamer_name, config.alchemist_name) if name]
    line = f"{' and '.join(names) or 'Unnamed worker'}, version {config.worker_version}"
    if config.horde_username:
        line += f", contributing as {config.horde_username}"
        return f"{line} for {human_duration(uptime)}" if uptime > 0 else line
    return f"{line}, contributing for {human_duration(uptime)}" if uptime > 0 else line


def _offer_words(config: WorkerConfigSummary) -> str:
    """Name what this worker takes on, in the words a requester would recognise.

    LoRA follows the effective setting rather than the configured one, so a worker whose pops are not
    currently advertising LoRA does not claim it.
    """
    allows_lora = config.allow_lora if config.effective_allow_lora is None else config.effective_allow_lora
    offers: list[str] = []
    if allows_lora:
        offers.append("LoRA styles")
    if config.allow_controlnet or config.allow_sdxl_controlnet:
        offers.append("ControlNet guidance")
    if config.allow_img2img:
        offers.append("image-to-image")
    if config.allow_post_processing:
        offers.append("post-processing")
    if config.alchemist:
        offers.append("alchemy")
    return ", ".join(offers) if offers else "plain image requests"


def _scale_words(config: WorkerConfigSummary) -> str:
    """Describe how much this worker serves and how much it takes on at once."""
    requests = f"up to {config.max_threads:,} request{'s' if config.max_threads != 1 else ''} at once"
    if not config.num_models:
        return f"Taking {requests}"
    return f"Serving {config.num_models:,} model{'s' if config.num_models != 1 else ''}, {requests}"


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
    ) -> Text:
        """Render the current indicator glyph, deferring to the health report when it is ``concerning``.

        Frame glyphs follow the process-wide rendering fidelity, the same source ``sparkline`` and
        ``mini_bar`` consult, so one detection covers every animated element.
        """
        if snapshot is None or not is_alive:
            return Text("○", "grey50")
        if concerning:
            return Text("!", "bold yellow")
        if snapshot.jobs_in_progress > 0:
            frames = _WORKING_FRAMES_ASCII if is_low_fidelity() else _WORKING_FRAMES
            return Text(frames[self._phase % len(frames)], "bold green")
        return Text(_IDLE_FRAMES[self._phase % len(_IDLE_FRAMES)], "cyan")


class PrimerCallout(NamedTuple):
    """One observation a technical destination can make about its own current numbers."""

    topic: str
    """A short lead-in naming what the observation is about."""
    read: Callable[[WorkerStateSnapshot, HealthReport], str | None]
    """Build the plain-language sentence for this frame, or None when the observation does not apply."""


_FAULT_SHARE_MIN_ATTEMPTS = 20
"""Attempted requests before a fault share is quoted, so a single early failure does not read as a trend."""

_NOTABLE_FAULT_SHARE = 0.05
"""Share of attempts faulting that is worth pointing a contributor at the log for."""

_NOTHING_FINISHED_SECONDS = 15 * 60
"""How long accepted work may sit unfinished before that is worth saying; a first model load is slow."""

_MODEL_LOADING_STATES = frozenset({"PROCESS_STARTING", "DOWNLOADING_MODEL", "PRELOADING_MODEL"})
"""Child states meaning a background program is preparing a model rather than serving a request."""

_DEMAND_LIMITED_MIN_SECONDS = 10 * 60
"""Session length before the idle share is representative of what the horde is offering."""

_DEMAND_LIMITED_SHARE = 0.5
"""Share of the session spent with no work on offer that makes demand the thing to explain."""

_NOTABLE_SLOWDOWNS = 5
"""Recorded slowdowns before they read as a sustained mismatch rather than ordinary variation."""

_WORKING_PHASES = frozenset(
    {WorkerPhase.SERVING, WorkerPhase.READY, WorkerPhase.IDLE, WorkerPhase.DEGRADED},
)
"""Phases in which the worker is meant to be turning accepted work into submissions."""


def _session_seconds(snapshot: WorkerStateSnapshot) -> float:
    """Return how long the session has run, measured against the frame's own clock rather than the view's."""
    if not snapshot.session_start_time:
        return 0.0
    return max(0.0, snapshot.timestamp - snapshot.session_start_time)


def _fault_share(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str | None:
    """Report the share of attempted requests that faulted, once there have been enough to mean anything."""
    attempts = snapshot.num_jobs_submitted + snapshot.num_jobs_faulted
    if snapshot.num_jobs_faulted == 0 or attempts < _FAULT_SHARE_MIN_ATTEMPTS:
        return None
    share = snapshot.num_jobs_faulted / attempts
    if share < _NOTABLE_FAULT_SHARE:
        return None
    return (
        f"{snapshot.num_jobs_faulted:,} of {attempts:,} requests this session could not be finished "
        f"({share * 100:.0f}%). The Logs tab carries the reason each one gave."
    )


def _nothing_finished_yet(snapshot: WorkerStateSnapshot, report: HealthReport) -> str | None:
    """Report work accepted but never sent back, once that has gone on longer than a model load explains."""
    if snapshot.num_jobs_submitted or not snapshot.num_jobs_popped or report.phase not in _WORKING_PHASES:
        return None
    elapsed = _session_seconds(snapshot)
    if elapsed < _NOTHING_FINISHED_SECONDS:
        return None
    return (
        f"{snapshot.num_jobs_popped:,} requests have been accepted over {human_duration(elapsed)} and none "
        "have been sent back yet. The Live tab shows where they are sitting."
    )


def _process_restarts(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str | None:
    """Report that the worker has replaced background programs on its own."""
    if snapshot.num_process_recoveries <= 0:
        return None
    return (
        "The worker has replaced background programs that stopped responding, without being asked "
        f"(restarts this session: {snapshot.num_process_recoveries}). Nothing is needed now, and a count "
        "that keeps climbing is the figure to quote when asking for help."
    )


def _waiting_on_a_model(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str | None:
    """Explain a queue that nothing is working on while a program is still getting a model ready."""
    pending = snapshot.jobs_pending_inference
    if pending <= 0 or any(process.is_busy for process in snapshot.processes):
        return None
    if not any(process.last_process_state in _MODEL_LOADING_STATES for process in snapshot.processes):
        return None
    return (
        f"Requests are queued ({pending:,} waiting to start) and no program is holding one yet, because a "
        "model is still being made ready. The first use of a model takes a few minutes."
    )


def _consecutive_failures(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str | None:
    """Report requests failing back to back."""
    if snapshot.consecutive_failed_jobs <= 0:
        return None
    return (
        f"Requests are failing back to back (currently {snapshot.consecutive_failed_jobs} in a row). The "
        "count resets on the next success, so quote it as it stands when asking for help."
    )


def _error_backoff(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str | None:
    """Report that the worker has throttled its own calls to the horde."""
    if not snapshot.in_error_backoff:
        return None
    return (
        "The worker has slowed its calls to the horde after repeated errors. It clears this itself once a "
        "call succeeds, so give it a few minutes before changing anything."
    )


def _horde_notices(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str | None:
    """Report messages the horde delivered to this worker."""
    if not snapshot.api_messages:
        return None
    return (
        f"The horde has sent this worker messages (currently {len(snapshot.api_messages)}), shown in the log "
        "alongside the worker's own lines. They concern maintenance or your account rather than anything "
        "this computer did."
    )


def _demand_limited(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str | None:
    """Report a session dominated by waiting for the horde to offer work."""
    elapsed = _session_seconds(snapshot)
    if elapsed < _DEMAND_LIMITED_MIN_SECONDS:
        return None
    share = snapshot.time_spent_no_jobs_available / elapsed
    if share < _DEMAND_LIMITED_SHARE:
        return None
    return (
        f"{human_duration(snapshot.time_spent_no_jobs_available)} of this session ({share * 100:.0f}%) has "
        "gone by with the horde having nothing to send. That is demand, and no setting on this computer "
        "shortens it."
    )


def _slowdowns(snapshot: WorkerStateSnapshot, _report: HealthReport) -> str | None:
    """Report requests that ran slower than the horde was promised."""
    if snapshot.num_job_slowdowns < _NOTABLE_SLOWDOWNS:
        return None
    return (
        f"{snapshot.num_job_slowdowns:,} requests have run slower than the horde was promised. Offering "
        "fewer models, or taking fewer requests at once, is the usual way to bring that back into line."
    )


PRIMERS: dict[str, tuple[PrimerCallout, ...]] = {
    "Stats": (
        PrimerCallout("Faults", _fault_share),
        PrimerCallout("Nothing finished yet", _nothing_finished_yet),
    ),
    "Control": (
        PrimerCallout("Restarts", _process_restarts),
        PrimerCallout("Waiting on a model", _waiting_on_a_model),
    ),
    "Logs": (
        PrimerCallout("Failures in a row", _consecutive_failures),
        PrimerCallout("Backing off", _error_backoff),
        PrimerCallout("Notices from the horde", _horde_notices),
    ),
    "Insights": (
        PrimerCallout("Waiting for work", _demand_limited),
        PrimerCallout("Slowdowns", _slowdowns),
    ),
}
"""The observations each framed destination can make about its own live numbers, keyed by tab caption.

A destination is listed only where the snapshot carries something a contributor can act on or would
otherwise misread. Every callout is conditional, so a destination with nothing to say shows its framing
line alone; a figure that merely restates the operator widget's own label earns no entry here.
"""


class TabPrimer(Vertical):
    """A plain-language framing of a technical destination, plus what its numbers say right now.

    Shown only in Simple, above the operator widget it explains. The framing line is always visible.
    Beneath it sits a callout carrying only the observations that currently apply, each one a live
    sentence built from the snapshot's own figures. With nothing to report the callout is hidden
    outright, so an unremarkable worker leaves the framing line standing alone instead of an empty frame.

    The callout folds, because several observations at once run to more rows than an
    eighty-by-twenty-four terminal has to spare and would otherwise push the widget being explained off
    the bottom of the screen. It opens expanded, so an observation is read before it is dismissed.
    """

    _CALLOUT_CLASS = "tab-primer-callout"

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

    def __init__(self, intro: str, callouts: Sequence[PrimerCallout] = ()) -> None:
        """Frame a destination with ``intro`` and surface each of ``callouts`` while it applies."""
        super().__init__(classes="horde-tab-intro level-simple-only")
        self._intro = intro
        self._callouts = tuple(callouts)
        self._shown: tuple[tuple[str, str], ...] | None = None

    def compose(self) -> ComposeResult:
        """Lay out the always-visible framing line, then the foldable callout."""
        yield Static(Text(self._intro, "grey70"))
        if not self._callouts:
            return
        with Collapsible(title="What these numbers say right now", collapsed=False):
            yield Static(classes=self._CALLOUT_CLASS)

    def on_mount(self) -> None:
        """Settle the callout out of sight until worker state gives it something to say."""
        self.update_view(None, None)

    def update_view(self, snapshot: WorkerStateSnapshot | None, report: HealthReport | None) -> None:
        """Rebuild the callout from one frame of worker state, hiding it when no observation applies.

        The callout is mounted once and toggled, and an unchanged set of observations is left alone, so a
        condition coming and going does not rebuild widgets under the reader every tick.
        """
        if not self._callouts:
            return
        observed: tuple[tuple[str, str], ...] = ()
        if snapshot is not None and report is not None:
            observed = tuple(
                (callout.topic, sentence)
                for callout in self._callouts
                if (sentence := callout.read(snapshot, report)) is not None
            )
        if observed == self._shown:
            return
        self._shown = observed
        with contextlib.suppress(NoMatches):
            self.query_one(Collapsible).display = bool(observed)
            if observed:
                self.query_one(f".{self._CALLOUT_CLASS}", Static).update(self._build(observed))

    def _build(self, observed: Sequence[tuple[str, str]]) -> RenderableType:
        """Stack each applying observation as a lead-in over the sentence reading this frame's figures."""
        body = Table.grid()
        body.add_column()
        for index, (topic, sentence) in enumerate(observed):
            if index:
                body.add_row(Text(""))
            body.add_row(Text(topic, "bold cyan"))
            body.add_row(Text(sentence, "grey70"))
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
    /* The cards above grow as the session produces output (the feed filling, jobs starting), so the
       actions are pinned to the container's bottom edge rather than flowed after them: a click aimed at
       a button must not land on whatever card happened to expand a frame earlier. */
    SimpleHomeView #simple-actions {
        dock: bottom;
        height: auto;
    }
    Screen.-narrow SimpleHomeView .simple-headline {
        margin-right: 0;
    }
    """

    def __init__(self) -> None:
        """Start with empty trend history and no observations."""
        super().__init__()
        self._liveness = LivenessIndicator()
        self._kudos_history: deque[tuple[float, float, float]] = deque(maxlen=_TREND_MAX_SAMPLES)
        self._completed_history: deque[tuple[float, int]] = deque(maxlen=_TREND_MAX_SAMPLES)
        self._trend_epoch: float | None = None
        self._last_sample_at = 0.0
        self._ticker: deque[str] = deque(maxlen=_TICKER_LINES)
        # Insertion-ordered so the oldest identifier can be evicted; a plain set would grow for the life
        # of a session that runs for days.
        self._seen_job_ids: dict[str, None] = {}
        self._setup_required = False

    def compose(self) -> ComposeResult:
        """Lay out the setup prompt, status hero, attention card, headline figures, live progress, and feed.

        Attention sits directly under the hero: an off-nominal posture must be visible without scrolling,
        and everything below it (headlines, progress, the feed) degrades gracefully when pushed down.
        """
        yield Static(id="simple-setup", classes="horde-card")
        yield Static(id="simple-status", classes="horde-hero")
        yield Static(id="simple-attention", classes="horde-card")
        with Horizontal(id="simple-headlines"):
            yield Static(id="simple-requests", classes="horde-card simple-headline")
            yield Static(id="simple-kudos", classes="horde-card simple-headline")
        yield Static(id="simple-current", classes="horde-card")
        yield Static(id="simple-ticker", classes="horde-card")
        with Horizontal(id="simple-actions", classes="horde-actions"):
            yield Button("Start contributing", id="simple-start-stop", variant="success")
            yield Button("Getting started", id="simple-setup-action", variant="success")
            yield Button("Activity", id="simple-show-activity")
            yield Button("Model downloads", id="simple-show-models")

    def on_mount(self) -> None:
        """Hide the conditional cards until a state update decides they are needed."""
        self.set_setup_required(False, force=True)
        self.query_one("#simple-attention", Static).display = False

    def set_setup_required(self, required: bool, *, force: bool = False) -> None:
        """Switch between the setup prompt and the everyday home.

        The Getting started action stays in the row either way: it explains what the worker offers and
        holds the presets, which is worth revisiting long after setup is done.

        Returns early when nothing changed: the render loop calls this every frame, and re-rendering
        unchanged prompt text costs a repaint for no visible difference.
        """
        if required == self._setup_required and not force:
            return
        self._setup_required = required
        self.query_one("#simple-setup", Static).display = required
        self.query_one("#simple-setup-action", Button).variant = "success" if required else "default"
        self.query_one("#simple-start-stop", Button).display = not required
        if required:
            self.query_one("#simple-setup", Static).update(
                Text.assemble(
                    ("Set this worker up\n", "bold"),
                    (
                        "This computer needs an AI Horde API key, a worker name, and models on disk "
                        "before it can take requests. The Getting started guide explains what is needed "
                        "and sets it up.",
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
    ) -> None:
        """Refresh every card from one frame of worker state."""
        self._liveness.update(snapshot, is_alive=is_alive)
        self._record_trend(snapshot)
        self._record_events(snapshot)
        self._render_status(report, snapshot, is_alive=is_alive)
        self._render_headlines(snapshot)
        self._render_current(snapshot, is_alive=is_alive)
        self._render_attention(report, snapshot)
        self._render_ticker()
        action = self.query_one("#simple-start-stop", Button)
        action.label = "Stop contributing" if is_alive else "Start contributing"
        action.variant = "warning" if is_alive else "success"

    def _record_trend(self, snapshot: WorkerStateSnapshot | None) -> None:
        """Retain a bounded, timestamped history of the cumulative counters the trends take deltas of.

        The counters are stored as reported and differenced at render time. Charting the counters
        themselves would draw a line that only ever rises, so a worker that stopped finishing anything
        would keep showing the plateau it reached rather than the stall it is in.
        """
        if snapshot is None:
            return
        # Monotonic: this measures an elapsed gap, and a wall-clock step would either stall the trend or
        # flood it with samples. The sample itself is stamped with the worker's own wall clock, which is
        # what the trend window is expressed in.
        now = time.monotonic()
        if now - self._last_sample_at < _TREND_SAMPLE_MIN_SECONDS:
            return
        self._last_sample_at = now
        stamp = snapshot.timestamp or time.time()
        if self._trend_epoch is None:
            self._trend_epoch = stamp
        self._kudos_history.append((stamp, float(snapshot.kudos_this_session or 0.0), snapshot.eligible_seconds_total))
        self._completed_history.append((stamp, snapshot.num_jobs_submitted))

    def _completed_series(self) -> list[float]:
        """Return per-bucket completed-request deltas across the charted window."""
        _rate, deltas, _sampled_span = fixed_counter_deltas(
            list(self._completed_history),
            _TREND_WINDOW,
            epoch=self._trend_epoch,
            buckets=_TREND_BUCKETS,
        )
        return deltas

    def _kudos_series(self) -> list[float]:
        """Return per-bucket kudos deltas across the charted window."""
        _rate, deltas, _sampled_span = fixed_ratio_deltas(
            list(self._kudos_history),
            _TREND_WINDOW,
            epoch=self._trend_epoch,
            buckets=_TREND_BUCKETS,
        )
        return deltas

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
            # The reward is only known for a job the horde accepted and paid for, so a faulted line says
            # nothing about kudos rather than showing a zero the contributor would read as a bad payout.
            earned = f" (+{job.kudos_reward:,.1f} kudos)" if job.kudos_reward is not None else ""
            named = f" with {shorten(job.model_name, 24)}" if job.model_name else ""
            if job.faulted:
                self._ticker.append(f"Could not finish a request{elapsed}")
            elif job.is_alchemy:
                self._ticker.append(f"Finished an alchemy request{named}{elapsed}{earned}")
            else:
                self._ticker.append(f"Finished an image request{named}{elapsed}{earned}")

    def _render_status(
        self,
        report: HealthReport,
        snapshot: WorkerStateSnapshot | None,
        *,
        is_alive: bool,
    ) -> None:
        """Render the headline phase with the liveness marker beside it, over who this worker is.

        Identity and offered features ride the hero rather than a card of their own: the home view fills
        the terminal it is designed for, and a further card would push the action buttons off the bottom
        of it.
        """
        concerning = report.severity >= HealthStatus.WARN
        marker = self._liveness.marker(
            snapshot,
            is_alive=is_alive,
            concerning=concerning,
        )
        phase = _PHASE_WORDS.get(report.phase, report.headline)
        style = "bold red" if report.severity >= HealthStatus.ERROR else "bold green"
        body = Text.assemble(marker, ("  ", ""), (f"{phase}\n", style), (report.detail, "grey70"))
        if snapshot is not None:
            uptime = time.time() - snapshot.session_start_time if snapshot.session_start_time else 0.0
            body.append(f"\n{_identity_words(snapshot.config, uptime=uptime)}", "grey62")
            body.append(f"\nOffers: {_offer_words(snapshot.config)}. {_scale_words(snapshot.config)}.", "grey62")
        self.query_one("#simple-status", Static).update(body)

    def _render_headlines(self, snapshot: WorkerStateSnapshot | None) -> None:
        """Render the two headline totals, each over the recent pace that is producing them.

        The figure is the session total and the trend beneath it is a rate, so a worker that has stopped
        earning shows a large number over a flat line rather than a number over a line that still climbs.
        """
        completed = snapshot.num_jobs_submitted if snapshot is not None else 0
        kudos = (snapshot.kudos_this_session or 0.0) if snapshot is not None else 0.0
        self.query_one("#simple-requests", Static).update(
            Text.assemble(
                (f"{completed:,}\n", "bold cyan"),
                ("community requests completed\n", "grey70"),
                (sparkline(self._completed_series()), "cyan"),
            ),
        )
        kudos_per_hour = snapshot.kudos_per_hour if snapshot is not None else None
        pace = f"{kudos_per_hour:,.0f} an hour working" if kudos_per_hour is not None else "hourly rate not yet known"
        self.query_one("#simple-kudos", Static).update(
            Text.assemble(
                (f"{kudos:,.1f}\n", "bold magenta"),
                (f"kudos this session, {pace}\n", "grey70"),
                (sparkline(self._kudos_series()), "magenta"),
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

    def _render_attention(self, report: HealthReport, snapshot: WorkerStateSnapshot | None) -> None:
        """Show what is off-nominal, and nothing at all while everything is.

        Alongside the first concerning health check this carries the postures that explain a worker which
        is running and yet earning nothing: maintenance and pop backoff both stop new work arriving, and
        an absorbed process restart says the totals were interrupted. A nominal worker adds no line here,
        so the card's presence is itself the signal.
        """
        lines = [check.detail for check in report.checks if check.status >= HealthStatus.WARN][:1]
        if snapshot is not None:
            if snapshot.maintenance_mode:
                lines.append("Paused for maintenance, so no new requests are being taken.")
            if snapshot.in_error_backoff:
                lines.append("Waiting before asking for work again, after repeated trouble reaching the horde.")
            recoveries = snapshot.num_process_recoveries
            if recoveries:
                plural = "es" if recoveries != 1 else ""
                lines.append(f"Restarted {recoveries:,} stuck worker process{plural} this session.")
        card = self.query_one("#simple-attention", Static)
        if not lines:
            card.display = False
            return
        card.display = True
        body = Text("Needs attention\n", "bold yellow")
        for index, line in enumerate(lines):
            body.append(line if index == len(lines) - 1 else f"{line}\n", "grey70")
        card.update(body)

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
        table.add_column("Kudos", justify="right", no_wrap=True)
        for job in list(snapshot.recent_jobs)[-12:][::-1]:
            result = Text("Not completed", "yellow") if job.faulted else Text("Completed", "green")
            work = "Alchemy request" if job.is_alchemy else (job.model_name or "Image request")
            elapsed = f"{job.e2e_seconds:.1f}s" if job.e2e_seconds is not None else "-"
            earned = f"{job.kudos_reward:,.1f}" if job.kudos_reward is not None else "-"
            table.add_row(result, work, elapsed, earned)
        if not snapshot.recent_jobs:
            table.add_row(Text("Waiting", "grey62"), "No requests have finished yet", "-", "-")
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
    "PrimerCallout",
    "SimpleDestination",
    "SimpleActivityView",
    "SimpleHomeView",
    "SimpleModelStatusView",
    "job_progress_fraction",
    "TabPrimer",
    "titled",
]
