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
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    DownloadPhase,
    WorkerEvent,
    WorkerEventKind,
    WorkLedgerEntry,
    WorkLedgerStage,
)
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind
from horde_worker_regen.tui.formatters import (
    human_bytes,
    human_duration,
    is_low_fidelity,
    mini_bar,
    shorten,
    sparkline,
)
from horde_worker_regen.tui.health import (
    TEXT_BACKEND_CHECK_NAME,
    TEXT_GENERATION_CHECK_NAME,
    HealthReport,
    HealthStatus,
    WorkerPhase,
)
from horde_worker_regen.tui.trends import fixed_counter_deltas, fixed_ratio_deltas

if TYPE_CHECKING:
    from horde_worker_regen.process_management.ipc.supervisor_channel import (
        CurrentDownloadStatus,
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


def _transfer_detail(item: CurrentDownloadStatus) -> str:
    """Render one transfer's bytes, rate, and remaining time as a single plain-language line.

    These figures are what separate a slow download from a stopped one, which a bar alone cannot say, and
    they are the only reading at all for a source that never declared the file's size.
    """
    if item.total_bytes > 0:
        size = f"{human_bytes(item.downloaded_bytes)} of {human_bytes(item.total_bytes)}"
    else:
        size = f"{human_bytes(item.downloaded_bytes)} so far"
    rate = f"{human_bytes(item.speed_bps)}/s" if item.speed_bps else "measuring speed"
    if item.eta_seconds is None:
        return f"{size} - {rate}"
    return f"{size} - {rate} - about {human_duration(item.eta_seconds)} left"


def _download_impact(snapshot: WorkerStateSnapshot) -> str:
    """State whether the worker can serve requests while these downloads run.

    Telling a contributor the worker keeps contributing is only true once a model is loaded. Said during a
    first run, when nothing can be served until the downloads land, it turns a long wait into an apparent
    fault, so the claim is made only when a loaded model backs it.
    """
    serving = any(process.loaded_horde_model_name for process in snapshot.processes if not process.is_external)
    if serving:
        return "Models download in the background; the worker keeps contributing meanwhile."
    return "The worker starts contributing once the first of these finishes. Leaving it running is enough."


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


def _text_entries(snapshot: WorkerStateSnapshot) -> list[WorkLedgerEntry]:
    """Return the in-flight text generations on this snapshot, in the order the worker popped them."""
    active_stages = (WorkLedgerStage.QUEUED, WorkLedgerStage.INFERENCE, WorkLedgerStage.SUBMIT)
    return [
        entry
        for entry in snapshot.work_ledger
        if entry.workload is WorkloadKind.TEXT_GENERATION and entry.stage in active_stages
    ]


def _text_progress_total(snapshot: WorkerStateSnapshot) -> int:
    """Return the summed progress of every in-flight text generation, whatever each one counts.

    A counter for the liveness indicator, not a figure anyone reads: tokens and stream records are
    summed together because the only question asked of it is whether it moved since the last frame.
    """
    return sum(entry.progress_current or 0 for entry in _text_entries(snapshot))


_TEXT_POSTURE_WORDS: dict[str, str] = {
    TEXT_BACKEND_CHECK_NAME: (
        "The text program on this computer has not reported a loaded model, so no text requests are being taken."
    ),
    TEXT_GENERATION_CHECK_NAME: "A text request has stopped producing words and is being given up on.",
}
"""Simple's wording for each text finding, keyed by the health check's name.

A finding this view has no wording for is left to the general first-finding line, so a check added later
degrades to its own detail rather than disappearing."""


def _text_posture_words(report: HealthReport) -> list[str]:
    """Return Simple's wording for whichever text findings the health report raised."""
    return [
        _TEXT_POSTURE_WORDS[check.name]
        for check in report.checks
        if check.status >= HealthStatus.WARN and check.name in _TEXT_POSTURE_WORDS
    ]


def _text_request_progress(entry: WorkLedgerEntry) -> tuple[Text, Text]:
    """Return the bar and the figure beside it for one in-flight text generation.

    A generation whose backend counts tokens has a requested length to measure against and gets a real
    bar. One whose backend counts only stream records has arrival but no fraction, so it gets an
    indeterminate bar and the count in the unit it was counted in, rather than a percentage that would
    read as "this far through" on evidence that cannot say. The unit keeps the word the operator
    surfaces use, so a contributor who is promoted meets the same term rather than a second one.
    """
    waiting = entry.stage is WorkLedgerStage.QUEUED
    if waiting:
        return Text("waiting", "cyan"), Text("")
    current = entry.progress_current
    if entry.progress_total and current is not None:
        fraction = max(0.0, min(1.0, current / entry.progress_total))
        return Text(mini_bar(fraction, 16), "green"), Text(f"{fraction * 100:.0f}%", "bold")
    if current is None:
        return Text("working", "green"), Text("")
    unit = entry.progress_unit.value.removesuffix("s") if current == 1 else entry.progress_unit.value
    return Text("?" * 16, "grey50"), Text(f"{current:,} {unit}", "grey70")


def event_sentence(event: WorkerEvent) -> str | None:
    """Return the plain-language line for one worker event, or None for a kind the feed does not show.

    Shared with the native page so one worker transition reads the same wherever it is shown. Two kinds
    are deliberately unsaid: a pop, because one line per accepted request would crowd out every other
    kind, and a first backend launch, because the readiness line that follows it seconds later says the
    same thing with a duration. Both stay on the ring for the surfaces that list every transition.
    """
    match event.kind:
        case WorkerEventKind.JOB_FINISHED | WorkerEventKind.JOB_FAULTED:
            return _finished_request_words(event)
        case WorkerEventKind.PRELOAD_STARTED:
            return f"Started loading {shorten(event.model, 24)}" if event.model else "Started loading a model"
        case WorkerEventKind.PRELOAD_READY:
            return f"{shorten(event.model, 24)} is ready to serve" if event.model else "A model is ready to serve"
        case WorkerEventKind.BACKEND_READY:
            if event.duration_seconds is None:
                return "Text backend ready"
            return f"Text backend ready after {event.duration_seconds:.0f} s"
        case WorkerEventKind.BACKEND_RELAUNCHED:
            return f"Text backend restarted (launch {event.payload.count})"
        case WorkerEventKind.PROCESS_RECOVERED:
            if event.payload.device_index is None:
                # A whole service lane was rebuilt rather than one card's slot, so there is no card to name.
                return "Recovered a worker process"
            return f"Recovered the inference process on GPU {event.payload.device_index}"
        case WorkerEventKind.DOWNLOAD_FINISHED:
            return f"Finished downloading {shorten(event.model, 24)}" if event.model else "Finished a download"
        case WorkerEventKind.MAINTENANCE_ON:
            # A horde-side hold is refused per flow and carries that flow; the operator's own pause is
            # worker-wide and carries none, and must not be reported as the horde's doing.
            if event.workload is None:
                return "Paused on this computer, so no new requests are being taken"
            return "The horde is holding requests back (maintenance)"
        case WorkerEventKind.MAINTENANCE_OFF:
            if event.workload is None:
                return "Resumed on this computer; requests are flowing again"
            return "Requests are flowing again"
        case WorkerEventKind.POP_BACKOFF_ENTERED:
            return "Backing off from the horde after errors"
        case WorkerEventKind.POP_BACKOFF_LEFT:
            return "Back to normal polling"
        case WorkerEventKind.JOB_POPPED | WorkerEventKind.BACKEND_LAUNCHED:
            return None


def _finished_request_words(event: WorkerEvent) -> str:
    """Return the feed's line for one finished or faulted request, in the words of its workload."""
    elapsed = f" in {event.duration_seconds:.0f}s" if event.duration_seconds is not None else ""
    # The reward is only known for a job the horde accepted and paid for, so a faulted line says
    # nothing about kudos rather than showing a zero the contributor would read as a bad payout.
    earned = f" (+{event.kudos:,.1f} kudos)" if event.kudos is not None else ""
    named = f" with {shorten(event.model, 24)}" if event.model else ""
    if event.kind is WorkerEventKind.JOB_FAULTED:
        return f"Could not finish a request{elapsed}"
    if event.workload is WorkloadKind.ALCHEMY:
        return f"Finished an alchemy request{named}{elapsed}{earned}"
    if event.workload is WorkloadKind.TEXT_GENERATION:
        # A text request has no resolution or step count to say how much work it was; the token
        # count is the only figure that does, and a backend that counts none leaves it out.
        tokens = f", {event.payload.count:,} tokens" if event.payload.count is not None else ""
        return f"Finished a text request{named}{tokens}{elapsed}{earned}"
    return f"Finished an image request{named}{elapsed}{earned}"


def _text_backend_row(snapshot: WorkerStateSnapshot) -> ProcessSnapshot | None:
    """Return the supervised text backend's row, or None when the worker has no backend row yet."""
    return next((process for process in snapshot.processes if process.text_backend is not None), None)


def _posture_words(snapshot: WorkerStateSnapshot | None, *, connected: bool) -> str:
    """State this dashboard's contact with the worker, when it last asked for work, and what is loaded.

    Unconditional, unlike the attention card: a contributor watching a worker that earns nothing needs the
    three facts that separate "nothing to do" from "not asking" and "nothing loaded to serve with", and a
    line that appears only when something is wrong cannot be checked against a healthy worker.
    """
    parts = ["In contact with the worker" if connected else "Not in contact with the worker"]
    if snapshot is None:
        return parts[0]
    since_pop = snapshot.seconds_since_last_pop
    parts.append(
        f"last asked the horde {human_duration(since_pop)} ago" if since_pop is not None else "never asked yet",
    )
    config = snapshot.config
    if config.dreamer:
        loaded = sum(
            1 for process in snapshot.processes if not process.is_external and process.loaded_horde_model_name
        )
        parts.append(f"{loaded} of {config.num_models} models loaded")
    if config.scribe:
        row = _text_backend_row(snapshot)
        state = row.display_state if row is not None and row.display_state else "not started"
        parts.append(f"text backend {state}")
    return " · ".join(parts)


def _has_work_in_hand(snapshot: WorkerStateSnapshot) -> bool:
    """Whether the worker is holding work of any kind, which decides what may prove it is alive."""
    return snapshot.jobs_in_progress > 0 or snapshot.text_jobs_in_flight > 0


def _identity_words(config: WorkerConfigSummary, *, uptime: float) -> str:
    """Name this worker, the build it is running, and who it is contributing for.

    Only the enabled roles are named: a name the worker never registers is not its identity, and the
    defaults the unselected roles carry would otherwise be read as workers a contributor is running.
    """
    line = f"{config.worker_display_name or 'Unnamed worker'}, version {config.worker_version}"
    if config.horde_username:
        line += f", contributing as {config.horde_username}"
        return f"{line} for {human_duration(uptime)}" if uptime > 0 else line
    return f"{line}, contributing for {human_duration(uptime)}" if uptime > 0 else line


def _offer_words(config: WorkerConfigSummary, *, text_model_name: str | None = None) -> str:
    """Name what this worker takes on, in the words a requester would recognise.

    The image offers are gated on the dreamer role: the flags behind them default on, so a worker that
    serves only text or only alchemy would otherwise claim image work it never pops.

    LoRA follows the effective setting rather than the configured one, so a worker whose pops are not
    currently advertising LoRA does not claim it. Text generation names the model the backend loaded
    where it has answered, because "text generation" alone does not tell a contributor which of their
    models the requests are being served by.
    """
    allows_lora = config.allow_lora if config.effective_allow_lora is None else config.effective_allow_lora
    offers: list[str] = []
    if config.dreamer:
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
    if config.scribe:
        offers.append(
            f"text generation with {shorten(text_model_name, 32)}" if text_model_name else "text generation",
        )
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

    A text generation has no child of its own: it runs in a separate program, and what it produces
    arrives as a stream the flow counts. That count is the same kind of signal as a sampling counter and
    is read the same way, so a scribe worker with a generation in hand animates on the text arriving and
    not on the supervisor still stamping snapshots over a backend that stopped.

    Whether a stall amounts to a fault is settled by
    [`derive`][horde_worker_regen.tui.health.derive], against tuned, download-aware thresholds the whole
    dashboard shares. This class supplies the animation and takes the verdict from the health report.
    """

    def __init__(self) -> None:
        """Start with no observations."""
        self._phase = 0
        self._last_steps: int | None = None
        self._last_child_heartbeat: float | None = None
        self._last_text_progress: int | None = None
        self._last_timestamp: float | None = None

    def update(self, snapshot: WorkerStateSnapshot | None, *, is_alive: bool) -> None:
        """Fold one frame of worker state into the indicator."""
        if snapshot is None or not is_alive:
            self._last_steps = None
            self._last_child_heartbeat = None
            self._last_text_progress = None
            self._last_timestamp = None
            return
        children = [process for process in snapshot.processes if not process.is_external]
        steps = sum(process.heartbeats_inference_steps for process in children)
        child_heartbeat = max((process.last_heartbeat_timestamp for process in children), default=0.0)
        text_progress = _text_progress_total(snapshot)
        working = _has_work_in_hand(snapshot)
        sampled = self._last_steps is not None and steps != self._last_steps
        child_reported = self._last_child_heartbeat is not None and child_heartbeat != self._last_child_heartbeat
        text_arrived = self._last_text_progress is not None and text_progress != self._last_text_progress
        supervisor_reported = self._last_timestamp is not None and snapshot.timestamp != self._last_timestamp
        advanced = (sampled or child_reported or text_arrived) if working else supervisor_reported
        if advanced:
            self._phase += 1
        self._last_steps = steps
        self._last_child_heartbeat = child_heartbeat
        self._last_text_progress = text_progress
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
        if _has_work_in_hand(snapshot):
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
    inference_processes = [process for process in snapshot.processes if not process.is_external]
    if pending <= 0 or any(process.is_busy for process in inference_processes):
        return None
    if not any(process.last_process_state in _MODEL_LOADING_STATES for process in inference_processes):
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
        self._backfilled_session_start: float | None = None
        self._last_sample_at = 0.0
        self._ticker: deque[str] = deque(maxlen=_TICKER_LINES)
        # The highest event sequence this feed has shown. One integer rather than a set of identifiers:
        # the worker stamps its ring monotonically, so everything at or below this has been seen.
        self._highest_event_sequence = 0
        self._backend_ready_seconds: float | None = None
        """How long the text backend's last completed launch took, or None before one completed."""
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
        connected: bool = True,
    ) -> None:
        """Refresh every card from one frame of worker state.

        ``connected`` is whether this dashboard still has its channel to the worker, which only a session
        attached to a worker host can lose; a dashboard that owns the worker process is always in contact
        with it, which is the default.
        """
        self._liveness.update(snapshot, is_alive=is_alive)
        self._record_trend(snapshot)
        self._record_events(snapshot)
        self._render_status(report, snapshot, is_alive=is_alive, connected=connected)
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
        self._restore_worker_trends(snapshot)
        # Monotonic: this measures an elapsed gap, and a wall-clock step would either stall the trend or
        # flood it with samples. The sample itself is stamped with the worker's own wall clock, which is
        # what the trend window is expressed in.
        now = time.monotonic()
        if now - self._last_sample_at < _TREND_SAMPLE_MIN_SECONDS:
            return
        self._last_sample_at = now
        stamp = snapshot.timestamp or time.time()
        if self._trend_epoch is None:
            self._trend_epoch = snapshot.session_start_time or stamp
        if not self._completed_history or stamp > self._completed_history[-1][0]:
            self._kudos_history.append(
                (stamp, float(snapshot.kudos_this_session or 0.0), snapshot.eligible_seconds_total)
            )
            self._completed_history.append((stamp, snapshot.num_jobs_submitted))

    def _restore_worker_trends(self, snapshot: WorkerStateSnapshot) -> None:
        """Restore Home's pace charts from worker-owned samples after a browser reconnect."""
        session_start = snapshot.session_start_time
        if not session_start or session_start == self._backfilled_session_start:
            return
        self._kudos_history.clear()
        self._completed_history.clear()
        self._ticker.clear()
        # A different worker session numbers its ring from the start again, so the highest sequence this
        # feed showed belongs to the session it just left.
        self._highest_event_sequence = 0
        self._backend_ready_seconds = None
        self._trend_epoch = session_start
        self._backfilled_session_start = session_start

        backfill = snapshot.stats_history_backfill
        if backfill is None:
            return
        samples_by_timestamp = {
            sample.timestamp: sample for sample in (*backfill.all_session_samples, *backfill.recent_samples)
        }
        for sample in sorted(samples_by_timestamp.values(), key=lambda item: item.timestamp):
            self._kudos_history.append(
                (sample.timestamp, float(sample.kudos_this_session or 0.0), sample.eligible_seconds_total)
            )
            self._completed_history.append((sample.timestamp, sample.jobs_submitted))

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
        """Append the worker transitions this feed has not shown yet, newest last.

        Deduplicated by the event's own sequence rather than by job id: the ring carries kinds that belong
        to no job, and the sequence is what a reconnecting dashboard can compare against what it has
        already shown. A snapshot whose lowest sequence is above the highest shown proves the ring evicted
        events unseen, and the feed then shows what did arrive rather than inventing the gap.
        """
        if snapshot is None:
            return
        for event in snapshot.recent_events:
            if event.sequence <= self._highest_event_sequence:
                continue
            self._highest_event_sequence = event.sequence
            if event.kind is WorkerEventKind.BACKEND_READY and event.duration_seconds is not None:
                # How long the backend took to come up last time is the only honest estimate of how long
                # the next launch will take, and the row cannot supply it: its launch detail is cleared at
                # the start of every attempt.
                self._backend_ready_seconds = event.duration_seconds
            sentence = event_sentence(event)
            if sentence is not None:
                self._ticker.append(sentence)

    def _render_status(
        self,
        report: HealthReport,
        snapshot: WorkerStateSnapshot | None,
        *,
        is_alive: bool,
        connected: bool,
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
        body.append(f"\n{_posture_words(snapshot, connected=connected)}", "grey70")
        if snapshot is not None:
            uptime = time.time() - snapshot.session_start_time if snapshot.session_start_time else 0.0
            body.append(f"\n{_identity_words(snapshot.config, uptime=uptime)}", "grey62")
            offers = _offer_words(snapshot.config, text_model_name=snapshot.text_model_name)
            body.append(f"\nOffers: {offers}. {_scale_words(snapshot.config)}.", "grey62")
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
        """Render a real progress bar per request in flight, image and text alike.

        A text generation has no process of its own, so its rows come from the work ledger rather than
        from the process list; both kinds sit in one card because a contributor watching it is asking
        what their machine is doing, not which subsystem is doing it.
        """
        card = self.query_one("#simple-current", Static)
        if snapshot is None or not is_alive:
            card.update(titled("Right now", Text("Nothing running yet.", "grey70")))
            return
        busy = [process for process in snapshot.processes if process.is_busy and process.current_job_id]
        text_entries = _text_entries(snapshot)
        if not busy and not text_entries:
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
        for entry in text_entries:
            bar, detail = _text_request_progress(entry)
            table.add_row(bar, Text(shorten(entry.model, 32) or "a text request", "grey70"), detail)
        card.update(titled("Right now", table))

    def _idle_detail(self, snapshot: WorkerStateSnapshot) -> str:
        """Explain what the worker is waiting for, one sentence per workload it serves.

        A worker serving two workloads is idle for two reasons at once, and a single sentence has to pick
        one of them, which leaves the other flow unexplained. Maintenance is the exception: it holds every
        flow's pops, so it is said once instead of per workload.
        """
        if snapshot.maintenance_mode:
            return "Maintenance is holding requests back."
        config = snapshot.config
        sentences: list[str] = []
        if config.dreamer:
            sentences.append(self._idle_image_words(snapshot))
        if config.alchemist:
            sentences.append("No alchemy requests are waiting.")
        if config.scribe:
            sentences.append(self._idle_text_words(snapshot))
        if not sentences:
            return "Waiting for a community request."
        return " ".join(sentences)

    @staticmethod
    def _idle_image_words(snapshot: WorkerStateSnapshot) -> str:
        """Say what image generation is waiting for: its own downloads, its queue, or the horde.

        Reads the download phase rather than merely testing it against idle, so a paused or failed
        download is never described as one in progress, and names the file in flight where the subsystem
        reports one: "downloading" for an hour says nothing about whether it is progressing.
        """
        pending = snapshot.jobs_pending_inference
        if pending:
            return f"Preparing {pending} request{'s' if pending != 1 else ''}."
        downloads = snapshot.downloads
        if downloads is not None:
            if downloads.phase is DownloadPhase.DOWNLOADING and downloads.current is not None:
                return f"Waiting for the download of {shorten(downloads.current.model_name, 32)}."
            detail = _DOWNLOAD_PHASE_DETAIL.get(downloads.phase)
            if detail is not None:
                return detail
        return "No image requests are waiting for your models."

    def _idle_text_words(self, snapshot: WorkerStateSnapshot) -> str:
        """Say what text generation is waiting for: its backend to come up, or the horde.

        The estimate is how long the backend's last completed launch took, which the feed remembers from
        the worker's own readiness event. Before one has completed there is no estimate, and the sentence
        says so rather than naming a number nothing supports.
        """
        if snapshot.text_backend_ready:
            return "No text requests are waiting for your model."
        if self._backend_ready_seconds is None:
            return "The text backend is starting."
        return f"The text backend is starting, ready in about {self._backend_ready_seconds:.0f} s."

    def _render_attention(self, report: HealthReport, snapshot: WorkerStateSnapshot | None) -> None:
        """Show what is off-nominal, and nothing at all while everything is.

        Alongside the first concerning health check this carries the postures that explain a worker which
        is running and yet earning nothing: maintenance and pop backoff both stop new work arriving, and
        an absorbed process restart says the totals were interrupted. A nominal worker adds no line here,
        so the card's presence is itself the signal.

        The two text postures are reworded from the health report's own findings rather than decided
        again here, so Simple and the rest of the dashboard cannot disagree about whether a scribe is in
        trouble. They are added beside the first finding rather than competing for its single slot: a
        scribe worker whose backend never started has nothing else to say, and a dreamer-and-scribe one
        would otherwise lose the text finding to any image one.
        """
        lines = [check.detail for check in report.checks if check.status >= HealthStatus.WARN][:1]
        lines.extend(_text_posture_words(report))
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
            if job.workload == WorkloadKind.ALCHEMY:
                work = "Alchemy request"
            elif job.model_name:
                work = job.model_name
            elif job.workload == WorkloadKind.TEXT_GENERATION:
                work = "Text request"
            else:
                work = "Image request"
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
        """Lay out any problem banner, the readiness summary, download progress, and the config link."""
        # Bordered and padded, so it must start hidden or an empty card sits above the page whenever
        # nothing is wrong (which is nearly always).
        problem = Static(id="simple-models-problem", classes="horde-card horde-problem")
        problem.display = False
        yield problem
        yield Static("Models and downloads", classes="horde-hero horde-card-title")
        yield Static(id="simple-models-state", classes="horde-card")
        yield Static(id="simple-models-downloads", classes="horde-card")
        with Horizontal(classes="horde-actions"):
            yield Button("Change which models to run", id="simple-models-manage", variant="primary")

    def update_view(self, snapshot: WorkerStateSnapshot | None) -> None:
        """Refresh the problem banner, readiness, and download progress."""
        problem = self.query_one("#simple-models-problem", Static)
        state = self.query_one("#simple-models-state", Static)
        downloads = self.query_one("#simple-models-downloads", Static)
        if snapshot is None:
            problem.update(Text(""))
            problem.display = False
            state.update(Text("Model status appears once the worker starts.", "grey70"))
            downloads.update(Text(""))
            return
        banner = self._render_problem(snapshot)
        problem.display = banner is not None
        problem.update(banner if banner is not None else Text(""))
        state.update(titled("Model status", self._render_state(snapshot)))
        downloads.update(titled("Downloads", self._render_downloads(snapshot)))

    @staticmethod
    def _render_problem(snapshot: WorkerStateSnapshot) -> RenderableType | None:
        """Render the banner for a download that has gone wrong, or None when nothing has.

        A first-time contributor reads this page to answer one question: is this working or is it stuck?
        Failures buried as a count in the readiness grid answer neither, so anything that stops a model
        arriving is lifted to the top of the page with its reason and what it costs.
        """
        activity = snapshot.downloads
        if activity is None:
            return None
        lines: list[Text] = []
        if activity.phase is DownloadPhase.ERROR:
            detail = activity.error_message or "The download service stopped and cannot fetch models."
            lines.append(Text(detail, "red"))
        for failure in activity.failures:
            line = Text()
            line.append(failure.model_name, "bold red")
            # The coarse downloads (the safety models among them) label the whole task with its feature
            # name, so naming both would read as a stutter.
            if failure.feature != failure.model_name:
                line.append(f" ({failure.feature})", "red")
            line.append(" could not be downloaded: ", "red")
            line.append(failure.reason, "grey70")
            lines.append(line)
        if not lines:
            return None

        body = Table.grid()
        body.add_column()
        for line in lines:
            body.add_row(line)
        body.add_row(
            Text(
                "\nThe worker keeps retrying. If this does not clear, check that this computer can reach "
                "the internet and has free disk space, then restart the worker.",
                "grey70",
            ),
        )
        return titled("Something needs attention", body)

    @staticmethod
    def _render_state(snapshot: WorkerStateSnapshot) -> RenderableType:
        """Summarise which configured models are loaded, on disk, fetching, or failed."""
        loaded = sorted(
            {
                process.loaded_horde_model_name
                for process in snapshot.processes
                if process.loaded_horde_model_name and not process.is_external
            },
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
            names.append("  (the reason is at the top of this page)", "grey62")
            body.add_row(names)
        return body

    @staticmethod
    def _render_downloads(snapshot: WorkerStateSnapshot) -> RenderableType:
        """Render live download progress, or say plainly what the download service is doing instead.

        Every in-flight transfer gets a bar, the bytes behind it, its rate and its estimate. Those figures
        are what separate "slow" from "stuck": a bar alone cannot, and a model whose size the source never
        declared has no bar at all, which is precisely when the byte counter has to carry the answer.
        """
        activity = snapshot.downloads
        if activity is None:
            return Text("This worker does not manage model downloads.", "grey70")

        wrapper = Table.grid()
        wrapper.add_column()

        if not activity.active:
            detail = _DOWNLOAD_PHASE_DETAIL.get(activity.phase)
            wrapper.add_row(Text(detail or "Nothing is downloading right now.", "grey70"))
            if activity.pending:
                wrapper.add_row(Text(f"\n{len(activity.pending)} more queued to download.", "grey62"))
            return wrapper

        table = Table.grid(padding=(0, 2))
        table.add_column(no_wrap=True)
        table.add_column()
        table.add_column(justify="right", no_wrap=True)
        for item in activity.active:
            percent = item.percent
            bar = Text(mini_bar(percent / 100.0, 16), "yellow") if percent is not None else Text("fetching", "yellow")
            share = f"{percent:.0f}%" if percent is not None else "size unknown"
            table.add_row(bar, Text(item.model_name, "grey70"), Text(share, "bold"))
            table.add_row(Text(""), Text(_transfer_detail(item), "grey62"), Text(""))

        wrapper.add_row(table)
        if activity.paused:
            wrapper.add_row(Text("\nDownloads are paused.", "yellow"))
        if activity.pending:
            wrapper.add_row(Text(f"\n{len(activity.pending)} more queued after these.", "grey62"))
        wrapper.add_row(Text("\n" + _download_impact(snapshot), "grey70"))
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
