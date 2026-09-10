"""One parse of a session's per-job lifecycle, lane placement, and status census, shared by detectors.

A job's wall-clock time splits into three segments the worker logs separately: pop to dispatch (the
scheduler's own queue wait), dispatch to inference-finished (generation), and inference-finished to
submit (safety, upload, submission). Judging a slow worker requires all three, because the remedies are
opposite: a long generation segment is a GPU/config problem, a long pre-inference segment is a
scheduling problem, and a long post-inference segment is a pipeline-balance problem. A detector that
sees only one of them will attribute the wait to whatever it happens to measure.

Every detector that needs job-shaped facts reads :class:`JobLifecycleModel` through
:func:`job_lifecycle_for`, which parses once per :class:`~horde_worker_regen.analysis.correlate.SessionContext`
and caches on it, rather than regexing job lines itself. Alongside the per-job records the model
carries the facts that give those records meaning: which card each inference lane sits on (including
re-spawns and migrations), how many cards the worker drives and what its intake budget is, a parsed
series of the ~20s status blocks (each lane's state and model, plus the pending queue), the line-skip
census, the model preload/unload movement counts, and the timestamps of the parent's IPC drains.

Parsing is regex over the orchestrator's own message text and degrades rather than raising: a truncated
capture that begins mid-run yields jobs with no ``popped_at``, a log without the "Driving N cards" line
yields ``card_count is None``, and either simply narrows what the derived views can say.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from .correlate import SessionContext
from .log_ingest import LogRecord
from .log_signatures import pattern_for

# Every pattern below is registered in :mod:`log_signatures` with the worker function that emits the
# line and a literal sample; the registry is what makes a reworded emit fail a test rather than silently
# retire a parser. Add a new parser there first, never here.
_POP_RE = pattern_for("popped_job")
_JOB_QUEUE_LINE_RE = pattern_for("job_queue")
_DISPATCH_RE = pattern_for("inference_dispatched")
_INFERENCE_FINISHED_RE = pattern_for("inference_finished")
_SAFETY_RE = pattern_for("safety_checked")
_SUBMIT_RE = pattern_for("submitted_generation")
_JOB_FAULTED_RE = pattern_for("job_faulted_on_process")
_FAULT_REPORTED_RE = pattern_for("fault_reported")
_BATCH_IDS_RE = pattern_for("batch_ids")
_BATCH_ID_ENTRY_RE = pattern_for("batch_id_entry")
_LINE_SKIP_RE = pattern_for("line_skip")

_INFERENCE_LANE_RE = pattern_for("inference_lane_started")
_POST_PROCESS_LANE_RE = pattern_for("post_process_lane_started")
_UTILITIES_LANE_RE = pattern_for("utilities_lane_started")
_SAFETY_LANE_RE = pattern_for("safety_lane_started")

_DRIVING_CARDS_RE = pattern_for("driving_cards")
_PER_CARD_INTAKE_RE = pattern_for("per_card_intake")
_CARD_CONCURRENCY_RE = pattern_for("per_card_concurrency")
_CARD_CONCURRENCY_ENTRY_RE = pattern_for("per_card_concurrency_entry")

_PRELOAD_RE = pattern_for("model_preloading")
_UNLOAD_RE = pattern_for("model_unloaded")
_CLEARED_PRELOAD_RE = pattern_for("preload_cleared")
_DISPLACED_EXPIRY_RE = pattern_for("displaced_entry_expired")
_LOADING_RE = pattern_for("model_loading")
_MOVED_TO_RAM_RE = pattern_for("model_moved_to_system_ram")

_STATUS_HEADER_RE = pattern_for("status_header")
_STATUS_INFERENCE_LANE_RE = pattern_for("status_inference_lane")
_STATUS_AUX_LANE_RE = pattern_for("status_aux_lane")
_SAMPLING_STATE_RE = pattern_for("status_sampling_state")
_JOBS_LINE_RE = pattern_for("status_jobs")
_QUEUE_ENTRY_RE = pattern_for("queue_entry")

_IPC_DRAIN_RE = pattern_for("ipc_drain")

_SHORT_ID_LENGTH = 8
"""The worker truncates job ids to eight hex characters in dispatch, finish and submit lines, so that
truncation is the only join key available across a job's whole lifecycle."""

_IDLE_LANE_STATE = "WAITING_FOR_JOB"


def short_job_id(job_id: str) -> str:
    """Return the eight-character form the worker uses to key a job across its lifecycle lines."""
    return job_id[:_SHORT_ID_LENGTH]


class LaneRole(StrEnum):
    """What a worker child process does, as the parent's spawn line declares it."""

    INFERENCE = "inference"
    SAFETY = "safety"
    POST_PROCESS = "post_process"
    UTILITIES = "utilities"


@dataclass(frozen=True)
class LanePlacement:
    """Represents one spawn of a child process onto a card: the role it plays and where it landed.

    ``device_index`` is None when the spawn line carries no card (the safety lane's does not) or when a
    truncated capture never showed the spawn.
    """

    process_id: int
    role: LaneRole
    device_index: int | None
    started_at: datetime | None


@dataclass(frozen=True)
class LaneMigration:
    """Represents an auxiliary lane re-spawning onto a different card than the one it previously held."""

    process_id: int
    role: LaneRole
    from_device_index: int | None
    to_device_index: int | None
    timestamp: datetime | None


@dataclass(frozen=True)
class CardIntake:
    """Represents one card's share of the worker-wide intake budget, as the boot summary states it."""

    device_index: int
    process_count: int
    intake: int


@dataclass
class JobRecord:
    """Represents one job's passage through the worker, keyed by its eight-character short id.

    Every timestamp is optional: a capture that begins mid-run has jobs whose pop is off the front of the
    log, and a session that ends mid-flight has jobs that never finished. The three wait segments are
    None whenever either of their bounds is.
    """

    job_id: str
    full_job_id: str | None = None
    model: str | None = None
    popped_at: datetime | None = None
    effective_megapixelsteps: int | None = None
    batch: int | None = None
    uses_loras: bool | None = None
    uses_post_processing: bool | None = None
    dispatched_at: datetime | None = None
    process_id: int | None = None
    device_index: int | None = None
    inference_finished_at: datetime | None = None
    generation_seconds: float | None = None
    safety_checked_at: datetime | None = None
    safety_seconds: float | None = None
    submitted_at: datetime | None = None
    faulted_at: datetime | None = None
    fault_reported: bool = False
    line_skipped: bool = False
    """Whether this job was seated ahead of the queue head at least once."""
    batch_lead_job_id: str | None = None
    """The job whose dispatch line covered this one, when several jobs ran as one batched inference.

    The worker names only the lead job in its dispatch and finish lines and lists the rest in the batch
    line that follows, so a batch member's timings are the lead's and only the batch line joins them."""

    @property
    def pre_inference_seconds(self) -> float | None:
        """Seconds from pop to dispatch: the scheduler's own queue wait."""
        if self.popped_at is None or self.dispatched_at is None:
            return None
        return (self.dispatched_at - self.popped_at).total_seconds()

    @property
    def inference_seconds(self) -> float | None:
        """Seconds spent generating, as the child reported it (falling back to the parent's own span)."""
        if self.generation_seconds is not None:
            return self.generation_seconds
        if self.dispatched_at is None or self.inference_finished_at is None:
            return None
        return (self.inference_finished_at - self.dispatched_at).total_seconds()

    @property
    def post_inference_seconds(self) -> float | None:
        """Seconds from inference finishing to submission: safety, upload and the submit call."""
        if self.inference_finished_at is None or self.submitted_at is None:
            return None
        return (self.submitted_at - self.inference_finished_at).total_seconds()

    @property
    def faulted(self) -> bool:
        """Whether the job faulted at any point, whether or not the fault reached the horde."""
        return self.faulted_at is not None or self.fault_reported

    @property
    def faulted_before_dispatch(self) -> bool:
        """Whether the job faulted without ever having been dispatched to an inference lane."""
        return self.faulted and self.dispatched_at is None


@dataclass(frozen=True)
class LaneStatus:
    """Represents one lane's line within a single status print."""

    process_id: int
    role: LaneRole
    state: str
    model: str | None
    sampling: bool
    """Whether the lane's state is the per-step sampling readout, i.e. it is actively generating."""

    @property
    def idle(self) -> bool:
        """Whether the lane is sitting in WAITING_FOR_JOB (available to be seated, not starting up)."""
        return self.state == _IDLE_LANE_STATE


@dataclass(frozen=True)
class PendingJob:
    """Represents one entry of the pending-inference queue as a status print lists it."""

    job_id: str
    model: str


@dataclass(frozen=True)
class StatusSnapshot:
    """Represents one ~20s status print: every lane's state and model, plus the pending queue."""

    timestamp: datetime | None
    lanes: tuple[LaneStatus, ...]
    pending_jobs: tuple[PendingJob, ...]

    @property
    def inference_lanes(self) -> tuple[LaneStatus, ...]:
        """The inference lanes only (the auxiliary lanes hold no queued job)."""
        return tuple(lane for lane in self.lanes if lane.role is LaneRole.INFERENCE)


@dataclass(frozen=True)
class LineSkipEvent:
    """Represents one job seated ahead of the queue head, and the scheduler's reason token for it."""

    timestamp: datetime | None
    job_id: str
    reason: str
    process_id: int
    displaced_job_id: str


@dataclass(frozen=True)
class ModelMovement:
    """Represents how much model loading a session did, as counted from the parent's own lines."""

    preloads: int = 0
    unloads: int = 0
    cleared_preloads: int = 0
    displaced_expiries: int = 0
    loads: int = 0
    moved_to_system_ram: int = 0


@dataclass(frozen=True)
class WaitSegment:
    """Represents the distribution of one lifecycle wait segment across a session's jobs."""

    name: str
    count: int
    median: float | None
    p90: float | None
    total: float


@dataclass(frozen=True)
class ConcurrencyProfile:
    """Represents how many cards held an in-flight sampling job, weighted by the time each state held.

    Built by sweeping every job's dispatch-to-finish interval onto the card its lane sits on, so it
    measures the fleet's actual parallelism rather than the number of lanes that exist.
    """

    seconds_by_card_count: dict[int, float]
    observed_seconds: float

    @property
    def mean_cards_busy(self) -> float | None:
        """Time-weighted mean number of distinct cards with an in-flight job, or None with no coverage."""
        if self.observed_seconds <= 0:
            return None
        return sum(count * seconds for count, seconds in self.seconds_by_card_count.items()) / self.observed_seconds

    def share_at_or_below(self, card_count: int) -> float:
        """The fraction of observed time during which at most ``card_count`` cards were busy."""
        if self.observed_seconds <= 0:
            return 0.0
        matching = sum(
            seconds for busy_cards, seconds in self.seconds_by_card_count.items() if busy_cards <= card_count
        )
        return matching / self.observed_seconds

    def describe(self, *, top: int = 6) -> str:
        """A compact histogram line, busiest states last: ``0 cards 12% | 1 card 48% | ...``."""
        if self.observed_seconds <= 0:
            return "no in-flight coverage"
        ordered = sorted(self.seconds_by_card_count.items())[:top]
        parts = [
            f"{busy} card{'s' if busy != 1 else ''} {100 * seconds / self.observed_seconds:.0f}%"
            for busy, seconds in ordered
            if seconds > 0
        ]
        return " | ".join(parts) if parts else "no in-flight coverage"


@dataclass(frozen=True)
class HeadStateCensus:
    """Represents why the pending queue's head was not running, counted over the status prints.

    ``resident_but_blocked`` is the signature of a dispatch that has stopped spreading across cards: the
    head's weights are already on a lane, but every lane holding them is on a card that is busy, so the
    job waits behind work the fleet had capacity to run elsewhere.
    """

    seatable: int = 0
    resident_but_blocked: int = 0
    cold: int = 0

    @property
    def total(self) -> int:
        """How many status prints had a pending head to classify."""
        return self.seatable + self.resident_but_blocked + self.cold


@dataclass(frozen=True)
class StatusCensus:
    """Represents the per-print seating census: what was queued against what was idle.

    A "seatable" pending job is one whose model is already resident on an idle lane sitting on a card
    with no sampling job: the worker could have started it that instant without loading anything.
    """

    snapshots: int = 0
    median_pending: float = 0.0
    median_idle_lanes: float = 0.0
    median_seatable_pending: float = 0.0
    max_seatable_pending: int = 0
    head_states: HeadStateCensus = field(default_factory=HeadStateCensus)


@dataclass(frozen=True)
class IpcDrainProfile:
    """Represents the cadence of the parent's IPC drain, and the longest silences in it."""

    drains: int
    median_gap_seconds: float | None
    gaps_over_threshold: tuple[tuple[datetime, float], ...]
    """``(gap end, gap seconds)`` for each silence longer than the threshold, longest-first."""
    threshold_seconds: float
    max_gap_seconds: float | None


@dataclass
class JobLifecycleModel:
    """Represents one session's parsed job lifecycle, lane placement, and status census.

    Read it through :func:`job_lifecycle_for` rather than constructing it: the parse is linear in the
    session's records and every detector that wants job-shaped facts needs the same one.
    """

    jobs: dict[str, JobRecord] = field(default_factory=dict)
    lane_history: list[LanePlacement] = field(default_factory=list)
    card_count: int | None = None
    intake_budget: int | None = None
    card_intake: dict[int, CardIntake] = field(default_factory=dict)
    resolved_process_counts: dict[int, int] = field(default_factory=dict)
    status_snapshots: list[StatusSnapshot] = field(default_factory=list)
    line_skips: list[LineSkipEvent] = field(default_factory=list)
    model_movement: ModelMovement = field(default_factory=ModelMovement)
    ipc_drain_timestamps: list[datetime] = field(default_factory=list)
    dispatch_events: int = 0
    """How many times the scheduler seated work on an inference lane.

    Counts dispatch lines, not jobs: a batched inference seats several jobs on one lane in one act, and a
    ratio against "how much loading did this session do" wants the act, not the passenger count."""

    @property
    def dispatched_jobs(self) -> list[JobRecord]:
        """Jobs that reached an inference lane, in dispatch order."""
        dispatched = [job for job in self.jobs.values() if job.dispatched_at is not None]
        dispatched.sort(key=lambda job: job.dispatched_at or datetime.min)
        return dispatched

    @property
    def dispatch_count(self) -> int:
        """How many times the scheduler seated work on an inference lane (one per dispatch line)."""
        return self.dispatch_events

    @property
    def has_lifecycle_lines(self) -> bool:
        """Whether the capture carried enough lifecycle lines to split a job's wait into segments.

        False for an older or coarser log, where a detector must say it could not locate the wait rather
        than attributing it to a stage it did not measure.
        """
        return any(
            job.dispatched_at is not None and job.inference_finished_at is not None for job in self.jobs.values()
        )

    def latest_placement(self, process_id: int, *, before: datetime | None = None) -> LanePlacement | None:
        """The most recent placement of ``process_id`` at (or before) ``before``, or its first if none is."""
        placements = [placement for placement in self.lane_history if placement.process_id == process_id]
        if not placements:
            return None
        if before is None:
            return placements[-1]
        prior = [
            placement
            for placement in placements
            if placement.started_at is not None and placement.started_at <= before
        ]
        return prior[-1] if prior else placements[0]

    def current_placements(self) -> dict[int, LanePlacement]:
        """The latest known placement of every lane the capture saw spawn, keyed by process id."""
        latest: dict[int, LanePlacement] = {}
        for placement in self.lane_history:
            latest[placement.process_id] = placement
        return latest

    def lane_migrations(self) -> list[LaneMigration]:
        """Every re-spawn that landed a lane on a different card than the one it previously held."""
        migrations: list[LaneMigration] = []
        seen: dict[int, LanePlacement] = {}
        for placement in self.lane_history:
            previous = seen.get(placement.process_id)
            if previous is not None and previous.device_index != placement.device_index:
                migrations.append(
                    LaneMigration(
                        process_id=placement.process_id,
                        role=placement.role,
                        from_device_index=previous.device_index,
                        to_device_index=placement.device_index,
                        timestamp=placement.started_at,
                    ),
                )
            seen[placement.process_id] = placement
        return migrations

    def card_occupancy(self) -> dict[int, list[LanePlacement]]:
        """The lanes currently sitting on each card, keyed by device index (cardless lanes excluded)."""
        occupancy: dict[int, list[LanePlacement]] = {}
        for placement in self.current_placements().values():
            if placement.device_index is None:
                continue
            occupancy.setdefault(placement.device_index, []).append(placement)
        for lanes in occupancy.values():
            lanes.sort(key=lambda placement: placement.process_id)
        return occupancy

    def wait_segments(self) -> tuple[WaitSegment, WaitSegment, WaitSegment]:
        """The pop->dispatch, dispatch->finished and finished->submit distributions, in that order."""
        pre = [job.pre_inference_seconds for job in self.jobs.values()]
        inference = [job.inference_seconds for job in self.jobs.values()]
        post = [job.post_inference_seconds for job in self.jobs.values()]
        return (
            _segment("pre_inference", pre),
            _segment("inference", inference),
            _segment("post_inference", post),
        )

    def sampling_concurrency(self) -> ConcurrencyProfile:
        """Time-weighted distinct cards holding an in-flight job across the session."""
        intervals = [
            (job.dispatched_at, job.inference_finished_at, job.device_index)
            for job in self.jobs.values()
            if job.dispatched_at is not None
            and job.inference_finished_at is not None
            and job.device_index is not None
            and job.inference_finished_at > job.dispatched_at
        ]
        return _sweep_concurrency(intervals)

    def line_skip_census(self) -> dict[str, int]:
        """How many jobs were seated ahead of the head, by the scheduler's reason token."""
        census: dict[str, int] = {}
        for skip in self.line_skips:
            census[skip.reason] = census.get(skip.reason, 0) + 1
        return census

    def status_census(self) -> StatusCensus:
        """The per-print seating census: queue depth, idle lanes, and how many pending jobs were seatable."""
        return _status_census(self.status_snapshots, self.current_placements())

    def ipc_drain_profile(self) -> IpcDrainProfile:
        """The parent loop's IPC drain cadence, with silences measured against the status-print cadence."""
        return _ipc_drain_profile(self.ipc_drain_timestamps, self.status_snapshots)

    def status_cadence_seconds(self) -> float | None:
        """The median interval between status prints, the worker's own slowest routine clock."""
        stamps = [snapshot.timestamp for snapshot in self.status_snapshots if snapshot.timestamp is not None]
        gaps = [(later - earlier).total_seconds() for earlier, later in zip(stamps, stamps[1:], strict=False)]
        return _median(gaps)


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return the nearest-rank percentile of ``values``, or None when empty.

    Nearest-rank rather than interpolated because these samples are frequently tiny (a session with two
    submitted jobs is common in a crash bundle) and an interpolated value there invents a number no job
    exhibited.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0, min(len(ordered) - 1, round(quantile * (len(ordered) - 1))))
    return ordered[rank]


def _median(values: Sequence[float]) -> float | None:
    """Return the median of ``values``, or None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _segment(name: str, values: Sequence[float | None]) -> WaitSegment:
    """Build one :class:`WaitSegment` from a column of per-job durations, ignoring the unmeasurable ones."""
    present = [value for value in values if value is not None]
    return WaitSegment(
        name=name,
        count=len(present),
        median=_median(present),
        p90=percentile(present, 0.9),
        total=sum(present),
    )


def _sweep_concurrency(
    intervals: Sequence[tuple[datetime, datetime, int]],
) -> ConcurrencyProfile:
    """Sweep in-flight intervals into a time-weighted histogram of distinct busy cards."""
    if not intervals:
        return ConcurrencyProfile(seconds_by_card_count={}, observed_seconds=0.0)

    events: list[tuple[datetime, int, int]] = []
    for start, end, device_index in intervals:
        events.append((start, 1, device_index))
        events.append((end, -1, device_index))
    events.sort(key=lambda event: (event[0], event[1]))

    per_card: dict[int, int] = {}
    busy_cards = 0
    histogram: dict[int, float] = {}
    previous = events[0][0]
    for timestamp, delta, device_index in events:
        elapsed = (timestamp - previous).total_seconds()
        if elapsed > 0:
            histogram[busy_cards] = histogram.get(busy_cards, 0.0) + elapsed
        previous = timestamp
        was_busy = per_card.get(device_index, 0) > 0
        per_card[device_index] = per_card.get(device_index, 0) + delta
        now_busy = per_card[device_index] > 0
        if now_busy and not was_busy:
            busy_cards += 1
        elif was_busy and not now_busy:
            busy_cards -= 1
    return ConcurrencyProfile(
        seconds_by_card_count=histogram,
        observed_seconds=sum(histogram.values()),
    )


def _classify_head(snapshot: StatusSnapshot, busy_cards: set[int], lane_card: dict[int, int | None]) -> str:
    """Classify the pending queue's head as ``seatable``, ``resident_but_blocked`` or ``cold``."""
    head = snapshot.pending_jobs[0]
    holders = [lane for lane in snapshot.inference_lanes if lane.model == head.model]
    if not holders:
        return "cold"
    for lane in holders:
        if lane.idle and lane_card.get(lane.process_id) not in busy_cards:
            return "seatable"
    return "resident_but_blocked"


def _snapshot_lane_cards(snapshot: StatusSnapshot, placements: dict[int, LanePlacement]) -> dict[int, int | None]:
    """Map each inference lane in a status print to the card it sits on (None when unknown)."""
    return {
        lane.process_id: placements[lane.process_id].device_index if lane.process_id in placements else None
        for lane in snapshot.inference_lanes
    }


def _status_census(
    snapshots: Sequence[StatusSnapshot],
    placements: dict[int, LanePlacement] | None = None,
) -> StatusCensus:
    """Count queue depth, idle lanes and seatable pending jobs across a series of status prints."""
    if not snapshots:
        return StatusCensus()
    lane_placements = placements or {}
    pending_depths: list[float] = []
    idle_counts: list[float] = []
    seatable_counts: list[float] = []
    head_seatable = head_blocked = head_cold = 0

    for snapshot in snapshots:
        lanes = snapshot.inference_lanes
        if not lanes:
            continue
        lane_card = _snapshot_lane_cards(snapshot, lane_placements)
        busy_cards = {card for lane in lanes if lane.sampling and (card := lane_card.get(lane.process_id)) is not None}
        idle_lanes = [lane for lane in lanes if lane.idle]
        idle_free_card_models = {
            lane.model
            for lane in idle_lanes
            if lane.model is not None and lane_card.get(lane.process_id) not in busy_cards
        }
        seatable = sum(1 for pending in snapshot.pending_jobs if pending.model in idle_free_card_models)

        pending_depths.append(len(snapshot.pending_jobs))
        idle_counts.append(len(idle_lanes))
        seatable_counts.append(seatable)
        if snapshot.pending_jobs:
            head_state = _classify_head(snapshot, busy_cards, lane_card)
            if head_state == "seatable":
                head_seatable += 1
            elif head_state == "resident_but_blocked":
                head_blocked += 1
            else:
                head_cold += 1

    return StatusCensus(
        snapshots=len(pending_depths),
        median_pending=_median(pending_depths) or 0.0,
        median_idle_lanes=_median(idle_counts) or 0.0,
        median_seatable_pending=_median(seatable_counts) or 0.0,
        max_seatable_pending=int(max(seatable_counts)) if seatable_counts else 0,
        head_states=HeadStateCensus(seatable=head_seatable, resident_but_blocked=head_blocked, cold=head_cold),
    )


_IPC_STALL_CADENCE_MULTIPLIER = 1.5
"""How many status-print periods of IPC silence count as the parent loop having stopped draining.

The status print is the worker's slowest routine clock, so a silence longer than one and a half of its
periods cannot be explained by an idle worker: the print itself runs on the same loop that drains IPC."""
_IPC_STALL_FLOOR_SECONDS = 15.0
"""A floor under the derived threshold, so a log with an unusually fast status cadence does not report
ordinary scheduling jitter as a stall."""
_IPC_STALL_FALLBACK_SECONDS = 30.0
"""The threshold used when no status prints are present to derive a cadence from."""


def _ipc_drain_profile(
    timestamps: Sequence[datetime],
    snapshots: Sequence[StatusSnapshot],
) -> IpcDrainProfile:
    """Measure IPC drain cadence and flag silences longer than the status-print-derived threshold."""
    snapshot_stamps = [snapshot.timestamp for snapshot in snapshots if snapshot.timestamp is not None]
    cadence = _median(
        [
            (later - earlier).total_seconds()
            for earlier, later in zip(snapshot_stamps, snapshot_stamps[1:], strict=False)
        ],
    )
    if cadence is None or cadence <= 0:
        threshold = _IPC_STALL_FALLBACK_SECONDS
    else:
        threshold = max(_IPC_STALL_FLOOR_SECONDS, cadence * _IPC_STALL_CADENCE_MULTIPLIER)

    gaps = [
        (later, (later - earlier).total_seconds()) for earlier, later in zip(timestamps, timestamps[1:], strict=False)
    ]
    over = sorted((gap for gap in gaps if gap[1] > threshold), key=lambda gap: gap[1], reverse=True)
    return IpcDrainProfile(
        drains=len(timestamps),
        median_gap_seconds=_median([seconds for _, seconds in gaps]),
        gaps_over_threshold=tuple(over),
        threshold_seconds=threshold,
        max_gap_seconds=max((seconds for _, seconds in gaps), default=None),
    )


def _parse_bool(token: str) -> bool | None:
    """Parse the worker's ``True``/``False`` rendering of a flag, or None for anything else."""
    if token == "True":
        return True
    if token == "False":
        return False
    return None


def _parse_queue_entries(body: str) -> tuple[PendingJob, ...]:
    """Parse a ``<id8: model>, <id8: model>`` queue listing into its entries."""
    return tuple(
        PendingJob(job_id=match.group("job"), model=match.group("model")) for match in _QUEUE_ENTRY_RE.finditer(body)
    )


class _LifecycleParser:
    """Accumulates one session's lifecycle facts as its records stream past, in log order."""

    def __init__(self) -> None:
        self._model = JobLifecycleModel()
        self._preloads = 0
        self._unloads = 0
        self._cleared = 0
        self._expiries = 0
        self._loads = 0
        self._moved_to_ram = 0
        self._dispatch_events = 0
        self._last_dispatched: JobRecord | None = None
        self._open_lanes: list[LaneStatus] = []
        self._open_timestamp: datetime | None = None
        self._snapshot_open = False

    def _job(self, job_id: str) -> JobRecord:
        """The record for ``job_id``'s short form, created on first sight of any of its lines."""
        key = short_job_id(job_id)
        record = self._model.jobs.get(key)
        if record is None:
            record = JobRecord(job_id=key)
            self._model.jobs[key] = record
        return record

    def feed(self, record: LogRecord) -> None:
        """Fold one orchestrator record into the model."""
        message = record.message
        if self._feed_status(record, message):
            return
        if self._feed_job(record, message):
            return
        if self._feed_lane(record, message):
            return
        if self._feed_model_movement(message):
            return
        self._feed_shape(message)

    # region status blocks

    def _feed_status(self, record: LogRecord, message: str) -> bool:
        """Fold a status-print line into the open snapshot, opening or closing one as the print dictates."""
        if _STATUS_HEADER_RE.match(message):
            self._close_snapshot()
            self._snapshot_open = True
            self._open_timestamp = record.timestamp
            return True
        if not self._snapshot_open:
            return False

        aux = _STATUS_AUX_LANE_RE.match(message)
        if aux is not None:
            self._open_lanes.append(
                LaneStatus(
                    process_id=int(aux.group("process")),
                    role=LaneRole(aux.group("role").lower()),
                    state=aux.group("state"),
                    model=None,
                    sampling=False,
                ),
            )
            return True

        inference = _STATUS_INFERENCE_LANE_RE.match(message)
        if inference is not None:
            state = inference.group("state")
            self._open_lanes.append(
                LaneStatus(
                    process_id=int(inference.group("process")),
                    role=LaneRole.INFERENCE,
                    state=state,
                    model=inference.group("model"),
                    sampling=_SAMPLING_STATE_RE.match(state) is not None,
                ),
            )
            return True

        jobs_line = _JOBS_LINE_RE.match(message)
        if jobs_line is not None:
            self._model.status_snapshots.append(
                StatusSnapshot(
                    timestamp=self._open_timestamp,
                    lanes=tuple(self._open_lanes),
                    pending_jobs=_parse_queue_entries(jobs_line.group("body")),
                ),
            )
            self._open_lanes = []
            self._snapshot_open = False
            return True
        return False

    def _close_snapshot(self) -> None:
        """Emit a snapshot whose print ended without a Jobs line (a truncated or interleaved capture)."""
        if self._open_lanes:
            self._model.status_snapshots.append(
                StatusSnapshot(timestamp=self._open_timestamp, lanes=tuple(self._open_lanes), pending_jobs=()),
            )
        self._open_lanes = []
        self._snapshot_open = False

    # endregion

    # region job lifecycle

    def _feed_job(self, record: LogRecord, message: str) -> bool:
        """Fold a per-job lifecycle line into that job's record."""
        pop = _POP_RE.search(message)
        if pop is not None:
            job = self._job(pop.group("job"))
            job.full_job_id = pop.group("job")
            job.model = pop.group("model")
            job.popped_at = record.timestamp
            job.effective_megapixelsteps = int(pop.group("emps"))
            job.batch = int(pop.group("batch"))
            job.uses_loras = _parse_bool(pop.group("loras"))
            job.uses_post_processing = _parse_bool(pop.group("post"))
            return True

        dispatch = _DISPATCH_RE.search(message)
        if dispatch is not None:
            job = self._job(dispatch.group("job"))
            if job.dispatched_at is None:
                job.dispatched_at = record.timestamp
            job.process_id = int(dispatch.group("process"))
            self._dispatch_events += 1
            self._last_dispatched = job
            return True

        batch = _BATCH_IDS_RE.match(message)
        if batch is not None:
            lead = self._last_dispatched
            if lead is not None:
                for entry in _BATCH_ID_ENTRY_RE.finditer(batch.group("ids")):
                    member = self._job(entry.group(0))
                    if member is not lead:
                        member.full_job_id = member.full_job_id or entry.group(0)
                        member.batch_lead_job_id = lead.job_id
            return True

        finished = _INFERENCE_FINISHED_RE.search(message)
        if finished is not None:
            job = self._job(finished.group("job"))
            if job.inference_finished_at is None:
                job.inference_finished_at = record.timestamp
                job.generation_seconds = float(finished.group("seconds"))
            job.model = job.model or finished.group("model")
            job.process_id = job.process_id if job.process_id is not None else int(finished.group("process"))
            return True

        safety = _SAFETY_RE.search(message)
        if safety is not None:
            job = self._job(safety.group("job"))
            job.full_job_id = job.full_job_id or safety.group("job")
            job.safety_checked_at = record.timestamp
            job.safety_seconds = float(safety.group("seconds"))
            return True

        submit = _SUBMIT_RE.search(message)
        if submit is not None:
            job = self._job(submit.group("job"))
            if job.submitted_at is None:
                job.submitted_at = record.timestamp
            job.model = job.model or submit.group("model")
            if job.popped_at is None and record.timestamp is not None:
                # A capture that begins mid-run has no pop line for this job, but the submit line states
                # how long ago the pop was, which reconstructs it exactly.
                job.popped_at = record.timestamp - timedelta(seconds=float(submit.group("popped_ago")))
            if job.generation_seconds is None:
                job.generation_seconds = float(submit.group("generate"))
            return True

        faulted = _JOB_FAULTED_RE.search(message)
        if faulted is not None:
            job = self._job(faulted.group("job"))
            job.full_job_id = job.full_job_id or faulted.group("job")
            job.faulted_at = job.faulted_at or record.timestamp
            return True

        reported = _FAULT_REPORTED_RE.search(message)
        if reported is not None:
            job = self._job(reported.group("job"))
            job.full_job_id = job.full_job_id or reported.group("job")
            job.fault_reported = True
            job.faulted_at = job.faulted_at or record.timestamp
            return True

        skip = _LINE_SKIP_RE.search(message)
        if skip is not None:
            job = self._job(skip.group("job"))
            job.line_skipped = True
            self._model.line_skips.append(
                LineSkipEvent(
                    timestamp=record.timestamp,
                    job_id=short_job_id(skip.group("job")),
                    reason=skip.group("reason"),
                    process_id=int(skip.group("process")),
                    displaced_job_id=short_job_id(skip.group("displaced")),
                ),
            )
            return True

        queue = _JOB_QUEUE_LINE_RE.match(message)
        if queue is not None:
            # The popper's own queue listing carries no lane states, so it is not a status snapshot; it is
            # read only to learn the model of a job whose pop line predates the capture.
            for pending in _parse_queue_entries(queue.group("body")):
                job = self._job(pending.job_id)
                job.model = job.model or pending.model
            return True
        return False

    # endregion

    # region lanes and worker shape

    def _feed_lane(self, record: LogRecord, message: str) -> bool:
        """Fold a child-process spawn line into the lane placement history."""
        for pattern, role in (
            (_INFERENCE_LANE_RE, LaneRole.INFERENCE),
            (_POST_PROCESS_LANE_RE, LaneRole.POST_PROCESS),
            (_UTILITIES_LANE_RE, LaneRole.UTILITIES),
        ):
            match = pattern.search(message)
            if match is not None:
                self._model.lane_history.append(
                    LanePlacement(
                        process_id=int(match.group("process")),
                        role=role,
                        device_index=int(match.group("device")),
                        started_at=record.timestamp,
                    ),
                )
                return True
        safety = _SAFETY_LANE_RE.search(message)
        if safety is not None:
            self._model.lane_history.append(
                LanePlacement(
                    process_id=int(safety.group("process")),
                    role=LaneRole.SAFETY,
                    device_index=None,
                    started_at=record.timestamp,
                ),
            )
            return True
        drain = _IPC_DRAIN_RE.search(message)
        if drain is not None and record.timestamp is not None:
            self._model.ipc_drain_timestamps.append(record.timestamp)
            return True
        return False

    def _feed_model_movement(self, message: str) -> bool:
        """Count one model-movement line (preload, unload, clear, displacement, load, RAM offload)."""
        if _PRELOAD_RE.search(message):
            self._preloads += 1
            return True
        if _CLEARED_PRELOAD_RE.search(message):
            self._cleared += 1
            return True
        if _UNLOAD_RE.search(message):
            self._unloads += 1
            return True
        if _DISPLACED_EXPIRY_RE.search(message):
            self._expiries += 1
            return True
        if _LOADING_RE.search(message):
            self._loads += 1
            return True
        if _MOVED_TO_RAM_RE.search(message):
            self._moved_to_ram += 1
            return True
        return False

    def _feed_shape(self, message: str) -> None:
        """Fold the boot lines that state how many cards the worker drives and its intake budget."""
        driving = _DRIVING_CARDS_RE.search(message)
        if driving is not None:
            self._model.card_count = int(driving.group("cards"))
            self._model.intake_budget = int(driving.group("intake"))
            self._model.card_intake = {
                int(entry.group("device")): CardIntake(
                    device_index=int(entry.group("device")),
                    process_count=int(entry.group("processes")),
                    intake=int(entry.group("intake")),
                )
                for entry in _PER_CARD_INTAKE_RE.finditer(driving.group("per_card"))
            }
            return
        concurrency = _CARD_CONCURRENCY_RE.search(message)
        if concurrency is not None:
            self._model.resolved_process_counts = {
                int(entry.group("device")): int(entry.group("processes"))
                for entry in _CARD_CONCURRENCY_ENTRY_RE.finditer(concurrency.group("body"))
            }

    # endregion

    def finish(self) -> JobLifecycleModel:
        """Close any open status print, resolve each job's card, and return the model."""
        self._close_snapshot()
        self._model.model_movement = ModelMovement(
            preloads=self._preloads,
            unloads=self._unloads,
            cleared_preloads=self._cleared,
            displaced_expiries=self._expiries,
            loads=self._loads,
            moved_to_system_ram=self._moved_to_ram,
        )
        self._model.dispatch_events = self._dispatch_events
        for job in self._model.jobs.values():
            lead_id = job.batch_lead_job_id
            if lead_id is None:
                continue
            lead = self._model.jobs.get(lead_id)
            if lead is None:
                continue
            job.dispatched_at = job.dispatched_at or lead.dispatched_at
            job.process_id = job.process_id if job.process_id is not None else lead.process_id
            job.inference_finished_at = job.inference_finished_at or lead.inference_finished_at
        for job in self._model.jobs.values():
            if job.process_id is None:
                continue
            placement = self._model.latest_placement(job.process_id, before=job.dispatched_at)
            if placement is not None:
                job.device_index = placement.device_index
        return self._model


def build_job_lifecycle(records: Sequence[LogRecord]) -> JobLifecycleModel:
    """Parse a session's orchestrator records into its job lifecycle model.

    Args:
        records: The session's orchestrator records, in log order.

    Returns:
        The parsed model. Fields the capture did not carry are empty or None rather than raising.
    """
    parser = _LifecycleParser()
    for record in records:
        parser.feed(record)
    return parser.finish()


def job_lifecycle_for(context: SessionContext) -> JobLifecycleModel:
    """Return the session's job lifecycle model, parsing it once and caching it on the context.

    Every detector that needs job-shaped facts must go through here rather than regexing job lines of
    its own, so the parse is single-sourced and the per-session cost is paid once.
    """
    cached = context.job_lifecycle
    if cached is None:
        cached = build_job_lifecycle(context.session.records)
        context.job_lifecycle = cached
    return cached
