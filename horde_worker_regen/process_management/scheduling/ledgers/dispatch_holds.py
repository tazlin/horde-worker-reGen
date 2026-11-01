"""The holds the dispatch and clearance gates place on jobs that fit the queue but not yet the card.

A dispatch-time residency hold parks an already-resident head while idle sibling VRAM is evicted so the head's
materialisation fits before it commits; the job keeps its queue position and is never faulted. A
post-processing defer hold parks a head whose sampling peak would collide with an in-flight post-processing
chain. A clearance hold withholds the clearance grant from a staged, primed child whose diffusion weights would
over-commit the card at the clearance moment. A utilities-lane hold keeps an extended-ControlNet job out of
selection while the image-utilities lane comes up to serve its control type. Each is bookkeeping the gate already
needs; keeping it in one place lets the stall classifier, the recovery supervisor and the liveness watchdog read
the gate's own truthful state instead of re-deriving it.

Alongside them it owns the current cycle's dispatch declines: which gate withheld each job the dispatch pass
selected, so the pass can route around a withheld job instead of re-asking the gate, and the stall explainer
can name the gate rather than reporting an unexplained stall.

This module owns the per-job hold records and the session counters. The scheduler decides the holds, records
the resource-state transitions and decision events, and issues the evictions.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from horde_worker_regen.process_management.resources.run_metrics import FlatScalarMap
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyBucket

DISPATCH_HOLD_LIVENESS_SECONDS = 190.0
"""How long a head's residency-reconciliation hold with a live remedy keeps the recovery supervisor from reading
an idle card with pending work as a structural wedge.

The bound is the gate's escalation end to end: the head-starvation clock the measured-load probe is gated on, the
idle-context teardown grace, and the heavy head's load window once admitted. A hold older than this has exhausted
every remedy the gate owns, so the supervisor takes over as it would for any other wedge. A hold whose latest
pass left no remedy live (:attr:`HoldRemedy.NONE`) does not get the bound at all. Its remedies are already spent,
and the bound is longer than the 150 s deadline the horde sets on some jobs, so waiting it out only loses the jobs
queued behind the hold."""


class HoldRemedy(enum.Enum):
    """The remedy that the latest pass of a dispatch hold left live, derived from that pass's verdict.

    Several remedies can be live at once. The ledger records the first live one in member order.
    """

    REROUTE = "reroute"
    """The arbiter denied the job on this card and another card can seat it, but the copy that pins the job here
    stays while its lane owns a job, is loading, or has an unload from RAM on the way."""
    RECLAIM = "reclaim"
    """An actuator accepted an eviction command for the hold on this pass."""
    TEARDOWN_GRACE = "teardown_grace"
    """The deficit is the head's own idle sibling contexts, which the arbiter tears down once the head has starved
    past the teardown grace."""
    PROBE = "probe"
    """The measured-load probe on this card is not spent: it is waiting on the head-starvation clock, eligible, or
    in progress."""
    NONE = "none"
    """No remedy is live. Either the verdict is a DENY and no other card can seat the job, or it is a DEFER whose
    probe on this card is spent, with nothing reclaimed and no teardown grace open."""

    @property
    def live(self) -> bool:
        """Whether this remedy can still release the hold."""
        return self is not HoldRemedy.NONE

    @property
    def phrase(self) -> str:
        """What the gate is doing about the hold, for a log line."""
        return _HOLD_REMEDY_PHRASES[self]


_HOLD_REMEDY_PHRASES: dict[HoldRemedy, str] = {
    HoldRemedy.REROUTE: "waiting to retire the copy that pins the head to a card that cannot seat it",
    HoldRemedy.RECLAIM: "reclaiming room for the head",
    HoldRemedy.TEARDOWN_GRACE: "waiting out the grace before it tears down idle contexts",
    HoldRemedy.PROBE: "waiting on its measured-load probe",
    HoldRemedy.NONE: "holding the head with no remedy left on this card",
}


def residency_hold_detail(remedy: HoldRemedy | None) -> str:
    """Why a residency-held job does not dispatch, for the dispatch decline and the stall summary.

    ``remedy`` is the one the ledger recorded for the hold's latest pass. It is None when that pass did not leave a
    hold standing, as when the gate unloaded the job's copy for a reroute.
    """
    if remedy is None:
        return "its materialisation would over-commit the card"
    return f"its materialisation would over-commit the card, and the residency gate is {remedy.phrase}"


def hold_remedy(
    *,
    denied: bool,
    reroute_pending: bool,
    reclaim_applied: bool,
    teardown_grace_open: bool,
    probe_pending: bool,
) -> HoldRemedy:
    """Return the remedy a hold's verdict left live on the pass that noted it.

    A DENY is terminal on its card. The arbiter issues it only after the ceiling attempt is refused or spent, and
    the achievable ceiling does not count the worker's own memory, so no reclaim on the card moves it. The only
    way out of a DENY is another card. A DEFER keeps the gate's remedies until each is spent.

    Args:
        denied: Whether the verdict is a DENY.
        reroute_pending: Whether another card can seat the denied job and the copy pinning it was kept this pass.
        reclaim_applied: Whether an actuator accepted an eviction command for the hold this pass.
        teardown_grace_open: Whether the verdict waits out the teardown grace for the head's idle sibling contexts.
        probe_pending: Whether the measured-load probe on the hold's card is not yet spent.
    """
    if denied:
        return HoldRemedy.REROUTE if reroute_pending else HoldRemedy.NONE
    if reclaim_applied:
        return HoldRemedy.RECLAIM
    if teardown_grace_open:
        return HoldRemedy.TEARDOWN_GRACE
    if probe_pending:
        return HoldRemedy.PROBE
    return HoldRemedy.NONE


class HoldLivenessEnd(enum.Enum):
    """Why a hold stopped keeping the recovery supervisor from reading the queue as wedged."""

    BOUND_LAPSED = "bound_lapsed"
    """The hold has a live remedy but has stood past :data:`DISPATCH_HOLD_LIVENESS_SECONDS`."""
    NO_REMEDY = "no_remedy"
    """The hold's latest pass left no remedy live, so the bound does not apply."""


@dataclass(frozen=True)
class HoldLiveness:
    """A dispatch hold's age, its liveness bound and the remedy its latest pass left live."""

    held_seconds: float
    bound_seconds: float
    remedy: HoldRemedy

    @property
    def active(self) -> bool:
        """Whether the hold keeps the recovery supervisor from reading the queue as wedged."""
        return self.ended_by is None

    @property
    def ended_by(self) -> HoldLivenessEnd | None:
        """Why the hold no longer keeps recovery waiting, or None while it does."""
        if not self.remedy.live:
            return HoldLivenessEnd.NO_REMEDY
        if self.held_seconds >= self.bound_seconds:
            return HoldLivenessEnd.BOUND_LAPSED
        return None


class HoldRelease(enum.Enum):
    """What released a dispatch hold."""

    MEASURED_ATTEMPT = "measured_attempt"
    """The measured-load probe admitted the held job."""
    RECLAIM = "reclaim"
    """The gate's own eviction commands freed the room."""
    NATURAL_FREE = "natural_free"
    """The card recovered on its own."""
    REROUTE = "reroute"
    """The arbiter denied the job on the card holding its copy, another card can seat it, and the gate retired
    that copy so the preload pass can place the job on a card that seats it."""


@dataclass(frozen=True)
class ReleasedHold:
    """A dispatch hold that just closed, with what the release record needs."""

    job_id: str
    release: HoldRelease
    hold_seconds: float
    room_inputs: FlatScalarMap
    """The room breakdown of the hold's most recent verdict, for the release record."""


@dataclass(frozen=True)
class AbandonedHold:
    """A dispatch hold whose job left the pending queue by some path other than a release through the gate."""

    job_id: str
    hold_seconds: float
    room_inputs: FlatScalarMap


@dataclass(frozen=True)
class DispatchDecline:
    """Why one gate withheld a job the dispatch pass had already selected, within one scheduling cycle.

    A dispatch that is withheld leaves the selected job undispatched without concluding anything: the job
    keeps its queue position and every clock the gate owns. Naming the gate that did it turns "the scheduler
    returned nothing" into a fact a reader can act on, and lets the rest of the cycle route around the
    withheld job instead of re-asking the same gate.
    """

    job_id: str
    bucket: SlotDutyBucket
    """The duty-attribution member naming the gate, so the stall text, the slot accounting and this record all
    name one cause."""
    device_index: int | None
    """The card the withheld dispatch would have landed on; None where the slot names no device."""
    detail: str
    """The gate's own words for what it is waiting on. Carries no quantity that advances with the clock, so
    successive cycles can be compared for a changed cause."""


@dataclass
class PostProcessingTurn:
    """The card's next turn, owed to a job that a running post-processing chain held.

    A job that cannot share the card with a chain and the lane's chains take the card in turns. The job a running
    chain held goes next, and the lane leaves its pending chains waiting until that job has its sampling window.
    """

    job_id: str
    device_index: int | None
    """The card the held job would sample on; None where its slot names no device."""
    claimed_at: float
    """When a running chain first held the job."""
    yield_began_at: float | None = None
    """When the lane first left a pending chain waiting for the job, or None while a chain still runs."""

    def yield_expired(self, now: float, *, floor_seconds: float) -> bool:
        """Whether the lane has waited on the job at least as long as the job waited on the lane.

        A job that is given the turn and still cannot start is waiting on something other than the lane, so the
        lane stops waiting once the two waits are even. ``floor_seconds`` covers a job the chain held only
        briefly, which still needs time to be dispatched and cleared.
        """
        if self.yield_began_at is None:
            return False
        waited_on_lane = max(0.0, self.yield_began_at - self.claimed_at)
        return (now - self.yield_began_at) > max(floor_seconds, waited_on_lane)


class DispatchHoldLedger:
    """Per-job hold records and session counters for the dispatch, post-processing defer and clearance gates."""

    def __init__(self, clock: Callable[[], float]) -> None:
        """Track holds against ``clock``."""
        self._clock = clock
        self.hold_since: dict[str, float] = {}
        """When each held job's dispatch hold first stood, keyed by job id."""
        self.reclaim_requested: set[str] = set()
        """Held jobs for which the gate's actuator accepted an eviction command, so a release is attributed to
        reclaim rather than to the card recovering on its own."""
        self.room_inputs: dict[str, FlatScalarMap] = {}
        """The room breakdown of the most recent hold verdict per held job."""
        self.remedies: dict[str, HoldRemedy] = {}
        """The remedy the most recent hold pass left live per held job."""
        self.holds = 0
        """Distinct held dispatches this session."""
        self.conflicts = 0
        """Hold passes this session (every pass a job was held counts one)."""
        self.hold_seconds = 0.0
        """Cumulative seconds released holds stood."""
        self.released_by_measured_attempt = 0
        self.released_by_reclaim = 0
        self.released_by_natural_free = 0
        self.released_by_reroute = 0
        self.pp_defer_holds: set[str] = set()
        """Heads the post-processing co-residency gate is deferring right now."""
        self.pp_turn: PostProcessingTurn | None = None
        """The turn a running chain's held job is owed, or None when the lane owes none."""
        self.clearance_hold_ids: set[str] = set()
        """Staged, primed job ids the clearance gate is currently withholding the grant from."""
        self.clearance_hold_spans: dict[str, tuple[float, float | None]] = {}
        """Per job: accumulated held seconds from closed clearance-hold spans, and the live span's start or
        None. Closed spans are kept until the job leaves the in-progress set so a resolved hold still widens
        the liveness grace it already consumed."""
        self.clearance_granted_ids: set[str] = set()
        """In-progress job ids the clearance gate has granted their load-and-sample window."""
        self.utilities_lane_holds: dict[str, float] = {}
        """Extended-ControlNet jobs held for the image-utilities lane to serve their control type, keyed by job id,
        with when each hold began. Selection skips a held job, so it does not stall the jobs behind it."""
        self.utilities_lane_hold_count = 0
        """Distinct utilities-lane holds this session."""
        self.cycle_declines: dict[str, DispatchDecline] = {}
        """The jobs a dispatch gate withheld during the scheduling cycle now running, keyed by job id. Scoped
        to the cycle and discarded at its start, so it states what this pass decided and never a stale verdict
        the world has moved past."""

    # ---- per-cycle dispatch declines

    def note_decline(self, decline: DispatchDecline) -> None:
        """Record that a gate withheld ``decline.job_id`` this cycle, replacing any earlier record for it."""
        self.cycle_declines[decline.job_id] = decline

    def clear_cycle_declines(self) -> None:
        """Discard the cycle's decline records, which the next cycle re-derives from its own gate passes."""
        self.cycle_declines.clear()

    def declined_cards(self) -> set[int]:
        """The cards a withheld dispatch was aimed at this cycle.

        A card whose dispatch a gate has already withheld is the withheld job's to claim as soon as its gate
        releases, so seating a later job there takes exactly the capacity the wait is for.
        """
        return {decline.device_index for decline in self.cycle_declines.values() if decline.device_index is not None}

    # ---- dispatch residency holds

    def note_hold(
        self,
        job_id: str,
        *,
        reclaim_applied: bool,
        room_inputs: FlatScalarMap | None,
        remedy: HoldRemedy,
    ) -> bool:
        """Count a held pass for ``job_id`` with the remedy it left live. Returns whether this is the first hold."""
        self.conflicts += 1
        if room_inputs is not None:
            self.room_inputs[job_id] = room_inputs
        self.remedies[job_id] = remedy
        first = job_id not in self.hold_since
        if first:
            self.hold_since[job_id] = self._clock()
            self.holds += 1
        if reclaim_applied:
            self.reclaim_requested.add(job_id)
        return first

    def resolve_hold(self, job_id: str, *, measured_attempt: bool, rerouted: bool = False) -> ReleasedHold | None:
        """Close ``job_id``'s hold, folding its duration into the counters; None when it was never held.

        The release is attributed to the measured-load probe when that admitted the job, to a reroute when the
        gate retired the job's copy on a card that cannot seat it, to reclaim when this gate emitted eviction
        commands during the hold, otherwise to the card freeing on its own.
        """
        held_since = self.hold_since.pop(job_id, None)
        self.remedies.pop(job_id, None)
        if held_since is None:
            self.reclaim_requested.discard(job_id)
            self.room_inputs.pop(job_id, None)
            return None
        held_seconds = max(0.0, self._clock() - held_since)
        self.hold_seconds += held_seconds
        if measured_attempt:
            self.released_by_measured_attempt += 1
            self.reclaim_requested.discard(job_id)
            release = HoldRelease.MEASURED_ATTEMPT
        elif rerouted:
            self.released_by_reroute += 1
            self.reclaim_requested.discard(job_id)
            release = HoldRelease.REROUTE
        elif job_id in self.reclaim_requested:
            self.released_by_reclaim += 1
            self.reclaim_requested.discard(job_id)
            release = HoldRelease.RECLAIM
        else:
            self.released_by_natural_free += 1
            release = HoldRelease.NATURAL_FREE
        return ReleasedHold(job_id, release, held_seconds, self.room_inputs.pop(job_id, {}))

    def prune(self, pending_ids: Iterable[str]) -> list[AbandonedHold]:
        """Forget holds for jobs no longer pending; an abandoned hold advances no release counter."""
        live = set(pending_ids)
        abandoned: list[AbandonedHold] = []
        for job_id in [job_id for job_id in self.hold_since if job_id not in live]:
            held_since = self.hold_since.pop(job_id)
            self.reclaim_requested.discard(job_id)
            self.remedies.pop(job_id, None)
            abandoned.append(
                AbandonedHold(job_id, max(0.0, self._clock() - held_since), self.room_inputs.pop(job_id, {})),
            )
        self.pp_defer_holds.intersection_update(live)
        return abandoned

    def held_seconds(self, job_id: str) -> float | None:
        """How long ``job_id``'s dispatch hold has stood, or None when it is not held."""
        held_since = self.hold_since.get(job_id)
        if held_since is None:
            return None
        return max(0.0, self._clock() - held_since)

    def is_standing(self, job_id: str, bound_seconds: float) -> bool:
        """Whether ``job_id``'s hold has stood at least ``bound_seconds``."""
        held = self.held_seconds(job_id)
        return held is not None and held >= bound_seconds

    def liveness(self, job_id: str) -> HoldLiveness | None:
        """``job_id``'s hold age against :data:`DISPATCH_HOLD_LIVENESS_SECONDS` and its live remedy, or None."""
        held = self.held_seconds(job_id)
        if held is None:
            return None
        return HoldLiveness(
            held_seconds=held,
            bound_seconds=DISPATCH_HOLD_LIVENESS_SECONDS,
            remedy=self.remedies.get(job_id, HoldRemedy.NONE),
        )

    # ---- utilities-lane holds

    def note_utilities_lane_hold(self, job_id: str) -> bool:
        """Hold ``job_id`` for the image-utilities lane and return whether this is the job's first hold."""
        if job_id in self.utilities_lane_holds:
            return False
        self.utilities_lane_holds[job_id] = self._clock()
        self.utilities_lane_hold_count += 1
        return True

    def release_utilities_lane_holds(self, keep: Iterable[str]) -> list[tuple[str, float]]:
        """Release every utilities-lane hold not in ``keep`` and return each released job id with its held seconds."""
        kept = set(keep)
        released: list[tuple[str, float]] = []
        for job_id in [job_id for job_id in self.utilities_lane_holds if job_id not in kept]:
            held_since = self.utilities_lane_holds.pop(job_id)
            released.append((job_id, max(0.0, self._clock() - held_since)))
        return released

    # ---- post-processing defer holds

    def note_pp_defer(self, job_id: str, *, deferred: bool) -> None:
        """Record whether the post-processing co-residency gate held ``job_id`` on this pass."""
        if deferred:
            self.pp_defer_holds.add(job_id)
        else:
            self.pp_defer_holds.discard(job_id)

    def claim_pp_turn(self, job_id: str, device_index: int | None) -> None:
        """Owe the card's next turn to ``job_id`` unless another held job already has it."""
        if self.pp_turn is None:
            self.pp_turn = PostProcessingTurn(job_id=job_id, device_index=device_index, claimed_at=self._clock())

    def begin_pp_yield(self) -> None:
        """Start the owed turn's clock at the first pass where the lane leaves a chain waiting for it."""
        if self.pp_turn is not None and self.pp_turn.yield_began_at is None:
            self.pp_turn.yield_began_at = self._clock()

    def pp_yield_expired(self, *, floor_seconds: float) -> bool:
        """Whether the lane has waited out the owed turn (see :meth:`PostProcessingTurn.yield_expired`)."""
        return self.pp_turn is not None and self.pp_turn.yield_expired(self._clock(), floor_seconds=floor_seconds)

    def clear_pp_turn(self) -> None:
        """Forget the owed turn: its job has its sampling window, left the queue, or was waited out."""
        self.pp_turn = None

    # ---- clearance holds

    def note_clearance_hold(self, job_id: str) -> None:
        """Record that ``job_id``'s clearance was withheld this pass, opening a live span if none is open."""
        self.clearance_hold_ids.add(job_id)
        accumulated, live_since = self.clearance_hold_spans.get(job_id, (0.0, None))
        if live_since is None:
            self.clearance_hold_spans[job_id] = (accumulated, self._clock())

    def resolve_clearance_hold(self, job_id: str) -> None:
        """Close ``job_id``'s live clearance span, if any, keeping its accumulated time (idempotent)."""
        self.clearance_hold_ids.discard(job_id)
        span = self.clearance_hold_spans.get(job_id)
        if span is not None:
            accumulated, live_since = span
            if live_since is not None:
                self.clearance_hold_spans[job_id] = (accumulated + max(0.0, self._clock() - live_since), None)

    def note_clearance_granted(self, job_id: str) -> None:
        """Record that ``job_id`` was granted its load-and-sample window."""
        self.clearance_granted_ids.add(job_id)

    def clearance_held_seconds(self, job_id: str) -> float:
        """Total seconds the clearance gate has withheld ``job_id``: closed spans plus any live one."""
        span = self.clearance_hold_spans.get(job_id)
        if span is None:
            return 0.0
        accumulated, live_since = span
        if live_since is not None:
            return accumulated + max(0.0, self._clock() - live_since)
        return accumulated

    def forget_clearance(self, live_job_ids: Iterable[str]) -> None:
        """Drop clearance records for jobs no longer in progress, so the maps self-heal."""
        live = set(live_job_ids)
        self.clearance_hold_ids.intersection_update(live)
        self.clearance_granted_ids.intersection_update(live)
        for job_id in [job_id for job_id in self.clearance_hold_spans if job_id not in live]:
            del self.clearance_hold_spans[job_id]
