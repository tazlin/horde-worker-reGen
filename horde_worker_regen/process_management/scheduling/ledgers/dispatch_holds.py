"""The holds the dispatch and clearance gates place on jobs that fit the queue but not yet the card.

A dispatch-time residency hold parks an already-resident head while idle sibling VRAM is evicted so the head's
materialisation fits before it commits; the job keeps its queue position and is never faulted. A
post-processing defer hold parks a head whose sampling peak would collide with an in-flight post-processing
chain. A clearance hold withholds the clearance grant from a staged, primed child whose diffusion weights would
over-commit the card at the clearance moment. Each is bookkeeping the gate already needs; keeping it in one place
lets the stall classifier, the recovery supervisor and the liveness watchdog read the gate's own truthful state
instead of re-deriving it.

This module owns the per-job hold records and the session counters. The scheduler decides the holds, records
the resource-state transitions and decision events, and issues the evictions.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from horde_worker_regen.process_management.resources.run_metrics import FlatScalarMap

DISPATCH_HOLD_LIVENESS_SECONDS = 190.0
"""How long a head's residency-reconciliation hold keeps the recovery supervisor from reading an idle card with
pending work as a structural wedge.

The hold has its own escalation, and this is that escalation end to end: the head-starvation clock the
measured-load probe is gated on, the idle-context teardown grace, and the heavy head's load window once admitted.
A hold older than this has exhausted every remedy the gate owns, so the supervisor takes over as it would for
any other wedge."""


class HoldRelease(enum.Enum):
    """What released a dispatch hold."""

    MEASURED_ATTEMPT = "measured_attempt"
    """The measured-load probe admitted the held job."""
    RECLAIM = "reclaim"
    """The gate's own eviction commands freed the room."""
    NATURAL_FREE = "natural_free"
    """The card recovered on its own."""


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
        self.holds = 0
        """Distinct held dispatches this session."""
        self.conflicts = 0
        """Hold passes this session (every pass a job was held counts one)."""
        self.hold_seconds = 0.0
        """Cumulative seconds released holds stood."""
        self.released_by_measured_attempt = 0
        self.released_by_reclaim = 0
        self.released_by_natural_free = 0
        self.pp_defer_holds: set[str] = set()
        """Heads the post-processing co-residency gate is deferring right now."""
        self.clearance_hold_ids: set[str] = set()
        """Staged, primed job ids the clearance gate is currently withholding the grant from."""
        self.clearance_hold_spans: dict[str, tuple[float, float | None]] = {}
        """Per job: accumulated held seconds from closed clearance-hold spans, and the live span's start or
        None. Closed spans are kept until the job leaves the in-progress set so a resolved hold still widens
        the liveness grace it already consumed."""

    # ---- dispatch residency holds

    def note_hold(self, job_id: str, *, reclaim_applied: bool, room_inputs: FlatScalarMap | None) -> bool:
        """Count a held pass for ``job_id``; return whether this is the job's first hold."""
        self.conflicts += 1
        if room_inputs is not None:
            self.room_inputs[job_id] = room_inputs
        first = job_id not in self.hold_since
        if first:
            self.hold_since[job_id] = self._clock()
            self.holds += 1
        if reclaim_applied:
            self.reclaim_requested.add(job_id)
        return first

    def resolve_hold(self, job_id: str, *, measured_attempt: bool) -> ReleasedHold | None:
        """Close ``job_id``'s hold, folding its duration into the counters; None when it was never held.

        The release is attributed to the measured-load probe when that admitted the job, to reclaim when this
        gate emitted eviction commands during the hold, otherwise to the card freeing on its own.
        """
        held_since = self.hold_since.pop(job_id, None)
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

    # ---- post-processing defer holds

    def note_pp_defer(self, job_id: str, *, deferred: bool) -> None:
        """Record whether the post-processing co-residency gate held ``job_id`` on this pass."""
        if deferred:
            self.pp_defer_holds.add(job_id)
        else:
            self.pp_defer_holds.discard(job_id)

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
        for job_id in [job_id for job_id in self.clearance_hold_spans if job_id not in live]:
            del self.clearance_hold_spans[job_id]
