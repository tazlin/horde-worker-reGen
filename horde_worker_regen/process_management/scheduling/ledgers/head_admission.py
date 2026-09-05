"""What the scheduler remembers about the head of the queue's admission across scheduling cycles.

The head of the queue is the job every gate is judged against: how long it has sat on an idle device, whether
its preload is being deferred behind live work, whether a crash-looping safety pool is holding admissions off
its card, what the last admission decision said, and why dispatch is stalled. Each of those is a clock or a
record that only means something over several cycles, and each has a bound so an intent cannot outlive the
evidence that produced it. The scheduler reads its collaborators and owns every log line, ledger event and
actuation; this module owns the state and the step decisions over it.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from horde_worker_regen.process_management.scheduling.diagnostic_throttle import DiagnosticThrottle
from horde_worker_regen.process_management.scheduling.governance.preload_admission import AdmissionDecision

HEAD_PROTECTION_MAX_STARVE_SECONDS = 120.0
"""How long a parked head may reserve card room from the jobs behind it before the reservation is released.

Reserving room is only worth its cost while the head is converging on a dispatch: a head whose own admission
keeps declining would otherwise hold an idle card against fitting siblings for as long as the queue lasts. Well
above the ordinary drain of an in-flight job, so a normal handoff never trips it, and far below the horizon at
which a stalled queue misses the horde's dispatch deadlines."""

HEAD_RAM_DEFER_BARRIER_SECONDS = 60.0
"""How long the head's preload may be continuously RAM-deferred behind live work before the dispatch barrier
latches.

Below this a running sibling legitimately holds the memory the head needs and the RAM branch keeps reclaiming
and re-asking. Past it, with reclaim freeing nothing, the head can never reach the no-live-consumer best-effort
admit on its own, so the barrier withholds new dispatch to other slots and lets the running jobs drain to that
escape. Above a normal reclaim-and-retry settle and above the reclaim-cycle grace, so a deliberate reclaim cycle
is never mistaken for starvation."""

HEAD_RAM_DEFER_BARRIER_CAP_SECONDS = 180.0
"""How long the dispatch barrier may hold before the head is declined for reissue.

Three times the engage bound: long enough that draining siblings win the race in practice, bounded so a head
that will not fit even as the card empties fails fast rather than rotting."""

SAFETY_RECOVERY_HOLD_TTL_SECONDS = 120.0
"""How long the safety-recovery admission hold may keep new preloads off a saturated card.

The hold lets the card drain so a deferred safety GPU start can succeed; if the safety pool still cannot start
this long after the hold engaged, holding inference longer starves the card without helping. Above a normal
drain-and-respawn window and bounded so a permanently stuck safety pool cannot wedge inference intake."""

DISPATCH_STALL_MIN_SECONDS = 10.0
"""How long the head must be continuously undispatched before a dispatch stall is diagnosed.

Reuses the head-starvation clock so an ordinary one-tick gap between jobs, or a model mid-preload, is never
reported."""

DISPATCH_STALL_LOG_INTERVAL_SECONDS = 30.0
"""Minimum gap between repeats of the dispatch-stall diagnostic for an unchanged reason. A changed reason logs
immediately."""

MISSING_MODEL_LATCH_FALLBACK_SECONDS = 150.0
"""Bound for the missing-model recovery latch when no numeric ``preload_timeout`` is configured. Matches the
``preload_timeout`` default."""

STAGING_DEFER_REPEAT_SECONDS = 300.0
"""How long an unchanged staging-defer reason stays suppressed before it is restated with its tally.

The cap is consulted on every dispatch decision, so the reason edge carries the information and a persistent
hold is worth one restatement every few minutes."""


class StagingDeferReason(enum.Enum):
    """Why the in-progress cap was held at the sampling-slot count instead of allowing another staged job.

    Deferring staging is ordinary backpressure, but it is also the clause that leaves spare inference processes
    idle while jobs queue, so a session has to be able to read which measurement held it.
    """

    MEASUREMENT_UNREAD = "unread"
    """No GPU-bearing child has reported its VRAM yet, so there is no evidence to admit staging on."""
    ENCODE_HEADROOM_SHORT = "headroom"
    """Measured free VRAM net of the reserve does not cover a staged job's encode working set."""


def format_staging_defer_tally(defers: Mapping[StagingDeferReason, int]) -> str | None:
    """A compact staging-deferral tally for the duty-cycle line, or None when staging was never held back.

    Reports the total and the share each measurement took of it, largest first, so a duty figure short of
    target can be read straight across to what kept spare inference processes out of the queue.
    """
    counted = {reason: count for reason, count in defers.items() if count}
    total = sum(counted.values())
    if not total:
        return None
    ranked = sorted(counted.items(), key=lambda entry: (-entry[1], entry[0].value))
    shares = ", ".join(f"{reason.value} {count / total:.0%}" for reason, count in ranked)
    return f"staging deferred: {total} ({shares})"


@dataclass(frozen=True)
class LatestPreloadAdmission:
    """Operator-facing record of the most recent preload-admission decision."""

    decision: AdmissionDecision
    model: str | None
    """Model whose queued job was judged, when available."""
    process_id: int | None
    """Target inference process selected by the decision, when one was selected."""
    reason: str
    timestamp: float
    """Worker wall-clock time when the decision was recorded."""


class HeadRamDeferStep(enum.Enum):
    """What the head RAM-defer clock asks the scheduler to do after one deferred cycle."""

    NONE = enum.auto()
    RELEASE_BARRIER = enum.auto()
    """Reclaim made progress for the barred head, so the barrier is no longer needed."""
    ENGAGE_BARRIER = enum.auto()
    """The head has deferred past the starvation bound with nothing freed: withhold other dispatch."""
    DECLINE_HEAD = enum.auto()
    """The barrier has held past its cap without admitting the head: fault it for reissue."""


class SafetyRecoveryHoldStep(enum.Enum):
    """The safety-recovery hold's transition on one admission query."""

    NOT_HOLDING = enum.auto()
    """No hold applies to this admission: the pool is healthy, the card is not the one waiting, or the episode
    already gave up at its TTL."""
    RELEASED = enum.auto()
    """The pool started or recovered while a hold was engaged; the hold is gone."""
    ENGAGED = enum.auto()
    """This query engaged the hold: announce it and nudge idle reclaim."""
    HOLDING = enum.auto()
    """An engaged hold still applies."""
    EXPIRED = enum.auto()
    """The hold outlived its TTL on this query and latched the episode; admissions proceed."""


class HeadAdmissionLedger:
    """Clocks, latches and records about the head of the queue, each bounded so no intent outlives its evidence."""

    def __init__(self, clock: Callable[[], float]) -> None:
        """Track head-of-queue admission state against ``clock``."""
        self._clock = clock
        self.starvation_job_id: str | None = None
        """The head whose idle-device wait is being timed, or None while a live job holds the device."""
        self.starvation_since: float = 0.0
        """When that wait began; 0.0 while nothing is timed."""
        self.ram_defer_job_id: str | None = None
        """The head whose preload the RAM verdict is continuously deferring behind live work."""
        self.ram_defer_since: float = 0.0
        self.barrier_job_id: str | None = None
        """The head the dispatch barrier is held for, or None while dispatch flows."""
        self.barrier_since: float = 0.0
        self.barrier_withhold_logged: bool = False
        """Whether the barrier's first withheld dispatch has been logged this episode."""
        self.recovery_hold_since: float = 0.0
        """When the safety-recovery hold engaged; 0.0 while inactive."""
        self.recovery_hold_expired: bool = False
        """The hold gave up at its TTL and must not re-engage until the crash-loop condition clears."""
        self.last_preload_admission: LatestPreloadAdmission | None = None
        self.admission_denials_by_device: dict[int, int] = {}
        self.admission_headroom_mb_by_device: dict[int, float | None] = {}
        """The last measured-floor admission headroom per card (worker-wide under key 0); None when unapplied."""
        self.staging_defers: dict[StagingDeferReason, int] = {}
        """How many staging deferrals each measurement accounted for this session."""
        self._staging_throttle = DiagnosticThrottle(clock, repeat_seconds=STAGING_DEFER_REPEAT_SECONDS)
        self.model_recently_missing: bool = False
        """The head's model was expected resident but no process held it; a fresh preload was released."""
        self.model_recently_missing_at: float = 0.0
        self.stall_reason: str | None = None
        """The constraint last recorded as holding the head back, or None while dispatch flows."""
        self._stall_logged_at: float = 0.0
        self.heavy_head_admitted_at: float = 0.0
        """When a heavy head was last admitted off the whole-card path; 0.0 when none is loading."""

    # ---- head starvation

    def track_head_starvation(self, head_id: str | None, *, work_in_progress: bool) -> None:
        """Time how long the current head has sat on an otherwise idle device.

        The clock runs only while no live job holds the device: a head waiting behind such work is queued, not
        starved. It restarts when a different job reaches the front so it measures this head's wait, not the
        queue's age.
        """
        if head_id is None or work_in_progress:
            self.starvation_job_id = None
            self.starvation_since = 0.0
            return
        if head_id != self.starvation_job_id:
            self.starvation_job_id = head_id
            self.starvation_since = self._clock()

    def starved_seconds(self, job_id: str | None) -> float:
        """Seconds ``job_id`` has been the idle-device head, or 0.0 when it is not the timed head."""
        if job_id is None or job_id != self.starvation_job_id or self.starvation_since == 0.0:
            return 0.0
        return self._clock() - self.starvation_since

    def clear_head_starvation(self) -> None:
        """Stop timing: a job dispatched, so the wedge, if any, is broken."""
        self.starvation_job_id = None
        self.starvation_since = 0.0

    # ---- head RAM defer and the dispatch barrier

    def govern_ram_defer(
        self,
        job_id: str,
        *,
        made_reclaim_progress: bool,
        reclaim_grace_active: bool,
    ) -> HeadRamDeferStep:
        """Advance the RAM-defer clock for a head deferred behind live work and say what follows.

        The clock starts at this head's first such defer and restarts on reclaim progress or a change of head,
        since only a head that keeps deferring with nothing freed is starving. Progress for the barred head
        releases the barrier. Past the engage bound the barrier latches, unless a deliberate reclaim cycle is
        still inside its grace; a barrier past its cap declines the head.
        """
        now = self._clock()
        if made_reclaim_progress or job_id != self.ram_defer_job_id:
            self.ram_defer_job_id = job_id
            self.ram_defer_since = now
            if made_reclaim_progress and self.barrier_job_id == job_id:
                return HeadRamDeferStep.RELEASE_BARRIER
            return HeadRamDeferStep.NONE
        if self.barrier_job_id == job_id and (now - self.barrier_since) >= HEAD_RAM_DEFER_BARRIER_CAP_SECONDS:
            return HeadRamDeferStep.DECLINE_HEAD
        if (now - self.ram_defer_since) < HEAD_RAM_DEFER_BARRIER_SECONDS or reclaim_grace_active:
            return HeadRamDeferStep.NONE
        return HeadRamDeferStep.ENGAGE_BARRIER

    def clear_ram_defer(self, job_id: str | None) -> None:
        """Forget the defer clock if it is timing ``job_id``."""
        if job_id is not None and job_id == self.ram_defer_job_id:
            self.ram_defer_job_id = None
            self.ram_defer_since = 0.0

    def resolve_ram_defer(self, job_id: str | None) -> bool:
        """Clear the defer clock for an admitted head; return whether the barrier is held for it."""
        self.clear_ram_defer(job_id)
        return job_id is not None and job_id == self.barrier_job_id

    def reconcile_to_head(self, head_id: str | None) -> bool:
        """Drop state for a head that left the queue front; return whether a barrier for it must release."""
        if self.ram_defer_job_id is not None and self.ram_defer_job_id != head_id:
            self.ram_defer_job_id = None
            self.ram_defer_since = 0.0
        return self.barrier_job_id is not None and self.barrier_job_id != head_id

    def engage_barrier(self, job_id: str) -> bool:
        """Latch the barrier for ``job_id``; False when it is already held for that job."""
        if self.barrier_job_id == job_id:
            return False
        self.barrier_job_id = job_id
        self.barrier_since = self._clock()
        self.barrier_withhold_logged = False
        return True

    def release_barrier(self) -> str | None:
        """Release the barrier, returning the job it was held for, or None when none was held."""
        released = self.barrier_job_id
        if released is None:
            return None
        self.barrier_job_id = None
        self.barrier_since = 0.0
        self.barrier_withhold_logged = False
        return released

    def barrier_withholds(self, job_id: str | None) -> bool:
        """Whether the barrier withholds dispatching ``job_id``: every job but the barred head, while held."""
        if self.barrier_job_id is None:
            return False
        return job_id != self.barrier_job_id

    # ---- safety-recovery admission hold

    def recovery_hold_step(self, *, crash_looping: bool, targets_card: bool) -> SafetyRecoveryHoldStep:
        """Advance the hold for one admission query.

        The hold engages when the safety pool is crash-looping with a deferred GPU start on the queried card,
        releases when the pool recovers, and expires at its TTL, latching the episode so a permanently stuck pool
        does not re-hold one preload per TTL window. A query for another card neither engages nor releases.
        """
        if not crash_looping:
            was_engaged = self.recovery_hold_since != 0.0
            self.recovery_hold_since = 0.0
            self.recovery_hold_expired = False
            return SafetyRecoveryHoldStep.RELEASED if was_engaged else SafetyRecoveryHoldStep.NOT_HOLDING
        if not targets_card or self.recovery_hold_expired:
            return SafetyRecoveryHoldStep.NOT_HOLDING
        now = self._clock()
        engaged_now = self.recovery_hold_since == 0.0
        if engaged_now:
            self.recovery_hold_since = now
        if (now - self.recovery_hold_since) >= SAFETY_RECOVERY_HOLD_TTL_SECONDS:
            self.recovery_hold_since = 0.0
            self.recovery_hold_expired = True
            return SafetyRecoveryHoldStep.EXPIRED
        return SafetyRecoveryHoldStep.ENGAGED if engaged_now else SafetyRecoveryHoldStep.HOLDING

    # ---- admission records

    def record_preload_admission(
        self,
        decision: AdmissionDecision,
        *,
        model: str | None,
        process_id: int | None,
        reason: str,
    ) -> None:
        """Remember one preload-admission decision for the status surfaces."""
        self.last_preload_admission = LatestPreloadAdmission(
            decision=decision,
            model=model,
            process_id=process_id,
            reason=reason,
            timestamp=self._clock(),
        )

    def admission_denials(self, device_index: int | None) -> int:
        """Measured-floor admission denials on a card this run (worker-wide under key 0)."""
        return self.admission_denials_by_device.get(device_index if device_index is not None else 0, 0)

    def admission_headroom_mb(self, device_index: int | None) -> float | None:
        """The last measured-floor admission headroom (MB) on a card, or None when the floor was unapplied."""
        return self.admission_headroom_mb_by_device.get(device_index if device_index is not None else 0)

    def note_admission_headroom(self, device_index: int | None, headroom_mb: float | None) -> None:
        """Record the measured-floor headroom the admission overlay computed for a card this cycle."""
        self.admission_headroom_mb_by_device[device_index if device_index is not None else 0] = headroom_mb

    def note_staging_defer(self, reason: StagingDeferReason) -> int | None:
        """Tally a staging deferral; return the suppressed repeat count when it should be logged, else None.

        The reason edge is what carries information; an unchanged run is restated only every
        :data:`STAGING_DEFER_REPEAT_SECONDS` with the count it stood for.
        """
        self.staging_defers[reason] = self.staging_defers.get(reason, 0) + 1
        return self._staging_throttle.suppressed_count("staging_defer", reason)

    # ---- missing-model recovery latch

    def latch_missing_model(self) -> None:
        """Record that the head's model was found missing and a fresh preload was released for it."""
        self.model_recently_missing = True
        self.model_recently_missing_at = self._clock()

    def clear_missing_model(self) -> None:
        """A dispatch selection committed, so the recovery the latch guarded has run its course."""
        self.model_recently_missing = False

    def missing_model_latched(self, budget_seconds: float) -> bool:
        """Whether the latch still holds within ``budget_seconds`` of being set.

        No dispatch is guaranteed to follow a recovery, so the latch expires on the preload budget: the fresh
        preload either lands inside that window or has failed. Unbounded, one recovery would bar every later one.
        """
        if not self.model_recently_missing:
            return False
        return (self._clock() - self.model_recently_missing_at) < budget_seconds

    # ---- dispatch stall diagnostic

    def note_stall(self, reason: str) -> bool:
        """Record why the head is parked; return whether the diagnostic should be logged now.

        A changed reason logs immediately; an unchanged one at most once per
        :data:`DISPATCH_STALL_LOG_INTERVAL_SECONDS`.
        """
        now = self._clock()
        if reason == self.stall_reason and (now - self._stall_logged_at) < DISPATCH_STALL_LOG_INTERVAL_SECONDS:
            return False
        self.stall_reason = reason
        self._stall_logged_at = now
        return True

    def clear_stall(self) -> None:
        """A job dispatched, so the recorded stall reason is stale."""
        self.stall_reason = None

    # ---- heavy head load grace

    def note_heavy_head_admitted(self) -> None:
        """A heavy head was admitted off the whole-card path; its load holds the queue for a bounded window."""
        self.heavy_head_admitted_at = self._clock()

    def heavy_head_load_grace_active(self, grace_seconds: float) -> bool:
        """Whether the last heavy-head admission is still inside its bounded load window."""
        if self.heavy_head_admitted_at == 0.0:
            return False
        return (self._clock() - self.heavy_head_admitted_at) < grace_seconds
