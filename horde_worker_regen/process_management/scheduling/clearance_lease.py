"""Per-process GPU denoise clearance lease: the parent clears each child into its VRAM-load-plus-sample window.

The sampling lease brackets each job's diffusion-model VRAM load and its denoise loop. Under the earlier
single shared per-card semaphore a staged child would autonomously grab a freed permit and immediately load
its 8-10GB of weights while the outgoing job's weights were still resident, tipping heavy pairs into WDDM
demand-paging: the parent had no checkpoint at the true VRAM moment. This module moves the lease to a
per-child clearance handshake so the parent decides, against measured device truth, exactly when each child
may enter its load-and-sample window.

Each inference child receives a :class:`ClearanceLeaseProxy` as its ``gpu_sampling_lease``. The proxy holds
two parent-created semaphores: ``clearance`` (the child blocks on it around ``comfy.sample.sample`` until the
parent grants one permit) and ``done`` (the child signals it on release so the parent learns the sampling
window closed). The child stages its pipeline (checkpoint disk load, prompt encode) freely up to the sample
call, then waits for clearance; the parent clears the best staged waiter once its full materialisation fits.
A multi-sample job (hires-fix, refiner) consumes one grant for its whole job: the second and later
``acquire`` calls pass through so a single job never blocks twice.

The decision of *which* child to clear and *when* is split house-style: :func:`decide_clearances` is a pure
function of a per-tick snapshot, and :class:`ClearanceController` owns the semaphore edge, the per-process
grant state, and the degradation latches. Liveness beats pricing: hordelib's bounded lease-acquire timeout
means a clearance-starved child eventually samples without a grant, so the controller detects an unpriced
sampling window and logs it once rather than ever wedging the pool.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import time
from collections.abc import Callable, Mapping
from typing import Protocol

from loguru import logger

TAIL_OVERLAP_MIN_PROGRESS_FOR_ESTIMATE = 0.15
"""How far through its denoise loop an in-flight sampler must be before its remaining sampling time is
estimated at all. A rate taken from the first step or two of a job is dominated by the one-off cost of entering
the loop, so an estimate below this fraction would be noise; a sampler under it reports no estimate and the
handoff is denied as progress-unknown rather than fired on a guess."""

_TAIL_OVERLAP_PAD_SECONDS = 1.0
"""Slack added to the incoming child's estimated load time when sizing the handoff window. Covers the tick
granularity of the parent's control loop and the child's own wake latency, so the incoming load starts a little
before the outgoing sampler's tail rather than a little after it."""

_DEFAULT_INCOMING_LOAD_SECONDS = 6.0
"""Assumed RAM-to-VRAM weight-load seconds for an incoming child when the parent has measured none yet.
Deliberately on the long side of a typical SDXL upload: over-estimating opens the handoff window earlier, which
the headroom clause still gates, whereas under-estimating leaves the load outside the outgoing tail entirely,
which is the failure this handoff exists to remove."""

_TAIL_OVERLAP_DENIAL_REPEAT_SECONDS = 30.0
"""How long an unchanged tail-overlap denial reason is suppressed before it is restated. The gate is evaluated
every control-loop tick, so only the reason *edge* is interesting; the periodic restatement keeps a persistent
starvation visible in a log that is otherwise silent about it."""

_TAIL_OVERLAP_MARGIN_MB = 3072.0
"""Measured free-VRAM headroom (device free minus the configured reserve) required before an early clear. The
incoming job's weights are typically already RAM-staged in its primed child, so the overlap cost is its early
activation working set plus any load remainder; this margin bounds that transient plus allocator fragmentation
and foreign churn, so the brief overlap does not tip the card into WDDM demand-paging."""

CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS = 60.0
"""How long a child blocks on its clearance grant before sampling anyway (hordelib's lease-acquire timeout).
Set well below the inference step-timeout kill deadline so a clearance-starved child degrades into unpriced
sampling, resumes emitting step heartbeats, and is never mistaken for a hung process. The parent's hung-process
watchdog extends a not-yet-cleared child's first-step grace by this same window so it is never killed while the
controller is legitimately holding it."""


class _ClearanceSemaphore(Protocol):
    """The subset of a multiprocessing semaphore the clearance handshake drives."""

    def acquire(self, block: bool = ..., timeout: float | None = ...) -> bool:
        """Acquire one permit; with ``block=False`` return whether a permit was available."""
        ...

    def release(self) -> None:
        """Return one permit."""
        ...


class ClearanceLeaseProxy:
    """The child-side sampling lease: block on the parent's clearance grant, signal when the window closes.

    Handed to one inference child as its ``gpu_sampling_lease`` and registered with hordelib, so it satisfies
    the same ``acquire(block, timeout) -> bool`` / ``release() -> None`` protocol hordelib wraps around
    ``comfy.sample.sample``. It additionally exposes :meth:`begin_job`, called by the child at job start, so a
    single grant covers a whole job's samples.

    The two semaphores are created by the parent so the parent can grant (release ``clearance``) and observe
    completion (drain ``done``). The proxy is picklable and shared into the child by spawn inheritance, exactly
    as the multiprocessing semaphores it wraps are. The per-job ``consumed`` flag is child-local instance
    state: the parent and child hold distinct unpickled copies and never read each other's flag, so the parent
    tracks grants through the controller instead.
    """

    def __init__(self, *, clearance: _ClearanceSemaphore, done: _ClearanceSemaphore) -> None:
        """Wrap the parent-created clearance and done semaphores.

        Args:
            clearance: A bounded semaphore the parent holds empty (its single permit acquired at creation) and
                releases to grant one child one load-and-sample window.
            done: A semaphore the child releases when a sampling window closes, so the parent (draining it
                non-blockingly) learns the grant was consumed and retired.
        """
        self._clearance = clearance
        self._done = done
        self._grant_consumed = False

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        """Wait for the parent's clearance grant, or pass through if this job already consumed one.

        The first ``acquire`` of a job blocks on the clearance permit up to ``timeout``; a later ``acquire``
        for the same job (a multi-sample workflow's subsequent sample) returns immediately so one job never
        waits for two grants. A blocked acquire that times out still marks the job's grant consumed: the job
        proceeds to sample unpriced (hordelib's degraded path), and its remaining samples must not each pay
        the timeout again. :meth:`begin_job` clears the flag for the next job.
        """
        if self._grant_consumed:
            return True
        acquired = bool(self._clearance.acquire(block, timeout))
        # Consume the per-job grant on any completed attempt, granted or timed out: either way the job now
        # samples, and its later samples must pass through rather than block again.
        self._grant_consumed = True
        return acquired

    def release(self) -> None:
        """Signal the parent that a sampling window closed; never touches the clearance permit.

        Releasing ``done`` (rather than returning a clearance permit) keeps granting one-directional: the
        parent alone decides the next grant, so a child releasing here can never hand itself a second window.
        """
        self._done.release()

    def begin_job(self) -> None:
        """Reset the per-job grant flag so the next job waits for its own clearance grant.

        Called by the child at job start, before the pipeline runs. Job execution in a child is
        single-threaded, so this un-consumes the grant exactly once per job with no race.
        """
        self._grant_consumed = False

    def _parent_grant(self) -> None:
        """Parent-side: release one clearance permit to grant this child a load-and-sample window.

        A bounded clearance semaphore raises ``ValueError`` if a permit is already available (a prior grant
        the child has not consumed), which the controller treats as already-cleared.
        """
        self._clearance.release()

    def _parent_drain_done(self) -> bool:
        """Parent-side: non-blockingly take one done signal, returning whether one was present."""
        return bool(self._done.acquire(False))


class GrantState(enum.Enum):
    """A registered child's clearance-grant state, owned by :class:`ClearanceController`."""

    IDLE = "idle"
    """No grant outstanding: the child holds no window and does not occupy a steady-state slot."""
    CLEARED = "cleared"
    """The parent released this child's clearance permit; the child has not yet entered its denoise loop."""
    SAMPLING = "sampling"
    """The child consumed its grant and is inside its denoise loop (observed in ``INFERENCE_STARTING``)."""


@dataclasses.dataclass(frozen=True)
class ClearanceWaiter:
    """One staged child that has primed its next job and is waiting for a clearance grant."""

    process_id: int
    """The child's logical slot id."""
    priority: int
    """Head-of-queue order (lower is closer to the head); ties broken by residency/affinity above this layer."""
    job_id: str | None = None
    """The staged job's id, so the controller can correlate a grant to the job it was issued for and retire it
    when the child moves on to a different job (or none). None when the primed child has no referenced job."""


@dataclasses.dataclass(frozen=True)
class ActiveSampler:
    """One child currently holding a grant (cleared or sampling), for the slot count and tail-overlap gate."""

    process_id: int
    """The child's logical slot id."""
    job_id: str
    """The granted job's id, binding a tail-overlap early clear to its outgoing sampler for one-per-job dedup."""
    progress_fraction: float
    """Denoise progress in ``[0.0, 1.0]``; a cleared-but-not-yet-sampling child reads ``0.0``."""
    remaining_sampling_seconds: float | None = None
    """Estimated wall seconds left in this sampler's denoise loop, or None while no rate can be trusted.

    Computed by the scheduler from the child's reported step position and the elapsed time since its first
    step. None until the job is far enough in (see :data:`TAIL_OVERLAP_MIN_PROGRESS_FOR_ESTIMATE`) for the
    rate to mean anything, and the tail-overlap gate never fires on a sampler without it."""


@dataclasses.dataclass(frozen=True)
class ClearanceInputs:
    """The per-tick truth the clearance decision reads, gathered by the scheduler for one card.

    A frozen snapshot so :func:`decide_clearances` is a pure function of its inputs. The controller feeds the
    grant populations it owns (``active_grants``) alongside the scheduler's staged-waiter and measured-VRAM
    view.
    """

    staged_waiters: tuple[ClearanceWaiter, ...]
    """Idle staged children waiting for a slot, in head-of-queue priority order."""
    active_grants: tuple[ActiveSampler, ...]
    """Children holding a grant (cleared or sampling); their count is the occupied steady-state slots."""
    device_free_mb: float | None
    """The parent's measured device-free VRAM (MB) for this card, or None when unread."""
    vram_reserve_mb: float
    """The configured VRAM reserve (MB) held back from admission."""
    paging_active: bool
    """Whether the parent's WDDM demand-paging detector is flagging this worker's allocations."""
    incoming_load_seconds: float | None = None
    """Measured RAM-to-VRAM weight-load seconds an incoming child is expected to pay, or None when unmeasured.

    The recent median of observed weight uploads on this card; the tail-overlap window is sized from it so the
    incoming load lands inside the outgoing sampler's tail. ``None`` falls back to
    :data:`_DEFAULT_INCOMING_LOAD_SECONDS`."""


class TailOverlapDenialReason(enum.Enum):
    """Why the tail-overlap handoff did not fire on a tick where it could conceptually have fired.

    Recorded whenever the feature is enabled and at least one child holds a grant, so a session's log and
    counters name the clause that is actually starving the handoff rather than only showing its absence.
    """

    NO_STAGED_WAITER = "no-waiter"
    """No staged child was waiting, so there was nobody to hand off to."""
    PAGING_ACTIVE = "paging"
    """The card is demand-paging; an overlap would deepen it."""
    ALREADY_GRANTED = "already-granted"
    """Every candidate outgoing sampler already triggered its one early clear (the one-per-job dedup)."""
    PROGRESS_UNKNOWN = "progress-unknown"
    """No active sampler is far enough along for its remaining sampling time to be estimated."""
    REMAINING_ABOVE_WINDOW = "remaining"
    """The soonest-finishing sampler has more time left than the incoming child's load needs to hide in."""
    HEADROOM_SHORT = "headroom"
    """Measured free VRAM net of the reserve does not cover the overlap margin."""
    BONUS_SLOT_UNUSED = "slot-unused"
    """The handoff slot opened but steady-state slots already covered every staged waiter.

    A statement of applicability rather than a refusal: every clause passed and the card simply had room to
    serve the queue without the bonus. Tallied separately from the refusing clauses for that reason, since a
    lightly loaded worker reaches this every tick and would otherwise swamp the shares that name a starver."""


@dataclasses.dataclass(frozen=True)
class TailOverlapDenial:
    """One tick's tail-overlap denial: the clause that failed first, with the arithmetic behind it."""

    reason: TailOverlapDenialReason
    """The first clause that failed, in evaluation order."""
    progress_fraction: float | None = None
    """The candidate sampler's denoise progress, where one was identified."""
    remaining_seconds: float | None = None
    """The candidate sampler's estimated remaining sampling seconds, where one could be estimated."""
    load_estimate_seconds: float | None = None
    """The incoming load estimate the remaining time was compared against."""
    headroom_shortfall_mb: float | None = None
    """How many megabytes short of the overlap margin the measured headroom was, for a headroom denial."""

    def describe(self) -> str:
        """A compact human-readable statement of the denial and its measured quantities."""
        parts: list[str] = [self.reason.value]
        if self.remaining_seconds is not None:
            parts.append(f"remaining {self.remaining_seconds:.1f}s")
        if self.load_estimate_seconds is not None:
            parts.append(f"load estimate {self.load_estimate_seconds:.1f}s")
        if self.progress_fraction is not None:
            parts.append(f"progress {self.progress_fraction:.2f}")
        if self.headroom_shortfall_mb is not None:
            parts.append(f"{self.headroom_shortfall_mb:.0f}MB short of margin")
        return ", ".join(parts)


@dataclasses.dataclass(frozen=True)
class ClearancePlan:
    """The clearance intents for one tick: which staged children to attempt to clear, and any tail-overlap id."""

    clear_process_ids: tuple[int, ...] = ()
    """Staged children to attempt to clear this tick, in priority order (steady slots plus any tail bonus)."""
    tail_cleared_for_job_id: str | None = None
    """The outgoing sampler's job id a tail-overlap early clear is bound to this tick, for one-per-job dedup."""
    tail_denial: TailOverlapDenial | None = None
    """Why the handoff did not fire, when it was applicable and did not; None when it fired or did not apply."""


def _evaluate_tail_overlap(
    inputs: ClearanceInputs,
    *,
    tail_overlap_pad_seconds: float,
    tail_overlap_margin_mb: float,
    already_tail_cleared_job_ids: frozenset[str],
) -> tuple[str | None, TailOverlapDenial | None]:
    """Pick the outgoing sampler whose tail can hide the incoming load, or say which clause refused.

    Returns the outgoing job id to bind the early clear to, or a denial naming the first failing clause.
    Clauses are evaluated cheapest-and-most-fundamental first (is there anyone to hand off to, is the card
    healthy, is a candidate still eligible) before the timing comparison, and headroom last: headroom only
    matters once the timing says a handoff is due, so evaluating it earlier would mask the timing answer.
    """
    if not inputs.staged_waiters:
        return None, TailOverlapDenial(TailOverlapDenialReason.NO_STAGED_WAITER)
    if inputs.paging_active:
        return None, TailOverlapDenial(TailOverlapDenialReason.PAGING_ACTIVE)

    eligible = [grant for grant in inputs.active_grants if grant.job_id not in already_tail_cleared_job_ids]
    if not eligible:
        return None, TailOverlapDenial(TailOverlapDenialReason.ALREADY_GRANTED)

    timed = [
        (grant.remaining_sampling_seconds, grant) for grant in eligible if grant.remaining_sampling_seconds is not None
    ]
    if not timed:
        best_progress = max(grant.progress_fraction for grant in eligible)
        return None, TailOverlapDenial(TailOverlapDenialReason.PROGRESS_UNKNOWN, progress_fraction=best_progress)

    # The soonest-finishing sampler owns the next free slot, so its tail is the one the handoff hides in.
    remaining_seconds, outgoing = min(timed, key=lambda entry: entry[0])
    load_seconds = inputs.incoming_load_seconds
    if load_seconds is None:
        load_seconds = _DEFAULT_INCOMING_LOAD_SECONDS
    window_seconds = load_seconds + tail_overlap_pad_seconds
    if remaining_seconds > window_seconds:
        return None, TailOverlapDenial(
            TailOverlapDenialReason.REMAINING_ABOVE_WINDOW,
            progress_fraction=outgoing.progress_fraction,
            remaining_seconds=remaining_seconds,
            load_estimate_seconds=load_seconds,
        )

    headroom_mb = None if inputs.device_free_mb is None else inputs.device_free_mb - inputs.vram_reserve_mb
    if headroom_mb is None or headroom_mb < tail_overlap_margin_mb:
        return None, TailOverlapDenial(
            TailOverlapDenialReason.HEADROOM_SHORT,
            progress_fraction=outgoing.progress_fraction,
            remaining_seconds=remaining_seconds,
            load_estimate_seconds=load_seconds,
            headroom_shortfall_mb=tail_overlap_margin_mb - (headroom_mb if headroom_mb is not None else 0.0),
        )
    return outgoing.job_id, None


def decide_clearances(
    inputs: ClearanceInputs,
    *,
    slot_cap: int,
    held_grant_count: int,
    tail_overlap_enabled: bool,
    tail_overlap_pad_seconds: float,
    tail_overlap_margin_mb: float,
    already_tail_cleared_job_ids: frozenset[str],
) -> ClearancePlan:
    """Decide which staged children to attempt to clear this tick.

    Pure: the return depends only on the arguments. The controller owns ``already_tail_cleared_job_ids`` and
    ``held_grant_count`` and applies the plan at the semaphore edge (running the scheduler's full-price
    admission per chosen child).

    Steady state admits at most ``slot_cap`` concurrent grants: the available slots are ``slot_cap`` minus
    ``held_grant_count`` (every child the controller has cleared or that is sampling, including a cleared child
    not yet in its denoise loop, which ``active_grants`` cannot yet show), and the best
    (lowest-priority-number) staged waiters fill them in head-of-queue order so queue position, not residency,
    decides who samples next.

    Tail overlap (only when ``tail_overlap_enabled``) adds exactly one extra grant for one handoff window. The
    handoff exists to hide the incoming child's RAM-to-VRAM weight upload inside the outgoing sampler's tail,
    so it is triggered on *time*: the soonest-finishing sampler's estimated remaining seconds must be at or
    under the incoming load estimate plus ``tail_overlap_pad_seconds``. A progress-fraction trigger cannot do
    this job, because a fixed fraction is a window whose width scales with the job's own duration: the same
    fraction that opens seconds of overlap on a slow job opens a window on a fast job too narrow for the
    parent's sampling of child progress to land inside, so exactly the jobs whose loads dominate their
    wall time are the ones the handoff never fires for. Alongside the timing clause a staged waiter must
    exist, the card must not be paging, and measured free net of the reserve must clear
    ``tail_overlap_margin_mb``. The early clear is bound to the outgoing sampler's job id and suppressed while
    that id is in ``already_tail_cleared_job_ids``, so a given outgoing sampler triggers at most one early
    clear. When the gate is applicable (enabled, with at least one active grant) and does not fire, the plan
    carries a :class:`TailOverlapDenial` naming the clause that refused.
    """
    base_slots = max(0, slot_cap - held_grant_count)

    tail_cleared_for_job_id: str | None = None
    tail_denial: TailOverlapDenial | None = None
    tail_bonus = 0
    if tail_overlap_enabled and inputs.active_grants:
        tail_cleared_for_job_id, tail_denial = _evaluate_tail_overlap(
            inputs,
            tail_overlap_pad_seconds=tail_overlap_pad_seconds,
            tail_overlap_margin_mb=tail_overlap_margin_mb,
            already_tail_cleared_job_ids=already_tail_cleared_job_ids,
        )
        if tail_cleared_for_job_id is not None:
            tail_bonus = 1

    available = base_slots + tail_bonus
    if available <= 0 or not inputs.staged_waiters:
        return ClearancePlan(tail_denial=tail_denial)

    ordered = sorted(inputs.staged_waiters, key=lambda waiter: waiter.priority)
    chosen = tuple(waiter.process_id for waiter in ordered[:available])
    if not chosen:
        return ClearancePlan(tail_denial=tail_denial)
    # The tail bonus only applies when its extra slot is actually used by a chosen waiter.
    if tail_bonus and len(chosen) <= base_slots:
        tail_cleared_for_job_id = None
        tail_denial = TailOverlapDenial(TailOverlapDenialReason.BONUS_SLOT_UNUSED)
    return ClearancePlan(
        clear_process_ids=chosen,
        tail_cleared_for_job_id=tail_cleared_for_job_id,
        tail_denial=tail_denial,
    )


@dataclasses.dataclass(frozen=True)
class ClearanceStepResult:
    """What one controller tick actually did, for the scheduler's slot-duty attribution and logging."""

    cleared_process_ids: tuple[int, ...] = ()
    """Children granted a clearance permit this tick."""
    held_process_ids: tuple[int, ...] = ()
    """Children the decision chose but whose full-price admission did not fit; their empty slot is a hold."""


class ClearanceController:
    """Owns the per-child clearance grants and applies the clearance decision at the semaphore edge.

    Registered proxies (one per live inference child) supply the clearance/done semaphores. Each tick:
    completed windows are retired by draining every child's ``done`` non-blockingly; the snapshot's sampling
    onset advances cleared children to sampling and flags any child sampling without a recorded grant (the
    degraded timeout path) with a single edge warning; :func:`decide_clearances` then chooses staged waiters to
    clear, and each chosen child is admitted through the injected ``admit_fn`` (the scheduler's full-price fit
    against measured device truth, running eviction as needed). A child that fits is cleared once; a child that
    does not is held, and its wait is reported so the scheduler attributes the empty slot.

    The controller owns its grant accounting for the session and self-heals rather than propagating: a
    double-clear is guarded, a replaced child's state is discarded, and a released-then-reclaimed grant that
    the child never consumed is bounded by the clearance semaphore's own capacity. Nothing here blocks the
    control loop; the caller isolates a raised tick fail-inert.
    """

    def __init__(
        self,
        *,
        device_index: int,
        slot_cap: int,
        tail_overlap: bool,
        tail_overlap_pad_seconds: float = _TAIL_OVERLAP_PAD_SECONDS,
        tail_overlap_margin_mb: float = _TAIL_OVERLAP_MARGIN_MB,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Configure the controller for one card.

        Args:
            device_index: The card this controller governs (for logging).
            slot_cap: The steady-state cap on concurrent grants (the configured sampling-lease slot count).
            tail_overlap: Whether the one-extra-grant handoff window is enabled for this card.
            tail_overlap_pad_seconds: Slack added to the incoming load estimate when sizing the handoff window.
            tail_overlap_margin_mb: The measured free-minus-reserve headroom an early clear requires.
            clock: Monotonic seconds source for the denial diagnostic's repeat throttle; defaults to
                ``time.monotonic``.
        """
        self._device_index = device_index
        self._slot_cap = slot_cap
        self._tail_overlap = tail_overlap
        self._tail_overlap_pad_seconds = tail_overlap_pad_seconds
        self._tail_overlap_margin_mb = tail_overlap_margin_mb
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._proxies: dict[int, ClearanceLeaseProxy] = {}
        self._grant_state: dict[int, GrantState] = {}
        # The job id each held grant was issued for, so a grant is retired exactly when its child moves off that
        # job (finished it, or picked up a different one), never by a stale done permit from a prior job.
        self._granted_job_id: dict[int, str] = {}
        # Outgoing sampler job ids an early clear already fired for, so a tail overlap fires once per job.
        self._tail_cleared_job_ids: set[str] = set()
        # Processes flagged as sampling without a recorded grant, so the unpriced warning is edge-triggered.
        self._unpriced_flagged: set[int] = set()
        # Session tallies for the handoff: grants issued, and denials by the clause that refused. Read into
        # the duty-cycle summary so one session's log answers which clause starves the overlap.
        self._tail_overlap_grants = 0
        self._grants_issued = 0
        self._unpriced_sampling_windows = 0
        self._tail_overlap_denials: collections.Counter[TailOverlapDenialReason] = collections.Counter()
        # The denial reason last logged, when it was logged, and how many identical ticks were suppressed.
        self._tail_denial_log_state: tuple[TailOverlapDenialReason, float, int] | None = None

    def register(self, process_id: int, proxy: ClearanceLeaseProxy) -> None:
        """Register a freshly spawned child's proxy so its clearance can be granted and its done drained."""
        self._proxies[process_id] = proxy
        self._grant_state[process_id] = GrantState.IDLE
        self._granted_job_id.pop(process_id, None)
        self._unpriced_flagged.discard(process_id)

    def note_child_replaced(self, process_id: int) -> None:
        """Discard a replaced or dead child's grant state; its per-child semaphores die with the process.

        Under the per-child lease the parent holds no shared permit on a dead child's behalf, so there is
        nothing to release: dropping the state (and any tail-overlap binding tied to a grant it held) is the
        whole reconciliation. A replacement child registers fresh.
        """
        self._proxies.pop(process_id, None)
        self._grant_state.pop(process_id, None)
        self._granted_job_id.pop(process_id, None)
        self._unpriced_flagged.discard(process_id)

    def grant_state(self, process_id: int) -> GrantState:
        """The current grant state for a registered child, or ``IDLE`` when unknown."""
        return self._grant_state.get(process_id, GrantState.IDLE)

    @property
    def held_grant_count(self) -> int:
        """How many registered children currently hold a grant (cleared or sampling)."""
        return sum(1 for state in self._grant_state.values() if state is not GrantState.IDLE)

    def step(
        self,
        inputs: ClearanceInputs,
        *,
        admit_fn: Callable[[int], bool],
    ) -> ClearanceStepResult:
        """Retire completed windows, reconcile sampling onset, and clear the chosen staged waiters.

        Args:
            inputs: The per-tick snapshot; its ``active_grants`` is derived from this controller's own held
                children (the scheduler reads :meth:`grant_state` to build it).
            admit_fn: The scheduler's full-price admission for a chosen child: returns whether the child's job
                materialisation fits the card now (running eviction as a side effect), so clearance is the true
                VRAM moment. A child that does not fit is held, not cleared.

        Returns:
            Which children were cleared and which were held, for slot-duty attribution.
        """
        self._drain_done_discard()
        self._reconcile_grants(inputs)
        self._prune_tail_dedup(inputs)

        # Only children the controller still holds IDLE are clearable; a cleared-but-not-yet-sampling child is
        # still reported as a primed waiter by the scheduler snapshot, so filter it out here rather than let a
        # slot pick be wasted on it (the per-child clear guard below would skip it anyway).
        job_id_by_pid = {waiter.process_id: waiter.job_id for waiter in inputs.staged_waiters}
        idle_waiters = tuple(
            waiter
            for waiter in inputs.staged_waiters
            if self._grant_state.get(waiter.process_id, GrantState.IDLE) is GrantState.IDLE
        )
        effective_inputs = dataclasses.replace(inputs, staged_waiters=idle_waiters)

        plan = decide_clearances(
            effective_inputs,
            slot_cap=self._slot_cap,
            held_grant_count=self.held_grant_count,
            tail_overlap_enabled=self._tail_overlap,
            tail_overlap_pad_seconds=self._tail_overlap_pad_seconds,
            tail_overlap_margin_mb=self._tail_overlap_margin_mb,
            already_tail_cleared_job_ids=frozenset(self._tail_cleared_job_ids),
        )
        if plan.tail_denial is not None:
            self._note_tail_overlap_denial(plan.tail_denial)

        cleared: list[int] = []
        held: list[int] = []
        for process_id in plan.clear_process_ids:
            if self._grant_state.get(process_id, GrantState.IDLE) is not GrantState.IDLE:
                # Already holds a grant; do not clear twice.
                continue
            if admit_fn(process_id):
                if self._clear(process_id, job_id_by_pid.get(process_id)):
                    cleared.append(process_id)
                    if plan.tail_cleared_for_job_id is not None and process_id == plan.clear_process_ids[-1]:
                        self._tail_cleared_job_ids.add(plan.tail_cleared_for_job_id)
                        self._tail_overlap_grants += 1
                        # A fired handoff ends the denial run, so the next denial reason logs on its own edge.
                        self._tail_denial_log_state = None
                        self._log_tail_overlap_clear(inputs, outgoing_job_id=plan.tail_cleared_for_job_id)
            else:
                held.append(process_id)

        return ClearanceStepResult(cleared_process_ids=tuple(cleared), held_process_ids=tuple(held))

    def _clear(self, process_id: int, job_id: str | None) -> bool:
        """Release one child's clearance permit exactly once, guarding a double clear. Returns success.

        Records the job the grant is issued for so the grant is later retired by job correlation (the child
        moving off that job), never by a stale done permit from a prior job.
        """
        proxy = self._proxies.get(process_id)
        if proxy is None:
            return False
        try:
            proxy._parent_grant()
        except ValueError:
            # The child's clearance permit is already available (a prior grant it has not consumed, e.g. the
            # degraded timeout path). The bounded semaphore caps it at one, so treat it as already cleared.
            self._grant_state[process_id] = GrantState.CLEARED
            if job_id is not None:
                self._granted_job_id[process_id] = job_id
            return False
        self._grant_state[process_id] = GrantState.CLEARED
        if job_id is not None:
            self._granted_job_id[process_id] = job_id
        self._grants_issued += 1
        logger.debug(
            f"Clearance lease on device {self._device_index}: cleared process {process_id} into its "
            f"load-and-sample window.",
        )
        return True

    def _log_tail_overlap_clear(self, inputs: ClearanceInputs, *, outgoing_job_id: str) -> None:
        """Emit the dedicated INFO signal for a tail-overlap early clear so its firing rate is measurable.

        The ordinary steady-state grant and the tail-overlap bonus grant share :meth:`_clear`'s debug line, so
        without this the bonus firing rate is invisible in the logs. Bound to the semaphore edge where the bonus
        is actually issued and gated by the one-per-job dedup, so it fires exactly once per outgoing sampler.
        Reports the outgoing sampler's denoise progress and the measured headroom (device free minus reserve)
        the bonus was granted against, the two quantities that decide whether the handoff window is well-tuned.
        """
        outgoing = next(
            (grant for grant in inputs.active_grants if grant.job_id == outgoing_job_id),
            None,
        )
        progress_fraction = outgoing.progress_fraction if outgoing is not None else 0.0
        headroom_mb = inputs.device_free_mb - inputs.vram_reserve_mb if inputs.device_free_mb is not None else 0.0
        logger.info(
            f"Clearance lease on device {self._device_index}: tail-overlap early clear granted behind outgoing "
            f"sampler {outgoing_job_id[:8]} at progress {progress_fraction:.2f} with {headroom_mb:.0f}MB measured "
            f"headroom.",
        )

    @property
    def grants_issued(self) -> int:
        """How many load-and-sample windows this controller has granted this session.

        The session tally against which sampling windows are read: a worker whose children enter more denoise
        loops than the parent granted is sampling unpriced, which is the condition
        :attr:`unpriced_sampling_windows` counts.
        """
        return self._grants_issued

    @property
    def unpriced_sampling_windows(self) -> int:
        """How many times a child was first seen inside a denoise loop holding no grant, this session.

        A child that waits out its bounded lease-acquire timeout samples anyway (liveness over pricing), and
        so does a child that never waits at all. Either way the parent priced nothing for that window, so a
        nonzero tally means the clearance handshake is not bracketing the work it is meant to bracket.
        """
        return self._unpriced_sampling_windows

    @property
    def tail_overlap_grant_count(self) -> int:
        """How many tail-overlap handoff grants this controller has issued this session."""
        return self._tail_overlap_grants

    @property
    def tail_overlap_denial_counts(self) -> Mapping[TailOverlapDenialReason, int]:
        """How many ticks each clause held the handoff back this session, keyed by the clause.

        Includes :attr:`TailOverlapDenialReason.BONUS_SLOT_UNUSED`, which records applicability rather than
        refusal; :func:`format_tail_overlap_tally` reports it apart from the refusing clauses.
        """
        return self._tail_overlap_denials

    def _note_tail_overlap_denial(self, denial: TailOverlapDenial) -> None:
        """Tally a denied handoff tick and log it when the refusing clause changes (or after a long run).

        The gate is evaluated every control-loop tick, so a per-tick line would be noise: the reason edge is
        what carries information, and a persistent run is restated only every
        ``_TAIL_OVERLAP_DENIAL_REPEAT_SECONDS`` with the count of ticks it stood for. Emitted at debug beside
        the per-child clearance line, since a denied handoff is ordinary operation rather than a fault.
        """
        self._tail_overlap_denials[denial.reason] += 1

        now = self._clock()
        suppressed = 0
        previous = self._tail_denial_log_state
        if previous is not None:
            previous_reason, previous_emit, previous_suppressed = previous
            if previous_reason == denial.reason and (now - previous_emit) < _TAIL_OVERLAP_DENIAL_REPEAT_SECONDS:
                self._tail_denial_log_state = (previous_reason, previous_emit, previous_suppressed + 1)
                return
            suppressed = previous_suppressed
        self._tail_denial_log_state = (denial.reason, now, 0)

        suffix = f" (suppressed {suppressed} unchanged repeats)" if suppressed > 0 else ""
        logger.debug(
            f"Clearance lease on device {self._device_index}: tail-overlap handoff denied "
            f"({denial.describe()}).{suffix}",
        )

    def _drain_done_discard(self) -> None:
        """Empty each child's done semaphore without using it to retire grants (bounded hygiene only).

        A child's ``release`` posts a done permit once per sample call, so a multi-sample job posts several for
        one grant, and a permit outlives the job that produced it. Retirement is therefore driven by job
        correlation in :meth:`_reconcile_grants`, not by counting these permits; draining here only keeps the
        semaphore from growing unbounded so it can never mis-retire a later grant.
        """
        for proxy in self._proxies.values():
            while True:
                try:
                    drained = proxy._parent_drain_done()
                except Exception:
                    drained = False
                if not drained:
                    break

    def _reconcile_grants(self, inputs: ClearanceInputs) -> None:
        """Retire, onset, or flag grants from the process snapshot by job correlation, not done permits.

        For each child holding a grant: if the snapshot no longer shows it staged or sampling *its granted
        job*, the child has moved on (finished that job or picked up a different one), so the grant retires and
        its slot reopens. A grant whose child is now sampling its granted job advances to sampling. A child
        sampling with no grant (its clearance timed out and it sampled anyway) is flagged once as an unpriced
        window. This is immune to a stale done permit retiring a fresh grant, the failure that wedged the pool.
        """
        primed_job_by_pid = {waiter.process_id: waiter.job_id for waiter in inputs.staged_waiters}
        sampling_job_by_pid = {grant.process_id: grant.job_id for grant in inputs.active_grants}

        for process_id in list(self._grant_state):
            state = self._grant_state[process_id]
            if state is GrantState.IDLE:
                continue
            in_sampling = process_id in sampling_job_by_pid
            in_primed = process_id in primed_job_by_pid
            if not in_sampling and not in_primed:
                # The child has left the staged-and-sampling states entirely (finished its job and went idle,
                # or died): retire the grant so its slot reopens. Immune to a stale done permit.
                self._retire_grant(process_id)
                continue
            granted_job = self._granted_job_id.get(process_id)
            current_job = sampling_job_by_pid.get(process_id) if in_sampling else primed_job_by_pid.get(process_id)
            if granted_job is not None and current_job is not None and current_job != granted_job:
                # The child moved on to a different job than this grant was issued for: retire the stale grant.
                self._retire_grant(process_id)
                continue
            if in_sampling and state is GrantState.CLEARED:
                self._grant_state[process_id] = GrantState.SAMPLING

        # A child sampling with no recorded grant sampled through its lease-acquire timeout (unpriced). Flag it
        # once (edge) and account it as a held slot so the slot cap is not overshot while it samples.
        for grant in inputs.active_grants:
            if self._grant_state.get(grant.process_id, GrantState.IDLE) is not GrantState.IDLE:
                continue
            if grant.progress_fraction <= 0.0:
                continue
            self._grant_state[grant.process_id] = GrantState.SAMPLING
            self._granted_job_id[grant.process_id] = grant.job_id
            if grant.process_id not in self._unpriced_flagged:
                self._unpriced_flagged.add(grant.process_id)
                self._unpriced_sampling_windows += 1
                logger.warning(
                    f"Clearance lease on device {self._device_index}: process {grant.process_id} entered its "
                    f"denoise loop without a recorded grant (unpriced sampling window); liveness preserved.",
                )

    def _retire_grant(self, process_id: int) -> None:
        """Return a held grant to idle so its steady-state slot reopens (grant accounting only)."""
        self._grant_state[process_id] = GrantState.IDLE
        self._granted_job_id.pop(process_id, None)
        self._unpriced_flagged.discard(process_id)

    def _prune_tail_dedup(self, inputs: ClearanceInputs) -> None:
        """Forget tail-overlap bindings whose outgoing sampler has left the active-grant set."""
        if not self._tail_cleared_job_ids:
            return
        live_job_ids = {grant.job_id for grant in inputs.active_grants}
        self._tail_cleared_job_ids &= live_job_ids


def format_tail_overlap_tally(
    granted: int,
    denials: Mapping[TailOverlapDenialReason, int],
    *,
    max_reasons: int = 3,
) -> str | None:
    """A compact handoff tally for the duty-cycle line, or None when the gate was never applicable.

    Reports grants against refusals and the share each clause took of those refusals, largest first, so a duty
    figure short of target can be read straight across to the clause holding the handoff shut.
    :attr:`TailOverlapDenialReason.BONUS_SLOT_UNUSED` is kept out of that arithmetic and trails the line in
    brackets: it records a tick where every clause passed and the steady slots simply covered the queue, so
    counting it as a refusal would drown the shares that name a starver on any lightly loaded worker.
    """
    slot_unused = denials.get(TailOverlapDenialReason.BONUS_SLOT_UNUSED, 0)
    refusals = {
        reason: count
        for reason, count in denials.items()
        if reason is not TailOverlapDenialReason.BONUS_SLOT_UNUSED and count
    }
    denied = sum(refusals.values())
    if granted == 0 and denied == 0 and slot_unused == 0:
        return None
    line = f"tail-overlap: {granted} granted / {denied} denied"
    if denied:
        ranked = sorted(refusals.items(), key=lambda entry: (-entry[1], entry[0].value))[:max_reasons]
        shares = ", ".join(f"{reason.value} {count / denied:.0%}" for reason, count in ranked)
        line = f"{line} ({shares})"
    if slot_unused:
        line = f"{line} [slot-unused {slot_unused}]"
    return line
