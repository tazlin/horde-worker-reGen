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

import dataclasses
import enum
from collections.abc import Callable
from typing import Protocol

from loguru import logger

_TAIL_OVERLAP_PROGRESS_THRESHOLD = 0.8
"""How far the most-advanced in-flight sampler must be through its denoise loop before the parent clears the
next staged child early. High enough that the outgoing job finishes soon after the incoming one begins (so the
two sampling windows only briefly overlap) rather than admitting a full second denoise."""

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


@dataclasses.dataclass(frozen=True)
class ClearancePlan:
    """The clearance intents for one tick: which staged children to attempt to clear, and any tail-overlap id."""

    clear_process_ids: tuple[int, ...] = ()
    """Staged children to attempt to clear this tick, in priority order (steady slots plus any tail bonus)."""
    tail_cleared_for_job_id: str | None = None
    """The outgoing sampler's job id a tail-overlap early clear is bound to this tick, for one-per-job dedup."""


def decide_clearances(
    inputs: ClearanceInputs,
    *,
    slot_cap: int,
    held_grant_count: int,
    tail_overlap_enabled: bool,
    tail_overlap_progress_threshold: float,
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
    decides who samples next. Tail overlap (only when ``tail_overlap_enabled``) adds exactly one extra grant
    for one handoff window: when the most advanced sampler is at or past ``tail_overlap_progress_threshold``, a
    staged waiter exists, the card is not paging, and measured free net of the reserve clears
    ``tail_overlap_margin_mb``, one more waiter is cleared early so the next sampling window opens before the
    outgoing one closes. That early clear is bound to the outgoing sampler's job id and suppressed while that id
    is in ``already_tail_cleared_job_ids``, so a given outgoing sampler triggers at most one early clear.
    """
    base_slots = max(0, slot_cap - held_grant_count)

    tail_cleared_for_job_id: str | None = None
    tail_bonus = 0
    if tail_overlap_enabled and inputs.staged_waiters and inputs.active_grants and not inputs.paging_active:
        most_advanced = max(inputs.active_grants, key=lambda grant: grant.progress_fraction)
        room = (
            inputs.device_free_mb is not None
            and inputs.device_free_mb - inputs.vram_reserve_mb >= tail_overlap_margin_mb
        )
        if (
            most_advanced.progress_fraction >= tail_overlap_progress_threshold
            and room
            and most_advanced.job_id not in already_tail_cleared_job_ids
        ):
            tail_bonus = 1
            tail_cleared_for_job_id = most_advanced.job_id

    available = base_slots + tail_bonus
    if available <= 0 or not inputs.staged_waiters:
        return ClearancePlan()

    ordered = sorted(inputs.staged_waiters, key=lambda waiter: waiter.priority)
    chosen = tuple(waiter.process_id for waiter in ordered[:available])
    if not chosen:
        return ClearancePlan()
    # The tail bonus only applies when its extra slot is actually used by a chosen waiter.
    if tail_bonus and len(chosen) <= base_slots:
        tail_cleared_for_job_id = None
    return ClearancePlan(clear_process_ids=chosen, tail_cleared_for_job_id=tail_cleared_for_job_id)


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
        tail_overlap_progress_threshold: float = _TAIL_OVERLAP_PROGRESS_THRESHOLD,
        tail_overlap_margin_mb: float = _TAIL_OVERLAP_MARGIN_MB,
    ) -> None:
        """Configure the controller for one card.

        Args:
            device_index: The card this controller governs (for logging).
            slot_cap: The steady-state cap on concurrent grants (the configured sampling-lease slot count).
            tail_overlap: Whether the one-extra-grant handoff window is enabled for this card.
            tail_overlap_progress_threshold: The outgoing sampler's tail fraction that opens an early clear.
            tail_overlap_margin_mb: The measured free-minus-reserve headroom an early clear requires.
        """
        self._device_index = device_index
        self._slot_cap = slot_cap
        self._tail_overlap = tail_overlap
        self._tail_overlap_progress_threshold = tail_overlap_progress_threshold
        self._tail_overlap_margin_mb = tail_overlap_margin_mb
        self._proxies: dict[int, ClearanceLeaseProxy] = {}
        self._grant_state: dict[int, GrantState] = {}
        # The job id each held grant was issued for, so a grant is retired exactly when its child moves off that
        # job (finished it, or picked up a different one), never by a stale done permit from a prior job.
        self._granted_job_id: dict[int, str] = {}
        # Outgoing sampler job ids an early clear already fired for, so a tail overlap fires once per job.
        self._tail_cleared_job_ids: set[str] = set()
        # Processes flagged as sampling without a recorded grant, so the unpriced warning is edge-triggered.
        self._unpriced_flagged: set[int] = set()

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
            tail_overlap_progress_threshold=self._tail_overlap_progress_threshold,
            tail_overlap_margin_mb=self._tail_overlap_margin_mb,
            already_tail_cleared_job_ids=frozenset(self._tail_cleared_job_ids),
        )

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
