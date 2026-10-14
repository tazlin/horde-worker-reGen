"""Per-process GPU denoise clearance lease: the parent clears each child into its VRAM-load-plus-sample window.

The sampling lease brackets each job's diffusion-model VRAM load and its denoise loop. Under the earlier
single shared per-card semaphore a staged child would autonomously grab a freed permit and immediately load
its 8-10GB of weights while the outgoing job's weights were still resident, tipping heavy pairs into WDDM
demand-paging: the parent had no checkpoint at the true VRAM moment. This module moves the lease to a
per-child clearance handshake so the parent decides, against measured device truth, exactly when each child
may enter its load-and-sample window.

Each inference child receives a :class:`ClearanceLeaseProxy` as its ``gpu_sampling_lease``. The proxy holds
two parent-created semaphores: ``clearance`` (the child blocks on it around ComfyUI's guider sample call,
``comfy.samplers.CFGGuider.sample``, until the parent grants one permit) and ``done`` (the child signals it on
release so the parent learns the sampling window closed). The child stages its pipeline (checkpoint disk load,
prompt encode) freely up to the sample call, then waits for clearance; the parent clears the best staged
waiter once its full materialisation fits.
A multi-sample job (hires-fix, refiner) consumes one grant for its whole job: the second and later
``acquire`` calls pass through so a single job never blocks twice.

A proxy built with a ``stage`` semaphore also carries a stage permit, taken by the child before its sample
permit to upload the diffusion weights ahead of the denoise loop. The weights alone fit beside a running
sampler's peak where the whole sampling peak does not, so the parent can grant the stage permit to the head
waiter while another child samples and the denoise onset waits only on the sample permit. A single clear
grants both permits, so a child never waits on a stage permit longer than on today's clearance.

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
import math
import time
from collections.abc import Callable, Mapping
from typing import Protocol

from loguru import logger

from horde_worker_regen.process_management.scheduling.diagnostic_throttle import DiagnosticThrottle, suppressed_suffix

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


class _SharedFigure(Protocol):
    """A float shared between the parent and one child, such as a ``multiprocessing`` ``Value('d')``."""

    value: float
    """The shared figure; NaN when unset."""


class ClearanceLeaseProxy:
    """The child-side sampling lease: block on the parent's clearance grant, signal when the window closes.

    Handed to one inference child as its ``gpu_sampling_lease`` and registered with hordelib, so it satisfies
    the same ``acquire(block, timeout) -> bool`` / ``release() -> None`` protocol hordelib wraps around
    ``comfy.samplers.CFGGuider.sample``. It additionally exposes :meth:`begin_job`, called by the child at job
    start, so a single grant covers a whole job's samples.

    The two semaphores are created by the parent so the parent can grant (release ``clearance``) and observe
    completion (drain ``done``). The proxy is picklable and shared into the child by spawn inheritance, exactly
    as the multiprocessing semaphores it wraps are. The per-job ``consumed`` flag is child-local instance
    state: the parent and child hold distinct unpickled copies and never read each other's flag, so the parent
    tracks grants through the controller instead.

    A grant also carries the device free VRAM the parent measured when it admitted the child, written to a
    shared figure before the permit is released. The child's free-VRAM clamp was set from the dispatch-time
    reading, taken while the previous sampler still held the card; rebasing it on the clearance figure lets the
    child judge its load against the card the parent just admitted it to. The child reads the figure only on
    an acquire that obtained the permit, so a timed-out or passed-through acquire yields none. Until the child
    takes the permit the parent rewrites the figure from each control step's reading, since the child can reach
    its sample call seconds after the grant, by which time it has moved its own encoder off the card. Children
    cleared in the same step each get the reading less what the children cleared before them take of it, since
    each child sizes its load to the figure it is handed.

    A grant also carries the activation reserve the parent priced the window's non-weight memory at
    (:func:`~horde_worker_regen.process_management.scheduling.admission.pricing.sampling_reserve_mb`), in a
    second shared figure written beside the first. ComfyUI sizes a partial load to leave only its own activation
    estimate, which can be smaller. hordelib raises that estimate to the granted figure, so a partial load leaves
    the room the parent priced. The child reads it on the same acquire and serves it once through
    :meth:`take_cleared_sampling_reserve_mb`. The parent does not refresh it, since the figure belongs to the job.

    The child can also register a wait-entry callback (:meth:`set_wait_entry_callback`), run before a job's first
    acquire of each permit blocks. The parent prices a staged child net of its last reported reservation, so the
    child reports memory there rather than leaving the parent on a reading from before it staged.

    Built with a ``stage`` semaphore, the proxy also exposes :attr:`stage_lease`, the view the child registers
    as hordelib's stage lease. The stage permit, its consumed flag and its shared figure follow the clearance
    permit's rules and are reset by the same :meth:`begin_job`.
    """

    _wait_entry_callback: Callable[[], None] | None = None
    """Child-local hook run before a job's first acquire blocks. A class-level default rather than an
    ``__init__`` field, so a proxy pickled across the spawn boundary never carries it and one built before it
    existed still reads None."""

    def __init__(
        self,
        *,
        clearance: _ClearanceSemaphore,
        done: _ClearanceSemaphore,
        cleared_device_free: _SharedFigure | None = None,
        stage: _ClearanceSemaphore | None = None,
        staged_device_free: _SharedFigure | None = None,
        cleared_sampling_reserve: _SharedFigure | None = None,
        staged_sampling_reserve: _SharedFigure | None = None,
    ) -> None:
        """Wrap the parent-created clearance and done semaphores, and the stage semaphore when there is one.

        Args:
            clearance: A bounded semaphore the parent holds empty (its single permit acquired at creation) and
                releases to grant one child one load-and-sample window.
            done: A semaphore the child releases when a sampling window closes, so the parent (draining it
                non-blockingly) learns the grant was consumed and retired.
            cleared_device_free: A parent-created shared float the parent sets to the admitted device free
                (MB) before each grant, NaN when it had no reading. None carries no figure with any grant.
            stage: A bounded semaphore the parent holds empty and releases to let the child upload its weights
                ahead of its sample window. None gives the proxy no stage lease, and the child loads inside
                its sample window.
            staged_device_free: The stage grant's shared figure, set the way ``cleared_device_free`` is. None
                carries no figure with any stage grant.
            cleared_sampling_reserve: A parent-created shared float the parent sets to the window's priced
                activation reserve (MB) before each grant, NaN when it priced none. None carries no reserve.
            staged_sampling_reserve: The stage grant's reserve, set the way ``cleared_sampling_reserve`` is. None
                carries no reserve with any stage grant.
        """
        self._clearance = clearance
        self._done = done
        self._cleared_device_free = cleared_device_free
        self._cleared_sampling_reserve = cleared_sampling_reserve
        self._grant_consumed = False
        self._granted_device_free_mb: float | None = None
        self._granted_sampling_reserve_mb: float | None = None
        self._stage = stage
        self._staged_device_free = staged_device_free
        self._staged_sampling_reserve = staged_sampling_reserve
        self._stage_consumed = False
        self._stage_granted_device_free_mb: float | None = None
        self._stage_granted_sampling_reserve_mb: float | None = None
        self._stage_lease = ClearanceStageLease(self) if stage is not None else None

    @property
    def stage_lease(self) -> ClearanceStageLease | None:
        """The child-facing stage lease to register with hordelib, or None when the proxy has no stage permit."""
        return self._stage_lease

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        """Wait for the parent's clearance grant, or pass through if this job already consumed one.

        The first ``acquire`` of a job blocks on the clearance permit up to ``timeout``; a later ``acquire``
        for the same job (a multi-sample workflow's subsequent sample) returns immediately so one job never
        waits for two grants. A blocked acquire that times out still marks the job's grant consumed: the job
        proceeds to sample unpriced (hordelib's degraded path), and its remaining samples must not each pay
        the timeout again. :meth:`begin_job` clears the flag for the next job.

        A blocking first acquire probes for a permit before it waits: a grant already waiting is taken at once,
        and otherwise the wait-entry callback runs and the acquire blocks for what is left of ``timeout``, so
        the callback never lengthens the total wait.
        """
        if self._grant_consumed:
            return True
        acquired = self._acquire_clearance(block, timeout)
        # Consume the per-job grant on any completed attempt, granted or timed out: either way the job now
        # samples, and its later samples must pass through rather than block again.
        self._grant_consumed = True
        if acquired:
            self._granted_device_free_mb = self._read_cleared_device_free_mb()
            self._granted_sampling_reserve_mb = _read_shared_figure(self._cleared_sampling_reserve)
        return acquired

    def set_wait_entry_callback(self, callback: Callable[[], None] | None) -> None:
        """Register (or clear) the child-local callback run before a job's first acquire of either permit blocks.

        Set in the child after the proxy crossed the spawn boundary; it is never pickled.
        """
        self._wait_entry_callback = callback

    def _acquire_clearance(self, block: bool, timeout: float | None) -> bool:
        """Take the clearance permit, running the wait-entry callback only when the acquire is about to block."""
        return self._acquire_with_wait_entry(self._clearance, block, timeout)

    def _acquire_with_wait_entry(self, permit: _ClearanceSemaphore, block: bool, timeout: float | None) -> bool:
        """Take ``permit``, running the wait-entry callback only when the acquire is about to block.

        Without a callback the permit is taken in the single call the semaphore was always given; the probe
        exists only to tell a waiting grant from a wait.
        """
        if not block or self._wait_entry_callback is None:
            return bool(permit.acquire(block, timeout))
        started_at = time.monotonic()
        if permit.acquire(False):
            return True
        self._notify_wait_entry()
        remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - started_at))
        return bool(permit.acquire(True, remaining))

    def _notify_wait_entry(self) -> None:
        """Run the wait-entry callback; a failure is logged and never reaches the acquire."""
        callback = self._wait_entry_callback
        if callback is None:
            return
        try:
            callback()
        except Exception as e:  # noqa: BLE001 - a reporting hook must never stop the child sampling
            logger.warning(f"Clearance wait-entry callback failed: {type(e).__name__} {e}")

    def take_cleared_device_free_mb(self) -> float | None:
        """Return the device free VRAM (MB) the parent admitted this child's grant against, once.

        Set by the :meth:`acquire` that obtained the permit and cleared on read, so a later sample of the same
        job (whose own allocations the figure does not account for) never rebases on it again. None when the
        last acquire timed out or passed through, or the parent had no reading.
        """
        figure = self._granted_device_free_mb
        self._granted_device_free_mb = None
        return figure

    def take_cleared_sampling_reserve_mb(self) -> float | None:
        """Return the activation reserve (MB) the parent priced this child's grant with, once.

        Like :meth:`take_cleared_device_free_mb`, it is set by the acquire that obtained the permit and cleared on
        read. None when the last acquire timed out or passed through, or the parent priced no reserve.
        """
        figure = self._granted_sampling_reserve_mb
        self._granted_sampling_reserve_mb = None
        return figure

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

        The shared figure is unset here too. A permit the parent released after the previous job had already
        timed out is still available to this job's acquire, and the figure written with it describes the card
        before the previous job sampled. The reserve figures and the stage permit's flag and figures are reset
        by the same rules.
        """
        self._grant_consumed = False
        self._granted_device_free_mb = None
        self._granted_sampling_reserve_mb = None
        _unset_shared_figure(self._cleared_device_free)
        _unset_shared_figure(self._cleared_sampling_reserve)
        self._stage_consumed = False
        self._stage_granted_device_free_mb = None
        self._stage_granted_sampling_reserve_mb = None
        _unset_shared_figure(self._staged_device_free)
        _unset_shared_figure(self._staged_sampling_reserve)

    def _read_cleared_device_free_mb(self) -> float | None:
        """Child-side: the figure the parent wrote with the grant just acquired, or None when unset."""
        return _read_shared_figure(self._cleared_device_free)

    def _acquire_stage(self, block: bool, timeout: float | None) -> bool:
        """Child-side: take the stage permit once per job; see :meth:`ClearanceStageLease.acquire`.

        A blocking acquire runs the wait-entry callback before it waits, as the clearance acquire does, since
        under two-phase clearance the child's first wait is here and the parent prices the stage against the
        report it sends.
        """
        if self._stage_consumed:
            return True
        if self._stage is None:
            return False
        acquired = self._acquire_with_wait_entry(self._stage, block, timeout)
        self._stage_consumed = True
        if acquired:
            self._stage_granted_device_free_mb = _read_shared_figure(self._staged_device_free)
            self._stage_granted_sampling_reserve_mb = _read_shared_figure(self._staged_sampling_reserve)
        return acquired

    def _take_staged_device_free_mb(self) -> float | None:
        """Child-side: return the stage grant's figure once, then clear it."""
        figure = self._stage_granted_device_free_mb
        self._stage_granted_device_free_mb = None
        return figure

    def _take_staged_sampling_reserve_mb(self) -> float | None:
        """Child-side: return the stage grant's activation reserve once, then clear it."""
        figure = self._stage_granted_sampling_reserve_mb
        self._stage_granted_sampling_reserve_mb = None
        return figure

    def _parent_grant(self, *, device_free_mb: float | None, sampling_reserve_mb: float | None = None) -> None:
        """Parent-side: release one clearance permit to grant this child a load-and-sample window.

        The admitted device free and the priced activation reserve are written before the release, so the
        child's acquire, which returns only after the release, reads this grant's figures.

        A bounded clearance semaphore raises ``ValueError`` if a permit is already available (a prior grant
        the child has not consumed), which the controller treats as already-cleared. The figures are still
        replaced, since they are the newer admission of the same child's waiting job.
        """
        _write_shared_figure(self._cleared_device_free, device_free_mb)
        _write_shared_figure(self._cleared_sampling_reserve, sampling_reserve_mb)
        self._clearance.release()

    def _parent_refresh_grant_figure(self, *, device_free_mb: float) -> bool:
        """Parent-side: rewrite the figure of a grant the child has not consumed, returning whether it did.

        The permit is held across the write. The child's acquire can return only once the permit is back, so it
        reads the new figure, and a permit the child already took leaves nothing to hold, so a consumed grant is
        never rewritten. A child that probes during the hold finds no permit and blocks until the release.
        """
        if self._cleared_device_free is None:
            return False
        if not self._clearance.acquire(False):
            return False
        try:
            self._cleared_device_free.value = device_free_mb
        finally:
            self._clearance.release()
        return True

    def _parent_reclaim_unconsumed_permit(self) -> bool:
        """Parent-side: non-blockingly take back a granted permit the child never consumed; return whether one was."""
        return bool(self._clearance.acquire(False))

    def _parent_grant_stage(self, *, device_free_mb: float | None, sampling_reserve_mb: float | None = None) -> None:
        """Parent-side: release the stage permit so the child may upload its weights ahead of its sample window.

        The figures are written before the release, as :meth:`_parent_grant` writes its own. A bounded stage
        semaphore raises ``ValueError`` when a permit is already available, which the controller treats as
        already staged. Raises ``ValueError`` too when the proxy has no stage semaphore; the controller checks
        :attr:`stage_lease` first.
        """
        if self._stage is None:
            raise ValueError("this proxy carries no stage permit")
        _write_shared_figure(self._staged_device_free, device_free_mb)
        _write_shared_figure(self._staged_sampling_reserve, sampling_reserve_mb)
        self._stage.release()

    def _parent_refresh_stage_figure(self, *, device_free_mb: float) -> bool:
        """Parent-side: rewrite an unconsumed stage grant's figure, as :meth:`_parent_refresh_grant_figure` does."""
        if self._stage is None or self._staged_device_free is None:
            return False
        if not self._stage.acquire(False):
            return False
        try:
            self._staged_device_free.value = device_free_mb
        finally:
            self._stage.release()
        return True

    def _parent_reclaim_unconsumed_stage_permit(self) -> bool:
        """Parent-side: non-blockingly take back a stage permit the child never consumed; return whether one was."""
        if self._stage is None:
            return False
        return bool(self._stage.acquire(False))

    def _parent_drain_done(self) -> bool:
        """Parent-side: non-blockingly take one done signal, returning whether one was present."""
        return bool(self._done.acquire(False))


def _read_shared_figure(shared_figure: _SharedFigure | None) -> float | None:
    """Return a shared figure's value, or None when there is no figure or it is unset (NaN)."""
    if shared_figure is None:
        return None
    figure = float(shared_figure.value)
    if math.isnan(figure):
        return None
    return figure


def _write_shared_figure(shared_figure: _SharedFigure | None, figure: float | None) -> None:
    """Set a shared figure to ``figure``, or unset it (NaN) for None. A proxy without the figure writes nothing."""
    if shared_figure is None:
        return
    shared_figure.value = math.nan if figure is None else figure


def _unset_shared_figure(shared_figure: _SharedFigure | None) -> None:
    """Unset a shared figure (NaN). A proxy without the figure writes nothing."""
    _write_shared_figure(shared_figure, None)


class ClearanceStageLease:
    """The child-side stage lease: block on the parent's stage grant before uploading the weights.

    A view over one :class:`ClearanceLeaseProxy`, which owns the permit and its per-job state, registered with
    hordelib as its stage lease. It satisfies hordelib's ``SamplingLease``, ``ClearanceTruthLease`` and
    ``ClearanceReserveLease`` protocols. The parent reconciles a stage grant by job correlation, so
    :meth:`release` signals nothing.
    """

    def __init__(self, proxy: ClearanceLeaseProxy) -> None:
        """Bind the view to the proxy that owns the stage permit."""
        self._proxy = proxy

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        """Wait for the parent's stage grant, or pass through if this job already took one.

        Follows the clearance permit's per-job rule: the first acquire of a job blocks up to ``timeout`` and
        marks the stage taken whether it was granted or timed out, and a later acquire of the same job returns
        True at once. :meth:`ClearanceLeaseProxy.begin_job` clears the flag for the next job.
        """
        return self._proxy._acquire_stage(block, timeout)

    def release(self) -> None:
        """End the stage window; the parent retires the stage grant by job correlation, so nothing is signalled."""

    def take_cleared_device_free_mb(self) -> float | None:
        """Return the device free VRAM (MB) the parent admitted this child's stage grant against, once.

        None when the last stage acquire timed out or passed through, or the parent had no reading.
        """
        return self._proxy._take_staged_device_free_mb()

    def take_cleared_sampling_reserve_mb(self) -> float | None:
        """Return the activation reserve (MB) the parent priced this child's stage grant with, once.

        None when the last stage acquire timed out or passed through, or the parent priced no reserve.
        """
        return self._proxy._take_staged_sampling_reserve_mb()


class GrantState(enum.Enum):
    """A registered child's clearance-grant state, owned by :class:`ClearanceController`."""

    IDLE = "idle"
    """No grant outstanding: the child holds no window and does not occupy a steady-state slot."""
    STAGED = "staged"
    """The parent released this child's stage permit, so its weights may land on the card; its sample permit is
    not yet released. Holds no steady-state sampling slot."""
    CLEARED = "cleared"
    """The parent released this child's clearance permit; the child has not yet entered its denoise loop."""
    SAMPLING = "sampling"
    """The child consumed its grant and is inside its denoise loop (observed in ``INFERENCE_STARTING``)."""


_SLOT_HOLDING_STATES = frozenset({GrantState.CLEARED, GrantState.SAMPLING})
"""Grant states that occupy a steady-state sampling slot."""

_CLEARABLE_STATES = frozenset({GrantState.IDLE, GrantState.STAGED})
"""Grant states from which a child may be granted its sample permit."""


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
    sampling_reserve_mb: float | None = None
    """The non-weight memory (MB) the staged job's sampling window is priced with
    (:func:`~horde_worker_regen.process_management.scheduling.admission.pricing.sampling_reserve_mb`), carried
    with the child's stage and sample grants. None when the job's figure cannot be priced."""
    weights_mb: float | None = None
    """The diffusion weights (MB) the staged job's grant puts on the card, net of any its slot already holds
    there, or None when they cannot be estimated. With :attr:`sampling_reserve_mb` it sizes what the child takes
    of its grant's free figure, which a child cleared after it in the same step does not get
    (:func:`_grant_charge_mb`)."""


def _grant_charge_mb(device_free_mb: float | None, waiter: ClearanceWaiter | None) -> float:
    """Return what (MB) a cleared child takes of the free figure its grant carries.

    The child loads as much of its weights as the figure allows less the reserve, and keeps the reserve free for
    its activation, so it takes its weights plus its reserve, or the whole figure when that is less. A figure or
    waiter the controller does not have takes nothing.
    """
    if device_free_mb is None or waiter is None:
        return 0.0
    need_mb = (waiter.weights_mb or 0.0) + (waiter.sampling_reserve_mb or 0.0)
    return min(max(0.0, device_free_mb), max(0.0, need_mb))


def _free_after_earlier_grants_mb(device_free_mb: float | None, earlier_charges_mb: float) -> float | None:
    """Return the measured free (MB) less what children cleared earlier in the step take, or None unread."""
    if device_free_mb is None:
        return None
    return max(0.0, device_free_mb - earlier_charges_mb)


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
    device_free_is_device_wide: bool = True
    """Whether ``device_free_mb`` covers every tenant of the card. The tail-overlap handoff spends measured
    headroom on the strength of the figure alone, so it is offered only on a device-wide one. The figure is
    still passed to a cleared child whatever its provenance."""
    incoming_load_seconds: float | None = None
    """Measured RAM-to-VRAM weight-load seconds an incoming child is expected to pay, or None when unmeasured.

    The recent median of observed weight uploads on this card; the tail-overlap window is sized from it so the
    incoming load lands inside the outgoing sampler's tail. ``None`` falls back to
    :data:`_DEFAULT_INCOMING_LOAD_SECONDS`."""


_TAIL_DENIAL_DIAGNOSTIC = "tail_overlap_denial"
_STAGE_REFUSAL_DIAGNOSTIC = "stage_refusal"


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

    headroom_mb = (
        None
        if inputs.device_free_mb is None or not inputs.device_free_is_device_wide
        else inputs.device_free_mb - inputs.vram_reserve_mb
    )
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
    does not is held, and its wait is reported so the scheduler attributes the empty slot. Given a stage
    admission, the tick then stages the head idle waiter's weights while another child samples, and that
    waiter's later sample grant releases only its sample permit.

    The controller owns its grant accounting for the session and self-heals rather than propagating: a
    double-clear is guarded, a replaced child's state is discarded, and a grant the child never consumed has
    its permit taken back when the grant retires. Nothing here blocks the control loop; the caller isolates a
    raised tick fail-inert.
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
        self._tail_denial_diagnostics = DiagnosticThrottle(self._clock)
        # Session tallies for the stage grant: weights staged ahead of a sample window, and head waiters the
        # stage admission refused.
        self._stage_grants_issued = 0
        self._stage_refusals = 0
        self._stage_refusal_diagnostics = DiagnosticThrottle(self._clock)

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

    def holds_sampling_slot(self, process_id: int) -> bool:
        """Whether a child holds a sampling slot (cleared or sampling), False when unknown.

        A staged child holds none, since its sample permit is not granted even with its weights on the card.
        """
        return self.grant_state(process_id) in _SLOT_HOLDING_STATES

    @property
    def held_grant_count(self) -> int:
        """How many registered children currently hold a sampling slot (see :meth:`holds_sampling_slot`)."""
        return sum(1 for process_id in self._grant_state if self.holds_sampling_slot(process_id))

    def step(
        self,
        inputs: ClearanceInputs,
        *,
        admit_fn: Callable[[int], bool],
        stage_admit_fn: Callable[[int], bool] | None = None,
    ) -> ClearanceStepResult:
        """Retire completed windows, reconcile sampling onset, clear the chosen staged waiters, then stage one.

        Each child cleared in the step is handed the measured free less what the children cleared before it take
        of it (:func:`_grant_charge_mb`), so the figures handed out in one step sum to the card's free memory.

        Args:
            inputs: The per-tick snapshot; its ``active_grants`` is derived from this controller's own held
                children (the scheduler reads :meth:`grant_state` to build it).
            admit_fn: The scheduler's full-price admission for a chosen child: returns whether the child's job
                materialisation fits the card now (running eviction as a side effect), so clearance is the true
                VRAM moment. A child that does not fit is held, not cleared.
            stage_admit_fn: The scheduler's admission for staging a child's weights ahead of its sample window.
                None issues no stage grant and keeps the tail-overlap handoff; a callable replaces the handoff
                with the stage pass (see :meth:`_stage_head_waiter`).

        Returns:
            Which children were cleared and which were held, for slot-duty attribution.
        """
        self._drain_done_discard()
        self._reconcile_grants(inputs)
        self._prune_tail_dedup(inputs)
        self._refresh_unconsumed_grant_figures(inputs.device_free_mb)

        # Only children the controller holds IDLE or STAGED are clearable; a cleared-but-not-yet-sampling child
        # is still reported as a primed waiter by the scheduler snapshot, so filter it out here rather than let a
        # slot pick be wasted on it (the per-child clear guard below would skip it anyway).
        job_id_by_pid = {waiter.process_id: waiter.job_id for waiter in inputs.staged_waiters}
        waiter_by_pid = {waiter.process_id: waiter for waiter in inputs.staged_waiters}
        sampling_reserve_by_pid = {waiter.process_id: waiter.sampling_reserve_mb for waiter in inputs.staged_waiters}
        clearable_waiters = tuple(
            waiter
            for waiter in inputs.staged_waiters
            if self._grant_state.get(waiter.process_id, GrantState.IDLE) in _CLEARABLE_STATES
        )
        effective_inputs = dataclasses.replace(inputs, staged_waiters=clearable_waiters)

        plan = decide_clearances(
            effective_inputs,
            slot_cap=self._slot_cap,
            held_grant_count=self.held_grant_count,
            tail_overlap_enabled=self._tail_overlap and stage_admit_fn is None,
            tail_overlap_pad_seconds=self._tail_overlap_pad_seconds,
            tail_overlap_margin_mb=self._tail_overlap_margin_mb,
            already_tail_cleared_job_ids=frozenset(self._tail_cleared_job_ids),
        )
        if plan.tail_denial is not None:
            self._note_tail_overlap_denial(plan.tail_denial)

        cleared: list[int] = []
        held: list[int] = []
        earlier_charges_mb = 0.0
        for process_id in plan.clear_process_ids:
            if self._grant_state.get(process_id, GrantState.IDLE) not in _CLEARABLE_STATES:
                # Already holds a sample grant; do not clear twice.
                continue
            if admit_fn(process_id):
                device_free_mb = _free_after_earlier_grants_mb(inputs.device_free_mb, earlier_charges_mb)
                if process_id in self._proxies:
                    # A clear that finds its permit already available still writes the figures, so the child
                    # sizes its load to this one either way.
                    earlier_charges_mb += _grant_charge_mb(device_free_mb, waiter_by_pid.get(process_id))
                if self._clear(
                    process_id,
                    job_id_by_pid.get(process_id),
                    device_free_mb=device_free_mb,
                    sampling_reserve_mb=sampling_reserve_by_pid.get(process_id),
                ):
                    cleared.append(process_id)
                    if plan.tail_cleared_for_job_id is not None and process_id == plan.clear_process_ids[-1]:
                        self._tail_cleared_job_ids.add(plan.tail_cleared_for_job_id)
                        self._tail_overlap_grants += 1
                        # A fired handoff ends the denial run, so the next denial reason logs on its own edge.
                        self._tail_denial_diagnostics.forget(_TAIL_DENIAL_DIAGNOSTIC)
                        self._log_tail_overlap_clear(inputs, outgoing_job_id=plan.tail_cleared_for_job_id)
            else:
                held.append(process_id)

        if stage_admit_fn is not None:
            self._stage_head_waiter(inputs, stage_admit_fn=stage_admit_fn, job_id_by_pid=job_id_by_pid)

        return ClearanceStepResult(cleared_process_ids=tuple(cleared), held_process_ids=tuple(held))

    def _stage_head_waiter(
        self,
        inputs: ClearanceInputs,
        *,
        stage_admit_fn: Callable[[int], bool],
        job_id_by_pid: Mapping[int, str | None],
    ) -> None:
        """Grant the head idle waiter its stage permit while another child samples, if the stage admission fits.

        At most one child per card is staged ahead: the pass runs only while a grant on the card is sampling and
        no grant is staged or cleared, so the staged weights land beside a running peak and the next sample
        grant goes to the child already holding them. The head is the lowest priority value, the order
        :func:`decide_clearances` uses. A refused head is asked again on the next step.
        """
        states = self._grant_state.values()
        if not any(state is GrantState.SAMPLING for state in states):
            return
        if any(state in (GrantState.STAGED, GrantState.CLEARED) for state in states):
            return
        idle_waiters = [
            waiter
            for waiter in inputs.staged_waiters
            if self._grant_state.get(waiter.process_id, GrantState.IDLE) is GrantState.IDLE
        ]
        if not idle_waiters:
            return
        head = min(idle_waiters, key=lambda waiter: waiter.priority)
        proxy = self._proxies.get(head.process_id)
        if proxy is None or proxy.stage_lease is None:
            return
        if stage_admit_fn(head.process_id):
            self._clear_stage(
                head.process_id,
                job_id_by_pid.get(head.process_id),
                device_free_mb=inputs.device_free_mb,
                sampling_reserve_mb=head.sampling_reserve_mb,
            )
            self._stage_refusal_diagnostics.forget(_STAGE_REFUSAL_DIAGNOSTIC)
            return
        self._note_stage_refusal(head)

    def _note_stage_refusal(self, waiter: ClearanceWaiter) -> None:
        """Tally a refused stage grant and log it once per refused waiter and job, restated periodically."""
        self._stage_refusals += 1
        suppressed = self._stage_refusal_diagnostics.suppressed_count(
            _STAGE_REFUSAL_DIAGNOSTIC,
            (waiter.process_id, waiter.job_id),
        )
        if suppressed is None:
            return
        logger.debug(
            f"Clearance lease on device {self._device_index}: stage grant for process {waiter.process_id} "
            f"refused by admission.{suppressed_suffix(suppressed)}",
        )

    def _clear_stage(
        self,
        process_id: int,
        job_id: str | None,
        *,
        device_free_mb: float | None,
        sampling_reserve_mb: float | None = None,
    ) -> bool:
        """Release one child's stage permit so its weights may land ahead of its sample window. Returns success.

        Records the job the grant is issued for, so a staged grant retires by the same job correlation as a
        cleared one. ``device_free_mb`` and ``sampling_reserve_mb`` carry the same meaning as in :meth:`_clear`.
        """
        proxy = self._proxies.get(process_id)
        if proxy is None or proxy.stage_lease is None:
            return False
        try:
            proxy._parent_grant_stage(device_free_mb=device_free_mb, sampling_reserve_mb=sampling_reserve_mb)
        except ValueError:
            # The stage permit is already available (one the child has not consumed); treat it as staged.
            self._grant_state[process_id] = GrantState.STAGED
            if job_id is not None:
                self._granted_job_id[process_id] = job_id
            return False
        self._grant_state[process_id] = GrantState.STAGED
        if job_id is not None:
            self._granted_job_id[process_id] = job_id
        self._stage_grants_issued += 1
        logger.debug(
            f"Clearance lease on device {self._device_index}: cleared process {process_id} to stage its weights "
            f"ahead of its sample window.",
        )
        return True

    def _clear(
        self,
        process_id: int,
        job_id: str | None,
        *,
        device_free_mb: float | None,
        sampling_reserve_mb: float | None = None,
    ) -> bool:
        """Release one child's clearance permit exactly once, guarding a double clear. Returns success.

        Records the job the grant is issued for so the grant is later retired by job correlation (the child
        moving off that job), never by a stale done permit from a prior job.

        ``device_free_mb`` is the measured device free the admission priced against, before any charge for
        this child's materialisation and less what children cleared before it in the same step take. The child's
        clamp subtracts its own allocator growth from the figure, so a figure already net of the child's charge
        would count that charge twice.

        ``sampling_reserve_mb`` is the non-weight memory the job's window is priced with, written beside the
        device free on both permits so the child's partial load leaves that room. None writes no reserve.

        An idle child with a stage permit is granted it in the same clear, so a single clear lets the child
        stage and sample without a second wait; a staged child already holds its stage permit and is granted
        only the sample permit. A stage permit already available is left as it is.
        """
        proxy = self._proxies.get(process_id)
        if proxy is None:
            return False
        if self._grant_state.get(process_id, GrantState.IDLE) is GrantState.IDLE and proxy.stage_lease is not None:
            try:
                proxy._parent_grant_stage(device_free_mb=device_free_mb, sampling_reserve_mb=sampling_reserve_mb)
            except ValueError:
                # The stage permit is already available (one the child has not consumed); the child takes it.
                logger.debug(
                    f"Clearance lease on device {self._device_index}: process {process_id} already holds an "
                    f"unconsumed stage permit.",
                )
        try:
            proxy._parent_grant(device_free_mb=device_free_mb, sampling_reserve_mb=sampling_reserve_mb)
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

    def _refresh_unconsumed_grant_figures(self, device_free_mb: float | None) -> None:
        """Rewrite every cleared child's grant figure from this step's reading until the child consumes it.

        Between the grant and the child's sample call the child releases its encoder to host RAM and siblings
        unload, so the figure frozen at the grant tick understates the card the child loads into. A step with no
        reading leaves the last figure in place. A staged child's stage figure is refreshed the same way, and a
        cleared child's stage figure too while its stage permit is untaken.
        """
        if device_free_mb is None:
            return
        for process_id, state in self._grant_state.items():
            if state not in (GrantState.STAGED, GrantState.CLEARED):
                continue
            proxy = self._proxies.get(process_id)
            if proxy is None:
                continue
            proxy._parent_refresh_stage_figure(device_free_mb=device_free_mb)
            if state is GrantState.CLEARED:
                proxy._parent_refresh_grant_figure(device_free_mb=device_free_mb)

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
    def stage_grants_issued(self) -> int:
        """How many stage permits this controller has granted ahead of a sample window this session.

        A stage permit granted together with its sample permit in a single clear is not counted.
        """
        return self._stage_grants_issued

    @property
    def stage_refusals(self) -> int:
        """How many steps the stage admission refused the head waiter this session."""
        return self._stage_refusals

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
        """Tally a denied handoff tick and log it on a reason edge or a periodic restatement.

        The gate is evaluated every control-loop tick, so only the reason edge carries information. Emitted at
        debug, since a denied handoff is ordinary operation rather than a fault.
        """
        self._tail_overlap_denials[denial.reason] += 1
        suppressed = self._tail_denial_diagnostics.suppressed_count(_TAIL_DENIAL_DIAGNOSTIC, denial.reason)
        if suppressed is None:
            return
        logger.debug(
            f"Clearance lease on device {self._device_index}: tail-overlap handoff denied "
            f"({denial.describe()}).{suppressed_suffix(suppressed)}",
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
            elif in_sampling and state is GrantState.STAGED:
                # A staged child seen sampling waited out its sample acquire: its weights were priced, its
                # activation was not. Flag the window once and hold the slot like any sampler.
                self._grant_state[process_id] = GrantState.SAMPLING
                if process_id not in self._unpriced_flagged:
                    self._unpriced_flagged.add(process_id)
                    self._unpriced_sampling_windows += 1
                    logger.warning(
                        f"Clearance lease on device {self._device_index}: process {process_id} entered its "
                        f"denoise loop without a recorded grant for its sample window (unpriced sampling window "
                        f"after its stage grant); liveness preserved.",
                    )

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
        """Return a held grant to idle so its steady-state slot reopens, taking back a permit never consumed.

        A grant can retire still cleared with its permit untaken, when the job ran a graph that never called the
        lease or sampled past its acquire timeout before the grant. Left in the semaphore, that permit would let
        the next job on the slot sample unpriced, so the controller takes it back and the semaphore stays matched
        to this ledger. A child already inside its acquire may win the permit and sample on it, which the
        controller then flags as an unpriced window, as before.

        An untaken stage permit is taken back by the same rule, whatever state the grant retires from: a child
        whose hordelib registered no stage lease samples on its sample permit alone and leaves the stage permit
        of its single clear untaken, which the next job's stage acquire would otherwise pass on unpriced.
        """
        state = self._grant_state.get(process_id)
        proxy = self._proxies.get(process_id)
        if proxy is not None:
            if state is GrantState.CLEARED:
                proxy._parent_reclaim_unconsumed_permit()
            proxy._parent_reclaim_unconsumed_stage_permit()
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
    stage_grants_issued: int = 0,
    stage_refusals: int = 0,
) -> str | None:
    """A compact handoff tally for the duty-cycle line, or None when neither handoff ever applied.

    Reports grants against refusals and the share each clause took of those refusals, largest first, so a duty
    figure short of target can be read straight across to the clause holding the handoff shut.
    :attr:`TailOverlapDenialReason.BONUS_SLOT_UNUSED` is kept out of that arithmetic and trails the line in
    brackets: it records a tick where every clause passed and the steady slots simply covered the queue, so
    counting it as a refusal would drown the shares that name a starver on any lightly loaded worker.

    The stage grant does the tail overlap's work where it is on, so its grants and refusals
    (:attr:`ClearanceController.stage_grants_issued`, :attr:`ClearanceController.stage_refusals`) share the
    line, after the tail-overlap figures when both are present.
    """
    parts: list[str] = []
    slot_unused = denials.get(TailOverlapDenialReason.BONUS_SLOT_UNUSED, 0)
    refusals = {
        reason: count
        for reason, count in denials.items()
        if reason is not TailOverlapDenialReason.BONUS_SLOT_UNUSED and count
    }
    denied = sum(refusals.values())
    if granted or denied or slot_unused:
        line = f"tail-overlap: {granted} granted / {denied} denied"
        if denied:
            ranked = sorted(refusals.items(), key=lambda entry: (-entry[1], entry[0].value))[:max_reasons]
            shares = ", ".join(f"{reason.value} {count / denied:.0%}" for reason, count in ranked)
            line = f"{line} ({shares})"
        if slot_unused:
            line = f"{line} [slot-unused {slot_unused}]"
        parts.append(line)
    if stage_grants_issued or stage_refusals:
        parts.append(f"stage: {stage_grants_issued} granted / {stage_refusals} refused")
    if not parts:
        return None
    return "; ".join(parts)
