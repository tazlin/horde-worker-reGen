"""The single VRAM arbiter: one authority that prices every device-memory request against measured truth.

The worker's device VRAM is contended by several independent consumers (model preloads, monolithic
dispatch, the disaggregated encode/sample/decode lanes, post-processing, safety, and cross-job retention).
Historically each priced the card with its own arithmetic and its own free-VRAM read, which under WDDM
demand-paging (where the driver silently spills an over-commit to host RAM and keeps reporting healthy
free VRAM) cannot be reconciled into one coherent admission picture. This module concentrates that
decision into a single object that reasons about one frozen measurement per control-loop cycle.

The arbiter separates four concerns deliberately:

- Measurement arrives from outside as a :class:`MeasuredVramSnapshot`, assembled once per control-loop
  iteration from figures the parent already holds. The arbiter performs no NVML reads and imports no
  torch: it is pure decision state in the torch-free parent.
- Estimation prices a candidate request's marginal device cost. The caller supplies the priced delta
  (or a stage's static spike figure); an unpriceable candidate is charged nothing, matching the
  predictive gate's admit-on-unknown-cost contract.
- Arbitration evaluates the measured-truth admission identity (see
  :mod:`~horde_worker_regen.process_management.resources.admission_identity`) plus the concurrent-sampling
  headroom, and resolves an actuator escalation ladder. A disaggregated encode/decode stage dispatch is the
  one demand priced by neither arithmetic: it targets a process already resident on the card and is never
  withheld, because the concurrent-sampling gate downstream is the pipeline's real admission point.
- Actuation is expressed as :class:`ActuatorCommand` values on the verdict. The arbiter itself executes
  nothing: it describes what would relieve the pressure through a :class:`VramActuator` and leaves the
  doing to that caller.

Admission reasons from device truth, not from a book. A request FITS iff its candidate outstanding cost fits
the frozen device-free reading net of the outstanding reservations the reading does not yet reflect and the
one noise buffer. Because the free reading already contains the shared baseline, every foreign allocation, and
every materialised worker load, there is no separate committed floor, no baseline term, and no foreign-pressure
concept: those quantities are physically inside the reading. The verdict for a request follows from why it
does or does not fit:

- A candidate that fits available room is admitted, even while a reclaim episode is running on the card: the
  episode exists to make room for demands that do NOT fit, and a fitting demand never waits on it.
- A candidate whose weights already occupy VRAM on its target process admits directly (dispatching materialises
  nothing new; its footprint is already physically in the free reading), the whole-card analogue of the
  disaggregated stage dispatch a resident lane never withholds.
- A non-head request that fits is still withheld when admitting it would leave the true head of queue unable to
  fit its own known demand: the head took precedence, so a line-skipper may not consume the room the head
  needs. This is queue priority, not a memory shortfall, so it defers with no reclaim.
- A candidate that does not fit DEFERS with an actuator ladder sized to relieve the deficit, and re-asks once
  the reclaim owner has freed the room (verified at device level). A head starved past a short grace whose
  remaining deficit is exactly its own idle sibling contexts (a bare CUDA context weight eviction cannot
  reclaim, freed only when the process exits) escalates to a verified context teardown: it DEFERS with a
  REDUCE_LIVE_CONTEXTS actuation and re-asks once the room frees.
- A candidate that cannot fit an even fully-cleared card DENIES: no escalation on this card could seat it.
- A card with no device-free reading yet DEFERS with a throttled diagnostic: the primary admission input is
  absent, so the arbiter neither denies nor fabricates a fictional free figure; it waits for the next reading.

There is no overcommit-admit path. A head that never becomes admittable while the device is idle is caught by
the structural-queue-wedge recovery supervisor (the deadlock detector feeding the worker recovery
coordinator), which soft-resets the pools and, failing that, faults the wedged jobs non-retryably so the
horde reissues them elsewhere. Before that, a genuinely-parked head past :data:`_STARVATION_DIAGNOSTIC_SECONDS`
emits a warning naming the arithmetic and increments :attr:`VramArbiter.starvation_diagnostics`, so the
inequality is in the log for the post-mortem.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger
from strenum import StrEnum

from horde_worker_regen.process_management.resources.admission_identity import (
    _ADMISSION_NOISE_BUFFER_MB,
    AdmissionVerdict,
    evaluate_admission,
)
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState

_FIRST_PARTY_TEARDOWN_GRACE_SECONDS = 10.0
"""Age (seconds) a head must be the undispatched head of an idle device before its own idle sibling contexts
are torn down, when those contexts provably hold the entire remaining deficit and nothing else can free it.

This is a short grace, not the diagnostic threshold. When weight reclaim is exhausted, no physically-available
VRAM exists to admit into, and the deficit is exactly the head's own idle sibling contexts (a bare CUDA context
weight eviction cannot reclaim), no alternative remedy can arrive: evicting a model or releasing a cache frees
nothing the contexts hold, and a busy sibling finishing does not surrender its context. Waiting past this point
is pure idle-card loss, so the escalation fires quickly. The grace exists only to ride out transient state
churn and measurement noise (a sibling about to pick up work, a snapshot mid-reconciliation), not to wait for a
remedy that cannot come. Rapid large/small model alternation is damped on the pop side by
``large_model_switch_min_seconds`` and the re-entry cooldown, not by lengthening this timer."""

_STARVATION_DIAGNOSTIC_SECONDS = 60.0
"""Age (seconds) past which a head-of-queue request that keeps deferring with no verified progress emits a
diagnostic warning naming the full admission arithmetic. This is observability, not an admit path: the head
stays queued. The structural-queue-wedge recovery supervisor is the mechanism that actually reroutes a
never-admittable idle-device head (it soft-resets the pools at the wedge horizon and then faults the wedged
jobs non-retryably for horde reissue); this warning is a slower, self-explaining backstop that fires if such a
head is somehow still parked, so the arithmetic is in the log for the post-mortem."""


class VramRequestKind(StrEnum):
    """The kind of device-memory demand a request represents.

    The kind selects which arithmetic prices it: the disaggregated sampling gate reasons about concurrent
    activation headroom, every other kind about the measured-truth admission identity.
    """

    PRELOAD = "preload"
    """A model preload (weights brought toward a target process)."""
    MONOLITHIC_DISPATCH = "monolithic_dispatch"
    """A whole-job dispatch onto an inference process that samples, decodes, and post-processes in one."""
    DISAGG_ENCODE = "disagg_encode"
    """A disaggregated text-encode or VAE-encode stage on the component service or image lane."""
    DISAGG_SAMPLE = "disagg_sample"
    """A disaggregated UNet sampling stage; priced against concurrent-sampling headroom."""
    DISAGG_DECODE = "disagg_decode"
    """A disaggregated VAE-decode stage (plus any bundled post-processing) on the image lane."""
    PP_JOB = "pp_job"
    """A post-processing chain on the dedicated post-processing lane."""
    SAFETY_LOAD = "safety_load"
    """A safety-model load onto the GPU safety context."""


class VramDisposition(StrEnum):
    """The outcome class of a verdict."""

    FITS = "fits"
    """The candidate fits available room, is a no-op onto already-resident weights, or the arbiter has no
    device state yet and relaxed to admit on the predictive path."""
    DEFER = "defer"
    """The candidate does not fit (or is withheld to protect the head, or has no device-free reading yet);
    any actuations are attached to relieve pressure and the request re-asks."""
    DENY = "deny"
    """The demand is structurally impossible even after full escalation (it cannot fit an empty card)."""


class ActuatorCommandKind(StrEnum):
    """A pressure-relief action the arbiter would take, expressed but never executed."""

    RELEASE_CACHE = "release_cache"
    """Ask an idle lane to release its cached allocator reservation back to the device."""
    EVICT_IDLE_MODEL = "evict_idle_model"
    """Evict an idle VRAM-resident model to reclaim its weights."""
    REDUCE_LIVE_CONTEXTS = "reduce_live_contexts"
    """Reduce the live inference context count so a retained per-context reservation returns to the device."""
    PAUSE_VAE_LANE = "pause_vae_lane"
    """Temporarily stop an idle VAE service lane so its CUDA context returns to the device."""
    PAUSE_COMPONENT_LANE = "pause_component_lane"
    """Temporarily stop an idle component service lane so its CUDA context returns to the device."""
    CYCLE_SAFETY_OFF_GPU = "cycle_safety_off_gpu"
    """Cycle the safety model off the GPU to reclaim its context."""


@dataclass(frozen=True)
class ActuatorCommand:
    """One described pressure-relief action, targeted at a card and optionally a specific process."""

    kind: ActuatorCommandKind
    device_index: int | None
    target_process_id: int | None = None


class VramActuator(Protocol):
    """The execution surface a caller supplies to run the pressure-relief commands a verdict describes.

    The arbiter reasons in the torch-free parent and executes nothing; the worker component that owns process
    state implements this Protocol and the adapter drives it once per deferred verdict. Each method maps one
    :class:`ActuatorCommandKind` onto the worker mechanism that already performs it (allocator-cache release,
    idle-model eviction, live-context reduction, safety off-GPU cycling) and reports whether it acted.
    """

    def release_cache(self, process_id: int) -> bool:
        """Ask one idle lane to release its cached allocator reservation back to the device."""
        ...

    def evict_idle_model(self, device_index: int | None, *, for_head_of_queue: bool) -> bool:
        """Evict an idle VRAM-resident model on the card to reclaim its weights."""
        ...

    def reduce_live_contexts(self, device_index: int | None) -> bool:
        """Reduce the live inference-context count so a retained per-context reservation returns to the card."""
        ...

    def pause_vae_lane(self, device_index: int | None) -> bool:
        """Pause an idle VAE service lane so its CUDA context returns to the card."""
        ...

    def pause_component_lane(self, device_index: int | None) -> bool:
        """Pause an idle component service lane so its CUDA context returns to the card."""
        ...

    def restore_vae_lane(self, device_index: int | None) -> bool:
        """Restore a VAE service lane this reclaim path previously paused."""
        ...

    def restore_component_lane(self, device_index: int | None) -> bool:
        """Restore a component service lane this reclaim path previously paused."""
        ...

    def cycle_safety_off_gpu(self, device_index: int | None) -> bool:
        """Cycle the safety model off the GPU to reclaim its context."""
        ...


@dataclass(frozen=True)
class VramRequest:
    """One priced demand on a device's VRAM, evaluated against the frozen cycle snapshot.

    ``candidate_delta_mb`` is the request's marginal device cost net of any weights already resident in the
    target process. None means the caller could not price it: the arbiter then charges nothing, so an
    unpriceable candidate is never denied by the measured identity. ``sampling_peak_mb`` is only consulted
    for :attr:`VramRequestKind.DISAGG_SAMPLE`, where the concurrent-sampling headroom sums whole-process
    peaks rather than marginal deltas.
    """

    kind: VramRequestKind
    job_label: str
    baseline: str | None
    device_index: int | None
    target_process_id: int | None = None
    candidate_delta_mb: float | None = None
    candidate_already_resident: bool = False
    """True when the candidate's weights are already materialised in VRAM on the target process, so admitting
    this request adds no new device footprint. A dispatch or preload onto an already-resident idle model moves
    nothing: its weights are already physically inside the device-free reading and its next activation is the
    monolithic status quo the card has already demonstrated it holds. Such a request is admitted directly, the
    same way a disaggregated stage dispatch to a resident lane is never withheld, because a resident model's own
    footprint can legitimately push the device-free reading below the candidate's activation cost, which would
    otherwise deny a dispatch that needs no memory. Set only when the target genuinely holds the weights in
    VRAM; a RAM-staged load whose weights materialise on dispatch leaves it False so the identity prices the
    materialisation."""
    own_planned_unmaterialized_mb: float = 0.0
    """The portion (MB) of the device's outstanding reservations attributable to this request's own target,
    netted out of the identity so a re-ask can never block on itself. A preload that was admitted, recorded a
    reservation, and then had its target reclaimed or its process die before the load materialised leaves that
    reservation outstanding until the overlay reconciles; if the head re-asks the same load while the reservation
    lingers, the unnetted identity would subtract the load's own reservation from its own available room and
    defer forever on its own footprint. Subtracting the target's own outstanding reservation makes the request's
    load count at most once, while every other unit's reservation stays fully charged."""
    is_head_of_queue: bool = False
    head_outstanding_mb: float | None = None
    """For a non-head request, the true head of queue's priced outstanding demand (MB) on this device, or None
    when unknown or when this request is itself the head. Head protection: a non-head request that fits is still
    withheld when admitting it would leave less than this room, because the head took precedence and the
    line-skipper may not consume the space the head needs. None skips the check (the head's demand is unknown at
    this seam), degrading to admitting the non-head."""
    first_of_kind: bool = False
    starved_seconds: float = 0.0
    """Seconds this head has been the undispatched head of an idle device, used only to time the
    starvation diagnostic and the first-party context teardown grace; zero when a live job holds the card (the
    head is then queued behind live work, not starved)."""
    sampling_peak_mb: float | None = None
    active_sampling_peaks_total_mb: float | None = None
    """The live sum (MB) of the in-flight disaggregated sampling peaks at the moment of this request, for
    :attr:`VramRequestKind.DISAGG_SAMPLE`. None defers to the cycle-frozen
    :attr:`DeviceVramState.active_sampling_peaks_total_mb`; the disaggregation orchestrator supplies the live
    figure so a peak booked earlier in the same tick is counted before the cycle snapshot is next refrozen."""
    has_reclaimable_idle_model: bool = False
    """True when an idle resident model on the card could be evicted to reclaim its weights for this request."""
    can_reduce_live_contexts: bool = False
    """True when reducing the live inference-context count is a warranted remedy for this head's over-commit."""
    idle_contexts_teardownable: bool = False
    """True when idle sibling inference contexts (a bare CUDA context, neither the head's own target slot nor a
    busy process) exist that a teardown could reclaim for this head. Distinct from ``can_reduce_live_contexts``:
    that flag is the ordinary activation-peak warrant, which a bare-context over-commit (idle siblings holding
    no evictable model or cache) does not trip, so a head whose remaining deficit is exactly those contexts
    would defer forever. This lets a head starved past the teardown grace escalate to a verified context
    teardown regardless of that warrant. Both a preload/whole-card head and a monolithic-dispatch head set it
    when they are the true head of queue; an ordinary (un-starved) dispatch and a line-skip never reach the
    escalation."""
    allow_idle_service_lane_reclaim: bool = False
    """Whether a re-checked PP drain head may borrow one idle disaggregation service-lane context.

    False on the first admission rejection, while existing cache/model reclaim is still being attempted. The
    PP orchestrator enables it only on a fresh measured re-check that remains non-fitting, so a reversible lane
    pause is an escalation after the softer actions rather than a competing first response."""


@dataclass(frozen=True)
class DeviceVramState:
    """The frozen per-device measurement the arbiter prices a cycle's requests against.

    The admission identity reads the truthful device-free reading, the outstanding reservations, the device
    total (to size the noise buffer), and the noise buffer itself. The concurrent-sampling headroom reads the
    device total net of baseline, the fixed per-process overhead, the marginal per-context cost, the live
    context counts, the operator reserve, the lane decode spike, and the in-flight sampling peaks. The
    committed-ledger fields (``committed_vram_mb``, ``committed_is_stale``) and the governor fields
    (``governor_state``, ``reclaim_unresolved``) are retained for the sampling-headroom path, diagnostics, and
    telemetry consumers; the admission path does not read them.
    """

    total_vram_mb: float | None
    """Raw device total VRAM (MB), or None at cold start; used to size the noise buffer and the sampling
    headroom, and to judge structural impossibility."""
    baseline_mb: float
    """The reconciler's measured shared-device baseline (MB); read by the sampling-headroom path only."""
    committed_vram_mb: float
    """The worker's committed-VRAM ledger sum (MB); retained for diagnostics and telemetry, not admission."""
    planned_unmaterialized_mb: float
    """Outstanding reservations (MB): work admitted (preload staged, dispatch admitted) whose allocation the
    device-free reading does not yet reflect, decayed per target as each materialises. Summed across every
    admission flow; the requester's own outstanding reservation is netted out per request via
    :attr:`VramRequest.own_planned_unmaterialized_mb`."""
    committed_is_stale: bool
    """True when a committed-ledger contributor's report has aged past the staleness bound; retained for
    diagnostics and telemetry, not admission."""
    preload_planned_unmaterialized_mb: float = 0.0
    """The preload-flow share (MB) of :attr:`planned_unmaterialized_mb`: charges for admitted loads still
    staged in system RAM, whose VRAM materialisation happens only when their dispatch is later admitted
    against fresh measured truth. A drain-side request (post-processing of an already-sampled job) is priced
    net of this share: the staged load cannot claim the card before that drain completes (its dispatch gate
    re-prices it, and the drain's completion is what frees the room it waits on), so charging it against the
    drain is a circular wait, not protection. Dispatch-flow reservations (in-flight sampling about to spike)
    stay fully charged for every requester."""
    noise_buffer_mb: float = _ADMISSION_NOISE_BUFFER_MB
    """The admission noise margin (MB) subtracted from the device-free reading, derived per device at snapshot
    assembly from the total via :func:`admission_noise_buffer_mb` so it scales with card capacity; direct
    construction defaults to the floor."""
    per_process_reserved_mb: Mapping[int, float] = field(default_factory=dict)
    """Live GPU processes' measured allocator reservation (MB) keyed by process id; carried for diagnostics."""
    idle_process_ids: frozenset[int] = frozenset()
    """Process ids whose lane is idle (a valid RELEASE_CACHE target)."""
    busy_process_ids: frozenset[int] = frozenset()
    """Process ids actively working (never a RELEASE_CACHE target); kept for the boundary invariant."""
    num_loaded_inference_processes: int = 0
    """Inference processes holding a model on this card (drives the extra-context count)."""
    safety_context_count: int = 0
    """Number of on-GPU safety contexts on this card."""
    safety_reclaim_allowed: bool = False
    """Whether policy permits moving this card's safety context off-GPU to relieve admission pressure."""
    post_process_context_count: int = 0
    """Number of on-GPU post-processing contexts on this card."""
    vae_lane_context_count: int = 0
    """Number of on-GPU VAE/image-lane contexts on this card."""
    vae_lane_reclaim_allowed: bool = False
    """Whether a VAE service lane on this card is live, idle, and policy-owned by no existing pause."""
    component_lane_context_count: int = 0
    """Number of on-GPU component/text-encode contexts on this card."""
    component_lane_reclaim_allowed: bool = False
    """Whether a component service lane on this card is live, idle, and policy-owned by no existing pause."""
    per_process_overhead_mb: float = 0.0
    """Fixed device overhead (MB) of the first/sole CUDA context."""
    marginal_mb: float = 0.0
    """Marginal device cost (MB) of each additional resident context."""
    vram_reserve_mb: float = 0.0
    """The operator's configured per-step sampling activation margin (MB)."""
    vae_lane_decode_spike_mb: float = 0.0
    """The image lane's concurrent tiled-decode spike (MB) reserved out of sampling headroom."""
    active_sampling_peaks_total_mb: float = 0.0
    """Sum of the in-flight disaggregated sampling peaks (MB) already admitted on this card."""
    governor_state: GovernorState | None = None
    """The device-free governor's committed state for this card, or None when it has not sampled yet; carried
    for diagnostics and telemetry, not admission."""
    device_free_mb: float | None = None
    """The truthful NVML device-level free VRAM (MB) for this card, or None before the first read. The primary
    admission input: a request FITS iff its candidate fits ``device_free_mb - outstanding_reservations -
    noise``. None yields a throttled-diagnostic DEFER, never a denial or a fabricated figure."""
    reclaim_unresolved: bool = False
    """True when the governor's verified reclaim ladder exhausted its rungs while still SATURATED; carried for
    diagnostics and telemetry, not admission."""

    def sampling_headroom_mb(self) -> float | None:
        """Reproduce the concurrent-sampling headroom (MB), or None when the total is unknown.

        Byte-for-byte with the live scheduler's ``sampling_headroom_mb``: the device total net of the
        measured baseline, minus the fixed first-context overhead, minus the marginal cost of every extra
        resident context, minus the operator reserve, minus the lane decode spike. None (cold start) admits
        rather than wedging on missing telemetry.
        """
        if self.total_vram_mb is None:
            return None
        sampling_total_mb = self.total_vram_mb - self.baseline_mb
        extra_contexts = (
            max(0, self.num_loaded_inference_processes - 1)
            + self.safety_context_count
            + self.post_process_context_count
            + self.vae_lane_context_count
        )
        headroom_mb = (
            sampling_total_mb
            - self.per_process_overhead_mb
            - self.marginal_mb * extra_contexts
            - self.vram_reserve_mb
            - self.vae_lane_decode_spike_mb
        )
        return max(0.0, headroom_mb)


@dataclass(frozen=True)
class MeasuredVramSnapshot:
    """The whole worker's frozen device picture for one control-loop cycle, keyed by device index."""

    devices: Mapping[int, DeviceVramState]

    def device(self, device_index: int | None) -> DeviceVramState | None:
        """Return the per-device state for a card, mapping the single-GPU/worker-wide key (None) to card 0."""
        return self.devices.get(device_index if device_index is not None else 0)


@dataclass(frozen=True)
class VramVerdict:
    """The arbiter's outcome for one request: disposition, reason, and described actuations.

    ``measured`` is always attached so a log line renders the full admission identity the disposition was
    reasoned from.
    """

    disposition: VramDisposition
    request_kind: VramRequestKind
    device_index: int | None
    reason: str
    measured: AdmissionVerdict
    required_actuations: tuple[ActuatorCommand, ...] = ()

    @property
    def admits(self) -> bool:
        """Whether this verdict would let the request proceed (FITS)."""
        return self.disposition is VramDisposition.FITS


def _relaxed_verdict(request: VramRequest, device_index: int | None) -> VramVerdict:
    """Build the degraded FITS verdict for a card with no cycle snapshot yet (arbiter not begun this tick).

    Distinct from a snapshot that exists but lacks a device-free reading (which DEFERS): here there is no
    frozen measurement at all, so the caller falls back to its predictive gate exactly as before the arbiter
    was wired.
    """
    measured = evaluate_admission(
        candidate_outstanding_mb=request.candidate_delta_mb if request.candidate_delta_mb is not None else 0.0,
        device_free_mb=None,
        outstanding_reservations_mb=0.0,
        total_vram_mb=None,
    )
    return VramVerdict(
        disposition=VramDisposition.FITS,
        request_kind=request.kind,
        device_index=device_index,
        reason="no device state this cycle (cold start); admitted on the predictive path",
        measured=measured,
    )


class VramArbiter:
    """The single VRAM authority: freezes a measurement per cycle and prices each request against it.

    Used from the parent's single-threaded control loop only. :meth:`begin_cycle` installs the cycle's
    frozen snapshot; :meth:`evaluate` reads it to price a request and never mutates the measurement. The one
    side effect is the starvation and missing-reading diagnostics: a head deferred past the diagnostic horizon
    with no verified progress, or a card asked to admit with no device-free reading, emits a throttled warning
    (once per cycle per device) and advances a counter.
    """

    def __init__(self) -> None:
        """Initialise with no cycle snapshot and zeroed observability counters."""
        self._cycle: MeasuredVramSnapshot | None = None
        self._cycle_seq = 0
        self._starvation_diag_cycle: dict[int, int] = {}
        self._device_free_missing_diag_cycle: dict[int, int] = {}
        self.admission_foreign_pressure_defers = 0
        self.first_party_context_defers = 0
        self.starvation_diagnostics = 0
        self.starvation_context_teardowns = 0
        self.device_free_missing_defers = 0

    def begin_cycle(self, snapshot: MeasuredVramSnapshot) -> None:
        """Freeze the measurement this cycle's requests are priced against."""
        self._cycle = snapshot
        self._cycle_seq += 1

    @property
    def has_cycle(self) -> bool:
        """Whether a cycle snapshot has been installed."""
        return self._cycle is not None

    def evaluate(self, request: VramRequest) -> VramVerdict:
        """Price one request against the frozen cycle snapshot and return its verdict."""
        if self._cycle is None:
            return _relaxed_verdict(request, request.device_index)
        state = self._cycle.device(request.device_index)
        if state is None:
            return _relaxed_verdict(request, request.device_index)

        if request.kind == VramRequestKind.DISAGG_SAMPLE:
            return self._evaluate_sampling(request, state)
        if request.kind in (VramRequestKind.DISAGG_ENCODE, VramRequestKind.DISAGG_DECODE):
            # Stage dispatches to already-resident lane processes are never withheld. The concurrent-sampling
            # gate downstream is the pipeline's real admission point (an encode only leads to sampling if that
            # gate admits the job), so gating the stages adds no admission control; it only serialises the
            # stage overlap the pipeline exists for. Decode in particular DRAINS the pipeline: completing it
            # releases the job's sampler hold, latents, and submit path, which is how memory pressure ends;
            # the image lane's tiled decode and its allocation self-heal bound the transient spike.
            measured = self._measured(request, state)
            return VramVerdict(
                disposition=VramDisposition.FITS,
                request_kind=request.kind,
                device_index=request.device_index,
                reason="stage dispatch to a resident lane; the sampling gate is the pipeline's admission point",
                measured=measured,
            )
        return self._evaluate_admission(request, state)

    def _measured(self, request: VramRequest, state: DeviceVramState) -> AdmissionVerdict:
        candidate_delta_mb = request.candidate_delta_mb if request.candidate_delta_mb is not None else 0.0
        # Net the request's own outstanding reservation out of the overlay before pricing it. That reservation
        # is this same load admitted on an earlier cycle and not yet materialised; the candidate delta already
        # represents it, so subtracting it from the request's own available room would count the load twice and
        # let a re-ask defer forever on its own footprint. Only the target's own reservation is removed; every
        # other unit's stays.
        overlay_mb = state.planned_unmaterialized_mb
        if request.kind == VramRequestKind.PP_JOB:
            # Post-processing drains the pipeline: completing it is what releases the finished job's holds and
            # frees the room a staged head waits on. A preload-flow charge is a load still in system RAM whose
            # VRAM claim only happens when its dispatch is later re-priced against fresh measured truth, so
            # charging it here inverts the dependency and deadlocks the drain behind bookkeeping (the head
            # cannot materialise until the drain completes, the drain defers on the head's charge). Price the
            # drain against physical truth plus the dispatch-flow reservations only (in-flight sampling really
            # is about to spike); the same reasoning already admits the disaggregated decode unconditionally.
            overlay_mb = max(0.0, overlay_mb - state.preload_planned_unmaterialized_mb)
        net_reservations_mb = max(
            0.0,
            overlay_mb - max(0.0, request.own_planned_unmaterialized_mb),
        )
        return evaluate_admission(
            candidate_outstanding_mb=candidate_delta_mb,
            device_free_mb=state.device_free_mb,
            outstanding_reservations_mb=net_reservations_mb,
            total_vram_mb=state.total_vram_mb,
            noise_buffer_mb=state.noise_buffer_mb,
        )

    def _evaluate_admission(self, request: VramRequest, state: DeviceVramState) -> VramVerdict:
        measured = self._measured(request, state)
        if request.candidate_already_resident:
            # The candidate's weights already occupy VRAM on the target process: dispatching (or preloading) onto
            # it materialises nothing. Its footprint is already physically inside the device-free reading and its
            # next activation is the monolithic status quo the card has already served, so there is nothing to
            # admit into. The identity cannot express this no-op, because the resident model's own footprint can
            # push the device-free reading below the candidate's activation cost; pricing it there would withhold
            # a dispatch to an idle, already-loaded model and wedge the head behind a slot that is already open.
            # This is the whole-card analogue of the disaggregated stage dispatch that a resident lane never
            # withholds.
            return VramVerdict(
                disposition=VramDisposition.FITS,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=f"candidate already resident on the target process; no new VRAM. {measured.reason()}",
                measured=measured,
            )

        if not measured.available_known:
            # The primary admission input (the device-free reading) is absent for this card. Defer rather than
            # deny or fabricate a figure: the next cycle's reading decides. Throttled diagnostic so a persistent
            # gap is visible without flooding the log.
            self._note_device_free_missing(request)
            return VramVerdict(
                disposition=VramDisposition.DEFER,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=measured.reason(),
                measured=measured,
            )

        if measured.fits:
            head_protection_verdict = self._head_protection_defer(request, measured)
            if head_protection_verdict is not None:
                return head_protection_verdict
            return VramVerdict(
                disposition=VramDisposition.FITS,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=measured.reason(),
                measured=measured,
            )

        # The candidate does not fit available room. If it cannot fit even an empty card, no escalation on this
        # card could ever seat it, so it DENIES. Otherwise it DEFERS: a starved head whose deficit is its own
        # idle sibling contexts escalates to a verified teardown, every other non-fitting demand rides the
        # per-cycle reclaim ladder and re-asks once the room frees.
        if self._structurally_impossible(request, state):
            return VramVerdict(
                disposition=VramDisposition.DENY,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=f"candidate cannot fit an empty card; measured: {measured.reason()}",
                measured=measured,
            )

        teardown_actuations = self._starvation_context_teardown(request)
        if teardown_actuations is not None:
            self.starvation_context_teardowns += 1
            return VramVerdict(
                disposition=VramDisposition.DEFER,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=(
                    f"head starved {request.starved_seconds:.0f}s (past the "
                    f"{_FIRST_PARTY_TEARDOWN_GRACE_SECONDS:.0f}s teardown grace) with weight reclaim exhausted and "
                    f"idle sibling contexts holding the deficit; tearing idle contexts down. measured: "
                    f"{measured.reason()}"
                ),
                measured=measured,
                required_actuations=teardown_actuations,
            )

        if self._has_first_party_context_reclaim(request):
            # The deficit is the head's own idle sibling contexts, still within the short teardown grace. The
            # head simply waits out the grace for the verified teardown above rather than routing weight reclaim
            # that frees nothing a bare context holds.
            self.first_party_context_defers += 1

        actuations = self._escalation_ladder(state, request)
        self._note_starvation_diagnostic(request, state, measured)
        return VramVerdict(
            disposition=VramDisposition.DEFER,
            request_kind=request.kind,
            device_index=request.device_index,
            reason=measured.reason(),
            measured=measured,
            required_actuations=actuations,
        )

    def _head_protection_defer(
        self,
        request: VramRequest,
        measured: AdmissionVerdict,
    ) -> VramVerdict | None:
        """Withhold a fitting non-head request when admitting it would starve the true head of queue.

        The candidate fits available room, but a non-head request (a line-skip job jumping the queue) may not
        consume room the head needs: the head took precedence, and after this admission the head's own priced
        demand must still fit. When it would not, the non-head DEFERS with no reclaim, holding the physical room
        for the head. Returns None (admit) for the head itself, when the head's demand is unknown at this seam,
        or when the head still fits after this admission.
        """
        if request.is_head_of_queue or request.head_outstanding_mb is None:
            return None
        available_mb = measured.available_mb
        if available_mb is None:
            return None
        remaining_after_admit_mb = available_mb - measured.candidate_outstanding_mb
        if request.head_outstanding_mb <= remaining_after_admit_mb:
            return None
        self.admission_foreign_pressure_defers += 1
        return VramVerdict(
            disposition=VramDisposition.DEFER,
            request_kind=request.kind,
            device_index=request.device_index,
            reason=(
                f"candidate {measured.candidate_outstanding_mb:.0f} MB fits available "
                f"{available_mb:.0f} MB, but admitting it would leave {remaining_after_admit_mb:.0f} MB, "
                f"below the head of queue's {request.head_outstanding_mb:.0f} MB demand; held for the head. "
                f"measured: {measured.reason()}"
            ),
            measured=measured,
        )

    def _evaluate_sampling(self, request: VramRequest, state: DeviceVramState) -> VramVerdict:
        measured = self._measured(request, state)
        # The empty-ledger first-of-kind admit mirrors the live gate: a single over-peak sampling is the
        # monolithic status quo (the driver streams: slow but correct), so denying it would wedge a small card.
        if request.first_of_kind:
            return VramVerdict(
                disposition=VramDisposition.FITS,
                request_kind=request.kind,
                device_index=request.device_index,
                reason="first concurrent sampling of its kind: empty ledger admits",
                measured=measured,
            )
        headroom_mb = state.sampling_headroom_mb()
        peak_mb = request.sampling_peak_mb
        if headroom_mb is None or peak_mb is None:
            return VramVerdict(
                disposition=VramDisposition.FITS,
                request_kind=request.kind,
                device_index=request.device_index,
                reason="sampling headroom or peak unsizable: admitted rather than wedging on telemetry",
                measured=measured,
            )
        active_total_mb = (
            request.active_sampling_peaks_total_mb
            if request.active_sampling_peaks_total_mb is not None
            else state.active_sampling_peaks_total_mb
        )
        demand_mb = active_total_mb + peak_mb
        if demand_mb <= headroom_mb:
            return VramVerdict(
                disposition=VramDisposition.FITS,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=(
                    f"sampling: active {active_total_mb:.0f} + peak {peak_mb:.0f} = "
                    f"{demand_mb:.0f} MB within headroom {headroom_mb:.0f} MB"
                ),
                measured=measured,
            )
        return VramVerdict(
            disposition=VramDisposition.DEFER,
            request_kind=request.kind,
            device_index=request.device_index,
            reason=(
                f"sampling: active {state.active_sampling_peaks_total_mb:.0f} + peak {peak_mb:.0f} = "
                f"{demand_mb:.0f} MB exceeds headroom {headroom_mb:.0f} MB"
            ),
            measured=measured,
            required_actuations=self._escalation_ladder(state, request),
        )

    @staticmethod
    def _escalation_ladder(state: DeviceVramState, request: VramRequest) -> tuple[ActuatorCommand, ...]:
        """Describe the pressure-relief ladder for a non-fitting demand, in escalation order.

        Only commands that could still free device memory are emitted. RELEASE_CACHE targets idle lanes (a
        busy lane is never a target, and the request's own target slot is never asked to release the cache it
        is about to load into); EVICT_IDLE_MODEL is emitted only when an idle resident model exists to evict;
        REDUCE_LIVE_CONTEXTS only when a warranted context reduction is available. An empty ladder therefore
        means the arbiter's own per-cycle reclamation is structurally exhausted: nothing the caller could run
        this cycle would free memory. The commands are described for the caller to execute; the arbiter runs
        none of them.
        """
        commands: list[ActuatorCommand] = []
        for pid in sorted(state.idle_process_ids):
            if pid in state.busy_process_ids or pid == request.target_process_id:
                continue
            commands.append(
                ActuatorCommand(
                    kind=ActuatorCommandKind.RELEASE_CACHE,
                    device_index=None,
                    target_process_id=pid,
                ),
            )
        if request.has_reclaimable_idle_model:
            commands.append(ActuatorCommand(kind=ActuatorCommandKind.EVICT_IDLE_MODEL, device_index=None))
        if request.can_reduce_live_contexts:
            commands.append(ActuatorCommand(kind=ActuatorCommandKind.REDUCE_LIVE_CONTEXTS, device_index=None))
        service_lane_added = False
        if request.kind is VramRequestKind.PP_JOB and request.allow_idle_service_lane_reclaim and not commands:
            # PP cannot pause its own lane. Offer one idle disaggregation service context in the verified
            # ladder's existing VAE -> component order. The caller caps this at one successful loan for the
            # drain episode and restores only a pause whose actuator reported that it actually acquired.
            if state.vae_lane_context_count > 0 and state.vae_lane_reclaim_allowed:
                commands.append(ActuatorCommand(kind=ActuatorCommandKind.PAUSE_VAE_LANE, device_index=None))
                service_lane_added = True
            elif state.component_lane_context_count > 0 and state.component_lane_reclaim_allowed:
                commands.append(ActuatorCommand(kind=ActuatorCommandKind.PAUSE_COMPONENT_LANE, device_index=None))
                service_lane_added = True
        if (
            request.kind is VramRequestKind.PP_JOB
            and not service_lane_added
            and state.safety_context_count > 0
            and state.safety_reclaim_allowed
        ):
            commands.append(ActuatorCommand(kind=ActuatorCommandKind.CYCLE_SAFETY_OFF_GPU, device_index=None))
        return tuple(commands)

    @staticmethod
    def _has_first_party_context_reclaim(request: VramRequest) -> bool:
        """Whether a head's remaining deficit could be relieved by tearing its own idle sibling contexts down.

        A bare CUDA context returns its VRAM only when the process exits, so the per-cycle ladder's
        weight-eviction rungs (release cache, evict idle model) cannot reclaim it. A head whose remaining
        over-commit is exactly its own idle sibling contexts holds a first-party reclaim the ordinary ladder
        cannot describe; the ordinary activation-peak warrant behind ``can_reduce_live_contexts`` does not trip
        on a bare-context over-commit, so without this the head would defer indefinitely on space it could
        itself free. This decides only reachability of that reclaim (whether such contexts exist and the caller
        is a head whose seam owns the teardown), not the timing: the short teardown grace gates when it fires.

        Both the preload/whole-card path and a monolithic-dispatch head reach it. Ordinary staged dispatch still
        never collapses the pool (``can_reduce_live_contexts`` stays False for a dispatch, so the ordinary
        activation-peak warrant never tears a context down); the starved-head escalation is the one exception,
        the same as the preload seam.
        """
        if request.kind not in (VramRequestKind.PRELOAD, VramRequestKind.MONOLITHIC_DISPATCH):
            return False
        return request.is_head_of_queue and request.idle_contexts_teardownable

    def _starvation_context_teardown(self, request: VramRequest) -> tuple[ActuatorCommand, ...] | None:
        """Return a context-teardown actuation once a starved first-party head clears the short teardown grace.

        The reclaim itself (idle sibling contexts the head could free) is decided by
        :meth:`_has_first_party_context_reclaim`; this adds the timing gate. Because the caller only reaches this
        with the candidate not fitting available room and the deficit held by the head's own idle sibling
        contexts, no other remedy can arrive: the escalation is evidence-based, not a long wait. Once such a head
        has been the undispatched head of an idle device past :data:`_FIRST_PARTY_TEARDOWN_GRACE_SECONDS` (a
        short grace that rides out state churn, not the 60s diagnostic threshold), it escalates to a single
        REDUCE_LIVE_CONTEXTS command: the caller reduces the live inference-context count, tearing the idle
        contexts down while protecting the head's own target slot and every busy process, and the freed room is
        verified at device level before the head is admitted (the escalation never force-admits). Returns None
        while the head is younger than the grace or has no such reclaim.
        """
        if not self._has_first_party_context_reclaim(request):
            return None
        if request.starved_seconds < _FIRST_PARTY_TEARDOWN_GRACE_SECONDS:
            return None
        return (ActuatorCommand(kind=ActuatorCommandKind.REDUCE_LIVE_CONTEXTS, device_index=None),)

    def _note_device_free_missing(self, request: VramRequest) -> None:
        """Emit the missing-device-free diagnostic, throttled per cycle per card, and advance its counter.

        A card asked to admit with no NVML device-free reading yet cannot be priced by the primary identity, so
        the request DEFERS. This records that gap so a persistent absence (a card the parent never reads) is
        visible in the log and the counter without flooding either.
        """
        device_key = request.device_index if request.device_index is not None else 0
        self.device_free_missing_defers += 1
        if self._device_free_missing_diag_cycle.get(device_key) == self._cycle_seq:
            return
        self._device_free_missing_diag_cycle[device_key] = self._cycle_seq
        logger.warning(
            f"VRAM admission for {request.job_label} deferred: no device-free reading for card {device_key} "
            "this cycle. Admission neither denies nor fabricates a free figure on a missing measurement; it "
            "waits for the next NVML reading.",
        )

    def _note_starvation_diagnostic(
        self,
        request: VramRequest,
        state: DeviceVramState,
        measured: AdmissionVerdict,
    ) -> None:
        """Emit the starvation diagnostic for a long-deferred idle-device head, throttled per cycle per card.

        Fires only for a head-of-queue request that has been the undispatched head of an idle device past
        :data:`_STARVATION_DIAGNOSTIC_SECONDS`. It is observability, not an admit: the caller still defers. The
        warning names the full measured arithmetic so a post-mortem of a parked head reads the exact inequality;
        the structural-queue-wedge recovery supervisor is what actually reroutes the job.
        """
        del state
        if not request.is_head_of_queue or request.starved_seconds < _STARVATION_DIAGNOSTIC_SECONDS:
            return
        device_key = request.device_index if request.device_index is not None else 0
        if self._starvation_diag_cycle.get(device_key) == self._cycle_seq:
            return
        self._starvation_diag_cycle[device_key] = self._cycle_seq
        self.starvation_diagnostics += 1
        logger.warning(
            f"Head-of-queue {request.job_label} deferred {request.starved_seconds:.0f}s "
            f">= {_STARVATION_DIAGNOSTIC_SECONDS:.0f}s with no verified progress; it stays queued for the "
            f"structural-wedge recovery supervisor to reroute. Measured: {measured.reason()}.",
        )

    @staticmethod
    def _structurally_impossible(request: VramRequest, state: DeviceVramState) -> bool:
        """Whether the candidate alone exceeds an empty card's room, so no escalation could seat it.

        An empty card offers ``total - noise_buffer`` (every reservation released, every allocation freed). A
        candidate larger than that can never fit on this card however much is reclaimed, so it DENIES rather
        than deferring forever. Unknown total cannot prove impossibility, so it returns False.
        """
        if state.total_vram_mb is None:
            return False
        candidate_delta_mb = request.candidate_delta_mb if request.candidate_delta_mb is not None else 0.0
        return candidate_delta_mb > state.total_vram_mb - state.noise_buffer_mb
