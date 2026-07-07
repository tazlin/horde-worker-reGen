"""The single VRAM arbiter: one authority that prices every device-memory request against measured state.

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
- Arbitration evaluates the ledger-driven admission identity (see
  :mod:`~horde_worker_regen.process_management.resources.admission_identity`) plus the concurrent-sampling
  headroom, and resolves an actuator escalation ladder. A disaggregated encode/decode stage dispatch is the
  one demand priced by neither arithmetic: it targets a process already resident on the card and is never
  withheld, because the concurrent-sampling gate downstream is the pipeline's real admission point.
- Actuation is expressed as :class:`ActuatorCommand` values on the verdict. The arbiter itself executes
  nothing: it describes what would relieve the pressure through a :class:`VramActuator` and leaves the
  doing to that caller.

Admission never admits work into a proven over-commit, because doing so triggers WDDM demotion and a
card-wide throughput cliff that is strictly worse than waiting. The verdict for a request that does not fit
measured capacity follows from why it does not fit:

- While reclaim can still free space, the demand DEFERS. Reclaim can still free space when the arbiter's own
  per-cycle escalation ladder still emits rungs, or when the governor's verified reclaim ladder is still
  running on the card (SATURATED and not yet proven unrelievable). Because reclaim is verified, a deferred
  demand either gets its space demonstrably freed within a bounded number of cycles or the ladder proves
  nothing remains to reclaim. The caller runs the described rungs and the request re-asks next cycle.
- Once reclaim is exhausted and the demand still does not fit, the verdict depends on the shortfall's cause.
  If the worker's own committed load exceeds capacity even after full reclaim, live samplers legitimately
  hold the card and the head DEFERS to wait for a slot like any queued job. If the shortfall is foreign
  (the worker's own committed load fits capacity but the card is consumed by baseline/desktop load), the
  candidate is admitted only when it physically fits the truthful device-free reading net of the noise
  buffer right now, and only for the true head of queue: fitting into reality, never into hope, and never
  handing that physical room to a non-head line-skipper the head would then starve behind. Otherwise it
  DEFERS.
- A candidate that cannot fit an even fully-cleared card DENIES: no escalation on this card could seat it.

There is no overcommit-admit path. A head that never becomes admittable while the device is idle is caught by
the structural-queue-wedge recovery supervisor (the deadlock detector feeding the worker recovery
coordinator), which soft-resets the pools and, failing that, faults the wedged jobs non-retryably so the
horde reissues them elsewhere. Before that, a head starved past :data:`_STARVATION_DIAGNOSTIC_SECONDS` whose
remaining deficit is held by its own idle sibling contexts (a bare CUDA context that weight eviction cannot
reclaim, freed only when the process exits) escalates to a verified context teardown: it DEFERS with a
REDUCE_LIVE_CONTEXTS actuation that reduces the live inference-context count, protecting the head's own target
slot and every busy process, and re-asks for a verified FITS once the room frees. That escalation is the
preload/whole-card path's alone; a dispatch-gate hold never tears a context down. A head deferred past the
threshold with the ladder exhausted and no such teardown target emits a warning naming the arithmetic and
increments :attr:`VramArbiter.starvation_diagnostics`, but it never admits: the job stays queued for that
recovery machinery to reroute.

Staleness and cold start relax to FITS, never deny: a stale committed ledger or an unknown device total
means the measured floor cannot be trusted, so the identity degrades to admit exactly as
``evaluate_admission`` does. An all-stale ledger can never block a request.
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

_STARVATION_DIAGNOSTIC_SECONDS = 60.0
"""Age (seconds) past which a head-of-queue request that keeps deferring with reclaim exhausted and no
verified progress emits a diagnostic warning naming the full admission arithmetic. This is observability, not
an admit path: the head stays queued. The structural-queue-wedge recovery supervisor is the mechanism that
actually reroutes a never-admittable idle-device head (it soft-resets the pools at the wedge horizon and then
faults the wedged jobs non-retryably for horde reissue); this warning is a slower, self-explaining backstop
that fires if such a head is somehow still parked, so the arithmetic is in the log for the post-mortem."""


class VramRequestKind(StrEnum):
    """The kind of device-memory demand a request represents.

    The kind selects which arithmetic prices it: the disaggregated sampling gate reasons about concurrent
    activation headroom, every other kind about the ledger-driven admission identity.
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
    """The demand is within capacity, physically fits the truthful device-free reading despite foreign load,
    or the measured floor could not apply and relaxed to admit."""
    DEFER = "defer"
    """The demand does not fit now; actuations are attached to relieve pressure and the request re-asks."""
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

    def cycle_safety_off_gpu(self, device_index: int | None) -> bool:
        """Cycle the safety model off the GPU to reclaim its context."""
        ...


@dataclass(frozen=True)
class VramRequest:
    """One priced demand on a device's VRAM, evaluated against the frozen cycle snapshot.

    ``candidate_delta_mb`` is the request's marginal device cost net of any weights already resident in the
    target process. None means the caller could not price it: the arbiter then charges nothing, so an
    unpriceable candidate is never denied by the measured overlay. ``sampling_peak_mb`` is only consulted
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
    nothing: its weights are already in the measured committed floor and its next activation is the monolithic
    status quo the card has already demonstrated it holds. Such a request is admitted directly, the same way a
    disaggregated stage dispatch to a resident lane is never withheld, because the ledger identity cannot
    express a no-op (the resident model's own reservation can legitimately sit above the noise-adjusted
    admission ceiling, which would otherwise deny a dispatch that needs no memory). Set only when the target
    genuinely holds the weights in VRAM; a RAM-staged load whose weights materialise on dispatch leaves it
    False so the identity prices the materialisation."""
    own_planned_unmaterialized_mb: float = 0.0
    """The portion (MB) of the device's planned overlay attributable to this request's own target process, netted
    out of the identity so a re-ask can never double-count itself. A preload that was admitted, recorded a
    planned charge, and then had its target reclaimed or its process die before the load materialised leaves
    that charge outstanding until the overlay reconciles; if the head re-asks the same load while the charge
    lingers, the unnetted identity would count the load once as this planned charge and once as the candidate
    delta and defer forever on its own footprint. Subtracting the target process's own planned charge makes the
    request's load count at most once (as the candidate), while every other process's planned load stays fully
    charged."""
    is_head_of_queue: bool = False
    first_of_kind: bool = False
    starved_seconds: float = 0.0
    """Seconds this head has been the undispatched head of an idle device, used only to time the
    starvation diagnostic; zero when a live job holds the card (the head is then queued behind live work,
    not starved)."""
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
    would defer forever. This lets a head starved past the escalation threshold escalate to a verified context
    teardown regardless of that warrant. Only the preload/whole-card path sets it; a dispatch-gate hold leaves
    it False so a line-skip or staged-dispatch reconciliation never tears a context down."""


@dataclass(frozen=True)
class DeviceVramState:
    """The frozen per-device measurement the arbiter prices a cycle's requests against.

    The fields carry exactly what the two arithmetics need. The admission identity reads the raw device
    total, the reconciler baseline, the committed floor, the planned overlay, and per-process reservations.
    The concurrent-sampling headroom reads the fixed per-process overhead, the marginal per-context cost,
    the live context counts, the operator reserve, the lane decode spike, and the in-flight sampling peaks.
    The governor state, truthful device-free reading, and verified-reclaim-exhaustion flag drive the
    foreign-pressure branch of admission once the escalation ladder is exhausted.
    """

    total_vram_mb: float | None
    """Raw device total VRAM (MB), or None at cold start (relaxes every verdict on this card to FITS)."""
    baseline_mb: float
    """The reconciler's measured shared-device baseline (MB) netted out of the total to form capacity."""
    committed_vram_mb: float
    """The worker's committed-VRAM ledger sum (MB): per-process context plus allocator reservation."""
    planned_unmaterialized_mb: float
    """VRAM (MB) admitted for in-flight work but not yet reflected in the committed floor."""
    committed_is_stale: bool
    """True when a committed-ledger contributor's report has aged past the staleness bound."""
    noise_buffer_mb: float = _ADMISSION_NOISE_BUFFER_MB
    """The admission noise slack (MB) subtracted on top of the baseline, derived per device at snapshot
    assembly from the total via :func:`admission_noise_buffer_mb` so it scales with card capacity; direct
    construction defaults to the floor."""
    per_process_reserved_mb: Mapping[int, float] = field(default_factory=dict)
    """Live GPU processes' measured allocator reservation (MB) keyed by process id."""
    idle_process_ids: frozenset[int] = frozenset()
    """Process ids whose lane is idle (a valid RELEASE_CACHE target)."""
    busy_process_ids: frozenset[int] = frozenset()
    """Process ids actively working (never a RELEASE_CACHE target); kept for the boundary invariant."""
    num_loaded_inference_processes: int = 0
    """Inference processes holding a model on this card (drives the extra-context count)."""
    safety_context_count: int = 0
    """Number of on-GPU safety contexts on this card."""
    post_process_context_count: int = 0
    """Number of on-GPU post-processing contexts on this card."""
    vae_lane_context_count: int = 0
    """Number of on-GPU VAE/image-lane contexts on this card."""
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
    """The device-free governor's committed state for this card at snapshot assembly, or None when the
    governor has not sampled yet. Carries the truthful NVML device-level proximity-to-cliff read into the
    arbiter: a card the governor calls SATURATED is over the paging cliff however the ledger prices it. When
    SATURATED and not yet :attr:`reclaim_unresolved`, the verified reclaim ladder is still working the card,
    so a non-fitting demand keeps deferring rather than consulting reality prematurely."""
    device_free_mb: float | None = None
    """The truthful NVML device-level free VRAM (MB) for this card, or None before the first read. The one
    figure that does not lie under WDDM: consulted only in the foreign-pressure branch, where a candidate is
    admitted iff it physically fits ``device_free_mb - noise_buffer_mb`` despite the card being consumed by
    load the worker did not commit."""
    reclaim_unresolved: bool = False
    """True when the governor's verified reclaim ladder exhausted its rungs on this card while still
    SATURATED (nothing the worker can give back relieved it). Signals that reclaim is finished, so admission
    may move past the defer-behind-reclaim rule to the foreign-versus-own-load shortfall analysis."""

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
    foreign_pressure_admit: bool = False
    """True on the one FITS that is not a ledger-capacity fit: the candidate does not fit the worker's own
    admission capacity, but with reclaim exhausted and the worker's own committed load within capacity, it
    physically fits the truthful device-free reading net of the noise buffer despite foreign load. The caller
    loads it under the heavy-head grace (it may sample slowly while foreign load holds the card) rather than
    as an ordinary co-resident admit."""

    @property
    def admits(self) -> bool:
        """Whether this verdict would let the request proceed (FITS)."""
        return self.disposition is VramDisposition.FITS


def _relaxed_verdict(request: VramRequest, device_index: int | None) -> VramVerdict:
    """Build the degraded FITS verdict for a card with no known state (cold start or missing snapshot)."""
    measured = evaluate_admission(
        measured_committed_mb=0.0,
        planned_unmaterialized_mb=0.0,
        candidate_delta_mb=request.candidate_delta_mb if request.candidate_delta_mb is not None else 0.0,
        total_vram_mb=None,
        baseline_mb=0.0,
        committed_is_stale=False,
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
    side effect is the starvation diagnostic: a head deferred past the diagnostic horizon with reclaim
    exhausted emits a warning naming the arithmetic and advances :attr:`starvation_diagnostics`, throttled to
    once per cycle per device.
    """

    def __init__(self) -> None:
        """Initialise with no cycle snapshot and zeroed observability counters."""
        self._cycle: MeasuredVramSnapshot | None = None
        self._cycle_seq = 0
        self._starvation_diag_cycle: dict[int, int] = {}
        self.admission_foreign_pressure_defers = 0
        self.starvation_diagnostics = 0
        self.starvation_context_teardowns = 0

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
        # Net the request's own outstanding planned charge out of the overlay before pricing it. That charge is
        # this same load admitted on an earlier cycle and not yet materialised; the candidate delta already
        # represents it, so leaving it in the overlay would count the load twice and let a re-ask defer forever
        # on its own footprint. Only the target process's own charge is removed; every other planned load stays.
        net_planned_mb = max(0.0, state.planned_unmaterialized_mb - max(0.0, request.own_planned_unmaterialized_mb))
        return evaluate_admission(
            measured_committed_mb=state.committed_vram_mb,
            planned_unmaterialized_mb=net_planned_mb,
            candidate_delta_mb=candidate_delta_mb,
            total_vram_mb=state.total_vram_mb,
            baseline_mb=state.baseline_mb,
            noise_buffer_mb=state.noise_buffer_mb,
            committed_is_stale=state.committed_is_stale,
        )

    def _evaluate_admission(self, request: VramRequest, state: DeviceVramState) -> VramVerdict:
        measured = self._measured(request, state)
        if request.candidate_already_resident:
            # The candidate's weights already occupy VRAM on the target process: dispatching (or preloading) onto
            # it materialises nothing. Its footprint is already in the measured committed floor and its next
            # activation is the monolithic status quo the card has already served, so there is nothing to admit
            # into. The ledger identity cannot express this no-op, because the resident model's own reservation
            # can legitimately sit above the noise-adjusted ceiling; pricing it there would withhold a dispatch
            # to an idle, already-loaded model and wedge the head behind a slot that is already open. This is the
            # whole-card analogue of the disaggregated stage dispatch that a resident lane never withholds.
            return VramVerdict(
                disposition=VramDisposition.FITS,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=f"candidate already resident on the target process; no new VRAM. {measured.reason()}",
                measured=measured,
            )
        if measured.fits:
            return VramVerdict(
                disposition=VramDisposition.FITS,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=measured.reason(),
                measured=measured,
            )

        actuations = self._escalation_ladder(state, request)
        if actuations or self._verified_reclaim_unfinished(state):
            # Reclaim can still free space on this card: the arbiter's own per-cycle ladder still emits rungs,
            # or the governor's verified reclaim ladder is still running (SATURATED and not proven
            # unrelievable). Because reclaim is verified, a deferred demand either gets its space demonstrably
            # freed within a bounded number of cycles or the ladder proves nothing remains. Never admit into an
            # over-commit that a rung could still relieve; the caller runs the rungs and the request re-asks.
            return VramVerdict(
                disposition=VramDisposition.DEFER,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=measured.reason(),
                measured=measured,
                required_actuations=actuations,
            )

        if self._structurally_impossible(request, measured):
            return VramVerdict(
                disposition=VramDisposition.DENY,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=f"candidate cannot fit an empty card; measured: {measured.reason()}",
                measured=measured,
            )

        # Reclaim is exhausted and the demand still does not fit. Before the shortfall analysis, a head starved
        # past the escalation threshold whose remaining deficit is held by its own idle sibling contexts (which
        # weight eviction cannot reclaim, because a context's VRAM returns only when its process exits) escalates
        # to a verified context teardown rather than deferring forever. It never force-admits: the teardown only
        # frees room and the head re-asks for a verified FITS next cycle.
        teardown_actuations = self._starvation_context_teardown(request)
        if teardown_actuations is not None:
            self.starvation_context_teardowns += 1
            return VramVerdict(
                disposition=VramDisposition.DEFER,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=(
                    f"head starved {request.starved_seconds:.0f}s with weight reclaim exhausted and idle sibling "
                    f"contexts holding the deficit; tearing idle contexts down. measured: {measured.reason()}"
                ),
                measured=measured,
                required_actuations=teardown_actuations,
            )

        # The verdict follows the shortfall's cause.
        capacity_mb = measured.capacity_mb
        candidate_delta_mb = request.candidate_delta_mb if request.candidate_delta_mb is not None else 0.0
        own_committed_demand_mb = measured.measured_committed_mb + measured.planned_unmaterialized_mb
        if capacity_mb is None or own_committed_demand_mb > capacity_mb:
            # The worker's own committed load exceeds capacity even after full reclaim: live samplers
            # legitimately hold the card. The head waits for a slot like any queued job.
            self._note_starvation_diagnostic(
                request,
                state,
                measured,
                cause="the worker's own committed load holds the card after full reclaim",
            )
            return VramVerdict(
                disposition=VramDisposition.DEFER,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=f"own committed load holds the card after full reclaim; measured: {measured.reason()}",
                measured=measured,
            )

        # Foreign shortfall: the worker's own committed load fits capacity, so the candidate tips the ledger
        # over only because the card is consumed by load the worker did not commit. Consult reality: admit iff
        # the candidate physically fits the truthful device-free reading net of the noise buffer right now.
        device_free_mb = state.device_free_mb
        physically_fits_reality = (
            device_free_mb is not None and candidate_delta_mb <= device_free_mb - state.noise_buffer_mb
        )
        if physically_fits_reality and request.is_head_of_queue:
            return VramVerdict(
                disposition=VramDisposition.FITS,
                request_kind=request.kind,
                device_index=request.device_index,
                reason=(
                    f"foreign pressure but candidate {candidate_delta_mb:.0f} MB physically fits device-free "
                    f"{device_free_mb:.0f} - noise {state.noise_buffer_mb:.0f} MB: admit into reality"
                ),
                measured=measured,
                foreign_pressure_admit=True,
            )
        # The best-effort over-budget admit belongs to the true head of queue alone. A non-head request (a
        # line-skip job jumping the queue) is denied it even when the card physically has room right now,
        # because materialising into that room starves the head it skipped: the head needs the same space and
        # took precedence. The non-head request defers so the head keeps first claim on the reclaimed card.
        self.admission_foreign_pressure_defers += 1
        self._note_starvation_diagnostic(
            request,
            state,
            measured,
            cause="foreign pressure and the candidate does not physically fit measured device-free",
        )
        if physically_fits_reality:
            reason = (
                f"foreign pressure; candidate {candidate_delta_mb:.0f} MB physically fits device-free but the "
                f"best-effort over-budget admit is reserved for the head of queue and {request.job_label} is not "
                f"the head; measured: {measured.reason()}"
            )
        else:
            free_render = f"{device_free_mb:.0f}" if device_free_mb is not None else "unknown"
            reason = (
                f"foreign pressure; candidate {candidate_delta_mb:.0f} MB does not fit device-free "
                f"{free_render} - noise {state.noise_buffer_mb:.0f} MB; measured: {measured.reason()}"
            )
        return VramVerdict(
            disposition=VramDisposition.DEFER,
            request_kind=request.kind,
            device_index=request.device_index,
            reason=reason,
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
        return tuple(commands)

    @staticmethod
    def _verified_reclaim_unfinished(state: DeviceVramState) -> bool:
        """Whether the governor's verified reclaim ladder is still working this card.

        The verified ladder runs only while the card is SATURATED (over the paging cliff). It is unfinished
        while the governor calls the card SATURATED and the ladder has not yet proven the card unrelievable
        (:attr:`DeviceVramState.reclaim_unresolved`). While it is unfinished a non-fitting demand keeps
        deferring, because rungs the arbiter's own per-cycle ladder does not describe (lane pauses, safety
        off-GPU) may still free the needed space.
        """
        return state.governor_state is GovernorState.SATURATED and not state.reclaim_unresolved

    @staticmethod
    def _starvation_context_teardown(request: VramRequest) -> tuple[ActuatorCommand, ...] | None:
        """Return a context-teardown actuation when a starved head's deficit is held by idle sibling contexts.

        The per-cycle ladder's weight-eviction rungs (release cache, evict idle model) cannot reclaim a bare
        CUDA context: its VRAM returns only when the process exits. A head whose remaining over-commit is
        exactly those idle sibling contexts, with no reclaimable model or cache left, would otherwise defer
        indefinitely (the ordinary activation-peak warrant behind ``can_reduce_live_contexts`` does not trip on
        a bare-context over-commit). Once such a head has been the undispatched head of an idle device past
        :data:`_STARVATION_DIAGNOSTIC_SECONDS`, this escalates to a single REDUCE_LIVE_CONTEXTS command: the
        caller reduces the live inference-context count, tearing the idle contexts down while protecting the
        head's own target slot and every busy process, and the freed room is verified at device level before the
        head is admitted (the escalation never force-admits). Restricted to the preload/whole-card path: a
        dispatch-gate hold is a MONOLITHIC_DISPATCH request and never tears a context down, matching the
        ``can_reduce_live_contexts`` semantics that gate stays under.
        """
        if request.kind is not VramRequestKind.PRELOAD:
            return None
        if not (request.is_head_of_queue and request.idle_contexts_teardownable):
            return None
        if request.starved_seconds < _STARVATION_DIAGNOSTIC_SECONDS:
            return None
        return (ActuatorCommand(kind=ActuatorCommandKind.REDUCE_LIVE_CONTEXTS, device_index=None),)

    def _note_starvation_diagnostic(
        self,
        request: VramRequest,
        state: DeviceVramState,
        measured: AdmissionVerdict,
        *,
        cause: str,
    ) -> None:
        """Emit the starvation diagnostic for a long-deferred idle-device head, throttled per cycle per card.

        Fires only for a head-of-queue request that has been the undispatched head of an idle device past
        :data:`_STARVATION_DIAGNOSTIC_SECONDS` while reclaim is exhausted and made no verified progress. It is
        observability, not an admit: the caller still defers. The warning names the full measured arithmetic
        so a post-mortem of a parked head reads the exact inequality; the structural-queue-wedge recovery
        supervisor is what actually reroutes the job.
        """
        if not request.is_head_of_queue or request.starved_seconds < _STARVATION_DIAGNOSTIC_SECONDS:
            return
        device_key = request.device_index if request.device_index is not None else 0
        if self._starvation_diag_cycle.get(device_key) == self._cycle_seq:
            return
        self._starvation_diag_cycle[device_key] = self._cycle_seq
        self.starvation_diagnostics += 1
        logger.warning(
            f"Head-of-queue {request.job_label} deferred {request.starved_seconds:.0f}s "
            f">= {_STARVATION_DIAGNOSTIC_SECONDS:.0f}s with the reclaim ladder exhausted and no verified "
            f"progress ({cause}); it stays queued for the structural-wedge recovery supervisor to reroute. "
            f"Measured: {measured.reason()}.",
        )

    @staticmethod
    def _structurally_impossible(request: VramRequest, measured: AdmissionVerdict) -> bool:
        """Whether the candidate alone exceeds capacity, so no escalation on this card could seat it."""
        capacity_mb = measured.capacity_mb
        if capacity_mb is None:
            return False
        candidate_delta_mb = request.candidate_delta_mb if request.candidate_delta_mb is not None else 0.0
        return candidate_delta_mb > capacity_mb
