"""Schedules model preloading, inference start, and model unloading."""

from __future__ import annotations

import enum
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import psutil
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger

from horde_worker_regen.compute_mode import is_cpu_only_install
from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import PopPauseOwner, WorkerState
from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
from horde_worker_regen.process_management.gpu.gpu_eligibility import eligible_card_indices_for
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeControlModelMessage,
    HordeInferenceControlMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo, LineSkip, NextJobAndProcess
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import (
    ALLOCATOR_CACHE_CAPABLE_PROCESS_TYPES,
    HordeProcessType,
)
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner, ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.process_management.models.model_sizing import ModelSizeTier, model_size_tier
from horde_worker_regen.process_management.resources.admission_identity import (
    admission_noise_buffer_mb,
)
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.resources.model_serviceability import (
    ModelServiceabilityVerdict,
    assess_model_serviceability,
    model_footprint_figures_for_baseline,
)
from horde_worker_regen.process_management.resources.reclaim_ladder import (
    CacheReleaseTarget,
    IdleResidentModel,
    LadderCandidates,
    LaneReclaimCandidate,
    ReclaimRung,
    ReclaimRungKind,
    VerifiedReclaimLadder,
    build_reclaim_ladder,
)
from horde_worker_regen.process_management.resources.resource_budget import (
    _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB,
    BudgetVerdict,
    CommittedReserveLedger,
    RamBudget,
    RamPressureVerdict,
    StreamForecast,
    VramBudget,
    WholeCardResidencyState,
    assess_ram_pressure,
    effective_inference_reserve_mb,
    forecast_weight_streaming,
    is_model_locally_unservable_for,
    platform_context_constant_mb,
    predict_job_decode_spike_mb,
    predict_job_footprint_mb,
    predict_job_post_processing_vram_mb,
    predict_job_sampler_only_vram_mb,
    predict_job_sampling_vram_mb,
    predict_job_weight_mb,
)
from horde_worker_regen.process_management.resources.run_metrics import ChurnKind
from horde_worker_regen.process_management.resources.vram_arbiter import (
    ActuatorCommand,
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
)
from horde_worker_regen.process_management.resources.vram_attribution import _REPORT_STALENESS_SECONDS
from horde_worker_regen.process_management.resources.vram_footprints import (
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    ResolutionBucket,
)
from horde_worker_regen.process_management.scheduling.context_overhead_model import ContextOverheadModel
from horde_worker_regen.process_management.scheduling.governance import (
    AdmissionDecision,
    CardProcessSnapshot,
    ClearProcessDraining,
    EvictIdleModels,
    GovernanceAction,
    HostMemorySnapshot,
    InferenceSlotSnapshot,
    MarkProcessDraining,
    PausePops,
    PreloadSlotSnapshot,
    RamGovernorState,
    RamReclaimOutcome,
    RecycleProcess,
    ReduceCardProcesses,
    ReduceWorkerProcesses,
    ResourceGovernor,
    RestoreCardProcess,
    RestoreWorkerProcess,
    SetPopHold,
    StopTrackingShedCard,
    StopTrackingWorkerShed,
    WholeCardResidency,
    WholeCardResidencyMachine,
    WorkerProcessShedState,
    card_preload_order,
    compute_preload_disallowed_processes,
    decide_degrade_response,
    decide_process_reduction,
    decide_ram_reclaim_outcome,
    decide_shed_card_restore,
    max_coresident_for_peak,
    preload_concurrency_blocked,
    select_head_room_process_id,
)
from horde_worker_regen.process_management.scheduling.model_affinity import affinity_active
from horde_worker_regen.process_management.scheduling.performance_model import PerformanceModel, signature_from_job
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyAccumulator, SlotDutyBucket
from horde_worker_regen.process_management.scheduling.workload_flow import (
    DISPATCH_ADMISSION_FLOW,
    POST_PROCESS_RESERVE_FLOW,
    PRELOAD_ADMISSION_FLOW,
)
from horde_worker_regen.telemetry_spans import span_preload_model
from horde_worker_regen.utils.config_coercion import config_number
from horde_worker_regen.utils.job_utils import (
    get_single_job_magnitude as _get_single_job_effective_megapixelsteps,
)
from horde_worker_regen.utils.job_utils import line_skip_candidate_emps_limit
from horde_worker_regen.utils.vram_quota import effective_post_process_vram_quota_mb

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData


@dataclass(frozen=True)
class LatestPreloadAdmission:
    """Operator-facing record of the most recent preload-admission decision."""

    decision: AdmissionDecision
    """The admission gate's decision."""
    model: str | None
    """Model whose queued job was judged, when available."""
    process_id: int | None
    """Target inference process selected by the decision, when one was selected."""
    reason: str
    """Short human-readable explanation for the decision."""
    timestamp: float
    """Worker wall-clock time when the decision was recorded."""


_SPECULATIVE_DISPATCH_MIN_FREE_VRAM_MB = 3000.0
"""Minimum device-wide free VRAM required to dispatch a job to a spare process *ahead* of a
sampling slot opening (speculative pre-staging under the GPU sampling lease). Staging a job loads
its model and encodes conditioning into VRAM before it can sample; this headroom guards against
over-committing the GPU when pre-staging. Below it, dispatch falls back to the sampling-slot cap."""

_RESIDENCY_GRACE_SECONDS = 30.0
"""How long a model stays protected from RAM eviction after its last live demand, in the
models-exceed-processes regime. Bridges the gap between a model's consecutive jobs so a
process does not disk-reload the very model it just used when the next job for it has not yet
been popped."""

_DEFAULT_VRAM_RESERVE_MB = 2048.0
"""Fallback VRAM reserve (MB) used until the live config value is read. Matches the
``vram_reserve_mb`` config default; covers transient spikes such as tiled VAE decode."""

_DEFAULT_RAM_RESERVE_MB = 4096.0
"""Fallback system-RAM reserve (MB) used until the live config value is read. Matches the
``ram_reserve_mb`` config default; keeps resident-in-RAM weights from forcing the OS to page."""

_STALE_RAM_UNLOAD_REPLACE_BYTES = 1024 * 1024 * 1024
"""RSS threshold above which a model-less idle process is still materially holding RAM after unload."""

_PRELOAD_FIRST_REPORT_GRACE_SECONDS = 5.0
"""How long a just-sent preload may still look idle before its first child state report arrives.

The parent records the model as ``LOADING`` immediately after sending ``PRELOAD_MODEL``, but the child
may still read as ``WAITING_FOR_JOB`` until it drains the control pipe and publishes its first preload
state. This short grace keeps stale-entry cleanup from expiring a healthy, just-sent preload while still
letting genuinely abandoned loading entries clear promptly.
"""
_LINE_SKIP_REJECTION_LOG_INTERVAL = 5.0
"""Minimum seconds between repeats of an identical line-skip rejection log line.

Line-skip is re-evaluated every (sub-second) scheduling pass while a head job is blocked, so an
unthrottled per-candidate rejection log floods the file with thousands of identical lines during a
stall. Repeats of the same (candidate, reason) are collapsed to one per this interval; a new candidate
or a changed reason still logs immediately, so no distinct information is lost."""

_LINE_SKIP_REJECTION_LOG_MAX_KEYS = 256
"""Cap on remembered (candidate, reason) throttle keys before stale ones are pruned."""

_RELEASE_CACHE_MIN_RECLAIMABLE_MB = 256.0
"""Minimum reclaimable allocator cache (MB) a GPU process must hold to qualify as a RELEASE_CACHE target.

A RELEASE_CACHE actuation runs ``torch.cuda.empty_cache`` on the lane, which returns only the allocator's
reserved-but-unallocated blocks (``process_reserved_mb - process_allocated_mb``); a process whose reservation
is its resident weights or components (a component/VAE/post-process lane holding encoders, allocated close to
reserved) has no such cache, so asking it to release frees nothing. Requiring this measured margin keeps a
resident-weight lane out of the release-target set so the escalation ladder does not emit a rung that can
never yield, which would otherwise keep the ladder non-empty forever and defer a head that reclaim can
never actually relieve."""

_SAFETY_GPU_LOAD_CHARGE_MB = 3044.0
"""The device VRAM (MB) charged when the safety process is loaded onto the GPU, for the arbiter's SAFETY_LOAD
gate. A documented conservative seed for the idle CLIP model plus its CUDA context. DeepDanbooru, BLIP, the
aesthetic head, and evaluation activations are explicitly reclaimable and are not fixed safety residency. The
worker holds no learned CLIP footprint, so this static figure prices restore rather than a per-run watermark;
erring high keeps safety off-GPU one more cycle rather than restoring it onto a card it would over-commit."""

_SAFETY_PLACEMENT_PAUSE_STREAK = 2
"""Consecutive control cycles the safety charge must fail to fit beside the largest learned solo sampling peak
before the runtime safety-placement policy moves safety off the GPU. Small but greater than one so a single
transient reading (a stale watermark, a momentary over-commit as a model loads) does not evict safety; paired
with :data:`_SAFETY_PLACEMENT_RESTORE_STREAK` it forms the hysteresis band that keeps the policy from flapping
the safety process on and off the card every cycle."""

_SAFETY_PLACEMENT_RESTORE_STREAK = 5
"""Consecutive control cycles the safety charge must fit beside the largest learned solo sampling peak *with a
proportional margin* before the runtime safety-placement policy restores safety to the GPU. Deliberately larger
than :data:`_SAFETY_PLACEMENT_PAUSE_STREAK` (evict quickly on real pressure, readmit slowly and only once the
card has proven durable headroom) so the readmit does not immediately re-trip the evict on the next heavy job."""

_SAFETY_BACKLOG_PRIORITY_DEPTH = 2
"""Safety backlog depth above which GPU safety restoration is prioritized over placement inertia."""

_DISPATCH_STALL_MIN_SECONDS = 10.0
"""How long the head must be continuously undispatched before the dispatch-stall diagnostic speaks.

Reuses the head-starvation clock so an ordinary one-tick gap between jobs (or a model mid-preload) is
never reported; only a head that has been parked this long, with nothing dispatching, is explained."""

_DISPATCH_STALL_LOG_INTERVAL_SECONDS = 30.0
"""Minimum gap between repeats of the dispatch-stall diagnostic for an unchanged reason, so the
sub-second control loop cannot spam it. A changed reason logs immediately (the stall's cause shifted)."""

_WHOLE_CARD_ESTABLISH_GRACE_SECONDS = 120.0
"""How long after a whole-card residency is established the worker may keep the queue intentionally held
(heavy head deferred while idle siblings stop, safety cycles off-GPU, and the model loads ~11GB) without
the recovery supervisor treating it as a structural wedge. The establishment is deliberately slow now
that it cycles the safety process, so the plain ``_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS`` (20s) window would
otherwise soft-reset the pools mid-setup. Bounded so a residency that genuinely never loads still trips
the supervisor."""

_WHOLE_CARD_DRAIN_SETTLE_SECONDS = 20.0
"""How long after a whole-card teardown reaches sole residency the head waits for the live free-VRAM reading to
confirm the drain before loading best-effort regardless.

A teardown frees the stopped siblings' VRAM asynchronously, so the live measurement can lag or be briefly
unavailable. The live reading dispatches the head the moment it confirms; this bound guarantees the head is
never parked indefinitely on a stuck or missing measurement; once the teardown has been structurally complete
this long it loads on the structural ``fits_alone`` guarantee (the grant precondition for a residency).
Comfortably under ``_WHOLE_CARD_ESTABLISH_GRACE_SECONDS`` so the head always dispatches before the recovery
supervisor would treat the held queue as a structural wedge."""

_WHOLE_CARD_RESTORE_GRACE_SECONDS = 60.0
"""How long after a whole-card residency is *restored* the recovery supervisor keeps ignoring a queue
wedge. Restoring respawns the torn-down sibling inference processes and cycles the safety process back
on-GPU, each a ~20s spawn during which the queue is briefly unservable. Without this grace that churn
looks like a structural wedge and soft-resets the pools, which then cascades into further whole-card
churn and more resets. Covers the respawn window; bounded so a genuine post-restore wedge still trips
the supervisor."""

_HEAVY_HEAD_LOAD_GRACE_SECONDS = 120.0
"""How long after a heavy head is admitted on the over-budget classification path (for example under
foreign-pressure fit-into-reality, when a model streams even with the whole card to itself) the recovery
supervisor keeps ignoring a queue wedge. Such a head bypasses the whole-card branch, so it is not covered
by ``_WHOLE_CARD_ESTABLISH_GRACE_SECONDS``, yet its multi-gigabyte load equally holds the queue and must
not be mistaken for a structural wedge that faults the never-run backlog. Bounded so a head that genuinely
never loads still trips the supervisor."""

_RAM_RECLAIM_CYCLE_GRACE_SECONDS = 60.0
"""How long after the worker deliberately cycles an idle inference process to reclaim allocator-retained
RAM (``_replace_stale_ram_unload_process``) the recovery supervisor keeps ignoring a queue wedge. The
cycle restarts the slot (a ~20s spawn) and the next head must then preload onto it (another ~20s+), a
window in which the queue is legitimately unservable through no fault of the pool. Without this grace
that deliberate, bounded hold ages past ``_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS`` (20s) and is mistaken for
a structural wedge, soft-resetting the pools and faulting the perfectly-servable backlog (in a
sole-process configuration this drops every queued job over a window the worker itself created).
Covers the respawn + preload window; bounded so a cycle that genuinely never recovers still trips the
supervisor."""

_SCHEDULER_DIAGNOSTIC_REPEAT_SECONDS = 30.0
"""Minimum cadence for unchanged high-frequency scheduler diagnostics.

These diagnostics are useful when reconstructing residency and performance behavior, but they sit inside
the scheduler's fast polling loop. Log immediately when the decision state changes, otherwise emit only
periodic reminders with a suppressed-repeat count.
"""

_SCHEDULER_DIAGNOSTIC_MB_BUCKET = 256.0
"""Bucket size for deciding whether memory telemetry changed enough to re-log a scheduler diagnostic."""


# The model-size tier classification (and its baseline value sets) lives in the shared, torch-free
# ``model_sizing`` module so the scheduler and the job popper's large-model pop limiters classify "very large"
# identically. Aliased to the historical private name so the existing references read unchanged.
_ModelSizeTier = ModelSizeTier

_OVERLAP_HEADWAY_MIXED_HEAVY = 0.5
"""Fraction of the in-flight job's sampling that must be done before a concurrent job joins it when
exactly one side of the overlap is heavy (e.g. an SDXL is running and a cheaper SD1.5 wants to join,
or vice versa). Gives the heavier job room to get past its memory-hungry startup before another
sampler adds pressure."""

_OVERLAP_HEADWAY_BOTH_HEAVY = 0.75
"""Fraction of the in-flight job's sampling that must be done before a *second heavy* job joins it.
Two SDXL jobs stacking their weight loads and activation peaks is the over-subscription that thrashes
a sampler into a watchdog teardown, so the running job must be most of the way done first."""

_OVERLAP_HEADWAY_AMPLE_VRAM = 0.15
"""Headway applied instead of the mixed/both-heavy fractions when the device's measured free VRAM
absorbs the candidate's full predicted sampling peak plus the configured reserve.

The strict fractions price every card as tight; on a high-VRAM card serving a heavy-only queue that
prices a second configured thread out of existence (a both-heavy candidate waits for 75% progress, so
two threads converge to ~one effective thread). When the measurement says the newcomer's whole peak
fits *now*, the over-subscription the strict headway guards against cannot occur; a small headway is
kept so the running job clears its memory-hungry startup before a sibling adds pressure."""

_OVERLAP_HEADWAY_SCALE_HIGH_PERFORMANCE = 0.5
"""Multiplier applied to the required overlap headway when the worker runs in high-performance mode.

High-performance operators have provisioned the card for aggressive co-sampling and want the next job's
sampling to overlap the tail of the current one sooner. Halving the headway brings the newcomer in
earlier while the VRAM arbiter still independently decides whether the card can hold the overlap."""

_OVERLAP_HEADWAY_SCALE_MODERATE_PERFORMANCE = 0.75
"""Multiplier applied to the required overlap headway in moderate-performance mode: a milder pull-in than
high-performance mode, still gated by the arbiter's memory verdict."""


def _performance_mode_headway_scale(bridge_data: reGenBridgeData) -> float:
    """Return the overlap-headway multiplier for the worker's performance mode (1.0 outside the fast modes).

    Higher performance modes shrink the sampling headway a newcomer must wait for, so concurrent inference
    starts sooner. The memory arbiter still gates whether the overlap fits, so this only moves *when* an
    admissible overlap begins, never *whether* an over-committing one is allowed.
    """
    if bridge_data.high_performance_mode:
        return _OVERLAP_HEADWAY_SCALE_HIGH_PERFORMANCE
    if bridge_data.moderate_performance_mode:
        return _OVERLAP_HEADWAY_SCALE_MODERATE_PERFORMANCE
    return 1.0


class _WholeCardDemandOutcome(enum.Enum):
    """How the whole-card residency decision resolves a budget-gated head's preload.

    Returned by :meth:`InferenceScheduler._decide_whole_card_demand` so the budget-admission orchestrator
    can map each outcome to proceed/defer without re-deriving the residency state.
    """

    FALL_THROUGH = enum.auto()
    """No whole-card reservation applies (not demanded, or declined as untrustworthy); continue to the
    ordinary VRAM/RAM verdict."""
    PRESTAGE = enum.auto()
    """The head's weights are pre-staging into spare RAM while a live job drains; skip the verdict and
    send the preload now (convergence collapses the card to sole residency before it samples)."""
    DEFER = enum.auto()
    """The reservation is mid-teardown (idle siblings stopping, safety cycling off-GPU, freed VRAM
    draining); defer this cycle and re-evaluate against the reduced topology next tick."""


class _PreloadJobOutcome(enum.Enum):
    """What one pending job's preload attempt means for the rest of this scheduling pass."""

    NEXT_JOB = enum.auto()
    """This job needs nothing (or was faulted); consider the next pending job."""
    STOP_PASS = enum.auto()
    """A gate deferred or consumed this cycle (RAM floor, no slot, serialization, budget); stop the pass."""
    PRELOAD_SENT = enum.auto()
    """A preload was issued for this job; the pass is done and reports success."""


@dataclass
class _PreloadActuation:
    """The head-preload context a described actuation needs when the adapter runs a deferred verdict.

    An EVICT_IDLE_MODEL or REDUCE_LIVE_CONTEXTS command targets the card on behalf of the specific head being
    adjudicated: the eviction spares the head's own target slot, and the reduction establishes whole-card
    residency for the head's job at the depth the verdict's rejected peak sized. The adapter records this for
    the current head immediately before running the verdict's commands and clears it once they have run.
    """

    job: ImageGenerateJobPopResponse
    available_process: HordeProcessInfo
    forecast: StreamForecast
    max_resident: int | None


def _preload_outcome_from_admission(decision: AdmissionDecision) -> _PreloadJobOutcome:
    """Map the public admission decision vocabulary onto the scheduler pass control enum."""
    match decision:
        case AdmissionDecision.ADMIT | AdmissionDecision.PRESTAGE:
            return _PreloadJobOutcome.PRELOAD_SENT
        case (
            AdmissionDecision.NEXT_JOB
            | AdmissionDecision.QUARANTINED
            | AdmissionDecision.UNSERVICEABLE
            | AdmissionDecision.ALREADY_LOADED
        ):
            return _PreloadJobOutcome.NEXT_JOB
        case _:
            return _PreloadJobOutcome.STOP_PASS


class InferenceScheduler:
    """Owns model preloading, inference start, and model unloading logic."""

    _state: WorkerState
    _process_map: ProcessMap
    _horde_model_map: HordeModelMap
    _job_tracker: JobTracker
    _process_lifecycle: ProcessLifecycleManager
    _runtime_config: RuntimeConfig
    _model_metadata: ModelMetadata
    _max_threads_ceiling: int
    _max_inference_processes: int
    _lru: LRUCache
    _performance_model: PerformanceModel | None

    _preload_delay_notified: bool
    _model_recently_missing: bool
    _model_recently_missing_time: float
    _batch_wait_log_time: float
    _pending_line_skip: NextJobAndProcess | None
    _model_last_in_demand: dict[str, float]
    _vram_budget: VramBudget
    _ram_budget: RamBudget
    _reserve_ledger: CommittedReserveLedger
    _vram_budget_defer_notified: bool
    _ram_budget_defer_notified: bool
    _ram_pressure_notified: bool
    _scheduler_diagnostic_log_state: dict[str, tuple[tuple[object, ...], float, int]]
    _last_preload_admission: LatestPreloadAdmission | None
    _post_processing_lane_commitments_provider: Callable[[], int]

    def __init__(
        self,
        *,
        state: WorkerState,
        process_map: ProcessMap,
        horde_model_map: HordeModelMap,
        job_tracker: JobTracker,
        process_lifecycle: ProcessLifecycleManager,
        runtime_config: RuntimeConfig,
        model_metadata: ModelMetadata,
        card_runtimes: dict[int, CardRuntime] | None = None,
        max_concurrent_inference_processes: int,
        max_inference_processes: int,
        lru: LRUCache,
        performance_model: PerformanceModel | None = None,
        reserve_ledger: CommittedReserveLedger | None = None,
        post_processing_lane_commitments_provider: Callable[[], int] | None = None,
    ) -> None:
        """Initialize the scheduler with references to the components it needs to manage.

        Args:
            state (WorkerState): The worker's state object, containing all of the mutable flags
                relating to the worker's active state and lifecycle.
            process_map (ProcessMap): The worker's ProcessMap, which tracks all active processes and
                their states.
            horde_model_map (HordeModelMap): The worker's HordeModelMap, which tracks the load state of all models
                and which processes they are loaded on.
            job_tracker (JobTracker): The worker's JobTracker, which tracks all jobs in-flight
                and is responsible for managing their state transitions.
            process_lifecycle (ProcessLifecycleManager): The worker's ProcessLifecycleManager, which is responsible
                for launching, monitoring, and killing processes as needed.
            runtime_config (RuntimeConfig): Holds the current bridge configuration snapshot.
            model_metadata (ModelMetadata): Provides lookups against the stable-diffusion model reference.
            card_runtimes (dict[int, CardRuntime] | None): The per-card runtime plan, keyed by stable device
                index, used to route a job to a card that can serve it on a multi-GPU host. ``None`` or a
                single entry means single-GPU: dispatch takes the original card-agnostic path unchanged.
            max_concurrent_inference_processes (int): The maximum number of inference processes to run at once.
            max_inference_processes (int): The maximum number of inference processes to have launched at once,
                including those that are preloading or downloading models.
            lru (LRUCache): The worker's LRU cache, used to track recently used models for unloading decisions.
            performance_model (PerformanceModel | None): Supplies an expected sampling time per dispatched
                job for the audit ledger (and, in a later phase, slow-job remediation). May be ``None``.
            reserve_ledger (CommittedReserveLedger | None): The shared committed-VRAM/RAM ledger every
                workload flow contributes to, so image generation and alchemy cannot independently admit
                against the same free VRAM. When ``None`` (unit tests driving the scheduler alone) a private
                ledger is created, so the scheduler still accounts for its own post-processing reserve.
            post_processing_lane_commitments_provider (Callable[[], int] | None): Optional count of
                non-JobTracker work already committed to the shared post-processing lane, such as
                graph-backed alchemy forms waiting for or running on that lane.
        """
        self._state = state
        self._process_map = process_map
        self._horde_model_map = horde_model_map
        self._job_tracker = job_tracker
        self._process_lifecycle = process_lifecycle
        self._runtime_config = runtime_config
        self._model_metadata = model_metadata
        self._post_processing_lane_commitments_provider = post_processing_lane_commitments_provider or (lambda: 0)
        # Per-card runtime plan for multi-GPU routing. A single entry (or None) means single-GPU, where the
        # dispatch path stays card-agnostic and byte-identical to before multi-GPU existed.
        self._card_runtimes: dict[int, CardRuntime] = card_runtimes if card_runtimes is not None else {}
        # The constructor value is the provisioned ceiling; the *live* concurrent cap is read from
        # the runtime config (see the _max_concurrent_inference_processes property) so it can change
        # at runtime without resizing the inference semaphore.
        self._max_threads_ceiling = max_concurrent_inference_processes
        self._max_inference_processes = max_inference_processes
        self._lru = lru
        self._performance_model = performance_model
        # Optional sink for between-jobs reload/respawn events, set by the manager to
        # WorkerRunMetrics.record_churn. None in unit tests that drive the scheduler directly.
        self._churn_observer: Callable[[ChurnKind], None] | None = None

        # The ledger-driven admission overlay. The baseline provider yields the reconciler's measured
        # shared-device baseline (MB) per card, wired by the manager (None until wired, and in standalone unit
        # tests: the overlay then reads baseline 0, so capacity is the raw total and the measured gate matches
        # the predictive gate). The per-card counters/headroom feed run-metrics calibration visibility.
        self._admission_baseline_provider: Callable[[int | None], float | None] | None = None
        self._admission_denials_by_device: dict[int, int] = {}
        self._admission_headroom_mb_by_device: dict[int, float | None] = {}
        # The device-free governor's growth hold per card, set each tick by the parent. True while a card is at
        # PRESSURE or SATURATED (device-level free VRAM below the soft floor): the scheduler must not grow the
        # card's VRAM footprint (no new model brought to VRAM on a process that does not already hold it, no
        # safety GPU restore, no paused-lane restart). In-flight sampling is never touched by this hold; it is
        # a truthful WDDM-cliff brake, orthogonal to the ledger admission gate. Empty (no hold) in standalone
        # unit tests, where the parent never wires the governor.
        self._vram_growth_hold_by_device: dict[int, bool] = {}
        # The device-free governor's committed state per card, pushed each governor tick by the parent alongside
        # the growth hold. Retention reads the STATE (not the derived hold boolean): weights only stay resident
        # while the card is HEALTHY, since a PRESSURE or SATURATED card is one the verified reclaim ladder is or
        # may soon be taking residents back from. Empty (defaults to HEALTHY) in standalone unit tests, where the
        # parent never wires the governor.
        self._governor_states_by_device: dict[int, GovernorState] = {}
        # Count of verified reclaim-ladder shortfalls the engine reported (a rung freed less than half its
        # promised device memory). Recorded here as a calibration counter: at reclaim time the freed figure is
        # not a footprint peak and no complete (baseline, resolution, stage) key is reconstructable, so the
        # raise-only footprint store does not apply and the signal is kept as a count. Calibration visibility.
        self._reclaim_calibration_events = 0
        # The single VRAM arbiter, injected by the manager. It is the live authority for the gated
        # preload/overlap/disaggregation/post-processing seams, pricing each demand against the cycle-frozen
        # measurement. None until wired (and in standalone unit tests), where those seams fall back to their
        # measured floors.
        self._vram_arbiter: VramArbiter | None = None
        # The truthful per-card device-free reading source, injected by the manager (parent NVML). The
        # manager-driven cycle passes its explicit reading map to build_vram_arbiter_snapshot; this provider is
        # the fallback for a self-primed snapshot (a scheduler consult before or outside a manager tick), so the
        # measured-truth identity keeps its primary input there too. None (unwired) leaves the reading absent,
        # and admission defers with the missing-reading diagnostic.
        self._device_free_mb_provider: Callable[[int], float | None] | None = None
        # The head-preload context the current deferred verdict's actuations act on, set immediately before the
        # adapter runs a verdict's commands and cleared once they have. None outside that window.
        self._preload_actuation: _PreloadActuation | None = None

        # Dispatch-time residency reconciliation state. The dispatch gate re-uses the arbiter's
        # MONOLITHIC_DISPATCH identity to check that a staged job's VRAM materialisation fits the card before it
        # is handed to a child (the moment RAM-staged weights actually commit to VRAM). A conflicting verdict
        # holds the dispatch (the job keeps its queue position, never faulted) and routes idle-resident eviction
        # through the single reclaim owner. The per-job map stamps when each held job first held, so a release is
        # attributed to reclaim (this gate emitted eviction commands for it) versus natural free (device-free
        # recovered on its own); the counters are calibration visibility only.
        self._dispatch_hold_since: dict[str, float] = {}
        self._dispatch_hold_reclaim_requested: set[str] = set()
        self._dispatch_reconciliation_holds = 0
        self._dispatch_reconciliation_conflicts = 0
        self._dispatch_reconciliation_hold_seconds = 0.0
        self._dispatch_reconciliation_released_by_reclaim = 0
        self._dispatch_reconciliation_released_by_natural_free = 0

        # The learned-footprint store, injected by the manager (one shared instance, the same the message
        # dispatcher observes into). Admission pricing of a job's sampling peak reads it so a measured
        # activation high-water raises the static per-model seed the predictor returns; a static seed
        # systematically undershoots the reserved peak (calibration saw ~11GB against a 6158MB seed). None
        # until wired (and in standalone unit tests), where every estimate falls back to the static seed.
        self._footprint_store: LearnedFootprintStore | None = None

        # Pipeline-disaggregation hooks, wired by the manager via set_disaggregation_hooks. The predicate
        # decides whether a job takes the disaggregated path (so its verdicts charge the UNet-only sampler
        # figure); the router registers an eligible job with the orchestrator, pinned to the process it was
        # scheduled onto, in place of the monolithic START_INFERENCE. Defaults keep the scheduler on the pure
        # monolithic path (every job disaggregated=False) for unit tests that drive it alone.
        self._is_disaggregatable_job: Callable[[ImageGenerateJobPopResponse], bool] = lambda _job: False
        # The stable class-eligibility predicate (no liveness/residency coupling): forecasting and VRAM
        # charging use this, so a job that *will* run disaggregated is always priced sampler-only, even during
        # a whole-card window when the lane is transiently paused. Defaults monolithic for standalone tests.
        self._is_disaggregation_class_eligible: Callable[[ImageGenerateJobPopResponse], bool] = lambda _job: False
        self._register_disaggregated_job: (
            Callable[[ImageGenerateJobPopResponse, HordeProcessInfo], Awaitable[bool]] | None
        ) = None
        # Read-only disaggregation diagnostics for the dispatch-stall classifier: the job pinning a given
        # process as its sampler, and the current in-flight sampling peaks. Defaults (no owner, empty peaks)
        # keep the classifier's disaggregation branch inert for standalone tests that never wire them.
        self._disaggregation_pin_owner: Callable[[int], str | None] = lambda _pid: None
        self._disaggregation_sampling_peaks: Callable[[], dict[str, float]] = dict

        # Runtime safety-placement policy hysteresis (see _reconcile_runtime_safety_placement). ``wants_off``
        # latches the policy's current verdict; the two streaks count consecutive cycles the safety charge did
        # not fit (drives the latch on) or fit with margin (drives it off), so a card oscillating around the
        # fit boundary does not flap the safety process on and off GPU every cycle.
        self._safety_placement_miss_streak = 0
        self._safety_placement_fit_streak = 0
        self._safety_placement_wants_off = False
        # Lifetime counts of runtime safety-placement policy actuations, for the run-metrics readback: a
        # demotion moves safety off-GPU (its charge did not fit beside the sampler), a promotion restores it
        # once the chosen card's measured free proved durable room. These count only policy-initiated moves,
        # not the whole-card residency's own safety pauses (which the lifecycle manager counts separately).
        self._safety_placement_demotions = 0
        self._safety_placement_promotions = 0

        self._preload_delay_notified = False
        self._model_recently_missing = False
        self._model_recently_missing_time = 0.0
        self._batch_wait_log_time = 0.0
        self._pending_line_skip = None
        self._model_last_in_demand = {}

        # Constructed with safe defaults; the live reserves are synced from the (reloadable) config each
        # scheduling cycle by _vram_budget_active(), which also tolerates partially-mocked test config.
        self._vram_budget = VramBudget(reserve_mb=_DEFAULT_VRAM_RESERVE_MB)
        self._ram_budget = RamBudget(reserve_mb=_DEFAULT_RAM_RESERVE_MB)
        self._reserve_ledger = reserve_ledger if reserve_ledger is not None else CommittedReserveLedger()
        self._vram_budget_defer_notified = False
        self._ram_budget_defer_notified = False
        self._ram_pressure_notified = False
        self._scheduler_diagnostic_log_state = {}
        self._last_preload_admission = None
        # One-shot log throttle, keyed by model, for the "held back as locally unservable" notice.
        self._unservable_admit_notified: dict[str, bool] = {}
        # One-shot log throttle, keyed by model, for the "declined a whole-card residency" notice (a teardown
        # demand the warrant gate did not trust; see _whole_card_warranted / _log_whole_card_declined).
        self._whole_card_declined_notified: dict[str, bool] = {}
        # Rate-limit state for line-skip rejection logs, keyed by "candidate_id:reason"; see
        # _log_line_skip_rejection. Maps the key to the monotonic time it was last emitted.
        self._line_skip_rejection_log_state: dict[str, float] = {}
        # Per-context VRAM overhead model: owns the startup-measured per-process and marginal context costs
        # and derives the figures the streaming forecast needs (see ContextOverheadModel). The manager feeds
        # it the probe measurements via set_measured_*, and its attribution tick feeds the truthful
        # NVML-derived bare-context readings via capture_idle_context_residency /
        # invalidate_idle_context_floor (per-child VRAM views are per-process artefacts under WDDM and are
        # never decomposed as device truth).
        self._overhead = ContextOverheadModel()
        # Whole-card exclusive-residency records, keyed by the device index a residency is held on. A heavy
        # model claims a card by stopping that card's idle sibling contexts (and cycling safety off-GPU on the
        # safety card); keying per card lets two heavy models on different cards each hold their own residency.
        # A single-GPU worker uses exactly one entry under the None key, identical to the prior scalar fields.
        # All reads/writes go through the ledger (see WholeCardResidencyLedger; _residency_state delegates).
        self._whole_card_ledger = WholeCardResidencyMachine()
        # The per-tick resource governor: the process manager ticks it once per control-loop iteration via
        # run_governance_tick(), independent of queue depth, so governance never depends on a particular
        # scheduling path executing (or on the inference queue being non-empty). It owns the RAM governor's
        # multi-tick bookkeeping (shed cards, draining processes), exposed under the historical attribute
        # names through the _ram_pressure_shed_cards / _processes_draining_for_ram properties.
        self._governor = ResourceGovernor(host=self)
        # When a heavy head was last admitted through the foreign-pressure physical-fit branch. Its load
        # equally holds the queue, so this bounds a wedge grace that the whole-card establishment grace does
        # not cover. 0.0 when none is loading.
        self._heavy_head_admitted_at: float = 0.0
        # When an idle inference slot was last deliberately cycled to reclaim allocator-retained RAM
        # (_replace_stale_ram_unload_process). The respawn + the next head's preload leave the queue
        # briefly unservable through no fault of the pool, so this bounds a wedge grace covering that
        # deliberate window. 0.0 when no reclaim cycle is in flight. See _RAM_RECLAIM_CYCLE_GRACE_SECONDS.
        self._ram_reclaim_cycle_at: float = 0.0
        # Head-of-queue starvation clock. Tracks the id of the job currently at the head of the queue and
        # when it first became budget-deferred onto an idle device. It only feeds
        # the arbiter's starvation diagnostic (a warning naming the arithmetic once a head is deferred past
        # the diagnostic horizon with reclaim exhausted). Reset when the head changes, a job dispatches, or a
        # live job takes the device.
        self._head_starvation_job_id: str | None = None
        self._head_starvation_since: float = 0.0

        # Dispatch-stall diagnostic throttle. When the queue has work but nothing dispatches, the scheduler
        # would otherwise return None silently; this records the last reason logged and when, so the
        # explanation is emitted at most once per interval (and immediately when the reason changes) rather
        # than every sub-second control-loop tick.
        self._dispatch_stall_last_reason: str | None = None
        self._dispatch_stall_log_time: float = 0.0

        # The parent's measured WDDM demand-paging verdict (per-process GPU shared-segment usage on the
        # worker's own children). While set, retention is denied; the rising edge triggers an idle-VRAM
        # reclaim. Always False on hosts without the telemetry.
        self._wddm_paging_active: bool = False
        # The parent's most recent WDDM paging attribution: the child PIDs whose VRAM the driver demoted to
        # system memory, mapped to their shared (system-backed) GPU MB, plus a monotonic stamp of when it was
        # recorded. Refreshed on every active verdict (not just the rising edge) so the paged-slowdown
        # watchdog reads a current victim set, and cleared the moment paging clears. See
        # :meth:`wddm_paging_victim_shared_mb_by_pid`.
        self._wddm_paging_victims_shared_mb_by_pid: dict[int, float] = {}
        self._wddm_paging_victims_updated_monotonic: float = 0.0
        # Edge-log throttle for the post-processing/sampling time-slice hold on dispatch.
        self._pp_mutex_hold_logged: bool = False

        # Capacity-normalized wall-clock accounting: every scheduler tick attributes each configured
        # inference slot's elapsed time to SAMPLING or to the gate/supply state that kept it empty, so
        # "active vs idle vs gated" is a direct read over any window. Fed once per scheduling cycle
        # (record_slot_duty); snapshotted into the stats stream and the periodic duty-cycle log line.
        self._slot_duty = SlotDutyAccumulator()
        self._slot_duty_current_hold: SlotDutyBucket | None = None

        # The head whose dispatch is currently held for post-processing-peak headroom (job id, shortfall MB),
        # or None. Set when a dispatch defers (the peak overflows the contended card now but fits it alone and
        # an in-flight sibling will free room); read by the dispatch-stall diagnostic so a held head reads as
        # an explained wait. Cleared the moment any job dispatches.

    def set_churn_observer(self, observer: Callable[[ChurnKind], None]) -> None:
        """Register the sink for between-jobs reload/respawn events (see :data:`ChurnKind`)."""
        self._churn_observer = observer

    def set_admission_baseline_provider(self, provider: Callable[[int | None], float | None]) -> None:
        """Register the source of the measured shared-device baseline (MB) per card for the admission overlay.

        The manager's :meth:`ProcessManager.latest_baseline_estimate_mb`; called with a device index (None for
        the single-GPU / worker-wide case). Until wired (and in standalone unit tests) the overlay reads a
        baseline of 0, so capacity is the raw device total and the measured gate never denies what the
        predictive gate admits.
        """
        self._admission_baseline_provider = provider

    def latest_admission_denials(self, *, device_index: int | None = None) -> int:
        """Return the count of measured-floor admission denials on a card this run (calibration visibility)."""
        return self._admission_denials_by_device.get(device_index if device_index is not None else 0, 0)

    def latest_admission_headroom_mb(self, *, device_index: int | None = None) -> float | None:
        """Return the last measured-floor admission headroom (MB) on a card, or None when the floor was unapplied."""
        return self._admission_headroom_mb_by_device.get(device_index if device_index is not None else 0)

    def set_disaggregation_hooks(
        self,
        *,
        is_disaggregatable: Callable[[ImageGenerateJobPopResponse], bool],
        is_disaggregation_class_eligible: Callable[[ImageGenerateJobPopResponse], bool],
        register_disaggregated: Callable[[ImageGenerateJobPopResponse, HordeProcessInfo], Awaitable[bool]],
        pin_owner: Callable[[int], str | None] | None = None,
        sampling_peaks: Callable[[], dict[str, float]] | None = None,
    ) -> None:
        """Wire the pipeline-disaggregation predicates and router (see the ``_is_disaggregatable_job`` attr).

        ``is_disaggregatable`` is the dispatch-time predicate (class-eligible AND role processes live AND no
        whole-card residency held): at the dispatch seam an eligible job is routed to ``register_disaggregated``
        (which pins the process the scheduler chose as its sampler) instead of being sent monolithic inference.
        ``is_disaggregation_class_eligible`` is the stable class predicate used by residency forecasting and
        VRAM charging, so a job that will run disaggregated is priced sampler-only regardless of transient lane
        state (a whole-card window pauses the lane without flipping the forecast to the monolithic footprint).

        ``pin_owner`` maps a process id to the job pinning it as its sampler, and ``sampling_peaks`` returns the
        in-flight sampling peaks; both are read-only, used only by the dispatch-stall classifier to name a head
        held behind a pinned sampler lane. They are optional so standalone tests need not wire the orchestrator.
        """
        self._is_disaggregatable_job = is_disaggregatable
        self._is_disaggregation_class_eligible = is_disaggregation_class_eligible
        self._register_disaggregated_job = register_disaggregated
        if pin_owner is not None:
            self._disaggregation_pin_owner = pin_owner
        if sampling_peaks is not None:
            self._disaggregation_sampling_peaks = sampling_peaks

    def _disaggregation_sibling_charge_mb(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        *,
        device_index: int | None,
    ) -> float:
        """The image lane's concurrent VAE-decode spike (MB) to charge against co-residency when disaggregating.

        Prefers the job's *bounded* tiled-decode activation (``predict_job_decode_spike_mb``): the lane
        decodes the previous job's latent while this one samples, so only that decode working set (not the
        lane's whole allocator-guard quota) is the concurrent commitment. Charging the full quota over-commits
        the card and denies a second sampler it can physically hold, collapsing the pipeline. Falls back to the
        full lane quota when the pinned hordelib does not yet expose the decode-spike figure (conservative:
        safe but not optimally packed).
        """
        decode_spike_mb = predict_job_decode_spike_mb(job, str(baseline) if baseline is not None else None)
        if decode_spike_mb is not None:
            return decode_spike_mb
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        return effective_post_process_vram_quota_mb(total_vram_mb)

    def latest_preload_admission(self) -> LatestPreloadAdmission | None:
        """Return the most recent preload-admission decision, for the supervisor snapshot."""
        return self._last_preload_admission

    def latest_host_memory_governance_snapshot(self) -> HostMemorySnapshot | None:
        """Return the latest host-memory governance input snapshot, or None before the first tick."""
        verdict = self._governor.last_ram_verdict
        if verdict is None:
            return None
        return self._build_host_memory_snapshot(verdict)

    def _record_churn(self, kind: ChurnKind) -> None:
        """Report one churn event to the observer if one is registered (no-op otherwise)."""
        if self._churn_observer is not None:
            self._churn_observer(kind)

    @property
    def _max_concurrent_inference_processes(self) -> int:
        """The live concurrent-inference cap (effective ``max_threads``), bounded by the ceiling."""
        return self._runtime_config.effective_max_threads

    def get_single_job_effective_megapixelsteps(self, job: ImageGenerateJobPopResponse) -> int:
        """Return the number of effective megapixelsteps for a single job."""
        return _get_single_job_effective_megapixelsteps(job)

    def _expected_sampling_seconds(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    ) -> float | None:
        """The performance model's expected sampling seconds for a job, or ``None`` when unavailable.

        Returns ``None`` when no model is wired, the baseline is unknown, or the job's signature has no
        seeded or calibrated rate yet (cold start), so the absence of an expectation is never an error.
        """
        if self._performance_model is None:
            return None
        signature = signature_from_job(job, str(baseline) if baseline is not None else None)
        if signature is None:
            return None
        return self._performance_model.expected_sampling_seconds(signature)

    def _diagnostic_mb_bucket(self, value: float | None) -> int | None:
        """Bucket memory telemetry so harmless measurement jitter does not spam diagnostics."""
        if value is None:
            return None
        return round(value / _SCHEDULER_DIAGNOSTIC_MB_BUCKET)

    def _scheduler_diagnostic_suppressed_count(
        self,
        name: str,
        state_key: tuple[object, ...],
    ) -> int | None:
        """Return suppressed-repeat count when a high-frequency diagnostic should be emitted.

        The first observation logs, a semantic state change logs immediately, and an unchanged observation
        logs periodically. ``None`` means "do not emit this time".
        """
        now = time.time()
        previous = self._scheduler_diagnostic_log_state.get(name)
        if previous is None:
            self._scheduler_diagnostic_log_state[name] = (state_key, now, 0)
            return 0

        previous_key, previous_emit, suppressed_count = previous
        if previous_key != state_key:
            self._scheduler_diagnostic_log_state[name] = (state_key, now, 0)
            return suppressed_count

        if (now - previous_emit) >= _SCHEDULER_DIAGNOSTIC_REPEAT_SECONDS:
            self._scheduler_diagnostic_log_state[name] = (state_key, now, 0)
            return suppressed_count

        self._scheduler_diagnostic_log_state[name] = (previous_key, previous_emit, suppressed_count + 1)
        return None

    def _suppressed_suffix(self, suppressed_count: int) -> str:
        """Return a compact suffix for diagnostics that skipped unchanged loop repeats."""
        if suppressed_count <= 0:
            return ""
        return f" (suppressed {suppressed_count} unchanged repeats)"

    def _log_stream_forecast(self, job: ImageGenerateJobPopResponse, forecast: StreamForecast) -> None:
        """Log the stream forecast when its decision or materially-relevant measurements change."""
        if not forecast.known:
            return

        job_id = str(job.id_) if job.id_ is not None else None
        state_key = (
            job.model,
            job_id,
            self._diagnostic_mb_bucket(forecast.weights_mb),
            self._diagnostic_mb_bucket(forecast.reserve_mb),
            self._diagnostic_mb_bucket(forecast.free_now_mb),
            self._diagnostic_mb_bucket(forecast.free_after_model_evict_mb),
            self._diagnostic_mb_bucket(forecast.free_if_alone_mb),
            self._process_map.num_loaded_inference_processes(),
            self._diagnostic_mb_bucket(self._per_process_overhead_mb()),
            self._diagnostic_mb_bucket(forecast.marginal_process_overhead_mb),
            forecast.fits_coresident,
            forecast.needs_exclusive_residency,
            forecast.requires_sibling_teardown,
            forecast.streams_unavoidably,
        )
        suppressed_count = self._scheduler_diagnostic_suppressed_count(f"stream_forecast:{job_id}", state_key)
        if suppressed_count is None:
            return

        marginal = self._overhead.marginal_breakdown(config_override_mb=self._config_overhead_override_mb())
        marginal_chosen = f"{marginal.chosen_mb:.0f}" if marginal.chosen_mb is not None else "?"
        marginal_probe = f"{marginal.probe_mb:.0f}" if marginal.probe_mb is not None else "?"
        marginal_floor = f"{marginal.idle_floor_mb:.0f}" if marginal.idle_floor_mb is not None else "?"
        logger.debug(
            f"Stream forecast for {job.model}: {forecast.reason()} "
            f"[free_now={forecast.free_now_mb}, after_model_evict={forecast.free_after_model_evict_mb}, "
            f"alone={forecast.free_if_alone_mb}, live_procs="
            f"{self._process_map.num_loaded_inference_processes()}, "
            f"overhead/proc={self._per_process_overhead_mb():.0f}MB, "
            f"marginal/ctx={marginal_chosen}MB(src={marginal.source},probe={marginal_probe},"
            f"idle_floor={marginal_floor})] -> "
            f"coresident={forecast.fits_coresident}, "
            f"needs_exclusive={forecast.needs_exclusive_residency}, "
            f"needs_teardown={forecast.requires_sibling_teardown}, "
            f"streams_unavoidably={forecast.streams_unavoidably}"
            f"{self._suppressed_suffix(suppressed_count)}",
        )

    def _log_next_models_for_vram_unload(
        self,
        next_n_models: list[str],
        *,
        under_pressure: bool,
        for_head_of_queue: bool,
    ) -> None:
        """Log the unload guard's next-model view without repeating it every reclaim attempt."""
        in_progress_models = tuple(sorted(str(job.model) for job in self._job_tracker.jobs_in_progress))
        state_key = (
            tuple(next_n_models),
            in_progress_models,
            under_pressure,
            for_head_of_queue,
            self._max_inference_processes,
        )
        suppressed_count = self._scheduler_diagnostic_suppressed_count("vram_unload_next_models", state_key)
        if suppressed_count is None:
            return
        logger.debug(f"Next n models: {next_n_models}{self._suppressed_suffix(suppressed_count)}")

    def _budget_active(self) -> bool:
        """Whether the measured VRAM/RAM budget gates preload/dispatch this cycle.

        Disabled by config (``enable_vram_budget=false``) restores the prior availability-only
        behavior. Both reserves are synced from the (live-reloadable) config here. Tests construct the
        scheduler with a mocked bridge_data whose attributes are Mocks rather than real values; in that
        case (or any partial config) fall back to the pre-budget behavior instead of acting on a
        non-numeric reserve.
        """
        bridge_data = self._runtime_config.bridge_data
        enabled = bridge_data.enable_vram_budget
        vram_reserve = config_number(bridge_data.vram_reserve_mb)
        ram_reserve = config_number(bridge_data.ram_reserve_mb)
        if not isinstance(enabled, bool) or vram_reserve is None or ram_reserve is None:
            return False
        if not enabled:
            return False
        self._vram_budget.set_reserve_mb(vram_reserve)
        self._ram_budget.set_reserve_mb(ram_reserve)
        return True

    def _is_model_locally_unservable(self, model: str | None, *, device_index: int | None = None) -> bool:
        """Return whether ``model`` is held back as locally unservable on a card (the shared breaker policy).

        Delegates to :func:`is_model_locally_unservable_for` so dispatch and the popper's model selection
        apply one identical policy: a model held back here is neither dispatched nor popped, so the worker
        stops dropping a model the device genuinely cannot run. ``device_index`` scopes the streak to the card
        the admit targets on a multi-GPU host; None is the single-GPU / worker-wide reading.
        """
        return is_model_locally_unservable_for(
            self._runtime_config.bridge_data,
            self._job_tracker,
            model,
            device_index=device_index,
        )

    def _log_overbudget_admit(self, job: ImageGenerateJobPopResponse) -> None:
        """Log a foreign-pressure physical-fit admit with the residency/measurement picture.

        Captures, in one greppable line, the model admitted outside the worker's own admission capacity
        because it physically fits the device-free read, whether it runs exclusively, its prior over-budget
        fault streak, and the per-slot residency plus device-wide free VRAM at admit time.
        """
        exclusive = self._job_tracker.is_admitted_exclusive(job)
        fault_count = self._job_tracker.get_model_overbudget_fault_count(job.model)
        logger.opt(ansi=True).warning(
            f"<fg #f0beff>VRAM budget cannot fit head-of-queue model {job.model} even after reclaiming all idle "
            f"VRAM/RAM, but it physically fits measured device-free VRAM; admitting it "
            f"({'exclusive' if exclusive else 'shared'}, prior_overbudget_faults={fault_count}) rather than "
            f"wedging the queue. {self._process_map.residency_snapshot()}</>",
        )

    def _mark_overbudget_admit(self, job: ImageGenerateJobPopResponse, forecast: StreamForecast | None) -> None:
        """Tag ``job`` as an over-budget physical-fit admit, opening the heavy-head load grace on first admit.

        Records the load-grace start the first time the job is admitted (so its multi-gigabyte load is not
        mistaken for a structural wedge; see :meth:`heavy_head_load_grace_active`). When over-budget
        exclusive mode is configured *and* the forecast shows the model's footprint dominates the card,
        also marks it exclusive so the scheduler suppresses concurrent pre-staging and dispatch for its
        duration, leaving the device un-contended while it completes.

        Exclusivity guards a heavy model against a concurrent sibling load pushing its weights into
        host-RAM streaming; that risk needs a footprint that dominates the device *on a card too small
        to host a sibling beside it* (see :attr:`StreamForecast.admit_requires_isolation`). A card-light
        model can reach this path purely through reserve arithmetic (free VRAM depressed by retained
        sibling contexts), and a card-dominating model on a roomy card co-resides safely; isolating
        either caps a multi-thread card at one job for the admit's whole lifetime while blocking every
        other preload. An unsized or missing forecast keeps the conservative isolation.
        """
        if not self._job_tracker.is_admitted_over_budget(job):
            self._heavy_head_admitted_at = time.time()
        self._job_tracker.mark_admitted_over_budget(job)
        if self._runtime_config.bridge_data.overbudget_exclusive_mode and (
            forecast is None or forecast.admit_requires_isolation
        ):
            self._job_tracker.mark_admitted_exclusive(job)

    def set_measured_per_process_overhead_mb(self, overhead_mb: int | float) -> None:
        """Record the startup-measured per-process VRAM overhead (MB) for the streaming forecast."""
        self._overhead.set_per_process_overhead_mb(overhead_mb)

    def set_measured_marginal_overhead_mb(self, marginal_mb: int | float) -> None:
        """Record the startup-measured *marginal* per-additional-context VRAM cost (MB) from the probe.

        Hard data (the probe's second-context delta) available from the first scheduling tick, so it fixes the
        startup-window over-count without waiting for siblings to reach idle. 0 (or unmeasurable) leaves the
        scheduler on its idle-residency fallback.
        """
        self._overhead.set_marginal_overhead_mb(marginal_mb)

    def _config_overhead_override_mb(self) -> float | None:
        """Return the coerced ``vram_per_process_overhead_mb`` config override, or None when unset/non-numeric.

        Tolerant of partially-mocked config: a non-numeric reading coerces to None so the overhead model falls
        back to its measured figures.
        """
        return config_number(self._runtime_config.bridge_data.vram_per_process_overhead_mb)

    def _per_process_overhead_mb(self) -> float:
        """Return the per-process VRAM overhead (MB) to assume: configured override, else measured, else 0.

        An explicit ``vram_per_process_overhead_mb`` config value (> 0) wins so operators can tune; otherwise
        the startup-measured figure is used. This is the *first/sole* context cost (it includes the one-time
        CUDA runtime allocation), used to size ``free_if_alone``; the per-additional-context cost is
        :meth:`_marginal_process_overhead_mb`.
        """
        return self._overhead.per_process_mb(config_override_mb=self._config_overhead_override_mb())

    def _bare_context_total_mb(
        self,
        *,
        device_used_mb: float,
        baseline_mb: float,
        device_index: int | None,
    ) -> tuple[float, int] | None:
        """Decompose a truthful device-used reading into the tenants' bare-context total and their count.

        The worker-attributable bare-context total is truthful device-used minus the shared device baseline
        minus every committed-ledger tenant's byte-exact allocator reservation: what remains is only the
        context costs (the one-time CUDA runtime plus one context each), the exact quantity the overhead
        model's marginal derivation is defined over. Charging anything else (the baseline, resident weights,
        another tenant's reservation) into that residual multiplies it across the process count and prices
        the card into a phantom over-commit. Keyed on the committed ledger's tenant set so the marginal
        derivation and the ledger can never disagree about who holds a context. Returns None when the card
        has no ledger tenants; the residual may be negative (a baseline estimate that absorbed context cost),
        which the capture path skips and the invalidation path clamps toward zero.
        """
        reserved_sum_mb = 0.0
        tenants = self._process_map.committed_ledger_processes(device_index)
        if not tenants:
            return None
        for process_info in tenants:
            reserved_sum_mb += (process_info.process_reserved_mb or 0.0) + (process_info.process_aimdo_mb or 0.0)
        context_total_mb = device_used_mb - baseline_mb - reserved_sum_mb
        return context_total_mb, len(tenants)

    def capture_idle_context_residency(
        self,
        *,
        device_used_mb: float,
        baseline_mb: float,
        device_index: int | None = None,
    ) -> None:
        """Record the tenants' bare-context total when every inference process is idle with no model resident.

        That measurement is the true combined cost of the GPU tenants' contexts (the one-time CUDA runtime
        plus one context each), which the forecast needs to size ``free_after_model_evict`` without
        multiplying the one-time cost by the process count. Inspects the process map for the clean
        precondition (every live inference process up, idle, and holding no model, and no GPU tenant busy)
        and feeds a confirmed reading to the overhead model, which keeps the relevant extremes.

        Fed by the parent's attribution tick, which owns the truthful NVML device-used reading and the
        reconciled shared-baseline estimate: per-child VRAM views are per-process artefacts under WDDM and
        must never be decomposed as if they were device truth.

        Args:
            device_used_mb: Truthful device-wide used VRAM (MB) from the parent-side NVML read.
            baseline_mb: The reconciler's shared-device baseline estimate (MB) for the card.
            device_index: The card the reading belongs to; None for the single-GPU/worker-wide case.
        """
        inference_count = 0
        for process_info in self._process_map.values():
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED):
                continue
            # Any busy GPU tenant (an in-flight safety evaluation, a post-processing form) is transient VRAM
            # the residual would misread as context cost, so the clean window requires full quiescence.
            if process_info.is_process_busy():
                return
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            inference_count += 1
            # A clean baseline requires every live inference process up, idle, and holding no model: any model
            # resident (even one offloaded to RAM but still tracked) means the reading includes weight VRAM.
            if (
                process_info.last_process_state != HordeProcessState.WAITING_FOR_JOB
                or process_info.loaded_horde_model_name is not None
            ):
                return
        if inference_count < 1:
            return
        decomposed = self._bare_context_total_mb(
            device_used_mb=device_used_mb,
            baseline_mb=baseline_mb,
            device_index=device_index,
        )
        if decomposed is None:
            return
        context_total_mb, context_count = decomposed
        if context_total_mb <= 0:
            # The baseline estimate absorbed the context cost (it was captured with tenants already up): there
            # is no attributable residual to latch, and the marginal correctly falls back to probe/seed.
            return
        self._overhead.observe_idle_residency(context_total_mb=context_total_mb, context_count=context_count)

    def invalidate_idle_context_floor(
        self,
        *,
        device_used_mb: float,
        baseline_mb: float,
        device_index: int | None = None,
    ) -> None:
        """Lower a latched effective idle floor once the device proves it was not a sustained reading.

        Complements :meth:`capture_idle_context_residency`. The capture keeps the worst clean all-idle
        reading; a transient spike would otherwise pin the per-context marginal high for the whole session
        and route ordinary models into teardown/exclusive admits. Unlike the capture this does not require
        the clean precondition: resident weights are netted out via the byte-exact reservations, and any
        residual transient VRAM only makes the correction conservative, so a bare-context reading below the
        latched floor (with at least as many tenants live) is unambiguous proof it was too high.

        Args:
            device_used_mb: Truthful device-wide used VRAM (MB) from the parent-side NVML read.
            baseline_mb: The reconciler's shared-device baseline estimate (MB) for the card.
            device_index: The card the reading belongs to; None for the single-GPU/worker-wide case.
        """
        decomposed = self._bare_context_total_mb(
            device_used_mb=device_used_mb,
            baseline_mb=baseline_mb,
            device_index=device_index,
        )
        if decomposed is None:
            return
        context_total_mb, context_count = decomposed
        self._overhead.observe_device_residency(
            context_total_mb=max(0.0, context_total_mb),
            context_count=context_count,
        )

    def _marginal_process_overhead_mb(self) -> float | None:
        """Return the per-additional-context VRAM cost (MB), or None to fall back to the first-context overhead.

        Prefers the probe's directly-measured second-context delta (hard data, available from the first tick,
        so it also covers the startup window where siblings have not yet reached idle). Failing that (the probe
        could not measure it on this backend), derives it from the measured all-contexts idle residency.
        Returns None when neither is available, in which case the forecast conservatively reuses the
        first-context overhead per additional context.
        """
        return self._overhead.marginal_mb(config_override_mb=self._config_overhead_override_mb())

    def resolved_context_constant_mb(self) -> float:
        """Return the per-process CUDA-context VRAM charge (MB) for the committed-VRAM attribution ledger.

        The measured per-additional-context marginal when the overhead model has one, else the platform seed
        (243 MB Windows / 144 MB Linux / the generic fallback), resolved by
        :func:`platform_context_constant_mb`. Consumed by the observational committed-VRAM ledger and drift
        reconciliation, not by admission.
        """
        return platform_context_constant_mb(self._marginal_process_overhead_mb())

    def _whole_card_residency_enabled(self) -> bool:
        """Whether preventative whole-card exclusive residency is on (config, tolerant of mocked config)."""
        enabled = self._runtime_config.bridge_data.whole_card_exclusive_residency
        return enabled is True

    def _whole_card_warranted(self, forecast: StreamForecast) -> bool:
        """Whether a teardown demand is trustworthy enough to engage the whole-card residency machinery.

        Reserving the whole card has a large blast radius: it stops sibling processes (which may be serving
        other queued heads), moves safety off-GPU, and holds the device through a cooldown, so it must only
        fire on a demand that is not a measurement artifact. Two signals qualify it:

        - a genuinely card-demanding model (its persistent footprint dominates the device, or its baseline is
          declared whole-card on intent): the teardown is warranted regardless of how contexts are counted; or
        - a per-additional-context cost that was actually *measured* (the probe's second-context delta or a
          derived idle-floor): the contention the demand rests on is real, not an over-count.

        When neither holds (a card-light model on a host where the marginal context cost could not be
        measured), the per-context overhead falls back to the full first-context cost, which charges the
        one-time CUDA runtime against every context and can collapse the structural floor below a model that
        physically co-resides with room to spare. Engaging a whole-card residency off that phantom reserves the
        card for a model that never needed it (and, held through the cooldown, can then starve a later head of a
        different model). So the caller falls through to the ordinary model-eviction path instead, whose
        admission still gates on real free VRAM, rather than reserving the device on an unmeasured guess.
        """
        if forecast.is_card_demanding:
            return True
        return self._marginal_process_overhead_mb() is not None

    def _log_whole_card_declined(self, job: ImageGenerateJobPopResponse, forecast: StreamForecast) -> None:
        """Record (once per model) that a whole-card teardown demand was declined as untrustworthy.

        Names why a model that the budget/forecast wanted to give the whole card was instead served by
        ordinary eviction: its footprint does not dominate the device and the per-additional-context cost was
        not measured, so the demand rests on the fallback that charges the one-time runtime cost against every
        context. Surfaces the numbers behind that call (the model's weight share of the card and whether the
        marginal was measured) so a teardown that does *not* happen is as visible in the logs as one that does.
        """
        if self._whole_card_declined_notified.get(job.model or "", False):
            return
        self._whole_card_declined_notified[job.model or ""] = True
        weights = forecast.weights_mb
        total = forecast.total_vram_mb
        share = f"{(weights / total) * 100:.0f}%" if weights is not None and total else "unknown"
        logger.opt(ansi=True).info(
            f"<fg #7b7d7d>Declined a whole-card residency for {job.model}: its weights (~{weights or 0:.0f}MB, "
            f"{share} of the {total or 0:.0f}MB card) do not dominate the device and the per-context overhead is "
            f"unmeasured (using the conservative first-context fallback), so a teardown demand cannot be trusted. "
            f"Serving it co-resident via model eviction instead of reserving the card.</>",
        )

    def _residency_state(self, device_index: int | None) -> WholeCardResidency:
        """Return the (lazily-created) whole-card residency state for ``device_index``.

        ``None`` is the single-GPU / worker-wide key, so a single-GPU host keeps exactly one residency state
        and behaves as the pre-multi-GPU scalar fields did.
        """
        return self._whole_card_ledger.state_for(device_index)

    def _held_residencies(self) -> list[tuple[int | None, WholeCardResidency]]:
        """Return ``(device_index, state)`` for every card currently holding a whole-card residency.

        A residency is "held" while its model is set. Used by the per-cycle convergence/restore passes and the
        supervisor-facing grace checks, which must consider every card's residency, not just one.
        """
        return self._whole_card_ledger.held()

    # The worker-wide (single-GPU) whole-card residency is the entry under the ``None`` key. These properties
    # expose its fields under their historical scalar names so single-GPU callers and tests read/write the
    # worker-wide residency exactly as before the per-card ``_whole_card_residencies`` map existed. The
    # multi-GPU admission path keys residency by real device index and does not go through these.
    @property
    def _sibling_teardown_for_model(self) -> str | None:
        """The model holding the worker-wide whole-card residency (the ``None``-keyed entry)."""
        return self._residency_state(None).model

    @_sibling_teardown_for_model.setter
    def _sibling_teardown_for_model(self, value: str | None) -> None:
        self._residency_state(None).model = value

    @property
    def _whole_card_forecast(self) -> StreamForecast | None:
        """The forecast that established the worker-wide whole-card residency."""
        return self._residency_state(None).forecast

    @_whole_card_forecast.setter
    def _whole_card_forecast(self, value: StreamForecast | None) -> None:
        self._residency_state(None).forecast = value

    @property
    def _whole_card_established_at(self) -> float:
        """When the worker-wide whole-card residency was established (0.0 when none)."""
        return self._residency_state(None).established_at

    @_whole_card_established_at.setter
    def _whole_card_established_at(self, value: float) -> None:
        self._residency_state(None).established_at = value

    @property
    def _whole_card_cooldown_until(self) -> float:
        """Cooldown deadline of the worker-wide whole-card residency."""
        return self._residency_state(None).cooldown_until

    @_whole_card_cooldown_until.setter
    def _whole_card_cooldown_until(self, value: float) -> None:
        self._residency_state(None).cooldown_until = value

    @property
    def _whole_card_restore_at(self) -> float:
        """When the worker-wide whole-card residency was last restored (0.0 when none)."""
        return self._residency_state(None).restore_at

    @_whole_card_restore_at.setter
    def _whole_card_restore_at(self, value: float) -> None:
        self._residency_state(None).restore_at = value

    def _max_coresident_for_peak_mb(
        self,
        peak_mb: float,
        reserve_mb: float,
        *,
        device_index: int | None = None,
    ) -> int | None:
        """Largest live inference-process count that still fits ``peak_mb`` plus ``reserve_mb``.

        Sizes the context-reduction depth from the *same* conservative figure the VRAM verdict rejects on
        (the burden estimate), not the forecast's resident-weight estimate. The two estimators differ: the
        forecast judges co-residence from the resident weight footprint while the admission verdict uses the
        fuller per-job burden peak, so a moderate head can read co-resident in the forecast yet be rejected
        by the verdict every tick, the gap that routes it into the evict-all admit. Reasoning the teardown
        depth from the verdict's own peak makes the structural remedy fire exactly when admission would
        otherwise reject and thrash. The loader's first context costs the full one-time overhead; each
        additional co-resident context costs only the marginal. Returns None when it cannot be sized.

        Args:
            peak_mb: The job's predicted peak VRAM (MB) that must fit alongside the live contexts.
            reserve_mb: The transient-spike reserve (MB) required on top of the peak.
            device_index: When given, size against that one card's total VRAM (the per-card context-reduction
                depth on a multi-GPU host); when None, the worker-wide total.
        """
        return max_coresident_for_peak(
            total_vram_mb=self._process_map.get_reported_total_vram_mb(device_index=device_index),
            per_process_overhead_mb=self._per_process_overhead_mb(),
            marginal_overhead_mb=self._marginal_process_overhead_mb(),
            peak_mb=peak_mb,
            reserve_mb=reserve_mb,
        )

    def _forecast_streaming(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        *,
        device_index: int | None = None,
    ) -> StreamForecast:
        """Return the weight-streaming forecast for loading ``job``'s model given the device's measured state.

        Combines the measured free VRAM and total VRAM (from the children's reports), the configured reserve
        floor, and the per-process overhead so the scheduler can tell a model that only streams because of
        co-resident siblings (curable by exclusive residency) from one that streams even alone.

        Args:
            job: The job whose model load is being forecast.
            baseline: The model's known image-generation baseline (or its string form), or None when unknown.
            device_index: When given, forecast against that one card's measured free/total VRAM and its live
                inference- and safety-context counts (the per-card forecast on a multi-GPU host); when None,
                the worker-wide reading. The per-context overhead is a CUDA-runtime/arch constant and stays
                worker-wide either way (per-card overhead probing is a hordelib-side follow-up).
        """
        configured_floor = self._runtime_config.bridge_data.vram_reserve_mb
        floor_mb = (
            float(configured_floor)
            if isinstance(configured_floor, (int, float))
            and not isinstance(
                configured_floor,
                bool,
            )
            else 0.0
        )
        # The structural floor (free once every process's context materialises) is keyed off the *live*
        # inference-process count, not the configured ceiling. Using the live count is what lets the
        # teardown converge: once idle siblings are stopped for a whole-card job the forecast re-evaluates
        # against the reduced contexts and admits the model, instead of perpetually demanding more teardown
        # against a ceiling that is no longer running. Processes are staged up front (or once a model is on
        # disk), so by the time a job is scheduled the live count already reflects the real contention.
        num_processes = self._process_map.num_loaded_inference_processes(device_index=device_index)
        # The safety process holds its own CUDA context on the card when safety_on_gpu is set; that VRAM is
        # not reclaimable by stopping idle inference siblings, so the forecast must count it against the
        # achievable-free figures (sole residency for a heavy model then implies moving safety off-GPU too).
        # Count the safety context only when safety is *actually* on the GPU right now: once a whole-card
        # job has paused it off-GPU, its context is freed, so continuing to charge it would keep the
        # structural floor (free_after_model_evict) below the model's demand forever and the whole-card
        # branch would defer the model every tick without ever loading it. The safety process is pinned to a
        # single card, so on a per-card forecast it is charged only against the card it actually sits on.
        safety_on_gpu = self._runtime_config.bridge_data.safety_on_gpu and (
            not self._process_lifecycle.is_safety_gpu_paused
        )
        num_safety_contexts = self._process_map.num_safety_processes(device_index=device_index) if safety_on_gpu else 0
        # The dedicated post-processing lane holds a CUDA context (and its resident post-processing models)
        # on the card it is pinned to; like the safety context, that is a real device-wide commitment idle
        # inference siblings cannot reclaim, so it is charged as an extra resident context here. Charge it only
        # while the lane is actually on the card: once a whole-card job has stopped the lane off-GPU its context
        # is freed, so continuing to charge it would keep the structural floor below the model's demand and
        # defer the head forever (the same reasoning as the paused safety context above).
        num_post_process_contexts = (
            0
            if self._process_lifecycle.is_post_process_gpu_paused
            else self._process_map.num_post_process_processes(device_index=device_index)
        )
        # The EXTRA_LARGE tier (extra-large baselines plus the named VRAM-heavy checkpoints) is the single
        # source of truth for "wants the whole card and never shares". Feed it to the forecast so a baseline
        # whose conservative weight seed happens to fit co-resident still claims sole residency on intent,
        # rather than co-residing and thrashing as Z-Image did.
        wants_whole_card = self._model_size_tier(job.model) >= _ModelSizeTier.EXTRA_LARGE
        # A disaggregated job's sampler holds only the UNet, so its forecast charges the sampler-only figure
        # (keeping two samplers co-resident where the whole-job charge collapses them), and the image lane's
        # concurrent decode spike is charged as a sibling context. Class-eligibility (not the liveness-coupled
        # dispatch predicate) is used so a job that will run disaggregated is charged sampler-only even during a
        # whole-card window when the lane is transiently paused.
        disaggregated = self._is_disaggregation_class_eligible(job)
        return forecast_weight_streaming(
            job,
            str(baseline) if baseline is not None else None,
            free_now_mb=self._measured_free_vram_mb(device_index=device_index),
            total_vram_mb=self._process_map.get_reported_total_vram_mb(device_index=device_index),
            per_process_overhead_mb=self._per_process_overhead_mb(),
            num_inference_processes=num_processes,
            configured_reserve_floor_mb=floor_mb,
            num_extra_resident_contexts=num_safety_contexts + num_post_process_contexts,
            committed_reserve_mb=self._committed_vram_reserve_mb(device_index=device_index),
            marginal_process_overhead_mb=self._marginal_process_overhead_mb(),
            wants_whole_card=wants_whole_card,
            disaggregated=disaggregated,
            disaggregation_sibling_charge_mb=(
                self._disaggregation_sibling_charge_mb(job, baseline, device_index=device_index)
                if disaggregated
                else 0.0
            ),
        )

    def _residency_should_pause_safety(self, device_index: int | None) -> bool:
        """Whether a whole-card residency on this card should also move the single safety process off-GPU.

        Requires safety configured-and-on-GPU (:meth:`_whole_card_safety_off_gpu_enabled`) and that this is
        the card the one safety process is pinned to (:meth:`_safety_gpu_card`, headroom-chosen, not a fixed
        index). A residency on a non-safety card never disturbs safety. The worker-wide key (``None``,
        single-GPU) always qualifies.
        """
        if not self._whole_card_safety_off_gpu_enabled():
            return False
        if device_index is None or not self._card_runtimes:
            return True
        return device_index == self._safety_gpu_card()

    def _has_safety_backlog(self) -> bool:
        """Return whether safety has work that should not be interrupted by residency churn."""
        return self._safety_backlog_depth() > 0

    def _safety_backlog_depth(self) -> int:
        """Return the total safety backlog: pending checks plus checks awaiting a verdict."""
        return len(self._job_tracker.jobs_pending_safety_check) + len(self._job_tracker.jobs_being_safety_checked)

    def _has_priority_safety_backlog(self) -> bool:
        """Return whether the safety backlog is deep enough to prioritize GPU restoration."""
        return self._safety_backlog_depth() > _SAFETY_BACKLOG_PRIORITY_DEPTH

    def _pause_safety_for_residency_if_idle(self, device_index: int | None) -> bool:
        """Pause safety for whole-card residency only when no safety job is pending or active."""
        if not self._residency_should_pause_safety(device_index):
            return False
        if self._process_lifecycle.is_safety_gpu_paused:
            return False
        if self._has_safety_backlog():
            return False
        return self._process_lifecycle.pause_safety_on_gpu()

    def _arbiter_admits_safety_gpu_load(self, device_index: int | None) -> bool:
        """Whether the safety process may (re)load onto the GPU now, the VRAM arbiter deciding the memory question.

        Charges :data:`_SAFETY_GPU_LOAD_CHARGE_MB` against the card's measured admission floor as a
        :attr:`VramRequestKind.SAFETY_LOAD`: a FITS verdict admits, a DEFER or DENY keeps safety off-GPU this
        cycle so the load re-asks. An unwired or cold arbiter admits, matching the
        every-gate-admits-on-missing-telemetry contract. No actuations run here (reclaim is single-owner, driven
        only by the preload path).
        """
        arbiter = self._vram_arbiter
        if arbiter is None or not arbiter.has_cycle:
            return True
        verdict = arbiter.evaluate(
            VramRequest(
                kind=VramRequestKind.SAFETY_LOAD,
                job_label="safety_load",
                baseline=None,
                device_index=device_index,
                candidate_delta_mb=_SAFETY_GPU_LOAD_CHARGE_MB,
            ),
        )
        return verdict.admits

    def _restore_deferred_safety_gpu_load(self) -> None:
        """Bring a safety-load-gate-deferred safety process back on-GPU once the card has room to hold it.

        The safety-load gate can keep safety off-GPU when the card is momentarily over-committed as a whole-card
        residency drains; this per-tick reconciler re-asks the arbiter so a deferred safety load is not stranded
        off-GPU for the rest of the session. It never fights the residency machinery: it acts only when no held
        whole-card residency still requires safety off its card. It avoids churning a shallow safety backlog,
        but a backlog deeper than :data:`_SAFETY_BACKLOG_PRIORITY_DEPTH` is urgent enough to let a paused
        safety process return to GPU service when the arbiter admits the load. Only the recurring
        residency-drain restore is gated this way; the initial cold-start safety load onto the GPU (at worker
        bring-up, before any heavy residency pressure) is not gated and always proceeds.
        """
        if not self._whole_card_safety_off_gpu_enabled():
            return
        if not self._process_lifecycle.is_safety_gpu_paused:
            return
        if self._has_safety_backlog() and not self._has_priority_safety_backlog():
            return
        # The runtime safety-placement policy is the other owner of the safety process's on/off-GPU state; while
        # it holds safety off (the charge does not fit beside the largest sampling peak) this residency-drain
        # restore must not fight it back on-GPU. The placement reconcile performs its own restore once the card
        # proves durable headroom.
        if self._safety_placement_wants_off:
            return
        if any(self._residency_should_pause_safety(device_index) for device_index, _ in self._held_residencies()):
            return
        safety_card = self._safety_gpu_card()
        if self._arbiter_admits_safety_gpu_load(safety_card):
            self._process_lifecycle.restore_safety_on_gpu()

    def _choose_safety_gpu_card(self) -> int | None:
        """Return the driven card safety should be placed on: the one with the most verified headroom.

        The single placement identity, consumed both at spawn (pushed to the lifecycle manager, which pins the
        safety process there) and when the runtime placement policy re-promotes safety onto the GPU, so the
        two never disagree about which card safety lands on. Headroom per card is its truthful measured
        device-free VRAM when reported (that figure already nets out whatever is resident and sampling on the
        card right now); absent a measured reading it falls back to the card total less the largest active
        sampling peak (the modeled expected peak). The card with the greatest headroom wins, ties resolving to
        the lowest index so the choice is stable. On a single-GPU host this is the one card, and with no
        headroom evidence at all it is the lowest-index card, both byte-identical to the historical fixed pin.
        """
        if not self._card_runtimes:
            return None
        modeled_peak_mb = self._largest_active_sampling_peak_mb() or 0.0
        best_index: int | None = None
        best_headroom_mb = float("-inf")
        for device_index in sorted(self._card_runtimes):
            measured_free_mb = self._measured_free_vram_mb(device_index=device_index)
            if measured_free_mb is not None:
                headroom_mb = measured_free_mb
            else:
                total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
                if total_vram_mb is None:
                    continue
                headroom_mb = total_vram_mb - modeled_peak_mb
            if headroom_mb > best_headroom_mb:
                best_headroom_mb = headroom_mb
                best_index = device_index
        return best_index if best_index is not None else min(self._card_runtimes)

    def _safety_gpu_card(self) -> int | None:
        """Return the card safety currently occupies, or the card it would be placed on when off-GPU.

        When safety is on-GPU this is the card it is physically pinned to (from the lifecycle manager), so
        residency and fit checks reason about the real card. When safety is off-GPU it is the headroom-aware
        candidate (:meth:`_choose_safety_gpu_card`) it would land on at the next re-promotion. None on a
        host with no driven cards.
        """
        if not self._card_runtimes:
            return None
        current_card = self._process_lifecycle.safety_gpu_card_index()
        if current_card is not None and current_card in self._card_runtimes:
            return current_card
        return self._choose_safety_gpu_card()

    def _safety_restore_headroom_fits(self, device_index: int | None) -> bool:
        """Whether the chosen card's *measured* free VRAM now holds safety's context plus a proportional buffer.

        The demotion side prices a modeled worst-case peak (device total less the largest learned sampling
        peak), which under sustained load is always populated, so a modeled restore predicate is unsatisfiable
        while the card keeps sampling: that is the defect this replaces. The restore side instead reads the
        card's truthful measured device-free between allocation peaks. The governor's NVML-derived figure
        already nets out whatever is resident and sampling right now, so a card that genuinely has room to
        reabsorb safety's context reports it directly, and a card that stays busy under load never accrues the
        restore streak (CPU placement remaining the correct steady state, with pop backpressure carrying the
        load). Additionally requires the device-free governor to be HEALTHY on the card, so a card hovering at
        the paging cliff never readmits safety. Missing telemetry (no measured free) does not restore: the
        policy promotes only on positive, measured evidence.
        """
        measured_free_mb = self._measured_free_vram_mb(device_index=device_index)
        if measured_free_mb is None:
            return False
        if self.governor_state(device_index) is not GovernorState.HEALTHY:
            return False
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        buffer_mb = admission_noise_buffer_mb(total_vram_mb)
        return (measured_free_mb - _SAFETY_GPU_LOAD_CHARGE_MB - buffer_mb) >= 0.0

    def _runtime_safety_placement_enabled(self) -> bool:
        """Whether the runtime safety-placement policy may act (safety configured on-GPU on a real device).

        The policy can only ever degrade the operator's placement (GPU to CPU), never promote it, so it is inert
        unless ``safety_on_gpu`` grants the maximum permission. On a CPU-only install safety is always off-GPU
        already, so there is nothing to place.
        """
        return bool(self._runtime_config.bridge_data.safety_on_gpu) and not is_cpu_only_install()

    def _largest_active_sampling_peak_mb(self) -> float | None:
        """The largest learned solo sampling peak (MB) among jobs in progress or queued for inference.

        Each job's static sampling-peak seed is raised by any learned SAMPLE-stage watermark for its footprint
        before the maximum is taken, so the policy prices the heaviest activation peak the device is committed
        to from measured high-waters, not a seed the hardware has already overshot. Returns None when no job can
        be priced (nothing sampling), in which case the safety charge trivially fits.
        """
        peaks: list[float] = []
        seen: set[int] = set()
        for job in (*self._job_tracker.jobs_in_progress, *self._job_tracker.jobs_pending_inference):
            if id(job) in seen:
                continue
            seen.add(id(job))
            if job.model is None:
                continue
            baseline = self._model_metadata.get_baseline(job.model)
            static_peak_mb = predict_job_sampling_vram_mb(job, baseline)
            if static_peak_mb is None:
                continue
            peaks.append(
                self._learned_sampling_peak_mb(
                    job,
                    baseline,
                    static_seed_mb=static_peak_mb,
                    stage=FootprintStage.SAMPLE,
                ),
            )
        if not peaks:
            return None
        return max(peaks)

    def _safety_fits_beside_largest_sampling_peak(
        self,
        device_index: int | None,
        *,
        require_margin: bool,
    ) -> bool:
        """Whether the safety charge fits on the card beside the largest active sampling peak, as arithmetic.

        Structural fit over (device total, largest learned sampling peak, proportional noise buffer, the static
        safety charge): ``total - peak - noise - safety_charge >= 0``. No constant is tuned to a card size; the
        noise buffer scales with the device total. With ``require_margin`` an extra proportional buffer must
        also clear, so the restore side of the hysteresis demands durable headroom rather than a bare fit. When
        the device total is unknown or nothing is sampling, the charge trivially fits (the policy never forces
        safety off on missing telemetry, matching the every-gate-admits-on-missing-measurement contract).
        """
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        if total_vram_mb is None or total_vram_mb <= 0:
            return True
        peak_mb = self._largest_active_sampling_peak_mb()
        if peak_mb is None:
            return True
        noise_mb = admission_noise_buffer_mb(total_vram_mb)
        margin_mb = admission_noise_buffer_mb(total_vram_mb) if require_margin else 0.0
        return (total_vram_mb - peak_mb - noise_mb - _SAFETY_GPU_LOAD_CHARGE_MB - margin_mb) >= 0.0

    def _reconcile_runtime_safety_placement(self) -> None:
        """Keep safety off the GPU while its charge cannot fit, and re-promote it once a card proves room.

        A scheduler-owned per-cycle policy that generalises the whole-card safety-off lever to the ordinary
        case: on a card too tight to hold the safety context beside the heaviest sampling activation it is
        committed to, safety cycles to a CPU-only process so its CUDA context stops competing for the card. The
        operator's ``safety_on_gpu`` remains the maximum permission (False leaves safety off forever); this
        policy can only degrade GPU to CPU and back, never beyond the operator's grant.

        The two sides read different signals so the re-promotion is satisfiable under sustained load. Demotion
        prices a *modeled* worst case: the charge must fail to fit beside the largest learned sampling peak
        (:meth:`_safety_fits_beside_largest_sampling_peak`), a predictive eviction that acts before the card
        reaches the paging cliff. Re-promotion instead reads the chosen card's *measured* device-free between
        allocation peaks (:meth:`_safety_restore_headroom_fits`): the modeled peak is always populated while
        jobs flow, so a modeled restore predicate could never be satisfied under load, whereas the measured
        free rises whenever the card genuinely has room. On a box where no card can host safety beside its
        sampler the measured streak never accrues and CPU placement is the correct steady state, with pop
        backpressure carrying the load.

        Hysteresis (:data:`_SAFETY_PLACEMENT_PAUSE_STREAK` / :data:`_SAFETY_PLACEMENT_RESTORE_STREAK`) guards
        against flapping: the off-latch turns on only after several consecutive modeled-non-fit cycles and off
        only after a longer run of measured-headroom cycles, with a deadband (modeled fit but measured room not
        yet proven) that advances neither streak. The asymmetric streaks double as a demote-again cooldown: a
        promotion resets the miss streak, so at least :data:`_SAFETY_PLACEMENT_PAUSE_STREAK` fresh non-fit
        cycles must pass before safety can be evicted again. Demotion actuation is skipped while a safety check
        is pending or active (no mid-backlog churn). If safety is already off-GPU and the backlog grows beyond
        :data:`_SAFETY_BACKLOG_PRIORITY_DEPTH`, restore actuation is allowed once measured headroom satisfies
        the normal restore hysteresis, because leaving a deep backlog on CPU safety is worse than preserving
        placement inertia. The restore is still withheld while a whole-card residency needs safety off its card
        and while the device-free governor holds growth, so this policy fights neither the residency machinery
        nor the cliff brake. The card safety is placed on is the headroom-aware choice
        (:meth:`_choose_safety_gpu_card`), pushed to the lifecycle manager so spawn and re-promotion agree.
        """
        # Push the headroom-aware placement choice to the lifecycle manager every cycle so any safety
        # (re)spawn (this policy's re-promotion, a residency restore, or a crash rebuild) pins to the current
        # best card. Single-GPU keeps the historical fixed pin (None), so its spawn path is byte-identical.
        if len(self._card_runtimes) > 1:
            self._process_lifecycle.set_desired_safety_card(self._choose_safety_gpu_card())

        if not self._runtime_safety_placement_enabled():
            self._safety_placement_miss_streak = 0
            self._safety_placement_fit_streak = 0
            self._safety_placement_wants_off = False
            return
        safety_backlog_depth = self._safety_backlog_depth()
        if safety_backlog_depth > 0 and not self._process_lifecycle.is_safety_gpu_paused:
            self._safety_placement_miss_streak = 0
            return
        if 0 < safety_backlog_depth <= _SAFETY_BACKLOG_PRIORITY_DEPTH:
            return

        safety_card = self._safety_gpu_card()
        modeled_fits = self._safety_fits_beside_largest_sampling_peak(safety_card, require_margin=False)
        measured_headroom_fits = self._safety_restore_headroom_fits(safety_card)
        if measured_headroom_fits:
            self._safety_placement_fit_streak += 1
            self._safety_placement_miss_streak = 0
        elif not modeled_fits:
            self._safety_placement_miss_streak += 1
            self._safety_placement_fit_streak = 0
        else:
            # Deadband: the modeled charge fits beside the peak but the card's measured free has not yet proven
            # durable room to reabsorb safety. Advance neither streak so a card on the boundary neither evicts
            # nor readmits safety.
            return

        if not self._safety_placement_wants_off:
            if self._safety_placement_miss_streak >= _SAFETY_PLACEMENT_PAUSE_STREAK:
                self._safety_placement_wants_off = True
        elif self._safety_placement_fit_streak >= _SAFETY_PLACEMENT_RESTORE_STREAK:
            self._safety_placement_wants_off = False

        if self._safety_placement_wants_off:
            if not self._process_lifecycle.is_safety_gpu_paused:
                peak_mb = self._largest_active_sampling_peak_mb()
                total_mb = self._process_map.get_reported_total_vram_mb(device_index=safety_card)
                if self._process_lifecycle.pause_safety_on_gpu():
                    self._safety_placement_demotions += 1
                    logger.info(
                        f"Runtime safety placement: moving safety off-GPU. Its "
                        f"~{_SAFETY_GPU_LOAD_CHARGE_MB / 1024:.1f}GB context does not fit beside the largest "
                        f"active sampling peak (~{(peak_mb or 0.0) / 1024:.1f}GB) on a "
                        f"~{(total_mb or 0.0) / 1024:.0f}GB card after "
                        f"{self._safety_placement_miss_streak} consecutive cycles.",
                    )
            return

        # The policy no longer wants safety off; restore it unless a whole-card residency still needs it off
        # that card, and only when the arbiter agrees the card can hold the load now.
        if not self._process_lifecycle.is_safety_gpu_paused:
            return
        if any(self._residency_should_pause_safety(device_index) for device_index, _ in self._held_residencies()):
            return
        # A safety GPU restore grows the card's committed footprint; withhold it while the device-free governor
        # holds growth (device-level free below the soft floor), the same brake the preload path honors.
        if self.is_vram_growth_held(safety_card):
            return
        if self._arbiter_admits_safety_gpu_load(safety_card) and self._process_lifecycle.restore_safety_on_gpu():
            self._safety_placement_promotions += 1
            logger.info(
                f"Runtime safety placement: restoring safety to card {safety_card} after "
                f"{self._safety_placement_fit_streak} consecutive cycles of measured device-free headroom for "
                f"its ~{_SAFETY_GPU_LOAD_CHARGE_MB / 1024:.1f}GB context.",
            )

    def _residency_should_pause_post_process(self, device_index: int | None) -> bool:
        """Whether a whole-card residency on this card should also stop the dedicated post-processing lane.

        Requires the lane to be enabled and to sit on the residency's card: its permanent CUDA context (and any
        warm upscaler models) is real device-wide VRAM that a sibling teardown cannot reclaim, so on a card too
        tight to host a whole-card model beside it (Flux on 16GB) the lane must vacate the card exactly as safety
        does. A residency on a card the lane does not occupy leaves it untouched. The worker-wide key (``None``,
        single-GPU) always qualifies: the lane shares the one card.
        """
        if not self._process_lifecycle.post_process_lane_enabled():
            return False
        if device_index is None or not self._card_runtimes:
            return True
        return device_index == self._process_lifecycle.post_process_lane_card_index()

    def _residency_should_pause_vae_lane(self, device_index: int | None) -> bool:
        """Whether a whole-card residency on this card should also stop the dedicated VAE lane.

        Mirrors :meth:`_residency_should_pause_post_process`: the lane's permanent CUDA context is real
        device-wide VRAM a sibling teardown cannot reclaim, so on a card too tight to host a whole-card model
        beside it the lane must vacate the card exactly as safety and the post-processing lane do. Requires
        the lane to be enabled and to sit on the residency's card; the worker-wide key (``None``, single-GPU)
        always qualifies.
        """
        if not self._process_lifecycle.vae_lane_enabled():
            return False
        if device_index is None or not self._card_runtimes:
            return True
        return device_index == self._process_lifecycle.vae_lane_card_index()

    def _residency_should_pause_component_lane(self, device_index: int | None) -> bool:
        """Whether a whole-card residency on this card should also stop the dedicated component lane.

        Mirrors :meth:`_residency_should_pause_vae_lane`: the component lane's permanent CUDA context and its
        resident text encoders are real device-wide VRAM a sibling teardown cannot reclaim, so on a card too
        tight to host a whole-card model beside it the lane must vacate the card exactly as safety, the
        post-processing lane, and the VAE lane do. Requires the lane to be enabled and to sit on the
        residency's card; the worker-wide key (``None``, single-GPU) always qualifies.
        """
        if not self._process_lifecycle.component_lane_enabled():
            return False
        if device_index is None or not self._card_runtimes:
            return True
        return device_index == self._process_lifecycle.component_lane_card_index()

    def _has_post_process_backlog(self) -> bool:
        """Return whether a post-processing job is pending or actively on the lane.

        Whole-card residency must leave a bounded window for post-processing jobs that peel off the resident
        model. Pending work therefore counts as backlog: the normal residency lever is to unload the idle lane's
        modules from VRAM, not to remove the lane and strand its queue. The one exception is a structurally
        incompatible card/model/lane combination, which first disables post-processing for the session and then
        stops the idle lane so the heavy model can fit.
        """
        return bool(self._job_tracker.jobs_pending_post_processing or self._job_tracker.jobs_being_post_processed)

    def _post_process_context_fits_with_residency(
        self,
        forecast: StreamForecast,
        *,
        device_index: int | None,
    ) -> bool:
        """Whether the residency model can load with the post-processing lane's bare context alive."""
        if not self._residency_should_pause_post_process(device_index):
            return True
        if forecast.weights_mb is None or forecast.total_vram_mb is None:
            return True
        target = self._whole_card_ledger.target_process_count(forecast)
        marginal = forecast._effective_marginal_overhead_mb  # noqa: SLF001 - same budget object owns the estimate.
        extra_contexts = max(0, target - 1) + 1  # surviving inference siblings plus the PP lane context.
        free_with_pp_lane_mb = max(
            0.0,
            float(forecast.total_vram_mb) - forecast.per_process_overhead_mb - marginal * extra_contexts,
        )
        return (free_with_pp_lane_mb - forecast.weights_mb) >= forecast._effective_base_reserve  # noqa: SLF001

    def _disable_post_processing_for_whole_card(self, model_name: str | None, forecast: StreamForecast) -> None:
        """Session-disable post-processing because a whole-card model cannot fit beside the lane context."""
        if self._state.post_processing_disabled_by_breaker:
            return
        model = model_name or "the whole-card model"
        self._state.post_processing_disabled_by_breaker = True
        self._state.post_processing_breaker_tripped_at = time.time()
        self._state.post_processing_disabled_reason = (
            f"Disabled: {model} needs whole-card residency and cannot fit beside the dedicated "
            "post-processing lane's GPU context. Disable post-processing or move this workload to a card "
            "with more VRAM, then restart."
        )
        logger.warning(
            f"Disabling post-processing for this session: {model} needs whole-card residency and cannot fit "
            "beside the dedicated post-processing lane's GPU context. Keeping post-processing enabled would "
            "thrash the large model in and out of VRAM for every post-processing job. To restore it, disable "
            "whole-card models or post-processing for this worker, move the workload to a larger card, and restart.",
        )

    def _pause_post_process_for_residency_if_idle(
        self,
        device_index: int | None,
        *,
        model_name: str | None,
        forecast: StreamForecast,
    ) -> bool:
        """Reclaim the post-processing lane for residency without stranding supported PP work."""
        if not self._residency_should_pause_post_process(device_index):
            return False
        if self._post_process_context_fits_with_residency(forecast, device_index=device_index):
            return self.unload_post_process_models_from_vram(device_index=device_index)
        self._disable_post_processing_for_whole_card(model_name, forecast)
        if self._process_lifecycle.is_post_process_gpu_paused or self._has_post_process_backlog():
            return False
        return self._process_lifecycle.pause_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD)

    def _establish_whole_card_residency(
        self,
        job: ImageGenerateJobPopResponse,
        forecast: StreamForecast,
        *,
        announce: bool,
        target_override: int | None = None,
        device_index: int | None = None,
    ) -> None:
        """Claim the device for a whole-card model: stop idle siblings and move safety off-GPU.

        The siblings' fixed per-process CUDA contexts (not their models) over-commit the device, and a context
        is only reclaimed by the process exiting (``torch.cuda.empty_cache`` returns cached blocks but never a
        context). Reduce the live inference-process count to the largest that still leaves room for this model's
        weights plus its activation reserve, and, on the very edge (Flux on a 16GB card), also move the
        safety process off-GPU so its context is freed too. The model is remembered so the residency is held
        and then restored once its job drains (after the configured cooldown). Only idle inference processes
        are stopped; a busy sibling is left to finish its job.

        ``device_index`` scopes the residency to one card on a multi-GPU host (only that card's processes are
        reduced, and safety is paused only if it sits on that card); None is the single-GPU / worker-wide case.
        """
        self._whole_card_ledger.record_grant(
            device_index,
            model=job.model,
            forecast=forecast,
            cooldown_until=time.time() + self._whole_card_cooldown_seconds(),
            now=time.time(),
            refresh_established=announce,
        )

        # ``target_override`` lets a caller size the depth from the admission verdict's rejected peak rather
        # than the forecast's lighter resident-weight estimate, for the activation-peak context over-commit the
        # weight-based gates leave co-resident.
        target = target_override if target_override is not None else (forecast.max_resident_processes() or 1)
        current = self._process_map.num_loaded_inference_processes(device_index=device_index)
        after = current
        if target < current:
            after = self._process_lifecycle.scale_inference_processes(
                target,
                device_index=device_index,
                whole_card_model=job.model,
            )

        safety_paused = self._pause_safety_for_residency_if_idle(device_index)
        post_process_paused = self._pause_post_process_for_residency_if_idle(
            device_index,
            model_name=job.model,
            forecast=forecast,
        )
        # The disaggregated pipeline's VAE lane holds an equivalent bare CUDA context; stop it off-GPU on the
        # residency's card so the heavy model's weights are not tipped into host-RAM streaming by it. Disagg
        # dispatch is already suppressed while a residency is active, so the lane is idle here. A no-op unless
        # disaggregation is enabled and the lane sits on this card.
        if self._residency_should_pause_vae_lane(device_index):
            self._process_lifecycle.pause_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD)
        # The disaggregated pipeline's component (text-encode) lane holds an equivalent bare CUDA context plus
        # resident encoders; stop it off-GPU on the residency's card for the same reason as the VAE lane.
        # Stopping it also drops the lane out of the disaggregation liveness predicate, so new jobs route
        # monolithic while the residency holds. A no-op unless disaggregation is enabled and the lane sits here.
        if self._residency_should_pause_component_lane(device_index):
            self._process_lifecycle.pause_component_off_gpu(owner=PauseOwner.WHOLE_CARD)

        if announce or after < current or safety_paused or post_process_paused:
            safety_note = " and moving safety off-GPU" if safety_paused else ""
            total_mb = forecast.total_vram_mb
            card_phrase = f"the whole ~{total_mb / 1024:.0f}GB card" if total_mb else "nearly the whole card"
            logger.opt(ansi=True).warning(
                f"<fg #f0beff>Whole-card residency: reserving the device for {job.model} "
                f"(inference processes {current} -> {after} of {self._max_inference_processes}, target "
                f"{target}){safety_note}. Its weights + activations need {card_phrase}; co-resident "
                f"siblings/safety would force the driver to stream activations to host RAM and run several "
                f"times slower. {self._process_map.residency_snapshot()}</>",
            )

    def _should_prestage_whole_card_head(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        forecast: StreamForecast,
        available_process: HordeProcessInfo,
        *,
        device_index: int | None = None,
    ) -> bool:
        """Whether a whole-card head should be pre-loaded into a spare's RAM while a live job holds the device.

        ``preload_model`` is a RAM-only load (the weights reach VRAM only at sampling), so a heavy head can
        load into an idle process's RAM concurrently with the in-flight job, and be ready to sample the
        instant the device frees, rather than its multi-GB disk->RAM load only starting after the drain.

        Pre-staging is worthwhile only when:

        - a live job actually holds the device (otherwise the normal whole-card path claims the idle card and
          loads immediately, with nothing to overlap);
        - the head is not already resident or loading somewhere (nothing left to pre-stage);
        - there is an idle spare to hand the preload to (never the live job's own process); and
        - system RAM can hold the head's *weights* alongside the in-flight job, i.e. the operator's "assuming the
          RAM can support it" (see :meth:`_prestage_weights_fit_ram`). A RAM shortfall falls back to the prior
          claim-the-card-and-wait behavior.

        ``device_index`` scopes "a live job holds the device" to one card on a multi-GPU host (the card the
        spare slot sits on); None is the single-GPU / worker-wide case.
        """
        if device_index is None:
            live_jobs_on_device = len(self._job_tracker.jobs_in_progress)
        else:
            live_jobs_on_device = len(self._jobs_in_progress_on_card(device_index))
        if live_jobs_on_device == 0:
            return False
        if self._is_model_forecast_to_load(job.model):
            return False
        if available_process.is_process_busy():
            return False
        return self._prestage_weights_fit_ram(job, baseline, forecast)

    def _prestage_weights_fit_ram(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        forecast: StreamForecast,
    ) -> bool:
        """Whether system RAM can hold the head's *weights* alongside the in-flight job.

        A RAM preload materialises only the model's weights on the CPU offload device; the activation working
        set that inflates the full :func:`predict_job_ram_mb` burden lives in VRAM at sampling time, not in
        RAM. Gating the pre-stage on that full burden over-rejects a head whose weights comfortably fit (a Flux
        fp8 head's ~11.5GB of weights versus its ~24GB activation-inclusive estimate), which is what forces the
        worker to tear every idle sibling down instead of staging. Worse, the establish path it then falls back
        to loads those same weights into RAM with no hard gate at all, so the burden gate held the pre-stage to
        a stricter standard than the path it defers to. So gate on the weight footprint (the forecast's
        ``weights_mb``, the persistent RAM cost of a preload) plus the configured RAM reserve.

        When the weight estimate is unavailable (it should not be once ``needs_exclusive_residency`` is True,
        which requires known weights) fall back to the conservative full-burden RAM budget, so a head whose
        footprint cannot be sized is never force-staged onto a RAM-pressured host.
        """
        available_ram_mb = self._measured_available_ram_mb()
        weights_mb = forecast.weights_mb
        committed_ram_mb = self._reserve_ledger.total_ram_mb()
        if weights_mb is None:
            return self._ram_budget.check_job(
                job,
                baseline,
                available_ram_mb,
                committed_reserve_mb=committed_ram_mb,
            ).fits
        return (available_ram_mb - committed_ram_mb) >= float(weights_mb) + self._ram_budget.reserve_mb

    def _begin_whole_card_residency(
        self,
        job: ImageGenerateJobPopResponse,
        forecast: StreamForecast,
        *,
        announce: bool,
        device_index: int | None = None,
    ) -> None:
        """Record a whole-card residency for a head being pre-staged into RAM, without claiming the card yet.

        The device cannot be claimed while a live job holds it, but the heavy head's weights can load into a
        spare's RAM now. This sets the same residency bookkeeping :meth:`_establish_whole_card_residency` does
        (so the cooldown, the restore, and the recovery-supervisor wedge grace all cover the pre-stage load and
        the convergence that follows), minus the process teardown and safety pause: those are deferred to
        :meth:`_converge_whole_card_residency`, which runs once the head is staged and the device frees.

        ``device_index`` scopes the pre-staged residency to one card on a multi-GPU host; None is the
        single-GPU / worker-wide case.
        """
        self._whole_card_ledger.record_grant(
            device_index,
            model=job.model,
            forecast=forecast,
            cooldown_until=time.time() + self._whole_card_cooldown_seconds(),
            now=time.time(),
            refresh_established=announce,
        )
        if announce:
            logger.opt(ansi=True).info(
                f"<fg #f0beff>Pre-staging whole-card head {job.model} into a spare process's RAM while the "
                f"in-flight job finishes; the device will be reserved for it (idle siblings stopped"
                f"{' and safety moved off-GPU' if self._residency_should_pause_safety(device_index) else ''}) "
                f"once it frees, so its weights are loaded before it samples instead of after. "
                f"{self._process_map.residency_snapshot()}</>",
            )

    def _converge_whole_card_residency(self) -> None:
        """Collapse an in-progress whole-card residency to sole VRAM residency once its model is staged.

        Driven each scheduling cycle while a residency is held. A pre-staged head is loaded into RAM before
        the device is claimed (see :meth:`_begin_whole_card_residency`); stopping idle siblings before the head
        is actually resident on a process could kill the very spare the pre-stage wants to use, so this waits
        until the head is resident or loading on a process. From then on the scale-down is told this is a
        whole-card collapse (``whole_card_model``), so it spares only that head's holder and stops the *other*
        idle siblings, including ones holding a model still queued behind the head, which the generic
        scale-down guard would otherwise protect and thereby pin the count above the target forever. Those
        queued jobs wait and reload once the head drains (see :meth:`_restore_siblings_after_whole_card`).
        Reclaiming the siblings' CUDA contexts and moving safety off-GPU leaves the staged head the whole card
        when it samples. A no-op until a residency is held and its model is staged; idempotent at the target.
        Converges every held residency, so on a multi-GPU host each card's pre-staged head collapses its own
        card independently.
        """
        for device_index, state in self._held_residencies():
            model = state.model
            if model is None or not self._whole_card_residency_has_holder(model, device_index):
                continue
            forecast = state.forecast
            target = self._whole_card_ledger.target_process_count(forecast)
            if self._process_map.num_loaded_inference_processes(device_index=device_index) > target:
                self._process_lifecycle.scale_inference_processes(
                    target,
                    device_index=device_index,
                    whole_card_model=model,
                )
            self._pause_safety_for_residency_if_idle(device_index)
            if forecast is not None:
                self._pause_post_process_for_residency_if_idle(device_index, model_name=model, forecast=forecast)

    def _whole_card_residency_has_holder(self, model: str, device_index: int | None) -> bool:
        """Whether a held whole-card model is staged or resident on a live process.

        Convergence must wait until the pre-staged head has a holder so the scale-down can spare that process.
        Once the holder reports ready it may sit in ``WAITING_FOR_JOB`` rather than a preload state, so the
        generic "forecast to load" predicate is too narrow here.
        """
        return any(
            process.process_type is HordeProcessType.INFERENCE
            and process.loaded_horde_model_name == model
            and (device_index is None or process.device_index == device_index)
            for process in self._process_map.values()
        )

    def _prestaged_whole_card_not_ready(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether ``job`` must wait for its in-progress whole-card residency to claim the card before sampling.

        A pre-staged whole-card head is loaded into RAM (see :meth:`_begin_whole_card_residency`) before the
        device is reserved, so dispatching it would commit its weights to VRAM while idle siblings (or the
        just-drained busy process) still hold their CUDA contexts, forcing the first step to stream. This
        returns True until the residency has converged, i.e. the live inference-process count is at the forecast's
        target, safety is off-GPU if this residency needs it, and the card has drained enough to load the
        weights (the same :meth:`_whole_card_teardown_exhausted` gate the non-pre-staged path loads under).

        Returns False for any job that is not the currently-held residency's model, so ordinary dispatch (and
        the non-pre-staged whole-card path, which only preloads once already at sole residency) is unaffected.
        """
        found, device_index = self._residency_holder_for_model(job.model)
        if not found:
            return False
        # Re-use the residency's stored forecast for the readiness check rather than re-deriving from the
        # current (possibly degraded) state: it carries the stable weight footprint, the budget-relative target,
        # and the fits_alone guarantee captured at establishment, which the live device reading and the bounded
        # drain backstop in _whole_card_teardown_exhausted then resolve against the real, post-teardown VRAM.
        stored = self._residency_state(device_index).forecast
        if stored is not None:
            # The stored forecast reflects the residency's actual budget-relative target and the
            # weight footprint at establishment time; only re-derive when it was never captured.
            return not self._whole_card_teardown_exhausted(stored, device_index=device_index)
        baseline = self._model_metadata.get_baseline(job.model) if job.model is not None else None
        forecast = self._forecast_streaming(job, baseline, device_index=device_index)
        return not self._whole_card_teardown_exhausted(forecast, device_index=device_index)

    def _resident_whole_card_head_ready(
        self,
        job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
    ) -> bool:
        """Ensure an already-resident whole-card head has sole residency before it samples.

        The ordinary whole-card path runs during preload admission, so it used to miss a heavy head whose
        model was already resident on an idle process while sibling processes still held their own models.
        That job is entitled to the same residency as a to-be-loaded head: establish the residency, evict
        sibling VRAM, and defer dispatch until the teardown is complete.
        """
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")
        if not self._budget_active() or not self._whole_card_residency_enabled():
            return True

        target_device_index = process_with_model.device_index if self._multi_gpu_routing_active else None
        baseline = self._model_metadata.get_baseline(job.model)
        forecast = self._forecast_streaming(job, baseline, device_index=target_device_index)
        if not self._whole_card_ledger.residency_demanded(
            forecast,
            enabled=self._whole_card_residency_enabled(),
            is_head_blocker=True,
        ):
            return True
        if not self._whole_card_warranted(forecast):
            self._log_whole_card_declined(job, forecast)
            return True

        first_time = not self._job_tracker.is_admitted_exclusive(job)
        self._job_tracker.mark_admitted_exclusive(job)
        self._establish_whole_card_residency(
            job,
            forecast,
            announce=first_time,
            device_index=target_device_index,
        )
        self.unload_models_from_vram(
            process_with_model,
            under_pressure=True,
            for_head_of_queue=True,
            device_index=target_device_index,
        )
        return self._whole_card_teardown_exhausted(forecast, device_index=target_device_index)

    def _residency_holder_for_model(self, model: str | None) -> tuple[bool, int | None]:
        """Return ``(found, device_index)`` for the card whose held whole-card residency is for ``model``.

        ``found`` distinguishes a genuine hit on the ``None`` (single-GPU / worker-wide) key from a miss, since
        ``None`` is itself a valid residency key.
        """
        return self._whole_card_ledger.holder_for_model(model)

    def _whole_card_teardown_exhausted(self, forecast: StreamForecast, *, device_index: int | None = None) -> bool:
        """Whether a whole-card residency has done all it can and the head can now load best-effort.

        The whole-card branch defers a heavy head while a teardown can still make room: idle siblings left to
        stop, the safety process still on-GPU, or their freed VRAM still draining. The first two are the
        *structural* hold and are decided on topology alone (live process count at or below the forecast's
        target, safety off-GPU if this residency needs it). Once both hold the teardown is structurally
        complete: the model fits alone (``fits_alone``, the grant precondition for a whole-card residency), so
        the only remaining question is whether the asynchronously-freed VRAM has actually materialised.

        That last step is resolved against the live device, not the stale establishment forecast (whose
        ``free_now_mb`` was captured before the teardown freed the siblings' VRAM, so reading it would park the
        head forever once it drains): the *live* free-VRAM reading dispatches the head the moment it confirms the
        drain (safe to read here, at sole residency, where it only rises as the stopped contexts release), and a
        bounded ``_WHOLE_CARD_DRAIN_SETTLE_SECONDS`` backstop admits it on the structural ``fits_alone`` guarantee
        if the measurement is unavailable or lags, so the head never parks indefinitely. A model that still
        cannot fit co-resident even at sole residency loads best-effort the same way and samples slowly under the
        over-budget step grace rather than wedging the queue until the recovery supervisor soft-resets.

        ``device_index`` scopes the live-context count and the safety check to one card on a multi-GPU host;
        None is the single-GPU / worker-wide case.
        """
        return self._whole_card_ledger.teardown_complete(
            forecast,
            loaded_process_count=self._process_map.num_loaded_inference_processes(device_index=device_index),
            safety_pause_required=self._residency_should_pause_safety(device_index),
            safety_paused=self._process_lifecycle.is_safety_gpu_paused,
            post_process_pause_required=(
                self._residency_should_pause_post_process(device_index)
                and not self._post_process_context_fits_with_residency(forecast, device_index=device_index)
            ),
            post_process_cleared=self._process_map.num_post_process_processes(device_index=device_index) == 0,
            component_lane_pause_required=self._residency_should_pause_component_lane(device_index),
            component_lane_cleared=self._process_map.num_component_processes(device_index=device_index) == 0,
            weights_fit_live=self._whole_card_weights_fit_live(forecast, device_index=device_index),
            drain_backstop_elapsed=self._whole_card_drain_backstop_elapsed(device_index),
        )

    def _whole_card_weights_fit_live(self, forecast: StreamForecast, *, device_index: int | None = None) -> bool:
        """Whether the residency model's weights fit the *live* measured free VRAM (read only at sole residency).

        Keyed on the live device reading rather than the forecast's stored ``free_now_mb`` (captured at
        establishment, before the teardown freed the siblings' VRAM). Only the caller's structural-completion
        guard makes this safe to trust: at sole residency the reading is monotonic, only rising as the
        stopped siblings' contexts release, so it never reads deceptively high the way an instantaneous
        reading does during startup (idle contexts not yet allocated reading as free). Unknown weight or
        measurement returns False so the bounded structural backstop, not a guess, drives the fallback.
        """
        if forecast.weights_mb is None:
            return False
        free_now = self._measured_free_vram_mb(device_index=device_index)
        if free_now is None:
            return False
        return (free_now - forecast.weights_mb) >= forecast._effective_base_reserve  # noqa: SLF001

    def _whole_card_drain_backstop_elapsed(self, device_index: int | None) -> bool:
        """Whether the bounded drain-settle window has elapsed since this residency was established.

        The deterministic backstop for the dispatch gate: once a structurally-complete teardown has held for
        ``_WHOLE_CARD_DRAIN_SETTLE_SECONDS`` without the live reading confirming the drain, the head is admitted
        on the structural ``fits_alone`` guarantee rather than parking forever. Measured from ``established_at``
        (the teardown completes shortly after establishment, and this is gated behind the structural-complete
        check at the call site), so a stuck or unavailable free-VRAM measurement can never wedge the head.
        """
        return self._whole_card_ledger.drain_backstop_elapsed(
            device_index,
            now=time.time(),
            settle_seconds=_WHOLE_CARD_DRAIN_SETTLE_SECONDS,
        )

    def is_whole_card_residency_active(self) -> bool:
        """Whether any card currently holds a whole-card residency lease (its cooldown still running).

        Mirrors the ``active`` field of :meth:`whole_card_residency_state` but without building the full
        snapshot, so the job popper's large-model re-entry cooldown can cheaply ask "is the lease up?" every
        pop cycle: the lease is up exactly when this returns False (no card holds a residency model).
        """
        return self._whole_card_ledger.any_held()

    def whole_card_residency_grace_active(self) -> bool:
        """Whether a whole-card residency is establishing, so the held queue is intentional (not a wedge).

        While true, the recovery supervisor must not treat the deliberately-deferred heavy head (waiting
        for idle siblings to stop, the safety process to cycle off-GPU, and ~11GB of weights to load) as a
        structural queue wedge and soft-reset the pools mid-setup. Bounded by
        ``_WHOLE_CARD_ESTABLISH_GRACE_SECONDS`` so a residency that genuinely never loads still trips the
        supervisor. Public: read by the process manager's wedge assessment.
        """
        return self._whole_card_ledger.grace_active(
            now=time.time(),
            establish_grace_seconds=_WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
            restore_grace_seconds=_WHOLE_CARD_RESTORE_GRACE_SECONDS,
        )

    def heavy_head_load_grace_active(self) -> bool:
        """Whether a heavy head admitted off the whole-card path is still inside its bounded load window.

        A model that streams even with the whole card to itself never enters the whole-card branch, so
        ``whole_card_residency_grace_active`` does not cover it; but its multi-gigabyte load holds the queue
        just the same. While true the recovery supervisor must not treat that deliberate hold as a structural
        wedge and give up the never-run backlog. Bounded by ``_HEAVY_HEAD_LOAD_GRACE_SECONDS`` so a head that
        genuinely never loads still trips the supervisor. Public: read by the process manager's wedge assessment.
        """
        if self._heavy_head_admitted_at == 0.0:
            return False
        return (time.time() - self._heavy_head_admitted_at) < _HEAVY_HEAD_LOAD_GRACE_SECONDS

    def ram_reclaim_cycle_grace_active(self) -> bool:
        """Whether a deliberate RAM-reclaim process cycle is still inside its bounded respawn/preload window.

        When the RAM budget cannot fit the next head and cycles an idle slot to return allocator-retained
        RAM to the OS (:meth:`_replace_stale_ram_unload_process`), the slot respawns and the head must then
        preload onto it. The queue is unservable across that window, but by the worker's own deliberate,
        bounded action, not a wedge. While true the recovery supervisor must not treat the held queue as a
        structural wedge and fault the servable backlog. Bounded by ``_RAM_RECLAIM_CYCLE_GRACE_SECONDS`` so a
        cycle that genuinely never recovers still trips the supervisor. Public: read by the process manager's
        wedge assessment.
        """
        if self._ram_reclaim_cycle_at == 0.0:
            return False
        return (time.time() - self._ram_reclaim_cycle_at) < _RAM_RECLAIM_CYCLE_GRACE_SECONDS

    def card_residency(self, device_index: int | None) -> tuple[str | None, str]:
        """Return ``(model, phase)`` for the whole-card residency held on ``device_index`` (per-card view).

        ``model`` is None when this card holds no residency; otherwise ``phase`` is ``establishing`` while the
        establish grace is still in effect, else ``holding``; this is the same phase split the worker-wide
        :meth:`whole_card_residency_state` reports. The single-GPU worker-wide residency lives under the
        ``None`` key, so a single-GPU caller reads it by passing ``device_index=None``. Reads without creating:
        a card with no residency is left absent from the map.
        """
        model, phase = self._whole_card_ledger.phase(
            device_index,
            now=time.time(),
            establish_grace_seconds=_WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        )
        if model is None:
            return None, ""
        return model, str(phase)

    def whole_card_residency_state(self) -> WholeCardResidencyState:
        """Return a read-only view of the whole-card residency posture, for the status snapshot/TUI.

        ``possible`` is config + topology only (feature on, the VRAM budget on, and something is actually
        tear-down-able: more than one inference process, or a safety process that can be moved off-GPU);
        it powers the operator heads-up so a teardown is not a surprise. The remaining fields describe a
        residency that is currently held (its model, the establish/hold phase, the reduced process count,
        the safety-pause state, and the establishing forecast's hard numbers for the detailed view).
        Tolerant of partially-mocked config (used in tests that build snapshots): config flags are read
        with boolean coercion so a non-bool never leaks a truthy Mock into ``possible``.
        """
        bridge_data = self._runtime_config.bridge_data
        enabled = self._whole_card_residency_enabled()
        budget_on = bridge_data.enable_vram_budget is True
        safety_off_enabled = bool(self._whole_card_safety_off_gpu_enabled())
        multi_process = self._max_inference_processes > 1
        possible = enabled and budget_on and (multi_process or safety_off_enabled)

        # Represent the posture with the first held residency (single-GPU has at most one).
        # ``active`` is true while any card holds a residency.
        representative = next((state for _index, state in self._held_residencies()), None)
        model = representative.model if representative is not None else None
        active = model is not None
        forecast = representative.forecast if representative is not None else None
        now = time.time()

        phase = ""
        cooldown_remaining: float | None = None
        processes_target = 0
        weights_mb = reserve_mb = free_now_mb = free_if_alone_mb = None
        max_resident_processes: int | None = None
        if active and representative is not None:
            establishing = (
                representative.established_at != 0.0
                and (now - representative.established_at) < _WHOLE_CARD_ESTABLISH_GRACE_SECONDS
            )
            phase = "establishing" if establishing else "holding"
            cooldown_remaining = max(0.0, representative.cooldown_until - now)
            if forecast is not None:
                weights_mb = forecast.weights_mb
                reserve_mb = forecast.reserve_mb
                free_now_mb = forecast.free_now_mb
                free_if_alone_mb = forecast.free_if_alone_mb
                max_resident_processes = forecast.max_resident_processes()
            processes_target = max_resident_processes or 1

        total_vram_mb = (
            forecast.total_vram_mb if forecast is not None else self._process_map.get_reported_total_vram_mb()
        )

        return WholeCardResidencyState(
            possible=possible,
            enabled=enabled,
            safety_off_gpu_enabled=safety_off_enabled,
            cooldown_seconds=self._whole_card_cooldown_seconds(),
            per_process_overhead_mb=self._per_process_overhead_mb(),
            total_vram_mb=total_vram_mb,
            active=active,
            model=model,
            phase=phase,
            safety_paused=bool(self._process_lifecycle.is_safety_gpu_paused),
            processes_now=self._process_map.num_loaded_inference_processes(),
            processes_target=processes_target,
            processes_max=self._max_inference_processes,
            cooldown_remaining_seconds=cooldown_remaining,
            weights_mb=weights_mb,
            reserve_mb=reserve_mb,
            free_now_mb=free_now_mb,
            free_if_alone_mb=free_if_alone_mb,
            max_resident_processes=max_resident_processes,
        )

    def _whole_card_safety_off_gpu_enabled(self) -> bool:
        """Whether a whole-card job should move the safety process off-GPU (config + safety actually on-GPU)."""
        bridge_data = self._runtime_config.bridge_data
        return bridge_data.whole_card_residency_safety_off_gpu and bridge_data.safety_on_gpu

    def _whole_card_cooldown_seconds(self) -> float:
        """Operator-configured seconds to hold a whole-card residency after its last job drains."""
        return float(self._runtime_config.bridge_data.whole_card_residency_cooldown_seconds)

    def _restore_siblings_after_whole_card(self) -> None:
        """Restore inference concurrency and safety-on-GPU after a whole-card residency has fully drained.

        Held while the residency model is still pending or in progress, and for the configured cooldown after
        that, so a burst of heavy jobs reuses one residency rather than each thrashing the process count and
        the safety process. Once neither condition holds, that card's sibling processes are grown back to its
        ceiling and, if the residency was on the safety card, the safety process is restored to the GPU.
        Restores every drained card's residency independently; a no-op when none is outstanding.
        """
        now = time.time()
        active_models = {j.model for j in self._job_tracker.jobs_in_progress}
        active_models.update(j.model for j in self._job_tracker.jobs_pending_inference)
        # The exclusive-job suppression is worker-wide: it holds every card's residency a little longer, which
        # only delays restoring concurrency (conservative-safe) rather than risking an over-commit.
        has_exclusive = self._job_tracker.has_exclusive_job_in_progress()
        post_processing_has_work = bool(
            self._job_tracker.jobs_pending_post_processing or self._job_tracker.jobs_being_post_processed
        )
        for device_index, state in self._held_residencies():
            model = state.model
            if model in active_models or has_exclusive:
                # Still serving the residency; keep it (refresh the cooldown so it survives the lull between
                # back-to-back heavy jobs).
                state.cooldown_until = now + self._whole_card_cooldown_seconds()
                continue
            if time.time() < state.cooldown_until and not self._ready_different_model_head_on_device(
                residency_model=model,
                device_index=device_index,
            ):
                # Drained, but hold the residency through the cooldown so an imminent heavy job reuses it.
                continue
            state.model = None
            state.established_at = 0.0
            state.forecast = None
            # The restore's own churn (respawning siblings, cycling safety back on-GPU) briefly makes the queue
            # unservable; mark its start so the wedge grace covers it (see _WHOLE_CARD_RESTORE_GRACE_SECONDS).
            state.restore_at = time.time()
            safety_restored = (
                self._process_lifecycle.restore_safety_on_gpu()
                if self._residency_should_pause_safety(device_index)
                and self._arbiter_admits_safety_gpu_load(device_index)
                and not post_processing_has_work
                else False
            )
            post_process_restored = (
                self._process_lifecycle.restore_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD)
                if self._residency_should_pause_post_process(device_index)
                and not self._state.post_processing_disabled_by_breaker
                else False
            )
            vae_lane_restored = (
                self._process_lifecycle.restore_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD)
                if self._residency_should_pause_vae_lane(device_index)
                else False
            )
            component_lane_restored = (
                self._process_lifecycle.restore_component_off_gpu(owner=PauseOwner.WHOLE_CARD)
                if self._residency_should_pause_component_lane(device_index)
                else False
            )
            ceiling = self._residency_restore_ceiling(device_index)
            current = self._process_map.num_loaded_inference_processes(device_index=device_index)
            if (
                current >= ceiling
                and not safety_restored
                and not post_process_restored
                and not vae_lane_restored
                and not component_lane_restored
            ):
                continue
            after = self._process_lifecycle.scale_inference_processes(ceiling, device_index=device_index)
            self._reconcile_worker_shed_to_pool()
            safety_note = " and restoring safety to the GPU" if safety_restored else ""
            post_process_note = " and restarting the post-processing lane" if post_process_restored else ""
            vae_lane_note = " and restarting the VAE lane" if vae_lane_restored else ""
            component_lane_note = " and restarting the component lane" if component_lane_restored else ""
            logger.opt(ansi=True).info(
                f"<fg #7b7d7d>Whole-card residency for {model} complete; restoring inference processes "
                f"({current} -> {after} of {ceiling})"
                f"{safety_note}{post_process_note}{vae_lane_note}{component_lane_note}.</>",
            )

    def _ready_different_model_head_on_device(
        self,
        *,
        residency_model: str | None,
        device_index: int | None,
    ) -> bool:
        """Return whether a ready queue head on this card should preempt a drained residency cooldown."""
        in_progress = set(self._job_tracker.jobs_in_progress)
        head = next((job for job in self._job_tracker.jobs_pending_inference if job not in in_progress), None)
        if head is None or head.model is None or head.model == residency_model:
            return False
        process_info = self._resident_process_for_job(head)
        if process_info is None or not process_info.can_accept_job():
            return False
        return device_index is None or process_info.device_index == device_index

    def _reconcile_worker_shed_to_pool(self) -> None:
        """Realign the RAM governor's worker-wide shed record with the live inference-process count.

        The RAM governor records a worker-wide shed so its own restore can grow the pool back once RAM
        proves headroom. When a different mechanism grows the pool instead (the whole-card residency
        restore), that record would otherwise persist as a stale claim that the pool is still short of
        plan, and while the host stays under its RAM floor the governor re-sheds the pool the residency
        just regrew. Recompute the record from the live count against the recorded plan: drop it once the
        pool is back at (or above) plan, otherwise set the shortfall to the true remaining gap. A no-op on a
        multi-GPU host, whose reduction tracks per-card shedding rather than a worker-wide record.
        """
        worker_shed = self._ram_governor_state.worker_shed
        if worker_shed is None:
            return
        loaded = self._process_map.num_loaded_inference_processes()
        if loaded >= worker_shed.planned_process_count:
            self._ram_governor_state.worker_shed = None
        else:
            worker_shed.shed_process_count = worker_shed.planned_process_count - loaded

    def _residency_restore_ceiling(self, device_index: int | None) -> int:
        """The process count to grow back to when a card's whole-card residency is restored.

        That card's own ``target_process_count`` on a multi-GPU host; the worker-wide launched-process
        ceiling for the single-GPU / worker-wide (``None``) case.
        """
        if device_index is not None and device_index in self._card_runtimes:
            return self._card_runtimes[device_index].target_process_count
        return self._max_inference_processes

    def _update_head_starvation_timer(self, head_job: ImageGenerateJobPopResponse | None) -> None:
        """Track how long the current head-of-queue job has been stuck on an otherwise-idle device.

        The clock only runs while no live job holds the device: a head waiting behind in-flight work is
        legitimately queued, not starved. It resets whenever the head changes (a different job reached the
        front) so the backstop measures *this* head's wait, not the queue's age.
        """
        head_id = str(head_job.id_) if head_job is not None and head_job.id_ is not None else None
        if head_id is None or len(self._job_tracker.jobs_in_progress) > 0:
            self._head_starvation_job_id = None
            self._head_starvation_since = 0.0
            return
        if head_id != self._head_starvation_job_id:
            self._head_starvation_job_id = head_id
            self._head_starvation_since = time.time()

    def _head_starved_seconds(self, job: ImageGenerateJobPopResponse) -> float:
        """Seconds this job has been the idle-device head, or 0.0 when it is not the tracked head."""
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None or job_id != self._head_starvation_job_id or self._head_starvation_since == 0.0:
            return 0.0
        return time.time() - self._head_starvation_since

    def _clear_head_starvation_timer(self) -> None:
        """Reset the head-starvation clock once a job is dispatched (the wedge, if any, is broken)."""
        self._head_starvation_job_id = None
        self._head_starvation_since = 0.0

    def _diagnose_dispatch_stall(
        self,
        head: ImageGenerateJobPopResponse,
        stable_diffusion_reference: dict[str, ImageGenerationModelRecord],
    ) -> str:
        """Return why the head-of-queue job is not being dispatched (read-only; never raises into the loop)."""
        return self._classify_dispatch_stall(head, stable_diffusion_reference)[1]

    def _classify_dispatch_stall(
        self,
        head: ImageGenerateJobPopResponse,
        stable_diffusion_reference: dict[str, ImageGenerationModelRecord],
    ) -> tuple[SlotDutyBucket, str]:
        """Name the gate parking the head-of-queue job, as a duty bucket plus the operator-facing text.

        The scheduler returns ``None`` from :meth:`get_next_job_and_process` at several points without saying
        why, so a stuck queue with idle processes leaves no record of which gate parked the head. This
        re-derives that reason, with the most detail for the genuinely suspicious case -- the head's model is
        resident on an *idle* process yet nothing dispatches, since that is the scheduler-bug-shaped stall
        that is otherwise invisible. The bucket half feeds the slot-duty accounting every tick
        (:meth:`record_slot_duty`), so the same derivation prices the empty slot's wall clock; the text half
        feeds the throttled parked-head log line. Read-only; never raises into the loop.
        """
        process = self._resident_process_for_job(head)
        if process is None:
            if head.model is not None and self._horde_model_map.is_model_loading(head.model):
                return SlotDutyBucket.MODEL_LOADING, "its model is loading (a preload is in progress)"
            # The head's model can be resident only on a disaggregation-pinned sampler lane, which the dispatch
            # query excludes. That is not a budget defer: the head is deliberately held for the pin to release
            # (rather than funding a second copy), so name the pin, the job holding it, and the in-flight
            # sampling that keeps the card busy, instead of reporting a generic not-resident preload defer.
            pinned_lane = self._pinned_lane_resident_for_job(head)
            if pinned_lane is not None:
                owner = self._disaggregation_pin_owner(pinned_lane.process_id)
                owner_text = f" holding disaggregated job {owner[:8]}" if owner else ""
                peaks = self._disaggregation_sampling_peaks()
                peaks_text = (
                    f"; {len(peaks)} sampling(s) in flight totalling {sum(peaks.values()):.0f} MB"
                    if peaks
                    else "; no sampling currently in flight"
                )
                return SlotDutyBucket.DISAGG_PIN_WAIT, (
                    f"its model is resident only on process {pinned_lane.process_id}, pinned as a disaggregation "
                    f"sampler{owner_text}; the head waits for that pin to release and dispatch onto the resident "
                    f"lane rather than fund a second copy that cannot fit beside the pinned residents{peaks_text}"
                )
            # A whole-card residency held for a *different* model reserves the card and tore its siblings down,
            # so a head of another model cannot load until that residency restores. Name it: otherwise this
            # reads as a generic VRAM-budget defer (the card looks idle with ample free VRAM) when the real
            # cause is a residency granted to a non-head model.
            nonhead_residency_model = next(
                (
                    state.model
                    for _, state in self._held_residencies()
                    if state.model is not None and state.model != head.model
                ),
                None,
            )
            if nonhead_residency_model is not None:
                return SlotDutyBucket.WHOLE_CARD_RESERVED, (
                    f"its model is not resident because a whole-card residency is held for non-head model "
                    f"{nonhead_residency_model!r}: the card is reserved for that model and its siblings were "
                    f"torn down, so this head cannot load until that residency restores"
                )
            return SlotDutyBucket.PRELOAD_DEFERRED, (
                "its model is not resident and no preload has been admitted "
                "(usually a VRAM/RAM budget defer; see the budget lines above)"
            )
        if not process.can_accept_job():
            return SlotDutyBucket.RESIDENT_SLOT_BUSY, (
                f"its model is resident on process {process.process_id}, but that process is busy "
                f"({process.last_process_state.name})"
            )

        # Resident on an idle process: the interesting case. Name the gate that is holding dispatch.
        keep_single, single_reason = self._process_map.keep_single_inference(
            stable_diffusion_model_reference=stable_diffusion_reference,
        )
        pending_and_active = len(self._job_tracker.jobs_pending_inference) + len(self._job_tracker.jobs_in_progress)
        if keep_single and pending_and_active > 1:
            return SlotDutyBucket.KEEP_SINGLE_INFERENCE, (
                f"its model is resident and idle on process {process.process_id}, but dispatch is held by "
                f"keep-single-inference ({single_reason})"
            )
        in_progress = len(self._job_tracker.jobs_in_progress)
        cap = self._max_jobs_in_progress_allowed()
        if in_progress >= cap:
            # The exclusive-admit hold collapses the cap to the running job; name it distinctly so the
            # serialization is attributed to the admit, not to a generic cap the operator would chase
            # through max_threads.
            if self._job_tracker.has_exclusive_job_in_progress() and not self._job_tracker.is_admitted_exclusive(
                head,
            ):
                return SlotDutyBucket.EXCLUSIVE_ISOLATION, (
                    f"its model is resident and idle on process {process.process_id}, but an exclusively-"
                    f"admitted over-budget job has the device to itself (in_progress={in_progress})"
                )
            return SlotDutyBucket.CONCURRENCY_CAP, (
                f"its model is resident and idle on process {process.process_id}, but the concurrency cap is "
                f"reached (in_progress={in_progress}, cap={cap})"
            )
        if not self._concurrent_overlap_allowed(head, target_device_index=process.device_index):
            return SlotDutyBucket.OVERLAP_HEADWAY, (
                f"its model is resident and idle on process {process.process_id}, but the overlap-headway gate "
                "is holding it (the in-flight job has not made enough progress to share the card)"
            )

        # A held whole-card residency parks its own pre-staged head until the live inference-process count
        # collapses to the forecast's target (sole residency). The convergence teardown is meant to stop the
        # idle siblings, including ones holding a model queued behind the head, sparing only the head's
        # holder. If the head is still parked with such a sibling un-torn-down, the convergence shrink has not
        # collapsed the pool, and the head will be deferred until the recovery supervisor soft-resets. Name
        # that specific state rather than reporting a gate-less "scheduler stall", so the post-mortem points at
        # the residency teardown rather than the dispatch path.
        found_residency, residency_device = self._residency_holder_for_model(head.model)
        if found_residency and self._prestaged_whole_card_not_ready(head):
            blockers = self._whole_card_convergence_blockers(process, residency_device)
            if blockers:
                pinned = ", ".join(f"process {pid} holds queued model {model!r}" for pid, model in blockers)
                return SlotDutyBucket.WHOLE_CARD_CONVERGENCE, (
                    f"its model is resident and idle on process {process.process_id}, but the whole-card "
                    f"residency stuck: cannot reach sole residency because {pinned}; the convergence teardown "
                    f"should have stopped that idle sibling (only the head's holder is spared), so the shrink "
                    f"has not collapsed the pool and the head never dispatches"
                )
            return SlotDutyBucket.WHOLE_CARD_CONVERGENCE, (
                f"its model is resident and idle on process {process.process_id}, but its whole-card residency "
                f"has not yet converged to sole residency (siblings still tearing down or the device draining)"
            )

        # A head whose next dispatch must run degraded (isolated) waits for the card to clear of other work
        # rather than share it. Named here so the isolation wait does not read as an unexplained scheduler
        # stall once the concurrency gates above have not claimed it.
        if self._job_tracker.is_degraded_dispatch_pending(head):
            return SlotDutyBucket.DEGRADED_ISOLATION_PENDING, (
                f"its model is resident and idle on process {process.process_id}, but its next dispatch must run "
                f"degraded/isolated and is waiting for the card to clear of other work"
            )

        return SlotDutyBucket.UNEXPLAINED, (
            f"its model is resident and idle on process {process.process_id} but dispatch was withheld with no "
            "matching gate; this is a scheduler stall worth reporting"
        )

    def _whole_card_convergence_blockers(
        self,
        head_process: HordeProcessInfo,
        device_index: int | None,
    ) -> list[tuple[int, str]]:
        """Return idle sibling processes still holding a queued model while a whole-card head is parked.

        Returns ``(process_id, model)`` for each inference process other than the head's own holder that is
        idle (not busy), pinned to ``device_index`` when scoped, and holds a model that is still queued. The
        whole-card convergence is meant to have torn these siblings down (sparing only the head's holder), so
        finding any while the head is still parked is the fingerprint of a teardown that did not collapse the
        pool. Read-only; used only to explain a stalled dispatch.
        """
        queued_models = {
            job.model
            for job in (*self._job_tracker.jobs_pending_inference, *self._job_tracker.jobs_in_progress)
            if job.model is not None
        }
        blockers: list[tuple[int, str]] = []
        for proc in self._process_map.values():
            if proc.process_type is not HordeProcessType.INFERENCE:
                continue
            if proc.process_id == head_process.process_id:
                continue
            if device_index is not None and proc.device_index != device_index:
                continue
            if proc.is_process_busy():
                continue
            model = proc.loaded_horde_model_name
            if model is not None and model in queued_models:
                blockers.append((proc.process_id, model))
        return blockers

    def _log_dispatch_stall_if_needed(
        self,
        stable_diffusion_reference: dict[str, ImageGenerationModelRecord],
    ) -> None:
        """Emit a throttled explanation when a parked head is not dispatching despite pending work.

        Only fires once the head has been undispatched past :data:`_DISPATCH_STALL_MIN_SECONDS` (so a normal
        between-jobs gap is silent), then at most once per :data:`_DISPATCH_STALL_LOG_INTERVAL_SECONDS` for an
        unchanged reason. Read-only: it explains the stall, it does not change scheduling.
        """
        head = next(
            (j for j in self._job_tracker.jobs_pending_inference if j not in self._job_tracker.jobs_in_progress),
            None,
        )
        if head is None or self._head_starved_seconds(head) < _DISPATCH_STALL_MIN_SECONDS:
            return
        try:
            reason = self._diagnose_dispatch_stall(head, stable_diffusion_reference)
        except Exception as e:  # noqa: BLE001 - a diagnostic must never crash the scheduling cycle
            reason = f"undiagnosed ({type(e).__name__}: {e})"

        now = time.monotonic()
        if (
            reason == self._dispatch_stall_last_reason
            and (now - self._dispatch_stall_log_time) < _DISPATCH_STALL_LOG_INTERVAL_SECONDS
        ):
            return
        self._dispatch_stall_last_reason = reason
        self._dispatch_stall_log_time = now
        logger.opt(ansi=True).warning(
            f"<fg #ff8c69>Inference dispatch stalled: head {str(head.id_)[:8]} ({head.model}) has been parked "
            f"{self._head_starved_seconds(head):.0f}s: {reason}.</>",
        )

    def record_slot_duty(self, stable_diffusion_reference: dict[str, ImageGenerationModelRecord]) -> None:
        """Attribute the wall clock since the last scheduling cycle across the configured inference slots.

        Called once per scheduling cycle. Busy slots accrue ``SAMPLING``; when capacity is spare and a
        queued job is waiting, the empty slots accrue the bucket the stall classifier names (the same
        derivation that explains a parked head, but priced every tick instead of only after a multi-second
        park); with no waiting work they accrue ``NO_LOCAL_WORK``. The classification is a read-only
        diagnostic: any failure inside it degrades to ``UNEXPLAINED`` rather than touching scheduling.
        """
        capacity = max(int(self._max_concurrent_inference_processes or 0), 0)
        in_progress = self._job_tracker.jobs_in_progress
        busy = len(in_progress)
        head = next((j for j in self._job_tracker.jobs_pending_inference if j not in in_progress), None)
        waiting = len(self._job_tracker.jobs_pending_inference) - busy

        hold: SlotDutyBucket | None = None
        if head is not None and busy < capacity:
            try:
                hold = self._classify_dispatch_stall(head, stable_diffusion_reference)[0]
            except Exception:  # noqa: BLE001 - a diagnostic must never crash the scheduling cycle
                hold = SlotDutyBucket.UNEXPLAINED
        self._slot_duty_current_hold = hold

        self._slot_duty.observe(
            time.time(),
            capacity=capacity,
            busy_slots=busy,
            waiting_jobs=max(waiting, 0),
            hold=hold,
        )

    def slot_duty_snapshot(self) -> tuple[dict[str, float], int, str | None]:
        """The cumulative slot-second totals, the current capacity, and the currently-named hold bucket.

        Consumers difference successive totals for a window's breakdown (the stats stream carries the
        cumulative figures; the periodic duty log line differences its own anchor).
        """
        hold = self._slot_duty_current_hold
        return self._slot_duty.totals(), self._slot_duty.capacity, str(hold) if hold is not None else None

    def _measured_free_vram_mb(self, *, device_index: int | None = None) -> float | None:
        """Return the most conservative measured free VRAM (MB), or None when not yet reported.

        Sourced from GPU-bearing child VRAM reports via :meth:`ProcessMap.get_free_vram_mb`, which the
        children compute through hordelib's backend-agnostic accelerator layer (comfy /
        ``torch.cuda.mem_get_info``, accurate and not NVIDIA-specific). The parent stays free of any direct
        GPU query, so this works on every backend the execution layer supports.

        Args:
            device_index: When given, the free VRAM of that one card (the per-card budget on a multi-GPU
                host); when None, the most conservative figure across every card (the single-GPU reading).
        """
        return self._process_map.get_free_vram_mb(device_index=device_index)

    def _measured_available_ram_mb(self) -> float:
        """The measured system-wide available RAM (MB), read live in the parent process."""
        return psutil.virtual_memory().available / (1024 * 1024)

    def _measured_total_ram_mb(self) -> float:
        """The measured system-wide total RAM (MB), read live in the parent process."""
        return psutil.virtual_memory().total / (1024 * 1024)

    def _ram_pressure_floor_config(self) -> tuple[float, float]:
        """The configured (pause_percent, min_free_mb) for the absolute RAM danger floor, read defensively.

        Tolerant of a partially-mocked config (the scheduler unit tests): a non-numeric value falls back to
        the module default so the pressure check never crashes the scheduling cycle on a bad attribute.
        """
        bridge_data = self._runtime_config.bridge_data
        pause = config_number(bridge_data.ram_pressure_pause_percent)
        min_free = config_number(bridge_data.ram_pressure_min_free_mb)
        # Fallbacks match reGenBridgeData's defaults for these fields (85% used / 1 GB) so a partially-mocked
        # config sees the same danger floor a real worker does.
        pause_pct = pause if pause is not None else 85.0
        min_free_mb = min_free if min_free is not None else 1024.0
        return pause_pct, min_free_mb

    def _ram_pressure_verdict(self) -> RamPressureVerdict:
        """Assess whether the host is below its absolute system-RAM danger floor right now."""
        pause_pct, min_free_mb = self._ram_pressure_floor_config()
        return assess_ram_pressure(
            self._measured_available_ram_mb(),
            self._measured_total_ram_mb(),
            pause_percent=pause_pct,
            min_free_mb=min_free_mb,
        )

    @property
    def _ram_governor_state(self) -> RamGovernorState:
        """The RAM governor's multi-tick bookkeeping (owned by the resource governor)."""
        return self._governor.ram_state

    @property
    def _ram_pressure_shed_cards(self) -> set[int]:
        """Device indices the RAM-pressure reduction shed below plan (see ``RamGovernorState.shed_cards``)."""
        return self._ram_governor_state.shed_cards

    @_ram_pressure_shed_cards.setter
    def _ram_pressure_shed_cards(self, value: set[int]) -> None:
        self._ram_governor_state.shed_cards = set(value)

    @property
    def _processes_draining_for_ram(self) -> set[int]:
        """Inference process ids draining for RAM reclaim (see ``RamGovernorState.draining_process_ids``)."""
        return self._ram_governor_state.draining_process_ids

    @_processes_draining_for_ram.setter
    def _processes_draining_for_ram(self, value: set[int]) -> None:
        self._ram_governor_state.draining_process_ids = set(value)

    def _build_host_memory_snapshot(self, verdict: RamPressureVerdict) -> HostMemorySnapshot:
        """Capture the host-RAM state and governor bookkeeping one governance decision runs over.

        The single measurement site for RAM governance: every reading the pure decision functions in
        [`ram_governor`][horde_worker_regen.process_management.scheduling.governance.ram_governor] consume
        is taken here, once, so a decision never re-measures mid-flight. Config values are read
        defensively (a partially-mocked config falls back to the field default) so snapshotting never
        crashes the scheduling cycle.
        """
        margin_mb = config_number(self._runtime_config.bridge_data.ram_reserve_mb)
        if margin_mb is None:
            margin_mb = 4096.0
        inference_slots = tuple(
            InferenceSlotSnapshot(
                process_id=process_info.process_id,
                device_index=process_info.device_index,
                resident_ram_mb=process_info.ram_usage_bytes / (1024 * 1024),
                is_busy=process_info.is_process_busy(),
            )
            for process_info in self._process_map.values()
            if process_info.process_type == HordeProcessType.INFERENCE
        )
        residency_held_cards = {index for index, _residency in self._held_residencies()}
        cards = tuple(
            CardProcessSnapshot(
                device_index=device_index,
                loaded_process_count=self._process_map.num_loaded_inference_processes(device_index=device_index),
                busy_process_count=self._card_inference_load(device_index),
                planned_process_count=card_runtime.target_process_count,
                held_by_whole_card_residency=device_index in residency_held_cards,
            )
            for device_index, card_runtime in sorted(self._card_runtimes.items())
        )
        return HostMemorySnapshot(
            verdict=verdict,
            now=time.time(),
            pop_pause_active=self._state.self_throttle_paused,
            pop_pause_until=self._state.self_throttle_paused_until,
            pop_hold_margin_mb=margin_mb,
            per_process_ceiling_mb=self._ram_per_process_ceiling_mb(),
            multi_gpu_routing_active=self._multi_gpu_routing_active,
            in_flight_job_count=len(self._job_tracker.jobs_in_progress),
            loaded_worker_process_count=self._process_map.num_loaded_inference_processes(),
            planned_worker_process_count=self._max_inference_processes,
            inference_slots=inference_slots,
            cards=cards,
            draining_process_ids=frozenset(self._ram_governor_state.draining_process_ids),
            shed_card_indices=frozenset(self._ram_governor_state.shed_cards),
            restore_headroom_mb=self._ram_headroom_for_additional_context_mb(),
            per_context_ram_estimate_mb=self._estimated_resident_context_ram_mb(),
            worker_shed_planned_process_count=(
                self._ram_governor_state.worker_shed.planned_process_count
                if self._ram_governor_state.worker_shed is not None
                else None
            ),
            worker_shed_process_count=(
                self._ram_governor_state.worker_shed.shed_process_count
                if self._ram_governor_state.worker_shed is not None
                else 0
            ),
        )

    def _execute_governance_actions(self, actions: list[GovernanceAction]) -> None:
        """Execute governance decisions against the live worker: the single act site for RAM remedies.

        The governor's multi-tick bookkeeping (draining marks, shed-card tracking) is mutated here, at
        execution time and with the measured result of each remedy (a card is only recorded as shed when
        its count actually fell), so the decision layer stays a pure function of its snapshot.
        """
        governor_state = self._ram_governor_state
        for action in actions:
            match action:
                case SetPopHold(active=hold_active):
                    self._state.ram_pressure_pop_hold = hold_active
                case PausePops(
                    until_time=until_time,
                    pause_seconds=pause_seconds,
                    reason=reason,
                    available_mb=available_mb,
                    floor_mb=floor_mb,
                ):
                    prior_owner = self._state.self_throttle_pause_owner
                    pause_reason = f"host RAM pressure: {reason}"
                    self._state.self_throttle_paused = True
                    self._state.self_throttle_paused_until = until_time
                    self._state.self_throttle_pause_owner = PopPauseOwner.RAM_PRESSURE
                    self._state.self_throttle_pause_reason = pause_reason
                    self._process_lifecycle.action_ledger.record(
                        LedgerEventType.POP_PAUSE_ARMED,
                        reason=pause_reason,
                        detail={
                            "owner": PopPauseOwner.RAM_PRESSURE.value,
                            "duration_seconds": round(pause_seconds, 1),
                            "available_ram_mb": round(available_mb, 1) if available_mb is not None else None,
                            "floor_ram_mb": round(floor_mb, 1) if floor_mb is not None else None,
                        },
                    )
                    # A still-standing pause from a different backstop is only superseded here when this
                    # RAM deadline is the later one (the decision layer emits PausePops only then), so name
                    # the transition rather than silently relabelling the shared deadline.
                    takeover = (
                        f" (superseding a standing {prior_owner.value} pause)"
                        if prior_owner is not None and prior_owner is not PopPauseOwner.RAM_PRESSURE
                        else ""
                    )
                    logger.opt(ansi=True).warning(
                        f"<fg #ff8c69>System RAM below the danger floor ({reason}); pausing job pops for "
                        f"{pause_seconds:.0f}s{takeover} and shedding idle footprint so the host is not driven "
                        "into an OS OOM kill. In-flight jobs finish; pops resume once RAM recovers.</>",
                    )
                case EvictIdleModels():
                    # Unload an idle resident model; when none remains to unload, the footprint left on the
                    # host is a slot whose allocator kept the freed model's pages, which only a process cycle
                    # returns to the OS. Mirrors the preload reclaim path so sustained pressure with a drained
                    # queue still reclaims RAM instead of pinning it and holding pops forever.
                    if not self.unload_models(under_pressure=True):
                        self._replace_stale_ram_unload_process()
                case ReduceWorkerProcesses(
                    target_count=target_count,
                    planned_count=planned_count,
                    pressure_shortfall_mb=pressure_shortfall_mb,
                ):
                    current = self._process_map.num_loaded_inference_processes()
                    planned = planned_count if planned_count > 0 else self._max_inference_processes
                    after = self._process_lifecycle.scale_inference_processes(
                        target_count,
                        device_index=None,
                        pressure_shortfall_mb=pressure_shortfall_mb,
                    )
                    if not isinstance(after, int):
                        after = current
                    if after < current:
                        # The record is the live shortfall below plan, not an accumulation of reductions: a
                        # whole-card residency restore can regrow the pool between reductions, and a running
                        # total would over-count every cycle without bound while the pool is back at plan.
                        governor_state.worker_shed = WorkerProcessShedState(
                            planned_process_count=planned,
                            shed_process_count=max(0, planned - after),
                        )
                        shortfall_note = (
                            f", shortfall ~{pressure_shortfall_mb:.0f} MB" if pressure_shortfall_mb is not None else ""
                        )
                        logger.opt(ansi=True).info(
                            f"<fg #ff8c69>RAM pressure reduced worker inference contexts "
                            f"({current} -> {after} of {planned}{shortfall_note}); the pool will be "
                            "restored incrementally once RAM has headroom.</>",
                        )
                case ReduceCardProcesses(device_index=device_index, target_count=target_count):
                    current = self._process_map.num_loaded_inference_processes(device_index=device_index)
                    after = self._process_lifecycle.scale_inference_processes(
                        target_count,
                        device_index=device_index,
                    )
                    if not isinstance(after, int):
                        after = current
                    if after < current:
                        governor_state.shed_cards.add(device_index)
                case MarkProcessDraining(
                    process_id=process_id,
                    resident_ram_mb=resident_ram_mb,
                    ceiling_mb=ceiling_mb,
                ):
                    governor_state.draining_process_ids.add(process_id)
                    logger.opt(ansi=True).warning(
                        f"<fg #ff8c69>Inference process {process_id} holds {resident_ram_mb:.0f} MB RAM (>= the "
                        f"{ceiling_mb:.0f} MB per-process ceiling) while the host is under its RAM floor; "
                        "draining it (no new work) so it can be recycled once its in-flight job finishes.</>",
                    )
                case ClearProcessDraining(process_id=process_id):
                    governor_state.draining_process_ids.discard(process_id)
                case RecycleProcess(process_id=process_id, resident_ram_mb=resident_ram_mb, ceiling_mb=ceiling_mb):
                    process_info = self._process_map.get(process_id)
                    if process_info is None:
                        # The process exited between snapshot and execution; nothing to reclaim.
                        governor_state.draining_process_ids.discard(process_id)
                        continue
                    logger.opt(ansi=True).warning(
                        f"<fg #ff8c69>Inference process {process_id} holds {resident_ram_mb:.0f} MB RAM (>= the "
                        f"{ceiling_mb:.0f} MB per-process ceiling); "
                        "recycling it to return the retained RAM to the OS.</>",
                    )
                    governor_state.draining_process_ids.discard(process_id)
                    self._process_lifecycle._replace_inference_process(process_info, intentional_reclaim=True)
                    self._ram_reclaim_cycle_at = time.time()
                    self._record_churn("process_cycle")
                case RestoreCardProcess(device_index=device_index, target_count=target_count, planned_count=planned):
                    current = self._process_map.num_loaded_inference_processes(device_index=device_index)
                    after = self._process_lifecycle.scale_inference_processes(
                        target_count,
                        device_index=device_index,
                    )
                    if not isinstance(after, int):
                        after = current
                    logger.opt(ansi=True).info(
                        f"<fg #7b7d7d>System RAM has headroom; restoring an inference context on device "
                        f"{device_index} ({current} -> {after} of {planned}) so the card resumes serving.</>",
                    )
                    if after >= planned:
                        governor_state.shed_cards.discard(device_index)
                case RestoreWorkerProcess(target_count=target_count, planned_count=planned):
                    current = self._process_map.num_loaded_inference_processes()
                    after = self._process_lifecycle.scale_inference_processes(target_count, device_index=None)
                    if not isinstance(after, int):
                        after = current
                    logger.opt(ansi=True).info(
                        f"<fg #7b7d7d>System RAM has headroom; restoring a worker inference context "
                        f"({current} -> {after} of {planned}).</>",
                    )
                    if after >= planned:
                        governor_state.worker_shed = None
                    elif governor_state.worker_shed is not None and after > current:
                        governor_state.worker_shed.shed_process_count = max(
                            0,
                            governor_state.worker_shed.shed_process_count - (after - current),
                        )
                case StopTrackingShedCard(device_index=device_index):
                    governor_state.shed_cards.discard(device_index)
                case StopTrackingWorkerShed():
                    governor_state.worker_shed = None

    def _govern_ram_pressure(self, verdict: RamPressureVerdict) -> None:
        """Degrade the worker's footprint and intake while system RAM is below the danger floor.

        The proactive counterpart to the marginal RAM budget: rather than admit a load that the absolute
        reading says will trip the kernel OOM-killer, the worker pauses job pops, evicts idle resident
        models, reduces the resident inference-process count, and reclaims a process whose resident RAM
        crossed the per-process ceiling. The decision logic lives in
        [`decide_degrade_response`][horde_worker_regen.process_management.scheduling.governance.ram_governor.decide_degrade_response].
        """
        snapshot = self._build_host_memory_snapshot(verdict)
        self._execute_governance_actions(decide_degrade_response(snapshot))

    def _govern_ram_pressure_if_pressured(self) -> bool:
        """Evaluate the absolute RAM danger floor and degrade the worker if it is breached.

        The per-tick entry point (distinct from the per-job :meth:`_preload_blocked_by_ram_pressure`),
        delegated to the resource governor: it updates the soft pop hold, runs the whole-host degrade
        response whenever the host is under its floor, and restores past shedding once the host recovers,
        so a worker that never attempts a new preload still throttles and reclaims instead of growing
        into an OS OOM kill. Returns whether the host was under pressure. Clears the one-shot notice when
        the host is healthy.
        """
        under_pressure = self._governor.tick()
        if not under_pressure:
            self._ram_pressure_notified = False
        return under_pressure

    def run_governance_tick(self) -> None:
        """Drive one resource-governance tick per control-loop iteration, independent of queue depth.

        The process manager calls this every iteration so the governor's degrade/restore response and the
        soft pop hold are re-evaluated even when the inference queue is empty. Gated on the same budget
        switch as the rest of the memory machinery, which also no-ops against partial/mocked or
        early-startup config.
        """
        if self._budget_active():
            self._govern_ram_pressure_if_pressured()

    def reset_governance_to_baseline(self, reason: str) -> None:
        """Return RAM-governance state to a clean baseline, re-derived from live measurement next tick.

        Clears the soft pop hold and the governor's shed/draining episode bookkeeping, and drops the
        RAM-pressure entry from the pop-skip reasons so a stale count stops surfacing (other reasons are
        left intact). Deliberately leaves alone flags owned by other subsystems or latched for the session:
        the shared self-throttle pause (safety/self-maintenance, which self-expires), the operator
        supervisor pause, the downloads-only hold, and the post-processing / torch-compat breakers. Safe to
        call under genuine pressure: the next governance tick re-arms whatever the live host warrants.
        """
        logger.warning(f"Resetting RAM governance to baseline: {reason}")
        self._state.ram_pressure_pop_hold = False
        self._state.last_pop_skipped_reasons.pop("ram_pressure", None)
        self._ram_reclaim_cycle_at = 0.0
        self._ram_pressure_notified = False
        self._governor.reset_bookkeeping()

    def governance_healthy_but_held(self) -> bool:
        """Whether the soft RAM pop hold is engaged while the host is measurably healthy.

        The signature of a governance latch: pops are held for RAM pressure, yet the most recent
        danger-floor verdict is healthy and nothing is draining, so the hold should already have cleared.
        Distinct from a merely idle worker (which never sets the hold) and from the deliberate held-queue
        windows (whole-card establishment, heavy-head load, RAM-reclaim cycle), which own their own
        resolution. Returns False before the first tick has measured a verdict (treated as not-yet-healthy).
        Read by the recovery coordinator's healthy-hold watchdog.
        """
        if not self._state.ram_pressure_pop_hold:
            return False
        verdict = self._governor.last_ram_verdict
        if verdict is None or verdict.under_pressure:
            return False
        if self._ram_governor_state.draining_process_ids:
            return False
        return not (
            self.whole_card_residency_grace_active()
            or self.heavy_head_load_grace_active()
            or self.ram_reclaim_cycle_grace_active()
        )

    def _ram_per_process_ceiling_mb(self) -> float | None:
        """The configured per-process resident-RAM ceiling (MB), or None when disabled/unset.

        Read defensively (a partially-mocked config yields None) so the pressure path never crashes on a bad
        attribute; a non-positive value disables the ceiling.
        """
        ceiling = config_number(self._runtime_config.bridge_data.ram_per_process_max_mb)
        if ceiling is None or ceiling <= 0:
            return None
        return ceiling

    def _reduce_processes_under_ram_pressure(self) -> None:
        """Shed idle resident inference processes to return their resident-weight RAM to the OS.

        The RAM analogue of :attr:`StreamForecast.needs_process_count_reduction`: with the host over the
        danger floor, the structural remedy is fewer resident contexts, not another load on top. Only idle
        processes are stopped (``scale_inference_processes`` never kills a busy slot), so live work is
        spared. The reduction targets are decided by
        [`decide_process_reduction`][horde_worker_regen.process_management.scheduling.governance.ram_governor.decide_process_reduction]
        (per card on a multi-GPU host, worker-wide otherwise).
        """
        snapshot = self._build_host_memory_snapshot(self._ram_pressure_verdict())
        self._execute_governance_actions(decide_process_reduction(snapshot))

    def _estimated_resident_context_ram_mb(self) -> float:
        """Conservative system-RAM cost (MB) of one more resident inference context.

        Taken as the largest live inference process's measured resident RAM, which captures the model
        working set the allocator retains and will not free without a respawn. Falls back to the configured
        RAM reserve when no process has reported usage yet (only before any model has loaded; a card is only
        ever restored after a reduction that itself implies loaded, RAM-holding processes, so the measured
        value is the normal case).
        """
        live_context_ram_mb = [
            process_info.ram_usage_bytes / (1024 * 1024)
            for process_info in self._process_map.values()
            if process_info.process_type == HordeProcessType.INFERENCE and process_info.ram_usage_bytes > 0
        ]
        if live_context_ram_mb:
            return max(live_context_ram_mb)
        return self._ram_budget.reserve_mb

    def _ram_headroom_for_additional_context_mb(self) -> float:
        """Measured system-RAM headroom (MB) above the reserve and committed reserves for one more context."""
        available_ram_mb = self._measured_available_ram_mb()
        committed_ram_mb = self._reserve_ledger.total_ram_mb()
        return available_ram_mb - committed_ram_mb - self._ram_budget.reserve_mb

    def _restore_processes_after_ram_pressure(self) -> None:
        """Grow RAM-pressure-shed inference contexts back toward plan as system RAM proves headroom.

        The reduction sheds idle contexts to walk the host back above its absolute RAM floor; nothing else
        re-establishes them, so without this a card or single-GPU worker-wide pool that lost contexts to a
        RAM spike stays reduced for the rest of the run. The restore grants (incremental, RAM-gated,
        residency-aware) are decided by
        [`decide_shed_card_restore`][horde_worker_regen.process_management.scheduling.governance.ram_governor.decide_shed_card_restore].
        """
        if not self._ram_governor_state.shed_cards and self._ram_governor_state.worker_shed is None:
            return
        snapshot = self._build_host_memory_snapshot(self._ram_pressure_verdict())
        self._execute_governance_actions(decide_shed_card_restore(snapshot))

    def _committed_vram_reserve_mb(self, *, device_index: int | None = None) -> float:
        """Return the combined committed VRAM (MB) across every flow in the shared ledger.

        The dedicated post-processing lane has a fixed resident context charged by the process residency
        forecast, but each active post-processing job still registers its estimated upscaler/face-fixer peak
        here until the lane result arrives. Alchemy forms use the same ledger. Admission and
        residency-forecast gates subtract the combined figure so a freshly released slot is not handed VRAM
        concurrent work is about to claim.

        Args:
            device_index: Accepted for call-site symmetry; ledger flows are not card-attributed, so the
                worker-wide total is charged conservatively against any card.
        """
        del device_index
        return self._reserve_ledger.total_vram_mb()

    def _admission_baseline_mb(self, device_index: int | None) -> float:
        """Return the measured shared-device baseline (MB) for a card, or 0.0 when none is available.

        Falls back to 0.0 (raw-total capacity) whenever no provider is wired or none has been captured yet,
        so a cold start or a standalone unit test degrades to a capacity of the whole device total and the
        measured overlay never denies what the predictive gate admits.
        """
        if self._admission_baseline_provider is None:
            return 0.0
        baseline = self._admission_baseline_provider(device_index)
        return baseline if baseline is not None else 0.0

    def _committed_process_reserved_by_pid(self, device_index: int | None) -> dict[int, float]:
        """Return the live GPU processes' measured allocator reservation (MB) keyed by process id, for a card.

        The snapshot the planned-reserve overlay decays each entry against (a planned charge shrinks as its
        target's reservation materialises). Keyed by :attr:`HordeProcessInfo.process_id`, matching the id the
        planned entries are registered under.
        """
        reserved_by_pid: dict[int, float] = {}
        for process_info in self._process_map.values():
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_reserved_mb is None:
                continue
            reserved_by_pid[process_info.process_id] = float(process_info.process_reserved_mb)
        return reserved_by_pid

    def _in_flight_admitted_planned_units(self) -> set[str]:
        """Return the loading-process ids (as ledger units) whose admitted VRAM has not yet materialised.

        The authoritative in-flight-admitted set, derived from the model map rather than any parallel registry
        so it cannot leak: a model is counted while its load state is ``LOADING`` or ``LOADED_IN_RAM``, since a
        just-admitted preload sits in ``LOADED_IN_RAM`` before its allocator reservation grows, and two loads
        admitted the same cycle must both keep their planned charge until then (the double-admit guard). Once
        the model is ``IN_USE`` or its process finishes, faults, or dies, that unit stops appearing here and the
        next :meth:`CommittedReserveLedger.reconcile_planned` drops its planned charge by omission.

        ``LOADED_IN_RAM`` is overloaded: it is both the pre-materialisation state of a fresh preload and the
        state a resident model returns to when it is evicted from VRAM back to system RAM, so this set alone
        cannot tell a materialised-then-evicted anchor apart from one still in flight. The ledger closes that
        gap directly: :meth:`CommittedReserveLedger.effective_planned_vram_mb` consumes each anchor
        monotonically, so an anchor whose reservation has already grown stays consumed regardless of the load
        state it later revisits. Keyed by process id to match the unit :meth:`_send_preload` registers each
        grant under.

        A planned charge only survives while its target process could still materialise it. A model-map entry
        that still reads ``LOADING`` on a process that has since died or entered its terminal shutdown states can
        outlive the (throttled, once-per-cooldown) missing-process recovery that expires it, and a dead target's
        reservation never grows, so its charge would otherwise decay by neither materialisation nor omission and
        pin the overlay at full weight indefinitely; a head re-asking that same load then finds its own stale
        planned charge holding the card against it, a self-deadlock the identity cannot escape. Excluding process
        ids that are absent from the process map or in a terminal state drops such a charge here, through the
        same reconcile-by-omission that
        releases a finished load, with no separate death-path delete to keep in sync. Mirrors the committed
        ledger's own exclusion of ending/ended tenants, so the two overlays agree on which processes are live.
        """
        units: set[str] = set()
        for model_info in self._horde_model_map.root.values():
            if model_info.process_id is None:
                continue
            if model_info.horde_model_load_state not in (ModelLoadState.LOADING, ModelLoadState.LOADED_IN_RAM):
                continue
            process_info = self._process_map.get(model_info.process_id)
            if process_info is None:
                continue
            if process_info.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED):
                continue
            units.add(str(model_info.process_id))
        return units

    def _in_flight_dispatch_units(self, device_index: int | None) -> set[str]:
        """Return the in-progress job ids (as reservation units) whose dispatch reservation is still live.

        A dispatch reservation protects an admitted job's activation-inclusive peak until it materialises over
        the sampling window the device-free reading does not yet reflect. The authoritative live set is the
        job tracker's in-progress jobs (``INFERENCE_IN_PROGRESS``): a job that finalises, faults, or whose
        process dies leaves that set, and the next :meth:`CommittedReserveLedger.reconcile_planned` drops its
        reservation by omission with no death-path delete to keep in sync. On a multi-GPU host only jobs
        dispatched to ``device_index`` are counted, matching the per-card reservation view; a single-GPU
        (``device_index`` None) call counts every in-progress job.
        """
        jobs = (
            self._jobs_in_progress_on_card(device_index)
            if self._multi_gpu_routing_active and device_index is not None
            else self._job_tracker.jobs_in_progress
        )
        return {str(job.id_) for job in jobs if job.id_ is not None}

    def _measured_admission_candidate_delta_mb(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: str | None,
        *,
        process_id: int | None,
        disaggregated: bool,
    ) -> float | None:
        """Return the candidate job's marginal predicted VRAM (MB) for the measured overlay, net of resident credit.

        The gross charge is the sampler-only figure for a disaggregation-class job, else the whole-job sampling
        peak, matching how the predictive gate prices the same job, then raised by any learned watermark for the
        job's (baseline, resolution, platform) footprint at the matching stage (SAMPLE_ISOLATED for the
        disaggregated sampler-only figure, SAMPLE for the whole-job peak) so the measured floor is never priced
        below a peak the hardware has already demonstrated. When the job's model is already resident in
        its target process its weights are already in the measured committed floor, so they are credited out
        (the same resident-credit reasoning the retention gate applies) to avoid re-charging them. None (no
        estimate) propagates so the overlay treats the candidate as a zero marginal, never denying on an
        unpriceable cost.
        """
        static_gross_mb = (
            predict_job_sampler_only_vram_mb(job, baseline)
            if disaggregated
            else predict_job_sampling_vram_mb(job, baseline)
        )
        if static_gross_mb is None:
            return None
        gross_mb = self._learned_sampling_peak_mb(
            job,
            baseline,
            static_seed_mb=static_gross_mb,
            stage=FootprintStage.SAMPLE_ISOLATED if disaggregated else FootprintStage.SAMPLE,
        )
        resident_credit_mb = 0.0
        if self._candidate_weights_resident_on_process(job.model, process_id):
            resident_credit_mb = predict_job_weight_mb(job, baseline) or 0.0
        return max(0.0, gross_mb - resident_credit_mb)

    def _candidate_weights_resident_on_process(self, model_name: str | None, process_id: int | None) -> bool:
        """Whether ``model_name``'s weights already occupy VRAM on ``process_id`` (dispatch materialises nothing).

        The single residency truth two admission concerns share: the resident-weight credit that keeps a
        candidate delta from re-charging weights the committed floor already counts, and the arbiter's
        ``candidate_already_resident`` no-op admit. Read primarily from the horde model map's residency state on
        the matching process. The committed floor charges those weights by the process's own measured
        reservation, keyed by the process map's ``loaded_horde_model_name``; when the model map's process pointer
        transiently lags that record the two disagree, so a fallback also credits residency when the target
        process itself reports this model loaded and the model map agrees the model is VRAM-resident. Aligning
        the credit with the floor's own truth stops the divergence from double-charging resident weights (once
        in the committed floor, again as the candidate delta) and wedging a dispatch to an idle resident model.
        """
        if model_name is None or process_id is None:
            return False
        model_info = self._horde_model_map.root.get(model_name)
        model_map_says_vram_resident = model_info is not None and model_info.horde_model_load_state in (
            ModelLoadState.LOADED_IN_VRAM,
            ModelLoadState.IN_USE,
        )
        if model_info is not None and model_info.process_id == process_id and model_map_says_vram_resident:
            return True
        process_info = self._process_map.get(process_id)
        return (
            process_info is not None
            and process_info.loaded_horde_model_name == model_name
            and model_map_says_vram_resident
        )

    def set_vram_arbiter(self, arbiter: VramArbiter) -> None:
        """Inject the single VRAM arbiter: the preload-admission authority and the observational overlay elsewhere."""
        self._vram_arbiter = arbiter

    def set_device_free_mb_provider(self, provider: Callable[[int], float | None]) -> None:
        """Inject the truthful per-card device-free reading source (the parent's NVML view).

        The manager-driven cycle passes its explicit reading map into :meth:`build_vram_arbiter_snapshot`;
        this provider covers the self-primed path (a scheduler consult before or outside a manager tick), so
        the measured-truth admission identity keeps its primary input on every snapshot the scheduler builds.
        """
        self._device_free_mb_provider = provider

    def set_footprint_store(self, store: LearnedFootprintStore) -> None:
        """Inject the shared learned-footprint store the message dispatcher also observes into.

        Admission pricing of a job's sampling peak consults it so a measured activation high-water raises the
        static per-model seed; one instance is shared across the parent so every observed peak and every priced
        estimate reference the same watermarks.
        """
        self._footprint_store = store

    def _sampling_footprint_key(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: str | None,
        *,
        stage: FootprintStage,
    ) -> FootprintKey | None:
        """Build the footprint key for ``job`` at ``stage``, or None when it cannot be keyed.

        The key is (baseline, resolution bucket by the job's maximum dimension, host platform, stage). The stage
        distinguishes a whole-job monolithic peak (:attr:`FootprintStage.SAMPLE`) from a disaggregated UNet-only
        sampler peak (:attr:`FootprintStage.SAMPLE_ISOLATED`): the two are physically different quantities and,
        since watermarks are raise-only, must not share a key. A None baseline or an absent width/height cannot
        be attributed to a footprint population, so it returns None and the caller keeps the static seed.
        """
        if baseline is None:
            return None
        width = job.payload.width
        height = job.payload.height
        if width is None or height is None:
            return None
        return FootprintKey(
            model_baseline=str(baseline),
            resolution_bucket=ResolutionBucket.from_dimensions(width, height, job.payload.n_iter or 1),
            platform=sys.platform,
            stage=stage,
        )

    def _learned_sampling_peak_mb(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: str | None,
        *,
        static_seed_mb: float,
        stage: FootprintStage,
    ) -> float:
        """Raise a static sampling-peak seed by any learned watermark for this job's ``stage`` footprint key.

        The static predictor stays the seed; the learned overlay can only ever RAISE it (a cold key, a None
        baseline, or an unkeyable job returns the seed unchanged). This is the single seam admission pricing of
        sampling work routes through so a measured activation peak is never undershot. Callers pricing whole-job
        sampling pass :attr:`FootprintStage.SAMPLE`; callers pricing a disaggregated UNet-only sampler pass
        :attr:`FootprintStage.SAMPLE_ISOLATED` so a monolithic whole-job watermark never over-prices it.
        """
        store = self._footprint_store
        if store is None:
            return static_seed_mb
        key = self._sampling_footprint_key(job, baseline, stage=stage)
        if key is None:
            return static_seed_mb
        return store.estimate_mb(key, static_seed_mb=static_seed_mb)

    def observe_disaggregated_sampling_peak(self, job_info: HordeJobInfo, peak_reserved_mb: float) -> None:
        """Fold a disaggregated sampler's measured peak into the store under this job's SAMPLE_ISOLATED key.

        The message dispatcher observes the monolithic case under :attr:`FootprintStage.SAMPLE`, but a
        disaggregated UNet-only sampler's peak arrives through the orchestrator (which alone binds the pinned
        sampler process to the job's stage), so this is the seam that closes that gap. It records under
        :attr:`FootprintStage.SAMPLE_ISOLATED`, a distinct key from the monolithic whole-job peak: mixed
        operation is designed (a stage fault re-routes a disaggregated job monolithic), so a single monolithic
        peak must not raise the isolated-sampler estimate and forfeit the second concurrent sampler. The peak
        figure is the pinned sampler process's latest reported ``process_peak_reserved_mb`` at sample completion:
        it is the allocator high-water since the process's previous memory report, so it can lag the true
        sampling peak by up to one report interval, but it is the best-attributable reading at this seam.
        Raise-only semantics apply (a non-positive reading is ignored by the store); a store-less, unkeyable, or
        model-less job is a no-op.
        """
        store = self._footprint_store
        if store is None:
            return
        job = job_info.sdk_api_job_info
        if job.model is None:
            return
        key = self._sampling_footprint_key(
            job,
            self._model_metadata.get_baseline(job.model),
            stage=FootprintStage.SAMPLE_ISOLATED,
        )
        if key is None:
            return
        store.observe_peak(key, peak_reserved_mb)

    def _gpu_process_activity_ids(self, device_index: int | None) -> tuple[frozenset[int], frozenset[int]]:
        """Return the idle and busy GPU-process ids on a card, for the arbiter's release-cache targeting.

        A RELEASE_CACHE target is an idle GPU process that plausibly still holds *reclaimable* allocator
        cache: its measured reservation exceeds its in-use allocation by at least
        ``_RELEASE_CACHE_MIN_RECLAIMABLE_MB``, so an ``empty_cache`` could return that reserved-but-unallocated
        margin to the card. A process whose reservation is its resident footprint (a component/VAE/post-process
        lane holding encoders or a still-loaded model, where allocated tracks reserved) has no such margin and
        is not a target: asking it to release frees nothing, and emitting a rung that can never yield would keep
        the escalation ladder non-empty forever. An idle process that still holds a horde model is an eviction
        candidate (a distinct ladder rung), not a cache-release target, so it is left out here to keep the two
        remedies separate. A busy process is never a target. When the in-use allocation has not yet been
        reported the reclaimable margin cannot be measured, so the process is not targeted rather than assumed
        to hold cache.
        """
        cache_bearing = (
            HordeProcessType.INFERENCE,
            HordeProcessType.POST_PROCESS,
            HordeProcessType.VAE_LANE,
            HordeProcessType.COMPONENT,
        )
        idle: set[int] = set()
        busy: set[int] = set()
        for process_info in self._process_map.values():
            if process_info.process_type not in cache_bearing:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.is_process_busy():
                busy.add(process_info.process_id)
                continue
            if process_info.loaded_horde_model_name is not None:
                continue
            reserved_mb = process_info.process_reserved_mb
            allocated_mb = process_info.process_allocated_mb
            if reserved_mb is None or allocated_mb is None:
                continue
            if reserved_mb - allocated_mb >= _RELEASE_CACHE_MIN_RECLAIMABLE_MB:
                idle.add(process_info.process_id)
        return frozenset(idle), frozenset(busy)

    def build_vram_arbiter_device_state(
        self,
        device_index: int | None,
        *,
        active_sampling_peaks_total_mb: float = 0.0,
        governor_state: GovernorState | None = None,
        device_free_mb: float | None = None,
        reclaim_unresolved: bool = False,
    ) -> DeviceVramState:
        """Assemble the frozen per-device VRAM measurement the arbiter prices this cycle's requests against.

        Sourced entirely from figures the scheduler already holds: the measured-truth admission identity's
        primary input (the frozen device-free reading, passed in) plus the outstanding reservations and the
        noise buffer, and the concurrent-sampling headroom's terms (baseline, fixed and marginal context
        overhead, the live context counts, the operator reserve, the lane decode spike). The committed floor
        and staleness are still assembled for diagnostics and telemetry but the admission path no longer reads
        them. No NVML read and no torch import; the measurement is the parent's already-reconciled state.

        Both admission-reservation flows are reconciled by omission before the outstanding total is read: a
        preload whose process finished, faulted, or died, or an in-progress job that finalised or died, drops
        its reservation here so a re-ask is never blocked by a dead unit's stale reservation.
        """
        raw_total_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        baseline_mb = self._admission_baseline_mb(device_index)
        committed_mb = self._process_map.committed_vram_mb(
            context_constant_mb=self.resolved_context_constant_mb(),
            device_index=device_index,
        )
        oldest_report_age = self._process_map.oldest_committed_report_age_seconds(
            now=time.time(),
            device_index=device_index,
        )
        committed_is_stale = oldest_report_age is not None and oldest_report_age > _REPORT_STALENESS_SECONDS
        self._reserve_ledger.reconcile_planned(PRELOAD_ADMISSION_FLOW, self._in_flight_admitted_planned_units())
        self._reserve_ledger.reconcile_planned(DISPATCH_ADMISSION_FLOW, self._in_flight_dispatch_units(device_index))
        per_process_reserved = self._committed_process_reserved_by_pid(device_index)
        planned_mb = self._reserve_ledger.effective_planned_vram_mb(per_process_reserved)
        noise_buffer_mb = admission_noise_buffer_mb(raw_total_mb)
        self._admission_headroom_mb_by_device[device_index if device_index is not None else 0] = (
            None if device_free_mb is None else device_free_mb - planned_mb - noise_buffer_mb
        )

        override_mb = self._config_overhead_override_mb()
        per_process_mb = self._overhead.per_process_mb(config_override_mb=override_mb)
        marginal_mb = self._overhead.marginal_mb(config_override_mb=override_mb)
        if marginal_mb is None or marginal_mb <= 0:
            marginal_mb = _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB
        bridge_data = self._runtime_config.bridge_data
        safety_contexts = (
            1 if bridge_data.safety_on_gpu is True and not self._process_lifecycle.is_safety_gpu_paused else 0
        )
        post_process_contexts = (
            0
            if self._process_lifecycle.is_post_process_gpu_paused
            else self._process_map.num_post_process_processes(device_index=device_index)
        )
        vae_lane_contexts = (
            0
            if self._process_lifecycle.is_vae_lane_gpu_paused
            else self._process_map.num_vae_lane_processes(device_index=device_index)
        )
        idle_process_ids, busy_process_ids = self._gpu_process_activity_ids(device_index)
        return DeviceVramState(
            total_vram_mb=raw_total_mb,
            baseline_mb=baseline_mb,
            committed_vram_mb=committed_mb,
            planned_unmaterialized_mb=planned_mb,
            committed_is_stale=committed_is_stale,
            noise_buffer_mb=noise_buffer_mb,
            per_process_reserved_mb=per_process_reserved,
            idle_process_ids=idle_process_ids,
            busy_process_ids=busy_process_ids,
            num_loaded_inference_processes=self._process_map.num_loaded_inference_processes(
                device_index=device_index,
            ),
            safety_context_count=safety_contexts,
            safety_reclaim_allowed=(
                self._residency_should_pause_safety(device_index) and not self._has_safety_backlog()
            ),
            post_process_context_count=post_process_contexts,
            vae_lane_context_count=vae_lane_contexts,
            per_process_overhead_mb=per_process_mb,
            marginal_mb=marginal_mb,
            vram_reserve_mb=self._vram_budget.reserve_mb,
            vae_lane_decode_spike_mb=self._vae_lane_decode_spike_charge_mb(device_index=device_index),
            active_sampling_peaks_total_mb=active_sampling_peaks_total_mb,
            governor_state=governor_state,
            device_free_mb=device_free_mb,
            reclaim_unresolved=reclaim_unresolved,
        )

    def build_vram_arbiter_snapshot(
        self,
        *,
        active_sampling_peaks_total_mb: float = 0.0,
        governor_states: Mapping[int, GovernorState] | None = None,
        device_free_mb_by_device: Mapping[int, float] | None = None,
        reclaim_unresolved_by_device: Mapping[int, bool] | None = None,
    ) -> MeasuredVramSnapshot:
        """Build the whole-worker frozen snapshot for one cycle, one entry per driven card.

        The single-GPU/worker-wide reading (device index None) is stored under card 0, matching how the
        scheduler keys its per-card admission counters, so a None-keyed request resolves to it. The parent
        supplies ``governor_states`` (the device-free governor's committed state per card), the truthful NVML
        ``device_free_mb_by_device`` reading, and ``reclaim_unresolved_by_device`` (whether the verified
        reclaim ladder has exhausted itself while SATURATED per card) so each device state carries the
        admission inputs. A missing readings map falls back to the injected device-free provider (see
        :meth:`set_device_free_mb_provider`) so a self-primed snapshot keeps the identity's primary input;
        with neither source the reading is absent and admission defers on it.
        """
        device_indices = {process_info.device_index for process_info in self._process_map.values()}
        if not device_indices:
            device_indices = {0}
        devices: dict[int, DeviceVramState] = {}
        for device_index in sorted(device_indices):
            governor_state = governor_states.get(device_index) if governor_states is not None else None
            if device_free_mb_by_device is not None:
                device_free_mb = device_free_mb_by_device.get(device_index)
            elif self._device_free_mb_provider is not None:
                device_free_mb = self._device_free_mb_provider(device_index)
            else:
                device_free_mb = None
            reclaim_unresolved = (
                reclaim_unresolved_by_device.get(device_index, False)
                if reclaim_unresolved_by_device is not None
                else False
            )
            devices[device_index] = self.build_vram_arbiter_device_state(
                device_index if self._multi_gpu_routing_active else None,
                active_sampling_peaks_total_mb=active_sampling_peaks_total_mb,
                governor_state=governor_state,
                device_free_mb=device_free_mb,
                reclaim_unresolved=reclaim_unresolved,
            )
        return MeasuredVramSnapshot(devices=devices)

    def _overlap_memory_verdict(
        self,
        candidate_job: ImageGenerateJobPopResponse,
        *,
        target_device_index: int | None,
    ) -> bool | None:
        """The arbiter's answer to ``candidate_job``'s overlap memory demand: admits, withholds, or unpriced.

        Prices the candidate's marginal device cost against the cycle-frozen admission floor as a
        :attr:`VramRequestKind.MONOLITHIC_DISPATCH`: True when a FITS verdict admits, False when a DEFER or
        DENY withholds. Returns None when the demand cannot be priced at all (the arbiter is unwired,
        no cycle snapshot is frozen, or the candidate is model-less): the caller then relaxes the memory answer
        to admit, matching the predictive gate's admit-on-missing-telemetry contract, without treating the
        absence of telemetry as positive confirmation of room. A disaggregation-class candidate is priced with
        its sampler-only delta (``disaggregated``), so the concurrent decode spike the sampling gate already
        reserves is never double-counted here. No actuations run: the arbiter verdict's actuations are ignored
        because reclaim stays single-owner (the preload path drives it).
        """
        arbiter = self._vram_arbiter
        if arbiter is None or not arbiter.has_cycle or candidate_job.model is None:
            return None
        baseline = self._model_metadata.get_baseline(candidate_job.model)
        resident_model_info = self._horde_model_map.root.get(candidate_job.model)
        resident_pid = resident_model_info.process_id if resident_model_info is not None else None
        request = VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label=str(candidate_job.model),
            baseline=baseline,
            device_index=target_device_index,
            target_process_id=resident_pid,
            candidate_delta_mb=self._measured_admission_candidate_delta_mb(
                candidate_job,
                baseline,
                process_id=resident_pid,
                disaggregated=self._is_disaggregation_class_eligible(candidate_job),
            ),
        )
        return arbiter.evaluate(request).admits

    def _concurrent_overlap_allowed(
        self,
        candidate_job: ImageGenerateJobPopResponse,
        *,
        target_device_index: int | None = None,
    ) -> bool:
        """Whether ``candidate_job`` may start while other jobs are already sampling, the arbiter deciding memory.

        The concurrency cap (``max_threads``) only counts in-flight jobs; it does not look at what those jobs
        are, how far along they are, or how much the card can hold. This gate adds both missing dimensions: a
        temporal/structural guard that keeps a newcomer off a running job's memory-hungry startup beat, and the
        VRAM arbiter's authoritative answer to whether the card can hold the overlap at all.

        The non-memory guards run first and can decline overlap on their own:
            * The first job (nothing in flight) always starts: with no overlap there is no memory question.
            * An extra-large (whole-card tier) model neither joins a busy card nor shares one, whatever the
              card's headroom; that contract is the tier's, not the card's.
            * A newcomer must let the running job make size-appropriate sampling headway (none for light+light,
              a startup beat when a memory-hungry pairing has room, the strictest headway behind a batch) before
              it joins, so two loads and activation peaks do not stack into a step-timeout teardown.

        The memory question is then the arbiter's: a :attr:`VramRequestKind.MONOLITHIC_DISPATCH` verdict that
        FITS admits the overlap, a DEFER or DENY withholds it this cycle and the dispatch re-asks
        naturally on the next scheduling pass. This seam runs no actuations on a DEFER (reclaim is single-owner,
        driven only by the preload path); a cold start or an unwired arbiter relaxes the memory answer to admit.

        The headway relaxation is driven by positive confirmation only: a heavy pairing's headway drops to the
        startup-beat constant (and a batch is bounded by the strictest headway rather than a hard block) only
        when the arbiter has actually confirmed room this cycle. A cold start (no cycle) keeps the strict
        headway fractions, since the admit-on-missing-telemetry relaxation is not evidence the card has room.

        A blocked job is not dropped: it keeps its queue position and dispatches once the in-flight job(s)
        progress or finish and the card has room.

        Args:
            candidate_job: The job being considered for dispatch.
            target_device_index: On a multi-GPU host, the card this candidate would run on; the headway check
                then considers only jobs already sampling on that same card (jobs on other cards do not contend
                for its VRAM or sampler), and the arbiter prices the demand against that card. ``None`` (and
                every single-GPU call) keeps the worker-wide comparison.
        """
        if self._multi_gpu_routing_active and target_device_index is not None:
            in_progress_jobs: tuple[ImageGenerateJobPopResponse, ...] | list[ImageGenerateJobPopResponse] = (
                self._jobs_in_progress_on_card(target_device_index)
            )
        else:
            in_progress_jobs = self._job_tracker.jobs_in_progress
        if not in_progress_jobs:
            return True

        candidate_tier = self._model_size_tier(candidate_job.model)
        if candidate_tier >= _ModelSizeTier.EXTRA_LARGE:
            return False

        # The memory question is the arbiter's, resolved once and only when a rule needs it. ``memory_ample``
        # is the arbiter's positive confirmation of room (a real cycle that admits), which relaxes the headway;
        # ``memory_admits`` is the veto, which withholds only when a real cycle denies and relaxes to admit when
        # the demand could not be priced. A cold start therefore keeps the strict headway yet admits on memory.
        memory_verdict_cache: bool | None = None
        memory_evaluated = False

        def memory_verdict() -> bool | None:
            nonlocal memory_verdict_cache, memory_evaluated
            if not memory_evaluated:
                memory_verdict_cache = self._overlap_memory_verdict(
                    candidate_job,
                    target_device_index=target_device_index,
                )
                memory_evaluated = True
            return memory_verdict_cache

        def memory_admits() -> bool:
            return memory_verdict() is not False

        def memory_ample() -> bool:
            return memory_verdict() is True

        candidate_batched = self._job_batch_amount(candidate_job) > 1
        if candidate_batched and not memory_ample():
            return False

        # Higher performance modes pull a newcomer's sampling into the current job's tail sooner by shrinking
        # the headway it must wait for; the arbiter's memory verdict below still independently gates the overlap.
        headway_scale = _performance_mode_headway_scale(self._runtime_config.bridge_data)

        for job in in_progress_jobs:
            running_tier = self._model_size_tier(job.model)
            if running_tier >= _ModelSizeTier.EXTRA_LARGE:
                return False

            if candidate_batched or self._job_batch_amount(job) > 1:
                # A batch multiplies the activation peak, so without confirmed room it keeps the hard block;
                # with room it is bounded instead by the strictest headway (never the startup-beat relaxation,
                # which is sized for single jobs).
                if not memory_ample():
                    return False
                required_headway = _OVERLAP_HEADWAY_BOTH_HEAVY
            else:
                required_headway = self._required_overlap_headway(running_tier, candidate_tier)
                if required_headway > 0.0 and memory_ample():
                    required_headway = _OVERLAP_HEADWAY_AMPLE_VRAM

            required_headway *= headway_scale

            if required_headway <= 0.0:
                continue
            if self._in_flight_progress_fraction(job) < required_headway:
                return False

        return memory_admits()

    def set_vram_growth_hold(self, device_index: int, active: bool) -> None:
        """Set or clear the device-free governor's growth hold for a card (parent control loop only).

        While ``active`` the scheduler withholds every action that would grow the card's VRAM footprint: a new
        model preload onto a process that does not already hold it, a safety GPU restore, and a paused-lane
        restart. In-flight sampling is untouched. Called each governor tick, so the hold tracks the card's
        live proximity-to-cliff state.
        """
        self._vram_growth_hold_by_device[device_index] = active

    def is_vram_growth_held(self, device_index: int | None) -> bool:
        """Whether the device-free governor is holding new VRAM growth on a card (default card 0 for None).

        The single-GPU/worker-wide key (None) maps to card 0, matching the arbiter snapshot and admission
        counters. False for any card the governor has not yet sampled (no hold), so a host without NVML never
        holds growth.
        """
        return self._vram_growth_hold_by_device.get(device_index if device_index is not None else 0, False)

    def set_governor_state(self, device_index: int, state: GovernorState) -> None:
        """Record the device-free governor's committed state for a card (parent control loop only).

        Pushed each governor tick alongside :meth:`set_vram_growth_hold`. Retention reads this state (not the
        derived hold) so a resident is only kept while the card is HEALTHY.
        """
        self._governor_states_by_device[device_index] = state

    def governor_state(self, device_index: int | None) -> GovernorState:
        """Return the device-free governor's committed state for a card (default card 0 for None).

        The single-GPU/worker-wide key (None) maps to card 0, matching the growth hold and the arbiter snapshot.
        HEALTHY for any card the governor has not yet sampled, so a host without NVML never denies retention on
        governor grounds.
        """
        return self._governor_states_by_device.get(
            device_index if device_index is not None else 0, GovernorState.HEALTHY
        )

    def reclaim_one_idle_model_under_pressure(self, *, device_index: int | None = None) -> bool:
        """Reclaim idle resident VRAM under a physical-overcommit pressure signal (no specific loading candidate).

        The no-candidate counterpart of the WDDM-paging rising-edge reclaim: one under-pressure sweep of idle
        resident models on the card, reusing :meth:`unload_models_from_vram`'s target selection and its existing
        protections (busy, in-progress, queued-lookahead, lane, and pinned processes are all skipped). The
        anchor passed is a process to protect (a busy inference process if one exists, else any inference
        process on the card); the sweep evicts the coldest eligible idle resident, never a live model. Returns
        True when an unload was issued.
        """
        anchor = self._pressure_reclaim_anchor(device_index=device_index)
        if anchor is None:
            return False
        return self.unload_models_from_vram(anchor, under_pressure=True, device_index=device_index)

    def recalibrate_committed_ledger(self, *, device_index: int | None = None) -> int:
        """Recalibrate the committed-VRAM ledger to device truth by releasing every idle lane's allocator cache.

        The committed ledger sums each lane's ``memory_reserved()``. That per-process figure can detach upward
        from device reality: an unloaded model's blocks the torch caching allocator has not returned, or a
        reservation the WDDM driver already spilled to host RAM, both keep counting against committed while the
        physical pages are free. Such a phantom over-count cannot be cured by evicting a model (there is nothing
        resident to evict) and would otherwise defer every admission forever on a figure the card does not hold.

        Emptying an idle lane's allocator cache returns those blocks and prompts a fresh memory report, so the
        parent's ``process_reserved_mb`` (hence committed) converges back to device truth. A busy (actively
        sampling) lane is skipped: its reservation is live, and its cache returns through the ordinary post-stage
        path. Terminal (ending/ended) lanes and lanes that have never reported a reservation are skipped too, as
        is any process whose dispatch contract does not include ``RELEASE_ALLOCATOR_CACHE`` (the fan-out targets
        only :data:`ALLOCATOR_CACHE_CAPABLE_PROCESS_TYPES`, so a routing-incapable process is never asked).
        Returns how many lanes were asked to release.

        Args:
            device_index: When given, recalibrate only lanes pinned to that card; when None, every card.
        """
        lanes_asked = 0
        for process_info in self._process_map.values():
            if process_info.process_type not in ALLOCATOR_CACHE_CAPABLE_PROCESS_TYPES:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_reserved_mb is None:
                continue
            if process_info.last_process_state in (
                HordeProcessState.PROCESS_ENDING,
                HordeProcessState.PROCESS_ENDED,
            ):
                continue
            if process_info.is_process_busy():
                continue
            if self.release_allocator_cache(process_info.process_id):
                lanes_asked += 1
        return lanes_asked

    def _pressure_reclaim_anchor(self, *, device_index: int | None) -> HordeProcessInfo | None:
        """Pick the process to protect for a no-candidate pressure reclaim: a busy inference process, else any.

        The reclaim sweep excludes its anchor, so anchoring on a busy inference process keeps a live job's
        model safe while the sweep evicts the coldest idle resident. Falls back to any inference process on the
        card (its own model is still protected by the in-progress guard) and to None when the card has none.
        """
        fallback: HordeProcessInfo | None = None
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if fallback is None:
                fallback = process_info
            if process_info.is_process_busy():
                return process_info
        return fallback

    def _max_jobs_in_progress_allowed(
        self,
        *,
        card: CardRuntime | None = None,
    ) -> int:
        """The cap on concurrently in-progress jobs for this scheduling decision.

        Without the GPU sampling lease, the inference semaphore is the sole denoise gate, so this
        is the concurrent-sampling count; dispatching more would over-subscribe the GPU. With the
        lease enabled, the lease (not this cap) limits actual concurrent sampling, so spare
        inference processes are allowed to receive jobs and stage their pipeline (model load,
        prompt encode) *ahead* while others sample, filling the inter-job gaps where the GPU
        would otherwise go dark. That pre-staging is permitted up to the full inference-process
        count, but only while there is enough free VRAM to hold another staged model; otherwise it
        falls back to the sampling-slot cap so speculation never over-commits the device.

        Args:
            card: When the worker drives more than one card, the card this decision is scoped to: its
                own sampling-slot and process ceilings are used so the big card's spare threads never
                inflate a small card's allowance. ``None`` keeps the worker-wide global ceilings, which
                is exactly the single-GPU case (byte-identical to before). The free-VRAM staging
                headroom is measured worker-wide either way (a deliberate conservatism until per-card
                memory-report attribution is implemented).
        """
        # An exclusively-admitted over-budget job needs the whole device; never dispatch another job
        # alongside it. Returning the current in-progress count (floored at 1 so the exclusive job itself
        # can still be dispatched when none is yet running) blocks any *additional* concurrent dispatch.
        # This stays worker-wide even in the per-card path: the over-budget exclusive admission is itself
        # whole-worker today, so an exclusive job suppresses dispatch on every card until it clears.
        if self._job_tracker.has_exclusive_job_in_progress():
            return max(1, len(self._job_tracker.jobs_in_progress))

        if card is not None:
            concurrent_ceiling = card.max_concurrent_inference
            process_ceiling = card.target_process_count
        else:
            concurrent_ceiling = self._max_concurrent_inference_processes
            process_ceiling = self._max_inference_processes

        base = concurrent_ceiling
        if not self._runtime_config.bridge_data.gpu_sampling_lease_enabled:
            return base

        # Floor the speculative-staging headroom at the budget's reserve when it is active, so
        # pre-staging never eats into the VRAM the budget is holding back for the in-flight job's
        # transient spikes; otherwise keep the standalone staging threshold.
        staging_floor = _SPECULATIVE_DISPATCH_MIN_FREE_VRAM_MB
        if self._budget_active():
            staging_floor = max(staging_floor, self._vram_budget.reserve_mb)

        free_vram_mb = self._process_map.get_free_vram_mb()
        if free_vram_mb is None or free_vram_mb < staging_floor:
            return base

        return process_ceiling

    def _model_size_tier(self, model_name: str | None) -> _ModelSizeTier:
        """Classify a model by how much of the device its inference is expected to want.

        Resolves the model's baseline from the loaded reference and delegates to the shared, torch-free
        :func:`~horde_worker_regen.process_management.models.model_sizing.model_size_tier`, so this and the
        popper's large-model pop limiters classify "very large" from the same single source of truth.
        """
        baseline = self._model_metadata.get_baseline(model_name) if model_name is not None else None
        baseline_value = baseline.value if isinstance(baseline, KNOWN_IMAGE_GENERATION_BASELINE) else baseline
        return model_size_tier(model_name, baseline_value)

    @staticmethod
    def _job_batch_amount(job: ImageGenerateJobPopResponse) -> int:
        """The batch size (``n_iter``) of a job, floored at 1 for malformed values."""
        n_iter = job.payload.n_iter
        return n_iter if isinstance(n_iter, int) and n_iter > 0 else 1

    def _process_running_job(self, job: ImageGenerateJobPopResponse) -> HordeProcessInfo | None:
        """The inference process currently dispatched the given in-flight job, if any.

        Matches on the slot's ``last_job_referenced`` (stamped at dispatch) so the overlap gate can read
        the running job's live step progress.
        """
        job_id = job.id_
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            referenced = process_info.last_job_referenced
            if referenced is not None and referenced.id_ == job_id:
                return process_info
        return None

    def _in_flight_progress_fraction(self, job: ImageGenerateJobPopResponse) -> float:
        """How far along the in-flight job's sampling is, in ``[0.0, 1.0]``.

        A freshly dispatched job that has not yet reported a step reads as ``0.0`` (the slot's progress
        fields are unset), which is exactly when a heavy overlap is most dangerous.
        """
        process_info = self._process_running_job(job)
        if process_info is None:
            return 0.0

        total_steps = process_info.last_total_steps
        current_step = process_info.last_current_step
        if total_steps is not None and total_steps > 0 and current_step is not None:
            return max(0.0, min(1.0, current_step / total_steps))

        percent_complete = process_info.last_heartbeat_percent_complete
        if percent_complete is not None:
            return max(0.0, min(1.0, percent_complete / 100.0))

        return 0.0

    @staticmethod
    def _required_overlap_headway(running_tier: _ModelSizeTier, candidate_tier: _ModelSizeTier) -> float:
        """Progress the running job must have made before a candidate joins it concurrently.

        Only called once both jobs are known to be non-extra-large and non-batched (those are hard
        blocks handled earlier). Two light jobs thread together freely; any pairing involving a heavy
        job requires headway, and two heavy jobs require the most.
        """
        if running_tier <= _ModelSizeTier.LIGHT and candidate_tier <= _ModelSizeTier.LIGHT:
            return 0.0
        if running_tier >= _ModelSizeTier.HEAVY and candidate_tier >= _ModelSizeTier.HEAVY:
            return _OVERLAP_HEADWAY_BOTH_HEAVY
        return _OVERLAP_HEADWAY_MIXED_HEAVY

    def _jobs_in_progress_on_card(self, device_index: int) -> list[ImageGenerateJobPopResponse]:
        """The in-progress jobs whose live inference process is pinned to ``device_index``.

        Cards are independent sampling/VRAM domains, so the per-card concurrency gates compare a candidate
        only against the jobs sharing its card. A job whose running slot cannot be identified is omitted (it
        is attributed to no card), which only ever relaxes the per-card count, never inflates it.
        """
        on_card: list[ImageGenerateJobPopResponse] = []
        for job in self._job_tracker.jobs_in_progress:
            running_process = self._process_running_job(job)
            if running_process is not None and running_process.device_index == device_index:
                on_card.append(job)
        return on_card

    def _expire_stale_model_map_entries(self) -> list[str]:
        """Expire model-map entries whose owning process can no longer be loading that model."""
        expired: list[str] = []
        loading_owner_states = {
            HordeProcessState.PROCESS_STARTING,
            HordeProcessState.DOWNLOADING_MODEL,
            HordeProcessState.PRELOADING_MODEL,
            HordeProcessState.DOWNLOADING_AUX_MODEL,
            HordeProcessState.DOWNLOAD_AUX_COMPLETE,
            HordeProcessState.UNLOADED_MODEL_FROM_RAM,
        }

        now = time.time()

        for model_name, model_info in list(self._horde_model_map.root.items()):
            process_info = self._process_map.get(model_info.process_id)
            if process_info is None:
                self._horde_model_map.expire_entry(model_name)
                expired.append(model_name)
                logger.warning(
                    f"Expiring stale model-map entry for {model_name}: process {model_info.process_id} is gone.",
                )
                continue

            recent_preload_request = (
                model_info.horde_model_load_state == ModelLoadState.LOADING
                and process_info.last_control_flag == HordeControlFlag.PRELOAD_MODEL
                and process_info.loaded_horde_model_name == model_name
                and (now - process_info.last_preload_requested_at) <= _PRELOAD_FIRST_REPORT_GRACE_SECONDS
            )
            if (
                model_info.horde_model_load_state == ModelLoadState.LOADING
                and process_info.last_process_state not in loading_owner_states
                and not recent_preload_request
            ):
                self._horde_model_map.expire_entry(model_name)
                expired.append(model_name)
                logger.warning(
                    f"Expiring stale loading entry for {model_name} on process {process_info.process_id}: "
                    f"process is {process_info.last_process_state.name}.",
                )

        return expired

    def _replace_stale_ram_unload_process(self) -> bool:
        """Cycle an idle inference process that did not actually release RAM after a RAM-unload request."""
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.is_process_busy():
                continue
            if process_info.loaded_horde_model_name is not None:
                continue
            if process_info.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
                continue
            if process_info.ram_usage_bytes < _STALE_RAM_UNLOAD_REPLACE_BYTES:
                continue

            logger.warning(
                f"Idle process {process_info.process_id} still holds {process_info.ram_usage_bytes} bytes "
                "after a RAM unload (the allocator retains the freed model's pages); cycling it to return "
                "the RAM to the OS.",
            )
            # A deliberate reclaim of a healthy idle slot, not a crash/hang: keep it out of the crash
            # bookkeeping (recovery count + crash-loop breaker) so sustained RAM pressure cannot
            # quarantine a perfectly healthy slot.
            self._process_lifecycle._replace_inference_process(process_info, intentional_reclaim=True)
            # Open the bounded reclaim-cycle grace: the slot now respawns and the next head must preload
            # onto it, a window in which the queue is unservable by the worker's own deliberate action, not
            # a wedge. ram_reclaim_cycle_grace_active() reads this so the recovery supervisor does not
            # soft-reset the pools and fault the servable backlog mid-reclaim.
            self._ram_reclaim_cycle_at = time.time()
            self._record_churn("process_cycle")
            return True

        return False

    def _preload_blocked_by_ram_pressure(self, job: ImageGenerateJobPopResponse) -> bool:
        """Return whether the host's absolute RAM danger floor forces this preload to defer.

        When system RAM is below its danger floor, governs the pressure (sheds idle footprint, pauses
        pops) and reports True so the caller defers rather than routing a new model's weights through a
        host already on the edge. Clears the one-shot notice and reports False when RAM is healthy.
        """
        # One scheduling cycle acts on one consistent reading: reuse the verdict the governor's tick
        # measured at the top of this cycle rather than re-measuring per job. Tests (and any path that
        # reaches here before a first tick) fall back to a live reading.
        ram_pressure = self._governor.last_ram_verdict
        if ram_pressure is None:
            ram_pressure = self._ram_pressure_verdict()
        if not ram_pressure.under_pressure:
            self._ram_pressure_notified = False
            return False
        # The governor's tick (run once per control-loop iteration via run_governance_tick) has already
        # driven the whole-host degrade response this cycle; here we only defer *this* preload and surface
        # the per-model notice once so the loop does not route a new model's weights through a host already
        # on the edge.
        if not self._ram_pressure_notified:
            logger.opt(ansi=True).warning(
                f"<fg #ff8c69>RAM danger floor reached: deferring preload of {job.model} "
                f"({ram_pressure.reason()}). Shedding idle footprint and pausing pops.</>",
            )
            self._ram_pressure_notified = True
        return True

    def _send_preload(self, job: ImageGenerateJobPopResponse, available_process: HordeProcessInfo) -> bool:
        """Send the preload command for ``job``'s model to ``available_process`` and record the load.

        Resets the preload-delay and head-starvation trackers, sends the PRELOAD_MODEL message inside a
        telemetry span, and on a successful send records the churn/ledger entry and advances the model map
        and process map into the LOADING state. Returns True (a preload was issued this cycle).
        """
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")

        self._preload_delay_notified = False
        self._clear_head_starvation_timer()
        logger.debug(f"Preloading model {job.model} on process {available_process.process_id}")
        logger.debug(f"Available inference processes: {self._process_map}")
        only_active_models = {
            model_name: model_info
            for model_name, model_info in self._horde_model_map.root.items()
            if model_info.horde_model_load_state.is_active()
        }
        logger.debug(f"Horde model map (active): {only_active_models}")

        will_load_loras = job.payload.loras is not None and len(job.payload.loras) > 0
        seamless_tiling_enabled = job.payload.tiling is not None and job.payload.tiling

        # A swap is a preload that displaces a *different* model already resident on this process;
        # that prior model's load work is thrown away. A fresh slot (None) or re-preload of the same
        # model is not churn. Captured before the send so the process's prior model is still readable.
        prior_model = available_process.loaded_horde_model_name
        is_model_swap = prior_model is not None and prior_model != job.model

        with span_preload_model(model_name=job.model, process_id=available_process.process_id):
            preload_sent = available_process.safe_send_message(
                HordePreloadInferenceModelMessage(
                    control_flag=HordeControlFlag.PRELOAD_MODEL,
                    horde_model_name=job.model,
                    will_load_loras=will_load_loras,
                    seamless_tiling_enabled=seamless_tiling_enabled,
                    sdk_api_job_info=job,
                    aux_download_deadline_seconds=self._process_lifecycle.aux_download_deadline_for_dispatch(
                        self._runtime_config.bridge_data,
                    ),
                ),
            )

        if preload_sent:
            available_process.last_control_flag = HordeControlFlag.PRELOAD_MODEL
            available_process.last_preload_requested_at = time.time()
            if is_model_swap:
                self._record_churn("model_swap")
            self._process_lifecycle.action_ledger.record(
                LedgerEventType.PRELOAD_REQUESTED,
                process_id=available_process.process_id,
                os_pid=available_process.os_pid,
                launch_identifier=available_process.process_launch_identifier,
                job_id=str(job.id_) if job.id_ is not None else None,
                detail={"model": job.model},
            )

            self._horde_model_map.update_entry(
                horde_model_name=job.model,
                load_state=ModelLoadState.LOADING,
                process_id=available_process.process_id,
            )

            model_baseline = self._model_metadata.get_baseline(job.model)

            self._process_map.on_model_load_state_change(
                process_id=available_process.process_id,
                horde_model_name=job.model,
                horde_model_baseline=model_baseline,
                last_job_referenced=job,
            )

            # Record the grant into the planned overlay the moment the load is admitted, so a second admission
            # in this same scheduling cycle (before the per-cycle reconcile runs) sees this charge and cannot
            # over-admit against the same measured floor. The charge is the candidate delta actually priced for
            # this preload (sampler-only or whole-job, net of any resident credit); the admit-time reservation
            # baseline is the target's reserved right now, so the charge decays one-for-one as this process's
            # own reservation materialises the load. The per-cycle reconcile then prunes it once the process
            # leaves the loading set (finished, faulted, or dead), with no explicit release on those paths.
            planned_charge_mb = self._measured_admission_candidate_delta_mb(
                job,
                model_baseline,
                process_id=available_process.process_id,
                disaggregated=self._is_disaggregation_class_eligible(job),
            )
            self._reserve_ledger.set_planned(
                PRELOAD_ADMISSION_FLOW,
                str(available_process.process_id),
                vram_mb=planned_charge_mb if planned_charge_mb is not None else 0.0,
                target_process_id=available_process.process_id,
                reserved_at_admit_mb=float(available_process.process_reserved_mb or 0),
            )

        return True

    def _decide_whole_card_demand(
        self,
        job: ImageGenerateJobPopResponse,
        available_process: HordeProcessInfo,
        forecast: StreamForecast,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        *,
        is_head_blocker: bool,
        target_device_index: int | None,
    ) -> _WholeCardDemandOutcome:
        """Decide whether the head should claim the whole card and drive the residency side effects.

        Whole-card exclusive residency (preventative): the forecast judges whether loading this model
        alongside the currently-resident models would drive the device into weight streaming. A heavy model
        loaded while others stay resident across sibling processes can collapse free VRAM to near zero, at
        which point ComfyUI offloads weights or the driver's system-memory fallback spills per-step
        activations; both stream over the bus and the slow job risks being mistaken for a hang and killed.
        When the model would stream co-resident but fits with the card to itself, it is given sole residency
        before it loads: marked exclusive (so ``has_exclusive_job_in_progress`` suppresses other staging),
        then enough VRAM is freed. The forecast distinguishes two remedies, applying the least-disruptive:
        evicting sibling *models* (their processes stay up), or stopping idle sibling *processes* when their
        fixed per-process contexts are themselves the over-commit (a context is only reclaimed by exit).

        Only the head may claim the card: reserving it tears down the siblings serving the lighter heads
        ahead of a deeper-queue job, so a non-head heavy job returns ``FALL_THROUGH`` and defers via the
        ordinary verdict until it becomes the head. See :class:`_WholeCardDemandOutcome` for each result.
        """
        # A disaggregation-class job never demands exclusive device residency: it runs as a UNet-only sampler
        # whose sampler-only footprint co-resides with the encode lane and other samplers by design. Coupled
        # with sampler-only charging in the forecast, this breaks the loop where a whole-card window (which
        # pauses the lane) would otherwise flip the job to its monolithic footprint and re-demand the card,
        # starving the encode lane. Decided on class-eligibility, not liveness, so the contract holds even
        # while the lane is transiently paused.
        if self._is_disaggregation_class_eligible(job):
            return _WholeCardDemandOutcome.FALL_THROUGH
        # A model needs the teardown path either because it is weight-dominant (needs sole residency) or
        # because the live sibling process contexts have squeezed its bounded weights off the card though it
        # co-resides once the process count is reduced. Both are served by the same machinery: establish
        # residency, stop idle siblings down to max_resident_processes, and admit once the weights fit.
        whole_card_demanded = self._whole_card_ledger.residency_demanded(
            forecast,
            enabled=self._whole_card_residency_enabled(),
            is_head_blocker=is_head_blocker,
        )
        if not whole_card_demanded:
            return _WholeCardDemandOutcome.FALL_THROUGH
        if not self._whole_card_warranted(forecast):
            # The teardown demand is not trustworthy (a card-light model on a host with no measured
            # per-context cost): decline the reservation and fall through to ordinary eviction rather than
            # reserving the device on an over-counted-context phantom.
            self._log_whole_card_declined(job, forecast)
            return _WholeCardDemandOutcome.FALL_THROUGH

        first_time = not self._job_tracker.is_admitted_exclusive(job)
        self._job_tracker.mark_admitted_exclusive(job)
        if self._should_prestage_whole_card_head(
            job,
            baseline,
            forecast,
            available_process,
            device_index=target_device_index,
        ):
            # A live job still holds the device, but the heavy head's weights can begin loading into a spare
            # process's RAM right now: preload_model is a RAM-only load (weights move to VRAM at sampling
            # time), so it does not contend with the in-flight job's VRAM. Record the residency and send the
            # preload; _converge_whole_card_residency then collapses the live process count to sole VRAM
            # residency before the staged model samples. The heavy disk->RAM load overlaps the in-flight job
            # instead of waiting for the device to drain first.
            self._begin_whole_card_residency(
                job,
                forecast,
                announce=first_time,
                device_index=target_device_index,
            )
            return _WholeCardDemandOutcome.PRESTAGE

        # Claim the device: stop idle siblings to the model's max-resident count and, on the very edge, move
        # safety off-GPU too. Announces (once) why, for the operator. Held through the cooldown so a burst of
        # heavy jobs reuses one residency instead of churning per job.
        self._establish_whole_card_residency(
            job,
            forecast,
            announce=first_time,
            device_index=target_device_index,
        )
        # Evict the idle resident models on the *other* processes (sparing the slot that will load this
        # model, and never a live in-progress model) so their VRAM returns to the driver. A live sibling is
        # left to drain; the preload simply waits until the device is clear.
        self.unload_models_from_vram(
            available_process,
            under_pressure=True,
            for_head_of_queue=True,
            device_index=target_device_index,
        )
        if not self._whole_card_teardown_exhausted(forecast, device_index=target_device_index):
            # Still tearing down idle siblings, cycling safety off-GPU, or waiting for their freed VRAM to
            # drain: defer and let a later tick re-evaluate against the reduced topology.
            return _WholeCardDemandOutcome.DEFER
        # Teardown is structurally exhausted (already at the target process count, safety settled). The card is
        # now cleared to sole residency, so the head's weights are priced against a drained card: fall through
        # to the measured arbiter evaluation, which admits when the weights fit the cleared card (the
        # activation peak is the sampling gate's concern, not preload admission) and denies when even the
        # cleared card cannot hold them (an unserviceable model the offering seam should have excluded).
        return _WholeCardDemandOutcome.FALL_THROUGH

    def _reclaim_ram_for_overbudget_admit(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    ) -> None:
        """Reclaim idle system RAM ahead of an over-budget classified load, only when the host is short.

        A heavy head loads its checkpoint through system RAM before it reaches the device, so an admit on
        a RAM-tight host must first evict an idle resident copy. On a host with ample available RAM that
        eviction buys nothing and costs a warm cache: the sibling's model drops to disk, and the next job
        for it pays a full checkpoint reload (with the allocator-stuck slot the unload leaves behind then
        recycled, compounding the churn). So the reclaim is gated on the RAM budget's own verdict for the
        incoming load rather than performed unconditionally.
        """
        ram_verdict = self._ram_budget.check_job(
            job,
            baseline,
            self._measured_available_ram_mb(),
            committed_reserve_mb=self._reserve_ledger.total_ram_mb(),
        )
        if ram_verdict.fits:
            return
        self.unload_models(under_pressure=True, for_head_of_queue=True)

    def _apply_ram_verdict(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        *,
        is_head_blocker: bool,
        no_live_resource_consumer: bool,
    ) -> bool:
        """Apply the system-RAM budget verdict for a preload: reclaim idle RAM or best-effort admit.

        When the predicted RAM cost fits, returns True immediately. Otherwise runs the reclaim attempts
        (gentle eviction, escalated for the head, then cycling an allocator-stuck idle slot) and
        dispatches on
        [`decide_ram_reclaim_outcome`][horde_worker_regen.process_management.scheduling.governance.preload_admission.decide_ram_reclaim_outcome]:
        reclaim progress is always worth waiting for, and only a head-of-queue blocker with no live job
        holding memory is admitted best-effort once nothing more can be reclaimed.
        """
        ram_verdict = self._ram_budget.check_job(
            job,
            baseline,
            self._measured_available_ram_mb(),
            committed_reserve_mb=self._reserve_ledger.total_ram_mb(),
        )
        if ram_verdict.fits:
            self._ram_budget_defer_notified = False
            return True

        if not self._ram_budget_defer_notified:
            logger.opt(ansi=True).warning(
                f"<fg #f0beff>RAM budget deferring preload of {job.model}: "
                f"{ram_verdict.reason()}. Reclaiming idle RAM.</>",
            )
            self._ram_budget_defer_notified = True
        reclaimed = self.unload_models(under_pressure=True)
        if not reclaimed and is_head_blocker:
            # Gentle reclaim freed nothing; for the head of the queue, escalate to reclaim a queued
            # model's RAM before falling back to cycling an allocator-stuck idle slot.
            reclaimed = self.unload_models(under_pressure=True, for_head_of_queue=True)
        cycled = False if reclaimed else self._replace_stale_ram_unload_process()

        outcome = decide_ram_reclaim_outcome(
            reclaimed=reclaimed,
            cycled_stale_slot=cycled,
            is_head_blocker=is_head_blocker,
            no_live_resource_consumer=no_live_resource_consumer,
        )
        if outcome is RamReclaimOutcome.DEFER:
            return False

        logger.opt(ansi=True).warning(
            f"<fg #f0beff>RAM budget cannot fit head-of-queue model {job.model} even after "
            "reclaiming all idle RAM, and no live job holds memory; admitting it best-effort "
            "rather than wedging the queue.</>",
        )
        return True

    def _admit_preload_under_budget(
        self,
        job: ImageGenerateJobPopResponse,
        available_process: HordeProcessInfo,
        *,
        is_head_blocker: bool,
    ) -> bool:
        """Return whether ``job`` may be admitted for preload, with the VRAM arbiter as the deciding authority.

        True means proceed to send the preload; False means defer this cycle. The whole-card residency state
        machine is consulted first (pre-stage, defer, or fall through once its teardown has cleared the card to
        sole residency). Otherwise the arbiter prices the preload against the frozen cycle measurement: a FITS
        admits and runs the RAM verdict, and a DEFER (or the structural-impossibility DENY) runs the described
        pressure-relief actuations so the over-commit is relieved before the request re-asks next cycle. There
        is no overcommit-admit path: a head that never becomes admittable while the device is idle is rerouted by the
        structural-queue-wedge recovery supervisor. Every decision is scoped to the card this preload would land
        on (None keeps the worker-wide reading on a single-GPU host).
        """
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")

        baseline = self._model_metadata.get_baseline(job.model)
        target_device_index = available_process.device_index if self._multi_gpu_routing_active else None
        # A head waiting behind live work is queued, not starved. With no live job holding this card, the
        # starved-seconds value feeds the arbiter diagnostic and the RAM branch can still decide that exhausted
        # system-RAM reclaim should proceed.
        if target_device_index is None:
            no_live_resource_consumer = len(self._job_tracker.jobs_in_progress) == 0
        else:
            no_live_resource_consumer = len(self._jobs_in_progress_on_card(target_device_index)) == 0

        forecast = self._forecast_streaming(job, baseline, device_index=target_device_index)
        # Trace the forecast for every budget-gated load so the logs show the residency dynamics, not just the
        # action taken. Unchanged observations are coalesced by _log_stream_forecast.
        self._log_stream_forecast(job, forecast)

        whole_card = self._decide_whole_card_demand(
            job,
            available_process,
            forecast,
            baseline,
            is_head_blocker=is_head_blocker,
            target_device_index=target_device_index,
        )
        if whole_card is _WholeCardDemandOutcome.DEFER:
            return False
        if whole_card is _WholeCardDemandOutcome.PRESTAGE:
            # A RAM-only pre-stage of a whole-card head: the VRAM budget deliberately does not fit it
            # co-resident (that is *why* it gets the whole card), so skip the verdict and send the preload.
            return True

        arbiter = self._ensure_preload_arbiter()

        # Price the predictive verdict once: it sources the candidate delta and the rejected peak the
        # context-reduction remedy is sized from.
        vram_verdict = self._vram_budget.check_job(
            job,
            baseline,
            self._measured_free_vram_mb(device_index=target_device_index),
            committed_reserve_mb=self._committed_vram_reserve_mb(device_index=target_device_index),
            disaggregated=self._is_disaggregation_class_eligible(job),
        )
        max_resident, can_reduce_live_contexts = self._context_reduction_demand(
            vram_verdict,
            forecast,
            is_head_blocker=is_head_blocker,
            target_device_index=target_device_index,
        )
        has_reclaimable_idle_model = self._has_reclaimable_idle_model(
            available_process,
            for_head_of_queue=is_head_blocker,
            device_index=target_device_index,
        )
        idle_contexts_teardownable = is_head_blocker and self._has_teardownable_idle_context(
            available_process,
            device_index=target_device_index,
        )
        request = self._build_preload_request(
            job,
            available_process,
            baseline,
            target_device_index=target_device_index,
            is_head_blocker=is_head_blocker,
            has_reclaimable_idle_model=has_reclaimable_idle_model,
            can_reduce_live_contexts=can_reduce_live_contexts,
            idle_contexts_teardownable=idle_contexts_teardownable,
        )
        verdict = arbiter.evaluate(request)

        if verdict.disposition is VramDisposition.FITS:
            self._vram_budget_defer_notified = False
            # A FITS is a real fit against the truthful device-free reading (the identity already accounts for
            # baseline and foreign load, which are physically inside that reading). The model still loads
            # through system RAM, so the marginal RAM verdict runs.
            return self._apply_ram_verdict(
                job,
                baseline,
                is_head_blocker=is_head_blocker,
                no_live_resource_consumer=no_live_resource_consumer,
            )

        # DEFER (or the structural-impossibility DENY, treated identically): run the described pressure-relief
        # commands so the over-commit is relieved, then defer. There is no overcommit admit: a head that never
        # becomes admittable while the device is idle is rerouted by the structural-queue-wedge recovery
        # supervisor, and the arbiter emits a starvation diagnostic naming the arithmetic before then.
        # Completion is observed via the next cycle's frozen snapshot; the actuations are never awaited inline.
        if not self._vram_budget_defer_notified and not vram_verdict.fits:
            logger.opt(ansi=True).warning(
                f"<fg #f0beff>VRAM arbiter deferring preload of {job.model}: {verdict.reason}. "
                "Reclaiming idle VRAM.</>",
            )
            self._vram_budget_defer_notified = True
        self._preload_actuation = _PreloadActuation(
            job=job,
            available_process=available_process,
            forecast=forecast,
            max_resident=max_resident,
        )
        try:
            self._execute_preload_actuations(
                verdict.required_actuations,
                device_index=target_device_index,
                for_head_of_queue=is_head_blocker,
            )
        finally:
            self._preload_actuation = None
        return False

    def _ensure_preload_arbiter(self) -> VramArbiter:
        """Return the preload-admission arbiter, priming a private one with the current measurement if unwired.

        In the running worker the manager injects the shared arbiter and freezes its cycle once per
        control-loop iteration, so this returns that instance with the tick's snapshot intact. A scheduler
        exercised on its own (no manager tick, no injected cycle) gets a private arbiter primed with a
        freshly-built snapshot, so admission stays fully governed rather than falling to an ungoverned path.
        """
        arbiter = self._vram_arbiter
        if arbiter is None:
            arbiter = VramArbiter()
            self._vram_arbiter = arbiter
        if not arbiter.has_cycle:
            arbiter.begin_cycle(self.build_vram_arbiter_snapshot())
        return arbiter

    def _build_preload_request(
        self,
        job: ImageGenerateJobPopResponse,
        available_process: HordeProcessInfo,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        *,
        target_device_index: int | None,
        is_head_blocker: bool,
        has_reclaimable_idle_model: bool,
        can_reduce_live_contexts: bool,
        idle_contexts_teardownable: bool,
    ) -> VramRequest:
        """Assemble the arbiter request for one preload, priced identically to the measured admission overlay."""
        return VramRequest(
            kind=VramRequestKind.PRELOAD,
            job_label=str(job.model),
            baseline=baseline,
            device_index=target_device_index,
            target_process_id=available_process.process_id,
            candidate_delta_mb=self._measured_admission_candidate_delta_mb(
                job,
                baseline,
                process_id=available_process.process_id,
                disaggregated=self._is_disaggregation_class_eligible(job),
            ),
            candidate_already_resident=self._candidate_weights_resident_on_process(
                job.model,
                available_process.process_id,
            ),
            own_planned_unmaterialized_mb=self._own_planned_charge_mb(
                device_index=target_device_index,
                target_process_id=available_process.process_id,
            ),
            is_head_of_queue=is_head_blocker,
            starved_seconds=self._head_starved_seconds(job),
            has_reclaimable_idle_model=has_reclaimable_idle_model,
            can_reduce_live_contexts=can_reduce_live_contexts,
            idle_contexts_teardownable=idle_contexts_teardownable,
        )

    def _own_planned_charge_mb(self, *, device_index: int | None, target_process_id: int | None) -> float:
        """Return the planned-overlay charge (MB) attributable to a request's own target process.

        The arbiter subtracts this from the device's planned overlay so a re-ask nets out the load it itself
        admitted on an earlier cycle (the candidate delta already represents it), preventing the head-of-queue
        self-deadlock where a load's own not-yet-materialised plan holds the card against its re-ask. Every
        other process's planned charge is left intact, so genuinely-concurrent admissions still count in full.
        """
        if target_process_id is None:
            return 0.0
        return self._reserve_ledger.planned_charge_for_unit(
            PRELOAD_ADMISSION_FLOW,
            str(target_process_id),
            self._committed_process_reserved_by_pid(device_index),
        )

    def _record_dispatch_reservation(
        self,
        job: ImageGenerateJobPopResponse,
        process_info: HordeProcessInfo,
        *,
        baseline: str | None,
    ) -> None:
        """Register the outstanding reservation for a dispatch the moment it is sent inference.

        A dispatch onto an already-resident model materialises the job's activation-inclusive peak (net of the
        resident-weight credit the model already holds) over the sampling window the device-free reading does
        not yet reflect. Recording that peak as a reservation keyed by the job id, targeting the sampling
        process and baselined at the process's current reservation, means a second admission in the same window
        sees it and cannot over-admit into the same physical room; the reservation decays one-for-one as this
        process's own reservation materialises the activation, and drops by omission once the job leaves the
        in-progress set (finalised, faulted, or process-dead). An unpriceable job reserves nothing rather than
        pinning the overlay at a fabricated figure.
        """
        if job.id_ is None:
            return
        charge_mb = self._measured_admission_candidate_delta_mb(
            job,
            baseline,
            process_id=process_info.process_id,
            disaggregated=False,
        )
        self._reserve_ledger.set_planned(
            DISPATCH_ADMISSION_FLOW,
            str(job.id_),
            vram_mb=charge_mb if charge_mb is not None else 0.0,
            target_process_id=process_info.process_id,
            reserved_at_admit_mb=float(process_info.process_reserved_mb or 0),
        )

    def release_dispatch_reservation(self, job: ImageGenerateJobPopResponse) -> None:
        """Drop a dispatch's outstanding reservation on clean finalization (a latency tightener over omission).

        Reconcile-by-omission already releases a dispatch reservation the next cycle a job leaves the
        in-progress set, so this is not the correctness guarantee: it only shortens the window between a job
        completing and its reservation clearing, so the freed room is available to the next admission sooner.
        Idempotent and safe for a job that never reserved (no-op).
        """
        if job.id_ is None:
            return
        self._reserve_ledger.release(DISPATCH_ADMISSION_FLOW, str(job.id_))

    def _displaced_head_outstanding_mb(
        self,
        displaced_head: ImageGenerateJobPopResponse,
        *,
        device_index: int | None,
    ) -> float | None:
        """Return the head-of-queue demand (MB) a line-skipper must not consume, or None when unpriceable.

        The head a line-skip jumped is still downloading, so its weights are not yet resident: its outstanding
        demand is its full priced candidate (no resident-weight credit). Head protection uses this to hold the
        skipper when admitting it would leave the card short of the room the head needs. None (an unpriceable
        head, or a model-less one) skips the protection rather than fabricating a figure, degrading to admitting
        the skipper.
        """
        if displaced_head.model is None:
            return None
        baseline = self._model_metadata.get_baseline(displaced_head.model)
        return self._measured_admission_candidate_delta_mb(
            displaced_head,
            baseline,
            process_id=None,
            disaggregated=self._is_disaggregation_class_eligible(displaced_head),
        )

    def _context_reduction_demand(
        self,
        vram_verdict: BudgetVerdict,
        forecast: StreamForecast,
        *,
        is_head_blocker: bool,
        target_device_index: int | None,
    ) -> tuple[int | None, bool]:
        """Return the head's context-reduction target and whether reducing live contexts is a warranted remedy.

        A moderate head's weights fit after a model eviction but its activation peak does not while this many
        contexts are live (each extra context retains VRAM the allocator never returns). Reducing the live
        inference-process count to the largest that still seats the rejected peak plus its structural reserve
        is the remedy. The depth keys on the honest streaming floor, not the operator's configured margin, so
        only a genuinely card-filling peak pushes the co-resident count below the live pool; a demand resting
        on untrusted (unmeasured-fallback) overhead figures is not warranted.
        """
        max_resident: int | None = None
        if vram_verdict.predicted_mb is not None:
            total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=target_device_index)
            structural_reserve_mb = (
                effective_inference_reserve_mb(total_vram_mb, 0.0)
                if total_vram_mb is not None
                else vram_verdict.reserve_mb
            )
            max_resident = self._max_coresident_for_peak_mb(
                vram_verdict.predicted_mb,
                structural_reserve_mb,
                device_index=target_device_index,
            )
        context_reduction_demanded = (
            self._whole_card_residency_enabled()
            and is_head_blocker
            and max_resident is not None
            and self._process_map.num_loaded_inference_processes(device_index=target_device_index) > max_resident
        )
        can_reduce = context_reduction_demanded and self._whole_card_warranted(forecast)
        return max_resident, can_reduce

    def _has_reclaimable_idle_model(
        self,
        process_with_model: HordeProcessInfo,
        *,
        for_head_of_queue: bool,
        device_index: int | None,
    ) -> bool:
        """Return whether an idle resident model could be evicted on the card to reclaim VRAM for this head.

        A read-only mirror of the eviction targeting :meth:`unload_models_from_vram` performs under pressure:
        a post-processing lane not already unloading, or an inference process holding a model that is not in
        progress, not spared by the queued-lookahead or residency guards (both of which the head escalation
        overrides), and not already unloading. It excludes the head's own target slot and never counts an
        in-progress model. When this is False, and no idle cache and no warranted context reduction remain,
        reclamation is structurally exhausted for this head.
        """
        wanted_models = self._compute_wanted_models()
        next_n_models = list(self.get_next_n_models(self._max_inference_processes))
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}
        for process_info in self._process_map.values():
            if process_info.process_id == process_with_model.process_id:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_type == HordeProcessType.POST_PROCESS:
                if process_info.is_process_busy():
                    continue
                if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                    continue
                return True
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.loaded_horde_model_name is None:
                continue
            # The single-model unload guard (skip when only one model is configured) applies only when the
            # reclaim is not under pressure; every preload reclaim here is under pressure, so it never spares.
            if process_info.loaded_horde_model_name in in_progress_models:
                continue
            if (
                process_info.loaded_horde_model_name in next_n_models
                and not for_head_of_queue
                and self._coresident_lookahead_affordable(
                    process_info.loaded_horde_model_name,
                    device_index=device_index,
                )
            ):
                continue
            if not for_head_of_queue and self._residency_protects_from_unload(
                process_info.loaded_horde_model_name,
                wanted_models,
                vram=True,
                under_pressure=True,
            ):
                continue
            if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                continue
            return True
        return False

    def _has_teardownable_idle_context(
        self,
        head_process: HordeProcessInfo,
        *,
        device_index: int | None,
    ) -> bool:
        """Return whether an idle sibling inference context could be torn down to reclaim VRAM for a starved head.

        A bare CUDA context's VRAM is reclaimed only when its process exits, which weight eviction (model
        unload, cache release) cannot achieve. An idle inference process on the card other than the head's own
        target slot, not busy and not serving an in-progress job, is a teardown candidate the starvation
        escalation can reduce via :meth:`reduce_live_contexts`. Excludes the head's target slot and every busy
        process, matching that actuator's own protections.

        This is independent of ``whole_card_exclusive_residency``: that flag governs whether the worker
        establishes exclusive residency as a steady-state preference, but the starvation escalation is an
        emergency liveness path (a head starved past the arbiter's threshold whose own idle contexts hold the
        deficit) that must be reachable regardless. The actuation runs through :meth:`reduce_live_contexts` ->
        :meth:`_establish_whole_card_residency` -> ``scale_inference_processes``, none of which gate on the flag,
        so tearing the idle contexts down proceeds when the flag is off.
        """
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.process_id == head_process.process_id:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.is_process_busy():
                continue
            if (
                process_info.loaded_horde_model_name is not None
                and process_info.loaded_horde_model_name in in_progress_models
            ):
                continue
            return True
        return False

    def _execute_preload_actuations(
        self,
        commands: tuple[ActuatorCommand, ...],
        *,
        device_index: int | None,
        for_head_of_queue: bool,
    ) -> None:
        """Run the pressure-relief commands a deferred preload verdict described, at most once each this cycle.

        RELEASE_CACHE returns an idle lane's cached allocator reservation to the card; EVICT_IDLE_MODEL frees
        an idle resident model's weights; REDUCE_LIVE_CONTEXTS collapses the live inference-process count so a
        retained per-context reservation returns; CYCLE_SAFETY_OFF_GPU frees the safety context. The arbiter
        guarantees RELEASE_CACHE targets only idle lanes, so a busy lane is never asked to release its cache.

        The command dispatch is routed through :meth:`VerifiedReclaimLadder.execute_arbiter_commands` so this
        DEFER path and the governor's SATURATED verified ladder share one reclaim execution surface (the
        single-owner rule): the two triggers can never become two mechanisms evicting the same card by
        different rules.
        """
        VerifiedReclaimLadder.execute_arbiter_commands(
            commands,
            self,
            device_index=device_index,
            for_head_of_queue=for_head_of_queue,
        )

    def release_cache(self, process_id: int) -> bool:
        """Return an idle lane's cached allocator reservation to the device (:class:`VramActuator`)."""
        return self.release_allocator_cache(process_id)

    def evict_idle_model(self, device_index: int | None, *, for_head_of_queue: bool) -> bool:
        """Evict an idle resident model on the card to reclaim its weights (:class:`VramActuator`).

        The head being admitted keeps its own target slot: the eviction protects that process and never
        touches a live in-progress model.
        """
        actuation = self._preload_actuation
        anchor = (
            actuation.available_process
            if actuation is not None
            else self._pressure_reclaim_anchor(device_index=device_index)
        )
        if anchor is None:
            return False
        return self.unload_models_from_vram(
            anchor,
            under_pressure=True,
            for_head_of_queue=for_head_of_queue,
            device_index=device_index,
        )

    def reduce_live_contexts(self, device_index: int | None) -> bool:
        """Reduce the live inference-context count for the current head (:class:`VramActuator`).

        Establishes whole-card residency for the head at the depth the rejected peak sized, then evicts the
        idle residents on the other processes so their contexts' retained VRAM returns to the card. A no-op
        when no head-preload context is recorded.
        """
        actuation = self._preload_actuation
        if actuation is None:
            return False
        first_time = not self._job_tracker.is_admitted_exclusive(actuation.job)
        self._job_tracker.mark_admitted_exclusive(actuation.job)
        self._establish_whole_card_residency(
            actuation.job,
            actuation.forecast,
            announce=first_time,
            target_override=actuation.max_resident,
            device_index=device_index,
        )
        self.unload_models_from_vram(
            actuation.available_process,
            under_pressure=True,
            for_head_of_queue=True,
            device_index=device_index,
        )
        return True

    def cycle_safety_off_gpu(self, device_index: int | None) -> bool:
        """Cycle the safety model off the GPU to reclaim its context (:class:`VramActuator`)."""
        return self._pause_safety_for_residency_if_idle(device_index)

    def build_reclaim_ladder_candidates(self, device_index: int | None) -> LadderCandidates:
        """Assemble the idle-filtered inputs the verified reclaim ladder orders into rungs for a card.

        Every actively-sampling process is excluded, so a busy tenant can never become a rung. Idle inference
        processes still holding a model (and not serving an in-progress job) are unload candidates, ranked by
        recency via their last model-state report. The reclaimable-cache targets reuse the arbiter's
        release-cache selection (idle processes whose reservation exceeds allocation by the release threshold,
        holding no model). Lane and safety candidates are included only while their context is on the GPU, in
        the fixed pause order the ladder escalates through. Promised-free figures are the tenants' measured
        reservations where known, so verification compares realized frees against measured expectations.
        """
        now_monotonic = time.monotonic()
        now_wall = time.time()
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}
        idle_residents: list[IdleResidentModel] = []
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.is_process_busy() or process_info.loaded_horde_model_name is None:
                continue
            if process_info.loaded_horde_model_name in in_progress_models:
                continue
            footprint_mb = float(process_info.process_reserved_mb) if process_info.process_reserved_mb else 0.0
            idle_residents.append(
                IdleResidentModel(
                    process_id=process_info.process_id,
                    tenant_label=process_info.loaded_horde_model_name,
                    materialized_monotonic=self._reclaim_recency_key(process_info, now_monotonic, now_wall),
                    footprint_mb=footprint_mb,
                ),
            )

        idle_cache_ids, _busy = self._gpu_process_activity_ids(device_index)
        cache_targets: list[CacheReleaseTarget] = []
        for process_id in idle_cache_ids:
            process_info = self._process_map.get(process_id)
            if process_info is None or process_info.process_reserved_mb is None:
                continue
            reclaimable_mb = float(process_info.process_reserved_mb) - float(process_info.process_allocated_mb or 0)
            cache_targets.append(
                CacheReleaseTarget(
                    process_id=process_id,
                    tenant_label=f"{process_info.process_type.name.lower()}#{process_id}",
                    materialized_monotonic=self._reclaim_recency_key(process_info, now_monotonic, now_wall),
                    reclaimable_mb=reclaimable_mb,
                ),
            )

        lanes = self._reclaim_lane_candidates(device_index)
        safety = self._reclaim_safety_candidate(device_index)
        return LadderCandidates(
            device_index=device_index,
            idle_residents=tuple(idle_residents),
            cache_targets=tuple(cache_targets),
            lanes=lanes,
            safety=safety,
        )

    @staticmethod
    def _reclaim_recency_key(process_info: HordeProcessInfo, now_monotonic: float, now_wall: float) -> float:
        """Return a monotonic-scale recency key for LIFO reclaim ranking of a process.

        Prefers the dedicated ``vram_materialized_monotonic`` stamp (set when the parent observed the process
        materialize VRAM). When that is unset (an older child, or a process that has not materialized since
        start) it falls back to the report-time proxy, mapped onto the monotonic timeline
        (``now_monotonic - (now_wall - last_received_timestamp)``) so stamped and unstamped processes remain
        comparable in one ranking rather than one scale sorting entirely above the other.
        """
        if process_info.vram_materialized_monotonic is not None:
            return process_info.vram_materialized_monotonic
        return now_monotonic - (now_wall - process_info.last_received_timestamp)

    def _post_processing_lane_has_committed_work(self) -> bool:
        """Return true if the shared post-processing lane has queued or active work.

        Image post-processing lives in JobTracker. Graph-backed alchemy shares the same child process but
        owns its queue in AlchemyCoordinator, so the manager wires that count in through a provider.
        """
        if self._job_tracker.jobs_pending_post_processing or self._job_tracker.jobs_being_post_processed:
            return True
        try:
            return self._post_processing_lane_commitments_provider() > 0
        except Exception:
            logger.exception("Failed to read post-processing lane commitments; preserving the lane this cycle")
            return True

    def _has_idle_post_process_process_for_reclaim(self, device_index: int | None) -> bool:
        """Return true when a post-processing process is live, idle, and on the requested card."""
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.POST_PROCESS:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.can_accept_job():
                return True
        return False

    def _reclaim_lane_candidates(self, device_index: int | None) -> tuple[LaneReclaimCandidate, ...]:
        """Build the lane-pause rungs in fixed escalation order for lanes currently on the GPU."""
        lifecycle = self._process_lifecycle
        bridge_data = self._runtime_config.bridge_data
        lanes: list[LaneReclaimCandidate] = []
        if (
            bridge_data.allow_post_processing
            and bridge_data.post_processing_lane_enabled
            and not lifecycle.is_post_process_gpu_paused
            and not self._post_processing_lane_has_committed_work()
            and self._has_idle_post_process_process_for_reclaim(device_index)
        ):
            lanes.append(
                LaneReclaimCandidate(
                    kind=ReclaimRungKind.PAUSE_PP_LANE,
                    tenant_label="post-processing lane",
                    promised_mb=self._reserved_mb_for_type(HordeProcessType.POST_PROCESS, device_index),
                ),
            )
        if (
            lifecycle.vae_lane_enabled()
            and not lifecycle.is_vae_lane_gpu_paused
            and self._process_map.num_vae_lane_processes(device_index=device_index) > 0
        ):
            lanes.append(
                LaneReclaimCandidate(
                    kind=ReclaimRungKind.PAUSE_VAE_LANE,
                    tenant_label="VAE lane",
                    promised_mb=self._reserved_mb_for_type(HordeProcessType.VAE_LANE, device_index),
                ),
            )
        if (
            lifecycle.component_lane_enabled()
            and not lifecycle.is_component_gpu_paused
            and self._process_map.num_component_processes(device_index=device_index) > 0
        ):
            lanes.append(
                LaneReclaimCandidate(
                    kind=ReclaimRungKind.PAUSE_COMPONENT_LANE,
                    tenant_label="component lane",
                    promised_mb=self._reserved_mb_for_type(HordeProcessType.COMPONENT, device_index),
                ),
            )
        return tuple(lanes)

    def _reclaim_safety_candidate(self, device_index: int | None) -> LaneReclaimCandidate | None:
        """Build the safety-off-GPU rung when the operator allows safety to leave the GPU."""
        bridge_data = self._runtime_config.bridge_data
        if (
            not bridge_data.safety_on_gpu
            or not bridge_data.whole_card_residency_safety_off_gpu
            or self._process_lifecycle.is_safety_gpu_paused
        ):
            return None
        reserved_mb = self._reserved_mb_for_type(HordeProcessType.SAFETY, device_index)
        return LaneReclaimCandidate(
            kind=ReclaimRungKind.SAFETY_OFF_GPU,
            tenant_label="safety",
            promised_mb=reserved_mb if reserved_mb > 0 else _SAFETY_GPU_LOAD_CHARGE_MB,
        )

    def _reserved_mb_for_type(self, process_type: HordeProcessType, device_index: int | None) -> float:
        """Sum the measured device reservation (MB) of a process type's live processes on a card."""
        total_mb = 0.0
        for process_info in self._process_map.values():
            if process_info.process_type != process_type:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_reserved_mb is not None:
                total_mb += float(process_info.process_reserved_mb)
        return total_mb

    def unload_idle_model(self, process_id: int, device_index: int | None = None) -> bool:
        """Unload one idle process's resident model from VRAM to RAM (reclaim-ladder actuator).

        Targets a single named process rather than sweeping the card, so the verified ladder controls exactly
        which resident it gives back and in what order. Never touches an actively-sampling process, and treats
        a process already unloading (or without a resident model) as a no-op so the engine does not open a
        verification window on a rung that frees nothing.
        """
        process_info = self._process_map.get(process_id)
        if process_info is None or process_info.loaded_horde_model_name is None:
            return False
        if process_info.is_process_busy():
            return False
        if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
            return False
        model_name = process_info.loaded_horde_model_name
        if not process_info.safe_send_message(
            HordeControlModelMessage(
                control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
                horde_model_name=model_name,
            ),
        ):
            return False
        process_info.last_job_referenced = None
        process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        self._record_churn("vram_eviction")
        logger.info(f"Reclaim ladder: unloading model {model_name} from VRAM on idle process {process_id}")
        return True

    def release_idle_cache(self, process_id: int) -> bool:
        """Release an idle process's reclaimable allocator cache back to the card (reclaim-ladder actuator)."""
        return self.release_allocator_cache(process_id)

    def pause_post_process_lane(self, device_index: int | None) -> bool:
        """Pause the post-processing lane off the GPU to reclaim its context (reclaim-ladder actuator)."""
        return self._process_lifecycle.pause_post_process_off_gpu(owner=PauseOwner.RECLAIM_LADDER)

    def pause_vae_lane(self, device_index: int | None) -> bool:
        """Pause the VAE lane off the GPU to reclaim its context (reclaim-ladder actuator)."""
        return self._process_lifecycle.pause_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER)

    def pause_component_lane(self, device_index: int | None) -> bool:
        """Pause the component lane off the GPU to reclaim its context (reclaim-ladder actuator)."""
        return self._process_lifecycle.pause_component_off_gpu(owner=PauseOwner.RECLAIM_LADDER)

    def safety_off_gpu(self, device_index: int | None) -> bool:
        """Move the on-GPU safety context off the card to reclaim it (reclaim-ladder actuator).

        Unlike the lane pauses, the ladder does not restore safety when the episode ends: the runtime
        safety-placement policy owns safety's on/off-GPU state and re-promotes it once the card demonstrably
        fits its context beside the largest active sampling peak, so a ladder-cycled safety pause has a live,
        independent restore path and is never stranded.
        """
        return self._process_lifecycle.pause_safety_on_gpu()

    def restore_post_process_lane(self, device_index: int | None) -> bool:
        """Restart a ladder-paused post-processing lane once the card has recovered (reclaim-ladder actuator)."""
        return self._process_lifecycle.restore_post_process_off_gpu(owner=PauseOwner.RECLAIM_LADDER)

    def restore_vae_lane(self, device_index: int | None) -> bool:
        """Restart a ladder-paused VAE lane once the card has recovered (reclaim-ladder actuator)."""
        return self._process_lifecycle.restore_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER)

    def restore_component_lane(self, device_index: int | None) -> bool:
        """Restart a ladder-paused component lane once the card has recovered (reclaim-ladder actuator)."""
        return self._process_lifecycle.restore_component_off_gpu(owner=PauseOwner.RECLAIM_LADDER)

    def record_calibration_event(self, rung: ReclaimRung, *, promised_mb: float, realized_mb: float) -> None:
        """Record a verified reclaim shortfall as a calibration counter (reclaim-ladder actuator).

        The freed figure is not a footprint peak and no complete footprint key is reconstructable at reclaim
        time, so the raise-only footprint store does not apply; the signal is kept as a count for calibration
        visibility that a rung's promised free over-stated what the hardware returned.
        """
        self._reclaim_calibration_events += 1

    def preload_models(self) -> bool:
        """Preload models that are likely to be used soon.

        Housekeeping first (whole-card residency restore/convergence, stale model-map expiry, clearing
        preloads the queue no longer needs), then one pass over the pending queue: each job runs through
        the admission pipeline (:meth:`_attempt_preload_for_job`) until one preloads or a gate stops the
        pass for this cycle.

        Returns:
            True if a model was preloaded, False otherwise.
        """
        self._restore_siblings_after_whole_card()
        self._reconcile_runtime_safety_placement()
        self._restore_deferred_safety_gpu_load()
        self._converge_whole_card_residency()
        self._expire_stale_model_map_entries()

        if self._pending_post_processing_should_hold_preload():
            return False

        loaded_models = {process.loaded_horde_model_name for process in self._process_map.values()}
        loaded_models = loaded_models.union(
            model.horde_model_name
            for model in self._horde_model_map.root.values()
            if model.horde_model_load_state.is_loaded() or model.horde_model_load_state == ModelLoadState.LOADING
        )

        pending_models = {job.model for job in self._job_tracker.jobs_pending_inference}
        for process in self._process_map.values():
            if (
                process.last_process_state == HordeProcessState.PRELOADED_MODEL
                and process.loaded_horde_model_name not in pending_models
            ):
                logger.debug(
                    f"Clearing preloaded model {process.loaded_horde_model_name} "
                    f"from process {process.process_id} as it is no longer needed",
                )
                self._process_map.on_process_state_change(
                    process_id=process.process_id,
                    new_state=HordeProcessState.WAITING_FOR_JOB,
                )

        if loaded_models == pending_models:
            return False

        # The first queued job not already in progress is the head of the queue. Only when *its* model
        # is the one that cannot be loaded may the budget gate escalate to evicting another queued
        # model (see the budget-defer branches in the admission pipeline); a later job whose turn has
        # not come never displaces a resident head.
        in_progress_jobs = self._job_tracker.jobs_in_progress
        head_job = next((j for j in self._job_tracker.jobs_pending_inference if j not in in_progress_jobs), None)
        self._update_head_starvation_timer(head_job)
        if self._resident_head_should_dispatch_before_preload(head_job):
            return False

        for job in self._job_tracker.jobs_pending_inference:
            outcome = self._attempt_preload_for_job(job, head_job=head_job, loaded_models=loaded_models)
            if outcome is _PreloadJobOutcome.NEXT_JOB:
                continue
            return outcome is _PreloadJobOutcome.PRELOAD_SENT

        return False

    def _pending_post_processing_should_hold_preload(self) -> bool:
        """Whether a pending post-processing chain should receive the next drain window before preloads."""
        return self._pending_post_processing_reserve_mb(device_index=None) > 0

    def _resident_head_should_dispatch_before_preload(self, head_job: ImageGenerateJobPopResponse | None) -> bool:
        """Whether the queue head can already try dispatch, so speculative preloading should yield."""
        if head_job is None or head_job.model is None:
            return False
        if self._process_lifecycle.is_model_load_quarantined(head_job.model):
            return False

        process_with_model = self._resident_process_for_job(head_job)
        if process_with_model is None or not process_with_model.can_accept_job():
            return False

        target_card: CardRuntime | None = None
        if self._multi_gpu_routing_active and not self._job_tracker.has_exclusive_job_in_progress():
            target_card = self._card_runtimes.get(process_with_model.device_index)

        if target_card is not None:
            jobs_in_progress_count = len(self._jobs_in_progress_on_card(target_card.device_index))
        else:
            jobs_in_progress_count = len(self._job_tracker.jobs_in_progress)
        return jobs_in_progress_count < self._max_jobs_in_progress_allowed(card=target_card)

    def _record_preload_admission(
        self,
        decision: AdmissionDecision,
        *,
        job: ImageGenerateJobPopResponse | None = None,
        process: HordeProcessInfo | None = None,
        reason: str = "",
    ) -> None:
        """Remember one preload-admission decision for operator diagnostics."""
        self._last_preload_admission = LatestPreloadAdmission(
            decision=decision,
            model=job.model if job is not None else None,
            process_id=process.process_id if process is not None else None,
            reason=reason,
            timestamp=time.time(),
        )

    def _preload_outcome(
        self,
        decision: AdmissionDecision,
        *,
        job: ImageGenerateJobPopResponse | None = None,
        process: HordeProcessInfo | None = None,
        reason: str = "",
    ) -> _PreloadJobOutcome:
        """Record a public admission decision and map it onto the preload pass control enum."""
        self._record_preload_admission(decision, job=job, process=process, reason=reason)
        return _preload_outcome_from_admission(decision)

    def _attempt_preload_for_job(
        self,
        job: ImageGenerateJobPopResponse,
        *,
        head_job: ImageGenerateJobPopResponse | None,
        loaded_models: set[str | None],
    ) -> _PreloadJobOutcome:
        """Run one pending job through the preload admission pipeline.

        The gates, in order: quarantine (faults the job), already-resident, the absolute RAM danger
        floor, the exclusive-job hold, target selection, the
        cycle-on-model-change replacement, the per-device load serialization gate, and the VRAM/RAM
        budget admission. The returned :class:`_PreloadJobOutcome` tells the pass whether to consider the
        next pending job, stop for this cycle, or record that a preload was issued.
        """
        bridge_data = self._runtime_config.bridge_data
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")

        if (unserviceable_reason := self._unserviceable_job_reason(job)) is not None:
            if job not in self._job_tracker.jobs_in_progress:
                self._fault_unserviceable_job(job, unserviceable_reason)
            return self._preload_outcome(
                AdmissionDecision.UNSERVICEABLE,
                job=job,
                reason=unserviceable_reason,
            )

        # A model quarantined for repeatedly failing to load must never be preloaded again: doing so only
        # re-arms the crash/recovery loop it was quarantined to stop. Fault the job so the horde reissues
        # it elsewhere rather than letting an unservable head wedge the queue.
        if self._process_lifecycle.is_model_load_quarantined(job.model):
            if job not in self._job_tracker.jobs_in_progress:
                logger.warning(
                    f"Skipping preload of quarantined model {job.model}; faulting its job for reissue.",
                )
                self._job_tracker.handle_job_fault_now(job, retryable=False)
            return self._preload_outcome(AdmissionDecision.QUARANTINED, job=job, reason="model load quarantined")

        if job.model in loaded_models:
            return self._preload_outcome(
                AdmissionDecision.ALREADY_LOADED, job=job, reason="model already resident or loading"
            )

        # Absolute system-RAM floor (degrade, never crash): loading a new model routes its weights through
        # system RAM first, so admitting one while the host is already below its danger floor is the OS
        # OOM kill, not progress. This gates every preload path independent of the marginal RAM budget, which
        # can pass on a job's small estimate while the whole host is on the edge (resident weights + the
        # safety process + other apps). The governor's tick has already degraded the host this cycle;
        # this only defers the load. Gated on the budget being active (the same switch the rest of the
        # memory machinery uses).
        if self._budget_active() and self._preload_blocked_by_ram_pressure(job):
            return self._preload_outcome(
                AdmissionDecision.DEFER_RAM_PRESSURE, job=job, reason="system RAM danger floor"
            )

        # An exclusively-admitted over-budget job has the whole device; do not stage another model's
        # weights concurrently (a second resident load is exactly what spills the exclusive job's
        # weights to system RAM). The exclusive job's own preload is still allowed through.
        if self._job_tracker.has_exclusive_job_in_progress() and not self._job_tracker.is_admitted_exclusive(job):
            return self._preload_outcome(
                AdmissionDecision.EXCLUSIVE_IN_PROGRESS, job=job, reason="exclusive over-budget job in progress"
            )

        is_head_blocker = head_job is not None and job is head_job

        # Which slots this preload may not displace: the queued-model guard, model->process affinity
        # (never displace the last resident copy of a still-wanted model; the working model set is
        # taken from live state, not bridge_data.image_models_to_load, because the harness/canned
        # path never resolves that config field), and slots draining for RAM reclaim. The guards are
        # target exclusions only, never a wedge: the head-starvation fallback below deliberately
        # overrides them, and the governor recycles a draining slot once it is idle.
        inference_process_models = {
            p.process_id: p.loaded_horde_model_name
            for p in self._process_map.values()
            if p.process_type == HordeProcessType.INFERENCE
        }
        wanted_models: set[str] = {m for m in inference_process_models.values() if m is not None}
        wanted_models.update(j.model for j in self._job_tracker.jobs_pending_inference if j.model is not None)
        wanted_models.update(j.model for j in self._job_tracker.jobs_in_progress if j.model is not None)
        preload_disallowed = compute_preload_disallowed_processes(
            queued_model_process_ids=self._process_lifecycle.get_processes_with_model_for_queued_job(),
            busy_process_ids=[p.process_id for p in self._process_map.values() if p.is_process_busy()],
            prefer_busy_only=self._process_map.num_loaded_inference_processes()
            < (len(self._job_tracker.jobs_pending_inference) + len(self._job_tracker.jobs_in_progress)),
            inference_process_models=inference_process_models,
            wanted_models=wanted_models,
            max_inference_processes=self._max_inference_processes,
            draining_process_ids=frozenset(self._processes_draining_for_ram),
        )

        # On a multi-GPU host this also chooses *which* card to load onto: an eligible card already
        # holding the model first, then the least-loaded eligible card. Single-GPU returns the first
        # available slot exactly as before.
        available_process = self._select_preload_process(job, sorted(preload_disallowed))

        if available_process is None and is_head_blocker:
            # The head of the queue could not get a slot because affinity (or the queued-model
            # guard) protected every idle process. Affinity is provisioned against the
            # inference-process *ceiling*, so with more resident models than running processes it
            # can pin every slot and starve a genuinely-queued head, wedging the whole worker. The
            # head must make progress regardless of whether the measured budget is active, so fall
            # back to a displacement target that spares live work and prefers an idle resident model
            # no queued job needs. This is the budget-independent counterpart to the budget-gated
            # make-room escalation in the admission pipeline.
            available_process = self._select_head_room_process()

        if available_process is None:
            return self._preload_outcome(
                AdmissionDecision.NO_TARGET, job=job, reason="no idle inference slot available"
            )

        # Device-free governor growth hold: while the target card's device-level free VRAM sits below the
        # soft floor (PRESSURE or SATURATED), bringing this model to a slot that does not already hold it
        # would grow a footprint already near the WDDM paging cliff. Defer until the card recovers. A job
        # already in progress is exempt: its preload is part of live work, not new speculative growth, and
        # withholding it would wedge a job the card is already committed to.
        growth_hold_device = available_process.device_index if self._multi_gpu_routing_active else 0
        if self.is_vram_growth_held(growth_hold_device) and job not in self._job_tracker.jobs_in_progress:
            return self._preload_outcome(
                AdmissionDecision.DEFER_VRAM_GROWTH_HOLD,
                job=job,
                process=available_process,
                reason="device-free governor holding VRAM growth (device near paging cliff)",
            )

        if (
            available_process.last_process_state != HordeProcessState.WAITING_FOR_JOB
            and available_process.loaded_horde_model_name is not None
            and bridge_data.cycle_process_on_model_change
            and not self._state.shutting_down
        ):
            self._process_lifecycle._replace_inference_process(available_process, intentional_reclaim=True)
            return self._preload_outcome(
                AdmissionDecision.REPLACE_PROCESS,
                job=job,
                process=available_process,
                reason="cycling process for model change",
            )

        # Serialize preloads per card, not worker-wide: the gate exists so two checkpoints do not load
        # onto the same device at once (disk-read + VRAM-allocation spike). On a multi-GPU host a load
        # onto an idle card is independent of one happening on another card, so scope the in-flight count
        # to the card this preload would land on. Worker-wide (device_index=None) on a single-GPU host
        # keeps the original behavior byte-identical. Without this, a card that is almost always mid-load
        # (the busy card) perpetually blocks the idle card from ever getting its first model -> starvation.
        preload_scope_device = available_process.device_index if self._multi_gpu_routing_active else None
        num_preloading_processes = self._process_map.num_preloading_processes(
            device_index=preload_scope_device,
        )

        if preload_concurrency_blocked(
            num_preloading=num_preloading_processes,
            max_concurrent_inference_processes=self._max_concurrent_inference_processes,
            very_fast_disk_mode=bool(bridge_data.very_fast_disk_mode),
        ):
            if not self._preload_delay_notified:
                logger.opt(ansi=True).info(
                    "<fg #7b7d7d>"
                    f"Already preloading {num_preloading_processes} models, waiting for one to finish before "
                    f"preloading {job.model}"
                    "</>",
                )
                self._preload_delay_notified = True
            return self._preload_outcome(
                AdmissionDecision.DEFER_CONCURRENCY,
                job=job,
                process=available_process,
                reason="preload concurrency gate",
            )

        # Resource budget gate: a fresh preload loads this model's weights into the shared device
        # (VRAM) and into system RAM, so admit it only when both measured free VRAM and available
        # RAM cover its estimated cost plus their reserves. This is the proactive guard against the
        # multi-process over-commit that OOMs the GPU and against resident weights paging RAM to
        # disk. When a resource does not fit, start reclaiming it from idle resident models
        # (overriding residency under pressure) and defer this preload rather than over-committing.
        if self._budget_active() and not self._admit_preload_under_budget(
            job,
            available_process,
            is_head_blocker=is_head_blocker,
        ):
            return self._preload_outcome(
                AdmissionDecision.DEFER_BUDGET, job=job, process=available_process, reason="VRAM/RAM budget gate"
            )

        if self._send_preload(job, available_process):
            return self._preload_outcome(
                AdmissionDecision.ADMIT, job=job, process=available_process, reason="preload sent"
            )
        return self._preload_outcome(
            AdmissionDecision.STOP_PASS, job=job, process=available_process, reason="preload send failed"
        )

    def _select_head_room_process(self) -> HordeProcessInfo | None:
        """Pick an idle inference process to free for a starved head-of-queue job, or None.

        Used when the normal preload picker found no slot because affinity (provisioned against the
        inference-process ceiling) or the queued-model guard protected every idle process. The head must
        still make progress, so this deliberately overrides those guards. It never returns a process
        running live work (only ``can_accept_job()`` slots, and never one whose model is in progress) and
        prefers the cheapest displacement: an empty slot, then one holding a resident model no pending or
        in-progress job needs, then, as a last resort, one holding a merely-queued model.
        """
        slots = tuple(
            PreloadSlotSnapshot(
                process_id=process_info.process_id,
                model_name=process_info.loaded_horde_model_name,
                can_accept_job=process_info.can_accept_job(),
            )
            for process_info in self._process_map.values()
            if process_info.process_type == HordeProcessType.INFERENCE
        )
        chosen_id = select_head_room_process_id(
            slots,
            in_progress_models={job.model for job in self._job_tracker.jobs_in_progress},
            pending_models={job.model for job in self._job_tracker.jobs_pending_inference if job.model is not None},
        )
        return self._process_map.get(chosen_id) if chosen_id is not None else None

    def _select_idle_thread_diversity_job(
        self,
        head_job: ImageGenerateJobPopResponse,
        candidates: list[ImageGenerateJobPopResponse],
    ) -> tuple[ImageGenerateJobPopResponse, HordeProcessInfo] | None:
        """A pending distinct-model job resident on a free process that may overlap the in-flight work.

        When the head's own process cannot take work right now because it is busy sampling the head's model,
        a later job for a *different* model that is already resident on an idle process can run concurrently
        instead of leaving the thread idle. Preferring a distinct model also avoids loading a duplicate copy
        of the head's model onto a second process: with several same-model jobs ahead of a lone different
        model (a run of one checkpoint followed by another), threading the different model alongside the run
        processes it for free under the run, rather than idling a thread and tacking the second model on at
        the end as its own load. The overlap-headway gate still applies (two heavy models are not stacked
        without headway), degraded retries that must run isolated are skipped, and the head keeps its queue
        position (the caller records a line-skip) so it dispatches the moment its process frees.
        """
        for candidate_job in candidates:
            if candidate_job.model is None or candidate_job.model == head_job.model:
                continue
            if self._job_tracker.is_degraded_dispatch_pending(candidate_job):
                continue
            candidate_process = self._resident_process_for_job(candidate_job)
            if candidate_process is None or not candidate_process.can_accept_job():
                continue
            if not self._concurrent_overlap_allowed(
                candidate_job,
                target_device_index=candidate_process.device_index,
            ):
                continue
            return candidate_job, candidate_process
        return None

    @property
    def _multi_gpu_routing_active(self) -> bool:
        """Whether per-card dispatch routing applies (the worker drives more than one card).

        A single card (or the empty plan unit tests construct) keeps the original card-agnostic dispatch,
        so all multi-GPU routing below is a strict no-op on a single-GPU host.
        """
        return len(self._card_runtimes) > 1

    def _eligible_card_indices(self, job: ImageGenerateJobPopResponse) -> set[int]:
        """Device indices of the cards whose effective config can serve ``job`` (see ``gpu_eligibility``).

        Restricts dispatch (and the resident-process search) to cards that offer the job's model, fit its
        weights, enable every feature it needs, and allow its resolution. An unknown fact never excludes a
        card (the eligibility primitive abstains), so this only ever narrows routing on a genuine mismatch.
        """
        baseline = self._model_metadata.get_baseline(job.model) if job.model is not None else None
        baseline_value = baseline.value if isinstance(baseline, KNOWN_IMAGE_GENERATION_BASELINE) else baseline
        weight_mb = predict_job_weight_mb(job, baseline)
        return eligible_card_indices_for(
            job,
            self._card_runtimes,
            baseline=baseline_value,
            weight_mb=weight_mb,
        )

    def _baseline_value_for_job(self, job: ImageGenerateJobPopResponse) -> str | None:
        """Return the job model's baseline value, or None when metadata is unavailable."""
        if job.model is None:
            return None
        baseline = self._model_metadata.get_baseline(job.model)
        return baseline.value if isinstance(baseline, KNOWN_IMAGE_GENERATION_BASELINE) else baseline

    def _model_serviceability_verdicts(
        self,
        job: ImageGenerateJobPopResponse,
    ) -> list[tuple[CardRuntime, ModelServiceabilityVerdict]]:
        """Return per-serving-card serviceability verdicts for ``job``.

        The check abstains when the runtime card map or model footprint is unknown. It considers only cards
        whose effective config serves the model, matching the offer and placement surfaces.
        """
        if job.model is None or not self._card_runtimes:
            return []
        baseline = self._baseline_value_for_job(job)
        figures = model_footprint_figures_for_baseline(baseline)
        if figures is None:
            return []
        verdicts: list[tuple[CardRuntime, ModelServiceabilityVerdict]] = []
        for card in self._card_runtimes.values():
            if job.model not in set(card.config.image_models_to_load):
                continue
            baseline_mb = self._admission_baseline_mb(card.device_index)
            verdicts.append(
                (
                    card,
                    assess_model_serviceability(
                        total_vram_mb=card.total_vram_mb,
                        baseline_mb=baseline_mb,
                        noise_buffer_mb=None,
                        figures=figures,
                    ),
                ),
            )
        return verdicts

    def _unserviceable_job_reason(self, job: ImageGenerateJobPopResponse) -> str | None:
        """Return a fault reason when no serving card can ever host this job's model minimum."""
        verdicts = self._model_serviceability_verdicts(job)
        if not verdicts or any(verdict.serviceable for _, verdict in verdicts):
            return None
        arithmetic = "; ".join(f"device {card.device_index}: {verdict.reason()}" for card, verdict in verdicts)
        return f"model minimum footprint cannot fit any serving card; {arithmetic}"

    def _fault_unserviceable_job(self, job: ImageGenerateJobPopResponse, reason: str) -> None:
        """Fault an unserviceable queued job before any child process touches VRAM for it."""
        logger.warning(f"Faulting unserviceable job {job.id_} for model {job.model}: {reason}")
        self._job_tracker.handle_job_fault_now(
            job,
            is_resource_failure=True,
            retryable=False,
            fault_reason=reason,
        )

    def _card_inference_load(self, device_index: int) -> int:
        """Count this card's inference processes currently busy: the least-loaded routing tie-breaker."""
        return sum(
            1
            for p in self._process_map.values()
            if p.process_type == HordeProcessType.INFERENCE and p.device_index == device_index and p.is_process_busy()
        )

    def _pick_best_resident_process(self, candidates: list[HordeProcessInfo]) -> HordeProcessInfo:
        """Choose which resident process to dispatch to: prefer one ready now, then the least-loaded card.

        The "sticky, then least-loaded" policy at the process level: every candidate already holds the model
        (sticky), so among them a process that can take work immediately is preferred, and ties break to the
        card running the fewest inference jobs so a hot model spreads across cards instead of queueing on one.
        """
        ready = [p for p in candidates if p.can_accept_job()]
        pool = ready or candidates
        return min(pool, key=lambda p: self._card_inference_load(p.device_index))

    def _resident_process_for_job(
        self,
        job: ImageGenerateJobPopResponse,
        *,
        include_reserved: bool = False,
    ) -> HordeProcessInfo | None:
        """The resident process to dispatch ``job`` to, honoring per-card eligibility on a multi-GPU host.

        Single-GPU: identical to :meth:`ProcessMap.get_process_by_horde_model_name` (the first resident
        process), so the dispatch path is byte-identical. Multi-GPU: restrict to processes pinned to cards
        eligible for this job, then apply the sticky-then-least-loaded policy. Returns None when the model is
        resident only on cards that cannot serve this particular job, or is not resident anywhere.

        Pinned disaggregation-sampler lanes are excluded by default (a dispatch may never land on a pinned
        lane). ``include_reserved=True`` includes them, for the residency and pricing queries that must see a
        model's weights even where they sit on a lane no job may be dispatched onto yet.
        """
        if job.model is None:
            return None
        if not self._multi_gpu_routing_active:
            return self._process_map.get_process_by_horde_model_name(job.model, include_reserved=include_reserved)
        allowed = self._eligible_card_indices(job)
        candidates = self._process_map.get_processes_by_horde_model_name(
            job.model,
            allowed_cards=allowed,
            include_reserved=include_reserved,
        )
        if not candidates:
            return None
        return self._pick_best_resident_process(candidates)

    def _idle_resident_process_excluding(
        self,
        job: ImageGenerateJobPopResponse,
        *,
        exclude_process_id: int | None,
    ) -> HordeProcessInfo | None:
        """A resident, idle (can-accept) process for ``job`` other than ``exclude_process_id``.

        Used by the line-skip when the candidate shares the blocked head's model: the head's own slot reads
        as able to accept work while it downloads auxiliary models, so a same-model skip must be routed onto
        a *different* idle process that already holds the model, never back onto the downloading slot.
        Returns None when the model is resident only on the excluded slot or on busy lanes, honoring per-card
        eligibility on a multi-GPU host. Ties break to the least-loaded card, matching dispatch selection.
        """
        if job.model is None:
            return None
        if self._multi_gpu_routing_active:
            candidates = self._process_map.get_processes_by_horde_model_name(
                job.model,
                allowed_cards=self._eligible_card_indices(job),
            )
        else:
            candidates = self._process_map.get_processes_by_horde_model_name(job.model)
        ready = [p for p in candidates if p.process_id != exclude_process_id and p.can_accept_job()]
        if not ready:
            return None
        return min(ready, key=lambda p: self._card_inference_load(p.device_index))

    def _pinned_lane_resident_for_job(self, job: ImageGenerateJobPopResponse) -> HordeProcessInfo | None:
        """The disaggregation-pinned lane holding ``job``'s model when that is the only resident copy, else None.

        The dispatch query (:meth:`_resident_process_for_job`) excludes pinned lanes, so a job whose model is
        resident only on a pinned sampler lane reads as not-resident and would otherwise be priced a fresh
        preload that cannot fit beside the pinned residents. This names that case: an unreserved resident copy
        does not exist, but a pinned lane holds the model. The head then holds for the pin to release (dispatch
        onto the resident lane, priced as resident) instead of funding a second copy; a job is never dispatched
        onto the returned pinned lane.
        """
        if self._resident_process_for_job(job) is not None:
            return None
        resident = self._resident_process_for_job(job, include_reserved=True)
        if resident is not None and self._process_map.is_reserved_for_disaggregation(resident.process_id):
            return resident
        return None

    def _select_preload_process(
        self,
        job: ImageGenerateJobPopResponse,
        disallowed_processes: list[int],
    ) -> HordeProcessInfo | None:
        """The inference slot to preload ``job``'s model onto, choosing the card on a multi-GPU host.

        Single-GPU: identical to :meth:`ProcessMap.get_first_available_inference_process`, so the preload path
        is byte-identical. Multi-GPU: restrict to cards eligible for this job and pick the placement card by the
        same sticky-then-least-loaded policy dispatch uses: a card already holding this model first (avoid a
        duplicate load), then the card running the fewest inference jobs (balance fresh loads). Returns the
        first available slot on the best such card, or None when no eligible card has a free slot.
        """
        if not self._multi_gpu_routing_active:
            return self._process_map.get_first_available_inference_process(disallowed_processes=disallowed_processes)
        eligible = self._eligible_card_indices(job)
        if not eligible:
            return None

        cards_already_serving_model = {
            process.device_index
            for process in self._process_map.values()
            if process.loaded_horde_model_name == job.model
        }
        placement_order = card_preload_order(
            eligible,
            cards_already_serving_model=cards_already_serving_model,
            card_busy_counts={device_index: self._card_inference_load(device_index) for device_index in eligible},
            card_free_vram_mb={
                device_index: self._measured_free_vram_mb(device_index=device_index) for device_index in eligible
            },
        )
        for device_index in placement_order:
            candidate = self._process_map.get_first_available_inference_process(
                disallowed_processes=disallowed_processes,
                device_index=device_index,
            )
            if candidate is not None:
                return candidate
        return None

    def _log_line_skip_rejection(self, candidate_id: str, reason_key: str, message: str) -> None:
        """Emit a line-skip rejection at DEBUG, rate-limited per (candidate, reason).

        Line-skip is re-evaluated every (sub-second) scheduling pass while a head job is blocked, so an
        unthrottled rejection log floods the file with thousands of identical lines during a stall. This
        keeps full fidelity (every distinct candidate and reason is still logged, and a changed reason
        logs immediately) while collapsing the repeats to at most one per
        ``_LINE_SKIP_REJECTION_LOG_INTERVAL``.
        """
        now = time.monotonic()
        key = f"{candidate_id}:{reason_key}"
        last = self._line_skip_rejection_log_state.get(key)
        if last is not None and (now - last) < _LINE_SKIP_REJECTION_LOG_INTERVAL:
            return
        self._line_skip_rejection_log_state[key] = now
        # Prune stale keys so a long-running worker's churn of candidate ids cannot grow this unboundedly.
        if len(self._line_skip_rejection_log_state) > _LINE_SKIP_REJECTION_LOG_MAX_KEYS:
            cutoff = now - _LINE_SKIP_REJECTION_LOG_INTERVAL
            self._line_skip_rejection_log_state = {
                k: t for k, t in self._line_skip_rejection_log_state.items() if t >= cutoff
            }
        logger.debug(f"Line-skip candidate {candidate_id} {message}")

    def _select_line_skip_candidate(
        self,
        displaced_job: ImageGenerateJobPopResponse,
        *,
        next_n_jobs: list[ImageGenerateJobPopResponse],
        candidate_job_size: int,
        displaced_process_id: int | None,
    ) -> NextJobAndProcess | None:
        """Select a small, ready job that may bypass ``displaced_job`` while its slot is non-sampling.

        Scans the pending jobs for the first resident on an idle process that carries no LoRAs, is not a
        degraded retry, and is within the per-performance-mode size limit. A candidate sharing the blocked
        head's model qualifies as long as that model is resident on a *different* idle process than the
        downloading head (``displaced_process_id``): during an aux-model download the head's own slot reads
        as able to accept work, so a same-model skip is routed onto a sibling copy, never back onto the
        downloading slot. This keeps a mono-model queue from starving the GPU whenever its one head stalls
        on a LoRA download. Returns a :class:`NextJobAndProcess` carrying the :class:`LineSkip` record, or
        None when nothing qualifies. Rejections are logged (rate-limited).
        """
        for candidate_small_job in next_n_jobs:
            candidate_id = str(candidate_small_job.id_)[:8]
            job_has_loras = (
                candidate_small_job.payload.loras is not None and len(candidate_small_job.payload.loras) > 0
            )
            if candidate_small_job.model is None:
                self._log_line_skip_rejection(candidate_id, "missing_model", "rejected: missing model.")
                continue
            if job_has_loras:
                self._log_line_skip_rejection(candidate_id, "has_loras", "rejected: candidate has LoRAs.")
                continue
            if self._job_tracker.is_degraded_dispatch_pending(candidate_small_job):
                self._log_line_skip_rejection(
                    candidate_id,
                    "degraded",
                    "rejected: degraded retry must run isolated.",
                )
                continue

            if candidate_small_job.model == displaced_job.model:
                candidate_process_with_model = self._idle_resident_process_excluding(
                    candidate_small_job,
                    exclude_process_id=displaced_process_id,
                )
                if candidate_process_with_model is None:
                    self._log_line_skip_rejection(
                        candidate_id,
                        "same_model",
                        f"rejected: same model as blocked job {str(displaced_job.id_)[:8]} with no other "
                        "idle resident lane.",
                    )
                    continue
            else:
                candidate_process_with_model = self._resident_process_for_job(candidate_small_job)
                if candidate_process_with_model is None:
                    self._log_line_skip_rejection(
                        candidate_id,
                        "not_resident",
                        f"rejected: model {candidate_small_job.model} is not resident.",
                    )
                    continue

            candidate_effective_mps = self.get_single_job_effective_megapixelsteps(candidate_small_job)
            if candidate_effective_mps > candidate_job_size:
                self._log_line_skip_rejection(
                    candidate_id,
                    "emps",
                    f"rejected: {candidate_effective_mps} eMPS exceeds {candidate_job_size} eMPS limit.",
                )
                continue

            if not candidate_process_with_model.can_accept_job():
                self._log_line_skip_rejection(
                    candidate_id,
                    "process_state",
                    f"rejected: process {candidate_process_with_model.process_id} is "
                    f"{candidate_process_with_model.last_process_state.name}.",
                )
                continue

            logger.debug(
                f"Line-skip candidate {candidate_id} accepted: {candidate_small_job.model}, "
                f"{candidate_effective_mps} eMPS <= {candidate_job_size}, process "
                f"{candidate_process_with_model.process_id} can accept work.",
            )
            return NextJobAndProcess(
                next_job=candidate_small_job,
                process_with_model=candidate_process_with_model,
                line_skip=LineSkip(displaced_job=displaced_job),
            )

        return None

    def _active_aux_download_blocker(
        self,
        *,
        device_index: int | None,
    ) -> tuple[ImageGenerateJobPopResponse, HordeProcessInfo] | None:
        """Return an in-progress job whose slot is downloading auxiliaries instead of sampling.

        The ordinary concurrency cap counts a popped job as in progress as soon as START_INFERENCE is sent,
        including the time its child spends fetching LoRAs. That work consumes no sampling slot. A different
        model already resident on an idle process may therefore use the card without increasing denoiser
        concurrency; this helper identifies the active download whose cap slot can be borrowed.
        """
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.last_process_state != HordeProcessState.DOWNLOADING_AUX_MODEL:
                continue
            job = process_info.last_job_referenced
            if job is not None and job in self._job_tracker.jobs_in_progress:
                return job, process_info
        return None

    async def _handle_process_missing(
        self,
        job: ImageGenerateJobPopResponse,
        *,
        process_with_model: HordeProcessInfo | None,
    ) -> None:
        """Recover when the head's model was expected resident but no process holds it.

        Expires the stale model-map entry, clears any process still tagged with the model, and releases
        the job from in-progress so a fresh preload can be scheduled. Guarded by ``_model_recently_missing``
        so the recovery runs at most once until a model loads again.
        """
        if self._model_recently_missing:
            return
        logger.warning(
            f"Expected to find a process with model {job.model} but none was found. Attempt to load it now...",
        )
        logger.debug(f"Horde model map: {self._horde_model_map}")
        logger.debug(f"Process map: {self._process_map}")

        if job.model is not None:
            logger.debug(f"Expiring entry for model {job.model}")
            self._horde_model_map.expire_entry(job.model)

            if process_with_model is not None:
                logger.debug(f"Clearing process {process_with_model.process_id} of model {job.model}")

                horde_model_baseline = self._model_metadata.get_baseline(job.model)

                self._process_map.on_model_load_state_change(
                    process_id=process_with_model.process_id,
                    horde_model_name=job.model,
                    horde_model_baseline=horde_model_baseline,
                )

            logger.debug(f"Horde model map: {self._horde_model_map}")
            logger.debug(f"Process map: {self._process_map}")

            self._model_recently_missing = True

            logger.debug(f"Last missing time: {self._model_recently_missing_time}")
            self._model_recently_missing_time = time.time()

            if not await self._job_tracker.release_in_progress(job):
                logger.debug(f"Job {job.id_} not found in jobs_in_progress.")

    async def get_next_job_and_process(
        self,
        information_only: bool = False,
    ) -> NextJobAndProcess | None:
        """Get the next job and process that can be started, if any.

        A single scheduling cycle calls this twice: once with ``information_only=True``
        (to look ahead for heavy-model / single-inference decisions) and once from
        :meth:`start_inference` to actually launch. The two calls must agree on which
        job is selected. Normal selection is deterministic given the queue and process
        states, but the line-skip branch below depends on transient process state
        (e.g. a process being mid aux-model download), so a line-skip decision is
        cached in ``self._pending_line_skip`` and returned on the second call. The
        cache is cleared at the start of each scheduling cycle (see
        :meth:`run_scheduling_cycle`) and at the end of :meth:`start_inference`.
        """
        cached = self._pending_line_skip
        if cached is not None:
            cached_job = cached.next_job
            if (
                cached_job in self._job_tracker.jobs_pending_inference
                and cached_job not in self._job_tracker.jobs_in_progress
                and cached.process_with_model.can_accept_job()
            ):
                return cached
            self._pending_line_skip = None

        bridge_data = self._runtime_config.bridge_data

        next_job: ImageGenerateJobPopResponse | None = None
        next_n_jobs: list[ImageGenerateJobPopResponse] = []
        for job in self._job_tracker.jobs_pending_inference:
            if job in self._job_tracker.jobs_in_progress:
                continue
            # Never make a quarantined model the dispatch head: it can never become resident (preload_models
            # skips it and faults it for reissue), so selecting it here would only stall the scheduler.
            if self._process_lifecycle.is_model_load_quarantined(job.model):
                continue
            if next_job is None:
                next_job = job
            next_n_jobs.append(job)

        if next_job is None:
            return None

        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

        if (unserviceable_reason := self._unserviceable_job_reason(next_job)) is not None:
            if not information_only:
                self._fault_unserviceable_job(next_job, unserviceable_reason)
            return None

        process_with_model = self._resident_process_for_job(next_job)
        line_skip: LineSkip | None = None

        candidate_job_size = line_skip_candidate_emps_limit(
            high_performance_mode=bool(bridge_data.high_performance_mode),
            moderate_performance_mode=bool(bridge_data.moderate_performance_mode),
        )

        # On a multi-GPU host the head's resident process names the card this dispatch would land on, so the
        # concurrency cap is scoped to that card: its own in-progress count vs its own ceilings. The scope is
        # dropped (worker-wide, as on a single-GPU host) when the head is not yet resident (no target card) or
        # while a worker-wide exclusive job is suppressing all dispatch.
        target_card: CardRuntime | None = None
        if (
            self._multi_gpu_routing_active
            and process_with_model is not None
            and not self._job_tracker.has_exclusive_job_in_progress()
        ):
            target_card = self._card_runtimes.get(process_with_model.device_index)

        if target_card is not None:
            jobs_in_progress_count = len(self._jobs_in_progress_on_card(target_card.device_index))
        else:
            jobs_in_progress_count = len(self._job_tracker.jobs_in_progress)
        max_jobs_allowed = self._max_jobs_in_progress_allowed(card=target_card)
        if self._state.wants_line_skip_candidate and (
            bridge_data.aux_model_download_line_skip_threshold_seconds is None
            or not self._process_map.any_model_downloading_aux_more_than_threshold(
                threshold_seconds=bridge_data.aux_model_download_line_skip_threshold_seconds,
            )
        ):
            # The aux download that justified bypassing the in-progress cap is no longer blocking (or the
            # breaker was disabled); without this reset the cap would stay bypassed for the whole session.
            self._state.wants_line_skip_candidate = False
        if jobs_in_progress_count >= max_jobs_allowed and not self._state.wants_line_skip_candidate:
            if self._job_tracker.has_exclusive_job_in_progress():
                logger.debug(
                    "Line-skip blocked: exclusive in-progress job requires isolation "
                    f"(jobs_in_progress={jobs_in_progress_count}, cap={max_jobs_allowed}).",
                )
                return None
            if (
                process_with_model is not None
                and process_with_model.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL
            ):
                logger.debug(
                    "Line-skip considering cap bypass: head job "
                    f"{str(next_job.id_)[:8]} is blocked by aux downloads on process "
                    f"{process_with_model.process_id} "
                    f"(jobs_in_progress={jobs_in_progress_count}, cap={max_jobs_allowed}, "
                    f"max_threads={self._max_concurrent_inference_processes}, "
                    f"gpu_sampling_lease_enabled={bridge_data.gpu_sampling_lease_enabled}).",
                )
                bypass = self._select_line_skip_candidate(
                    next_job,
                    next_n_jobs=next_n_jobs,
                    candidate_job_size=candidate_job_size,
                    displaced_process_id=process_with_model.process_id,
                )
                if bypass is not None:
                    self._pending_line_skip = bypass
                    return bypass

                # So no already popped job was suitable, let's set the state accordingly so we can attempt
                # to pop a smaller job next tick
                if (
                    bridge_data.aux_model_download_line_skip_threshold_seconds is not None
                    and self._process_map.any_model_downloading_aux_more_than_threshold(
                        device_index=process_with_model.device_index,
                        threshold_seconds=bridge_data.aux_model_download_line_skip_threshold_seconds,
                    )
                ):
                    self._state.wants_line_skip_candidate = True

            # The cap may instead be occupied by an *already in-progress* job fetching its auxiliaries. In
            # that shape the pending head is perfectly ready on another resident process, so inspecting only
            # the head's process misses the idle-card opportunity. Borrow the non-sampling slot using the same
            # modest/resident/non-LoRA line-skip contract and preserve the active download as the displaced job.
            aux_blocker = self._active_aux_download_blocker(
                device_index=target_card.device_index if target_card is not None else None,
            )
            if aux_blocker is not None:
                active_aux_job, aux_process = aux_blocker
                bypass = self._select_line_skip_candidate(
                    active_aux_job,
                    next_n_jobs=next_n_jobs,
                    candidate_job_size=candidate_job_size,
                    displaced_process_id=aux_process.process_id,
                )
                if bypass is not None:
                    self._pending_line_skip = bypass
                    return bypass

            return None

        if process_with_model is None:
            if next_job.model is None:
                raise ValueError(f"next_job.model is None ({next_job})")

            # The head's model may be resident only on a disaggregation-pinned sampler lane, which the dispatch
            # query excludes so no job is ever dispatched onto a pinned lane. That copy becomes dispatchable when
            # the pin releases (its disaggregated job's sampling finishes), so the head must wait for it rather
            # than fund a second copy that cannot fit beside the pinned residents.
            pinned_resident = self._pinned_lane_resident_for_job(next_job)

            # The head's model is not resident on any dispatchable process. If it is forecast to load (a preload
            # is already on the way), or its only resident copy is on a pinned lane it is waiting to reuse, let a
            # later already-resident job bypass it so the GPU is not idle while the head waits; this reduces
            # churn versus evicting to run the head right now. If it is NOT forecast to load and not pin-waiting,
            # do not bypass: fall through so the head is the one that gets a process (and the budget gate makes
            # room for it), rather than being starved behind perpetual bypassers.
            if pinned_resident is not None or self._is_model_forecast_to_load(next_job.model):
                for candidate_job in next_n_jobs:
                    if candidate_job.model is None or candidate_job.model == next_job.model:
                        continue
                    candidate_process = self._resident_process_for_job(candidate_job)
                    if candidate_process is not None and candidate_process.can_accept_job():
                        line_skip = LineSkip(displaced_job=next_job)
                        next_job = candidate_job
                        process_with_model = candidate_process
                        break

            if process_with_model is None:
                if pinned_resident is not None:
                    # The head's only resident copy is on a disaggregation-pinned sampler lane. Never fund a
                    # fresh preload (it cannot fit beside the pinned residents and would wedge the card); hold
                    # the head's queue position until the pin releases and the resident lane becomes
                    # dispatchable, then it dispatches onto that lane priced as already resident.
                    return None

                next_job_model = next_job.model
                if next_job_model is None:
                    raise ValueError(f"next_job.model is None ({next_job})")

                if (
                    self._preload_delay_notified
                    or self._horde_model_map.is_model_loading(next_job_model)
                    or information_only
                ):
                    return None
                await self._handle_process_missing(next_job, process_with_model=process_with_model)
                return None

        if not process_with_model.can_accept_job():
            if process_with_model.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL:
                line_skip_selection = self._select_line_skip_candidate(
                    next_job,
                    next_n_jobs=next_n_jobs,
                    candidate_job_size=candidate_job_size,
                    displaced_process_id=process_with_model.process_id,
                )
                if line_skip_selection is None:
                    return None
                next_job = line_skip_selection.next_job
                process_with_model = line_skip_selection.process_with_model
                line_skip = line_skip_selection.line_skip
            else:
                # The head's own process is busy sampling its model, so the head cannot run yet. Rather than
                # idle a free inference process, fill it with a pending job for a *different* model already
                # resident there: a multi-threaded worker covers more distinct models per concurrent slot and
                # avoids duplicate-loading the head's model. The head keeps its queue position via the
                # line-skip. Falls through to None when nothing distinct is runnable (so a run of same-model
                # jobs still waits for the busy process rather than duplicating its model).
                diversity = self._select_idle_thread_diversity_job(next_job, next_n_jobs)
                if diversity is None:
                    return None
                diversity_job, diversity_process = diversity
                line_skip = LineSkip(displaced_job=next_job)
                next_job = diversity_job
                process_with_model = diversity_process

        self._model_recently_missing = False

        if (
            not information_only
            and line_skip is None
            and not self._resident_whole_card_head_ready(next_job, process_with_model)
        ):
            self._pending_line_skip = None
            return None

        # Hold a would-be concurrent job back until the in-flight job(s) have made size-appropriate
        # headway, so two heavy models (or a batch / extra-large model) do not stack their loads and
        # activation peaks on the card and thrash a sampler into a watchdog teardown. The line-skip
        # bypass is exempt: it deliberately keeps the GPU fed with a small job while another slot is
        # only downloading aux models, and is already size-limited. information_only look-ahead is not
        # gated here so callers still see the next job; the real dispatch path below enforces the hold.
        if (
            not information_only
            and line_skip is None
            and not self._concurrent_overlap_allowed(next_job, target_device_index=process_with_model.device_index)
        ):
            self._pending_line_skip = None
            return None

        next_job_and_process = NextJobAndProcess(
            next_job=next_job,
            process_with_model=process_with_model,
            line_skip=line_skip,
        )

        if line_skip is not None:
            self._pending_line_skip = next_job_and_process

        return next_job_and_process

    def _should_keep_model_resident(
        self,
        dispatched_job: ImageGenerateJobPopResponse,
        *,
        process_with_model: HordeProcessInfo,
        device_index: int | None,
    ) -> bool:
        """Whether ``dispatched_job``'s model should stay resident in VRAM after it runs.

        hordelib evicts the model from VRAM after every job so sibling GPU instances never collectively
        over-commit; that eviction forces a RAM->VRAM reload on the next job, which is the dominant
        non-sampling cost on small jobs (a same-model successor on the same process pays it for weights that
        were still on the card). Retention suppresses that eviction for one job. Because eviction is now both
        on-demand and *proven* (the device-free governor reads truthful NVML device-free, and the verified
        reclaim ladder takes residents back rung by rung with each free confirmed at the device level),
        retention no longer has to be preemptively stingy: weights stay resident while the card is healthy,
        and the ladder reclaims them the instant any overcommit picture appears. The grant needs only:

        - **Card healthy**: the device-free governor's committed state for this card is HEALTHY. A PRESSURE
          or SATURATED card is one the ladder is or may soon be reclaiming from, so it is handed no new
          resident to evict. This state is derived from the one figure a WDDM driver cannot misreport under
          demand-paging (NVML device-free), so it holds precisely in the regime measured free VRAM lies.
        - **Static fit**: the card's reported total VRAM must absorb this job's sampling peak plus the
          measurement noise buffer (and any committed in-flight reserves), after charging the sibling CUDA
          contexts and the job's own post-processing that share the card while the weights are held. The margin
          added on top of the peak is the admission noise buffer, not the operator's configured
          ``vram_reserve_mb``: that reserve is a sampling / co-residency headroom term, and enforcing it as a
          hard static-fit floor stacks it on an already activation-inclusive learned peak, denying retention on
          a small card by a few dozen MB and forcing a re-transfer every job. The total is a constant the driver
          cannot misreport under pressure, so a model too large to hold at all is refused regardless of what
          "free" claims.

        The measured admission floor is deliberately *not* re-checked here: it is the admission/dispatch
        gate's job, and retaining already-materialized weights adds zero new bytes to the card, so a measured
        veto in this seam only reintroduces the never-fires problem via committed-figure noise. Nor is sole
        residency required: the governor plus the verified ladder make a second idle resident safe (it is a
        first-class ladder reclaim candidate), so retention may keep weights warm even while a sibling holds
        its own resident model.

        No queue lookahead gates the grant. The pop cycle refills the queue immediately *after* a dispatch
        drains it, so at the dispatch instant a same-model successor is almost never visible in the pending
        set even when one arrives milliseconds later; requiring one makes retention structurally unreachable.
        Eviction is instead just-in-time: a cross-model preload that no longer fits because idle residents
        hold the card defers while the ladder evicts them, and the under-pressure reclaim overrides retention
        outright, so an unused hold costs only the interval until the next dispatch.

        A missing budget, unreported total, or unpriceable sibling overhead yields False: retention is granted
        on evidence, never assumed. Even when granted, hordelib's force-load overflow guard remains the hard
        backstop, so a wrong call degrades to a reload rather than an OOM.
        """
        model = dispatched_job.model
        if model is None:
            return False
        if not self._budget_active():
            return False
        if self._wddm_paging_active:
            # The driver is already demand-paging the worker's allocations; holding weights across jobs
            # in that regime can only deepen it.
            return False
        governor_state = self.governor_state(device_index)
        if governor_state is not GovernorState.HEALTHY:
            self._log_retention_decision(
                model,
                process_with_model,
                granted=False,
                reason=f"governor: card {governor_state.value} (the reclaim ladder holds priority over new residents)",
            )
            return False
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        if total_vram_mb is None:
            return False
        baseline = self._model_metadata.get_baseline(model)
        committed_reserve_mb = self._committed_vram_reserve_mb(device_index=device_index)
        static_charges_mb = self._retention_static_charges_mb(
            dispatched_job,
            baseline,
            process_with_model=process_with_model,
            device_index=device_index,
        )
        if static_charges_mb is None:
            self._log_retention_decision(
                model,
                process_with_model,
                granted=False,
                reason="static: sibling contexts present but per-context overhead not yet measured",
            )
            return False
        static_available_mb = total_vram_mb - static_charges_mb
        static_verdict = self._vram_budget.check_job(
            dispatched_job,
            baseline,
            static_available_mb,
            committed_reserve_mb=committed_reserve_mb,
            disaggregated=self._is_disaggregation_class_eligible(dispatched_job),
        )
        # De-stack the margin: the learned sampling peak is already activation-inclusive, and the operator's
        # configured vram_reserve_mb is a sampling / co-residency headroom term, not a static load-feasibility
        # floor. Enforcing that full reserve on top of the peak (as check_job's fits does) prices already
        # materialised weights off a small card and forces a re-transfer every job. The measurement margin for a
        # static fit is the admission noise buffer, the same slack the admission identity uses. The sibling
        # contexts (charged above at their truthful marginal) and the job's own post-processing are already
        # netted out of static_available_mb.
        predicted_mb = static_verdict.predicted_mb
        noise_mb = admission_noise_buffer_mb(total_vram_mb)
        effective_available_mb = static_available_mb - committed_reserve_mb
        granted = predicted_mb is None or (predicted_mb + noise_mb) <= effective_available_mb
        self._log_retention_decision(
            model,
            process_with_model,
            granted=granted,
            reason=(
                f"static: peak {predicted_mb} + noise {noise_mb:.0f} vs {effective_available_mb:.0f}MB "
                f"(total {total_vram_mb:.0f}MB minus sibling contexts, the job's own post-processing, and "
                f"in-flight commitments)"
            ),
        )
        return granted

    def _log_retention_decision(
        self,
        model: str,
        process_with_model: HordeProcessInfo,
        *,
        granted: bool,
        reason: str,
    ) -> None:
        """Emit the per-dispatch retention verdict with the gate figures that produced it."""
        logger.debug(
            f"VRAM retention for {model} on process {process_with_model.process_id}: "
            f"{'granted' if granted else 'denied'} ({reason})",
        )

    def _retention_static_charges_mb(
        self,
        dispatched_job: ImageGenerateJobPopResponse,
        baseline: str | None,
        *,
        process_with_model: HordeProcessInfo,
        device_index: int | None,
    ) -> float | None:
        """VRAM (MB) the retention static gate must charge on top of the job's own sampling peak.

        Two costs share the card with retained weights but are invisible to the sampling-peak estimate
        and to the committed-reserve ledger at grant time:

        - **Sibling CUDA contexts**: every other live GPU process (inference siblings, the
          post-processing lane, the on-GPU safety process) holds a context whether or not it holds a
          model. Charged at the measured marginal per-context cost (first-context overhead when no
          marginal was measured), matching how the streaming forecast counts them. Returns None when
          sibling contexts exist but no per-context cost has been measured yet: an unpriceable charge
          must deny the grant, not be waved through at zero.
        - **The job's own post-processing chain**: a job that requests post-processing runs its
          upscaler/face-fixer right after sampling, precisely while retention is holding the weights.
          Its estimated peak only enters the committed ledger after inference finishes, one dispatch
          too late for this grant, so it is charged here up front.

        Both are static estimates: the gate must hold even when the driver's free figure cannot be
        trusted (WDDM demand-paging), so nothing here reads measured free VRAM.
        """
        bridge_data = self._runtime_config.bridge_data
        safety_on_gpu = bridge_data.safety_on_gpu is True and not self._process_lifecycle.is_safety_gpu_paused
        sibling_contexts = 0
        for process_info in self._process_map.values():
            if process_info.process_id == process_with_model.process_id:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_type in (HordeProcessType.INFERENCE, HordeProcessType.POST_PROCESS) or (
                process_info.process_type == HordeProcessType.SAFETY and safety_on_gpu
            ):
                sibling_contexts += 1

        charges_mb = 0.0
        if sibling_contexts > 0:
            override_mb = self._config_overhead_override_mb()
            per_context_mb = self._overhead.marginal_mb(config_override_mb=override_mb)
            if per_context_mb is None:
                per_context_mb = self._overhead.per_process_mb(config_override_mb=override_mb)
            if per_context_mb <= 0:
                return None
            charges_mb = sibling_contexts * per_context_mb

        if dispatched_job.payload.post_processing:
            own_post_processing_mb = predict_job_post_processing_vram_mb(dispatched_job, baseline)
            if own_post_processing_mb is not None:
                charges_mb += max(0.0, own_post_processing_mb)

        return charges_mb

    def _coresident_lookahead_affordable(self, resident_model: str, *, device_index: int | None) -> bool:
        """Whether an idle resident copy of a queued model can coexist with the imminent job's sampling.

        Static accounting against the card's reported total (a constant the driver cannot misreport
        under memory pressure): the resident's full weight footprint plus the head-of-queue job's
        sampling peak plus the configured reserve must fit. On a card where they cannot, keeping the
        copy warm forces driver demand-paging during the head's sampling, which costs far more than the
        one reload the protection would have saved. Unknown figures (no total reported, no head job, no
        estimate) keep the protection: the affordability gate only ever *removes* protection on
        evidence.
        """
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        if total_vram_mb is None:
            return True
        pending = self._job_tracker.jobs_pending_inference
        head_job = pending[0] if len(pending) > 0 else None
        if head_job is None or head_job.model is None:
            return True
        head_peak_mb = predict_job_sampling_vram_mb(head_job, self._model_metadata.get_baseline(head_job.model))
        # The footprint estimator keys on the baseline alone (weights do not scale with job shape); the
        # job argument only satisfies its signature.
        resident_footprint_mb = predict_job_footprint_mb(head_job, self._model_metadata.get_baseline(resident_model))
        if head_peak_mb is None or resident_footprint_mb is None:
            return True
        return total_vram_mb - self._vram_budget.reserve_mb - head_peak_mb - resident_footprint_mb >= 0

    def pp_sampling_coresidency_affordable(
        self,
        *,
        sampling_peak_mb: float | None,
        pp_reserve_mb: float,
        device_index: int | None = None,
    ) -> bool:
        """Whether a sampling job and a post-processing chain can run on this card at the same time.

        Static accounting against the card's reported total (the driver's free figure is untrustworthy
        under WDDM demand-paging, precisely the failure this predicate prevents): the sampling peak, the
        chain's estimated peak, every extra live GPU context, and the configured reserve must all fit
        together. On a card where they cannot, co-running the two silently demand-pages both (sampling
        collapses to a fraction of its rate for the whole overlap), so the dispatch gates time-slice the
        card instead: whichever side arrives second waits for the first to finish. Unknown figures (no
        total, no peak estimate, unmeasured context cost) leave co-running allowed: this gate only ever
        restricts on evidence.
        """
        if sampling_peak_mb is None or pp_reserve_mb <= 0:
            return True
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        if total_vram_mb is None:
            return True
        override_mb = self._config_overhead_override_mb()
        per_context_mb = self._overhead.marginal_mb(config_override_mb=override_mb)
        if per_context_mb is None:
            per_context_mb = self._overhead.per_process_mb(config_override_mb=override_mb)
        if per_context_mb <= 0:
            return True
        bridge_data = self._runtime_config.bridge_data
        safety_context = (
            1 if bridge_data.safety_on_gpu is True and not self._process_lifecycle.is_safety_gpu_paused else 0
        )
        extra_contexts = (
            max(0, self._process_map.num_loaded_inference_processes(device_index=device_index) - 1) + safety_context
        )
        return (
            total_vram_mb
            - self._vram_budget.reserve_mb
            - sampling_peak_mb
            - pp_reserve_mb
            - extra_contexts * per_context_mb
            >= 0
        )

    def max_in_progress_sampling_peak_mb(self) -> float | None:
        """The largest sampling peak (MB) among jobs currently in progress, or None when idle.

        Each job's static sampling peak is raised by any learned SAMPLE-stage watermark for its footprint before
        the maximum is taken, so the post-processing co-residency gate this feeds prices in-flight sampling from
        measured activation high-waters, not a seed the hardware has already overshot.
        """
        peaks: list[float] = []
        for job in self._job_tracker.jobs_in_progress:
            if job.model is None:
                continue
            baseline = self._model_metadata.get_baseline(job.model)
            static_peak_mb = predict_job_sampling_vram_mb(job, baseline)
            if static_peak_mb is None:
                continue
            peaks.append(
                self._learned_sampling_peak_mb(
                    job,
                    baseline,
                    static_seed_mb=static_peak_mb,
                    stage=FootprintStage.SAMPLE,
                ),
            )
        if not peaks:
            return None
        return max(peaks)

    def estimate_disaggregated_sampling_peak_mb(self, job_info: HordeJobInfo) -> float | None:
        """Return a disaggregated job's estimated sampling-phase peak VRAM (MB), or None when unavailable.

        Injected into the disaggregation orchestrator's concurrent-sampling gate. Charges the whole-job
        sampling peak (:func:`predict_job_sampling_vram_mb`, weights plus the per-step activation working set),
        a deliberately conservative monolithic bound for v1 rather than the leaner sampler-only figure: the gate
        arbitrates whether two activation peaks may over-commit the card, and erring high there defers a second
        sampler that would otherwise drive the device into WDDM demand-paging. The static figure is the seed of a
        learned SAMPLE_ISOLATED-stage estimate (the disaggregated UNet-only sampler's own key, distinct from the
        monolithic whole-job SAMPLE key), so a measured isolated-sampler high-water for this (baseline,
        resolution, platform) raises the booked peak above the seed and never below it. None (no model, or no
        estimate) leaves the gate to admit, so a missing estimate never wedges the pipeline.
        """
        job = job_info.sdk_api_job_info
        if job.model is None:
            return None
        baseline = self._model_metadata.get_baseline(job.model)
        static_peak_mb = predict_job_sampling_vram_mb(job, baseline)
        if static_peak_mb is None:
            return None
        return self._learned_sampling_peak_mb(
            job,
            baseline,
            static_seed_mb=static_peak_mb,
            stage=FootprintStage.SAMPLE_ISOLATED,
        )

    def estimate_disaggregated_decode_spike_mb(self, job_info: HordeJobInfo) -> float | None:
        """Return a disaggregated job's bundled VAE-decode-plus-post-processing spike (MB), or None if unsizable.

        Injected into the disaggregation orchestrator's decode gate. The bounded tiled-decode activation working
        set (:func:`predict_job_decode_spike_mb`, the same figure the lane co-residency charge derives from) is
        the concurrent device commitment a decode adds while a sibling samples. None (no model, or a pinned
        hordelib that predates the decode-spike figure) prices the decode as unpriced, so the gate then only
        withholds a decode onto an already over-committed card rather than charging a phantom cost.
        """
        job = job_info.sdk_api_job_info
        if job.model is None:
            return None
        baseline = self._model_metadata.get_baseline(job.model)
        return predict_job_decode_spike_mb(job, str(baseline) if baseline is not None else None)

    def _vae_lane_decode_spike_charge_mb(self, *, device_index: int | None) -> float:
        """The VAE lane's concurrent decode spike (MB) to reserve out of sampling headroom, 0 when off-card.

        Under disaggregation the image lane VAE-decodes the previous job's latent while a sampler runs, so that
        bounded tiled-decode activation is a real concurrent device commitment that must not be handed to a
        second sampler. Charged only while the lane is enabled and on the GPU; sized from an in-flight job's
        bounded decode-spike estimate (via :meth:`_disaggregation_sibling_charge_mb`, which itself falls back to
        the conservative full lane quota when the pinned hordelib predates the decode-spike figure). Zero when
        no in-flight job can size it, so the headroom is never charged a phantom lane spike.
        """
        if not self._process_lifecycle.vae_lane_enabled() or self._process_lifecycle.is_vae_lane_gpu_paused:
            return 0.0
        if self._process_map.num_vae_lane_processes(device_index=device_index) <= 0:
            return 0.0
        for job in self._job_tracker.jobs_in_progress:
            if job.model is None:
                continue
            return self._disaggregation_sibling_charge_mb(
                job,
                self._model_metadata.get_baseline(job.model),
                device_index=device_index,
            )
        return 0.0

    def _sampling_peak_mb(self, job: ImageGenerateJobPopResponse) -> float | None:
        """Return the sampling peak estimate for ``job``, raised by any learned SAMPLE-stage watermark."""
        if job.model is None:
            return None
        baseline = self._model_metadata.get_baseline(job.model)
        static_peak_mb = predict_job_sampling_vram_mb(job, baseline)
        if static_peak_mb is None:
            return None
        return self._learned_sampling_peak_mb(
            job,
            baseline,
            static_seed_mb=static_peak_mb,
            stage=FootprintStage.SAMPLE,
        )

    def _pending_post_processing_reserve_mb(self, *, device_index: int | None) -> float:
        """Return the smallest known pending post-processing peak for an idle lane on ``device_index``.

        The lane orchestrator scans for the first pending chain that can run. For the dispatch-side hold, the
        smallest known pending peak is enough: if the next sampler cannot share the card with even that chain,
        starting it would extend the no-drain window for every pending chain. Unknown estimates do not hold
        inference; every memory gate in this scheduler restricts only on evidence.
        """
        post_process_process = self._process_map.get_first_available_post_process_process()
        if post_process_process is None:
            return 0.0
        if device_index is not None and post_process_process.device_index != device_index:
            return 0.0

        estimates_mb: list[float] = []
        for job_info in self._job_tracker.jobs_pending_post_processing:
            sdk_job = job_info.sdk_api_job_info
            baseline = self._model_metadata.get_baseline(sdk_job.model) if sdk_job.model is not None else None
            baseline_name = str(getattr(baseline, "value", baseline)) if baseline is not None else None
            estimate = predict_job_post_processing_vram_mb(sdk_job, baseline_name)
            if estimate is None or estimate <= 0:
                continue
            estimates_mb.append(estimate)

        if not estimates_mb:
            return 0.0
        return min(estimates_mb)

    def _should_defer_dispatch_for_post_processing(
        self,
        next_job: ImageGenerateJobPopResponse,
        *,
        process_with_model: HordeProcessInfo | None = None,
    ) -> bool:
        """Whether this dispatch must wait for post-processing to release or receive the card.

        The counterpart of the orchestrator's chain-admission gate: together they time-slice a card that
        cannot hold a sampling peak and an upscale chain at once. Active chains hold dispatch until their
        result lands. Pending chains can also hold dispatch before the next sampler starts; otherwise a
        fresh sampler can keep the card never-idle and prevent the pending lane work from ever getting its
        turn.
        """
        device_index = process_with_model.device_index if process_with_model is not None else None

        pp_committed_mb = self._reserve_ledger.total_vram_mb() - self._reserve_ledger.total_vram_mb_excluding(
            POST_PROCESS_RESERVE_FLOW,
        )
        pp_busy = any(
            process_info.process_type == HordeProcessType.POST_PROCESS
            and process_info.is_process_busy()
            and (device_index is None or process_info.device_index == device_index)
            for process_info in self._process_map.values()
        )
        if pp_committed_mb > 0 and pp_busy:
            sampling_peak_mb = self._sampling_peak_mb(next_job)
            affordable = self.pp_sampling_coresidency_affordable(
                sampling_peak_mb=sampling_peak_mb,
                pp_reserve_mb=pp_committed_mb,
                device_index=device_index,
            )
            if not affordable and not self._pp_mutex_hold_logged:
                self._pp_mutex_hold_logged = True
                logger.info(
                    f"Holding dispatch of {next_job.model}: an in-flight post-processing chain "
                    f"({pp_committed_mb:.0f}MB committed) and this job's sampling peak cannot share the card; "
                    "dispatching when the chain finishes.",
                )
            if affordable:
                self._pp_mutex_hold_logged = False
            return not affordable

        pending_pp_reserve_mb = self._pending_post_processing_reserve_mb(device_index=device_index)
        if pending_pp_reserve_mb <= 0:
            return False

        sampling_peak_mb = self._sampling_peak_mb(next_job)
        affordable = self.pp_sampling_coresidency_affordable(
            sampling_peak_mb=sampling_peak_mb,
            pp_reserve_mb=pending_pp_reserve_mb,
            device_index=device_index,
        )
        if not affordable and not self._pp_mutex_hold_logged:
            self._pp_mutex_hold_logged = True
            logger.info(
                f"Holding dispatch of {next_job.model}: pending post-processing "
                f"({pending_pp_reserve_mb:.0f}MB estimated) needs the next drain window and this job's "
                "sampling peak cannot share the card; dispatching after the lane gets its turn.",
            )
        if affordable:
            self._pp_mutex_hold_logged = False
        return not affordable

    def note_wddm_paging(self, elevated_shared_mb_by_pid: dict[int, float], *, active: bool) -> None:
        """Record the parent's WDDM demand-paging verdict and reclaim idle VRAM on its rising edge.

        ``elevated_shared_mb_by_pid`` names the worker child PIDs whose shared (system-backed) GPU usage
        crossed the paging threshold: measured attribution that the *worker's own* allocations were
        demoted out of dedicated VRAM. While active, retention is denied outright (holding weights in a
        regime the driver is already paging can only deepen it). The rising edge additionally reclaims idle
        resident VRAM, routed through the same LIFO reclaim policy the governor's ladder uses (newest idle
        resident first).

        The PDH-flagged process is deliberately NOT protected. The old sweep spared it on the assumption its
        model was the one in use, but under WDDM the driver demotes the least-recently-touched allocator, so
        the flagged process is usually the idle newcomer that just materialized weights, not the active
        sampler. Protecting it therefore spared exactly the squatter that should be evicted first. Immunity is
        instead structural: the reclaim ladder's candidate assembly excludes every actively-sampling process,
        so a busy slot is never swept whatever PDH flagged, and the newest idle resident (the likeliest
        squatter) is the first eviction target.
        """
        was_active = self._wddm_paging_active
        self._wddm_paging_active = active
        # Persist the victim set on every active verdict so recency tracks the latest sample, and clear it
        # the instant paging clears so a stale set cannot outlive the pressure that produced it.
        if active:
            self._wddm_paging_victims_shared_mb_by_pid = dict(elevated_shared_mb_by_pid)
            self._wddm_paging_victims_updated_monotonic = time.monotonic()
        else:
            self._wddm_paging_victims_shared_mb_by_pid = {}
        if not active or was_active:
            return

        detail = ", ".join(
            f"pid {pid}: {shared_mb:.0f}MB shared" for pid, shared_mb in sorted(elevated_shared_mb_by_pid.items())
        )
        logger.warning(
            "WDDM demand-paging detected on worker processes "
            f"({detail}); the driver demoted their VRAM allocations to system memory. "
            "Denying model retention and reclaiming idle resident VRAM (newest idle resident first).",
        )

        # Reclaim idle resident models in LIFO order through the single reclaim policy: build the ordered
        # ladder and issue each idle-model unload rung via the same actuator the governor's ladder uses. A
        # busy process is never a candidate, so an actively-sampling slot is untouched.
        ladder = build_reclaim_ladder(self.build_reclaim_ladder_candidates(None))
        for rung in ladder:
            if rung.kind is ReclaimRungKind.UNLOAD_IDLE_MODEL and rung.target_process_id is not None:
                self.unload_idle_model(rung.target_process_id, rung.device_index)

    def wddm_paging_victim_shared_mb_by_pid(self, max_age_seconds: float) -> dict[int, float]:
        """Return the fresh WDDM paging-victim map (os_pid -> shared MB), or empty when none is current.

        The map names the worker child PIDs whose VRAM the driver most recently demoted to system memory
        and by how much (their shared, system-backed GPU MB). It is returned only while it is younger than
        ``max_age_seconds``; a stale or absent verdict yields an empty map, so a caller can never act on a
        paging episode that has already cleared or whose telemetry has stopped arriving. The per-PID
        figures are diagnostic hints only: the counter is unreliable sample-to-sample and the demoted PID
        is usually the idle newcomer rather than the slow sampler, so no reclaim or kill decision gates on
        membership in this map.
        """
        if not self._wddm_paging_victims_shared_mb_by_pid:
            return {}
        if (time.monotonic() - self._wddm_paging_victims_updated_monotonic) > max_age_seconds:
            return {}
        return dict(self._wddm_paging_victims_shared_mb_by_pid)

    def _log_job_dispatch_details(self, next_job: ImageGenerateJobPopResponse) -> None:
        """Log the model, conditioning extras, and the resolution/steps/sampler line for a dispatching job.

        Side-effect-only diagnostics emitted just before an inference dispatch; it reads the job payload
        and writes log lines, mutating no scheduler state.
        """
        color_format_string = "<fg #f0beff>{message}</>"

        logger.opt(ansi=True).info(
            color_format_string.format(
                message=f"  Model: {next_job.model}",
            ),
        )
        if next_job.source_image is not None:
            logger.opt(ansi=True).info(
                color_format_string.format(
                    message="  Using source image",
                ),
            )

        extra_info = ""
        if next_job.payload.control_type is not None:
            extra_info += f"Control type: {next_job.payload.control_type}"
        if next_job.payload.loras:
            if extra_info:
                extra_info += ", "
            extra_info += f"{len(next_job.payload.loras)} LoRAs"
        if next_job.payload.tis:
            if extra_info:
                extra_info += ", "
            extra_info += f"{len(next_job.payload.tis)} TIs"
        if next_job.payload.post_processing is not None and len(next_job.payload.post_processing) > 0:
            if extra_info:
                extra_info += ", "
            extra_info += f"Post processing: {next_job.payload.post_processing}"
        if next_job.payload.hires_fix:
            if extra_info:
                extra_info += ", "
            extra_info += "HiRes fix"

        if next_job.payload.workflow is not None:
            if extra_info:
                extra_info += ", "
            extra_info += f"Workflow: {next_job.payload.workflow}"

        if extra_info:
            logger.opt(ansi=True).info(
                color_format_string.format(
                    message=f"  {extra_info}",
                ),
            )

        logger.opt(ansi=True).info(
            color_format_string.format(
                message=f"  {next_job.payload.width}x{next_job.payload.height} for "
                f"{next_job.payload.ddim_steps} steps "
                f"with sampler {next_job.payload.sampler_name} for a batch of {next_job.payload.n_iter}",
            ),
        )

        logger.debug(f"All Batch IDs: {next_job.ids}")

    async def _dispatch_disaggregated(
        self,
        next_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
        *,
        dispatched_device_index: int | None,
        degraded_dispatch: bool,
    ) -> bool:
        """Register a disaggregation-eligible job with the orchestrator, pinned to its scheduled process.

        Replaces the monolithic START_INFERENCE at this seam: the orchestrator reserves ``process_with_model``
        as the job's sampler (so the scheduler cannot double-book it), and this applies the same job-progress
        marking a monolithic dispatch does, so concurrency accounting and the orphaned-job watchdog see the
        job as owned. Returns False when the router declines (a role went unhealthy), so the caller falls back
        to a monolithic dispatch.
        """
        assert self._register_disaggregated_job is not None
        model = next_job.model
        if model is None:
            return False
        registered = await self._register_disaggregated_job(next_job, process_with_model)
        if not registered:
            return False

        await self._job_tracker.mark_inference_started(next_job, device_index=dispatched_device_index)
        # The pinned process references this job so the orphaned-job watchdog credits it as owned across the
        # whole encode-and-sample window (the reservation, not a START_INFERENCE flag, is that ownership
        # record: see WorkerRecoveryCoordinator.inference_slot_owns_job). No sampling-timing stamp is set here;
        # the sampler reports its own INFERENCE_STARTING when the sample stage runs.
        process_with_model.last_job_referenced = next_job
        process_with_model.loaded_horde_model_name = model
        process_with_model.loaded_horde_model_baseline = self._model_metadata.get_baseline(model)
        if degraded_dispatch:
            self._job_tracker.clear_degraded_dispatch(next_job)
        self._process_lifecycle.action_ledger.record(
            LedgerEventType.INFERENCE_DISPATCHED,
            process_id=process_with_model.process_id,
            os_pid=process_with_model.os_pid,
            launch_identifier=process_with_model.process_launch_identifier,
            job_id=str(next_job.id_) if next_job.id_ is not None else None,
            detail={"model": model, "disaggregated": True},
        )
        logger.opt(ansi=True).info(
            f"<fg #f0beff>Job {str(next_job.id_)[:8]} routed to the disaggregated pipeline; sampler pinned "
            f"to process {process_with_model.process_id}.</>",
        )
        return True

    async def _dispatch_inference_message(
        self,
        next_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
        *,
        keep_model_resident_after: bool,
        dispatched_device_index: int | None,
        degraded_dispatch: bool,
    ) -> None:
        """Send the START_INFERENCE command and record the outcome: mark started on success, fault on failure.

        On a successful send, marks the job started, stamps the slot's in-flight timing for the
        graded-slowdown monitor, records the dispatch in the action ledger, and advances the process state.
        On a failed send, faults the job so the horde reissues it.

        A disaggregation-eligible job is not sent monolithic inference here: this is the single admission
        point, so the scheduler has already preloaded its model onto ``process_with_model`` exactly as for a
        monolithic job, and instead of START_INFERENCE the job is registered with the orchestrator pinned to
        that process as its sampler. All the in-flight/job-progress marking a monolithic dispatch performs is
        applied identically (so the process cannot be double-booked), minus the sampling-timing stamps, which
        the sampler reports itself when the sample stage runs.
        """
        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

        # A disaggregation-eligible job is registered with the orchestrator (its sampler pinned to
        # process_with_model) instead of being sent monolithic inference. If registration is declined (a role
        # went unhealthy between the eligibility check and here) the dispatch falls through to the monolithic
        # path so the job still runs whole rather than being dropped.
        if (
            self._register_disaggregated_job is not None
            and self._is_disaggregatable_job(next_job)
            and await self._dispatch_disaggregated(
                next_job,
                process_with_model,
                dispatched_device_index=dispatched_device_index,
                degraded_dispatch=degraded_dispatch,
            )
        ):
            return

        bridge_data = self._runtime_config.bridge_data
        if process_with_model.safe_send_message(
            HordeInferenceControlMessage(
                control_flag=HordeControlFlag.START_INFERENCE,
                horde_model_name=next_job.model,
                sdk_api_job_info=next_job,
                keep_model_resident_after=keep_model_resident_after,
                aux_download_deadline_seconds=self._process_lifecycle.aux_download_deadline_for_dispatch(
                    bridge_data,
                ),
            ),
        ):
            await self._job_tracker.mark_inference_started(next_job, device_index=dispatched_device_index)
            horde_model_baseline = self._model_metadata.get_baseline(next_job.model)
            self._record_dispatch_reservation(next_job, process_with_model, baseline=horde_model_baseline)

            dispatch_detail: dict[str, str | int | float | bool | None] = {
                "model": next_job.model,
                "steps": next_job.payload.ddim_steps,
            }
            expected_seconds = self._expected_sampling_seconds(next_job, horde_model_baseline)
            if expected_seconds is not None:
                dispatch_detail["expected_sampling_seconds"] = round(expected_seconds, 2)

            # Stamp the in-flight timing onto the slot so the graded-slowdown monitor can measure this
            # job against its expected sampling time; the level resets so notices escalate per dispatch.
            process_with_model.current_inference_started_at = time.time()
            process_with_model.current_first_step_at = None
            process_with_model.current_job_expected_sampling_seconds = expected_seconds
            process_with_model.current_job_slowdown_level = 0
            process_with_model.consecutive_slow_per_steps = 0
            process_with_model.current_job_per_step_floor_tripped = False

            if degraded_dispatch:
                self._job_tracker.clear_degraded_dispatch(next_job)
                dispatch_detail["degraded_retry"] = True
                logger.warning(
                    f"  Degraded, isolated retry dispatched for job {str(next_job.id_)[:8]} "
                    "after a prior resource failure.",
                )

            self._process_lifecycle.action_ledger.record(
                LedgerEventType.INFERENCE_DISPATCHED,
                process_id=process_with_model.process_id,
                os_pid=process_with_model.os_pid,
                launch_identifier=process_with_model.process_launch_identifier,
                job_id=str(next_job.id_) if next_job.id_ is not None else None,
                detail=dispatch_detail,
            )

            process_with_model.last_control_flag = HordeControlFlag.START_INFERENCE
            process_with_model.last_job_referenced = next_job
            process_with_model.loaded_horde_model_name = next_job.model
            process_with_model.loaded_horde_model_baseline = horde_model_baseline
            self._process_map.on_process_state_change(
                process_id=process_with_model.process_id,
                new_state=HordeProcessState.INFERENCE_STARTING,
            )

        else:
            logger.error(
                f"Failed to start inference for job {next_job.id_} on process {process_with_model.process_id}",
            )
            await self._job_tracker.handle_job_fault(
                faulted_job=next_job,
                process_info=process_with_model,
                process_timeout=bridge_data.process_timeout,
            )

    def _prune_abandoned_dispatch_holds(self) -> None:
        """Drop hold bookkeeping for jobs no longer pending inference (rerouted, faulted, or dispatched).

        An abandoned hold (its job left the pending queue by some path other than a release through the gate)
        is not a release, so it advances neither release counter; it is simply forgotten so the maps stay
        bounded to the live queue.
        """
        pending_ids = {str(job.id_) for job in self._job_tracker.jobs_pending_inference if job.id_ is not None}
        for held_id in [held_id for held_id in self._dispatch_hold_since if held_id not in pending_ids]:
            self._dispatch_hold_since.pop(held_id, None)
            self._dispatch_hold_reclaim_requested.discard(held_id)

    def _note_dispatch_hold(self, job: ImageGenerateJobPopResponse, *, reclaim_requested: bool) -> None:
        """Record that the dispatch of ``job`` was held this pass, stamping the first hold and its cause.

        Every hold pass counts a conflict; the first hold for a job also stamps the hold-start instant and
        counts a distinct held dispatch. A pass that emitted eviction commands marks the job so its eventual
        release is attributed to reclaim rather than to the card recovering on its own.
        """
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None:
            return
        self._dispatch_reconciliation_conflicts += 1
        if job_id not in self._dispatch_hold_since:
            self._dispatch_hold_since[job_id] = time.time()
            self._dispatch_reconciliation_holds += 1
        if reclaim_requested:
            self._dispatch_hold_reclaim_requested.add(job_id)

    def _resolve_dispatch_hold(self, job: ImageGenerateJobPopResponse) -> None:
        """Close out any dispatch hold on ``job`` now that it fits, folding its duration and release cause.

        A no-op for a job that was never held (the common admit-first-pass case). A held job's accumulated
        wait folds into the cumulative hold seconds, and the release is attributed to reclaim when this gate
        emitted eviction commands during the hold, otherwise to the card freeing on its own.
        """
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None:
            return
        held_since = self._dispatch_hold_since.pop(job_id, None)
        if held_since is None:
            self._dispatch_hold_reclaim_requested.discard(job_id)
            return
        self._dispatch_reconciliation_hold_seconds += max(0.0, time.time() - held_since)
        if job_id in self._dispatch_hold_reclaim_requested:
            self._dispatch_reconciliation_released_by_reclaim += 1
            self._dispatch_hold_reclaim_requested.discard(job_id)
        else:
            self._dispatch_reconciliation_released_by_natural_free += 1

    def _dispatch_residency_reconciliation_holds(
        self,
        next_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
        *,
        is_head_of_queue: bool = True,
        head_outstanding_mb: float | None = None,
    ) -> bool:
        """Return whether ``next_job``'s dispatch must be held because its VRAM would over-commit the card now.

        The dispatch of an already-RAM-staged job is the moment its weights and first activation actually
        materialise on the device. Admission is consulted at preload and at the second-concurrent-sampler seam,
        but not here, so a job whose materialisation lands beside an idle sibling's still-resident weights can
        tip the card over the paging cliff faster than the tick-paced reclaim reacts. This gate closes that
        seam by pricing the dispatch through the arbiter's single MONOLITHIC_DISPATCH identity (the same
        measured-truth admission math the preload and overlap seams use, testing the candidate against the
        truthful device-free reading net of the outstanding reservations and the noise buffer): a FITS releases
        the dispatch, a DEFER or DENY holds it.

        On a hold the job is never faulted: it keeps its queue position and re-asks on the next scheduling pass.
        The conflicting idle residents are evicted through the one reclaim owner (the same
        :meth:`_execute_preload_actuations` surface the arbiter's preload-DEFER path drives), never inline and
        never through a second ladder; the head's own target slot is protected. The hold releases only once the
        arbiter next verdicts FITS, matching the verified-reclaim doctrine that a demand is admitted into
        measured reality rather than into hope. Can't-fit-ever models are excluded upstream by model
        serviceability, so this gate only ever holds a can't-fit-now dispatch.

        ``is_head_of_queue`` is the truth of whether this dispatch is the genuine head of queue: a line-skip
        dispatch (a smaller ready job selected ahead of a downloading head) is not, so it presents
        ``is_head_of_queue=False`` and ``head_outstanding_mb`` priced from the head it jumped. Head protection
        then holds the line-skipper when admitting it would leave the card without the room the head needs, so
        the head keeps first claim on the physical space rather than being starved behind the skipper. A
        line-skip likewise routes its reclaim through the non-head eviction path, which respects the residency
        and queued-lookahead guards the head escalation would override.
        """
        self._prune_abandoned_dispatch_holds()

        if next_job.model is None:
            return False

        device_index = process_with_model.device_index if self._multi_gpu_routing_active else None
        baseline = self._model_metadata.get_baseline(next_job.model)
        has_reclaimable_idle_model = self._has_reclaimable_idle_model(
            process_with_model,
            for_head_of_queue=is_head_of_queue,
            device_index=device_index,
        )
        candidate_delta_mb = self._measured_admission_candidate_delta_mb(
            next_job,
            baseline,
            process_id=process_with_model.process_id,
            disaggregated=self._is_disaggregation_class_eligible(next_job),
        )
        forecast = self._forecast_streaming(next_job, baseline, device_index=device_index)
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        structural_reserve_mb = (
            effective_inference_reserve_mb(total_vram_mb, 0.0)
            if total_vram_mb is not None
            else forecast._effective_base_reserve  # noqa: SLF001 - same budget owner sizes teardown depth.
        )
        max_resident = (
            self._max_coresident_for_peak_mb(
                candidate_delta_mb,
                structural_reserve_mb,
                device_index=device_index,
            )
            if candidate_delta_mb is not None
            else None
        )
        live_inference_processes = self._process_map.num_loaded_inference_processes(device_index=device_index)
        idle_contexts_teardownable = (
            is_head_of_queue
            and max_resident is not None
            and max_resident < live_inference_processes
            and self._has_teardownable_idle_context(process_with_model, device_index=device_index)
        )
        request = VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label=str(next_job.model),
            baseline=baseline,
            device_index=device_index,
            target_process_id=process_with_model.process_id,
            candidate_delta_mb=candidate_delta_mb,
            candidate_already_resident=self._candidate_weights_resident_on_process(
                next_job.model,
                process_with_model.process_id,
            ),
            own_planned_unmaterialized_mb=self._own_planned_charge_mb(
                device_index=device_index,
                target_process_id=process_with_model.process_id,
            ),
            is_head_of_queue=is_head_of_queue,
            head_outstanding_mb=head_outstanding_mb,
            starved_seconds=self._head_starved_seconds(next_job),
            has_reclaimable_idle_model=has_reclaimable_idle_model,
            # An ordinary staged dispatch never reduces the live inference-context count: it evicts idle
            # residents to make room, it does not collapse the co-resident pool (can_reduce_live_contexts stays
            # False, so the ordinary activation-peak warrant never tears a context down). The one exception is a
            # starved head whose deficit is held by its own bare idle sibling contexts with no reality-admit and
            # no weight reclaim left: it escalates to the same verified teardown the preload seam uses, so this
            # head-only signal is reported for that path.
            can_reduce_live_contexts=False,
            idle_contexts_teardownable=idle_contexts_teardownable,
        )
        verdict = self._ensure_preload_arbiter().evaluate(request)

        if verdict.admits:
            self._resolve_dispatch_hold(next_job)
            return False

        # The dispatch cannot land yet. Route the described idle-resident eviction through the single reclaim
        # owner, protecting the head's own slot, then hold. The job re-asks next pass and releases once the
        # arbiter verdicts FITS (the governor's device-free reading having verified the reclaimed room).
        actuations = verdict.required_actuations
        self._preload_actuation = _PreloadActuation(
            job=next_job,
            available_process=process_with_model,
            forecast=forecast,
            max_resident=max_resident,
        )
        try:
            self._execute_preload_actuations(
                actuations,
                device_index=device_index,
                for_head_of_queue=is_head_of_queue,
            )
        finally:
            self._preload_actuation = None
        self._note_dispatch_hold(next_job, reclaim_requested=bool(actuations))

        suppressed = self._scheduler_diagnostic_suppressed_count(
            "dispatch_residency_hold",
            (str(next_job.id_), verdict.disposition.value),
        )
        if suppressed is not None:
            logger.opt(ansi=True).warning(
                f"<fg #f0beff>Holding dispatch of {next_job.model} to reconcile residency: {verdict.reason}. "
                "Evicting idle VRAM so the job's materialisation fits the card before it commits to VRAM.</>",
            )
        return True

    def latest_dispatch_reconciliation_holds(self) -> int:
        """Return the count of dispatches held for residency reconciliation this run (calibration visibility)."""
        return self._dispatch_reconciliation_holds

    def latest_dispatch_reconciliation_conflicts(self) -> int:
        """Return the count of dispatch-time residency conflicts detected this run (calibration visibility)."""
        return self._dispatch_reconciliation_conflicts

    def latest_dispatch_reconciliation_hold_seconds(self) -> float:
        """Return the cumulative seconds dispatches spent held for residency reconciliation (calibration)."""
        return self._dispatch_reconciliation_hold_seconds

    def latest_dispatch_reconciliation_released_by_reclaim(self) -> int:
        """Return the count of held dispatches released after this gate's eviction freed room (calibration)."""
        return self._dispatch_reconciliation_released_by_reclaim

    def latest_dispatch_reconciliation_released_by_natural_free(self) -> int:
        """Return the count of held dispatches released by the card recovering on its own (calibration)."""
        return self._dispatch_reconciliation_released_by_natural_free

    def latest_safety_placement_demotions(self) -> int:
        """Return how many times the runtime safety-placement policy moved safety off-GPU this run."""
        return self._safety_placement_demotions

    def latest_safety_placement_promotions(self) -> int:
        """Return how many times the runtime safety-placement policy restored safety to the GPU this run."""
        return self._safety_placement_promotions

    def latest_safety_placement_card(self) -> int | None:
        """Return the card the safety process currently occupies, or None when safety is off-GPU (on CPU)."""
        return self._process_lifecycle.safety_gpu_card_index()

    async def start_inference(self) -> bool:
        """Start inference for the next job in jobs_pending_inference, if possible.

        During graceful shutdown the worker keeps draining the queue it already popped rather than
        faulting it: the job popper stops accepting NEW jobs once shutdown is armed, so the only jobs
        that can dispatch here are ones accepted before the stop. They are given a chance to finish,
        bounded by the per-job-scaled shutdown grace and the force-kill backstop; whatever genuinely
        cannot finish in time is still fault-reported so the horde reissues it promptly.
        """
        next_job_and_process = await self.get_next_job_and_process()

        if next_job_and_process is None:
            return False

        bridge_data = self._runtime_config.bridge_data
        process_with_model = next_job_and_process.process_with_model
        next_job = next_job_and_process.next_job

        degraded_dispatch = self._job_tracker.is_degraded_dispatch_pending(next_job)
        if degraded_dispatch and len(self._job_tracker.jobs_in_progress) > 0:
            # A degraded retry (after a resource/OOM failure) runs in isolation to minimise VRAM
            # pressure: defer it until no other job is sampling. It keeps its head-of-queue position, so
            # it dispatches as soon as the in-flight jobs drain rather than being starved.
            return False

        if self._prestaged_whole_card_not_ready(next_job):
            # The head was pre-staged into RAM while another job held the device; sampling commits its
            # weights to VRAM, so it must wait for the residency to finish collapsing to sole residency
            # (idle siblings stopped, safety off-GPU, the card drained) before it starts. Otherwise a
            # lingering sibling context would force its first step to stream over the bus. It keeps its
            # head-of-queue position, so it dispatches the moment the residency converges.
            return False

        if self._should_defer_dispatch_for_post_processing(next_job, process_with_model=process_with_model):
            # Post-processing either holds the card or is waiting for the active sampler to drain, and this
            # job's sampling peak cannot share the card with it.
            return False

        line_skip = next_job_and_process.line_skip
        if self._dispatch_residency_reconciliation_holds(
            next_job,
            process_with_model,
            # A line-skip dispatch is not the true head of queue, so it presents is_head_of_queue=False and the
            # head it jumped priced as head_outstanding_mb. Head protection then reserves the card's physical
            # room for that head rather than letting the skipper consume the space the head needs.
            is_head_of_queue=line_skip is None,
            head_outstanding_mb=(
                None
                if line_skip is None
                else self._displaced_head_outstanding_mb(
                    line_skip.displaced_job,
                    device_index=process_with_model.device_index if self._multi_gpu_routing_active else None,
                )
            ),
        ):
            # The staged job's VRAM materialisation would over-commit the card against an idle sibling's
            # still-resident weights: hold the dispatch (the job keeps its head-of-queue position) while the
            # single reclaim owner evicts the idle residents, and re-ask next pass once the arbiter verifies the
            # reclaimed room fits it. This is the seam where RAM-staged weights actually commit to VRAM, which
            # neither the preload nor the second-sampler admission consult.
            return False

        if next_job_and_process.line_skip is not None:
            logger.info(
                f"Job {next_job_and_process.next_job.id_} skipped the line and will be run on process "
                f"{process_with_model.process_id} before job {next_job_and_process.line_skip.displaced_job.id_}"
                " which is currently downloading extra models.",
            )

        if bridge_data.unload_models_from_vram_often:
            self.unload_models_from_vram(process_with_model)

        if degraded_dispatch:
            # Reclaim VRAM from idle slots before the degraded retry so the job has the best chance to
            # fit, independent of the unload_models_from_vram_often setting.
            self.unload_models_from_vram(process_with_model)

        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

        process_with_model.batch_amount = next_job.payload.n_iter
        # Record the card this job runs on (None on a single-GPU host) so its over-budget fault streak is
        # kept per card: a model unservable on a small card can still be advertised and run on a larger one.
        dispatched_device_index = process_with_model.device_index if self._multi_gpu_routing_active else None
        keep_model_resident_after = self._should_keep_model_resident(
            next_job,
            process_with_model=process_with_model,
            device_index=dispatched_device_index,
        )

        # Past every hold/fault gate: this job is dispatching now, so emit the start logging here rather than
        # before the reclaim decision (where a deferred or faulted job would mislead the log as "starting").
        color_format_string = "<fg #f0beff>{message}</>"
        logger.opt(ansi=True).info(
            color_format_string.format(
                message=f"Starting inference for job {str(next_job.id_)[:8]} "
                f"on process {process_with_model.process_id}",
            ),
        )
        self._log_job_dispatch_details(next_job)

        await self._dispatch_inference_message(
            next_job,
            process_with_model,
            keep_model_resident_after=keep_model_resident_after,
            dispatched_device_index=dispatched_device_index,
            degraded_dispatch=degraded_dispatch,
        )

        self._pending_line_skip = None

        # A job dispatched: any prior stall reason is now stale. Clear it so the
        # orchestrator intent's "Holding dispatch" does not stick after the stall resolves.
        self._dispatch_stall_last_reason = None

        return True

    def _compute_wanted_models(self) -> set[str]:
        """The set of models the worker is actively serving right now.

        Derived from live scheduler state; every model currently resident on an inference
        process, plus every model referenced by a pending or in-progress job. This mirrors the
        affinity computation in :meth:`preload_models`; ``bridge_data.image_models_to_load`` is
        deliberately not used because the harness/canned path never resolves that config field,
        so live state is the only reliable source.
        """
        wanted: set[str] = {
            p.loaded_horde_model_name
            for p in self._process_map.values()
            if p.process_type == HordeProcessType.INFERENCE and p.loaded_horde_model_name is not None
        }
        wanted.update(j.model for j in self._job_tracker.jobs_pending_inference if j.model is not None)
        wanted.update(j.model for j in self._job_tracker.jobs_in_progress if j.model is not None)
        return wanted

    def _is_model_forecast_to_load(self, model_name: str | None) -> bool:
        """Whether ``model_name`` is already on track to become resident soon.

        True when the model map marks it loading, or an inference process is currently preloading it or
        already holds it preloaded. In that case the job needing it will get a process shortly, so a
        later already-resident job may bypass it to keep the GPU fed rather than the worker idling until
        the load completes. When the model is *not* forecast to load, no bypass is allowed so the budget
        gate's room-making runs and that job makes progress instead of being starved behind bypassers.
        """
        if model_name is None:
            return False
        if self._horde_model_map.is_model_loading(model_name):
            return True
        return any(
            process.process_type == HordeProcessType.INFERENCE
            and process.loaded_horde_model_name == model_name
            and process.last_process_state in (HordeProcessState.PRELOADING_MODEL, HordeProcessState.PRELOADED_MODEL)
            for process in self._process_map.values()
        )

    def _refresh_model_demand(self) -> None:
        """Stamp the current time against every model with live demand (pending/in-progress job).

        Feeds the residency grace period (:meth:`_is_recently_demanded`). Only genuine demand,
        not mere residency, refreshes the stamp, so a loaded-but-idle model's grace still
        expires. Entries well past the grace window are pruned to bound the dict.
        """
        now = time.time()
        for job in (*self._job_tracker.jobs_pending_inference, *self._job_tracker.jobs_in_progress):
            if job.model is not None:
                self._model_last_in_demand[job.model] = now

        cutoff = now - (_RESIDENCY_GRACE_SECONDS * 4)
        for model_name in [m for m, last in self._model_last_in_demand.items() if last < cutoff]:
            del self._model_last_in_demand[model_name]

    def _is_recently_demanded(self, model_name: str) -> bool:
        """Whether the model had a pending/in-progress job within the residency grace window."""
        last = self._model_last_in_demand.get(model_name)
        return last is not None and (time.time() - last) <= _RESIDENCY_GRACE_SECONDS

    def _residency_protects_from_unload(
        self,
        model_name: str | None,
        wanted_models: set[str],
        *,
        vram: bool,
        under_pressure: bool = False,
    ) -> bool:
        """Whether residency policy should keep ``model_name`` loaded rather than evict it now.

        Two regimes:

        - **Working set fits the process count** (``affinity_active``): every actively-served
          model can have its own home process, so keep them all resident in both RAM and VRAM.
          This is the regime the soak measures and the dominant duty-cycle win; it stops a
          process evicting the very model it just used (and is about to reuse) the instant its
          next job has not yet been popped.
        - **More models than processes**: residency cannot be guaranteed, so apply only a RAM
          grace period; cheap to hold, and it avoids the expensive disk reload between a model's
          consecutive jobs. VRAM, the scarce resource, is still reclaimed promptly.

        ``under_pressure`` is the measured-budget override (the WS-1 "aggregate budget"): the
        fits-regime assumption that model-count <= process-count implies the resident set fits the
        device only holds for sd15-class weights, so when the VRAM (or RAM) budget reports the resource
        cannot absorb the next job, residency protection for that resource is dropped to let an idle
        resident model be evicted. It never overrides the in-progress / next-model guards in the caller,
        so live and imminent work is still never evicted.
        """
        if model_name is None:
            return False

        if under_pressure:
            # A model holding a whole-card residency must never be evicted from VRAM, even under
            # budget pressure: evicting it undermines the residency convergence (the pre-staged head
            # cannot reach sole residency and dispatch is permanently blocked until save-our-ship
            # soft-resets the pools). Only the residency holder is spared; other models are still
            # reclaimable.
            return any(state.model == model_name for _, state in self._held_residencies())

        if affinity_active(len(wanted_models), self._max_inference_processes) and model_name in wanted_models:
            return True

        return not vram and self._is_recently_demanded(model_name)

    def unload_post_process_models_from_vram(self, *, device_index: int | None = None) -> bool:
        """Ask an idle post-processing lane to unload its modules while keeping the lane alive."""
        unloaded_any = False
        for process_info in self._process_map.values():
            if process_info.process_type is not HordeProcessType.POST_PROCESS:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.is_process_busy():
                logger.debug(f"Post-processing process {process_info.process_id} is busy")
                continue
            if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                continue

            logger.info(f"Unloading post-processing models from VRAM on process {process_info.process_id}")
            if (
                not process_info.safe_send_message(
                    HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM),
                )
                and not self._state.shutting_down
            ):
                logger.warning(
                    f"Failed to send UNLOAD_MODELS_FROM_VRAM to post-processing process "
                    f"{process_info.process_id}; marking the lane for replacement.",
                )
                self._process_lifecycle.post_process_processes_should_be_replaced = True
            process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
            unloaded_any = True
            self._record_churn("vram_eviction")
        return unloaded_any

    def release_allocator_cache(self, process_id: int) -> bool:
        """Ask one process to release its torch allocator cache without unloading its models.

        The cache-only reclaim actuator: an arbiter RELEASE_CACHE command and the post-stage lane policy
        both land here, returning an allocator's reserved-but-unused device blocks to the card without
        evicting any resident model. Mirrors the safe-send path of the unload senders. Returns True if
        the flag was delivered, False if the process is absent or the send failed.
        """
        process_info = self._process_map.get(process_id)
        if process_info is None:
            return False
        delivered = process_info.safe_send_message(
            HordeControlMessage(control_flag=HordeControlFlag.RELEASE_ALLOCATOR_CACHE),
        )
        if delivered:
            logger.debug(f"Asked process {process_id} to release its allocator cache")
        return delivered

    def unload_models_from_vram(
        self,
        process_with_model: HordeProcessInfo,
        *,
        under_pressure: bool = False,
        for_head_of_queue: bool = False,
        device_index: int | None = None,
    ) -> bool:
        """Unload models from VRAM from processes that are not running a job.

        ``under_pressure`` (set by the VRAM budget when the next job does not fit) drops residency
        protection and the single-model hold-back so the coldest idle resident copy is reclaimed,
        while still never touching an in-progress or next-up model.

        ``for_head_of_queue`` is the last-resort escalation when the head-of-queue job cannot be loaded
        and gentle reclaim freed nothing because every idle resident copy is another *queued* job's
        model: it additionally overrides the queued-lookahead guard so the head can be given room. It
        never evicts an in-progress (live) model.

        ``device_index`` restricts eviction to idle resident copies on that one card: reclaiming VRAM for a
        load onto card C must evict from card C, since freeing another card's model returns no VRAM to C.
        None (the single-GPU / worker-wide case) considers every card's idle residents.

        Returns True if an idle resident model's unload was issued (room is on the way), False if there
        was nothing to reclaim.
        """
        bridge_data = self._runtime_config.bridge_data
        wanted_models = self._compute_wanted_models()
        next_n_models = list(self.get_next_n_models(self._max_inference_processes))
        self._log_next_models_for_vram_unload(
            next_n_models,
            under_pressure=under_pressure,
            for_head_of_queue=for_head_of_queue,
        )

        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}

        unloaded_any = False
        for process_info in self._process_map.values():
            if process_info.process_id == process_with_model.process_id:
                continue

            if process_info.process_type == HordeProcessType.POST_PROCESS:
                if device_index is not None and process_info.device_index != device_index:
                    continue

                if process_info.is_process_busy():
                    logger.debug(f"Post-processing process {process_info.process_id} is busy")
                    continue

                if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                    continue

                logger.info(f"Unloading post-processing models from VRAM on process {process_info.process_id}")
                if (
                    not process_info.safe_send_message(
                        HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM),
                    )
                    and not self._state.shutting_down
                ):
                    logger.warning(
                        f"Failed to send UNLOAD_MODELS_FROM_VRAM to post-processing process "
                        f"{process_info.process_id}; marking the lane for replacement.",
                    )
                    self._process_lifecycle.post_process_processes_should_be_replaced = True
                process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
                unloaded_any = True
                self._record_churn("vram_eviction")
                continue

            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if device_index is not None and process_info.device_index != device_index:
                continue

            if process_info.is_process_busy():
                logger.debug(f"Process {process_info.process_id} is busy")

            if process_info.loaded_horde_model_name is not None:
                if len(bridge_data.image_models_to_load) == 1 and not under_pressure:
                    logger.debug("Not unloading models from VRAM because there is only one model to load.")
                    continue

                if process_info.loaded_horde_model_name in in_progress_models:
                    continue

                # Spare the resident copy of ANY model in the queue lookahead, not just one of them:
                # evicting a queued model's weights trades the room gained for a guaranteed reload when
                # its job's turn comes. The protection is affordability-gated: on a card that cannot
                # physically hold the resident's footprint alongside the imminent job's sampling peak,
                # sparing it would force driver demand-paging (far costlier than the reload it saves),
                # so the copy is evicted. The head-of-queue escalation overrides the protection outright
                # (the head has priority for the room when every idle resident copy is another queued
                # job's model).
                if (
                    process_info.loaded_horde_model_name in next_n_models
                    and not for_head_of_queue
                    and self._coresident_lookahead_affordable(
                        process_info.loaded_horde_model_name,
                        device_index=device_index,
                    )
                ):
                    continue

                if not for_head_of_queue and self._residency_protects_from_unload(
                    process_info.loaded_horde_model_name,
                    wanted_models,
                    vram=True,
                    under_pressure=under_pressure,
                ):
                    continue

                if process_info.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                    logger.info(
                        f"Unloading model {process_info.loaded_horde_model_name} from VRAM on process "
                        f"{process_info.process_id}",
                    )
                    process_info.safe_send_message(
                        HordeControlModelMessage(
                            control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
                            horde_model_name=process_info.loaded_horde_model_name,
                        ),
                    )
                    process_info.last_job_referenced = None
                    process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
                    unloaded_any = True
                    self._record_churn("vram_eviction")
            else:
                if process_info.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                    logger.debug(f"Unloading all models from VRAM on process {process_info.process_id}")
                    if (
                        not process_info.safe_send_message(
                            HordeControlMessage(
                                control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
                            ),
                        )
                        and not self._state.shutting_down
                    ):
                        logger.warning(
                            f"Failed to send UNLOAD_MODELS_FROM_VRAM to process {process_info.process_id}. ",
                            "This may indicate the process is unresponsive or has already exited. "
                            "Attempting to replace the process with a new one.",
                        )
                        self._process_lifecycle._replace_inference_process(process_info)
                    process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
                    unloaded_any = True

        return unloaded_any

    def unload_from_ram(self, process_id: int) -> None:
        """Unload models from a process."""
        if process_id not in self._process_map:
            raise ValueError(f"process_id {process_id} is not in the process map")

        process_info = self._process_map[process_id]

        if process_info.process_type == HordeProcessType.POST_PROCESS:
            if process_info.is_process_busy():
                logger.warning(f"Post-processing process {process_id} is busy, not unloading models from RAM")
                return

            logger.debug(f"Unloading post-processing models from RAM on process {process_id}")
            process_info.safe_send_message(
                HordeControlMessage(
                    control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
                ),
            )
            process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM
            self._process_map.on_model_ram_clear(process_id=process_id)
            return

        if process_info.process_type != HordeProcessType.INFERENCE:
            logger.warning(
                f"Process {process_id} is not an inference or post-processing process, not unloading models"
            )
            return

        if process_info.recently_unloaded_from_ram:
            return

        if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            return

        if process_info.loaded_horde_model_name is not None and self._horde_model_map.is_model_loaded(
            process_info.loaded_horde_model_name,
        ):
            logger.debug(f"Unloading model {process_info.loaded_horde_model_name} from RAM on process {process_id}")
            process_info.safe_send_message(
                HordeControlModelMessage(
                    control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
                    horde_model_name=process_info.loaded_horde_model_name,
                ),
            )

            self._horde_model_map.update_entry(
                horde_model_name=process_info.loaded_horde_model_name,
                load_state=ModelLoadState.ON_DISK,
                process_id=process_id,
            )

            process_info.last_job_referenced = None
            process_info.loaded_horde_model_name = None
            process_info.loaded_horde_model_baseline = None
            process_info.recently_unloaded_from_ram = True
            process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM

        else:
            if (
                process_info.last_process_state == HordeProcessState.PROCESS_ENDING
                or process_info.last_process_state == HordeProcessState.PROCESS_ENDED
            ):
                return

            logger.debug(f"Unloading all models from RAM on process {process_id}")
            process_info.safe_send_message(
                HordeControlMessage(
                    control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
                ),
            )
        logger.debug(f"Clearing process {process_id} of model {process_info.loaded_horde_model_name}")
        self._process_map.on_model_ram_clear(process_id=process_id)

    def get_next_n_models(self, n: int) -> list[str]:
        """Get the next n models that will be used in the job deque."""
        next_n_models: list[str] = []
        jobs_traversed = 0
        while len(next_n_models) < n:
            if jobs_traversed >= len(self._job_tracker.jobs_pending_inference):
                break

            model_name = self._job_tracker.jobs_pending_inference[jobs_traversed].model

            if model_name is None:
                raise ValueError(f"job_deque[{jobs_traversed}].model is None")

            if model_name not in next_n_models:
                next_n_models.append(model_name)

            jobs_traversed += 1

        return next_n_models

    def unload_models(self, *, under_pressure: bool = False, for_head_of_queue: bool = False) -> bool:
        """Unload one idle model from RAM that is no longer needed; return True if one was unloaded.

        ``under_pressure`` (set by the RAM budget when the next job's RAM cost does not fit available
        system memory) drops the RAM residency grace so an idle resident copy is reclaimed rather than
        held, the guard against resident-in-RAM weights forcing the OS to page.

        ``for_head_of_queue`` is the last-resort escalation when the head-of-queue job cannot be loaded
        and gentle reclaim freed nothing because every idle resident copy is another *queued* job's
        model: it additionally overrides the still-needed-by-a-pending-job guard so the head can be
        given room. It never evicts an in-progress (live) model.
        """
        bridge_data = self._runtime_config.bridge_data

        # An empty queue short-circuits the *normal* path (nothing to make room for), but not the pressure
        # path: under the RAM danger floor the held pops drain the queue, and the idle resident footprint
        # left behind is precisely what must be reclaimed to get the host back off its floor. Gating the
        # reclaim on queued work would leave that footprint pinned forever, so the pop hold never lifts.
        if len(self._job_tracker.jobs_pending_inference) == 0 and not under_pressure:
            return False

        if (
            self._max_concurrent_inference_processes == 1
            and len(bridge_data.image_models_to_load) == 1
            and not under_pressure
        ):
            return False

        wanted_models = self._compute_wanted_models()
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}

        eligible: list[HordeProcessInfo] = []
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if process_info.is_process_busy() or process_info.last_process_state == HordeProcessState.PRELOADED_MODEL:
                continue

            if process_info.loaded_horde_model_name is not None:
                if self._horde_model_map.is_model_loading(process_info.loaded_horde_model_name):
                    continue

                # The map entry can be expired out from under a still-set loaded_horde_model_name (the
                # stale-loading sweep, or a dead process's entries): a missing entry is not IN_USE, so a
                # raw [] index here would crash the control loop. Treat absence as "free to reclaim".
                model_info = self._horde_model_map.root.get(process_info.loaded_horde_model_name)
                if model_info is not None and model_info.horde_model_load_state == ModelLoadState.IN_USE:
                    continue

                # Live (in-progress) work is never evicted, even when making room for the head. Pending
                # (merely queued) models are protected too in the normal path, but the head-of-queue
                # escalation may reclaim one of them since the head has priority for room.
                if process_info.loaded_horde_model_name in in_progress_models:
                    continue
                if not for_head_of_queue:
                    pending_models = {
                        job.model for job in self._job_tracker.jobs_pending_inference if job.model is not None
                    }
                    if process_info.loaded_horde_model_name in pending_models:
                        continue

                if not for_head_of_queue and self._residency_protects_from_unload(
                    process_info.loaded_horde_model_name,
                    wanted_models,
                    vram=False,
                    under_pressure=under_pressure,
                ):
                    continue

                eligible.append(process_info)

        if not eligible:
            return False

        # Among the reclaimable idle residents, sacrifice the cheapest cache to rebuild: a light model's
        # checkpoint reloads from disk in a fraction of a card-dominating one's time, so evicting by size
        # tier (map order breaking ties) keeps the most expensive warm copy alive whenever any cheaper
        # candidate can free the RAM instead. A lone heavy resident is still evicted, so the tier
        # preference can never wedge the reclaim.
        victim = min(eligible, key=lambda p: self._model_size_tier(p.loaded_horde_model_name))
        self.unload_from_ram(victim.process_id)
        return True

    async def run_scheduling_cycle(self, stable_diffusion_reference: dict[str, ImageGenerationModelRecord]) -> None:
        """Run a single scheduling cycle: preload, start inference, unload.

        This absorbs the inline orchestration block from _process_control_loop.
        """
        self._pending_line_skip = None
        bridge_data = self._runtime_config.bridge_data

        self._refresh_model_demand()
        self.record_slot_duty(stable_diffusion_reference)

        # Resource governance is not driven here: the process manager runs run_governance_tick() every
        # control-loop iteration, so the danger-floor verdict and shed/restore response are already fresh
        # for this cycle regardless of whether any preload or dispatch happens.
        if not self.preload_models():
            keep_single_inference, single_inf_reason = self._process_map.keep_single_inference(
                stable_diffusion_model_reference=stable_diffusion_reference,
            )

            pending_and_active = len(self._job_tracker.jobs_pending_inference) + len(
                self._job_tracker.jobs_in_progress,
            )
            if keep_single_inference and pending_and_active > 1:
                if (time.time() - self._batch_wait_log_time > 10) and bridge_data.max_threads > 1:
                    logger.opt(ansi=True).info(
                        f"<fg #7b7d7d><i>Blocking further inference due to {single_inf_reason}.</i></>",
                    )
                    self._batch_wait_log_time = time.time()

            else:
                # Fill every free inference slot this cycle rather than one per ~0.5s control-loop
                # tick: when several jobs complete close together, dispatching them one tick apart
                # leaves the GPU underfed. start_inference() returns False once no more can start
                # (its own concurrency gate: jobs_in_progress >= max_concurrent, no free process,
                # or no eligible job, stops the loop), so this cannot over-subscribe.
                started_any = False
                while await self.start_inference():
                    started_any = True

                if not started_any:
                    # Nothing dispatched this cycle though the queue has work: if the head has been parked
                    # long enough to be a real stall (not a between-jobs gap), explain *why* it is not
                    # dispatching. Throttled, read-only; it never changes scheduling.
                    self._log_dispatch_stall_if_needed(stable_diffusion_reference)
                    self.unload_models()
