"""Schedules model preloading, inference start, and model unloading."""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass

import psutil
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger

from horde_worker_regen.consts import KNOWN_SLOW_WORKFLOWS, VRAM_HEAVY_MODELS
from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import WorkerState
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
from horde_worker_regen.process_management.jobs.job_models import LineSkip, NextJobAndProcess
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.process_management.models.model_sizing import ModelSizeTier, model_size_tier
from horde_worker_regen.process_management.resources.resource_budget import (
    CommittedReserveLedger,
    RamBudget,
    RamPressureVerdict,
    StreamForecast,
    VramBudget,
    WholeCardResidencyState,
    assess_ram_pressure,
    forecast_weight_streaming,
    is_model_locally_unservable_for,
    predict_job_weight_mb,
)
from horde_worker_regen.process_management.resources.run_metrics import ChurnKind
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
    VramGateResult,
    VramReclaimOutcome,
    WholeCardResidency,
    WholeCardResidencyMachine,
    WorkerProcessShedState,
    card_preload_order,
    compute_preload_disallowed_processes,
    decide_degrade_response,
    decide_process_reduction,
    decide_ram_reclaim_outcome,
    decide_shed_card_restore,
    decide_vram_reclaim_outcome,
    max_coresident_for_peak,
    preload_concurrency_blocked,
    select_head_room_process_id,
)
from horde_worker_regen.process_management.scheduling.model_affinity import affinity_active
from horde_worker_regen.process_management.scheduling.performance_model import PerformanceModel, signature_from_job
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyAccumulator, SlotDutyBucket
from horde_worker_regen.telemetry_spans import span_preload_model
from horde_worker_regen.utils.config_coercion import config_number
from horde_worker_regen.utils.job_utils import (
    get_single_job_magnitude as _get_single_job_effective_megapixelsteps,
)
from horde_worker_regen.utils.job_utils import line_skip_candidate_emps_limit


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

_HEAD_STARVATION_FORCE_ADMIT_SECONDS = 15.0
"""How long the head-of-queue job may be budget-deferred onto an otherwise-idle device before it is
force-admitted best-effort. Deliberately under the recovery supervisor's
``_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS`` (20s) so a self-inflicted budget-defer wedge resolves by
admitting one head onto an idle card, rather than tripping a save-our-ship soft reset that respawns
every pool and faults the whole backlog. Only runs while no live job holds the device, so it never
re-introduces the multi-process over-commit the budget guards against."""

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
"""How long after a heavy head is admitted best-effort (the over-budget exclusive path, taken when a
model streams even with the whole card to itself, e.g. an fp16 checkpoint on a small device) the recovery
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
    TERMINAL_ADMIT = enum.auto()
    """Teardown is structurally exhausted yet the activation-inclusive peak still overflows even sole
    residency; admit best-effort onto the cleared card under the over-budget step grace."""


class _PreloadJobOutcome(enum.Enum):
    """What one pending job's preload attempt means for the rest of this scheduling pass."""

    NEXT_JOB = enum.auto()
    """This job needs nothing (or was faulted); consider the next pending job."""
    STOP_PASS = enum.auto()
    """A gate deferred or consumed this cycle (RAM floor, no slot, serialization, budget); stop the pass."""
    PRELOAD_SENT = enum.auto()
    """A preload was issued for this job; the pass is done and reports success."""


def _preload_outcome_from_admission(decision: AdmissionDecision) -> _PreloadJobOutcome:
    """Map the public admission decision vocabulary onto the scheduler pass control enum."""
    match decision:
        case AdmissionDecision.ADMIT | AdmissionDecision.PRESTAGE | AdmissionDecision.TERMINAL_ADMIT:
            return _PreloadJobOutcome.PRELOAD_SENT
        case AdmissionDecision.NEXT_JOB | AdmissionDecision.QUARANTINED | AdmissionDecision.ALREADY_LOADED:
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
        """
        self._state = state
        self._process_map = process_map
        self._horde_model_map = horde_model_map
        self._job_tracker = job_tracker
        self._process_lifecycle = process_lifecycle
        self._runtime_config = runtime_config
        self._model_metadata = model_metadata
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
        # it the probe measurements via set_measured_*; the scheduler feeds it clean idle-residency readings
        # captured each tick by _maybe_capture_idle_context_residency.
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
        # When a heavy head was last admitted best-effort off the whole-card path (the over-budget exclusive
        # branch, for a model that streams even alone). Its load equally holds the queue, so this bounds a
        # wedge grace that the whole-card establishment grace does not cover. 0.0 when none is loading.
        self._heavy_head_admitted_at: float = 0.0
        # When an idle inference slot was last deliberately cycled to reclaim allocator-retained RAM
        # (_replace_stale_ram_unload_process). The respawn + the next head's preload leave the queue
        # briefly unservable through no fault of the pool, so this bounds a wedge grace covering that
        # deliberate window. 0.0 when no reclaim cycle is in flight. See _RAM_RECLAIM_CYCLE_GRACE_SECONDS.
        self._ram_reclaim_cycle_at: float = 0.0
        # Head-of-queue starvation backstop. Tracks the id of the job currently at the head of the queue
        # and when it first became budget-deferred onto an idle device, so a head that the budget gate
        # cannot fit (reclamation structurally exhausted) is force-admitted before the sustained-wedge
        # window trips the recovery supervisor. Reset when the head changes, a job dispatches, or a live
        # job takes the device. See _HEAD_STARVATION_FORCE_ADMIT_SECONDS.
        self._head_starvation_job_id: str | None = None
        self._head_starvation_since: float = 0.0

        # Dispatch-stall diagnostic throttle. When the queue has work but nothing dispatches, the scheduler
        # would otherwise return None silently; this records the last reason logged and when, so the
        # explanation is emitted at most once per interval (and immediately when the reason changes) rather
        # than every sub-second control-loop tick.
        self._dispatch_stall_last_reason: str | None = None
        self._dispatch_stall_log_time: float = 0.0

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

        Delegates to :func:`is_model_locally_unservable_for` so the scheduler's best-effort-admit gate and
        the popper's model selection apply one identical policy: a model held back here is neither
        best-effort-admitted nor popped, so the worker stops force-admitting and dropping a model the
        device genuinely cannot run. ``device_index`` scopes the streak to the card the admit targets on a
        multi-GPU host; None is the single-GPU / worker-wide reading.
        """
        return is_model_locally_unservable_for(
            self._runtime_config.bridge_data,
            self._job_tracker,
            model,
            device_index=device_index,
        )

    def _log_overbudget_admit(self, job: ImageGenerateJobPopResponse) -> None:
        """Log a best-effort over-budget admit with the residency/measurement picture (live diagnostics).

        Captures, in one greppable line, the model admitted against the budget, whether it runs
        exclusively, its prior over-budget fault streak, and the per-slot residency + device-wide free
        VRAM at admit time (the over-commit signature: e.g. another slot resident while this loads).
        """
        exclusive = self._job_tracker.is_admitted_exclusive(job)
        fault_count = self._job_tracker.get_model_overbudget_fault_count(job.model)
        logger.opt(ansi=True).warning(
            f"<fg #f0beff>VRAM budget cannot fit head-of-queue model {job.model} even after reclaiming all idle "
            f"VRAM/RAM, and no live job holds the device; admitting it best-effort "
            f"({'exclusive' if exclusive else 'shared'}, prior_overbudget_faults={fault_count}) rather than "
            f"wedging the queue. {self._process_map.residency_snapshot()}</>",
        )

    def _mark_overbudget_admit(self, job: ImageGenerateJobPopResponse, forecast: StreamForecast | None) -> None:
        """Tag ``job`` as an over-budget best-effort admit, opening the heavy-head load grace on first admit.

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

    def _maybe_capture_idle_context_residency(self) -> None:
        """Record the device-wide used VRAM when every inference process is idle with no model resident.

        That measurement is the true combined cost of all process contexts (the one-time CUDA runtime plus one
        context each), which the forecast needs to size ``free_after_model_evict`` without multiplying the
        one-time cost by the process count. Inspects the process map for the clean precondition (every live
        inference process up, idle, and holding no model) and feeds a confirmed reading to the overhead model,
        which keeps the relevant extremes. Cheap and side-effect-free beyond the cached figure, so it is safe
        to call every scheduling tick.
        """
        free_mb = self._process_map.get_free_vram_mb()
        total_mb = self._process_map.get_reported_total_vram_mb()
        if free_mb is None or total_mb is None:
            return
        process_count = 0
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED):
                continue
            process_count += 1
            # A clean baseline requires every live inference process up, idle, and holding no model: any model
            # resident (even one offloaded to RAM but still tracked) means the reading includes weight VRAM.
            if (
                process_info.last_process_state != HordeProcessState.WAITING_FOR_JOB
                or process_info.loaded_horde_model_name is not None
            ):
                return
        if process_count < 1:
            return
        used_mb = total_mb - free_mb
        if used_mb <= 0:
            return
        self._overhead.observe_idle_residency(used_mb=used_mb, idle_inference_process_count=process_count)

    def _maybe_invalidate_idle_context_floor(self) -> None:
        """Lower a latched effective idle floor once the device proves it was not a sustained reading.

        Complements :meth:`_maybe_capture_idle_context_residency`. The capture keeps the worst clean all-idle
        reading; a transient spike (taken before the allocator returned a just-unloaded model's cache) would
        otherwise pin the per-context marginal high for the whole session and route ordinary models into
        teardown/exclusive admits. Unlike the capture this does not require the clean precondition: a reading
        with resident models can only make the correction conservative, so any device-wide used reading below
        the latched floor (with at least as many inference contexts live) is unambiguous proof it was too high.
        Read-only beyond the cached figure, so it is safe every scheduling tick.
        """
        free_mb = self._process_map.get_free_vram_mb()
        total_mb = self._process_map.get_reported_total_vram_mb()
        if free_mb is None or total_mb is None:
            return
        used_mb = total_mb - free_mb
        if used_mb <= 0:
            return
        live_inference_processes = sum(
            1
            for process_info in self._process_map.values()
            if process_info.process_type == HordeProcessType.INFERENCE
            and process_info.last_process_state
            not in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED)
        )
        self._overhead.observe_device_residency(
            used_mb=used_mb,
            live_inference_process_count=live_inference_processes,
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
        terminal admit still gates on real free VRAM, rather than reserving the device on an unmeasured guess.
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
        # inference siblings cannot reclaim, so it is charged as an extra resident context here.
        num_post_process_contexts = self._process_map.num_post_process_processes(device_index=device_index)
        # Refresh the clean all-contexts idle baseline (a no-op once startup has passed) so the marginal
        # per-context cost reflects measurement rather than the one-time-cost-times-N over-count, then let a
        # later lower reading invalidate a latched floor that was a transient spike rather than sustained.
        self._maybe_capture_idle_context_residency()
        self._maybe_invalidate_idle_context_floor()
        # The EXTRA_LARGE tier (extra-large baselines plus the named VRAM-heavy checkpoints) is the single
        # source of truth for "wants the whole card and never shares". Feed it to the forecast so a baseline
        # whose conservative weight seed happens to fit co-resident still claims sole residency on intent,
        # rather than co-residing and thrashing as Z-Image did.
        wants_whole_card = self._model_size_tier(job.model) >= _ModelSizeTier.EXTRA_LARGE
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
        )

    def _residency_should_pause_safety(self, device_index: int | None) -> bool:
        """Whether a whole-card residency on this card should also move the single safety process off-GPU.

        Requires safety configured-and-on-GPU (:meth:`_whole_card_safety_off_gpu_enabled`) and that this is
        the card the one safety process is pinned to, i.e. the lowest-index driven card. A residency on a
        non-safety card never disturbs safety. The worker-wide key (``None``, single-GPU) always qualifies.
        """
        if not self._whole_card_safety_off_gpu_enabled():
            return False
        if device_index is None or not self._card_runtimes:
            return True
        return device_index == min(self._card_runtimes)

    def _has_safety_backlog(self) -> bool:
        """Return whether safety has work that should not be interrupted by residency churn."""
        return bool(self._job_tracker.jobs_pending_safety_check or self._job_tracker.jobs_being_safety_checked)

    def _pause_safety_for_residency_if_idle(self, device_index: int | None) -> bool:
        """Pause safety for whole-card residency only when no safety job is pending or active."""
        if not self._residency_should_pause_safety(device_index):
            return False
        if self._process_lifecycle.is_safety_gpu_paused:
            return False
        if self._has_safety_backlog():
            return False
        return self._process_lifecycle.pause_safety_on_gpu()

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

        if announce or after < current or safety_paused:
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
        for device_index, state in self._held_residencies():
            model = state.model
            if model in active_models or has_exclusive:
                # Still serving the residency; keep it (refresh the cooldown so it survives the lull between
                # back-to-back heavy jobs).
                state.cooldown_until = now + self._whole_card_cooldown_seconds()
                continue
            if time.time() < state.cooldown_until:
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
                else False
            )
            ceiling = self._residency_restore_ceiling(device_index)
            current = self._process_map.num_loaded_inference_processes(device_index=device_index)
            if current >= ceiling and not safety_restored:
                continue
            after = self._process_lifecycle.scale_inference_processes(ceiling, device_index=device_index)
            self._reconcile_worker_shed_to_pool()
            safety_note = " and restoring safety to the GPU" if safety_restored else ""
            logger.opt(ansi=True).info(
                f"<fg #7b7d7d>Whole-card residency for {model} complete; restoring inference processes "
                f"({current} -> {after} of {ceiling}){safety_note}.</>",
            )

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

    def _log_head_starvation_force_admit(self, job: ImageGenerateJobPopResponse) -> None:
        """Announce a head-of-queue force-admit, with the residency snapshot for the post-mortem."""
        logger.opt(ansi=True).warning(
            f"<fg #ff8c69>Head-of-queue {job.model} was budget-deferred on an idle device for "
            f"{self._head_starved_seconds(job):.0f}s (reclamation exhausted); force-admitting it best-effort "
            f"to break the wedge before the recovery supervisor soft-resets the pools and faults the backlog. "
            f"{self._process_map.residency_snapshot()}</>",
        )

    def _log_whole_card_terminal_admit(self, job: ImageGenerateJobPopResponse) -> None:
        """Announce a whole-card head admitted best-effort after its teardown was structurally exhausted."""
        logger.opt(ansi=True).warning(
            f"<fg #ff8c69>Whole-card head {job.model} reached its target sole residency but its activation "
            f"peak still overflows the card; admitting it best-effort to load onto the cleared device (it will "
            f"sample slowly under the over-budget step grace) rather than wedge the queue until save-our-ship. "
            f"{self._process_map.residency_snapshot()}</>",
        )

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
                case PausePops(until_time=until_time, pause_seconds=pause_seconds, reason=reason):
                    self._state.self_throttle_paused = True
                    self._state.self_throttle_paused_until = until_time
                    logger.opt(ansi=True).warning(
                        f"<fg #ff8c69>System RAM below the danger floor ({reason}); pausing job pops for "
                        f"{pause_seconds:.0f}s and shedding idle footprint so the host is not driven into "
                        "an OS OOM kill. In-flight jobs finish; pops resume once RAM recovers.</>",
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

    def _concurrent_overlap_allowed(
        self,
        candidate_job: ImageGenerateJobPopResponse,
        *,
        target_device_index: int | None = None,
    ) -> bool:
        """Whether ``candidate_job`` may start while other jobs are already sampling.

        The concurrency cap (``max_threads``) only counts in-flight jobs; it does not look at what those
        jobs are or how far along they are. That lets two heavy SDXL jobs (plus a speculatively-staged
        third) stack their weight loads and activation peaks on the card at once, thrashing a sampler
        into a step-timeout teardown. This gate adds the missing dimension: a new overlap is admitted
        only when the in-flight work can tolerate it.

        Rules, scaled by model size and measured headroom:
            * The first job (nothing in flight) always starts.
            * An extra-large candidate never joins a busy card, and an extra-large job in flight never
              shares it: the whole-card tier's contract is independent of how roomy the card is.
            * A batched candidate or a batched job in flight blocks overlap on a tight card; when the
              device's measured free VRAM absorbs the candidate's full predicted peak plus reserve
              (:meth:`_overlap_headroom_ample`), the batch instead imposes the strictest headway.
            * Otherwise the running job must have made size-appropriate headway: none for light+light,
              modest when one side is heavy, considerable for two heavy jobs. With ample measured
              headroom the heavy fractions relax to a small constant, since the over-subscription they
              guard against cannot occur when the newcomer's whole peak fits free VRAM now.

        A blocked job is not dropped; it keeps its queue position and dispatches once the in-flight
        job(s) progress or finish.

        Args:
            candidate_job: The job being considered for dispatch.
            target_device_index: On a multi-GPU host, the card this candidate would run on; the headway
                check then considers only jobs already sampling on that same card (jobs on other cards do
                not contend for its VRAM or sampler). ``None`` (and every single-GPU call) keeps the
                worker-wide comparison, byte-identical to before.
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

        # Resolved once per call (it reads the live VRAM measurement) and only when a rule needs it.
        headroom_ample: bool | None = None

        def ample() -> bool:
            nonlocal headroom_ample
            if headroom_ample is None:
                headroom_ample = self._overlap_headroom_ample(candidate_job, device_index=target_device_index)
            return headroom_ample

        candidate_batched = self._job_batch_amount(candidate_job) > 1
        if candidate_batched and not ample():
            return False

        for job in in_progress_jobs:
            running_tier = self._model_size_tier(job.model)
            if running_tier >= _ModelSizeTier.EXTRA_LARGE:
                return False

            if candidate_batched or self._job_batch_amount(job) > 1:
                # A batch multiplies the activation peak, so on a tight card it keeps the hard block;
                # with measured room for the newcomer's whole peak the batch is bounded instead by the
                # strictest headway (never the ample relaxation, which is sized for single jobs).
                if not ample():
                    return False
                required_headway = _OVERLAP_HEADWAY_BOTH_HEAVY
            else:
                required_headway = self._required_overlap_headway(running_tier, candidate_tier)
                if required_headway > 0.0 and ample():
                    required_headway = _OVERLAP_HEADWAY_AMPLE_VRAM

            if required_headway <= 0.0:
                continue
            if self._in_flight_progress_fraction(job) < required_headway:
                return False

        return True

    def _overlap_headroom_ample(
        self,
        candidate_job: ImageGenerateJobPopResponse,
        *,
        device_index: int | None = None,
    ) -> bool:
        """Whether the device's live free VRAM absorbs ``candidate_job``'s full predicted sampling peak.

        The overlap gate's strict headway fractions guard against a newcomer's weight load and
        activation peak over-subscribing a card that cannot host both samplers; this predicate is the
        measurement that decides whether that hazard exists at all. It reuses the VRAM budget's own
        verdict (predicted sampling peak + configured reserve against measured free, net of committed
        reserves), so the overlap gate and the admission gate price a job identically. The check is
        deliberately conservative in the candidate's favor: dispatch requires the candidate's model to
        be resident already, so its weights are on the card and the prediction double-counts them as
        margin. No measurement (cold start) or an inactive budget reads as tight, keeping the strict
        gate wherever the relaxation cannot be justified by data.
        """
        if candidate_job.model is None or not self._budget_active():
            return False
        free_vram_mb = self._measured_free_vram_mb(device_index=device_index)
        if free_vram_mb is None:
            return False
        baseline = self._model_metadata.get_baseline(candidate_job.model)
        verdict = self._vram_budget.check_job(
            candidate_job,
            baseline,
            free_vram_mb,
            committed_reserve_mb=self._committed_vram_reserve_mb(device_index=device_index),
        )
        return verdict.fits

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
        # Teardown is structurally exhausted (already at the target process count, safety settled) yet the
        # activation-inclusive peak still overflows even sole residency, so co-residency can never be reached.
        # The weights DO fit alone, so admit best-effort onto the now-cleared card under the over-budget step
        # grace instead of deferring every tick until the supervisor soft-resets.
        return _WholeCardDemandOutcome.TERMINAL_ADMIT

    def _apply_resource_verdicts(
        self,
        job: ImageGenerateJobPopResponse,
        available_process: HordeProcessInfo,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        forecast: StreamForecast,
        *,
        is_head_blocker: bool,
        target_device_index: int | None,
        no_live_resource_consumer: bool,
    ) -> bool:
        """Apply the VRAM then RAM budget verdicts for a preload, reclaiming or admitting as needed.

        Returns True to proceed with the preload (the resources fit, or a head was admitted best-effort
        after reclamation was exhausted), False to defer this cycle. When a resource does not fit, idle
        residents are reclaimed (overriding residency under pressure, escalating for the head of the queue);
        a head on an otherwise-idle device whose shortfall no reclaim can close is admitted best-effort and
        tagged over-budget rather than wedging the queue. The VRAM-not-fits path may instead reduce the live
        process count when the contexts (not the resident models) are the over-commit.
        """
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")

        vram_result = self._apply_vram_verdict(
            job,
            available_process,
            baseline,
            forecast,
            is_head_blocker=is_head_blocker,
            target_device_index=target_device_index,
            no_live_resource_consumer=no_live_resource_consumer,
        )
        if vram_result is VramGateResult.DEFER:
            return False
        if vram_result is VramGateResult.ADMIT_OVER_BUDGET:
            # The best-effort admit already reclaimed system RAM (a heavy head loads its checkpoint
            # through RAM first) and deliberately bypasses the marginal RAM verdict: the whole point of
            # the admit is that the estimates reject a head the device must nevertheless host.
            return True
        return self._apply_ram_verdict(
            job,
            baseline,
            is_head_blocker=is_head_blocker,
            no_live_resource_consumer=no_live_resource_consumer,
        )

    def _apply_vram_verdict(
        self,
        job: ImageGenerateJobPopResponse,
        available_process: HordeProcessInfo,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        forecast: StreamForecast,
        *,
        is_head_blocker: bool,
        target_device_index: int | None,
        no_live_resource_consumer: bool,
    ) -> VramGateResult:
        """Apply the VRAM budget verdict for a preload: reclaim, reduce contexts, or best-effort admit.

        When the predicted peak fits, answers ``FITS`` immediately (the caller proceeds to the RAM gate).
        Otherwise runs the reclaim attempts (gentle eviction, escalated for the head of the queue) and
        dispatches on
        [`decide_vram_reclaim_outcome`][horde_worker_regen.process_management.scheduling.governance.preload_admission.decide_vram_reclaim_outcome],
        which holds the escalation policy: defer while reclaim makes progress or a live job holds the
        device, hold a breaker-tripped model, reduce the live context count when contexts are the
        over-commit, or admit the head best-effort (tagged over-budget) rather than wedge the queue.
        """
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")

        vram_verdict = self._vram_budget.check_job(
            job,
            baseline,
            self._measured_free_vram_mb(device_index=target_device_index),
            committed_reserve_mb=self._committed_vram_reserve_mb(device_index=target_device_index),
        )
        if vram_verdict.fits:
            self._vram_budget_defer_notified = False
            return VramGateResult.FITS

        if not self._vram_budget_defer_notified:
            logger.opt(ansi=True).warning(
                f"<fg #f0beff>VRAM budget deferring preload of {job.model}: {vram_verdict.reason()}. "
                "Reclaiming idle VRAM.</>",
            )
            self._vram_budget_defer_notified = True
        freed = self.unload_models_from_vram(
            available_process,
            under_pressure=True,
            device_index=target_device_index,
        )
        if not freed and is_head_blocker:
            # Gentle reclaim found nothing to free because every idle resident copy is another queued
            # job's model. The head of the queue must still make progress, so escalate and reclaim one of
            # them to give the head room.
            freed = self.unload_models_from_vram(
                available_process,
                under_pressure=True,
                for_head_of_queue=True,
                device_index=target_device_index,
            )

        # Before evicting every resident model and admitting the head exclusively, check whether the live
        # process *contexts* are the over-commit rather than the resident models. The weight-based teardown
        # gates leave a moderate head co-resident because its weights fit after a model eviction, but its
        # activation peak does not fit while this many contexts are live (the threads>1 regime, where each
        # extra context retains VRAM the allocator never returns). When more inference contexts are live
        # than the head's weights-plus-reserve can co-reside with (``max_resident_processes``, sized from
        # the measured per-context cost), reducing the process count is the structural remedy: it returns a
        # context's retained VRAM so the head and a sibling model co-reside and pipeline. The depth is
        # sized from the verdict's own rejected peak so the reduction fires exactly when admission would
        # reject; a demand resting on untrusted (unmeasured-fallback) overhead figures is declined.
        max_resident = None
        if vram_verdict.predicted_mb is not None:
            max_resident = self._max_coresident_for_peak_mb(
                vram_verdict.predicted_mb,
                vram_verdict.reserve_mb,
                device_index=target_device_index,
            )
        context_reduction_demanded = (
            self._whole_card_residency_enabled()
            and is_head_blocker
            and max_resident is not None
            and self._process_map.num_loaded_inference_processes(device_index=target_device_index) > max_resident
        )

        outcome = decide_vram_reclaim_outcome(
            freed=freed,
            is_head_blocker=is_head_blocker,
            no_live_resource_consumer=no_live_resource_consumer,
            model_unservable=self._is_model_locally_unservable(job.model, device_index=target_device_index),
            context_reduction_demanded=context_reduction_demanded,
            whole_card_warranted=self._whole_card_warranted(forecast),
        )

        if outcome is VramReclaimOutcome.DEFER:
            # Reclamation made progress, the job is not the head blocker, or a live job holds the device:
            # wait for the freed room (or the live job's completion) rather than over-committing.
            return VramGateResult.DEFER

        if outcome is VramReclaimOutcome.HOLD_UNSERVABLE:
            # Circuit-breaker: a model the device genuinely cannot run faults every over-budget attempt no
            # matter how it is isolated. Once its consecutive-fault streak crosses the configured threshold
            # it is held back (not admitted here, not popped in the popper) for a cooldown, so the worker
            # stops dropping jobs faster than the horde server tolerates and is never forced into
            # maintenance.
            if not self._unservable_admit_notified.get(job.model, False):
                logger.opt(ansi=True).warning(
                    f"<fg #ff8c69>Model {job.model} keeps faulting over the VRAM budget; held "
                    f"back as locally unservable and not admitted. "
                    f"{self._process_map.residency_snapshot()}</>",
                )
                self._unservable_admit_notified[job.model] = True
            return VramGateResult.DEFER
        self._unservable_admit_notified.pop(job.model, None)

        if outcome is VramReclaimOutcome.REDUCE_CONTEXTS:
            first_time = not self._job_tracker.is_admitted_exclusive(job)
            self._job_tracker.mark_admitted_exclusive(job)
            self._establish_whole_card_residency(
                job,
                forecast,
                announce=first_time,
                target_override=max_resident,
                device_index=target_device_index,
            )
            self.unload_models_from_vram(
                available_process,
                under_pressure=True,
                for_head_of_queue=True,
                device_index=target_device_index,
            )
            return VramGateResult.DEFER

        if outcome is VramReclaimOutcome.ADMIT_DECLINING_REDUCTION:
            self._log_whole_card_declined(job, forecast)

        # Best-effort admit: reclamation is exhausted (the predicted peak + reserve exceeds achievable free
        # VRAM even with every idle resident copy evicted) and nothing live holds the device. The burden
        # estimate is a deliberately conservative single-resident-peak figure, but a large combined
        # checkpoint is streamed through VRAM component-by-component by the backend, so its true peak is
        # well under the summed estimate. Give the head the device rather than deferring it forever (which
        # would wedge the queue and fault the head anyway), after reclaiming system RAM from idle residents
        # when the host actually lacks the room (a heavy head loads its checkpoint through RAM first). Tag
        # it so a crash/hang of its over-committed slot is classified as a resource failure (earning the
        # bounded, isolated retry) instead of a plain re-dispatch.
        self._reclaim_ram_for_overbudget_admit(job, baseline)
        self._mark_overbudget_admit(job, forecast)
        self._log_overbudget_admit(job)
        return VramGateResult.ADMIT_OVER_BUDGET

    def _reclaim_ram_for_overbudget_admit(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    ) -> None:
        """Reclaim idle system RAM ahead of a best-effort over-budget load, only when the host is short.

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
        """Return whether ``job`` may be admitted for preload under the active VRAM/RAM budget.

        True means proceed to send the preload; False means defer this cycle. Performs the residency,
        reclamation, and best-effort-admit side effects the verdict requires. The decision is: forecast
        weight streaming, resolve a whole-card residency demand (which may pre-stage, defer, or terminally
        admit the head), apply the head-of-queue starvation backstop, then fall to the ordinary VRAM/RAM
        verdict. Every decision below is scoped to the card this preload would land on (None keeps the
        worker-wide reading on a single-GPU host, so the path is byte-identical there).
        """
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")

        baseline = self._model_metadata.get_baseline(job.model)
        target_device_index = available_process.device_index if self._multi_gpu_routing_active else None
        # A single model loaded onto an otherwise-idle GPU cannot reintroduce the multi-process over-commit
        # the budget guards against; the over-commit case is several *concurrent* resident models. So when no
        # job is in-flight (holding this card), a starved head may be admitted best-effort rather than
        # deferred forever.
        if target_device_index is None:
            no_live_resource_consumer = len(self._job_tracker.jobs_in_progress) == 0
        else:
            no_live_resource_consumer = len(self._jobs_in_progress_on_card(target_device_index)) == 0

        forecast = self._forecast_streaming(job, baseline, device_index=target_device_index)
        # Trace the forecast for every budget-gated load so the logs show the residency dynamics, not just the
        # action taken. Unchanged observations are coalesced by _log_stream_forecast; decision or headroom
        # changes still log immediately.
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
        if whole_card is _WholeCardDemandOutcome.TERMINAL_ADMIT:
            self._log_whole_card_terminal_admit(job)
            self._mark_overbudget_admit(job, forecast)
            return True

        # Head-of-queue starvation backstop (see _HEAD_STARVATION_FORCE_ADMIT_SECONDS): once the head has been
        # budget-deferred on an otherwise-idle device past the wedge horizon, stop deferring and admit it
        # best-effort. Reclamation is structurally exhausted by then, so continuing to defer only wedges the
        # queue until the recovery supervisor soft-resets every pool and faults the whole backlog: strictly
        # worse than loading one head onto an idle card. This rescues a plain over-budget head that the
        # verdicts keep rejecting (e.g. a head failing the RAM budget against allocator-stranded idle RAM).
        force_admit_starved_head = (
            is_head_blocker and self._head_starved_seconds(job) >= _HEAD_STARVATION_FORCE_ADMIT_SECONDS
        )
        if force_admit_starved_head:
            self._log_head_starvation_force_admit(job)
            self._mark_overbudget_admit(job, forecast)
            return True

        return self._apply_resource_verdicts(
            job,
            available_process,
            baseline,
            forecast,
            is_head_blocker=is_head_blocker,
            target_device_index=target_device_index,
            no_live_resource_consumer=no_live_resource_consumer,
        )

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
        self._converge_whole_card_residency()
        self._expire_stale_model_map_entries()

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

        for job in self._job_tracker.jobs_pending_inference:
            outcome = self._attempt_preload_for_job(job, head_job=head_job, loaded_models=loaded_models)
            if outcome is _PreloadJobOutcome.NEXT_JOB:
                continue
            return outcome is _PreloadJobOutcome.PRELOAD_SENT

        return False

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
        floor, the exclusive-job hold, target selection (with the starved-head fallback), the
        cycle-on-model-change replacement, the per-device load serialization gate, and the VRAM/RAM
        budget admission. The returned :class:`_PreloadJobOutcome` tells the pass whether to consider the
        next pending job, stop for this cycle, or record that a preload was issued.
        """
        bridge_data = self._runtime_config.bridge_data
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")

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
        # OOM kill, not progress. This gates every admit path (best-effort, head-starvation force-admit,
        # and whole-card-terminal all sit below this point), independent of the marginal RAM budget, which
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

    def _resident_process_for_job(self, job: ImageGenerateJobPopResponse) -> HordeProcessInfo | None:
        """The resident process to dispatch ``job`` to, honoring per-card eligibility on a multi-GPU host.

        Single-GPU: identical to :meth:`ProcessMap.get_process_by_horde_model_name` (the first resident
        process), so the dispatch path is byte-identical. Multi-GPU: restrict to processes pinned to cards
        eligible for this job, then apply the sticky-then-least-loaded policy. Returns None when the model is
        resident only on cards that cannot serve this particular job, or is not resident anywhere.
        """
        if job.model is None:
            return None
        if not self._multi_gpu_routing_active:
            return self._process_map.get_process_by_horde_model_name(job.model)
        allowed = self._eligible_card_indices(job)
        candidates = self._process_map.get_processes_by_horde_model_name(job.model, allowed_cards=allowed)
        if not candidates:
            return None
        return self._pick_best_resident_process(candidates)

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
    ) -> NextJobAndProcess | None:
        """Select a small, ready job that may bypass ``displaced_job`` while its slot is non-sampling.

        Scans the pending jobs for the first resident on an idle process that holds a *different* model
        than the blocked head, carries no LoRAs, is not a degraded retry, and is within the
        per-performance-mode size limit. Returns a :class:`NextJobAndProcess` carrying the
        :class:`LineSkip` record, or None when nothing qualifies. Rejections are logged (rate-limited).
        """
        for candidate_small_job in next_n_jobs:
            candidate_id = str(candidate_small_job.id_)[:8]
            job_has_loras = (
                candidate_small_job.payload.loras is not None and len(candidate_small_job.payload.loras) > 0
            )
            if candidate_small_job.model is None:
                self._log_line_skip_rejection(candidate_id, "missing_model", "rejected: missing model.")
                continue
            if candidate_small_job.model == displaced_job.model:
                self._log_line_skip_rejection(
                    candidate_id,
                    "same_model",
                    f"rejected: same model as blocked job {str(displaced_job.id_)[:8]}.",
                )
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

            return None

        if process_with_model is None:
            if next_job.model is None:
                raise ValueError(f"next_job.model is None ({next_job})")

            # The head's model is not resident. If it is forecast to load (a preload is already on the
            # way), let a later already-resident job bypass it so the GPU is not idle while the head
            # loads; this reduces churn versus evicting to run the head right now. If it is NOT forecast
            # to load, do not bypass: fall through so the head is the one that gets a process (and the
            # budget gate makes room for it), rather than being starved behind perpetual bypassers.
            if self._is_model_forecast_to_load(next_job.model):
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
        device_index: int | None,
    ) -> bool:
        """Whether ``dispatched_job``'s model should stay resident in VRAM after it runs.

        hordelib evicts the model from VRAM after every job so sibling GPU instances never collectively
        over-commit; that eviction forces a RAM->VRAM reload on the next job, which is the dominant
        non-sampling cost on small jobs. Retention skips that reload, but only earns its keep when the
        reuse is imminent and the headroom is real, so both gates must hold:

        - **same model queued next**: another pending-inference job (not this one, not already in flight)
          reuses this model, so the retained weights are actually consumed rather than idly pinned.
        - **budget fits**: the VRAM budget confirms the card could still admit this job from scratch.
          The measured free figure is taken *while the job's weights occupy the card*, so they are
          credited back before the check: retention holds a state the admission already approved, and
          demanding the footprint fit inside the remaining free VRAM as well would double-charge the
          weights and refuse retention on precisely the contended cards where the reload skip pays.
          A missing budget or unmeasured VRAM yields False: retention is granted on evidence, never
          assumed.

        Even when granted, hordelib's force-load overflow guard remains the hard backstop, and the
        worker's under-pressure eviction can still reclaim the retained model, so a wrong call degrades
        to a reload rather than an OOM.
        """
        model = dispatched_job.model
        if model is None:
            return False
        same_model_pending = any(
            job.model == model
            for job in self._job_tracker.jobs_pending_inference
            if job is not dispatched_job and job not in self._job_tracker.jobs_in_progress
        )
        if not same_model_pending:
            return False
        if not self._budget_active():
            return False
        free_vram_mb = self._measured_free_vram_mb(device_index=device_index)
        if free_vram_mb is None:
            return False
        baseline = self._model_metadata.get_baseline(model)
        resident_weights_mb = predict_job_weight_mb(dispatched_job, baseline) or 0.0
        verdict = self._vram_budget.check_job(
            dispatched_job,
            baseline,
            free_vram_mb + resident_weights_mb,
            committed_reserve_mb=self._committed_vram_reserve_mb(device_index=device_index),
        )
        return verdict.fits

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
        """
        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

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

        else:
            logger.error(
                f"Failed to start inference for job {next_job.id_} on process {process_with_model.process_id}",
            )
            await self._job_tracker.handle_job_fault(
                faulted_job=next_job,
                process_info=process_with_model,
                process_timeout=bridge_data.process_timeout,
            )

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
        model: it additionally overrides the next-up guard so the head can be given room. It never
        evicts an in-progress (live) model.

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
        next_model = None
        if len(next_n_models) > 0:
            next_model = next_n_models.pop()

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

                if process_info.loaded_horde_model_name == next_model and not for_head_of_queue:
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

    def _is_heavy_model_and_workflow(
        self,
        job: ImageGenerateJobPopResponse,
        stable_diffusion_reference: dict[str, ImageGenerationModelRecord],
    ) -> bool:
        """Return whether the job's model and workflow are heavy enough to serialise behind in-flight work.

        True for an SDXL model running a known-slow workflow, or any model in ``VRAM_HEAVY_MODELS``. Used to
        hold a heavy batch head back while a thread is already busy, so stacked weight loads and activation
        peaks do not thrash a sampler into a watchdog teardown.
        """
        model = job.model
        if model is None:
            return False
        next_model_baseline = stable_diffusion_reference.get(model)
        next_workflow = job.payload.workflow
        heavy = (
            next_model_baseline is not None
            and next_model_baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
            and next_workflow in KNOWN_SLOW_WORKFLOWS
        )
        if model in VRAM_HEAVY_MODELS:
            heavy = True
        return heavy

    async def run_scheduling_cycle(self, stable_diffusion_reference: dict[str, ImageGenerationModelRecord]) -> None:
        """Run a single scheduling cycle: preload, detect heavy model/batch, start inference, unload.

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
            next_job_and_process = await self.get_next_job_and_process(information_only=True)

            next_job_heavy_model_and_workflow = next_job_and_process is not None and self._is_heavy_model_and_workflow(
                next_job_and_process.next_job, stable_diffusion_reference
            )

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

            elif (
                next_job_and_process is not None
                and (next_job_and_process.next_job.payload.n_iter > 1 or next_job_heavy_model_and_workflow)
                and (
                    self._process_map.num_busy_with_inference() > 0
                    or self._process_map.num_busy_with_post_processing() > 0
                )
            ):
                if time.time() - self._batch_wait_log_time > 10:
                    logger.opt(ansi=True).info(
                        "<fg #7b7d7d>"
                        f"<i>Blocking starting batch job {next_job_and_process.next_job.id_} "
                        "because a thread is already busy with a heavy model/workflow or batch job"
                        ".</i>"
                        "</>",
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
