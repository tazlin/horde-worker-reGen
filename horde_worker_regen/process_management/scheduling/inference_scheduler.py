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
    predict_job_post_processing_vram_mb,
    predict_job_weight_mb,
)
from horde_worker_regen.process_management.resources.run_metrics import ChurnKind
from horde_worker_regen.process_management.scheduling.context_overhead_model import ContextOverheadModel
from horde_worker_regen.process_management.scheduling.model_affinity import (
    affinity_active,
    compute_protected_processes,
)
from horde_worker_regen.process_management.scheduling.performance_model import PerformanceModel, signature_from_job
from horde_worker_regen.telemetry_spans import span_preload_model
from horde_worker_regen.utils.config_coercion import config_number
from horde_worker_regen.utils.job_utils import (
    get_single_job_magnitude as _get_single_job_effective_megapixelsteps,
)


class PostProcessingReclaimAction(enum.Enum):
    """What the scheduler should do to host a running job's imminent post-processing-phase VRAM peak.

    A job's upscaler/face-fixer peak (``predict_job_post_processing_vram_mb``) lands *after* sampling, on
    a card whose process contexts and any warm sibling models are already committed. The planner sizes that
    peak against the measured headroom and the room the job's own (idle-during-upscale) weights would free,
    and picks the cheapest action that fits it:

    1. :attr:`DELEGATE_IN_PROCESS`: the peak fits once the job's own weights are freed, which ComfyUI's
       per-process ``free_memory`` already does in-child, so the orchestrator need do nothing.
    2. :attr:`EVICT_SIBLING_MODEL`: own-weights room is not enough, so evict a *different* model resident
       on an idle sibling process (the running model may still be demanded by the queue, so it is kept).
    3. :attr:`REDUCE_CONTEXT`: nothing idle holds an evictable model, so stop a sibling process to reclaim
       its context. Expected only on the smallest cards or one over-committed by out-of-worker allocation.
    4. :attr:`DEFER`: nothing idle is reclaimable right now, but the peak fits the card drained to this
       job's process alone and a sibling is mid-inference whose completion will free the room, so hold the
       dispatch (the job keeps its head-of-queue position) until that room appears rather than faulting it.

    :attr:`NONE` means the peak fits as-is (or the job does no post-processing). :attr:`INSUFFICIENT` means
    no reclaim the orchestrator can perform and no in-flight sibling will free room hosts the peak (e.g. a
    single-process worker on a tiny card, or a peak that overflows even the card alone), so the job must fault
    gracefully rather than thrash; that fault feeds the post-processing circuit breaker.
    """

    NONE = enum.auto()
    """The post-processing peak fits the measured headroom as-is; no reclaim required."""
    DELEGATE_IN_PROCESS = enum.auto()
    """Freeing the job's own (now idle) weights suffices; ComfyUI does this in-child, no orchestrator action."""
    EVICT_SIBLING_MODEL = enum.auto()
    """Evict a different model resident on an idle sibling process (cross-process room only the parent frees)."""
    REDUCE_CONTEXT = enum.auto()
    """Stop a sibling process to reclaim its context when nothing idle is evictable (the rare last rung)."""
    DEFER = enum.auto()
    """Hold the dispatch (keep the head's position) until an in-flight sibling frees room the peak fits into."""
    INSUFFICIENT = enum.auto()
    """No orchestrator-reclaimable room hosts the peak; fault gracefully and feed the circuit breaker."""


@dataclass(frozen=True)
class PostProcessingReclaimPlan:
    """Represents the scheduler's decision for hosting a job's imminent post-processing VRAM peak.

    Pairs the chosen :class:`PostProcessingReclaimAction` with the sibling process it targets (for the
    eviction/teardown rungs) and the measured shortfall that drove the decision, so the enactment site can
    act without re-deriving the fit math and the logs can report why room was reclaimed.
    """

    action: PostProcessingReclaimAction
    target_process_id: int | None = None
    shortfall_mb: float = 0.0


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

_RAM_PRESSURE_PAUSE_SECONDS = 30.0
"""How long the worker pauses job pops once system RAM crosses its danger floor.

A short, self-expiring pop-pause (it auto-resumes via the manager's self-throttle cooldown) so intake
stops adding memory pressure while idle footprint is shed and the host recovers, without wedging a worker
whose RAM frees up moments later. Re-armed each scheduling pass that still reads under the floor."""

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


@dataclass
class _WholeCardResidency:
    """Mutable whole-card exclusive-residency state for one card (the worker, on a single-GPU host).

    A heavy model can claim a whole card to itself by stopping that card's idle sibling inference contexts
    (a context's VRAM is only reclaimed when its process exits) and, on the card the safety process sits
    on, moving safety off-GPU. The scheduler keys one of these per device index so two heavy models on
    different cards each hold their own residency independently. A single-GPU worker keeps exactly one
    instance under the ``None`` key, so its behaviour is identical to the pre-multi-GPU scalar fields.
    """

    model: str | None = None
    """The model holding (or being given) sole residency on this card; None when no residency is held."""
    forecast: StreamForecast | None = None
    """The streaming forecast that established this residency, cached for the status snapshot's hard numbers."""
    established_at: float = 0.0
    """When this residency was first established (stop siblings, cycle safety, load weights); 0.0 when none.

    The establishment intentionally holds the queue, which the recovery supervisor must not mistake for a
    structural wedge until the establish grace elapses."""
    cooldown_until: float = 0.0
    """Wall-clock time until which this residency is held even after its heavy job drains, so a burst of
    heavy jobs reuses one residency instead of each churning a teardown/restore + safety cycle."""
    restore_at: float = 0.0
    """When this residency was last restored (siblings respawned, safety cycled back on-GPU); 0.0 when none.
    The restore churn also briefly makes the queue unservable, so the wedge grace must cover it too."""


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
        # Whole-card exclusive-residency state, keyed by the device index a residency is held on. A heavy
        # model claims a card by stopping that card's idle sibling contexts (and cycling safety off-GPU on the
        # safety card); keying per card lets two heavy models on different cards each hold their own residency.
        # A single-GPU worker uses exactly one entry under the None key, identical to the prior scalar fields.
        # See _WholeCardResidency for the per-card fields and _residency_state for the accessor.
        self._whole_card_residencies: dict[int | None, _WholeCardResidency] = {}
        # Device indices whose idle inference contexts the RAM-pressure footprint reduction shed below their
        # planned per-card count. The reduction keeps one context per driven card so no GPU is stranded, and
        # records each card it shrank here so the recovery path can grow it back once the host clears the
        # danger floor. Multi-GPU only; the worker-wide (single-GPU) reduction does not populate it.
        self._ram_pressure_shed_cards: set[int] = set()
        # Inference process ids marked to DRAIN because their resident RAM crossed the per-process ceiling while
        # the host was under its danger floor. A draining process is fed no new dispatch/preload so its in-flight
        # job can finish, after which the governor recycles it to return its allocator-retained pages. Cleared
        # when a process falls back under the ceiling (or is recycled). Distinct from _ram_pressure_shed_cards,
        # which tracks *idle* contexts shed by count; this tracks a *specific busy* process being wound down.
        self._processes_draining_for_ram: set[int] = set()
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

        # The head whose dispatch is currently held for post-processing-peak headroom (job id, shortfall MB),
        # or None. Set when a dispatch defers (the peak overflows the contended card now but fits it alone and
        # an in-flight sibling will free room); read by the dispatch-stall diagnostic so a held head reads as
        # an explained wait. Cleared the moment any job dispatches.
        self._post_processing_dispatch_defer: tuple[str | None, float] | None = None

    def set_churn_observer(self, observer: Callable[[ChurnKind], None]) -> None:
        """Register the sink for between-jobs reload/respawn events (see :data:`ChurnKind`)."""
        self._churn_observer = observer

    def _record_churn(self, kind: ChurnKind) -> None:
        """Report one churn event to the observer if one is registered (no-op otherwise)."""
        if self._churn_observer is not None:
            self._churn_observer(kind)

    @property
    def _max_concurrent_inference_processes(self) -> int:
        """The live concurrent-inference cap (effective ``max_threads``), bounded by the ceiling."""
        return self._runtime_config.effective_max_threads

    @property
    def post_process_job_overlap_allowed(self) -> bool:
        """Return true if post processing jobs are allowed to overlap."""
        bd = self._runtime_config.bridge_data
        return (bd.moderate_performance_mode or bd.high_performance_mode) and bd.post_process_job_overlap

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
        exclusive = self._runtime_config.bridge_data.overbudget_exclusive_mode
        fault_count = self._job_tracker.get_model_overbudget_fault_count(job.model)
        logger.opt(ansi=True).warning(
            f"<fg #f0beff>VRAM budget cannot fit head-of-queue model {job.model} even after reclaiming all idle "
            f"VRAM/RAM, and no live job holds the device; admitting it best-effort "
            f"({'exclusive' if exclusive else 'shared'}, prior_overbudget_faults={fault_count}) rather than "
            f"wedging the queue. {self._process_map.residency_snapshot()}</>",
        )

    def _mark_overbudget_admit(self, job: ImageGenerateJobPopResponse) -> None:
        """Tag ``job`` as an over-budget best-effort admit, opening the heavy-head load grace on first admit.

        Records the load-grace start the first time the job is admitted (so its multi-gigabyte load is not
        mistaken for a structural wedge; see :meth:`heavy_head_load_grace_active`). When over-budget
        exclusive mode is configured, also marks it exclusive so the scheduler suppresses concurrent
        pre-staging and dispatch for its duration, leaving the device un-contended while it completes.
        """
        if not self._job_tracker.is_admitted_over_budget(job):
            self._heavy_head_admitted_at = time.time()
        self._job_tracker.mark_admitted_over_budget(job)
        if self._runtime_config.bridge_data.overbudget_exclusive_mode:
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

    def _residency_state(self, device_index: int | None) -> _WholeCardResidency:
        """Return the (lazily-created) whole-card residency state for ``device_index``.

        ``None`` is the single-GPU / worker-wide key, so a single-GPU host keeps exactly one residency state
        and behaves as the pre-multi-GPU scalar fields did.
        """
        state = self._whole_card_residencies.get(device_index)
        if state is None:
            state = _WholeCardResidency()
            self._whole_card_residencies[device_index] = state
        return state

    def _held_residencies(self) -> list[tuple[int | None, _WholeCardResidency]]:
        """Return ``(device_index, state)`` for every card currently holding a whole-card residency.

        A residency is "held" while its model is set. Used by the per-cycle convergence/restore passes and the
        supervisor-facing grace checks, which must consider every card's residency, not just one.
        """
        return [(index, state) for index, state in self._whole_card_residencies.items() if state.model is not None]

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
        total_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        per_process = self._per_process_overhead_mb()
        if total_mb is None or per_process <= 0:
            return None
        marginal = self._marginal_process_overhead_mb() or per_process
        if marginal <= 0:
            return None
        budget = total_mb - peak_mb - reserve_mb
        if budget <= per_process:
            return 1
        return max(1, 1 + int((budget - per_process) // marginal))

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
            num_extra_resident_contexts=num_safety_contexts,
            post_processing_reserve_mb=self._committed_vram_reserve_mb(device_index=device_index),
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
        state = self._residency_state(device_index)
        if announce or state.established_at == 0.0:
            # Mark the establishment start (first admit of this heavy job, or a fresh residency) so the
            # recovery supervisor's grace window is measured from when the intentional hold began.
            state.established_at = time.time()
        state.model = job.model
        state.forecast = forecast
        state.cooldown_until = time.time() + self._whole_card_cooldown_seconds()

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
        state = self._residency_state(device_index)
        if announce or state.established_at == 0.0:
            state.established_at = time.time()
        state.model = job.model
        state.forecast = forecast
        state.cooldown_until = time.time() + self._whole_card_cooldown_seconds()
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
            target = (forecast.max_resident_processes() or 1) if forecast is not None else 1
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

    def _residency_holder_for_model(self, model: str | None) -> tuple[bool, int | None]:
        """Return ``(found, device_index)`` for the card whose held whole-card residency is for ``model``.

        ``found`` distinguishes a genuine hit on the ``None`` (single-GPU / worker-wide) key from a miss, since
        ``None`` is itself a valid residency key.
        """
        if model is None:
            return (False, None)
        for device_index, state in self._whole_card_residencies.items():
            if state.model == model:
                return (True, device_index)
        return (False, None)

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
        target = forecast.max_resident_processes() or 1
        if self._process_map.num_loaded_inference_processes(device_index=device_index) > target:
            return False
        if self._residency_should_pause_safety(device_index) and not self._process_lifecycle.is_safety_gpu_paused:
            return False
        if self._whole_card_weights_fit_live(forecast, device_index=device_index):
            return True
        return forecast.fits_alone and self._whole_card_drain_backstop_elapsed(device_index)

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
        state = self._whole_card_residencies.get(device_index)
        if state is None or state.established_at == 0.0:
            return False
        return (time.time() - state.established_at) >= _WHOLE_CARD_DRAIN_SETTLE_SECONDS

    def is_whole_card_residency_active(self) -> bool:
        """Whether any card currently holds a whole-card residency lease (its cooldown still running).

        Mirrors the ``active`` field of :meth:`whole_card_residency_state` but without building the full
        snapshot, so the job popper's large-model re-entry cooldown can cheaply ask "is the lease up?" every
        pop cycle: the lease is up exactly when this returns False (no card holds a residency model).
        """
        return any(state.model is not None for state in self._whole_card_residencies.values())

    def whole_card_residency_grace_active(self) -> bool:
        """Whether a whole-card residency is establishing, so the held queue is intentional (not a wedge).

        While true, the recovery supervisor must not treat the deliberately-deferred heavy head (waiting
        for idle siblings to stop, the safety process to cycle off-GPU, and ~11GB of weights to load) as a
        structural queue wedge and soft-reset the pools mid-setup. Bounded by
        ``_WHOLE_CARD_ESTABLISH_GRACE_SECONDS`` so a residency that genuinely never loads still trips the
        supervisor. Public: read by the process manager's wedge assessment.
        """
        now = time.time()
        for state in self._whole_card_residencies.values():
            establishing = (
                state.model is not None
                and state.established_at != 0.0
                and (now - state.established_at) < _WHOLE_CARD_ESTABLISH_GRACE_SECONDS
            )
            restoring = state.restore_at != 0.0 and (now - state.restore_at) < _WHOLE_CARD_RESTORE_GRACE_SECONDS
            if establishing or restoring:
                return True
        return False

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
        state = self._whole_card_residencies.get(device_index)
        if state is None or state.model is None:
            return None, ""
        establishing = (
            state.established_at != 0.0 and (time.time() - state.established_at) < _WHOLE_CARD_ESTABLISH_GRACE_SECONDS
        )
        return state.model, ("establishing" if establishing else "holding")

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
            safety_note = " and restoring safety to the GPU" if safety_restored else ""
            logger.opt(ansi=True).info(
                f"<fg #7b7d7d>Whole-card residency for {model} complete; restoring inference processes "
                f"({current} -> {after} of {ceiling}){safety_note}.</>",
            )

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
        """Return why the head-of-queue job is not being dispatched (read-only; never raises into the loop).

        The scheduler returns ``None`` from :meth:`get_next_job_and_process` at several points without saying
        why, so a stuck queue with idle processes leaves no record of which gate parked the head. This
        re-derives that reason for the diagnostic, with the most detail for the genuinely suspicious case --
        the head's model is resident on an *idle* process yet nothing dispatches, since that is the
        scheduler-bug-shaped stall that is otherwise invisible.
        """
        process = self._resident_process_for_job(head)
        if process is None:
            if head.model is not None and self._horde_model_map.is_model_loading(head.model):
                return "its model is loading (a preload is in progress)"
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
                return (
                    f"its model is not resident because a whole-card residency is held for non-head model "
                    f"{nonhead_residency_model!r}: the card is reserved for that model and its siblings were "
                    f"torn down, so this head cannot load until that residency restores"
                )
            return (
                "its model is not resident and no preload has been admitted "
                "(usually a VRAM/RAM budget defer; see the budget lines above)"
            )
        if not process.can_accept_job():
            return (
                f"its model is resident on process {process.process_id}, but that process is busy "
                f"({process.last_process_state.name})"
            )

        # Resident on an idle process: the interesting case. Name the gate that is holding dispatch.
        keep_single, single_reason = self._process_map.keep_single_inference(
            stable_diffusion_model_reference=stable_diffusion_reference,
            post_process_job_overlap=self._runtime_config.bridge_data.post_process_job_overlap,
        )
        pending_and_active = len(self._job_tracker.jobs_pending_inference) + len(self._job_tracker.jobs_in_progress)
        if keep_single and pending_and_active > 1:
            return (
                f"its model is resident and idle on process {process.process_id}, but dispatch is held by "
                f"keep-single-inference ({single_reason})"
            )
        in_progress = len(self._job_tracker.jobs_in_progress)
        cap = self._max_jobs_in_progress_allowed(0)
        if in_progress >= cap:
            return (
                f"its model is resident and idle on process {process.process_id}, but the concurrency cap is "
                f"reached (in_progress={in_progress}, cap={cap})"
            )
        if not self._concurrent_overlap_allowed(head, target_device_index=process.device_index):
            return (
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
                return (
                    f"its model is resident and idle on process {process.process_id}, but the whole-card "
                    f"residency stuck: cannot reach sole residency because {pinned}; the convergence teardown "
                    f"should have stopped that idle sibling (only the head's holder is spared), so the shrink "
                    f"has not collapsed the pool and the head never dispatches"
                )
            return (
                f"its model is resident and idle on process {process.process_id}, but its whole-card residency "
                f"has not yet converged to sole residency (siblings still tearing down or the device draining)"
            )

        if (
            self._post_processing_dispatch_defer is not None
            and head.id_ is not None
            and self._post_processing_dispatch_defer[0] == str(head.id_)
        ):
            shortfall_mb = self._post_processing_dispatch_defer[1]
            return (
                f"its model is resident and idle on process {process.process_id}, but dispatch is held for "
                f"post-processing-peak headroom (~{shortfall_mb:.0f}MB short on the contended card); it keeps "
                "its head-of-queue position and dispatches once an in-flight sibling frees room"
            )

        return (
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

        Sourced from the inference processes' VRAM reports via :meth:`ProcessMap.get_free_vram_mb`, which
        the children compute through hordelib's backend-agnostic accelerator layer (comfy /
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

    def _govern_ram_pressure(self, verdict: RamPressureVerdict) -> None:
        """Degrade the worker's footprint and intake while system RAM is below the danger floor.

        The proactive counterpart to the marginal RAM budget: rather than admit a load that the absolute
        reading says will trip the kernel OOM-killer, the worker (1) pauses job pops so intake stops adding
        pressure (self-throttle, auto-resumed on cooldown), (2) evicts idle resident models, and (3) reduces
        the resident inference-process count so the multi-GB of resident weights each idle context pins is
        returned to the OS. Together these walk the host back above the floor instead of crash-looping.
        """
        now = time.time()
        until = now + _RAM_PRESSURE_PAUSE_SECONDS
        if not (self._state.self_throttle_paused and self._state.self_throttle_paused_until >= until):
            self._state.self_throttle_paused = True
            self._state.self_throttle_paused_until = until
            logger.opt(ansi=True).warning(
                f"<fg #ff8c69>System RAM below the danger floor ({verdict.reason()}); pausing job pops for "
                f"{_RAM_PRESSURE_PAUSE_SECONDS:.0f}s and shedding idle footprint so the host is not driven into "
                "an OS OOM kill. In-flight jobs finish; pops resume once RAM recovers.</>",
            )
        # Evict idle resident models (their RAM is returned to the OS once the slot respawns), then reduce the
        # resident process count: each idle context pins GB of resident weights the allocator only frees on exit.
        self.unload_models(under_pressure=True)
        self._reduce_processes_under_ram_pressure()
        # Idle-shed keeps at least one context per card and cannot help when every process is busy, so a single
        # process whose retained RAM has ballooned past the ceiling is never reclaimed by count alone. Bound it
        # directly: recycle the largest over-ceiling process (draining it first if it is busy).
        self._reclaim_over_ceiling_processes()

    def _govern_ram_pressure_if_pressured(self) -> bool:
        """Evaluate the absolute RAM danger floor and degrade the worker if it is breached.

        The per-tick entry point (distinct from the per-job :meth:`_preload_blocked_by_ram_pressure`): it runs
        the whole-host degrade response (pause pops, evict idle models, shed idle contexts, reclaim an
        over-ceiling process) whenever the host is under its floor, so a worker that never attempts a new
        preload still throttles and reclaims instead of growing into an OS OOM kill. Returns whether the host
        was under pressure. Clears the one-shot notice when the host is healthy.
        """
        verdict = self._ram_pressure_verdict()
        self._update_ram_pop_hold(verdict)
        if not verdict.under_pressure:
            self._ram_pressure_notified = False
            return False
        self._govern_ram_pressure(verdict)
        return True

    def _update_ram_pop_hold(self, verdict: RamPressureVerdict) -> None:
        """Set the soft, pre-floor pop hold when RAM is approaching the danger floor or a drain is in flight.

        The hard floor engages :attr:`self_throttle_paused`; this softer band stops the popper starting a new
        job's ttl clock *before* the host is critical, so a job does not age past its ttl waiting on a degraded
        worker and get aborted by the horde as too slow. The approach margin reuses the marginal RAM reserve
        (no separate knob): the hold engages once measured available RAM is within ``ram_reserve_mb`` of the
        floor, and also while any process is being drained for reclaim. Cleared when RAM recovers and no
        process is draining. Never raises on a bad config value.
        """
        margin_mb = config_number(self._runtime_config.bridge_data.ram_reserve_mb)
        if margin_mb is None:
            margin_mb = 4096.0
        approaching = verdict.available_mb is not None and (verdict.available_mb - verdict.floor_mb) < margin_mb
        self._state.ram_pressure_pop_hold = bool(verdict.under_pressure or approaching) or bool(
            self._processes_draining_for_ram,
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

    def _reclaim_over_ceiling_processes(self) -> None:
        """Reclaim the largest inference process whose resident RAM is at/above the per-process ceiling.

        Only ever called while the host is under its RAM danger floor, so a roomy host never recycles. Acts on
        one process per invocation (the largest over-ceiling one) to avoid emptying every card at once: if it is
        idle it is recycled now (its allocator-retained pages return to the OS on respawn); if it is busy it is
        marked draining (fed no new work by the dispatch/preload target selection) so its in-flight job finishes,
        and a later invocation recycles it once it is idle. A process that has fallen back under the ceiling is
        cleared from the drain set.
        """
        ceiling_mb = self._ram_per_process_ceiling_mb()
        if ceiling_mb is None:
            self._processes_draining_for_ram.clear()
            return

        over_ceiling: list[HordeProcessInfo] = []
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.ram_usage_bytes / (1024 * 1024) >= ceiling_mb:
                over_ceiling.append(process_info)
            else:
                self._processes_draining_for_ram.discard(process_info.process_id)

        if not over_ceiling:
            return

        target = max(over_ceiling, key=lambda p: p.ram_usage_bytes)
        used_mb = target.ram_usage_bytes / (1024 * 1024)
        if target.is_process_busy():
            if target.process_id not in self._processes_draining_for_ram:
                self._processes_draining_for_ram.add(target.process_id)
                logger.opt(ansi=True).warning(
                    f"<fg #ff8c69>Inference process {target.process_id} holds {used_mb:.0f} MB RAM (>= the "
                    f"{ceiling_mb:.0f} MB per-process ceiling) while the host is under its RAM floor; draining it "
                    "(no new work) so it can be recycled once its in-flight job finishes.</>",
                )
            return

        logger.opt(ansi=True).warning(
            f"<fg #ff8c69>Inference process {target.process_id} holds {used_mb:.0f} MB RAM (>= the "
            f"{ceiling_mb:.0f} MB per-process ceiling) while the host is under its RAM floor; recycling it to "
            "return the retained RAM to the OS.</>",
        )
        self._processes_draining_for_ram.discard(target.process_id)
        self._process_lifecycle._replace_inference_process(target, intentional_reclaim=True)
        self._ram_reclaim_cycle_at = time.time()
        self._record_churn("process_cycle")

    def _reduce_processes_under_ram_pressure(self) -> None:
        """Shed idle resident inference processes to return their resident-weight RAM to the OS.

        The RAM analogue of :attr:`StreamForecast.needs_process_count_reduction`: with the host over the
        danger floor, the structural remedy is fewer resident contexts, not another load on top. Targets the
        count needed for in-flight work (at least one), shedding at least one idle sibling; only idle
        processes are stopped (``scale_inference_processes`` never kills a busy slot), so live work is spared.

        On a multi-GPU host the reduction is applied per card so it never empties a card of every context: a
        worker-wide shrink would let the worldwide victim search stop every idle process regardless of card,
        leaving a card with zero contexts (and so an idle GPU) until restored. The single-GPU / worker-wide
        path is unchanged.
        """
        if self._multi_gpu_routing_active:
            self._reduce_processes_per_card_under_ram_pressure()
            return
        current = self._process_map.num_loaded_inference_processes()
        if current <= 1:
            return
        needed = max(1, len(self._job_tracker.jobs_in_progress))
        target = max(1, min(current - 1, needed))
        self._process_lifecycle.scale_inference_processes(target, device_index=None)

    def _reduce_processes_per_card_under_ram_pressure(self) -> None:
        """Shed idle contexts on each driven card while leaving every card at least one resident context.

        Each card is reduced toward the count its own in-flight work needs (at least one), so no GPU is
        stranded by the worker-wide victim search. Every card actually shrunk is recorded in
        :attr:`_ram_pressure_shed_cards` so :meth:`_restore_processes_after_ram_pressure` grows it back once
        the host clears the danger floor.
        """
        for device_index in sorted(self._card_runtimes):
            current = self._process_map.num_loaded_inference_processes(device_index=device_index)
            if current <= 1:
                continue
            needed = max(1, self._card_inference_load(device_index))
            target = max(1, min(current - 1, needed))
            after = self._process_lifecycle.scale_inference_processes(target, device_index=device_index)
            if after < current:
                self._ram_pressure_shed_cards.add(device_index)

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
        """Grow cards shed by the RAM-pressure reduction back toward plan as system RAM proves it can hold them.

        The reduction sheds idle contexts to walk the host back above its absolute RAM floor; nothing else
        re-establishes them, so without this a card that lost its contexts to a RAM spike sits idle for the
        rest of the run while the surviving card serializes the work. Mirrors the whole-card residency restore
        and is scoped to the pressure episode: only cards the reduction recorded are grown back, only once RAM
        is no longer under pressure and the self-throttle pop-pause has lapsed, and never a card a whole-card
        residency is deliberately holding down (that path runs its own restore).

        Growth is RAM-gated and incremental: one context per card per cycle, and only while measured RAM
        headroom can hold another resident working set (estimated from the largest live context). This keeps
        the restore from fighting the budget on a host whose full process plan over-commits the shared RAM
        pool, where growing straight back to plan would re-trip the reduction and oscillate. A card stays
        pending until it reaches its plan or RAM proves it cannot sustain more.
        """
        if not self._ram_pressure_shed_cards:
            return
        if self._ram_pressure_verdict().under_pressure:
            return
        if self._state.self_throttle_paused and time.time() < self._state.self_throttle_paused_until:
            return

        cards_held_by_residency = {index for index, _residency in self._held_residencies()}
        per_context_ram_mb = self._estimated_resident_context_ram_mb()
        cards_to_stop_tracking: set[int] = set()
        for device_index in sorted(self._ram_pressure_shed_cards):
            card = self._card_runtimes.get(device_index)
            if card is None or device_index in cards_held_by_residency:
                # A residency-held (or unknown) card is restored by its own path; stop tracking it here.
                cards_to_stop_tracking.add(device_index)
                continue
            current = self._process_map.num_loaded_inference_processes(device_index=device_index)
            if current >= card.target_process_count:
                cards_to_stop_tracking.add(device_index)
                continue
            if self._ram_headroom_for_additional_context_mb() < per_context_ram_mb:
                # No RAM to sustain another resident context yet; keep the card pending and retry next cycle.
                continue
            after = self._process_lifecycle.scale_inference_processes(current + 1, device_index=device_index)
            logger.opt(ansi=True).info(
                f"<fg #7b7d7d>System RAM has headroom; restoring an inference context on device {device_index} "
                f"({current} -> {after} of {card.target_process_count}) so the card resumes serving.</>",
            )
            if after >= card.target_process_count:
                cards_to_stop_tracking.add(device_index)

        self._ram_pressure_shed_cards -= cards_to_stop_tracking

    def _committed_post_processing_reserve_mb(self, *, device_index: int | None = None) -> float:
        """Sum the imminent post-processing-phase VRAM peaks of jobs currently in post-processing (MB).

        When a job finishes sampling it releases its inference slot for overlap *before* its
        upscaler/face-fixer allocates, so the measured free-VRAM figure still reads as if that headroom
        were available. This derives the VRAM that is therefore spoken-for but not yet realised: for each
        inference process in ``INFERENCE_POST_PROCESSING``, the predicted post-processing peak of the job it
        is running. Subtracted from measured free VRAM at the dispatch/overlap gates and folded into the
        residency forecast, it stops a freshly-released slot being handed VRAM an in-flight job is about to
        claim. Returns 0.0 when nothing is post-processing (so the reserve self-scales away on roomy cards)
        or when the feature is disabled. Never raises: a bad per-job estimate is skipped.

        Args:
            device_index: When given, sum only the post-processing peaks on that card (the per-card reserve,
                since a post-processing peak only spends the VRAM of the card running it); when None, sum
                across every card.
        """
        if not self._budget_active() or not self._runtime_config.bridge_data.post_processing_budget_reserve_enabled:
            return 0.0
        return sum(
            (
                self._job_post_processing_peak_mb(job)
                for process_info, job in self._in_flight_jobs_by_inference_process(device_index=device_index)
                if process_info.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
            ),
            0.0,
        )

    def _job_post_processing_peak_mb(self, job: ImageGenerateJobPopResponse) -> float:
        """Return ``job``'s predicted post-processing-phase VRAM peak (MB), or 0.0 when none/unknown."""
        baseline = self._model_metadata.get_baseline(job.model) if job.model is not None else None
        peak_mb = predict_job_post_processing_vram_mb(job, str(baseline) if baseline is not None else None)
        return peak_mb if peak_mb is not None else 0.0

    def _in_flight_jobs_by_inference_process(
        self,
        *,
        device_index: int | None,
    ) -> list[tuple[HordeProcessInfo, ImageGenerateJobPopResponse]]:
        """Return each inference process running an in-flight job, paired with that job.

        A process's ``last_job_referenced`` still points at its job through the post-processing stage (the
        result that ends the inference stage only arrives once post-processing completes), so a process whose
        reference has already left the in-progress set is skipped as stale. ``device_index`` restricts the
        scan to one card (a post-processing peak only spends the VRAM of the card running it); None spans
        every card.
        """
        jobs_in_progress = self._job_tracker.jobs_in_progress
        pairs: list[tuple[HordeProcessInfo, ImageGenerateJobPopResponse]] = []
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            job = process_info.last_job_referenced
            if job is None or job not in jobs_in_progress:
                continue
            pairs.append((process_info, job))
        return pairs

    def _imminent_post_processing_reserve_mb(self, *, device_index: int | None = None) -> float:
        """Sum the post-processing peaks of in-flight jobs that *will* post-process but have not reached it.

        The committed reserve only counts a peak once its process is in ``INFERENCE_POST_PROCESSING``. But the
        over-commit that stalls an upscaler is set up earlier: while a job is still sampling, the overlap /
        pre-staging path can admit a *second* concurrent job, and by the time the first job reaches its
        upscaler the card is already committed to both. This counts the not-yet-realised peak of every
        in-flight job currently sampling or staging (its process busy but not yet post-processing), so the
        overlap gate can withhold a fresh sample when an imminent peak would not co-fit. Disjoint from the
        committed reserve (which counts the realised peaks), so the two sum without double-counting. Returns
        0.0 when nothing in flight will post-process or the reserve is disabled.

        Args:
            device_index: When given, sum only the imminent peaks on that card; when None, sum across cards.
        """
        if not self._budget_active() or not self._runtime_config.bridge_data.post_processing_budget_reserve_enabled:
            return 0.0
        return sum(
            (
                self._job_post_processing_peak_mb(job)
                for process_info, job in self._in_flight_jobs_by_inference_process(device_index=device_index)
                if process_info.last_process_state != HordeProcessState.INFERENCE_POST_PROCESSING
                and process_info.is_process_busy()
            ),
            0.0,
        )

    _IMAGE_PP_RESERVE_FLOW = "image_post_processing"
    """The shared-ledger flow namespace under which this scheduler registers its post-processing reserve."""

    def _committed_vram_reserve_mb(self, *, device_index: int | None = None) -> float:
        """Return the combined committed VRAM (MB) across every flow, refreshing this scheduler's own entry.

        Each call re-derives the image-generation post-processing reserve from live process state and
        publishes it into the shared :class:`CommittedReserveLedger` as a single aggregate entry, then
        returns the ledger total (which also includes any in-flight alchemy forms other flows registered).
        Re-publishing every call keeps the entry self-healing: it always reflects current state and never
        leaks. This combined figure is what the VRAM admission and residency-forecast gates subtract, so a
        freshly-released slot is not handed VRAM that image post-processing *or* concurrent alchemy is about
        to claim.

        Args:
            device_index: When given, return that card's committed reserve: its own post-processing reserve
                plus the worker-global non-image-post-processing flows (alchemy is not card-attributed, so it
                is charged conservatively against the card). When None, return the worker-wide ledger total
                (the single-GPU reading). The worker-wide image-post-processing aggregate is refreshed in the
                ledger either way, so the status snapshot and other readers are unaffected.
        """
        self._reserve_ledger.set(
            self._IMAGE_PP_RESERVE_FLOW,
            "aggregate",
            vram_mb=self._committed_post_processing_reserve_mb(),
        )
        if device_index is None:
            return self._reserve_ledger.total_vram_mb()
        per_card_post_processing = self._committed_post_processing_reserve_mb(device_index=device_index)
        return per_card_post_processing + self._reserve_ledger.total_vram_mb_excluding(self._IMAGE_PP_RESERVE_FLOW)

    def _max_jobs_in_progress_allowed(
        self,
        processes_post_processing: int,
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
            processes_post_processing: How many processes are mid post-processing (per the caller's
                scope, global or per-card), used for the overlap bump.
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

        # The post-processing overlap bump raises the cap *because* a process is post-processing, but that
        # process is holding (or about to hold) its upscaler/face-fixer VRAM peak, the worst moment to admit
        # another job. Grant the bump only while the device still has staging headroom once the in-flight
        # post-processing reserve is held back; otherwise drop it so the post-proc peaks of several jobs
        # cannot align and over-commit the card. Gated on the feature flag so behavior is unchanged when off.
        post_processing_bump = processes_post_processing
        if (
            post_processing_bump > 0
            and self._budget_active()
            and self._runtime_config.bridge_data.post_processing_budget_reserve_enabled
        ):
            free_vram_mb = self._process_map.get_free_vram_mb()
            if free_vram_mb is not None:
                bump_floor = max(_SPECULATIVE_DISPATCH_MIN_FREE_VRAM_MB, self._vram_budget.reserve_mb)
                available_mb = (
                    free_vram_mb - self._committed_vram_reserve_mb() - self._imminent_post_processing_reserve_mb()
                )
                if available_mb < bump_floor:
                    post_processing_bump = 0

        base = concurrent_ceiling + post_processing_bump
        if not self._runtime_config.bridge_data.gpu_sampling_lease_enabled:
            return base

        # Floor the speculative-staging headroom at the budget's reserve when it is active, so
        # pre-staging never eats into the VRAM the budget is holding back for the in-flight job's
        # transient spikes; otherwise keep the standalone staging threshold.
        staging_floor = _SPECULATIVE_DISPATCH_MIN_FREE_VRAM_MB
        if self._budget_active():
            staging_floor = max(staging_floor, self._vram_budget.reserve_mb)

        free_vram_mb = self._process_map.get_free_vram_mb()
        # Subtract the imminent post-processing peak of any in-flight job that is still sampling: pre-staging a
        # second concurrent sample onto a card already owed a big upscale peak is exactly the overlap that
        # over-commits the device and stalls the upscaler. Self-scales to zero when nothing in flight will
        # post-process (and when the reserve is disabled), so ordinary overlap is unaffected.
        if free_vram_mb is None or (free_vram_mb - self._imminent_post_processing_reserve_mb()) < staging_floor:
            return base

        return process_ceiling + post_processing_bump

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

        Rules, scaled by model size:
            * The first job (nothing in flight) always starts.
            * An extra-large or batched candidate never joins a busy card; it wants the card to itself.
            * An extra-large or batched job already in flight never shares the card.
            * Otherwise the running job must have made size-appropriate headway: none for light+light,
              modest when one side is heavy, considerable for two heavy jobs.

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
        if candidate_tier >= _ModelSizeTier.EXTRA_LARGE or self._job_batch_amount(candidate_job) > 1:
            return False

        for job in in_progress_jobs:
            running_tier = self._model_size_tier(job.model)
            if running_tier >= _ModelSizeTier.EXTRA_LARGE or self._job_batch_amount(job) > 1:
                return False

            required_headway = self._required_overlap_headway(running_tier, candidate_tier)
            if required_headway <= 0.0:
                continue
            if self._in_flight_progress_fraction(job) < required_headway:
                return False

        return True

    def _expire_stale_model_map_entries(self) -> list[str]:
        """Expire model-map entries whose owning process can no longer be loading that model."""
        expired: list[str] = []
        loading_owner_states = {
            HordeProcessState.PROCESS_STARTING,
            HordeProcessState.DOWNLOADING_MODEL,
            HordeProcessState.PRELOADING_MODEL,
        }

        for model_name, model_info in list(self._horde_model_map.root.items()):
            process_info = self._process_map.get(model_info.process_id)
            if process_info is None:
                self._horde_model_map.expire_entry(model_name)
                expired.append(model_name)
                logger.warning(
                    f"Expiring stale model-map entry for {model_name}: process {model_info.process_id} is gone.",
                )
                continue

            if (
                model_info.horde_model_load_state == ModelLoadState.LOADING
                and process_info.last_process_state not in loading_owner_states
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
        ram_pressure = self._ram_pressure_verdict()
        if not ram_pressure.under_pressure:
            self._ram_pressure_notified = False
            return False
        # The per-tick governor (_govern_ram_pressure_if_pressured, run at the top of preload_models) has
        # already driven the whole-host degrade response this cycle; here we only defer *this* preload and
        # surface the per-model notice once so the loop does not route a new model's weights through a host
        # already on the edge.
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
        needs_teardown_path = forecast.needs_exclusive_residency or forecast.needs_process_count_reduction
        whole_card_demanded = self._whole_card_residency_enabled() and needs_teardown_path and is_head_blocker
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

        vram_verdict = self._vram_budget.check_job(
            job,
            baseline,
            self._measured_free_vram_mb(device_index=target_device_index),
            committed_reserve_mb=self._committed_vram_reserve_mb(device_index=target_device_index),
        )
        if not vram_verdict.fits:
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
            # Reclamation is exhausted when nothing more could be freed: the predicted peak + reserve exceeds
            # achievable free VRAM even with every idle resident copy evicted. The burden estimate is a
            # deliberately conservative single-resident-peak figure, but a large combined checkpoint is
            # streamed through VRAM component-by-component by the backend, so its true peak is well under the
            # summed estimate. A head-of-queue job must therefore be given the device rather than deferred
            # forever (which would wedge the queue and fault the head anyway). Admit it best-effort when no
            # live job holds the device, after also reclaiming system RAM from idle residents: a heavy head
            # loads its checkpoint through RAM first, so admitting it onto a RAM-pressured host is a likely
            # load-time fault. Tag it so a crash/hang of its over-committed slot is classified as a resource
            # failure (earning the bounded, isolated retry) instead of a plain re-dispatch.
            if not (is_head_blocker and not freed and no_live_resource_consumer):
                return False

            # Circuit-breaker: a model the device genuinely cannot run faults every over-budget attempt no
            # matter how it is isolated. Once its consecutive-fault streak crosses the configured threshold it
            # is held back (not admitted here, not popped in the popper) for a cooldown, so the worker stops
            # dropping jobs faster than the horde server tolerates and is never forced into maintenance.
            if self._is_model_locally_unservable(job.model, device_index=target_device_index):
                if not self._unservable_admit_notified.get(job.model, False):
                    logger.opt(ansi=True).warning(
                        f"<fg #ff8c69>Model {job.model} keeps faulting over the VRAM budget; held "
                        f"back as locally unservable and not admitted. "
                        f"{self._process_map.residency_snapshot()}</>",
                    )
                    self._unservable_admit_notified[job.model] = True
                return False
            self._unservable_admit_notified.pop(job.model, None)

            # Before evicting every resident model and admitting the head exclusively, check whether the live
            # process *contexts* are the over-commit rather than the resident models. The weight-based teardown
            # gates leave a moderate head co-resident because its weights fit after a model eviction, but its
            # activation peak does not fit while this many contexts are live (the threads>1 regime, where each
            # extra context retains VRAM the allocator never returns). When more inference contexts are live
            # than the head's weights-plus-reserve can co-reside with (``max_resident_processes``, sized from
            # the measured per-context cost), reducing the process count is the structural remedy: it returns a
            # context's retained VRAM so the head and a sibling model co-reside and pipeline. Evicting every
            # model and loading exclusively instead strands the models the next jobs reuse and churns a full
            # reload per job. The depth is sized from the verdict's own rejected peak so the reduction fires
            # exactly when admission would reject.
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
            if context_reduction_demanded and not self._whole_card_warranted(forecast):
                # The reduction depth rests on the per-context overhead; when that is the unmeasured fallback
                # for a card-light model, the demand is an over-count phantom. Decline it and fall through to
                # the ordinary evict-all admit rather than reserving the card.
                self._log_whole_card_declined(job, forecast)
            elif context_reduction_demanded:
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
                return False

            self.unload_models(under_pressure=True, for_head_of_queue=True)
            self._mark_overbudget_admit(job)
            self._log_overbudget_admit(job)
            return True

        self._vram_budget_defer_notified = False

        ram_verdict = self._ram_budget.check_job(
            job,
            baseline,
            self._measured_available_ram_mb(),
            committed_reserve_mb=self._reserve_ledger.total_ram_mb(),
        )
        if not ram_verdict.fits:
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
            if not reclaimed:
                cycled = self._replace_stale_ram_unload_process()
                # Cycling a stuck idle slot reclaims RAM by restarting it, so wait for that. Only when even
                # cycling finds nothing to reclaim (and no live job holds RAM) is the head truly unservable by
                # waiting; admit it best-effort then, mirroring the VRAM branch, rather than starving it.
                if not (is_head_blocker and not cycled and no_live_resource_consumer):
                    return False
                logger.opt(ansi=True).warning(
                    f"<fg #f0beff>RAM budget cannot fit head-of-queue model {job.model} even after "
                    "reclaiming all idle RAM, and no live job holds memory; admitting it best-effort "
                    "rather than wedging the queue.</>",
                )
            else:
                return False
        else:
            self._ram_budget_defer_notified = False

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
            self._mark_overbudget_admit(job)
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
            self._mark_overbudget_admit(job)
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

        Returns:
            True if a model was preloaded, False otherwise.
        """
        bridge_data = self._runtime_config.bridge_data
        self._restore_siblings_after_whole_card()
        self._converge_whole_card_residency()
        self._expire_stale_model_map_entries()

        # Absolute system-RAM floor, evaluated every tick rather than only when a new model needs preloading.
        # A steady-state worker whose pending jobs all target already-resident models reaches the
        # ``loaded_models == pending_models`` early-return below without ever checking the floor, so its
        # resident set can grow into an OS OOM kill with the governor asleep. Governing here (before that
        # return) drives the shed/throttle/per-process-reclaim response regardless of whether anything preloads.
        # Gated on the budget being active, the same switch the rest of the memory machinery uses.
        if self._budget_active():
            self._govern_ram_pressure_if_pressured()

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
        # model (see the budget-defer branches below); a later job whose turn has not come never
        # displaces a resident head.
        in_progress_jobs = self._job_tracker.jobs_in_progress
        head_job = next((j for j in self._job_tracker.jobs_pending_inference if j not in in_progress_jobs), None)
        self._update_head_starvation_timer(head_job)

        for job in self._job_tracker.jobs_pending_inference:
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
                continue

            if job.model in loaded_models:
                continue

            # Absolute system-RAM floor (degrade, never crash): loading a new model routes its weights through
            # system RAM first, so admitting one while the host is already below its danger floor is the OS
            # OOM kill, not progress. This gates every admit path (best-effort, head-starvation force-admit,
            # and whole-card-terminal all sit below this point), independent of the marginal RAM budget, which
            # can pass on a job's small estimate while the whole host is on the edge (resident weights + the
            # safety process + other apps). Instead of loading, shed idle footprint and pause pops until RAM
            # recovers. Gated on the budget being active (the same switch the rest of the memory machinery uses).
            if self._budget_active() and self._preload_blocked_by_ram_pressure(job):
                return False

            # An exclusively-admitted over-budget job has the whole device; do not stage another model's
            # weights concurrently (a second resident load is exactly what spills the exclusive job's
            # weights to system RAM). The exclusive job's own preload is still allowed through.
            if self._job_tracker.has_exclusive_job_in_progress() and not self._job_tracker.is_admitted_exclusive(
                job,
            ):
                continue

            is_head_blocker = head_job is not None and job is head_job

            processes_with_model_for_queued_job: list[int] = (
                self._process_lifecycle.get_processes_with_model_for_queued_job()
            )

            if self._process_map.num_loaded_inference_processes() < (
                len(self._job_tracker.jobs_pending_inference) + len(self._job_tracker.jobs_in_progress)
            ):
                processes_with_model_for_queued_job = [
                    p.process_id for p in self._process_map.values() if p.is_process_busy()
                ]

            # Model->process affinity: in the models<=processes regime, never displace the last
            # resident copy of a still-wanted model. Without this, hot models' second instances
            # (the 2-per-model cap) consume the spare processes and the fallback evicts a cold
            # model's only copy, forcing it to disk-reload on its next job; the dominant
            # GPU-duty-cycle tax measured on the multi-model soak. The working model set is taken
            # from live state (loaded + in-flight + pending), not bridge_data.image_models_to_load,
            # because the harness/canned path never resolves that config field.
            inference_process_models = {
                p.process_id: p.loaded_horde_model_name
                for p in self._process_map.values()
                if p.process_type == HordeProcessType.INFERENCE
            }
            wanted_models: set[str] = {m for m in inference_process_models.values() if m is not None}
            wanted_models.update(j.model for j in self._job_tracker.jobs_pending_inference if j.model is not None)
            wanted_models.update(j.model for j in self._job_tracker.jobs_in_progress if j.model is not None)
            if affinity_active(len(wanted_models), self._max_inference_processes):
                protected = compute_protected_processes(inference_process_models, wanted_models)
                if protected:
                    processes_with_model_for_queued_job = list(
                        set(processes_with_model_for_queued_job) | protected,
                    )

            # A process marked draining for RAM (its resident footprint crossed the per-process ceiling under
            # pressure) must be fed no new work so it can go idle and be recycled; exclude it as a preload target.
            # It is only ever excluded here as a target, never wedging the queue: the head-starvation fallback
            # below can still reach it, and the governor recycles it before dispatch once it is idle.
            preload_disallowed = processes_with_model_for_queued_job
            if self._processes_draining_for_ram:
                preload_disallowed = list(
                    set(processes_with_model_for_queued_job) | self._processes_draining_for_ram,
                )

            # On a multi-GPU host this also chooses *which* card to load onto: an eligible card already
            # holding the model first, then the least-loaded eligible card. Single-GPU returns the first
            # available slot exactly as before.
            available_process = self._select_preload_process(job, preload_disallowed)

            if available_process is None and is_head_blocker:
                # The head of the queue could not get a slot because affinity (or the queued-model
                # guard) protected every idle process. Affinity is provisioned against the
                # inference-process *ceiling*, so with more resident models than running processes it
                # can pin every slot and starve a genuinely-queued head, wedging the whole worker. The
                # head must make progress regardless of whether the measured budget is active, so fall
                # back to a displacement target that spares live work and prefers an idle resident model
                # no queued job needs. This is the budget-independent counterpart to the budget-gated
                # make-room escalation further below.
                available_process = self._select_head_room_process()

            if available_process is None:
                return False

            if (
                available_process.last_process_state != HordeProcessState.WAITING_FOR_JOB
                and available_process.loaded_horde_model_name is not None
                and bridge_data.cycle_process_on_model_change
                and not self._state.shutting_down
            ):
                self._process_lifecycle._replace_inference_process(available_process)
                return False

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

            at_least_one_preloading_process = num_preloading_processes >= 1
            very_fast_disk_mode_enabled = bridge_data.very_fast_disk_mode
            if very_fast_disk_mode_enabled:
                max_concurrent_inference_processes_reached = num_preloading_processes >= (
                    self._max_concurrent_inference_processes + 1
                )
            else:
                max_concurrent_inference_processes_reached = (
                    num_preloading_processes >= self._max_concurrent_inference_processes
                )

            if (not very_fast_disk_mode_enabled and at_least_one_preloading_process) or (
                very_fast_disk_mode_enabled and max_concurrent_inference_processes_reached
            ):
                if not self._preload_delay_notified:
                    logger.opt(ansi=True).info(
                        "<fg #7b7d7d>"
                        f"Already preloading {num_preloading_processes} models, waiting for one to finish before "
                        f"preloading {job.model}"
                        "</>",
                    )
                    self._preload_delay_notified = True
                return False

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
                return False

            return self._send_preload(job, available_process)

        return False

    def _select_head_room_process(self) -> HordeProcessInfo | None:
        """Pick an idle inference process to free for a starved head-of-queue job, or None.

        Used when the normal preload picker found no slot because affinity (provisioned against the
        inference-process ceiling) or the queued-model guard protected every idle process. The head must
        still make progress, so this deliberately overrides those guards. It never returns a process
        running live work (only ``can_accept_job()`` slots, and never one whose model is in progress) and
        prefers the cheapest displacement: an empty slot, then one holding a resident model no pending or
        in-progress job needs, then, as a last resort, one holding a merely-queued model.
        """
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}
        pending_models = {job.model for job in self._job_tracker.jobs_pending_inference if job.model is not None}

        candidates = [
            process_info
            for process_info in self._process_map.values()
            if process_info.process_type == HordeProcessType.INFERENCE
            and process_info.can_accept_job()
            and process_info.loaded_horde_model_name not in in_progress_models
        ]
        if not candidates:
            return None

        def _displacement_cost(process_info: HordeProcessInfo) -> int:
            model_name = process_info.loaded_horde_model_name
            if model_name is None:
                return 0
            if model_name not in pending_models:
                return 1
            return 2

        return min(candidates, key=_displacement_cost)

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

        def card_placement_key(device_index: int) -> tuple[int, int]:
            already_serves_model = any(
                process.loaded_horde_model_name == job.model and process.device_index == device_index
                for process in self._process_map.values()
            )
            return (0 if already_serves_model else 1, self._card_inference_load(device_index))

        for device_index in sorted(eligible, key=card_placement_key):
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

        candidate_job_size = 25

        if bridge_data.high_performance_mode:
            candidate_job_size = 100
        elif bridge_data.moderate_performance_mode:
            candidate_job_size = 50

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

        processes_post_processing = 0
        if self.post_process_job_overlap_allowed:
            processes_post_processing = self._process_map.num_busy_with_post_processing(
                device_index=target_card.device_index if target_card is not None else None,
            )

        if target_card is not None:
            jobs_in_progress_count = len(self._jobs_in_progress_on_card(target_card.device_index))
        else:
            jobs_in_progress_count = len(self._job_tracker.jobs_in_progress)
        max_jobs_allowed = self._max_jobs_in_progress_allowed(processes_post_processing, card=target_card)
        if jobs_in_progress_count >= max_jobs_allowed:
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
            if (process_with_model.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL) or (
                self.post_process_job_overlap_allowed
                and process_with_model.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
            ):
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
        - **budget fits**: the VRAM budget confirms the model's footprint fits the *measured* free VRAM,
          so retention never starves a different model another process must load. A missing budget or
          unmeasured VRAM yields False: retention is granted on evidence, never assumed.

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
        verdict = self._vram_budget.check_job(
            dispatched_job,
            baseline,
            free_vram_mb,
            committed_reserve_mb=self._committed_vram_reserve_mb(device_index=device_index),
        )
        return verdict.fits

    def _idle_post_processing_reclaim_siblings(
        self,
        *,
        dispatching_process_id: int | None,
        device_index: int | None,
    ) -> list[HordeProcessInfo]:
        """Return the idle inference processes whose VRAM a post-processing reclaim may target.

        Excludes the process dispatching the post-processing job (it is the one that needs the room) and,
        when ``device_index`` is given, any process pinned to a different card (freeing another card's VRAM
        returns nothing to this one). A busy process is never a candidate: only a process that has finished
        its work holds reclaimable room without interrupting live inference. Processes already ending are
        skipped since their context is on its way out regardless.
        """
        candidates: list[HordeProcessInfo] = []
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.process_id == dispatching_process_id:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.is_process_busy():
                continue
            if process_info.last_process_state in (
                HordeProcessState.PROCESS_ENDING,
                HordeProcessState.PROCESS_ENDED,
            ):
                continue
            candidates.append(process_info)
        return candidates

    def _evictable_sibling_model_process(
        self,
        dispatched_model: str | None,
        idle_siblings: list[HordeProcessInfo],
    ) -> HordeProcessInfo | None:
        """Return the first idle sibling holding a model whose eviction would free cross-process VRAM.

        Mirrors the eviction eligibility :meth:`unload_models_from_vram` enforces, so the planner only
        promises room the enactment can actually reclaim: the sibling must hold a resident model that is not
        the dispatched job's own model, not the model of an in-progress job, and not protected by a held
        whole-card residency. A bare idle context (no resident model) is not an eviction target; it is the
        :attr:`PostProcessingReclaimAction.REDUCE_CONTEXT` rung instead.
        """
        wanted_models = self._compute_wanted_models()
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}
        for process_info in idle_siblings:
            model_name = process_info.loaded_horde_model_name
            if model_name is None or model_name == dispatched_model:
                continue
            if model_name in in_progress_models:
                continue
            if self._residency_protects_from_unload(model_name, wanted_models, vram=True, under_pressure=True):
                continue
            return process_info
        return None

    def _plan_post_processing_reclaim(
        self,
        dispatched_job: ImageGenerateJobPopResponse,
        *,
        device_index: int | None,
        dispatching_process_id: int | None = None,
    ) -> PostProcessingReclaimPlan:
        """Return how to free room for ``dispatched_job``'s imminent post-processing peak (see the actions).

        The peak (:func:`predict_job_post_processing_vram_mb`) is the upscaler/face-fixer cost that lands
        after sampling, once the card already holds this job's weights, its process contexts, and any warm
        sibling models. When it overflows the available headroom the worker must reclaim room before the
        upscaler allocates, or the allocation streams/thrashes until the post-processing watchdog replaces
        the process.

        The peak is measured against *effective* free VRAM: the measured free reading less the VRAM in-flight
        sibling work has committed or will imminently commit (the same not-yet-realised reserve the
        concurrent-dispatch gate subtracts), so an optimistic or stale reading does not let the peak look like
        it fits a card that is about to fill.

        The decision is evidence-gated: an unavailable peak estimate or unmeasured free VRAM yields a
        :attr:`PostProcessingReclaimAction.NONE` plan (never act on absent telemetry, mirroring
        :meth:`_should_keep_model_resident`). When the peak fits the effective headroom there is nothing to
        do. Otherwise the action is chosen for the room it can actually reclaim: when an idle sibling holds an
        evictable model, freeing it (:attr:`EVICT_SIBLING_MODEL`) is preferred, since that resident model is
        the cross-process room the upscaler needs and ComfyUI's in-child free cannot reach it; failing a
        reclaimable sibling, delegating to ComfyUI freeing the job's own (idle-during-upscale) weights
        in-child (:attr:`DELEGATE_IN_PROCESS`) when that alone suffices on an uncontended card; else stopping
        an idle sibling's bare context (:attr:`REDUCE_CONTEXT`); or, when none of those can host it (a
        single-process worker on a tiny card), :attr:`INSUFFICIENT`, which faults the job gracefully rather
        than thrashing the card.
        """
        baseline = self._model_metadata.get_baseline(dispatched_job.model) if dispatched_job.model else None
        post_processing_peak_mb = predict_job_post_processing_vram_mb(dispatched_job, baseline)
        if post_processing_peak_mb is None or post_processing_peak_mb <= 0.0:
            logger.debug(
                f"Post-processing reclaim: no post-processing peak estimate for job "
                f"{str(dispatched_job.id_)[:8]} ({dispatched_job.model}); not reserving (action NONE).",
            )
            return PostProcessingReclaimPlan(action=PostProcessingReclaimAction.NONE)

        measured_free_mb = self._measured_free_vram_mb(device_index=device_index)
        if measured_free_mb is None:
            logger.debug(
                f"Post-processing reclaim: no measured free VRAM for device {device_index}; not reserving "
                f"for job {str(dispatched_job.id_)[:8]} ({dispatched_job.model}) (action NONE).",
            )
            return PostProcessingReclaimPlan(action=PostProcessingReclaimAction.NONE)

        # The measured free reading lags VRAM that in-flight sibling work has committed (peaks already
        # post-processing) or will imminently commit (jobs still sampling/staging that will post-process):
        # the same not-yet-realised cost the concurrent-dispatch gate subtracts. Charging it here too keeps
        # an optimistic or stale free reading from making the peak look like it fits a card that is about to
        # fill, which would otherwise leave the upscaler to allocate into a full card. (This job is not yet
        # in-flight at plan time, so it is excluded from both reserves and never charged against itself.)
        committed_reserve_mb = self._committed_vram_reserve_mb(device_index=device_index)
        imminent_reserve_mb = self._imminent_post_processing_reserve_mb(device_index=device_index)
        effective_free_mb = measured_free_mb - committed_reserve_mb - imminent_reserve_mb

        own_weights_mb = predict_job_weight_mb(dispatched_job, baseline) or 0.0
        idle_siblings = self._idle_post_processing_reclaim_siblings(
            dispatching_process_id=dispatching_process_id,
            device_index=device_index,
        )
        evictable_sibling = self._evictable_sibling_model_process(dispatched_job.model, idle_siblings)
        shortfall_mb = post_processing_peak_mb - effective_free_mb

        if post_processing_peak_mb <= effective_free_mb:
            plan = PostProcessingReclaimPlan(action=PostProcessingReclaimAction.NONE)
        elif evictable_sibling is not None:
            # The peak does not fit the effective free VRAM and an idle sibling holds an evictable model: free
            # it. This is preferred over the in-child own-weights delegation even when freeing this job's own
            # weights would nominally cover the peak, because on a contended card the room the upscaler needs
            # is occupied by the *sibling's* resident model, which ComfyUI's in-child free cannot reclaim.
            # Relying on the own-weights credit while a sibling fills the card lets the upscaler allocate into
            # a full card and stall until the watchdog reaps the slot.
            plan = PostProcessingReclaimPlan(
                action=PostProcessingReclaimAction.EVICT_SIBLING_MODEL,
                target_process_id=evictable_sibling.process_id,
                shortfall_mb=shortfall_mb,
            )
        elif post_processing_peak_mb <= effective_free_mb + own_weights_mb:
            # No reclaimable sibling holds the room, but freeing this job's own (idle-during-upscale) weights,
            # which ComfyUI does in-child once sampling ends, yields enough on an otherwise-uncontended card.
            plan = PostProcessingReclaimPlan(action=PostProcessingReclaimAction.DELEGATE_IN_PROCESS)
        elif idle_siblings:
            plan = PostProcessingReclaimPlan(
                action=PostProcessingReclaimAction.REDUCE_CONTEXT,
                target_process_id=idle_siblings[0].process_id,
                shortfall_mb=shortfall_mb,
            )
        elif self._post_processing_peak_fits_solo(
            post_processing_peak_mb,
            device_index=device_index,
        ) and self._post_processing_reclaim_can_wait(
            dispatching_process_id=dispatching_process_id,
            device_index=device_index,
        ):
            # Nothing idle is reclaimable right now, but the peak would fit once the card drains to this job's
            # process alone, and a sibling is mid-inference whose completion frees that room. Hold the dispatch
            # (the job keeps its head-of-queue position) rather than faulting a job the card can host moments
            # from now. The wait is wedge-exempt while sibling inference is in progress, and self-bounds: once
            # no sibling is left in flight to free room, this condition fails and the plan faults terminally.
            plan = PostProcessingReclaimPlan(
                action=PostProcessingReclaimAction.DEFER,
                shortfall_mb=shortfall_mb,
            )
        else:
            plan = PostProcessingReclaimPlan(
                action=PostProcessingReclaimAction.INSUFFICIENT,
                shortfall_mb=shortfall_mb,
            )

        logger.debug(
            f"Post-processing reclaim plan for job {str(dispatched_job.id_)[:8]} ({dispatched_job.model}) on "
            f"device {device_index}: {plan.action.name} (peak ~{post_processing_peak_mb:.0f}MB vs effective "
            f"free {effective_free_mb:.0f}MB [measured {measured_free_mb:.0f}MB - committed "
            f"{committed_reserve_mb:.0f}MB - imminent {imminent_reserve_mb:.0f}MB]; own-weights "
            f"~{own_weights_mb:.0f}MB; {len(idle_siblings)} idle sibling(s), evictable="
            f"{evictable_sibling is not None}).",
        )
        return plan

    def _post_processing_peak_fits_solo(self, post_processing_peak_mb: float, *, device_index: int | None) -> bool:
        """Whether a post-processing peak fits the card drained to this job's process alone.

        The achievable headroom under sole residency is the device total less one process context (the
        per-process overhead the upscaler's own process keeps) and the configured VRAM reserve; the job's own
        diffusion weights are freed in-child during the upscale, so they are not charged here (mirroring the
        :attr:`PostProcessingReclaimAction.DELEGATE_IN_PROCESS` rung). A peak that exceeds even this can never
        be hosted on the card no matter how it drains, so waiting cannot help and the caller faults instead of
        deferring. Unknown total VRAM (cold start) reads False, the conservative direction (do not defer on a
        guess).
        """
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        if total_vram_mb is None:
            return False
        configured_reserve = config_number(self._runtime_config.bridge_data.vram_reserve_mb)
        reserve_mb = configured_reserve if configured_reserve is not None and configured_reserve >= 0 else None
        if reserve_mb is None:
            reserve_mb = _DEFAULT_VRAM_RESERVE_MB
        solo_free_mb = float(total_vram_mb) - self._per_process_overhead_mb() - reserve_mb
        return post_processing_peak_mb <= solo_free_mb

    def _post_processing_reclaim_can_wait(
        self,
        *,
        dispatching_process_id: int | None,
        device_index: int | None,
    ) -> bool:
        """Whether an in-flight sibling will free cross-process room as it completes.

        A peak that overflows the contended card now but fits the card alone is hostable by waiting, not
        faulting, only while a sibling is actively *inferencing* (sampling or post-processing): its completion
        releases the activations/upscaler VRAM and leaves its model idle and evictable, which is the room the
        upscaler needs. Siblings that are merely preloading or starting are excluded, since those grow the
        card's commitment rather than freeing it. With no such sibling (a single-process worker, or every
        sibling already idle and unevictable) waiting cannot help, so the caller faults. This is also what
        keeps the deferral wedge-safe: it holds the dispatch only over a window the recovery supervisor
        already exempts as inference-in-progress, and the moment that window closes the dispatch faults
        rather than parking.
        """
        for process_info in self._process_map.values():
            if process_info.process_type is not HordeProcessType.INFERENCE:
                continue
            if dispatching_process_id is not None and process_info.process_id == dispatching_process_id:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.last_process_state in (
                HordeProcessState.INFERENCE_STARTING,
                HordeProcessState.INFERENCE_POST_PROCESSING,
            ):
                return True
        return False

    def _note_post_processing_dispatch_defer(
        self,
        job: ImageGenerateJobPopResponse,
        shortfall_mb: float,
    ) -> None:
        """Record, and throttled-log, that a job's dispatch is held for post-processing-peak headroom.

        The job keeps its head-of-queue position; the recorded state lets the dispatch-stall diagnostic name
        the hold rather than report a gate-less scheduler stall. The log is collapsed to one line per changed
        (job, shortfall) so a multi-second hold does not flood the file each control-loop tick.
        """
        job_id = str(job.id_) if job.id_ is not None else None
        self._post_processing_dispatch_defer = (job_id, shortfall_mb)
        state_key = (job_id, self._diagnostic_mb_bucket(shortfall_mb))
        suppressed_count = self._scheduler_diagnostic_suppressed_count("post_processing_dispatch_defer", state_key)
        if suppressed_count is None:
            return
        logger.info(
            f"Post-processing reclaim: holding dispatch of job {str(job_id)[:8]} ({job.model}); its "
            f"post-processing peak overflows the contended card by ~{shortfall_mb:.0f}MB now but fits the card "
            "alone, and an in-flight sibling will free the room. Keeping its head-of-queue position rather "
            f"than faulting it{self._suppressed_suffix(suppressed_count)}.",
        )

    async def _enact_post_processing_reclaim(
        self,
        plan: PostProcessingReclaimPlan,
        dispatched_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
        *,
        device_index: int | None,
    ) -> bool:
        """Carry out a post-processing reclaim ``plan`` before the job dispatches; return whether to dispatch.

        Reuses the existing reclaim primitives: an idle sibling model is freed via
        :meth:`unload_models_from_vram` (which already enforces the in-progress/next-model/residency guards),
        and an idle context is shed via the lifecycle's ``scale_inference_processes`` (which never stops a busy
        slot). The async eviction completes during this job's sampling, so the room is ready by the time the
        post-processing peak lands. :attr:`PostProcessingReclaimAction.INSUFFICIENT` cannot be hosted, so the
        job is faulted gracefully (the horde reissues it) rather than dispatched into a guaranteed stall, and
        the caller does not send ``START_INFERENCE``.
        """
        action = plan.action
        if action in (PostProcessingReclaimAction.NONE, PostProcessingReclaimAction.DELEGATE_IN_PROCESS):
            return True

        if action is PostProcessingReclaimAction.EVICT_SIBLING_MODEL:
            logger.info(
                f"Post-processing reclaim: freeing an idle sibling model on process {plan.target_process_id} "
                f"to host job {str(dispatched_job.id_)[:8]}'s ~{plan.shortfall_mb:.0f}MB post-processing "
                "shortfall before its upscaler allocates.",
            )
            self.unload_models_from_vram(process_with_model, under_pressure=True, device_index=device_index)
            return True

        if action is PostProcessingReclaimAction.REDUCE_CONTEXT:
            # Size the reduction from the contended card itself, not the worker-wide pool: on a multi-GPU host
            # a worker-wide count would leave the per-card scale a no-op (the target never drops below the
            # card's own context count) and the peak would go unrelieved. None is the single-GPU/worker-wide
            # case, sized as before.
            if device_index is None:
                current_loaded = self._process_map.num_loaded_inference_processes()
                needed = max(1, len(self._job_tracker.jobs_in_progress))
            else:
                current_loaded = self._process_map.num_loaded_inference_processes(device_index=device_index)
                needed = max(1, self._card_inference_load(device_index))
            target = max(1, min(current_loaded - 1, needed))
            logger.warning(
                f"Post-processing reclaim: reducing inference contexts on device {device_index} to {target} "
                f"(from {current_loaded}) to host job {str(dispatched_job.id_)[:8]}'s "
                f"~{plan.shortfall_mb:.0f}MB post-processing shortfall; no idle sibling held an evictable model.",
            )
            self._process_lifecycle.scale_inference_processes(target, device_index=device_index)
            return True

        # INSUFFICIENT: no orchestrator-reclaimable room hosts the peak (e.g. a single-process worker on a
        # tiny card). Fault gracefully so the horde reissues the job, rather than dispatch it into a card it
        # cannot fit and let the post-processing watchdog reap the slot once its timeout elapses.
        logger.warning(
            f"Post-processing reclaim: job {str(dispatched_job.id_)[:8]}'s post-processing peak overflows the "
            f"card by ~{plan.shortfall_mb:.0f}MB and no cross-process VRAM can be reclaimed to host it; "
            "faulting it rather than stalling the device.",
        )
        # Terminal, not retryable: a local retry would only re-dispatch into the same unchanged, still
        # overflowing card (a guaranteed second fault) and feed the breaker a second over-commit count for one
        # job. The horde reissues the faulted job elsewhere, which is the intended recovery here.
        await self._job_tracker.handle_job_fault(
            faulted_job=dispatched_job,
            process_info=process_with_model,
            process_timeout=self._runtime_config.bridge_data.process_timeout,
            retryable=False,
        )
        self._job_tracker.note_post_processing_overcommit_fault()
        return False

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

        processes_post_processing = 0
        if self.post_process_job_overlap_allowed:
            processes_post_processing = self._process_map.num_busy_with_post_processing()

        if (
            processes_post_processing > 0
            and len(self._job_tracker.jobs_in_progress) >= self._max_concurrent_inference_processes
        ):
            logger.debug(
                "Proceeding with inference, but post processing is still running on "
                f"{processes_post_processing} processes",
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

        if self._runtime_config.bridge_data.post_processing_active_reclaim_enabled and self._budget_active():
            # Charge the job's *own* imminent post-processing peak against the device now and, if it will not
            # fit, reclaim cross-process room before sampling so the upscaler does not allocate into a full
            # card and stall. The async reclaim completes during sampling; an unhostable peak faults the job
            # here instead of dispatching it into a guaranteed stall.
            reclaim_plan = self._plan_post_processing_reclaim(
                next_job,
                device_index=dispatched_device_index,
                dispatching_process_id=process_with_model.process_id,
            )
            if reclaim_plan.action is PostProcessingReclaimAction.DEFER:
                # The peak overflows the contended card now but fits the card alone, and an in-flight sibling
                # will free the room as it completes. Hold the dispatch and keep the head's position rather
                # than faulting a job the card can host shortly; the next pass re-plans against the drained
                # card and dispatches (or, once no sibling is left in flight, faults terminally).
                self._note_post_processing_dispatch_defer(next_job, reclaim_plan.shortfall_mb)
                return False
            should_dispatch = await self._enact_post_processing_reclaim(
                reclaim_plan,
                next_job,
                process_with_model,
                device_index=dispatched_device_index,
            )
            if not should_dispatch:
                self._pending_line_skip = None
                self._post_processing_dispatch_defer = None
                return True

        self._post_processing_dispatch_defer = None

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
                        self._process_lifecycle._replace_inference_process(process_info)
                    process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
                    unloaded_any = True

        return unloaded_any

    def unload_from_ram(self, process_id: int) -> None:
        """Unload models from a process."""
        if process_id not in self._process_map:
            raise ValueError(f"process_id {process_id} is not in the process map")

        process_info = self._process_map[process_id]

        if process_info.process_type != HordeProcessType.INFERENCE:
            logger.warning(f"Process {process_id} is not an inference process, not unloading models")
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

        if len(self._job_tracker.jobs_pending_inference) == 0:
            return False

        if (
            self._max_concurrent_inference_processes == 1
            and len(bridge_data.image_models_to_load) == 1
            and not under_pressure
        ):
            return False

        wanted_models = self._compute_wanted_models()
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}

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

                self.unload_from_ram(process_info.process_id)
                return True

        return False

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
        self._restore_processes_after_ram_pressure()

        if not self.preload_models():
            next_job_and_process = await self.get_next_job_and_process(information_only=True)

            next_job_heavy_model_and_workflow = next_job_and_process is not None and self._is_heavy_model_and_workflow(
                next_job_and_process.next_job, stable_diffusion_reference
            )

            keep_single_inference, single_inf_reason = self._process_map.keep_single_inference(
                stable_diffusion_model_reference=stable_diffusion_reference,
                post_process_job_overlap=bridge_data.post_process_job_overlap,
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
