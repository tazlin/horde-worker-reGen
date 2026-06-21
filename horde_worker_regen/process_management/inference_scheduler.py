"""Schedules model preloading, inference start, and model unloading."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import IntEnum

import psutil
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger

from horde_worker_regen.consts import KNOWN_SLOW_WORKFLOWS, VRAM_HEAVY_MODELS
from horde_worker_regen.process_management.action_ledger import LedgerEventType
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.job_models import LineSkip, NextJobAndProcess
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.lru_cache import LRUCache
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeControlModelMessage,
    HordeInferenceControlMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.model_affinity import affinity_active, compute_protected_processes
from horde_worker_regen.process_management.model_metadata import ModelMetadata
from horde_worker_regen.process_management.performance_model import PerformanceModel, signature_from_job
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.resource_budget import (
    CommittedReserveLedger,
    RamBudget,
    StreamForecast,
    VramBudget,
    WholeCardResidencyState,
    forecast_weight_streaming,
    is_model_locally_unservable_for,
    predict_job_post_processing_vram_mb,
)
from horde_worker_regen.process_management.run_metrics import ChurnKind
from horde_worker_regen.process_management.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.worker_state import WorkerState
from horde_worker_regen.telemetry_spans import span_preload_model
from horde_worker_regen.utils.job_utils import (
    get_single_job_magnitude as _get_single_job_effective_megapixelsteps,
)

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

_HEAD_STARVATION_FORCE_ADMIT_SECONDS = 15.0
"""How long the head-of-queue job may be budget-deferred onto an otherwise-idle device before it is
force-admitted best-effort. Deliberately under the recovery supervisor's
``_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS`` (20s) so a self-inflicted budget-defer wedge resolves by
admitting one head onto an idle card, rather than tripping a save-our-ship soft reset that respawns
every pool and faults the whole backlog. Only runs while no live job holds the device, so it never
re-introduces the multi-process over-commit the budget guards against."""

_WHOLE_CARD_ESTABLISH_GRACE_SECONDS = 120.0
"""How long after a whole-card residency is established the worker may keep the queue intentionally held
(heavy head deferred while idle siblings stop, safety cycles off-GPU, and the model loads ~11GB) without
the recovery supervisor treating it as a structural wedge. The establishment is deliberately slow now
that it cycles the safety process, so the plain ``_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS`` (20s) window would
otherwise soft-reset the pools mid-setup. Bounded so a residency that genuinely never loads still trips
the supervisor."""

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

_SCHEDULER_DIAGNOSTIC_REPEAT_SECONDS = 30.0
"""Minimum cadence for unchanged high-frequency scheduler diagnostics.

These diagnostics are useful when reconstructing residency and performance behavior, but they sit inside
the scheduler's fast polling loop. Log immediately when the decision state changes, otherwise emit only
periodic reminders with a suppressed-repeat count.
"""

_SCHEDULER_DIAGNOSTIC_MB_BUCKET = 256.0
"""Bucket size for deciding whether memory telemetry changed enough to re-log a scheduler diagnostic."""


class _ModelSizeTier(IntEnum):
    """How much of the device a model's inference is expected to want, for concurrency decisions.

    Ordered so heavier tiers compare greater. ``LIGHT`` jobs (SD1.5/SD2) are cheap enough to sample
    side by side; ``HEAVY`` (SDXL) jobs need the job they would overlap to be well underway first; and
    ``EXTRA_LARGE`` (Cascade/Flux/Qwen/Z-Image and the named VRAM-heavy checkpoints) effectively want
    the whole card and never share it.
    """

    LIGHT = 0
    HEAVY = 1
    EXTRA_LARGE = 2


_LIGHT_BASELINE_VALUES = frozenset(
    {
        KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1.value,
        KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_512.value,
        KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_768.value,
    },
)
"""Baselines treated as light enough to thread together without headway."""

_HEAVY_BASELINE_VALUES = frozenset(
    {
        KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl.value,
    },
)
"""Baselines that need the job they overlap to be well underway before joining the card."""

_EXTRA_LARGE_BASELINE_VALUES = frozenset(
    {
        KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade.value,
        KNOWN_IMAGE_GENERATION_BASELINE.flux_1.value,
        KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell.value,
        KNOWN_IMAGE_GENERATION_BASELINE.flux_dev.value,
        KNOWN_IMAGE_GENERATION_BASELINE.qwen_image.value,
        KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo.value,
    },
)
"""Baselines that effectively want the whole card and never share it with a concurrent job."""

_OVERLAP_HEADWAY_MIXED_HEAVY = 0.5
"""Fraction of the in-flight job's sampling that must be done before a concurrent job joins it when
exactly one side of the overlap is heavy (e.g. an SDXL is running and a cheaper SD1.5 wants to join,
or vice versa). Gives the heavier job room to get past its memory-hungry startup before another
sampler adds pressure."""

_OVERLAP_HEADWAY_BOTH_HEAVY = 0.75
"""Fraction of the in-flight job's sampling that must be done before a *second heavy* job joins it.
Two SDXL jobs stacking their weight loads and activation peaks is the over-subscription that thrashes
a sampler into a watchdog teardown, so the running job must be most of the way done first."""


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
        self._scheduler_diagnostic_log_state = {}
        # One-shot log throttle, keyed by model, for the "held back as locally unservable" notice.
        self._unservable_admit_notified: dict[str, bool] = {}
        # Startup-measured per-process VRAM overhead (one torch/CUDA context, no model), set by the manager
        # via set_measured_per_process_overhead_mb. The streaming forecast subtracts it from total VRAM to
        # estimate the free achievable under sole residency. 0 until measured (free-if-alone == total then).
        # NB: this is the *first/sole* context cost (it includes the one-time, device-wide CUDA runtime
        # allocation), NOT the marginal cost of an additional sibling context -- see
        # _measured_idle_context_residency_mb below, from which the marginal is derived.
        self._measured_per_process_overhead_mb: float = 0.0
        # Startup-measured *marginal* VRAM cost of each additional sibling context (the probe's second-context
        # delta, set by the manager via set_measured_marginal_overhead_mb). This is hard data available from
        # the first scheduling tick, so it sizes free_after_model_evict correctly even in the startup window
        # before any sibling reaches idle. 0 until measured (or unmeasurable), where the scheduler falls back
        # to the idle-residency derivation below and then to the conservative overhead-per-context sizing.
        self._measured_marginal_overhead_mb: float = 0.0
        # Lowest device-wide *used* VRAM observed while every loaded inference process is idle with no model
        # resident (the clean all-contexts baseline, typically at startup). This is the true combined cost of
        # all process contexts -- the one-time CUDA runtime plus one context each -- so the marginal cost of an
        # additional context is (residency - per_process_overhead) / (count - 1). A runtime fallback for the
        # probe's direct marginal measurement: sizes free_after_model_evict from measurement instead of
        # multiplying the one-time cost by the process count. None until seen.
        self._measured_idle_context_residency_mb: float | None = None
        self._idle_residency_process_count: int = 0
        # Highest device-wide *used* VRAM observed while every loaded inference process is idle with no model
        # resident: the floor reclaim can never get below. The clean baseline above keeps the *minimum* on the
        # assumption a model's cache returns to the device when it unloads; when that assumption fails (the
        # allocator/runtime retains multi-GB per context, as a real inference context does once it has loaded a
        # checkpoint), the *effective* floor is the maximum, not the minimum. A probe measured against a minimal
        # holder under-counts this, so once the effective floor is known it supersedes the probe in deriving the
        # per-context marginal -- otherwise the forecast believes in reclaimable VRAM the device never returns
        # and routes every load into an evict-all admit. None until seen.
        self._measured_effective_idle_used_mb: float | None = None
        self._effective_idle_process_count: int = 0
        # Set while a whole-card model is being given sole residency by stopping idle sibling processes to
        # reclaim their CUDA contexts (a context is only freed by the process exiting). Holds the model name
        # so the teardown is logged once and the process count is restored once the exclusive job drains.
        self._sibling_teardown_for_model: str | None = None
        # Whole-card residency cooldown: when a whole-card residency is established it is held until this
        # wall-clock time even after the heavy job drains, so a burst of heavy jobs reuses one residency
        # instead of each churning a teardown/restore + safety cycle. Refreshed on each whole-card admit.
        self._whole_card_cooldown_until: float = 0.0
        # When the current whole-card residency was first established. The establishment (stop siblings,
        # cycle safety off-GPU, load ~11GB) intentionally holds the queue, which must not be mistaken for a
        # structural wedge until this grace elapses. 0.0 when no residency is establishing.
        self._whole_card_established_at: float = 0.0
        # When a whole-card residency was last restored (siblings respawned, safety cycled back on-GPU).
        # That churn also briefly makes the queue unservable, so the wedge grace must cover it too.
        self._whole_card_restore_at: float = 0.0
        # The streaming forecast that established the current whole-card residency, cached so the status
        # snapshot can show the hard numbers (weights, reserve, free-if-alone) without re-running it.
        # None when no residency is held.
        self._whole_card_forecast: StreamForecast | None = None
        # When a heavy head was last admitted best-effort off the whole-card path (the over-budget exclusive
        # branch, for a model that streams even alone). Its load equally holds the queue, so this bounds a
        # wedge grace that the whole-card establishment grace does not cover. 0.0 when none is loading.
        self._heavy_head_admitted_at: float = 0.0
        # Head-of-queue starvation backstop. Tracks the id of the job currently at the head of the queue
        # and when it first became budget-deferred onto an idle device, so a head that the budget gate
        # cannot fit (reclamation structurally exhausted) is force-admitted before the sustained-wedge
        # window trips the recovery supervisor. Reset when the head changes, a job dispatches, or a live
        # job takes the device. See _HEAD_STARVATION_FORCE_ADMIT_SECONDS.
        self._head_starvation_job_id: str | None = None
        self._head_starvation_since: float = 0.0

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
            forecast.fits_coresident,
            forecast.needs_exclusive_residency,
            forecast.requires_sibling_teardown,
            forecast.streams_unavoidably,
        )
        suppressed_count = self._scheduler_diagnostic_suppressed_count(f"stream_forecast:{job_id}", state_key)
        if suppressed_count is None:
            return

        logger.debug(
            f"Stream forecast for {job.model}: {forecast.reason()} "
            f"[free_now={forecast.free_now_mb}, after_model_evict={forecast.free_after_model_evict_mb}, "
            f"alone={forecast.free_if_alone_mb}, live_procs="
            f"{self._process_map.num_loaded_inference_processes()}, "
            f"overhead/proc={self._per_process_overhead_mb():.0f}MB] -> "
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
        vram_reserve = bridge_data.vram_reserve_mb
        ram_reserve = bridge_data.ram_reserve_mb
        numeric_reserves = (
            isinstance(vram_reserve, (int, float))
            and not isinstance(vram_reserve, bool)
            and isinstance(ram_reserve, (int, float))
            and not isinstance(ram_reserve, bool)
        )
        if not isinstance(enabled, bool) or not numeric_reserves:
            return False
        if not enabled:
            return False
        self._vram_budget.set_reserve_mb(float(vram_reserve))
        self._ram_budget.set_reserve_mb(float(ram_reserve))
        return True

    def _is_model_locally_unservable(self, model: str | None) -> bool:
        """Return whether ``model`` is held back as locally unservable (the shared breaker policy).

        Delegates to :func:`is_model_locally_unservable_for` so the scheduler's best-effort-admit gate and
        the popper's model selection apply one identical policy: a model held back here is neither
        best-effort-admitted nor popped, so the worker stops force-admitting and dropping a model the
        device genuinely cannot run.
        """
        return is_model_locally_unservable_for(self._runtime_config.bridge_data, self._job_tracker, model)

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

    def set_measured_per_process_overhead_mb(self, overhead_mb: int | float) -> None:
        """Record the startup-measured per-process VRAM overhead (MB) for the streaming forecast."""
        if isinstance(overhead_mb, (int, float)) and not isinstance(overhead_mb, bool) and overhead_mb >= 0:
            self._measured_per_process_overhead_mb = float(overhead_mb)

    def set_measured_marginal_overhead_mb(self, marginal_mb: int | float) -> None:
        """Record the startup-measured *marginal* per-additional-context VRAM cost (MB) from the probe.

        Hard data (the probe's second-context delta) available from the first scheduling tick, so it fixes the
        startup-window over-count without waiting for siblings to reach idle. 0 (or unmeasurable) leaves the
        scheduler on its idle-residency fallback.
        """
        if isinstance(marginal_mb, (int, float)) and not isinstance(marginal_mb, bool) and marginal_mb >= 0:
            self._measured_marginal_overhead_mb = float(marginal_mb)

    def _per_process_overhead_mb(self) -> float:
        """Return the per-process VRAM overhead (MB) to assume: configured override, else measured, else 0.

        An explicit ``vram_per_process_overhead_mb`` config value (> 0) wins so operators can tune; otherwise
        the startup-measured figure is used. Tolerant of partially-mocked config (non-numeric -> measured).
        This is the *first/sole* context cost (it includes the one-time CUDA runtime allocation), used to size
        ``free_if_alone``; the per-additional-context cost is :meth:`_marginal_process_overhead_mb`.
        """
        configured = self._runtime_config.bridge_data.vram_per_process_overhead_mb
        if isinstance(configured, (int, float)) and not isinstance(configured, bool) and configured > 0:
            return float(configured)
        return self._measured_per_process_overhead_mb

    def _maybe_capture_idle_context_residency(self) -> None:
        """Record the device-wide used VRAM when every inference process is idle with no model resident.

        That measurement is the true combined cost of all process contexts (the one-time CUDA runtime plus one
        context each), which the forecast needs to size ``free_after_model_evict`` without multiplying the
        one-time cost by the process count. The clean window is at startup, before any model loads; once a
        model has loaded this rarely holds again, so the minimum observed value is kept (later, cache-dirtied
        observations read higher and are ignored). Cheap and side-effect-free beyond the cached figure, so it
        is safe to call every scheduling tick.
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
        if self._measured_idle_context_residency_mb is None or used_mb < self._measured_idle_context_residency_mb:
            self._measured_idle_context_residency_mb = used_mb
            self._idle_residency_process_count = process_count
        # The effective floor is the *worst* (highest) fully-idle, fully-evicted reading: the VRAM reclaim
        # provably cannot return. Kept per the live context count so a later, fewer-process reading does not
        # mask an earlier over-commit.
        if (
            self._measured_effective_idle_used_mb is None
            or process_count > self._effective_idle_process_count
            or (process_count == self._effective_idle_process_count and used_mb > self._measured_effective_idle_used_mb)
        ):
            self._measured_effective_idle_used_mb = used_mb
            self._effective_idle_process_count = process_count

    def _marginal_process_overhead_mb(self) -> float | None:
        """Return the per-additional-context VRAM cost (MB), or None to fall back to the first-context overhead.

        Prefers the probe's directly-measured second-context delta (hard data, available from the first tick,
        so it also covers the startup window where siblings have not yet reached idle). Failing that (the probe
        could not measure it on this backend), derives it from the measured all-contexts idle residency:
        ``residency = per_process_overhead + (count - 1) * marginal``, so ``marginal = (residency -
        per_process_overhead) / (count - 1)``. Returns None when neither is available -- no probe delta, and no
        clean idle baseline (or only one process up, or a residency at/below the first-context overhead, an
        inconsistent reading) -- in which case the forecast conservatively reuses the first-context overhead
        per additional context.
        """
        per_process = self._per_process_overhead_mb()

        def _derive(residency: float | None, count: int) -> float | None:
            if residency is None or count < 2 or per_process <= 0 or residency <= per_process:
                return None
            return (residency - per_process) / (count - 1)

        # The measured *effective* floor is ground truth for what reclaim achieves: when a real inference
        # context retains more cache than the probe's minimal holder allocated, the probe under-counts the
        # marginal and the forecast over-counts reclaimable VRAM. Take the larger of the probe estimate and the
        # effective-floor derivation so the forecast never believes in headroom the device will not return; the
        # effective floor only rises above the probe once contexts genuinely over-commit (the threads>1 regime),
        # so a roomy card keeps the probe estimate unchanged.
        effective_derived = _derive(self._measured_effective_idle_used_mb, self._effective_idle_process_count)
        probe = self._measured_marginal_overhead_mb if self._measured_marginal_overhead_mb > 0 else None
        candidates = [c for c in (probe, effective_derived) if c is not None]
        if candidates:
            return max(candidates)
        # No probe and no effective floor: fall back to the clean idle-residency derivation (startup path).
        return _derive(self._measured_idle_context_residency_mb, self._idle_residency_process_count)

    def _whole_card_residency_enabled(self) -> bool:
        """Whether preventative whole-card exclusive residency is on (config, tolerant of mocked config)."""
        enabled = self._runtime_config.bridge_data.whole_card_exclusive_residency
        return enabled is True

    def _max_coresident_for_peak_mb(self, peak_mb: float, reserve_mb: float) -> int | None:
        """Largest live inference-process count that still fits ``peak_mb`` plus ``reserve_mb``.

        Sizes the context-reduction depth from the *same* conservative figure the VRAM verdict rejects on
        (the burden estimate), not the forecast's resident-weight estimate. The two estimators differ -- the
        forecast judges co-residence from the resident weight footprint while the admission verdict uses the
        fuller per-job burden peak -- so a moderate head can read co-resident in the forecast yet be rejected
        by the verdict every tick, the gap that routes it into the evict-all admit. Reasoning the teardown
        depth from the verdict's own peak makes the structural remedy fire exactly when admission would
        otherwise reject and thrash. The loader's first context costs the full one-time overhead; each
        additional co-resident context costs only the marginal. Returns None when it cannot be sized.
        """
        total_mb = self._process_map.get_reported_total_vram_mb()
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
    ) -> StreamForecast:
        """Return the weight-streaming forecast for loading ``job``'s model given the device's measured state.

        Combines the measured free VRAM and total VRAM (from the children's reports), the configured reserve
        floor, and the per-process overhead so the scheduler can tell a model that only streams because of
        co-resident siblings (curable by exclusive residency) from one that streams even alone.
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
        num_processes = self._process_map.num_loaded_inference_processes()
        # The safety process holds its own CUDA context on the card when safety_on_gpu is set; that VRAM is
        # not reclaimable by stopping idle inference siblings, so the forecast must count it against the
        # achievable-free figures (sole residency for a heavy model then implies moving safety off-GPU too).
        # Count the safety context only when safety is *actually* on the GPU right now: once a whole-card
        # job has paused it off-GPU, its context is freed, so continuing to charge it would keep the
        # structural floor (free_after_model_evict) below the model's demand forever and the whole-card
        # branch would defer the model every tick without ever loading it.
        safety_on_gpu = self._runtime_config.bridge_data.safety_on_gpu and (
            not self._process_lifecycle.is_safety_gpu_paused
        )
        num_safety_contexts = self._process_map.num_safety_processes() if safety_on_gpu else 0
        # Refresh the clean all-contexts idle baseline (a no-op once startup has passed) so the marginal
        # per-context cost reflects measurement rather than the one-time-cost-times-N over-count.
        self._maybe_capture_idle_context_residency()
        return forecast_weight_streaming(
            job,
            str(baseline) if baseline is not None else None,
            free_now_mb=self._measured_free_vram_mb(),
            total_vram_mb=self._process_map.get_reported_total_vram_mb(),
            per_process_overhead_mb=self._per_process_overhead_mb(),
            num_inference_processes=num_processes,
            configured_reserve_floor_mb=floor_mb,
            num_extra_resident_contexts=num_safety_contexts,
            post_processing_reserve_mb=self._committed_vram_reserve_mb(),
            marginal_process_overhead_mb=self._marginal_process_overhead_mb(),
        )

    def _establish_whole_card_residency(
        self,
        job: ImageGenerateJobPopResponse,
        forecast: StreamForecast,
        *,
        announce: bool,
        target_override: int | None = None,
    ) -> None:
        """Claim the device for a whole-card model: stop idle siblings and move safety off-GPU.

        The siblings' fixed per-process CUDA contexts (not their models) over-commit the device, and a context
        is only reclaimed by the process exiting (``torch.cuda.empty_cache`` returns cached blocks but never a
        context). Reduce the live inference-process count to the largest that still leaves room for this model's
        weights plus its activation reserve, and -- on the very edge (Flux on a 16GB card) -- also move the
        safety process off-GPU so its context is freed too. The model is remembered so the residency is held
        and then restored once its job drains (after the configured cooldown). Only idle inference processes
        are stopped; a busy sibling is left to finish its job.
        """
        if announce or self._whole_card_established_at == 0.0:
            # Mark the establishment start (first admit of this heavy job, or a fresh residency) so the
            # recovery supervisor's grace window is measured from when the intentional hold began.
            self._whole_card_established_at = time.time()
        self._sibling_teardown_for_model = job.model
        self._whole_card_forecast = forecast
        self._whole_card_cooldown_until = time.time() + self._whole_card_cooldown_seconds()

        # ``target_override`` lets a caller size the depth from the admission verdict's rejected peak rather
        # than the forecast's lighter resident-weight estimate, for the activation-peak context over-commit the
        # weight-based gates leave co-resident.
        target = target_override if target_override is not None else (forecast.max_resident_processes() or 1)
        current = self._process_map.num_loaded_inference_processes()
        after = current
        if target < current:
            after = self._process_lifecycle.scale_inference_processes(target)

        safety_paused = False
        if self._whole_card_safety_off_gpu_enabled():
            safety_paused = self._process_lifecycle.pause_safety_on_gpu()

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
        - system RAM can hold the head's *weights* alongside the in-flight job -- the operator's "assuming the
          RAM can support it" (see :meth:`_prestage_weights_fit_ram`). A RAM shortfall falls back to the prior
          claim-the-card-and-wait behavior.
        """
        if len(self._job_tracker.jobs_in_progress) == 0:
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
    ) -> None:
        """Record a whole-card residency for a head being pre-staged into RAM, without claiming the card yet.

        The device cannot be claimed while a live job holds it, but the heavy head's weights can load into a
        spare's RAM now. This sets the same residency bookkeeping :meth:`_establish_whole_card_residency` does
        (so the cooldown, the restore, and the recovery-supervisor wedge grace all cover the pre-stage load and
        the convergence that follows), minus the process teardown and safety pause -- those are deferred to
        :meth:`_converge_whole_card_residency`, which runs once the head is staged and the device frees.
        """
        if announce or self._whole_card_established_at == 0.0:
            self._whole_card_established_at = time.time()
        self._sibling_teardown_for_model = job.model
        self._whole_card_forecast = forecast
        self._whole_card_cooldown_until = time.time() + self._whole_card_cooldown_seconds()
        if announce:
            logger.opt(ansi=True).info(
                f"<fg #f0beff>Pre-staging whole-card head {job.model} into a spare process's RAM while the "
                f"in-flight job finishes; the device will be reserved for it (idle siblings stopped"
                f"{' and safety moved off-GPU' if self._whole_card_safety_off_gpu_enabled() else ''}) once it "
                f"frees, so its weights are loaded before it samples instead of after. "
                f"{self._process_map.residency_snapshot()}</>",
            )

    def _converge_whole_card_residency(self) -> None:
        """Collapse an in-progress whole-card residency to sole VRAM residency once its model is staged.

        Driven each scheduling cycle while a residency is held. A pre-staged head is loaded into RAM before
        the device is claimed (see :meth:`_begin_whole_card_residency`); stopping idle siblings before the head
        is actually resident on a process could kill the very spare the pre-stage wants to use, so this waits
        until the head is resident or loading on a process. From then on that process is the queued-model holder
        ``scale_inference_processes`` spares, so reducing the live inference-process count to the forecast's
        target stops the *other* idle siblings (and the former busy process once its job drains) to reclaim
        their CUDA contexts, and safety is moved off-GPU, leaving the staged head the whole card when it samples.
        A no-op until a residency is held and its model is staged; idempotent at the target.
        """
        model = self._sibling_teardown_for_model
        if model is None:
            return
        if not self._is_model_forecast_to_load(model):
            return
        forecast = self._whole_card_forecast
        target = (forecast.max_resident_processes() or 1) if forecast is not None else 1
        if self._process_map.num_loaded_inference_processes() > target:
            self._process_lifecycle.scale_inference_processes(target)
        if self._whole_card_safety_off_gpu_enabled() and not self._process_lifecycle.is_safety_gpu_paused:
            self._process_lifecycle.pause_safety_on_gpu()

    def _prestaged_whole_card_not_ready(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether ``job`` must wait for its in-progress whole-card residency to claim the card before sampling.

        A pre-staged whole-card head is loaded into RAM (see :meth:`_begin_whole_card_residency`) before the
        device is reserved, so dispatching it would commit its weights to VRAM while idle siblings (or the
        just-drained busy process) still hold their CUDA contexts, forcing the first step to stream. This
        returns True until the residency has converged -- the live inference-process count is at the forecast's
        target, safety is off-GPU if this residency needs it, and the card has drained enough to load the
        weights (the same :meth:`_whole_card_teardown_exhausted` gate the non-pre-staged path loads under).

        Returns False for any job that is not the currently-held residency's model, so ordinary dispatch -- and
        the non-pre-staged whole-card path, which only preloads once already at sole residency -- is unaffected.
        """
        model = self._sibling_teardown_for_model
        if model is None or job.model != model:
            return False
        baseline = self._model_metadata.get_baseline(job.model) if job.model is not None else None
        forecast = self._forecast_streaming(job, baseline)
        return not self._whole_card_teardown_exhausted(forecast)

    def _whole_card_teardown_exhausted(self, forecast: StreamForecast) -> bool:
        """Whether a whole-card residency has done all it can and the head can now load best-effort.

        The whole-card branch defers a heavy head while a teardown can still make room: idle siblings left to
        stop, the safety process still on-GPU, or their freed VRAM still draining. This returns True only once
        none of those remain: the live inference-process count is already at (or below) the forecast's target,
        safety is off-GPU if this residency needs it, and the device has drained enough that the weights will
        load now (``fits_weights_now``). A model that reaches this state but still cannot fit co-resident (its
        activation peak overflows even sole residency) can never converge by deferring, so the scheduler loads
        it onto the cleared card best-effort and lets it sample slowly under the over-budget step grace rather
        than wedging the queue until the recovery supervisor soft-resets the pools.
        """
        target = forecast.max_resident_processes() or 1
        if self._process_map.num_loaded_inference_processes() > target:
            return False
        if self._whole_card_safety_off_gpu_enabled() and not self._process_lifecycle.is_safety_gpu_paused:
            return False
        return forecast.fits_weights_now

    def whole_card_residency_grace_active(self) -> bool:
        """Whether a whole-card residency is establishing, so the held queue is intentional (not a wedge).

        While true, the recovery supervisor must not treat the deliberately-deferred heavy head (waiting
        for idle siblings to stop, the safety process to cycle off-GPU, and ~11GB of weights to load) as a
        structural queue wedge and soft-reset the pools mid-setup. Bounded by
        ``_WHOLE_CARD_ESTABLISH_GRACE_SECONDS`` so a residency that genuinely never loads still trips the
        supervisor. Public: read by the process manager's wedge assessment.
        """
        now = time.time()
        establishing = (
            self._sibling_teardown_for_model is not None
            and self._whole_card_established_at != 0.0
            and (now - self._whole_card_established_at) < _WHOLE_CARD_ESTABLISH_GRACE_SECONDS
        )
        restoring = (
            self._whole_card_restore_at != 0.0
            and (now - self._whole_card_restore_at) < _WHOLE_CARD_RESTORE_GRACE_SECONDS
        )
        return establishing or restoring

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

        model = self._sibling_teardown_for_model
        active = model is not None
        forecast = self._whole_card_forecast
        now = time.time()

        phase = ""
        cooldown_remaining: float | None = None
        processes_target = 0
        weights_mb = reserve_mb = free_now_mb = free_if_alone_mb = None
        max_resident_processes: int | None = None
        if active:
            establishing = (
                self._whole_card_established_at != 0.0
                and (now - self._whole_card_established_at) < _WHOLE_CARD_ESTABLISH_GRACE_SECONDS
            )
            phase = "establishing" if establishing else "holding"
            cooldown_remaining = max(0.0, self._whole_card_cooldown_until - now)
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
        the safety process. Once neither condition holds, sibling processes are grown back to the ceiling and
        the safety process is restored to the GPU. A no-op when no residency is outstanding.
        """
        model = self._sibling_teardown_for_model
        if model is None:
            return
        active_models = {j.model for j in self._job_tracker.jobs_in_progress}
        active_models.update(j.model for j in self._job_tracker.jobs_pending_inference)
        if model in active_models or self._job_tracker.has_exclusive_job_in_progress():
            # Still serving the residency; keep it (and refresh the cooldown so it survives the lull between
            # back-to-back heavy jobs).
            self._whole_card_cooldown_until = time.time() + self._whole_card_cooldown_seconds()
            return
        if time.time() < self._whole_card_cooldown_until:
            # Drained, but hold the residency through the cooldown so an imminent heavy job reuses it.
            return
        self._sibling_teardown_for_model = None
        self._whole_card_established_at = 0.0
        self._whole_card_forecast = None
        # The restore's own churn (respawning siblings, cycling safety back on-GPU) briefly makes the queue
        # unservable; mark its start so the wedge grace covers it (see _WHOLE_CARD_RESTORE_GRACE_SECONDS).
        self._whole_card_restore_at = time.time()
        safety_restored = self._process_lifecycle.restore_safety_on_gpu()
        current = self._process_map.num_loaded_inference_processes()
        if current >= self._max_inference_processes and not safety_restored:
            return
        after = self._process_lifecycle.scale_inference_processes(self._max_inference_processes)
        safety_note = " and restoring safety to the GPU" if safety_restored else ""
        logger.opt(ansi=True).info(
            f"<fg #7b7d7d>Whole-card residency for {model} complete; restoring inference processes "
            f"({current} -> {after} of {self._max_inference_processes}){safety_note}.</>",
        )

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

    def _measured_free_vram_mb(self) -> float | None:
        """Return the most conservative measured device-wide free VRAM (MB), or None when not yet reported.

        Sourced from the inference processes' VRAM reports via :meth:`ProcessMap.get_free_vram_mb`, which
        the children compute through hordelib's backend-agnostic accelerator layer (comfy /
        ``torch.cuda.mem_get_info``, accurate and not NVIDIA-specific). The parent stays free of any direct
        GPU query, so this works on every backend the execution layer supports.
        """
        return self._process_map.get_free_vram_mb()

    def _measured_available_ram_mb(self) -> float:
        """The measured system-wide available RAM (MB), read live in the parent process."""
        return psutil.virtual_memory().available / (1024 * 1024)

    def _committed_post_processing_reserve_mb(self) -> float:
        """Sum the imminent post-processing-phase VRAM peaks of jobs currently in post-processing (MB).

        When a job finishes sampling it releases its inference slot for overlap *before* its
        upscaler/face-fixer allocates, so the measured free-VRAM figure still reads as if that headroom
        were available. This derives the VRAM that is therefore spoken-for but not yet realised: for each
        inference process in ``INFERENCE_POST_PROCESSING``, the predicted post-processing peak of the job it
        is running. Subtracted from measured free VRAM at the dispatch/overlap gates and folded into the
        residency forecast, it stops a freshly-released slot being handed VRAM an in-flight job is about to
        claim. Returns 0.0 when nothing is post-processing (so the reserve self-scales away on roomy cards)
        or when the feature is disabled. Never raises: a bad per-job estimate is skipped.
        """
        if not self._budget_active() or not self._runtime_config.bridge_data.post_processing_budget_reserve_enabled:
            return 0.0

        total_mb = 0.0
        jobs_in_progress = self._job_tracker.jobs_in_progress
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.last_process_state != HordeProcessState.INFERENCE_POST_PROCESSING:
                continue
            job = process_info.last_job_referenced
            # A post-processing process's job is still mid-inference in the tracker (the result that ends
            # the inference stage only arrives once post-processing completes). Guard against a stale
            # reference that has already left that stage.
            if job is None or job not in jobs_in_progress:
                continue
            baseline = self._model_metadata.get_baseline(job.model) if job.model is not None else None
            peak_mb = predict_job_post_processing_vram_mb(job, str(baseline) if baseline is not None else None)
            if peak_mb is not None:
                total_mb += peak_mb
        return total_mb

    _IMAGE_PP_RESERVE_FLOW = "image_post_processing"
    """The shared-ledger flow namespace under which this scheduler registers its post-processing reserve."""

    def _committed_vram_reserve_mb(self) -> float:
        """Return the combined committed VRAM (MB) across every flow, refreshing this scheduler's own entry.

        Each call re-derives the image-generation post-processing reserve from live process state and
        publishes it into the shared :class:`CommittedReserveLedger` as a single aggregate entry, then
        returns the ledger total (which also includes any in-flight alchemy forms other flows registered).
        Re-publishing every call keeps the entry self-healing: it always reflects current state and never
        leaks. This combined figure is what the VRAM admission and residency-forecast gates subtract, so a
        freshly-released slot is not handed VRAM that image post-processing *or* concurrent alchemy is about
        to claim.
        """
        self._reserve_ledger.set(
            self._IMAGE_PP_RESERVE_FLOW,
            "aggregate",
            vram_mb=self._committed_post_processing_reserve_mb(),
        )
        return self._reserve_ledger.total_vram_mb()

    def _max_jobs_in_progress_allowed(self, processes_post_processing: int) -> int:
        """The cap on concurrently in-progress jobs for this scheduling decision.

        Without the GPU sampling lease, the inference semaphore is the sole denoise gate, so this
        is the concurrent-sampling count — dispatching more would over-subscribe the GPU. With the
        lease enabled, the lease (not this cap) limits actual concurrent sampling, so spare
        inference processes are allowed to receive jobs and stage their pipeline (model load,
        prompt encode) *ahead* while others sample — filling the inter-job gaps where the GPU
        would otherwise go dark. That pre-staging is permitted up to the full inference-process
        count, but only while there is enough free VRAM to hold another staged model; otherwise it
        falls back to the sampling-slot cap so speculation never over-commits the device.
        """
        # An exclusively-admitted over-budget job needs the whole device; never dispatch another job
        # alongside it. Returning the current in-progress count (floored at 1 so the exclusive job itself
        # can still be dispatched when none is yet running) blocks any *additional* concurrent dispatch.
        if self._job_tracker.has_exclusive_job_in_progress():
            return max(1, len(self._job_tracker.jobs_in_progress))

        # The post-processing overlap bump raises the cap *because* a process is post-processing -- but that
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
                if (free_vram_mb - self._committed_vram_reserve_mb()) < bump_floor:
                    post_processing_bump = 0

        base = self._max_concurrent_inference_processes + post_processing_bump
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

        return self._max_inference_processes + post_processing_bump

    def _model_size_tier(self, model_name: str | None) -> _ModelSizeTier:
        """Classify a model by how much of the device its inference is expected to want.

        A model named in the VRAM-heavy list, or carrying an extra-large baseline, wants the whole card.
        SDXL is heavy; SD1.5/SD2 are light. An unknown baseline (no loaded reference for the model)
        falls back to light so the overlap gate stays permissive rather than starving dispatch on
        missing metadata; a genuinely large model is always classified by its known baseline.
        """
        if model_name is not None and model_name in VRAM_HEAVY_MODELS:
            return _ModelSizeTier.EXTRA_LARGE

        baseline = self._model_metadata.get_baseline(model_name) if model_name is not None else None
        baseline_value = baseline.value if isinstance(baseline, KNOWN_IMAGE_GENERATION_BASELINE) else baseline
        if baseline_value in _EXTRA_LARGE_BASELINE_VALUES:
            return _ModelSizeTier.EXTRA_LARGE
        if baseline_value in _HEAVY_BASELINE_VALUES:
            return _ModelSizeTier.HEAVY
        return _ModelSizeTier.LIGHT

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

    def _concurrent_overlap_allowed(self, candidate_job: ImageGenerateJobPopResponse) -> bool:
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
        """
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
            self._record_churn("process_cycle")
            return True

        return False

    def preload_models(self) -> bool:
        """Preload models that are likely to be used soon.

        Returns:
            True if a model was preloaded, False otherwise.
        """
        bridge_data = self._runtime_config.bridge_data
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
        # model (see the budget-defer branches below); a later job whose turn has not come never
        # displaces a resident head.
        in_progress_jobs = self._job_tracker.jobs_in_progress
        head_job = next((j for j in self._job_tracker.jobs_pending_inference if j not in in_progress_jobs), None)
        self._update_head_starvation_timer(head_job)

        for job in self._job_tracker.jobs_pending_inference:
            if job.model is None:
                raise ValueError(f"job.model is None ({job})")

            if job.model in loaded_models:
                continue

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
            # model's only copy, forcing it to disk-reload on its next job — the dominant
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

            available_process = self._process_map.get_first_available_inference_process(
                disallowed_processes=processes_with_model_for_queued_job,
            )

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

            num_preloading_processes = self._process_map.num_preloading_processes()

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
            if self._budget_active():
                baseline = self._model_metadata.get_baseline(job.model)
                # A single model loaded onto an otherwise-idle GPU cannot reintroduce the multi-process
                # over-commit the budget guards against; the over-commit case is several *concurrent*
                # resident models. So when no job is in-flight (holding the device), a starved head may be
                # admitted best-effort rather than deferred forever (see the branches below).
                no_live_resource_consumer = len(self._job_tracker.jobs_in_progress) == 0

                # Whole-card exclusive residency (preventative): forecast whether loading this model alongside
                # the currently-resident models would drive the device into weight streaming. A heavy model
                # (such as a large diffusion checkpoint) loaded while other models stay resident across sibling
                # processes can collapse free VRAM to near zero, at which point either ComfyUI offloads weights
                # or, once free VRAM nears zero, the GPU driver's system-memory fallback spills the per-step
                # activations; both stream over the bus, sampling slows by several times, and the slow job
                # risks being mistaken for a hang and killed. When the model would stream co-resident but fits
                # with the card to itself, give it sole residency *before* it loads: mark it exclusive (the
                # has_exclusive_job_in_progress hook then suppresses any other model's staging/dispatch), then
                # free enough VRAM and defer until the device recovers. The forecast distinguishes two
                # remedies, applying the least-disruptive one that works:
                #   - evicting sibling *models* is enough -> just unload them (their processes stay up);
                #   - the siblings' fixed per-process contexts (~1 GB each, the cost of importing torch) are
                #     themselves the over-commit -> stop idle sibling processes, because a context is only
                #     reclaimed by the process exiting, never by emptying the allocator cache.
                # A live sibling job is never disturbed (for_head_of_queue spares live work, and only idle
                # processes are stopped), so the model simply waits for the device to drain. Models that would
                # stream even with the whole card to themselves (streams_unavoidably) fall through to the
                # best-effort admit below.
                forecast = self._forecast_streaming(job, baseline)
                # Trace the forecast for every budget-gated load so the logs show the residency dynamics
                # (the numbers behind a stream/no-stream decision), not just the action taken. Kept at DEBUG
                # because it can be per-pending-job per-tick; unchanged observations are coalesced by
                # _log_stream_forecast, while decision or headroom changes still log immediately.
                self._log_stream_forecast(job, forecast)
                whole_card_terminal = False
                early_prestage = False
                # A model needs the teardown path either because it is weight-dominant (needs sole residency)
                # or because the live sibling *process contexts* have squeezed its bounded weights off the card
                # though it co-resides once the process count is reduced (needs_process_count_reduction -- the
                # soak's blind spot, where a moderate SDXL head was deferred until the starvation backstop
                # force-admitted it into an OOM). Both are served by the same machinery: establish residency,
                # stop idle siblings down to max_resident_processes (sole residency for the former, a partial
                # reduction for the latter), and admit once the weights fit. The process-count-reduction case is
                # held exclusive through the teardown so a second model cannot re-fill the card mid-reduction;
                # the residency cooldown then restores concurrency.
                needs_teardown_path = forecast.needs_exclusive_residency or forecast.needs_process_count_reduction
                if self._whole_card_residency_enabled() and needs_teardown_path:
                    first_time = not self._job_tracker.is_admitted_exclusive(job)
                    self._job_tracker.mark_admitted_exclusive(job)
                    if self._should_prestage_whole_card_head(job, baseline, forecast, available_process):
                        # A live job still holds the device, so the card cannot be claimed yet, but the heavy
                        # head's weights can begin loading into a spare process's RAM right now: preload_model is
                        # a RAM-only load (the weights move to VRAM at sampling time), so it does not contend with
                        # the in-flight job's VRAM. Record the residency and fall through to the preload send;
                        # _converge_whole_card_residency then collapses the live process count to sole VRAM
                        # residency (stopping idle siblings, then the former busy process once its job drains)
                        # before the staged model samples. RAM-gated so a second multi-GB checkpoint is never
                        # forced onto a RAM-pressured host. This is the load-ASAP win: the heavy disk->RAM load
                        # overlaps the in-flight job instead of waiting for the device to drain first.
                        self._begin_whole_card_residency(job, forecast, announce=first_time)
                        early_prestage = True
                    else:
                        # Claim the device: stop idle siblings to the model's max-resident count and, on the very
                        # edge, move safety off-GPU too. Announces (once) why, for the operator. Held through the
                        # cooldown so a burst of heavy jobs reuses one residency instead of churning per job.
                        self._establish_whole_card_residency(job, forecast, announce=first_time)
                        # Evict the idle resident models on the *other* processes (sparing the slot that will load
                        # this model, and never a live in-progress model) so their VRAM returns to the driver. A
                        # live sibling is left to drain; the preload simply waits until the device is clear.
                        self.unload_models_from_vram(available_process, under_pressure=True, for_head_of_queue=True)
                        if not self._whole_card_teardown_exhausted(forecast):
                            # Still tearing down idle siblings, cycling safety off-GPU, or waiting for their freed
                            # VRAM to drain: defer and let a later tick re-evaluate against the reduced topology.
                            return False
                        # Teardown is structurally exhausted (already at the target process count, safety settled)
                        # yet the activation-inclusive peak still overflows even sole residency, so co-residency
                        # can never be reached. The weights DO fit alone (fits_weights_now), so fall through to the
                        # best-effort admit and load onto the now-cleared card, where it samples slowly under the
                        # over-budget step grace, instead of deferring every tick until the supervisor soft-resets.
                        whole_card_terminal = True

                # Head-of-queue starvation backstop (see _HEAD_STARVATION_FORCE_ADMIT_SECONDS): once the head
                # has been budget-deferred on an otherwise-idle device past the wedge horizon, stop deferring
                # and admit it best-effort. Reclamation is structurally exhausted by then (an earlier tick
                # would have freed room otherwise), so continuing to defer only wedges the queue until the
                # recovery supervisor soft-resets every pool and faults the whole backlog: strictly worse than
                # loading one head onto an idle card. A whole-card head reaches the shared admit only once its
                # teardown is exhausted (``whole_card_terminal`` above); this timer-based backstop rescues a
                # plain over-budget head that the verdicts keep rejecting (e.g. a head failing the RAM budget
                # against allocator-stranded idle RAM that no reclaim path can return).
                force_admit_starved_head = (
                    not early_prestage
                    and is_head_blocker
                    and self._head_starved_seconds(job) >= _HEAD_STARVATION_FORCE_ADMIT_SECONDS
                )
                if early_prestage:
                    # A RAM-only pre-stage of a whole-card head: the VRAM budget deliberately does not fit it
                    # co-resident (that is *why* it gets the whole card), so skip the VRAM/RAM verdict and fall
                    # through to the preload send. The RAM fit was already checked in the pre-stage decision.
                    pass
                elif whole_card_terminal or force_admit_starved_head:
                    if whole_card_terminal:
                        self._log_whole_card_terminal_admit(job)
                    else:
                        self._log_head_starvation_force_admit(job)
                    if not self._job_tracker.is_admitted_over_budget(job):
                        # First admit of this heavy head: open the bounded load grace so the supervisor does
                        # not read its multi-gigabyte load as a structural wedge (see heavy_head_load_grace_active).
                        self._heavy_head_admitted_at = time.time()
                    self._job_tracker.mark_admitted_over_budget(job)
                    if self._runtime_config.bridge_data.overbudget_exclusive_mode:
                        self._job_tracker.mark_admitted_exclusive(job)
                else:
                    vram_verdict = self._vram_budget.check_job(
                        job,
                        baseline,
                        self._measured_free_vram_mb(),
                        committed_reserve_mb=self._committed_vram_reserve_mb(),
                    )
                    if not vram_verdict.fits:
                        if not self._vram_budget_defer_notified:
                            logger.opt(ansi=True).warning(
                                f"<fg #f0beff>VRAM budget deferring preload of {job.model}: {vram_verdict.reason()}. "
                                "Reclaiming idle VRAM.</>",
                            )
                            self._vram_budget_defer_notified = True
                        freed = self.unload_models_from_vram(available_process, under_pressure=True)
                        if not freed and is_head_blocker:
                            # Gentle reclaim found nothing to free because every idle resident copy is
                            # another queued job's model. The head of the queue must still make progress,
                            # so escalate and reclaim one of them to give the head room.
                            freed = self.unload_models_from_vram(
                                available_process,
                                under_pressure=True,
                                for_head_of_queue=True,
                            )
                        # Reclamation is exhausted when nothing more could be freed: the predicted peak + reserve
                        # exceeds achievable free VRAM even with every idle resident copy evicted. The burden
                        # estimate is a deliberately conservative single-resident-peak figure, but a large
                        # combined checkpoint (text encoder + diffusion weights + VAE in one file) is streamed
                        # through VRAM component-by-component by the backend, so its true peak is the largest
                        # single component, well under the summed estimate. A head-of-queue job must therefore
                        # be given the device rather than deferred forever (which would wedge the queue and
                        # fault the head anyway). Admit it best-effort when no live job holds the device, after
                        # also reclaiming system RAM from idle residents: a heavy head loads its checkpoint
                        # through RAM first, so admitting it onto a RAM-pressured host is a likely load-time
                        # fault. Tag it so a crash/hang of its over-committed slot is classified as a resource
                        # failure (earning the bounded, isolated retry) instead of a plain re-dispatch onto
                        # another equally over-committed slot.
                        if not (is_head_blocker and not freed and no_live_resource_consumer):
                            return False

                        # Circuit-breaker: a model the device genuinely cannot run faults every over-budget
                        # attempt no matter how it is isolated. Once its consecutive-fault streak crosses the
                        # configured threshold it is held back (not admitted here, not popped in the popper) for
                        # a cooldown, so the worker stops dropping jobs faster than the horde server tolerates
                        # and is never forced into maintenance. The self-throttle backstop catches the aggregate.
                        if self._is_model_locally_unservable(job.model):
                            if not self._unservable_admit_notified.get(job.model, False):
                                logger.opt(ansi=True).warning(
                                    f"<fg #ff8c69>Model {job.model} keeps faulting over the VRAM budget; held "
                                    f"back as locally unservable and not admitted. "
                                    f"{self._process_map.residency_snapshot()}</>",
                                )
                                self._unservable_admit_notified[job.model] = True
                            return False
                        self._unservable_admit_notified.pop(job.model, None)

                        # Before evicting every resident model and admitting the head exclusively, check whether
                        # the live process *contexts* are the over-commit rather than the resident models. The
                        # weight-based teardown gates above leave a moderate head co-resident because its weights
                        # fit after a model eviction, but its activation peak does not fit while this many
                        # contexts are live (the threads>1 regime, where each extra context retains VRAM the
                        # allocator never returns). When more inference contexts are live than the head's
                        # weights-plus-reserve can co-reside with (``max_resident_processes``, sized from the
                        # measured per-context cost), reducing the process count is the structural remedy: it
                        # returns a context's retained VRAM so the head and a sibling model co-reside and
                        # pipeline. Evicting every model and loading exclusively instead strands the models the
                        # next jobs reuse and churns a full reload per job. Only idle siblings are stopped (the
                        # all-idle branch we are in), and the residency cooldown restores concurrency afterwards.
                        # The depth is sized from the verdict's own rejected peak (not the forecast's lighter
                        # resident-weight estimate) so the reduction fires exactly when admission would reject.
                        max_resident = None
                        if vram_verdict.predicted_mb is not None:
                            max_resident = self._max_coresident_for_peak_mb(
                                vram_verdict.predicted_mb,
                                vram_verdict.reserve_mb,
                            )
                        if (
                            self._whole_card_residency_enabled()
                            and max_resident is not None
                            and self._process_map.num_loaded_inference_processes() > max_resident
                        ):
                            first_time = not self._job_tracker.is_admitted_exclusive(job)
                            self._job_tracker.mark_admitted_exclusive(job)
                            self._establish_whole_card_residency(
                                job,
                                forecast,
                                announce=first_time,
                                target_override=max_resident,
                            )
                            self.unload_models_from_vram(
                                available_process,
                                under_pressure=True,
                                for_head_of_queue=True,
                            )
                            return False

                        self.unload_models(under_pressure=True, for_head_of_queue=True)
                        if not self._job_tracker.is_admitted_over_budget(job):
                            # First admit of this heavy head: open the bounded load grace (see
                            # heavy_head_load_grace_active) so its load is not mistaken for a structural wedge.
                            self._heavy_head_admitted_at = time.time()
                        self._job_tracker.mark_admitted_over_budget(job)
                        # Exclusive-first: contention arises when a *second* process loads another model while
                        # the over-budget job samples, pushing free VRAM to ~0 and spilling its weights to
                        # system RAM. Flag the job exclusive so the scheduler suppresses concurrent pre-staging
                        # and dispatch for its duration, leaving the device un-contended so it can complete
                        # (slowly, under the over-budget step grace) instead of being killed as a hang.
                        if self._runtime_config.bridge_data.overbudget_exclusive_mode:
                            self._job_tracker.mark_admitted_exclusive(job)
                        self._log_overbudget_admit(job)
                    else:
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
                                # Gentle reclaim freed nothing; for the head of the queue, escalate to reclaim a
                                # queued model's RAM before falling back to cycling an allocator-stuck idle slot.
                                reclaimed = self.unload_models(under_pressure=True, for_head_of_queue=True)
                            if not reclaimed:
                                cycled = self._replace_stale_ram_unload_process()
                                # Cycling a stuck idle slot reclaims RAM by restarting it, so wait for that.
                                # Only when even cycling finds nothing to reclaim (and no live job holds RAM)
                                # is the head truly unservable by waiting; admit it best-effort then, mirroring
                                # the VRAM branch, rather than starving it.
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
            candidate_process = self._process_map.get_process_by_horde_model_name(candidate_job.model)
            if candidate_process is None or not candidate_process.can_accept_job():
                continue
            if not self._concurrent_overlap_allowed(candidate_job):
                continue
            return candidate_job, candidate_process
        return None

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
            if next_job is None:
                next_job = job
            next_n_jobs.append(job)

        if next_job is None:
            return None

        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

        process_with_model = self._process_map.get_process_by_horde_model_name(next_job.model)
        line_skip: LineSkip | None = None

        candidate_job_size = 25

        if bridge_data.high_performance_mode:
            candidate_job_size = 100
        elif bridge_data.moderate_performance_mode:
            candidate_job_size = 50

        def select_line_skip_candidate(displaced_job: ImageGenerateJobPopResponse) -> NextJobAndProcess | None:
            """Select a small, ready job that may bypass ``displaced_job`` while its slot is non-sampling."""
            for candidate_small_job in next_n_jobs:
                candidate_id = str(candidate_small_job.id_)[:8]
                job_has_loras = (
                    candidate_small_job.payload.loras is not None and len(candidate_small_job.payload.loras) > 0
                )
                if candidate_small_job.model is None:
                    logger.debug(f"Line-skip candidate {candidate_id} rejected: missing model.")
                    continue
                if candidate_small_job.model == displaced_job.model:
                    logger.debug(
                        f"Line-skip candidate {candidate_id} rejected: same model as blocked job "
                        f"{str(displaced_job.id_)[:8]}.",
                    )
                    continue
                if job_has_loras:
                    logger.debug(f"Line-skip candidate {candidate_id} rejected: candidate has LoRAs.")
                    continue
                if self._job_tracker.is_degraded_dispatch_pending(candidate_small_job):
                    logger.debug(f"Line-skip candidate {candidate_id} rejected: degraded retry must run isolated.")
                    continue

                candidate_process_with_model = self._process_map.get_process_by_horde_model_name(
                    candidate_small_job.model,
                )
                if candidate_process_with_model is None:
                    logger.debug(
                        f"Line-skip candidate {candidate_id} rejected: model {candidate_small_job.model} "
                        "is not resident.",
                    )
                    continue

                candidate_effective_mps = self.get_single_job_effective_megapixelsteps(candidate_small_job)
                if candidate_effective_mps > candidate_job_size:
                    logger.debug(
                        f"Line-skip candidate {candidate_id} rejected: {candidate_effective_mps} eMPS exceeds "
                        f"{candidate_job_size} eMPS limit.",
                    )
                    continue

                if not candidate_process_with_model.can_accept_job():
                    logger.debug(
                        f"Line-skip candidate {candidate_id} rejected: process "
                        f"{candidate_process_with_model.process_id} is "
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

        processes_post_processing = 0
        if self.post_process_job_overlap_allowed:
            processes_post_processing = self._process_map.num_busy_with_post_processing()

        jobs_in_progress_count = len(self._job_tracker.jobs_in_progress)
        max_jobs_allowed = self._max_jobs_in_progress_allowed(processes_post_processing)
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
                bypass = select_line_skip_candidate(next_job)
                if bypass is not None:
                    self._pending_line_skip = bypass
                    return bypass
            return None

        async def handle_process_missing(job: ImageGenerateJobPopResponse) -> None:
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
                    candidate_process = self._process_map.get_process_by_horde_model_name(candidate_job.model)
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
                await handle_process_missing(next_job)
                return None

        if not process_with_model.can_accept_job():
            if (process_with_model.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL) or (
                self.post_process_job_overlap_allowed
                and process_with_model.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
            ):
                line_skip_selection = select_line_skip_candidate(next_job)
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
        if not information_only and line_skip is None and not self._concurrent_overlap_allowed(next_job):
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

        color_format_string = "<fg #f0beff>{message}</>"

        logger.opt(ansi=True).info(
            color_format_string.format(
                message=f"Starting inference for job {str(next_job.id_)[:8]} "
                f"on process {process_with_model.process_id}",
            ),
        )

        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

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

        process_with_model.batch_amount = next_job.payload.n_iter
        if process_with_model.safe_send_message(
            HordeInferenceControlMessage(
                control_flag=HordeControlFlag.START_INFERENCE,
                horde_model_name=next_job.model,
                sdk_api_job_info=next_job,
            ),
        ):
            await self._job_tracker.mark_inference_started(next_job)
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

        self._pending_line_skip = None

        return True

    def _compute_wanted_models(self) -> set[str]:
        """The set of models the worker is actively serving right now.

        Derived from live scheduler state — every model currently resident on an inference
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

        Feeds the residency grace period (:meth:`_is_recently_demanded`). Only genuine demand —
        not mere residency — refreshes the stamp, so a loaded-but-idle model's grace still
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
          This is the regime the soak measures and the dominant duty-cycle win — it stops a
          process evicting the very model it just used (and is about to reuse) the instant its
          next job has not yet been popped.
        - **More models than processes**: residency cannot be guaranteed, so apply only a RAM
          grace period — cheap to hold, and it avoids the expensive disk reload between a model's
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
            return False

        if affinity_active(len(wanted_models), self._max_inference_processes) and model_name in wanted_models:
            return True

        return not vram and self._is_recently_demanded(model_name)

    def unload_models_from_vram(
        self,
        process_with_model: HordeProcessInfo,
        *,
        under_pressure: bool = False,
        for_head_of_queue: bool = False,
    ) -> bool:
        """Unload models from VRAM from processes that are not running a job.

        ``under_pressure`` (set by the VRAM budget when the next job does not fit) drops residency
        protection and the single-model hold-back so the coldest idle resident copy is reclaimed,
        while still never touching an in-progress or next-up model.

        ``for_head_of_queue`` is the last-resort escalation when the head-of-queue job cannot be loaded
        and gentle reclaim freed nothing because every idle resident copy is another *queued* job's
        model: it additionally overrides the next-up guard so the head can be given room. It never
        evicts an in-progress (live) model.

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

    async def run_scheduling_cycle(self, stable_diffusion_reference: dict[str, ImageGenerationModelRecord]) -> None:
        """Run a single scheduling cycle: preload, detect heavy model/batch, start inference, unload.

        This absorbs the inline orchestration block from _process_control_loop.
        """
        self._pending_line_skip = None
        bridge_data = self._runtime_config.bridge_data

        self._refresh_model_demand()

        if not self.preload_models():
            next_job_and_process = await self.get_next_job_and_process(information_only=True)

            next_job_heavy_model_and_workflow = False
            if next_job_and_process is not None:
                next_model = next_job_and_process.next_job.model
                if next_model is not None:
                    next_model_baseline = stable_diffusion_reference.get(next_model)
                    next_workflow = next_job_and_process.next_job.payload.workflow

                    next_job_heavy_model_and_workflow = (
                        next_model_baseline is not None
                        and next_model_baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
                        and next_workflow in KNOWN_SLOW_WORKFLOWS
                    )

                    if next_model in VRAM_HEAVY_MODELS:
                        next_job_heavy_model_and_workflow = True

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
                # (its own concurrency gate — jobs_in_progress >= max_concurrent, no free process,
                # or no eligible job — stops the loop), so this cannot over-subscribe.
                started_any = False
                while await self.start_inference():
                    started_any = True

                if not started_any:
                    self.unload_models()
