"""Handles job popping from the AI Horde API."""

from __future__ import annotations

import asyncio
import collections
import random
import time
from asyncio import CancelledError
from collections.abc import Callable
from typing import TYPE_CHECKING

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopRequest,
    ImageGenerateJobPopResponse,
)
from loguru import logger

from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.gpu.gpu_eligibility import eligible_card_indices_for
from horde_worker_regen.process_management.gpu.gpu_pop_shaping import (
    AdvertisedCapabilities,
    advertised_capabilities,
    under_fed_card,
)
from horde_worker_regen.process_management.ipc.api_sessions import ApiSessions
from horde_worker_regen.process_management.jobs.job_models import APIWorkerMessage
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.jobs.large_model_pop_governor import (
    LargeModelGovernorStatus,
    LargeModelPopGovernor,
)
from horde_worker_regen.process_management.jobs.source_image_downloader import SourceImageDownloader
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.feature_readiness import (
    CONTROLNET_ANNOTATOR_FAILED_DETAIL,
    FeatureInputs,
    GatedFeature,
    build_feature_readiness,
    is_offered,
)
from horde_worker_regen.process_management.models.model_availability import ModelAvailability
from horde_worker_regen.process_management.models.model_sizing import is_extra_large_model
from horde_worker_regen.process_management.resources.resource_budget import (
    is_model_locally_unservable_for,
    predict_job_weight_mb,
)
from horde_worker_regen.process_management.scheduling.pop_throttler import (
    CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS,
    PopThrottler,
)
from horde_worker_regen.process_management.simulation._canned_scenarios import (
    CannedJobSource,
    make_default_dry_run_source,
)
from horde_worker_regen.reporting.maintenance_messenger import MaintenanceModeMessenger
from horde_worker_regen.runtime_version import runtime_version
from horde_worker_regen.telemetry_spans import queue_depth_counter, span_job_pop
from horde_worker_regen.utils.job_utils import get_single_job_magnitude

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
    from horde_worker_regen.process_management.lifecycle.shutdown_manager import ShutdownManager
    from horde_worker_regen.process_management.models.model_metadata import ModelMetadata

# Post-inference backpressure tuning. The safety stage sits downstream of inference and (unlike the
# pre-inference queue, bounded by queue_size) had no bound: when inference outran a slow/CPU safety
# stage the post-inference backlog grew until jobs aged past their horde ttl and were server-aborted as
# "too slow", which the horde answers with forced maintenance. The popper therefore refuses to pop while
# the backlog already represents more than a budget's worth of safety work.
_DEFAULT_SAFETY_SECONDS = 8.0
"""Per-check safety cost assumed before any real measurement exists (typical CPU safety check)."""
_DEFAULT_JOB_TTL_SECONDS = 150.0
"""Deadline assumed when the horde does not supply a job ttl; conservative so backpressure still bounds
the backlog. Real ttls (when present) override this."""
_POST_INFERENCE_WAIT_BUDGET_FRACTION = 0.5
"""Fraction of the job ttl the post-inference (safety+submit) tail is allowed to consume. Holding the
backlog under this keeps headroom for the inference and submit stages plus per-job variance, so a job
admitted now still clears with margin before its deadline."""
_MIN_POST_INFERENCE_BACKLOG = 2
"""Always allow at least this much post-inference backlog per safety process, so a balanced pipeline
still overlaps inference with safety instead of running them strictly one-at-a-time."""


def _select_models_for_pop(
    bridge_data: reGenBridgeData,
    process_map: ProcessMap,
    job_tracker: JobTracker,
    max_inference_processes: int,
    *,
    last_pop_had_no_jobs: bool,
    model_availability: ModelAvailability | None = None,
    configured_models: set[str] | None = None,
    card_runtimes: dict[int, CardRuntime] | None = None,
) -> set[str] | None:
    """Choose which models to include in a pop request.

    Args:
        bridge_data: The global worker config (for stickiness, custom models, and the unservable breaker).
        process_map: The live process map (for loaded/free-model stickiness).
        job_tracker: The job tracker (for the one-running-plus-one-queued cap and the unservable streak).
        max_inference_processes: The provisioned inference-process ceiling.
        last_pop_had_no_jobs: Whether the previous pop returned nothing (relaxes stickiness).
        model_availability: When provided, drops models not yet on disk.
        configured_models: The candidate model set to draw from. On a multi-GPU host this is the union of
            every card's configured models; when None it defaults to the global ``image_models_to_load``,
            byte-identical to the single-GPU behaviour.
        card_runtimes: The per-card runtime plan. On a multi-GPU host a model is held back as unservable only
            when it is unservable on *every* card that serves it (so a model fine on a big card keeps being
            advertised); when None or single-card the worker-wide streak decides, as before.

    Returns:
        A set of model names, or ``None`` if no models are eligible (caller should skip the pop).
    """
    configured = set(bridge_data.image_models_to_load) if configured_models is None else set(configured_models)
    models = set(configured)

    # Never advertise a model that is not on disk: a job for it would be popped only to fault when the
    # inference process cannot find the checkpoint. While availability is unknown (no download process)
    # this is a no-op, preserving the behaviour of workers that pre-download everything.
    if model_availability is not None:
        models = model_availability.filter_present(models)

    loaded_models = {
        process.loaded_horde_model_name
        for process in process_map.values()
        if process.loaded_horde_model_name is not None
    }

    if len(configured) > max_inference_processes and len(loaded_models) == max_inference_processes:
        if (
            (not last_pop_had_no_jobs)
            and bridge_data.horde_model_stickiness > 0
            and random.random() < bridge_data.horde_model_stickiness
        ):
            free_models = {
                process.loaded_horde_model_name
                for process in process_map.values()
                if not process.is_process_busy() and process.loaded_horde_model_name is not None
            }
            if len(loaded_models) >= 1:
                # free_models may be empty when all inference processes are
                # busy; in that case no pop occurs (intentional: there is
                # no process available to accept a new job).
                models = free_models
            logger.debug(f"Sticky models: popping only {models}")
            if len(configured) > 10:
                logger.warning(
                    "Model stickiness is intended mostly for slow disks and works best with few models. "
                    f"You have {len(configured)} models configured.",
                )
        elif bridge_data.horde_model_stickiness > 0:
            logger.debug("Models unstuck: asking to pop for all available models.")

    # Only allow one running plus one queued for a given model
    models_to_remove = {
        model
        for model, count in collections.Counter(
            [job.model for job in job_tracker.jobs_pending_inference],
        ).items()
        if count >= 2
    }
    if len(models_to_remove) > 0:
        models = models.difference(models_to_remove)

    # Hold back models the device has shown it genuinely cannot run. A model that faults every
    # over-budget attempt would otherwise be popped only to be dropped, and a steady drop stream trips
    # the horde's "dropping too many jobs" maintenance guard. Shares the scheduler's best-effort-admit
    # breaker policy so popping and admitting agree on which models are locally unservable.
    if card_runtimes is not None and len(card_runtimes) > 1:
        # Multi-GPU: hold a model back only when every card that serves it has flagged it unservable. A model
        # still servable on at least one card keeps being advertised; the worker routes it to that card.
        held_back = set()
        for model in models:
            serving_cards = [
                device_index
                for device_index, card in card_runtimes.items()
                if model in card.config.image_models_to_load
            ]
            if serving_cards and all(
                is_model_locally_unservable_for(bridge_data, job_tracker, model, device_index=device_index)
                for device_index in serving_cards
            ):
                held_back.add(model)
    else:
        held_back = {model for model in models if is_model_locally_unservable_for(bridge_data, job_tracker, model)}
    if held_back:
        logger.debug(f"Not popping models held back as locally unservable: {sorted(held_back)}")
        models = models.difference(held_back)

    if bridge_data.custom_models is not None and len(bridge_data.custom_models) > 0:
        logger.debug("Custom models are enabled, adding them to the list of models to pop")
        custom_model_names = {model["name"] for model in bridge_data.custom_models}
        models.update(custom_model_names)

    if len(models) == 0:
        if (
            model_availability is not None
            and model_availability.is_known
            and (model_availability.currently_downloading or model_availability.pending)
        ):
            logger.info(
                "No configured models are on disk yet; waiting for downloads "
                f"(downloading: {model_availability.currently_downloading}, "
                f"pending: {len(model_availability.pending)})",
            )
        else:
            logger.debug("Not eligible to pop a job yet")
        return None

    return models


class JobPopper:
    """Owns job pop logic: requesting new jobs from the API and downloading source images."""

    _state: WorkerState
    _process_map: ProcessMap
    _job_tracker: JobTracker
    _shutdown_manager: ShutdownManager
    _runtime_config: RuntimeConfig
    _api_sessions: ApiSessions

    _pop_throttler: PopThrottler
    _source_image_downloader: SourceImageDownloader

    _replaced_due_to_maintenance: bool
    _api_messages_received: dict[str, APIWorkerMessage]
    _api_call_loop_interval: float
    _fast_pop_interval: float

    _canned_job_source: CannedJobSource | None
    _model_availability: ModelAvailability | None

    _max_inference_processes: int
    _max_threads_ceiling: int
    _card_runtimes: dict[int, CardRuntime]
    _model_metadata: ModelMetadata | None

    def __init__(
        self,
        *,
        state: WorkerState,
        process_map: ProcessMap,
        job_tracker: JobTracker,
        shutdown_manager: ShutdownManager,
        runtime_config: RuntimeConfig,
        api_sessions: ApiSessions,
        max_inference_processes: int,
        max_concurrent_inference_processes: int,
        dry_run_skip_api: bool = False,
        canned_job_source: CannedJobSource | None = None,
        model_availability: ModelAvailability | None = None,
        card_runtimes: dict[int, CardRuntime] | None = None,
        model_metadata: ModelMetadata | None = None,
        whole_card_residency_active: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize with all required dependencies for job popping.

        When `dry_run_skip_api` is set, jobs come from `canned_job_source` instead of
        the live API; if no source is given, an endlessly-cycling default is used.

        When `model_availability` is provided, only models present on disk are advertised in
        pop requests (a missing model would otherwise be popped and then fault).

        When `card_runtimes` has more than one card, the pop advertises the union of the cards'
        capabilities (models, features, resolution, threads); a single card (or None) advertises the global
        config exactly as before.

        `whole_card_residency_active` is queried by the large-model re-entry cooldown to know whether a
        whole-card residency lease is still held; it defaults to "never held" so a worker wired without it
        (and the tests) behaves as if no lease is ever active.
        """
        self._state = state
        self._process_map = process_map
        self._job_tracker = job_tracker
        self._shutdown_manager = shutdown_manager
        self._runtime_config = runtime_config
        self._api_sessions = api_sessions
        self._card_runtimes = card_runtimes if card_runtimes is not None else {}
        self._model_metadata = model_metadata
        self._whole_card_residency_active = (
            whole_card_residency_active if whole_card_residency_active is not None else (lambda: False)
        )
        self._large_model_pop_governor = LargeModelPopGovernor()

        self._max_inference_processes = max_inference_processes
        # The constructor value is the provisioned ceiling; the threads advertised in pop requests
        # track the live effective cap (see the _max_concurrent_inference_processes property).
        self._max_threads_ceiling = max_concurrent_inference_processes
        self._dry_run_skip_api = dry_run_skip_api

        self._canned_job_source = canned_job_source
        if dry_run_skip_api and self._canned_job_source is None:
            self._canned_job_source = make_default_dry_run_source()

        self._model_availability = model_availability

        self._pop_throttler = PopThrottler(job_tracker=job_tracker)
        self._source_image_downloader = SourceImageDownloader(
            api_sessions=api_sessions,
            job_tracker=job_tracker,
        )

        self._replaced_due_to_maintenance = False
        self._api_messages_received = {}
        self._api_call_loop_interval = 1
        self._fast_pop_interval = 0.05

    @property
    def _max_concurrent_inference_processes(self) -> int:
        """The live concurrent-inference cap (effective ``max_threads``) advertised to the API."""
        return self._runtime_config.effective_max_threads

    @property
    def _multi_gpu_advertise(self) -> bool:
        """Whether the pop advertises a union across cards (the worker drives more than one)."""
        return len(self._card_runtimes) > 1

    def _advertised_capabilities(self) -> AdvertisedCapabilities | None:
        """The union pop envelope on a multi-GPU host, or None on a single-GPU host (use the global config).

        Returns None when the worker drives one card (or none), so the caller advertises the global config
        exactly as before; otherwise the union of every card's capabilities (see
        :func:`~horde_worker_regen.process_management.gpu.gpu_pop_shaping.advertised_capabilities`).
        """
        if not self._multi_gpu_advertise:
            return None
        return advertised_capabilities(self._card_runtimes)

    def _gpu_pop_balance_threshold(self, bridge_data: reGenBridgeData) -> float:
        """The configured fraction of held work a card must be unable to serve before a pop targets it.

        Read tolerantly (a non-numeric mocked value falls back to 0.5) and clamped to ``[0, 1]``.
        """
        raw = bridge_data.gpu_pop_balance_threshold
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return 0.5
        return max(0.0, min(1.0, float(raw)))

    def _targeted_under_fed_card(self, bridge_data: reGenBridgeData) -> int | None:
        """The under-fed card this pop should be scoped to, or None to keep union-popping.

        Computes, for every held job (queued, including those already in flight), which cards could serve it,
        then asks :func:`~horde_worker_regen.process_management.gpu.gpu_pop_shaping.under_fed_card` whether one
        card is starved past the configured balance threshold. None on a single-GPU host (no targeting) or
        when model metadata is unavailable (eligibility cannot be judged, so the worker union-pops).
        """
        if not self._multi_gpu_advertise or self._model_metadata is None:
            return None
        held_jobs = self._job_tracker.jobs_pending_inference
        if not held_jobs:
            return None
        eligible_sets: list[set[int]] = []
        for job in held_jobs:
            baseline = self._model_metadata.get_baseline(job.model) if job.model is not None else None
            baseline_value = baseline.value if isinstance(baseline, KNOWN_IMAGE_GENERATION_BASELINE) else baseline
            weight_mb = predict_job_weight_mb(job, baseline)
            eligible_sets.append(
                eligible_card_indices_for(job, self._card_runtimes, baseline=baseline_value, weight_mb=weight_mb),
            )
        return under_fed_card(
            eligible_sets,
            self._card_runtimes.keys(),
            balance_threshold=self._gpu_pop_balance_threshold(bridge_data),
        )

    def _baseline_value_for(self, model_name: str | None) -> str | None:
        """The model's baseline value from the loaded reference, or None when metadata is unavailable.

        With no metadata (or no name) the classifier still recognizes the named VRAM-heavy checkpoints by name,
        so Flux/Cascade compact checkpoints are caught even before a reference is loaded.
        """
        if self._model_metadata is None or model_name is None:
            return None
        baseline = self._model_metadata.get_baseline(model_name)
        return baseline.value if isinstance(baseline, KNOWN_IMAGE_GENERATION_BASELINE) else baseline

    def _is_large_model(self, model_name: str | None) -> bool:
        """Whether a model is in the EXTRA_LARGE ('very large') tier the pop limiters govern."""
        return model_name is not None and is_extra_large_model(model_name, self._baseline_value_for(model_name))

    def _large_models_loaded_or_queued(self) -> frozenset[str]:
        """The very-large models currently resident on a process or held in the local queue (incl. in flight)."""
        in_play: set[str] = set()
        for process in self._process_map.values():
            model = process.loaded_horde_model_name
            if model is not None and self._is_large_model(model):
                in_play.add(model)
        for job in self._job_tracker.jobs_pending_inference:
            if job.model is not None and self._is_large_model(job.model):
                in_play.add(job.model)
        return frozenset(in_play)

    @staticmethod
    def _coerce_seconds(value: object, *, default: float) -> float:
        """Coerce a config duration to float, falling back to ``default`` for a non-numeric (e.g. mocked) value."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return float(value)

    def _resolve_large_model_pop_durations(self, bridge_data: reGenBridgeData) -> tuple[float, float]:
        """Return the effective (switch_min_seconds, reentry_cooldown_seconds), resolving the -1 inherit.

        A negative re-entry value inherits ``whole_card_residency_cooldown_seconds`` (the lease it complements);
        a non-numeric config (a partial mock) reads as disabled, so the limiter never crashes the pop cycle.
        """
        switch_min = self._coerce_seconds(getattr(bridge_data, "large_model_switch_min_seconds", 0), default=0.0)
        reentry_raw = getattr(bridge_data, "large_model_reentry_cooldown_seconds", -1)
        if isinstance(reentry_raw, bool) or not isinstance(reentry_raw, (int, float)):
            reentry = 0.0
        elif reentry_raw < 0:
            reentry = self._coerce_seconds(
                getattr(bridge_data, "whole_card_residency_cooldown_seconds", 0),
                default=0.0,
            )
        else:
            reentry = float(reentry_raw)
        return switch_min, reentry

    def _apply_large_model_pop_limits(self, models: set[str], bridge_data: reGenBridgeData) -> set[str]:
        """Withhold very-large models from the offer per the switch throttle and re-entry cooldown.

        Both limiters are off by default (zero durations) and yield to an idle escape (the worker holds no
        work locally), so the worker never sits idle when the only work it could take is a large model. See
        :class:`~horde_worker_regen.process_management.jobs.large_model_pop_governor.LargeModelPopGovernor`.
        """
        switch_min, reentry = self._resolve_large_model_pop_durations(bridge_data)
        if switch_min <= 0 and reentry <= 0:
            return models

        candidate_large = frozenset(model for model in models if self._is_large_model(model))
        decision = self._large_model_pop_governor.evaluate(
            candidate_large_models=candidate_large,
            incumbent_large_models=self._large_models_loaded_or_queued(),
            residency_active=bool(self._whole_card_residency_active()),
            now=time.time(),
            switch_min_seconds=switch_min,
            reentry_cooldown_seconds=reentry,
            idle_escape=self._job_tracker.num_jobs_total == 0,
        )
        if decision.withheld:
            logger.debug(
                f"Large-model pop limiter ({decision.reason}): withholding {sorted(decision.withheld)} from "
                "this pop offer.",
            )
            return models.difference(decision.withheld)
        return models

    def large_model_governor_status(self, *, now: float, residency_active: bool) -> LargeModelGovernorStatus:
        """Report the live engagement of the two large-model limiters, for the governor registry.

        Resolves the configured durations and the current large-model incumbents the same way the pop filter
        does, then asks the governor for a read-only status (no mutation of its throttle timers).
        """
        bridge_data = self._runtime_config.bridge_data
        switch_min, reentry = self._resolve_large_model_pop_durations(bridge_data)
        return self._large_model_pop_governor.describe(
            incumbent_large_models=self._large_models_loaded_or_queued(),
            residency_active=residency_active,
            now=now,
            switch_min_seconds=switch_min,
            reentry_cooldown_seconds=reentry,
        )

    def is_post_inference_backlogged(self) -> bool:
        """Public read of the post-inference backpressure gate, for the governor registry."""
        return self._is_post_inference_backlogged()

    @property
    def is_in_error_backoff(self) -> bool:
        """Whether the pop throttler is backing off the API after recent pop errors."""
        return self._pop_throttler.is_in_error_backoff

    def megapixelstep_wait_remaining(self, bridge_data: reGenBridgeData, *, now: float) -> float | None:
        """Seconds the megapixelstep wait is still holding pops, or None when it is not engaged."""
        return self._pop_throttler.megapixelstep_wait_remaining(bridge_data, now=now)

    def set_canned_job_source(self, source: CannedJobSource | None) -> None:
        """Swap the canned job source at runtime (a warm benchmark worker's level boundary)."""
        self._canned_job_source = source

    @property
    def api_messages_received(self) -> dict[str, APIWorkerMessage]:
        """Return the worker messages received from the API, keyed by message ID."""
        return self._api_messages_received

    @property
    def time_spent_no_jobs_available(self) -> float:
        """Return the cumulative seconds spent with no jobs available."""
        return self._pop_throttler._time_spent_no_jobs_available

    @property
    def max_time_spent_no_jobs_available(self) -> float:
        """Return the longest stretch of seconds spent with no jobs available."""
        return self._pop_throttler._max_time_spent_no_jobs_available

    # region api_job_pop helper methods

    def _handle_consecutive_failures(self, bridge_data: reGenBridgeData, cur_time: float) -> bool:
        """Check and handle consecutive job failure state.

        Returns:
            True if the pop should be skipped this cycle.
        """
        if self._state.too_many_consecutive_failed_jobs:
            if cur_time - self._state.too_many_consecutive_failed_jobs_time > CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS:
                self._state.consecutive_failed_jobs = 0
                self._state.too_many_consecutive_failed_jobs = False
                logger.debug("Resuming job pops after too many consecutive failed jobs")
            return True

        if self._state.consecutive_failed_jobs >= 3:
            logger.error(
                "Too many consecutive failed jobs, pausing job pops. "
                "Please look into what happened and let the devs know. "
                f"Waiting {CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS} seconds...",
            )
            if bridge_data.exit_on_unhandled_faults:
                logger.error("Exiting due to exit_on_unhandled_faults being enabled")
                self._shutdown_manager.shutdown()
            self._state.too_many_consecutive_failed_jobs = True
            self._state.too_many_consecutive_failed_jobs_time = cur_time
            return True

        return False

    def _is_queue_full(self, bridge_data: reGenBridgeData) -> bool:
        """Return True if the job queue already has enough jobs."""
        max_jobs_in_queue = bridge_data.queue_size + 1
        if bridge_data.max_threads > 1:
            max_jobs_in_queue += bridge_data.max_threads - 1
        return len(self._job_tracker.jobs_pending_inference) >= max_jobs_in_queue

    _SAFETY_BACKLOG_LOG_INTERVAL_SECONDS = 30.0
    """Minimum gap between repeats of the "withholding pops: safety backlog" line, so the sub-second pop
    loop cannot spam it while the backpressure stays engaged."""

    _safety_backlog_log_time: float = 0.0
    """Monotonic-ish wall-clock of the last safety-backlog backpressure log (throttle state)."""

    def _max_safe_safety_backlog(self) -> int:
        """How many jobs may wait for safety before a newly popped job would risk aging out.

        Sized from the measured safety cost and the horde-supplied job ttl: a job admitted now must pass
        the whole backlog ahead of it through the (often single, CPU-bound) safety stage before it can be
        submitted, so the backlog the worker tolerates is the deadline budget divided by the per-check
        cost (scaled by the number of safety processes). Self-tunes: faster safety (or a longer ttl)
        raises the cap, a slow safety stage lowers it, with no operator knob.
        """
        avg_safety = self._state.avg_safety_seconds if self._state.avg_safety_seconds > 0 else _DEFAULT_SAFETY_SECONDS
        ttl = self._state.recent_job_ttl if self._state.recent_job_ttl is not None else _DEFAULT_JOB_TTL_SECONDS
        num_safety = max(1, self._process_map.num_safety_processes())
        budget_seconds = ttl * _POST_INFERENCE_WAIT_BUDGET_FRACTION
        capacity = int(budget_seconds * num_safety / avg_safety)
        return max(_MIN_POST_INFERENCE_BACKLOG * num_safety, capacity)

    def _is_post_inference_backlogged(self) -> bool:
        """Return True if the post-inference (safety) backlog is too deep to admit more work.

        This is the backpressure the worker previously lacked: inference completions pile into the
        ungated safety queue, so a safety stage even slightly slower than inference grows that queue until
        jobs exceed their ttl and the horde aborts them as too slow. Counting the jobs already waiting for
        (or in) safety against a deadline-derived cap lets the worker stop popping before the backlog ages
        jobs out, throttling intake to the pipeline's slowest stage instead of spiralling into
        forced maintenance.
        """
        backlog = len(self._job_tracker.jobs_pending_safety_check) + len(self._job_tracker.jobs_being_safety_checked)
        if backlog == 0:
            return False
        return backlog >= self._max_safe_safety_backlog()

    @property
    def _lora_disk_permits(self) -> bool:
        """Whether the worker-wide LoRA disk guard currently permits advertising LoRA support.

        Independent of any per-card ``allow_lora`` choice: LoRA storage is one shared cache, so a full disk
        or an in-progress background download suppresses LoRA advertising for the whole worker.
        """
        if self._state.lora_disk_exhausted:
            return False
        # Repeated ad-hoc download teardowns withhold LoRA support for an escalating window; popping
        # more LoRA jobs while the download path is failing only churns slots (see LoraDownloadBackoff).
        if self._state.lora_download_backoff.pops_suppressed(time.time()):
            return False
        return not (self._model_availability is not None and self._model_availability.background_download_active)

    def _effective_allow_lora(self, bridge_data: reGenBridgeData) -> bool:
        """Return whether this pop should advertise LoRA support (config flag and the worker-wide disk guard)."""
        return bool(bridge_data.allow_lora) and self._lora_disk_permits

    def _lora_queue_cap_reached(self) -> bool:
        """Whether the local queue already holds the most concurrently-queued LoRA jobs we allow.

        Each LoRA job blocks its slot on an ad-hoc download before it can sample, so letting LoRA jobs
        fill every queue slot leaves no non-LoRA job for the scheduler to slip past a blocked LoRA head
        (the line-skip path rejects LoRA candidates). Capping concurrently-queued LoRA jobs to one fewer
        than the inference-process count keeps at least one slot's worth of room for a skippable
        non-LoRA job, while always allowing at least one LoRA job so single-process workers still serve
        them.
        """
        cap = max(1, self._max_inference_processes - 1)
        queued_lora_jobs = sum(
            1
            for job in self._job_tracker.jobs_pending_inference
            if job.payload.loras is not None and len(job.payload.loras) > 0
        )
        return queued_lora_jobs >= cap

    def _is_hungry(self, bridge_data: reGenBridgeData) -> bool:
        """Whether the worker should pop again immediately instead of waiting the poll interval.

        True only when work is actively flowing (the last pop returned a job), the local queue
        has room (`_is_queue_full` is False), an inference process is free to take a job, and we
        are not in post-error backoff. In that state the fixed ~1s poll cadence would leave a
        freed GPU slot starved while a job is readily available; popping back-to-back fills the
        buffer so the slot refills without delay. When the queue is full, no process is free, the
        source has no work, or we are backing off, this is False and the loop reverts to polite
        interval polling; so this never increases pressure on the API beyond filling the buffer.
        """
        if self._state.last_pop_no_jobs_available:
            return False
        if self._pop_throttler.is_in_error_backoff:
            return False
        if self._is_queue_full(bridge_data):
            return False
        if self._is_post_inference_backlogged():
            return False
        return self._process_map.get_first_available_inference_process() is not None

    def _process_api_messages(self, job_pop_response: object) -> None:
        """Extract and store any worker messages from the pop response."""
        try:
            if not (
                hasattr(job_pop_response, "messages")
                and job_pop_response.messages is not None  # type: ignore[union-attr]
                and len(job_pop_response.messages) > 0  # type: ignore[union-attr]
            ):
                return

            for message in job_pop_response.messages:  # type: ignore[union-attr]
                raw_message = APIWorkerMessage.from_raw_dict(message)
                if raw_message.message_id not in self._api_messages_received:
                    self._api_messages_received[raw_message.message_id] = raw_message
                    logger.debug(
                        f"Message {raw_message.message_id} from {raw_message.message_origin} "
                        f"(expires {raw_message.message_expiry}): {raw_message.message_text}",
                    )
        except Exception as e:
            logger.error(f"Failed to process API messages: {e}")

    def _handle_pop_error_response(self, response: RequestErrorResponse) -> None:
        """Log and categorize an error response from the pop API."""
        message_lower = response.message.lower()

        if "maintenance mode" in message_lower:
            if not self._state.last_pop_maintenance_mode:
                logger.warning(f"Failed to pop job (Maintenance Mode): {response}")
                MaintenanceModeMessenger.print_maintenance_mode_messages()
                self._state.last_pop_maintenance_mode = True
                self._state.server_maintenance_cleared_by_job_pop = False
        elif "we cannot accept workers serving" in message_lower:
            logger.warning(f"Failed to pop job (Unrecognized Model): {response}")
            logger.error(
                "Your worker is configured to use a model that is not accepted by the API. "
                "Please check your models_to_load and make sure they are all valid.",
            )
        elif "wrong credentials" in message_lower:
            logger.warning(f"Failed to pop job (Wrong Credentials): {response}")
            logger.error("Did you forget to set your worker name (`dreamer_name` in bridgeData.yaml)?")
            logger.error(
                "Horde Worker names must be unique horde-wide. If you haven't used this name before, "
                "try changing your worker name.",
            )
        else:
            logger.error(f"Failed to pop job (API Error): {response}")

        self._pop_throttler.on_pop_error()
        self._state.last_pop_no_jobs_available = True

    @staticmethod
    def _apply_sdk_workarounds(
        job_pop_response: ImageGenerateJobPopResponse,
    ) -> ImageGenerateJobPopResponse:
        """Fix up payload fields that the SDK does not handle correctly yet.

        TODO: move to horde_sdk once the SDK is updated.
        """
        needs_rebuild = False
        new_response_dict = None

        if job_pop_response.payload.seed is None:
            logger.warning(f"Job {job_pop_response.id_} has no seed!")
            new_response_dict = job_pop_response.model_dump(by_alias=True)
            new_response_dict["payload"]["seed"] = random.randint(0, (2**32) - 1)
            needs_rebuild = True

        if job_pop_response.payload.denoising_strength is not None and job_pop_response.source_image is None:
            if new_response_dict is None:
                new_response_dict = job_pop_response.model_dump(by_alias=True)
            new_response_dict["payload"]["denoising_strength"] = None
            needs_rebuild = True

        if needs_rebuild and new_response_dict is not None:
            job_pop_response = ImageGenerateJobPopResponse(**new_response_dict)

        return job_pop_response

    async def _enqueue_popped_job(
        self,
        job_pop_response: ImageGenerateJobPopResponse,
    ) -> None:
        """Add a successfully popped job to the pending inference queue."""
        await self._job_tracker.record_popped_job(job_pop_response)
        # Remember the horde-supplied deadline so post-inference backpressure can be sized to it; the
        # field stays at its last known value (or None) when a pop omits the ttl.
        if job_pop_response.ttl is not None:
            self._state.recent_job_ttl = float(job_pop_response.ttl)
        jobs = []
        for job in self._job_tracker.jobs_pending_inference:
            if job.id_ is not None:
                jobs.append(f"<{str(job.id_)[:8]}: {job.model}>")
            else:
                jobs.append(f"<{job.model}>")
        logger.info(f"Job queue: {', '.join(jobs)}")

    # endregion

    @logger.catch(reraise=True)
    async def api_job_pop(self, *, urgent: bool = False) -> None:
        """Pop a job from the API if the queue is not full and preconditions are met.

        Args:
            urgent: When True, skip the inter-pop frequency gate so the local queue can be
                refilled back-to-back while a GPU slot is starved. The caller is responsible for
                only setting this when the worker is genuinely hungry (see :meth:`_is_hungry`);
                all other preconditions (queue-full, free process, megapixelstep wait, error
                backoff) are still enforced below.
        """
        if self._state.shutting_down:
            self._state.last_pop_no_jobs_available = False
            return

        if self._state.supervisor_paused or self._state.self_throttle_paused:
            self._state.last_pop_no_jobs_available = False
            return

        if self._state.gpu_torch_incompatible:
            # The installed PyTorch has no kernels for this GPU: every job would fail at the first kernel
            # launch, so never pop. Sticky for the session (a build/hardware mismatch); fixed by reinstalling.
            self._state.last_pop_no_jobs_available = False
            return

        if self._state.downloads_only_hold:
            # Download-only posture: pre-fetch models without committing the GPU; pop nothing until GO_LIVE.
            self._state.last_pop_no_jobs_available = False
            return

        cur_time = time.time()
        bridge_data = self._runtime_config.bridge_data

        if self._handle_consecutive_failures(bridge_data, cur_time):
            return

        if self._is_queue_full(bridge_data):
            return

        # Post-inference backpressure: if the safety stage is backed up enough that a job admitted now
        # would likely age past its ttl waiting for it, stop popping until the backlog drains. Without
        # this the worker keeps accepting work a slow (often CPU) safety stage cannot clear, the backlog
        # grows unbounded, and the horde aborts the aged jobs as too slow and forces maintenance.
        if self._is_post_inference_backlogged():
            self._state.last_pop_no_jobs_available = False
            self._state.last_pop_skipped_reasons["safety_backlog"] = (
                self._state.last_pop_skipped_reasons.get("safety_backlog", 0) + 1
            )
            # Surface the backpressure in prose, throttled so the sub-second pop loop never spams it: a
            # bundle should show pops were stopped *because the safety stage is backed up*, not merely that
            # pops stopped. Names the depth, the self-tuned cap, and the oldest waiting safety job so a
            # slow downstream stage (typically CPU safety) is unmistakable.
            now = time.time()
            if (now - self._safety_backlog_log_time) >= self._SAFETY_BACKLOG_LOG_INTERVAL_SECONDS:
                self._safety_backlog_log_time = now
                backlog = len(self._job_tracker.jobs_pending_safety_check) + len(
                    self._job_tracker.jobs_being_safety_checked,
                )
                safety_ages = self._job_tracker.stage_age_summary()
                oldest = max(
                    safety_ages.get(JobStage.PENDING_SAFETY_CHECK, (0, 0.0))[1],
                    safety_ages.get(JobStage.SAFETY_CHECKING, (0, 0.0))[1],
                )
                logger.warning(
                    f"Withholding job pops: post-inference safety backlog {backlog} >= cap "
                    f"{self._max_safe_safety_backlog()} (oldest waiting safety job {oldest:.0f}s). The safety "
                    "stage is slower than inference; if this persists, enable safety_on_gpu or speed safety up.",
                )
            return

        # Warm-up rule: until the first job of the session has completed, don't queue
        # ahead (if we're doomed to fail with 1 job, we're doomed to fail with 2).
        if len(self._job_tracker.jobs_pending_inference) != 0 and self._job_tracker.total_num_completed_jobs == 0:
            return

        if self._process_map.get_first_available_safety_process() is None:
            return

        if self._process_map.get_first_available_inference_process() is None:
            return

        if len(bridge_data.image_models_to_load) == 0:
            logger.error("No models are configured to be loaded, please check your config (models_to_load).")
            await asyncio.sleep(3)
            return

        if self._pop_throttler.should_wait_for_megapixelsteps(bridge_data):
            return

        if not urgent and self._pop_throttler.is_pop_too_soon(self._state.last_job_pop_time):
            return

        self._state.last_job_pop_time = time.time()

        # On a multi-GPU host advertise the union of every card's capabilities so the horde returns work
        # any card can run (the worker then routes each job to an eligible card); single-GPU advertises the
        # global config unchanged.
        advertised = self._advertised_capabilities()

        # Adaptive targeting: when the local queue is lopsided away from one card (most held work is servable
        # only by other cards), scope THIS pop to the under-fed card's capabilities so the horde returns work
        # it can actually run, instead of more work for the already-fed cards. Union-pop otherwise.
        if advertised is not None:
            under_fed = self._targeted_under_fed_card(bridge_data)
            if under_fed is not None:
                advertised = advertised_capabilities({under_fed: self._card_runtimes[under_fed]})
                logger.debug(
                    f"Adaptive pop: local queue is lopsided away from card {under_fed}; scoping this pop to "
                    "its capabilities.",
                )

        models = _select_models_for_pop(
            bridge_data,
            self._process_map,
            self._job_tracker,
            self._max_inference_processes,
            last_pop_had_no_jobs=self._state.last_pop_no_jobs_available,
            model_availability=self._model_availability,
            configured_models=set(advertised.models) if advertised is not None else None,
            card_runtimes=self._card_runtimes if self._multi_gpu_advertise else None,
        )
        if models is None:
            return

        # Tame pathological mixed very-large-model queues: withhold a switched-to or just-drained large model
        # from this offer so the worker is not whipsawed into repeated whole-card teardowns and multi-GB
        # reloads. A no-op unless the operator configures a switch interval or re-entry cooldown.
        models = self._apply_large_model_pop_limits(models, bridge_data)
        if len(models) == 0:
            return

        pop_nsfw = advertised.nsfw if advertised is not None else bridge_data.nsfw
        pop_threads = advertised.threads if advertised is not None else self._max_concurrent_inference_processes
        pop_max_power = advertised.max_power if advertised is not None else bridge_data.max_power
        pop_allow_img2img = advertised.allow_img2img if advertised is not None else bridge_data.allow_img2img
        pop_allow_painting = advertised.allow_inpainting if advertised is not None else bridge_data.allow_inpainting
        pop_allow_post_processing = (
            advertised.allow_post_processing if advertised is not None else bridge_data.allow_post_processing
        )
        # Session-latched off by the post-processing fault breaker: once repeated post-processing peaks could
        # not be hosted, stop advertising post-processing so the worker is not handed more upscale/face-fix
        # jobs it cannot host (which would keep faulting toward the horde's forced-maintenance).
        if self._state.post_processing_disabled_by_breaker:
            pop_allow_post_processing = False
        pop_allow_controlnet = advertised.allow_controlnet if advertised is not None else bridge_data.allow_controlnet
        pop_allow_sdxl_controlnet = (
            advertised.allow_sdxl_controlnet if advertised is not None else bridge_data.allow_sdxl_controlnet
        )
        # Union LoRA: any card opting in, still subject to the worker-wide LoRA disk guard.
        pop_allow_lora = (
            (advertised.allow_lora and self._lora_disk_permits)
            if advertised is not None
            else self._effective_allow_lora(bridge_data)
        )
        # Stop advertising LoRA support once the queue is already carrying its allowed share of LoRA
        # jobs, so a non-LoRA job can still be popped and line-skip past a blocked LoRA head.
        if pop_allow_lora and self._lora_queue_cap_reached():
            pop_allow_lora = False

        # First-class feature readiness: withhold a gated feature (ControlNet, SDXL-ControlNet,
        # post-processing) until its models/annotators are actually on disk, so the worker never
        # advertises a capability whose aux downloads are still in flight (a job for it would only fault).
        # While availability is unknown (no download process / no report yet) this is a no-op, preserving
        # the behaviour of workers that pre-download everything.
        if self._model_availability is not None:
            readiness = build_feature_readiness(
                {
                    GatedFeature.CONTROLNET: FeatureInputs(
                        enabled=pop_allow_controlnet,
                        present=self._model_availability.controlnet_present,
                        failed=self._model_availability.controlnet_failed,
                        failed_detail=CONTROLNET_ANNOTATOR_FAILED_DETAIL,
                    ),
                    GatedFeature.SDXL_CONTROLNET: FeatureInputs(
                        enabled=pop_allow_sdxl_controlnet,
                        present=self._model_availability.sdxl_controlnet_present,
                        failed=self._model_availability.controlnet_failed,
                        failed_detail=CONTROLNET_ANNOTATOR_FAILED_DETAIL,
                    ),
                    GatedFeature.POST_PROCESSING: FeatureInputs(
                        enabled=pop_allow_post_processing,
                        present=self._model_availability.post_processing_present,
                    ),
                },
            )
            pop_allow_controlnet = is_offered(readiness, GatedFeature.CONTROLNET)
            pop_allow_sdxl_controlnet = is_offered(readiness, GatedFeature.SDXL_CONTROLNET)
            pop_allow_post_processing = is_offered(readiness, GatedFeature.POST_PROCESSING)

        try:
            job_pop_request = ImageGenerateJobPopRequest(
                apikey=bridge_data.api_key,
                name=bridge_data.dreamer_worker_name,
                bridge_agent=f"AI Horde Worker reGen:{runtime_version()}:https://github.com/Haidra-Org/horde-worker-reGen",
                models=list(models),
                blacklist=bridge_data.blacklist,
                nsfw=pop_nsfw,
                threads=pop_threads,
                max_pixels=pop_max_power * 8 * 64 * 64,
                require_upfront_kudos=bridge_data.require_upfront_kudos,
                allow_img2img=pop_allow_img2img,
                allow_painting=pop_allow_painting,
                allow_unsafe_ipaddr=bridge_data.allow_unsafe_ip,
                allow_post_processing=pop_allow_post_processing,
                allow_controlnet=pop_allow_controlnet,
                allow_sdxl_controlnet=pop_allow_sdxl_controlnet,
                extra_slow_worker=bridge_data.extra_slow_worker,
                limit_max_steps=bridge_data.limit_max_steps,
                allow_lora=pop_allow_lora,
                amount=bridge_data.max_batch,
            )

            if self._dry_run_skip_api:
                if self._canned_job_source is None:
                    raise RuntimeError("dry_run_skip_api is set but no canned job source is configured")

                job_pop_response = self._canned_job_source.next_pop_response()
                if job_pop_response.id_ is not None:
                    queue_depth_counter.add(1)
            else:
                with span_job_pop(models=",".join(sorted(models))):
                    job_pop_response = await self._api_sessions.require_horde_client_session().submit_request(
                        job_pop_request,
                        ImageGenerateJobPopResponse,
                    )

            self._process_api_messages(job_pop_response)

            if isinstance(job_pop_response, RequestErrorResponse):
                self._handle_pop_error_response(job_pop_response)
                return

        except Exception as e:
            if self._pop_throttler.current_pop_frequency == self._pop_throttler._error_pop_frequency:
                logger.error(f"Failed to pop job (Unexpected Error): {e}")
            else:
                logger.warning(f"Failed to pop job (Unexpected Error): {e}")
            self._pop_throttler.on_pop_error()
            return

        self._pop_throttler.on_pop_success()

        info_string = "No job available. "
        if len(self._job_tracker.jobs_pending_inference) > 0:
            info_string += f"Current number of popped jobs: {len(self._job_tracker.jobs_pending_inference)}. "

        skipped_reasons = job_pop_response.skipped.model_dump(exclude_defaults=True)
        if job_pop_response.skipped.model_extra is not None:
            skipped_reasons.update(job_pop_response.skipped.model_extra)

        skipped_reasons = {k: v for k, v in skipped_reasons.items() if v != 0}

        info_string += f"(Skipped reasons: {skipped_reasons})"

        if job_pop_response.id_ is None:
            self._state.last_pop_no_jobs_available = True
            self._state.last_pop_skipped_reasons = skipped_reasons
            logger.info(info_string)
            self._pop_throttler.on_no_jobs_available(
                cur_time,
                # Active alchemy work counts as the worker being busy, so an alchemy-only
                # stretch does not accrue "time without jobs".
                queue_empty=(
                    len(self._job_tracker.jobs_pending_inference) == 0 and self._state.alchemy_forms_in_flight == 0
                ),
            )
            return

        if self._state.last_pop_maintenance_mode:
            logger.info("Clearing horde maintenance latch: a new job was popped successfully.")
            self._state.server_maintenance_cleared_by_job_pop = True
        self._state.last_pop_maintenance_mode = False
        self._replaced_due_to_maintenance = False
        self._state.last_pop_no_jobs_available = False
        self._state.last_pop_skipped_reasons = {}
        self._pop_throttler.on_job_popped()

        has_loras = job_pop_response.payload.loras is not None and len(job_pop_response.payload.loras) > 0
        has_post_processing = (
            job_pop_response.payload.post_processing is not None
            and len(
                job_pop_response.payload.post_processing,
            )
            > 0
        )
        logger.opt(ansi=True).info(
            "<fg #a200ff>"
            f"Popped job {job_pop_response.id_} "
            f"({get_single_job_magnitude(job_pop_response)} eMPS) "
            f"(model: {job_pop_response.model}, batch: {job_pop_response.payload.n_iter}, "
            f"loras: {has_loras}, post_processing: {has_post_processing})"
            "</>",
        )

        job_pop_response = self._apply_sdk_workarounds(job_pop_response)
        job_pop_response = await self._source_image_downloader.download_source_images(job_pop_response)

        if job_pop_response.id_ is None:
            logger.error("Job has no id!")
            return

        await self._enqueue_popped_job(job_pop_response)

    async def run(self) -> None:
        """Run the API call loop for popping jobs.

        The loop normally polls at ``_api_call_loop_interval`` (~1s). When the worker is hungry
        (a GPU slot is free, the queue has room, and work is flowing; see :meth:`_is_hungry`),
        it instead pops back-to-back at ``_fast_pop_interval`` to refill the local queue, so a
        process that just finished a job does not sit idle waiting for the next poll tick. It
        reverts to the slow cadence the moment the queue is full or no work is available.
        """
        logger.debug("In JobPopper.run")

        while True:
            urgent = self._is_hungry(self._runtime_config.bridge_data)
            with logger.catch():
                try:
                    await self.api_job_pop(urgent=urgent)
                except CancelledError as e:
                    self._shutdown_manager.shutdown()
                    logger.debug(f"CancelledError: {e}")

            # Checked outside the catch block so persistent errors cannot prevent shutdown.
            if self._shutdown_manager.is_time_for_shutdown() or self._state.shut_down:
                break

            still_hungry = self._is_hungry(self._runtime_config.bridge_data)
            await asyncio.sleep(self._fast_pop_interval if still_hungry else self._api_call_loop_interval)
