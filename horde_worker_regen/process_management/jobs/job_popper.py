"""Handles job popping from the AI Horde API."""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import hashlib
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
from horde_sdk.worker.dispatch.ai_horde.image.convert import apply_image_worker_feature_flags_to_pop_request
from loguru import logger

from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import PopGate, WorkerState
from horde_worker_regen.process_management.gpu.card_runtime import safety_permitted_card_indices
from horde_worker_regen.process_management.gpu.gpu_eligibility import eligible_card_indices_for
from horde_worker_regen.process_management.gpu.gpu_pop_shaping import (
    AdvertisedCapabilities,
    advertised_capabilities,
    requires_card_scoped_pops,
    under_fed_card,
)
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.ipc.api_sessions import ApiSessions
from horde_worker_regen.process_management.jobs.job_models import APIWorkerMessage
from horde_worker_regen.process_management.jobs.job_tracker import JobFaultOrigin, JobStage, JobTracker
from horde_worker_regen.process_management.jobs.large_model_pop_governor import (
    LargeModelGovernorStatus,
    LargeModelPopGovernor,
)
from horde_worker_regen.process_management.jobs.pool_lanes import (
    LaneDecision,
    PoolLaneState,
    PoolLaneTally,
    decide_pool_lane,
    fold_pool_lane_outcome,
    record_fixed_pop_outcome,
)
from horde_worker_regen.process_management.jobs.source_image_downloader import SourceImageDownloader
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.feature_readiness import (
    CONTROLNET_ANNOTATOR_FAILED_DETAIL,
    FeatureInputs,
    GatedFeature,
    build_feature_readiness,
    is_offered,
)
from horde_worker_regen.process_management.models.model_availability import ModelAvailability
from horde_worker_regen.process_management.models.model_sizing import (
    ModelSizeTier,
    is_extra_large_model,
    model_size_tier,
)
from horde_worker_regen.process_management.resources.model_serviceability import (
    assess_model_serviceability,
    model_footprint_figures_for_baseline,
)
from horde_worker_regen.process_management.resources.resource_budget import (
    is_model_locally_unservable_for,
    predict_job_weight_mb,
)
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    WholeCardPopClaim,
    offer_under_pop_claim,
)
from horde_worker_regen.process_management.scheduling.model_pool import PopLane
from horde_worker_regen.process_management.scheduling.pop_affinity import (
    ResidencyBiasDecision,
    ResidencyBiasState,
    decide_residency_advertising,
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
from horde_worker_regen.server_capabilities import server_supports_extended_controlnet
from horde_worker_regen.telemetry_spans import queue_depth_counter, span_job_pop
from horde_worker_regen.utils.job_utils import get_single_job_magnitude, small_pop_max_power

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
    from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger
    from horde_worker_regen.process_management.lifecycle.shutdown_manager import ShutdownManager
    from horde_worker_regen.process_management.models.model_metadata import ModelMetadata

# Post-inference backpressure tuning. The safety stage sits downstream of inference and (unlike the
# pre-inference queue, bounded by queue_size) had no bound: when inference outran a slow/CPU safety
# stage the post-inference backlog grew until jobs aged past their horde ttl and were server-aborted as
# "too slow", which the horde answers with forced maintenance. The popper therefore refuses to pop while
# the backlog already represents more than a budget's worth of safety work.
POP_REQUEST_TIMEOUT_SECONDS: float = 30.0
"""Ceiling on a single job-pop HTTP request.

The pop loop is a single coroutine, so an unanswered request silences every intake decision the worker
makes until it returns. The shared client session is deliberately left without a session-wide timeout so
submit and upload behaviour is unchanged; the ceiling is applied per await instead. A pop is cheap to
retry and the loop re-issues one on the next tick, so a bound well under the transport default costs
nothing and keeps an unresponsive peer from stalling intake."""

SOURCE_IMAGE_DOWNLOAD_TIMEOUT_SECONDS: float = 120.0
"""Ceiling on fetching one job's source media.

The job is already popped and its ttl clock is running when the download starts, so waiting indefinitely
on a slow or dead media host both wastes the job and blocks the pop loop behind it. On expiry the job is
faulted for whatever media is still missing and proceeds, which is what an exhausted retry loop inside
the downloader produces."""

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
_SAFETY_BACKLOG_RELEASE_FRACTION = 0.5
"""Fraction of the self-tuning backlog cap the backlog must drain below before intake resumes.

The engage bound and the release bound differ (hysteresis): backpressure engages when the backlog reaches
the cap, then stays engaged until the backlog has drained to half the cap, rather than releasing the instant
one job clears. Without the gap, a backlog sitting at the cap would re-admit one job on every safety
completion and re-engage on the next inference completion, popping the worker in and out of backpressure each
tick (thrash) and defeating the purpose of letting the slow stage catch up. Half the cap gives the safety
stage a full margin to work down before more inference work is admitted, and because it is a fraction of the
same deadline-derived cap it tracks that cap as measured safety speed and the horde ttl move."""

_SERVER_FORCED_MAINTENANCE_MARKER = "dropping too many jobs"
"""Phrase in the horde's maintenance reason that identifies maintenance the horde imposed on the worker.

The horde returns the same return code whichever side set maintenance and carries the reason as free text,
so the reason is the only discriminator available: the horde writes its own reason when it pauses a worker
for dropping too many jobs, where an owner-set pause carries the owner's (or the default owner) message. The
worker treats a reason it does not recognise as operator intent and leaves it alone, so a phrasing change on
the horde costs an auto-clear, never an unwanted one.
"""


def _is_server_forced_maintenance(message_lower: str) -> bool:
    """Return whether a maintenance pop-rejection reason is one the horde imposed on the worker itself."""
    return _SERVER_FORCED_MAINTENANCE_MARKER in message_lower


_POST_PROCESSING_OFFER_COMMITMENT_LIMIT = 2
"""Accepted post-processing chains at which the next pop stops advertising post-processing.

This is offer shaping, not intake backpressure. The inference queue can keep accepting ordinary image jobs
while the single post-processing lane catches up; only the capability that would add more downstream work is
temporarily withheld.
"""

_MAX_CONCURRENT_LORA_JOBS = 2
"""Maximum LoRA-bearing jobs retained in the local inference queue.

The process-count reserve still lowers this ceiling on small pools so at least one process-sized share of
queue capacity remains available for non-LoRA work. Wide pools must not scale LoRA intake without bound:
auxiliary downloads are serialized and can occupy concurrency slots without feeding the GPU.
"""

_MODEL_POP_QUEUE_CAP = 2
"""Concurrently-queued jobs (running plus queued) a single non-resident model may hold before the popper
stops advertising it. Caps how much of the shallow local queue any one cold model can occupy."""

_RESIDENT_MODEL_POP_QUEUE_CAP = 3
"""Raised per-model queue cap for a model currently resident on a sampler slot. A resident model can absorb
one more queued job than a cold one because it runs without a swap, so keeping the card fed with resident
work is cheaper than admitting a cold model that forces an unload+stage+preload."""


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
    model_metadata: ModelMetadata | None = None,
    admission_baseline_provider: Callable[[int | None], float | None] | None = None,
    serviceability_logged: set[str] | None = None,
) -> set[str] | None:
    """Choose which models to include in a pop request.

    Args:
        bridge_data: The global worker config (for stickiness, custom models, and the unservable breaker).
        process_map: The live process map (for loaded/free-model stickiness).
        job_tracker: The job tracker (for the one-running-plus-one-queued cap and the unservable streak).
        max_inference_processes: The provisioned inference-process ceiling.
        last_pop_had_no_jobs: Whether the previous pop returned nothing (relaxes stickiness).
        model_availability: When provided, drops models not yet on disk.
        configured_models: The candidate model set in the current safe offer scope. When None it defaults
            to the global ``image_models_to_load``, byte-identical to the legacy single-GPU behaviour.
        card_runtimes: The per-card runtime plan. On a multi-GPU host a model is held back as unservable only
            when it is unservable on *every* card that serves it (so a model fine on a big card keeps being
            advertised); when None or single-card the worker-wide streak decides, as before.
        model_metadata: Loaded model reference metadata for baseline lookups. When unavailable,
            serviceability abstains.
        admission_baseline_provider: Source of the shared device baseline (MB) per card. A missing baseline
            reads as zero so a quiet baseline that has not yet been captured does not falsely de-list models.
        serviceability_logged: Mutable set of log keys already emitted for serviceability exclusions.

    Returns:
        A set of model names, or ``None`` if no models are eligible (caller should skip the pop).
    """
    configured = set(bridge_data.image_models_to_load) if configured_models is None else set(configured_models)
    models = set(configured)

    # Never advertise a model that is not on disk: a job for it would be popped only to fault when the
    # inference process cannot find the checkpoint. While availability is unknown (no download process)
    # this is a no-op, preserving the behaviour of workers that pre-download everything.
    if model_availability is not None:
        ready_custom_models = set(bridge_data.custom_model_ready_names).intersection(configured)
        models = model_availability.filter_present(models).union(ready_custom_models)

    loaded_models = {
        process.loaded_horde_model_name
        for process in process_map.values()
        if process.loaded_horde_model_name is not None
    }

    # The fixed model pool supersedes probabilistic stickiness: when the pool is enabled its advertising
    # lanes own the narrowing, so the deprecated stickiness roll must not run (nor emit its logs).
    stickiness_superseded = bridge_data.model_pool.enabled
    if (
        not stickiness_superseded
        and len(configured) > max_inference_processes
        and len(loaded_models) == max_inference_processes
    ):
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

    # Cap how many jobs any one model may hold in the shallow local queue. A resident model gets a higher
    # cap than a cold one: it runs without a swap, so keeping the card fed with resident work is cheaper
    # than admitting a cold model that forces an unload+stage+preload.
    models_to_remove = {
        model
        for model, count in collections.Counter(
            [job.model for job in job_tracker.jobs_pending_inference],
        ).items()
        if count >= (_RESIDENT_MODEL_POP_QUEUE_CAP if model in loaded_models else _MODEL_POP_QUEUE_CAP)
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

    serviceability_held_back = _serviceability_held_back_models(
        models,
        card_runtimes=card_runtimes,
        model_metadata=model_metadata,
        admission_baseline_provider=admission_baseline_provider,
        serviceability_logged=serviceability_logged,
    )
    if serviceability_held_back:
        models = models.difference(serviceability_held_back)

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


_ADVERTISED_MODELS_LOG_LIMIT = 12
"""Model count above which the per-pop advertised-offer line is summarised rather than listed in full."""


def _describe_advertised_models(models: set[str]) -> str:
    """Render the advertised model set for the per-pop log line.

    A short offer is listed in full. A long one is summarised as its size, a stable digest of the whole set,
    and a sample, which keeps the line readable on a wide worker while still letting two pops be compared and
    an unexpected entry (a blank name among them) be spotted. Names are quoted so a blank or whitespace-only
    entry is visible rather than rendering as nothing.
    """
    ordered = sorted(models)
    if len(ordered) <= _ADVERTISED_MODELS_LOG_LIMIT:
        return f"{len(ordered)} {[repr(name) for name in ordered]}"
    digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()[:12]
    sample = [repr(name) for name in ordered[:_ADVERTISED_MODELS_LOG_LIMIT]]
    return f"{len(ordered)} (sha256:{digest}) sample={sample}"


_MAX_IDLE_FILL_RUNG = 3
"""Highest idle-fill ladder index: the four rungs are (light, small), (light, large), (heavy, small),
(heavy, large). The escalation counter is capped here; the shaping helper re-clamps to the concrete rung
count for a worker that has no model for some tier."""


def _baseline_value_for_model(model_metadata: ModelMetadata | None, model_name: str) -> str | None:
    """Return a model's baseline value, or None when metadata is unavailable."""
    if model_metadata is None:
        return None
    baseline = model_metadata.get_baseline(model_name)
    return baseline.value if isinstance(baseline, KNOWN_IMAGE_GENERATION_BASELINE) else baseline


def _serviceability_held_back_models(
    models: set[str],
    *,
    card_runtimes: dict[int, CardRuntime] | None,
    model_metadata: ModelMetadata | None,
    admission_baseline_provider: Callable[[int | None], float | None] | None,
    serviceability_logged: set[str] | None,
) -> set[str]:
    """Return models whose minimum footprint cannot fit any card in this pop scope."""
    if model_metadata is None or card_runtimes is None or len(card_runtimes) == 0:
        return set()

    held_back: set[str] = set()
    for model in models:
        serving_cards = [card for card in card_runtimes.values() if model in set(card.config.image_models_to_load)]
        if not serving_cards:
            continue
        baseline = _baseline_value_for_model(model_metadata, model)
        figures = model_footprint_figures_for_baseline(baseline)
        verdicts = []
        for card in serving_cards:
            baseline_mb = (
                admission_baseline_provider(card.device_index) if admission_baseline_provider is not None else None
            )
            verdicts.append(
                assess_model_serviceability(
                    total_vram_mb=card.total_vram_mb,
                    baseline_mb=0.0 if baseline_mb is None else baseline_mb,
                    noise_buffer_mb=None,
                    figures=figures,
                ),
            )
        if any(verdict.serviceable for verdict in verdicts):
            continue
        held_back.add(model)
        if serviceability_logged is None:
            continue
        key = f"{model}:{','.join(str(card.device_index) for card in serving_cards)}"
        if key in serviceability_logged:
            continue
        serviceability_logged.add(key)
        arithmetic = "; ".join(
            f"device {card.device_index}: {verdict.reason()}"
            for card, verdict in zip(serving_cards, verdicts, strict=True)
        )
        logger.info(f"Not offering unserviceable model {model}: {arithmetic}")
    return held_back


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
    _post_processing_lane_commitments_provider: Callable[[], int]
    _extended_controlnet_ready_provider: Callable[[], bool]
    _post_processing_lane_paused_provider: Callable[[], bool]
    _safety_off_gpu_provider: Callable[[], bool]
    _vram_pressure_provider: Callable[[], bool]

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
        whole_card_pop_claim: Callable[[], WholeCardPopClaim | None] | None = None,
        whole_card_pop_outcome: Callable[..., None] | None = None,
        admission_baseline_provider: Callable[[int | None], float | None] | None = None,
        post_processing_lane_commitments_provider: Callable[[], int] | None = None,
        extended_controlnet_ready_provider: Callable[[], bool] | None = None,
        post_processing_lane_paused_provider: Callable[[], bool] | None = None,
        safety_off_gpu_provider: Callable[[], bool] | None = None,
        vram_pressure_provider: Callable[[], bool] | None = None,
        staged_models_provider: Callable[[], frozenset[str]] | None = None,
        action_ledger: ActionLedger | None = None,
        on_job_popped: Callable[[ImageGenerateJobPopResponse], None] | None = None,
        background_downloads_enabled: bool = True,
        pool_active_seats_provider: Callable[[], frozenset[str]] | None = None,
        pool_pop_outcome_sink: Callable[..., None] | None = None,
        quarantined_models_provider: Callable[[], frozenset[str]] | None = None,
    ) -> None:
        """Initialize with all required dependencies for job popping.

        When `dry_run_skip_api` is set, jobs come from `canned_job_source` instead of
        the live API; if no source is given, an endlessly-cycling default is used.

        When `model_availability` is provided, only models present on disk are advertised in
        pop requests (a missing model would otherwise be popped and then fault).

        When `card_runtimes` has more than one card, equivalent card offers are safely unioned. Heterogeneous
        offers rotate through one card at a time because the API cannot preserve correlations between models,
        features, policy and resolution in one union. A single card (or None) advertises the global config.

        `whole_card_residency_active` is queried by the large-model re-entry cooldown to know whether a
        whole-card residency lease is still held; it defaults to "never held" so a worker wired without it
        (and the tests) behaves as if no lease is ever active.

        `whole_card_pop_claim` returns the residency's standing claim over the offer, which narrows this
        cycle's advertised models to the resident one; `whole_card_pop_outcome` reports back whether the
        attempt made under that claim was answered with work, which is the evidence that ends a claim whose
        model the horde has nothing for. Both default to inert, so a worker wired without them advertises
        exactly as it always has.

        `admission_baseline_provider` supplies the shared device baseline for model serviceability offer
        shaping. A missing provider or uncaptured baseline reads as zero and only defers the exclusion until
        the arithmetic is known.

        `post_processing_lane_commitments_provider` reports non-image work that also occupies the dedicated
        post-processing lane (currently graph alchemy). It is added to image-job post-processing commitments
        for offer shaping, so the worker does not accept more image post-processing work while the shared lane
        is already committed elsewhere.

        `extended_controlnet_ready_provider` reports whether the extended controlnet annotators are servable
        right now. It is ANDed with the operator's `extended_controlnet` flag to decide the per-pop
        `allow_extended_controlnet` offer, so a fresh install advertises extended only once its annotators
        land. It defaults to "never ready", keeping the offer fail-closed when a popper is wired without it.

        `post_processing_lane_paused_provider` reports whether the dedicated post-processing lane is currently
        held off the GPU. A paused lane cannot run an upscale or face-fix, so the pop stops advertising
        post-processing while it is down. It defaults to "never paused", which is the truth for a worker wired
        without a lifecycle manager (and for the tests).

        `safety_off_gpu_provider` reports whether resource governance currently holds the safety process off
        its card. It only shapes the safety-backlog diagnostic's advice, which otherwise tells an operator who
        already set `safety_on_gpu` to set it. It defaults to "on the card", the truth for a worker wired
        without a lifecycle manager.

        `vram_pressure_provider` reports whether every governed card is at or below the device-free governor's
        PRESSURE floor. It gates the very-large-model offer narrowing, which stops advertising models that want
        the whole card while no card has room for one. It defaults to "never under pressure", so a popper wired
        without it advertises the full configured set.

        `staged_models_provider` reports the models staged in RAM (loadable without a fresh download/stage).
        It widens the residency-bias floor beyond the VRAM-resident set so a narrowed pop can also offer work
        the card can start cheaply from RAM. It defaults to "nothing staged", so the floor is the resident set
        alone (still safe: a narrowed offer never empties, falling back to the full offered set).

        `action_ledger`, when provided, records the edge-triggered engage/release of residency advertising
        narrowing for offline diagnosis. It defaults to None (in-memory logging only), so a directly
        constructed popper (and the tests) records no ledger events.

        `pool_active_seats_provider` reports the models the fixed model pool currently seats. When it is
        provided, the pool is enabled, and it reports a non-empty seat set, each non-idle-fill pop is routed
        through a fixed/free advertising lane (see
        :func:`~horde_worker_regen.process_management.jobs.pool_lanes.decide_pool_lane`). It defaults to None,
        so a popper wired without it (and the tests) behaves exactly as before, leaving model selection and the
        residency-bias call untouched. This byte-identical no-provider path is a hard regression contract.

        `pool_pop_outcome_sink` receives each pool-routed pop's outcome (its lane, advertised set, popped model
        name or None, whether that model was already resident, and a monotonic timestamp), matching
        :meth:`~horde_worker_regen.process_management.scheduling.model_pool.ModelPool.on_pop_outcome`. It is
        called only for cycles the pool actually routed, never for pool-disabled or idle-fill pops. It defaults
        to None (no outcome reporting).

        `quarantined_models_provider` reports the models the lifecycle manager has taken out of rotation for
        repeatedly killing the slots they are dispatched to. They come off the offer so the horde stops
        assigning their jobs, which is what ends the drop stream a quarantine would otherwise keep feeding. It
        defaults to "nothing quarantined", so a popper wired without it (and the tests) advertises as before.
        """
        self._state = state
        self._process_map = process_map
        self._job_tracker = job_tracker
        self._shutdown_manager = shutdown_manager
        self._runtime_config = runtime_config
        self._api_sessions = api_sessions
        self._card_runtimes = card_runtimes if card_runtimes is not None else {}
        self._model_metadata = model_metadata
        self._admission_baseline_provider = admission_baseline_provider
        self._serviceability_exclusion_logged: set[str] = set()
        self._whole_card_residency_active = (
            whole_card_residency_active if whole_card_residency_active is not None else (lambda: False)
        )
        self._whole_card_pop_claim = whole_card_pop_claim if whole_card_pop_claim is not None else (lambda: None)
        self._whole_card_pop_outcome = whole_card_pop_outcome
        self._post_processing_lane_commitments_provider = (
            post_processing_lane_commitments_provider
            if post_processing_lane_commitments_provider is not None
            else (lambda: 0)
        )
        # Fail-closed: a popper wired without an extended-readiness provider (and the tests) never advertises
        # the extended controlnet types, so an old bridge cannot be tricked into offering them.
        self._extended_controlnet_ready_provider = (
            extended_controlnet_ready_provider if extended_controlnet_ready_provider is not None else (lambda: False)
        )
        # Both default to the benign reading, so a popper wired without them advertises exactly as before: the
        # lane is not paused and no card is under VRAM pressure.
        self._post_processing_lane_paused_provider = (
            post_processing_lane_paused_provider
            if post_processing_lane_paused_provider is not None
            else (lambda: False)
        )
        self._safety_off_gpu_provider = (
            safety_off_gpu_provider if safety_off_gpu_provider is not None else (lambda: False)
        )
        self._vram_pressure_provider = (
            vram_pressure_provider if vram_pressure_provider is not None else (lambda: False)
        )
        self._staged_models_provider = (
            staged_models_provider if staged_models_provider is not None else (lambda: frozenset())
        )
        self._quarantined_models_provider = (
            quarantined_models_provider if quarantined_models_provider is not None else (lambda: frozenset())
        )
        # Latch for the edge-triggered warning about a quarantine exclusion that would have emptied the offer.
        self._quarantine_offer_floor_held = False
        self._action_ledger = action_ledger
        # Duty-cycle state for residency-biased advertising, advanced once per built pop request. The
        # narrowing latch is the edge-log/ledger anchor (only offer-narrowing transitions are surfaced), and
        # the last-offered count is the readable status snapshot.
        self._residency_bias_state = ResidencyBiasState()
        self._residency_bias_narrowing = False
        self._residency_bias_offered_count = 0
        # Fixed-pool advertising: the seat provider and outcome sink are inert by default, so a popper wired
        # without them is byte-identical to the pre-pool worker. The lane state carries the fixed/free
        # interleave across pops; the narrowing latch anchors the edge log/ledger; the last-fixed count and
        # most recently routed lane are readable status snapshots.
        self._pool_active_seats_provider = pool_active_seats_provider
        self._pool_pop_outcome_sink = pool_pop_outcome_sink
        self._pool_lane_state = PoolLaneState()
        self._pool_lane_tally = PoolLaneTally()
        self._pool_lane_narrowing = False
        self._pool_last_fixed_seat_count = 0
        self._pool_lane_this_cycle: PopLane | None = None
        self._pool_last_routed_lane: PopLane | None = None
        # Heterogeneous card offers rotate deterministically when no queue imbalance names a more urgent card.
        # The cursor is local scheduling state; it does not alter or duplicate any capability facts.
        self._card_scoped_pop_cursor = 0
        self._large_model_pop_governor = LargeModelPopGovernor()
        # Edge-log anchor for the VRAM-pressure offer narrowing, so a steady narrow (or steady full) offer is
        # never re-logged pop after pop.
        self._vram_pressure_narrowing = False
        # Notified once per popped job. With background downloads enabled this is the aux-prefetch
        # coordinator's pop trigger (place the job's LoRAs/TIs on disk while it is still pending); without
        # them the manager wires a guard that faults auxiliary-bearing jobs immediately, since nothing could
        # ever prepare one. The no-op default is for tests that construct a popper directly.
        self._on_job_popped = on_job_popped if on_job_popped is not None else (lambda _job: None)
        # LoRAs are placed on disk only by the dedicated background download process, so a worker running
        # without it can never prepare a LoRA job. Advertising LoRA support then would only pop jobs that can
        # never be fed. Defaults True so a popper constructed directly (and the tests) keeps advertising.
        self._background_downloads_enabled = background_downloads_enabled

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

        # Last (effective allow_lora, withholding cause) actually logged, so the per-pop advertising line is
        # edge-triggered: it fires only when the outgoing capability (or the reason it is withheld) changes,
        # never once per pop at steady state.
        self._last_logged_lora_advertise: tuple[bool, str] | None = None

    @property
    def _max_concurrent_inference_processes(self) -> int:
        """The live concurrent-inference cap (effective ``max_threads``) advertised to the API."""
        return self._runtime_config.effective_max_threads

    @property
    def _multi_gpu_advertise(self) -> bool:
        """Whether the worker shapes offers from more than one card runtime."""
        return len(self._card_runtimes) > 1

    def _advertised_capabilities(self) -> AdvertisedCapabilities | None:
        """Return the canonical card-profile union, or None before a card plan exists.

        A single-card plan reduces to that card's effective configuration. Returning the same canonical
        envelope for one or several cards keeps pop shaping and post-pop routing on one feature seam. See
        :func:`~horde_worker_regen.process_management.gpu.gpu_pop_shaping.advertised_capabilities`.
        """
        if not self._card_runtimes:
            return None
        return advertised_capabilities(self._card_runtimes)

    def _next_card_scoped_pop(self) -> int:
        """Return the next card in the stable fair rotation for a heterogeneous offer."""
        card_indices = sorted(self._card_runtimes)
        if not card_indices:
            raise RuntimeError("A card-scoped pop requires at least one card runtime.")
        selected = card_indices[self._card_scoped_pop_cursor % len(card_indices)]
        self._card_scoped_pop_cursor = (self._card_scoped_pop_cursor + 1) % len(card_indices)
        return selected

    def _gpu_pop_balance_threshold(self, bridge_data: reGenBridgeData) -> float:
        """The configured fraction of held work a card must be unable to serve before a pop targets it.

        Read tolerantly (a non-numeric mocked value falls back to 0.5) and clamped to ``[0, 1]``.
        """
        raw = bridge_data.gpu_pop_balance_threshold
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return 0.5
        return max(0.0, min(1.0, float(raw)))

    def _targeted_under_fed_card(self, bridge_data: reGenBridgeData) -> int | None:
        """The under-fed card this pop should be scoped to, or None for the topology's normal strategy.

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

    def _apply_idle_fill_ladder(
        self,
        models: set[str],
        pop_max_power: int,
        bridge_data: reGenBridgeData,
    ) -> tuple[set[str], int]:
        """Shape a fill pop to the current idle-fill rung: a smallest-fastest-first, size-narrowed slice.

        Groups the already-serviceable offered models by size tier (LIGHT = sd15/sd2, HEAVY = sdxl; the
        whole-card EXTRA_LARGE tier can never quick-start, so it is dropped) and builds a smallest-fastest-first
        rung list -- (light, small), (light, large), (heavy, small), (heavy, large) -- skipping any rung whose
        group the worker has no model for. The current ``idle_fill_rung`` (clamped to the concrete rung count)
        selects the offered subset and its max-power cap. Falls back to a flat small offer when model metadata
        is unavailable (baselines would all read light) or no light/heavy model is configured, so the fill
        degrades to a single small pop rather than mislabelling a heavy model as light. The caller sets
        ``allow_lora=False``.
        """
        small_cap = min(
            pop_max_power,
            small_pop_max_power(
                high_performance_mode=bool(bridge_data.high_performance_mode),
                moderate_performance_mode=bool(bridge_data.moderate_performance_mode),
            ),
        )
        large_cap = pop_max_power

        if self._model_metadata is None:
            return models, small_cap

        light: set[str] = set()
        heavy: set[str] = set()
        for model in models:
            tier = model_size_tier(model, self._baseline_value_for(model))
            if tier == ModelSizeTier.LIGHT:
                light.add(model)
            elif tier == ModelSizeTier.HEAVY:
                heavy.add(model)
            # EXTRA_LARGE (Flux/Cascade/Qwen/...) wants the whole card and can never quick-start; drop it.

        rungs: list[tuple[set[str], int]] = []
        if light:
            rungs.append((light, small_cap))
            rungs.append((light, large_cap))
        if heavy:
            rungs.append((heavy, small_cap))
            rungs.append((heavy, large_cap))

        if not rungs:
            return models, small_cap

        rung_index = min(self._state.idle_fill_rung, len(rungs) - 1)
        rung_models, rung_cap = rungs[rung_index]
        return rung_models, rung_cap

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

    def _apply_vram_pressure_model_narrowing(self, models: set[str]) -> set[str]:
        """Drop whole-card (EXTRA_LARGE) models from the offer while every governed card is under VRAM pressure.

        A model that wants the whole card cannot be hosted on a card already below the device-free governor's
        soft floor, so advertising it only earns work that waits on a reclaim that may never come. The pressure
        reading is the governor's own debounced state (two agreeing samples), so the offer does not flap on a
        single dip.

        Two rails bound the narrowing:

        - It never applies while the worker holds no work locally. A worker configured only with whole-card
          models would otherwise offer nothing, be sent nothing, and so never generate the activity that would
          relieve the pressure it is waiting on.
        - It never returns an empty set. When every offered model is whole-card, the full set is offered
          unchanged for the same reason.
        """
        if not self._vram_pressure_provider():
            self._log_vram_pressure_narrowing_edge(narrowing=False, withheld_count=0)
            return models
        if self._job_tracker.num_jobs_total == 0:
            self._log_vram_pressure_narrowing_edge(narrowing=False, withheld_count=0)
            return models
        narrowed = {model for model in models if not self._is_large_model(model)}
        if not narrowed:
            self._log_vram_pressure_narrowing_edge(narrowing=False, withheld_count=0)
            return models
        self._log_vram_pressure_narrowing_edge(narrowing=True, withheld_count=len(models) - len(narrowed))
        return narrowed

    def _apply_quarantine_model_exclusion(self, models: set[str]) -> set[str]:
        """Drop quarantined models from the offer, unless doing so would leave nothing to advertise.

        A quarantined model kills the slot it is dispatched to, so every job the horde sends for it is
        faulted. Continuing to advertise it therefore keeps the drop stream running until the horde force-sets
        maintenance for "dropping too many jobs", which is the failure the quarantine exists to prevent; the
        offer is where that loop is actually cut.

        The exclusion never empties the offer. A worker whose only configured model is quarantined would
        otherwise advertise nothing, be sent nothing, and never reach the work that could clear the
        quarantine, so it keeps offering the model and takes the faults instead of going permanently silent.
        """
        quarantined = self._quarantined_models_provider()
        if not quarantined:
            self._quarantine_offer_floor_held = False
            return models
        remaining = models.difference(quarantined)
        if not remaining:
            if not self._quarantine_offer_floor_held:
                self._quarantine_offer_floor_held = True
                logger.warning(
                    f"Every offered model is quarantined ({sorted(models & quarantined)}); still advertising "
                    "them because a worker that offers nothing is sent nothing and can never recover. Their jobs "
                    "will keep faulting until the quarantine clears or the configuration changes.",
                )
            return models
        self._quarantine_offer_floor_held = False
        withheld = models - remaining
        if withheld:
            logger.debug(f"Not popping quarantined models: {sorted(withheld)}")
        return remaining

    def _log_vram_pressure_narrowing_edge(self, *, narrowing: bool, withheld_count: int) -> None:
        """Emit the edge-triggered engage/release line for the VRAM-pressure offer narrowing."""
        if narrowing == self._vram_pressure_narrowing:
            return
        self._vram_pressure_narrowing = narrowing
        if narrowing:
            logger.info(
                f"VRAM-pressure offer narrowing engaged: withholding {withheld_count} whole-card model(s) from "
                "the pop offer while every governed card sits below the device-free governor's soft floor.",
            )
            return
        logger.info("VRAM-pressure offer narrowing released: advertising the whole-card models again.")

    def _resident_model_names(self) -> frozenset[str]:
        """The models currently resident on a sampler slot (loaded in VRAM on some inference process)."""
        return frozenset(
            process.loaded_horde_model_name
            for process in self._process_map.values()
            if process.process_type is HordeProcessType.INFERENCE
            and process.is_process_alive()
            and process.loaded_horde_model_name is not None
        )

    def _residency_swap_backlog(self, resident_models: frozenset[str]) -> bool:
        """Whether the worker is currently paying model swaps: a queued job needs a non-resident model.

        Requires at least one resident model (nothing to bias toward on a cold start) and at least one queued
        job whose model is not resident (the direct evidence of an imminent unload+stage+preload). An empty
        local queue is not a backlog: the worker should take any work when idle rather than narrow itself.
        """
        if not resident_models:
            return False
        return any(
            job.model is not None and job.model not in resident_models
            for job in self._job_tracker.jobs_pending_inference
        )

    def _apply_residency_advertising_bias(self, models: set[str]) -> set[str]:
        """Narrow the offered set toward residents while a swap backlog persists, duty-cycled.

        Delegates the decision to :func:`decide_residency_advertising` (pure, duty-cycled, floored) and
        advances the stored duty-cycle state. Never returns an empty set when ``models`` was non-empty, and
        never adds a model the worker does not already offer.
        """
        resident = self._resident_model_names()
        staged = self._staged_models_provider()
        decision = decide_residency_advertising(
            self._residency_bias_state,
            swap_backlog=self._residency_swap_backlog(resident),
            resident_models=resident,
            staged_models=staged,
            offered_models=frozenset(models),
        )
        self._residency_bias_state = decision.next_state
        self._residency_bias_offered_count = len(decision.advertised_models)
        self._log_residency_bias_edge(decision)
        return set(decision.advertised_models)

    def _apply_whole_card_pop_claim(self, models: set[str], claim: WholeCardPopClaim | None) -> set[str]:
        """Narrow the offer to the model a whole-card residency claims the card for.

        The strongest of the offer stages and deliberately unfloored: every other model leaves the offer for
        the duration of the claim. A residency is a commitment the rest of the worker has already paid for
        (idle sibling contexts stopped, safety cycled off the card, multiple gigabytes of weights loaded), and
        every foreign job accepted while it stands forces those weights to page back to host RAM or be re-read
        from disk. The other narrowings floor themselves because they are preferences; this one is the offer
        matching what the worker has actually committed to serving.

        Returning an empty set is a real outcome rather than a fault: the claimed model was already withheld
        by an earlier stage, so there is nothing this worker should be asking for this cycle. The caller
        withholds the pop and names the gate.
        """
        return set(offer_under_pop_claim(frozenset(models), claim=claim))

    def _note_whole_card_pop_outcome(self, claim: WholeCardPopClaim | None, *, served: bool) -> None:
        """Report back what an attempt made under the residency's claim came back with.

        Only an attempt that reached the horde and was answered carries evidence: a request that errored or
        timed out says nothing about whether the claimed model has demand, so those paths report nothing and
        the run of empty answers keeps whatever length it had.
        """
        if claim is None or self._whole_card_pop_outcome is None:
            return
        self._whole_card_pop_outcome(served=served)

    def _log_residency_bias_edge(self, decision: ResidencyBiasDecision) -> None:
        """Emit the edge-triggered engage/release log line and ledger event for offer narrowing.

        Coalesced against the last narrowing state actually surfaced, so a steady narrow (or steady full)
        offer is never re-logged pop after pop; only a transition in whether the offer is narrowed fires.
        """
        if decision.narrowed_offer == self._residency_bias_narrowing:
            return
        self._residency_bias_narrowing = decision.narrowed_offer
        if decision.narrowed_offer:
            logger.info(
                "Residency advertising bias engaged: narrowing the pop offer to "
                f"{len(decision.advertised_models)} resident/staged model(s) while a model-swap backlog "
                "persists.",
            )
            self._record_residency_bias_ledger(
                LedgerEventType.RESIDENCY_ADVERTISING_NARROWED,
                reason="model_swap_backlog",
                offered_count=len(decision.advertised_models),
            )
        else:
            logger.info("Residency advertising bias released: re-advertising the full offered model set.")
            self._record_residency_bias_ledger(
                LedgerEventType.RESIDENCY_ADVERTISING_RELEASED,
                reason="backlog_cleared_or_duty_open",
                offered_count=len(decision.advertised_models),
            )

    def _record_residency_bias_ledger(
        self,
        event_type: LedgerEventType,
        *,
        reason: str,
        offered_count: int,
    ) -> None:
        """Record a residency-advertising edge to the action ledger when one is wired (never raises)."""
        if self._action_ledger is None:
            return
        self._action_ledger.record(event_type, reason=reason, detail={"offered_models": offered_count})

    def latest_residency_bias_narrowing(self) -> bool:
        """Whether the most recent pop actually narrowed its offer toward residents (status snapshot)."""
        return self._residency_bias_narrowing

    def latest_residency_bias_offered_count(self) -> int:
        """The model count advertised on the most recent residency-bias decision (status snapshot)."""
        return self._residency_bias_offered_count

    def _apply_pool_lane(
        self,
        models: set[str],
        bridge_data: reGenBridgeData,
        *,
        idle_fill_wanted: bool,
    ) -> LaneDecision | None:
        """Route this pop through a fixed/free advertising lane, or None when the pool does not run.

        The pool runs only for a non-idle-fill pop when a seat provider is wired, the pool is enabled, and it
        reports at least one seated model. It then chooses the lane (advancing the interleave state) and
        surfaces the edge log/ledger. Returns None whenever the pool does not run, leaving the offer and the
        residency-bias call untouched, so the pool-disabled path is byte-identical to the pre-pool worker.
        """
        self._pool_lane_this_cycle = None
        if idle_fill_wanted or self._pool_active_seats_provider is None or not bridge_data.model_pool.enabled:
            return None
        active_seats = self._pool_active_seats_provider()
        if not active_seats:
            return None

        # The free-lane weight must be measured against the same capacity the auto seat count resolves
        # from (the provisioned process ceiling), or a lower live concurrency cap would hand the free lane
        # capacity the seats already claim and dilute the fixed lane with churn traffic.
        decision = decide_pool_lane(
            self._pool_lane_state,
            eligible=frozenset(models),
            active_seats=active_seats,
            process_count=self._max_inference_processes,
        )
        self._pool_lane_state = decision.next_state
        self._pool_lane_this_cycle = decision.lane
        self._pool_last_routed_lane = decision.lane
        if decision.lane is PopLane.FIXED:
            self._pool_last_fixed_seat_count = len(decision.advertised)
        self._log_pool_lane_edge(decision)
        return decision

    def _report_pool_pop_outcome(self, pool_lane: LaneDecision | None, *, popped_model: str | None) -> None:
        """Report a pool-routed pop's outcome and fold fixed-lane emptiness into the interleave.

        A no-op for cycles the pool did not route (``pool_lane`` is None). Every pool-routed pop advances the
        cumulative lane tally (its lane's pop count, its matched count, and whether a returned model was already
        resident), which the status snapshot reads. A fixed-lane pop also records whether it came back empty
        (feeding the free-weight boost that yields the offer back when the fixed lane stops earning work)
        independently of the sink, then the wired sink is handed the lane, advertised set, popped model (or
        None), and a monotonic timestamp.
        """
        if pool_lane is None:
            return
        resident_hit = popped_model is not None and popped_model in self._resident_model_names()
        self._pool_lane_tally = fold_pool_lane_outcome(
            self._pool_lane_tally,
            lane=pool_lane.lane,
            fulfilled=popped_model is not None,
            resident_hit=resident_hit,
        )
        if pool_lane.lane is PopLane.FIXED:
            self._pool_lane_state = record_fixed_pop_outcome(
                self._pool_lane_state,
                was_empty=popped_model is None,
            )
        if self._pool_pop_outcome_sink is None:
            return
        self._pool_pop_outcome_sink(
            lane=pool_lane.lane,
            advertised=pool_lane.advertised,
            popped_model=popped_model,
            popped_model_was_resident=resident_hit,
            now=time.monotonic(),
        )

    def _log_pool_lane_edge(self, decision: LaneDecision) -> None:
        """Emit the edge-triggered lane-narrowing log line and ledger event.

        Coalesced against the last lane surfaced, so a steady fixed (or steady free) lane is never re-logged
        pop after pop; only a transition between the narrowed fixed lane and the wider free lane fires.
        """
        narrowing = decision.lane is PopLane.FIXED
        if narrowing == self._pool_lane_narrowing:
            return
        self._pool_lane_narrowing = narrowing
        if narrowing:
            logger.info(
                "Fixed model pool: entered the fixed advertising lane, offering "
                f"{len(decision.advertised)} seated model(s); resident-hit telemetry will show whether matches "
                "avoid a model load.",
            )
            self._record_pool_lane_ledger(
                LedgerEventType.MODEL_POOL_LANE_FIXED,
                reason="fixed_lane",
                offered_count=len(decision.advertised),
            )
        else:
            logger.info("Fixed model pool: returned to the free advertising lane, re-opening the wider offer.")
            self._record_pool_lane_ledger(
                LedgerEventType.MODEL_POOL_LANE_FREE,
                reason="free_lane",
                offered_count=len(decision.advertised),
            )

    def _record_pool_lane_ledger(self, event_type: LedgerEventType, *, reason: str, offered_count: int) -> None:
        """Record a pool-lane edge to the action ledger when one is wired (never raises)."""
        if self._action_ledger is None:
            return
        self._action_ledger.record(event_type, reason=reason, detail={"offered_models": offered_count})

    def latest_pool_lane(self) -> PopLane | None:
        """The most recent pool-routed advertising lane, retained across unrelated or idle-fill cycles."""
        return self._pool_last_routed_lane

    def latest_pool_fixed_seat_count(self) -> int:
        """The seated-model count advertised on the most recent fixed-lane pop (status snapshot)."""
        return self._pool_last_fixed_seat_count

    def latest_pool_lane_tally(self) -> PoolLaneTally:
        """The session-cumulative fixed/free lane pop and fulfillment counts (status snapshot)."""
        return self._pool_lane_tally

    def latest_pool_lane_narrowing(self) -> bool:
        """Whether the most recent pool-routed pop narrowed to the fixed lane (status snapshot)."""
        return self._pool_lane_narrowing

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
            self._state.consecutive_failed_jobs_pause_count += 1
            return True

        # A run of failed submit attempts means the remote submit endpoint is stalled: withhold pops so the
        # worker stops accepting work onto a backlog it cannot drain. This is deliberately softer than the
        # consecutive-failure pause above: the stall is remote and self-healing (the in-flight submit retries
        # are the probes; their first success clears the counter), so it neither latches the timed pause nor
        # shuts the worker down under exit_on_unhandled_faults.
        if self._state.consecutive_failed_submit_attempts >= 3:
            if not self._submit_stall_withholding:
                self._submit_stall_withholding = True
                logger.warning(
                    "Withholding job pops: the submit endpoint is not accepting finished generations "
                    f"({self._state.consecutive_failed_submit_attempts} consecutive failed submit attempts). "
                    "Pops resume as soon as a submit succeeds.",
                )
            return True

        if self._submit_stall_withholding:
            self._submit_stall_withholding = False
            logger.info("Resuming job pops: the submit endpoint is accepting generations again.")

        return False

    def _note_pop_gate(self, gate: PopGate | None) -> None:
        """Record which gate ended this pop cycle, or None when the cycle reached the API.

        The since-stamp moves only when the gate name changes, so it measures how long the current gate has
        held rather than when it was last observed. Kept to one comparison and at most two writes: this runs
        on every tick of the sub-second pop loop. The stamp is stored as a plain string so every reader of
        :attr:`WorkerState.last_pop_gate` (the sentinel, the recovery coordinator, the wire snapshots) keeps
        comparing names rather than enum identity.
        """
        name = str(gate) if gate is not None else None
        if self._state.last_pop_gate == name:
            return
        self._state.last_pop_gate = name
        self._state.last_pop_gate_since = time.time()

    def _is_queue_full(self, bridge_data: reGenBridgeData, *, extra_allowance: int = 0) -> bool:
        """Return True if the job queue already has enough jobs.

        Args:
            bridge_data: The active bridge configuration (supplies queue_size / max_threads).
            extra_allowance: Additional queue slots to tolerate beyond the configured depth. Used by the
                idle-fill ladder to admit one fill job that is expected to dispatch onto an idle sibling
                immediately rather than buffer, so the normal depth cap would otherwise strand the GPU while
                the head waits on its load.
        """
        max_jobs_in_queue = bridge_data.queue_size + 1
        if bridge_data.max_threads > 1:
            max_jobs_in_queue += bridge_data.max_threads - 1
        max_jobs_in_queue += extra_allowance
        return len(self._job_tracker.jobs_pending_inference) >= max_jobs_in_queue

    _SAFETY_BACKLOG_LOG_INTERVAL_SECONDS = 30.0
    """Minimum gap between repeats of the "withholding pops: safety backlog" line, so the sub-second pop
    loop cannot spam it while the backpressure stays engaged."""

    _safety_backlog_log_time: float = 0.0
    """Monotonic-ish wall-clock of the last safety-backlog backpressure log (throttle state)."""

    _safety_backpressure_engaged: bool = False
    """Whether safety-backlog backpressure is currently withholding pops (the hysteresis latch state).

    Engaged when the backlog reaches the cap, cleared only once it drains below the release bound (see
    :data:`_SAFETY_BACKLOG_RELEASE_FRACTION`). Holding this state between ticks is what makes the gate
    hysteretic rather than a bare threshold, so the worker does not flap in and out of backpressure while
    the backlog hovers near the cap."""

    _submit_stall_withholding: bool = False
    """Whether the pop side is currently withholding pops because the submit endpoint is stalled.

    A state-change latch so the withhold/resume decision logs only on its edges, never re-emitted each
    sub-second pop tick while the stall persists."""

    _SUBMIT_BACKLOG_MIN = 6
    """Floor on the pending-submit backlog that counts as post-inference backpressure.

    A backlog this deep means finished generations are piling up faster than the submit stage can deliver
    them, so admitting still more work only ages jobs toward their ttl. The effective threshold scales with
    concurrency (see :meth:`_submit_backlog_cap`) but never drops below this floor, so a low-concurrency
    worker still reacts before a modest backlog becomes a stampede."""

    _submit_backpressure_engaged: bool = False
    """Whether a deep pending-submit backlog is currently counted as post-inference backpressure (log latch)."""

    def _safety_backlog_advice(self) -> str:
        """Return the remediation clause for the safety-backlog diagnostic, given where safety is running.

        Telling an operator to enable ``safety_on_gpu`` is only useful advice while no card permits it. Where
        one does, the stage is either already on that card (so the bottleneck is elsewhere) or resource
        governance has moved it off to protect the card's memory, and in that case the backlog is the cost of
        that placement rather than a configuration mistake. A popper wired without card runtimes reads the
        global flag, which is the same answer on the single-GPU worker that has no per-card deltas.
        """
        safety_permitted = (
            bool(safety_permitted_card_indices(self._card_runtimes))
            if self._card_runtimes
            else bool(self._runtime_config.bridge_data.safety_on_gpu)
        )
        if not safety_permitted:
            return "enable safety_on_gpu or speed safety up"
        if self._safety_off_gpu_provider():
            return (
                "note that safety_on_gpu is set but resource governance is holding safety off its card for "
                "memory headroom, so the check is running on the CPU; give the card more room (a smaller "
                "max_power, fewer threads, or a larger card) or speed safety up"
            )
        return "speed safety up (safety_on_gpu is already set), or reduce inference throughput"

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

    def _submit_backlog_cap(self) -> int:
        """How deep the pending-submit backlog may grow before it counts as post-inference backpressure.

        Scales with the worker's concurrent-inference ceiling (a wider worker can keep more finished
        generations in flight legitimately) but never falls below :data:`_SUBMIT_BACKLOG_MIN`.
        """
        return max(self._SUBMIT_BACKLOG_MIN, 3 * self._max_threads_ceiling)

    def _is_submit_backlogged(self) -> bool:
        """Return True if finished generations are piling up unsubmitted past the backlog cap."""
        return len(self._job_tracker.jobs_pending_submit) >= self._submit_backlog_cap()

    def _is_post_inference_backlogged(self) -> bool:
        """Return True if the post-inference (safety or submit) backlog is too deep to admit more work.

        This is the backpressure the worker previously lacked: inference completions pile into the
        ungated safety queue, so a safety stage even slightly slower than inference grows that queue until
        jobs exceed their ttl and the horde aborts them as too slow. Counting the jobs already waiting for
        (or in) safety against a deadline-derived cap lets the worker stop popping before the backlog ages
        jobs out, throttling intake to the pipeline's slowest stage instead of spiralling into
        forced maintenance.

        The submit stage sits at the same tail: when the remote submit endpoint stalls, finished generations
        pile up unsubmitted just as readily, so a pending-submit backlog past :meth:`_submit_backlog_cap`
        counts as the same backpressure and withholds intake until it drains.

        Hysteretic (safety term): the gate engages when the backlog reaches the cap and stays engaged until
        the backlog drains below :data:`_SAFETY_BACKLOG_RELEASE_FRACTION` of that cap, so intake resumes only
        once the slow stage has made real headway rather than the instant a single job clears. The latch is
        deterministic in the current backlog, so the several read sites that call this each tick (the pop
        gate, the hungry check, the orchestration-intent readback) all observe the same verdict.
        """
        if self._is_submit_backlogged():
            if not self._submit_backpressure_engaged:
                self._submit_backpressure_engaged = True
                logger.warning(
                    f"Withholding job pops: {len(self._job_tracker.jobs_pending_submit)} finished generations "
                    f"are waiting to submit (cap {self._submit_backlog_cap()}) and the submit stage is not "
                    "draining them. Intake is paused so the backlog does not deepen.",
                )
            return True
        if self._submit_backpressure_engaged:
            self._submit_backpressure_engaged = False
            logger.info("Resuming job pops: the pending-submit backlog has drained.")

        backlog = len(self._job_tracker.jobs_pending_safety_check) + len(self._job_tracker.jobs_being_safety_checked)
        cap = self._max_safe_safety_backlog()
        if self._safety_backpressure_engaged:
            if backlog <= cap * _SAFETY_BACKLOG_RELEASE_FRACTION:
                self._safety_backpressure_engaged = False
            return self._safety_backpressure_engaged
        if backlog > 0 and backlog >= cap:
            self._safety_backpressure_engaged = True
        return self._safety_backpressure_engaged

    def _post_processing_commitment_depth(self) -> int:
        """Return accepted jobs that still require the post-processing lane.

        Count jobs before they reach the lane too: a burst of already-popped large PP jobs can otherwise sit in
        the inference queue and only become visible as post-processing pressure after the worker has accepted
        several more PP jobs. Graph alchemy forms share the same single lane, so an injected provider adds
        their pending/running count without coupling the popper to the alchemy coordinator.
        """
        accepted_before_lane = sum(
            1 for job in self._job_tracker.jobs_pending_inference if bool(job.payload.post_processing)
        )
        try:
            shared_lane_commitments = max(0, int(self._post_processing_lane_commitments_provider()))
        except Exception as e:  # advisory shaping must never break image popping
            logger.debug(f"Post-processing lane commitment read failed: {type(e).__name__} {e}")
            shared_lane_commitments = 0
        return (
            accepted_before_lane
            + len(self._job_tracker.jobs_pending_post_processing)
            + len(self._job_tracker.jobs_being_post_processed)
            + shared_lane_commitments
        )

    def _post_processing_offer_withheld(self) -> bool:
        """Whether the worker's post-processing self-protection withholds the capability from this pop.

        Three independent reasons, any of which stops the advertising so the worker is not handed more
        upscale/face-fix work it cannot host (which would keep faulting toward the horde's forced-maintenance):
        the reactive fault breaker (repeated unhostable peaks), the proactive headroom gate (the parent measures
        the card's free VRAM below the post-processing peak), and the dedicated lane being held off the GPU.

        The paused-lane read is derived live rather than latched. Whoever paused the lane owns its restore, so
        the offer follows the lane back up on its own; borrowing either latch would instead leave the offer
        gated on a third party clearing state it does not own.
        """
        return (
            self._state.post_processing_disabled_by_breaker
            or self._state.post_processing_withheld_for_headroom
            or self._post_processing_lane_paused_provider()
        )

    def _should_withhold_post_processing_offer(self, bridge_data: reGenBridgeData) -> bool:
        """Return whether this pop should stop advertising post-processing until the lane catches up."""
        if not bridge_data.post_processing_lane_enabled:
            return False
        return self._post_processing_commitment_depth() >= _POST_PROCESSING_OFFER_COMMITMENT_LIMIT

    @property
    def _lora_disk_permits(self) -> bool:
        """Whether the worker-wide LoRA disk guard currently permits advertising LoRA support.

        Independent of any per-card ``allow_lora`` choice: LoRA storage is one shared cache, so a full disk
        or an in-progress background download suppresses LoRA advertising for the whole worker.
        """
        if not self._background_downloads_enabled:
            # No background downloader means no path to place a job's LoRAs on disk, so LoRA support must not
            # be advertised regardless of the per-card config flag.
            return False
        if self._model_availability is not None and self._model_availability.downloader_lost:
            # The download process died and the parent has stopped restarting it: with no downloader there is
            # no path to place LoRAs on disk, exactly as if background downloads were never enabled.
            return False
        if self._state.lora_disk_exhausted:
            return False
        # Repeated ad-hoc download teardowns withhold LoRA support for an escalating window; popping
        # more LoRA jobs while the download path is failing only churns slots (see AuxDownloadBackoff).
        if self._state.lora_download_backoff.pops_suppressed(time.time()):
            return False
        return not (self._model_availability is not None and self._model_availability.background_download_active)

    def _effective_allow_lora(self, bridge_data: reGenBridgeData) -> bool:
        """Return whether this pop should advertise LoRA support (config flag and the worker-wide disk guard)."""
        return bool(bridge_data.allow_lora) and self._lora_disk_permits

    def _lora_advertise_cause(self, bridge_data: reGenBridgeData, *, idle_fill: bool, queue_cap: bool) -> str:
        """Return the first reason this pop is withholding LoRA advertising, or ``"offered"`` if it is not.

        Names the cause the forensics of a LoRA-download-backoff incident needs to read from the logs (the
        outgoing ``allow_lora`` and, when it is off, the specific gate withholding it), in priority order so
        the download backoff is called out distinctly from a plain config opt-out or a full disk.
        """
        if not bridge_data.allow_lora:
            return "config_opt_out"
        if not self._background_downloads_enabled:
            return "no_background_downloader"
        if self._model_availability is not None and self._model_availability.downloader_lost:
            return "no_background_downloader"
        if self._state.lora_disk_exhausted:
            return "disk_exhausted"
        if self._state.lora_download_backoff.pops_suppressed(time.time()):
            return "download_backoff"
        if self._model_availability is not None and self._model_availability.background_download_active:
            return "background_download_active"
        if idle_fill:
            return "idle_fill"
        if queue_cap:
            return "queue_cap"
        return "offered"

    def _log_lora_advertising(self, bridge_data: reGenBridgeData, *, pop_allow_lora: bool, idle_fill: bool) -> None:
        """Emit one edge-triggered debug line for the outgoing pop's effective LoRA advertising.

        Coalesced against the last line actually logged so an unchanged verdict is never repeated pop after
        pop; it fires only when the effective ``allow_lora`` or the reason it is withheld changes.
        """
        cause = self._lora_advertise_cause(bridge_data, idle_fill=idle_fill, queue_cap=self._lora_queue_cap_reached())
        current = (pop_allow_lora, cause)
        if current == self._last_logged_lora_advertise:
            return
        self._last_logged_lora_advertise = current
        if pop_allow_lora:
            logger.debug("Pop advertising LoRA support (allow_lora=True).")
        else:
            logger.debug(f"Pop withholding LoRA support (allow_lora=False); cause: {cause}.")

    def _lora_queue_cap_reached(self) -> bool:
        """Whether the local queue already holds the most concurrently-queued LoRA jobs we allow.

        Each LoRA job blocks its slot on an ad-hoc download before it can sample, so letting LoRA jobs
        fill every queue slot leaves no non-LoRA job for the scheduler to slip past a blocked LoRA head
        (unless all of the candidate's LoRAs are already cached). The process-count reserve keeps at least
        one slot's worth of room for a skippable non-LoRA job on small pools, while the absolute ceiling
        prevents a wide pool from accumulating an unbounded serialized-download backlog. At least one LoRA
        job remains allowed so single-process workers still serve them.
        """
        cap = min(_MAX_CONCURRENT_LORA_JOBS, max(1, self._max_inference_processes - 1))
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
                self._state.server_maintenance_latched_at = time.time()
                self._state.server_maintenance_pop_rejections = 0
                self._state.server_maintenance_forced_by_server = _is_server_forced_maintenance(message_lower)
            # Counted on every rejection, not only on the edge: the log line above fires once per episode, so
            # this counter is the worker's only measure of how much work the maintenance is costing it.
            self._state.server_maintenance_pop_rejections += 1
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

    async def _reject_malformed_pop(self, job_pop_response: ImageGenerateJobPopResponse) -> bool:
        """Hand a pop back to the horde when it carries no usable model identity; return whether it was rejected.

        A pop whose model name is absent or blank names nothing this worker (or any worker) can load. Accepted,
        it would be preloaded as a literal empty identity, fault the slot it is dispatched to, and then be
        counted against that empty string as if it were a model, poisoning the per-model incident and quarantine
        state with an identity no job can ever satisfy. The job is therefore faulted terminally at the boundary
        so the horde reissues it immediately, and the fault is attributed to the malformed pop rather than to a
        model, keeping it out of the per-model breakers.
        """
        model = job_pop_response.model
        if model is not None and model.strip():
            return False

        await self._job_tracker.record_popped_job(job_pop_response)
        self._job_tracker.handle_job_fault_now(
            job_pop_response,
            retryable=False,
            fault_reason="malformed pop: no model name",
            fault_origin=JobFaultOrigin.MALFORMED_POP,
        )
        logger.error(
            f"Popped job {job_pop_response.id_} carries no model name (got {model!r}); returning it to the horde "
            "for reissue without queueing it. This is a malformed pop response, not a model failure.",
        )
        return True

    async def _enqueue_popped_job(
        self,
        job_pop_response: ImageGenerateJobPopResponse,
    ) -> None:
        """Add a successfully popped job to the pending inference queue."""
        await self._job_tracker.record_popped_job(job_pop_response)
        # Kick off the pending-queue LoRA/TI prefetch for this job (no-op when it carries none, or when the
        # worker runs without a background download process). Never blocks the pop path on network IO.
        self._on_job_popped(job_pop_response)
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
        if self._state.workload_intake_paused:
            # Every workload flow shares this boundary (shutdown, operator/self pause, download-only hold, and
            # terminal-recovery park). Keep it centralized on WorkerState so a new flow cannot accidentally
            # accept work under a worker-wide hold.
            self._state.last_pop_no_jobs_available = False
            self._note_pop_gate(PopGate.INTAKE_PAUSED)
            return

        if self._state.ram_pressure_pop_hold:
            # Soft, pre-floor RAM hold: system RAM is approaching its danger floor (or an over-ceiling process
            # is being drained). Do not start a new job's ttl clock on work the degraded worker cannot promptly
            # serve, or it ages past its ttl in-queue and the horde aborts it as too slow. In-flight work is
            # unaffected; the hold clears as soon as RAM recovers and no process is draining.
            self._state.last_pop_no_jobs_available = False
            self._state.last_pop_skipped_reasons["ram_pressure"] = (
                self._state.last_pop_skipped_reasons.get("ram_pressure", 0) + 1
            )
            self._note_pop_gate(PopGate.RAM_PRESSURE)
            return

        self._state.last_pop_skipped_reasons.pop("ram_pressure", None)

        if self._state.gpu_torch_incompatible:
            # The installed PyTorch has no kernels for this GPU: every job would fail at the first kernel
            # launch, so never pop. Sticky for the session (a build/hardware mismatch); fixed by reinstalling.
            self._state.last_pop_no_jobs_available = False
            self._note_pop_gate(PopGate.TORCH_UNUSABLE)
            return

        if self._state.torch_build_cpu_only:
            # CPU-only torch build: image generation is impractically slow and is disabled, so this (image)
            # popper never pops. Alchemy runs on its own loop and is unaffected. This is the runtime
            # equivalent of a 'cpu' install sentinel; sticky for the session (a build fact).
            self._state.last_pop_no_jobs_available = False
            self._note_pop_gate(PopGate.TORCH_UNUSABLE)
            return

        cur_time = time.time()
        bridge_data = self._runtime_config.bridge_data

        idle_fill_wanted = self._state.wants_idle_fill_candidate
        if idle_fill_wanted:
            urgent = True

        if self._handle_consecutive_failures(bridge_data, cur_time):
            self._note_pop_gate(PopGate.CONSECUTIVE_FAILURE_PAUSE)
            return

        # Admit one extra job past the configured depth when an idle-fill is wanted: that job is expected to
        # leave the queue immediately for the idle sibling, so bounding the relaxation to a single slot keeps
        # intake from running away if it cannot be placed this cycle.
        if self._is_queue_full(bridge_data, extra_allowance=1 if idle_fill_wanted else 0):
            self._note_pop_gate(PopGate.QUEUE_FULL)
            return

        # Post-inference backpressure: if the safety stage is backed up enough that a job admitted now
        # would likely age past its ttl waiting for it, stop popping until the backlog drains. Without
        # this the worker keeps accepting work a slow (often CPU) safety stage cannot clear, the backlog
        # grows unbounded, and the horde aborts the aged jobs as too slow and forces maintenance.
        if self._is_post_inference_backlogged():
            self._state.last_pop_no_jobs_available = False
            # The hold can come from either post-inference stage; attribute the skipped reason (and any
            # prose) to the latch actually engaged, or an alert reader chases the wrong stage.
            backlog_gate = PopGate.SAFETY_BACKLOG if self._safety_backpressure_engaged else PopGate.SUBMIT_BACKLOG
            backlog_reason = str(backlog_gate)
            self._state.last_pop_skipped_reasons[backlog_reason] = (
                self._state.last_pop_skipped_reasons.get(backlog_reason, 0) + 1
            )
            self._note_pop_gate(backlog_gate)
            # Surface safety backpressure in prose, throttled so the sub-second pop loop never spams it: a
            # bundle should show pops were stopped *because the safety stage is backed up*, not merely that
            # pops stopped. Names the depth, the self-tuned cap, and the oldest waiting safety job so a
            # slow downstream stage (typically CPU safety) is unmistakable. The submit latch logs its own
            # engage/release lines, so only the safety latch needs this periodic reminder.
            now = time.time()
            if (
                self._safety_backpressure_engaged
                and (now - self._safety_backlog_log_time) >= self._SAFETY_BACKLOG_LOG_INTERVAL_SECONDS
            ):
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
                    f"stage is slower than inference; if this persists, {self._safety_backlog_advice()}.",
                )
            return

        self._state.last_pop_skipped_reasons.pop("safety_backlog", None)
        self._state.last_pop_skipped_reasons.pop("submit_backlog", None)

        # Warm-up rule: until the first job of the session has completed, don't queue
        # ahead (if we're doomed to fail with 1 job, we're doomed to fail with 2).
        if len(self._job_tracker.jobs_pending_inference) != 0 and self._job_tracker.total_num_completed_jobs == 0:
            self._note_pop_gate(PopGate.WARMUP_FIRST_JOB)
            return

        if self._process_map.get_first_available_safety_process() is None:
            self._note_pop_gate(PopGate.NO_SAFETY_PROCESS)
            return

        if self._process_map.get_first_available_inference_process() is None:
            self._note_pop_gate(PopGate.NO_INFERENCE_PROCESS)
            return

        if len(bridge_data.image_models_to_load) == 0:
            logger.error("No models are configured to be loaded, please check your config (models_to_load).")
            self._note_pop_gate(PopGate.NO_MODELS_CONFIGURED)
            await asyncio.sleep(3)
            return

        # The megapixelstep governor holds pops so large in-flight jobs can drain; an idle-fill job is small
        # by construction and fills a GPU the blocked head has left idle, so it must not be held behind the
        # very backlog it is meant to relieve.
        if not idle_fill_wanted and self._pop_throttler.should_wait_for_megapixelsteps(
            bridge_data,
        ):
            self._note_pop_gate(PopGate.MEGAPIXELSTEP_WAIT)
            return

        if not urgent and self._pop_throttler.is_pop_too_soon(self._state.last_job_pop_time):
            self._note_pop_gate(PopGate.POP_FREQUENCY_GATE)
            return

        self._state.last_job_pop_time = time.time()

        # Equivalent cards form one safe union. Heterogeneous cards cannot: the wire shape would combine
        # models/features/limits contributed by different cards into jobs no card necessarily serves.
        advertised = self._advertised_capabilities()
        advertised_card_runtimes = self._card_runtimes

        # Adaptive targeting: when the local queue is lopsided away from one card (most held work is servable
        # only by other cards), scope THIS pop to the under-fed card's capabilities. When no card is under-fed,
        # heterogeneous plans still rotate card-scoped offers; only equivalent profiles are safe to union.
        if advertised is not None:
            target_card = self._targeted_under_fed_card(bridge_data)
            target_reason = "queue imbalance"
            if target_card is None and requires_card_scoped_pops(self._card_runtimes):
                target_card = self._next_card_scoped_pop()
                target_reason = "heterogeneous offer rotation"
            if target_card is not None:
                advertised_card_runtimes = {target_card: self._card_runtimes[target_card]}
                # A popped job is not pinned to the card whose offer fetched it and carries no NSFW marker, so
                # a card-scoped offer must still carry the fleet-wide nsfw policy or an NSFW job could route
                # to a card configured SFW.
                advertised = dataclasses.replace(
                    advertised_capabilities(advertised_card_runtimes),
                    nsfw=advertised.nsfw,
                )
                logger.debug(
                    f"Card-scoped pop: selecting card {target_card} due to {target_reason}; the request carries "
                    "only combinations that card can serve.",
                )

        models = _select_models_for_pop(
            bridge_data,
            self._process_map,
            self._job_tracker,
            self._max_inference_processes,
            last_pop_had_no_jobs=self._state.last_pop_no_jobs_available,
            model_availability=self._model_availability,
            configured_models=set(advertised.models) if advertised is not None else None,
            card_runtimes=advertised_card_runtimes,
            model_metadata=self._model_metadata,
            admission_baseline_provider=self._admission_baseline_provider,
            serviceability_logged=self._serviceability_exclusion_logged,
        )
        if models is None:
            self._note_pop_gate(PopGate.NO_ELIGIBLE_MODELS)
            return

        # Stop advertising a model the lifecycle manager has taken out of rotation, so the horde stops sending
        # its jobs for the worker to fault. Floored so the offer never empties.
        models = self._apply_quarantine_model_exclusion(models)

        # Tame pathological mixed very-large-model queues: withhold a switched-to or just-drained large model
        # from this offer so the worker is not whipsawed into repeated whole-card teardowns and multi-GB
        # reloads. A no-op unless the operator configures a switch interval or re-entry cooldown.
        models = self._apply_large_model_pop_limits(models, bridge_data)
        if len(models) == 0:
            self._note_pop_gate(PopGate.LARGE_MODEL_LIMITS)
            return

        # Stop promising what the card cannot host: while every governed card is under VRAM pressure, the
        # whole-card models come off the offer. Floored so the offer never empties and skipped while the worker
        # is idle, since a whale-only worker that advertised nothing could never recover.
        models = self._apply_vram_pressure_model_narrowing(models)

        # Fixed-pool advertising lane: when the pool holds seats, route this pop through the fixed lane (the
        # seated models, so the horde returns work the card runs without a swap) or the free lane (the rest,
        # so cold demand still reaches the worker), interleaved by a weighted round-robin. A fixed-lane offer
        # is already the narrowing, so it must NOT then pass through the residency-bias floor (which could cut
        # a not-yet-resident seat from its own lane); the free lane and every pool-disabled pop keep the bias.
        pool_lane = self._apply_pool_lane(models, bridge_data, idle_fill_wanted=idle_fill_wanted)
        if pool_lane is not None:
            models = set(pool_lane.advertised)

        # Residency-biased advertising: while the worker is paying model swaps (a queued head needs a
        # non-resident model), narrow the offer toward the resident+staged set so the horde returns work the
        # card can run without a swap. Duty-cycled so the full set is periodically re-advertised, and skipped
        # entirely on an idle-fill pop, whose job is to grab the quickest available work regardless of model.
        on_fixed_lane = pool_lane is not None and pool_lane.lane is PopLane.FIXED
        if not idle_fill_wanted and not on_fixed_lane:
            models = self._apply_residency_advertising_bias(models)

        pop_nsfw = advertised.nsfw if advertised is not None else bridge_data.nsfw
        pop_threads = advertised.threads if advertised is not None else self._max_concurrent_inference_processes
        pop_max_power = advertised.max_power if advertised is not None else bridge_data.max_power
        pop_max_batch = advertised.max_batch if advertised is not None else bridge_data.max_batch
        pop_allow_img2img = advertised.allow_img2img if advertised is not None else bridge_data.allow_img2img
        pop_allow_painting = advertised.allow_inpainting if advertised is not None else bridge_data.allow_inpainting
        pop_allow_post_processing = (
            advertised.allow_post_processing if advertised is not None else bridge_data.allow_post_processing
        )
        if self._post_processing_offer_withheld():
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
        # Stop advertising LoRA support once the queue is already carrying its allowed share of LoRA jobs.
        # A LoRA job stays pending until its files finish prefetching, so capping the LoRA queue share keeps
        # non-LoRA work poppable while those jobs wait and the GPU is never starved by an all-LoRA queue.
        if pop_allow_lora and self._lora_queue_cap_reached():
            pop_allow_lora = False

        if idle_fill_wanted:
            # Idle-fill ladder: offer a no-LoRA, smallest-fastest-first slice of the models (small sd15 ->
            # large sd15 -> small sdxl -> large sdxl) so a card idled behind a download is fed the quickest
            # work the horde currently has, escalating only when it has nothing lighter.
            models, pop_max_power = self._apply_idle_fill_ladder(models, pop_max_power, bridge_data)
            pop_allow_lora = False

        # Whole-card pop claim: while a residency holds the card, ask for its model and nothing else, so the
        # horde stops sending work whose arrival would evict the weights the residency exists to keep resident.
        # Applied last, and to the idle-fill path too: a card held by a residency is not free to be filled with
        # whatever is quickest, and a residency with no work of its own releases on its own evidence instead.
        pop_claim = self._whole_card_pop_claim()
        if pop_claim is not None:
            models = self._apply_whole_card_pop_claim(models, pop_claim)
            if len(models) == 0:
                self._note_pop_gate(PopGate.WHOLE_CARD_POP_CLAIM)
                return

        # The effective allow_lora is now settled for this pop; surface it (and, when withheld, why) so a
        # LoRA-download-backoff incident is verifiable from the logs. Edge-triggered, so steady state is quiet.
        self._log_lora_advertising(bridge_data, pop_allow_lora=pop_allow_lora, idle_fill=idle_fill_wanted)

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

        if pop_allow_post_processing and self._should_withhold_post_processing_offer(bridge_data):
            pop_allow_post_processing = False

        # Extended controlnet is a dynamic, per-pop opt-in: the operator flag AND live annotator readiness
        # AND proof that the connected server understands the field. It is additionally clamped to the
        # classic offer, since a worker offering extended must always also serve the classic set (extended
        # implies legacy). Every conjunct fails closed, so a fresh install (or one talking to a server that
        # predates the feature) advertises extended only once every gate is satisfied, no restart required.
        pop_allow_extended_controlnet = (
            bridge_data.extended_controlnet is True
            and pop_allow_controlnet
            and (advertised is None or advertised.allow_extended_controlnet)
            and self._extended_controlnet_ready_provider()
            and server_supports_extended_controlnet()
        )

        # Past every gate: the offer is settled and the request is about to go out, so nothing is holding
        # pops back. Cleared before the request rather than after it, so a request that never returns leaves
        # no gate named; that pairing (no gate, no completed attempt) is what distinguishes a wedged request
        # from a held one.
        self._note_pop_gate(None)

        # The exact offer that goes out, so a later capture can tell an empty model name this worker advertised
        # apart from one the horde answered with. Without it the two are indistinguishable after the fact.
        logger.debug(f"Advertising models in pop request: {_describe_advertised_models(models)}")

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
                allow_extended_controlnet=pop_allow_extended_controlnet,
                allow_sdxl_controlnet=pop_allow_sdxl_controlnet,
                extra_slow_worker=bridge_data.extra_slow_worker,
                limit_max_steps=bridge_data.limit_max_steps,
                allow_lora=pop_allow_lora,
                amount=pop_max_batch,
            )
            if advertised is not None:
                job_pop_request = apply_image_worker_feature_flags_to_pop_request(
                    job_pop_request,
                    advertised.image_worker_features,
                )
            # Live readiness, storage pressure, server support and idle-fill policy may only narrow the static
            # canonical projection. Keeping these overrides at the final boundary makes that ordering explicit.
            job_pop_request = job_pop_request.model_copy(
                update={
                    "allow_img2img": pop_allow_img2img,
                    "allow_painting": pop_allow_painting,
                    "allow_post_processing": pop_allow_post_processing,
                    "allow_controlnet": pop_allow_controlnet,
                    "allow_extended_controlnet": pop_allow_extended_controlnet,
                    "allow_sdxl_controlnet": pop_allow_sdxl_controlnet,
                    "allow_lora": pop_allow_lora,
                },
            )

            if self._dry_run_skip_api:
                if self._canned_job_source is None:
                    raise RuntimeError("dry_run_skip_api is set but no canned job source is configured")

                job_pop_response = self._canned_job_source.next_pop_response(job_pop_request)
                if job_pop_response.id_ is not None:
                    queue_depth_counter.add(1)
            else:
                with span_job_pop(models=",".join(sorted(models))):
                    job_pop_response = await asyncio.wait_for(
                        self._api_sessions.require_horde_client_session().submit_request(
                            job_pop_request,
                            ImageGenerateJobPopResponse,
                        ),
                        timeout=POP_REQUEST_TIMEOUT_SECONDS,
                    )

            # The attempt reached the horde and got an answer of some kind, which is the proof the worker's
            # only intake path is still running end to end. Recorded before the answer is interpreted, since
            # an error response is as much an answer as a job is.
            self._state.last_pop_attempt_completed_at = time.time()

            self._process_api_messages(job_pop_response)

            if isinstance(job_pop_response, RequestErrorResponse):
                self._handle_pop_error_response(job_pop_response)
                return

        except Exception as e:
            # Name the exception type: several failure modes here (a timed-out request among them) carry an
            # empty str(), which would otherwise leave the operator with a bare, undiagnosable line.
            # A failed attempt still concluded, so it counts toward pop liveness: the popper is reaching the
            # network and hearing back, however badly. Only an attempt that never returns leaves this unset.
            self._state.last_pop_attempt_completed_at = time.time()
            failure = f"Failed to pop job (Unexpected Error): {type(e).__name__}: {e}"
            if self._pop_throttler.current_pop_frequency == self._pop_throttler._error_pop_frequency:
                logger.error(failure)
            else:
                logger.warning(failure)
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
            self._note_whole_card_pop_outcome(pop_claim, served=False)
            self._state.last_pop_no_jobs_available = True
            self._state.last_pop_skipped_reasons = skipped_reasons
            if idle_fill_wanted:
                # The horde had nothing at this rung; climb one so the next fill tick offers the
                # next-heaviest quick-start work. Clamped; the shaping helper re-clamps per worker.
                self._state.idle_fill_rung = min(self._state.idle_fill_rung + 1, _MAX_IDLE_FILL_RUNG)
            logger.info(info_string)
            self._pop_throttler.on_no_jobs_available(
                cur_time,
                # Active alchemy work counts as the worker being busy, so an alchemy-only
                # stretch does not accrue "time without jobs".
                queue_empty=(
                    len(self._job_tracker.jobs_pending_inference) == 0 and self._state.alchemy_forms_in_flight == 0
                ),
            )
            self._report_pool_pop_outcome(pool_lane, popped_model=None)
            return

        self._note_whole_card_pop_outcome(pop_claim, served=True)
        if self._state.last_pop_maintenance_mode:
            logger.info("Clearing horde maintenance latch: a new job was popped successfully.")
            self._state.server_maintenance_cleared_by_job_pop = True
        self._state.last_pop_maintenance_mode = False
        self._state.server_maintenance_latched_at = 0.0
        self._state.server_maintenance_forced_by_server = False
        self._state.server_maintenance_pop_rejections = 0
        self._replaced_due_to_maintenance = False
        self._state.last_pop_no_jobs_available = False
        self._state.last_pop_skipped_reasons = {}
        if idle_fill_wanted:
            # Fed at this rung; restart the ladder at the smallest, quickest rung for the next idle episode.
            self._state.idle_fill_rung = 0
        self._pop_throttler.on_job_popped()
        self._report_pool_pop_outcome(pool_lane, popped_model=job_pop_response.model)

        has_loras = job_pop_response.payload.loras is not None and len(job_pop_response.payload.loras) > 0
        has_post_processing = (
            job_pop_response.payload.post_processing is not None
            and len(
                job_pop_response.payload.post_processing,
            )
            > 0
        )
        logger.opt(colors=True).info(
            "<fg #a200ff>"
            "Popped job {} "
            f"({get_single_job_magnitude(job_pop_response)} eMPS) "
            "(model: {}, "
            f"batch: {job_pop_response.payload.n_iter}, "
            f"loras: {has_loras}, post_processing: {has_post_processing})"
            "</>",
            job_pop_response.id_,
            job_pop_response.model,
        )

        # Checked before any preparation work: a job with no model identity can never be dispatched, so
        # downloading its source media (and prefetching its auxiliaries) would only spend the worker's time on
        # a job that is about to be handed straight back.
        if job_pop_response.id_ is not None and await self._reject_malformed_pop(job_pop_response):
            return

        job_pop_response = self._apply_sdk_workarounds(job_pop_response)
        try:
            job_pop_response = await asyncio.wait_for(
                self._source_image_downloader.download_source_images(job_pop_response),
                timeout=SOURCE_IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.error(
                f"Source image download for job {job_pop_response.id_} did not complete within "
                f"{SOURCE_IMAGE_DOWNLOAD_TIMEOUT_SECONDS:.0f} seconds; continuing with whatever media arrived.",
            )
            await self._source_image_downloader.record_download_faults(job_pop_response)

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

        # Seed the pop-liveness clock from here rather than the epoch, so a worker that never completes an
        # attempt measures its silence from when its intake loop started.
        self._state.last_pop_attempt_completed_at = time.time()

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
