from __future__ import annotations

import asyncio
import asyncio.exceptions
import dataclasses
import os
import ssl
import sys
import threading
import time
from asyncio import CancelledError
from collections.abc import Awaitable, Callable, Mapping
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import BoundedSemaphore as BoundedSemaphore_MultiProcessing
from multiprocessing.synchronize import Lock as Lock_MultiProcessing
from multiprocessing.synchronize import Semaphore
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
import aiohttp.client_exceptions
import certifi
from aiohttp import ClientSession
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.ai_horde_clients import (
    AIHordeAPIAsyncClientSession,
    AIHordeAPISimpleClient,
)
from horde_sdk.ai_horde_api.apimodels import (
    FindUserRequest,
    ImageGenerateJobPopResponse,
    ModifyWorkerRequest,
    UserDetailsResponse,
    WorkerDetailItem,
)
from loguru import logger

from horde_worker_regen.app_state import AppStateStore, WorkerRunRecord, default_app_state_dir
from horde_worker_regen.bridge_data.beta_source import beta_aware_image_records
from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.bridge_data.gpu_config import resolve_all_effective_gpu_configs
from horde_worker_regen.capabilities import coerce_bridge_data_to_capabilities, enabled_workloads
from horde_worker_regen.consts import (
    BRIDGE_CONFIG_FILENAME,
    VRAM_HEAVY_MODELS,
)
from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.config.bridge_data_reloader import BridgeDataReloader
from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_identity import lookup_worker_by_name
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger
from horde_worker_regen.process_management.ipc.api_sessions import ApiSessions
from horde_worker_regen.process_management.ipc.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.ipc.messages import (
    AlchemyFormSpec,
    HordeProcessState,
)
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    PENDING_JOBS_IN_SNAPSHOT,
    RECENT_JOBS_IN_SNAPSHOT,
    WORK_LEDGER_ENTRIES_IN_SNAPSHOT,
    CardSnapshot,
    FeatureInfoRow,
    FeatureReadinessSummary,
    JobFeatureSummary,
    JobQueueEntry,
    OrchestrationIntentSnapshot,
    PopGovernorsSnapshot,
    PopGovernorStatus,
    PreloadAdmissionSnapshot,
    ProcessSnapshot,
    RamGovernanceSnapshot,
    RecentJobRecord,
    SchedulingGovernanceSnapshot,
    StatsSample,
    SupervisorChannel,
    SupervisorCommand,
    SupervisorControlMessage,
    SystemMemorySnapshot,
    WholeCardResidencyStatus,
    WorkerConfigSummary,
    WorkerStateSnapshot,
    WorkLedgerEntry,
    WorkLedgerStage,
)
from horde_worker_regen.process_management.jobs.alchemy_popper import (
    DEFAULT_ALCHEMY_FORMS,
    AlchemyCoordinator,
    AlchemyFormStatus,
)
from horde_worker_regen.process_management.jobs.image_coordinator import ImageGenerationCoordinator
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.jobs.job_submitter import JobSubmitter
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.owned_process_registry import OwnedProcessRegistry
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.lifecycle.process_temperature import classify_process_temperature
from horde_worker_regen.process_management.lifecycle.shutdown_manager import ShutdownManager
from horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator import WorkerRecoveryCoordinator
from horde_worker_regen.process_management.models.desired_state import DesiredState
from horde_worker_regen.process_management.models.download_coordinator import ModelDownloadCoordinator
from horde_worker_regen.process_management.models.feature_readiness import (
    CONTROLNET_ANNOTATOR_FAILED_DETAIL,
    FeatureInputs,
    GatedFeature,
    build_feature_readiness,
)
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lora_disk_guard import (
    free_mb,
    is_lora_disk_exhausted,
    read_evictable_adhoc_mb,
)
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.models.model_availability import ModelAvailability
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.resources.duty_cycle import DutyCycleSummary, summarize_duty_cycle
from horde_worker_regen.process_management.resources.resource_budget import (
    CommittedReserveLedger,
    is_model_locally_unservable_for,
)
from horde_worker_regen.process_management.resources.run_metrics import RunMetricsSnapshot, WorkerRunMetrics
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.performance_model import (
    PERF_MODEL_FILENAME,
    PerformanceModel,
    load_seed_its_by_signature,
)
from horde_worker_regen.process_management.scheduling.pop_governor_registry import (
    PopGovernorReading,
    PopGovernorRegistry,
)
from horde_worker_regen.process_management.scheduling.pop_throttler import CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS
from horde_worker_regen.process_management.scheduling.workload_flow import FlowCoordinator, WorkloadKind
from horde_worker_regen.process_management.simulation._canned_scenarios import CannedAlchemySource, CannedJobSource
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from horde_worker_regen.process_management.workers.safety_orchestrator import SafetyOrchestrator
from horde_worker_regen.reporting.kudos_logger import KudosLogger
from horde_worker_regen.reporting.maintenance_messenger import MaintenanceModeMessenger
from horde_worker_regen.reporting.status_reporter import StatusReporter
from horde_worker_regen.utils.config_coercion import config_number
from horde_worker_regen.utils.disk_monitor import DiskSpaceMonitor
from horde_worker_regen.utils.gpu_monitor import GpuUtilizationSampler
from horde_worker_regen.utils.kudos_calculator import KudosCalculator
from horde_worker_regen.utils.kudos_utils import generate_kudos_info_string as _generate_kudos_info_string

if TYPE_CHECKING:
    from horde_worker_regen.process_management.ipc.messages import HordeJobMetricsMessage
    from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
    from horde_worker_regen.process_management.jobs.job_tracker import TrackedJob
    from horde_worker_regen.process_management.resources.system_memory import SystemMemorySummary


@dataclasses.dataclass(frozen=True)
class SystemResources:
    """Hardware information detected at startup."""

    total_ram_bytes: int
    device_map: TorchDeviceMap
    per_process_overhead_mb: int = 0
    """Approx. VRAM (MB) the *first/sole* inference process consumes for its torch/CUDA context with no model
    loaded (the one-time runtime cost plus one context), measured by the accelerator probe on the idle
    device. The streaming forecast subtracts this from total VRAM to estimate the free achievable under sole
    residency. 0 when unmeasured."""
    marginal_process_overhead_mb: int = 0
    """Approx. VRAM (MB) each *additional* inference process's context costs once the first has paid the
    shared one-time runtime cost, measured by the probe's second-context delta. The forecast multiplies this
    (not the one-time-inclusive ``per_process_overhead_mb``) by the sibling count for free-after-model-evict.
    0 when unmeasured, where the forecast falls back to charging the full overhead per context."""

    @classmethod
    def detect(cls) -> SystemResources:
        """Detect system resources via psutil and hordelib's backend-agnostic accelerator inventory.

        Device discovery goes through the out-of-process accelerator probe rather than ``torch.cuda``
        directly, for two reasons. It stays backend-agnostic (every ComfyUI-supported backend (CUDA/ROCm,
        Intel XPU, Apple MPS, DirectML, CPU) populates the device map; a bare ``torch.cuda.device_count()``
        loop would yield no devices on non-CUDA backends). And it keeps this process torch-free: detect()
        runs in the long-lived orchestrator, enumerating accelerators loads torch (~500MB), so
        :func:`probe_accelerators` pays that cost in a short-lived subprocess that frees it on exit.
        """
        import psutil

        from horde_worker_regen.utils.accelerator_probe import probe_accelerators

        total_ram = psutil.virtual_memory().total

        # Pin CUDA enumeration to physical PCI-bus order so a device index maps to a fixed physical slot
        # across reboots/driver changes. With multi-GPU this is what makes gpu_device_indices/gpu_overrides
        # (and the per-card device pinning in the children, which inherit this env) refer to stable cards.
        # setdefault so a deliberate operator override still wins; the probe subprocess inherits it.
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

        accelerators = probe_accelerators()

        # Warn once if the declared compute backend (bin/backend) disagrees with what the hardware probe
        # actually found: a GPU install whose driver is broken (would run on CPU or fault), or a CPU
        # install on a box that does have a usable accelerator.
        from horde_worker_regen.compute_mode import log_compute_mode_reconciliation

        log_compute_mode_reconciliation([accelerator.kind for accelerator in accelerators])

        device_map = TorchDeviceMap(root={})
        for accelerator in accelerators:
            device_map.root[accelerator.index] = TorchDeviceInfo(
                device_name=accelerator.name,
                device_index=accelerator.index,
                total_memory=accelerator.total_vram_mb * 1024 * 1024,
                kind=accelerator.kind,
            )

        per_process_overhead_mb = max((a.runtime_overhead_mb for a in accelerators), default=0)
        marginal_process_overhead_mb = max((a.marginal_overhead_mb for a in accelerators), default=0)

        return cls(
            total_ram_bytes=total_ram,
            device_map=device_map,
            per_process_overhead_mb=per_process_overhead_mb,
            marginal_process_overhead_mb=marginal_process_overhead_mb,
        )


def _select_driven_devices(
    device_map: TorchDeviceMap,
    configured_indices: list[int] | None,
) -> TorchDeviceMap:
    """Filter the detected device map to the cards this worker should drive.

    ``configured_indices`` None means auto-detect: drive every detected accelerator. An explicit list opts
    the worker into a subset (e.g. a multi-GPU box pinned to one card); the indices are stable PCI-bus
    indices (see :meth:`SystemResources.detect`). Indices not present in the detected map are warned about
    and ignored, and a list that matches nothing falls back to all detected devices rather than leaving the
    worker with no cards. Returns the map unchanged when no accelerators were detected (CPU/dry-run/test
    paths) so those never break.
    """
    # Anything that is not an explicit list (None, or a partially-mocked bridge_data in tests) means
    # auto-detect: drive every detected device.
    if not device_map.root or not isinstance(configured_indices, list):
        return device_map
    requested = list(dict.fromkeys(configured_indices))  # de-dup, preserve operator order
    missing = [index for index in requested if index not in device_map.root]
    if missing:
        logger.warning(
            f"gpu_device_indices requested device(s) {missing} that are not present "
            f"(detected: {sorted(device_map.root)}); ignoring those.",
        )
    selected = {index: device_map.root[index] for index in requested if index in device_map.root}
    if not selected:
        logger.warning(
            f"gpu_device_indices {requested} matched no detected devices "
            f"(detected: {sorted(device_map.root)}); driving all detected devices instead.",
        )
        return device_map
    return TorchDeviceMap(root=selected)


def _resolve_inference_concurrency(
    *,
    gpu_sampling_lease_enabled: bool,
    configured_lease_slots: int | None,
    max_concurrent_inference_processes: int,
    max_inference_processes: int,
) -> tuple[int, int]:
    """Resolve ``(inference_semaphore_size, gpu_sampling_lease_slots)`` for the IPC primitives.

    Without the lease the whole-job inference semaphore is the only GPU gate, so it stays at the
    concurrent-sampling count. With the lease the *lease* becomes the denoise gate and the
    inference semaphore opens up to every inference process, letting spare processes stage their
    next pipeline (model load, prompt encode) ahead instead of blocking at job start.

    When ``configured_lease_slots`` is ``None`` (the default) the slot count tracks the
    concurrent-inference ceiling, so enabling the lease never serializes the denoise below the
    concurrency ``max_threads`` already admits. An explicit value overrides that, e.g. to pin a
    single denoise loop on time-slicing (no-MPS) hardware while still staging the next job ahead.
    Either way the slot count is clamped to ``[1, max_inference_processes]`` (at least one denoise
    may always run, and more slots than processes can never be used).
    """
    desired_slots = max_concurrent_inference_processes if configured_lease_slots is None else configured_lease_slots
    lease_slots = min(max(1, desired_slots), max(1, max_inference_processes))
    if gpu_sampling_lease_enabled:
        return max_inference_processes, lease_slots
    return max_concurrent_inference_processes, lease_slots


@dataclasses.dataclass(frozen=True)
class CardConcurrency:
    """Resolved concurrency sizes for one card, mirroring the pre-multi-GPU single-pool computation.

    On a single-GPU host this reproduces exactly the values the old global computation produced, so a
    one-card worker spawns the same process count with the same semaphore sizes as before.
    """

    target_process_count: int
    """How many inference processes this card runs (``queue_size`` + concurrency ceiling)."""
    max_concurrent_inference: int
    """This card's concurrent-sampling ceiling (the inference-semaphore size when the lease is disabled)."""
    inference_semaphore_size: int
    """Permits on this card's inference semaphore (opened up to the process count when the lease is on)."""
    vae_decode_semaphore_size: int
    """Permits on this card's VAE-decode semaphore (always 1 today)."""
    gpu_sampling_lease_slots: int
    """Concurrent denoise loops allowed on this card when the GPU sampling lease is enabled."""


def resolve_card_concurrency(
    *,
    max_threads: int,
    queue_size: int,
    num_models_to_load: int,
    gpu_sampling_lease_enabled: bool,
    gpu_sampling_lease_slots: int | None,
    max_threads_ceiling: int,
    serves_image_generation: bool = True,
) -> CardConcurrency:
    """Resolve one card's concurrency sizes from its effective config (the per-card analogue of the globals).

    Mirrors the old single-pool computation: a ceiling of ``max(max_threads, max_threads_ceiling)``, a
    process count of ``queue_size + ceiling`` collapsed to 1 for the single-model/single-thread case, and
    the lease-aware inference-semaphore size from :func:`_resolve_inference_concurrency`. Passing one card's
    effective values reproduces today's globals exactly for a single-GPU host.

    When this worker does not serve image generation (``serves_image_generation`` is false: an
    alchemist-only worker, whether by CPU install or a deliberate ``dreamer: false`` opt-out) the process
    count is forced to one. The image-generation fleet exists to run many concurrent samplers; without
    image work the only inference-process consumer is graph alchemy (upscale, face-fix), which serializes
    fine through a single process while CLIP/text forms run on the safety process. The ceiling and
    semaphore sizes are left intact so the single process is still sized correctly for what it runs.
    """
    ceiling = max(max_threads, max_threads_ceiling)
    max_concurrent = ceiling
    target_process_count = queue_size + ceiling
    if num_models_to_load == 1 and max_concurrent == 1:
        target_process_count = 1
    if not serves_image_generation:
        target_process_count = 1
    inference_semaphore_size, lease_slots = _resolve_inference_concurrency(
        gpu_sampling_lease_enabled=gpu_sampling_lease_enabled,
        configured_lease_slots=gpu_sampling_lease_slots,
        max_concurrent_inference_processes=max_concurrent,
        max_inference_processes=target_process_count,
    )
    return CardConcurrency(
        target_process_count=target_process_count,
        max_concurrent_inference=max_concurrent,
        inference_semaphore_size=inference_semaphore_size,
        vae_decode_semaphore_size=1,
        gpu_sampling_lease_slots=lease_slots,
    )


class _EstimatedContextFootprint:
    """Conservative plan-time estimates (MB) of one resident inference context's memory cost.

    Used only to right-size the per-card process count before any model reference or runtime measurement
    exists (see :func:`cap_card_process_counts`). Deliberately coarse and conservative: the measured runtime
    budget refines the *live* count downward under real pressure, so a slightly-low plan self-corrects,
    whereas an over-provisioned plan is what produces the system-RAM and post-processing-VRAM reaps. Tunable.
    """

    SDXL_CONTEXT_VRAM_MB = 7000.0
    """Resident VRAM working set of a typical SDXL/SD context (weights plus the per-context CUDA overhead)."""
    HEAVY_CONTEXT_VRAM_MB = 13000.0
    """Resident VRAM working set when a VRAM-heavy family (Flux/Cascade) is offered."""
    CONTEXT_RAM_MB = 9000.0
    """System RAM a resident context retains (CPU-side weights plus allocator retention freed only by a respawn)."""
    POST_PROCESSING_PEAK_MB = 8500.0
    """Free VRAM a card must keep for the post-processing peak that lands after sampling (a 4x SDXL upscale)."""


# Coarse system-RAM a co-hosted worker role consumes beside the image inference contexts, reserved up front so
# the resident-context count is sized against what is actually left of the shared pool. An alchemist runs its
# own CLIP/BLIP/aesthetic models in-worker; a scribe is a separate text-generation server whose weights still
# sit in the same RAM. Both are deliberately conservative floors: under-reserving is what lets the summed
# footprint (contexts + co-tenants) drive an OS OOM kill, the failure this guards.
_ALCHEMIST_CO_TENANT_RAM_BYTES = 4 * 1024 * 1024 * 1024
_SCRIBE_CO_TENANT_RAM_BYTES = 4 * 1024 * 1024 * 1024


def co_tenant_ram_reserve_bytes(bridge_data: reGenBridgeData) -> int:
    """System RAM (bytes) to keep free for co-hosted worker roles (alchemist, scribe) beside image inference.

    Raised into ``target_ram_overhead_bytes`` so the per-card process-count cap sizes resident contexts against
    the RAM the co-tenants actually take, not the whole pool. Returns 0 for an image-only worker (byte-identical
    to prior sizing). An alchemist reserve is the larger of its configured per-form RAM headroom and a floor; a
    configured scribe adds a fixed floor (its text model's size is not known here, so a coarse guard is used).
    Never raises on a bad config value.
    """
    reserve = 0
    if WorkloadKind.ALCHEMY in enabled_workloads(bridge_data):
        alchemy_headroom_mb = config_number(getattr(bridge_data, "alchemy_ram_headroom_mb", None)) or 0.0
        reserve += max(_ALCHEMIST_CO_TENANT_RAM_BYTES, int(alchemy_headroom_mb * 1024 * 1024))
    if getattr(bridge_data, "scribe_name", None):
        reserve += _SCRIBE_CO_TENANT_RAM_BYTES
    return reserve


def cap_card_process_counts(
    *,
    per_card_target_processes: Mapping[int, int],
    total_ram_bytes: int,
    target_ram_overhead_bytes: int,
    total_vram_mb_by_card: Mapping[int, float | None],
    allow_post_processing: bool,
    has_vram_heavy_models: bool,
) -> dict[int, int]:
    """Right-size each card's inference-process count to the shared RAM pool and the card's VRAM headroom.

    The resolved per-card plan (``queue_size + ceiling``) is sound per card but, summed across cards,
    double-counts the single shared system-RAM pool (a second card doubles VRAM, not RAM) and ignores the
    post-processing VRAM peak that lands after sampling. This lowers the per-card target so each card keeps
    enough free VRAM for that peak and the worker-wide resident-context count fits system RAM. It only ever
    reduces a card below its resolved plan, and never below one context per card (a card must keep at least
    one to serve at all). The measured runtime budget remains the authority that gates actual loads; this is
    a coarse front-end that keeps the spawn plan from over-committing in the first place.

    Args:
        per_card_target_processes: The resolved ``target_process_count`` per device index.
        total_ram_bytes: Total system RAM (the shared pool across all cards).
        target_ram_overhead_bytes: RAM the worker keeps free (subtracted before sizing the context count).
        total_vram_mb_by_card: Each card's total VRAM in MB, or None where capacity is unknown (then the
            per-card VRAM cap abstains for that card and only the worker-wide RAM cap applies).
        allow_post_processing: Whether the worker offers post-processing (reserves the upscale peak per card).
        has_vram_heavy_models: Whether any offered model is VRAM-heavy (Flux/Cascade), raising the per-context
            VRAM estimate.

    Returns:
        The capped ``target_process_count`` per device index (each at most its input, at least 1).
    """
    capped = dict(per_card_target_processes)
    per_context_vram_mb = (
        _EstimatedContextFootprint.HEAVY_CONTEXT_VRAM_MB
        if has_vram_heavy_models
        else _EstimatedContextFootprint.SDXL_CONTEXT_VRAM_MB
    )
    post_processing_reserve_mb = _EstimatedContextFootprint.POST_PROCESSING_PEAK_MB if allow_post_processing else 0.0

    # Per-card VRAM / post-processing cap: a card must keep the post-processing peak free on top of its
    # resident contexts. Abstain when the card's capacity is unknown (the runtime budget still gates loads).
    for device_index, total_vram_mb in total_vram_mb_by_card.items():
        if total_vram_mb is None or device_index not in capped:
            continue
        vram_for_contexts_mb = total_vram_mb - post_processing_reserve_mb
        max_contexts_for_vram = max(1, int(vram_for_contexts_mb // per_context_vram_mb))
        capped[device_index] = min(capped[device_index], max_contexts_for_vram)

    # Worker-wide RAM cap: every resident context retains RAM that no other card's VRAM offsets. Trim the
    # most-provisioned cards first (their discretionary pipelining slots), never below one context per card.
    usable_ram_mb = (total_ram_bytes - target_ram_overhead_bytes) / (1024 * 1024)
    max_total_contexts = max(len(capped), int(usable_ram_mb // _EstimatedContextFootprint.CONTEXT_RAM_MB))
    while sum(capped.values()) > max_total_contexts:
        trimmable_indices = [index for index, count in capped.items() if count > 1]
        if not trimmable_indices:
            break
        victim = max(trimmable_indices, key=lambda index: capped[index])
        capped[victim] -= 1

    return capped


@dataclasses.dataclass
class MultiprocessingPrimitives:
    """Multiprocessing primitives created for IPC.

    The GPU-concurrency gates (inference / VAE-decode / sampling-lease semaphores) are held **per card**:
    each driven GPU gets its own so one card's sampling cannot block another's. On a single-GPU host each
    map has exactly one entry keyed by index 0, sized identically to the old single semaphores. The
    process message queue, disk/aux locks, and download-bandwidth semaphore are genuinely shared across
    all cards and stay singular.
    """

    process_message_queue: ProcessQueue
    inference_semaphores: dict[int, Semaphore]
    disk_lock: Lock_MultiProcessing
    aux_model_lock: Lock_MultiProcessing
    vae_decode_semaphores: dict[int, Semaphore]
    gpu_sampling_leases: dict[int, Semaphore]
    """Per-card GPU sampling lease: serializes that card's denoising loop across its inference processes so
    they pipeline (one samples while others stage their next pipeline) rather than idling the GPU."""
    download_bandwidth_semaphore: Semaphore
    """Held by the background download process while it is actively downloading, so the parent can
    coordinate pop policy around WAN-bandwidth contention. Shared (downloads are not per-card)."""

    @classmethod
    def create(
        cls,
        ctx: BaseContext,
        *,
        per_card: dict[int, CardConcurrency],
    ) -> MultiprocessingPrimitives:
        """Create real multiprocessing primitives from a context, one semaphore set per driven card.

        The GPU-concurrency gates are BoundedSemaphores, not plain Semaphores, for a reason that is
        load-bearing for crash recovery: a child acquires these inside its own process, so when it
        dies or hangs the parent must release on its behalf or the slot's concurrency is lost forever
        (a single orphaned inference permit at ``max_threads=1`` wedges the whole worker). The parent
        cannot always know whether the dead child actually held a permit, so it releases
        unconditionally. With a plain Semaphore an over-release would silently raise the ceiling and
        admit more concurrent sampling than configured (an eventual VRAM OOM); a BoundedSemaphore
        instead rejects the over-release with ``ValueError``, making the blind release a safe no-op
        when the child held nothing. The child's own acquire/release stay paired and idempotent, so
        the bound is never hit in normal operation.
        """
        return cls(
            # ctx.Queue(), NOT multiprocessing.Queue(): a Queue carries an internal SemLock, and the
            # children are started from this (spawn) ctx. The global multiprocessing module defaults to
            # fork on Linux, so a global Queue() yields a fork-context SemLock that cannot be pickled into
            # a spawn child ("A SemLock created in a fork context is being shared with a process in a spawn
            # context"). The worker happens to dodge this only because _prepare_runtime forces the global
            # start method to spawn; the benchmark never does, so binding to ctx is the real fix.
            process_message_queue=ctx.Queue(),
            inference_semaphores={
                index: BoundedSemaphore_MultiProcessing(card.inference_semaphore_size, ctx=ctx)
                for index, card in per_card.items()
            },
            disk_lock=Lock_MultiProcessing(ctx=ctx),
            aux_model_lock=Lock_MultiProcessing(ctx=ctx),
            vae_decode_semaphores={
                index: BoundedSemaphore_MultiProcessing(card.vae_decode_semaphore_size, ctx=ctx)
                for index, card in per_card.items()
            },
            gpu_sampling_leases={
                index: BoundedSemaphore_MultiProcessing(max(1, card.gpu_sampling_lease_slots), ctx=ctx)
                for index, card in per_card.items()
            },
            download_bandwidth_semaphore=BoundedSemaphore_MultiProcessing(1, ctx=ctx),
        )


sslcontext = ssl.create_default_context(cafile=certifi.where())


# As of 3.11, asyncio.TimeoutError is deprecated and is an alias for builtins.TimeoutError
_async_client_exceptions: tuple[type[Exception], ...] = (TimeoutError, aiohttp.client_exceptions.ClientError, OSError)

if sys.version_info[:2] == (3, 10):
    _async_client_exceptions = (asyncio.exceptions.TimeoutError, aiohttp.client_exceptions.ClientError, OSError)

_caught_signal = False


class HordeWorkerProcessManager:
    """Manages and controls processes to act as a horde worker."""

    _runtime_config: RuntimeConfig
    _api_sessions: ApiSessions
    _model_metadata: ModelMetadata

    @property
    def bridge_data(self) -> reGenBridgeData:
        """The bridge data for this worker."""
        return self._runtime_config.bridge_data

    @bridge_data.setter
    def bridge_data(self, value: reGenBridgeData) -> None:
        self._runtime_config.update(value)

    horde_model_reference_manager: ModelReferenceManager | None
    """The model reference manager for this worker. None only when a pre-loaded reference was injected."""

    max_inference_processes: int
    """The maximum number of inference processes that can be active. This is not the number of jobs that
    can run at once. Use `max_concurrent_inference_processes` to control that behavior."""

    _max_concurrent_inference_processes: int
    """The provisioned concurrency *ceiling* (the size the inference semaphore was created for). \
        The live effective cap is exposed by the ``max_concurrent_inference_processes`` property."""

    @property
    def max_concurrent_inference_processes(self) -> int:
        """The live concurrent-inference cap (effective ``max_threads``), adjustable at runtime."""
        return self._runtime_config.effective_max_threads

    max_safety_processes: int
    """The maximum number of safety processes that can run at once."""

    total_ram_bytes: int
    """The total amount of RAM on the system."""

    @property
    def total_ram_megabytes(self) -> int:
        """The total amount of RAM on the system in megabytes."""
        return self.total_ram_bytes // 1024 // 1024

    @property
    def total_ram_gigabytes(self) -> int:
        """The total amount of RAM on the system in gigabytes."""
        return self.total_ram_bytes // 1024 // 1024 // 1024

    target_ram_overhead_bytes: int
    """The target amount of RAM to keep free."""

    target_vram_overhead_bytes_map: Mapping[int, int] | None = None

    @property
    def max_queue_size(self) -> int:
        """The maximum number of jobs that can be queued."""
        return self.bridge_data.queue_size

    @property
    def current_queue_size(self) -> int:
        """The current number of jobs queued for inference."""
        return self._job_tracker.current_queue_size

    @property
    def target_ram_bytes_used(self) -> int:
        """The target amount of RAM to use."""
        return self.total_ram_bytes - self.target_ram_overhead_bytes

    def get_process_total_ram_usage(self) -> int:
        """Return the total amount of RAM used by all processes."""
        total = 0
        for process_info in self._process_map.values():
            total += process_info.ram_usage_bytes
        return total

    _job_tracker: JobTracker
    """Tracks all job collections, locks, and job lifecycle state."""

    _process_lifecycle: ProcessLifecycleManager
    """Manages process start, stop, replace, and hung-process detection."""

    @property
    def num_jobs_total(self) -> int:
        """The total number of jobs across all live stages."""
        return self._job_tracker.num_jobs_total

    session_start_time: float = 0
    """The time at which the session started in epoch time."""

    @property
    def stable_diffusion_reference(self) -> dict[str, ImageGenerationModelRecord] | None:
        """The class which contains the list of models from horde_model_reference."""
        return self._model_metadata.reference

    @stable_diffusion_reference.setter
    def stable_diffusion_reference(self, value: dict[str, ImageGenerationModelRecord] | None) -> None:
        self._model_metadata.set_reference(value)

    def get_model_baseline(self, model_name: str) -> KNOWN_IMAGE_GENERATION_BASELINE | str | None:
        """Return the baseline of the model."""
        return self._model_metadata.get_baseline(model_name)

    @property
    def horde_client_session(self) -> AIHordeAPIAsyncClientSession:
        """The context manager for the horde sdk client."""
        return self._api_sessions.require_horde_client_session()

    @horde_client_session.setter
    def horde_client_session(self, value: AIHordeAPIAsyncClientSession) -> None:
        self._api_sessions.set_horde_client_session(value)

    user_info: UserDetailsResponse | None = None
    """The user info for the user that this worker is logged in as."""

    _process_map: ProcessMap
    """Shared by reference with all sub-managers. Created once; never reassigned after __init__."""

    _horde_model_map: HordeModelMap

    _device_map: TorchDeviceMap
    """A mapping (dict) of device IDs to TorchDeviceInfo objects. Contains some helper methods."""

    _loop_interval: float = 0.20
    """The number of seconds to wait between each loop of the main process (inter process management) loop."""

    _sleep: Callable[[float], Awaitable[None]]
    """Pacing sleep used by the control loop. Defaults to asyncio.sleep; tests inject a no-op
    so the loop can be driven tick-by-tick without wall-clock delays."""
    _api_get_user_info_interval = 15
    """The number of seconds to wait between each fetch of the user info."""

    _last_get_user_info_time: float = 0
    """The time at which the user info was last fetched."""

    _process_message_queue: ProcessQueue
    """A queue of messages sent from child processes."""

    _card_runtimes: dict[int, CardRuntime]
    """Per-card runtime plan keyed by stable device index: each card's effective config, concurrency
    semaphores, and process count. A single-GPU host has one entry (index 0) sized as before."""

    _disk_lock: Lock_MultiProcessing
    """A lock to prevent multiple processes from accessing the disk at once."""

    _aux_model_lock: Lock_MultiProcessing
    """A lock to prevent multiple processes from accessing the auxiliary models at once (such as LoRas)."""

    _lru: LRUCache
    """A simple LRU cache. This is used to keep track of the most recently used models."""

    _amd_gpu: bool
    """Whether or not the GPU is an AMD GPU."""

    _directml: int | None
    """ID of the potential directml device."""

    @property
    def post_process_job_overlap_allowed(self) -> bool:
        """Return true if a new inference job can start while the previous job's post-processing is still running.

        Distinct from ``allow_post_processing`` (from the SDK), which advertises
        to the horde API that this worker accepts post-processing jobs at all.
        """
        return (
            self.bridge_data.moderate_performance_mode or self.bridge_data.high_performance_mode
        ) and self.bridge_data.post_process_job_overlap

    def __init__(
        self,
        *,
        ctx: BaseContext,
        bridge_data: reGenBridgeData,
        horde_model_reference_manager: ModelReferenceManager | None,
        target_ram_overhead_bytes: int = 9 * 1024 * 1024 * 1024,
        target_vram_overhead_bytes_map: Mapping[int, int] | None = None,  # FIXME
        max_safety_processes: int = 1,
        amd_gpu: bool = False,
        directml: int | None = None,
        supervisor_connection: Connection | None = None,
        system_resources: SystemResources | None = None,
        mp_primitives: MultiprocessingPrimitives | None = None,
        skip_api_init: bool = False,
        stable_diffusion_reference: dict[str, ImageGenerationModelRecord] | None = None,
        process_entry_points: ProcessEntryPoints | None = None,
        canned_job_source: CannedJobSource | None = None,
        canned_alchemy_source: CannedAlchemySource | None = None,
        enable_background_downloads: bool = False,
        max_threads_ceiling: int | None = None,
    ) -> None:
        """Initialise the process manager.

        Args:
            ctx: The multiprocessing context to use.
            bridge_data: The bridge data for this worker.
            horde_model_reference_manager: The model reference manager for this worker. May only be None \
                when a pre-loaded `stable_diffusion_reference` is provided (bridge data reloads from disk \
                are then disabled).
            target_ram_overhead_bytes: The target amount of RAM to keep free.
            target_vram_overhead_bytes_map: The target amount of VRAM to keep free.
            max_safety_processes: The maximum number of safety processes that can run at once.
            amd_gpu: Whether or not the GPU is an AMD GPU.
            directml: ID of the potential directml device.
            supervisor_connection: When launched by a supervising frontend (the TUI), the worker's \
                end of the duplex pipe. State snapshots are pushed and control commands drained over \
                it. None for the standard headless run.
            system_resources: Pre-detected system resources. If None, auto-detects via torch/psutil.
            mp_primitives: Pre-created multiprocessing primitives. If None, creates real ones from ctx.
            skip_api_init: If True, skip the remove_maintenance API call during init.
            stable_diffusion_reference: Pre-loaded model reference. If None, fetches from ModelReferenceManager.
            process_entry_points: Multiprocessing targets for child processes. If None, uses the real \
                (hordelib-backed) entry points. Test harnesses can inject fakes here.
            canned_job_source: Source of predetermined jobs used when `dry_run_skip_api` is set. \
                If None, an endlessly-cycling default scenario is used.
            canned_alchemy_source: Source of predetermined alchemy forms; when set, the alchemy \
                coordinator pops from it and records submits locally instead of touching the API.
            enable_background_downloads: If True, start a background download process that reports \
                on-disk model availability and fetches missing models, gating pops and inference \
                startup on disk presence. Off by default (tests/harness pre-load everything).
            max_threads_ceiling: The largest concurrent-inference count this session may scale to. \
                The IPC semaphores are provisioned for this many slots, and runtime thread changes \
                (config reload, supervisor command) are clamped to it. Defaults to ``max_threads`` \
                (no runtime growth beyond the configured value); the benchmark raises it.
        """
        self.session_start_time = time.time()
        self._state = WorkerState()
        self._sleep = asyncio.sleep

        ceiling = max(bridge_data.max_threads, max_threads_ceiling if max_threads_ceiling is not None else 0)
        self._runtime_config = RuntimeConfig(initial=bridge_data, max_threads_ceiling=ceiling)
        self._api_sessions = ApiSessions()
        self._model_metadata = ModelMetadata()
        self._model_availability = ModelAvailability()
        self._desired_state = DesiredState()
        self._enable_background_downloads = enable_background_downloads
        # Periodic, parent-owned reference refresh: subprocesses never download references, so the
        # parent re-downloads on this cadence and tells every subprocess to reload from disk. The same
        # reload also re-reads lora.json/ti.json, which is how cross-process LoRa/TI downloads become
        # visible without a restart.
        self._last_reference_refresh = time.time()
        self._reference_refresh_in_progress = False
        self._pending_reference_reload_broadcast = False
        logger.debug(f"Models to load: {bridge_data.image_models_to_load}")
        logger.debug(f"Custom Models to load: {bridge_data.custom_models}")

        self.horde_model_reference_manager = horde_model_reference_manager

        self._bridge_data_reloader = BridgeDataReloader(
            state=self._state,
            bridge_data_provider=lambda: self.bridge_data,
            model_reference_manager_provider=lambda: self.horde_model_reference_manager,
            apply_bridge_data=self._apply_reloaded_bridge_data,
            enable_performance_mode=self.enable_performance_mode,
            shutdown_callback=self._shutdown,
        )

        self._process_map = ProcessMap({})
        self._horde_model_map = HordeModelMap(root={})

        self.max_safety_processes = max_safety_processes

        # This attribute is the provisioned concurrency *ceiling* (semaphore size). The live
        # effective cap is read via the max_concurrent_inference_processes property below.
        self._max_concurrent_inference_processes = ceiling

        self._amd_gpu = amd_gpu
        self._directml = directml

        self._job_tracker = JobTracker()

        self.target_vram_overhead_bytes_map = target_vram_overhead_bytes_map  # TODO

        if system_resources is None:
            system_resources = SystemResources.detect()

        self.total_ram_bytes = system_resources.total_ram_bytes
        # Restrict to the cards this worker drives (all detected unless gpu_device_indices opts into a subset).
        self._device_map = _select_driven_devices(
            system_resources.device_map,
            self.bridge_data.gpu_device_indices,
        )
        logger.debug(f"Driving device indices: {sorted(self._device_map.root)}")

        # System RAM the worker keeps free. Computed before the runtime plan so the per-card process-count
        # cap can size the resident-context count against the rest of the shared pool.
        self.target_ram_overhead_bytes = min(int(self.total_ram_bytes / 2), 9 * 1024 * 1024 * 1024)

        if any(model in VRAM_HEAVY_MODELS for model in self.bridge_data.image_models_to_load):
            if self.total_ram_bytes < (24 * 1024 * 1024 * 1024):
                raise ValueError(
                    "VRAM heavy models detected. Total RAM is less than 24GB. "
                    "This is not enough RAM to run the worker."
                    "Disable the large models by adding it to your `models_to_skip` or remove it from your "
                    "`models_to_load`. Large models include: " + ", ".join(VRAM_HEAVY_MODELS),
                )

            self.target_ram_overhead_bytes = min(self.target_ram_overhead_bytes, int(20 * 1024 * 1024 * 1024 / 2))

        # Reserve additional RAM for co-hosted worker roles (an alchemist's in-worker CLIP/BLIP, a scribe's text
        # model) so the per-card process-count cap does not size resident contexts as if the whole pool were
        # theirs. Clamped so the reserve never claims more than 75% of RAM (always leaving room for at least one
        # inference context): the summed image + co-tenant footprint over-committing the shared pool is what
        # drove the observed OS OOM kill on a box running a dreamer, an alchemist, and a scribe together.
        co_tenant_reserve = co_tenant_ram_reserve_bytes(self.bridge_data)
        if co_tenant_reserve > 0:
            self.target_ram_overhead_bytes = min(
                self.target_ram_overhead_bytes + co_tenant_reserve,
                int(self.total_ram_bytes * 0.75),
            )
            logger.debug(
                f"Reserving an additional {co_tenant_reserve / 1024 / 1024 / 1024:.1f} GB RAM for co-hosted "
                f"worker roles (alchemist/scribe); target RAM overhead now "
                f"{self.target_ram_overhead_bytes / 1024 / 1024 / 1024:.1f} GB.",
            )

        if self.target_ram_overhead_bytes > self.total_ram_bytes:
            raise ValueError(
                f"target_ram_overhead_bytes ({self.target_ram_overhead_bytes}) is greater than "
                f"total_ram_bytes ({self.total_ram_bytes})",
            )

        # Build the per-card runtime plan (effective config + concurrency sizes + per-card semaphores) and,
        # unless a test injected them, the multiprocessing primitives sized for it. A single-GPU host yields
        # a one-entry map whose process count and semaphore sizes equal the old global computation, so the
        # total process count and LRU capacity stay identical for one card.
        self._card_runtimes, mp_primitives = self._build_card_runtimes(
            ctx=ctx,
            mp_primitives=mp_primitives,
            max_threads_ceiling=ceiling,
        )
        self.max_inference_processes = sum(card.target_process_count for card in self._card_runtimes.values())
        self._lru = LRUCache(self.max_inference_processes)

        # Multi-GPU is auto-all by default, so a host that previously only used card 0 now drives every
        # card under this one identity. Warn prominently in that auto-detected case (not when the operator
        # explicitly chose the cards) so anyone still running a separate worker per card notices and opts out.
        if len(self._card_runtimes) > 1 and self.bridge_data.gpu_device_indices is None:
            logger.warning(
                f"Multi-GPU: auto-detected {len(self._card_runtimes)} GPUs (indices "
                f"{sorted(self._card_runtimes)}); this single worker identity now drives all of them under "
                "one name and one job queue. If you run a separate worker per card, set gpu_device_indices "
                "(or pass --gpu-device-indices) to pin this worker to specific card(s).",
            )

        # The legacy --directml=N flag is inherently a single-device selection, so it stays authoritative:
        # every inference process targets that one adapter. Multi-GPU DirectML is instead opted into via
        # gpu_device_indices *without* --directml, where each card derives its own --directml index. Warn so
        # an operator on a multi-adapter DirectML box is not surprised that the explicit flag pins them all.
        if self._directml is not None and len(self._card_runtimes) > 1:
            logger.warning(
                f"--directml={self._directml} selects a single DirectML adapter, so all "
                f"{len(self._card_runtimes)} driven cards' inference processes will target adapter "
                f"{self._directml}. For multi-GPU DirectML, omit --directml and set gpu_device_indices to "
                "the adapter indices instead.",
            )

        self._log_resource_budget_posture()

        self._status_message_frequency = bridge_data.stats_output_frequency

        logger.debug(f"Total RAM: {self.total_ram_bytes / 1024 / 1024 / 1024} GB")
        logger.debug(f"Target RAM overhead: {self.target_ram_overhead_bytes / 1024 / 1024 / 1024} GB")

        self.enable_performance_mode()

        if not skip_api_init and self.bridge_data.remove_maintenance_on_init:
            try:
                self.remove_maintenance()
            except Exception as e:
                logger.warning(e)
                logger.warning("Error trying to unset maintenance. Did this worker not exist yet?")

        # The per-card inference / VAE / sampling-lease semaphores live on self._card_runtimes (built
        # above); only the genuinely shared primitives are read out here.
        self._process_message_queue = mp_primitives.process_message_queue
        self._disk_lock = mp_primitives.disk_lock
        self._aux_model_lock = mp_primitives.aux_model_lock
        self._download_bandwidth_semaphore = mp_primitives.download_bandwidth_semaphore

        # Take ownership of child OS pids so a parent that died hard can have its orphaned children
        # reaped on the next startup. Disabled under test: it touches real OS processes and a shared
        # on-disk registry, neither of which belongs in CI. The registry itself is unit-tested directly.
        self._owned_registry: OwnedProcessRegistry | None = None
        if not os.environ.get("AI_HORDE_TESTING"):
            self._owned_registry = OwnedProcessRegistry()
            reaped = self._owned_registry.reap_orphans_from_previous_run()
            if reaped:
                logger.warning(
                    f"Reaped {len(reaped)} orphaned child process(es) left by a previous run: {reaped}",
                )

        # Self-audited record of lifecycle actions, for post-mortems of a hang/crash. Always kept in
        # memory (so the timeout diagnostics dump works); mirrored to a JSONL file only outside tests.
        ledger_path = None if os.environ.get("AI_HORDE_TESTING") else (default_app_state_dir() / "action_ledger.jsonl")
        self._action_ledger = ActionLedger(path=ledger_path)

        self._process_lifecycle = ProcessLifecycleManager(
            ctx=ctx,
            process_map=self._process_map,
            horde_model_map=self._horde_model_map,
            job_tracker=self._job_tracker,
            process_message_queue=self._process_message_queue,
            card_runtimes=self._card_runtimes,
            disk_lock=self._disk_lock,
            aux_model_lock=self._aux_model_lock,
            download_bandwidth_semaphore=self._download_bandwidth_semaphore,
            gpu_sampling_lease_enabled=self.bridge_data.gpu_sampling_lease_enabled,
            runtime_config=self._runtime_config,
            max_safety_processes=self.max_safety_processes,
            amd_gpu=self._amd_gpu,
            directml=self._directml,
            abort_callback=lambda: self._abort(),
            state=self._state,
            entry_points=process_entry_points,
            owned_registry=self._owned_registry,
            action_ledger=self._action_ledger,
        )

        self._download_coordinator = ModelDownloadCoordinator(
            state=self._state,
            process_map=self._process_map,
            process_lifecycle=self._process_lifecycle,
            model_availability=self._model_availability,
            desired_state=self._desired_state,
            bridge_data_provider=lambda: self.bridge_data,
            stable_diffusion_reference_provider=lambda: self.stable_diffusion_reference,
            enable_background_downloads=self._enable_background_downloads,
        )

        self._message_dispatcher = MessageDispatcher(
            process_map=self._process_map,
            horde_model_map=self._horde_model_map,
            job_tracker=self._job_tracker,
            process_message_queue=self._process_message_queue,
            runtime_config=self._runtime_config,
            model_metadata=self._model_metadata,
            action_ledger=self._action_ledger,
            on_unload_vram=self.unload_models_from_vram,
            state=self._state,
        )

        self._run_metrics = WorkerRunMetrics(baseline_resolver=self._safe_model_baseline)

        # Measure real GPU core uptime (the duty cycle) for the whole worker session, not just the
        # benchmark. A coarse 1s poll is plenty for the rolling-window trend and threshold logs and
        # is far cheaper than the benchmark's 0.1s sampler. It no-ops on CPU/fake/non-NVIDIA backends
        # (no telemetry -> no thread), so creating it here is always safe.
        self._gpu_sampler = GpuUtilizationSampler(interval_seconds=1.0)
        self._last_duty_cycle_log_time = 0.0
        self._last_no_jobs_seconds_at_duty_log = 0.0
        self._first_inference_started_at: float | None = None
        """Epoch second the worker's first-ever inference began sampling, cached once. The duty cycle
        excludes everything before it: cold-boot model loading is one-time warm-up, not inter-job
        inefficiency, so counting it only depresses the headline with noise."""

        # Expected-time-to-complete model: seeds from the last benchmark's per-tier reference it/s and
        # self-calibrates from this worker's own jobs, so a "slow" job becomes measurable rather than
        # guessed. Disabled-to-memory under test (no app-state read, no benchmark import, no perf file);
        # the model is unit-tested directly.
        self._performance_model = self._build_performance_model()

        self._message_dispatcher.set_metrics_handlers(
            on_job_metrics=self._on_job_metrics,
            on_download_metrics=self._run_metrics.on_download_metrics,
        )
        self._message_dispatcher.set_download_availability_handler(self._download_coordinator.on_download_availability)
        self._message_dispatcher.set_model_load_failure_handler(self._on_model_load_failure)
        self._job_tracker.set_finalize_observer(self._on_job_finalized)
        self._process_lifecycle.set_process_recovery_observer(self._record_process_crash)

        self._disk_monitor = DiskSpaceMonitor(self._disk_paths_to_monitor())
        self._last_disk_sample_time = 0.0
        self._lora_paths = self._resolve_lora_paths()

        self._supervisor = SupervisorChannel(supervisor_connection) if supervisor_connection is not None else None
        self._last_supervisor_publish_time = 0.0
        self._last_supervisor_signature: tuple[object, ...] | None = None

        self._safety_orchestrator = SafetyOrchestrator(
            process_map=self._process_map,
            job_tracker=self._job_tracker,
            process_lifecycle=self._process_lifecycle,
            runtime_config=self._runtime_config,
            model_metadata=self._model_metadata,
            state=self._state,
        )

        self._shutdown_manager = ShutdownManager(
            state=self._state,
            job_tracker=self._job_tracker,
            process_map=self._process_map,
            process_lifecycle=self._process_lifecycle,
        )

        # One committed-VRAM/RAM reserve ledger shared by every workload flow (image generation and
        # alchemy today; audio/video later), so they account for one another's in-flight cost and cannot
        # independently admit against the same free VRAM.
        self._reserve_ledger = CommittedReserveLedger()

        self._inference_scheduler = InferenceScheduler(
            state=self._state,
            process_map=self._process_map,
            horde_model_map=self._horde_model_map,
            job_tracker=self._job_tracker,
            process_lifecycle=self._process_lifecycle,
            runtime_config=self._runtime_config,
            model_metadata=self._model_metadata,
            card_runtimes=self._card_runtimes,
            max_concurrent_inference_processes=self._max_concurrent_inference_processes,
            max_inference_processes=self.max_inference_processes,
            lru=self._lru,
            performance_model=self._performance_model,
            reserve_ledger=self._reserve_ledger,
        )
        # Feed the startup-measured per-process VRAM overhead to the scheduler's streaming forecast, so it
        # can estimate the free VRAM achievable under sole residency (total - one process's context) and,
        # from the probe's second-context delta, the marginal cost of each additional sibling context (so
        # free-after-model-evict is not the one-time runtime cost multiplied by the process count).
        self._inference_scheduler.set_measured_per_process_overhead_mb(system_resources.per_process_overhead_mb)
        self._inference_scheduler.set_measured_marginal_overhead_mb(system_resources.marginal_process_overhead_mb)
        # Attribute between-jobs reload/respawn churn (model swaps, VRAM evictions, process cycles) into
        # the run metrics so the periodic duty-cycle line can name it alongside the per-job phase gaps.
        self._inference_scheduler.set_churn_observer(self._run_metrics.record_churn)

        self._recovery_coordinator = WorkerRecoveryCoordinator(
            state=self._state,
            runtime_config=self._runtime_config,
            job_tracker=self._job_tracker,
            process_map=self._process_map,
            process_lifecycle=self._process_lifecycle,
            message_dispatcher=self._message_dispatcher,
            inference_scheduler=self._inference_scheduler,
            action_ledger=self._action_ledger,
            bridge_data_provider=lambda: self.bridge_data,
            max_inference_processes_provider=lambda: self.max_inference_processes,
            abort_callback=lambda: self._abort(),
        )

        self._job_submitter = JobSubmitter(
            state=self._state,
            job_tracker=self._job_tracker,
            shutdown_manager=self._shutdown_manager,
            runtime_config=self._runtime_config,
            api_sessions=self._api_sessions,
            model_metadata=self._model_metadata,
            dry_run_skip_api=bridge_data.dry_run_skip_api,
        )

        self._job_popper = JobPopper(
            state=self._state,
            process_map=self._process_map,
            job_tracker=self._job_tracker,
            shutdown_manager=self._shutdown_manager,
            runtime_config=self._runtime_config,
            api_sessions=self._api_sessions,
            max_inference_processes=self.max_inference_processes,
            max_concurrent_inference_processes=self._max_concurrent_inference_processes,
            dry_run_skip_api=bridge_data.dry_run_skip_api,
            canned_job_source=canned_job_source,
            model_availability=self._model_availability,
            card_runtimes=self._card_runtimes,
            model_metadata=self._model_metadata,
            whole_card_residency_active=self._inference_scheduler.is_whole_card_residency_active,
        )

        # Tracks the live spell and session totals of every pop/scheduling governor, fed once per control-loop
        # tick by _update_pop_governors so its ENTER/EXIT log lines fire regardless of whether a TUI is attached.
        self._pop_governor_registry = PopGovernorRegistry()

        self._alchemy_coordinator = AlchemyCoordinator(
            state=self._state,
            process_map=self._process_map,
            job_tracker=self._job_tracker,
            shutdown_manager=self._shutdown_manager,
            runtime_config=self._runtime_config,
            api_sessions=self._api_sessions,
            reserve_ledger=self._reserve_ledger,
            canned_alchemy_source=canned_alchemy_source,
            run_metrics=self._run_metrics,
        )
        self._message_dispatcher.set_alchemy_result_handler(self._alchemy_coordinator.on_alchemy_result)

        self._image_coordinator = ImageGenerationCoordinator(
            job_popper=self._job_popper,
            job_submitter=self._job_submitter,
            job_tracker=self._job_tracker,
            subtask_done_callback=self._handle_exception,
        )
        # One WorkloadKind-keyed registry of every workload flow this worker runs. Both flows are
        # always present and self-gate (the image popper does not pop without models; the alchemy
        # coordinator does not pop unless alchemist is set), so the registry mirrors today's
        # always-running loops while giving the snapshot and TUI a uniform per-workload surface. Which
        # workloads the worker is *for* (process sizing, the dashboard's mode identity) is a separate,
        # declarative question answered by capabilities.enabled_workloads.
        self._flows: dict[WorkloadKind, FlowCoordinator] = {
            WorkloadKind.IMAGE_GENERATION: self._image_coordinator,
            WorkloadKind.ALCHEMY: self._alchemy_coordinator,
        }

        if stable_diffusion_reference is not None:
            self.stable_diffusion_reference = stable_diffusion_reference
        else:
            if horde_model_reference_manager is None:
                raise ValueError(
                    "horde_model_reference_manager may only be None when stable_diffusion_reference is provided",
                )
            self.stable_diffusion_reference = None
            self._init_model_reference()

    def _build_card_runtimes(
        self,
        *,
        ctx: BaseContext,
        mp_primitives: MultiprocessingPrimitives | None,
        max_threads_ceiling: int,
    ) -> tuple[dict[int, CardRuntime], MultiprocessingPrimitives]:
        """Build the per-card runtime plan and, unless injected, the multiprocessing primitives for it.

        Resolves each driven card's effective config and concurrency sizes, creates (or reuses an injected)
        :class:`MultiprocessingPrimitives` sized per card, and assembles one :class:`CardRuntime` per card.
        Masking is enabled only when there is a real choice to make: more than one card, or an explicit
        ``gpu_device_indices`` selection, so a default single-GPU host stays unmasked and byte-identical.
        A host with no detected accelerator (CPU/dry-run) yields a single notional card 0.

        Args:
            ctx: The multiprocessing context (used only when primitives must be created).
            mp_primitives: Pre-created primitives (injected by tests), or None to create real ones.
            max_threads_ceiling: The session-wide concurrency ceiling applied to each card.

        Returns:
            A 2-tuple of the per-card runtime map (keyed by stable device index) and the primitives used.
        """
        bridge_data = self.bridge_data
        device_indices = sorted(self._device_map.root) or [0]
        effective_configs = resolve_all_effective_gpu_configs(bridge_data, device_indices)
        # The dreamer/alchemist roles are worker-wide (not per-card overrides), so whether image
        # generation is served is decided once for the whole worker. An alchemist-only worker collapses
        # every card to a single inference process (graph alchemy serializes through it).
        serves_image_generation = WorkloadKind.IMAGE_GENERATION in enabled_workloads(bridge_data)
        per_card_concurrency = {
            index: resolve_card_concurrency(
                max_threads=effective_configs[index].max_threads,
                queue_size=effective_configs[index].queue_size,
                num_models_to_load=len(effective_configs[index].image_models_to_load),
                gpu_sampling_lease_enabled=effective_configs[index].gpu_sampling_lease_enabled,
                gpu_sampling_lease_slots=effective_configs[index].gpu_sampling_lease_slots,
                max_threads_ceiling=max_threads_ceiling,
                serves_image_generation=serves_image_generation,
            )
            for index in device_indices
        }
        if mp_primitives is None:
            mp_primitives = MultiprocessingPrimitives.create(ctx=ctx, per_card=per_card_concurrency)

        # Right-size the resolved per-card plan to the hardware when this worker drives more than one card:
        # the summed plan would otherwise over-commit the single shared system-RAM pool (a second card doubles
        # VRAM, not RAM) and ignore the post-processing VRAM peak. The cap only ever lowers the spawn count and
        # never below one context per card; the single-GPU path is left byte-identical. The concurrency
        # ceiling and semaphore sizes are unchanged: only how many processes are spawned is reduced.
        target_process_counts = {index: per_card_concurrency[index].target_process_count for index in device_indices}
        total_vram_mb_by_card = {
            index: (info.total_memory / (1024 * 1024))
            if (info := self._device_map.root.get(index)) is not None
            else None
            for index in device_indices
        }
        if len(device_indices) > 1:
            target_process_counts = cap_card_process_counts(
                per_card_target_processes=target_process_counts,
                total_ram_bytes=self.total_ram_bytes,
                target_ram_overhead_bytes=self.target_ram_overhead_bytes,
                total_vram_mb_by_card=total_vram_mb_by_card,
                allow_post_processing=bridge_data.allow_post_processing,
                has_vram_heavy_models=any(m in VRAM_HEAVY_MODELS for m in bridge_data.image_models_to_load),
            )
            for index in device_indices:
                resolved = per_card_concurrency[index].target_process_count
                if target_process_counts[index] < resolved:
                    logger.warning(
                        f"Right-sizing device {index} to {target_process_counts[index]} inference process(es) "
                        f"(resolved plan was {resolved}): the shared system RAM and this card's "
                        f"post-processing VRAM headroom cannot sustain the full plan across "
                        f"{len(device_indices)} cards. Lower max_threads/queue_size to raise it intentionally.",
                    )

        should_mask = len(device_indices) > 1 or bridge_data.gpu_device_indices is not None
        card_runtimes: dict[int, CardRuntime] = {}
        for index in device_indices:
            device_info = self._device_map.root.get(index)
            kind = device_info.kind if device_info is not None else "cuda"
            concurrency = per_card_concurrency[index]
            card_runtimes[index] = CardRuntime(
                device_index=index,
                kind=kind,
                config=effective_configs[index],
                # Total VRAM for the heterogeneous weight-fit check; None when capacity is unknown
                # (a notional CPU/dry-run card 0 has no device_info), where the eligibility check abstains.
                total_vram_mb=total_vram_mb_by_card[index],
                inference_semaphore=mp_primitives.inference_semaphores[index],
                vae_decode_semaphore=mp_primitives.vae_decode_semaphores[index],
                gpu_sampling_lease=mp_primitives.gpu_sampling_leases[index],
                target_process_count=target_process_counts[index],
                max_concurrent_inference=concurrency.max_concurrent_inference,
                mask_kind=(kind if should_mask else None),
            )
        return card_runtimes, mp_primitives

    def _init_model_reference(self) -> None:
        """Fetch the stable diffusion model reference, retrying on failure."""
        while self.stable_diffusion_reference is None:
            try:
                horde_model_reference_manager = ModelReferenceManager.get_instance()

                # The orchestrator builds its own copy of the image reference, separate from the inference
                # subprocesses; beta_aware_image_records keeps the two in agreement when beta (pending-queue)
                # models such as qwen/Z-Image are opted into (see bridge_data.beta_source).
                records = beta_aware_image_records(horde_model_reference_manager)
                if not records:
                    raise RuntimeError(
                        "horde_model_reference returned no image_generation models; the reference may have "
                        "failed to download; cannot continue with an empty reference.",
                    )

                self.stable_diffusion_reference = records
            except Exception as e:
                logger.error(e)
                time.sleep(5)

    def _apply_self_maintenance_throttle(self) -> None:
        """Local-pause popping when resource/OOM faults approach the horde's server-side drop tolerance.

        A backstop above the per-model circuit-breaker: if terminal resource faults across all models
        accumulate fast enough within the configured window, enter a worker-initiated local pop-pause
        (in-flight jobs finish) for a cooldown, so the worker stops the bleeding on its own terms before
        the horde forces it into maintenance for "dropping too many jobs". Auto-resumes after the cooldown.
        """
        now = time.time()
        if self._state.self_throttle_paused:
            if now >= self._state.self_throttle_paused_until:
                self._state.self_throttle_paused = False
                self._state.self_throttle_paused_until = 0.0
                logger.info("Self-throttle cooldown elapsed; resuming job pops.")
            return

        threshold = self.bridge_data.self_maintenance_fault_threshold
        if threshold <= 0:
            return
        window = self.bridge_data.self_maintenance_window_seconds
        recent = self._job_tracker.count_recent_resource_faults(window, now=now)
        if recent < threshold:
            return
        cooldown = self.bridge_data.self_maintenance_cooldown_seconds
        self._state.self_throttle_paused = True
        self._state.self_throttle_paused_until = now + cooldown
        logger.warning(
            f"Self-throttle engaged: {recent} resource/OOM faults in the last {window:.0f}s (threshold "
            f"{threshold}); pausing job pops locally for {cooldown:.0f}s so the horde does not force the "
            "worker into maintenance. In-flight jobs will finish.",
        )

    def _apply_post_processing_fault_breaker(self) -> None:
        """Session-latch off post-processing after repeated unhostable post-processing-peak faults.

        A post-processing peak that cannot be hosted (a single-process worker on a tiny card, or a card a job
        over-commits) faults the job; the horde reissues it, but a worker that keeps faulting trips the
        horde's forced-maintenance, the very spiral this guards against. When such faults (the scheduler's
        unhostable-peak faults and the lifecycle's reaped post-processing stalls) exceed the configured
        threshold within the window, stop advertising post-processing so the worker is no longer handed
        upscale/face-fix jobs it cannot host, and advise the operator to downgrade. Session-latched: the
        over-commit is structural, so it clears only on restart (auto-recovery would simply re-trip it).
        """
        if not self.bridge_data.post_processing_fault_breaker_enabled:
            return
        if self._state.post_processing_disabled_by_breaker:
            return
        threshold = self.bridge_data.post_processing_fault_threshold
        window = self.bridge_data.post_processing_fault_window_seconds
        now = time.time()
        recent = self._job_tracker.count_recent_post_processing_faults(window, now=now)
        if recent <= threshold:
            return
        self._state.post_processing_disabled_by_breaker = True
        self._state.post_processing_breaker_tripped_at = now
        logger.warning(
            f"Post-processing fault breaker tripped: {recent} post-processing over-commit fault(s) in the last "
            f"{window:.0f}s (threshold {threshold}). Disabling post-processing on this worker for the rest of "
            "the session so it stops being handed upscale/face-fix jobs it cannot host (which keep faulting and "
            "risk the horde forcing maintenance). To restore it, downgrade settings (lower max_threads or "
            "queue_size, or enable post_processing_active_reclaim_enabled) and restart the worker.",
        )

    def set_maintenance(self, enabled: bool) -> None:
        """Set the named worker's *server-side* maintenance flag via the horde API (blocking).

        ``enabled=True`` puts the worker into maintenance on the horde (it stops being sent jobs);
        ``enabled=False`` clears it. This is the true horde-side "maintenance mode" the job-pop response
        signals, distinct from the local pop-pause (:attr:`WorkerState.supervisor_paused`). Runs a
        blocking API call, so live callers from the control loop must invoke it off-loop (see
        :meth:`_apply_supervisor_command`).
        """
        simple_client = AIHordeAPISimpleClient()
        worker_details = lookup_worker_by_name(simple_client, self.bridge_data.dreamer_worker_name)
        if worker_details is None:
            logger.debug(
                f"Worker with name {self.bridge_data.dreamer_worker_name} is not registered yet "
                f"(the horde creates it on first pop); nothing to set maintenance={enabled} on.",
            )
            return
        modify_worker_request = ModifyWorkerRequest(
            apikey=self.bridge_data.api_key,
            worker_id=worker_details.id_,
            maintenance=enabled,
        )

        simple_client.worker_modify(modify_worker_request)

        verb = "placed into" if enabled else "removed from"
        logger.debug(
            f"Ensured worker with name {self.bridge_data.dreamer_worker_name} "
            f"({worker_details.id_}) is {verb} maintenance.",
        )

    def remove_maintenance(self) -> None:
        """Remove the server-side maintenance from the named worker.

        Thin convenience wrapper over :meth:`set_maintenance` (``set_maintenance(False)``).
        """
        self.set_maintenance(False)

    def _set_server_maintenance_safe(self, enabled: bool) -> None:
        """Best-effort off-loop ``set_maintenance`` for the supervisor toggle; never raises into the thread."""
        try:
            self.set_maintenance(enabled)
        except Exception as e:
            logger.warning(f"Failed to set server-side maintenance={enabled}: {type(e).__name__} {e}")

    def enable_performance_mode(self) -> None:
        """Enable performance mode."""
        # Re-applied here (init + every config reload) so a live change to max_inference_attempts takes
        # effect without a restart, alongside the other performance-related thresholds.
        self._job_tracker.set_retry_policy(self.bridge_data.max_inference_attempts)

        if self.bridge_data.high_performance_mode:
            self._job_tracker.set_performance_mode_thresholds(80)
            logger.info("High performance mode enabled")
            if not self.bridge_data.safety_on_gpu:
                logger.warning(
                    "If you have a high-end GPU, you should enable safety on GPU (safety_on_gpu in the config).",
                )

        elif self.bridge_data.moderate_performance_mode:
            self._job_tracker.set_performance_mode_thresholds(60)
            logger.info("Moderate performance mode enabled")
        else:
            self._job_tracker.set_performance_mode_thresholds(15)
            logger.info("Normal performance mode enabled")

        if self.bridge_data.high_performance_mode and self.bridge_data.moderate_performance_mode:
            logger.warning("Both high and moderate performance modes are enabled. Using high performance mode.")

    def is_time_for_shutdown(self) -> bool:
        """Return true if it is time to shut down."""
        return self._shutdown_manager.is_time_for_shutdown()

    def is_free_inference_process_available(self) -> bool:
        """Return true if there is an inference process available which can accept a job."""
        return self._process_map.num_available_inference_processes() > 0

    def is_any_model_preloaded(self) -> bool:
        """Return true if any model is preloaded."""
        return self._process_map.num_preloaded_processes() > 0

    def has_queued_jobs(self) -> bool:
        """Return true if there are any jobs not already in progress but are popped."""
        return any(job not in self._job_tracker.jobs_in_progress for job in self._job_tracker.jobs_pending_inference)

    def get_expected_ram_usage(self, horde_model_name: str) -> int:  # TODO: Use or rework this
        """Return the expected RAM usage of the given model, in bytes."""
        if self.stable_diffusion_reference is None:
            raise ValueError("stable_diffusion_reference is None")

        horde_model_record = self.stable_diffusion_reference[horde_model_name]

        if horde_model_record.baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1:
            return int(3 * 1024 * 1024 * 1024)
        if horde_model_record.baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_512:
            return 4 * 1024 * 1024 * 1024
        if horde_model_record.baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_768:
            return 5 * 1024 * 1024 * 1024
        if horde_model_record.baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl:
            return int(5.75 * 1024 * 1024 * 1024)

        raise ValueError(f"Model {horde_model_name} has an unknown baseline {horde_model_record.baseline}")

    async def receive_and_handle_process_messages(self) -> None:
        """Receive and handle any messages from the child processes.

        Delegates to MessageDispatcher.
        """
        await self._message_dispatcher.receive_and_handle_process_messages()

    def _log_resource_budget_posture(self) -> None:
        """Log, once at startup, whether the VRAM/RAM budget is active and how it will behave.

        This is the "warn loudly" half of the auto-throttle: the operator is told up front that the budget
        gates preloads and concurrent dispatch on measured free VRAM/RAM and evicts idle resident models
        under measured pressure (so any eviction churn they see later is expected and the remedy is to reduce
        the model set). The runtime budget is the actual enforcement; this only surfaces the posture.
        """
        if not self.bridge_data.enable_vram_budget:
            logger.warning(
                "VRAM/RAM budget is disabled (enable_vram_budget=false): the worker will not guard "
                "against multiple inference processes over-committing the GPU. Not recommended on a "
                "shared or consumer GPU.",
            )
            return

        primary_device = self._device_map.root.get(0)
        total_vram_mb = round(primary_device.total_memory / (1024 * 1024)) if primary_device is not None else None
        total_vram_note = f"{total_vram_mb} MB VRAM" if total_vram_mb is not None else "an unknown amount of VRAM"

        logger.info(
            f"VRAM/RAM budget active (reserve {self.bridge_data.vram_reserve_mb} MB VRAM / "
            f"{self.bridge_data.ram_reserve_mb} MB RAM on {total_vram_note}): preloads and concurrent "
            "dispatch are gated on measured free VRAM/RAM, and idle resident models are evicted under "
            "pressure to prevent out-of-memory crashes.",
        )

    async def unload_models_from_vram(
        self,
        process_with_model: HordeProcessInfo,
    ) -> None:
        """Unload models from VRAM from processes that are not running a job."""
        self._inference_scheduler.unload_models_from_vram(process_with_model)

    async def start_evaluate_safety(self) -> None:
        """Start evaluating the safety of the next job pending a safety check, if any."""
        await self._safety_orchestrator.start_evaluate_safety()

    _user_info_failed = False
    """Whether the API request to fetch user info failed."""
    _user_info_failed_reason: str | None = None
    """The reason the API request to fetch user info failed."""

    _worker_details_maintenance: bool = False
    """Whether the horde reports this worker in maintenance (from the worker-details API, polled)."""
    _worker_details_paused: bool = False
    """Whether the horde reports this worker paused (from the worker-details API, polled)."""

    def calculate_kudos_info(self) -> None:
        """Calculate and log information about the kudos generated in the current session."""
        # Use KudosCalculator to compute all metrics
        (
            time_since_session_start,
            kudos_per_hour_session,
            kudos_total_past_hour,
            active_kudos_per_hour,
            cleaned_events,
        ) = KudosCalculator.calculate_all_metrics(
            self._state.kudos_generated_this_session,
            self.session_start_time,
            self._job_popper.time_spent_no_jobs_available,
            self._state.kudos_events,
        )

        # Update the events deque with cleaned version
        self._state.kudos_events = cleaned_events

        kudos_info_string = self.generate_kudos_info_string(
            time_since_session_start,
            kudos_per_hour_session,
            kudos_total_past_hour,
            active_kudos_per_hour,
        )

        self.log_kudos_info(kudos_info_string)

    def calculate_kudos_totals(self) -> float:
        """Calculate the total kudos generated in the past hour.

        Returns:
            float: The total kudos generated in the past hour.
        """
        # Delegate to KudosCalculator
        kudos_total_past_hour, cleaned_events = KudosCalculator.calculate_kudos_totals_past_hour(
            self._state.kudos_events,
        )
        self._state.kudos_events = cleaned_events
        return kudos_total_past_hour

    def generate_kudos_info_string(
        self,
        time_since_session_start: float,
        kudos_per_hour_session: float,
        kudos_total_past_hour: float,
        active_kudos_per_hour: float,
    ) -> str:
        """Generate a string with information about the kudos generated in the current session.

        Args:
            time_since_session_start: The time since the session started.
            kudos_per_hour_session: The kudos per hour generated in the current session.
            kudos_total_past_hour: The total kudos generated in the past hour.
            active_kudos_per_hour: The kudos per hour generated while active (jobs available).

        Returns:
            A string with information about the kudos generated in the current session.
        """
        return _generate_kudos_info_string(
            kudos_generated_this_session=self._state.kudos_generated_this_session,
            time_since_session_start=time_since_session_start,
            kudos_per_hour_session=kudos_per_hour_session,
            kudos_total_past_hour=kudos_total_past_hour,
            active_kudos_per_hour=active_kudos_per_hour,
            time_spent_no_jobs_available=self._job_popper.time_spent_no_jobs_available,
            max_time_spent_no_jobs_available=self._job_popper._pop_throttler._max_time_spent_no_jobs_available,
        )

    def log_kudos_info(self, kudos_info_string: str) -> None:
        """Log the kudos information string.

        Args:
            kudos_info_string: The kudos information string to log.
        """
        logger.debug(f"len(kudos_events): {len(self._state.kudos_events)}")
        KudosLogger.log_kudos_info(
            kudos_info_string=kudos_info_string,
            kudos_generated_this_session=self._state.kudos_generated_this_session,
            user_info=self.user_info,
            limited_console_messages=self.bridge_data.limited_console_messages,
        )

    async def api_get_user_info(self) -> None:
        """Get the information associated with this API key from the API."""
        if self._state.shutting_down or self._state.last_pop_maintenance_mode:
            return

        if self.bridge_data.dry_run_skip_api:
            return

        request = FindUserRequest(apikey=self.bridge_data.api_key)
        try:
            response = await self.horde_client_session.submit_request(request, UserDetailsResponse)
            if isinstance(response, RequestErrorResponse):
                logger.error(f"Failed to get user info (API Error): {response}")
                self._user_info_failed = True
                return
            # if self.user_info is None:
            # logger.info(f"Got user info: {response}")  # FIXME

            self.user_info = response
            self._user_info_failed = False
            self._user_info_failed_reason = None

            if self.user_info.kudos_details is not None:
                self.calculate_kudos_info()

        except _async_client_exceptions as e:
            self._user_info_failed = True
            self._user_info_failed_reason = f"HTTP error (({type(e).__name__}) {e})"

        except Exception as e:
            self._user_info_failed = True
            self._user_info_failed_reason = f"Unexpected error (({type(e).__name__}) {e})"

        finally:
            if self._user_info_failed:
                logger.debug(f"Failed to get user info: {self._user_info_failed_reason}")
                logger.error("The server failed to respond. Is the horde or your internet down?")
            await logger.complete()

    def _fetch_worker_details(self, worker_name: str) -> WorkerDetailItem | None:
        """Synchronous worker-details lookup (run in a thread); mirrors :meth:`remove_maintenance`.

        Returns None for a name that is not yet registered (the normal first-run state, since the horde
        registers a worker on its first pop) instead of raising and logging at ERROR. See
        :func:`lookup_worker_by_name`.
        """
        simple_client = AIHordeAPISimpleClient()
        return lookup_worker_by_name(simple_client, worker_name)

    async def api_get_worker_details(self) -> None:
        """Best-effort refresh of this worker's maintenance/paused flags from the worker-details API.

        These flags can be toggled remotely (web UI or another tool), so polling them lets the
        dashboard show a worker the horde has placed in maintenance even when the local pop loop is not
        the cause. Deliberately not gated on ``last_pop_maintenance_mode`` (unlike user-info) so the
        worker keeps observing when maintenance is lifted. Any failure is swallowed: this is advisory
        context, never load-bearing.
        """
        if self._state.shutting_down or self.bridge_data.dry_run_skip_api:
            return
        worker_name = self.bridge_data.dreamer_worker_name
        if not worker_name:
            return
        try:
            worker_details = await asyncio.to_thread(self._fetch_worker_details, worker_name)
        except Exception as e:  # noqa: BLE001 - advisory poll must never disturb the worker
            logger.trace(f"Worker-details refresh failed: {type(e).__name__} {e}")
            return
        if worker_details is None:
            return
        polled_maintenance = bool(worker_details.maintenance_mode)
        if not polled_maintenance:
            self._state.server_maintenance_cleared_by_job_pop = False
        self._worker_details_maintenance = polled_maintenance and not self._state.server_maintenance_cleared_by_job_pop
        self._worker_details_paused = bool(worker_details.paused)

    async def _api_get_user_info_loop(self) -> None:
        """Run the API get user info loop."""
        logger.debug("In _api_get_user_info_loop")
        while True:
            with logger.catch():
                try:
                    if time.time() - self._last_get_user_info_time >= self._api_get_user_info_interval:
                        self._last_get_user_info_time = time.time()
                        await self.api_get_user_info()
                        await self.api_get_worker_details()
                except CancelledError as e:
                    self._shutdown()
                    logger.debug(f"CancelledError: {e}")

            # Checked outside the catch block so persistent errors cannot prevent shutdown.
            if self.is_time_for_shutdown() or self._state.shut_down:
                break

            # Sleep briefly (not the full user info interval) so shutdown is detected promptly.
            await asyncio.sleep(1)

    _UPDATE_CHECK_SHUTDOWN_POLL_SECONDS = 1.0
    """How often the update-check loop wakes to notice an in-flight shutdown while waiting out its
    30-minute interval. The interval itself is long, but waiting it out in one ``asyncio.sleep`` would
    keep this task (and so the ``asyncio.gather`` over every loop, and so the whole process) alive until
    the next wake-up after a shutdown was requested. With the control loop already done, nothing would
    stamp liveness during that gap and the TUI would age the worker into a false UNRESPONSIVE long after
    it had really stopped. Polling on this cadence instead lets the loop exit within a second of shutdown,
    matching the other background loops."""

    async def _periodic_update_check_loop(self) -> None:
        """Re-check for a newer worker release every 30 minutes and surface in logs.

        The initial check runs from ``_start_release_update_check`` at startup; this loop keeps
        the verdict current for long-running workers so a release published mid-session is noticed.
        The result is recorded via :func:`horde_worker_regen.update_check.apply_update_check_result`
        so the periodic status report picks it up without further plumbing.

        The long interval is waited out in short slices (not a single sleep) so a shutdown requested
        mid-interval is honoured promptly rather than pinning the process open until the next wake-up;
        see :attr:`_UPDATE_CHECK_SHUTDOWN_POLL_SECONDS`.
        """
        from horde_worker_regen.update_check import (
            UPDATE_CHECK_INTERVAL_SECONDS,
            apply_update_check_result,
            check_for_update,
            current_version,
            update_check_disabled,
        )

        # This is one of the worker's main-loop tasks, which the loop's done-callback expects to run for the
        # worker's whole lifetime: a task returning while the worker is live is treated as a fatal anomaly and
        # triggers a graceful shutdown. So when update checks are disabled there is simply nothing to do each
        # tick; the loop still idles until shutdown rather than returning early and tripping that path (which
        # would otherwise shut the worker down at startup whenever checks are disabled, e.g. under the test
        # env or an operator's opt-out).
        checks_enabled = not update_check_disabled()
        next_check_at = time.monotonic() + UPDATE_CHECK_INTERVAL_SECONDS
        while True:
            await asyncio.sleep(self._UPDATE_CHECK_SHUTDOWN_POLL_SECONDS)
            if self.is_time_for_shutdown() or self._state.shut_down:
                break
            if not checks_enabled or time.monotonic() < next_check_at:
                continue
            next_check_at = time.monotonic() + UPDATE_CHECK_INTERVAL_SECONDS
            try:
                info = check_for_update()
            except Exception:  # noqa: BLE001 - an update check must never affect the worker
                continue
            apply_update_check_result(info)
            if info is None:
                logger.info(f"Worker v{current_version()} is up to date.")
            else:
                logger.warning(
                    f"Update available: v{current_version()} -> v{info.latest_version}. Update with "
                    "'update.cmd'/'update.sh', or by re-running the installer.",
                )

    _SERVER_CAPABILITIES_POLL_SECONDS = 5.0
    """How often the server-capability loop wakes. The probe self-throttles on its own TTL, so this only
    bounds how promptly a shutdown is noticed and how soon the first probe runs after startup; the actual
    swagger fetch cadence is governed by :mod:`horde_worker_regen.server_capabilities`."""

    async def _periodic_server_capabilities_loop(self) -> None:
        """Keep the server-capability probe fresh, off any job hot path.

        Some worker features (a newly-added interrogation form, the aesthetic-score ``gen_metadata``
        attachment) must be withheld until the server advertises support, or the server rejects the
        whole pop/submit. The probe reads the server's Swagger document and gates those features
        fail-closed. It is refreshed here, on its own task, rather than inside the pop loops: a slow or
        hung swagger fetch must never delay job popping, and the single owner keeps one fetch serving
        both the image and alchemy flows. The probe self-throttles on its TTL, so calling it each tick
        is cheap once primed.
        """
        from horde_worker_regen.server_capabilities import refresh_server_capabilities

        while True:
            with logger.catch():
                try:
                    await refresh_server_capabilities()
                except CancelledError as e:
                    self._shutdown()
                    logger.debug(f"CancelledError: {e}")

            if self.is_time_for_shutdown() or self._state.shut_down:
                break

            await asyncio.sleep(self._SERVER_CAPABILITIES_POLL_SECONDS)

    _status_message_frequency = 20.0
    """The rate in seconds at which to print status messages with details about the current state of the worker."""
    _last_status_message_time = 0.0
    """The epoch time of the last status message."""

    async def _control_loop_tick(self) -> bool:
        """Run a single iteration of the process control loop.

        Pacing sleeps go through ``self._sleep`` so tests can drive ticks with a
        no-op sleep and assert effects deterministically.

        Returns:
            True if the loop should keep running, False when it is time to shut down.
        """
        if self.stable_diffusion_reference is None:
            raise ValueError("stable_diffusion_reference is None; cannot run the control loop")

        if self._supervisor is not None:
            self._supervisor.note_alive()

        with logger.catch(reraise=True):
            await self._sleep(self._loop_interval)

            await self.receive_and_handle_process_messages()
            self._download_coordinator.maybe_start_safety_processes()
            self._download_coordinator.maybe_start_inference_processes()
            # A child may have just reported a CPU-only torch build (image generation disabled); collapse the
            # fleet to one inference process per card so a sentinel-less CPU install matches the startup sizing.
            self._enforce_alchemist_only_scale_down()
            self._apply_self_maintenance_throttle()
            self._apply_post_processing_fault_breaker()
            if self._state.server_maintenance_cleared_by_job_pop and self._worker_details_maintenance:
                logger.info("Clearing cached worker-details maintenance: a new job was popped successfully.")
                self._worker_details_maintenance = False
            self.detect_deadlock()

            if len(self._job_tracker.jobs_pending_safety_check) > 0:
                await self.start_evaluate_safety()

            free_process_or_model_loaded = self.is_free_inference_process_available() or self.is_any_model_preloaded()

            if (
                self._state.last_pop_maintenance_mode
                and self.num_jobs_total == 0
                and not self._job_popper._replaced_due_to_maintenance
            ):
                logger.warning("Reloading all process due to maintenance mode")
                for process_info in self._process_map.values():
                    if process_info.process_type == HordeProcessType.INFERENCE:
                        # This is a deliberate, operational reload of healthy slots, not a crash recovery:
                        # flag it so each replacement is not mislabelled "crashed or hung" in the recovery
                        # diagnostics nor counted as a process recovery (which would otherwise make a routine
                        # maintenance episode look like a crash storm to the recovery diagnostics).
                        self._process_lifecycle._replace_inference_process(
                            process_info,
                            intentional_reason="maintenance-mode pool reload",
                        )
                    self._job_popper._replaced_due_to_maintenance = True
                MaintenanceModeMessenger.print_maintenance_mode_messages()

            # Drive resource governance every iteration, independent of queue depth. The governor tick is the
            # only thing that clears the soft RAM pop hold and restores shed contexts once RAM recovers;
            # gating it on a non-empty inference queue would let a pop hold that drained the queue latch
            # forever (the hold keeps the queue empty, and an empty queue would keep the tick from ever
            # running). Placed after the housekeeping above so the governor snapshots the same process-map
            # and pause state a scheduling cycle would.
            self._inference_scheduler.run_governance_tick()

            if free_process_or_model_loaded and len(self._job_tracker.jobs_pending_inference) > 0:
                await self._inference_scheduler.run_scheduling_cycle(self.stable_diffusion_reference)

            await self._sleep(self._loop_interval)
            await self.receive_and_handle_process_messages()
            self._handle_supervisor_commands()
            if self._process_lifecycle.replace_hung_processes():
                await self._sleep(self._loop_interval / 2)
                await self._sleep(self._loop_interval / 2)
            self._process_lifecycle._replace_all_safety_process()

            # Backstop the per-slot recovery: punt any job left in-progress with no owning live slot
            # before it can wedge the head of the queue.
            self._recovery_coordinator.reconcile_orphaned_in_progress_jobs()

            # The safety-stage analogue: recover any job stranded in SAFETY_CHECKING whose verdict was lost
            # (re-check it, or fault it with no image if safety is pathological) before the backlog wedges
            # the pipeline.
            await self._recovery_coordinator.reconcile_orphaned_safety_jobs()

            # Save-our-ship: above the per-slot recovery, escalate a worker that is wedged as a whole
            # (no live process for pending work) to a soft reset and finally to giving up cleanly.
            self._recovery_coordinator.run_recovery_supervisor()

            # During graceful shutdown, keep the inference processes up until the queue the worker
            # already accepted has drained, so those jobs get a chance to finish (the popper has
            # already stopped accepting new work). Only wind the processes down once no inference job
            # remains pending or in progress. Once the safety queue is also drained, wind down the safety
            # pool too; otherwise an idle or still-starting safety child can outlive the control loop and
            # only be killed by the timed-shutdown backstop.
            if (
                self._state.shutting_down
                and not self._state.last_pop_recently()
                and len(self._job_tracker.jobs_pending_inference) == 0
                and len(self._job_tracker.jobs_in_progress) == 0
            ):
                self._process_lifecycle.end_inference_processes()
                if (
                    len(self._job_tracker.jobs_pending_safety_check) == 0
                    and len(self._job_tracker.jobs_being_safety_checked) == 0
                    and self._alchemy_coordinator.num_forms_pending == 0
                    and self._alchemy_coordinator.num_forms_in_flight == 0
                    and self._alchemy_coordinator.num_forms_awaiting_submit == 0
                ):
                    self._process_lifecycle.end_safety_processes()

            if self.is_time_for_shutdown():
                return False

        self._maybe_refresh_references()
        self.print_status_method()
        self._maybe_log_duty_cycle()
        self._sample_disk_space()
        self._update_pop_governors()
        self._publish_supervisor_snapshot()

        await self._sleep(self._loop_interval / 2)
        return True

    _REFERENCE_REFRESH_INTERVAL_SECONDS = 1800.0
    """How often the parent re-downloads the model reference and tells subprocesses to reload from
    disk. References change rarely, so this is intentionally infrequent; it also refreshes
    cross-process LoRa/TI visibility (subprocesses re-read lora.json/ti.json on the same reload)."""

    def _maybe_refresh_references(self) -> None:
        """Periodically re-download references in the parent and broadcast a reload to subprocesses.

        Subprocesses never download references themselves: the parent owns it. The network
        re-download runs off the event loop in a daemon thread (writing fresh JSON to disk); once it
        finishes, the next tick broadcasts a reload so every subprocess re-reads the files.
        """
        if self._pending_reference_reload_broadcast:
            self._pending_reference_reload_broadcast = False
            self._process_lifecycle.broadcast_reload_model_database()

        if self._reference_refresh_in_progress:
            return
        if (time.time() - self._last_reference_refresh) < self._REFERENCE_REFRESH_INTERVAL_SECONDS:
            return
        manager = self.horde_model_reference_manager
        if manager is None:
            return

        self._last_reference_refresh = time.time()
        self._reference_refresh_in_progress = True

        def _refresh() -> None:
            try:
                manager.get_all_model_references_or_none(overwrite_existing=True)
                logger.info("Refreshed model reference from source; broadcasting reload to subprocesses")
                self._pending_reference_reload_broadcast = True
            except Exception as e:  # noqa: BLE001 - a refresh failure must not crash the worker
                logger.warning(f"Periodic model reference refresh failed: {type(e).__name__}: {e}")
            finally:
                self._reference_refresh_in_progress = False

        threading.Thread(target=_refresh, name="reference-refresh", daemon=True).start()

    def _on_model_load_failure(self, process_id: int, model_name: str) -> None:
        """Handle a child's report that it failed to load ``model_name`` (the poison-model path).

        Records the failure against the model (not the slot). Once the model crosses the quarantine
        threshold it is taken out of rotation: every queued job for it is faulted non-retryably so the horde
        reissues them elsewhere instead of the worker re-dispatching the same unloadable model into a
        pool-wide recovery storm. The scheduler separately refuses to preload a quarantined model. The job the
        failing process itself held is faulted by the slot-replacement path; this sweeps the *rest* of the
        backlog for that model.
        """
        newly_quarantined = self._process_lifecycle.record_model_load_failure(process_id, model_name)
        if not newly_quarantined:
            return

        faulted = 0
        for job in list(self._job_tracker.jobs_pending_inference):
            if job.model == model_name and job not in self._job_tracker.jobs_in_progress:
                self._job_tracker.handle_job_fault_now(job, retryable=False)
                faulted += 1
        if faulted:
            logger.warning(
                f"Quarantined model {model_name}: faulted {faulted} queued job(s) for reissue to the horde.",
            )

    async def _process_control_loop(self) -> None:
        self._download_coordinator.download_wait_started = time.time()
        self._gpu_sampler.start()
        if self._enable_background_downloads:
            self._process_lifecycle.start_download_process()
            # Both the safety and inference processes are started lazily once their required models are on
            # disk; the download coordinator checks model availability from the handler and each tick.
            # Starting the safety process up front would make it
            # download the ~2.3GB safety models synchronously (and invisibly) in its constructor.
        else:
            # Without a download process there is nothing to defer to: start everything up front and let
            # the safety process fetch its own models (the legacy behaviour for tests/harness/dry-run).
            self._process_lifecycle.start_safety_processes()
            self._download_coordinator.safety_processes_started = True
            self._process_lifecycle.start_inference_processes()
            self._download_coordinator.inference_processes_started = True

        while True:
            try:
                if self.stable_diffusion_reference is None:
                    return
                # Watch for an externally-created .abort file as a signal-less
                # abort trigger (e.g. for process managers that cannot send signals).
                if os.path.exists(".abort"):
                    logger.warning("Found .abort file; aborting immediately")
                    self._abort()
                    break
                if not await self._control_loop_tick():
                    self._start_timed_shutdown()
                    break
            except CancelledError as e:
                self._shutdown()
                logger.debug(f"CancelledError: {e}")
            except Exception as e:
                # A failure in a control tick must not abandon in-flight work by killing the loop:
                # log, initiate a graceful shutdown, and fall through to the orderly teardown below
                # (the timed-shutdown backstop bounds it).
                logger.error(f"Unexpected error in control loop; shutting down gracefully: {e}")
                logger.exception(e)
                self._shutdown()
                self._start_timed_shutdown()
                break

        while len(self._job_tracker.jobs_pending_inference) > 0:
            await asyncio.sleep(0.2)
            await self.receive_and_handle_process_messages()
            self.detect_deadlock()
            self._process_lifecycle.replace_hung_processes()
            await asyncio.sleep(0.2)

        self._gpu_sampler.stop()
        self._process_lifecycle.end_inference_processes(force=True)
        self._process_lifecycle.end_safety_processes()
        self._process_lifecycle.end_download_process()

        logger.info("Shutting down process manager")
        self._state.shut_down = True
        for process in self._process_map.values():
            process.mp_process.terminate()
            process.mp_process.join(0.2)

        await asyncio.sleep(0.2)

        return

    def detect_deadlock(self) -> None:
        """Detect if there are jobs in the queue but no processes doing anything."""
        self._message_dispatcher.detect_deadlock()

    _DISK_SAMPLE_INTERVAL_SECONDS = 30.0

    @staticmethod
    def _disk_paths_to_monitor() -> list[Path]:
        """Return the disk paths whose free space matters to the worker."""
        paths = [Path.cwd()]
        cache_home = os.getenv("AIWORKER_CACHE_HOME")
        if cache_home:
            paths.append(Path(cache_home))
        return paths

    @staticmethod
    def _resolve_lora_paths() -> tuple[Path, Path] | None:
        """Return ``(lora_reference_json, lora_volume_dir)`` for disk-floor checks, or ``None``.

        Resolved from ``horde_model_reference`` (torch-free), so the main process can read the
        persisted ad-hoc cache size and sample the cache volume without importing the inference stack.
        """
        try:
            from horde_model_reference import horde_model_reference_paths

            legacy_path = Path(horde_model_reference_paths.legacy_path)
        except Exception as resolve_error:  # noqa: BLE001 - the disk floor is best-effort
            logger.warning(f"Could not resolve LoRA reference path for disk-floor checks: {resolve_error}")
            return None
        return legacy_path / "lora.json", legacy_path

    def _sample_disk_space(self) -> None:
        """Sample disk free space at most every `_DISK_SAMPLE_INTERVAL_SECONDS`."""
        if time.time() - self._last_disk_sample_time < self._DISK_SAMPLE_INTERVAL_SECONDS:
            return
        self._last_disk_sample_time = time.time()
        self._disk_monitor.sample()
        self._evaluate_lora_disk_exhaustion()

    def _evaluate_lora_disk_exhaustion(self) -> None:
        """Update ``lora_disk_exhausted`` from free space vs. the floor and the evictable ad-hoc cache.

        LoRAs are disabled only when the volume is below its floor *and* evicting every ad-hoc LoRA
        still would not clear it, so a recoverable shortfall is left to the inference-side eviction
        (which runs per LoRA job) rather than latching the worker out of LoRA work. Transitions are
        logged prominently because, left unaddressed, this stops the worker serving any LoRA jobs.
        """
        floor_mb = self.bridge_data.min_lora_disk_free_gb * 1024
        if self._lora_paths is None or not self.bridge_data.allow_lora or floor_mb <= 0:
            self._state.lora_disk_exhausted = False
            return

        reference_path, volume_dir = self._lora_paths
        free_mb_value = free_mb(volume_dir)
        if free_mb_value is None:
            return  # Keep the prior verdict when the volume can't be sampled.

        evictable_mb = read_evictable_adhoc_mb(reference_path)
        exhausted = is_lora_disk_exhausted(
            free_mb_value=free_mb_value,
            floor_mb=floor_mb,
            evictable_adhoc_mb=evictable_mb,
        )
        if exhausted and not self._state.lora_disk_exhausted:
            logger.warning(
                f"LoRA cache volume is critically low ({free_mb_value / 1024:.1f} GB free, floor "
                f"{floor_mb / 1024:.1f} GB) and evicting all ad-hoc LoRAs ({evictable_mb / 1024:.1f} GB) "
                "cannot clear it. Suppressing LoRA support until disk space is freed.",
            )
        elif not exhausted and self._state.lora_disk_exhausted:
            logger.success("LoRA cache volume recovered above its free-space floor; resuming LoRA support.")
        self._state.lora_disk_exhausted = exhausted

    def _build_performance_model(self) -> PerformanceModel:
        """Construct the performance model, seeding from the last benchmark report when one exists.

        Under test the model is purely in-memory: no app-state read, no benchmark import chain, and no
        perf file, so CI never touches the working directory or a prior benchmark. The model is unit
        tested directly. A missing or unreadable benchmark report yields an empty seed (the model then
        relies on self-calibration alone), never an error.
        """

        def resolve_baseline(model_name: str) -> str | None:
            baseline = self._model_metadata.get_baseline(model_name)
            return str(baseline) if baseline is not None else None

        if os.environ.get("AI_HORDE_TESTING"):
            return PerformanceModel(baseline_resolver=resolve_baseline)

        seed: dict[str, float] = {}
        last_benchmark = AppStateStore().load().last_benchmark
        if last_benchmark is not None:
            seed = load_seed_its_by_signature(last_benchmark.results_dir)
            if seed:
                logger.info(f"Seeded performance model with {len(seed)} tier baseline(s) from the last benchmark.")

        return PerformanceModel(
            seed_its_by_signature=seed,
            path=default_app_state_dir() / PERF_MODEL_FILENAME,
            baseline_resolver=resolve_baseline,
        )

    def _on_job_metrics(self, message: HordeJobMetricsMessage) -> None:
        """Fan a child's per-job metrics message out to the run metrics and the performance model."""
        self._run_metrics.on_job_metrics(message)
        self._performance_model.on_job_metrics(message)

    def _on_job_finalized(self, tracked: TrackedJob, completed_job_info: HordeJobInfo) -> None:
        """Fan a finalized job out to the run metrics (stage latencies) and the performance model (calibration)."""
        self._run_metrics.on_job_finalized(tracked, completed_job_info)
        self._performance_model.on_job_finalized(tracked, completed_job_info)

    def _record_process_crash(self, process_info: HordeProcessInfo, reason: str) -> None:
        """Forward a process recovery event to the run-metrics aggregator."""
        self._run_metrics.record_process_crash(
            process_id=process_info.process_id,
            process_launch_identifier=process_info.process_launch_identifier,
            last_state=process_info.last_process_state.name,
            reason=reason,
        )

    def get_run_metrics_snapshot(self) -> RunMetricsSnapshot:
        """Return the run-wide metrics snapshot (stage latencies, downloads, high-waters, crashes)."""
        phase, process_state_summary = self.describe_run_phase()
        return self._run_metrics.snapshot(
            num_process_recoveries=self._process_lifecycle._num_process_recoveries,
            num_job_slowdowns=self._job_submitter.num_job_slowdowns,
            time_spent_no_jobs_available=self._job_popper.time_spent_no_jobs_available,
            disk_min_free_bytes=self._disk_monitor.min_free_bytes,
            phase=phase,
            process_state_summary=process_state_summary,
        )

    def describe_run_phase(self) -> tuple[str, str]:
        """Describe what the worker is doing right now as ``(phase, per-process summary)``.

        Gives benchmark live progress a human-readable sense of motion through the long, otherwise
        silent cold start (process spawn, hordelib/GPU init, model load, first job), so a slow level
        reads as "still working" rather than "hung". Cheap and side-effect-free; safe to call often.
        """
        inference = [p for p in self._process_map.values() if p.process_type == HordeProcessType.INFERENCE]
        safety = [p for p in self._process_map.values() if p.process_type == HordeProcessType.SAFETY]

        # Lead each slot with its temperature so a primed slot reads as primed, not idle: a resident model a
        # queued job will use (next) is distinct from a resident model nothing needs yet (warm) and from an
        # empty slot (cold), though all three report WAITING_FOR_JOB. The raw state is kept after the colon so
        # existing log greps on state names still match.
        pending_models = frozenset(
            job.model for job in self._job_tracker.jobs_pending_inference if job.model is not None
        )

        def _slot(prefix: str, process_info: HordeProcessInfo) -> str:
            temperature = classify_process_temperature(
                state=process_info.last_process_state.name,
                loaded_model=process_info.loaded_horde_model_name,
                pending_models=pending_models,
            )
            return f"{prefix}#{process_info.process_id}={temperature.value}:{process_info.last_process_state.name}"

        summary_parts = [_slot("inf", p) for p in inference]
        summary_parts += [_slot("safety", p) for p in safety]
        process_summary = " ".join(summary_parts)

        if self._state.shutting_down:
            return "draining in-flight work", process_summary

        jobs_in_progress = len(self._job_tracker.jobs_in_progress)
        if jobs_in_progress > 0:
            return f"running inference ({jobs_in_progress} in progress)", process_summary
        if self._job_tracker.total_num_completed_jobs > 0:
            return "waiting for next job", process_summary
        if any(p.can_accept_job() for p in inference):
            return "ready; waiting for first job", process_summary
        if any(p.last_process_state == HordeProcessState.PROCESS_STARTING for p in inference):
            return "initializing inference process (loading GPU/model stack; first start is slow)", process_summary
        if self._enable_background_downloads and not self._download_coordinator.inference_processes_started:
            return "waiting for model download / disk scan", process_summary
        return "starting worker processes", process_summary

    _DUTY_CYCLE_SNAPSHOT_WINDOW_SECONDS = 60.0
    """Rolling window for the duty-cycle figure published to the TUI/insights (recent, lightly smoothed)."""

    _DUTY_CYCLE_REPORT_INTERVAL_SECONDS = 180.0
    """How often the duty-cycle health line is logged, and the window each report covers, so the NVML
    mean, the per-job attribution, and the no-jobs share all describe the same elapsed period."""

    _DUTY_CYCLE_TARGET_PERCENT = 90.0
    """The duty cycle the worker drives toward on a reference machine; below it leaves uptime on the table."""

    _DUTY_CYCLE_WARN_PERCENT = 75.0
    """At or above this (but below target) the shortfall is noted at INFO; below it escalates to WARNING."""

    def _maybe_log_duty_cycle(self) -> None:
        """Periodically log GPU duty cycle and, when it is low, where the wall-clock went.

        The number is the same one the TUI shows; the value added here is the *attribution* on the same
        line (per-job queue/safety/submit/model-load gaps plus a demand-vs-efficiency split), so an
        operator grepping ``GPU duty cycle`` across many workers' logs can see *why* uptime dropped with
        no tracing backend. Throttled, and quiet (DEBUG) when the worker is healthy; a worker the horde
        simply left idle is reported as demand-limited, never as a worker fault.
        """
        now = time.time()

        # Seed the baseline on the first call so the first real report measures a known interval.
        if self._last_duty_cycle_log_time == 0.0:
            self._last_duty_cycle_log_time = now
            self._last_no_jobs_seconds_at_duty_log = self._job_popper.time_spent_no_jobs_available
            return

        window_seconds = now - self._last_duty_cycle_log_time
        if window_seconds < self._DUTY_CYCLE_REPORT_INTERVAL_SECONDS:
            return

        metrics = self.get_run_metrics_snapshot()

        # Cache the first-ever inference start (monotonically the minimum across all retained jobs) so the
        # duty cycle measures only from when the GPU started doing job work, never the cold-boot warm-up.
        if self._first_inference_started_at is None:
            inference_starts = [
                started
                for job in metrics.jobs
                if (started := job.stage_timestamps.get("INFERENCE_IN_PROGRESS")) is not None
            ]
            if inference_starts:
                self._first_inference_started_at = min(inference_starts)

        nvml_mean = self._gpu_sampler.mean_percent(
            window_seconds=window_seconds,
            not_before=self._first_inference_started_at,
        )
        nvml_busy = self._gpu_sampler.busy_fraction(
            window_seconds=window_seconds,
            not_before=self._first_inference_started_at,
        )

        window_start = self._last_duty_cycle_log_time
        jobs_in_window = [
            job for job in metrics.jobs if (job.stage_timestamps.get("FINALIZED") or 0.0) >= window_start
        ]

        no_jobs_total = self._job_popper.time_spent_no_jobs_available
        no_jobs_in_window = max(0.0, no_jobs_total - self._last_no_jobs_seconds_at_duty_log)

        churn_counts: dict[str, int] = {
            kind: sum(1 for stamp in times if stamp >= window_start)
            for kind, times in metrics.churn_event_times.items()
        }

        # Advance the window before logging so a quiet report never widens the next one's denominator.
        self._last_duty_cycle_log_time = now
        self._last_no_jobs_seconds_at_duty_log = no_jobs_total

        summary = summarize_duty_cycle(
            jobs_in_window,
            window_seconds=window_seconds,
            time_spent_no_jobs_available=no_jobs_in_window,
            nvml_mean_percent=nvml_mean,
            nvml_busy_fraction=nvml_busy,
            churn_counts=churn_counts,
        )
        self._log_duty_cycle_summary(summary, metrics.process_state_summary)

    def _log_duty_cycle_summary(self, summary: DutyCycleSummary, process_state_summary: str) -> None:
        """Emit one structured ``GPU duty cycle`` line for ``summary`` at a severity matched to the cause."""
        duty = summary.effective_duty_percent()
        if duty is None:
            return  # Nothing measured this window (no GPU telemetry and no completed jobs to attribute).

        # A worker the horde left without work is not inefficient; never alarm for demand-limited idle.
        demand_limited = summary.completed_jobs == 0 and summary.is_demand_limited()

        busy_str = f"{summary.nvml_busy_fraction:.0%}" if summary.nvml_busy_fraction is not None else "n/a"
        head = (
            f"GPU duty cycle {duty:.0f}% over last {summary.window_seconds:.0f}s "
            f"(target {self._DUTY_CYCLE_TARGET_PERCENT:.0f}%, source={summary.headline_source()}, busy={busy_str})"
        )

        explanation_parts: list[str] = []
        if summary.no_jobs_available_fraction:
            explanation_parts.append(
                f"{summary.no_jobs_available_fraction:.0%} of the window had no jobs available "
                "(horde demand, not the worker)",
            )
        gaps = summary.format_gap_summary()
        if gaps:
            explanation_parts.append(f"biggest worker-side gaps: {gaps}")
        churn = summary.format_churn_summary()
        if churn:
            explanation_parts.append(f"reload churn: {churn}")
        explanation = "; ".join(explanation_parts) if explanation_parts else "no per-job attribution yet"

        context = (
            f"jobs: {summary.completed_jobs} done | {len(self._job_tracker.jobs_pending_inference)} pending | "
            f"{len(self._job_tracker.jobs_in_progress)} in-flight; processes: {process_state_summary or 'n/a'}"
        )
        message = f"{head}. {explanation}. {context}"

        if demand_limited:
            logger.info(message)
        elif duty >= self._DUTY_CYCLE_TARGET_PERCENT:
            logger.debug(message)
        elif duty >= self._DUTY_CYCLE_WARN_PERCENT:
            logger.info(message)
        else:
            logger.warning(message)

    def _build_stage_age_line(self) -> str | None:
        """A one-line per-stage census with the oldest age in each stage, or None when nothing is tracked.

        Ordered along the pipeline so a backlog that is *aging* (not just deep), e.g. jobs sitting in
        SAFETY_CHECKING while inference keeps finishing, is obvious. Emitted only inside the already-rate-
        limited status dump, so it adds no new log frequency.
        """
        summary = self._job_tracker.stage_age_summary()
        if not summary:
            return None
        order = (
            JobStage.PENDING_INFERENCE,
            JobStage.INFERENCE_IN_PROGRESS,
            JobStage.PENDING_SAFETY_CHECK,
            JobStage.SAFETY_CHECKING,
            JobStage.PENDING_SUBMIT,
        )
        parts = [
            f"{stage.name.lower()}={summary[stage][0]} (oldest {summary[stage][1]:.0f}s)"
            for stage in order
            if stage in summary
        ]
        return "Pipeline stages: " + " | ".join(parts) if parts else None

    def print_status_method(self) -> None:
        """Print the status of the worker if it's time to do so."""
        reporter = StatusReporter(
            last_status_message_time=self._last_status_message_time,
            status_message_frequency=self._status_message_frequency,
        )

        if not reporter.should_print_status(self._state.last_pop_maintenance_mode):
            return

        # Gather active models
        active_models = {
            process.loaded_horde_model_name
            for process in self._process_map.values()
            if process.loaded_horde_model_name is not None
        }

        # Print status and get updated frequency
        updated_frequency = reporter.print_status(
            bridge_data=self.bridge_data,
            process_info_strings=self._process_map.get_process_info_strings(),
            api_messages_received=self._job_popper.api_messages_received,
            jobs_pending_inference=self._job_tracker.jobs_pending_inference,
            active_models=active_models,
            pending_megapixelsteps=self._job_tracker.get_pending_megapixelsteps(),
            num_jobs_total=self.num_jobs_total,
            total_num_completed_jobs=self._job_tracker.total_num_completed_jobs,
            num_jobs_faulted=self._job_tracker.num_jobs_faulted,
            num_job_slowdowns=self._job_submitter.num_job_slowdowns,
            num_process_recoveries=self._process_lifecycle._num_process_recoveries,
            time_spent_no_jobs_available=self._job_popper.time_spent_no_jobs_available,
            user_info=self.user_info,
            max_concurrent_inference_processes=self.max_concurrent_inference_processes,
            device_map=self._device_map,
            too_many_consecutive_failed_jobs=self._state.too_many_consecutive_failed_jobs,
            too_many_consecutive_failed_jobs_time=self._state.too_many_consecutive_failed_jobs_time,
            too_many_consecutive_failed_jobs_wait_time=self._state.too_many_consecutive_failed_jobs_wait_time,
            session_start_time=self.session_start_time,
            shutting_down=self._state.shutting_down,
            jobs_pending_safety_check=len(self._job_tracker.jobs_pending_safety_check),
            jobs_being_safety_checked=len(self._job_tracker.jobs_being_safety_checked),
            jobs_in_progress=len(self._job_tracker.jobs_in_progress),
            total_ram_gigabytes=self.total_ram_gigabytes,
            system_memory=self._sample_system_memory(),
            download_status=self._model_availability.status,
            download_plan=self._download_coordinator.get_download_plan_summary(),
            stage_age_line=self._build_stage_age_line(),
        )

        self._last_status_message_time = reporter.last_status_message_time
        self._status_message_frequency = updated_frequency

    _supervisor_publish_min_interval = 0.0
    """A hard floor between snapshots regardless of change (0 = publish every tick when state changed)."""
    _supervisor_publish_floor_interval = 1.0
    """Maximum seconds between snapshots when nothing changes; a heartbeat so the TUI knows we're alive."""

    def _handle_supervisor_commands(self) -> None:
        """Drain and apply any control commands from a supervising frontend (no-op if unsupervised)."""
        if self._supervisor is None:
            return
        for command in self._supervisor.drain_commands():
            try:
                self._apply_supervisor_command(command)
            except Exception as e:
                logger.warning(f"Failed to apply supervisor command {command.command.name}: {e}")
        if self._supervisor.closed:
            self._supervisor = None

    def _apply_supervisor_command(self, command: SupervisorControlMessage) -> None:
        """Dispatch one supervisor command onto the worker's existing control mechanisms."""
        match command.command:
            case SupervisorCommand.PAUSE | SupervisorCommand.DRAIN:
                self._state.supervisor_paused = True
                logger.info("Supervisor requested pause: no new jobs will be popped (in-flight jobs finish).")
            case SupervisorCommand.RESUME:
                self._state.supervisor_paused = False
                logger.info("Supervisor requested resume: job popping re-enabled.")
                # An operator resume may also lift any horde-side maintenance the worker is in, but only
                # when the operator opted into that via remove_maintenance_on_init; otherwise a local
                # resume must never silently clear server-side maintenance (use the explicit toggle).
                if self.bridge_data.remove_maintenance_on_init:
                    threading.Thread(
                        target=self._set_server_maintenance_safe,
                        args=(False,),
                        name="resume-clear-maintenance",
                        daemon=True,
                    ).start()
            case SupervisorCommand.RESTART_PROCESS:
                if command.process_id is None:
                    logger.warning("RESTART_PROCESS supervisor command missing process_id; ignoring.")
                    return
                process_info = self._process_map.get(command.process_id)
                if process_info is None or process_info.process_type != HordeProcessType.INFERENCE:
                    logger.warning(f"RESTART_PROCESS: no inference process with id {command.process_id}.")
                    return
                logger.warning(f"Supervisor requested restart of inference process {command.process_id}.")
                self._process_lifecycle._replace_inference_process(process_info)
            case SupervisorCommand.RELOAD_CONFIG:
                logger.info("Supervisor requested config reload from disk.")
                # Off the control loop: the resolve is network-bound and must not stall the worker.
                self._bridge_data_reloader.schedule_config_reload()
            case SupervisorCommand.SET_CONCURRENCY:
                self._apply_set_concurrency(command.target_threads, command.target_processes)
            case SupervisorCommand.PAUSE_DOWNLOADS:
                logger.info("Supervisor requested download pause.")
                self._process_lifecycle.set_download_controls(paused=True)
            case SupervisorCommand.RESUME_DOWNLOADS:
                logger.info("Supervisor requested download resume.")
                self._process_lifecycle.set_download_controls(paused=False)
            case SupervisorCommand.SET_DOWNLOAD_RATE_LIMIT:
                rate = command.download_rate_limit_kbps or 0
                limit_label = "unlimited" if rate == 0 else f"{rate} KB/s"
                logger.info(f"Supervisor set download rate limit to {limit_label}.")
                self._process_lifecycle.set_download_controls(rate_limit_kbps=rate)
            case SupervisorCommand.DOWNLOADS_ONLY_HOLD:
                self._download_coordinator.enter_downloads_only_hold()
            case SupervisorCommand.GO_LIVE:
                self._download_coordinator.leave_downloads_only_hold()
            case SupervisorCommand.DOWNLOAD_MODELS:
                self._download_coordinator.download_models_on_demand(
                    command.download_model_names, include_aux=command.download_include_aux
                )
            case SupervisorCommand.SET_SERVER_MAINTENANCE:
                enabled = bool(command.server_maintenance_enabled)
                logger.warning(
                    f"Supervisor requested server-side maintenance {'ON' if enabled else 'OFF'} (horde API).",
                )
                # The horde API call is blocking; run it off the control loop so a slow or unreachable
                # horde can never stall the worker's tick.
                threading.Thread(
                    target=self._set_server_maintenance_safe,
                    args=(enabled,),
                    name="set-server-maintenance",
                    daemon=True,
                ).start()
            case SupervisorCommand.SHUTDOWN:
                logger.warning("Supervisor requested shutdown.")
                # Graceful first: drain in-flight work via the normal shutdown path; the
                # (drain-aware, idempotent) timed-shutdown is only the force-kill backstop.
                self._shutdown()
                self._start_timed_shutdown()
            case SupervisorCommand.SET_STATS_EXPORT:
                enabled = bool(command.stats_export_enabled)
                import horde_worker_regen

                self._run_metrics.set_stats_export(enabled, worker_version=horde_worker_regen.__version__)
                logger.info(f"Supervisor {'enabled' if enabled else 'disabled'} stats JSONL export.")

    def _apply_set_concurrency(self, target_threads: int | None, target_processes: int | None) -> None:
        """Adjust the live concurrent-inference cap and/or running inference-process count.

        Both knobs are independent: ``target_threads`` changes how many inferences may run at once
        (clamped to the provisioned ceiling), while ``target_processes`` changes how many inference
        processes are staged. The benchmark drives both per level; an operator can use either as a
        memory/VRAM-pressure lever.
        """
        if target_threads is not None:
            applied = self._runtime_config.set_effective_max_threads(target_threads)
            logger.info(f"Supervisor set concurrent-inference cap to {applied} (requested {target_threads}).")
        if target_processes is not None:
            result = self._process_lifecycle.scale_inference_processes(target_processes)
            self._download_coordinator.inference_processes_started = True
            logger.info(f"Supervisor scaled inference processes to {result} (requested {target_processes}).")

    def install_benchmark_scenario(
        self,
        *,
        jobs: list[ImageGenerateJobPopResponse] | None,
        alchemy_forms: list[AlchemyFormSpec] | None = None,
    ) -> None:
        """Swap in a fresh canned scenario and reset per-level metrics (warm benchmark worker).

        The worker keeps running between levels; this replaces the job/alchemy sources it pops from
        and clears the aggregated run metrics so the next level's numbers start clean. Completion is
        tracked by the caller via job-tracker count deltas (the tracker itself is not reset).
        """
        # Always install concrete (possibly empty) sources: a None job source under skip_api would
        # make the popper fall back to the default cycling scenario, polluting the level.
        self._job_popper.set_canned_job_source(CannedJobSource(jobs or []))
        self._alchemy_coordinator.set_canned_alchemy_source(CannedAlchemySource(alchemy_forms or []))
        self._run_metrics.reset()
        # The recovery counter is cumulative for the worker's lifetime; the warm benchmark reuses one
        # worker across levels, so it must be zeroed here too or each level after the first recovery
        # would inherit a non-zero count and be failed for a recovery it never had.
        self._process_lifecycle.reset_recovery_counter()

    def _supervisor_state_signature(self) -> tuple[object, ...]:
        """A cheap fingerprint of the display-relevant worker state.

        Publishing is gated on this changing (plus a periodic floor): it captures the per-process states,
        sampling progress, and headline counters that the dashboards render; but deliberately omits
        constantly-jittering memory/kudos figures, which ride the floor refresh instead. Computing it each
        tick avoids the cost of a full snapshot build (notably ``run_metrics.snapshot``) when idle.
        """
        per_process = tuple(
            (
                info.process_id,
                info.last_process_state,
                info.last_current_step,
                info.last_total_steps,
                info.loaded_horde_model_name,
            )
            for info in self._process_map.values()
        )
        # The download process lives outside the process map, so its phase/current-download is folded in
        # here too; otherwise a startup that is only downloading (e.g. the required safety models, before
        # any process is up) would ride the 2s heartbeat and read as a frozen frame for that window.
        download_status = self._model_availability.status
        download_fingerprint: tuple[object, ...] = (
            (download_status.phase, download_status.current.model_name if download_status.current else None)
            if download_status is not None
            else ()
        )
        return (
            per_process,
            download_fingerprint,
            self.num_jobs_total,
            self._job_tracker.total_num_completed_jobs,
            self._job_tracker.num_jobs_faulted,
            len(self._job_tracker.jobs_in_progress),
            len(self._job_tracker.jobs_pending_inference),
            len(self._job_tracker.jobs_pending_safety_check),
            len(self._job_tracker.jobs_being_safety_checked),
            self._state.last_pop_maintenance_mode or self._state.supervisor_paused,
            self._state.shutting_down,
            self._user_info_failed,
        )

    def _publish_supervisor_snapshot(self) -> None:
        """Push a worker-state snapshot when display state changed, or at the periodic heartbeat floor.

        Snapshots go out whenever the cheap state signature changes (so transitions and sampling progress
        surface within one control-loop tick, ~2 Hz) and at least every ``_supervisor_publish_floor_interval``
        seconds otherwise. A hard ``_supervisor_publish_min_interval`` floor can rate-limit bursts. The send
        itself is non-blocking (the channel's daemon thread owns the pipe), so this never stalls the loop.
        """
        if self._supervisor is None:
            return
        now = time.time()
        since_last = now - self._last_supervisor_publish_time
        if since_last < self._supervisor_publish_min_interval:
            return

        signature = self._supervisor_state_signature()
        changed = signature != self._last_supervisor_signature
        if not changed and since_last < self._supervisor_publish_floor_interval:
            return

        self._last_supervisor_publish_time = now
        self._last_supervisor_signature = signature
        try:
            snapshot = self._build_worker_state_snapshot()
        except Exception as e:
            logger.debug(f"Failed to build supervisor snapshot: {e}")
            return
        if not self._supervisor.send_snapshot(snapshot):
            self._supervisor = None

    def _safe_model_baseline(self, model_name: str | None) -> str | None:
        """Resolve a model's baseline as a plain string for the wire, swallowing lookup misses.

        The metadata may not know a model (custom checkpoints, a reference that has not loaded yet), so a
        miss yields None rather than failing snapshot assembly, which must never raise on the control loop.
        """
        if not model_name:
            return None
        try:
            baseline = self.get_model_baseline(model_name)
        except Exception:
            return None
        return str(baseline) if baseline is not None else None

    @staticmethod
    def _job_id_text(job_id: object | None) -> str:
        """Return a stable string form for Horde generation IDs and SDK job IDs."""
        if job_id is None:
            return ""
        root = getattr(job_id, "root", None)
        return str(root if root is not None else job_id)

    @staticmethod
    def _ledger_stage(stage: JobStage) -> WorkLedgerStage:
        """Map the internal job stage to the operator-facing work-ledger stage."""
        return {
            JobStage.PENDING_INFERENCE: WorkLedgerStage.QUEUED,
            JobStage.INFERENCE_IN_PROGRESS: WorkLedgerStage.INFERENCE,
            JobStage.DETACHED: WorkLedgerStage.PREPARING,
            JobStage.PENDING_SAFETY_CHECK: WorkLedgerStage.SAFETY,
            JobStage.SAFETY_CHECKING: WorkLedgerStage.SAFETY,
            JobStage.PENDING_SUBMIT: WorkLedgerStage.SUBMIT,
        }[stage]

    def _process_for_job(self, job_id: str) -> ProcessSnapshot | None:
        """Return the process snapshot currently referencing ``job_id``, if any."""
        for process in (ProcessSnapshot.from_process_info(info) for info in self._process_map.values()):
            if process.current_job_id == job_id:
                return process
        return None

    def _tracked_job_to_ledger_entry(self, tracked: TrackedJob, *, now: float) -> WorkLedgerEntry:
        """Project an active tracked job into the Overview work ledger."""
        sdk_job = tracked.sdk_api_job_info
        payload = sdk_job.payload
        features = JobFeatureSummary.from_payload(payload)
        job_id = self._job_id_text(tracked.job_id)
        process = self._process_for_job(job_id)
        stage = self._ledger_stage(tracked.stage)
        intent = {
            WorkLedgerStage.QUEUED: "waiting for model/process",
            WorkLedgerStage.PREPARING: "between orchestration stages",
            WorkLedgerStage.INFERENCE: "sampling" if process is not None and process.is_busy else "dispatched",
            WorkLedgerStage.SAFETY: "safety check",
            WorkLedgerStage.SUBMIT: "awaiting submit",
        }.get(stage)
        return WorkLedgerEntry(
            job_id=job_id,
            stage=stage,
            model=str(sdk_job.model) if sdk_job.model is not None else None,
            baseline=self._safe_model_baseline(str(sdk_job.model) if sdk_job.model is not None else None),
            process_id=process.process_id if process is not None else None,
            device_index=process.device_index if process is not None else tracked.last_dispatched_device_index,
            progress_current=process.last_current_step if process is not None else None,
            progress_total=process.last_total_steps if process is not None else None,
            iterations_per_second=process.last_iterations_per_second if process is not None else None,
            width=payload.width,
            height=payload.height,
            steps=payload.ddim_steps,
            features=features if not features.is_empty() else None,
            age_seconds=max(0.0, now - tracked.current_stage_since) if tracked.current_stage_since else None,
            intent=intent,
            raw_reason=getattr(self._inference_scheduler, "_dispatch_stall_last_reason", None),
        )

    def _recent_job_to_ledger_entry(self, job: RecentJobRecord) -> WorkLedgerEntry:
        """Project a finished job record into the Overview work ledger."""
        return WorkLedgerEntry(
            job_id=job.job_id,
            stage=WorkLedgerStage.FAULTED if job.faulted else WorkLedgerStage.COMPLETED,
            model=job.model_name,
            baseline=job.baseline,
            width=job.width,
            height=job.height,
            steps=job.steps,
            features=job.features,
            queue_wait_seconds=job.queue_wait_seconds,
            safety_seconds=job.safety_seconds,
            e2e_seconds=job.e2e_seconds,
            faulted=job.faulted,
            intent="recent fault" if job.faulted else "recent completion",
        )

    _ALCHEMY_STAGE_TO_LEDGER: dict[str, WorkLedgerStage] = {
        "pending": WorkLedgerStage.QUEUED,
        "in_flight": WorkLedgerStage.INFERENCE,
        "awaiting_submit": WorkLedgerStage.SUBMIT,
    }
    """Maps an alchemy form's pipeline stage onto the work-ledger's job stages so a form reads in the same
    queued -> processing -> submit vocabulary as an image job."""

    def _alchemy_status_to_ledger_entry(self, status: AlchemyFormStatus) -> WorkLedgerEntry:
        """Project one active alchemy form into a work-ledger row (form as model, resolution as size)."""
        return WorkLedgerEntry(
            job_id=status.form_id,
            stage=self._ALCHEMY_STAGE_TO_LEDGER.get(status.stage, WorkLedgerStage.INFERENCE),
            model=f"⚗ {status.form}",
            baseline=None,
            process_id=status.process_id,
            width=status.width,
            height=status.height,
        )

    def _build_work_ledger(self, recent_jobs: list[RecentJobRecord]) -> list[WorkLedgerEntry]:
        """Build active rows first, then recent completed/faulted rows for the Overview work ledger.

        Active image jobs come first, then an alchemist worker's active forms (each shown with the form as
        its model and the source-image resolution as its size), then recent finished work, all capped.
        """
        now = time.time()
        rows = [self._tracked_job_to_ledger_entry(tracked, now=now) for tracked in self._job_tracker.tracked_jobs()]
        rows.extend(
            self._alchemy_status_to_ledger_entry(status) for status in self._alchemy_coordinator.active_form_statuses()
        )
        active_ids = {row.job_id for row in rows}
        for job in reversed(recent_jobs):
            if len(rows) >= WORK_LEDGER_ENTRIES_IN_SNAPSHOT:
                break
            if job.job_id in active_ids:
                continue
            rows.append(self._recent_job_to_ledger_entry(job))
        return rows[:WORK_LEDGER_ENTRIES_IN_SNAPSHOT]

    @staticmethod
    def _format_remaining_seconds(seconds: float) -> str:
        """Format a remaining-time value compactly for the orchestrator intent (e.g. ``2m 30s``).

        Kept here rather than importing ``human_duration`` from the TUI layer to avoid a dependency
        inversion: the orchestrator owns the intent, the TUI only renders it.
        """
        total = int(max(seconds, 0))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    # Mapping of AI Horde API skip-reason keys to operator-facing phrases, so the "Why" line
    # in the Now/Next/Why strip is intelligible rather than raw key=value pairs. Unknown keys
    # fall through unchanged.
    _SKIP_REASON_PHRASES: dict[str, str] = {
        "models": "no matching models offered",
        "nsfw": "NSFW mismatch",
        "max_pixels": "max pixels exceeded",
        "trusted": "trusted-worker-only jobs",
        "allow_img2img": "img2img not allowed",
        "allow_controlnet": "controlnet not allowed",
        "allow_sdxl_controlnet": "SDXL controlnet not allowed",
        "allow_lora": "LoRA not offered",
        "allow_post_processing": "post-processing not allowed",
        "allow_painting": "inpainting not allowed",
        "safety_backlog": "safety backlogged",
        "kudos": "insufficient kudos",
        "worker_id": "worker-id mismatch",
    }

    def _build_orchestration_intent(self) -> OrchestrationIntentSnapshot:
        """Summarize what the orchestrator is doing now and what it is waiting on next.

        The cascade is priority-ordered: the first matching condition wins, so higher-priority
        states (shutdown, pauses, holds) dominate lower-priority operational states (sampling,
        safety, idle). Each early-return path that stops job popping should have a distinct
        intent so the operator can see *why* the worker is not taking new work.
        """
        raw_gate = getattr(self._inference_scheduler, "_dispatch_stall_last_reason", None)
        head = next(
            (job for job in self._job_tracker.jobs_pending_inference if job not in self._job_tracker.jobs_in_progress),
            None,
        )
        target_job_id = self._job_id_text(head.id_) if head is not None else None
        target_model = str(head.model) if head is not None and head.model is not None else None

        # Terminal / operator-commanded states
        if self._state.shutting_down:
            return OrchestrationIntentSnapshot(
                summary="Draining before shutdown.",
                why="In-flight work finishes first.",
            )

        if self._state.supervisor_paused:
            return OrchestrationIntentSnapshot(summary="Paused by operator.", why="No new jobs are being popped.")

        # Download-only hold (operator-initiated pre-fetch posture)
        if self._state.downloads_only_hold:
            return OrchestrationIntentSnapshot(
                summary="Download-only mode: pre-fetching models without serving.",
                next_action="Use the Downloads tab to go live when ready.",
                why="Inference and safety are held; the GPU is not committed.",
            )

        # Self-throttle (worker-initiated, auto-resuming)
        if self._state.self_throttle_paused:
            remaining = max(0.0, self._state.self_throttle_paused_until - time.time())
            why = "The worker is backing off before the horde forces maintenance."
            if remaining > 0:
                why += f" Resumes in ~{self._format_remaining_seconds(remaining)}."
            return OrchestrationIntentSnapshot(
                summary="Self-throttled after resource faults.",
                why=why,
            )

        # Background model download
        if self._model_availability.background_download_active:
            model_hint = ""
            status = self._model_availability.status
            if status is not None and status.active:
                model_hint = f" ({status.active[0].model_name})"
            return OrchestrationIntentSnapshot(
                summary=f"Downloading model assets{model_hint}.",
                next_action="Serve each model as soon as it is ready.",
                why="Feature/model readiness gates job popping.",
            )

        # Horde-forced maintenance (server-side pause)
        if self._state.last_pop_maintenance_mode:
            return OrchestrationIntentSnapshot(
                summary="In horde maintenance mode.",
                next_action="Press the Maintenance (horde) key to clear it when ready.",
                why="The horde has paused this worker; it will not be given new jobs until cleared.",
            )

        # Dispatch stall (head parked, no job dispatched this cycle)
        if raw_gate:
            return OrchestrationIntentSnapshot(
                summary="Holding dispatch.",
                next_action="Re-check the head job on the next scheduler tick.",
                why=raw_gate,
                raw_gate=raw_gate,
                target_job_id=target_job_id,
                target_model=target_model,
            )

        # Jobs in progress (sampling or post-processing)
        if self._job_tracker.jobs_in_progress:
            count = len(self._job_tracker.jobs_in_progress)
            sampling = self._process_map.num_busy_with_inference()
            post = self._process_map.num_busy_with_post_processing()
            if sampling > 0:
                summary = f"Sampling {sampling} job{'s' if sampling != 1 else ''}"
                if post > 0:
                    summary += f" (+ {post} in post-processing)"
                summary += "."
            elif post > 0:
                summary = f"Post-processing {post} job{'s' if post != 1 else ''}."
            else:
                summary = f"{count} job{'s' if count != 1 else ''} in progress."

            return OrchestrationIntentSnapshot(
                summary=summary,
                next_action=(f"Prepare {target_model}." if target_model else "Keep the pipeline fed."),
                target_job_id=target_job_id,
                target_model=target_model,
            )

        # A queued job is waiting but nothing is in progress
        if head is not None:
            if target_model and self._horde_model_map.is_model_loading(target_model):
                summary = f"Preloading {target_model}."
                why = "The queued job's model is loading into a process."
            else:
                summary = f"Preparing queued job {target_job_id[:8] if target_job_id else ''}.".strip()
                _find_resident = getattr(self._inference_scheduler, "_resident_process_for_job", None)
                resident = _find_resident(head) if _find_resident is not None else None
                if resident is not None:
                    if resident.last_process_state.name == "INFERENCE_STARTING":
                        why = f"Process {resident.process_id} is busy sampling."
                    elif resident.last_process_state.name == "INFERENCE_POST_PROCESSING":
                        why = f"Process {resident.process_id} is finishing post-processing."
                    elif resident.last_process_state.name == "DOWNLOADING_AUX_MODEL":
                        why = f"Process {resident.process_id} is downloading auxiliary models."
                    else:
                        state_phrase = resident.last_process_state.name.lower().replace("_", " ")
                        why = f"Process {resident.process_id} is {state_phrase}."
                else:
                    why = "Waiting for a resident model and an accepting process."
            return OrchestrationIntentSnapshot(
                summary=summary,
                next_action="Dispatch when the process can accept work.",
                why=why,
                target_job_id=target_job_id,
                target_model=target_model,
            )

        # Consecutive failure backoff
        if self._state.too_many_consecutive_failed_jobs:
            remaining = max(
                0.0,
                self._state.too_many_consecutive_failed_jobs_time + CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS - time.time(),
            )
            why = f"{self._state.consecutive_failed_jobs} consecutive job failures."
            if remaining > 0:
                why += f" Resumes in ~{self._format_remaining_seconds(remaining)}."
            return OrchestrationIntentSnapshot(
                summary="Paused after consecutive job failures.",
                next_action="Check logs and diagnostics to identify the fault.",
                why=why,
            )

        # Safety backlog backpressure (pops withheld to let the safety stage catch up)
        safety_backlog = len(self._job_tracker.jobs_pending_safety_check) + len(
            self._job_tracker.jobs_being_safety_checked
        )
        if safety_backlog > 0 and getattr(self._job_popper, "_is_post_inference_backlogged", lambda: False)():
            return OrchestrationIntentSnapshot(
                summary=f"Safety backlogged: withholding pops ({safety_backlog} waiting).",
                next_action="Pops resume when the safety stage catches up.",
                why="The safety stage is slower than inference; enable safety_on_gpu or speed safety up.",
            )

        # API error backoff
        if self._job_popper._pop_throttler.is_in_error_backoff:
            return OrchestrationIntentSnapshot(
                summary="Backing off after API errors.",
                next_action="Pops resume automatically after the backoff interval.",
                why="The horde API returned errors on recent pop attempts.",
            )

        # Warm-up rule: don't queue ahead until the first job completes
        if self._job_tracker.total_num_completed_jobs == 0 and self._job_tracker.jobs_pending_inference:
            return OrchestrationIntentSnapshot(
                summary="Warming up: withholding queue-ahead until the first job completes.",
                next_action="The queue will fill once the inaugural job succeeds.",
                why="If the worker is doomed to fail with one job, it is doomed to fail with two.",
            )

        # Safety stage activity (normal, not backlogged)
        safety = len(self._job_tracker.jobs_pending_safety_check) + len(self._job_tracker.jobs_being_safety_checked)
        if safety:
            return OrchestrationIntentSnapshot(
                summary=f"Safety checking {safety} job{'s' if safety != 1 else ''}.",
                next_action="Jobs will be submitted once safety completes.",
            )

        # Submission stage
        if self._job_tracker.jobs_pending_submit:
            return OrchestrationIntentSnapshot(
                summary="Submitting completed jobs to the horde.",
                next_action="Results are being uploaded and reported.",
            )

        # Alchemy active with no inference work
        alchemy_active = self._alchemy_coordinator.num_forms_in_flight + self._alchemy_coordinator.num_forms_pending
        if alchemy_active > 0:
            return OrchestrationIntentSnapshot(
                summary=f"Processing alchemy: {alchemy_active} form{'s' if alchemy_active != 1 else ''}.",
                next_action="Pop inference jobs when alchemy work completes.",
                why="Alchemy (upscale/face-fix/interrogate/caption) is using the GPU.",
            )

        # Last pop returned no job; surface the API's skip reasons in friendly prose
        if self._state.last_pop_skipped_reasons:
            phrases: list[str] = []
            for key, value in sorted(self._state.last_pop_skipped_reasons.items()):
                phrase = self._SKIP_REASON_PHRASES.get(key, key)
                phrases.append(f"{value} {phrase}")
            reasons = " · ".join(phrases)
            return OrchestrationIntentSnapshot(
                summary="Looking for matching horde work.",
                next_action="Pop again after the throttle interval.",
                why=f"Last pop skipped: {reasons}.",
            )

        # Default: ready and waiting
        return OrchestrationIntentSnapshot(summary="Ready for work.", next_action="Pop the next matching job.")

    def _build_pending_jobs_list(self) -> list[JobQueueEntry]:
        """Build a capped list of pending work for the overview queue display.

        Image jobs awaiting inference come first; an alchemist worker's popped-but-not-yet-dispatched forms
        follow, each shown with the form as its model and the source-image resolution as its size, so the
        queue reads the same way for both workloads.
        """
        entries: list[JobQueueEntry] = []
        for api_job in self._job_tracker.jobs_pending_inference[:PENDING_JOBS_IN_SNAPSHOT]:
            payload = api_job.payload
            candidate = JobFeatureSummary.from_payload(payload)
            features = candidate if not candidate.is_empty() else None
            model_name = str(api_job.model) if api_job.model is not None else "?"
            entries.append(
                JobQueueEntry(
                    job_id=str(api_job.id_.root) if api_job.id_ is not None else "",
                    model=model_name,
                    baseline=self._safe_model_baseline(model_name),
                    steps=payload.ddim_steps,
                    width=payload.width,
                    height=payload.height,
                    features=features,
                )
            )
        for status in self._alchemy_coordinator.active_form_statuses():
            if len(entries) >= PENDING_JOBS_IN_SNAPSHOT:
                break
            if status.stage != "pending":
                continue
            entries.append(
                JobQueueEntry(
                    job_id=status.form_id,
                    model=f"⚗ {status.form}",
                    baseline=None,
                    steps=None,
                    width=status.width,
                    height=status.height,
                    features=None,
                ),
            )
        return entries

    @staticmethod
    def _to_int_mb(value: float | None) -> int | None:
        """Round an MB figure to a whole MB for the wire, preserving None."""
        return int(round(value)) if value is not None else None

    @staticmethod
    def _process_rss_bytes(os_pid: int | None) -> int:
        """Sample one process's resident-set size (bytes) by OS pid, returning 0 if it cannot be read.

        Used for processes the worker does not get a self-reported RAM figure from (the download
        process). A dead pid, a permission error, or a missing pid all yield 0 rather than raising on
        the control loop.
        """
        if not os_pid:
            return 0
        import psutil

        try:
            return int(psutil.Process(os_pid).memory_info().rss)
        except (psutil.Error, OSError):
            return 0

    def _sample_system_memory(self) -> SystemMemorySummary:
        """Build the current system-RAM summary: total/available plus the worker's per-role RSS share.

        Inference and safety processes self-report their RSS in their periodic memory reports (already
        kept on the process map), so those are summed from there. The orchestrator (this process) and the
        background download process do not, so their RSS is sampled directly via psutil. All figures are
        resident-set size; see :mod:`system_memory` for why the per-role sum is an upper bound.
        """
        import psutil

        from horde_worker_regen.process_management.resources.system_memory import (
            ROLE_DOWNLOAD,
            ROLE_INFERENCE,
            ROLE_ORCHESTRATOR,
            ROLE_SAFETY,
            build_system_memory_summary,
        )

        virtual_memory = psutil.virtual_memory()

        inference_rss = 0
        safety_rss = 0
        for process_info in self._process_map.values():
            if process_info.process_type == HordeProcessType.INFERENCE:
                inference_rss += max(0, process_info.ram_usage_bytes)
            elif process_info.process_type == HordeProcessType.SAFETY:
                safety_rss += max(0, process_info.ram_usage_bytes)

        download_info = self._process_lifecycle.download_process_info
        download_rss = self._process_rss_bytes(download_info.os_pid if download_info is not None else None)

        try:
            orchestrator_rss = int(psutil.Process().memory_info().rss)
        except (psutil.Error, OSError):
            orchestrator_rss = 0

        return build_system_memory_summary(
            total_bytes=virtual_memory.total,
            available_bytes=virtual_memory.available,
            worker_rss_by_role={
                ROLE_ORCHESTRATOR: orchestrator_rss,
                ROLE_INFERENCE: inference_rss,
                ROLE_SAFETY: safety_rss,
                ROLE_DOWNLOAD: download_rss,
            },
        )

    def _whole_card_residency_status(self) -> WholeCardResidencyStatus:
        """Project the scheduler's whole-card residency state onto the wire model (MB rounded to int)."""
        state = self._inference_scheduler.whole_card_residency_state()
        return WholeCardResidencyStatus(
            possible=state.possible,
            enabled=state.enabled,
            safety_off_gpu_enabled=state.safety_off_gpu_enabled,
            cooldown_seconds=int(round(state.cooldown_seconds)),
            per_process_overhead_mb=int(round(state.per_process_overhead_mb)),
            total_vram_mb=int(round(state.total_vram_mb)) if state.total_vram_mb else 0,
            active=state.active,
            model=state.model,
            phase=state.phase,
            safety_paused=state.safety_paused,
            processes_now=state.processes_now,
            processes_target=state.processes_target,
            processes_max=state.processes_max,
            cooldown_remaining_seconds=state.cooldown_remaining_seconds,
            weights_mb=self._to_int_mb(state.weights_mb),
            reserve_mb=self._to_int_mb(state.reserve_mb),
            free_now_mb=self._to_int_mb(state.free_now_mb),
            free_if_alone_mb=self._to_int_mb(state.free_if_alone_mb),
            max_resident_processes=state.max_resident_processes,
        )

    def _update_pop_governors(self) -> None:
        """Feed the governor registry this tick's readings, so spells advance and ENTER/EXIT lines log.

        Runs every control-loop tick (not only when a snapshot is published), so the grep-friendly boundary
        lines and the session aggregates are produced whether or not a TUI is attached. Wrapped so a transient
        read failure on one cycle skips that update rather than disturbing the control loop, which must never
        raise here.
        """
        now = time.time()
        try:
            self._pop_governor_registry.update(self._collect_pop_governor_readings(now), now=now)
        except Exception as e:  # never let observability bookkeeping break the control loop
            logger.debug(f"Pop-governor update skipped this tick: {type(e).__name__} {e}")

    def _collect_pop_governor_readings(self, now: float) -> list[PopGovernorReading]:
        """Read every known pop/scheduling governor's current engagement into a list of readings.

        Each governor is a *condition* that holds back or reshapes job pops; the readings are levels (engaged
        now, why, expected remaining), and the registry derives the spell edges. Timers are supplied where a
        governor releases on a clock (residency cooldown, the switch/re-entry windows, the consecutive-failure
        and self-throttle pauses, the megapixelstep wait); the rest clear when their underlying state changes
        and report no fixed remaining.
        """
        bridge_data = self.bridge_data
        readings: list[PopGovernorReading] = []

        residency = self._inference_scheduler.whole_card_residency_state()
        readings.append(
            PopGovernorReading(
                name="whole_card_residency",
                label="Whole-card residency",
                active=residency.active,
                reason=(f"{residency.model} holds the card ({residency.phase})" if residency.active else None),
                expected_remaining_seconds=residency.cooldown_remaining_seconds if residency.active else None,
            ),
        )

        large = self._job_popper.large_model_governor_status(
            now=now,
            residency_active=self._inference_scheduler.is_whole_card_residency_active(),
        )
        readings.append(
            PopGovernorReading(
                name="large_model_switch",
                label="Large-model switch throttle",
                active=large.switch_active,
                reason=large.switch_reason,
                expected_remaining_seconds=large.switch_remaining_seconds,
            ),
        )
        readings.append(
            PopGovernorReading(
                name="large_model_reentry",
                label="Large-model re-entry cooldown",
                active=large.reentry_active,
                reason=large.reentry_reason,
                expected_remaining_seconds=large.reentry_remaining_seconds,
            ),
        )

        backlogged = self._job_popper.is_post_inference_backlogged()
        readings.append(
            PopGovernorReading(
                name="post_inference_backpressure",
                label="Post-inference backpressure",
                active=backlogged,
                reason="safety stage backlogged; holding pops so jobs do not age past their deadline"
                if backlogged
                else None,
            ),
        )

        held_unservable = [
            model
            for model in bridge_data.image_models_to_load
            if is_model_locally_unservable_for(bridge_data, self._job_tracker, model)
        ]
        readings.append(
            PopGovernorReading(
                name="unservable_model_holdback",
                label="Unservable-model holdback",
                active=bool(held_unservable),
                reason=(
                    f"{len(held_unservable)} model(s) held after repeated faults: {', '.join(sorted(held_unservable))}"
                )
                if held_unservable
                else None,
            ),
        )

        failure_pause = bool(self._state.too_many_consecutive_failed_jobs)
        failure_remaining = (
            CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS - (now - self._state.too_many_consecutive_failed_jobs_time)
            if failure_pause
            else None
        )
        readings.append(
            PopGovernorReading(
                name="consecutive_failure_pause",
                label="Consecutive-failure pause",
                active=failure_pause,
                reason="paused pops after three consecutive faulted jobs" if failure_pause else None,
                expected_remaining_seconds=max(0.0, failure_remaining) if failure_remaining is not None else None,
            ),
        )

        readings.append(
            PopGovernorReading(
                name="pop_error_backoff",
                label="Pop error-backoff",
                active=self._job_popper.is_in_error_backoff,
                reason="backing off the API after recent pop errors" if self._job_popper.is_in_error_backoff else None,
            ),
        )

        lora_by_disk = bool(bridge_data.allow_lora and self._state.lora_disk_exhausted)
        lora_by_download = bool(bridge_data.allow_lora and self._model_availability.background_download_active)
        lora_reason = None
        if lora_by_disk:
            lora_reason = "LoRA cache volume below its free-space floor"
        elif lora_by_download:
            lora_reason = "LoRA support suppressed while background downloads run"
        readings.append(
            PopGovernorReading(
                name="lora_pop_backoff",
                label="LoRA pop backoff",
                active=lora_by_disk or lora_by_download,
                reason=lora_reason,
            ),
        )

        self_throttle = bool(self._state.self_throttle_paused)
        throttle_remaining = (self._state.self_throttle_paused_until - now) if self_throttle else None
        readings.append(
            PopGovernorReading(
                name="self_throttle_pause",
                label="Self-throttle pause",
                active=self_throttle,
                reason="paused itself after resource/OOM faults to avoid dropping jobs" if self_throttle else None,
                expected_remaining_seconds=max(0.0, throttle_remaining) if throttle_remaining is not None else None,
            ),
        )

        mps_remaining = self._job_popper.megapixelstep_wait_remaining(bridge_data, now=now)
        readings.append(
            PopGovernorReading(
                name="megapixelstep_wait",
                label="Megapixelstep wait",
                active=mps_remaining is not None,
                reason="letting long-running jobs make progress before popping more"
                if mps_remaining is not None
                else None,
                expected_remaining_seconds=mps_remaining,
            ),
        )

        stickiness = getattr(bridge_data, "horde_model_stickiness", 0)
        loaded_count = sum(1 for p in self._process_map.values() if p.loaded_horde_model_name is not None)
        stickiness_active = bool(
            isinstance(stickiness, (int, float))
            and not isinstance(stickiness, bool)
            and stickiness > 0
            and len(bridge_data.image_models_to_load) > self.max_inference_processes
            and loaded_count >= self.max_inference_processes,
        )
        readings.append(
            PopGovernorReading(
                name="model_stickiness",
                label="Model stickiness",
                active=stickiness_active,
                reason="preferring already-resident models (stickiness)" if stickiness_active else None,
            ),
        )

        return readings

    @staticmethod
    def _round_optional_mb(value: float | None) -> int | None:
        """Round an optional MB reading for the supervisor snapshot."""
        return int(round(value)) if value is not None else None

    def _scheduling_governance_status(self) -> SchedulingGovernanceSnapshot:
        """Project scheduler governance decisions onto the wire snapshot the TUI renders."""
        ram_snapshot = self._inference_scheduler.latest_host_memory_governance_snapshot()
        if ram_snapshot is None:
            ram = RamGovernanceSnapshot()
        else:
            pause_remaining = (
                max(0.0, ram_snapshot.pop_pause_until - time.time()) if ram_snapshot.pop_pause_active else None
            )
            ram = RamGovernanceSnapshot(
                measured=True,
                under_pressure=ram_snapshot.verdict.under_pressure,
                reason=ram_snapshot.verdict.reason(),
                available_mb=self._round_optional_mb(ram_snapshot.verdict.available_mb),
                floor_mb=int(round(ram_snapshot.verdict.floor_mb)),
                total_mb=self._round_optional_mb(ram_snapshot.verdict.total_mb),
                pop_hold_active=self._state.ram_pressure_pop_hold,
                pop_pause_active=ram_snapshot.pop_pause_active,
                pop_pause_remaining_seconds=pause_remaining,
                draining_process_ids=sorted(ram_snapshot.draining_process_ids),
                shed_card_indices=sorted(ram_snapshot.shed_card_indices),
                restore_headroom_mb=int(round(ram_snapshot.restore_headroom_mb)),
                per_context_ram_estimate_mb=int(round(ram_snapshot.per_context_ram_estimate_mb)),
                per_process_ceiling_mb=self._round_optional_mb(ram_snapshot.per_process_ceiling_mb),
            )

        preload_status = self._inference_scheduler.latest_preload_admission()
        if preload_status is None:
            preload = PreloadAdmissionSnapshot()
        else:
            preload = PreloadAdmissionSnapshot(
                decision=str(preload_status.decision),
                model=preload_status.model,
                process_id=preload_status.process_id,
                reason=preload_status.reason,
                timestamp=preload_status.timestamp,
            )

        return SchedulingGovernanceSnapshot(ram=ram, preload=preload)

    def _pop_governors_status(self) -> PopGovernorsSnapshot:
        """Project the governor registry onto the wire snapshot the TUI renders."""
        elapsed = max(0.0, time.time() - self.session_start_time)
        views = self._pop_governor_registry.views(now=time.time(), session_elapsed_seconds=elapsed)
        governors = [
            PopGovernorStatus(
                name=view.name,
                label=view.label,
                active=view.active,
                reason=view.reason,
                current_spell_seconds=view.current_spell_seconds,
                expected_remaining_seconds=view.expected_remaining_seconds,
                triggers=view.triggers,
                total_active_seconds=view.total_active_seconds,
                fraction_of_session=view.fraction_of_session,
            )
            for view in views
        ]
        return PopGovernorsSnapshot(governors=governors, any_active=any(v.active for v in views))

    def _build_card_snapshots(self) -> list[CardSnapshot]:
        """Project per-card multi-GPU state onto wire models, one per driven card.

        A single-GPU host has exactly one card runtime, so it reports one ``CardSnapshot`` (the collapsed card
        the dashboard renders). Per-card residency, fault streaks and the jobs/hr source are keyed by real
        device index only when the worker drives more than one card; on a single-GPU host the worker-wide
        ``None`` key is used, matching how dispatch and streak bookkeeping key those facts. The VRAM/context
        figures filter the process map by the slot's pinned ``device_index`` (a real attribute, 0 on a
        single-GPU host), so the single-card figure equals the worker-wide one.
        """
        multi_gpu = len(self._card_runtimes) > 1
        cards: list[CardSnapshot] = []
        for device_index, card_runtime in sorted(self._card_runtimes.items()):
            fault_key = device_index if multi_gpu else None
            device_info = self._device_map.root.get(device_index)

            busy_contexts = sum(
                1
                for info in self._process_map.values()
                if info.process_type == HordeProcessType.INFERENCE
                and info.device_index == device_index
                and info.is_process_alive()
                and info.last_process_state
                in (HordeProcessState.INFERENCE_STARTING, HordeProcessState.INFERENCE_POST_PROCESSING)
            )

            residency_model, residency_phase = self._inference_scheduler.card_residency(fault_key)

            unservable_models: list[str] = []
            worst_fault_streak = 0
            for model in card_runtime.config.image_models_to_load:
                worst_fault_streak = max(
                    worst_fault_streak,
                    self._job_tracker.get_model_overbudget_fault_count(model, device_index=fault_key),
                )
                if is_model_locally_unservable_for(
                    card_runtime.config,
                    self._job_tracker,
                    model,
                    device_index=fault_key,
                ):
                    unservable_models.append(model)

            total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
            if total_vram_mb is None:
                total_vram_mb = card_runtime.total_vram_mb

            cards.append(
                CardSnapshot(
                    device_index=device_index,
                    device_name=device_info.device_name if device_info is not None else None,
                    kind=card_runtime.kind,
                    total_vram_mb=total_vram_mb,
                    free_vram_mb=self._process_map.get_free_vram_mb(device_index=device_index),
                    loaded_contexts=self._process_map.num_loaded_inference_processes(device_index=device_index),
                    busy_contexts=busy_contexts,
                    target_process_count=card_runtime.target_process_count,
                    max_concurrent_inference=card_runtime.max_concurrent_inference,
                    jobs_completed=self._job_tracker.get_card_inference_results(fault_key),
                    residency_model=residency_model,
                    residency_phase=residency_phase,
                    unservable_models=unservable_models,
                    worst_fault_streak=worst_fault_streak,
                ),
            )
        return cards

    def _enforce_alchemist_only_scale_down(self) -> None:
        """Collapse the inference fleet to one process per card once a runtime CPU-only build is detected.

        A worker whose install sentinel was never set (a manual CPU torch install) comes up sized for image
        generation. When an inference child reports a CPU-only torch build at runtime
        (``torch_build_cpu_only``), image generation is disabled and a single inference process per card is
        enough for the graph alchemy forms, so the extra contexts are reaped. This brings the runtime path
        to parity with the startup ``serves_image_generation=False`` sizing the install sentinel would have
        produced.

        Each card's ``target_process_count`` is lowered to one (authoritative for recovery placement, the
        VRAM/RAM budget, and any target-based scale-up), then idle contexts are reaped toward one. Run every
        control tick while the flag is set: lowering the target happens once (idempotent thereafter), and the
        reap retries cheaply until each card is at one, so a context that was briefly mid-alchemy at detection
        is collapsed on a later tick. Busy processes are never killed.
        """
        if not self._state.torch_build_cpu_only:
            return

        from dataclasses import replace

        target_lowered = False
        for index, card in list(self._card_runtimes.items()):
            if card.target_process_count != 1:
                self._card_runtimes[index] = replace(card, target_process_count=1)
                target_lowered = True
        if target_lowered:
            self.max_inference_processes = sum(card.target_process_count for card in self._card_runtimes.values())
            self._process_lifecycle.refresh_max_inference_processes()
            logger.info(
                "CPU-only torch build detected at runtime; collapsing inference processes to one per card "
                "(image generation disabled, alchemy continues).",
            )

        for index in sorted(self._card_runtimes):
            if self._process_map.num_loaded_inference_processes(device_index=index) > 1:
                self._process_lifecycle.scale_inference_processes(1, device_index=index)

    def _served_workloads(self, bridge_data: reGenBridgeData) -> list[str]:
        """The workloads actually served, as sorted ``WorkloadKind`` values, for the snapshot.

        Starts from the declarative :func:`enabled_workloads` (config + install sentinel) and additionally
        drops image generation when an inference child has reported a CPU-only torch build at runtime
        (``torch_build_cpu_only``). That keeps the dashboard's mode identity truthful for a CPU torch build
        whose ``bin/backend`` sentinel was never set: it reshapes to alchemist-only just as the install
        sentinel would, matching the image popper which is already gated on the same flag.
        """
        workloads = enabled_workloads(bridge_data)
        if self._state.torch_build_cpu_only:
            workloads = workloads - {WorkloadKind.IMAGE_GENERATION}
        return [workload.value for workload in workloads]

    def _build_worker_state_snapshot(self) -> WorkerStateSnapshot:
        """Assemble current worker state for the supervisor pipe (mirrors what StatusReporter prints)."""
        import horde_worker_regen

        bridge_data = self.bridge_data
        processes = [ProcessSnapshot.from_process_info(info) for info in self._process_map.values()]
        active_models = sorted(
            {info.loaded_horde_model_name for info in self._process_map.values() if info.loaded_horde_model_name},
        )

        run_metrics = self._run_metrics.snapshot(
            num_process_recoveries=self._process_lifecycle._num_process_recoveries,
            num_job_slowdowns=self._job_submitter.num_job_slowdowns,
            time_spent_no_jobs_available=self._job_popper.time_spent_no_jobs_available,
            disk_min_free_bytes=self._disk_monitor.min_free_bytes,
        )

        session_hours = max((time.time() - self.session_start_time) / 3600.0, 1e-6)
        kudos_session = self._state.kudos_generated_this_session
        kudos_per_hour = kudos_session / session_hours if kudos_session else None

        now = time.time()
        gpu_utilization_mean_percent = self._gpu_sampler.mean_percent(
            window_seconds=self._DUTY_CYCLE_SNAPSHOT_WINDOW_SECONDS,
        )
        gpu_utilization_busy_fraction = self._gpu_sampler.busy_fraction(
            window_seconds=self._DUTY_CYCLE_SNAPSHOT_WINDOW_SECONDS,
        )
        last_pop_time = self._state.last_job_pop_time
        seconds_since_last_pop = (now - last_pop_time) if last_pop_time else None
        api_messages: list[str] = []
        for api_message in self._job_popper.api_messages_received.values():
            if api_message.message_text:
                api_messages.append(api_message.message_text)

        recent_jobs = [
            RecentJobRecord.from_metrics_record(job, baseline=self._safe_model_baseline(job.model_name))
            for job in run_metrics.jobs[-RECENT_JOBS_IN_SNAPSHOT:]
        ]
        orchestration_intent = self._build_orchestration_intent()
        process_state_summary = " ".join(
            f"{process.process_type.lower()}#{process.process_id}={process.last_process_state}"
            for process in processes
        )
        maintenance_mode = (
            self._state.last_pop_maintenance_mode or self._state.supervisor_paused or self._state.self_throttle_paused
        )

        stats_sample = self._run_metrics.record_stats_sample(
            StatsSample(
                timestamp=now,
                jobs_submitted=self._job_tracker.total_num_completed_jobs,
                jobs_faulted=self._job_tracker.num_jobs_faulted,
                kudos_per_hour=kudos_per_hour,
                gpu_duty_percent=gpu_utilization_mean_percent,
                gpu_busy_fraction=gpu_utilization_busy_fraction,
                pending_megapixelsteps=self._job_tracker.get_pending_megapixelsteps(),
                jobs_pending_inference=len(self._job_tracker.jobs_pending_inference),
                jobs_in_progress=len(self._job_tracker.jobs_in_progress),
                jobs_pending_safety_check=len(self._job_tracker.jobs_pending_safety_check),
                jobs_being_safety_checked=len(self._job_tracker.jobs_being_safety_checked),
                jobs_pending_submit=len(self._job_tracker.jobs_pending_submit),
                time_spent_no_jobs_available=self._job_popper.time_spent_no_jobs_available,
                num_process_recoveries=self._process_lifecycle._num_process_recoveries,
                num_job_slowdowns=self._job_submitter.num_job_slowdowns,
                alchemy_forms_pending=self._alchemy_coordinator.num_forms_pending,
                alchemy_forms_in_flight=self._alchemy_coordinator.num_forms_in_flight,
                alchemy_forms_awaiting_submit=self._alchemy_coordinator.num_forms_awaiting_submit,
                alchemy_total_submitted=self._alchemy_coordinator.num_forms_submitted,
                alchemy_total_faulted=self._alchemy_coordinator.num_forms_faulted,
                process_state_summary=process_state_summary,
                orchestration_intent_summary=orchestration_intent.summary,
                orchestration_next_action=orchestration_intent.next_action,
                orchestration_why=orchestration_intent.why,
                orchestration_raw_gate=orchestration_intent.raw_gate,
                maintenance_mode=maintenance_mode,
                self_throttle_paused=self._state.self_throttle_paused,
                supervisor_paused=self._state.supervisor_paused,
                last_pop_maintenance_mode=self._state.last_pop_maintenance_mode,
                worker_details_maintenance=self._worker_details_maintenance,
                in_error_backoff=self._job_popper._pop_throttler.is_in_error_backoff,
                last_pop_no_jobs_available=self._state.last_pop_no_jobs_available,
                last_pop_skipped_reasons=dict(self._state.last_pop_skipped_reasons),
                churn_counts={kind: len(times) for kind, times in run_metrics.churn_event_times.items()},
            ),
        )
        latest_stats_sample = stats_sample or self._run_metrics.latest_stats_sample()

        config = WorkerConfigSummary(
            dreamer_name=bridge_data.dreamer_worker_name,
            alchemist_name=bridge_data.alchemist_name,
            worker_version=horde_worker_regen.__version__,
            horde_username=self.user_info.username if self.user_info is not None else None,
            num_models=len(bridge_data.image_models_to_load),
            custom_models=bool(bridge_data.custom_models),
            max_power=bridge_data.max_power,
            max_threads=self.max_concurrent_inference_processes,
            queue_size=bridge_data.queue_size,
            max_batch=bridge_data.max_batch,
            safety_on_gpu=bridge_data.safety_on_gpu,
            allow_img2img=bridge_data.allow_img2img,
            allow_lora=bridge_data.allow_lora,
            effective_allow_lora=(
                bridge_data.allow_lora
                and not self._model_availability.background_download_active
                and not self._state.lora_disk_exhausted
            ),
            allow_controlnet=bridge_data.allow_controlnet,
            allow_sdxl_controlnet=bridge_data.allow_sdxl_controlnet,
            allow_post_processing=bridge_data.allow_post_processing,
            high_performance_mode=bridge_data.high_performance_mode,
            moderate_performance_mode=bridge_data.moderate_performance_mode,
            extra_slow_worker=bridge_data.extra_slow_worker,
            alchemist=bridge_data.alchemist,
            alchemy_concurrent=bridge_data.alchemy_allow_concurrent,
            alchemy_max_concurrency=bridge_data.alchemy_max_concurrency,
            alchemy_vram_headroom_mb=bridge_data.alchemy_vram_headroom_mb,
            alchemy_caption_enabled=bridge_data.alchemy_caption_enabled,
            alchemy_forms=list(bridge_data.forms) if bridge_data.forms else list(DEFAULT_ALCHEMY_FORMS),
        )

        return WorkerStateSnapshot(
            session_start_time=self.session_start_time,
            shutting_down=self._state.shutting_down,
            maintenance_mode=maintenance_mode,
            self_throttle_paused=self._state.self_throttle_paused,
            supervisor_paused=self._state.supervisor_paused,
            last_pop_maintenance_mode=self._state.last_pop_maintenance_mode,
            worker_details_maintenance=self._worker_details_maintenance,
            worker_details_paused=self._worker_details_paused,
            too_many_consecutive_failed_jobs=self._state.too_many_consecutive_failed_jobs,
            gpu_torch_incompatible=self._state.gpu_torch_incompatible,
            gpu_torch_incompatible_reason=(self._state.gpu_torch_incompatible_reason or None),
            torch_build_cpu_only=self._state.torch_build_cpu_only,
            torch_build_cpu_only_reason=(self._state.torch_build_cpu_only_reason or None),
            worker_registered=self.user_info is not None,
            user_info_failed=self._user_info_failed,
            user_info_failed_reason=self._user_info_failed_reason,
            in_error_backoff=self._job_popper._pop_throttler.is_in_error_backoff,
            consecutive_failed_jobs=self._state.consecutive_failed_jobs,
            seconds_since_last_pop=seconds_since_last_pop,
            last_pop_no_jobs_available=self._state.last_pop_no_jobs_available,
            last_pop_skipped_reasons=dict(self._state.last_pop_skipped_reasons),
            api_messages=api_messages,
            config=config,
            processes=processes,
            num_jobs_popped=self.num_jobs_total,
            num_jobs_submitted=self._job_tracker.total_num_completed_jobs,
            num_jobs_faulted=self._job_tracker.num_jobs_faulted,
            num_job_slowdowns=self._job_submitter.num_job_slowdowns,
            num_process_recoveries=self._process_lifecycle._num_process_recoveries,
            pending_megapixelsteps=self._job_tracker.get_pending_megapixelsteps(),
            jobs_pending_inference=len(self._job_tracker.jobs_pending_inference),
            jobs_in_progress=len(self._job_tracker.jobs_in_progress),
            jobs_pending_safety_check=len(self._job_tracker.jobs_pending_safety_check),
            jobs_being_safety_checked=len(self._job_tracker.jobs_being_safety_checked),
            jobs_pending_submit=len(self._job_tracker.jobs_pending_submit),
            time_spent_no_jobs_available=self._job_popper.time_spent_no_jobs_available,
            kudos_per_hour=kudos_per_hour,
            kudos_this_session=kudos_session,
            active_models=active_models,
            gpu_utilization_mean_percent=gpu_utilization_mean_percent,
            gpu_utilization_busy_fraction=gpu_utilization_busy_fraction,
            gpu_utilization_samples=self._gpu_sampler.sample_count,
            vram_high_water_mb_per_process=run_metrics.vram_used_high_water_mb_per_process,
            ram_high_water_mb_per_process=run_metrics.ram_used_high_water_mb_per_process,
            disk_free_bytes=dict(self._disk_monitor.current_free_bytes),
            recent_jobs=recent_jobs,
            latest_stats_sample=latest_stats_sample,
            stats_model_rollups=self._run_metrics.model_rollups(),
            stats_baseline_rollups=self._run_metrics.baseline_rollups(),
            stats_form_rollups=self._run_metrics.form_rollups(),
            stats_export=self._run_metrics.stats_export_state(),
            stats_history_backfill=self._run_metrics.stats_history_backfill(),
            downloads=self._model_availability.status,
            download_plan=self._download_coordinator.get_download_plan_summary(),
            feature_readiness=self._build_feature_readiness_summary(bridge_data),
            lora_pops_blocked_by_downloads=(
                bridge_data.allow_lora and self._model_availability.background_download_active
            ),
            lora_pops_blocked_by_disk=(bridge_data.allow_lora and self._state.lora_disk_exhausted),
            alchemy_forms_pending=self._alchemy_coordinator.num_forms_pending,
            alchemy_forms_in_flight=self._alchemy_coordinator.num_forms_in_flight,
            alchemy_forms_awaiting_submit=self._alchemy_coordinator.num_forms_awaiting_submit,
            alchemy_total_submitted=self._alchemy_coordinator.num_forms_submitted,
            alchemy_total_faulted=self._alchemy_coordinator.num_forms_faulted,
            enabled_workloads=sorted(self._served_workloads(bridge_data)),
            pending_jobs=self._build_pending_jobs_list(),
            orchestration_intent=orchestration_intent,
            work_ledger=self._build_work_ledger(recent_jobs),
            whole_card_residency=self._whole_card_residency_status(),
            pop_governors=self._pop_governors_status(),
            scheduling_governance=self._scheduling_governance_status(),
            per_card=self._build_card_snapshots(),
            system_memory=SystemMemorySnapshot.from_summary(self._sample_system_memory()),
        )

    def _build_feature_readiness_summary(self, bridge_data: reGenBridgeData) -> FeatureReadinessSummary:
        """Build the per-feature readiness shown in the TUI, matching the pop gate's offer decision.

        The gated rows fuse the (post-coercion) opt-in flag, the live dependency probe, and the on-disk
        presence reported by the download process; the informational rows surface LoRA and safety, which
        keep their own gating. Built from the same inputs the pop gate uses, so the table never disagrees
        with what the worker actually advertises.
        """
        from horde_worker_regen.capabilities import (
            controlnet_available,
            controlnet_install_hint,
            post_processing_install_hint,
            strip_background_available,
        )

        availability = self._model_availability
        controlnet_deps = controlnet_available()
        post_processing_deps = strip_background_available()
        controlnet_hint = controlnet_install_hint() if not controlnet_deps else ""
        post_processing_hint = post_processing_install_hint() if not post_processing_deps else ""

        gated = build_feature_readiness(
            {
                GatedFeature.CONTROLNET: FeatureInputs(
                    enabled=bridge_data.allow_controlnet,
                    present=availability.controlnet_present,
                    deps_available=controlnet_deps,
                    deps_hint=controlnet_hint,
                    failed=availability.controlnet_failed,
                    failed_detail=CONTROLNET_ANNOTATOR_FAILED_DETAIL,
                ),
                GatedFeature.SDXL_CONTROLNET: FeatureInputs(
                    enabled=bridge_data.allow_sdxl_controlnet,
                    present=availability.sdxl_controlnet_present,
                    deps_available=controlnet_deps,
                    deps_hint=controlnet_hint,
                    failed=availability.controlnet_failed,
                    failed_detail=CONTROLNET_ANNOTATOR_FAILED_DETAIL,
                ),
                GatedFeature.POST_PROCESSING: FeatureInputs(
                    enabled=bridge_data.allow_post_processing,
                    present=availability.post_processing_present,
                    deps_available=post_processing_deps,
                    deps_hint=post_processing_hint,
                ),
            },
        )

        informational = [
            self._lora_info_row(bridge_data),
            self._safety_info_row(),
        ]
        return FeatureReadinessSummary(gated=list(gated), informational=informational)

    def _lora_info_row(self, bridge_data: reGenBridgeData) -> FeatureInfoRow:
        """Read-only LoRA readiness: enabled, or paused by the download/disk guards (its own gating)."""
        if not bridge_data.allow_lora:
            return FeatureInfoRow(label="LoRA", status="not enabled in config", ok=False)
        if self._state.lora_disk_exhausted:
            return FeatureInfoRow(label="LoRA", status="paused: low disk on the LoRA volume", ok=False)
        if self._model_availability.background_download_active:
            return FeatureInfoRow(label="LoRA", status="paused while models download", ok=False)
        return FeatureInfoRow(label="LoRA", status="enabled (fetched per job)", ok=True)

    def _safety_info_row(self) -> FeatureInfoRow:
        """Read-only safety-model readiness: present (image jobs can run) or still being fetched."""
        if self._model_availability.safety_present:
            return FeatureInfoRow(label="Safety models", status="present", ok=True)
        if self._model_availability.safety_attempted:
            return FeatureInfoRow(label="Safety models", status="unavailable (see logs)", ok=False)
        return FeatureInfoRow(label="Safety models", status="verifying / downloading", ok=False)

    def build_run_record(self) -> WorkerRunRecord:
        """Return a durable summary of this worker session for app-state persistence.

        Read after the main loop ends (the counters remain valid on the manager). ``clean_exit`` is
        False when the session tripped the consecutive-failure circuit breaker, which both flags a
        bad run and disqualifies the active config from being recorded as known-good.
        """
        import horde_worker_regen

        ended_at = time.time()
        return WorkerRunRecord(
            started_at=self.session_start_time,
            ended_at=ended_at,
            duration_seconds=max(0.0, ended_at - self.session_start_time),
            worker_version=horde_worker_regen.__version__,
            jobs_submitted=self._job_tracker.total_num_completed_jobs,
            jobs_faulted=self._job_tracker.num_jobs_faulted,
            kudos_this_session=self._state.kudos_generated_this_session,
            clean_exit=not self._state.too_many_consecutive_failed_jobs,
        )

    def _apply_reloaded_bridge_data(self, bridge_data: reGenBridgeData) -> None:
        """Swap in freshly-loaded bridge data and re-derive dependent state."""
        previous_effective = self._runtime_config.effective_max_threads
        previously_configured = set(self.bridge_data.image_models_to_load)
        previous_download_flags = self._download_coordinator.download_process_flags()

        self.bridge_data = bridge_data
        coerce_bridge_data_to_capabilities(self.bridge_data)
        new_effective = self._runtime_config.effective_max_threads
        if new_effective != previous_effective:
            logger.info(
                f"Concurrent-inference cap changed {previous_effective} -> {new_effective} "
                f"from {BRIDGE_CONFIG_FILENAME}.",
            )
        if self.bridge_data.max_threads > self._max_concurrent_inference_processes:
            logger.warning(
                f"max_threads={self.bridge_data.max_threads} exceeds this session's ceiling of "
                f"{self._max_concurrent_inference_processes} (capped to {new_effective}); "
                "restart the worker to raise the ceiling.",
            )
        logger.debug(f"Models to load: {self.bridge_data.image_models_to_load}")
        logger.debug(f"Custom models: {self.bridge_data.custom_models}")

        self._process_lifecycle.set_download_controls(
            paused=self.bridge_data.downloads_paused,
            rate_limit_kbps=self.bridge_data.download_rate_limit_kbps or 0,
            max_parallel_downloads=self.bridge_data.download_max_parallel_downloads,
            per_host_concurrency=self.bridge_data.download_per_host_concurrency,
            connections_per_file=self.bridge_data.download_connections_per_file,
        )
        self._download_coordinator.reconcile_downloads(
            run_aux_if_incomplete=False,
            previously_configured=previously_configured,
        )
        self._download_coordinator.forward_download_gating_if_changed(previous_download_flags)

    def _handle_exception(self, task: asyncio.Task[None]) -> None:
        """Supervise a finished main-loop task; shut down gracefully if one ends unexpectedly.

        Each main-loop coroutine is meant to run for the worker's whole life. If one finishes while
        the worker is not already shutting down, whether by raising or by returning early, the
        worker would otherwise limp on with a dead loop (e.g. jobs popped but never submitted, which
        orphans them). Instead, initiate a graceful shutdown so in-flight work drains and any
        supervising frontend relaunches us.

        This runs on the event-loop thread, so it must not block or ``sys.exit``: setting the
        shutdown flag lets the existing drain-and-exit path run, bounded by the timed-shutdown
        backstop.
        """
        if task.cancelled():
            return

        ex = task.exception()

        if self._state.shutting_down or self._state.shut_down:
            if ex is not None:
                logger.debug(f"main loop task ended during shutdown: {ex}")
            return

        if ex is not None:
            # Format the traceback from the exception object directly. This is a done-callback, not an
            # ``except`` block, so there is no active exception for ``logger.exception()`` to pick up;
            # calling it here logged only the (often empty) message and silently dropped the traceback,
            # which made a crashing main-loop task nearly impossible to diagnose.
            import traceback

            tb_text = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            logger.error(f"main loop task ended unexpectedly: {ex!r}\n{tb_text}")
        else:
            logger.error("A main loop task returned unexpectedly while the worker was running; shutting down.")

        self._shutdown()
        self._start_timed_shutdown()

    async def _main_loop(self) -> None:
        aiohttp_session = ClientSession(requote_redirect_url=False)

        from horde_worker_regen.telemetry import instrument_aiohttp_client

        instrument_aiohttp_client()

        horde_session = AIHordeAPIAsyncClientSession(
            aiohttp_session=aiohttp_session,
            apikey=self.bridge_data.api_key,
        )

        self._api_sessions.set_aiohttp_session(aiohttp_session)
        self._api_sessions.set_horde_client_session(horde_session)

        async with aiohttp_session, horde_session:  # pyrefly: ignore
            # The per-workload flows (image generation, alchemy) are launched uniformly from the flow
            # registry; each flow owns its own pop/dispatch/submit loops. The image flow internally
            # supervises the job popper and submitter, carrying the same per-loop shutdown supervision
            # they had as top-level tasks (see ImageGenerationCoordinator.run).
            coroutines = [
                self._process_control_loop(),
                self._api_get_user_info_loop(),
                self._periodic_update_check_loop(),
                self._periodic_server_capabilities_loop(),
                *(flow.run() for flow in self._flows.values()),
            ]
            if not self.bridge_data._loaded_from_env_vars:
                coroutines.append(self._bridge_data_reloader.bridge_data_loop())

            tasks = [asyncio.create_task(coro) for coro in coroutines]
            for task in tasks:
                task.add_done_callback(self._handle_exception)

            # return_exceptions=True so one failing loop does not cancel its siblings mid-flight:
            # _handle_exception has already initiated a graceful shutdown, and we want the other
            # loops (notably the submitter) to keep draining in-flight work until done.
            results = await asyncio.gather(*tasks, return_exceptions=True)

            if not self._state.shut_down:
                self._shutdown()
            for result in results:
                if isinstance(result, BaseException) and not isinstance(result, CancelledError):
                    logger.error(f"main loop task raised during shutdown: {result}")

    def start(self) -> None:
        """Start the process manager."""
        import atexit
        import signal

        # Backstop the clean-shutdown path: if the interpreter exits without ending children (a crash
        # that still unwinds to atexit, or a stray exit), kill any child we still own by OS pid so it
        # cannot linger holding the GPU. Identity is re-verified per pid, so a reused pid is never hit.
        atexit.register(self._kill_owned_children_on_exit)

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        asyncio.run(self._main_loop())

    def _kill_owned_children_on_exit(self) -> None:
        """Best-effort kill of any child still owned at interpreter exit (the atexit backstop)."""
        killed = self._process_lifecycle.kill_owned_children()
        if killed:
            logger.warning(f"atexit: killed {len(killed)} still-running owned child process(es): {killed}")

    def signal_handler(self, sig: int, frame: object) -> None:
        """Handle SIGINT and SIGTERM."""
        self._shutdown_manager.signal_handler(sig, frame)

        global _caught_signal
        _caught_signal = True

    def _start_timed_shutdown(self) -> None:
        self._shutdown_manager.start_timed_shutdown()

    def _shutdown(self) -> None:
        # Flush the latest self-calibration before exit so the next run starts warm.
        self._performance_model.save()
        self._shutdown_manager.shutdown()

    def _abort(self) -> None:
        """Exit as soon as possible, aborting all processes and jobs immediately."""
        self._shutdown_manager.abort()
