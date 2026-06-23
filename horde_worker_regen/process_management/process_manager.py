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
from horde_model_reference import SourceSelector
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE, MODEL_REFERENCE_CATEGORY
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
from horde_sdk.ai_horde_api.fields import GenerationID
from loguru import logger
from pydantic import ValidationError

from horde_worker_regen.app_state import AppStateStore, WorkerRunRecord, default_app_state_dir
from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader
from horde_worker_regen.capabilities import coerce_bridge_data_to_capabilities
from horde_worker_regen.consts import (
    BRIDGE_CONFIG_FILENAME,
    VRAM_HEAVY_MODELS,
)
from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management._canned_scenarios import CannedAlchemySource, CannedJobSource
from horde_worker_regen.process_management.action_ledger import ActionLedger, LedgerEventType
from horde_worker_regen.process_management.alchemy_popper import DEFAULT_ALCHEMY_FORMS, AlchemyCoordinator
from horde_worker_regen.process_management.api_sessions import ApiSessions
from horde_worker_regen.process_management.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.duty_cycle import DutyCycleSummary, summarize_duty_cycle
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.job_popper import JobPopper
from horde_worker_regen.process_management.job_submitter import JobSubmitter
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.lora_disk_guard import (
    free_mb,
    is_lora_disk_exhausted,
    read_evictable_adhoc_mb,
)
from horde_worker_regen.process_management.lru_cache import LRUCache
from horde_worker_regen.process_management.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.messages import (
    AlchemyFormSpec,
    HordeDownloadAvailabilityMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.model_availability import ModelAvailability
from horde_worker_regen.process_management.model_metadata import ModelMetadata
from horde_worker_regen.process_management.owned_process_registry import OwnedProcessRegistry
from horde_worker_regen.process_management.performance_model import (
    PERF_MODEL_FILENAME,
    PerformanceModel,
    load_seed_its_by_signature,
)
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.process_temperature import classify_process_temperature
from horde_worker_regen.process_management.recovery_supervisor import RecoveryAction, RecoverySupervisor
from horde_worker_regen.process_management.resource_budget import CommittedReserveLedger
from horde_worker_regen.process_management.run_metrics import RunMetricsSnapshot, WorkerRunMetrics
from horde_worker_regen.process_management.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.safety_orchestrator import SafetyOrchestrator
from horde_worker_regen.process_management.shutdown_manager import ShutdownManager
from horde_worker_regen.process_management.supervisor_channel import (
    PENDING_JOBS_IN_SNAPSHOT,
    RECENT_JOBS_IN_SNAPSHOT,
    DownloadPlanSummary,
    JobFeatureSummary,
    JobQueueEntry,
    ProcessSnapshot,
    RecentJobRecord,
    SupervisorChannel,
    SupervisorCommand,
    SupervisorControlMessage,
    SystemMemorySnapshot,
    WholeCardResidencyStatus,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from horde_worker_regen.process_management.worker_identity import lookup_worker_by_name
from horde_worker_regen.process_management.worker_state import WorkerState
from horde_worker_regen.reporting.kudos_logger import KudosLogger
from horde_worker_regen.reporting.maintenance_messenger import MaintenanceModeMessenger
from horde_worker_regen.reporting.status_reporter import StatusReporter
from horde_worker_regen.utils.disk_monitor import DiskSpaceMonitor
from horde_worker_regen.utils.gpu_monitor import GpuUtilizationSampler
from horde_worker_regen.utils.kudos_calculator import KudosCalculator
from horde_worker_regen.utils.kudos_utils import generate_kudos_info_string as _generate_kudos_info_string

if TYPE_CHECKING:
    from horde_worker_regen.process_management.job_models import HordeJobInfo
    from horde_worker_regen.process_management.job_tracker import TrackedJob
    from horde_worker_regen.process_management.messages import HordeJobMetricsMessage
    from horde_worker_regen.process_management.system_memory import SystemMemorySummary


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
        directly, for two reasons. It stays backend-agnostic (every ComfyUI-supported backend -- CUDA/ROCm,
        Intel XPU, Apple MPS, DirectML, CPU -- populates the device map; a bare ``torch.cuda.device_count()``
        loop would yield no devices on non-CUDA backends). And it keeps this process torch-free: detect()
        runs in the long-lived orchestrator, enumerating accelerators loads torch (~500MB), so
        :func:`probe_accelerators` pays that cost in a short-lived subprocess that frees it on exit.
        """
        import psutil

        from horde_worker_regen.utils.accelerator_probe import probe_accelerators

        total_ram = psutil.virtual_memory().total

        accelerators = probe_accelerators()
        device_map = TorchDeviceMap(root={})
        for accelerator in accelerators:
            device_map.root[accelerator.index] = TorchDeviceInfo(
                device_name=accelerator.name,
                device_index=accelerator.index,
                total_memory=accelerator.total_vram_mb * 1024 * 1024,
            )

        per_process_overhead_mb = max((a.runtime_overhead_mb for a in accelerators), default=0)
        marginal_process_overhead_mb = max((a.marginal_overhead_mb for a in accelerators), default=0)

        return cls(
            total_ram_bytes=total_ram,
            device_map=device_map,
            per_process_overhead_mb=per_process_overhead_mb,
            marginal_process_overhead_mb=marginal_process_overhead_mb,
        )


def _resolve_inference_concurrency(
    *,
    gpu_sampling_lease_enabled: bool,
    configured_lease_slots: int,
    max_concurrent_inference_processes: int,
    max_inference_processes: int,
) -> tuple[int, int]:
    """Resolve ``(inference_semaphore_size, gpu_sampling_lease_slots)`` for the IPC primitives.

    Without the lease the whole-job inference semaphore is the only GPU gate, so it stays at the
    concurrent-sampling count. With the lease the *lease* becomes the denoise gate and the
    inference semaphore opens up to every inference process, letting spare processes stage their
    next pipeline (model load, prompt encode) ahead instead of blocking at job start.

    The lease slot count is independent of ``max_threads``: it bounds concurrent denoise loops
    only, clamped to ``[1, max_inference_processes]`` (at least one denoise may always run, and
    more slots than processes can never be used).
    """
    lease_slots = min(max(1, configured_lease_slots), max(1, max_inference_processes))
    if gpu_sampling_lease_enabled:
        return max_inference_processes, lease_slots
    return max_concurrent_inference_processes, lease_slots


@dataclasses.dataclass
class MultiprocessingPrimitives:
    """Multiprocessing primitives created for IPC."""

    process_message_queue: ProcessQueue
    inference_semaphore: Semaphore
    disk_lock: Lock_MultiProcessing
    aux_model_lock: Lock_MultiProcessing
    vae_decode_semaphore: Semaphore
    gpu_sampling_lease: Semaphore
    """Serializes the GPU denoising loop across inference processes so they pipeline (one
    samples while others stage their next pipeline) rather than idling the GPU in lockstep.
    Sized to the number of GPU sampling slots (1 for a single GPU)."""
    download_bandwidth_semaphore: Semaphore
    """Held by the background download process while it is actively downloading, so the parent can
    coordinate pop policy around WAN-bandwidth contention."""

    @classmethod
    def create(
        cls,
        ctx: BaseContext,
        max_concurrent_inference: int,
        vae_decode_semaphore_max: int,
        gpu_sampling_lease_slots: int = 1,
    ) -> MultiprocessingPrimitives:
        """Create real multiprocessing primitives from a context.

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
            inference_semaphore=BoundedSemaphore_MultiProcessing(max_concurrent_inference, ctx=ctx),
            disk_lock=Lock_MultiProcessing(ctx=ctx),
            aux_model_lock=Lock_MultiProcessing(ctx=ctx),
            vae_decode_semaphore=BoundedSemaphore_MultiProcessing(vae_decode_semaphore_max, ctx=ctx),
            gpu_sampling_lease=BoundedSemaphore_MultiProcessing(max(1, gpu_sampling_lease_slots), ctx=ctx),
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

    _inference_semaphore: Semaphore
    """A semaphore that limits the number of inference processes that can run at once."""

    _vae_decode_semaphore: Semaphore

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
        self._enable_background_downloads = enable_background_downloads
        self._inference_processes_started = False
        self._safety_processes_started = False
        self._initial_download_requested = False
        self._download_wait_started = 0.0
        self._download_plan_summary: DownloadPlanSummary | None = None
        self._download_plan_refreshed_at = 0.0
        """Monotonic time the disk plan was last (re)computed; it refreshes on a throttle so the presence
        counts track downloads completing, all from the single horde_model_reference presence authority."""
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

        self._process_map = ProcessMap({})
        self._horde_model_map = HordeModelMap(root={})

        self.max_safety_processes = max_safety_processes

        # This attribute is the provisioned concurrency *ceiling* (semaphore size). The live
        # effective cap is read via the max_concurrent_inference_processes property below.
        self._max_concurrent_inference_processes = ceiling

        self.max_inference_processes = self.bridge_data.queue_size + ceiling

        self._lru = LRUCache(self.max_inference_processes)

        self._amd_gpu = amd_gpu
        self._directml = directml

        if len(self.bridge_data.image_models_to_load) == 1 and self.max_concurrent_inference_processes == 1:
            self.max_inference_processes = 1

        self._job_tracker = JobTracker()

        self.target_vram_overhead_bytes_map = target_vram_overhead_bytes_map  # TODO

        if system_resources is None:
            system_resources = SystemResources.detect()

        self.total_ram_bytes = system_resources.total_ram_bytes
        self._device_map = system_resources.device_map

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

        if self.target_ram_overhead_bytes > self.total_ram_bytes:
            raise ValueError(
                f"target_ram_overhead_bytes ({self.target_ram_overhead_bytes}) is greater than "
                f"total_ram_bytes ({self.total_ram_bytes})",
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

        if mp_primitives is None:
            vae_decode_semaphore_max = 1

            # The lease is an independent denoise gate (see _resolve_inference_concurrency): with
            # it enabled the whole-job inference semaphore opens to every process so spare
            # processes can stage their next pipeline (model load, prompt encode) ahead while
            # others sample; the lease itself bounds concurrent denoise loops. Without it the
            # inference semaphore stays the sole sampling gate at the concurrent-sampling count.
            inference_semaphore_size, gpu_sampling_lease_slots = _resolve_inference_concurrency(
                gpu_sampling_lease_enabled=self.bridge_data.gpu_sampling_lease_enabled,
                configured_lease_slots=self.bridge_data.gpu_sampling_lease_slots,
                max_concurrent_inference_processes=self._max_concurrent_inference_processes,
                max_inference_processes=self.max_inference_processes,
            )

            mp_primitives = MultiprocessingPrimitives.create(
                ctx=ctx,
                max_concurrent_inference=inference_semaphore_size,
                vae_decode_semaphore_max=vae_decode_semaphore_max,
                gpu_sampling_lease_slots=gpu_sampling_lease_slots,
            )

        self._process_message_queue = mp_primitives.process_message_queue
        self._inference_semaphore = mp_primitives.inference_semaphore
        self._disk_lock = mp_primitives.disk_lock
        self._aux_model_lock = mp_primitives.aux_model_lock
        self._vae_decode_semaphore = mp_primitives.vae_decode_semaphore
        self._gpu_sampling_lease = mp_primitives.gpu_sampling_lease
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
            inference_semaphore=self._inference_semaphore,
            disk_lock=self._disk_lock,
            aux_model_lock=self._aux_model_lock,
            vae_decode_semaphore=self._vae_decode_semaphore,
            gpu_sampling_lease=self._gpu_sampling_lease,
            download_bandwidth_semaphore=self._download_bandwidth_semaphore,
            gpu_sampling_lease_enabled=self.bridge_data.gpu_sampling_lease_enabled,
            runtime_config=self._runtime_config,
            max_inference_processes=self.max_inference_processes,
            max_safety_processes=self.max_safety_processes,
            amd_gpu=self._amd_gpu,
            directml=self._directml,
            abort_callback=self._abort,
            state=self._state,
            entry_points=process_entry_points,
            owned_registry=self._owned_registry,
            action_ledger=self._action_ledger,
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

        self._run_metrics = WorkerRunMetrics()

        # Measure real GPU core uptime (the duty cycle) for the whole worker session, not just the
        # benchmark. A coarse 1s poll is plenty for the rolling-window trend and threshold logs and
        # is far cheaper than the benchmark's 0.1s sampler. It no-ops on CPU/fake/non-NVIDIA backends
        # (no telemetry -> no thread), so creating it here is always safe.
        self._gpu_sampler = GpuUtilizationSampler(interval_seconds=1.0)
        self._last_duty_cycle_log_time = 0.0
        self._last_no_jobs_seconds_at_duty_log = 0.0

        # Expected-time-to-complete model: seeds from the last benchmark's per-tier reference it/s and
        # self-calibrates from this worker's own jobs, so a "slow" job becomes measurable rather than
        # guessed. Disabled-to-memory under test (no app-state read, no benchmark import, no perf file);
        # the model is unit-tested directly.
        self._performance_model = self._build_performance_model()

        self._message_dispatcher.set_metrics_handlers(
            on_job_metrics=self._on_job_metrics,
            on_download_metrics=self._run_metrics.on_download_metrics,
        )
        self._message_dispatcher.set_download_availability_handler(self._on_download_availability)
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

        # Save-our-ship: escalates a worker that has stopped making progress on accepted work from an
        # in-place soft reset (rebuild pools, limp-by) to giving up cleanly on jobs it cannot serve, so
        # the worker keeps running rather than wedging. See RecoverySupervisor for the escalation policy.
        self._recovery_supervisor = RecoverySupervisor()
        self._limp_by_active = False

        # Orphaned-in-progress-job watchdog: a job left INFERENCE_IN_PROGRESS that no live inference
        # slot owns will never produce a result, so it wedges the head of the queue forever unless
        # something punts it. `_orphan_in_progress_since` records when each such job was first seen
        # un-owned (so a brief dispatch race is ridden out before punting); `_orphan_punt_history`
        # records recent punts so a recurring orphan storm escalates into the save-our-ship wedge path
        # (soft reset + limp-by) rather than silently punting jobs forever at full settings.
        self._orphan_in_progress_since: dict[GenerationID, float] = {}
        self._orphan_punt_history: list[float] = []

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
        )

        self._alchemy_coordinator = AlchemyCoordinator(
            state=self._state,
            process_map=self._process_map,
            job_tracker=self._job_tracker,
            shutdown_manager=self._shutdown_manager,
            runtime_config=self._runtime_config,
            api_sessions=self._api_sessions,
            reserve_ledger=self._reserve_ledger,
            canned_alchemy_source=canned_alchemy_source,
        )
        self._message_dispatcher.set_alchemy_result_handler(self._alchemy_coordinator.on_alchemy_result)

        if stable_diffusion_reference is not None:
            self.stable_diffusion_reference = stable_diffusion_reference
        else:
            if horde_model_reference_manager is None:
                raise ValueError(
                    "horde_model_reference_manager may only be None when stable_diffusion_reference is provided",
                )
            self.stable_diffusion_reference = None
            self._init_model_reference()

    def _init_model_reference(self) -> None:
        """Fetch the stable diffusion model reference, retrying on failure."""
        while self.stable_diffusion_reference is None:
            try:
                horde_model_reference_manager = ModelReferenceManager.get_instance()

                source = self._beta_aware_image_source(horde_model_reference_manager)
                # query() keeps the per-category record type through the source-bearing overload (the
                # image_generation overload returns an ImageGenerationQuery), so to_list() is typed as
                # list[ImageGenerationModelRecord] with no cast, unlike get_model_reference + source.
                records = horde_model_reference_manager.query(
                    MODEL_REFERENCE_CATEGORY.image_generation,
                    source=source,
                ).to_list()
                if not records:
                    raise RuntimeError(
                        "horde_model_reference returned no image_generation models; the reference may have "
                        "failed to download; cannot continue with an empty reference.",
                    )

                self.stable_diffusion_reference = {record.name: record for record in records}
            except Exception as e:
                logger.error(e)
                time.sleep(5)

    @staticmethod
    def _beta_aware_image_source(manager: ModelReferenceManager) -> SourceSelector:
        """Return the image-generation source selector, registering the beta (pending) provider if opted in.

        The orchestrator builds its own copy of the image reference (``stable_diffusion_reference``) and
        would otherwise stay canonical-only, while the inference subprocesses load the PRIMARY pending-queue
        (beta) models such as qwen whenever ``HORDELIB_BETA_MODEL_CATEGORIES`` is set. A beta model the
        children can load but the orchestrator has never heard of is never offered or scheduled, and a job
        for it would ``KeyError`` on the reference lookups. Mirroring the subprocess contract here keeps the
        two in agreement. Beta is best-effort: any failure degrades to the canonical source rather than
        blocking reference init. ``hordelib.beta_models`` is torch-free, so importing it here does not
        violate the torch-free orchestrator invariant.
        """
        try:
            from horde_model_reference import PENDING_SOURCE_ID
            from hordelib.beta_models import beta_source_for, build_pending_provider

            if manager.get_provider(PENDING_SOURCE_ID) is None:
                provider = build_pending_provider()
                if provider is not None:
                    manager.register_provider(provider, replace=True)

            return beta_source_for(MODEL_REFERENCE_CATEGORY.image_generation, manager)
        except Exception as e:  # noqa: BLE001 - beta is best-effort; never block reference init
            from horde_model_reference import HORDE_SOURCE_ID

            logger.warning(f"Could not enable beta models for the orchestrator reference: {type(e).__name__}: {e}")
            return HORDE_SOURCE_ID

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
        self._worker_details_maintenance = bool(worker_details.maintenance_mode)
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
            self._maybe_start_safety_processes()
            self._maybe_start_inference_processes()
            self._apply_self_maintenance_throttle()
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
                        self._process_lifecycle._replace_inference_process(process_info)
                    self._job_popper._replaced_due_to_maintenance = True
                MaintenanceModeMessenger.print_maintenance_mode_messages()

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
            self._reconcile_orphaned_in_progress_jobs()

            # Save-our-ship: above the per-slot recovery, escalate a worker that is wedged as a whole
            # (no live process for pending work) to a soft reset and finally to giving up cleanly.
            self._run_recovery_supervisor()

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
        self._publish_supervisor_snapshot()

        await self._sleep(self._loop_interval / 2)
        return True

    _DOWNLOAD_STARTUP_GRACE_SECONDS = 90.0
    """How long to wait for the download process's first availability report before starting
    inference anyway, so a missing/failed download process can never wedge startup forever."""

    _DOWNLOAD_PLAN_REFRESH_SECONDS = 2.0
    """How often the disk plan is recomputed so its presence counts track downloads completing live,
    while keeping the existence checks off the hot path."""

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

    def _on_download_availability(self, message: HordeDownloadAvailabilityMessage) -> None:
        """Record an on-disk availability snapshot from the download process.

        On the first authoritative (post-scan) report, request background downloads of any
        configured-but-missing models, then (re)check whether inference processes can be started now
        that disk presence is known. Early initializing/scanning reports update the live status only;
        acting on them would request downloads against an as-yet-incomplete present set.
        """
        self._model_availability.update(
            present=set(message.available_model_names),
            currently_downloading=message.currently_downloading,
            pending=tuple(message.pending_downloads),
            failed=tuple(message.failed_downloads),
            status=message.status,
            scan_complete=message.scan_complete,
            safety_present=message.safety_models_present,
            safety_attempted=message.safety_models_attempted,
        )

        # A completed download changed the on-disk reference (a new image model, or the LoRa/TI/aux
        # pass). Tell the inference subprocesses to reload from disk so newly downloaded auxiliary
        # models become visible cross-process without a restart. Subprocesses never download.
        if message.reference_changed:
            self._process_lifecycle.broadcast_reload_model_database()

        if message.scan_complete and not self._initial_download_requested:
            self._initial_download_requested = True
            plan = self._get_download_plan_summary()
            if plan is not None:
                StatusReporter.log_startup_download_plan(plan)
            # Only run the (heavier) auxiliary pass on a genuinely incomplete install; a worker that
            # already has all its image models almost certainly has its aux models too.
            self._request_downloads_for_configured_missing(run_aux_if_incomplete=True)

        self._maybe_start_safety_processes()
        self._maybe_start_inference_processes()

    def _request_downloads_for_configured_missing(
        self,
        *,
        run_aux_if_incomplete: bool,
        previously_configured: set[str] | None = None,
    ) -> None:
        """Background-download any configured image models not yet on disk, and prune ones config dropped.

        Shared by the initial scan-complete trigger and the config-reload path, so a config change that
        adds a model fetches it without restarting the worker. The download process dedups against what it
        already has or is in-flight on, so re-sending the full missing set on every reload is safe. The
        auxiliary pass (LoRa/ControlNet/post-processing/safety) is one-shot in the download process, so
        ``run_aux_if_incomplete`` only matters for the very first request.

        ``previously_configured`` is the image-model set before a reload; when a reload *removes* a model
        we send the now-authoritative configured set so the download process stops any queued/in-flight
        download of the dropped model. The configured set is sent only when there is work or a removal, so
        an unchanged reload stays a no-op (it never sends a redundant reconcile).
        """
        if not self._enable_background_downloads:
            return
        present = self._model_availability.present or set()
        configured = set(self.bridge_data.image_models_to_load)
        missing = sorted(configured - present)
        removed = (previously_configured or set()) - configured
        download_aux = run_aux_if_incomplete and len(missing) > 0
        if not missing and not removed and not download_aux:
            return
        if removed:
            logger.info(f"Config removed {len(removed)} image model(s); stopping their downloads: {sorted(removed)}")
        if missing:
            logger.info(
                f"Worker has {len(present)} of {len(configured)} configured models on disk; "
                f"background-downloading {len(missing)} missing: {missing}",
            )
        self._process_lifecycle.request_downloads(
            missing,
            download_aux=download_aux,
            desired_image_models=sorted(configured),
        )

    def _download_process_flags(self) -> tuple[object, ...]:
        """The bridge-data fields baked into the download process at construction (for change detection).

        These gate *which* models the download process fetches (aux categories, nsfw filtering, LoRa
        purging); unlike pause/rate/parallelism they are constructor arguments, so a change to any of them
        only takes effect by restarting the process. Order is irrelevant; only equality is compared.
        """
        return (
            self.bridge_data.nsfw,
            self.bridge_data.allow_lora,
            self.bridge_data.allow_controlnet,
            self.bridge_data.allow_sdxl_controlnet,
            self.bridge_data.allow_post_processing,
            self.bridge_data.purge_loras_on_download,
        )

    def _reload_download_process_if_flags_changed(self, previous_flags: tuple[object, ...]) -> None:
        """Restart the download process when a reload changed its construction-time download gating.

        Live controls (pause/rate/parallelism) are forwarded without a restart; the aux/nsfw/purge flags
        cannot be, so a change to them stops the current process and starts a fresh one with the new
        config, then lets the next scan-complete re-trigger the configured downloads (including a fresh
        aux pass). Inference and safety keep running throughout: only the (jobless) download process
        cycles, and the present-set is held across the brief gap so popping is unaffected.
        """
        if not self._enable_background_downloads:
            return
        if self._download_process_flags() == previous_flags:
            return
        logger.info("Download-affecting config changed on reload; restarting the background download process.")
        # A download-process restart failure must not abort the rest of the config reload (or wedge the
        # worker): the worker keeps serving whatever is present, and the next reload can retry the restart.
        try:
            self._process_lifecycle.restart_download_process()
            # The fresh process must be told the live controls (its constructor took the config defaults, but
            # a prior live TUI override is intentionally re-asserted from config on every reload anyway).
            self._process_lifecycle.set_download_controls(
                paused=self.bridge_data.downloads_paused,
                rate_limit_kbps=self.bridge_data.download_rate_limit_kbps or 0,
                max_parallel_downloads=self.bridge_data.download_max_parallel_downloads,
                per_host_concurrency=self.bridge_data.download_per_host_concurrency,
            )
            # Let the restarted process's first authoritative scan re-request configured-missing models plus
            # a fresh aux pass (one-shot guard reset), so newly-enabled aux categories actually download.
            self._initial_download_requested = False
        except Exception as e:  # noqa: BLE001 - a reload must never crash on a download-process restart
            logger.error(f"Failed to restart the download process on reload (continuing): {type(e).__name__}: {e}")

    def _enter_downloads_only_hold(self) -> None:
        """Enter the download-only posture: keep fetching models but hold inference/safety/popping.

        Lets the operator pre-fetch models without committing the GPU. The download process is ensured
        running (it is the thing that does the work and the availability oracle); inference and safety
        stay deferred via the hold gate, and the job popper stops popping. Idempotent.
        """
        if not self._enable_background_downloads:
            logger.warning("Download-only hold requested but background downloads are disabled; ignoring.")
            return
        if self._state.downloads_only_hold:
            return
        self._state.downloads_only_hold = True
        self._process_lifecycle.start_download_process()
        logger.info("Entered download-only mode: pre-fetching models; inference and job popping are held.")

    def _leave_downloads_only_hold(self) -> None:
        """Leave the download-only posture and bring the worker fully up (GO_LIVE).

        Clears the hold and re-checks the deferred starts immediately; inference/safety come up once a
        model is present (the normal availability gate), and popping resumes. In-flight downloads are
        untouched, and the present-set pop gate keeps the worker from advertising a still-downloading model.
        """
        if not self._state.downloads_only_hold:
            return
        self._state.downloads_only_hold = False
        logger.info("Leaving download-only mode (GO_LIVE): starting inference/safety and resuming job popping.")
        self._maybe_start_safety_processes()
        self._maybe_start_inference_processes()

    def _download_models_on_demand(self, model_names: list[str], *, include_aux: bool) -> None:
        """Fetch an operator-chosen set of models now, without changing config (drives the TUI picker).

        Additive only: the names are enqueued into the background download process alongside whatever the
        config already requested (no authoritative-set reconcile, so this never prunes configured
        downloads). ``include_aux`` also kicks the one-time aux/default pass.
        """
        if not self._enable_background_downloads:
            logger.warning("On-demand download requested but background downloads are disabled; ignoring.")
            return
        if not model_names and not include_aux:
            return
        self._process_lifecycle.start_download_process()
        logger.info(f"On-demand download of {len(model_names)} model(s) (aux={include_aux}): {sorted(model_names)}")
        self._process_lifecycle.request_downloads(list(model_names), download_aux=include_aux)

    def _maybe_start_safety_processes(self) -> None:
        """Start safety processes once the required safety models are on disk (background-download mode).

        The safety models (DeepDanbooru + CLIP, ~2.3GB) are fetched by the dedicated download process.
        Starting the safety process before they are present would make it download them synchronously in
        its constructor: minutes of work with no parent-visible state change, which reads as a hung
        worker (especially under the TUI, whose console is redirected). So we defer the launch until the
        download process reports them present.

        The fallback is deliberately narrow to avoid a duplicate, concurrent download: we only start the
        safety process early (to let it self-fetch) once the download process has *finished* its one-shot
        ensure without producing them (``safety_attempted`` but not ``safety_present`` -- e.g. the download
        failed), or when it never reported at all (a crashed import, bounded by the startup grace). While
        the ensure is still pending or in flight we simply wait, which is the whole point: the operator
        sees a real "downloading safety models" phase instead of a freeze. A transient post-scan idle
        report (ensure not yet attempted) must not trip the fallback, which is why this keys on
        ``safety_attempted`` rather than the download phase.
        """
        if self._safety_processes_started or not self._enable_background_downloads:
            return
        if self._state.downloads_only_hold:
            return

        availability = self._model_availability
        if availability.safety_present:
            logger.info("Required safety models are present on disk; starting safety processes")
            self._process_lifecycle.start_safety_processes()
            self._safety_processes_started = True
            return

        if availability.safety_attempted:
            logger.warning(
                "Download process finished without providing the safety models; starting the safety "
                "process to fetch them directly (it will surface any download error)",
            )
            self._process_lifecycle.start_safety_processes()
            self._safety_processes_started = True
            return

        # A download process that never reported at all (crashed/hung import) must not wedge startup.
        if not availability.is_known and (time.time() - self._download_wait_started) > (
            self._DOWNLOAD_STARTUP_GRACE_SECONDS
        ):
            logger.warning(
                "No model availability report after "
                f"{self._DOWNLOAD_STARTUP_GRACE_SECONDS:.0f}s; starting safety processes anyway",
            )
            self._process_lifecycle.start_safety_processes()
            self._safety_processes_started = True

    def _maybe_start_inference_processes(self) -> None:
        """Start inference processes once at least one model is present (background-download mode).

        In the default (non-background-download) mode inference processes are started up front, so
        this is a no-op. With background downloads enabled, starting before any model is on disk
        would crash the inference children ("no models available"); this defers them until disk
        presence is known, with a grace-period fallback so a silent download process cannot hang us.
        """
        if self._inference_processes_started or not self._enable_background_downloads:
            return
        if self._state.downloads_only_hold:
            return

        availability = self._model_availability
        if availability.scan_complete and len(availability.present or set()) > 0:
            logger.info("At least one model is present on disk; starting inference processes")
            self._process_lifecycle.start_inference_processes()
            self._inference_processes_started = True
            return

        # While the download process is still initializing/scanning, or reporting an empty disk, we
        # keep waiting: starting inference with no models would only crash and churn the children. The
        # grace fallback exists solely for a download process that never reports at all (crashed/hung
        # import); an actively-initializing or -scanning process counts as alive and resets the clock.
        if availability.is_known and not availability.scan_complete:
            self._download_wait_started = time.time()
            return

        if not availability.is_known and (time.time() - self._download_wait_started) > (
            self._DOWNLOAD_STARTUP_GRACE_SECONDS
        ):
            logger.warning(
                "No model availability report after "
                f"{self._DOWNLOAD_STARTUP_GRACE_SECONDS:.0f}s; starting inference processes anyway",
            )
            self._process_lifecycle.start_inference_processes()
            self._inference_processes_started = True

    def _is_inference_capacity_available(self) -> bool:
        """Whether any inference process is alive to serve pending inference work."""
        return any(
            process_info.process_type == HordeProcessType.INFERENCE and process_info.is_process_alive()
            for process_info in self._process_map.values()
        )

    def _is_safety_capacity_available(self) -> bool:
        """Whether any safety process is alive to serve pending safety checks."""
        return any(
            process_info.process_type == HordeProcessType.SAFETY and process_info.is_process_alive()
            for process_info in self._process_map.values()
        )

    def _is_safety_pool_ready(self) -> bool:
        """Whether at least one safety process is alive and able to accept a check (genuine recovery)."""
        return any(
            process_info.process_type == HordeProcessType.SAFETY and process_info.can_accept_job()
            for process_info in self._process_map.values()
        )

    def _is_inference_pool_unrecoverable(self) -> bool:
        """Whether the crash-loop breaker has quarantined every inference slot (definitive: cannot serve).

        Quarantine requires repeated rapid failures per slot, so this never fires during a normal slot
        replacement or a slow model load (neither of which quarantines), only when every slot has
        crash-looped out of the pool.
        """
        return len(self._process_lifecycle.quarantined_inference_slots) >= self.max_inference_processes

    def _is_safety_pool_unrecoverable(self) -> bool:
        """Whether the safety pool is crash-looping (rebuilt too often) and not currently ready to serve.

        The readiness gate keeps a pool that has recovered (a healthy safety process is up) from being
        treated as wedged while its recent rebuild count ages out of the window.
        """
        return self._process_lifecycle.safety_pool_failing and not self._is_safety_pool_ready()

    _ORPHAN_IN_PROGRESS_GRACE_SECONDS = 30.0
    """How long a job may sit INFERENCE_IN_PROGRESS with no owning live slot before it is punted.

    Long enough to ride out the brief window between a job being marked in-progress and its owning
    slot's reference being recorded (and any in-flight result still on the wire), short enough that a
    truly orphaned job drains in well under a minute instead of wedging the queue head indefinitely."""

    _ORPHAN_PUNT_WINDOW_SECONDS = 300.0
    """Sliding window over which repeated orphan punts are counted toward the wedge escalation."""

    _ORPHAN_PUNT_WEDGE_THRESHOLD = 3
    """Orphan punts within the window that escalate to the save-our-ship wedge path (soft reset/limp-by)."""

    def _inference_slot_owns_job(self, job_id: GenerationID) -> bool:
        """Whether some live inference slot is currently working on (references) the given job.

        ``last_job_referenced`` is not cleared when a job completes or its slot returns to idle, so a
        reference match alone is not ownership: an idle slot (one that ``can_accept_job``) carrying a
        stale reference will never produce a result for the job. Counting such a slot as the owner lets
        a job whose result was lost (e.g. dropped by the launch-identifier guard during a recovery
        storm) sit in progress forever, shielded from the orphaned-job watchdog, until it wedges the
        whole worker. A slot only owns the job while it is genuinely processing it, i.e. it is not
        available for new work.
        """
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if not process_info.is_process_alive():
                continue
            if process_info.can_accept_job():
                continue
            referenced = process_info.last_job_referenced
            if referenced is not None and referenced.id_ == job_id:
                return True
        return False

    def _reconcile_orphaned_in_progress_jobs(self) -> None:
        """Punt jobs stuck INFERENCE_IN_PROGRESS that no live inference slot owns.

        Per-slot recovery faults the job of the slot it replaces, but a mis-association, a lost result,
        or a requeue race can still leave a *different* job marked in-progress with no owning slot. No
        result will ever arrive for it, so it pins the head of the queue forever . This watchdog is the
        backstop: an in-progress job that no live slot has referenced for ``_ORPHAN_IN_PROGRESS_GRACE``
        seconds is faulted (retryable, so it requeues or, once attempts are exhausted, is reported
        faulted and drains). Recurring orphans feed the wedge escalation so the worker limps by.
        """
        now = time.time()
        in_progress = self._job_tracker.jobs_in_progress
        live_ids = {job.id_ for job in in_progress if job.id_ is not None and self._inference_slot_owns_job(job.id_)}

        # Forget jobs that are owned again or no longer in progress, so the grace clock only runs while
        # a job is continuously orphaned.
        current_ids = {job.id_ for job in in_progress if job.id_ is not None}
        for job_id in list(self._orphan_in_progress_since):
            if job_id not in current_ids or job_id in live_ids:
                del self._orphan_in_progress_since[job_id]

        for job in in_progress:
            job_id = job.id_
            if job_id is None or job_id in live_ids:
                continue
            first_seen = self._orphan_in_progress_since.setdefault(job_id, now)
            if (now - first_seen) < self._ORPHAN_IN_PROGRESS_GRACE_SECONDS:
                continue

            logger.error(
                f"Job {job_id} has been in progress with no live inference slot for "
                f"{now - first_seen:.0f}s; punting it so the queue can drain (orphaned-job watchdog).",
            )
            self._action_ledger.record(
                LedgerEventType.INFERENCE_FAULTED,
                job_id=str(job_id),
                reason="orphaned in-progress job (no owning live inference slot)",
                detail={"stuck_seconds": round(now - first_seen, 1)},
            )
            self._job_tracker.handle_job_fault_now(
                faulted_job=job,
                process_timeout=self.bridge_data.process_timeout,
                retryable=True,
            )
            del self._orphan_in_progress_since[job_id]
            self._orphan_punt_history.append(now)

    def _orphan_wedge_active(self) -> bool:
        """Whether orphaned-job punts have recurred often enough to count as a worker-level wedge.

        A single orphan is handled by punting it; a *storm* of them means something upstream keeps
        stranding jobs (a flaky GPU that hangs each inference, say), which the punt alone does not fix.
        Surfacing it as a wedge lets the recovery supervisor soft-reset the pools and limp by at
        reduced concurrency, then restore settings once the storm subsides.
        """
        now = time.time()
        self._orphan_punt_history = [
            t for t in self._orphan_punt_history if (now - t) <= self._ORPHAN_PUNT_WINDOW_SECONDS
        ]
        return len(self._orphan_punt_history) >= self._ORPHAN_PUNT_WEDGE_THRESHOLD

    def _assess_wedge(self) -> bool:
        """Whether the worker structurally cannot make progress (the SOS/save-our-ship trigger).

        Keyed on the crash-loop signals (every inference slot quarantined, or the safety pool
        crash-looping with no healthy process), a sustained *queue* deadlock (pending inference work with
        every process idle), plus a recurring orphaned-job storm, not on transient capacity gaps. A merely
        slow, busy, replacing, or model-loading worker trips none of these, so a healthy worker is never
        wedged. Note the *general* deadlock flag is intentionally not a wedge signal: it also fires for a
        job draining through the safety/submit tail during a queue lull (see
        ``DeadlockSnapshot.indicates_structural_wedge``).
        """
        if self._state.shutting_down:
            return False
        structural_queue_wedge = self._message_dispatcher.get_deadlock_snapshot().indicates_structural_wedge()
        if structural_queue_wedge and (
            self._inference_scheduler.whole_card_residency_grace_active()
            or self._inference_scheduler.heavy_head_load_grace_active()
        ):
            # The queue is deliberately held while a heavy head loads: either a whole-card residency
            # establishing (idle siblings stopping, the safety process cycling off-GPU, ~11GB of weights
            # loading) or a streams-even-alone head admitted best-effort off that path. Both are the worker
            # doing the right thing, not a wedge, so do not let it soft-reset the pools mid-load. Both graces
            # are bounded, so a load that genuinely never completes still trips the supervisor.
            structural_queue_wedge = False
        return (
            self._is_inference_pool_unrecoverable()
            or self._is_safety_pool_unrecoverable()
            or structural_queue_wedge
            or self._orphan_wedge_active()
        )

    def _run_recovery_supervisor(self) -> None:
        """Drive the save-our-ship escalation one tick and perform any action it returns."""
        if self._state.shutting_down:
            return
        action = self._recovery_supervisor.evaluate(is_wedged=self._assess_wedge())
        if action is RecoveryAction.SOFT_RESET:
            self._perform_soft_reset()
            self._limp_by_active = True
        elif action is RecoveryAction.GIVE_UP:
            self._give_up_on_wedged_jobs()
        elif self._limp_by_active and not self._recovery_supervisor.is_in_episode:
            # The episode recovered after a sustained clean streak: undo limp-by exactly once, restoring
            # the *configured* concurrency (not the ceiling, which would override a user max_threads or a
            # live TUI override). RuntimeConfig clamps to the ceiling.
            self._limp_by_active = False
            self._runtime_config.set_effective_max_threads(self.bridge_data.max_threads)
            logger.info("Save-our-ship: pools recovered; restored configured concurrency (limp-by cleared).")

    def _perform_soft_reset(self) -> None:
        """Rebuild the worker's process pools in place and drop one limp-by notch (reduced concurrency)."""
        level = self._recovery_supervisor.limp_by_level
        # Limp by one notch *down from the current* effective concurrency (clamped to >= 1 by
        # RuntimeConfig), never up from a lower configured value.
        applied = self._runtime_config.set_effective_max_threads(self._runtime_config.effective_max_threads - 1)
        logger.error(
            f"Save-our-ship soft reset #{level}: rebuilding process pools and limping by "
            f"(effective max_threads -> {applied}).",
        )
        self._action_ledger.record(
            LedgerEventType.SOFT_RESET,
            reason=f"save-our-ship soft reset #{level}",
            detail={"limp_by_level": level, "effective_max_threads": applied},
        )
        self._process_lifecycle.rebuild_inference_pool(reason=f"soft reset #{level}")
        self._process_lifecycle.rebuild_safety_pool(reason=f"soft reset #{level}")

    def _give_up_on_wedged_jobs(self) -> None:
        """Last resort: fault unservable jobs, and if no pool can recover, shut down cleanly.

        Soft resets did not restore a working pool. Any jobs that cannot be served are reported faulted
        so the horde reissues them rather than holding them forever. If the worker structurally cannot
        serve at all (inference pool unrecoverable, or safety pool failing), it shuts down cleanly: the
        sanctioned last resort, so a permanently-broken worker stops rather than spinning, instead of
        hanging. A worker whose pools later recover never reaches here (the episode closes first).
        """
        faulted = 0
        # Reissue stuck pending work when the pool cannot serve it, OR when the pool is healthy but the
        # scheduler is structurally wedged (a sustained queue deadlock: pending inference work with every
        # process idle and no progress). The latter is the "healthy pool, starved scheduler" wedge: the
        # capacity check alone would fault nothing and let the worker spin forever, so the structural
        # queue-deadlock signal must also reissue the head so the horde reassigns it and the queue unblocks.
        structural_queue_wedge = self._message_dispatcher.get_deadlock_snapshot().indicates_structural_wedge()
        if not self._is_inference_capacity_available() or structural_queue_wedge:
            for job in list(self._job_tracker.jobs_pending_inference):
                if job not in self._job_tracker.jobs_in_progress:
                    self._job_tracker.handle_job_fault_now(job, retryable=False)
                    faulted += 1
        if not self._is_safety_capacity_available():
            stuck_safety = list(self._job_tracker.jobs_pending_safety_check) + list(
                self._job_tracker.jobs_being_safety_checked,
            )
            for info in stuck_safety:
                self._job_tracker.handle_job_fault_now(info.sdk_api_job_info, retryable=False)
                faulted += 1
        if faulted > 0:
            logger.critical(
                f"Save-our-ship: gave up on {faulted} unservable job(s) and reported them faulted so the "
                "horde reissues them.",
            )

        structurally_broken = self._is_inference_pool_unrecoverable() or self._is_safety_pool_unrecoverable()
        self._action_ledger.record(
            LedgerEventType.RECOVERY_ABANDONED,
            reason="save-our-ship: soft resets could not restore a working pool",
            detail={"jobs_faulted": faulted, "structurally_broken": structurally_broken},
        )
        if structurally_broken and not self._state.shutting_down:
            logger.critical(
                "Save-our-ship: the worker cannot restore a working process pool after repeated soft "
                "resets; abandoning ship (the last resort) rather than spinning indefinitely.",
            )
            # Abort rather than graceful shutdown: a graceful drain is gated by `recently_recovered`
            # (which the soft resets just set) and would stall for the watchdog window, and there is
            # nothing to drain gracefully when the pools are dead. The .abort sentinel stops promptly.
            self._abort()

    async def _process_control_loop(self) -> None:
        self._download_wait_started = time.time()
        self._gpu_sampler.start()
        if self._enable_background_downloads:
            self._process_lifecycle.start_download_process()
            # Both the safety and inference processes are started lazily once their required models are on
            # disk; see _maybe_start_safety_processes / _maybe_start_inference_processes (called from the
            # availability handler and each tick). Starting the safety process up front would make it
            # download the ~2.3GB safety models synchronously (and invisibly) in its constructor.
        else:
            # Without a download process there is nothing to defer to: start everything up front and let
            # the safety process fetch its own models (the legacy behaviour for tests/harness/dry-run).
            self._process_lifecycle.start_safety_processes()
            self._safety_processes_started = True
            self._process_lifecycle.start_inference_processes()
            self._inference_processes_started = True

        while True:
            try:
                if self.stable_diffusion_reference is None:
                    return
                # Watch for an externally-created .abort file as a signal-less
                # abort trigger (e.g. for process managers that cannot send signals).
                if os.path.exists(".abort"):
                    logger.warning("Found .abort file — aborting immediately")
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
        if self._enable_background_downloads and not self._inference_processes_started:
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

        nvml_mean = self._gpu_sampler.mean_percent(window_seconds=window_seconds)
        nvml_busy = self._gpu_sampler.busy_fraction(window_seconds=window_seconds)

        metrics = self.get_run_metrics_snapshot()
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
            download_plan=self._get_download_plan_summary(),
        )

        self._last_status_message_time = reporter.last_status_message_time
        self._status_message_frequency = updated_frequency

    _supervisor_publish_min_interval = 0.0
    """A hard floor between snapshots regardless of change (0 = publish every tick when state changed)."""
    _supervisor_publish_floor_interval = 2.0
    """Maximum seconds between snapshots when nothing changes — a heartbeat so the TUI knows we're alive."""

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
                self._schedule_config_reload()
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
                self._enter_downloads_only_hold()
            case SupervisorCommand.GO_LIVE:
                self._leave_downloads_only_hold()
            case SupervisorCommand.DOWNLOAD_MODELS:
                self._download_models_on_demand(command.download_model_names, include_aux=command.download_include_aux)
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
            self._inference_processes_started = True
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
        sampling progress, and headline counters that the dashboards render — but deliberately omits
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

    def _get_download_plan_summary(self) -> DownloadPlanSummary | None:
        """Compute the config's disk-implications summary, refreshed on a short throttle.

        Existence-only and torch-free (see :mod:`model_download_plan`); the live download process stays
        authoritative about integrity. Presence comes from the single ``horde_model_reference`` on-disk
        authority, so re-running it as downloads complete is what lets ``num_present`` (and thus the TUI's
        live readiness) climb without ever disagreeing with the disk budget. The throttle keeps the
        existence checks off the hot path; the last result is held between refreshes (and when the
        reference is not yet loaded the snapshot simply omits the plan).
        """
        now = time.monotonic()
        fresh = (now - self._download_plan_refreshed_at) < self._DOWNLOAD_PLAN_REFRESH_SECONDS
        if self._download_plan_summary is not None and fresh:
            return self._download_plan_summary

        reference = self.stable_diffusion_reference
        if reference is None:
            return self._download_plan_summary

        from horde_worker_regen import model_download_plan

        plan = model_download_plan.compute_download_plan(
            list(self.bridge_data.image_models_to_load),
            reference,
            extra_model_directories=self.bridge_data.extra_model_directories,
        )
        self._download_plan_summary = DownloadPlanSummary(
            present_bytes=plan.present_bytes,
            to_download_bytes=plan.to_download_bytes,
            total_bytes=plan.total_bytes,
            free_disk_bytes=plan.free_disk_bytes,
            fits=plan.fits,
            shortfall_bytes=plan.shortfall_bytes,
            num_present=plan.num_present,
            num_to_download=plan.num_to_download,
            sizes_complete=plan.sizes_complete,
        )
        self._download_plan_refreshed_at = now
        return self._download_plan_summary

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

    def _build_pending_jobs_list(self) -> list[JobQueueEntry]:
        """Build a capped list of pending-inference jobs for the overview queue display."""
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

        from horde_worker_regen.process_management.system_memory import (
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

        last_pop_time = self._state.last_job_pop_time
        seconds_since_last_pop = (time.time() - last_pop_time) if last_pop_time else None
        api_messages: list[str] = []
        for api_message in self._job_popper.api_messages_received.values():
            if api_message.message_text:
                api_messages.append(api_message.message_text)

        config = WorkerConfigSummary(
            dreamer_name=bridge_data.dreamer_worker_name,
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
            maintenance_mode=(
                self._state.last_pop_maintenance_mode
                or self._state.supervisor_paused
                or self._state.self_throttle_paused
            ),
            self_throttle_paused=self._state.self_throttle_paused,
            worker_details_maintenance=self._worker_details_maintenance,
            worker_details_paused=self._worker_details_paused,
            too_many_consecutive_failed_jobs=self._state.too_many_consecutive_failed_jobs,
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
            gpu_utilization_mean_percent=(
                self._gpu_sampler.mean_percent(window_seconds=self._DUTY_CYCLE_SNAPSHOT_WINDOW_SECONDS)
            ),
            gpu_utilization_busy_fraction=(
                self._gpu_sampler.busy_fraction(window_seconds=self._DUTY_CYCLE_SNAPSHOT_WINDOW_SECONDS)
            ),
            gpu_utilization_samples=self._gpu_sampler.sample_count,
            vram_high_water_mb_per_process=run_metrics.vram_used_high_water_mb_per_process,
            ram_high_water_mb_per_process=run_metrics.ram_used_high_water_mb_per_process,
            disk_free_bytes=dict(self._disk_monitor.current_free_bytes),
            recent_jobs=[
                RecentJobRecord.from_metrics_record(job, baseline=self._safe_model_baseline(job.model_name))
                for job in run_metrics.jobs[-RECENT_JOBS_IN_SNAPSHOT:]
            ],
            downloads=self._model_availability.status,
            download_plan=self._get_download_plan_summary(),
            lora_pops_blocked_by_downloads=(
                bridge_data.allow_lora and self._model_availability.background_download_active
            ),
            lora_pops_blocked_by_disk=(bridge_data.allow_lora and self._state.lora_disk_exhausted),
            alchemy_forms_pending=self._alchemy_coordinator.num_forms_pending,
            alchemy_forms_in_flight=self._alchemy_coordinator.num_forms_in_flight,
            alchemy_forms_awaiting_submit=self._alchemy_coordinator.num_forms_awaiting_submit,
            alchemy_total_submitted=self._alchemy_coordinator.num_forms_submitted,
            alchemy_total_faulted=self._alchemy_coordinator.num_forms_faulted,
            pending_jobs=self._build_pending_jobs_list(),
            whole_card_residency=self._whole_card_residency_status(),
            system_memory=SystemMemorySnapshot.from_summary(self._sample_system_memory()),
        )

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

    _bridge_data_loop_interval = 1.0
    """The interval between bridge data loop iterations."""
    _last_bridge_data_reload_time = 0.0
    """The epoch time of the last bridge data reload."""

    _bridge_data_last_modified_time = 0.0
    """The time the bridge data file on disk was last modified."""

    _bridge_data_reload_lock: asyncio.Lock | None = None
    """Serialises off-loop reloads so the mtime watcher and an explicit reload command never overlap."""

    _config_reload_tasks: set[asyncio.Task[None]] | None = None
    """Strong references to in-flight off-loop reload tasks (asyncio only weakly references tasks)."""

    def _load_bridge_data_blocking(self) -> reGenBridgeData | None:
        """Read and resolve bridge data from disk, returning the new model (or None when nothing to apply).

        This is the network-bound half of a reload: ``BridgeDataLoader.load`` resolves meta instructions
        (``top N`` etc.) against the horde stats API, which can take many seconds. It is split out so the
        event loop can run it in a thread (see :meth:`_reload_bridge_data_off_loop`) instead of stalling
        every other coroutine (job popping, heartbeats, submission) for the duration. Safe to call off the
        event loop; all the dependent state changes happen in :meth:`_apply_reloaded_bridge_data`.
        """
        if self.bridge_data._loaded_from_env_vars:
            return None

        if self.horde_model_reference_manager is None:
            logger.debug("No model reference manager available; skipping bridge data reload")
            return None

        try:
            return BridgeDataLoader.load(
                file_path=BRIDGE_CONFIG_FILENAME,
                horde_model_reference_manager=self.horde_model_reference_manager,
            )
        except Exception as e:
            logger.debug(e)

            if "No such file or directory" in str(e):
                logger.error(f"Could not find {BRIDGE_CONFIG_FILENAME}. Please create it and try again.")

            if isinstance(e, ValidationError):
                logger.error(f"The following fields in {BRIDGE_CONFIG_FILENAME} failed validation:")
                for error in e.errors():
                    logger.error(f"{error['loc'][0]}: {error['msg']}")

            return None

    def _apply_reloaded_bridge_data(self, bridge_data: reGenBridgeData) -> None:
        """Swap in freshly-loaded bridge data and re-derive dependent state (must run on the event loop)."""
        previous_effective = self._runtime_config.effective_max_threads
        # Captured before the swap so the download request below can detect models the reload dropped and
        # tell the download process to stop fetching them.
        previously_configured = set(self.bridge_data.image_models_to_load)
        # The aux/download gating is baked into the download process at construction, so a change to it
        # requires a restart; captured before the swap to compare against the reloaded config.
        previous_download_flags = self._download_process_flags()
        # The setter calls RuntimeConfig.update, which re-derives the effective concurrency cap
        # (clamped to the session ceiling) from the reloaded max_threads.
        self.bridge_data = bridge_data
        # Re-coerce on every reload so a config edit re-enabling a feature whose packages are still
        # missing is caught again; the warning only fires when a flag actually flips, not each tick.
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
        # Re-assert the config's download controls (config is authoritative on reload, overriding any
        # prior live TUI override); a None rate-limit means unlimited, sent as 0 to clear any cap.
        self._process_lifecycle.set_download_controls(
            paused=self.bridge_data.downloads_paused,
            rate_limit_kbps=self.bridge_data.download_rate_limit_kbps or 0,
            max_parallel_downloads=self.bridge_data.download_max_parallel_downloads,
            per_host_concurrency=self.bridge_data.download_per_host_concurrency,
        )
        # A config change can add image models that are not yet on disk (fetch them in the background so a
        # newly-configured model becomes servable without a restart) or remove models (stop their queued/
        # in-flight downloads); the startup trigger is one-shot, so the reload owns both directions.
        self._request_downloads_for_configured_missing(
            run_aux_if_incomplete=False,
            previously_configured=previously_configured,
        )
        # A change to the download process's construction-time gating (aux flags, nsfw, purge) is applied
        # by restarting it; live controls were already forwarded above.
        self._reload_download_process_if_flags_changed(previous_download_flags)

    def get_bridge_data_from_disk(self) -> None:
        """Load the bridge data from disk (blocking).

        Prefer :meth:`_reload_bridge_data_off_loop` when called from the event loop; this synchronous
        form is for startup (before the loop runs) and is kept for callers that are not on the loop.
        """
        bridge_data = self._load_bridge_data_blocking()
        if bridge_data is not None:
            self._apply_reloaded_bridge_data(bridge_data)

    async def _reload_bridge_data_off_loop(self) -> None:
        """Reload bridge data without stalling the event loop (the resolve is run in a worker thread)."""
        if self._bridge_data_reload_lock is None:
            self._bridge_data_reload_lock = asyncio.Lock()
        async with self._bridge_data_reload_lock:
            bridge_data = await asyncio.to_thread(self._load_bridge_data_blocking)
            if bridge_data is not None:
                self._apply_reloaded_bridge_data(bridge_data)

    def _schedule_config_reload(self) -> None:
        """Kick off an off-loop reload from synchronous, on-loop code (e.g. a supervisor command)."""
        if self._config_reload_tasks is None:
            self._config_reload_tasks = set()
        task = asyncio.create_task(self._reload_bridge_data_off_loop())
        self._config_reload_tasks.add(task)
        task.add_done_callback(self._config_reload_tasks.discard)

    async def _bridge_data_loop(self) -> None:
        while True:
            try:
                if self._state.shutting_down:
                    break

                self._bridge_data_last_modified_time = os.path.getmtime(BRIDGE_CONFIG_FILENAME)

                if self._last_bridge_data_reload_time < self._bridge_data_last_modified_time:
                    logger.info(f"Reloading {BRIDGE_CONFIG_FILENAME}")
                    await self._reload_bridge_data_off_loop()
                    # Capture mtime immediately after a successful load so that
                    # a modification during the load does not go undetected.
                    self._last_bridge_data_reload_time = os.path.getmtime(BRIDGE_CONFIG_FILENAME)
                    logger.success(f"Reloaded {BRIDGE_CONFIG_FILENAME}")
                    self.enable_performance_mode()
                await asyncio.sleep(self._bridge_data_loop_interval)
            except CancelledError as e:
                self._shutdown()
                logger.debug(f"CancelledError: {e}")
            except Exception as e:
                # Best-effort config watcher: a transient read/parse error (e.g. the file briefly
                # missing mid-rewrite) must never take the worker down. Log and retry next interval.
                logger.warning(f"Error while watching {BRIDGE_CONFIG_FILENAME} for changes: {e}")
                await asyncio.sleep(self._bridge_data_loop_interval)

    def _handle_exception(self, task: asyncio.Task[None]) -> None:
        """Supervise a finished main-loop task; shut down gracefully if one ends unexpectedly.

        Each main-loop coroutine is meant to run for the worker's whole life. If one finishes while
        the worker is not already shutting down — whether by raising or by returning early — the
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
            coroutines = [
                self._process_control_loop(),
                self._job_popper.run(),
                self._api_get_user_info_loop(),
                self._job_submitter.run(),
                self._alchemy_coordinator.run(),
            ]
            if not self.bridge_data._loaded_from_env_vars:
                coroutines.append(self._bridge_data_loop())

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
