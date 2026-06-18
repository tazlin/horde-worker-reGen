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
    SingleWorkerDetailsResponse,
    UserDetailsResponse,
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
from horde_worker_regen.process_management.alchemy_popper import AlchemyCoordinator
from horde_worker_regen.process_management.api_sessions import ApiSessions
from horde_worker_regen.process_management.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.job_popper import JobPopper
from horde_worker_regen.process_management.job_submitter import JobSubmitter
from horde_worker_regen.process_management.job_tracker import JobTracker
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
from horde_worker_regen.process_management.recovery_supervisor import RecoveryAction, RecoverySupervisor
from horde_worker_regen.process_management.run_metrics import RunMetricsSnapshot, WorkerRunMetrics
from horde_worker_regen.process_management.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.safety_orchestrator import SafetyOrchestrator
from horde_worker_regen.process_management.shutdown_manager import ShutdownManager
from horde_worker_regen.process_management.supervisor_channel import (
    RECENT_JOBS_IN_SNAPSHOT,
    DownloadPlanSummary,
    ProcessSnapshot,
    RecentJobRecord,
    SupervisorChannel,
    SupervisorCommand,
    SupervisorControlMessage,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from horde_worker_regen.process_management.worker_state import WorkerState
from horde_worker_regen.reporting.kudos_logger import KudosLogger
from horde_worker_regen.reporting.maintenance_messenger import MaintenanceModeMessenger
from horde_worker_regen.reporting.status_reporter import StatusReporter
from horde_worker_regen.utils.disk_monitor import DiskSpaceMonitor
from horde_worker_regen.utils.kudos_calculator import KudosCalculator
from horde_worker_regen.utils.kudos_utils import generate_kudos_info_string as _generate_kudos_info_string

if TYPE_CHECKING:
    from horde_worker_regen.process_management.job_models import HordeJobInfo
    from horde_worker_regen.process_management.job_tracker import TrackedJob
    from horde_worker_regen.process_management.messages import HordeJobMetricsMessage


@dataclasses.dataclass(frozen=True)
class SystemResources:
    """Hardware information detected at startup."""

    total_ram_bytes: int
    device_map: TorchDeviceMap

    @classmethod
    def detect(cls) -> SystemResources:
        """Detect system resources via psutil and hordelib's backend-agnostic accelerator inventory.

        Device discovery goes through ``enumerate_accelerators`` rather than ``torch.cuda`` directly so
        every ComfyUI-supported backend (CUDA/ROCm, Intel XPU, Apple MPS, DirectML, CPU) populates the
        device map; a bare ``torch.cuda.device_count()`` loop would yield no devices on non-CUDA backends.
        """
        import psutil
        from hordelib.api import enumerate_accelerators

        total_ram = psutil.virtual_memory().total

        device_map = TorchDeviceMap(root={})
        for accelerator in enumerate_accelerators():
            device_map.root[accelerator.index] = TorchDeviceInfo(
                device_name=accelerator.name,
                device_index=accelerator.index,
                total_memory=accelerator.total_vram_mb * 1024 * 1024,
            )

        return cls(total_ram_bytes=total_ram, device_map=device_map)


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
        self._download_plan_computed = False
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
            if self.bridge_data.high_memory_mode:
                vae_decode_semaphore_max = self.max_inference_processes

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

        # Expected-time-to-complete model: seeds from the last benchmark's per-tier reference it/s and
        # self-calibrates from this worker's own jobs, so a "slow" job becomes measurable rather than
        # guessed. Disabled-to-memory under test (no app-state read, no benchmark import, no perf file);
        # the model is unit-tested directly. Phase 4 consumes its expected_sampling_seconds.
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
        )

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

                # all_refs = horde_model_reference_manager.get_all_model_references(False)
                # _sd_ref = all_refs[MODEL_REFERENCE_CATEGORY.image_generation]
                _sd_ref = horde_model_reference_manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)

                if not isinstance(_sd_ref, dict):
                    raise ValueError(
                        "Expected dict[str, ImageGenerationModelRecord] for stable diffusion reference, got "
                        + str(type(_sd_ref)),
                    )

                self.stable_diffusion_reference = _sd_ref
            except Exception as e:
                logger.error(e)
                time.sleep(5)

    def remove_maintenance(self) -> None:
        """Removes the maintenance from the named worker."""
        simple_client = AIHordeAPISimpleClient()
        worker_details: SingleWorkerDetailsResponse = simple_client.worker_details_by_name(
            worker_name=self.bridge_data.dreamer_worker_name,
        )
        if worker_details is None:
            logger.debug(
                f"Worker with name {self.bridge_data.dreamer_worker_name} "
                "does not appear to exist already to remove maintenance.",
            )
            return
        modify_worker_request = ModifyWorkerRequest(
            apikey=self.bridge_data.api_key,
            worker_id=worker_details.id_,
            maintenance=False,
        )

        simple_client.worker_modify(modify_worker_request)

        logger.debug(
            f"Ensured worker with name {self.bridge_data.dreamer_worker_name} "
            f"({worker_details.id_}) is removed from maintenance.",
        )

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

    def _fetch_worker_details(self, worker_name: str) -> SingleWorkerDetailsResponse | None:
        """Synchronous worker-details lookup (run in a thread); mirrors :meth:`remove_maintenance`."""
        simple_client = AIHordeAPISimpleClient()
        return simple_client.worker_details_by_name(worker_name=worker_name)

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

        with logger.catch(reraise=True):
            await self._sleep(self._loop_interval)

            await self.receive_and_handle_process_messages()
            self._maybe_start_safety_processes()
            self._maybe_start_inference_processes()
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
            # before it can wedge the head of the queue (the 2026-06-18 overnight wedge).
            self._reconcile_orphaned_in_progress_jobs()

            # Save-our-ship: above the per-slot recovery, escalate a worker that is wedged as a whole
            # (no live process for pending work) to a soft reset and finally to giving up cleanly.
            self._run_recovery_supervisor()

            if self._state.shutting_down and not self._state.last_pop_recently():
                self._process_lifecycle.end_inference_processes()

            if self.is_time_for_shutdown():
                return False

        self._maybe_refresh_references()
        self.print_status_method()
        self._sample_disk_space()
        self._publish_supervisor_snapshot()

        await self._sleep(self._loop_interval / 2)
        return True

    _DOWNLOAD_STARTUP_GRACE_SECONDS = 90.0
    """How long to wait for the download process's first availability report before starting
    inference anyway, so a missing/failed download process can never wedge startup forever."""

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
            present = self._model_availability.present or set()
            configured = set(self.bridge_data.image_models_to_load)
            missing = sorted(configured - present)
            # Only run the (heavier) auxiliary pass on a genuinely incomplete install; a worker that
            # already has all its image models almost certainly has its aux models too.
            download_aux = len(missing) > 0
            if missing or download_aux:
                logger.info(
                    f"Worker has {len(present)} of {len(configured)} configured models on disk; "
                    f"background-downloading {len(missing)} missing: {missing}",
                )
                self._process_lifecycle.request_downloads(missing, download_aux=download_aux)

        self._maybe_start_safety_processes()
        self._maybe_start_inference_processes()

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
        """Whether some live inference slot is currently working on (references) the given job."""
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if not process_info.is_process_alive():
                continue
            referenced = process_info.last_job_referenced
            if referenced is not None and referenced.id_ == job_id:
                return True
        return False

    def _reconcile_orphaned_in_progress_jobs(self) -> None:
        """Punt jobs stuck INFERENCE_IN_PROGRESS that no live inference slot owns.

        Per-slot recovery faults the job of the slot it replaces, but a mis-association, a lost result,
        or a requeue race can still leave a *different* job marked in-progress with no owning slot. No
        result will ever arrive for it, so it pins the head of the queue forever (the 2026-06-18
        overnight wedge: one orphaned job stalled all image inference for 6.5h). This watchdog is the
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
        crash-looping with no healthy process) plus a recurring orphaned-job storm, not on transient
        capacity gaps. A merely slow, busy, replacing, or model-loading worker trips none of these, so
        a healthy worker is never wedged.
        """
        if self._state.shutting_down:
            return False
        return (
            self._is_inference_pool_unrecoverable()
            or self._is_safety_pool_unrecoverable()
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
        if not self._is_inference_capacity_available():
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

    def _sample_disk_space(self) -> None:
        """Sample disk free space at most every `_DISK_SAMPLE_INTERVAL_SECONDS`."""
        if time.time() - self._last_disk_sample_time < self._DISK_SAMPLE_INTERVAL_SECONDS:
            return
        self._last_disk_sample_time = time.time()
        self._disk_monitor.sample()

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

        summary_parts = [f"inf#{p.process_id}={p.last_process_state.name}" for p in inference]
        summary_parts += [f"safety#{p.process_id}={p.last_process_state.name}" for p in safety]
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
                self.get_bridge_data_from_disk()
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
        """Compute the config's disk-implications summary once, then return the cached value.

        Existence-only and torch-free (see :mod:`model_download_plan`); the live download process stays
        authoritative about integrity. Returns None until the model reference is available, so the
        snapshot simply omits the plan rather than blocking on a not-yet-loaded reference.
        """
        if self._download_plan_computed:
            return self._download_plan_summary

        reference = self.stable_diffusion_reference
        if reference is None:
            return None

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
        self._download_plan_computed = True
        return self._download_plan_summary

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
            allow_controlnet=bridge_data.allow_controlnet,
            allow_sdxl_controlnet=bridge_data.allow_sdxl_controlnet,
            allow_post_processing=bridge_data.allow_post_processing,
            high_performance_mode=bridge_data.high_performance_mode,
            moderate_performance_mode=bridge_data.moderate_performance_mode,
            high_memory_mode=bridge_data.high_memory_mode,
            very_high_memory_mode=bridge_data.very_high_memory_mode,
            extra_slow_worker=bridge_data.extra_slow_worker,
        )

        return WorkerStateSnapshot(
            session_start_time=self.session_start_time,
            shutting_down=self._state.shutting_down,
            maintenance_mode=self._state.last_pop_maintenance_mode or self._state.supervisor_paused,
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
            time_spent_no_jobs_available=self._job_popper.time_spent_no_jobs_available,
            kudos_per_hour=kudos_per_hour,
            kudos_this_session=kudos_session,
            active_models=active_models,
            gpu_utilization_mean_percent=run_metrics.gpu_utilization_mean_percent,
            gpu_utilization_busy_fraction=run_metrics.gpu_utilization_busy_fraction,
            vram_high_water_mb_per_process=run_metrics.vram_used_high_water_mb_per_process,
            ram_high_water_mb_per_process=run_metrics.ram_used_high_water_mb_per_process,
            disk_free_bytes=dict(self._disk_monitor.current_free_bytes),
            recent_jobs=[
                RecentJobRecord.from_metrics_record(job) for job in run_metrics.jobs[-RECENT_JOBS_IN_SNAPSHOT:]
            ],
            downloads=self._model_availability.status,
            download_plan=self._get_download_plan_summary(),
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

    def get_bridge_data_from_disk(self) -> None:
        """Load the bridge data from disk."""
        if self.bridge_data._loaded_from_env_vars:
            return

        if self.horde_model_reference_manager is None:
            logger.debug("No model reference manager available; skipping bridge data reload")
            return

        try:
            previous_effective = self._runtime_config.effective_max_threads
            # The setter calls RuntimeConfig.update, which re-derives the effective concurrency cap
            # (clamped to the session ceiling) from the reloaded max_threads.
            self.bridge_data = BridgeDataLoader.load(
                file_path=BRIDGE_CONFIG_FILENAME,
                horde_model_reference_manager=self.horde_model_reference_manager,
            )
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
            )
        except Exception as e:
            logger.debug(e)

            if "No such file or directory" in str(e):
                logger.error(f"Could not find {BRIDGE_CONFIG_FILENAME}. Please create it and try again.")

            if isinstance(e, ValidationError):
                logger.error(f"The following fields in {BRIDGE_CONFIG_FILENAME} failed validation:")
                for error in e.errors():
                    logger.error(f"{error['loc'][0]}: {error['msg']}")

            return

    async def _bridge_data_loop(self) -> None:
        while True:
            try:
                if self._state.shutting_down:
                    break

                self._bridge_data_last_modified_time = os.path.getmtime(BRIDGE_CONFIG_FILENAME)

                if self._last_bridge_data_reload_time < self._bridge_data_last_modified_time:
                    logger.info(f"Reloading {BRIDGE_CONFIG_FILENAME}")
                    self.get_bridge_data_from_disk()
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
