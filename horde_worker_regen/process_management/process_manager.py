from __future__ import annotations

import asyncio
import asyncio.exceptions
import dataclasses
import multiprocessing
import os
import ssl
import sys
import time
from asyncio import CancelledError
from collections import deque
from collections.abc import Mapping
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import Lock as Lock_MultiProcessing
from multiprocessing.synchronize import Semaphore

import aiohttp
import aiohttp.client_exceptions
import certifi
from aiohttp import ClientSession
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY, STABLE_DIFFUSION_BASELINE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import StableDiffusion_ModelReference
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.ai_horde_clients import (
    AIHordeAPIAsyncClientSession,
    AIHordeAPISimpleClient,
)
from horde_sdk.ai_horde_api.apimodels import (
    FindUserRequest,
    ModifyWorkerRequest,
    SingleWorkerDetailsResponse,
    UserDetailsResponse,
)
from loguru import logger
from pydantic import ValidationError

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader
from horde_worker_regen.consts import (
    BRIDGE_CONFIG_FILENAME,
    VRAM_HEAVY_MODELS,
)
from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.job_models import APIWorkerMessage
from horde_worker_regen.process_management.job_popper import JobPopper
from horde_worker_regen.process_management.job_submitter import JobSubmitter
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.lru_cache import LRUCache
from horde_worker_regen.process_management.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.safety_orchestrator import SafetyOrchestrator
from horde_worker_regen.process_management.shutdown_manager import ShutdownManager
from horde_worker_regen.process_management.worker_state import WorkerState
from horde_worker_regen.reporting.kudos_logger import KudosLogger
from horde_worker_regen.reporting.maintenance_messenger import MaintenanceModeMessenger
from horde_worker_regen.reporting.status_reporter import StatusReporter
from horde_worker_regen.utils.kudos_calculator import KudosCalculator
from horde_worker_regen.utils.kudos_utils import generate_kudos_info_string as _generate_kudos_info_string


@dataclasses.dataclass(frozen=True)
class SystemResources:
    """Hardware information detected at startup."""

    total_ram_bytes: int
    device_map: TorchDeviceMap

    @classmethod
    def detect(cls) -> SystemResources:
        """Detect system resources by probing psutil and torch.cuda."""
        import psutil
        import torch

        total_ram = psutil.virtual_memory().total

        device_map = TorchDeviceMap(root={})
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            device_map.root[i] = TorchDeviceInfo(
                device_name=device.name,
                device_index=i,
                total_memory=device.total_memory,
            )

        return cls(total_ram_bytes=total_ram, device_map=device_map)


@dataclasses.dataclass
class MultiprocessingPrimitives:
    """Multiprocessing primitives created for IPC."""

    process_message_queue: ProcessQueue
    inference_semaphore: Semaphore
    disk_lock: Lock_MultiProcessing
    aux_model_lock: Lock_MultiProcessing
    vae_decode_semaphore: Semaphore

    @classmethod
    def create(
        cls,
        ctx: BaseContext,
        max_concurrent_inference: int,
        vae_decode_semaphore_max: int,
    ) -> MultiprocessingPrimitives:
        """Create real multiprocessing primitives from a context."""
        return cls(
            process_message_queue=multiprocessing.Queue(),
            inference_semaphore=Semaphore(max_concurrent_inference, ctx=ctx),
            disk_lock=Lock_MultiProcessing(ctx=ctx),
            aux_model_lock=Lock_MultiProcessing(ctx=ctx),
            vae_decode_semaphore=Semaphore(vae_decode_semaphore_max, ctx=ctx),
        )


sslcontext = ssl.create_default_context(cafile=certifi.where())


# As of 3.11, asyncio.TimeoutError is deprecated and is an alias for builtins.TimeoutError
_async_client_exceptions: tuple[type[Exception], ...] = (TimeoutError, aiohttp.client_exceptions.ClientError, OSError)

if sys.version_info[:2] == (3, 10):
    _async_client_exceptions = (asyncio.exceptions.TimeoutError, aiohttp.client_exceptions.ClientError, OSError)

_caught_signal = False


class HordeWorkerProcessManager:
    """Manages and controls processes to act as a horde worker."""

    bridge_data: reGenBridgeData
    """The bridge data for this worker."""

    horde_model_reference_manager: ModelReferenceManager
    """The model reference manager for this worker."""

    max_inference_processes: int
    """The maximum number of inference processes that can be active. This is not the number of jobs that
    can run at once. Use `max_concurrent_inference_processes` to control that behavior."""

    _max_concurrent_inference_processes: int
    """The maximum number of inference processes that can run jobs concurrently. \
        This is set at initialization to prevent changing the value at runtime."""

    @property
    def max_concurrent_inference_processes(self) -> int:
        """The maximum number of inference processes that can run jobs concurrently."""
        return self._max_concurrent_inference_processes

    max_safety_processes: int
    """The maximum number of safety processes that can run at once."""
    max_download_processes: int
    """The maximum number of download processes that can run at once."""

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

    @property
    def kudos_generated_this_session(self) -> float:
        """The amount of kudos generated this entire session."""
        return self._state.kudos_generated_this_session

    @kudos_generated_this_session.setter
    def kudos_generated_this_session(self, value: float) -> None:
        self._state.kudos_generated_this_session = value

    @property
    def kudos_events(self) -> deque[tuple[float, float]]:
        """A deque of kudos events, each is a tuple of the time the event occurred and the kudos generated."""
        return self._state.kudos_events

    @kudos_events.setter
    def kudos_events(self, value: deque[tuple[float, float]]) -> None:
        self._state.kudos_events = value

    session_start_time: float = 0
    """The time at which the session started in epoch time."""

    _aiohttp_client_session: aiohttp.ClientSession
    """The aiohttp client session to use for making network calls."""

    stable_diffusion_reference: StableDiffusion_ModelReference | None
    """The class which contains the list of models from horde_model_reference."""

    def get_model_baseline(self, model_name: str) -> STABLE_DIFFUSION_BASELINE_CATEGORY | str | None:
        """Return the baseline of the model."""
        if self.stable_diffusion_reference is None:
            return None

        if model_name not in self.stable_diffusion_reference.root:
            return None

        return self.stable_diffusion_reference.root[model_name].baseline

    horde_client_session: AIHordeAPIAsyncClientSession
    """The context manager for the horde sdk client."""

    user_info: UserDetailsResponse | None = None
    """The user info for the user that this worker is logged in as."""

    _process_map: ProcessMap
    """Shared by reference with all sub-managers. Created once; never reassigned after __init__."""

    _horde_model_map: HordeModelMap

    _device_map: TorchDeviceMap
    """A mapping (dict) of device IDs to TorchDeviceInfo objects. Contains some helper methods."""

    _loop_interval: float = 0.20
    """The number of seconds to wait between each loop of the main process (inter process management) loop."""
    _api_get_user_info_interval = 15
    """The number of seconds to wait between each fetch of the user info."""

    _last_get_user_info_time: float = 0
    """The time at which the user info was last fetched."""

    @property
    def num_total_processes(self) -> int:
        """The total number of processes that can be running at once (inference, safety, and download)."""
        return self.max_inference_processes + self.max_safety_processes + self.max_download_processes

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
    def _api_messages_received(self) -> dict[str | None, APIWorkerMessage]:
        return self._job_popper._api_messages_received

    @property
    def post_process_job_overlap_allowed(self) -> bool:
        """Return true if post processing jobs are allowed to overlap."""
        return (
            self.bridge_data.moderate_performance_mode or self.bridge_data.high_performance_mode
        ) and self.bridge_data.post_process_job_overlap

    def __init__(
        self,
        *,
        ctx: BaseContext,
        bridge_data: reGenBridgeData,
        horde_model_reference_manager: ModelReferenceManager,
        target_ram_overhead_bytes: int = 9 * 1024 * 1024 * 1024,
        target_vram_overhead_bytes_map: Mapping[int, int] | None = None,  # FIXME
        max_safety_processes: int = 1,
        max_download_processes: int = 1,
        amd_gpu: bool = False,
        directml: int | None = None,
        system_resources: SystemResources | None = None,
        mp_primitives: MultiprocessingPrimitives | None = None,
        skip_api_init: bool = False,
        stable_diffusion_reference: StableDiffusion_ModelReference | None = None,
    ) -> None:
        """Initialise the process manager.

        Args:
            ctx: The multiprocessing context to use.
            bridge_data: The bridge data for this worker.
            horde_model_reference_manager: The model reference manager for this worker.
            target_ram_overhead_bytes: The target amount of RAM to keep free.
            target_vram_overhead_bytes_map: The target amount of VRAM to keep free.
            max_safety_processes: The maximum number of safety processes that can run at once.
            max_download_processes: The maximum number of download processes that can run at once.
            amd_gpu: Whether or not the GPU is an AMD GPU.
            directml: ID of the potential directml device.
            system_resources: Pre-detected system resources. If None, auto-detects via torch/psutil.
            mp_primitives: Pre-created multiprocessing primitives. If None, creates real ones from ctx.
            skip_api_init: If True, skip the remove_maintenance API call during init.
            stable_diffusion_reference: Pre-loaded model reference. If None, fetches from ModelReferenceManager.
        """
        self.session_start_time = time.time()
        self._state = WorkerState()

        self.bridge_data = bridge_data
        logger.debug(f"Models to load: {bridge_data.image_models_to_load}")
        logger.debug(f"Custom Models to load: {bridge_data.custom_models}")

        self.horde_model_reference_manager = horde_model_reference_manager

        self._process_map = ProcessMap({})
        self._horde_model_map = HordeModelMap(root={})

        self.max_safety_processes = max_safety_processes
        self.max_download_processes = max_download_processes

        self._max_concurrent_inference_processes = bridge_data.max_threads

        self.max_inference_processes = self.bridge_data.queue_size + self.bridge_data.max_threads

        self._lru = LRUCache(self.max_inference_processes)

        self._amd_gpu = amd_gpu
        self._directml = directml

        if len(self.bridge_data.image_models_to_load) == 1 and self.max_concurrent_inference_processes == 1:
            self.max_inference_processes = 1

        self._job_tracker = JobTracker()

        self.target_vram_overhead_bytes_map = target_vram_overhead_bytes_map  # TODO

        # Detect or use provided system resources
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

        # Create or use provided multiprocessing primitives
        if mp_primitives is None:
            vae_decode_semaphore_max = 1
            if self.bridge_data.high_memory_mode:
                vae_decode_semaphore_max = self.max_inference_processes

            mp_primitives = MultiprocessingPrimitives.create(
                ctx=ctx,
                max_concurrent_inference=self._max_concurrent_inference_processes,
                vae_decode_semaphore_max=vae_decode_semaphore_max,
            )

        self._process_message_queue = mp_primitives.process_message_queue
        self._inference_semaphore = mp_primitives.inference_semaphore
        self._disk_lock = mp_primitives.disk_lock
        self._aux_model_lock = mp_primitives.aux_model_lock
        self._vae_decode_semaphore = mp_primitives.vae_decode_semaphore

        self._process_lifecycle = ProcessLifecycleManager(
            process_map=self._process_map,
            horde_model_map=self._horde_model_map,
            job_tracker=self._job_tracker,
            process_message_queue=self._process_message_queue,
            inference_semaphore=self._inference_semaphore,
            disk_lock=self._disk_lock,
            aux_model_lock=self._aux_model_lock,
            vae_decode_semaphore=self._vae_decode_semaphore,
            get_bridge_data=lambda: self.bridge_data,
            max_inference_processes=self.max_inference_processes,
            max_safety_processes=self.max_safety_processes,
            amd_gpu=self._amd_gpu,
            directml=self._directml,
            abort_callback=self._abort,
            state=self._state,
        )

        self._message_dispatcher = MessageDispatcher(
            process_map=self._process_map,
            horde_model_map=self._horde_model_map,
            job_tracker=self._job_tracker,
            process_message_queue=self._process_message_queue,
            get_model_baseline=self.get_model_baseline,
            get_bridge_data=lambda: self.bridge_data,
            on_unload_vram=self.unload_models_from_vram,
            state=self._state,
        )

        self._safety_orchestrator = SafetyOrchestrator(
            process_map=self._process_map,
            job_tracker=self._job_tracker,
            process_lifecycle=self._process_lifecycle,
            get_bridge_data=lambda: self.bridge_data,
            get_stable_diffusion_reference=lambda: self.stable_diffusion_reference,
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
            get_bridge_data=lambda: self.bridge_data,
            get_model_baseline=self.get_model_baseline,
            get_stable_diffusion_reference=lambda: self.stable_diffusion_reference,
            max_concurrent_inference_processes=self._max_concurrent_inference_processes,
            max_inference_processes=self.max_inference_processes,
            lru=self._lru,
        )

        self._job_submitter = JobSubmitter(
            state=self._state,
            job_tracker=self._job_tracker,
            shutdown_manager=self._shutdown_manager,
            get_bridge_data=lambda: self.bridge_data,
            get_stable_diffusion_reference=lambda: self.stable_diffusion_reference,
            get_horde_client_session=lambda: self.horde_client_session,
            get_aiohttp_session=lambda: self._aiohttp_client_session,
            dry_run_skip_api=bridge_data.dry_run_skip_api,
        )

        self._job_popper = JobPopper(
            state=self._state,
            process_map=self._process_map,
            job_tracker=self._job_tracker,
            shutdown_manager=self._shutdown_manager,
            get_bridge_data=lambda: self.bridge_data,
            get_horde_client_session=lambda: self.horde_client_session,
            get_aiohttp_session=lambda: self._aiohttp_client_session,
            get_effective_megapixelsteps=lambda job: self._inference_scheduler.get_single_job_effective_megapixelsteps(
                job,
            ),
            max_inference_processes=self.max_inference_processes,
            max_concurrent_inference_processes=self._max_concurrent_inference_processes,
            dry_run_skip_api=bridge_data.dry_run_skip_api,
        )

        if stable_diffusion_reference is not None:
            self.stable_diffusion_reference = stable_diffusion_reference
        else:
            self.stable_diffusion_reference = None
            self._init_model_reference()

    def _init_model_reference(self) -> None:
        """Fetch the stable diffusion model reference, retrying on failure."""
        while self.stable_diffusion_reference is None:
            try:
                horde_model_reference_manager = ModelReferenceManager(
                    download_and_convert_legacy_dbs=False,
                    override_existing=False,
                )
                all_refs = horde_model_reference_manager.get_all_model_references(False)
                _sd_ref = all_refs[MODEL_REFERENCE_CATEGORY.stable_diffusion]

                if not isinstance(_sd_ref, StableDiffusion_ModelReference):
                    raise ValueError("Expected StableDiffusion_ModelReference")

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

        horde_model_record = self.stable_diffusion_reference.root[horde_model_name]

        if horde_model_record.baseline == STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_1:
            return int(3 * 1024 * 1024 * 1024)
        if horde_model_record.baseline == STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_2_512:
            return 4 * 1024 * 1024 * 1024
        if horde_model_record.baseline == STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_2_768:
            return 5 * 1024 * 1024 * 1024
        if horde_model_record.baseline == STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_xl:
            return int(5.75 * 1024 * 1024 * 1024)

        raise ValueError(f"Model {horde_model_name} has an unknown baseline {horde_model_record.baseline}")

    def receive_and_handle_process_messages(self) -> None:
        """Receive and handle any messages from the child processes.

        Delegates to MessageDispatcher.
        """
        self._message_dispatcher.receive_and_handle_process_messages()

    def unload_models_from_vram(
        self,
        process_with_model: HordeProcessInfo,
    ) -> None:
        """Unload models from VRAM from processes that are not running a job."""
        self._inference_scheduler.unload_models_from_vram(process_with_model)

    def start_evaluate_safety(self) -> None:
        """Start evaluating the safety of the next job pending a safety check, if any."""
        self._safety_orchestrator.start_evaluate_safety()

    @property
    def _num_job_slowdowns(self) -> int:
        return self._job_submitter._num_job_slowdowns

    def _last_pop_recently(self) -> bool:
        return self._state.last_pop_recently()

    @property
    def _last_pop_maintenance_mode(self) -> bool:
        return self._state.last_pop_maintenance_mode

    @_last_pop_maintenance_mode.setter
    def _last_pop_maintenance_mode(self, value: bool) -> None:
        self._state.last_pop_maintenance_mode = value

    @property
    def _time_spent_no_jobs_available(self) -> float:
        return self._job_popper._time_spent_no_jobs_available

    @property
    def _too_many_consecutive_failed_jobs(self) -> bool:
        return self._state.too_many_consecutive_failed_jobs

    @_too_many_consecutive_failed_jobs.setter
    def _too_many_consecutive_failed_jobs(self, value: bool) -> None:
        self._state.too_many_consecutive_failed_jobs = value

    @property
    def _too_many_consecutive_failed_jobs_time(self) -> float:
        return self._state.too_many_consecutive_failed_jobs_time

    @_too_many_consecutive_failed_jobs_time.setter
    def _too_many_consecutive_failed_jobs_time(self, value: float) -> None:
        self._state.too_many_consecutive_failed_jobs_time = value

    @property
    def _too_many_consecutive_failed_jobs_wait_time(self) -> int:
        return self._job_popper._too_many_consecutive_failed_jobs_wait_time

    _user_info_failed = False
    """Whether the API request to fetch user info failed."""
    _user_info_failed_reason: str | None = None
    """The reason the API request to fetch user info failed."""

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
            self.kudos_generated_this_session,
            self.session_start_time,
            self._time_spent_no_jobs_available,
            self.kudos_events,
        )

        # Update the events deque with cleaned version
        self.kudos_events = cleaned_events

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
            self.kudos_events,
        )
        self.kudos_events = cleaned_events
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
            kudos_generated_this_session=self.kudos_generated_this_session,
            time_since_session_start=time_since_session_start,
            kudos_per_hour_session=kudos_per_hour_session,
            kudos_total_past_hour=kudos_total_past_hour,
            active_kudos_per_hour=active_kudos_per_hour,
            time_spent_no_jobs_available=self._time_spent_no_jobs_available,
            max_time_spent_no_jobs_available=self._job_popper._max_time_spent_no_jobs_available,
        )

    def log_kudos_info(self, kudos_info_string: str) -> None:
        """Log the kudos information string.

        Args:
            kudos_info_string: The kudos information string to log.
        """
        logger.debug(f"len(kudos_events): {len(self.kudos_events)}")
        KudosLogger.log_kudos_info(
            kudos_info_string=kudos_info_string,
            kudos_generated_this_session=self.kudos_generated_this_session,
            user_info=self.user_info,
            limited_console_messages=self.bridge_data.limited_console_messages,
        )

    async def api_get_user_info(self) -> None:
        """Get the information associated with this API key from the API."""
        if self._shutting_down or self._last_pop_maintenance_mode:
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

    async def _api_get_user_info_loop(self) -> None:
        """Run the API get user info loop."""
        logger.debug("In _api_get_user_info_loop")
        while True:
            with logger.catch():
                try:
                    await self.api_get_user_info()
                    if self.is_time_for_shutdown() or self._shut_down:
                        break
                except CancelledError as e:
                    self._shutdown()
                    logger.debug(f"CancelledError: {e}")

            await asyncio.sleep(self._api_get_user_info_interval)

    _status_message_frequency = 20.0
    """The rate in seconds at which to print status messages with details about the current state of the worker."""
    _last_status_message_time = 0.0
    """The epoch time of the last status message."""

    @property
    def _replaced_due_to_maintenance(self) -> bool:
        return self._job_popper._replaced_due_to_maintenance

    @_replaced_due_to_maintenance.setter
    def _replaced_due_to_maintenance(self, value: bool) -> None:
        self._job_popper._replaced_due_to_maintenance = value

    async def _process_control_loop(self) -> None:
        self._process_lifecycle.start_safety_processes()
        self._process_lifecycle.start_inference_processes()

        while True:
            try:
                if self.stable_diffusion_reference is None:
                    return
                with logger.catch(reraise=True):
                    await asyncio.sleep(self._loop_interval)

                    async with self._job_tracker.all_locks():
                        self.receive_and_handle_process_messages()
                        self.detect_deadlock()

                    if len(self._job_tracker.jobs_pending_safety_check) > 0:
                        async with self._job_tracker.safety_check_lock:
                            self.start_evaluate_safety()

                    free_process_or_model_loaded = (
                        self.is_free_inference_process_available() or self.is_any_model_preloaded()
                    )

                    if (
                        self._last_pop_maintenance_mode
                        and len(self._job_tracker.jobs_pending_inference) == 0
                        and len(self._job_tracker.jobs_in_progress) == 0
                        and len(self._job_tracker.jobs_pending_safety_check) == 0
                        and len(self._job_tracker.jobs_being_safety_checked) == 0
                        and len(self._job_tracker.jobs_pending_submit) == 0
                        and not self._replaced_due_to_maintenance
                    ):
                        logger.warning("Reloading all process due to maintenance mode")
                        for process_info in self._process_map.values():
                            if process_info.process_type == HordeProcessType.INFERENCE:
                                self._process_lifecycle._replace_inference_process(process_info)
                            self._replaced_due_to_maintenance = True
                        MaintenanceModeMessenger.print_maintenance_mode_messages()

                    if free_process_or_model_loaded and len(self._job_tracker.jobs_pending_inference) > 0:
                        async with self._job_tracker.all_locks(include_timestamps=True):
                            self._inference_scheduler.run_scheduling_cycle(self.stable_diffusion_reference)

                    async with self._job_tracker.all_locks():
                        await asyncio.sleep(self._loop_interval)
                        self.receive_and_handle_process_messages()
                        if self._process_lifecycle.replace_hung_processes():
                            await asyncio.sleep(self._loop_interval / 2)
                            await asyncio.sleep(self._loop_interval / 2)
                        self._process_lifecycle._replace_all_safety_process()

                    if self._shutting_down and not self._last_pop_recently():
                        self._process_lifecycle.end_inference_processes()

                    if self.is_time_for_shutdown():
                        self._start_timed_shutdown()
                        break

                self.print_status_method()

                await asyncio.sleep(self._loop_interval / 2)
            except CancelledError as e:
                self._shutdown()
                logger.debug(f"CancelledError: {e}")

        while len(self._job_tracker.jobs_pending_inference) > 0:
            await asyncio.sleep(0.2)
            async with self._job_tracker.all_locks():
                self.receive_and_handle_process_messages()
                self.detect_deadlock()
                self._process_lifecycle.replace_hung_processes()
            await asyncio.sleep(0.2)

        self._process_lifecycle.end_inference_processes(force=True)
        self._process_lifecycle.end_safety_processes()

        logger.info("Shutting down process manager")
        self._shut_down = True
        for process in self._process_map.values():
            process.mp_process.terminate()
            process.mp_process.join(0.2)

        await asyncio.sleep(0.2)

        return

    def detect_deadlock(self) -> None:
        """Detect if there are jobs in the queue but no processes doing anything."""
        self._message_dispatcher.detect_deadlock()

    def print_status_method(self) -> None:
        """Print the status of the worker if it's time to do so."""
        reporter = StatusReporter(
            last_status_message_time=self._last_status_message_time,
            status_message_frequency=self._status_message_frequency,
        )

        if not reporter.should_print_status(self._last_pop_maintenance_mode):
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
            api_messages_received=self._api_messages_received,
            jobs_pending_inference=self._job_tracker.jobs_pending_inference,
            active_models=active_models,
            pending_megapixelsteps=self._job_tracker.get_pending_megapixelsteps(),
            num_jobs_total=self.num_jobs_total,
            total_num_completed_jobs=self._job_tracker.total_num_completed_jobs,
            num_jobs_faulted=self._job_tracker._num_jobs_faulted,
            num_job_slowdowns=self._num_job_slowdowns,
            num_process_recoveries=self._process_lifecycle._num_process_recoveries,
            time_spent_no_jobs_available=self._time_spent_no_jobs_available,
            user_info=self.user_info,
            max_concurrent_inference_processes=self.max_concurrent_inference_processes,
            device_map=self._device_map,
            too_many_consecutive_failed_jobs=self._too_many_consecutive_failed_jobs,
            too_many_consecutive_failed_jobs_time=self._too_many_consecutive_failed_jobs_time,
            too_many_consecutive_failed_jobs_wait_time=self._too_many_consecutive_failed_jobs_wait_time,
            session_start_time=self.session_start_time,
            shutting_down=self._shutting_down,
            jobs_pending_safety_check=len(self._job_tracker.jobs_pending_safety_check),
            jobs_being_safety_checked=len(self._job_tracker.jobs_being_safety_checked),
            jobs_in_progress=len(self._job_tracker.jobs_in_progress),
            total_ram_gigabytes=self.total_ram_gigabytes,
        )

        # Update state from reporter
        self._last_status_message_time = reporter.last_status_message_time
        self._status_message_frequency = updated_frequency

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

        try:
            self.bridge_data = BridgeDataLoader.load(
                file_path=BRIDGE_CONFIG_FILENAME,
                horde_model_reference_manager=self.horde_model_reference_manager,
            )
            if self.bridge_data.max_threads != self._max_concurrent_inference_processes:
                logger.warning(
                    f"max_threads in {BRIDGE_CONFIG_FILENAME} cannot be changed while the worker is running.",
                )
            logger.debug(f"Models to load: {self.bridge_data.image_models_to_load}")
            logger.debug(f"Custom models: {self.bridge_data.custom_models}")
        except Exception as e:
            logger.debug(e)

            if "No such file or directory" in str(e):
                logger.error(f"Could not find {BRIDGE_CONFIG_FILENAME}. Please create it and try again.")

            if isinstance(e, ValidationError):
                # Print a list of fields that failed validation
                logger.error(f"The following fields in {BRIDGE_CONFIG_FILENAME} failed validation:")
                for error in e.errors():
                    logger.error(f"{error['loc'][0]}: {error['msg']}")

            return

    async def _bridge_data_loop(self) -> None:
        while True:
            try:
                if self._shutting_down:
                    break

                self._bridge_data_last_modified_time = os.path.getmtime(BRIDGE_CONFIG_FILENAME)

                if self._last_bridge_data_reload_time < self._bridge_data_last_modified_time:
                    logger.info(f"Reloading {BRIDGE_CONFIG_FILENAME}")
                    self.get_bridge_data_from_disk()
                    self._last_bridge_data_reload_time = time.time()
                    logger.success(f"Reloaded {BRIDGE_CONFIG_FILENAME}")
                    self.enable_performance_mode()
                await asyncio.sleep(self._bridge_data_loop_interval)
            except CancelledError as e:
                self._shutdown()
                logger.debug(f"CancelledError: {e}")

    def _handle_exception(self, future: asyncio.Future) -> None:
        """Logs exceptions from asyncio tasks.

        :param future: asyncio task to monitor
        :return: None
        """
        ex = future.exception()
        if ex is not None:
            if self._shutting_down:
                logger.debug(f"exception thrown by a main loop task: {ex}")
            else:
                logger.error(f"exception thrown by a main loop task: {ex}")
                logger.exception(ex)

    async def _main_loop(self) -> None:
        self._aiohttp_client_session = ClientSession(requote_redirect_url=False)

        import logfire

        logfire.instrument_aiohttp_client()

        self.horde_client_session = AIHordeAPIAsyncClientSession(
            aiohttp_session=self._aiohttp_client_session,
            apikey=self.bridge_data.api_key,
        )

        async with self._aiohttp_client_session, self.horde_client_session:
            coroutines = [
                self._process_control_loop(),
                self._job_popper.run(),
                self._api_get_user_info_loop(),
                self._job_submitter.run(),
            ]
            if not self.bridge_data._loaded_from_env_vars:
                coroutines.append(self._bridge_data_loop())

            tasks = [asyncio.create_task(coro) for coro in coroutines]
            for task in tasks:
                task.add_done_callback(self._handle_exception)

            await asyncio.gather(*tasks)

    def start(self) -> None:
        """Start the process manager."""
        import signal

        signal.signal(signal.SIGINT, self.signal_handler)
        asyncio.run(self._main_loop())

    def signal_handler(self, sig: int, frame: object) -> None:
        """Handle SIGINT and SIGTERM."""
        self._shutdown_manager.signal_handler(sig, frame)

        global _caught_signal
        _caught_signal = True

    def _start_timed_shutdown(self) -> None:
        self._shutdown_manager.start_timed_shutdown()

    @property
    def _shutting_down(self) -> bool:
        return self._state.shutting_down

    @_shutting_down.setter
    def _shutting_down(self, value: bool) -> None:
        self._state.shutting_down = value

    @property
    def _shut_down(self) -> bool:
        return self._state.shut_down

    @_shut_down.setter
    def _shut_down(self, value: bool) -> None:
        self._state.shut_down = value

    def _shutdown(self) -> None:
        self._shutdown_manager.shutdown()

    def _abort(self) -> None:
        """Exit as soon as possible, aborting all processes and jobs immediately."""
        self._shutdown_manager.abort()
