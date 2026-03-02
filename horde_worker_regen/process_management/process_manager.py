import asyncio
import asyncio.exceptions
import collections
import multiprocessing
import os
import queue
import random
import ssl
import sys
import time
from asyncio import CancelledError, Task
from collections import deque
from collections.abc import Mapping
from io import BytesIO
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import Lock as Lock_MultiProcessing
from multiprocessing.synchronize import Semaphore

import aiohttp
import aiohttp.client_exceptions
import certifi
import psutil
import yarl
from aiohttp import ClientSession
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY, STABLE_DIFFUSION_BASELINE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import StableDiffusion_ModelReference
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.ai_horde_clients import (
    AIHordeAPIAsyncClientSession,
    AIHordeAPISimpleClient,
)
from horde_sdk.ai_horde_api.apimodels import (
    FindUserRequest,
    GenMetadataEntry,
    ImageGenerateJobPopRequest,
    ImageGenerateJobPopResponse,
    JobSubmitResponse,
    ModifyWorkerRequest,
    SingleWorkerDetailsResponse,
    UserDetailsResponse,
)
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from loguru import logger
from pydantic import ValidationError

import horde_worker_regen
from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader
from horde_worker_regen.consts import (
    BRIDGE_CONFIG_FILENAME,
    KNOWN_SLOW_WORKFLOWS,
    MAX_SOURCE_IMAGE_RETRIES,
    VRAM_HEAVY_MODELS,
)
from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.job_models import (
    APIWorkerMessage,
    HordeJobInfo,
    NextJobAndProcess,
    PendingSubmitJob,
)
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.lru_cache import LRUCache
from horde_worker_regen.process_management.messages import (
    HordeAuxModelStateChangeMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeControlModelMessage,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeModelStateChangeMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    HordeSafetyControlMessage,
    HordeSafetyResultMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.reporting.kudos_logger import KudosLogger
from horde_worker_regen.reporting.kudos_training_recorder import KudosTrainingRecorder
from horde_worker_regen.reporting.maintenance_messenger import MaintenanceModeMessenger
from horde_worker_regen.reporting.status_reporter import StatusReporter
from horde_worker_regen.utils.image_utils import base64_image_to_stream_buffer as _base64_image_to_stream_buffer
from horde_worker_regen.utils.job_utils import (
    get_single_job_effective_megapixelsteps as _get_single_job_effective_megapixelsteps,
)
from horde_worker_regen.utils.kudos_calculator import KudosCalculator
from horde_worker_regen.utils.kudos_utils import generate_kudos_info_string as _generate_kudos_info_string

sslcontext = ssl.create_default_context(cafile=certifi.where())


# As of 3.11, asyncio.TimeoutError is deprecated and is an alias for builtins.TimeoutError
_async_client_exceptions: tuple[type[Exception], ...] = (TimeoutError, aiohttp.client_exceptions.ClientError, OSError)

if sys.version_info[:2] == (3, 10):
    _async_client_exceptions = (asyncio.exceptions.TimeoutError, aiohttp.client_exceptions.ClientError, OSError)

_excludes_for_job_dump = {
    "job_image_results": True,
    "sdk_api_job_info": {
        "payload": {"prompt": True, "special": True},
        "skipped": True,
        "source_image": True,
        "source_mask": True,
        "extra_source_images": True,
        "r2_upload": True,
        "r2_uploads": True,
    },
}

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

    kudos_generated_this_session: float = 0
    """The amount of kudos generated this entire session."""
    kudos_events: deque[tuple[float, float]]
    """A deque of kudos events, each is a tuple of the time the event occurred and the kudos generated."""
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
    _last_user_info_fetch_time: float = 0
    """The time at which the user info was last fetched."""
    _user_info_fetch_interval: float = 10
    """The number of seconds between each fetch of the user info."""

    _process_map: ProcessMap
    """A mapping (dict) of process IDs to HordeProcessInfo objects. Contains some helper methods."""
    _horde_model_map: HordeModelMap
    """A mapping (dict) of horde model names to ModelInfo objects. Contains some helper methods."""
    _device_map: TorchDeviceMap
    """A mapping (dict) of device IDs to TorchDeviceInfo objects. Contains some helper methods."""

    _loop_interval: float = 0.20
    """The number of seconds to wait between each loop of the main process (inter process management) loop."""
    _api_call_loop_interval = 1
    """The number of seconds to wait between each loop of the main API call loop."""

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

    _api_messages_received: dict[str, APIWorkerMessage]

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
    ) -> None:
        """Initialise the process manager.

        Args:
            ctx (BaseContext): The multiprocessing context to use.
            bridge_data (reGenBridgeData): The bridge data for this worker.
            horde_model_reference_manager (ModelReferenceManager): The model reference manager for this worker.
            target_ram_overhead_bytes (int, optional): The target amount of RAM to keep free. \
                Defaults to 9 * 1024 * 1024 * 1024.
            target_vram_overhead_bytes_map (Mapping[int, int] | None, optional): The target amount of VRAM to keep \
                free. Defaults to None.
            max_safety_processes (int, optional): The maximum number of safety processes that can run at once. \
                Defaults to 1.
            max_download_processes (int, optional): The maximum number of download processes that can run at once. \
                Defaults to 1.
            amd_gpu (bool, optional): Whether or not the GPU is an AMD GPU. Defaults to False.
            directml (int, optional): ID of the potential directml device. Defaults to None.
        """
        self.session_start_time = time.time()

        self.bridge_data = bridge_data
        logger.debug(f"Models to load: {bridge_data.image_models_to_load}")
        logger.debug(f"Custom Models to load: {bridge_data.custom_models}")

        self.horde_model_reference_manager = horde_model_reference_manager

        self._process_map = ProcessMap({})
        self._horde_model_map = HordeModelMap(root={})

        self.max_safety_processes = max_safety_processes
        self.max_download_processes = max_download_processes

        self._max_concurrent_inference_processes = bridge_data.max_threads
        self._inference_semaphore = Semaphore(self._max_concurrent_inference_processes, ctx=ctx)

        self._aux_model_lock = Lock_MultiProcessing(ctx=ctx)

        self.max_inference_processes = self.bridge_data.queue_size + self.bridge_data.max_threads

        vae_decode_semaphore_max = 1

        if self.bridge_data.high_memory_mode:
            vae_decode_semaphore_max = self.max_inference_processes

        self._vae_decode_semaphore = Semaphore(vae_decode_semaphore_max, ctx=ctx)

        self._lru = LRUCache(self.max_inference_processes)

        self._amd_gpu = amd_gpu
        self._directml = directml

        self._replaced_due_to_maintenance = False

        # If there is only one model to load and only one inference process, then we can only run one job at a time
        # and there is no point in having more than one inference process
        if len(self.bridge_data.image_models_to_load) == 1 and self.max_concurrent_inference_processes == 1:
            self.max_inference_processes = 1

        self._disk_lock = Lock_MultiProcessing(ctx=ctx)

        self._job_tracker = JobTracker()

        self.target_vram_overhead_bytes_map = target_vram_overhead_bytes_map  # TODO

        self.total_ram_bytes = psutil.virtual_memory().total

        self.target_ram_overhead_bytes = target_ram_overhead_bytes
        self.target_ram_overhead_bytes = min(int(self.total_ram_bytes / 2), 9)

        if any(model in VRAM_HEAVY_MODELS for model in self.bridge_data.image_models_to_load):
            # If the system ram is less than 24GB, then we're going to exit with an error
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
        if self.bridge_data.remove_maintenance_on_init:
            try:
                self.remove_maintenance()
            except Exception as e:
                logger.warning(e)
                logger.warning("Error trying to unset maintenance. Did this worker not exist yet?")

        # Get the total memory of each GPU
        import torch

        self._device_map = TorchDeviceMap(root={})
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            self._device_map.root[i] = TorchDeviceInfo(
                device_name=device.name,
                device_index=i,
                total_memory=device.total_memory,
            )

        self._process_message_queue = multiprocessing.Queue()

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
            is_shutting_down=lambda: self._shutting_down,
            get_shutting_down_time=lambda: self._shutting_down_time,
        )

        self.kudos_events: deque[tuple[float, float]] = deque()

        self._api_messages_received = {}

        self.stable_diffusion_reference = None

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
        if not self._shutting_down:
            return False

        if self._process_lifecycle.recently_recovered:
            return False

        # If any job hasn't been submitted to the API yet, then we can't shut down
        if len(self._job_tracker.jobs_pending_submit) > 0:
            return False
        # If there are any jobs in progress, then we can't shut down
        if (
            len(self._job_tracker.jobs_being_safety_checked) > 0
            or len(self._job_tracker.jobs_pending_safety_check) > 0
        ):
            return False
        if len(self._job_tracker.jobs_in_progress) > 0:
            return False
        if len(self._job_tracker.jobs_pending_inference) > 0:
            return False
        if len(self._job_tracker.jobs_pending_submit) > 0:
            return False

        if all(
            inference_process.last_process_state == HordeProcessState.PROCESS_ENDING
            or inference_process.last_process_state == HordeProcessState.PROCESS_ENDED
            or inference_process.last_process_state == HordeProcessState.PROCESS_STARTING
            for inference_process in self._process_map.get_inference_processes()
        ):
            return True

        any_process_alive = False

        for process_info in self._process_map.values():
            # The safety process gets shut down last and is part of cleanup
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if (process_info.last_process_state == HordeProcessState.INFERENCE_STARTING) or (
                process_info.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
            ):
                any_process_alive = True
                continue

        # If there are any inference processes still alive, then we can't shut down
        return not any_process_alive

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

        This is the backbone of the inter-process communication system and is the main way that the parent process \
             knows what is going on in the child processes.

        **Note** also that this is a synchronous function and any interaction with objects that are shared between \
            coroutines should be done with care. Critically, this function should be called with locks already \
            acquired on any shared objects.

        See also `._process_map` and `._horde_model_map`, which are updated by this function, and `HordeProcessState` \
            and `ModelLoadState` for the possible states that the processes and models can be in.
        """
        while not self._process_message_queue.empty():
            try:
                message: HordeProcessMessage = self._process_message_queue.get(block=False)
            except queue.Empty:
                logger.debug("Queue was empty, breaking")
                break

            self._in_deadlock = False
            self._in_queue_deadlock = False

            if isinstance(message, HordeProcessHeartbeatMessage):
                self._handle_heartbeat(message)
            else:
                logger.debug(
                    f"Received {type(message).__name__} from process {message.process_id}: {message.info}",
                )

            if not isinstance(message, HordeProcessMessage):
                raise ValueError(f"Received a message that is not a HordeProcessMessage: {message}")
            if message.process_id not in self._process_map:
                raise ValueError(f"Received a message from an unknown process: {message}")

            known_launch_identifier = self._process_map[message.process_id].process_launch_identifier

            if message.process_launch_identifier != known_launch_identifier:
                if self._process_map[message.process_id].last_process_state != HordeProcessState.PROCESS_STARTING:
                    logger.error(
                        f"Received a message from process {message.process_id} with launch identifier "
                        f"{message.process_launch_identifier}, but expected {known_launch_identifier}",
                    )
                    logger.error("This is probably due to a process being replaced. Ignoring.")
                    logger.error(f"Message: {message}")
                else:
                    logger.debug(
                        f"Received a message from process {message.process_id} with launch identifier "
                        f"{message.process_launch_identifier}, but expected {known_launch_identifier}",
                    )
                continue

            if isinstance(message, HordeProcessMemoryMessage):
                self._handle_memory_report(message)
                continue

            if isinstance(message, HordeProcessStateChangeMessage):
                self._handle_process_state_change(message)

            if isinstance(message, HordeAuxModelStateChangeMessage):
                self._handle_aux_model_state_change(message)

            if isinstance(message, HordeModelStateChangeMessage):
                self._handle_model_state_change(message)

            if isinstance(message, HordeInferenceResultMessage):
                self._handle_inference_result(message)
            elif isinstance(message, HordeSafetyResultMessage):
                self._handle_safety_result(message)

    def _handle_heartbeat(self, message: HordeProcessHeartbeatMessage) -> None:
        """Handle a heartbeat message from a child process."""
        self._process_map.on_heartbeat(
            message.process_id,
            heartbeat_type=message.heartbeat_type,
            percent_complete=message.percent_complete,
        )

        in_progress_job_info = self._process_map[message.process_id].last_job_referenced

        if message.process_warning is not None and (
            in_progress_job_info is not None and in_progress_job_info.payload.n_iter < 4
        ):
            logger.warning(f"Process {message.process_id} warning: {message.process_warning}")

            model_name = self._process_map[message.process_id].loaded_horde_model_name
            model_baseline = self.get_model_baseline(model_name) if model_name is not None else None

            if model_baseline is not None:
                logger.warning(f"Model baseline triggering warning: {model_baseline}")

            if in_progress_job_info.payload.n_iter != 1:
                logger.warning(f"Batched job triggering warning: {in_progress_job_info.payload.n_iter} images")
                logger.warning("If you think this is in error, please contact the devs on github or discord.")

    def _handle_memory_report(self, message: HordeProcessMemoryMessage) -> None:
        """Handle a memory usage report from a child process."""
        self._process_map.on_memory_report(
            process_id=message.process_id,
            ram_usage_bytes=message.ram_usage_bytes,
            vram_usage_bytes=message.vram_usage_bytes,
            total_vram_bytes=message.vram_total_bytes,
        )

    def _handle_process_state_change(self, message: HordeProcessStateChangeMessage) -> None:
        """Handle a process state change message."""
        if self._process_map[message.process_id].last_process_state == message.process_state:
            return

        self._process_map.on_process_state_change(
            process_id=message.process_id,
            new_state=message.process_state,
        )

        if message.process_state == HordeProcessState.PROCESS_ENDING:
            logger.info(f"Process {message.process_id} is ending")
            self._process_map.on_process_ending(process_id=message.process_id)

        if message.process_state == HordeProcessState.PROCESS_ENDED:
            logger.info(f"Process {message.process_id} has ended with message: {message.info}")
        else:
            logger.debug(f"Process {message.process_id} changed state to {message.process_state}")

        if message.process_state == HordeProcessState.INFERENCE_STARTING:
            loaded_model_name = self._process_map[message.process_id].loaded_horde_model_name
            if loaded_model_name is None:
                raise ValueError(
                    f"Process {message.process_id} has no model loaded, but is starting inference",
                )
            batch_amount = self._process_map[message.process_id].batch_amount
            if batch_amount is None:
                raise ValueError(
                    f"Process {message.process_id} has batch_amount, but is starting inference",
                )
            self._horde_model_map.update_entry(
                horde_model_name=loaded_model_name,
                load_state=ModelLoadState.IN_USE,
                process_id=message.process_id,
            )

        if (
            message.process_state == HordeProcessState.UNLOADED_MODEL_FROM_RAM
            and self._process_map[message.process_id].last_process_state
            != HordeProcessState.UNLOADED_MODEL_FROM_RAM
        ):
            logger.opt(ansi=True).info(
                f"<fg #7b7d7d>Process {message.process_id} cleared RAM: {message.info}</>",
            )
            self._process_map.on_model_ram_clear(process_id=message.process_id)

    def _handle_aux_model_state_change(self, message: HordeAuxModelStateChangeMessage) -> None:
        """Handle an auxiliary model state change message (e.g., LoRa downloads)."""
        if message.process_state == HordeProcessState.DOWNLOADING_AUX_MODEL:
            logger.opt(ansi=True).info(
                f"<fg #7b7d7d>Process {message.process_id} is downloading extra models (LoRas, etc.)</>",
            )
            self._process_map.on_last_job_reference_change(
                process_id=message.process_id,
                last_job_referenced=message.sdk_api_job_info,
            )

        if message.process_state == HordeProcessState.DOWNLOAD_AUX_COMPLETE:
            logger.opt(ansi=True).info(
                "<fg #7b7d7d>"
                f"Process {message.process_id} finished downloading extra models in {message.time_elapsed}"
                "</>",
            )
            if message.sdk_api_job_info not in self._job_tracker.jobs_lookup:
                if message.sdk_api_job_info is not None:
                    logger.warning(
                        f"Job {message.sdk_api_job_info.id_} not found in jobs_lookup."
                        f" (Process {message.process_id})",
                    )
                else:
                    logger.warning(
                        f"Job not found in jobs_lookup. (Process {message.process_id})",
                    )
                logger.debug(f"Jobs lookup: {self._job_tracker.jobs_lookup}")
            else:
                job_info = self._job_tracker.jobs_lookup[message.sdk_api_job_info]
                job_info.time_to_download_aux_models = message.time_elapsed

    def _handle_model_state_change(self, message: HordeModelStateChangeMessage) -> None:
        """Handle a model state change message."""
        self._horde_model_map.update_entry(
            horde_model_name=message.horde_model_name,
            load_state=message.horde_model_state,
            process_id=message.process_id,
        )

        model_baseline = self.get_model_baseline(message.horde_model_name)

        if message.horde_model_state != ModelLoadState.ON_DISK:
            self._process_map.on_model_load_state_change(
                process_id=message.process_id,
                horde_model_name=message.horde_model_name,
                horde_model_baseline=model_baseline,
            )

            if message.horde_model_state == ModelLoadState.LOADING:
                logger.debug(f"Process {message.process_id} is loading model {message.horde_model_name}")

            if (
                message.horde_model_state == ModelLoadState.LOADED_IN_VRAM
                or message.horde_model_state == ModelLoadState.LOADED_IN_RAM
            ):
                if message.horde_model_state == ModelLoadState.LOADED_IN_VRAM:
                    loaded_message = (
                        f"Process {message.process_id} just finished inference, and has "
                        f"{message.horde_model_name} in VRAM."
                    )
                    logger.debug(loaded_message)
                elif message.horde_model_state == ModelLoadState.LOADED_IN_RAM:
                    loaded_message = (
                        f"Process {message.process_id} moved model {message.horde_model_name} to system RAM. "
                    )

                    if message.time_elapsed is not None:
                        loaded_message += f"Loading took {message.time_elapsed:.2f} seconds."

                    logger.opt(ansi=True).info(f"<fg #7b7d7d>{loaded_message}</>")

        else:
            # FIXME this message is wrong for download processes
            logger.opt(ansi=True).info(
                f"<fg #7b7d7d>Process {message.process_id} unloaded model {message.horde_model_name}</>",
            )

    def _handle_inference_result(self, message: HordeInferenceResultMessage) -> None:
        """Handle an inference job result message."""
        if message.sdk_api_job_info not in self._job_tracker.jobs_lookup:
            logger.error(
                f"Job {message.sdk_api_job_info.id_} not found in jobs_lookup. (Process {message.process_id})",
            )
            if message.sdk_api_job_info in self._job_tracker.jobs_in_progress:
                logger.error(
                    f"Job {message.sdk_api_job_info.id_} found in jobs_in_progress. "
                    f"(Process {message.process_id})",
                )
                self._job_tracker.jobs_in_progress.remove(message.sdk_api_job_info)
            if message.sdk_api_job_info in self._job_tracker.jobs_pending_inference:
                logger.error(
                    f"Job {message.sdk_api_job_info.id_} found in job_deque. (Process {message.process_id})",
                )
                self._job_tracker.jobs_pending_inference.remove(message.sdk_api_job_info)
            return

        job_info = self._job_tracker.jobs_lookup[message.sdk_api_job_info]

        if message.sdk_api_job_info in self._job_tracker.jobs_in_progress:
            self._job_tracker.jobs_in_progress.remove(message.sdk_api_job_info)
        else:
            logger.error(
                f"Job {message.sdk_api_job_info.id_} not found in jobs_in_progress. "
                "Did it fault? "
                f"(Process {message.process_id})",
            )

        for job in self._job_tracker.jobs_pending_inference:
            if job.id_ == message.sdk_api_job_info.id_:
                self._job_tracker.jobs_pending_inference.remove(job)
                break

        self._job_tracker.total_num_completed_jobs += 1
        if self.bridge_data.unload_models_from_vram_often:
            self.unload_models_from_vram(process_with_model=self._process_map[message.process_id])

        if message.time_elapsed is not None:
            inference_finished_string = (
                "\0<fg #da9dff>"
                f"Inference finished for job {str(message.sdk_api_job_info.id_)[:8]} "
                f"<u>({message.sdk_api_job_info.model})</u> on process {message.process_id}. "
                f"It took {round(message.time_elapsed, 2)} seconds, finishing at {message.info} "
                f"and reported {message.faults_count} faults."
                "</>"
            )

            logger.opt(ansi=True).info(inference_finished_string)

        else:
            logger.info(f"Inference finished for job {message.sdk_api_job_info.id_}")
            logger.debug(f"Job didn't include time_elapsed: {message.sdk_api_job_info}")
        if message.state != GENERATION_STATE.faulted:
            job_info.state = message.state
            job_info.time_to_generate = message.time_elapsed
            job_info.job_image_results = message.job_image_results

            self._job_tracker.jobs_pending_safety_check.append(job_info)
        else:
            logger.error(
                f"Job {message.sdk_api_job_info.id_} faulted on process {message.process_id}: {message.info}",
            )

            logger.debug(
                f"Job data: {message.sdk_api_job_info.model_dump(exclude=_excludes_for_job_dump)}",  # type: ignore
            )

            self._job_tracker.jobs_pending_submit.append(job_info)

    def _handle_safety_result(self, message: HordeSafetyResultMessage) -> None:
        """Handle a safety check result message."""
        completed_job_info: HordeJobInfo | None = None
        for i, job_being_safety_checked in enumerate(self._job_tracker.jobs_being_safety_checked):
            if job_being_safety_checked.sdk_api_job_info.id_ == message.job_id:
                completed_job_info = self._job_tracker.jobs_being_safety_checked.pop(i)
                break

        if completed_job_info is None or completed_job_info.job_image_results is None:
            logger.error(
                f"Expected to find a completed job with ID {message.job_id} but none was found. "
                "This should only happen when certain process crashes occur.",
            )
            return

        num_images_censored = 0
        num_images_csam = 0

        any_safety_failed = False

        for i in range(len(completed_job_info.job_image_results)):
            if completed_job_info.sdk_api_job_info.id_ is None:
                continue
            completed_job_info.job_image_results[i].generation_faults += self._job_tracker.job_faults[
                completed_job_info.sdk_api_job_info.id_
            ]
            replacement_image = message.safety_evaluations[i].replacement_image_base64

            if message.safety_evaluations[i].failed:
                logger.error(
                    f"Job {message.job_id} image #{i} faulted during safety checks. "
                    "Check the safety process logs for more information.",
                )
                any_safety_failed = True
                continue

            if replacement_image is not None:
                completed_job_info.job_image_results[i].image_base64 = replacement_image
                num_images_censored += 1
                if message.safety_evaluations[i].is_csam:
                    num_images_csam += 1
        if (
            completed_job_info.sdk_api_job_info.id_ is not None
            and completed_job_info.sdk_api_job_info.id_ in self._job_tracker.job_faults
        ):
            del self._job_tracker.job_faults[completed_job_info.sdk_api_job_info.id_]
        else:
            logger.error(
                f"Job {message.job_id} was not found in job_faults. This is unexpected.",
            )

        logger.debug(
            f"Job {message.job_id} had {num_images_censored} images censored and took "
            f"{message.time_elapsed:.2f} seconds to check safety",
        )

        if any_safety_failed:
            completed_job_info.state = GENERATION_STATE.faulted
        completed_job_info.censored = False
        for i in range(len(completed_job_info.job_image_results)):
            if message.safety_evaluations[i].is_csam:
                new_meta_entry = GenMetadataEntry(
                    type=METADATA_TYPE.censorship,
                    value=METADATA_VALUE.csam,
                )
                completed_job_info.job_image_results[i].generation_faults.append(new_meta_entry)
                completed_job_info.state = GENERATION_STATE.csam
                completed_job_info.censored = True
            elif message.safety_evaluations[i].is_nsfw:
                if message.safety_evaluations[i].replacement_image_base64 is None:
                    new_meta_entry = GenMetadataEntry(
                        type=METADATA_TYPE.information,
                        value=METADATA_VALUE.nsfw,
                    )
                    completed_job_info.job_image_results[i].generation_faults.append(new_meta_entry)
                else:
                    new_meta_entry = GenMetadataEntry(
                        type=METADATA_TYPE.censorship,
                        value=METADATA_VALUE.nsfw,
                    )
                    completed_job_info.job_image_results[i].generation_faults.append(new_meta_entry)
                    completed_job_info.censored = True
                    if completed_job_info.state != GENERATION_STATE.csam:
                        completed_job_info.state = GENERATION_STATE.censored

        self._job_tracker.jobs_pending_submit.append(completed_job_info)

    _preload_delay_notified = False

    def preload_models(self) -> bool:
        """Preload models that are likely to be used soon.

        Returns:
            True if a model was preloaded, False otherwise.
        """
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

        # logger.debug(f"Loaded models: {loaded_models}, queued: {queued_models}")
        # Starting from the left of the deque, preload models that are not yet loaded up to the
        # number of inference processes that are available
        for job in self._job_tracker.jobs_pending_inference:
            if job.model is None:
                raise ValueError(f"job.model is None ({job})")

            if job.model in loaded_models:
                continue

            processes_with_model_for_queued_job: list[int] = (
                self._process_lifecycle.get_processes_with_model_for_queued_job()
            )

            # If the number of still active inference processes is less than the number of jobs in the deque or in
            # progress then we use all processes that are active
            if self._process_map.num_loaded_inference_processes() < (
                len(self._job_tracker.jobs_pending_inference) + len(self._job_tracker.jobs_in_progress)
            ):
                processes_with_model_for_queued_job = [
                    p.process_id for p in self._process_map.values() if p.is_process_busy()
                ]

            available_process = self._process_map.get_first_available_inference_process(
                disallowed_processes=processes_with_model_for_queued_job,
            )

            if available_process is None:
                return False

            if (
                available_process.last_process_state != HordeProcessState.WAITING_FOR_JOB
                and available_process.loaded_horde_model_name is not None
                and self.bridge_data.cycle_process_on_model_change
                and not self._shutting_down
            ):
                # We're going to restart the process and then exit the loop, because
                # available_process is very quickly _not_ going to be available.
                # We also don't want to block waiting for the newly forked job to become
                # available, so we'll wait for it to become ready before scheduling a model
                # to be loaded on it.
                self._process_lifecycle._replace_inference_process(available_process)
                return False

            num_preloading_processes = self._process_map.num_preloading_processes()

            at_least_one_preloading_process = num_preloading_processes >= 1
            very_fast_disk_mode_enabled = self.bridge_data.very_fast_disk_mode
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

            self._preload_delay_notified = False
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

            if available_process.safe_send_message(
                HordePreloadInferenceModelMessage(
                    control_flag=HordeControlFlag.PRELOAD_MODEL,
                    horde_model_name=job.model,
                    will_load_loras=will_load_loras,
                    seamless_tiling_enabled=seamless_tiling_enabled,
                    sdk_api_job_info=job,
                ),
            ):
                available_process.last_control_flag = HordeControlFlag.PRELOAD_MODEL

                self._horde_model_map.update_entry(
                    horde_model_name=job.model,
                    load_state=ModelLoadState.LOADING,
                    process_id=available_process.process_id,
                )

                model_baseline = self.get_model_baseline(job.model)

                self._process_map.on_model_load_state_change(
                    process_id=available_process.process_id,
                    horde_model_name=job.model,
                    horde_model_baseline=model_baseline,
                    last_job_referenced=job,
                )

            # Even if the message fails to send, we still want to return True so that we can let the main loop
            # catch up and potentially replace the process.
            return True

        return False

    _model_recently_missing = False
    _model_recently_missing_time = 0.0

    def get_next_job_and_process(
        self,
        information_only: bool = False,
    ) -> NextJobAndProcess | None:
        """Get the next job and process that can be started, if any.

        Returns:
            NextJobAndProcess if a job can be started, None otherwise.
        """
        if self._job_tracker._skipped_line_next_job_and_process is not None:
            return self._job_tracker._skipped_line_next_job_and_process

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

        processes_post_processing = 0
        if self.post_process_job_overlap_allowed:
            processes_post_processing = self._process_map.num_busy_with_post_processing()

        jobs_in_progress_count = len(self._job_tracker.jobs_in_progress)
        if jobs_in_progress_count >= (self.max_concurrent_inference_processes + processes_post_processing):
            # if self.max_concurrent_inference_processes > 1:
            #     logger.debug(
            #         f"Waiting for {jobs_in_progress_count} jobs to finish "
            #         f"before starting inference for job {next_job.id_}",
            #     )
            return None

        process_with_model = self._process_map.get_process_by_horde_model_name(next_job.model)
        skipped_line = False
        skipped_line_for = None

        def handle_process_missing(job: ImageGenerateJobPopResponse) -> None:
            if self._model_recently_missing:
                # We don't want to spam the logs
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

                    horde_model_baseline = self.get_model_baseline(job.model)

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

                try:
                    self._job_tracker.jobs_in_progress.remove(job)
                except ValueError:
                    logger.debug(f"Job {job.id_} not found in jobs_in_progress.")

        if process_with_model is None:
            if (
                self._preload_delay_notified
                or self._horde_model_map.is_model_loading(next_job.model)
                or information_only
            ):
                return None
            handle_process_missing(next_job)
            return None

        candidate_job_size = 25

        if self.bridge_data.high_performance_mode:
            candidate_job_size = 100

        elif self.bridge_data.moderate_performance_mode:
            candidate_job_size = 50

        if not process_with_model.can_accept_job():
            if (process_with_model.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL) or (
                self.post_process_job_overlap_allowed
                and process_with_model.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
            ):
                # If any of the next n jobs (other than this one) aren't using the same model, see if that job
                # has a model that's already loaded.
                # If it does, we'll start inference on that job instead.
                for candidate_small_job in next_n_jobs:
                    job_has_loras = (
                        candidate_small_job.payload.loras is not None and len(candidate_small_job.payload.loras) > 0
                    )
                    if (
                        candidate_small_job.model is not None
                        and candidate_small_job.model != next_job.model
                        and not job_has_loras
                    ):
                        candidate_process_with_model = self._process_map.get_process_by_horde_model_name(
                            candidate_small_job.model,
                        )
                        if (
                            candidate_process_with_model is not None
                            and self.get_single_job_effective_megapixelsteps(candidate_small_job) <= candidate_job_size
                            and candidate_process_with_model.can_accept_job()
                        ):
                            skipped_line = True
                            skipped_line_for = next_job

                            next_job = candidate_small_job
                            self._skipped_line_job = next_job
                            process_with_model = candidate_process_with_model
                            break
                else:
                    return None
            else:
                return None

        self._model_recently_missing = False

        next_job_and_process = NextJobAndProcess(
            next_job=next_job,
            process_with_model=process_with_model,
            skipped_line=skipped_line,
            skipped_line_for=skipped_line_for,
        )

        if skipped_line:
            self._job_tracker._skipped_line_next_job_and_process = next_job_and_process

        return next_job_and_process

    def start_inference(self) -> bool:
        """Start inference for the next job in jobs_pending_inference, if possible.

        Returns:
            True if inference was started, False otherwise.
        """
        next_job_and_process = self.get_next_job_and_process()

        if next_job_and_process is None:
            return False

        process_with_model = next_job_and_process.process_with_model
        next_job = next_job_and_process.next_job

        if next_job_and_process.skipped_line and next_job_and_process.skipped_line_for is not None:
            logger.info(
                f"Job {next_job_and_process.next_job.id_} skipped the line and will be run on process "
                f"{process_with_model.process_id} before job {next_job_and_process.skipped_line_for.id_}"
                "which is currently downloading extra models.",
            )

        processes_post_processing = 0
        if self.post_process_job_overlap_allowed:
            processes_post_processing = self._process_map.num_busy_with_post_processing()

        if (
            processes_post_processing > 0
            and len(self._job_tracker.jobs_in_progress) >= self.max_concurrent_inference_processes
        ):
            logger.debug(
                "Proceeding with inference, but post processing is still running on "
                f"{processes_post_processing} processes",
            )

        # Unload all models from vram from any other process that isn't running a job if configured to do so
        if self.bridge_data.unload_models_from_vram_often:
            self.unload_models_from_vram(process_with_model)

        color_format_string = "<fg #f0beff>{message}</>"

        logger.opt(ansi=True).info(
            color_format_string.format(
                message=f"Starting inference for job {str(next_job.id_)[:8]} "
                f"on process {process_with_model.process_id}",
            ),
        )

        # region Log job info
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
            # logger.info("  " + extra_info)
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
        # endregion

        # We store the amount of batches this job will do,
        # as we use that later to check if we should start inference in parallel
        process_with_model.batch_amount = next_job.payload.n_iter
        if process_with_model.safe_send_message(
            HordeInferenceControlMessage(
                control_flag=HordeControlFlag.START_INFERENCE,
                horde_model_name=next_job.model,
                sdk_api_job_info=next_job,
            ),
        ):
            self._job_tracker.jobs_in_progress.append(next_job)

            process_with_model.last_control_flag = HordeControlFlag.START_INFERENCE
            process_with_model.last_job_referenced = next_job
            process_with_model.loaded_horde_model_name = next_job.model
            horde_model_baseline = self.get_model_baseline(next_job.model)
            process_with_model.loaded_horde_model_baseline = horde_model_baseline

        else:
            logger.error(
                f"Failed to start inference for job {next_job.id_} on process {process_with_model.process_id}",
            )
            self._job_tracker.handle_job_fault(
                faulted_job=next_job,
                process_info=process_with_model,
                process_timeout=self.bridge_data.process_timeout,
            )

        self._job_tracker._skipped_line_next_job_and_process = None

        return True

    def unload_models_from_vram(
        self,
        process_with_model: HordeProcessInfo,
    ) -> None:
        """Unload models from VRAM from processes that are not running a job.

        Args:
            process_with_model: The process that is running a job.
        """
        next_n_models = list(self.get_next_n_models(self.max_inference_processes))
        logger.debug(f"Next n models: {next_n_models}")
        next_model = None
        if len(next_n_models) > 0:
            next_model = next_n_models.pop()

        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}

        for process_info in self._process_map.values():
            if process_info.process_id == process_with_model.process_id:
                continue

            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if process_info.is_process_busy():
                logger.debug(f"Process {process_info.process_id} is busy")
                # continue

            if process_info.loaded_horde_model_name is not None:
                if len(self.bridge_data.image_models_to_load) == 1:
                    logger.debug("Not unloading models from VRAM because there is only one model to load.")
                    continue

                # If this models is in progress, don't unload it
                if process_info.loaded_horde_model_name in in_progress_models:
                    continue

                # If the model would be used next, don't unload it
                if process_info.loaded_horde_model_name == next_model:
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
            else:
                logger.debug(f"Unloading all models from VRAM on process {process_info.process_id}")
                if (
                    not process_info.safe_send_message(
                        HordeControlMessage(
                            control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
                        ),
                    )
                    and not self._shutting_down
                ):
                    self._process_lifecycle._replace_inference_process(process_info)

    def unload_from_ram(self, process_id: int) -> None:
        """Unload models from a process.

        Args:
            process_id: The process to unload models from.
        """
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
            # Check the process is not ending
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
        """Get the next n models that will be used in the job deque.

        Args:
            n: The number of models to get.

        Returns:
            A list of the next n models that will be used in the job deque.
        """
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

    def unload_models(self) -> bool:
        """Unload models that are no longer needed and would use above the limit specified.

        Returns:
            True if a model was unloaded, False otherwise.
        """
        if len(self._job_tracker.jobs_pending_inference) == 0:
            return False

        # 1 thread, 1 model, no need to unload as it should always be in use (or at least available)
        if self._max_concurrent_inference_processes == 1 and len(self.bridge_data.image_models_to_load) == 1:
            return False

        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if process_info.is_process_busy() or process_info.last_process_state == HordeProcessState.PRELOADED_MODEL:
                continue

            if process_info.loaded_horde_model_name is not None:
                if self._horde_model_map.is_model_loading(process_info.loaded_horde_model_name):
                    continue

                if (
                    self._horde_model_map.root[process_info.loaded_horde_model_name].horde_model_load_state
                    == ModelLoadState.IN_USE
                ):
                    continue

                if (
                    process_info.loaded_horde_model_name in self._job_tracker.jobs_pending_inference
                    or process_info.loaded_horde_model_name in self._job_tracker.jobs_in_progress
                ):
                    continue

                self.unload_from_ram(process_info.process_id)
                return True

        return False

    def start_evaluate_safety(self) -> None:
        """Start evaluating the safety of the next job pending a safety check, if any."""
        if len(self._job_tracker.jobs_pending_safety_check) == 0:
            return

        safety_process = self._process_map.get_first_available_safety_process()

        if safety_process is None:
            return

        completed_job_info = self._job_tracker.jobs_pending_safety_check[0]

        if self.stable_diffusion_reference is None:
            raise ValueError("stable_diffusion_reference is None")

        critical_fault = False

        if completed_job_info.job_image_results is None:
            logger.error("completed_job_info.job_image_results is None")
            critical_fault = True

        if completed_job_info.sdk_api_job_info.id_ is None:
            logger.error("completed_job_info.sdk_api_job_info.id_ is None")
            critical_fault = True

        if completed_job_info.sdk_api_job_info.model is None:
            logger.error("completed_job_info.sdk_api_job_info.model is None")
            critical_fault = True

        if completed_job_info.sdk_api_job_info.payload.prompt is None:
            logger.error("completed_job_info.sdk_api_job_info.payload.prompt is None")
            critical_fault = True

        if critical_fault:
            self._job_tracker.handle_job_fault(
                faulted_job=completed_job_info.sdk_api_job_info,
                process_info=safety_process,
                process_timeout=self.bridge_data.process_timeout,
            )
            logger.error(f"Failed to start safety evaluation for job {completed_job_info.sdk_api_job_info.id_}")
            self._job_tracker.jobs_pending_safety_check.remove(completed_job_info)

            return

        # Duplicated for static type checking
        if completed_job_info.sdk_api_job_info.id_ is None:
            raise ValueError("completed_job_info.sdk_api_job_info.id_ is None")
        if completed_job_info.sdk_api_job_info.payload.prompt is None:
            raise ValueError("completed_job_info.sdk_api_job_info.payload.prompt is None")
        if completed_job_info.sdk_api_job_info.model is None:
            raise ValueError("completed_job_info.sdk_api_job_info.model is None")

        # Custom models don't appear in the downloaded model reference
        model_info = {}
        if completed_job_info.sdk_api_job_info.model in self.stable_diffusion_reference.root:
            model_info = self.stable_diffusion_reference.root[completed_job_info.sdk_api_job_info.model].model_dump()
        safety_message_sent_succeeded = safety_process.safe_send_message(
            HordeSafetyControlMessage(
                control_flag=HordeControlFlag.EVALUATE_SAFETY,
                job_id=completed_job_info.sdk_api_job_info.id_,
                images_base64=completed_job_info.images_base64,
                prompt=completed_job_info.sdk_api_job_info.payload.prompt,
                censor_nsfw=completed_job_info.sdk_api_job_info.payload.use_nsfw_censor,
                sfw_worker=not self.bridge_data.nsfw,
                horde_model_info=model_info,
                # TODO: update this to use a class instead of a dict?
            ),
        )

        safety_process = self._process_map.get_safety_process()
        if not safety_message_sent_succeeded:
            if safety_process is None:
                return

            if (
                not safety_process.is_process_alive()
                or safety_process.last_process_state == HordeProcessState.PROCESS_STARTING
            ):
                return

            logger.error(f"Failed to start safety evaluation for job {completed_job_info.sdk_api_job_info.id_}")
            self._process_lifecycle.safety_processes_should_be_replaced = True
            if len(self._job_tracker.jobs_being_safety_checked) > 0:
                for job_info in self._job_tracker.jobs_being_safety_checked:
                    self._job_tracker.jobs_pending_safety_check.append(job_info)
        else:
            self._job_tracker.jobs_pending_safety_check.remove(completed_job_info)
            self._job_tracker.jobs_being_safety_checked.append(completed_job_info)

    def base64_image_to_stream_buffer(self, image_base64: str) -> BytesIO | None:
        """Convert a base64 image to a BytesIO stream buffer.

        Args:
            image_base64: The base64 image to convert.

        Returns:
            A BytesIO stream buffer containing the image, or None if the conversion failed.
        """
        return _base64_image_to_stream_buffer(image_base64)

    _num_job_slowdowns = 0
    """The number of jobs which did not meet the minimum expected kudos/second rate."""

    @logger.catch(reraise=True)
    async def submit_single_generation(self, new_submit: PendingSubmitJob) -> PendingSubmitJob:
        """Tries to upload and submit a single image from a batch.

        Args:
            new_submit: The job to attempt to submit.

        Returns:
            The modified in place job with the results of the submission attempt.
        """
        logger.debug(f"Preparing to submit job {new_submit.job_id}")

        if new_submit.image_result is None and not new_submit.is_faulted:
            logger.error(f"Job {new_submit.job_id} has no image result")
            new_submit.fault()
            return new_submit

        if new_submit.image_result is not None:
            image_in_buffer = self.base64_image_to_stream_buffer(
                new_submit.image_result.image_base64,
            )
            if image_in_buffer is None:
                logger.critical(
                    f"There is an invalid image in the job results for {new_submit.job_id}, "
                    "removing from completed jobs",
                )
                for (
                    follow_up_request
                ) in new_submit.completed_job_info.sdk_api_job_info.get_follow_up_failure_cleanup_request():
                    follow_up_response = await self.horde_client_session.submit_request(
                        follow_up_request,
                        JobSubmitResponse,
                    )

                    if isinstance(follow_up_response, RequestErrorResponse):
                        logger.error(f"Failed to submit followup request: {follow_up_response}")
                new_submit.fault()
                return new_submit

            async def _do_upload(new_submit: PendingSubmitJob, image_in_buffer_bytes: bytes) -> bool:
                async with self._aiohttp_client_session.put(
                    yarl.URL(new_submit.r2_upload, encoded=True),
                    data=image_in_buffer_bytes,
                    skip_auto_headers=["content-type"],
                    timeout=aiohttp.ClientTimeout(total=10),
                    ssl=sslcontext,
                ) as response:
                    if response.status == 500:
                        logger.warning(
                            "Retrying upload to R2. This is a cloudflare issue and only is a concern if "
                            "you see this message 5 or more times a minute.",
                        )
                        new_submit.retry()
                        return False
                    if response.status != 200:
                        logger.error(f"Failed to upload image to R2: {response}")
                        new_submit.retry()
                        return False
                return True

            try:
                submit_success = await asyncio.wait_for(
                    _do_upload(new_submit, image_in_buffer.getvalue()),
                    timeout=10 + 1,
                )
                if not submit_success:
                    return new_submit
            except _async_client_exceptions as e:
                logger.warning("Upload to AI Horde R2 timed out. Will retry.")
                logger.debug(f"{type(e).__name__}: {e}")
                new_submit.retry()
                return new_submit
            except Exception as e:
                logger.error(f"Failed to upload image to R2: {e}")
                logger.debug(f"{type(e).__name__}: {e}")
                new_submit.retry()
                return new_submit
        metadata = []
        if new_submit.image_result is not None:
            metadata = new_submit.image_result.generation_faults
            if new_submit.batch_count > 1:
                metadata.append(
                    GenMetadataEntry(
                        type=METADATA_TYPE.batch_index,
                        value=METADATA_VALUE.see_ref,
                        ref=str(new_submit.gen_iter),
                    ),
                )
        seed = 0
        if new_submit.completed_job_info.sdk_api_job_info.payload.seed is not None:
            seed = int(new_submit.completed_job_info.sdk_api_job_info.payload.seed)
        submit_job_request_type = new_submit.completed_job_info.sdk_api_job_info.get_follow_up_default_request_type()
        if new_submit.completed_job_info.state is None:
            logger.error(f"Job {new_submit.job_id} has no state, assuming faulted")
            new_submit.completed_job_info.state = GENERATION_STATE.faulted
            return new_submit
        submit_job_request = submit_job_request_type(
            apikey=self.bridge_data.api_key,
            id=new_submit.job_id,
            seed=seed,
            generation="R2",  # TODO # FIXME
            state=new_submit.completed_job_info.state,
            censored=bool(new_submit.completed_job_info.censored),  # TODO: is this cast problematic?
            gen_metadata=metadata,
        )
        logger.debug(f"Submitting job {new_submit.job_id}")
        job_submit_response = None
        try:
            job_submit_response = await asyncio.wait_for(
                self.horde_client_session.submit_request(
                    submit_job_request,
                    JobSubmitResponse,
                ),
                timeout=10 + 1,
            )
        except _async_client_exceptions:
            logger.error(f"Job {new_submit.job_id} submission timed out")
            new_submit.retry()
            return new_submit
        except Exception as e:
            logger.error(f"Failed to submit job {new_submit.job_id}: {e}")
            new_submit.retry()
            return new_submit

        # If the job submit response is an error,
        # log it and increment the number of consecutive failed job submits
        if isinstance(job_submit_response, RequestErrorResponse):
            if (
                "Processing Job with ID" in job_submit_response.message
                and "does not exist" in job_submit_response.message
            ):
                logger.warning(f"Job {new_submit.job_id} does not exist, removing from completed jobs")
                new_submit.fault()
                return new_submit

            if "already submitted" in job_submit_response.message:
                logger.debug(
                    f"Job {new_submit.job_id} has already been submitted, removing from completed jobs",
                )
                new_submit.fault()
                return new_submit

            if "Please check your worker speed" in job_submit_response.message:
                logger.error(job_submit_response.message)
                new_submit.fault()
                return new_submit

            error_string = (
                f"Failed to submit job (API Error) {new_submit.retry_attempts_string}: {job_submit_response}"
            )
            logger.error(error_string)
            new_submit.retry()
            return new_submit

        if job_submit_response is None:
            logger.error(f"Failed to submit job {new_submit.job_id}")
            new_submit.retry()
            return new_submit

        # Get the time the job was popped from the job deque
        async with self._job_tracker.pop_timestamps_lock:
            time_popped = self._job_tracker.job_pop_timestamps.get(new_submit.completed_job_info.sdk_api_job_info)
            if time_popped is None:
                logger.warning(
                    f"Failed to get time_popped for job {new_submit.completed_job_info.sdk_api_job_info.id_}. "
                    "This is likely a bug.",
                )
                time_popped = time.time()

            elif time_popped == -1:
                logger.warning(
                    f"Job {new_submit.completed_job_info.sdk_api_job_info.id_} will have an incorrect kudos/second "
                    "calculation.",
                )
                time_popped = time.time()

        time_taken = round(time.time() - time_popped, 2)

        kudos_per_second = 0.0

        if new_submit.completed_job_info.time_to_generate is None:
            logger.error(
                f"Job {new_submit.job_id} has no time_to_generate, ignoring.",
            )
            new_submit.completed_job_info.time_to_generate = 0.0
        else:
            kudos_per_second = job_submit_response.reward / new_submit.completed_job_info.time_to_generate

        # If the job was not faulted, log the job submission as a success
        if new_submit.completed_job_info.state != GENERATION_STATE.faulted:
            logger.opt(ansi=True).success(
                f"Submitted generation {str(new_submit.job_id)[:8]} (model: "
                f"<u>{new_submit.completed_job_info.sdk_api_job_info.model})</u> "
                f"for {job_submit_response.reward:,.2f} "
                f"kudos. Job popped {time_taken} seconds ago "
                f"and took {new_submit.completed_job_info.time_to_generate:.2f} "
                f"to generate. ({kudos_per_second * new_submit.batch_count:.2f} "
                "kudos/second for the whole batch. 0.4 or greater is ideal)",
            )
            # If slower than 0.4 kudos per second, log a warning
            if (kudos_per_second * new_submit.batch_count) < 0.4:
                logger.warning(
                    f"Job {new_submit.job_id} took longer than is ideal; if this persists consider "
                    "lowering your max_power, using less threads, disabling post processing and/or controlnets.",
                )
                logger.warning("Be sure your models are on an SSD. Freeing up RAM or VRAM may also help.")
                self._num_job_slowdowns += 1
        # If the job was faulted, log an error
        else:
            logger.error(
                f"{new_submit.job_id} faulted. Reported fault to the horde. "
                f"Job popped {time_taken} seconds ago and took "
                f"{new_submit.completed_job_info.time_to_generate:.2f} to generate.",
            )
            self._job_tracker._num_jobs_faulted += 1

        self.kudos_generated_this_session += job_submit_response.reward
        self.kudos_events.append((time.time(), job_submit_response.reward))
        new_submit.succeed(new_submit.kudos_reward, new_submit.kudos_per_second)
        return new_submit

    @logger.catch(reraise=True)
    async def api_submit_job(self) -> None:
        """Submit a job result to the API, if any are completed (safety checked too) and ready to be submitted."""
        if len(self._job_tracker.jobs_pending_submit) == 0:
            return

        completed_job_info = self._job_tracker.jobs_pending_submit[0]
        job_info = completed_job_info.sdk_api_job_info

        if completed_job_info.state is None:
            logger.error(f"Job {job_info.ids} has no state, assuming faulted")
            completed_job_info.state = GENERATION_STATE.faulted

        if completed_job_info.state == GENERATION_STATE.faulted:
            logger.error(
                f"Job {job_info.ids} faulted, removing from completed jobs after submitting the faults to the horde",
            )
            self._consecutive_failed_jobs += 1

        if completed_job_info.job_image_results is not None:
            if len(completed_job_info.job_image_results) != completed_job_info.sdk_api_job_info.payload.n_iter:
                logger.warning(
                    f"Needed to generate {completed_job_info.sdk_api_job_info.payload.n_iter} images "
                    f"but only {len(completed_job_info.job_image_results)} returned by the inference process "
                    "We will continue, but you might get put into maintenance if this keeps happening.",
                )
            elif len(completed_job_info.job_image_results) > 1:
                logger.info("Attempting to return batched jobs results")

            if completed_job_info.censored is None:
                raise ValueError("completed_job_info.censored is None")
        if job_info.id_ is None:
            raise ValueError("job_info.id_ is None")

        if job_info.payload.seed is None:
            raise ValueError("job_info.payload.seed is None")

        if job_info.r2_upload is None:  # TODO: r2_upload should be being set somewhere
            raise ValueError("job_info.r2_upload is None")

        highest_reward = 0
        highest_kudos_per_second = 0.0
        submit_tasks: list[Task[PendingSubmitJob]] = []
        finished_submit_jobs: list[PendingSubmitJob] = []
        iterations = 1
        job_faulted = False
        if completed_job_info.job_image_results is not None:
            iterations = len(completed_job_info.job_image_results)
        for gen_iter in range(iterations):
            new_submit = PendingSubmitJob(completed_job_info=completed_job_info, gen_iter=gen_iter)
            submit_tasks.append(asyncio.create_task(self.submit_single_generation(new_submit)))
        while len(submit_tasks) > 0:
            retry_submits: list[PendingSubmitJob] = []
            results = await asyncio.gather(*submit_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.exception(f"Exception in job submit task: {result}")
                    job_faulted = True
                elif isinstance(result, PendingSubmitJob):
                    if not result.is_finished:
                        retry_submits.append(result)
                    else:
                        finished_submit_jobs.append(result)
                    if highest_reward < result.kudos_reward:
                        highest_reward = result.kudos_reward
                    if highest_kudos_per_second < result.kudos_per_second:
                        highest_kudos_per_second = result.kudos_per_second
            submit_tasks = []
            for retry_submit in retry_submits:
                submit_tasks.append(asyncio.create_task(self.submit_single_generation(retry_submit)))

        # Get the time the job was popped from the job deque
        async with self._job_tracker.pop_timestamps_lock:
            time_popped = self._job_tracker.job_pop_timestamps.get(completed_job_info.sdk_api_job_info)
            if time_popped is None:
                logger.warning(
                    f"Failed to get time_popped for job {completed_job_info.sdk_api_job_info.id_}. "
                    "This is likely a bug.",
                )
                time_popped = time.time()
        time_taken = round(time.time() - time_popped, 2)
        # If the job took a long time to generate, log a warning (unless speed warnings are suppressed)
        if not self.bridge_data.suppress_speed_warnings:
            if highest_reward > 0 and (highest_reward / time_taken) < 0.1:
                logger.warning(
                    f"This job ({completed_job_info.sdk_api_job_info.id_}) "
                    "may have been in the queue for a long time. ",
                )

            if highest_reward > 0 and highest_kudos_per_second < 0.4:
                logger.warning(
                    f"This job ({completed_job_info.sdk_api_job_info.id_}) "
                    "took longer than is ideal; if this persists consider "
                    "lowering your max_power, using less threads, "
                    "disabling post processing and/or controlnets.",
                )

        # Finally, remove the job from the completed jobs list and reset the number of consecutive failed job
        async with self._job_tracker.lookup_lock, self._job_tracker.completed_jobs_lock:
            for submit_job in finished_submit_jobs:
                if submit_job.is_faulted:
                    job_faulted = True
                    self._consecutive_failed_jobs += 1
                    break
            if not job_faulted:
                # If any of the submits failed, we consider the whole job failed
                self._consecutive_failed_jobs = 0
            try:
                if completed_job_info.sdk_api_job_info in self._job_tracker.jobs_lookup:
                    self._job_tracker.jobs_lookup[completed_job_info.sdk_api_job_info].time_submitted = time.time()
                else:
                    self._job_tracker.jobs_lookup[completed_job_info.sdk_api_job_info] = HordeJobInfo(
                        sdk_api_job_info=completed_job_info.sdk_api_job_info,
                        time_popped=-1,
                        job_image_results=completed_job_info.job_image_results,
                        state=completed_job_info.state,
                        censored=completed_job_info.censored,
                        time_to_generate=completed_job_info.time_to_generate,
                        time_to_download_aux_models=completed_job_info.time_to_download_aux_models,
                    )
                    logger.error(
                        f"Job {completed_job_info.sdk_api_job_info.id_} not found in jobs_lookup "
                        "during submit. Creating a new HordeJobInfo object.",
                    )

                # Record kudos training data if enabled
                if self.bridge_data.capture_kudos_training_data:
                    recorder = KudosTrainingRecorder(
                        training_data_file=self.bridge_data.kudos_training_data_file,
                        stable_diffusion_reference=self.stable_diffusion_reference,
                    )

                    if completed_job_info.sdk_api_job_info in self._job_tracker.jobs_lookup:
                        recorder.record_job_data(self._job_tracker.jobs_lookup[completed_job_info.sdk_api_job_info])
                    else:
                        logger.error(
                            f"Job {completed_job_info.sdk_api_job_info.id_} not found in jobs_lookup "
                            "during kudos training data capture.",
                        )

                if completed_job_info in self._job_tracker.jobs_pending_submit:
                    self._job_tracker.jobs_pending_submit.remove(completed_job_info)
                else:
                    logger.warning(f"Job {completed_job_info.sdk_api_job_info.id_} not found in completed_jobs")

                if completed_job_info.sdk_api_job_info in self._job_tracker.jobs_lookup:
                    del self._job_tracker.jobs_lookup[completed_job_info.sdk_api_job_info]
                else:
                    logger.warning(f"Job {completed_job_info.sdk_api_job_info.id_} not found in jobs_lookup")

                self._job_tracker._last_job_submitted_time = time.time()

            except ValueError:
                # This means another fault catch removed the faulted job so it's OK
                # But we post a log anyway, just in case
                logger.debug(
                    f"Tried to remove completed_job_info "
                    f"{completed_job_info.sdk_api_job_info.id_} but it has already been removed.",
                )

            if completed_job_info.sdk_api_job_info in self._job_tracker.job_pop_timestamps:
                self._job_tracker.job_pop_timestamps.pop(completed_job_info.sdk_api_job_info)
                logger.debug(f"Removed {completed_job_info.sdk_api_job_info.id_} from job_pop_timestamps")

            if completed_job_info.sdk_api_job_info in self._job_tracker.jobs_in_progress:
                self._job_tracker.jobs_in_progress.remove(completed_job_info.sdk_api_job_info)
                logger.debug(f"Removed {completed_job_info.sdk_api_job_info.id_} from jobs_in_progress")

            if completed_job_info.sdk_api_job_info in self._job_tracker.jobs_lookup:
                self._job_tracker.jobs_lookup.pop(completed_job_info.sdk_api_job_info)
                logger.debug(f"Removed {completed_job_info.sdk_api_job_info.id_} from jobs_lookup")

    # _testing_max_jobs = 10000
    # _testing_jobs_added = 0
    # _testing_job_queue_length = 1

    _default_job_pop_frequency = 1.0
    """The default frequency at which to pop jobs from the API."""
    _error_job_pop_frequency = 5.0
    """The frequency at which to pop jobs from the API when an error occurs."""
    _job_pop_frequency = 1.0
    """The frequency at which to pop jobs from the API. Can be altered if an error occurs."""
    _last_job_pop_time = 0.0
    """The time at which the last job was popped from the API."""

    def _last_pop_recently(self) -> bool:
        return (time.time() - self._last_job_pop_time) < 10

    _batch_wait_log_time = 0.0
    """The last time we informed that we're waiting for batched jobs to finish."""

    _consecutive_failed_jobs = 0

    def get_single_job_effective_megapixelsteps(self, job: ImageGenerateJobPopResponse) -> int:
        """Return the number of megapixelsteps for a single job.

        Args:
            job: The job to get the number of megapixelsteps for.

        Returns:
            The number of effective megapixelsteps for the job.
        """
        return _get_single_job_effective_megapixelsteps(job)

    async def _get_source_images(self, job_pop_response: ImageGenerateJobPopResponse) -> ImageGenerateJobPopResponse:
        # Adding this to stop mypy complaining
        if job_pop_response.id_ is None:
            logger.error("Received ImageGenerateJobPopResponse with id_ is None. Please let the devs know!")
            return job_pop_response

        download_tasks: list[Task] = []

        source_image_is_url = False
        if job_pop_response.source_image is not None and job_pop_response.source_image.startswith("http"):
            source_image_is_url = True
            logger.debug(f"Source image for job {job_pop_response.id_} is a URL")

        source_mask_is_url = False
        if job_pop_response.source_mask is not None and job_pop_response.source_mask.startswith("http"):
            source_mask_is_url = True
            logger.debug(f"Source mask for job {job_pop_response.id_} is a URL")

        any_extra_source_images_are_urls = False
        if job_pop_response.extra_source_images is not None:
            for extra_source_image in job_pop_response.extra_source_images:
                if extra_source_image.image.startswith("http"):
                    any_extra_source_images_are_urls = True
                    logger.debug(f"Extra source image for job {job_pop_response.id_} is a URL")

        attempts = 0
        while attempts < MAX_SOURCE_IMAGE_RETRIES:
            if (
                source_image_is_url
                and job_pop_response.source_image is not None
                and job_pop_response.get_downloaded_source_image() is None
            ):
                download_tasks.append(job_pop_response.async_download_source_image(self._aiohttp_client_session))
            if (
                source_mask_is_url
                and job_pop_response.source_mask is not None
                and job_pop_response.get_downloaded_source_mask() is None
            ):
                download_tasks.append(job_pop_response.async_download_source_mask(self._aiohttp_client_session))

            download_extra_source_images = job_pop_response.get_downloaded_extra_source_images()
            if (
                any_extra_source_images_are_urls
                and job_pop_response.extra_source_images is not None
                or (
                    download_extra_source_images is not None
                    and job_pop_response.extra_source_images is not None
                    and len(download_extra_source_images) != len(job_pop_response.extra_source_images)
                )
            ):
                download_tasks.append(
                    asyncio.create_task(
                        job_pop_response.async_download_extra_source_images(
                            self._aiohttp_client_session,
                            max_retries=MAX_SOURCE_IMAGE_RETRIES,
                        ),
                    ),
                )

            gather_results = await asyncio.gather(*download_tasks, return_exceptions=True)

            for result in gather_results:
                if isinstance(result, Exception):
                    logger.error(f"Failed to download source image: {result}")
                    attempts += 1
                    break
            else:
                break

        if attempts >= MAX_SOURCE_IMAGE_RETRIES:
            if source_image_is_url and job_pop_response.get_downloaded_source_image() is None:
                if self._job_tracker.job_faults.get(job_pop_response.id_) is None:
                    self._job_tracker.job_faults[job_pop_response.id_] = []

                logger.error(f"Failed to download source image for job {job_pop_response.id_}")
                self._job_tracker.job_faults[job_pop_response.id_].append(
                    GenMetadataEntry(
                        type=METADATA_TYPE.source_image,
                        value=METADATA_VALUE.download_failed,
                        ref="source_image",
                    ),
                )

            if source_mask_is_url and job_pop_response.get_downloaded_source_mask() is None:
                if self._job_tracker.job_faults.get(job_pop_response.id_) is None:
                    self._job_tracker.job_faults[job_pop_response.id_] = []
                logger.error(f"Failed to download source mask for job {job_pop_response.id_}")

                self._job_tracker.job_faults[job_pop_response.id_].append(
                    GenMetadataEntry(
                        type=METADATA_TYPE.source_mask,
                        value=METADATA_VALUE.download_failed,
                        ref="source_mask",
                    ),
                )
            downloaded_extra_source_images = job_pop_response.get_downloaded_extra_source_images()
            if (
                any_extra_source_images_are_urls
                and downloaded_extra_source_images is None
                or (
                    downloaded_extra_source_images is not None
                    and job_pop_response.extra_source_images is not None
                    and len(downloaded_extra_source_images) != len(job_pop_response.extra_source_images)
                )
            ):
                if self._job_tracker.job_faults.get(job_pop_response.id_) is None:
                    self._job_tracker.job_faults[job_pop_response.id_] = []
                logger.error(f"Failed to download extra source images for job {job_pop_response.id_}")

                ref = []
                if job_pop_response.extra_source_images is not None and downloaded_extra_source_images is not None:
                    for predownload_extra_source_image in job_pop_response.extra_source_images:
                        if predownload_extra_source_image.image.startswith("http"):
                            if any(
                                predownload_extra_source_image.original_url == extra_source_image.image
                                for extra_source_image in downloaded_extra_source_images
                            ):
                                continue

                            ref.append(str(job_pop_response.extra_source_images.index(predownload_extra_source_image)))
                elif job_pop_response.extra_source_images is not None and downloaded_extra_source_images is None:
                    ref = [str(i) for i in range(len(job_pop_response.extra_source_images))]

                for r in ref:
                    self._job_tracker.job_faults[job_pop_response.id_].append(
                        GenMetadataEntry(
                            type=METADATA_TYPE.extra_source_images,
                            value=METADATA_VALUE.download_failed,
                            ref=r,
                        ),
                    )

        return job_pop_response

    _last_pop_maintenance_mode: bool = False
    """Whether the last job pop showed the worker was in maintenance mode."""
    _last_pop_no_jobs_available: bool = False
    """Whether the last job pop attempt had a no jobs available response."""
    _last_pop_no_jobs_available_time: float = 0.0
    """The time at which the last job pop attempt had a no jobs available response."""
    _time_spent_no_jobs_available: float = 0.0
    """The number of seconds spent with no jobs popped or available."""
    _max_time_spent_no_jobs_available: float = 60.0 * 60.0
    """The maximum number of seconds to spend with no jobs popped or available before warning the user."""
    _too_many_consecutive_failed_jobs: bool = False
    """Whether too many consecutive failed jobs have occurred and job pops are paused."""
    _too_many_consecutive_failed_jobs_time: float = 0.0
    """The time at which too many consecutive failed jobs occurred."""
    _too_many_consecutive_failed_jobs_wait_time = 180
    """The time to wait after too many consecutive failed jobs before resuming job pops."""

    def print_maint_mode_messages(self) -> None:
        """Print the information about maintenance mode to the user."""
        MaintenanceModeMessenger.print_maintenance_mode_messages()

    @logger.catch(reraise=True)
    async def api_job_pop(self) -> None:
        """If the job deque is not full, add any jobs that are available to the job deque."""
        if self._shutting_down:
            self._last_pop_no_jobs_available = False
            return

        cur_time = time.time()

        if self._too_many_consecutive_failed_jobs:
            if (
                cur_time - self._too_many_consecutive_failed_jobs_time
                > self._too_many_consecutive_failed_jobs_wait_time
            ):
                self._consecutive_failed_jobs = 0
                self._too_many_consecutive_failed_jobs = False
                logger.debug("Resuming job pops after too many consecutive failed jobs")
            return

        if self._consecutive_failed_jobs >= 3:
            logger.error(
                "Too many consecutive failed jobs, pausing job pops. "
                "Please look into what happened and let the devs know. ",
                f"Waiting {self._too_many_consecutive_failed_jobs_wait_time} seconds...",
            )
            if self.bridge_data.exit_on_unhandled_faults:
                logger.error("Exiting due to exit_on_unhandled_faults being enabled")
                self._shutdown()
            self._too_many_consecutive_failed_jobs = True
            self._too_many_consecutive_failed_jobs_time = cur_time
            return

        max_jobs_in_queue = self.bridge_data.queue_size + 1

        if self.bridge_data.max_threads > 1:
            max_jobs_in_queue += self.bridge_data.max_threads - 1

        if len(self._job_tracker.jobs_pending_inference) >= max_jobs_in_queue:
            return

        # We let the first job run through to make sure things are working
        # (if we're doomed to fail with 1 job, we're doomed to fail with 2 jobs)
        if len(self._job_tracker.jobs_pending_inference) != 0 and self._job_tracker.jobs_pending_submit == 0:
            return

        # if self._testing_jobs_added >= self._testing_max_jobs:
        #   return

        # Don't start jobs if we can't evaluate safety (NSFW/CSAM)
        if self._process_map.get_first_available_safety_process() is None:
            return

        # Don't start jobs if we can't run inference
        if self._process_map.get_first_available_inference_process() is None:
            return

        if len(self.bridge_data.image_models_to_load) == 0:
            logger.error("No models are configured to be loaded, please check your config (models_to_load).")
            await asyncio.sleep(3)
            return

        # If there are long running jobs, don't start any more even if there is space in the deque
        if self._job_tracker.should_wait_for_pending_megapixelsteps():
            if self._job_tracker.get_pending_megapixelsteps() < 40:
                seconds_to_wait = self._job_tracker.get_pending_megapixelsteps() * 0.5
            elif self._job_tracker.get_pending_megapixelsteps() < 80:
                seconds_to_wait = self._job_tracker.get_pending_megapixelsteps() * 0.7
            else:
                seconds_to_wait = self._job_tracker.get_pending_megapixelsteps() * 0.8

            if self.bridge_data.max_threads > 1:
                seconds_to_wait *= 0.75

            if self.bridge_data.high_performance_mode:
                seconds_to_wait *= 0.2
                if seconds_to_wait < 35:
                    seconds_to_wait = 1
            elif self.bridge_data.moderate_performance_mode:
                seconds_to_wait *= 0.4
                if seconds_to_wait < 20:
                    seconds_to_wait = 1

            if self._job_tracker._triggered_max_pending_megapixelsteps is False:
                self._job_tracker._triggered_max_pending_megapixelsteps = True
                self._job_tracker._triggered_max_pending_megapixelsteps_time = time.time()
                if seconds_to_wait > 2:
                    logger.opt(ansi=True).info(
                        f"<fg #7dcea0><i>Pausing job pops for {round(seconds_to_wait, 2)} seconds "
                        "so some long running jobs can make some progress.</i></>",
                    )
                logger.debug(
                    "Paused job pops for pending megapixelsteps to decrease below "
                    f"{self._job_tracker._max_pending_megapixelsteps}",
                )
                logger.debug(
                    f"Pending megapixelsteps: {self._job_tracker.get_pending_megapixelsteps()} | "
                    f"Max pending megapixelsteps: {self._job_tracker._max_pending_megapixelsteps} | "
                    f"Scheduled to wait for {seconds_to_wait} seconds",
                )
                logger.debug(
                    f"high_performance_mode: {self.bridge_data.high_performance_mode} | "
                    f"moderate_performance_mode: {self.bridge_data.moderate_performance_mode}",
                )
                return

            if not (time.time() - self._job_tracker._triggered_max_pending_megapixelsteps_time) > seconds_to_wait:
                return

            self._job_tracker._triggered_max_pending_megapixelsteps = False
            logger.debug(
                "Pending megapixelsteps decreased below "
                f"{self._job_tracker._max_pending_megapixelsteps}, continuing with job pops",
            )

        self._job_tracker._triggered_max_pending_megapixelsteps = False

        # We don't want to pop jobs too frequently, so we wait a bit between each pop
        if time.time() - self._last_job_pop_time < self._job_pop_frequency:
            return

        self._last_job_pop_time = time.time()

        # dummy_jobs = get_n_dummy_jobs(1)
        # async with self._job_deque_lock:
        #     self.job_deque.extend(dummy_jobs)
        # logger.debug(f"Added {len(dummy_jobs)} dummy jobs to the job deque")
        # # log a list of the current model names in the deque
        # logger.debug(f"Current models in job deque: {[job.model for job in self.job_deque]}")

        models = set(self.bridge_data.image_models_to_load)

        loaded_models = {
            process.loaded_horde_model_name
            for process in self._process_map.values()
            if process.loaded_horde_model_name is not None
        }

        if (
            len(self.bridge_data.image_models_to_load) > self.max_inference_processes
            and len(loaded_models) == self.max_inference_processes
        ):
            if (
                (not self._last_pop_no_jobs_available)
                and self.bridge_data.horde_model_stickiness > 0
                and random.random() < self.bridge_data.horde_model_stickiness
            ):
                free_models = {
                    process.loaded_horde_model_name
                    for process in self._process_map.values()
                    if not process.is_process_busy() and process.loaded_horde_model_name is not None
                }
                if len(loaded_models) >= 1:
                    models = free_models
                logger.debug(f"Sticky models -- popping only {models}")
                if len(self.bridge_data.image_models_to_load) > 10:
                    logger.warning(
                        "Model stickiness is intended mostly for slow disks and works best with few models. "
                        f"You have {len(self.bridge_data.image_models_to_load)} models configured.",
                    )
            elif self.bridge_data.horde_model_stickiness > 0:
                logger.debug("Models unstuck: asking to pop for all available models.")

        # We'll only allow one running plus one queued for a given model.
        models_to_remove = {
            model
            for model, count in collections.Counter(
                [job.model for job in self._job_tracker.jobs_pending_inference],
            ).items()
            if count >= 2
        }
        if len(models_to_remove) > 0:
            models = models.difference(models_to_remove)

        if self.bridge_data.custom_models is not None and len(self.bridge_data.custom_models) > 0:
            logger.debug("Custom models are enabled, adding them to the list of models to pop")
            custom_model_names = {model["name"] for model in self.bridge_data.custom_models}
            models.update(custom_model_names)

        if len(models) == 0:
            logger.debug("Not eligible to pop a job yet")
            return

        try:
            job_pop_request = ImageGenerateJobPopRequest(
                apikey=self.bridge_data.api_key,
                name=self.bridge_data.dreamer_worker_name,
                bridge_agent=f"AI Horde Worker reGen:{horde_worker_regen.__version__}:https://github.com/Haidra-Org/horde-worker-reGen",
                models=list(models),
                blacklist=self.bridge_data.blacklist,
                nsfw=self.bridge_data.nsfw,
                threads=self.max_concurrent_inference_processes,
                max_pixels=self.bridge_data.max_power * 8 * 64 * 64,
                require_upfront_kudos=self.bridge_data.require_upfront_kudos,
                allow_img2img=self.bridge_data.allow_img2img,
                allow_painting=self.bridge_data.allow_inpainting,
                allow_unsafe_ipaddr=self.bridge_data.allow_unsafe_ip,
                allow_post_processing=self.bridge_data.allow_post_processing,
                allow_controlnet=self.bridge_data.allow_controlnet,
                allow_sdxl_controlnet=self.bridge_data.allow_sdxl_controlnet,
                extra_slow_worker=self.bridge_data.extra_slow_worker,
                limit_max_steps=self.bridge_data.limit_max_steps,
                allow_lora=self.bridge_data.allow_lora,
                amount=self.bridge_data.max_batch,
            )

            job_pop_response = await self.horde_client_session.submit_request(
                job_pop_request,
                ImageGenerateJobPopResponse,
            )
            try:
                if (
                    hasattr(job_pop_response, "messages")
                    and job_pop_response.messages is not None
                    and len(job_pop_response.messages) > 0
                ):
                    for message in job_pop_response.messages:
                        message_id = message.get("id", None)
                        message_text = str(message.get("message", None))
                        message_origin = message.get("origin", None)
                        message_expiry = message.get("expiry", None)

                        if message_id not in self._api_messages_received:
                            if message_id is not None:
                                message_id = str(message_id)
                            self._api_messages_received[message_id] = APIWorkerMessage(
                                message_id=message_id,
                                message_text=message_text,
                                message_origin=message_origin,
                                message_expiry=message_expiry,
                            )
                            logger.debug(
                                f"Message {message_id} from {message_origin} (expires {message_expiry}): "
                                f"{message_text}",
                            )
            except Exception as e:
                logger.error(f"Failed to process API messages: {e}")

            # TODO: horde_sdk should handle this and return a field with a enum(?) of the reason
            if isinstance(job_pop_response, RequestErrorResponse):
                if "maintenance mode" in job_pop_response.message.lower():
                    if not self._last_pop_maintenance_mode:
                        logger.warning(f"Failed to pop job (Maintenance Mode): {job_pop_response}")
                        self.print_maint_mode_messages()
                        self._last_pop_maintenance_mode = True

                elif "we cannot accept workers serving" in job_pop_response.message.lower():
                    logger.warning(f"Failed to pop job (Unrecognized Model): {job_pop_response}")
                    logger.error(
                        "Your worker is configured to use a model that is not accepted by the API. "
                        "Please check your models_to_load and make sure they are all valid.",
                    )
                elif "wrong credentials" in job_pop_response.message.lower():
                    logger.warning(f"Failed to pop job (Wrong Credentials): {job_pop_response}")
                    logger.error("Did you forget to set your worker name (`dreamer_name` in bridgeData.yaml)?")
                    logger.error(
                        "Horde Worker names must be unique horde-wide. If you haven't used this name before, "
                        "try changing your worker name.",
                    )
                else:
                    logger.error(f"Failed to pop job (API Error): {job_pop_response}")
                self._job_pop_frequency = self._error_job_pop_frequency
                self._last_pop_no_jobs_available = True
                return

        except Exception as e:
            if self._job_pop_frequency == self._error_job_pop_frequency:
                logger.error(f"Failed to pop job (Unexpected Error): {e}")
            else:
                logger.warning(f"Failed to pop job (Unexpected Error): {e}")

            self._job_pop_frequency = self._error_job_pop_frequency
            return

        self._last_pop_maintenance_mode = False
        self._replaced_due_to_maintenance = False

        self._job_pop_frequency = self._default_job_pop_frequency

        info_string = "No job available. "
        if len(self._job_tracker.jobs_pending_inference) > 0:
            info_string += f"Current number of popped jobs: {len(self._job_tracker.jobs_pending_inference)}. "

        skipped_reasons = job_pop_response.skipped.model_dump(exclude_defaults=True)
        # Include the extra fields as well
        if job_pop_response.skipped.model_extra is not None:
            skipped_reasons.update(job_pop_response.skipped.model_extra)

        # Remove any '0' values
        skipped_reasons = {k: v for k, v in skipped_reasons.items() if v != 0}

        info_string += f"(Skipped reasons: {skipped_reasons})"

        if job_pop_response.id_ is None:
            self._last_pop_no_jobs_available = True
            logger.info(info_string)
            if len(self._job_tracker.jobs_pending_inference) == 0:
                if self._last_pop_no_jobs_available_time == 0.0:
                    self._last_pop_no_jobs_available_time = cur_time

                self._time_spent_no_jobs_available += cur_time - self._last_pop_no_jobs_available_time
                self._last_pop_no_jobs_available_time = cur_time
            return

        self._job_tracker.job_faults[job_pop_response.id_] = []

        self._last_pop_no_jobs_available = False
        self._last_pop_no_jobs_available_time = 0.0

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
            f"({self.get_single_job_effective_megapixelsteps(job_pop_response)} eMPS) "
            f"(model: {job_pop_response.model}, batch: {job_pop_response.payload.n_iter}, "
            f"loras: {has_loras}, post_processing: {has_post_processing})"
            "</>",
        )

        # region TODO: move to horde_sdk
        if job_pop_response.payload.seed is None:  # TODO # FIXME
            logger.warning(f"Job {job_pop_response.id_} has no seed!")
            new_response_dict = job_pop_response.model_dump(by_alias=True)
            new_response_dict["payload"]["seed"] = random.randint(0, (2**32) - 1)

        if job_pop_response.payload.denoising_strength is not None and job_pop_response.source_image is None:
            new_response_dict = job_pop_response.model_dump(by_alias=True)
            new_response_dict["payload"]["denoising_strength"] = None

        if job_pop_response.payload.seed is None or (
            job_pop_response.payload.denoising_strength is not None and job_pop_response.source_image is None
        ):
            job_pop_response = ImageGenerateJobPopResponse(**new_response_dict)

        # Initiate the job faults list for this job, so that we don't need to check if it exists every time
        job_pop_response = await self._get_source_images(job_pop_response)

        # endregion

        if job_pop_response.id_ is None:
            logger.error("Job has no id!")
            return

        async with self._job_tracker.pending_inference_lock, self._job_tracker.pop_timestamps_lock:
            self._job_tracker.jobs_pending_inference.append(job_pop_response)
            jobs = []
            for job in self._job_tracker.jobs_pending_inference:
                if job.id_ is not None:
                    jobs.append(f"<{str(job.id_)[:8]}: {job.model}>")
                else:
                    jobs.append(f"<{job.model}>")
            logger.info(f"Job queue: {', '.join(jobs)}")
            # self._testing_jobs_added += 1
            self._job_tracker.job_pop_timestamps[job_pop_response] = time.time()
            self._job_tracker.jobs_lookup[job_pop_response] = HordeJobInfo(
                sdk_api_job_info=job_pop_response,
                state=None,
                time_popped=self._job_tracker.job_pop_timestamps[job_pop_response],
            )

    _user_info_failed = False
    """Whether the API request to fetch user info failed."""
    _user_info_failed_reason: str | None = None
    """The reason the API request to fetch user info failed."""

    _current_worker_id: str | None = None
    """The current worker ID."""

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
            max_time_spent_no_jobs_available=self._max_time_spent_no_jobs_available,
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

    _job_submit_loop_interval = 0.02
    """The interval between job submit loop iterations."""

    async def _job_submit_loop(self) -> None:
        """Run the job submit loop."""
        logger.debug("In _job_submit_loop")
        while True:
            with logger.catch():
                try:
                    await self.api_submit_job()
                    if self.is_time_for_shutdown():
                        break
                except CancelledError as e:
                    self._shutdown()
                    logger.debug(f"CancelledError: {e}")

            await asyncio.sleep(self._job_submit_loop_interval)

    async def _api_call_loop(self) -> None:
        """Run the API call loop for popping jobs and doing miscellaneous API calls."""
        logger.debug("In _api_call_loop")

        while True:
            with logger.catch():
                try:
                    await self.api_job_pop()

                    if self.is_time_for_shutdown() or self._shut_down:
                        break
                except CancelledError as e:
                    self._shutdown()
                    logger.debug(f"CancelledError: {e}")

            await asyncio.sleep(self._api_call_loop_interval)

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
    _replaced_due_to_maintenance = False

    async def _process_control_loop(self) -> None:
        self._process_lifecycle.start_safety_processes()
        self._process_lifecycle.start_inference_processes()

        while True:
            try:
                if self.stable_diffusion_reference is None:
                    return
                with logger.catch(reraise=True):
                    await asyncio.sleep(self._loop_interval)

                    async with (
                        self._job_tracker.lookup_lock,
                        self._job_tracker.pending_inference_lock,
                        self._job_tracker.safety_check_lock,
                        self._job_tracker.completed_jobs_lock,
                    ):
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
                        # We're in maintenance mode and there are no jobs to run, so we're going to unload all models
                        logger.warning("Reloading all process due to maintenance mode")
                        for process_info in self._process_map.values():
                            if process_info.process_type == HordeProcessType.INFERENCE:
                                self._process_lifecycle._replace_inference_process(process_info)
                            self._replaced_due_to_maintenance = True
                        self.print_maint_mode_messages()

                    if free_process_or_model_loaded and len(self._job_tracker.jobs_pending_inference) > 0:
                        # Theres a job pending inference and a process available to
                        # preload the model or start inference
                        async with (
                            self._job_tracker.lookup_lock,
                            self._job_tracker.pending_inference_lock,
                            self._job_tracker.safety_check_lock,
                            self._job_tracker.completed_jobs_lock,
                            self._job_tracker.pop_timestamps_lock,
                        ):
                            # So long as we didn't preload a model this cycle, we can start inference
                            # We want to get any messages next cycle from preloading processes to make sure
                            # the state of everything is up to date
                            if not self.preload_models():
                                next_job_and_process = self.get_next_job_and_process(information_only=True)

                                next_job_heavy_model_and_workflow = False
                                if next_job_and_process is not None:
                                    next_model = next_job_and_process.next_job.model
                                    if next_model is not None:
                                        next_model_baseline = self.stable_diffusion_reference.root.get(next_model)
                                        next_workflow = next_job_and_process.next_job.payload.workflow

                                        next_job_heavy_model_and_workflow = (
                                            next_model_baseline is not None
                                            and next_model_baseline
                                            == STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_xl
                                            and next_workflow in KNOWN_SLOW_WORKFLOWS
                                        )

                                        if next_model in VRAM_HEAVY_MODELS:
                                            next_job_heavy_model_and_workflow = True

                                keep_single_inference, single_inf_reason = self._process_map.keep_single_inference(
                                    stable_diffusion_model_reference=self.stable_diffusion_reference,
                                    post_process_job_overlap=self.bridge_data.post_process_job_overlap,
                                )

                                pending_and_active = (
                                    len(self._job_tracker.jobs_pending_inference)
                                    + len(self._job_tracker.jobs_in_progress)
                                )
                                if keep_single_inference and pending_and_active > 1:
                                    if (
                                        time.time() - self._batch_wait_log_time > 10
                                    ) and self.bridge_data.max_threads > 1:
                                        logger.opt(ansi=True).info(
                                            "<fg #7b7d7d>"
                                            f"<i>Blocking further inference due to {single_inf_reason}.</i>"
                                            "</>",
                                        )
                                        self._batch_wait_log_time = time.time()

                                elif (
                                    next_job_and_process is not None
                                    and (
                                        next_job_and_process.next_job.payload.n_iter > 1
                                        or next_job_heavy_model_and_workflow
                                    )
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
                                    if not self.start_inference():
                                        self.unload_models()

                    async with (
                        self._job_tracker.lookup_lock,
                        self._job_tracker.pending_inference_lock,
                        self._job_tracker.safety_check_lock,
                        self._job_tracker.completed_jobs_lock,
                    ):
                        await asyncio.sleep(self._loop_interval)
                        self.receive_and_handle_process_messages()
                        self._process_lifecycle._last_pop_no_jobs_available = self._last_pop_no_jobs_available
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
            jt = self._job_tracker
            async with jt.pending_inference_lock, jt.safety_check_lock, jt.completed_jobs_lock:
                self.receive_and_handle_process_messages()
                self.detect_deadlock()
                self._process_lifecycle._last_pop_no_jobs_available = self._last_pop_no_jobs_available
                self._process_lifecycle.replace_hung_processes()
            await asyncio.sleep(0.2)

        self._process_lifecycle.end_inference_processes(force=True)
        self._process_lifecycle.end_safety_processes()

        logger.info("Shutting down process manager")
        self._shut_down = True
        for process in self._process_map.values():
            process.mp_process.terminate()
            process.mp_process.terminate()
            process.mp_process.terminate()

            process.mp_process.join(0.2)

        await asyncio.sleep(0.2)

        return

    _last_deadlock_detected_time: float = 0.0
    """The epoch time of the last deadlock detected."""
    _in_deadlock: bool = False
    """Whether the worker is in a deadlock state."""
    _in_queue_deadlock: bool = False
    """Whether the worker is in a queue deadlock state."""
    _last_queue_deadlock_detected_time: float = 0.0
    """The epoch time of the last queue deadlock detected."""
    _queue_deadlock_model: str | None = None
    """The model causing the queue deadlock."""
    _queue_deadlock_process_id: int | None = None
    """The process ID causing the queue deadlock."""

    def detect_deadlock(self) -> None:
        """Detect if there are jobs in the queue but no processes doing anything."""

        def _print_deadlock_info() -> None:
            logger.debug(f"Jobs in queue: {len(self._job_tracker.jobs_pending_inference)}")
            logger.debug(f"Jobs in progress: {len(self._job_tracker.jobs_in_progress)}")
            logger.debug(f"Jobs pending safety check: {len(self._job_tracker.jobs_pending_safety_check)}")
            logger.debug(f"Jobs being safety checked: {len(self._job_tracker.jobs_being_safety_checked)}")
            logger.debug(f"Jobs completed: {len(self._job_tracker.jobs_pending_submit)}")
            logger.debug(f"Jobs faulted: {self._job_tracker._num_jobs_faulted}")
            logger.debug(f"horde_model_map: {self._horde_model_map}")
            logger.debug(f"process_map: {self._process_map}")

        if self._last_pop_recently():
            # We just popped a job, lets allow some time for gears to start turning
            # before we assume we're in a deadlock
            return

        if (
            not self._in_queue_deadlock
            and self._process_map.all_waiting_for_job()
            and len(self._job_tracker.jobs_pending_inference) > 0
        ):
            currently_loaded_models = set()
            model_process_map: dict[str, int] = {}
            for process in self._process_map.values():
                if process.loaded_horde_model_name is not None:
                    currently_loaded_models.add(process.loaded_horde_model_name)
                    model_process_map[process.loaded_horde_model_name] = process.process_id

            for job in self._job_tracker.jobs_pending_inference:
                if job.model in currently_loaded_models:
                    self._in_queue_deadlock = True
                    self._last_queue_deadlock_detected_time = time.time()
                    self._queue_deadlock_model = job.model
                    self._queue_deadlock_process_id = model_process_map[job.model]
                    break
            else:
                logger.debug("Queue deadlock detected without a model causing it.")
                _print_deadlock_info()
                self._in_queue_deadlock = True
                self._last_queue_deadlock_detected_time = time.time()
                # we're going to fall back to the next model in the deque
                self._queue_deadlock_model = self._job_tracker.jobs_pending_inference[0].model

        elif self._in_queue_deadlock and (self._last_queue_deadlock_detected_time + 30) < time.time():
            if self._process_map.num_starting_processes() > 0:
                logger.debug("Queue deadlock detected but some processes are starting. Waiting.")
                self._last_queue_deadlock_detected_time = time.time()
                return

            logger.debug("Queue deadlock detected")
            _print_deadlock_info()

            if self._queue_deadlock_model is not None:
                logger.debug(f"Model causing deadlock: {self._queue_deadlock_model}")
            else:
                logger.warning("Queue deadlock detected but no model causing it.")

            self._in_queue_deadlock = False
            self._queue_deadlock_model = None
            self._queue_deadlock_process_id = None

        if (
            (not self._in_deadlock)
            and (
                len(self._job_tracker.jobs_pending_inference) > 0
                or len(self._job_tracker.jobs_in_progress) > 0
                or len(self._job_tracker.jobs_lookup) > 0
            )
            and self._process_map.num_busy_processes() == 0
        ):
            self._last_deadlock_detected_time = time.time()
            self._in_deadlock = True
            logger.debug("Deadlock detected")
            _print_deadlock_info()
        elif (
            self._in_deadlock
            and (self._last_deadlock_detected_time + 10) < time.time()
            and self._process_map.num_busy_processes() == 0
        ):
            logger.debug("Deadlock still detected after 10 seconds.")

            self._in_deadlock = False
        elif (
            self._in_deadlock
            and (self._last_deadlock_detected_time + 5) < time.time()
            and self._process_map.num_busy_processes() > 0
        ):
            logger.debug("Deadlock was likely false-alarm.")
            self._in_deadlock = False

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
        process_control_loop = asyncio.create_task(self._process_control_loop(), name="process_control_loop")
        process_control_loop.add_done_callback(self._handle_exception)

        api_call_loop = asyncio.create_task(self._api_call_loop(), name="api_call_loop")
        api_call_loop.add_done_callback(self._handle_exception)

        api_get_user_info_loop = asyncio.create_task(self._api_get_user_info_loop(), name="api_get_user_info_loop")
        api_get_user_info_loop.add_done_callback(self._handle_exception)

        job_submit_loop = asyncio.create_task(self._job_submit_loop(), name="job_submit_loop")
        job_submit_loop.add_done_callback(self._handle_exception)

        bridge_data_loop = None
        if not self.bridge_data._loaded_from_env_vars:
            bridge_data_loop = asyncio.create_task(self._bridge_data_loop(), name="bridge_data_loop")
            bridge_data_loop.add_done_callback(self._handle_exception)

        tasks = [process_control_loop, api_call_loop, api_get_user_info_loop, job_submit_loop]

        if bridge_data_loop is not None:
            tasks.append(bridge_data_loop)

        self._aiohttp_client_session = ClientSession(requote_redirect_url=False)
        self.horde_client_session = AIHordeAPIAsyncClientSession(
            aiohttp_session=self._aiohttp_client_session,
            apikey=self.bridge_data.api_key,
        )

        async with self._aiohttp_client_session, self.horde_client_session:
            await asyncio.gather(*tasks)

    _caught_sigints = 0
    """The number of SIGINTs or SIGTERMs caught."""

    def start(self) -> None:
        """Start the process manager."""
        import signal

        signal.signal(signal.SIGINT, self.signal_handler)
        asyncio.run(self._main_loop())

    def signal_handler(self, sig: int, frame: object) -> None:
        """Handle SIGINT and SIGTERM."""
        if self._caught_sigints >= 2:
            logger.warning("Caught SIGINT or SIGTERM three times, exiting immediately")
            self._start_timed_shutdown()
            sys.exit(1)

        self._caught_sigints += 1
        logger.warning("Shutting down after current jobs are finished...")
        self._shutdown()

        global _caught_signal
        _caught_signal = True

    def _start_timed_shutdown(self) -> None:
        import threading

        def hard_shutdown() -> None:
            # Just in case the process manager gets stuck on shutdown
            time.sleep((len(self._job_tracker.jobs_pending_submit) * 4) + 2)

            for process in self._process_map.values():
                try:
                    process.mp_process.kill()
                    process.mp_process.kill()

                    process.mp_process.join(1)
                except Exception as e:
                    logger.error(f"Failed to kill process {process}: {e}")

            sys.exit(1)

        threading.Thread(target=hard_shutdown).start()

    _shutting_down = False
    """If true, the worker is scheduled to shut down."""
    _shutting_down_time = 0.0
    """The epoch time of when the worker started shutting down."""
    _shut_down = False
    """If true, the worker is out of the process control loop and should halt."""

    def _shutdown(self) -> None:
        if not self._shutting_down:
            self._shutting_down = True
            self._shutting_down_time = time.time()

    def _abort(self) -> None:
        """Exit as soon as possible, aborting all processes and jobs immediately."""
        with logger.catch(), open(".abort", "w") as f:
            f.write("")

        self._job_tracker._purge_jobs()

        self._shutdown()
        self._process_lifecycle._hard_kill_processes()
        self._start_timed_shutdown()

