from __future__ import annotations

import queue
import time
from collections.abc import Callable
from multiprocessing import Queue
from typing import TYPE_CHECKING

from horde_model_reference.meta_consts import STABLE_DIFFUSION_BASELINE_CATEGORY
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from loguru import logger

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.telemetry_spans import (
    inference_duration_histogram,
    jobs_completed_counter,
    jobs_faulted_counter,
    queue_depth_counter,
)
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import (
    HordeAuxModelStateChangeMessage,
    HordeInferenceResultMessage,
    HordeModelStateChangeMessage,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    HordeSafetyResultMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.protocols import BridgeDataProvider
from horde_worker_regen.process_management.worker_state import WorkerState

if TYPE_CHECKING:
    from horde_worker_regen.process_management.job_models import HordeJobInfo

_excludes_for_job_dump = {"source_image", "source_mask", "extra_source_images", "r2_upload"}


class MessageDispatcher:
    """Drains the IPC message queue and updates process/model state."""

    _process_map: ProcessMap
    _horde_model_map: HordeModelMap
    _job_tracker: JobTracker
    _process_message_queue: Queue  # type: ignore[type-arg]

    _get_model_baseline: Callable[[str], STABLE_DIFFUSION_BASELINE_CATEGORY | str | None]
    _get_bridge_data: BridgeDataProvider
    _on_unload_vram: Callable[[HordeProcessInfo], None]

    _last_deadlock_detected_time: float = 0.0
    _in_deadlock: bool = False
    _in_queue_deadlock: bool = False
    _last_queue_deadlock_detected_time: float = 0.0
    _queue_deadlock_model: str | None = None
    _queue_deadlock_process_id: int | None = None

    def __init__(
        self,
        *,
        process_map: ProcessMap,
        horde_model_map: HordeModelMap,
        job_tracker: JobTracker,
        process_message_queue: Queue,  # type: ignore[type-arg]
        get_model_baseline: Callable[[str], STABLE_DIFFUSION_BASELINE_CATEGORY | str | None],
        get_bridge_data: BridgeDataProvider,
        on_unload_vram: Callable[[HordeProcessInfo], None],
        state: WorkerState,
    ) -> None:
        self._process_map = process_map
        self._horde_model_map = horde_model_map
        self._job_tracker = job_tracker
        self._process_message_queue = process_message_queue
        self._get_model_baseline = get_model_baseline
        self._get_bridge_data = get_bridge_data
        self._on_unload_vram = on_unload_vram
        self._state = state

    def receive_and_handle_process_messages(self) -> None:
        """Receive and handle any messages from the child processes."""
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
            model_baseline = self._get_model_baseline(model_name) if model_name is not None else None

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
            and self._process_map[message.process_id].last_process_state != HordeProcessState.UNLOADED_MODEL_FROM_RAM
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
                        f"Job {message.sdk_api_job_info.id_} not found in jobs_lookup. (Process {message.process_id})",
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

        model_baseline = self._get_model_baseline(message.horde_model_name)

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
                    f"Job {message.sdk_api_job_info.id_} found in jobs_in_progress. (Process {message.process_id})",
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
        queue_depth_counter.add(-1)
        bridge_data = self._get_bridge_data()
        if bridge_data.unload_models_from_vram_often:
            self._on_unload_vram(self._process_map[message.process_id])

        if message.time_elapsed is not None:
            inference_duration_histogram.record(message.time_elapsed)
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

            jobs_completed_counter.add(1)
            self._job_tracker.jobs_pending_safety_check.append(job_info)
        else:
            jobs_faulted_counter.add(1)
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

        if self._state.last_pop_recently():
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
