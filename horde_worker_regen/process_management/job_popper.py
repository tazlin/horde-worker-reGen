"""Handles job popping from the AI Horde API."""

from __future__ import annotations

import asyncio
import collections
import random
import time
from asyncio import CancelledError, Task
from collections.abc import Callable
from typing import TYPE_CHECKING

from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.apimodels import (
    GenMetadataEntry,
    ImageGenerateJobPopRequest,
    ImageGenerateJobPopResponse,
)
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from loguru import logger

import horde_worker_regen
from horde_worker_regen.consts import MAX_SOURCE_IMAGE_RETRIES
from horde_worker_regen.process_management.job_models import (
    APIWorkerMessage,
    HordeJobInfo,
)
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState
from horde_worker_regen.reporting.maintenance_messenger import MaintenanceModeMessenger

if TYPE_CHECKING:
    from aiohttp import ClientSession

    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.shutdown_manager import ShutdownManager


class JobPopper:
    """Owns job pop logic: requesting new jobs from the API and downloading source images."""

    _state: WorkerState
    _process_map: ProcessMap
    _job_tracker: JobTracker
    _shutdown_manager: ShutdownManager
    _get_bridge_data: Callable[[], reGenBridgeData]
    _get_horde_client_session: Callable[[], object]
    _get_aiohttp_session: Callable[[], ClientSession]
    _get_effective_megapixelsteps: Callable[[ImageGenerateJobPopResponse], int]

    _default_job_pop_frequency: float
    _error_job_pop_frequency: float
    _job_pop_frequency: float
    _max_time_spent_no_jobs_available: float
    _too_many_consecutive_failed_jobs_wait_time: int
    _replaced_due_to_maintenance: bool
    _api_messages_received: dict[str | None, APIWorkerMessage]
    _last_pop_no_jobs_available_time: float
    _time_spent_no_jobs_available: float
    _api_call_loop_interval: float

    max_inference_processes: int
    max_concurrent_inference_processes: int

    def __init__(
        self,
        *,
        state: WorkerState,
        process_map: ProcessMap,
        job_tracker: JobTracker,
        shutdown_manager: ShutdownManager,
        get_bridge_data: Callable[[], reGenBridgeData],
        get_horde_client_session: Callable[[], object],
        get_aiohttp_session: Callable[[], ClientSession],
        get_effective_megapixelsteps: Callable[[ImageGenerateJobPopResponse], int],
        max_inference_processes: int,
        max_concurrent_inference_processes: int,
    ) -> None:
        self._state = state
        self._process_map = process_map
        self._job_tracker = job_tracker
        self._shutdown_manager = shutdown_manager
        self._get_bridge_data = get_bridge_data
        self._get_horde_client_session = get_horde_client_session
        self._get_aiohttp_session = get_aiohttp_session
        self._get_effective_megapixelsteps = get_effective_megapixelsteps
        self.max_inference_processes = max_inference_processes
        self.max_concurrent_inference_processes = max_concurrent_inference_processes

        self._default_job_pop_frequency = 1.0
        self._error_job_pop_frequency = 5.0
        self._job_pop_frequency = 1.0
        self._max_time_spent_no_jobs_available = 60.0 * 60.0
        self._too_many_consecutive_failed_jobs_wait_time = 180
        self._replaced_due_to_maintenance = False
        self._api_messages_received = {}
        self._last_pop_no_jobs_available_time = 0.0
        self._time_spent_no_jobs_available = 0.0
        self._api_call_loop_interval = 1

    @property
    def bridge_data(self) -> reGenBridgeData:
        return self._get_bridge_data()

    @property
    def horde_client_session(self) -> object:
        return self._get_horde_client_session()

    @property
    def aiohttp_client_session(self) -> ClientSession:
        return self._get_aiohttp_session()

    async def _get_source_images(
        self, job_pop_response: ImageGenerateJobPopResponse,
    ) -> ImageGenerateJobPopResponse:
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
                download_tasks.append(job_pop_response.async_download_source_image(self.aiohttp_client_session))
            if (
                source_mask_is_url
                and job_pop_response.source_mask is not None
                and job_pop_response.get_downloaded_source_mask() is None
            ):
                download_tasks.append(job_pop_response.async_download_source_mask(self.aiohttp_client_session))

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
                            self.aiohttp_client_session,
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

    @logger.catch(reraise=True)
    async def api_job_pop(self) -> None:
        """If the job deque is not full, add any jobs that are available to the job deque."""
        if self._state.shutting_down:
            self._state.last_pop_no_jobs_available = False
            return

        cur_time = time.time()

        if self._state.too_many_consecutive_failed_jobs:
            if (
                cur_time - self._state.too_many_consecutive_failed_jobs_time
                > self._too_many_consecutive_failed_jobs_wait_time
            ):
                self._state.consecutive_failed_jobs = 0
                self._state.too_many_consecutive_failed_jobs = False
                logger.debug("Resuming job pops after too many consecutive failed jobs")
            return

        if self._state.consecutive_failed_jobs >= 3:
            logger.error(
                "Too many consecutive failed jobs, pausing job pops. "
                "Please look into what happened and let the devs know. ",
                f"Waiting {self._too_many_consecutive_failed_jobs_wait_time} seconds...",
            )
            if self.bridge_data.exit_on_unhandled_faults:
                logger.error("Exiting due to exit_on_unhandled_faults being enabled")
                self._shutdown_manager.shutdown()
            self._state.too_many_consecutive_failed_jobs = True
            self._state.too_many_consecutive_failed_jobs_time = cur_time
            return

        max_jobs_in_queue = self.bridge_data.queue_size + 1

        if self.bridge_data.max_threads > 1:
            max_jobs_in_queue += self.bridge_data.max_threads - 1

        if len(self._job_tracker.jobs_pending_inference) >= max_jobs_in_queue:
            return

        # We let the first job run through to make sure things are working
        if len(self._job_tracker.jobs_pending_inference) != 0 and self._job_tracker.jobs_pending_submit == 0:
            return

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
        if time.time() - self._state.last_job_pop_time < self._job_pop_frequency:
            return

        self._state.last_job_pop_time = time.time()

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
                (not self._state.last_pop_no_jobs_available)
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
                    and len(job_pop_response.messages) > 0  # type: ignore # FIXME: requires updated sdk
                ):
                    for message in job_pop_response.messages:  # type: ignore # FIXME: requires updated sdk
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

            if isinstance(job_pop_response, RequestErrorResponse):
                if "maintenance mode" in job_pop_response.message.lower():
                    if not self._state.last_pop_maintenance_mode:
                        logger.warning(f"Failed to pop job (Maintenance Mode): {job_pop_response}")
                        MaintenanceModeMessenger.print_maintenance_mode_messages()
                        self._state.last_pop_maintenance_mode = True

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
                self._state.last_pop_no_jobs_available = True
                return

        except Exception as e:
            if self._job_pop_frequency == self._error_job_pop_frequency:
                logger.error(f"Failed to pop job (Unexpected Error): {e}")
            else:
                logger.warning(f"Failed to pop job (Unexpected Error): {e}")

            self._job_pop_frequency = self._error_job_pop_frequency
            return

        self._state.last_pop_maintenance_mode = False
        self._replaced_due_to_maintenance = False

        self._job_pop_frequency = self._default_job_pop_frequency

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
            logger.info(info_string)
            if len(self._job_tracker.jobs_pending_inference) == 0:
                if self._last_pop_no_jobs_available_time == 0.0:
                    self._last_pop_no_jobs_available_time = cur_time

                self._time_spent_no_jobs_available += cur_time - self._last_pop_no_jobs_available_time
                self._last_pop_no_jobs_available_time = cur_time
            return

        self._job_tracker.job_faults[job_pop_response.id_] = []

        self._state.last_pop_no_jobs_available = False
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
            f"({self._get_effective_megapixelsteps(job_pop_response)} eMPS) "
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
            self._job_tracker.job_pop_timestamps[job_pop_response] = time.time()
            self._job_tracker.jobs_lookup[job_pop_response] = HordeJobInfo(
                sdk_api_job_info=job_pop_response,
                state=None,
                time_popped=self._job_tracker.job_pop_timestamps[job_pop_response],
            )

    async def run(self) -> None:
        """Run the API call loop for popping jobs."""
        logger.debug("In JobPopper.run")

        while True:
            with logger.catch():
                try:
                    await self.api_job_pop()

                    if self._shutdown_manager.is_time_for_shutdown() or self._state.shut_down:
                        break
                except CancelledError as e:
                    self._shutdown_manager.shutdown()
                    logger.debug(f"CancelledError: {e}")

            await asyncio.sleep(self._api_call_loop_interval)
