"""Handles job popping from the AI Horde API."""

from __future__ import annotations

import asyncio
import collections
import random
import time
from asyncio import CancelledError
from typing import TYPE_CHECKING

from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopRequest,
    ImageGenerateJobPopResponse,
)
from loguru import logger

import horde_worker_regen
from horde_worker_regen.process_management._canned_scenarios import CannedJobSource, make_default_dry_run_source
from horde_worker_regen.process_management.api_sessions import ApiSessions
from horde_worker_regen.process_management.job_models import APIWorkerMessage
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.pop_throttler import (
    CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS,
    PopThrottler,
)
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.source_image_downloader import SourceImageDownloader
from horde_worker_regen.process_management.worker_state import WorkerState
from horde_worker_regen.reporting.maintenance_messenger import MaintenanceModeMessenger
from horde_worker_regen.telemetry_spans import queue_depth_counter, span_job_pop
from horde_worker_regen.utils.job_utils import get_single_job_magnitude

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.shutdown_manager import ShutdownManager


def _select_models_for_pop(
    bridge_data: reGenBridgeData,
    process_map: ProcessMap,
    job_tracker: JobTracker,
    max_inference_processes: int,
    *,
    last_pop_had_no_jobs: bool,
) -> set[str] | None:
    """Choose which models to include in a pop request.

    Returns:
        A set of model names, or ``None`` if no models are eligible
        (caller should skip the pop).
    """
    models = set(bridge_data.image_models_to_load)

    loaded_models = {
        process.loaded_horde_model_name
        for process in process_map.values()
        if process.loaded_horde_model_name is not None
    }

    if (
        len(bridge_data.image_models_to_load) > max_inference_processes
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
                # busy; in that case no pop occurs (intentional — there is
                # no process available to accept a new job).
                models = free_models
            logger.debug(f"Sticky models -- popping only {models}")
            if len(bridge_data.image_models_to_load) > 10:
                logger.warning(
                    "Model stickiness is intended mostly for slow disks and works best with few models. "
                    f"You have {len(bridge_data.image_models_to_load)} models configured.",
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

    if bridge_data.custom_models is not None and len(bridge_data.custom_models) > 0:
        logger.debug("Custom models are enabled, adding them to the list of models to pop")
        custom_model_names = {model["name"] for model in bridge_data.custom_models}
        models.update(custom_model_names)

    if len(models) == 0:
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

    _max_inference_processes: int
    _max_concurrent_inference_processes: int

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
    ) -> None:
        """Initialize with all required dependencies for job popping.

        When `dry_run_skip_api` is set, jobs come from `canned_job_source` instead of
        the live API; if no source is given, an endlessly-cycling default is used.
        """
        self._state = state
        self._process_map = process_map
        self._job_tracker = job_tracker
        self._shutdown_manager = shutdown_manager
        self._runtime_config = runtime_config
        self._api_sessions = api_sessions

        self._max_inference_processes = max_inference_processes
        self._max_concurrent_inference_processes = max_concurrent_inference_processes
        self._dry_run_skip_api = dry_run_skip_api

        self._canned_job_source = canned_job_source
        if dry_run_skip_api and self._canned_job_source is None:
            self._canned_job_source = make_default_dry_run_source()

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

    def _is_hungry(self, bridge_data: reGenBridgeData) -> bool:
        """Whether the worker should pop again immediately instead of waiting the poll interval.

        True only when work is actively flowing (the last pop returned a job), the local queue
        has room (`_is_queue_full` is False), an inference process is free to take a job, and we
        are not in post-error backoff. In that state the fixed ~1s poll cadence would leave a
        freed GPU slot starved while a job is readily available; popping back-to-back fills the
        buffer so the slot refills without delay. When the queue is full, no process is free, the
        source has no work, or we are backing off, this is False and the loop reverts to polite
        interval polling — so this never increases pressure on the API beyond filling the buffer.
        """
        if self._state.last_pop_no_jobs_available:
            return False
        if self._pop_throttler.is_in_error_backoff:
            return False
        if self._is_queue_full(bridge_data):
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

        cur_time = time.time()
        bridge_data = self._runtime_config.bridge_data

        if self._handle_consecutive_failures(bridge_data, cur_time):
            return

        if self._is_queue_full(bridge_data):
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

        models = _select_models_for_pop(
            bridge_data,
            self._process_map,
            self._job_tracker,
            self._max_inference_processes,
            last_pop_had_no_jobs=self._state.last_pop_no_jobs_available,
        )
        if models is None:
            return

        try:
            job_pop_request = ImageGenerateJobPopRequest(
                apikey=bridge_data.api_key,
                name=bridge_data.dreamer_worker_name,
                bridge_agent=f"AI Horde Worker reGen:{horde_worker_regen.__version__}:https://github.com/Haidra-Org/horde-worker-reGen",
                models=list(models),
                blacklist=bridge_data.blacklist,
                nsfw=bridge_data.nsfw,
                threads=self._max_concurrent_inference_processes,
                max_pixels=bridge_data.max_power * 8 * 64 * 64,
                require_upfront_kudos=bridge_data.require_upfront_kudos,
                allow_img2img=bridge_data.allow_img2img,
                allow_painting=bridge_data.allow_inpainting,
                allow_unsafe_ipaddr=bridge_data.allow_unsafe_ip,
                allow_post_processing=bridge_data.allow_post_processing,
                allow_controlnet=bridge_data.allow_controlnet,
                allow_sdxl_controlnet=bridge_data.allow_sdxl_controlnet,
                extra_slow_worker=bridge_data.extra_slow_worker,
                limit_max_steps=bridge_data.limit_max_steps,
                allow_lora=bridge_data.allow_lora,
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

        self._state.last_pop_maintenance_mode = False
        self._replaced_due_to_maintenance = False
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

        self._state.last_pop_no_jobs_available = False
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
        (a GPU slot is free, the queue has room, and work is flowing — see :meth:`_is_hungry`),
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
