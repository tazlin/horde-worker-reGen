"""Handles job submission to the AI Horde API."""

from __future__ import annotations

import asyncio
import contextlib
import ssl
import time
from asyncio import CancelledError, Task
from typing import TYPE_CHECKING

import aiohttp
import aiohttp.client_exceptions
import certifi
import yarl
from horde_sdk.ai_horde_api import GENERATION_STATE, AIHordeAPIAsyncClientSession
from horde_sdk.ai_horde_api.apimodels import (
    GenMetadataEntry,
    JobSubmitResponse,
)
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from horde_sdk.generic_api.apimodels import RequestErrorResponse
from loguru import logger

from horde_worker_regen.process_management.api_sessions import ApiSessions
from horde_worker_regen.process_management.job_models import (
    PendingSubmitJob,
)
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.model_metadata import ModelMetadata
from horde_worker_regen.process_management.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.worker_state import WorkerState
from horde_worker_regen.reporting.kudos_training_recorder import KudosTrainingRecorder
from horde_worker_regen.utils.image_utils import base64_image_to_stream_buffer

if TYPE_CHECKING:
    from horde_model_reference.model_reference_records import ImageGenerationModelRecord

    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.shutdown_manager import ShutdownManager

_sslcontext = ssl.create_default_context(cafile=certifi.where())

_async_client_exceptions: tuple[type[Exception], ...] = (TimeoutError, aiohttp.client_exceptions.ClientError, OSError)
with contextlib.suppress(AttributeError):
    _async_client_exceptions = (asyncio.exceptions.TimeoutError, aiohttp.client_exceptions.ClientError, OSError)


class JobSubmitter:
    """Owns job submission logic: uploading images to R2 and submitting results to the API."""

    _state: WorkerState
    _job_tracker: JobTracker
    _shutdown_manager: ShutdownManager
    _runtime_config: RuntimeConfig
    _api_sessions: ApiSessions
    _model_metadata: ModelMetadata

    _num_job_slowdowns: int
    _job_submit_loop_interval: float

    def __init__(
        self,
        *,
        state: WorkerState,
        job_tracker: JobTracker,
        shutdown_manager: ShutdownManager,
        runtime_config: RuntimeConfig,
        api_sessions: ApiSessions,
        model_metadata: ModelMetadata,
        dry_run_skip_api: bool = False,
    ) -> None:
        """Initialize the submitter with references to shared state and a shutdown manager.

        Args:
            state (WorkerState): The mutable flags relating to the worker's active state and lifecycle.
            job_tracker (JobTracker): The shared job tracker.
            shutdown_manager (ShutdownManager): The shutdown manager for coordinating shutdown across components.
            runtime_config (RuntimeConfig): Holds the current bridge configuration snapshot.
            api_sessions (ApiSessions): Holds the horde-sdk client and aiohttp sessions.
            model_metadata (ModelMetadata): Provides lookups against the stable-diffusion model reference.
            dry_run_skip_api (bool, optional): If true, skip all real API interactions and return dummy results. \
                Defaults to False.

        """
        self._state = state
        self._job_tracker = job_tracker
        self._shutdown_manager = shutdown_manager
        self._runtime_config = runtime_config
        self._api_sessions = api_sessions
        self._model_metadata = model_metadata
        self._dry_run_skip_api = dry_run_skip_api

        self._num_job_slowdowns = 0
        self._job_submit_loop_interval = 0.02

    @property
    def num_job_slowdowns(self) -> int:
        """Return how many submitted jobs were slower than the ideal kudos rate."""
        return self._num_job_slowdowns

    @property
    def bridge_data(self) -> reGenBridgeData:
        """Return the current bridge configuration."""
        return self._runtime_config.bridge_data

    @property
    def horde_client_session(self) -> AIHordeAPIAsyncClientSession:
        """Return the horde client session, or raise if it is not set."""
        return self._api_sessions.require_horde_client_session()

    @property
    def aiohttp_client_session(self) -> aiohttp.ClientSession:
        """Return the aiohttp client session, or raise if it is not set."""
        return self._api_sessions.require_aiohttp_session()

    @property
    def stable_diffusion_reference(self) -> dict[str, ImageGenerationModelRecord] | None:
        """Return the current stable diffusion reference, or None if it is not set."""
        return self._model_metadata.reference

    @logger.catch(reraise=True)
    async def submit_single_generation(self, new_submit: PendingSubmitJob) -> PendingSubmitJob:
        """Tries to upload and submit a single image from a batch.

        Args:
            new_submit: The job to attempt to submit.

        Returns:
            The modified in place job with the results of the submission attempt.
        """
        if self._dry_run_skip_api:
            logger.debug(f"Dry-run: skipping upload/submit for job {new_submit.job_id}")
            new_submit.succeed(0, 0.0)
            return new_submit

        logger.debug(f"Preparing to submit job {new_submit.job_id}")

        if new_submit.image_result is None and not new_submit.is_faulted:
            logger.error(f"Job {new_submit.job_id} has no image result")
            new_submit.fault()
            return new_submit

        if new_submit.image_result is not None:
            image_in_buffer = base64_image_to_stream_buffer(
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
                async with self.aiohttp_client_session.put(
                    yarl.URL(new_submit.r2_upload, encoded=True),
                    data=image_in_buffer_bytes,
                    skip_auto_headers=["content-type"],
                    timeout=aiohttp.ClientTimeout(total=10),
                    ssl=_sslcontext,
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
        metadata: list[GenMetadataEntry] = []
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
        time_popped = await self._job_tracker.get_time_popped(new_submit.completed_job_info.sdk_api_job_info)
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
            await self._job_tracker.increment_jobs_faulted()

        self._state.kudos_generated_this_session += job_submit_response.reward
        self._state.kudos_events.append((time.time(), job_submit_response.reward))
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
            self._state.consecutive_failed_jobs += 1

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
        time_popped = await self._job_tracker.get_time_popped(completed_job_info.sdk_api_job_info)
        if time_popped is None:
            logger.warning(
                f"Failed to get time_popped for job {completed_job_info.sdk_api_job_info.id_}. This is likely a bug.",
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
        for submit_job in finished_submit_jobs:
            if submit_job.is_faulted:
                job_faulted = True
                self._state.consecutive_failed_jobs += 1
                break
        if not job_faulted:
            # If any of the submits failed, we consider the whole job failed
            self._state.consecutive_failed_jobs = 0

        tracked_job_info = await self._job_tracker.ensure_submitted_job_info(completed_job_info)

        if self.bridge_data.capture_kudos_training_data:
            recorder = KudosTrainingRecorder(
                training_data_file=self.bridge_data.kudos_training_data_file,
                stable_diffusion_reference=self.stable_diffusion_reference,
            )
            recorder.record_job_data(tracked_job_info)

        try:
            await self._job_tracker.finalize_submitted(completed_job_info)
        except ValueError:
            logger.debug(
                f"Tried to remove completed_job_info "
                f"{completed_job_info.sdk_api_job_info.id_} but it has already been removed.",
            )

    async def run(self) -> None:
        """Run the job submit loop."""
        logger.debug("In JobSubmitter.run")
        while True:
            with logger.catch():
                try:
                    await self.api_submit_job()
                except CancelledError as e:
                    self._shutdown_manager.shutdown()
                    logger.debug(f"CancelledError: {e}")

            # Checked outside the catch block so persistent errors cannot prevent shutdown.
            if self._shutdown_manager.is_time_for_shutdown() or self._state.shut_down:
                break

            await asyncio.sleep(self._job_submit_loop_interval)
