from __future__ import annotations

from loguru import logger

from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessState,
    HordeSafetyControlMessage,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.telemetry_spans import span_safety_check


class SafetyOrchestrator:
    """Send completed inference jobs to the safety process for evaluation."""

    _process_map: ProcessMap
    _job_tracker: JobTracker
    _process_lifecycle: ProcessLifecycleManager
    _runtime_config: RuntimeConfig
    _model_metadata: ModelMetadata

    def __init__(
        self,
        *,
        process_map: ProcessMap,
        job_tracker: JobTracker,
        process_lifecycle: ProcessLifecycleManager,
        runtime_config: RuntimeConfig,
        model_metadata: ModelMetadata,
        state: WorkerState,
    ) -> None:
        """Initialize the orchestrator with references to its dependencies.

        Args:
            process_map (ProcessMap): The process map to use for finding safety processes.
            job_tracker (JobTracker): The job tracker to use for moving jobs between pending and being
                checked.
            process_lifecycle (ProcessLifecycleManager): The process lifecycle manager to signal if safety
                processes need to be replaced.
            runtime_config (RuntimeConfig): Holds the current bridge configuration snapshot.
            model_metadata (ModelMetadata): Provides lookups against the stable-diffusion model reference.
            state (WorkerState): The worker state to check for dry-run mode.
        """
        self._process_map = process_map
        self._job_tracker = job_tracker
        self._process_lifecycle = process_lifecycle
        self._runtime_config = runtime_config
        self._model_metadata = model_metadata
        self._state = state

    async def start_evaluate_safety(self) -> None:
        """Start evaluating the safety of the next job pending a safety check, if any."""
        if len(self._job_tracker.jobs_pending_safety_check) == 0:
            return

        safety_process = self._process_map.get_first_available_safety_process()

        if safety_process is None:
            return

        completed_job_info = self._job_tracker.jobs_pending_safety_check[0]

        stable_diffusion_reference = self._model_metadata.require_reference()

        bridge_data = self._runtime_config.bridge_data

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
            # A post-inference safety setup failure (missing images/id/model/prompt) cannot be fixed by
            # re-running inference, so it is faulted terminally rather than requeued.
            await self._job_tracker.handle_job_fault(
                faulted_job=completed_job_info.sdk_api_job_info,
                process_info=safety_process,
                process_timeout=bridge_data.process_timeout,
                retryable=False,
            )
            logger.error(f"Failed to start safety evaluation for job {completed_job_info.sdk_api_job_info.id_}")
            await self._job_tracker.abandon_pending_safety(completed_job_info)

            return

        if completed_job_info.sdk_api_job_info.payload.prompt is None:
            # For static type checking the use below; this should never be hit at runtime, see above
            raise ValueError("completed_job_info.sdk_api_job_info.payload.prompt is None")

        model_info = None
        if completed_job_info.sdk_api_job_info.model in stable_diffusion_reference:
            model_info = stable_diffusion_reference[completed_job_info.sdk_api_job_info.model]
        with span_safety_check(job_id=str(completed_job_info.sdk_api_job_info.id_)):
            safety_message_sent_succeeded = safety_process.safe_send_message(
                HordeSafetyControlMessage(
                    control_flag=HordeControlFlag.EVALUATE_SAFETY,
                    job_id=completed_job_info.sdk_api_job_info.id_,
                    images_base64=completed_job_info.images_base64,
                    prompt=completed_job_info.sdk_api_job_info.payload.prompt,
                    censor_nsfw=completed_job_info.sdk_api_job_info.payload.use_nsfw_censor,
                    sfw_worker=not bridge_data.nsfw,
                    horde_model_info=model_info,
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
            await self._job_tracker.requeue_being_safety_checked()
        else:
            await self._job_tracker.begin_safety_check(completed_job_info)
