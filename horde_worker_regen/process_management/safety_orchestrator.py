from __future__ import annotations

from collections.abc import Callable

from horde_model_reference.model_reference_records import StableDiffusion_ModelReference
from loguru import logger

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeProcessState,
    HordeSafetyControlMessage,
)
from horde_worker_regen.telemetry_spans import span_safety_check
from horde_worker_regen.process_management.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.protocols import BridgeDataProvider
from horde_worker_regen.process_management.worker_state import WorkerState


class SafetyOrchestrator:
    """Send completed inference jobs to the safety process for evaluation."""

    _process_map: ProcessMap
    _job_tracker: JobTracker
    _process_lifecycle: ProcessLifecycleManager

    _get_bridge_data: BridgeDataProvider
    _get_stable_diffusion_reference: Callable[[], StableDiffusion_ModelReference | None]

    def __init__(
        self,
        *,
        process_map: ProcessMap,
        job_tracker: JobTracker,
        process_lifecycle: ProcessLifecycleManager,
        get_bridge_data: BridgeDataProvider,
        get_stable_diffusion_reference: Callable[[], StableDiffusion_ModelReference | None],
        state: WorkerState,
    ) -> None:
        self._process_map = process_map
        self._job_tracker = job_tracker
        self._process_lifecycle = process_lifecycle
        self._get_bridge_data = get_bridge_data
        self._get_stable_diffusion_reference = get_stable_diffusion_reference
        self._state = state

    def start_evaluate_safety(self) -> None:
        """Start evaluating the safety of the next job pending a safety check, if any."""
        if len(self._job_tracker.jobs_pending_safety_check) == 0:
            return

        safety_process = self._process_map.get_first_available_safety_process()

        if safety_process is None:
            return

        completed_job_info = self._job_tracker.jobs_pending_safety_check[0]

        stable_diffusion_reference = self._get_stable_diffusion_reference()
        if stable_diffusion_reference is None:
            raise ValueError("stable_diffusion_reference is None")

        bridge_data = self._get_bridge_data()

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
                process_timeout=bridge_data.process_timeout,
            )
            logger.error(f"Failed to start safety evaluation for job {completed_job_info.sdk_api_job_info.id_}")
            self._job_tracker.jobs_pending_safety_check.remove(completed_job_info)

            return

        if completed_job_info.sdk_api_job_info.id_ is None:
            raise ValueError("completed_job_info.sdk_api_job_info.id_ is None")
        if completed_job_info.sdk_api_job_info.payload.prompt is None:
            raise ValueError("completed_job_info.sdk_api_job_info.payload.prompt is None")
        if completed_job_info.sdk_api_job_info.model is None:
            raise ValueError("completed_job_info.sdk_api_job_info.model is None")

        model_info = {}
        if completed_job_info.sdk_api_job_info.model in stable_diffusion_reference.root:
            model_info = stable_diffusion_reference.root[completed_job_info.sdk_api_job_info.model].model_dump()
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
            if len(self._job_tracker.jobs_being_safety_checked) > 0:
                for job_info in self._job_tracker.jobs_being_safety_checked:
                    self._job_tracker.jobs_pending_safety_check.append(job_info)
        else:
            self._job_tracker.jobs_pending_safety_check.remove(completed_job_info)
            self._job_tracker.jobs_being_safety_checked.append(completed_job_info)
