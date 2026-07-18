"""The dedicated text-encode service process (the disaggregated pipeline's encode stage).

Under pipeline disaggregation a job's prompts are encoded to CONDITIONING in this process, which loads
only the text encoders (not the UNet or VAE), so the sampler that consumes the conditioning never
carries the text-encoder weights. It holds a hordelib backend across jobs (the text encoders are the
components it keeps resident) and serves ``START_TEXT_ENCODE`` requests, returning the positive and
negative CONDITIONING blobs the sampler injects.

This replaced the earlier component-sharing lane (which published CUDA-IPC weight handles for siblings
to adopt); that mechanism was removed in favour of moving small activations between processes rather
than sharing weights. The ``HordeProcessType.COMPONENT`` role and its spawn/replace lifecycle are
reused unchanged.
"""

from __future__ import annotations

import gc
import sys
from typing import TYPE_CHECKING, override

from loguru import logger

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock

from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.ipc.messages import (
    GENERATION_STATE,
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessState,
    HordeTextEncodeControlMessage,
    HordeTextEncodeResultMessage,
    PipelineStageTag,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcess, HordeProcessType
from horde_worker_regen.utils.oom_signature import is_resource_class_exception

if TYPE_CHECKING:
    from hordelib.api import HordeLib


class HordeComponentLaneProcess(HordeProcess):
    """The dedicated text-encode service: encodes prompts to CONDITIONING, loading only the text encoders."""

    _horde_model_names: list[str]
    _dry_run: bool
    _horde: HordeLib

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        device_index: int = 0,
        horde_model_names: list[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        """Initialise the text-encode service.

        Args:
            process_id (int): The reserved id for this process (not the OS PID).
            process_message_queue (ProcessQueue): The queue used to send messages to the main process.
            pipe_connection (Connection): Receives ``HordeControlMessage``s from the main process.
            disk_lock (Lock): The lock used when accessing disk.
            process_launch_identifier (int): The unique identifier for this launch.
            device_index (int, optional): The stable index of the GPU this process is pinned to. Defaults to 0.
            horde_model_names (list[str] | None, optional): The worker's configured models (retained for the
                spawn signature; text-encode derives its model per job). Defaults to None.
            dry_run (bool, optional): Skip the hordelib backend (used by tests and the dry-run worker).
                Defaults to False.
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
        )

        self.process_type = HordeProcessType.COMPONENT
        # The service pages the text encoders onto the device per encode, so its periodic report samples VRAM
        # like inference and the post-process lane, keeping the parent's per-card free-VRAM view fresh.
        self._periodic_report_includes_vram = True
        self._horde_model_names = horde_model_names or []
        self._dry_run = dry_run

        if not dry_run:
            self._bring_up_backend()

        logger.info("HordeComponentLaneProcess (text-encode service) initialised")
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Text-encode service ready",
        )

    def _bring_up_backend(self) -> None:
        """Initialise the hordelib backend the service needs to run the text-encode stage."""
        try:
            with logger.catch(reraise=True):
                from hordelib.api import HordeLib, SharedModelManager
        except Exception as import_error:
            logger.critical(f"Text-encode service backend import failed: {type(import_error).__name__} {import_error}")
            sys.exit(1)

        from horde_worker_regen.reference_helper import ensure_offline_reference_manager

        ensure_offline_reference_manager()
        SharedModelManager(do_not_load_model_mangers=True)
        SharedModelManager.load_model_managers(multiprocessing_lock=self.disk_lock, lora_reference_backups=False)
        # Held across jobs (no per-job reload); the text encoders are the components it keeps resident.
        self._horde = HordeLib(aggressive_unloading=False)

    def _run_text_encode(self, message: HordeTextEncodeControlMessage) -> None:
        """Encode a job's prompts to positive/negative CONDITIONING blobs, loading only the text encoders."""
        import time

        from horde_sdk.worker.dispatch.ai_horde.image.convert import (
            convert_image_job_pop_response_to_parameters,
        )

        from horde_worker_regen.reference_helper import ensure_offline_reference_manager

        self.send_process_state_change_message(
            HordeProcessState.INFERENCE_STARTING, info=f"Text-encode {message.job_id}"
        )
        time_start = time.time()
        state = GENERATION_STATE.ok
        fault_is_resource_class = False
        fault_reason: str | None = None
        positive_bytes: bytes | None = None
        negative_bytes: bytes | None = None
        try:
            if self._dry_run:
                # No backend in dry-run: return opaque CONDITIONING stand-ins. Nothing deserializes them
                # (a dry-run sampler skips sampling entirely), so the fake pipeline still flows end to end.
                positive_bytes, negative_bytes = b"dry-run-conditioning", b"dry-run-conditioning"
            else:
                params = convert_image_job_pop_response_to_parameters(
                    api_response=message.sdk_api_job_info,
                    model_reference_manager=ensure_offline_reference_manager(),
                ).generation_parameters
                positive_bytes, negative_bytes = self._horde.encode_text_stage(params)
        except Exception as encode_error:  # noqa: BLE001 - a stage fault is reported, never crashes the service
            logger.error(f"Text-encode failed for job {message.job_id}: {type(encode_error).__name__} {encode_error}")
            state = GENERATION_STATE.faulted
            fault_is_resource_class = is_resource_class_exception(encode_error)
            fault_reason = f"{type(encode_error).__name__}: {encode_error}"

        self.process_message_queue.put(
            HordeTextEncodeResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Text-encode for job {message.job_id}",
                time_elapsed=time.time() - time_start,
                job_id=message.job_id,
                positive_conditioning_bytes=positive_bytes,
                negative_conditioning_bytes=negative_bytes,
                state=state,
                fault_is_resource_class=fault_is_resource_class,
                fault_reason=fault_reason,
            ),
        )
        self.send_stage_job_metrics_message(str(message.job_id), stage=PipelineStageTag.TEXT_ENCODE)
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, info="Text-encode service ready")

    @staticmethod
    def clear_gc_and_torch_cache() -> None:
        """Clear Python garbage and the active backend's device cache."""
        gc.collect()
        from hordelib.api import clear_accelerator_cache

        clear_accelerator_cache()

    @logger.catch(reraise=True)
    def unload_models_from_vram(self) -> None:
        """Unload the resident text encoders from VRAM and report the refreshed memory sample.

        The service keeps the process alive (it re-pages the encoders on the next encode); only their
        device residency is released so the parent can reclaim the card for another tenant.
        """
        if not self._dry_run:
            self._horde.backend.free_vram()
            self.clear_gc_and_torch_cache()

        self.send_process_state_change_message(
            process_state=HordeProcessState.UNLOADED_MODEL_FROM_VRAM,
            info="Unloaded text-encode service models from VRAM",
        )
        self.send_memory_report_message(include_vram=True)
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Text-encode service ready")

    @logger.catch(reraise=True)
    def unload_models_from_ram(self) -> None:
        """Unload the resident text encoders from RAM/VRAM and report the refreshed memory sample."""
        if not self._dry_run:
            self._horde.backend.free_ram()
            self.clear_gc_and_torch_cache()

        self.send_process_state_change_message(
            process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
            info="Unloaded text-encode service models from RAM",
        )
        self.send_memory_report_message(include_vram=True)
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Text-encode service ready")

    @logger.catch(reraise=True)
    def release_allocator_cache(self) -> None:
        """Release the torch allocator's cached free blocks without unloading models, then report memory.

        Empties the caching allocator's reserved-but-unused device blocks so the reservation returns to
        the card while the resident text encoders stay loaded, then reports the refreshed memory sample.
        Deliberately emits no model state change: nothing was unloaded.
        """
        logger.debug("Releasing allocator cache (resident encoders stay loaded)")
        if not self._dry_run:
            self.clear_gc_and_torch_cache()
        self.send_memory_report_message(include_vram=True)

    @override
    def cleanup_for_exit(self) -> None:
        """Free the service's VRAM on teardown."""
        if not self._dry_run:
            try:
                from hordelib.api import clear_gc_and_torch_cache

                clear_gc_and_torch_cache()  # type: ignore - PEP 562 lazy imported; pyrefly see this as `object``
            except Exception as cleanup_error:  # noqa: BLE001 - teardown must not raise
                logger.debug(f"Text-encode service VRAM cleanup failed ({cleanup_error})")

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Handle a control message. ``END_PROCESS`` is handled by the base loop."""
        logger.debug(f"Text-encode service received control message: {message.control_flag.name}")
        if message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
            self.unload_models_from_vram()
            return

        if message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            self.unload_models_from_ram()
            return

        if message.control_flag == HordeControlFlag.RELEASE_ALLOCATOR_CACHE:
            self.release_allocator_cache()
            return

        if isinstance(message, HordeTextEncodeControlMessage):
            self._run_text_encode(message)
