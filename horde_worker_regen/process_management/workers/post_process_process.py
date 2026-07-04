"""The dedicated post-processing process.

This process keeps the post-processing models (ESRGAN upscalers, GFPGAN/CodeFormer face-fixers)
resident and runs the post-processing phase of image jobs off the inference processes, so a job's
upscale/face-fix does not contend for VRAM with a fresh generation on the same slot and the models
are not reloaded per job. It also serves the graph-backed alchemy forms (upscale/facefix/
strip_background) that would otherwise run on an inference process.

It owns a hordelib backend (for the post-processing graphs) but never loads an image-generation
checkpoint: its only entry points are the per-operation ``post_process`` calls.
"""

from __future__ import annotations

import gc
import io
import sys
import time

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock
from typing import TYPE_CHECKING, override

import PIL.Image
from horde_sdk.ai_horde_api import GENERATION_STATE
from loguru import logger

from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.ipc.messages import (
    AlchemyFormSpec,
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeImageResult,
    HordePostProcessControlMessage,
    HordePostProcessResultMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcess, HordeProcessType

if TYPE_CHECKING:
    from hordelib.api import HordeLib, SharedModelManager
else:

    class HordeLib:
        """Dummy class to prevent type errors."""

    class SharedModelManager:
        """Dummy class to prevent type errors."""


def _sort_facefixers_last(post_processing: list[str]) -> list[str]:
    """Return the requested post-processors with face-fixers ordered after upscalers.

    Mirrors the inline post-processing order: an upscaler runs on the base image, then a face-fixer
    refines the upscaled result. ``classify_post_processor`` decides which names are face-fixers so an
    unknown name keeps its position rather than being dropped.
    """
    from hordelib.api import PostProcessorKind, classify_post_processor

    def _key(name: str) -> int:
        return 1 if classify_post_processor(name) is PostProcessorKind.facefixer else 0

    return sorted(post_processing, key=_key)


class HordePostProcessProcess(HordeProcess):
    """The dedicated post-processing process."""

    _horde: HordeLib
    _shared_model_manager: SharedModelManager
    _dry_run_skip_post_processing: bool

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        device_index: int = 0,
        dry_run_skip_post_processing: bool = False,
    ) -> None:
        """Initialise the post-processing process.

        Args:
            process_id (int): The ID of the process (not the OS PID).
            process_message_queue (ProcessQueue): The queue used to send messages to the main process.
            pipe_connection (Connection): Receives ``HordeControlMessage``s from the main process.
            disk_lock (Lock): The lock used when accessing the disk.
            process_launch_identifier (int): The unique identifier for this launch.
            device_index (int, optional): The stable index of the GPU this process is pinned to. Defaults to 0.
            dry_run_skip_post_processing (bool, optional): Skip real post-processing and echo the input
                images back. Defaults to False.
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
        )

        self.process_type = HordeProcessType.POST_PROCESS
        self._dry_run_skip_post_processing = dry_run_skip_post_processing
        # This process holds a CUDA context, so its periodic report should sample VRAM (like inference)
        # and unlike the CPU-only safety process, so the parent's per-card free-VRAM view stays fresh.
        self._periodic_report_includes_vram = True

        if dry_run_skip_post_processing:
            logger.info("Dry-run mode: skipping HordeLib/SharedModelManager initialisation")
            self._horde = None  # type: ignore[assignment]
            self._shared_model_manager = None  # type: ignore[assignment]
        else:
            try:
                with logger.catch(reraise=True):
                    from hordelib.api import HordeLib, SharedModelManager
            except Exception as e:
                logger.critical(f"Failed to import HordeLib or SharedModelManager: {type(e).__name__} {e}")
                sys.exit(1)

            try:
                logger.info("Initialising HordeLib for post-processing")
                with logger.catch(reraise=True):
                    self._horde = HordeLib(aggressive_unloading=False)
                    self._shared_model_manager = SharedModelManager(do_not_load_model_mangers=True)
            except Exception as e:
                logger.critical(f"Failed to initialise HordeLib: {type(e).__name__} {e}")
                sys.exit(1)

            # Subprocesses never download model references; the parent owns that and has written the
            # converted files to disk, so reuse an offline reference manager (no per-process fetch).
            from horde_worker_regen.reference_helper import ensure_offline_reference_manager

            ensure_offline_reference_manager()

            SharedModelManager.load_model_managers(
                multiprocessing_lock=self.disk_lock,
                lora_reference_backups=False,
            )

        logger.info("HordePostProcessProcess initialised")
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    def _post_process_one_image(self, image_bytes: bytes, post_processing: list[str]) -> HordeImageResult | None:
        """Run the requested post-processors over a single image, threading each result into the next.

        Returns the post-processed image (encoded as PNG bytes, matching the pre-post-processing format)
        with any faults the operations recorded, or None if no output image survived.
        """
        from horde_sdk.ai_horde_api.apimodels.base import GenMetadataEntry

        current_image = PIL.Image.open(io.BytesIO(image_bytes))
        faults: list[GenMetadataEntry] = []

        for operation in _sort_facefixers_last(post_processing):
            # Upscales can run for many seconds; a heartbeat per operation keeps the parent's liveness
            # view fresh so a legitimately busy post-processing pass is not mistaken for a hung process.
            self.send_heartbeat_message(heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE)
            result = self._horde.post_process(
                {
                    "model": operation,
                    "source_image": current_image,
                },
            )
            if result.image is None:
                logger.error(f"Post-processor produced no image; aborting remaining operations: op={operation}")
                return None
            current_image = result.image
            faults += result.faults

        buffer = io.BytesIO()
        current_image.save(buffer, format="PNG")
        return HordeImageResult(image_bytes=buffer.getvalue(), generation_faults=faults)

    def _run_post_processing(self, message: HordePostProcessControlMessage) -> None:
        """Run an image job's post-processing phase and return the processed images."""
        self.send_process_state_change_message(
            process_state=HordeProcessState.POST_PROCESSING,
            info=f"Post-processing job {message.job_id}",
        )
        self.send_memory_report_message(include_vram=True)

        time_start = time.time()
        state = GENERATION_STATE.ok
        job_image_results: list[HordeImageResult] | None

        if self._dry_run_skip_post_processing:
            job_image_results = [HordeImageResult(image_bytes=image_bytes) for image_bytes in message.images_bytes]
        else:
            processed_images: list[HordeImageResult] = []
            try:
                for image_bytes in message.images_bytes:
                    processed = self._post_process_one_image(image_bytes, message.post_processing)
                    if processed is None:
                        raise RuntimeError("post-processing produced no output image")
                    processed_images.append(processed)
                job_image_results = processed_images
            except Exception as e:
                logger.error(f"Post-processing failed for job {message.job_id}: {type(e).__name__} {e}")
                state = GENERATION_STATE.faulted
                job_image_results = None

        self.process_message_queue.put(
            HordePostProcessResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Post-processing for job {message.job_id}",
                time_elapsed=time.time() - time_start,
                job_id=message.job_id,
                job_image_results=job_image_results,
                state=state,
            ),
        )

        process_state = (
            HordeProcessState.POST_PROCESSING_COMPLETE
            if state == GENERATION_STATE.ok
            else HordeProcessState.POST_PROCESSING_FAILED
        )
        self.send_process_state_change_message(process_state=process_state, info=f"Finished job {message.job_id}")
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    def _run_graph_alchemy(self, form: AlchemyFormSpec) -> None:
        """Run a graph-backed alchemy form (upscale/facefix/strip_background) and report the result.

        The result image is WebP-encoded (quality 95, matching the legacy alchemist) so the main process
        can upload it to R2 without re-encoding.
        """
        from hordelib.api import classify_post_processor

        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_STARTING,
            info=f"Starting alchemy form {form.form} ({form.form_id})",
        )

        time_start = time.time()
        state = GENERATION_STATE.faulted
        image_bytes: bytes | None = None

        try:
            if classify_post_processor(form.form) is None:
                raise ValueError(f"Unknown alchemy form for post-processing process: {form.form}")

            source_image = PIL.Image.open(io.BytesIO(form.source_image_bytes))
            result = self._horde.post_process(
                {
                    "model": form.form,
                    "source_image": source_image,
                },
            )
            if result.image is None:
                raise RuntimeError("Alchemy form produced no image")

            buffer = io.BytesIO()
            result.image.save(buffer, format="WebP", quality=95, method=6)
            image_bytes = buffer.getvalue()
            state = GENERATION_STATE.ok
        except Exception as e:
            logger.error(f"Alchemy form {form.form} ({form.form_id}) failed: {type(e).__name__} {e}")

        self.process_message_queue.put(
            HordeAlchemyResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Alchemy form {form.form} ({form.form_id})",
                time_elapsed=time.time() - time_start,
                form_id=form.form_id,
                form=form.form,
                state=state,
                image_bytes=image_bytes,
            ),
        )

        process_state = (
            HordeProcessState.ALCHEMY_COMPLETE if state == GENERATION_STATE.ok else HordeProcessState.ALCHEMY_FAILED
        )
        self.send_process_state_change_message(
            process_state=process_state,
            info=f"Finished alchemy form {form.form} ({form.form_id})",
        )
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    @staticmethod
    def clear_gc_and_torch_cache() -> None:
        """Clear Python garbage and the active backend's device cache."""
        gc.collect()
        from hordelib.api import clear_accelerator_cache

        clear_accelerator_cache()

    @logger.catch(reraise=True)
    def unload_models_from_vram(self) -> None:
        """Unload post-processing modules from VRAM and report the refreshed memory sample."""
        if not self._dry_run_skip_post_processing:
            self._horde.backend.free_vram()
            self.clear_gc_and_torch_cache()

        self.send_process_state_change_message(
            process_state=HordeProcessState.UNLOADED_MODEL_FROM_VRAM,
            info="Unloaded post-processing models from VRAM",
        )
        self.send_memory_report_message(include_vram=True)
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    @logger.catch(reraise=True)
    def unload_models_from_ram(self) -> None:
        """Unload post-processing modules from RAM/VRAM and report the refreshed memory sample."""
        if not self._dry_run_skip_post_processing:
            self._horde.backend.free_ram()
            self.clear_gc_and_torch_cache()

        self.send_process_state_change_message(
            process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
            info="Unloaded post-processing models from RAM",
        )
        self.send_memory_report_message(include_vram=True)
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        if message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
            self.unload_models_from_vram()
            return

        if message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            self.unload_models_from_ram()
            return

        if isinstance(message, HordePostProcessControlMessage):
            if message.control_flag != HordeControlFlag.START_POST_PROCESS:
                raise ValueError(f"Expected {HordeControlFlag.START_POST_PROCESS}, got {message.control_flag}")
            self._run_post_processing(message)
            return

        if isinstance(message, HordeAlchemyControlMessage):
            if message.control_flag != HordeControlFlag.START_ALCHEMY:
                raise ValueError(f"Expected {HordeControlFlag.START_ALCHEMY}, got {message.control_flag}")
            if self._dry_run_skip_post_processing:
                self.process_message_queue.put(
                    HordeAlchemyResultMessage(
                        process_id=self.process_id,
                        process_launch_identifier=self.process_launch_identifier,
                        info="Dry-run alchemy form",
                        time_elapsed=0.0,
                        form_id=message.form.form_id,
                        form=message.form.form,
                        state=GENERATION_STATE.ok,
                        image_bytes=message.form.source_image_bytes,
                    ),
                )
                self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")
                return
            self._run_graph_alchemy(message.form)
            return

        raise TypeError(f"Expected a post-process or alchemy control message, got {type(message)}")

    @override
    def cleanup_for_exit(self) -> None:
        if not self._dry_run_skip_post_processing:
            self._horde.backend.free_ram()
            self.clear_gc_and_torch_cache()
        return
