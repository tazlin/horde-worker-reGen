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
from collections.abc import Callable
from multiprocessing.synchronize import Lock
from typing import TYPE_CHECKING, TypeVar, override

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
    UnsupportedControlMessageError,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcess, HordeProcessType
from horde_worker_regen.utils.oom_signature import is_out_of_memory_text

if TYPE_CHECKING:
    from hordelib.api import HordeLib, SharedModelManager
else:

    class HordeLib:
        """Dummy class to prevent type errors."""

    class SharedModelManager:
        """Dummy class to prevent type errors."""


_T = TypeVar("_T")


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

    def _post_process_all_images(self, message: HordePostProcessControlMessage) -> list[HordeImageResult]:
        """Post-process every image in the job, raising if any operation yields no output image."""
        processed_images: list[HordeImageResult] = []
        for image_bytes in message.images_bytes:
            processed = self._post_process_one_image(image_bytes, message.post_processing)
            if processed is None:
                raise RuntimeError("post-processing produced no output image")
            processed_images.append(processed)
        return processed_images

    def _reclaim_own_vram_for_retry(self) -> None:
        """Evict this lane's own resident post-processing models and cached pool before a retry.

        hordelib's node-level tiling has already tried to fit the chain within the currently committed
        VRAM, so merely emptying the cache would not free room; only unloading the lane's retained models
        gives a retried chain a clean allocator. This recovers the case where a chain OOMs because a prior
        chain's pool is still resident, not because the chain genuinely exceeds the card.
        """
        if self._dry_run_skip_post_processing:
            return
        self._horde.backend.free_vram()
        self.clear_gc_and_torch_cache()
        self.send_memory_report_message(include_vram=True)

    def _run_with_oom_retry(self, run: Callable[[], _T], *, context: str) -> _T:
        """Run ``run``; on a CUDA out-of-memory failure, reclaim the lane's VRAM and retry it once.

        The OOM reaches the lane as a generic error wrapping the CUDA text (ComfyUI swallows the typed
        error), so it is recognized by fingerprint. A non-OOM failure, or a second OOM after reclaiming,
        propagates to the caller's fault handling unchanged.
        """
        try:
            return run()
        except Exception as first_error:
            if not is_out_of_memory_text(f"{type(first_error).__name__}: {first_error}"):
                raise
            logger.warning(
                f"Post-processing hit CUDA out-of-memory for {context}; reclaiming lane VRAM and retrying "
                f"once before faulting: {type(first_error).__name__} {first_error}",
            )
            self._reclaim_own_vram_for_retry()
            return run()

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
        fault_is_resource_class = False
        fault_reason: str | None = None

        if self._dry_run_skip_post_processing:
            job_image_results = [HordeImageResult(image_bytes=image_bytes) for image_bytes in message.images_bytes]
        else:
            try:
                job_image_results = self._run_with_oom_retry(
                    lambda: self._post_process_all_images(message),
                    context=f"job {message.job_id}",
                )
            except Exception as e:
                fault_reason = f"{type(e).__name__}: {e}"
                fault_is_resource_class = is_out_of_memory_text(fault_reason)
                logger.error(f"Post-processing failed for job {message.job_id}: {fault_reason}")
                state = GENERATION_STATE.faulted
                job_image_results = None

        self.process_message_queue.put(
            HordePostProcessResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=fault_reason or f"Post-processing for job {message.job_id}",
                time_elapsed=time.time() - time_start,
                job_id=message.job_id,
                job_image_results=job_image_results,
                state=state,
                fault_is_resource_class=fault_is_resource_class,
                fault_reason=fault_reason,
            ),
        )

        process_state = (
            HordeProcessState.POST_PROCESSING_COMPLETE
            if state == GENERATION_STATE.ok
            else HordeProcessState.POST_PROCESSING_FAILED
        )
        self.send_process_state_change_message(process_state=process_state, info=f"Finished job {message.job_id}")
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    def _run_alchemy_form_bytes(self, form: AlchemyFormSpec) -> bytes:
        """Run a single graph-backed alchemy form and return its WebP-encoded result bytes.

        Raises on an unknown form or a graph that yields no image, so the caller's OOM retry and fault
        handling see a clean exception. Encoding matches the legacy alchemist (WebP quality 95) so the main
        process can upload the result to R2 without re-encoding.
        """
        from hordelib.api import classify_post_processor

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
        return buffer.getvalue()

    def _run_graph_alchemy(self, form: AlchemyFormSpec) -> None:
        """Run a graph-backed alchemy form (upscale/facefix/strip_background) and report the result.

        The result image is WebP-encoded (quality 95, matching the legacy alchemist) so the main process
        can upload it to R2 without re-encoding.
        """
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_STARTING,
            info=f"Starting alchemy form {form.form} ({form.form_id})",
        )

        time_start = time.time()
        state = GENERATION_STATE.faulted
        image_bytes: bytes | None = None

        try:
            image_bytes = self._run_with_oom_retry(
                lambda: self._run_alchemy_form_bytes(form),
                context=f"alchemy form {form.form} ({form.form_id})",
            )
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
    def release_allocator_cache(self) -> None:
        """Release the torch allocator's cached free blocks without unloading models, then report memory.

        Empties the caching allocator's reserved-but-unused device blocks so the reservation returns to
        the card while the resident post-processing modules stay loaded, then reports the refreshed memory
        sample. Deliberately emits no model state change: nothing was unloaded.
        """
        logger.debug("Releasing allocator cache (resident modules stay loaded)")
        if not self._dry_run_skip_post_processing:
            self.clear_gc_and_torch_cache()
        self.send_memory_report_message(include_vram=True)

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

    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        if message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
            self.unload_models_from_vram()
            return

        if message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            self.unload_models_from_ram()
            return

        if message.control_flag == HordeControlFlag.RELEASE_ALLOCATOR_CACHE:
            self.release_allocator_cache()
            return

        if isinstance(message, HordePostProcessControlMessage):
            if message.control_flag != HordeControlFlag.START_POST_PROCESS:
                raise UnsupportedControlMessageError(
                    f"Expected {HordeControlFlag.START_POST_PROCESS}, got {message.control_flag}",
                )
            self._run_post_processing(message)
            return

        if isinstance(message, HordeAlchemyControlMessage):
            if message.control_flag != HordeControlFlag.START_ALCHEMY:
                raise UnsupportedControlMessageError(
                    f"Expected {HordeControlFlag.START_ALCHEMY}, got {message.control_flag}",
                )
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

        raise UnsupportedControlMessageError(
            f"Expected a post-process or alchemy control message, got {type(message)}",
        )

    @override
    def cleanup_for_exit(self) -> None:
        if not self._dry_run_skip_post_processing:
            self._horde.backend.free_ram()
            self.clear_gc_and_torch_cache()
        return
