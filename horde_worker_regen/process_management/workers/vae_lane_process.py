"""The dedicated VAE lane process (the disaggregated pipeline's VAE-encode/decode stage).

Under pipeline disaggregation a job's source image is VAE-encoded to a start LATENT (img2img/remix
front-end) and its sampled LATENT is VAE-decoded (and post-processed) to final images in this process,
which loads only the VAE. VAE stages are critical-path for every disaggregated job, so they get a lane
that post-processing work can never block, and whose co-residency charge (the tiled-decode spike) is
honest because nothing else is resident in its context.

It owns a hordelib backend (for the VAE stages and the decode's optional post-processing graphs) but
never loads an image-generation checkpoint: its only entry points are the per-stage ``vae_encode``/
``decode`` calls and the post-processing graphs a decode's requested upscale/face-fix runs.
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
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeImageResult,
    HordePostProcessControlMessage,
    HordeProcessState,
    HordeVaeDecodeControlMessage,
    HordeVaeDecodeResultMessage,
    HordeVaeEncodeControlMessage,
    HordeVaeEncodeResultMessage,
    UnsupportedControlMessageError,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcess, HordeProcessType
from horde_worker_regen.utils.oom_signature import is_out_of_memory_text, is_resource_class_exception

if TYPE_CHECKING:
    from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
    from horde_sdk.generation_parameters.image import ImageGenerationParameters
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


class HordeVaeLaneProcess(HordeProcess):
    """The dedicated VAE lane: VAE-encodes/decodes disaggregated stages, loading only the VAE."""

    _horde: HordeLib
    _shared_model_manager: SharedModelManager
    _dry_run: bool

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        device_index: int = 0,
        dry_run: bool = False,
    ) -> None:
        """Initialise the VAE lane process.

        Args:
            process_id (int): The ID of the process (not the OS PID).
            process_message_queue (ProcessQueue): The queue used to send messages to the main process.
            pipe_connection (Connection): Receives ``HordeControlMessage``s from the main process.
            disk_lock (Lock): The lock used when accessing the disk.
            process_launch_identifier (int): The unique identifier for this launch.
            device_index (int, optional): The stable index of the GPU this process is pinned to. Defaults to 0.
            dry_run (bool, optional): Skip the hordelib backend and return plausible stand-in latent/image
                bytes (used by tests and the dry-run worker). Defaults to False.
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
        )

        self.process_type = HordeProcessType.VAE_LANE
        self._dry_run = dry_run
        # This process holds a CUDA context, so its periodic report should sample VRAM (like inference)
        # and unlike the CPU-only safety process, so the parent's per-card free-VRAM view stays fresh.
        self._periodic_report_includes_vram = True

        if dry_run:
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
                logger.info("Initialising HordeLib for the VAE lane")
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

        logger.info("HordeVaeLaneProcess initialised")
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
        """Evict this lane's own resident VAE/post-processing models and cached pool before a retry.

        hordelib's node-level tiling has already tried to fit the stage within the currently committed
        VRAM, so merely emptying the cache would not free room; only unloading the lane's retained models
        gives a retried stage a clean allocator. This recovers the case where a stage OOMs because a prior
        stage's pool is still resident, not because the stage genuinely exceeds the card.
        """
        if self._dry_run:
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
                f"VAE lane hit CUDA out-of-memory for {context}; reclaiming lane VRAM and retrying "
                f"once before faulting: {type(first_error).__name__} {first_error}",
            )
            self._reclaim_own_vram_for_retry()
            return run()

    @staticmethod
    def clear_gc_and_torch_cache() -> None:
        """Clear Python garbage and the active backend's device cache."""
        gc.collect()
        from hordelib.api import clear_accelerator_cache

        clear_accelerator_cache()

    @logger.catch(reraise=True)
    def unload_models_from_vram(self) -> None:
        """Unload VAE/post-processing modules from VRAM and report the refreshed memory sample."""
        if not self._dry_run:
            self._horde.backend.free_vram()
            self.clear_gc_and_torch_cache()

        self.send_process_state_change_message(
            process_state=HordeProcessState.UNLOADED_MODEL_FROM_VRAM,
            info="Unloaded VAE lane models from VRAM",
        )
        self.send_memory_report_message(include_vram=True)
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    @logger.catch(reraise=True)
    def release_allocator_cache(self) -> None:
        """Release the torch allocator's cached free blocks without unloading models, then report memory.

        Empties the caching allocator's reserved-but-unused device blocks so the reservation returns to
        the card while the resident VAE/post-processing modules stay loaded, then reports the refreshed
        memory sample. Deliberately emits no model state change: nothing was unloaded.
        """
        logger.debug("Releasing allocator cache (resident modules stay loaded)")
        if not self._dry_run:
            self.clear_gc_and_torch_cache()
        self.send_memory_report_message(include_vram=True)

    @logger.catch(reraise=True)
    def unload_models_from_ram(self) -> None:
        """Unload VAE/post-processing modules from RAM/VRAM and report the refreshed memory sample."""
        if not self._dry_run:
            self._horde.backend.free_ram()
            self.clear_gc_and_torch_cache()

        self.send_process_state_change_message(
            process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
            info="Unloaded VAE lane models from RAM",
        )
        self.send_memory_report_message(include_vram=True)
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    def _job_generation_parameters(
        self,
        sdk_api_job_info: ImageGenerateJobPopResponse,
    ) -> ImageGenerationParameters:
        """Convert an API job into hordelib generation parameters (shared by the VAE stages)."""
        from horde_sdk.worker.dispatch.ai_horde.image.convert import (
            convert_image_job_pop_response_to_parameters,
        )

        from horde_worker_regen.reference_helper import ensure_offline_reference_manager

        return convert_image_job_pop_response_to_parameters(
            api_response=sdk_api_job_info,
            model_reference_manager=ensure_offline_reference_manager(),
        ).generation_parameters

    def _run_vae_encode(self, message: HordeVaeEncodeControlMessage) -> None:
        """VAE-encode a job's source image to a LATENT (img2img/remix front-end), loading only the VAE."""
        self.send_process_state_change_message(HordeProcessState.POST_PROCESSING, info=f"VAE-encode {message.job_id}")
        time_start = time.time()
        state = GENERATION_STATE.ok
        fault_is_resource_class = False
        fault_reason: str | None = None
        latent_bytes: bytes | None = None
        try:
            if self._dry_run:
                # No backend in dry-run: return an opaque LATENT stand-in. Nothing deserializes it (a
                # dry-run sampler skips sampling entirely), so the fake pipeline still flows end to end.
                latent_bytes = b"dry-run-source-latent"
            else:
                params = self._job_generation_parameters(message.sdk_api_job_info)
                latent_bytes = self._run_with_oom_retry(
                    lambda: self._horde.vae_encode_stage(params),
                    context=f"vae-encode job {message.job_id}",
                )
        except Exception as e:  # noqa: BLE001 - a stage fault is reported, never crashes the lane
            logger.error(f"VAE-encode failed for job {message.job_id}: {type(e).__name__} {e}")
            state = GENERATION_STATE.faulted
            fault_is_resource_class = is_resource_class_exception(e)
            fault_reason = f"{type(e).__name__}: {e}"

        self.process_message_queue.put(
            HordeVaeEncodeResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"VAE-encode for job {message.job_id}",
                time_elapsed=time.time() - time_start,
                job_id=message.job_id,
                latent_bytes=latent_bytes,
                state=state,
                fault_is_resource_class=fault_is_resource_class,
                fault_reason=fault_reason,
            ),
        )
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    def _run_vae_decode(self, message: HordeVaeDecodeControlMessage) -> None:
        """Decode a LATENT to images (loading only the VAE), then run any requested post-processing."""
        self.send_process_state_change_message(HordeProcessState.POST_PROCESSING, info=f"VAE-decode {message.job_id}")
        time_start = time.time()
        state = GENERATION_STATE.ok
        fault_is_resource_class = False
        fault_reason: str | None = None
        job_image_results: list[HordeImageResult] | None = None
        try:
            if self._dry_run:
                # No backend in dry-run: synthesize one valid PNG per requested iteration so the fake
                # pipeline hands real image bytes to the downstream safety/submit flow.
                from horde_worker_regen.process_management.simulation._dummy_images import make_dummy_png_bytes

                n_iter = message.sdk_api_job_info.payload.n_iter or 1
                job_image_results = [HordeImageResult(image_bytes=make_dummy_png_bytes()) for _ in range(n_iter)]
            else:
                params = self._job_generation_parameters(message.sdk_api_job_info)

                def _decode_and_pp() -> list[HordeImageResult]:
                    results, _faults = self._horde.decode_stage(params, latent_bytes=message.latent_bytes)
                    images = [
                        HordeImageResult(image_bytes=r.rawpng.getvalue(), generation_faults=r.faults)
                        for r in results
                        if r.rawpng is not None
                    ]
                    if message.post_processing:
                        pp_message = HordePostProcessControlMessage(
                            job_id=message.job_id,
                            images_bytes=[image_result.image_bytes for image_result in images],
                            post_processing=message.post_processing,
                        )
                        return self._post_process_all_images(pp_message)
                    return images

                job_image_results = self._run_with_oom_retry(
                    _decode_and_pp,
                    context=f"vae-decode job {message.job_id}",
                )
        except Exception as e:  # noqa: BLE001 - a stage fault is reported, never crashes the lane
            logger.error(f"VAE-decode failed for job {message.job_id}: {type(e).__name__} {e}")
            state = GENERATION_STATE.faulted
            fault_is_resource_class = is_resource_class_exception(e)
            fault_reason = f"{type(e).__name__}: {e}"

        self.process_message_queue.put(
            HordeVaeDecodeResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"VAE-decode for job {message.job_id}",
                time_elapsed=time.time() - time_start,
                job_id=message.job_id,
                job_image_results=job_image_results,
                state=state,
                fault_is_resource_class=fault_is_resource_class,
                fault_reason=fault_reason,
            ),
        )
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    @override
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

        if isinstance(message, HordeVaeEncodeControlMessage):
            self._run_vae_encode(message)
            return

        if isinstance(message, HordeVaeDecodeControlMessage):
            self._run_vae_decode(message)
            return

        raise UnsupportedControlMessageError(f"Expected a VAE encode/decode control message, got {type(message)}")

    @override
    def cleanup_for_exit(self) -> None:
        if not self._dry_run:
            self._horde.backend.free_ram()
            self.clear_gc_and_torch_cache()
        return
