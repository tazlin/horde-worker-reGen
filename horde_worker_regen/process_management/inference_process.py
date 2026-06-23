"""Contains the classes to form an inference process, which generate images."""

from __future__ import annotations

import base64
import contextlib
import gc
import io
import os
import sys
import threading
import time
from dataclasses import dataclass, field

from horde_worker_regen.consts import BASE_LORA_DOWNLOAD_TIMEOUT, EXTRA_LORA_DOWNLOAD_TIMEOUT
from horde_worker_regen.process_management.lora_disk_guard import (
    configured_lora_budget_mb_from_env,
    constrain_lora_cache_to_disk,
    lora_disk_floor_mb_from_env,
)

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock, Semaphore
from typing import TYPE_CHECKING, override

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import (
    GenMetadataEntry,
    ImageGenerateJobPopResponse,
)
from loguru import logger

from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.horde_process import HordeProcess
from horde_worker_regen.process_management.messages import (
    AlchemyFormSpec,
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeAuxModelStateChangeMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeControlModelMessage,
    HordeDownloadMetricsMessage,
    HordeHeartbeatType,
    HordeImageResult,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeJobMetricsMessage,
    HordeModelStateChangeMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    ModelLoadState,
)

AUX_DOWNLOAD_HEARTBEAT_INTERVAL_SECONDS = 5.0
"""How often a child reports liveness while blocked in an ad-hoc AUX-model download."""

if TYPE_CHECKING:
    from hordelib.api import HordeLib, ProgressReport, ResultingImageReturn, SharedModelManager
else:
    # Create a dummy class to prevent type errors at runtime
    # This is so we can defer the import of these classes until runtime
    class HordeLib:  # noqa
        pass

    class SharedModelManager:  # noqa
        pass

    class ProgressReport:  # noqa
        pass


@dataclass
class _DryRunResultingImage:
    """Duck-type stand-in for hordelib's `ResultingImageReturn` used in dry-run mode."""

    rawpng: io.BytesIO
    faults: list[GenMetadataEntry] = field(default_factory=list)


class HordeInferenceProcess(HordeProcess):
    """Represents an inference process, which generates images."""

    _periodic_report_includes_vram: bool = True
    """Inference processes own the GPU, so their interval memory report samples device-wide VRAM."""

    _inference_semaphore: Semaphore
    """A semaphore used to limit the number of concurrent inference jobs."""

    _vae_decode_semaphore: Semaphore

    _horde: HordeLib
    """The HordeLib instance used by this process. It is not shared between processes."""
    _shared_model_manager: SharedModelManager
    """The SharedModelManager instance used by this process. It is not shared between processes (despite the name)."""

    _active_model_name: str | None = None
    """The name of the currently active model. Note that other models may be loaded in RAM or VRAM."""
    _aux_model_lock: Lock

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        inference_semaphore: Semaphore,
        vae_decode_semaphore: Semaphore,
        aux_model_lock: Lock,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        device_index: int = 0,
        dry_run_skip_inference: bool = False,
        dry_run_inference_delay: float = 1.0,
        gpu_sampling_lease: Semaphore | None = None,
    ) -> None:
        """Initialise the HordeInferenceProcess.

        Args:
            process_id (int): The ID of the process. This is not the same as the process PID.
            process_message_queue (ProcessQueue): The queue the main process uses to receive messages from all worker \
                processes.
            pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
            inference_semaphore (Semaphore): A semaphore used to limit the number of concurrent inference jobs.
            vae_decode_semaphore (Semaphore): A semaphore used to limit the number of concurrent VAE decode jobs.
            aux_model_lock (Lock): A lock used to prevent multiple processes from downloading auxiliary models at the \
            disk_lock (Lock): A lock used to prevent multiple processes from accessing disk at the same time.
            process_launch_identifier (int): The identifier for the process launch.
            dry_run_skip_inference (bool, optional): Skip real inference and return a dummy image. Defaults to False.
            dry_run_inference_delay (float, optional): Seconds to sleep when dry-run inference is active. \
                Defaults to 1.0.
            gpu_sampling_lease (Semaphore | None, optional): Shared lease registered with hordelib to \
                serialize the GPU denoising loop across processes. None disables coordination. Defaults to None.
            device_index (int, optional): The stable index of the GPU this process is pinned to. Defaults to 0.
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
        )

        self._aux_model_lock = aux_model_lock
        self._dry_run_skip_inference = dry_run_skip_inference
        self._dry_run_inference_delay = dry_run_inference_delay

        self._inference_semaphore = inference_semaphore
        self._vae_decode_semaphore = vae_decode_semaphore

        if dry_run_skip_inference:
            logger.info("Dry-run mode: skipping HordeLib/SharedModelManager initialisation")
            self._horde = None  # type: ignore[assignment]
            self._shared_model_manager = None  # type: ignore[assignment]
        else:
            # We import these here to guard against potentially importing them in the main process
            # which would create shared objects, potentially causing issues
            try:
                with logger.catch(reraise=True):
                    from hordelib.api import HordeLib, SharedModelManager
            except Exception as e:
                logger.critical(f"Failed to import HordeLib or SharedModelManager: {type(e).__name__} {e}")
                sys.exit(1)

            try:
                logger.info("Initialising HordeLib")
                with logger.catch(reraise=True):
                    self._horde = HordeLib(
                        comfyui_callback=self._comfyui_callback,
                        aggressive_unloading=True,
                    )
                    self._shared_model_manager = SharedModelManager(do_not_load_model_mangers=True)
            except Exception as e:
                logger.critical(f"Failed to initialise HordeLib: {type(e).__name__} {e}")
                sys.exit(1)

            if gpu_sampling_lease is not None:
                # Coordinate the GPU denoising loop across inference processes: this process
                # samples only while holding the shared lease, but stages its pipeline (model
                # load, prompt encode) freely, so the GPU stays busy back-to-back across jobs.
                from hordelib.api import set_gpu_sampling_lease

                set_gpu_sampling_lease(gpu_sampling_lease)
                logger.info("Registered GPU sampling lease for cross-process pipelining")

            # Subprocesses never download model references: the parent process owns downloading and
            # has already written the converted files to disk. Pre-build an offline reference manager
            # so hordelib reuses it (instead of forcing a per-subprocess network fetch/convert).
            from horde_worker_regen.reference_helper import ensure_offline_reference_manager

            ensure_offline_reference_manager()

            SharedModelManager.load_model_managers(
                multiprocessing_lock=self.disk_lock,
                # Reference saves are coordinated by this process under disk_lock; the lora
                # manager's per-process backup copies are unnecessary churn.
                lora_reference_backups=False,
            )

            if SharedModelManager.manager.compvis is None:
                logger.critical("Failed to initialise SharedModelManager")
                self.send_process_state_change_message(
                    process_state=HordeProcessState.PROCESS_ENDED,
                    info="Failed to initialise compvis in SharedModelManager",
                )
                sys.exit(1)

            if len(SharedModelManager.manager.compvis.available_models) == 0:
                # A model may have only just landed via the background download process; re-scan the
                # on-disk database a few times before giving up, so we do not hard-crash on a race
                # with a just-completed download. The main process only starts inference once at least
                # one model is present, so reaching this branch at all is already unusual.
                for _ in range(5):
                    time.sleep(2)
                    SharedModelManager.manager.compvis.load_model_database()
                    if len(SharedModelManager.manager.compvis.available_models) > 0:
                        break
                else:
                    logger.critical("No models available in SharedModelManager")
                    self.send_process_state_change_message(
                        process_state=HordeProcessState.PROCESS_ENDED,
                        info="No models available in SharedModelManager",
                    )
                    sys.exit(1)

        logger.info("HordeInferenceProcess initialised")

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    def _comfyui_callback(self, label: str, data: dict, _id: str) -> None:  # pyrefly: ignore[implicit-any-type-argument] - we don't control the type signature of this callback
        self.send_heartbeat_message(heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE)

    @override
    def get_vram_usage_mb(self) -> int:
        """Return VRAM used, or 0 in dry-run mode where torch is unavailable."""
        if self._dry_run_skip_inference:
            return 0
        return super().get_vram_usage_mb()

    @override
    def get_vram_total_mb(self) -> int:
        """Return total VRAM, or 0 in dry-run mode where torch is unavailable."""
        if self._dry_run_skip_inference:
            return 0
        return super().get_vram_total_mb()

    @override
    def send_memory_report_message(self, include_vram: bool = False) -> bool:
        """Send a memory report message to the main process.

        Args:
            include_vram (bool, optional): Whether or not to include VRAM usage in the report. Defaults to False.

        Returns:
            bool: Whether or not the message was sent successfully.
        """
        if not super().send_memory_report_message(include_vram=include_vram):
            self._end_process = True

        return not self._end_process

    @logger.catch(reraise=True)
    def on_horde_model_state_change(
        self,
        horde_model_name: str,
        process_state: HordeProcessState,
        horde_model_state: ModelLoadState,
        time_elapsed: float | None = None,
    ) -> None:
        """Update the main process with the current process state and model state.

        Args:
            horde_model_name (str): The name of the model.
            process_state (HordeProcessState): The state of the process.
            horde_model_state (ModelLoadState): The state of the model.
            time_elapsed (float | None, optional): The time elapsed during the last operation, if applicable. \
                Defaults to None.
        """
        model_update_message = HordeModelStateChangeMessage(
            process_state=process_state,
            process_id=self.process_id,
            process_launch_identifier=self.process_launch_identifier,
            info=f"Model {horde_model_name} {horde_model_state.name}",
            horde_model_name=horde_model_name,
            horde_model_state=horde_model_state,
            time_elapsed=time_elapsed,
        )
        self.process_message_queue.put(model_update_message)

        self.send_memory_report_message(include_vram=True)

    def download_callback(
        self,
        downloaded_bytes: int,
        total_bytes: int,
    ) -> None:
        """Handle the callback for progress when a model is being downloaded.

        Args:
            downloaded_bytes (int): The number of bytes downloaded so far.
            total_bytes (int): The total number of bytes to download.
        """
        # TODO
        if downloaded_bytes % (total_bytes / 20) == 0:
            self.send_process_state_change_message(
                process_state=HordeProcessState.DOWNLOADING_MODEL,
                info=f"Downloading model ({downloaded_bytes} / {total_bytes})",
            )

    def download_model(self, horde_model_name: str) -> None:
        """Download a model as defined in the horde model reference.

        Args:
            horde_model_name (str): The name of the model to download.\
        """
        # TODO
        self.send_process_state_change_message(
            process_state=HordeProcessState.DOWNLOADING_MODEL,
            info=f"Downloading model {horde_model_name}",
        )

        if self._shared_model_manager.manager.is_model_available(horde_model_name):
            logger.info(f"Model {horde_model_name} already downloaded")

        time_start = time.time()

        success = self._shared_model_manager.manager.download_model(horde_model_name, self.download_callback)

        if success:
            self.send_process_state_change_message(
                process_state=HordeProcessState.DOWNLOAD_COMPLETE,
                info=f"Downloaded model {horde_model_name}",
                time_elapsed=time.time() - time_start,
            )

        self.on_horde_model_state_change(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            horde_model_name=horde_model_name,
            horde_model_state=ModelLoadState.ON_DISK,
        )

    @logger.catch(reraise=True)
    def download_aux_models(self, job_info: ImageGenerateJobPopResponse) -> float | None:
        """Download auxiliary models required for the job.

        Args:
            job_info (ImageGenerateJobPopResponse): The job to download auxiliary models for.

        Returns:
            float | None: The time elapsed during downloading, or None if no models were downloaded.
        """
        # Not a plain ``with self._aux_model_lock:`` because the bound block-exit release can raise
        # when the supervisor has force-released the shared lock out from under us; see
        # ``_release_aux_model_lock`` for why that over-release is benign and must not be fatal.
        self._aux_model_lock.acquire()
        try:
            time_start = time.time()

            lora_manager = self._shared_model_manager.manager.lora
            if lora_manager is None:
                raise RuntimeError("Failed to load LORA model manager")

            loras = job_info.payload.loras or []
            if not loras:
                logger.info("No auxiliary models to download")
                return None

            # Publish the busy state and start the liveness heartbeat *before* any blocking drain. Both
            # the in-flight-download drains below (``reset_adhoc_cache`` and the already-available-LoRA
            # ``wait_for_downloads``) are unbounded and can stall indefinitely on a wedged background
            # download (e.g. one retrying ENOSPC on a full disk). Until the parent has seen
            # DOWNLOADING_AUX_MODEL it still reads this slot as WAITING_FOR_JOB (``can_accept_job``), so
            # the orphaned-job watchdog punts the in-progress job; without a heartbeat the same stall
            # reads as a hung process. Either verdict escalates to a Save-our-ship soft reset (an
            # observed disk-full recovery storm). Starting both signals up front keeps the slot visibly
            # busy-and-alive for the whole aux phase, however long a drain blocks.
            self.send_aux_model_message(
                process_state=HordeProcessState.DOWNLOADING_AUX_MODEL,
                info="Resolving auxiliary models",
                time_elapsed=0.0,
                job_info=job_info,
            )
            self.send_heartbeat_message(HordeHeartbeatType.OTHER)
            aux_heartbeat_stop, aux_heartbeat_thread = self._start_aux_download_heartbeat_thread()

            performed_a_download = False
            try:
                try:
                    lora_manager.load_model_database()
                    lora_manager.reset_adhoc_cache()
                except Exception as e:
                    logger.error(f"Failed to reset adhoc loras: {type(e).__name__} {e}")

                # Make room before fetching: shrink the cache to fit free space and evict
                # least-recently-used ad-hoc LoRAs so this job's LoRAs can be written without
                # pushing the volume past its floor (and into ENOSPC failures).
                self._enforce_lora_disk_floor(lora_manager)

                time_to_wait_for_downloads = 0
                for lora_entry in loras:
                    # Already on disk — nothing to fetch, but still drain any in-flight downloads. Bound
                    # the drain by at least the base LoRA budget: a bare 0 (the accumulator's value before
                    # any fetch this job) reaches the manager as "wait forever", which a wedged background
                    # download (e.g. one retrying a full disk) would turn into an unbounded stall.
                    if lora_manager.is_model_available(lora_entry.name):
                        logger.info(f"Model {lora_entry.name} already downloaded")
                        try:
                            lora_manager.wait_for_downloads(
                                max(time_to_wait_for_downloads, BASE_LORA_DOWNLOAD_TIMEOUT)
                            )
                        except Exception as e:
                            logger.error(f"Failed to wait for downloads: {type(e).__name__} {e}")
                        continue

                    # --- Model needs downloading ---
                    performed_a_download = True
                    lora_manager.fetch_adhoc_lora(lora_entry.name, timeout=None, is_version=lora_entry.is_version)
                    time_to_wait_for_downloads = (
                        BASE_LORA_DOWNLOAD_TIMEOUT
                        if time_to_wait_for_downloads == 0
                        else time_to_wait_for_downloads + EXTRA_LORA_DOWNLOAD_TIMEOUT
                    )
                    try:
                        lora_manager.wait_for_downloads(time_to_wait_for_downloads)
                    except Exception as e:
                        logger.error(f"Failed to wait for downloads: {type(e).__name__} {e}")
            finally:
                aux_heartbeat_stop.set()
                aux_heartbeat_thread.join(timeout=1.0)

            time_elapsed = round(time.time() - time_start, 2)
            lora_manager.save_reference_to_disk()
            self._send_download_metrics_if_any()

            if performed_a_download:
                logger.info(f"Downloaded auxiliary models in {time_elapsed} seconds")
                return time_elapsed

            logger.info("No auxiliary models downloaded")
            return None
        finally:
            self._release_aux_model_lock()

    def _release_aux_model_lock(self) -> None:
        """Release the shared aux-model lock, tolerating a benign supervisor-forced over-release.

        ``_aux_model_lock`` is a single *bounded* multiprocessing lock created once in the manager and
        shared by every inference child and the supervisor. When the supervisor replaces this slot it
        reclaims that lock on our behalf (``HordeProcessLifecycleManager._release_held_primitives``).
        If that reclaim lands while we are still inside the critical section, our own release pushes the
        bounded lock past its ceiling and raises "released too many times". The protected work is
        already finished and the lock is genuinely free, so the over-release is benign: swallowing it
        keeps a slow-but-alive aux download from tearing the whole inference process down (an observed
        crash-loop that re-loaded the model on every LoRA job under supervisor pressure). This mirrors
        the supervisor side, which already swallows the symmetric ``ValueError`` when it force-releases.
        """
        with contextlib.suppress(ValueError):
            self._aux_model_lock.release()

    def _enforce_lora_disk_floor(self, lora_manager: object) -> None:
        """Constrain the ad-hoc LoRA cache to the disk floor before fetching this job's LoRAs.

        Shrinks the effective ad-hoc budget to fit free space and evicts least-recently-used ad-hoc
        LoRAs to make room. When even evicting everything cannot clear the floor, the surplus is
        non-LoRA data: a prominent warning is logged and new LoRA downloads will be skipped (the main
        process independently stops advertising LoRA support once it sees the same unsolvable state).
        """
        floor_mb = lora_disk_floor_mb_from_env(os.getenv)
        if floor_mb <= 0:
            return
        configured_budget_mb = configured_lora_budget_mb_from_env(os.getenv)
        try:
            result = constrain_lora_cache_to_disk(
                lora_manager,  # type: ignore[arg-type]  # structural LoraCacheManager; avoids a hordelib import here
                floor_mb=floor_mb,
                configured_budget_mb=configured_budget_mb,
            )
        except Exception as guard_error:  # noqa: BLE001 - the guard must never break the download path
            logger.error(f"LoRA disk guard failed: {type(guard_error).__name__} {guard_error}")
            return

        if result.acted:
            logger.info(
                f"LoRA disk guard: free {result.free_mb_before:.0f} -> {result.free_mb_after:.0f} MB "
                f"(floor {floor_mb:.0f} MB), evicted {result.evicted_count} ad-hoc LoRA(s), "
                f"ad-hoc budget {result.budget_mb_before} -> {result.budget_mb_after} MB",
            )
        if not result.solved and result.free_mb_after is not None:
            logger.warning(
                "LoRA cache volume is below its free-space floor and eviction could not clear it "
                f"({result.free_mb_after:.0f} MB free < {floor_mb:.0f} MB floor). New LoRA downloads "
                "will be skipped until disk space is freed.",
            )

    @logger.catch(reraise=True)
    def preload_model(
        self,
        horde_model_name: str,
        will_load_loras: bool,
        seamless_tiling_enabled: bool,
        job_info: ImageGenerateJobPopResponse,
    ) -> None:
        """Preload a model into RAM.

        Args:
            horde_model_name (str): The name of the model to preload.
            will_load_loras (bool): Whether or not the model will be loaded into VRAM.
            seamless_tiling_enabled (bool): Whether or not seamless tiling is enabled.
            job_info (ImageGenerateJobPopResponse): The job to preload the model for.
        """
        logger.debug(f"Currently active model is {self._active_model_name}. Requested model is {horde_model_name}")

        if self._active_model_name == horde_model_name:
            return

        if self._is_busy:
            logger.warning("Cannot preload model while busy")

        if not self._dry_run_skip_inference:
            self.clear_gc_and_torch_cache()

        logger.debug(f"Preloading model {horde_model_name}")

        if self._active_model_name is not None and self._active_model_name != horde_model_name:
            self.on_horde_model_state_change(
                process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
                horde_model_name=self._active_model_name,
                horde_model_state=ModelLoadState.ON_DISK,
            )

        download_time = self.download_aux_models(job_info)

        if download_time is not None:
            self.send_aux_model_message(
                process_state=HordeProcessState.DOWNLOAD_AUX_COMPLETE,
                info="Downloaded auxiliary models",
                time_elapsed=download_time,
                job_info=job_info,
            )

        self.on_horde_model_state_change(
            process_state=HordeProcessState.PRELOADING_MODEL,
            horde_model_name=horde_model_name,
            horde_model_state=ModelLoadState.LOADING,
        )

        time_start = time.time()

        if not self._dry_run_skip_inference:
            with contextlib.nullcontext():  # self.disk_lock:
                try:
                    self._horde.preload_model(
                        horde_model_name,
                        will_load_loras=will_load_loras,
                        seamless_tiling_enabled=seamless_tiling_enabled,
                    )
                except Exception as preload_error:
                    # A load failure is a property of the *model* (an unsupported/corrupt checkpoint the
                    # backend cannot load), not of this process, but the backend may have left torch/ComfyUI
                    # in an indeterminate state, so the process still ends after reporting. Naming the model in
                    # a FAILED state lets the parent quarantine that specific model after repeated failures
                    # rather than mistaking a deterministically-unloadable model for a sick slot and churning
                    # the pool. The control-message handler logs and ends the process when this re-raises.
                    logger.error(
                        f"Failed to preload model {horde_model_name}: {type(preload_error).__name__} {preload_error}",
                    )
                    self.on_horde_model_state_change(
                        process_state=HordeProcessState.PRELOADING_FAILED,
                        horde_model_name=horde_model_name,
                        horde_model_state=ModelLoadState.FAILED,
                    )
                    raise

        logger.info(f"Preloaded model {horde_model_name}")
        self._active_model_name = horde_model_name
        self.on_horde_model_state_change(
            process_state=HordeProcessState.PRELOADED_MODEL,
            horde_model_name=horde_model_name,
            horde_model_state=ModelLoadState.LOADED_IN_RAM,
            time_elapsed=time.time() - time_start,
        )

        self.send_memory_report_message(include_vram=True)

    _is_busy: bool = False

    _start_inference_time: float = 0.0

    _in_post_processing: bool = False

    _current_job_inference_steps_complete: bool = False
    _vae_lock_was_acquired: bool = False
    _inference_slot_released: bool = False
    _post_processing_memory_report_sent: bool = False

    _last_job_inference_rate: str | None = None
    _last_inference_error: str | None = None
    """Summary of the exception that failed the current job's inference, or None if it succeeded.

    Surfaced as the faulted result's ``info`` so the main process can both log a real reason (previously
    a faulted result carried only the empty rate string) and classify a resource/OOM failure for retry."""

    def _send_inference_memory_report(self) -> None:
        """Send an inference-path VRAM report and reset the periodic report throttle."""
        self._last_periodic_memory_report_time = time.time()
        self.send_memory_report_message(include_vram=True)

    def _release_inference_slot(self) -> None:
        """Release this job's sampling-concurrency slot, at most once per job.

        The slot is freed the moment sampling finishes (before VAE decode) so a queued job can
        begin sampling on the GPU while this job decodes VAE, rather than the slot sitting idle
        through VAE and result hand-off. This mirrors the long-standing early release at the
        post-processing boundary, extended to the sampling->VAE boundary so plain (no-PP) jobs
        overlap too. Idempotent: the post-sampling release and the ``finally`` cleanup both call
        this, so the semaphore can never be released twice (which would over-subscribe it and
        admit more concurrent sampling than ``max_threads``).
        """
        if self._inference_slot_released:
            return
        self._inference_slot_released = True
        try:
            self._inference_semaphore.release()
            logger.debug("Released inference semaphore (sampling slot freed for the next job)")
        except Exception as e:
            logger.error(f"Failed to release inference semaphore: {type(e).__name__} {e}")

    def progress_callback(
        self,
        progress_report: ProgressReport,
    ) -> None:
        """Handle progress updates from the HordeLib instance.

        Args:
            progress_report (ProgressReport): The progress report from the HordeLib instance.
        """
        from hordelib.api import ComfyUIProgressUnit, ProgressState, log_free_ram

        if progress_report.hordelib_progress_state == ProgressState.post_processing or (
            self._in_post_processing and progress_report.hordelib_progress_state == ProgressState.progress
        ):
            self.send_process_state_change_message(
                process_state=HordeProcessState.INFERENCE_POST_PROCESSING,
                info="Post Processing",
                time_elapsed=time.time() - self._start_inference_time,
            )
            self._in_post_processing = True
            self._release_inference_slot()
            if not self._post_processing_memory_report_sent:
                self._post_processing_memory_report_sent = True
                self._send_inference_memory_report()

        if self._current_job_inference_steps_complete:
            if not self._vae_lock_was_acquired:
                self._vae_lock_was_acquired = True
                # Sampling is done; free the slot so the next job can sample on the GPU while we
                # wait our turn for and then perform the (VRAM-serialized) VAE decode.
                self._release_inference_slot()
                self._vae_decode_semaphore.acquire()
                log_free_ram()
                logger.debug("Acquired VAE decode semaphore")
                self._send_inference_memory_report()

            self.send_heartbeat_message(heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE)
            self._maybe_send_periodic_memory_report()
            return

        if progress_report.comfyui_progress is not None and progress_report.comfyui_progress.current_step == (
            progress_report.comfyui_progress.total_steps
        ):
            self.send_heartbeat_message(heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE)
            self._current_job_inference_steps_complete = True
            self._send_inference_memory_report()
            logger.debug("Current job inference steps complete")
        elif progress_report.comfyui_progress is not None and progress_report.comfyui_progress.current_step > 0:
            warning = None

            if progress_report.comfyui_progress.rate_unit == ComfyUIProgressUnit.SECONDS_PER_ITERATION and (
                progress_report.comfyui_progress.rate > 2.5 and progress_report.comfyui_progress.current_step > 1
            ):
                warning = (
                    f"{progress_report.comfyui_progress.rate} seconds *per iteration* for step "
                    f"{progress_report.comfyui_progress.current_step}/{progress_report.comfyui_progress.total_steps} "
                    f"for model {self._active_model_name}. "
                    "These are the typical expected speeds: "
                    "SD15: >4 it/s, SDXL >2 it/s, Flux >0.5 it/s. "
                    "If you see this message for most jobs, consider using fewer threads, adjusting the batch size, "
                    "removing the model type triggering this message or turning off other features."
                )

            self._last_job_inference_rate = (
                f"{progress_report.comfyui_progress.rate:.2f} "
                f"{progress_report.comfyui_progress.rate_unit.name.lower().replace('_', ' ')}"
            )

            # Normalize the rate to iterations/second regardless of how it was reported
            # (-1.0 means not yet known and is passed through).
            rate = progress_report.comfyui_progress.rate
            if progress_report.comfyui_progress.rate_unit == ComfyUIProgressUnit.SECONDS_PER_ITERATION:
                rate = 1.0 / rate if rate > 0 else -1.0

            self.send_heartbeat_message(
                heartbeat_type=HordeHeartbeatType.INFERENCE_STEP,
                process_warning=warning,
                percent_complete=progress_report.comfyui_progress.percent,
                current_step=progress_report.comfyui_progress.current_step,
                total_steps=progress_report.comfyui_progress.total_steps,
                iterations_per_second=rate,
            )
        else:
            self.send_heartbeat_message(heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE)

        self._maybe_send_periodic_memory_report()

    def _send_job_metrics_message(self, job_id: str, *, is_alchemy: bool = False) -> None:
        """Snapshot hordelib's per-job metrics and forward them to the main process.

        Sent even in dry-run mode (with an empty snapshot) so the message flow is
        identical with and without real inference.
        """
        try:
            from hordelib.api import get_metrics_collector

            self.process_message_queue.put(
                HordeJobMetricsMessage(
                    process_id=self.process_id,
                    process_launch_identifier=self.process_launch_identifier,
                    info=f"Job metrics for {job_id}",
                    job_id=job_id,
                    is_alchemy=is_alchemy,
                    phase_metrics=get_metrics_collector().snapshot_and_reset_job(),
                ),
            )
        except Exception as e:
            # Metrics must never take down a job that otherwise succeeded.
            logger.warning(f"Failed to send job metrics: {type(e).__name__} {e}")

    def _send_download_metrics_if_any(self) -> None:
        """Forward any ad-hoc download events hordelib observed since the last drain."""
        try:
            from hordelib.api import get_metrics_collector

            events = get_metrics_collector().drain_download_events()
            if not events:
                return

            self.process_message_queue.put(
                HordeDownloadMetricsMessage(
                    process_id=self.process_id,
                    process_launch_identifier=self.process_launch_identifier,
                    info=f"{len(events)} download event(s)",
                    events=events,
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to send download metrics: {type(e).__name__} {e}")

    def _start_aux_download_heartbeat_thread(self) -> tuple[threading.Event, threading.Thread]:
        """Start a short-lived liveness loop for blocking AUX-model downloads.

        The LoRA manager's ad-hoc download path can block inside ``fetch_adhoc_lora`` /
        ``wait_for_downloads`` without returning to the child process main loop. Without this loop, the
        parent and TUI only see the initial ``DOWNLOADING_AUX_MODEL`` state change until the download
        completes, so a healthy WAN transfer can look silent. These are ``OTHER`` heartbeats, not
        ``INFERENCE_STEP`` heartbeats, so mid-sampling hang detection remains unchanged.
        """
        stop_event = threading.Event()

        def _heartbeat_loop() -> None:
            while not stop_event.wait(AUX_DOWNLOAD_HEARTBEAT_INTERVAL_SECONDS):
                self.send_heartbeat_message(HordeHeartbeatType.OTHER)
                self._send_download_metrics_if_any()

        thread = threading.Thread(
            target=_heartbeat_loop,
            name=f"horde-aux-download-heartbeat-{self.process_id}",
            daemon=True,
        )
        thread.start()
        return stop_event, thread

    def start_inference(self, job_info: ImageGenerateJobPopResponse) -> list[ResultingImageReturn] | None:
        """Start an inference job in the HordeLib instance.

        Args:
            job_info (ImageGenerateJobPopResponse): The job to start inference on.

        Returns:
            list[Image] | None: The generated images, or None if inference failed.
        """
        logger.info("Checking if too many inference jobs are already running...")
        self._inference_semaphore.acquire()
        logger.info("Acquired inference semaphore.")
        self._is_busy = True
        self._current_job_inference_steps_complete = False
        self._inference_slot_released = False
        self._vae_lock_was_acquired = False
        self._post_processing_memory_report_sent = False
        self._last_job_inference_rate = None
        self._last_inference_error = None

        try:
            self.send_heartbeat_message(heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE)
            logger.info(f"Starting inference for job(s) {job_info.ids}")
            esi_count = len(job_info.extra_source_images) if job_info.extra_source_images is not None else 0
            logger.debug(
                f"has source_image: {job_info.source_image is not None}, "
                f"has source_mask: {job_info.source_mask is not None}, "
                f"extra_source_images: {esi_count}",
            )
            logger.debug(f"{job_info.payload.model_dump(exclude={'prompt'})}")

            with logger.catch(reraise=True):
                self._start_inference_time = time.time()
                if self._dry_run_skip_inference:
                    time.sleep(self._dry_run_inference_delay)
                    results = self._make_dummy_inference_result(job_info)
                else:
                    from horde_worker_regen.telemetry_spans import span_inference

                    with span_inference(
                        model=job_info.model or "unknown",
                        steps=job_info.payload.ddim_steps,
                        width=job_info.payload.width,
                        height=job_info.payload.height,
                    ):
                        results = self._horde.basic_inference(job_info, progress_callback=self.progress_callback)
        except Exception as e:
            # Keep a reason for the faulted result: the main process logs it and classifies a
            # resource/OOM failure (which earns a degraded retry) from this text. The full message is
            # preserved so torch's "CUDA out of memory" wording reaches the failure classifier intact.
            self._last_inference_error = f"{type(e).__name__}: {e}"
            logger.critical(f"Inference failed: {self._last_inference_error}")
            return None
        finally:
            self._is_busy = False
            self._in_post_processing = False
            self._current_job_inference_steps_complete = False

            self._send_job_metrics_message(str(job_info.id_))
            self._send_download_metrics_if_any()

            # Idempotent: a no-op if sampling completed and the slot was already freed early;
            # the real release for jobs that errored before reaching the VAE boundary.
            self._release_inference_slot()
            # Only release the VAE lock if this job actually acquired it (a job that faulted
            # mid-sampling never did), so the semaphore is never over-released.
            if self._vae_lock_was_acquired:
                with contextlib.suppress(Exception):
                    self._vae_decode_semaphore.release()
            self._vae_lock_was_acquired = False
        return results

    @staticmethod
    def _make_dummy_inference_result(
        job_info: ImageGenerateJobPopResponse,
    ) -> list[ResultingImageReturn]:
        """Create minimal 1x1 PNG results for dry-run mode.

        The returned objects duck-type the parts of hordelib's ``ResultingImageReturn``
        that ``send_inference_result_message`` consumes (``rawpng`` and ``faults``).
        """
        import io

        from horde_worker_regen.process_management._dummy_images import make_dummy_png_bytes

        png_bytes = make_dummy_png_bytes()

        n_iter = job_info.payload.n_iter if job_info.payload.n_iter else 1
        results = []
        for _ in range(n_iter):
            results.append(_DryRunResultingImage(rawpng=io.BytesIO(png_bytes)))
        return results  # type: ignore[return-value]

    def start_alchemy(self, form: AlchemyFormSpec) -> None:
        """Run a graph-backed alchemy form (upscale/facefix/strip_background) and report the result.

        The result image is WebP-encoded here (quality 95, matching the legacy alchemist)
        so the main process can upload it to R2 without re-encoding.
        """
        import PIL.Image
        from hordelib.api import classify_post_processor

        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_STARTING,
            info=f"Starting alchemy form {form.form} ({form.form_id})",
        )

        time_start = time.time()
        state = GENERATION_STATE.faulted
        image_base64: str | None = None

        try:
            kind = classify_post_processor(form.form)
            if kind is None:
                raise ValueError(f"Unknown alchemy form for inference process: {form.form}")

            source_image = PIL.Image.open(io.BytesIO(base64.b64decode(form.source_image_base64)))

            result = self._horde.post_process(
                {
                    "model": form.form,
                    "source_image": source_image,
                },
            )
            if result.image is None:
                raise RuntimeError("Alchemy form produced no image")

            buffer = io.BytesIO()
            # WebP keeps submit bandwidth low; quality/method match the legacy alchemist.
            result.image.save(buffer, format="WebP", quality=95, method=6)
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
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
                image_base64=image_base64,
            ),
        )

        self._send_job_metrics_message(form.form_id, is_alchemy=True)

        process_state = (
            HordeProcessState.ALCHEMY_COMPLETE if state == GENERATION_STATE.ok else HordeProcessState.ALCHEMY_FAILED
        )
        self.send_process_state_change_message(
            process_state=process_state,
            info=f"Finished alchemy form {form.form} ({form.form_id})",
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @staticmethod
    def clear_gc_and_torch_cache() -> None:
        """Clear the garbage collector and the active backend's device cache."""
        gc.collect()
        from hordelib.api import clear_accelerator_cache

        clear_accelerator_cache()

    @logger.catch(reraise=True)
    def unload_models_from_vram(self) -> None:
        """Unload all models from VRAM."""
        if not self._dry_run_skip_inference:
            self._horde.backend.free_vram()

            self.clear_gc_and_torch_cache()

        if self._active_model_name is not None:
            self.on_horde_model_state_change(
                process_state=HordeProcessState.UNLOADED_MODEL_FROM_VRAM,
                horde_model_name=self._active_model_name,
                horde_model_state=ModelLoadState.LOADED_IN_RAM,
            )

            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="Unloaded models from VRAM",
            )
        else:
            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="No models to unload from VRAM",
            )

    @logger.catch(reraise=True)
    def unload_models_from_ram(self) -> None:
        """Unload all models from RAM."""
        if not self._dry_run_skip_inference:
            self._horde.backend.free_ram()

            self.clear_gc_and_torch_cache()

        self.send_memory_report_message(include_vram=True)
        if self._active_model_name is not None:
            self.on_horde_model_state_change(
                process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
                horde_model_name=self._active_model_name,
                horde_model_state=ModelLoadState.ON_DISK,
            )

            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="Unloaded models from RAM",
            )
        else:
            self.send_process_state_change_message(
                process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
                info="No models to unload from RAM",
            )

            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="Waiting for job",
            )
        logger.info("Unloaded all models from RAM")
        self._active_model_name = None

    def reload_model_database(self) -> None:
        """Reload the model managers' references from disk (no download).

        Triggered by the parent after it refreshes the on-disk reference, or after the download
        process reports new LoRa/TI availability. Reloading the adhoc (LoRa/TI) managers picks up
        records other processes wrote, which is how newly downloaded auxiliary models become visible
        here without a restart.
        """
        if self._dry_run_skip_inference or self._shared_model_manager is None:
            return
        try:
            self._shared_model_manager.manager.reload_database()
            logger.info("Reloaded model database from disk")
        except Exception as e:  # noqa: BLE001 - a reload failure must not crash the inference process
            logger.error(f"Failed to reload model database: {type(e).__name__}: {e}")

    @logger.catch(reraise=True)
    @override
    def cleanup_for_exit(self) -> None:
        """Cleanup the process pending a shutdown."""
        self.unload_models_from_ram()
        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDED,
            info="Process ended",
        )

    def send_aux_model_message(
        self,
        job_info: ImageGenerateJobPopResponse,
        time_elapsed: float,
        process_state: HordeProcessState,
        info: str,
    ) -> None:
        """Send an auxiliary model download complete message to the main process.

        Args:
            job_info (ImageGenerateJobPopResponse): The job that was inferred.
            time_elapsed (float): The time elapsed during the last operation.
            process_state (HordeProcessState): The state of the process.
            info (str): Additional information about the message.
        """
        message = HordeAuxModelStateChangeMessage(
            process_state=process_state,
            process_id=self.process_id,
            process_launch_identifier=self.process_launch_identifier,
            info=info,
            time_elapsed=time_elapsed,
            sdk_api_job_info=job_info,
        )
        self.process_message_queue.put(message)

    def send_inference_result_message(
        self,
        process_state: HordeProcessState,
        job_info: ImageGenerateJobPopResponse,
        results: list[ResultingImageReturn] | None,
        time_elapsed: float,
    ) -> None:
        """Send an inference result message to the main process.

        Args:
            process_state (HordeProcessState): The state of the process.
            job_info (ImageGenerateJobPopResponse): The job that was inferred.
            results (list[ResultingImageReturn] | None): The generated images, or None if inference failed.
            time_elapsed (float): The time elapsed during the last operation.
        """
        all_image_results = []

        if results is not None:
            for result in results:
                if result.rawpng is None:
                    logger.critical("Result or result image is None")
                    continue

                image_base64 = base64.b64encode(result.rawpng.getvalue()).decode("utf-8")
                all_image_results.append(
                    HordeImageResult(
                        image_base64=image_base64,
                        generation_faults=result.faults,
                    ),
                )

        is_faulted = results is None or len(results) == 0
        # A faulted result's info doubles as the diagnostic + the resource-failure classification signal,
        # so prefer the captured exception summary over the (empty, for a fault) inference-rate string.
        if is_faulted and self._last_inference_error is not None:
            info = self._last_inference_error
        else:
            info = self._last_job_inference_rate if self._last_job_inference_rate is not None else ""

        message = HordeInferenceResultMessage(
            process_id=self.process_id,
            process_launch_identifier=self.process_launch_identifier,
            info=info,
            state=GENERATION_STATE.faulted if is_faulted else GENERATION_STATE.ok,
            time_elapsed=time_elapsed,
            job_image_results=all_image_results,
            sdk_api_job_info=job_info,
        )
        self.process_message_queue.put(message)

        if self._active_model_name is None:
            logger.critical("No active model name, cannot update model state")
            return

        self.on_horde_model_state_change(
            process_state=process_state,
            horde_model_name=self._active_model_name,
            horde_model_state=ModelLoadState.LOADED_IN_VRAM,
        )

        self.send_process_state_change_message(
            HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    @logger.catch(reraise=True)
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Receive and handle a control message from the main process.

        Args:
            message (HordeControlMessage): The message to handle.
        """
        logger.debug(f"Received ({type(message)}): {message.control_flag}")

        if isinstance(message, HordePreloadInferenceModelMessage):
            self.preload_model(
                horde_model_name=message.horde_model_name,
                will_load_loras=message.will_load_loras,
                seamless_tiling_enabled=message.seamless_tiling_enabled,
                job_info=message.sdk_api_job_info,
            )
        elif isinstance(message, HordeInferenceControlMessage):
            if message.control_flag == HordeControlFlag.START_INFERENCE:
                if self._active_model_name is None or message.horde_model_name != self._active_model_name:
                    if message.horde_model_name != self._active_model_name:
                        logger.warning(
                            f"Received START_INFERENCE control message for model {message.horde_model_name} "
                            f"but currently active model is {self._active_model_name}",
                        )

                    self.preload_model(
                        horde_model_name=message.horde_model_name,
                        will_load_loras=message.sdk_api_job_info.payload.loras is not None
                        and len(
                            message.sdk_api_job_info.payload.loras,
                        )
                        > 0,
                        seamless_tiling_enabled=message.sdk_api_job_info.payload.tiling,
                        job_info=message.sdk_api_job_info,
                    )
                else:
                    # The model is already resident, so the scheduler dispatched inference without a
                    # fresh preload. The aux-model (LoRA/TI) download lives inside preload_model, so
                    # without this call those per-job downloads fall through to a lazy fetch inside
                    # basic_inference while the slot reads INFERENCE_STARTING. A slow CivitAI download
                    # there emits no step heartbeat, so the parent's inference_step_timeout watchdog
                    # mistakes it for a hang and kills the process. download_aux_models runs under the
                    # heartbeat-protected DOWNLOADING_AUX_MODEL path and is idempotent (a no-op when
                    # the loras are already on disk or the job has none).
                    self.download_aux_models(message.sdk_api_job_info)

                if message.horde_model_name != self._active_model_name:
                    error_message = f"Received START_INFERENCE control message for model {message.horde_model_name} "
                    error_message += f"but currently active model is {self._active_model_name}"
                    logger.error(error_message)

                    self.send_process_state_change_message(
                        process_state=HordeProcessState.INFERENCE_FAILED,
                        info=error_message,
                    )

                self.on_horde_model_state_change(
                    horde_model_name=message.horde_model_name,
                    process_state=HordeProcessState.INFERENCE_STARTING,
                    horde_model_state=ModelLoadState.IN_USE,
                )

                time_start = time.time()

                results = self.start_inference(message.sdk_api_job_info)

                if results is None or len(results) == 0:
                    self.send_memory_report_message(include_vram=True)
                    self.send_inference_result_message(
                        process_state=HordeProcessState.INFERENCE_FAILED,
                        job_info=message.sdk_api_job_info,
                        results=None,
                        time_elapsed=time.time() - time_start,
                    )

                    active_model_name = self._active_model_name
                    logger.debug("Unloading models from RAM")
                    self.unload_models_from_ram()
                    logger.debug("Unloaded models from RAM")
                    self.send_memory_report_message(include_vram=True)

                    if active_model_name is None:
                        logger.critical("No active model name, cannot update model state")

                    else:
                        self.preload_model(
                            active_model_name,
                            will_load_loras=True,
                            seamless_tiling_enabled=False,
                            job_info=message.sdk_api_job_info,
                        )
                        logger.warning("A non-blocking LoRas/TIs preload didn't occur!. This is a bug.")

                    self.send_process_state_change_message(
                        process_state=HordeProcessState.WAITING_FOR_JOB,
                        info="Waiting for job",
                    )
                    return

                process_state = HordeProcessState.INFERENCE_COMPLETE if results else HordeProcessState.INFERENCE_FAILED
                logger.debug(f"Finished inference with process state {process_state}")
                self.send_inference_result_message(
                    process_state=process_state,
                    job_info=message.sdk_api_job_info,
                    results=results,
                    time_elapsed=time.time() - time_start,
                )
            else:
                logger.critical(f"Received unexpected message: {message}")
                return
        elif isinstance(message, HordeAlchemyControlMessage):
            if message.control_flag == HordeControlFlag.START_ALCHEMY:
                self.start_alchemy(message.form)
            else:
                logger.critical(f"Received unexpected message: {message}")
            return
        elif message.control_flag == HordeControlFlag.END_PROCESS:
            self.send_process_state_change_message(
                process_state=HordeProcessState.PROCESS_ENDING,
                info="Process stopping",
            )

            self._end_process = True
            return

        if isinstance(message, HordeControlModelMessage) and message.control_flag == HordeControlFlag.DOWNLOAD_MODEL:
            self.download_model(horde_model_name=message.horde_model_name)

        if isinstance(message, HordeControlMessage):
            if message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                self.unload_models_from_vram()
            elif message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
                self.unload_models_from_ram()
            elif message.control_flag == HordeControlFlag.RELOAD_MODEL_DATABASE:
                self.reload_model_database()
