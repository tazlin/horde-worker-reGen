"""Contains the classes to form an inference process, which generate images."""

from __future__ import annotations

import contextlib
import gc
import io
import sys
import time
from dataclasses import dataclass, field

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

from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.ipc.messages import (
    AUX_RESOLVE_FAILED_INFO,
    AuxModelKind,
    AuxModelRef,
    HordeControlFlag,
    HordeControlMessage,
    HordeControlModelMessage,
    HordeDownloadMetricsMessage,
    HordeEvictComponentsControlMessage,
    HordeHeartbeatType,
    HordeImageResult,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeJobMetricsMessage,
    HordeModelStateChangeMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    HordeSampleControlMessage,
    HordeSampleResultMessage,
    ModelLoadState,
    PipelineStageTag,
    SampleSliceResult,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcess
from horde_worker_regen.process_management.scheduling.clearance_lease import ClearanceLeaseProxy


def inject_premade_control_map(
    generation_parameters: ImageGenerationParameters,
    control_map_bytes: bytes,
) -> bool:
    """Assign a pre-computed control map onto a job's controlnet parameters, returning whether it applied.

    The image-utilities lane annotates a ControlNet job's control map ahead of dispatch; setting it here
    makes hordelib run the ``none`` preprocessor over the supplied map (``control_uses_premade_map``)
    instead of re-deriving the annotation in the main venv. A no-op (returns False) when the job carries
    no controlnet component, which should not happen for an annotated job but is guarded so a mismatched
    dispatch degrades to normal in-graph preprocessing rather than raising.
    """
    controlnet_params = generation_parameters.additional_params.controlnet_params
    if controlnet_params is None:
        return False
    controlnet_params.control_map = control_map_bytes
    return True


if TYPE_CHECKING:
    from horde_sdk.generation_parameters.image.object_models import ImageGenerationParameters
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


def _gpu_arch_supported(arch_list: list[str], capability: tuple[int, int]) -> bool:
    """Whether the installed CUDA torch build has a usable kernel/PTX for a device of ``capability``.

    ``torch.cuda.get_arch_list()`` reports the architectures the wheel was compiled for, as ``sm_<n>``
    binary cubins and/or ``compute_<n>`` PTX. A device of compute capability (major, minor) can run:

    * a binary ``sm_<n>`` kernel of the *same major* whose minor is <= the device minor (cubins are
      forward-compatible only within a major), or
    * any ``compute_<n>`` PTX whose (major, minor) <= the device, JIT-compiled at load time.

    If neither exists, every kernel launch raises ``cudaErrorNoKernelImageForDevice``, which ComfyUI
    swallows into a generic "no images produced" fault. This predicts that before the first launch.
    """
    dev_major, dev_minor = capability
    for entry in arch_list:
        kind, _, ver = entry.partition("_")
        if not ver.isdigit() or len(ver) < 2:
            continue
        major, minor = int(ver[:-1]), int(ver[-1])
        if kind == "sm" and major == dev_major and minor <= dev_minor:
            return True
        if kind == "compute" and (major, minor) <= (dev_major, dev_minor):
            return True
    return False


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

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        inference_semaphore: Semaphore,
        vae_decode_semaphore: Semaphore,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        device_index: int = 0,
        dry_run_skip_inference: bool = False,
        dry_run_inference_delay: float = 1.0,
        gpu_sampling_lease: ClearanceLeaseProxy | None = None,
        expect_image_models: bool = True,
    ) -> None:
        """Initialise the HordeInferenceProcess.

        Args:
            process_id (int): The ID of the process. This is not the same as the process PID.
            process_message_queue (ProcessQueue): The queue the main process uses to receive messages from all worker \
                processes.
            pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
            inference_semaphore (Semaphore): A semaphore used to limit the number of concurrent inference jobs.
            vae_decode_semaphore (Semaphore): A semaphore used to limit the number of concurrent VAE decode jobs.
            disk_lock (Lock): A lock used to prevent multiple processes from accessing disk at the same time.
            process_launch_identifier (int): The identifier for the process launch.
            dry_run_skip_inference (bool, optional): Skip real inference and return a dummy image. Defaults to False.
            dry_run_inference_delay (float, optional): Seconds to sleep when dry-run inference is active. \
                Defaults to 1.0.
            gpu_sampling_lease (ClearanceLeaseProxy | None, optional): This process's own GPU denoise \
                clearance proxy, registered with hordelib so the process blocks at the sample call until the \
                parent clears it into its weight-load-plus-denoise window. None disables coordination. \
                Defaults to None.
            device_index (int, optional): The stable index of the GPU this process is pinned to. Defaults to 0.
            expect_image_models (bool, optional): Whether this worker is configured to serve image \
                generation. False for an alchemist-only (e.g. CPU) worker that loads no image models, where \
                an empty image-model database is expected rather than a fatal misconfiguration. Defaults to True.
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
        )

        self._dry_run_skip_inference = dry_run_skip_inference
        self._dry_run_inference_delay = dry_run_inference_delay
        self._expect_image_models = expect_image_models
        # Only a loaded backend has a component cache to report residency from (and to evict from under RAM
        # pressure); a dry-run process has none and must not import hordelib on the report/evict paths.
        self._reports_held_components = not dry_run_skip_inference

        self._inference_semaphore = inference_semaphore
        self._vae_decode_semaphore = vae_decode_semaphore
        # The per-process GPU denoise clearance proxy (hordelib's sampling lease), or None when the lease is
        # disabled. Retained so each job can reset its per-job grant at start (see :meth:`start_inference`).
        self._gpu_sampling_lease = gpu_sampling_lease

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

            self._verify_torch_supports_gpu()
            self._report_if_cpu_only_build()

            if gpu_sampling_lease is not None:
                # Coordinate the GPU denoise loop across inference processes: this process stages its
                # pipeline (checkpoint load, prompt encode) freely, then blocks at the sample call until the
                # parent clears it into its weight-load-plus-denoise window, so the GPU stays busy
                # back-to-back across jobs without heavy pairs co-loading weights.
                from hordelib.api import set_gpu_sampling_lease

                from horde_worker_regen.process_management.scheduling.clearance_lease import (
                    CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS,
                )

                # A shorter acquire timeout than the inference step-timeout kill deadline: a clearance-starved
                # child degrades into unpriced sampling (resuming step heartbeats) well before the hung-process
                # watchdog could mistake its clearance wait for a hang.
                set_gpu_sampling_lease(
                    gpu_sampling_lease,
                    acquire_timeout_seconds=CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS,
                )
                logger.info("Registered GPU denoise clearance proxy for cross-process pipelining")

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
                # The dedicated download process is the sole writer of the ad-hoc LoRA/TI caches; an
                # inference child only resolves already-present files, so it opens those managers read
                # only. Reads still coordinate with the writer through ``multiprocessing_lock``.
                adhoc_read_only=True,
            )

            if SharedModelManager.manager.compvis is None:
                logger.critical("Failed to initialise SharedModelManager")
                self.send_process_state_change_message(
                    process_state=HordeProcessState.PROCESS_ENDED,
                    info="Failed to initialise compvis in SharedModelManager",
                )
                sys.exit(1)

            # An alchemist-only worker (e.g. a CPU install) legitimately loads no image models; alchemy
            # graph and CLIP forms run through their own managers, so an empty image-model database is
            # expected there rather than a fatal misconfiguration. Only image-serving workers require a
            # model to be present.
            if self._expect_image_models and len(SharedModelManager.manager.compvis.available_models) == 0:
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

    def _report_if_cpu_only_build(self) -> None:
        """Tell the parent when the installed torch is a CPU-only build, so it stops popping image jobs.

        Image generation on CPU is impractically slow, so a CPU-only torch build must not serve dreamer
        work even when the torch-free orchestrator's install sentinel (``bin/backend``) was never set to
        ``cpu`` (a manually installed CPU torch is the motivating case). The torch-bearing child is
        authoritative about the build, so it reports the fact; the parent latches it and the image popper
        stops, while alchemy (which runs acceptably on CPU) keeps serving on this same process. Unlike
        :meth:`_verify_torch_supports_gpu` this never exits: the process is still useful for alchemy.

        Build-based on purpose: it keys on the torch *build* having no GPU backend, so a merely masked or
        broken GPU on a real GPU build is not mistaken for a CPU install (which would wrongly disable image
        generation). Advisory and self-contained: any error determining the build type is logged and
        ignored (the check must never crash the process).
        """
        try:
            from hordelib.utils.torch_memory import torch_build_is_cpu_only

            is_cpu_build = torch_build_is_cpu_only()
        except Exception as e:
            logger.debug(f"Could not determine whether torch is a CPU-only build: {type(e).__name__} {e}")
            return

        if not is_cpu_build:
            return

        logger.warning(
            "The installed PyTorch is a CPU-only build, so image generation is disabled (it would run "
            "~100x slower); this worker will serve alchemy only. Reinstall a GPU build to enable image "
            "generation.",
        )
        # The info string is the operator-facing reason the parent relays verbatim to the TUI; keep it
        # self-contained (no torch references the parent would have to interpret).
        self.send_process_state_change_message(
            process_state=HordeProcessState.TORCH_BUILD_CPU_ONLY,
            info=(
                "Installed PyTorch is a CPU-only build; image generation is disabled (CPU inference is "
                "impractically slow). Alchemy forms continue to run. Reinstall a GPU build to enable image "
                "generation."
            ),
        )

    def _verify_torch_supports_gpu(self) -> None:
        """Fail fast (with an actionable message) when the installed torch has no kernels for this GPU.

        A torch wheel built for the wrong CUDA architecture set (e.g. a cu126 build on a Blackwell card,
        or a CUDA 13 build on a pre-Turing card) raises ``cudaErrorNoKernelImageForDevice`` at the first
        kernel launch. ComfyUI swallows that into a generic "no images produced" RuntimeError, so without
        this check the cause surfaces only as a stream of faults, the resource breaker never fires, and
        the worker self-pauses without ever naming the real problem. Detecting it here turns a silent,
        every-job failure into one clear instruction to reinstall the matching backend.

        On a mismatch the child reports ``TORCH_GPU_INCOMPATIBLE`` (so the torch-free orchestrator latches
        a stop-popping flag and the TUI surfaces it) and exits; the crash-loop breaker then quarantines the
        slot rather than respawning into the same failure. Reporting *before* the slot ever advertises
        readiness guarantees the parent latches the flag before it could dispatch a job here.

        Advisory and self-contained: any error querying torch is logged and ignored (the check must never
        be what crashes the process), and non-CUDA builds (ROCm/DirectML/CPU) are skipped since their
        arch tags do not apply.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return
            arch_list = torch.cuda.get_arch_list()
            # Only CUDA builds tag architectures as sm_/compute_; ROCm reports gfx*, so skip those.
            if not any(arch.startswith("sm_") for arch in arch_list):
                return
            capability = torch.cuda.get_device_capability(self.device_index)
            if _gpu_arch_supported(arch_list, capability):
                return
            device_name = torch.cuda.get_device_name(self.device_index)
            # The wheel's build tag (e.g. "12.6" -> a cu126 build) names *which* build was installed, so a
            # support log ties the mismatch straight back to the installer's choice without guesswork.
            torch_build = getattr(torch.version, "cuda", None)
            torch_version = getattr(torch, "__version__", "?")
        except Exception as e:
            logger.debug(f"Could not verify torch GPU architecture support: {type(e).__name__} {e}")
            return

        cap_tag = f"sm_{capability[0]}{capability[1]}"
        build_tag = f"CUDA {torch_build}" if torch_build else "an unknown CUDA"
        logger.critical(
            f"The installed PyTorch ({torch_version}, {build_tag} build) has no CUDA kernels for this GPU. "
            f"{device_name} is compute capability {capability[0]}.{capability[1]} ({cap_tag}), but this "
            f"torch build only supports {' '.join(arch_list)}. Every job would fail at the first kernel "
            f"launch with 'no kernel image is available for execution on the device'. Reinstall the worker "
            f"so the installer picks the matching CUDA backend for your GPU and driver (update.cmd, or "
            f"re-run the installer), and update your NVIDIA driver if the installer asks for a newer one.",
        )
        # The info string is the operator-facing reason the parent relays verbatim to the TUI; keep it
        # self-contained (no torch references the parent would have to interpret).
        self.send_process_state_change_message(
            process_state=HordeProcessState.TORCH_GPU_INCOMPATIBLE,
            info=(
                f"The installed PyTorch ({build_tag} build) has no CUDA kernels for {device_name} "
                f"(compute capability {cap_tag}). Reinstall the worker to get the matching CUDA backend, "
                f"and update your NVIDIA driver if asked. The worker will not pop jobs until this is fixed."
            ),
        )
        sys.exit(1)

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

    def _resolve_aux_models(
        self,
        job_info: ImageGenerateJobPopResponse,
        skipped_aux: list[AuxModelRef] | None = None,
    ) -> bool:
        """Confirm every LoRA/TI a job references is already on disk, using no network.

        The dedicated download process places a job's auxiliary files on disk before the job becomes
        dispatchable, so by inference time they are expected to be present. This performs a disk-and-memory
        only recheck: a stale-reference reload (to pick up files another process wrote) followed by a
        per-entry presence probe. It never fetches, so the ``inference_step_timeout`` watchdog cannot
        mistake a download for a hung sampler. A file listed in ``skipped_aux`` was terminally rejected from
        ad-hoc download (invalid, too large, or NSFW on an SFW-only worker) and is tolerated as intentionally
        absent: the generator skips it and records a fault-metadata entry. Any other missing file returns
        ``False`` and the caller faults the job retryably, which the parent's prefetch reconcile sweep
        re-requests, so a raced eviction self-heals.

        Args:
            job_info (ImageGenerateJobPopResponse): The job whose auxiliary references are checked.
            skipped_aux (list[AuxModelRef] | None): Auxiliary files terminally rejected from ad-hoc download,
                tolerated as intentionally absent instead of faulting the job.

        Returns:
            bool: ``True`` when every referenced LoRA/TI resolves on disk or is tolerated as skipped (or the
                job references none), ``False`` on the first unresolved reference not in ``skipped_aux``.
        """
        if self._dry_run_skip_inference or self._shared_model_manager is None:
            return True

        loras = job_info.payload.loras or []
        tis = job_info.payload.tis or []
        if not loras and not tis:
            return True

        skipped = skipped_aux or []
        skipped_loras = {(ref.name, bool(ref.is_version)) for ref in skipped if ref.kind is AuxModelKind.LORA}
        skipped_tis = {ref.name for ref in skipped if ref.kind is AuxModelKind.TI}

        manager = self._shared_model_manager.manager
        lora_manager = manager.lora
        ti_manager = manager.ti

        if loras and lora_manager is not None:
            lora_manager.refresh_reference_if_stale()
        if tis and ti_manager is not None:
            ti_manager.refresh_reference_if_stale()

        for lora_entry in loras:
            if lora_manager is None or not lora_manager.is_lora_available(
                lora_entry.name,
                is_version=bool(lora_entry.is_version),
            ):
                if (lora_entry.name, bool(lora_entry.is_version)) in skipped_loras:
                    logger.info(
                        f"LoRA {lora_entry.name!r} for job {job_info.id_} was rejected from ad-hoc download; "
                        "generating without it.",
                    )
                    continue
                logger.warning(
                    f"LoRA {lora_entry.name!r} for job {job_info.id_} is not present on disk at dispatch; "
                    "faulting the job so the parent re-requests its prefetch.",
                )
                return False

        for ti_entry in tis:
            if ti_manager is None or ti_manager.fuzzy_find_ti_key(ti_entry.name) is None:
                if ti_entry.name in skipped_tis:
                    logger.info(
                        f"Textual inversion {ti_entry.name!r} for job {job_info.id_} was rejected from ad-hoc "
                        "download; generating without it.",
                    )
                    continue
                logger.warning(
                    f"Textual inversion {ti_entry.name!r} for job {job_info.id_} is not present on disk at "
                    "dispatch; faulting the job so the parent re-requests its prefetch.",
                )
                return False

        return True

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
            job_info (ImageGenerateJobPopResponse): The job the model is being preloaded for.
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

    _current_job_inference_steps_complete: bool = False
    _vae_lock_was_acquired: bool = False
    _inference_slot_released: bool = False

    _last_progress_step_seen: int | None = None
    """The sampling step reported by the previous progress callback for the current job, or None."""
    _nonadvancing_progress_repeats: int = 0
    """Consecutive progress callbacks at the same step without advancing (0 while sampling advances).

    A healthy generation reports each step (including the last) once, so this stays 0. It climbs only
    when ComfyUI loops on a single step and never returns; it is forwarded on every heartbeat so the
    parent's stuck-step watchdog can reap this slot, since the child cannot abort the wedged call itself
    (hordelib swallows exceptions raised inside the progress callback)."""

    _last_job_inference_rate: str | None = None
    _current_job_kept_model_resident: bool = False
    """Whether the job now finishing was dispatched with the keep-model-resident hint.

    Decides the post-job model-state report: LOADED_IN_VRAM only when the eviction was actually
    deferred, LOADED_IN_RAM otherwise. The parent's retention gate reads that map to enforce a single
    VRAM-resident model per card, so reporting VRAM residency for weights hordelib just evicted would
    make every recently-active sibling look like a resident and starve retention."""
    _last_inference_error: str | None = None
    """Summary of the exception that failed the current job's inference, or None if it succeeded.

    Surfaced as the faulted result's ``info`` so the main process can both log a real reason (previously
    a faulted result carried only the empty rate string) and classify a resource/OOM failure for retry."""

    def _send_inference_memory_report(self) -> None:
        """Send a precise boundary VRAM report at an inference stage transition (sampling done / VAE decode).

        These stage-boundary reports carry the working set at a known moment and complement the reporter
        thread's interval sampling, which runs independently while the main loop is blocked in the GPU op.
        """
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
        from hordelib.api import ComfyUIProgressUnit, log_free_ram

        # Track non-advancing sampling progress before any early return, so the post-completion repeats
        # (the wedge signature: ComfyUI re-reporting the final step forever) are counted too. A healthy
        # job reports each step once, so an advancing step resets the counter to 0.
        reported_step = (
            progress_report.comfyui_progress.current_step if progress_report.comfyui_progress is not None else None
        )
        if reported_step is not None:
            if reported_step == self._last_progress_step_seen:
                self._nonadvancing_progress_repeats += 1
            else:
                self._nonadvancing_progress_repeats = 0
                self._last_progress_step_seen = reported_step

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

            self.send_heartbeat_message(
                heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE,
                nonadvancing_step_repeats=self._nonadvancing_progress_repeats,
            )
            return

        if progress_report.comfyui_progress is not None and progress_report.comfyui_progress.current_step == (
            progress_report.comfyui_progress.total_steps
        ):
            self.send_heartbeat_message(
                heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE,
                nonadvancing_step_repeats=self._nonadvancing_progress_repeats,
            )
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
                nonadvancing_step_repeats=self._nonadvancing_progress_repeats,
            )
        else:
            self.send_heartbeat_message(
                heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE,
                nonadvancing_step_repeats=self._nonadvancing_progress_repeats,
            )

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

    def start_inference(
        self,
        job_info: ImageGenerateJobPopResponse,
        *,
        keep_model_resident: bool = False,
        premade_control_map_bytes: bytes | None = None,
    ) -> list[ResultingImageReturn] | None:
        """Start an inference job in the HordeLib instance.

        Args:
            job_info (ImageGenerateJobPopResponse): The job to start inference on.
            keep_model_resident (bool, optional): Keep the model resident in VRAM after this job
                instead of evicting it, so a following same-model job skips the RAM->VRAM reload. The
                scheduler sets this only when it has confirmed the next job reuses the model and the
                VRAM budget allows it. Defaults to False.
            premade_control_map_bytes (bytes | None, optional): A ControlNet control map the
                image-utilities lane annotated ahead of dispatch, injected into the job's controlnet
                parameters so hordelib runs the ``none`` preprocessor over it. None for a job with no
                pre-annotated control map. Defaults to None.

        Returns:
            list[Image] | None: The generated images, or None if inference failed.
        """
        # Reset the clearance grant for this job before the pipeline runs: one grant covers a whole job (a
        # multi-sample workflow passes through after the first), and the next job must wait for its own grant.
        if self._gpu_sampling_lease is not None:
            self._gpu_sampling_lease.begin_job()

        logger.info("Checking if too many inference jobs are already running...")
        self._inference_semaphore.acquire()
        logger.info("Acquired inference semaphore.")
        self._is_busy = True
        self._current_job_inference_steps_complete = False
        self._inference_slot_released = False
        self._vae_lock_was_acquired = False
        self._current_job_kept_model_resident = keep_model_resident
        self._last_job_inference_rate = None
        self._last_inference_error = None
        self._last_progress_step_seen = None
        self._nonadvancing_progress_repeats = 0

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
                        results = self._run_typed_inference(
                            job_info,
                            keep_model_resident=keep_model_resident,
                            premade_control_map_bytes=premade_control_map_bytes,
                        )
        except Exception as e:
            # Keep a reason for the faulted result: the main process logs it and classifies a
            # resource/OOM failure (which earns a degraded retry) from this text. The full message is
            # preserved so torch's "CUDA out of memory" wording reaches the failure classifier intact.
            self._last_inference_error = f"{type(e).__name__}: {e}"
            logger.critical(f"Inference failed: {self._last_inference_error}")
            return None
        finally:
            self._is_busy = False
            self._current_job_inference_steps_complete = False
            self._last_progress_step_seen = None
            self._nonadvancing_progress_repeats = 0

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

    def _run_typed_inference(
        self,
        job_info: ImageGenerateJobPopResponse,
        *,
        keep_model_resident: bool,
        premade_control_map_bytes: bytes | None = None,
    ) -> list[ResultingImageReturn]:
        """Run a job through hordelib's typed generation path.

        The AI-Horde job is translated to backend-agnostic generation parameters by the SDK
        (hordelib itself has no knowledge of the AI-Horde API models) and inference runs via
        `HordeLib.generate` with automatic pipeline selection. Any post-processing the job
        requested is not run here: the raw images are returned and the main process routes them
        to the dedicated post-processing lane.

        When ``premade_control_map_bytes`` is set, the image-utilities lane already annotated this job's
        ControlNet control map, so it is injected into the converted parameters before generation and the
        in-graph preprocessor runs ``none`` over it rather than re-deriving the annotation here.
        """
        from horde_sdk.worker.dispatch.ai_horde.image.convert import (
            convert_image_job_pop_response_to_parameters,
        )

        from horde_worker_regen.reference_helper import ensure_offline_reference_manager

        conversion_result = convert_image_job_pop_response_to_parameters(
            api_response=job_info,
            model_reference_manager=ensure_offline_reference_manager(),
        )

        if premade_control_map_bytes is not None and not inject_premade_control_map(
            conversion_result.generation_parameters,
            premade_control_map_bytes,
        ):
            logger.warning(
                f"Job {job_info.id_} carried a pre-annotated control map but has no controlnet parameters; "
                "falling back to in-graph preprocessing.",
            )

        from hordelib.api import AUTO_PIPELINE

        results = self._horde.generate(
            conversion_result.generation_parameters,
            pipeline=AUTO_PIPELINE,
            progress_callback=self.progress_callback,
            defer_vram_unload=keep_model_resident,
        )

        for single_result in results:
            single_result.faults = conversion_result.faults + single_result.faults
        return results

    def _run_sample_stage(self, message: HordeSampleControlMessage) -> None:
        """Run the disaggregated sample stage for each slice, returning a LATENT per job.

        A sampler process loads only the UNet (via HordeLib.sample_stage's subset-flagged loader) and
        consumes each slice's injected CONDITIONING (and optional source LATENT for img2img), producing
        a LATENT the image lane decodes. Each slice is faulted independently; a slice failure never
        drops its siblings in the batch.
        """
        import time

        from horde_sdk.worker.dispatch.ai_horde.image.convert import (
            convert_image_job_pop_response_to_parameters,
        )

        from horde_worker_regen.reference_helper import ensure_offline_reference_manager

        # Primed for the sample stage: the pinned sampler advances to INFERENCE_STARTING on its first step.
        self.send_process_state_change_message(HordeProcessState.INFERENCE_PRIMED, info="Priming")
        start_time = time.time()
        reference_manager = ensure_offline_reference_manager()
        results: list[SampleSliceResult] = []

        for job_slice in message.slices:
            try:
                if self._dry_run_skip_inference:
                    latent_bytes = self._make_dummy_latent_bytes()
                else:
                    conversion = convert_image_job_pop_response_to_parameters(
                        api_response=job_slice.sdk_api_job_info,
                        model_reference_manager=reference_manager,
                    )
                    latent_bytes = self._horde.sample_stage(
                        conversion.generation_parameters,
                        positive_conditioning_bytes=job_slice.positive_conditioning_bytes,
                        negative_conditioning_bytes=job_slice.negative_conditioning_bytes,
                        source_latent_bytes=job_slice.source_latent_bytes,
                    )
                results.append(
                    SampleSliceResult(job_id=job_slice.job_id, latent_bytes=latent_bytes, state=GENERATION_STATE.ok),
                )
            except Exception as sample_error:  # noqa: BLE001 - one slice's fault must not drop the batch
                logger.error(f"Sample stage faulted for job {job_slice.job_id}: {sample_error}")
                results.append(
                    SampleSliceResult(job_id=job_slice.job_id, latent_bytes=None, state=GENERATION_STATE.faulted),
                )
            # Drained per slice (success or fault) so the slice that triggered a UNet load owns the
            # load events; same-model siblings in the batch then correctly report none.
            self.send_stage_job_metrics_message(str(job_slice.job_id), stage=PipelineStageTag.SAMPLE)

        self.process_message_queue.put(
            HordeSampleResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info="",
                time_elapsed=time.time() - start_time,
                results=results,
            ),
        )
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, info="Waiting for job")

    def _make_dummy_latent_bytes(self) -> bytes:
        """A minimal serialized LATENT for dry-run mode (no backend), shaped like an SDXL 1MP latent."""
        import torch
        from hordelib.execution.stage_payloads import serialize_latent

        return serialize_latent({"samples": torch.zeros([1, 4, 128, 128])})

    @staticmethod
    def _make_dummy_inference_result(
        job_info: ImageGenerateJobPopResponse,
    ) -> list[ResultingImageReturn]:
        """Create minimal 1x1 PNG results for dry-run mode.

        The returned objects duck-type the parts of hordelib's ``ResultingImageReturn``
        that ``send_inference_result_message`` consumes (``rawpng`` and ``faults``).
        """
        import io

        from horde_worker_regen.process_management.simulation._dummy_images import make_dummy_png_bytes

        png_bytes = make_dummy_png_bytes()

        n_iter = job_info.payload.n_iter if job_info.payload.n_iter else 1
        results = []
        for _ in range(n_iter):
            results.append(_DryRunResultingImage(rawpng=io.BytesIO(png_bytes)))
        return results  # type: ignore[return-value]

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

        # Emit a fresh VRAM report now that the cache has been emptied: the allocator's reservation just fell,
        # so the parent's committed-VRAM ledger must see the release promptly rather than at the next periodic
        # report up to a report interval later, keeping committed tracking device truth after an unload.
        self.send_memory_report_message(include_vram=True)

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
    def release_allocator_cache(self) -> None:
        """Release the torch allocator's cached free blocks without unloading models, then report memory.

        Empties the caching allocator's reserved-but-unused device blocks so the reservation returns to
        the card while the resident model stays loaded, then sends a fresh memory report so the parent's
        per-process ``process_reserved_mb`` reflects the release promptly. Deliberately emits no model
        state change: nothing was unloaded.
        """
        logger.debug("Releasing allocator cache (resident models stay loaded)")
        if not self._dry_run_skip_inference:
            self.clear_gc_and_torch_cache()
        self.send_memory_report_message(include_vram=True)

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

                all_image_results.append(
                    HordeImageResult(
                        image_bytes=result.rawpng.getvalue(),
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
            # Report where the weights actually are: hordelib evicts them from VRAM after the job
            # unless the dispatch deferred it, and the parent's retention gate enforces its
            # one-resident-per-card rule from this map entry.
            horde_model_state=(
                ModelLoadState.LOADED_IN_VRAM
                if self._current_job_kept_model_resident
                else ModelLoadState.LOADED_IN_RAM
            ),
        )

        self.send_process_state_change_message(
            HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    def _fault_job_for_unresolved_aux(self, job_info: ImageGenerateJobPopResponse) -> None:
        """Report a job faulted because its auxiliary files could not be resolved on disk, then go idle.

        Routes through the ordinary faulted-result path (so the parent's retry/backoff brain owns the
        outcome), tagging the result's ``info`` with :data:`AUX_RESOLVE_FAILED_INFO` so the parent can
        classify it as a retryable prefetch race rather than a generation failure. The process is kept
        alive with its model resident and returned to ``WAITING_FOR_JOB``.
        """
        self._last_inference_error = AUX_RESOLVE_FAILED_INFO
        self.send_inference_result_message(
            process_state=HordeProcessState.INFERENCE_FAILED,
            job_info=job_info,
            results=None,
            time_elapsed=0.0,
        )
        self._last_inference_error = None
        self.send_process_state_change_message(
            HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    @logger.catch(reraise=True)
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Dispatch one control message to the appropriate handler (preload/inference/alchemy/lifecycle)."""
        logger.debug(f"Received ({type(message)}): {message.control_flag}")

        if isinstance(message, HordeEvictComponentsControlMessage):
            self.evict_held_components(message.identities)
            return

        if isinstance(message, HordePreloadInferenceModelMessage):
            self.preload_model(
                horde_model_name=message.horde_model_name,
                will_load_loras=message.will_load_loras,
                seamless_tiling_enabled=message.seamless_tiling_enabled,
                job_info=message.sdk_api_job_info,
            )
        elif isinstance(message, HordeSampleControlMessage):
            self._run_sample_stage(message)
        elif isinstance(message, HordeInferenceControlMessage):
            if message.control_flag == HordeControlFlag.START_INFERENCE:
                # The parent's download process placed this job's LoRAs/TIs on disk before the job became
                # dispatchable, so here the child only confirms them read-only (no network). An unresolved
                # file means a raced eviction between prefetch and dispatch: fault the job retryably and go
                # idle, keeping the model resident, so the parent's prefetch reconcile sweep re-requests it.
                if not self._resolve_aux_models(message.sdk_api_job_info, message.skipped_aux_models):
                    self._fault_job_for_unresolved_aux(message.sdk_api_job_info)
                    return

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

                if message.horde_model_name != self._active_model_name:
                    error_message = f"Received START_INFERENCE control message for model {message.horde_model_name} "
                    error_message += f"but currently active model is {self._active_model_name}"
                    logger.error(error_message)

                    self.send_process_state_change_message(
                        process_state=HordeProcessState.INFERENCE_FAILED,
                        info=error_message,
                    )

                # Primed, not yet sampling: the model is IN_USE and the slot owns the job, but the denoise
                # loop is entered inside start_inference() below (after acquiring the inference slot and the
                # GPU sampling lease). The parent advances this slot to INFERENCE_STARTING on the first step.
                self.on_horde_model_state_change(
                    horde_model_name=message.horde_model_name,
                    process_state=HordeProcessState.INFERENCE_PRIMED,
                    horde_model_state=ModelLoadState.IN_USE,
                )

                time_start = time.time()

                results = self.start_inference(
                    message.sdk_api_job_info,
                    keep_model_resident=message.keep_model_resident_after,
                    premade_control_map_bytes=message.premade_control_map_bytes,
                )

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
            elif message.control_flag == HordeControlFlag.RELEASE_ALLOCATOR_CACHE:
                self.release_allocator_cache()
            elif message.control_flag == HordeControlFlag.RELOAD_MODEL_DATABASE:
                self.reload_model_database()
