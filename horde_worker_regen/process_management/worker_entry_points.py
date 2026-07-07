import contextlib
import os
import sys
import time
from typing import Protocol

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock, Semaphore

from loguru import logger

from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.lifecycle.child_crash_capture import (
    enable_child_faulthandler,
    neutralize_inherited_argv,
    write_startup_crash,
)
from horde_worker_regen.process_management.lifecycle.debug_attach import maybe_wait_for_process_debugger

# Env var the parent process sets (from its own ``-v`` count) so spawned workers inherit the
# operator's verbosity intent instead of a hardcoded value. Read by ``resolve_worker_log_verbosity``.
WORKER_LOG_VERBOSITY_ENV = "AIWORKER_PROCESS_LOG_VERBOSITY"

# Floor for worker processes. Maps to DEBUG, the level the bridge.log file sink uses; previously
# every worker was hardcoded to 5 (TRACE), which forced trace.log's diagnose output on permanently.
_DEFAULT_WORKER_LOG_VERBOSITY = 4

_SPAWN_TIMING_ENV = "AIWORKER_SPAWN_TIMING"


def _apply_device_pin(
    *,
    process_id: int,
    device_index: int,
    accelerator_kind: str | None,
    role: str = "inference",
) -> None:
    """Mask this process to one GPU before torch loads, when an accelerator kind is provided.

    Applies the env-var portion of hordelib's ``device_pin_env`` (e.g. ``CUDA_VISIBLE_DEVICES``) so the
    child sees only its assigned card as ``cuda:0`` (or the backend equivalent), keeping every
    single-device assumption in ComfyUI/hordelib correct without changes there. Must run before the first
    torch import. A ``None`` kind (the default single-GPU path with no explicit card selection) is a
    no-op, so that case writes no env var and is byte-identical to before. DirectML has no env-var mask (it
    is pinned by the per-card ``--directml`` comfy arg instead) and cpu/mps need no masking, so those are
    skipped here. Best-effort: a failure to pin logs a warning and the process runs unmasked rather than
    crashing at startup.

    Args:
        process_id: The logical slot id, for the log line.
        device_index: The stable index of the card to pin to.
        accelerator_kind: The backend kind (``cuda``/``rocm``/...), or None to skip pinning.
        role: The process role (``inference``/``safety``) for the log line. Defaults to ``inference``.
    """
    if accelerator_kind is None:
        return
    try:
        from hordelib.utils.device_pinning import device_pin_env
        from hordelib.utils.torch_memory import AcceleratorKind

        kind = AcceleratorKind(accelerator_kind)
        if kind in (AcceleratorKind.directml, AcceleratorKind.cpu, AcceleratorKind.mps):
            return
        pin_env, _extra_args = device_pin_env(kind, device_index)
        os.environ.update(pin_env)
        logger.info(f"Pinned {role} process {process_id} to device {device_index} ({kind.value}): {pin_env}")
    except Exception as pin_error:  # noqa: BLE001 - pinning must never crash a starting child
        logger.warning(
            f"Could not pin {role} process {process_id} to device {device_index} "
            f"({accelerator_kind}): {type(pin_error).__name__} {pin_error}. Running unmasked.",
        )


def _spawn_timing_mark(process_id: int, kind: str, label: str) -> None:
    """Diagnostic: write a raw wall-clock marker to fd 2 for spawn/import-phase timing analysis.

    Opt-in via ``AIWORKER_SPAWN_TIMING``; a no-op otherwise. Uses ``os.write(2, ...)`` rather than
    ``logger`` so the marker lands in the process's inherited stderr (the benchmark's per-level
    ``*.subprocess.log``) regardless of how loguru is later reconfigured, and so the very first
    marker can fire *before any import*, making the spawn-bootstrap + arg-unpickle window
    (parent ``process.start()`` to child entry) directly measurable.
    """
    if not os.environ.get(_SPAWN_TIMING_ENV):
        return
    with contextlib.suppress(Exception):
        os.write(2, f"[spawn-timing] {kind} process_id={process_id} {label} t={time.time():.3f}\n".encode())


_CUDA_ALLOC_CONF_ENV = "PYTORCH_CUDA_ALLOC_CONF"
_HIP_ALLOC_CONF_ENV = "PYTORCH_HIP_ALLOC_CONF"
_EXPANDABLE_SEGMENTS_VALUE = "expandable_segments:True"


def _enable_expandable_segments(*, amd_gpu: bool, directml: int | None) -> None:
    """Opt the inference child's CUDA/ROCm caching allocator into expandable segments.

    Fragmentation (large ``reserved but unallocated`` pools) is a frequent cause of the OOM that
    killed live jobs even when the device still reported free memory; torch's own OOM message
    recommends this setting. It must be set *before* torch is imported, so this runs at the very top
    of the spawned child. We only touch the variable when the operator has not set their own value,
    and we skip DirectML (a different allocator entirely). The env name differs by build: CUDA builds
    read ``PYTORCH_CUDA_ALLOC_CONF``; ROCm/HIP builds read ``PYTORCH_HIP_ALLOC_CONF`` (older ROCm
    builds still honor the CUDA name), so for AMD we set both.
    """
    if directml is not None:
        return

    if _CUDA_ALLOC_CONF_ENV not in os.environ:
        os.environ[_CUDA_ALLOC_CONF_ENV] = _EXPANDABLE_SEGMENTS_VALUE
    if amd_gpu and _HIP_ALLOC_CONF_ENV not in os.environ:
        os.environ[_HIP_ALLOC_CONF_ENV] = _EXPANDABLE_SEGMENTS_VALUE


def resolve_worker_log_verbosity() -> int:
    """Resolve the ``verbosity_count`` a spawned worker process should initialise logging with.

    Defaults to DEBUG. The parent may raise it (e.g. to TRACE) via ``WORKER_LOG_VERBOSITY_ENV``;
    worker *file* logs are never dropped below DEBUG, matching the bridge.log sink level.
    """
    raw = os.environ.get(WORKER_LOG_VERBOSITY_ENV)
    if raw is None:
        return _DEFAULT_WORKER_LOG_VERBOSITY
    try:
        return max(int(raw), _DEFAULT_WORKER_LOG_VERBOSITY)
    except ValueError:
        return _DEFAULT_WORKER_LOG_VERBOSITY


def _seed_extra_comfyui_args(*, comfy_smart_memory: bool) -> list[str]:
    """Build the base ``extra_comfyui_args`` for a ComfyUI-running child from the smart-memory policy.

    ComfyUI's ``--disable-smart-memory`` makes it offload every model to RAM after each job, so a
    back-to-back same-model job re-uploads the UNet/CLIP/VAE from RAM even when the worker asked hordelib
    to keep them resident (``defer_vram_unload``): the flag acts below worker retention. With smart memory
    on (the default, no flag) ComfyUI keeps weights device-resident across jobs. The parent's device-free
    governor and verified reclaim ladder remain the authoritative evictor and force an actual VRAM free on
    any idle child, so residency never overcommits the card. The flag is restored only when an operator
    opts out via ``comfy_smart_memory=False``.
    """
    if comfy_smart_memory:
        return []
    return ["--disable-smart-memory"]


class InferenceProcessEntryPoint(Protocol):
    """The signature a callable must have to serve as an inference process target."""

    def __call__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        inference_semaphore: Semaphore,
        disk_lock: Lock,
        aux_model_lock: Lock,
        vae_decode_semaphore: Semaphore,
        process_launch_identifier: int,
        *,
        low_memory_mode: bool = False,
        amd_gpu: bool = False,
        directml: int | None = None,
        vram_heavy_models: bool = False,
        dry_run_skip_inference: bool = False,
        dry_run_inference_delay: float = 1.0,
        comfy_smart_memory: bool = False,
    ) -> None:
        """Run an inference process until told to end."""


class SafetyProcessEntryPoint(Protocol):
    """The signature a callable must have to serve as a safety process target."""

    def __call__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        cpu_only: bool = True,
        *,
        device_index: int = 0,
        accelerator_kind: str | None = None,
        amd_gpu: bool = False,
        directml: int | None = None,
        dry_run_skip_safety: bool = False,
        comfy_smart_memory: bool = False,
    ) -> None:
        """Run a safety process until told to end."""


class PostProcessProcessEntryPoint(Protocol):
    """The signature a callable must have to serve as the dedicated post-processing process target."""

    def __call__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        device_index: int = 0,
        accelerator_kind: str | None = None,
        amd_gpu: bool = False,
        directml: int | None = None,
        dry_run_skip_post_processing: bool = False,
        comfy_smart_memory: bool = False,
    ) -> None:
        """Run a post-processing process until told to end."""


class VaeLaneProcessEntryPoint(Protocol):
    """The signature a callable must have to serve as the dedicated VAE lane process target."""

    def __call__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        device_index: int = 0,
        accelerator_kind: str | None = None,
        amd_gpu: bool = False,
        directml: int | None = None,
        dry_run_skip_vae_lane: bool = False,
        comfy_smart_memory: bool = False,
    ) -> None:
        """Run the VAE lane process until told to end."""


class ComponentProcessEntryPoint(Protocol):
    """The signature a callable must have to serve as the dedicated component lane process target."""

    def __call__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        device_index: int = 0,
        accelerator_kind: str | None = None,
        amd_gpu: bool = False,
        directml: int | None = None,
        horde_model_names: list[str] | None = None,
        dry_run_skip_component_lane: bool = False,
        comfy_smart_memory: bool = False,
    ) -> None:
        """Run the component lane process until told to end."""


class DownloadProcessEntryPoint(Protocol):
    """The signature a callable must have to serve as the background download process target."""

    def __call__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        download_bandwidth_semaphore: Semaphore,
        process_launch_identifier: int,
        *,
        nsfw: bool = True,
        allow_lora: bool = False,
        allow_controlnet: bool = False,
        allow_sdxl_controlnet: bool = False,
        allow_post_processing: bool = True,
        purge_loras: bool = False,
        amd_gpu: bool = False,
        directml: int | None = None,
        rate_limit_kbps: int | None = None,
        paused: bool = False,
    ) -> None:
        """Run the download process until told to end."""


def start_inference_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    inference_semaphore: Semaphore,
    disk_lock: Lock,
    aux_model_lock: Lock,
    vae_decode_semaphore: Semaphore,
    process_launch_identifier: int,
    *,
    device_index: int = 0,
    accelerator_kind: str | None = None,
    low_memory_mode: bool = False,
    amd_gpu: bool = False,
    directml: int | None = None,
    vram_heavy_models: bool = False,
    dry_run_skip_inference: bool = False,
    dry_run_inference_delay: float = 1.0,
    gpu_sampling_lease: Semaphore | None = None,
    expect_image_models: bool = True,
    comfy_smart_memory: bool = False,
) -> None:
    """Start an inference process.

    Args:
        process_id (int): The ID of the process. This is not the same as the PID.
        process_message_queue (ProcessQueue): The queue to send messages to the main process.
        pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
        inference_semaphore (Semaphore): The semaphore to use to limit concurrent inference.
        disk_lock (Lock): The lock to use for disk access.
        aux_model_lock (Lock): The lock to use for auxiliary model downloading.
        vae_decode_semaphore (Semaphore): The semaphore to use to limit concurrent VAE decoding.
        process_launch_identifier (int): The unique identifier for this launch.
        device_index (int, optional): The stable index of the GPU this process is assigned to. Reported back \
            on memory messages so the parent can attribute VRAM per card. Defaults to 0.
        accelerator_kind (str | None, optional): The accelerator backend of the assigned device \
            (``cuda``/``rocm``/...), used to pin the process to its card. None applies no pinning. \
            Defaults to None.
        low_memory_mode (bool, optional): If true, the process will attempt to use less memory. Defaults to True.
        amd_gpu (bool, optional): If true, the process will attempt to use AMD GPU-specific optimisations.
            Defaults to False.
        directml (int | None, optional): If not None, the process will attempt to use DirectML \
            with the specified device
        vram_heavy_models (bool, optional): If true, the process will attempt to reserve more VRAM. Defaults to False.
        dry_run_skip_inference (bool, optional): If true, skip real inference and return a dummy image.
            Defaults to False.
        dry_run_inference_delay (float, optional): Seconds to sleep when dry-run inference is active. Defaults to 1.0.
        gpu_sampling_lease (Semaphore | None, optional): Shared lease for cross-process GPU sampling
            coordination, registered with hordelib. None disables it. Defaults to None.
        expect_image_models (bool, optional): Whether this worker serves image generation. False for an
            alchemist-only worker (e.g. a CPU install) that loads no image models, so an empty image-model
            database is expected rather than a fatal error. Defaults to True.
        comfy_smart_memory (bool, optional): Keep ComfyUI's smart memory management on so model weights stay
            device-resident across jobs. False restores the old ``--disable-smart-memory`` behavior that
            offloads every model to RAM after each job. Defaults to False.
    """
    _spawn_timing_mark(process_id, "inference", "entry")
    # Must precede the first torch/hordelib import below so the allocator reads it, and the device mask
    # must be applied before that too so torch only ever sees this process's assigned card.
    if not dry_run_skip_inference:
        _apply_device_pin(process_id=process_id, device_index=device_index, accelerator_kind=accelerator_kind)
        _enable_expandable_segments(amd_gpu=amd_gpu, directml=directml)
    enable_child_faulthandler(f"inference_{process_id}")
    neutralize_inherited_argv()
    with contextlib.nullcontext():  # contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        logger.remove()
        maybe_wait_for_process_debugger(process_id, "inference")

        try:
            # Before the first hordelib import: its import-time logfire init must defer to ours,
            # and OpenTelemetry tracing must be forced off by default. hordelib spans every
            # ComfyUI internal op (hundreds/job); with no collector the SDK still processes them
            # on GIL-contending threads and starves the inference loop. See telemetry.py.
            from horde_worker_regen.telemetry import claim_logfire_ownership, enforce_telemetry_default_off

            claim_logfire_ownership()
            enforce_telemetry_default_off()

            from hordelib.api import HordeLog

            _spawn_timing_mark(process_id, "inference", "imported-hordelib-api")

            HordeLog.initialise(
                setup_logging=True,
                process_id=process_id,
                verbosity_count=resolve_worker_log_verbosity(),
            )

            _spawn_timing_mark(process_id, "inference", "hordelog-initialised")

            from horde_worker_regen.telemetry import configure_child_telemetry

            configure_child_telemetry(process_id)

            if not dry_run_skip_inference:
                import hordelib

                logger.debug(
                    f"Initialising hordelib with process_id={process_id}, "
                    f"process_launch_identifier={process_launch_identifier}, "
                    f"amd_gpu={amd_gpu}, low_memory_mode={low_memory_mode}",
                )

                extra_comfyui_args = _seed_extra_comfyui_args(comfy_smart_memory=comfy_smart_memory)

                if amd_gpu:
                    extra_comfyui_args.append("--use-pytorch-cross-attention")

                if directml is not None:
                    extra_comfyui_args.append(f"--directml={directml}")

                from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

                # Force-load policy is expressed in horde baselines; hordelib owns the
                # mapping to comfy model class names.
                models_not_to_force_load: list[str] = [KNOWN_IMAGE_GENERATION_BASELINE.flux_1]

                if low_memory_mode:
                    extra_comfyui_args.append("--novram")
                    models_not_to_force_load.extend(
                        [
                            KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
                            KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade,
                        ],
                    )
                elif not vram_heavy_models:
                    logger.info("Reserving 1.4GB VRAM.")
                    extra_comfyui_args.extend(["--reserve-vram", "1.4"])

                if "--reserve-vram" not in extra_comfyui_args:
                    logger.warning("No VRAM reservation specified.")

                with logger.catch(reraise=True):
                    logger.debug(f"Using extra comfyui args: {extra_comfyui_args}")
                    hordelib.initialise(
                        setup_logging=None,
                        process_id=process_id,
                        logging_verbosity=0,
                        force_normal_vram_mode=False,
                        models_not_to_force_load=models_not_to_force_load,
                        extra_comfyui_args=extra_comfyui_args,
                    )
            else:
                logger.info(f"Dry-run mode: skipping hordelib initialisation for process {process_id}")

        except Exception as e:
            # logger.critical reaches nowhere when the crash precedes HordeLog.initialise (no sink yet);
            # the startup file is the loguru-independent backstop for that window.
            logger.critical(f"Failed to initialise hordelib: {type(e).__name__} {e}")
            write_startup_crash(
                f"inference_{process_id}",
                e,
                os_pid=os.getpid(),
                launch_identifier=process_launch_identifier,
            )
            sys.exit(1)

        _spawn_timing_mark(process_id, "inference", "hordelib-initialised")

        from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess

        worker_process = HordeInferenceProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            inference_semaphore=inference_semaphore,
            disk_lock=disk_lock,
            aux_model_lock=aux_model_lock,
            vae_decode_semaphore=vae_decode_semaphore,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
            dry_run_skip_inference=dry_run_skip_inference,
            dry_run_inference_delay=dry_run_inference_delay,
            gpu_sampling_lease=gpu_sampling_lease,
            expect_image_models=expect_image_models,
        )

        _spawn_timing_mark(process_id, "inference", "process-constructed")

        worker_process.main_loop()


def start_safety_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    process_launch_identifier: int,
    cpu_only: bool = True,
    *,
    device_index: int = 0,
    accelerator_kind: str | None = None,
    amd_gpu: bool = False,
    directml: int | None = None,
    dry_run_skip_safety: bool = False,
    comfy_smart_memory: bool = False,
) -> None:
    """Start a safety process.

    Args:
        process_id (int): The ID of the process. This is not the same as the PID of the system.
        process_message_queue (ProcessQueue): The queue to send messages to the main process.
        pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
        disk_lock (Lock): The lock to use for disk access.
        process_launch_identifier (int): The unique identifier for this launch.
        cpu_only (bool, optional): If true, the process will not use the GPU. Defaults to True.
        device_index (int, optional): The stable index of the GPU to pin to when running on-GPU (i.e. \
            ``cpu_only`` is False). On a multi-GPU host the safety model lives on the first configured card. \
            Defaults to 0.
        accelerator_kind (str | None, optional): The backend kind (``cuda``/``rocm``/...) of the assigned \
            card, used to pin the on-GPU safety process. None applies no pinning. Defaults to None.
        amd_gpu (bool, optional): If true, the process will attempt to use AMD GPU-specific optimizations.
            Defaults to False.
        directml (int | None, optional): If not None, the process will attempt to use DirectML \
            with the specified device
        dry_run_skip_safety (bool, optional): If true, skip real safety checks and return a dummy result.
            Defaults to False.
        comfy_smart_memory (bool, optional): Keep ComfyUI's smart memory management on so model weights stay
            device-resident across jobs. False restores the old ``--disable-smart-memory`` behavior that
            offloads every model to RAM after each job. Defaults to False.
    """
    _spawn_timing_mark(process_id, "safety", "entry")
    # The on-GPU safety model (cpu_only False) must be masked to its assigned card before torch loads, the
    # same as an inference process. cpu_only safety needs no GPU, so it is never pinned; the default
    # single-GPU on-GPU case passes accelerator_kind None and so also writes no env var (byte-identical).
    if not cpu_only and not dry_run_skip_safety:
        _apply_device_pin(
            process_id=process_id,
            device_index=device_index,
            accelerator_kind=accelerator_kind,
            role="safety",
        )
    enable_child_faulthandler(f"safety_{process_id}")
    neutralize_inherited_argv()
    with contextlib.nullcontext():  # contextlib.redirect_stdout(), contextlib.redirect_stderr():
        logger.remove()
        maybe_wait_for_process_debugger(process_id, "safety")

        try:
            # Before the first hordelib import: its import-time logfire init must defer to ours,
            # and OpenTelemetry tracing must be forced off by default. hordelib spans every
            # ComfyUI internal op (hundreds/job); with no collector the SDK still processes them
            # on GIL-contending threads and starves the inference loop. See telemetry.py.
            from horde_worker_regen.telemetry import claim_logfire_ownership, enforce_telemetry_default_off

            claim_logfire_ownership()
            enforce_telemetry_default_off()

            from hordelib.api import HordeLog

            _spawn_timing_mark(process_id, "safety", "imported-hordelib-api")

            HordeLog.initialise(
                setup_logging=True,
                process_id=process_id,
                verbosity_count=resolve_worker_log_verbosity(),
            )

            _spawn_timing_mark(process_id, "safety", "hordelog-initialised")

            from horde_worker_regen.telemetry import configure_child_telemetry

            configure_child_telemetry(process_id)

            logger.debug(f"Initialising hordelib with process_id={process_id}")

            extra_comfyui_args = _seed_extra_comfyui_args(comfy_smart_memory=comfy_smart_memory)

            if amd_gpu:
                extra_comfyui_args.append("--use-pytorch-cross-attention")

            if directml is not None:
                extra_comfyui_args.append(f"--directml={directml}")

        except Exception as e:
            logger.critical(f"Failed to initialise: {type(e).__name__} {e}")
            write_startup_crash(
                f"safety_{process_id}",
                e,
                os_pid=os.getpid(),
                launch_identifier=process_launch_identifier,
            )
            sys.exit(1)

        if not cpu_only and not dry_run_skip_safety:
            # Cap the on-GPU safety process's caching allocator so its evaluation models and retained pool
            # take only the share of the card their role justifies; an eval that cannot fit the cap faults
            # inside this process (which is recycled) rather than silently demand-paging the whole device.
            # After the device pin the process sees its card as device 0. A no-op on non-CUDA backends.
            from horde_worker_regen.utils.vram_quota import SAFETY_VRAM_QUOTA_MB, apply_process_vram_quota_mb

            apply_process_vram_quota_mb(SAFETY_VRAM_QUOTA_MB, device_index=0)

        from horde_worker_regen.process_management.workers.safety_process import HordeSafetyProcess

        logger.debug(
            f"Initialising hordelib with process_id={process_id}, "
            f"process_launch_identifier={process_launch_identifier}, "
            f"cpu_only={cpu_only} and amd_gpu={amd_gpu}",
        )
        worker_process = HordeSafetyProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            cpu_only=cpu_only,
            dry_run_skip_safety=dry_run_skip_safety,
        )

        worker_process.main_loop()


def start_post_process_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    process_launch_identifier: int,
    *,
    device_index: int = 0,
    accelerator_kind: str | None = None,
    amd_gpu: bool = False,
    directml: int | None = None,
    dry_run_skip_post_processing: bool = False,
    comfy_smart_memory: bool = False,
) -> None:
    """Start the dedicated post-processing process.

    Mirrors the inference process's hordelib bring-up (the post-processing graphs run on the same comfy
    backend) but never loads an image-generation checkpoint. The process is pinned to its assigned card
    before torch loads, the same as an inference process.

    Args:
        process_id (int): The reserved id for this process.
        process_message_queue (ProcessQueue): The queue to send messages to the main process.
        pipe_connection (Connection): Receives ``HordeControlMessage``s from the main process.
        disk_lock (Lock): The lock to use for disk access.
        process_launch_identifier (int): The unique identifier for this launch.
        device_index (int, optional): The stable index of the GPU this process is assigned to. Defaults to 0.
        accelerator_kind (str | None, optional): The backend kind (``cuda``/``rocm``/...) used to pin the \
            process to its card. None applies no pinning. Defaults to None.
        amd_gpu (bool, optional): Whether this is an AMD GPU. Defaults to False.
        directml (int | None, optional): The DirectML device index, if any. Defaults to None.
        dry_run_skip_post_processing (bool, optional): Skip real post-processing (and hordelib init) and \
            echo images back. Defaults to False.
        comfy_smart_memory (bool, optional): Keep ComfyUI's smart memory management on so model weights stay
            device-resident across jobs. False restores the old ``--disable-smart-memory`` behavior that
            offloads every model to RAM after each job. Defaults to False.
    """
    _spawn_timing_mark(process_id, "post_process", "entry")
    if not dry_run_skip_post_processing:
        _apply_device_pin(
            process_id=process_id,
            device_index=device_index,
            accelerator_kind=accelerator_kind,
            role="post_process",
        )
        _enable_expandable_segments(amd_gpu=amd_gpu, directml=directml)
    enable_child_faulthandler(f"post_process_{process_id}")
    neutralize_inherited_argv()
    with contextlib.nullcontext():
        logger.remove()
        maybe_wait_for_process_debugger(process_id, "post_process")

        try:
            from horde_worker_regen.telemetry import claim_logfire_ownership, enforce_telemetry_default_off

            claim_logfire_ownership()
            enforce_telemetry_default_off()

            from hordelib.api import HordeLog

            HordeLog.initialise(
                setup_logging=True,
                process_id=process_id,
                verbosity_count=resolve_worker_log_verbosity(),
            )

            from horde_worker_regen.telemetry import configure_child_telemetry

            configure_child_telemetry(process_id)

            if not dry_run_skip_post_processing:
                import hordelib

                extra_comfyui_args = _seed_extra_comfyui_args(comfy_smart_memory=comfy_smart_memory)
                if amd_gpu:
                    extra_comfyui_args.append("--use-pytorch-cross-attention")
                if directml is not None:
                    extra_comfyui_args.append(f"--directml={directml}")

                with logger.catch(reraise=True):
                    hordelib.initialise(
                        setup_logging=None,
                        process_id=process_id,
                        logging_verbosity=0,
                        force_normal_vram_mode=False,
                        extra_comfyui_args=extra_comfyui_args,
                    )
            else:
                logger.info(f"Dry-run mode: skipping hordelib initialisation for post-process process {process_id}")

        except Exception as e:
            logger.critical(f"Failed to initialise post-process process: {type(e).__name__} {e}")
            write_startup_crash(
                f"post_process_{process_id}",
                e,
                os_pid=os.getpid(),
                launch_identifier=process_launch_identifier,
            )
            sys.exit(1)

        if not dry_run_skip_post_processing:
            # Cap the lane's caching allocator as a runaway/leak guard: sized above the largest realistic
            # chain where the card allows, so legitimate upscale/face-fix jobs run in VRAM, while still
            # bounding a pool that would otherwise squat the card between chains. Deciding *when* a chain
            # co-resides with sampling is the orchestrator's admission gate, not this cap. After the device
            # pin the lane sees its card as device 0. A no-op on non-CUDA backends.
            from horde_worker_regen.utils.vram_quota import apply_post_process_vram_quota

            apply_post_process_vram_quota(device_index=0)

        from horde_worker_regen.process_management.workers.post_process_process import HordePostProcessProcess

        worker_process = HordePostProcessProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
            dry_run_skip_post_processing=dry_run_skip_post_processing,
        )

        worker_process.main_loop()


def start_vae_lane_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    process_launch_identifier: int,
    *,
    device_index: int = 0,
    accelerator_kind: str | None = None,
    amd_gpu: bool = False,
    directml: int | None = None,
    dry_run_skip_vae_lane: bool = False,
    comfy_smart_memory: bool = False,
) -> None:
    """Start the dedicated VAE lane process.

    Mirrors the post-processing lane's hordelib bring-up (the VAE stages and the decode's optional
    post-processing graphs run on the same comfy backend) but never loads an image-generation checkpoint.
    The process is pinned to its assigned card before torch loads, the same as an inference process.

    Args:
        process_id (int): The reserved id for this process.
        process_message_queue (ProcessQueue): The queue to send messages to the main process.
        pipe_connection (Connection): Receives ``HordeControlMessage``s from the main process.
        disk_lock (Lock): The lock to use for disk access.
        process_launch_identifier (int): The unique identifier for this launch.
        device_index (int, optional): The stable index of the GPU this process is assigned to. Defaults to 0.
        accelerator_kind (str | None, optional): The backend kind (``cuda``/``rocm``/...) used to pin the \
            process to its card. None applies no pinning. Defaults to None.
        amd_gpu (bool, optional): Whether this is an AMD GPU. Defaults to False.
        directml (int | None, optional): The DirectML device index, if any. Defaults to None.
        dry_run_skip_vae_lane (bool, optional): Skip the backend (and hordelib init) and return plausible \
            stand-in latent/image bytes. Defaults to False.
        comfy_smart_memory (bool, optional): Keep ComfyUI's smart memory management on so model weights stay
            device-resident across jobs. False restores the old ``--disable-smart-memory`` behavior that
            offloads every model to RAM after each job. Defaults to False.
    """
    _spawn_timing_mark(process_id, "vae_lane", "entry")
    if not dry_run_skip_vae_lane:
        _apply_device_pin(
            process_id=process_id,
            device_index=device_index,
            accelerator_kind=accelerator_kind,
            role="vae_lane",
        )
        _enable_expandable_segments(amd_gpu=amd_gpu, directml=directml)
    enable_child_faulthandler(f"vae_lane_{process_id}")
    neutralize_inherited_argv()
    with contextlib.nullcontext():
        logger.remove()
        maybe_wait_for_process_debugger(process_id, "vae_lane")

        try:
            from horde_worker_regen.telemetry import claim_logfire_ownership, enforce_telemetry_default_off

            claim_logfire_ownership()
            enforce_telemetry_default_off()

            from hordelib.api import HordeLog

            HordeLog.initialise(
                setup_logging=True,
                process_id=process_id,
                verbosity_count=resolve_worker_log_verbosity(),
            )

            from horde_worker_regen.telemetry import configure_child_telemetry

            configure_child_telemetry(process_id)

            if not dry_run_skip_vae_lane:
                import hordelib

                extra_comfyui_args = _seed_extra_comfyui_args(comfy_smart_memory=comfy_smart_memory)
                if amd_gpu:
                    extra_comfyui_args.append("--use-pytorch-cross-attention")
                if directml is not None:
                    extra_comfyui_args.append(f"--directml={directml}")

                with logger.catch(reraise=True):
                    hordelib.initialise(
                        setup_logging=None,
                        process_id=process_id,
                        logging_verbosity=0,
                        force_normal_vram_mode=False,
                        extra_comfyui_args=extra_comfyui_args,
                    )
            else:
                logger.info(f"Dry-run mode: skipping hordelib initialisation for VAE lane {process_id}")

        except Exception as e:
            logger.critical(f"Failed to initialise VAE lane process: {type(e).__name__} {e}")
            write_startup_crash(
                f"vae_lane_{process_id}",
                e,
                os_pid=os.getpid(),
                launch_identifier=process_launch_identifier,
            )
            sys.exit(1)

        from horde_worker_regen.process_management.workers.vae_lane_process import HordeVaeLaneProcess

        worker_process = HordeVaeLaneProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
            dry_run=dry_run_skip_vae_lane,
        )

        worker_process.main_loop()


def start_component_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    process_launch_identifier: int,
    *,
    device_index: int = 0,
    accelerator_kind: str | None = None,
    amd_gpu: bool = False,
    directml: int | None = None,
    horde_model_names: list[str] | None = None,
    dry_run_skip_component_lane: bool = False,
    comfy_smart_memory: bool = False,
) -> None:
    """Start the dedicated component lane process.

    Mirrors the post-processing lane's hordelib bring-up, then installs the sharing client in PRODUCER role
    (the lane is the single stable producer) before constructing the process, which materialises and publishes
    the hot-set. The lane is pinned to its card before torch loads.

    Args:
        process_id (int): The reserved id for this process.
        process_message_queue (ProcessQueue): The queue to send messages to the main process.
        pipe_connection (Connection): Receives ``HordeControlMessage``s from the main process.
        disk_lock (Lock): The lock to use for disk access.
        process_launch_identifier (int): The unique identifier for this launch.
        device_index (int, optional): The stable index of the GPU this process is assigned to. Defaults to 0.
        accelerator_kind (str | None, optional): The backend kind (``cuda``/``rocm``/...) used to pin the \
            process to its card. None applies no pinning. Defaults to None.
        amd_gpu (bool, optional): Whether this is an AMD GPU. Defaults to False.
        directml (int | None, optional): The DirectML device index, if any. Defaults to None.
        horde_model_names (list[str] | None, optional): The worker's configured models; the lane holds the \
            components shared across them. Defaults to None.
        dry_run_skip_component_lane (bool, optional): Skip the backend and materialisation. Defaults to False.
        comfy_smart_memory (bool, optional): Keep ComfyUI's smart memory management on so model weights stay
            device-resident across jobs. False restores the old ``--disable-smart-memory`` behavior that
            offloads every model to RAM after each job. Defaults to False.
    """
    _spawn_timing_mark(process_id, "component", "entry")
    if not dry_run_skip_component_lane:
        _apply_device_pin(
            process_id=process_id,
            device_index=device_index,
            accelerator_kind=accelerator_kind,
            role="component",
        )
        _enable_expandable_segments(amd_gpu=amd_gpu, directml=directml)
    enable_child_faulthandler(f"component_{process_id}")
    neutralize_inherited_argv()
    with contextlib.nullcontext():
        logger.remove()
        maybe_wait_for_process_debugger(process_id, "component")

        try:
            from horde_worker_regen.telemetry import claim_logfire_ownership, enforce_telemetry_default_off

            claim_logfire_ownership()
            enforce_telemetry_default_off()

            from hordelib.api import HordeLog

            HordeLog.initialise(
                setup_logging=True,
                process_id=process_id,
                verbosity_count=resolve_worker_log_verbosity(),
            )

            from horde_worker_regen.telemetry import configure_child_telemetry

            configure_child_telemetry(process_id)

            if not dry_run_skip_component_lane:
                import hordelib

                extra_comfyui_args = _seed_extra_comfyui_args(comfy_smart_memory=comfy_smart_memory)
                if amd_gpu:
                    extra_comfyui_args.append("--use-pytorch-cross-attention")
                if directml is not None:
                    extra_comfyui_args.append(f"--directml={directml}")

                with logger.catch(reraise=True):
                    hordelib.initialise(
                        setup_logging=None,
                        process_id=process_id,
                        logging_verbosity=0,
                        force_normal_vram_mode=False,
                        extra_comfyui_args=extra_comfyui_args,
                    )
            else:
                logger.info(f"Dry-run mode: skipping hordelib initialisation for component lane {process_id}")

        except Exception as e:
            logger.critical(f"Failed to initialise component lane process: {type(e).__name__} {e}")
            write_startup_crash(
                f"component_{process_id}",
                e,
                os_pid=os.getpid(),
                launch_identifier=process_launch_identifier,
            )
            sys.exit(1)

        from horde_worker_regen.process_management.workers.component_lane_process import HordeComponentLaneProcess

        worker_process = HordeComponentLaneProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
            device_index=device_index,
            horde_model_names=horde_model_names,
            dry_run=dry_run_skip_component_lane,
        )

        worker_process.main_loop()


def start_download_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    download_bandwidth_semaphore: Semaphore,
    process_launch_identifier: int,
    *,
    nsfw: bool = True,
    allow_lora: bool = False,
    allow_controlnet: bool = False,
    allow_sdxl_controlnet: bool = False,
    allow_post_processing: bool = True,
    purge_loras: bool = False,
    amd_gpu: bool = False,
    directml: int | None = None,
    rate_limit_kbps: int | None = None,
    paused: bool = False,
    max_parallel_downloads: int = 4,
    per_host_concurrency: int = 1,
    connections_per_file: int = 4,
) -> None:
    """Start the background model-download process.

    Args:
        process_id (int): The reserved id for this process (see ``download_process.DOWNLOAD_PROCESS_ID``).
        process_message_queue (ProcessQueue): The queue to send messages to the main process.
        pipe_connection (Connection): Receives ``HordeControlMessage``s from the main process.
        disk_lock (Lock): Coordinates disk access with the inference/safety processes.
        download_bandwidth_semaphore (Semaphore): Held while this process is actively downloading.
        process_launch_identifier (int): The unique identifier for this launch.
        nsfw (bool): Whether NSFW default LoRas may be fetched. Defaults to True.
        allow_lora (bool): Whether to fetch the default LoRas during an aux pass. Defaults to False.
        allow_controlnet (bool): Whether to fetch ControlNet models/annotators. Defaults to False.
        allow_sdxl_controlnet (bool): Whether to fetch SDXL ControlNet/miscellaneous models. Defaults to False.
        allow_post_processing (bool): Whether to fetch post-processing models. Defaults to True.
        purge_loras (bool): Whether to purge unused LoRas during an aux pass. Defaults to False.
        amd_gpu (bool): Whether this is an AMD GPU. Defaults to False.
        directml (int | None): The DirectML device index, if any. Defaults to None.
        rate_limit_kbps (int | None): Initial bandwidth cap in KB/s (None/0 = unlimited). Defaults to None.
        paused (bool): Whether downloads start paused. Defaults to False.
        max_parallel_downloads (int): Global concurrent-download ceiling across all hosts. Defaults to 4.
        per_host_concurrency (int): Concurrent downloads allowed per source host. Defaults to 1.
        connections_per_file (int): Max concurrent connections used to fetch a single large file. Defaults to 4.
    """
    enable_child_faulthandler(f"download_{process_id}")
    neutralize_inherited_argv()
    with contextlib.nullcontext():
        logger.remove()
        maybe_wait_for_process_debugger(process_id, "download")

        try:
            from horde_worker_regen.telemetry import claim_logfire_ownership, enforce_telemetry_default_off

            claim_logfire_ownership()
            enforce_telemetry_default_off()

            from hordelib.api import HordeLog

            HordeLog.initialise(
                setup_logging=True,
                process_id=process_id,
                verbosity_count=resolve_worker_log_verbosity(),
            )

            from horde_worker_regen.telemetry import configure_child_telemetry

            configure_child_telemetry(process_id)
        except Exception as e:
            logger.critical(f"Failed to initialise download process: {type(e).__name__} {e}")
            write_startup_crash(
                f"download_{process_id}",
                e,
                os_pid=os.getpid(),
                launch_identifier=process_launch_identifier,
            )
            sys.exit(1)

        from horde_worker_regen.process_management.workers.download_process import HordeDownloadProcess

        worker_process = HordeDownloadProcess(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            download_bandwidth_semaphore=download_bandwidth_semaphore,
            process_launch_identifier=process_launch_identifier,
            nsfw=nsfw,
            allow_lora=allow_lora,
            allow_controlnet=allow_controlnet,
            allow_sdxl_controlnet=allow_sdxl_controlnet,
            allow_post_processing=allow_post_processing,
            purge_loras=purge_loras,
            amd_gpu=amd_gpu,
            directml=directml,
            rate_limit_kbps=rate_limit_kbps,
            paused=paused,
            max_parallel_downloads=max_parallel_downloads,
            per_host_concurrency=per_host_concurrency,
            connections_per_file=connections_per_file,
        )

        worker_process.main_loop()


class ProcessEntryPoints:
    """The multiprocessing targets used when launching child worker processes.

    The defaults are the real (hordelib-backed) entry points. Test harnesses can
    substitute the fakes from ``fake_worker_processes`` to exercise the
    orchestration layer without the ML dependency stack.

    Entry points must be module-level functions (or otherwise picklable) so they
    survive the trip to a spawned child process.
    """

    inference_entry_point: InferenceProcessEntryPoint
    safety_entry_point: SafetyProcessEntryPoint
    post_process_entry_point: PostProcessProcessEntryPoint
    vae_lane_entry_point: VaeLaneProcessEntryPoint
    component_entry_point: ComponentProcessEntryPoint
    download_entry_point: DownloadProcessEntryPoint

    def __init__(
        self,
        *,
        inference_entry_point: InferenceProcessEntryPoint | None = None,
        safety_entry_point: SafetyProcessEntryPoint | None = None,
        post_process_entry_point: PostProcessProcessEntryPoint | None = None,
        vae_lane_entry_point: VaeLaneProcessEntryPoint | None = None,
        component_entry_point: ComponentProcessEntryPoint | None = None,
        download_entry_point: DownloadProcessEntryPoint | None = None,
    ) -> None:
        """Initialise with the given entry points, defaulting to the real ones.

        Args:
            inference_entry_point (InferenceProcessEntryPoint | None, optional): The target for \
                inference processes. Defaults to `start_inference_process`.
            safety_entry_point (SafetyProcessEntryPoint | None, optional): The target for \
                safety processes. Defaults to `start_safety_process`.
            post_process_entry_point (PostProcessProcessEntryPoint | None, optional): The target for \
                the dedicated post-processing process. Defaults to `start_post_process_process`.
            vae_lane_entry_point (VaeLaneProcessEntryPoint | None, optional): The target for \
                the dedicated VAE lane process. Defaults to `start_vae_lane_process`.
            component_entry_point (ComponentProcessEntryPoint | None, optional): The target for \
                the dedicated component lane process. Defaults to `start_component_process`.
            download_entry_point (DownloadProcessEntryPoint | None, optional): The target for \
                the background download process. Defaults to `start_download_process`.
        """
        self.inference_entry_point = (
            inference_entry_point if inference_entry_point is not None else start_inference_process
        )
        self.safety_entry_point = safety_entry_point if safety_entry_point is not None else start_safety_process
        self.post_process_entry_point = (
            post_process_entry_point if post_process_entry_point is not None else start_post_process_process
        )
        self.vae_lane_entry_point = (
            vae_lane_entry_point if vae_lane_entry_point is not None else start_vae_lane_process
        )
        self.component_entry_point = (
            component_entry_point if component_entry_point is not None else start_component_process
        )
        self.download_entry_point = (
            download_entry_point if download_entry_point is not None else start_download_process
        )
