"""Render a text backend launch for koboldcpp.

This is the koboldcpp half of a launch: the flag spellings upstream's argument parser accepts and the
choices the worker makes when running it unattended. Everything backend-independent is in
[`launch_spec`][horde_worker_regen.text_backends.launch_spec]; the supervisor never imports this module
directly, it receives the rendered spec through
[`launch.build_launch_spec`][horde_worker_regen.text_backends.launch.build_launch_spec].
"""

from __future__ import annotations

from horde_worker_regen.compute_mode import TextBackendAccelerator
from horde_worker_regen.text_backends.launch_spec import (
    LOOPBACK_HOST,
    TextBackendLaunchSettings,
    TextBackendLaunchSpec,
)


class KoboldcppArguments:
    """koboldcpp's command-line flags, as spelled by upstream's argument parser."""

    MODEL = "--model"
    PORT = "--port"
    HOST = "--host"
    USE_CUDA = "--usecuda"
    USE_VULKAN = "--usevulkan"
    USE_CPU = "--usecpu"
    GPU_LAYERS = "--gpulayers"
    CONTEXT_SIZE = "--contextsize"
    SKIP_LAUNCHER = "--skiplauncher"
    QUIET = "--quiet"
    PARALLEL_REQUESTS = "--parallelrequests"


_DEVICE_FLAGS: dict[TextBackendAccelerator, str] = {
    TextBackendAccelerator.CUDA: KoboldcppArguments.USE_CUDA,
    TextBackendAccelerator.VULKAN: KoboldcppArguments.USE_VULKAN,
}
"""The flag that both selects a compute path and names the device on it."""

_CUDA_DEVICE_IDS = range(4)
"""The device ids ``--usecuda`` accepts; upstream's parser lists them as fixed choices."""


def _device_arguments(settings: TextBackendLaunchSettings) -> list[str]:
    """Return the compute-path, device and offload arguments for the settings.

    The compute path is always stated, with or without a device id, because koboldcpp given no backend
    flag picks one itself and the binary variant was chosen for the path the settings name.
    """
    if settings.accelerator is TextBackendAccelerator.CPU:
        return [KoboldcppArguments.USE_CPU, KoboldcppArguments.GPU_LAYERS, "0"]
    arguments = [_DEVICE_FLAGS[settings.accelerator]]
    if settings.device_index is not None:
        if settings.accelerator is TextBackendAccelerator.CUDA and settings.device_index not in _CUDA_DEVICE_IDS:
            raise ValueError(
                f"koboldcpp's {KoboldcppArguments.USE_CUDA} accepts device ids {_CUDA_DEVICE_IDS.start} to "
                f"{_CUDA_DEVICE_IDS.stop - 1}; text_gpu_device_index {settings.device_index} is outside that.",
            )
        arguments.append(str(settings.device_index))
    arguments.extend([KoboldcppArguments.GPU_LAYERS, str(settings.gpu_layers)])
    return arguments


def koboldcpp_launch_spec(settings: TextBackendLaunchSettings) -> TextBackendLaunchSpec:
    """Create the koboldcpp command line for the given settings.

    koboldcpp's own embedded horde worker stays off (no API key is passed) because the worker pops. The
    launcher GUI is skipped and output is quietened because the process runs unattended. On CUDA the device
    index becomes the CUDA ordinal: the parent sets ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` before any child
    starts, so the worker's stable index and koboldcpp's ordinal agree. On Vulkan the device id is
    koboldcpp's own enumeration of what the Vulkan loader reports, which need not equal the worker's stable
    index, so the index is passed through unmapped and a multi-card host may have to name the other one. A
    None device index states the compute path without a device, which koboldcpp reads as every device on
    that path. koboldcpp caps an oversized layer count at the model's layer count, so a large `gpu_layers`
    means the whole model. One parallel request is koboldcpp's own default, so the flag is left off.
    """
    arguments: list[str] = [
        KoboldcppArguments.MODEL,
        str(settings.model_path),
        KoboldcppArguments.PORT,
        str(settings.port),
        KoboldcppArguments.HOST,
        LOOPBACK_HOST,
        *_device_arguments(settings),
        KoboldcppArguments.CONTEXT_SIZE,
        str(settings.context_length),
        KoboldcppArguments.SKIP_LAUNCHER,
        KoboldcppArguments.QUIET,
    ]
    if settings.parallel_requests > 1:
        arguments.extend([KoboldcppArguments.PARALLEL_REQUESTS, str(settings.parallel_requests)])
    return TextBackendLaunchSpec(
        executable=settings.executable,
        arguments=tuple(arguments),
        port=settings.port,
        log_path=settings.log_path,
        device_index=settings.device_index,
    )


__all__ = ["KoboldcppArguments", "koboldcpp_launch_spec"]
