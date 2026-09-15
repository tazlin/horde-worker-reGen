"""Render a text backend launch for koboldcpp.

This is the koboldcpp half of a launch: the flag spellings upstream's argument parser accepts and the
choices the worker makes when running it unattended. Everything backend-independent is in
[`launch_spec`][horde_worker_regen.text_backends.launch_spec]; the supervisor never imports this module
directly, it receives the rendered spec through
[`launch.build_launch_spec`][horde_worker_regen.text_backends.launch.build_launch_spec].
"""

from __future__ import annotations

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
    GPU_LAYERS = "--gpulayers"
    CONTEXT_SIZE = "--contextsize"
    SKIP_LAUNCHER = "--skiplauncher"
    QUIET = "--quiet"


def koboldcpp_launch_spec(settings: TextBackendLaunchSettings) -> TextBackendLaunchSpec:
    """Create the koboldcpp command line for the given settings.

    koboldcpp's own embedded horde worker stays off (no API key is passed) because the worker pops. The
    launcher GUI is skipped and output is quietened because the process runs unattended. The device index
    becomes the CUDA ordinal: the parent sets ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` before any child starts, so
    the worker's stable index and koboldcpp's ordinal agree. A None device index passes no CUDA flag (a
    no-CUDA build, or a CPU-only host). koboldcpp caps an oversized layer count at the model's layer count,
    so a large `gpu_layers` means the whole model.
    """
    arguments: list[str] = [
        KoboldcppArguments.MODEL,
        str(settings.model_path),
        KoboldcppArguments.PORT,
        str(settings.port),
        KoboldcppArguments.HOST,
        LOOPBACK_HOST,
    ]
    if settings.device_index is not None:
        arguments.extend([KoboldcppArguments.USE_CUDA, str(settings.device_index)])
    arguments.extend(
        [
            KoboldcppArguments.GPU_LAYERS,
            str(settings.gpu_layers),
            KoboldcppArguments.CONTEXT_SIZE,
            str(settings.context_length),
            KoboldcppArguments.SKIP_LAUNCHER,
            KoboldcppArguments.QUIET,
        ],
    )
    return TextBackendLaunchSpec(
        executable=settings.executable,
        arguments=tuple(arguments),
        port=settings.port,
        log_path=settings.log_path,
        device_index=settings.device_index,
    )


__all__ = ["KoboldcppArguments", "koboldcpp_launch_spec"]
