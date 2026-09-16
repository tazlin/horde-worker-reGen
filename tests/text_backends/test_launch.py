"""Launch rendering: backend-independent settings become one backend's command line, keyed by horde name."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from horde_model_reference.text_backend_names import TEXT_BACKENDS

from horde_worker_regen.text_backends.koboldcpp_launch import KoboldcppArguments
from horde_worker_regen.text_backends.launch import (
    LAUNCH_SPEC_BUILDERS,
    UnsupportedTextBackendError,
    build_launch_spec,
    launchable_backends,
)
from horde_worker_regen.text_backends.launch_spec import LOOPBACK_HOST, TextBackendLaunchSettings


def settings_in(tmp_path: Path, *, device_index: int | None = 1) -> TextBackendLaunchSettings:
    """Return launch settings pointing at files under ``tmp_path``."""
    return TextBackendLaunchSettings(
        executable=tmp_path / "backend.exe",
        model_path=tmp_path / "model.gguf",
        port=5001,
        device_index=device_index,
        gpu_layers=99,
        context_length=4096,
        log_path=tmp_path / "text_backend.log",
    )


def test_every_launchable_backend_is_a_horde_backend_name() -> None:
    """The registry is keyed by the reference package's backend vocabulary, so names never drift."""
    assert launchable_backends() == frozenset(LAUNCH_SPEC_BUILDERS)
    assert all(isinstance(kind, TEXT_BACKENDS) for kind in launchable_backends())
    assert TEXT_BACKENDS.koboldcpp in launchable_backends()


def test_koboldcpp_renders_the_expected_argv(tmp_path: Path) -> None:
    """Koboldcpp's argv carries the model, loopback port and host, CUDA ordinal, layers, context, and quiet flags."""
    settings = settings_in(tmp_path)

    spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings)

    assert spec.command == [
        str(settings.executable),
        KoboldcppArguments.MODEL,
        str(settings.model_path),
        KoboldcppArguments.PORT,
        "5001",
        KoboldcppArguments.HOST,
        LOOPBACK_HOST,
        KoboldcppArguments.USE_CUDA,
        "1",
        KoboldcppArguments.GPU_LAYERS,
        "99",
        KoboldcppArguments.CONTEXT_SIZE,
        "4096",
        KoboldcppArguments.SKIP_LAUNCHER,
        KoboldcppArguments.QUIET,
    ]
    assert spec.base_url == f"http://{LOOPBACK_HOST}:5001"
    assert spec.device_index == 1
    assert spec.log_path == settings.log_path


def test_a_script_executable_runs_under_the_worker_interpreter(tmp_path: Path) -> None:
    """A `.py` executable (a source checkout) is launched by the worker's own Python, arguments unchanged."""
    settings = settings_in(tmp_path).model_copy(update={"executable": tmp_path / "koboldcpp.py"})

    spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings)

    assert spec.runs_from_source
    assert spec.command[:2] == [sys.executable, str(settings.executable)]
    assert tuple(spec.command[2:]) == spec.arguments


def test_a_binary_executable_is_the_command_itself(tmp_path: Path) -> None:
    """A binary executable starts directly; no interpreter is prepended."""
    spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings_in(tmp_path))

    assert not spec.runs_from_source
    assert spec.command[0] == str(spec.executable)


def test_koboldcpp_omits_the_cuda_flag_off_gpu(tmp_path: Path) -> None:
    """Without a device index the CUDA flag is absent and the footprint device is None."""
    spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings_in(tmp_path, device_index=None))

    assert KoboldcppArguments.USE_CUDA not in spec.arguments
    assert spec.device_index is None


def test_a_backend_without_a_renderer_is_refused_by_name(tmp_path: Path) -> None:
    """A horde backend name the worker cannot launch yet raises an error naming it and the supported list."""
    unsupported = next(kind for kind in TEXT_BACKENDS if kind not in LAUNCH_SPEC_BUILDERS)

    with pytest.raises(UnsupportedTextBackendError) as raised:
        build_launch_spec(unsupported, settings_in(tmp_path))

    assert raised.value.kind is unsupported
    assert str(unsupported) in str(raised.value)
    assert str(TEXT_BACKENDS.koboldcpp) in str(raised.value)
