"""Launch rendering: backend-independent settings become one backend's command line, keyed by horde name."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from horde_model_reference.text_backend_names import TEXT_BACKENDS
from pydantic import ValidationError

from horde_worker_regen.compute_mode import TextBackendAccelerator
from horde_worker_regen.text_backends.koboldcpp_launch import KoboldcppArguments
from horde_worker_regen.text_backends.launch import (
    LAUNCH_SPEC_BUILDERS,
    UnsupportedTextBackendError,
    build_launch_spec,
    launchable_backends,
)
from horde_worker_regen.text_backends.launch_spec import LOOPBACK_HOST, TextBackendLaunchSettings


def settings_in(
    tmp_path: Path,
    *,
    device_index: int | None = 1,
    accelerator: TextBackendAccelerator = TextBackendAccelerator.CUDA,
    parallel_requests: int = 1,
) -> TextBackendLaunchSettings:
    """Return launch settings pointing at files under ``tmp_path``."""
    return TextBackendLaunchSettings(
        executable=tmp_path / "backend.exe",
        model_path=tmp_path / "model.gguf",
        port=5001,
        device_index=device_index,
        gpu_layers=99,
        context_length=4096,
        log_path=tmp_path / "text_backend.log",
        accelerator=accelerator,
        parallel_requests=parallel_requests,
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


def test_koboldcpp_states_cuda_without_a_device_when_no_card_is_assigned(tmp_path: Path) -> None:
    """Without a device index the path is still stated, bare, and the footprint device is None."""
    spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings_in(tmp_path, device_index=None))

    index = spec.arguments.index(KoboldcppArguments.USE_CUDA)
    assert spec.arguments[index + 1] == KoboldcppArguments.GPU_LAYERS
    assert spec.device_index is None


def test_koboldcpp_refuses_a_cuda_device_id_upstream_does_not_list(tmp_path: Path) -> None:
    """Upstream's parser fixes the CUDA ids as choices, so a higher index fails here with the key named."""
    with pytest.raises(ValueError, match="text_gpu_device_index"):
        build_launch_spec(TEXT_BACKENDS.koboldcpp, settings_in(tmp_path, device_index=4))


class TestKoboldcppAcceleratorRendering:
    """Each compute path renders the flag koboldcpp accepts for it, and only that one."""

    def test_the_default_cuda_line_is_unchanged(self, tmp_path: Path) -> None:
        """A single-generation CUDA launch renders exactly the command line it rendered before."""
        settings = settings_in(tmp_path)

        spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings)

        assert spec.arguments == (
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
        )

    def test_vulkan_names_the_device_on_its_own_flag(self, tmp_path: Path) -> None:
        """An AMD or Intel host selects Vulkan, which takes the device id on its own flag."""
        settings = settings_in(tmp_path, accelerator=TextBackendAccelerator.VULKAN)

        spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings)

        assert KoboldcppArguments.USE_CUDA not in spec.arguments
        index = spec.arguments.index(KoboldcppArguments.USE_VULKAN)
        assert spec.arguments[index + 1] == "1"
        assert spec.device_index == 1

    def test_vulkan_without_a_device_index_still_selects_vulkan(self, tmp_path: Path) -> None:
        """Koboldcpp given no backend flag picks a path itself, so the flag is stated bare."""
        settings = settings_in(tmp_path, accelerator=TextBackendAccelerator.VULKAN, device_index=None)

        spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings)

        assert KoboldcppArguments.USE_CUDA not in spec.arguments
        index = spec.arguments.index(KoboldcppArguments.USE_VULKAN)
        assert spec.arguments[index + 1 : index + 3] == (KoboldcppArguments.GPU_LAYERS, "99")

    def test_vulkan_accepts_a_device_id_cuda_would_refuse(self, tmp_path: Path) -> None:
        """Only the CUDA flag has fixed choices upstream; Vulkan ids are plain integers."""
        settings = settings_in(tmp_path, accelerator=TextBackendAccelerator.VULKAN, device_index=5)

        spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings)

        assert spec.arguments[spec.arguments.index(KoboldcppArguments.USE_VULKAN) + 1] == "5"

    @pytest.mark.parametrize("device_index", [1, None])
    def test_cpu_says_so_and_offloads_nothing(self, tmp_path: Path, device_index: int | None) -> None:
        """A CPU launch names the CPU path; with no backend flag koboldcpp would pick a device itself."""
        settings = settings_in(tmp_path, accelerator=TextBackendAccelerator.CPU, device_index=device_index)

        spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings)

        assert KoboldcppArguments.USE_CUDA not in spec.arguments
        assert KoboldcppArguments.USE_VULKAN not in spec.arguments
        assert KoboldcppArguments.USE_CPU in spec.arguments
        index = spec.arguments.index(KoboldcppArguments.GPU_LAYERS)
        assert spec.arguments[index + 1] == "0"

    def test_one_parallel_request_renders_nothing(self, tmp_path: Path) -> None:
        """One is koboldcpp's own default, so the flag would only restate it."""
        spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings_in(tmp_path, parallel_requests=1))

        assert KoboldcppArguments.PARALLEL_REQUESTS not in spec.arguments

    def test_several_parallel_requests_size_the_batch(self, tmp_path: Path) -> None:
        """`text_threads` above one needs the backend to serve that many generations at once."""
        spec = build_launch_spec(TEXT_BACKENDS.koboldcpp, settings_in(tmp_path, parallel_requests=4))

        index = spec.arguments.index(KoboldcppArguments.PARALLEL_REQUESTS)
        assert spec.arguments[index + 1] == "4"

    def test_an_unresolved_compute_path_is_refused(self, tmp_path: Path) -> None:
        """`auto` has no flag anywhere, so it is caught where it is set rather than rendered as something."""
        with pytest.raises(ValidationError):
            settings_in(tmp_path, accelerator=TextBackendAccelerator.AUTO)


def test_a_backend_without_a_renderer_is_refused_by_name(tmp_path: Path) -> None:
    """A horde backend name the worker cannot launch yet raises an error naming it and the supported list."""
    unsupported = next(kind for kind in TEXT_BACKENDS if kind not in LAUNCH_SPEC_BUILDERS)

    with pytest.raises(UnsupportedTextBackendError) as raised:
        build_launch_spec(unsupported, settings_in(tmp_path))

    assert raised.value.kind is unsupported
    assert str(unsupported) in str(raised.value)
    assert str(TEXT_BACKENDS.koboldcpp) in str(raised.value)
