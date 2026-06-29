"""Unit tests for cross-platform GPU/torch-build detection."""

from __future__ import annotations

import pytest

from worker_bootstrap import detect


@pytest.mark.parametrize(
    ("smi_output", "expected"),
    [
        ("NVIDIA-SMI 560.94   Driver Version: 560.94   CUDA Version: 12.8", (12, 8)),
        ("CUDA Version: 13.0", (13, 0)),
        ("CUDA Version: 13.2", (13, 2)),
        ("CUDA Version:13.2", (13, 2)),
        ("no cuda header here", (0, 0)),
        ("", (0, 0)),
    ],
)
def test_parse_cuda_version(smi_output: str, expected: tuple[int, int]) -> None:
    """The (major, minor) CUDA version is parsed from the nvidia-smi header ((0, 0) when absent)."""
    assert detect.parse_cuda_version(smi_output) == expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((12, 6), detect.CU126),
        ((12, 8), detect.CU126),
        ((13, 0), detect.CU130),
        ((13, 1), detect.CU130),  # 13.1 cannot run a 13.2-built wheel, so it stays on cu130
        ((13, 2), detect.CU132),
        ((13, 5), detect.CU132),
        ((14, 0), detect.CU132),  # a newer major still covers the 13.2 build via backward-compat
        ((0, 0), detect.CU126),  # version unreadable -> safe 12.6 build
    ],
)
def test_cuda_build(version: tuple[int, int], expected: str) -> None:
    """_cuda_build picks the newest locked CUDA build the driver's max CUDA version can run."""
    assert detect._cuda_build(version) == expected


@pytest.mark.parametrize(
    ("smi_output", "expected"),
    [
        ("8.6", (8, 6)),
        ("12.0\n", (12, 0)),
        ("8.9\n12.0\n", (12, 0)),  # mixed GPUs -> highest wins
        ("compute_cap\n7.5", (7, 5)),
        ("", (0, 0)),
        ("no caps here", (0, 0)),
    ],
)
def test_parse_compute_cap(smi_output: str, expected: tuple[int, int]) -> None:
    """parse_compute_cap returns the highest compute capability in the query output ((0, 0) when absent)."""
    assert detect.parse_compute_cap(smi_output) == expected


@pytest.mark.parametrize(
    ("version", "compute_cap", "expected"),
    [
        # Floor -- Blackwell (sm_120 / sm_100) has no kernel image in cu126, so lift to cu130.
        ((12, 8), (12, 0), detect.CU130),  # old CUDA 12.x driver, known Blackwell card
        ((0, 0), (12, 0), detect.CU130),  # driver CUDA unreadable but the card is known Blackwell
        ((10, 0), (10, 0), detect.CU130),  # sm_100 datacenter Blackwell, same floor
        ((13, 0), (12, 0), detect.CU130),  # driver already covers Blackwell -> floor is a no-op
        ((13, 2), (12, 0), detect.CU132),  # newest driver-supported build still wins for Blackwell
        # Ceiling -- pre-Turing (Maxwell/Pascal/Volta) was dropped from the CUDA 13 wheels, so hold cu126
        # even when the driver is new enough for cu130/cu132 (cu126 still runs on the newer driver).
        ((13, 2), (6, 1), detect.CU126),  # Pascal GTX 10-series on a CUDA 13.2 driver
        ((13, 0), (5, 2), detect.CU126),  # Maxwell on a CUDA 13.0 driver
        ((13, 0), (7, 0), detect.CU126),  # Volta sm_70 is pre-Turing and dropped too
        # In-window cards (Turing..Hopper) take the highest build the driver supports, unchanged.
        ((13, 2), (7, 5), detect.CU132),  # Turing sm_75 is the first arch the CUDA 13 wheels keep
        ((12, 8), (9, 0), detect.CU126),  # Hopper on a CUDA 12.x driver
        ((12, 8), (8, 6), detect.CU126),  # Ampere on a CUDA 12.x driver
        ((13, 2), (8, 6), detect.CU132),  # Ampere on a CUDA 13.2 driver
        # Arch unreadable -> keep the driver-only pick (prior behaviour).
        ((0, 0), (0, 0), detect.CU126),
        ((13, 2), (0, 0), detect.CU132),
    ],
)
def test_cuda_build_architecture_window(
    version: tuple[int, int],
    compute_cap: tuple[int, int],
    expected: str,
) -> None:
    """The driver-based build is clamped into the GPU's valid arch window (cu126 floor and ceiling)."""
    assert detect._cuda_build(version, compute_cap) == expected


@pytest.mark.parametrize(
    ("nvidia", "cuda_version", "compute_cap", "amd", "rocm", "expected"),
    [
        (True, (12, 8), (8, 6), False, False, detect.CU126),
        (True, (13, 0), (8, 6), False, False, detect.CU130),
        (True, (13, 2), (8, 6), False, False, detect.CU132),
        (True, (0, 0), (0, 0), False, False, detect.CU126),  # nvidia present but unreadable -> safe 12.6 build
        # Blackwell card on an old CUDA 12.x driver is floored to cu130 by the architecture check.
        (True, (12, 8), (12, 0), False, False, detect.CU130),
        (False, (0, 0), (0, 0), True, False, detect.AMD_UNSUPPORTED),
        (False, (0, 0), (0, 0), True, True, detect.ROCM),
        (False, (0, 0), (0, 0), False, False, detect.CPU),
    ],
)
def test_detect_backend(
    monkeypatch: pytest.MonkeyPatch,
    nvidia: bool,
    cuda_version: tuple[int, int],
    compute_cap: tuple[int, int],
    amd: bool,
    rocm: bool,
    expected: str,
) -> None:
    """detect_backend maps hardware presence + CUDA version + GPU arch onto the right build token."""
    monkeypatch.setattr(detect, "_nvidia_present", lambda: nvidia)
    monkeypatch.setattr(detect, "_nvidia_cuda_version", lambda: cuda_version)
    monkeypatch.setattr(detect, "_nvidia_compute_cap", lambda: compute_cap)
    monkeypatch.setattr(detect, "_amd_present", lambda: amd)
    monkeypatch.setattr(detect, "_rocm_runtime_present", lambda: rocm)
    monkeypatch.setattr(detect, "_windows_amd_rocm_backend", lambda: None)
    assert detect.detect_backend() == expected


@pytest.mark.parametrize(
    ("adapter_name", "expected"),
    [
        ("AMD Radeon RX 7900 XTX", detect.ROCM_WINDOWS),
        ("AMD Radeon RX 7700 XT", detect.ROCM_WINDOWS),
        ("AMD Radeon PRO W7900", detect.ROCM_WINDOWS),
        ("AMD Radeon RX 9070 XT", detect.ROCM_WINDOWS),
        ("AMD Radeon AI PRO R9700", detect.ROCM_WINDOWS),
        ("AMD Radeon 8060S Graphics", detect.ROCM_WINDOWS),
        ("AMD Ryzen AI 9 HX 370 w/ Radeon Graphics", detect.ROCM_WINDOWS),
        ("AMD Ryzen AI Max+ 395 w/ Radeon Graphics", detect.ROCM_WINDOWS),
    ],
)
def test_windows_amd_rocm_backend_supported_profiles(
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
    expected: str,
) -> None:
    """Supported AMD Windows adapter families map to the ROCm Windows profile."""
    monkeypatch.setattr(detect, "_is_windows", lambda: True)
    monkeypatch.setattr(detect, "_windows_display_adapters", lambda: [adapter_name])
    assert detect._windows_amd_rocm_backend() == expected


def test_windows_amd_rocm_backend_unknown_card_is_not_guessed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown AMD Windows cards remain unsupported instead of getting an unsafe ROCm wheel family."""
    monkeypatch.setattr(detect, "_is_windows", lambda: True)
    monkeypatch.setattr(detect, "_windows_display_adapters", lambda: ["AMD Radeon RX 6800 XT"])
    assert detect._windows_amd_rocm_backend() is None


def test_detect_backend_uses_windows_amd_rocm_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """An AMD Windows profile returned by the matcher wins over the generic unsupported path."""
    monkeypatch.setattr(detect, "_nvidia_present", lambda: False)
    monkeypatch.setattr(detect, "_amd_present", lambda: True)
    monkeypatch.setattr(detect, "_windows_amd_rocm_backend", lambda: detect.ROCM_WINDOWS)
    monkeypatch.setattr(detect, "_rocm_runtime_present", lambda: False)
    assert detect.detect_backend() == detect.ROCM_WINDOWS
