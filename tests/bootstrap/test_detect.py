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
    ("nvidia", "cuda_version", "amd", "rocm", "expected"),
    [
        (True, (12, 8), False, False, detect.CU126),
        (True, (13, 0), False, False, detect.CU130),
        (True, (13, 2), False, False, detect.CU132),
        (True, (0, 0), False, False, detect.CU126),  # nvidia present but version unreadable -> safe 12.6 build
        (False, (0, 0), True, False, detect.AMD_UNSUPPORTED),
        (False, (0, 0), True, True, detect.ROCM),
        (False, (0, 0), False, False, detect.CPU),
    ],
)
def test_detect_backend(
    monkeypatch: pytest.MonkeyPatch,
    nvidia: bool,
    cuda_version: tuple[int, int],
    amd: bool,
    rocm: bool,
    expected: str,
) -> None:
    """detect_backend maps hardware presence + CUDA version onto the right build token."""
    monkeypatch.setattr(detect, "_nvidia_present", lambda: nvidia)
    monkeypatch.setattr(detect, "_nvidia_cuda_version", lambda: cuda_version)
    monkeypatch.setattr(detect, "_amd_present", lambda: amd)
    monkeypatch.setattr(detect, "_rocm_runtime_present", lambda: rocm)
    assert detect.detect_backend() == expected
