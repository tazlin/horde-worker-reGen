"""Unit tests for ad-hoc ROCm dependency installation profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from worker_bootstrap import detect, rocm


def _write_pyproject(root: Path) -> None:
    """Write the minimal pyproject needed by the base cpu sync."""
    (root / "pyproject.toml").write_text(
        '[project.optional-dependencies]\ncpu = ["torch"]\n',
        encoding="utf-8",
    )


def test_linux_rocm_install_keeps_stable_overlay_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The generic Linux ROCm profile keeps the stable PyTorch ROCm index and triton sidecar."""
    _write_pyproject(tmp_path)
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(rocm.runner, "uv_sync", lambda uv, extra, **kw: calls.append(("sync", extra)) or 0)
    monkeypatch.setattr(rocm.runner, "run_uv", lambda uv, args, **kw: calls.append(("pip", args)) or 0)
    monkeypatch.setattr(rocm.runner, "uv_run", lambda uv, command, **kw: 0)
    monkeypatch.setattr(rocm, "_patch_wsl_libhsa", lambda root: None)

    assert rocm.sync_rocm("UV", root=tmp_path, token=detect.ROCM) == 0

    assert calls == [
        ("sync", "cpu"),
        (
            "pip",
            [
                "pip",
                "install",
                "--reinstall",
                "torch==2.9.1",
                "torchvision",
                "torchaudio",
                "pytorch-triton-rocm",
                "--extra-index-url",
                "https://download.pytorch.org/whl/rocm6.4",
            ],
        ),
        ("pip", ["pip", "uninstall", "pynvml", "nvidia-ml-py"]),
    ]


def test_windows_rocm_profile_uses_amd_official_wheels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The AMD Windows profile installs the official AMD ROCm SDK + PyTorch wheels."""
    _write_pyproject(tmp_path)
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(rocm.runner, "uv_sync", lambda uv, extra, **kw: calls.append(("sync", extra)) or 0)
    monkeypatch.setattr(rocm.runner, "run_uv", lambda uv, args, **kw: calls.append(("pip", args)) or 0)
    monkeypatch.setattr(rocm.runner, "uv_run", lambda uv, command, **kw: 0)
    monkeypatch.setattr(rocm, "_patch_wsl_libhsa", lambda root: None)

    assert rocm.sync_rocm("UV", root=tmp_path, token=detect.ROCM_WINDOWS) == 0

    assert calls == [
        ("sync", "cpu"),
        (
            "pip",
            [
                "pip",
                "install",
                "--reinstall",
                "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/amd_smi-7.2.1-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/hip_sdk-7.2.1-py3-none-win_amd64.whl",
                "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
            ],
        ),
        ("pip", ["pip", "uninstall", "pynvml", "nvidia-ml-py"]),
    ]
