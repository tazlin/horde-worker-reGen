"""Tests for the explicit JetPack 5 / CUDA 11.4 legacy runtime profile."""

from __future__ import annotations

from pathlib import Path

import pytest

from worker_bootstrap import jetpack5


def test_parse_l4t_major_release() -> None:
    """The NVIDIA release marker identifies JetPack 5 by its L4T R35 major."""
    assert jetpack5.parse_l4t_major_release("# R35 (release), REVISION: 6.4") == 35
    assert jetpack5.parse_l4t_major_release("# R36 (release), REVISION: 4.3") == 36
    assert jetpack5.parse_l4t_major_release("not an NVIDIA release marker") is None


def test_validate_host_accepts_jetpack5_aarch64() -> None:
    """Only Linux aarch64 running L4T R35 enters the CUDA 11.4 compatibility profile."""
    jetpack5.validate_host(system="Linux", machine="aarch64", l4t_release="# R35 (release), REVISION: 6.4")


@pytest.mark.parametrize(
    ("system", "machine", "release"),
    [
        ("Linux", "x86_64", "# R35 (release), REVISION: 6.4"),
        ("Linux", "aarch64", "# R36 (release), REVISION: 4.3"),
        ("Windows", "ARM64", "# R35 (release), REVISION: 6.4"),
    ],
)
def test_validate_host_rejects_non_jetpack5(system: str, machine: str, release: str) -> None:
    """An explicit legacy backend fails clearly instead of installing incompatible wheels elsewhere."""
    with pytest.raises(jetpack5.JetPack5Error, match="JetPack 5.*L4T R35.*aarch64"):
        jetpack5.validate_host(system=system, machine=machine, l4t_release=release)


def test_resolve_wheels_requires_the_tested_cuda114_set(tmp_path: Path) -> None:
    """The profile refuses an incomplete wheel set before creating or mutating its runtime."""
    for name in jetpack5.REQUIRED_WHEEL_NAMES:
        (tmp_path / name).touch()
    xformers = tmp_path / "xformers-0.0.23+e1b36f7.d20260803-cp310-cp310-linux_aarch64.whl"
    xformers.touch()

    wheels = jetpack5.resolve_wheels(tmp_path)

    assert wheels[:3] == tuple(tmp_path / name for name in jetpack5.REQUIRED_WHEEL_NAMES)
    assert wheels[3] == xformers


def test_resolve_wheels_rejects_missing_torch(tmp_path: Path) -> None:
    """A generic cu118 wheel is not accepted in place of NVIDIA's JetPack CUDA 11.4 build."""
    with pytest.raises(jetpack5.JetPack5Error, match="Required JetPack 5 wheel not found"):
        jetpack5.resolve_wheels(tmp_path)


def test_runtime_root_lives_in_preserved_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Updating the modern installer must not delete the separately pinned legacy runtime."""
    data = tmp_path / "worker-data"
    monkeypatch.setenv("HORDE_WORKER_DATA_DIR", str(data))
    assert jetpack5.runtime_root(tmp_path / "worker") == data / "jetpack5-runtime"
