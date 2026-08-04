"""Tests for the explicit JetPack 5 / CUDA 11.4 legacy runtime profile."""

from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
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
    """The profile checks the pinned NVIDIA wheels and a local xFormers checksum sidecar."""
    wheel_hashes: dict[str, str] = {}
    for name in jetpack5.REQUIRED_WHEEL_NAMES:
        content = name.encode()
        (tmp_path / name).write_bytes(content)
        wheel_hashes[name] = hashlib.sha256(content).hexdigest()
    xformers = tmp_path / "xformers-0.0.23+e1b36f7.d20260803-cp310-cp310-linux_aarch64.whl"
    xformers.write_bytes(b"local xformers build")
    xformers.with_suffix(f"{xformers.suffix}.sha256").write_text(
        f"{hashlib.sha256(xformers.read_bytes()).hexdigest()}  {xformers.name}\n",
        encoding="ascii",
    )

    wheels = jetpack5.resolve_wheels(
        tmp_path,
        required_hashes=wheel_hashes,
        xformers_hash=hashlib.sha256(xformers.read_bytes()).hexdigest(),
    )

    assert wheels[:3] == tuple(tmp_path / name for name in jetpack5.REQUIRED_WHEEL_NAMES)
    assert wheels[3] == xformers


def test_resolve_wheels_rejects_missing_torch(tmp_path: Path) -> None:
    """A generic cu118 wheel is not accepted in place of NVIDIA's JetPack CUDA 11.4 build."""
    with pytest.raises(jetpack5.JetPack5Error, match="Required JetPack 5 wheel not found"):
        jetpack5.resolve_wheels(tmp_path)


def test_resolve_wheels_rejects_checksum_mismatch(tmp_path: Path) -> None:
    """Renaming an arbitrary file to a pinned NVIDIA wheel name does not bypass validation."""
    for name in jetpack5.REQUIRED_WHEEL_NAMES:
        (tmp_path / name).write_bytes(b"not the tested wheel")

    with pytest.raises(jetpack5.JetPack5Error, match="checksum mismatch"):
        jetpack5.resolve_wheels(tmp_path)


def test_resolve_wheels_rejects_xformers_wheel_and_sidecar_substitution(tmp_path: Path) -> None:
    """A replaced local xFormers wheel cannot authorize itself by replacing its sidecar too."""
    wheel_hashes: dict[str, str] = {}
    for name in jetpack5.REQUIRED_WHEEL_NAMES:
        content = name.encode()
        (tmp_path / name).write_bytes(content)
        wheel_hashes[name] = hashlib.sha256(content).hexdigest()
    xformers = tmp_path / "xformers-0.0.23+e1b36f7.d20260803-cp310-cp310-linux_aarch64.whl"
    xformers.write_bytes(b"substituted build")
    substituted = hashlib.sha256(xformers.read_bytes()).hexdigest()
    xformers.with_suffix(f"{xformers.suffix}.sha256").write_text(
        f"{substituted}  {xformers.name}\n",
        encoding="ascii",
    )

    with pytest.raises(jetpack5.JetPack5Error, match="checksum mismatch"):
        jetpack5.resolve_wheels(
            tmp_path,
            required_hashes=wheel_hashes,
            xformers_hash="0" * 64,
        )


def test_source_archive_follows_recorded_install_origin(tmp_path: Path) -> None:
    """A Tazlin or other fork install fetches the pinned historical tree through the same origin."""
    install_info = tmp_path / "bin" / "install-info"
    install_info.parent.mkdir(parents=True)
    install_info.write_text("method=one-line\nrepo=tazlin/horde-worker-reGen\n", encoding="utf-8")

    assert jetpack5._source_archive_url(tmp_path) == (  # noqa: SLF001 - focused profile unit test
        f"https://github.com/tazlin/horde-worker-reGen/archive/{jetpack5.LEGACY_COMMIT}.tar.gz"
    )


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    """The verified archive still cannot write outside its extraction directory."""
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        entry = tarfile.TarInfo("../escape")
        entry.size = 1
        bundle.addfile(entry, io.BytesIO(b"x"))

    with pytest.raises(jetpack5.JetPack5Error, match="unsafe path"):
        jetpack5._safe_extract(archive, tmp_path / "extract")  # noqa: SLF001 - safety regression test


def test_python_validation_pins_patch_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The compatibility runtime uses the exact tested Python 3.10.20 interpreter."""
    python = tmp_path / "python"
    python.touch()
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(jetpack5.subprocess, "run", run)

    assert jetpack5._validate_python310(python) == python  # noqa: SLF001 - version pin regression test
    assert "(3, 10, 20)" in commands[0][2]


def test_runtime_root_lives_in_preserved_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Updating the modern installer must not delete the separately pinned legacy runtime."""
    data = tmp_path / "worker-data"
    monkeypatch.setenv("HORDE_WORKER_DATA_DIR", str(data))
    assert jetpack5.runtime_root(tmp_path / "worker") == data / "jetpack5-runtime"
