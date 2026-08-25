"""Regression coverage for bootstrap-time uv version reconciliation."""

from __future__ import annotations

import hashlib
import http.client
import io
import os
import zipfile
from pathlib import Path

import pytest

from worker_bootstrap import uvbin


def _write_project(root: Path, required: str = "0.12.1") -> None:
    """Write the smallest project carrying an exact uv pin."""
    (root / "pyproject.toml").write_text(
        f'[tool.uv]\nrequired-version = "{required}"\n',
        encoding="utf-8",
    )


def _bundled_path(root: Path) -> Path:
    """Return the platform-specific private uv path."""
    return root / "bin" / ("uv.exe" if os.name == "nt" else "uv")


def test_required_uv_version_reads_exact_pin(tmp_path: Path) -> None:
    """The repair target comes from the overlaid project, not a stale launcher constant."""
    _write_project(tmp_path, "==0.12.1")
    assert uvbin.required_uv_version(tmp_path) == "0.12.1"


def test_required_uv_version_rejects_non_exact_target(tmp_path: Path) -> None:
    """A range cannot silently choose an arbitrary self-update target."""
    _write_project(tmp_path, ">=0.12.1")
    with pytest.raises(uvbin.UvCompatibilityError, match="exact version"):
        uvbin.required_uv_version(tmp_path)


def test_compatible_versioned_sidecar_is_reused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A repaired sidecar survives the preserved bin directory and avoids another download."""
    _write_project(tmp_path)
    bundled = _bundled_path(tmp_path)
    bundled.parent.mkdir()
    bundled.write_bytes(b"old")
    sidecar = bundled.parent / ("uv-0.12.1.exe" if os.name == "nt" else "uv-0.12.1")
    sidecar.write_bytes(b"new")
    monkeypatch.setattr(
        uvbin,
        "_reported_version",
        lambda executable, **kw: "0.12.1" if Path(executable) == sidecar else "0.11.21",
    )

    assert uvbin.ensure_compatible_uv(tmp_path) == str(sidecar)


def _uv_zip(payload: bytes = b"downloaded-uv") -> bytes:
    """Return a minimal uv release ZIP for the host executable name."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        executable_name = "uv.exe" if os.name == "nt" else "uv"
        archive.writestr(f"uv-test/{executable_name}", payload)
    return buffer.getvalue()


def test_stale_bundled_uv_is_repaired_through_verified_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stale private uv downloads a verified sidecar without invoking that uv's self-updater."""
    _write_project(tmp_path)
    bundled = _bundled_path(tmp_path)
    bundled.parent.mkdir()
    bundled.write_bytes(b"old-private-uv")
    archive_payload = _uv_zip()
    archive_name = "uv-x86_64-pc-windows-msvc.zip"
    checksum_payload = f"{hashlib.sha256(archive_payload).hexdigest()}  {archive_name}\n".encode()
    downloads: list[str] = []
    command_directories: list[Path] = []

    def fake_reported_version(executable: str, *, cwd: Path) -> str | None:
        command_directories.append(cwd)
        path = Path(executable)
        if path.read_bytes() == b"downloaded-uv":
            return "0.12.1"
        return "0.11.21" if path.exists() else None

    def fake_download(url: str) -> bytes:
        downloads.append(url)
        return checksum_payload if url.endswith(".sha256") else archive_payload

    monkeypatch.setattr(uvbin, "_release_archive_name", lambda: archive_name)
    monkeypatch.setattr(uvbin, "_download", fake_download)
    monkeypatch.setattr(uvbin, "_reported_version", fake_reported_version)

    repaired = Path(uvbin.ensure_compatible_uv(tmp_path))

    assert downloads == [
        f"https://github.com/astral-sh/uv/releases/download/0.12.1/{archive_name}.sha256",
        f"https://github.com/astral-sh/uv/releases/download/0.12.1/{archive_name}",
    ]
    assert command_directories
    assert all(not directory.is_relative_to(tmp_path) for directory in command_directories)
    assert repaired.name == ("uv-0.12.1.exe" if os.name == "nt" else "uv-0.12.1")
    assert repaired.read_bytes() == b"downloaded-uv"
    assert bundled.read_bytes() == b"old-private-uv"


def test_incompatible_path_uv_gets_private_sidecar_without_modification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A git checkout gets a compatible private sidecar without modifying developer-owned PATH tooling."""
    _write_project(tmp_path)
    path_uv = tmp_path / "developer-uv"
    path_uv.write_bytes(b"developer-owned")
    sidecar = _bundled_path(tmp_path).parent / ("uv-0.12.1.exe" if os.name == "nt" else "uv-0.12.1")

    def fake_reported_version(executable: str, **kw: object) -> str:
        return "0.12.1" if Path(executable) == sidecar else "0.11.21"

    def fake_download(version: str, target: Path, **kw: object) -> None:
        assert version == "0.12.1"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"private-sidecar")

    monkeypatch.setattr(uvbin, "_reported_version", fake_reported_version)
    monkeypatch.setattr(uvbin, "_download_verified_uv", fake_download)

    assert uvbin.ensure_compatible_uv(tmp_path, str(path_uv)) == str(sidecar)
    assert path_uv.read_bytes() == b"developer-owned"
    assert sidecar.read_bytes() == b"private-sidecar"


def test_checksum_mismatch_never_publishes_sidecar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A corrupt or intercepted archive cannot replace the last sidecar."""
    target = _bundled_path(tmp_path).parent / ("uv-0.12.1.exe" if os.name == "nt" else "uv-0.12.1")
    target.parent.mkdir()
    target.write_bytes(b"last-runnable-sidecar")
    archive_name = "uv-x86_64-pc-windows-msvc.zip"
    monkeypatch.setattr(uvbin, "_release_archive_name", lambda: archive_name)
    monkeypatch.setattr(
        uvbin,
        "_download",
        lambda url: (b"0" * 64 + b"  " + archive_name.encode()) if url.endswith(".sha256") else b"corrupt",
    )

    with pytest.raises(uvbin.UvCompatibilityError, match="failed SHA-256 verification"):
        uvbin._download_verified_uv("0.12.1", target, probe_directory=tmp_path)
    assert target.read_bytes() == b"last-runnable-sidecar"


def test_interrupted_http_body_becomes_actionable_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A truncated HTTP response cannot escape bootstrap as a raw traceback."""
    monkeypatch.setattr(
        uvbin,
        "_download",
        lambda url: (_ for _ in ()).throw(http.client.IncompleteRead(b"partial", 100)),
    )

    with pytest.raises(uvbin.UvCompatibilityError, match="Could not download uv 0.12.1"):
        uvbin._download_verified_uv("0.12.1", tmp_path / "uv.exe", probe_directory=tmp_path)


def test_publish_filesystem_failure_becomes_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A locked or unwritable sidecar path is reported without a raw filesystem traceback."""
    archive_payload = _uv_zip()
    archive_name = "uv-x86_64-pc-windows-msvc.zip"
    checksum_payload = f"{hashlib.sha256(archive_payload).hexdigest()}  {archive_name}\n".encode()
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")
    monkeypatch.setattr(uvbin, "_release_archive_name", lambda: archive_name)
    monkeypatch.setattr(
        uvbin,
        "_download",
        lambda url: checksum_payload if url.endswith(".sha256") else archive_payload,
    )

    with pytest.raises(uvbin.UvCompatibilityError, match="Could not publish the worker's private uv"):
        uvbin._download_verified_uv("0.12.1", blocked_parent / "uv.exe", probe_directory=tmp_path)
