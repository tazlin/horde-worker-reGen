"""Regression coverage for bootstrap-time uv version reconciliation."""

from __future__ import annotations

import os
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


def test_stale_bundled_uv_is_repaired_through_verified_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stale private uv updates a sidecar, never the binary hosting the current bootstrap."""
    _write_project(tmp_path)
    bundled = _bundled_path(tmp_path)
    bundled.parent.mkdir()
    bundled.write_bytes(b"old-private-uv")
    updated_copies: set[Path] = set()
    commands: list[list[str]] = []

    def fake_reported_version(executable: str, **kw: object) -> str | None:
        path = Path(executable)
        if path in updated_copies:
            return "0.12.1"
        return "0.11.21" if path.exists() else None

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command: list[str], **kw: object) -> _Completed:
        commands.append(command)
        candidate = Path(command[0])
        assert candidate != bundled
        assert candidate.read_bytes() == b"old-private-uv"
        assert command[1:] == ["self", "update", "0.12.1"]
        updated_copies.add(candidate)
        return _Completed()

    monkeypatch.setattr(uvbin, "_reported_version", fake_reported_version)
    monkeypatch.setattr(uvbin.subprocess, "run", fake_run)

    repaired = Path(uvbin.ensure_compatible_uv(tmp_path))

    assert commands
    assert repaired.name == ("uv-0.12.1.exe" if os.name == "nt" else "uv-0.12.1")
    assert repaired.read_bytes() == b"old-private-uv"
    assert bundled.read_bytes() == b"old-private-uv"


def test_incompatible_path_uv_is_not_modified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Developer-owned PATH tooling remains outside the managed install's repair boundary."""
    _write_project(tmp_path)
    path_uv = tmp_path / "developer-uv"
    path_uv.write_bytes(b"developer-owned")
    monkeypatch.setattr(uvbin, "_reported_version", lambda executable, **kw: "0.11.21")

    with pytest.raises(uvbin.UvCompatibilityError, match="PATH-provided"):
        uvbin.ensure_compatible_uv(tmp_path, str(path_uv))
    assert path_uv.read_bytes() == b"developer-owned"
