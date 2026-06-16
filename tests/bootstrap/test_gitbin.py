"""Unit tests for git resolution (prefer system git; Windows MinGit fallback; POSIX guidance)."""

from __future__ import annotations

from pathlib import Path

import pytest

from worker_bootstrap import gitbin


class _Probe:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_find_system_git_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """A git on PATH that answers --version is returned."""
    monkeypatch.setattr(gitbin.shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(gitbin.subprocess, "run", lambda *a, **k: _Probe(0, "git version 2.47.1"))
    assert gitbin.find_system_git() == "/usr/bin/git"


def test_find_system_git_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """No git on PATH yields None (no probe attempted)."""
    monkeypatch.setattr(gitbin.shutil, "which", lambda _: None)
    assert gitbin.find_system_git() is None


def test_find_system_git_broken_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolvable shim that cannot actually run is treated as absent."""
    monkeypatch.setattr(gitbin.shutil, "which", lambda _: "/usr/bin/git")

    def _raise(*_a: object, **_k: object) -> _Probe:
        raise OSError("cannot exec")

    monkeypatch.setattr(gitbin.subprocess, "run", _raise)
    assert gitbin.find_system_git() is None


def test_notice_line_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """The notice line reflects which path will be taken for git."""
    assert "already on your PATH" in gitbin.notice_line("/usr/bin/git")

    monkeypatch.setattr(gitbin, "_can_provision", lambda: True)
    windows_line = gitbin.notice_line(None)
    assert "MinGit" in windows_line and "download" in windows_line.lower()

    monkeypatch.setattr(gitbin, "_can_provision", lambda: False)
    posix_line = gitbin.notice_line(None)
    assert "install" in posix_line.lower() and "git" in posix_line.lower()


def test_ensure_git_uses_system_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When system git exists it is used as-is: no download, source 'system'."""
    monkeypatch.setattr(gitbin, "find_system_git", lambda: "/usr/bin/git")
    called = {"provisioned": False}
    monkeypatch.setattr(gitbin, "provision_mingit", lambda root=None: called.__setitem__("provisioned", True))

    resolution = gitbin.ensure_git(tmp_path)

    assert resolution.ok
    assert resolution.source == "system"
    assert called["provisioned"] is False


def test_ensure_git_missing_on_posix_is_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No git and no provisioning (POSIX) fails with install guidance, not a download."""
    monkeypatch.setattr(gitbin, "find_system_git", lambda: None)
    monkeypatch.setattr(gitbin, "_can_provision", lambda: False)

    resolution = gitbin.ensure_git(tmp_path)

    assert not resolution.ok
    assert resolution.source == "missing"
    assert "install git" in resolution.message.lower()


def test_ensure_git_reuses_existing_mingit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An already-unpacked bundled git is reused without downloading again."""
    monkeypatch.setattr(gitbin, "find_system_git", lambda: None)
    monkeypatch.setattr(gitbin, "_can_provision", lambda: True)
    git_exe = gitbin.mingit_git_exe(tmp_path)
    git_exe.parent.mkdir(parents=True)
    git_exe.write_text("", encoding="utf-8")

    def _fail(root: Path | None = None) -> Path:
        raise AssertionError("should not download when a bundled git already exists")

    monkeypatch.setattr(gitbin, "provision_mingit", _fail)
    resolution = gitbin.ensure_git(tmp_path)

    assert resolution.ok
    assert resolution.source == "mingit"


def test_ensure_git_downloads_mingit_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No git on a provisionable platform triggers a MinGit download."""
    monkeypatch.setattr(gitbin, "find_system_git", lambda: None)
    monkeypatch.setattr(gitbin, "_can_provision", lambda: True)

    def _provision(root: Path | None = None) -> Path:
        exe = gitbin.mingit_git_exe(root)
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("", encoding="utf-8")
        return exe

    monkeypatch.setattr(gitbin, "provision_mingit", _provision)
    resolution = gitbin.ensure_git(tmp_path)

    assert resolution.ok
    assert resolution.source == "mingit"


def test_ensure_git_download_failure_is_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failed MinGit download surfaces a clear error rather than crashing."""
    monkeypatch.setattr(gitbin, "find_system_git", lambda: None)
    monkeypatch.setattr(gitbin, "_can_provision", lambda: True)

    def _boom(root: Path | None = None) -> Path:
        raise OSError("network down")

    monkeypatch.setattr(gitbin, "provision_mingit", _boom)
    resolution = gitbin.ensure_git(tmp_path)

    assert not resolution.ok
    assert "could not provision" in resolution.message.lower()


def test_mingit_url_pins_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MinGit URL embeds the pinned version, overridable via env."""
    monkeypatch.setenv("HORDE_WORKER_MINGIT_VERSION", "9.9.9")
    url = gitbin.mingit_url()
    assert "v9.9.9.windows.1/MinGit-9.9.9-64-bit.zip" in url
