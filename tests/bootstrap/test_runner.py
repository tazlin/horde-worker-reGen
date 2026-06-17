"""Unit tests for the uv subprocess wrappers (isolation env + faithful exit codes)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from worker_bootstrap import paths, runner


def test_build_child_env_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The child env neutralizes user site/PYTHONPATH/conda and points uv at the on-drive locations."""
    monkeypatch.setenv("PYTHONPATH", "C:/junk")
    monkeypatch.setenv("CONDA_SHLVL", "2")
    monkeypatch.setenv("VIRTUAL_ENV", "C:/some/script-env")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("UV_PYTHON_INSTALL_DIR", raising=False)

    env = runner.build_child_env(tmp_path)

    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env
    assert "CONDA_SHLVL" not in env
    assert "VIRTUAL_ENV" not in env  # inherited from the outer `uv run --script`; must not leak to inner uv
    assert env["UV_CACHE_DIR"] == str(paths.uv_cache_dir(tmp_path))
    assert env["UV_PYTHON_INSTALL_DIR"] == str(paths.python_install_dir(tmp_path))
    assert env["UV_PYTHON_PREFERENCE"] == "only-managed"


def test_build_child_env_respects_preset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A caller-set UV_CACHE_DIR is preserved (power users can redirect the cache)."""
    monkeypatch.setenv("UV_CACHE_DIR", "C:/shared/uvcache")
    assert runner.build_child_env(tmp_path)["UV_CACHE_DIR"] == "C:/shared/uvcache"


def test_build_child_env_prepends_bundled_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A provisioned bundled git is put first on PATH so hordelib's child `git` clone resolves to it."""
    monkeypatch.setenv("PATH", "/existing/path")
    assert str(paths.git_cmd_dir(tmp_path)) not in runner.build_child_env(tmp_path)["PATH"]

    paths.git_cmd_dir(tmp_path).mkdir(parents=True)
    path = runner.build_child_env(tmp_path)["PATH"]
    assert path.split(os.pathsep)[0] == str(paths.git_cmd_dir(tmp_path))
    assert "/existing/path" in path


class _FakeCompleted:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def test_uv_sync_argv_and_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """uv_sync builds `sync --locked --extra <build>` and returns the child's real exit code."""
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], cwd: str, env: dict[str, str], check: bool) -> _FakeCompleted:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return _FakeCompleted(7)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc = runner.uv_sync("UV", "cu126", root=tmp_path)

    assert rc == 7  # a failed sync must propagate, never be masked as success
    assert captured["cmd"] == ["UV", "sync", "--locked", "--extra", "cu126"]
    assert captured["cwd"] == str(tmp_path)


def test_uv_sync_appends_feature_extras(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each feature extra is passed as its own --extra flag after the build extra."""
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], cwd: str, env: dict[str, str], check: bool) -> _FakeCompleted:
        captured["cmd"] = cmd
        return _FakeCompleted(0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.uv_sync("UV", "cu132", extras=("controlnet", "post-processing"), root=tmp_path)

    assert captured["cmd"] == [
        "UV",
        "sync",
        "--locked",
        "--extra",
        "cu132",
        "--extra",
        "controlnet",
        "--extra",
        "post-processing",
    ]


def test_uv_run_no_sync_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """uv_run builds `run --no-sync <command...>` with passthrough args intact."""
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], cwd: str, env: dict[str, str], check: bool) -> _FakeCompleted:
        captured["cmd"] = cmd
        return _FakeCompleted(0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc = runner.uv_run("UV", ["horde-worker-web", "--host", "0.0.0.0"], root=tmp_path)

    assert rc == 0
    assert captured["cmd"] == ["UV", "run", "--no-sync", "horde-worker-web", "--host", "0.0.0.0"]
