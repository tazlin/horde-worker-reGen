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


class _FakePopen:
    """Stand-in for ``subprocess.Popen`` that records argv and returns a canned exit code from wait()."""

    def __init__(self, returncode: int, *, wait_side_effects: list[BaseException] | None = None) -> None:
        self._returncode = returncode
        # Exceptions to raise from successive wait() calls before finally returning the code, used to
        # simulate KeyboardInterrupt arriving while the parent is blocked waiting on the child.
        self._wait_side_effects = list(wait_side_effects or [])
        self.wait_calls = 0

    def wait(self) -> int:
        self.wait_calls += 1
        if self._wait_side_effects:
            raise self._wait_side_effects.pop(0)
        return self._returncode


def test_uv_sync_argv_and_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """uv_sync builds `sync --locked --extra <build>` and returns the child's real exit code."""
    captured: dict[str, object] = {}

    def fake_popen(cmd: list[str], cwd: str, env: dict[str, str]) -> _FakePopen:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return _FakePopen(7)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    rc = runner.uv_sync("UV", "cu126", root=tmp_path)

    assert rc == 7  # a failed sync must propagate, never be masked as success
    assert captured["cmd"] == ["UV", "sync", "--locked", "--extra", "cu126"]
    assert captured["cwd"] == str(tmp_path)


def test_uv_sync_appends_feature_extras(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each feature extra is passed as its own --extra flag after the build extra."""
    captured: dict[str, object] = {}

    def fake_popen(cmd: list[str], cwd: str, env: dict[str, str]) -> _FakePopen:
        captured["cmd"] = cmd
        return _FakePopen(0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
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

    def fake_popen(cmd: list[str], cwd: str, env: dict[str, str]) -> _FakePopen:
        captured["cmd"] = cmd
        return _FakePopen(0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    rc = runner.uv_run("UV", ["horde-worker-web", "--host", "0.0.0.0"], root=tmp_path)

    assert rc == 0
    assert captured["cmd"] == ["UV", "run", "--no-sync", "horde-worker-web", "--host", "0.0.0.0"]


def test_run_uv_keeps_waiting_through_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ctrl+C must not abandon the child: run_uv swallows KeyboardInterrupt and waits for the real exit.

    Ctrl+C reaches the child through the shared process group, which runs its own graceful shutdown.
    The launcher must keep waiting (here two interrupts) rather than unwind with a traceback and orphan
    the still-draining worker.
    """
    fake = _FakePopen(0, wait_side_effects=[KeyboardInterrupt(), KeyboardInterrupt()])

    def fake_popen(cmd: list[str], cwd: str, env: dict[str, str]) -> _FakePopen:
        return fake

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    rc = runner.uv_run("UV", ["python", "-s", "run_worker.py"], root=tmp_path)

    assert rc == 0  # the child's real exit code, returned after the interrupts were absorbed
    assert fake.wait_calls == 3  # two interrupted waits, then the one that returned
