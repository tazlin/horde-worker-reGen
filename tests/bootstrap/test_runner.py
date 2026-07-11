"""Unit tests for the uv subprocess wrappers (isolation env + faithful exit codes)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from worker_bootstrap import paths, runner, utilities_env


def test_build_child_env_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The child env neutralizes user site/PYTHONPATH/conda and points uv at the on-drive locations."""
    monkeypatch.setenv("PYTHONPATH", "C:/junk")
    monkeypatch.setenv("CONDA_SHLVL", "2")
    monkeypatch.setenv("VIRTUAL_ENV", "C:/some/script-env")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("UV_PYTHON_INSTALL_DIR", raising=False)
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)
    monkeypatch.delenv("HORDE_WORKER_DATA_DIR", raising=False)

    env = runner.build_child_env(tmp_path)

    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env
    assert "CONDA_SHLVL" not in env
    assert "VIRTUAL_ENV" not in env  # inherited from the outer `uv run --script`; must not leak to inner uv
    assert env["UV_CACHE_DIR"] == str(paths.uv_cache_dir(tmp_path))
    assert env["UV_PYTHON_INSTALL_DIR"] == str(paths.python_install_dir(tmp_path))
    # Only the data-dir LOCATION is propagated; AIWORKER_CACHE_HOME is left for the worker to derive at the
    # lowest precedence so it never outranks bridgeData.yaml `cache_home`.
    assert env["HORDE_WORKER_DATA_DIR"] == str(paths.data_root(tmp_path))
    assert "AIWORKER_CACHE_HOME" not in env
    assert env["UV_PYTHON_PREFERENCE"] == "only-managed"


def test_build_child_env_respects_preset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Caller-set cache/model/data locations are preserved (power users can redirect them)."""
    monkeypatch.setenv("UV_CACHE_DIR", "C:/shared/uvcache")
    monkeypatch.setenv("AIWORKER_CACHE_HOME", "C:/shared/models")
    monkeypatch.setenv("HORDE_WORKER_DATA_DIR", "C:/shared/horde-data")
    env = runner.build_child_env(tmp_path)
    assert env["UV_CACHE_DIR"] == "C:/shared/uvcache"
    # A user/system-set AIWORKER_CACHE_HOME is never dropped or overridden by the launcher.
    assert env["AIWORKER_CACHE_HOME"] == "C:/shared/models"
    assert env["HORDE_WORKER_DATA_DIR"] == "C:/shared/horde-data"


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


def test_build_child_env_shared_cache_mode_omits_uv_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Shared cache mode leaves UV_CACHE_DIR unset so uv uses its own (system) default cache."""
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.setenv("HORDE_WORKER_UV_CACHE_MODE", "shared")
    env = runner.build_child_env(tmp_path)
    assert "UV_CACHE_DIR" not in env
    # The data dir and managed Python still live in the peered location regardless of cache mode.
    assert env["UV_PYTHON_INSTALL_DIR"] == str(paths.python_install_dir(tmp_path))


class _FakeCompleted:
    """Stand-in for ``subprocess.run`` result with canned returncode/stdout/stderr."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_uv_sync_dry_run_argv_and_capture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """uv_sync_dry_run adds --dry-run, keeps --locked, and returns the captured output."""
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kw: object) -> _FakeCompleted:
        captured["cmd"] = cmd
        captured["capture_output"] = kw.get("capture_output")
        return _FakeCompleted(0, stdout="+ torch==2.12.1+cu132\n")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc, out = runner.uv_sync_dry_run("UV", "cu132", extras=("controlnet",), root=tmp_path)

    assert rc == 0
    assert "torch==2.12.1+cu132" in out
    assert captured["capture_output"] is True
    assert captured["cmd"] == ["UV", "sync", "--dry-run", "--locked", "--extra", "cu132", "--extra", "controlnet"]


def test_uv_sync_held_drops_locked_and_adds_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The opt-out path drops --locked (only here) and passes --override to hold packages."""
    captured: dict[str, object] = {}

    def fake_popen(cmd: list[str], cwd: str, env: dict[str, str]) -> _FakePopen:
        captured["cmd"] = cmd
        return _FakePopen(0)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    overrides = tmp_path / "sync-overrides.txt"
    rc = runner.uv_sync_held("UV", "cu132", overrides_path=overrides, extras=("post-processing",), root=tmp_path)

    assert rc == 0
    cmd = captured["cmd"]
    assert "--locked" not in cmd  # the load-bearing difference from the normal path
    assert cmd == ["UV", "sync", "--extra", "cu132", "--extra", "post-processing", "--override", str(overrides)]


def test_uv_sync_held_dry_run_returns_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The held dry-run (feasibility probe) captures output and returns uv's exit code."""

    def fake_run(cmd: list[str], **kw: object) -> _FakeCompleted:
        assert "--dry-run" in cmd
        assert "--locked" not in cmd
        return _FakeCompleted(1)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc = runner.uv_sync_held("UV", "cu132", overrides_path=tmp_path / "o.txt", root=tmp_path, dry_run=True)
    assert rc == 1  # non-zero => the hold is infeasible => the upgrade is mandatory


def test_uv_cache_prune_parses_reclaimed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """uv_cache_prune runs `cache prune` (never clean) and parses the reclaimed size."""
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kw: object) -> _FakeCompleted:
        captured["cmd"] = cmd
        captured["timeout"] = kw.get("timeout")
        return _FakeCompleted(0, stdout="Pruning cache at ...\nRemoved 1248 files (3.4 GiB)\n")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc, reclaimed = runner.uv_cache_prune("UV", root=tmp_path)

    assert rc == 0
    assert captured["cmd"] == ["UV", "cache", "prune"]
    assert reclaimed == int(3.4 * 1024**3)
    assert captured["timeout"] == runner._PRUNE_TIMEOUT_SECONDS  # bounded by default so it cannot hang forever


def test_uv_cache_prune_times_out_non_fatally(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A stalled prune is bounded by the timeout and reported as a skip, never a crash."""

    def fake_run(cmd: list[str], **kw: object) -> _FakeCompleted:
        raise runner.subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc, reclaimed = runner.uv_cache_prune("UV", root=tmp_path)

    assert rc == runner.PRUNE_TIMED_OUT
    assert reclaimed == 0


def test_uv_cache_prune_swallows_ctrl_c(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ctrl+C during the final cleanup skips the prune instead of unwinding with a traceback."""

    def fake_run(cmd: list[str], **kw: object) -> _FakeCompleted:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc, reclaimed = runner.uv_cache_prune("UV", root=tmp_path)

    assert rc == runner.PRUNE_INTERRUPTED
    assert reclaimed == 0


def test_prune_timeout_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """HORDE_WORKER_PRUNE_TIMEOUT overrides the bound; 0/garbage fall back safely."""
    monkeypatch.setenv("HORDE_WORKER_PRUNE_TIMEOUT", "45")
    assert runner._prune_timeout_seconds() == 45.0
    monkeypatch.setenv("HORDE_WORKER_PRUNE_TIMEOUT", "0")
    assert runner._prune_timeout_seconds() == 0.0  # disables the bound
    monkeypatch.setenv("HORDE_WORKER_PRUNE_TIMEOUT", "nonsense")
    assert runner._prune_timeout_seconds() == runner._PRUNE_TIMEOUT_SECONDS


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


def _write_utilities_requirements(root: Path, token: str) -> None:
    """Write a minimal utilities requirements pin for *token* so provisioning has something to install."""
    path = utilities_env.utilities_requirements_file(token=token, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("horde-image-utilities==1.0.0\n", encoding="utf-8")


def test_provision_utilities_runs_plan_then_stamps(tmp_path: Path) -> None:
    """A successful provision runs each planned command in order and only then writes the stamp."""
    _write_utilities_requirements(tmp_path, "cu126")
    ran: list[list[str]] = []

    def fake_runner(argv: list[str]) -> int:
        ran.append(argv)
        return 0

    runner.provision_utilities("UV", backend_token="cu126", root=tmp_path, command_runner=fake_runner)

    expected = utilities_env.plan_utilities_provision(uv="UV", backend_token="cu126", root=tmp_path)
    assert ran == expected
    stamp = utilities_env.read_utilities_stamp(tmp_path)
    assert stamp is not None and stamp.backend_token == "cu126"


def test_provision_utilities_raises_and_does_not_stamp_on_failure(tmp_path: Path) -> None:
    """A non-zero step aborts with an actionable error and leaves no stamp (so it retries next sync)."""
    _write_utilities_requirements(tmp_path, "cu126")

    def failing_runner(argv: list[str]) -> int:
        return 3

    with pytest.raises(utilities_env.UtilitiesProvisionError, match="exit code 3"):
        runner.provision_utilities("UV", backend_token="cu126", root=tmp_path, command_runner=failing_runner)

    assert utilities_env.read_utilities_stamp(tmp_path) is None


def test_provision_utilities_stops_at_first_failure(tmp_path: Path) -> None:
    """The install step is not attempted when the venv-create step fails."""
    _write_utilities_requirements(tmp_path, "cu126")
    ran: list[list[str]] = []

    def failing_first(argv: list[str]) -> int:
        ran.append(argv)
        return 1  # the create step fails

    with pytest.raises(utilities_env.UtilitiesProvisionError):
        runner.provision_utilities("UV", backend_token="cu126", root=tmp_path, command_runner=failing_first)

    assert len(ran) == 1  # stopped before the pip install step


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
