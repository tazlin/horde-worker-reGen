"""Unit tests for CLI dispatch (no real uv/subprocess is invoked)."""

from __future__ import annotations

from pathlib import Path

import pytest

from worker_bootstrap import backend, cli


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, list[tuple[str, object]]]:
    """Isolate the install root in a tmp dir and record uv calls instead of running them."""
    monkeypatch.setattr(cli.paths, "install_root", lambda: tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project.optional-dependencies]\ncu126 = ["torch"]\ncu130 = ["torch"]\ncpu = ["torch"]\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("HORDE_WORKER_BACKEND", raising=False)
    monkeypatch.setattr(cli.detect, "detect_backend", lambda: "cu126")

    # Keep the install path hermetic: don't really prompt for consent or shell out to git.
    monkeypatch.setattr(cli.consent, "ensure_consent", lambda **kw: True)
    monkeypatch.setattr(cli.gitbin, "find_system_git", lambda: "git")
    monkeypatch.setattr(
        cli.gitbin,
        "ensure_git",
        lambda root=None, **kw: cli.gitbin.GitResolution("git", "system", "ok"),
    )

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(cli.runner, "uv_sync", lambda uv, extra, **kw: calls.append(("sync", extra)) or 0)
    monkeypatch.setattr(cli.runner, "uv_run", lambda uv, command, **kw: calls.append(("run", command)) or 0)
    # Pruning runs after every successful sync; fake it so tests never shell out and `calls` stays clean.
    monkeypatch.setattr(cli.runner, "uv_cache_prune", lambda uv, **kw: (0, 0))
    return tmp_path, calls


def _make_venv(root: Path) -> None:
    (root / ".venv").mkdir()


def test_detect_prints_token(env: tuple[Path, list], capsys: pytest.CaptureFixture[str]) -> None:
    """`detect` prints the resolved token and exits 0."""
    assert cli.main(["detect"]) == 0
    assert capsys.readouterr().out.strip() == "cu126"


def test_detect_write_persists_backend(env: tuple[Path, list]) -> None:
    """`detect --write` persists the token to bin/backend."""
    root, _ = env
    assert cli.main(["detect", "--write"]) == 0
    assert backend.read_backend_file(root / "bin" / "backend") == "cu126"


def test_sync_calls_uv_sync(env: tuple[Path, list]) -> None:
    """`sync` resolves the default build and runs uv sync."""
    _, calls = env
    assert cli.main(["sync"]) == 0
    assert calls == [("sync", "cu126")]


def test_sync_backend_flag_overrides(env: tuple[Path, list]) -> None:
    """`sync --backend cu130` forces that build."""
    _, calls = env
    assert cli.main(["sync", "--backend", "cu130"]) == 0
    assert calls == [("sync", "cu130")]


def test_sync_shortcut_flag(env: tuple[Path, list]) -> None:
    """The legacy `--cu130` shortcut maps to `--backend cu130` (back-compat with update-runtime.cmd)."""
    _, calls = env
    assert cli.main(["sync", "--cu130"]) == 0
    assert calls == [("sync", "cu130")]


def test_launch_web_passthrough(env: tuple[Path, list]) -> None:
    """`launch web` runs the web console script with passthrough args, no re-sync when .venv exists."""
    root, calls = env
    _make_venv(root)
    assert cli.main(["launch", "web", "--host", "0.0.0.0"]) == 0
    assert calls == [("run", ["horde-worker-web", "--host", "0.0.0.0"])]


def test_launch_bridge_downloads_then_runs(env: tuple[Path, list]) -> None:
    """`launch bridge` downloads models, then starts run_worker.py with passthrough args."""
    root, calls = env
    _make_venv(root)
    assert cli.main(["launch", "bridge", "--amd"]) == 0
    assert calls == [
        ("run", ["python", "-s", "download_models.py"]),
        ("run", ["python", "-s", "run_worker.py", "--amd"]),
    ]


def test_launch_syncs_when_no_venv(env: tuple[Path, list]) -> None:
    """A first launch with no .venv syncs before running."""
    _, calls = env
    assert cli.main(["launch", "terminal"]) == 0
    assert calls == [("sync", "cu126"), ("run", ["horde-worker"])]


def test_preload_runs_download(env: tuple[Path, list]) -> None:
    """`preload` runs the model downloader."""
    root, calls = env
    _make_venv(root)
    assert cli.main(["preload"]) == 0
    assert calls == [("run", ["python", "-s", "download_models.py"])]


def test_install_writes_backend_syncs_no_launch(env: tuple[Path, list]) -> None:
    """`install --no-launch` detects+persists the backend and syncs, but does not start the worker."""
    root, calls = env
    assert cli.main(["install", "--no-launch"]) == 0
    assert backend.read_backend_file(root / "bin" / "backend") == "cu126"
    assert calls == [("sync", "cu126")]


def test_install_aborts_when_consent_declined(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """Declining the consent gate aborts before any dependency sync."""
    _, calls = env
    monkeypatch.setattr(cli.consent, "ensure_consent", lambda **kw: False)
    assert cli.main(["install", "--no-launch"]) == 1
    assert calls == []


def test_sync_aborts_when_git_unavailable(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """An unresolvable git (e.g. POSIX with no git) aborts the sync with a non-zero code."""
    _, calls = env
    monkeypatch.setattr(
        cli.gitbin,
        "ensure_git",
        lambda root=None, **kw: cli.gitbin.GitResolution(None, "missing", "install git"),
    )
    assert cli.main(["sync"]) == 1
    assert calls == []


def test_run_passthrough(env: tuple[Path, list]) -> None:
    """`run <command>` runs an arbitrary command in the venv via uv run --no-sync."""
    _, calls = env
    assert cli.main(["run", "python", "-s", "download_models.py"]) == 0
    assert calls == [("run", ["python", "-s", "download_models.py"])]


def test_bare_command_falls_back_to_run(env: tuple[Path, list]) -> None:
    """A bare command (old runtime.cmd contract, e.g. the Dockerfiles README) is treated as `run ...`."""
    _, calls = env
    assert cli.main(["python", "-s", "-m", "convert_config_to_env"]) == 0
    assert calls == [("run", ["python", "-s", "-m", "convert_config_to_env"])]


def test_amd_unsupported_aborts(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """An AMD-on-Windows detection makes `detect` exit non-zero rather than silently choosing CPU."""
    monkeypatch.setattr(cli.detect, "detect_backend", lambda: "amd-unsupported")
    assert cli.main(["detect"]) == 2


def test_env_override_beats_detection(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """HORDE_WORKER_BACKEND=cpu lets an AMD user opt into the CPU build."""
    root, _ = env
    monkeypatch.setattr(cli.detect, "detect_backend", lambda: "amd-unsupported")
    monkeypatch.setenv("HORDE_WORKER_BACKEND", "cpu")
    assert cli.main(["detect", "--write"]) == 0
    assert backend.read_backend_file(root / "bin" / "backend") == "cpu"


# --- preview / hold / prune path (existing .venv, so the dry-run preview runs) ---------------------

_TORCH_BUMP_DRY_RUN = (
    " - torch==2.11.0+cu126\n + torch==2.12.1+cu126\n - torchvision==0.26.0\n + torchvision==0.27.1\n"
)


def _fake_preview(monkeypatch: pytest.MonkeyPatch, calls: list, *, dry_run_output: str, hold_feasible_rc: int) -> None:
    """Wire the dry-run preview: the normal dry-run returns *dry_run_output*; the held probe returns a rc."""
    monkeypatch.setattr(cli.runner, "uv_sync_dry_run", lambda uv, extra, **kw: (0, dry_run_output))

    def fake_held(uv: str, extra: str, *, dry_run: bool = False, **kw: object) -> int:
        if dry_run:
            return hold_feasible_rc
        calls.append(("held", extra))
        return 0

    monkeypatch.setattr(cli.runner, "uv_sync_held", fake_held)


def test_sync_preview_proceeds_full_when_no_torch_change(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a venv but no large upgrade in the dry-run, sync proceeds normally (locked)."""
    root, calls = env
    _make_venv(root)
    monkeypatch.setattr(cli.runner, "uv_sync_dry_run", lambda uv, extra, **kw: (0, " + somepkg==1.0.0\n"))
    assert cli.main(["sync"]) == 0
    assert calls == [("sync", "cu126")]


def test_sync_hold_takes_held_path_for_optional_upgrade(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--hold-torch on a resolvable (optional) torch bump runs the held sync, not the locked one."""
    root, calls = env
    _make_venv(root)
    _fake_preview(monkeypatch, calls, dry_run_output=_TORCH_BUMP_DRY_RUN, hold_feasible_rc=0)
    assert cli.main(["sync", "--hold-torch"]) == 0
    assert calls == [("held", "cu126")]


def test_sync_hold_refused_when_upgrade_mandatory(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """When the held dry-run cannot resolve, the upgrade is mandatory: the normal locked sync runs."""
    root, calls = env
    _make_venv(root)
    _fake_preview(monkeypatch, calls, dry_run_output=_TORCH_BUMP_DRY_RUN, hold_feasible_rc=1)
    assert cli.main(["sync", "--hold-torch"]) == 0
    assert calls == [("sync", "cu126")]  # forced full upgrade, never the held path


def test_sync_prunes_only_when_cache_owned(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-prune fires for the owned isolated cache but never in shared mode."""
    root, _ = env
    pruned: list[bool] = []
    monkeypatch.setattr(cli.runner, "uv_cache_prune", lambda uv, **kw: pruned.append(True) or (0, 0))

    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("HORDE_WORKER_UV_CACHE_MODE", raising=False)
    assert cli.main(["sync"]) == 0
    assert pruned == [True]  # owned (isolated default) => pruned

    pruned.clear()
    monkeypatch.setenv("HORDE_WORKER_UV_CACHE_MODE", "shared")
    assert cli.main(["sync"]) == 0
    assert pruned == []  # shared cache is never auto-pruned

    pruned.clear()
    monkeypatch.delenv("HORDE_WORKER_UV_CACHE_MODE", raising=False)
    assert cli.main(["sync", "--no-prune"]) == 0
    assert pruned == []  # explicitly disabled


def test_sync_prune_timeout_is_non_fatal(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stalled/timed-out prune still leaves the install reported as success, with an honest skip note."""
    root, _ = env
    monkeypatch.setattr(cli.runner, "uv_cache_prune", lambda uv, **kw: (cli.runner.PRUNE_TIMED_OUT, 0))
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("HORDE_WORKER_UV_CACHE_MODE", raising=False)

    assert cli.main(["sync"]) == 0  # install succeeded; cleanup failure must not change that
    err = capsys.readouterr().err
    assert "Cache cleanup timed out" in err
