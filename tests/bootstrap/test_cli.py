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
    # Keep launch tests hermetic: no launch-time update check reaches the network unless a test opts in.
    monkeypatch.setenv("HORDE_WORKER_AUTO_UPDATE", "off")

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


def test_launch_resyncs_when_lock_changed(env: tuple[Path, list]) -> None:
    """An in-place update overlays a new uv.lock but keeps the venv; launch must re-sync to match it."""
    root, calls = env
    _make_venv(root)
    (root / "uv.lock").write_text("lock-after-update", encoding="utf-8")
    # No stamp recorded for this lock -> the venv is stale -> re-sync before running, then record the lock.
    assert cli.main(["launch", "terminal"]) == 0
    assert calls == [("sync", "cu126"), ("run", ["horde-worker"])]
    assert (root / ".venv" / ".horde-sync-stamp").read_text(encoding="utf-8").strip() == cli._lock_fingerprint(root)


def test_launch_skips_sync_when_stamp_matches_lock(env: tuple[Path, list]) -> None:
    """An unchanged install (stamp matches the current lock) starts immediately, without re-syncing."""
    root, calls = env
    _make_venv(root)
    (root / "uv.lock").write_text("lock-current", encoding="utf-8")
    (root / ".venv" / ".horde-sync-stamp").write_text(cli._lock_fingerprint(root), encoding="utf-8")
    assert cli.main(["launch", "terminal"]) == 0
    assert calls == [("run", ["horde-worker"])]


def _available_info() -> object:
    """An UpdateInfo describing an available update (assets present)."""
    return cli.updater.UpdateInfo(
        current="1.0.0",
        latest="v2.0.0",
        available=True,
        bundle_url="https://example/horde-worker-reGen.zip",
        checksums_url="https://example/SHA256SUMS",
    )


def test_update_check_reports_available(
    env: tuple[Path, list],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`update --check` reports an available update without applying it."""
    monkeypatch.setattr(cli.updater, "check_for_update", lambda root: _available_info())
    assert cli.main(["update", "--check"]) == 0
    assert "Update available" in capsys.readouterr().out


def test_update_applies_then_resyncs(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """`update --yes` applies the update and then re-syncs so the new deps are installed."""
    _, calls = env
    monkeypatch.setattr(cli.updater, "check_for_update", lambda root: _available_info())
    applied: list[bool] = []
    monkeypatch.setattr(
        cli.updater,
        "perform_update",
        lambda root, info: (
            applied.append(True) or cli.updater.UpdateResult(True, "Updated to v2.0.0.", "1.0.0", "2.0.0")
        ),
    )
    assert cli.main(["update", "--yes"]) == 0
    assert applied == [True]
    assert calls == [("sync", "cu126")]  # reconcile after the overlay


def test_launch_auto_update_reconciles_deps_when_lock_changes(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch-time update that moves uv.lock must re-sync before running, not start on the old deps."""
    root, calls = env
    _make_venv(root)
    # Begin in sync: a lockfile and a matching stamp, so a plain launch would otherwise skip the sync.
    (root / "uv.lock").write_text("lock-old\n", encoding="utf-8")
    cli.paths.sync_stamp_file(root).write_text(cli.hashlib.sha256(b"lock-old\n").hexdigest(), encoding="utf-8")
    monkeypatch.setenv("HORDE_WORKER_AUTO_UPDATE", "auto")
    monkeypatch.setattr(cli.updater, "check_for_update", lambda r: _available_info())

    def fake_apply(r: Path, info: object) -> object:
        # Mimic a real overlay: a new lockfile lands and the sync stamp is invalidated.
        (root / "uv.lock").write_text("lock-new\n", encoding="utf-8")
        cli.updater._invalidate_sync_stamp(root)
        return cli.updater.UpdateResult(True, "Updated.", "1.0.0", "2.0.0")

    monkeypatch.setattr(cli.updater, "perform_update", fake_apply)
    assert cli.main(["launch", "terminal"]) == 0
    assert calls == [("sync", "cu126"), ("run", ["horde-worker"])]


def test_launch_auto_update_applies_before_sync(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """With policy=auto, launch applies an available update, then proceeds to sync/run."""
    root, calls = env
    _make_venv(root)
    monkeypatch.setenv("HORDE_WORKER_AUTO_UPDATE", "auto")
    monkeypatch.setattr(cli.updater, "check_for_update", lambda r: _available_info())
    applied: list[bool] = []
    monkeypatch.setattr(
        cli.updater,
        "perform_update",
        lambda r, info: applied.append(True) or cli.updater.UpdateResult(True, "Updated.", "1.0.0", "2.0.0"),
    )
    assert cli.main(["launch", "terminal"]) == 0
    assert applied == [True]
    assert calls == [("run", ["horde-worker"])]


def test_launch_update_off_never_checks(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """With policy=off (the fixture default), the launch path never calls the update check."""
    root, calls = env
    _make_venv(root)
    checked: list[bool] = []
    monkeypatch.setattr(cli.updater, "check_for_update", lambda r: checked.append(True) or _available_info())
    assert cli.main(["launch", "terminal"]) == 0
    assert checked == []


def test_update_refused_on_winget_install(
    env: tuple[Path, list],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`update` refuses to self-apply a winget-managed install (winget owns the version) and never checks."""
    _, calls = env
    monkeypatch.setattr(cli.updater, "resolve_install_method", lambda root=None: "winget")
    checked: list[bool] = []
    monkeypatch.setattr(cli.updater, "check_for_update", lambda root: checked.append(True) or _available_info())
    assert cli.main(["update", "--yes"]) == 1
    assert "winget upgrade" in capsys.readouterr().err
    assert checked == []  # gated before the network call
    assert calls == []  # nothing applied or synced


def test_launch_bows_out_of_self_update_on_dev_checkout(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a git checkout, launch never runs the self-updater (it would overlay the working tree)."""
    root, _ = env
    _make_venv(root)
    monkeypatch.setenv("HORDE_WORKER_AUTO_UPDATE", "auto")
    monkeypatch.setattr(cli.updater, "resolve_install_method", lambda root=None: "dev")
    checked: list[bool] = []
    monkeypatch.setattr(cli.updater, "check_for_update", lambda r: checked.append(True) or _available_info())
    assert cli.main(["launch", "terminal"]) == 0
    assert checked == []


def test_launch_skip_persists_and_suppresses_reoffer(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """Answering 'skip' records the version and a later launch does not re-offer it."""
    root, _ = env
    _make_venv(root)
    state = root / ".update-state.json"
    monkeypatch.setattr(cli.updater.paths, "update_state_file", lambda root=None: state)
    monkeypatch.setenv("HORDE_WORKER_AUTO_UPDATE", "prompt")
    # Force the check to always be due so this exercises skip suppression, not the time throttle.
    monkeypatch.setattr(cli.updater, "should_check_now", lambda root: True)
    monkeypatch.setattr(cli.consent, "is_interactive", lambda: True)
    monkeypatch.setattr(cli.updater, "check_for_update", lambda r: _available_info())
    applied: list[bool] = []
    monkeypatch.setattr(
        cli.updater,
        "perform_update",
        lambda r, info: applied.append(True) or cli.updater.UpdateResult(True, "Updated.", "1.0.0", "2.0.0"),
    )

    monkeypatch.setattr("builtins.input", lambda prompt="": "skip")
    assert cli.main(["launch", "terminal"]) == 0
    assert applied == []  # the user skipped, so nothing was applied
    assert cli.updater.is_version_skipped(root, "v2.0.0") is True

    # A second launch must not re-prompt for the same version; a stray prompt would raise here.
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(AssertionError("re-prompted")))
    assert cli.main(["launch", "terminal"]) == 0
    assert applied == []


def test_sync_writes_lock_stamp(env: tuple[Path, list]) -> None:
    """A successful sync records the lock it installed so later launches can detect staleness."""
    root, _ = env
    (root / "uv.lock").write_text("lock-v1", encoding="utf-8")
    assert cli.main(["sync"]) == 0
    assert (root / ".venv" / ".horde-sync-stamp").read_text(encoding="utf-8").strip() == cli._lock_fingerprint(root)


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
