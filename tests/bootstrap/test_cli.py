"""Unit tests for CLI dispatch (no real uv/subprocess is invoked)."""

from __future__ import annotations

from pathlib import Path

import pytest

from worker_bootstrap import backend, cli, detect, paths, utilities_env


def _decision(token: str) -> detect.BackendDecision:
    """A minimal detect-stage BackendDecision resolving to *token* (the hardware-probe seam in tests)."""
    return detect.BackendDecision(stage="detect", final_token=token, reason="test")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, list[tuple[str, object]]]:
    """Isolate the install root in a tmp dir and record uv calls instead of running them."""
    monkeypatch.setattr(cli.paths, "install_root", lambda: tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project.optional-dependencies]\ncu126 = ["torch"]\ncu130 = ["torch"]\ncpu = ["torch"]\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("HORDE_WORKER_BACKEND", raising=False)
    # The hardware-probe seam: the CLI resolves the detected token (and records its decision) through
    # describe_backend_selection. Stub it to cu126 so tests never touch real nvidia-smi.
    monkeypatch.setattr(cli.detect, "describe_backend_selection", lambda: _decision("cu126"))
    # The sync path reconciles the resolved token against the live GPU's compute capability. Stub it to
    # "unreadable" so the default is a no-op; tests that exercise the arch self-heal override this.
    monkeypatch.setattr(cli.detect, "_nvidia_compute_cap", lambda: (0, 0))
    # The post-sync self-check would otherwise spawn `uv run python -c import torch`. Stub it to "no arch
    # list" so it is a no-op by default; the dedicated post-sync tests override this to a real arch list.
    monkeypatch.setattr(cli.runner, "query_torch_arch_list", lambda uv, **kw: None)
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


def test_sync_self_heals_stale_cu126_on_blackwell(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """A persisted cu126 on a Blackwell GPU is clamped up to cu130 (and re-persisted), not reinstalled.

    cu126 carries no sm_120 kernel image, so reinstalling it would leave torch unable to launch a single
    kernel ("PyTorch cannot run this GPU"). The live compute capability must override the stale token.
    """
    root, calls = env
    backend.write_backend_file(root / "bin" / "backend", "cu126")
    monkeypatch.setattr(cli.detect, "_nvidia_compute_cap", lambda: (12, 0))
    assert cli.main(["sync"]) == 0
    assert calls == [("sync", "cu130")]
    assert backend.read_backend_file(root / "bin" / "backend") == "cu130"


def test_sync_detects_backend_when_no_file(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """With no persisted bin/backend, sync uses live detection rather than blindly defaulting to cu126."""
    _, calls = env
    monkeypatch.setattr(cli.detect, "describe_backend_selection", lambda: _decision("cu130"))
    assert cli.main(["sync"]) == 0
    assert calls == [("sync", "cu130")]


def test_sync_overrides_unrunnable_explicit_backend(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """Even an explicit --backend cu126 is clamped on a Blackwell GPU: an unrunnable build helps nobody."""
    root, calls = env
    monkeypatch.setattr(cli.detect, "_nvidia_compute_cap", lambda: (12, 0))
    assert cli.main(["sync", "--backend", "cu126"]) == 0
    assert calls == [("sync", "cu130")]
    assert backend.read_backend_file(root / "bin" / "backend") == "cu130"


def test_sync_keeps_cpu_choice_without_probing(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """A deliberate CPU/alchemist token is never probed or clamped (reconcile is CUDA-only)."""
    root, calls = env
    backend.write_backend_file(root / "bin" / "backend", "cpu")

    def _boom() -> tuple[int, int]:
        raise AssertionError("reconcile probed nvidia-smi for a non-CUDA token")

    monkeypatch.setattr(cli.detect, "_nvidia_compute_cap", _boom)
    assert cli.main(["sync"]) == 0
    assert calls == [("sync", "cpu")]


def test_sync_warns_when_installed_wheel_lacks_kernels(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The post-sync check flags a wheel that cannot run the GPU as a worker bug (stale build map)."""
    _, calls = env
    # Blackwell sm_120 card: the arch floor first upgrades the install to cu130, but (in this stubbed
    # world) the installed wheel still only covers through Hopper -- a residual gap the table failed to
    # predict, which reinstalling the same build cannot fix.
    monkeypatch.setattr(cli.runner, "query_torch_arch_list", lambda uv, **kw: ["sm_80", "sm_90"])
    monkeypatch.setattr(cli.detect, "_nvidia_compute_cap", lambda: (12, 0))
    assert cli.main(["sync"]) == 0  # the install still succeeded; the warning never fails it
    err = capsys.readouterr().err
    assert "no CUDA kernels for this GPU" in err
    assert "sm_120" in err
    assert "worker bug" in err
    assert calls == [("sync", "cu130")]  # the floor upgraded cu126 -> cu130 before the post-check ran


def test_sync_silent_when_installed_wheel_supports_gpu(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A wheel that does carry kernels for the live GPU produces no post-sync warning."""
    _, _ = env
    monkeypatch.setattr(cli.runner, "query_torch_arch_list", lambda uv, **kw: ["sm_90", "sm_100", "sm_120"])
    monkeypatch.setattr(cli.detect, "_nvidia_compute_cap", lambda: (12, 0))
    assert cli.main(["sync"]) == 0
    assert "no CUDA kernels" not in capsys.readouterr().err


def test_sync_post_check_skips_non_cuda_wheel(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ROCm/CPU wheel (no sm_ tags) short-circuits before the compatibility predicate is consulted."""
    _, _ = env
    monkeypatch.setattr(cli.runner, "query_torch_arch_list", lambda uv, **kw: ["gfx1100", "gfx1151"])

    def _boom(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("ran the arch-compatibility predicate on a non-CUDA wheel")

    # Spy on the predicate (not the compute-cap probe, which the reconcile step legitimately uses): the
    # post-sync check must return on the missing sm_ tags before it ever evaluates compatibility.
    monkeypatch.setattr(cli.detect, "gpu_arch_supported", _boom)
    assert cli.main(["sync"]) == 0
    assert "no CUDA kernels" not in capsys.readouterr().err


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


def test_launch_provisions_utilities_when_main_venv_is_already_current(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch repairs the independent utilities venv even when it can skip the main sync."""
    root, calls = env
    _in_sync_venv(root)
    _write_utilities_lock(root)
    provisioned: list[str] = []
    monkeypatch.setattr(
        cli.runner,
        "provision_utilities",
        lambda uv, *, backend_token, **kw: provisioned.append(backend_token),
    )

    assert cli.main(["launch", "terminal"]) == 0
    assert provisioned == ["cu126"]
    assert calls == [("run", ["horde-worker"])]


def _in_sync_venv(root: Path) -> None:
    """A venv whose recorded sync stamp matches the current lock, so a launch would otherwise skip syncing."""
    _make_venv(root)
    (root / "uv.lock").write_text("lock-current", encoding="utf-8")
    (root / ".venv" / ".horde-sync-stamp").write_text(cli._lock_fingerprint(root), encoding="utf-8")


def test_launch_forces_resync_when_installed_torch_cannot_run_gpu(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lock-matching venv whose torch has no kernels for the live GPU is re-synced before starting.

    This is the launch-path backstop for a persisted build that cannot run the card actually present (a
    cu126 install on a Blackwell GPU): the runtime guard alone would refuse every job on every relaunch
    forever, because nothing on the pure-launch path re-runs the installer.
    """
    root, calls = env
    _in_sync_venv(root)
    backend.write_backend_file(root / "bin" / "backend", "cu126")
    # Live Blackwell card; the installed cu126 wheel carries no sm_120 kernel image.
    monkeypatch.setattr(cli.detect, "_nvidia_compute_cap", lambda: (12, 0))
    monkeypatch.setattr(cli.runner, "query_torch_arch_list", lambda uv, **kw: ["sm_80", "sm_90"])
    assert cli.main(["launch", "terminal"]) == 0
    # The forced sync reconciles cu126 -> cu130 (the runnable build) before the worker starts.
    assert calls == [("sync", "cu130"), ("run", ["horde-worker"])]
    assert backend.read_backend_file(root / "bin" / "backend") == "cu130"
    assert "no CUDA kernels for this GPU" in capsys.readouterr().err


def test_launch_stamps_healthy_torch_and_skips_reprobe(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wheel that runs the live GPU is verified once, stamped, and not re-probed on the next launch."""
    root, calls = env
    _in_sync_venv(root)
    monkeypatch.setattr(cli.detect, "_nvidia_compute_cap", lambda: (12, 0))
    monkeypatch.setattr(cli.runner, "query_torch_arch_list", lambda uv, **kw: ["sm_90", "sm_100", "sm_120"])
    assert cli.main(["launch", "terminal"]) == 0
    assert calls == [("run", ["horde-worker"])]
    assert cli.paths.gpu_check_stamp_file(root).is_file()

    # A second launch must not re-run the torch-arch probe now that the check is stamped current.
    def _boom(uv: str, **kw: object) -> list[str]:
        raise AssertionError("re-probed torch arch despite a current GPU-check stamp")

    monkeypatch.setattr(cli.runner, "query_torch_arch_list", _boom)
    calls.clear()
    assert cli.main(["launch", "terminal"]) == 0
    assert calls == [("run", ["horde-worker"])]


def test_launch_warns_but_starts_when_no_corrective_build_exists(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the installed build is already the best for the card yet still cannot run it, do not loop-resync.

    This is the stale-selection-table case (a worker bug): reinstalling the same build cannot help, so the
    launch starts and lets the worker's runtime guard surface the report-worthy specifics.
    """
    root, calls = env
    _in_sync_venv(root)
    backend.write_backend_file(root / "bin" / "backend", "cu130")
    monkeypatch.setattr(cli.detect, "_nvidia_compute_cap", lambda: (12, 0))
    # The installed wheel still cannot run sm_120 even though cu130 is the correct token for the card.
    monkeypatch.setattr(cli.runner, "query_torch_arch_list", lambda uv, **kw: ["sm_80", "sm_90"])
    assert cli.main(["launch", "terminal"]) == 0
    assert calls == [("run", ["horde-worker"])]  # started, never re-synced
    assert "no alternative locked build is known" in capsys.readouterr().err


def test_detect_records_decision_breadcrumb(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """`detect` writes the full selection decision to bin/backend-decision.json for a support bundle."""
    root, _ = env
    monkeypatch.setattr(
        cli.detect,
        "describe_backend_selection",
        lambda: detect.BackendDecision(
            stage="detect",
            final_token="cu130",
            reason="floored for Blackwell",
            driver_cuda_version=(12, 8),
            compute_capability=(12, 0),
            clamp_action=detect.CLAMP_FLOORED,
        ),
    )
    assert cli.main(["detect"]) == 0
    recorded = cli.json.loads(cli.paths.backend_decision_file(root).read_text(encoding="utf-8"))
    assert recorded["detect"]["final_token"] == "cu130"
    assert recorded["detect"]["clamp_action"] == detect.CLAMP_FLOORED
    assert recorded["detect"]["compute_capability"] == "12.0"


def test_sync_reconcile_records_decision_breadcrumb(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """The reconcile step records its own decision stage (why a stale token was clamped) in the breadcrumb."""
    root, _ = env
    backend.write_backend_file(root / "bin" / "backend", "cu126")
    monkeypatch.setattr(cli.detect, "_nvidia_compute_cap", lambda: (12, 0))
    assert cli.main(["sync"]) == 0
    recorded = cli.json.loads(cli.paths.backend_decision_file(root).read_text(encoding="utf-8"))
    assert recorded["reconcile"]["final_token"] == "cu130"
    assert recorded["reconcile"]["input_token"] == "cu126"
    assert recorded["reconcile"]["clamp_action"] == detect.CLAMP_FLOORED


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
    monkeypatch.setattr(cli.updater, "check_for_update", lambda root, **kw: _available_info())
    assert cli.main(["update", "--check"]) == 0
    assert "Update available" in capsys.readouterr().out


def test_update_applies_then_resyncs(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """`update --yes` applies the update and then re-syncs so the new deps are installed."""
    _, calls = env
    monkeypatch.setattr(cli.updater, "check_for_update", lambda root, **kw: _available_info())
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
    monkeypatch.setattr(cli.detect, "describe_backend_selection", lambda: _decision("amd-unsupported"))
    assert cli.main(["detect"]) == 2


def test_env_override_beats_detection(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """HORDE_WORKER_BACKEND=cpu lets an AMD user opt into the CPU build."""
    root, _ = env
    monkeypatch.setattr(cli.detect, "describe_backend_selection", lambda: _decision("amd-unsupported"))
    monkeypatch.setenv("HORDE_WORKER_BACKEND", "cpu")
    assert cli.main(["detect", "--write"]) == 0
    assert backend.read_backend_file(root / "bin" / "backend") == "cpu"


def test_sync_amd_windows_profile_uses_rocm_path(
    env: tuple[Path, list],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AMD Windows ROCm profile tokens bypass locked-extra validation and use the ad-hoc ROCm path."""
    _, calls = env

    def fake_rocm_sync(uv: str, *, token: str, **kw: object) -> int:
        calls.append(("rocm", token))
        return 0

    from worker_bootstrap import rocm

    monkeypatch.setattr(rocm, "sync_rocm", fake_rocm_sync)
    assert cli.main(["sync", "--backend", "rocm-windows"]) == 0
    assert calls == [("rocm", "rocm-windows")]


# --- utilities-venv provisioning wire-in (after a successful sync) ---------------------------------


def _write_utilities_lock(root: Path) -> None:
    """Write a stand-in utilities uv.lock so provisioning has a lock to sync from."""
    lock = paths.utilities_lock_file(root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("utilities-lock\n", encoding="utf-8")


def test_sync_provisions_utilities_when_wanted_and_pinned(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful sync provisions the utilities venv when features are wanted and a pin exists."""
    root, _ = env
    _write_utilities_lock(root)
    provisioned: list[str] = []
    monkeypatch.setattr(
        cli.runner,
        "provision_utilities",
        lambda uv, *, backend_token, **kw: provisioned.append(backend_token),
    )
    assert cli.main(["sync"]) == 0
    assert provisioned == ["cu126"]


def test_sync_provisions_utilities_before_pruning_cache(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The utilities venv is built before `uv cache prune`, so it reuses the wheels the main sync fetched.

    `uv cache prune` reclaims the reusable unpacked wheels (torch and its multi-GB CUDA stack among them),
    and a venv's own hardlinks survive a prune but can no longer serve a fresh install. Pruning before the
    second venv is created would therefore force it to re-download torch; provisioning must come first so
    both environments share one download.
    """
    root, _ = env
    # Make the cache "owned" so the auto-prune actually fires (mirrors test_sync_prunes_only_when_cache_owned).
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("HORDE_WORKER_UV_CACHE_MODE", raising=False)
    _write_utilities_lock(root)
    order: list[str] = []
    monkeypatch.setattr(cli.runner, "provision_utilities", lambda *a, **k: order.append("provision"))
    monkeypatch.setattr(cli.runner, "uv_cache_prune", lambda uv, **kw: (order.append("prune"), (0, 0))[1])
    assert cli.main(["sync"]) == 0
    assert order == ["provision", "prune"]


def test_sync_warns_and_skips_when_full_backend_lock_is_missing(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A full backend whose utilities lock is absent warns (it should ship one) and provisions nothing.

    This is the broken-install signal: the default token (cu126) ships the utilities project by policy, so a
    missing lock means the tree never received requirements/utilities/ (the release-bundle omission this
    guards). It must be diagnosable, not a silent lane-disable, while never failing the completed worker sync.
    """
    _, _ = env
    called: list[bool] = []
    monkeypatch.setattr(cli.runner, "provision_utilities", lambda *a, **k: called.append(True))
    assert cli.main(["sync"]) == 0
    assert called == []
    err = capsys.readouterr().err
    assert "image-utilities lane is enabled" in err
    assert "dependency lock is missing" in err


def test_sync_skip_utilities_flag_suppresses_provision(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--skip-utilities suppresses provisioning even when it would otherwise run."""
    root, _ = env
    _write_utilities_lock(root)
    called: list[bool] = []
    monkeypatch.setattr(cli.runner, "provision_utilities", lambda *a, **k: called.append(True))
    assert cli.main(["sync", "--skip-utilities"]) == 0
    assert called == []


def test_sync_skip_utilities_env_suppresses_provision(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """HORDE_WORKER_SKIP_UTILITIES=1 suppresses provisioning (the emergency env guard)."""
    root, _ = env
    _write_utilities_lock(root)
    monkeypatch.setenv("HORDE_WORKER_SKIP_UTILITIES", "1")
    called: list[bool] = []
    monkeypatch.setattr(cli.runner, "provision_utilities", lambda *a, **k: called.append(True))
    assert cli.main(["sync"]) == 0
    assert called == []


def test_sync_features_none_does_not_want_utilities(env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch) -> None:
    """HORDE_WORKER_FEATURES=none strips the feature set, so the utilities venv is not wanted."""
    root, _ = env
    _write_utilities_lock(root)
    monkeypatch.setenv("HORDE_WORKER_FEATURES", "none")
    called: list[bool] = []
    monkeypatch.setattr(cli.runner, "provision_utilities", lambda *a, **k: called.append(True))
    assert cli.main(["sync"]) == 0
    assert called == []


def test_sync_utilities_provision_failure_is_non_fatal(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A utilities-provision failure warns but never turns a completed worker sync into a failure."""
    root, calls = env
    _write_utilities_lock(root)

    def boom(uv: str, *, backend_token: str, **kw: object) -> None:
        raise utilities_env.UtilitiesProvisionError("could not create the utilities venv (exit code 2).")

    monkeypatch.setattr(cli.runner, "provision_utilities", boom)
    assert cli.main(["sync"]) == 0  # the worker sync succeeded; the utilities failure is post-success
    assert calls == [("sync", "cu126")]
    assert "could not create the utilities venv" in capsys.readouterr().err


def test_maybe_provision_lean_backend_missing_lock_is_silent(
    env: tuple[Path, list], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lean backend opted into a feature it ships no utilities lock for stays a quiet no-op (no warning).

    Unlike a full build, a lean backend (e.g. rocm) is not expected to ship a utilities lock, so an absent
    one is the benign "no pin published yet" case, not a broken install: provision nothing and say nothing.
    """
    root, _ = env
    monkeypatch.setenv("HORDE_WORKER_FEATURES", "controlnet")  # opts the lean backend into wanting the lane
    called: list[bool] = []
    monkeypatch.setattr(cli.runner, "provision_utilities", lambda *a, **k: called.append(True))
    options = cli._SyncOptions(
        preview=False,
        hold=False,
        confirm_threshold_bytes=0,
        headless_policy="proceed",
        prune=False,
        skip_utilities=False,
    )
    cli._maybe_provision_utilities("UV", root, "rocm", options)
    assert called == []
    assert capsys.readouterr().err == ""


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
