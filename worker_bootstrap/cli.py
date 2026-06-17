"""Command-line entry for the worker bootstrap brain (detect / sync / launch / preload / install)."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from worker_bootstrap import backend as backend_mod
from worker_bootstrap import config_seed, consent, detect, gitbin, paths, runner, sync_plan, uvbin

_BACKEND_ENV = "HORDE_WORKER_BACKEND"
_FEATURES_ENV = "HORDE_WORKER_FEATURES"

_DEFAULT_CONFIRM_MB = 1500

# launch mode -> the uv-run command (console scripts from pyproject [project.scripts]). "bridge" is handled
# separately because it downloads models before starting the worker, matching the old horde-bridge.cmd.
_LAUNCH_COMMANDS: dict[str, list[str]] = {
    "web": ["horde-worker-web"],
    "terminal": ["horde-worker"],
    "host": ["horde-worker-host"],
    "benchmark": ["horde-benchmark"],
}


def _print_amd_unsupported() -> None:
    """Explain that no usable AMD backend was found and how to force a choice."""
    print(
        "An AMD GPU was detected, but no usable GPU backend is available (no ROCm runtime found; ROCm is "
        "Linux-only and DirectML is removed). Re-run with HORDE_WORKER_BACKEND=cpu for the CPU build "
        "(~100x slower), or HORDE_WORKER_BACKEND=rocm after installing ROCm (Linux).",
        file=sys.stderr,
    )


def _print_cpu_notice() -> None:
    """Warn that no GPU was found and the (slow) CPU build will be used."""
    print(
        "No NVIDIA or AMD GPU detected; using the CPU build (~100x slower, mainly for testing).",
        file=sys.stderr,
    )


@dataclass(frozen=True)
class _SyncOptions:
    """Resolved knobs for the preview/hold/prune behaviour of a sync (CLI flag > env var > default)."""

    preview: bool
    hold: bool
    confirm_threshold_bytes: int
    headless_policy: str
    prune: bool


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var; unset/empty falls back to *default*; ``0/false/no/off`` mean False."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _sync_options(args: argparse.Namespace) -> _SyncOptions:
    """Resolve sync behaviour from CLI flags (when present), then env vars, then safe defaults.

    ``args`` may lack the sync flags (e.g. a ``launch`` first-run sync), in which case every flag reads
    ``None`` and the env/default takes over.
    """
    preview = not getattr(args, "no_sync_preview", False) and _env_bool("HORDE_WORKER_SYNC_PREVIEW", True)

    hold_flag = getattr(args, "hold_torch", None)
    hold = hold_flag if hold_flag is not None else _env_bool("HORDE_WORKER_SYNC_HOLD", False)

    confirm_mb = getattr(args, "confirm_above_mb", None)
    if confirm_mb is None:
        raw_mb = os.environ.get("HORDE_WORKER_SYNC_CONFIRM_MB", "")
        try:
            confirm_mb = int(raw_mb) if raw_mb.strip() else _DEFAULT_CONFIRM_MB
        except ValueError:
            confirm_mb = _DEFAULT_CONFIRM_MB

    policy = getattr(args, "headless_policy", None) or os.environ.get("HORDE_WORKER_SYNC_HEADLESS_POLICY") or "proceed"
    policy = policy.strip().lower()
    if policy not in ("proceed", "hold"):
        policy = "proceed"

    prune = not getattr(args, "no_prune", False) and _env_bool("HORDE_WORKER_SYNC_PRUNE", True)

    return _SyncOptions(
        preview=preview,
        hold=hold,
        confirm_threshold_bytes=max(confirm_mb, 0) * 1024 * 1024,
        headless_policy=policy,
        prune=prune,
    )


def _apply_cache_mode_flag(args: argparse.Namespace) -> None:
    """Honour ``--cache-mode`` by setting the env var the shims/runner read (CLI flag > env var).

    Shared mode also clears any ``UV_CACHE_DIR`` a shim pre-set, so uv falls back to its own default
    (system) cache rather than the isolated one for this run.
    """
    mode = getattr(args, "cache_mode", None)
    if mode is None:
        return
    os.environ["HORDE_WORKER_UV_CACHE_MODE"] = mode
    if mode == "shared":
        os.environ.pop("UV_CACHE_DIR", None)


def _cache_is_owned(root: Path) -> bool:
    """Return whether uv's cache for this run is the isolated cache we created (so safe to auto-prune)."""
    if paths.uv_cache_mode() == "shared":
        return False
    preset = os.environ.get("UV_CACHE_DIR")
    return preset is None or Path(preset) == paths.uv_cache_dir(root)


def _effective_cache_dir(root: Path) -> str:
    """Return a human label for the uv cache uv will use this run (for the preview footer)."""
    if paths.uv_cache_mode() == "shared":
        return "uv default (shared) cache"
    return os.environ.get("UV_CACHE_DIR") or str(paths.uv_cache_dir(root))


def _maybe_prune(uv: str, root: Path, options: _SyncOptions) -> None:
    """Auto-prune the owned uv cache after a successful sync (never a shared/redirected cache)."""
    if not options.prune or not _cache_is_owned(root):
        return
    rc, reclaimed = runner.uv_cache_prune(uv, root=root)
    if rc == 0 and reclaimed:
        print(
            f"Reclaimed {sync_plan.human_bytes(reclaimed)} from the worker's uv cache ({_effective_cache_dir(root)})."
        )


def _is_headless() -> bool:
    """Return whether this run is non-interactive (no terminal) or consent was captured upstream."""
    return consent.consent_env_var() is not None or not consent.is_interactive()


def _run_sync(uv: str, root: Path, token: str, feature_extras: tuple[str, ...], options: _SyncOptions) -> int:
    """Run the sync, first showing a download preview and honouring a hold/cancel when one applies.

    Falls back to the plain locked sync whenever the preview is disabled, there is no venv yet (nothing
    to limp along on), or the dry-run cannot be produced/parsed: the preview must never block an update.
    """
    if not options.preview or not paths.venv_dir(root).exists():
        return runner.uv_sync(uv, token, extras=feature_extras, root=root)

    rc_dry, output = runner.uv_sync_dry_run(uv, token, extras=feature_extras, root=root)
    changes = sync_plan.parse_dry_run(output) if rc_dry == 0 else []
    if rc_dry != 0 or not changes:
        if rc_dry != 0:
            print("Could not preview the sync; proceeding with the normal locked sync.", file=sys.stderr)
        return runner.uv_sync(uv, token, extras=feature_extras, root=root)

    overrides_path = paths.sync_overrides_file(root)
    installed = sync_plan.installed_versions(paths.venv_dir(root))
    overrides_text = sync_plan.held_overrides_text(changes, installed)
    holdable = False
    if overrides_text is not None and _write_overrides(overrides_path, overrides_text):
        holdable = (
            runner.uv_sync_held(
                uv,
                token,
                overrides_path=overrides_path,
                extras=feature_extras,
                root=root,
                dry_run=True,
            )
            == 0
        )

    plan = sync_plan.build_plan(
        changes,
        holdable=holdable,
        cache_dir=_effective_cache_dir(root),
        cache_is_owned=_cache_is_owned(root),
        free_disk_bytes=sync_plan.free_bytes(root),
    )
    print(sync_plan.format_sync_plan(plan))

    action = sync_plan.decide(
        plan,
        hold_requested=options.hold,
        headless=_is_headless(),
        headless_policy=options.headless_policy,
        confirm_threshold_bytes=options.confirm_threshold_bytes,
        interactive=consent.is_interactive(),
    )
    if action == "abort":
        print("Sync cancelled; keeping the current environment.")
        return 1
    if action == "hold":
        print("Limping along: holding torch/torchvision at the installed version; updating everything else.")
        return runner.uv_sync_held(uv, token, overrides_path=overrides_path, extras=feature_extras, root=root)
    return runner.uv_sync(uv, token, extras=feature_extras, root=root)


def _write_overrides(path: Path, text: str) -> bool:
    """Write the uv override file used to hold packages; return False (skip the hold) if it cannot be written."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        return False
    return True


def _sync(uv: str, root: Path, *, cli_flag: str | None, options: _SyncOptions) -> int:
    """Disclose, gain consent, ensure git, seed config, then run the sync (with preview) or ROCm path."""
    token = backend_mod.resolve_backend(
        cli_flag=cli_flag,
        env_value=os.environ.get(_BACKEND_ENV),
        file_value=backend_mod.read_backend_file(paths.backend_file(root)),
    )
    if token == detect.AMD_UNSUPPORTED:
        _print_amd_unsupported()
        return 2

    # Disclose what is about to be installed (and from where) and gain consent before any heavy download.
    # The git line tells the user up front whether their existing git is used or a portable one is fetched.
    system_git = gitbin.find_system_git()
    if not consent.ensure_consent(
        notice_path=paths.install_notice(root),
        marker_path=paths.consent_marker(root),
        detail_lines=[f"  - GPU backend to install: {token}", gitbin.notice_line(system_git)],
    ):
        return 1

    # Resolve git now (during the long install), not mid-job: hordelib clones ComfyUI with a bare `git`.
    git_resolution = gitbin.ensure_git(root)
    if not git_resolution.ok:
        print(git_resolution.message, file=sys.stderr)
        return 1

    config_seed.seed_config(template=paths.template_config(root), target=paths.bridge_config(root))
    if token == detect.ROCM:
        from worker_bootstrap import rocm

        rc = rocm.sync_rocm(uv, root=root, hold=options.hold)
        if rc == 0:
            _maybe_prune(uv, root, options)
        return rc
    try:
        backend_mod.validate_locked_extra(token, paths.pyproject_path(root))
        feature_extras = backend_mod.desired_feature_extras(token, env_value=os.environ.get(_FEATURES_ENV))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    features_note = ", ".join(feature_extras) if feature_extras else "none (lean base)"
    print(f"Installing dependencies for GPU backend: {token} (features: {features_note})")
    rc = _run_sync(uv, root, token, feature_extras, options)
    if rc == 0:
        _maybe_prune(uv, root, options)
    return rc


def _ensure_synced(uv: str, root: Path, *, cli_flag: str | None, options: _SyncOptions) -> int:
    """Sync the venv only when it does not yet exist; existing installs start without re-syncing."""
    if paths.venv_dir(root).exists():
        return 0
    return _sync(uv, root, cli_flag=cli_flag, options=options)


def _cmd_detect(args: argparse.Namespace, root: Path, uv: str) -> int:  # noqa: ARG001  (uv unused here)
    """Detect (and optionally persist) the backend token, honouring a flag/env override."""
    token = backend_mod.resolve_backend(
        cli_flag=args.backend,
        env_value=os.environ.get(_BACKEND_ENV),
        detected=detect.detect_backend(),
    )
    if token == detect.AMD_UNSUPPORTED:
        _print_amd_unsupported()
        return 2
    if token == detect.CPU:
        _print_cpu_notice()
    if args.write:
        backend_mod.write_backend_file(paths.backend_file(root), token)
    print(token)
    return 0


def _cmd_sync(args: argparse.Namespace, root: Path, uv: str) -> int:
    """Install/update dependencies for the resolved backend."""
    _apply_cache_mode_flag(args)
    return _sync(uv, root, cli_flag=args.backend, options=_sync_options(args))


def _cmd_launch(args: argparse.Namespace, root: Path, uv: str) -> int:
    """Start the worker in the requested mode, syncing first if the venv is missing."""
    rc = _ensure_synced(uv, root, cli_flag=args.backend, options=_sync_options(args))
    if rc != 0:
        return rc
    if args.mode == "bridge":
        rc = runner.uv_run(uv, ["python", "-s", "download_models.py"], root=root)
        if rc != 0:
            return rc
        return runner.uv_run(uv, ["python", "-s", "run_worker.py", *args.rest], root=root)
    return runner.uv_run(uv, [*_LAUNCH_COMMANDS[args.mode], *args.rest], root=root)


def _cmd_preload(args: argparse.Namespace, root: Path, uv: str) -> int:
    """Download/verify models, then exit."""
    rc = _ensure_synced(uv, root, cli_flag=None, options=_sync_options(args))
    if rc != 0:
        return rc
    return runner.uv_run(uv, ["python", "-s", "download_models.py"], root=root)


def _cmd_run(args: argparse.Namespace, root: Path, uv: str) -> int:
    """Run an arbitrary command in the worker venv (uv run --no-sync), for back-compat passthrough."""
    return runner.uv_run(uv, list(args.rest), root=root)


def _cmd_install(args: argparse.Namespace, root: Path, uv: str) -> int:
    """One-shot first run: detect + persist backend, sync, then launch the web dashboard."""
    token = backend_mod.resolve_backend(
        cli_flag=args.backend,
        env_value=os.environ.get(_BACKEND_ENV),
        detected=detect.detect_backend(),
    )
    if token == detect.AMD_UNSUPPORTED:
        _print_amd_unsupported()
        return 2
    if token == detect.CPU:
        _print_cpu_notice()
    backend_mod.write_backend_file(paths.backend_file(root), token)
    _apply_cache_mode_flag(args)
    rc = _sync(uv, root, cli_flag=token, options=_sync_options(args))
    if rc != 0 or args.no_launch:
        return rc
    return runner.uv_run(uv, _LAUNCH_COMMANDS["web"], root=root)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with one subcommand per bootstrap action."""
    parser = argparse.ArgumentParser(prog="bootstrap.py", description="AI Horde Worker bootstrap.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_backend_flag(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--backend",
            default=None,
            help="Force a torch build (cu126/cu130/cu132/cpu/rocm) instead of detecting/reading bin/backend.",
        )
        # Convenience shortcuts kept for back-compat with the old update-runtime.cmd/sh flag interface
        # (e.g. `update-runtime.cmd --cu126`); each is just `--backend <build>`.
        for build in ("cu126", "cu130", "cu132", "cpu", "rocm"):
            target.add_argument(
                f"--{build}",
                dest="backend",
                action="store_const",
                const=build,
                help=f"Shortcut for --backend {build}.",
            )

    def add_sync_flags(target: argparse.ArgumentParser) -> None:
        """Add the preview/hold/prune/cache-mode knobs (CLI flag > env var > default)."""
        target.add_argument(
            "--no-sync-preview",
            action="store_true",
            help="Skip the pre-sync download preview (env: HORDE_WORKER_SYNC_PREVIEW=0).",
        )
        target.add_argument(
            "--hold-torch",
            dest="hold_torch",
            action="store_const",
            const=True,
            default=None,
            help="Limp along: keep the installed torch/torchvision when no dependency requires the upgrade.",
        )
        target.add_argument(
            "--no-hold-torch",
            dest="hold_torch",
            action="store_const",
            const=False,
            help="Always take torch/torchvision upgrades (opposite of --hold-torch).",
        )
        target.add_argument(
            "--confirm-above-mb",
            type=int,
            default=None,
            help=f"Confirm before downloads larger than N MB on an interactive run (default {_DEFAULT_CONFIRM_MB}).",
        )
        target.add_argument(
            "--headless-policy",
            choices=["proceed", "hold"],
            default=None,
            help="Non-interactive behaviour for big optional upgrades: take them (proceed) or hold (default proceed).",
        )
        target.add_argument(
            "--no-prune",
            action="store_true",
            help="Do not auto-prune the worker's owned uv cache after a successful sync.",
        )
        target.add_argument(
            "--cache-mode",
            choices=["isolated", "shared"],
            default=None,
            help="Use the isolated worker cache (default) or uv's shared cache (env HORDE_WORKER_UV_CACHE_MODE).",
        )

    p_detect = sub.add_parser("detect", help="Detect the GPU/torch build for this machine.")
    add_backend_flag(p_detect)
    p_detect.add_argument("--write", action="store_true", help="Persist the result to bin/backend.")

    p_sync = sub.add_parser("sync", help="Install/update dependencies for the selected build.")
    add_backend_flag(p_sync)
    add_sync_flags(p_sync)

    p_launch = sub.add_parser("launch", help="Start the worker (syncing first if needed).")
    p_launch.add_argument("mode", choices=["web", "terminal", "bridge", "host", "benchmark"])
    add_backend_flag(p_launch)
    # No sync flags here: launch only syncs on first run (no venv), where the preview is skipped anyway,
    # and argparse.REMAINDER below would otherwise swallow them as worker passthrough.
    p_launch.add_argument("rest", nargs=argparse.REMAINDER, help="Arguments passed through to the worker.")

    sub.add_parser("preload", help="Download/verify models, then exit.")

    p_install = sub.add_parser("install", help="Detect, sync, and launch (one-shot first run).")
    add_backend_flag(p_install)
    add_sync_flags(p_install)
    p_install.add_argument("--no-launch", action="store_true", help="Install only; do not start the worker.")

    p_run = sub.add_parser("run", help="Run an arbitrary command in the worker venv (uv run --no-sync).")
    p_run.add_argument("rest", nargs=argparse.REMAINDER, help="The command and its arguments.")

    return parser


_HANDLERS = {
    "detect": _cmd_detect,
    "sync": _cmd_sync,
    "launch": _cmd_launch,
    "preload": _cmd_preload,
    "install": _cmd_install,
    "run": _cmd_run,
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the matching bootstrap action; return its exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Back-compat: the old runtime.cmd/.sh were a generic `uv run` wrapper, so things like
    # `runtime.cmd python -s download_models.py` (and the Dockerfiles README) pass a bare command. If the
    # first token is not a known subcommand (and not a flag), treat the whole line as `run <command...>`.
    if argv and not argv[0].startswith("-") and argv[0] not in _HANDLERS:
        argv = ["run", *argv]
    args = _build_parser().parse_args(argv)
    root = paths.install_root()
    uv = uvbin.uv_executable(root)
    return _HANDLERS[args.command](args, root, uv)
