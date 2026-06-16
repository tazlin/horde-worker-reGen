"""Command-line entry for the worker bootstrap brain (detect / sync / launch / preload / install)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from worker_bootstrap import backend as backend_mod
from worker_bootstrap import config_seed, consent, detect, gitbin, paths, runner, uvbin

_BACKEND_ENV = "HORDE_WORKER_BACKEND"

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


def _sync(uv: str, root: Path, *, cli_flag: str | None) -> int:
    """Disclose, gain consent, ensure git, seed config, then run the locked sync (or ad-hoc ROCm path)."""
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

        return rocm.sync_rocm(uv, root=root)
    try:
        backend_mod.validate_locked_extra(token, paths.pyproject_path(root))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Installing dependencies for GPU backend: {token}")
    return runner.uv_sync(uv, token, root=root)


def _ensure_synced(uv: str, root: Path, *, cli_flag: str | None) -> int:
    """Sync the venv only when it does not yet exist; existing installs start without re-syncing."""
    if paths.venv_dir(root).exists():
        return 0
    return _sync(uv, root, cli_flag=cli_flag)


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
    return _sync(uv, root, cli_flag=args.backend)


def _cmd_launch(args: argparse.Namespace, root: Path, uv: str) -> int:
    """Start the worker in the requested mode, syncing first if the venv is missing."""
    rc = _ensure_synced(uv, root, cli_flag=args.backend)
    if rc != 0:
        return rc
    if args.mode == "bridge":
        rc = runner.uv_run(uv, ["python", "-s", "download_models.py"], root=root)
        if rc != 0:
            return rc
        return runner.uv_run(uv, ["python", "-s", "run_worker.py", *args.rest], root=root)
    return runner.uv_run(uv, [*_LAUNCH_COMMANDS[args.mode], *args.rest], root=root)


def _cmd_preload(args: argparse.Namespace, root: Path, uv: str) -> int:  # noqa: ARG001  (args unused)
    """Download/verify models, then exit."""
    rc = _ensure_synced(uv, root, cli_flag=None)
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
    rc = _sync(uv, root, cli_flag=token)
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

    p_detect = sub.add_parser("detect", help="Detect the GPU/torch build for this machine.")
    add_backend_flag(p_detect)
    p_detect.add_argument("--write", action="store_true", help="Persist the result to bin/backend.")

    p_sync = sub.add_parser("sync", help="Install/update dependencies for the selected build.")
    add_backend_flag(p_sync)

    p_launch = sub.add_parser("launch", help="Start the worker (syncing first if needed).")
    p_launch.add_argument("mode", choices=["web", "terminal", "bridge", "host", "benchmark"])
    add_backend_flag(p_launch)
    p_launch.add_argument("rest", nargs=argparse.REMAINDER, help="Arguments passed through to the worker.")

    sub.add_parser("preload", help="Download/verify models, then exit.")

    p_install = sub.add_parser("install", help="Detect, sync, and launch (one-shot first run).")
    add_backend_flag(p_install)
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
