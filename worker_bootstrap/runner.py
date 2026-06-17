"""Subprocess wrappers that invoke uv with a clean, isolated child environment.

Centralizes the isolation triplet (``PYTHONNOUSERSITE`` / ``PYTHONPATH`` / ``CONDA_SHLVL``) and the
on-drive cache and managed-Python locations the shell scripts used to each set by hand, and returns the
child's real exit code so a failure can never be misread as success (the old ``errorlevel`` / ``exit /b``
trap that printed "Installation complete" after uv had failed).
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from worker_bootstrap import paths


def build_child_env(root: Path | None = None) -> dict[str, str]:
    """Return a copy of ``os.environ`` hardened for a reproducible uv/Python child process."""
    env = dict(os.environ)
    # Isolation: ignore user site-packages, a stray PYTHONPATH, and a half-activated conda env.
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    env.pop("CONDA_SHLVL", None)
    # We are invoked from `uv run --script`, which exports VIRTUAL_ENV pointing at its throwaway script
    # environment. Drop it so the uv we spawn here targets the project's .venv cleanly (and picks the
    # managed Python) instead of warning that VIRTUAL_ENV does not match and falling back to a system one.
    env.pop("VIRTUAL_ENV", None)
    # Keep uv's cache, managed CPython, and downloaded models in the peered data dir (a sibling of the
    # worker folder, preserved when the worker folder is deleted or reinstalled) rather than the home
    # drive. Respect caller-set values so power users can redirect them (the runtime shims set them first).
    # In shared cache mode we deliberately leave UV_CACHE_DIR unset so uv uses its own (system) default
    # cache the user already populates for other projects; we then never auto-prune a cache we do not own.
    if paths.uv_cache_mode() != "shared":
        env.setdefault("UV_CACHE_DIR", str(paths.uv_cache_dir(root)))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(paths.python_install_dir(root)))
    # Propagate only the data-dir LOCATION, not AIWORKER_CACHE_HOME itself: forcing the models env var here
    # would outrank `cache_home` in bridgeData.yaml. The worker derives the <data>/models default from this
    # at the lowest precedence (load_env_vars.py), keeping env var > cache_home > peered default.
    env.setdefault("HORDE_WORKER_DATA_DIR", str(paths.data_root(root)))
    # Use a uv-managed CPython, never a system one: a packaged install must be self-contained, so a user
    # uninstalling their own Python can't break .venv. (pyproject's "managed" only *prefers* managed.)
    env.setdefault("UV_PYTHON_PREFERENCE", "only-managed")
    # If a bundled portable git was provisioned (Windows, no system git), put it on PATH for the child.
    # hordelib's inference subprocesses inherit this env and call a bare `git` to clone ComfyUI on first
    # run, so this is what makes that clone succeed with no system git and no hordelib change.
    git_cmd_dir = paths.git_cmd_dir(root)
    if git_cmd_dir.is_dir():
        env["PATH"] = os.pathsep.join([str(git_cmd_dir), env.get("PATH", "")])
    return env


def run_uv(uv: str, args: list[str], *, root: Path | None = None) -> int:
    """Run uv with ``args`` in the install dir and the hardened env; return its exit code.

    Ctrl+C is delivered by the OS to the whole foreground process group, so the child (uv and,
    underneath it, the worker) receives the same SIGINT we do and runs its own graceful shutdown.
    We therefore swallow ``KeyboardInterrupt`` and keep waiting for the child to exit rather than
    letting ``subprocess.run`` SIGKILL it and unwind with a traceback, which would orphan the worker
    mid-drain. Repeated Ctrl+C simply loops here while the worker's handler escalates its own exit.
    """
    root = root or paths.install_root()
    process = subprocess.Popen([uv, *args], cwd=str(root), env=build_child_env(root))
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            continue


def uv_sync(
    uv: str,
    extra: str,
    *,
    extras: Sequence[str] = (),
    root: Path | None = None,
    locked: bool = True,
) -> int:
    """Run ``uv sync [--locked] --extra <extra> [--extra <e> ...]`` and return its exit code.

    ``extra`` is the torch build; ``extras`` are any additional (feature) extras to install alongside
    it, each passed as its own ``--extra`` flag.
    """
    args = ["sync"]
    if locked:
        args.append("--locked")
    args += ["--extra", extra]
    for feature_extra in extras:
        args += ["--extra", feature_extra]
    return run_uv(uv, args, root=root)


def uv_sync_dry_run(
    uv: str,
    extra: str,
    *,
    extras: Sequence[str] = (),
    root: Path | None = None,
    locked: bool = True,
) -> tuple[int, str]:
    """Run ``uv sync ... --dry-run`` and return ``(exit_code, combined_output)`` without touching the env.

    Builds the same argv as :func:`uv_sync` plus ``--dry-run`` so the preview reflects exactly what the
    real sync would do. Output is captured (stdout+stderr) for the plan parser rather than streamed, so
    this uses a plain :func:`subprocess.run` instead of the foreground ``Popen`` wrapper.
    """
    root = root or paths.install_root()
    args = ["sync", "--dry-run"]
    if locked:
        args.append("--locked")
    args += ["--extra", extra]
    for feature_extra in extras:
        args += ["--extra", feature_extra]
    result = subprocess.run(
        [uv, *args],
        cwd=str(root),
        env=build_child_env(root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def uv_sync_held(
    uv: str,
    extra: str,
    *,
    overrides_path: Path,
    extras: Sequence[str] = (),
    root: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Run an opt-out sync that holds packages at the versions named in ``overrides_path``.

    This is the "limp along" path: it drops ``--locked`` (the lock would force the held packages to
    advance) and passes ``--override`` so uv keeps torch/torchvision at the currently-installed version
    while still installing the rest of the new lock. ``--locked`` is intentionally absent here and ONLY
    here; the normal :func:`uv_sync` path always keeps it.

    With ``dry_run=True`` this only resolves (no install) and returns uv's exit code, which is the
    authoritative REQUIRED-vs-OPTIONAL test: a non-zero code means a real dependency floor forbids the
    hold, so the upgrade is mandatory. The dry-run output is captured (not streamed) to stay quiet.
    """
    args = ["sync", "--extra", extra]
    for feature_extra in extras:
        args += ["--extra", feature_extra]
    args += ["--override", str(overrides_path)]
    if dry_run:
        args.append("--dry-run")
        result = subprocess.run(
            [uv, *args],
            cwd=str(root or paths.install_root()),
            env=build_child_env(root),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode
    return run_uv(uv, args, root=root)


_PRUNE_REMOVED_RE = re.compile(r"\(([\d.]+)\s*([KMGT]?i?B)\)")
_UNIT_BYTES = {
    "B": 1,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
}


def uv_cache_prune(uv: str, *, root: Path | None = None) -> tuple[int, int]:
    """Run ``uv cache prune`` (never ``clean``) and return ``(exit_code, reclaimed_bytes)``.

    ``uv cache prune`` removes only unused cache entries and is hardlink-safe (it can never break an
    existing venv), so it is safe to run after a successful sync to reclaim superseded wheels. The
    reclaimed size is parsed best-effort from uv's "Removed N files (X MiB)" summary; 0 when not found.
    """
    root = root or paths.install_root()
    result = subprocess.run(
        [uv, "cache", "prune"],
        cwd=str(root),
        env=build_child_env(root),
        capture_output=True,
        text=True,
        check=False,
    )
    reclaimed = 0
    match = _PRUNE_REMOVED_RE.search((result.stdout or "") + (result.stderr or ""))
    if match:
        amount, unit = match.group(1), match.group(2)
        try:
            reclaimed = int(float(amount) * _UNIT_BYTES.get(unit, 1))
        except ValueError:
            reclaimed = 0
    return result.returncode, reclaimed


def uv_run(uv: str, command: list[str], *, root: Path | None = None, no_sync: bool = True) -> int:
    """Run ``uv run [--no-sync] <command...>`` and return its exit code."""
    args = ["run"]
    if no_sync:
        args.append("--no-sync")
    args += command
    return run_uv(uv, args, root=root)
