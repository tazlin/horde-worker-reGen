"""Subprocess wrappers that invoke uv with a clean, isolated child environment.

Centralizes the isolation triplet (``PYTHONNOUSERSITE`` / ``PYTHONPATH`` / ``CONDA_SHLVL``) and the
on-drive cache and managed-Python locations the shell scripts used to each set by hand, and returns the
child's real exit code so a failure can never be misread as success (the old ``errorlevel`` / ``exit /b``
trap that printed "Installation complete" after uv had failed).
"""

from __future__ import annotations

import os
import subprocess
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
    # Keep uv's cache and managed CPython on the install drive (and under bin/, which uninstall removes)
    # rather than the home drive. Respect caller-set values so power users can redirect them.
    env.setdefault("UV_CACHE_DIR", str(paths.uv_cache_dir(root)))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(paths.python_install_dir(root)))
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
    """Run uv with ``args`` in the install dir and the hardened env; return its exit code."""
    root = root or paths.install_root()
    completed = subprocess.run([uv, *args], cwd=str(root), env=build_child_env(root), check=False)
    return completed.returncode


def uv_sync(uv: str, extra: str, *, root: Path | None = None, locked: bool = True) -> int:
    """Run ``uv sync [--locked] --extra <extra>`` and return its exit code."""
    args = ["sync"]
    if locked:
        args.append("--locked")
    args += ["--extra", extra]
    return run_uv(uv, args, root=root)


def uv_run(uv: str, command: list[str], *, root: Path | None = None, no_sync: bool = True) -> int:
    """Run ``uv run [--no-sync] <command...>`` and return its exit code."""
    args = ["run"]
    if no_sync:
        args.append("--no-sync")
    args += command
    return run_uv(uv, args, root=root)
