"""Filesystem locations for a worker install, derived from where this package is bundled.

The bundle lays ``worker_bootstrap/`` next to ``bootstrap.py`` at the install root, so every path is found
relative to this file without needing the caller to pass the root in.
"""

from __future__ import annotations

import os
from pathlib import Path


def install_root() -> Path:
    """Return the install directory (the parent of this bundled ``worker_bootstrap/`` package)."""
    return Path(__file__).resolve().parent.parent


def data_root(root: Path | None = None) -> Path:
    """Return the sibling data directory that holds the reusable, expensive-to-rebuild artifacts.

    The uv cache, the managed CPython, and downloaded models live here, peered alongside the worker
    folder rather than nested inside it, so deleting or reinstalling the worker folder cannot take the
    user's models and cached dependencies with it. The default is ``<worker_root>-data`` (same name with
    a ``-data`` suffix); ``HORDE_WORKER_DATA_DIR`` overrides the location outright (e.g. another drive).
    This resolution must stay identical to the shell shims (runtime.sh / runtime.cmd) that set the same
    env before Python runs.
    """
    override = os.environ.get("HORDE_WORKER_DATA_DIR")
    if override:
        return Path(override)
    r = root or install_root()
    return r.parent / (r.name + "-data")


def bin_dir(root: Path | None = None) -> Path:
    """Return the ``bin/`` directory that holds uv, the managed Python, and ``bin/backend``."""
    return (root or install_root()) / "bin"


def uv_cache_dir(root: Path | None = None) -> Path:
    """Return uv's package cache directory, in the peered data dir (preserved across worker reinstalls)."""
    return data_root(root) / "uv_cache"


def python_install_dir(root: Path | None = None) -> Path:
    """Return where uv provisions managed CPython, in the peered data dir (preserved across reinstalls)."""
    return data_root(root) / "python"


def uv_cache_mode() -> str:
    """Return the uv cache mode: ``"shared"`` or ``"isolated"`` (default).

    ``isolated`` (the default) keeps a private uv cache under the peered data dir so a managed install
    can prune it safely and never duplicates wheels with another tool's cache by accident. ``shared``
    leaves ``UV_CACHE_DIR`` unset so uv uses its own default (system) cache, which a power user already
    populates for other projects: this avoids duplicating 7-10 GB, at the cost of us never auto-pruning a
    cache we do not own. Set via ``HORDE_WORKER_UV_CACHE_MODE``; any value other than ``shared`` (case
    insensitive) is treated as ``isolated``. This resolution must match the shell shims.
    """
    return "shared" if os.environ.get("HORDE_WORKER_UV_CACHE_MODE", "").strip().lower() == "shared" else "isolated"


def sync_overrides_file(root: Path | None = None) -> Path:
    """Return the path of the generated uv override file used to hold packages during an opt-out sync.

    Written to the writable peered data dir (not the worker folder, which a release update overwrites)
    so "limp along" holds survive a reinstall and never collide with bundled files.
    """
    return data_root(root) / "sync-overrides.txt"


def models_dir(root: Path | None = None) -> Path:
    """Return the model-weights directory (``AIWORKER_CACHE_HOME``), in the peered, preserved data dir."""
    return data_root(root) / "models"


def backend_file(root: Path | None = None) -> Path:
    """Return the path of the persisted backend token (``bin/backend``)."""
    return bin_dir(root) / "backend"


def consent_marker(root: Path | None = None) -> Path:
    """Return the marker that records install consent was captured (``bin/install-consent``).

    Lives under ``bin/`` so the uninstaller removes it (a reinstall then re-asks), and so the .exe's
    deferred first-launch sync and later dependency updates do not re-prompt once consent is recorded.
    """
    return bin_dir(root) / "install-consent"


def git_dir(root: Path | None = None) -> Path:
    """Return where a bundled portable git (MinGit, Windows fallback) is unpacked (``bin/git``)."""
    return bin_dir(root) / "git"


def git_cmd_dir(root: Path | None = None) -> Path:
    """Return the bundled git's ``cmd/`` directory (holds ``git.exe``), to prepend to a child PATH."""
    return git_dir(root) / "cmd"


def install_notice(root: Path | None = None) -> Path:
    """Return the bundled plain-language install notice (``INSTALL_NOTICE.txt``)."""
    return (root or install_root()) / "INSTALL_NOTICE.txt"


def venv_dir(root: Path | None = None) -> Path:
    """Return the project virtual environment directory (``.venv``)."""
    return (root or install_root()) / ".venv"


def pyproject_path(root: Path | None = None) -> Path:
    """Return the bundled ``pyproject.toml`` path."""
    return (root or install_root()) / "pyproject.toml"


def lock_path(root: Path | None = None) -> Path:
    """Return the bundled ``uv.lock`` path (the resolved versions a sync installs)."""
    return (root or install_root()) / "uv.lock"


def sync_stamp_file(root: Path | None = None) -> Path:
    """Return the stamp recording the lockfile the venv was last synced against (``.venv/.horde-sync-stamp``).

    It lives inside the venv so it is discarded whenever the venv is recreated, keeping the recorded
    fingerprint and the actually-installed packages consistent. An in-place update overlays a new
    ``uv.lock`` but preserves the venv, so comparing this stamp to the current lock is what tells a plain
    launch the dependencies changed and a re-sync is due.
    """
    return venv_dir(root) / ".horde-sync-stamp"


def template_config(root: Path | None = None) -> Path:
    """Return the bundled ``bridgeData_template.yaml`` path."""
    return (root or install_root()) / "bridgeData_template.yaml"


def bridge_config(root: Path | None = None) -> Path:
    """Return the user's ``bridgeData.yaml`` path."""
    return (root or install_root()) / "bridgeData.yaml"
