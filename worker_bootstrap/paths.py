"""Filesystem locations for a worker install, derived from where this package is bundled.

The bundle lays ``worker_bootstrap/`` next to ``bootstrap.py`` at the install root, so every path is found
relative to this file without needing the caller to pass the root in.
"""

from __future__ import annotations

from pathlib import Path


def install_root() -> Path:
    """Return the install directory (the parent of this bundled ``worker_bootstrap/`` package)."""
    return Path(__file__).resolve().parent.parent


def bin_dir(root: Path | None = None) -> Path:
    """Return the ``bin/`` directory that holds uv, the managed Python, and ``bin/backend``."""
    return (root or install_root()) / "bin"


def uv_cache_dir(root: Path | None = None) -> Path:
    """Return uv's package cache directory, co-located with the install (not on the home drive)."""
    return bin_dir(root) / "uv_cache"


def python_install_dir(root: Path | None = None) -> Path:
    """Return where uv should provision managed CPython, kept on the install drive under ``bin/``."""
    return bin_dir(root) / "python"


def backend_file(root: Path | None = None) -> Path:
    """Return the path of the persisted backend token (``bin/backend``)."""
    return bin_dir(root) / "backend"


def venv_dir(root: Path | None = None) -> Path:
    """Return the project virtual environment directory (``.venv``)."""
    return (root or install_root()) / ".venv"


def pyproject_path(root: Path | None = None) -> Path:
    """Return the bundled ``pyproject.toml`` path."""
    return (root or install_root()) / "pyproject.toml"


def template_config(root: Path | None = None) -> Path:
    """Return the bundled ``bridgeData_template.yaml`` path."""
    return (root or install_root()) / "bridgeData_template.yaml"


def bridge_config(root: Path | None = None) -> Path:
    """Return the user's ``bridgeData.yaml`` path."""
    return (root or install_root()) / "bridgeData.yaml"
