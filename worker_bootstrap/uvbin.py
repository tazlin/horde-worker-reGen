"""Locate the uv executable: bundled ``bin/uv`` preferred, falling back to ``uv`` on PATH (dev checkouts)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from worker_bootstrap import paths


def uv_executable(root: Path | None = None) -> str:
    """Return the path to uv: bundled ``bin/uv[.exe]`` if present, else ``uv`` from PATH.

    Mirrors ``runtime.cmd``'s precedence so a packaged install uses its pinned uv while a dev checkout uses
    whatever uv is on PATH.
    """
    name = "uv.exe" if os.name == "nt" else "uv"
    bundled = paths.bin_dir(root) / name
    if bundled.exists():
        return str(bundled)
    return shutil.which("uv") or name
