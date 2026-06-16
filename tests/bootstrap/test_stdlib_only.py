"""Guard: the bootstrap brain must stay standard-library only.

The platform shims run ``bootstrap.py`` through uv *before* the project venv exists, so a single stray
third-party import (e.g. ``loguru``) anywhere in ``worker_bootstrap`` or ``bootstrap.py`` would reintroduce
the chicken-and-egg this whole design removes. This test fails CI if that happens.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import worker_bootstrap

_PKG_DIR = Path(worker_bootstrap.__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
# Standard library plus the bootstrap's own package are the only allowed top-level imports.
_ALLOWED = set(sys.stdlib_module_names) | {"worker_bootstrap"}


def _top_level_import_roots(path: Path) -> set[str]:
    """Return the root module name of every import (at any nesting) in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_bootstrap_brain_is_stdlib_only() -> None:
    """No module in worker_bootstrap/ (or bootstrap.py) imports a non-stdlib package."""
    files = [*sorted(_PKG_DIR.glob("*.py")), _REPO_ROOT / "bootstrap.py"]
    offenders = {path.name: sorted(bad) for path in files if (bad := _top_level_import_roots(path) - _ALLOWED)}
    assert not offenders, f"non-stdlib imports in the bootstrap brain: {offenders}"
