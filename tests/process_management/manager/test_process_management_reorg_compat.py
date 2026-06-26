"""Tests for the process-management namespace reorganization."""

from __future__ import annotations

import importlib
from pathlib import Path

EXPECTED_TOP_LEVEL_FILES = {
    "__init__.py",
    "main_entry_point.py",
    "process_manager.py",
    "worker_entry_points.py",
}


def test_process_management_top_level_stays_thin() -> None:
    """Only the facade and entry-point modules remain at the package root."""
    import horde_worker_regen.process_management as process_management

    package_dir = Path(process_management.__file__).parent
    top_level_files = {path.name for path in package_dir.glob("*.py")}

    assert top_level_files == EXPECTED_TOP_LEVEL_FILES


def test_parent_package_module_import_returns_canonical_module() -> None:
    """Canonical package imports return the same module object as importlib."""
    from horde_worker_regen.process_management.resources import resource_budget

    canonical_module = importlib.import_module("horde_worker_regen.process_management.resources.resource_budget")
    assert resource_budget is canonical_module


def test_private_helper_imports_resolve_from_canonical_module() -> None:
    """Private helpers used by tests resolve from the grouped module path."""
    from horde_worker_regen.process_management.jobs.job_popper import _select_models_for_pop as canonical_helper

    assert canonical_helper.__name__ == "_select_models_for_pop"
