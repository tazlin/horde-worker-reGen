"""Unit tests for the peered data-dir path resolution (uv cache / managed Python / models)."""

from __future__ import annotations

from pathlib import Path

import pytest

from worker_bootstrap import paths


def test_data_root_is_peered_sibling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The data dir is a sibling of the worker folder (same name + '-data'), not nested inside it."""
    monkeypatch.delenv("HORDE_WORKER_DATA_DIR", raising=False)
    worker = tmp_path / "HordeWorker"

    assert paths.data_root(worker) == tmp_path / "HordeWorker-data"
    assert paths.uv_cache_dir(worker) == tmp_path / "HordeWorker-data" / "uv_cache"
    assert paths.python_install_dir(worker) == tmp_path / "HordeWorker-data" / "python"
    assert paths.models_dir(worker) == tmp_path / "HordeWorker-data" / "models"


def test_data_root_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """HORDE_WORKER_DATA_DIR overrides the location outright (e.g. to place models on another drive)."""
    override = tmp_path / "elsewhere" / "horde-data"
    monkeypatch.setenv("HORDE_WORKER_DATA_DIR", str(override))

    assert paths.data_root(tmp_path / "HordeWorker") == override
    assert paths.models_dir(tmp_path / "HordeWorker") == override / "models"


def test_utilities_venv_is_peered_data_dir_sibling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The utilities venv lives in the peered data dir alongside the managed Python and uv cache."""
    monkeypatch.delenv("HORDE_WORKER_DATA_DIR", raising=False)
    worker = tmp_path / "HordeWorker"

    assert paths.utilities_venv_dir(worker) == tmp_path / "HordeWorker-data" / "utilities-venv"
    assert paths.utilities_stamp_file(worker) == (
        tmp_path / "HordeWorker-data" / "utilities-venv" / ".horde-utilities-stamp"
    )


def test_utilities_python_is_platform_aware(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The utilities interpreter is Scripts/python.exe on Windows and bin/python elsewhere."""
    venv = paths.utilities_venv_dir(tmp_path)

    monkeypatch.setattr(paths.os, "name", "nt")
    assert paths.utilities_python(tmp_path) == venv / "Scripts" / "python.exe"

    monkeypatch.setattr(paths.os, "name", "posix")
    assert paths.utilities_python(tmp_path) == venv / "bin" / "python"
