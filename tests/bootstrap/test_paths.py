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
