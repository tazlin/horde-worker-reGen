"""Benchmark wiring: the harness must hand the real worker's model directory to hordelib.

Deterministic, GPU-free regression tests for the ``cache_home`` propagation fix and the
no-local-models pre-flight. The bug: the benchmark never set ``AIWORKER_CACHE_HOME``, so real
inference children resolved hordelib's CWD-relative ``./models`` fallback, found no checkpoints,
and crash-looped with "No models available", wedging the first level until its timeout.
"""

from __future__ import annotations

import os
from pathlib import Path

import horde_model_reference
import pytest

# Import the worker env module at collection time so its module-level ``load_dotenv()`` runs once
# now, before any test body calls ``monkeypatch.delenv``. Otherwise the first test to trigger the
# lazy import inside ``ensure_worker_env`` would have ``load_dotenv()`` re-populate AIWORKER_CACHE_HOME
# from the developer's local .env after the delenv, masking the bridgeData-driven assignment.
from horde_worker_regen import load_env_vars as _load_env_vars  # noqa: F401
from horde_worker_regen.benchmark.controller import BenchmarkController
from horde_worker_regen.benchmark.worker_env import ensure_worker_env


def _write_bridge_data(directory: Path, cache_home: Path) -> None:
    # Forward slashes only: load_env_vars_from_config hard-exits on backslashes in the config.
    (directory / "bridgeData.yaml").write_text(
        f'cache_home: "{cache_home.as_posix()}"\n',
        encoding="utf-8",
    )


def test_ensure_worker_env_sets_cache_home_from_bridge_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real-mode benchmark adopts the worker's configured model directory."""
    models_dir = tmp_path / "mymodels"
    models_dir.mkdir()
    _write_bridge_data(tmp_path, models_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)

    # The failing precondition: with no env var set, hordelib's get_model_directory() falls back to
    # the CWD-relative "models" dir (hordelib/settings.py), which holds none of the worker's models.
    assert os.getenv("AIWORKER_CACHE_HOME") is None
    assert os.environ.get("AIWORKER_CACHE_HOME", "models") == "models"

    ensure_worker_env("real")

    assert os.environ["AIWORKER_CACHE_HOME"] == models_dir.as_posix()


def test_ensure_worker_env_missing_config_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing bridgeData.yaml is best-effort: no exception, no env mutation (fake/CI runs)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)

    ensure_worker_env("fake")

    assert os.getenv("AIWORKER_CACHE_HOME") is None


def test_ensure_worker_env_respects_existing_cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit AIWORKER_CACHE_HOME wins over the config value, matching the worker's precedence."""
    shell_dir = tmp_path / "shellmodels"
    _write_bridge_data(tmp_path, tmp_path / "configmodels")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIWORKER_CACHE_HOME", shell_dir.as_posix())

    ensure_worker_env("real")

    assert os.environ["AIWORKER_CACHE_HOME"] == shell_dir.as_posix()


def test_no_local_models_reason_flags_empty_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-flight returns an actionable skip reason when the weights root holds no checkpoints."""
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    monkeypatch.delenv("AIWORKER_EXTRA_MODEL_DIRECTORIES", raising=False)
    monkeypatch.setattr(horde_model_reference, "resolve_weights_root", lambda _base: empty_root)

    reason = BenchmarkController._no_local_models_reason()

    assert reason is not None
    assert str(empty_root) in reason
    assert "cache_home" in reason


def test_no_local_models_reason_passes_when_checkpoint_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A weights root containing a checkpoint file is not flagged."""
    root = tmp_path / "weights"
    (root / "compvis").mkdir(parents=True)
    (root / "compvis" / "Deliberate.safetensors").write_bytes(b"stub")
    monkeypatch.delenv("AIWORKER_EXTRA_MODEL_DIRECTORIES", raising=False)
    monkeypatch.setattr(horde_model_reference, "resolve_weights_root", lambda _base: root)

    assert BenchmarkController._no_local_models_reason() is None


def test_no_local_models_reason_skipped_with_extra_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With extra model directories configured, the primary-root check is skipped entirely."""
    monkeypatch.setenv("AIWORKER_EXTRA_MODEL_DIRECTORIES", str(tmp_path / "extra"))
    resolved: list[object] = []
    monkeypatch.setattr(
        horde_model_reference,
        "resolve_weights_root",
        lambda _base: resolved.append(_base) or tmp_path,
    )

    assert BenchmarkController._no_local_models_reason() is None
    assert not resolved, "resolve_weights_root must not be consulted when extra dirs are set"
