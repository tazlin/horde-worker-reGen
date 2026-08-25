"""Regression tests for the AIWORKER_CACHE_HOME precedence ladder applied by ``load_env_vars``.

The peered-data layout introduced a bug: the runtime shim used to export ``AIWORKER_CACHE_HOME``
unconditionally, which outranked a user's ``cache_home`` in ``bridgeData.yaml`` (the worker treats a
pre-set env var as higher precedence than config). The fix has the shim export only the data-dir
*location* (``HORDE_WORKER_DATA_DIR``) and lets ``load_env_vars`` derive ``<data>/models`` at the
LOWEST precedence, so the ladder is: user/system env var > config ``cache_home`` > peered default.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Run the module-level load_dotenv() once now, before any delenv below, so a developer's local .env
# cannot repopulate AIWORKER_CACHE_HOME mid-test and mask the assignment under test.
from horde_worker_regen.load_env_vars import load_env_vars_from_config


def _write_bridge_data(directory: Path, *, cache_home: Path | None = None) -> None:
    body = "dreamer_name: test\n"
    if cache_home is not None:
        body += f'cache_home: "{cache_home.as_posix()}"\n'
    (directory / "bridgeData.yaml").write_text(body, encoding="utf-8")


def test_env_var_wins_over_config_and_peered_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A user/system-set AIWORKER_CACHE_HOME is never overridden by config or the peered default."""
    _write_bridge_data(tmp_path, cache_home=tmp_path / "configmodels")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HORDE_WORKER_DATA_DIR", str(tmp_path / "HordeWorker-data"))
    monkeypatch.setenv("AIWORKER_CACHE_HOME", (tmp_path / "usermodels").as_posix())

    load_env_vars_from_config()

    assert os.environ["AIWORKER_CACHE_HOME"] == (tmp_path / "usermodels").as_posix()


def test_config_cache_home_wins_over_peered_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env var, bridgeData.yaml `cache_home` beats the peered <data>/models default."""
    config_models = tmp_path / "configmodels"
    _write_bridge_data(tmp_path, cache_home=config_models)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HORDE_WORKER_DATA_DIR", str(tmp_path / "HordeWorker-data"))
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)

    load_env_vars_from_config()

    assert os.environ["AIWORKER_CACHE_HOME"] == config_models.as_posix()


def test_peered_default_applies_when_neither_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env var and no `cache_home`, models default to <HORDE_WORKER_DATA_DIR>/models."""
    _write_bridge_data(tmp_path)  # no cache_home line
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "HordeWorker-data"
    monkeypatch.setenv("HORDE_WORKER_DATA_DIR", str(data_dir))
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)

    load_env_vars_from_config()

    assert os.environ["AIWORKER_CACHE_HOME"] == os.path.join(str(data_dir), "models")


def test_no_peered_default_without_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside the scripted installer (no HORDE_WORKER_DATA_DIR), nothing is forced (git/manual users)."""
    _write_bridge_data(tmp_path)  # no cache_home line
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HORDE_WORKER_DATA_DIR", raising=False)
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)

    load_env_vars_from_config()

    assert os.getenv("AIWORKER_CACHE_HOME") is None


def test_plain_yaml_windows_paths_do_not_abort_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A native custom-model path is a valid YAML scalar and must survive the environment preflight."""
    (tmp_path / "bridgeData.yaml").write_text(
        "dreamer_name: test\n"
        "custom_models:\n"
        "- name: Local model\n"
        "  baseline: stable_diffusion_xl\n"
        "  filepath: T:\\models\\local.safetensors\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HORDE_WORKER_DATA_DIR", raising=False)
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)

    load_env_vars_from_config()

    assert os.getenv("AIWORKER_CACHE_HOME") is None


def test_huggingface_cache_follows_the_cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The hub cache lands under AIWORKER_CACHE_HOME and an ambient HF_HUB_CACHE no longer outranks it."""
    _write_bridge_data(tmp_path, cache_home=tmp_path / "configmodels")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)
    monkeypatch.delenv("HORDE_WORKER_DATA_DIR", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "elsewhere"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    load_env_vars_from_config()

    expected = Path(os.environ["AIWORKER_CACHE_HOME"], "hf_transformers")
    assert Path(os.environ["HF_HOME"]) == expected
    assert "HF_HUB_CACHE" not in os.environ
    legacy = os.environ["AIWORKER_HF_LEGACY_HUB_CACHES"].split(os.pathsep)
    assert str(tmp_path / "elsewhere") in legacy, "the ambient hub cache is recorded for migration"
    assert str(expected / "hub") not in legacy


def test_huggingface_cache_untouched_without_a_cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no cache root to isolate into, the hub stack keeps whatever the environment said."""
    _write_bridge_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)
    monkeypatch.delenv("HORDE_WORKER_DATA_DIR", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "ambient"))

    load_env_vars_from_config()

    assert os.environ["HF_HOME"] == str(tmp_path / "ambient")
