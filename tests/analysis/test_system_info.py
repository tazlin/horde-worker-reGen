"""Unit tests for torch-free system-info and cache-inventory collection."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_worker_regen.analysis.cache_inventory import collect_cache_inventory
from horde_worker_regen.analysis.system_info import (
    collect_system_info,
    config_secret_values,
    resolve_cache_home,
)


class TestSystemInfo:
    """The host context block, collected without touching torch in-process."""

    def test_has_core_keys_without_gpu_probe(self) -> None:
        """Core fields are present and the GPU probe is not run unless asked."""
        info = collect_system_info(probe_gpu=False)
        assert info["worker_version"]
        assert "platform" in info and "python_version" in info
        assert info["ram"]["total_bytes"] > 0
        assert "accelerators" not in info

    def test_cache_home_disk_when_provided(self, tmp_path: Path) -> None:
        """A provided cache_home gets a disk-free reading."""
        info = collect_system_info(cache_home=str(tmp_path))
        assert info["cache_home"] == str(tmp_path)
        assert info["disk"]["cache_home"]["total_bytes"] > 0


class TestConfigReads:
    """Best-effort reads from bridgeData.yaml that feed redaction and cache resolution."""

    def test_resolve_cache_home_prefers_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit AIWORKER_CACHE_HOME wins over the config value."""
        monkeypatch.setenv("AIWORKER_CACHE_HOME", str(tmp_path / "env_cache"))
        config = tmp_path / "bridgeData.yaml"
        config.write_text("cache_home: T:/config_cache\n", encoding="utf-8")
        assert resolve_cache_home(config) == str(tmp_path / "env_cache")

    def test_resolve_cache_home_falls_back_to_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no env var, the config's cache_home is used."""
        monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)
        config = tmp_path / "bridgeData.yaml"
        config.write_text("cache_home: T:/config_cache\n", encoding="utf-8")
        assert resolve_cache_home(config) == "T:/config_cache"

    def test_config_secret_values_extracted(self, tmp_path: Path) -> None:
        """The api_key and civitai token are read from the config for value-based redaction."""
        config = tmp_path / "bridgeData.yaml"
        config.write_text("api_key: SECRETKEY123456789012\ncivitai_api_token: civitoken123\n", encoding="utf-8")
        values = config_secret_values(config)
        assert "SECRETKEY123456789012" in values
        assert "civitoken123" in values

    def test_malformed_config_is_empty_not_fatal(self, tmp_path: Path) -> None:
        """A malformed config yields no secrets rather than raising (the pattern backstop covers it)."""
        config = tmp_path / "bridgeData.yaml"
        config.write_text("this: : : not valid yaml: [", encoding="utf-8")
        assert config_secret_values(config) == [None, None, None]


class TestCacheInventory:
    """The on-disk model listing."""

    def test_lists_model_files_with_sizes(self, tmp_path: Path) -> None:
        """Model-like files are listed with sizes and totalled; non-model files are ignored."""
        (tmp_path / "a.safetensors").write_bytes(b"x" * 100)
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.ckpt").write_bytes(b"y" * 50)
        (tmp_path / "notes.txt").write_bytes(b"ignore me")
        inventory = collect_cache_inventory(str(tmp_path))
        assert inventory["present"] is True
        assert inventory["model_file_count"] == 2
        assert inventory["total_model_bytes"] == 150
        # Largest first.
        assert inventory["files"][0]["path"] == "a.safetensors"

    def test_missing_cache_is_a_stub(self) -> None:
        """An unresolved cache_home is recorded as not-present rather than crashing."""
        assert collect_cache_inventory(None)["present"] is False
