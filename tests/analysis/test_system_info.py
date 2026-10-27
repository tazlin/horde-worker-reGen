"""Unit tests for torch-free system-info and cache-inventory collection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from horde_worker_regen.analysis import system_info
from horde_worker_regen.analysis.cache_inventory import collect_cache_inventory
from horde_worker_regen.analysis.system_info import (
    collect_system_info,
    config_secret_values,
    resolve_cache_home,
    rotational_storage,
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


@pytest.fixture
def fake_sysfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the probe at a sysfs tree under tmp_path, with device entries named ``major_minor``.

    The real entries are named ``major:minor``, which is not a valid file name on Windows.
    """
    monkeypatch.setattr(system_info, "_SYSFS_ROOT", tmp_path)
    monkeypatch.setattr(
        system_info, "_block_device_entry", lambda *, major, minor: tmp_path / "dev" / "block" / f"{major}_{minor}"
    )
    return tmp_path


def _write_block_device(sysfs: Path, *, number: str, device_name: str, device_type: str) -> None:
    entry = sysfs / "dev" / "block" / number
    entry.mkdir(parents=True)
    (entry / "uevent").write_text(f"MAJOR=8\nDEVNAME={device_name}\nDEVTYPE={device_type}\n", encoding="utf-8")


class TestRotationalStorage:
    """The storage-type probe, which reports None whenever the media type cannot be read."""

    @pytest.mark.parametrize(("flag", "expected"), [("1\n", True), ("0\n", False)])
    def test_linux_partition_reads_its_disk_flag(self, fake_sysfs: Path, flag: str, expected: bool) -> None:
        """A partition is walked to its parent disk, whose queue/rotational flag determines the result."""
        _write_block_device(fake_sysfs, number="8_1", device_name="sda1", device_type="partition")
        (fake_sysfs / "block" / "sda" / "sda1").mkdir(parents=True)
        (fake_sysfs / "block" / "sda" / "queue").mkdir()
        (fake_sysfs / "block" / "sda" / "queue" / "rotational").write_text(flag, encoding="utf-8")
        assert system_info._linux_rotational(major=8, minor=1) is expected

    def test_linux_whole_disk_reads_its_own_flag(self, fake_sysfs: Path) -> None:
        """A device-mapper or unpartitioned disk carries the flag itself."""
        _write_block_device(fake_sysfs, number="253_0", device_name="dm-0", device_type="disk")
        (fake_sysfs / "block" / "dm-0" / "queue").mkdir(parents=True)
        (fake_sysfs / "block" / "dm-0" / "queue" / "rotational").write_text("0\n", encoding="utf-8")
        assert system_info._linux_rotational(major=253, minor=0) is False

    @pytest.mark.usefixtures("fake_sysfs")
    def test_linux_anonymous_device_is_undetermined(self) -> None:
        """A filesystem with no block device entry (btrfs subvolume, overlay, tmpfs) reports None."""
        assert system_info._linux_rotational(major=0, minor=42) is None

    def test_linux_partition_without_parent_is_undetermined(self, fake_sysfs: Path) -> None:
        """A partition whose disk cannot be found reports None rather than guessing."""
        _write_block_device(fake_sysfs, number="8_1", device_name="sda1", device_type="partition")
        (fake_sysfs / "block").mkdir()
        assert system_info._linux_rotational(major=8, minor=1) is None

    def test_unsupported_platform_is_undetermined(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """MacOS and other platforms report None."""
        monkeypatch.setattr(system_info.sys, "platform", "darwin")
        assert rotational_storage(tmp_path) is None

    @pytest.mark.skipif(sys.platform != "win32", reason="a drive letter only parses as one on Windows")
    @pytest.mark.parametrize(
        ("stdout", "expected"), [("HDD\r\n", True), ("SSD\r\n", False), ("Unspecified\r\n", None), ("", None)]
    )
    def test_windows_media_type_mapping(
        self, monkeypatch: pytest.MonkeyPatch, stdout: str, expected: bool | None
    ) -> None:
        """The probe returns the physical disk's media type, or None when the type is unknown or missing."""

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(system_info.subprocess, "run", fake_run)
        assert system_info._windows_rotational(Path("C:/models")) is expected

    @pytest.mark.skipif(sys.platform != "win32", reason="a drive letter only parses as one on Windows")
    def test_windows_probe_timeout_is_undetermined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A PowerShell call that exceeds its bound reports None instead of raising."""

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(command, 10.0)

        monkeypatch.setattr(system_info.subprocess, "run", fake_run)
        assert system_info._windows_rotational(Path("C:/models")) is None

    def test_windows_unc_path_is_undetermined(self) -> None:
        """A network share has no drive letter to map to a physical disk."""
        assert system_info._windows_rotational(Path("//server/share/models")) is None


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
