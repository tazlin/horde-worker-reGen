"""Unit tests for the LoRA cache disk-space fail-safes (``lora_disk_guard``).

These exercise the constrain/evict logic and the main-process exhaustion verdict in isolation, with a
fake LoRA manager and a simulated volume, so there is no hordelib, no GPU, and no real disk pressure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from horde_worker_regen.process_management import lora_disk_guard
from horde_worker_regen.process_management.lora_disk_guard import (
    configured_lora_budget_mb_from_env,
    constrain_lora_cache_to_disk,
    is_lora_disk_exhausted,
    lora_disk_floor_mb_from_env,
    read_evictable_adhoc_mb,
)


class FakeLoraManager:
    """A structural stand-in for the hordelib LoRA manager's disk-guard surface."""

    def __init__(self, adhoc_sizes_mb: list[float], max_adhoc_disk: int) -> None:
        """Seed the fake with ad-hoc entry sizes (oldest first) and a starting byte budget."""
        self.model_folder_path = Path("/fake/lora")
        self.max_adhoc_disk = max_adhoc_disk
        # Oldest first, so delete_oldest pops index 0 (least-recently-used).
        self._entries = list(adhoc_sizes_mb)
        self.save_count = 0

    def calculate_adhoc_cache(self) -> float:
        """Return the total size of the remaining ad-hoc entries, in megabytes."""
        return sum(self._entries)

    def find_oldest_adhoc_entry(self) -> object | None:
        """Return a truthy marker for the oldest entry, or ``None`` when none remain."""
        return self._entries[0] if self._entries else None

    def delete_oldest(self) -> None:
        """Evict the least-recently-used ad-hoc entry."""
        if self._entries:
            self._entries.pop(0)

    def save_reference_to_disk(self) -> None:
        """Count persistence calls so tests can assert eviction was saved exactly once."""
        self.save_count += 1


def _simulate_volume(monkeypatch: pytest.MonkeyPatch, manager: FakeLoraManager, base_free_mb: float) -> None:
    """Patch the guard's free-space probe so deleting ad-hoc entries frees their bytes back."""
    initial_total = manager.calculate_adhoc_cache()
    monkeypatch.setattr(
        lora_disk_guard,
        "free_mb",
        lambda path: base_free_mb + (initial_total - manager.calculate_adhoc_cache()),
    )


class TestConstrainLoraCacheToDisk:
    """The inference-side constrain/evict pass."""

    def test_evicts_until_floor_cleared(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Below the floor, the oldest ad-hoc LoRAs are evicted until free space clears it."""
        manager = FakeLoraManager(adhoc_sizes_mb=[200, 200, 200, 200, 200], max_adhoc_disk=10240)
        _simulate_volume(monkeypatch, manager, base_free_mb=600.0)  # 600 < 1024 floor

        result = constrain_lora_cache_to_disk(manager, floor_mb=1024, configured_budget_mb=10240)

        assert result.solved is True
        assert result.evicted_count >= 1
        assert (result.free_mb_after or 0) >= 1024
        assert manager.save_count == 1  # persisted once after eviction
        # Budget was constrained below the configured value while the disk was tight.
        assert result.budget_mb_after <= 10240

    def test_unsolvable_when_adhoc_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When evicting every ad-hoc entry still can't clear the floor, the pass reports failure."""
        manager = FakeLoraManager(adhoc_sizes_mb=[50, 50], max_adhoc_disk=10240)
        _simulate_volume(monkeypatch, manager, base_free_mb=200.0)  # 200 + 100 < 1024 floor

        result = constrain_lora_cache_to_disk(manager, floor_mb=1024, configured_budget_mb=10240)

        assert result.solved is False
        assert manager.calculate_adhoc_cache() == 0  # tried everything
        assert result.free_mb_after == pytest.approx(300.0)

    def test_restores_budget_when_healthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ample free space restores a previously shrunk budget to its configured value."""
        manager = FakeLoraManager(adhoc_sizes_mb=[200], max_adhoc_disk=512)  # previously shrunk to 512
        monkeypatch.setattr(lora_disk_guard, "free_mb", lambda path: 50_000.0)  # plenty of room

        result = constrain_lora_cache_to_disk(manager, floor_mb=1024, configured_budget_mb=10240)

        assert result.solved is True
        assert result.evicted_count == 0
        assert manager.max_adhoc_disk == 10240  # restored to configured
        assert manager.save_count == 0  # nothing evicted, nothing to persist

    def test_floor_disabled_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A floor of 0 disables the guard entirely, even on a nearly full volume."""
        manager = FakeLoraManager(adhoc_sizes_mb=[200, 200], max_adhoc_disk=10240)
        monkeypatch.setattr(lora_disk_guard, "free_mb", lambda path: 10.0)

        result = constrain_lora_cache_to_disk(manager, floor_mb=0, configured_budget_mb=10240)

        assert result.acted is False
        assert result.evicted_count == 0


class TestExhaustionVerdict:
    """The main-process decision to stop advertising LoRA support."""

    def test_not_exhausted_above_floor(self) -> None:
        """Free space above the floor is never exhausted."""
        assert is_lora_disk_exhausted(free_mb_value=2000, floor_mb=1024, evictable_adhoc_mb=0) is False

    def test_not_exhausted_when_eviction_would_clear(self) -> None:
        """A shortfall that eviction could clear is recoverable, so LoRAs stay offered."""
        # 600 free + 5000 evictable ad-hoc > 1024 floor: recoverable.
        assert is_lora_disk_exhausted(free_mb_value=600, floor_mb=1024, evictable_adhoc_mb=5000) is False

    def test_exhausted_when_eviction_cannot_clear(self) -> None:
        """A structural shortfall (eviction can't help) disables LoRAs."""
        # 600 free + 100 evictable < 1024 floor.
        assert is_lora_disk_exhausted(free_mb_value=600, floor_mb=1024, evictable_adhoc_mb=100) is True

    def test_unsampleable_is_not_exhausted(self) -> None:
        """An unsampleable volume is treated as not exhausted (no false positives)."""
        assert is_lora_disk_exhausted(free_mb_value=None, floor_mb=1024, evictable_adhoc_mb=0) is False

    def test_floor_disabled_is_not_exhausted(self) -> None:
        """A floor of 0 never marks the disk exhausted."""
        assert is_lora_disk_exhausted(free_mb_value=10, floor_mb=0, evictable_adhoc_mb=0) is False


class TestReadEvictableAdhocMb:
    """Reading the evictable ad-hoc footprint from the persisted reference."""

    def test_sums_only_adhoc_versions(self, tmp_path: Path) -> None:
        """Only ad-hoc versions count toward the evictable total; default-set LoRAs are excluded."""
        reference = {
            "lora_a": {"versions": {"1": {"size_mb": 200, "adhoc": True}, "2": {"size_mb": 50, "adhoc": False}}},
            "lora_b": {"versions": {"3": {"size_mb": 144, "adhoc": True}}},
        }
        path = tmp_path / "lora.json"
        path.write_text(json.dumps(reference), encoding="utf-8")

        assert read_evictable_adhoc_mb(path) == pytest.approx(344.0)  # 200 + 144, not the default 50

    def test_missing_file_is_zero(self, tmp_path: Path) -> None:
        """A missing reference yields zero evictable megabytes."""
        assert read_evictable_adhoc_mb(tmp_path / "absent.json") == 0.0

    def test_malformed_file_is_zero(self, tmp_path: Path) -> None:
        """A corrupt reference yields zero evictable megabytes rather than raising."""
        path = tmp_path / "lora.json"
        path.write_text("{not json", encoding="utf-8")
        assert read_evictable_adhoc_mb(path) == 0.0


class TestEnvHelpers:
    """The env-derived floor and budget readers shared by both processes."""

    def test_floor_from_env_default(self) -> None:
        """An unset floor env var falls back to the 1024 MB default."""
        assert lora_disk_floor_mb_from_env(lambda key: None) == 1024.0

    def test_floor_from_env_value(self) -> None:
        """A set floor env var is parsed to megabytes."""
        assert lora_disk_floor_mb_from_env({"AIWORKER_LORA_MIN_DISK_FREE_MB": "2048"}.get) == 2048.0

    def test_floor_from_env_invalid_falls_back(self) -> None:
        """An unparseable floor env var falls back to the default."""
        assert lora_disk_floor_mb_from_env({"AIWORKER_LORA_MIN_DISK_FREE_MB": "abc"}.get) == 1024.0

    def test_budget_from_env_default(self) -> None:
        """An unset budget env var falls back to the 10 GB default (in MB)."""
        assert configured_lora_budget_mb_from_env(lambda key: None) == 10 * 1024

    def test_budget_from_env_value(self) -> None:
        """A set budget env var is parsed to megabytes."""
        assert configured_lora_budget_mb_from_env({"AIWORKER_LORA_CACHE_SIZE": "20480"}.get) == 20480
