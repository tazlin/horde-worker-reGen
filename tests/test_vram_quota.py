"""Tests for the allocator-enforced per-process VRAM quota helper.

The quota is an optimization applied inside GPU child processes; its contract is graceful absence:
on non-CUDA backends, missing torch, or any error it returns False and never raises, so XPU/DirectML/
CPU deployments and Linux hosts are unaffected.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from horde_worker_regen.utils.vram_quota import (
    POST_PROCESS_VRAM_QUOTA_CEILING_MB,
    POST_PROCESS_VRAM_QUOTA_FLOOR_MB,
    SAFETY_VRAM_QUOTA_CEILING_MB,
    SAFETY_VRAM_QUOTA_FLOOR_MB,
    apply_post_process_vram_quota,
    apply_process_vram_quota_mb,
    apply_safety_vram_quota,
    effective_post_process_vram_quota_mb,
    effective_safety_vram_quota_mb,
)


def _fake_torch(*, cuda_available: bool, total_memory_bytes: int = 16 * 1024**3) -> SimpleNamespace:
    cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_properties=lambda _index: SimpleNamespace(total_memory=total_memory_bytes),
        set_per_process_memory_fraction=Mock(),
    )
    return SimpleNamespace(cuda=cuda)


def test_caps_allocator_at_quota_fraction(monkeypatch) -> None:  # noqa: ANN001
    """The cap is the quota's fraction of the device total, applied to the pinned device."""
    fake = _fake_torch(cuda_available=True, total_memory_bytes=16 * 1024**3)
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert apply_process_vram_quota_mb(4096.0, device_index=0) is True
    fraction, device = fake.cuda.set_per_process_memory_fraction.call_args.args
    assert abs(fraction - 0.25) < 1e-6
    assert device == 0


def test_quota_larger_than_device_clamps_to_full_card(monkeypatch) -> None:  # noqa: ANN001
    """A quota above the device total is clamped to 1.0 rather than rejected."""
    fake = _fake_torch(cuda_available=True, total_memory_bytes=2 * 1024**3)
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert apply_process_vram_quota_mb(4096.0) is True
    fraction, _device = fake.cuda.set_per_process_memory_fraction.call_args.args
    assert fraction == 1.0


def test_non_cuda_backend_is_a_noop(monkeypatch) -> None:  # noqa: ANN001
    """XPU/DirectML/CPU backends (CUDA unavailable) decline the quota without raising."""
    fake = _fake_torch(cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert apply_process_vram_quota_mb(4096.0) is False
    fake.cuda.set_per_process_memory_fraction.assert_not_called()


def test_allocator_error_is_swallowed(monkeypatch) -> None:  # noqa: ANN001
    """Any failure applying the cap degrades to running unquota'd, never a startup crash."""
    fake = _fake_torch(cuda_available=True)
    fake.cuda.set_per_process_memory_fraction = Mock(side_effect=RuntimeError("driver said no"))
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert apply_process_vram_quota_mb(4096.0) is False


class TestEffectivePostProcessQuota:
    """The lane's guard is sized to the card: above the largest realistic chain where the card allows."""

    @pytest.mark.parametrize(
        ("total_mb", "expected_mb"),
        [
            (None, POST_PROCESS_VRAM_QUOTA_FLOOR_MB),  # no card reading -> floor
            (0.0, POST_PROCESS_VRAM_QUOTA_FLOOR_MB),  # nonsense reading -> floor
            (
                2048.0,
                POST_PROCESS_VRAM_QUOTA_FLOOR_MB,
            ),  # card smaller than the floor -> floor (clamps to card on apply)
            (8192.0, 4096.0),  # 8GB card: 8192 - 4096 headroom = 4096 (== floor)
            (10240.0, 6144.0),  # 10GB card: 10240 - 4096 = 6144, below the ceiling
            (16384.0, POST_PROCESS_VRAM_QUOTA_CEILING_MB),  # 16GB card: guard clamps to the ceiling
            (24576.0, POST_PROCESS_VRAM_QUOTA_CEILING_MB),  # 24GB card: guard clamps to the ceiling
        ],
    )
    def test_guard_scales_with_card(self, total_mb: float | None, expected_mb: float) -> None:
        """The guard never dips below the floor nor rises above the ceiling, leaving inference headroom."""
        assert effective_post_process_vram_quota_mb(total_mb) == expected_mb

    def test_guard_covers_the_largest_observed_chain_on_a_headroom_card(self) -> None:
        """A card with headroom hosts an upscale/face-fix peak (~6.4GB) that the old fixed 4GB cap faulted."""
        assert effective_post_process_vram_quota_mb(24576.0) >= 6429.0


def test_apply_post_process_quota_sizes_fraction_to_card(monkeypatch) -> None:  # noqa: ANN001
    """The applied fraction is the card-sized guard over the device total, not a fixed 4GB."""
    fake = _fake_torch(cuda_available=True, total_memory_bytes=24 * 1024**3)
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert apply_post_process_vram_quota(device_index=0) is True
    fraction, device = fake.cuda.set_per_process_memory_fraction.call_args.args
    # 24GB card -> ceiling guard 8192MB -> 8192 / 24576 = 0.333...
    assert abs(fraction - (POST_PROCESS_VRAM_QUOTA_CEILING_MB / (24 * 1024))) < 1e-6
    assert device == 0


def test_apply_post_process_quota_is_a_noop_without_cuda(monkeypatch) -> None:  # noqa: ANN001
    """On a non-CUDA backend the lane runs unquota'd rather than raising."""
    fake = _fake_torch(cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert apply_post_process_vram_quota() is False
    fake.cuda.set_per_process_memory_fraction.assert_not_called()


class TestEffectiveSafetyQuota:
    """The safety guard grows on roomy cards so CLIP alchemy is not capped at the startup floor."""

    @pytest.mark.parametrize(
        ("total_mb", "expected_mb"),
        [
            (None, SAFETY_VRAM_QUOTA_FLOOR_MB),
            (0.0, SAFETY_VRAM_QUOTA_FLOOR_MB),
            (8192.0, SAFETY_VRAM_QUOTA_FLOOR_MB),
            (10240.0, SAFETY_VRAM_QUOTA_FLOOR_MB),
            (12288.0, SAFETY_VRAM_QUOTA_CEILING_MB),
            (16384.0, SAFETY_VRAM_QUOTA_CEILING_MB),
            (24576.0, SAFETY_VRAM_QUOTA_CEILING_MB),
        ],
    )
    def test_guard_scales_with_card(self, total_mb: float | None, expected_mb: float) -> None:
        """The safety cap preserves small-card protection and expands on cards with room."""
        assert effective_safety_vram_quota_mb(total_mb) == expected_mb


def test_apply_safety_quota_sizes_fraction_to_card(monkeypatch) -> None:  # noqa: ANN001
    """The safety process gets the card-sized guard rather than the fixed 4GB floor on a 24GB card."""
    fake = _fake_torch(cuda_available=True, total_memory_bytes=24 * 1024**3)
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert apply_safety_vram_quota(device_index=0) is True
    fraction, device = fake.cuda.set_per_process_memory_fraction.call_args.args
    assert abs(fraction - (SAFETY_VRAM_QUOTA_CEILING_MB / (24 * 1024))) < 1e-6
    assert device == 0
