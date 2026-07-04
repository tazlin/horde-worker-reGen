"""Tests for the allocator-enforced per-process VRAM quota helper.

The quota is an optimization applied inside GPU child processes; its contract is graceful absence:
on non-CUDA backends, missing torch, or any error it returns False and never raises, so XPU/DirectML/
CPU deployments and Linux hosts are unaffected.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from horde_worker_regen.utils.vram_quota import apply_process_vram_quota_mb


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
