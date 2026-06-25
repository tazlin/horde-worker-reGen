"""Tests for device-map selection by gpu_device_indices (Phase A2 of multi-GPU)."""

from __future__ import annotations

from horde_worker_regen.process_management.process_manager import _select_driven_devices
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap


def _device_map(*indices: int) -> TorchDeviceMap:
    """Build a device map with one entry per given index."""
    return TorchDeviceMap(
        root={
            index: TorchDeviceInfo(
                device_name=f"GPU{index}",
                device_index=index,
                total_memory=24 * 1024 * 1024 * 1024,
            )
            for index in indices
        },
    )


def test_none_indices_drives_all() -> None:
    """No gpu_device_indices means auto-detect: every detected device is driven."""
    detected = _device_map(0, 1, 2)
    selected = _select_driven_devices(detected, None)
    assert sorted(selected.root) == [0, 1, 2]


def test_explicit_subset_is_selected() -> None:
    """An explicit index list opts the worker into exactly that subset, in order."""
    detected = _device_map(0, 1, 2)
    selected = _select_driven_devices(detected, [0, 2])
    assert sorted(selected.root) == [0, 2]


def test_missing_indices_are_ignored() -> None:
    """A requested index that is not present is dropped (the present ones still select)."""
    detected = _device_map(0, 1)
    selected = _select_driven_devices(detected, [1, 5])
    assert sorted(selected.root) == [1]


def test_no_match_falls_back_to_all() -> None:
    """A list that matches nothing falls back to all detected devices rather than zero cards."""
    detected = _device_map(0, 1)
    selected = _select_driven_devices(detected, [7, 8])
    assert sorted(selected.root) == [0, 1]


def test_empty_device_map_unchanged() -> None:
    """No detected accelerators (CPU/dry-run) stays empty regardless of configured indices."""
    detected = TorchDeviceMap(root={})
    assert _select_driven_devices(detected, [0]).root == {}
