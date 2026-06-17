"""Tests for backend-agnostic hardware detection in ``SystemResources.detect``.

These assert that device discovery flows through hordelib's ``enumerate_accelerators`` (which
covers every ComfyUI backend) rather than ``torch.cuda`` directly, so non-NVIDIA backends -
including a CPU-only machine - still yield a populated device map. The hordelib call is mocked so
the tests need no GPU, no network, and no installed accelerator backend.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from horde_worker_regen.process_management.process_manager import SystemResources

_MB = 1024 * 1024


class _FakeAccelerator:
    """Stand-in for ``hordelib.api.AcceleratorInfo`` (only the fields ``detect`` reads)."""

    def __init__(self, *, index: int, name: str, total_vram_mb: int) -> None:
        self.index = index
        self.name = name
        self.total_vram_mb = total_vram_mb


@pytest.fixture
def fake_hordelib_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_FakeAccelerator]]:
    """Install a fake ``hordelib.api`` whose ``enumerate_accelerators`` returns a mutable list."""
    accelerators: list[_FakeAccelerator] = []
    fake_module = types.ModuleType("hordelib.api")
    fake_module.enumerate_accelerators = lambda: list(accelerators)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hordelib.api", fake_module)
    yield accelerators


def test_detect_maps_multiple_devices(fake_hordelib_api: list[_FakeAccelerator]) -> None:
    """Every enumerated accelerator becomes a TorchDeviceInfo keyed by its index, MB converted to bytes."""
    fake_hordelib_api.extend(
        [
            _FakeAccelerator(index=0, name="NVIDIA RTX 4090", total_vram_mb=24564),
            _FakeAccelerator(index=1, name="NVIDIA RTX 3090", total_vram_mb=24576),
        ],
    )

    resources = SystemResources.detect()

    assert set(resources.device_map.root) == {0, 1}
    assert resources.device_map.root[0].device_name == "NVIDIA RTX 4090"
    # MB are converted back to bytes for the TorchDeviceInfo contract.
    assert resources.device_map.root[0].total_memory == 24564 * _MB
    assert resources.device_map.root[1].device_index == 1


def test_detect_yields_cpu_pseudo_device_without_gpu(fake_hordelib_api: list[_FakeAccelerator]) -> None:
    """A CPU-only machine must still produce a device, where a bare torch.cuda loop would yield none."""
    fake_hordelib_api.append(_FakeAccelerator(index=0, name="CPU", total_vram_mb=65455))

    resources = SystemResources.detect()

    assert list(resources.device_map.root) == [0]
    assert resources.device_map.root[0].device_name == "CPU"
    assert resources.total_ram_bytes > 0
