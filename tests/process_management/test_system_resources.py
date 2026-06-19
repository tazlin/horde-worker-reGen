"""Tests for backend-agnostic hardware detection in ``SystemResources.detect``.

These assert that device discovery flows through the out-of-process accelerator probe (which itself uses
hordelib's backend-agnostic ``enumerate_accelerators``, covering every ComfyUI backend) rather than
``torch.cuda`` directly, so non-NVIDIA backends - including a CPU-only machine - still yield a populated
device map. The probe is mocked so the tests need no GPU, no network, and no subprocess: enumeration runs
out-of-process precisely to keep the orchestrator torch-free (see ``test_orchestrator_torch_free.py``).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import horde_worker_regen.utils.accelerator_probe as accelerator_probe_module
from horde_worker_regen.process_management.process_manager import SystemResources
from horde_worker_regen.utils.accelerator_probe import ProbedAccelerator

_MB = 1024 * 1024


@pytest.fixture
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[ProbedAccelerator]]:
    """Patch ``probe_accelerators`` to return a mutable list, so ``detect`` runs without a real subprocess."""
    accelerators: list[ProbedAccelerator] = []
    monkeypatch.setattr(
        accelerator_probe_module,
        "probe_accelerators",
        lambda **_kwargs: list(accelerators),
    )
    yield accelerators


def test_detect_maps_multiple_devices(fake_probe: list[ProbedAccelerator]) -> None:
    """Every probed accelerator becomes a TorchDeviceInfo keyed by its index, MB converted to bytes."""
    fake_probe.extend(
        [
            ProbedAccelerator(index=0, name="NVIDIA RTX 4090", total_vram_mb=24564),
            ProbedAccelerator(index=1, name="NVIDIA RTX 3090", total_vram_mb=24576),
        ],
    )

    resources = SystemResources.detect()

    assert set(resources.device_map.root) == {0, 1}
    assert resources.device_map.root[0].device_name == "NVIDIA RTX 4090"
    # MB are converted back to bytes for the TorchDeviceInfo contract.
    assert resources.device_map.root[0].total_memory == 24564 * _MB
    assert resources.device_map.root[1].device_index == 1


def test_detect_yields_cpu_pseudo_device_without_gpu(fake_probe: list[ProbedAccelerator]) -> None:
    """A CPU-only machine must still produce a device, where a bare torch.cuda loop would yield none."""
    fake_probe.append(ProbedAccelerator(index=0, name="CPU", total_vram_mb=65455))

    resources = SystemResources.detect()

    assert list(resources.device_map.root) == [0]
    assert resources.device_map.root[0].device_name == "CPU"
    assert resources.total_ram_bytes > 0
