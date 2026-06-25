"""Tests for backend-agnostic hardware detection in ``SystemResources.detect``.

These assert that device discovery flows through the out-of-process accelerator probe (which itself uses
hordelib's backend-agnostic ``enumerate_accelerators``, covering every ComfyUI backend) rather than
``torch.cuda`` directly, so non-NVIDIA backends - including a CPU-only machine - still yield a populated
device map. The probe is mocked so the tests need no GPU, no network, and no subprocess: enumeration runs
out-of-process precisely to keep the orchestrator torch-free (see ``test_orchestrator_torch_free.py``).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

import horde_worker_regen.utils.accelerator_probe as accelerator_probe_module
from horde_worker_regen.process_management.process_manager import SystemResources
from horde_worker_regen.utils.accelerator_probe import ProbedAccelerator, probe_accelerators

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


def test_detect_carries_overhead_and_marginal(fake_probe: list[ProbedAccelerator]) -> None:
    """detect() surfaces the probe's first-context overhead and the (smaller) per-additional-context marginal.

    These are the two figures the streaming forecast needs: the overhead sizes free-if-alone, the marginal
    sizes free-after-model-evict. Across multiple devices it takes the max of each (single-GPU is the norm;
    the max is the conservative choice).
    """
    fake_probe.extend(
        [
            ProbedAccelerator(
                index=0, name="GPU0", total_vram_mb=24564, runtime_overhead_mb=4112, marginal_overhead_mb=455,
            ),
            ProbedAccelerator(
                index=1, name="GPU1", total_vram_mb=24564, runtime_overhead_mb=3000, marginal_overhead_mb=480,
            ),
        ],
    )

    resources = SystemResources.detect()

    assert resources.per_process_overhead_mb == 4112
    assert resources.marginal_process_overhead_mb == 480


def test_detect_marginal_defaults_zero_for_old_probe(fake_probe: list[ProbedAccelerator]) -> None:
    """A probe result without the marginal (older serialisation) leaves it 0 -> forecast falls back."""
    fake_probe.append(ProbedAccelerator(index=0, name="GPU0", total_vram_mb=16375, runtime_overhead_mb=1288))

    resources = SystemResources.detect()

    assert resources.per_process_overhead_mb == 1288
    assert resources.marginal_process_overhead_mb == 0


@pytest.mark.gpu
def test_probe_measures_overhead_and_marginal_on_real_device() -> None:
    """On real hardware the probe reports a positive first-context overhead and a sane marginal.

    The marginal is measured by bringing up a second process and reading the device-wide used delta. That is
    only visible cross-process where the platform reports true device-wide VRAM: Linux does, Windows WDDM does
    not (a process cannot see a sibling's allocation), so there the marginal degrades to 0 and the worker
    falls back to charging the full overhead per context. The non-Windows assertion is the real validation:
    the marginal is positive and clearly smaller than the one-time-inclusive overhead.
    """
    accelerators = probe_accelerators(timeout_seconds=240)
    assert accelerators, "probe found no accelerators on a GPU box"
    primary = accelerators[0]
    assert primary.total_vram_mb > 0
    assert primary.runtime_overhead_mb > 0, "a fresh process must show some context/runtime VRAM"
    assert primary.marginal_overhead_mb >= 0

    if sys.platform != "win32":
        assert primary.marginal_overhead_mb > 0, (
            "on Linux the device-wide second-context delta must be measurable; "
            "a 0 here means the marginal measurement regressed"
        )
        assert primary.marginal_overhead_mb < primary.runtime_overhead_mb, (
            "the per-additional-context marginal must be smaller than the one-time-inclusive first-context "
            f"overhead (got marginal={primary.marginal_overhead_mb} >= overhead={primary.runtime_overhead_mb})"
        )
