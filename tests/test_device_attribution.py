"""Tests for the per-device attribution plumbing (Phase A3, increment 1: device_index + kind)."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import HordeProcessMemoryMessage
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessState, HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_manager import _select_driven_devices
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo
from horde_worker_regen.utils.accelerator_probe import ProbedAccelerator


def test_torch_device_info_kind_default_and_explicit() -> None:
    """TorchDeviceInfo defaults its backend to cuda but preserves an explicit kind."""
    assert TorchDeviceInfo(device_name="g", device_index=0, total_memory=1).kind == "cuda"
    rocm = TorchDeviceInfo(device_name="g", device_index=1, total_memory=1, kind="rocm")
    assert rocm.kind == "rocm"


def test_probed_accelerator_kind_default() -> None:
    """A probe entry without a kind (older serialisation) defaults to cuda."""
    assert ProbedAccelerator(index=0, name="g", total_vram_mb=8192).kind == "cuda"
    assert ProbedAccelerator(index=0, name="g", total_vram_mb=8192, kind="xpu").kind == "xpu"


def test_select_driven_devices_preserves_kind() -> None:
    """Filtering the device map to a subset keeps each kept device's backend kind."""
    from horde_worker_regen.process_management.resources.device_info import TorchDeviceMap

    detected = TorchDeviceMap(
        root={
            0: TorchDeviceInfo(device_name="a", device_index=0, total_memory=1, kind="cuda"),
            1: TorchDeviceInfo(device_name="b", device_index=1, total_memory=1, kind="rocm"),
        },
    )
    selected = _select_driven_devices(detected, [1])
    assert selected.root[1].kind == "rocm"


def test_memory_message_device_index_default_and_explicit() -> None:
    """A memory message defaults its device index to 0 but carries an explicit one."""
    base = HordeProcessMemoryMessage(process_id=0, process_launch_identifier=0, info="m", ram_usage_bytes=1)
    assert base.device_index == 0
    pinned = HordeProcessMemoryMessage(
        process_id=0,
        process_launch_identifier=0,
        info="m",
        ram_usage_bytes=1,
        device_index=2,
    )
    assert pinned.device_index == 2


def test_process_info_carries_device_index() -> None:
    """A HordeProcessInfo records the device it was spawned on (default 0)."""
    mp_process = Mock()
    mp_process.pid = 1234
    info = HordeProcessInfo(
        mp_process=mp_process,
        pipe_connection=Mock(),
        process_id=0,
        process_type=HordeProcessType.INFERENCE,
        last_process_state=HordeProcessState.PROCESS_STARTING,
        process_launch_identifier=0,
        device_index=3,
    )
    assert info.device_index == 3
