"""Production control boundaries for capacity increases on a drained worker."""

from __future__ import annotations

import multiprocessing
from dataclasses import replace
from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.inference_gate import ResizableInferenceGate
from horde_worker_regen.process_management.ipc.supervisor_channel import SupervisorCommand, SupervisorControlMessage
from tests.process_management.conftest import make_testable_process_manager


def test_capacity_change_requires_pause_and_returns_correlated_refusal() -> None:
    """An active intake must keep its original gate and concurrency ceiling."""
    manager = make_testable_process_manager()
    manager._state.supervisor_paused = False
    original = manager._runtime_config.max_threads_ceiling
    manager._apply_resize_capacity(
        SupervisorControlMessage(
            command=SupervisorCommand.RESIZE_CAPACITY,
            device_index=0,
            target_threads=2,
            request_id="request",
        )
    )
    assert manager._capacity_change is not None
    assert not manager._capacity_change.accepted
    assert manager._capacity_change.request_id == "request"
    assert manager._runtime_config.max_threads_ceiling == original


def test_paused_capacity_growth_uses_normal_lifecycle_and_preserves_existing_gate() -> None:
    """A drained change updates shared capacity and uses existing process scaling."""
    manager = make_testable_process_manager()
    manager._state.supervisor_paused = True
    gate = ResizableInferenceGate(1, ctx=multiprocessing.get_context("spawn"))
    manager._card_runtimes[0] = replace(manager._card_runtimes[0], inference_semaphore=gate, total_vram_mb=24576)
    manager._process_lifecycle.scale_inference_processes = Mock(return_value=3)
    manager._apply_resize_capacity(
        SupervisorControlMessage(
            command=SupervisorCommand.RESIZE_CAPACITY,
            device_index=0,
            target_threads=2,
        )
    )
    assert manager._capacity_change is not None and manager._capacity_change.accepted
    assert manager._card_runtimes[0].inference_semaphore is gate
    assert gate.capacity == 2
    assert manager._runtime_config.max_threads_ceiling >= 2
    assert gate.acquire(block=False) and gate.acquire(block=False)
    assert not gate.acquire(block=False)
    gate.release()
    gate.release()
    manager._process_lifecycle.scale_inference_processes.assert_called_once()


def test_held_permit_rejects_change_and_work_can_continue() -> None:
    """A refused change must leave the current capacity usable after the holder finishes."""
    manager = make_testable_process_manager()
    manager._state.supervisor_paused = True
    gate = ResizableInferenceGate(1, ctx=multiprocessing.get_context("spawn"))
    manager._card_runtimes[0] = replace(manager._card_runtimes[0], inference_semaphore=gate)
    gate.acquire()
    manager._apply_resize_capacity(
        SupervisorControlMessage(
            command=SupervisorCommand.RESIZE_CAPACITY,
            device_index=0,
            target_threads=3,
        )
    )
    assert manager._capacity_change is not None and not manager._capacity_change.accepted
    assert gate.capacity == 1
    gate.release()
    assert gate.acquire(block=False)
    gate.release()
