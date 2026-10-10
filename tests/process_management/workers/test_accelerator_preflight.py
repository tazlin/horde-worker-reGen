"""The inference child's start check that the accelerator accepts a tensor before the slot reports ready."""

from __future__ import annotations

import queue
import sys
import types
from unittest.mock import Mock

import pytest
import torch

from horde_worker_regen.process_management.ipc.messages import HordeProcessState, HordeProcessStateChangeMessage
from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess

_DRIVER_FAULT = (
    'NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED at "PeerToPeerAccess.cpp":83, '
    "please report a bug to PyTorch."
)


def _child(monkeypatch: pytest.MonkeyPatch, *, device: torch.device) -> HordeInferenceProcess:
    """Return a child built without its backend, whose execution device is ``device``."""
    backend = types.ModuleType("hordelib.comfy_horde")
    backend.get_torch_device = lambda: device  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hordelib.comfy_horde", backend)
    proc = object.__new__(HordeInferenceProcess)
    proc.process_id = 3
    proc.process_launch_identifier = 0
    proc.process_message_queue = queue.Queue()  # pyrefly: ignore
    return proc


def _sent_states(proc: HordeInferenceProcess) -> list[HordeProcessStateChangeMessage]:
    """Return every process-state message the child has queued for the parent."""
    sent = []
    while not proc.process_message_queue.empty():
        sent.append(proc.process_message_queue.get_nowait())
    return [message for message in sent if isinstance(message, HordeProcessStateChangeMessage)]


def test_a_device_that_refuses_the_move_is_reported_and_ends_the_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device that raises on the first tensor moved to it is reported as unusable before the child exits."""
    proc = _child(monkeypatch, device=torch.device("cuda", 0))
    refused = Mock()
    refused.to.side_effect = RuntimeError(_DRIVER_FAULT)
    monkeypatch.setattr(torch, "zeros", lambda *_args, **_kwargs: refused)

    with pytest.raises(SystemExit):
        proc._verify_accelerator_accepts_work()

    states = _sent_states(proc)
    assert [message.process_state for message in states] == [HordeProcessState.ACCELERATOR_UNUSABLE]
    assert "nvmlInit_v2_" in states[0].info
    assert "Reboot the host" in states[0].info


def test_a_device_that_takes_the_move_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A working device is left as it was: no message, no exit, and the move was made to that device."""
    device = torch.device("cuda", 0)
    proc = _child(monkeypatch, device=device)
    accepted = Mock()
    monkeypatch.setattr(torch, "zeros", lambda *_args, **_kwargs: accepted)

    proc._verify_accelerator_accepts_work()

    accepted.to.assert_called_once_with(device)
    assert _sent_states(proc) == []


def test_a_cpu_device_is_not_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU execution device has no driver to refuse the move, so nothing is allocated for the check."""
    proc = _child(monkeypatch, device=torch.device("cpu"))
    allocate = Mock()
    monkeypatch.setattr(torch, "zeros", allocate)

    proc._verify_accelerator_accepts_work()

    allocate.assert_not_called()
    assert _sent_states(proc) == []


def test_a_device_that_cannot_be_resolved_does_not_stop_the_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error finding the execution device is ignored, since the check must not stop a working process."""
    proc = _child(monkeypatch, device=torch.device("cpu"))

    def unresolvable() -> torch.device:
        raise RuntimeError("no backend")

    monkeypatch.setattr(sys.modules["hordelib.comfy_horde"], "get_torch_device", unresolvable)

    proc._verify_accelerator_accepts_work()

    assert _sent_states(proc) == []
