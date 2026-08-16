"""Pins how an inference child reports what the device did with its weights, rather than what it was asked.

Two directions of the same discipline. A retention grant is a parent-side prediction the backend can break by
freeing the checkpoint mid-run; an unload is a parent-side command the backend can decline by leaving the
weights loaded behind a live reference. In both the parent's record is right only if the child reports the
device rather than the request.

A retention grant is a parent-side prediction made at dispatch: the slot keeps its checkpoint on the
device so the next same-model job skips the upload. ComfyUI can still free every other model on the card
to fund an allocation during the run, taking the granted checkpoint with it, and nothing outside the
process can observe that. The engine reports it on the result; what these pin is the child turning that
report into the state change the parent already understands for a VRAM unload, so the existing handler
drops the record rather than a new message type having to be taught to every consumer.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import (
    HordeModelStateChangeMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess


class _FakeQueue:
    """A minimal stand-in for the process message queue that records what the child sends."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        """Record a message the child sent to the parent."""
        self.messages.append(message)


def _child(queue: _FakeQueue, *, evicted: bool, active_model: str | None = "sdxl-checkpoint") -> HordeInferenceProcess:
    """An inference child with only the state the eviction report reads, and no backend.

    Constructing the real process would stand up hordelib, a model manager and a device; the report is a
    pure message-emitting path over four fields, so it is exercised against those directly. The memory
    report is stubbed because it reads the device.
    """
    child = object.__new__(HordeInferenceProcess)
    child.process_id = 3
    child.process_launch_identifier = 1
    child.process_message_queue = queue  # type: ignore[assignment]
    child._active_model_name = active_model
    child._retained_weights_evicted = evicted
    child.send_memory_report_message = Mock(return_value=True)  # type: ignore[method-assign]
    return child


def _model_state_changes(queue: _FakeQueue) -> list[HordeModelStateChangeMessage]:
    return [m for m in queue.messages if isinstance(m, HordeModelStateChangeMessage)]


def _process_state_changes(queue: _FakeQueue) -> list[HordeProcessStateChangeMessage]:
    return [
        m
        for m in queue.messages
        if isinstance(m, HordeProcessStateChangeMessage) and not isinstance(m, HordeModelStateChangeMessage)
    ]


def test_an_evicted_grant_is_reported_as_a_vram_unload() -> None:
    """The child reports the model out of VRAM, which is what clears the parent's retained record."""
    queue = _FakeQueue()
    child = _child(queue, evicted=True)

    child.report_retained_weights_evicted()

    model_changes = _model_state_changes(queue)
    assert len(model_changes) == 1, f"expected exactly one model state change, got {queue.messages}"
    assert model_changes[0].process_state == HordeProcessState.UNLOADED_MODEL_FROM_VRAM
    assert model_changes[0].horde_model_state == ModelLoadState.LOADED_IN_RAM
    assert model_changes[0].horde_model_name == "sdxl-checkpoint"
    assert child.send_memory_report_message.called, (
        "the parent re-prices the slot from a fresh device reading, so the report has to accompany the "
        "state change rather than waiting for the next periodic one"
    )
    assert [m.process_state for m in _process_state_changes(queue)] == [HordeProcessState.WAITING_FOR_JOB]


def test_a_job_whose_grant_held_reports_nothing() -> None:
    """The ordinary retained job is silent here: its weights are where the parent thinks they are."""
    queue = _FakeQueue()
    child = _child(queue, evicted=False)

    child.report_retained_weights_evicted()

    assert queue.messages == []
    assert not child.send_memory_report_message.called


def test_the_report_fires_once_per_job() -> None:
    """The latch is spent by the report, so a second call for the same job cannot re-clear a fresh grant.

    The next job's dispatch stamps a new grant the moment this one is settled, and a repeat report would
    drop weights that job is entitled to keep.
    """
    queue = _FakeQueue()
    child = _child(queue, evicted=True)

    child.report_retained_weights_evicted()
    sent_after_first = len(queue.messages)
    child.report_retained_weights_evicted()

    assert len(queue.messages) == sent_after_first
    assert child._retained_weights_evicted is False


def test_a_slot_with_no_active_model_still_reports_its_memory() -> None:
    """With no model to name, the device reading is still worth sending and no model change is invented."""
    queue = _FakeQueue()
    child = _child(queue, evicted=True, active_model=None)

    child.report_retained_weights_evicted()

    assert _model_state_changes(queue) == []
    assert child.send_memory_report_message.called


class _StubUnloadResult:
    """Stands in for hordelib's ``VramUnloadResult`` so these need no engine install."""

    def __init__(self, *, complete: bool, remaining_loaded_models: int = 0, freed_mb: float = 0.0) -> None:
        self.complete = complete
        self.remaining_loaded_models = remaining_loaded_models
        self.freed_mb = freed_mb


class _StubBackend:
    """A backend whose full-VRAM free reports whatever the test says the device was left holding."""

    def __init__(self, result: _StubUnloadResult) -> None:
        self._result = result
        self.free_vram_calls = 0

    def free_vram(self) -> _StubUnloadResult:
        self.free_vram_calls += 1
        return self._result


class _StubHorde:
    def __init__(self, backend: _StubBackend) -> None:
        self.backend = backend


def _unloading_child(queue: _FakeQueue, result: _StubUnloadResult) -> HordeInferenceProcess:
    """An inference child wired to a stub backend, with only the state the unload handler reads."""
    child = object.__new__(HordeInferenceProcess)
    child.process_id = 4
    child.process_launch_identifier = 1
    child.process_message_queue = queue  # type: ignore[assignment]
    child._active_model_name = "sdxl-checkpoint"
    child._dry_run_skip_inference = False
    child._horde = _StubHorde(_StubBackend(result))  # type: ignore[assignment]
    child.send_memory_report_message = Mock(return_value=True)  # type: ignore[method-assign]
    child.clear_gc_and_torch_cache = Mock()  # type: ignore[method-assign]
    return child


def test_a_completed_unload_reports_the_model_in_host_ram() -> None:
    """The ordinary unload: the device gave the weights back, so RAM residency is the truth."""
    queue = _FakeQueue()
    child = _unloading_child(queue, _StubUnloadResult(complete=True))

    child.unload_models_from_vram()

    model_changes = _model_state_changes(queue)
    assert len(model_changes) == 1
    assert model_changes[0].process_state == HordeProcessState.UNLOADED_MODEL_FROM_VRAM
    assert model_changes[0].horde_model_state == ModelLoadState.LOADED_IN_RAM
    assert model_changes[0].vram_unload_refused is False


def test_a_refused_unload_reports_the_model_still_in_vram() -> None:
    """Weights the backend could not free stay reported as VRAM-resident, refusal and all.

    A full free is a request the backend answers by dropping what it can, skipping any model a live reference
    still pins. Reporting host-RAM residency on the strength of having issued the command hands the parent
    room the card is still holding, and every admission decision after it is made against that room.
    """
    queue = _FakeQueue()
    child = _unloading_child(queue, _StubUnloadResult(complete=False, remaining_loaded_models=6, freed_mb=1966.0))

    child.unload_models_from_vram()

    model_changes = _model_state_changes(queue)
    assert len(model_changes) == 1
    assert model_changes[0].horde_model_state == ModelLoadState.LOADED_IN_VRAM
    assert model_changes[0].vram_unload_refused is True
    assert model_changes[0].process_state != HordeProcessState.UNLOADED_MODEL_FROM_VRAM, (
        "naming the state that means the weights left VRAM would let the parent's own unload reconciliation "
        "clear a residency the device still has"
    )
    assert child.send_memory_report_message.called


def test_a_dry_run_lane_never_claims_a_refusal() -> None:
    """A lane with no backend runs no free, so it has nothing to refuse and reports the ordinary unload."""
    queue = _FakeQueue()
    child = _unloading_child(queue, _StubUnloadResult(complete=False, remaining_loaded_models=6))
    child._dry_run_skip_inference = True

    child.unload_models_from_vram()

    model_changes = _model_state_changes(queue)
    assert model_changes[0].vram_unload_refused is False
    assert model_changes[0].horde_model_state == ModelLoadState.LOADED_IN_RAM
