"""Pins how an inference child recognises a CUDA runtime error and reports it on the faulted result.

A CUDA runtime error can leave a process's CUDA context failing every later kernel while its siblings on the card
run normally, so the parent replaces the process on the first one. The child recognises it by type: torch raises
``AcceleratorError`` directly, and the backend's pipeline error carries the type name ComfyUI recorded for the
failing node. The allocator's out-of-memory error leaves the context usable and must not be mistaken for one.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from hordelib.execution.interface import PipelineExecutionError

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
)
from horde_worker_regen.process_management.workers.inference_process import (
    HordeInferenceProcess,
    _is_cuda_context_fault,
)
from tests.process_management.conftest import make_job_pop_response

_MODEL = "Anima-Turbo-v1.1"
_ACCELERATOR_ERROR_NAME = f"{torch.AcceleratorError.__module__}.{torch.AcceleratorError.__qualname__}"
_ALLOCATOR_OOM_NAME = f"{torch.OutOfMemoryError.__module__}.{torch.OutOfMemoryError.__qualname__}"


def _pipeline_error(exception_type: str | None) -> PipelineExecutionError:
    return PipelineExecutionError(
        "Pipeline failed to run - declared output node(s) ['output_image'] produced no results.",
        exception_type=exception_type,
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(_pipeline_error(_ACCELERATOR_ERROR_NAME), True, id="pipeline_error_naming_accelerator_error"),
        pytest.param(_pipeline_error(_ALLOCATOR_OOM_NAME), False, id="pipeline_error_naming_allocator_oom"),
        pytest.param(_pipeline_error(None), False, id="pipeline_error_without_a_node_error"),
        pytest.param(
            RuntimeError(f"sampler (KSampler): {_ACCELERATOR_ERROR_NAME}: CUDA error: out of memory"),
            False,
            id="plain_runtime_error_with_the_name_in_its_text",
        ),
    ],
)
def test_a_cuda_runtime_error_is_recognised_by_type(error: BaseException, expected: bool) -> None:
    """Only the type decides. Text naming the error does not count."""
    assert _is_cuda_context_fault(error) is expected


class _RecordingQueue:
    """A minimal stand-in for the process message queue that records what the child sends."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        """Record a message the child sent to the parent."""
        self.messages.append(message)


def _failing_child(queue: _RecordingQueue, *, cuda_context_fault: bool) -> HordeInferenceProcess:
    """An inference child whose job fails, with the backend replaced by the state its failure leaves behind."""
    child = object.__new__(HordeInferenceProcess)
    child.process_id = 2
    child.process_launch_identifier = 1
    child.process_message_queue = queue  # type: ignore[assignment]
    child._active_model_name = _MODEL
    child._dry_run_skip_inference = False
    child._end_process = False
    child._last_inference_error = None
    child._last_job_inference_rate = None
    child._current_job_kept_model_resident = False
    child._retained_weights_evicted = False

    def _fail(*_args: object, **_kwargs: object) -> None:
        child._last_inference_error = f"RuntimeError: sampler (KSampler): {_ACCELERATOR_ERROR_NAME}: CUDA error"
        child._last_inference_cuda_context_fault = cuda_context_fault

    child.send_memory_report_message = Mock(return_value=True)  # type: ignore[method-assign]
    child._resolve_aux_models = Mock(return_value=True)  # type: ignore[method-assign]
    child.start_inference = Mock(side_effect=_fail)  # type: ignore[method-assign]
    child.on_horde_model_state_change = Mock()  # type: ignore[method-assign]
    child.preload_model = Mock()  # type: ignore[method-assign]
    child.unload_models_from_ram = Mock()  # type: ignore[method-assign]
    child.send_process_state_change_message = Mock()  # type: ignore[method-assign]
    return child


@pytest.mark.parametrize("cuda_context_fault", [True, False])
def test_the_faulted_result_carries_the_verdict(cuda_context_fault: bool) -> None:
    """The parent reads the verdict from a field on the result message."""
    queue = _RecordingQueue()
    child = _failing_child(queue, cuda_context_fault=cuda_context_fault)

    child._receive_and_handle_control_message(
        HordeInferenceControlMessage(
            control_flag=HordeControlFlag.START_INFERENCE,
            horde_model_name=_MODEL,
            sdk_api_job_info=make_job_pop_response(model=_MODEL),
        ),
    )

    results = [message for message in queue.messages if isinstance(message, HordeInferenceResultMessage)]
    assert len(results) == 1
    assert results[0].cuda_context_fault is cuda_context_fault
