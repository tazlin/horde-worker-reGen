"""Pins how an inference child recognises the host refusing a checkpoint commit inside a running job.

The backend's mapping guard raises ``HostCommitError`` before any weight is adopted. Inside a job, ComfyUI catches
the node's exception, so the backend's pipeline error carries the type by name. The child marks the faulted result
so the parent requeues the job against host memory, and it does not retry the job itself.
"""

from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest
import torch
from hordelib.execution.interface import PipelineExecutionError
from hordelib.execution.zero_copy_load import HostCommitError

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
)
from horde_worker_regen.process_management.workers.inference_process import (
    HordeInferenceProcess,
    _is_host_commit_error,
)
from tests.process_management.conftest import make_job_pop_response

_MODEL = "Anima-Turbo-v1.1"
_HOST_COMMIT_ERROR_NAME = f"{HostCommitError.__module__}.{HostCommitError.__qualname__}"
_ALLOCATOR_OOM_NAME = f"{torch.OutOfMemoryError.__module__}.{torch.OutOfMemoryError.__qualname__}"


def _pipeline_error(exception_type: str | None) -> PipelineExecutionError:
    return PipelineExecutionError(
        "Pipeline failed to run - declared output node(s) ['output_image'] produced no results.",
        exception_type=exception_type,
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(HostCommitError("commit refused"), True, id="direct_host_commit_error"),
        pytest.param(_pipeline_error(_HOST_COMMIT_ERROR_NAME), True, id="pipeline_error_naming_host_commit_error"),
        pytest.param(_pipeline_error(_ALLOCATOR_OOM_NAME), False, id="pipeline_error_naming_allocator_oom"),
        pytest.param(_pipeline_error(None), False, id="pipeline_error_without_a_node_error"),
        pytest.param(
            RuntimeError(f"CheckpointLoader: {_HOST_COMMIT_ERROR_NAME}: commit refused"),
            False,
            id="plain_runtime_error_with_the_name_in_its_text",
        ),
    ],
)
def test_a_host_commit_refusal_is_recognised_by_type(error: BaseException, expected: bool) -> None:
    """Only the type decides. Text naming the error does not count."""
    assert _is_host_commit_error(error) is expected


class _RecordingQueue:
    """A minimal stand-in for the process message queue that records what the child sends."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        """Record a message the child sent to the parent."""
        self.messages.append(message)


def _child_whose_pipeline_raises(queue: _RecordingQueue, error: BaseException) -> HordeInferenceProcess:
    """An inference child that runs its real job path over a backend raising ``error``."""
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
    child._gpu_sampling_lease = None
    child._inference_semaphore = threading.Semaphore(1)  # type: ignore[assignment]
    child._vae_decode_semaphore = threading.Semaphore(1)  # type: ignore[assignment]

    child._run_typed_inference = Mock(side_effect=error)  # type: ignore[method-assign]
    child.send_heartbeat_message = Mock()  # type: ignore[method-assign]
    child._send_job_metrics_message = Mock()  # type: ignore[method-assign]
    child._send_download_metrics_if_any = Mock()  # type: ignore[method-assign]
    child._release_inference_slot = Mock()  # type: ignore[method-assign]
    child.send_memory_report_message = Mock(return_value=True)  # type: ignore[method-assign]
    child._resolve_aux_models = Mock(return_value=True)  # type: ignore[method-assign]
    child.on_horde_model_state_change = Mock()  # type: ignore[method-assign]
    child.preload_model = Mock()  # type: ignore[method-assign]
    child.unload_models_from_ram = Mock()  # type: ignore[method-assign]
    child.send_process_state_change_message = Mock()  # type: ignore[method-assign]
    return child


def _run_one_job(child: HordeInferenceProcess, queue: _RecordingQueue) -> HordeInferenceResultMessage:
    child._receive_and_handle_control_message(
        HordeInferenceControlMessage(
            control_flag=HordeControlFlag.START_INFERENCE,
            horde_model_name=_MODEL,
            sdk_api_job_info=make_job_pop_response(model=_MODEL),
        ),
    )
    results = [message for message in queue.messages if isinstance(message, HordeInferenceResultMessage)]
    assert len(results) == 1
    return results[0]


def test_an_in_run_host_commit_refusal_marks_the_faulted_result() -> None:
    """The refusal reaches the parent as a typed mark, and the child ran the job exactly once."""
    queue = _RecordingQueue()
    child = _child_whose_pipeline_raises(queue, _pipeline_error(_HOST_COMMIT_ERROR_NAME))

    result = _run_one_job(child, queue)

    assert result.host_commit_refused is True
    assert result.cuda_context_fault is False
    assert child._run_typed_inference.call_count == 1  # type: ignore[attr-defined]


def test_an_allocator_oom_is_not_marked_as_a_host_commit_refusal() -> None:
    """An ordinary out-of-memory fault keeps the ordinary fault path."""
    queue = _RecordingQueue()
    child = _child_whose_pipeline_raises(queue, _pipeline_error(_ALLOCATOR_OOM_NAME))

    result = _run_one_job(child, queue)

    assert result.host_commit_refused is False
