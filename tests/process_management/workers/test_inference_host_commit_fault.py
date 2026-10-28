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
from horde_sdk.ai_horde_api import GENERATION_STATE
from hordelib.execution.interface import PipelineExecutionError
from hordelib.execution.zero_copy_load import HostCommitError

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeProcessState,
    ModelLoadState,
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


def _child_whose_dispatch_preload_raises(queue: _RecordingQueue, error: BaseException) -> HordeInferenceProcess:
    """An inference child with no model loaded whose real preload runs over a backend raising ``error``."""
    child = _child_whose_pipeline_raises(queue, RuntimeError("the pipeline must not run"))
    child._active_model_name = None
    child._is_busy = False
    child._horde = Mock()  # type: ignore[assignment]
    child._horde.preload_model.side_effect = error
    child.clear_gc_and_torch_cache = Mock()  # type: ignore[method-assign]
    del child.preload_model
    return child


def _dispatch_one_job(child: HordeInferenceProcess) -> None:
    child._receive_and_handle_control_message(
        HordeInferenceControlMessage(
            control_flag=HordeControlFlag.START_INFERENCE,
            horde_model_name=_MODEL,
            sdk_api_job_info=make_job_pop_response(model=_MODEL),
        ),
    )


def _reported_process_states(child: HordeInferenceProcess) -> list[HordeProcessState]:
    return [
        call.kwargs.get("process_state", call.args[0] if call.args else None)
        for call in child.send_process_state_change_message.call_args_list  # type: ignore[attr-defined]
    ]


def test_a_dispatch_whose_preload_the_host_refuses_is_faulted_back_unrun() -> None:
    """A refused mapping at dispatch faults the job for re-admission, and the pipeline never loads it itself."""
    queue = _RecordingQueue()
    child = _child_whose_dispatch_preload_raises(queue, HostCommitError("commit refused"))

    _dispatch_one_job(child)

    results = [message for message in queue.messages if isinstance(message, HordeInferenceResultMessage)]
    assert len(results) == 1
    assert results[0].state == GENERATION_STATE.faulted
    assert results[0].host_commit_refused is True
    assert child._run_typed_inference.call_count == 0  # type: ignore[attr-defined]
    assert [call[0] for call in child._horde.method_calls] == ["preload_model"]  # type: ignore[attr-defined]
    assert HordeProcessState.WAITING_FOR_JOB in _reported_process_states(child)
    assert _reported_process_states(child)[-1] == HordeProcessState.WAITING_FOR_JOB

    model_states = [
        call.kwargs["horde_model_state"]
        for call in child.on_horde_model_state_change.call_args_list  # type: ignore[attr-defined]
    ]
    assert model_states == [ModelLoadState.LOADING, ModelLoadState.FAILED]
    assert child._last_inference_error is None


def test_a_dispatch_whose_preload_fails_otherwise_still_ends_the_process() -> None:
    """Any other preload failure leaves torch in an unknown state, so it escapes to the handler that ends it."""
    queue = _RecordingQueue()
    child = _child_whose_dispatch_preload_raises(queue, RuntimeError("corrupt checkpoint"))

    with pytest.raises(RuntimeError, match="corrupt checkpoint"):
        _dispatch_one_job(child)

    assert [message for message in queue.messages if isinstance(message, HordeInferenceResultMessage)] == []
    assert child._run_typed_inference.call_count == 0  # type: ignore[attr-defined]
    assert HordeProcessState.WAITING_FOR_JOB not in _reported_process_states(child)
    failed_report = child.on_horde_model_state_change.call_args_list[-1]  # type: ignore[attr-defined]
    assert failed_report.kwargs["horde_model_state"] == ModelLoadState.FAILED
    assert failed_report.kwargs.get("info") is None


def test_a_failed_run_unloads_and_goes_idle_without_reloading() -> None:
    """A failed run leaves the slot empty, so the retry's load goes through the parent's preload admission."""
    queue = _RecordingQueue()
    child = _child_whose_pipeline_raises(queue, _pipeline_error(_HOST_COMMIT_ERROR_NAME))
    child._is_busy = False
    child._horde = Mock()  # type: ignore[assignment]
    child.clear_gc_and_torch_cache = Mock()  # type: ignore[method-assign]
    del child.preload_model
    del child.unload_models_from_ram

    result = _run_one_job(child, queue)

    assert result.state == GENERATION_STATE.faulted
    assert child._run_typed_inference.call_count == 1  # type: ignore[attr-defined]
    assert child._horde.backend.free_ram.call_count == 1  # type: ignore[attr-defined]
    assert child._horde.preload_model.call_count == 0  # type: ignore[attr-defined]
    assert child._active_model_name is None
    model_states = [
        call.kwargs["horde_model_state"]
        for call in child.on_horde_model_state_change.call_args_list  # type: ignore[attr-defined]
    ]
    assert model_states[-1] == ModelLoadState.ON_DISK
    assert _reported_process_states(child)[-1] == HordeProcessState.WAITING_FOR_JOB


def test_a_refused_preload_leaves_no_active_model() -> None:
    """A preload refused after the previous model was reported unloaded leaves the child without an active model."""
    other_model = "AlbedoBase XL (SDXL)"
    queue = _RecordingQueue()
    child = _child_whose_dispatch_preload_raises(queue, HostCommitError("commit refused"))
    child._active_model_name = "Deliberate"
    child._active_model_load_state = ModelLoadState.LOADED_IN_RAM

    child.preload_model(
        horde_model_name=_MODEL,
        will_load_loras=False,
        seamless_tiling_enabled=False,
        job_info=make_job_pop_response(model=_MODEL),
    )

    assert child._active_model_name is None
    assert child._active_model_load_state is None
    model_states = [
        (call.kwargs["horde_model_name"], call.kwargs["horde_model_state"])
        for call in child.on_horde_model_state_change.call_args_list  # type: ignore[attr-defined]
    ]
    assert model_states == [
        ("Deliberate", ModelLoadState.ON_DISK),
        (_MODEL, ModelLoadState.LOADING),
        (_MODEL, ModelLoadState.FAILED),
    ]

    child._receive_and_handle_control_message(
        HordeInferenceControlMessage(
            control_flag=HordeControlFlag.START_INFERENCE,
            horde_model_name=other_model,
            sdk_api_job_info=make_job_pop_response(model=other_model),
        ),
    )

    failed_reports = [
        call.kwargs.get("info")
        for call in child.send_process_state_change_message.call_args_list  # type: ignore[attr-defined]
        if call.kwargs.get("process_state") == HordeProcessState.INFERENCE_FAILED
    ]
    assert len(failed_reports) == 1
    assert failed_reports[0] is not None
    assert failed_reports[0].endswith("currently active model is None")
