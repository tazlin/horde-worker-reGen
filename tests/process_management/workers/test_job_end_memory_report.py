"""Pins the fresh memory report an inference child sends ahead of its job result.

The parent decides its next dispatch on receipt of a result, and prices an idle lane by the reservation that lane
last reported. hordelib drops the job's weights at the end of the run, so a lane whose last report came from
inside the sampling stage reads several GB fuller than the card holds until the reporter thread's next tick.
Reporting ahead of the result, on both the completed and the failed branch, means the parent has applied the
post-job reservation by the time it reads the result.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
)
from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess
from tests.process_management.conftest import make_job_pop_response

_MODEL = "sdxl-checkpoint"


class _MemoryReportMarker:
    """Stands in on the queue for a memory report, so its position among the child's messages is observable."""


class _RecordingQueue:
    """A minimal stand-in for the process message queue that records what the child sends, in order."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        """Record a message the child sent to the parent."""
        self.messages.append(message)


def _child(queue: _RecordingQueue, *, results: list[SimpleNamespace] | None) -> HordeInferenceProcess:
    """An inference child wired only for the START_INFERENCE job-end path, with no backend.

    The memory report is replaced by a marker on the same queue because it reads the device; everything that
    decides message order (the branch itself and the result builder) stays real.
    """
    child = object.__new__(HordeInferenceProcess)
    child.process_id = 3
    child.process_launch_identifier = 1
    child.process_message_queue = queue  # type: ignore[assignment]
    child._active_model_name = _MODEL
    child._dry_run_skip_inference = False
    child._end_process = False
    child._last_inference_error = None
    child._last_job_inference_rate = None
    child._current_job_kept_model_resident = False
    child._retained_weights_evicted = False

    def _record_memory_report(*_args: object, **_kwargs: object) -> bool:
        queue.put(_MemoryReportMarker())
        return True

    child.send_memory_report_message = Mock(side_effect=_record_memory_report)  # type: ignore[method-assign]
    child._resolve_aux_models = Mock(return_value=True)  # type: ignore[method-assign]
    child.start_inference = Mock(return_value=results)  # type: ignore[method-assign]
    child.on_horde_model_state_change = Mock()  # type: ignore[method-assign]
    child.preload_model = Mock()  # type: ignore[method-assign]
    child.unload_models_from_ram = Mock()  # type: ignore[method-assign]
    child.send_process_state_change_message = Mock()  # type: ignore[method-assign]
    return child


def _start_inference(child: HordeInferenceProcess) -> None:
    child._receive_and_handle_control_message(
        HordeInferenceControlMessage(
            control_flag=HordeControlFlag.START_INFERENCE,
            horde_model_name=_MODEL,
            sdk_api_job_info=make_job_pop_response(model=_MODEL),
        ),
    )


def _result_index(queue: _RecordingQueue) -> int:
    indices = [i for i, message in enumerate(queue.messages) if isinstance(message, HordeInferenceResultMessage)]
    assert len(indices) == 1, f"expected exactly one result, got {queue.messages}"
    return indices[0]


def test_a_completed_job_reports_memory_before_its_result() -> None:
    """The report the parent prices the emptied lane from lands ahead of the result it dispatches on."""
    queue = _RecordingQueue()
    child = _child(queue, results=[SimpleNamespace(rawpng=io.BytesIO(b"\x89PNG\r\n"), faults=[])])

    _start_inference(child)

    result_at = _result_index(queue)
    assert any(isinstance(message, _MemoryReportMarker) for message in queue.messages[:result_at]), (
        "a completed job's result must follow a fresh memory report, or the parent's next dispatch reads the "
        f"lane at its sampling-time reservation: {queue.messages}"
    )


def test_a_failed_job_reports_memory_before_its_result() -> None:
    """The failed branch keeps the report it has always sent ahead of its result."""
    queue = _RecordingQueue()
    child = _child(queue, results=None)

    _start_inference(child)

    result_at = _result_index(queue)
    assert any(isinstance(message, _MemoryReportMarker) for message in queue.messages[:result_at]), (
        f"a failed job's result must follow a fresh memory report: {queue.messages}"
    )
