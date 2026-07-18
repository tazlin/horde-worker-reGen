"""Tests for the dry-run component lane process (its ML paths are rig-only)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import (
    GENERATION_STATE,
    HordeJobMetricsMessage,
    HordeProcessState,
    HordeTextEncodeControlMessage,
    HordeTextEncodeResultMessage,
    PipelineStageTag,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.workers.component_lane_process import HordeComponentLaneProcess

from .conftest import make_job_pop_response


class _FakeQueue:
    """A minimal stand-in for the process message queue that records what the lane sends."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        """Record a message the lane sent to the parent."""
        self.messages.append(message)


def _make_dry_run_lane(queue: _FakeQueue) -> HordeComponentLaneProcess:
    return HordeComponentLaneProcess(
        process_id=5,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
        dry_run=True,
    )


def test_dry_run_lane_is_component_kind_and_reports_ready() -> None:
    """A dry-run lane constructs without the backend, is the COMPONENT kind, and signals it is ready."""
    queue = _FakeQueue()
    lane = _make_dry_run_lane(queue)

    assert lane.process_type is HordeProcessType.COMPONENT
    states = [getattr(message, "process_state", None) for message in queue.messages]
    assert HordeProcessState.WAITING_FOR_JOB in states


def test_dry_run_lane_cleanup_is_safe_without_a_client() -> None:
    """Teardown with no sharing client installed must not raise."""
    lane = _make_dry_run_lane(_FakeQueue())
    lane.cleanup_for_exit()


def test_text_encode_fault_carries_exception_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A faulted text-encode result carries the originating exception summary, not a blank reason.

    Drives the real (non-dry-run) fault path with a backend whose encode raises: the reported result must be
    faulted and its ``fault_reason`` must be the ``"{type}: {message}"`` summary the parent threads on so the
    orchestrator and detectors are not blind to the stage fault.
    """
    import horde_sdk.worker.dispatch.ai_horde.image.convert as convert_module

    import horde_worker_regen.reference_helper as reference_helper

    queue = _FakeQueue()
    lane = _make_dry_run_lane(queue)
    lane._dry_run = False
    lane._horde = Mock()
    lane._horde.encode_text_stage.side_effect = RuntimeError("CUDA out of memory")
    monkeypatch.setattr(reference_helper, "ensure_offline_reference_manager", lambda: Mock())
    monkeypatch.setattr(
        convert_module,
        "convert_image_job_pop_response_to_parameters",
        lambda **_kwargs: Mock(generation_parameters=Mock()),
    )

    job = make_job_pop_response(model="SDXL 1.0")
    queue.messages.clear()
    lane._run_text_encode(
        HordeTextEncodeControlMessage(horde_model_name="SDXL 1.0", job_id=job.id_, sdk_api_job_info=job),
    )

    results = [message for message in queue.messages if isinstance(message, HordeTextEncodeResultMessage)]
    assert len(results) == 1
    assert results[0].state == GENERATION_STATE.faulted
    assert results[0].fault_reason == "RuntimeError: CUDA out of memory"


def _patch_metrics_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point hordelib's metrics collector at a stand-in that yields one disk->RAM model-load event."""
    from unittest.mock import Mock

    from hordelib.metrics import JobPhaseMetrics, ModelLoadEvent

    def _snapshot() -> JobPhaseMetrics:
        return JobPhaseMetrics(
            model_loads=[
                ModelLoadEvent(model_name="text_encoders", phase="disk_to_ram", duration_seconds=1.0, timestamp=0.0),
            ],
        )

    collector = Mock()
    collector.snapshot_and_reset_job.side_effect = _snapshot
    monkeypatch.setattr("hordelib.api.get_metrics_collector", lambda: collector)


def test_text_encode_emits_stage_tagged_job_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a text-encode stage the lane forwards a stage-tagged HordeJobMetricsMessage for the job."""
    _patch_metrics_collector(monkeypatch)
    queue = _FakeQueue()
    lane = _make_dry_run_lane(queue)

    job = make_job_pop_response(model="SDXL 1.0")
    queue.messages.clear()
    lane._run_text_encode(
        HordeTextEncodeControlMessage(horde_model_name="SDXL 1.0", job_id=job.id_, sdk_api_job_info=job),
    )

    metrics = [message for message in queue.messages if isinstance(message, HordeJobMetricsMessage)]
    assert len(metrics) == 1
    assert metrics[0].stage is PipelineStageTag.TEXT_ENCODE
    assert metrics[0].job_id == str(job.id_)
    assert any(load.phase == "disk_to_ram" for load in metrics[0].phase_metrics.model_loads)
