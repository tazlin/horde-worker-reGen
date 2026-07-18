"""Tests for the dry-run VAE lane process's per-stage metrics emission (its ML paths are rig-only)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import (
    HordeJobMetricsMessage,
    HordeVaeDecodeControlMessage,
    HordeVaeEncodeControlMessage,
    PipelineStageTag,
)
from horde_worker_regen.process_management.workers.vae_lane_process import HordeVaeLaneProcess

from .conftest import make_job_pop_response


class _FakeQueue:
    """A minimal stand-in for the process message queue that records what the lane sends."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        """Record a message the lane sent to the parent."""
        self.messages.append(message)


def _make_dry_run_lane(queue: _FakeQueue) -> HordeVaeLaneProcess:
    return HordeVaeLaneProcess(
        process_id=7,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
        dry_run=True,
    )


def _patch_metrics_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point hordelib's metrics collector at a stand-in that yields one disk->RAM model-load event."""
    from hordelib.metrics import JobPhaseMetrics, ModelLoadEvent

    def _snapshot() -> JobPhaseMetrics:
        return JobPhaseMetrics(
            model_loads=[
                ModelLoadEvent(model_name="vae", phase="disk_to_ram", duration_seconds=1.0, timestamp=0.0),
            ],
        )

    collector = Mock()
    collector.snapshot_and_reset_job.side_effect = _snapshot
    monkeypatch.setattr("hordelib.api.get_metrics_collector", lambda: collector)


def _sole_metrics_message(queue: _FakeQueue) -> HordeJobMetricsMessage:
    metrics = [message for message in queue.messages if isinstance(message, HordeJobMetricsMessage)]
    assert len(metrics) == 1
    return metrics[0]


def test_vae_encode_emits_vae_encode_stage_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a VAE-encode stage the lane forwards metrics tagged VAE_ENCODE for the job."""
    _patch_metrics_collector(monkeypatch)
    queue = _FakeQueue()
    lane = _make_dry_run_lane(queue)

    job = make_job_pop_response(model="SDXL 1.0")
    queue.messages.clear()
    lane._run_vae_encode(
        HordeVaeEncodeControlMessage(horde_model_name="SDXL 1.0", job_id=job.id_, sdk_api_job_info=job),
    )

    message = _sole_metrics_message(queue)
    assert message.stage is PipelineStageTag.VAE_ENCODE
    assert message.job_id == str(job.id_)
    assert any(load.phase == "disk_to_ram" for load in message.phase_metrics.model_loads)


def test_vae_decode_emits_vae_decode_stage_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a VAE-decode stage the lane forwards metrics tagged VAE_DECODE for the job."""
    _patch_metrics_collector(monkeypatch)
    queue = _FakeQueue()
    lane = _make_dry_run_lane(queue)

    job = make_job_pop_response(model="SDXL 1.0")
    queue.messages.clear()
    lane._run_vae_decode(
        HordeVaeDecodeControlMessage(
            horde_model_name="SDXL 1.0",
            job_id=job.id_,
            sdk_api_job_info=job,
            latent_bytes=b"dry-run-latent",
        ),
    )

    message = _sole_metrics_message(queue)
    assert message.stage is PipelineStageTag.VAE_DECODE
    assert message.job_id == str(job.id_)
