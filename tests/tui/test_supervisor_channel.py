"""Unit tests for the supervisor channel: model round-trips and the threaded channel wrapper."""

from __future__ import annotations

import multiprocessing
import pickle
import time
from types import SimpleNamespace

from horde_worker_regen.process_management.supervisor_channel import (
    SUPERVISOR_PROTOCOL_VERSION,
    ProcessSnapshot,
    RecentJobRecord,
    SupervisorChannel,
    SupervisorCommand,
    SupervisorControlMessage,
    WorkerConfigSummary,
    WorkerLivenessFrame,
    WorkerStateSnapshot,
)


def _make_snapshot() -> WorkerStateSnapshot:
    config = WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0", num_models=2)
    process = ProcessSnapshot(
        process_id=0,
        process_type="INFERENCE",
        last_process_state="INFERENCE_STARTING",
        is_alive=True,
        is_busy=True,
        last_current_step=10,
        last_total_steps=30,
        last_iterations_per_second=8.0,
        vram_usage_mb=8000,
        total_vram_mb=24000,
    )
    return WorkerStateSnapshot(config=config, processes=[process], num_jobs_submitted=7)


def test_snapshot_pickle_roundtrip() -> None:
    """A snapshot survives a pickle round-trip (the multiprocessing pipe transport)."""
    snapshot = _make_snapshot()
    restored = pickle.loads(pickle.dumps(snapshot))
    assert restored == snapshot
    assert restored.protocol_version == SUPERVISOR_PROTOCOL_VERSION


def test_snapshot_json_roundtrip() -> None:
    """A snapshot survives a JSON round-trip (the socket-fallback transport)."""
    snapshot = _make_snapshot()
    restored = WorkerStateSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored.processes[0].last_iterations_per_second == 8.0
    assert restored.num_jobs_submitted == 7


def test_recent_job_record_from_metrics_record() -> None:
    """RecentJobRecord projects the lean fields (including job features) from a metrics record."""
    from horde_worker_regen.process_management.run_metrics import JobMetricsRecord

    record = JobMetricsRecord(
        job_id="abc",
        is_alchemy=True,
        faulted=False,
        queue_wait_seconds=1.5,
        e2e_seconds=12.0,
        safety_seconds=0.4,
        model_name="Deliberate",
        steps=30,
        loras_count=2,
        control_type="canny",
    )
    lean = RecentJobRecord.from_metrics_record(record)
    assert lean.job_id == "abc"
    assert lean.is_alchemy is True
    assert lean.e2e_seconds == 12.0
    assert lean.model_name == "Deliberate"
    assert lean.features is not None
    assert lean.features.loras == 2
    assert lean.features.control_type == "canny"


def test_recent_job_record_without_features() -> None:
    """A plain job (no LoRAs/controlnet/etc.) projects with no feature summary."""
    from horde_worker_regen.process_management.run_metrics import JobMetricsRecord

    record = JobMetricsRecord(job_id="plain", e2e_seconds=3.0, steps=20)
    lean = RecentJobRecord.from_metrics_record(record)
    assert lean.features is None
    assert lean.steps == 20


def _fake_process_info() -> SimpleNamespace:
    """A duck-typed stand-in for HordeProcessInfo (only the attributes ProcessSnapshot reads)."""
    return SimpleNamespace(
        process_id=1,
        process_type=SimpleNamespace(name="INFERENCE"),
        last_process_state=SimpleNamespace(name="WAITING_FOR_JOB"),
        is_process_alive=lambda: True,
        is_process_busy=lambda: False,
        loaded_horde_model_name="Deliberate",
        loaded_horde_model_baseline="stable_diffusion_1",
        last_job_referenced=SimpleNamespace(id_=SimpleNamespace(root="job-9")),
        last_heartbeat_timestamp=time.time(),
        last_heartbeat_delta=0.2,
        last_heartbeat_type=SimpleNamespace(name="OTHER"),
        heartbeats_inference_steps=3,
        last_heartbeat_percent_complete=50,
        ram_usage_bytes=1024,
        vram_usage_mb=2000,
        total_vram_mb=24000,
        batch_amount=1,
        last_iterations_per_second=None,
        last_current_step=None,
        last_total_steps=None,
        vram_used_high_water_mb=2200,
        ram_used_high_water_mb=512,
        num_jobs_completed=7,
    )


def test_process_snapshot_from_process_info() -> None:
    """ProcessSnapshot.from_process_info reads enum names and the current job id without coupling."""
    snapshot = ProcessSnapshot.from_process_info(_fake_process_info())  # type: ignore[arg-type]
    assert snapshot.process_type == "INFERENCE"
    assert snapshot.last_process_state == "WAITING_FOR_JOB"
    assert snapshot.current_job_id == "job-9"
    assert snapshot.loaded_horde_model_baseline == "stable_diffusion_1"
    # An idle process carries no current-job resolution/steps.
    assert snapshot.current_job_width is None
    assert snapshot.current_job_height is None
    assert snapshot.current_job_steps is None


def test_process_snapshot_carries_busy_job_resolution() -> None:
    """A busy process surfaces the active job's resolution and step count for the dashboards."""
    info = _fake_process_info()
    info.is_process_busy = lambda: True
    info.batch_amount = 4
    info.last_job_referenced = SimpleNamespace(
        id_=SimpleNamespace(root="job-9"),
        payload=SimpleNamespace(
            width=832,
            height=1216,
            ddim_steps=28,
            loras=[],
            tis=[],
            control_type=None,
            post_processing=[],
            hires_fix=False,
            workflow=None,
        ),
    )
    snapshot = ProcessSnapshot.from_process_info(info)  # type: ignore[arg-type]
    assert snapshot.current_job_width == 832
    assert snapshot.current_job_height == 1216
    assert snapshot.current_job_steps == 28
    assert snapshot.batch_amount == 4


def test_snapshot_roundtrip_preserves_new_pipeline_and_job_fields() -> None:
    """The new pipeline-tail count and per-job resolution survive a JSON round-trip."""
    snapshot = _make_snapshot()
    snapshot.jobs_pending_submit = 3
    snapshot.processes[0].current_job_width = 1024
    snapshot.processes[0].current_job_height = 1024
    snapshot.processes[0].current_job_steps = 30
    restored = WorkerStateSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored.jobs_pending_submit == 3
    assert restored.processes[0].current_job_width == 1024
    assert restored.processes[0].current_job_steps == 30


def _recv_first(parent: object, frame_type: type, *, timeout: float = 3.0) -> object | None:
    """Drain the pipe until a frame of ``frame_type`` arrives (liveness frames are interleaved now)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if parent.poll(0.1):  # type: ignore[attr-defined]
            message = parent.recv()  # type: ignore[attr-defined]
            if isinstance(message, frame_type):
                return message
    return None


def test_channel_sends_snapshot_without_blocking_and_receives_commands() -> None:
    """The threaded channel delivers snapshots and drains commands over a real in-process pipe."""
    parent, child = multiprocessing.Pipe(duplex=True)
    channel = SupervisorChannel(child)  # pyrefly: ignore
    try:
        assert channel.send_snapshot(_make_snapshot()) is True

        received = _recv_first(parent, WorkerStateSnapshot)
        assert isinstance(received, WorkerStateSnapshot)
        assert received.num_jobs_submitted == 7

        parent.send(SupervisorControlMessage(command=SupervisorCommand.PAUSE))
        commands: list[SupervisorControlMessage] = []
        deadline = time.time() + 3.0
        while time.time() < deadline and not commands:
            commands = channel.drain_commands()
            time.sleep(0.05)
        assert len(commands) == 1
        assert commands[0].command is SupervisorCommand.PAUSE
    finally:
        channel.close()
        parent.close()
        child.close()


def test_channel_emits_liveness_frames_without_any_snapshot() -> None:
    """The channel heartbeats a liveness frame on its own cadence even when no snapshot is ever queued."""
    parent, child = multiprocessing.Pipe(duplex=True)
    channel = SupervisorChannel(child)  # pyrefly: ignore
    try:
        frame = _recv_first(parent, WorkerLivenessFrame)
        assert isinstance(frame, WorkerLivenessFrame)
        assert frame.protocol_version == SUPERVISOR_PROTOCOL_VERSION
        assert frame.loop_alive_wall_time > 0.0
    finally:
        channel.close()
        parent.close()
        child.close()


def test_note_alive_advances_the_carried_loop_time() -> None:
    """``note_alive`` advances the ``loop_alive_wall_time`` carried by subsequent liveness frames."""
    parent, child = multiprocessing.Pipe(duplex=True)
    channel = SupervisorChannel(child)  # pyrefly: ignore
    try:
        first = _recv_first(parent, WorkerLivenessFrame)
        assert isinstance(first, WorkerLivenessFrame)

        time.sleep(0.05)
        channel.note_alive()
        later = _recv_first(parent, WorkerLivenessFrame, timeout=4.0)
        assert isinstance(later, WorkerLivenessFrame)
        assert later.loop_alive_wall_time > first.loop_alive_wall_time
    finally:
        channel.close()
        parent.close()
        child.close()


def test_closed_channel_emits_no_liveness_frame() -> None:
    """A channel whose pipe has died stops emitting frames rather than spinning on a dead connection."""
    parent, child = multiprocessing.Pipe(duplex=True)
    channel = SupervisorChannel(child)  # pyrefly: ignore
    # Kill the consumer end so the next send fails and the channel marks itself closed.
    parent.close()
    deadline = time.time() + 3.0
    while not channel.closed and time.time() < deadline:
        time.sleep(0.05)
    assert channel.closed is True
    channel.close()
    child.close()
