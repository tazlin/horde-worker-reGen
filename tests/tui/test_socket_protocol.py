"""Tests for the worker-host socket framing and message round-trips."""

from __future__ import annotations

import socket

import pytest

from horde_worker_regen.process_management.supervisor_channel import (
    SupervisorCommand,
    SupervisorControlMessage,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui import socket_protocol as sp


def _snapshot() -> WorkerStateSnapshot:
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Worker", worker_version="12.0.0"),
        num_jobs_submitted=3,
    )


def test_frame_roundtrip() -> None:
    """A message dict survives a send/recv across a socket pair."""
    left, right = socket.socketpair()
    try:
        sp.send_frame(left, {"type": "x", "n": 1})
        assert sp.recv_frame(right) == {"type": "x", "n": 1}
    finally:
        left.close()
        right.close()


def test_snapshot_roundtrip() -> None:
    """A snapshot serializes and reconstructs with its fields intact."""
    left, right = socket.socketpair()
    try:
        sp.send_frame(left, sp.snapshot_message(_snapshot()))
        message = sp.recv_frame(right)
        assert message is not None and message["type"] == sp.MSG_SNAPSHOT
        restored = sp.parse_snapshot(message)
        assert restored.num_jobs_submitted == 3
        assert restored.config.dreamer_name == "Worker"
    finally:
        left.close()
        right.close()


def test_command_roundtrip() -> None:
    """A control command (with its enum and extra fields) reconstructs exactly."""
    command = SupervisorControlMessage(
        command=SupervisorCommand.SET_DOWNLOAD_RATE_LIMIT,
        download_rate_limit_kbps=500,
    )
    left, right = socket.socketpair()
    try:
        sp.send_frame(left, sp.command_message(command))
        message = sp.recv_frame(right)
        assert message is not None
        restored = sp.parse_command(message)
        assert restored.command is SupervisorCommand.SET_DOWNLOAD_RATE_LIMIT
        assert restored.download_rate_limit_kbps == 500
    finally:
        left.close()
        right.close()


def test_recv_frame_returns_none_on_clean_close() -> None:
    """recv_frame yields None (not an error) when the peer closes the connection."""
    left, right = socket.socketpair()
    left.close()
    try:
        assert sp.recv_frame(right) is None
    finally:
        right.close()


def test_resolve_attach_address() -> None:
    """host:port parsing handles IPv4 and bracketed IPv6 literals."""
    assert sp.resolve_attach_address("127.0.0.1:7717") == ("127.0.0.1", 7717)
    assert sp.resolve_attach_address("[::1]:9000") == ("::1", 9000)
    with pytest.raises(ValueError, match="host:port"):
        sp.resolve_attach_address("no-port")


def test_snapshot_full_fidelity_roundtrip() -> None:
    """A rich snapshot survives the JSON transport intact, including the parts JSON does not natively keep.

    The pipe transport uses pickle (which preserves int dict keys and nested types); the socket uses JSON
    (which stringifies dict keys). This guards that the dashboard sees identical data in attach mode:
    int-keyed per-process maps, nested process/download models, enum fields, and derived properties.
    """
    from horde_worker_regen.process_management.supervisor_channel import (
        CurrentDownloadStatus,
        DownloadPhase,
        DownloadStatusSnapshot,
        ProcessSnapshot,
        RecentJobRecord,
    )

    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="W", worker_version="12.0.0"),
        processes=[
            ProcessSnapshot(
                process_id=2,
                process_type="INFERENCE",
                last_process_state="WAITING_FOR_JOB",
                is_alive=True,
                is_busy=False,
                total_vram_mb=16000,
            ),
        ],
        vram_high_water_mb_per_process={0: 123, 1: 456},
        disk_free_bytes={"/models": 5000},
        downloads=DownloadStatusSnapshot(
            phase=DownloadPhase.DOWNLOADING,
            current=CurrentDownloadStatus(
                model_name="M",
                feature="image model",
                target_dir="/x",
                downloaded_bytes=10,
                total_bytes=100,
            ),
        ),
        recent_jobs=[RecentJobRecord(job_id="j1", faulted=True)],
        kudos_per_hour=42.5,
    )

    left, right = socket.socketpair()
    try:
        sp.send_frame(left, sp.snapshot_message(snapshot))
        message = sp.recv_frame(right)
        assert message is not None
        restored = sp.parse_snapshot(message)
    finally:
        left.close()
        right.close()

    assert restored.vram_high_water_mb_per_process == {0: 123, 1: 456}
    assert all(isinstance(key, int) for key in restored.vram_high_water_mb_per_process)
    assert restored.processes[0].process_id == 2
    assert restored.processes[0].total_vram_mb == 16000
    assert restored.downloads is not None
    assert restored.downloads.phase is DownloadPhase.DOWNLOADING
    assert restored.downloads.current is not None
    assert restored.downloads.current.percent == 10.0
    assert restored.recent_jobs[0].faulted is True
    assert restored.kudos_per_hour == 42.5
