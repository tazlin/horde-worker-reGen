"""Tests for the base HordeProcess idle-heartbeat (keeps an idle process from looking unresponsive)."""

from __future__ import annotations

import queue
import time
from typing import override
from unittest.mock import Mock

from horde_worker_regen.process_management.horde_process import HordeProcess
from horde_worker_regen.process_management.messages import (
    HordeControlMessage,
    HordeHeartbeatType,
    HordeProcessHeartbeatMessage,
    HordeProcessState,
)


class _StubProcess(HordeProcess):
    """A minimal concrete HordeProcess for exercising base-class behaviour without a real subprocess."""

    @override
    def cleanup_for_exit(self) -> None:
        return

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        return


def _make_stub() -> _StubProcess:
    """Build a stub process and drop the PROCESS_STARTING message its constructor emits."""
    proc = _StubProcess(
        process_id=3,
        process_message_queue=Mock(spec=queue.Queue),
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
    )
    proc.process_message_queue.reset_mock()  # pyrefly: ignore
    return proc


def test_idle_heartbeat_emitted_when_waiting() -> None:
    """An idle (WAITING_FOR_JOB) process emits an OTHER heartbeat so its liveness keeps refreshing."""
    proc = _make_stub()
    proc._last_sent_process_state = HordeProcessState.WAITING_FOR_JOB
    proc._last_heartbeat_time = 0.0

    proc._maybe_send_idle_heartbeat()

    assert proc.process_message_queue.put.call_count == 1  # pyrefly: ignore
    sent = proc.process_message_queue.put.call_args[0][0]  # pyrefly: ignore
    assert isinstance(sent, HordeProcessHeartbeatMessage)
    assert sent.heartbeat_type is HordeHeartbeatType.OTHER


def test_idle_heartbeat_skipped_while_busy() -> None:
    """A process mid-inference must not emit the idle heartbeat (it would disturb stuck-detection)."""
    proc = _make_stub()
    proc._last_sent_process_state = HordeProcessState.INFERENCE_STARTING
    proc._last_heartbeat_time = 0.0

    proc._maybe_send_idle_heartbeat()

    proc.process_message_queue.put.assert_not_called()  # pyrefly: ignore


def test_idle_heartbeat_is_throttled() -> None:
    """A heartbeat sent within the interval suppresses the next idle heartbeat."""
    proc = _make_stub()
    proc._last_sent_process_state = HordeProcessState.WAITING_FOR_JOB
    proc._last_heartbeat_time = time.time()

    proc._maybe_send_idle_heartbeat()

    proc.process_message_queue.put.assert_not_called()  # pyrefly: ignore
