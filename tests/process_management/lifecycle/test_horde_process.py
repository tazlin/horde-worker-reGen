"""Tests for the base HordeProcess idle-heartbeat (keeps an idle process from looking unresponsive)."""

from __future__ import annotations

import enum
import multiprocessing
import queue
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any, cast, override
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeProcessHeartbeatMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcess
from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess


class _StubProcess(HordeProcess):
    """A minimal concrete HordeProcess for exercising base-class behaviour without a real subprocess."""

    handled: list[HordeControlMessage]

    @override
    def cleanup_for_exit(self) -> None:
        return

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        if not hasattr(self, "handled"):
            self.handled = []
        self.handled.append(message)


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


def test_heartbeat_type_change_sent_immediately_within_interval() -> None:
    """A heartbeat whose type changed must be sent at once, even inside the throttle window.

    Per the docstring, the throttle suppresses only repeated *same-type* heartbeats; a type change is
    a meaningful transition (e.g. INFERENCE_STEP -> OTHER when a job hands off to a blocking aux-model
    drain) and must reach the parent so its view of the slot stays accurate. The inverted condition did
    the opposite: it dropped the transition and let same-type spam through.
    """
    proc = _make_stub()
    proc._last_heartbeat_type = HordeHeartbeatType.INFERENCE_STEP
    proc._last_heartbeat_time = time.time()  # squarely inside the 1s throttle window

    proc.send_heartbeat_message(HordeHeartbeatType.OTHER)

    assert proc.process_message_queue.put.call_count == 1  # pyrefly: ignore
    sent = proc.process_message_queue.put.call_args[0][0]  # pyrefly: ignore
    assert sent.heartbeat_type is HordeHeartbeatType.OTHER


def test_same_type_heartbeat_throttled_within_interval() -> None:
    """A repeated same-type heartbeat inside the window is throttled (the actual intent of the gate)."""
    proc = _make_stub()
    proc._last_heartbeat_type = HordeHeartbeatType.OTHER
    proc._last_heartbeat_time = time.time()

    proc.send_heartbeat_message(HordeHeartbeatType.OTHER)

    proc.process_message_queue.put.assert_not_called()  # pyrefly: ignore


def test_same_type_heartbeat_sent_after_interval() -> None:
    """Once the throttle window has elapsed, even a same-type heartbeat is sent again."""
    proc = _make_stub()
    proc._last_heartbeat_type = HordeHeartbeatType.OTHER
    proc._last_heartbeat_time = time.time() - (proc._heartbeat_limit_interval_seconds + 1.0)

    proc.send_heartbeat_message(HordeHeartbeatType.OTHER)

    assert proc.process_message_queue.put.call_count == 1  # pyrefly: ignore


def _make_piped_stub() -> tuple[_StubProcess, Any]:
    """Build a stub wired to a real duplex pipe; returns the stub and the parent's send end."""
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    proc = _StubProcess(
        process_id=7,
        process_message_queue=Mock(spec=queue.Queue),
        pipe_connection=child_conn,
        disk_lock=Mock(),
        process_launch_identifier=0,
    )
    return proc, parent_conn


def test_control_reader_drains_pipe_without_the_main_loop() -> None:
    """The reader thread keeps the control pipe drained even if the main loop never consumes.

    This is the anti-wedge guarantee: the parent's blocking ``send()`` can never back up on a child
    that is busy (e.g. mid aux-model download), because a dedicated thread always reads the pipe.
    """
    proc, parent_conn = _make_piped_stub()
    proc._start_control_pipe_reader()
    try:
        for _ in range(50):
            parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM))

        # Without ever calling receive_and_handle_control_messages, every message should land in the
        # inbox: the reader thread drained the pipe on its own.
        deadline = time.time() + 5.0
        while proc._control_inbox.qsize() < 50 and time.time() < deadline:
            time.sleep(0.01)
        assert proc._control_inbox.qsize() == 50
    finally:
        proc._control_reader_stop.set()


def test_control_messages_are_handled_in_order_from_the_inbox() -> None:
    """Drained messages are handled on the main loop, in order, and END_PROCESS stops the loop."""
    proc, parent_conn = _make_piped_stub()
    proc._start_control_pipe_reader()
    try:
        parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM))
        parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM))

        deadline = time.time() + 5.0
        while proc._control_inbox.qsize() < 2 and time.time() < deadline:
            time.sleep(0.01)

        proc.receive_and_handle_control_messages()
        assert [m.control_flag for m in proc.handled] == [
            HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
            HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
        ]

        parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
        deadline = time.time() + 5.0
        while proc._control_inbox.qsize() < 1 and time.time() < deadline:
            time.sleep(0.01)
        proc.receive_and_handle_control_messages()
        assert proc._end_process is True
    finally:
        proc._control_reader_stop.set()


class _FakeProgressState(enum.Enum):
    progress = enum.auto()
    post_processing = enum.auto()


class _FakeComfyUIProgressUnit(enum.Enum):
    ITERATIONS_PER_SECOND = enum.auto()
    SECONDS_PER_ITERATION = enum.auto()


def _install_fake_hordelib_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the small hordelib.api surface progress_callback imports."""
    fake_api = ModuleType("hordelib.api")
    fake_api.__dict__["ComfyUIProgressUnit"] = _FakeComfyUIProgressUnit
    fake_api.__dict__["ProgressState"] = _FakeProgressState
    fake_api.__dict__["log_free_ram"] = Mock()
    fake_hordelib = ModuleType("hordelib")
    fake_hordelib.__dict__["api"] = fake_api
    monkeypatch.setitem(sys.modules, "hordelib", fake_hordelib)
    monkeypatch.setitem(sys.modules, "hordelib.api", fake_api)


def _make_inference_proc_for_progress() -> HordeInferenceProcess:
    """Build a bare HordeInferenceProcess with only progress-callback collaborators."""
    proc = object.__new__(HordeInferenceProcess)
    proc._active_model_name = "test-model"
    proc._current_job_inference_steps_complete = False
    proc._in_post_processing = False
    proc._post_processing_memory_report_sent = False
    proc._vae_lock_was_acquired = False
    proc._start_inference_time = time.time()
    proc._release_inference_slot = Mock()  # pyrefly: ignore
    proc.send_process_state_change_message = Mock()  # pyrefly: ignore
    proc.send_heartbeat_message = Mock()  # pyrefly: ignore
    proc.send_memory_report_message = Mock(return_value=True)  # pyrefly: ignore
    return proc


def _progress_report(*, step: int = 1, total: int = 20) -> SimpleNamespace:
    """Return a fake hordelib progress report carrying one ComfyUI step."""
    return SimpleNamespace(
        hordelib_progress_state=_FakeProgressState.progress,
        comfyui_progress=SimpleNamespace(
            current_step=step,
            total_steps=total,
            rate=8.0,
            rate_unit=_FakeComfyUIProgressUnit.ITERATIONS_PER_SECOND,
            percent=round(step / total * 100),
        ),
    )


def test_progress_callback_midstep_does_not_poll_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mid-inference step no longer polls VRAM: the reporter thread owns interval sampling now.

    The progress callback runs on the main thread, which is blocked for the whole GPU op, so any report it
    emitted would still be an on-the-main-thread snapshot. Periodic sampling moved to the dedicated reporter
    thread, so a plain mid-step callback emits only a heartbeat and no memory report.
    """
    _install_fake_hordelib_api(monkeypatch)
    proc = _make_inference_proc_for_progress()

    proc.progress_callback(cast(Any, _progress_report(step=3)))
    proc.progress_callback(cast(Any, _progress_report(step=4)))

    cast(Mock, proc.send_memory_report_message).assert_not_called()


def test_sampling_complete_emits_one_boundary_vram_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sampling-complete boundary still sends one precise stage-transition VRAM report."""
    _install_fake_hordelib_api(monkeypatch)
    proc = _make_inference_proc_for_progress()

    proc.progress_callback(cast(Any, _progress_report(step=20, total=20)))

    cast(Mock, proc.send_memory_report_message).assert_called_once_with(include_vram=True)


class _ReporterStubProcess(HordeProcess):
    """A stub whose ``send_memory_report_message`` records each call's ``include_vram`` on a thread-safe list."""

    @override
    def cleanup_for_exit(self) -> None:
        return

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        return

    @override
    def send_memory_report_message(self, include_vram: bool = False) -> bool:
        self.reported_include_vram.append(include_vram)
        return True


def _make_reporter_stub(*, includes_vram: bool, sampling_ready: bool) -> _ReporterStubProcess:
    """Build a reporter-thread stub with a fast cadence and a stubbed device-init guard."""
    proc = _ReporterStubProcess(
        process_id=11,
        process_message_queue=Mock(spec=queue.Queue),
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
    )
    proc.reported_include_vram: list[bool] = []
    proc._periodic_report_includes_vram = includes_vram
    proc._memory_report_interval = 0.02
    proc._offthread_vram_sampling_ready = Mock(return_value=sampling_ready)  # pyrefly: ignore
    return proc


def test_reporter_thread_emits_reports_at_cadence_off_the_main_loop() -> None:
    """The reporter thread sends interval reports on its own, never touching the main loop, and stops cleanly."""
    proc = _make_reporter_stub(includes_vram=True, sampling_ready=True)
    proc._start_memory_reporter_thread()
    try:
        deadline = time.time() + 5.0
        while len(proc.reported_include_vram) < 3 and time.time() < deadline:
            time.sleep(0.01)
        assert len(proc.reported_include_vram) >= 3
        assert all(include_vram is True for include_vram in proc.reported_include_vram)
    finally:
        proc._memory_reporter_stop.set()

    proc._memory_reporter_thread.join(timeout=2.0)  # pyrefly: ignore
    assert not proc._memory_reporter_thread.is_alive()  # pyrefly: ignore


def test_reporter_thread_withholds_vram_until_device_context_ready() -> None:
    """Before CUDA is initialised the thread must report Nones (include_vram False), never triggering init.

    Simulates the pre-init guard: with the stats source not ready, a VRAM-inclusive process still emits its
    interval report but with ``include_vram=False`` so it reads only RAM/FDs and never calls a device-init
    primitive off the main thread.
    """
    proc = _make_reporter_stub(includes_vram=True, sampling_ready=False)
    proc._start_memory_reporter_thread()
    try:
        deadline = time.time() + 5.0
        while len(proc.reported_include_vram) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert len(proc.reported_include_vram) >= 2
        assert all(include_vram is False for include_vram in proc.reported_include_vram)
    finally:
        proc._memory_reporter_stop.set()
    proc._memory_reporter_thread.join(timeout=2.0)  # pyrefly: ignore
