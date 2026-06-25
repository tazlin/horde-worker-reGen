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
from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeInferenceControlMessage,
    HordeProcessHeartbeatMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcess
from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess
from tests.process_management.conftest import make_job_pop_response


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


def _make_inference_proc_for_start(active_model: str) -> HordeInferenceProcess:
    """Build a bare HordeInferenceProcess wired only for the START_INFERENCE handler path.

    A real construction spins up HordeLib/SharedModelManager; the START_INFERENCE branch only
    touches a handful of methods, so we mock those and pre-set the already-resident model.
    """
    proc = object.__new__(HordeInferenceProcess)
    proc._active_model_name = active_model
    proc.preload_model = Mock()  # pyrefly: ignore
    proc.download_aux_models = Mock()  # pyrefly: ignore
    proc.on_horde_model_state_change = Mock()  # pyrefly: ignore
    proc.start_inference = Mock(return_value=[object()])  # pyrefly: ignore
    proc.send_inference_result_message = Mock()  # pyrefly: ignore
    return proc


def test_start_inference_for_resident_model_still_downloads_aux_models() -> None:
    """A job reusing an already-loaded model must still fetch its LoRAs before inference.

    When the base model is already resident the scheduler dispatches inference without a fresh preload,
    so the only heartbeat-protected aux download (inside ``preload_model``) is skipped. The job's
    LoRAs then download lazily inside ``basic_inference`` while the slot reads INFERENCE_STARTING,
    which the parent's ``inference_step_timeout`` watchdog mistakes for a hang and kills. The handler must run the
    aux download itself in that case.
    """
    model = "CyberRealistic Pony"
    proc = _make_inference_proc_for_start(model)

    job = make_job_pop_response(
        model=model,
        loras=[LorasPayloadEntry(name="2498503", model=0.9, clip=1.0, is_version=True)],
    )
    message = HordeInferenceControlMessage(
        control_flag=HordeControlFlag.START_INFERENCE,
        horde_model_name=model,
        sdk_api_job_info=job,
    )

    proc._receive_and_handle_control_message(message)

    proc.preload_model.assert_not_called()  # pyrefly: ignore
    # The resident path forwards the per-job aux-download deadline carried on the control message (None here).
    proc.download_aux_models.assert_called_once_with(job, aux_download_deadline_seconds=None)  # pyrefly: ignore
    proc.start_inference.assert_called_once()  # pyrefly: ignore


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
    proc._last_periodic_memory_report_time = 0.0
    proc._memory_report_interval = 5.0
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


def test_progress_callback_emits_periodic_vram_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mid-inference progress callback refreshes VRAM when the report interval has elapsed."""
    _install_fake_hordelib_api(monkeypatch)
    proc = _make_inference_proc_for_progress()

    proc.progress_callback(cast(Any, _progress_report(step=3)))

    cast(Mock, proc.send_memory_report_message).assert_called_once_with(include_vram=True)


def test_progress_callback_throttles_vram_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated progress callbacks inside the interval must not poll VRAM every step."""
    _install_fake_hordelib_api(monkeypatch)
    proc = _make_inference_proc_for_progress()

    proc.progress_callback(cast(Any, _progress_report(step=3)))
    proc.progress_callback(cast(Any, _progress_report(step=4)))

    cast(Mock, proc.send_memory_report_message).assert_called_once_with(include_vram=True)


def test_sampling_complete_emits_one_boundary_vram_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sampling-complete boundary sends a fresh report and resets the periodic throttle."""
    _install_fake_hordelib_api(monkeypatch)
    proc = _make_inference_proc_for_progress()

    proc.progress_callback(cast(Any, _progress_report(step=20, total=20)))

    cast(Mock, proc.send_memory_report_message).assert_called_once_with(include_vram=True)
