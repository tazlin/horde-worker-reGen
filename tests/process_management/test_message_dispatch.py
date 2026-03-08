"""Tests for IPC message handling in MessageDispatcher."""

from __future__ import annotations

import queue
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.messages import (
    HordeInferenceResultMessage,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import make_mock_bridge_data, make_mock_process_info


def _make_dispatcher(
    *,
    state: WorkerState | None = None,
    process_map: ProcessMap | None = None,
    horde_model_map: HordeModelMap | None = None,
    job_tracker: JobTracker | None = None,
    bridge_data: Mock | None = None,
    process_message_queue: queue.Queue | None = None,
) -> MessageDispatcher:
    """Build a MessageDispatcher with mostly-mocked dependencies."""
    if state is None:
        state = WorkerState()
    if process_map is None:
        process_map = ProcessMap({})
    if horde_model_map is None:
        horde_model_map = HordeModelMap(root={})
    if job_tracker is None:
        job_tracker = JobTracker()
    if bridge_data is None:
        bridge_data = make_mock_bridge_data()
    if process_message_queue is None:
        process_message_queue = Mock(spec=queue.Queue)

    return MessageDispatcher(
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        process_message_queue=process_message_queue,
        get_model_baseline=lambda name: None,
        get_bridge_data=lambda: bridge_data,
        on_unload_vram=Mock(),
        state=state,
    )


def _enqueue(md: MessageDispatcher, message: object) -> None:
    """Helper to enqueue a single message into the dispatcher's mock queue."""
    md._process_message_queue.empty.side_effect = [False, True]
    md._process_message_queue.get.return_value = message


class TestReceiveAndHandleProcessMessages:
    """Tests for receive_and_handle_process_messages."""

    def test_empty_queue_does_nothing(self) -> None:
        md = _make_dispatcher()
        md._process_message_queue.empty.return_value = True
        md.receive_and_handle_process_messages()

    def test_heartbeat_updates_process_map(self) -> None:
        proc = make_mock_process_info(0)
        proc.process_launch_identifier = 0
        pm = ProcessMap({0: proc})
        md = _make_dispatcher(process_map=pm)

        msg = Mock(spec=HordeProcessHeartbeatMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.heartbeat_type = Mock()
        msg.percent_complete = None
        msg.process_warning = None
        msg.info = "heartbeat"

        _enqueue(md, msg)
        md.receive_and_handle_process_messages()

    def test_memory_report_updates_process_map(self) -> None:
        proc = make_mock_process_info(0)
        proc.process_launch_identifier = 0
        pm = ProcessMap({0: proc})
        pm.on_memory_report = Mock()

        md = _make_dispatcher(process_map=pm)

        msg = Mock(spec=HordeProcessMemoryMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.ram_usage_bytes = 1024
        msg.vram_usage_bytes = 2048
        msg.vram_total_bytes = 4096
        msg.info = "memory"

        _enqueue(md, msg)
        md.receive_and_handle_process_messages()

        pm.on_memory_report.assert_called_once_with(
            process_id=0,
            ram_usage_bytes=1024,
            vram_usage_bytes=2048,
            total_vram_bytes=4096,
        )

    def test_state_change_to_inference_starting_updates_model_map(self) -> None:
        proc = make_mock_process_info(0, model_name="test_model")
        proc.process_launch_identifier = 0
        proc.loaded_horde_model_name = "test_model"
        proc.batch_amount = 1
        proc.last_process_state = HordeProcessState.PRELOADED_MODEL
        pm = ProcessMap({0: proc})
        hmm = HordeModelMap(root={})

        md = _make_dispatcher(process_map=pm, horde_model_map=hmm)

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.process_state = HordeProcessState.INFERENCE_STARTING
        msg.info = "starting inference"

        _enqueue(md, msg)
        md.receive_and_handle_process_messages()

        assert hmm.root.get("test_model") is not None

    def test_mismatched_launch_identifier_is_ignored(self) -> None:
        proc = make_mock_process_info(0)
        proc.process_launch_identifier = 5
        pm = ProcessMap({0: proc})

        md = _make_dispatcher(process_map=pm)

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 3
        msg.process_state = HordeProcessState.WAITING_FOR_JOB
        msg.info = "stale"

        _enqueue(md, msg)
        md.receive_and_handle_process_messages()

    def test_unknown_process_id_raises(self) -> None:
        md = _make_dispatcher()

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 99
        msg.process_launch_identifier = 0
        msg.process_state = HordeProcessState.WAITING_FOR_JOB
        msg.info = "state change"

        _enqueue(md, msg)
        with pytest.raises(ValueError, match="unknown process"):
            md.receive_and_handle_process_messages()

    def test_process_ending_calls_on_process_ending(self) -> None:
        proc = make_mock_process_info(0)
        proc.process_launch_identifier = 0
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
        pm = ProcessMap({0: proc})
        pm.on_process_ending = Mock()
        pm.on_process_state_change = Mock()

        md = _make_dispatcher(process_map=pm)

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.process_state = HordeProcessState.PROCESS_ENDING
        msg.info = "ending"

        _enqueue(md, msg)
        md.receive_and_handle_process_messages()

        pm.on_process_ending.assert_called_once_with(process_id=0)


class TestHandleInferenceResult:
    """Tests for _handle_inference_result."""

    def test_inference_result_moves_job_to_safety_check(self) -> None:
        proc = make_mock_process_info(0)
        proc.process_launch_identifier = 0
        pm = ProcessMap({0: proc})
        jt = JobTracker()

        job = Mock()
        job.id_ = "test-id"
        job.model = "stable_diffusion"
        job.payload = Mock()
        job.payload.n_iter = 1

        job_info = Mock()
        job_info.sdk_api_job_info = job
        jt.jobs_lookup[job] = job_info
        jt.jobs_in_progress.append(job)

        md = _make_dispatcher(process_map=pm, job_tracker=jt)

        msg = Mock(spec=HordeInferenceResultMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.sdk_api_job_info = job
        msg.time_elapsed = 5.0
        msg.info = "50%"
        msg.state = Mock()
        msg.state.__eq__ = lambda self, other: False  # not faulted
        msg.job_image_results = [Mock()]
        msg.faults_count = 0

        _enqueue(md, msg)
        md.receive_and_handle_process_messages()

        assert job not in jt.jobs_in_progress
        assert job_info in jt.jobs_pending_safety_check

    def test_faulted_inference_result_moves_to_pending_submit(self) -> None:
        from horde_sdk.ai_horde_api import GENERATION_STATE

        proc = make_mock_process_info(0)
        proc.process_launch_identifier = 0
        pm = ProcessMap({0: proc})
        jt = JobTracker()

        job = Mock()
        job.id_ = "test-id"
        job.model = "stable_diffusion"

        job_info = Mock()
        job_info.sdk_api_job_info = job
        jt.jobs_lookup[job] = job_info
        jt.jobs_in_progress.append(job)

        md = _make_dispatcher(process_map=pm, job_tracker=jt)

        msg = Mock(spec=HordeInferenceResultMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.sdk_api_job_info = job
        msg.time_elapsed = 5.0
        msg.info = "faulted"
        msg.state = GENERATION_STATE.faulted
        msg.job_image_results = None
        msg.faults_count = 1

        _enqueue(md, msg)
        md.receive_and_handle_process_messages()

        assert job_info in jt.jobs_pending_submit

    def test_inference_result_unknown_job_is_handled(self) -> None:
        proc = make_mock_process_info(0)
        proc.process_launch_identifier = 0
        pm = ProcessMap({0: proc})

        md = _make_dispatcher(process_map=pm)

        job = Mock()
        job.id_ = "unknown-job"
        job.model = "stable_diffusion"

        msg = Mock(spec=HordeInferenceResultMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.sdk_api_job_info = job
        msg.time_elapsed = 1.0
        msg.info = "done"
        msg.state = Mock()
        msg.job_image_results = None
        msg.faults_count = 0

        _enqueue(md, msg)
        md.receive_and_handle_process_messages()


class TestHandleSafetyResult:
    """Tests for _handle_safety_result."""

    def test_safety_result_moves_job_to_pending_submit(self) -> None:
        jt = JobTracker()

        job = Mock()
        job.id_ = "safety-test-id"

        image_result = Mock()
        image_result.generation_faults = []

        job_info = Mock()
        job_info.sdk_api_job_info = job
        job_info.job_image_results = [image_result]
        jt.jobs_being_safety_checked.append(job_info)
        jt.job_faults[job.id_] = []

        md = _make_dispatcher(job_tracker=jt)

        safety_eval = Mock()
        safety_eval.failed = False
        safety_eval.replacement_image_base64 = None
        safety_eval.is_csam = False
        safety_eval.is_nsfw = False

        md._handle_safety_result(
            Mock(
                job_id=job.id_,
                safety_evaluations=[safety_eval],
                time_elapsed=1.0,
            ),
        )

        assert job_info in jt.jobs_pending_submit
        assert job_info not in jt.jobs_being_safety_checked

    def test_safety_result_not_found_logs_error(self) -> None:
        md = _make_dispatcher()

        md._handle_safety_result(
            Mock(
                job_id="nonexistent",
                safety_evaluations=[],
                time_elapsed=0.5,
            ),
        )


class TestHandleModelStateChange:
    """Tests for _handle_model_state_change."""

    def test_model_loaded_in_ram_updates_maps(self) -> None:
        proc = make_mock_process_info(0)
        pm = ProcessMap({0: proc})
        hmm = HordeModelMap(root={})

        md = _make_dispatcher(process_map=pm, horde_model_map=hmm)

        md._handle_model_state_change(
            Mock(
                horde_model_name="test_model",
                horde_model_state=ModelLoadState.LOADED_IN_RAM,
                process_id=0,
                time_elapsed=2.5,
                info="loaded",
            ),
        )

        assert hmm.root.get("test_model") is not None

    def test_model_on_disk_does_not_update_process_map(self) -> None:
        proc = make_mock_process_info(0)
        pm = ProcessMap({0: proc})
        pm.on_model_load_state_change = Mock()

        md = _make_dispatcher(process_map=pm)

        md._handle_model_state_change(
            Mock(
                horde_model_name="test_model",
                horde_model_state=ModelLoadState.ON_DISK,
                process_id=0,
                time_elapsed=None,
                info="unloaded",
            ),
        )

        pm.on_model_load_state_change.assert_not_called()


class TestHandleAuxModelStateChange:
    """Tests for _handle_aux_model_state_change."""

    def test_downloading_aux_model_updates_last_job(self) -> None:
        proc = make_mock_process_info(0)
        pm = ProcessMap({0: proc})
        pm.on_last_job_reference_change = Mock()

        md = _make_dispatcher(process_map=pm)

        job = Mock()

        md._handle_aux_model_state_change(
            Mock(
                process_state=HordeProcessState.DOWNLOADING_AUX_MODEL,
                process_id=0,
                sdk_api_job_info=job,
                time_elapsed=None,
            ),
        )

        pm.on_last_job_reference_change.assert_called_once_with(
            process_id=0,
            last_job_referenced=job,
        )

    def test_download_aux_complete_records_time(self) -> None:
        jt = JobTracker()
        job = Mock()
        job.id_ = "aux-test"
        job_info = Mock()
        jt.jobs_lookup[job] = job_info

        md = _make_dispatcher(job_tracker=jt)

        md._handle_aux_model_state_change(
            Mock(
                process_state=HordeProcessState.DOWNLOAD_AUX_COMPLETE,
                process_id=0,
                sdk_api_job_info=job,
                time_elapsed=3.5,
            ),
        )

        assert job_info.time_to_download_aux_models == 3.5
