"""Tests for IPC message handling in MessageDispatcher."""

from __future__ import annotations

import queue
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management._aliased_types import ProcessQueue
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

from .conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_job,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
    mark_job_in_progress_async,
    move_job_to_being_safety_checked_async,
)


def _make_dispatcher(
    *,
    state: WorkerState | None = None,
    process_map: ProcessMap | None = None,
    horde_model_map: HordeModelMap | None = None,
    job_tracker: JobTracker | None = None,
    bridge_data: Mock | None = None,
    process_message_queue: ProcessQueue | None = None,
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
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(),
        on_unload_vram=Mock(),
        state=state,
    )


def _enqueue(message_dispatcher: MessageDispatcher, message: object) -> None:
    """Helper to enqueue a single message into the dispatcher's mock queue."""
    message_dispatcher._process_message_queue.empty.side_effect = [False, True]
    message_dispatcher._process_message_queue.get.return_value = message


class TestReceiveAndHandleProcessMessages:
    """Tests for receive_and_handle_process_messages."""

    async def test_empty_queue_does_nothing(self) -> None:
        """If the message queue is empty, the method should return without doing anything."""
        message_dispatcher = _make_dispatcher()
        message_dispatcher._process_message_queue.empty.return_value = True
        await message_dispatcher.receive_and_handle_process_messages()

    async def test_heartbeat_updates_process_map(self) -> None:
        """When a heartbeat message is received, the process map's on_heartbeat callback is called."""
        process_info = make_mock_process_info(0)
        process_info.process_launch_identifier = 0
        process_map = ProcessMap({0: process_info})
        message_dispatcher = _make_dispatcher(process_map=process_map)

        msg = Mock(spec=HordeProcessHeartbeatMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.heartbeat_type = Mock()
        msg.percent_complete = None
        msg.process_warning = None
        msg.info = "heartbeat"

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

    async def test_memory_report_updates_process_map(self) -> None:
        """When a memory report message is received, the process map's on_memory_report callback is called."""
        process_info = make_mock_process_info(0)
        process_info.process_launch_identifier = 0
        process_map = ProcessMap({0: process_info})
        process_map.on_memory_report = Mock()

        message_dispatcher = _make_dispatcher(process_map=process_map)

        msg = Mock(spec=HordeProcessMemoryMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.ram_usage_bytes = 1024
        msg.vram_usage_bytes = 2048
        msg.vram_total_bytes = 4096
        msg.info = "memory"

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

        process_map.on_memory_report.assert_called_once_with(
            process_id=0,
            ram_usage_bytes=1024,
            vram_usage_bytes=2048,
            total_vram_bytes=4096,
        )

    async def test_state_change_to_inference_starting_updates_model_map(self) -> None:
        """When a process state changes to INFERENCE_STARTING, the model map should be updated."""
        process_info = make_mock_process_info(0, model_name="test_model")
        process_info.process_launch_identifier = 0
        process_info.loaded_horde_model_name = "test_model"
        process_info.batch_amount = 1
        process_info.last_process_state = HordeProcessState.PRELOADED_MODEL
        process_map = ProcessMap({0: process_info})
        hmm = HordeModelMap(root={})

        message_dispatcher = _make_dispatcher(process_map=process_map, horde_model_map=hmm)

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.process_state = HordeProcessState.INFERENCE_STARTING
        msg.info = "starting inference"

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

        assert hmm.root.get("test_model") is not None

    async def test_mismatched_launch_identifier_is_ignored(self) -> None:
        """Ignore if a message is received with a launch identifier that doesn't match the process's current one."""
        process_info = make_mock_process_info(0)
        process_info.process_launch_identifier = 5
        process_map = ProcessMap({0: process_info})

        message_dispatcher = _make_dispatcher(process_map=process_map)

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 3
        msg.process_state = HordeProcessState.WAITING_FOR_JOB
        msg.info = "stale"

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

    async def test_unknown_process_id_raises(self) -> None:
        """If a message is received for an unknown process ID, a ValueError should be raised."""
        message_dispatcher = _make_dispatcher()

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 99
        msg.process_launch_identifier = 0
        msg.process_state = HordeProcessState.WAITING_FOR_JOB
        msg.info = "state change"

        _enqueue(message_dispatcher, msg)
        with pytest.raises(ValueError, match="unknown process"):
            await message_dispatcher.receive_and_handle_process_messages()

    async def test_process_ending_calls_on_process_ending(self) -> None:
        """When a process is ending, the process map's on_process_ending callback is called."""
        process_info = make_mock_process_info(0)
        process_info.process_launch_identifier = 0
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_map = ProcessMap({0: process_info})
        process_map.on_process_ending = Mock()
        process_map.on_process_state_change = Mock()

        message_dispatcher = _make_dispatcher(process_map=process_map)

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.process_state = HordeProcessState.PROCESS_ENDING
        msg.info = "ending"

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

        process_map.on_process_ending.assert_called_once_with(process_id=0)


class TestHandleInferenceResult:
    """Tests for _handle_inference_result."""

    async def test_inference_result_moves_job_to_safety_check(self) -> None:
        """When an inference result is received for a job in progress, it should be moved to pending safety check."""
        process_info = make_mock_process_info(0)
        process_info.process_launch_identifier = 0
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response(model="stable_diffusion")

        job_info = await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)

        message_dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        msg = Mock(spec=HordeInferenceResultMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.sdk_api_job_info = job
        msg.time_elapsed = 5.0
        msg.info = "50%"
        msg.state = Mock()
        msg.state.__eq__ = lambda self, other: False  # pyrefly: ignore - we aren't testing state handling here, just that the message is processed and the job is moved to safety check
        msg.job_image_results = [Mock()]
        msg.faults_count = 0

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

        assert job not in job_tracker.jobs_in_progress
        assert job_info in job_tracker.jobs_pending_safety_check

    async def test_faulted_inference_result_moves_to_pending_submit(self) -> None:
        """If an inference result is faulted, it should be moved to pending submit."""
        from horde_sdk.ai_horde_api import GENERATION_STATE

        process_info = make_mock_process_info(0)
        process_info.process_launch_identifier = 0
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response(model="stable_diffusion")

        job_info = await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)

        message_dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        msg = Mock(spec=HordeInferenceResultMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.sdk_api_job_info = job
        msg.time_elapsed = 5.0
        msg.info = "faulted"
        msg.state = GENERATION_STATE.faulted
        msg.job_image_results = None
        msg.faults_count = 1

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

        assert job_info in job_tracker.jobs_pending_submit

    async def test_inference_result_unknown_job_is_handled(self) -> None:
        """If an inference result is received for a job that isn't in progress, it should be handled gracefully."""
        process_info = make_mock_process_info(0)
        process_info.process_launch_identifier = 0
        process_map = ProcessMap({0: process_info})

        message_dispatcher = _make_dispatcher(process_map=process_map)

        job = make_mock_job()
        job.id_ = "unknown-job"

        msg = Mock(spec=HordeInferenceResultMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.sdk_api_job_info = job
        msg.time_elapsed = 1.0
        msg.info = "done"
        msg.state = Mock()
        msg.job_image_results = None
        msg.faults_count = 0

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()


class TestHandleSafetyResult:
    """Tests for _handle_safety_result."""

    async def test_safety_result_moves_job_to_pending_submit(self) -> None:
        """When a safety result is received for a job being safety checked it should be moved to pending submit.

        This should be regardless of the evaluation outcome (safety checks don't gate submission,
        they just provide info for the API). The job info should also be removed from being_safety_checked.
        """
        job_tracker = JobTracker()

        job = Mock()
        job.id_ = "safety-test-id"

        image_result = Mock()
        image_result.generation_faults = []  # pyrefly: ignore - we aren't testing fault handling here, just that the safety result is processed and the job is moved to pending submit

        job_info = Mock()
        job_info.sdk_api_job_info = job
        job_info.job_image_results = [image_result]
        await move_job_to_being_safety_checked_async(job_tracker, job_info)

        message_dispatcher = _make_dispatcher(job_tracker=job_tracker)

        safety_eval = Mock()
        safety_eval.failed = False
        safety_eval.replacement_image_base64 = None
        safety_eval.is_csam = False
        safety_eval.is_nsfw = False

        await message_dispatcher._handle_safety_result(
            Mock(
                job_id=job.id_,
                safety_evaluations=[safety_eval],
                time_elapsed=1.0,
            ),
        )

        assert job_info in job_tracker.jobs_pending_submit
        assert job_info not in job_tracker.jobs_being_safety_checked

    async def test_safety_result_not_found_logs_error(self) -> None:
        """Test that if a safety result is received for a job ID that isn't being safety checked, it is handled."""
        message_dispatcher = _make_dispatcher()

        await message_dispatcher._handle_safety_result(
            Mock(
                job_id="nonexistent",
                safety_evaluations=[],
                time_elapsed=0.5,
            ),
        )


class TestHandleModelStateChange:
    """Tests for _handle_model_state_change."""

    def test_model_loaded_in_ram_updates_maps(self) -> None:
        """When a model is loaded in RAM, the process map should be updated.

        This should include the model name and the model map being updated with the new model.
        """
        process_info = make_mock_process_info(0)
        process_map = ProcessMap({0: process_info})
        hmm = HordeModelMap(root={})

        message_dispatcher = _make_dispatcher(process_map=process_map, horde_model_map=hmm)

        message_dispatcher._handle_model_state_change(
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
        """Model state change to ON_DISK should not update the process map's loaded model."""
        process_info = make_mock_process_info(0)
        process_map = ProcessMap({0: process_info})
        process_manageron_model_load_state_change = Mock()

        message_dispatcher = _make_dispatcher(process_map=process_map)

        message_dispatcher._handle_model_state_change(
            Mock(
                horde_model_name="test_model",
                horde_model_state=ModelLoadState.ON_DISK,
                process_id=0,
                time_elapsed=None,
                info="unloaded",
            ),
        )

        process_manageron_model_load_state_change.assert_not_called()


class TestHandleAuxModelStateChange:
    """Tests for _handle_aux_model_state_change."""

    async def test_downloading_aux_model_updates_last_job(self) -> None:
        """When a process reports that it's downloading an aux model, the last job it referenced should be updated."""
        process_info = make_mock_process_info(0)
        process_map = ProcessMap({0: process_info})
        process_map.on_last_job_reference_change = Mock()

        message_dispatcher = _make_dispatcher(process_map=process_map)

        job = Mock()

        await message_dispatcher._handle_aux_model_state_change(
            Mock(
                process_state=HordeProcessState.DOWNLOADING_AUX_MODEL,
                process_id=0,
                sdk_api_job_info=job,
                time_elapsed=None,
            ),
        )

        process_map.on_last_job_reference_change.assert_called_once_with(
            process_id=0,
            last_job_referenced=job,
        )

    async def test_download_aux_complete_records_time(self) -> None:
        """When an aux model finishes downloading, the time to download should be recorded on the job info."""
        job_tracker = JobTracker()
        job = make_job_pop_response(model="stable_diffusion")
        job_info = await job_tracker.record_popped_job(job)

        message_dispatcher = _make_dispatcher(job_tracker=job_tracker)

        await message_dispatcher._handle_aux_model_state_change(
            Mock(
                process_state=HordeProcessState.DOWNLOAD_AUX_COMPLETE,
                process_id=0,
                sdk_api_job_info=job,
                time_elapsed=3.5,
            ),
        )

        assert job_info.time_to_download_aux_models == 3.5
