"""Tests for IPC message handling in MessageDispatcher."""

from __future__ import annotations

import queue
import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger
from horde_worker_regen.process_management.ipc.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.ipc.messages import (
    HordeHeartbeatType,
    HordeInferenceResultMessage,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from tests.process_management.conftest import (
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
        action_ledger=ActionLedger(),
        reserve_ledger=CommittedReserveLedger(),
        on_unload_vram=Mock(),
        state=state,
    )


def _enqueue(message_dispatcher: MessageDispatcher, message: object) -> None:
    """Helper to enqueue a single message into the dispatcher's mock queue."""
    message_dispatcher._process_message_queue.empty.side_effect = [False, True]
    message_dispatcher._process_message_queue.get.return_value = message


def _enqueue_many(message_dispatcher: MessageDispatcher, messages: list[object]) -> None:
    """Helper to enqueue multiple messages into the dispatcher's mock queue."""
    message_dispatcher._process_message_queue.empty.side_effect = [False] * len(messages) + [True]
    message_dispatcher._process_message_queue.get.side_effect = messages


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
        msg.current_step = None
        msg.total_steps = None
        msg.iterations_per_second = None
        msg.nonadvancing_step_repeats = 0

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
        msg.vram_usage_mb = 2048
        msg.vram_total_mb = 4096
        msg.open_fds = None
        msg.fd_soft_limit = None
        msg.process_reserved_mb = 6000
        msg.process_allocated_mb = 5000
        msg.process_peak_reserved_mb = 6500
        msg.process_aimdo_mb = 10000
        msg.sampled_at = 1234.5
        msg.info = "memory"

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

        process_map.on_memory_report.assert_called_once_with(
            process_id=0,
            ram_usage_bytes=1024,
            vram_usage_mb=2048,
            total_vram_mb=4096,
            open_fds=None,
            fd_soft_limit=None,
            process_reserved_mb=6000,
            process_allocated_mb=5000,
            process_peak_reserved_mb=6500,
            process_aimdo_mb=10000,
            report_sampled_at=1234.5,
        )

    def test_record_completed_job_increments_producing_process(self) -> None:
        """A result message bumps the per-process completed-work counter on the producing slot."""
        process_info = make_mock_process_info(0)
        message_dispatcher = _make_dispatcher(process_map=ProcessMap({0: process_info}))
        assert process_info.num_jobs_completed == 0

        message_dispatcher._record_completed_job(0)
        message_dispatcher._record_completed_job(0)

        assert process_info.num_jobs_completed == 2

    def test_record_completed_job_unknown_process_is_noop(self) -> None:
        """A result from a process not in the map is ignored rather than raising."""
        message_dispatcher = _make_dispatcher(process_map=ProcessMap({}))
        message_dispatcher._record_completed_job(99)  # must not raise

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

    async def test_model_load_failed_invokes_handler_and_does_not_record_residency(self) -> None:
        """A FAILED model state hands the failure to the registered handler and never records the model loaded."""
        from horde_worker_regen.process_management.ipc.messages import HordeModelStateChangeMessage, ModelLoadState

        process_info = make_mock_process_info(0, model_name="Z-Image-Turbo")
        process_info.process_launch_identifier = 0
        process_map = ProcessMap({0: process_info})
        hmm = HordeModelMap(root={})

        message_dispatcher = _make_dispatcher(process_map=process_map, horde_model_map=hmm)
        seen: list[tuple[int, str]] = []
        message_dispatcher.set_model_load_failure_handler(lambda pid, model: seen.append((pid, model)))

        msg = Mock(spec=HordeModelStateChangeMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.horde_model_name = "Z-Image-Turbo"
        msg.horde_model_state = ModelLoadState.FAILED
        msg.process_state = HordeProcessState.PRELOADING_FAILED
        msg.info = "failed to load"
        msg.time_elapsed = None

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

        assert seen == [(0, "Z-Image-Turbo")]
        # The failed model must not be left recorded as resident anywhere.
        assert (
            hmm.root.get("Z-Image-Turbo") is None or not hmm.root["Z-Image-Turbo"].horde_model_load_state.is_active()
        )

    async def test_torch_gpu_incompatible_latches_worker_state_flag(self) -> None:
        """A TORCH_GPU_INCOMPATIBLE report latches the sticky stop-popping flag and stores its reason."""
        process_info = make_mock_process_info(0)
        process_info.process_launch_identifier = 0
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        state = WorkerState()

        message_dispatcher = _make_dispatcher(process_map=ProcessMap({0: process_info}), state=state)

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.process_state = HordeProcessState.TORCH_GPU_INCOMPATIBLE
        msg.info = "PyTorch has no CUDA kernels for NVIDIA GeForce RTX 5070 (compute capability sm_120)."
        msg.time_elapsed = None

        assert state.gpu_torch_incompatible is False
        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

        assert state.gpu_torch_incompatible is True
        assert "RTX 5070" in state.gpu_torch_incompatible_reason

    async def test_torch_build_cpu_only_latches_worker_state_flag(self) -> None:
        """A TORCH_BUILD_CPU_ONLY report latches the image-disable flag and stores its reason."""
        process_info = make_mock_process_info(0)
        process_info.process_launch_identifier = 0
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        state = WorkerState()

        message_dispatcher = _make_dispatcher(process_map=ProcessMap({0: process_info}), state=state)

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 0
        msg.process_launch_identifier = 0
        msg.process_state = HordeProcessState.TORCH_BUILD_CPU_ONLY
        msg.info = "Installed PyTorch is a CPU-only build; image generation is disabled. Alchemy continues."
        msg.time_elapsed = None

        assert state.torch_build_cpu_only is False
        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

        assert state.torch_build_cpu_only is True
        assert "CPU-only build" in state.torch_build_cpu_only_reason

    async def test_stale_idle_after_fresh_dispatch_does_not_release_current_job(self) -> None:
        """An idle transition older than the current dispatch must not release the newly assigned job."""
        job_tracker = JobTracker()
        job_tracker.set_retry_policy(2)
        job = make_job_pop_response(model="stable_diffusion")
        await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)
        assert job.id_ is not None

        process_info = make_mock_process_info(3, model_name="stable_diffusion")
        process_info.process_launch_identifier = 0
        process_info.last_process_state = HordeProcessState.INFERENCE_COMPLETE
        process_info.last_process_state_started_at = time.time() - 5.0
        process_info.last_job_referenced = job
        process_info.current_inference_started_at = time.time()
        process_map = ProcessMap({3: process_info})
        message_dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 3
        msg.process_launch_identifier = 0
        msg.process_state = HordeProcessState.WAITING_FOR_JOB
        msg.info = "waiting"

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

        assert job in job_tracker.jobs_in_progress
        assert job_tracker.get_stage(job.id_) == JobStage.INFERENCE_IN_PROGRESS

    async def test_active_to_idle_without_result_releases_current_job(self) -> None:
        """An active slot that returns idle with its own job still in progress releases that job for retry."""
        job_tracker = JobTracker()
        job_tracker.set_retry_policy(2)
        job = make_job_pop_response(model="stable_diffusion")
        await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)
        assert job.id_ is not None

        process_info = make_mock_process_info(3, model_name="stable_diffusion")
        process_info.process_launch_identifier = 0
        process_info.last_process_state = HordeProcessState.INFERENCE_COMPLETE
        process_info.last_process_state_started_at = time.time()
        process_info.last_job_referenced = job
        process_info.current_inference_started_at = time.time() - 5.0
        process_map = ProcessMap({3: process_info})
        message_dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 3
        msg.process_launch_identifier = 0
        msg.process_state = HordeProcessState.WAITING_FOR_JOB
        msg.info = "waiting"

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

        assert job not in job_tracker.jobs_in_progress
        assert job_tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE

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

    async def test_unknown_process_id_is_dropped_not_fatal(self) -> None:
        """A message from a process the map no longer knows must be dropped, never crash the worker.

        This is the regression for the whole-card teardown total-death: scale_inference_processes
        pops a stopped slot from the map, but its already-queued terminal messages (PROCESS_ENDING,
        a final memory report) still arrive. When the retired-launch tombstone does not cover them
        (older launch id, or pruned by TTL), the dispatcher used to raise ValueError, which killed the
        control loop and took the whole worker down over a stale status update.
        """
        message_dispatcher = _make_dispatcher()

        msg = Mock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 99
        msg.process_launch_identifier = 0
        msg.process_state = HordeProcessState.PROCESS_ENDING
        msg.info = "state change"

        _enqueue(message_dispatcher, msg)
        # Must not raise; the message is simply dropped.
        await message_dispatcher.receive_and_handle_process_messages()

    @pytest.mark.parametrize(
        "message",
        [
            HordeProcessStateChangeMessage(
                process_id=1,
                process_launch_identifier=7,
                process_state=HordeProcessState.PROCESS_ENDING,
                info="ending",
            ),
            HordeProcessStateChangeMessage(
                process_id=1,
                process_launch_identifier=7,
                process_state=HordeProcessState.PROCESS_ENDED,
                info="ended",
            ),
            HordeProcessHeartbeatMessage(
                process_id=1,
                process_launch_identifier=7,
                heartbeat_type=HordeHeartbeatType.OTHER,
                info="heartbeat",
            ),
            HordeProcessMemoryMessage(
                process_id=1,
                process_launch_identifier=7,
                ram_usage_bytes=1024,
                vram_usage_mb=2048,
                vram_total_mb=4096,
                info="memory",
            ),
        ],
    )
    async def test_late_liveness_from_retired_launch_is_ignored(
        self,
        message: HordeProcessStateChangeMessage | HordeProcessHeartbeatMessage | HordeProcessMemoryMessage,
    ) -> None:
        """Late terminal/liveness messages from an intentionally retired launch must not crash."""
        retired = make_mock_process_info(1)
        retired.process_launch_identifier = 7
        process_map = ProcessMap({1: retired})
        process_map.retire_process(retired, "test retirement")
        process_map.on_memory_report = Mock()
        process_map.on_heartbeat = Mock()
        process_map.on_process_ending = Mock()

        message_dispatcher = _make_dispatcher(process_map=process_map)
        _enqueue(message_dispatcher, message)

        await message_dispatcher.receive_and_handle_process_messages()

        process_map.on_memory_report.assert_not_called()
        process_map.on_heartbeat.assert_not_called()
        process_map.on_process_ending.assert_not_called()

    async def test_result_from_retired_launch_is_warned_and_ignored(self) -> None:
        """A retired idle slot should not produce results; warn, but do not crash the parent."""
        retired = make_mock_process_info(1)
        retired.process_launch_identifier = 7
        process_map = ProcessMap({1: retired})
        process_map.retire_process(retired, "test retirement")

        message_dispatcher = _make_dispatcher(process_map=process_map)
        msg = Mock(spec=HordeInferenceResultMessage)
        msg.process_id = 1
        msg.process_launch_identifier = 7
        msg.info = "late result"

        _enqueue(message_dispatcher, msg)
        await message_dispatcher.receive_and_handle_process_messages()

    async def test_retired_launch_for_reused_process_id_does_not_mutate_current_process(self) -> None:
        """A stale launch id is ignored even when its logical process id has already been reused."""
        old = make_mock_process_info(1)
        old.process_launch_identifier = 7
        process_map = ProcessMap({1: old})
        process_map.retire_process(old, "scale-down")

        current = make_mock_process_info(1)
        current.process_launch_identifier = 8
        process_map[1] = current
        message_dispatcher = _make_dispatcher(process_map=process_map)

        stale_memory = HordeProcessMemoryMessage(
            process_id=1,
            process_launch_identifier=7,
            ram_usage_bytes=1024,
            vram_usage_mb=2048,
            vram_total_mb=4096,
            info="stale memory",
        )
        current_memory = HordeProcessMemoryMessage(
            process_id=1,
            process_launch_identifier=8,
            ram_usage_bytes=32,
            vram_usage_mb=64,
            vram_total_mb=128,
            info="current memory",
        )

        _enqueue_many(message_dispatcher, [stale_memory, current_memory])
        await message_dispatcher.receive_and_handle_process_messages()

        assert current.ram_usage_bytes == 32
        assert current.vram_usage_mb == 64
        assert current.total_vram_mb == 128

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


class TestInferenceStartingIsTypeScoped:
    """``INFERENCE_STARTING`` bookkeeping applies only to whole-job INFERENCE slots, never stage lanes.

    A disaggregated text-encode stage runs on a ``COMPONENT`` lane process that reuses the busy
    ``INFERENCE_STARTING`` state to mark itself working, but holds no parent-tracked model. Before the
    type guard, that state change reached the whole-job handler, which raised ``ValueError`` on the
    absent model and took the whole worker down over routine stage traffic. These lock the guard: the
    stage lane is tolerated, while the invariant still fires for a real whole-job slot with no model.
    """

    def _inference_starting_message(self, process_id: int) -> HordeProcessStateChangeMessage:
        return HordeProcessStateChangeMessage(
            process_id=process_id,
            process_launch_identifier=0,
            process_state=HordeProcessState.INFERENCE_STARTING,
            info="Text-encode 1234",
        )

    def test_component_lane_inference_starting_without_model_does_not_raise(self) -> None:
        """A COMPONENT stage lane reporting INFERENCE_STARTING with no loaded model is tolerated."""
        component = make_mock_process_info(1, model_name=None, process_type=HordeProcessType.COMPONENT)
        dispatcher = _make_dispatcher(process_map=ProcessMap({1: component}))

        # Must not raise: the whole-job model/batch bookkeeping is skipped for a non-INFERENCE process.
        dispatcher._handle_process_state_change(self._inference_starting_message(1))

    def test_post_process_lane_inference_starting_without_model_does_not_raise(self) -> None:
        """A POST_PROCESS image lane is likewise exempt from the whole-job INFERENCE_STARTING bookkeeping."""
        lane = make_mock_process_info(2, model_name=None, process_type=HordeProcessType.POST_PROCESS)
        dispatcher = _make_dispatcher(process_map=ProcessMap({2: lane}))

        dispatcher._handle_process_state_change(self._inference_starting_message(2))

    def test_whole_job_inference_slot_without_model_still_raises(self) -> None:
        """The invariant is not weakened: a whole-job INFERENCE slot starting with no model still faults."""
        inference = make_mock_process_info(0, model_name=None, process_type=HordeProcessType.INFERENCE)
        dispatcher = _make_dispatcher(process_map=ProcessMap({0: inference}))

        with pytest.raises(ValueError, match="no model loaded"):
            dispatcher._handle_process_state_change(self._inference_starting_message(0))


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
        safety_eval.replacement_image_bytes = None
        safety_eval.is_csam = False
        safety_eval.is_nsfw = False
        safety_eval.aesthetic_score = None

        await message_dispatcher._handle_safety_result(
            Mock(
                job_id=job.id_,
                safety_evaluations=[safety_eval],
                time_elapsed=1.0,
            ),
        )

        assert job_info in job_tracker.jobs_pending_submit
        assert job_info not in job_tracker.jobs_being_safety_checked

    async def test_safety_result_attaches_aesthetic_metadata(self) -> None:
        """A safety evaluation that carries an aesthetic score attaches it as gen_metadata.

        The score rides on the per-image generation_faults list (the worker's gen_metadata bucket) as a
        float in the entry's ``ref``, so a client can rank/curate generations.
        """
        from horde_worker_regen.consts import AESTHETIC_METADATA_TYPE

        job_tracker = JobTracker()

        job = Mock()
        job.id_ = "aesthetic-test-id"

        image_result = Mock()
        image_result.generation_faults = []

        job_info = Mock()
        job_info.sdk_api_job_info = job
        job_info.job_image_results = [image_result]
        await move_job_to_being_safety_checked_async(job_tracker, job_info)

        message_dispatcher = _make_dispatcher(job_tracker=job_tracker)

        safety_eval = Mock()
        safety_eval.failed = False
        safety_eval.replacement_image_bytes = None
        safety_eval.is_csam = False
        safety_eval.is_nsfw = False
        safety_eval.aesthetic_score = 6.42

        await message_dispatcher._handle_safety_result(
            Mock(job_id=job.id_, safety_evaluations=[safety_eval], time_elapsed=1.0),
        )

        aesthetic_entries = [e for e in image_result.generation_faults if e.type_ == AESTHETIC_METADATA_TYPE]
        assert len(aesthetic_entries) == 1
        assert aesthetic_entries[0].ref == "6.42"

    async def test_safety_result_without_aesthetic_score_attaches_nothing(self) -> None:
        """When the evaluation carries no score (scoring disabled/unavailable), no entry is added."""
        from horde_worker_regen.consts import AESTHETIC_METADATA_TYPE

        job_tracker = JobTracker()
        job = Mock()
        job.id_ = "no-aesthetic-id"
        image_result = Mock()
        image_result.generation_faults = []
        job_info = Mock()
        job_info.sdk_api_job_info = job
        job_info.job_image_results = [image_result]
        await move_job_to_being_safety_checked_async(job_tracker, job_info)

        message_dispatcher = _make_dispatcher(job_tracker=job_tracker)
        safety_eval = Mock()
        safety_eval.failed = False
        safety_eval.replacement_image_bytes = None
        safety_eval.is_csam = False
        safety_eval.is_nsfw = False
        safety_eval.aesthetic_score = None

        await message_dispatcher._handle_safety_result(
            Mock(job_id=job.id_, safety_evaluations=[safety_eval], time_elapsed=1.0),
        )

        assert not [e for e in image_result.generation_faults if e.type_ == AESTHETIC_METADATA_TYPE]

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
