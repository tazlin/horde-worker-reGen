"""Tests for deadlock detection in MessageDispatcher."""

from __future__ import annotations

import queue
import time
from unittest.mock import Mock

from horde_worker_regen.process_management.action_ledger import ActionLedger
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.messages import HordeProcessMemoryMessage, HordeProcessState
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import (
    make_mock_bridge_data,
    make_mock_job,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
    track_popped_job_async,
)


def _make_message_dispatcher(
    *,
    state: WorkerState | None = None,
    process_map: ProcessMap | None = None,
    horde_model_map: HordeModelMap | None = None,
    job_tracker: JobTracker | None = None,
    bridge_data: Mock | None = None,
    process_message_queue: queue.Queue[object] | None = None,
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

    return MessageDispatcher(
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        process_message_queue=process_message_queue or Mock(spec=queue.Queue),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(),
        action_ledger=ActionLedger(),
        on_unload_vram=Mock(),
        state=state,
    )


class TestDetectDeadlock:
    """Tests for detect_deadlock."""

    def test_no_jobs_no_deadlock(self) -> None:
        """Deadlocks should never be considered to exist if there are no jobs."""
        message_dispatcher = _make_message_dispatcher()
        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_deadlock is False
        assert message_dispatcher._in_queue_deadlock is False

    async def test_recent_pop_skips_detection(self) -> None:
        """Deadlocks should not be detected if a job was just popped."""
        state = WorkerState(last_job_pop_time=time.time())
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())

        message_dispatcher = _make_message_dispatcher(state=state, job_tracker=job_tracker)
        message_dispatcher._in_deadlock = True
        message_dispatcher._in_queue_deadlock = True
        message_dispatcher._queue_deadlock_model = "stable_diffusion"
        message_dispatcher._queue_deadlock_process_id = 0

        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_deadlock is False
        assert message_dispatcher._in_queue_deadlock is False
        assert message_dispatcher._queue_deadlock_model is None

    async def test_detects_queue_deadlock_when_all_waiting_with_matching_model(self) -> None:
        """When all processes are waiting and one has the needed model, it's a queue deadlock."""
        state = WorkerState(last_job_pop_time=time.time() - 20)
        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_info.loaded_horde_model_name = "stable_diffusion"
        process_map = ProcessMap({0: process_info})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_queue_deadlock is True
        assert message_dispatcher._queue_deadlock_model == "stable_diffusion"
        assert message_dispatcher._queue_deadlock_process_id == 0

    async def test_detects_queue_deadlock_no_model_match_uses_first_job(self) -> None:
        """When all processes are waiting but none has the needed model, still a queue deadlock."""
        state = WorkerState(last_job_pop_time=time.time() - 20)
        process_info = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_info.loaded_horde_model_name = None
        process_map = ProcessMap({0: process_info})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job(model="some_model"))

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_queue_deadlock is True
        assert message_dispatcher._queue_deadlock_model == "some_model"

    async def test_detects_general_deadlock_no_processes(self) -> None:
        """General deadlock: jobs exist but no processes are busy and none waiting."""
        state = WorkerState(last_job_pop_time=time.time() - 20)
        process_map = ProcessMap({})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_deadlock is True

    async def test_deadlock_clears_when_processes_become_busy(self) -> None:
        """If a process starts working, the deadlock should clear."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        process_info = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({0: process_info})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        message_dispatcher._in_deadlock = True
        message_dispatcher._last_deadlock_detected_time = time.time() - 8

        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_deadlock is False

    async def test_queue_deadlock_persists_after_timeout(self) -> None:
        """Queue deadlock should remain active after timeout so the supervisor can recover it."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        process_info = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job(model="stable_diffusion"))

        message_dispatcher = _make_message_dispatcher(
            state=state,
            process_map=process_map,
            job_tracker=job_tracker,
        )
        message_dispatcher._in_queue_deadlock = True
        message_dispatcher._last_queue_deadlock_detected_time = time.time() - 35
        message_dispatcher._queue_deadlock_model = "stable_diffusion"
        message_dispatcher._queue_deadlock_process_id = 0

        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_queue_deadlock is True
        assert message_dispatcher._queue_deadlock_model == "stable_diffusion"

    def test_queue_deadlock_waits_if_processes_starting(self) -> None:
        """Queue deadlock should wait if processes are starting."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        process_info = make_mock_process_info(0, state=HordeProcessState.PROCESS_STARTING)
        process_info.last_process_state = HordeProcessState.PROCESS_STARTING
        process_map = ProcessMap({0: process_info})

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map)
        message_dispatcher._in_queue_deadlock = True
        message_dispatcher._last_queue_deadlock_detected_time = time.time() - 35

        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_queue_deadlock is True

    async def test_deadlock_persists_after_10_seconds(self) -> None:
        """Deadlock should remain active after timeout so the supervisor can recover it."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        process_map = ProcessMap({})
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())

        message_dispatcher = _make_message_dispatcher(
            state=state,
            process_map=process_map,
            job_tracker=job_tracker,
        )
        message_dispatcher._in_deadlock = True
        message_dispatcher._last_deadlock_detected_time = time.time() - 12

        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_deadlock is True

    async def test_memory_report_does_not_clear_deadlock_signal(self) -> None:
        """Passive child messages should not mask an active deadlock episode."""
        process_info = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: process_info})
        process_message_queue: queue.Queue[object] = queue.Queue()
        process_message_queue.put(
            HordeProcessMemoryMessage(
                process_id=0,
                process_launch_identifier=0,
                info="memory",
                ram_usage_bytes=1024,
            ),
        )

        message_dispatcher = _make_message_dispatcher(
            process_map=process_map,
            process_message_queue=process_message_queue,
        )
        message_dispatcher._in_deadlock = True
        message_dispatcher._in_queue_deadlock = True

        await message_dispatcher.receive_and_handle_process_messages()

        assert message_dispatcher._in_deadlock is True
        assert message_dispatcher._in_queue_deadlock is True
