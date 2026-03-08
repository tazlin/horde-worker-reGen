"""Tests for deadlock detection in MessageDispatcher."""

from __future__ import annotations

import queue
import time
from unittest.mock import Mock

from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.messages import HordeProcessState
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
        process_message_queue=Mock(spec=queue.Queue),
        get_model_baseline=lambda name: None,
        get_bridge_data=lambda: bridge_data,
        on_unload_vram=Mock(),
        state=state,
    )


class TestDetectDeadlock:
    """Tests for detect_deadlock."""

    def test_no_jobs_no_deadlock(self) -> None:
        """Deadlocks should never be considered to exist if there are no jobs."""
        md = _make_dispatcher()
        md.detect_deadlock()
        assert md._in_deadlock is False
        assert md._in_queue_deadlock is False

    def test_recent_pop_skips_detection(self) -> None:
        """Deadlocks should not be detected if a job was just popped."""
        state = WorkerState(last_job_pop_time=time.time())
        jt = JobTracker()
        job = Mock()
        job.model = "stable_diffusion"
        jt.jobs_pending_inference.append(job)

        md = _make_dispatcher(state=state, job_tracker=jt)
        md.detect_deadlock()
        assert md._in_deadlock is False

    def test_detects_queue_deadlock_when_all_waiting_with_matching_model(self) -> None:
        """When all processes are waiting and one has the needed model, it's a queue deadlock."""
        state = WorkerState(last_job_pop_time=time.time() - 20)
        proc = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.WAITING_FOR_JOB)
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
        proc.loaded_horde_model_name = "stable_diffusion"
        pm = ProcessMap({0: proc})

        jt = JobTracker()
        job = Mock()
        job.model = "stable_diffusion"
        jt.jobs_pending_inference.append(job)

        md = _make_dispatcher(state=state, process_map=pm, job_tracker=jt)
        md.detect_deadlock()
        assert md._in_queue_deadlock is True
        assert md._queue_deadlock_model == "stable_diffusion"
        assert md._queue_deadlock_process_id == 0

    def test_detects_queue_deadlock_no_model_match_uses_first_job(self) -> None:
        """When all processes are waiting but none has the needed model, still a queue deadlock."""
        state = WorkerState(last_job_pop_time=time.time() - 20)
        proc = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
        proc.loaded_horde_model_name = None
        pm = ProcessMap({0: proc})

        jt = JobTracker()
        job = Mock()
        job.model = "some_model"
        jt.jobs_pending_inference.append(job)

        md = _make_dispatcher(state=state, process_map=pm, job_tracker=jt)
        md.detect_deadlock()
        assert md._in_queue_deadlock is True
        assert md._queue_deadlock_model == "some_model"

    def test_detects_general_deadlock_no_processes(self) -> None:
        """General deadlock: jobs exist but no processes are busy and none waiting."""
        state = WorkerState(last_job_pop_time=time.time() - 20)
        pm = ProcessMap({})

        jt = JobTracker()
        job = Mock()
        job.model = "stable_diffusion"
        jt.jobs_pending_inference.append(job)

        md = _make_dispatcher(state=state, process_map=pm, job_tracker=jt)
        md.detect_deadlock()
        assert md._in_deadlock is True

    def test_deadlock_clears_when_processes_become_busy(self) -> None:
        """If a process starts working, the deadlock should clear."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        proc = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        pm = ProcessMap({0: proc})

        jt = JobTracker()
        job = Mock()
        job.model = "stable_diffusion"
        jt.jobs_pending_inference.append(job)

        md = _make_dispatcher(state=state, process_map=pm, job_tracker=jt)
        md._in_deadlock = True
        md._last_deadlock_detected_time = time.time() - 8

        md.detect_deadlock()
        assert md._in_deadlock is False

    def test_queue_deadlock_resolves_after_timeout(self) -> None:
        """Queue deadlock should resolve after a timeout period."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        pm = ProcessMap({})

        md = _make_dispatcher(state=state, process_map=pm)
        md._in_queue_deadlock = True
        md._last_queue_deadlock_detected_time = time.time() - 35
        md._queue_deadlock_model = "stable_diffusion"
        md._queue_deadlock_process_id = 0

        md.detect_deadlock()
        assert md._in_queue_deadlock is False
        assert md._queue_deadlock_model is None

    def test_queue_deadlock_waits_if_processes_starting(self) -> None:
        """Queue deadlock should wait if processes are starting."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        proc = make_mock_process_info(0, state=HordeProcessState.PROCESS_STARTING)
        proc.last_process_state = HordeProcessState.PROCESS_STARTING
        pm = ProcessMap({0: proc})

        md = _make_dispatcher(state=state, process_map=pm)
        md._in_queue_deadlock = True
        md._last_queue_deadlock_detected_time = time.time() - 35

        md.detect_deadlock()
        assert md._in_queue_deadlock is True

    def test_deadlock_persists_then_clears_after_10_seconds(self) -> None:
        """Deadlock should persist for a short period and then clear after 10 seconds."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        pm = ProcessMap({})

        md = _make_dispatcher(state=state, process_map=pm)
        md._in_deadlock = True
        md._last_deadlock_detected_time = time.time() - 12

        md.detect_deadlock()
        assert md._in_deadlock is False
