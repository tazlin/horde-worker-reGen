"""A lane that owns a dispatched inference attempt must not accept a second job on a child preload report.

A job can be dispatched to a lane in the same scheduling cycle that its model's preload is sent to that lane.
The dispatch stamps the lane ``INFERENCE_PRIMED`` and binds the job to the lane's launch, but the child then
reports the preload's own states (``UNLOADED_MODEL_FROM_RAM``, ``PRELOADING_MODEL``, ``PRELOADED_MODEL``),
and ``ProcessMap.on_process_state_change`` applies each one. ``PRELOADED_MODEL`` is a
``HordeProcessInfo.can_accept_job`` state, so a second job is dispatched onto the lane while the first is
still in flight. When the first job finishes and the lane reports idle, the lost-result reap finds the lane
owning the second job and requeues it, though the child still holds it.

These tests drive the real ``ProcessMap``, ``JobTracker`` and ``MessageDispatcher`` state-change handler. The
dispatch itself is the bookkeeping ``InferenceScheduler._dispatch_inference_message`` applies once the
START_INFERENCE send succeeds, applied directly so the scenario does not depend on admission.
"""

from __future__ import annotations

import time

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessState,
    HordeProcessStateChangeMessage,
)
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)

_MODEL = "AlbedoBase XL 3.1"
_LANE = 1

_PRELOAD_REPORTS = (
    HordeProcessState.UNLOADED_MODEL_FROM_RAM,
    HordeProcessState.PRELOADING_MODEL,
    HordeProcessState.PRELOADED_MODEL,
)


_ACCEPTING_STATES = frozenset(
    {
        HordeProcessState.WAITING_FOR_JOB,
        HordeProcessState.PRELOADED_MODEL,
        HordeProcessState.INFERENCE_COMPLETE,
        HordeProcessState.ALCHEMY_COMPLETE,
    },
)


async def _noop_sleep(_seconds: float) -> None:
    """Let a control-loop tick run without pacing."""


def _report(pm: HordeWorkerProcessManager, state: HordeProcessState) -> None:
    """Feed a child state report through the dispatcher's state-change entry point."""
    lane = pm._process_map[_LANE]
    pm._message_dispatcher._handle_process_state_change(
        HordeProcessStateChangeMessage(
            process_id=_LANE,
            process_launch_identifier=lane.process_launch_identifier,
            info=f"child reports {state.name}",
            process_state=state,
        ),
    )


async def _dispatch(pm: HordeWorkerProcessManager, job: ImageGenerateJobPopResponse) -> None:
    """Apply the parent-side bookkeeping of a successful START_INFERENCE send to the lane."""
    lane = pm._process_map[_LANE]
    await pm._job_tracker.mark_inference_started(job)
    lane.last_control_flag = HordeControlFlag.START_INFERENCE
    lane.record_inference_ownership(job, attempt_ordinal=1)
    lane.loaded_horde_model_name = _MODEL
    pm._process_map.on_process_state_change(process_id=_LANE, new_state=HordeProcessState.INFERENCE_PRIMED)


async def _lane_with_job_dispatched_alongside_preload() -> tuple[
    HordeWorkerProcessManager, ImageGenerateJobPopResponse
]:
    """Build a manager whose lane received job A with its preload, then reported the preload's states."""
    pm = make_testable_process_manager()
    pm._process_map[_LANE] = make_mock_process_info(_LANE, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    job_a = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(pm._job_tracker, job_a)
    await _dispatch(pm, job_a)
    for state in _PRELOAD_REPORTS:
        _report(pm, state)
    assert job_a in pm._job_tracker.jobs_in_progress
    return pm, job_a


async def test_lane_running_a_job_does_not_accept_a_second() -> None:
    """While job A is in progress on the lane, the lane is not offered as able to accept another job."""
    pm, _job_a = await _lane_with_job_dispatched_alongside_preload()

    assert pm._process_map[_LANE].last_process_state == HordeProcessState.PRELOADED_MODEL
    assert pm._process_map[_LANE].can_accept_job() is False


async def test_second_job_survives_first_job_completion() -> None:
    """B waits for A to finish, then a dispatch of B onto the lane is not requeued as lost."""
    pm, job_a = await _lane_with_job_dispatched_alongside_preload()
    job_b = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(pm._job_tracker, job_b)
    assert pm._process_map[_LANE].can_accept_job() is False

    # The child is still running A: it samples, completes, returns A's result, then reports idle.
    _report(pm, HordeProcessState.INFERENCE_STARTING)
    _report(pm, HordeProcessState.INFERENCE_COMPLETE)
    pm._process_map[_LANE].retire_inference_ownership(job_a)
    assert await pm._job_tracker.release_in_progress(job_a)
    _report(pm, HordeProcessState.WAITING_FOR_JOB)

    assert pm._process_map[_LANE].can_accept_job() is True
    await _dispatch(pm, job_b)
    _report(pm, HordeProcessState.INFERENCE_STARTING)

    # jobs_pending_inference includes in-progress jobs by design, so the lost-result requeue is read from the
    # in-progress set and the lane's ownership.
    assert job_b in pm._job_tracker.jobs_in_progress
    assert pm._process_map[_LANE].current_inference_job() == job_b


async def test_lane_accepts_again_in_the_same_message_after_the_lost_result_reap() -> None:
    """The reap releases the job and the lane's ownership together, so the lane is offered at once."""
    pm, job_a = await _lane_with_job_dispatched_alongside_preload()
    _report(pm, HordeProcessState.INFERENCE_STARTING)
    # The idle report must begin after the dispatch stamps for the reap to read it as this job's return.
    pm._process_map[_LANE].last_process_state_started_at = time.time() + 1.0
    _report(pm, HordeProcessState.WAITING_FOR_JOB)

    assert job_a not in pm._job_tracker.jobs_in_progress
    assert pm._process_map[_LANE].current_inference_job() is None
    assert pm._process_map[_LANE].can_accept_job() is True


async def test_lane_accepts_again_after_one_tick_when_its_job_faulted_without_a_result() -> None:
    """An ownership whose job left progress by a fault that sent no result is retired by the next tick."""
    pm, job_a = await _lane_with_job_dispatched_alongside_preload()
    _report(pm, HordeProcessState.WAITING_FOR_JOB)
    pm._job_tracker.handle_job_fault_now(faulted_job=job_a, retryable=False)
    assert job_a not in pm._job_tracker.jobs_in_progress
    assert pm._process_map[_LANE].can_accept_job() is False

    pm._sleep = _noop_sleep
    pm._last_status_message_time = time.time()
    await pm._control_loop_tick()

    assert pm._process_map[_LANE].current_inference_job() is None
    assert pm._process_map[_LANE].can_accept_job() is True


@pytest.mark.parametrize("state", sorted(_ACCEPTING_STATES, key=lambda s: s.name))
def test_idle_lane_without_ownership_accepts_in_each_accepting_state(state: HordeProcessState) -> None:
    """Without ownership, the accepting states are the ones the predicate always accepted."""
    lane = make_mock_process_info(_LANE, model_name=_MODEL, state=state)
    assert lane.current_inference_job() is None
    assert lane.can_accept_job() is True
