"""Lane recovery and re-ask idempotence for the emergency starvation teardown with whole-card residency off.

The starvation context teardown is an emergency-liveness path that must run even when
``whole_card_exclusive_residency`` is disabled: a weight-dominant head starved behind its own idle sibling
contexts tears them down to admit. Two properties of that path were previously never exercised with the flag
off, because whole-card residency (and therefore its restore) was unreachable in that configuration:

* The worker must get its lanes back. The exclusive hold the teardown takes releases when the head's job leaves
  the in-progress stages, and the torn-down sibling processes regrow once the residency drains. If either the
  hold release or the regrowth gated on the flag, every large-model job would permanently degrade the worker to
  a single lane. Neither does: the release is stage-based in the job tracker and the regrowth runs through
  ``_restore_siblings_after_whole_card`` unconditionally.
* Re-asks must be safe. Past the escalation threshold the head re-asks every scheduler cycle, so a second
  ``REDUCE_LIVE_CONTEXTS`` can arrive while the first teardown's processes have already retired. The scale-down
  targets a fixed process count and retires victims from the map immediately, and the residency establish only
  scales when the count still exceeds the target, so repeated commands converge on the target rather than
  tearing down additional processes.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.vram_arbiter import ActuatorCommand, ActuatorCommandKind
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler, _PreloadActuation
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MAX_INFERENCE = 2
_HEAD_MODEL = "AlbedoBase XL (SDXL)"


class _ScaleRecorder:
    """A ``scale_inference_processes`` stand-in that mutates the process map and records what it removed.

    The real lifecycle retires a scaled-down process from the map synchronously, so the live count drops the
    instant the teardown runs. This mirrors that (grow/shrink the map of idle contexts toward the target) and
    counts the removals, so a test can assert both the resulting lane count and that a repeated command did not
    tear down additional processes beyond the target.
    """

    def __init__(self, process_map: ProcessMap) -> None:
        self._process_map = process_map
        self.removed_total = 0

    def __call__(self, target_count: int, *, device_index: int | None = None, **_kwargs: object) -> int:
        loaded = self._process_map.num_loaded_inference_processes()
        while loaded > target_count:
            victim = next(
                pid
                for pid, info in self._process_map.items()
                if info.last_process_state == HordeProcessState.WAITING_FOR_JOB
            )
            del self._process_map[victim]
            self.removed_total += 1
            loaded -= 1
        while loaded < target_count:
            new_pid = (max(self._process_map.keys()) + 1) if self._process_map.keys() else 0
            self._process_map[new_pid] = make_mock_process_info(
                new_pid,
                model_name=None,
                state=HordeProcessState.WAITING_FOR_JOB,
                device_index=0,
            )
            loaded += 1
        return self._process_map.num_loaded_inference_processes()


async def _flag_off_scheduler_with_tracked_head() -> tuple[
    InferenceScheduler, ProcessMap, _ScaleRecorder, ImageGenerateJobPopResponse
]:
    """A flag-off scheduler with two idle lanes, a tracked pending head, and a map-mutating scale stub.

    The teardown actuation's residency establish and the sibling unload run for real; only the lifecycle
    scale-down (replaced by the recorder) and the VRAM unload send (mocked) are stood in, so the assertions
    observe real lane-count and residency-hold state, not call wiring.
    """
    process_map = ProcessMap(
        {
            0: make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
            1: make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
        },
    )
    job_tracker = JobTracker()
    job = make_job_pop_response(_HEAD_MODEL)
    await track_popped_job_async(job_tracker, job)
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(enable_vram_budget=True, whole_card_exclusive_residency=False),
        max_inference=_MAX_INFERENCE,
    )
    recorder = _ScaleRecorder(process_map)
    scheduler._process_lifecycle.scale_inference_processes = recorder  # type: ignore[method-assign]
    scheduler._process_lifecycle.restore_safety_on_gpu = lambda: False  # type: ignore[method-assign]
    scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]
    scheduler._pause_post_process_for_residency_if_idle = Mock(return_value=False)  # type: ignore[method-assign]

    forecast = Mock()
    forecast.max_resident_processes = Mock(return_value=1)
    forecast.total_vram_mb = 24000.0
    forecast.fits_weights_now = True
    scheduler._preload_actuation = _PreloadActuation(
        job=job,
        available_process=process_map[0],
        forecast=forecast,
        max_resident=1,
    )
    return scheduler, process_map, recorder, job


def _issue_reduce_contexts(scheduler: InferenceScheduler) -> None:
    """Deliver one arbiter REDUCE_LIVE_CONTEXTS command through the preload actuation surface."""
    commands = (ActuatorCommand(kind=ActuatorCommandKind.REDUCE_LIVE_CONTEXTS, device_index=None),)
    scheduler._execute_preload_actuations(commands, device_index=None, for_head_of_queue=True)


class TestFlagOffStarvationTeardownRecovery:
    """The full emergency cycle recovers the worker's lanes even with steady-state residency disabled."""

    async def test_teardown_then_completion_restores_lane_count(self) -> None:
        """Head tears its siblings down, holds exclusive through dispatch, then the drain regrows the pool."""
        scheduler, process_map, _recorder, job = await _flag_off_scheduler_with_tracked_head()

        _issue_reduce_contexts(scheduler)
        assert process_map.num_loaded_inference_processes() == 1
        assert scheduler._job_tracker.has_exclusive_job_in_progress() is True
        assert scheduler._residency_state(None).model == _HEAD_MODEL

        # The head dispatches and its job runs, then finishes: it leaves the in-progress stages for a
        # post-inference stage, which is what releases the exclusive hold (a stage transition, not the flag).
        await scheduler._job_tracker.mark_inference_started(job)
        tracked = scheduler._job_tracker._tracked_for(job)
        assert tracked is not None
        assert scheduler._job_tracker._set_stage(tracked, JobStage.PENDING_SAFETY_CHECK) is True
        assert scheduler._job_tracker.has_exclusive_job_in_progress() is False

        # The residency's cooldown lapses with no heavy job left in flight, so the restore pass releases the
        # residency and grows the pool back to its pre-emergency ceiling.
        scheduler._residency_state(None).cooldown_until = 0.0
        scheduler._restore_siblings_after_whole_card()

        assert process_map.num_loaded_inference_processes() == _MAX_INFERENCE
        assert scheduler._residency_state(None).model is None


class TestReAskTeardownIsIdempotent:
    """A REDUCE_LIVE_CONTEXTS re-issued after the first teardown retired its victims tears nothing more down."""

    async def test_second_command_does_not_reduce_below_the_target(self) -> None:
        """Two commands in succession converge on the target lane count and re-establish nothing."""
        scheduler, process_map, recorder, _job = await _flag_off_scheduler_with_tracked_head()

        _issue_reduce_contexts(scheduler)
        assert process_map.num_loaded_inference_processes() == 1
        removed_after_first = recorder.removed_total
        established_at_after_first = scheduler._residency_state(None).established_at
        assert removed_after_first == 1
        assert established_at_after_first != 0.0

        # The head re-asks the next cycle while the first teardown has already retired its victim: the count is
        # already at the target, so the establish does not scale again and the residency is not re-stamped.
        _issue_reduce_contexts(scheduler)
        assert process_map.num_loaded_inference_processes() == 1
        assert recorder.removed_total == removed_after_first
        assert scheduler._residency_state(None).established_at == established_at_after_first
        assert scheduler._residency_state(None).model == _HEAD_MODEL
