"""Composition test: a healthy multi-thread pool actually reaches its configured concurrency.

The dispatch path is a stack of independently-correct gates (the in-progress cap, the exclusive-job
hold, the overlap headway, line-skip, the whole-card residency). Each gate has unit coverage, but a
worker can still end up effectively single-threaded when their composition serializes dispatch, and no
unit test notices: every gate is individually behaving as specified. This module pins the composed
outcome for the baseline healthy case, so a future gate (or a changed default) that silently costs the
second thread fails a test instead of a field deployment.

The scenario is deliberately the easiest possible dispatch: two idle slots, each already holding a
different resident light model, one queued job per model, no budget pressure, no exclusive admits, no
batches. If this cannot reach two concurrent inferences, nothing can. The controls then pin the two
sanctioned serializers (the thread cap and an exclusive admit) as the only things standing between
this scenario and full concurrency.
"""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap, ModelLoadState
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MODEL_A = "model_alpha"
_MODEL_B = "model_beta"


async def _two_lane_scheduler(job_tracker: JobTracker, *, max_concurrent: int = 2):  # noqa: ANN202
    """Two idle slots with distinct resident models and one queued job for each."""
    slot_a = make_mock_process_info(1, model_name=_MODEL_A, state=HordeProcessState.PRELOADED_MODEL)
    slot_b = make_mock_process_info(2, model_name=_MODEL_B, state=HordeProcessState.PRELOADED_MODEL)
    process_map = ProcessMap({1: slot_a, 2: slot_b})
    horde_model_map = HordeModelMap(root={})
    horde_model_map.update_entry(horde_model_name=_MODEL_A, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
    horde_model_map.update_entry(horde_model_name=_MODEL_B, load_state=ModelLoadState.LOADED_IN_RAM, process_id=2)

    job_a = make_job_pop_response(model=_MODEL_A)
    job_b = make_job_pop_response(model=_MODEL_B)
    await job_tracker.record_popped_job(job_a)
    await job_tracker.record_popped_job(job_b)

    scheduler = _make_inference_scheduler(
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(max_threads=max_concurrent),
        max_concurrent=max_concurrent,
        max_inference=2,
    )
    return scheduler, (job_a, slot_a), (job_b, slot_b)


async def _dispatch_first(scheduler, job_tracker: JobTracker):  # noqa: ANN001, ANN202
    """Dispatch the first job and reflect its consequences on the pool, as the process manager would.

    The scheduler only *selects*; the manager marks the job started and the slot transitions to a busy
    state. Both must be mirrored here or the second selection would see an idle pool.
    """
    first = await scheduler.get_next_job_and_process()
    assert first is not None
    await job_tracker.mark_inference_started(first.next_job)
    first.process_with_model.last_job_referenced = first.next_job
    scheduler._process_map.on_process_state_change(
        process_id=first.process_with_model.process_id,
        new_state=HordeProcessState.INFERENCE_STARTING,
    )
    return first


class TestHealthyPoolReachesConfiguredConcurrency:
    """With two dispatchable lanes and no pressure, the second thread must actually dispatch."""

    async def test_two_lanes_dispatch_concurrently(self, job_tracker: JobTracker) -> None:
        """The second selection returns the other lane's job while the first is in flight."""
        scheduler, _, _ = await _two_lane_scheduler(job_tracker)

        first = await _dispatch_first(scheduler, job_tracker)
        second = await scheduler.get_next_job_and_process()

        assert second is not None
        assert second.next_job is not first.next_job
        assert second.process_with_model.process_id != first.process_with_model.process_id
        assert len(job_tracker.jobs_in_progress) == 1
        await job_tracker.mark_inference_started(second.next_job)
        assert len(job_tracker.jobs_in_progress) == 2

    async def test_thread_cap_is_the_serializer_at_one(self, job_tracker: JobTracker) -> None:
        """CONTROL: the identical pool at max_threads=1 withholds the second dispatch."""
        scheduler, _, _ = await _two_lane_scheduler(job_tracker, max_concurrent=1)

        await _dispatch_first(scheduler, job_tracker)
        second = await scheduler.get_next_job_and_process()

        assert second is None

    async def test_exclusive_admit_is_the_serializer_at_two(self, job_tracker: JobTracker) -> None:
        """CONTROL: an exclusively-admitted in-flight job withholds the second dispatch at any cap.

        Together with the healthy case this pins that the thread cap and an exclusive admit are the
        serializers, and an idle-pool dispatch is never withheld by gate composition alone.
        """
        scheduler, _, _ = await _two_lane_scheduler(job_tracker)

        first = await _dispatch_first(scheduler, job_tracker)
        job_tracker.mark_admitted_over_budget(first.next_job)
        job_tracker.mark_admitted_exclusive(first.next_job)
        second = await scheduler.get_next_job_and_process()

        assert second is None
