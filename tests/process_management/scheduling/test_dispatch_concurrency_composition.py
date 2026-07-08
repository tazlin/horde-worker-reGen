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

from collections.abc import Awaitable, Callable
from unittest.mock import Mock

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap, ModelLoadState
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    mark_job_in_progress_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MODEL_A = "model_alpha"
_MODEL_B = "model_beta"
_MODEL_C = "model_gamma"
_SDXL_A = "sdxl_alpha"
_SDXL_B = "sdxl_beta"
_SLOW_WORKFLOW = "qr_code"

SchedulerFactory = Callable[[JobTracker], Awaitable[tuple[InferenceScheduler, ImageGenerateJobPopResponse]]]


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


async def _run_cycle_scheduler(
    job_tracker: JobTracker,
    *,
    max_concurrent: int = 2,
    job_a: ImageGenerateJobPopResponse | None = None,
    job_b: ImageGenerateJobPopResponse | None = None,
) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse]:
    """Two resident lanes ready for a full scheduling-cycle dispatch."""
    slot_a = make_mock_process_info(1, model_name=_MODEL_A, state=HordeProcessState.PRELOADED_MODEL)
    slot_b = make_mock_process_info(2, model_name=_MODEL_B, state=HordeProcessState.PRELOADED_MODEL)
    process_map = ProcessMap({1: slot_a, 2: slot_b})
    horde_model_map = HordeModelMap(root={})
    horde_model_map.update_entry(horde_model_name=_MODEL_A, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
    horde_model_map.update_entry(horde_model_name=_MODEL_B, load_state=ModelLoadState.LOADED_IN_RAM, process_id=2)

    first_job = job_a if job_a is not None else make_job_pop_response(model=_MODEL_A)
    second_job = job_b if job_b is not None else make_job_pop_response(model=_MODEL_B)
    await job_tracker.record_popped_job(first_job)
    await job_tracker.record_popped_job(second_job)

    scheduler = _make_inference_scheduler(
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(max_threads=max_concurrent),
        max_concurrent=max_concurrent,
        max_inference=2,
    )
    scheduler.preload_models = Mock(return_value=False)  # type: ignore[method-assign]
    scheduler._should_keep_model_resident = Mock(return_value=False)  # type: ignore[method-assign]
    return scheduler, second_job


async def _run_cycle_and_count_in_progress(job_tracker: JobTracker, factory: SchedulerFactory) -> int:
    """Run one scheduler cycle and return how many jobs it started."""
    scheduler, _second_job = await factory(job_tracker)
    await scheduler.run_scheduling_cycle({})
    return len(job_tracker.jobs_in_progress)


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


class TestSchedulingCycleSerializers:
    """Mechanisms that can make a configured two-thread worker behave as one-threaded."""

    async def test_cycle_starts_two_small_jobs_when_no_serializer_applies(self, job_tracker: JobTracker) -> None:
        """The whole scheduling cycle fills both ready lanes for ordinary small jobs."""
        scheduler, second_job = await _run_cycle_scheduler(job_tracker)

        await scheduler.run_scheduling_cycle({})

        assert second_job in job_tracker.jobs_in_progress
        assert len(job_tracker.jobs_in_progress) == 2

    async def test_runtime_effective_cap_below_configured_threads_serializes(
        self,
        job_tracker: JobTracker,
    ) -> None:
        """A live cap reduced below ``bridge_data.max_threads`` withholds the second small job."""
        scheduler, _second_job = await _run_cycle_scheduler(job_tracker)
        scheduler._runtime_config = RuntimeConfig(  # type: ignore[assignment]
            initial=make_mock_bridge_data(max_threads=1),
            max_threads_ceiling=2,
        )
        scheduler._runtime_config.bridge_data.max_threads = 2

        await scheduler.run_scheduling_cycle({})

        assert len(job_tracker.jobs_in_progress) == 1
        assert scheduler._runtime_config.bridge_data.max_threads == 2
        assert scheduler._runtime_config.effective_max_threads == 1

    async def test_one_live_process_serializes_even_with_two_thread_cap(self, job_tracker: JobTracker) -> None:
        """A shed or scaled-down pool cannot run a second job without a second accepting process."""
        slot_a = make_mock_process_info(1, model_name=_MODEL_A, state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: slot_a})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(horde_model_name=_MODEL_A, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
        job_a = make_job_pop_response(model=_MODEL_A)
        job_b = make_job_pop_response(model=_MODEL_B)
        await job_tracker.record_popped_job(job_a)
        await job_tracker.record_popped_job(job_b)
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(max_threads=2),
            max_concurrent=2,
            max_inference=2,
        )
        scheduler.preload_models = Mock(return_value=False)  # type: ignore[method-assign]
        scheduler._should_keep_model_resident = Mock(return_value=False)  # type: ignore[method-assign]

        await scheduler.run_scheduling_cycle({})

        assert len(job_tracker.jobs_in_progress) == 1

    async def test_batched_running_job_serializes_even_small_candidate(self, job_tracker: JobTracker) -> None:
        """The keep-single batched-job rule blocks every candidate while the batch samples."""
        batch_job = make_job_pop_response(model=_MODEL_A, n_iter=2)

        async def factory(tracker: JobTracker):  # noqa: ANN202
            return await _run_cycle_scheduler(tracker, job_a=batch_job)

        assert await _run_cycle_and_count_in_progress(job_tracker, factory) == 1

    async def test_exclusive_in_progress_job_serializes_small_candidate(self, job_tracker: JobTracker) -> None:
        """An exclusive over-budget admit suppresses otherwise-ready small work."""
        scheduler, _second_job = await _run_cycle_scheduler(job_tracker)
        first = await scheduler.start_inference()
        assert first is True
        job_tracker.mark_admitted_over_budget(job_tracker.jobs_in_progress[0])
        job_tracker.mark_admitted_exclusive(job_tracker.jobs_in_progress[0])

        second = await scheduler.start_inference()

        assert second is False
        assert len(job_tracker.jobs_in_progress) == 1

    async def test_degraded_retry_serializes_while_other_work_is_in_progress(self, job_tracker: JobTracker) -> None:
        """A resource-failure retry runs isolated, even when its model is already resident."""
        scheduler, second_job = await _run_cycle_scheduler(job_tracker)
        first = await scheduler.start_inference()
        assert first is True
        tracked = job_tracker._tracked_for(second_job)  # noqa: SLF001
        assert tracked is not None
        tracked.needs_degraded_dispatch = True

        second = await scheduler.start_inference()

        assert second is False
        assert second_job not in job_tracker.jobs_in_progress

    async def test_post_processing_hold_serializes_otherwise_ready_small_job(self, job_tracker: JobTracker) -> None:
        """The post-processing mutex can hold the second job after the first starts."""
        scheduler, second_job = await _run_cycle_scheduler(job_tracker)
        first = await scheduler.start_inference()
        assert first is True
        scheduler._should_defer_dispatch_for_post_processing = Mock(return_value=True)  # type: ignore[method-assign]

        second = await scheduler.start_inference()

        assert second is False
        assert second_job not in job_tracker.jobs_in_progress

    async def test_dispatch_residency_hold_serializes_otherwise_ready_small_job(self, job_tracker: JobTracker) -> None:
        """Residency reconciliation can hold a resident candidate before it materializes VRAM."""
        scheduler, second_job = await _run_cycle_scheduler(job_tracker)
        first = await scheduler.start_inference()
        assert first is True
        scheduler._dispatch_residency_reconciliation_holds = Mock(return_value=True)  # type: ignore[method-assign]

        second = await scheduler.start_inference()

        assert second is False
        assert second_job not in job_tracker.jobs_in_progress

    async def test_same_model_queue_waits_on_busy_resident_slot_without_duplicate_copy(
        self,
        job_tracker: JobTracker,
    ) -> None:
        """A run of same-model jobs is serial unless another process already holds a distinct queued model."""
        slot_a = make_mock_process_info(1, model_name=_MODEL_A, state=HordeProcessState.PRELOADED_MODEL)
        slot_b = make_mock_process_info(2, model_name=_MODEL_C, state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: slot_a, 2: slot_b})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(horde_model_name=_MODEL_A, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
        horde_model_map.update_entry(horde_model_name=_MODEL_C, load_state=ModelLoadState.LOADED_IN_RAM, process_id=2)
        job_a = make_job_pop_response(model=_MODEL_A)
        job_b = make_job_pop_response(model=_MODEL_A)
        await job_tracker.record_popped_job(job_a)
        await job_tracker.record_popped_job(job_b)
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(max_threads=2),
            max_concurrent=2,
            max_inference=2,
        )
        scheduler.preload_models = Mock(return_value=False)  # type: ignore[method-assign]
        scheduler._should_keep_model_resident = Mock(return_value=False)  # type: ignore[method-assign]

        await scheduler.run_scheduling_cycle({})

        assert len(job_tracker.jobs_in_progress) == 1

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


class TestSchedulingCycleOverlapComposition:
    """The scheduling cycle should reach the overlap gate before serializing heavy work."""

    async def test_cycle_starts_second_heavy_job_after_progress_and_headroom(self, job_tracker: JobTracker) -> None:
        """A progressed heavy job with confirmed room should not keep the second slot idle."""
        running_slot = make_mock_process_info(1, model_name=_SDXL_A, state=HordeProcessState.INFERENCE_STARTING)
        candidate_slot = make_mock_process_info(2, model_name=_SDXL_B, state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: running_slot, 2: candidate_slot})

        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(horde_model_name=_SDXL_A, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
        horde_model_map.update_entry(horde_model_name=_SDXL_B, load_state=ModelLoadState.LOADED_IN_RAM, process_id=2)

        reference = {
            _SDXL_A: make_mock_model_reference_record(
                _SDXL_A,
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            ),
            _SDXL_B: make_mock_model_reference_record(
                _SDXL_B,
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            ),
        }
        running_job = make_job_pop_response(_SDXL_A, ddim_steps=20, workflow=_SLOW_WORKFLOW)
        candidate_job = make_job_pop_response(_SDXL_B, ddim_steps=20, workflow=_SLOW_WORKFLOW)
        await job_tracker.record_popped_job(running_job)
        await mark_job_in_progress_async(job_tracker, running_job)
        await job_tracker.record_popped_job(candidate_job)

        running_slot.last_job_referenced = running_job
        running_slot.loaded_horde_model_baseline = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
        running_slot.batch_amount = 1
        running_slot.last_total_steps = 20
        running_slot.last_current_step = 18
        running_slot.last_heartbeat_percent_complete = 90

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(max_threads=2, gpu_sampling_lease_enabled=False),
            max_concurrent=2,
            max_inference=2,
            model_metadata=make_test_model_metadata(reference),
        )
        scheduler.preload_models = Mock(return_value=False)  # type: ignore[method-assign]
        scheduler._overlap_memory_verdict = Mock(return_value=True)  # type: ignore[method-assign]
        scheduler._should_keep_model_resident = Mock(return_value=False)  # type: ignore[method-assign]

        await scheduler.run_scheduling_cycle(reference)

        assert candidate_job in job_tracker.jobs_in_progress
