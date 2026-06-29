"""The scheduler plans room for a job's own post-processing VRAM peak before that peak can stall.

A job's upscaler/face-fixer peak (``predict_job_post_processing_vram_mb``, roughly 8.5 GB for a 4x SDXL
upscale) lands *after* sampling. The VRAM budget admits the job on its sampling footprint alone, so on a
card already carrying warm sibling models and several process contexts the peak allocates into a device
with no headroom and thrashes until the post-processing watchdog reaps the slot. ``free_memory`` inside the
inference child can release only that process's own weights; the sibling models and contexts that fill the
card are cross-process and reclaimable only by the orchestrator.

``_plan_post_processing_reclaim`` therefore sizes the peak against the post-sampling headroom and the room
the job's own (idle-during-upscale) weights would free, and returns the cheapest action that hosts it:

1. :attr:`PostProcessingReclaimAction.DELEGATE_IN_PROCESS`: the peak fits once the job's own weights are
   freed, which the inference child's ``free_memory`` does in-process, so no orchestrator action is needed.
2. :attr:`PostProcessingReclaimAction.EVICT_SIBLING_MODEL`: own-weights room is not enough, so a
   *different* model resident on an idle sibling is evicted (cross-process room only the orchestrator frees).
3. :attr:`PostProcessingReclaimAction.REDUCE_CONTEXT`: nothing idle holds an evictable model, so a sibling
   process is stopped to reclaim its context (the rare last rung).
4. :attr:`PostProcessingReclaimAction.INSUFFICIENT`: no orchestrator-reclaimable room hosts the peak (a
   single-process worker on a tiny card), so the job faults gracefully rather than thrashing the card.

Each test fixes a card layout and asserts the planner returns the action that layout calls for.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling import inference_scheduler as inference_scheduler_module
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    InferenceScheduler,
    PostProcessingReclaimAction,
    PostProcessingReclaimPlan,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MODEL = "WAI-NSFW-illustrious-SDXL"
_OTHER_MODEL = "CyberRealistic Pony"

# A 4x upscale + face-fixer on an SDXL job, sized as hordelib's burden estimate does, against a 16 GB
# card whose process contexts and a warm sibling leave nowhere near this much free; with the job's own
# SDXL diffusion weights resident (idle during the upscale).
_POST_PROCESSING_PEAK_MB = 8533.0
_OWN_WEIGHTS_MB = 4900.0
# Headroom that hosts the peak only once the job's own weights are freed (peak > free, peak <= free + own).
_FREE_VRAM_OWN_WEIGHTS_SUFFICE_MB = 5000.0
# Headroom so low that even freeing the job's own weights leaves a shortfall (peak > free + own).
_FREE_VRAM_NEEDS_SIBLING_MB = 1000.0
# Ample headroom that hosts the peak as-is.
_FREE_VRAM_AMPLE_MB = 12000.0


def _scheduler_with_post_processing_peak(
    job_tracker: JobTracker,
    *,
    monkeypatch: pytest.MonkeyPatch,
    free_vram_mb: float,
    peak_for_model: str | None,
    process_map: ProcessMap | None = None,
) -> InferenceScheduler:
    """A budget-active scheduler with a controllable measured free VRAM, peak, and own-weights footprint.

    ``peak_for_model`` names the model whose jobs report the large post-processing peak and the resident
    weight footprint; every other job reports 0 (no post-processing), so the reserve self-scales away when
    nothing is upscaling.
    """
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
    )
    scheduler = _make_inference_scheduler(
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        process_map=process_map,
    )
    scheduler._measured_free_vram_mb = Mock(return_value=free_vram_mb)  # type: ignore[method-assign]

    def _fake_peak(job: object, baseline: str | None) -> float:
        return _POST_PROCESSING_PEAK_MB if getattr(job, "model", None) == peak_for_model else 0.0

    def _fake_weight(job: object, baseline: str | None) -> float:
        return _OWN_WEIGHTS_MB if getattr(job, "model", None) == peak_for_model else 0.0

    monkeypatch.setattr(inference_scheduler_module, "predict_job_post_processing_vram_mb", _fake_peak)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_weight_mb", _fake_weight)
    return scheduler


async def test_delegates_in_process_when_own_weights_suffice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The peak overflows the bare headroom but fits once the job's own weights are freed.

    Freeing the job's own (idle-during-upscale) diffusion weights is exactly what ComfyUI's per-process
    ``free_memory`` does in-child, so the orchestrator delegates to it rather than evicting a sibling.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await job_tracker.mark_inference_started(dispatched)

    scheduler = _scheduler_with_post_processing_peak(
        job_tracker,
        monkeypatch=monkeypatch,
        free_vram_mb=_FREE_VRAM_OWN_WEIGHTS_SUFFICE_MB,
        peak_for_model=_MODEL,
    )

    plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None)
    assert plan.action is PostProcessingReclaimAction.DELEGATE_IN_PROCESS


async def test_evicts_idle_sibling_when_own_weights_not_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Own-weights room leaves a shortfall, so free a different model resident on an idle sibling."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await job_tracker.mark_inference_started(dispatched)
    # Another queued (not in-flight) job reuses the dispatched model -> its weights are still demanded.
    await track_popped_job_async(job_tracker, make_job_pop_response(model=_MODEL))

    process_map = ProcessMap(
        {
            0: make_mock_process_info(
                process_id=0,
                model_name=_MODEL,
                state=HordeProcessState.INFERENCE_POST_PROCESSING,
            ),
            # A different model sits resident on an idle sibling: the evictable cross-process room.
            1: make_mock_process_info(
                process_id=1,
                model_name=_OTHER_MODEL,
                state=HordeProcessState.WAITING_FOR_JOB,
            ),
        },
    )

    scheduler = _scheduler_with_post_processing_peak(
        job_tracker,
        monkeypatch=monkeypatch,
        free_vram_mb=_FREE_VRAM_NEEDS_SIBLING_MB,
        peak_for_model=_MODEL,
        process_map=process_map,
    )

    plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None)
    assert plan.action is PostProcessingReclaimAction.EVICT_SIBLING_MODEL
    assert plan.target_process_id == 1


async def test_reduces_context_when_no_idle_sibling_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shortfall remains and no idle sibling holds an evictable model -> stop a context (the rare last rung)."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await job_tracker.mark_inference_started(dispatched)

    process_map = ProcessMap(
        {
            0: make_mock_process_info(
                process_id=0,
                model_name=_MODEL,
                state=HordeProcessState.INFERENCE_POST_PROCESSING,
            ),
            # An idle sibling holding no model: a context to reclaim, but nothing to evict first.
            1: make_mock_process_info(
                process_id=1,
                model_name=None,
                state=HordeProcessState.WAITING_FOR_JOB,
            ),
        },
    )

    scheduler = _scheduler_with_post_processing_peak(
        job_tracker,
        monkeypatch=monkeypatch,
        free_vram_mb=_FREE_VRAM_NEEDS_SIBLING_MB,
        peak_for_model=_MODEL,
        process_map=process_map,
    )

    plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None)
    assert plan.action is PostProcessingReclaimAction.REDUCE_CONTEXT
    assert plan.target_process_id == 1


async def test_insufficient_when_single_process_and_peak_overflows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-process worker on a tiny card cannot reclaim cross-process room -> fault, never park.

    There is no idle sibling to evict and no context to reduce (the one process is running the job), so the
    planner reports the peak as unhostable rather than demanding a teardown it cannot satisfy. The graceful
    fault this drives feeds the post-processing circuit breaker.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await job_tracker.mark_inference_started(dispatched)

    process_map = ProcessMap(
        {
            0: make_mock_process_info(
                process_id=0,
                model_name=_MODEL,
                state=HordeProcessState.INFERENCE_POST_PROCESSING,
            ),
        },
    )

    scheduler = _scheduler_with_post_processing_peak(
        job_tracker,
        monkeypatch=monkeypatch,
        free_vram_mb=_FREE_VRAM_NEEDS_SIBLING_MB,
        peak_for_model=_MODEL,
        process_map=process_map,
    )

    plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None)
    assert plan.action is PostProcessingReclaimAction.INSUFFICIENT


async def test_no_reclaim_when_peak_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ample free VRAM hosts the post-processing peak as-is; no reclaim is planned."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await job_tracker.mark_inference_started(dispatched)

    scheduler = _scheduler_with_post_processing_peak(
        job_tracker,
        monkeypatch=monkeypatch,
        free_vram_mb=_FREE_VRAM_AMPLE_MB,
        peak_for_model=_MODEL,
    )

    plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None)
    assert plan.action is PostProcessingReclaimAction.NONE


class TestPostProcessingReclaimEnactment:
    """Each action carries out the right reclaim primitive, and INSUFFICIENT faults instead of dispatching."""

    async def test_evict_sibling_model_unloads_and_dispatches(self) -> None:
        """The eviction rung reclaims an idle sibling via ``unload_models_from_vram`` and dispatches."""
        job_tracker = JobTracker()
        dispatched = make_job_pop_response(model=_MODEL)
        await track_popped_job_async(job_tracker, dispatched)
        process_map = ProcessMap(
            {
                0: make_mock_process_info(process_id=0, model_name=_MODEL, state=HordeProcessState.INFERENCE_STARTING),
                1: make_mock_process_info(
                    process_id=1,
                    model_name=_OTHER_MODEL,
                    state=HordeProcessState.WAITING_FOR_JOB,
                ),
            },
        )
        scheduler = _make_inference_scheduler(job_tracker=job_tracker, process_map=process_map)
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]

        plan = PostProcessingReclaimPlan(
            action=PostProcessingReclaimAction.EVICT_SIBLING_MODEL,
            target_process_id=1,
            shortfall_mb=2633.0,
        )
        should_dispatch = await scheduler._enact_post_processing_reclaim(
            plan,
            dispatched,
            process_map[0],
            device_index=None,
        )

        assert should_dispatch is True
        scheduler.unload_models_from_vram.assert_called_once()
        assert scheduler.unload_models_from_vram.call_args.kwargs.get("under_pressure") is True

    async def test_reduce_context_scales_down_and_dispatches(self) -> None:
        """The context rung sheds one idle context via ``scale_inference_processes`` and dispatches."""
        job_tracker = JobTracker()
        dispatched = make_job_pop_response(model=_MODEL)
        await track_popped_job_async(job_tracker, dispatched)
        process_map = ProcessMap(
            {
                0: make_mock_process_info(process_id=0, model_name=_MODEL, state=HordeProcessState.INFERENCE_STARTING),
                1: make_mock_process_info(process_id=1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB),
            },
        )
        scheduler = _make_inference_scheduler(job_tracker=job_tracker, process_map=process_map)

        plan = PostProcessingReclaimPlan(
            action=PostProcessingReclaimAction.REDUCE_CONTEXT,
            target_process_id=1,
            shortfall_mb=2633.0,
        )
        should_dispatch = await scheduler._enact_post_processing_reclaim(
            plan,
            dispatched,
            process_map[0],
            device_index=None,
        )

        assert should_dispatch is True
        scheduler._process_lifecycle.scale_inference_processes.assert_called_once()

    async def test_insufficient_faults_and_does_not_dispatch(self) -> None:
        """The unhostable rung faults the job gracefully and signals the caller not to dispatch it."""
        job_tracker = JobTracker()
        dispatched = make_job_pop_response(model=_MODEL)
        await track_popped_job_async(job_tracker, dispatched)
        process_map = ProcessMap(
            {0: make_mock_process_info(process_id=0, model_name=_MODEL, state=HordeProcessState.INFERENCE_STARTING)},
        )
        scheduler = _make_inference_scheduler(job_tracker=job_tracker, process_map=process_map)
        scheduler._job_tracker.handle_job_fault = AsyncMock()  # type: ignore[method-assign]

        plan = PostProcessingReclaimPlan(action=PostProcessingReclaimAction.INSUFFICIENT, shortfall_mb=3000.0)
        should_dispatch = await scheduler._enact_post_processing_reclaim(
            plan,
            dispatched,
            process_map[0],
            device_index=None,
        )

        assert should_dispatch is False
        scheduler._job_tracker.handle_job_fault.assert_awaited_once()

    @pytest.mark.parametrize(
        "action",
        [PostProcessingReclaimAction.NONE, PostProcessingReclaimAction.DELEGATE_IN_PROCESS],
    )
    async def test_no_action_rungs_dispatch_without_reclaim(self, action: PostProcessingReclaimAction) -> None:
        """NONE and DELEGATE_IN_PROCESS dispatch unchanged: ComfyUI handles own-weights in-child."""
        job_tracker = JobTracker()
        dispatched = make_job_pop_response(model=_MODEL)
        await track_popped_job_async(job_tracker, dispatched)
        process_map = ProcessMap(
            {0: make_mock_process_info(process_id=0, model_name=_MODEL, state=HordeProcessState.INFERENCE_STARTING)},
        )
        scheduler = _make_inference_scheduler(job_tracker=job_tracker, process_map=process_map)
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]
        scheduler._job_tracker.handle_job_fault = AsyncMock()  # type: ignore[method-assign]

        plan = PostProcessingReclaimPlan(action=action)
        should_dispatch = await scheduler._enact_post_processing_reclaim(
            plan,
            dispatched,
            process_map[0],
            device_index=None,
        )

        assert should_dispatch is True
        scheduler.unload_models_from_vram.assert_not_called()
        scheduler._process_lifecycle.scale_inference_processes.assert_not_called()
        scheduler._job_tracker.handle_job_fault.assert_not_awaited()


async def test_no_reclaim_when_job_has_no_post_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job with no upscaler/face-fixer reports a 0 peak, so the reserve self-scales away."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await job_tracker.mark_inference_started(dispatched)

    scheduler = _scheduler_with_post_processing_peak(
        job_tracker,
        monkeypatch=monkeypatch,
        free_vram_mb=_FREE_VRAM_NEEDS_SIBLING_MB,
        peak_for_model="a-different-model-entirely",
    )

    plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None)
    assert plan.action is PostProcessingReclaimAction.NONE
