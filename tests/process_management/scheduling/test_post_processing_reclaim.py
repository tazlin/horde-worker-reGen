"""The scheduler plans room for a job's own post-processing VRAM peak before that peak can stall.

A job's upscaler/face-fixer peak (``predict_job_post_processing_vram_mb``, roughly 8.5 GB for a 4x SDXL
upscale) lands *after* sampling. The VRAM budget admits the job on its sampling footprint alone, so on a
card already carrying warm sibling models and several process contexts the peak allocates into a device
with no headroom and thrashes until the post-processing watchdog reaps the slot. ``free_memory`` inside the
inference child can release only that process's own weights; the sibling models and contexts that fill the
card are cross-process and reclaimable only by the orchestrator.

``_plan_post_processing_reclaim`` therefore sizes the peak against the *effective* post-sampling headroom,
the measured free VRAM less the room in-flight sibling work has committed or will imminently commit (the
same not-yet-realised reserve the concurrent-dispatch gate subtracts), and returns the action for the room
it can actually reclaim:

1. :attr:`PostProcessingReclaimAction.NONE`: the peak fits the effective headroom, so nothing is needed.
2. :attr:`PostProcessingReclaimAction.EVICT_SIBLING_MODEL`: the peak overflows the effective headroom and an
   idle sibling holds an evictable model. Freeing it is preferred over the in-child own-weights delegation,
   because that resident model is the cross-process room the upscaler needs and the child's ``free_memory``
   cannot reach it.
3. :attr:`PostProcessingReclaimAction.DELEGATE_IN_PROCESS`: no reclaimable sibling holds room, but freeing
   the job's own (idle-during-upscale) weights, which the inference child's ``free_memory`` does in-process,
   suffices on an otherwise-uncontended card, so no orchestrator action is needed.
4. :attr:`PostProcessingReclaimAction.REDUCE_CONTEXT`: nothing idle holds an evictable model, so a sibling
   process is stopped to reclaim its (contended-card-scoped) context (the rare last rung).
5. :attr:`PostProcessingReclaimAction.INSUFFICIENT`: no orchestrator-reclaimable room hosts the peak (a
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


async def test_evicts_idle_sibling_over_delegating_when_card_is_contended(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the peak overflows free VRAM and an idle sibling holds an evictable model, free the sibling.

    Freeing the sibling is preferred over delegating to ComfyUI's in-child own-weights free, because on a
    contended card the room the upscaler needs is occupied by the sibling's resident model, which the
    in-child free cannot reach. Even though the own-weights credit would nominally cover the peak here, the
    sibling is the actual contention, so it is evicted.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await job_tracker.mark_inference_started(dispatched)
    # A queued (not in-flight) job reuses the dispatched model so its weights stay demanded.
    await track_popped_job_async(job_tracker, make_job_pop_response(model=_MODEL))

    process_map = ProcessMap(
        {
            0: make_mock_process_info(
                process_id=0,
                model_name=_MODEL,
                state=HordeProcessState.INFERENCE_POST_PROCESSING,
            ),
            # A different model sits resident on an idle sibling: the cross-process room the upscaler needs.
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
        # Peak overflows free but fits once own weights are freed: the rung that used to delegate in-child.
        free_vram_mb=_FREE_VRAM_OWN_WEIGHTS_SUFFICE_MB,
        peak_for_model=_MODEL,
        process_map=process_map,
    )

    plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None, dispatching_process_id=0)
    assert plan.action is PostProcessingReclaimAction.EVICT_SIBLING_MODEL
    assert plan.target_process_id == 1


async def test_committed_sibling_reserve_reclaims_when_measured_free_alone_would_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measured free reading that alone fits the peak still reclaims when a sibling's peak is committed.

    The measured reading lags VRAM an in-flight sibling already post-processing has committed but not yet
    realised. Charging that committed reserve (the effective-free reading) keeps an optimistic measurement
    from letting the peak look like it fits a card that is in fact already spoken-for.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)
    await job_tracker.mark_inference_started(dispatched)
    # An in-flight sibling already in post-processing commits its own peak against the card.
    sibling_job = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, sibling_job)
    await job_tracker.mark_inference_started(sibling_job)

    sibling_in_post_processing = make_mock_process_info(
        process_id=1,
        model_name=_MODEL,
        state=HordeProcessState.INFERENCE_POST_PROCESSING,
    )
    sibling_in_post_processing.last_job_referenced = sibling_job
    process_map = ProcessMap(
        {
            0: make_mock_process_info(process_id=0, model_name=_MODEL, state=HordeProcessState.INFERENCE_STARTING),
            1: sibling_in_post_processing,
            # An idle sibling holding a different model is the reclaimable room once the reserve is charged.
            2: make_mock_process_info(process_id=2, model_name=_OTHER_MODEL, state=HordeProcessState.WAITING_FOR_JOB),
        },
    )

    scheduler = _scheduler_with_post_processing_peak(
        job_tracker,
        monkeypatch=monkeypatch,
        # Ample by the bare measurement (peak fits 12 GB), but the sibling's committed peak eats the headroom.
        free_vram_mb=_FREE_VRAM_AMPLE_MB,
        peak_for_model=_MODEL,
        process_map=process_map,
    )

    plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None, dispatching_process_id=0)
    assert plan.action is not PostProcessingReclaimAction.NONE, (
        "the committed sibling reserve should pull effective free below the peak and trigger reclaim"
    )
    assert plan.action is PostProcessingReclaimAction.EVICT_SIBLING_MODEL
    assert plan.target_process_id == 2


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

    async def test_reduce_context_targets_the_contended_card_on_multi_gpu(self) -> None:
        """On a multi-GPU host the context reduction is sized from the contended card, not the worker-wide pool.

        A worker-wide count would leave the per-card scale a no-op (the target never drops below the card's
        own context count) and the peak would go unrelieved. Card 1 holds two contexts (one busy, one idle)
        and only one job runs on it, so the reduction targets one context on card 1.
        """
        job_tracker = JobTracker()
        dispatched = make_job_pop_response(model=_MODEL)
        await track_popped_job_async(job_tracker, dispatched)
        await job_tracker.mark_inference_started(dispatched)
        # A second in-flight job on the other card: a worker-wide "needed" of 2 would wrongly hold card 1 at 2.
        other_card_job = make_job_pop_response(model=_OTHER_MODEL)
        await track_popped_job_async(job_tracker, other_card_job)
        await job_tracker.mark_inference_started(other_card_job)

        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    process_id=0,
                    model_name=_OTHER_MODEL,
                    state=HordeProcessState.INFERENCE_STARTING,
                    device_index=0,
                ),
                1: make_mock_process_info(
                    process_id=1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0
                ),
                2: make_mock_process_info(
                    process_id=2,
                    model_name=_MODEL,
                    state=HordeProcessState.INFERENCE_STARTING,
                    device_index=1,
                ),
                3: make_mock_process_info(
                    process_id=3, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=1
                ),
            },
        )
        scheduler = _make_inference_scheduler(job_tracker=job_tracker, process_map=process_map, max_inference=4)

        plan = PostProcessingReclaimPlan(
            action=PostProcessingReclaimAction.REDUCE_CONTEXT,
            target_process_id=3,
            shortfall_mb=2633.0,
        )
        should_dispatch = await scheduler._enact_post_processing_reclaim(
            plan,
            dispatched,
            process_map[2],
            device_index=1,
        )

        assert should_dispatch is True
        scheduler._process_lifecycle.scale_inference_processes.assert_called_once_with(1, device_index=1)

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
