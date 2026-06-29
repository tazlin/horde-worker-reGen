"""The planner's chosen action and the simulated-VRAM physics agree for each representative card layout.

Each case pairs a card and process layout with the action the planner should pick *and* the physical
outcome of enacting that action on the cross-process VRAM ledger (``sim_vram``). Together they assert the
chosen action is the one that actually turns an unhostable post-processing peak into a fitting one (or, on a
single-process tiny card, the one that declines rather than thrashing). These run in-process; the spawned
end-to-end variant exercising a real process replacement lives in
``tests/e2e/test_sim_vram_post_processing_e2e.py`` (CI-gated).
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling import inference_scheduler as inference_scheduler_module
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    InferenceScheduler,
    PostProcessingReclaimAction,
)
from horde_worker_regen.process_management.simulation.sim_vram import (
    SimProcessVram,
    SimVramLedger,
    SimVramSpec,
    simulate_post_processing_allocation,
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

# hordelib's burden shapes for a 4x-upscale SDXL job and the per-resident footprints the ledger charges.
_PP_PEAK_MB = 8533.0
_OWN_WEIGHTS_MB = 4900.0
_CONTEXT_MB = 1354.0


def _scheduler_reading_ledger(
    ledger: SimVramLedger,
    process_map: ProcessMap,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> InferenceScheduler:
    """A budget-active scheduler whose measured free VRAM and burden estimates mirror ``ledger``.

    The planner then decides against the same numbers the simulated card reports, so its action and the
    ledger's physical fit are evaluated on one consistent world.
    """
    bridge_data = make_mock_bridge_data(enable_vram_budget=True, vram_reserve_mb=2048, ram_reserve_mb=4096)
    job_tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        process_map=process_map,
    )
    scheduler._measured_free_vram_mb = Mock(return_value=ledger.device_free_mb(0))  # type: ignore[method-assign]

    def _fake_peak(job: object, baseline: str | None) -> float:
        return _PP_PEAK_MB if getattr(job, "model", None) == _MODEL else 0.0

    def _fake_weight(job: object, baseline: str | None) -> float:
        return _OWN_WEIGHTS_MB if getattr(job, "model", None) == _MODEL else 0.0

    monkeypatch.setattr(inference_scheduler_module, "predict_job_post_processing_vram_mb", _fake_peak)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_weight_mb", _fake_weight)
    return scheduler


def _seed(ledger: SimVramLedger, total_mb: float, processes: list[SimProcessVram]) -> None:
    SimVramSpec(device_index=0, total_vram_mb=total_mb, processes=processes).seed(ledger)


async def _dispatched_job(scheduler: InferenceScheduler) -> object:
    job = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(scheduler._job_tracker, job)
    await scheduler._job_tracker.mark_inference_started(job)
    return job


async def test_contended_16gb_with_warm_sibling_evicts_sibling_and_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contended cell: a 16 GB card with a warm idle sibling -> evict the sibling; the peak then fits."""
    ledger = SimVramLedger.in_process()
    _seed(
        ledger,
        16375.0,
        [
            SimProcessVram(process_id=0, weights_mb=_OWN_WEIGHTS_MB, context_mb=_CONTEXT_MB),  # the running job
            SimProcessVram(process_id=1, weights_mb=_OWN_WEIGHTS_MB, context_mb=_CONTEXT_MB),  # warm idle sibling
            SimProcessVram(process_id=2, weights_mb=0.0, context_mb=_CONTEXT_MB),
            SimProcessVram(process_id=3, weights_mb=0.0, context_mb=_CONTEXT_MB),
        ],
    )
    process_map = ProcessMap(
        {
            0: make_mock_process_info(
                process_id=0, model_name=_MODEL, state=HordeProcessState.INFERENCE_POST_PROCESSING
            ),
            1: make_mock_process_info(process_id=1, model_name=_OTHER_MODEL, state=HordeProcessState.WAITING_FOR_JOB),
            2: make_mock_process_info(process_id=2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB),
            3: make_mock_process_info(process_id=3, model_name=None, state=HordeProcessState.WAITING_FOR_JOB),
        },
    )
    scheduler = _scheduler_reading_ledger(ledger, process_map, monkeypatch=monkeypatch)
    job = await _dispatched_job(scheduler)

    plan = scheduler._plan_post_processing_reclaim(job, device_index=None, dispatching_process_id=0)
    assert plan.action is PostProcessingReclaimAction.EVICT_SIBLING_MODEL
    assert plan.target_process_id == 1

    # As-is the peak does not fit (the stall). Enacting the planned sibling eviction frees cross-process
    # VRAM, and the same allocation now fits.
    assert (
        simulate_post_processing_allocation(ledger, device_index=0, process_id=0, post_processing_peak_mb=_PP_PEAK_MB)
        is False
    )
    ledger.free_own_models(device_index=0, process_id=plan.target_process_id)
    assert (
        simulate_post_processing_allocation(ledger, device_index=0, process_id=0, post_processing_peak_mb=_PP_PEAK_MB)
        is True
    )


async def test_single_process_under_mild_pressure_delegates_and_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Low-proc cell: one process, peak overflows bare headroom but fits once own weights free in-child."""
    # A 16 GB card with ~4 GB spoken for outside the worker (modeled as a reduced usable total).
    ledger = SimVramLedger.in_process()
    _seed(ledger, 12000.0, [SimProcessVram(process_id=0, weights_mb=_OWN_WEIGHTS_MB, context_mb=_CONTEXT_MB)])
    process_map = ProcessMap(
        {
            0: make_mock_process_info(
                process_id=0, model_name=_MODEL, state=HordeProcessState.INFERENCE_POST_PROCESSING
            )
        },
    )
    scheduler = _scheduler_reading_ledger(ledger, process_map, monkeypatch=monkeypatch)
    job = await _dispatched_job(scheduler)

    plan = scheduler._plan_post_processing_reclaim(job, device_index=None, dispatching_process_id=0)
    assert plan.action is PostProcessingReclaimAction.DELEGATE_IN_PROCESS

    # ComfyUI's own free_memory is exactly the in-child own-weights free the ledger models; after it, fits.
    assert (
        simulate_post_processing_allocation(ledger, device_index=0, process_id=0, post_processing_peak_mb=_PP_PEAK_MB)
        is True
    )


async def test_tiny_8gb_single_process_is_insufficient_and_cannot_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiny cell: an 8 GB single-process card cannot host the peak even alone -> INSUFFICIENT (fault, not stall)."""
    ledger = SimVramLedger.in_process()
    _seed(ledger, 8192.0, [SimProcessVram(process_id=0, weights_mb=_OWN_WEIGHTS_MB, context_mb=_CONTEXT_MB)])
    process_map = ProcessMap(
        {
            0: make_mock_process_info(
                process_id=0, model_name=_MODEL, state=HordeProcessState.INFERENCE_POST_PROCESSING
            )
        },
    )
    scheduler = _scheduler_reading_ledger(ledger, process_map, monkeypatch=monkeypatch)
    job = await _dispatched_job(scheduler)

    plan = scheduler._plan_post_processing_reclaim(job, device_index=None, dispatching_process_id=0)
    assert plan.action is PostProcessingReclaimAction.INSUFFICIENT

    # Even freeing its own weights (the most the card can give) leaves the peak unhostable: the planner's
    # graceful fault is the correct call, not a dispatch into a guaranteed stall.
    assert (
        simulate_post_processing_allocation(ledger, device_index=0, process_id=0, post_processing_peak_mb=_PP_PEAK_MB)
        is False
    )


async def test_roomy_24gb_needs_no_reclaim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Roomy cell: a 24 GB card hosts the peak as-is -> NONE, and the allocation fits without any reclaim."""
    ledger = SimVramLedger.in_process()
    _seed(
        ledger,
        24576.0,
        [
            SimProcessVram(process_id=0, weights_mb=_OWN_WEIGHTS_MB, context_mb=_CONTEXT_MB),
            SimProcessVram(process_id=1, weights_mb=_OWN_WEIGHTS_MB, context_mb=_CONTEXT_MB),
        ],
    )
    process_map = ProcessMap(
        {
            0: make_mock_process_info(
                process_id=0, model_name=_MODEL, state=HordeProcessState.INFERENCE_POST_PROCESSING
            ),
            1: make_mock_process_info(process_id=1, model_name=_OTHER_MODEL, state=HordeProcessState.WAITING_FOR_JOB),
        },
    )
    scheduler = _scheduler_reading_ledger(ledger, process_map, monkeypatch=monkeypatch)
    job = await _dispatched_job(scheduler)

    plan = scheduler._plan_post_processing_reclaim(job, device_index=None, dispatching_process_id=0)
    assert plan.action is PostProcessingReclaimAction.NONE
    assert (
        simulate_post_processing_allocation(ledger, device_index=0, process_id=0, post_processing_peak_mb=_PP_PEAK_MB)
        is True
    )
