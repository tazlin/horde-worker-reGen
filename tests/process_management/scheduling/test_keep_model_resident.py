"""Tests for governed VRAM retention across jobs.

hordelib evicts a job's model from VRAM after every run so sibling GPU instances never collectively
over-commit. That eviction forces a RAM->VRAM reload on the next job, the dominant non-sampling cost on
small jobs (a same-model successor on the same process re-uploads weights that were still on the card).
:meth:`InferenceScheduler._should_keep_model_resident` decides when to suppress that eviction for one
dispatch. Because eviction is now on-demand and *proven* (the device-free governor reads truthful NVML
device-free and the verified reclaim ladder takes residents back rung by rung, each free confirmed at the
device level), retention no longer has to be preemptively stingy:

- **Card healthy**: the device-free governor's committed state for the card is HEALTHY. A PRESSURE or
  SATURATED card is one the ladder is or may soon be reclaiming from, so it is handed no new resident.
- **Static fit**: the card's reported total VRAM (a constant the driver cannot misreport) must absorb the
  job's sampling peak plus the reserve, after charging sibling CUDA contexts and the job's own
  post-processing that share the card while the weights are held.

The measured admission floor is deliberately not re-checked in this seam (that is the admission/dispatch
gate's job; retaining already-materialized weights adds zero new bytes), and sole residency is not
required: a second idle resident is safe because it is a first-class reclaim-ladder candidate. Eviction is
just-in-time: a cross-model preload that no longer fits because idle residents hold the card defers while
the ladder evicts them.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import (
    HeldComponentSnapshot,
    HordeControlFlag,
    HordeInferenceControlMessage,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.scheduling import inference_scheduler as inference_scheduler_module
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MODEL = "WAI-NSFW-illustrious-SDXL"
_OTHER_MODEL = "CyberRealistic Pony"
_TOTAL_VRAM_MB = 16376
_PROCESS_ID = 2
_SIBLING_PROCESS_ID = 3


def _dispatch_process(model_name: str | None = _MODEL):  # noqa: ANN202
    """The process receiving the dispatch, reporting the card's total VRAM (needed by the static gate)."""
    process_info = make_mock_process_info(_PROCESS_ID, model_name=model_name)
    process_info.total_vram_mb = _TOTAL_VRAM_MB
    return process_info


def _budget_on_scheduler(
    job_tracker: JobTracker,
    *,
    process_map: ProcessMap | None = None,
    horde_model_map: HordeModelMap | None = None,
    legacy_comfy_vram_unload: bool = False,
) -> InferenceScheduler:
    """A scheduler with the VRAM budget active, the governor unsampled (HEALTHY), and repeat evidence seeded.

    Retention is also gated on the dispatched model having repeated in the slot's trailing dispatches, which
    every test here would otherwise fail at before reaching the gate it is about. Seeding that evidence for
    both models on both slots is what keeps each test a statement about its own gate; the evidence gate itself
    is exercised by the tests that build a scheduler with an unseeded slot.
    """
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
        legacy_comfy_vram_unload=legacy_comfy_vram_unload,
    )
    scheduler = _make_inference_scheduler(
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        process_map=process_map if process_map is not None else ProcessMap({_PROCESS_ID: _dispatch_process()}),
        horde_model_map=horde_model_map,
    )
    for process_id in (_PROCESS_ID, _SIBLING_PROCESS_ID):
        for model in (_MODEL, _OTHER_MODEL):
            _seed_repeat_evidence(scheduler, process_id, model)
    return scheduler


def _seed_repeat_evidence(scheduler: InferenceScheduler, process_id: int, model: str) -> None:
    """Record a prior dispatch of ``model`` on ``process_id``, the evidence a retention grant is gated on."""
    scheduler._record_slot_dispatch(process_id, model)


def _map_with_model_on_process(
    model: str,
    process_id: int,
    load_state: ModelLoadState = ModelLoadState.LOADED_IN_VRAM,
) -> HordeModelMap:
    model_map = HordeModelMap(root={})
    model_map.update_entry(model, load_state=load_state, process_id=process_id)
    return model_map


async def test_retains_when_healthy_and_budget_fits_without_queue_lookahead() -> None:
    """A healthy card with a fitting budget retains, even with an empty pending queue.

    The pop cycle refills the queue after a dispatch drains it, so requiring a visible same-model
    successor would make retention structurally unreachable.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )


async def test_retains_even_when_another_process_holds_a_resident_model() -> None:
    """Sole residency is no longer required: a sibling resident does not deny retention on a healthy card.

    A second idle resident is safe under the governed policy because it is a first-class reclaim-ladder
    candidate that the verified ladder takes back on demand; the old WDDM-driven sole-residency denial is
    superseded.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=_OTHER_MODEL)
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    model_map = _map_with_model_on_process(_OTHER_MODEL, _SIBLING_PROCESS_ID)
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map, horde_model_map=model_map)
    # Price the sibling's CUDA context so the static gate is decided on fit, not on an unmeasured charge.
    scheduler._overhead.set_marginal_overhead_mb(1354.0)
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )


async def test_no_retain_when_governor_reports_pressure() -> None:
    """A PRESSURE card denies retention: the reclaim ladder may soon reclaim, so it gains no new resident."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    scheduler.set_governor_state(0, GovernorState.PRESSURE)
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_no_retain_when_governor_reports_saturated() -> None:
    """A SATURATED card denies retention: the ladder is running, so it gains no new resident to evict."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    scheduler.set_governor_state(0, GovernorState.SATURATED)
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_no_retain_when_budget_inactive() -> None:
    """Without the VRAM budget the worker cannot vouch for the headroom, so it evicts as before."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    # Default mock bridge_data leaves enable_vram_budget unset (non-bool), so _budget_active() is False.
    scheduler = _make_inference_scheduler(
        job_tracker=job_tracker,
        process_map=ProcessMap({_PROCESS_ID: _dispatch_process()}),
    )
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_no_retain_when_total_vram_unreported() -> None:
    """Without a reported card total the static fit cannot be judged, so retention is refused."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    unreporting = make_mock_process_info(_PROCESS_ID, model_name=_MODEL)  # total_vram_mb defaults to 0
    scheduler = _budget_on_scheduler(job_tracker, process_map=ProcessMap({_PROCESS_ID: unreporting}))

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=unreporting, device_index=None) is False
    )


async def test_sibling_context_overhead_is_charged_against_the_static_fit(monkeypatch) -> None:  # noqa: ANN001
    """Live sibling GPU processes cost a context each, and the static gate must charge them.

    A CUDA context is held whether or not the sibling holds a model, and it is invisible to both the
    sampling-peak estimate and the committed ledger. Without this charge, retention granted alongside a
    process pool plus the post-processing lane commits VRAM the card does not have, and the overflow is
    silent driver paging on WDDM rather than a visible failure.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=None)
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map)
    scheduler._overhead.set_marginal_overhead_mb(1354.0)
    seen_free: list[float] = []

    def record_check(job, baseline, free_vram_mb, committed_reserve_mb=0.0, *, disaggregated=False):  # noqa: ANN001, ANN202
        seen_free.append(free_vram_mb)
        return Mock(fits=True, predicted_mb=None, reserve_mb=0.0)

    scheduler._vram_budget.check_job = record_check  # type: ignore[method-assign]
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )
    assert seen_free == [float(_TOTAL_VRAM_MB) - 1354.0]


async def test_no_retain_when_context_overhead_unmeasured_with_siblings() -> None:
    """Sibling contexts whose cost has not been measured deny retention rather than charging zero."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=None)
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map)
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_jobs_own_post_processing_peak_is_charged_against_the_static_fit(monkeypatch) -> None:  # noqa: ANN001
    """A post-processing job's upscaler peak is charged at grant time, not when its reserve lands.

    The chain runs right after sampling, precisely while retention holds the weights, but its committed
    reserve only registers once inference finishes: one dispatch too late for this grant.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL, post_processing=["RealESRGAN_x4plus"])
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    monkeypatch.setattr(
        inference_scheduler_module,
        "predict_job_post_processing_vram_mb",
        lambda job, baseline: 3100.0,
    )
    seen_free: list[float] = []

    def record_check(job, baseline, free_vram_mb, committed_reserve_mb=0.0, *, disaggregated=False):  # noqa: ANN001, ANN202
        seen_free.append(free_vram_mb)
        return Mock(fits=True, predicted_mb=None, reserve_mb=0.0)

    scheduler._vram_budget.check_job = record_check  # type: ignore[method-assign]
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )
    assert seen_free[0] == float(_TOTAL_VRAM_MB) - 3100.0


async def test_soak_geometry_sdxl_peak_retains_on_a_16gb_card_with_a_sibling() -> None:
    """The soak geometry: SDXL peak 8258 on a 16375MB card with a sibling context retains.

    The old static fit stacked the full operator reserve (4096) on top of the learned peak plus the sibling
    context charge, denying retention on a 16GB card by a few dozen MB and forcing a weight re-transfer every
    job. Charging only the measurement noise buffer instead flips it to a grant, which the baseline tree ran
    with zero corroborated paging.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=None)
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map)
    scheduler._overhead.set_marginal_overhead_mb(2030.0)

    def peak_8258(job, baseline, free_vram_mb, committed_reserve_mb=0.0, *, disaggregated=False):  # noqa: ANN001, ANN202
        return Mock(fits=free_vram_mb >= 8258.0 + 4096.0, predicted_mb=8258.0, reserve_mb=4096.0)

    scheduler._vram_budget.check_job = peak_8258  # type: ignore[method-assign]
    process_info = scheduler._process_map[_PROCESS_ID]

    # check_job's own fits (peak + operator reserve 4096) would deny: 8258 + 4096 = 12354 > 16375 - 2030 = 14345
    # is false, but net of only the noise buffer it fits, so the de-stacked gate grants.
    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )


async def test_conflicting_sibling_working_set_still_denies_retention() -> None:
    """A peak that cannot coexist with a sibling's resident working set is still refused.

    De-stacking the operator reserve does not remove the protection: a genuinely too-large peak (net of the
    sibling contexts already charged and the noise buffer) still fails the static fit.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=_OTHER_MODEL)
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    model_map = _map_with_model_on_process(_OTHER_MODEL, _SIBLING_PROCESS_ID)
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map, horde_model_map=model_map)
    scheduler._overhead.set_marginal_overhead_mb(2030.0)

    # A peak larger than the card net of the sibling context and the noise buffer cannot coexist.
    def oversized_peak(job, baseline, free_vram_mb, committed_reserve_mb=0.0, *, disaggregated=False):  # noqa: ANN001, ANN202
        return Mock(fits=False, predicted_mb=15000.0, reserve_mb=4096.0)

    scheduler._vram_budget.check_job = oversized_peak  # type: ignore[method-assign]
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_no_retain_when_footprint_exceeds_the_static_fit() -> None:
    """A sampling peak that cannot fit the card even net of the noise buffer refuses retention.

    Retention charges the learned peak against the card total (net of sibling contexts and the job's own
    post-processing) plus the measurement noise buffer, not the operator reserve. A peak that overshoots that
    static fit is refused, so retention never starves a genuinely-conflicting swap.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    # A peak larger than the whole card cannot fit even net of only the noise buffer.
    scheduler._vram_budget.check_job = Mock(  # type: ignore[method-assign]
        return_value=Mock(fits=False, predicted_mb=float(_TOTAL_VRAM_MB), reserve_mb=0.0),
    )
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_no_retain_while_wddm_paging_active() -> None:
    """A measured demand-paging verdict denies retention regardless of every other gate passing."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    scheduler.note_wddm_paging({12345: 900.0}, active=True)
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_wddm_paging_rising_edge_reclaims_idle_vram_once() -> None:
    """The paging verdict's rising edge reclaims idle residents (LIFO) once per edge, NOT protecting the flag.

    Under WDDM the driver demotes the least-recently-touched allocator, so the PDH-flagged process is usually
    the idle newcomer, not the active sampler. The reworked sweep therefore evicts the flagged idle resident
    rather than protecting it, routed through the same reclaim actuator the governor's ladder uses.
    """
    job_tracker = JobTracker()
    scheduler = _budget_on_scheduler(job_tracker)
    paging_process = scheduler._process_map[_PROCESS_ID]
    unloaded: list[int] = []

    def record_unload(process_id, device_index=None):  # noqa: ANN001, ANN202
        unloaded.append(process_id)
        return True

    scheduler.unload_idle_model = record_unload  # type: ignore[method-assign]

    paging_pid = paging_process.os_pid
    assert paging_pid is not None
    scheduler.note_wddm_paging({paging_pid: 900.0}, active=True)
    scheduler.note_wddm_paging({paging_pid: 900.0}, active=True)
    scheduler.note_wddm_paging({}, active=False)
    scheduler.note_wddm_paging({paging_pid: 900.0}, active=True)

    # One reclaim per rising edge (not per tick); the PDH-flagged idle resident is a target, not protected.
    assert unloaded == [_PROCESS_ID, _PROCESS_ID]


async def test_same_model_redispatch_leaves_the_map_residency_intact() -> None:
    """Granting retention for a model already VRAM-resident on the process does not regress the map to RAM.

    The retention decision only sets the defer-unload flag; the parent's model-map entry keeps its
    LOADED_IN_VRAM state, which is what lets the child skip the RAM->VRAM re-transfer on the next same-model
    job (its cache still holds the weights).
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    model_map = _map_with_model_on_process(_MODEL, _PROCESS_ID)
    scheduler = _budget_on_scheduler(job_tracker, horde_model_map=model_map)
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )
    entry = scheduler._horde_model_map.root[_MODEL]
    assert entry.process_id == _PROCESS_ID
    assert entry.horde_model_load_state is ModelLoadState.LOADED_IN_VRAM


async def test_idle_retained_resident_is_evicted_on_demand_for_a_cross_model_head(monkeypatch) -> None:  # noqa: ANN001
    """A retained idle resident is a reclaim-ladder candidate the head-of-queue eviction targets.

    Process A holds model X in VRAM (retained under the healthy policy) while idle; the head job wants a
    different model Y on process B. The card cannot hold both, so the eviction routed through the
    single-owner reclaim path targets A, and once A's copy is freed the room it held is no longer a ladder
    candidate (available for Y).
    """
    process_a = make_mock_process_info(1, model_name=_MODEL)
    process_a.total_vram_mb = _TOTAL_VRAM_MB
    requester_b = make_mock_process_info(2, model_name=None)
    requester_b.total_vram_mb = _TOTAL_VRAM_MB
    process_map = ProcessMap({1: process_a, 2: requester_b})
    model_map = _map_with_model_on_process(_MODEL, 1)

    scheduler = _make_inference_scheduler(
        process_map=process_map,
        horde_model_map=model_map,
        job_tracker=JobTracker(),
        bridge_data=make_mock_bridge_data(enable_vram_budget=True, max_threads=2),
        max_concurrent=2,
        max_inference=2,
    )
    monkeypatch.setattr(scheduler, "get_next_n_models", lambda n: [_OTHER_MODEL])

    candidates = scheduler.build_reclaim_ladder_candidates(None)
    assert 1 in {resident.process_id for resident in candidates.idle_residents}

    freed = scheduler.unload_models_from_vram(requester_b, under_pressure=True, for_head_of_queue=True)
    assert freed is True
    assert process_a.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM

    # Simulate the child reporting the model gone: the freed room is no longer an idle-resident candidate.
    process_a.loaded_horde_model_name = None
    remaining = scheduler.build_reclaim_ladder_candidates(None)
    assert 1 not in {resident.process_id for resident in remaining.idle_residents}


async def test_sibling_retained_resident_is_charged_and_can_deny_the_static_fit(monkeypatch) -> None:  # noqa: ANN001
    """Weights another slot holds under an earlier grant are charged against this grant's static fit.

    Retention is cumulative: a granted model stays on the card until an eviction actuates, so a later
    grant's sampling peak has to fit beside it. A fit that counts only live contexts prices each grant as
    if it were the only one, and a run of grants across sibling slots then sums past the card.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=_OTHER_MODEL)
    sibling.retained_resident_model = _OTHER_MODEL
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map)
    scheduler._overhead.set_marginal_overhead_mb(2030.0)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)

    def peak_8258(job, baseline, free_vram_mb, committed_reserve_mb=0.0, *, disaggregated=False):  # noqa: ANN001, ANN202
        return Mock(fits=False, predicted_mb=8258.0, reserve_mb=4096.0)

    scheduler._vram_budget.check_job = peak_8258  # type: ignore[method-assign]
    process_info = scheduler._process_map[_PROCESS_ID]

    # 16375 - 2030 (sibling context) - 6800 (sibling's retained weights) leaves 7545MB, under the 8258 peak.
    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_static_fit_grants_when_the_sibling_retains_nothing(monkeypatch) -> None:  # noqa: ANN001
    """The same geometry grants once the sibling holds no retained weights: only the context is charged."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=_OTHER_MODEL)
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map)
    scheduler._overhead.set_marginal_overhead_mb(2030.0)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)

    def peak_8258(job, baseline, free_vram_mb, committed_reserve_mb=0.0, *, disaggregated=False):  # noqa: ANN001, ANN202
        return Mock(fits=False, predicted_mb=8258.0, reserve_mb=4096.0)

    scheduler._vram_budget.check_job = peak_8258  # type: ignore[method-assign]
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )


async def test_same_model_re_grant_does_not_charge_its_own_retained_weights(monkeypatch) -> None:  # noqa: ANN001
    """A same-model streak's re-grant charges nothing for the weights it is reusing.

    The retained weights and the dispatched job's weights are the same bytes; double-charging them would
    deny exactly the case retention exists to serve.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    process_info = scheduler._process_map[_PROCESS_ID]
    process_info.retained_resident_model = _MODEL
    seen_free: list[float] = []

    def record_check(job, baseline, free_vram_mb, committed_reserve_mb=0.0, *, disaggregated=False):  # noqa: ANN001, ANN202
        seen_free.append(free_vram_mb)
        return Mock(fits=True, predicted_mb=None, reserve_mb=0.0)

    scheduler._vram_budget.check_job = record_check  # type: ignore[method-assign]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )
    assert seen_free == [float(_TOTAL_VRAM_MB)]


async def test_own_cross_model_retained_resident_is_charged(monkeypatch) -> None:  # noqa: ANN001
    """A slot's own retained weights for a *different* model are still on the card and are charged."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    process_info = scheduler._process_map[_PROCESS_ID]
    process_info.retained_resident_model = _OTHER_MODEL
    seen_free: list[float] = []

    def record_check(job, baseline, free_vram_mb, committed_reserve_mb=0.0, *, disaggregated=False):  # noqa: ANN001, ANN202
        seen_free.append(free_vram_mb)
        return Mock(fits=True, predicted_mb=None, reserve_mb=0.0)

    scheduler._vram_budget.check_job = record_check  # type: ignore[method-assign]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )
    assert seen_free == [float(_TOTAL_VRAM_MB) - 6800.0]


async def test_unpriceable_retained_resident_denies_retention(monkeypatch) -> None:  # noqa: ANN001
    """A tracked resident whose footprint cannot be estimated denies the grant rather than charging zero."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: None)
    process_info = scheduler._process_map[_PROCESS_ID]
    process_info.retained_resident_model = _OTHER_MODEL

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


def _record_retention_reasons(scheduler: InferenceScheduler) -> list[str]:
    """Capture the reason text of every retention verdict the scheduler emits."""
    reasons: list[str] = []

    def capture(model, process_with_model, *, granted, reason, denial_reason=None):  # noqa: ANN001, ANN202
        reasons.append(reason)

    scheduler._log_retention_decision = capture  # type: ignore[method-assign]
    return reasons


async def test_retention_decision_reports_the_retained_charge(monkeypatch) -> None:  # noqa: ANN001
    """The grant/denial verdict carries the retained-resident charge, so it names what held the card."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    process_info = scheduler._process_map[_PROCESS_ID]
    process_info.retained_resident_model = _OTHER_MODEL
    scheduler._vram_budget.check_job = Mock(  # type: ignore[method-assign]
        return_value=Mock(fits=True, predicted_mb=1000.0, reserve_mb=0.0),
    )
    reasons = _record_retention_reasons(scheduler)

    scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None)

    assert any("retained residents 6800MB" in reason for reason in reasons)


async def test_retention_decision_omits_a_zero_retained_charge() -> None:
    """With nothing retained the verdict keeps its existing composition; the charge is not reported as zero."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    process_info = scheduler._process_map[_PROCESS_ID]
    reasons = _record_retention_reasons(scheduler)

    scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None)

    assert len(reasons) == 1
    assert "retained residents" not in reasons[0]


def test_granted_completion_records_the_retained_resident() -> None:
    """A job dispatched with a grant leaves its model recorded as this slot's retained resident."""
    process_info = make_mock_process_info(_PROCESS_ID, model_name=_MODEL)
    process_info.note_retention_grant(_MODEL)

    process_info.settle_retention_after_job()

    assert process_info.retained_resident_model == _MODEL
    assert process_info.retention_granted_model is None


def test_ungranted_completion_clears_the_retained_resident() -> None:
    """An ungranted job ends with the explicit evictor returning the card, so the slot retains nothing."""
    process_info = make_mock_process_info(_PROCESS_ID, model_name=_MODEL)
    process_info.retained_resident_model = _MODEL
    process_info.note_retention_grant(None)

    process_info.settle_retention_after_job()

    assert process_info.retained_resident_model is None


def test_granted_completion_for_another_model_replaces_the_retained_resident() -> None:
    """A granted job for a different model replaces what the slot is recorded as holding."""
    process_info = make_mock_process_info(_PROCESS_ID, model_name=_OTHER_MODEL)
    process_info.retained_resident_model = _MODEL
    process_info.note_retention_grant(_OTHER_MODEL)

    process_info.settle_retention_after_job()

    assert process_info.retained_resident_model == _OTHER_MODEL


def test_process_death_clears_the_retained_resident() -> None:
    """A dead slot's device weights go with it, so it is no longer charged or evicted as a resident."""
    process_info = make_mock_process_info(_PROCESS_ID, model_name=_MODEL)
    process_info.retained_resident_model = _MODEL
    process_map = ProcessMap({_PROCESS_ID: process_info})

    process_map.on_process_ending(_PROCESS_ID)

    assert process_info.retained_resident_model is None


async def test_ladder_unload_clears_the_retained_resident() -> None:
    """The reclaim ladder's idle-model unload takes the slot out of the retained-resident set."""
    job_tracker = JobTracker()
    scheduler = _budget_on_scheduler(job_tracker)
    process_info = scheduler._process_map[_PROCESS_ID]
    process_info.retained_resident_model = _MODEL

    assert scheduler.unload_idle_model(_PROCESS_ID) is True
    assert process_info.retained_resident_model is None


async def test_paging_reclaim_clears_the_retained_resident() -> None:
    """The demand-paging rising edge reclaims through the ladder actuator, clearing the residency record."""
    job_tracker = JobTracker()
    scheduler = _budget_on_scheduler(job_tracker)
    process_info = scheduler._process_map[_PROCESS_ID]
    process_info.retained_resident_model = _MODEL

    paging_pid = process_info.os_pid
    assert paging_pid is not None
    scheduler.note_wddm_paging({paging_pid: 900.0}, active=True)

    assert process_info.retained_resident_model is None


async def test_cross_model_dispatch_evicts_the_retained_resident_first() -> None:
    """A dispatch for a different model returns the retained weights to the card before the new load.

    Without this the slot carries both models' weights through the job, which is the overcommit the
    static fit cannot price away once it has already happened.
    """
    job_tracker = JobTracker()
    scheduler = _budget_on_scheduler(job_tracker)
    process_info = scheduler._process_map[_PROCESS_ID]
    process_info.retained_resident_model = _OTHER_MODEL

    assert scheduler.evict_retained_resident_for_model_change(process_info, _MODEL) is True
    assert process_info.retained_resident_model is None
    sent = process_info.pipe_connection.send.call_args.args[0]  # pyrefly: ignore
    assert sent.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
    assert sent.horde_model_name == _OTHER_MODEL


async def test_same_model_dispatch_does_not_evict_the_retained_resident() -> None:
    """A same-model dispatch keeps the retained weights: reusing them is the whole saving."""
    job_tracker = JobTracker()
    scheduler = _budget_on_scheduler(job_tracker)
    process_info = scheduler._process_map[_PROCESS_ID]
    process_info.retained_resident_model = _MODEL

    assert scheduler.evict_retained_resident_for_model_change(process_info, _MODEL) is False
    assert process_info.retained_resident_model == _MODEL
    process_info.pipe_connection.send.assert_not_called()  # type: ignore[attr-defined]


async def test_legacy_comfy_unload_denies_retention_with_its_own_reason() -> None:
    """The legacy regime denies every grant, named as such: the child unloads below anything a grant reaches."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker, legacy_comfy_vram_unload=True)
    process_info = scheduler._process_map[_PROCESS_ID]
    reasons = _record_retention_reasons(scheduler)

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )
    assert any("legacy_comfy_vram_unload" in reason for reason in reasons)


async def test_legacy_regime_records_no_retained_resident_across_a_dispatch() -> None:
    """Through a full dispatch and completion under the legacy regime the slot is never tracked as retaining.

    The tracking is what later dispatches wait on and are charged for, so a phantom entry for weights the
    child has already unloaded would hold and price the card against memory that is free.
    """
    job = make_job_pop_response(model=_MODEL)
    job_tracker = JobTracker()
    await track_popped_job_async(job_tracker, job)
    process_info = _dispatch_process()
    process_info.last_process_state = HordeProcessState.PRELOADED_MODEL
    scheduler = _budget_on_scheduler(
        job_tracker,
        process_map=ProcessMap({_PROCESS_ID: process_info}),
        horde_model_map=_map_with_model_on_process(_MODEL, _PROCESS_ID),
        legacy_comfy_vram_unload=True,
    )

    assert await scheduler.start_inference() is True
    assert process_info.retention_granted_model is None

    process_info.settle_retention_after_job()
    assert process_info.retained_resident_model is None


async def test_actuated_regime_records_the_grant_across_the_same_dispatch() -> None:
    """The same dispatch with the hatch off does record the grant: the gate, not the geometry, is the change."""
    job = make_job_pop_response(model=_MODEL)
    job_tracker = JobTracker()
    await track_popped_job_async(job_tracker, job)
    process_info = _dispatch_process()
    process_info.last_process_state = HordeProcessState.PRELOADED_MODEL
    scheduler = _budget_on_scheduler(
        job_tracker,
        process_map=ProcessMap({_PROCESS_ID: process_info}),
        horde_model_map=_map_with_model_on_process(_MODEL, _PROCESS_ID),
    )

    assert await scheduler.start_inference() is True
    assert process_info.retention_granted_model == _MODEL

    process_info.settle_retention_after_job()
    assert process_info.retained_resident_model == _MODEL


def _admission_scheduler(
    *,
    target_retains: str | None = None,
    sibling_retains: str | None = None,
    predicted_peak_mb: float,
) -> tuple[InferenceScheduler, HordeProcessInfo, HordeProcessInfo]:
    """A two-slot card whose dispatch target loads weights beside a sibling's tracked retained residents.

    The static geometry: a 16376MB card, one sibling CUDA context at 2030MB, retained weights priced at
    6800MB each, and a sampling peak the caller chooses to sit either side of what is left.
    """
    target = make_mock_process_info(_PROCESS_ID, model_name=None)
    target.total_vram_mb = _TOTAL_VRAM_MB
    target.retained_resident_model = target_retains
    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=sibling_retains)
    sibling.total_vram_mb = _TOTAL_VRAM_MB
    sibling.retained_resident_model = sibling_retains
    sibling.process_reserved_mb = 7000
    horde_model_map = HordeModelMap(root={})
    if sibling_retains is not None:
        horde_model_map.update_entry(
            sibling_retains,
            load_state=ModelLoadState.LOADED_IN_VRAM,
            process_id=_SIBLING_PROCESS_ID,
        )
    scheduler = _budget_on_scheduler(
        JobTracker(),
        process_map=ProcessMap({_PROCESS_ID: target, _SIBLING_PROCESS_ID: sibling}),
        horde_model_map=horde_model_map,
    )
    scheduler._overhead.set_marginal_overhead_mb(2030.0)
    scheduler._vram_budget.check_job = Mock(  # type: ignore[method-assign]
        return_value=Mock(fits=False, predicted_mb=predicted_peak_mb, reserve_mb=4096.0),
    )
    return scheduler, target, sibling


async def test_dispatch_beside_a_non_fitting_retained_resident_defers_and_evicts(monkeypatch) -> None:  # noqa: ANN001
    """A load that cannot fit beside a sibling's retained weights holds, and asks for those weights back.

    16376 total, less the sibling context (2030) and its retained weights (6800), leaves 7546MB: an 8258MB
    sampling peak does not fit, so the dispatch is held rather than materialised into occupied memory.
    """
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    scheduler, target, sibling = _admission_scheduler(sibling_retains=_OTHER_MODEL, predicted_peak_mb=8258.0)
    job = make_job_pop_response(model=_MODEL)

    assert scheduler._retained_resident_dispatch_holds(job, target) is True
    assert sibling.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
    assert sibling.retained_resident_model is None
    assert target.last_control_flag != HordeControlFlag.START_INFERENCE


async def test_dispatch_stays_held_until_the_child_evidences_the_free(monkeypatch) -> None:  # noqa: ANN001
    """The hold releases on the child's own post-free report, never on the eviction merely having been sent.

    Retention tracking clears the instant the unload goes out, so a dispatch that trusted it would load into
    memory the card has not returned yet. The child frees, then reports its fallen reservation; that report
    is what admits the dispatch.
    """
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    scheduler, target, sibling = _admission_scheduler(sibling_retains=_OTHER_MODEL, predicted_peak_mb=8258.0)
    job = make_job_pop_response(model=_MODEL)

    assert scheduler._retained_resident_dispatch_holds(job, target) is True
    # Nothing has come back from the child yet: the room is owed, not returned.
    assert scheduler._retained_resident_dispatch_holds(job, target) is True

    sibling.process_reserved_mb = 200
    assert scheduler._retained_resident_dispatch_holds(job, target) is False


async def test_dispatch_that_fits_beside_the_retained_resident_proceeds_untouched(monkeypatch) -> None:  # noqa: ANN001
    """A peak that fits beside the retained weights dispatches with no eviction: retention keeps its value."""
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    scheduler, target, sibling = _admission_scheduler(sibling_retains=_OTHER_MODEL, predicted_peak_mb=3000.0)
    job = make_job_pop_response(model=_MODEL)

    assert scheduler._retained_resident_dispatch_holds(job, target) is False
    assert sibling.retained_resident_model == _OTHER_MODEL
    sibling.pipe_connection.send.assert_not_called()  # type: ignore[attr-defined]


async def test_dispatch_onto_the_retaining_slot_itself_is_not_double_evicted(monkeypatch) -> None:  # noqa: ANN001
    """A cross-model dispatch onto a retaining slot keeps its own pre-load eviction and is not held for it.

    Those weights are returned ahead of the same slot's START_INFERENCE, so charging them here would hold a
    dispatch against a tenant that is already leaving, and evicting them here would ask for them twice.
    """
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    scheduler, target, _sibling = _admission_scheduler(target_retains=_OTHER_MODEL, predicted_peak_mb=8258.0)
    job = make_job_pop_response(model=_MODEL)

    assert scheduler._retained_resident_dispatch_holds(job, target) is False
    target.pipe_connection.send.assert_not_called()  # type: ignore[attr-defined]

    assert scheduler.evict_retained_resident_for_model_change(target, _MODEL) is True
    assert target.pipe_connection.send.call_count == 1  # type: ignore[attr-defined]


async def test_dispatch_onto_the_slot_retaining_this_model_loads_nothing(monkeypatch) -> None:  # noqa: ANN001
    """Seating a job on the slot that already holds its weights materialises nothing, so nothing is held."""
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    scheduler, target, sibling = _admission_scheduler(
        target_retains=_MODEL,
        sibling_retains=_OTHER_MODEL,
        predicted_peak_mb=8258.0,
    )
    job = make_job_pop_response(model=_MODEL)

    assert scheduler._retained_resident_dispatch_holds(job, target) is False
    assert sibling.retained_resident_model == _OTHER_MODEL


def test_inference_control_message_defaults_to_eviction() -> None:
    """The dispatch message preserves today's aggressive eviction unless the scheduler opts in."""
    message = HordeInferenceControlMessage(
        control_flag=HordeControlFlag.START_INFERENCE,
        horde_model_name=_MODEL,
        sdk_api_job_info=make_job_pop_response(model=_MODEL),
    )

    assert message.keep_model_resident_after is False


def test_inference_control_message_carries_no_device_reading_by_default() -> None:
    """A dispatch built without a device reading leaves the child on its own free view, as before."""
    message = HordeInferenceControlMessage(
        control_flag=HordeControlFlag.START_INFERENCE,
        horde_model_name=_MODEL,
        sdk_api_job_info=make_job_pop_response(model=_MODEL),
    )

    assert message.device_free_mb is None


_SDXL_UNET_RESIDUAL_BYTES = 5 * 1024 * 1024 * 1024
"""An SDXL-class UNet residual (~5GB), the tensor bytes a disaggregated sampler's checkpoint leaves it holding."""


class _Sidecar:
    """A component-identity sidecar stand-in: only its residual tensor bytes are read by the charge."""

    def __init__(self, residual_tensor_bytes: int) -> None:
        self.residual_tensor_bytes = residual_tensor_bytes


async def test_a_component_only_retained_resident_is_charged_its_unet_alone(monkeypatch) -> None:  # noqa: ANN001
    """A disaggregated sampler's residency is priced at its UNet, so a grant the whole checkpoint denies fits.

    The same geometry the whole-checkpoint charge denies (see the sibling-charge test above: 16375 less the
    2030 sibling context and 6800 of retained weights leaves 7545MB against an 8258MB peak). A disaggregated
    sampler holds only the core diffusion weights, its text encoders having run in the encode service and its
    VAE in the image lane, so charging it the whole checkpoint prices support weights no process is holding and
    collapses exactly the co-residency disaggregation exists to buy.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=_OTHER_MODEL)
    sibling.retained_resident_model = _OTHER_MODEL
    sibling.retained_resident_component_only = True
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map)
    scheduler._overhead.set_marginal_overhead_mb(2030.0)
    # Patched to the whole-checkpoint figure the aggregated path would charge, so a component residency reading
    # it at all is a visible failure rather than an arithmetic coincidence.
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    monkeypatch.setattr(
        InferenceScheduler,
        "_read_component_sidecar",
        lambda _self, _model: _Sidecar(_SDXL_UNET_RESIDUAL_BYTES),
    )
    scheduler._vram_budget.check_job = Mock(  # type: ignore[method-assign]
        return_value=Mock(fits=False, predicted_mb=8258.0, reserve_mb=4096.0),
    )
    process_info = scheduler._process_map[_PROCESS_ID]

    # 16375 - 2030 (sibling context) - 5120 (its retained UNet) leaves 9225MB, over the 8258 peak.
    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )


async def test_a_component_only_resident_charge_reads_the_unet_residual(monkeypatch) -> None:  # noqa: ANN001
    """The figure charged is the sidecar's floored UNet residual, not the model's full resident footprint."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=_OTHER_MODEL)
    sibling.retained_resident_model = _OTHER_MODEL
    sibling.retained_resident_component_only = True
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    monkeypatch.setattr(
        InferenceScheduler,
        "_read_component_sidecar",
        lambda _self, _model: _Sidecar(_SDXL_UNET_RESIDUAL_BYTES),
    )

    charges_mb = scheduler._retained_resident_charges_mb(
        dispatched,
        _MODEL,
        process_with_model=scheduler._process_map[_PROCESS_ID],
        device_index=None,
    )

    assert charges_mb == pytest.approx(5120.0)


async def test_an_unpriceable_component_resident_denies_the_grant(monkeypatch) -> None:  # noqa: ANN001
    """A component residency whose sidecar cannot be read denies, rather than falling back to the checkpoint.

    The fallback would be the over-charge the component figure exists to remove, and an unpriceable tenant is
    handled the way an unpriceable sibling context is: deny the grant rather than wave it through at zero.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=_OTHER_MODEL)
    sibling.retained_resident_model = _OTHER_MODEL
    sibling.retained_resident_component_only = True
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map)
    scheduler._overhead.set_marginal_overhead_mb(2030.0)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    monkeypatch.setattr(InferenceScheduler, "_read_component_sidecar", lambda _self, _model: None)
    scheduler._vram_budget.check_job = Mock(  # type: ignore[method-assign]
        return_value=Mock(fits=False, predicted_mb=1000.0, reserve_mb=4096.0),
    )
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_a_whole_job_retained_resident_still_charges_the_full_footprint(monkeypatch) -> None:  # noqa: ANN001
    """A monolithic residency is unaffected: it holds the whole checkpoint and is charged for it.

    The component figure is keyed to what the slot actually holds, so a slot that ran a whole job keeps the
    aggregated charge even on a worker that also disaggregates.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=_OTHER_MODEL)
    sibling.retained_resident_model = _OTHER_MODEL
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_footprint_mb", lambda job, baseline: 6800.0)
    monkeypatch.setattr(
        InferenceScheduler,
        "_read_component_sidecar",
        lambda _self, _model: _Sidecar(_SDXL_UNET_RESIDUAL_BYTES),
    )

    charges_mb = scheduler._retained_resident_charges_mb(
        dispatched,
        _MODEL,
        process_with_model=scheduler._process_map[_PROCESS_ID],
        device_index=None,
    )

    assert charges_mb == pytest.approx(6800.0)


async def test_a_second_concurrent_lease_grant_is_charged_against_the_card(monkeypatch) -> None:  # noqa: ANN001
    """A sibling staged under the lease will materialise beside these weights, so its peak is charged now.

    The failure this pins: with more than one lease slot the parent can clear two children into their
    load-and-sample windows at once. Retention priced against the card minus contexts alone reads as if this
    grant were the only claim in flight, so both fit "alone" and jointly overflow the card, which the driver
    settles by demand-paging or by failing the allocation.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    staged_job = make_job_pop_response(model=_OTHER_MODEL)
    await track_popped_job_async(job_tracker, staged_job)
    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=_OTHER_MODEL)
    sibling.last_process_state = HordeProcessState.INFERENCE_PRIMED
    sibling.last_job_referenced = staged_job
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})

    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
        gpu_sampling_lease_enabled=True,
    )
    scheduler = _make_inference_scheduler(job_tracker=job_tracker, bridge_data=bridge_data, process_map=process_map)
    for process_id in (_PROCESS_ID, _SIBLING_PROCESS_ID):
        for model in (_MODEL, _OTHER_MODEL):
            _seed_repeat_evidence(scheduler, process_id, model)
    scheduler._overhead.set_marginal_overhead_mb(500.0)
    scheduler._vram_budget.check_job = Mock(  # type: ignore[method-assign]
        return_value=Mock(fits=False, predicted_mb=8000.0, reserve_mb=4096.0),
    )
    process_info = scheduler._process_map[_PROCESS_ID]

    # Without the staged sibling's window the card reads as 16376 - 500 = 15876MB against an 8000MB peak, an
    # easy fit. Its 10000MB materialisation less the 2048MB encode charge the ledger already carries leaves
    # 7924MB, which the peak plus the noise buffer does not fit into.
    monkeypatch.setattr(inference_scheduler_module, "predict_job_sampling_vram_mb", lambda job, baseline: 10000.0)

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_an_idle_lane_holding_components_is_charged_as_a_tenant() -> None:
    """A lane's component cache outlives every job boundary, so retention is priced beside it.

    The card is only as big as what nothing is holding. A cache no job boundary returns is as real a tenant as
    a retained checkpoint, and the parent is told what each lane holds on every memory report, so pricing the
    grant as if the lane held nothing packs the card past what it physically has.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    lane = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=None)
    lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
    lane.held_components = [
        HeldComponentSnapshot(kind="unet", identity="staged-checkpoint", approx_ram_mb=7600.0),
    ]
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: lane})
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map)
    scheduler._overhead.set_marginal_overhead_mb(500.0)
    scheduler._vram_budget.check_job = Mock(  # type: ignore[method-assign]
        return_value=Mock(fits=False, predicted_mb=8000.0, reserve_mb=4096.0),
    )
    process_info = scheduler._process_map[_PROCESS_ID]

    # 16376 less the 500MB sibling context and the 7600MB the lane is holding leaves 8276MB, under the
    # 8000MB peak plus its admission noise buffer. Charged as if the lane held nothing, the same grant reads
    # as a comfortable fit.
    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )
