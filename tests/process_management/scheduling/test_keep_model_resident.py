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

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeInferenceControlMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
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
) -> InferenceScheduler:
    """A scheduler with the VRAM budget active and the governor unsampled (defaults to HEALTHY)."""
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
    )
    return _make_inference_scheduler(
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        process_map=process_map if process_map is not None else ProcessMap({_PROCESS_ID: _dispatch_process()}),
        horde_model_map=horde_model_map,
    )


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


def test_inference_control_message_defaults_to_eviction() -> None:
    """The dispatch message preserves today's aggressive eviction unless the scheduler opts in."""
    message = HordeInferenceControlMessage(
        control_flag=HordeControlFlag.START_INFERENCE,
        horde_model_name=_MODEL,
        sdk_api_job_info=make_job_pop_response(model=_MODEL),
    )

    assert message.keep_model_resident_after is False
