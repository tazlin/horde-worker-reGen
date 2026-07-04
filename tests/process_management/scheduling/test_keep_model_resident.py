"""Tests for budget-gated VRAM retention across jobs.

hordelib evicts a job's model from VRAM after every run so sibling GPU instances never collectively
over-commit. That eviction forces a RAM->VRAM reload on the next job, the dominant non-sampling cost on
small jobs. :meth:`InferenceScheduler._should_keep_model_resident` decides when to suppress that eviction
for one dispatch, under gates that cannot over-commit the card even when the driver's free-VRAM figure
lies (a WDDM driver in demand-paging keeps reporting generous free VRAM while the card is saturated):

- **Sole residency**: no other process on the card may hold a VRAM-resident model, judged from the
  parent's model-map bookkeeping rather than device telemetry. Retention only extends the card's single
  resident across consecutive jobs; it never creates a second resident.
- **Static fit**: the card's reported total VRAM (a constant the driver cannot misreport) must absorb the
  job's sampling peak plus the reserve.
- **Measured veto**: the measured-free check can still deny, never solely grant. The job's own weights
  are credited back only when the dispatching process already holds this model in VRAM, since only then
  was the reading taken with the weights occupying the card.

No queue lookahead gates the grant: the pop cycle refills the queue immediately after a dispatch drains
it, so a same-model successor is almost never visible at the dispatch instant even when one arrives
milliseconds later. Reclaim is instead just-in-time via the per-dispatch VRAM sweep and the
under-pressure eviction.
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
_AMPLE_FREE_VRAM_MB = 12000.0
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
    free_vram_mb: float | None = _AMPLE_FREE_VRAM_MB,
    process_map: ProcessMap | None = None,
    horde_model_map: HordeModelMap | None = None,
) -> InferenceScheduler:
    """A scheduler with the VRAM budget active and a controllable measured-free-VRAM reading."""
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
    )
    scheduler = _make_inference_scheduler(
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        process_map=process_map if process_map is not None else ProcessMap({_PROCESS_ID: _dispatch_process()}),
        horde_model_map=horde_model_map,
    )
    scheduler._measured_free_vram_mb = Mock(return_value=free_vram_mb)  # type: ignore[method-assign]
    return scheduler


def _map_with_model_on_process(
    model: str,
    process_id: int,
    load_state: ModelLoadState = ModelLoadState.LOADED_IN_VRAM,
) -> HordeModelMap:
    model_map = HordeModelMap(root={})
    model_map.update_entry(model, load_state=load_state, process_id=process_id)
    return model_map


async def test_retains_when_budget_fits_without_queue_lookahead() -> None:
    """An empty pending queue does not block retention: the successor is popped after dispatch."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
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
    scheduler._measured_free_vram_mb = Mock(return_value=_AMPLE_FREE_VRAM_MB)  # type: ignore[method-assign]
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_no_retain_when_free_vram_unmeasured() -> None:
    """A cold start with no VRAM telemetry must not assume headroom; retention is evidence-gated."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker, free_vram_mb=None)
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


async def test_no_retain_when_another_process_holds_a_resident_model() -> None:
    """A VRAM-resident model on any sibling denies retention outright, regardless of measured free.

    This is the WDDM guard: a driver in demand-paging reports generous free VRAM while the card is
    saturated, so co-residency questions are answered by the parent's own bookkeeping. Retention must
    never create a second resident model on the card.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=_OTHER_MODEL)
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    model_map = _map_with_model_on_process(_OTHER_MODEL, _SIBLING_PROCESS_ID)
    scheduler = _budget_on_scheduler(job_tracker, process_map=process_map, horde_model_map=model_map)
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_stale_resident_entry_for_dead_process_does_not_block() -> None:
    """A model-map entry whose process no longer exists is staleness, not residency."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    model_map = _map_with_model_on_process(_OTHER_MODEL, 99)  # process 99 is not in the map
    scheduler = _budget_on_scheduler(job_tracker, horde_model_map=model_map)
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )


async def test_credits_weights_when_model_resident_on_this_process(monkeypatch) -> None:  # noqa: ANN001
    """A model already in VRAM on the dispatching process gets its weights credited back into free.

    The free reading was taken with the weights occupying the card; asking the footprint to also fit
    inside the remainder would charge the weights twice and deny retention on any busy card. The credit
    is clamped to the card total so a stale reading can never grant against more VRAM than exists.
    """
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    model_map = _map_with_model_on_process(_MODEL, _PROCESS_ID)
    scheduler = _budget_on_scheduler(job_tracker, free_vram_mb=3000.0, horde_model_map=model_map)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_weight_mb", lambda job, baseline: 5000.0)
    seen_free: list[float] = []

    def record_check(job, baseline, free_vram_mb, committed_reserve_mb=0.0):  # noqa: ANN001, ANN202
        seen_free.append(free_vram_mb)
        return Mock(fits=True, predicted_mb=None, reserve_mb=0.0)

    scheduler._vram_budget.check_job = record_check  # type: ignore[method-assign]
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )
    # First check is the static (total-based) gate; the second is the measured gate with the credit.
    assert seen_free == [float(_TOTAL_VRAM_MB), 8000.0]


async def test_no_credit_when_model_resident_on_other_process(monkeypatch) -> None:  # noqa: ANN001
    """A copy resident on a different process denies retention before any credit question arises."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    sibling = make_mock_process_info(_SIBLING_PROCESS_ID, model_name=_MODEL)
    process_map = ProcessMap({_PROCESS_ID: _dispatch_process(), _SIBLING_PROCESS_ID: sibling})
    model_map = _map_with_model_on_process(_MODEL, _SIBLING_PROCESS_ID)
    scheduler = _budget_on_scheduler(
        job_tracker, free_vram_mb=3000.0, process_map=process_map, horde_model_map=model_map
    )
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )


async def test_no_credit_when_model_only_in_ram(monkeypatch) -> None:  # noqa: ANN001
    """A RAM-only (preloaded) model occupies no VRAM, so its weights must not be credited."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    model_map = _map_with_model_on_process(_MODEL, _PROCESS_ID, load_state=ModelLoadState.LOADED_IN_RAM)
    scheduler = _budget_on_scheduler(job_tracker, free_vram_mb=3000.0, horde_model_map=model_map)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_weight_mb", lambda job, baseline: 5000.0)
    seen_free: list[float] = []

    def record_check(job, baseline, free_vram_mb, committed_reserve_mb=0.0):  # noqa: ANN001, ANN202
        seen_free.append(free_vram_mb)
        return Mock(fits=len(seen_free) == 1, predicted_mb=None, reserve_mb=0.0)

    scheduler._vram_budget.check_job = record_check  # type: ignore[method-assign]
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is False
    )
    assert seen_free == [float(_TOTAL_VRAM_MB), 3000.0]


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

    def record_check(job, baseline, free_vram_mb, committed_reserve_mb=0.0):  # noqa: ANN001, ANN202
        seen_free.append(free_vram_mb)
        return Mock(fits=True, predicted_mb=None, reserve_mb=0.0)

    scheduler._vram_budget.check_job = record_check  # type: ignore[method-assign]
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )
    assert seen_free[0] == float(_TOTAL_VRAM_MB) - 1354.0


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

    def record_check(job, baseline, free_vram_mb, committed_reserve_mb=0.0):  # noqa: ANN001, ANN202
        seen_free.append(free_vram_mb)
        return Mock(fits=True, predicted_mb=None, reserve_mb=0.0)

    scheduler._vram_budget.check_job = record_check  # type: ignore[method-assign]
    process_info = scheduler._process_map[_PROCESS_ID]

    assert (
        scheduler._should_keep_model_resident(dispatched, process_with_model=process_info, device_index=None) is True
    )
    assert seen_free[0] == float(_TOTAL_VRAM_MB) - 3100.0


async def test_no_retain_when_budget_rejects_footprint() -> None:
    """Under VRAM pressure the budget says the model does not fit, so retention is refused (would starve a swap)."""
    job_tracker = JobTracker()
    dispatched = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, dispatched)

    scheduler = _budget_on_scheduler(job_tracker)
    scheduler._vram_budget.check_job = Mock(  # type: ignore[method-assign]
        return_value=Mock(fits=False, predicted_mb=None, reserve_mb=0.0),
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
    """The paging verdict's rising edge runs one idle-VRAM sweep, excluding the paging process itself."""
    job_tracker = JobTracker()
    scheduler = _budget_on_scheduler(job_tracker)
    paging_process = scheduler._process_map[_PROCESS_ID]
    sweeps: list[int] = []

    def record_sweep(process_with_model, **kwargs):  # noqa: ANN001, ANN003, ANN202
        sweeps.append(process_with_model.process_id)
        return True

    scheduler.unload_models_from_vram = record_sweep  # type: ignore[method-assign]

    paging_pid = paging_process.os_pid
    assert paging_pid is not None
    scheduler.note_wddm_paging({paging_pid: 900.0}, active=True)
    scheduler.note_wddm_paging({paging_pid: 900.0}, active=True)
    scheduler.note_wddm_paging({}, active=False)
    scheduler.note_wddm_paging({paging_pid: 900.0}, active=True)

    # One sweep per rising edge (not per tick), each excluding the paging process.
    assert sweeps == [_PROCESS_ID, _PROCESS_ID]


def test_inference_control_message_defaults_to_eviction() -> None:
    """The dispatch message preserves today's aggressive eviction unless the scheduler opts in."""
    message = HordeInferenceControlMessage(
        control_flag=HordeControlFlag.START_INFERENCE,
        horde_model_name=_MODEL,
        sdk_api_job_info=make_job_pop_response(model=_MODEL),
    )

    assert message.keep_model_resident_after is False
