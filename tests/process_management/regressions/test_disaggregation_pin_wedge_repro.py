"""Regression: a monolithic head whose model is resident only on a disaggregation-pinned lane must not wedge.

Incident shape: on a card whose disaggregation-pinned sampler lanes hold resident SDXL models whose sampling
peaks cannot co-fit, a monolithic head for one of those resident models read as not-resident, because the
dispatch query excludes pinned lanes so no job is dispatched onto one. The head then churned toward a fresh
full-weight preload that could never fit beside the pinned residents, and the queue deadlocked.

The head must instead be recognised as resident on the pinned lane (a residency/pricing query that includes
pinned lanes), priced as already resident rather than a fresh copy, and held for the pin to release, then
dispatched onto that resident lane. A job is never dispatched onto a lane while it is pinned. The dispatch
stall classifier must name the pin and the disaggregated job holding it rather than a generic budget defer.
"""

from __future__ import annotations

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.process_management.ipc.messages import HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyBucket
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MODEL = "SDXL 1.0"


def _reference() -> dict[str, object]:
    return {
        _MODEL: make_mock_model_reference_record(_MODEL, baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl)
    }


async def _scheduler_with_pinned_resident_head(*, with_free_slot: bool):  # noqa: ANN202
    """Head for ``_MODEL``, whose only resident copy is on process 0, pinned as a disaggregation sampler.

    Process 0 is PRELOADED_MODEL (so it would read as can-accept-job absent the reservation, the trap the
    dispatch exclusion guards). ``with_free_slot`` adds an empty idle process 1 the old path would have
    preloaded a second copy onto.
    """
    reference = _reference()
    metadata = make_test_model_metadata(reference)  # type: ignore[arg-type]

    pinned = make_mock_process_info(0, model_name=_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    process_map = ProcessMap({0: pinned})
    if with_free_slot:
        process_map[1] = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    process_map.reserve_for_disaggregation(0)

    horde_model_map = HordeModelMap(root={})
    horde_model_map.update_entry(horde_model_name=_MODEL, load_state=ModelLoadState.LOADED_IN_VRAM, process_id=0)

    job_tracker = JobTracker()
    head = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, head)

    scheduler = _make_inference_scheduler(
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(max_threads=2),
        max_concurrent=2,
        max_inference=2,
        model_metadata=metadata,
    )
    return scheduler, head, process_map


async def test_pinned_lane_is_seen_only_by_residency_queries_not_dispatch() -> None:
    """The pinned lane is excluded from dispatch/can-accept queries but visible to residency/pricing queries."""
    scheduler, head, process_map = await _scheduler_with_pinned_resident_head(with_free_slot=False)

    # Dispatch selection excludes the pinned lane (never dispatch onto it): the model reads as not resident.
    assert process_map.get_process_by_horde_model_name(_MODEL) is None
    assert scheduler._resident_process_for_job(head) is None

    # Residency/pricing queries include it, so the scheduler can know the model's weights are resident there.
    assert process_map.get_process_by_horde_model_name(_MODEL, include_reserved=True) is process_map[0]
    assert scheduler._resident_process_for_job(head, include_reserved=True) is process_map[0]
    assert scheduler._pinned_lane_resident_for_job(head) is process_map[0]


async def test_head_resident_only_on_pinned_lane_holds_and_never_preloads_a_second_copy() -> None:
    """The head holds (no dispatch) for the pin rather than dispatching onto it or funding a fresh copy."""
    scheduler, _head, process_map = await _scheduler_with_pinned_resident_head(with_free_slot=True)

    result = await scheduler.get_next_job_and_process()

    assert result is None  # held for the pin to release, not dispatched
    # Neither the pinned lane nor the free slot was dispatched onto.
    process_map[0].pipe_connection.send.assert_not_called()
    process_map[1].pipe_connection.send.assert_not_called()
    # The free slot stays idle: no fresh second copy was staged onto it.
    assert process_map[1].last_process_state == HordeProcessState.WAITING_FOR_JOB
    # The head was held, not routed through the "model expected resident but missing" recovery: that recovery
    # expires the model-map entry (the churn the wedge fed on), so a preserved entry proves the head held.
    assert _MODEL in scheduler._horde_model_map.root
    assert scheduler._horde_model_map.root[_MODEL].horde_model_load_state == ModelLoadState.LOADED_IN_VRAM


async def test_pin_aware_pricing_counts_the_pinned_resident_model_as_loaded_not_a_fresh_copy() -> None:
    """The preload pass prices the pinned-lane model as already resident, so it never stages a second copy."""
    scheduler, _head, process_map = await _scheduler_with_pinned_resident_head(with_free_slot=True)

    preloaded = scheduler.preload_models()

    assert preloaded is False  # nothing preloaded: the model is priced resident on the pinned lane
    process_map[1].pipe_connection.send.assert_not_called()  # no fresh copy dispatched to the free slot


async def test_head_dispatches_onto_the_resident_lane_once_the_pin_releases() -> None:
    """When the pin releases, the head dispatches onto the now-free resident lane (the reused resident copy)."""
    scheduler, head, process_map = await _scheduler_with_pinned_resident_head(with_free_slot=True)

    assert await scheduler.get_next_job_and_process() is None  # held while pinned

    # Sampling finished for the disaggregated job: its sampler pin releases, returning the resident lane.
    process_map.release_disaggregation_reservation(0)

    result = await scheduler.get_next_job_and_process()
    assert result is not None
    assert result.next_job is head
    assert result.process_with_model is process_map[0]  # dispatched onto the resident lane, not a fresh slot


async def test_classifier_names_the_pinned_residency_wait_with_owner_and_peaks() -> None:
    """The dispatch-stall classifier attributes the held head to the pin, its owning job, and the in-flight peaks."""
    scheduler, head, _process_map = await _scheduler_with_pinned_resident_head(with_free_slot=False)
    scheduler.set_disaggregation_hooks(
        is_disaggregatable=lambda _job: False,
        is_disaggregation_class_eligible=lambda _job: False,
        register_disaggregated=None,  # type: ignore[arg-type]
        pin_owner=lambda pid: "abcd1234-owner" if pid == 0 else None,
        sampling_peaks=lambda: {"abcd1234-owner": 8000.0},
    )

    bucket, text = scheduler._classify_dispatch_stall(head, _reference())  # type: ignore[arg-type]

    assert bucket is SlotDutyBucket.DISAGG_PIN_WAIT
    assert "process 0" in text
    assert "abcd1234"[:8] in text  # names the disaggregated job holding the pin
    assert "8000" in text  # surfaces the in-flight sampling peak keeping the card busy


async def test_classifier_names_degraded_isolation_pending() -> None:
    """A resident-idle head that must run isolated is named DEGRADED_ISOLATION_PENDING, not UNEXPLAINED."""
    reference = _reference()
    metadata = make_test_model_metadata(reference)  # type: ignore[arg-type]
    idle = make_mock_process_info(0, model_name=_MODEL, state=HordeProcessState.WAITING_FOR_JOB)
    process_map = ProcessMap({0: idle})
    horde_model_map = HordeModelMap(root={})
    horde_model_map.update_entry(horde_model_name=_MODEL, load_state=ModelLoadState.LOADED_IN_VRAM, process_id=0)

    job_tracker = JobTracker()
    head = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, head)

    scheduler = _make_inference_scheduler(
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(max_threads=2),
        max_concurrent=2,
        max_inference=2,
        model_metadata=metadata,
    )
    tracked = job_tracker.get_tracked_job(head.id_)
    assert tracked is not None
    tracked.needs_degraded_dispatch = True  # the next dispatch must run isolated

    bucket, text = scheduler._classify_dispatch_stall(head, reference)  # type: ignore[arg-type]

    assert bucket is SlotDutyBucket.DEGRADED_ISOLATION_PENDING
    assert "degraded" in text
