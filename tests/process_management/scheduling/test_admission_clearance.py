"""The clearance admit decision over the scheduling snapshot."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from unittest.mock import Mock

import pytest
from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeImageResult, HordeProcessState
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.process_management.resources.resource_budget import (
    predict_job_weight_mb,
    predict_model_weight_mb,
)
from horde_worker_regen.process_management.resources.vram_arbiter import (
    _FIRST_PARTY_TEARDOWN_GRACE_SECONDS,
    ActuatorCommandKind,
)
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.admission.clearance import (
    ClearanceDecision,
    StagedWaiterClock,
    decide_clearance_admit,
    staged_attempt_deadline_seconds,
    staged_held_mb,
    staged_materialization_delta_mb,
)
from horde_worker_regen.process_management.scheduling.admission.materialization import head_starved_seconds
from horde_worker_regen.process_management.scheduling.clearance_lease import (
    CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS,
    ClearanceController,
    ClearanceLeaseProxy,
    GrantState,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.workload_flow import DISPATCH_ADMISSION_FLOW
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_JOB_MODEL = "stable_diffusion"
_JOB_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
_OTHER_MODEL = "other_checkpoint"
_OTHER_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1


def _two_model_metadata() -> ModelMetadata:
    """Model metadata that knows the job's model and one other, so both have predicted weights."""
    return make_test_model_metadata(
        {
            _JOB_MODEL: make_mock_model_reference_record(_JOB_MODEL, baseline=_JOB_BASELINE),
            _OTHER_MODEL: make_mock_model_reference_record(_OTHER_MODEL, baseline=_OTHER_BASELINE),
        },
    )


async def _staged_waiter(
    *, device_free_mb: float, budget: bool = True, model_metadata: ModelMetadata | None = None
) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse, HordeProcessInfo]:
    """One primed child holding a dispatched job and its encode-only staging reservation."""
    tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        bridge_data=make_mock_bridge_data(
            gpu_sampling_lease_enabled=True,
            enable_vram_budget=budget,
            vram_reserve_mb=2048,
            ram_reserve_mb=4096,
        ),
        job_tracker=tracker,
        device_free_mb=device_free_mb,
        model_metadata=model_metadata,
    )
    waiter = make_mock_process_info(0, model_name=_JOB_MODEL, state=HordeProcessState.INFERENCE_PRIMED)
    waiter.total_vram_mb = 16000
    waiter.process_reserved_mb = 900
    scheduler._process_map = ProcessMap({0: waiter})
    job = make_job_pop_response(_JOB_MODEL)
    await track_popped_job_async(tracker, job)
    await mark_job_in_progress_async(tracker, job)
    waiter.last_job_referenced = job
    scheduler._record_dispatch_reservation(job, waiter, baseline=None, staging_only=True)  # type: ignore[attr-defined]
    return scheduler, job, waiter


def _decide(scheduler: InferenceScheduler, process_id: int, *, deferred: bool = False):  # noqa: ANN202
    arbiter = scheduler._ensure_preload_arbiter()  # type: ignore[attr-defined]
    return decide_clearance_admit(scheduler.snapshot(), process_id, arbiter=arbiter, post_processing_deferred=deferred)


async def test_an_unknown_process_is_withheld() -> None:
    """A process id naming no slot is withheld: there is nothing to clear."""
    scheduler, _job, _waiter = await _staged_waiter(device_free_mb=24000.0)
    plan = _decide(scheduler, 7)
    assert plan.decision is ClearanceDecision.NO_PROCESS
    assert plan.grants is False
    assert plan.actuations == ()


async def test_a_child_with_nothing_priceable_is_granted() -> None:
    """A primed child referencing no job is granted rather than wedged on nothing priceable."""
    scheduler, _job, waiter = await _staged_waiter(device_free_mb=24000.0)
    waiter.last_job_referenced = None
    plan = _decide(scheduler, 0)
    assert plan.decision is ClearanceDecision.UNPRICED
    assert plan.grants is True
    assert plan.priced is None


async def test_the_post_processing_mutex_holds_before_pricing() -> None:
    """The co-residency mutex holds the child without pricing, keyed on its job."""
    scheduler, job, _waiter = await _staged_waiter(device_free_mb=24000.0)
    plan = _decide(scheduler, 0, deferred=True)
    assert plan.decision is ClearanceDecision.HOLD_POST_PROCESSING
    assert plan.grants is False
    assert plan.reason == "post_processing_coresidency"
    assert plan.job_id == str(job.id_)
    assert plan.priced is None


async def test_an_inactive_budget_grants_without_pricing() -> None:
    """With the VRAM budget off the child is granted, no verdict taken."""
    scheduler, _job, _waiter = await _staged_waiter(device_free_mb=100.0, budget=False)
    plan = _decide(scheduler, 0)
    assert plan.decision is ClearanceDecision.BUDGET_INACTIVE
    assert plan.grants is True
    assert plan.verdict is None


async def test_an_ample_card_admits_the_staged_peak_net_of_its_own_reservation() -> None:
    """A fitting card admits: the head's request at the staged delta, netting the job's own staging charge."""
    scheduler, job, _waiter = await _staged_waiter(device_free_mb=24000.0)
    plan = _decide(scheduler, 0)
    assert plan.decision is ClearanceDecision.ADMIT
    assert plan.grants is True
    assert plan.priced is not None and plan.verdict is not None
    assert plan.verdict.admits
    assert plan.reason == plan.verdict.disposition.value
    snapshot = scheduler.snapshot()
    job_id = str(job.id_)
    assert plan.candidate_delta_mb == staged_materialization_delta_mb(snapshot, job_id, 0)
    request = plan.priced.request
    assert request.is_head_of_queue is True
    assert request.head_outstanding_mb is None
    own_staging_mb = snapshot.services.reserve_ledger.planned_charge_for_unit(
        DISPATCH_ADMISSION_FLOW, job_id, dict(snapshot.card(None).reserved_by_pid)
    )
    assert own_staging_mb > 0.0
    assert request.own_dispatch_unmaterialized_mb == own_staging_mb


async def test_a_short_card_holds_and_carries_the_verdict_evictions() -> None:
    """A card short of the peak holds, naming the disposition and the evictions the verdict describes."""
    scheduler, _job, _waiter = await _staged_waiter(device_free_mb=100.0)
    plan = _decide(scheduler, 0)
    assert plan.decision is ClearanceDecision.HOLD
    assert plan.grants is False
    assert plan.verdict is not None and not plan.verdict.admits
    assert plan.reason == plan.verdict.disposition.value
    assert plan.actuations == plan.verdict.required_actuations


async def test_the_staged_delta_nets_what_the_child_already_holds() -> None:
    """The staged delta is the gross peak net of the child reservation; an unreported slot is charged gross."""
    scheduler, job, waiter = await _staged_waiter(device_free_mb=24000.0)
    job_id = str(job.id_)
    snapshot = scheduler.snapshot()
    gross = pricing.candidate_delta_mb(
        snapshot, job, snapshot.queue.jobs[job_id].baseline, process_id=0, disaggregated=False
    )
    assert gross is not None
    assert staged_materialization_delta_mb(snapshot, job_id, 0) == max(0.0, gross - 900.0)

    waiter.process_reserved_mb = None
    assert staged_materialization_delta_mb(scheduler.snapshot(), job_id, 0) == gross


async def test_own_weights_on_the_slot_are_netted_whole_or_partial() -> None:
    """Weights of the job's own model the slot reports are netted from the uncredited peak, whole or partial.

    Credited whole instead, a partial copy prices as fully loaded, and a whole copy leaves the staged encode set
    charged on top of the activation.
    """
    scheduler, job, waiter = await _staged_waiter(device_free_mb=24000.0, model_metadata=_two_model_metadata())
    job_id = str(job.id_)
    # A primed lane's job model is resident only under a retention grant (the weights are otherwise in RAM).
    waiter.retained_resident_model = _JOB_MODEL
    snapshot = scheduler.snapshot()
    assert pricing.candidate_weights_resident(snapshot, _JOB_MODEL, 0)
    baseline = snapshot.queue.jobs[job_id].baseline
    weights_mb = predict_job_weight_mb(job, baseline)
    credited_mb = pricing.candidate_delta_mb(snapshot, job, baseline, process_id=0, disaggregated=False)
    assert weights_mb is not None and weights_mb > 0.0 and credited_mb is not None
    peak_mb = credited_mb + weights_mb

    waiter.process_reserved_mb = int(weights_mb) + 900
    whole = staged_materialization_delta_mb(scheduler.snapshot(), job_id, 0)
    assert whole == max(0.0, peak_mb - (int(weights_mb) + 900))
    assert whole is not None and whole < credited_mb

    partial_copy_mb = int(weights_mb) // 4
    waiter.process_reserved_mb = partial_copy_mb + 900
    partial = staged_materialization_delta_mb(scheduler.snapshot(), job_id, 0)
    assert partial == max(0.0, peak_mb - (partial_copy_mb + 900))
    assert partial is not None and partial > credited_mb


async def test_another_models_weights_on_the_slot_are_not_netted() -> None:
    """Weights another model keeps on the slot stay beside the job, so only the rest of the report is netted."""
    scheduler, job, waiter = await _staged_waiter(device_free_mb=24000.0, model_metadata=_two_model_metadata())
    job_id = str(job.id_)
    waiter.retained_resident_model = _OTHER_MODEL
    other_weights_mb = predict_model_weight_mb(_OTHER_MODEL, str(_OTHER_BASELINE))
    assert other_weights_mb is not None and other_weights_mb > 0.0
    waiter.process_reserved_mb = int(other_weights_mb) + 900
    snapshot = scheduler.snapshot()
    assert _OTHER_MODEL in snapshot.slots[0].resident_weight_models
    assert not pricing.candidate_weights_resident(snapshot, _JOB_MODEL, 0)
    gross = pricing.candidate_delta_mb(
        snapshot, job, snapshot.queue.jobs[job_id].baseline, process_id=0, disaggregated=False
    )
    assert gross is not None

    own_holdings_mb = waiter.process_reserved_mb - other_weights_mb
    assert staged_materialization_delta_mb(snapshot, job_id, 0) == max(0.0, gross - own_holdings_mb)


async def test_a_cleared_job_books_what_clearance_priced_and_spends_it_at_its_peak() -> None:
    """The cleared job's reservation is the staged delta against the same report, so it is spent at the peak.

    Booked at the gross peak against a baseline that already holds the staged set, that set stays outstanding
    through the whole sample and is charged against the next waiter.
    """
    scheduler, job, _waiter = await _staged_waiter(device_free_mb=24000.0)
    job_id = str(job.id_)
    snapshot = scheduler.snapshot()
    gross = pricing.candidate_delta_mb(
        snapshot, job, snapshot.queue.jobs[job_id].baseline, process_id=0, disaggregated=False
    )
    remaining = staged_materialization_delta_mb(snapshot, job_id, 0)
    assert gross is not None and remaining is not None and remaining == gross - 900.0

    assert scheduler.clearance_admit_process(0) is True

    ledger = scheduler._reserve_ledger  # type: ignore[attr-defined]
    assert ledger.planned_charge_for_unit(DISPATCH_ADMISSION_FLOW, job_id, {0: 900.0}) == remaining
    assert ledger.planned_charge_for_unit(DISPATCH_ADMISSION_FLOW, job_id, {0: gross}) == 0.0


async def test_clearance_presents_what_the_staged_job_already_holds() -> None:
    """The request carries the netted holdings, so the outstanding charge plus them is the job's whole peak.

    Without it the ceiling test reads the netted charge alone, and a job whose whole need exceeds the emptied
    card reads as possible on it.
    """
    scheduler, job, _waiter = await _staged_waiter(device_free_mb=100.0)
    job_id = str(job.id_)
    snapshot = scheduler.snapshot()
    gross = pricing.candidate_delta_mb(
        snapshot, job, snapshot.queue.jobs[job_id].baseline, process_id=0, disaggregated=False
    )
    plan = _decide(scheduler, 0)
    assert plan.priced is not None and gross is not None
    request = plan.priced.request
    assert request.candidate_held_mb == staged_held_mb(snapshot, job_id, 0) == 900.0
    assert request.candidate_delta_mb is not None
    assert request.candidate_delta_mb + request.candidate_held_mb == gross


async def test_the_waiter_clock_times_the_probe_and_its_deadline_precedes_the_lease_timeout() -> None:
    """Clearance prices a staged waiter on its own clock, with a probe deadline inside the acquire timeout.

    The deadline is the timeout less the longer of the card's probe delay and its measured load seconds.
    """
    scheduler, job, _waiter = await _staged_waiter(device_free_mb=100.0)
    probe_seconds = float(scheduler.snapshot().config_for(None).measured_load_probe_seconds)
    arbiter = scheduler._ensure_preload_arbiter()  # type: ignore[attr-defined]

    def priced_with(clock: StagedWaiterClock):  # noqa: ANN202
        plan = decide_clearance_admit(
            scheduler.snapshot(), 0, arbiter=arbiter, post_processing_deferred=False, waiter_clock=clock
        )
        assert plan.priced is not None
        return plan.priced.request

    quick_load = priced_with(StagedWaiterClock(starved_seconds=12.0, load_seconds=probe_seconds / 2))
    assert quick_load.starved_seconds == 12.0
    assert quick_load.attempt_deadline_seconds == CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS - probe_seconds
    slow_load = priced_with(StagedWaiterClock(starved_seconds=12.0, load_seconds=probe_seconds + 5.0))
    assert slow_load.attempt_deadline_seconds == CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS - probe_seconds - 5.0
    assert staged_attempt_deadline_seconds(probe_after_seconds=0.0, load_seconds=None) == (
        CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
    )
    assert _decide(scheduler, 0).priced.request.starved_seconds == 0.0  # type: ignore[union-attr]


async def test_a_held_lone_waiter_is_timed_until_another_lanes_work_can_help() -> None:
    """Clearance times a lone held waiter and stops the clock while another lane on the card is sampling.

    The stop happens every tick the inputs are built, since a waiter behind another grant is not offered to
    the admission at all.
    """
    scheduler, job, waiter = await _staged_waiter(device_free_mb=100.0)
    now = [1000.0]
    scheduler._clock = lambda: now[0]  # type: ignore[method-assign]

    assert scheduler.clearance_admit_process(0) is False
    now[0] += 7.0
    assert scheduler._staged_waiter_clock(waiter, job).starved_seconds == 7.0  # type: ignore[attr-defined]

    sibling = make_mock_process_info(1, model_name=_OTHER_MODEL, state=HordeProcessState.INFERENCE_STARTING)
    sibling.last_job_referenced = make_job_pop_response(_OTHER_MODEL)
    scheduler._process_map = ProcessMap({0: waiter, 1: sibling})
    scheduler.build_clearance_inputs(device_index=0)
    assert scheduler._staged_waiter_clock(waiter, job).starved_seconds == 0.0  # type: ignore[attr-defined]

    assert scheduler.clearance_admit_process(0) is False
    now[0] += 7.0
    assert scheduler._staged_waiter_clock(waiter, job).starved_seconds == 0.0  # type: ignore[attr-defined]


async def test_a_starved_waiter_beside_a_fresh_preload_is_offered_the_post_processing_lane() -> None:
    """A sibling still busy on a fresh preload is no eviction rung, so a starved waiter reaches the lane rung.

    The eviction actuator skips a busy slot until its preload is parked past the retention horizon. Counted as
    reclaimable regardless, the slot kept the ladder non-empty with a rung that never acted, and the lane rung
    behind it was unreachable for that whole horizon.
    """
    scheduler, _job, waiter = await _staged_waiter(device_free_mb=24000.0, model_metadata=_two_model_metadata())
    now = [1000.0]
    scheduler._clock = lambda: now[0]  # type: ignore[method-assign]
    preloading_sibling = make_mock_process_info(1, model_name=_OTHER_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    preloading_sibling.total_vram_mb = 16000
    preloading_sibling.process_reserved_mb = 3200
    preloading_sibling.process_allocated_mb = 3200
    post_process_lane = make_mock_process_info(
        2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, process_type=HordeProcessType.POST_PROCESS
    )
    post_process_lane.process_reserved_mb = 1500
    post_process_lane.process_allocated_mb = 1500
    # Its models were already unloaded, so the lane's context is all it still holds.
    post_process_lane.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
    scheduler._process_map = ProcessMap({0: waiter, 1: preloading_sibling, 2: post_process_lane})
    scheduler._process_lifecycle.is_post_process_gpu_paused = False  # type: ignore[attr-defined]
    preloading_sibling.last_process_state_started_at = now[0]
    snapshot = scheduler.snapshot()
    assert snapshot.slots[1].is_busy and not snapshot.slots[1].parked_preload

    arbiter = scheduler._ensure_preload_arbiter()  # type: ignore[attr-defined]
    starved = StagedWaiterClock(starved_seconds=_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 1.0, load_seconds=None)
    ample = decide_clearance_admit(snapshot, 0, arbiter=arbiter, post_processing_deferred=False, waiter_clock=starved)
    assert ample.priced is not None and ample.verdict is not None
    candidate_mb = ample.priced.request.candidate_delta_mb
    noise_mb = ample.verdict.measured.noise_buffer_mb
    reservations_mb = ample.verdict.measured.outstanding_reservations_mb
    assert candidate_mb is not None
    # Short by less than the post-processing lane's reservation alone, so pausing it closes the deficit.
    scheduler.set_device_free_mb_provider(lambda _device_index: candidate_mb + noise_mb + reservations_mb - 500.0)

    arbiter = scheduler._ensure_preload_arbiter()  # type: ignore[attr-defined]
    plan = decide_clearance_admit(
        scheduler.snapshot(), 0, arbiter=arbiter, post_processing_deferred=False, waiter_clock=starved
    )

    assert plan.decision is ClearanceDecision.HOLD
    assert [command.kind for command in plan.actuations] == [ActuatorCommandKind.PAUSE_POST_PROCESS_LANE]


async def test_a_dispatched_job_is_never_timed_by_the_head_clock() -> None:
    """The head-starvation clock answers nothing for a job in progress, and is released once it is dispatched.

    The preload pass that retimes the head returns early once every pending model is accounted for, so a job
    whose model needed no preload kept the clock it started at pop through dispatch and staging.
    """
    scheduler, job, _waiter = await _staged_waiter(device_free_mb=100.0)
    job_id = str(job.id_)
    head_admission = scheduler.head_admission
    head_admission.track_head_starvation(job_id, work_in_progress=False)
    snapshot = scheduler.snapshot()
    later = replace(snapshot, now=head_admission.starvation_since + 30.0)
    assert head_starved_seconds(later, job_id) == 0.0
    still_pending = replace(
        later,
        queue=replace(later.queue, jobs={job_id: replace(later.queue.jobs[job_id], in_progress=False)}),
    )
    assert head_starved_seconds(still_pending, job_id) == 30.0

    scheduler.reconcile_head_starvation()
    assert head_admission.starvation_job_id is None


async def test_a_job_awaits_admission_until_its_sample_finishes() -> None:
    """A lane pause's beneficiary reads as waiting until its inference result lands, sampling included.

    The restore evidence must not read the card: the context the pause freed is what makes the card healthy.
    The sample keeps growing after its first step, so a lane restored at that step lands on the same card.
    A warm slot still reports the previous job's ``INFERENCE_COMPLETE`` when the head is dispatched to it.
    """
    job_tracker = JobTracker()
    job = make_job_pop_response("stable_diffusion")
    await track_popped_job_async(job_tracker, job)
    lane = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.INFERENCE_COMPLETE)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({0: lane}), job_tracker=job_tracker)
    job_id = str(job.id_)

    assert scheduler.job_awaits_admission(job_id) is True

    await mark_job_in_progress_async(job_tracker, job)
    lane.record_inference_ownership(job, attempt_ordinal=1)
    for slot_state in (
        HordeProcessState.INFERENCE_COMPLETE,
        HordeProcessState.PRELOADED_MODEL,
        HordeProcessState.INFERENCE_PRIMED,
        HordeProcessState.INFERENCE_STARTING,
    ):
        lane.last_process_state = slot_state
        assert scheduler.job_awaits_admission(job_id) is True, slot_state

    lane.last_process_state = HordeProcessState.INFERENCE_COMPLETE
    await job_tracker.queue_for_safety(
        HordeJobInfo(
            sdk_api_job_info=job,
            job_image_results=[HordeImageResult(image_bytes=b"raw")],
            state=GENERATION_STATE.ok,
            censored=False,
            time_popped=0.0,
        ),
    )
    assert scheduler.job_awaits_admission(job_id) is False
    assert scheduler.job_awaits_admission("no-such-job") is False


@pytest.mark.parametrize("primed_tick_observed", [True, False], ids=["primed_tick", "first_step_before_tick"])
async def test_a_grant_survives_the_staged_childs_preload_reports(primed_tick_observed: bool) -> None:
    """A cleared child keeps its grant through the preload states it reports and samples priced.

    The parent marks a slot primed at dispatch and the controller clears it there. A child sent its job with a
    preload then reports the preload's states before it primes, and the first step can flip the slot to
    sampling between two controller ticks. A grant retired on those reports leaves the onset reading as an
    unpriced window although the child samples on the permit it was granted.
    """
    scheduler, _job, waiter = await _staged_waiter(device_free_mb=24000.0)
    controller = ClearanceController(device_index=0, slot_cap=1, tail_overlap=False)
    clearance = threading.BoundedSemaphore(1)
    clearance.acquire()
    controller.register(0, ClearanceLeaseProxy(clearance=clearance, done=threading.Semaphore(0)))

    def tick() -> None:
        controller.step(scheduler.build_clearance_inputs(device_index=0), admit_fn=scheduler.clearance_admit_process)

    tick()
    assert controller.grant_state(0) is GrantState.CLEARED

    staged_reports = [
        HordeProcessState.UNLOADED_MODEL_FROM_RAM,
        HordeProcessState.PRELOADING_MODEL,
        HordeProcessState.PRELOADED_MODEL,
    ]
    if primed_tick_observed:
        staged_reports.append(HordeProcessState.INFERENCE_PRIMED)
    for reported_state in staged_reports:
        waiter.last_process_state = reported_state
        waiter.last_process_state_started_at = time.time()
        tick()
        assert controller.grant_state(0) is GrantState.CLEARED, reported_state

    waiter.last_process_state = HordeProcessState.INFERENCE_STARTING
    waiter.last_process_state_started_at = time.time()
    waiter.current_first_step_at = time.time()
    waiter.last_current_step = 3
    waiter.last_total_steps = 20
    tick()

    assert controller.grant_state(0) is GrantState.SAMPLING
    assert controller.unpriced_sampling_windows == 0
    assert controller.grants_issued == 1

    # The completion report clears the first-step stamp while the slot still owns the job until its result
    # lands; the finished sample retires its grant and is never cleared again.
    waiter.last_process_state = HordeProcessState.INFERENCE_COMPLETE
    waiter.last_process_state_started_at = time.time()
    waiter.current_first_step_at = None
    tick()

    assert controller.grant_state(0) is GrantState.IDLE
    assert controller.grants_issued == 1


async def test_a_pinned_disaggregated_sampler_waits_on_clearance_only_from_its_sample_prime() -> None:
    """A sampler pinned to a disaggregated job is not a clearance waiter during the encode, only once it primes.

    The pin binds the sampler at admission, before the encode runs on another lane, so the slot owns the job
    while it sits idle. A grant issued then would hold a clearance slot through the whole encode.
    """
    scheduler, job, waiter = await _staged_waiter(device_free_mb=24000.0)
    waiter.record_inference_ownership(job, attempt_ordinal=1, disaggregated=True)
    ownership = waiter.inference_ownership
    assert ownership is not None

    def staged_waiter_ids() -> list[int]:
        return [staged.process_id for staged in scheduler.build_clearance_inputs(device_index=0).staged_waiters]

    waiter.last_process_state = HordeProcessState.WAITING_FOR_JOB
    waiter.last_process_state_started_at = ownership.recorded_at + 1.0
    assert waiter.is_staged_short_of_sampling() is False
    assert staged_waiter_ids() == [], "a pinned sampler is not a waiter while its encode runs elsewhere"

    waiter.last_process_state = HordeProcessState.INFERENCE_PRIMED
    waiter.last_process_state_started_at = ownership.recorded_at + 2.0
    assert waiter.is_staged_short_of_sampling() is True
    assert staged_waiter_ids() == [0], "the sample stage's prime makes the pinned sampler a waiter"


async def _refused_past_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    wddm_paging_active: bool,
    model_crashed_a_child_natively: bool,
) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse, HordeProcessInfo, Mock]:
    """A staged waiter the arbiter still DENYs past its attempt deadline, with nothing to reclaim or wait for.

    Returns the scheduler, the staged job, the waiter, and the stand-in for the lane replacement.
    """
    from horde_worker_regen.process_management.resources.vram_arbiter import VramDisposition
    from horde_worker_regen.process_management.scheduling import inference_scheduler as scheduler_module

    scheduler, job, waiter = await _staged_waiter(device_free_mb=100.0)
    real_plan = _decide(scheduler, 0)
    assert real_plan.verdict is not None and real_plan.priced is not None
    # Refused, and waiting past the deadline the one real load was bounded by.
    starved_request = replace(
        real_plan.priced.request,
        starved_seconds=50.0,
        lease_wait_seconds=50.0,
        attempt_deadline_seconds=40.0,
        wddm_paging_active=wddm_paging_active,
    )
    denied = replace(
        real_plan,
        verdict=replace(real_plan.verdict, disposition=VramDisposition.DENY),
        priced=replace(real_plan.priced, request=starved_request),
    )
    monkeypatch.setattr(scheduler_module, "decide_clearance_admit", lambda *args, **kwargs: denied)
    scheduler._actuate_materialization_verdict = Mock(return_value=())  # type: ignore[method-assign]
    scheduler._waiting_can_help_staged_waiter = Mock(return_value=False)  # type: ignore[method-assign]
    scheduler._process_lifecycle.model_has_crashed_a_child_natively = Mock(  # type: ignore[method-assign]
        return_value=model_crashed_a_child_natively,
    )
    replaced = Mock()
    scheduler._process_lifecycle._replace_inference_process = replaced  # type: ignore[method-assign]
    return scheduler, job, waiter, replaced


async def test_a_terminal_refusal_faults_the_staged_job_and_replaces_the_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DENY past the deadline on a paging card ends the wait, not the lease timeout.

    While the driver pages the worker's allocations, the child's unpriced sample at its lease-acquire timeout
    fails natively instead of degrading. The job is faulted for reissue and the lane replaced deliberately.
    """
    scheduler, _job, waiter, replaced = await _refused_past_its_deadline(
        monkeypatch,
        wddm_paging_active=True,
        model_crashed_a_child_natively=False,
    )

    assert scheduler.clearance_admit_process(0) is False

    replaced.assert_called_once()
    assert replaced.call_args.args[0] is waiter
    assert replaced.call_args.kwargs["intentional_reason"]
    assert replaced.call_args.kwargs["resource_fault_reason"]


async def test_a_refusal_past_the_deadline_without_crash_evidence_keeps_the_waiter_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without paging or a prior native crash of the model, a refusal past the deadline stays a hold.

    Clearance prices the whole weights, and ComfyUI can serve a checkpoint it loads partially, so the child's
    sample at its lease-acquire timeout is the job's remaining path to success.
    """
    scheduler, job, _waiter, replaced = await _refused_past_its_deadline(
        monkeypatch,
        wddm_paging_active=False,
        model_crashed_a_child_natively=False,
    )

    assert scheduler.clearance_admit_process(0) is False

    replaced.assert_not_called()
    assert str(job.id_) in scheduler._clearance_starved_since, "the waiter is held, its starvation recorded"


async def test_a_prior_native_crash_of_the_jobs_model_makes_the_refusal_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that has crashed an inference child natively this session makes a refusal past the deadline final.

    Its unpriced sample is known to end in a native crash, which is never classified as a resource failure.
    """
    scheduler, job, waiter, replaced = await _refused_past_its_deadline(
        monkeypatch,
        wddm_paging_active=False,
        model_crashed_a_child_natively=True,
    )

    assert scheduler.clearance_admit_process(0) is False

    scheduler._process_lifecycle.model_has_crashed_a_child_natively.assert_called_with(job.model)
    replaced.assert_called_once()
    assert replaced.call_args.args[0] is waiter
    assert replaced.call_args.kwargs["resource_fault_reason"]


_SIBLING_HOLDS_MB = 554
"""What a light staged sibling holds on the card once its encode is done: far below the staging charge."""


async def _head_beside_staged_sibling(
    *, device_free_mb: float
) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse, ImageGenerateJobPopResponse, HordeProcessInfo]:
    """The clearance head on process 0 and, behind it in slot order, a primed sibling on process 1.

    The sibling's staging charge is booked against the reading it already shows, so the whole charge stands
    outstanding whatever its encode turns out to hold.
    """
    scheduler, job, _head = await _staged_waiter(device_free_mb=device_free_mb)
    sibling = make_mock_process_info(1, model_name=_JOB_MODEL, state=HordeProcessState.INFERENCE_PRIMED)
    sibling.total_vram_mb = 16000
    sibling.process_reserved_mb = _SIBLING_HOLDS_MB
    scheduler._process_map[1] = sibling
    sibling_job = make_job_pop_response(_JOB_MODEL)
    await track_popped_job_async(scheduler._job_tracker, sibling_job)  # type: ignore[attr-defined]
    await mark_job_in_progress_async(scheduler._job_tracker, sibling_job)  # type: ignore[attr-defined]
    sibling.last_job_referenced = sibling_job
    scheduler._record_dispatch_reservation(sibling_job, sibling, baseline=None, staging_only=True)  # type: ignore[attr-defined]
    return scheduler, job, sibling_job, sibling


async def _free_that_fits_the_head_only_without_the_sibling_charge() -> float:
    """A device-free reading the head fits within by less than the staging charge.

    Read off the head priced alone, so the figure follows the head's own price and the card's noise buffer.
    """
    scheduler, _job, _waiter = await _staged_waiter(device_free_mb=24000.0)
    plan = _decide(scheduler, 0)
    assert plan.verdict is not None and plan.candidate_delta_mb is not None
    noise_mb = plan.verdict.measured.noise_buffer_mb
    assert noise_mb is not None
    margin_mb = pricing.STAGING_ENCODE_VRAM_MB / 2
    return plan.candidate_delta_mb + noise_mb + margin_mb


async def test_a_head_beside_a_sibling_waiting_on_clearance_is_not_charged_its_staging() -> None:
    """A sibling that has entered its clearance wait is charged nothing beyond what the card already shows.

    Its encode is done, so everything it holds is inside the device-free reading. Charged the staging charge
    as well, the head reads short by room the card already has and waits on lane reclaim that cannot help.
    """
    device_free_mb = await _free_that_fits_the_head_only_without_the_sibling_charge()
    scheduler, _job, sibling_job, sibling = await _head_beside_staged_sibling(device_free_mb=device_free_mb)
    sibling.note_clearance_wait_entered(time.time())
    snapshot = scheduler.snapshot()
    outstanding_mb = snapshot.services.reserve_ledger.planned_charge_for_unit(
        DISPATCH_ADMISSION_FLOW, str(sibling_job.id_), dict(snapshot.card(None).reserved_by_pid)
    )
    assert outstanding_mb == pricing.STAGING_ENCODE_VRAM_MB

    plan = _decide(scheduler, 0)

    assert plan.priced is not None and plan.verdict is not None
    assert plan.priced.request.sibling_staging_relief_mb == outstanding_mb
    assert plan.verdict.measured.outstanding_reservations_mb == 0.0
    assert plan.decision is ClearanceDecision.ADMIT


async def test_a_head_beside_a_sibling_still_encoding_keeps_the_staging_charge_less_its_holding() -> None:
    """A sibling whose encode is not done stays charged the staging charge less what it already holds."""
    device_free_mb = await _free_that_fits_the_head_only_without_the_sibling_charge()
    scheduler, _job, _sibling_job, _sibling = await _head_beside_staged_sibling(device_free_mb=device_free_mb)

    plan = _decide(scheduler, 0)

    assert plan.priced is not None and plan.verdict is not None
    assert plan.priced.request.sibling_staging_relief_mb == _SIBLING_HOLDS_MB
    assert plan.verdict.measured.outstanding_reservations_mb == pricing.STAGING_ENCODE_VRAM_MB - _SIBLING_HOLDS_MB
    assert plan.decision is ClearanceDecision.HOLD


async def test_a_wait_entry_older_than_the_siblings_dispatch_is_no_evidence_its_encode_is_done() -> None:
    """A wait entry stamped before the sibling took its job belongs to an earlier job and earns no relief."""
    device_free_mb = await _free_that_fits_the_head_only_without_the_sibling_charge()
    scheduler, _job, _sibling_job, sibling = await _head_beside_staged_sibling(device_free_mb=device_free_mb)
    ownership = sibling.inference_ownership
    assert ownership is not None
    sibling.note_clearance_wait_entered(ownership.recorded_at - 1.0)

    plan = _decide(scheduler, 0)

    assert plan.priced is not None
    assert plan.priced.request.sibling_staging_relief_mb == _SIBLING_HOLDS_MB
    assert plan.decision is ClearanceDecision.HOLD


async def test_a_cleared_sibling_keeps_its_whole_charge() -> None:
    """A sibling already granted clearance is about to load its weights, so nothing it is charged is waived."""
    device_free_mb = await _free_that_fits_the_head_only_without_the_sibling_charge()
    scheduler, _job, sibling_job, sibling = await _head_beside_staged_sibling(device_free_mb=device_free_mb)
    sibling.note_clearance_wait_entered(time.time())
    scheduler._dispatch_holds.note_clearance_granted(str(sibling_job.id_))  # type: ignore[attr-defined]

    plan = _decide(scheduler, 0)

    assert plan.priced is not None and plan.verdict is not None
    assert plan.priced.request.sibling_staging_relief_mb == 0.0
    assert plan.verdict.measured.outstanding_reservations_mb == pricing.STAGING_ENCODE_VRAM_MB
    assert plan.decision is ClearanceDecision.HOLD


async def test_a_sampling_sibling_keeps_its_whole_charge() -> None:
    """A sibling past its staging (sampling, here unpriced after its acquire timed out) is not a staged waiter."""
    device_free_mb = await _free_that_fits_the_head_only_without_the_sibling_charge()
    scheduler, _job, _sibling_job, sibling = await _head_beside_staged_sibling(device_free_mb=device_free_mb)
    sibling.note_clearance_wait_entered(time.time())
    sibling.last_process_state = HordeProcessState.INFERENCE_STARTING
    sibling.last_process_state_started_at = time.time()

    plan = _decide(scheduler, 0)

    assert plan.priced is not None
    assert plan.priced.request.sibling_staging_relief_mb == 0.0
    assert plan.decision is ClearanceDecision.HOLD


async def test_a_grant_is_recorded_for_the_cleared_job() -> None:
    """The scheduler records each job it grants clearance, which is what keeps a cleared sibling charged."""
    scheduler, job, _waiter = await _staged_waiter(device_free_mb=24000.0)

    assert scheduler.clearance_admit_process(0) is True

    assert str(job.id_) in scheduler.snapshot().ledgers.dispatch_holds.clearance_granted_ids


_KREA2_MODEL = "Krea2-Turbo_fp8"
_KREA2_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.krea2_turbo
_KREA2_SHAPE = (768, 1344)
_SIXTEEN_GB_MB = 16384
_KREA2_DEVICE_FREE_MB = 13000.0
"""What a converged 16 GB card reads free with the staged child's context and encode set on it."""


async def _krea2_staged_waiter(
    *, with_lora: bool = False, paging: bool = False
) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse, HordeProcessInfo]:
    """One primed child on a 16 GB card holding a staged Krea2 job whose one real load on the card is spent.

    The job's whole sampling peak sits above the card's achievable ceiling, and a spent attempt leaves the
    ceiling trigger nothing to offer, so the full price alone refuses it.
    """
    from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry

    tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        bridge_data=make_mock_bridge_data(
            gpu_sampling_lease_enabled=True,
            enable_vram_budget=True,
            vram_reserve_mb=2048,
            ram_reserve_mb=4096,
        ),
        job_tracker=tracker,
        device_free_mb=_KREA2_DEVICE_FREE_MB,
        model_metadata=make_test_model_metadata(
            {_KREA2_MODEL: make_mock_model_reference_record(_KREA2_MODEL, baseline=_KREA2_BASELINE)},
        ),
    )
    waiter = make_mock_process_info(0, model_name=_KREA2_MODEL, state=HordeProcessState.INFERENCE_PRIMED)
    waiter.total_vram_mb = _SIXTEEN_GB_MB
    waiter.process_reserved_mb = 900
    scheduler._process_map = ProcessMap({0: waiter})
    width, height = _KREA2_SHAPE
    loras = [LorasPayloadEntry(name="some_lora", model=1.0, clip=1.0)] if with_lora else None
    job = make_job_pop_response(_KREA2_MODEL, width=width, height=height, loras=loras)
    await track_popped_job_async(tracker, job)
    await mark_job_in_progress_async(tracker, job)
    assert job.id_ is not None
    tracked = tracker.get_tracked_job(job.id_)
    assert tracked is not None
    # A single-card worker scopes the card as None. The retry's earlier attempt ran there.
    tracked.measured_attempted_device_indices.add(None)
    waiter.last_job_referenced = job
    scheduler._record_dispatch_reservation(job, waiter, baseline=None, staging_only=True)  # type: ignore[attr-defined]
    if paging:
        scheduler.note_wddm_paging({100001: 512.0}, active=True)
    return scheduler, job, waiter


def _expected_seat_mb(
    scheduler: InferenceScheduler, job: ImageGenerateJobPopResponse, weight_fraction: float
) -> float:
    """The seat the pricing owners describe: a share of the weights plus the larger non-weight charge."""
    from horde_worker_regen.process_management.resources.resource_budget import effective_inference_reserve_mb

    snapshot = scheduler.snapshot()
    weights_mb = predict_job_weight_mb(job, str(_KREA2_BASELINE))
    activation_mb = pricing.partial_seat_activation_mb(snapshot, job, str(_KREA2_BASELINE))
    assert weights_mb is not None and activation_mb is not None
    return weight_fraction * weights_mb + max(activation_mb, effective_inference_reserve_mb(_SIXTEEN_GB_MB, 0.0))


async def test_an_extra_large_job_over_the_ceiling_is_admitted_at_its_partial_load_seat() -> None:
    """A Krea2 job without LoRAs on a converged 16 GB card is granted at its seat, which the full price refuses.

    The full sampling peak exceeds what the emptied card offers, and with its one real load spent the job has
    no other priced way onto the card. ComfyUI serves it by loading the weights partially, so clearance
    prices that seat: half the weights without a measured upload rate, plus the larger of the activation and
    ComfyUI's inference reserve.
    """
    scheduler, job, _waiter = await _krea2_staged_waiter()
    plan = _decide(scheduler, 0)

    assert plan.priced is not None and plan.verdict is not None
    request = plan.priced.request
    assert request.partial_seat_mb == pytest.approx(
        _expected_seat_mb(scheduler, job, pricing.PARTIAL_SEAT_FALLBACK_WEIGHT_FRACTION),
    )
    assert plan.decision is ClearanceDecision.ADMIT
    assert plan.verdict.partial_seat is True
    assert plan.verdict.measured_attempt is False


def _add_idle_siblings_holding_host_ram_checkpoints(scheduler: InferenceScheduler) -> None:
    """Two idle lanes that last unloaded their SDXL checkpoints from VRAM and keep them only in host RAM."""
    from horde_worker_regen.process_management.ipc.messages import HeldComponentSnapshot

    for process_id in (1, 2):
        sibling = make_mock_process_info(process_id, model_name=f"sdxl-{process_id}")
        sibling.total_vram_mb = _SIXTEEN_GB_MB
        sibling.process_reserved_mb = 60
        sibling.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        sibling.held_components = [
            HeldComponentSnapshot(kind="checkpoint", identity=f"sdxl-{process_id}", approx_ram_mb=6800.0),
        ]
        scheduler._process_map[process_id] = sibling


async def test_idle_siblings_holding_host_ram_checkpoints_do_not_hold_off_the_seat() -> None:
    """Checkpoints parked in host RAM return no VRAM, so the card still reads converged-empty for the seat.

    The failure this encodes: every idle lane holding a component cache entry counted as card tenancy the head
    could still reclaim, so the seat was refused on a card whose idle lanes held nothing on the device, and the
    child waited out its lease-acquire timeout to make the same partial load unpriced.
    """
    scheduler, _job, _waiter = await _krea2_staged_waiter()
    _add_idle_siblings_holding_host_ram_checkpoints(scheduler)

    plan = _decide(scheduler, 0)

    assert plan.priced is not None and plan.verdict is not None
    assert plan.priced.request.has_reclaimable_idle_tenancy is False
    assert plan.decision is ClearanceDecision.ADMIT
    assert plan.verdict.partial_seat is True


async def test_counting_host_ram_checkpoints_as_tenancy_refuses_the_seat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect reinjection: with host-RAM components counted as card tenancy, the same waiter is held."""

    def ram_counting_tenancy(snapshot, head_model, target_process_id, *, device_index):  # noqa: ANN001, ANN202
        for slot in pricing.card_slots(snapshot, device_index):
            if slot.process_type is not HordeProcessType.INFERENCE or slot.process_id == target_process_id:
                continue
            if head_model is not None and slot.model == head_model:
                continue
            if slot.held_component_count > 0 or slot.parked_preload:
                return True
        return False

    monkeypatch.setattr(pricing, "has_reclaimable_idle_tenancy", ram_counting_tenancy)
    scheduler, _job, _waiter = await _krea2_staged_waiter()
    _add_idle_siblings_holding_host_ram_checkpoints(scheduler)

    plan = _decide(scheduler, 0)

    assert plan.verdict is not None
    assert plan.decision is ClearanceDecision.HOLD
    assert plan.verdict.partial_seat is False


async def test_a_seat_admit_books_no_more_than_the_room_the_child_was_granted() -> None:
    """The cleared job is booked the room it may load into, which decays to nothing once it reaches its peak.

    Booked at the full outstanding charge, the part above the granted room would stay outstanding through the
    whole sample, because the child only grows into what it was given.
    """
    scheduler, job, _waiter = await _krea2_staged_waiter()
    plan = _decide(scheduler, 0)
    assert plan.verdict is not None and plan.verdict.partial_seat is True
    available_mb = plan.verdict.measured.available_mb
    assert available_mb is not None and plan.candidate_delta_mb is not None
    assert available_mb < plan.candidate_delta_mb
    assert plan.booking_mb == pytest.approx(available_mb)

    assert scheduler.clearance_admit_process(0) is True

    ledger = scheduler._reserve_ledger  # type: ignore[attr-defined]
    assert ledger.planned_charge_for_unit(DISPATCH_ADMISSION_FLOW, str(job.id_), {0: 900.0}) == pytest.approx(
        available_mb,
    )


async def test_a_lora_job_keeps_the_full_price() -> None:
    """A LoRA job is never seated partially, since its patch temporaries crash a child natively in that regime."""
    scheduler, _job, _waiter = await _krea2_staged_waiter(with_lora=True)
    plan = _decide(scheduler, 0)

    assert plan.priced is not None and plan.verdict is not None
    assert plan.priced.request.partial_seat_mb is None
    assert plan.decision is ClearanceDecision.HOLD
    assert plan.verdict.partial_seat is False


async def test_a_paging_card_keeps_the_full_price() -> None:
    """A card whose driver is paging the worker's allocations never seats a job partially."""
    scheduler, _job, _waiter = await _krea2_staged_waiter(paging=True)
    plan = _decide(scheduler, 0)

    assert plan.priced is not None and plan.verdict is not None
    assert plan.priced.request.partial_seat_mb is None
    assert plan.decision is ClearanceDecision.HOLD
    assert plan.verdict.partial_seat is False


async def test_the_seat_weight_share_follows_the_waiter_clocks_upload_rate_and_step_time() -> None:
    """With an upload rate and a step time on the clock, the seat's weight share is derived from them."""
    scheduler, job, _waiter = await _krea2_staged_waiter()
    weights_mb = predict_job_weight_mb(job, str(_KREA2_BASELINE))
    assert weights_mb is not None
    upload_mb_per_second = 0.6 * weights_mb
    clock = StagedWaiterClock(
        starved_seconds=0.0,
        load_seconds=None,
        seconds_per_step=1.0,
        upload_mb_per_second=upload_mb_per_second,
    )
    arbiter = scheduler._ensure_preload_arbiter()  # type: ignore[attr-defined]
    plan = decide_clearance_admit(
        scheduler.snapshot(), 0, arbiter=arbiter, post_processing_deferred=False, waiter_clock=clock
    )

    assert plan.priced is not None
    assert plan.priced.request.partial_seat_mb == pytest.approx(_expected_seat_mb(scheduler, job, 0.4))


async def test_the_waiter_clock_carries_the_cards_measured_upload_rate() -> None:
    """The clock reads the card's upload rate from the last job its inference lanes reported."""
    from hordelib.metrics import JobPhaseMetrics, JobVramFootprint, ModelLoadEvent

    scheduler, job, waiter = await _staged_waiter(device_free_mb=24000.0)
    assert scheduler._staged_waiter_clock(waiter, job).upload_mb_per_second is None  # type: ignore[attr-defined]

    waiter.last_job_metrics = JobPhaseMetrics(
        model_loads=[ModelLoadEvent(model_name="m", phase="ram_to_vram", duration_seconds=4.0, timestamp=0.0)],
        vram_footprint=JobVramFootprint(peak_resident_weights_mb=8000.0),
    )
    clock = scheduler._staged_waiter_clock(waiter, job)  # type: ignore[attr-defined]
    assert clock.upload_mb_per_second == pytest.approx(2000.0)
