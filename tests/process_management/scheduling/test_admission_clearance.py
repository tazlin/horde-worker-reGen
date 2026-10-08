"""The clearance admit decision over the scheduling snapshot."""

from __future__ import annotations

from dataclasses import replace

from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
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
from horde_worker_regen.process_management.scheduling.clearance_lease import CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
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


async def test_a_job_awaits_admission_until_its_lane_samples() -> None:
    """A lane pause's beneficiary reads as waiting while queued or staged, and as served once it samples.

    The restore evidence must not read the card: the context the pause freed is what makes the card healthy.
    """
    job_tracker = JobTracker()
    job = make_job_pop_response("stable_diffusion")
    await track_popped_job_async(job_tracker, job)
    lane = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.WAITING_FOR_JOB)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({0: lane}), job_tracker=job_tracker)
    job_id = str(job.id_)

    assert scheduler.job_awaits_admission(job_id) is True

    await mark_job_in_progress_async(job_tracker, job)
    lane.record_inference_ownership(job, attempt_ordinal=1)
    for staged_state in (HordeProcessState.PRELOADED_MODEL, HordeProcessState.INFERENCE_PRIMED):
        lane.last_process_state = staged_state
        assert scheduler.job_awaits_admission(job_id) is True

    lane.last_process_state = HordeProcessState.INFERENCE_STARTING
    assert scheduler.job_awaits_admission(job_id) is False
    assert scheduler.job_awaits_admission("no-such-job") is False
