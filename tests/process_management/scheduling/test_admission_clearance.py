"""The clearance admit decision over the scheduling snapshot."""

from __future__ import annotations

from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.process_management.resources.resource_budget import (
    predict_job_weight_mb,
    predict_model_weight_mb,
)
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.admission.clearance import (
    ClearanceDecision,
    decide_clearance_admit,
    staged_materialization_delta_mb,
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
