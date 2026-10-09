"""The materialisation request: pricing a job's VRAM landing on a slot through the MONOLITHIC_DISPATCH identity.

Shared by the dispatch residency-reconciliation gate, the clearance gate and the preload budget step. The
request is built from the snapshot alone; evaluating it is the arbiter's, and running the verdict's actuations
is the executor's.
"""

from __future__ import annotations

from dataclasses import dataclass

from horde_worker_regen.process_management.resources.resource_budget import (
    StreamForecast,
    effective_inference_reserve_mb,
    predict_job_weight_mb,
)
from horde_worker_regen.process_management.resources.vram_arbiter import (
    LaneRungGrade,
    VramArbiter,
    VramRequest,
    VramRequestKind,
)
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.admission.snapshot import SchedulingSnapshot
from horde_worker_regen.process_management.scheduling.workload_flow import (
    DISPATCH_ADMISSION_FLOW,
    PRELOAD_ADMISSION_FLOW,
)


@dataclass(frozen=True)
class ContextReduction:
    """How deep a context reduction on behalf of the job may go, and whether it is a warranted remedy.

    Dispatch and clearance size the depth from the candidate delta and never offer the live-context reduction;
    a preload sizes it from the predictive verdict's rejected peak (the conservative burden estimate the static
    budget declines on, so the structural remedy fires exactly where admission would otherwise thrash) and
    offers the reduction only for a head whose demand the forecast makes trustworthy.
    """

    max_resident: int | None
    """The largest live inference-process count that still seats the peak, or None when it cannot be sized."""
    can_reduce_live_contexts: bool


@dataclass(frozen=True)
class MaterializationRequest:
    """A priced materialisation, ready for the arbiter, with the figures its actuations act on."""

    request: VramRequest
    max_resident: int | None
    """The context-reduction depth a REDUCE_LIVE_CONTEXTS actuation collapses the card to, or None."""
    candidate_delta_mb: float | None
    device_index: int | None


@dataclass(frozen=True)
class StagedWaiterTerms:
    """Represents what clearance knows about a staged waiter beyond its outstanding charge."""

    held_mb: float
    """What the job already holds on the card (MB), netted out of its outstanding charge."""
    starved_seconds: float
    """The clearance clock: seconds the waiter has been held where waiting could not help."""
    attempt_deadline_seconds: float
    """The clock reading by which its measured-load probe must be eligible."""
    lane_rung_grade: LaneRungGrade = LaneRungGrade.NO_RUNG
    """Where the waiter's starvation episode stands with the service-lane rungs applied for it."""


def head_starved_seconds(snapshot: SchedulingSnapshot, job_id: str) -> float:
    """Seconds the job has been the idle-device head, or 0.0 when it is not the timed head.

    A job already in progress is never the timed head: a staged waiter's wait is clearance's clock.
    """
    view = snapshot.ledgers.head_admission
    if view.starvation_job_id != job_id or view.starvation_since == 0.0:
        return 0.0
    job = snapshot.queue.jobs.get(job_id)
    if job is not None and job.in_progress:
        return 0.0
    return snapshot.now - view.starvation_since


def build_materialization_request(
    snapshot: SchedulingSnapshot,
    job_id: str,
    process_id: int,
    *,
    arbiter: VramArbiter,
    is_head_of_queue: bool,
    head_outstanding_mb: float | None,
    candidate_delta_override_mb: float | None = None,
    staged_waiter: StagedWaiterTerms | None = None,
    nets_own_dispatch_reservation: bool = False,
    sibling_staging_relief_mb: float = 0.0,
    kind: VramRequestKind = VramRequestKind.MONOLITHIC_DISPATCH,
    context_reduction: ContextReduction | None = None,
    forecast: StreamForecast | None = None,
    prepared_head_reprices_activation: bool = True,
) -> MaterializationRequest:
    """Price the job's landing on the slot as the arbiter will see it.

    ``candidate_delta_override_mb`` prices a smaller charge than the full peak (the staged child's remaining
    materialisation at clearance, the staging-capped charge of a preload under the lease); ``staged_waiter``
    carries clearance's held figure and clock for that staged child, in place of the head-starvation clock;
    ``nets_own_dispatch_reservation`` nets the job's own outstanding dispatch reservation out of the overlay,
    for the clearance re-price of an already-dispatched job; ``sibling_staging_relief_mb`` is the part of other
    staged waiters' staging charges that re-price nets out. ``context_reduction`` supplies a depth already sized
    by the caller (the preload's, from the predictive peak); without it the depth is sized from the candidate
    delta and the live-context reduction is never offered. ``prepared_head_reprices_activation`` is the dispatch
    rule that an aux-prepared job beside live work on the card re-prices its activation even where its weights
    are resident; a preload never does. The idle-context teardown is widened in the measured frame through the
    arbiter's deficit, the one thing here that is not read from the snapshot.
    """
    job = snapshot.queue.jobs[job_id]
    payload = snapshot.queue.payloads[job_id]
    if job.model is None:
        raise ValueError("materialisation admission requires a job with a model")
    slot = snapshot.slots[process_id]
    device_index = snapshot.routing_device_index(slot)
    card = snapshot.card(device_index)
    ledger = snapshot.services.reserve_ledger
    own_dispatch_mb = 0.0
    if nets_own_dispatch_reservation:
        own_dispatch_mb = ledger.planned_charge_for_unit(DISPATCH_ADMISSION_FLOW, job_id, dict(card.reserved_by_pid))
    baseline = job.baseline
    has_reclaimable_idle_model = pricing.has_reclaimable_idle_model(
        snapshot,
        process_id,
        for_head_of_queue=is_head_of_queue,
        device_index=device_index,
        make_room_for_model=job.model,
    )
    candidate_delta_mb = (
        candidate_delta_override_mb
        if candidate_delta_override_mb is not None
        else pricing.candidate_delta_mb(
            snapshot,
            payload,
            baseline,
            process_id=process_id,
            disaggregated=job.disaggregation_class_eligible,
        )
    )
    if context_reduction is None:
        if forecast is None:
            forecast = pricing.forecast_streaming(snapshot, payload, baseline, device_index=device_index)
        structural_reserve_mb = (
            effective_inference_reserve_mb(card.total_vram_mb, 0.0)
            if card.total_vram_mb is not None
            else forecast._effective_base_reserve  # noqa: SLF001 - the same budget owner sizes the teardown depth.
        )
        context_reduction = ContextReduction(
            max_resident=(
                pricing.max_coresident_for_peak_mb(
                    snapshot, candidate_delta_mb, structural_reserve_mb, device_index=device_index
                )
                if candidate_delta_mb is not None
                else None
            ),
            can_reduce_live_contexts=False,
        )
    max_resident = context_reduction.max_resident
    live = pricing.loaded_inference_count(snapshot, device_index)
    idle_contexts_teardownable = (
        is_head_of_queue
        and max_resident is not None
        and max_resident < live
        and pricing.has_teardownable_idle_context(snapshot, process_id, device_index=device_index)
    )
    reprices_activation = (
        prepared_head_reprices_activation
        and job.aux_models_prepared
        and bool(pricing.active_jobs_on_card(snapshot, device_index))
    )
    config = snapshot.config_for(device_index)
    request = VramRequest(
        kind=kind,
        job_label=str(job.model),
        baseline=baseline,
        device_index=device_index,
        target_process_id=process_id,
        candidate_delta_mb=candidate_delta_mb,
        candidate_held_mb=staged_waiter.held_mb if staged_waiter is not None else 0.0,
        candidate_weights_mb=predict_job_weight_mb(payload, baseline),
        accepted_work=job.tracked,
        candidate_already_resident=(
            pricing.candidate_weights_resident(snapshot, job.model, process_id) and not reprices_activation
        ),
        own_planned_unmaterialized_mb=ledger.planned_charge_for_unit(
            PRELOAD_ADMISSION_FLOW,
            str(process_id),
            dict(card.reserved_by_pid),
        ),
        own_dispatch_unmaterialized_mb=own_dispatch_mb,
        sibling_staging_relief_mb=sibling_staging_relief_mb,
        is_head_of_queue=is_head_of_queue,
        head_job_id=job_id,
        wddm_paging_active=snapshot.ledgers.retention.wddm_paging_active,
        candidate_measured=pricing.sampling_peak_measured(
            snapshot,
            payload,
            baseline,
            process_id=process_id,
            disaggregated=job.disaggregation_class_eligible,
        ),
        measured_attempt_in_progress=device_index in job.measured_attempt_devices,
        measured_attempt_already_spent=device_index in job.measured_attempt_spent_devices,
        head_outstanding_mb=head_outstanding_mb,
        starved_seconds=(
            staged_waiter.starved_seconds if staged_waiter is not None else head_starved_seconds(snapshot, job_id)
        ),
        probe_after_seconds=float(config.measured_load_probe_seconds),
        attempt_deadline_seconds=staged_waiter.attempt_deadline_seconds if staged_waiter is not None else None,
        lane_rung_grade=staged_waiter.lane_rung_grade if staged_waiter is not None else LaneRungGrade.NO_RUNG,
        has_reclaimable_idle_tenancy=pricing.has_reclaimable_idle_tenancy(
            snapshot,
            job.model,
            process_id,
            device_index=device_index,
        ),
        lane_reclaim_permitted=bool(config.starved_head_lane_reclaim),
        has_reclaimable_idle_model=has_reclaimable_idle_model,
        can_reduce_live_contexts=context_reduction.can_reduce_live_contexts,
        idle_contexts_teardownable=idle_contexts_teardownable,
    )
    request, max_resident = pricing.apply_measured_context_teardown(
        snapshot,
        request,
        arbiter,
        process_id,
        structural_max_resident=max_resident,
        device_index=device_index,
    )
    return MaterializationRequest(
        request=request,
        max_resident=max_resident,
        candidate_delta_mb=candidate_delta_mb,
        device_index=device_index,
    )
