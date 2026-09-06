"""The materialisation request: pricing a job's VRAM landing on a slot through the MONOLITHIC_DISPATCH identity.

Shared by the dispatch residency-reconciliation gate and the clearance gate. The request is built from the
snapshot alone; evaluating it is the arbiter's, and running the verdict's actuations is the executor's.
"""

from __future__ import annotations

from dataclasses import dataclass

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.resources.resource_budget import (
    effective_inference_reserve_mb,
    predict_job_weight_mb,
)
from horde_worker_regen.process_management.resources.vram_arbiter import VramArbiter, VramRequest, VramRequestKind
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.admission.snapshot import SchedulingSnapshot
from horde_worker_regen.process_management.scheduling.workload_flow import (
    DISPATCH_ADMISSION_FLOW,
    PRELOAD_ADMISSION_FLOW,
)


@dataclass(frozen=True)
class MaterializationRequest:
    """A priced materialisation, ready for the arbiter, with the figures its actuations act on."""

    request: VramRequest
    max_resident: int | None
    """The context-reduction depth a REDUCE_LIVE_CONTEXTS actuation collapses the card to, or None."""
    candidate_delta_mb: float | None
    device_index: int | None


def head_starved_seconds(snapshot: SchedulingSnapshot, job_id: str) -> float:
    """Seconds the job has been the idle-device head, or 0.0 when it is not the timed head."""
    view = snapshot.ledgers.head_admission
    if view.starvation_job_id != job_id or view.starvation_since == 0.0:
        return 0.0
    return snapshot.now - view.starvation_since


def active_jobs_on_card(snapshot: SchedulingSnapshot, device_index: int | None) -> tuple[str, ...]:
    """The in-progress jobs whose running slot sits on the card, or every in-progress job when card-agnostic."""
    if not snapshot.multi_gpu_routing_active or device_index is None:
        return snapshot.queue.in_progress
    running = {
        slot.current_job_id
        for slot in snapshot.slots.values()
        if slot.process_type is HordeProcessType.INFERENCE and slot.device_index == device_index
    }
    return tuple(job_id for job_id in snapshot.queue.in_progress if job_id in running)


def build_materialization_request(
    snapshot: SchedulingSnapshot,
    job_id: str,
    process_id: int,
    *,
    arbiter: VramArbiter,
    is_head_of_queue: bool,
    head_outstanding_mb: float | None,
    candidate_delta_override_mb: float | None = None,
    nets_own_dispatch_reservation: bool = False,
) -> MaterializationRequest:
    """Price the job's landing on the slot as the arbiter will see it.

    ``candidate_delta_override_mb`` prices a smaller charge than the full peak (the staged child's remaining
    materialisation at clearance); ``nets_own_dispatch_reservation`` nets the job's own outstanding dispatch
    reservation out of the overlay, for the clearance re-price of an already-dispatched job. The idle-context
    teardown is widened in the measured frame through the arbiter's deficit, the one thing here that is not
    read from the snapshot.
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
    forecast = pricing.forecast_streaming(snapshot, payload, baseline, device_index=device_index)
    structural_reserve_mb = (
        effective_inference_reserve_mb(card.total_vram_mb, 0.0)
        if card.total_vram_mb is not None
        else forecast._effective_base_reserve  # noqa: SLF001 - the same budget owner sizes the teardown depth.
    )
    max_resident = (
        pricing.max_coresident_for_peak_mb(
            snapshot, candidate_delta_mb, structural_reserve_mb, device_index=device_index
        )
        if candidate_delta_mb is not None
        else None
    )
    live = pricing.loaded_inference_count(snapshot, device_index)
    idle_contexts_teardownable = (
        is_head_of_queue
        and max_resident is not None
        and max_resident < live
        and pricing.has_teardownable_idle_context(snapshot, process_id, device_index=device_index)
    )
    prepared_head_reprices_activation = job.aux_models_prepared and bool(active_jobs_on_card(snapshot, device_index))
    config = snapshot.config_for(device_index)
    request = VramRequest(
        kind=VramRequestKind.MONOLITHIC_DISPATCH,
        job_label=str(job.model),
        baseline=baseline,
        device_index=device_index,
        target_process_id=process_id,
        candidate_delta_mb=candidate_delta_mb,
        candidate_weights_mb=predict_job_weight_mb(payload, baseline),
        accepted_work=job.tracked,
        candidate_already_resident=(
            pricing.candidate_weights_resident(snapshot, job.model, process_id)
            and not prepared_head_reprices_activation
        ),
        own_planned_unmaterialized_mb=ledger.planned_charge_for_unit(
            PRELOAD_ADMISSION_FLOW,
            str(process_id),
            dict(card.reserved_by_pid),
        ),
        own_dispatch_unmaterialized_mb=own_dispatch_mb,
        is_head_of_queue=is_head_of_queue,
        head_job_id=job_id,
        measured_attempt_in_progress=device_index in job.measured_attempt_devices,
        measured_attempt_already_spent=device_index in job.measured_attempt_spent_devices,
        head_outstanding_mb=head_outstanding_mb,
        starved_seconds=head_starved_seconds(snapshot, job_id),
        probe_after_seconds=float(config.measured_load_probe_seconds),
        has_reclaimable_idle_tenancy=pricing.has_reclaimable_idle_tenancy(
            snapshot,
            job.model,
            process_id,
            device_index=device_index,
        ),
        lane_reclaim_permitted=bool(config.starved_head_lane_reclaim),
        has_reclaimable_idle_model=has_reclaimable_idle_model,
        can_reduce_live_contexts=False,
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
