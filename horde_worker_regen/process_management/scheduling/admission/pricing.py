"""Pure VRAM pricing over a scheduling snapshot: what a job costs a card and what the card could give back.

Each function here is the snapshot form of a scheduler pricing method and gives the same answer over the same
worker; the differential tests hold the two together while both exist. Nothing here reads a collaborator or
records anything.
"""

from __future__ import annotations

import sys

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.models.model_sizing import ModelSizeTier, model_size_tier
from horde_worker_regen.process_management.resources.admission_identity import admission_margin_mb
from horde_worker_regen.process_management.resources.resource_budget import (
    _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB,
    StreamForecast,
    forecast_weight_streaming,
    predict_job_decode_spike_mb,
    predict_job_footprint_mb,
    predict_job_sampler_only_vram_mb,
    predict_job_sampling_vram_mb,
    predict_job_weight_mb,
)
from horde_worker_regen.process_management.resources.vram_arbiter import VramArbiter, VramRequest
from horde_worker_regen.process_management.resources.vram_footprints import (
    FootprintKey,
    FootprintStage,
    sampling_footprint_key,
)
from horde_worker_regen.process_management.scheduling.admission.snapshot import SchedulingSnapshot, SlotSnapshot
from horde_worker_regen.process_management.scheduling.governance.whole_card import max_coresident_for_peak
from horde_worker_regen.utils.config_coercion import config_number
from horde_worker_regen.utils.vram_quota import effective_post_process_vram_quota_mb

type Baseline = KNOWN_IMAGE_GENERATION_BASELINE | str | None

STAGING_ENCODE_VRAM_MB = 2048.0
"""VRAM a staged (dispatched but not-yet-cleared) job actually charges the device under the clearance
lease: the text-encoder footprint plus the conditioning working set for the largest supported family
(SDXL's dual CLIP encode lands near 1.5-2GB). Under the clearance lease the diffusion weights load
*inside* the leased sample call, at clearance, not at dispatch, so a staged job's device footprint is
only this encode working set until it is cleared. Dispatch admits staging while measured device free net
of the reserve covers this charge; the full materialisation is priced at clearance instead."""

# ---- context overheads


def config_overhead_override_mb(snapshot: SchedulingSnapshot) -> float | None:
    """The coerced ``vram_per_process_overhead_mb`` override, or None when unset or non-numeric."""
    return config_number(snapshot.config.vram_per_process_overhead_mb)


def per_process_overhead_mb(snapshot: SchedulingSnapshot, device_index: int | None) -> float:
    """The first-context VRAM overhead (MB) to assume on a card: configured override, else measured, else 0."""
    return snapshot.services.overhead.per_process_mb(
        config_override_mb=config_overhead_override_mb(snapshot),
        device_index=device_index,
    )


def marginal_process_overhead_mb(snapshot: SchedulingSnapshot, device_index: int | None) -> float | None:
    """The per-additional-context VRAM cost (MB), or None when unmeasured (callers fall back to the seed)."""
    return snapshot.services.overhead.marginal_mb(
        config_override_mb=config_overhead_override_mb(snapshot),
        device_index=device_index,
    )


def marginal_or_seed_mb(snapshot: SchedulingSnapshot, device_index: int | None) -> float:
    """The marginal context cost, or the seeded figure when it is unmeasured or non-positive."""
    marginal_mb = marginal_process_overhead_mb(snapshot, device_index)
    if marginal_mb is None or marginal_mb <= 0.0:
        return _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB
    return marginal_mb


def whole_card_warranted(forecast: StreamForecast, *, marginal_overhead_mb: float | None) -> bool:
    """Whether a teardown demand is trustworthy enough to reserve the card or reduce its live contexts.

    Reserving the card stops siblings, moves safety off-GPU and holds the device through a cooldown, so it must
    not fire on a measurement artifact. A card-demanding model warrants it outright; otherwise the per-context
    cost must have been measured, since the unmeasured fallback charges the one-time CUDA runtime against every
    context and can manufacture a demand for a model that co-resides with room to spare.
    """
    return forecast.is_card_demanding or marginal_overhead_mb is not None


def admission_noise_mb(snapshot: SchedulingSnapshot, device_index: int | None, total_vram_mb: float | None) -> float:
    """The admission margin (MB) for a card: its operator override, else the platform default."""
    override = snapshot.config_for(device_index).vram_admission_noise_mb
    return admission_margin_mb(total_vram_mb, override_mb=None if override is None else float(override))


# ---- slots on a card


def card_slots(snapshot: SchedulingSnapshot, device_index: int | None) -> tuple[SlotSnapshot, ...]:
    """The slots on a card, or every slot for the worker-wide key."""
    return tuple(slot for slot in snapshot.slots.values() if device_index is None or slot.device_index == device_index)


def loaded_inference_count(snapshot: SchedulingSnapshot, device_index: int | None) -> int:
    """How many inference processes are live on the card, the count the structural floor is keyed off."""
    return sum(
        1
        for slot in card_slots(snapshot, device_index)
        if slot.process_type is HordeProcessType.INFERENCE
        and slot.state not in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED)
    )


def in_progress_models(snapshot: SchedulingSnapshot) -> set[str | None]:
    """The models of every in-progress job."""
    return {snapshot.queue.jobs[job_id].model for job_id in snapshot.queue.in_progress}


def wanted_models(snapshot: SchedulingSnapshot) -> set[str]:
    """Every model resident on an inference slot or referenced by a pending or in-progress job."""
    wanted = {
        slot.model
        for slot in snapshot.slots.values()
        if slot.process_type is HordeProcessType.INFERENCE and slot.model is not None
    }
    wanted.update(job.model for job in snapshot.queue.jobs.values() if job.model is not None)
    return wanted


def next_models(snapshot: SchedulingSnapshot, count: int) -> list[str]:
    """The next ``count`` distinct models in the tracker's pending order."""
    models: list[str] = []
    for job_id in snapshot.queue.pending_in_pop_order:
        if len(models) >= count:
            break
        model = snapshot.queue.jobs[job_id].model
        if model is not None and model not in models:
            models.append(model)
    return models


def target_slot_is_spared(slot: SlotSnapshot, *, make_room_for_model: str | None) -> bool:
    """Whether the slot a load is aimed at is spared from the reclaim being performed for it.

    Spared outright when no model is named, when it is not an inference slot, when it is mid-load, or when it
    holds nothing or the model being seated; a different idle model on it is the ordinary cost of a swap.
    """
    if make_room_for_model is None or slot.process_type is not HordeProcessType.INFERENCE or slot.is_busy:
        return True
    return slot.model is None or slot.model == make_room_for_model


# ---- learned and measured footprints


def learned_resident_footprint_mb(snapshot: SchedulingSnapshot, model: str | None, baseline: Baseline) -> float | None:
    """The measured at-rest footprint (MB) of a checkpoint net of its context, or None when unmeasured."""
    store = snapshot.services.footprint_store
    if store is None or model is None or baseline is None:
        return None
    watermark_mb = store.estimate_mb(
        FootprintKey(
            model_baseline=str(baseline),
            resolution_bucket=None,
            platform=sys.platform,
            stage=FootprintStage.RESIDENT,
            checkpoint=model,
        ),
        static_seed_mb=0.0,
    )
    if watermark_mb <= 0.0:
        return None
    return max(0.0, watermark_mb - snapshot.context_constant_mb)


def measured_resident_footprint_mb(
    snapshot: SchedulingSnapshot,
    model: str | None,
    baseline: Baseline,
) -> tuple[float | None, int]:
    """The margined measured at-rest charge (MB) and its observation count; ``(None, 0)`` without authority."""
    store = snapshot.services.footprint_store
    if store is None or model is None or baseline is None or model not in snapshot.models_with_results:
        return (None, 0)
    key = FootprintKey(
        model_baseline=str(baseline),
        resolution_bucket=None,
        platform=sys.platform,
        stage=FootprintStage.RESIDENT,
        checkpoint=model,
    )
    measured_mb = store.measured_estimate_mb(key)
    if measured_mb is None:
        return (None, 0)
    return (max(0.0, measured_mb - snapshot.context_constant_mb), store.observation_count(key))


def learned_sampling_peak_mb(
    snapshot: SchedulingSnapshot,
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
    *,
    static_seed_mb: float,
    stage: FootprintStage,
) -> float:
    """The static sampling-peak seed raised by the learned watermark, lowered only by a trusted measurement."""
    store = snapshot.services.footprint_store
    if store is None:
        return static_seed_mb
    key = sampling_footprint_key(job, baseline, stage=stage)
    if key is None:
        return static_seed_mb
    raised_mb = store.estimate_mb(key, static_seed_mb=static_seed_mb)
    measured_mb: float | None = None
    if job.model is not None and job.model in snapshot.models_with_results:
        raw_mb = store.measured_estimate_mb(key)
        if raw_mb is not None:
            measured_mb = max(0.0, raw_mb - snapshot.context_constant_mb)
    if measured_mb is None or measured_mb >= raised_mb:
        return raised_mb
    floor_mb = predict_job_weight_mb(job, baseline) or 0.0
    return min(raised_mb, max(measured_mb, floor_mb))


def candidate_weights_resident(snapshot: SchedulingSnapshot, model: str | None, process_id: int | None) -> bool:
    """Whether the model's weights already occupy VRAM on the slot, so a dispatch onto it materialises nothing."""
    if model is None or process_id is None:
        return False
    slot = snapshot.slots.get(process_id)
    return slot is not None and model in slot.resident_weight_models


def candidate_delta_mb(
    snapshot: SchedulingSnapshot,
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
    *,
    process_id: int | None,
    disaggregated: bool,
) -> float | None:
    """The job's marginal predicted VRAM (MB) for the measured overlay, net of resident-weight credit."""
    static_gross_mb = (
        predict_job_sampler_only_vram_mb(job, baseline)
        if disaggregated
        else predict_job_sampling_vram_mb(job, baseline)
    )
    if static_gross_mb is None:
        return None
    gross_mb = learned_sampling_peak_mb(
        snapshot,
        job,
        baseline,
        static_seed_mb=static_gross_mb,
        stage=FootprintStage.SAMPLE_ISOLATED if disaggregated else FootprintStage.SAMPLE,
    )
    resident_credit_mb = 0.0
    if candidate_weights_resident(snapshot, job.model, process_id):
        resident_credit_mb = predict_job_weight_mb(job, baseline) or 0.0
    return max(0.0, gross_mb - resident_credit_mb)


# ---- the streaming forecast and the co-resident maximum


def whole_card_pinned(
    snapshot: SchedulingSnapshot, model: str | None, baseline: Baseline, device_index: int | None
) -> bool:
    """Whether the operator pinned the model (by name or baseline id) as whole-card on the card."""
    if model is None:
        return False
    pins = snapshot.config_for(device_index).whole_card_models
    if not pins:
        return False
    baseline_id = str(baseline) if baseline is not None else None
    return model in pins or (baseline_id is not None and baseline_id in pins)


def size_tier(snapshot: SchedulingSnapshot, model: str | None) -> ModelSizeTier:
    """The model's size tier from the shared classifier."""
    baseline = snapshot.services.model_metadata.get_baseline(model) if model is not None else None
    baseline_value = baseline.value if isinstance(baseline, KNOWN_IMAGE_GENERATION_BASELINE) else baseline
    return model_size_tier(model, baseline_value)


def disaggregation_sibling_charge_mb(
    snapshot: SchedulingSnapshot,
    job: ImageGenerateJobPopResponse,
    baseline: Baseline,
    device_index: int | None,
) -> float:
    """The image lane's concurrent decode spike (MB) to charge when disaggregating, else the lane quota."""
    decode_spike_mb = predict_job_decode_spike_mb(job, str(baseline) if baseline is not None else None)
    if decode_spike_mb is not None:
        return decode_spike_mb
    return effective_post_process_vram_quota_mb(snapshot.card(device_index).total_vram_mb)


def unpausable_tenancy_mb(snapshot: SchedulingSnapshot, device_index: int | None) -> float:
    """Device tenancy (MB) on the card that no sole-residency teardown returns."""
    tenancy_mb = max(0.0, snapshot.card(device_index).foreign_floor_mb or 0.0)
    config = snapshot.config_for(device_index)
    if bool(config.starved_head_lane_reclaim) and bool(config.starved_head_utilities_pause):
        return tenancy_mb
    marginal_mb = marginal_or_seed_mb(snapshot, device_index)
    for slot in card_slots(snapshot, device_index):
        if slot.process_type is HordeProcessType.UTILITIES:
            tenancy_mb += max(0.0, float(slot.reserved_mb or 0)) + marginal_mb
    return tenancy_mb


def forecast_streaming(
    snapshot: SchedulingSnapshot,
    job: ImageGenerateJobPopResponse,
    baseline: Baseline,
    *,
    device_index: int | None = None,
) -> StreamForecast:
    """The weight-streaming forecast for loading the job's model against the card's measured state."""
    card = snapshot.card(device_index)
    configured_floor = snapshot.config.vram_reserve_mb
    floor_mb = (
        float(configured_floor)
        if isinstance(configured_floor, (int, float)) and not isinstance(configured_floor, bool)
        else 0.0
    )
    slots = card_slots(snapshot, device_index)
    safety_on_gpu = snapshot.safety_on_gpu_permitted and not snapshot.lifecycle.safety_gpu_paused
    num_safety_contexts = (
        sum(1 for slot in slots if slot.process_type is HordeProcessType.SAFETY) if safety_on_gpu else 0
    )
    num_post_process_contexts = (
        0
        if snapshot.lifecycle.post_process_gpu_paused
        else sum(1 for slot in slots if slot.process_type is HordeProcessType.POST_PROCESS)
    )
    pinned = whole_card_pinned(snapshot, job.model, baseline, device_index)
    wants_whole_card = pinned or size_tier(snapshot, job.model) >= ModelSizeTier.EXTRA_LARGE
    job_id = str(job.id_) if job.id_ is not None else None
    queued = snapshot.queue.jobs.get(job_id) if job_id is not None else None
    disaggregated = queued.disaggregation_class_eligible if queued is not None else False
    measured_mb, observation_count = measured_resident_footprint_mb(snapshot, job.model, baseline)
    return forecast_weight_streaming(
        job,
        str(baseline) if baseline is not None else None,
        free_now_mb=card.measured_free_mb,
        total_vram_mb=card.total_vram_mb,
        per_process_overhead_mb=per_process_overhead_mb(snapshot, device_index),
        num_inference_processes=loaded_inference_count(snapshot, device_index),
        configured_reserve_floor_mb=floor_mb,
        num_extra_resident_contexts=num_post_process_contexts,
        safety_context_charge_mb=num_safety_contexts * snapshot.safety_footprint_mb,
        learned_resident_footprint_mb=learned_resident_footprint_mb(snapshot, job.model, baseline),
        measured_resident_footprint_mb=measured_mb,
        measured_observation_count=observation_count,
        committed_reserve_mb=snapshot.committed_vram_reserve_mb,
        marginal_process_overhead_mb=marginal_process_overhead_mb(snapshot, device_index),
        wants_whole_card=wants_whole_card,
        disaggregated=disaggregated,
        disaggregation_sibling_charge_mb=(
            disaggregation_sibling_charge_mb(snapshot, job, baseline, device_index) if disaggregated else 0.0
        ),
        admission_noise_mb=admission_noise_mb(snapshot, device_index, card.total_vram_mb),
        whole_card_pinned=pinned,
        unpausable_tenancy_mb=unpausable_tenancy_mb(snapshot, device_index),
    )


def max_coresident_for_peak_mb(
    snapshot: SchedulingSnapshot,
    peak_mb: float,
    reserve_mb: float,
    *,
    device_index: int | None = None,
) -> int | None:
    """The largest live inference-process count that still fits the peak plus the reserve on the card."""
    return max_coresident_for_peak(
        total_vram_mb=snapshot.card(device_index).total_vram_mb,
        per_process_overhead_mb=per_process_overhead_mb(snapshot, device_index),
        marginal_overhead_mb=marginal_process_overhead_mb(snapshot, device_index),
        peak_mb=peak_mb,
        reserve_mb=reserve_mb,
    )


# ---- what the card could give back


def coresident_lookahead_affordable(
    snapshot: SchedulingSnapshot, resident_model: str, *, device_index: int | None
) -> bool:
    """Whether an idle resident copy of a queued model can coexist with the imminent head's sampling.

    Unknown figures keep the protection: the gate only ever removes protection on evidence.
    """
    total_vram_mb = snapshot.card(device_index).total_vram_mb
    if total_vram_mb is None:
        return True
    head_id = snapshot.queue.pending_in_pop_order[0] if snapshot.queue.pending_in_pop_order else None
    head_job = snapshot.queue.payloads.get(head_id) if head_id is not None else None
    if head_job is None or head_job.model is None:
        return True
    metadata = snapshot.services.model_metadata
    head_peak_mb = predict_job_sampling_vram_mb(head_job, metadata.get_baseline(head_job.model))
    resident_footprint_mb = predict_job_footprint_mb(head_job, metadata.get_baseline(resident_model))
    if head_peak_mb is None or resident_footprint_mb is None:
        return True
    return total_vram_mb - snapshot.vram_reserve_mb - head_peak_mb - resident_footprint_mb >= 0


def has_reclaimable_idle_model(
    snapshot: SchedulingSnapshot,
    target_process_id: int,
    *,
    for_head_of_queue: bool,
    device_index: int | None,
    make_room_for_model: str | None = None,
) -> bool:
    """Whether an idle resident model could be evicted on the card to reclaim VRAM for a head.

    A read-only mirror of the pressure eviction's targeting: an idle post-processing lane not already
    unloading, or an inference slot holding a model that is not in progress, not spared by the lookahead or
    whole-card guards (which the head escalation overrides), and not already unloading. The head's own target
    slot is excluded on the terms :func:`target_slot_is_spared` sets.
    """
    running = in_progress_models(snapshot)
    lookahead = next_models(snapshot, snapshot.max_inference_processes)
    for slot in card_slots(snapshot, device_index):
        if slot.process_id == target_process_id and target_slot_is_spared(
            slot, make_room_for_model=make_room_for_model
        ):
            continue
        if slot.process_type is HordeProcessType.POST_PROCESS:
            if slot.is_busy or slot.last_control_flag is HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                continue
            return True
        if slot.process_type is not HordeProcessType.INFERENCE or slot.model is None:
            continue
        if slot.model in running:
            continue
        if (
            slot.model in lookahead
            and not for_head_of_queue
            and coresident_lookahead_affordable(snapshot, slot.model, device_index=device_index)
        ):
            continue
        if not for_head_of_queue and slot.model in snapshot.whole_card_held_models:
            continue
        if slot.last_control_flag is HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
            continue
        return True
    return False


def teardownable_idle_contexts(
    snapshot: SchedulingSnapshot, head_process_id: int, *, device_index: int | None
) -> tuple[SlotSnapshot, ...]:
    """The idle inference slots on the card, other than the head's own, not serving an in-progress model."""
    running = in_progress_models(snapshot)
    return tuple(
        slot
        for slot in card_slots(snapshot, device_index)
        if slot.process_type is HordeProcessType.INFERENCE
        and slot.process_id != head_process_id
        and not slot.is_busy
        and not (slot.model is not None and slot.model in running)
    )


def has_teardownable_idle_context(
    snapshot: SchedulingSnapshot, head_process_id: int, *, device_index: int | None
) -> bool:
    """Whether an idle sibling inference context could be torn down to reclaim VRAM for a starved head."""
    return bool(teardownable_idle_contexts(snapshot, head_process_id, device_index=device_index))


def teardownable_idle_context_returns_mb(
    snapshot: SchedulingSnapshot,
    head_process_id: int,
    *,
    device_index: int | None,
) -> list[float]:
    """What each bare idle sibling context on the card would return (MB): its reservation plus a context."""
    marginal_mb = marginal_or_seed_mb(snapshot, device_index)
    return [
        max(0.0, float(slot.reserved_mb or 0)) + marginal_mb
        for slot in teardownable_idle_contexts(snapshot, head_process_id, device_index=device_index)
        if slot.model is None and slot.held_component_count == 0
    ]


def has_reclaimable_idle_tenancy(
    snapshot: SchedulingSnapshot,
    head_model: str | None,
    target_process_id: int,
    *,
    device_index: int | None,
) -> bool:
    """Whether a lane holds warm components or a parked preload the head could ask the card back from."""
    for slot in card_slots(snapshot, device_index):
        if slot.process_type is not HordeProcessType.INFERENCE or slot.process_id == target_process_id:
            continue
        if head_model is not None and slot.model == head_model:
            continue
        if slot.held_component_count > 0 or slot.parked_preload:
            return True
    return False


def apply_measured_context_teardown(
    snapshot: SchedulingSnapshot,
    request: VramRequest,
    arbiter: VramArbiter,
    head_process_id: int,
    *,
    structural_max_resident: int | None,
    device_index: int | None,
) -> tuple[VramRequest, int | None]:
    """Widen a head's request with the idle-context teardown when the measured deficit is closable by it.

    The structural count knows nothing of the noise buffer, the lanes or foreign VRAM; the measured deficit the
    arbiter refuses on knows all of them. When bare idle sibling contexts would close that deficit, the request
    is marked teardownable and the depth sized to exactly the contexts needed, never below one live process.
    """
    from dataclasses import replace

    if not request.is_head_of_queue or request.idle_contexts_teardownable or request.has_reclaimable_idle_model:
        return request, structural_max_resident
    deficit_mb = arbiter.measured_deficit_mb(request)
    if deficit_mb is None or deficit_mb <= 0.0:
        return request, structural_max_resident
    marginal_mb = marginal_or_seed_mb(snapshot, device_index)
    returns = teardownable_idle_context_returns_mb(snapshot, head_process_id, device_index=device_index)
    if not returns:
        return request, structural_max_resident
    covered = 0.0
    needed = 0
    for returned_mb in sorted(returns, reverse=True):
        needed += 1
        covered += max(returned_mb, marginal_mb)
        if covered >= deficit_mb:
            break
    if covered < deficit_mb:
        return request, structural_max_resident
    live = loaded_inference_count(snapshot, device_index)
    measured_target = max(1, live - needed)
    if measured_target >= live:
        return request, structural_max_resident
    max_resident = (
        measured_target if structural_max_resident is None else min(structural_max_resident, measured_target)
    )
    return replace(request, idle_contexts_teardownable=True), max_resident
