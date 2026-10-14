"""Pure VRAM pricing over a scheduling snapshot: what a job costs a card and what the card could give back.

Each function here is the snapshot form of a scheduler pricing method and gives the same answer over the same
worker; the differential tests hold the two together while both exist. Nothing here reads a collaborator or
records anything.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from hordelib.metrics import JobPhaseMetrics

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.models.model_sizing import ModelSizeTier, model_size_tier
from horde_worker_regen.process_management.resources.admission_identity import admission_margin_mb
from horde_worker_regen.process_management.resources.resource_budget import (
    _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB,
    StreamForecast,
    _job_feature_kinds,
    baseline_burden_entry,
    effective_inference_reserve_mb,
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
    LearnedFootprintStore,
    ResolutionBucket,
    job_class_key,
    sampling_footprint_key,
)
from horde_worker_regen.process_management.scheduling.admission.snapshot import SchedulingSnapshot, SlotSnapshot
from horde_worker_regen.process_management.scheduling.clearance_lease import CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
from horde_worker_regen.process_management.scheduling.concurrent_overlap import prices_sampler_only_beside
from horde_worker_regen.process_management.scheduling.diagnostic_throttle import DIAGNOSTIC_MB_BUCKET
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    DominanceRung,
    max_coresident_for_peak,
)
from horde_worker_regen.utils.config_coercion import config_number
from horde_worker_regen.utils.vram_quota import effective_post_process_vram_quota_mb

type Baseline = KNOWN_IMAGE_GENERATION_BASELINE | str | None

STAGING_ENCODE_ALLOWANCE_MB = 348.0
"""VRAM (MB) a staged job's encode allocates beyond its family's text-encoder and VAE weights: the latents and
the conditioning working set the encode holds before the child acquires the sampling lease. Sized so SDXL,
whose support weights are 1700 MB, stages at the 2048 MB its dual CLIP encode was measured near."""

_STAGING_FLOOR_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
"""The family whose staging charge is every family's floor, and the charge of a job whose family is unknown.

A family with lighter support weights still allocates the encode working set, so pricing it below SDXL's
measured figure would admit staging into room the encode then overruns."""


def staging_charge_mb(baseline: Baseline) -> float:
    """Return the VRAM (MB) a staged (dispatched, not yet cleared) job of the family charges the card.

    Under the clearance lease the diffusion weights load inside the leased sample call, so a staged job holds
    only its encode until clearance: the family's support weights (hordelib's ``vram_support_weights_mb``) plus
    :data:`STAGING_ENCODE_ALLOWANCE_MB`, floored at the SDXL figure. A heavy text encoder (Flux, Qwen,
    Z-Image) therefore stages at its own size, where one flat SDXL-sized figure admitted encodes the card
    could not hold.

    Args:
        baseline: The job's model family; None or a family hordelib has no seed for is charged the floor.

    Returns:
        The staging charge (MB).
    """
    floor_mb = _family_support_weights_mb(_STAGING_FLOOR_BASELINE) + STAGING_ENCODE_ALLOWANCE_MB
    if baseline is None:
        return floor_mb
    return max(floor_mb, _family_support_weights_mb(baseline) + STAGING_ENCODE_ALLOWANCE_MB)


def _family_support_weights_mb(baseline: KNOWN_IMAGE_GENERATION_BASELINE | str) -> float:
    """Return the family's text-encoder and VAE weights (MB) from hordelib's seed, or 0 when it has none."""
    burden = baseline_burden_entry(str(baseline), None)
    return float(burden.vram_support_weights_mb) if burden is not None else 0.0


def dispatch_staging_charge_mb(*, clearance_lease_active: bool, baseline: Baseline) -> float | None:
    """Return what a dispatch puts on the card when it only stages the job, or None when it is the VRAM moment.

    Under the clearance lease the weights land at clearance, which prices the full materialisation, so the
    dispatch charges the encode working set alone (:func:`staging_charge_mb`). The residency gate charges this
    with no duration term: its head-protection arithmetic is stated against it, and the overlap gate, which
    governs a dispatch beside a running sampler, adds the duration term through
    :func:`overlap_staging_charge_mb`.
    """
    return staging_charge_mb(baseline) if clearance_lease_active else None


def overlap_staging_charge_mb(
    *,
    clearance_lease_active: bool,
    baseline: Baseline,
    running_remaining_sampling_seconds: Sequence[float | None],
    incoming_load_seconds: float | None,
) -> float | None:
    """Return the staging charge for a dispatch beside running samplers, or None to price the full job.

    A staged child waits for clearance at most :data:`CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS` and then samples
    without a grant, unpriced. The staging charge therefore holds only while clearance is expected well inside
    that bound: every running sampler's remaining seconds is known, and the longest of them plus the card's
    measured weight-load seconds fits the timeout twice over, so the estimate may be wrong by its own size. An
    unknown remaining time or an unmeasured load keeps the full price.
    """
    dispatch_charge_mb = dispatch_staging_charge_mb(clearance_lease_active=clearance_lease_active, baseline=baseline)
    if dispatch_charge_mb is None or incoming_load_seconds is None:
        return None
    known_remaining_seconds = [seconds for seconds in running_remaining_sampling_seconds if seconds is not None]
    if len(known_remaining_seconds) != len(running_remaining_sampling_seconds):
        return None
    expected_wait_seconds = max(known_remaining_seconds, default=0.0) + incoming_load_seconds
    margin_seconds = expected_wait_seconds
    if expected_wait_seconds + margin_seconds >= CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS:
        return None
    return dispatch_charge_mb


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
    measured_mb = store.measured_estimate_net_of_context_mb(key)
    if measured_mb is None:
        return (None, 0)
    return (measured_mb, store.observation_count(key))


def learned_sampling_peak_mb(
    snapshot: SchedulingSnapshot,
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
    *,
    static_seed_mb: float,
    stage: FootprintStage,
) -> float:
    """The static sampling-peak seed raised by the learned watermark, lowered by the key's trusted measurement.

    Never lowered below the job's own core weights, which is what stops a light sibling's measurements in the
    same band from underpricing a heavy checkpoint. A LoRA job's measured figure carries the LoRA delta the seed
    charges (:func:`lora_feature_delta_mb`): the key has no LoRA axis, so its measurement is the maximum over
    mixed jobs and cannot see the patch transient of a LoRA job that faulted before reporting it.
    """
    store = snapshot.services.footprint_store
    if store is None:
        return static_seed_mb
    key = sampling_footprint_key(job, baseline, stage=stage)
    if key is None:
        return static_seed_mb
    raised_mb = store.estimate_mb(key, static_seed_mb=static_seed_mb)
    measured_mb: float | None = None
    if job.model is not None:
        net_mb = store.measured_job_estimate_net_of_context_mb(key, job_class_key(key, job))
        if net_mb is not None:
            measured_mb = net_mb + lora_feature_delta_mb(job, baseline)
    if measured_mb is None or measured_mb >= raised_mb:
        return raised_mb
    floor_mb = predict_job_weight_mb(job, baseline) or 0.0
    return min(raised_mb, max(measured_mb, floor_mb))


def lora_feature_delta_mb(job: ImageGenerateJobPopResponse, baseline: str | None) -> float:
    """The sampling-phase VRAM (MB) the static seed charges for the job's LoRAs, zero for a job without any.

    Read as hordelib's burden estimate with the LoRA feature less the same estimate without it, so a
    per-baseline or per-megapixel term in the feature table carries over unchanged.
    """
    from hordelib.feature_impact import FEATURE_KIND, estimate_job_burden

    if FEATURE_KIND.lora not in _job_feature_kinds(job):
        return 0.0
    baseline_name = baseline if baseline is not None else ""
    batch = max(1, job.payload.n_iter)
    with_lora = estimate_job_burden(
        baseline=baseline_name,
        width=job.payload.width,
        height=job.payload.height,
        batch=batch,
        features=[FEATURE_KIND.lora],
    )
    without_lora = estimate_job_burden(
        baseline=baseline_name,
        width=job.payload.width,
        height=job.payload.height,
        batch=batch,
        features=[],
    )
    return float(max(0, with_lora.vram_sampling_mb - without_lora.vram_sampling_mb))


def candidate_weights_resident(snapshot: SchedulingSnapshot, model: str | None, process_id: int | None) -> bool:
    """Whether the model's weights already occupy VRAM on the slot, so a dispatch onto it materialises nothing."""
    if model is None or process_id is None:
        return False
    slot = snapshot.slots.get(process_id)
    return slot is not None and model in slot.resident_weight_models


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


def prices_sampler_only_on_card(
    snapshot: SchedulingSnapshot,
    job: ImageGenerateJobPopResponse,
    device_index: int | None,
) -> bool:
    """Whether the job shares the card with a co-tenant that makes one side of the pairing extra-large.

    Such a job is priced at its sampler-only footprint (:func:`prices_sampler_only_beside`). On a multi-card
    worker with no card named there is no co-tenant to judge, so the whole-job price stands.
    """
    if snapshot.multi_gpu_routing_active and device_index is None:
        return False
    job_id = str(job.id_)
    co_tenant_tiers = [
        size_tier(snapshot, snapshot.queue.jobs[running_id].model)
        for running_id in active_jobs_on_card(snapshot, device_index)
        if running_id != job_id and running_id in snapshot.queue.jobs
    ]
    return prices_sampler_only_beside(size_tier(snapshot, job.model), co_tenant_tiers)


def sampling_peak_measured(
    snapshot: SchedulingSnapshot,
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
    *,
    process_id: int | None,
    disaggregated: bool,
) -> bool:
    """Whether the key :func:`candidate_delta_mb` prices this job under carries a trusted measured peak."""
    store = snapshot.services.footprint_store
    if store is None:
        return False
    slot = snapshot.slots.get(process_id) if process_id is not None else None
    device_index = snapshot.routing_device_index(slot) if slot is not None else None
    sampler_only = disaggregated or prices_sampler_only_on_card(snapshot, job, device_index)
    key = sampling_footprint_key(
        job, baseline, stage=FootprintStage.SAMPLE_ISOLATED if sampler_only else FootprintStage.SAMPLE
    )
    return key is not None and store.measured_estimate_mb(key) is not None


def candidate_delta_mb(
    snapshot: SchedulingSnapshot,
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
    *,
    process_id: int | None,
    disaggregated: bool,
) -> float | None:
    """The job's marginal predicted VRAM (MB) for the measured overlay, net of resident-weight credit.

    Sampler-only for a disaggregation-class job, and for a job sharing its card with an extra-large co-tenant
    (:func:`prices_sampler_only_on_card`); the whole-job sampling peak otherwise.
    """
    slot = snapshot.slots.get(process_id) if process_id is not None else None
    device_index = snapshot.routing_device_index(slot) if slot is not None else None
    sampler_only = disaggregated or prices_sampler_only_on_card(snapshot, job, device_index)
    static_gross_mb = (
        predict_job_sampler_only_vram_mb(job, baseline)
        if sampler_only
        else predict_job_sampling_vram_mb(job, baseline)
    )
    if static_gross_mb is None:
        return None
    gross_mb = learned_sampling_peak_mb(
        snapshot,
        job,
        baseline,
        static_seed_mb=static_gross_mb,
        stage=FootprintStage.SAMPLE_ISOLATED if sampler_only else FootprintStage.SAMPLE,
    )
    resident_credit_mb = 0.0
    if candidate_weights_resident(snapshot, job.model, process_id):
        resident_credit_mb = predict_job_weight_mb(job, baseline) or 0.0
    return max(0.0, gross_mb - resident_credit_mb)


# ---- the partial-load seat

PARTIAL_SEAT_FALLBACK_WEIGHT_FRACTION = 0.5
"""The share of an extra-large checkpoint's weights a partial-load seat must hold while no step time or upload
rate is measured, and the most the derived share may ask.

It is also the ceiling of the derived share. A seat is only priced on a card that has nothing left to reclaim,
where a refusal leaves the child to sample at its lease-acquire timeout with whatever share fits anyway, so a
larger share would only delay the same load by the timeout."""

PARTIAL_SEAT_MIN_WEIGHT_FRACTION = 0.25
"""The smallest share of the weights a seat may hold however fast the bus, so a sample never streams most of
its model across it on every step."""

_PLAUSIBLE_UPLOAD_MB_PER_SECOND = (256.0, 65536.0)
"""The range (MB/s) a measured weight-upload rate must fall in to size a seat.

The top is a PCIe 5.0 x16 link. A rate below the bottom is a load that waited on disk, which says nothing
about the bus a partial load streams its offloaded weights across."""


def job_upload_mb_per_second(phase_metrics: JobPhaseMetrics) -> float | None:
    """Return the rate (MB/s) a finished job put its weights on the card, or None when it uploaded nothing.

    The weights the job held at its peak over the seconds its device loads took. The loads include the text
    encoder's and reading the checkpoint's pages back from the host cache, so the figure is below the bus's
    raw rate, which keeps the share of weights a seat derives from it on the cautious side.
    """
    footprint = phase_metrics.vram_footprint
    if footprint is None or footprint.peak_resident_weights_mb is None or footprint.peak_resident_weights_mb <= 0:
        return None
    upload_seconds = sum(
        model_load.duration_seconds for model_load in phase_metrics.model_loads if model_load.phase == "ram_to_vram"
    )
    if upload_seconds <= 0.0:
        return None
    return footprint.peak_resident_weights_mb / upload_seconds


def partial_seat_weight_fraction(
    *,
    weights_mb: float | None,
    seconds_per_step: float | None,
    upload_mb_per_second: float | None,
) -> float:
    """Return the share of a checkpoint's weights its partial-load seat must hold.

    The weights left off the card cross the bus on every step, so the share is the one whose offloaded part the
    measured upload rate moves within one step's time: ``1 - seconds_per_step * upload_mb_per_second /
    weights_mb``, clamped to :data:`PARTIAL_SEAT_MIN_WEIGHT_FRACTION` and
    :data:`PARTIAL_SEAT_FALLBACK_WEIGHT_FRACTION`. A missing or non-positive figure, or an upload rate outside
    :data:`_PLAUSIBLE_UPLOAD_MB_PER_SECOND`, prices the fallback share.
    """
    low_rate_mb, high_rate_mb = _PLAUSIBLE_UPLOAD_MB_PER_SECOND
    if (
        weights_mb is None
        or weights_mb <= 0.0
        or seconds_per_step is None
        or seconds_per_step <= 0.0
        or upload_mb_per_second is None
        or not low_rate_mb <= upload_mb_per_second <= high_rate_mb
    ):
        return PARTIAL_SEAT_FALLBACK_WEIGHT_FRACTION
    streamed_share = seconds_per_step * upload_mb_per_second / weights_mb
    return min(PARTIAL_SEAT_FALLBACK_WEIGHT_FRACTION, max(PARTIAL_SEAT_MIN_WEIGHT_FRACTION, 1.0 - streamed_share))


def partial_seat_activation_mb(
    snapshot: SchedulingSnapshot,
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
) -> float | None:
    """Return the non-weight memory (MB) a whole job needs, or None when its weights or peak cannot be predicted.

    The figure is the measured activation of the job's class, else of its pooled band, once that window holds
    enough jobs (:meth:`LearnedFootprintStore.measured_job_estimate_net_of_context_mb`). Until then it is the
    static seed, the sampler-only peak less the weights. The measured figure carries the VAE and the encoded
    conditioning the seed leaves out, and it may fall below the seed.

    The key's raise-only watermark is never read here. A grant carries this figure as room the child's load must
    leave free, so one reading that also held streamed weights or another model's cache would hold that room
    free on every later grant and force partial loads.
    """
    return _partial_seat_activation_from_store_mb(snapshot.services.footprint_store, job, baseline)


def _partial_seat_activation_from_store_mb(
    store: LearnedFootprintStore | None,
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
) -> float | None:
    """Return :func:`partial_seat_activation_mb` priced from ``store`` rather than a snapshot's."""
    weights_mb = predict_job_weight_mb(job, baseline)
    sampler_only_mb = predict_job_sampler_only_vram_mb(job, baseline)
    if weights_mb is None or sampler_only_mb is None:
        return None
    static_seed_mb = max(0.0, sampler_only_mb - weights_mb)
    key = sampling_footprint_key(job, baseline, stage=FootprintStage.SAMPLE_ACTIVATION)
    if store is None or key is None:
        return static_seed_mb
    measured_mb = store.measured_job_estimate_net_of_context_mb(key, job_class_key(key, job))
    return static_seed_mb if measured_mb is None else measured_mb


def _sampling_non_weight_mb(
    snapshot: SchedulingSnapshot, activation_mb: float | None, *, device_index: int | None
) -> float:
    """Return the larger of the job's non-weight memory and ComfyUI's inference reserve (MB), or the reserve alone."""
    return _non_weight_on_card_mb(snapshot.card(device_index).total_vram_mb, activation_mb)


def _non_weight_on_card_mb(total_vram_mb: float | None, activation_mb: float | None) -> float:
    """Return :func:`_sampling_non_weight_mb` for a card of ``total_vram_mb``."""
    comfy_reserve_mb = effective_inference_reserve_mb(total_vram_mb, 0.0)
    if activation_mb is None:
        return comfy_reserve_mb
    return max(activation_mb, comfy_reserve_mb)


def sampling_reserve_mb(
    snapshot: SchedulingSnapshot,
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
    *,
    device_index: int | None,
) -> float | None:
    """Return the non-weight memory (MB) a job's sampling window is priced with, or None when there is none.

    The term :func:`partial_seat_mb` adds to the weights it seats is the job's activation
    (:func:`partial_seat_activation_mb`) raised to ComfyUI's inference reserve, or the reserve alone when the
    job's activation cannot be predicted. A clearance grant carries it to the child, whose partial load would
    otherwise leave only ComfyUI's activation estimate. None when the card's total is unknown and nothing is
    predicted.
    """
    non_weight_mb = _sampling_non_weight_mb(
        snapshot, partial_seat_activation_mb(snapshot, job, baseline), device_index=device_index
    )
    return non_weight_mb if non_weight_mb > 0.0 else None


def partial_seat_mb(
    snapshot: SchedulingSnapshot,
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
    *,
    device_index: int | None,
    weight_fraction: float,
) -> float | None:
    """Return the whole need (MB) of a partial load of the job's checkpoint, or None when the job does not qualify.

    ComfyUI serves an extra-large checkpoint the card cannot hold whole by loading a share of its weights. The
    seat is that share plus the larger of the job's non-weight memory and ComfyUI's inference reserve, since
    the reserve is the free memory ComfyUI keeps for those same activations. A LoRA job never qualifies, because
    its patch temporaries crash a child natively in this regime, and neither does any job on a card whose driver
    is paging the worker's allocations.
    """
    return partial_seat_on_card_mb(
        job,
        baseline,
        store=snapshot.services.footprint_store,
        total_vram_mb=snapshot.card(device_index).total_vram_mb,
        wddm_paging_active=snapshot.ledgers.retention.wddm_paging_active,
        model_tier=size_tier(snapshot, job.model),
        weight_fraction=weight_fraction,
    )


def partial_seat_on_card_mb(
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
    *,
    store: LearnedFootprintStore | None,
    total_vram_mb: float | None,
    wddm_paging_active: bool,
    model_tier: ModelSizeTier,
    weight_fraction: float,
) -> float | None:
    """Return :func:`partial_seat_mb` from explicit figures, for a caller that prices a seat outside a snapshot."""
    if wddm_paging_active:
        return None
    if model_tier < ModelSizeTier.EXTRA_LARGE:
        return None
    from hordelib.feature_impact import FEATURE_KIND

    if FEATURE_KIND.lora in _job_feature_kinds(job):
        return None
    weights_mb = predict_job_weight_mb(job, baseline)
    activation_mb = _partial_seat_activation_from_store_mb(store, job, baseline)
    if weights_mb is None or activation_mb is None:
        return None
    return weight_fraction * weights_mb + _non_weight_on_card_mb(total_vram_mb, activation_mb)


# ---- the streaming forecast and the co-resident maximum


def whole_card_pinned(
    snapshot: SchedulingSnapshot, model: str | None, baseline: Baseline, device_index: int | None
) -> bool:
    """Whether the operator pinned the model (by name or baseline id) as whole-card on the card."""
    return pinned_by(whole_card_pins(snapshot.config_for(device_index)), model, baseline)


def whole_card_pins(config: reGenBridgeData) -> frozenset[str]:
    """Return the operator's whole-card pins (model names and baseline ids) for a card's config."""
    pins = config.whole_card_models
    if not isinstance(pins, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(pin) for pin in pins)


def pinned_by(pins: frozenset[str], model: str | None, baseline: Baseline) -> bool:
    """Whether ``pins`` names the model or its baseline id."""
    if model is None or not pins:
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
    verdict = model_dominance(card.dominance, snapshot.services.footprint_store, job.model, baseline)
    wants_whole_card = pinned or (verdict is not None and verdict.cannot_co_reside)
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
        whole_card_tier=size_tier(snapshot, job.model) >= ModelSizeTier.EXTRA_LARGE,
        sampling_peak_observations=verdict.peak.observation_count if verdict is not None else 0,
    )


# ---- the dominance verdict


DOMINANCE_SEED_BAND = ResolutionBucket.LE_1024
"""The band a model with no observed band is priced at."""

DOMINANCE_HYSTERESIS_MB = DIAGNOSTIC_MB_BUCKET
"""How far (MB) a rung's slack must cross zero before a held rung changes, and the room a pairing with a dominant
side must show to co-sample. It is the width the recompute key buckets the floor by, so a figure moving inside
one bucket never moves a rung."""

_SEED_SIDE_PX_BY_BAND: dict[ResolutionBucket, int] = {
    ResolutionBucket.LE_512: 512,
    ResolutionBucket.LE_768: 768,
    ResolutionBucket.LE_1024: 1024,
    ResolutionBucket.GT_1024: 1536,
}
"""The square side (px) an unmeasured model's static seed is taken at for each band. It is the band's upper side,
or for the open top band the largest side a common job asks for."""


def seed_side_px(band: ResolutionBucket) -> int:
    """Return the square side (px) a representative job of ``band`` is priced at."""
    return _SEED_SIDE_PX_BY_BAND[band]


@dataclass(frozen=True)
class ModelSamplingPeak:
    """Represents the sampling peak (MB, net of context) the dominance verdict prices one model at."""

    model: str
    peak_mb: float | None
    """None when neither the store nor the static seed can price the model; such a model reads as fitting."""
    band: ResolutionBucket
    """The band the figure belongs to, which is the most-observed band with a measured figure, else the seed band."""
    observation_count: int
    """Observations behind a measured figure. Zero for the static seed."""


@dataclass(frozen=True)
class ModelDominance:
    """Represents one model's rung on a card, with the peak it was priced at."""

    peak: ModelSamplingPeak
    rung: DominanceRung

    @property
    def cannot_co_reside(self) -> bool:
        """Whether the model is on rung 2, the rung the teardown serves."""
        return self.rung is DominanceRung.CANNOT_CO_RESIDE


@dataclass(frozen=True)
class DominanceInputs:
    """Represents every input the dominance verdict depends on, compared to decide whether it must be recomputed.

    The floor enters bucketed, so its ordinary movement within a bucket never recomputes. The footprint evidence
    enters as the observation counts of the offered baselines' pooled sampling keys, which move exactly when a
    figure the verdict reads can.
    """

    offered_models: frozenset[str]
    whole_card_pins: frozenset[str]
    total_vram_mb: float
    first_context_mb: float
    marginal_context_mb: float
    admission_noise_mb: float
    foreign_floor_bucket: int | None
    sample_observation_counts: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class CardDominance:
    """Represents one card's dominance verdict over its offered set, computed once per change of its inputs.

    Rung 2 for model M: ``total - P(M) - first context - marginal context - floor - margin < 0``. Rung 1: rung 2,
    or no offered model B (M itself included) has ``total - P(M) - P(B) - first - marginal - floor - margin >= 0``.
    ``P`` is the measured sampling peak at the most-observed band, else the static seed. A model the previous
    verdict judged keeps its rung until a test's slack crosses zero by more than :data:`DOMINANCE_HYSTERESIS_MB`.
    """

    inputs: DominanceInputs
    foreign_floor_mb: float
    """The floor (MB) the verdict charged, read when it was computed."""
    verdicts: Mapping[str, ModelDominance]
    """Each offered model's rung."""
    seats_mb: Mapping[str, float | None] = field(default_factory=dict)
    """The partial-load seat (MB) of each offered rung-2 or pinned model, None where the model has none."""

    @property
    def _charges_mb(self) -> float:
        """Return what every pairing charges besides the peaks, which is both contexts, the floor and the margin."""
        return (
            self.inputs.first_context_mb
            + self.inputs.marginal_context_mb
            + self.foreign_floor_mb
            + self.inputs.admission_noise_mb
        )

    def pair_room_mb(self, peak_mb: float | None, partner_mb: float | None) -> float | None:
        """Return the room (MB) the card leaves two models sampling together, or None when either is unpriced."""
        if peak_mb is None or partner_mb is None:
            return None
        return self.inputs.total_vram_mb - peak_mb - partner_mb - self._charges_mb

    def judge(self, peak: ModelSamplingPeak) -> ModelDominance:
        """Return the rung of a model priced at ``peak`` against this card's offered set."""
        return ModelDominance(
            peak=peak,
            rung=dominance_rung(
                peak.peak_mb,
                partner_peaks_mb=[verdict.peak.peak_mb for verdict in self.verdicts.values()],
                total_vram_mb=self.inputs.total_vram_mb,
                charges_mb=self._charges_mb,
            ),
        )

    def describe(self) -> str:
        """Return the verdict on one line, the charges first and then each model's rung, band, figure and evidence."""
        rows = ", ".join(
            f"{model} {verdict.rung} ({verdict.peak.band}, "
            f"{'?' if verdict.peak.peak_mb is None else f'{verdict.peak.peak_mb:.0f}'}MB, "
            f"n={verdict.peak.observation_count})"
            for model, verdict in sorted(self.verdicts.items())
        )
        return (
            f"total {self.inputs.total_vram_mb:.0f}MB, contexts {self.inputs.first_context_mb:.0f}+"
            f"{self.inputs.marginal_context_mb:.0f}MB, floor {self.foreign_floor_mb:.0f}MB, margin "
            f"{self.inputs.admission_noise_mb:.0f}MB: {rows}"
        )


def _crossed(slack_mb: float, *, held: bool | None, band_mb: float) -> bool:
    """Return whether ``slack_mb`` says the rung's test is failed (no room), with hysteresis about a held answer.

    With no earlier answer the test is the bare sign of the slack. A rung already held is left only once the slack
    shows room by more than ``band_mb``. A rung not held is entered only once the slack is short by more than it.
    """
    if held is None:
        return slack_mb < 0.0
    if held:
        return slack_mb < band_mb
    return slack_mb < -band_mb


def dominance_rung(
    peak_mb: float | None,
    *,
    partner_peaks_mb: Iterable[float | None],
    total_vram_mb: float,
    charges_mb: float,
    previous: DominanceRung | None = None,
    band_mb: float = 0.0,
) -> DominanceRung:
    """Return the rung of a model of ``peak_mb`` among partners of ``partner_peaks_mb`` (its own copy included).

    ``charges_mb`` is everything a pairing charges besides the two peaks. An unpriced model has no rung, and an
    unpriced partner reads as fitting, so a missing figure never withholds a co-sample.

    ``previous`` is the rung the card's last verdict gave the model. Each rung's test then moves only when its
    slack crosses zero by more than ``band_mb``, so a measured peak drifting about a threshold does not flip the
    rung, and with it the teardown, on every observation.
    """
    if peak_mb is None:
        return DominanceRung.NONE
    held_reside = None if previous is None else previous is DominanceRung.CANNOT_CO_RESIDE
    if _crossed(total_vram_mb - peak_mb - charges_mb, held=held_reside, band_mb=band_mb):
        return DominanceRung.CANNOT_CO_RESIDE
    best_room_mb = max(
        (
            float("inf") if partner_mb is None else total_vram_mb - peak_mb - partner_mb - charges_mb
            for partner_mb in (*partner_peaks_mb, peak_mb)
        ),
    )
    held_sample = None if previous is None else previous is not DominanceRung.NONE
    if _crossed(best_room_mb, held=held_sample, band_mb=band_mb):
        return DominanceRung.CANNOT_CO_SAMPLE
    return DominanceRung.NONE


def decide_card_dominance(
    inputs: DominanceInputs,
    *,
    foreign_floor_mb: float,
    peaks: Sequence[ModelSamplingPeak],
    seats_mb: Mapping[str, float | None] | None = None,
    previous: CardDominance | None = None,
) -> CardDominance:
    """Return the card's dominance verdict for every model in ``peaks``, all of which are its partners.

    ``previous`` is the card's last verdict. A model it judged keeps its rung until the new figures cross the
    rung's threshold by more than :data:`DOMINANCE_HYSTERESIS_MB`.
    """
    charges_mb = (
        inputs.first_context_mb + inputs.marginal_context_mb + max(0.0, foreign_floor_mb) + inputs.admission_noise_mb
    )
    partner_peaks_mb = [peak.peak_mb for peak in peaks]
    previous_verdicts = previous.verdicts if previous is not None else {}
    verdicts = {}
    for peak in peaks:
        held = previous_verdicts.get(peak.model)
        verdicts[peak.model] = ModelDominance(
            peak=peak,
            rung=dominance_rung(
                peak.peak_mb,
                partner_peaks_mb=partner_peaks_mb,
                total_vram_mb=inputs.total_vram_mb,
                charges_mb=charges_mb,
                previous=held.rung if held is not None else None,
                band_mb=DOMINANCE_HYSTERESIS_MB,
            ),
        )
    return CardDominance(
        inputs=inputs,
        foreign_floor_mb=max(0.0, foreign_floor_mb),
        verdicts=verdicts,
        seats_mb=dict(seats_mb or {}),
    )


def pooled_sample_key(baseline: str, band: ResolutionBucket, *, platform: str = sys.platform) -> FootprintKey:
    """Return the pooled (all job classes) SAMPLE key for a baseline and band."""
    return FootprintKey(
        model_baseline=baseline, resolution_bucket=band, platform=platform, stage=FootprintStage.SAMPLE
    )


def representative_job(model: str, *, side_px: int) -> ImageGenerateJobPopResponse:
    """Return a plain batch-1 square text-to-image job for ``model``, priced where no real job exists yet.

    It carries no LoRA, controlnet, source image or post-processing, so every per-job figure read from it is the
    model's figure at that size.
    """
    return ImageGenerateJobPopResponse.model_validate(
        {
            "id": str(_REPRESENTATIVE_JOB_ID),
            "ids": [str(_REPRESENTATIVE_JOB_ID)],
            "model": model,
            "payload": {"width": side_px, "height": side_px, "n_iter": 1, "ddim_steps": 1},
            "skipped": {},
            "source_processing": "txt2img",
        },
    )


_REPRESENTATIVE_JOB_ID = uuid.UUID(int=0)
"""The id every representative job carries. No tracker ever holds it."""


def model_sampling_peak(
    store: LearnedFootprintStore | None,
    model: str,
    baseline: Baseline,
) -> ModelSamplingPeak:
    """Return the sampling peak the dominance verdict prices ``model`` at.

    The figure is the measured SAMPLE window, net of context, at the baseline's most-observed band among the bands
    with a measured figure. Without one, it is the static sampling seed of a plain batch-1 job at the most-observed
    band's upper side, or at :data:`DOMINANCE_SEED_BAND` when the store has seen no band of it. The raise-only
    watermark is never read, since it carries card-sized child peaks and would price a model off the card on one
    reading.
    """
    baseline_id = str(baseline) if baseline is not None else None
    seed_band = DOMINANCE_SEED_BAND
    if store is not None and baseline_id is not None:
        measured: tuple[ResolutionBucket, float, int] | None = None
        most_seen: tuple[ResolutionBucket, int] | None = None
        for band in ResolutionBucket:
            key = pooled_sample_key(baseline_id, band)
            count = store.observation_count(key)
            if count > 0 and (most_seen is None or count > most_seen[1]):
                most_seen = (band, count)
            figure_mb = store.measured_estimate_net_of_context_mb(key)
            if figure_mb is not None and (measured is None or count > measured[2]):
                measured = (band, figure_mb, count)
        if measured is not None:
            return ModelSamplingPeak(model=model, peak_mb=measured[1], band=measured[0], observation_count=measured[2])
        if most_seen is not None:
            seed_band = most_seen[0]
    seed_mb = predict_job_sampling_vram_mb(
        representative_job(model, side_px=seed_side_px(seed_band)),
        baseline_id,
    )
    return ModelSamplingPeak(model=model, peak_mb=seed_mb, band=seed_band, observation_count=0)


def model_dominance(
    dominance: CardDominance | None,
    store: LearnedFootprintStore | None,
    model: str | None,
    baseline: Baseline,
) -> ModelDominance | None:
    """Return a model's rung on a card, cached when the model is offered there and otherwise judged against the offer.

    None when the card has no verdict (its total is not yet known) or there is no model.
    """
    if dominance is None or model is None:
        return None
    cached = dominance.verdicts.get(model)
    if cached is not None:
        return cached
    return dominance.judge(model_sampling_peak(store, model, baseline))


def pair_co_samples(
    dominance: CardDominance | None,
    store: LearnedFootprintStore | None,
    *,
    candidate: tuple[str | None, Baseline],
    running: tuple[str | None, Baseline],
) -> bool:
    """Whether a candidate may sample beside a running job on a card, by the dominance arithmetic.

    A pairing with a side on rung 1 or above samples together only when ``total - P(A) - P(B) - contexts - floor -
    margin`` shows at least :data:`DOMINANCE_HYSTERESIS_MB` of room. That is the verdict's rung-1 test and band, so
    a model the verdict holds on rung 1 never has a partner this admits. A pairing of two models with partners is
    left to the arbiter's memory verdict. An unpriced side, or a card with no verdict, admits.
    """
    if dominance is None:
        return True
    candidate_verdict = model_dominance(dominance, store, candidate[0], candidate[1])
    running_verdict = model_dominance(dominance, store, running[0], running[1])
    if candidate_verdict is None or running_verdict is None:
        return True
    if candidate_verdict.rung is DominanceRung.NONE and running_verdict.rung is DominanceRung.NONE:
        return True
    room_mb = dominance.pair_room_mb(candidate_verdict.peak.peak_mb, running_verdict.peak.peak_mb)
    return room_mb is None or room_mb >= DOMINANCE_HYSTERESIS_MB


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


def card_concurrency_cap(snapshot: SchedulingSnapshot, device_index: int | None) -> int:
    """How many inference jobs the card may run at once: its own cap, else the worker-wide ceiling."""
    card = snapshot.cards.get(device_index)
    if card is None or card.max_concurrent_inference is None:
        return snapshot.max_concurrent_inference_processes
    return card.max_concurrent_inference


def card_running_job_count(snapshot: SchedulingSnapshot, device_index: int | None) -> int:
    """How many in-progress jobs the card's inference lanes are running: the count dispatch holds to its cap.

    ``device_index`` is a routing key, so None counts the whole worker where routing is card-agnostic.
    """
    in_progress = frozenset(snapshot.queue.in_progress)
    return sum(
        1
        for slot in snapshot.slots.values()
        if slot.process_type is HordeProcessType.INFERENCE
        and snapshot.routing_device_index(slot) == device_index
        and slot.current_job_id in in_progress
    )


def cards_able_to_start_job(snapshot: SchedulingSnapshot) -> frozenset[int | None]:
    """The routing keys running fewer jobs than their sampling cap, so an idle lane there can start one this cycle.

    Each card on a multi-GPU host, else the one worker-wide key, None. Busy lanes are a different count. A lane
    loading or holding a freshly loaded model counts as busy while running nothing, so it keeps placement from
    stacking copies on its card, but it does not stop dispatch seating a job on the card's other lane.
    """
    return frozenset(
        device_index
        for device_index in snapshot.cards
        if (device_index is not None) is snapshot.multi_gpu_routing_active
        and card_running_job_count(snapshot, device_index) < card_concurrency_cap(snapshot, device_index)
    )


def ready_seat_process_ids(snapshot: SchedulingSnapshot, claimants: Iterable[str] | None = None) -> frozenset[int]:
    """The idle lanes a pending job could be dispatched onto this cycle with no load, by process id.

    A ready seat holds the model a pending job names, can take work, is not a pinned disaggregation sampler,
    and sits where that job may run and a job can start now (:func:`cards_able_to_start_job`), which on a
    multi-GPU host is a card eligible for the job and otherwise the worker. Dispatch runs after the preload pass
    in the same cycle and seats that job at once. Jobs awaiting auxiliary preparation are invisible to dispatch and
    claim no seat, and so are jobs a dispatch gate is holding with no reclaim able to release them (a room hold
    the permitted rungs cannot close, or the post-processing hold), which will not start this cycle whatever their
    lane holds. A job held on room the rungs can close keeps its seat, since the sweep that frees that room would
    otherwise take the copy the job is about to sample on and the job pays a reload. ``claimants`` restricts the
    jobs that may claim a seat, and None lets every pending job claim one.
    """
    startable_cards = cards_able_to_start_job(snapshot)
    in_progress = frozenset(snapshot.queue.in_progress)
    holds = snapshot.ledgers.dispatch_holds
    seat_cards_by_model: dict[str, set[int | None]] = {}
    job_ids = snapshot.queue.pending_in_placement_order if claimants is None else tuple(claimants)
    for job_id in job_ids:
        job = snapshot.queue.jobs[job_id]
        if job_id in in_progress or job.model is None or job.requires_aux_preparation:
            continue
        held_without_reclaim = job_id in holds.hold_since and job_id not in holds.closable_holds
        if held_without_reclaim or job_id in holds.pp_defer_holds:
            continue
        allowed: frozenset[int | None] = (
            frozenset(job.eligible_cards) if snapshot.multi_gpu_routing_active else frozenset({None})
        )
        seat_cards_by_model.setdefault(job.model, set()).update(allowed & startable_cards)
    return frozenset(
        slot.process_id
        for slot in snapshot.slots.values()
        if slot.process_type is HordeProcessType.INFERENCE
        and slot.model is not None
        and slot.can_accept_job
        and not slot.reserved_for_disaggregation
        and snapshot.routing_device_index(slot) in seat_cards_by_model.get(slot.model, ())
    )


def sweep_spares_idle_copy(
    snapshot: SchedulingSnapshot,
    slot: SlotSnapshot,
    *,
    target_process_id: int,
    for_head_of_queue: bool,
    device_index: int | None,
) -> bool:
    """Whether a VRAM sweep leaves an idle lane's resident copy where it is.

    The eviction (`unload_models_from_vram`) and its read-only mirror (:func:`has_reclaimable_idle_model`) both
    ask it, so the sweep's protection of queued models has one owner.

    A ready seat (:func:`ready_seat_process_ids`) is spared first. Dispatch runs later in the same cycle and
    starts its job there with no load. Unloading the copy frees room the reload takes back a tick later, and
    the job's start pays a load it did not need.
    The head escalation keeps only the seat the head itself (the first queued job not yet dispatched) claims, so
    the head's model is never unloaded from the lane about to receive it, while a follower's warm start yields to
    a starved head. A seat whose job a dispatch gate holds is claimed by no one, because that job will not start
    and a copy spared for it would hold back a staged job's room until the lease times it out. The lane the sweep
    is clearing for its beneficiary is not spared as a seat, since :func:`target_slot_is_spared` alone decides
    it.

    Any other copy of a model in the queue's lookahead is spared while it can share the card with the head's
    sampling (:func:`coresident_lookahead_affordable`). On a card that cannot hold both, keeping it forces
    demand-paging during the head's sampling, which costs more than the reload. The head escalation overrides
    this protection outright.
    """
    if slot.model is None:
        return False
    in_progress = frozenset(snapshot.queue.in_progress)
    head_id = next((job_id for job_id in snapshot.queue.pending_in_pop_order if job_id not in in_progress), None)
    claimants = None if not for_head_of_queue else (head_id,) if head_id is not None else ()
    if slot.process_id != target_process_id and slot.process_id in ready_seat_process_ids(snapshot, claimants):
        return True
    if for_head_of_queue:
        return False
    return slot.model in next_models(snapshot, snapshot.max_inference_processes) and coresident_lookahead_affordable(
        snapshot, slot.model, device_index=device_index
    )


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
    unloading, or an inference slot holding a model that is idle or parked on a stale preload, owns no
    dispatched job, is not in progress, not spared by :func:`sweep_spares_idle_copy` or the whole-card guard
    (which the head escalation overrides), and not already unloading. The head's own target slot is excluded
    on the terms :func:`target_slot_is_spared` sets.

    The busy-or-parked and job-ownership terms are the actuator's own (`unload_models_from_vram` skips a busy
    slot that is not a parked preload, and a slot that owns a dispatched job). A mirror that counted either as
    evictable would keep the arbiter's ladder non-empty with a rung that frees nothing, and the lane rungs
    behind it would never be offered.
    """
    running = in_progress_models(snapshot)
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
        if slot.is_busy and not slot.parked_preload:
            continue
        if slot.current_job_id is not None:
            continue
        if slot.model in running:
            continue
        if sweep_spares_idle_copy(
            snapshot,
            slot,
            target_process_id=target_process_id,
            for_head_of_queue=for_head_of_queue,
            device_index=device_index,
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
    """The idle inference slots on the card, other than the head's own, not serving an in-progress model.

    A slot that owns a dispatched job reads idle until its child reports the job. It is left out because the
    scale-down that performs the teardown never takes it, and counting it would price room no rung returns.
    """
    running = in_progress_models(snapshot)
    return tuple(
        slot
        for slot in card_slots(snapshot, device_index)
        if slot.process_type is HordeProcessType.INFERENCE
        and slot.process_id != head_process_id
        and not slot.is_busy
        and slot.current_job_id is None
        and not (slot.model is not None and slot.model in running)
    )


def has_teardownable_idle_context(
    snapshot: SchedulingSnapshot, head_process_id: int, *, device_index: int | None
) -> bool:
    """Whether an idle sibling inference context could be torn down to reclaim VRAM for a starved head."""
    return bool(teardownable_idle_contexts(snapshot, head_process_id, device_index=device_index))


def is_bare_context(*, model: str | None, held_component_count: int) -> bool:
    """Whether an idle inference slot is a bare context: no model and no held components.

    Only such a slot is a teardown candidate, since a slot holding a model or components has a cheaper rung
    that unloads or releases them. It is also the only idle slot whose context the admission room may count
    as returnable.
    """
    return model is None and held_component_count == 0


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
        if is_bare_context(model=slot.model, held_component_count=slot.held_component_count)
    ]


def has_reclaimable_idle_tenancy(
    snapshot: SchedulingSnapshot,
    head_model: str | None,
    target_process_id: int,
    *,
    device_index: int | None,
) -> bool:
    """Whether a lane holds warm components or a parked preload on the device that the head could ask back.

    Only tenancy with weights on the card counts (``resident_weight_models``). A lane whose held components and
    parked preload live in host RAM returns no VRAM when it is unloaded, so counting it would keep the card from
    reading converged-empty for room the card does not have: the waiter's partial-load seat and measured attempt
    would wait out the lease-acquire timeout and load the same way unpriced.
    """
    for slot in card_slots(snapshot, device_index):
        if slot.process_type is not HordeProcessType.INFERENCE or slot.process_id == target_process_id:
            continue
        if head_model is not None and slot.model == head_model:
            continue
        if (slot.held_component_count > 0 or slot.parked_preload) and slot.resident_weight_models:
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
