"""The preload gate ladder: whether one pending job may stage its model this cycle, and onto which slot.

The preload pass walks the pending queue and asks, per job, the questions that stand before the resource
budget: is the job serviceable at all, is its model quarantined, is a copy already resident where the job can
use it, is the host below its RAM floor, which slot may take the load without displacing work another job
needs, and is that slot held by an exclusive job, a growth hold, a model change or the per-card load
serialization gate. Each answer is a decision over the :class:`SchedulingSnapshot` and returns a
:class:`PreloadGatePlan`: the decision, the slot chosen, and the commands (a fault, a process cycle) the
executor runs. The budget step and the send follow an admitting plan on the scheduler.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass

from horde_worker_regen.process_management.ipc.messages import HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType, WorkerCapability
from horde_worker_regen.process_management.resources.resource_budget import (
    BudgetVerdict,
    StreamForecast,
    effective_inference_reserve_mb,
)
from horde_worker_regen.process_management.resources.vram_arbiter import VramArbiter, VramRequestKind
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.admission.commands import (
    FaultCause,
    FaultJob,
    ReplaceProcess,
    SchedulerCommand,
)
from horde_worker_regen.process_management.scheduling.admission.materialization import (
    ContextReduction,
    MaterializationRequest,
    build_materialization_request,
)
from horde_worker_regen.process_management.scheduling.admission.snapshot import (
    JobSnapshot,
    SchedulingSnapshot,
    SlotSnapshot,
)
from horde_worker_regen.process_management.scheduling.dispatch_affinity import head_aged_past_anti_starvation
from horde_worker_regen.process_management.scheduling.governance.preload_admission import (
    AdmissionDecision,
    PreloadSlotSnapshot,
    card_preload_order,
    compute_preload_disallowed_processes,
    preload_concurrency_blocked,
    select_head_room_process_id,
)


class PreloadPassControl(enum.Enum):
    """What one pending job's preload decision means for the rest of the scheduling pass."""

    NEXT_JOB = enum.auto()
    """This job needs nothing (or was faulted); consider the next pending job."""
    STOP_PASS = enum.auto()
    """A gate deferred or consumed this cycle (RAM floor, no slot, serialization, budget); stop the pass."""
    PRELOAD_SENT = enum.auto()
    """A preload was issued for this job; the pass is done and reports success."""


def pass_control_for(decision: AdmissionDecision) -> PreloadPassControl:
    """Map an admission decision onto the pass control it implies."""
    match decision:
        case AdmissionDecision.ADMIT | AdmissionDecision.PRESTAGE:
            return PreloadPassControl.PRELOAD_SENT
        case (
            AdmissionDecision.NEXT_JOB
            | AdmissionDecision.QUARANTINED
            | AdmissionDecision.UNSERVICEABLE
            | AdmissionDecision.ALREADY_LOADED
        ):
            return PreloadPassControl.NEXT_JOB
        case _:
            return PreloadPassControl.STOP_PASS


@dataclass(frozen=True)
class PreloadGatePlan:
    """One pending job's answer from the gate ladder.

    ``ADMIT`` means every gate passed and the budget step decides next; any other decision is final for this
    cycle. ``commands`` are the actions the decision requires before the pass continues. ``notice`` is the
    operator line an edge-triggered decision wants logged once, and ``records_admission`` is False for the one
    exit that never considered the job (an aux-gated job yielding to a sibling that can sample).
    """

    decision: AdmissionDecision
    job_id: str
    reason: str
    target_process_id: int | None
    commands: tuple[SchedulerCommand, ...] = ()
    is_head_blocker: bool = False
    disallowed_process_ids: frozenset[int] = frozenset()
    """The slots the preload may not displace, for the re-selection an admission that retires the target needs."""
    notice: str | None = None
    records_admission: bool = True

    @property
    def admits(self) -> bool:
        """Whether the gates passed and the budget step is next."""
        return self.decision is AdmissionDecision.ADMIT

    @property
    def pass_control(self) -> PreloadPassControl:
        """What the pass does after this plan, before any budget step."""
        return pass_control_for(self.decision)


def loaded_or_loading_models(snapshot: SchedulingSnapshot) -> frozenset[str | None]:
    """Every model a slot holds (None for an empty slot) plus every model the map records as loaded or loading.

    The empty-slot None is kept deliberately: the pass's fast exit compares this set with the pending models,
    and an empty slot keeps that comparison from short-circuiting the head's clocks.
    """
    models: set[str | None] = {slot.model for slot in snapshot.slots.values()}
    models.update(
        name
        for name, entry in snapshot.model_map.items()
        if entry.load_state.is_loaded() or entry.load_state is ModelLoadState.LOADING
    )
    return frozenset(models)


def pending_models(snapshot: SchedulingSnapshot) -> frozenset[str | None]:
    """The models the pending queue wants."""
    return frozenset(snapshot.queue.jobs[job_id].model for job_id in snapshot.queue.pending_in_placement_order)


def preload_head(snapshot: SchedulingSnapshot) -> str | None:
    """The first pending job that is neither in progress nor awaiting auxiliary preparation.

    An aux-unprepared job never anchors preload as the priority head: it holds no sampling reservation and
    nothing prices around it, so the first job eligible to hold capacity is the head that may escalate eviction.
    """
    return next(
        (
            job_id
            for job_id in snapshot.queue.pending_in_placement_order
            if job_id not in snapshot.queue.in_progress and not snapshot.queue.jobs[job_id].requires_aux_preparation
        ),
        None,
    )


def every_pending_model_accounted_for(snapshot: SchedulingSnapshot, loaded_models: frozenset[str | None]) -> bool:
    """Whether the pass can exit before considering any job: every pending model is resident where its job can use it.

    Set equality alone is not enough on a multi-GPU host, where a model can be loaded on a card that cannot
    serve the job that wants it, so each pending job must also find its copy on a card eligible for it.
    """
    return loaded_models == pending_models(snapshot) and all(
        model_loaded_for_job(snapshot, job_id, loaded_models) for job_id in snapshot.queue.pending_in_placement_order
    )


def model_loaded_for_job(snapshot: SchedulingSnapshot, job_id: str, loaded_models: frozenset[str | None]) -> bool:
    """Whether the job's model already counts as resident or loading for this job's routing.

    Single-GPU: the card-blind membership test. Multi-GPU: a copy only counts when it sits on a card eligible
    to serve this job; a model resident solely on a card that cannot serve it leaves the job needing its own
    copy, since dispatch will not seat it on the ineligible copy and counting that copy here would withhold
    the preload that gives it one. A job with no eligible card at all keeps the card-blind answer; the
    eligibility fault path owns that case. A map entry whose owning process is gone records no residency.
    """
    job = snapshot.queue.jobs[job_id]
    model = job.model
    if model is None:
        return False
    if not snapshot.multi_gpu_routing_active:
        return model in loaded_models
    allowed = job.eligible_cards
    if not allowed:
        return model in loaded_models
    if any(slot.model == model and slot.device_index in allowed for slot in snapshot.slots.values()):
        return True
    entry = snapshot.model_map.get(model)
    if entry is None:
        return False
    if not (entry.load_state.is_loaded() or entry.load_state is ModelLoadState.LOADING):
        return False
    owner = snapshot.slots.get(entry.process_id)
    return owner is not None and owner.device_index in allowed


def duplicate_copy_may_serve(snapshot: SchedulingSnapshot, job_id: str) -> bool:
    """Whether a second copy of the job's already-resident model may be preloaded onto another card.

    On one card a duplicate is pure waste. Across cards the single-copy rule inverts when every eligible copy
    is busy running other work: the queued job would otherwise wait a whole sampling window for weights an idle
    card could be given. A duplicate is considered only when the worker routes across cards, at least one
    eligible copy exists, every such copy is busy, and no copy is still loading (a load in flight is about to
    provide a serving copy). Whether it lands stays the rest of the ladder's decision.
    """
    job = snapshot.queue.jobs[job_id]
    if not snapshot.multi_gpu_routing_active or job.model is None:
        return False
    entry = snapshot.model_map.get(job.model)
    if entry is not None and entry.load_state is ModelLoadState.LOADING:
        return False
    allowed = job.eligible_cards
    if not allowed:
        return False
    copies = [slot for slot in snapshot.slots.values() if slot.model == job.model and slot.device_index in allowed]
    if not copies:
        return False
    return all(not copy.can_accept_job for copy in copies)


def first_available_inference_slot(
    snapshot: SchedulingSnapshot,
    *,
    disallowed: Iterable[int] = (),
    device_index: int | None = None,
) -> SlotSnapshot | None:
    """The first image-generation slot that can take a load, empty slots first, or None.

    Mirrors the process map's first-available selection: a pinned disaggregation sampler is never a target,
    an unoccupied idle slot is the cheapest, and otherwise any slot that can accept a job.
    """
    excluded = set(disallowed)
    capable = [slot for slot in snapshot.slots.values() if WorkerCapability.IMAGE_GEN in slot.capabilities]
    for slot in capable:
        if device_index is not None and slot.device_index != device_index:
            continue
        if slot.reserved_for_disaggregation or slot.process_id in excluded:
            continue
        if slot.state in (HordeProcessState.WAITING_FOR_JOB, HordeProcessState.PRELOADED_MODEL) and slot.model is None:
            return slot
    for slot in capable:
        if device_index is not None and slot.device_index != device_index:
            continue
        if slot.reserved_for_disaggregation or slot.process_id in excluded:
            continue
        if slot.can_accept_job:
            return slot
    return None


def head_is_starving(snapshot: SchedulingSnapshot, head: JobSnapshot) -> bool:
    """Whether the queued head has aged past the anti-starvation fraction of its ttl."""
    return head_aged_past_anti_starvation(
        now=snapshot.now,
        popped_at=head.popped_at,
        ttl=head.ttl,
        fallback_ttl=snapshot.recent_job_ttl,
    )


def slots_holding_the_head_model(snapshot: SchedulingSnapshot, job_id: str, disallowed: Iterable[int]) -> list[int]:
    """Slots holding the queue head's model that a preload for ``job_id`` must not take, else an empty list.

    A preload placed on the head's own warm slot displaces the copy the head is waiting to run: a priority
    inversion that is worst exactly when the head reaching its anti-starvation window is what pushed a later
    job's preload past it. A slot the head's copy is on is refused outright whenever another idle slot can
    take the preload; with no other slot free the refusal holds only while the head is starving, so a
    single-slot worker still swaps models between jobs. A preload for the head itself, or for the model the
    head needs, displaces nothing and is never refused. The head here is the queue's own first job, since the
    inversion is about the job the horde is waiting on.
    """
    head_id = next(iter(snapshot.queue.pending_in_pop_order), None)
    if head_id is None:
        return []
    head = snapshot.queue.jobs[head_id]
    job = snapshot.queue.jobs[job_id]
    if head.model is None or head_id == job_id or head.model == job.model:
        return []
    excluded = set(disallowed)
    holders = [
        slot.process_id
        for slot in snapshot.slots.values()
        if slot.process_type is HordeProcessType.INFERENCE
        and slot.model == head.model
        and slot.process_id not in excluded
    ]
    if not holders:
        return []
    alternative = first_available_inference_slot(snapshot, disallowed=[*excluded, *holders])
    if alternative is not None or head_is_starving(snapshot, head):
        return holders
    return []


def card_inference_load(snapshot: SchedulingSnapshot, device_index: int) -> int:
    """How many of a card's inference slots are busy: the least-loaded routing tie-breaker."""
    return sum(
        1
        for slot in snapshot.slots.values()
        if slot.process_type is HordeProcessType.INFERENCE and slot.device_index == device_index and slot.is_busy
    )


def select_preload_target(snapshot: SchedulingSnapshot, job_id: str, disallowed: Iterable[int] = ()) -> int | None:
    """The inference slot to preload the job's model onto, choosing the card on a multi-GPU host, or None.

    Single-GPU: the first available slot. Multi-GPU: among the cards eligible for this job, the same
    sticky-then-least-loaded policy dispatch uses (a card already holding the model first, then the card
    running the fewest jobs, then the most measured free VRAM), taking the first available slot on the best
    card. Slots carrying the queue head's own copy are excluded first on either topology.
    """
    job = snapshot.queue.jobs[job_id]
    excluded = sorted(set(disallowed))
    head_slots = slots_holding_the_head_model(snapshot, job_id, excluded)
    if head_slots:
        excluded = [*excluded, *head_slots]
    if not snapshot.multi_gpu_routing_active:
        slot = first_available_inference_slot(snapshot, disallowed=excluded)
        return slot.process_id if slot is not None else None
    eligible = set(job.eligible_cards)
    if not eligible:
        return None
    cards_already_serving_model = {slot.device_index for slot in snapshot.slots.values() if slot.model == job.model}
    placement_order = card_preload_order(
        eligible,
        cards_already_serving_model=cards_already_serving_model,
        card_busy_counts={device_index: card_inference_load(snapshot, device_index) for device_index in eligible},
        card_free_vram_mb={device_index: snapshot.cards[device_index].measured_free_mb for device_index in eligible},
    )
    for device_index in placement_order:
        slot = first_available_inference_slot(snapshot, disallowed=excluded, device_index=device_index)
        if slot is not None:
            return slot.process_id
    return None


def select_head_room_target(snapshot: SchedulingSnapshot, job_id: str) -> int | None:
    """An eligible idle slot to free for a starved head, overriding the affinity and queued-model guards, or None.

    Never overrides card eligibility, never takes live work (a pinned disaggregation sampler is live work even
    while it idles between stages), and prefers the cheapest displacement.
    """
    job = snapshot.queue.jobs[job_id]
    eligible = job.eligible_cards if snapshot.multi_gpu_routing_active else None
    slots = tuple(
        PreloadSlotSnapshot(process_id=slot.process_id, model_name=slot.model, can_accept_job=slot.can_accept_job)
        for slot in snapshot.slots.values()
        if slot.process_type is HordeProcessType.INFERENCE
        and (eligible is None or slot.device_index in eligible)
        and not slot.reserved_for_disaggregation
    )
    return select_head_room_process_id(
        slots,
        in_progress_models=pricing.in_progress_models(snapshot),
        pending_models={
            model
            for model in (snapshot.queue.jobs[job_id].model for job_id in snapshot.queue.pending_in_placement_order)
            if model is not None
        },
    )


def preload_disallowed_processes(snapshot: SchedulingSnapshot, job: JobSnapshot) -> set[int]:
    """The slots this job's preload may not displace.

    The queued-model guard, model-process affinity and the RAM-draining marks compose in
    :func:`compute_preload_disallowed_processes`; a job that is not the admitted exclusive one may also not
    take a slot whose card an exclusive job holds. The guards are target exclusions only, never a wedge: the
    head-room fallback deliberately overrides them.
    """
    inference_slots = [slot for slot in snapshot.slots.values() if slot.process_type is HordeProcessType.INFERENCE]
    disallowed = compute_preload_disallowed_processes(
        queued_model_process_ids=sorted(snapshot.lifecycle.processes_with_model_for_queued_job),
        busy_process_ids=[slot.process_id for slot in snapshot.slots.values() if slot.is_busy],
        prefer_busy_only=pricing.loaded_inference_count(snapshot, None)
        < len(snapshot.queue.pending_in_placement_order) + len(snapshot.queue.in_progress),
        inference_process_models={slot.process_id: slot.model for slot in inference_slots},
        wanted_models=pricing.wanted_models(snapshot),
        max_inference_processes=snapshot.max_inference_processes,
        draining_process_ids=snapshot.draining_process_ids,
    )
    if not job.admitted_exclusive:
        disallowed.update(
            slot.process_id
            for slot in inference_slots
            if snapshot.card(snapshot.routing_device_index(slot)).exclusive_job_in_progress
        )
    return disallowed


def preloading_count(snapshot: SchedulingSnapshot, device_index: int | None) -> int:
    """How many processes are mid-preload on the card (or worker-wide for None)."""
    return sum(
        1
        for slot in snapshot.slots.values()
        if (device_index is None or slot.device_index == device_index)
        and slot.state is HordeProcessState.PRELOADING_MODEL
    )


def decide_preload_gates(
    snapshot: SchedulingSnapshot,
    job_id: str,
    *,
    head_job_id: str | None,
    loaded_models: frozenset[str | None],
) -> PreloadGatePlan:
    """Run one pending job through the gate ladder, up to and including target selection.

    The gates, in order: aux preparation (yields to a sibling able to sample), unserviceable (faults),
    quarantine (faults), already resident for this job's routing, the absolute RAM danger floor, target
    selection over the disallowed set with the head-room fallback, the exclusive-job hold on the target's
    card, the aux-gated occupied-slot refusal, the growth hold, cycle-on-model-change (replaces the child), and
    the per-card load serialization gate. An admitting plan names the target; the budget step follows.

    Args:
        snapshot: The cycle's snapshot.
        job_id: The pending job to consider.
        head_job_id: The queue head (see :func:`preload_head`), which alone may escalate to displacing another
            queued model.
        loaded_models: The models already resident or loading (see :func:`loaded_or_loading_models`).
    """
    job = snapshot.queue.jobs[job_id]
    if job.model is None:
        raise ValueError(f"job.model is None ({snapshot.queue.payloads[job_id]})")

    def plan(
        decision: AdmissionDecision,
        reason: str,
        *,
        target: int | None = None,
        commands: tuple[SchedulerCommand, ...] = (),
        notice: str | None = None,
        records_admission: bool = True,
        is_head_blocker: bool = False,
        disallowed: frozenset[int] = frozenset(),
    ) -> PreloadGatePlan:
        return PreloadGatePlan(
            decision=decision,
            job_id=job_id,
            reason=reason,
            target_process_id=target,
            commands=commands,
            is_head_blocker=is_head_blocker,
            disallowed_process_ids=disallowed,
            notice=notice,
            records_admission=records_admission,
        )

    # An aux-unprepared job must not compete for capacity: staging its model reserves a lane and prices VRAM
    # around work that cannot sample yet, so any job able to run outranks it. With no such sibling the
    # reservation costs nobody anything, and withholding it only serializes the checkpoint load behind the
    # auxiliary download the load could have run alongside.
    aux_gated = job.requires_aux_preparation
    if aux_gated and head_job_id is not None:
        return plan(
            AdmissionDecision.NEXT_JOB, "aux-gated job yields to a sibling able to sample", records_admission=False
        )

    payload = snapshot.queue.payloads[job_id]
    if job.unserviceable_reason is not None:
        commands = () if job.in_progress else (FaultJob(payload, FaultCause.UNSERVICEABLE, job.unserviceable_reason),)
        return plan(AdmissionDecision.UNSERVICEABLE, job.unserviceable_reason, commands=commands)

    if job.model in snapshot.lifecycle.quarantined_models:
        reason = "model load quarantined"
        commands = () if job.in_progress else (FaultJob(payload, FaultCause.QUARANTINED, reason),)
        return plan(AdmissionDecision.QUARANTINED, reason, commands=commands)

    if model_loaded_for_job(snapshot, job_id, loaded_models) and not duplicate_copy_may_serve(snapshot, job_id):
        return plan(AdmissionDecision.ALREADY_LOADED, "model already resident or loading")

    # Loading a new model routes its weights through system RAM first, so admitting one while the host is
    # already below its danger floor is the OS OOM kill, not progress. The governor's tick has already
    # degraded the host this cycle; this only defers the load.
    if snapshot.budget_active and snapshot.host_ram.pressure.under_pressure:
        return plan(
            AdmissionDecision.DEFER_RAM_PRESSURE,
            "system RAM danger floor",
            notice=(
                f"RAM danger floor reached: deferring preload of {job.model} ({snapshot.host_ram.pressure.reason()}). "
                "Shedding idle footprint and pausing pops."
            ),
        )

    is_head_blocker = head_job_id is not None and job_id == head_job_id
    disallowed = frozenset(preload_disallowed_processes(snapshot, job))
    target = select_preload_target(snapshot, job_id, disallowed)
    if target is None and is_head_blocker:
        # Affinity is provisioned against the inference-process ceiling, so with more resident models than
        # running processes it can pin every slot and starve a genuinely queued head. The head must make
        # progress whether or not the budget is active, so fall back to a displacement target that spares live
        # work. A slot retaining weights a queued job reuses is not spared here: the placement order has
        # already moved that job ahead of this one.
        target = select_head_room_target(snapshot, job_id)
    if target is None:
        return plan(
            AdmissionDecision.NO_TARGET,
            "no idle inference slot available",
            is_head_blocker=is_head_blocker,
            disallowed=disallowed,
        )

    slot = snapshot.slots[target]
    scope_card = snapshot.card(snapshot.routing_device_index(slot))
    if scope_card.exclusive_job_in_progress and not job.admitted_exclusive:
        return plan(
            AdmissionDecision.EXCLUSIVE_IN_PROGRESS,
            "exclusive over-budget job in progress on target card",
            target=target,
        )

    # Displacing a resident model, cycling a process or evicting to make room are all costs paid on behalf of a
    # job that cannot sample until its auxiliary files land, and the model thrown away may be one a
    # dispatchable job still wants.
    if aux_gated and not slot.is_unoccupied:
        return plan(AdmissionDecision.NEXT_JOB, "aux-gated preload would displace an occupied slot", target=target)

    # While the target card's device-level free VRAM sits below the soft floor, bringing a model to a slot that
    # does not already hold it would grow a footprint already near the WDDM paging cliff. A job already in
    # progress is exempt: its preload is part of live work the card is committed to.
    if scope_card.growth_held and not job.in_progress:
        return plan(
            AdmissionDecision.DEFER_VRAM_GROWTH_HOLD,
            "device-free governor holding VRAM growth (device near paging cliff)",
            target=target,
        )

    if (
        slot.state is not HordeProcessState.WAITING_FOR_JOB
        and slot.model is not None
        and snapshot.config.cycle_process_on_model_change
        and not snapshot.shutting_down
    ):
        return plan(
            AdmissionDecision.REPLACE_PROCESS,
            "cycling process for model change",
            target=target,
            commands=(ReplaceProcess(target),),
        )

    # Serialize preloads per card, not worker-wide: two checkpoints loading onto one device stack their
    # disk-read and allocation spikes, while a load onto an idle card is independent of one on another card.
    preloading = preloading_count(snapshot, snapshot.routing_device_index(slot))
    if preload_concurrency_blocked(
        num_preloading=preloading,
        max_concurrent_inference_processes=snapshot.max_concurrent_inference_processes,
        very_fast_disk_mode=bool(snapshot.config.very_fast_disk_mode),
    ):
        return plan(
            AdmissionDecision.DEFER_CONCURRENCY,
            "preload concurrency gate",
            target=target,
            notice=f"Already preloading {preloading} models, waiting for one to finish before preloading {job.model}",
        )

    return plan(
        AdmissionDecision.ADMIT,
        "gates passed",
        target=target,
        is_head_blocker=is_head_blocker,
        disallowed=disallowed,
    )


# ---- the budget step's pricing


@dataclass(frozen=True)
class PricedPreload:
    """A preload priced for the arbiter, with the predictive verdict and forecast its diagnostics read."""

    priced: MaterializationRequest
    predictive: BudgetVerdict
    """The static budget's verdict: its rejected peak sizes the context reduction, and its ``fits`` annotates a
    defer the measured arbiter takes against a candidate the static forecast called a fit."""
    forecast: StreamForecast
    context_reduction: ContextReduction

    @property
    def max_resident(self) -> int | None:
        """The context-reduction depth after the measured widening."""
        return self.priced.max_resident


def preload_candidate_delta_mb(snapshot: SchedulingSnapshot, job_id: str, process_id: int | None) -> float | None:
    """The VRAM (MB) a preload of the job charges the card at admission.

    Without the clearance lease a preload is the VRAM moment (the child loads the weights when the job starts),
    so it is priced at the job's full marginal sampling charge. Under the lease a preload only stages the job
    in system RAM and the weights load inside the leased sample call, so the charge is capped at the staging
    encode footprint, the same figure a staged dispatch books; the full fit-or-evict runs at clearance. Pricing
    the stage at the full peak would park the next model's disk read behind the current sample on every model
    switch, which is exactly the work the stage exists to overlap.
    """
    job = snapshot.queue.jobs[job_id]
    delta_mb = pricing.candidate_delta_mb(
        snapshot,
        snapshot.queue.payloads[job_id],
        job.baseline,
        process_id=process_id,
        disaggregated=job.disaggregation_class_eligible,
    )
    if delta_mb is None or not bool(snapshot.config.gpu_sampling_lease_enabled):
        return delta_mb
    return min(delta_mb, pricing.STAGING_ENCODE_VRAM_MB)


def predictive_vram_verdict(snapshot: SchedulingSnapshot, job_id: str, device_index: int | None) -> BudgetVerdict:
    """The static VRAM budget's verdict for the job against the card's measured free and the committed reserve."""
    job = snapshot.queue.jobs[job_id]
    return snapshot.services.vram_budget.check_job(
        snapshot.queue.payloads[job_id],
        job.baseline,
        snapshot.card(device_index).measured_free_mb,
        committed_reserve_mb=snapshot.committed_vram_reserve_mb,
        disaggregated=job.disaggregation_class_eligible,
    )


def context_reduction_demand(
    snapshot: SchedulingSnapshot,
    predictive: BudgetVerdict,
    forecast: StreamForecast,
    *,
    is_head_blocker: bool,
    device_index: int | None,
) -> ContextReduction:
    """The head's context-reduction depth and whether reducing live contexts is a warranted remedy.

    A moderate head's weights fit after a model eviction but its activation peak does not while this many
    contexts are live (each extra context retains VRAM the allocator never returns). Reducing the live
    inference-process count to the largest that still seats the rejected peak plus its structural reserve is
    the remedy. The depth keys on the honest streaming floor, not the operator's configured margin, so only a
    genuinely card-filling peak pushes the co-resident count below the live pool; a demand resting on untrusted
    (unmeasured-fallback) overhead figures is not warranted. The warrant is measured, never an operator
    preference: ``whole_card_exclusive_residency`` governs exclusive residencies, and a context reduction is
    not one.
    """
    max_resident: int | None = None
    if predictive.predicted_mb is not None:
        total_vram_mb = snapshot.card(device_index).total_vram_mb
        structural_reserve_mb = (
            effective_inference_reserve_mb(total_vram_mb, 0.0) if total_vram_mb is not None else predictive.reserve_mb
        )
        max_resident = pricing.max_coresident_for_peak_mb(
            snapshot,
            predictive.predicted_mb,
            structural_reserve_mb,
            device_index=device_index,
        )
    demanded = (
        is_head_blocker
        and max_resident is not None
        and pricing.loaded_inference_count(snapshot, device_index) > max_resident
    )
    warranted = pricing.whole_card_warranted(
        forecast,
        marginal_overhead_mb=pricing.marginal_process_overhead_mb(snapshot, None),
    )
    return ContextReduction(max_resident=max_resident, can_reduce_live_contexts=demanded and warranted)


def price_preload(
    snapshot: SchedulingSnapshot,
    job_id: str,
    process_id: int,
    *,
    arbiter: VramArbiter,
    is_head_blocker: bool,
    forecast: StreamForecast,
) -> PricedPreload:
    """Price a preload onto the slot: the predictive verdict, the context reduction it sizes, the arbiter request."""
    slot = snapshot.slots[process_id]
    device_index = snapshot.routing_device_index(slot)
    predictive = predictive_vram_verdict(snapshot, job_id, device_index)
    context_reduction = context_reduction_demand(
        snapshot,
        predictive,
        forecast,
        is_head_blocker=is_head_blocker,
        device_index=device_index,
    )
    priced = build_materialization_request(
        snapshot,
        job_id,
        process_id,
        arbiter=arbiter,
        is_head_of_queue=is_head_blocker,
        head_outstanding_mb=None,
        candidate_delta_override_mb=preload_candidate_delta_mb(snapshot, job_id, process_id),
        kind=VramRequestKind.PRELOAD,
        context_reduction=context_reduction,
        forecast=forecast,
        prepared_head_reprices_activation=False,
    )
    return PricedPreload(priced=priced, predictive=predictive, forecast=forecast, context_reduction=context_reduction)
