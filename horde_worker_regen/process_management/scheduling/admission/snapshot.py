"""The immutable picture of the worker that one scheduling cycle decides from.

Every field is something the preload, dispatch or clearance pipeline reads from a collaborator today. Building
it once at the start of a cycle, and again only at a declared phase boundary, is what makes those pipelines
decisions rather than sequences of reads interleaved with actuations: two gates in one cycle can no longer see
two different cards.

The snapshot carries values, not collaborators. The three pricing services it does carry (model metadata, the
learned footprint store, the context overhead model) are read-only within a cycle and are consumed only by pure
pricing functions; the committed reserve ledger is carried the same way until its planned charges are folded
into the card snapshot.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner, ProcessLifecycleManager
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger, RamPressureVerdict
from horde_worker_regen.process_management.resources.vram_arbiter import DeviceVramState
from horde_worker_regen.process_management.resources.vram_footprints import LearnedFootprintStore
from horde_worker_regen.process_management.scheduling.context_overhead_model import ContextOverheadModel
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    WholeCardPhase,
    WholeCardResidencyLedger,
)
from horde_worker_regen.process_management.scheduling.ledgers.dispatch_holds import DispatchHoldLedger
from horde_worker_regen.process_management.scheduling.ledgers.head_admission import (
    HeadAdmissionLedger,
    LatestPreloadAdmission,
)
from horde_worker_regen.process_management.scheduling.ledgers.ram_reclaim import RamReclaimLedger
from horde_worker_regen.process_management.scheduling.ledgers.retention import (
    RETENTION_STALE_HOLD_SECONDS,
    RetentionLedger,
)
from horde_worker_regen.process_management.scheduling.ledgers.safety_placement import SafetyPlacementLedger


def job_key(job: ImageGenerateJobPopResponse) -> str | None:
    """The string id the snapshot keys a job by, or None for a job without one."""
    return str(job.id_) if job.id_ is not None else None


@dataclass(frozen=True)
class SlotSnapshot:
    """One child process as the pipelines see it."""

    process_id: int
    os_pid: int | None
    device_index: int
    process_type: HordeProcessType
    state: HordeProcessState
    model: str | None
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None
    last_control_flag: HordeControlFlag | None
    reserved_mb: int | None
    allocated_mb: int | None
    ram_usage_bytes: int
    retained_resident_model: str | None
    retention_granted_model: str | None
    is_busy: bool
    can_accept_job: bool
    is_alive: bool
    is_unoccupied: bool
    current_job_id: str | None
    held_component_count: int
    resident_weight_models: frozenset[str]
    """Models whose weights are in VRAM on this slot right now, as the model map answers it."""
    aimdo_mb: int | None
    parked_preload: bool
    """Whether the slot has sat on a completed preload past the retention stale horizon with no job."""


@dataclass(frozen=True)
class CardSnapshot:
    """One card's measured and governed state."""

    device_index: int | None
    """The card, or None for the worker-wide view a single-GPU host schedules against."""
    total_vram_mb: float | None
    measured_free_mb: float | None
    governor_state: GovernorState
    growth_held: bool
    arbiter_state: DeviceVramState | None
    """The arbiter's frozen measurement for this cycle, or None before the arbiter has a cycle."""
    whole_card_model: str | None
    """The model a whole-card residency holds this card for, or None."""
    whole_card_phase: WholeCardPhase | None
    max_concurrent_inference: int | None
    target_process_count: int | None
    config: reGenBridgeData
    """The card's effective config: its per-card resolution on a multi-GPU host, else the global config."""
    reserved_by_pid: Mapping[int, float]
    """The live GPU processes' measured allocator reservation (MB) by process id, the figure planned
    reserve charges decay against."""
    foreign_floor_mb: float | None
    """The sustained VRAM the OS, desktop or other processes hold on the card, or None when unmeasured."""


@dataclass(frozen=True)
class JobSnapshot:
    """A pending or in-progress job as the gates judge it."""

    job_id: str
    model: str | None
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None
    in_progress: bool
    admitted_exclusive: bool
    admitted_over_budget: bool
    aux_models_prepared: bool
    degraded_dispatch_pending: bool
    disaggregation_class_eligible: bool
    """Whether the job belongs to the disaggregation class (sampler-only pricing), by the wired predicate."""
    tracked: bool
    """Whether the tracker holds a record for the job (accepted work rather than a speculative offer)."""
    measured_attempt_devices: frozenset[int | None]
    """Cards (or the worker-wide key) on which the job has an active measured-load attempt."""
    measured_attempt_spent_devices: frozenset[int | None]
    """Cards (or the worker-wide key) on which the job has ever started a measured-load attempt."""
    eligible_cards: frozenset[int]
    """Cards whose effective config can serve the job; every card on a single-GPU host."""


@dataclass(frozen=True)
class QueueSnapshot:
    """The queue in placement order, with the in-progress set and each job's gate facts."""

    pending_in_placement_order: tuple[str, ...]
    pending_in_pop_order: tuple[str, ...]
    """The pending queue in the order the tracker holds it, which the lookahead and demand reads use."""
    in_progress: tuple[str, ...]
    jobs: Mapping[str, JobSnapshot]
    payloads: Mapping[str, ImageGenerateJobPopResponse]
    """The pop responses themselves, for the pricing functions that read the payload."""

    def head(self) -> str | None:
        """The first pending job no process is running: the head of the queue."""
        return next((job_id for job_id in self.pending_in_placement_order if job_id not in self.in_progress), None)


@dataclass(frozen=True)
class ModelMapEntry:
    """One model-map entry: which process claims the model and how far its load has got."""

    model: str
    load_state: ModelLoadState
    process_id: int


@dataclass(frozen=True)
class LifecycleSnapshot:
    """The lifecycle manager's flags the gates read."""

    safety_gpu_paused: bool
    safety_pause_owner: PauseOwner | None
    safety_transition_pending: bool
    safety_gpu_card: int | None
    post_process_gpu_paused: bool
    safety_pool_failing: bool
    pending_safety_starts: bool
    pending_safety_start_devices: frozenset[int]
    quarantined_models: frozenset[str]
    """The queued models whose loads are quarantined."""
    processes_with_model_for_queued_job: frozenset[int]


@dataclass(frozen=True)
class HostRamSnapshot:
    """System RAM as the RAM gates read it."""

    available_mb: float
    total_mb: float
    reserve_mb: float
    danger_floor_mb: float
    pressure: RamPressureVerdict | None
    """The governor's verdict for this tick, or None before the first tick."""


@dataclass(frozen=True)
class HeadAdmissionView:
    """The head-admission ledger's fields the gates read, frozen for the cycle."""

    starvation_job_id: str | None
    starvation_since: float
    barrier_job_id: str | None
    ram_defer_job_id: str | None
    recovery_hold_since: float
    last_preload_admission: LatestPreloadAdmission | None
    model_recently_missing: bool
    heavy_head_admitted_at: float


@dataclass(frozen=True)
class DispatchHoldView:
    """The dispatch-hold ledger's fields the gates read, frozen for the cycle."""

    hold_since: Mapping[str, float]
    reclaim_requested: frozenset[str]
    pp_defer_holds: frozenset[str]
    clearance_hold_ids: frozenset[str]


@dataclass(frozen=True)
class RetentionView:
    """The retention ledger's fields the gates read, frozen for the cycle."""

    wddm_paging_active: bool
    slot_dispatch_history: Mapping[int, tuple[str, ...]]
    pending_eviction_process_ids: frozenset[int]


@dataclass(frozen=True)
class SafetyPlacementView:
    """The safety-placement ledger's fields the gates read, frozen for the cycle."""

    reclaim_pause_requested: bool
    weights_demoted: bool


@dataclass(frozen=True)
class RamReclaimView:
    """The RAM-reclaim ledger's fields the gates read, frozen for the cycle."""

    cycle_at: float
    pending_reuse_credit_process_ids: frozenset[int]


@dataclass(frozen=True)
class LedgerViews:
    """Every ledger's read view for the cycle."""

    head_admission: HeadAdmissionView
    dispatch_holds: DispatchHoldView
    retention: RetentionView
    safety_placement: SafetyPlacementView
    ram_reclaim: RamReclaimView


@dataclass(frozen=True)
class PricingServices:
    """Read-only collaborators the pure pricing functions consume.

    Each is stable within a cycle: the metadata and overhead model change only on a config reload, the footprint
    store only when a child reports, and the reserve ledger only when the executor applies a plan.
    """

    model_metadata: ModelMetadata
    footprint_store: LearnedFootprintStore | None
    overhead: ContextOverheadModel
    reserve_ledger: CommittedReserveLedger


@dataclass(frozen=True)
class SchedulingSnapshot:
    """Everything one scheduling cycle decides from."""

    now: float
    multi_gpu_routing_active: bool
    cards: Mapping[int | None, CardSnapshot]
    slots: Mapping[int, SlotSnapshot]
    queue: QueueSnapshot
    model_map: Mapping[str, ModelMapEntry]
    lifecycle: LifecycleSnapshot
    host_ram: HostRamSnapshot
    config: reGenBridgeData
    budget_active: bool
    vram_reserve_mb: float
    safety_footprint_mb: float
    """The device VRAM (MB) the safety process costs while it sits on the GPU: the one safety price."""
    safety_on_gpu_permitted: bool
    context_constant_mb: float
    """The per-process CUDA-context charge (MB) the committed ledger and the learned store net out."""
    max_inference_processes: int
    committed_vram_reserve_mb: float
    """The reserve ledger's combined committed VRAM (MB) across every flow, charged against any card."""
    models_with_results: frozenset[str]
    """Models that have completed at least one job on this worker, the trust gate for measured footprints."""
    whole_card_held_models: frozenset[str]
    """Models holding a whole-card residency somewhere, which pressure eviction never takes."""
    ledgers: LedgerViews
    services: PricingServices

    def card(self, device_index: int | None) -> CardSnapshot:
        """The card a slot's memory is scoped to, or the worker-wide view when routing is card-agnostic."""
        key = device_index if self.multi_gpu_routing_active else None
        return self.cards[key]

    def config_for(self, device_index: int | None) -> reGenBridgeData:
        """The effective config for a card: its per-card resolution when one exists, else the global."""
        card = self.cards.get(device_index)
        return card.config if card is not None else self.config

    def routing_device_index(self, slot: SlotSnapshot) -> int | None:
        """The card a slot's memory is scoped to: its device on a multi-GPU host, the whole worker otherwise."""
        return slot.device_index if self.multi_gpu_routing_active else None

    def slots_of_type(self, process_type: HordeProcessType) -> tuple[SlotSnapshot, ...]:
        """The slots of one process type, in process-id order."""
        return tuple(slot for slot in self.slots.values() if slot.process_type is process_type)

    def model_loaded(self, model: str) -> bool:
        """Whether the model map records the model as loaded on some process."""
        entry = self.model_map.get(model)
        return entry is not None and entry.load_state is ModelLoadState.LOADED_IN_VRAM


def snapshot_slot(process_info: HordeProcessInfo, horde_model_map: HordeModelMap, *, now: float) -> SlotSnapshot:
    """Freeze one process record, answering weight residency through the model map for the models it names."""
    job = process_info.current_inference_job()
    candidates = {name for name, info in horde_model_map.root.items() if info.process_id == process_info.process_id}
    candidates.update(
        name for name in (process_info.loaded_horde_model_name, process_info.retained_resident_model) if name
    )
    return SlotSnapshot(
        process_id=process_info.process_id,
        os_pid=process_info.os_pid,
        device_index=process_info.device_index,
        process_type=process_info.process_type,
        state=process_info.last_process_state,
        model=process_info.loaded_horde_model_name,
        baseline=process_info.loaded_horde_model_baseline,
        last_control_flag=process_info.last_control_flag,
        reserved_mb=process_info.process_reserved_mb,
        allocated_mb=process_info.process_allocated_mb,
        ram_usage_bytes=process_info.ram_usage_bytes,
        retained_resident_model=process_info.retained_resident_model,
        retention_granted_model=process_info.retention_granted_model,
        is_busy=process_info.is_process_busy(),
        can_accept_job=process_info.can_accept_job(),
        is_alive=process_info.is_process_alive(),
        is_unoccupied=process_info.is_unoccupied(),
        current_job_id=job_key(job) if job is not None else None,
        held_component_count=len(process_info.held_components or ()),
        resident_weight_models=frozenset(
            name for name in candidates if horde_model_map.weights_resident_on_process(name, process_info)
        ),
        aimdo_mb=process_info.process_aimdo_mb,
        parked_preload=process_info.is_parked_preload(now=now, dwell_seconds=RETENTION_STALE_HOLD_SECONDS),
    )


def snapshot_ledgers(
    *,
    head_admission: HeadAdmissionLedger,
    dispatch_holds: DispatchHoldLedger,
    retention: RetentionLedger,
    safety_placement: SafetyPlacementLedger,
    ram_reclaim: RamReclaimLedger,
) -> LedgerViews:
    """Freeze the five ledgers' gate-facing fields."""
    return LedgerViews(
        head_admission=HeadAdmissionView(
            starvation_job_id=head_admission.starvation_job_id,
            starvation_since=head_admission.starvation_since,
            barrier_job_id=head_admission.barrier_job_id,
            ram_defer_job_id=head_admission.ram_defer_job_id,
            recovery_hold_since=head_admission.recovery_hold_since,
            last_preload_admission=head_admission.last_preload_admission,
            model_recently_missing=head_admission.model_recently_missing,
            heavy_head_admitted_at=head_admission.heavy_head_admitted_at,
        ),
        dispatch_holds=DispatchHoldView(
            hold_since=dict(dispatch_holds.hold_since),
            reclaim_requested=frozenset(dispatch_holds.reclaim_requested),
            pp_defer_holds=frozenset(dispatch_holds.pp_defer_holds),
            clearance_hold_ids=frozenset(dispatch_holds.clearance_hold_ids),
        ),
        retention=RetentionView(
            wddm_paging_active=retention.wddm_paging_active,
            slot_dispatch_history=retention.slot_dispatch_history(),
            pending_eviction_process_ids=retention.pending_eviction_process_ids(),
        ),
        safety_placement=SafetyPlacementView(
            reclaim_pause_requested=safety_placement.reclaim_pause_requested,
            weights_demoted=safety_placement.weights_demoted,
        ),
        ram_reclaim=RamReclaimView(
            cycle_at=ram_reclaim.cycle_at,
            pending_reuse_credit_process_ids=frozenset(ram_reclaim.pending_reuse_credits),
        ),
    )


def build_scheduling_snapshot(
    *,
    now: float,
    process_map: Mapping[int, HordeProcessInfo],
    job_tracker: JobTracker,
    pending_in_placement_order: Iterable[ImageGenerateJobPopResponse],
    horde_model_map: HordeModelMap,
    process_lifecycle: ProcessLifecycleManager,
    card_runtimes: Mapping[int, CardRuntime],
    bridge_data: reGenBridgeData,
    model_metadata: ModelMetadata,
    footprint_store: LearnedFootprintStore | None,
    overhead: ContextOverheadModel,
    reserve_ledger: CommittedReserveLedger,
    whole_card_ledger: WholeCardResidencyLedger,
    whole_card_phase: Callable[[int | None], tuple[str | None, WholeCardPhase]],
    measured_free_mb: Callable[[int | None], float | None],
    reported_total_mb: Callable[[int | None], float | None],
    governor_state: Callable[[int | None], GovernorState],
    growth_held: Callable[[int | None], bool],
    arbiter_state: Callable[[int | None], DeviceVramState | None],
    foreign_floor_mb: Callable[[int | None], float | None],
    eligible_cards: Callable[[ImageGenerateJobPopResponse], set[int]],
    disaggregation_class_eligible: Callable[[ImageGenerateJobPopResponse], bool],
    host_ram: HostRamSnapshot,
    budget_active: bool,
    vram_reserve_mb: float,
    safety_footprint_mb: float,
    safety_on_gpu_permitted: bool,
    context_constant_mb: float,
    max_inference_processes: int,
    models_with_results: frozenset[str],
    ledgers: LedgerViews,
) -> SchedulingSnapshot:
    """Freeze the worker for one cycle.

    Takes the collaborators explicitly so the builder is testable without a scheduler; the scheduler's
    ``snapshot`` method is the one production caller and supplies its own accessors for the per-card
    measurements.
    """
    multi_gpu = len(card_runtimes) > 1
    card_keys: list[int | None] = [None]
    if multi_gpu:
        card_keys.extend(sorted(card_runtimes))
    cards: dict[int | None, CardSnapshot] = {}
    reserved_by_pid_all = {
        info.process_id: (info.device_index, float(info.process_reserved_mb))
        for info in process_map.values()
        if info.process_reserved_mb is not None
    }
    for key in card_keys:
        runtime = card_runtimes.get(key) if key is not None else None
        held_model, phase = whole_card_phase(key)
        cards[key] = CardSnapshot(
            device_index=key,
            total_vram_mb=reported_total_mb(key),
            measured_free_mb=measured_free_mb(key),
            governor_state=governor_state(key),
            growth_held=growth_held(key),
            arbiter_state=arbiter_state(key),
            whole_card_model=held_model,
            whole_card_phase=phase if held_model is not None else None,
            max_concurrent_inference=runtime.max_concurrent_inference if runtime is not None else None,
            target_process_count=runtime.target_process_count if runtime is not None else None,
            config=runtime.config if runtime is not None else bridge_data,
            reserved_by_pid={
                pid: reserved
                for pid, (device, reserved) in reserved_by_pid_all.items()
                if key is None or device == key
            },
            foreign_floor_mb=foreign_floor_mb(key),
        )

    in_progress_jobs = list(job_tracker.jobs_in_progress)
    pending_jobs = list(pending_in_placement_order)
    pop_order = tuple(key for key in (job_key(job) for job in job_tracker.jobs_pending_inference) if key)
    payloads: dict[str, ImageGenerateJobPopResponse] = {}
    jobs: dict[str, JobSnapshot] = {}
    in_progress_ids: list[str] = []
    pending_ids: list[str] = []
    for job, is_in_progress in [(job, True) for job in in_progress_jobs] + [(job, False) for job in pending_jobs]:
        key = job_key(job)
        if key is None:
            continue
        if key in jobs:
            if not is_in_progress:
                pending_ids.append(key)
            continue
        payloads[key] = job
        jobs[key] = JobSnapshot(
            job_id=key,
            model=job.model,
            baseline=model_metadata.get_baseline(job.model) if job.model is not None else None,
            in_progress=is_in_progress,
            admitted_exclusive=job_tracker.is_admitted_exclusive(job),
            admitted_over_budget=job_tracker.is_admitted_over_budget(job),
            aux_models_prepared=job_tracker.are_job_aux_models_prepared(job),
            degraded_dispatch_pending=job_tracker.is_degraded_dispatch_pending(job),
            disaggregation_class_eligible=disaggregation_class_eligible(job),
            tracked=job.id_ is not None and job_tracker.get_tracked_job(job.id_) is not None,
            measured_attempt_devices=frozenset(
                device for device in card_keys if job_tracker.is_measured_attempt_on_device(job, device)
            ),
            measured_attempt_spent_devices=frozenset(
                device for device in card_keys if job_tracker.has_spent_measured_attempt_on_device(job, device)
            ),
            eligible_cards=frozenset(eligible_cards(job)),
        )
        (in_progress_ids if is_in_progress else pending_ids).append(key)

    queued_models = {job.model for job in pending_jobs if job.model is not None}
    # The waiting cards are read only while a safety start is pending, as the recovery hold reads them: the
    # lifecycle manager computes the set on demand and a stand-in need not answer when nothing is pending.
    pending_safety_starts = process_lifecycle.has_pending_safety_starts() is True
    pending_devices = (
        frozenset(process_lifecycle.pending_safety_start_device_indices()) if pending_safety_starts else frozenset()
    )
    lifecycle = LifecycleSnapshot(
        safety_gpu_paused=bool(process_lifecycle.is_safety_gpu_paused),
        safety_pause_owner=process_lifecycle.safety_pause_owner,
        safety_transition_pending=process_lifecycle.safety_placement_transition_pending is True,
        safety_gpu_card=process_lifecycle.safety_gpu_card_index(),
        post_process_gpu_paused=bool(process_lifecycle.is_post_process_gpu_paused),
        safety_pool_failing=process_lifecycle.safety_pool_failing is True,
        pending_safety_starts=pending_safety_starts,
        pending_safety_start_devices=pending_devices,
        quarantined_models=frozenset(
            model for model in queued_models if process_lifecycle.is_model_load_quarantined(model)
        ),
        processes_with_model_for_queued_job=frozenset(process_lifecycle.get_processes_with_model_for_queued_job()),
    )

    return SchedulingSnapshot(
        now=now,
        multi_gpu_routing_active=multi_gpu,
        cards=cards,
        slots={
            process_id: snapshot_slot(info, horde_model_map, now=now)
            for process_id, info in sorted(process_map.items())
        },
        queue=QueueSnapshot(
            pending_in_placement_order=tuple(pending_ids),
            pending_in_pop_order=pop_order,
            in_progress=tuple(in_progress_ids),
            jobs=jobs,
            payloads=payloads,
        ),
        model_map={
            name: ModelMapEntry(model=name, load_state=info.horde_model_load_state, process_id=info.process_id)
            for name, info in horde_model_map.root.items()
        },
        lifecycle=lifecycle,
        host_ram=host_ram,
        config=bridge_data,
        budget_active=budget_active,
        vram_reserve_mb=vram_reserve_mb,
        safety_footprint_mb=safety_footprint_mb,
        safety_on_gpu_permitted=safety_on_gpu_permitted,
        context_constant_mb=context_constant_mb,
        max_inference_processes=max_inference_processes,
        committed_vram_reserve_mb=reserve_ledger.total_vram_mb(),
        models_with_results=models_with_results,
        whole_card_held_models=frozenset(
            state.model for _, state in whole_card_ledger.held() if state.model is not None
        ),
        ledgers=ledgers,
        services=PricingServices(
            model_metadata=model_metadata,
            footprint_store=footprint_store,
            overhead=overhead,
            reserve_ledger=reserve_ledger,
        ),
    )
