"""Pure decision functions for governing host system RAM.

This module decides, from a single
[`HostMemorySnapshot`][horde_worker_regen.process_management.scheduling.governance.snapshots.HostMemorySnapshot],
which remedies the worker applies this tick to stay above its absolute RAM danger floor: pausing and
holding job pops, evicting idle models, shedding idle inference contexts (per card on a multi-GPU host),
reclaiming a process whose resident RAM crossed the per-process ceiling, and growing shed cards back once
the host recovers. Every function here is side-effect free; the scheduler executes the returned actions
through its single dispatcher.

Critical public members:

* [`RamGovernorState`][horde_worker_regen.process_management.scheduling.governance.ram_governor.RamGovernorState]:
  the multi-tick bookkeeping (draining processes, shed cards) carried between cycles.
* [`decide_pressure_governance`]
  [horde_worker_regen.process_management.scheduling.governance.ram_governor.decide_pressure_governance]:
  the per-tick entry point (pop hold plus the full degrade response when under the floor).
* [`decide_shed_card_restore`]
  [horde_worker_regen.process_management.scheduling.governance.ram_governor.decide_shed_card_restore]:
  the recovery counterpart that grows shed cards back toward plan.
* [`decide_draining_followthrough`]
  [horde_worker_regen.process_management.scheduling.governance.ram_governor.decide_draining_followthrough]:
  resolves pressure-initiated draining marks after the floor clears, so a drain never outlives its episode
  as a permanent hold on intake.
* [`RAM_PRESSURE_PAUSE_SECONDS`]
  [horde_worker_regen.process_management.scheduling.governance.ram_governor.RAM_PRESSURE_PAUSE_SECONDS]:
  how long one pressure reading pauses job pops.
"""

from __future__ import annotations

from dataclasses import dataclass

from horde_worker_regen.process_management.scheduling.governance.actions import (
    ClearProcessDraining,
    EvictIdleModels,
    GovernanceAction,
    MarkProcessDraining,
    PausePops,
    RecycleProcess,
    ReduceCardProcesses,
    ReduceWorkerProcesses,
    RestoreCardProcess,
    RestoreWorkerProcess,
    SetPopHold,
    StopTrackingShedCard,
    StopTrackingWorkerShed,
)
from horde_worker_regen.process_management.scheduling.governance.snapshots import HostMemorySnapshot

__all__ = [
    "RAM_PRESSURE_PAUSE_SECONDS",
    "RamGovernorState",
    "WorkerProcessShedState",
    "decide_degrade_response",
    "decide_draining_followthrough",
    "decide_over_ceiling_reclaim",
    "decide_pop_hold",
    "decide_pressure_governance",
    "decide_process_reduction",
    "decide_shed_card_restore",
    "decide_shed_restore",
]

RAM_PRESSURE_PAUSE_SECONDS = 30.0
"""How long the worker pauses job pops once system RAM crosses its danger floor.

A short, self-expiring pop-pause (it auto-resumes via the manager's self-throttle cooldown) so intake
stops adding memory pressure while idle footprint is shed and the host recovers, without wedging a worker
whose RAM frees up moments later. Re-armed each scheduling pass that still reads under the floor."""


@dataclass(slots=True)
class WorkerProcessShedState:
    """Worker-wide RAM-pressure shedding to restore after a single-GPU pressure episode."""

    planned_process_count: int
    """The normal worker-wide process target before pressure shedding."""
    shed_process_count: int = 0
    """The live shortfall below plan (planned minus currently-loaded), not a running total of reductions.

    Every mutation recomputes this from the measured process count against ``planned_process_count`` so it
    stays the true outstanding gap. It must not accumulate across reductions: another mechanism (a
    whole-card residency restore) can regrow the pool between reductions, and a running total would then
    over-count each cycle without bound while the pool is actually back at plan."""


class RamGovernorState:
    """Multi-tick bookkeeping the RAM governor carries between scheduling cycles.

    Decisions read this state through the frozen copies embedded in a snapshot; only the scheduler's
    action dispatcher mutates it, at execution time, so the decision layer stays pure. Both sets are
    scoped to a pressure episode and empty on a healthy host.

    Thread Safety:
        Owned and mutated exclusively by the scheduler's control loop; not safe for concurrent mutation.
    """

    def __init__(self) -> None:
        """Initialize an empty (healthy-host) governor state."""
        self.shed_cards: set[int] = set()
        """Device indices the RAM-pressure reduction shed below their planned per-card process count.

        The reduction keeps one context per driven card so no GPU is stranded, and records each card it
        shrank here so the restore path can grow it back once the host clears the danger floor.
        Multi-GPU only; the worker-wide (single-GPU) reduction does not populate it.
        """
        self.worker_shed: WorkerProcessShedState | None = None
        """Worker-wide single-GPU process shedding that should be restored after RAM recovers.

        Multi-GPU pressure uses :attr:`shed_cards`; single-GPU pressure has no per-card identity to track,
        so it records the planned worker-wide count and how many idle contexts actually stopped.
        """
        self.draining_process_ids: set[int] = set()
        """Inference process ids marked to drain because their resident RAM crossed the per-process ceiling.

        A draining process is fed no new dispatch/preload so its in-flight job can finish, after which the
        governor recycles it to return its allocator-retained pages. Cleared when a process falls back
        under the ceiling (or is recycled). Distinct from :attr:`shed_cards`, which tracks *idle* contexts
        shed by count; this tracks a *specific busy* process being wound down.
        """


def decide_pop_hold(snapshot: HostMemorySnapshot) -> SetPopHold:
    """Return the soft, pre-floor pop hold setting for this tick.

    The hard floor pauses pops outright; this softer band stops the popper starting a new job's ttl clock
    *before* the host is critical, so a job does not age past its ttl waiting on a degraded worker and get
    aborted by the horde as too slow. The hold engages while the host is under the floor, while any process
    is being drained for reclaim, or while measured available RAM is within the margin of the floor *and*
    work is in flight whose completion will change the reading. An idle worker whose steady-state resident
    footprint simply sits inside the margin is not held: nothing on an idle host frees RAM on its own, so
    holding there starves the worker permanently, and a popped job is served immediately (no ttl-aging
    risk) with the hard floor still guarding actual overgrowth.
    """
    verdict = snapshot.verdict
    approaching = (
        verdict.available_mb is not None
        and (verdict.available_mb - verdict.floor_mb) < snapshot.pop_hold_margin_mb
        and snapshot.in_flight_job_count > 0
    )
    active = bool(verdict.under_pressure or approaching) or bool(snapshot.draining_process_ids)
    return SetPopHold(active=active)


def decide_process_reduction(snapshot: HostMemorySnapshot) -> list[GovernanceAction]:
    """Return the idle-context reductions that shed resident-weight RAM back to the OS.

    The structural remedy while the host is over its danger floor: fewer resident contexts, not another
    load on top. Each pool is reduced toward the count its in-flight work needs (at least one), shedding
    at least one idle sibling; only idle processes are ever stopped by execution, so live work is spared.

    On a multi-GPU host the reduction is per card so it never empties a card of every context (a
    worker-wide shrink would let the victim search stop every idle process regardless of card, leaving a
    GPU idle until restored). The single-GPU / worker-wide path reduces the whole pool.
    """
    if snapshot.multi_gpu_routing_active:
        reductions: list[GovernanceAction] = []
        for card in sorted(snapshot.cards, key=lambda card_snapshot: card_snapshot.device_index):
            if card.loaded_process_count <= 1:
                continue
            needed = max(1, card.busy_process_count)
            target = max(1, min(card.loaded_process_count - 1, needed))
            reductions.append(ReduceCardProcesses(device_index=card.device_index, target_count=target))
        return reductions

    current = snapshot.loaded_worker_process_count
    if current <= 1:
        return []
    target = max(1, current - 1)
    planned = max(snapshot.planned_worker_process_count, current)
    shortfall = None
    if snapshot.verdict.available_mb is not None:
        shortfall = max(0.0, snapshot.verdict.floor_mb - snapshot.verdict.available_mb)
    return [ReduceWorkerProcesses(target_count=target, planned_count=planned, pressure_shortfall_mb=shortfall)]


def decide_over_ceiling_reclaim(snapshot: HostMemorySnapshot) -> list[GovernanceAction]:
    """Return the reclaim response for processes whose resident RAM is at/above the per-process ceiling.

    Only meaningful while the host is under its RAM danger floor (a roomy host never recycles); the
    caller gates on pressure. Acts on one process per tick (the largest over-ceiling one) to avoid
    emptying every card at once: an idle offender is recycled now (its allocator-retained pages return to
    the OS on respawn), a busy one is marked draining so its in-flight job finishes and a later tick
    recycles it. Processes that have fallen back under the ceiling (or no longer exist) have their
    draining marks cleared; a disabled ceiling clears every mark. Marks that outlive the pressure episode
    are resolved by :func:`decide_draining_followthrough`.
    """
    ceiling_mb = snapshot.per_process_ceiling_mb
    if ceiling_mb is None:
        return [ClearProcessDraining(process_id=process_id) for process_id in sorted(snapshot.draining_process_ids)]

    actions: list[GovernanceAction] = []
    live_process_ids = {slot.process_id for slot in snapshot.inference_slots}
    for process_id in sorted(snapshot.draining_process_ids):
        # A mark whose process no longer exists (exited or replaced) can never resolve through the slot
        # scan below; left in place it would gate pops and restore forever.
        if process_id not in live_process_ids:
            actions.append(ClearProcessDraining(process_id=process_id))
    over_ceiling = []
    for slot in snapshot.inference_slots:
        if slot.resident_ram_mb >= ceiling_mb:
            over_ceiling.append(slot)
        elif slot.process_id in snapshot.draining_process_ids:
            actions.append(ClearProcessDraining(process_id=slot.process_id))

    if not over_ceiling:
        return actions

    target = max(over_ceiling, key=lambda slot: slot.resident_ram_mb)
    if target.is_busy:
        if target.process_id not in snapshot.draining_process_ids:
            actions.append(
                MarkProcessDraining(
                    process_id=target.process_id,
                    resident_ram_mb=target.resident_ram_mb,
                    ceiling_mb=ceiling_mb,
                ),
            )
        return actions

    actions.append(
        RecycleProcess(
            process_id=target.process_id,
            resident_ram_mb=target.resident_ram_mb,
            ceiling_mb=ceiling_mb,
        ),
    )
    return actions


def decide_degrade_response(snapshot: HostMemorySnapshot) -> list[GovernanceAction]:
    """Return the whole-host degrade response for a host under its RAM danger floor.

    The proactive counterpart to the marginal RAM budget: rather than admit a load that the absolute
    reading says will trip the kernel OOM-killer, the worker (1) pauses job pops so intake stops adding
    pressure, (2) evicts idle resident models, (3) reduces the resident inference-process count so the
    multi-GB of resident weights each idle context pins is returned to the OS, and (4) reclaims a process
    whose resident RAM crossed the per-process ceiling. Returns no actions on a healthy host.
    """
    if not snapshot.verdict.under_pressure:
        return []
    actions: list[GovernanceAction] = []
    until = snapshot.now + RAM_PRESSURE_PAUSE_SECONDS
    if not (snapshot.pop_pause_active and snapshot.pop_pause_until >= until):
        actions.append(
            PausePops(
                until_time=until,
                pause_seconds=RAM_PRESSURE_PAUSE_SECONDS,
                reason=snapshot.verdict.reason(),
                available_mb=snapshot.verdict.available_mb,
                floor_mb=snapshot.verdict.floor_mb,
            ),
        )
    actions.append(EvictIdleModels())
    actions.extend(decide_process_reduction(snapshot))
    actions.extend(decide_over_ceiling_reclaim(snapshot))
    return actions


def decide_draining_followthrough(snapshot: HostMemorySnapshot) -> list[GovernanceAction]:
    """Return the resolution of existing draining marks on a host that is no longer under its RAM floor.

    Draining marks are *initiated* only under the danger floor (see :func:`decide_over_ceiling_reclaim`),
    but a mark gates job pops and shed restore until it resolves, and the pressure that placed it often
    clears before the drained process finishes its in-flight job (pausing pops and shedding footprint is
    the point of the degrade response). The mark must therefore resolve on a healthy host too: a marked
    process that fell back under the ceiling (or exited) is unmarked, a marked process that went idle is
    recycled (one per tick, largest first) so its allocator-retained RAM actually returns to the OS, and a
    marked process still busy over the ceiling keeps draining. New marks are never placed here.
    """
    if not snapshot.draining_process_ids:
        return []
    ceiling_mb = snapshot.per_process_ceiling_mb
    if ceiling_mb is None:
        return [ClearProcessDraining(process_id=process_id) for process_id in sorted(snapshot.draining_process_ids)]

    actions: list[GovernanceAction] = []
    slots_by_id = {slot.process_id: slot for slot in snapshot.inference_slots}
    recyclable = []
    for process_id in sorted(snapshot.draining_process_ids):
        slot = slots_by_id.get(process_id)
        if slot is None or slot.resident_ram_mb < ceiling_mb:
            actions.append(ClearProcessDraining(process_id=process_id))
        elif not slot.is_busy:
            recyclable.append(slot)
    if recyclable:
        target = max(recyclable, key=lambda slot: slot.resident_ram_mb)
        actions.append(
            RecycleProcess(
                process_id=target.process_id,
                resident_ram_mb=target.resident_ram_mb,
                ceiling_mb=ceiling_mb,
            ),
        )
    return actions


def decide_pressure_governance(snapshot: HostMemorySnapshot) -> list[GovernanceAction]:
    """Return the complete per-tick RAM governance for this snapshot.

    Always yields the soft pop-hold setting (engaged or cleared), then the full degrade response when the
    host is under its danger floor, or the draining follow-through when it is not (so a drain initiated
    under pressure still resolves after the floor clears). The restore of shed cards is decided separately
    by :func:`decide_shed_card_restore`, because it applies on a *healthy* host.
    """
    actions: list[GovernanceAction] = [decide_pop_hold(snapshot)]
    if snapshot.verdict.under_pressure:
        actions.extend(decide_degrade_response(snapshot))
    else:
        actions.extend(decide_draining_followthrough(snapshot))
    return actions


def decide_shed_restore(snapshot: HostMemorySnapshot) -> list[GovernanceAction]:
    """Return incremental RAM-pressure restore actions for card-scoped and worker-wide shedding.

    The reduction sheds idle contexts to walk the host back above its absolute RAM floor. Restore is
    conservative and RAM-gated: the host must be healthy, the self-throttle pause must have lapsed, no
    over-ceiling process may be draining, and each grant charges the per-context estimate against the
    measured restore headroom so one reading is not double-spent.
    """
    if not snapshot.shed_card_indices and snapshot.worker_shed_planned_process_count is None:
        return []
    if snapshot.verdict.under_pressure:
        return []
    if snapshot.pop_pause_active and snapshot.now < snapshot.pop_pause_until:
        return []
    if snapshot.draining_process_ids:
        return []

    actions: list[GovernanceAction] = []
    remaining_headroom_mb = snapshot.restore_headroom_mb

    worker_planned = snapshot.worker_shed_planned_process_count
    if worker_planned is not None:
        if snapshot.loaded_worker_process_count >= worker_planned or snapshot.worker_shed_process_count <= 0:
            actions.append(StopTrackingWorkerShed())
        elif remaining_headroom_mb >= snapshot.per_context_ram_estimate_mb:
            remaining_headroom_mb -= snapshot.per_context_ram_estimate_mb
            actions.append(
                RestoreWorkerProcess(
                    target_count=snapshot.loaded_worker_process_count + 1,
                    planned_count=worker_planned,
                ),
            )

    for device_index in sorted(snapshot.shed_card_indices):
        card = snapshot.card(device_index)
        if card is None or card.held_by_whole_card_residency:
            # A residency-held (or unknown) card is restored by its own path; stop tracking it here.
            actions.append(StopTrackingShedCard(device_index=device_index))
            continue
        if card.loaded_process_count >= card.planned_process_count:
            actions.append(StopTrackingShedCard(device_index=device_index))
            continue
        if remaining_headroom_mb < snapshot.per_context_ram_estimate_mb:
            # No RAM to sustain another resident context yet; keep the card pending and retry next tick.
            continue
        remaining_headroom_mb -= snapshot.per_context_ram_estimate_mb
        actions.append(
            RestoreCardProcess(
                device_index=device_index,
                target_count=card.loaded_process_count + 1,
                planned_count=card.planned_process_count,
            ),
        )
    return actions


def decide_shed_card_restore(snapshot: HostMemorySnapshot) -> list[GovernanceAction]:
    """Return the incremental restore of RAM-pressure shedding as RAM proves headroom.

    Backward-compatible public name retained for callers/tests; it now handles both the original multi-GPU
    card restore and the worker-wide single-GPU restore.
    """
    return decide_shed_restore(snapshot)
