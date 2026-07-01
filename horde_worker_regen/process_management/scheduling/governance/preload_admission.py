"""Pure decision functions for the preload admission pipeline.

The scheduler's preload loop walks the pending queue and decides, per job, whether to stage its model
onto an inference slot. The judgment calls in that pipeline live here as pure functions: which slots a
preload may not displace, whether the per-device load serialization gate blocks this cycle, which slot to
free for a starved head, in which order eligible cards should receive a fresh load, and what the
escalating VRAM/RAM reclamation resolves to once its reclaim attempts have run. The scheduler keeps the
measurement (budget verdicts, live process state) and the actions (evictions, process cycling); the
policy that turns those readings into an outcome is testable here without either.

Critical public members:

* [`compute_preload_disallowed_processes`]
  [horde_worker_regen.process_management.scheduling.governance.preload_admission.compute_preload_disallowed_processes]:
  the slots a preload may not target (queued-model guard, affinity protection, RAM draining).
* [`preload_concurrency_blocked`]
  [horde_worker_regen.process_management.scheduling.governance.preload_admission.preload_concurrency_blocked]:
  the per-device model-load serialization gate.
* ``decide_vram_reclaim_outcome`` / ``decide_ram_reclaim_outcome``: what an exhausted reclamation pass
  resolves to (defer, hold, reduce contexts, or best-effort admit).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, auto

from horde_worker_regen.process_management.scheduling.model_affinity import (
    affinity_active,
    compute_protected_processes,
)

__all__ = [
    "PreloadSlotSnapshot",
    "RamReclaimOutcome",
    "VramGateResult",
    "VramReclaimOutcome",
    "card_preload_order",
    "compute_preload_disallowed_processes",
    "decide_ram_reclaim_outcome",
    "decide_vram_reclaim_outcome",
    "preload_concurrency_blocked",
    "select_head_room_process_id",
]


@dataclass(frozen=True)
class PreloadSlotSnapshot:
    """Represents one inference slot's occupancy as the preload pipeline sees it."""

    process_id: int
    """The stable logical slot id."""
    model_name: str | None
    """The horde model resident on the slot, or None when it is empty."""
    can_accept_job: bool
    """Whether the slot can take work right now (idle and healthy)."""


def compute_preload_disallowed_processes(
    *,
    queued_model_process_ids: list[int],
    busy_process_ids: list[int],
    prefer_busy_only: bool,
    inference_process_models: Mapping[int, str | None],
    wanted_models: set[str],
    max_inference_processes: int,
    draining_process_ids: frozenset[int],
) -> set[int]:
    """Return the slots a preload may not displace.

    Three guards compose:

    1. The queued-model guard: slots holding a model another queued job needs (or, when fewer slots are
       loaded than there is work, only the busy slots, so idle slots stay reachable).
    2. Model-process affinity: in the models<=processes regime, the last resident copy of a still-wanted
       model is protected so a hot model's spare instance does not displace a cold model's only copy into
       a disk reload.
    3. RAM draining: a slot being drained for RAM reclaim is fed no new work so it can go idle and be
       recycled.

    The head-of-queue fallback (:func:`select_head_room_process_id`) deliberately overrides guards 1 and
    2 so a fully-protected pool can never starve the head.
    """
    disallowed = set(busy_process_ids) if prefer_busy_only else set(queued_model_process_ids)
    if affinity_active(len(wanted_models), max_inference_processes):
        disallowed |= compute_protected_processes(inference_process_models, wanted_models)
    disallowed |= draining_process_ids
    return disallowed


def preload_concurrency_blocked(
    *,
    num_preloading: int,
    max_concurrent_inference_processes: int,
    very_fast_disk_mode: bool,
) -> bool:
    """Return whether the per-device model-load serialization gate blocks another preload now.

    Loading two checkpoints onto one device at once stacks their disk-read and VRAM-allocation spikes, so
    by default one in-flight load blocks the next. ``very_fast_disk_mode`` relaxes the gate to the
    concurrent-inference ceiling plus one, for hosts whose storage can feed parallel loads.
    """
    if very_fast_disk_mode:
        return num_preloading >= max_concurrent_inference_processes + 1
    return num_preloading >= 1


def select_head_room_process_id(
    slots: tuple[PreloadSlotSnapshot, ...],
    *,
    in_progress_models: set[str | None],
    pending_models: set[str],
) -> int | None:
    """Return the slot to free for a starved head-of-queue job, or None when every slot is live.

    Deliberately overrides the affinity and queued-model guards (the head must make progress even when
    they protect every idle slot) while never displacing live work: only slots that can accept a job
    qualify, and never one whose model an in-progress job is using. Prefers the cheapest displacement: an
    empty slot, then a resident model no pending job needs, then, last, a merely-queued model.
    """
    candidates = [slot for slot in slots if slot.can_accept_job and slot.model_name not in in_progress_models]
    if not candidates:
        return None

    def _displacement_cost(slot: PreloadSlotSnapshot) -> int:
        if slot.model_name is None:
            return 0
        if slot.model_name not in pending_models:
            return 1
        return 2

    return min(candidates, key=_displacement_cost).process_id


def card_preload_order(
    eligible_card_indices: set[int],
    *,
    cards_already_serving_model: set[int],
    card_busy_counts: Mapping[int, int],
) -> list[int]:
    """Return the eligible cards in placement preference order for a fresh model load.

    The same sticky-then-least-loaded policy dispatch uses: a card already holding this model first
    (avoid a duplicate copy), then the card running the fewest inference jobs (balance fresh loads).
    """

    def _placement_key(device_index: int) -> tuple[int, int]:
        already_serves = 0 if device_index in cards_already_serving_model else 1
        return (already_serves, card_busy_counts.get(device_index, 0))

    return sorted(eligible_card_indices, key=_placement_key)


class VramGateResult(StrEnum):
    """The VRAM gate's answer for one preload attempt."""

    FITS = auto()
    """The predicted peak fits measured free VRAM: proceed to the RAM gate."""
    DEFER = auto()
    """Does not fit and reclaim is the answer this tick: defer the preload."""
    ADMIT_OVER_BUDGET = auto()
    """Admitted best-effort past an exhausted reclamation: skip the RAM gate (its reclaim already ran)."""


class VramReclaimOutcome(StrEnum):
    """What an exhausted VRAM reclamation pass resolves to for the job that could not fit."""

    DEFER = auto()
    """Wait: something was freed, the job is not the head blocker, or a live job still holds the device."""
    HOLD_UNSERVABLE = auto()
    """The model keeps faulting over the budget (circuit breaker): hold it back, do not admit."""
    REDUCE_CONTEXTS = auto()
    """Live process contexts are the over-commit: establish a whole-card reduction and defer."""
    ADMIT_DECLINING_REDUCTION = auto()
    """A reduction was demanded on untrusted overhead figures: decline it and admit best-effort instead."""
    BEST_EFFORT_ADMIT = auto()
    """Reclamation is structurally exhausted and nothing live holds the device: admit over-budget."""


def decide_vram_reclaim_outcome(
    *,
    freed: bool,
    is_head_blocker: bool,
    no_live_resource_consumer: bool,
    model_unservable: bool,
    context_reduction_demanded: bool,
    whole_card_warranted: bool,
) -> VramReclaimOutcome:
    """Return what the VRAM branch does after its reclaim attempts have run.

    The escalation policy in one place: a preload whose predicted peak does not fit defers while reclaim
    makes progress or a live job holds the device; only a head-of-queue blocker on an otherwise-idle
    device may go further, and then in order of preference: hold a breaker-tripped model, reduce the live
    context count when contexts (not models) are the over-commit and the overhead figures warrant it, or
    admit the head best-effort rather than wedge the queue.
    """
    if freed or not is_head_blocker or not no_live_resource_consumer:
        return VramReclaimOutcome.DEFER
    if model_unservable:
        return VramReclaimOutcome.HOLD_UNSERVABLE
    if context_reduction_demanded and whole_card_warranted:
        return VramReclaimOutcome.REDUCE_CONTEXTS
    if context_reduction_demanded:
        return VramReclaimOutcome.ADMIT_DECLINING_REDUCTION
    return VramReclaimOutcome.BEST_EFFORT_ADMIT


class RamReclaimOutcome(StrEnum):
    """What an exhausted system-RAM reclamation pass resolves to for the job that could not fit."""

    DEFER = auto()
    """Wait: RAM was reclaimed (or a stuck slot was cycled to reclaim it), or a live job holds memory."""
    BEST_EFFORT_ADMIT = auto()
    """Nothing reclaimable remains and nothing live holds memory: admit the head rather than starve it."""


def decide_ram_reclaim_outcome(
    *,
    reclaimed: bool,
    cycled_stale_slot: bool,
    is_head_blocker: bool,
    no_live_resource_consumer: bool,
) -> RamReclaimOutcome:
    """Return what the RAM branch does after its reclaim attempts have run.

    Mirrors the VRAM policy's final rung: reclaim progress (an eviction, or cycling an allocator-stuck
    idle slot) is always worth waiting for, and only a head-of-queue blocker with no live job holding
    memory may be admitted best-effort once nothing more can be reclaimed.
    """
    if reclaimed or cycled_stale_slot:
        return RamReclaimOutcome.DEFER
    if is_head_blocker and no_live_resource_consumer:
        return RamReclaimOutcome.BEST_EFFORT_ADMIT
    return RamReclaimOutcome.DEFER
