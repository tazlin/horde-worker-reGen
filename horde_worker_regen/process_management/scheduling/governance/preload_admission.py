"""Pure decision functions for the preload admission pipeline.

The scheduler's preload loop walks the pending queue and decides, per job, whether to stage its model
onto an inference slot. The judgment calls in that pipeline live here as pure functions: which slots a
preload may not displace, whether the per-device load serialization gate blocks this cycle, which slot to
free for a starved head, in which order eligible cards should receive a fresh load, and what the
system-RAM reclamation resolves to once its reclaim attempts have run. The scheduler keeps the
measurement (budget verdicts, live process state) and the actions (evictions, process cycling); the
policy that turns those readings into an outcome is testable here without either.

Critical public members:

* [`compute_preload_disallowed_processes`]
  [horde_worker_regen.process_management.scheduling.governance.preload_admission.compute_preload_disallowed_processes]:
  the slots a preload may not target (queued-model guard, affinity protection, RAM draining).
* [`preload_concurrency_blocked`]
  [horde_worker_regen.process_management.scheduling.governance.preload_admission.preload_concurrency_blocked]:
  the per-device model-load serialization gate.
* ``decide_ram_reclaim_outcome``: what an exhausted system-RAM reclamation pass resolves to.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Protocol

from horde_worker_regen.process_management.scheduling.model_affinity import (
    affinity_active,
    compute_protected_processes,
)

__all__ = [
    "AdmissionDecision",
    "AdmissionResult",
    "PreloadSlotSnapshot",
    "ReclamationExecutor",
    "RamReclaimOutcome",
    "card_preload_order",
    "compute_preload_disallowed_processes",
    "decide_ram_reclaim_outcome",
    "preload_concurrency_blocked",
    "select_head_room_process_id",
]


class AdmissionDecision(StrEnum):
    """The preload pass decision for one pending job."""

    ADMIT = auto()
    """The job may send a preload now."""
    NEXT_JOB = auto()
    """Skip this job and continue scanning the pending queue."""
    STOP_PASS = auto()
    """Stop the preload pass for this scheduling cycle."""
    QUARANTINED = auto()
    """The model is quarantined and the job should be faulted/reissued before continuing."""
    UNSERVICEABLE = auto()
    """The model's minimum device footprint cannot fit any serving card, so the job is faulted for reissue."""
    ALREADY_LOADED = auto()
    """The model is already resident or loading, so no preload is needed."""
    DEFER_RAM_PRESSURE = auto()
    """The absolute host-RAM floor blocks a new preload this cycle."""
    DEFER_VRAM_GROWTH_HOLD = auto()
    """The device-free governor is holding new VRAM growth on the target card this cycle: device-level free
    VRAM is below the soft floor, so bringing another model to the card would grow a footprint already near
    the WDDM paging cliff. Deferred until the card recovers (foreign pressure eases, or reclaim frees room)."""
    EXCLUSIVE_IN_PROGRESS = auto()
    """An exclusive over-budget job is in progress and suppresses unrelated staging."""
    NO_TARGET = auto()
    """No suitable inference slot is available for this preload."""
    REPLACE_PROCESS = auto()
    """The chosen slot should be cycled before the model change is attempted again."""
    DEFER_CONCURRENCY = auto()
    """The per-device preload concurrency gate is closed."""
    DEFER_BUDGET = auto()
    """The resource budget or reclamation ladder deferred the preload."""
    PRESTAGE = auto()
    """A whole-card head should be pre-staged into RAM before it samples."""


@dataclass(frozen=True)
class AdmissionResult:
    """Result returned by a preload-admission stage."""

    decision: AdmissionDecision
    """The gate's decision."""
    reason: str = ""
    """Optional human-readable reason for logs or tests."""
    process_id: int | None = None
    """Optional target process id when the decision selected a slot."""


class ReclamationExecutor(Protocol):
    """Side-effect surface for the same-tick preload reclamation ladder.

    The pure policy in this module decides which rung should win after attempts run; the scheduler owns
    these operations because they touch process state, lifecycle replacement, and operator logging.
    """

    def reclaim_idle_vram(self, *, for_head_of_queue: bool) -> bool:
        """Try to free idle VRAM and return whether anything was reclaimed."""
        ...

    def reclaim_idle_ram(self, *, for_head_of_queue: bool) -> bool:
        """Try to free idle resident RAM and return whether anything was reclaimed."""
        ...

    def cycle_stale_ram_slot(self) -> bool:
        """Cycle an allocator-stuck idle slot and return whether a cycle was started."""
        ...

    def reduce_contexts_for_head(self) -> None:
        """Start the context-reduction remedy for a head whose live contexts are the over-commit."""
        ...

    def admit_head_best_effort(self) -> None:
        """Record that the head is being admitted over budget after reclamation is exhausted."""
        ...


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
