"""Immutable inputs to the resource-governance decision functions.

A snapshot captures everything one governance decision needs at a single instant: the RAM danger-floor
verdict, per-process and per-card footprint readings, and the governor's own multi-tick bookkeeping
(draining processes, shed cards) as frozen values. Decision functions in
[`ram_governor`][horde_worker_regen.process_management.scheduling.governance.ram_governor] are pure
functions of a snapshot, so tests construct one directly and assert on the returned actions without a
scheduler, a process map, or monkeypatching.

Critical public classes:

* [`InferenceSlotSnapshot`]
  [horde_worker_regen.process_management.scheduling.governance.snapshots.InferenceSlotSnapshot]:
  one inference process's memory-relevant state.
* [`CardProcessSnapshot`][horde_worker_regen.process_management.scheduling.governance.snapshots.CardProcessSnapshot]:
  one driven card's inference-process counts against its plan.
* [`HostMemorySnapshot`][horde_worker_regen.process_management.scheduling.governance.snapshots.HostMemorySnapshot]:
  the complete host-RAM picture one governance tick decides over.
"""

from __future__ import annotations

from dataclasses import dataclass

from horde_worker_regen.process_management.resources.resource_budget import RamPressureVerdict

__all__ = [
    "CardProcessSnapshot",
    "HostMemorySnapshot",
    "InferenceSlotSnapshot",
]


@dataclass(frozen=True)
class InferenceSlotSnapshot:
    """Represents one inference process's memory-relevant state at snapshot time."""

    process_id: int
    """The stable logical slot id of the inference process (not an OS pid)."""
    device_index: int
    """The stable index of the GPU the process is pinned to."""
    resident_ram_mb: float
    """The process's measured resident system RAM (MB) from its most recent report."""
    is_busy: bool
    """Whether the process is mid-job (a busy process is drained, never stopped outright)."""


@dataclass(frozen=True)
class CardProcessSnapshot:
    """Represents one driven card's inference-process counts against its planned target."""

    device_index: int
    """The stable index of the card."""
    loaded_process_count: int
    """How many inference processes are currently resident on this card."""
    busy_process_count: int
    """How many of this card's inference processes are currently mid-job."""
    planned_process_count: int
    """The per-card process count the startup plan sized this card for."""
    held_by_whole_card_residency: bool = False
    """Whether a whole-card exclusive residency is deliberately holding this card's count down."""


@dataclass(frozen=True)
class HostMemorySnapshot:
    """Represents the host-RAM state and governor bookkeeping one governance tick decides over.

    All readings are captured once, before deciding: the decision functions never re-measure, so a
    single snapshot yields a single, internally-consistent set of actions.
    """

    verdict: RamPressureVerdict
    """Whether the host is below its absolute system-RAM danger floor, with the measured figures."""
    now: float
    """Wall-clock time (``time.time()``) the snapshot was taken."""
    pop_pause_active: bool
    """Whether the hard self-throttle pop pause is currently armed."""
    pop_pause_until: float
    """Wall-clock time the armed pop pause lapses (meaningless when not armed)."""
    pop_hold_margin_mb: float
    """How close (MB) available RAM may approach the danger floor before the soft pop hold engages."""
    per_process_ceiling_mb: float | None
    """The per-process resident-RAM ceiling (MB), or None when the ceiling is disabled."""
    multi_gpu_routing_active: bool
    """Whether the worker routes jobs across more than one card (per-card remedies apply)."""
    in_flight_job_count: int
    """How many jobs are currently in progress across the worker."""
    loaded_worker_process_count: int
    """How many inference processes are resident worker-wide (the single-GPU reduction input)."""
    inference_slots: tuple[InferenceSlotSnapshot, ...]
    """Every live inference process's memory-relevant state."""
    cards: tuple[CardProcessSnapshot, ...]
    """Every driven card's process counts against its plan."""
    draining_process_ids: frozenset[int]
    """Process ids currently marked draining for RAM reclaim (the governor's own multi-tick state)."""
    shed_card_indices: frozenset[int]
    """Device indices the RAM-pressure reduction shed below plan and not yet restored."""
    restore_headroom_mb: float
    """Measured RAM headroom (MB) above the reserve and committed reserves, for restore gating."""
    per_context_ram_estimate_mb: float
    """Conservative resident-RAM cost (MB) of one more inference context, for restore gating."""

    def card(self, device_index: int) -> CardProcessSnapshot | None:
        """Return the snapshot for ``device_index``, or None when the card is not driven."""
        for card_snapshot in self.cards:
            if card_snapshot.device_index == device_index:
                return card_snapshot
        return None
