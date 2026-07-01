"""Typed remedy commands the governance decision functions return.

Each action names one concrete remedy and carries the values its execution (and its log line) needs.
Decisions return ``list[GovernanceAction]``; the scheduler owns a single dispatcher that executes them
against the live process map, lifecycle manager, and worker state. Keeping the actions as inert values
separates deciding from acting: tests assert on the returned actions, and every side effect has exactly
one execution site.

The multi-tick bookkeeping actions (draining marks, shed-card tracking) mutate
[`RamGovernorState`][horde_worker_regen.process_management.scheduling.governance.ram_governor.RamGovernorState]
at execution time, so the decision layer stays a pure function of its snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ClearProcessDraining",
    "EvictIdleModels",
    "GovernanceAction",
    "MarkProcessDraining",
    "PausePops",
    "RecycleProcess",
    "ReduceCardProcesses",
    "ReduceWorkerProcesses",
    "RestoreCardProcess",
    "RestoreWorkerProcess",
    "SetPopHold",
    "StopTrackingShedCard",
    "StopTrackingWorkerShed",
]


@dataclass(frozen=True)
class PausePops:
    """Arm (or extend) the hard self-throttle pop pause while the host is under its RAM danger floor."""

    until_time: float
    """Wall-clock time the pause lapses (intake auto-resumes)."""
    pause_seconds: float
    """The pause length, for the announcement."""
    reason: str
    """A short human-readable pressure reason, for the announcement."""


@dataclass(frozen=True)
class SetPopHold:
    """Set or clear the soft, pre-floor pop hold that stops new job ttl clocks starting."""

    active: bool
    """Whether the hold is engaged."""


@dataclass(frozen=True)
class EvictIdleModels:
    """Evict one idle resident model from RAM (the residency grace is dropped under pressure)."""


@dataclass(frozen=True)
class ReduceWorkerProcesses:
    """Shed idle inference processes toward a worker-wide target count (the single-GPU reduction)."""

    target_count: int
    """The worker-wide process count to reduce toward (never below one)."""
    planned_count: int = 0
    """The planned worker-wide process count, for later restoration."""
    pressure_shortfall_mb: float | None = None
    """How far below the RAM floor the host is, used to choose the smallest sufficient idle victim."""


@dataclass(frozen=True)
class ReduceCardProcesses:
    """Shed idle inference processes on one card toward a per-card target count.

    Execution records the card as shed when the count actually fell, so the restore path knows which
    cards to grow back once the host recovers.
    """

    device_index: int
    """The card to reduce."""
    target_count: int
    """The per-card process count to reduce toward (never below one)."""


@dataclass(frozen=True)
class MarkProcessDraining:
    """Mark a busy over-ceiling process draining: fed no new work so it can be recycled once idle."""

    process_id: int
    """The process to drain."""
    resident_ram_mb: float
    """The process's measured resident RAM (MB), for the announcement."""
    ceiling_mb: float
    """The per-process ceiling (MB) it crossed, for the announcement."""


@dataclass(frozen=True)
class ClearProcessDraining:
    """Clear a process's draining mark (it fell back under the ceiling, or the ceiling was disabled)."""

    process_id: int
    """The process to clear."""


@dataclass(frozen=True)
class RecycleProcess:
    """Recycle an idle over-ceiling process so its allocator-retained pages return to the OS."""

    process_id: int
    """The process to recycle."""
    resident_ram_mb: float
    """The process's measured resident RAM (MB), for the announcement."""
    ceiling_mb: float
    """The per-process ceiling (MB) it crossed, for the announcement."""


@dataclass(frozen=True)
class RestoreCardProcess:
    """Grow one shed card back by one inference context now that RAM has proven headroom for it.

    Execution stops tracking the card once it reaches its planned count.
    """

    device_index: int
    """The card to grow."""
    target_count: int
    """The process count to grow to (one more than currently resident)."""
    planned_count: int
    """The card's planned per-card process count, for the announcement and tracking cutoff."""


@dataclass(frozen=True)
class RestoreWorkerProcess:
    """Grow the worker-wide pool by one inference context after RAM pressure recovers."""

    target_count: int
    """The worker-wide process count to grow to (one more than currently resident)."""
    planned_count: int
    """The planned worker-wide process count, for the announcement and tracking cutoff."""


@dataclass(frozen=True)
class StopTrackingShedCard:
    """Stop tracking a shed card whose restoration belongs to another path (or is already complete)."""

    device_index: int
    """The card to stop tracking."""


@dataclass(frozen=True)
class StopTrackingWorkerShed:
    """Stop tracking worker-wide RAM-pressure shedding once it is restored or stale."""


type GovernanceAction = (
    PausePops
    | SetPopHold
    | EvictIdleModels
    | ReduceWorkerProcesses
    | ReduceCardProcesses
    | MarkProcessDraining
    | ClearProcessDraining
    | RecycleProcess
    | RestoreCardProcess
    | RestoreWorkerProcess
    | StopTrackingShedCard
    | StopTrackingWorkerShed
)
"""The tagged union of every remedy a governance decision can return."""
