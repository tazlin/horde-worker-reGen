"""The per-tick resource-governance entry point.

[`ResourceGovernor`][horde_worker_regen.process_management.scheduling.governance.governor.ResourceGovernor]
owns the measure/decide/execute cycle for host resources and the multi-tick governor bookkeeping. The
scheduling loop calls
[`tick`][horde_worker_regen.process_management.scheduling.governance.governor.ResourceGovernor.tick]
unconditionally at the top of every cycle, so governance never depends on any particular scheduling path
(a preload attempt, a dispatch) happening to execute: a steady-state worker that never loads a new model
is governed exactly as often as one that does.

The governor does not measure or act itself: its host (the scheduler) provides the snapshot and executes
the returned actions, keeping this module free of process-map and lifecycle dependencies.
"""

from __future__ import annotations

from typing import Protocol

from horde_worker_regen.process_management.resources.resource_budget import RamPressureVerdict
from horde_worker_regen.process_management.scheduling.governance.actions import GovernanceAction
from horde_worker_regen.process_management.scheduling.governance.ram_governor import (
    RamGovernorState,
    decide_pressure_governance,
    decide_shed_restore,
)
from horde_worker_regen.process_management.scheduling.governance.snapshots import HostMemorySnapshot

__all__ = [
    "GovernanceHost",
    "ResourceGovernor",
]


class GovernanceHost(Protocol):
    """The measurement and execution surface a governor's host provides.

    Implemented by the inference scheduler: it owns the live process map, worker state, and lifecycle
    manager, so measuring a snapshot and executing remedies belong to it. The governor stays a pure
    orchestration of decide functions over that surface.
    """

    def _ram_pressure_verdict(self) -> RamPressureVerdict:
        """Assess whether the host is below its absolute system-RAM danger floor right now."""
        ...

    def _build_host_memory_snapshot(self, verdict: RamPressureVerdict) -> HostMemorySnapshot:
        """Capture the host-RAM state and governor bookkeeping one governance decision runs over."""
        ...

    def _execute_governance_actions(self, actions: list[GovernanceAction]) -> None:
        """Execute governance decisions against the live worker."""
        ...


class ResourceGovernor:
    """Owns per-tick resource governance: measure once, decide purely, execute through the host.

    One tick covers both regimes: on a pressured host it applies the degrade response (pop pause and
    hold, idle-model eviction, footprint reduction, over-ceiling reclaim); on a healthy host it restores
    what a past pressure episode shed. The multi-tick bookkeeping
    ([`RamGovernorState`][horde_worker_regen.process_management.scheduling.governance.ram_governor.RamGovernorState])
    lives here rather than on the scheduler.

    Thread Safety:
        Driven exclusively by the scheduler's control loop; not safe for concurrent use.
    """

    def __init__(self, host: GovernanceHost) -> None:
        """Bind the governor to the host that measures snapshots and executes actions for it.

        Args:
            host: The measurement/execution surface (the inference scheduler).
        """
        self._host = host
        self.ram_state = RamGovernorState()
        """The RAM governor's multi-tick bookkeeping (shed cards, draining processes)."""
        self.last_ram_verdict: RamPressureVerdict | None = None
        """The danger-floor verdict measured by the most recent tick, or None before the first tick.

        Per-job gates within the same scheduling cycle read this instead of re-measuring, so one cycle
        acts on one consistent reading.
        """

    def tick(self) -> bool:
        """Run one governance tick and return whether the host is under RAM pressure.

        Measures the danger-floor verdict and one snapshot, then decides and executes the complete
        response: the soft pop hold (always), the degrade response (pressured host), and the shed-card
        restore (recovered host). The two regimes are mutually exclusive by construction, so a single
        combined execution never both sheds and restores.
        """
        verdict = self._host._ram_pressure_verdict()
        self.last_ram_verdict = verdict
        snapshot = self._host._build_host_memory_snapshot(verdict)
        actions = decide_pressure_governance(snapshot)
        actions.extend(decide_shed_restore(snapshot))
        self._host._execute_governance_actions(actions)
        return verdict.under_pressure
