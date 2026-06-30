"""Order a set of probes by their dependency edges into a runnable plan.

The catalog declares probes with ``requires`` edges but no order; :func:`build_plan` topologically
sorts them so every probe follows its prerequisites, preserving catalog order as the tiebreak for a
stable, readable plan. The result is pure data: the CLI and TUI render the dependency DAG (and the
executor walks it) without booting anything.
"""

from __future__ import annotations

from pydantic import BaseModel

from horde_worker_regen.benchmark.capabilities.capability import Capability
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe


class CapabilityPlan(BaseModel):
    """An ordered list of probes plus the dependency edges they were ordered by."""

    probes: list[CapabilityProbe]
    """Topologically ordered: every probe appears after all of its prerequisites."""

    def dependents_of(self, capability: Capability) -> list[CapabilityProbe]:
        """Return the probes that directly require ``capability`` (its immediate children in the DAG)."""
        return [probe for probe in self.probes if capability in probe.requires]


def build_plan(probes: list[CapabilityProbe]) -> CapabilityPlan:
    """Topologically sort ``probes`` so each follows its prerequisites; raise on a missing edge or cycle.

    Catalog order is preserved wherever the dependencies allow it (Kahn's algorithm drains ready probes
    in their original order), so the plan reads conservative-to-demanding like the source catalog.

    Raises:
        ValueError: if a probe requires a capability no probe provides, or the edges form a cycle.
    """
    provider: dict[Capability, CapabilityProbe] = {}
    for probe in probes:
        if probe.capability in provider:
            raise ValueError(f"Duplicate probe for capability {probe.capability.slug!r}")
        provider[probe.capability] = probe

    for probe in probes:
        for required in probe.requires:
            if required not in provider:
                raise ValueError(
                    f"Probe {probe.probe_id!r} requires capability {required.slug!r}, which no probe provides",
                )

    unmet_count: dict[Capability, int] = {
        probe.capability: sum(1 for required in probe.requires if required in provider) for probe in probes
    }
    # Children indexed by prerequisite, so completing a probe can decrement exactly its dependents.
    children: dict[Capability, list[Capability]] = {probe.capability: [] for probe in probes}
    for probe in probes:
        for required in probe.requires:
            children[required].append(probe.capability)

    ready = [probe.capability for probe in probes if unmet_count[probe.capability] == 0]
    ordered: list[CapabilityProbe] = []
    while ready:
        current = ready.pop(0)
        ordered.append(provider[current])
        for child in children[current]:
            unmet_count[child] -= 1
            if unmet_count[child] == 0:
                ready.append(child)

    if len(ordered) != len(probes):
        unresolved = sorted(cap.slug for cap, count in unmet_count.items() if count > 0)
        raise ValueError(f"Cyclic capability dependencies among: {unresolved}")

    return CapabilityPlan(probes=ordered)


__all__ = ["CapabilityPlan", "build_plan"]
