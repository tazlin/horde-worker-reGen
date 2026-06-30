"""Unit tests for the capability dependency plan (topological ordering and edge validation)."""

from __future__ import annotations

import pytest

from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityKind
from horde_worker_regen.benchmark.capabilities.plan import build_plan
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.scenarios import CannedImageJobSpec, Scenario

_BASELINE = Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE)
_QUEUE = Capability(tier=BenchTier.SD15, kind=CapabilityKind.QUEUE_SIZE)
_BATCH2 = Capability(tier=BenchTier.SD15, kind=CapabilityKind.BATCH, magnitude=2)
_BATCH4 = Capability(tier=BenchTier.SD15, kind=CapabilityKind.BATCH, magnitude=4)


def _probe(capability: Capability, *requires: Capability) -> CapabilityProbe:
    """A minimal probe with the given prerequisites."""
    return CapabilityProbe(
        capability=capability,
        scenario=Scenario(name=capability.slug, image_jobs=[CannedImageJobSpec(count=1)]),
        requires=tuple(requires),
    )


def _order(probes: list[CapabilityProbe]) -> list[str]:
    return [probe.probe_id for probe in build_plan(probes).probes]


def test_orders_prerequisites_before_dependents() -> None:
    """A chain (baseline -> batch2 -> batch4) is ordered so each follows its prerequisite."""
    order = _order([_probe(_BATCH4, _BATCH2), _probe(_BATCH2, _BASELINE), _probe(_BASELINE)])
    assert order.index(_BASELINE.slug) < order.index(_BATCH2.slug) < order.index(_BATCH4.slug)


def test_preserves_catalog_order_among_independent_probes() -> None:
    """Probes that only depend on the baseline keep their input order (stable topological sort)."""
    probes = [_probe(_BASELINE), _probe(_QUEUE, _BASELINE), _probe(_BATCH2, _BASELINE)]
    assert _order(probes) == [_BASELINE.slug, _QUEUE.slug, _BATCH2.slug]


def test_dependents_of_returns_direct_children() -> None:
    """``dependents_of`` returns exactly the probes that directly require a capability."""
    plan = build_plan([_probe(_BASELINE), _probe(_QUEUE, _BASELINE), _probe(_BATCH2, _BASELINE)])
    dependents = {probe.probe_id for probe in plan.dependents_of(_BASELINE)}
    assert dependents == {_QUEUE.slug, _BATCH2.slug}


def test_missing_prerequisite_raises() -> None:
    """A probe requiring a capability no probe provides is a build-time error."""
    with pytest.raises(ValueError, match="which no probe provides"):
        build_plan([_probe(_QUEUE, _BASELINE)])


def test_cycle_raises() -> None:
    """Mutually dependent probes form a cycle and cannot be ordered."""
    with pytest.raises(ValueError, match="Cyclic"):
        build_plan([_probe(_BATCH2, _BATCH4), _probe(_BATCH4, _BATCH2)])


def test_duplicate_capability_raises() -> None:
    """Two probes claiming the same capability is a build-time error."""
    with pytest.raises(ValueError, match="Duplicate"):
        build_plan([_probe(_BASELINE), _probe(_BASELINE)])
