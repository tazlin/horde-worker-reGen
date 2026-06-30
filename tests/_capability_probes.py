"""The shared capability catalog the e2e (fake) and gpu (real) probe tests parametrize over.

Built once here so a benchmark scenario is literally runnable as a test: ``pytest -m gpu -k controlnet``
selects exactly the ``sd15-controlnet`` probe, because the parametrize id is the capability slug. The
e2e suite runs a small, cheap subset in fake mode (CI, no GPU); the gpu suite runs the whole catalog on
real hardware, self-skipping the probes the machine cannot host.
"""

from __future__ import annotations

from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind
from horde_worker_regen.benchmark.capabilities.catalog import CatalogOptions, build_capability_catalog
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.enums import BenchTier

ALL_PROBES: list[CapabilityProbe] = build_capability_catalog(
    CatalogOptions(tiers=[BenchTier.SD15, BenchTier.SDXL], jobs_per_level=2),
)
"""Every static probe across the two conservative tiers, for the real-hardware gpu suite."""

_LIGHT_KINDS: frozenset[CapabilityKind] = frozenset(
    {CapabilityKind.BASELINE, CapabilityKind.QUEUE_SIZE, CapabilityKind.THREADS},
)
"""The cheap, model-light SD1.5 shapes the fake worker reliably completes (no feature/alchemy weights)."""

LIGHT_PROBES: list[CapabilityProbe] = [
    probe
    for probe in build_capability_catalog(
        CatalogOptions(
            tiers=[BenchTier.SD15],
            jobs_per_level=2,
            include_features=False,
            include_alchemy=False,
        ),
    )
    if probe.capability.kind in _LIGHT_KINDS
]
"""A bounded SD1.5 subset (baseline, queue depth, threads) for the fake-mode CI e2e suite."""
