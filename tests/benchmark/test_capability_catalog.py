"""Shape tests for the capability catalog: dependencies, exclusions, and tier coverage."""

from __future__ import annotations

from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityKind
from horde_worker_regen.benchmark.capabilities.catalog import (
    CatalogOptions,
    build_capability_catalog,
    build_sustained_probe,
)
from horde_worker_regen.benchmark.capabilities.plan import build_plan
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.report import SuggestedBridgeData


def _kinds(probes: list, tier: BenchTier) -> set[CapabilityKind]:
    return {probe.capability.kind for probe in probes if probe.capability.tier is tier}


def test_baseline_is_first_and_establishes_the_reference() -> None:
    """Each tier leads with its baseline probe, which establishes the it/s reference and requires nothing."""
    probes = build_capability_catalog(CatalogOptions(tiers=[BenchTier.SD15]))
    baseline = probes[0]
    assert baseline.capability.kind is CapabilityKind.BASELINE
    assert baseline.establishes_baseline
    assert baseline.requires == ()


def test_every_non_baseline_probe_depends_on_something() -> None:
    """No non-baseline probe is a root: each names a prerequisite, so the cascade always has an anchor."""
    probes = build_capability_catalog(CatalogOptions(tiers=[BenchTier.SD15], include_downloads=True))
    for probe in probes:
        if probe.capability.kind is CapabilityKind.BASELINE:
            continue
        assert probe.requires, f"{probe.probe_id} has no prerequisites"


def test_full_catalog_forms_a_valid_dag() -> None:
    """The whole catalog topologically sorts: no dangling prerequisite, no cycle, no duplicate."""
    probes = build_capability_catalog(
        CatalogOptions(tiers=[BenchTier.SD15, BenchTier.SDXL], include_downloads=True),
    )
    plan = build_plan(probes)
    assert len(plan.probes) == len(probes)


def test_batch_and_post_processing_rungs_chain() -> None:
    """The higher rungs build on the lower ones: batch 4 requires batch 2, pp-resolutions requires the sweep."""
    probes = build_capability_catalog(CatalogOptions(tiers=[BenchTier.SD15]))
    by_id = {probe.probe_id: probe for probe in probes}

    batch2 = Capability(tier=BenchTier.SD15, kind=CapabilityKind.BATCH, magnitude=2)
    batch4 = by_id["sd15-batch-4"]
    assert batch2 in batch4.requires

    sweep = Capability(tier=BenchTier.SD15, kind=CapabilityKind.POST_PROCESSING)
    resolutions = next(
        probe
        for probe in probes
        if probe.capability.kind is CapabilityKind.POST_PROCESSING and probe.capability.magnitude
    )
    assert sweep in resolutions.requires


def test_excluding_a_kind_drops_its_probes() -> None:
    """An excluded kind (and therefore all its rungs) is absent from the catalog."""
    probes = build_capability_catalog(
        CatalogOptions(tiers=[BenchTier.SD15], excluded_kinds={CapabilityKind.POST_PROCESSING}),
    )
    assert CapabilityKind.POST_PROCESSING not in _kinds(probes, BenchTier.SD15)


def test_controlnet_is_sd15_only_and_qr_code_is_on_both() -> None:
    """Classic controlnet is SD1.5-only; the QR-code workflow is the SDXL controlnet capability too."""
    probes = build_capability_catalog(CatalogOptions(tiers=[BenchTier.SD15, BenchTier.SDXL]))
    assert CapabilityKind.CONTROLNET in _kinds(probes, BenchTier.SD15)
    assert CapabilityKind.CONTROLNET not in _kinds(probes, BenchTier.SDXL)
    assert CapabilityKind.QR_CODE in _kinds(probes, BenchTier.SD15)
    assert CapabilityKind.QR_CODE in _kinds(probes, BenchTier.SDXL)


def test_sustained_probe_carries_requires_and_a_soak_scenario() -> None:
    """The soak probe is the SUSTAINED capability, honours its passed prerequisites, and is a soak."""
    baseline = Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE)
    probe = build_sustained_probe(
        SuggestedBridgeData(),
        BenchTier.SD15,
        soak_seconds=30.0,
        requires=(baseline,),
    )
    assert probe.capability.kind is CapabilityKind.SUSTAINED
    assert probe.requires == (baseline,)
    assert probe.scenario.soak_seconds == 30.0
