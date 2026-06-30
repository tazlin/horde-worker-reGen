"""Rendering tests for the capability report markdown (pure, no harness)."""

from __future__ import annotations

from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityKind, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.recommendation import synthesize_bridge_data, synthesize_capabilities
from horde_worker_regen.benchmark.capabilities.report_render import describe_decision, render_markdown
from horde_worker_regen.benchmark.capabilities.result import (
    CapabilityProbeResult,
    CapabilityReport,
    Finding,
    MachineInfo,
    SuggestionBasis,
    SuggestionDecision,
)
from horde_worker_regen.benchmark.criteria import LevelStats
from horde_worker_regen.benchmark.enums import BenchTier, FindingKind


def _proven(kind: CapabilityKind, *, tier: BenchTier = BenchTier.SD15, magnitude: int = 0) -> CapabilityProbeResult:
    return CapabilityProbeResult(
        capability=Capability(tier=tier, kind=kind, magnitude=magnitude),
        verdict=CapabilityVerdict.PROVEN,
        stats=LevelStats(its_p50=6.0, vram_used_high_water_mb=4000),
    )


def test_render_markdown_includes_every_section() -> None:
    """A populated report renders the capability table, the recommendation, provenance, and remediation."""
    probes = [
        _proven(CapabilityKind.BASELINE),
        _proven(CapabilityKind.THREADS, magnitude=2),
        _proven(CapabilityKind.POST_PROCESSING),
    ]
    report = CapabilityReport(
        machine=MachineInfo(gpu_name="Test GPU", total_vram_mb=16000, total_ram_bytes=32 * 1024**3),
        probes=probes,
        capabilities=synthesize_capabilities(probes, total_vram_mb=16000),
        suggested_bridge_data=synthesize_bridge_data(probes, total_vram_mb=16000),
    )

    markdown = render_markdown(report)

    assert "# Worker Benchmark Report" in markdown
    assert "Test GPU" in markdown
    assert "## Capabilities" in markdown
    assert "## Probes" in markdown
    assert "sd15-baseline" in markdown
    assert "## Suggested bridgeData" in markdown
    assert "Why each value (provenance):" in markdown
    assert "No robustness findings" in markdown


def test_render_markdown_lists_findings_with_reproduce_hint() -> None:
    """A finding is rendered in the remediation queue with a slug-targeted reproduce command."""
    probe = CapabilityProbeResult(
        capability=Capability(tier=BenchTier.SD15, kind=CapabilityKind.CONTROLNET),
        verdict=CapabilityVerdict.CRASHED,
        findings=[Finding(kind=FindingKind.CRASH, level_id="sd15-controlnet", evidence="worker died")],
    )
    report = CapabilityReport(probes=[probe])

    markdown = render_markdown(report)

    assert "## Remediation queue" in markdown
    assert "sd15-controlnet" in markdown
    assert "horde-benchmark run --only sd15-controlnet" in markdown


def test_describe_decision_glosses_the_basis() -> None:
    """describe_decision renders a value, a plain-language basis label, and the detail."""
    value_text, label, detail = describe_decision(
        SuggestionDecision(
            setting="allow_lora",
            value=False,
            basis=SuggestionBasis.UNTESTED_SKIPPED,
            detail="never tested",
        ),
    )
    assert value_text == "off"
    assert "never tested (skipped)" in label
    assert detail == "never tested"
