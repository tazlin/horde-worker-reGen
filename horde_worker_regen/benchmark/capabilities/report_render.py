"""Render a :class:`CapabilityReport` to the human-readable markdown an operator reads after a run.

The capability-engine port of ``report.render_markdown``: the same report shape (capability surface,
suggested bridgeData with per-setting provenance, sustained-load soak, per-job time breakdown, and the
remediation queue), read off ``report.probes`` (capability + verdict) instead of the old ``levels``.
Kept separate from :mod:`recommendation` so the pure synthesis has no rendering concern, and torch-free
so any surface can render a stored report without booting anything.
"""

from __future__ import annotations

from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind
from horde_worker_regen.benchmark.capabilities.recommendation import verify_suggestion_consistency
from horde_worker_regen.benchmark.capabilities.result import (
    CapabilityReport,
    SuggestionBasis,
    SuggestionDecision,
    WorkerCapabilities,
)
from horde_worker_regen.process_management.resources.duty_cycle import PHASE_ORDER

_BASIS_LABELS: dict[SuggestionBasis, str] = {
    SuggestionBasis.PROVEN: "proven (a probe passed)",
    SuggestionBasis.DISABLED_FAILED: "off: tested and failed",
    SuggestionBasis.UNTESTED_SKIPPED: "off: never tested (skipped)",
    SuggestionBasis.NOT_IN_LADDER: "off: not in this run",
    SuggestionBasis.CAPPED_VRAM: "held back: VRAM headroom",
    SuggestionBasis.CAPPED_SOAK: "held back: soak unstable",
}
"""Plain-language gloss for each basis, shared by the markdown report and the TUI."""


def _format_setting_value(value: bool | int | list[str]) -> str:
    """Render a decision value compactly for the provenance table."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return ", ".join(value) if value else "(none)"
    return str(value)


def describe_decision(decision: SuggestionDecision) -> tuple[str, str, str]:
    """Return a decision's (value_text, plain-language basis label, detail) for display surfaces.

    Shared by the markdown report and the live progress projection so the TUI and the report agree on how
    a basis reads, without either reimplementing the gloss.
    """
    label = _BASIS_LABELS.get(decision.basis, decision.basis.value)
    return _format_setting_value(decision.value), label, decision.detail or ""


def _render_capabilities(capabilities: WorkerCapabilities) -> list[str]:
    """Render the capability surface block (what the worker proved it can do)."""
    lines = ["## Capabilities (what this worker can do)", ""]
    lines.append("| Tier | Model | Baseline | it/s p50 | Max stable batch | Peak VRAM (MB) | Fits with headroom |")
    lines.append("|---|---|---|---|---|---|---|")
    for tier_capability in capabilities.tiers:
        its = f"{tier_capability.observed_its_p50:.2f}" if tier_capability.observed_its_p50 is not None else "-"
        peak = str(tier_capability.peak_vram_mb) if tier_capability.peak_vram_mb is not None else "-"
        baseline = "pass" if tier_capability.baseline_passed else "fail"
        fits = "yes" if tier_capability.fits_with_headroom else "no"
        lines.append(
            f"| {tier_capability.tier} | {tier_capability.model_name} | {baseline} | {its} | "
            f"{tier_capability.max_stable_batch} | {peak} | {fits} |",
        )
    lines.append("")
    features = [
        ("hires-fix", capabilities.supports_hires_fix),
        ("post-processing", capabilities.supports_post_processing),
        ("controlnet (SD1.5)", capabilities.supports_controlnet),
        ("qr-code workflow (SDXL controlnet)", capabilities.supports_qr_code),
        ("alchemy CLIP lane", capabilities.supports_alchemy_clip),
        ("alchemy graph lane", capabilities.supports_alchemy_graph),
        ("concurrent alchemy", capabilities.supports_alchemy_concurrent),
        ("ad-hoc lora", capabilities.supports_lora),
    ]
    for name, supported in features:
        lines.append(f"- {'✓' if supported else '✗'} {name}")
    lines.append("")
    return lines


def _render_provenance(decisions: list[SuggestionDecision]) -> list[str]:
    """Render the per-setting provenance table: why each suggested value holds.

    The point is to let an operator tell a capability that was *proven off* (tested and failed) from one
    that is merely *untested* (skipped or absent), which the YAML alone cannot convey.
    """
    lines = ["Why each value (provenance):", "", "| Setting | Value | Basis | Detail |", "|---|---|---|---|"]
    for decision in decisions:
        value_text, label, detail = describe_decision(decision)
        lines.append(f"| {decision.setting} | {value_text} | {label} | {detail} |")
    lines.append("")
    return lines


def render_markdown(report: CapabilityReport) -> str:
    """Render the human-readable report, ending with the remediation queue."""
    lines: list[str] = ["# Worker Benchmark Report", ""]

    if report.machine.gpu_name:
        lines.append(f"- GPU: {report.machine.gpu_name} ({report.machine.total_vram_mb} MB VRAM)")
    if report.machine.total_ram_bytes:
        lines.append(f"- RAM: {report.machine.total_ram_bytes / 1024**3:.0f} GB")
    lines.append("")

    if report.capabilities.tiers:
        lines.extend(_render_capabilities(report.capabilities))

    lines.append("## Probes")
    lines.append("")
    lines.append("| Probe | Verdict | it/s p50 | GPU busy % | VRAM HW (MB) | Notes |")
    lines.append("|---|---|---|---|---|---|")
    for result in report.probes:
        stats = result.stats
        its = f"{stats.its_p50:.2f}" if stats and stats.its_p50 is not None else "-"
        gpu = (
            f"{stats.gpu_utilization_mean_percent:.0f}"
            if stats and stats.gpu_utilization_mean_percent is not None
            else "-"
        )
        vram = str(stats.vram_used_high_water_mb) if stats and stats.vram_used_high_water_mb is not None else "-"
        notes = "; ".join(result.reasons + result.advisories)
        lines.append(f"| {result.capability.slug} | {result.verdict} | {its} | {gpu} | {vram} | {notes} |")
    lines.append("")

    lines.append("## Suggested bridgeData")
    lines.append("")
    lines.append(
        "A conservative recommendation: only models that fit with VRAM headroom are loaded, the batch "
        "size is the largest that passed cleanly, and concurrent alchemy is enabled only if the soak held "
        "up. See the capability table above for everything the worker *can* do.",
    )
    lines.append("")
    lines.append("```yaml")
    lines.append(report.suggested_bridge_data.as_yaml_block())
    lines.append("```")
    lines.append("")
    if report.suggested_bridge_data.decisions:
        lines.extend(_render_provenance(report.suggested_bridge_data.decisions))
    if report.suggested_bridge_data.notes:
        lines.append("Conservative choices:")
        lines.append("")
        for note in report.suggested_bridge_data.notes:
            lines.append(f"- {note}")
        lines.append("")
    inconsistencies = verify_suggestion_consistency(report)
    if inconsistencies:
        lines.append("> **Consistency check flagged the recommendation:**")
        for problem in inconsistencies:
            lines.append(f"> - {problem}")
        lines.append("")

    soak_results = [result for result in report.probes if result.capability.kind is CapabilityKind.SUSTAINED]
    if soak_results:
        lines.append("## Validation (sustained load)")
        lines.append("")
        lines.append(
            "Soaks the recommended config above under continuous, mostly-max-config mixed traffic, "
            "confirming throughput holds and nothing degrades over time.",
        )
        lines.append("")
        lines.append("| Tier | Verdict | Duration (s) | Jobs done | GPU duty | it/s retained | Faults | Recoveries |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for result in soak_results:
            stats = result.stats
            duration = result.harness.elapsed_seconds if result.harness is not None else None
            duration_str = f"{duration:.0f}" if duration else "-"
            if stats is not None:
                jobs = str(stats.num_jobs_completed)
                gpu_duty = (
                    f"{stats.gpu_utilization_mean_percent:.0f}%"
                    if stats.gpu_utilization_mean_percent is not None
                    else "-"
                )
                retention = f"{stats.its_retention_fraction:.0%}" if stats.its_retention_fraction is not None else "-"
                faults = str(stats.num_jobs_faulted + stats.num_alchemy_forms_faulted)
                recoveries = str(stats.num_process_recoveries)
            else:
                jobs = gpu_duty = retention = faults = recoveries = "-"
            lines.append(
                f"| {result.capability.tier} | {result.verdict} | {duration_str} | {jobs} | "
                f"{gpu_duty} | {retention} | {faults} | {recoveries} |",
            )
        lines.append("")

    timed = [
        result for result in (soak_results or report.probes) if result.stats and result.stats.phase_breakdown_seconds
    ]
    if timed:
        lines.append("## Where the time goes (per job, median seconds)")
        lines.append("")
        lines.append(
            "Median time a typical image job spends in each phase. `other_inference` is the inference time "
            "outside model load, sampling and VAE (graph build, prompt encode, image encode, IPC); the prime "
            "target for raising GPU duty.",
        )
        lines.append("")
        for result in timed:
            assert result.stats is not None
            breakdown = result.stats.phase_breakdown_seconds
            total = sum(breakdown.values())
            busy_ratio = result.stats.span_derived_busy_ratio
            busy_note = f", GPU-busy phases ≈ {busy_ratio:.0%} of it" if busy_ratio is not None else ""
            lines.append(f"**{result.capability.slug}** (total ≈ {total:.2f}s/job{busy_note})")
            lines.append("")
            lines.append("| Phase | Seconds | % of total |")
            lines.append("|---|---|---|")
            for phase in PHASE_ORDER:
                if phase not in breakdown:
                    continue
                seconds = breakdown[phase]
                pct = (seconds / total * 100) if total > 0 else 0.0
                lines.append(f"| {phase} | {seconds:.3f} | {pct:.0f}% |")
            no_jobs = result.stats.time_spent_no_jobs_available
            if no_jobs:
                lines.append("")
                lines.append(
                    f"Plus ~{no_jobs:.0f}s of the run with no jobs available (horde demand, not the worker): "
                    "demand-limited idle, separate from the per-job gaps above.",
                )
            lines.append("")

    lines.append("## Remediation queue")
    lines.append("")
    if not report.findings:
        lines.append("No robustness findings; no crashes, hangs, lost jobs, or stalls observed.")
    else:
        for finding in report.findings:
            lines.append(f"- **{finding.kind}** ({finding.level_id}): {finding.evidence}")
            lines.append(f"  - reproduce: `horde-benchmark run --only {finding.level_id}`")
    lines.append("")

    return "\n".join(lines)


__all__ = ["describe_decision", "render_markdown"]
