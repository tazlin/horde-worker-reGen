"""Render a :class:`CapabilityReport` to the human-readable markdown an operator reads after a run.

The capability-engine port of ``report.render_markdown``: the same report shape (capability surface,
suggested bridgeData with per-setting provenance, sustained-load soak, per-job time breakdown, and the
findings), read off ``report.probes`` (capability + verdict) instead of the old ``levels``.
Kept separate from :mod:`recommendation` so the pure synthesis has no rendering concern, and torch-free
so any surface can render a stored report without booting anything.
"""

from __future__ import annotations

from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind
from horde_worker_regen.benchmark.capabilities.recommendation import verify_suggestion_consistency
from horde_worker_regen.benchmark.capabilities.result import (
    CapabilityReport,
    EvidenceStatus,
    OperatingProfile,
    ReferenceRateBasis,
    ReferenceSamplingRate,
    SuggestionBasis,
    SuggestionDecision,
    WorkerCapabilities,
    WorkloadEvidence,
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
        ("alchemy caption, interrogation and NSFW forms", capabilities.supports_alchemy_clip),
        ("alchemy upscale, face-fix and background-removal forms", capabilities.supports_alchemy_graph),
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
    lines = ["Why each value:", "", "| Setting | Value | Basis | Detail |", "|---|---|---|---|"]
    for decision in decisions:
        value_text, label, detail = describe_decision(decision)
        lines.append(f"| {decision.setting} | {value_text} | {label} | {detail} |")
    lines.append("")
    return lines


def render_markdown(report: CapabilityReport) -> str:
    """Render the human-readable report, ending with the findings section."""
    if report.operator_assessment:
        return _render_assessment(report)
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
    lines.append("| Probe | Verdict | it/s p50 | GPU busy % | Peak VRAM (MB) | Notes |")
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
        "These settings are conservative. Only models that fit with VRAM headroom are loaded. The batch "
        "size is the largest that passed with no findings. Concurrent alchemy is on only if the soak was "
        "stable. The capability table above lists everything the worker can do.",
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
        lines.append("> **Some suggested settings are not backed by a passed probe:**")
        for problem in inconsistencies:
            lines.append(f"> - {problem}")
        lines.append("")

    soak_results = [result for result in report.probes if result.capability.kind is CapabilityKind.SUSTAINED]
    if soak_results:
        lines.append("## Validation (sustained load)")
        lines.append("")
        lines.append(
            "The soak runs the suggested settings above under a continuous mix of jobs, most of them at the "
            "limits those settings allow. It checks that throughput holds and nothing gets worse over time.",
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
        lines.append("## Time per job by phase (median seconds)")
        lines.append("")
        lines.append(
            "Median time an image job spends in each phase. `other_inference` is the inference time "
            "outside model load, sampling and VAE (graph build, prompt encode, image encode, IPC). "
            "It is the first place to look when raising GPU duty.",
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
                    f"The run also spent ~{no_jobs:.0f}s with no jobs available. That idle time is from a "
                    "lack of jobs, not from the worker, and is not part of the per-job times above.",
                )
            lines.append("")

    lines.append("## Findings")
    lines.append("")
    if not report.findings:
        lines.append("No findings. No crashes, hangs, lost jobs or stalls were seen.")
    else:
        for finding in report.findings:
            lines.append(f"- **{finding.kind}** ({finding.level_id}): {finding.evidence}")
            lines.append(f"  - reproduce: `horde-benchmark run --only {finding.level_id}`")
    lines.append("")

    return "\n".join(lines)


def describe_sampling_rates(rates: dict[str, float], references: dict[str, ReferenceSamplingRate]) -> str:
    """Format each model's measured rate alongside the single-lane reference rate."""
    parts: list[str] = []
    for model_name, rate in rates.items():
        reference = references.get(model_name)
        if reference is None:
            parts.append(f"{model_name} {rate:.1f} it/s with no single-lane rate")
        elif reference.basis is ReferenceRateBasis.SINGLE_LANE:
            parts.append(f"{model_name} {rate:.1f} it/s against {reference.iterations_per_second:.1f} alone")
        else:
            parts.append(f"{model_name} {rate:.1f} it/s against a run best of {reference.iterations_per_second:.1f}")
    return ", ".join(parts)


def describe_lane_throughput(item: WorkloadEvidence) -> str | None:
    """Return a confirmed concurrency check's lane-equivalent throughput description, or None if unavailable.

    A check that fell short of one lane already states the figure in its reason.
    """
    if item.lane_equivalent_throughput is None or item.status is not EvidenceStatus.MEASURED_SUCCESS:
        return None
    rates = describe_sampling_rates(item.model_sampling_rates, item.model_reference_rates)
    return f"The lanes together sampled {item.lane_equivalent_throughput:.2f} times one lane's rate: {rates}"


def describe_jobs_per_minute(
    profile: OperatingProfile, *, jobs_per_minute: float | None, single_lane_jobs_per_minute: float | None
) -> str:
    """State the jobs per minute a concurrency check finished at its thread count and at one lane."""
    lanes = f"{profile.threads} lanes"
    if jobs_per_minute is None or single_lane_jobs_per_minute is None:
        return f"Jobs per minute were not measured for both {lanes} and one lane"
    return (
        f"{lanes} finished {jobs_per_minute:.1f} jobs per minute against {single_lane_jobs_per_minute:.1f} on one "
        f"lane at {profile.width}x{profile.height}, batch {profile.batch}, {profile.steps} steps"
    )


def describe_wall_throughput(item: WorkloadEvidence) -> str | None:
    """Return a confirmed concurrency check's throughput gain description, or None if unavailable.

    An inconclusive check already includes the jobs per minute in the reason.
    """
    if item.wall_gain is None or item.status is not EvidenceStatus.MEASURED_SUCCESS:
        return None
    jobs = describe_jobs_per_minute(
        item.profile,
        jobs_per_minute=item.jobs_per_minute,
        single_lane_jobs_per_minute=item.single_lane_jobs_per_minute,
    )
    required = f" (minimum {item.wall_gain_required:.2f})" if item.wall_gain_required is not None else ""
    return f"{jobs}. Throughput ratio: {item.wall_gain:.2f}{required}"


def _render_assessment(report: CapabilityReport) -> str:
    lines = ["# Worker capability assessment", ""]
    if report.budget is not None:
        budget = report.budget
        lines.extend(
            [
                f"Elapsed: {budget.elapsed_seconds:.1f}s. Offered work: {budget.jobs_offered}/{budget.max_jobs}.",
                f"Stop reason: {budget.stop_reason}.",
            ]
        )
        configured = [item.configure_seconds for item in report.evidence if item.configure_seconds is not None]
        if configured:
            elapsed = sum(item.elapsed_seconds for item in report.evidence)
            sampling = sum(item.sampling_seconds or 0.0 for item in report.evidence)
            lines.append(
                f"Checks took {elapsed:.1f}s in total, of which {sum(configured):.1f}s went to applying settings "
                f"between checks and {sampling:.1f}s to sampling."
            )
        lines.append("")
    lines.extend(
        [
            "Measured results apply to the named card, checkpoint and workload. Predictions are estimates. "
            "Other setting combinations and long-term reliability were not tested.",
            "",
            "| Scope / image card | Model | Check | Evidence | Workload | "
            "Sampling overlap | Device peak MB | Detail |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in report.evidence:
        profile = item.profile
        workload = f"{profile.width}x{profile.height}, batch {profile.batch}, threads {profile.threads}"
        features = [*profile.post_processing, *profile.alchemy_forms]
        if profile.control_type:
            features.append(profile.control_type)
        if profile.workflow:
            features.append(profile.workflow)
        if profile.hires_fix:
            features.append("hires-fix")
        if features:
            workload += "; " + ", ".join(features)
        peak = str(item.device_peak_mb) if item.device_peak_mb is not None else "unknown"
        check = "Combined settings" if item.combined_validation else item.capability.label
        status = {
            "measured_success": "Passed",
            "measured_failure": "Failed",
            "predicted": "Estimated",
            "unknown": "Inconclusive" if item.attempted or item.job_ids else "Not tested",
        }[item.status.value]
        if item.reason_code == "unavailable":
            status = "Unavailable"
        if item.reason_code == "not_applicable":
            status = "Not applicable"
        detail = item.reason.replace("|", "/").replace("\n", " ")
        if item.notes:
            detail = " ".join(item.notes) + " " + detail
        scope = (
            f"shared worker (image card {profile.device_index})"
            if profile.alchemy_forms
            else str(profile.device_index)
        )
        if item.service_placements:
            placements = ", ".join(
                f"{placement.form or placement.job_id}: reported lane card {placement.reported_device_index}"
                if placement.reported_device_index is not None
                else f"{placement.form or placement.job_id}: lane unreported"
                for placement in item.service_placements
            )
            detail += f"; form placement: {placements}; execution device unreported"
        for description in (describe_wall_throughput(item), describe_lane_throughput(item)):
            if description is not None:
                detail += f"; {description.replace('|', '/')}"
        lines.append(
            f"| {scope} | {profile.model_name} | {check} | "
            f"{status} | {workload} | "
            f"{item.observed_sampling_concurrency}/{item.expected_sampling_concurrency or profile.threads} | "
            f"{peak} | {detail} |"
        )
    lines.extend(
        [
            "",
            "## Model loading",
            "",
            "Observed loading includes deserialization and may use the OS cache. "
            "Physical disk speed was not measured.",
            "",
            "| Model | First observed load | Repeat loads | Samples |",
            "|---|---|---|---|",
        ]
    )
    for loading in report.model_loading:
        repeats = (
            f"{min(loading.repeat_seconds):.2f} to {max(loading.repeat_seconds):.2f}s"
            if loading.repeat_seconds
            else "Not reported"
        )
        lines.append(
            f"| {loading.model_name} | {loading.first_seconds:.2f}s | {repeats} | {len(loading.load_seconds)} |"
        )
    lines.extend(["", *report.storage_notes, ""])
    if report.model_guidance:
        lines.extend(["", "## Model guidance", "", *[f"- {line}" for line in report.model_guidance], ""])
    if report.suggested_bridge_data.safe_to_apply:
        lines.extend(
            ["", "## Suggested bridgeData", "", "```yaml", report.suggested_bridge_data.as_yaml_block(), "```", ""]
        )
    else:
        lines.extend(
            ["", "Combined settings were not confirmed. Review the results before changing the configuration.", ""]
        )
    lines.extend(_render_provenance(report.suggested_bridge_data.decisions))
    lines.extend(f"- {note}" for note in report.suggested_bridge_data.notes)
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "describe_decision",
    "describe_jobs_per_minute",
    "describe_lane_throughput",
    "describe_sampling_rates",
    "describe_wall_throughput",
    "render_markdown",
]
