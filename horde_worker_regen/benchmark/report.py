"""Per-level and per-run benchmark reports, plus the suggested-bridgeData synthesis."""

from __future__ import annotations

import statistics
from typing import Literal

from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.criteria import LevelCriteria, LevelStats, LevelVerdict
from horde_worker_regen.benchmark.ladder import BENCH_TIER_MODELS, RampLevel
from horde_worker_regen.process_management.run_metrics import RunMetricsSnapshot

FindingKind = Literal[
    "oom",
    "hang",
    "crash",
    "lost_job",
    "double_submit",
    "process_recovery",
    "download_stall",
    "swallowed_error",
]


class Finding(BaseModel):
    """One robustness problem observed during a level, for the remediation queue."""

    kind: FindingKind
    level_id: str
    evidence: str


class HarnessSummary(BaseModel):
    """The JSON-friendly subset of a HarnessResult."""

    num_jobs_expected: int = 0
    num_jobs_completed: int = 0
    num_jobs_faulted: int = 0
    num_alchemy_forms_expected: int = 0
    num_alchemy_forms_completed: int = 0
    num_alchemy_forms_faulted: int = 0
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    audit_failures: list[str] = Field(default_factory=list)
    exit_reason: str = ""
    diagnostics: list[str] = Field(default_factory=list)


class LevelRunResult(BaseModel):
    """What the level runner writes to disk: raw outcomes, no policy applied."""

    level_id: str
    harness: HarnessSummary
    metrics: RunMetricsSnapshot | None = None
    runner_error: str | None = None
    """Set when the runner itself failed before/while producing results."""


class LevelReport(BaseModel):
    """One level's complete record: definition, outcome, stats, verdict, findings."""

    level: RampLevel
    outcome: Literal["passed", "failed", "skipped", "crashed", "crashed_hang"]
    reasons: list[str] = Field(default_factory=list)
    advisories: list[str] = Field(default_factory=list)
    stats: LevelStats | None = None
    harness: HarnessSummary | None = None
    findings: list[Finding] = Field(default_factory=list)
    log_tail: list[str] = Field(default_factory=list)


class MachineInfo(BaseModel):
    """The hardware the benchmark ran on."""

    gpu_name: str | None = None
    total_vram_mb: int | None = None
    total_ram_bytes: int | None = None


class SuggestedBridgeData(BaseModel):
    """Concrete bridgeData values derived from the highest stable levels."""

    max_threads: int = 1
    queue_size: int = 1
    max_batch: int = 1
    allow_lora: bool = False
    allow_controlnet: bool = False
    allow_post_processing: bool = False
    models_to_load: list[str] = Field(default_factory=list)
    alchemist: bool = False
    alchemy_allow_concurrent: bool = False

    def as_yaml_block(self) -> str:
        """Render as a bridgeData.yaml-compatible snippet."""
        lines = [
            f"max_threads: {self.max_threads}",
            f"queue_size: {self.queue_size}",
            f"max_batch: {self.max_batch}",
            f"allow_lora: {str(self.allow_lora).lower()}",
            f"allow_controlnet: {str(self.allow_controlnet).lower()}",
            f"allow_post_processing: {str(self.allow_post_processing).lower()}",
            "models_to_load:",
            *[f'  - "{model}"' for model in self.models_to_load],
            f"alchemist: {str(self.alchemist).lower()}",
            f"alchemy_allow_concurrent: {str(self.alchemy_allow_concurrent).lower()}",
        ]
        return "\n".join(lines)


class BenchmarkReport(BaseModel):
    """The full ramp outcome: per-level reports plus the synthesized recommendation."""

    machine: MachineInfo = Field(default_factory=MachineInfo)
    levels: list[LevelReport] = Field(default_factory=list)
    suggested_bridge_data: SuggestedBridgeData = Field(default_factory=SuggestedBridgeData)
    tier_baselines_its: dict[str, float] = Field(default_factory=dict)

    @property
    def findings(self) -> list[Finding]:
        """All findings across levels, in ladder order."""
        return [finding for level in self.levels for finding in level.findings]


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def compute_level_stats(result: LevelRunResult, *, total_vram_mb: int | None = None) -> LevelStats:
    """Distill a raw level run into the criteria-relevant statistics."""
    harness = result.harness
    metrics = result.metrics

    its_values: list[float] = []
    disk_loads: list[float] = []
    vram_loads: list[float] = []
    queue_waits: list[float] = []
    e2es: list[float] = []
    download_rates: list[float] = []
    vram_high_water: int | None = None

    if metrics is not None:
        for job in metrics.jobs:
            if job.phase_metrics is not None:
                if job.phase_metrics.sampling is not None and job.phase_metrics.sampling.iterations_per_second > 0:
                    its_values.append(job.phase_metrics.sampling.iterations_per_second)
                for load in job.phase_metrics.model_loads:
                    if load.phase == "disk_to_ram":
                        disk_loads.append(load.duration_seconds)
                    else:
                        vram_loads.append(load.duration_seconds)
            if job.queue_wait_seconds is not None:
                queue_waits.append(job.queue_wait_seconds)
            if job.e2e_seconds is not None:
                e2es.append(job.e2e_seconds)
        for download in metrics.downloads:
            if download.success and download.megabytes_per_second > 0:
                download_rates.append(download.megabytes_per_second)
        if metrics.vram_used_high_water_mb_per_process:
            vram_high_water = max(metrics.vram_used_high_water_mb_per_process.values())

    return LevelStats(
        num_jobs_expected=harness.num_jobs_expected,
        num_jobs_completed=harness.num_jobs_completed,
        num_jobs_faulted=harness.num_jobs_faulted,
        num_alchemy_forms_faulted=harness.num_alchemy_forms_faulted,
        num_audit_failures=len(harness.audit_failures),
        num_process_recoveries=metrics.num_process_recoveries if metrics is not None else 0,
        timed_out=harness.timed_out,
        its_p50=statistics.median(its_values) if its_values else None,
        its_min=min(its_values) if its_values else None,
        vram_used_high_water_mb=vram_high_water,
        total_vram_mb=total_vram_mb,
        disk_min_free_bytes=(min(metrics.disk_min_free_bytes.values()) if metrics and metrics.disk_min_free_bytes else None),
        download_mbps_min=min(download_rates) if download_rates else None,
        model_load_disk_seconds_median=statistics.median(disk_loads) if disk_loads else None,
        model_load_vram_seconds_median=statistics.median(vram_loads) if vram_loads else None,
        queue_wait_seconds_p95=_percentile(queue_waits, 0.95),
        e2e_seconds_p95=_percentile(e2es, 0.95),
    )


def synthesize_bridge_data(levels: list[LevelReport]) -> SuggestedBridgeData:
    """Derive bridgeData values from the highest-passing levels."""
    suggestion = SuggestedBridgeData()
    passing_tiers: set[str] = set()

    for report in levels:
        if report.outcome != "passed":
            continue
        level = report.level

        if level.stage == "A":
            passing_tiers.add(level.tier)
        elif level.axis == "queue_size":
            suggestion.queue_size = max(
                suggestion.queue_size,
                int(level.bridge_data_overrides.get("queue_size", 1)),
            )
        elif level.axis == "threads":
            suggestion.max_threads = max(
                suggestion.max_threads,
                int(level.bridge_data_overrides.get("max_threads", 1)),
            )
        elif level.axis == "batch":
            batch_sizes = [job.n_iter for job in level.scenario.image_jobs]
            if batch_sizes:
                suggestion.max_batch = max(suggestion.max_batch, max(batch_sizes))
        elif level.axis == "controlnet":
            suggestion.allow_controlnet = True
        elif level.axis == "post_processing":
            suggestion.allow_post_processing = True
        elif level.axis == "downloads":
            suggestion.allow_lora = True
        elif level.axis == "alchemy":
            suggestion.alchemist = True
            if level.rung >= 2:
                suggestion.alchemy_allow_concurrent = True

    suggestion.models_to_load = [BENCH_TIER_MODELS[tier] for tier in BENCH_TIER_MODELS if tier in passing_tiers]
    return suggestion


def render_markdown(report: BenchmarkReport) -> str:
    """Render the human-readable report, ending with the remediation queue."""
    lines: list[str] = ["# Worker Benchmark Report", ""]

    if report.machine.gpu_name:
        lines.append(f"- GPU: {report.machine.gpu_name} ({report.machine.total_vram_mb} MB VRAM)")
    if report.machine.total_ram_bytes:
        lines.append(f"- RAM: {report.machine.total_ram_bytes / 1024**3:.0f} GB")
    lines.append("")

    lines.append("## Levels")
    lines.append("")
    lines.append("| Level | Outcome | it/s p50 | VRAM HW (MB) | Notes |")
    lines.append("|---|---|---|---|---|")
    for level_report in report.levels:
        stats = level_report.stats
        its = f"{stats.its_p50:.2f}" if stats and stats.its_p50 is not None else "-"
        vram = str(stats.vram_used_high_water_mb) if stats and stats.vram_used_high_water_mb is not None else "-"
        notes = "; ".join(level_report.reasons + level_report.advisories)
        lines.append(f"| {level_report.level.id} | {level_report.outcome} | {its} | {vram} | {notes} |")
    lines.append("")

    lines.append("## Suggested bridgeData")
    lines.append("")
    lines.append("```yaml")
    lines.append(report.suggested_bridge_data.as_yaml_block())
    lines.append("```")
    lines.append("")

    lines.append("## Remediation queue")
    lines.append("")
    if not report.findings:
        lines.append("No robustness findings — no crashes, hangs, lost jobs, or stalls observed.")
    else:
        for finding in report.findings:
            lines.append(f"- **{finding.kind}** ({finding.level_id}): {finding.evidence}")
            lines.append(f"  - reproduce: `horde-benchmark ramp --only-level {finding.level_id}`")
    lines.append("")

    return "\n".join(lines)
