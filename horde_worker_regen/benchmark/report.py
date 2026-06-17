"""Per-level and per-run benchmark reports, plus the recommendation synthesis.

The report carries two distinct things derived from the ramp:

- a :class:`WorkerCapabilities` summary of *everything the worker proved it can do* (every tier and
  feature that passed in isolation), and
- a :class:`SuggestedBridgeData` *conservative recommendation* of what to actually turn on by
  default, which keeps VRAM/disk headroom, prefers stability over peak throughput, and is downgraded
  when the post-ramp soak shows the combined config does not hold up under sustained load.

Separating the two means an operator sees the full capability surface without being pushed into an
aggressive default that only just fit or only just held together.
"""

from __future__ import annotations

import statistics
import time

from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.criteria import LevelStats
from horde_worker_regen.benchmark.enums import BenchAxis, BenchStage, BenchTier, FindingKind, LevelOutcome
from horde_worker_regen.benchmark.ladder import BENCH_TIER_MODELS, RampLevel
from horde_worker_regen.process_management.run_metrics import JobMetricsRecord, RunMetricsSnapshot

BENCHMARK_REPORT_SCHEMA_VERSION = 2
"""Bumped when the report schema changes incompatibly; stamped into every report for later reads.

v2 adds :class:`WorkerCapabilities` and the conservative-recommendation rationale alongside the
existing ``suggested_bridge_data``.
"""

_MIN_VRAM_HEADROOM_MB = 1500
"""A tier's model is only recommended for ``models_to_load`` if its peak VRAM leaves this much free."""


def _current_worker_version() -> str:
    """Return the running worker version (local import keeps this module import-light)."""
    from horde_worker_regen import __version__

    return __version__


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
    outcome: LevelOutcome
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
    """A conservative bridgeData recommendation derived from the stable levels.

    Unlike a raw "highest passing rung" readout, this keeps VRAM headroom (only models that fit with
    headroom are loaded), prefers a batch size that passed without robustness findings, and has
    concurrent alchemy disabled if the sustained-load soak did not hold up. ``notes`` records why each
    conservative choice was made, for the report.
    """

    max_threads: int = 1
    queue_size: int = 1
    max_batch: int = 1
    allow_lora: bool = False
    allow_controlnet: bool = False
    allow_sdxl_controlnet: bool = False
    allow_post_processing: bool = False
    models_to_load: list[str] = Field(default_factory=list)
    alchemist: bool = False
    alchemy_allow_concurrent: bool = False
    alchemy_max_concurrency: int = 1
    notes: list[str] = Field(default_factory=list)
    """Human-readable rationale for the conservative choices (not part of the bridgeData itself)."""

    def as_yaml_block(self) -> str:
        """Render as a bridgeData.yaml-compatible snippet."""
        lines = [
            f"max_threads: {self.max_threads}",
            f"queue_size: {self.queue_size}",
            f"max_batch: {self.max_batch}",
            f"allow_lora: {str(self.allow_lora).lower()}",
            f"allow_controlnet: {str(self.allow_controlnet).lower()}",
            f"allow_sdxl_controlnet: {str(self.allow_sdxl_controlnet).lower()}",
            f"allow_post_processing: {str(self.allow_post_processing).lower()}",
            "models_to_load:",
            *[f'  - "{model}"' for model in self.models_to_load],
            f"alchemist: {str(self.alchemist).lower()}",
            f"alchemy_allow_concurrent: {str(self.alchemy_allow_concurrent).lower()}",
            f"alchemy_max_concurrency: {self.alchemy_max_concurrency}",
        ]
        return "\n".join(lines)

    def to_bridge_overrides(self) -> dict[str, object]:
        """The worker bridge-data fields this recommendation sets (for the validation run).

        ``max_batch`` is intentionally excluded: batch size is a per-job payload value, not a
        worker config field, so the soak applies it through its job templates instead.
        """
        return {
            "max_threads": self.max_threads,
            "queue_size": self.queue_size,
            "allow_lora": self.allow_lora,
            "allow_controlnet": self.allow_controlnet,
            "allow_sdxl_controlnet": self.allow_sdxl_controlnet,
            "allow_post_processing": self.allow_post_processing,
            "models_to_load": list(self.models_to_load),
            "alchemist": self.alchemist,
            "alchemy_allow_concurrent": self.alchemy_allow_concurrent,
            "alchemy_max_concurrency": self.alchemy_max_concurrency,
        }


class TierCapability(BaseModel):
    """What one model tier proved during the ramp."""

    tier: BenchTier
    model_name: str
    baseline_passed: bool
    observed_its_p50: float | None = None
    max_stable_batch: int = 1
    """The largest batch rung that passed with no robustness findings."""
    peak_vram_mb: int | None = None
    fits_with_headroom: bool = False
    """Whether the tier's baseline peak VRAM left the headroom reserve free (drives models_to_load)."""


class WorkerCapabilities(BaseModel):
    """Everything the worker proved it can do, independent of the conservative recommendation."""

    tiers: list[TierCapability] = Field(default_factory=list)
    supports_hires_fix: bool = False
    supports_post_processing: bool = False
    supports_controlnet: bool = False
    """Classic SD1.5 preprocessor controlnet (canny/depth/openpose)."""
    supports_qr_code: bool = False
    """The QR-code controlnet workflow (the SDXL controlnet capability)."""
    supports_alchemy_clip: bool = False
    supports_alchemy_graph: bool = False
    supports_alchemy_concurrent: bool = False
    supports_lora: bool = False


class BenchmarkReport(BaseModel):
    """The full ramp outcome: per-level reports, the capability surface, and the recommendation.

    Self-describing: ``worker_version``, ``created_at`` and ``run_id`` stamp every report so a later
    run (or a version bump) can tell whether the stored recommendation still applies. The defaults
    make reports written before these fields were added still parse.
    """

    report_schema_version: int = BENCHMARK_REPORT_SCHEMA_VERSION
    worker_version: str = Field(default_factory=_current_worker_version)
    created_at: float = Field(default_factory=time.time)
    run_id: str = ""

    machine: MachineInfo = Field(default_factory=MachineInfo)
    levels: list[LevelReport] = Field(default_factory=list)
    capabilities: WorkerCapabilities = Field(default_factory=WorkerCapabilities)
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


def _its_retention(jobs: list[JobMetricsRecord]) -> float | None:
    """Median sampling rate of the second half of completed image jobs ÷ that of the first half.

    Ordered by completion time, this catches throughput decay over a sustained run (thermal
    throttling, a VRAM/RAM leak, queue backpressure). Returns None without enough samples.
    """
    timed: list[tuple[float, float]] = []
    for job in jobs:
        if job.is_alchemy or job.phase_metrics is None or job.phase_metrics.sampling is None:
            continue
        its = job.phase_metrics.sampling.iterations_per_second
        finalized = job.stage_timestamps.get("FINALIZED")
        if its > 0 and finalized is not None:
            timed.append((finalized, its))

    if len(timed) < 4:
        return None

    timed.sort(key=lambda entry: entry[0])
    midpoint = len(timed) // 2
    first_half = statistics.median([its for _, its in timed[:midpoint]])
    second_half = statistics.median([its for _, its in timed[midpoint:]])
    if first_half <= 0:
        return None
    return second_half / first_half


_PHASE_ORDER = [
    "queue_wait",
    "disk_load",
    "vram_load",
    "sampling",
    "vae",
    "other_inference",
    "safety",
    "submit",
]


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _phase_breakdown(jobs: list[JobMetricsRecord]) -> dict[str, float]:
    """Median per-job seconds in each pipeline phase, to show where a typical job's time goes.

    Built from worker stage timestamps (queue_wait/inference/safety/submit) and the child's
    phase metrics (disk->RAM, RAM->VRAM, sampling, VAE); ``other_inference`` is the inference
    time not otherwise accounted for (graph build, prompt encode, image encode, IPC).
    """
    samples: dict[str, list[float]] = {phase: [] for phase in _PHASE_ORDER}

    for job in jobs:
        if job.is_alchemy:
            continue
        stamps = job.stage_timestamps or {}
        if job.queue_wait_seconds is not None:
            samples["queue_wait"].append(job.queue_wait_seconds)

        inference_start = stamps.get("INFERENCE_IN_PROGRESS")
        inference_end = stamps.get("PENDING_SAFETY_CHECK")
        submit_ready = stamps.get("PENDING_SUBMIT")
        finalized = stamps.get("FINALIZED")

        if submit_ready is not None and finalized is not None:
            samples["submit"].append(max(0.0, finalized - submit_ready))
        if inference_end is not None and submit_ready is not None:
            samples["safety"].append(max(0.0, submit_ready - inference_end))

        pm = job.phase_metrics
        if pm is None or inference_start is None or inference_end is None:
            continue
        inference_total = max(0.0, inference_end - inference_start)

        disk_load = sum(m.duration_seconds for m in pm.model_loads if m.phase == "disk_to_ram")
        vram_load = sum(m.duration_seconds for m in pm.model_loads if m.phase == "ram_to_vram")
        sampling = pm.sampling.duration_seconds if pm.sampling is not None else 0.0
        vae = sum((pm.phase_seconds or {}).values())
        other = max(0.0, inference_total - disk_load - vram_load - sampling - vae)

        samples["disk_load"].append(disk_load)
        samples["vram_load"].append(vram_load)
        samples["sampling"].append(sampling)
        samples["vae"].append(vae)
        samples["other_inference"].append(other)

    return {phase: median for phase in _PHASE_ORDER if (median := _median(samples[phase])) is not None}


_GPU_BUSY_PHASES = ("vram_load", "sampling", "vae")


def _span_derived_busy_ratio(breakdown: dict[str, float]) -> float | None:
    """GPU-touching phases (vram_load + sampling + vae) ÷ the whole per-job wall, or None.

    A phase-attributed duty-cycle proxy that needs no tracing backend: it says what share of a
    typical job's lifecycle is actual GPU work versus idle hand-off (queue_wait, disk_load,
    other_inference, safety, submit). The NVML mean is still the headline; this explains it.
    """
    total = sum(breakdown.values())
    if total <= 0:
        return None
    busy = sum(breakdown.get(phase, 0.0) for phase in _GPU_BUSY_PHASES)
    return busy / total


def _post_warmup_vram_reloads(jobs: list[JobMetricsRecord]) -> int | None:
    """RAM->VRAM reloads across image jobs after the first completed one (warm-up excluded).

    Orders jobs by completion and drops the first (the unavoidable cold load), then counts every
    subsequent ``ram_to_vram`` model load. Under expected residency this is 0; a positive count
    means a resident model was evicted and reloaded (residency defeated / stickiness thrash).
    Returns None when no job carried phase metrics.
    """
    timed: list[tuple[float, int]] = []
    for job in jobs:
        if job.is_alchemy or job.phase_metrics is None:
            continue
        finalized = job.stage_timestamps.get("FINALIZED")
        if finalized is None:
            continue
        reloads = sum(1 for load in job.phase_metrics.model_loads if load.phase == "ram_to_vram")
        timed.append((finalized, reloads))

    if not timed:
        return None
    timed.sort(key=lambda entry: entry[0])
    return sum(reloads for _, reloads in timed[1:])


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

    disk_min_free = (
        min(metrics.disk_min_free_bytes.values()) if metrics is not None and metrics.disk_min_free_bytes else None
    )

    phase_breakdown = _phase_breakdown(metrics.jobs) if metrics is not None else {}

    return LevelStats(
        num_jobs_expected=harness.num_jobs_expected,
        num_jobs_completed=harness.num_jobs_completed,
        num_jobs_faulted=harness.num_jobs_faulted,
        num_alchemy_forms_expected=harness.num_alchemy_forms_expected,
        num_alchemy_forms_completed=harness.num_alchemy_forms_completed,
        num_alchemy_forms_faulted=harness.num_alchemy_forms_faulted,
        num_audit_failures=len(harness.audit_failures),
        num_process_recoveries=metrics.num_process_recoveries if metrics is not None else 0,
        timed_out=harness.timed_out,
        its_p50=statistics.median(its_values) if its_values else None,
        its_min=min(its_values) if its_values else None,
        its_retention_fraction=_its_retention(metrics.jobs) if metrics is not None else None,
        gpu_utilization_mean_percent=metrics.gpu_utilization_mean_percent if metrics is not None else None,
        gpu_utilization_busy_fraction=metrics.gpu_utilization_busy_fraction if metrics is not None else None,
        span_derived_busy_ratio=_span_derived_busy_ratio(phase_breakdown),
        post_warmup_vram_reloads=(_post_warmup_vram_reloads(metrics.jobs) if metrics is not None else None),
        vram_used_high_water_mb=vram_high_water,
        total_vram_mb=total_vram_mb,
        disk_min_free_bytes=disk_min_free,
        download_mbps_min=min(download_rates) if download_rates else None,
        model_load_disk_seconds_median=statistics.median(disk_loads) if disk_loads else None,
        model_load_vram_seconds_median=statistics.median(vram_loads) if vram_loads else None,
        queue_wait_seconds_p95=_percentile(queue_waits, 0.95),
        e2e_seconds_p95=_percentile(e2es, 0.95),
        phase_breakdown_seconds=phase_breakdown,
    )


def _passed(report: LevelReport) -> bool:
    """Whether a level passed cleanly (terminal pass with no robustness findings)."""
    return report.outcome == LevelOutcome.PASSED and not report.findings


def _axis_passed(levels: list[LevelReport], axis: BenchAxis) -> bool:
    """Whether any level on *axis* passed."""
    return any(report.outcome == LevelOutcome.PASSED for report in levels if report.level.axis == axis)


def synthesize_capabilities(levels: list[LevelReport], *, total_vram_mb: int | None = None) -> WorkerCapabilities:
    """Summarize everything the worker proved it can do across the ramp.

    This is the unfiltered capability surface: every tier whose baseline passed and every feature axis
    that passed in isolation, independent of whether the conservative recommendation turns it on.
    """
    capabilities = WorkerCapabilities(
        supports_hires_fix=_axis_passed(levels, BenchAxis.HIRES_FIX),
        supports_post_processing=_axis_passed(levels, BenchAxis.POST_PROCESSING),
        supports_controlnet=_axis_passed(levels, BenchAxis.CONTROLNET),
        supports_qr_code=_axis_passed(levels, BenchAxis.QR_CODE),
        supports_alchemy_clip=_axis_passed(levels, BenchAxis.ALCHEMY_CLIP),
        supports_alchemy_graph=_axis_passed(levels, BenchAxis.ALCHEMY_GRAPH),
        supports_alchemy_concurrent=_axis_passed(levels, BenchAxis.ALCHEMY_CONCURRENT),
        supports_lora=_axis_passed(levels, BenchAxis.DOWNLOADS),
    )

    for tier, model_name in BENCH_TIER_MODELS.items():
        baseline_reports = [
            report for report in levels if report.level.tier == tier and report.level.establishes_tier_baseline
        ]
        if not baseline_reports:
            continue
        baseline = baseline_reports[0]
        baseline_passed = baseline.outcome == LevelOutcome.PASSED
        peak_vram = baseline.stats.vram_used_high_water_mb if baseline.stats is not None else None
        capabilities.tiers.append(
            TierCapability(
                tier=tier,
                model_name=model_name,
                baseline_passed=baseline_passed,
                observed_its_p50=baseline.stats.its_p50 if baseline.stats is not None else None,
                max_stable_batch=_max_stable_batch(levels, tier),
                peak_vram_mb=peak_vram,
                fits_with_headroom=_fits_with_headroom(
                    baseline_passed=baseline_passed,
                    peak_vram_mb=peak_vram,
                    total_vram_mb=total_vram_mb,
                ),
            ),
        )

    return capabilities


def _max_stable_batch(levels: list[LevelReport], tier: BenchTier) -> int:
    """Return the largest batch size that passed for *tier* with no robustness findings."""
    stable = 1
    for report in levels:
        if report.level.tier != tier or report.level.axis != BenchAxis.BATCH or not _passed(report):
            continue
        batch_sizes = [job.n_iter for job in report.level.scenario.image_jobs]
        if batch_sizes:
            stable = max(stable, max(batch_sizes))
    return stable


def _fits_with_headroom(*, baseline_passed: bool, peak_vram_mb: int | None, total_vram_mb: int | None) -> bool:
    """Whether a tier's model is safe to keep resident: it passed and leaves the VRAM reserve free.

    When the machine VRAM or the observed peak is unknown, a passing baseline is taken to fit (the
    level ran, so it physically fit); the headroom guard only *excludes* a model when it can prove
    the reserve would be breached.
    """
    if not baseline_passed:
        return False
    if total_vram_mb is None or peak_vram_mb is None:
        return True
    return (total_vram_mb - peak_vram_mb) >= _MIN_VRAM_HEADROOM_MB


def _soak_unstable_tiers(levels: list[LevelReport]) -> tuple[set[BenchTier], bool]:
    """Return the tiers whose sustained-load soak did not pass, and whether any soak was unstable.

    "Unstable" means a soak failed with a hard robustness finding (faults, recoveries, OOM, hang) as
    opposed to merely missing a throughput/duty advisory; that is the signal to drop concurrent
    alchemy from the recommendation.
    """
    failed_tiers: set[BenchTier] = set()
    any_unstable = False
    hard = {FindingKind.OOM, FindingKind.HANG, FindingKind.CRASH, FindingKind.PROCESS_RECOVERY, FindingKind.LOST_JOB}
    for report in levels:
        if report.level.stage != BenchStage.VALIDATION or report.outcome == LevelOutcome.PASSED:
            continue
        failed_tiers.add(report.level.tier)
        if any(finding.kind in hard for finding in report.findings):
            any_unstable = True
    return failed_tiers, any_unstable


def synthesize_bridge_data(levels: list[LevelReport], *, total_vram_mb: int | None = None) -> SuggestedBridgeData:
    """Derive a conservative bridgeData recommendation from the stable levels.

    Conservative means: only models that passed and fit with VRAM headroom are loaded; ``max_batch``
    is the largest batch that passed without a robustness finding (not merely the highest attempted);
    and concurrent alchemy is recommended only if it passed and the sustained-load soak held up. When
    the soak (stage V) is present and a tier's soak failed, that tier's model is dropped; a hard soak
    failure also disables concurrent alchemy. ``notes`` records each downgrade.
    """
    suggestion = SuggestedBridgeData()
    notes: list[str] = []
    passing_tiers: list[BenchTier] = []

    for report in levels:
        level = report.level
        if report.outcome != LevelOutcome.PASSED:
            continue
        if level.establishes_tier_baseline:
            passing_tiers.append(level.tier)
        elif level.axis == BenchAxis.QUEUE_SIZE:
            suggestion.queue_size = max(suggestion.queue_size, _int_override(level, "queue_size", 1))
        elif level.axis == BenchAxis.THREADS:
            suggestion.max_threads = max(suggestion.max_threads, _int_override(level, "max_threads", 1))
        elif level.axis == BenchAxis.CONTROLNET:
            suggestion.allow_controlnet = True
        elif level.axis == BenchAxis.QR_CODE:
            suggestion.allow_controlnet = True
            if level.tier == BenchTier.SDXL:
                suggestion.allow_sdxl_controlnet = True
        elif level.axis == BenchAxis.POST_PROCESSING:
            suggestion.allow_post_processing = True
        elif level.axis == BenchAxis.DOWNLOADS:
            suggestion.allow_lora = True
        elif level.axis in (BenchAxis.ALCHEMY_CLIP, BenchAxis.ALCHEMY_GRAPH):
            suggestion.alchemist = True
        elif level.axis == BenchAxis.ALCHEMY_CONCURRENT:
            suggestion.alchemist = True
            suggestion.alchemy_allow_concurrent = True
            suggestion.alchemy_max_concurrency = 2

    # max_batch: the largest batch that passed cleanly across any tier (a per-job payload cap).
    suggestion.max_batch = max((_max_stable_batch(levels, tier) for tier in passing_tiers), default=1)

    failed_soak_tiers, soak_unstable = _soak_unstable_tiers(levels)

    models_to_load: list[str] = []
    for tier in passing_tiers:
        baseline = _baseline_report(levels, tier)
        peak_vram = baseline.stats.vram_used_high_water_mb if baseline is not None and baseline.stats else None
        if not _fits_with_headroom(baseline_passed=True, peak_vram_mb=peak_vram, total_vram_mb=total_vram_mb):
            notes.append(
                f"{BENCH_TIER_MODELS[tier]} omitted from models_to_load: its peak VRAM leaves under "
                f"{_MIN_VRAM_HEADROOM_MB} MB headroom on this GPU.",
            )
            continue
        if tier in failed_soak_tiers:
            notes.append(
                f"{BENCH_TIER_MODELS[tier]} omitted from models_to_load: its sustained-load soak did not pass.",
            )
            continue
        models_to_load.append(BENCH_TIER_MODELS[tier])
    suggestion.models_to_load = models_to_load

    if suggestion.alchemy_allow_concurrent and soak_unstable:
        suggestion.alchemy_allow_concurrent = False
        suggestion.alchemy_max_concurrency = 1
        notes.append("Concurrent alchemy disabled: the sustained-load soak showed instability under combined load.")

    suggestion.notes = notes
    return suggestion


def _int_override(level: RampLevel, key: str, default: int) -> int:
    """Read an integer bridge-data override off a level, defaulting when absent or non-numeric."""
    value = level.bridge_data_overrides.get(key, default)
    return int(value) if isinstance(value, (int, float)) else default


def _baseline_report(levels: list[LevelReport], tier: BenchTier) -> LevelReport | None:
    """Return the tier's stage-A baseline report, or None when it is absent."""
    for report in levels:
        if report.level.tier == tier and report.level.establishes_tier_baseline:
            return report
    return None


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


def render_markdown(report: BenchmarkReport) -> str:
    """Render the human-readable report, ending with the remediation queue."""
    lines: list[str] = ["# Worker Benchmark Report", ""]

    if report.machine.gpu_name:
        lines.append(f"- GPU: {report.machine.gpu_name} ({report.machine.total_vram_mb} MB VRAM)")
    if report.machine.total_ram_bytes:
        lines.append(f"- RAM: {report.machine.total_ram_bytes / 1024**3:.0f} GB")
    lines.append("")

    if report.capabilities.tiers:
        lines.extend(_render_capabilities(report.capabilities))

    lines.append("## Levels")
    lines.append("")
    lines.append("| Level | Outcome | it/s p50 | GPU busy % | VRAM HW (MB) | Notes |")
    lines.append("|---|---|---|---|---|---|")
    for level_report in report.levels:
        stats = level_report.stats
        its = f"{stats.its_p50:.2f}" if stats and stats.its_p50 is not None else "-"
        gpu = (
            f"{stats.gpu_utilization_mean_percent:.0f}"
            if stats and stats.gpu_utilization_mean_percent is not None
            else "-"
        )
        vram = str(stats.vram_used_high_water_mb) if stats and stats.vram_used_high_water_mb is not None else "-"
        notes = "; ".join(level_report.reasons + level_report.advisories)
        lines.append(f"| {level_report.level.id} | {level_report.outcome} | {its} | {gpu} | {vram} | {notes} |")
    lines.append("")

    lines.append("## Suggested bridgeData")
    lines.append("")
    lines.append(
        "A conservative recommendation: only models that fit with VRAM headroom are loaded, the batch "
        "size is the largest that passed cleanly, and concurrent alchemy is enabled only if the soak "
        "held up. See the capability table above for everything the worker *can* do.",
    )
    lines.append("")
    lines.append("```yaml")
    lines.append(report.suggested_bridge_data.as_yaml_block())
    lines.append("```")
    lines.append("")
    if report.suggested_bridge_data.notes:
        lines.append("Conservative choices:")
        lines.append("")
        for note in report.suggested_bridge_data.notes:
            lines.append(f"- {note}")
        lines.append("")

    validation_levels = [level for level in report.levels if level.level.stage == BenchStage.VALIDATION]
    if validation_levels:
        lines.append("## Validation (sustained load)")
        lines.append("")
        lines.append(
            "Soaks the recommended config above under continuous, mostly-max-config mixed "
            "traffic, confirming throughput holds and nothing degrades over time.",
        )
        lines.append("")
        lines.append(
            "| Tier | Verdict | Duration (s) | Jobs done | GPU duty | it/s retained | Faults | Recoveries |",
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for level_report in validation_levels:
            stats = level_report.stats
            duration = level_report.level.scenario.soak_seconds
            duration_str = f"{duration:.0f}" if duration is not None else "-"
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
                f"| {level_report.level.tier} | {level_report.outcome} | {duration_str} | {jobs} | "
                f"{gpu_duty} | {retention} | {faults} | {recoveries} |",
            )
        lines.append("")

    timed_levels = [
        level for level in (validation_levels or report.levels) if level.stats and level.stats.phase_breakdown_seconds
    ]
    if timed_levels:
        lines.append("## Where the time goes (per job, median seconds)")
        lines.append("")
        lines.append(
            "Median time a typical image job spends in each phase. `other_inference` is the "
            "inference time outside model load, sampling and VAE (graph build, prompt encode, "
            "image encode, IPC) — the prime target for raising GPU duty.",
        )
        lines.append("")
        for level_report in timed_levels:
            breakdown = level_report.stats.phase_breakdown_seconds  # type: ignore[union-attr]
            total = sum(breakdown.values())
            busy_ratio = level_report.stats.span_derived_busy_ratio  # type: ignore[union-attr]
            busy_note = f", GPU-busy phases ≈ {busy_ratio:.0%} of it" if busy_ratio is not None else ""
            lines.append(f"**{level_report.level.id}** (total ≈ {total:.2f}s/job{busy_note})")
            lines.append("")
            lines.append("| Phase | Seconds | % of total |")
            lines.append("|---|---|---|")
            for phase in _PHASE_ORDER:
                if phase not in breakdown:
                    continue
                seconds = breakdown[phase]
                pct = (seconds / total * 100) if total > 0 else 0.0
                lines.append(f"| {phase} | {seconds:.3f} | {pct:.0f}% |")
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


__all__ = [
    "BENCHMARK_REPORT_SCHEMA_VERSION",
    "BenchmarkReport",
    "Finding",
    "HarnessSummary",
    "LevelReport",
    "LevelRunResult",
    "MachineInfo",
    "SuggestedBridgeData",
    "TierCapability",
    "WorkerCapabilities",
    "compute_level_stats",
    "render_markdown",
    "synthesize_bridge_data",
    "synthesize_capabilities",
]
