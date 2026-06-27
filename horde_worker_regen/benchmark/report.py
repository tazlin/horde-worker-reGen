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
from enum import StrEnum

from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.criteria import LevelStats
from horde_worker_regen.benchmark.enums import BenchAxis, BenchStage, BenchTier, FindingKind, LevelOutcome
from horde_worker_regen.benchmark.ladder import BENCH_TIER_MODELS, RampLevel
from horde_worker_regen.process_management.resources.duty_cycle import (
    PHASE_ORDER,
    phase_breakdown,
    span_derived_busy_ratio,
)
from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord, RunMetricsSnapshot

BENCHMARK_REPORT_SCHEMA_VERSION = 4
"""Bumped when the report schema changes incompatibly; stamped into every report for later reads.

v2 added :class:`WorkerCapabilities` and the conservative-recommendation rationale alongside the
existing ``suggested_bridge_data``. v3 adds per-setting :class:`SuggestionDecision` provenance (the
basis behind every suggested value); older reports parse with an empty ``decisions`` list. v4 adds
``LevelStats.time_spent_no_jobs_available`` and makes the soak's GPU duty cycle an advisory rather
than a hard gate; older reports parse with that field None.
"""

_FAILED_OUTCOMES = frozenset({LevelOutcome.FAILED, LevelOutcome.CRASHED, LevelOutcome.CRASHED_HANG})
"""Outcomes that mean a level *ran and did not pass* (as opposed to never running)."""

_MIN_VRAM_HEADROOM_MB = 1500
"""A tier's model is only recommended for ``models_to_load`` if its peak VRAM leaves this much free."""

_HEADROOM_HINT_MB = 4000
"""When the busiest level still left at least this much VRAM unused, the recommendation adds an
advisory that higher concurrency/batch than the benchmarked rungs is likely safe. The ladder ramps
only to a fixed top rung, so on a large card (e.g. a 24 GB 4090 that peaked ~12 GB) the proven values
under-provision the hardware; the operator should be told the ceiling is the ladder's, not the GPU's."""


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


class SuggestionBasis(StrEnum):
    """Why a suggested setting holds the value it does, so a reader can tell proof from absence.

    The crucial distinction is between a setting that is off because it was *tested and did not work*
    (:attr:`DISABLED_FAILED`) and one that is off merely because it was *never tested*
    (:attr:`UNTESTED_SKIPPED` / :attr:`NOT_IN_LADDER`); the recommendation looks identical in both
    cases but means very different things to an operator.
    """

    PROVEN = "proven"
    """A level on the relevant axis passed; the value is grounded in a real result."""
    DISABLED_FAILED = "disabled_failed"
    """A level on the axis ran and failed/crashed, so the capability is left off."""
    UNTESTED_SKIPPED = "untested_skipped"
    """The axis's only levels were skipped (pre-flight gate or cascade); never proven either way."""
    NOT_IN_LADDER = "not_in_ladder"
    """The axis was not part of this ladder at all (e.g. excluded by --no-features)."""
    CAPPED_VRAM = "capped_vram"
    """Held back to keep VRAM headroom on this machine, not for lack of capability."""
    CAPPED_SOAK = "capped_soak"
    """Downgraded because the sustained-load soak did not hold up under combined load."""


class SuggestionDecision(BaseModel):
    """The provenance of one suggested setting: its value and why the synthesis chose it."""

    setting: str
    value: bool | int | list[str]
    basis: SuggestionBasis
    detail: str = ""


class SuggestedBridgeData(BaseModel):
    """A conservative bridgeData recommendation derived from the stable levels.

    Unlike a raw "highest passing rung" readout, this keeps VRAM headroom (only models that fit with
    headroom are loaded), prefers a batch size that passed without robustness findings, and has
    concurrent alchemy disabled if the sustained-load soak did not hold up. ``decisions`` records the
    basis behind every value (proven / failed / untested / capped) and ``notes`` the human-readable
    downgrades, for the report.
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
    decisions: list[SuggestionDecision] = Field(default_factory=list)
    """Per-setting provenance: the basis behind each suggested value (not part of the bridgeData)."""
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

    level_phase_breakdown = phase_breakdown(metrics.jobs) if metrics is not None else {}

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
        span_derived_busy_ratio=span_derived_busy_ratio(level_phase_breakdown),
        post_warmup_vram_reloads=(_post_warmup_vram_reloads(metrics.jobs) if metrics is not None else None),
        vram_used_high_water_mb=vram_high_water,
        total_vram_mb=total_vram_mb,
        disk_min_free_bytes=disk_min_free,
        download_mbps_min=min(download_rates) if download_rates else None,
        model_load_disk_seconds_median=statistics.median(disk_loads) if disk_loads else None,
        model_load_vram_seconds_median=statistics.median(vram_loads) if vram_loads else None,
        queue_wait_seconds_p95=_percentile(queue_waits, 0.95),
        e2e_seconds_p95=_percentile(e2es, 0.95),
        phase_breakdown_seconds=level_phase_breakdown,
        time_spent_no_jobs_available=metrics.time_spent_no_jobs_available if metrics is not None else None,
    )


def _passed(report: LevelReport) -> bool:
    """Whether a level passed cleanly (terminal pass with no robustness findings)."""
    return report.outcome == LevelOutcome.PASSED and not report.findings


def _axis_passed(levels: list[LevelReport], axis: BenchAxis) -> bool:
    """Whether any level on *axis* passed."""
    return any(report.outcome == LevelOutcome.PASSED for report in levels if report.level.axis == axis)


def _level_did_nothing(report: LevelReport) -> bool:
    """True when a level ran to its end without completing or faulting any job or alchemy form.

    A level that timed out having dispatched no work at all proves nothing about the capability:
    it is closer to 'never exercised' than to 'tested and failed'. Treating it as untested rather
    than failed keeps the recommendation from reporting an axis that never actually ran as a hard,
    proven-off failure.
    """
    stats = report.stats
    if stats is None:
        return False
    activity = (
        stats.num_jobs_completed
        + stats.num_jobs_faulted
        + stats.num_alchemy_forms_completed
        + stats.num_alchemy_forms_faulted
    )
    expected = stats.num_jobs_expected + stats.num_alchemy_forms_expected
    return activity == 0 and expected > 0


def _axis_status(levels: list[LevelReport], *axes: BenchAxis) -> tuple[SuggestionBasis, str]:
    """Classify how one or more axes fared, so a suggestion can record proof versus absence.

    Accepts several axes for capabilities that any of multiple axes can establish (alchemy is proven
    by either lane). Precedence is proven > failed > skipped/untested > not-in-ladder: a single pass
    proves the capability regardless of other rungs, and a level that ran-and-failed *with real
    activity* is a stronger signal than one that was never reached. A failed level that dispatched no
    work (see :func:`_level_did_nothing`) is downgraded to untested rather than reported as a failure.
    """
    relevant = [report for report in levels if report.level.axis in axes]
    if not relevant:
        return SuggestionBasis.NOT_IN_LADDER, ""
    if any(report.outcome == LevelOutcome.PASSED for report in relevant):
        return SuggestionBasis.PROVEN, ""
    failed = [report for report in relevant if report.outcome in _FAILED_OUTCOMES]
    real_failures = [report for report in failed if not _level_did_nothing(report)]
    if real_failures:
        report = real_failures[0]
        return SuggestionBasis.DISABLED_FAILED, "; ".join(report.reasons) or report.outcome.value
    if failed:
        # Every failure on the axis dispatched no work, so the axis was never actually exercised.
        report = failed[0]
        detail = "; ".join(report.reasons) or "level ran but dispatched no work"
        return SuggestionBasis.UNTESTED_SKIPPED, f"never exercised: {detail}"
    for report in relevant:
        if report.outcome == LevelOutcome.SKIPPED:
            return SuggestionBasis.UNTESTED_SKIPPED, "; ".join(report.reasons) or "level was skipped"
    return SuggestionBasis.NOT_IN_LADDER, ""


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


def _vram_headroom_note(
    levels: list[LevelReport],
    models_to_load: list[str],
    total_vram_mb: int | None,
) -> str | None:
    """Advise when the GPU has substantial VRAM the benchmarked rungs never touched.

    ``max_threads``/``queue_size``/``max_batch`` are taken as the highest *tested* rung, which is the
    ladder's fixed ceiling, not the hardware's. On a large card the proven values can badly
    under-provision it. Rather than fabricate untested numbers (which would break the proven-only
    contract), surface the unused headroom so the operator knows higher is likely safe. Returns None
    when VRAM is unknown, nothing is loaded, or the busiest level already used most of the card.
    """
    if total_vram_mb is None or not models_to_load:
        return None
    observed_peaks = [
        report.stats.vram_used_high_water_mb
        for report in levels
        if report.stats is not None and report.stats.vram_used_high_water_mb is not None
    ]
    if not observed_peaks:
        return None
    spare = total_vram_mb - max(observed_peaks)
    if spare < _HEADROOM_HINT_MB:
        return None
    return (
        f"This GPU never used more than {max(observed_peaks)} of {total_vram_mb} MB VRAM across the "
        f"whole run (~{spare} MB free at the busiest point). max_threads, queue_size, and max_batch "
        "above are the highest rungs the ladder tested, not the highest this card can sustain; higher "
        "values are likely safe but were not benchmarked."
    )


_SOAK_MEANINGFUL_MIN_JOBS = 4
"""A failed soak is only proof of *sustained-load* instability if it completed at least this many
jobs. Below it the soak never exercised sustained load at all (it wedged at startup or crashed
early), which is a worker/startup bug to fix, not grounds to stop loading a model whose isolated
baseline passed cleanly."""


class _SoakOutcome(BaseModel):
    """How the sustained-load soak fared, split by what its failure actually proves.

    The conservative recommendation must not drop a model whose isolated baseline passed unless the
    soak gives *real* evidence the model is unstable under load. So a soak failure is graded:

    - ``unstable_tiers``: the soak ran a meaningful number of jobs and then showed hard trouble
      (faults, recoveries, OOM, hang). Genuine evidence against keeping the model resident -> dropped.
    - ``inconclusive_tiers``: the soak failed for a reason that is *not* load instability: it
      completed too few jobs to test sustained load (a startup wedge), or it failed only on a soft
      gate (e.g. the GPU duty-cycle floor). The baseline still passed, so the model is kept resident
      with a caveat rather than silently dropped.
    - ``skipped_tiers``: the soak existed but never ran (a pre-flight gate); no evidence either way.
    """

    unstable_tiers: set[BenchTier] = Field(default_factory=set)
    """Tiers whose soak ran meaningfully and then proved unstable (model dropped)."""
    inconclusive_tiers: set[BenchTier] = Field(default_factory=set)
    """Tiers whose soak failed for a non-stability reason (model kept with a caveat)."""
    inconclusive_reasons: dict[BenchTier, str] = Field(default_factory=dict)
    """Per-tier human-readable explanation of why an inconclusive soak does not disqualify the model."""
    skipped_tiers: set[BenchTier] = Field(default_factory=set)
    """Tiers whose soak existed in the ladder but was skipped (no sustained-load evidence)."""
    any_unstable: bool = False
    """Whether any soak failed with a hard robustness finding (faults, recoveries, OOM, hang). Gates
    concurrent alchemy regardless of job count: a crash is a crash, even if it happened early."""


def _inconclusive_soak_reason(report: LevelReport, completed: int) -> str:
    """Explain why a failed-but-inconclusive soak does not count against keeping the model resident."""
    if completed == 0:
        return (
            "its soak completed no jobs (a worker/startup issue, not sustained-load instability), so "
            "load stability was neither proven nor disproven"
        )
    detail = "; ".join(report.reasons)
    if detail:
        return f"its soak failed without a sustained-load instability finding ({detail})"
    return "its soak failed on a non-stability gate"


def _soak_outcome(levels: list[LevelReport]) -> _SoakOutcome:
    """Summarize the validation soak, grading each failure by what it actually proves about the model."""
    outcome = _SoakOutcome()
    hard = {FindingKind.OOM, FindingKind.HANG, FindingKind.CRASH, FindingKind.PROCESS_RECOVERY, FindingKind.LOST_JOB}
    for report in levels:
        if report.level.stage != BenchStage.VALIDATION or report.outcome == LevelOutcome.PASSED:
            continue
        tier = report.level.tier
        if report.outcome == LevelOutcome.SKIPPED:
            outcome.skipped_tiers.add(tier)
            continue
        has_hard_finding = any(finding.kind in hard for finding in report.findings)
        if has_hard_finding:
            outcome.any_unstable = True
        completed = report.stats.num_jobs_completed if report.stats is not None else 0
        if has_hard_finding and completed >= _SOAK_MEANINGFUL_MIN_JOBS:
            outcome.unstable_tiers.add(tier)
        else:
            outcome.inconclusive_tiers.add(tier)
            outcome.inconclusive_reasons[tier] = _inconclusive_soak_reason(report, completed)
    return outcome


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
            # The QR-code workflow proves the *SDXL-controlnet* path, not classic preprocessor
            # controlnet. allow_controlnet (canny/depth/openpose) is therefore left to the CONTROLNET
            # axis alone: enabling it off a QR-code pass made the worker advertise and accept classic
            # controlnet jobs that the CONTROLNET level had just shown to crash (the 4090 run did
            # exactly this: allow_controlnet:on while the report's own capability table said ✗).
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

    soak = _soak_outcome(levels)

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
        if tier in soak.unstable_tiers:
            notes.append(
                f"{BENCH_TIER_MODELS[tier]} omitted from models_to_load: its sustained-load soak ran and did "
                "not hold up.",
            )
            continue
        if tier in soak.skipped_tiers:
            notes.append(
                f"{BENCH_TIER_MODELS[tier]} omitted from models_to_load: its sustained-load soak was skipped, so "
                "stability under combined load was never validated.",
            )
            continue
        if tier in soak.inconclusive_tiers:
            # The baseline passed and fits with headroom; the soak failure was not load instability, so
            # keep the model resident rather than zeroing out the recommendation, but say so plainly.
            notes.append(
                f"{BENCH_TIER_MODELS[tier]} kept in models_to_load despite an inconclusive soak: "
                f"{soak.inconclusive_reasons[tier]}.",
            )
        models_to_load.append(BENCH_TIER_MODELS[tier])
    suggestion.models_to_load = models_to_load

    headroom_note = _vram_headroom_note(levels, models_to_load, total_vram_mb)
    if headroom_note is not None:
        notes.append(headroom_note)

    concurrent_capped_by_soak = False
    if suggestion.alchemy_allow_concurrent and soak.any_unstable:
        suggestion.alchemy_allow_concurrent = False
        suggestion.alchemy_max_concurrency = 1
        concurrent_capped_by_soak = True
        notes.append("Concurrent alchemy disabled: the sustained-load soak showed instability under combined load.")

    suggestion.notes = notes
    suggestion.decisions = _build_decisions(
        levels,
        suggestion,
        passing_tiers=passing_tiers,
        soak=soak,
        concurrent_capped_by_soak=concurrent_capped_by_soak,
    )
    return suggestion


def _build_decisions(
    levels: list[LevelReport],
    suggestion: SuggestedBridgeData,
    *,
    passing_tiers: list[BenchTier],
    soak: _SoakOutcome,
    concurrent_capped_by_soak: bool,
) -> list[SuggestionDecision]:
    """Record the basis behind every suggested value, distinguishing proof from untested absence.

    The values are already decided; this only annotates *why* each holds, by re-reading how the
    governing axis fared. The single subtlety is concurrent alchemy, whose value may have been
    downgraded from a real pass by an unstable soak: that is reported as a soak cap, not a flag basis.
    """

    def axis_decision(setting: str, value: bool | int, *axes: BenchAxis) -> SuggestionDecision:
        basis, detail = _axis_status(levels, *axes)
        return SuggestionDecision(setting=setting, value=value, basis=basis, detail=detail)

    decisions = [
        axis_decision("max_threads", suggestion.max_threads, BenchAxis.THREADS),
        axis_decision("queue_size", suggestion.queue_size, BenchAxis.QUEUE_SIZE),
        axis_decision("max_batch", suggestion.max_batch, BenchAxis.BATCH),
        axis_decision("allow_controlnet", suggestion.allow_controlnet, BenchAxis.CONTROLNET),
        axis_decision("allow_sdxl_controlnet", suggestion.allow_sdxl_controlnet, BenchAxis.QR_CODE),
        axis_decision("allow_post_processing", suggestion.allow_post_processing, BenchAxis.POST_PROCESSING),
        axis_decision("allow_lora", suggestion.allow_lora, BenchAxis.DOWNLOADS),
        axis_decision("alchemist", suggestion.alchemist, BenchAxis.ALCHEMY_CLIP, BenchAxis.ALCHEMY_GRAPH),
    ]

    if concurrent_capped_by_soak:
        decisions.append(
            SuggestionDecision(
                setting="alchemy_allow_concurrent",
                value=suggestion.alchemy_allow_concurrent,
                basis=SuggestionBasis.CAPPED_SOAK,
                detail="proved out in isolation but the sustained-load soak was unstable under combined load",
            ),
        )
    else:
        decisions.append(
            axis_decision(
                "alchemy_allow_concurrent",
                suggestion.alchemy_allow_concurrent,
                BenchAxis.ALCHEMY_CONCURRENT,
            ),
        )

    decisions.append(_models_to_load_decision(levels, suggestion, passing_tiers=passing_tiers, soak=soak))
    return decisions


def _models_to_load_decision(
    levels: list[LevelReport],
    suggestion: SuggestedBridgeData,
    *,
    passing_tiers: list[BenchTier],
    soak: _SoakOutcome,
) -> SuggestionDecision:
    """Classify the models_to_load recommendation: proven, capped by VRAM/soak, or never proven."""
    if suggestion.models_to_load:
        return SuggestionDecision(
            setting="models_to_load",
            value=list(suggestion.models_to_load),
            basis=SuggestionBasis.PROVEN,
            detail="tier baselines passed; loaded with VRAM headroom",
        )
    if passing_tiers:
        # Every passing tier was held back; report the dominant reason so it does not read as a failure.
        # (Inconclusive soaks keep their model, so they never reach this empty-list branch.)
        basis = (
            SuggestionBasis.CAPPED_SOAK if soak.unstable_tiers or soak.skipped_tiers else SuggestionBasis.CAPPED_VRAM
        )
        return SuggestionDecision(
            setting="models_to_load",
            value=[],
            basis=basis,
            detail="all passing tiers were held back; see notes",
        )
    basis, detail = _axis_status(levels, BenchAxis.BASELINE)
    return SuggestionDecision(setting="models_to_load", value=[], basis=basis, detail=detail)


def verify_suggestion_consistency(report: BenchmarkReport) -> list[str]:
    """Return a message for any enabled capability or loaded model not grounded in a PROVEN basis.

    This is the literal "is the suggestion consistent with what actually ran" guard: a recommendation
    should never turn something *on* (or keep a model resident) on the strength of a level that was
    skipped, failed, or absent. Returns an empty list when the recommendation is fully grounded.
    """
    suggestion = report.suggested_bridge_data
    by_setting = {decision.setting: decision for decision in suggestion.decisions}
    enabled: dict[str, bool | list[str]] = {
        "allow_controlnet": suggestion.allow_controlnet,
        "allow_sdxl_controlnet": suggestion.allow_sdxl_controlnet,
        "allow_post_processing": suggestion.allow_post_processing,
        "allow_lora": suggestion.allow_lora,
        "alchemist": suggestion.alchemist,
        "alchemy_allow_concurrent": suggestion.alchemy_allow_concurrent,
        "models_to_load": list(suggestion.models_to_load),
    }
    problems: list[str] = []
    for setting, value in enabled.items():
        if not value:
            continue
        decision = by_setting.get(setting)
        if decision is None:
            problems.append(f"{setting} is set but carries no provenance decision.")
        elif decision.basis != SuggestionBasis.PROVEN:
            problems.append(f"{setting} is enabled on a '{decision.basis}' basis, not a proven pass.")
    return problems


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


_BASIS_LABELS: dict[SuggestionBasis, str] = {
    SuggestionBasis.PROVEN: "proven (a level passed)",
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

    Shared by the markdown report and the live progress projection so the TUI and the report agree on
    how a basis reads, without either reimplementing the gloss.
    """
    label = _BASIS_LABELS.get(decision.basis, decision.basis.value)
    return _format_setting_value(decision.value), label, decision.detail or ""


def _render_provenance(decisions: list[SuggestionDecision]) -> list[str]:
    """Render the per-setting provenance table: why each suggested value holds.

    The point is to let an operator tell a capability that was *proven off* (tested and failed) from
    one that is merely *untested* (skipped or absent), which the YAML alone cannot convey.
    """
    lines = ["Why each value (provenance):", "", "| Setting | Value | Basis | Detail |", "|---|---|---|---|"]
    for decision in decisions:
        value_text, label, detail = describe_decision(decision)
        lines.append(f"| {decision.setting} | {value_text} | {label} | {detail} |")
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
            "image encode, IPC); the prime target for raising GPU duty.",
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
            for phase in PHASE_ORDER:
                if phase not in breakdown:
                    continue
                seconds = breakdown[phase]
                pct = (seconds / total * 100) if total > 0 else 0.0
                lines.append(f"| {phase} | {seconds:.3f} | {pct:.0f}% |")
            no_jobs = level_report.stats.time_spent_no_jobs_available  # type: ignore[union-attr]
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
    "SuggestionBasis",
    "SuggestionDecision",
    "TierCapability",
    "WorkerCapabilities",
    "compute_level_stats",
    "describe_decision",
    "render_markdown",
    "synthesize_bridge_data",
    "synthesize_capabilities",
    "verify_suggestion_consistency",
]
