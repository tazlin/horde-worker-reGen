"""Pure pass/fail evaluation of one benchmark level's observed statistics.

Kept free of harness/process imports so the policy is trivially table-testable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from horde_worker_regen.process_management.duty_cycle import format_phase_gaps


class LevelCriteria(BaseModel):
    """The stability requirements a level must meet to pass."""

    max_faulted_jobs: int = 0
    max_faulted_alchemy_forms: int = 0
    max_audit_failures: int = 0
    max_process_recoveries: int = 0
    min_its_fraction_of_baseline: float = 0.85
    """Observed it/s p50 must stay within this fraction of the tier's stage-A baseline."""
    gate_its_against_baseline: bool = True
    """Whether the baseline it/s comparison is a pass/fail gate (True) or advisory (False).

    Only meaningful where the level does the same per-step work as the baseline (e.g. the
    queue-depth axis). Batch levels (more images per step), extra-thread levels (per-job
    it/s drops while aggregate throughput rises), and feature levels (hires-fix's second
    pass, post-processing upscalers, a controlnet model) all legitimately lower raw it/s,
    so gating them on the baseline would wrongly fail a healthy configuration. For those,
    set this False: the ratio is still reported as an advisory, but stability (no faults,
    timeouts, OOM, or recoveries) decides the verdict."""
    min_vram_headroom_mb: int = 1500
    """Total VRAM minus the observed high-water mark must never drop below this."""
    min_disk_free_gb: float = 1.0
    min_download_mbps: float | None = None
    """Advisory floor for download levels (a miss is reported but does not fail the level)."""
    min_its_retention: float | None = None
    """Soak only: the sampling rate in the second half of the run must stay at least this
    fraction of the first half. Below it means throughput degraded under sustained load
    (thermal throttling, a VRAM/RAM leak, queue backpressure). None disables the check."""
    min_completed_jobs: int = 0
    """Soak only: the run must complete at least this many jobs to count as a real soak."""
    target_gpu_utilization_percent: float | None = None
    """Advisory GPU duty-cycle target. If the measured mean utilization is below this, the
    level is flagged (but not failed) as leaving GPU uptime on the table. None disables it."""
    min_gpu_duty_cycle_percent: float | None = None
    """Opt-in hard gate on the measured mean NVML GPU core utilization: below this percent the level
    fails. Off by default (None), because a baseline soak is *expected* to miss the 90% north-star until
    the residency/overlap levers land, so failing on it would just cry wolf. The soak instead advises via
    ``target_gpu_utilization_percent`` and reports the full attribution; set this (``--strict-duty``)
    only when enforcing the target on a reference machine."""
    expect_vram_residency: bool = False
    """When True (a residency soak: ``--highvram`` + a worker VRAM budget), models should stay
    resident across jobs, so post-warm-up RAM->VRAM reloads should be ~0. Any such reload is
    flagged as an advisory ("memory pressure defeated residency"). At NORMAL_VRAM (the baseline)
    reloads are expected every job, so this stays False and the signal is not flagged."""


class TierBaseline(BaseModel):
    """Reference numbers established by a tier's stage-A level."""

    tier: str
    its_p50: float


class LevelStats(BaseModel):
    """The distilled, criteria-relevant numbers observed during one level run."""

    num_jobs_expected: int = 0
    num_jobs_completed: int = 0
    num_jobs_faulted: int = 0
    num_alchemy_forms_expected: int = 0
    num_alchemy_forms_completed: int = 0
    num_alchemy_forms_faulted: int = 0
    num_audit_failures: int = 0
    num_process_recoveries: int = 0
    timed_out: bool = False
    its_p50: float | None = None
    its_min: float | None = None
    its_retention_fraction: float | None = None
    """Soak only: median it/s of the second half of completed jobs ÷ that of the first half."""
    gpu_utilization_mean_percent: float | None = None
    """Average GPU core utilization (duty cycle) over the run, when measured."""
    gpu_utilization_busy_fraction: float | None = None
    """Fraction of the run the GPU was at or above the busy threshold, when measured."""
    span_derived_busy_ratio: float | None = None
    """Diagnostic duty-cycle proxy from the per-job phase medians: (vram_load + sampling + vae)
    ÷ the whole per-job wall. Complements the NVML headline by attributing busy time to phases,
    so "where the missing %" went is explainable offline (no Jaeger needed). None without a
    phase breakdown."""
    post_warmup_vram_reloads: int | None = None
    """RAM->VRAM model reloads observed after the first completed job (warm-up excluded). Under
    expected residency this should be 0; a positive count means a model was evicted and reloaded
    — memory pressure defeated residency, or stickiness broke and a process thrashed its
    single-slot RAM cache. None when no phase metrics were captured."""
    vram_used_high_water_mb: int | None = None
    total_vram_mb: int | None = None
    disk_min_free_bytes: int | None = None
    download_mbps_min: float | None = None
    model_load_disk_seconds_median: float | None = None
    model_load_vram_seconds_median: float | None = None
    queue_wait_seconds_p95: float | None = None
    e2e_seconds_p95: float | None = None
    phase_breakdown_seconds: dict[str, float] = {}
    """Median per-job seconds in each pipeline phase (queue_wait, disk_load, vram_load,
    sampling, vae, other_inference, safety, submit) — a "where the time goes" view."""
    time_spent_no_jobs_available: float | None = None
    """Seconds in the run the worker idled because the horde offered no jobs. Lets a low duty cycle be
    split into demand-limited idle (this) versus worker-side hand-off gaps. None when not measured."""


class LevelVerdict(BaseModel):
    """The outcome of evaluating one level."""

    passed: bool
    reasons: list[str] = Field(default_factory=list)
    """Why the level failed (empty when passed)."""
    advisories: list[str] = Field(default_factory=list)
    """Non-fatal observations (e.g. slow downloads)."""


def _duty_cycle_attribution(stats: LevelStats) -> str:
    """A short parenthetical attributing a low duty cycle, or "" when nothing is attributable.

    Turns the headline "the GPU idled" into something actionable by naming the biggest worker-side
    per-job gaps and separating out demand-limited idle (the horde had no work), so an operator reads
    *where* the uptime went straight from the verdict.
    """
    parts: list[str] = []
    gaps = format_phase_gaps(stats.phase_breakdown_seconds)
    if gaps:
        parts.append(f"biggest worker-side gaps: {gaps}")
    if stats.time_spent_no_jobs_available:
        parts.append(
            f"~{stats.time_spent_no_jobs_available:.0f}s with no jobs available (horde demand, not the worker)",
        )
    if stats.span_derived_busy_ratio is not None:
        parts.append(f"GPU-touching phases ~{stats.span_derived_busy_ratio:.0%} of a typical job's wall")
    return f" ({'; '.join(parts)})" if parts else ""


def evaluate_level(
    stats: LevelStats,
    criteria: LevelCriteria,
    baseline: TierBaseline | None = None,
) -> LevelVerdict:
    """Evaluate one level's stats against the criteria (and tier baseline, when known)."""
    reasons: list[str] = []
    advisories: list[str] = []

    if stats.timed_out:
        reasons.append("level timed out before the scenario completed")
    if stats.num_jobs_completed < stats.num_jobs_expected:
        reasons.append(f"only {stats.num_jobs_completed}/{stats.num_jobs_expected} jobs completed")
    if stats.num_alchemy_forms_completed < stats.num_alchemy_forms_expected:
        reasons.append(
            f"only {stats.num_alchemy_forms_completed}/{stats.num_alchemy_forms_expected} alchemy forms completed",
        )
    if criteria.min_completed_jobs > 0 and stats.num_jobs_completed < criteria.min_completed_jobs:
        reasons.append(
            f"soak completed only {stats.num_jobs_completed} jobs (need {criteria.min_completed_jobs} "
            "to be a meaningful sustained-load test)",
        )
    if stats.num_jobs_faulted > criteria.max_faulted_jobs:
        reasons.append(f"{stats.num_jobs_faulted} jobs faulted (max {criteria.max_faulted_jobs})")
    if stats.num_alchemy_forms_faulted > criteria.max_faulted_alchemy_forms:
        reasons.append(
            f"{stats.num_alchemy_forms_faulted} alchemy forms faulted (max {criteria.max_faulted_alchemy_forms})",
        )
    if stats.num_audit_failures > criteria.max_audit_failures:
        reasons.append(f"{stats.num_audit_failures} job-lifecycle audit failures")
    if stats.num_process_recoveries > criteria.max_process_recoveries:
        reasons.append(
            f"{stats.num_process_recoveries} process recoveries (crashed/hung children; "
            f"max {criteria.max_process_recoveries})",
        )

    if baseline is not None and stats.its_p50 is not None and baseline.its_p50 > 0:
        fraction = stats.its_p50 / baseline.its_p50
        if fraction < criteria.min_its_fraction_of_baseline:
            if criteria.gate_its_against_baseline:
                reasons.append(
                    f"sampling rate degraded to {fraction:.0%} of the tier baseline "
                    f"({stats.its_p50:.2f} vs {baseline.its_p50:.2f} it/s)",
                )
            else:
                # The level legitimately does more work per step than the baseline; report
                # the cost so an operator sees the throughput trade-off without it failing.
                advisories.append(
                    f"sampling rate is {fraction:.0%} of the tier baseline "
                    f"({stats.its_p50:.2f} vs {baseline.its_p50:.2f} it/s) — expected for this "
                    "batch/feature profile, not a regression",
                )

    if criteria.min_its_retention is not None and stats.its_retention_fraction is not None:
        if stats.its_retention_fraction < criteria.min_its_retention:
            reasons.append(
                f"throughput degraded under sustained load: second-half sampling rate fell to "
                f"{stats.its_retention_fraction:.0%} of the first half "
                f"(floor {criteria.min_its_retention:.0%})",
            )
        else:
            advisories.append(
                f"sustained-load throughput held at {stats.its_retention_fraction:.0%} of the "
                "first-half rate across the soak",
            )

    if criteria.target_gpu_utilization_percent is not None and stats.gpu_utilization_mean_percent is not None:
        if stats.gpu_utilization_mean_percent < criteria.target_gpu_utilization_percent:
            advisories.append(
                f"GPU duty cycle {stats.gpu_utilization_mean_percent:.0f}% is below the "
                f"{criteria.target_gpu_utilization_percent:.0f}% target — the GPU idled between jobs"
                f"{_duty_cycle_attribution(stats)}; "
                "see the uptime levers (post-processing overlap, queue depth, thread count)",
            )
        else:
            advisories.append(
                f"GPU duty cycle {stats.gpu_utilization_mean_percent:.0f}% met the "
                f"{criteria.target_gpu_utilization_percent:.0f}% target",
            )

    if criteria.min_gpu_duty_cycle_percent is not None and stats.gpu_utilization_mean_percent is not None:
        if stats.gpu_utilization_mean_percent < criteria.min_gpu_duty_cycle_percent:
            reasons.append(
                f"GPU duty cycle {stats.gpu_utilization_mean_percent:.0f}% is below the "
                f"{criteria.min_gpu_duty_cycle_percent:.0f}% floor — the GPU idled between jobs "
                f"under sustained load{_duty_cycle_attribution(stats)}",
            )
        else:
            advisories.append(
                f"GPU duty cycle {stats.gpu_utilization_mean_percent:.0f}% cleared the "
                f"{criteria.min_gpu_duty_cycle_percent:.0f}% floor",
            )

    if criteria.expect_vram_residency and stats.post_warmup_vram_reloads:
        advisories.append(
            f"memory pressure defeated residency: {stats.post_warmup_vram_reloads} RAM->VRAM "
            "reload(s) after warm-up — the soak is not exercising resident models as intended",
        )

    if (
        stats.vram_used_high_water_mb is not None
        and stats.total_vram_mb is not None
        and stats.total_vram_mb > 0
        and (stats.total_vram_mb - stats.vram_used_high_water_mb) < criteria.min_vram_headroom_mb
    ):
        reasons.append(
            f"VRAM headroom dropped to {stats.total_vram_mb - stats.vram_used_high_water_mb} MB "
            f"(floor {criteria.min_vram_headroom_mb} MB)",
        )

    if stats.disk_min_free_bytes is not None and stats.disk_min_free_bytes < criteria.min_disk_free_gb * 1024**3:
        reasons.append(
            f"disk free space dropped to {stats.disk_min_free_bytes / 1024**3:.1f} GB "
            f"(floor {criteria.min_disk_free_gb:.0f} GB)",
        )

    if (
        criteria.min_download_mbps is not None
        and stats.download_mbps_min is not None
        and stats.download_mbps_min < criteria.min_download_mbps
    ):
        advisories.append(
            f"download bandwidth dipped to {stats.download_mbps_min:.1f} MB/s "
            f"(advisory floor {criteria.min_download_mbps:.1f} MB/s) — "
            "ad-hoc lora/ti features may cause job timeouts on this connection",
        )

    return LevelVerdict(passed=not reasons, reasons=reasons, advisories=advisories)
