"""Pure pass/fail evaluation of one benchmark level's observed statistics.

Kept free of harness/process imports so the policy is trivially table-testable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LevelCriteria(BaseModel):
    """The stability requirements a level must meet to pass."""

    max_faulted_jobs: int = 0
    max_faulted_alchemy_forms: int = 0
    max_audit_failures: int = 0
    max_process_recoveries: int = 0
    min_its_fraction_of_baseline: float = 0.85
    """Observed it/s p50 must stay within this fraction of the tier's stage-A baseline."""
    min_vram_headroom_mb: int = 1500
    """Total VRAM minus the observed high-water mark must never drop below this."""
    min_disk_free_gb: float = 10.0
    min_download_mbps: float | None = None
    """Advisory floor for download levels (a miss is reported but does not fail the level)."""


class TierBaseline(BaseModel):
    """Reference numbers established by a tier's stage-A level."""

    tier: str
    its_p50: float


class LevelStats(BaseModel):
    """The distilled, criteria-relevant numbers observed during one level run."""

    num_jobs_expected: int = 0
    num_jobs_completed: int = 0
    num_jobs_faulted: int = 0
    num_alchemy_forms_faulted: int = 0
    num_audit_failures: int = 0
    num_process_recoveries: int = 0
    timed_out: bool = False
    its_p50: float | None = None
    its_min: float | None = None
    vram_used_high_water_mb: int | None = None
    total_vram_mb: int | None = None
    disk_min_free_bytes: int | None = None
    download_mbps_min: float | None = None
    model_load_disk_seconds_median: float | None = None
    model_load_vram_seconds_median: float | None = None
    queue_wait_seconds_p95: float | None = None
    e2e_seconds_p95: float | None = None


class LevelVerdict(BaseModel):
    """The outcome of evaluating one level."""

    passed: bool
    reasons: list[str] = Field(default_factory=list)
    """Why the level failed (empty when passed)."""
    advisories: list[str] = Field(default_factory=list)
    """Non-fatal observations (e.g. slow downloads)."""


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
            reasons.append(
                f"sampling rate degraded to {fraction:.0%} of the tier baseline "
                f"({stats.its_p50:.2f} vs {baseline.its_p50:.2f} it/s)",
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
