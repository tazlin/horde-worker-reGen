"""Table-driven tests for the level pass/fail criteria."""

from __future__ import annotations

import pytest

from horde_worker_regen.benchmark.criteria import (
    LevelCriteria,
    LevelStats,
    TierBaseline,
    evaluate_level,
)


def _clean_stats(**overrides: object) -> LevelStats:
    """A stats object that passes the default criteria, with overrides applied."""
    base = {
        "num_jobs_expected": 4,
        "num_jobs_completed": 4,
        "num_jobs_faulted": 0,
        "num_alchemy_forms_faulted": 0,
        "num_audit_failures": 0,
        "num_process_recoveries": 0,
        "timed_out": False,
        "its_p50": 5.0,
        "vram_used_high_water_mb": 6000,
        "total_vram_mb": 24000,
        "disk_min_free_bytes": 100 * 1024**3,
    }
    base.update(overrides)
    return LevelStats(**base)  # type: ignore[arg-type]


class TestEvaluateLevel:
    """Pass/fail decisions over synthetic stats."""

    def test_clean_run_passes(self) -> None:
        """A clean run with healthy stats passes with no reasons."""
        verdict = evaluate_level(_clean_stats(), LevelCriteria())
        assert verdict.passed
        assert verdict.reasons == []

    @pytest.mark.parametrize(
        ("overrides", "reason_fragment"),
        [
            ({"timed_out": True}, "timed out"),
            ({"num_jobs_completed": 2}, "2/4 jobs completed"),
            ({"num_jobs_faulted": 1}, "1 jobs faulted"),
            ({"num_alchemy_forms_faulted": 1}, "alchemy forms faulted"),
            ({"num_audit_failures": 1}, "audit failures"),
            ({"num_process_recoveries": 1}, "process recoveries"),
            ({"vram_used_high_water_mb": 23500}, "VRAM headroom"),
            ({"disk_min_free_bytes": 1 * 1024**3}, "disk free space"),
        ],
    )
    def test_single_violation_fails(self, overrides: dict, reason_fragment: str) -> None:
        """Each individual violation fails the level with a descriptive reason."""
        verdict = evaluate_level(_clean_stats(**overrides), LevelCriteria())
        assert not verdict.passed
        assert any(reason_fragment in reason for reason in verdict.reasons), verdict.reasons

    def test_its_degradation_vs_baseline(self) -> None:
        """A sampling rate below the baseline fraction fails the level."""
        baseline = TierBaseline(tier="sd15", its_p50=5.0)
        verdict = evaluate_level(_clean_stats(its_p50=3.0), LevelCriteria(), baseline)
        assert not verdict.passed
        assert any("degraded" in reason for reason in verdict.reasons)

    def test_its_within_baseline_fraction_passes(self) -> None:
        """A small dip within the allowed fraction still passes."""
        baseline = TierBaseline(tier="sd15", its_p50=5.0)
        verdict = evaluate_level(_clean_stats(its_p50=4.5), LevelCriteria(), baseline)
        assert verdict.passed

    def test_no_baseline_means_no_its_check(self) -> None:
        """Without a tier baseline the it/s check is skipped."""
        verdict = evaluate_level(_clean_stats(its_p50=0.1), LevelCriteria())
        assert verdict.passed

    def test_slow_downloads_are_advisory_only(self) -> None:
        """A slow download is reported as an advisory, not a failure."""
        verdict = evaluate_level(
            _clean_stats(download_mbps_min=0.5),
            LevelCriteria(min_download_mbps=5.0),
        )
        assert verdict.passed
        assert any("download bandwidth" in advisory for advisory in verdict.advisories)

    def test_missing_optional_stats_do_not_fail(self) -> None:
        """Absent optional metrics (no VRAM, no disk data) never fail a level."""
        verdict = evaluate_level(
            _clean_stats(its_p50=None, vram_used_high_water_mb=None, total_vram_mb=None, disk_min_free_bytes=None),
            LevelCriteria(),
        )
        assert verdict.passed
