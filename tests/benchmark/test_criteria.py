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

    def test_its_degradation_is_advisory_when_gate_disabled(self) -> None:
        """With the baseline gate off, a large it/s drop is an advisory, not a failure.

        This is how batch/thread/feature levels are scored: a controlnet or hires-fix pass
        legitimately runs slower than plain txt2img, so it must not fail a healthy machine.
        """
        baseline = TierBaseline(tier="sd15", its_p50=5.0)
        verdict = evaluate_level(
            _clean_stats(its_p50=1.4),  # 28% of baseline, like the real controlnet level
            LevelCriteria(gate_its_against_baseline=False),
            baseline,
        )
        assert verdict.passed
        assert verdict.reasons == []
        assert any("of the tier baseline" in advisory for advisory in verdict.advisories)

    def test_no_baseline_means_no_its_check(self) -> None:
        """Without a tier baseline the it/s check is skipped."""
        verdict = evaluate_level(_clean_stats(its_p50=0.1), LevelCriteria())
        assert verdict.passed

    def test_soak_throughput_retention_failure(self) -> None:
        """A second-half sampling rate below the retention floor fails a soak level."""
        verdict = evaluate_level(
            _clean_stats(its_retention_fraction=0.6),
            LevelCriteria(min_its_retention=0.85, gate_its_against_baseline=False),
        )
        assert not verdict.passed
        assert any("degraded under sustained load" in reason for reason in verdict.reasons)

    def test_soak_throughput_retention_pass_is_advisory(self) -> None:
        """Held-up throughput passes and is noted as an advisory."""
        verdict = evaluate_level(
            _clean_stats(its_retention_fraction=0.97),
            LevelCriteria(min_its_retention=0.85, gate_its_against_baseline=False),
        )
        assert verdict.passed
        assert any("throughput held" in advisory for advisory in verdict.advisories)

    def test_soak_min_completed_jobs(self) -> None:
        """A soak that processed too few jobs is not a meaningful sustained-load test."""
        verdict = evaluate_level(
            _clean_stats(num_jobs_expected=2, num_jobs_completed=2),
            LevelCriteria(min_completed_jobs=10),
        )
        assert not verdict.passed
        assert any("meaningful sustained-load test" in reason for reason in verdict.reasons)

    def test_retention_check_skipped_without_soak_criteria(self) -> None:
        """A low retention value is ignored when the soak criterion is not set (normal levels)."""
        verdict = evaluate_level(_clean_stats(its_retention_fraction=0.1), LevelCriteria())
        assert verdict.passed

    def test_gpu_duty_below_target_is_advisory(self) -> None:
        """A GPU duty cycle below target is flagged as an advisory, not a failure."""
        verdict = evaluate_level(
            _clean_stats(gpu_utilization_mean_percent=55.0),
            LevelCriteria(target_gpu_utilization_percent=90.0),
        )
        assert verdict.passed
        assert any("below the 90% target" in advisory for advisory in verdict.advisories)

    def test_gpu_duty_meets_target(self) -> None:
        """Meeting the GPU duty-cycle target is noted as a satisfied advisory."""
        verdict = evaluate_level(
            _clean_stats(gpu_utilization_mean_percent=94.0),
            LevelCriteria(target_gpu_utilization_percent=90.0),
        )
        assert verdict.passed
        assert any("met the 90% target" in advisory for advisory in verdict.advisories)

    def test_gpu_duty_below_floor_fails(self) -> None:
        """The duty-cycle floor is a hard gate: below it, the soak level fails."""
        verdict = evaluate_level(
            _clean_stats(gpu_utilization_mean_percent=72.0),
            LevelCriteria(min_gpu_duty_cycle_percent=90.0),
        )
        assert not verdict.passed
        assert any("below the 90% floor" in reason for reason in verdict.reasons)

    def test_gpu_duty_clears_floor_passes(self) -> None:
        """Meeting the duty-cycle floor passes and is noted as a cleared gate."""
        verdict = evaluate_level(
            _clean_stats(gpu_utilization_mean_percent=93.0),
            LevelCriteria(min_gpu_duty_cycle_percent=90.0),
        )
        assert verdict.passed
        assert any("cleared the 90% floor" in advisory for advisory in verdict.advisories)

    def test_gpu_duty_floor_skipped_without_measurement(self) -> None:
        """Without a measured duty cycle the floor cannot gate (no NVML, e.g. CI)."""
        verdict = evaluate_level(
            _clean_stats(gpu_utilization_mean_percent=None),
            LevelCriteria(min_gpu_duty_cycle_percent=90.0),
        )
        assert verdict.passed

    def test_residency_defeated_is_advisory_when_expected(self) -> None:
        """Post-warm-up reloads are flagged only when residency is expected."""
        verdict = evaluate_level(
            _clean_stats(post_warmup_vram_reloads=3),
            LevelCriteria(expect_vram_residency=True),
        )
        assert verdict.passed
        assert any("defeated residency" in advisory for advisory in verdict.advisories)

    def test_reloads_not_flagged_when_residency_not_expected(self) -> None:
        """At NORMAL_VRAM (the baseline) per-job reloads are expected, so they are not flagged."""
        verdict = evaluate_level(
            _clean_stats(post_warmup_vram_reloads=12),
            LevelCriteria(expect_vram_residency=False),
        )
        assert verdict.passed
        assert not any("residency" in advisory for advisory in verdict.advisories)

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
