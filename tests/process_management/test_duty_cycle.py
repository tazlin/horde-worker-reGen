"""Unit tests for the shared duty-cycle math and idle attribution (no processes, no GPU)."""

from __future__ import annotations

import time

from hordelib.metrics import JobPhaseMetrics, ModelLoadEvent, SamplingStats

from horde_worker_regen.process_management.duty_cycle import (
    format_phase_gaps,
    phase_breakdown,
    summarize_duty_cycle,
)
from horde_worker_regen.process_management.run_metrics import JobMetricsRecord


def _phase_rich_job(job_id: str = "a") -> JobMetricsRecord:
    """A finished image job whose timeline and phase metrics yield every phase bucket.

    Timeline: popped@100, inference 102.5->115 (12.5s), safety to 116, submit-ready, finalized@116.5.
    Phase metrics fit inside the 12.5s inference window so ``other_inference`` stays positive.
    """
    return JobMetricsRecord(
        job_id=job_id,
        time_popped=100.0,
        stage_timestamps={
            "PENDING_INFERENCE": 100.0,
            "INFERENCE_IN_PROGRESS": 102.5,
            "PENDING_SAFETY_CHECK": 115.0,
            "PENDING_SUBMIT": 116.0,
            "FINALIZED": 116.5,
        },
        queue_wait_seconds=2.5,
        phase_metrics=JobPhaseMetrics(
            model_loads=[
                ModelLoadEvent(model_name="Deliberate", phase="disk_to_ram", duration_seconds=4.2, timestamp=0.0),
                ModelLoadEvent(model_name="Deliberate", phase="ram_to_vram", duration_seconds=1.1, timestamp=0.0),
            ],
            sampling=SamplingStats(
                steps_completed=30,
                total_steps=30,
                duration_seconds=6.0,
                iterations_per_second=5.0,
            ),
            vram_used_high_water_mb=5000,
            ram_used_high_water_mb=9000,
        ),
    )


class TestPhaseBreakdownAndGaps:
    """Phase attribution and the friendly gap summary used in logs and the report."""

    def test_breakdown_covers_every_phase(self) -> None:
        """A phase-rich job yields queue_wait/disk/vram/sampling/safety/submit and a residual other."""
        breakdown = phase_breakdown([_phase_rich_job()])
        assert breakdown["queue_wait"] == 2.5
        assert breakdown["disk_load"] == 4.2
        assert breakdown["vram_load"] == 1.1
        assert breakdown["sampling"] == 6.0
        assert breakdown["safety"] == 1.0
        assert round(breakdown["submit"], 3) == 0.5
        # inference window 12.5s minus disk(4.2)+vram(1.1)+sampling(6.0)+vae(0) = 1.2s of other.
        assert round(breakdown["other_inference"], 3) == 1.2

    def test_gap_summary_names_biggest_worker_side_phases(self) -> None:
        """The summary lists the largest non-GPU per-job gaps with human labels, biggest first."""
        gaps = format_phase_gaps(phase_breakdown([_phase_rich_job()]))
        assert gaps == "model load (disk) 4.2s/job, queue wait 2.5s/job"

    def test_alchemy_jobs_are_excluded(self) -> None:
        """Alchemy forms never pass through the image phases, so they contribute nothing to attribution."""
        alchemy = JobMetricsRecord(job_id="x", is_alchemy=True)
        assert phase_breakdown([alchemy]) == {}

    def test_encode_and_graph_overhead_peeled_out_of_residual(self) -> None:
        """When the engine reports clip/vae-encode and pipeline framing, they become named buckets.

        The same 12.5s inference window now also carries clip_encode 0.8 + vae_encode 0.2 (encode 1.0)
        and pipeline_setup 0.3 + pipeline_validate 0.1 + pipeline_finalize 0.2 (graph_overhead 0.6),
        so ``other_inference`` shrinks from 1.2 to 1.2 - 1.0 - 0.6 floored at 0.
        """
        job = _phase_rich_job()
        assert job.phase_metrics is not None
        job.phase_metrics.phase_seconds = {
            "vae_decode": 0.0,
            "clip_encode": 0.8,
            "vae_encode": 0.2,
            "pipeline_setup": 0.3,
            "pipeline_validate": 0.1,
            "pipeline_finalize": 0.2,
        }
        breakdown = phase_breakdown([job])
        assert round(breakdown["encode"], 3) == 1.0
        assert round(breakdown["graph_overhead"], 3) == 0.6
        # 12.5 - disk(4.2) - vram(1.1) - sampling(6.0) - vae(0) - encode(1.0) - graph(0.6) = -0.4 -> 0.
        assert breakdown["other_inference"] == 0.0

    def test_encode_and_graph_overhead_absent_on_older_engines(self) -> None:
        """With no encode/pipeline phase_seconds, the new buckets are omitted and other_inference is unchanged."""
        breakdown = phase_breakdown([_phase_rich_job()])
        assert "encode" not in breakdown
        assert "graph_overhead" not in breakdown
        assert round(breakdown["other_inference"], 3) == 1.2

    def test_model_unload_surfaced_without_touching_the_residual(self) -> None:
        """Engine-reported model_unload (a between-jobs eviction) is its own gap, not part of other_inference."""
        job = _phase_rich_job()
        assert job.phase_metrics is not None
        job.phase_metrics.phase_seconds = {"model_unload": 0.9}
        breakdown = phase_breakdown([job])
        assert breakdown["model_unload"] == 0.9
        # other_inference is unchanged from the no-phase-seconds case: unload is outside the window.
        assert round(breakdown["other_inference"], 3) == 1.2


class TestSummarizeDutyCycle:
    """The headline figure and the demand-limited vs efficiency-limited idle split."""

    def test_nvml_is_the_headline_when_measured(self) -> None:
        """When NVML reports, it is the headline duty cycle and the source says so."""
        summary = summarize_duty_cycle(
            [_phase_rich_job()],
            window_seconds=300.0,
            time_spent_no_jobs_available=30.0,
            nvml_mean_percent=72.0,
            nvml_busy_fraction=0.9,
        )
        assert summary.effective_duty_percent() == 72.0
        assert summary.headline_source() == "nvml"
        assert summary.completed_jobs == 1
        assert summary.no_jobs_available_fraction == 30.0 / 300.0

    def test_falls_back_to_phase_derived_without_nvml(self) -> None:
        """Off NVML (e.g. non-NVIDIA), the phase-derived ratio carries the headline."""
        summary = summarize_duty_cycle([_phase_rich_job()], window_seconds=300.0)
        assert summary.headline_source() == "phase-derived"
        duty = summary.effective_duty_percent()
        assert duty is not None
        # GPU-touching phases (vram 1.1 + sampling 6.0) / total per-job wall (16.3s).
        assert 40.0 < duty < 50.0

    def test_demand_limited_when_idle_and_no_completions(self) -> None:
        """A window with no completed jobs but meaningful no-jobs time reads as demand-limited."""
        summary = summarize_duty_cycle(
            [],
            window_seconds=200.0,
            time_spent_no_jobs_available=60.0,
            nvml_mean_percent=5.0,
        )
        assert summary.completed_jobs == 0
        assert summary.is_demand_limited() is True

    def test_not_demand_limited_when_idle_is_trivial(self) -> None:
        """A tiny no-jobs share is not treated as demand-limited."""
        summary = summarize_duty_cycle([], window_seconds=200.0, time_spent_no_jobs_available=2.0)
        assert summary.is_demand_limited() is False

    def test_no_jobs_fraction_is_clamped(self) -> None:
        """No-jobs time exceeding the window cannot push the fraction above 1.0."""
        summary = summarize_duty_cycle([], window_seconds=100.0, time_spent_no_jobs_available=250.0)
        assert summary.no_jobs_available_fraction == 1.0

    def test_empty_window_has_no_headline(self) -> None:
        """With neither NVML nor completed jobs there is nothing to report."""
        summary = summarize_duty_cycle([], window_seconds=300.0)
        assert summary.effective_duty_percent() is None
        assert summary.headline_source() == "none"
        assert summary.format_gap_summary() == ""

    def test_churn_summary_lists_nonzero_counts_biggest_first(self) -> None:
        """Reload/respawn churn renders largest-first with friendly labels, omitting zeros."""
        summary = summarize_duty_cycle(
            [],
            window_seconds=180.0,
            nvml_mean_percent=50.0,
            churn_counts={"vram_eviction": 18, "model_swap": 23, "process_cycle": 0},
        )
        assert summary.format_churn_summary() == "23 model swaps, 18 VRAM evictions"

    def test_churn_summary_empty_when_no_churn(self) -> None:
        """No churn in the window renders as an empty string (nothing to append to the duty line)."""
        summary = summarize_duty_cycle([], window_seconds=180.0, nvml_mean_percent=90.0)
        assert summary.format_churn_summary() == ""


def test_window_uses_wall_clock_safely() -> None:
    """A zero window does not divide by zero; the no-jobs fraction is simply unknown."""
    summary = summarize_duty_cycle([], window_seconds=0.0, time_spent_no_jobs_available=5.0)
    assert summary.no_jobs_available_fraction is None
    # Sanity: a real time.time() based caller would never pass 0, but the helper must not crash.
    assert time.time() > 0
