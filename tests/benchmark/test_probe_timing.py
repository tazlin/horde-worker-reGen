"""Unit tests for the pure warmup-versus-inference timing split (no harness, no GPU)."""

from __future__ import annotations

from hordelib.metrics import JobPhaseMetrics, ModelLoadEvent, SamplingStats

from horde_worker_regen.benchmark.capabilities.timing import probe_timing
from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord


def _image_job(
    *,
    inference_start: float,
    finalized: float,
    disk_load: float = 0.0,
    vram_load: float = 0.0,
    sampling: float = 0.0,
    is_alchemy: bool = False,
) -> JobMetricsRecord:
    """A finished image job at the given absolute stage timestamps with optional phase durations."""
    model_loads: list[ModelLoadEvent] = []
    if disk_load > 0:
        model_loads.append(
            ModelLoadEvent(model_name="m", phase="disk_to_ram", duration_seconds=disk_load, timestamp=0)
        )
    if vram_load > 0:
        model_loads.append(
            ModelLoadEvent(model_name="m", phase="ram_to_vram", duration_seconds=vram_load, timestamp=0)
        )
    return JobMetricsRecord(
        job_id="j",
        is_alchemy=is_alchemy,
        stage_timestamps={"INFERENCE_IN_PROGRESS": inference_start, "FINALIZED": finalized},
        phase_metrics=JobPhaseMetrics(
            model_loads=model_loads,
            sampling=(
                SamplingStats(steps_completed=20, total_steps=20, duration_seconds=sampling, iterations_per_second=1.0)
                if sampling > 0
                else None
            ),
        ),
    )


def test_cold_boot_attributes_startup_active_and_teardown() -> None:
    """A per-probe cold boot reads as a long startup before a short productive window, plus drain."""
    jobs = [
        _image_job(inference_start=1040.0, finalized=1070.0, disk_load=10.0, vram_load=8.0, sampling=4.0),
        _image_job(inference_start=1071.0, finalized=1085.0, sampling=4.0, vram_load=2.0),
    ]
    timing = probe_timing(started_at_epoch=1000.0, elapsed_seconds=90.0, jobs=jobs)

    assert timing.startup_seconds == 40.0
    assert timing.active_window_seconds == 45.0
    assert timing.teardown_seconds == 5.0
    assert timing.cold_model_load_seconds == 18.0  # first job: 10 disk + 8 vram
    assert timing.gpu_active_seconds == 8.0 + 4.0 + 4.0 + 2.0  # both jobs' vram_load + sampling
    assert timing.jobs_completed == 2


def test_gpu_active_fraction_exposes_low_duty() -> None:
    """Most of a cold-boot probe's wall is not inference, which the fraction makes explicit."""
    jobs = [_image_job(inference_start=1040.0, finalized=1070.0, vram_load=2.0, sampling=4.0)]
    timing = probe_timing(started_at_epoch=1000.0, elapsed_seconds=80.0, jobs=jobs)

    assert timing.gpu_active_seconds == 6.0
    assert timing.gpu_active_fraction is not None
    assert abs(timing.gpu_active_fraction - 6.0 / 80.0) < 1e-9
    assert "% inference" in timing.summary()


def test_warm_session_shows_negligible_startup() -> None:
    """When the driver starts the clock at the measured pass, boot is already amortized (startup ~0)."""
    first_inference = 5000.0
    jobs = [
        _image_job(inference_start=first_inference, finalized=first_inference + 12.0, sampling=10.0, vram_load=1.0)
    ]
    timing = probe_timing(started_at_epoch=first_inference, elapsed_seconds=13.0, jobs=jobs)

    assert timing.startup_seconds == 0.0
    assert timing.active_window_seconds == 12.0
    assert timing.teardown_seconds == 1.0


def test_unknown_start_epoch_reports_only_the_window() -> None:
    """A driver that did not record the run-start epoch still gets the inference window, not startup."""
    jobs = [_image_job(inference_start=200.0, finalized=230.0, sampling=8.0)]
    timing = probe_timing(started_at_epoch=0.0, elapsed_seconds=60.0, jobs=jobs)

    assert timing.startup_seconds is None
    assert timing.teardown_seconds is None
    assert timing.active_window_seconds == 30.0
    assert timing.total_seconds == 60.0


def test_no_timed_jobs_yields_total_only() -> None:
    """With no completed image job to bound the window, only the total wall is known."""
    timing = probe_timing(started_at_epoch=1000.0, elapsed_seconds=45.0, jobs=[])

    assert timing.total_seconds == 45.0
    assert timing.startup_seconds is None
    assert timing.active_window_seconds is None
    assert timing.gpu_active_seconds is None
    assert timing.jobs_completed == 0


def test_alchemy_jobs_do_not_bound_the_image_window() -> None:
    """Alchemy forms run on other lanes, so they are excluded from the image inference window."""
    jobs = [
        _image_job(inference_start=9000.0, finalized=9000.0, is_alchemy=True, sampling=99.0),
        _image_job(inference_start=1040.0, finalized=1070.0, sampling=4.0),
    ]
    timing = probe_timing(started_at_epoch=1000.0, elapsed_seconds=90.0, jobs=jobs)

    assert timing.startup_seconds == 40.0
    assert timing.active_window_seconds == 30.0
    assert timing.jobs_completed == 1
