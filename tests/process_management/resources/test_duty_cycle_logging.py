"""Tests for the worker's periodic GPU duty-cycle logging (no processes, no GPU)."""

from __future__ import annotations

import time

from loguru import logger

from horde_worker_regen.process_management.resources.duty_cycle import summarize_duty_cycle
from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord
from tests.process_management.conftest import make_testable_process_manager


def _capture_logs(level: str = "DEBUG") -> tuple[list[tuple[str, str]], int]:
    """Attach a loguru sink that records ``(level_name, message)`` and return it plus its handler id."""
    messages: list[tuple[str, str]] = []
    handler_id = logger.add(
        lambda m: messages.append((m.record["level"].name, m.record["message"])),
        level=level,
    )
    return messages, handler_id


def _job_with_gaps(job_id: str = "a", *, finalized_at: float = 12.0) -> JobMetricsRecord:
    """A finished image job with queue/safety/submit gaps (no phase metrics needed for attribution)."""
    base = finalized_at - 12.0
    return JobMetricsRecord(
        job_id=job_id,
        time_popped=base,
        stage_timestamps={
            "INFERENCE_IN_PROGRESS": base + 2.0,
            "PENDING_SAFETY_CHECK": base + 10.0,
            "PENDING_SUBMIT": base + 11.0,
            "FINALIZED": finalized_at,
        },
        queue_wait_seconds=2.0,
    )


class TestLogDutyCycleSummary:
    """The structured ``GPU duty cycle`` line and its severity, by cause."""

    def test_low_duty_logs_warning_with_attribution(self) -> None:
        """Below the warn band, the line is a WARNING that names where the time went and the processes."""
        manager = make_testable_process_manager()
        summary = summarize_duty_cycle(
            [_job_with_gaps()],
            window_seconds=180.0,
            nvml_mean_percent=60.0,
            nvml_busy_fraction=0.8,
        )
        messages, handler_id = _capture_logs()
        try:
            manager._log_duty_cycle_summary(summary, "inf#0=WAITING_FOR_JOB")
        finally:
            logger.remove(handler_id)

        levels = [level for level, _ in messages]
        text = " ".join(message for _, message in messages)
        assert "WARNING" in levels
        assert "GPU duty cycle 60%" in text
        assert "biggest worker-side gaps" in text
        assert "inf#0=WAITING_FOR_JOB" in text

    def test_near_target_is_info(self) -> None:
        """Between the warn band and the target, the shortfall is a gentle INFO, not a warning."""
        manager = make_testable_process_manager()
        summary = summarize_duty_cycle([_job_with_gaps()], window_seconds=180.0, nvml_mean_percent=80.0)
        messages, handler_id = _capture_logs()
        try:
            manager._log_duty_cycle_summary(summary, "inf#0=WAITING_FOR_JOB")
        finally:
            logger.remove(handler_id)
        assert any(level == "INFO" for level, _ in messages)
        assert not any(level == "WARNING" for level, _ in messages)

    def test_healthy_is_debug(self) -> None:
        """At or above target the line is DEBUG, so a healthy worker does not spam INFO."""
        manager = make_testable_process_manager()
        summary = summarize_duty_cycle([_job_with_gaps()], window_seconds=180.0, nvml_mean_percent=95.0)
        messages, handler_id = _capture_logs()
        try:
            manager._log_duty_cycle_summary(summary, "inf#0=JOB_RECEIVED")
        finally:
            logger.remove(handler_id)
        assert messages  # something was logged
        assert all(level == "DEBUG" for level, _ in messages)

    def test_churn_named_on_the_line_when_present(self) -> None:
        """When reload/respawn churn occurred in the window, the duty line names it alongside the gaps."""
        manager = make_testable_process_manager()
        summary = summarize_duty_cycle(
            [_job_with_gaps()],
            window_seconds=180.0,
            nvml_mean_percent=60.0,
            churn_counts={"vram_eviction": 18, "model_swap": 23, "process_cycle": 0},
        )
        messages, handler_id = _capture_logs()
        try:
            manager._log_duty_cycle_summary(summary, "inf#0=WAITING_FOR_JOB")
        finally:
            logger.remove(handler_id)
        text = " ".join(message for _, message in messages)
        assert "reload churn: 23 model swaps, 18 VRAM evictions" in text

    def test_demand_limited_is_info_and_blames_the_horde(self) -> None:
        """A worker the horde left idle reads as demand-limited at INFO, never as a worker fault."""
        manager = make_testable_process_manager()
        summary = summarize_duty_cycle(
            [],
            window_seconds=200.0,
            time_spent_no_jobs_available=120.0,
            nvml_mean_percent=15.0,
        )
        messages, handler_id = _capture_logs()
        try:
            manager._log_duty_cycle_summary(summary, "inf#0=WAITING_FOR_JOB")
        finally:
            logger.remove(handler_id)
        levels = [level for level, _ in messages]
        text = " ".join(message for _, message in messages)
        assert "INFO" in levels
        assert "WARNING" not in levels
        assert "no jobs available" in text


class TestMaybeLogDutyCycle:
    """The throttle/seed gate around the duty-cycle report."""

    def test_seeds_then_throttles(self) -> None:
        """The first call only seeds the baseline; an immediate second call is throttled silent."""
        manager = make_testable_process_manager()
        assert manager._last_duty_cycle_log_time == 0.0

        messages, handler_id = _capture_logs()
        try:
            manager._maybe_log_duty_cycle()
            assert manager._last_duty_cycle_log_time != 0.0
            manager._maybe_log_duty_cycle()
        finally:
            logger.remove(handler_id)

        assert not any("GPU duty cycle" in message for _, message in messages)

    def test_full_path_emits_after_interval(self) -> None:
        """Once a report interval has elapsed, the full path measures the window and logs a line."""
        manager = make_testable_process_manager()
        now = time.time()
        manager._gpu_sampler._timeline.extend([(now - 1.0, 60), (now - 2.0, 60), (now - 3.0, 60)])
        manager._run_metrics._jobs.append(_job_with_gaps(finalized_at=now - 10.0))
        manager._last_duty_cycle_log_time = now - 200.0
        manager._last_no_jobs_seconds_at_duty_log = 0.0

        messages, handler_id = _capture_logs()
        try:
            manager._maybe_log_duty_cycle()
        finally:
            logger.remove(handler_id)

        assert any("GPU duty cycle 60%" in message for _, message in messages)

    def test_churn_observer_is_wired_and_counted_in_window(self) -> None:
        """The scheduler's churn observer feeds run metrics, and the duty line counts in-window churn."""
        manager = make_testable_process_manager()
        # The manager wires the scheduler's churn observer to the run-metrics recorder at construction.
        assert manager._inference_scheduler._churn_observer == manager._run_metrics.record_churn

        now = time.time()
        manager._gpu_sampler._timeline.extend([(now - 1.0, 60), (now - 2.0, 60), (now - 3.0, 60)])
        manager._run_metrics._jobs.append(_job_with_gaps(finalized_at=now - 10.0))
        manager._last_duty_cycle_log_time = now - 200.0
        manager._last_no_jobs_seconds_at_duty_log = 0.0
        manager._inference_scheduler._record_churn("vram_eviction")
        manager._inference_scheduler._record_churn("vram_eviction")
        manager._inference_scheduler._record_churn("model_swap")

        messages, handler_id = _capture_logs()
        try:
            manager._maybe_log_duty_cycle()
        finally:
            logger.remove(handler_id)

        text = " ".join(message for _, message in messages)
        assert "reload churn: 2 VRAM evictions, 1 model swaps" in text


class TestSnapshotPopulatesDutyCycle:
    """The supervisor snapshot (what the TUI trend reads) is fed by the manager's own sampler."""

    def test_snapshot_reads_live_sampler(self) -> None:
        """In normal operation the GPU fields come from the worker's sampler, lighting up the TUI trend."""
        manager = make_testable_process_manager()
        now = time.time()
        manager._gpu_sampler._timeline.extend([(now - 1.0, 70), (now - 2.0, 70), (now - 3.0, 0)])
        manager._gpu_sampler._samples.extend([70, 70, 0])

        snapshot = manager._build_worker_state_snapshot()

        assert snapshot.gpu_utilization_mean_percent is not None
        assert 40.0 < snapshot.gpu_utilization_mean_percent < 50.0
        assert snapshot.gpu_utilization_busy_fraction == 2 / 3
        assert snapshot.gpu_utilization_samples == 3

    def test_snapshot_gpu_fields_none_without_telemetry(self) -> None:
        """With no samples (CPU/fake/non-NVIDIA) the GPU fields stay None/0, exactly as before."""
        manager = make_testable_process_manager()
        snapshot = manager._build_worker_state_snapshot()
        assert snapshot.gpu_utilization_mean_percent is None
        assert snapshot.gpu_utilization_busy_fraction is None
        assert snapshot.gpu_utilization_samples == 0
