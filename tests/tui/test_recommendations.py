"""Unit tests for the live insights/recommendations engine."""

from __future__ import annotations

from horde_worker_regen.process_management.supervisor_channel import (
    ProcessSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.recommendations import Severity, analyze


def _config(**overrides: object) -> WorkerConfigSummary:
    base: dict[str, object] = {"dreamer_name": "Test", "worker_version": "12.0.0"}
    base.update(overrides)
    return WorkerConfigSummary(**base)  # type: ignore[arg-type]


def _snapshot(**overrides: object) -> WorkerStateSnapshot:
    base: dict[str, object] = {"config": _config()}
    base.update(overrides)
    return WorkerStateSnapshot(**base)  # type: ignore[arg-type]


def test_healthy_snapshot_reports_no_issues() -> None:
    """A nominal snapshot yields a single informational 'no issues' item."""
    result = analyze(_snapshot())
    assert len(result) == 1
    assert result[0].severity is Severity.INFO
    assert "No issues" in result[0].title


def test_consecutive_failures_is_critical_and_sorted_first() -> None:
    """A consecutive-failure flag produces a CRITICAL item sorted to the top."""
    result = analyze(_snapshot(too_many_consecutive_failed_jobs=True))
    assert result[0].severity is Severity.CRITICAL


def test_high_fault_rate_warns() -> None:
    """A high fault rate over a meaningful sample warns."""
    result = analyze(_snapshot(num_jobs_submitted=80, num_jobs_faulted=20))
    assert any(item.severity is Severity.WARNING and "fault rate" in item.title.lower() for item in result)


def test_vram_pressure_warns() -> None:
    """A process near its VRAM ceiling warns."""
    process = ProcessSnapshot(
        process_id=0,
        process_type="INFERENCE",
        last_process_state="INFERENCE_STARTING",
        is_alive=True,
        is_busy=True,
        vram_usage_mb=23000,
        vram_used_high_water_mb=23500,
        total_vram_mb=24000,
    )
    result = analyze(_snapshot(processes=[process]))
    assert any("VRAM pressure" in item.title for item in result)


def test_low_duty_cycle_with_work_suggests_tuning() -> None:
    """A low GPU duty cycle while work is queued yields a tuning suggestion."""
    result = analyze(
        _snapshot(
            gpu_utilization_mean_percent=30.0,
            jobs_pending_inference=2,
            config=_config(max_threads=1, queue_size=0),
        ),
    )
    assert any("duty cycle" in item.title.lower() for item in result)
