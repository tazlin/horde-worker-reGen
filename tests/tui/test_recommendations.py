"""Unit tests for the live insights engine: which finding each worker-state signal produces, and its copy."""

from __future__ import annotations

from horde_worker_regen.analysis.finding_kinds import FindingKind, Severity
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    ModelPoolSeatRow,
    ModelPoolSnapshot,
    ProcessSnapshot,
    RecentJobRecord,
    StatsSample,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.recommendations import analyze
from tests.analysis.test_finding_copy_style import _emit_problems


def _config(**overrides: object) -> WorkerConfigSummary:
    base: dict[str, object] = {"dreamer_name": "Test", "worker_version": "12.0.0"}
    base.update(overrides)
    return WorkerConfigSummary(**base)  # type: ignore[arg-type]


def _snapshot(**overrides: object) -> WorkerStateSnapshot:
    base: dict[str, object] = {"config": _config()}
    base.update(overrides)
    return WorkerStateSnapshot(**base)  # type: ignore[arg-type]


def test_healthy_snapshot_reports_no_issues() -> None:
    """A nominal snapshot yields no findings; the view says the worker looks healthy itself."""
    assert analyze(_snapshot()) == []


def test_consecutive_failures_is_critical_and_sorted_first() -> None:
    """A consecutive-failure flag produces a CRITICAL item sorted to the top."""
    result = analyze(_snapshot(too_many_consecutive_failed_jobs=True))
    assert result[0].severity is Severity.CRITICAL


def test_high_fault_rate_warns() -> None:
    """A high fault rate over a meaningful sample warns."""
    result = analyze(_snapshot(num_jobs_submitted=80, num_jobs_faulted=20))
    assert any(item.severity is Severity.WARNING and item.kind is FindingKind.FAULT_RATE for item in result)


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
    assert any(item.kind is FindingKind.VRAM_PRESSURE for item in result)


def test_low_duty_cycle_with_work_suggests_tuning() -> None:
    """A low GPU duty cycle while work is queued yields a tuning suggestion."""
    result = analyze(
        _snapshot(
            gpu_utilization_mean_percent=30.0,
            jobs_pending_inference=2,
            config=_config(max_threads=1, queue_size=0),
        ),
    )
    assert any(item.kind is FindingKind.LOW_DUTY_CYCLE for item in result)


def _jobs(models: list[str]) -> list[RecentJobRecord]:
    return [RecentJobRecord(job_id=f"j{index}", model_name=model) for index, model in enumerate(models)]


def _seat(model: str, **overrides: object) -> ModelPoolSeatRow:
    base: dict[str, object] = {"model": model, "source": "RANKER", "state": "ACTIVE"}
    base.update(overrides)
    return ModelPoolSeatRow(**base)  # type: ignore[arg-type]


def test_pool_off_measured_swaps_offer_the_trade() -> None:
    """With the pool off and actual model-swap churn, a variety trade-off nudge appears."""
    result = analyze(_snapshot(latest_stats_sample=StatsSample(churn_counts={"model_swap": 4})))
    hit = next((item for item in result if item.kind is FindingKind.MODEL_POOL_OFF_SWAPS), None)
    assert hit is not None
    assert hit.severity is Severity.SUGGESTION
    # Framed as a genuine preference, not a directive.
    assert "variety" in hit.detail.lower()


def test_pool_off_distinct_models_without_measured_swaps_stays_quiet() -> None:
    """Distinct recent models alone do not prove that the worker displaced any resident model."""
    result = analyze(_snapshot(recent_jobs=_jobs(["A", "B", "C", "D", "E", "F", "G", "H"])))
    assert not any(item.kind is FindingKind.MODEL_POOL_OFF_SWAPS for item in result)


def test_pool_on_stale_demand_warns() -> None:
    """A pool whose demand reading is far too old warns that the ranker is flying blind."""
    pool = ModelPoolSnapshot(enabled=True, seats=[_seat("A")], demand_age_seconds=1200.0)
    result = analyze(_snapshot(model_pool=pool))
    assert any(
        item.severity is Severity.WARNING and item.kind is FindingKind.MODEL_POOL_STALE_DEMAND for item in result
    )


def test_pool_on_unproductive_seat_suggests_review() -> None:
    """A seat taking many empty pops is flagged for rotation/ranker review."""
    pool = ModelPoolSnapshot(enabled=True, seats=[_seat("A", empty_pops=6)], demand_age_seconds=30.0)
    result = analyze(_snapshot(model_pool=pool))
    assert any(item.kind is FindingKind.MODEL_POOL_UNPRODUCTIVE_SEATS for item in result)


def test_pool_on_charged_download_is_not_reported_as_budget_blocked() -> None:
    """A charged pending transfer may finish and must not be described as blocked by its own budget charge."""
    pool = ModelPoolSnapshot(
        enabled=True,
        seats=[_seat("A", state="PENDING_DOWNLOAD", pending_model="Flux.1-dev")],
        demand_age_seconds=30.0,
        download_budget_gb=10.0,
        download_bytes_charged=10 * 1024**3,
    )
    result = analyze(_snapshot(model_pool=pool))
    assert not any("budget" in item.headline.lower() for item in result)


def test_pool_on_resident_matches_are_noted_positively() -> None:
    """A recent pop match is positive only when its model was resident at acceptance time."""
    pool = ModelPoolSnapshot(
        enabled=True,
        seats=[
            _seat(
                "A",
                last_fulfilled_age_seconds=20.0,
                last_match_was_resident=True,
                dwell_seconds=300.0,
            ),
        ],
        demand_age_seconds=30.0,
    )
    result = analyze(_snapshot(model_pool=pool))
    assert any(item.kind is FindingKind.MODEL_POOL_RESIDENT_MATCHES for item in result)


def test_pool_on_cold_match_does_not_claim_residency_benefit() -> None:
    """A fresh cold pop match does not claim that the pool avoided a model load."""
    pool = ModelPoolSnapshot(
        enabled=True,
        seats=[_seat("A", last_fulfilled_age_seconds=20.0, last_match_was_resident=False, dwell_seconds=300.0)],
        demand_age_seconds=30.0,
    )

    result = analyze(_snapshot(model_pool=pool))

    assert not any(item.kind is FindingKind.MODEL_POOL_RESIDENT_MATCHES for item in result)


def test_pool_on_resident_match_line_suppressed_when_a_pool_issue_fires() -> None:
    """The positive line is withheld while a pool problem (here, stale demand) is outstanding."""
    pool = ModelPoolSnapshot(
        enabled=True,
        seats=[
            _seat(
                "A",
                last_fulfilled_age_seconds=20.0,
                last_match_was_resident=True,
                dwell_seconds=300.0,
            ),
        ],
        demand_age_seconds=1200.0,
    )
    result = analyze(_snapshot(model_pool=pool))
    assert not any(item.kind is FindingKind.MODEL_POOL_RESIDENT_MATCHES for item in result)
    assert any(item.kind is FindingKind.MODEL_POOL_STALE_DEMAND for item in result)


def test_every_live_finding_follows_the_copy_guide() -> None:
    """Each signal's finding, once emitted, reads as docs/how-to/write-a-finding.md asks."""
    process = ProcessSnapshot(
        process_id=3,
        process_type="INFERENCE",
        last_process_state="INFERENCE_STARTING",
        is_alive=True,
        is_busy=True,
        vram_usage_mb=22500,
        vram_used_high_water_mb=22800,
        total_vram_mb=24000,
    )
    snapshots = [
        _snapshot(too_many_consecutive_failed_jobs=True, maintenance_mode=True),
        _snapshot(num_jobs_submitted=80, num_jobs_faulted=20, processes=[process]),
        _snapshot(
            gpu_utilization_mean_percent=20.0,
            jobs_in_progress=1,
            config=_config(max_threads=1, queue_size=0, extra_slow_worker=True, max_batch=4),
        ),
        _snapshot(gpu_utilization_mean_percent=20.0, jobs_in_progress=1, config=_config(max_threads=2, queue_size=1)),
        _snapshot(time_spent_no_jobs_available=1200.0),
        _snapshot(latest_stats_sample=StatsSample(churn_counts={"model_swap": 4})),
        _snapshot(model_pool=ModelPoolSnapshot(enabled=True, seats=[], demand_age_seconds=2000.0)),
        _snapshot(model_pool=ModelPoolSnapshot(enabled=True, seats=[_seat("stable_diffusion", empty_pops=6)])),
        _snapshot(
            model_pool=ModelPoolSnapshot(
                enabled=True,
                seats=[_seat("Deliberate", last_fulfilled_age_seconds=30.0, last_match_was_resident=True)],
            ),
        ),
    ]
    findings = [finding for snapshot in snapshots for finding in analyze(snapshot)]
    assert {finding.kind for finding in findings} >= {
        FindingKind.CONSECUTIVE_FAILURE_PAUSE,
        FindingKind.FORCED_MAINTENANCE,
        FindingKind.FAULT_RATE,
        FindingKind.VRAM_PRESSURE,
        FindingKind.LOW_DUTY_CYCLE,
        FindingKind.EXTRA_SLOW_BATCHING,
        FindingKind.LOW_DEMAND_IDLE,
        FindingKind.MODEL_POOL_OFF_SWAPS,
        FindingKind.MODEL_POOL_STALE_DEMAND,
        FindingKind.MODEL_POOL_UNPRODUCTIVE_SEATS,
        FindingKind.MODEL_POOL_RESIDENT_MATCHES,
    }
    problems = [f"{finding.id}: {problem}" for finding in findings for problem in _emit_problems(finding)]
    assert not problems, "\n".join(problems)
