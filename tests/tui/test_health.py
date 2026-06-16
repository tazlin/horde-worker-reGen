"""Unit tests for the worker health/phase derivation."""

from __future__ import annotations

from horde_worker_regen.process_management.supervisor_channel import (
    ProcessSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.health import HealthStatus, WorkerPhase, derive, summarize_skips
from horde_worker_regen.tui.worker_launcher import SupervisorStatus


def _process(state: str, **overrides: object) -> ProcessSnapshot:
    base: dict[str, object] = {
        "process_id": 0,
        "process_type": "INFERENCE",
        "last_process_state": state,
        "is_alive": True,
        "is_busy": state != "WAITING_FOR_JOB",
    }
    base.update(overrides)
    return ProcessSnapshot(**base)  # type: ignore[arg-type]


def _snapshot(**overrides: object) -> WorkerStateSnapshot:
    base: dict[str, object] = {
        "config": WorkerConfigSummary(dreamer_name="Test", worker_version="12.0.0"),
        "worker_registered": True,
    }
    base.update(overrides)
    return WorkerStateSnapshot(**base)  # type: ignore[arg-type]


def test_no_snapshot_is_initializing() -> None:
    """Before the first snapshot (while starting) the phase is INITIALIZING."""
    report = derive(None, SupervisorStatus.STARTING, None)
    assert report.phase is WorkerPhase.INITIALIZING
    assert report.animated is True


def test_supervisor_crashed_and_restarting() -> None:
    """Supervisor-level crash/restart states surface directly."""
    assert derive(None, SupervisorStatus.CRASHED, None).phase is WorkerPhase.CRASHED
    assert derive(None, SupervisorStatus.RESTARTING, None).phase is WorkerPhase.RESTARTING
    assert derive(None, SupervisorStatus.STOPPED, None).phase is WorkerPhase.STOPPED


def test_serving_when_a_process_is_inferencing() -> None:
    """A process mid-inference yields the SERVING phase (OK)."""
    report = derive(
        _snapshot(processes=[_process("INFERENCE_STARTING")], num_jobs_submitted=5), SupervisorStatus.RUNNING, 0.5
    )
    assert report.phase is WorkerPhase.SERVING
    assert report.severity is HealthStatus.OK
    assert report.animated is True


def test_disconnected_on_user_info_failure() -> None:
    """A failed user-info call surfaces as DISCONNECTED with an ERROR connectivity check."""
    report = derive(
        _snapshot(
            processes=[_process("WAITING_FOR_JOB")],
            user_info_failed=True,
            user_info_failed_reason="HTTP error",
        ),
        SupervisorStatus.RUNNING,
        0.5,
    )
    assert report.phase is WorkerPhase.DISCONNECTED
    assert report.severity is HealthStatus.ERROR
    connectivity = next(check for check in report.checks if check.name == "API connectivity")
    assert connectivity.status is HealthStatus.ERROR


def test_unresponsive_on_stale_snapshot() -> None:
    """A snapshot older than the staleness threshold means the worker is unresponsive."""
    report = derive(_snapshot(processes=[_process("WAITING_FOR_JOB")]), SupervisorStatus.RUNNING, 30.0)
    assert report.phase is WorkerPhase.UNRESPONSIVE
    assert report.severity is HealthStatus.ERROR


def test_degraded_on_consecutive_failures() -> None:
    """The repeated-failure flag surfaces as DEGRADED."""
    report = derive(
        _snapshot(
            processes=[_process("WAITING_FOR_JOB")], too_many_consecutive_failed_jobs=True, consecutive_failed_jobs=4
        ),
        SupervisorStatus.RUNNING,
        0.5,
    )
    assert report.phase is WorkerPhase.DEGRADED


def test_paused_phase() -> None:
    """Maintenance/paused mode surfaces as PAUSED (warning)."""
    report = derive(
        _snapshot(processes=[_process("WAITING_FOR_JOB")], maintenance_mode=True), SupervisorStatus.RUNNING, 0.5
    )
    assert report.phase is WorkerPhase.PAUSED
    assert report.severity is HealthStatus.WARN


def test_warming_up_while_loading() -> None:
    """Processes still loading, with no job done, is WARMING_UP."""
    report = derive(
        _snapshot(
            processes=[_process("DOWNLOADING_MODEL", loaded_horde_model_name="Deliberate")], num_jobs_submitted=0
        ),
        SupervisorStatus.RUNNING,
        0.5,
    )
    assert report.phase is WorkerPhase.WARMING_UP
    assert "downloading" in report.headline.lower()


def test_ready_and_idle() -> None:
    """Waiting processes are READY; long idle time becomes IDLE."""
    ready = derive(
        _snapshot(processes=[_process("WAITING_FOR_JOB")], seconds_since_last_pop=5.0), SupervisorStatus.RUNNING, 0.5
    )
    assert ready.phase is WorkerPhase.READY

    idle = derive(
        _snapshot(processes=[_process("WAITING_FOR_JOB")], time_spent_no_jobs_available=1200.0),
        SupervisorStatus.RUNNING,
        0.5,
    )
    assert idle.phase is WorkerPhase.IDLE


def test_worker_details_maintenance_is_paused_and_names_the_horde() -> None:
    """A worker the horde has placed in maintenance surfaces as PAUSED, attributed to the horde."""
    report = derive(
        _snapshot(processes=[_process("WAITING_FOR_JOB")], worker_details_maintenance=True),
        SupervisorStatus.RUNNING,
        0.5,
    )
    assert report.phase is WorkerPhase.PAUSED
    assert "horde" in report.detail.lower()


def test_short_term_no_jobs_is_ready_with_skip_reasons() -> None:
    """A pop that returned no job stays READY but explains why, including the skip-reason breakdown."""
    report = derive(
        _snapshot(
            processes=[_process("WAITING_FOR_JOB")],
            last_pop_no_jobs_available=True,
            last_pop_skipped_reasons={"models": 3, "nsfw": 1},
        ),
        SupervisorStatus.RUNNING,
        0.5,
    )
    assert report.phase is WorkerPhase.READY
    assert "no jobs available" in report.headline.lower()
    assert "3 models" in report.detail
    work = next(check for check in report.checks if check.name == "Work")
    assert "models" in work.detail


def test_summarize_skips_orders_by_count_and_drops_zeros() -> None:
    """The skip summary is count-ordered and omits zero-count reasons."""
    assert summarize_skips({"nsfw": 1, "models": 3, "untouched": 0}) == "3 models · 1 nsfw"
    assert summarize_skips({}) == ""


def test_checks_cover_core_dimensions() -> None:
    """The checklist always reports the core health dimensions."""
    report = derive(
        _snapshot(processes=[_process("WAITING_FOR_JOB")], active_models=["Deliberate"]), SupervisorStatus.RUNNING, 0.5
    )
    names = {check.name for check in report.checks}
    assert {"API connectivity", "Registration", "Processes", "Models", "GPU", "Disk", "Job health"} <= names
