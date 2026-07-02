"""Unit tests for the worker health/phase derivation."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    ProcessSnapshot,
    WholeCardResidencyStatus,
    WorkerConfigSummary,
    WorkerFatalConfigError,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.health import (
    HealthStatus,
    WorkerPhase,
    build_offline_checks,
    derive,
    summarize_skips,
)
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


def test_crashed_with_fatal_error_shows_specific_reason() -> None:
    """A fatal config error replaces the generic crash message with its title and remedy detail."""
    fatal = WorkerFatalConfigError(
        title="Worker name problem",
        detail="Worker name 'Foo' is already registered to another account; choose a different name.",
    )
    report = derive(None, SupervisorStatus.CRASHED, None, fatal_error=fatal)
    assert report.phase is WorkerPhase.CRASHED
    assert report.headline == "Worker name problem"
    assert "already registered to another account" in report.detail
    assert "will not restart until this is fixed" in report.detail


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
    connectivity = next(check for check in report.checks if check.name == "API")
    assert connectivity.status is HealthStatus.ERROR


def test_unresponsive_on_stale_snapshot() -> None:
    """A snapshot older than the staleness threshold means the worker is unresponsive."""
    report = derive(_snapshot(processes=[_process("WAITING_FOR_JOB")]), SupervisorStatus.RUNNING, 30.0)
    assert report.phase is WorkerPhase.UNRESPONSIVE
    assert report.severity is HealthStatus.ERROR


def test_stale_but_shutting_down_reads_as_shutting_down_not_unresponsive() -> None:
    """A worker that announced shutdown and then went quiet is tearing down, not wedged.

    This is the clean-stop false alarm: once the control loop ends the inference/safety children and
    unwinds, it stops stamping liveness, so the snapshot ages past the staleness threshold. Because the
    last snapshot said ``shutting_down``, that silence must read as SHUTTING_DOWN, not UNRESPONSIVE.
    """
    report = derive(
        _snapshot(processes=[_process("WAITING_FOR_JOB")], shutting_down=True),
        SupervisorStatus.RUNNING,
        30.0,
    )
    assert report.phase is WorkerPhase.SHUTTING_DOWN
    assert report.severity is HealthStatus.INFO
    assert "teardown" in report.detail.lower()


def test_shutting_down_with_fresh_snapshot_is_still_shutting_down() -> None:
    """A shutting-down worker that is still reporting shows the in-flight-drain detail."""
    report = derive(
        _snapshot(processes=[_process("INFERENCE_STARTING")], shutting_down=True),
        SupervisorStatus.RUNNING,
        0.5,
    )
    assert report.phase is WorkerPhase.SHUTTING_DOWN
    assert "in-flight" in report.detail.lower()


def test_brief_silence_under_base_threshold_is_not_unresponsive() -> None:
    """A silence between the old (8s) and new (20s) base threshold must not read as unresponsive."""
    report = derive(_snapshot(processes=[_process("WAITING_FOR_JOB")]), SupervisorStatus.RUNNING, 15.0)
    assert report.phase is not WorkerPhase.UNRESPONSIVE


def test_download_in_flight_gets_a_longer_staleness_grace() -> None:
    """While a download/load is in flight the unresponsive alarm holds off past the base threshold."""
    downloading = _snapshot(processes=[_process("DOWNLOADING_AUX_MODEL", loaded_horde_model_name="SomeModel")])
    # Past the 20s base budget but within the 90s download budget: not unresponsive (it is warming up).
    report = derive(downloading, SupervisorStatus.RUNNING, 45.0)
    assert report.phase is not WorkerPhase.UNRESPONSIVE
    # Far enough past even the download budget: genuinely stuck.
    stuck = derive(downloading, SupervisorStatus.RUNNING, 120.0)
    assert stuck.phase is WorkerPhase.UNRESPONSIVE


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


def test_gpu_torch_incompatible_is_degraded_error_and_names_the_reason() -> None:
    """An incompatible-PyTorch worker surfaces as a prominent DEGRADED/ERROR report carrying the reason."""
    reason = "PyTorch has no CUDA kernels for NVIDIA GeForce RTX 5070 (compute capability sm_120)."
    report = derive(
        _snapshot(
            processes=[_process("WAITING_FOR_JOB")],
            gpu_torch_incompatible=True,
            gpu_torch_incompatible_reason=reason,
        ),
        SupervisorStatus.RUNNING,
        0.5,
    )
    assert report.phase is WorkerPhase.DEGRADED
    assert report.severity is HealthStatus.ERROR
    assert reason in report.detail
    assert any(check.status is HealthStatus.ERROR and "RTX 5070" in check.detail for check in report.checks)


def test_gpu_torch_incompatible_beats_serving() -> None:
    """The hardware/build mismatch dominates even a (transient) inferencing process state."""
    report = derive(
        _snapshot(processes=[_process("INFERENCE_STARTING")], gpu_torch_incompatible=True),
        SupervisorStatus.RUNNING,
        0.5,
    )
    assert report.phase is WorkerPhase.DEGRADED
    assert report.severity is HealthStatus.ERROR


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
    assert report.phase is WorkerPhase.MAINTENANCE
    assert "horde" in report.detail.lower()


def test_pop_maintenance_mode_detail_is_specific_not_local_pause() -> None:
    """A pop-response maintenance error produces a distinct detail, not the generic 'locally paused' text.

    There is a gap between when the pop loop first sees a maintenance-mode error and when the 15 s
    advisory poll confirms it. During that window maintenance_mode is True but worker_details_maintenance
    is still False. The detail must not mislead the operator into thinking this is a local pause.
    """
    report = derive(
        _snapshot(
            processes=[_process("WAITING_FOR_JOB")],
            maintenance_mode=True,
            last_pop_maintenance_mode=True,
            worker_details_maintenance=False,
            supervisor_paused=False,
        ),
        SupervisorStatus.RUNNING,
        0.5,
    )
    assert report.phase is WorkerPhase.MAINTENANCE
    assert "locally paused" not in report.detail
    assert "maintenance" in report.detail.lower()


def test_maintenance_mode_beats_api_backoff_and_labels_connectivity() -> None:
    """Maintenance is not a network disconnect, even when pop backoff is active."""
    report = derive(
        _snapshot(
            processes=[_process("WAITING_FOR_JOB")],
            last_pop_maintenance_mode=True,
            in_error_backoff=True,
        ),
        SupervisorStatus.RUNNING,
        0.5,
    )

    assert report.phase is WorkerPhase.MAINTENANCE
    connectivity = next(check for check in report.checks if check.name == "API")
    assert connectivity.status is HealthStatus.INFO
    assert "maintenance" in connectivity.detail.lower()


def test_optimistic_server_maintenance_is_shown_before_poll_confirmation() -> None:
    """After the TUI sends maintenance ON, health shows maintenance before worker-details catches up."""
    report = derive(
        _snapshot(processes=[_process("WAITING_FOR_JOB")], worker_details_maintenance=False),
        SupervisorStatus.RUNNING,
        0.5,
        optimistic_server_maintenance=True,
    )

    assert report.phase is WorkerPhase.MAINTENANCE
    assert "requested" in report.detail.lower()
    connectivity = next(check for check in report.checks if check.name == "API")
    assert "maintenance" in connectivity.detail.lower()


def _residency_check(report: object) -> object | None:
    """Return the Residency health check from a report, or None if it was not added."""
    return next((check for check in report.checks if check.name == "Residency"), None)  # type: ignore[attr-defined]


def test_residency_is_not_a_health_row() -> None:
    """Whole-card residency is communicated in the hero/process/panel, not as a health checklist row.

    Even with residency active or armed, no "Residency" row is added: it is not a pass/warn/fail health
    dimension, so it stays out of the checklist to avoid duplicating what the residency banner already says.
    """
    active = derive(
        _snapshot(
            processes=[_process("INFERENCE_STARTING")],
            whole_card_residency=WholeCardResidencyStatus(active=True, model="Flux.1-dev"),
        ),
        SupervisorStatus.RUNNING,
        0.5,
    )
    possible = derive(
        _snapshot(
            processes=[_process("WAITING_FOR_JOB")],
            whole_card_residency=WholeCardResidencyStatus(possible=True),
        ),
        SupervisorStatus.RUNNING,
        0.5,
    )
    assert _residency_check(active) is None
    assert _residency_check(possible) is None


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


def test_offline_checks_surface_in_stopped_report(tmp_path: Path) -> None:
    """A stopped worker shows pre-flight checks (config + disk) instead of an empty checklist."""
    config_path = tmp_path / "bridgeData.yaml"
    config_path.write_text("dreamer_worker_name: Test\n", encoding="utf-8")
    offline = build_offline_checks(config_path)
    report = derive(None, SupervisorStatus.STOPPED, None, offline_checks=offline)
    assert report.phase is WorkerPhase.STOPPED
    names = {check.name for check in report.checks}
    assert {"Config", "Disk"} <= names
    config_check = next(check for check in report.checks if check.name == "Config")
    assert config_check.status is HealthStatus.OK


def test_offline_config_check_warns_when_missing(tmp_path: Path) -> None:
    """A missing config file is a warning (the setup wizard is the intended remedy), not an error."""
    checks = build_offline_checks(tmp_path / "absent.yaml")
    config_check = next(check for check in checks if check.name == "Config")
    assert config_check.status is HealthStatus.WARN


def test_offline_config_check_errors_on_unparseable_yaml(tmp_path: Path) -> None:
    """A present-but-corrupt config surfaces as an error."""
    config_path = tmp_path / "bridgeData.yaml"
    config_path.write_text("dreamer_worker_name: [unterminated\n", encoding="utf-8")
    checks = build_offline_checks(config_path)
    config_check = next(check for check in checks if check.name == "Config")
    assert config_check.status is HealthStatus.ERROR


def test_offline_disk_check_warns_below_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The offline disk check warns when free space is under the floor."""
    import horde_worker_regen.tui.health as health_module

    monkeypatch.setattr(
        health_module.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"total": 0, "used": 0, "free": 1024})(),
    )
    monkeypatch.setenv("AIWORKER_CACHE_HOME", str(tmp_path))
    checks = build_offline_checks(tmp_path / "bridgeData.yaml")
    disk_check = next(check for check in checks if check.name == "Disk")
    assert disk_check.status is HealthStatus.WARN


def test_checks_cover_core_dimensions() -> None:
    """The checklist always reports the core health dimensions."""
    report = derive(
        _snapshot(processes=[_process("WAITING_FOR_JOB")], active_models=["Deliberate"]), SupervisorStatus.RUNNING, 0.5
    )
    names = {check.name for check in report.checks}
    assert {"API", "Disk", "Job health"} <= names


def test_api_check_folds_reachability_and_registration() -> None:
    """A reachable, registered worker reports one 'API' row naming the registered dreamer name."""
    report = derive(
        _snapshot(processes=[_process("WAITING_FOR_JOB")], worker_registered=True),
        SupervisorStatus.RUNNING,
        0.5,
    )
    api = next(check for check in report.checks if check.name == "API")
    assert api.status is HealthStatus.OK
    assert "registered as Test" in api.detail


def test_api_check_notes_pending_registration_when_reachable() -> None:
    """A reachable but not-yet-acknowledged worker reports an INFO 'API' row, not a separate row."""
    report = derive(
        _snapshot(processes=[_process("WAITING_FOR_JOB")], worker_registered=False),
        SupervisorStatus.RUNNING,
        0.5,
    )
    api = next(check for check in report.checks if check.name == "API")
    assert api.status is HealthStatus.INFO
    assert "not yet acknowledged" in api.detail


def test_processes_models_and_gpu_duty_are_not_health_rows() -> None:
    """Process count, model count, and GPU duty moved out of the checklist to titles/Trends."""
    report = derive(
        _snapshot(processes=[_process("WAITING_FOR_JOB")], active_models=["Deliberate"]),
        SupervisorStatus.RUNNING,
        0.5,
    )
    names = {check.name for check in report.checks}
    assert {"Processes", "Models", "GPU", "Registration", "API connectivity"}.isdisjoint(names)


def test_is_gpu_duty_low_flags_idle_gpu_during_a_job() -> None:
    """The Trends low-duty predicate is True only when a job is running against a near-idle GPU."""
    from horde_worker_regen.tui.health import is_gpu_duty_low

    running_idle = _snapshot(processes=[_process("INFERENCE_STARTING")], gpu_utilization_mean_percent=1.0)
    running_busy = _snapshot(processes=[_process("INFERENCE_STARTING")], gpu_utilization_mean_percent=80.0)
    waiting_idle = _snapshot(processes=[_process("WAITING_FOR_JOB")], gpu_utilization_mean_percent=1.0)
    unsampled = _snapshot(processes=[_process("INFERENCE_STARTING")])

    assert is_gpu_duty_low(running_idle) is True
    assert is_gpu_duty_low(running_busy) is False
    assert is_gpu_duty_low(waiting_idle) is False
    assert is_gpu_duty_low(unsampled) is False
