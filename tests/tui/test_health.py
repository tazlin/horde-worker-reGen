"""Unit tests for the worker health/phase derivation."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    ProcessSnapshot,
    TextBackendDetail,
    WholeCardResidencyStatus,
    WorkerConfigSummary,
    WorkerFatalConfigError,
    WorkerStateSnapshot,
    WorkLedgerEntry,
    WorkLedgerProgressUnit,
    WorkLedgerStage,
)
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind
from horde_worker_regen.tui.health import (
    TEXT_BACKEND_CHECK_NAME,
    TEXT_GENERATION_CHECK_NAME,
    HealthCheck,
    HealthStatus,
    WorkerPhase,
    build_offline_checks,
    derive,
    gpu_duty_low_cards,
    summarize_reduced_skips,
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
    downloading = _snapshot(processes=[_process("DOWNLOADING_MODEL", loaded_horde_model_name="SomeModel")])
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


def test_reduced_size_skips_are_named_with_their_size_only_when_outstanding() -> None:
    """A reduced-size pop's skips render as their own phrase with the size asked; none outstanding renders nothing."""
    assert summarize_reduced_skips(_snapshot()) == ""
    snapshot = _snapshot(last_reduced_pop_skipped_reasons={"max_pixels": 4}, last_reduced_pop_max_power=17)
    assert summarize_reduced_skips(snapshot) == "reduced-size pop (max_power 17): 4 max_pixels"


def test_ready_report_lists_the_regular_and_reduced_lanes_separately() -> None:
    """With both lanes unanswered the report carries a Work line for each; with one, only that one."""
    both = derive(
        _snapshot(
            processes=[_process("WAITING_FOR_JOB")],
            last_pop_no_jobs_available=True,
            last_pop_skipped_reasons={"models": 3},
            last_reduced_pop_skipped_reasons={"max_pixels": 4},
            last_reduced_pop_max_power=17,
        ),
        SupervisorStatus.RUNNING,
        0.0,
    )
    work = [check.detail for check in both.checks if check.name == "Work"]
    assert len(work) == 2
    assert "3 models" in work[0]
    assert "max_power 17" in work[1]
    assert "max_power 17" in both.detail

    regular_only = derive(
        _snapshot(
            processes=[_process("WAITING_FOR_JOB")],
            last_pop_no_jobs_available=True,
            last_pop_skipped_reasons={"models": 3},
        ),
        SupervisorStatus.RUNNING,
        0.0,
    )
    assert len([check for check in regular_only.checks if check.name == "Work"]) == 1


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


def test_gpu_duty_low_is_decided_per_card() -> None:
    """A card is flagged only on its own evidence: its processes mid-job and its own duty near idle."""
    from horde_worker_regen.tui.health import gpu_duty_low_cards

    busy_card_one = _snapshot(
        processes=[_process("WAITING_FOR_JOB", device_index=0), _process("INFERENCE_STARTING", device_index=1)],
        gpu_utilization_mean_percent=45.0,
        gpu_utilization_mean_percent_per_card={0: 89.0, 1: 1.0},
    )
    assert gpu_duty_low_cards(busy_card_one) == [1]

    # Card 0 is idle because it has no work, not because it is starved; only the working card can qualify.
    idle_card_zero = _snapshot(
        processes=[_process("WAITING_FOR_JOB", device_index=0), _process("INFERENCE_STARTING", device_index=1)],
        gpu_utilization_mean_percent=45.0,
        gpu_utilization_mean_percent_per_card={0: 0.0, 1: 90.0},
    )
    assert gpu_duty_low_cards(idle_card_zero) == []

    both_starved = _snapshot(
        processes=[_process("INFERENCE_STARTING", device_index=0), _process("INFERENCE_STARTING", device_index=1)],
        gpu_utilization_mean_percent=1.0,
        gpu_utilization_mean_percent_per_card={0: 1.0, 1: 2.0},
    )
    assert gpu_duty_low_cards(both_starved) == [0, 1]


def test_gpu_duty_low_falls_back_to_the_worker_wide_figure() -> None:
    """Without per-card duty (a single-card worker, or an older snapshot) the scalar decides."""
    from horde_worker_regen.tui.health import gpu_duty_low_cards, is_gpu_duty_low

    running_idle = _snapshot(processes=[_process("INFERENCE_STARTING")], gpu_utilization_mean_percent=1.0)
    assert gpu_duty_low_cards(running_idle) == [0]
    assert is_gpu_duty_low(running_idle) is True


def test_per_card_duty_survives_a_snapshot_round_trip() -> None:
    """The per-card duty fields cross the supervisor channel intact, keyed by device index."""
    snapshot = _snapshot(
        gpu_utilization_mean_percent=45.0,
        gpu_utilization_busy_fraction=0.5,
        gpu_utilization_samples=8,
        gpu_utilization_mean_percent_per_card={0: 89.0, 1: 1.0},
        gpu_utilization_busy_fraction_per_card={0: 0.9, 1: 0.1},
        gpu_utilization_samples_per_card={0: 4, 1: 4},
    )
    restored = WorkerStateSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored.gpu_utilization_mean_percent_per_card == {0: 89.0, 1: 1.0}
    assert restored.gpu_utilization_busy_fraction_per_card == {0: 0.9, 1: 0.1}
    assert restored.gpu_utilization_samples_per_card == {0: 4, 1: 4}
    assert restored.gpu_utilization_mean_percent == 45.0


def test_model_fit_rows_name_unserviceable_and_constrained_models() -> None:
    """A model the card cannot fit is an ERROR row; one offered at reduced size is a WARN naming the cap."""
    from horde_worker_regen.process_management.ipc.supervisor_channel import CardSnapshot

    snapshot = _snapshot(
        per_card=[
            CardSnapshot(
                device_index=0,
                kind="cuda",
                unserviceable_models=["Flux.1-Schnell fp8 (Compact)"],
                constrained_models={"AlbedoBase XL (SDXL)": 40},
            ),
        ],
    )

    report = derive(snapshot, snapshot_age=1.0, supervisor_status=SupervisorStatus.RUNNING)
    rows = {check.name: check for check in report.checks if check.name == "Models"}
    statuses = [(check.status, check.detail) for check in report.checks if check.name == "Models"]

    assert rows
    assert any(status is HealthStatus.ERROR and "Flux.1-Schnell" in detail for status, detail in statuses)
    assert any(status is HealthStatus.WARN and "max_power 40" in detail for status, detail in statuses)


def _text_backend_process(state: str = "SERVING") -> ProcessSnapshot:
    """The supervised text backend's row in the given supervisor state."""
    return ProcessSnapshot(
        process_id=9100,
        process_type="TEXT_BACKEND",
        device_index=None,
        last_process_state=state,
        is_alive=True,
        is_busy=False,
        is_external=True,
        display_state="ready",
        loaded_horde_model_name="Llama-3.2-3B",
    )


def test_a_ready_text_backend_is_not_an_inference_lane_serving() -> None:
    """A serving external backend beside an idle slot leaves the worker reading as ready, not serving."""
    snapshot = _snapshot(processes=[_process("WAITING_FOR_JOB"), _text_backend_process()])

    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)

    assert report.phase is not WorkerPhase.SERVING


def test_a_text_backend_alone_does_not_end_the_warm_up() -> None:
    """A worker whose only row is its external backend is still warming up, as one with no rows is."""
    snapshot = _snapshot(processes=[_text_backend_process()])

    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)

    assert report.phase is WorkerPhase.WARMING_UP


def test_a_text_backend_is_not_a_card_with_low_duty() -> None:
    """The near-idle-card check reads inference lanes only; an external row names no card of its own."""
    snapshot = _snapshot(
        processes=[_text_backend_process()],
        gpu_utilization_mean_percent_per_card={0: 0.0},
    )

    assert gpu_duty_low_cards(snapshot) == []


_SCRIBE_CONFIG = WorkerConfigSummary(
    dreamer_name="Test",
    scribe_name="Test-Scribe",
    worker_version="12.0.0",
    scribe=True,
    text_stall_seconds=30.0,
    text_backend_ready_patience_seconds=120.0,
)
"""A scribe worker's config summary carrying the two patiences the text checks derive from."""


def _text_checks(snapshot: WorkerStateSnapshot) -> list[HealthCheck]:
    """Return the text rows of a derived report, which is where both text findings land."""
    report = derive(snapshot, SupervisorStatus.RUNNING, 1.0)
    return [check for check in report.checks if check.name in (TEXT_BACKEND_CHECK_NAME, TEXT_GENERATION_CHECK_NAME)]


def _unready_scribe(*, seconds: float) -> WorkerStateSnapshot:
    """A scribe worker whose readiness gate has been open for ``seconds``."""
    return _snapshot(
        config=_SCRIBE_CONFIG,
        timestamp=10_000.0,
        text_backend_ready=False,
        text_backend_not_ready_since=10_000.0 - seconds,
    )


def _generating_scribe(*, silent_for: float) -> WorkerStateSnapshot:
    """A scribe worker with one generation in hand that last produced text ``silent_for`` ago."""
    return _snapshot(
        config=_SCRIBE_CONFIG,
        timestamp=10_000.0,
        text_backend_ready=True,
        text_jobs_in_flight=1,
        snapshot_interval_seconds=1.0,
        work_ledger=[
            WorkLedgerEntry(
                job_id="running",
                stage=WorkLedgerStage.INFERENCE,
                workload=WorkloadKind.TEXT_GENERATION,
                progress_current=12,
                progress_unit=WorkLedgerProgressUnit.CHUNKS,
                progress_age_seconds=silent_for,
            ),
        ],
    )


def test_a_backend_within_its_readiness_patience_raises_nothing() -> None:
    """A cold start legitimately runs to minutes, so the wait itself is not a finding."""
    assert _text_checks(_unready_scribe(seconds=90.0)) == []


def test_a_refused_credential_is_an_error_at_once_with_the_password_as_the_remedy() -> None:
    """The backend is up and waiting changes nothing, so the finding does not wait out the patience window."""
    snapshot = _unready_scribe(seconds=5.0).model_copy(update={"text_backend_credentials_refused": True})

    checks = _text_checks(snapshot)

    assert [(check.name, check.status) for check in checks] == [(TEXT_BACKEND_CHECK_NAME, HealthStatus.ERROR)]
    assert "refused the worker's credentials" in checks[0].detail
    assert "text_backend_password" in checks[0].detail


def test_a_backend_past_its_readiness_patience_warns_and_says_why_no_jobs_are_popped() -> None:
    """Past the gate's own patience the wait stops being a cold start and is worth an operator's eye."""
    checks = _text_checks(_unready_scribe(seconds=150.0))

    assert [check.name for check in checks] == [TEXT_BACKEND_CHECK_NAME]
    assert checks[0].status is HealthStatus.WARN
    assert "no text jobs are being popped" in checks[0].detail


def test_a_backend_past_twice_its_readiness_patience_is_a_fault() -> None:
    """At that point the backend is not starting at all, which is a different situation from slow."""
    checks = _text_checks(_unready_scribe(seconds=260.0))

    assert checks[0].status is HealthStatus.ERROR


def test_a_generation_within_its_stall_bound_raises_nothing() -> None:
    """The bound is the worker's own patience; below it the generation is merely slow."""
    assert _text_checks(_generating_scribe(silent_for=20.0)) == []


def test_a_generation_that_has_produced_nothing_past_its_bound_warns() -> None:
    """Silence is the one signal a deadline cannot give, and the worker faults the job on it."""
    checks = _text_checks(_generating_scribe(silent_for=45.0))

    assert [check.name for check in checks] == [TEXT_GENERATION_CHECK_NAME]
    assert checks[0].status is HealthStatus.WARN
    assert "produced nothing" in checks[0].detail


def test_a_generation_at_the_stall_bound_is_given_the_snapshot_interval() -> None:
    """The worker abandons the job at its own bound, so the dashboard must not warn about it first."""
    assert _text_checks(_generating_scribe(silent_for=30.5)) == []


def test_a_worker_without_the_scribe_role_never_sees_a_text_check() -> None:
    """A row about a backend nobody configured is noise on nearly every dashboard."""
    dreamer = _snapshot(
        timestamp=10_000.0,
        text_backend_ready=False,
        text_backend_not_ready_since=1.0,
        text_jobs_in_flight=1,
        work_ledger=[
            WorkLedgerEntry(
                job_id="running",
                stage=WorkLedgerStage.INFERENCE,
                workload=WorkloadKind.TEXT_GENERATION,
                progress_age_seconds=900.0,
            ),
        ],
    )

    assert _text_checks(dreamer) == []


_MANAGED_SCRIBE_CONFIG = _SCRIBE_CONFIG.model_copy(update={"text_backend_managed": True})
"""A scribe worker that launches its own backend, which has a different clock and a different remedy."""


def _managed_scribe(*, launching_for: float | None, gate_open_for: float) -> WorkerStateSnapshot:
    """A worker launching its own backend, with a launch attempt begun ``launching_for`` ago.

    ``launching_for`` of None is a worker still obtaining the program, which has no launch to time, and
    ``gate_open_for`` is the flow's own clock, deliberately set much larger so a row proves which of the two
    the check reads. The idle inference lane is what a scribe-only worker's own children look like once they
    are up, so the image warming-up phase does not stand in front of the text one.
    """
    processes: list[ProcessSnapshot] = [_process("WAITING_FOR_JOB")]
    if launching_for is not None:
        processes.append(
            ProcessSnapshot(
                process_id=9100,
                process_type="TEXT_BACKEND",
                device_index=None,
                last_process_state="LAUNCHING",
                is_alive=True,
                is_busy=False,
                is_external=True,
                display_state="launching",
                text_backend=TextBackendDetail(
                    kind="koboldcpp",
                    port=5011,
                    launching_since=10_000.0 - launching_for,
                ),
            ),
        )
    return _snapshot(
        config=_MANAGED_SCRIBE_CONFIG,
        timestamp=10_000.0,
        processes=processes,
        text_backend_ready=False,
        text_backend_not_ready_since=10_000.0 - gate_open_for,
    )


def test_a_managed_backend_still_being_obtained_is_warming_up_and_raises_nothing() -> None:
    """Downloading the program on a first run is work in progress, not a backend that will not start."""
    snapshot = _managed_scribe(launching_for=None, gate_open_for=400.0)

    report = derive(snapshot, SupervisorStatus.RUNNING, 1.0)

    assert report.phase is WorkerPhase.WARMING_UP
    assert "obtaining the text backend" in report.headline
    assert _text_checks(snapshot) == []


def test_a_managed_backends_patience_is_measured_from_its_launch_not_the_flows_first_look() -> None:
    """The flow's clock starts with the worker, so a long provision would read as a late backend."""
    snapshot = _managed_scribe(launching_for=30.0, gate_open_for=400.0)

    report = derive(snapshot, SupervisorStatus.RUNNING, 1.0)

    assert report.phase is WorkerPhase.WARMING_UP
    assert "starting the text backend" in report.headline
    assert _text_checks(snapshot) == []


def test_a_managed_backend_past_its_patience_is_a_warn_headline_and_a_warn_row() -> None:
    """A worker serving nothing while its own backend will not answer is not a ready worker."""
    snapshot = _managed_scribe(launching_for=150.0, gate_open_for=400.0)

    report = derive(snapshot, SupervisorStatus.RUNNING, 1.0)

    assert report.phase is WorkerPhase.DEGRADED
    assert report.severity is HealthStatus.WARN
    assert "Text backend not ready" in report.headline
    assert _text_checks(snapshot)[0].status is HealthStatus.WARN


def test_a_managed_backend_past_twice_its_patience_names_the_worker_owned_remedy() -> None:
    """The worker started it, so there is nothing for the operator to start: its output is the evidence."""
    snapshot = _managed_scribe(launching_for=260.0, gate_open_for=400.0)

    report = derive(snapshot, SupervisorStatus.RUNNING, 1.0)

    assert report.severity is HealthStatus.ERROR
    assert "logs/text_backend.log" in report.detail
    row = _text_checks(snapshot)[0]
    assert row.status is HealthStatus.ERROR
    assert "logs/text_backend.log" in row.detail


def test_an_attached_backend_past_its_patience_points_at_kai_url() -> None:
    """A backend the operator runs is theirs to check, at the address they configured."""
    row = _text_checks(_unready_scribe(seconds=260.0))[0]

    assert "`kai_url` is its address" in row.detail


def test_a_failed_check_is_never_reported_under_an_ok_headline() -> None:
    """A headline at OK over a failed row tells an operator the opposite of what the row says."""
    snapshot = _snapshot(processes=[_process("WAITING_FOR_JOB")], lora_pops_blocked_by_disk=True)

    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)

    assert report.phase is WorkerPhase.READY
    assert report.severity is HealthStatus.ERROR


def test_a_program_that_could_not_be_obtained_is_an_error_not_a_warm_up() -> None:
    """Nothing is retried after that failure, so a headline still reading "obtaining" would never resolve."""
    snapshot = _managed_scribe(launching_for=None, gate_open_for=400.0)
    row = ProcessSnapshot(
        process_id=9100,
        process_type="TEXT_BACKEND",
        device_index=None,
        last_process_state="STOPPED",
        is_alive=False,
        is_busy=False,
        is_external=True,
        display_state="not obtained",
        text_backend=TextBackendDetail(kind="koboldcpp", provision_error="the release download failed"),
    )
    snapshot.processes = [_process("WAITING_FOR_JOB"), row]

    report = derive(snapshot, SupervisorStatus.RUNNING, 1.0)

    assert report.phase is WorkerPhase.DEGRADED
    assert report.severity is HealthStatus.ERROR
    assert report.headline == "Text backend could not be obtained"
    assert "the release download failed" in report.detail
    checks = _text_checks(snapshot)
    assert [check.name for check in checks] == [TEXT_BACKEND_CHECK_NAME]
    assert checks[0].status is HealthStatus.ERROR
