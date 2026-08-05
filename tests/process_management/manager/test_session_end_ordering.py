"""``session_end`` terminates the stats stream: it is written after the drain, never before it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, TrackedJob
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.simulation._dummy_jobs import dummy_job_factory
from tests.process_management.conftest import make_testable_process_manager


def _exported_events(tmp_path: Path) -> list[dict[str, object]]:
    """Every export event the session wrote, oldest-first."""
    events: list[dict[str, object]] = []
    for path in sorted((tmp_path / ".horde_worker_regen" / "stats").glob("stats-v*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    return events


def _finalize_one_job(process_manager: HordeWorkerProcessManager, *, faulted: bool = False) -> None:
    """Drive one job through the manager's finalize observer, as the submitter does."""
    job = dummy_job_factory("Deliberate")
    assert job.id_ is not None
    tracked = TrackedJob(
        job_id=job.id_,
        sdk_api_job_info=job,
        stage=JobStage.PENDING_SUBMIT,
        time_popped=100.0,
        stage_timestamps={"PENDING_INFERENCE": 100.0, "FINALIZED": 112.0},
    )
    job_info = HordeJobInfo(
        sdk_api_job_info=job,
        state=GENERATION_STATE.faulted if faulted else GENERATION_STATE.ok,
        time_popped=100.0,
    )
    process_manager._on_job_finalized(tracked, job_info)


def _close_session(process_manager: HordeWorkerProcessManager) -> None:
    """Write the marker the way the main loop does once every job-finalizing loop has returned."""
    process_manager._record_session_end_once(process_manager._session_end_reason or "graceful_shutdown")


@pytest.fixture
def exporting_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HordeWorkerProcessManager:
    """A process manager writing its stats stream under an isolated working directory.

    The export is switched on after construction (as the runtime command does) rather than through
    bridge data, whose mock fields cannot serialize into a session_start config snapshot.
    """
    monkeypatch.chdir(tmp_path)
    process_manager = make_testable_process_manager()
    process_manager._run_metrics.set_stats_export(True, worker_version="1.0.0")
    return process_manager


def test_initiating_shutdown_does_not_terminate_the_stream(
    exporting_manager: HordeWorkerProcessManager,
    tmp_path: Path,
) -> None:
    """Shutdown only begins the drain, so it must not write the marker that closes the stream."""
    exporting_manager._shutdown()

    assert [event for event in _exported_events(tmp_path) if event["event"] == "session_end"] == []


def test_jobs_finalized_during_the_drain_precede_session_end(
    exporting_manager: HordeWorkerProcessManager,
    tmp_path: Path,
) -> None:
    """A job the drain submits after shutdown begins still lands above the marker, and is counted."""
    _finalize_one_job(exporting_manager)
    exporting_manager._shutdown()
    _finalize_one_job(exporting_manager)
    _finalize_one_job(exporting_manager, faulted=True)

    _close_session(exporting_manager)

    events = _exported_events(tmp_path)
    completed = [event for event in events if event["event"] == "job_completed"]
    assert len(completed) == 3
    assert events[-1]["event"] == "session_end"
    assert events[-1]["reason"] == "graceful_shutdown"
    assert events[-1]["jobs_submitted"] == 2
    assert events[-1]["jobs_faulted"] == 1


def test_session_end_is_written_at_most_once(
    exporting_manager: HordeWorkerProcessManager,
    tmp_path: Path,
) -> None:
    """Repeated teardown paths cannot append a second marker."""
    _close_session(exporting_manager)
    _close_session(exporting_manager)

    assert len([event for event in _exported_events(tmp_path) if event["event"] == "session_end"]) == 1


def test_an_abort_names_itself_as_the_reason(
    exporting_manager: HordeWorkerProcessManager,
    tmp_path: Path,
) -> None:
    """The reason recorded is the one that began the teardown, not the path that completed it."""
    exporting_manager._abort()
    # An abort arms the force-kill backstop, whose thread would otherwise force-exit the test process
    # once its grace expires; the harness neutralizes it the same way once a lifecycle has returned.
    exporting_manager._cancel_timed_shutdown()
    exporting_manager._shutdown()

    _close_session(exporting_manager)

    end_events = [event for event in _exported_events(tmp_path) if event["event"] == "session_end"]
    assert [event["reason"] for event in end_events] == ["aborted"]
