"""Contracts for attributable full-worker harness deadlines."""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_worker_regen.harness import (
    HarnessStageDeadlines,
    _capture_harness_stage_snapshot,
    _stage_deadline_violation,
    _watch_for_scenario_completion,
)
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.simulation._canned_scenarios import CannedJobSource
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


async def test_pending_job_deadline_names_the_job_and_stage() -> None:
    """A queue-stage bound reports the exact oldest job rather than only an overall timeout."""
    manager = make_testable_process_manager()
    job = await track_popped_job_async(manager._job_tracker, make_job_pop_response())
    assert job.id_ is not None
    tracked = manager._job_tracker.get_tracked_job(job.id_)
    assert tracked is not None
    tracked.current_stage_since = time.time() - 5.0

    violation = _stage_deadline_violation(
        manager,
        HarnessStageDeadlines(pending_inference_seconds=1.0),
    )

    assert violation is not None
    stage, subject, age = violation
    assert stage == "pending_inference"
    assert subject == str(job.id_)
    assert age >= 4.0


def test_starting_process_deadline_captures_process_generation_and_receipt_state() -> None:
    """A boot-stage bound preserves the process census before teardown clears it."""
    manager = make_testable_process_manager()
    process_info = make_mock_process_info(7, state=HordeProcessState.PROCESS_STARTING)
    process_info.last_process_state_started_at = time.time() - 5.0
    process_info.has_ever_reported = False
    manager._process_map[7] = process_info

    violation = _stage_deadline_violation(
        manager,
        HarnessStageDeadlines(process_starting_seconds=1.0),
    )
    assert violation is not None
    stage, subject, age = violation
    snapshot = _capture_harness_stage_snapshot(
        manager,
        reason="stage_deadline_exceeded",
        exceeded_stage=stage,
        exceeded_subject=subject,
        exceeded_age_seconds=age,
    )

    assert snapshot.exceeded_stage == "process_starting"
    assert snapshot.exceeded_subject == "7"
    assert snapshot.process_states[7] == "PROCESS_STARTING"
    assert snapshot.process_types[7] == process_info.process_type.name
    assert snapshot.process_models[7] == process_info.loaded_horde_model_name
    assert snapshot.process_launches[7] == process_info.process_launch_identifier
    assert snapshot.processes_reported[7] is False


def test_snapshot_captures_intake_gate_and_finite_source_progress() -> None:
    """A timeout with no tracked job still explains whether intake consumed the scripted queue."""
    manager = make_testable_process_manager()
    source = CannedJobSource([make_job_pop_response(), make_job_pop_response()])
    manager._job_popper.set_canned_job_source(source)
    source.next_pop_response()
    manager._state.last_pop_gate = "queue_full"
    manager._state.last_pop_gate_since = time.time() - 3.0
    manager._state.last_pop_attempt_completed_at = time.time() - 2.0

    snapshot = _capture_harness_stage_snapshot(
        manager,
        reason="overall_deadline_exceeded",
        exceeded_stage=None,
        exceeded_subject=None,
        exceeded_age_seconds=5.0,
    )

    assert snapshot.pop_gate == "queue_full"
    assert snapshot.pop_gate_age_seconds is not None and snapshot.pop_gate_age_seconds >= 3.0
    assert snapshot.last_pop_attempt_age_seconds is not None and snapshot.last_pop_attempt_age_seconds >= 2.0
    assert snapshot.source_exhausted is False
    assert snapshot.source_progress == "1/2"
    assert snapshot.aux_prefetch_summary is None
    assert snapshot.whole_card_summary.startswith("active=False")


async def test_drained_source_with_missing_terminal_count_fails_at_accounting_deadline() -> None:
    """A fully finalized queue cannot hide a missing completion transition behind the overall timeout."""
    manager = make_testable_process_manager()
    source = CannedJobSource([make_job_pop_response()])
    manager._job_popper.set_canned_job_source(source)
    source.next_pop_response()
    manager._abort = Mock()  # type: ignore[method-assign]
    snapshots = []

    timed_out = await _watch_for_scenario_completion(
        manager,
        num_jobs_expected=1,
        timeout_seconds=2.0,
        stage_deadlines=HarnessStageDeadlines(terminal_accounting_seconds=0.01),
        timeout_snapshots=snapshots,
    )

    assert timed_out is True
    manager._abort.assert_called_once()
    assert len(snapshots) == 1
    assert snapshots[0].exceeded_stage == "terminal_accounting"
    assert snapshots[0].exceeded_subject == "0/1"
    assert snapshots[0].source_progress == "1/1"
