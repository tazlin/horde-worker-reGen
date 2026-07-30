"""Tests for when the recovery escalation is allowed to spend its last rung, a fresh process.

Exiting is only a recovery if something starts the worker again. A supervising frontend does that on the
unexpected exit; an operator running unattended arranges the same with a service manager and signals it by
setting ``exit_on_unhandled_faults``. With neither, exiting would end the worker's usefulness instead of
restoring it, so the rung is withheld and escalation continues with the remedies it can apply in place. When
even those are spent the worker holds itself quiescent instead of churning, which the doomed-pool regressions
cover.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.config.worker_state import RecoveryParkReason
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator import WorkerRecoveryCoordinator
from tests.process_management.conftest import make_testable_process_manager


class TestRecoveryAbortRequiresSomethingToRelaunch:
    """The gate on the escalation's terminal rung."""

    def test_attached_supervisor_permits_the_exit(self) -> None:
        """With a supervisor watching, the exit is a restart, so the abort proceeds."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = Mock()
        process_manager._shutdown_manager = Mock()

        process_manager._abort_for_recovery()

        process_manager._shutdown_manager.abort.assert_called_once()

    def test_opt_in_permits_the_exit_without_a_supervisor(self) -> None:
        """An operator who set exit_on_unhandled_faults has arranged an external restart, so exiting is theirs."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=True)
        process_manager._supervisor = None
        process_manager._shutdown_manager = Mock()

        process_manager._abort_for_recovery()

        process_manager._shutdown_manager.abort.assert_called_once()

    def test_unattended_worker_stays_up_instead_of_exiting(self) -> None:
        """With nothing to relaunch it, the worker keeps escalating in place rather than exiting."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = None
        process_manager._shutdown_manager = Mock()

        process_manager._abort_for_recovery()

        process_manager._shutdown_manager.abort.assert_not_called()

    def test_a_closed_supervisor_channel_counts_as_unattended(self) -> None:
        """A supervisor that has gone away cannot relaunch, so its absence withholds the rung.

        The channel is set to None when it closes, which is the same fact as never having had one.
        """
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = None
        process_manager._shutdown_manager = Mock()

        process_manager._abort_for_recovery()

        process_manager._shutdown_manager.abort.assert_not_called()

    @pytest.mark.parametrize(
        ("supervised", "exit_on_unhandled_faults"),
        [(True, False), (True, True), (False, True)],
        ids=["supervisor", "both-restart-contracts", "service-manager"],
    )
    def test_accepted_recovery_abort_exits_with_failure_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
        supervised: bool,
        exit_on_unhandled_faults: bool,
    ) -> None:
        """A terminal recovery exit is observable as failure by every supported relauncher.

        The recovery abort first performs the normal child cleanup and lets the async worker loop unwind. Once
        that loop has returned, the process-manager entry point must still propagate a nonzero status; otherwise
        ``Restart=on-failure`` and equivalent service policies interpret the terminal rung as a clean stop.
        """
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=exit_on_unhandled_faults)
        process_manager._supervisor = Mock() if supervised else None
        process_manager._shutdown_manager = Mock()

        def _run_worker_loop(coroutine: object) -> None:
            # ``start`` creates the real coroutine before handing it to asyncio. Close it in this synchronous
            # harness, then emulate the terminal callback occurring during that loop.
            assert hasattr(coroutine, "close")
            coroutine.close()  # type: ignore[union-attr]
            process_manager._abort_for_recovery()

        monkeypatch.setattr(asyncio, "run", _run_worker_loop)
        monkeypatch.setattr("atexit.register", Mock())
        monkeypatch.setattr("signal.signal", Mock())

        with pytest.raises(SystemExit) as exit_info:
            process_manager.start()

        assert exit_info.value.code != 0
        process_manager._shutdown_manager.abort.assert_called_once()

    def test_accepted_recovery_abort_marks_session_unclean(self) -> None:
        """A session that needed the terminal recovery rung cannot become a known-good configuration."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = Mock()
        process_manager._shutdown_manager = Mock()

        process_manager._abort_for_recovery()

        assert process_manager.build_run_record().clean_exit is False


class TestLifecycleAbortPolicy:
    """Lifecycle hard-timeout aborts are cleanup guarantees, not restart-policy decisions."""

    @pytest.mark.parametrize(
        ("supervised", "exit_on_unhandled_faults"),
        [(False, False), (True, False), (False, True), (True, True)],
        ids=["headless-default", "supervised", "service-managed", "both"],
    )
    def test_shutdown_timeout_callback_always_aborts(
        self,
        supervised: bool,
        exit_on_unhandled_faults: bool,
    ) -> None:
        """Once graceful shutdown is already in progress, its hard timeout cannot be refused.

        ``ProcessLifecycleManager`` invokes this callback after accepted work failed to drain. Whether another
        process will relaunch the worker is irrelevant at that point: refusing the abort leaves the existing
        process permanently inside shutdown and prevents the timed-shutdown backstop from being armed.
        """
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=exit_on_unhandled_faults)
        process_manager._supervisor = Mock() if supervised else None
        process_manager._shutdown_manager = Mock()
        process_manager._state.initiate_shutdown()

        process_manager._process_lifecycle._abort_callback()

        process_manager._shutdown_manager.abort.assert_called_once()


class TestRefusedAbortIsReportedHonestly:
    """A withheld terminal rung must never be reported to the caller as an exit that took."""

    def test_runaway_recovery_check_reports_not_aborted_when_the_exit_is_withheld(self) -> None:
        """The flapping verdict stands, but the worker is still running, and the caller is told so.

        ``run_recovery_supervisor`` treats a True reading as "the worker is on its way out". A withheld exit
        reported as True would hide a worker that is still up and still holding accepted work.
        """
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = None
        process_manager._shutdown_manager = Mock()
        coordinator = process_manager._recovery_coordinator

        process_manager._process_lifecycle._num_process_recoveries = coordinator.RUNAWAY_RECOVERY_CEILING + 1

        assert coordinator.maybe_abort_on_runaway_recoveries() is False
        assert process_manager._shutdown_manager.abort.called is False

    def test_the_refusal_is_recorded_once_not_every_tick(self) -> None:
        """The breached ceiling persists, so the verdict is edge-logged rather than repeated per tick."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = None
        process_manager._shutdown_manager = Mock()
        coordinator = process_manager._recovery_coordinator

        process_manager._process_lifecycle._num_process_recoveries = coordinator.RUNAWAY_RECOVERY_CEILING + 1
        coordinator.maybe_abort_on_runaway_recoveries()
        abandoned_after_first = _abandoned_count(coordinator)

        for _ in range(5):
            coordinator.maybe_abort_on_runaway_recoveries()

        assert _abandoned_count(coordinator) == abandoned_after_first

    def test_an_attached_supervisor_still_stops_escalation_on_abort(self) -> None:
        """The control: when the abort takes, the caller must stop driving escalation."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = Mock()
        process_manager._shutdown_manager = Mock()
        coordinator = process_manager._recovery_coordinator

        # A real abort marks the worker shutting down, which is the fact the caller keys on.
        def _abort_marks_shutting_down() -> None:
            coordinator._state.shutting_down = True

        process_manager._shutdown_manager.abort.side_effect = _abort_marks_shutting_down
        process_manager._process_lifecycle._num_process_recoveries = coordinator.RUNAWAY_RECOVERY_CEILING + 1

        assert coordinator.maybe_abort_on_runaway_recoveries() is True
        process_manager._shutdown_manager.abort.assert_called_once()


class TestRecoveryParkReprobeState:
    """A park re-probe begins a new recovery attempt, independent of the exhausted episode."""

    @pytest.mark.parametrize("reason", list(RecoveryParkReason))
    def test_lifting_park_clears_every_escalation_episode_field(self, reason: RecoveryParkReason) -> None:
        """No wedge verdict, progress baseline, or remedy budget survives across the park boundary."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        coordinator = process_manager._recovery_coordinator
        coordinator._state.recovery_parked = True
        coordinator._state.recovery_park_reason = reason
        coordinator._state.recovery_park_since = coordinator._clock() - 30.0

        coordinator.limp_by_active = True
        coordinator.episode_saw_unrecoverable_pool = True
        coordinator.episode_progress_baseline = 11
        coordinator.episode_inference_start_baseline = 12
        coordinator.episode_post_processing_progress_baseline = 13
        coordinator.pp_reclaim_remedy_issued_at = 14.0
        coordinator.head_recovery_in_flight_since = 15.0
        coordinator.healthy_hold_since = 16.0
        coordinator.governance_reset_at = 17.0
        coordinator.reclaim_rungs = ()
        coordinator.reclaim_cursor = 2
        coordinator.reclaim_rungs_issued_in_allotment = 2
        coordinator.reclaim_rung_issued_at = 18.0
        coordinator.reclaim_remedy_started_at = 19.0
        coordinator.give_up_yields_spent = 2
        coordinator.restore_reclaimed_lanes = Mock()  # type: ignore[method-assign]

        coordinator.leave_recovery_park()

        assert coordinator._state.recovery_parked is False
        assert coordinator.limp_by_active is False
        assert coordinator.episode_saw_unrecoverable_pool is False
        assert coordinator.episode_progress_baseline is None
        assert coordinator.episode_inference_start_baseline is None
        assert coordinator.episode_post_processing_progress_baseline is None
        assert coordinator.pp_reclaim_remedy_issued_at is None
        assert coordinator.head_recovery_in_flight_since is None
        assert coordinator.healthy_hold_since is None
        assert coordinator.governance_reset_at is None
        assert coordinator.reclaim_rungs is None
        assert coordinator.reclaim_cursor == 0
        assert coordinator.reclaim_rungs_issued_in_allotment == 0
        assert coordinator.reclaim_rung_issued_at is None
        assert coordinator.reclaim_remedy_started_at is None
        assert coordinator.give_up_yields_spent == 0
        coordinator.restore_reclaimed_lanes.assert_called_once()


def _abandoned_count(coordinator: WorkerRecoveryCoordinator) -> int:
    """Count the RECOVERY_ABANDONED events currently in the coordinator's action ledger."""
    return sum(
        1
        for event in coordinator._action_ledger.recent(limit=1000)
        if event.event_type == LedgerEventType.RECOVERY_ABANDONED
    )
