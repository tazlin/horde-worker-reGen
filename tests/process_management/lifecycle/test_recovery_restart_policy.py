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

from horde_worker_regen.process_management import main_entry_point
from horde_worker_regen.process_management.config.worker_state import RecoveryParkReason
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.lifecycle import worker_recovery_coordinator
from horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator import (
    RecoveryDisposition,
    WorkerRecoveryCoordinator,
)
from horde_worker_regen.process_management.resources.reclaim_ladder import ReclaimRung, ReclaimRungKind
from tests.process_management.conftest import make_testable_process_manager


class TestRecoveryAbortRequiresSomethingToRelaunch:
    """The gate on the escalation's terminal rung."""

    def test_attached_supervisor_permits_the_exit(self) -> None:
        """With a supervisor watching, the exit is a restart, so the abort proceeds."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = Mock()
        process_manager._shutdown_manager = Mock()

        disposition = process_manager._request_terminal_recovery()

        assert disposition is RecoveryDisposition.RESTART_PROCESS
        process_manager._shutdown_manager.abort.assert_called_once()

    def test_opt_in_permits_the_exit_without_a_supervisor(self) -> None:
        """An operator who set exit_on_unhandled_faults has arranged an external restart, so exiting is theirs."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=True)
        process_manager._supervisor = None
        process_manager._shutdown_manager = Mock()

        disposition = process_manager._request_terminal_recovery()

        assert disposition is RecoveryDisposition.RESTART_PROCESS
        process_manager._shutdown_manager.abort.assert_called_once()

    def test_unattended_worker_stays_up_instead_of_exiting(self) -> None:
        """With nothing to relaunch it, the worker keeps escalating in place rather than exiting."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = None
        process_manager._shutdown_manager = Mock()

        disposition = process_manager._request_terminal_recovery()

        assert disposition is RecoveryDisposition.CONTINUE_IN_PROCESS
        process_manager._shutdown_manager.abort.assert_not_called()

    def test_a_closed_supervisor_channel_counts_as_unattended(self) -> None:
        """A supervisor that has gone away cannot relaunch, so its absence withholds the rung.

        The channel is set to None when it closes, which is the same fact as never having had one.
        """
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = None
        process_manager._shutdown_manager = Mock()

        disposition = process_manager._request_terminal_recovery()

        assert disposition is RecoveryDisposition.CONTINUE_IN_PROCESS
        process_manager._shutdown_manager.abort.assert_not_called()

    @pytest.mark.parametrize(
        ("supervised", "exit_on_unhandled_faults"),
        [(True, False), (True, True), (False, True)],
        ids=["supervisor", "both-restart-contracts", "service-manager"],
    )
    def test_process_manager_reports_accepted_recovery_restart(
        self,
        monkeypatch: pytest.MonkeyPatch,
        supervised: bool,
        exit_on_unhandled_faults: bool,
    ) -> None:
        """The process manager reports terminal recovery through its typed session outcome.

        Cleanup still runs through the ordinary async loop. The manager returns the retained disposition instead
        of raising inside the lifecycle layer, leaving process-status policy to the outer entry point.
        """
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=exit_on_unhandled_faults)
        process_manager._supervisor = Mock() if supervised else None
        process_manager._shutdown_manager = Mock()

        def _run_worker_loop(coroutine: object) -> None:
            # ``start`` creates the real coroutine before handing it to asyncio. Close it in this synchronous
            # harness, then emulate the terminal callback occurring during that loop.
            assert hasattr(coroutine, "close")
            coroutine.close()  # type: ignore[union-attr]
            process_manager._request_terminal_recovery()

        monkeypatch.setattr(asyncio, "run", _run_worker_loop)
        monkeypatch.setattr("atexit.register", Mock())
        monkeypatch.setattr("signal.signal", Mock())

        disposition = process_manager.start()

        assert disposition is RecoveryDisposition.RESTART_PROCESS
        process_manager._shutdown_manager.abort.assert_called_once()

    def test_accepted_recovery_abort_marks_session_unclean(self) -> None:
        """A session that needed the terminal recovery rung cannot become a known-good configuration."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = Mock()
        process_manager._shutdown_manager = Mock()

        process_manager._request_terminal_recovery()

        assert process_manager.build_run_record().clean_exit is False

    def test_corrupt_message_channel_uses_the_typed_terminal_restart(self) -> None:
        """The dispatcher's real corruption callback must retain a restart disposition before aborting.

        This is the production seam the terminal-recovery tests previously skipped: wiring corruption
        directly to ``_abort`` cleaned up children but returned a successful process status, and it bypassed
        the disposition the supervisor relies on to identify a recovery restart.
        """
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = Mock()
        process_manager._shutdown_manager = Mock()

        handler = process_manager._message_dispatcher._on_channel_corrupt
        assert handler is not None
        handler("test channel corruption")

        assert process_manager._recovery_disposition is RecoveryDisposition.RESTART_PROCESS
        process_manager._shutdown_manager.abort.assert_called_once()

    def test_irreparable_channel_exits_nonzero_without_a_known_relauncher(self) -> None:
        """A process unable to receive any child result cannot truthfully remain running headless."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = None
        process_manager._shutdown_manager = Mock()

        process_manager._restart_after_message_channel_corruption("test channel corruption")

        assert process_manager._recovery_disposition is RecoveryDisposition.RESTART_PROCESS
        process_manager._shutdown_manager.abort.assert_called_once()


class TestRecoveryDispositionEntryPoint:
    """The outer entry point owns conversion from a session outcome to process status."""

    @pytest.mark.parametrize(
        ("disposition", "expects_failure_exit"),
        [
            (RecoveryDisposition.CONTINUE_IN_PROCESS, False),
            (RecoveryDisposition.RESTART_PROCESS, True),
        ],
        ids=["ordinary-stop", "recovery-restart"],
    )
    def test_session_is_persisted_before_process_status_is_applied(
        self,
        monkeypatch: pytest.MonkeyPatch,
        disposition: RecoveryDisposition,
        expects_failure_exit: bool,
    ) -> None:
        """Both outcomes persist once; only a requested restart becomes a failed process exit."""
        events: list[str] = []
        process_manager = Mock()
        process_manager.start.side_effect = lambda: events.append("start") or disposition
        manager_factory = Mock(return_value=process_manager)
        persist = Mock(side_effect=lambda *_args: events.append("persist"))

        monkeypatch.setattr(main_entry_point, "verify_worker_identity", Mock())
        monkeypatch.setattr(main_entry_point, "coerce_bridge_data_to_capabilities", Mock())
        monkeypatch.setattr(main_entry_point, "HordeWorkerProcessManager", manager_factory)
        monkeypatch.setattr(main_entry_point, "_persist_session_state", persist)

        if expects_failure_exit:
            with pytest.raises(SystemExit) as exit_info:
                main_entry_point.start_working(Mock(), Mock(), Mock())
            assert exit_info.value.code != 0
        else:
            main_entry_point.start_working(Mock(), Mock(), Mock())

        assert events == ["start", "persist"]
        persist.assert_called_once_with(process_manager, manager_factory.call_args.kwargs["bridge_data"])


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

    def test_an_attached_supervisor_stops_escalation_from_the_typed_disposition(self) -> None:
        """The coordinator trusts the returned contract instead of observing a shutdown side effect."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        process_manager._supervisor = Mock()
        process_manager._shutdown_manager = Mock()
        coordinator = process_manager._recovery_coordinator

        process_manager._process_lifecycle._num_process_recoveries = coordinator.RUNAWAY_RECOVERY_CEILING + 1

        assert coordinator.maybe_abort_on_runaway_recoveries() is True
        assert coordinator._state.shutting_down is False
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


class TestConstructiveRemedySelection:
    """A remedy is only constructive if it can change the resource condition the wedge is about."""

    def test_a_rung_promising_nothing_is_skipped_for_one_that_frees_memory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A zero-promise rung must not consume the settling window; the next rung that can free room runs."""
        process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
        coordinator = process_manager._recovery_coordinator

        empty_lane = ReclaimRung(
            kind=ReclaimRungKind.PAUSE_PP_LANE,
            tenant_label="post-processing lane",
            promised_freed_mb=0.0,
            device_index=0,
        )
        real_relief = ReclaimRung(
            kind=ReclaimRungKind.UNLOAD_IDLE_MODEL,
            tenant_label="model#4",
            promised_freed_mb=4096.0,
            device_index=0,
            target_process_id=4,
        )
        coordinator.reclaim_rungs = (empty_lane, real_relief)

        executed: list[ReclaimRung] = []

        def _execute(rung: ReclaimRung, _scheduler: object) -> bool:
            executed.append(rung)
            return True

        monkeypatch.setattr(worker_recovery_coordinator, "execute_reclaim_rung", _execute)

        issued = coordinator.issue_next_constructive_remedy()

        assert issued is real_relief
        assert executed == [real_relief], "the zero-promise rung must never be issued"
        assert coordinator.reclaim_paused_lanes == [], "a skipped lane pause leaves no restore obligation"


def _abandoned_count(coordinator: WorkerRecoveryCoordinator) -> int:
    """Count the RECOVERY_ABANDONED events currently in the coordinator's action ledger."""
    return sum(
        1
        for event in coordinator._action_ledger.recent(limit=1000)
        if event.event_type == LedgerEventType.RECOVERY_ABANDONED
    )
