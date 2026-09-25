"""Server-side (horde) maintenance toggle: API call, supervisor command, and the F2-resume gating.

The worker can put itself into, or out of, *server-side* maintenance on the horde (so the horde stops or
resumes sending it jobs), distinct from the local pop-pause. This is exposed as a dedicated supervisor
command and a TUI key. An operator resume (F2) additionally clears server maintenance only when the
``remove_maintenance_on_init`` config opts into it.

Also covers the snapshot maintenance fields: ``supervisor_paused`` and ``last_pop_maintenance_mode`` are
exposed as separate fields on ``WorkerStateSnapshot`` so the TUI can toggle and display each source of a
paused state independently.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import process_manager as process_manager_module
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    HordeWorkerDetailsSnapshot,
    SupervisorCommand,
    SupervisorControlMessage,
)
from horde_worker_regen.process_management.scheduling.workload_kind import WorkloadKind
from tests.process_management.conftest import make_testable_process_manager


class _InlineThread:
    """A ``threading.Thread`` stand-in that runs its target inline on ``start()``.

    The supervisor handler dispatches the (blocking) horde maintenance call off the control loop on a
    daemon thread; tests patch this in so that dispatch is synchronous and deterministic, with no real
    background threads to race the assertions.
    """

    def __init__(self, *, target: Callable[..., Any], args: tuple[Any, ...] = (), **_kwargs: Any) -> None:  # noqa: ANN401
        """Capture the target and positional args, ignoring thread-only kwargs (name/daemon)."""
        self._target = target
        self._args = args

    def start(self) -> None:
        """Run the captured target synchronously."""
        self._target(*self._args)


def _run_off_loop_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the supervisor handler's off-loop thread dispatch run inline for deterministic assertions."""
    monkeypatch.setattr(process_manager_module.threading, "Thread", _InlineThread)


class TestSetMaintenanceApiCall:
    """``set_maintenance`` issues a ModifyWorkerRequest with the requested flag."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_builds_modify_request_with_flag(self, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """``set_maintenance(enabled)`` looks up the worker and modifies it with ``maintenance=enabled``."""
        captured: dict[str, object] = {}

        class FakeClient:
            def workers_all_details(self, worker_name: str | None = None, *, api_key: str | None = None) -> list[Mock]:
                details = Mock()
                details.id_ = "worker-1"
                details.name = worker_name
                return [details]

            def worker_modify(self, request: object) -> None:
                captured["maintenance"] = request.maintenance  # type: ignore[attr-defined]

        monkeypatch.setattr(process_manager_module, "AIHordeAPISimpleClient", lambda: FakeClient())
        manager = make_testable_process_manager()

        manager.set_maintenance(enabled)

        assert captured["maintenance"] is enabled

    @pytest.mark.parametrize("enabled", [True, False])
    def test_unregistered_worker_is_a_quiet_no_op(self, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """A not-yet-registered name (empty list result) returns without modifying or raising.

        The horde registers a worker implicitly on its first pop, so a brand-new name is unknown until
        then; that must be treated as the normal first-run case, not an error.
        """
        modified = False

        class FakeClient:
            def workers_all_details(self, worker_name: str | None = None, *, api_key: str | None = None) -> list[Mock]:
                return []

            def worker_modify(self, request: object) -> None:
                nonlocal modified
                modified = True

        monkeypatch.setattr(process_manager_module, "AIHordeAPISimpleClient", lambda: FakeClient())
        manager = make_testable_process_manager()

        manager.set_maintenance(enabled)

        assert modified is False

    def test_the_flag_goes_to_every_enabled_roles_worker_and_no_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each enabled role is its own worker on the horde, and a role that is off has none to change."""
        looked_up: list[str | None] = []

        class FakeClient:
            def workers_all_details(self, worker_name: str | None = None, *, api_key: str | None = None) -> list[Mock]:
                looked_up.append(worker_name)
                details = Mock()
                details.id_ = f"id-of-{worker_name}"
                details.name = worker_name
                return [details]

            def worker_modify(self, request: object) -> None:
                return None

        monkeypatch.setattr(process_manager_module, "AIHordeAPISimpleClient", lambda: FakeClient())
        manager = make_testable_process_manager(
            dreamer=False,
            alchemist=True,
            alchemist_name="An Alchemist",
            scribe=True,
            scribe_name="A Scribe",
        )

        manager.set_maintenance(True)

        assert looked_up == ["An Alchemist", "A Scribe"]

    def test_remove_maintenance_clears_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``remove_maintenance`` is the ``set_maintenance(False)`` convenience wrapper."""
        manager = make_testable_process_manager()
        recorder = Mock()
        monkeypatch.setattr(manager, "set_maintenance", recorder)
        manager.remove_maintenance()
        recorder.assert_called_once_with(False)


class TestSupervisorMaintenanceCommand:
    """The SET_SERVER_MAINTENANCE supervisor command dispatches the API call off the control loop."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_command_dispatches_set_maintenance(self, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """The command runs ``_set_server_maintenance_safe(enabled)`` on a background thread."""
        manager = make_testable_process_manager()
        _run_off_loop_inline(monkeypatch)
        recorded: list[bool] = []
        monkeypatch.setattr(
            manager, "_set_server_maintenance_safe", lambda value, workloads=None: recorded.append(value)
        )

        manager._apply_supervisor_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_SERVER_MAINTENANCE,
                server_maintenance_enabled=enabled,
            ),
        )

        assert recorded == [enabled]


class TestResumeClearsMaintenanceOnlyWhenConfigured:
    """F2-resume clears *server* maintenance only when ``remove_maintenance_on_init`` is set (per config)."""

    def test_resume_clears_when_flag_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the flag on, a resume also clears horde-side maintenance (off-loop)."""
        manager = make_testable_process_manager(remove_maintenance_on_init=True)
        manager._state.supervisor_paused = True
        _run_off_loop_inline(monkeypatch)
        recorded: list[bool] = []
        monkeypatch.setattr(
            manager, "_set_server_maintenance_safe", lambda value, workloads=None: recorded.append(value)
        )

        manager._apply_supervisor_command(SupervisorControlMessage(command=SupervisorCommand.RESUME))

        assert manager._state.supervisor_paused is False
        assert recorded == [False]

    def test_resume_does_not_clear_when_flag_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the flag off (the default), a resume only lifts the local pause; server state is untouched."""
        manager = make_testable_process_manager(remove_maintenance_on_init=False)
        manager._state.supervisor_paused = True
        recorded: list[bool] = []
        monkeypatch.setattr(
            manager, "_set_server_maintenance_safe", lambda value, workloads=None: recorded.append(value)
        )

        manager._apply_supervisor_command(SupervisorControlMessage(command=SupervisorCommand.RESUME))

        assert manager._state.supervisor_paused is False
        assert recorded == []


class TestSnapshotMaintenanceFields:
    """``supervisor_paused`` and ``last_pop_maintenance_mode`` surface as their own snapshot fields.

    The TUI needs to distinguish which source is causing the aggregate ``maintenance_mode=True`` so that
    F2 can toggle the right lever and the status bar can name the source. Both fields must be present in
    the snapshot independently of the aggregate.
    """

    def test_supervisor_paused_surfaces_as_own_field_and_in_aggregate(self) -> None:
        """A local F2 pause sets supervisor_paused and maintenance_mode but not last_pop_maintenance_mode."""
        manager = make_testable_process_manager()
        manager._state.supervisor_paused = True
        manager._state.last_pop_maintenance_mode = False

        snapshot = manager._build_worker_state_snapshot()

        assert snapshot.supervisor_paused is True
        assert snapshot.last_pop_maintenance_mode is False
        assert snapshot.maintenance_mode is True

    def test_last_pop_maintenance_mode_surfaces_as_own_field_and_in_aggregate(self) -> None:
        """A pop-response maintenance error sets maintenance fields without supervisor pause."""
        manager = make_testable_process_manager()
        manager._state.last_pop_maintenance_mode = True
        manager._state.supervisor_paused = False

        snapshot = manager._build_worker_state_snapshot()

        assert snapshot.last_pop_maintenance_mode is True
        assert snapshot.supervisor_paused is False
        assert snapshot.maintenance_mode is True

    def test_neither_paused_flag_leaves_maintenance_mode_false(self) -> None:
        """When neither local flag is set, maintenance_mode is False and both granular fields are False."""
        manager = make_testable_process_manager()
        manager._state.supervisor_paused = False
        manager._state.last_pop_maintenance_mode = False
        manager._state.self_throttle_paused = False

        snapshot = manager._build_worker_state_snapshot()

        assert snapshot.supervisor_paused is False
        assert snapshot.last_pop_maintenance_mode is False
        assert snapshot.maintenance_mode is False


class TestWorkerDetailsMaintenanceRefresh:
    """Worker-details maintenance is advisory and can be stale behind successful pops."""

    async def test_successful_job_pop_suppresses_stale_worker_details_maintenance(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A new popped job is stronger evidence than a stale worker-details maintenance=True poll."""
        manager = make_testable_process_manager()
        manager._state.server_maintenance_cleared_by_job_pop = True
        monkeypatch.setattr(
            manager,
            "_fetch_worker_details",
            lambda _worker_name: Mock(maintenance_mode=True, paused=False),
        )

        await manager.api_get_worker_details()

        assert manager._worker_details_maintenance is False
        assert manager._state.server_maintenance_cleared_by_job_pop is True

    async def test_worker_details_maintenance_false_reconciles_successful_pop_signal(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Once the advisory poll catches up to maintenance=False, the stale-poll suppression resets."""
        manager = make_testable_process_manager()
        manager._state.server_maintenance_cleared_by_job_pop = True
        monkeypatch.setattr(
            manager,
            "_fetch_worker_details",
            lambda _worker_name: Mock(maintenance_mode=False, paused=False),
        )

        await manager.api_get_worker_details()

        assert manager._worker_details_maintenance is False
        assert manager._state.server_maintenance_cleared_by_job_pop is False


class TestPerWorkerMaintenance:
    """Each enabled role is its own worker on the horde, and the flag is set and read per worker."""

    @staticmethod
    def _client(looked_up: list[str | None], modified: list[tuple[str, bool]]) -> object:
        class FakeClient:
            def workers_all_details(self, worker_name: str | None = None, *, api_key: str | None = None) -> list[Mock]:
                looked_up.append(worker_name)
                details = Mock()
                details.id_ = f"id-of-{worker_name}"
                details.name = worker_name
                return [details]

            def worker_modify(self, request: object) -> None:
                modified.append((request.worker_id, request.maintenance))  # type: ignore[attr-defined]

        return FakeClient()

    def test_naming_workloads_changes_only_those_workers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dreamer plus scribe host holding only the scribe leaves the dreamer's flag alone."""
        looked_up: list[str | None] = []
        modified: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            process_manager_module, "AIHordeAPISimpleClient", lambda: self._client(looked_up, modified)
        )
        manager = make_testable_process_manager(scribe=True, scribe_name="A Scribe")

        manager.set_maintenance(True, [WorkloadKind.TEXT_GENERATION])

        assert looked_up == ["A Scribe"]
        assert modified == [("id-of-A Scribe", True)]

    def test_a_workload_whose_role_is_off_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A role that is off has no worker on the horde, so naming it changes nothing."""
        looked_up: list[str | None] = []
        monkeypatch.setattr(process_manager_module, "AIHordeAPISimpleClient", lambda: self._client(looked_up, []))
        manager = make_testable_process_manager()

        manager.set_maintenance(True, [WorkloadKind.TEXT_GENERATION])

        assert looked_up == []

    def test_the_command_carries_the_workloads_to_the_setter_and_the_intent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The workloads a surface names reach the horde call and the per-worker intent the auto-clear reads."""
        manager = make_testable_process_manager(scribe=True, scribe_name="A Scribe")
        _run_off_loop_inline(monkeypatch)
        recorded: list[tuple[bool, object]] = []
        monkeypatch.setattr(
            manager, "_set_server_maintenance_safe", lambda value, workloads=None: recorded.append((value, workloads))
        )

        manager._apply_supervisor_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_SERVER_MAINTENANCE,
                server_maintenance_enabled=True,
                server_maintenance_workloads=[WorkloadKind.TEXT_GENERATION],
            ),
        )

        assert recorded == [(True, [WorkloadKind.TEXT_GENERATION])]
        assert manager._state.server_maintenance_locally_intended_workloads == {WorkloadKind.TEXT_GENERATION}
        assert manager._state.server_maintenance_locally_intended is False, "the image worker was not held"

    def test_a_whole_worker_command_holds_every_enabled_role(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No workloads named means every enabled role, and a later OFF for one releases only that one."""
        manager = make_testable_process_manager(scribe=True, scribe_name="A Scribe")
        _run_off_loop_inline(monkeypatch)
        monkeypatch.setattr(manager, "_set_server_maintenance_safe", lambda value, workloads=None: None)

        manager._apply_supervisor_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_SERVER_MAINTENANCE, server_maintenance_enabled=True
            ),
        )
        assert manager._state.server_maintenance_locally_intended_workloads == {
            WorkloadKind.IMAGE_GENERATION,
            WorkloadKind.TEXT_GENERATION,
        }

        manager._apply_supervisor_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_SERVER_MAINTENANCE,
                server_maintenance_enabled=False,
                server_maintenance_workloads=[WorkloadKind.IMAGE_GENERATION],
            ),
        )
        assert manager._state.server_maintenance_locally_intended_workloads == {WorkloadKind.TEXT_GENERATION}

    async def test_the_poll_records_each_worker_and_the_aggregates_are_any_of_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A held scribe beside a serving dreamer reads as held on the whole and as itself per worker."""
        manager = make_testable_process_manager(scribe=True, scribe_name="A Scribe")

        def fetch(worker_name: str) -> Mock | None:
            if worker_name == "A Scribe":
                return Mock(maintenance_mode=True, paused=False)
            return Mock(maintenance_mode=False, paused=False)

        monkeypatch.setattr(manager, "_fetch_worker_details", fetch)

        await manager.api_get_worker_details()
        snapshot = manager._build_worker_state_snapshot()

        assert snapshot.worker_details_maintenance is True
        assert snapshot.worker_details_by_workload == {
            WorkloadKind.IMAGE_GENERATION: HordeWorkerDetailsSnapshot(
                worker_name=manager.bridge_data.dreamer_worker_name, registered=True
            ),
            WorkloadKind.TEXT_GENERATION: HordeWorkerDetailsSnapshot(
                worker_name="A Scribe", registered=True, maintenance=True
            ),
        }
