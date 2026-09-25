"""The TUI's horde-maintenance action addresses one logical worker at a time.

Each enabled role is its own worker on the horde with its own maintenance flag. With one role on, the key
toggles it from the snapshot's per-worker reading, overlaid by a request not yet confirmed; with more than
one, the key opens a modal whose rows offer one press per worker plus an all-workers pair.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    HordeWorkerDetailsSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.scheduling.workload_kind import WorkloadKind
from horde_worker_regen.tui.app import HordeWorkerTUI, ServerMaintenanceChoice, ServerMaintenanceModal

IMAGE = WorkloadKind.IMAGE_GENERATION
TEXT = WorkloadKind.TEXT_GENERATION


class _Supervisor:
    """Records each maintenance request the action sends."""

    def __init__(self, snapshot: WorkerStateSnapshot | None) -> None:
        self.latest_snapshot = snapshot
        self.requests: list[tuple[bool, tuple[WorkloadKind, ...] | None]] = []
        self.accept = True

    def request_set_server_maintenance(self, enabled: bool, workloads: Sequence[WorkloadKind] | None = None) -> bool:
        """Record the request and answer whether the worker took it."""
        self.requests.append((enabled, None if workloads is None else tuple(workloads)))
        return self.accept


class _Harness:
    """The app's maintenance methods over the state they read, without a running Textual app."""

    action_toggle_server_maintenance = HordeWorkerTUI.action_toggle_server_maintenance
    _on_server_maintenance_choice = HordeWorkerTUI._on_server_maintenance_choice
    _whole_worker_in_maintenance = HordeWorkerTUI._whole_worker_in_maintenance
    _server_maintenance_rows = HordeWorkerTUI._server_maintenance_rows
    _request_server_maintenance = HordeWorkerTUI._request_server_maintenance
    _worker_names_for = HordeWorkerTUI._worker_names_for

    def __init__(self, snapshot: WorkerStateSnapshot | None) -> None:
        self._supervisor = _Supervisor(snapshot)
        self._intended_server_maintenance: dict[WorkloadKind, bool] = {}
        self._server_maintenance_intent_at: dict[WorkloadKind, float] = {}
        self.notices: list[str] = []
        self.pushed: list[Any] = []

    def notify(self, message: str, **_kwargs: Any) -> None:  # noqa: ANN401 - Textual's signature
        self.notices.append(message)

    def push_screen(self, screen: Any, callback: Any = None) -> None:  # noqa: ANN401 - Textual's signature
        self.pushed.append((screen, callback))


def _snapshot(*, scribe: bool = False, held: dict[WorkloadKind, bool] | None = None) -> WorkerStateSnapshot:
    config = WorkerConfigSummary(
        dreamer_name="A Dreamer",
        worker_version="0.0.0",
        scribe=scribe,
        scribe_name="A Scribe" if scribe else None,
    )
    held = held or {}
    by_workload = {
        workload: HordeWorkerDetailsSnapshot(worker_name=name, registered=True, maintenance=held.get(workload, False))
        for workload, name in config.enabled_roles
    }
    return WorkerStateSnapshot(
        config=config,
        timestamp=time.time(),
        worker_details_maintenance=any(held.values()),
        worker_details_by_workload=by_workload,
    )


def test_one_role_toggles_that_worker_on_when_not_held() -> None:
    """A single role needs no modal: the key puts its worker into maintenance."""
    app = _Harness(_snapshot())

    app.action_toggle_server_maintenance()

    assert app._supervisor.requests == [(True, (IMAGE,))]
    assert app.pushed == []
    assert app._intended_server_maintenance == {IMAGE: True}
    assert "A Dreamer" in app.notices[-1]


def test_one_role_toggles_that_worker_off_when_held() -> None:
    """The horde holds the one worker, so the key takes it out."""
    app = _Harness(_snapshot(held={IMAGE: True}))

    app.action_toggle_server_maintenance()

    assert app._supervisor.requests == [(False, (IMAGE,))]
    assert app._intended_server_maintenance == {IMAGE: False}


def test_a_rapid_second_press_reverses_the_first() -> None:
    """The pending intent outranks the stale poll, so the second press sends the opposite state."""
    app = _Harness(_snapshot())

    app.action_toggle_server_maintenance()
    app.action_toggle_server_maintenance()

    assert [enabled for enabled, _ in app._supervisor.requests] == [True, False]


def test_a_refused_request_records_no_intent() -> None:
    """A worker that is not running takes nothing, so nothing is shown as requested."""
    app = _Harness(_snapshot())
    app._supervisor.accept = False

    app.action_toggle_server_maintenance()

    assert app._intended_server_maintenance == {}
    assert "not sent" in app.notices[-1]


def test_no_snapshot_sends_a_whole_worker_toggle() -> None:
    """Before any snapshot the roles are unknown, so the request addresses every enabled role."""
    app = _Harness(None)

    app.action_toggle_server_maintenance()

    assert app._supervisor.requests == [(True, None)]


def test_two_roles_open_the_modal_with_one_row_per_worker() -> None:
    """A dreamer plus scribe host has two workers on the horde, so the key asks which one."""
    app = _Harness(_snapshot(scribe=True, held={TEXT: True}))

    app.action_toggle_server_maintenance()

    assert app._supervisor.requests == []
    assert len(app.pushed) == 1
    modal, callback = app.pushed[0]
    assert isinstance(modal, ServerMaintenanceModal)
    assert [(row.workload, row.worker_name, row.in_maintenance) for row in modal._rows] == [
        (IMAGE, "A Dreamer", False),
        (TEXT, "A Scribe", True),
    ]
    assert callback == app._on_server_maintenance_choice


def test_the_modal_choice_addresses_only_the_chosen_worker() -> None:
    """Taking the scribe out leaves the dreamer's flag untouched and records intent for the scribe alone."""
    app = _Harness(_snapshot(scribe=True, held={TEXT: True}))
    choice: ServerMaintenanceChoice = ((TEXT,), False)

    app._on_server_maintenance_choice(choice)

    assert app._supervisor.requests == [(False, (TEXT,))]
    assert app._intended_server_maintenance == {TEXT: False}
    assert "A Scribe" in app.notices[-1]
    assert "A Dreamer" not in app.notices[-1]


def test_the_all_choice_addresses_every_worker() -> None:
    """The all-workers row sets one state on each enabled role's worker."""
    app = _Harness(_snapshot(scribe=True))

    app._on_server_maintenance_choice(((IMAGE, TEXT), True))

    assert app._supervisor.requests == [(True, (IMAGE, TEXT))]
    assert app._intended_server_maintenance == {IMAGE: True, TEXT: True}


def test_a_cancelled_modal_sends_nothing() -> None:
    """Escape or Cancel leaves every flag as it was."""
    app = _Harness(_snapshot(scribe=True))

    app._on_server_maintenance_choice(None)

    assert app._supervisor.requests == []
    assert app._intended_server_maintenance == {}


def test_the_rows_overlay_a_pending_request_on_the_polled_state() -> None:
    """A row shows the state the operator asked for until the horde confirms it, and says so."""
    app = _Harness(_snapshot(scribe=True))
    app._intended_server_maintenance[TEXT] = True

    rows = app._server_maintenance_rows()

    assert [(row.workload, row.in_maintenance, row.pending) for row in rows] == [
        (IMAGE, False, False),
        (TEXT, True, True),
    ]
