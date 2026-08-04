"""Tests for making the supervisor's own liveness visible to an operator.

The wedge backstop measures the worker, and every tick gap the supervisor forgives is a stretch in which
that measurement was suspended. A supervisor being starved by its host therefore presents as an absence:
no detections, no alarm, nothing on any surface that describes the worker. These tests pin the counters
onto the three places an operator or an attached session actually reads: the dashboard status bar, the
host's status frame, and the attach client that reflects it.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.tui import socket_protocol as sp
from horde_worker_regen.tui.app import HordeWorkerTUI
from horde_worker_regen.tui.attach import AttachedWorkerSupervisor
from horde_worker_regen.tui.worker_host import WorkerHost
from horde_worker_regen.tui.worker_launcher import SupervisorStallStats, SupervisorStatus, WorkerProcessMode
from tests.tui.test_supervisor_stall_forgiveness import (
    _MAX_FORGIVEN_RESETS,
    _establish_baseline,
    _install,
    _tick_after_gap,
)


def _stats(*, resets_in_window: int, budget_spent: bool, largest_gap: float = 0.0) -> SupervisorStallStats:
    """Stall counters with only the fields the status bar reads set away from their quiet values."""
    return SupervisorStallStats(
        forgiven_resets=resets_in_window,
        forgiven_seconds=0.0,
        refused_resets=0,
        resets_in_window=resets_in_window,
        forgiven_seconds_in_window=0.0,
        largest_tick_gap_seconds=largest_gap,
        budget_spent=budget_spent,
    )


def test_a_supervisor_with_nothing_to_report_adds_no_status_segment() -> None:
    """The healthy case is the absence of the segment; the bar is a scarce, priority-ordered resource."""
    assert HordeWorkerTUI._supervisor_stall_markup(SupervisorStallStats.quiet()) is None


def test_forgiven_gaps_show_the_count_against_the_allowance() -> None:
    """While forgiving, the segment warns and says how much of the window's allowance is gone.

    A bare "the supervisor stalled" would not distinguish the resume everyone has from the starvation that
    is about to leave a wedged worker undetected; the fraction is what separates them at a glance.
    """
    markup = HordeWorkerTUI._supervisor_stall_markup(
        _stats(resets_in_window=2, budget_spent=False, largest_gap=64.0),
    )

    assert markup is not None
    assert f"2/{_MAX_FORGIVEN_RESETS}" in markup
    assert "gap 64s" in markup
    assert "yellow" in markup


def test_a_spent_budget_reads_as_an_error_state() -> None:
    """Once the budget is spent the supervisor is no longer protecting the worker from its own outages."""
    markup = HordeWorkerTUI._supervisor_stall_markup(_stats(resets_in_window=3, budget_spent=True))

    assert markup is not None
    assert "red" in markup


def test_the_host_status_frame_carries_the_supervisor_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client sees only what the host puts on the wire, so the host's own stalls have to be on it."""
    supervisor, ctx, clock, _killed = _install(monkeypatch)
    _establish_baseline(supervisor, ctx, clock)
    for _ in range(2):
        _tick_after_gap(supervisor, ctx, clock, gap=70.0, worker_stamp=clock.now() + 70.0)

    frame = WorkerHost(supervisor, host="127.0.0.1", port=0)._status_message()

    assert frame["stall_resets_in_window"] == 2
    assert frame["stall_forgiven_resets"] == 2
    assert frame["stall_forgiven_seconds"] == 140.0
    assert frame["stall_refused_resets"] == 0
    assert frame["stall_forgiven_seconds_in_window"] == 140.0
    assert frame["stall_max_forgiven_resets"] == _MAX_FORGIVEN_RESETS
    assert frame["stall_budget_spent"] is False
    assert frame["largest_tick_gap_seconds"] == 70.0


def test_the_attach_client_reflects_the_host_counters() -> None:
    """A browser-mode session presents the host supervisor's stalls, not its own (it has no worker)."""
    client = AttachedWorkerSupervisor(("127.0.0.1", 1), mode=WorkerProcessMode.FAKE)
    try:
        assert client.stall_stats.resets_in_window == 0

        client._apply(
            sp.status_message(
                status=SupervisorStatus.RUNNING.value,
                restart_attempts=0,
                mode=WorkerProcessMode.FAKE.value,
                worker_running=True,
                stall_forgiven_resets=7,
                stall_forgiven_seconds=411.0,
                stall_refused_resets=2,
                stall_resets_in_window=3,
                stall_forgiven_seconds_in_window=201.0,
                stall_max_forgiven_resets=3,
                stall_budget_spent=True,
                largest_tick_gap_seconds=91.0,
            ),
        )

        assert client.stall_stats.resets_in_window == 3
        assert client.stall_stats.forgiven_resets == 7
        assert client.stall_stats.forgiven_seconds == 411.0
        assert client.stall_stats.refused_resets == 2
        assert client.stall_stats.forgiven_seconds_in_window == 201.0
        assert client.stall_stats.budget_spent is True
        assert client.stall_stats.largest_tick_gap_seconds == 91.0
        assert HordeWorkerTUI._supervisor_stall_markup(client.stall_stats) is not None
    finally:
        client.close()


def test_a_status_frame_without_the_counters_reads_as_quiet() -> None:
    """A host that predates these fields must not manufacture an alarm out of their absence."""
    client = AttachedWorkerSupervisor(("127.0.0.1", 1), mode=WorkerProcessMode.FAKE)
    try:
        client._apply(
            {
                "type": sp.MSG_STATUS,
                "status": SupervisorStatus.RUNNING.value,
                "restart_attempts": 0,
                "mode": WorkerProcessMode.FAKE.value,
                "worker_running": True,
            },
        )

        assert client.stall_stats.resets_in_window == 0
        assert client.stall_stats.budget_spent is False
        assert HordeWorkerTUI._supervisor_stall_markup(client.stall_stats) is None
    finally:
        client.close()
