"""Tests for the bound on how often the supervisor may excuse its own tick gaps.

The supervisor observes the worker only when it ticks, so a large gap between two consecutive ticks is
time it could not observe and must not charge to the worker; it re-graces the wedge baseline instead. That
forgiveness is correct exactly once per genuine outage and corrosive when it repeats: a host that starves
the supervisor faster than ``WEDGE_LIVENESS_TIMEOUT_SECONDS`` moves the baseline forward before staleness
can ever accrue, which silently turns the wedge detector off while leaving every other signal looking
healthy. A wedged worker then stays wedged for as long as the starvation lasts.

The contract under test is budget-then-detect. A rolling window bounds both the number of re-graces and the
already-forgiven time they represent; the first gap in a quiet window is always forgiven (an overnight host
sleep is one event), and once the window's budget is spent further gaps are charged to the worker so
staleness accrues and detection proceeds. Charging a gap is not by itself a restart: a worker that resumed
along with its supervisor advances its stamp on the next drain and is untouched. Only a worker that is also
silent is restarted, which is the accepted cost of keeping the detector armed.
"""

from __future__ import annotations

import pytest
from loguru import logger

from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui import worker_launcher
from horde_worker_regen.tui.worker_launcher import WorkerProcessMode, WorkerSupervisor
from tests.tui.test_supervisor_wedge_recovery import _Clock, _FakeCtx, _feed_liveness

_WEDGE_TIMEOUT = 180.0
"""The production wedge window; the fake clock makes using it as-is free."""

_STALL_RESET = 30.0
"""The production tick-gap threshold above which a gap is a supervisor stall rather than jitter."""

_FORGIVENESS_WINDOW = 3600.0
_FORGIVENESS_BUDGET = 300.0
_MAX_FORGIVEN_RESETS = 3

_FIELD_TICK_GAPS = (41.0, 62.0, 87.0, 53.0, 44.0, 71.0)
"""Successive supervisor tick gaps of the shape observed while a worker stayed wedged for hours.

Every one of them exceeds the stall threshold while staying under the wedge window, so unbounded
forgiveness re-graced the baseline before it could ever accrue.
"""


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    forgiveness_window: float = _FORGIVENESS_WINDOW,
    forgiveness_budget: float = _FORGIVENESS_BUDGET,
    max_forgiven_resets: int = _MAX_FORGIVEN_RESETS,
) -> tuple[WorkerSupervisor, _FakeCtx, _Clock, list[int]]:
    """Build a started supervisor on a hand-cranked clock, with the forgiveness bounds under test.

    Mirrors the wedge-recovery harness: fake process tree, fake pipe, and a tree-kill spy that both records
    the killed pid and flips the process dead so the following tick sees the exit a real kill would cause.
    """
    clock = _Clock()
    monkeypatch.setattr(worker_launcher.time, "time", clock.now)
    monkeypatch.setattr(worker_launcher, "WEDGE_LIVENESS_TIMEOUT_SECONDS", _WEDGE_TIMEOUT)
    monkeypatch.setattr(worker_launcher, "_SUPERVISOR_STALL_RESET_SECONDS", _STALL_RESET)
    monkeypatch.setattr(worker_launcher, "_SUPERVISOR_STALL_FORGIVENESS_WINDOW_SECONDS", forgiveness_window)
    monkeypatch.setattr(worker_launcher, "_SUPERVISOR_STALL_FORGIVENESS_BUDGET_SECONDS", forgiveness_budget)
    monkeypatch.setattr(worker_launcher, "_SUPERVISOR_STALL_MAX_FORGIVEN_RESETS", max_forgiven_resets)

    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
        restart_backoff_seconds=0.0,
    )

    killed_pids: list[int] = []

    def _spy_tree_kill(pid: int, **_kwargs: object) -> list[int]:
        killed_pids.append(pid)
        if ctx.last_process is not None and ctx.last_process.pid == pid:
            ctx.last_process.die(exitcode=-9)
        return [pid]

    monkeypatch.setattr(worker_launcher, "kill_process_tree", _spy_tree_kill)

    supervisor.start()
    return supervisor, ctx, clock, killed_pids


def _establish_baseline(supervisor: WorkerSupervisor, ctx: _FakeCtx, clock: _Clock) -> float:
    """Deliver the worker's first liveness frame so the wedge detector has something to measure from."""
    stamp = clock.now()
    _feed_liveness(supervisor, ctx, stamp)
    supervisor.tick()
    return stamp


def _tick_after_gap(
    supervisor: WorkerSupervisor,
    ctx: _FakeCtx,
    clock: _Clock,
    *,
    gap: float,
    worker_stamp: float | None,
) -> None:
    """Advance the clock by ``gap`` with no intervening ticks, then tick once.

    ``worker_stamp`` is the loop stamp the worker reports on that tick; None means it sent nothing at all.
    """
    clock.advance(gap)
    if worker_stamp is not None:
        _feed_liveness(supervisor, ctx, worker_stamp)
    supervisor.tick()


def test_a_single_long_gap_is_forgiven_and_leaves_detection_armed(monkeypatch: pytest.MonkeyPatch) -> None:
    """One outage (a laptop asleep overnight) is excused, and the detector still works afterwards.

    This is the case forgiveness exists for, and the bound must not break it: the window is empty, so the
    gap is forgiven however long it was. What the bound must also preserve is the state it leaves behind:
    the worker is re-graced, not exempted, so a wedge that begins after the resume is still caught.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)
    _establish_baseline(supervisor, ctx, clock)

    # The host sleeps for eight hours; both processes resume together and the worker reports a fresh stamp.
    _tick_after_gap(supervisor, ctx, clock, gap=8 * 60 * 60, worker_stamp=clock.now() + 8 * 60 * 60)

    assert killed_pids == [], "a single supervisor outage was charged to the worker"
    assert supervisor.stall_stats.forgiven_resets == 1
    assert supervisor.stall_stats.refused_resets == 0

    # The worker now genuinely wedges, with the supervisor ticking normally throughout.
    frozen = supervisor.last_liveness_wall_time
    for _ in range(int(_WEDGE_TIMEOUT / 5) + 2):
        _tick_after_gap(supervisor, ctx, clock, gap=5.0, worker_stamp=frozen)

    assert killed_pids, "forgiveness left the wedge detector disarmed for the rest of the session"


def test_repeated_gaps_spend_the_budget_and_detection_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the window's re-graces are spent, staleness accrues across gaps and the detector fires.

    The defect this pins: every gap re-graced the baseline, so the wedge window restarted from zero on each
    one and a worker that never advanced again was never detected. After the budget is spent the gaps stop
    resetting the baseline, so the same silence adds up to the detection threshold.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)
    _establish_baseline(supervisor, ctx, clock)

    # The host starves the supervisor repeatedly while the worker stays healthy and advancing.
    for _ in range(_MAX_FORGIVEN_RESETS):
        _tick_after_gap(supervisor, ctx, clock, gap=60.0, worker_stamp=clock.now() + 60.0)

    assert killed_pids == [], "a healthy, advancing worker was killed while the budget was still available"
    assert supervisor.stall_stats.forgiven_resets == _MAX_FORGIVEN_RESETS
    assert supervisor.stall_stats.budget_spent is True

    # The worker now wedges and the starvation continues. Each gap alone stays under the wedge window, so
    # only accrual across the unforgiven gaps can detect it.
    frozen = supervisor.last_liveness_wall_time
    for _ in range(4):
        if killed_pids:
            break
        _tick_after_gap(supervisor, ctx, clock, gap=60.0, worker_stamp=frozen)

    assert killed_pids, "gaps kept being forgiven past the budget, so the wedge was never detected"
    assert supervisor.stall_stats.refused_resets >= 1


def test_a_worker_still_advancing_survives_an_unforgiven_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spending the budget does not restart workers; it only stops excusing unobserved time.

    The accepted cost of budget-then-detect is a possible restart of a *silent* worker after repeated host
    stalls. A worker that resumed with its supervisor advances its stamp on the very next drain, which
    beats the staleness check in the same tick, so it must be untouched however spent the budget is.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)
    _establish_baseline(supervisor, ctx, clock)

    for _ in range(_MAX_FORGIVEN_RESETS + 6):
        _tick_after_gap(supervisor, ctx, clock, gap=90.0, worker_stamp=clock.now() + 90.0)

    assert supervisor.stall_stats.budget_spent is True
    assert supervisor.stall_stats.refused_resets >= 1, "the budget was never actually enforced"
    assert killed_pids == [], "an advancing worker was killed merely because the supervisor was starved"


def test_the_budget_replenishes_as_the_window_slides(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host that stalls in one hour and behaves in the next is forgiven again.

    The bound is a rolling window, not a session-lifetime allowance: keeping a supervisor permanently
    unable to re-grace because of an hour it has long left behind would reintroduce the false restarts
    forgiveness exists to prevent.
    """
    window = 200.0
    supervisor, ctx, clock, killed_pids = _install(monkeypatch, forgiveness_window=window)
    _establish_baseline(supervisor, ctx, clock)

    for _ in range(_MAX_FORGIVEN_RESETS):
        _tick_after_gap(supervisor, ctx, clock, gap=40.0, worker_stamp=clock.now() + 40.0)
    assert supervisor.stall_stats.budget_spent is True

    # A quiet stretch longer than the window, ticking normally with a healthy worker.
    for _ in range(int(window / 20) + 2):
        _tick_after_gap(supervisor, ctx, clock, gap=20.0, worker_stamp=clock.now() + 20.0)

    assert supervisor.stall_stats.resets_in_window == 0
    assert supervisor.stall_stats.budget_spent is False

    _tick_after_gap(supervisor, ctx, clock, gap=40.0, worker_stamp=clock.now() + 40.0)

    assert supervisor.stall_stats.forgiven_resets == _MAX_FORGIVEN_RESETS + 1, "the window never replenished"
    assert killed_pids == []


def test_the_field_tick_gap_sequence_exhausts_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The observed sequence of gaps must stop being excused and end in a recovered worker.

    Each gap is above the stall threshold and below the wedge window, which is precisely the shape that
    survived indefinitely under unbounded forgiveness. Replaying it against a wedged worker has to end with
    the tree killed rather than with an unbounded run of re-graces.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)
    frozen = _establish_baseline(supervisor, ctx, clock)

    for gap in _FIELD_TICK_GAPS:
        _tick_after_gap(supervisor, ctx, clock, gap=gap, worker_stamp=frozen)

    stats = supervisor.stall_stats
    assert stats.forgiven_resets == _MAX_FORGIVEN_RESETS, "the field sequence was forgiven past its budget"
    assert stats.refused_resets == len(_FIELD_TICK_GAPS) - _MAX_FORGIVEN_RESETS
    assert stats.largest_tick_gap_seconds == max(_FIELD_TICK_GAPS)

    # Keep replaying the same pattern; the accrued staleness must reach the detection threshold.
    for gap in _FIELD_TICK_GAPS:
        if killed_pids:
            break
        _tick_after_gap(supervisor, ctx, clock, gap=gap, worker_stamp=frozen)

    assert killed_pids, "the field tick-gap shape kept the wedge detector disarmed"


def test_forgiveness_is_reported_once_per_event_with_running_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each re-grace is visible at INFO, and the spent budget is edge-logged rather than repeated per gap.

    Forgiveness reported only at DEBUG is what let a long run of re-gracing pass unnoticed, so the event
    has to appear at a level operators actually keep, carrying the counter that makes a repeated pattern
    legible as one. The refusal that follows is a persistent condition, so it is announced when it begins
    instead of once per gap.
    """
    supervisor, ctx, clock, _killed = _install(monkeypatch)
    _establish_baseline(supervisor, ctx, clock)

    lines: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda message: lines.append((message.record["level"].name, message.record["message"])),
        level="INFO",
    )
    try:
        for _ in range(_MAX_FORGIVEN_RESETS + 3):
            _tick_after_gap(supervisor, ctx, clock, gap=60.0, worker_stamp=clock.now() + 60.0)
    finally:
        logger.remove(sink_id)

    forgiven = [line for line in lines if "resetting the worker wedge baseline" in line[1]]
    refused = [line for line in lines if "stall-forgiveness budget is spent" in line[1]]

    assert len(forgiven) == _MAX_FORGIVEN_RESETS
    assert {level for level, _ in forgiven} == {"INFO"}
    assert f"{_MAX_FORGIVEN_RESETS}/{_MAX_FORGIVEN_RESETS}" in forgiven[-1][1], "no running counter in the notice"
    assert len(refused) == 1, "the spent budget was reported per gap rather than on its edge"
