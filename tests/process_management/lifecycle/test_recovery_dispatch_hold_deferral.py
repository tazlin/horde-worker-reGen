"""The wedge assessment defers to a standing dispatch hold inside the hold's own liveness bound.

An idle card with pending work is the raw shape a structural wedge is read from, and a residency
reconciliation hold presents exactly that shape while the gate reclaims or waits on its probe. The recovery
coordinator answered it with pool resets and, past its budget, faulted the head the gate was about to admit.
The hold is now one of the deliberate holds the assessment excuses, bounded so a hold that never resolves
still trips the supervisor, and the deferral is logged once on each edge. The end line says what ended the
deferral, since a hold with no remedy left ends it at once rather than at the bound.

The dispatcher's queue-deadlock latch keeps its age underneath an excuse or an inference-slot boot, so the
wedge bound is measured again from the moment either ends: otherwise the wedge reads as sustained the tick
the hold lifts and a soft reset rebuilds a pool that was about to dispatch.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen.analysis.log_signatures import SIGNATURES
from horde_worker_regen.process_management.ipc.message_dispatcher import (
    _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS,
    DeadlockSnapshot,
)
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator import WorkerRecoveryCoordinator
from horde_worker_regen.process_management.scheduling.ledgers.dispatch_holds import (
    DISPATCH_HOLD_LIVENESS_SECONDS,
    HoldLiveness,
    HoldRemedy,
)
from tests.process_management.conftest import FakeClock, make_test_recovery_coordinator


def test_a_live_dispatch_hold_is_not_a_structural_wedge() -> None:
    """With the hold inside its bound the assessment reads no wedge; with no hold the raw shape stands."""
    clock = FakeClock()
    held = make_test_recovery_coordinator(
        job_tracker=JobTracker(clock=clock),
        clock=clock,
        structural_wedge=True,
        dispatch_hold_liveness_active=True,
    )
    assert held.structural_queue_wedge_active() is False
    assert held._deferring_to_dispatch_hold is True

    bare = make_test_recovery_coordinator(
        job_tracker=JobTracker(clock=clock),
        clock=clock,
        structural_wedge=True,
    )
    assert bare.structural_queue_wedge_active() is True
    assert bare._deferring_to_dispatch_hold is False


def test_the_deferral_lapses_with_the_hold() -> None:
    """Once the scheduler reports the hold past its bound, the assessment reads the queue on its own again."""
    clock = FakeClock()
    coordinator = make_test_recovery_coordinator(
        job_tracker=JobTracker(clock=clock),
        clock=clock,
        structural_wedge=True,
        dispatch_hold_liveness_active=True,
    )
    assert coordinator.structural_queue_wedge_active() is False

    coordinator._inference_scheduler.dispatch_hold_liveness_active.return_value = False
    assert coordinator.structural_queue_wedge_active() is True
    assert coordinator._deferring_to_dispatch_hold is False


def _latch_queue_deadlock(coordinator: WorkerRecoveryCoordinator, clock: FakeClock) -> None:
    """Latch a queue deadlock at the clock's present and read its age on the same driven clock."""
    snapshot = DeadlockSnapshot(
        in_deadlock=False,
        in_queue_deadlock=True,
        deadlock_started_at=0.0,
        queue_deadlock_started_at=clock(),
        queue_deadlock_model=None,
        queue_deadlock_process_id=None,
    )
    clocked = Mock()
    clocked.indicates_structural_wedge.side_effect = lambda measured_from=None: snapshot.indicates_structural_wedge(
        now=clock(), measured_from=measured_from
    )
    coordinator._message_dispatcher.get_deadlock_snapshot.return_value = clocked


def test_time_under_an_excuse_does_not_count_toward_the_wedge() -> None:
    """A latch older than the bound reads no wedge until the bound passes again after the excuse lifts."""
    clock = FakeClock(1_000.0)
    coordinator = make_test_recovery_coordinator(
        job_tracker=JobTracker(clock=clock),
        clock=clock,
        whole_card_governor_defer_active=True,
    )
    _latch_queue_deadlock(coordinator, clock)

    clock.advance(_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS + 40.0)
    assert coordinator.structural_queue_wedge_active() is False

    coordinator._inference_scheduler.whole_card_governor_defer_active.return_value = False
    clock.advance(1.0)
    assert coordinator.structural_queue_wedge_active() is False, "the latch's age under the excuse was counted"

    clock.advance(_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS - 2.0)
    assert coordinator.structural_queue_wedge_active() is False

    clock.advance(1.0)
    assert coordinator.structural_queue_wedge_active() is True


def test_an_inference_slot_boot_restarts_the_wedge_bound_when_it_ends() -> None:
    """A boot ending under a latch older than the bound leaves the wedge unsustained for one more bound."""
    clock = FakeClock(1_000.0)
    coordinator = make_test_recovery_coordinator(
        job_tracker=JobTracker(clock=clock),
        clock=clock,
        lane_state=HordeProcessState.PROCESS_STARTING,
    )
    _latch_queue_deadlock(coordinator, clock)

    clock.advance(_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS + 10.0)
    # A live boot is held by the supervisor, not excused here, so a boot that hangs still reaches its cap.
    assert coordinator.structural_queue_wedge_active() is True

    coordinator._process_map[1].last_process_state = HordeProcessState.WAITING_FOR_JOB
    clock.advance(0.5)
    assert coordinator.structural_queue_wedge_active() is False, "the latch's age through the boot was counted"

    clock.advance(_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS)
    assert coordinator.structural_queue_wedge_active() is True


def _deferring_coordinator(clock: FakeClock, liveness: HoldLiveness) -> WorkerRecoveryCoordinator:
    """A coordinator over a queue deadlock that has started deferring to a hold with ``liveness``."""
    coordinator = make_test_recovery_coordinator(
        job_tracker=JobTracker(clock=clock),
        clock=clock,
        structural_wedge=True,
        dispatch_hold_liveness_active=True,
    )
    scheduler = coordinator._inference_scheduler
    assert isinstance(scheduler, Mock)
    scheduler.dispatch_hold_liveness_seconds.return_value = liveness
    assert coordinator.structural_queue_wedge_active() is False
    return coordinator


def _capture_warnings() -> tuple[list[str], int]:
    """Collect the messages of INFO and louder log lines until the returned handler is removed."""
    captured: list[str] = []
    handler_id = logger.add(lambda message: captured.append(message.record["message"]), level="INFO")
    return captured, handler_id


def test_the_deferral_line_names_the_live_remedy() -> None:
    """The start line says what the gate is doing about the hold, read from the hold's recorded remedy."""
    clock = FakeClock()
    captured, handler_id = _capture_warnings()
    try:
        _deferring_coordinator(clock, HoldLiveness(12.0, DISPATCH_HOLD_LIVENESS_SECONDS, HoldRemedy.RECLAIM))
    finally:
        logger.remove(handler_id)
    started = [line for line in captured if line.startswith("Recovery is deferring")]
    assert len(started) == 1
    assert HoldRemedy.RECLAIM.phrase in started[0]


@pytest.mark.parametrize(
    ("liveness", "queue_deadlocked", "expected"),
    [
        pytest.param(
            HoldLiveness(34.0, DISPATCH_HOLD_LIVENESS_SECONDS, HoldRemedy.NONE),
            True,
            "the hold has no remedy left after 34s",
            id="no_remedy",
        ),
        pytest.param(
            HoldLiveness(DISPATCH_HOLD_LIVENESS_SECONDS, DISPATCH_HOLD_LIVENESS_SECONDS, HoldRemedy.PROBE),
            True,
            f"the hold outlived its {DISPATCH_HOLD_LIVENESS_SECONDS:.0f}s liveness bound",
            id="bound_lapsed",
        ),
        pytest.param(None, True, "the head is no longer held", id="released"),
        pytest.param(
            HoldLiveness(40.0, DISPATCH_HOLD_LIVENESS_SECONDS, HoldRemedy.PROBE),
            False,
            "the queue no longer reads as deadlocked",
            id="queue_moving",
        ),
    ],
)
def test_the_deferral_end_line_says_what_ended_it(
    liveness: HoldLiveness | None,
    queue_deadlocked: bool,
    expected: str,
) -> None:
    """The end line says whether the hold had no remedy, outlived its bound, released, or the queue moved."""
    clock = FakeClock()
    coordinator = _deferring_coordinator(
        clock,
        HoldLiveness(5.0, DISPATCH_HOLD_LIVENESS_SECONDS, HoldRemedy.PROBE),
    )
    scheduler = coordinator._inference_scheduler
    dispatcher = coordinator._message_dispatcher
    assert isinstance(scheduler, Mock) and isinstance(dispatcher, Mock)
    scheduler.dispatch_hold_liveness_active.return_value = False
    scheduler.dispatch_hold_liveness_seconds.return_value = liveness
    dispatcher.get_deadlock_snapshot.return_value.indicates_structural_wedge.return_value = queue_deadlocked
    captured, handler_id = _capture_warnings()
    try:
        coordinator.structural_queue_wedge_active()
    finally:
        logger.remove(handler_id)

    ended = [line for line in captured if line.startswith("Recovery stopped deferring")]
    assert len(ended) == 1, f"expected one end line, got {captured}"
    assert expected in ended[0], ended[0]
    match = SIGNATURES["dispatch_hold_deferral_end"].pattern.search(ended[0])
    assert match is not None and match.group("ended") == expected, "the registered signature no longer reads the line"
    assert coordinator._deferring_to_dispatch_hold is False
