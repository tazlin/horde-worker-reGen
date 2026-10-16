"""The wedge assessment defers to a standing dispatch hold inside the hold's own liveness bound.

An idle card with pending work is the raw shape a structural wedge is read from, and a residency
reconciliation hold presents exactly that shape while the gate reclaims or waits on its probe. The recovery
coordinator answered it with pool resets and, past its budget, faulted the head the gate was about to admit.
The hold is now one of the deliberate holds the assessment excuses, bounded so a hold that never resolves
still trips the supervisor, and the deferral is logged once on each edge.

The dispatcher's queue-deadlock latch keeps its age underneath an excuse or an inference-slot boot, so the
wedge bound is measured again from the moment either ends: otherwise the wedge reads as sustained the tick
the hold lifts and a soft reset rebuilds a pool that was about to dispatch.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.message_dispatcher import (
    _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS,
    DeadlockSnapshot,
)
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator import WorkerRecoveryCoordinator
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
