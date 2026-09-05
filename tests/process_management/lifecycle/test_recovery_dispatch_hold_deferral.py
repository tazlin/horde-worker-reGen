"""The wedge assessment defers to a standing dispatch hold inside the hold's own liveness bound.

An idle card with pending work is the raw shape a structural wedge is read from, and a residency
reconciliation hold presents exactly that shape while the gate reclaims or waits on its probe. The recovery
coordinator answered it with pool resets and, past its budget, faulted the head the gate was about to admit.
The hold is now one of the deliberate holds the assessment excuses, bounded so a hold that never resolves
still trips the supervisor, and the deferral is logged once on each edge.
"""

from __future__ import annotations

from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
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
