"""Reproduction: a transient queue deadlock must not trip the save-our-ship wedge.

Observed behavior: on a worker that auto-scaled to several inference processes serving many heavy
models, a head-of-queue job whose model was *actively being preloaded* was faulted by save-our-ship.
The queue-deadlock detector flags the brief all-idle window between a job finishing and the scheduler
preloading the next model as a deadlock, and holds that flag across the preload (its anti-flap guard
keeps it set while a process is starting). That instantaneous flag feeds the recovery supervisor,
whose short give-up budget is calibrated only for *definitive* crash-loop signals (a slow model load
"never trips" it, per its docstring). So a head that is merely waiting for its model to load is treated
as an unrecoverable wedge and faulted.

The fix makes a queue deadlock count as a *structural* wedge only once it has persisted past any normal
model-load / churn window, restoring the supervisor's definitive-signal assumption. These tests assert
that fixed behavior and are therefore RED against the pre-fix code (which treats an instantaneous queue
deadlock as structural).
"""

from __future__ import annotations

import time

from horde_worker_regen.process_management.message_dispatcher import DeadlockSnapshot

from .conftest import make_testable_process_manager

# The normal all-idle gap between a job finishing and the scheduler preloading the next model: the
# scheduler fills it within about one control-loop tick, so it is not a wedge.
_TRANSIENT_AGE_SECONDS = 0.0
# A queue deadlock that has outlasted any model-load / churn window: a genuine structural wedge.
_SUSTAINED_AGE_SECONDS = 60.0


def _queue_deadlock_snapshot(*, started_at: float) -> DeadlockSnapshot:
    """A snapshot whose queue deadlock began at ``started_at`` (head model not yet resident)."""
    return DeadlockSnapshot(
        in_deadlock=True,
        in_queue_deadlock=True,
        deadlock_started_at=started_at,
        queue_deadlock_started_at=started_at,
        queue_deadlock_model="Abyss OrangeMix",
        queue_deadlock_process_id=None,
    )


class TestTransientQueueWedgeIsNotStructural:
    """The signal itself: a just-detected queue deadlock is churn, a sustained one is a wedge."""

    def test_just_detected_queue_deadlock_is_not_structural(self) -> None:
        """A queue deadlock detected this instant is the normal between-jobs window, not a wedge."""
        snapshot = _queue_deadlock_snapshot(started_at=time.time() - _TRANSIENT_AGE_SECONDS)
        assert snapshot.indicates_structural_wedge() is False

    def test_sustained_queue_deadlock_is_structural(self) -> None:
        """A queue deadlock that outlasts any model-load window is a genuine structural wedge."""
        snapshot = _queue_deadlock_snapshot(started_at=time.time() - _SUSTAINED_AGE_SECONDS)
        assert snapshot.indicates_structural_wedge() is True


class TestAssessWedgeIgnoresTransientChurn:
    """Ties the signal to the real save-our-ship trigger (`_assess_wedge`) that faulted the head."""

    def test_assess_wedge_false_for_transient_queue_deadlock(self) -> None:
        """A worker mid-preload (transient queue deadlock) is not the SOS wedge that faults the head."""
        pm = make_testable_process_manager()
        # Precondition: nothing else makes this worker wedged (no quarantined pool, no orphan storm).
        assert pm._is_inference_pool_unrecoverable() is False
        dispatcher = pm._message_dispatcher
        dispatcher._in_queue_deadlock = True
        dispatcher._last_queue_deadlock_detected_time = time.time() - _TRANSIENT_AGE_SECONDS

        assert pm._assess_wedge() is False

    def test_assess_wedge_true_for_sustained_queue_deadlock(self) -> None:
        """A sustained queue deadlock still trips the SOS wedge so a genuine wedge is not ignored."""
        pm = make_testable_process_manager()
        dispatcher = pm._message_dispatcher
        dispatcher._in_queue_deadlock = True
        dispatcher._last_queue_deadlock_detected_time = time.time() - _SUSTAINED_AGE_SECONDS

        assert pm._assess_wedge() is True
