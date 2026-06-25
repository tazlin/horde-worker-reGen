"""Reproduction: a deterministically-doomed inference pool flaps forever instead of giving up.

Observed behavior: a worker whose every inference child crashed on start (here, a CPU-only torch in
the child's env raising ``Torch not compiled with CUDA enabled`` during hordelib init) racked up 24
process recoveries in a single session and never terminated on its own; it kept respawning the doomed
pool until an operator stopped it. Sibling sessions on the same broken env *did* give up and exit
cleanly, the only difference being how fast each crash burst recurred.

Root cause: the save-our-ship abort fires only when both conditions hold *at the same tick*:

  1. the recovery episode has aged past ``give_up_after_seconds`` (the supervisor returns ``GIVE_UP``), and
  2. ``_is_inference_pool_unrecoverable()`` is True (every slot quarantined right now).

But the path to give-up runs a soft reset first, and the soft reset's ``rebuild_inference_pool`` clears
the quarantine set to respawn the slots. So while the episode ages toward give-up the pool looks
*recoverable* (slots un-quarantined, merely starting), and ``_give_up_on_wedged_jobs`` skips the abort.
Whether the abort is ever reached then hinges on a race: if the freshly respawned slots crash and
re-quarantine *before* a clean streak closes the episode, the still-open episode catches the pool fully
quarantined and aborts (the tight-loop sessions). But if the respawned slots are slow to crash again,
e.g. the lazy inference start is gated behind a failing/slow download, the not-wedged window outlasts
``clean_streak_seconds``; the episode closes and the give-up clock resets. The next crash burst opens a
*fresh* episode (age 0), the next soft reset un-quarantines before that episode can age past give-up,
and the worker flaps between soft reset and re-crash indefinitely, accumulating recoveries without ever
aborting.

These tests drive the real recovery supervisor, soft-reset, and give-up paths through that slow-restart
cycle (the pool's respawn/crash is emulated by toggling the quarantine set, since no real children run
in unit tests) and assert the worker eventually aborts. It does not, so they are RED until the abort
no longer depends on the pool being quarantined at the exact give-up tick.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.lifecycle.recovery_supervisor import (
    _DEFAULT_CLEAN_STREAK_SECONDS,
    RecoverySupervisor,
)

from .conftest import make_testable_process_manager


class _FakeClock:
    """A monotonic clock the test advances explicitly, so escalation timing is deterministic."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# A respawn-to-recrash window longer than the recovery clean streak: the doomed slots are slow to crash
# again (lazy inference start gated behind a failing download), so each soft reset's un-quarantine looks
# like a recovery and closes the episode before give-up can catch the pool quarantined.
_SLOW_RESTART_SECONDS = _DEFAULT_CLEAN_STREAK_SECONDS + 6.0


class TestDoomedPoolEventuallyAborts:
    """The whole point of give-up: a pool that can never serve must stop the worker, not loop forever."""

    def test_doomed_pool_with_slow_restart_eventually_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A doomed pool whose respawn is slower than the clean streak must still abort, not flap forever."""
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        max_procs = pm.max_inference_processes

        # Deterministic clock so the episode/give-up/clean-streak timing is exact.
        clock = _FakeClock()
        pm._recovery_supervisor = RecoverySupervisor(clock=clock)

        # A soft reset rebuilds the pool in place, which clears the quarantine set and respawns the
        # slots. Emulate just that effect (un-quarantine every slot) without launching real children.
        def _fake_rebuild_inference(*, reason: str) -> None:
            lifecycle._quarantined_inference_slots.clear()

        monkeypatch.setattr(lifecycle, "rebuild_inference_pool", _fake_rebuild_inference)
        monkeypatch.setattr(lifecycle, "rebuild_safety_pool", lambda *, reason: None)

        # The terminal action we care about: does the worker ever decide to abort the doomed pool?
        aborted = {"called": False}
        monkeypatch.setattr(pm, "_abort", lambda: aborted.__setitem__("called", True))

        def _crash_burst_quarantines_pool() -> None:
            """The doomed children crashed on start: the breaker quarantines every slot."""
            lifecycle._quarantined_inference_slots = set(range(max_procs))

        # Many crash/recover cycles: far more give-up opportunities than the 24 recoveries observed live.
        for _cycle in range(15):
            _crash_burst_quarantines_pool()

            # The supervisor notices the wedge and (after its grace) soft-resets, which un-quarantines
            # the pool. Tick through the soft reset and past the give-up age while un-quarantined.
            for _ in range(5):
                clock.advance(2.0)
                pm._run_recovery_supervisor()
                if aborted["called"]:
                    break

            # The respawned slots are slow to crash again: a not-wedged window that outlasts the clean
            # streak, closing the episode and resetting the give-up clock.
            for _ in range(int(_SLOW_RESTART_SECONDS // 2) + 1):
                clock.advance(2.0)
                pm._run_recovery_supervisor()

            if aborted["called"]:
                break

        assert aborted["called"], (
            "A deterministically-doomed inference pool never aborted: the save-our-ship loop soft-reset "
            "the pool indefinitely (racking up unbounded process recoveries) instead of giving up, "
            "because the abort requires the pool to be fully quarantined at the exact give-up tick and "
            "the soft reset's un-quarantine keeps that from ever coinciding."
        )

    def test_give_up_after_soft_reset_unquarantine_does_not_abort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Isolated phase mismatch: GIVE_UP arriving just after a soft-reset un-quarantine skips the abort.

        This is the single tick at the heart of the loop above. The supervisor has decided the episode is
        doomed (``GIVE_UP``), but the soft reset that preceded it already cleared the quarantine set, so
        ``_give_up_on_wedged_jobs`` sees a "recoverable" pool and declines to abort.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle

        aborted = {"called": False}
        monkeypatch.setattr(pm, "_abort", lambda: aborted.__setitem__("called", True))

        # The pool is doomed (it will re-crash), but a soft reset has *just* un-quarantined every slot, so
        # at this instant nothing is quarantined: exactly the state the episode reaches when give-up is due.
        lifecycle._quarantined_inference_slots = set()
        assert pm._is_inference_pool_unrecoverable() is False

        pm._give_up_on_wedged_jobs()

        assert aborted["called"], (
            "Give-up fired on a doomed pool but did not abort because the pool was transiently "
            "un-quarantined by the preceding soft reset; the worker keeps running and loops."
        )
