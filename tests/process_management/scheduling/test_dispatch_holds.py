"""The dispatch-hold ledger, stated directly: hold records, release attribution and the clearance spans.

The gate tests cover the decisions that place a hold and the records the scheduler emits; these rows pin the
bookkeeping itself, so a change to a counter or an attribution rule is caught where it lives.
"""

from __future__ import annotations

from horde_worker_regen.process_management.scheduling.ledgers.dispatch_holds import (
    DispatchHoldLedger,
    HoldRelease,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _ledger() -> tuple[DispatchHoldLedger, _Clock]:
    clock = _Clock()
    return DispatchHoldLedger(clock), clock


class TestDispatchHolds:
    """Every held pass counts a conflict; the first stamps the hold; the release names what freed the room."""

    def test_first_hold_stamps_and_counts_once(self) -> None:
        """Repeated holds of one job count conflicts each pass but a single distinct hold."""
        ledger, clock = _ledger()
        assert ledger.note_hold("j", reclaim_applied=False, room_inputs={"free_mb": 100}) is True
        clock.now += 5.0
        assert ledger.note_hold("j", reclaim_applied=False, room_inputs=None) is False
        assert (ledger.holds, ledger.conflicts) == (1, 2)
        assert ledger.hold_since["j"] == 1_000.0
        assert ledger.room_inputs["j"] == {"free_mb": 100}, "a pass without inputs keeps the last breakdown"
        assert ledger.held_seconds("j") == 5.0
        assert ledger.held_seconds("other") is None

    def test_release_attribution(self) -> None:
        """Measured attempt beats reclaim beats natural free, and each release folds its duration."""
        ledger, clock = _ledger()
        ledger.note_hold("natural", reclaim_applied=False, room_inputs=None)
        ledger.note_hold("reclaimed", reclaim_applied=True, room_inputs={"free_mb": 1})
        ledger.note_hold("probed", reclaim_applied=True, room_inputs=None)
        clock.now += 10.0
        natural = ledger.resolve_hold("natural", measured_attempt=False)
        reclaimed = ledger.resolve_hold("reclaimed", measured_attempt=False)
        probed = ledger.resolve_hold("probed", measured_attempt=True)
        assert natural is not None and natural.release is HoldRelease.NATURAL_FREE
        assert reclaimed is not None and reclaimed.release is HoldRelease.RECLAIM
        assert reclaimed.room_inputs == {"free_mb": 1}
        assert probed is not None and probed.release is HoldRelease.MEASURED_ATTEMPT
        assert (ledger.released_by_natural_free, ledger.released_by_reclaim, ledger.released_by_measured_attempt) == (
            1,
            1,
            1,
        )
        assert ledger.hold_seconds == 30.0
        assert ledger.hold_since == {} and ledger.reclaim_requested == set() and ledger.room_inputs == {}

    def test_resolving_an_unheld_job_is_a_silent_no_op(self) -> None:
        """The common admit-first-pass case leaves every counter untouched."""
        ledger, _ = _ledger()
        assert ledger.resolve_hold("never", measured_attempt=False) is None
        assert (ledger.holds, ledger.conflicts, ledger.hold_seconds) == (0, 0, 0.0)

    def test_prune_forgets_departed_jobs_without_counting_a_release(self) -> None:
        """A job that left the queue by another path is abandoned, not released, and its defer hold goes too."""
        ledger, clock = _ledger()
        ledger.note_hold("gone", reclaim_applied=True, room_inputs={"free_mb": 7})
        ledger.note_hold("kept", reclaim_applied=False, room_inputs=None)
        ledger.note_pp_defer("gone", deferred=True)
        ledger.note_pp_defer("kept", deferred=True)
        clock.now += 3.0
        abandoned = ledger.prune({"kept"})
        assert [(a.job_id, a.hold_seconds, a.room_inputs) for a in abandoned] == [("gone", 3.0, {"free_mb": 7})]
        assert set(ledger.hold_since) == {"kept"}
        assert ledger.pp_defer_holds == {"kept"}
        assert ledger.released_by_natural_free == 0 and ledger.hold_seconds == 0.0

    def test_standing_hold_needs_the_bound(self) -> None:
        """A hold is standing once it has stood the bound; an unheld job never is."""
        ledger, clock = _ledger()
        assert ledger.is_standing("j", 10.0) is False
        ledger.note_hold("j", reclaim_applied=False, room_inputs=None)
        clock.now += 9.0
        assert ledger.is_standing("j", 10.0) is False
        clock.now += 1.0
        assert ledger.is_standing("j", 10.0) is True

    def test_pp_defer_tracks_only_currently_deferred_heads(self) -> None:
        """A cleared gate forgets the head."""
        ledger, _ = _ledger()
        ledger.note_pp_defer("j", deferred=True)
        assert ledger.pp_defer_holds == {"j"}
        ledger.note_pp_defer("j", deferred=False)
        assert ledger.pp_defer_holds == set()


class TestClearanceHolds:
    """Clearance spans accumulate across re-armed holds and survive until the job leaves the in-progress set."""

    def test_spans_accumulate_across_holds(self) -> None:
        """Closed spans keep their time, a live span counts to now, and re-noting an open span is a no-op."""
        ledger, clock = _ledger()
        assert ledger.clearance_held_seconds("j") == 0.0
        ledger.note_clearance_hold("j")
        clock.now += 4.0
        ledger.note_clearance_hold("j")
        assert ledger.clearance_held_seconds("j") == 4.0
        ledger.resolve_clearance_hold("j")
        clock.now += 100.0
        assert ledger.clearance_held_seconds("j") == 4.0, "a resolved hold stops counting but keeps its time"
        assert "j" not in ledger.clearance_hold_ids
        ledger.note_clearance_hold("j")
        clock.now += 6.0
        assert ledger.clearance_held_seconds("j") == 10.0
        ledger.resolve_clearance_hold("j")
        ledger.resolve_clearance_hold("j")
        assert ledger.clearance_held_seconds("j") == 10.0

    def test_forget_drops_jobs_no_longer_in_progress(self) -> None:
        """Records self-heal to the live in-progress set."""
        ledger, _ = _ledger()
        ledger.note_clearance_hold("live")
        ledger.note_clearance_hold("done")
        ledger.forget_clearance(["live"])
        assert ledger.clearance_hold_ids == {"live"}
        assert set(ledger.clearance_hold_spans) == {"live"}
