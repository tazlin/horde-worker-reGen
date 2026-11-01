"""The dispatch-hold ledger, stated directly: hold records, release attribution and the clearance spans.

The gate tests cover the decisions that place a hold and the records the scheduler emits; these rows pin the
bookkeeping itself, so a change to a counter or an attribution rule is caught where it lives.
"""

from __future__ import annotations

from horde_worker_regen.process_management.scheduling.ledgers.dispatch_holds import (
    DISPATCH_HOLD_LIVENESS_SECONDS,
    DispatchHoldLedger,
    HoldLivenessEnd,
    HoldRelease,
    HoldRemedy,
    hold_remedy,
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
        assert (
            ledger.note_hold("j", reclaim_applied=False, room_inputs={"free_mb": 100}, remedy=HoldRemedy.PROBE) is True
        )
        clock.now += 5.0
        assert ledger.note_hold("j", reclaim_applied=False, room_inputs=None, remedy=HoldRemedy.PROBE) is False
        assert (ledger.holds, ledger.conflicts) == (1, 2)
        assert ledger.hold_since["j"] == 1_000.0
        assert ledger.room_inputs["j"] == {"free_mb": 100}, "a pass without inputs keeps the last breakdown"
        assert ledger.held_seconds("j") == 5.0
        assert ledger.held_seconds("other") is None

    def test_release_attribution(self) -> None:
        """Measured attempt beats reclaim beats natural free, and each release folds its duration."""
        ledger, clock = _ledger()
        ledger.note_hold("natural", reclaim_applied=False, room_inputs=None, remedy=HoldRemedy.PROBE)
        ledger.note_hold("reclaimed", reclaim_applied=True, room_inputs={"free_mb": 1}, remedy=HoldRemedy.RECLAIM)
        ledger.note_hold("probed", reclaim_applied=True, room_inputs=None, remedy=HoldRemedy.RECLAIM)
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

    def test_a_reroute_release_is_counted_as_a_reroute_even_after_reclaim(self) -> None:
        """A rerouted hold is attributed to the reroute, though the gate had reclaimed for it earlier."""
        ledger, _ = _ledger()
        ledger.note_hold("rerouted", reclaim_applied=True, room_inputs={"free_mb": 2}, remedy=HoldRemedy.RECLAIM)
        released = ledger.resolve_hold("rerouted", measured_attempt=False, rerouted=True)
        assert released is not None and released.release is HoldRelease.REROUTE
        assert released.room_inputs == {"free_mb": 2}
        assert (ledger.released_by_reroute, ledger.released_by_reclaim) == (1, 0)
        assert ledger.reclaim_requested == set()

    def test_resolving_an_unheld_job_is_a_silent_no_op(self) -> None:
        """The common admit-first-pass case leaves every counter untouched."""
        ledger, _ = _ledger()
        assert ledger.resolve_hold("never", measured_attempt=False) is None
        assert (ledger.holds, ledger.conflicts, ledger.hold_seconds) == (0, 0, 0.0)

    def test_prune_forgets_departed_jobs_without_counting_a_release(self) -> None:
        """A job that left the queue by another path is abandoned, not released, and its defer hold goes too."""
        ledger, clock = _ledger()
        ledger.note_hold("gone", reclaim_applied=True, room_inputs={"free_mb": 7}, remedy=HoldRemedy.RECLAIM)
        ledger.note_hold("kept", reclaim_applied=False, room_inputs=None, remedy=HoldRemedy.PROBE)
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
        ledger.note_hold("j", reclaim_applied=False, room_inputs=None, remedy=HoldRemedy.PROBE)
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


class TestHoldRemedy:
    """Which remedy a hold pass left live, read from that pass's verdict."""

    @staticmethod
    def _remedy(
        *,
        denied: bool = False,
        reroute_pending: bool = False,
        reclaim_applied: bool = False,
        teardown_grace_open: bool = False,
        probe_pending: bool = False,
    ) -> HoldRemedy:
        return hold_remedy(
            denied=denied,
            reroute_pending=reroute_pending,
            reclaim_applied=reclaim_applied,
            teardown_grace_open=teardown_grace_open,
            probe_pending=probe_pending,
        )

    def test_a_deny_with_no_other_card_has_no_remedy(self) -> None:
        """No reclaim moves the ceiling and its attempt is refused or spent, so only another card is left."""
        assert self._remedy(denied=True, reclaim_applied=True, probe_pending=True) is HoldRemedy.NONE
        assert self._remedy(denied=True, reroute_pending=True) is HoldRemedy.REROUTE

    def test_a_defer_keeps_each_remedy_until_it_is_spent(self) -> None:
        """Reclaim, then the teardown grace, then the probe. A DEFER with all three spent has none."""
        assert self._remedy(reclaim_applied=True, probe_pending=True) is HoldRemedy.RECLAIM
        assert self._remedy(teardown_grace_open=True, probe_pending=True) is HoldRemedy.TEARDOWN_GRACE
        assert self._remedy(probe_pending=True) is HoldRemedy.PROBE
        assert self._remedy() is HoldRemedy.NONE


class TestHoldLiveness:
    """A hold keeps recovery waiting while its latest pass left a remedy live, up to the liveness bound."""

    def test_a_hold_with_no_remedy_is_inactive_at_once(self) -> None:
        """A terminal DENY on its first pass ends the deferral without waiting out any of the bound."""
        ledger, _ = _ledger()
        ledger.note_hold("dead", reclaim_applied=False, room_inputs=None, remedy=HoldRemedy.NONE)
        liveness = ledger.liveness("dead")
        assert liveness is not None
        assert liveness.held_seconds == 0.0
        assert liveness.active is False
        assert liveness.ended_by is HoldLivenessEnd.NO_REMEDY

    def test_a_defer_with_reclaim_accepted_stays_active_to_the_bound(self) -> None:
        """Reclaim on the way keeps the hold live up to the bound, and the bound still ends it."""
        ledger, clock = _ledger()
        ledger.note_hold("reclaiming", reclaim_applied=True, room_inputs=None, remedy=HoldRemedy.RECLAIM)
        clock.now += DISPATCH_HOLD_LIVENESS_SECONDS - 1.0
        liveness = ledger.liveness("reclaiming")
        assert liveness is not None and liveness.active is True and liveness.ended_by is None
        clock.now += 1.0
        lapsed = ledger.liveness("reclaiming")
        assert lapsed is not None and lapsed.active is False
        assert lapsed.ended_by is HoldLivenessEnd.BOUND_LAPSED, "a live remedy does not outlast the bound"

    def test_a_ceiling_attempt_pending_convergence_stays_active(self) -> None:
        """The arbiter holds back the ceiling DENY as a DEFER while the card converges; its probe keeps it live."""
        remedy = hold_remedy(
            denied=False,
            reroute_pending=False,
            reclaim_applied=False,
            teardown_grace_open=False,
            probe_pending=True,
        )
        ledger, clock = _ledger()
        ledger.note_hold("converging", reclaim_applied=False, room_inputs=None, remedy=remedy)
        clock.now += 60.0
        liveness = ledger.liveness("converging")
        assert liveness is not None and liveness.remedy is HoldRemedy.PROBE and liveness.active is True

    def test_the_latest_pass_decides_the_remedy(self) -> None:
        """A hold whose probe is spent on a later pass stops being live on that pass."""
        ledger, clock = _ledger()
        ledger.note_hold("j", reclaim_applied=False, room_inputs=None, remedy=HoldRemedy.PROBE)
        clock.now += 30.0
        ledger.note_hold("j", reclaim_applied=False, room_inputs=None, remedy=HoldRemedy.NONE)
        liveness = ledger.liveness("j")
        assert liveness is not None and liveness.held_seconds == 30.0 and liveness.active is False

    def test_an_unheld_job_has_no_liveness_and_a_release_forgets_the_remedy(self) -> None:
        """A released or pruned hold leaves no remedy behind for a later hold of the same job to inherit."""
        ledger, _ = _ledger()
        assert ledger.liveness("never") is None
        ledger.note_hold("released", reclaim_applied=False, room_inputs=None, remedy=HoldRemedy.PROBE)
        ledger.note_hold("pruned", reclaim_applied=False, room_inputs=None, remedy=HoldRemedy.PROBE)
        ledger.resolve_hold("released", measured_attempt=False)
        ledger.prune(set())
        assert ledger.liveness("released") is None and ledger.liveness("pruned") is None
        assert ledger.remedies == {}


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


class TestPostProcessingTurn:
    """The turn a running chain's held job is owed, and how long the lane waits on it."""

    def test_the_first_held_job_keeps_the_turn(self) -> None:
        """A later claim does not displace the job a running chain held first."""
        ledger, clock = _ledger()
        ledger.claim_pp_turn("first", 0)
        clock.now += 3.0
        ledger.claim_pp_turn("second", 0)
        assert ledger.pp_turn is not None
        assert (ledger.pp_turn.job_id, ledger.pp_turn.claimed_at) == ("first", 1_000.0)

    def test_the_lane_waits_as_long_as_the_job_waited(self) -> None:
        """A job held twenty seconds is waited on for twenty, counted from the first pass the lane yields."""
        ledger, clock = _ledger()
        ledger.claim_pp_turn("j", None)
        assert ledger.pp_yield_expired(floor_seconds=5.0) is False, "no yield has begun while the chain runs"
        clock.now += 20.0
        ledger.begin_pp_yield()
        clock.now += 4.0
        ledger.begin_pp_yield()
        clock.now += 16.0
        assert ledger.pp_yield_expired(floor_seconds=5.0) is False
        clock.now += 0.5
        assert ledger.pp_yield_expired(floor_seconds=5.0) is True

    def test_a_briefly_held_job_is_waited_on_for_the_floor(self) -> None:
        """A job held for a moment still gets the floor to be dispatched and cleared."""
        ledger, clock = _ledger()
        ledger.claim_pp_turn("j", None)
        clock.now += 1.0
        ledger.begin_pp_yield()
        clock.now += 5.0
        assert ledger.pp_yield_expired(floor_seconds=5.0) is False
        clock.now += 0.5
        assert ledger.pp_yield_expired(floor_seconds=5.0) is True
        ledger.clear_pp_turn()
        assert ledger.pp_turn is None
        assert ledger.pp_yield_expired(floor_seconds=5.0) is False


class TestUtilitiesLaneHolds:
    """A lane hold is stamped once per job and released with its held seconds when the job is no longer waiting."""

    def test_hold_counts_once_and_release_reports_the_held_seconds(self) -> None:
        """Re-holding a held job is not a new hold, and a job left out of ``keep`` is released."""
        ledger, clock = _ledger()
        assert ledger.note_utilities_lane_hold("a") is True
        assert ledger.note_utilities_lane_hold("b") is True
        clock.now += 4.0
        assert ledger.note_utilities_lane_hold("a") is False
        assert ledger.utilities_lane_hold_count == 2

        assert ledger.release_utilities_lane_holds(keep={"b"}) == [("a", 4.0)]
        assert set(ledger.utilities_lane_holds) == {"b"}
        assert ledger.release_utilities_lane_holds(keep=()) == [("b", 4.0)]
        assert ledger.utilities_lane_holds == {}
