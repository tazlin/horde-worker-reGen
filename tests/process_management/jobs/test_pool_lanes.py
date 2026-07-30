"""Table tests for the pure pool-lane advertising decision.

These exercise :func:`decide_pool_lane` and :func:`record_fixed_pop_outcome` in isolation: the weighted
round-robin proportionality, the empty-offer edge rules, the two-empties free-weight boost, and the seats-
saturate-processes case that starves the free lane. No popper or manager is involved.
"""

from __future__ import annotations

from horde_worker_regen.process_management.jobs.pool_lanes import (
    PoolLaneState,
    PoolLaneTally,
    decide_pool_lane,
    fold_pool_lane_outcome,
    record_fixed_pop_outcome,
)
from horde_worker_regen.process_management.scheduling.model_pool import PopLane


def _run_lanes(
    *,
    eligible: frozenset[str],
    active_seats: frozenset[str],
    process_count: int,
    iterations: int,
    initial_state: PoolLaneState | None = None,
) -> list[PopLane]:
    """Run ``iterations`` lane decisions threading the state forward, returning the chosen lane each time."""
    state = initial_state if initial_state is not None else PoolLaneState()
    chosen: list[PopLane] = []
    for _ in range(iterations):
        decision = decide_pool_lane(
            state,
            eligible=eligible,
            active_seats=active_seats,
            process_count=process_count,
        )
        chosen.append(decision.lane)
        state = decision.next_state
    return chosen


class TestEdgeRules:
    """An empty offer on either lane forces the other; both empty falls back to the free lane."""

    def test_empty_fixed_offer_forces_free_with_free_offer(self) -> None:
        """Test empty fixed offer forces free with free offer."""
        decision = decide_pool_lane(
            PoolLaneState(),
            eligible=frozenset({"a", "b"}),
            active_seats=frozenset({"z"}),
            process_count=2,
        )
        assert decision.lane is PopLane.FREE
        assert decision.advertised == frozenset({"a", "b"})

    def test_empty_free_offer_forces_fixed(self) -> None:
        """Test empty free offer forces fixed."""
        decision = decide_pool_lane(
            PoolLaneState(),
            eligible=frozenset({"a", "b"}),
            active_seats=frozenset({"a", "b"}),
            process_count=2,
        )
        assert decision.lane is PopLane.FIXED
        assert decision.advertised == frozenset({"a", "b"})

    def test_both_empty_falls_back_to_free_with_full_eligible(self) -> None:
        """Test both empty falls back to free with full eligible."""
        decision = decide_pool_lane(
            PoolLaneState(),
            eligible=frozenset(),
            active_seats=frozenset({"a"}),
            process_count=2,
        )
        assert decision.lane is PopLane.FREE
        assert decision.advertised == frozenset()

    def test_forced_choice_leaves_round_robin_credit_untouched(self) -> None:
        """Test forced choice leaves round robin credit untouched."""
        state = PoolLaneState(fixed_credit=3, free_credit=5, recent_fixed_empty=(True,))
        decision = decide_pool_lane(
            state,
            eligible=frozenset({"a", "b"}),
            active_seats=frozenset({"a", "b"}),
            process_count=2,
        )
        assert decision.next_state == state


class TestWeightedRoundRobin:
    """Both offers non-empty: the lane is chosen in proportion to its weight, smoothly interleaved."""

    def test_proportional_selection_over_a_sequence(self) -> None:
        """Test proportional selection over a sequence."""
        # fixed weight 3 (three seated eligible), free weight 1 (one uncommitted slot): a 3:1 interleave.
        chosen = _run_lanes(
            eligible=frozenset({"f1", "f2", "f3", "r1"}),
            active_seats=frozenset({"f1", "f2", "f3"}),
            process_count=4,
            iterations=8,
        )
        assert chosen.count(PopLane.FIXED) == 6
        assert chosen.count(PopLane.FREE) == 2

    def test_interleave_is_smooth_not_bursty(self) -> None:
        """Test interleave is smooth not bursty."""
        chosen = _run_lanes(
            eligible=frozenset({"f1", "f2", "f3", "r1"}),
            active_seats=frozenset({"f1", "f2", "f3"}),
            process_count=4,
            iterations=4,
        )
        # A 3:1 smooth interleave never emits the single free turn first, nor two free turns adjacent.
        assert chosen == [PopLane.FIXED, PopLane.FIXED, PopLane.FREE, PopLane.FIXED]


class TestFreeLaneStarvation:
    """When the seats saturate the inference slots, the free lane weight collapses to zero."""

    def test_seats_at_or_above_processes_starve_the_free_lane(self) -> None:
        """Test seats at or above processes starve the free lane."""
        chosen = _run_lanes(
            eligible=frozenset({"f1", "f2", "r1"}),
            active_seats=frozenset({"f1", "f2"}),
            process_count=1,
            iterations=6,
        )
        assert all(lane is PopLane.FIXED for lane in chosen)


class TestTwoEmptiesBoost:
    """Two consecutive empty fixed pops boost the free-lane weight so the offer yields back to the wider set."""

    def test_boost_engages_only_after_two_empty_fixed_pops(self) -> None:
        """Test boost engages only after two empty fixed pops."""
        eligible = frozenset({"f1", "f2", "r1"})
        active_seats = frozenset({"f1", "f2"})

        # With the free weight collapsed (seats >= processes) and no remembered emptiness, the free lane
        # never runs.
        without_boost = _run_lanes(
            eligible=eligible,
            active_seats=active_seats,
            process_count=2,
            iterations=3,
        )
        assert all(lane is PopLane.FIXED for lane in without_boost)

        # After two empty fixed pops the boost lifts the free weight to one, so the free lane earns a turn.
        boosted_state = record_fixed_pop_outcome(
            record_fixed_pop_outcome(PoolLaneState(), was_empty=True),
            was_empty=True,
        )
        with_boost = _run_lanes(
            eligible=eligible,
            active_seats=active_seats,
            process_count=2,
            iterations=3,
            initial_state=boosted_state,
        )
        assert PopLane.FREE in with_boost

    def test_single_empty_fixed_pop_does_not_boost(self) -> None:
        """Test single empty fixed pop does not boost."""
        one_empty = record_fixed_pop_outcome(PoolLaneState(), was_empty=True)
        chosen = _run_lanes(
            eligible=frozenset({"f1", "f2", "r1"}),
            active_seats=frozenset({"f1", "f2"}),
            process_count=2,
            iterations=3,
            initial_state=one_empty,
        )
        assert all(lane is PopLane.FIXED for lane in chosen)

    def test_a_fulfilled_fixed_pop_clears_the_boost(self) -> None:
        """Test a fulfilled fixed pop clears the boost."""
        two_empty = record_fixed_pop_outcome(
            record_fixed_pop_outcome(PoolLaneState(), was_empty=True),
            was_empty=True,
        )
        after_fulfillment = record_fixed_pop_outcome(two_empty, was_empty=False)
        chosen = _run_lanes(
            eligible=frozenset({"f1", "f2", "r1"}),
            active_seats=frozenset({"f1", "f2"}),
            process_count=2,
            iterations=3,
            initial_state=after_fulfillment,
        )
        assert all(lane is PopLane.FIXED for lane in chosen)


class TestRecordFixedPopOutcome:
    """The emptiness memory keeps only the two most recent fixed-lane outcomes."""

    def test_memory_is_bounded_to_two(self) -> None:
        """Test memory is bounded to two."""
        state = PoolLaneState()
        for was_empty in (True, False, True, True):
            state = record_fixed_pop_outcome(state, was_empty=was_empty)
        assert state.recent_fixed_empty == (True, True)

    def test_records_do_not_disturb_credit(self) -> None:
        """Test records do not disturb credit."""
        state = PoolLaneState(fixed_credit=2, free_credit=1)
        updated = record_fixed_pop_outcome(state, was_empty=True)
        assert updated.fixed_credit == 2
        assert updated.free_credit == 1


class TestPoolLaneTally:
    """Resident-hit counts advance only for successful pops whose model was already loaded."""

    def test_resident_fixed_match_advances_all_fixed_counts(self) -> None:
        """A resident fixed-lane match counts the pop, match, and resident hit together."""
        tally = fold_pool_lane_outcome(
            PoolLaneTally(),
            lane=PopLane.FIXED,
            fulfilled=True,
            resident_hit=True,
        )

        assert tally == PoolLaneTally(fixed_pops=1, fixed_fulfilled=1, fixed_resident_hits=1)

    def test_empty_pop_cannot_count_as_resident(self) -> None:
        """The resident flag is ignored when the pop did not return a model."""
        tally = fold_pool_lane_outcome(
            PoolLaneTally(),
            lane=PopLane.FREE,
            fulfilled=False,
            resident_hit=True,
        )

        assert tally == PoolLaneTally(free_pops=1)
