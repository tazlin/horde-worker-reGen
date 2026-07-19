"""Behavioral tests for the fixed model pool seat/bench/decay engine.

Time is injected in seconds throughout; the engine never reads the clock, so every scenario drives ``now``
explicitly and asserts the observable seat/bench outcome (which models serve, which are benched, and the
transitions emitted) rather than internal bookkeeping.
"""

from __future__ import annotations

from collections.abc import Sequence

from horde_worker_regen.process_management.scheduling.model_pool import (
    DemotionReason,
    ModelPool,
    PinnedModel,
    PoolParams,
    PopLane,
    RankedCandidate,
    SeatSource,
    SeatState,
    SeatTransition,
    TransitionKind,
)

_MINUTE = 60.0


def _params(seat_count: int, **overrides: object) -> PoolParams:
    """Build pool params with the given seat count and any field overrides."""
    return PoolParams(seat_count=seat_count, **overrides)  # type: ignore[arg-type]


def _cand(
    name: str,
    score: float,
    *,
    on_disk: bool = True,
    eta: float | None = None,
    queued_per_worker: float | None = None,
) -> RankedCandidate:
    """Build a ranked candidate with compact keyword defaults."""
    return RankedCandidate(
        name=name, score=score, on_disk=on_disk, eta_seconds=eta, queued_per_worker=queued_per_worker
    )


def _kinds(transitions: Sequence[SeatTransition]) -> list[TransitionKind]:
    """Return the transition kinds in order."""
    return [transition.kind for transition in transitions]


def _of_kind(transitions: Sequence[SeatTransition], kind: TransitionKind) -> list[SeatTransition]:
    """Return the transitions of a given kind."""
    return [transition for transition in transitions if transition.kind == kind]


class TestSeatFill:
    """Empty seats fill from manual pins in affinity order, then from ranker candidates by score."""

    def test_manual_pins_fill_by_affinity_order(self) -> None:
        """The highest-affinity pin claims the first empty seat, the next-highest the second."""
        pool = ModelPool(_params(2, pinned=(PinnedModel("low", 0.5), PinnedModel("high", 1.0)), ranker_enabled=False))

        transitions = pool.tick(0.0, ranked=None, demand_is_stale=False)

        seats = pool.seats()
        assert seats[0].model == "high"
        assert seats[1].model == "low"
        assert all(seat.source is SeatSource.MANUAL for seat in seats)
        assert [transition.model for transition in _of_kind(transitions, TransitionKind.SEATED)] == ["high", "low"]

    def test_ranker_fills_remaining_seats_by_score(self) -> None:
        """A single pin takes one seat; the ranker fills the rest highest-score first."""
        pool = ModelPool(_params(3, pinned=(PinnedModel("pinned", 1.0),), ranker_enabled=True))

        pool.tick(0.0, ranked=[_cand("weak", 5.0), _cand("strong", 10.0)], demand_is_stale=False)

        seats = pool.seats()
        assert seats[0].model == "pinned"
        assert seats[0].source is SeatSource.MANUAL
        assert seats[1].model == "strong"
        assert seats[2].model == "weak"
        assert seats[1].source is SeatSource.RANKER


class TestMinDwell:
    """A freshly-seated model is protected from every demotion until its dwell elapses."""

    def test_dwell_blocks_empty_pop_demotion_then_allows(self) -> None:
        """A seat past its empty-pop threshold with dead demand survives within dwell and demotes after it."""
        pool = ModelPool(
            _params(
                1,
                pinned=(PinnedModel("pinned", 1.0),),
                ranker_enabled=False,
                min_dwell_minutes=10.0,
                empty_pop_demotion_threshold=40,
            ),
        )
        pool.tick(0.0, ranked=None, demand_is_stale=False)
        for _ in range(40):
            pool.on_pop_outcome(lane=PopLane.FIXED, advertised=frozenset({"pinned"}), popped_model=None, now=1.0)

        dead_demand = [_cand("pinned", 1.0, queued_per_worker=0.0)]
        within_dwell = pool.tick(5.0 * _MINUTE, ranked=dead_demand, demand_is_stale=False)
        assert pool.active_seat_models() == frozenset({"pinned"})
        assert _of_kind(within_dwell, TransitionKind.DEMOTED) == []

        past_dwell = pool.tick(11.0 * _MINUTE, ranked=dead_demand, demand_is_stale=False)
        assert pool.active_seat_models() == frozenset()
        demotions = _of_kind(past_dwell, TransitionKind.DEMOTED)
        assert len(demotions) == 1
        assert demotions[0].reason is DemotionReason.EMPTY_POPS


class TestEmptyPopDemotion:
    """Empty-pop demotion requires both a threshold breach and a near-zero demand signal, and resets on work."""

    def test_threshold_with_near_zero_demand_demotes(self) -> None:
        """A seat over the empty-pop threshold with near-zero queue is demoted after dwell."""
        pool = ModelPool(_params(1, pinned=(PinnedModel("m", 1.0),), ranker_enabled=False, min_dwell_minutes=0.0))
        pool.tick(0.0, ranked=None, demand_is_stale=False)
        for _ in range(40):
            pool.on_pop_outcome(lane=PopLane.FIXED, advertised=frozenset({"m"}), popped_model=None, now=1.0)

        transitions = pool.tick(2.0, ranked=[_cand("m", 1.0, queued_per_worker=0.0)], demand_is_stale=False)

        assert pool.active_seat_models() == frozenset()
        assert _of_kind(transitions, TransitionKind.DEMOTED)[0].reason is DemotionReason.EMPTY_POPS

    def test_threshold_without_near_zero_demand_does_not_demote(self) -> None:
        """The same empty-pop count with live queued demand does not demote (the AND condition holds)."""
        pool = ModelPool(_params(1, pinned=(PinnedModel("m", 1.0),), ranker_enabled=False, min_dwell_minutes=0.0))
        pool.tick(0.0, ranked=None, demand_is_stale=False)
        for _ in range(40):
            pool.on_pop_outcome(lane=PopLane.FIXED, advertised=frozenset({"m"}), popped_model=None, now=1.0)

        transitions = pool.tick(2.0, ranked=[_cand("m", 1.0, queued_per_worker=5.0)], demand_is_stale=False)

        assert pool.active_seat_models() == frozenset({"m"})
        assert _of_kind(transitions, TransitionKind.DEMOTED) == []

    def test_fulfillment_resets_the_counter(self) -> None:
        """A fulfillment clears the empty-pop count, so the seat is no longer a demotion candidate."""
        pool = ModelPool(_params(1, pinned=(PinnedModel("m", 1.0),), ranker_enabled=False, min_dwell_minutes=0.0))
        pool.tick(0.0, ranked=None, demand_is_stale=False)
        for _ in range(40):
            pool.on_pop_outcome(lane=PopLane.FIXED, advertised=frozenset({"m"}), popped_model=None, now=1.0)
        pool.on_pop_outcome(lane=PopLane.FIXED, advertised=frozenset({"m"}), popped_model="m", now=2.0)

        transitions = pool.tick(3.0, ranked=[_cand("m", 1.0, queued_per_worker=0.0)], demand_is_stale=False)

        assert pool.active_seat_models() == frozenset({"m"})
        assert _of_kind(transitions, TransitionKind.DEMOTED) == []
        assert pool.seats()[0].empty_pops == 0


class TestZeroFulfillmentDemotion:
    """A seat that never produces work for the zero-fulfillment window is demoted regardless of pop counts."""

    def test_no_fulfillment_window_demotes(self) -> None:
        """After the zero-fulfillment window with no work, the seat demotes even with no empty pops recorded."""
        pool = ModelPool(
            _params(
                1,
                pinned=(PinnedModel("m", 1.0),),
                ranker_enabled=False,
                min_dwell_minutes=10.0,
                zero_fulfillment_demotion_minutes=15.0,
            ),
        )
        pool.tick(0.0, ranked=None, demand_is_stale=False)

        before_window = pool.tick(14.0 * _MINUTE, ranked=None, demand_is_stale=False)
        assert pool.active_seat_models() == frozenset({"m"})
        assert _of_kind(before_window, TransitionKind.DEMOTED) == []

        after_window = pool.tick(16.0 * _MINUTE, ranked=None, demand_is_stale=False)
        assert pool.active_seat_models() == frozenset()
        assert _of_kind(after_window, TransitionKind.DEMOTED)[0].reason is DemotionReason.ZERO_FULFILLMENT


class TestTimerRecontest:
    """At its rotation deadline a seat holds unless a challenger beats its affinity-boosted score by the margin."""

    def _recontest_params(self, **overrides: object) -> PoolParams:
        return _params(
            1,
            ranker_enabled=True,
            rotation_minutes=10.0,
            min_dwell_minutes=5.0,
            rotation_margin=0.25,
            affinity_bonus_weight=0.3,
            zero_fulfillment_demotion_minutes=100000.0,
            **overrides,
        )

    def test_incumbent_holds_against_sub_margin_challenger(self) -> None:
        """A challenger inside the rotation margin does not unseat the incumbent at its deadline."""
        pool = ModelPool(self._recontest_params())
        pool.tick(0.0, ranked=[_cand("incumbent", 10.0)], demand_is_stale=False)

        transitions = pool.tick(
            10.0 * _MINUTE,
            ranked=[_cand("incumbent", 10.0), _cand("challenger", 12.0)],
            demand_is_stale=False,
        )

        assert pool.active_seat_models() == frozenset({"incumbent"})
        assert _of_kind(transitions, TransitionKind.SEATED) == []

    def test_incumbent_displaced_by_margin_beating_challenger(self) -> None:
        """A challenger clearing the margin unseats the incumbent, which benches with the timer cooldown."""
        pool = ModelPool(self._recontest_params())
        pool.tick(0.0, ranked=[_cand("incumbent", 10.0)], demand_is_stale=False)

        transitions = pool.tick(
            10.0 * _MINUTE,
            ranked=[_cand("incumbent", 10.0), _cand("challenger", 13.0)],
            demand_is_stale=False,
        )

        assert pool.active_seat_models() == frozenset({"challenger"})
        assert _of_kind(transitions, TransitionKind.DEMOTED)[0].reason is DemotionReason.TIMER_LOST
        assert _of_kind(transitions, TransitionKind.SEATED)[0].model == "challenger"
        assert "incumbent" in {entry.model for entry in pool.bench()}

    def test_affinity_bonus_and_extended_deadline_keep_a_pin(self) -> None:
        """A pin's affinity bonus and extended deadline hold a seat a bare ranker incumbent would have lost."""
        pool = ModelPool(self._recontest_params(pinned=(PinnedModel("pinned", 1.0),)))
        pool.tick(0.0, ranked=[_cand("pinned", 10.0)], demand_is_stale=False)

        # A bare ranker incumbent (effective 10.0, required 12.5) would lose to a 12.5 challenger. The pin's
        # affinity bonus raises its effective score to 10.3 (required 12.875), and its deadline is doubled to
        # 20 minutes, so at 21 minutes with the same challenger it still holds.
        transitions = pool.tick(
            21.0 * _MINUTE,
            ranked=[_cand("pinned", 10.0), _cand("challenger", 12.5)],
            demand_is_stale=False,
        )

        assert pool.active_seat_models() == frozenset({"pinned"})
        assert _of_kind(transitions, TransitionKind.SEATED) == []


class TestBenchCooldown:
    """A demoted model is held out of seating for its cooldown, then becomes eligible again."""

    def test_cooldown_blocks_then_allows_reseating(self) -> None:
        """A benched pin cannot refill its own seat until the cooldown lapses, then it does."""
        pool = ModelPool(
            _params(
                1,
                pinned=(PinnedModel("m", 1.0),),
                ranker_enabled=False,
                min_dwell_minutes=0.0,
                empty_pop_demotion_threshold=40,
                bench_cooldown_empty_pops_minutes=20.0,
            ),
        )
        pool.tick(0.0, ranked=None, demand_is_stale=False)
        for _ in range(40):
            pool.on_pop_outcome(lane=PopLane.FIXED, advertised=frozenset({"m"}), popped_model=None, now=1.0)
        pool.tick(2.0, ranked=[_cand("m", 1.0, queued_per_worker=0.0)], demand_is_stale=False)
        assert pool.active_seat_models() == frozenset()

        still_benched = pool.tick(10.0 * _MINUTE, ranked=None, demand_is_stale=False)
        assert pool.active_seat_models() == frozenset()
        assert _of_kind(still_benched, TransitionKind.SEATED) == []

        after_cooldown = pool.tick(25.0 * _MINUTE, ranked=None, demand_is_stale=False)
        assert pool.active_seat_models() == frozenset({"m"})
        assert _of_kind(after_cooldown, TransitionKind.SEATED)[0].model == "m"


class TestDemandStale:
    """Stale demand freezes rotation, re-contest, and rescue, but still fills empty seats from manual pins."""

    def test_stale_fills_manual_but_not_ranker(self) -> None:
        """Under stale demand a manual pin fills its seat while the ranker seat is left empty."""
        pool = ModelPool(_params(2, pinned=(PinnedModel("pinned", 1.0),), ranker_enabled=True))

        pool.tick(0.0, ranked=[_cand("hot", 100.0)], demand_is_stale=True)

        assert pool.active_seat_models() == frozenset({"pinned"})
        assert any(seat.model is None for seat in pool.seats())

    def test_stale_freezes_recontest(self) -> None:
        """A margin-beating challenger cannot unseat an incumbent while demand is stale, even past deadline."""
        pool = ModelPool(_params(1, ranker_enabled=True, rotation_minutes=10.0, min_dwell_minutes=5.0))
        pool.tick(0.0, ranked=[_cand("incumbent", 10.0)], demand_is_stale=False)

        transitions = pool.tick(
            60.0 * _MINUTE,
            ranked=[_cand("incumbent", 10.0), _cand("challenger", 1000.0)],
            demand_is_stale=True,
        )

        assert pool.active_seat_models() == frozenset({"incumbent"})
        assert _of_kind(transitions, TransitionKind.SEATED) == []


class TestRescue:
    """Opt-in rescue lets a starved, high-wait model borrow the weakest ranker seat, with a per-model cooldown."""

    def _rescue_params(self, seat_count: int, **overrides: object) -> PoolParams:
        return _params(
            seat_count,
            ranker_enabled=True,
            rescue_enabled=True,
            rescue_eta_seconds=10000.0,
            rescue_window_minutes=15.0,
            rescue_model_cooldown_hours=6.0,
            min_dwell_minutes=0.0,
            rotation_minutes=100000.0,
            zero_fulfillment_demotion_minutes=100000.0,
            **overrides,
        )

    def test_rescue_claims_the_weakest_ranker_seat(self) -> None:
        """A candidate above the rescue ETA displaces the lowest-scoring ranker seat and benches it."""
        pool = ModelPool(self._rescue_params(2))
        pool.tick(0.0, ranked=[_cand("strong", 10.0), _cand("weak", 8.0)], demand_is_stale=False)

        transitions = pool.tick(
            1.0,
            ranked=[_cand("strong", 10.0), _cand("weak", 8.0), _cand("starved", 0.5, eta=20000.0)],
            demand_is_stale=False,
        )

        assert pool.active_seat_models() == frozenset({"strong", "starved"})
        assert _of_kind(transitions, TransitionKind.RESCUE_ENGAGED)[0].model == "starved"
        assert _of_kind(transitions, TransitionKind.DEMOTED)[0].reason is DemotionReason.RESCUE_DISPLACED
        assert "weak" in {entry.model for entry in pool.bench()}

    def test_rescue_releases_early_when_demand_recovers(self) -> None:
        """Once the rescued model's wait drops back below the ETA it releases its borrowed seat."""
        pool = ModelPool(self._rescue_params(2))
        pool.tick(0.0, ranked=[_cand("strong", 10.0), _cand("weak", 8.0)], demand_is_stale=False)
        pool.tick(1.0, ranked=[_cand("strong", 10.0), _cand("starved", 0.5, eta=20000.0)], demand_is_stale=False)
        assert any(seat.source is SeatSource.RESCUE for seat in pool.seats())

        transitions = pool.tick(
            2.0, ranked=[_cand("strong", 10.0), _cand("starved", 0.5, eta=5000.0)], demand_is_stale=False
        )

        assert _of_kind(transitions, TransitionKind.RESCUE_RELEASED)[0].model == "starved"
        assert all(seat.source is not SeatSource.RESCUE for seat in pool.seats())

    def test_rescue_releases_on_window_expiry(self) -> None:
        """A rescue whose demand never recovers is released when its window closes."""
        pool = ModelPool(self._rescue_params(2))
        pool.tick(0.0, ranked=[_cand("strong", 10.0), _cand("weak", 8.0)], demand_is_stale=False)
        pool.tick(1.0, ranked=[_cand("strong", 10.0), _cand("starved", 0.5, eta=20000.0)], demand_is_stale=False)

        transitions = pool.tick(
            1.0 + 16.0 * _MINUTE,
            ranked=[_cand("strong", 10.0), _cand("starved", 0.5, eta=20000.0)],
            demand_is_stale=False,
        )

        assert _of_kind(transitions, TransitionKind.RESCUE_RELEASED)[0].model == "starved"

    def test_rescue_cooldown_blocks_reengagement(self) -> None:
        """A model just rescued cannot be rescued again within its cooldown, even when starved anew."""
        pool = ModelPool(self._rescue_params(2))
        pool.tick(0.0, ranked=[_cand("strong", 10.0), _cand("weak", 8.0)], demand_is_stale=False)
        pool.tick(
            1.0,
            ranked=[_cand("strong", 10.0), _cand("weak", 8.0), _cand("starved", 0.5, eta=20000.0)],
            demand_is_stale=False,
        )
        # Recover demand so the rescue releases, then let a fresh candidate take the freed seat.
        pool.tick(
            2.0,
            ranked=[_cand("strong", 10.0), _cand("filler", 9.0), _cand("starved", 0.5, eta=5000.0)],
            demand_is_stale=False,
        )
        assert pool.active_seat_models() == frozenset({"strong", "filler"})

        transitions = pool.tick(
            3.0,
            ranked=[_cand("strong", 10.0), _cand("filler", 9.0), _cand("starved", 0.9, eta=40000.0)],
            demand_is_stale=False,
        )

        assert _of_kind(transitions, TransitionKind.RESCUE_ENGAGED) == []
        assert pool.active_seat_models() == frozenset({"strong", "filler"})


class TestPendingDownload:
    """A re-contest won by a not-yet-downloaded challenger keeps the incumbent serving until the file lands."""

    def _pending_params(self) -> PoolParams:
        return _params(
            1,
            ranker_enabled=True,
            rotation_minutes=10.0,
            min_dwell_minutes=0.0,
            zero_fulfillment_demotion_minutes=100000.0,
        )

    def test_incumbent_serves_while_download_pending(self) -> None:
        """A winning off-disk challenger marks the seat pending while its incumbent keeps serving."""
        pool = ModelPool(self._pending_params())
        pool.tick(0.0, ranked=[_cand("incumbent", 10.0)], demand_is_stale=False)

        transitions = pool.tick(
            10.0 * _MINUTE,
            ranked=[_cand("incumbent", 10.0), _cand("challenger", 100.0, on_disk=False)],
            demand_is_stale=False,
        )

        seat = pool.seats()[0]
        assert seat.model == "incumbent"
        assert seat.state is SeatState.PENDING_DOWNLOAD
        assert seat.pending_model == "challenger"
        assert pool.active_seat_models() == frozenset({"incumbent"})
        assert _of_kind(transitions, TransitionKind.DOWNLOAD_PENDING)[0].model == "challenger"

    def test_download_ready_flips_and_benches_incumbent(self) -> None:
        """When the download completes the challenger takes the seat and the incumbent benches."""
        pool = ModelPool(self._pending_params())
        pool.tick(0.0, ranked=[_cand("incumbent", 10.0)], demand_is_stale=False)
        pool.tick(
            10.0 * _MINUTE,
            ranked=[_cand("incumbent", 10.0), _cand("challenger", 100.0, on_disk=False)],
            demand_is_stale=False,
        )

        transitions = pool.on_download_ready("challenger", now=11.0 * _MINUTE)

        assert pool.active_seat_models() == frozenset({"challenger"})
        assert pool.seats()[0].state is SeatState.ACTIVE
        assert pool.seats()[0].pending_model is None
        assert _of_kind(transitions, TransitionKind.DOWNLOAD_READY)[0].model == "challenger"
        assert "incumbent" in {entry.model for entry in pool.bench()}

    def test_download_failed_keeps_incumbent_and_cools_the_model(self) -> None:
        """A failed download leaves the incumbent serving and holds the failed model out for its cooldown."""
        pool = ModelPool(self._pending_params())
        pool.tick(0.0, ranked=[_cand("incumbent", 10.0)], demand_is_stale=False)
        pool.tick(
            10.0 * _MINUTE,
            ranked=[_cand("incumbent", 10.0), _cand("challenger", 100.0, on_disk=False)],
            demand_is_stale=False,
        )

        transitions = pool.on_download_failed("challenger", now=11.0 * _MINUTE)

        seat = pool.seats()[0]
        assert seat.model == "incumbent"
        assert seat.state is SeatState.ACTIVE
        assert seat.pending_model is None
        assert _of_kind(transitions, TransitionKind.DEMOTED)[0].reason is DemotionReason.DOWNLOAD_FAILED

        # The failed model is cooled: it does not immediately seat even when it is a strong on-disk candidate.
        next_transitions = pool.tick(
            21.0 * _MINUTE,
            ranked=[_cand("incumbent", 10.0), _cand("challenger", 100.0)],
            demand_is_stale=False,
        )
        assert "challenger" not in pool.active_seat_models()
        assert _of_kind(next_transitions, TransitionKind.SEATED) == []


class TestReplaceParams:
    """A configuration change reconciles seats, demoting removed pins without benching them."""

    def test_removed_pin_is_demoted_without_bench(self) -> None:
        """Dropping a pin from the config frees its seat and leaves the model free to seat again."""
        pool = ModelPool(_params(2, pinned=(PinnedModel("keep", 1.0), PinnedModel("drop", 1.0)), ranker_enabled=False))
        pool.tick(0.0, ranked=None, demand_is_stale=False)
        assert pool.active_seat_models() == frozenset({"keep", "drop"})

        transitions = pool.replace_params(
            _params(2, pinned=(PinnedModel("keep", 1.0),), ranker_enabled=False),
            now=1.0,
        )

        assert pool.active_seat_models() == frozenset({"keep"})
        demotion = _of_kind(transitions, TransitionKind.DEMOTED)[0]
        assert demotion.model == "drop"
        assert demotion.reason is DemotionReason.CONFIG_CHANGED
        assert "drop" not in {entry.model for entry in pool.bench()}

    def test_shrinking_seat_count_releases_least_committed_first(self) -> None:
        """Reducing the seat count removes a ranker seat before a manual pin."""
        pool = ModelPool(_params(2, pinned=(PinnedModel("pinned", 1.0),), ranker_enabled=True))
        pool.tick(0.0, ranked=[_cand("ranked", 5.0)], demand_is_stale=False)
        assert pool.active_seat_models() == frozenset({"pinned", "ranked"})

        pool.replace_params(_params(1, pinned=(PinnedModel("pinned", 1.0),), ranker_enabled=True), now=1.0)

        assert len(pool.seats()) == 1
        assert pool.active_seat_models() == frozenset({"pinned"})

    def test_growing_seat_count_adds_empty_seats(self) -> None:
        """Increasing the seat count adds empty seats without disturbing the occupied one."""
        pool = ModelPool(_params(1, pinned=(PinnedModel("pinned", 1.0),), ranker_enabled=False))
        pool.tick(0.0, ranked=None, demand_is_stale=False)

        pool.replace_params(_params(3, pinned=(PinnedModel("pinned", 1.0),), ranker_enabled=False), now=1.0)

        seats = pool.seats()
        assert len(seats) == 3
        assert seats[0].model == "pinned"
        assert sum(1 for seat in seats if seat.model is None) == 2


class TestPressureNote:
    """An external pressure eviction is recorded but never unseats the model."""

    def test_pressure_note_keeps_the_seat(self) -> None:
        """A pressure eviction of a seated model surfaces a note and leaves the seat serving."""
        pool = ModelPool(_params(1, pinned=(PinnedModel("m", 1.0),), ranker_enabled=False))
        pool.tick(0.0, ranked=None, demand_is_stale=False)

        transitions = pool.on_pressure_eviction("m", now=1.0)

        assert pool.active_seat_models() == frozenset({"m"})
        assert _kinds(transitions) == [TransitionKind.DEMOTED]
        assert transitions[0].reason is DemotionReason.PRESSURE_NOTED


class TestRescueSingleSeat:
    """At most one rescue seat may hold at a time, no matter how many models are starving."""

    def test_second_starving_model_cannot_claim_another_seat(self) -> None:
        """While one rescue holds, a new starving candidate does not displace a second ranker seat."""
        pool = ModelPool(
            PoolParams(
                seat_count=3,
                ranker_enabled=True,
                rescue_enabled=True,
                rescue_eta_seconds=10000.0,
                rescue_window_minutes=15.0,
                min_dwell_minutes=0.0,
                rotation_minutes=100000.0,
                zero_fulfillment_demotion_minutes=100000.0,
            ),
        )
        baseline = [_cand("strong", 10.0), _cand("mid", 9.0), _cand("weak", 8.0)]
        pool.tick(0.0, ranked=baseline, demand_is_stale=False)
        pool.tick(1.0, ranked=[*baseline, _cand("starved_one", 0.5, eta=20000.0)], demand_is_stale=False)
        assert sum(1 for seat in pool.seats() if seat.source is SeatSource.RESCUE) == 1

        transitions = pool.tick(
            2.0,
            ranked=[*baseline, _cand("starved_one", 0.5, eta=20000.0), _cand("starved_two", 0.4, eta=30000.0)],
            demand_is_stale=False,
        )

        assert _of_kind(transitions, TransitionKind.RESCUE_ENGAGED) == []
        assert sum(1 for seat in pool.seats() if seat.source is SeatSource.RESCUE) == 1


class TestPinIdentityPreservation:
    """A pinned model keeps its manual source and affinity regardless of which path seats it."""

    def test_pin_seated_through_recontest_keeps_manual_identity(self) -> None:
        """A benched pin that wins a timer re-contest as the challenger seats as MANUAL with its affinity."""
        pool = ModelPool(
            PoolParams(
                seat_count=1,
                pinned=(PinnedModel("pinned", 1.0),),
                ranker_enabled=True,
                min_dwell_minutes=0.0,
                rotation_minutes=0.01,
                empty_pop_demotion_threshold=1,
                bench_cooldown_empty_pops_minutes=1.0,
                zero_fulfillment_demotion_minutes=100000.0,
            ),
        )
        pool.tick(0.0, ranked=[_cand("other", 5.0)], demand_is_stale=False)
        assert pool.active_seat_models() == frozenset({"pinned"})

        pool.on_pop_outcome(lane=PopLane.FIXED, advertised=frozenset({"pinned"}), popped_model=None, now=1.0)
        pool.tick(
            2.0,
            ranked=[_cand("pinned", 1.0, queued_per_worker=0.0), _cand("other", 5.0)],
            demand_is_stale=False,
        )
        assert pool.active_seat_models() == frozenset({"other"})

        transitions = pool.tick(
            2.0 + 6.0 * _MINUTE,
            ranked=[_cand("pinned", 100.0), _cand("other", 5.0)],
            demand_is_stale=False,
        )

        seated = _of_kind(transitions, TransitionKind.SEATED)
        assert [transition.model for transition in seated] == ["pinned"]
        assert seated[0].source is SeatSource.MANUAL
        seat = pool.seats()[0]
        assert seat.model == "pinned"
        assert seat.source is SeatSource.MANUAL


class TestOffDiskRescue:
    """An off-disk rescue pick marks the weakest ranker seat pending, keeping its incumbent until the file lands."""

    def _off_disk_params(self, seat_count: int = 2, **overrides: object) -> PoolParams:
        return _params(
            seat_count,
            ranker_enabled=True,
            rescue_enabled=True,
            rescue_eta_seconds=10000.0,
            rescue_window_minutes=15.0,
            rescue_model_cooldown_hours=6.0,
            min_dwell_minutes=0.0,
            rotation_minutes=100000.0,
            zero_fulfillment_demotion_minutes=100000.0,
            **overrides,
        )

    def test_off_disk_rescue_marks_pending_and_retains_incumbent(self) -> None:
        """A starved off-disk pick marks the weakest ranker seat pending while its incumbent keeps serving."""
        pool = ModelPool(self._off_disk_params())
        pool.tick(0.0, ranked=[_cand("strong", 10.0), _cand("weak", 8.0)], demand_is_stale=False)

        transitions = pool.tick(
            1.0,
            ranked=[_cand("strong", 10.0), _cand("weak", 8.0), _cand("starved", 0.5, on_disk=False, eta=20000.0)],
            demand_is_stale=False,
        )

        weak_seat = next(seat for seat in pool.seats() if seat.model == "weak")
        assert weak_seat.state is SeatState.PENDING_DOWNLOAD
        assert weak_seat.pending_model == "starved"
        assert pool.active_seat_models() == frozenset({"strong", "weak"})
        assert pool.pending_download_models() == frozenset({"starved"})
        assert _of_kind(transitions, TransitionKind.DOWNLOAD_PENDING)[0].model == "starved"
        assert _of_kind(transitions, TransitionKind.RESCUE_ENGAGED) == []
        assert _of_kind(transitions, TransitionKind.DEMOTED) == []

    def test_off_disk_rescue_ready_activates_as_rescue_with_window(self) -> None:
        """When the rescue download lands the model seats as RESCUE with its window started at activation."""
        pool = ModelPool(self._off_disk_params())
        pool.tick(0.0, ranked=[_cand("strong", 10.0), _cand("weak", 8.0)], demand_is_stale=False)
        pool.tick(
            1.0,
            ranked=[_cand("strong", 10.0), _cand("weak", 8.0), _cand("starved", 0.5, on_disk=False, eta=20000.0)],
            demand_is_stale=False,
        )

        transitions = pool.on_download_ready("starved", now=2.0)

        rescue_seat = next(seat for seat in pool.seats() if seat.model == "starved")
        assert rescue_seat.source is SeatSource.RESCUE
        assert rescue_seat.state is SeatState.ACTIVE
        assert rescue_seat.rescue_expires_at == 2.0 + 15.0 * _MINUTE
        assert pool.active_seat_models() == frozenset({"strong", "starved"})
        assert _of_kind(transitions, TransitionKind.DOWNLOAD_READY)[0].model == "starved"
        assert "weak" in {entry.model for entry in pool.bench()}

    def test_pending_off_disk_rescue_blocks_a_second_rescue(self) -> None:
        """While an off-disk rescue is still downloading, a second starving candidate cannot engage a rescue."""
        pool = ModelPool(self._off_disk_params(seat_count=3))
        baseline = [_cand("strong", 10.0), _cand("mid", 9.0), _cand("weak", 8.0)]
        pool.tick(0.0, ranked=baseline, demand_is_stale=False)
        pool.tick(
            1.0,
            ranked=[*baseline, _cand("starved_one", 0.5, on_disk=False, eta=20000.0)],
            demand_is_stale=False,
        )
        assert pool.pending_download_models() == frozenset({"starved_one"})

        transitions = pool.tick(
            2.0,
            ranked=[
                *baseline,
                _cand("starved_one", 0.5, on_disk=False, eta=20000.0),
                _cand("starved_two", 0.4, on_disk=False, eta=30000.0),
            ],
            demand_is_stale=False,
        )

        assert _of_kind(transitions, TransitionKind.DOWNLOAD_PENDING) == []
        assert pool.pending_download_models() == frozenset({"starved_one"})


class TestNonProductiveDemotionReasons:
    """The empty-pop and zero-fulfillment demotion triggers surface as distinct reasons for the caller."""

    def test_empty_pop_and_zero_fulfillment_triggers_differ(self) -> None:
        """The empty-pop path reports EMPTY_POPS while the zero-fulfillment path reports ZERO_FULFILLMENT."""
        empty_pop_pool = ModelPool(
            _params(
                1,
                pinned=(PinnedModel("m", 1.0),),
                ranker_enabled=False,
                min_dwell_minutes=0.0,
                zero_fulfillment_demotion_minutes=100000.0,
            ),
        )
        empty_pop_pool.tick(0.0, ranked=None, demand_is_stale=False)
        for _ in range(40):
            empty_pop_pool.on_pop_outcome(lane=PopLane.FIXED, advertised=frozenset({"m"}), popped_model=None, now=1.0)
        empty_pop_transitions = empty_pop_pool.tick(
            2.0, ranked=[_cand("m", 1.0, queued_per_worker=0.0)], demand_is_stale=False
        )
        assert _of_kind(empty_pop_transitions, TransitionKind.DEMOTED)[0].reason is DemotionReason.EMPTY_POPS

        zero_fulfillment_pool = ModelPool(
            _params(
                1,
                pinned=(PinnedModel("m", 1.0),),
                ranker_enabled=False,
                min_dwell_minutes=0.0,
                zero_fulfillment_demotion_minutes=15.0,
            ),
        )
        zero_fulfillment_pool.tick(0.0, ranked=None, demand_is_stale=False)
        zero_fulfillment_transitions = zero_fulfillment_pool.tick(16.0 * _MINUTE, ranked=None, demand_is_stale=False)
        assert (
            _of_kind(zero_fulfillment_transitions, TransitionKind.DEMOTED)[0].reason is DemotionReason.ZERO_FULFILLMENT
        )


class TestReplaceParamsPromotion:
    """A newly pinned model already holding a ranker seat converts to manual in place on a config change."""

    def test_replace_params_promotes_ranker_seat_of_newly_pinned_model(self) -> None:
        """Pinning a model that already holds a ranker seat converts the seat to manual in place."""
        pool = ModelPool(PoolParams(seat_count=1, ranker_enabled=True, min_dwell_minutes=0.0))
        pool.tick(0.0, ranked=[_cand("promoted", 5.0)], demand_is_stale=False)
        assert pool.seats()[0].source is SeatSource.RANKER

        transitions = pool.replace_params(
            PoolParams(seat_count=1, pinned=(PinnedModel("promoted", 0.8),), ranker_enabled=True), now=1.0
        )

        assert transitions == []
        seat = pool.seats()[0]
        assert seat.model == "promoted"
        assert seat.source is SeatSource.MANUAL
