"""Table-driven tests for duty-cycled residency-biased pop advertising."""

from __future__ import annotations

from horde_worker_regen.process_management.scheduling.pop_affinity import (
    ResidencyBiasState,
    decide_residency_advertising,
)


def _run(
    state: ResidencyBiasState,
    *,
    swap_backlog: bool,
    resident: set[str] | None = None,
    staged: set[str] | None = None,
    offered: set[str] | None = None,
    narrow_cycles: int = 6,
    open_cycles: int = 2,
) -> tuple[frozenset[str], bool, bool, ResidencyBiasState]:
    """Run one decision with compact defaults, returning the decision's fields as a tuple."""
    decision = decide_residency_advertising(
        state,
        swap_backlog=swap_backlog,
        resident_models=resident if resident is not None else {"A"},
        staged_models=staged if staged is not None else set(),
        offered_models=offered if offered is not None else {"A", "B", "C"},
        narrow_cycles=narrow_cycles,
        open_cycles=open_cycles,
    )
    return (
        decision.advertised_models,
        decision.narrowing,
        decision.narrowed_offer,
        decision.next_state,
    )


class TestEngage:
    """A swap backlog engages the narrow phase on its first cycle and narrows toward residents."""

    def test_no_backlog_advertises_full_and_stays_idle(self) -> None:
        """With no swap backlog the full offered set is advertised and the state stays idle."""
        advertised, narrowing, narrowed, next_state = _run(ResidencyBiasState(), swap_backlog=False)
        assert advertised == frozenset({"A", "B", "C"})
        assert not narrowing
        assert not narrowed
        assert next_state == ResidencyBiasState()

    def test_first_backlog_cycle_narrows_immediately(self) -> None:
        """The first backlogged cycle engages the narrow phase (does not wait out an open phase first)."""
        advertised, narrowing, narrowed, next_state = _run(
            ResidencyBiasState(),
            swap_backlog=True,
            resident={"A"},
            offered={"A", "B", "C"},
        )
        assert advertised == frozenset({"A"})
        assert narrowing
        assert narrowed
        assert next_state.active
        assert next_state.narrowing
        assert next_state.cycles_in_phase == 1

    def test_staged_models_widen_the_floor(self) -> None:
        """Staged (RAM) models join residents in the narrowed floor."""
        advertised, _narrowing, narrowed, _next = _run(
            ResidencyBiasState(),
            swap_backlog=True,
            resident={"A"},
            staged={"B"},
            offered={"A", "B", "C"},
        )
        assert advertised == frozenset({"A", "B"})
        assert narrowed


class TestDutyCycleAlternation:
    """Under a persistent backlog the offer alternates narrow (N) and open (M) phases, bounded."""

    def test_alternates_six_narrow_then_two_open(self) -> None:
        """A persistent backlog yields exactly six narrowed cycles, then two full-set cycles, repeating."""
        state = ResidencyBiasState()
        narrowing_sequence: list[bool] = []
        for _ in range(16):
            advertised, narrowing, _narrowed, state = _run(
                state,
                swap_backlog=True,
                resident={"A"},
                offered={"A", "B", "C"},
            )
            narrowing_sequence.append(narrowing)
            # The advertised set is never empty regardless of phase.
            assert advertised

        expected = ([True] * 6 + [False] * 2) * 2
        assert narrowing_sequence == expected

    def test_open_phase_advertises_full_set(self) -> None:
        """During the open phase of a persistent backlog the full offered set is re-advertised."""
        state = ResidencyBiasState()
        # Advance through the six narrow cycles.
        for _ in range(6):
            _adv, narrowing, _narrowed, state = _run(state, swap_backlog=True)
            assert narrowing
        # The seventh cycle is the first open cycle.
        advertised, narrowing, narrowed, state = _run(state, swap_backlog=True)
        assert not narrowing
        assert not narrowed
        assert advertised == frozenset({"A", "B", "C"})

    def test_narrow_window_is_bounded_even_if_backlog_never_clears(self) -> None:
        """No narrow window exceeds N consecutive narrowed cycles under an unbroken backlog."""
        state = ResidencyBiasState()
        max_consecutive = 0
        run_length = 0
        for _ in range(100):
            _adv, narrowing, _narrowed, state = _run(state, swap_backlog=True)
            run_length = run_length + 1 if narrowing else 0
            max_consecutive = max(max_consecutive, run_length)
        assert max_consecutive == 6


class TestFloorRail:
    """The narrowed offer is floored so it can never empty and never expands past the offer."""

    def test_no_resident_or_staged_in_offer_falls_back_to_full(self) -> None:
        """When residents/staged intersect the offer to nothing, the full offered set is advertised."""
        advertised, narrowing, narrowed, _next = _run(
            ResidencyBiasState(),
            swap_backlog=True,
            resident={"Z"},  # not in the offer
            staged=set(),
            offered={"A", "B", "C"},
        )
        assert narrowing
        assert not narrowed  # the offer was not actually reduced
        assert advertised == frozenset({"A", "B", "C"})

    def test_narrowing_never_adds_a_model_not_offered(self) -> None:
        """A resident model not in the offered set is never advertised."""
        advertised, _narrowing, _narrowed, _next = _run(
            ResidencyBiasState(),
            swap_backlog=True,
            resident={"A", "Z"},  # Z resident but not offered
            offered={"A", "B", "C"},
        )
        assert "Z" not in advertised
        assert advertised == frozenset({"A"})

    def test_single_offered_resident_model_narrows_to_itself_not_empty(self) -> None:
        """A one-model offer that is resident narrows to that model, never to the empty set."""
        advertised, _narrowing, _narrowed, _next = _run(
            ResidencyBiasState(),
            swap_backlog=True,
            resident={"A"},
            offered={"A"},
        )
        assert advertised == frozenset({"A"})


class TestReleaseOnBacklogClearing:
    """The duty cycle resets to idle the moment the backlog clears, re-engaging fresh next time."""

    def test_backlog_clearing_resets_to_idle(self) -> None:
        """A cleared backlog mid-narrow-phase advertises the full set and resets the state to idle."""
        state = ResidencyBiasState()
        # Engage and advance a couple of narrow cycles.
        for _ in range(3):
            _adv, narrowing, _narrowed, state = _run(state, swap_backlog=True)
            assert narrowing
        advertised, narrowing, narrowed, state = _run(state, swap_backlog=False)
        assert not narrowing
        assert not narrowed
        assert advertised == frozenset({"A", "B", "C"})
        assert state == ResidencyBiasState()

    def test_reengage_after_clear_starts_in_narrow_phase(self) -> None:
        """After clearing, the next backlog engages a fresh narrow phase from cycle one."""
        state = ResidencyBiasState()
        for _ in range(4):
            _adv, _narrowing, _narrowed, state = _run(state, swap_backlog=True)
        _adv, _narrowing, _narrowed, state = _run(state, swap_backlog=False)
        assert state == ResidencyBiasState()
        _adv, narrowing, narrowed, state = _run(state, swap_backlog=True)
        assert narrowing
        assert narrowed
        assert state.cycles_in_phase == 1


class TestOffSwitch:
    """A non-positive narrow length disables narrowing entirely."""

    def test_zero_narrow_cycles_never_narrows(self) -> None:
        """With ``narrow_cycles`` <= 0 the offer is never narrowed even under a persistent backlog."""
        state = ResidencyBiasState()
        for _ in range(10):
            advertised, narrowing, narrowed, state = _run(
                state,
                swap_backlog=True,
                narrow_cycles=0,
            )
            assert not narrowing
            assert not narrowed
            assert advertised == frozenset({"A", "B", "C"})
            assert state == ResidencyBiasState()
