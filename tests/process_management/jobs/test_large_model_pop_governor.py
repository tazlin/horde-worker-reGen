"""Tests for the large-model pop limiters: the switch throttle and the re-entry cooldown.

The governor is pure and time-injectable, so each scenario drives ``evaluate`` across a sequence of pop
cycles with explicit timestamps and asserts which large models it withholds from the offer.
"""

from __future__ import annotations

from horde_worker_regen.process_management.jobs.large_model_pop_governor import LargeModelPopGovernor

_FLUX = "Flux.1-Schnell fp8 (Compact)"
_ZIMAGE = "Z-Image-Turbo"
_CASCADE = "Stable Cascade 1.0"


def _evaluate(
    governor: LargeModelPopGovernor,
    *,
    candidate: set[str],
    incumbent: set[str],
    now: float,
    switch: float = 0.0,
    reentry: float = 0.0,
    residency_active: bool = False,
    idle_escape: bool = False,
) -> frozenset[str]:
    """Drive one governor cycle and return the withheld set."""
    decision = governor.evaluate(
        candidate_large_models=frozenset(candidate),
        incumbent_large_models=frozenset(incumbent),
        residency_active=residency_active,
        now=now,
        switch_min_seconds=switch,
        reentry_cooldown_seconds=reentry,
        idle_escape=idle_escape,
    )
    return decision.withheld


class TestDisabled:
    """Zero durations disable the limiter entirely."""

    def test_nothing_withheld_when_both_durations_zero(self) -> None:
        """With both durations 0, even a different large model alongside an incumbent is offered."""
        governor = LargeModelPopGovernor()
        withheld = _evaluate(
            governor,
            candidate={_FLUX, _ZIMAGE},
            incumbent={_FLUX},
            now=10.0,
            switch=0.0,
            reentry=0.0,
        )
        assert withheld == frozenset()


class TestSwitchThrottle:
    """Once a large model is in play, a *different* large model is withheld for the switch interval."""

    def test_different_large_model_withheld_within_interval(self) -> None:
        """A different large model offered within the interval is withheld; the incumbent is not."""
        governor = LargeModelPopGovernor()
        # Introduce Flux at t=0 (stamps the switch anchor).
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, switch=30.0)
        # 10s later a different large model is offered: it must be withheld, Flux must not.
        withheld = _evaluate(governor, candidate={_FLUX, _ZIMAGE}, incumbent={_FLUX}, now=10.0, switch=30.0)
        assert withheld == frozenset({_ZIMAGE})

    def test_same_large_model_never_withheld(self) -> None:
        """The large model already in play stays offerable regardless of the switch interval."""
        governor = LargeModelPopGovernor()
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, switch=30.0)
        withheld = _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=5.0, switch=30.0)
        assert withheld == frozenset()

    def test_different_large_model_allowed_after_interval(self) -> None:
        """Once the interval elapses, the different large model is offered again."""
        governor = LargeModelPopGovernor()
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, switch=30.0)
        withheld = _evaluate(governor, candidate={_FLUX, _ZIMAGE}, incumbent={_FLUX}, now=31.0, switch=30.0)
        assert withheld == frozenset()

    def test_first_large_model_not_throttled(self) -> None:
        """With no large model in play, the first one is not a 'switch' and must be offered."""
        governor = LargeModelPopGovernor()
        withheld = _evaluate(governor, candidate={_FLUX}, incumbent=set(), now=0.0, switch=30.0)
        assert withheld == frozenset()

    def test_introducing_a_new_distinct_large_resets_the_clock(self) -> None:
        """Each distinct introduction re-anchors the interval, so churn stays throttled."""
        governor = LargeModelPopGovernor()
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, switch=30.0)
        # Z-Image becomes resident at t=31 (allowed); the anchor moves to t=31.
        _evaluate(governor, candidate={_FLUX, _ZIMAGE}, incumbent={_FLUX, _ZIMAGE}, now=31.0, switch=30.0)
        # At t=40 a third distinct large is still within 30s of the t=31 introduction: withheld.
        withheld = _evaluate(
            governor,
            candidate={_FLUX, _ZIMAGE, _CASCADE},
            incumbent={_FLUX, _ZIMAGE},
            now=40.0,
            switch=30.0,
        )
        assert withheld == frozenset({_CASCADE})


class TestReentryCooldown:
    """After the lease is up and all large models drain, any large model is withheld for the cooldown."""

    def test_all_large_withheld_after_drain_once_lease_up(self) -> None:
        """The window opens only once the lease clears; then every large model is withheld."""
        governor = LargeModelPopGovernor()
        # Flux in play, lease held.
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, reentry=45.0, residency_active=True)
        # Flux drains but the lease is still held: the re-entry window has not opened, so it is offerable.
        withheld_during_lease = _evaluate(
            governor,
            candidate={_FLUX},
            incumbent=set(),
            now=10.0,
            reentry=45.0,
            residency_active=True,
        )
        assert withheld_during_lease == frozenset()
        # Lease up at t=20: the window opens and every large model is withheld.
        withheld = _evaluate(governor, candidate={_FLUX, _ZIMAGE}, incumbent=set(), now=20.0, reentry=45.0)
        assert withheld == frozenset({_FLUX, _ZIMAGE})

    def test_large_offered_again_after_cooldown(self) -> None:
        """Large models are withheld through the cooldown window and offered once it elapses."""
        governor = LargeModelPopGovernor()
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, reentry=45.0)
        # Window opens at t=10 (no incumbent, no lease).
        assert _evaluate(governor, candidate={_FLUX}, incumbent=set(), now=10.0, reentry=45.0) == frozenset({_FLUX})
        # Still within the cooldown at t=50 (10 + 45 = 55).
        assert _evaluate(governor, candidate={_FLUX}, incumbent=set(), now=50.0, reentry=45.0) == frozenset({_FLUX})
        # Past the cooldown at t=56.
        assert _evaluate(governor, candidate={_FLUX}, incumbent=set(), now=56.0, reentry=45.0) == frozenset()

    def test_no_cooldown_when_no_large_ever_ran(self) -> None:
        """A worker that has never held a large model must not start a cooldown out of nowhere."""
        governor = LargeModelPopGovernor()
        withheld = _evaluate(governor, candidate={_FLUX}, incumbent=set(), now=0.0, reentry=45.0)
        assert withheld == frozenset()

    def test_new_large_in_play_clears_the_cooldown(self) -> None:
        """A large model coming back into play cancels an in-progress cooldown."""
        governor = LargeModelPopGovernor()
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, reentry=45.0)
        assert _evaluate(governor, candidate={_FLUX}, incumbent=set(), now=10.0, reentry=45.0) == frozenset({_FLUX})
        # A large model comes back into play (e.g. via the idle escape elsewhere): the cooldown is cleared.
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=15.0, reentry=45.0)
        assert _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=16.0, reentry=45.0) == frozenset()


class TestIdleEscape:
    """A fully idle worker (empty local queue) is never left withholding its only available work."""

    def test_idle_escape_bypasses_switch_throttle(self) -> None:
        """An idle worker offers a different large model even within the switch interval."""
        governor = LargeModelPopGovernor()
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, switch=30.0)
        withheld = _evaluate(
            governor,
            candidate={_FLUX, _ZIMAGE},
            incumbent={_FLUX},
            now=10.0,
            switch=30.0,
            idle_escape=True,
        )
        assert withheld == frozenset()

    def test_idle_escape_bypasses_reentry_cooldown(self) -> None:
        """An idle worker offers large models even within the re-entry cooldown."""
        governor = LargeModelPopGovernor()
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, reentry=45.0)
        withheld = _evaluate(
            governor,
            candidate={_FLUX},
            incumbent=set(),
            now=10.0,
            reentry=45.0,
            idle_escape=True,
        )
        assert withheld == frozenset()
