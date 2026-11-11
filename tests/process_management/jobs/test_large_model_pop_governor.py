"""Tests for the large-model pop limiters: the switch throttle, the re-entry cooldown and the paced return.

The governor is pure and time-injectable, so each scenario drives ``evaluate`` across a sequence of pop
cycles with explicit timestamps and asserts which large models it withholds from the offer.
"""

from __future__ import annotations

from horde_worker_regen.process_management.jobs.large_model_pop_governor import (
    LargeModelPopGovernor,
    LowEarningPacingDecision,
    PacedReturn,
    PacedReturnCause,
)

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


class TestDescribe:
    """The read-only ``describe`` reports window engagement without perturbing the throttle timers."""

    def test_switch_window_reports_active_and_remaining(self) -> None:
        """With a large model in play and the interval not elapsed, the switch window reads active."""
        governor = LargeModelPopGovernor()
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, switch=30.0)
        status = governor.describe(
            incumbent_large_models=frozenset({_FLUX}),
            residency_active=False,
            now=10.0,
            switch_min_seconds=30.0,
            reentry_cooldown_seconds=0.0,
        )
        assert status.switch_active is True
        assert status.switch_remaining_seconds == 20.0
        assert status.reentry_active is False

    def test_describe_does_not_mutate_timers(self) -> None:
        """Calling describe must not advance the switch anchor (a status poll is side-effect free)."""
        governor = LargeModelPopGovernor()
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, switch=30.0)
        for _ in range(3):
            governor.describe(
                incumbent_large_models=frozenset({_FLUX}),
                residency_active=False,
                now=10.0,
                switch_min_seconds=30.0,
                reentry_cooldown_seconds=0.0,
            )
        # The throttle still withholds a different large model exactly as if describe had never been called.
        assert _evaluate(governor, candidate={_FLUX, _ZIMAGE}, incumbent={_FLUX}, now=10.0, switch=30.0) == frozenset(
            {_ZIMAGE},
        )

    def test_reentry_window_reports_active_after_drain(self) -> None:
        """Once the re-entry window has opened, describe reports it active with the remaining cooldown."""
        governor = LargeModelPopGovernor()
        _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, reentry=45.0)
        _evaluate(governor, candidate={_FLUX}, incumbent=set(), now=10.0, reentry=45.0)  # window opens at t=10
        status = governor.describe(
            incumbent_large_models=frozenset(),
            residency_active=False,
            now=20.0,
            switch_min_seconds=0.0,
            reentry_cooldown_seconds=45.0,
        )
        assert status.reentry_active is True
        assert status.reentry_remaining_seconds == 35.0


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


_RETURN_SECONDS = 180.0


def _pace(
    governor: LargeModelPopGovernor,
    *,
    now: float,
    withheld: str | None = None,
    with_work: set[str] | None = None,
    exempt: set[str] | None = None,
    return_seconds: float = _RETURN_SECONDS,
    idle_escape: bool = False,
) -> LowEarningPacingDecision:
    """Drive one pacing evaluation and return the decision."""
    return governor.evaluate_pacing(
        withheld_claim_model=withheld,
        models_with_work=frozenset(with_work or set()),
        exempt_models=frozenset(exempt or set()),
        now=now,
        return_seconds=return_seconds,
        idle_escape=idle_escape,
    )


class TestPacedReturn:
    """The paced return of a model whose claim is withheld."""

    def test_a_withheld_model_leaves_the_offer_while_its_jobs_finish(self) -> None:
        """The model leaves when its claim is withheld, and stays out while a job for it is in hand."""
        governor = LargeModelPopGovernor()
        decision = _pace(governor, now=0.0, withheld=_FLUX, with_work={_FLUX})
        assert decision.withheld == frozenset({_FLUX})
        assert decision.started == (_FLUX,)
        later = _pace(governor, now=1000.0, with_work={_FLUX})
        assert later.withheld == frozenset({_FLUX})
        assert later.started == ()

    def test_the_model_returns_the_interval_after_its_last_job_finished(self) -> None:
        """The interval is measured from the cycle in which the model's last job finished."""
        governor = LargeModelPopGovernor()
        _pace(governor, now=0.0, withheld=_FLUX, with_work={_FLUX})
        governor.note_paced_model_work(models_with_work=frozenset(), now=50.0)
        assert _pace(governor, now=50.0 + _RETURN_SECONDS - 1.0).withheld == frozenset({_FLUX})
        decision = _pace(governor, now=50.0 + _RETURN_SECONDS)
        assert decision.withheld == frozenset()
        assert decision.returned == (PacedReturn(model=_FLUX, cause=PacedReturnCause.INTERVAL_PASSED),)
        assert governor.paced_models == frozenset()

    def test_the_model_stays_out_while_its_claim_is_still_withheld(self) -> None:
        """A model whose claim is still withheld does not return when the interval has passed."""
        governor = LargeModelPopGovernor()
        _pace(governor, now=0.0, withheld=_FLUX)
        assert _pace(governor, now=_RETURN_SECONDS + 10.0, withheld=_FLUX).withheld == frozenset({_FLUX})
        assert _pace(governor, now=_RETURN_SECONDS + 20.0).withheld == frozenset()

    def test_a_job_for_the_model_restarts_its_interval(self) -> None:
        """A job for the paced model in hand clears its finish time, so the interval runs from its end."""
        governor = LargeModelPopGovernor()
        _pace(governor, now=0.0, withheld=_FLUX)
        _pace(governor, now=100.0, with_work={_FLUX})
        governor.note_paced_model_work(models_with_work=frozenset(), now=101.0)
        assert _pace(governor, now=_RETURN_SECONDS + 1.0).withheld == frozenset({_FLUX})
        assert _pace(governor, now=_RETURN_SECONDS + 100.0).withheld == frozenset({_FLUX})
        assert _pace(governor, now=_RETURN_SECONDS + 101.0).withheld == frozenset()

    def test_an_empty_answer_returns_the_model_until_another_model_is_popped(self) -> None:
        """An empty answer to an offer that left the model out returns it, and only another model's pop ends that."""
        governor = LargeModelPopGovernor()
        _pace(governor, now=0.0, withheld=_FLUX, with_work={_FLUX})
        returned = governor.note_pop_outcome(popped_model=None)
        assert returned == (PacedReturn(model=_FLUX, cause=PacedReturnCause.HORDE_HAD_NO_OTHER_WORK),)
        assert _pace(governor, now=10.0, withheld=_FLUX, with_work={_FLUX}).withheld == frozenset()
        assert governor.note_pop_outcome(popped_model=_FLUX) == ()
        assert _pace(governor, now=20.0, withheld=_FLUX, with_work={_FLUX}).withheld == frozenset()
        governor.note_pop_outcome(popped_model=_ZIMAGE)
        decision = _pace(governor, now=30.0, withheld=_FLUX, with_work={_FLUX})
        assert decision.withheld == frozenset({_FLUX})
        assert decision.started == (_FLUX,)

    def test_an_empty_answer_to_an_offer_with_the_model_returns_nothing(self) -> None:
        """An empty answer is evidence only for a model the last offer left out."""
        governor = LargeModelPopGovernor()
        _pace(governor, now=0.0)
        assert governor.note_pop_outcome(popped_model=None) == ()

    def test_the_idle_escape_returns_the_model_and_starts_no_window(self) -> None:
        """A worker with no work asks for every paced model, and a withheld claim does not start a window then."""
        governor = LargeModelPopGovernor()
        _pace(governor, now=0.0, withheld=_FLUX, with_work={_FLUX})
        decision = _pace(governor, now=10.0, withheld=_FLUX, idle_escape=True)
        assert decision.withheld == frozenset()
        assert decision.returned == (PacedReturn(model=_FLUX, cause=PacedReturnCause.WORKER_IDLE),)
        assert _pace(governor, now=20.0, withheld=_FLUX, idle_escape=True).started == ()

    def test_an_exemption_returns_the_model(self) -> None:
        """A paced model that the policy or an exemption now holds the card for is asked for again at once."""
        governor = LargeModelPopGovernor()
        _pace(governor, now=0.0, withheld=_FLUX, with_work={_FLUX})
        decision = _pace(governor, now=10.0, with_work={_FLUX}, exempt={_FLUX})
        assert decision.withheld == frozenset()
        assert decision.returned == (PacedReturn(model=_FLUX, cause=PacedReturnCause.NO_LONGER_PACED),)

    def test_a_model_without_a_withheld_claim_is_not_paced(self) -> None:
        """Without a withheld claim the window does not open, whatever the model's work."""
        governor = LargeModelPopGovernor()
        assert _pace(governor, now=0.0, with_work={_FLUX}).withheld == frozenset()
        assert governor.paced_models == frozenset()

    def test_pacing_leaves_the_switch_throttle_and_the_cooldown_unchanged(self) -> None:
        """The paced window does not change what the switch throttle and re-entry cooldown withhold."""
        paced = LargeModelPopGovernor()
        plain = LargeModelPopGovernor()
        _pace(paced, now=0.0, withheld=_FLUX, with_work={_FLUX})
        for governor in (paced, plain):
            _evaluate(governor, candidate={_FLUX}, incumbent={_FLUX}, now=0.0, switch=30.0, reentry=45.0)
        for governor in (paced, plain):
            assert _evaluate(
                governor, candidate={_FLUX, _ZIMAGE}, incumbent={_FLUX}, now=10.0, switch=30.0, reentry=45.0
            ) == frozenset({_ZIMAGE})
            _evaluate(governor, candidate={_FLUX}, incumbent=set(), now=20.0, switch=30.0, reentry=45.0)
            assert _evaluate(
                governor, candidate={_FLUX}, incumbent=set(), now=30.0, switch=30.0, reentry=45.0
            ) == frozenset({_FLUX})

    def test_describe_reports_the_return_once_the_jobs_finish(self) -> None:
        """The status has no return time while a job is in hand, and the interval after the last finish."""
        governor = LargeModelPopGovernor()
        _pace(governor, now=5.0, withheld=_FLUX, with_work={_FLUX})
        (status,) = governor.describe_pacing(return_seconds=_RETURN_SECONDS)
        assert status.model == _FLUX
        assert status.left_at == 5.0
        assert status.returns_at is None
        governor.note_paced_model_work(models_with_work=frozenset(), now=40.0)
        (status,) = governor.describe_pacing(return_seconds=_RETURN_SECONDS)
        assert status.returns_at == 40.0 + _RETURN_SECONDS
