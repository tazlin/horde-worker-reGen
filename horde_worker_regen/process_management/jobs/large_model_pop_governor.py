"""Two pop-time limiters that tame the process/disk thrash of mixed very-large-model queues.

A queue that alternates distinct very-large models (Flux -> Z-Image -> Flux -> Flux -> Z-Image) is
pathological for a single GPU: each switch to a *different* large model forces the whole-card residency
machinery to tear the pool down, evict the resident model, and stream a fresh multi-GB checkpoint from disk,
so the worker spends most of its time loading rather than generating. These limiters act at the only place
the worker controls what work it takes: the set of models it *offers* in the horde pop request, so no job
is ever popped and then dropped (dropping is what trips the horde's "too many drops" maintenance).

Two independent mechanisms, each disabled by a zero duration:

* **Switch throttle** (``switch_min_seconds``): once a large model is loaded or queued, a *different* large
  model is withheld from the offer until this many seconds have elapsed since the last *distinct* large model
  was introduced. Jobs for the large model already in play stay offerable; only churning to a new one is
  throttled.
* **Re-entry cooldown** (``reentry_cooldown_seconds``): once the whole-card residency lease is up *and* no
  large model remains loaded or queued, *any* large model is withheld for this long, so the worker does
  ordinary work for a beat before it is allowed back into large-model territory rather than immediately
  re-thrashing.

Both yield to an idle escape: when the worker is genuinely idle with an empty local queue, nothing is
withheld, so a limiter never leaves the worker sitting idle when the only work it could take is a large model.

A third window paces a model that earns below the rest of its card (:meth:`LargeModelPopGovernor.evaluate_pacing`).
It keys on the model whose whole-card pop claim the scheduler withholds, whatever its size tier. The model leaves
the offer while its claim is withheld, so its jobs in hand finish and no more arrive. It stays out until
``whole_card_low_earning_return_seconds`` have passed since its last job finished. It returns early after the horde
answers an offer that left it out with no job, until a job for another model is popped, and when the worker holds
no work. Without the window, jobs for other models taken while the model holds the card would wait behind a steady
stream of its own jobs.

The governor is pure and time-injectable (no clock, no I/O): the caller passes the current large-model
incumbents, the residency-lease flag, and ``now`` each pop cycle, and the governor tracks the timing state
needed to decide what to withhold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class LargeModelPopDecision:
    """The outcome of one governor evaluation: which large models to withhold, and why (for logging)."""

    withheld: frozenset[str]
    """Large model names to drop from this pop's offered set. Empty when nothing is throttled."""
    reason: str | None
    """A short human-readable cause (``"switch throttle"`` / ``"re-entry cooldown"``), or None when nothing
    was withheld."""


@dataclass(frozen=True)
class LargeModelGovernorStatus:
    """A read-only view of the two large-model limiters' current engagement, for observability.

    Distinct from :class:`LargeModelPopDecision` (which models to withhold *this* pop): this reports whether
    each window is open right now and how much longer it is expected to last, so the dashboard and the
    governor registry can show the limiter as engaged independent of whether a candidate happens to be queued.
    """

    switch_active: bool
    switch_remaining_seconds: float | None
    switch_reason: str | None
    reentry_active: bool
    reentry_remaining_seconds: float | None
    reentry_reason: str | None


class PacedReturnCause(StrEnum):
    """Why a paced model is asked for again."""

    INTERVAL_PASSED = "interval_passed"
    """The interval has passed since the model's last job finished."""
    HORDE_HAD_NO_OTHER_WORK = "horde_had_no_other_work"
    """The horde answered an offer that left the model out with no job."""
    WORKER_IDLE = "worker_idle"
    """The worker holds no work at all."""
    NO_LONGER_PACED = "no_longer_paced"
    """The policy or an exemption now holds the card for the model."""


@dataclass(frozen=True)
class PacedReturn:
    """Represents a paced model that is asked for again, and why.

    Attributes:
        model: The model.
        cause: Why it is asked for again.
    """

    model: str
    cause: PacedReturnCause


@dataclass(frozen=True)
class LowEarningPacingDecision:
    """Represents one evaluation of the pacing window.

    Attributes:
        withheld: The models to leave out of this pop's offer.
        started: The models that left the offer in this evaluation.
        returned: The models asked for again in this evaluation, with the cause.
    """

    withheld: frozenset[str]
    started: tuple[str, ...]
    returned: tuple[PacedReturn, ...]


@dataclass(frozen=True)
class PacedModelStatus:
    """Represents a paced model as the governor registry and the residency status show it.

    Attributes:
        model: The paced model.
        left_at: When the model left the offer, on the caller's clock.
        returns_at: When the model is asked for again, or None while a job for it is pending or in progress.
    """

    model: str
    left_at: float
    returns_at: float | None


@dataclass
class _PacedModel:
    """Represents the timing of one paced model.

    Attributes:
        left_at: When the model left the offer.
        finished_at: When the model's last job in hand finished, or None while one is pending or in progress.
    """

    left_at: float
    finished_at: float | None


class LargeModelPopGovernor:
    """Stateful tracker for the large-model switch throttle and re-entry cooldown.

    One instance per worker (the pop path is worker-wide). :meth:`evaluate` is called once per pop cycle; it
    updates the internal timing state from the observed incumbents and returns the withholding decision.
    """

    def __init__(self) -> None:
        """Initialize with no large-model history."""
        self._known_large_incumbents: frozenset[str] = frozenset()
        # When the last *distinct* large model was first observed loaded/queued (the switch-throttle anchor).
        self._last_distinct_large_introduced_at: float = 0.0
        # Whether a large model has been in play since the re-entry window last opened, so a worker that never
        # ran a large model does not start a cooldown, and an expired cooldown does not re-open without new
        # large-model activity.
        self._had_large_since_reentry: bool = False
        # When the post-large window opened (no large model in play and no lease held), or None when not open.
        self._reentry_started_at: float | None = None
        # The models the pacing window keeps out of the offer, by name.
        self._paced: dict[str, _PacedModel] = {}
        # The models the last pacing evaluation left out, which an empty answer to that pop returns.
        self._paced_out_last_evaluation: frozenset[str] = frozenset()
        # Models returned by an empty answer. They are not paced again until a job for another model is popped.
        self._returned_until_other_pop: set[str] = set()

    @property
    def paced_models(self) -> frozenset[str]:
        """Return the models the pacing window keeps out of the offer now."""
        return frozenset(self._paced)

    def note_paced_model_work(self, *, models_with_work: frozenset[str], now: float) -> None:
        """Mutate each paced model's finish time from whether a job for it is pending or in progress.

        The caller calls this on every pop cycle, before any gate, so the interval is measured from the cycle in
        which the last job finished, even when later gates end those cycles before the offer is built.

        Args:
            models_with_work: The models with a job pending or in progress.
            now: The caller's clock.
        """
        for model, paced in self._paced.items():
            if model in models_with_work:
                paced.finished_at = None
            elif paced.finished_at is None:
                paced.finished_at = now

    def evaluate_pacing(
        self,
        *,
        withheld_claim_model: str | None,
        models_with_work: frozenset[str],
        exempt_models: frozenset[str],
        now: float,
        return_seconds: float,
        idle_escape: bool,
    ) -> LowEarningPacingDecision:
        """Return which models the pacing window leaves out of this pop, updating its state.

        A model leaves the offer when its claim is withheld. It stays out while its claim is withheld, and
        afterwards until ``return_seconds`` have passed since its last job finished. The idle escape returns
        every paced model and starts no new window, as it lets every large model through the other two limits.

        Args:
            withheld_claim_model: The model whose whole-card pop claim is withheld now, or None.
            models_with_work: The models with a job pending or in progress.
            exempt_models: The paced models that the policy or an exemption now holds the card for.
            now: The caller's clock.
            return_seconds: The interval after the last job finishes. At or below zero, nothing is paced.
            idle_escape: Whether the worker holds no work at all.
        """
        returned: list[PacedReturn] = []
        if return_seconds <= 0 or idle_escape:
            cause = PacedReturnCause.WORKER_IDLE if idle_escape else PacedReturnCause.NO_LONGER_PACED
            returned.extend(PacedReturn(model=model, cause=cause) for model in self._paced)
            self._paced.clear()
            self._paced_out_last_evaluation = frozenset()
            return LowEarningPacingDecision(withheld=frozenset(), started=(), returned=tuple(returned))

        for model in sorted(exempt_models & self._paced.keys()):
            del self._paced[model]
            returned.append(PacedReturn(model=model, cause=PacedReturnCause.NO_LONGER_PACED))

        started: tuple[str, ...] = ()
        starts_window = (
            withheld_claim_model is not None
            and withheld_claim_model not in self._paced
            and withheld_claim_model not in self._returned_until_other_pop
            and withheld_claim_model not in exempt_models
        )
        if withheld_claim_model is not None and starts_window:
            self._paced[withheld_claim_model] = _PacedModel(left_at=now, finished_at=None)
            started = (withheld_claim_model,)
        self.note_paced_model_work(models_with_work=models_with_work, now=now)

        for model, paced in list(self._paced.items()):
            interval_passed = paced.finished_at is not None and (now - paced.finished_at) >= return_seconds
            if model != withheld_claim_model and interval_passed:
                del self._paced[model]
                returned.append(PacedReturn(model=model, cause=PacedReturnCause.INTERVAL_PASSED))

        self._paced_out_last_evaluation = frozenset(self._paced)
        return LowEarningPacingDecision(
            withheld=self._paced_out_last_evaluation,
            started=started,
            returned=tuple(returned),
        )

    def note_pop_outcome(self, *, popped_model: str | None) -> tuple[PacedReturn, ...]:
        """Mutate the pacing state from what the horde answered the last pop with, and return the models returned.

        Call it only for a pop the horde answered. A request that errored or timed out says nothing about the work
        the horde has. An empty answer to an offer that left a paced model out shows the horde has nothing else,
        so that model is asked for again until a job for another model is popped.

        Args:
            popped_model: The popped job's model, or None when the horde answered with no job.
        """
        if popped_model is not None:
            self._returned_until_other_pop = {
                model for model in self._returned_until_other_pop if model == popped_model
            }
            return ()
        returned: list[PacedReturn] = []
        for model in sorted(self._paced_out_last_evaluation & self._paced.keys()):
            del self._paced[model]
            self._returned_until_other_pop.add(model)
            returned.append(PacedReturn(model=model, cause=PacedReturnCause.HORDE_HAD_NO_OTHER_WORK))
        self._paced_out_last_evaluation = frozenset()
        return tuple(returned)

    def describe_pacing(self, *, return_seconds: float) -> tuple[PacedModelStatus, ...]:
        """Return each paced model with when it left and when it returns, without mutating state.

        Args:
            return_seconds: The interval after the last job finishes.
        """
        return tuple(
            PacedModelStatus(
                model=model,
                left_at=paced.left_at,
                returns_at=paced.finished_at + return_seconds if paced.finished_at is not None else None,
            )
            for model, paced in sorted(self._paced.items())
        )

    def evaluate(
        self,
        *,
        candidate_large_models: frozenset[str],
        incumbent_large_models: frozenset[str],
        residency_active: bool,
        now: float,
        switch_min_seconds: float,
        reentry_cooldown_seconds: float,
        idle_escape: bool,
    ) -> LargeModelPopDecision:
        """Return which of ``candidate_large_models`` to withhold from this pop, updating timing state.

        Args:
            candidate_large_models: The large models the offer set currently contains (the withhold candidates).
            incumbent_large_models: The large models currently loaded or queued (already "in play").
            residency_active: Whether a whole-card residency lease is currently held (its cooldown still
                running). The re-entry window only opens once this is False (the lease is up).
            now: The current monotonic-ish timestamp (``time.time()`` at the call site).
            switch_min_seconds: Minimum seconds between introducing distinct large models; 0 disables.
            reentry_cooldown_seconds: Seconds to withhold all large models after the last drains; 0 disables.
            idle_escape: When True (worker fully idle, local queue empty), nothing is withheld.
        """
        self._update_state(incumbent_large_models, residency_active=residency_active, now=now)

        if idle_escape:
            return LargeModelPopDecision(withheld=frozenset(), reason=None)

        # Re-entry cooldown is the broader gate (withholds every large model), so it is checked first: when it
        # applies, the switch throttle is moot.
        if (
            reentry_cooldown_seconds > 0
            and not incumbent_large_models
            and not residency_active
            and self._reentry_started_at is not None
            and (now - self._reentry_started_at) < reentry_cooldown_seconds
        ):
            if candidate_large_models:
                return LargeModelPopDecision(withheld=candidate_large_models, reason="re-entry cooldown")
            return LargeModelPopDecision(withheld=frozenset(), reason=None)

        # Switch throttle: a large model is already in play and the interval since the last distinct large
        # model was introduced has not elapsed, so withhold only the *different* large models.
        if (
            switch_min_seconds > 0
            and incumbent_large_models
            and (now - self._last_distinct_large_introduced_at) < switch_min_seconds
        ):
            different = candidate_large_models - incumbent_large_models
            if different:
                return LargeModelPopDecision(withheld=different, reason="switch throttle")

        return LargeModelPopDecision(withheld=frozenset(), reason=None)

    def describe(
        self,
        *,
        incumbent_large_models: frozenset[str],
        residency_active: bool,
        now: float,
        switch_min_seconds: float,
        reentry_cooldown_seconds: float,
    ) -> LargeModelGovernorStatus:
        """Report whether each limiter window is open right now, and how much longer, without mutating state.

        Reads the timing state the last :meth:`evaluate` recorded (the pop loop calls evaluate every cycle, so
        it is current) and applies the same gate conditions read-only, so a status poll never perturbs the
        throttle. ``incumbent_large_models`` and ``residency_active`` are the live observations the caller
        already has; the windows themselves are anchored on the recorded timers.
        """
        switch_active = False
        switch_remaining: float | None = None
        switch_reason: str | None = None
        if switch_min_seconds > 0 and incumbent_large_models:
            elapsed = now - self._last_distinct_large_introduced_at
            if elapsed < switch_min_seconds:
                switch_active = True
                switch_remaining = max(0.0, switch_min_seconds - elapsed)
                switch_reason = "holding off a different very-large model while one is loaded/queued"

        reentry_active = False
        reentry_remaining: float | None = None
        reentry_reason: str | None = None
        if (
            reentry_cooldown_seconds > 0
            and not incumbent_large_models
            and not residency_active
            and self._reentry_started_at is not None
        ):
            elapsed = now - self._reentry_started_at
            if elapsed < reentry_cooldown_seconds:
                reentry_active = True
                reentry_remaining = max(0.0, reentry_cooldown_seconds - elapsed)
                reentry_reason = "cooling down before serving any very-large model after the last drained"

        return LargeModelGovernorStatus(
            switch_active=switch_active,
            switch_remaining_seconds=switch_remaining,
            switch_reason=switch_reason,
            reentry_active=reentry_active,
            reentry_remaining_seconds=reentry_remaining,
            reentry_reason=reentry_reason,
        )

    def _update_state(
        self,
        incumbent_large_models: frozenset[str],
        *,
        residency_active: bool,
        now: float,
    ) -> None:
        """Fold this cycle's observation into the switch-anchor and re-entry-window timers."""
        # Switch anchor: stamp when any large model not previously in play appears.
        if incumbent_large_models - self._known_large_incumbents:
            self._last_distinct_large_introduced_at = now
        self._known_large_incumbents = incumbent_large_models

        # Re-entry window: open it only when a large model has been in play and both conditions (no large
        # model in play, no lease held) are now true; a held lease or a resident large model resets it.
        if incumbent_large_models:
            self._had_large_since_reentry = True
            self._reentry_started_at = None
        elif residency_active:
            self._reentry_started_at = None
        elif self._had_large_since_reentry and self._reentry_started_at is None:
            self._reentry_started_at = now
            self._had_large_since_reentry = False
