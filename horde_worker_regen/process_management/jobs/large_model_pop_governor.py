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

The governor is pure and time-injectable (no clock, no I/O): the caller passes the current large-model
incumbents, the residency-lease flag, and ``now`` each pop cycle, and the governor tracks the timing state
needed to decide what to withhold.
"""

from __future__ import annotations

from dataclasses import dataclass


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
