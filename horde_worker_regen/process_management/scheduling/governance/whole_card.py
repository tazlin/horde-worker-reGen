"""Whole-card exclusive-residency state and its pure queries.

A heavy model can claim a whole card to itself by stopping that card's idle sibling inference contexts
(a context's VRAM is only reclaimed when its process exits) and, on the card the safety process sits on,
moving safety off-GPU. This module owns the per-card residency records and every question that can be
answered from them alone: which cards hold a residency, which card holds a given model, what phase a
residency is in, whether an establish/restore grace window is active, and whether the bounded drain
backstop has elapsed. It also owns the churn governors that bound how fast residencies may be cycled: a
minimum hold before a grant may be released early, a per-card establishment rate limit, and a rolling budget
on how much recovery-supervisor grace a card may open per window. The budget is an admission gate: it holds
off a *new* establishment until the spend replenishes, and never withdraws a window already granted. Each is
stored and answered here, and actuated at the scheduler's call sites.

A held residency also claims the worker's intake: while the claim stands the offer is exactly the resident
model, so the horde stops sending work that would force the resident weights back out of the card. The claim
is a residency fact, so its state and its two end conditions (the operator-configured maximum hold, and the
run of empty pops that says the resident model's demand has dried up) are answered here too.

The scheduler keeps the transitions that touch live processes (establish, converge,
restore); it reads and writes residency state exclusively through
[`WholeCardResidencyLedger`][horde_worker_regen.process_management.scheduling.governance.whole_card.WholeCardResidencyLedger].

Also home to [`max_coresident_for_peak`]
[horde_worker_regen.process_management.scheduling.governance.whole_card.max_coresident_for_peak], the
pure sizing rule for how many live inference contexts a rejected peak can co-reside with.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum, auto

from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.scheduler_budget import SchedulerBudget

__all__ = [
    "GraceBudgetStatus",
    "WholeCardGovernor",
    "WholeCardGrantKind",
    "WholeCardPhase",
    "WholeCardPopClaim",
    "WholeCardPopClaimRelease",
    "WholeCardResidency",
    "WholeCardResidencyLedger",
    "WholeCardResidencyMachine",
    "max_coresident_for_peak",
    "offer_under_pop_claim",
]

_MIN_HOLD_SECONDS = 90.0
"""How long a fresh whole-card residency is immune to early release by a ready different-model head.

Releasing a residency is not free: every sibling inference process the establishment stopped has to be
respawned (~20s each, plus the preload that follows) and the safety process cycled back on-GPU, and the
next heavy head then pays the whole teardown again. A residency released seconds after it was granted
therefore buys one lighter job at the price of two full pool rebuilds, and a queue that alternates heavy
and light heads can repeat that indefinitely. This floor amortizes the actuation already paid over at
least a few jobs' worth of holding; it is deliberately shorter than the establish grace window so it can
never be the thing that keeps a residency alive past the point the recovery supervisor is watching."""

_ESTABLISH_WINDOW_SECONDS = 240.0
"""Rolling window over which a card's whole-card establishments are counted by the rate limiter.

Kept well under the pop-gate structural-wedge backstop (900s) so a head deferred by the limiter can never
accrue toward it: the longest a deferral can last is one window, after which the oldest establishment
falls out and the head is admitted."""

_ESTABLISH_WINDOW_LIMIT = 2
"""Establishments per card per :data:`_ESTABLISH_WINDOW_SECONDS` before a further one is deferred.

Two lets a genuine retry follow a first establishment immediately (an establishment that failed to converge
and was restored still gets a second attempt without waiting), while a third inside the same window is the
signature of a pricing oscillation rather than of demand, and is made to wait out the window."""

_GRACE_BUDGET_WINDOW_SECONDS = 1200.0
"""Rolling window over which the grace granted to a card's residencies is accumulated.

Sized to the field's model-pool rotation period, so the budget is spent and replenished on the same cadence
that legitimately drives one establish/restore cycle per card."""

_GRACE_BUDGET_SECONDS = 360.0
"""Total grace seconds a card may open per :data:`_GRACE_BUDGET_WINDOW_SECONDS` before establishments defer.

One nominal rotation costs a card one establish grant (120s) plus one restore grant (60s), so this allows a
full cycle plus one complete retry and holds the third off until the spend replenishes. Without a ceiling,
back-to-back establish/restore churn re-arms the grace window faster than it expires and the recovery
supervisor is disarmed indefinitely, which is precisely the state in which a real wedge goes unnoticed.
Capping the spend at 30% of the window bounds how much of any window residency churn can cover, and it does
so at admission rather than by withdrawing a window already granted: a granted window covers a teardown the
scheduler itself commanded, so withdrawing it part-way would have the supervisor judge that deliberate action
as a wedge. The bound on a residency that never completes is the granted window's own duration."""

_GOVERNOR_DEFER_DWELL_SECONDS = _ESTABLISH_WINDOW_SECONDS
"""How long a head may be held off the card by a churn governor before it stops asking for the card.

Both governors brake the *rate* at which a card may be rotated; neither is a finding that the head cannot be
served. Left unbounded they become an absolute park: a card measurably able to hold the model sits idle while
the queue stands still, which costs far more than the churn the brake was protecting against. Sized at one
establishment window, the longest a rate deferral can itself last, so a governor that is going to release on
its own is given the chance to before the preference is abandoned. Past it the head falls through to ordinary
measured admission and runs co-resident where the device reading allows: slower than sole residency, and
strictly better than not running."""


_POP_CLAIM_EMPTY_POP_RUN = 3
"""Consecutive pops that must come back with no work for the resident model before the claim releases.

One empty pop is the horde's ordinary answer between jobs, not evidence that a burst is over, and ending a
residency on it would pay a full teardown and regrowth for a lull of a few seconds. Three is the shortest run
that cannot be produced by that ordinary spacing, and it is reached within seconds at any pop cadence, so the
hysteresis costs the worker almost nothing when the demand really has gone."""

_POP_CLAIM_EMPTY_POP_WINDOW_SECONDS = 30.0
"""How long the run of empty pops must have spanned before it counts as the demand having dried up.

The count alone is cadence-dependent: a worker popping every second reaches three empties in a gap far too
short to conclude anything from. Requiring the run to also cover a span makes the evidence a statement about
the horde's queue rather than about this worker's pop rate. Deliberately short next to the maximum hold, so
an idle residency is given back long before the cap has to end it.

Both rails are fixed rather than operator-configured: they describe how much evidence is needed to believe a
reading, which is not a preference, and the operator-facing lever over how long a residency may own the
intake is the maximum hold."""


class WholeCardGovernor(StrEnum):
    """The churn governors that may bar a card from taking on a *new* whole-card residency.

    Both brake the rate at which a card is rotated; neither is a finding that the head cannot be served, so
    each has a bounded dwell past which the head falls through to ordinary measured admission. Enumerating
    them keeps the set of churn brakes closed: a further one has to take a member, so it cannot ship without
    a declared release path in the gate registry.
    """

    ESTABLISH_RATE = "establish_rate"
    """The rolling per-card establishment rate limit is at its ceiling."""
    GRACE_BUDGET = "grace_budget"
    """The card has spent its rolling recovery-supervisor grace allowance."""


@dataclass(frozen=True)
class GraceBudgetStatus:
    """One card's rolling grace spend against its allowance, and the wait for the next replenishment."""

    allowance_seconds: float
    """Total grace the card may have open across the rolling window."""
    spent_seconds: float
    """Grace granted inside the rolling window, which is what the allowance is measured against."""
    remaining_seconds: float
    """Allowance still available, floored at zero once the spend has passed it."""
    replenish_in_seconds: float
    """Wait until enough charges age out for the spend to fall back inside the allowance; 0.0 when it already is."""


@dataclass(frozen=True)
class WholeCardPopClaim:
    """One card's standing claim over the worker's pop offer, held for the duration of its residency.

    While a claim stands the advertised model set is exactly :attr:`model`. Every other model leaves the
    offer, because a foreign job accepted now forces the resident weights to page back to host RAM or be
    re-read from disk, and the scheduler then spends an establish/restore cycle on contention that intake
    manufactured. The claim is a promise about what the worker asks for, never about what it will serve:
    work already accepted keeps its queue position and drains after the claim ends.
    """

    model: str
    """The resident model, and so the only model the worker advertises while this claim stands."""
    device_index: int | None
    """The card whose residency holds the claim (``None`` on a single-GPU host)."""
    held_since: float
    """When the residency behind the claim was established, which is where the maximum hold is measured from."""
    expires_at: float
    """When the maximum hold elapses, after which the pool returns whatever the residency itself is doing."""


class WholeCardPopClaimRelease(StrEnum):
    """Why a standing claim over the pop offer stopped narrowing what the worker advertises.

    The three ends call for different responses, so they are distinguished rather than reported as one
    "claim over" event: a cap expiry says the burst outlasted the window the operator set, the empty-pop
    finding says the demand went away, and a residency release says the card simply stopped holding the
    model. Enumerated so the status surfaces and the disclosure line name the same set.
    """

    MAXIMUM_HOLD = "maximum_hold"
    """The operator's maximum hold elapsed while the residency still held the card."""
    NO_FURTHER_WORK = "no_further_work"
    """The run of empty pops established that the resident model's demand had dried up."""
    RESIDENCY_RELEASED = "residency_released"
    """The residency behind the claim let go, and the claim went with it."""


@dataclass
class WholeCardResidency:
    """Mutable whole-card exclusive-residency state for one card (the worker, on a single-GPU host).

    The ledger keys one of these per device index so two heavy models on different cards each hold their
    own residency independently. A single-GPU worker keeps exactly one instance under the ``None`` key,
    so its behaviour is identical to the pre-multi-GPU scalar fields.
    """

    model: str | None = None
    """The model holding (or being given) sole residency on this card; None when no residency is held."""
    forecast: StreamForecast | None = None
    """The immutable forecast that granted this residency, cached for diagnostics and its fit guarantee."""
    repriced_target: int | None = None
    """The tighten-only live inference-process target derived after the grant.

    The grant forecast is never rewritten while the residency is held: doing so would erase the evidence and
    guarantee that justified admission. New allocator evidence may lower this separate target, but it never
    raises it mid-hold because restoring contexts while the heavy model still owns the card would manufacture
    the same pressure the residency removed.
    """
    established_at: float = 0.0
    """When this residency was first established (stop siblings, cycle safety, load weights); 0.0 when none.

    The establishment intentionally holds the queue, which the recovery supervisor must not mistake for a
    structural wedge until the establish grace elapses."""
    cooldown_until: float = 0.0
    """Wall-clock time until which this residency is held even after its heavy job drains, so a burst of
    heavy jobs reuses one residency instead of each churning a teardown/restore + safety cycle."""
    restore_at: float = 0.0
    """When this residency was last restored (siblings respawned, safety cycled back on-GPU); 0.0 when none.
    The restore churn also briefly makes the queue unservable, so the wedge grace must cover it too."""
    min_hold_until: float = 0.0
    """Wall-clock time before which this residency may not be released early for a different-model head.

    Distinct from ``cooldown_until``, which is an operator-configured hold that a ready different-model head
    may preempt: this floor is not preemptable, because the point of it is to amortize the teardown and
    regrowth the establishment has already paid for. 0.0 when no residency is held."""
    structural_complete_at: float = 0.0
    """When this residency's teardown first satisfied every structural leg (process count, safety, lanes).

    The drain backstop measures from here rather than from ``established_at`` so a slow teardown cannot burn
    the backstop before there is anything to back off from: until the structure is complete there is no
    sole-residency guarantee for the backstop to admit the head against. 0.0 while incomplete."""
    grace_charges: deque[tuple[float, float]] = field(default_factory=deque)
    """``(granted_at, seconds)`` for each grace window this card has been granted, newest last.

    Pruned to :data:`_GRACE_BUDGET_WINDOW_SECONDS` whenever the budget is consulted, so the sum over the
    deque is the grace spent in the current rolling window."""
    establishments: deque[float] = field(default_factory=deque)
    """When each of this card's whole-card residencies was established, newest last.

    Pruned to :data:`_ESTABLISH_WINDOW_SECONDS` whenever the rate limiter is consulted."""
    prestage_process_id: int | None = None
    """The slot a pre-staged head's weights are being loaded into, before any lane carries its model name.

    A convergence shrink spares the residency holder by model name, which a slot mid-load does not yet have,
    so the pre-stage's own target would be a legal victim of the collapse it is loading for. None once the
    residency was established outright (its target is spared at the establish itself) or once no pre-stage is
    outstanding."""
    governor_deferred_model: str | None = None
    """The head model a churn governor is currently holding off this card; None when none is held."""
    governor_deferred_since: float = 0.0
    """When the current governor deferral of ``governor_deferred_model`` began; 0.0 when none is in force."""
    pop_claim_empty_pops: int = 0
    """Consecutive pops that came back with no work while this residency claimed the offer; 0 when the last
    one was served or none has been reported."""
    pop_claim_empty_pop_since: float = 0.0
    """When the current run of empty pops began, so the run is judged over a span as well as a count."""
    pop_claim_released_at: float = 0.0
    """When the empty-pop evidence ended this residency's claim over the offer; 0.0 while it still stands.

    Distinct from the maximum hold, which is a clock on the grant: this is a finding that the resident model
    has no demand left, so continuing to hold the pool to it would starve the worker on its own claim."""


class WholeCardGrantKind(StrEnum):
    """What a residency grant cost the card it was recorded against."""

    ESTABLISH = auto()
    """The card took the model on physically (it held nothing, or held something else), so the grant opened a
    grace window, counted toward the establishment rate, and reset the holds that a teardown resets."""
    REUSE = auto()
    """A further job asked for the model the card already holds, which costs no teardown and so charges
    nothing: it extends the cooldown and leaves every clock the physical episode started where it was."""


class WholeCardPhase(StrEnum):
    """The externally-visible phase of one card's whole-card residency."""

    NONE = auto()
    """No residency is held on the card."""
    ESTABLISHING = auto()
    """A residency is held and still inside its establish grace (teardown/load in progress)."""
    HOLDING = auto()
    """A residency is held past its establish grace (the heavy model owns the card)."""


class WholeCardResidencyLedger:
    """Owns the per-card whole-card residency records and their pure queries.

    Thread Safety:
        Owned and mutated exclusively by the scheduler's control loop; not safe for concurrent mutation.
    """

    def __init__(self) -> None:
        """Initialize an empty ledger (no card holds a residency)."""
        self._residencies: dict[int | None, WholeCardResidency] = {}

    def state_for(self, device_index: int | None) -> WholeCardResidency:
        """Return the (lazily-created) residency state for ``device_index``.

        ``None`` is the single-GPU / worker-wide key, so a single-GPU host keeps exactly one residency
        state.
        """
        state = self._residencies.get(device_index)
        if state is None:
            state = WholeCardResidency()
            self._residencies[device_index] = state
        return state

    def get(self, device_index: int | None) -> WholeCardResidency | None:
        """Return the residency state for ``device_index`` without creating one, or None when absent."""
        return self._residencies.get(device_index)

    def held(self) -> list[tuple[int | None, WholeCardResidency]]:
        """Return ``(device_index, state)`` for every card currently holding a residency (model set)."""
        return [(index, state) for index, state in self._residencies.items() if state.model is not None]

    def any_held(self) -> bool:
        """Return whether any card currently holds a whole-card residency."""
        return any(state.model is not None for state in self._residencies.values())

    def holder_for_model(self, model: str | None) -> tuple[bool, int | None]:
        """Return ``(found, device_index)`` for the card whose held residency is for ``model``.

        ``found`` distinguishes a genuine hit on the ``None`` (single-GPU / worker-wide) key from a miss,
        since ``None`` is itself a valid residency key.
        """
        if model is None:
            return (False, None)
        for device_index, state in self._residencies.items():
            if state.model == model:
                return (True, device_index)
        return (False, None)

    def record_grant(
        self,
        device_index: int | None,
        *,
        model: str | None,
        forecast: StreamForecast | None,
        cooldown_until: float,
        now: float,
        establish_grace_seconds: float = 0.0,
    ) -> WholeCardGrantKind:
        """Record a residency grant (an establishment or a RAM pre-stage) for ``device_index``.

        Sets the model and cooldown, and returns which kind of grant this was. Everything else the grant
        touches is scoped to a *physical* grace window: the card taking the model on when it held nothing or
        held something else. That is what costs a teardown, opens the window the recovery supervisor honours,
        and resets the clocks a teardown resets, so it stamps ``established_at``, opens a fresh min-hold
        floor, clears the previous grant's structural-completion latch (a re-established residency tears down
        again, so it must not inherit an elapsed drain backstop), records the establishment for the rate
        limiter, and charges ``establish_grace_seconds`` against the card's rolling grace budget. Passing zero
        seconds records no charge, which is what a caller not claiming a grace window should do.

        A further job asking for the model the card already holds is a reuse: it rides the residency the
        cooldown is deliberately keeping alive and physically costs nothing, so it only extends the cooldown.
        Charging it, counting it, or re-opening its establish window would spend an allowance sized for
        teardown churn on jobs that caused none, hand the supervisor a fresh excuse for a queue nothing is
        tearing down, and restart the drain backstop of a residency that has already converged. The grant
        forecast is likewise captured only for a physical grant, keeping it immutable across repeated asks.
        """
        state = self.state_for(device_index)
        kind = (
            WholeCardGrantKind.REUSE
            if state.model is not None and state.model == model and state.established_at != 0.0
            else WholeCardGrantKind.ESTABLISH
        )
        # A grant of either kind is a job asking for this model, which is the direct contradiction of the
        # empty-pop evidence: the run restarts from the demand that just arrived. An establishment additionally
        # restarts the maximum hold, since the clock measures one physical residency episode.
        state.pop_claim_empty_pops = 0
        state.pop_claim_empty_pop_since = 0.0
        if kind is WholeCardGrantKind.ESTABLISH:
            state.established_at = now
            state.min_hold_until = now + _MIN_HOLD_SECONDS
            state.structural_complete_at = 0.0
            state.pop_claim_released_at = 0.0
            state.establishments.append(now)
            if establish_grace_seconds > 0.0:
                state.grace_charges.append((now, establish_grace_seconds))
            state.forecast = forecast
            state.repriced_target = forecast.max_resident_processes() if forecast is not None else None
        elif state.forecast is None:
            state.forecast = forecast
            state.repriced_target = forecast.max_resident_processes() if forecast is not None else None
        state.model = model
        state.cooldown_until = cooldown_until
        return kind

    def record_restore(self, device_index: int | None, *, now: float, restore_grace_seconds: float = 0.0) -> None:
        """Clear a drained residency on ``device_index`` and open its restore window.

        The restore's own churn (respawning the stopped siblings, cycling safety back on-GPU) briefly makes
        the queue unservable, so the stamp is what the wedge grace reads; the grant's holds (min-hold,
        structural completion) are released with it and ``restore_grace_seconds`` is charged against the
        card's rolling grace budget.

        The window is granted for churn, so a caller that finds it had none to do withdraws the grant with
        :meth:`close_restore_window`. That is a narrower grant, not a shortened one: a window this card is
        genuinely inside still runs its full duration and nothing cuts it short.
        """
        state = self.state_for(device_index)
        state.model = None
        state.forecast = None
        state.repriced_target = None
        state.established_at = 0.0
        state.min_hold_until = 0.0
        state.structural_complete_at = 0.0
        state.prestage_process_id = None
        state.pop_claim_empty_pops = 0
        state.pop_claim_empty_pop_since = 0.0
        state.pop_claim_released_at = 0.0
        state.restore_at = now
        if restore_grace_seconds > 0.0:
            state.grace_charges.append((now, restore_grace_seconds))

    def close_restore_window(self, device_index: int | None, *, granted_at: float) -> None:
        """Withdraw a restore window granted at ``granted_at`` whose restore had no churn to cover.

        A restore that respawns nothing and restarts no lane leaves the card exactly as it found it, so the
        wedge grace has no deliberate action to excuse. Left standing, the window would tell the recovery
        supervisor to ignore a genuine wedge for the rest of its duration after every such release, and the
        allowance it charged would price teardown churn that never happened. Both are undone here: the stamp
        is cleared and the charge this grant added is dropped. A no-op once any other grant has been recorded
        against the card, so only the caller's own window is ever withdrawn.
        """
        state = self._residencies.get(device_index)
        if state is None or state.restore_at != granted_at:
            return
        state.restore_at = 0.0
        if state.grace_charges and state.grace_charges[-1][0] == granted_at:
            state.grace_charges.pop()

    def min_hold_active(self, device_index: int | None, *, now: float) -> bool:
        """Return whether this card's residency is still inside its non-preemptable minimum hold."""
        state = self._residencies.get(device_index)
        if state is None or state.model is None:
            return False
        return now < state.min_hold_until

    def min_hold_disclosure(self, device_index: int | None, *, now: float) -> str | None:
        """Return what the minimum hold is withholding right now, or None when it is not in force.

        The stamp names :attr:`SchedulerBudget.WHOLE_CARD_MIN_HOLD`, which is the only runtime surface the
        floor has: it holds a ready head without returning an admission verdict, so a reader who sees a
        different-model head waiting behind a drained residency has nothing else to attribute it to. The
        figures tick, so this belongs to a formatted line and never to a compared reason.
        """
        state = self._residencies.get(device_index)
        if state is None or state.model is None or now >= state.min_hold_until:
            return None
        return (
            f"{SchedulerBudget.WHOLE_CARD_MIN_HOLD.value}: {state.model} took the card "
            f"{now - state.established_at:.0f}s ago and is not preemptable for a further "
            f"{state.min_hold_until - now:.0f}s"
        )

    def pop_claim(
        self,
        device_index: int | None,
        *,
        now: float,
        max_hold_seconds: float,
    ) -> WholeCardPopClaim | None:
        """Return this card's standing claim over the pop offer, or None when it holds none.

        A claim stands from the establishment until the first of its two ends: ``max_hold_seconds`` elapsing
        from the grant, or the empty-pop evidence finding that the resident model has no demand left. A
        non-positive ``max_hold_seconds`` means the caller is not running the claim at all (the operator
        disabled it, or the host drives more than one card), and no claim is ever reported.

        Deliberately live rather than latched: the claim is derived from the residency and the clock on every
        ask, so releasing the residency releases the claim with it and nothing can outlive the thing it was
        taken for.
        """
        state = self._residencies.get(device_index)
        if state is None or state.model is None or max_hold_seconds <= 0.0:
            return None
        if state.pop_claim_released_at != 0.0 or state.established_at == 0.0:
            return None
        expires_at = state.established_at + max_hold_seconds
        if now >= expires_at:
            return None
        return WholeCardPopClaim(
            model=state.model,
            device_index=device_index,
            held_since=state.established_at,
            expires_at=expires_at,
        )

    def note_pop_outcome(self, device_index: int | None, *, served: bool, now: float) -> bool:
        """Record what one pop taken under this card's claim came back with, and report an early release.

        A pop that returned work is proof the resident model still has demand, so it restarts the run. A pop
        that returned nothing extends it, and once the run has both reached :data:`_POP_CLAIM_EMPTY_POP_RUN`
        and covered :data:`_POP_CLAIM_EMPTY_POP_WINDOW_SECONDS` the claim is released: a residency whose model
        nobody is asking for would otherwise hold the whole pool to itself until the maximum hold expired,
        which is the worker starving on its own claim.

        Returns:
            True on the pop that released the claim, so the caller can disclose the edge once.
        """
        state = self._residencies.get(device_index)
        if state is None or state.model is None:
            return False
        if served:
            state.pop_claim_empty_pops = 0
            state.pop_claim_empty_pop_since = 0.0
            return False
        if state.pop_claim_empty_pops == 0:
            state.pop_claim_empty_pop_since = now
        state.pop_claim_empty_pops += 1
        if state.pop_claim_released_at != 0.0:
            return False
        if state.pop_claim_empty_pops < _POP_CLAIM_EMPTY_POP_RUN:
            return False
        if (now - state.pop_claim_empty_pop_since) < _POP_CLAIM_EMPTY_POP_WINDOW_SECONDS:
            return False
        state.pop_claim_released_at = now
        return True

    def pop_claim_retention_ended(
        self,
        device_index: int | None,
        *,
        now: float,
        max_hold_seconds: float,
    ) -> bool:
        """Return whether this residency's claim has ended, so the residency is no longer retained.

        The two ends of a claim are also the two statements that the residency has had its window: the
        maximum hold caps how long one grant may own the card and the intake, and the empty-pop release says
        the demand it was holding for is gone. Past either, the release path stops refreshing the cooldown and
        stops honouring a standing one, so the residency lets go as soon as its own work has drained.

        It ends retention, never an action in flight: a granted establish or restore window runs its full
        duration, and work already dispatched is left to finish.
        """
        state = self._residencies.get(device_index)
        if state is None or state.model is None or max_hold_seconds <= 0.0:
            return False
        if state.pop_claim_released_at != 0.0:
            return True
        if state.established_at == 0.0:
            return False
        return now >= (state.established_at + max_hold_seconds)

    def establish_rate_exceeded(self, device_index: int | None, *, now: float) -> bool:
        """Return whether this card has already established as often as the rolling window allows.

        A caller that sees True should defer the new establishment and re-ask, never deny it outright: the
        window is short enough (:data:`_ESTABLISH_WINDOW_SECONDS`) that the deferral always resolves well
        inside the structural-wedge backstop the deferred head would otherwise accrue toward.
        """
        state = self._residencies.get(device_index)
        if state is None:
            return False
        _prune_before(state.establishments, cutoff=now - _ESTABLISH_WINDOW_SECONDS)
        return len(state.establishments) >= _ESTABLISH_WINDOW_LIMIT

    def phase(
        self,
        device_index: int | None,
        *,
        now: float,
        establish_grace_seconds: float,
    ) -> tuple[str | None, WholeCardPhase]:
        """Return ``(model, phase)`` for the residency held on ``device_index``.

        ``model`` is None (with phase ``NONE``) when the card holds no residency. Reads without creating:
        a card with no residency is left absent from the ledger.
        """
        state = self._residencies.get(device_index)
        if state is None or state.model is None:
            return None, WholeCardPhase.NONE
        establishing = state.established_at != 0.0 and (now - state.established_at) < establish_grace_seconds
        return state.model, (WholeCardPhase.ESTABLISHING if establishing else WholeCardPhase.HOLDING)

    def grace_active(
        self,
        *,
        now: float,
        establish_grace_seconds: float,
        restore_grace_seconds: float,
    ) -> bool:
        """Return whether any residency sits inside a granted establish or restore window.

        A granted window excuses a held queue for its whole duration and nothing shortens it. The teardown or
        restore it covers is an action the scheduler itself commanded, so withdrawing the excuse part-way
        would have the recovery supervisor classify that deliberate action as a structural wedge. The
        liveness bound on a residency that never finishes loading (or a restore that never completes) is the
        window's own duration, measured from the grant. How often a card may open a *new* window is governed
        at admission by :meth:`grace_budget_exhausted`, not here.

        The grant condition is churn: a restore window covers siblings actually being respawned or a service
        lane actually being restarted, and a release that changed nothing is granted none (see
        :meth:`close_restore_window`). Narrowing what earns a window is distinct from cutting one short, which
        nothing does.
        """
        return any(
            self._window_active(
                state,
                now=now,
                establish_grace_seconds=establish_grace_seconds,
                restore_grace_seconds=restore_grace_seconds,
            )
            for state in self._residencies.values()
        )

    def grace_budget_exhausted(self, device_index: int | None, *, now: float) -> bool:
        """Return whether this card has spent its rolling-window grace allowance.

        The admission-side query on the whole-card path: a caller about to establish a *new* residency on
        this card defers while this is True, so the card stops opening fresh grace windows until its spend
        replenishes. It says nothing about a window already granted, which runs to its own duration
        regardless; refreshes and restores of a residency the card already holds are not gated by it.

        Spend is the sum of the grace windows granted to the card inside
        :data:`_GRACE_BUDGET_WINDOW_SECONDS`; it replenishes as those grants age out of the window.
        """
        state = self._residencies.get(device_index)
        if state is None:
            return False
        _prune_charges_before(state.grace_charges, cutoff=now - _GRACE_BUDGET_WINDOW_SECONDS)
        return sum(seconds for _granted_at, seconds in state.grace_charges) > _GRACE_BUDGET_SECONDS

    def grace_budget_status(self, device_index: int | None, *, now: float) -> GraceBudgetStatus:
        """Return this card's rolling grace spend, remaining allowance, and wait for the next replenishment.

        The disclosure companion to :meth:`grace_budget_exhausted`: a refusal that quotes the arithmetic can be
        acted on, where a bare "deferred" cannot. The replenishment wait is derived from the charges
        themselves, being how long until enough of them age out of the rolling window for the spend to sit back
        inside the allowance.
        """
        state = self._residencies.get(device_index)
        if state is None:
            return GraceBudgetStatus(
                allowance_seconds=_GRACE_BUDGET_SECONDS,
                spent_seconds=0.0,
                remaining_seconds=_GRACE_BUDGET_SECONDS,
                replenish_in_seconds=0.0,
            )
        _prune_charges_before(state.grace_charges, cutoff=now - _GRACE_BUDGET_WINDOW_SECONDS)
        spent = sum(seconds for _granted_at, seconds in state.grace_charges)
        replenish_in = 0.0
        if spent > _GRACE_BUDGET_SECONDS:
            outstanding = spent
            for granted_at, seconds in state.grace_charges:
                outstanding -= seconds
                if outstanding <= _GRACE_BUDGET_SECONDS:
                    replenish_in = max(0.0, (granted_at + _GRACE_BUDGET_WINDOW_SECONDS) - now)
                    break
        return GraceBudgetStatus(
            allowance_seconds=_GRACE_BUDGET_SECONDS,
            spent_seconds=spent,
            remaining_seconds=max(0.0, _GRACE_BUDGET_SECONDS - spent),
            replenish_in_seconds=replenish_in,
        )

    @property
    def governor_defer_dwell_seconds(self) -> float:
        """How long a head may be held off a card by a churn governor before it stops asking for the card."""
        return _GOVERNOR_DEFER_DWELL_SECONDS

    def note_governor_defer(self, device_index: int | None, *, model: str | None, now: float) -> tuple[float, bool]:
        """Record that a churn governor is holding ``model`` off this card, and report the wait so far.

        Returns ``(elapsed_seconds, dwell_exhausted)``. The wait is anchored at the first of a run of
        deferrals for this model on this card and restarts whenever the deferred model changes, so one head's
        wait is never inherited by the next. ``dwell_exhausted`` is the caller's cue to stop preferring the
        whole card for this head and let ordinary measured admission decide instead; see
        :data:`_GOVERNOR_DEFER_DWELL_SECONDS` for why the preference is bounded at all.
        """
        state = self.state_for(device_index)
        if state.governor_deferred_model != model or state.governor_deferred_since == 0.0:
            state.governor_deferred_model = model
            state.governor_deferred_since = now
        elapsed = max(0.0, now - state.governor_deferred_since)
        return elapsed, elapsed >= _GOVERNOR_DEFER_DWELL_SECONDS

    def clear_governor_defer(self, device_index: int | None) -> None:
        """Forget any governor deferral recorded for this card, so the next one is timed from its own start."""
        state = self._residencies.get(device_index)
        if state is None:
            return
        state.governor_deferred_model = None
        state.governor_deferred_since = 0.0

    def governor_deferred_head(self, device_index: int | None, *, now: float) -> str | None:
        """Return the model this card's governor deferral is still holding off, or None when none is in force.

        Only a deferral inside its dwell counts. Past the dwell the head stops asking for the card and is
        served by ordinary admission, so it is no longer a governance hold and no longer a head that has
        deliberately stood down.
        """
        state = self._residencies.get(device_index)
        if state is None or state.governor_deferred_since == 0.0:
            return None
        if (now - state.governor_deferred_since) >= _GOVERNOR_DEFER_DWELL_SECONDS:
            return None
        return state.governor_deferred_model

    def any_governor_defer_active(self, *, now: float) -> bool:
        """Return whether any card is inside a governor deferral's dwell."""
        return any(
            self.governor_deferred_head(device_index, now=now) is not None for device_index in self._residencies
        )

    @staticmethod
    def _window_active(
        state: WholeCardResidency,
        *,
        now: float,
        establish_grace_seconds: float,
        restore_grace_seconds: float,
    ) -> bool:
        """Return whether one card's residency sits inside a nominal establish or restore window."""
        establishing = (
            state.model is not None
            and state.established_at != 0.0
            and (now - state.established_at) < establish_grace_seconds
        )
        restoring = state.restore_at != 0.0 and (now - state.restore_at) < restore_grace_seconds
        return establishing or restoring

    def drain_backstop_elapsed(self, device_index: int | None, *, now: float, settle_seconds: float) -> bool:
        """Return whether the bounded drain-settle window has elapsed since the teardown became structural.

        The deterministic backstop for the dispatch gate: once a structurally-complete teardown has held
        for ``settle_seconds`` without the live free-VRAM reading confirming the drain, the head is
        admitted on the structural guarantee rather than parking forever. Measured from
        ``structural_complete_at``, so the clock only runs once there is a sole-residency guarantee to admit
        the head against: a slow establishment (siblings still exiting, safety still cycling) does not burn
        the backstop before the drain it backstops has anything to drain.
        """
        state = self._residencies.get(device_index)
        if state is None or state.structural_complete_at == 0.0:
            return False
        return (now - state.structural_complete_at) >= settle_seconds


class WholeCardResidencyMachine(WholeCardResidencyLedger):
    """Whole-card residency state machine plus pure transition queries.

    The scheduler still executes side effects (process scale-down, safety cycling, VRAM eviction), but this
    class owns the multi-tick residency state and the policy questions that can be answered without touching
    live process objects. It extends :class:`WholeCardResidencyLedger` so existing adapter properties can be
    migrated incrementally without changing behavior.
    """

    def residency_demanded(
        self,
        forecast: StreamForecast,
        *,
        enabled: bool,
        is_head_blocker: bool,
        live_inference_process_count: int,
    ) -> bool:
        """Return whether a job should enter the whole-card residency pipeline.

        Args:
            forecast: The head's streaming forecast against the card it would claim.
            enabled: Whether whole-card exclusive residency is configured on.
            is_head_blocker: Whether this job is the head; only the head may claim the card.
            live_inference_process_count: Inference processes currently holding a context on the card. A
                process-count-reduction claim is priced against this, since it is what the claim proposes to
                reduce.

        Returns:
            Whether the job enters the residency pipeline.
        """
        if not (enabled and is_head_blocker):
            return False
        if forecast.needs_exclusive_residency:
            return True
        if not forecast.needs_process_count_reduction:
            return False
        # The process-count-reduction branch claims the live inference contexts are themselves the
        # over-commit, and its remedy is stopping some of them. Two things must both hold for that claim to
        # be incoherent, and only then is it refused. The budget must already hold at least as many contexts
        # as are running, so the reduction has nothing to take; and the deficit must be no larger than the
        # charges an inference teardown cannot reclaim (safety's footprint, the service lanes' contexts, the
        # image lane's decode spike: StreamForecast.unreclaimable_charge_mb), so it is attributable to them
        # rather than to the contexts. Granting whole-card semantics there spends the card's pop monopoly,
        # retention hold, and sibling eviction to reach a topology the card is already in, and the teardown
        # then deletes the very charges the deficit was made of. Requiring both keeps a genuine context
        # over-commit, on a card carrying no such charges, on the residency path. Weight-dominant and
        # whole-card-on-intent claims are decided above and never reach here.
        target = forecast.max_resident_processes()
        if target is None or target < max(1, live_inference_process_count):
            return True
        return not self._deficit_is_unreclaimable(forecast)

    @staticmethod
    def _deficit_is_unreclaimable(forecast: StreamForecast) -> bool:
        """Whether the weight shortfall against the siblings-present floor is within the unreclaimable charges.

        Args:
            forecast: The forecast whose process-count-reduction verdict is being attributed.

        Returns:
            True when the shortfall is no larger than the charges stopping inference siblings cannot give
            back, so the reduction remedy is not what the deficit calls for. False when the forecast reports
            no such charges (a directly-constructed forecast, or a card carrying neither safety nor a service
            lane), which is the conservative direction: the deficit is then treated as a real context
            over-commit.
        """
        if forecast.weights_mb is None or forecast.free_after_model_evict_mb is None:
            return False
        floor_mb = forecast.base_reserve_mb if forecast.base_reserve_mb is not None else forecast.reserve_mb
        shortfall_mb = forecast.weights_mb + floor_mb - forecast.free_after_model_evict_mb
        return shortfall_mb <= forecast.unreclaimable_charge_mb

    def target_process_count(self, forecast: StreamForecast | None) -> int | None:
        """Return the live inference-process target for a held residency, or None when it cannot be sized.

        None means the forecast cannot say how many contexts the card holds (no weight estimate, no reported
        total VRAM, or no per-process overhead to reason about), which is a different statement from "one".
        Coercing it to sole residency tears the pool down to a single process on the strength of a figure
        nobody measured, on exactly the hosts where the measurement is missing. A caller that cannot size the
        card leaves the pool where it is instead; the residency's other gates (the live weight fit and the
        drain backstop) still govern when the head loads.
        """
        if forecast is None:
            return None
        return forecast.max_resident_processes()

    def effective_target(self, state: WholeCardResidency) -> int | None:
        """Return the held residency's tighten-only process target.

        ``repriced_target`` is initialized from the grant forecast and may only move down. Falling back to the
        immutable forecast keeps directly-constructed legacy state useful in tests and adapters.
        """
        if state.repriced_target is not None:
            return state.repriced_target
        return self.target_process_count(state.forecast)

    def tighten_target(self, device_index: int | None, target: int | None) -> bool:
        """Lower a held residency's effective process target, never grow it mid-hold.

        Args:
            device_index: Card whose residency is being re-priced.
            target: Newly calculated target, or None when the current evidence cannot size it.

        Returns:
            True when the effective target became stricter.
        """
        state = self._residencies.get(device_index)
        if state is None or state.model is None or target is None:
            return False
        current = self.effective_target(state)
        if current is not None and target >= current:
            return False
        state.repriced_target = target
        return True

    def teardown_complete(
        self,
        forecast: StreamForecast,
        *,
        loaded_process_count: int,
        safety_clear_of_card: bool,
        process_target: int | None = None,
        post_process_pause_required: bool = False,
        post_process_cleared: bool = True,
        component_lane_pause_required: bool = False,
        component_lane_cleared: bool = True,
        weights_fit_live: bool,
        drain_backstop_elapsed: bool,
        resident_context_charge_mb: float = 0.0,
        device_index: int | None = None,
        now: float | None = None,
    ) -> bool:
        """Return whether a held residency has cleared enough room for the head to sample.

        The head must not be admitted until every VRAM consumer the residency displaces has actually vacated
        the card: the live inference-process count is at (or below) the effective target, safety is physically
        clear of this card, and the dedicated post-processing and component lanes have left the card if
        they need to. A lane's context is only freed when its process exits, so ``post_process_cleared`` and
        ``component_lane_cleared`` are structural checks (the lane is gone), distinct from the pause merely
        having been requested; admitting the head while a lane's context is still resident is exactly what
        leaves too little room and streams the weights.

        Once the structural checks hold, a live free-VRAM reading that holds the weights releases the head
        immediately; otherwise the bounded drain backstop releases it on the forecast's sole-residency
        guarantee. That guarantee describes a card every other context has left, so ``resident_context_charge_mb``
        prices back in whatever the residency is leaving on the card (the safety context, where the
        configuration forbids moving it off-GPU). The caller supplies the figure because which contexts stay
        is a configuration and lifecycle fact, not residency state; zero (nothing stays) is the plain
        sole-residency guarantee.

        A forecast that cannot size the card (:meth:`target_process_count` returns None) leaves the
        process-count leg satisfied: no teardown depth was ever demanded of the pool, so waiting on one would
        park the head forever. The remaining legs still gate, so the head is admitted on a measured fit or the
        drain backstop rather than on the unsized count.

        Passing ``now`` (with the ``device_index`` the other arguments were gathered for) latches the moment
        the structural legs first all pass onto the residency, which is what the drain backstop measures
        from. The latch lives here because the structural legs are computed here and nowhere else; callers
        that only want the question answered omit ``now`` and record nothing.
        """
        target = process_target if process_target is not None else self.target_process_count(forecast)
        structurally_complete = not (
            (target is not None and loaded_process_count > target)
            or not safety_clear_of_card
            or (post_process_pause_required and not post_process_cleared)
            or (component_lane_pause_required and not component_lane_cleared)
        )
        if now is not None and structurally_complete:
            state = self._residencies.get(device_index)
            if state is not None and state.model is not None and state.structural_complete_at == 0.0:
                state.structural_complete_at = now
        if not structurally_complete:
            return False
        if weights_fit_live:
            return True
        return forecast.fits_alone_beside(resident_context_charge_mb) and drain_backstop_elapsed


def offer_under_pop_claim(models: frozenset[str], *, claim: WholeCardPopClaim | None) -> frozenset[str]:
    """Return the model set a worker may advertise while ``claim`` stands.

    With no claim the offer passes through untouched. Under a claim it is exactly the resident model, and
    nothing else: the point of the claim is that a foreign job accepted now costs the residency the card.

    The result is empty when the claimed model is not in ``models`` at all, which is a real state and not an
    error: an earlier offer stage (a quarantine, a serviceability exclusion) has already found the resident
    model unofferable this cycle. Advertising the rest of the pool instead would hand the card exactly the
    foreign work the claim exists to keep off it, so the caller withholds the pop and says so.
    """
    if claim is None:
        return models
    return models & {claim.model}


def _prune_before(stamps: deque[float], *, cutoff: float) -> None:
    """Drop timestamps at or before ``cutoff`` from the front of an oldest-first deque.

    Inclusive at the boundary so a stamp exactly one window old is already out: the disclosed replenish
    wait (``granted_at + window - now``) is then exact rather than short by an instant.
    """
    while stamps and stamps[0] <= cutoff:
        stamps.popleft()


def _prune_charges_before(charges: deque[tuple[float, float]], *, cutoff: float) -> None:
    """Drop ``(granted_at, seconds)`` charges granted at or before ``cutoff`` from an oldest-first deque.

    Inclusive at the boundary for the same reason as :func:`_prune_before`.
    """
    while charges and charges[0][0] <= cutoff:
        charges.popleft()


def max_coresident_for_peak(
    *,
    total_vram_mb: float | None,
    per_process_overhead_mb: float,
    marginal_overhead_mb: float | None,
    peak_mb: float,
    reserve_mb: float,
) -> int | None:
    """Return the largest live inference-process count that still fits ``peak_mb`` plus ``reserve_mb``.

    The loader's first context costs the full one-time overhead; each additional co-resident context
    costs only the marginal (falling back to the full figure when unmeasured or zero). Returns None when
    the depth cannot be sized (no reported total VRAM, or non-positive overhead figures); never below
    one, since the job's own context always exists.
    """
    if total_vram_mb is None or per_process_overhead_mb <= 0:
        return None
    marginal = marginal_overhead_mb or per_process_overhead_mb
    if marginal <= 0:
        return None
    budget = total_vram_mb - peak_mb - reserve_mb
    if budget <= per_process_overhead_mb:
        return 1
    return max(1, 1 + int((budget - per_process_overhead_mb) // marginal))
