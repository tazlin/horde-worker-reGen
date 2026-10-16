"""Whole-card exclusive-residency state and its pure queries.

A heavy model can claim a whole card to itself by stopping that card's idle sibling inference contexts
(a context's VRAM is only reclaimed when its process exits) and, on the card the safety process sits on,
moving safety off-GPU. This module owns the per-card residency records and every question that can be
answered from them alone: which cards hold a residency, which card holds a given model, what phase a
residency is in, whether an establish/restore grace window is active, and whether the bounded drain
backstop has elapsed. It also owns the churn governors that bound how fast residencies may be cycled: a
minimum hold before a grant may be released early, a per-card establishment rate limit, and a rolling budget
on how much recovery-supervisor grace a card may open per window. Only the rate limit defers a new
establishment. The budget limits the excuse. Once a card has spent it, an establishment or restore proceeds
without opening a window until the spend replenishes, and a window already granted is never withdrawn. Each is
stored and answered here, and actuated at the scheduler's call sites.

A held residency also claims the worker's intake: while the claim stands the offer is exactly the resident
model, so the horde stops sending work that would force the resident weights back out of the card. The claim
is a residency fact, so its state and its early ends (the operator-configured maximum hold, the run of empty
pops that says the resident model's demand has dried up, and a terminal fault of the claimed model) are
answered here too.

Each grant carries the dominance rung it was made on (:class:`DominanceRung`). Rung 2 carries the teardown. The
readers that stop siblings, move safety or pause lanes read :meth:`WholeCardResidencyLedger.teardown_held`, so a rung-1
grant shares the claim and the cooldown without tearing anything down. A rung-1 grant is a run: the worker keeps
earning on the dominant model's loaded weights instead of swapping to another large model and back. Since it does
not tear anything down, it does not open a grace window, is not counted by either churn governor, has no minimum
hold, and the run's end does not regrow anything (:meth:`WholeCardResidencyLedger.record_run_end`).

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
from typing import TYPE_CHECKING, Literal

from loguru import logger

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.scheduler_budget import SchedulerBudget

if TYPE_CHECKING:
    from collections.abc import Iterable

    from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo

__all__ = [
    "HEAVY_HEAD_LOAD_GRACE_SECONDS",
    "POP_CLAIM_RELEASE_VISIBLE_SECONDS",
    "WHOLE_CARD_DRAIN_SETTLE_SECONDS",
    "WHOLE_CARD_ESTABLISH_GRACE_SECONDS",
    "WHOLE_CARD_RESTORE_GRACE_SECONDS",
    "DominanceRung",
    "GraceBudgetStatus",
    "WholeCardGovernor",
    "WholeCardGovernorHold",
    "WholeCardPopClaimTracker",
    "WholeCardGrantKind",
    "WholeCardPhase",
    "WholeCardPopClaim",
    "WholeCardPopClaimRelease",
    "WholeCardResidency",
    "WholeCardResidencyLedger",
    "WholeCardResidencyMachine",
    "convergence_blockers",
    "max_coresident_for_peak",
    "offer_under_pop_claim",
    "post_process_context_fits",
    "residency_has_holder",
]

WHOLE_CARD_ESTABLISH_GRACE_SECONDS = 120.0
"""How long after a residency is established the queue may be held (siblings stopping, safety cycling off-GPU,
a multi-GB load) before the recovery supervisor reads it as a structural wedge. Bounded so a residency that never
loads still trips the supervisor."""

WHOLE_CARD_DRAIN_SETTLE_SECONDS = 20.0
"""The longest a teardown at sole residency holds the head for its drain to settle
(:meth:`WholeCardResidencyLedger.drain_settled`) before loading on the structural fits-alone guarantee. An upper
bound for a reading that never settles or never arrives; a settled reading releases the head sooner. Kept under
the establish grace so the head always dispatches before the supervisor would act."""

WHOLE_CARD_RESTORE_GRACE_SECONDS = 60.0
"""How long after a residency is restored the supervisor keeps ignoring a queue wedge: respawning siblings and
cycling safety back on-GPU briefly make the queue unservable, and reading that as a wedge would cascade into
further churn."""

POP_CLAIM_RELEASE_VISIBLE_SECONDS = 120.0
"""How long a released pop claim's end is still reported beside the residency posture, so the widening of the
offer is explained while the operator is looking at it and not minutes later."""

HEAVY_HEAD_LOAD_GRACE_SECONDS = 120.0
"""How long after a heavy head is admitted off the whole-card path its multi-GB load may hold the queue before
the supervisor reads a wedge. Such a head bypasses the establish grace yet holds the queue the same way."""

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
falls out and the head is admitted. The same window bounds the record of a deferral
(:meth:`WholeCardResidencyLedger.governor_deferred_head`), so a record nobody clears stops excusing the queue."""

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
"""Total grace seconds a card may open per :data:`_GRACE_BUDGET_WINDOW_SECONDS`.
Once it is spent, an establishment or restore does not open a window.

One nominal rotation costs a card one establish grant (120s) plus one restore grant (60s), so this allows a
full cycle plus one complete retry before a further establishment or restore proceeds without a window. Without a
ceiling, back-to-back establish/restore churn re-arms the grace window faster than it expires and the recovery
supervisor is disarmed indefinitely, which is precisely the state in which a real wedge goes unnoticed.
Capping the spend at 30% of the window bounds how much of any window residency churn can cover. It does so
when a window opens, never by withdrawing one already granted. A granted window covers a teardown the
scheduler itself commanded, so withdrawing it part-way would have the supervisor judge that deliberate action
as a wedge. The bound on a residency that never completes is the granted window's own duration. The spend
never refuses the card itself. The rotation rate is the establish-rate limiter's to brake, and a card held
off for want of an excuse would sit idle with the head unserved."""


_POP_CLAIM_EMPTY_POP_RUN = 3
"""Consecutive pops that must come back with no work for the resident model before the claim releases.

One empty pop is the horde's ordinary answer between jobs, not evidence that a burst is over, and ending a
residency on it would pay a full teardown and regrowth for a lull of a few seconds. Three is the shortest run
that cannot be produced by that ordinary spacing, and it is reached within seconds at any pop cadence, so the
hysteresis costs the worker almost nothing when the demand really has gone."""

_POP_CLAIM_EMPTY_POP_WINDOW_SECONDS = 10.0
"""How long the run of empty pops must have spanned before it counts as the demand having dried up.

The count alone is cadence-dependent: a worker popping every second reaches three empties in a gap far too
short to conclude anything from. Requiring the run to also cover a span makes the evidence a statement about
the horde's queue rather than about this worker's pop rate. Sized to the ordinary pop cadence of a few
seconds, so the run of three empties ends the claim about as soon as it is complete: every second the claim
outlives its demand is a second the whole pool is withheld from an idle card, and a single-job residency
already spends more time establishing and releasing than sampling.

Both rails are fixed rather than operator-configured: they describe how much evidence is needed to believe a
reading, which is not a preference, and the operator-facing lever over how long a residency may own the
intake is the maximum hold."""


class WholeCardGovernor(StrEnum):
    """The churn governors over a card taking on a *new* whole-card residency.

    Neither is a finding that the head cannot be served. The rate limiter defers the establishment until its
    window lets one through, and the head waits for the card meanwhile; the grace budget never defers, it only
    withholds the recovery supervisor's excuse window. Enumerating them keeps the set of churn brakes closed. A
    further one has to take a member, so it cannot be added without a declared release path in the gate registry.
    """

    ESTABLISH_RATE = "establish_rate"
    """The rolling per-card establishment rate limit is at its ceiling, so a new residency is deferred."""
    GRACE_BUDGET = "grace_budget"
    """The card has spent its rolling recovery-supervisor grace allowance, so a new establishment or restore does not
    open a window."""


class DominantRunEndCause(StrEnum):
    """Why a dominant-model run (rung 1) ended, as recorded in the action ledger's run-end event."""

    DRAINED = "drained"
    """The run's own jobs drained and nothing held the run through its cooldown."""
    STARVED_HEAD = "starved_head"
    """A head for a different model waited past the anti-starvation bound while the run still had work."""


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
class WholeCardGovernorHold:
    """A churn governor's refusal to open a new whole-card residency on one card, with its arithmetic."""

    governor: WholeCardGovernor
    """Which governor refused; keys the throttled disclosure so a changed objection always speaks."""
    reason: str
    """Operator-facing sentence naming the refusal; also the recorded defer reason."""
    detail: str | None = None
    """The governor's figures, when it has any to show."""


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

    The ends call for different responses, so they are distinguished rather than reported as one "claim over"
    event: a cap expiry says the burst outlasted the window the operator set, the empty-pop finding says the
    demand went away, a terminal fault says the claimed model's work is failing on the card, and a residency
    release says the card simply stopped holding the model. Enumerated so the status surfaces and the
    disclosure line name the same set.
    """

    MAXIMUM_HOLD = "maximum_hold"
    """The operator's maximum hold elapsed while the residency still held the card."""
    NO_FURTHER_WORK = "no_further_work"
    """The run of empty pops established that the resident model's demand had dried up."""
    RESIDENCY_RELEASED = "residency_released"
    """The residency behind the claim let go, and the claim went with it."""
    TERMINAL_FAULT = "terminal_fault"
    """A job for the claimed model faulted terminally. Narrowing the offer to a model whose jobs fault feeds the
    card more of the same failure, so the claim lets go at the first one."""


class DominanceRung(StrEnum):
    """How far a model dominates a card, given the card, the offered set, the footprints and the foreign floor.

    The rungs nest, since a model that cannot sample beside one context cannot sample beside another model
    either. The verdict is computed in ``scheduling.admission.pricing``, and the residency grant carries it.
    """

    NONE = "none"
    """Some offered model can sample beside it."""
    CANNOT_CO_SAMPLE = "cannot_co_sample"
    """Rung 1: no offered model, a second copy of itself included, can sample beside it."""
    CANNOT_CO_RESIDE = "cannot_co_reside"
    """Rung 2: its sampling peak beside one sibling context exceeds the card, so only the teardown seats it."""


@dataclass
class WholeCardResidency:
    """Mutable whole-card exclusive-residency state for one card (the worker, on a single-GPU host).

    The ledger keys one of these per device index so two heavy models on different cards each hold their
    own residency independently. A single-GPU worker keeps exactly one instance under the ``None`` key,
    so its behaviour is identical to the pre-multi-GPU scalar fields.
    """

    model: str | None = None
    """The model holding (or being given) sole residency on this card; None when no residency is held."""
    rung: DominanceRung = DominanceRung.CANNOT_CO_RESIDE
    """The dominance rung the grant was made on. Rung 2 carries the teardown (stopped siblings, safety off the
    card, paused lanes). Rung 1 is a run, which shares the claim and the cooldown and does not tear anything down."""
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
    establish_grace_withheld: bool = False
    """Whether this establishment asked for a grace window on a card whose allowance was spent, so none opened.

    The residency is still physically establishing (the phase reads so); only the recovery supervisor's excuse
    is withheld."""
    cooldown_until: float = 0.0
    """Wall-clock time until which this residency is held even after its heavy job drains, so a burst of
    heavy jobs reuses one residency instead of each churning a teardown/restore + safety cycle."""
    restore_at: float = 0.0
    """When this residency was last restored with a grace window (siblings respawned, safety cycled back
    on-GPU); 0.0 when none. The restore churn also briefly makes the queue unservable, so the wedge grace must
    cover it too. A restore on a card whose allowance is spent does not open a window and does not touch this stamp."""
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
    drain_reference_at: float = 0.0
    """When the free reading the drain is judged against was taken; 0.0 before the first post-structural reading.

    :meth:`WholeCardResidencyLedger.drain_settled` moves it forward whenever the reading rises, so the drain is
    settled once a later reading shows no rise across the spacing."""
    drain_reference_mb: float = 0.0
    """The free reading (MB) taken at ``drain_reference_at``."""
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
    """When the empty-pop evidence or a terminal fault ended this residency's claim over the offer; 0.0 while it
    still stands.

    Distinct from the maximum hold, which is a clock on the grant: this is a finding that the resident model
    has no demand left, or that its jobs fail on the card, so continuing to hold the pool to it would starve the
    worker on its own claim."""


class WholeCardGrantKind(StrEnum):
    """What a residency grant cost the card it was recorded against."""

    ESTABLISH = auto()
    """The card took the model on physically (it held nothing, or held something else), so the grant opened a
    grace window where the card's allowance had room, counted toward the establishment rate, and reset the holds
    that a teardown resets."""
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

    def teardown_held(self) -> list[tuple[int | None, WholeCardResidency]]:
        """Return the held residencies granted on rung 2, the ones whose readers tear the card down."""
        return [(index, state) for index, state in self.held() if state.rung is DominanceRung.CANNOT_CO_RESIDE]

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
        new_demand: bool = True,
        rung: DominanceRung = DominanceRung.CANNOT_CO_RESIDE,
    ) -> WholeCardGrantKind:
        """Record a residency grant (an establishment or a RAM pre-stage) for ``device_index``.

        Sets the model and cooldown, and returns which kind of grant this was. Everything else the grant
        touches is scoped to a *physical* grace window: the card taking the model on when it held nothing or
        held something else. That is what costs a teardown, opens the window the recovery supervisor honours,
        and resets the clocks a teardown resets, so it stamps ``established_at``, opens a fresh min-hold
        floor, clears the previous grant's structural-completion latch (a re-established residency tears down
        again, so it must not inherit an elapsed drain backstop), records the establishment for the rate
        limiter, and opens the ``establish_grace_seconds`` window :meth:`grace_seconds_to_open` allows, charging
        what it opened against the card's rolling grace budget. A card whose allowance is spent is still taken,
        without a window and without a charge. Passing zero seconds does not record a charge, which is what a
        caller not claiming a grace window should do.

        A further job asking for the model the card already holds is a reuse: it rides the residency the
        cooldown is deliberately keeping alive and physically costs nothing, so it only extends the cooldown.
        Charging it, counting it, or re-opening its establish window would spend an allowance sized for
        teardown churn on jobs that caused none, hand the supervisor a fresh excuse for a queue nothing is
        tearing down, and restart the drain backstop of a residency that has already converged. The grant
        forecast is likewise captured only for a physical grant, keeping it immutable across repeated asks.

        ``new_demand`` is False when the asker is a head that already asked and is re-asking while it waits
        for the residency to converge. Such a reuse is no new evidence of demand, so it leaves the empty-pop
        run alone; resetting the run on every re-ask would keep the claim from ever releasing while the head
        is parked.

        ``rung`` is the dominance rung the grant is made on. A rung-1 grant is a run and costs no teardown, so
        the run's establishment does not open a minimum hold and is not counted by the rate limiter. A rung-2
        ask for the model a run holds is the teardown starting, so it is an establishment and restarts every
        clock. A rung-1 ask never demotes a held rung-2 grant.
        """
        state = self.state_for(device_index)
        promoting_run = (
            state.model == model
            and state.rung is not DominanceRung.CANNOT_CO_RESIDE
            and rung is DominanceRung.CANNOT_CO_RESIDE
        )
        kind = (
            WholeCardGrantKind.REUSE
            if state.model is not None and state.model == model and state.established_at != 0.0 and not promoting_run
            else WholeCardGrantKind.ESTABLISH
        )
        # A job asking for this model is the direct contradiction of the empty-pop evidence, so the run restarts
        # from the demand that just arrived. An establishment additionally restarts the maximum hold, since the
        # clock measures one physical residency episode.
        if kind is WholeCardGrantKind.ESTABLISH or new_demand:
            state.pop_claim_empty_pops = 0
            state.pop_claim_empty_pop_since = 0.0
        if kind is WholeCardGrantKind.ESTABLISH:
            tears_down = rung is DominanceRung.CANNOT_CO_RESIDE
            state.established_at = now
            state.min_hold_until = now + _MIN_HOLD_SECONDS if tears_down else now
            state.structural_complete_at = 0.0
            state.drain_reference_at = 0.0
            state.drain_reference_mb = 0.0
            state.pop_claim_released_at = 0.0
            if tears_down:
                state.establishments.append(now)
            opened = self.grace_seconds_to_open(device_index, window_seconds=establish_grace_seconds, now=now)
            state.establish_grace_withheld = establish_grace_seconds > 0.0 and opened == 0.0
            if opened > 0.0:
                state.grace_charges.append((now, opened))
            elif state.establish_grace_withheld:
                self._disclose_grace_withheld(device_index, model=model, action="establishment", now=now)
            state.forecast = forecast
            state.repriced_target = forecast.max_resident_processes() if forecast is not None else None
        elif state.forecast is None:
            state.forecast = forecast
            state.repriced_target = forecast.max_resident_processes() if forecast is not None else None
        if kind is WholeCardGrantKind.ESTABLISH or rung is DominanceRung.CANNOT_CO_RESIDE:
            state.rung = rung
        state.model = model
        state.cooldown_until = cooldown_until
        return kind

    def record_restore(self, device_index: int | None, *, now: float, restore_grace_seconds: float = 0.0) -> None:
        """Clear a drained residency on ``device_index`` and open its restore window.

        The restore's own churn (respawning the stopped siblings, cycling safety back on-GPU) briefly makes
        the queue unservable, so the stamp is what the wedge grace reads; the grant's holds (min-hold,
        structural completion) are released with it, and the ``restore_grace_seconds`` window
        :meth:`grace_seconds_to_open` allows is opened and charged against the card's rolling grace budget. On a
        card whose allowance is spent the restore still runs, without a window and without a charge.

        The window is granted for churn, so a caller that finds it had none to do withdraws the grant with
        :meth:`close_restore_window`. That is a narrower grant, not a shortened one: a window this card is
        genuinely inside still runs its full duration and nothing cuts it short.
        """
        state = self.state_for(device_index)
        model = state.model
        _clear_grant(state)
        opened = self.grace_seconds_to_open(device_index, window_seconds=restore_grace_seconds, now=now)
        if restore_grace_seconds > 0.0 and opened == 0.0:
            self._disclose_grace_withheld(device_index, model=model, action="restore", now=now)
            return
        state.restore_at = now
        if opened > 0.0:
            state.grace_charges.append((now, opened))

    def record_run_end(self, device_index: int | None) -> None:
        """Clear a rung-1 run on ``device_index`` once its own work has drained.

        A run did not stop a sibling or pause a lane, so the run's end has no churn to cover. Unlike
        :meth:`record_restore`, it does not open a restore window and does not charge the grace budget.
        """
        state = self._residencies.get(device_index)
        if state is None:
            return
        _clear_grant(state)

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

    def note_terminal_fault(self, model: str | None, *, now: float) -> bool:
        """Release the claim a residency holds for ``model`` because one of its jobs faulted terminally.

        A fault of a model the card does not claim leaves every claim standing. The release ends the claim's
        retention as the empty-pop release does, so the residency lets go once its accepted work has drained.

        Returns:
            True when this fault released a standing claim, so the caller can name the end once.
        """
        found, device_index = self.holder_for_model(model)
        if not found:
            return False
        state = self._residencies[device_index]
        if state.pop_claim_released_at != 0.0 or state.established_at == 0.0:
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

    def governor_hold(self, device_index: int | None, *, now: float) -> WholeCardGovernorHold | None:
        """Return the churn governor barring a new residency on this card, or None when the card may take one.

        Only the establish-rate limiter bars the card, and only at admission; a held residency keeps converging.
        A spent grace allowance is not a bar. The establishment proceeds and does not open a window
        (:meth:`grace_seconds_to_open`).
        """
        if not self.establish_rate_exceeded(device_index, now=now):
            return None
        state = self._residencies.get(device_index)
        detail = None
        if state is not None and state.establishments:
            admitted_in = max(0.0, state.establishments[0] + _ESTABLISH_WINDOW_SECONDS - now)
            detail = (
                f"{len(state.establishments)} establishments inside the {_ESTABLISH_WINDOW_SECONDS:.0f}s window, "
                f"the next is admitted in {admitted_in:.0f}s"
            )
        return WholeCardGovernorHold(
            governor=WholeCardGovernor.ESTABLISH_RATE,
            reason="this card has already cycled whole-card residency as often as the rolling window allows",
            detail=detail,
        )

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
        return state.model, (
            WholeCardPhase.ESTABLISHING
            if _establishing(state, now=now, establish_grace_seconds=establish_grace_seconds)
            else WholeCardPhase.HOLDING
        )

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
        window's own duration, measured from the grant. How often a card may open a *new* window is decided
        when it opens, by :meth:`grace_seconds_to_open`, not here; an establishment that did not open one is not
        inside a window however recently it was stamped.

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

        While this is True the card does not open a fresh grace window. An establishment or restore still
        proceeds, without the recovery supervisor's excuse, until the spend replenishes. It does not describe a
        window already granted, which runs to its own duration regardless, and reuse of a residency the card
        already holds does not open a window either way.

        Spend is the sum of the grace windows granted to the card inside
        :data:`_GRACE_BUDGET_WINDOW_SECONDS`; it replenishes as those grants age out of the window. Spend equal
        to the allowance is the allowance used up.
        """
        state = self._residencies.get(device_index)
        if state is None:
            return False
        _prune_charges_before(state.grace_charges, cutoff=now - _GRACE_BUDGET_WINDOW_SECONDS)
        return sum(seconds for _granted_at, seconds in state.grace_charges) >= _GRACE_BUDGET_SECONDS

    def grace_seconds_to_open(self, device_index: int | None, *, window_seconds: float, now: float) -> float:
        """Return the grace seconds this card may open now for a window of ``window_seconds``.

        The whole window while the card has allowance left, and 0.0 once it has spent it. A window is opened
        whole or not at all, so the excuse the supervisor reads is always the nominal one or none. Every
        establish and restore opens its window through this, so the rule has one owner.
        """
        if window_seconds <= 0.0 or self.grace_budget_exhausted(device_index, now=now):
            return 0.0
        return window_seconds

    def grace_budget_status(self, device_index: int | None, *, now: float) -> GraceBudgetStatus:
        """Return this card's rolling grace spend, remaining allowance, and wait for the next replenishment.

        The disclosure companion to :meth:`grace_budget_exhausted`. A withheld window that quotes the arithmetic
        can be acted on, where a bare "withheld" cannot. The replenishment wait is derived from the charges
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
        if spent >= _GRACE_BUDGET_SECONDS:
            outstanding = spent
            for granted_at, seconds in state.grace_charges:
                outstanding -= seconds
                if outstanding < _GRACE_BUDGET_SECONDS:
                    replenish_in = max(0.0, (granted_at + _GRACE_BUDGET_WINDOW_SECONDS) - now)
                    break
        return GraceBudgetStatus(
            allowance_seconds=_GRACE_BUDGET_SECONDS,
            spent_seconds=spent,
            remaining_seconds=max(0.0, _GRACE_BUDGET_SECONDS - spent),
            replenish_in_seconds=replenish_in,
        )

    def _disclose_grace_withheld(
        self,
        device_index: int | None,
        *,
        model: str | None,
        action: Literal["establishment", "restore"],
        now: float,
    ) -> None:
        """Log that an establishment or restore went ahead with no grace window, quoting the spent allowance.

        Each such action is a physical rotation the rate limiter bounds, so the line cannot fire every tick.
        """
        status = self.grace_budget_status(device_index, now=now)
        logger.info(
            f"Whole-card residency {action} for {model} on card {device_index} does not open a "
            f"recovery-supervisor grace window: {status.spent_seconds:.0f}s of grace opened against a "
            f"{status.allowance_seconds:.0f}s rolling allowance, next replenish in {status.replenish_in_seconds:.0f}s."
        )

    def note_governor_defer(self, device_index: int | None, *, model: str | None, now: float) -> float:
        """Record that the establish-rate limiter is holding ``model`` off this card, and return the wait so far.

        The wait is anchored at the first of a run of deferrals for this model on this card and restarts
        whenever the deferred model changes, so one head's wait is never inherited by the next. A record older
        than :data:`_ESTABLISH_WINDOW_SECONDS` starts a new run, since no single deferral outlasts that window.
        """
        state = self.state_for(device_index)
        record_expired = (now - state.governor_deferred_since) >= _ESTABLISH_WINDOW_SECONDS
        if state.governor_deferred_model != model or state.governor_deferred_since == 0.0 or record_expired:
            state.governor_deferred_model = model
            state.governor_deferred_since = now
        return max(0.0, now - state.governor_deferred_since)

    def clear_governor_defer(self, device_index: int | None, *, now: float) -> None:
        """Forget any governor deferral recorded for this card, so the next one is timed from its own start.

        Clearing a recorded deferral logs how long the limiter held its model. The caller clears on every head
        the limiter lets through, so a card with no record logs nothing.
        """
        state = self._residencies.get(device_index)
        if state is None:
            return
        if state.governor_deferred_since != 0.0:
            logger.info(
                f"Whole-card governor deferral of {state.governor_deferred_model} on card {device_index} cleared "
                f"after {now - state.governor_deferred_since:.0f}s held.",
            )
        state.governor_deferred_model = None
        state.governor_deferred_since = 0.0

    def governor_deferred_head(self, device_index: int | None, *, now: float) -> str | None:
        """Return the model this card's governor deferral is still holding off, or None when none is in force.

        The record tells the recovery supervisor and head protection that the head stood down deliberately.
        It counts for at most :data:`_ESTABLISH_WINDOW_SECONDS` from the first deferral of the run, the longest
        the rate limiter can hold one head off, so a record left behind by a head that left the queue stops
        excusing the card on its own.
        """
        state = self._residencies.get(device_index)
        if state is None or state.governor_deferred_since == 0.0:
            return None
        if (now - state.governor_deferred_since) >= _ESTABLISH_WINDOW_SECONDS:
            return None
        return state.governor_deferred_model

    def any_governor_defer_active(self, *, now: float) -> bool:
        """Return whether any card holds a governor deferral record inside its bound."""
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
        """Return whether one card's residency sits inside a granted establish or restore window."""
        restoring = state.restore_at != 0.0 and (now - state.restore_at) < restore_grace_seconds
        establish_window_open = not state.establish_grace_withheld and _establishing(
            state,
            now=now,
            establish_grace_seconds=establish_grace_seconds,
        )
        return establish_window_open or restoring

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

    def drain_settled(
        self,
        device_index: int | None,
        *,
        now: float,
        free_mb: float | None,
        spacing_seconds: float,
        progress_epsilon_mb: float,
        upper_bound_seconds: float,
    ) -> bool:
        """Return whether the card's free reading has stopped rising since the teardown became structural.

        A stopped context's memory returns to the driver some time after its process exits, so the head waits
        for the free reading to stop climbing, never a fixed window that outlasts the drain: the drain is
        settled once a reading ``spacing_seconds`` after the reference shows no rise beyond
        ``progress_epsilon_mb``, and a larger rise moves the reference forward. Releasing the head hands any
        shortfall left on the card to clearance, which re-prices the load against the live reading.
        ``upper_bound_seconds`` (:meth:`drain_backstop_elapsed`) releases the head on a reading that never
        settles or never arrives.
        """
        if self.drain_backstop_elapsed(device_index, now=now, settle_seconds=upper_bound_seconds):
            return True
        state = self._residencies.get(device_index)
        if state is None or state.structural_complete_at == 0.0 or free_mb is None:
            return False
        if state.drain_reference_at == 0.0 or free_mb > state.drain_reference_mb + progress_epsilon_mb:
            state.drain_reference_at = now
            state.drain_reference_mb = free_mb
            return False
        return (now - state.drain_reference_at) >= spacing_seconds


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

    def set_forecast(self, device_index: int | None, forecast: StreamForecast | None) -> None:
        """Replace a card's establishing forecast and re-derive its process target from it.

        A grant records both together; this is the seam for replacing the forecast alone, which resets the
        tighten-only target rather than tightening it.
        """
        state = self.state_for(device_index)
        state.forecast = forecast
        state.repriced_target = self.target_process_count(forecast)

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
        immediately; otherwise a settled drain (``drain_backstop_elapsed``, which the caller derives from
        :meth:`drain_settled`) releases it on the forecast's sole-residency guarantee. That guarantee describes a
        card every other context has left, so ``resident_context_charge_mb``
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


def _establishing(state: WholeCardResidency, *, now: float, establish_grace_seconds: float) -> bool:
    """Return whether a held rung-2 residency is still inside its establish grace.

    A run tears nothing down, so it never holds the queue for a teardown and is never establishing.
    """
    return (
        state.model is not None
        and state.rung is DominanceRung.CANNOT_CO_RESIDE
        and state.established_at != 0.0
        and (now - state.established_at) < establish_grace_seconds
    )


def _clear_grant(state: WholeCardResidency) -> None:
    """Reset every field a grant sets, leaving the card's churn history (restore stamp, charges, establishments)."""
    state.model = None
    state.rung = DominanceRung.CANNOT_CO_RESIDE
    state.forecast = None
    state.repriced_target = None
    state.established_at = 0.0
    state.establish_grace_withheld = False
    state.min_hold_until = 0.0
    state.structural_complete_at = 0.0
    state.drain_reference_at = 0.0
    state.drain_reference_mb = 0.0
    state.prestage_process_id = None
    state.pop_claim_empty_pops = 0
    state.pop_claim_empty_pop_since = 0.0
    state.pop_claim_released_at = 0.0


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


class WholeCardPopClaimTracker:
    """Edge-triggered disclosure of the pop claim, and the memory of why the last claim ended.

    The claim itself is derived live from the residency ledger; this only remembers what was last surfaced so a
    claim held across many ticks is stated once, and a released one is stated once with the end that fired.
    """

    def __init__(self) -> None:
        """Start with no claim disclosed and no release remembered."""
        self._disclosed: WholeCardPopClaim | None = None
        self._early_release_pending: WholeCardPopClaimRelease | None = None
        self._release: tuple[WholeCardPopClaimRelease, float] | None = None
        self._multi_gpu_disclosed = False

    def note_empty_pop_release(self) -> None:
        """Remember that the empty-pop evidence ended the claim, so the next edge names that end."""
        self._early_release_pending = WholeCardPopClaimRelease.NO_FURTHER_WORK

    def note_terminal_fault_release(self) -> None:
        """Remember that a terminal fault of the claimed model ended the claim, so the next edge names that end."""
        self._early_release_pending = WholeCardPopClaimRelease.TERMINAL_FAULT

    def disclose_skipped_on_multi_gpu(self, *, any_held: bool) -> None:
        """State once that a residency on a multi-card host does not claim the worker-wide offer.

        Narrowing the offer to one card's model would starve the other cards; silence would read as the feature
        being broken rather than deliberately inapplicable.
        """
        if self._multi_gpu_disclosed or not any_held:
            return
        self._multi_gpu_disclosed = True
        logger.info(
            "Whole-card residency is held on a multi-card host, so it does not claim the pop offer: the offer "
            "is worker-wide and narrowing it to one card's model would starve the others.",
        )

    def disclose_edge(self, claim: WholeCardPopClaim | None, *, now: float) -> None:
        """Log the engage or release line when the standing claim actually changes."""
        previous = self._disclosed
        if claim is not None:
            if previous is not None and previous.model == claim.model:
                return
            self._disclosed = claim
            self._early_release_pending = None
            self._release = None
            logger.info(
                f"Whole-card pop claim engaged for {claim.model}: advertising that model alone while it holds "
                f"the card, for at most {claim.expires_at - claim.held_since:.0f}s.",
            )
            return
        if previous is None:
            return
        self._disclosed = None
        if self._early_release_pending is WholeCardPopClaimRelease.NO_FURTHER_WORK:
            release = WholeCardPopClaimRelease.NO_FURTHER_WORK
            reason = "the horde had no further work for it"
        elif self._early_release_pending is WholeCardPopClaimRelease.TERMINAL_FAULT:
            release = WholeCardPopClaimRelease.TERMINAL_FAULT
            reason = "a job for it faulted terminally"
        elif now >= previous.expires_at:
            release = WholeCardPopClaimRelease.MAXIMUM_HOLD
            reason = "the maximum hold elapsed"
        else:
            release = WholeCardPopClaimRelease.RESIDENCY_RELEASED
            reason = "the residency released"
        self._early_release_pending = None
        self._release = (release, now)
        logger.info(f"Whole-card pop claim released for {previous.model}: {reason}; advertising the full pool again.")

    def recent_release(self, now: float) -> WholeCardPopClaimRelease | None:
        """Why the last claim ended, while that still explains the widened offer; else None."""
        if self._release is None:
            return None
        release, released_at = self._release
        if (now - released_at) >= POP_CLAIM_RELEASE_VISIBLE_SECONDS:
            return None
        return release


def residency_has_holder(processes: Iterable[HordeProcessInfo], model: str, device_index: int | None) -> bool:
    """Whether a held whole-card model is staged or resident on a live inference process of the card.

    A holder that has finished loading may sit in WAITING_FOR_JOB rather than a preload state, so the generic
    forecast-to-load predicate is too narrow for convergence, which must spare that process from the shrink.
    """
    return any(
        p.process_type is HordeProcessType.INFERENCE
        and p.loaded_horde_model_name == model
        and (device_index is None or p.device_index == device_index)
        for p in processes
    )


def convergence_blockers(
    processes: Iterable[HordeProcessInfo],
    *,
    head_process_id: int,
    device_index: int | None,
    queued_models: set[str],
) -> list[tuple[int, str]]:
    """Idle sibling processes still holding a queued model while a whole-card head is parked.

    Convergence is meant to have torn these down, sparing only the head's holder, so finding any while the head
    is still parked is the fingerprint of a shrink that did not collapse the pool. Read-only diagnostics.
    """
    return [
        (p.process_id, p.loaded_horde_model_name)
        for p in processes
        if p.process_type is HordeProcessType.INFERENCE
        and p.process_id != head_process_id
        and (device_index is None or p.device_index == device_index)
        and not p.is_process_busy()
        and p.loaded_horde_model_name is not None
        and p.loaded_horde_model_name in queued_models
    ]


def post_process_context_fits(forecast: StreamForecast, target: int | None) -> bool:
    """Whether the residency model can load with the post-processing lane's bare context still on the card.

    Unknown weights, total or target read as fitting: the gate only ever pauses the lane on evidence.
    """
    if forecast.weights_mb is None or forecast.total_vram_mb is None or target is None:
        return True
    marginal = forecast._effective_marginal_overhead_mb  # noqa: SLF001 - same budget object owns the estimate.
    extra_contexts = max(0, target - 1) + 1  # surviving inference siblings plus the lane context
    free_with_pp_lane_mb = max(
        0.0,
        float(forecast.total_vram_mb) - forecast.per_process_overhead_mb - marginal * extra_contexts,
    )
    return (free_with_pp_lane_mb - forecast.weights_mb) >= forecast._effective_base_reserve  # noqa: SLF001
