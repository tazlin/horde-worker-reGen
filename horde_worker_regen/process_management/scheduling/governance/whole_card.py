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

__all__ = [
    "WholeCardPhase",
    "WholeCardResidency",
    "WholeCardResidencyLedger",
    "WholeCardResidencyMachine",
    "max_coresident_for_peak",
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
        refresh_established: bool,
        establish_grace_seconds: float = 0.0,
    ) -> WholeCardResidency:
        """Record a residency grant (an establishment or a RAM pre-stage) for ``device_index``.

        Sets the model and cooldown; captures the grant forecast only when the residency is fresh, keeping it
        immutable through repeated asks for the same held model. Stamps ``established_at`` when
        ``refresh_established`` is set or the residency is fresh, so the recovery supervisor's grace window is
        measured from when the intentional hold began. A stamped establishment also opens a fresh min-hold
        floor and clears any structural-completion latch from the previous grant. A re-established residency
        tears down again, so it must not inherit an elapsed drain backstop. The stamp also records the
        establishment for the rate limiter and
        charges ``establish_grace_seconds`` against the card's rolling grace budget. Passing zero seconds
        records no charge, which is what a caller that is not claiming a grace window should do.

        Returns the updated state.
        """
        state = self.state_for(device_index)
        fresh_grant = state.model is None or state.model != model
        if refresh_established or state.established_at == 0.0:
            state.established_at = now
            state.min_hold_until = now + _MIN_HOLD_SECONDS
            state.structural_complete_at = 0.0
            state.establishments.append(now)
            if establish_grace_seconds > 0.0:
                state.grace_charges.append((now, establish_grace_seconds))
        if fresh_grant or state.forecast is None:
            state.forecast = forecast
            state.repriced_target = forecast.max_resident_processes() if forecast is not None else None
        state.model = model
        state.cooldown_until = cooldown_until
        return state

    def record_restore(self, device_index: int | None, *, now: float, restore_grace_seconds: float = 0.0) -> None:
        """Clear a drained residency on ``device_index`` and open its restore window.

        The restore's own churn (respawning the stopped siblings, cycling safety back on-GPU) briefly makes
        the queue unservable, so the stamp is what the wedge grace reads; the grant's holds (min-hold,
        structural completion) are released with it and ``restore_grace_seconds`` is charged against the
        card's rolling grace budget.
        """
        state = self.state_for(device_index)
        state.model = None
        state.forecast = None
        state.repriced_target = None
        state.established_at = 0.0
        state.min_hold_until = 0.0
        state.structural_complete_at = 0.0
        state.restore_at = now
        if restore_grace_seconds > 0.0:
            state.grace_charges.append((now, restore_grace_seconds))

    def min_hold_active(self, device_index: int | None, *, now: float) -> bool:
        """Return whether this card's residency is still inside its non-preemptable minimum hold."""
        state = self._residencies.get(device_index)
        if state is None or state.model is None:
            return False
        return now < state.min_hold_until

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
    ) -> bool:
        """Return whether a job should enter the whole-card residency pipeline."""
        needs_teardown = forecast.needs_exclusive_residency or forecast.needs_process_count_reduction
        return enabled and needs_teardown and is_head_blocker

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


def _prune_before(stamps: deque[float], *, cutoff: float) -> None:
    """Drop timestamps older than ``cutoff`` from the front of an oldest-first deque."""
    while stamps and stamps[0] < cutoff:
        stamps.popleft()


def _prune_charges_before(charges: deque[tuple[float, float]], *, cutoff: float) -> None:
    """Drop ``(granted_at, seconds)`` charges granted before ``cutoff`` from an oldest-first deque."""
    while charges and charges[0][0] < cutoff:
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
