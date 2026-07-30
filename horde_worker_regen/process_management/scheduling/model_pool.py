"""Fixed model pool: a deterministic seat/bench/decay engine for a persistently-served model set.

A worker with a fixed pool keeps a bounded number of "seats", each holding one model the worker commits to
serving for a while, so the horde returns work the card can run without a per-job model swap. Seats are filled
first by operator manual pins (in affinity order) and then, when a ranker is enabled, by the highest-scoring
ranker candidates. A seat is held for a minimum dwell, then becomes eligible for rotation: a timed re-contest
against the best challenger, or a demotion when the seat stops earning its place (sustained empty pops against
near-zero demand, or a stretch with no fulfillment at all). A demoted model benches with a cooldown so it does
not immediately re-seat and thrash. An opt-in rescue path lets a starved, high-wait model briefly claim the
weakest ranker seat.

The engine is pure and table-testable: it imports only the standard library, ``strenum``, and the sibling
scheduling value objects. It never imports the process manager, the popper, the scheduler, ``torch``, or
``hordelib``, and it never reads the clock. Every method that advances state is handed a monotonic ``now`` by
the caller, so a test drives time explicitly and the same inputs always produce the same transitions. The
caller owns the mutable :class:`ModelPool`, feeds it pop outcomes and download results, ticks it against a
freshly ranked candidate set, and logs or ledgers the returned :class:`SeatTransition` list.

The engine is deliberately unaware of the download budget and of how candidates are ranked or how demand is
measured: those are caller concerns. It consumes a :class:`RankedCandidate` sequence (name, score, on-disk,
optional wait ETA, optional per-worker queue depth) and emits seat decisions; a candidate that is not yet on
disk becomes a pending-download seat the caller resolves with :meth:`ModelPool.on_download_ready` or
:meth:`ModelPool.on_download_failed`.

Public surface: :class:`ModelPool`, :class:`PoolParams`, :class:`PinnedModel`, :class:`RankedCandidate`,
:class:`SeatTransition`, :class:`SeatView`, :class:`BenchView`, and the enums :class:`SeatSource`,
:class:`SeatState`, :class:`DemotionReason`, :class:`TransitionKind`, :class:`PopLane`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import auto

from strenum import StrEnum

__all__ = [
    "BenchView",
    "DemotionReason",
    "ModelPool",
    "PinnedModel",
    "PoolParams",
    "PopLane",
    "RankedCandidate",
    "SeatSource",
    "SeatState",
    "SeatTransition",
    "SeatView",
    "TransitionKind",
]

_SECONDS_PER_MINUTE = 60.0
_SECONDS_PER_HOUR = 3600.0

_NEAR_ZERO_QUEUE_PER_WORKER = 1.0
"""At or below this queued-jobs-per-worker figure a model's demand counts as near-zero for empty-pop demotion.

The empty-pop counter alone cannot tell a genuinely dead model from one the worker simply keeps losing pops
for while real demand exists, so an empty-pop demotion additionally requires the caller-supplied per-worker
queue depth to sit at or below one queued job per worker. A model with no such demand signal this tick is
treated as unknown rather than near-zero, so it is not demoted on empty-pop grounds until its demand is
observed to have dried up.
"""


class SeatSource(StrEnum):
    """How a seat's current model came to hold the seat."""

    MANUAL = auto()
    RANKER = auto()
    RESCUE = auto()


class SeatState(StrEnum):
    """Whether a seat is serving its model or is mid-download to swap it."""

    ACTIVE = auto()
    PENDING_DOWNLOAD = auto()


class DemotionReason(StrEnum):
    """Why a seat's model left the seat (or, for pressure, why it was recorded but kept)."""

    EMPTY_POPS = auto()
    ZERO_FULFILLMENT = auto()
    TIMER_LOST = auto()
    RESCUE_DISPLACED = auto()
    DOWNLOAD_FAILED = auto()
    CONFIG_CHANGED = auto()
    PRESSURE_NOTED = auto()


class TransitionKind(StrEnum):
    """The kind of seat change a :class:`SeatTransition` records."""

    SEATED = auto()
    DEMOTED = auto()
    RESCUE_ENGAGED = auto()
    RESCUE_RELEASED = auto()
    DOWNLOAD_PENDING = auto()
    DOWNLOAD_READY = auto()


class PopLane(StrEnum):
    """Which advertising lane a pop outcome came from.

    ``FIXED`` is the pool-narrowed lane whose empty pops drive seat demotion; ``FREE`` is the wider,
    non-pool lane whose empty pops never charge a seat (though a fulfillment on either lane still credits
    a seated model).
    """

    FIXED = auto()
    FREE = auto()


@dataclass(frozen=True)
class PinnedModel:
    """Represents an operator manual pin: a model name and its persistent seating bias.

    The affinity is a bias, never a lock. It orders which pins fill empty seats first (highest first), extends
    the pinned model's rotation deadline, and adds a small bonus to its re-contest score, so a pinned model is
    sticky but can still yield its seat to a sufficiently stronger challenger and re-seat later.
    """

    name: str
    affinity: float = 1.0


@dataclass(frozen=True)
class PoolParams:
    """Represents the frozen configuration and tunables the engine reconciles against.

    Built at wiring time from operator config (the engine never reads ``bridge_data``). ``seat_count`` is the
    resolved number of seats. ``pinned`` is the ordered manual-pin set. ``ranker_enabled`` gates ranker fills
    and re-contests. The minute-valued rotation and dwell figures, the rescue window and ETA, and the
    tunables below govern the seat/bench/decay dynamics.
    """

    seat_count: int
    pinned: tuple[PinnedModel, ...] = ()
    ranker_enabled: bool = True
    rotation_minutes: float = 60.0
    min_dwell_minutes: float = 10.0
    rescue_enabled: bool = False
    rescue_eta_seconds: float = 10000.0
    rescue_window_minutes: float = 15.0
    rotation_margin: float = 0.25
    affinity_bonus_weight: float = 0.3
    empty_pop_demotion_threshold: int = 40
    zero_fulfillment_demotion_minutes: float = 15.0
    bench_cooldown_empty_pops_minutes: float = 20.0
    bench_cooldown_timer_minutes: float = 5.0
    rescue_model_cooldown_hours: float = 6.0


@dataclass(frozen=True)
class RankedCandidate:
    """Represents one scored model the caller's ranker offers the pool this tick.

    ``score`` orders candidates for filling and re-contest (higher wins). ``on_disk`` decides whether a
    chosen candidate seats immediately or becomes a pending download. ``eta_seconds`` is the estimated wait
    a requester currently faces for this model (high means starved, which the rescue path targets), and
    ``queued_per_worker`` is the demand signal the empty-pop demotion consults for near-zero queue depth.
    Both optional figures are ``None`` when the caller has no measurement this tick.
    """

    name: str
    score: float
    on_disk: bool
    eta_seconds: float | None = None
    queued_per_worker: float | None = None


@dataclass(frozen=True)
class SeatTransition:
    """Represents one seat change for the caller to log or ledger.

    ``reason`` is set only for :attr:`TransitionKind.DEMOTED`; ``source`` is the seat source the change
    concerns (the intended source for a pending download, the resolved source for a ready one). ``at`` is the
    monotonic time the caller passed in.
    """

    kind: TransitionKind
    model: str
    source: SeatSource | None
    reason: DemotionReason | None
    at: float


@dataclass(frozen=True)
class SeatView:
    """Represents an immutable snapshot of one seat for the caller to inspect."""

    model: str | None
    source: SeatSource | None
    state: SeatState
    seated_at: float
    last_fulfilled_at: float | None
    last_match_was_resident: bool | None
    empty_pops: int
    pending_model: str | None
    rescue_expires_at: float | None


@dataclass(frozen=True)
class BenchView:
    """Represents an immutable snapshot of one benched model and its cooldown."""

    model: str
    cooldown_until: float
    reason: DemotionReason


@dataclass
class _Seat:
    """Mutable per-seat state. Private to the engine; exposed only through :class:`SeatView` copies."""

    model: str | None = None
    source: SeatSource | None = None
    state: SeatState = SeatState.ACTIVE
    seated_at: float = 0.0
    last_fulfilled_at: float | None = None
    last_match_was_resident: bool | None = None
    empty_pops: int = 0
    score: float = 0.0
    affinity: float = 0.0
    timer_deadline: float | None = None
    rescue_expires_at: float | None = None
    pending_model: str | None = None
    pending_source: SeatSource | None = None
    pending_affinity: float = 0.0


@dataclass
class _BenchEntry:
    """Mutable bench record: a model held out of seating until its cooldown elapses."""

    cooldown_until: float
    reason: DemotionReason


class ModelPool:
    """A deterministic fixed-pool seat engine driven by caller-injected time.

    Holds ``seat_count`` seats, a bench of cooled-down models, and per-model download and rescue cooldowns.
    Seats fill from manual pins (affinity order) then ranker candidates; they hold for a minimum dwell, then
    rotate by timed re-contest or demote on sustained non-productivity. All mutation flows through the public
    methods, each of which takes the caller's monotonic ``now`` and returns the transitions it produced, so
    the engine never reads the clock and the same call sequence always yields the same result.

    Thread Safety:
        Not thread-safe. The owning caller serializes all access (the popper's single event loop in practice).
    """

    def __init__(self, params: PoolParams) -> None:
        """Build a pool with all seats empty, ready to fill on the first :meth:`tick`."""
        self._params = params
        self._seats: list[_Seat] = [_Seat() for _ in range(max(0, params.seat_count))]
        self._bench: dict[str, _BenchEntry] = {}
        self._download_cooldowns: dict[str, float] = {}
        self._rescue_cooldowns: dict[str, float] = {}

    # -- Read accessors --

    def active_seat_models(self) -> frozenset[str]:
        """Return the set of models currently serving a seat (a pending download's incumbent still serves)."""
        return frozenset(seat.model for seat in self._seats if seat.model is not None)

    def pending_download_models(self) -> frozenset[str]:
        """Return every model a seat is currently waiting to download, for the caller to command and protect.

        The caller reads this to issue the actual fetch for each pending target and to union those targets into
        any authoritative desired-model set it sends the download subsystem, so a config reconciliation does not
        prune a pool-initiated download out from under a seat.
        """
        return frozenset(seat.pending_model for seat in self._seats if seat.pending_model is not None)

    def seats(self) -> tuple[SeatView, ...]:
        """Return an immutable snapshot of every seat in seat order."""
        return tuple(
            SeatView(
                model=seat.model,
                source=seat.source,
                state=seat.state,
                seated_at=seat.seated_at,
                last_fulfilled_at=seat.last_fulfilled_at,
                last_match_was_resident=seat.last_match_was_resident,
                empty_pops=seat.empty_pops,
                pending_model=seat.pending_model,
                rescue_expires_at=seat.rescue_expires_at,
            )
            for seat in self._seats
        )

    def bench(self) -> tuple[BenchView, ...]:
        """Return an immutable snapshot of every benched model and its cooldown."""
        return tuple(
            BenchView(model=model, cooldown_until=entry.cooldown_until, reason=entry.reason)
            for model, entry in self._bench.items()
        )

    # -- Configuration reconciliation --

    def replace_params(self, params: PoolParams, now: float) -> list[SeatTransition]:
        """Adopt a new configuration, reconciling seats to it and returning the transitions that caused.

        Reconciles two kinds of change. A changed manual-pin set demotes any seat holding a model no longer
        pinned as a manual seat; a seat whose model is (still or newly) pinned keeps serving with its manual
        identity and affinity refreshed and its rotation deadline re-anchored. A changed seat
        count grows the pool with empty seats or shrinks it, releasing the least-committed seats first (empty,
        then rescue, then the lowest-scoring ranker, then the lowest-affinity manual). Reconciliation demotions
        do not bench their model, since the model left for a configuration reason rather than for failing to
        earn its seat; the freed model is eligible to seat again immediately.
        """
        transitions: list[SeatTransition] = []
        pinned_by_name = {pin.name: pin for pin in params.pinned}
        self._params = params

        for seat in self._seats:
            if seat.model is None:
                continue
            pin = pinned_by_name.get(seat.model)
            if seat.source is SeatSource.MANUAL and pin is None:
                transitions.append(
                    self._demote_seat(seat, reason=DemotionReason.CONFIG_CHANGED, now=now, bench_cooldown_minutes=None)
                )
            elif pin is not None and seat.source in (SeatSource.MANUAL, SeatSource.RANKER):
                # A newly-pinned model already holding a seat adopts its manual identity in place, and an
                # affinity change re-anchors the rotation deadline so the bias takes effect immediately.
                seat.source = SeatSource.MANUAL
                seat.affinity = pin.affinity
                seat.timer_deadline = self._rotation_deadline(seat.seated_at, pin.affinity)

        transitions.extend(self._resize_seats(params.seat_count, now=now))
        return transitions

    def _resize_seats(self, target_count: int, now: float) -> list[SeatTransition]:
        """Grow the seat list with empty seats or shrink it, releasing the least-committed seats first."""
        current_count = len(self._seats)
        if target_count > current_count:
            self._seats.extend(_Seat() for _ in range(target_count - current_count))
            return []
        if target_count == current_count:
            return []

        removals_needed = current_count - target_count
        removal_order = sorted(range(current_count), key=self._seat_release_priority)
        indices_to_remove = set(removal_order[:removals_needed])

        transitions: list[SeatTransition] = []
        kept_seats: list[_Seat] = []
        for seat_index, seat in enumerate(self._seats):
            if seat_index not in indices_to_remove:
                kept_seats.append(seat)
                continue
            if seat.model is not None:
                transitions.append(
                    self._demote_seat(
                        seat, reason=DemotionReason.CONFIG_CHANGED, now=now, bench_cooldown_minutes=None
                    ),
                )
        self._seats = kept_seats
        return transitions

    def _seat_release_priority(self, seat_index: int) -> tuple[int, float]:
        """Return a sort key ordering seats from most-releasable (empty) to least (a high-affinity pin)."""
        seat = self._seats[seat_index]
        if seat.model is None:
            return (0, 0.0)
        if seat.source is SeatSource.RESCUE:
            return (1, 0.0)
        if seat.source is SeatSource.RANKER:
            return (2, seat.score)
        return (3, seat.affinity)

    # -- Outcome and download callbacks --

    def on_pop_outcome(
        self,
        *,
        lane: PopLane,
        advertised: frozenset[str],
        popped_model: str | None,
        popped_model_was_resident: bool = False,
        now: float,
    ) -> None:
        """Record a pop outcome against the seats, updating the signals a later :meth:`tick` acts on.

        A fixed-lane empty pop charges one empty-pop count to every seated model that was advertised, since the
        horde returned no work for the narrowed offer. A match (either lane) credits the seated model: its
        empty-pop count resets, its last-match time is stamped, and whether it was resident at pop time is
        retained separately. Free-lane empty pops charge nothing. This method never seats or demotes.
        """
        if popped_model is not None:
            for seat in self._seats:
                if seat.model == popped_model:
                    seat.empty_pops = 0
                    seat.last_fulfilled_at = now
                    seat.last_match_was_resident = popped_model_was_resident
            return

        if lane is not PopLane.FIXED:
            return

        for seat in self._seats:
            if seat.model is not None and seat.model in advertised:
                seat.empty_pops += 1

    def on_download_ready(self, model: str, now: float) -> list[SeatTransition]:
        """Swap a completed pending download into its seat, benching the incumbent it replaces.

        The pending model becomes the seat's active model; any incumbent it displaced benches with the
        re-contest cooldown, since a pending download only ever attaches to a seat a challenger won on the
        timer. An empty seat that was reserved for the download simply activates with no incumbent to bench.
        """
        transitions: list[SeatTransition] = []
        for seat in self._seats:
            if seat.pending_model != model:
                continue
            if seat.model is not None:
                transitions.append(
                    self._bench_and_clear(
                        seat,
                        reason=DemotionReason.TIMER_LOST,
                        now=now,
                        bench_cooldown_minutes=self._params.bench_cooldown_timer_minutes,
                    ),
                )
            self._activate_seat(
                seat,
                model=model,
                source=seat.pending_source if seat.pending_source is not None else SeatSource.RANKER,
                affinity=seat.pending_affinity,
                now=now,
            )
            self._clear_pending(seat)
            transitions.append(
                SeatTransition(
                    kind=TransitionKind.DOWNLOAD_READY, model=model, source=seat.source, reason=None, at=now
                ),
            )
            break
        return transitions

    def on_download_failed(self, model: str, now: float) -> list[SeatTransition]:
        """Abandon a failed pending download, leaving its incumbent in place and cooling the failed model.

        The seat keeps whatever it was serving (an incumbent, or empty), the pending target is cleared, and
        the failed model is held out of seating for an hour so a broken download does not retry every tick.
        """
        transitions: list[SeatTransition] = []
        for seat in self._seats:
            if seat.pending_model != model:
                continue
            failed_source = seat.pending_source
            self._clear_pending(seat)
            self._download_cooldowns[model] = now + _SECONDS_PER_HOUR
            transitions.append(
                SeatTransition(
                    kind=TransitionKind.DEMOTED,
                    model=model,
                    source=failed_source,
                    reason=DemotionReason.DOWNLOAD_FAILED,
                    at=now,
                ),
            )
            break
        return transitions

    def on_pressure_eviction(self, model: str, now: float) -> list[SeatTransition]:
        """Record an external VRAM-pressure eviction of a seated model without unseating it.

        Pressure never unseats: the seat survives so the pool keeps committing to the model, and the event is
        surfaced as a noted-but-kept transition for the caller to observe. If the model holds no seat, nothing
        is recorded.
        """
        for seat in self._seats:
            if seat.model == model:
                return [
                    SeatTransition(
                        kind=TransitionKind.DEMOTED,
                        model=model,
                        source=seat.source,
                        reason=DemotionReason.PRESSURE_NOTED,
                        at=now,
                    ),
                ]
        return []

    # -- The periodic advance --

    def tick(
        self,
        now: float,
        *,
        ranked: Sequence[RankedCandidate] | None,
        demand_is_stale: bool,
    ) -> list[SeatTransition]:
        """Advance the pool one step against a fresh ranking, returning every seat change it produced.

        In order: release any rescue seat whose window closed or whose demand recovered; run timer re-contests;
        run empty-pop and zero-fulfillment demotions; fill empty seats from manual pins and then ranker
        candidates; engage a rescue if one is warranted. When ``demand_is_stale`` the ranking cannot be
        trusted, so rotations, re-contests, and both rescue engagement and release are frozen (a bench-hold);
        only genuinely empty seats are still filled, and only from manual pins, so the pool never sits emptier
        than the operator explicitly asked for.

        Args:
            now: The caller's monotonic time.
            ranked: The ranker's scored candidates this tick, or ``None`` when no ranking is available.
            demand_is_stale: Whether the demand measurement backing ``ranked`` is too old to act on.

        Returns:
            The transitions produced, in the order they occurred.
        """
        self._purge_expired(now)
        candidates = list(ranked) if ranked is not None else []
        ranked_by_name = {candidate.name: candidate for candidate in candidates}
        self._refresh_seat_scores(ranked_by_name)

        transitions: list[SeatTransition] = []

        if not demand_is_stale:
            transitions.extend(self._release_rescues(ranked_by_name, now=now))
            transitions.extend(self._run_recontests(candidates, ranked_by_name, now=now))
            transitions.extend(self._run_demotions(ranked_by_name, now=now))

        transitions.extend(self._fill_manual(now=now))

        if not demand_is_stale and self._params.ranker_enabled:
            transitions.extend(self._fill_ranker(candidates, now=now))

        if not demand_is_stale and self._params.rescue_enabled:
            transitions.extend(self._engage_rescue(candidates, now=now))

        return transitions

    # -- Internal helpers --

    def _purge_expired(self, now: float) -> None:
        """Drop bench and cooldown records whose hold has elapsed so the accessors and checks stay current."""
        self._bench = {model: entry for model, entry in self._bench.items() if entry.cooldown_until > now}
        self._download_cooldowns = {model: until for model, until in self._download_cooldowns.items() if until > now}
        self._rescue_cooldowns = {model: until for model, until in self._rescue_cooldowns.items() if until > now}

    def _refresh_seat_scores(self, ranked_by_name: dict[str, RankedCandidate]) -> None:
        """Refresh each seated model's score from this tick's ranking so re-contests use current demand."""
        for seat in self._seats:
            if seat.model is not None and seat.model in ranked_by_name:
                seat.score = ranked_by_name[seat.model].score

    def _committed_models(self) -> set[str]:
        """Return every model a seat is serving or downloading, so a fill never double-seats a model."""
        committed: set[str] = set()
        for seat in self._seats:
            if seat.model is not None:
                committed.add(seat.model)
            if seat.pending_model is not None:
                committed.add(seat.pending_model)
        return committed

    def _is_seatable(self, model: str, now: float, committed: set[str]) -> bool:
        """Return whether a model may take a seat now (not already committed, benched, or download-cooled)."""
        if model in committed:
            return False
        if model in self._download_cooldowns and self._download_cooldowns[model] > now:
            return False
        bench_entry = self._bench.get(model)
        return bench_entry is None or bench_entry.cooldown_until <= now

    def _seating_identity(self, model: str) -> tuple[SeatSource, float]:
        """Return the source and affinity a model seats with, preserving manual-pin identity everywhere.

        A pinned model keeps its manual source and affinity no matter which path seats it (manual fill, ranker
        fill, or a won re-contest), so the operator's bias survives the model re-earning its seat through the
        ranker. Any other model seats as a plain ranker candidate.
        """
        for pin in self._params.pinned:
            if pin.name == model:
                return (SeatSource.MANUAL, pin.affinity)
        return (SeatSource.RANKER, 0.0)

    def _rescue_seat_exists(self) -> bool:
        """Return whether any seat is serving or downloading a rescue model; at most one rescue may hold."""
        return any(
            (seat.source is SeatSource.RESCUE and seat.model is not None) or seat.pending_source is SeatSource.RESCUE
            for seat in self._seats
        )

    def _past_dwell(self, seat: _Seat, now: float) -> bool:
        """Return whether a seat has served past its minimum dwell and so may be rotated or demoted."""
        return (now - seat.seated_at) >= self._params.min_dwell_minutes * _SECONDS_PER_MINUTE

    def _rotation_deadline(self, seated_at: float, affinity: float) -> float:
        """Return the monotonic time a seat becomes due for a timed re-contest, extended by affinity."""
        return seated_at + self._params.rotation_minutes * (1.0 + affinity) * _SECONDS_PER_MINUTE

    def _activate_seat(self, seat: _Seat, *, model: str, source: SeatSource, affinity: float, now: float) -> None:
        """Seat a model into a seat as its active model, resetting the per-seat productivity and timer state."""
        seat.model = model
        seat.source = source
        seat.state = SeatState.ACTIVE
        seat.seated_at = now
        seat.last_fulfilled_at = None
        seat.last_match_was_resident = None
        seat.empty_pops = 0
        seat.affinity = affinity
        seat.timer_deadline = self._rotation_deadline(now, affinity)
        seat.rescue_expires_at = (
            now + self._params.rescue_window_minutes * _SECONDS_PER_MINUTE if source is SeatSource.RESCUE else None
        )

    def _clear_pending(self, seat: _Seat) -> None:
        """Clear a seat's pending-download target and return it to the active state."""
        seat.pending_model = None
        seat.pending_source = None
        seat.pending_affinity = 0.0
        seat.state = SeatState.ACTIVE

    def _bench_model(self, model: str, *, reason: DemotionReason, now: float, cooldown_minutes: float) -> None:
        """Hold a model out of seating for a cooldown so a just-demoted model does not immediately re-seat."""
        self._bench[model] = _BenchEntry(cooldown_until=now + cooldown_minutes * _SECONDS_PER_MINUTE, reason=reason)

    def _empty_active_seat(self, seat: _Seat) -> None:
        """Reset a seat's active model to empty, leaving any pending-download target untouched."""
        seat.model = None
        seat.source = None
        seat.state = SeatState.PENDING_DOWNLOAD if seat.pending_model is not None else SeatState.ACTIVE
        seat.seated_at = 0.0
        seat.last_fulfilled_at = None
        seat.last_match_was_resident = None
        seat.empty_pops = 0
        seat.score = 0.0
        seat.affinity = 0.0
        seat.timer_deadline = None
        seat.rescue_expires_at = None

    def _demote_seat(
        self,
        seat: _Seat,
        *,
        reason: DemotionReason,
        now: float,
        bench_cooldown_minutes: float | None,
    ) -> SeatTransition:
        """Empty a seat's active model, optionally benching it, and return the demotion transition."""
        model = seat.model
        source = seat.source
        if model is None:
            raise ValueError("cannot demote an empty seat")
        if bench_cooldown_minutes is not None:
            self._bench_model(model, reason=reason, now=now, cooldown_minutes=bench_cooldown_minutes)
        self._empty_active_seat(seat)
        return SeatTransition(kind=TransitionKind.DEMOTED, model=model, source=source, reason=reason, at=now)

    def _bench_and_clear(
        self,
        seat: _Seat,
        *,
        reason: DemotionReason,
        now: float,
        bench_cooldown_minutes: float,
    ) -> SeatTransition:
        """Bench a seat's incumbent and empty its active slot, used when a ready download replaces it."""
        return self._demote_seat(seat, reason=reason, now=now, bench_cooldown_minutes=bench_cooldown_minutes)

    def _release_rescues(self, ranked_by_name: dict[str, RankedCandidate], now: float) -> list[SeatTransition]:
        """Release rescue seats whose window closed or whose model's wait dropped back below the rescue ETA."""
        transitions: list[SeatTransition] = []
        for seat in self._seats:
            if seat.source is not SeatSource.RESCUE or seat.model is None:
                continue
            window_closed = seat.rescue_expires_at is not None and now >= seat.rescue_expires_at
            candidate = ranked_by_name.get(seat.model)
            demand_recovered = (
                candidate is not None
                and candidate.eta_seconds is not None
                and candidate.eta_seconds < self._params.rescue_eta_seconds
            )
            if not window_closed and not demand_recovered:
                continue
            released_model = seat.model
            self._empty_active_seat(seat)
            transitions.append(
                SeatTransition(
                    kind=TransitionKind.RESCUE_RELEASED,
                    model=released_model,
                    source=SeatSource.RESCUE,
                    reason=None,
                    at=now,
                ),
            )
        return transitions

    def _run_recontests(
        self,
        candidates: Sequence[RankedCandidate],
        ranked_by_name: dict[str, RankedCandidate],
        now: float,
    ) -> list[SeatTransition]:
        """Re-contest each dwell-eligible seat at its deadline, displacing it only for a margin-beating win."""
        if not self._params.ranker_enabled:
            return []

        transitions: list[SeatTransition] = []
        for seat in self._seats:
            if seat.model is None or seat.pending_model is not None:
                continue
            if seat.timer_deadline is None or now < seat.timer_deadline:
                continue
            if not self._past_dwell(seat, now):
                seat.timer_deadline = self._rotation_deadline(now, seat.affinity)
                continue

            committed = self._committed_models()
            challenger = self._best_challenger(candidates, now=now, committed=committed)
            incumbent_effective = seat.score + seat.affinity * self._params.affinity_bonus_weight
            required_score = incumbent_effective * (1.0 + self._params.rotation_margin)

            if challenger is None or challenger.score < required_score:
                seat.timer_deadline = self._rotation_deadline(now, seat.affinity)
                continue

            transitions.extend(self._displace_with_challenger(seat, challenger, now=now))
        return transitions

    def _best_challenger(
        self,
        candidates: Sequence[RankedCandidate],
        *,
        now: float,
        committed: set[str],
    ) -> RankedCandidate | None:
        """Return the highest-scoring seatable candidate not already committed, or ``None`` if none qualify."""
        best: RankedCandidate | None = None
        for candidate in candidates:
            if not self._is_seatable(candidate.name, now, committed):
                continue
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def _displace_with_challenger(self, seat: _Seat, challenger: RankedCandidate, now: float) -> list[SeatTransition]:
        """Hand a seat to a winning challenger, seating it now if on disk or marking a pending download."""
        challenger_source, challenger_affinity = self._seating_identity(challenger.name)
        if challenger.on_disk:
            transitions = [
                self._demote_seat(
                    seat,
                    reason=DemotionReason.TIMER_LOST,
                    now=now,
                    bench_cooldown_minutes=self._params.bench_cooldown_timer_minutes,
                ),
            ]
            self._activate_seat(
                seat, model=challenger.name, source=challenger_source, affinity=challenger_affinity, now=now
            )
            transitions.append(
                SeatTransition(
                    kind=TransitionKind.SEATED, model=challenger.name, source=challenger_source, reason=None, at=now
                ),
            )
            return transitions

        seat.pending_model = challenger.name
        seat.pending_source = challenger_source
        seat.pending_affinity = challenger_affinity
        seat.state = SeatState.PENDING_DOWNLOAD
        return [
            SeatTransition(
                kind=TransitionKind.DOWNLOAD_PENDING,
                model=challenger.name,
                source=challenger_source,
                reason=None,
                at=now,
            ),
        ]

    def _run_demotions(self, ranked_by_name: dict[str, RankedCandidate], now: float) -> list[SeatTransition]:
        """Demote dwell-eligible seats that have stopped earning their place (empty pops or no fulfillment)."""
        transitions: list[SeatTransition] = []
        for seat in self._seats:
            if seat.model is None or seat.pending_model is not None:
                continue
            if not self._past_dwell(seat, now):
                continue

            reason = self._demotion_reason(seat, ranked_by_name, now=now)
            if reason is None:
                continue
            transitions.append(
                self._demote_seat(
                    seat,
                    reason=reason,
                    now=now,
                    bench_cooldown_minutes=self._params.bench_cooldown_empty_pops_minutes,
                ),
            )
        return transitions

    def _demotion_reason(
        self,
        seat: _Seat,
        ranked_by_name: dict[str, RankedCandidate],
        now: float,
    ) -> DemotionReason | None:
        """Return why a dwell-eligible seat should demote, or ``None`` if it still earns its place.

        Two distinct non-productive triggers each carry their own reason for the caller to tell apart. A seat
        charged the empty-pop threshold while its model's observed queue is near zero demotes as
        :attr:`DemotionReason.EMPTY_POPS`; a seat that has served without any fulfillment for the zero-fulfillment
        window demotes as :attr:`DemotionReason.ZERO_FULFILLMENT`. Whichever condition holds first ends the seat.
        """
        empty_pops_exhausted = seat.empty_pops >= self._params.empty_pop_demotion_threshold
        if empty_pops_exhausted and self._demand_is_near_zero(seat, ranked_by_name):
            return DemotionReason.EMPTY_POPS

        reference_time = seat.last_fulfilled_at if seat.last_fulfilled_at is not None else seat.seated_at
        no_fulfillment_seconds = now - reference_time
        if no_fulfillment_seconds >= self._params.zero_fulfillment_demotion_minutes * _SECONDS_PER_MINUTE:
            return DemotionReason.ZERO_FULFILLMENT
        return None

    def _demand_is_near_zero(self, seat: _Seat, ranked_by_name: dict[str, RankedCandidate]) -> bool:
        """Return whether the seat's model has an observed queue depth at or below the near-zero threshold."""
        if seat.model is None:
            return False
        candidate = ranked_by_name.get(seat.model)
        if candidate is None or candidate.queued_per_worker is None:
            return False
        return candidate.queued_per_worker <= _NEAR_ZERO_QUEUE_PER_WORKER

    def _fill_manual(self, now: float) -> list[SeatTransition]:
        """Fill empty seats from manual pins in affinity order, seating each eligible pin immediately."""
        transitions: list[SeatTransition] = []
        ordered_pins = sorted(self._params.pinned, key=lambda pin: pin.affinity, reverse=True)
        for pin in ordered_pins:
            empty_seat = self._next_empty_seat()
            if empty_seat is None:
                break
            committed = self._committed_models()
            if not self._is_seatable(pin.name, now, committed):
                continue
            self._activate_seat(empty_seat, model=pin.name, source=SeatSource.MANUAL, affinity=pin.affinity, now=now)
            transitions.append(
                SeatTransition(
                    kind=TransitionKind.SEATED, model=pin.name, source=SeatSource.MANUAL, reason=None, at=now
                ),
            )
        return transitions

    def _fill_ranker(self, candidates: Sequence[RankedCandidate], now: float) -> list[SeatTransition]:
        """Fill remaining empty seats from the highest-scoring seatable candidates."""
        transitions: list[SeatTransition] = []
        ordered_candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
        for candidate in ordered_candidates:
            empty_seat = self._next_empty_seat()
            if empty_seat is None:
                break
            committed = self._committed_models()
            if not self._is_seatable(candidate.name, now, committed):
                continue
            candidate_source, candidate_affinity = self._seating_identity(candidate.name)
            if candidate.on_disk:
                self._activate_seat(
                    empty_seat, model=candidate.name, source=candidate_source, affinity=candidate_affinity, now=now
                )
                transitions.append(
                    SeatTransition(
                        kind=TransitionKind.SEATED, model=candidate.name, source=candidate_source, reason=None, at=now
                    ),
                )
                continue
            empty_seat.pending_model = candidate.name
            empty_seat.pending_source = candidate_source
            empty_seat.pending_affinity = candidate_affinity
            empty_seat.state = SeatState.PENDING_DOWNLOAD
            transitions.append(
                SeatTransition(
                    kind=TransitionKind.DOWNLOAD_PENDING,
                    model=candidate.name,
                    source=candidate_source,
                    reason=None,
                    at=now,
                ),
            )
        return transitions

    def _next_empty_seat(self) -> _Seat | None:
        """Return the first seat that is neither serving a model nor holding a pending download."""
        for seat in self._seats:
            if seat.model is None and seat.pending_model is None:
                return seat
        return None

    def _engage_rescue(self, candidates: Sequence[RankedCandidate], now: float) -> list[SeatTransition]:
        """Let the most-starved eligible candidate claim the weakest ranker seat for a bounded rescue window.

        An on-disk pick displaces the weakest ranker seat at once and benches the model it evicts. An off-disk
        pick does not evict anything yet: it marks that seat pending against the rescue download, so the
        incumbent keeps serving until :meth:`on_download_ready` swaps the rescued model in and starts its window.
        Either way the per-model rescue cooldown is charged at the engagement decision, so a pick that is still
        downloading is not re-selected while its fetch is in flight.
        """
        if self._rescue_seat_exists():
            return []
        rescue_pick = self._best_rescue_candidate(candidates, now=now)
        if rescue_pick is None:
            return []

        target_seat = self._weakest_ranker_seat()
        if target_seat is None or target_seat.model is None:
            return []
        if not self._past_dwell(target_seat, now):
            return []

        self._rescue_cooldowns[rescue_pick.name] = now + self._params.rescue_model_cooldown_hours * _SECONDS_PER_HOUR

        if not rescue_pick.on_disk:
            target_seat.pending_model = rescue_pick.name
            target_seat.pending_source = SeatSource.RESCUE
            target_seat.pending_affinity = 0.0
            target_seat.state = SeatState.PENDING_DOWNLOAD
            return [
                SeatTransition(
                    kind=TransitionKind.DOWNLOAD_PENDING,
                    model=rescue_pick.name,
                    source=SeatSource.RESCUE,
                    reason=None,
                    at=now,
                ),
            ]

        transitions = [
            self._demote_seat(
                target_seat,
                reason=DemotionReason.RESCUE_DISPLACED,
                now=now,
                bench_cooldown_minutes=self._params.bench_cooldown_timer_minutes,
            ),
        ]
        self._activate_seat(target_seat, model=rescue_pick.name, source=SeatSource.RESCUE, affinity=0.0, now=now)
        transitions.append(
            SeatTransition(
                kind=TransitionKind.RESCUE_ENGAGED,
                model=rescue_pick.name,
                source=SeatSource.RESCUE,
                reason=None,
                at=now,
            ),
        )
        return transitions

    def _best_rescue_candidate(self, candidates: Sequence[RankedCandidate], now: float) -> RankedCandidate | None:
        """Return the uncommitted, uncooled candidate with the highest wait ETA above the threshold.

        On-disk membership is not required: an off-disk pick engages as a pending download rather than seating
        at once, and the caller is responsible for having budget-gated any off-disk candidate before offering it.
        """
        committed = self._committed_models()
        best: RankedCandidate | None = None
        for candidate in candidates:
            if candidate.eta_seconds is None:
                continue
            if candidate.eta_seconds < self._params.rescue_eta_seconds:
                continue
            if candidate.name in committed:
                continue
            if candidate.name in self._rescue_cooldowns and self._rescue_cooldowns[candidate.name] > now:
                continue
            bench_entry = self._bench.get(candidate.name)
            if bench_entry is not None and bench_entry.cooldown_until > now:
                continue
            if best is None or candidate.eta_seconds > best.eta_seconds:  # type: ignore[operator]
                best = candidate
        return best

    def _weakest_ranker_seat(self) -> _Seat | None:
        """Return the lowest-scoring active ranker seat a rescue may claim, or ``None`` if there is none."""
        ranker_seats = [
            seat
            for seat in self._seats
            if seat.source is SeatSource.RANKER and seat.model is not None and seat.pending_model is None
        ]
        if not ranker_seats:
            return None
        return min(ranker_seats, key=lambda seat: seat.score)

    def params(self) -> PoolParams:
        """Return the pool's current (frozen) parameters."""
        return self._params
