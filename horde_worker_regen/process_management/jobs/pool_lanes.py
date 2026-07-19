"""Pool-lane pop advertising: interleave a fixed-pool-narrowed offer with the wider free offer.

When the fixed model pool holds seats, a pop can advertise one of two lanes. The ``FIXED`` lane narrows the
offer to the seated models the pool commits to serving, so the horde returns work the card can run without a
per-job model swap. The ``FREE`` lane advertises the models the pool is not seating, so cold and rare-model
demand still reaches the worker and a seated model does not monopolise the offer. This module decides which
lane a pop advertises and advances the interleave state, staying pure so the popper owns the mutable
:class:`PoolLaneState` and the decision is table-testable.

The interleave is a smooth weighted round-robin between the two lanes. The fixed lane is weighted by how many
seated models it can offer this cycle; the free lane is weighted by how many inference slots are not committed
to a seat, plus a small boost when the last two fixed-lane pops both came back empty (so a fixed lane the
horde is not feeding yields the offer back to the wider set rather than starving). Two edge rules override the
round-robin: an empty fixed offer forces ``FREE`` and an empty free offer forces ``FIXED``, and when both are
empty the lane is ``FREE`` advertising the full eligible set (the caller handles an empty offer as it does
today). A forced choice does not disturb the round-robin credit, so the interleave resumes proportionally once
both lanes can offer again.

The fixed-lane emptiness memory the free-weight boost reads is folded in by a separate hook,
:func:`record_fixed_pop_outcome`, which the popper calls with each fixed-lane pop's empty-or-not outcome. This
keeps :func:`decide_pool_lane` free of outcome bookkeeping: it reads the memory and carries it forward
unchanged, while the outcome hook is the sole writer.

Public surface: :class:`PoolLaneState`, :class:`LaneDecision`, :func:`decide_pool_lane`, and
:func:`record_fixed_pop_outcome`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from horde_worker_regen.process_management.scheduling.model_pool import PopLane

__all__ = [
    "LaneDecision",
    "PoolLaneState",
    "decide_pool_lane",
    "record_fixed_pop_outcome",
]

_RECENT_FIXED_MEMORY = 2
"""How many recent fixed-lane pop outcomes the free-weight boost remembers. The boost engages only when the
whole remembered run is empty, so a single unlucky empty pop does not yet tilt the interleave back toward the
free lane."""


@dataclass(frozen=True)
class PoolLaneState:
    """The interleave state carried between pop cycles by the popper.

    ``fixed_credit`` and ``free_credit`` are the smooth weighted round-robin accumulators (only advanced by a
    weighted decision, never by a forced edge choice). ``recent_fixed_empty`` remembers the last
    :data:`_RECENT_FIXED_MEMORY` fixed-lane pop outcomes as empty flags (most recent last), which the
    free-weight boost reads. The empty default (zero credit, no remembered outcomes) starts the interleave
    balanced with no boost.
    """

    fixed_credit: int = 0
    free_credit: int = 0
    recent_fixed_empty: tuple[bool, ...] = field(default=())


@dataclass(frozen=True)
class LaneDecision:
    """The advertising decision for one pop cycle, plus the advanced interleave state.

    ``lane`` is the chosen advertising lane; ``advertised`` is the model set to put on this pop's request (the
    chosen lane's offer, or the full eligible set when both offers are empty). ``next_state`` is the interleave
    state the popper must store. ``reason`` is a short human-readable account of the choice for edge logging.
    """

    lane: PopLane
    advertised: frozenset[str]
    next_state: PoolLaneState
    reason: str


def _recent_run_all_empty(recent_fixed_empty: tuple[bool, ...]) -> bool:
    """Return whether the remembered fixed-lane run is full and every remembered outcome was empty."""
    return len(recent_fixed_empty) >= _RECENT_FIXED_MEMORY and all(recent_fixed_empty)


def decide_pool_lane(
    state: PoolLaneState,
    *,
    eligible: frozenset[str],
    active_seats: frozenset[str],
    process_count: int,
) -> LaneDecision:
    """Decide this pop cycle's advertising lane and advance the interleave.

    The fixed offer is the eligible models that are also seated; the free offer is the eligible models that are
    not. An empty fixed offer forces ``FREE`` and an empty free offer forces ``FIXED``; both empty yields
    ``FREE`` advertising the full eligible set. Otherwise a smooth weighted round-robin picks the lane, the
    fixed lane weighted by the count of seated offerings and the free lane by the uncommitted slot count plus a
    one-unit boost when the last two fixed-lane pops both came back empty. A forced choice leaves the
    round-robin credit untouched; only a weighted choice advances it.

    Args:
        state: The current interleave state (the popper's stored :class:`PoolLaneState`).
        eligible: The models this pop would otherwise advertise (the pool-narrowed candidate set).
        active_seats: The models the pool currently seats.
        process_count: The live inference-slot count, weighting how much of the offer the free lane deserves.

    Returns:
        The :class:`LaneDecision` for this cycle.
    """
    fixed_offer = eligible & active_seats
    free_offer = eligible - active_seats

    if not fixed_offer and not free_offer:
        return LaneDecision(
            lane=PopLane.FREE,
            advertised=eligible,
            next_state=state,
            reason="no eligible models to lane; free lane with the full eligible set",
        )
    if not fixed_offer:
        return LaneDecision(
            lane=PopLane.FREE,
            advertised=free_offer,
            next_state=state,
            reason="no seated model is eligible this cycle; free lane",
        )
    if not free_offer:
        return LaneDecision(
            lane=PopLane.FIXED,
            advertised=fixed_offer,
            next_state=state,
            reason="every eligible model is seated; fixed lane",
        )

    fixed_weight = len(fixed_offer)
    empty_run_boost = 1 if _recent_run_all_empty(state.recent_fixed_empty) else 0
    free_weight = max(0, process_count - len(active_seats)) + empty_run_boost

    fixed_credit = state.fixed_credit + fixed_weight
    free_credit = state.free_credit + free_weight
    total_weight = fixed_weight + free_weight

    if fixed_credit >= free_credit:
        next_state = replace(state, fixed_credit=fixed_credit - total_weight, free_credit=free_credit)
        return LaneDecision(
            lane=PopLane.FIXED,
            advertised=fixed_offer,
            next_state=next_state,
            reason=f"fixed lane (weight {fixed_weight} vs free {free_weight})",
        )

    next_state = replace(state, fixed_credit=fixed_credit, free_credit=free_credit - total_weight)
    return LaneDecision(
        lane=PopLane.FREE,
        advertised=free_offer,
        next_state=next_state,
        reason=f"free lane (weight {free_weight} vs fixed {fixed_weight})",
    )


def record_fixed_pop_outcome(state: PoolLaneState, *, was_empty: bool) -> PoolLaneState:
    """Fold one fixed-lane pop outcome into the emptiness memory the free-weight boost reads.

    Appends the empty-or-not flag to the remembered run, keeping only the most recent
    :data:`_RECENT_FIXED_MEMORY` outcomes. Only fixed-lane pops are recorded, since only they can charge the
    fixed lane with returning no work; the round-robin credit is left untouched.
    """
    updated_run = (*state.recent_fixed_empty, was_empty)[-_RECENT_FIXED_MEMORY:]
    return replace(state, recent_fixed_empty=updated_run)
