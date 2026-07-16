"""Residency-biased pop advertising: duty-cycled narrowing of the offered model set toward residents.

Under diverse traffic a small-VRAM card pays a model swap (unload + stage + preload) for most popped
jobs, because the horde freely returns work for any offered model while only one or two checkpoints are
resident. Dispatch-side resident-model bypass cannot help when the shallow local queue holds no resident
candidate, so the shaping must move upstream to what the worker OFFERS: when a model-swap backlog exists,
narrowing the advertised set toward the currently-resident (and RAM-staged) checkpoints makes the horde
return work the card can run without a swap.

The narrowing is duty-cycled. It engages for a bounded run of pop cycles, then re-opens the full offered
set for a run of cycles, so cold demand still reaches the worker and rare-model kudos stay reachable. The
window is bounded by the duty cycle even if the backlog never clears; there is no path to a permanent
narrowing. Two safety rails hold regardless of phase: the narrowed offer is floored at resident+staged
(and falls back to the full offered set when that floor is empty), and narrowing only ever removes models
from the offer, never adds one the worker does not already offer, so an advertised model is always one the
worker would serve.

Pure and table-testable; no popper/scheduler imports. The popper owns the mutable ``ResidencyBiasState``
and advances it once per built pop request.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

_RESIDENCY_BIAS_NARROW_CYCLES = 6
"""Pop cycles the offer stays narrowed toward residents once a swap backlog engages the duty cycle.

Sized against the measured swap cost: a small-VRAM card under diverse traffic pays a swap for most jobs, so
the narrow phase dominates the duty cycle (a 3:1 narrow:open ratio) to keep residents fed. It is a hard
bound on one narrow window: even with a persistent backlog the offer re-opens after this many narrowed pops.
"""

_RESIDENCY_BIAS_OPEN_CYCLES = 2
"""Pop cycles the full offered set is re-advertised between narrow phases while a backlog persists.

Guarantees cold and rare-model demand keeps reaching the worker (one full-set advertise for every three
narrowed pops), so residency biasing never permanently hides the worker's less-resident models from the
horde and the rare-model kudos bonus stays reachable.
"""


@dataclass(frozen=True)
class ResidencyBiasState:
    """The duty-cycle phase of residency-biased advertising.

    ``active`` is whether a backlog-driven duty cycle is running (it clears the moment the swap backlog
    clears, so the next engagement starts fresh in the narrow phase). ``narrowing`` is whether the current
    phase narrows the offer toward residents, and ``cycles_in_phase`` counts pop cycles already spent in the
    current phase. The empty default (idle, open, zero cycles) advertises the full set until a swap backlog
    engages the narrow phase.
    """

    active: bool = False
    narrowing: bool = False
    cycles_in_phase: int = 0


@dataclass(frozen=True)
class ResidencyBiasDecision:
    """The advertising decision for one pop cycle, plus the advanced duty-cycle state.

    ``advertised_models`` is the set to put on this pop's request (never empty when the offered set was not).
    ``narrowing`` is whether this cycle is in a narrow phase; ``narrowed_offer`` is whether the offer was
    actually reduced below the full offered set (a narrow phase whose resident+staged floor already covers
    every offered model does not reduce it). ``next_state`` is the duty-cycle state the popper must store.
    """

    advertised_models: frozenset[str]
    narrowing: bool
    narrowed_offer: bool
    next_state: ResidencyBiasState


def _advance_phase(
    *,
    narrowing: bool,
    cycles_consumed: int,
    narrow_cycles: int,
    open_cycles: int,
) -> ResidencyBiasState:
    """Return the active duty-cycle state after consuming ``cycles_consumed`` cycles of the current phase.

    The phase flips (narrow to open, or open to narrow) once its length is reached, resetting the counter;
    otherwise the phase holds and the counter carries forward. A phase length below one is treated as one so
    a phase always consumes at least the cycle that entered it.
    """
    phase_len = max(1, narrow_cycles if narrowing else open_cycles)
    if cycles_consumed >= phase_len:
        return ResidencyBiasState(active=True, narrowing=not narrowing, cycles_in_phase=0)
    return ResidencyBiasState(active=True, narrowing=narrowing, cycles_in_phase=cycles_consumed)


def decide_residency_advertising(
    state: ResidencyBiasState,
    *,
    swap_backlog: bool,
    resident_models: Iterable[str],
    staged_models: Iterable[str],
    offered_models: Iterable[str],
    narrow_cycles: int = _RESIDENCY_BIAS_NARROW_CYCLES,
    open_cycles: int = _RESIDENCY_BIAS_OPEN_CYCLES,
) -> ResidencyBiasDecision:
    """Decide this pop cycle's advertised model set and advance the duty cycle.

    With no swap backlog (or a non-positive ``narrow_cycles`` off-switch) the full offered set is advertised
    and the duty cycle is reset to idle, so the next backlog engages a fresh narrow phase. With a swap
    backlog the duty cycle runs: the first backlogged cycle engages the narrow phase, subsequent cycles
    alternate narrow (``narrow_cycles`` long) and open (``open_cycles`` long). During a narrow phase the
    offer is intersected with resident+staged, floored back to the full offered set when that intersection is
    empty so the offer is never emptied, and never expanded beyond the offered set.

    Args:
        state: The current duty-cycle state (the popper's stored ``ResidencyBiasState``).
        swap_backlog: Whether the worker is currently paying model swaps (a queued head needs a
            non-resident model while residents exist).
        resident_models: Models currently resident on a sampler slot.
        staged_models: Models staged in RAM (loadable without a fresh download/stage).
        offered_models: The full eligible set this pop would otherwise advertise (assumed non-empty).
        narrow_cycles: Narrow-phase length in pop cycles; non-positive disables narrowing entirely.
        open_cycles: Open-phase length in pop cycles.

    Returns:
        The :class:`ResidencyBiasDecision` for this cycle.
    """
    offered = frozenset(offered_models)

    if not swap_backlog or narrow_cycles <= 0:
        return ResidencyBiasDecision(
            advertised_models=offered,
            narrowing=False,
            narrowed_offer=False,
            next_state=ResidencyBiasState(),
        )

    if not state.active:
        narrowing_now = True
        next_state = _advance_phase(
            narrowing=True,
            cycles_consumed=1,
            narrow_cycles=narrow_cycles,
            open_cycles=open_cycles,
        )
    else:
        narrowing_now = state.narrowing
        next_state = _advance_phase(
            narrowing=state.narrowing,
            cycles_consumed=state.cycles_in_phase + 1,
            narrow_cycles=narrow_cycles,
            open_cycles=open_cycles,
        )

    if not narrowing_now:
        return ResidencyBiasDecision(
            advertised_models=offered,
            narrowing=False,
            narrowed_offer=False,
            next_state=next_state,
        )

    floor = (frozenset(resident_models) | frozenset(staged_models)) & offered
    advertised = floor if floor else offered
    return ResidencyBiasDecision(
        advertised_models=advertised,
        narrowing=True,
        narrowed_offer=advertised != offered,
        next_state=next_state,
    )
