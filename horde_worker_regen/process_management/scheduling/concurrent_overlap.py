"""Whether a new job may start sampling beside jobs already sampling on the same card.

The concurrency cap only counts in-flight jobs. This rule adds what the count leaves out: how far along the
running jobs are, how heavy each side of the overlap is, and whether the card has been measured to hold the
pairing. A newcomer is held off a running job's memory-hungry startup so two weight loads and activation peaks
do not stack into a step-timeout teardown. The memory question itself belongs to the VRAM arbiter and is passed
in as a verdict callable; this module only decides when to ask it and how its answer relaxes the headway.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from horde_worker_regen.process_management.models.model_sizing import ModelSizeTier

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData

OVERLAP_HEADWAY_MIXED_HEAVY = 0.5
"""Progress the running job needs before a newcomer joins when exactly one side of the pairing is heavy."""

OVERLAP_HEADWAY_BOTH_HEAVY = 0.75
"""Progress the running job needs before a second heavy job joins it, and the bound a batched pairing uses."""

OVERLAP_HEADWAY_AMPLE_VRAM = 0.15
"""Headway used in place of the strict fractions once the arbiter has confirmed the card holds the overlap.

The strict fractions price every card as tight, which on a large card serving a heavy-only queue collapses two
configured threads into one effective thread. A small headway is kept so the running job clears its startup.
"""

OVERLAP_HEADWAY_SCALE_HIGH_PERFORMANCE = 0.5
"""Headway multiplier in high-performance mode: the newcomer joins the running job's tail sooner."""

OVERLAP_HEADWAY_SCALE_MODERATE_PERFORMANCE = 0.75
"""Headway multiplier in moderate-performance mode."""


@dataclass(frozen=True, slots=True)
class RunningSampler:
    """What the overlap rule needs to know about one job already sampling."""

    tier: ModelSizeTier
    batched: bool
    progress_fraction: float


def performance_mode_headway_scale(bridge_data: reGenBridgeData) -> float:
    """The headway multiplier for the worker's performance mode; ``1.0`` outside the fast modes.

    The multiplier moves when an admissible overlap begins. The arbiter still decides whether it fits.
    """
    if bridge_data.high_performance_mode:
        return OVERLAP_HEADWAY_SCALE_HIGH_PERFORMANCE
    if bridge_data.moderate_performance_mode:
        return OVERLAP_HEADWAY_SCALE_MODERATE_PERFORMANCE
    return 1.0


def required_overlap_headway(running_tier: ModelSizeTier, candidate_tier: ModelSizeTier) -> float:
    """Progress the running job must have made before a candidate joins it, by size pairing.

    Assumes neither side is extra-large or batched; those are hard blocks decided before headway is consulted.
    """
    if running_tier <= ModelSizeTier.LIGHT and candidate_tier <= ModelSizeTier.LIGHT:
        return 0.0
    if running_tier >= ModelSizeTier.HEAVY and candidate_tier >= ModelSizeTier.HEAVY:
        return OVERLAP_HEADWAY_BOTH_HEAVY
    return OVERLAP_HEADWAY_MIXED_HEAVY


def concurrent_overlap_permitted(
    *,
    candidate_tier: ModelSizeTier,
    candidate_batched: bool,
    running: Sequence[RunningSampler],
    headway_scale: float,
    memory_verdict: Callable[[], bool | None],
) -> bool:
    """Whether a candidate may start beside ``running`` samplers.

    ``memory_verdict`` is the arbiter's answer for the candidate's demand: ``True`` when a real cycle confirmed
    room, ``False`` when one denied it, ``None`` when the demand could not be priced. It is consulted at most
    once and only when a rule needs it. Positive confirmation relaxes the headway; a denial vetoes the overlap;
    an unpriced demand keeps the strict headway yet admits on memory, since missing telemetry is not evidence of
    room.

    With nothing running the candidate always starts. An extra-large model neither joins a busy card nor shares
    one. A batch multiplies the activation peak, so a batched side needs confirmed room and is then bounded by
    the strictest headway rather than the relaxed one.
    """
    if not running:
        return True
    if candidate_tier >= ModelSizeTier.EXTRA_LARGE:
        return False

    verdict_cache: list[bool | None] = []

    def memory_ample() -> bool:
        if not verdict_cache:
            verdict_cache.append(memory_verdict())
        return verdict_cache[0] is True

    def memory_admits() -> bool:
        if not verdict_cache:
            verdict_cache.append(memory_verdict())
        return verdict_cache[0] is not False

    if candidate_batched and not memory_ample():
        return False

    for sampler in running:
        if sampler.tier >= ModelSizeTier.EXTRA_LARGE:
            return False

        if candidate_batched or sampler.batched:
            if not memory_ample():
                return False
            required_headway = OVERLAP_HEADWAY_BOTH_HEAVY
        else:
            required_headway = required_overlap_headway(sampler.tier, candidate_tier)
            if required_headway > 0.0 and memory_ample():
                required_headway = OVERLAP_HEADWAY_AMPLE_VRAM

        required_headway *= headway_scale
        if required_headway <= 0.0:
            continue
        if sampler.progress_fraction < required_headway:
            return False

    return memory_admits()
