"""The enumerable surface for the scheduler-side budgets that hold work on a clock of their own.

Two scheduling decisions defer work under a bound rather than by returning an admission verdict: the affinity
line-skip window (:mod:`~horde_worker_regen.process_management.scheduling.dispatch_affinity`) lets
resident-model jobs pass a cold FIFO head for a bounded span, and the whole-card minimum hold
(:mod:`~horde_worker_regen.process_management.scheduling.governance.whole_card`) refuses an early release of a
fresh residency. Neither takes a member of an admission, pop-gate, or governor enum, so without a surface of
their own they can only be declared by hand, and a third one could ship with nothing to notice it.

This enum is that surface. Each budget stamps its member on the disclosure a reader sees when it engages, and
the gate registry's guardrail enumerates the members, so a further budget has to take one and therefore has to
declare a release path and a bound alongside it.

The enum sits beside the budgets rather than inside either of them because each module owns one member and
neither owns the other's. Nothing here imports scheduler state, so the pure budget modules can take it.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["SchedulerBudget"]


class SchedulerBudget(StrEnum):
    """A scheduler-side budget that holds work for a bounded span before releasing it.

    Enumerating them keeps the set closed: a further budget has to take a member, so it cannot ship without a
    declared release path in the gate registry.
    """

    AFFINITY_LINE_SKIP = "affinity_line_skip"
    """The count + wall-clock window inside which resident-model jobs may pass a cold FIFO head."""
    WHOLE_CARD_MIN_HOLD = "whole_card_min_hold"
    """The floor under a fresh whole-card residency, inside which a different-model head cannot release it."""
