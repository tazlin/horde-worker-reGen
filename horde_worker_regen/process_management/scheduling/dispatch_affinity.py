"""Bounded affinity line-skips: let resident-model jobs pass a cold FIFO head under a time+count budget.

Under diverse traffic (many distinct models per hour), strict-FIFO dispatch forces an unload+stage+preload
cycle per job while the head's model is cold. The existing resident-model bypass only fires when the head's
model is forecast to load, so during the head's preload-defer windows a job whose model is already resident
sits idle behind the cold head. Letting the resident job pass keeps the card fed, but an unbounded bypass
could starve the head behind a steady stream of resident work.

This module bounds the bypass. A displaced head may be passed by resident-model jobs only while inside a
skip budget: a wall-clock window and a hard skip count, both keyed to the head's identity. When either bound
is reached the bypass stops and dispatch falls back to making room for the head, which never lost its queue
position. The bound is unconditional: there is no path that raises it to infinity. A ``max_skips`` of 0 is a
full off-switch (no affinity skips at all).

Pure and table-testable; no scheduler/process imports. The scheduler owns the mutable ``AffinitySkipState``
and advances it only on committed dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

_AFFINITY_MAX_SKIPS = 6
"""The hard ceiling on how many times one head may be passed by affinity line-skips before it reclaims."""

_DEFAULT_AFFINITY_TTL_SECONDS = 150.0
"""Assumed job ttl when the horde has not supplied one, used to size the skip budget conservatively."""

_AFFINITY_BUDGET_FRACTION = 0.2
"""Fraction of the job ttl a displaced head may spend being bypassed before it reclaims the slot."""

_AFFINITY_BUDGET_MIN_SECONDS = 15.0
"""Floor so a short ttl still allows at least one useful bypass window."""

_AFFINITY_BUDGET_MAX_SECONDS = 45.0
"""Ceiling so a long ttl does not let the head sit bypassed for an unbounded stretch."""


@dataclass(frozen=True)
class AffinitySkipState:
    """The skip window for the currently-tracked displaced head.

    ``head_job_id`` names the head the window belongs to; a different head resets the window. ``first_skip_time``
    stamps the first committed skip against that head (the wall-clock budget runs from there), and ``skip_count``
    counts committed skips. The empty default (no head, zero skips) is the initial state before any bypass.
    """

    head_job_id: str | None = None
    first_skip_time: float = 0.0
    skip_count: int = 0


def affinity_budget_seconds(recent_job_ttl: float | None) -> float:
    """Return the wall-clock seconds a displaced head may be bypassed, derived from the job ttl.

    The displaced head's worst case (the skip window, then its own staging/preload, sampling, and
    post-inference tail) must sit well inside the ttl or the head ages past its deadline and the horde aborts
    it. Spending only a fraction of the ttl on bypassing leaves most of it for the head's own load and run; the
    clamp keeps the window sane when the ttl is very short or very long.

    Args:
        recent_job_ttl: The most recent horde-supplied job ttl in seconds, or None if none was supplied.

    Returns:
        The budget in seconds, clamped to ``[_AFFINITY_BUDGET_MIN_SECONDS, _AFFINITY_BUDGET_MAX_SECONDS]``.
    """
    ttl = recent_job_ttl if recent_job_ttl is not None else _DEFAULT_AFFINITY_TTL_SECONDS
    budget = _AFFINITY_BUDGET_FRACTION * ttl
    return max(_AFFINITY_BUDGET_MIN_SECONDS, min(budget, _AFFINITY_BUDGET_MAX_SECONDS))


def affinity_skip_allowed(
    state: AffinitySkipState,
    head_job_id: str | None,
    now: float,
    budget_seconds: float,
    max_skips: int,
) -> bool:
    """Whether a resident-model job may bypass the head named by ``head_job_id`` right now.

    A fresh head (one the window does not yet track) is always allowed a first bypass. A head already tracked
    is allowed only while it is under both bounds: fewer than ``max_skips`` committed skips and less than
    ``budget_seconds`` elapsed since its first skip. A ``max_skips`` of 0 (or a non-positive budget) is a full
    off-switch. A None head id (no dispatchable head) is never bypassable.

    Args:
        state: The current skip window.
        head_job_id: The identity of the FIFO head being considered for bypass.
        now: The current wall-clock time in seconds.
        budget_seconds: The wall-clock bypass budget from :func:`affinity_budget_seconds`.
        max_skips: The hard skip ceiling (0 disables affinity skips entirely).

    Returns:
        True if the head may be bypassed by a resident-model job, else False.
    """
    if head_job_id is None or max_skips <= 0 or budget_seconds <= 0:
        return False
    if state.head_job_id != head_job_id:
        return True
    if state.skip_count >= max_skips:
        return False
    return (now - state.first_skip_time) < budget_seconds


def record_affinity_skip(state: AffinitySkipState, head_job_id: str, now: float) -> AffinitySkipState:
    """Return the window advanced by one committed skip against ``head_job_id``.

    A skip against a head the window does not yet track starts a fresh window (count 1, budget clock starting
    now). A skip against the tracked head increments the count and keeps the original budget start, so the
    wall-clock budget always measures from the first skip.

    Args:
        state: The current skip window.
        head_job_id: The head that was just bypassed.
        now: The current wall-clock time in seconds.

    Returns:
        The advanced :class:`AffinitySkipState`.
    """
    if state.head_job_id != head_job_id:
        return AffinitySkipState(head_job_id=head_job_id, first_skip_time=now, skip_count=1)
    return AffinitySkipState(
        head_job_id=head_job_id,
        first_skip_time=state.first_skip_time,
        skip_count=state.skip_count + 1,
    )
