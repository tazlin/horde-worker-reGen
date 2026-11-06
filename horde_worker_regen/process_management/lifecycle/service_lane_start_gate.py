"""Consecutive start-failure streak and respawn backoff for one service lane.

A service lane (safety, post-processing, component, VAE, image utilities) is restarted by the lifecycle as soon as
its child is found dead. A lane that cannot start at all (a missing interpreter, a broken environment) then dies
within milliseconds of every start, and an immediate respawn turns that into a spawn and a reap every control-loop
tick. :class:`ServiceLaneStartGate` holds the shared arithmetic that spaces those starts out. The lifecycle holds
one instance per lane, and what counts as a start failure or as readiness is defined there.

- :class:`ServiceLaneStartGate`: the streak, the doubling backoff, and the earliest time of the next start.
- :data:`SERVICE_LANE_START_BACKOFF_BASE_SECONDS` and :data:`SERVICE_LANE_START_BACKOFF_MAX_SECONDS`: the first
  non-zero delay and its ceiling.
"""

from __future__ import annotations

__all__ = [
    "SERVICE_LANE_START_BACKOFF_BASE_SECONDS",
    "SERVICE_LANE_START_BACKOFF_MAX_SECONDS",
    "ServiceLaneStartGate",
]

SERVICE_LANE_START_BACKOFF_BASE_SECONDS: float = 2.0
"""First non-zero delay before restarting a service lane that has failed to reach readiness.

The delay doubles per consecutive failure up to ``SERVICE_LANE_START_BACKOFF_MAX_SECONDS``. The first restart
after a lone failure is immediate: that is the ordinary recovery path, and delaying it would leave the worker
without the lane for no reason. Only a repeat failure earns a wait, so a lane that cannot start stops consuming a
spawn, a full cold start and a reap several times a second."""

SERVICE_LANE_START_BACKOFF_MAX_SECONDS: float = 30.0
"""Ceiling on the service-lane start backoff.

Well under the safety crash-loop window, so a safety pool that cannot start still accumulates rebuilds fast enough
to register on both of its breakers. Low enough that a lane whose external cause has cleared (a card released by a
co-tenant, an environment repaired) comes back promptly rather than sitting out a long sleep."""


class ServiceLaneStartGate:
    """Track one service lane's consecutive start failures and the earliest time its next start may run.

    The streak counts failures since the lane last reached readiness, so it measures a lane that cannot initialise
    rather than one that came up, served and later died. Every method takes the caller's wall clock instead of
    reading one, so the lifecycle's clock is the only clock involved.
    """

    def __init__(self) -> None:
        """Create a gate with no failures recorded, which allows a start at once."""
        self._consecutive_start_failures = 0
        self._next_start_allowed_at = 0.0

    @property
    def consecutive_start_failures(self) -> int:
        """Return how many starts have failed in a row since the lane last reached readiness."""
        return self._consecutive_start_failures

    @property
    def next_start_allowed_at(self) -> float:
        """Return the earliest wall-clock time the next start may run; zero when no failure holds it."""
        return self._next_start_allowed_at

    def backoff_seconds(self) -> float:
        """Return the delay armed by the current streak.

        Zero up to the first failure, so the restart after a lone failure stays immediate. Each further failure
        doubles the delay from ``SERVICE_LANE_START_BACKOFF_BASE_SECONDS``, up to
        ``SERVICE_LANE_START_BACKOFF_MAX_SECONDS``.
        """
        if self._consecutive_start_failures <= 1:
            return 0.0
        delay = SERVICE_LANE_START_BACKOFF_BASE_SECONDS * (2 ** (self._consecutive_start_failures - 2))
        return min(delay, SERVICE_LANE_START_BACKOFF_MAX_SECONDS)

    def record_start_failure(self, now: float) -> None:
        """Count one more failed start and hold the next start back by the new backoff.

        Args:
            now: The caller's wall clock at the failure.
        """
        self._consecutive_start_failures += 1
        self._next_start_allowed_at = now + self.backoff_seconds()

    def start_allowed(self, now: float) -> bool:
        """Return whether a start may run at ``now``.

        Args:
            now: The caller's wall clock.
        """
        return now >= self._next_start_allowed_at

    def observe_ready(self) -> None:
        """Clear the streak and the backoff: the lane has proved it can initialise."""
        self.reset()

    def reset(self) -> None:
        """Return the gate to its initial state for a fresh attempt that is not a readiness report.

        A supervised rebuild starts the lane over on purpose. If the lane still cannot start, its failures
        accumulate again from zero.
        """
        self._consecutive_start_failures = 0
        self._next_start_allowed_at = 0.0
