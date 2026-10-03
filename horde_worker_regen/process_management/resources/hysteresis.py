"""Shared pure threshold latch for resource and backlog holds."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HysteresisLatch:
    """Represents a hold that engages at a high reading and releases at a lower reading."""

    active: bool = False

    def update(self, reading: float, *, engage_at: float, release_at: float, inclusive: bool = True) -> bool:
        """Return the next hold state; the caller owns storing it and acting on edges."""
        if self.active:
            return reading > release_at
        return reading >= engage_at if inclusive else reading > engage_at
