"""Trust-gated marginal RAM estimates, separated by model and staging accounting kind."""

from collections import deque
from dataclasses import dataclass, field
from math import isfinite

_MIN_OBSERVATIONS = 5
_RECENT_WINDOW = 20
_MEASUREMENT_MARGIN = 1.10


@dataclass
class _RamObservation:
    recent_mb: deque[float] = field(default_factory=lambda: deque(maxlen=_RECENT_WINDOW))
    count: int = 0


class LearnedRamStore:
    """Record settled load growth, trusting bidirectional estimates after five observations.

    Whole, reused-page and component loads never share evidence: growth for a reused slot cannot
    demonstrate that a cold checkpoint load is cheap. The recent maximum and ten-percent margin mirror
    the trust policy of the bidirectional VRAM estimate. This store is scoped to one worker launch.
    """

    def __init__(self) -> None:
        """Initialize an empty observation store."""
        self._observations: dict[tuple[str, str], _RamObservation] = {}

    def observe(self, model: str, kind: str, growth_mb: float) -> None:
        """Record nonnegative, finite settled growth for this staging identity."""
        if not isfinite(growth_mb) or growth_mb < 0:
            return
        observation = self._observations.setdefault((model, kind), _RamObservation())
        observation.recent_mb.append(growth_mb)
        observation.count += 1

    def measured_estimate_mb(self, model: str, kind: str) -> float | None:
        """Return a margined recent maximum, or None until the identity is sufficiently observed."""
        observation = self._observations.get((model, kind))
        if observation is None or observation.count < _MIN_OBSERVATIONS:
            return None
        return max(observation.recent_mb) * _MEASUREMENT_MARGIN
