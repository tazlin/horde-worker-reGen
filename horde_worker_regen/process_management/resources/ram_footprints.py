"""Trust-gated marginal RAM estimates per model and staging kind, with a per-baseline fallback."""

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
    """Record a checkpoint load's peak growth, trusting bidirectional estimates after five observations.

    Observations come from load completion, before a job runs, so they hold no feature RAM and one price
    serves every feature mix. Whole, reused-page and component loads never share evidence: growth for a
    reused slot cannot demonstrate that a cold checkpoint load is cheap. The recent maximum and ten-percent
    margin mirror the trust policy of the bidirectional VRAM estimate. This store is scoped to one launch.

    A worker whose models stay resident completes few cold loads of any one checkpoint, so each load also
    counts toward its baseline as growth per megabyte of the staged file. A checkpoint without trusted
    evidence of its own is priced from its baseline's ratio and its own size, which carries between fp16
    and fp8 files of one baseline where a flat figure would not.
    """

    def __init__(self) -> None:
        """Initialize an empty observation store."""
        self._observations: dict[tuple[str, str], _RamObservation] = {}
        self._baseline_ratios: dict[tuple[str, str], _RamObservation] = {}

    def observe(
        self,
        model: str,
        kind: str,
        settled_mb: float,
        peak_mb: float | None = None,
        *,
        baseline: str | None = None,
        size_mb: float | None = None,
    ) -> None:
        """Record a load's growth; the peak counts when reported, never below the settled growth.

        With a baseline and the staged file's size, the growth per megabyte also counts toward the baseline.
        """
        growth_mb = settled_mb if peak_mb is None else max(settled_mb, peak_mb)
        if not isfinite(growth_mb) or growth_mb < 0:
            return
        observation = self._observations.setdefault((model, kind), _RamObservation())
        observation.recent_mb.append(growth_mb)
        observation.count += 1
        if baseline is None or size_mb is None or not isfinite(size_mb) or size_mb <= 0:
            return
        ratio = self._baseline_ratios.setdefault((baseline, kind), _RamObservation())
        ratio.recent_mb.append(growth_mb / size_mb)
        ratio.count += 1

    def measured_estimate_mb(
        self,
        model: str,
        kind: str,
        *,
        baseline: str | None = None,
        size_mb: float | None = None,
    ) -> float | None:
        """Return the model's margined recent maximum, else its baseline's ratio applied to ``size_mb``.

        None until either identity is sufficiently observed.
        """
        observation = self._observations.get((model, kind))
        if observation is not None and observation.count >= _MIN_OBSERVATIONS:
            return max(observation.recent_mb) * _MEASUREMENT_MARGIN
        if baseline is None or size_mb is None or size_mb <= 0:
            return None
        ratio = self._baseline_ratios.get((baseline, kind))
        if ratio is None or ratio.count < _MIN_OBSERVATIONS:
            return None
        return max(ratio.recent_mb) * size_mb * _MEASUREMENT_MARGIN
