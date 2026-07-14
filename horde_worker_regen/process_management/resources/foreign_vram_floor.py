"""Sustained-minimum tracker for measured non-worker (foreign) VRAM usage per card.

The VRAM arbiter judges a candidate structurally impossible when it exceeds the card's achievable ceiling
(``total - noise - foreign_floor``). The foreign floor is the VRAM the OS/desktop/other processes sustain,
which the worker cannot reclaim by evicting its own residents. It is computed outside the arbiter (the
arbiter stays a pure function of measured state) from ``foreign_now = total - device_free - worker_footprint``,
and smoothed here into a *sustained* floor: the minimum over a trailing window.

Taking the minimum makes the floor robust in the safe direction. A transient foreign spike (a game or browser
briefly allocating) raises ``foreign_now`` for a few samples but not the window minimum, so it never
permanently lowers the ceiling and wrongly denies a servable model. A momentary dip only lowers the floor,
which can only raise the ceiling: the worst it does is defer-and-reclaim a load that then does not fit, never a
false terminal denial. The floor is withheld (None) until a full window of observation has elapsed, so a
cold-start transient cannot set it prematurely; None preserves the arbiter's pre-foreign behaviour exactly.
"""

from __future__ import annotations

from collections import deque

FOREIGN_FLOOR_WINDOW_SECONDS = 120.0
"""Trailing window (seconds) the sustained foreign floor is the minimum over. Also the warm-up span: no floor
is reported until observation has covered a full window, so a startup transient cannot set it early."""


class ForeignVramFloorTracker:
    """Per-card trailing-window minimum of measured foreign VRAM usage, on a monotonic clock.

    One instance serves every card; samples are keyed by device index (the arbiter's single-GPU/worker-wide
    key is 0). Cheap and allocation-light per update: an append and a bounded prune of a per-card deque, so it
    is safe to drive once per control tick.
    """

    def __init__(self, *, window_seconds: float = FOREIGN_FLOOR_WINDOW_SECONDS) -> None:
        """Initialise with an empty per-card sample history and the given trailing-window span."""
        self._window_seconds = window_seconds
        self._samples: dict[int, deque[tuple[float, float]]] = {}
        self._first_sample_at: dict[int, float] = {}

    def update(self, device_key: int, foreign_now_mb: float | None, *, now: float) -> float | None:
        """Record this tick's foreign reading (when available) and return the sustained floor, or None.

        Args:
            device_key: The card index (0 for the single-GPU/worker-wide key).
            foreign_now_mb: This tick's ``total - device_free - worker_footprint`` (MB), clamped to >= 0 on
                entry, or None when it cannot be measured (children not yet reporting VRAM). None contributes
                no sample; the existing window still ages so a card whose reports stop eventually forgets its
                floor.
            now: The monotonic-clock timestamp of this tick.

        Returns:
            The minimum foreign reading over the trailing window once observation has covered a full window,
            else None (cold start / warm-up).
        """
        samples = self._samples.setdefault(device_key, deque())
        if foreign_now_mb is not None:
            samples.append((now, max(0.0, foreign_now_mb)))
            self._first_sample_at.setdefault(device_key, now)
        cutoff = now - self._window_seconds
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        if not samples:
            self._first_sample_at.pop(device_key, None)
            return None
        first_sample_at = self._first_sample_at.get(device_key)
        if first_sample_at is None or (now - first_sample_at) < self._window_seconds:
            return None
        return min(mb for _timestamp, mb in samples)

    def current_floor_mb(self, device_key: int, *, now: float) -> float | None:
        """Return the sustained floor for a card without recording a new sample (a read-only ceiling read).

        Ages the trailing window against ``now`` (stale samples are pruned exactly as in :meth:`update`) and
        returns the current minimum, or None during warm-up. Used by the live ceiling read that decides when a
        conditional hold may lift, which must not itself contribute a sample.
        """
        return self.update(device_key, None, now=now)
