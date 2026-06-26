"""Shared fixed-window trend helpers for dashboard sparkline data."""

from __future__ import annotations

import time
from collections.abc import Sequence

from horde_worker_regen.app_state import OverviewTrendWindow

TREND_WINDOW_SECONDS: dict[OverviewTrendWindow, float | None] = {
    OverviewTrendWindow.FIVE_MINUTES: 5 * 60.0,
    OverviewTrendWindow.FIFTEEN_MINUTES: 15 * 60.0,
    OverviewTrendWindow.THIRTY_MINUTES: 30 * 60.0,
    OverviewTrendWindow.SIXTY_MINUTES: 60 * 60.0,
    OverviewTrendWindow.TWO_HOURS: 120 * 60.0,
    OverviewTrendWindow.ALL: None,
}


def trend_bounds(
    window: OverviewTrendWindow,
    *,
    now: float | None = None,
    session_start: float | None = None,
    epoch: float | None = None,
) -> tuple[float, float, float | None]:
    """Return ``(start, end, configured_seconds)`` for a selected trend window."""
    end = now if now is not None else time.time()
    configured = TREND_WINDOW_SECONDS[window]
    if configured is None:
        start = session_start if session_start is not None else epoch if epoch is not None else end
        if epoch is not None:
            start = max(start, epoch)
        return start, end, None
    start = end - configured
    if epoch is not None:
        start = max(start, epoch)
    return start, end, configured


def fixed_float_buckets(
    samples: Sequence[tuple[float, float]],
    window: OverviewTrendWindow,
    *,
    now: float | None = None,
    session_start: float | None = None,
    epoch: float | None = None,
    buckets: int = 48,
) -> list[float]:
    """Bucket timestamped float samples across the whole selected span, filling empty buckets with zero."""
    start, end, configured = trend_bounds(window, now=now, session_start=session_start, epoch=epoch)
    span = configured if configured is not None else max(end - start, 1.0)
    if buckets <= 0 or span <= 0:
        return []
    bucket_seconds = span / buckets
    totals = [0.0 for _ in range(buckets)]
    counts = [0 for _ in range(buckets)]
    for timestamp, value in samples:
        if timestamp < start or timestamp > end:
            continue
        index = min(buckets - 1, max(0, int((timestamp - start) / bucket_seconds)))
        totals[index] += value
        counts[index] += 1
    return [totals[index] / counts[index] if counts[index] else 0.0 for index in range(buckets)]


def fixed_counter_deltas(
    samples: Sequence[tuple[float, int]],
    window: OverviewTrendWindow,
    *,
    now: float | None = None,
    session_start: float | None = None,
    epoch: float | None = None,
    buckets: int = 48,
) -> tuple[float | None, list[float], float]:
    """Return jobs/hr, bucketed completion deltas, and sampled span for cumulative counter samples."""
    start, end, configured = trend_bounds(window, now=now, session_start=session_start, epoch=epoch)
    span = configured if configured is not None else max(end - start, 1.0)
    if buckets <= 0 or span <= 0:
        return None, [], 0.0
    bucket_seconds = span / buckets
    in_window = [(timestamp, count) for timestamp, count in samples if start <= timestamp <= end]
    if len(in_window) < 2:
        return None, [0.0 for _ in range(buckets)], 0.0
    deltas = [0.0 for _ in range(buckets)]
    for previous, current in zip(in_window, in_window[1:], strict=False):
        delta = max(0, current[1] - previous[1])
        index = min(buckets - 1, max(0, int((current[0] - start) / bucket_seconds)))
        deltas[index] += float(delta)
    completed = sum(deltas)
    rate_span = span if configured is not None else max(in_window[-1][0] - in_window[0][0], 1.0)
    sampled_span = max(in_window[-1][0] - in_window[0][0], 0.0)
    return (completed / rate_span * 3600.0 if rate_span > 0 else None), deltas, sampled_span
