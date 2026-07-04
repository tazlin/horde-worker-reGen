"""Tests for fixed-window dashboard trend bucketing."""

from __future__ import annotations

from horde_worker_regen.app_state import OverviewTrendWindow
from horde_worker_regen.tui.trends import fixed_counter_deltas, fixed_float_buckets, fixed_ratio_deltas


def test_five_minute_window_produces_fixed_full_span_buckets() -> None:
    """A warming 5m window still renders the requested bucket count with early zero buckets."""
    series = fixed_float_buckets(
        [(280.0, 50.0), (300.0, 60.0)],
        OverviewTrendWindow.FIVE_MINUTES,
        now=300.0,
        buckets=5,
    )

    assert series == [0.0, 0.0, 0.0, 0.0, 55.0]


def test_sixty_minute_window_spans_full_duration() -> None:
    """A 60m selection buckets against now-60m, not just the number of samples present."""
    series = fixed_float_buckets(
        [(3590.0, 90.0)],
        OverviewTrendWindow.SIXTY_MINUTES,
        now=3600.0,
        buckets=6,
    )

    assert len(series) == 6
    assert series[:-1] == [0.0] * 5
    assert series[-1] == 90.0


def test_all_window_spans_session_start() -> None:
    """All mode uses the worker session start as the left edge."""
    series = fixed_float_buckets(
        [(110.0, 10.0), (190.0, 30.0)],
        OverviewTrendWindow.ALL,
        now=200.0,
        session_start=100.0,
        buckets=2,
    )

    assert series == [10.0, 30.0]


def test_jobs_per_hour_uses_completion_deltas_in_selected_window() -> None:
    """Jobs/hr is derived from cumulative-counter deltas inside the selected span."""
    rate, deltas, sampled_span = fixed_counter_deltas(
        [(0.0, 0), (60.0, 1), (120.0, 3)],
        OverviewTrendWindow.FIVE_MINUTES,
        now=300.0,
        buckets=5,
    )

    assert rate == 36.0
    assert deltas == [0.0, 1.0, 2.0, 0.0, 0.0]
    assert sampled_span == 120.0


def test_kudos_per_hour_divides_by_productive_seconds() -> None:
    """Windowed kudos/hr divides kudos earned by productive seconds earned, with per-bucket kudos deltas."""
    rate, deltas, sampled_span = fixed_ratio_deltas(
        [(0.0, 0.0, 0.0), (60.0, 10.0, 60.0), (120.0, 30.0, 120.0)],
        OverviewTrendWindow.FIVE_MINUTES,
        now=300.0,
        buckets=5,
    )

    # 30 kudos over 120 productive seconds = 900/hr.
    assert rate == 900.0
    assert deltas == [0.0, 10.0, 20.0, 0.0, 0.0]
    assert sampled_span == 120.0


def test_kudos_per_hour_excludes_idle_from_denominator() -> None:
    """When productive seconds trail wall time, the rate reflects the active pace, not the wall-clock pace."""
    rate, _deltas, _sampled_span = fixed_ratio_deltas(
        # 60 wall seconds elapse but only 30 are productive (an empty pipeline for the other 30).
        [(0.0, 0.0, 0.0), (60.0, 10.0, 30.0)],
        OverviewTrendWindow.FIVE_MINUTES,
        now=300.0,
        buckets=5,
    )

    # 10 kudos over 30 productive seconds = 1200/hr (a raw-wall-clock rate would read only 600/hr).
    assert rate == 1200.0


def test_kudos_per_hour_warming_up_without_two_samples() -> None:
    """With fewer than two in-window samples the rate is undefined and buckets stay zero."""
    rate, deltas, sampled_span = fixed_ratio_deltas(
        [(0.0, 5.0, 5.0)],
        OverviewTrendWindow.FIVE_MINUTES,
        now=300.0,
        buckets=5,
    )

    assert rate is None
    assert deltas == [0.0] * 5
    assert sampled_span == 0.0
