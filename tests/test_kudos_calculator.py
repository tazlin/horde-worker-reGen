"""Tests for kudos calculator."""

import time
from collections import deque

from horde_worker_regen.utils.kudos_calculator import KudosCalculator


def test_calculate_kudos_per_hour_normal() -> None:
    """Test kudos per hour calculation over productive time."""
    kudos_generated = 100.0
    eligible_seconds = 1800  # 30 productive minutes

    result = KudosCalculator.calculate_kudos_per_hour(kudos_generated, eligible_seconds)

    # 100 kudos in 30 productive minutes = 200 kudos per hour
    assert result == 200.0


def test_calculate_kudos_per_hour_no_productive_time() -> None:
    """With no productive time accrued yet the rate is undefined (the worker is still warming up)."""
    result = KudosCalculator.calculate_kudos_per_hour(100.0, 0)

    assert result is None


def test_calculate_kudos_per_hour_one_hour() -> None:
    """Test kudos per hour calculation with exactly one productive hour."""
    kudos_generated = 150.0
    eligible_seconds = 3600  # 1 productive hour

    result = KudosCalculator.calculate_kudos_per_hour(kudos_generated, eligible_seconds)

    assert result == 150.0


def test_calculate_kudos_totals_past_hour_all_recent() -> None:
    """Test calculating kudos totals when all events are within past hour."""
    current_time = time.time()
    kudos_events = deque(
        [
            (current_time - 1800, 50.0),  # 30 minutes ago
            (current_time - 900, 30.0),  # 15 minutes ago
            (current_time - 300, 20.0),  # 5 minutes ago
        ],
        maxlen=100,
    )

    total, cleaned = KudosCalculator.calculate_kudos_totals_past_hour(kudos_events)

    assert total == 100.0
    assert len(cleaned) == 3


def test_calculate_kudos_totals_past_hour_some_old() -> None:
    """Test calculating kudos totals when some events are older than 1 hour."""
    current_time = time.time()
    kudos_events = deque(
        [
            (current_time - 7200, 100.0),  # 2 hours ago (should be excluded)
            (current_time - 3700, 75.0),  # 1 hour 2 minutes ago (should be excluded)
            (current_time - 1800, 50.0),  # 30 minutes ago
            (current_time - 900, 30.0),  # 15 minutes ago
        ],
        maxlen=100,
    )

    total, cleaned = KudosCalculator.calculate_kudos_totals_past_hour(kudos_events)

    assert total == 80.0  # Only the last two events
    assert len(cleaned) == 2


def test_calculate_kudos_totals_past_hour_all_old() -> None:
    """Test calculating kudos totals when all events are older than 1 hour."""
    current_time = time.time()
    kudos_events = deque(
        [
            (current_time - 7200, 100.0),  # 2 hours ago
            (current_time - 5400, 75.0),  # 1.5 hours ago
        ],
        maxlen=100,
    )

    total, cleaned = KudosCalculator.calculate_kudos_totals_past_hour(kudos_events)

    assert total == 0.0
    assert len(cleaned) == 0


def test_calculate_kudos_totals_past_hour_empty() -> None:
    """Test calculating kudos totals with empty events."""
    kudos_events: deque[tuple[float, float]] = deque([], maxlen=100)

    total, cleaned = KudosCalculator.calculate_kudos_totals_past_hour(kudos_events)

    assert total == 0.0
    assert len(cleaned) == 0


def test_calculate_kudos_totals_past_hour_maxlen_preserved() -> None:
    """Test that maxlen is preserved in cleaned deque."""
    current_time = time.time()
    original_maxlen = 50
    kudos_events = deque(
        [
            (current_time - 1800, 50.0),
            (current_time - 900, 30.0),
        ],
        maxlen=original_maxlen,
    )

    _total, cleaned = KudosCalculator.calculate_kudos_totals_past_hour(kudos_events)

    assert cleaned.maxlen == original_maxlen


def test_calculate_all_metrics_integration() -> None:
    """Test calculate_all_metrics returns the session rate over productive seconds and the past-hour total."""
    current_time = time.time()
    kudos_generated_this_session = 200.0
    eligible_seconds_total = 3000.0  # 50 productive minutes (10 of the last hour were idle)
    kudos_events = deque(
        [
            (current_time - 1800, 100.0),  # 30 minutes ago
            (current_time - 900, 100.0),  # 15 minutes ago
        ],
        maxlen=100,
    )

    (
        echoed_eligible_seconds,
        kudos_per_hour_session,
        kudos_total_past_hour,
        cleaned_events,
    ) = KudosCalculator.calculate_all_metrics(
        kudos_generated_this_session,
        eligible_seconds_total,
        kudos_events,
    )

    assert echoed_eligible_seconds == eligible_seconds_total

    # 200 kudos over 50 productive minutes = 240/hr
    assert kudos_per_hour_session is not None
    assert abs(kudos_per_hour_session - 240.0) < 1

    # Both events are within the hour
    assert kudos_total_past_hour == 200.0
    assert len(cleaned_events) == 2


def test_calculate_all_metrics_with_old_events() -> None:
    """Test calculate_all_metrics properly cleans old events."""
    current_time = time.time()
    kudos_generated_this_session = 300.0
    eligible_seconds_total = 6000.0
    kudos_events = deque(
        [
            (current_time - 7000, 50.0),  # ~2 hours ago (should be excluded)
            (current_time - 5000, 50.0),  # ~1.4 hours ago (should be excluded)
            (current_time - 1800, 100.0),  # 30 minutes ago
            (current_time - 900, 100.0),  # 15 minutes ago
        ],
        maxlen=100,
    )

    (
        _echoed_eligible_seconds,
        _kudos_per_hour_session,
        kudos_total_past_hour,
        cleaned_events,
    ) = KudosCalculator.calculate_all_metrics(
        kudos_generated_this_session,
        eligible_seconds_total,
        kudos_events,
    )

    # Only the recent events count toward the past-hour total.
    assert kudos_total_past_hour == 200.0
    assert len(cleaned_events) == 2


def test_calculate_all_metrics_warming_up() -> None:
    """Before any productive time accrues, the session rate is None ("warming up")."""
    (
        _echoed_eligible_seconds,
        kudos_per_hour_session,
        _kudos_total_past_hour,
        _cleaned_events,
    ) = KudosCalculator.calculate_all_metrics(
        0.0,
        0.0,
        deque([], maxlen=100),
    )

    assert kudos_per_hour_session is None
