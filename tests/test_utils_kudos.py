"""Tests for kudos utility functions."""

from horde_worker_regen.utils.kudos_utils import generate_kudos_info_string


def test_generate_kudos_info_string_short_session() -> None:
    """A session with under an hour of productive time reports the span in minutes."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=100.0,
        eligible_seconds_total=1800.0,  # 30 productive minutes
        kudos_per_hour_session=200.0,
        kudos_total_past_hour=100.0,
    )

    assert "30.00 productive minutes" in result
    assert "100.00" in result  # kudos generated
    assert "Session: 200.00 kudos/hr" in result


def test_generate_kudos_info_string_long_session() -> None:
    """A session with over an hour of productive time reports the span in hours."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=500.0,
        eligible_seconds_total=7200.0,  # 2 productive hours
        kudos_per_hour_session=250.0,
        kudos_total_past_hour=300.0,
    )

    assert "2.00 productive hours" in result
    assert "500.00" in result  # kudos generated
    assert "Session: 250.00 kudos/hr" in result


def test_generate_kudos_info_string_warming_up() -> None:
    """Before any productive time accrues the rate reads as warming up, not a misleading number."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=0.0,
        eligible_seconds_total=0.0,
        kudos_per_hour_session=None,
        kudos_total_past_hour=0.0,
    )

    assert "warming up" in result
    assert "kudos/hr" not in result.split("|")[1]  # the rate element carries no number yet


def test_generate_kudos_info_string_past_hour_always_shown() -> None:
    """The rolling past-hour total is always part of the line."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=500.0,
        eligible_seconds_total=7200.0,
        kudos_per_hour_session=250.0,
        kudos_total_past_hour=321.0,
    )

    assert "Past hour: 321.00 kudos" in result


def test_generate_kudos_info_string_format() -> None:
    """The line is pipe-separated into its span, rate, and past-hour parts."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=100.0,
        eligible_seconds_total=1800.0,
        kudos_per_hour_session=200.0,
        kudos_total_past_hour=100.0,
    )

    parts = result.split("|")
    assert len(parts) == 3


def test_generate_kudos_info_string_large_values() -> None:
    """Large values are comma-formatted."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=1_234_567.89,
        eligible_seconds_total=36000.0,  # 10 productive hours
        kudos_per_hour_session=123_456.78,
        kudos_total_past_hour=120_000.0,
    )

    assert "1,234,567.89" in result
    assert "123,456.78" in result
