"""Kudos calculation and formatting utility functions."""

from __future__ import annotations


def generate_kudos_info_string(
    kudos_generated_this_session: float,
    eligible_seconds_total: float,
    kudos_per_hour_session: float | None,
    kudos_total_past_hour: float,
) -> str:
    """Generate a string with information about the kudos generated in the current session.

    The rate is measured over *productive* time (see the kudos calculator), so it is honest from the first
    job onward: there is no cold-start extrapolation and no separate "active" figure, because idle,
    maintenance, and drained-pause time are already excluded from the denominator.

    Args:
        kudos_generated_this_session: The kudos generated in the current session.
        eligible_seconds_total: Productive seconds since the first submit (the rate's denominator).
        kudos_per_hour_session: The kudos per hour over productive time, or None while still warming up.
        kudos_total_past_hour: The total kudos generated in the past hour.

    Returns:
        A string with information about the kudos generated in the current session.
    """
    if eligible_seconds_total <= 3600:
        span_element = (
            f"Total Session Kudos: {kudos_generated_this_session:,.2f} over "
            f"{eligible_seconds_total / 60:.2f} productive minutes"
        )
    else:
        span_element = (
            f"Total Session Kudos: {kudos_generated_this_session:,.2f} over "
            f"{eligible_seconds_total / 3600:.2f} productive hours"
        )

    if kudos_per_hour_session is None:
        rate_element = "Session: warming up (measured after the first job)"
    else:
        rate_element = f"Session: {kudos_per_hour_session:,.2f} kudos/hr"

    past_hour_element = f"Past hour: {kudos_total_past_hour:,.2f} kudos"

    return " | ".join([span_element, rate_element, past_hour_element])
