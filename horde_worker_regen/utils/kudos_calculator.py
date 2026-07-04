"""Kudos calculation and analysis."""

from __future__ import annotations

import time
from collections import deque


class KudosCalculator:
    """Handles kudos calculations and metrics."""

    @staticmethod
    def calculate_kudos_per_hour(kudos_generated: float, eligible_seconds: float) -> float | None:
        """Calculate kudos per hour over productive time.

        Args:
            kudos_generated: Total kudos generated.
            eligible_seconds: Productive (pipeline-occupied) seconds since the first submit. This is the
                honest denominator: it excludes the cold-start lead-in and any idle, maintenance, or
                drained-pause time during which no kudos could be earned.

        Returns:
            Kudos per hour rate, or None until productive time has accrued (a cold worker is "warming up").
        """
        if eligible_seconds <= 0:
            return None
        return kudos_generated / eligible_seconds * 3600

    @staticmethod
    def calculate_kudos_totals_past_hour(
        kudos_events: deque[tuple[float, float]],
    ) -> tuple[float, deque[tuple[float, float]]]:
        """Calculate the total kudos generated in the past hour and clean up old events.

        Args:
            kudos_events: Deque of (timestamp, kudos) tuples.

        Returns:
            Tuple of (total kudos in past hour, cleaned kudos events deque).
        """
        kudos_total_past_hour = 0.0
        num_events_found = 0
        current_time = time.time()

        for event_time, kudos in reversed(kudos_events):
            if current_time - event_time > 3600:
                break

            num_events_found += 1
            kudos_total_past_hour += kudos

        # Remove events older than 1 hour
        elements_to_remove = len(kudos_events) - num_events_found
        if elements_to_remove > 0:
            # Create new deque with only recent events
            if num_events_found > 0:
                cleaned_events = deque(list(kudos_events)[-num_events_found:], maxlen=kudos_events.maxlen)
            else:
                # All events are old, return empty deque
                cleaned_events = deque(maxlen=kudos_events.maxlen)
        else:
            cleaned_events = kudos_events

        return kudos_total_past_hour, cleaned_events

    @staticmethod
    def calculate_all_metrics(
        kudos_generated_this_session: float,
        eligible_seconds_total: float,
        kudos_events: deque[tuple[float, float]],
    ) -> tuple[float, float | None, float, deque[tuple[float, float]]]:
        """Calculate the session kudos metrics over productive time.

        Args:
            kudos_generated_this_session: Total kudos generated in this session.
            eligible_seconds_total: Productive (pipeline-occupied) seconds since the first submit; the
                honest rate denominator (see :meth:`calculate_kudos_per_hour`).
            kudos_events: Deque of (timestamp, kudos) tuples.

        Returns:
            Tuple of:
            - eligible_seconds_total (echoed back for display)
            - kudos_per_hour_session (None until productive time has accrued)
            - kudos_total_past_hour
            - cleaned kudos_events deque
        """
        kudos_per_hour_session = KudosCalculator.calculate_kudos_per_hour(
            kudos_generated_this_session,
            eligible_seconds_total,
        )

        kudos_total_past_hour, cleaned_events = KudosCalculator.calculate_kudos_totals_past_hour(
            kudos_events,
        )

        return (
            eligible_seconds_total,
            kudos_per_hour_session,
            kudos_total_past_hour,
            cleaned_events,
        )
