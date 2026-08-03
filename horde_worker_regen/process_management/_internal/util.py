import time
from datetime import datetime

_time_units = [
    ("year", 365 * 24 * 3600),
    ("month", 30 * 24 * 3600),
    ("day", 24 * 3600),
    ("hour", 3600),
    ("minute", 60),
]


def dt_to_td_str(dt: datetime) -> str | None:
    """Convert a datetime to a human-readable time difference string."""
    now = datetime.now()
    time_difference = (now - dt).total_seconds()

    chosen: tuple[str, int] | None = None

    for unit, seconds_in_unit in _time_units:
        if time_difference >= seconds_in_unit:
            chosen = (unit, seconds_in_unit)

    if chosen is None:
        chosen = ("second", 1)

    unit, seconds_in_unit = chosen
    count = int(time_difference / seconds_in_unit)
    return f"{count} {unit}{'' if count == 1 else 's'} ago"


_log_throttle_last_emission: dict[str, float] = {}
"""Monotonic time each throttle key last emitted at its normal level."""


def throttled_log_level(
    key: str,
    interval_seconds: float,
    *,
    normal_level: str = "DEBUG",
    suppressed_level: str = "TRACE",
    now: float | None = None,
) -> str:
    """Pick the level a repeating log line should use, letting one emission per key per interval through.

    Telemetry that fires at a tick rate (memory reports, per-message dispatch notices) is worth keeping
    at full fidelity but is not worth a ``DEBUG`` line every tick: at multiple lines per second it
    crowds out everything else in the operator-facing log. Callers pass a key that identifies the
    repeating line (typically the call site plus the process it describes) and log at the returned
    level, so at most one emission per interval lands at the normal level and the rest are demoted to
    the suppressed level rather than dropped.

    Suppressed emissions do not restart the window, so a key logging continuously still emits at the
    normal level once per interval instead of falling silent.

    Args:
        key (str): Identifies the repeating line. Distinct keys throttle independently.
        interval_seconds (float): Minimum seconds between normal-level emissions for this key.
        normal_level (str): The level to use when the key is due to emit.
        suppressed_level (str): The level to demote to while the key is inside its interval.
        now (float | None): Monotonic-clock reading to evaluate against, or None to sample the clock.

    Returns:
        The loguru level name the caller should log this emission at.
    """
    current = time.monotonic() if now is None else now
    last_emission = _log_throttle_last_emission.get(key)

    if last_emission is not None and current - last_emission < interval_seconds:
        return suppressed_level

    _log_throttle_last_emission[key] = current
    return normal_level


def reset_log_throttles() -> None:
    """Forget every throttle key, so the next emission of each lands at its normal level."""
    _log_throttle_last_emission.clear()
