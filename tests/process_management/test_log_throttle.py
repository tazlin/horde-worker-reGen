"""Tests for the shared per-key log throttle."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management._internal.util import (
    reset_log_throttles,
    throttled_log_level,
)


@pytest.fixture(autouse=True)
def _clear_throttles() -> None:
    """Start every test with an empty throttle registry."""
    reset_log_throttles()


class TestThrottledLogLevel:
    """The throttle must pass the first emission per key and demote repeats within the interval."""

    def test_first_emission_uses_the_normal_level(self) -> None:
        """A key that has never been emitted logs at its normal level."""
        assert throttled_log_level("key", 30.0, now=100.0) == "DEBUG"

    def test_repeat_within_the_interval_is_demoted(self) -> None:
        """A repeat inside the interval is demoted rather than dropped."""
        assert throttled_log_level("key", 30.0, now=100.0) == "DEBUG"
        assert throttled_log_level("key", 30.0, now=110.0) == "TRACE"
        assert throttled_log_level("key", 30.0, now=129.9) == "TRACE"

    def test_emission_after_the_interval_passes_again(self) -> None:
        """Once the interval has elapsed the key emits at its normal level again."""
        assert throttled_log_level("key", 30.0, now=100.0) == "DEBUG"
        assert throttled_log_level("key", 30.0, now=115.0) == "TRACE"
        assert throttled_log_level("key", 30.0, now=130.0) == "DEBUG"

    def test_a_demoted_emission_does_not_extend_the_interval(self) -> None:
        """Suppressed repeats do not reset the window, so a busy key still emits once per interval."""
        assert throttled_log_level("key", 30.0, now=0.0) == "DEBUG"
        for tick in range(1, 30):
            assert throttled_log_level("key", 30.0, now=float(tick)) == "TRACE"
        assert throttled_log_level("key", 30.0, now=30.0) == "DEBUG"

    def test_keys_are_throttled_independently(self) -> None:
        """One key's emission must not suppress another's."""
        assert throttled_log_level("a", 30.0, now=100.0) == "DEBUG"
        assert throttled_log_level("b", 30.0, now=100.0) == "DEBUG"
        assert throttled_log_level("a", 30.0, now=101.0) == "TRACE"
        assert throttled_log_level("b", 30.0, now=101.0) == "TRACE"

    def test_levels_are_configurable(self) -> None:
        """Callers can choose the levels the throttle selects between."""
        assert throttled_log_level("key", 30.0, normal_level="INFO", suppressed_level="DEBUG", now=0.0) == "INFO"
        assert throttled_log_level("key", 30.0, normal_level="INFO", suppressed_level="DEBUG", now=1.0) == "DEBUG"

    def test_reset_clears_every_key(self) -> None:
        """Resetting the registry lets each key emit at its normal level again."""
        assert throttled_log_level("key", 30.0, now=0.0) == "DEBUG"
        assert throttled_log_level("key", 30.0, now=1.0) == "TRACE"

        reset_log_throttles()

        assert throttled_log_level("key", 30.0, now=1.0) == "DEBUG"

    def test_default_clock_is_monotonic(self) -> None:
        """Omitting the clock uses the process monotonic clock, so real callers need no argument."""
        assert throttled_log_level("key", 30.0) == "DEBUG"
        assert throttled_log_level("key", 30.0) == "TRACE"
