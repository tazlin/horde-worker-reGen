"""Tests for the control-loop diagnostic throttle."""

from __future__ import annotations

from horde_worker_regen.process_management.scheduling.diagnostic_throttle import (
    DIAGNOSTIC_MB_BUCKET,
    DiagnosticThrottle,
    diagnostic_mb_bucket,
    suppressed_suffix,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestDiagnosticThrottle:
    """Emission decisions for one named diagnostic."""

    def test_unchanged_diagnostic_is_suppressed_until_interval(self) -> None:
        """An unchanged state emits once, then again only after the repeat interval with the swallowed count."""
        clock = _Clock()
        throttle = DiagnosticThrottle(clock, repeat_seconds=30.0)

        assert throttle.suppressed_count("diagnostic", ("same",)) == 0
        assert throttle.suppressed_count("diagnostic", ("same",)) is None
        assert throttle.suppressed_count("diagnostic", ("same",)) is None

        clock.now += 31.0
        assert throttle.suppressed_count("diagnostic", ("same",)) == 2
        assert throttle.suppressed_count("diagnostic", ("same",)) is None

    def test_changed_diagnostic_logs_immediately(self) -> None:
        """A state change emits at once, carrying the repeats suppressed under the previous state."""
        throttle = DiagnosticThrottle(_Clock())

        assert throttle.suppressed_count("diagnostic", ("old",)) == 0
        assert throttle.suppressed_count("diagnostic", ("old",)) is None
        assert throttle.suppressed_count("diagnostic", ("new",)) == 1

    def test_names_are_independent(self) -> None:
        """Two diagnostics under different names do not suppress each other."""
        throttle = DiagnosticThrottle(_Clock())

        assert throttle.suppressed_count("a", ("k",)) == 0
        assert throttle.suppressed_count("b", ("k",)) == 0
        assert throttle.suppressed_count("a", ("k",)) is None

    def test_forget_and_reset_restart_the_run(self) -> None:
        """Forgetting a name (or resetting all) makes its next observation a first observation again."""
        throttle = DiagnosticThrottle(_Clock())
        assert throttle.suppressed_count("a", ("k",)) == 0
        assert throttle.suppressed_count("b", ("k",)) == 0
        assert throttle.suppressed_count("a", ("k",)) is None

        throttle.forget("a")
        assert throttle.suppressed_count("a", ("k",)) == 0
        assert throttle.suppressed_count("b", ("k",)) is None

        throttle.reset()
        assert throttle.suppressed_count("b", ("k",)) == 0


def test_suppressed_suffix_is_empty_without_repeats() -> None:
    """Only a positive repeat count produces a suffix."""
    assert suppressed_suffix(0) == ""
    assert suppressed_suffix(-1) == ""
    assert suppressed_suffix(3) == " (suppressed 3 unchanged repeats)"


def test_diagnostic_mb_bucket_folds_jitter() -> None:
    """Figures within one bucket share a value; None passes through."""
    assert diagnostic_mb_bucket(None) is None
    assert diagnostic_mb_bucket(0.0) == 0
    assert diagnostic_mb_bucket(DIAGNOSTIC_MB_BUCKET * 4 + 10) == diagnostic_mb_bucket(DIAGNOSTIC_MB_BUCKET * 4 - 10)
    assert diagnostic_mb_bucket(DIAGNOSTIC_MB_BUCKET * 8) != diagnostic_mb_bucket(DIAGNOSTIC_MB_BUCKET * 4)
