"""Unit tests for the auxiliary-download backoff escalation/decay logic."""

from __future__ import annotations

from horde_worker_regen.process_management.models.aux_download_backoff import (
    BASE_BACKOFF_SECONDS,
    MAX_BACKOFF_SECONDS,
    STRIKE_DECAY_SECONDS,
    AuxDownloadBackoff,
)


def test_first_strike_uses_base_window_and_suppresses() -> None:
    """The first strike suppresses pops for exactly the base window."""
    backoff = AuxDownloadBackoff()
    assert not backoff.pops_suppressed(now=1000.0)

    window = backoff.register_timeout(now=1000.0)

    assert window == BASE_BACKOFF_SECONDS
    assert backoff.strikes == 1
    assert backoff.pops_suppressed(now=1000.0)
    assert backoff.pops_suppressed(now=1000.0 + BASE_BACKOFF_SECONDS - 1)
    assert not backoff.pops_suppressed(now=1000.0 + BASE_BACKOFF_SECONDS + 1)


def test_consecutive_strikes_double_the_window() -> None:
    """Strikes within an incident escalate the window geometrically."""
    backoff = AuxDownloadBackoff()
    now = 1000.0
    windows = []
    for _ in range(4):
        windows.append(backoff.register_timeout(now=now))
        # Re-strike before the window or the decay elapses: a continuing incident.
        now += 1.0
    assert windows == [60.0, 120.0, 240.0, 480.0]
    assert backoff.strikes == 4


def test_window_is_capped() -> None:
    """A sustained incident holds at the maximum window rather than growing unbounded."""
    backoff = AuxDownloadBackoff()
    now = 1000.0
    window = 0.0
    for _ in range(20):
        window = backoff.register_timeout(now=now)
        now += 1.0
    assert window == MAX_BACKOFF_SECONDS


def test_strike_after_quiet_period_resets_escalation() -> None:
    """A strike after the decay window starts a fresh incident at the base window."""
    backoff = AuxDownloadBackoff()
    backoff.register_timeout(now=1000.0)
    backoff.register_timeout(now=1001.0)
    assert backoff.strikes == 2

    # A strike well after the previous one starts a fresh incident at the base window.
    later = 1001.0 + STRIKE_DECAY_SECONDS + 1
    window = backoff.register_timeout(now=later)
    assert window == BASE_BACKOFF_SECONDS
    assert backoff.strikes == 1


def test_is_escalation_active_tracks_suppression_then_decays() -> None:
    """The escalation stays active through suppression and the decay window, then self-expires."""
    backoff = AuxDownloadBackoff()
    assert backoff.is_escalation_active(now=1000.0) is False

    backoff.register_timeout(now=1000.0)

    # Active while suppressed.
    assert backoff.is_escalation_active(now=1000.0 + 1) is True
    # Still active after the (short) window but within the decay window: a recent strike.
    assert backoff.is_escalation_active(now=1000.0 + BASE_BACKOFF_SECONDS + 1) is True
    # Inactive once the decay window has fully elapsed, without mutating the strike count.
    assert backoff.is_escalation_active(now=1000.0 + STRIKE_DECAY_SECONDS + 1) is False


def test_remaining_seconds() -> None:
    """remaining_seconds counts down to the resume time and floors at zero."""
    backoff = AuxDownloadBackoff()
    backoff.register_timeout(now=1000.0)
    assert backoff.remaining_seconds(now=1000.0) == BASE_BACKOFF_SECONDS
    assert backoff.remaining_seconds(now=1000.0 + BASE_BACKOFF_SECONDS + 5) == 0.0
