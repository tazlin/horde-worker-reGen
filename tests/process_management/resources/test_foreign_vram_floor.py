"""Unit tests for the sustained foreign-VRAM floor tracker.

The tracker smooths the per-tick foreign reading (``total - device_free - worker_footprint``) into the
minimum over a trailing window, on a hand-advanced monotonic clock. Taking the minimum makes a transient
foreign spike unable to raise the floor (and so unable to wrongly deny a servable model), while the warm-up
gate withholds any floor until a full window has been observed so a cold-start transient cannot set it early.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources.foreign_vram_floor import (
    FOREIGN_FLOOR_WINDOW_SECONDS,
    ForeignVramFloorTracker,
)


def test_no_floor_until_a_full_window_is_observed() -> None:
    """Within the warm-up window the floor is withheld (None), preserving the pre-foreign boundary."""
    tracker = ForeignVramFloorTracker()
    assert tracker.update(0, 1900.0, now=0.0) is None
    assert tracker.update(0, 1900.0, now=FOREIGN_FLOOR_WINDOW_SECONDS - 1.0) is None
    # The first sample at or beyond a full window of observation yields the sustained minimum.
    assert tracker.update(0, 1900.0, now=FOREIGN_FLOOR_WINDOW_SECONDS) == 1900.0


def test_sustained_minimum_over_the_window() -> None:
    """The reported floor is the minimum foreign reading across the trailing window's samples."""
    tracker = ForeignVramFloorTracker()
    readings = [2200.0, 1900.0, 2100.0, 2000.0]
    floor = None
    for index, foreign in enumerate(readings):
        floor = tracker.update(0, foreign, now=float(index))
    # Still inside warm-up (span 3s << window), so no floor is reported yet.
    assert floor is None
    assert tracker.update(0, 2050.0, now=FOREIGN_FLOOR_WINDOW_SECONDS) == 1900.0


def test_transient_spike_does_not_raise_the_floor() -> None:
    """A brief foreign spike raises the instantaneous reading but not the window minimum."""
    tracker = ForeignVramFloorTracker()
    # A steady 1900 floor established over a full window.
    step = 10.0
    now = 0.0
    while now <= FOREIGN_FLOOR_WINDOW_SECONDS:
        tracker.update(0, 1900.0, now=now)
        now += step
    # A game opens: one sample spikes to 9000. The sustained floor stays at the steady minimum.
    spiked = tracker.update(0, 9000.0, now=now)
    assert spiked == 1900.0


def test_missing_worker_report_contributes_no_sample() -> None:
    """A None reading (children not yet reporting VRAM) adds no sample and holds the current floor."""
    tracker = ForeignVramFloorTracker()
    now = 0.0
    while now <= FOREIGN_FLOOR_WINDOW_SECONDS:
        tracker.update(0, 1900.0, now=now)
        now += 10.0
    established = tracker.update(0, 1900.0, now=now)
    assert established == 1900.0
    # A tick with no measurable foreign reading must not disturb the established floor.
    assert tracker.update(0, None, now=now + 1.0) == 1900.0


def test_stale_samples_age_out_and_the_floor_is_forgotten() -> None:
    """When readings stop for longer than the window, the aged-out history forgets the floor (None)."""
    tracker = ForeignVramFloorTracker()
    now = 0.0
    while now <= FOREIGN_FLOOR_WINDOW_SECONDS:
        tracker.update(0, 1900.0, now=now)
        now += 10.0
    assert tracker.update(0, 1900.0, now=now) == 1900.0
    # A long gap with no reports: every sample falls outside the trailing window and is pruned.
    assert tracker.update(0, None, now=now + 2 * FOREIGN_FLOOR_WINDOW_SECONDS) is None


def test_per_card_isolation() -> None:
    """Each card keeps its own sample history and floor."""
    tracker = ForeignVramFloorTracker()
    now = 0.0
    while now <= FOREIGN_FLOOR_WINDOW_SECONDS:
        tracker.update(0, 1900.0, now=now)
        tracker.update(1, 3500.0, now=now)
        now += 10.0
    assert tracker.update(0, 1900.0, now=now) == 1900.0
    assert tracker.update(1, 3500.0, now=now) == 3500.0
