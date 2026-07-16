"""Contracts for the shared-channel corruption watchdog.

A physical queue read that cannot return (a torn length-prefixed frame, a stalled shared writer lock) is
unrecoverable in place, because every child inherits the same queue. The dispatcher's reader stamps when it
enters a read and clears the stamp on return; the watchdog judges the channel corrupt once that stamp has
stood past a threshold while the queue still reports a backlog, and escalates exactly once through the
worker's terminal machinery. These lock that behaviour without spinning up the real reader thread: the stamp
and the queue depth are driven directly.
"""

from __future__ import annotations

import queue
import time

from tests.process_management.ipc.test_dispatcher_drain_liveness import _make_dispatcher


class _DepthQueue:
    """A queue double that only answers ``qsize()``; the watchdog never reads or polls it."""

    def __init__(self, depth: int) -> None:
        self._depth = depth

    def empty(self) -> bool:
        return self._depth == 0

    def get(self, block: bool = False, timeout: float | None = None) -> object:
        raise queue.Empty

    def qsize(self) -> int:
        return self._depth


class _NoDepthQueue(_DepthQueue):
    """A queue double whose depth cannot be reported, as on platforms without a shared counter."""

    def qsize(self) -> int:
        raise NotImplementedError


class TestChannelCorruptionWatchdog:
    """A read stuck past the threshold with backlog escalates once; a healthy reader never escalates."""

    def test_stuck_read_with_backlog_fires_escalation_once(self) -> None:
        """A stale read stamp plus a reported backlog logs and fires the terminal handler exactly once."""
        dispatcher = _make_dispatcher(process_message_queue=_DepthQueue(depth=3))
        fired: list[str] = []
        dispatcher.set_channel_corruption_handler(fired.append)

        # A read that entered well past the threshold and never returned.
        dispatcher._reader_read_started_at = time.monotonic() - (
            dispatcher._CHANNEL_STUCK_READ_THRESHOLD_SECONDS + 5.0
        )

        assert dispatcher.check_message_channel_health() is True
        assert len(fired) == 1

        # A second maintenance tick still reports corrupt but must not re-fire the terminal handler.
        assert dispatcher.check_message_channel_health() is True
        assert len(fired) == 1

    def test_healthy_reader_never_escalates(self) -> None:
        """With no read in flight, the channel is healthy and the handler is never called."""
        dispatcher = _make_dispatcher(process_message_queue=_DepthQueue(depth=3))
        fired: list[str] = []
        dispatcher.set_channel_corruption_handler(fired.append)

        dispatcher._reader_read_started_at = None

        assert dispatcher.check_message_channel_health() is False
        assert fired == []

    def test_brief_read_is_not_corruption(self) -> None:
        """A read in flight for less than the threshold is normal and does not escalate."""
        dispatcher = _make_dispatcher(process_message_queue=_DepthQueue(depth=3))
        fired: list[str] = []
        dispatcher.set_channel_corruption_handler(fired.append)

        dispatcher._reader_read_started_at = time.monotonic() - 1.0

        assert dispatcher.check_message_channel_health() is False
        assert fired == []

    def test_stuck_read_without_backlog_is_not_corruption(self) -> None:
        """A stale stamp but an empty queue is not judged corrupt: there is no unconsumed frame to explain it."""
        dispatcher = _make_dispatcher(process_message_queue=_DepthQueue(depth=0))
        fired: list[str] = []
        dispatcher.set_channel_corruption_handler(fired.append)

        dispatcher._reader_read_started_at = time.monotonic() - (
            dispatcher._CHANNEL_STUCK_READ_THRESHOLD_SECONDS + 5.0
        )

        assert dispatcher.check_message_channel_health() is False
        assert fired == []

    def test_missing_depth_counter_does_not_veto_corruption(self) -> None:
        """A queue that cannot report depth must not block the verdict: a read stuck this long is evidence."""
        dispatcher = _make_dispatcher(process_message_queue=_NoDepthQueue(depth=0))
        fired: list[str] = []
        dispatcher.set_channel_corruption_handler(fired.append)

        dispatcher._reader_read_started_at = time.monotonic() - (
            dispatcher._CHANNEL_STUCK_READ_THRESHOLD_SECONDS + 5.0
        )

        assert dispatcher.check_message_channel_health() is True
        assert len(fired) == 1
