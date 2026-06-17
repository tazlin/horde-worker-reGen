"""Unit tests for the level runner's stall watchdog (the hang-diagnosis dump trigger).

The watchdog is what turns a silent wedge (the observed pre-spawn startup hang, where the manager log
simply stopped and the ``.faulthandler`` file was empty) into a thread-stack dump. Its timing is driven
by an injectable clock so these assertions are deterministic and need no real waiting.
"""

from __future__ import annotations

from horde_worker_regen.benchmark.level_runner import _StallWatchdog


def test_dumps_after_threshold_without_progress() -> None:
    """With no progress, the watchdog dumps exactly when the stall threshold is crossed (not before)."""
    now = [100.0]
    dumps: list[float] = []
    watchdog = _StallWatchdog(stall_seconds=30.0, dump=dumps.append, clock=lambda: now[0])

    now[0] = 129.0  # 29s since construction
    assert watchdog.check() is False
    assert not dumps

    now[0] = 131.0  # 31s >= 30s
    assert watchdog.check() is True
    assert len(dumps) == 1
    assert dumps[0] >= 30.0


def test_dumps_only_once_per_stall_episode() -> None:
    """A persistent wedge yields a single dump, not one on every poll (no log/file spam)."""
    now = [0.0]
    dumps: list[float] = []
    watchdog = _StallWatchdog(stall_seconds=10.0, dump=dumps.append, clock=lambda: now[0])

    now[0] = 50.0
    assert watchdog.check() is True
    now[0] = 100.0
    assert watchdog.check() is False
    assert len(dumps) == 1


def test_progress_resets_timer_and_rearms_the_dump() -> None:
    """Changing progress keeps resetting the timer; once it stops, the watchdog dumps and can rearm."""
    now = [0.0]
    dumps: list[float] = []
    watchdog = _StallWatchdog(stall_seconds=10.0, dump=dumps.append, clock=lambda: now[0])

    # Progress keeps arriving inside the threshold: never dumps.
    for moment in (5.0, 9.0, 14.0, 19.0):
        now[0] = moment
        watchdog.note_progress(("phase", int(moment), 0, 0))
        assert watchdog.check() is False
    assert not dumps

    # Progress stops after t=19; the threshold is measured from that last change.
    now[0] = 30.0
    assert watchdog.check() is True
    assert len(dumps) == 1

    # An identical signature is a no-op (does not rearm); a genuinely new signature rearms the dump.
    watchdog.note_progress(("phase", 19, 0, 0))
    watchdog.note_progress(("phase", 99, 0, 0))
    now[0] = 41.0
    assert watchdog.check() is True
    assert len(dumps) == 2
