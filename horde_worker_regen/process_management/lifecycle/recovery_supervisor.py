"""Save-our-ship escalation policy: decide when a wedged worker should soft-reset or give up.

The lower layers detect and recover individual failures (a slot crashes -> replaced; a slot crash-loops
-> quarantined; a job faults -> bounded/degraded retry). This supervisor sits above them and answers a
different question: *the worker as a whole has stopped making progress on work it has accepted. Now
what?* Continued operation is the paramount goal, so the escalation is, in order:

1. **Soft reset** (bounded): rebuild the worker's process pools in-place (kill and re-spawn every child,
   un-quarantine slots, reduce settings a notch for "limp-by") without restarting the parent process or
   detaching the TUI. A transient wedge (a bad model load, a one-off deadlock) recovers here.
2. **Give up cleanly**: once the worker has been wedged long enough that resets clearly are not helping
   (e.g. a deterministic crash-on-start), stop fighting: fault the jobs that cannot be served so the
   horde reissues them, rather than wedging forever. The worker keeps running and keeps popping.

This module is *only the policy*: it tracks how long the worker has been wedged and returns an action.
The manager owns the wedge assessment (what "wedged" means in terms of live processes and pending work)
and the actions (rebuilding pools, applying limp-by, faulting stuck jobs). Keeping the policy pure makes
the escalation timing unit-testable with a fake clock.
"""

from __future__ import annotations

import enum
import math
import time
from collections.abc import Callable

_DEFAULT_WEDGE_GRACE_SECONDS = 1.5
"""How long a wedge must persist continuously before the first soft reset, to ride out brief blips."""

_DEFAULT_RESET_INTERVAL_SECONDS = 2.0
"""Minimum spacing between soft resets, so a reset has a chance to take effect before the next."""

_DEFAULT_MAX_SOFT_RESETS = 1
"""Soft resets attempted within one wedge episode before escalating to give-up.

One is enough to clear a genuinely transient wedge; a persistent fault (crash-on-start) will not be
fixed by repeating the reset, so the time-based give-up below is what ends the episode."""

_DEFAULT_GIVE_UP_AFTER_SECONDS = 7.0
"""Episode age after which the worker gives up: only reached when the wedge signal is definitive.

The manager assesses a wedge only on the crash-loop signals (every inference slot quarantined, or the
safety pool crash-looping with no healthy process), which a slow model load or normal replacement never
trips, so this short budget is safe: it is measured over an episode that stays open only because those
definitive signals persist, and a soft reset's brief un-quarantine does not close it (see clean streak)."""

_DEFAULT_CLEAN_STREAK_SECONDS = 30.0
"""Continuous non-wedge time that ends a wedge episode and restores full settings (limp-by recovery).

Set longer than a soft reset's transient un-quarantine + re-quarantine window so a doomed pool's soft
reset does not look like a recovery and reopen a fresh (give-up-resetting) episode each time."""


class RecoveryAction(enum.Enum):
    """What the recovery supervisor wants the manager to do this tick."""

    NONE = enum.auto()
    """No action: not wedged, or still within a grace/backoff window."""
    SOFT_RESET = enum.auto()
    """Rebuild the process pools in-place and drop one limp-by notch."""
    GIVE_UP = enum.auto()
    """Fault the jobs that cannot be served so they drain; the worker keeps running."""


class RecoverySupervisor:
    """Tracks how long the worker has been wedged and escalates soft reset -> give up.

    Drive it once per control-loop tick with :meth:`evaluate`, passing whether the worker is currently
    wedged (pending work it structurally cannot make progress on). The returned :class:`RecoveryAction`
    tells the manager what to do; :attr:`limp_by_level` is the number of soft resets done in the current
    episode, which the manager maps to reduced settings.
    """

    def __init__(
        self,
        *,
        wedge_grace_seconds: float = _DEFAULT_WEDGE_GRACE_SECONDS,
        reset_interval_seconds: float = _DEFAULT_RESET_INTERVAL_SECONDS,
        max_soft_resets: int = _DEFAULT_MAX_SOFT_RESETS,
        give_up_after_seconds: float = _DEFAULT_GIVE_UP_AFTER_SECONDS,
        clean_streak_seconds: float = _DEFAULT_CLEAN_STREAK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the supervisor's thresholds and (injectable) clock."""
        self._wedge_grace_seconds = wedge_grace_seconds
        self._reset_interval_seconds = reset_interval_seconds
        self._max_soft_resets = max(0, max_soft_resets)
        self._give_up_after_seconds = give_up_after_seconds
        self._clean_streak_seconds = clean_streak_seconds
        self._clock = clock

        self._episode_start: float | None = None
        self._wedge_since: float | None = None
        self._clean_since: float | None = None
        self._last_action_time: float = -math.inf
        self._soft_resets_done = 0

    @property
    def limp_by_level(self) -> int:
        """Soft resets done in the current episode; the manager reduces settings by this many notches."""
        return self._soft_resets_done

    @property
    def is_in_episode(self) -> bool:
        """Whether a wedge episode is currently open (wedged, or not yet recovered for a clean streak)."""
        return self._episode_start is not None

    def evaluate(self, *, is_wedged: bool) -> RecoveryAction:
        """Advance the escalation state machine one tick and return the action to take.

        Args:
            is_wedged: Whether the worker currently has accepted work it structurally cannot progress
                (no live process of the kind the pending work needs).
        """
        now = self._clock()

        if is_wedged:
            self._clean_since = None
            if self._episode_start is None:
                self._episode_start = now
            if self._wedge_since is None:
                self._wedge_since = now
        else:
            self._wedge_since = None
            if self._clean_since is None:
                self._clean_since = now

        # Sustained recovery closes the episode first, so a transient wedge a soft reset actually fixed
        # never reaches give-up (clean_streak is shorter than give_up_after for exactly this reason).
        if (
            self._episode_start is not None
            and not is_wedged
            and self._clean_since is not None
            and (now - self._clean_since) >= self._clean_streak_seconds
        ):
            self._episode_start = None
            self._soft_resets_done = 0
            return RecoveryAction.NONE

        if self._episode_start is None:
            return RecoveryAction.NONE

        # Give-up is measured over the whole episode regardless of the instantaneous wedge state: the
        # episode is only still open because recovery has not held for a clean streak, so a doomed pool
        # that a soft reset keeps briefly reviving still reaches give-up instead of escaping it by flapping.
        if (now - self._episode_start) >= self._give_up_after_seconds:
            self._last_action_time = now
            return RecoveryAction.GIVE_UP

        if (
            is_wedged
            and self._wedge_since is not None
            and (now - self._wedge_since) >= self._wedge_grace_seconds
            and (now - self._last_action_time) >= self._reset_interval_seconds
            and self._soft_resets_done < self._max_soft_resets
        ):
            self._soft_resets_done += 1
            self._last_action_time = now
            return RecoveryAction.SOFT_RESET

        return RecoveryAction.NONE
