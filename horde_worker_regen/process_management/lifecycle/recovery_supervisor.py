"""Save-our-ship escalation policy: decide when a wedged worker should soft-reset or give up.

The lower layers detect and recover individual failures (a slot crashes -> replaced; a slot crash-loops
-> quarantined; a job faults -> bounded/degraded retry). This supervisor sits above them and answers a
different question: *the worker as a whole has stopped making progress on work it has accepted. Now
what?* Continued operation is the paramount goal, so the escalation is, in order:

1. **Soft reset** (bounded): rebuild the worker's process pools in-place (kill and re-spawn every child,
   un-quarantine slots) without restarting the parent process or detaching the TUI, preserving the
   configured concurrency. A transient wedge (a bad model load, a one-off deadlock) recovers here.
2. **Give up cleanly**: once the worker has been wedged long enough that resets clearly are not helping
   (e.g. a deterministic crash-on-start), stop fighting: fault the jobs that cannot be served so the
   horde reissues them, rather than wedging forever. The worker keeps running and keeps popping.
3. **Abandon ship** (last resort): a second give-up whose wedge outlived even a fresh soft-reset cycle
   escalates to a deliberate abort, rather than faulting jobs every tick forever.

Readiness gating is the crux. A soft reset rebuilds the pools, and the replacement children spend real
time booting (importing torch) before they can accept a job. During that boot window the pool looks
"alive" (the replacement processes exist) but is not yet *ready* (no lane has reached an accepting
state). Escalating to give-up in that window faults jobs the just-rebuilt pool was about to serve. So
the give-up clock does not advance while the pool is still booting: give-up may fire only once the pool
has reached an accepting state (or a bounded boot allowance has elapsed for a pool that never does) and
the wedge has then persisted for a further grace. The manager supplies the readiness fact each tick.

This module is *only the policy*: it tracks how long the worker has been wedged, whether its pool is
ready, and returns an action. The manager owns the wedge assessment (what "wedged" means in terms of
live processes and pending work), the readiness assessment, and the actions (rebuilding pools, faulting
stuck jobs, aborting). Keeping the policy pure makes the escalation timing unit-testable with a fake clock.
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
"""Soft resets attempted within one escalation cycle before escalating to give-up.

One is enough to clear a genuinely transient wedge; a persistent fault (crash-on-start) will not be
fixed by repeating the reset, so the readiness-gated give-up below is what ends the cycle."""

_DEFAULT_POOL_READY_GRACE_SECONDS = 5.0
"""How long the wedge must persist *after the rebuilt pool is ready* before give-up fires.

Measured only while the pool is ready (an inference lane accepting, or the boot allowance elapsed) and
the worker is still wedged. This is the honesty margin that separates "the pool just came back and is
about to drain the queue" (the wedge clears within a tick or two of readiness, so this grace is never
satisfied) from "the pool is ready and the work still cannot move" (a genuine structural wedge)."""

_DEFAULT_BOOT_ALLOWANCE_SECONDS = 30.0
"""Bounded time a rebuilt pool may spend not-yet-ready before it is treated as ready for escalation.

A pool whose replacement children come up healthy reaches an accepting state well within this window, so
the allowance never gates it (the accepting-state signal fires first). The allowance exists only to bound
the pathological case where children never leave PROCESS_STARTING (a child wedged importing torch): give-up
would otherwise be held off forever. Set comfortably above the observed replacement boot time so a merely
slow boot is not cut off (a boot that eventually succeeds clears the wedge and give-up never fires anyway)."""

_DEFAULT_GIVE_UP_COOLDOWN_SECONDS = 10.0
"""Cool-down after a give-up before one further soft-reset cycle is permitted for a still-persisting wedge.

Bounds the continuation: rather than faulting jobs every tick forever, a give-up whose wedge persists over
a ready pool for this long re-opens the escalation cycle once (a fresh soft reset), and if that cycle also
ends in give-up the escalation becomes a deliberate abort."""

_DEFAULT_MAX_GIVE_UP_CYCLES = 2
"""Give-ups permitted in one wedge episode before the escalation is terminal (abort).

The first faults the unservable jobs so the horde reissues them; a continuation permits one more soft-reset
cycle; a second give-up is flagged terminal so the manager abandons ship deliberately instead of by chance."""

_DEFAULT_CLEAN_STREAK_SECONDS = 30.0
"""Continuous non-wedge time required (alongside real progress) to end a wedge episode.

The time streak alone is insufficient once a soft reset has been attempted: a rebuild transiently reads as
not-wedged (the un-quarantine to re-quarantine window, or a queue deadlock that momentarily clears while the
pool boots) and that window can outlast this streak. So an episode that has already spent a soft reset closes
only when the streak holds *and* accepted work has actually moved forward since the most recent soft reset
(see :meth:`RecoverySupervisor.evaluate`). Without that progress requirement the counter would reset on the
transient window and every soft reset would re-log as the first, never reaching the give-up backstop."""


class RecoveryAction(enum.Enum):
    """What the recovery supervisor wants the manager to do this tick."""

    NONE = enum.auto()
    """No action: not wedged, still within a grace/backoff window, or holding after a latched give-up."""
    SOFT_RESET = enum.auto()
    """Rebuild the process pools in-place; the configured concurrency is preserved across the rebuild."""
    GIVE_UP = enum.auto()
    """Fault the jobs that cannot be served so they drain; the worker keeps running.

    Consult :attr:`give_up_is_terminal` when acting on this: a terminal give-up is the deliberate
    abandon-ship escalation, not merely another job-faulting pass."""


class RecoverySupervisor:
    """Tracks how long the worker has been wedged, whether its pool is ready, and escalates.

    Drive it once per control-loop tick with :meth:`evaluate`, passing whether the worker is currently
    wedged (pending work it structurally cannot make progress on), whether its inference pool is ready
    (a lane has reached an accepting state), and whether accepted work has moved forward since the most
    recent soft reset. The returned :class:`RecoveryAction` tells the manager what to do; :attr:`limp_by_level`
    is the number of soft resets done in the current cycle (an escalation counter, not a settings reduction);
    :attr:`give_up_is_terminal` marks the give-up that should abandon ship.

    Progress, not merely a quiet wedge signal, ends an escalation. Once a soft reset has been attempted the
    counter resets only when the clean streak is backed by real forward progress, so a pool that keeps
    rebuilding without ever serving work climbs the ladder to give-up instead of re-attempting the first reset.
    """

    def __init__(
        self,
        *,
        wedge_grace_seconds: float = _DEFAULT_WEDGE_GRACE_SECONDS,
        reset_interval_seconds: float = _DEFAULT_RESET_INTERVAL_SECONDS,
        max_soft_resets: int = _DEFAULT_MAX_SOFT_RESETS,
        pool_ready_grace_seconds: float = _DEFAULT_POOL_READY_GRACE_SECONDS,
        boot_allowance_seconds: float = _DEFAULT_BOOT_ALLOWANCE_SECONDS,
        give_up_cooldown_seconds: float = _DEFAULT_GIVE_UP_COOLDOWN_SECONDS,
        max_give_up_cycles: int = _DEFAULT_MAX_GIVE_UP_CYCLES,
        clean_streak_seconds: float = _DEFAULT_CLEAN_STREAK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the supervisor's thresholds and (injectable) clock."""
        self._wedge_grace_seconds = wedge_grace_seconds
        self._reset_interval_seconds = reset_interval_seconds
        self._max_soft_resets = max(0, max_soft_resets)
        self._pool_ready_grace_seconds = pool_ready_grace_seconds
        self._boot_allowance_seconds = boot_allowance_seconds
        self._give_up_cooldown_seconds = give_up_cooldown_seconds
        self._max_give_up_cycles = max(1, max_give_up_cycles)
        self._clean_streak_seconds = clean_streak_seconds
        self._clock = clock

        self._episode_start: float | None = None
        self._wedge_since: float | None = None
        self._clean_since: float | None = None
        self._last_action_time: float = -math.inf
        self._soft_resets_done = 0
        # When the pool last (re)entered a booting state within this episode: episode open, or a soft reset.
        # The boot allowance is measured from here so a pool that never reaches an accepting state still
        # eventually counts as ready for escalation.
        self._pool_rebuilt_at: float | None = None
        # When the pool first became ready while the worker was still wedged, in the current continuity.
        # Cleared whenever the wedge clears or the pool goes un-ready (a soft reset rebuild).
        self._ready_wedged_since: float | None = None
        # Give-up latch + continuation bookkeeping. The latch makes GIVE_UP fire at most once per cycle;
        # the cycle count decides when a further give-up is terminal.
        self._gave_up_latched = False
        self._gave_up_at: float | None = None
        self._give_up_cycles = 0
        self._give_up_is_terminal = False

    @property
    def limp_by_level(self) -> int:
        """Soft resets done in the current cycle; an escalation counter (the concurrency cap is preserved)."""
        return self._soft_resets_done

    @property
    def is_in_episode(self) -> bool:
        """Whether a wedge episode is currently open (wedged, or not yet recovered for a clean streak)."""
        return self._episode_start is not None

    @property
    def give_up_is_terminal(self) -> bool:
        """Whether the most recent GIVE_UP is the deliberate abandon-ship escalation.

        Meaningful only on the tick :meth:`evaluate` returned GIVE_UP: True when the wedge outlived a
        continuation soft-reset cycle, so the manager should abort rather than merely fault jobs again.
        """
        return self._give_up_is_terminal

    def _close_episode(self) -> None:
        """Reset all episode state after a sustained recovery."""
        self._episode_start = None
        self._wedge_since = None
        self._clean_since = None
        self._last_action_time = -math.inf
        self._soft_resets_done = 0
        self._pool_rebuilt_at = None
        self._ready_wedged_since = None
        self._gave_up_latched = False
        self._gave_up_at = None
        self._give_up_cycles = 0
        self._give_up_is_terminal = False

    def evaluate(self, *, is_wedged: bool, pool_ready: bool, made_progress: bool = False) -> RecoveryAction:
        """Advance the escalation state machine one tick and return the action to take.

        Args:
            is_wedged: Whether the worker currently has accepted work it structurally cannot progress
                (no live process of the kind the pending work needs, or a sustained structural deadlock).
            pool_ready: Whether the inference pool has reached an accepting state (a lane can take a job).
                False while a just-rebuilt pool's replacement children are still booting; the give-up clock
                does not advance until this is True or the bounded boot allowance has elapsed.
            made_progress: Whether accepted work has moved forward (a job completed, an inference started, or
                post-processing advanced) since the most recent soft reset. Gates the episode close once a soft
                reset has been spent: the clean streak alone cannot reset the escalation counter, because a
                rebuild's transient not-wedged window can satisfy the streak without any work moving. Ignored
                before the first soft reset (a self-healing wedge that never needed a reset closes on the streak
                alone). Defaults to False so an un-wired caller escalates rather than silently resetting.
        """
        now = self._clock()
        self._give_up_is_terminal = False

        if is_wedged:
            self._clean_since = None
            if self._episode_start is None:
                self._episode_start = now
                self._pool_rebuilt_at = now
            if self._wedge_since is None:
                self._wedge_since = now
        else:
            self._wedge_since = None
            if self._clean_since is None:
                self._clean_since = now

        # Sustained recovery closes the episode first, so a transient wedge a soft reset actually fixed
        # never reaches give-up (clean_streak is shorter than the readiness-gated give-up for this reason).
        # Once a soft reset has been spent, the streak must be corroborated by real forward progress: a rebuild
        # reads as not-wedged while it boots, so the streak alone would reset the counter on a pool that never
        # actually recovered, stranding it below the give-up backstop. Before any reset there is no such
        # transient window to guard against, so the streak alone suffices (a self-healed wedge needs no proof).
        if (
            self._episode_start is not None
            and not is_wedged
            and self._clean_since is not None
            and (now - self._clean_since) >= self._clean_streak_seconds
            and (self._soft_resets_done == 0 or made_progress)
        ):
            self._close_episode()
            return RecoveryAction.NONE

        if self._episode_start is None:
            return RecoveryAction.NONE

        # The pool counts as ready for escalation once a lane is accepting, or once a bounded boot allowance
        # has elapsed since it last began rebuilding (so a pool whose children never come up still terminates).
        boot_allowance_elapsed = (
            self._pool_rebuilt_at is not None and (now - self._pool_rebuilt_at) >= self._boot_allowance_seconds
        )
        effective_ready = pool_ready or boot_allowance_elapsed

        if is_wedged and effective_ready:
            if self._ready_wedged_since is None:
                self._ready_wedged_since = now
        else:
            self._ready_wedged_since = None

        # After a give-up, hold (no repeated faulting) unless the wedge persists over a ready pool past the
        # cool-down and a continuation cycle remains, in which case re-open the escalation cycle cleanly.
        if self._gave_up_latched:
            can_continue = (
                is_wedged
                and self._ready_wedged_since is not None
                and self._gave_up_at is not None
                and (now - self._gave_up_at) >= self._give_up_cooldown_seconds
                and self._give_up_cycles < self._max_give_up_cycles
            )
            if not can_continue:
                return RecoveryAction.NONE
            self._gave_up_latched = False
            self._soft_resets_done = 0
            self._last_action_time = now
            self._pool_rebuilt_at = now
            self._ready_wedged_since = None
            return RecoveryAction.NONE

        # Give-up is readiness-gated: only after the soft-reset budget is spent and the wedge has persisted
        # over a ready pool for the grace. A second give-up (a wedge that outlived a continuation) is terminal.
        if (
            self._soft_resets_done >= self._max_soft_resets
            and self._ready_wedged_since is not None
            and (now - self._ready_wedged_since) >= self._pool_ready_grace_seconds
        ):
            self._gave_up_latched = True
            self._gave_up_at = now
            self._give_up_is_terminal = self._give_up_cycles >= 1
            self._give_up_cycles += 1
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
            # The rebuild makes the pool un-ready; anchor the boot allowance here and restart the ready grace.
            self._pool_rebuilt_at = now
            self._ready_wedged_since = None
            return RecoveryAction.SOFT_RESET

        return RecoveryAction.NONE
