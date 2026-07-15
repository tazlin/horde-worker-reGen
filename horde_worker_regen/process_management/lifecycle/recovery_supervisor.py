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
has reached an accepting state (or a boot backstop has elapsed) and the wedge has then persisted for a
further grace. Two backstops bound the boot so a pool that never comes up still escalates: a boot
*allowance* for an absent boot (no live child progressing), and a larger boot *hard cap* for a hung boot
(a child alive and still booting but never accepting). While a replacement child is alive and booting the
allowance is suppressed up to the hard cap, so a merely slow-but-healthy boot is not cut off. The manager
supplies the readiness fact and whether a live boot is in progress each tick.

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

from loguru import logger

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
"""Backstop for an *absent* boot: how long a pool with no live booting child may sit not-yet-ready.

A pool whose replacement children come up healthy reaches an accepting state well within this window, so
the allowance never gates it (the accepting-state signal fires first). The allowance exists to bound the
case where no replacement child is progressing at all (none ever entered PROCESS_STARTING, or the one that
did has died) so give-up stays reachable. It is deliberately *not* the bound for a child that is still
alive and booting: a live-but-slow boot can outlast this allowance yet still succeed, and faulting there
would drop the very jobs the finishing boot serves. A live boot suppresses this allowance up to the
separate boot hard cap below."""

_DEFAULT_BOOT_HARD_CAP_MULTIPLE = 4.0
"""Boot hard cap default, as a multiple of the boot allowance, when no explicit cap is given.

The hard cap is the backstop for a *hung* boot: a replacement child that stays alive and PROCESS_STARTING
but never reaches an accepting state. While a live boot is in progress the ordinary boot allowance is held
off (a slow-but-healthy boot must not be cut short), so a second, larger bound is needed or a permanently
hung boot would defer give-up forever. Defaulted generously above observed healthy replacement boot times
(which have reached well past the allowance) so only a genuinely stuck boot trips it."""

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
    (a lane has reached an accepting state), whether accepted work has moved forward since the most recent
    soft reset, and whether a replacement child is still alive and booting. The returned
    :class:`RecoveryAction` tells the manager what to do; :attr:`limp_by_level` is the number of soft resets
    done in the current cycle (an escalation counter, not a settings reduction); :attr:`give_up_is_terminal`
    marks the give-up that should abandon ship.

    Give-up readiness has two backstops, not one. The **boot allowance** bounds an *absent* boot (no live
    child is progressing) so a pool that never comes up still escalates. The **boot hard cap** bounds a
    *hung* boot: while a replacement child is alive and still booting the allowance is suppressed (a slow but
    healthy boot must not be faulted), and the hard cap is the larger bound past which even a live-but-stuck
    boot lets the allowance apply again. Both are backstops; a boot that actually completes clears the wedge
    and give-up never fires.

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
        boot_hard_cap_seconds: float | None = None,
        give_up_cooldown_seconds: float = _DEFAULT_GIVE_UP_COOLDOWN_SECONDS,
        max_give_up_cycles: int = _DEFAULT_MAX_GIVE_UP_CYCLES,
        clean_streak_seconds: float = _DEFAULT_CLEAN_STREAK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the supervisor's thresholds and (injectable) clock.

        Args:
            wedge_grace_seconds: How long a wedge must persist before the first soft reset.
            reset_interval_seconds: Minimum spacing between soft resets.
            max_soft_resets: Soft resets attempted in one cycle before give-up becomes eligible.
            pool_ready_grace_seconds: How long the wedge must persist after the pool is ready before give-up.
            boot_allowance_seconds: Backstop for an *absent* boot (no live booting child) before the pool is
                treated as ready for escalation. Suppressed while a live boot is in flight (see
                ``boot_hard_cap_seconds``).
            boot_hard_cap_seconds: Backstop for a *hung* boot: the larger bound past which a still-live but
                never-completing boot stops suppressing the allowance, so give-up stays reachable. Defaults to
                ``boot_allowance_seconds`` times :data:`_DEFAULT_BOOT_HARD_CAP_MULTIPLE` when not given.
            give_up_cooldown_seconds: Cool-down before a continuation soft-reset cycle is permitted.
            max_give_up_cycles: Give-ups permitted in one episode before the escalation is terminal.
            clean_streak_seconds: Continuous non-wedge time (with progress) required to end an episode.
            clock: Injectable monotonic clock.
        """
        self._wedge_grace_seconds = wedge_grace_seconds
        self._reset_interval_seconds = reset_interval_seconds
        self._max_soft_resets = max(0, max_soft_resets)
        self._pool_ready_grace_seconds = pool_ready_grace_seconds
        self._boot_allowance_seconds = boot_allowance_seconds
        self._boot_hard_cap_seconds = (
            boot_hard_cap_seconds
            if boot_hard_cap_seconds is not None
            else _DEFAULT_BOOT_HARD_CAP_MULTIPLE * boot_allowance_seconds
        )
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
        # Edge latch so the live-boot give-up deferral is logged once per hold engagement, not every tick.
        self._boot_hold_deferral_logged = False

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
        self._boot_hold_deferral_logged = False

    def yield_give_up(self) -> None:
        """Un-count the give-up just returned because the caller deferred it to an in-flight remedy.

        A caller that receives :attr:`RecoveryAction.GIVE_UP` may judge the wedge still curable (a reclaim it
        just issued has not had time to land) and yield instead of faulting. Without this refund the yielded
        give-up would still consume a continuation cycle, and enough yields would latch the escalation with
        the safety valve never having actually fired: the bounded yield would decay into an unbounded park.
        The refund keeps the cool-down (the caller may not re-ask sooner than a taken give-up could) but
        restores the cycle and the terminal flag, so the eventual real give-up escalates exactly as if the
        yields had never happened.
        """
        if not self._gave_up_latched:
            return
        self._give_up_cycles = max(0, self._give_up_cycles - 1)
        self._give_up_is_terminal = False

    def evaluate(
        self,
        *,
        is_wedged: bool,
        pool_ready: bool,
        made_progress: bool = False,
        head_recovery_in_flight: bool = False,
        boot_in_progress: bool = False,
    ) -> RecoveryAction:
        """Advance the escalation state machine one tick and return the action to take.

        Args:
            is_wedged: Whether the worker currently has accepted work it structurally cannot progress
                (no live process of the kind the pending work needs, or a sustained structural deadlock).
            pool_ready: Whether the inference pool has reached an accepting state (a lane can take a job).
                False while a just-rebuilt pool's replacement children are still booting; the give-up clock
                does not advance until this is True or a boot backstop (allowance/hard cap) has elapsed.
            made_progress: Whether accepted work has moved forward (a job completed, an inference started, or
                post-processing advanced) since the most recent soft reset. Gates the episode close once a soft
                reset has been spent: the clean streak alone cannot reset the escalation counter, because a
                rebuild's transient not-wedged window can satisfy the streak without any work moving. Ignored
                before the first soft reset (a self-healing wedge that never needed a reset closes on the streak
                alone). Defaults to False so an un-wired caller escalates rather than silently resetting.
            head_recovery_in_flight: Whether the pool is actively materialising the head-of-queue job's model
                (a preload/load underway over an otherwise idle lane). A ready lane whose head model is still
                loading is capacity in flight, not a wedge over a healthy pool, so the ready-wedged give-up
                anchor is held while this is True: give-up must not fault a job the pool is loading. The caller
                bounds this by the preload budget, so a load that never lands still anchors and escalates.
            boot_in_progress: Whether a replacement child is alive and still booting (state PROCESS_STARTING
                with a live OS process). A live boot is capacity in flight, so it suppresses the boot
                *allowance* backstop: the allowance would otherwise treat a merely slow-but-healthy boot as
                ready and fault the jobs the finishing boot serves. The suppression is bounded by the boot
                hard cap, past which even a still-live boot is deemed hung and the allowance applies again, so
                a permanently stuck boot still escalates. A dead boot must report False here (the caller uses a
                liveness-aware count) so it does not hold give-up. Defaults to False for an unwired caller.
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

        # The pool counts as ready for escalation once a lane is accepting, or once a boot backstop has
        # elapsed since it last began rebuilding (so a pool whose boot never lands still terminates). There are
        # two backstops: the allowance bounds an *absent* boot (no live child progressing), and the larger hard
        # cap bounds a *hung* boot. While a replacement child is alive and still booting the allowance is
        # suppressed up to the hard cap: a slow-but-healthy boot can outlast the allowance yet still succeed,
        # and faulting there drops the jobs the finishing boot serves. Past the hard cap even a still-live boot
        # is deemed hung, so the allowance applies again and a permanently stuck boot escalates.
        boot_elapsed = 0.0 if self._pool_rebuilt_at is None else now - self._pool_rebuilt_at
        boot_hard_cap_reached = self._pool_rebuilt_at is not None and boot_elapsed >= self._boot_hard_cap_seconds
        live_boot_holds = boot_in_progress and not boot_hard_cap_reached
        boot_allowance_reached = self._pool_rebuilt_at is not None and boot_elapsed >= self._boot_allowance_seconds
        boot_allowance_elapsed = boot_allowance_reached and not live_boot_holds
        effective_ready = pool_ready or boot_allowance_elapsed

        # Edge-triggered: log once when a live boot first suppresses an allowance that would otherwise have
        # marked the pool ready for give-up, so the deferral is visible without a per-tick repeat.
        if is_wedged and live_boot_holds and boot_allowance_reached:
            if not self._boot_hold_deferral_logged:
                logger.info(
                    "Save-our-ship: deferring give-up while a replacement child is still alive and booting "
                    f"(boot elapsed {boot_elapsed:.0f}s, allowance {self._boot_allowance_seconds:.0f}s, hard "
                    f"cap {self._boot_hard_cap_seconds:.0f}s). The finishing boot is expected to serve the "
                    "queued work rather than have it faulted.",
                )
                self._boot_hold_deferral_logged = True
        else:
            self._boot_hold_deferral_logged = False

        # A ready lane whose head-of-queue model is still materialising is capacity in flight, not a wedge
        # over a healthy pool: hold the ready-wedged anchor so give-up cannot fault a job the pool is loading.
        # The caller bounds ``head_recovery_in_flight`` by the preload budget, so a load that never lands
        # stops deferring and the anchor sets, letting the wedge escalate exactly as before.
        if is_wedged and effective_ready and not head_recovery_in_flight:
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
