"""Unit tests for the save-our-ship escalation policy (soft-reset -> readiness-gated give-up timing)."""

from __future__ import annotations

from horde_worker_regen.process_management.lifecycle.recovery_supervisor import RecoveryAction, RecoverySupervisor


class _FakeClock:
    """A monotonic clock the test advances explicitly, so escalation timing is deterministic."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make(**overrides: float | int) -> tuple[RecoverySupervisor, _FakeClock]:
    clock = _FakeClock()
    params: dict[str, float | int] = {
        "wedge_grace_seconds": 2,
        "reset_interval_seconds": 2,
        "max_soft_resets": 1,
        "pool_ready_grace_seconds": 2,
        # A boot allowance far larger than any test timeline, so tests drive readiness explicitly through
        # ``pool_ready`` unless they are specifically exercising the allowance fallback.
        "boot_allowance_seconds": 1000,
        "give_up_cooldown_seconds": 1000,
        "max_give_up_cycles": 2,
        "clean_streak_seconds": 10,
    }
    params.update(overrides)
    return RecoverySupervisor(clock=clock, **params), clock  # type: ignore[arg-type]


def test_not_wedged_is_always_none() -> None:
    """A worker that is not wedged never gets a recovery action and opens no episode."""
    supervisor, clock = _make()
    for _ in range(5):
        assert supervisor.evaluate(is_wedged=False, pool_ready=True) is RecoveryAction.NONE
        clock.advance(5)
    assert not supervisor.is_in_episode


def test_grace_precedes_first_soft_reset() -> None:
    """A wedge shorter than the grace window does not trigger a reset (rides out brief blips)."""
    supervisor, clock = _make()
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.NONE
    clock.advance(1)
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.NONE


def test_escalates_soft_reset_then_give_up() -> None:
    """A persistent wedge over a ready pool attempts a soft reset, then escalates to giving up."""
    supervisor, clock = _make()
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.NONE

    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.SOFT_RESET
    assert supervisor.limp_by_level == 1

    clock.advance(2)
    # Pool ready and wedged, but the ready-grace has not yet elapsed since readiness resumed.
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.NONE

    clock.advance(2)
    action = supervisor.evaluate(is_wedged=True, pool_ready=True)
    assert action is RecoveryAction.GIVE_UP
    assert supervisor.give_up_is_terminal is False


def test_give_up_held_off_while_pool_not_ready() -> None:
    """The incident: after a soft reset, give-up must not fire while the rebuilt pool is still booting.

    A structural wedge triggers a soft reset; the replacement children then spend the boot window not-ready
    (no lane accepting). Through that window the give-up clock must not advance. When the children finish
    booting and the wedge clears, the episode recovers with no give-up and no faulting.
    """
    # Boot allowance longer than the boot window so only the accepting-state signal (not the allowance)
    # can mark the pool ready; clean streak short so the recovery is recognized quickly.
    supervisor, clock = _make(boot_allowance_seconds=1000, clean_streak_seconds=5)

    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.NONE  # pool idle-accepting
    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.SOFT_RESET

    # Boot window: children are PROCESS_STARTING (pool not ready) while the wedge persists. No give-up.
    for _ in range(6):
        clock.advance(2)
        assert supervisor.evaluate(is_wedged=True, pool_ready=False) is RecoveryAction.NONE

    # Children reach an accepting state and the wedge clears simultaneously; recovery, not give-up.
    clock.advance(1)
    assert supervisor.evaluate(is_wedged=False, pool_ready=True) is RecoveryAction.NONE
    clock.advance(5)
    assert supervisor.evaluate(is_wedged=False, pool_ready=True) is RecoveryAction.NONE
    assert not supervisor.is_in_episode


def test_give_up_survives_flapping() -> None:
    """A brief healthy blip (a soft reset's slot looking alive) does not restart the give-up clock."""
    supervisor, clock = _make()
    supervisor.evaluate(is_wedged=True, pool_ready=True)
    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.SOFT_RESET

    clock.advance(1)
    assert supervisor.evaluate(is_wedged=False, pool_ready=True) is RecoveryAction.NONE  # blip
    clock.advance(1)
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.NONE  # wedged again

    # The blip reset the ready-grace, so give-up follows the grace after readiness+wedge resumed, not the blip.
    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.GIVE_UP


def test_sustained_recovery_closes_episode() -> None:
    """A sustained clean streak ends the episode and resets the limp-by level before give-up is due."""
    supervisor, clock = _make(clean_streak_seconds=3)
    supervisor.evaluate(is_wedged=True, pool_ready=True)
    clock.advance(2)
    supervisor.evaluate(is_wedged=True, pool_ready=True)  # SOFT_RESET, limp_by_level == 1
    assert supervisor.limp_by_level == 1

    clock.advance(1)
    supervisor.evaluate(is_wedged=False, pool_ready=True)  # clean streak starts
    clock.advance(3)
    supervisor.evaluate(is_wedged=False, pool_ready=True)  # clean streak satisfied -> episode closes

    assert not supervisor.is_in_episode
    assert supervisor.limp_by_level == 0


def test_multiple_soft_resets_before_give_up() -> None:
    """With a larger reset budget, several resets are attempted (each a deeper limp-by) before give-up."""
    supervisor, clock = _make(max_soft_resets=2)
    supervisor.evaluate(is_wedged=True, pool_ready=True)

    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.SOFT_RESET
    assert supervisor.limp_by_level == 1

    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.SOFT_RESET
    assert supervisor.limp_by_level == 2

    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.NONE  # ready-grace not met

    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.GIVE_UP


def test_give_up_latches_once_per_cycle() -> None:
    """After a give-up, further wedged ticks within the cool-down produce no repeated give-up (spam guard)."""
    supervisor, clock = _make()
    supervisor.evaluate(is_wedged=True, pool_ready=True)
    clock.advance(2)
    supervisor.evaluate(is_wedged=True, pool_ready=True)  # SOFT_RESET
    clock.advance(2)
    supervisor.evaluate(is_wedged=True, pool_ready=True)  # ready-grace begins after the rebuild
    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.GIVE_UP

    give_ups = 0
    for _ in range(20):
        clock.advance(2)
        if supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.GIVE_UP:
            give_ups += 1
    assert give_ups == 0  # the cool-down here is longer than the whole loop, so no continuation fires


def test_pool_never_ready_bounded_escalation() -> None:
    """Hostile self-infliction: a pool that never becomes ready reaches a terminal give-up in bounded time.

    Children never leave the boot state (``pool_ready`` stays False). The give-up clock must neither fire at
    a fixed age while the pool might still be booting nor spin forever: the bounded boot allowance eventually
    marks the pool ready, and a continuation cycle escalates a persisting wedge to a terminal give-up.
    """
    supervisor, clock = _make(boot_allowance_seconds=6, give_up_cooldown_seconds=4)

    actions: list[RecoveryAction] = []
    terminal_seen_at: float | None = None
    for _ in range(40):
        clock.advance(2)
        action = supervisor.evaluate(is_wedged=True, pool_ready=False)
        actions.append(action)
        if action is RecoveryAction.GIVE_UP and supervisor.give_up_is_terminal:
            terminal_seen_at = clock.now
            break

    assert RecoveryAction.SOFT_RESET in actions
    assert actions.count(RecoveryAction.GIVE_UP) == 2  # one job-faulting give-up, then the terminal one
    assert terminal_seen_at is not None  # a defined terminal outcome, not an infinite silent spin


def test_continuation_then_terminal_over_ready_pool() -> None:
    """A give-up whose wedge persists over a ready pool permits one more soft-reset cycle, then aborts.

    Exactly one continuation: soft reset -> give-up -> (cool-down) -> soft reset -> terminal give-up.
    """
    supervisor, clock = _make(give_up_cooldown_seconds=3)

    soft_resets = 0
    give_ups: list[bool] = []  # whether each give-up was terminal
    for _ in range(30):
        clock.advance(1)
        action = supervisor.evaluate(is_wedged=True, pool_ready=True)
        if action is RecoveryAction.SOFT_RESET:
            soft_resets += 1
        elif action is RecoveryAction.GIVE_UP:
            give_ups.append(supervisor.give_up_is_terminal)
            if supervisor.give_up_is_terminal:
                break

    assert soft_resets == 2
    assert give_ups == [False, True]
