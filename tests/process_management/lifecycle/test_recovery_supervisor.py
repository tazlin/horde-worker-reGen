"""Unit tests for the save-our-ship escalation policy (soft-reset -> give-up timing)."""

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
        "give_up_after_seconds": 6,
        "clean_streak_seconds": 10,
    }
    params.update(overrides)
    return RecoverySupervisor(clock=clock, **params), clock  # type: ignore[arg-type]


def test_not_wedged_is_always_none() -> None:
    """A worker that is not wedged never gets a recovery action and opens no episode."""
    supervisor, clock = _make()
    for _ in range(5):
        assert supervisor.evaluate(is_wedged=False) is RecoveryAction.NONE
        clock.advance(5)
    assert not supervisor.is_in_episode


def test_grace_precedes_first_soft_reset() -> None:
    """A wedge shorter than the grace window does not trigger a reset (rides out brief blips)."""
    supervisor, clock = _make()
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.NONE
    clock.advance(1)
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.NONE


def test_escalates_soft_reset_then_give_up() -> None:
    """A persistent wedge attempts a bounded soft reset, then escalates to giving up."""
    supervisor, clock = _make()
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.NONE

    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.SOFT_RESET
    assert supervisor.limp_by_level == 1

    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.NONE  # resets exhausted, give-up not yet due

    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.GIVE_UP


def test_give_up_survives_flapping() -> None:
    """A brief healthy blip (a soft reset's slot looking alive) does not restart the give-up clock."""
    supervisor, clock = _make()
    supervisor.evaluate(is_wedged=True)
    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.SOFT_RESET

    clock.advance(1)
    assert supervisor.evaluate(is_wedged=False) is RecoveryAction.NONE  # blip
    clock.advance(1)
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.NONE  # wedged again, episode intact

    clock.advance(2)  # episode age is now 6s despite the blip
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.GIVE_UP


def test_sustained_recovery_closes_episode() -> None:
    """A sustained clean streak ends the episode and resets the limp-by level before give-up is due."""
    # clean_streak (3) is shorter than give_up_after (10), so a recovery is recognized before give-up.
    supervisor, clock = _make(clean_streak_seconds=3, give_up_after_seconds=10)
    supervisor.evaluate(is_wedged=True)
    clock.advance(2)
    supervisor.evaluate(is_wedged=True)  # SOFT_RESET, limp_by_level == 1
    assert supervisor.limp_by_level == 1

    clock.advance(1)
    supervisor.evaluate(is_wedged=False)  # clean streak starts
    clock.advance(3)
    supervisor.evaluate(is_wedged=False)  # clean streak satisfied -> episode closes

    assert not supervisor.is_in_episode
    assert supervisor.limp_by_level == 0


def test_multiple_soft_resets_before_give_up() -> None:
    """With a larger reset budget, several resets are attempted (each a deeper limp-by) before give-up."""
    supervisor, clock = _make(max_soft_resets=2, give_up_after_seconds=10)
    supervisor.evaluate(is_wedged=True)

    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.SOFT_RESET
    assert supervisor.limp_by_level == 1

    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.SOFT_RESET
    assert supervisor.limp_by_level == 2

    clock.advance(2)
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.NONE  # budget spent, give-up not yet due

    clock.advance(4)
    assert supervisor.evaluate(is_wedged=True) is RecoveryAction.GIVE_UP
