"""Last-resort watchdog for a RAM pop hold that stays engaged after the host is healthy.

If the per-iteration governance tick ever fails to clear the soft pop hold once RAM recovers, the hold
blocks image pops, the queue drains, and the worker never pops again despite a healthy, idle host. The
watchdog observes that healthy-but-held condition and, after a grace, resets governance to baseline; if the
hold re-latches despite a healthy host, it escalates to rebuilding the (all-idle) inference pool.

The contract these tests pin:

* The scheduler's ``governance_healthy_but_held`` predicate is True only for a genuine latch (hold set, host
  measurably healthy, nothing draining, no deliberate held-queue grace active) and False for a merely idle
  worker, genuine pressure, an unmeasured verdict, or an active drain.
* The coordinator watchdog resets governance after the grace (Tier 1), escalates to a pool rebuild only if
  the hold survives that (Tier 2), and never fires while pressured, busy, backlogged, or shutting down.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator import WorkerRecoveryCoordinator
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import make_testable_process_manager
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_TOTAL_RAM_MB = 64000.0
_HEALTHY_AVAILABLE_RAM_MB = 30000.0
_CRITICAL_AVAILABLE_RAM_MB = 500.0

_GRACE = WorkerRecoveryCoordinator.HEALTHY_HOLD_WATCHDOG_GRACE_SECONDS
_ESCALATION = WorkerRecoveryCoordinator.HEALTHY_HOLD_ESCALATION_GRACE_SECONDS


def _pin_available_ram(scheduler: InferenceScheduler, monkeypatch: pytest.MonkeyPatch, available_mb: float) -> None:
    """Pin measured system RAM so the danger-floor verdict is deterministic on any host."""
    monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: available_mb)
    monkeypatch.setattr(scheduler, "_measured_total_ram_mb", lambda: _TOTAL_RAM_MB)


class TestGovernanceHealthyButHeldPredicate:
    """The instantaneous latch predicate distinguishes a stuck hold from every benign case."""

    def _scheduler_with_verdict(self, monkeypatch: pytest.MonkeyPatch, available_mb: float) -> InferenceScheduler:
        scheduler = _make_inference_scheduler(job_tracker=JobTracker())
        _pin_available_ram(scheduler, monkeypatch, available_mb)
        scheduler._governor.last_ram_verdict = scheduler._ram_pressure_verdict()
        return scheduler

    def test_true_when_hold_set_and_host_healthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hold engaged, verdict healthy, nothing draining, no grace: a genuine latch."""
        scheduler = self._scheduler_with_verdict(monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)
        scheduler._state.ram_pressure_pop_hold = True

        assert scheduler.governance_healthy_but_held() is True

    def test_false_when_hold_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A merely idle worker never set the hold, so it is not a latch (the key discriminator)."""
        scheduler = self._scheduler_with_verdict(monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)
        scheduler._state.ram_pressure_pop_hold = False

        assert scheduler.governance_healthy_but_held() is False

    def test_false_before_first_verdict_measured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hold with no verdict yet measured is treated as not-yet-healthy (do not fire)."""
        scheduler = _make_inference_scheduler(job_tracker=JobTracker())
        _pin_available_ram(scheduler, monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)
        scheduler._state.ram_pressure_pop_hold = True
        assert scheduler._governor.last_ram_verdict is None

        assert scheduler.governance_healthy_but_held() is False

    def test_false_when_host_under_pressure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hold under genuine pressure is correct, not a latch."""
        scheduler = self._scheduler_with_verdict(monkeypatch, _CRITICAL_AVAILABLE_RAM_MB)
        scheduler._state.ram_pressure_pop_hold = True
        assert scheduler._governor.last_ram_verdict is not None
        assert scheduler._governor.last_ram_verdict.under_pressure is True

        assert scheduler.governance_healthy_but_held() is False

    def test_false_while_a_process_is_draining(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A drain legitimately holds pops on a healthy host; its own path resolves it, not the watchdog."""
        scheduler = self._scheduler_with_verdict(monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)
        scheduler._state.ram_pressure_pop_hold = True
        scheduler._ram_governor_state.draining_process_ids = {1}

        assert scheduler.governance_healthy_but_held() is False


class _FakeClock:
    """A settable monotonic clock for driving grace windows deterministically."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestHealthyHoldWatchdog:
    """The coordinator watchdog resets governance after the grace and escalates only if it does not stick."""

    def _coordinator_with_latch(
        self, *, held: bool = True
    ) -> tuple[WorkerRecoveryCoordinator, _FakeClock, Mock, Mock]:
        pm = make_testable_process_manager()
        coordinator = pm._recovery_coordinator
        clock = _FakeClock()
        coordinator._clock = clock  # type: ignore[assignment]
        pm._inference_scheduler.governance_healthy_but_held = lambda: held  # type: ignore[method-assign]
        reset = Mock()
        rebuild = Mock()
        pm._inference_scheduler.reset_governance_to_baseline = reset  # type: ignore[method-assign]
        pm._process_lifecycle.rebuild_inference_pool = rebuild  # type: ignore[method-assign]
        assert pm._process_map.has_inference_in_progress() is False
        assert len(pm._job_tracker.jobs_pending_inference) == 0
        return coordinator, clock, reset, rebuild

    def test_tier1_resets_governance_after_grace_without_rebuild(self) -> None:
        """A sustained healthy-but-held condition resets governance, but does not churn the healthy pool."""
        coordinator, clock, reset, rebuild = self._coordinator_with_latch()

        coordinator.maybe_reset_stuck_governance_hold()  # first observation records the timestamp
        reset.assert_not_called()

        clock.now += _GRACE - 1.0
        coordinator.maybe_reset_stuck_governance_hold()
        reset.assert_not_called()

        clock.now += 2.0  # now past the grace
        coordinator.maybe_reset_stuck_governance_hold()
        reset.assert_called_once_with("healthy-hold watchdog")
        rebuild.assert_not_called()

    def test_tier2_escalates_to_pool_rebuild_if_hold_relatches(self) -> None:
        """If the hold survives the governance reset, escalate to rebuilding the idle inference pool."""
        coordinator, clock, reset, rebuild = self._coordinator_with_latch()

        coordinator.maybe_reset_stuck_governance_hold()
        clock.now += _GRACE + 1.0
        coordinator.maybe_reset_stuck_governance_hold()  # Tier 1 fires
        reset.assert_called_once()
        rebuild.assert_not_called()

        clock.now += _ESCALATION - 1.0
        coordinator.maybe_reset_stuck_governance_hold()
        rebuild.assert_not_called()

        clock.now += 2.0  # past the escalation grace, hold still latched
        coordinator.maybe_reset_stuck_governance_hold()
        rebuild.assert_called_once_with(reason="healthy-hold watchdog escalation")
        assert coordinator.healthy_hold_since is None
        assert coordinator.governance_reset_at is None

    def test_resolved_hold_clears_episode_without_firing(self) -> None:
        """If the hold clears on its own before the grace, the episode resets and nothing fires."""
        coordinator, clock, reset, rebuild = self._coordinator_with_latch()

        coordinator.maybe_reset_stuck_governance_hold()
        assert coordinator.healthy_hold_since is not None

        coordinator._inference_scheduler.governance_healthy_but_held = lambda: False  # type: ignore[method-assign]
        clock.now += _GRACE + 100.0
        coordinator.maybe_reset_stuck_governance_hold()

        reset.assert_not_called()
        rebuild.assert_not_called()
        assert coordinator.healthy_hold_since is None

    def test_does_not_fire_while_inference_in_progress(self) -> None:
        """A busy worker is not the wedge this watchdog recovers."""
        coordinator, clock, reset, rebuild = self._coordinator_with_latch()
        coordinator._process_map.has_inference_in_progress = lambda: True  # type: ignore[method-assign]

        coordinator.maybe_reset_stuck_governance_hold()
        clock.now += _GRACE + 100.0
        coordinator.maybe_reset_stuck_governance_hold()

        reset.assert_not_called()
        rebuild.assert_not_called()

    def test_does_not_fire_while_shutting_down(self) -> None:
        """Shutdown suppresses the watchdog; the timestamps stay clear."""
        coordinator, clock, reset, rebuild = self._coordinator_with_latch()
        coordinator._state.shutting_down = True

        coordinator.maybe_reset_stuck_governance_hold()
        clock.now += _GRACE + 100.0
        coordinator.maybe_reset_stuck_governance_hold()

        reset.assert_not_called()
        assert coordinator.healthy_hold_since is None

    def test_does_not_fire_during_downloads_only_hold(self) -> None:
        """A download-only posture legitimately holds pops; the watchdog must not fight it."""
        coordinator, clock, reset, rebuild = self._coordinator_with_latch()
        coordinator._state.downloads_only_hold = True

        coordinator.maybe_reset_stuck_governance_hold()
        clock.now += _GRACE + 100.0
        coordinator.maybe_reset_stuck_governance_hold()

        reset.assert_not_called()
        assert coordinator.healthy_hold_since is None
