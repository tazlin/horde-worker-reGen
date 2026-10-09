"""The RAM pop hold must be re-evaluated (and cleared) even when the inference queue is empty.

The soft RAM pop hold blocks image job pops, so once it engages the inference queue drains to empty and
stays there. The governor tick is the only thing that clears the hold, so if the tick were gated behind a
non-empty queue the hold would latch forever: the hold keeps the queue empty, and the empty queue would
keep the only thing that clears the hold from ever running. Its pop-skip counter would then climb without
bound while the worker never pops again, despite RAM having fully recovered.

The contract these tests pin:

* The process manager drives one governance tick per control-loop iteration regardless of queue depth, so
  a latched pop hold on a healthy, idle worker is cleared without any pending job to trigger a scheduling
  cycle.
* The scheduling cycle itself no longer drives the governor, so a busy iteration ticks it exactly once.
* The danger floor and the hold read admissible RAM, the lower of physical available and available commit,
  so a host whose commit limit binds first holds pops before a child is asked to map what it cannot commit.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, Mock

import pytest

from horde_worker_regen.analysis.log_signatures import pattern_for
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import assess_ram_pressure
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_mock_process_info,
    make_testable_process_manager,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_TOTAL_RAM_MB = 64000.0
_HEALTHY_AVAILABLE_RAM_MB = 30000.0


def _pin_available_ram(scheduler: InferenceScheduler, monkeypatch: pytest.MonkeyPatch, available_mb: float) -> None:
    """Pin measured system RAM so the danger-floor verdict is deterministic on any host."""
    monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: available_mb)
    monkeypatch.setattr(scheduler, "_measured_total_ram_mb", lambda: _TOTAL_RAM_MB)


def _budget_enabled_scheduler(process_map: ProcessMap | None = None) -> InferenceScheduler:
    """A scheduler whose measured RAM/VRAM budget is active, so the governance tick actually runs."""
    return _make_inference_scheduler(
        process_map=process_map,
        job_tracker=JobTracker(),
        bridge_data=make_mock_bridge_data(
            enable_vram_budget=True,
            vram_reserve_mb=1024.0,
            ram_reserve_mb=4096.0,
        ),
    )


class TestGovernanceTickClearsLatchedHold:
    """The governance tick clears a stale pop hold with no pending or in-flight work."""

    def test_run_governance_tick_clears_pop_hold_on_healthy_empty_queue(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hold set by a past pressure episode is cleared once RAM is healthy, queue empty or not."""
        process_info = make_mock_process_info(0, model_name=None)
        scheduler = _budget_enabled_scheduler(process_map=ProcessMap({0: process_info}))
        _pin_available_ram(scheduler, monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)
        scheduler._state.ram_pressure_pop_hold = True

        scheduler.run_governance_tick()

        assert scheduler._state.ram_pressure_pop_hold is False, (
            "a healthy host must clear the pop hold even with an empty inference queue"
        )

    def test_run_governance_tick_is_noop_when_budget_disabled(self) -> None:
        """A disabled budget must leave governance untouched.

        The gate is the same one the rest of the memory machinery uses, and a partially-mocked config must
        not act on a non-numeric reserve.
        """
        scheduler = _make_inference_scheduler(job_tracker=JobTracker())  # default bridge data leaves budget off
        scheduler._state.ram_pressure_pop_hold = True

        scheduler.run_governance_tick()

        assert scheduler._state.ram_pressure_pop_hold is True, "a disabled budget must leave governance untouched"

    async def test_run_scheduling_cycle_does_not_drive_the_governor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The scheduling cycle no longer ticks the governor, so a busy iteration ticks it exactly once."""
        scheduler = _budget_enabled_scheduler()
        _pin_available_ram(scheduler, monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)

        governed = Mock()
        monkeypatch.setattr(scheduler, "_govern_ram_pressure_if_pressured", governed)
        # Stub the post-preload dispatch half inert so the cycle runs without probing for a next job.
        monkeypatch.setattr(scheduler, "get_next_job_and_process", AsyncMock(return_value=None))
        monkeypatch.setattr(scheduler, "start_inference", AsyncMock(return_value=False))

        await scheduler.run_scheduling_cycle({})

        governed.assert_not_called()


class TestControlLoopGovernsWithEmptyQueue:
    """The control loop governs every iteration, independent of the queue-gated scheduling cycle."""

    async def test_control_loop_tick_governs_but_skips_scheduling_on_empty_queue(self) -> None:
        """An idle worker with no pending jobs still drives governance, without a scheduling cycle."""
        process_manager = make_testable_process_manager()

        async def _noop_sleep(_delay: float) -> None:
            return None

        process_manager._sleep = _noop_sleep  # type: ignore[method-assign]
        # Pretend a status message was just printed so the tick does not exercise the status reporter (which
        # divides a Mock bridge-data field in this harness); its behavior is covered by its own tests.
        process_manager._last_status_message_time = time.time()
        governance = Mock()
        scheduling = AsyncMock()
        process_manager._inference_scheduler.run_governance_tick = governance  # type: ignore[method-assign]
        process_manager._inference_scheduler.run_scheduling_cycle = scheduling  # type: ignore[method-assign]

        assert len(process_manager._job_tracker.jobs_pending_inference) == 0
        await process_manager._control_loop_tick()

        governance.assert_called_once()
        scheduling.assert_not_called()


_COMMIT_BELOW_FLOOR_MB = 500.0
"""Available commit under any danger floor while physical RAM reads healthy: a host whose commit limit binds
before its physical memory, where a checkpoint mapping fails on commit with physical pages still free."""


class TestCommitBoundGovernance:
    """The danger floor and the soft pop hold read the lower of physical available RAM and available commit."""

    def test_commit_below_the_floor_engages_the_hold_on_a_physically_roomy_host(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Physical RAM far above the floor with commit below it puts the host under pressure and holds pops."""
        scheduler = _budget_enabled_scheduler()
        _pin_available_ram(scheduler, monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)
        scheduler.set_available_commit_mb_provider(lambda: _COMMIT_BELOW_FLOOR_MB)

        scheduler.run_governance_tick()

        verdict = scheduler._governor.last_ram_verdict
        assert verdict is not None
        assert verdict.under_pressure is True, "commit under the floor must read as pressure"
        assert verdict.available_mb == _COMMIT_BELOW_FLOOR_MB
        assert verdict.physical_available_mb == _HEALTHY_AVAILABLE_RAM_MB
        assert verdict.commit_bound is True
        assert scheduler._state.ram_pressure_pop_hold is True, (
            "a host whose commit is below the danger floor must hold pops even with physical RAM to spare"
        )

    def test_unreported_commit_governs_on_physical_ram_alone(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A host that reports no commit figure (POSIX) is governed exactly as before, on physical RAM."""
        scheduler = _budget_enabled_scheduler()
        _pin_available_ram(scheduler, monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)
        scheduler.set_available_commit_mb_provider(lambda: None)

        scheduler.run_governance_tick()

        verdict = scheduler._governor.last_ram_verdict
        assert verdict is not None
        assert verdict.under_pressure is False
        assert verdict.available_mb == _HEALTHY_AVAILABLE_RAM_MB
        assert verdict.commit_bound is False
        assert scheduler._state.ram_pressure_pop_hold is False

    def test_commit_bound_reason_matches_the_registered_pop_hold_line(self) -> None:
        """The pop-hold line carries the commit note in a form the registered log signature still parses."""
        verdict = assess_ram_pressure(
            _HEALTHY_AVAILABLE_RAM_MB,
            _TOTAL_RAM_MB,
            available_commit_mb=_COMMIT_BELOW_FLOOR_MB,
        )
        line = (
            f"Host RAM pop hold engaged: {verdict.reason()}, soft hold 8500 MB, preload 14500 MB, restore 32500 MB; "
            "in-flight jobs continue."
        )

        match = pattern_for("host_ram_pop_hold").search(line)

        assert "(commit-bound; physical 30000 MB)" in verdict.reason()
        assert match is not None, line
        assert float(match.group("available")) == _COMMIT_BELOW_FLOOR_MB
        assert float(match.group("physical")) == _HEALTHY_AVAILABLE_RAM_MB
