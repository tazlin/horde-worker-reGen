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
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, Mock

import pytest

from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
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
