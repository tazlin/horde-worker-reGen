"""The VRAM arbiter as the preload-admission authority: the adapter seam, and actuation execution.

These pin the contracts the authority path adds: a preload proceeds only on a FITS verdict (never on a
DEFER/DENY, and there is no overcommit-admit override), a FITS runs the marginal RAM verdict, and a deferred
verdict's described commands are actually run against the worker's reclaim mechanisms (the allocator-cache
release being the new one). A FITS is a real fit against the truthful device-free reading net of outstanding
reservations and the noise buffer; a full card (device-free at zero) never admits.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.vram_arbiter import (
    ActuatorCommand,
    ActuatorCommandKind,
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import _WholeCardDemandOutcome
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _full_card_state() -> DeviceVramState:
    """A device state whose card is physically full (device-free at zero), so no candidate fits."""
    return DeviceVramState(
        total_vram_mb=16000.0,
        baseline_mb=0.0,
        committed_vram_mb=16000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=0.0,
    )


def _fitting_state() -> DeviceVramState:
    """A fresh device state with ample device-free room, so an ordinary candidate fits."""
    return DeviceVramState(
        total_vram_mb=24000.0,
        baseline_mb=1000.0,
        committed_vram_mb=2000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=21000.0,
    )


async def _budgeted_scheduler_with_head():  # noqa: ANN202
    """A scheduler with an empty target slot and one pending head, the budget gate active."""
    target = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    process_map = ProcessMap({0: target})
    job_tracker = JobTracker()
    job = make_job_pop_response("model_a")
    await track_popped_job_async(job_tracker, job)
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(enable_vram_budget=True, vram_reserve_mb=2000, ram_reserve_mb=4096),
        max_concurrent=2,
        max_inference=2,
    )
    scheduler._decide_whole_card_demand = Mock(return_value=_WholeCardDemandOutcome.FALL_THROUGH)
    scheduler._measured_available_ram_mb = lambda: 64000.0  # type: ignore[method-assign]
    return scheduler, job, target


def _install_cycle(scheduler, state: DeviceVramState) -> None:  # noqa: ANN001
    """Freeze a crafted arbiter cycle on the scheduler's arbiter."""
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
    scheduler._vram_arbiter = arbiter


class TestAdapterSeamAdmitsOnlyOnFits:
    """No preload proceeds without a FITS verdict at the adapter seam."""

    async def test_deferred_verdict_does_not_admit(self) -> None:
        """A preload onto a full card defers (device-free is zero): the adapter returns False and sends nothing."""
        scheduler, job, target = await _budgeted_scheduler_with_head()
        _install_cycle(scheduler, _full_card_state())

        admitted = scheduler._admit_preload_under_budget(job, target, is_head_blocker=True)

        assert admitted is False
        assert target.last_control_flag != HordeControlFlag.PRELOAD_MODEL

    async def test_fitting_verdict_admits(self) -> None:
        """A candidate within the device-free room is admitted (the RAM verdict then passes)."""
        scheduler, job, target = await _budgeted_scheduler_with_head()
        _install_cycle(scheduler, _fitting_state())

        admitted = scheduler._admit_preload_under_budget(job, target, is_head_blocker=True)

        assert admitted is True

    async def test_starved_head_on_full_card_still_defers(self) -> None:
        """A head deferred on the clock still defers on a physically full card, with no over-budget tag."""
        scheduler, job, target = await _budgeted_scheduler_with_head()
        _install_cycle(scheduler, _full_card_state())
        scheduler._head_starved_seconds = Mock(return_value=120.0)

        admitted = scheduler._admit_preload_under_budget(job, target, is_head_blocker=True)

        assert admitted is False
        assert scheduler._job_tracker.is_admitted_over_budget(job) is False


class TestFitRunsRamVerdict:
    """A FITS runs the marginal RAM verdict and tags nothing over-budget (there is no admit-into-overcommit path)."""

    async def test_fit_runs_the_ram_verdict(self) -> None:
        """An ordinary within-room FITS runs the marginal RAM verdict and tags nothing over-budget."""
        scheduler, job, target = await _budgeted_scheduler_with_head()
        _install_cycle(scheduler, _fitting_state())
        scheduler._apply_ram_verdict = Mock(return_value=True)  # type: ignore[method-assign]

        admitted = scheduler._admit_preload_under_budget(job, target, is_head_blocker=True)

        assert admitted is True
        assert scheduler._job_tracker.is_admitted_over_budget(job) is False
        scheduler._apply_ram_verdict.assert_called_once()


class TestActuationExecution:
    """A deferred verdict's RELEASE_CACHE command reaches the allocator-cache sender for the idle lane only."""

    def test_release_cache_command_sends_to_idle_target_only(self) -> None:
        """RELEASE_CACHE(idle) sends one release_allocator_cache to the idle pid and never to a busy one."""
        target = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: target}),
            job_tracker=JobTracker(),
            bridge_data=make_mock_bridge_data(enable_vram_budget=True),
            max_concurrent=2,
            max_inference=2,
        )
        scheduler.release_allocator_cache = Mock(return_value=True)  # type: ignore[method-assign]

        idle_pid, busy_pid = 5, 6
        commands = (
            ActuatorCommand(kind=ActuatorCommandKind.RELEASE_CACHE, device_index=None, target_process_id=idle_pid),
        )
        scheduler._execute_preload_actuations(commands, device_index=None, for_head_of_queue=True)

        scheduler.release_allocator_cache.assert_called_once_with(idle_pid)
        assert all(call.args != (busy_pid,) for call in scheduler.release_allocator_cache.call_args_list)

    async def test_deferred_preload_runs_the_release_then_readmits_next_cycle(self) -> None:
        """A full-card preload defers and releases the idle lane cache; the relieved re-ask admits."""
        scheduler, job, target = await _budgeted_scheduler_with_head()
        scheduler.release_allocator_cache = Mock(return_value=True)  # type: ignore[method-assign]

        pressured = DeviceVramState(
            total_vram_mb=16000.0,
            baseline_mb=0.0,
            committed_vram_mb=16000.0,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            device_free_mb=0.0,
            idle_process_ids=frozenset({5}),
        )
        _install_cycle(scheduler, pressured)

        assert scheduler._admit_preload_under_budget(job, target, is_head_blocker=True) is False
        scheduler.release_allocator_cache.assert_called_once_with(5)

        # Next cycle: the released lane cache has returned VRAM to the card (device-free now shows room), so the
        # re-ask admits.
        _install_cycle(scheduler, _fitting_state())
        assert scheduler._admit_preload_under_budget(job, target, is_head_blocker=True) is True
