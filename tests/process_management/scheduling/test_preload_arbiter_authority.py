"""The VRAM arbiter as the preload-admission authority: the adapter seam, and actuation execution.

These pin the contracts the authority path adds over the observational overlay: a preload proceeds only on
a FITS verdict (never on a DEFER/DENY, and there is no overcommit-admit override), the foreign-pressure
fit-into-reality FITS routes through the over-budget marking (heavy-head grace, RAM reclaim), and a deferred
verdict's described commands are actually run against the worker's reclaim mechanisms (the allocator-cache
release being the new one).
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.admission_identity import evaluate_admission
from horde_worker_regen.process_management.resources.vram_arbiter import (
    ActuatorCommand,
    ActuatorCommandKind,
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequestKind,
    VramVerdict,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import _WholeCardDemandOutcome
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _over_committed_state() -> DeviceVramState:
    """A device state whose committed floor sits over the card's capacity (the worker's own load holds it)."""
    return DeviceVramState(
        total_vram_mb=16000.0,
        baseline_mb=0.0,
        committed_vram_mb=16000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
    )


def _fitting_state() -> DeviceVramState:
    """A fresh device state with ample capacity, so an ordinary candidate fits."""
    return DeviceVramState(
        total_vram_mb=24000.0,
        baseline_mb=1000.0,
        committed_vram_mb=2000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
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


def _install_foreign_pressure_verdict(scheduler) -> None:  # noqa: ANN001
    """Force the arbiter to return a foreign-pressure fit-into-reality FITS for the next preload evaluation."""
    measured = evaluate_admission(
        measured_committed_mb=16000.0,
        planned_unmaterialized_mb=0.0,
        candidate_delta_mb=5000.0,
        total_vram_mb=24000.0,
        baseline_mb=1000.0,
        committed_is_stale=False,
    )
    verdict = VramVerdict(
        disposition=VramDisposition.FITS,
        request_kind=VramRequestKind.PRELOAD,
        device_index=None,
        reason="foreign pressure but candidate physically fits device-free: admit into reality",
        measured=measured,
        foreign_pressure_admit=True,
    )
    fake_arbiter = Mock()
    fake_arbiter.evaluate.return_value = verdict
    scheduler._ensure_preload_arbiter = Mock(return_value=fake_arbiter)  # type: ignore[method-assign]


class TestAdapterSeamAdmitsOnlyOnFits:
    """No preload proceeds without a FITS verdict at the adapter seam."""

    async def test_deferred_verdict_does_not_admit(self) -> None:
        """An over-committed preload defers (its own committed load holds the card): the adapter returns False."""
        scheduler, job, target = await _budgeted_scheduler_with_head()
        _install_cycle(scheduler, _over_committed_state())

        admitted = scheduler._admit_preload_under_budget(job, target, is_head_blocker=True)

        assert admitted is False
        assert target.last_control_flag != HordeControlFlag.PRELOAD_MODEL

    async def test_fitting_verdict_admits(self) -> None:
        """A candidate within the measured floor is admitted (the RAM verdict then passes)."""
        scheduler, job, target = await _budgeted_scheduler_with_head()
        _install_cycle(scheduler, _fitting_state())

        admitted = scheduler._admit_preload_under_budget(job, target, is_head_blocker=True)

        assert admitted is True

    async def test_starved_head_over_committed_still_defers(self) -> None:
        """A head deferred on the clock still defers when its own committed load holds the card."""
        scheduler, job, target = await _budgeted_scheduler_with_head()
        _install_cycle(scheduler, _over_committed_state())
        scheduler._head_starved_seconds = Mock(return_value=120.0)

        admitted = scheduler._admit_preload_under_budget(job, target, is_head_blocker=True)

        assert admitted is False
        assert scheduler._job_tracker.is_admitted_over_budget(job) is False


class TestForeignPressureAdmitMarksOverBudget:
    """The foreign-pressure fit-into-reality FITS routes through the over-budget marking, bypassing the RAM verdict."""

    async def test_foreign_pressure_admit_marks_over_budget_and_skips_ram(self) -> None:
        """A foreign-pressure FITS admits, tags the job over-budget, and does not run the marginal RAM verdict."""
        scheduler, job, target = await _budgeted_scheduler_with_head()
        _install_foreign_pressure_verdict(scheduler)
        scheduler._apply_ram_verdict = Mock(return_value=True)  # type: ignore[method-assign]

        admitted = scheduler._admit_preload_under_budget(job, target, is_head_blocker=True)

        assert admitted is True
        assert scheduler._job_tracker.is_admitted_over_budget(job) is True
        scheduler._apply_ram_verdict.assert_not_called()

    async def test_ordinary_fit_runs_the_ram_verdict(self) -> None:
        """The control: an ordinary within-capacity FITS runs the marginal RAM verdict and tags nothing."""
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
        """The over-committed preload defers and releases the idle lane cache; the relieved re-ask admits."""
        scheduler, job, target = await _budgeted_scheduler_with_head()
        scheduler.release_allocator_cache = Mock(return_value=True)  # type: ignore[method-assign]

        pressured = DeviceVramState(
            total_vram_mb=16000.0,
            baseline_mb=0.0,
            committed_vram_mb=16000.0,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            idle_process_ids=frozenset({5}),
        )
        _install_cycle(scheduler, pressured)

        assert scheduler._admit_preload_under_budget(job, target, is_head_blocker=True) is False
        scheduler.release_allocator_cache.assert_called_once_with(5)

        # Next cycle: the released lane cache drops the committed floor under capacity, so the re-ask admits.
        _install_cycle(scheduler, _fitting_state())
        assert scheduler._admit_preload_under_budget(job, target, is_head_blocker=True) is True
