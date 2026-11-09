"""Tests for what the recovery coordinator accepts as forward progress when closing a wedge episode.

The escalation ladder resets only on progress the failure mode cannot manufacture. An attempt counter
(inference starts) rises freely while a downstream stage is the thing that is stuck, so a backlog waiting
on that stage must not be credited as recovery.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.recovery_supervisor import RecoverySupervisor
from horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator import WorkerRecoveryCoordinator
from tests.process_management.conftest import (
    FakeClock,
    make_job_pop_response,
    make_test_recovery_coordinator,
    make_testable_process_manager,
    mark_job_in_progress_async,
    move_job_to_being_safety_checked_async,
    queue_job_for_safety_async,
    track_popped_job_async,
)


class TestSafetyBacklogIsNotProgress:
    """A safety-stage backlog is not disproved by more upstream inference starting."""

    async def test_inference_starts_do_not_credit_progress_while_safety_is_backlogged(self) -> None:
        """Starting more inference while safety checks pile up is not forward progress.

        A safety pool that cannot serve leaves accepted work parked after generation. Inference keeps
        starting behind it, so crediting the start counter would close the wedge episode on motion that
        the stalled stage itself produces, resetting the escalation instead of climbing it.
        """
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = make_test_recovery_coordinator(job_tracker=job_tracker, clock=clock)

        parked = await track_popped_job_async(job_tracker, make_job_pop_response())
        # The generated job parks waiting on safety, and a fresh job starts inference behind it.
        await queue_job_for_safety_async(job_tracker, _job_info_for(parked))
        coordinator._capture_progress_baseline()
        follower = await track_popped_job_async(job_tracker, make_job_pop_response())
        await mark_job_in_progress_async(job_tracker, follower)

        assert job_tracker.total_num_inference_starts > 0
        assert job_tracker.jobs_pending_safety_check
        assert coordinator.made_progress_since_episode() is False

    async def test_in_flight_safety_check_also_withholds_progress(self) -> None:
        """A safety check that has begun but never returns withholds credit just as a queued one does."""
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = make_test_recovery_coordinator(job_tracker=job_tracker, clock=clock)

        parked = await track_popped_job_async(job_tracker, make_job_pop_response())
        await move_job_to_being_safety_checked_async(job_tracker, _job_info_for(parked))
        coordinator._capture_progress_baseline()
        follower = await track_popped_job_async(job_tracker, make_job_pop_response())
        await mark_job_in_progress_async(job_tracker, follower)

        assert job_tracker.jobs_being_safety_checked
        assert coordinator.made_progress_since_episode() is False

    async def test_a_drained_safety_stage_restores_progress_credit(self) -> None:
        """With no safety backlog, an inference start is ordinary forward progress again.

        The control for the guard: withholding credit must be scoped to an actual safety backlog, or a
        healthy worker's wedge episode could never close.
        """
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = make_test_recovery_coordinator(job_tracker=job_tracker, clock=clock)

        coordinator._capture_progress_baseline()

        follower = await track_popped_job_async(job_tracker, make_job_pop_response())
        await mark_job_in_progress_async(job_tracker, follower)

        assert not job_tracker.jobs_pending_safety_check
        assert not job_tracker.jobs_being_safety_checked
        assert coordinator.made_progress_since_episode() is True


class TestFaultedWorkIsNotRecoveryProgress:
    """A terminal fault drains accounting but does not prove that the worker can serve accepted work."""

    async def test_terminal_fault_cannot_close_the_episode_that_produced_it(self) -> None:
        """Moving a pending job to faulted submission is not successful forward progress."""
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = make_test_recovery_coordinator(job_tracker=job_tracker, clock=clock)

        job = await track_popped_job_async(job_tracker, make_job_pop_response())
        coordinator._capture_progress_baseline()

        job_tracker.handle_job_fault_now(job, retryable=False)

        assert coordinator.episode_progress_baseline is not None
        assert job_tracker.total_num_completed_jobs > coordinator.episode_progress_baseline
        assert coordinator.made_progress_since_episode() is False

    async def test_a_baseline_finalized_as_faulted_is_not_progress(self) -> None:
        """Finalize stops tracking a faulted baseline job, and that removal is not read as a submit."""
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = make_test_recovery_coordinator(job_tracker=job_tracker, clock=clock)
        job_tracker.set_finalize_observer(coordinator.on_job_finalized)

        job = await track_popped_job_async(job_tracker, make_job_pop_response())
        coordinator._capture_progress_baseline()

        job_tracker.handle_job_fault_now(job, retryable=False)
        await _submit(job_tracker, job)

        assert job_tracker.num_jobs_total == 0
        assert coordinator.made_progress_since_episode() is False


_TICK_SECONDS = 1.0
_ESCALATION_TICK_BUDGET = 60


def _make_escalating_coordinator(clock: FakeClock, job_tracker: JobTracker) -> WorkerRecoveryCoordinator:
    """Build a coordinator over a structural queue wedge with tight, legible escalation windows.

    The give-up cool-down outlasts every test, so a give-up stays the single non-terminal one. The tracker's
    finalize observer is wired to the coordinator as the process manager's fan-out does.
    """
    coordinator = make_test_recovery_coordinator(job_tracker=job_tracker, clock=clock, structural_wedge=True)
    coordinator.recovery_supervisor = RecoverySupervisor(
        clock=clock,
        wedge_grace_seconds=1,
        reset_interval_seconds=1,
        max_soft_resets=1,
        pool_ready_grace_seconds=3,
        boot_allowance_seconds=20,
        give_up_cooldown_seconds=10_000,
        max_give_up_cycles=2,
        clean_streak_seconds=5,
    )
    job_tracker.set_finalize_observer(coordinator.on_job_finalized)
    return coordinator


def _set_structural_wedge(coordinator: WorkerRecoveryCoordinator, *, active: bool) -> None:
    """Set what the dispatcher's deadlock snapshot reports."""
    dispatcher = coordinator._message_dispatcher
    assert isinstance(dispatcher, Mock)
    dispatcher.get_deadlock_snapshot.return_value.indicates_structural_wedge.return_value = active


def _spy_on_lane_restore(coordinator: WorkerRecoveryCoordinator, monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Wrap the coordinator's lane restore, which runs when the episode's accounting is cleared."""
    spy = Mock(wraps=coordinator.restore_reclaimed_lanes)
    monkeypatch.setattr(coordinator, "restore_reclaimed_lanes", spy)
    return spy


def _tick_until(coordinator: WorkerRecoveryCoordinator, clock: FakeClock, done: Callable[[], bool]) -> None:
    """Drive the supervisor until ``done`` holds, failing if the escalation budget runs out first."""
    for _ in range(_ESCALATION_TICK_BUDGET):
        coordinator.run_recovery_supervisor()
        clock.advance(_TICK_SECONDS)
        if done():
            return
    raise AssertionError("the escalation did not reach the expected step")


def _tick_for(coordinator: WorkerRecoveryCoordinator, clock: FakeClock, seconds: float) -> None:
    """Drive the supervisor for ``seconds`` of fake time."""
    for _ in range(int(seconds / _TICK_SECONDS)):
        coordinator.run_recovery_supervisor()
        clock.advance(_TICK_SECONDS)


def _was_faulted(job_tracker: JobTracker, job: ImageGenerateJobPopResponse) -> bool:
    """Return whether ``job`` is waiting to submit a fault."""
    assert job.id_ is not None
    tracked = job_tracker.get_tracked_job(job.id_)
    return (
        tracked is not None
        and tracked.job_info is not None
        and tracked.job_info.state is GENERATION_STATE.faulted
        and tracked.stage is JobStage.PENDING_SUBMIT
    )


async def _submit(job_tracker: JobTracker, job: ImageGenerateJobPopResponse) -> None:
    """Queue ``job`` for submit and finalize it with whatever state its job info holds."""
    assert job.id_ is not None
    job_info = await job_tracker.get_job_info(job)
    assert job_info is not None
    if job_tracker.get_stage(job.id_) is not JobStage.PENDING_SUBMIT:
        await job_tracker.queue_for_submit(job_info)
    await job_tracker.finalize_submitted(job_info)


class TestRecoveryEpisodeCloses:
    """An episode that has spent a soft reset closes once real work moves past the most recent baseline."""

    async def test_a_baseline_that_submits_after_the_soft_reset_closes_the_episode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A baseline job finalized as submitted is forward progress, although finalize stops tracking it.

        The tracker deletes a job at finalize whether it submitted or faulted, so the frontier check alone
        cannot tell the two apart. The coordinator receives the outcome from the finalize observer.
        """
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = _make_escalating_coordinator(clock, job_tracker)
        baseline_job = await track_popped_job_async(job_tracker, make_job_pop_response())

        _tick_until(coordinator, clock, lambda: coordinator.recovery_supervisor.limp_by_level >= 1)
        assert coordinator.recovery_supervisor.is_in_episode
        restore_spy = _spy_on_lane_restore(coordinator, monkeypatch)

        _set_structural_wedge(coordinator, active=False)
        clock.advance(10)
        await mark_job_in_progress_async(job_tracker, baseline_job)
        await _submit(job_tracker, baseline_job)
        _tick_for(coordinator, clock, 20)

        assert coordinator.recovery_supervisor.is_in_episode is False
        assert restore_spy.called

    async def test_a_baseline_faulted_by_the_give_up_closes_once_later_work_advances(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The give-up re-captures the baseline, so new work moving forward afterwards closes the episode."""
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = _make_escalating_coordinator(clock, job_tracker)
        baseline_job = await track_popped_job_async(job_tracker, make_job_pop_response())

        _tick_until(coordinator, clock, lambda: _was_faulted(job_tracker, baseline_job))
        assert coordinator.recovery_supervisor.give_up_is_terminal is False
        assert coordinator.recovery_supervisor.is_in_episode
        restore_spy = _spy_on_lane_restore(coordinator, monkeypatch)

        _set_structural_wedge(coordinator, active=False)
        later_job = await track_popped_job_async(job_tracker, make_job_pop_response())
        await mark_job_in_progress_async(job_tracker, later_job)
        _tick_for(coordinator, clock, 20)

        assert coordinator.recovery_supervisor.is_in_episode is False
        assert restore_spy.called

    async def test_a_faulted_baseline_without_later_progress_keeps_the_episode_open(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A clean streak after the give-up is not proof of recovery while no work has moved."""
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = _make_escalating_coordinator(clock, job_tracker)
        baseline_job = await track_popped_job_async(job_tracker, make_job_pop_response())

        _tick_until(coordinator, clock, lambda: _was_faulted(job_tracker, baseline_job))
        restore_spy = _spy_on_lane_restore(coordinator, monkeypatch)

        _set_structural_wedge(coordinator, active=False)
        _tick_for(coordinator, clock, 30)

        assert coordinator.recovery_supervisor.is_in_episode
        assert not restore_spy.called

    async def test_the_give_up_faults_do_not_close_the_episode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Submitting the faults the give-up produced is the recovery's own bookkeeping, not progress.

        The give-up leaves no unresolved job here, so the aggregate fallback is what reads the faults.
        """
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = _make_escalating_coordinator(clock, job_tracker)
        baseline_job = await track_popped_job_async(job_tracker, make_job_pop_response())
        follower_job = await track_popped_job_async(job_tracker, make_job_pop_response())

        _tick_until(coordinator, clock, lambda: _was_faulted(job_tracker, baseline_job))
        assert _was_faulted(job_tracker, follower_job)
        restore_spy = _spy_on_lane_restore(coordinator, monkeypatch)

        _set_structural_wedge(coordinator, active=False)
        await _submit(job_tracker, baseline_job)
        await _submit(job_tracker, follower_job)
        _tick_for(coordinator, clock, 30)

        assert job_tracker.num_jobs_total == 0
        assert coordinator.recovery_supervisor.is_in_episode
        assert not restore_spy.called


class TestFinalizeOutcomeReachesRecovery:
    """The process manager's finalize fan-out reports each job's outcome to the recovery coordinator."""

    async def test_a_submitted_baseline_reads_as_progress_through_the_process_manager(self) -> None:
        """A baseline job submitted through the real tracker wiring is progress for the open baseline."""
        manager = make_testable_process_manager()
        coordinator = manager._recovery_coordinator
        job_tracker = manager._job_tracker
        baseline_job = await track_popped_job_async(job_tracker, make_job_pop_response())
        coordinator._capture_progress_baseline()

        await _submit(job_tracker, baseline_job)

        assert coordinator.made_progress_since_episode() is True


def _job_info_for(job: object) -> Mock:
    """Wrap a popped job in the job-info shape the safety queue expects."""
    job_info = Mock()
    job_info.sdk_api_job_info = job
    job_info.job_image_results = []
    return job_info
