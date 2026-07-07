"""Dispatch-time residency reconciliation: the gate that prices a staged job's VRAM against the card.

These pin the contracts the dispatch gate adds. Admission is consulted at preload and at the second
concurrent sampler, but the moment an already-RAM-staged job's weights actually commit to VRAM crosses no
gate on its own, so a job materialising beside an idle sibling's resident weights can over-commit the card
faster than tick-paced reclaim reacts. The gate prices that materialisation through the arbiter's single
MONOLITHIC_DISPATCH identity: a FITS releases the dispatch, a conflict holds it (the job keeps its queue
position, never faulted) and routes idle-resident eviction through the one reclaim owner, protecting the
head's own slot. The held job is not reaped by the watchdogs that time the preloaded-to-inference-started
transition, because it stays pending with its model resident and never enters in-progress.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.message_dispatcher import (
    _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS,
    DeadlockSnapshot,
)
from horde_worker_regen.process_management.ipc.messages import (
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _fitting_state() -> DeviceVramState:
    """A device state with ample capacity, so a staged dispatch fits without reclaim."""
    return DeviceVramState(
        total_vram_mb=24000.0,
        baseline_mb=1000.0,
        committed_vram_mb=2000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
    )


def _over_committed_state() -> DeviceVramState:
    """A device state whose committed floor already sits over capacity (idle residents hold the card)."""
    return DeviceVramState(
        total_vram_mb=16000.0,
        baseline_mb=0.0,
        committed_vram_mb=16000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
    )


def _install_cycle(scheduler, state: DeviceVramState) -> None:  # noqa: ANN001
    """Freeze a crafted arbiter cycle on the scheduler's arbiter."""
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
    scheduler._vram_arbiter = arbiter


async def _scheduler_with_idle_sibling():  # noqa: ANN202
    """A scheduler whose head model is staged on its slot beside an evictable idle resident sibling."""
    target = make_mock_process_info(0, model_name="model_a", state=HordeProcessState.PRELOADED_MODEL)
    sibling = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.WAITING_FOR_JOB)
    process_map = ProcessMap({0: target, 1: sibling})
    job_tracker = JobTracker()
    job = make_job_pop_response("model_a")
    await track_popped_job_async(job_tracker, job)
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(image_models_to_load=["model_a", "model_b"]),
        max_concurrent=2,
        max_inference=2,
    )
    return scheduler, job, target, sibling


async def _scheduler_without_sibling():  # noqa: ANN202
    """A scheduler with only the head's slot: an over-commit here has no idle resident to evict."""
    target = make_mock_process_info(0, model_name="model_a", state=HordeProcessState.PRELOADED_MODEL)
    process_map = ProcessMap({0: target})
    job_tracker = JobTracker()
    job = make_job_pop_response("model_a")
    await track_popped_job_async(job_tracker, job)
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(image_models_to_load=["model_a"]),
        max_concurrent=2,
        max_inference=2,
    )
    return scheduler, job, target


class TestGatePredicate:
    """The gate releases a fitting dispatch and holds a conflicting one."""

    async def test_fitting_dispatch_is_released(self) -> None:
        """A staged job that fits the card is not held and records no conflict."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        _install_cycle(scheduler, _fitting_state())

        held = scheduler._dispatch_residency_reconciliation_holds(job, target)

        assert held is False
        assert scheduler.latest_dispatch_reconciliation_holds() == 0
        assert scheduler.latest_dispatch_reconciliation_conflicts() == 0

    async def test_conflict_holds_and_routes_eviction_protecting_the_head(self) -> None:
        """An over-committing dispatch holds and evicts the idle resident through the reclaim owner."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        _install_cycle(scheduler, _over_committed_state())
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]

        held = scheduler._dispatch_residency_reconciliation_holds(job, target)

        assert held is True
        # The eviction ran through unload_models_from_vram anchored on the head's own slot (which the sweep
        # protects), so the idle sibling is evicted and the head's staged model is spared.
        scheduler.unload_models_from_vram.assert_called_once()
        assert scheduler.unload_models_from_vram.call_args.args[0] is target
        # The job is never faulted: it keeps its queue position.
        assert job in scheduler._job_tracker.jobs_pending_inference
        assert scheduler.latest_dispatch_reconciliation_holds() == 1
        assert scheduler.latest_dispatch_reconciliation_conflicts() == 1

    async def test_hold_releases_by_reclaim_after_verified_free(self) -> None:
        """A held dispatch whose eviction ran is released, on a later fitting pass, as reclaim-attributed."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]

        _install_cycle(scheduler, _over_committed_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

        # The reclaimed room drops the committed floor under capacity, so the re-ask releases the dispatch.
        _install_cycle(scheduler, _fitting_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False

        assert scheduler.latest_dispatch_reconciliation_released_by_reclaim() == 1
        assert scheduler.latest_dispatch_reconciliation_released_by_natural_free() == 0
        assert scheduler.latest_dispatch_reconciliation_hold_seconds() >= 0.0

    async def test_hold_releases_by_natural_free_when_no_eviction_was_emitted(self) -> None:
        """A held dispatch with nothing to evict is released as natural-free when the card recovers on its own."""
        scheduler, job, target = await _scheduler_without_sibling()

        _install_cycle(scheduler, _over_committed_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

        _install_cycle(scheduler, _fitting_state())
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False

        assert scheduler.latest_dispatch_reconciliation_released_by_reclaim() == 0
        assert scheduler.latest_dispatch_reconciliation_released_by_natural_free() == 1


class TestHeldDispatchSurvivesWatchdogs:
    """A held dispatch is not reaped by the clocks that time the preloaded-to-inference-started transition."""

    async def test_stale_model_map_expiry_spares_the_held_resident(self) -> None:
        """The stale-entry expiry never touches the held head's resident model (it is not a LOADING entry)."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]
        scheduler._horde_model_map.update_entry(
            "model_a",
            load_state=ModelLoadState.LOADED_IN_VRAM,
            process_id=target.process_id,
        )
        _install_cycle(scheduler, _over_committed_state())

        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

        expired = scheduler._expire_stale_model_map_entries()

        assert "model_a" not in expired
        assert "model_a" in scheduler._horde_model_map.root
        # The held job is still queued (never faulted) and never entered in-progress, so the lost-result reap
        # and the orphaned-in-progress reconciler have nothing to act on.
        assert job in scheduler._job_tracker.jobs_pending_inference
        assert job not in scheduler._job_tracker.jobs_in_progress

    async def test_fresh_hold_is_not_a_structural_wedge(self) -> None:
        """A just-formed hold is far below the structural-wedge horizon, so the recovery supervisor stays out."""
        scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]
        _install_cycle(scheduler, _over_committed_state())

        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

        # The held dispatch presents to the deadlock detector as an all-idle queue whose head model is resident;
        # that is a queue deadlock, but only a queue deadlock that persists past the horizon is a structural
        # wedge. A fresh hold (reclaim resolves within a few ticks) never reaches it.
        snapshot = DeadlockSnapshot(
            in_deadlock=False,
            in_queue_deadlock=True,
            deadlock_started_at=0.0,
            queue_deadlock_started_at=100.0,
            queue_deadlock_model="model_a",
            queue_deadlock_process_id=target.process_id,
        )
        assert snapshot.indicates_structural_wedge(now=100.0) is False
        assert snapshot.indicates_structural_wedge(now=100.0 + _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS + 1.0) is True
