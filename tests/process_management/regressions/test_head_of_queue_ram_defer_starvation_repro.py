"""Head-of-queue starvation when a system-RAM verdict persistently defers while a sibling job runs.

The preload admission path prices a head-of-queue model against the frozen cycle measurement; a VRAM FITS
runs the marginal system-RAM verdict (:meth:`InferenceScheduler._apply_ram_verdict`). When that verdict does
not fit, the RAM branch runs its reclaim attempts and then dispatches on :func:`decide_ram_reclaim_outcome`.
Its only escape from an indefinite defer is ``BEST_EFFORT_ADMIT``, gated on ``no_live_resource_consumer`` (no
job in progress anywhere on the card). While any sibling job is in progress that gate is closed, so a head
whose RAM verdict keeps failing and whose reclaim attempts free nothing would defer every cycle with no way to
reach the escape on its own.

The head-priority barrier resolves that. When a head-of-queue preload has been continuously RAM-deferred
behind live work past ``_HEAD_RAM_DEFER_BARRIER_SECONDS`` with reclaim freeing nothing, the scheduler latches a
barrier: new inference dispatch to other slots is withheld so the running siblings drain to the
no-live-consumer best-effort admit that seats the head. The barrier is edge-triggered, releases the moment the
head is admitted, dispatched, faulted, or departs, and is hard-capped: a barrier that has held past
``_HEAD_RAM_DEFER_BARRIER_CAP_SECONDS`` without admitting the head declines it for reissue rather than holding
dispatch forever.

Contract asserted here: past the bound with a sibling busy and reclaim freeing nothing, the head stays pending
and new dispatch to other slots is withheld (the dispatch seam); once the sibling completes the head is
admitted best-effort and the barrier releases so dispatch resumes; a head still unadmittable past the hard cap
is declined for reissue rather than left pending. The controls bound the reproduction: with no sibling in
progress the head is admitted best-effort at once, and a head whose RAM verdict fits is admitted with no
barrier at all.
"""

from __future__ import annotations

import time
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_models import NextJobAndProcess
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import BudgetVerdict
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
)
from horde_worker_regen.process_management.scheduling import inference_scheduler as _sched_mod
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _HEAD_RAM_DEFER_BARRIER_CAP_SECONDS,
    _HEAD_RAM_DEFER_BARRIER_SECONDS,
    InferenceScheduler,
    _WholeCardDemandOutcome,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


class _ClockShim(ModuleType):
    """Stands in for the scheduler module's ``time`` so ``time.time()`` is hand-advanceable.

    Only ``time`` (wall clock) and ``monotonic`` are overridden with the controllable value; every other
    attribute delegates to the real module, so any incidental time call in the admission path still works.
    """

    def __init__(self) -> None:
        super().__init__("time")
        self.now = 10_000.0

    def time(self) -> float:  # type: ignore[override]
        return self.now

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name: str) -> object:
        return getattr(time, name)


def _fitting_state() -> DeviceVramState:
    """A device state with ample device-free room, so the VRAM arbiter admits (reaching the RAM verdict)."""
    return DeviceVramState(
        total_vram_mb=24000.0,
        baseline_mb=1000.0,
        committed_vram_mb=2000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=21000.0,
    )


def _install_fitting_cycle(scheduler: InferenceScheduler) -> None:
    """Freeze a fitting arbiter cycle so every preload evaluation this cycle is a VRAM FITS."""
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: _fitting_state()}))
    scheduler._vram_arbiter = arbiter


def _hard_ram_defer_verdict() -> BudgetVerdict:
    """A RAM verdict that does not fit: the head's predicted cost exceeds available RAM by a wide margin."""
    return BudgetVerdict(fits=False, predicted_mb=12000.0, available_mb=200.0, reserve_mb=4096.0)


async def _scheduler_with_head_and_target() -> tuple[
    InferenceScheduler, ImageGenerateJobPopResponse, HordeProcessInfo
]:
    """A budget-gated scheduler with one empty target slot and one pending head-of-queue job."""
    target = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    process_map = ProcessMap({0: target})
    job_tracker = JobTracker()
    head = make_job_pop_response("head_model")
    await track_popped_job_async(job_tracker, head)
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(enable_vram_budget=True, vram_reserve_mb=2000, ram_reserve_mb=4096),
        max_concurrent=2,
        max_inference=2,
    )
    scheduler._decide_whole_card_demand = Mock(return_value=_WholeCardDemandOutcome.FALL_THROUGH)  # type: ignore[method-assign]
    scheduler._measured_available_ram_mb = lambda: 200.0  # type: ignore[method-assign]
    return scheduler, head, target


async def _add_busy_sibling(
    scheduler: InferenceScheduler, *, model: str = "sibling_model"
) -> ImageGenerateJobPopResponse:
    """Mark a sibling job in progress so the worker has a live resource consumer during the head's wait."""
    sibling = make_job_pop_response(model)
    await track_popped_job_async(scheduler._job_tracker, sibling)
    await scheduler._job_tracker.mark_inference_started(sibling)
    assert len(scheduler._job_tracker.jobs_in_progress) == 1
    return sibling


def _force_unreclaimable_ram(scheduler: InferenceScheduler) -> None:
    """Pin the RAM verdict to a persistent non-fit whose reclaim attempts free nothing.

    Models the starvation premise: the head's weights do not fit available RAM and no idle RAM or
    allocator-stuck slot can be reclaimed to make room, so the RAM branch can only defer or best-effort admit.
    """
    scheduler._ram_budget.check_job = Mock(return_value=_hard_ram_defer_verdict())  # type: ignore[method-assign]
    scheduler.unload_models = Mock(return_value=False)  # type: ignore[method-assign]
    scheduler._replace_stale_ram_unload_process = Mock(return_value=False)  # type: ignore[method-assign]


class TestHeadRamDeferBarrier:
    """A RAM-deferred head behind a busy sibling latches a dispatch barrier rather than starving."""

    async def test_barrier_engages_and_withholds_dispatch_after_bound_with_sibling_busy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Past the bound with a sibling busy, the head stays pending and new sibling dispatch is withheld.

        The VRAM arbiter fits, so admission reaches the RAM verdict, which never fits; reclaim frees nothing
        and a sibling is in progress, so the head cannot reach the no-live-consumer escape. After the admission
        clock advances past the barrier bound, the head defers again (it is not force-admitted) and the
        head-priority barrier latches: at the dispatch seam a sibling job is withheld while the head's own job
        would not be.
        """
        clock = _ClockShim()
        monkeypatch.setattr(_sched_mod, "time", clock)

        scheduler, head, target = await _scheduler_with_head_and_target()
        sibling = await _add_busy_sibling(scheduler)
        _force_unreclaimable_ram(scheduler)
        _install_fitting_cycle(scheduler)

        # First evaluation: the head defers (RAM does not fit, a sibling holds memory) and the clock starts.
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False
        assert scheduler._head_priority_barrier_job_id is None

        # Time passes past the barrier bound with the sibling still busy and the RAM verdict still failing.
        clock.now += _HEAD_RAM_DEFER_BARRIER_SECONDS + 1.0
        _install_fitting_cycle(scheduler)
        assert len(scheduler._job_tracker.jobs_in_progress) == 1

        # The head is not force-admitted; instead the barrier latches on this head.
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False
        assert scheduler._head_priority_barrier_job_id == str(head.id_)

        # At the dispatch seam the barrier withholds a sibling dispatch while sparing the head's own job.
        assert scheduler._head_priority_barrier_withholds_dispatch(sibling) is True
        assert scheduler._head_priority_barrier_withholds_dispatch(head) is False

        scheduler.get_next_job_and_process = AsyncMock(  # type: ignore[method-assign]
            return_value=NextJobAndProcess(next_job=sibling, process_with_model=target),
        )
        assert await scheduler.start_inference() is False, "the barrier must withhold a new sibling dispatch"
        assert not target.pipe_connection.send.called

    async def test_head_admitted_and_barrier_releases_when_sibling_drains(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the sibling completes the head is admitted best-effort and the barrier releases (liveness).

        A permanent-hold barrier would be a new wedge class. This proves the release edge: once the barrier is
        engaged and the sibling drains, the existing no-live-consumer escape admits the head, the barrier is
        released, and the dispatch seam stops withholding new work.
        """
        clock = _ClockShim()
        monkeypatch.setattr(_sched_mod, "time", clock)

        scheduler, head, target = await _scheduler_with_head_and_target()
        sibling = await _add_busy_sibling(scheduler)
        _force_unreclaimable_ram(scheduler)
        _install_fitting_cycle(scheduler)

        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False
        clock.now += _HEAD_RAM_DEFER_BARRIER_SECONDS + 1.0
        _install_fitting_cycle(scheduler)
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False
        assert scheduler._head_priority_barrier_job_id == str(head.id_)
        assert scheduler._head_priority_barrier_withholds_dispatch(sibling) is True

        # The barred dispatch let the sibling finish; with no live consumer the escape now admits the head.
        await scheduler._job_tracker.handle_job_fault(sibling, retryable=False)
        assert len(scheduler._job_tracker.jobs_in_progress) == 0
        _install_fitting_cycle(scheduler)

        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is True
        assert scheduler._head_priority_barrier_job_id is None
        assert scheduler._head_priority_barrier_withholds_dispatch(sibling) is False

    async def test_head_declined_for_reissue_after_hard_cap_with_sibling_busy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A head still unadmittable past the hard cap is declined for reissue rather than left pending.

        Barring dispatch normally drains the siblings; if a sibling never completes (so the head never reaches
        the escape) the barrier would otherwise hold forever. The hard cap fails the head fast through the
        existing retryable fault machinery so the horde reissues it, and releases the barrier.
        """
        clock = _ClockShim()
        monkeypatch.setattr(_sched_mod, "time", clock)

        scheduler, head, target = await _scheduler_with_head_and_target()
        await _add_busy_sibling(scheduler)
        _force_unreclaimable_ram(scheduler)
        _install_fitting_cycle(scheduler)

        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False
        clock.now += _HEAD_RAM_DEFER_BARRIER_SECONDS + 1.0
        _install_fitting_cycle(scheduler)
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False
        assert scheduler._head_priority_barrier_job_id == str(head.id_)

        # The sibling never finishes; past the hard cap the head is declined for reissue and the barrier drops.
        clock.now += _HEAD_RAM_DEFER_BARRIER_CAP_SECONDS + 1.0
        _install_fitting_cycle(scheduler)
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False
        assert scheduler._head_priority_barrier_job_id is None
        assert head.id_ is not None
        assert scheduler._job_tracker.get_stage(head.id_) is JobStage.PENDING_SUBMIT

    async def test_head_with_no_live_consumer_is_admitted_best_effort_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no sibling in progress the head is admitted best-effort at once (control, passes today).

        This is the existing escape: an exhausted RAM reclaim with no live job holding memory admits the head
        rather than wedging the queue. It bounds the reproduction to the sibling-busy case.
        """
        clock = _ClockShim()
        monkeypatch.setattr(_sched_mod, "time", clock)

        scheduler, head, target = await _scheduler_with_head_and_target()
        _force_unreclaimable_ram(scheduler)
        _install_fitting_cycle(scheduler)

        assert len(scheduler._job_tracker.jobs_in_progress) == 0
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is True
        assert scheduler._head_priority_barrier_job_id is None

    async def test_head_whose_ram_verdict_fits_is_admitted_with_sibling_busy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A head whose RAM verdict fits is admitted even while a sibling runs (control, passes today).

        No starvation arises when RAM has room: the sibling-busy gate only matters once the verdict fails, so
        a fitting verdict admits immediately and no barrier latches. Confirms the reproduction isolates the
        failing-verdict path.
        """
        clock = _ClockShim()
        monkeypatch.setattr(_sched_mod, "time", clock)

        scheduler, head, target = await _scheduler_with_head_and_target()
        await _add_busy_sibling(scheduler)
        scheduler._ram_budget.check_job = Mock(  # type: ignore[method-assign]
            return_value=BudgetVerdict(fits=True, predicted_mb=1000.0, available_mb=64000.0, reserve_mb=4096.0),
        )
        _install_fitting_cycle(scheduler)

        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is True
        assert scheduler._head_priority_barrier_job_id is None
