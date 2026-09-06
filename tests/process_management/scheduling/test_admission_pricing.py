"""Each pure pricing function gives the scheduler's own answer over a snapshot of the same worker.

These are differential rows: while the scheduler still prices inline, the snapshot form must agree with it on
every configuration a gate can meet. When a scheduler method is removed, its row here becomes the function's
own specification.
"""

from __future__ import annotations

import sys
from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.vram_footprints import (
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
)
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _slot(
    process_id: int,
    *,
    model: str | None,
    state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB,
    process_type: HordeProcessType = HordeProcessType.INFERENCE,
    reserved_mb: int | None = 3000,
) -> object:
    slot = make_mock_process_info(process_id, model_name=model, state=state, process_type=process_type)
    slot.process_reserved_mb = reserved_mb
    return slot


async def _worker(
    *,
    slots: dict[int, object],
    pending: list[str],
    in_progress: list[str] | None = None,
    learned: dict[tuple[str, FootprintStage], float] | None = None,
) -> tuple[InferenceScheduler, list[object]]:
    """A scheduler over the given slots and queue, with a learned store seeded as requested."""
    tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        process_map=ProcessMap(slots),  # type: ignore[arg-type]
        job_tracker=tracker,
        device_free_mb=6000.0,
        max_inference=2,
    )
    scheduler._process_map.get_free_vram_mb = Mock(return_value=6000.0)  # type: ignore[method-assign]
    scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)  # type: ignore[method-assign]
    jobs = []
    for model in pending:
        job = make_job_pop_response(model=model)
        await track_popped_job_async(tracker, job)
        jobs.append(job)
    for model in in_progress or []:
        job = make_job_pop_response(model=model)
        await track_popped_job_async(tracker, job)
        await mark_job_in_progress_async(tracker, job)
        jobs.append(job)
    if learned:
        store = LearnedFootprintStore()
        for (baseline, stage), peak_mb in learned.items():
            store.observe_peak(
                FootprintKey(model_baseline=baseline, resolution_bucket=None, platform=sys.platform, stage=stage),
                peak_mb,
            )
        scheduler.set_footprint_store(store)
    return scheduler, jobs


class TestForecastAndDeltas:
    """The forecast, the candidate delta and the co-resident maximum agree with the scheduler."""

    async def test_streaming_forecast_matches(self) -> None:
        """The full forecast dataclass is identical, resident credit and overheads included."""
        scheduler, jobs = await _worker(
            slots={0: _slot(0, model="stable_diffusion"), 1: _slot(1, model=None)},
            pending=["stable_diffusion", "other"],
        )
        snapshot = scheduler.snapshot()
        for job in jobs:
            baseline = scheduler._model_metadata.get_baseline(job.model)  # type: ignore[attr-defined]
            assert pricing.forecast_streaming(snapshot, job, baseline) == scheduler._forecast_streaming(  # type: ignore[attr-defined]
                job,
                baseline,
            )

    async def test_candidate_delta_and_learned_peak_match(self) -> None:
        """Resident credit on one slot and a learned raise on the sampling key both carry over."""
        scheduler, jobs = await _worker(
            slots={0: _slot(0, model="stable_diffusion"), 1: _slot(1, model=None)},
            pending=["stable_diffusion"],
            learned={("stable_diffusion_1", FootprintStage.SAMPLE): 9000.0},
        )
        snapshot = scheduler.snapshot()
        job = jobs[0]
        baseline = scheduler._model_metadata.get_baseline(job.model)  # type: ignore[attr-defined]
        for process_id in (0, 1, None):
            for disaggregated in (False, True):
                assert pricing.candidate_delta_mb(
                    snapshot,
                    job,
                    baseline,
                    process_id=process_id,
                    disaggregated=disaggregated,
                ) == scheduler._measured_admission_candidate_delta_mb(  # type: ignore[attr-defined]
                    job,
                    baseline,
                    process_id=process_id,
                    disaggregated=disaggregated,
                )

    async def test_max_coresident_sizes_from_the_card_total_and_overheads(self) -> None:
        """The structural depth is the loader's context plus the marginal contexts the remaining total seats."""
        scheduler, _ = await _worker(slots={0: _slot(0, model=None)}, pending=[])
        scheduler.set_measured_per_process_overhead_mb(1000.0, device_index=0)
        scheduler.set_measured_marginal_overhead_mb(500.0, device_index=0)
        snapshot = scheduler.snapshot()
        # 16000 - 4000 - 1024 = 10976 budget; (10976 - 1000) // 500 = 19 extra contexts beside the loader's.
        assert pricing.max_coresident_for_peak_mb(snapshot, 4000.0, 1024.0) == 20
        # The loader's own context always counts, so a peak past the budget still sizes to one.
        assert pricing.max_coresident_for_peak_mb(snapshot, 15500.0, 1024.0) == 1

    async def test_unpausable_tenancy_matches(self) -> None:
        """A utilities lane on the card is charged the same way, per the lane-reclaim policy."""
        scheduler, _ = await _worker(
            slots={0: _slot(0, model=None), 5: _slot(5, model=None, process_type=HordeProcessType.UTILITIES)},
            pending=[],
        )
        snapshot = scheduler.snapshot()
        assert pricing.unpausable_tenancy_mb(snapshot, None) == scheduler._unpausable_tenancy_mb(None)  # type: ignore[attr-defined]


class TestReclaimPredicates:
    """The reclaim and teardown predicates across the guard combinations."""

    async def _reclaim_worker(self):  # noqa: ANN202
        unloading = _slot(2, model="queued")
        unloading.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM  # type: ignore[attr-defined]
        scheduler, _ = await _worker(
            slots={
                0: _slot(0, model="resident"),
                1: _slot(1, model="busy", state=HordeProcessState.INFERENCE_STARTING),
                2: unloading,
                3: _slot(3, model="queued"),
            },
            pending=["queued", "other"],
            in_progress=["busy"],
        )
        return scheduler.snapshot()

    async def test_the_target_slot_is_spared_unless_it_holds_another_model(self) -> None:
        """Making room on the target itself counts only when it holds a model other than the one being seated."""
        snapshot = await self._reclaim_worker()
        reclaimable = {
            make_room: pricing.has_reclaimable_idle_model(
                snapshot, 0, for_head_of_queue=False, device_index=None, make_room_for_model=make_room
            )
            for make_room in (None, "resident", "other")
        }
        # The busy, unloading and affordable queued-lookahead slots are never candidates for a follower, so
        # the target's own resident model is the only one an "other" load may displace.
        assert reclaimable == {None: False, "resident": False, "other": True}

    async def test_the_head_overrides_the_lookahead_guard(self) -> None:
        """A queued model's idle copy is spared for a follower but reclaimable for the head."""
        snapshot = await self._reclaim_worker()
        assert pricing.has_reclaimable_idle_model(snapshot, 3, for_head_of_queue=False, device_index=None) is True
        assert pricing.has_reclaimable_idle_model(snapshot, 0, for_head_of_queue=True, device_index=None) is True
        assert pricing.has_reclaimable_idle_model(snapshot, 0, for_head_of_queue=False, device_index=None) is False

    async def test_teardown_predicates(self) -> None:
        """The teardownable siblings, what the bare ones return, and the warm tenancy the head could reclaim."""
        components = _slot(2, model=None)
        components.held_components = [Mock()]  # type: ignore[attr-defined]
        scheduler, _jobs = await _worker(
            slots={
                0: _slot(0, model="head"),
                1: _slot(1, model=None, reserved_mb=800),
                2: components,
                3: _slot(3, model="busy", state=HordeProcessState.INFERENCE_STARTING),
            },
            pending=["head"],
            in_progress=["busy"],
        )
        snapshot = scheduler.snapshot()
        assert pricing.has_teardownable_idle_context(snapshot, 0, device_index=None) is True
        assert [slot.process_id for slot in pricing.teardownable_idle_contexts(snapshot, 0, device_index=None)] == [
            1,
            2,
        ]
        # Only the bare idle sibling returns its reservation plus a context; the one holding components does not.
        assert pricing.teardownable_idle_context_returns_mb(snapshot, 0, device_index=None) == [
            800.0 + pricing.marginal_or_seed_mb(snapshot, None),
        ]
        assert pricing.has_reclaimable_idle_tenancy(snapshot, "head", 0, device_index=None) is True
        assert pricing.has_reclaimable_idle_tenancy(snapshot, "head", 2, device_index=None) is False

    async def test_lookahead_affordability_matches(self) -> None:
        """The static lookahead fit reads the same head and the same reserve."""
        scheduler, _ = await _worker(slots={0: _slot(0, model="resident")}, pending=["head"])
        snapshot = scheduler.snapshot()
        assert pricing.coresident_lookahead_affordable(snapshot, "resident", device_index=None) is (
            scheduler._coresident_lookahead_affordable("resident", device_index=None)  # type: ignore[attr-defined]
        )
