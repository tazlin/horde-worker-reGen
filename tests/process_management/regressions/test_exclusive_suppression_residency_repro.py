"""Reproduces worker-wide dispatch suppression by an exclusive admit that holds nothing on the card.

``TrackedJob.admitted_exclusive`` is sticky: it is set when a whole-card head is admitted and stays set for
the life of the job because it also carries fault attribution and the over-budget step grace. The dispatch
guard read that flag alone, so a marked head whose whole-card establishment was deferred by the residency
rate limiter suppressed every other job's dispatch, silently, for as long as the deferral lasted. Nothing
held the card during that window: the residency had not been granted and the job was not sampling.

Dispatch suppression follows a live claim on the card instead: an exclusive job actually sampling, or a
staged one whose whole-card residency is held or being established. Both of those still suppress, so a
running over-budget job keeps the isolation it was admitted for.
"""

from __future__ import annotations

import time

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap, ModelLoadState
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyBucket
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_RESIDENT_MODEL = "model_alpha"
_HEAVY_MODEL = "model_heavy"


async def _scheduler_with_deferred_exclusive_head(
    job_tracker: JobTracker,
) -> tuple[InferenceScheduler, object]:
    """One idle resident slot, a dispatchable job, and a queued exclusive job holding no residency.

    The exclusive job's model is not resident anywhere: it is the whole-card head waiting for its
    establishment, which the rate limiter has deferred, so no residency exists on the card.
    """
    slot = make_mock_process_info(1, model_name=_RESIDENT_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    process_map = ProcessMap({1: slot})
    horde_model_map = HordeModelMap(root={})
    horde_model_map.update_entry(
        horde_model_name=_RESIDENT_MODEL,
        load_state=ModelLoadState.LOADED_IN_RAM,
        process_id=1,
    )

    ready_job = make_job_pop_response(model=_RESIDENT_MODEL)
    heavy_job = make_job_pop_response(model=_HEAVY_MODEL)
    await job_tracker.record_popped_job(ready_job)
    await job_tracker.record_popped_job(heavy_job)
    job_tracker.mark_admitted_over_budget(heavy_job)
    job_tracker.mark_admitted_exclusive(heavy_job)

    scheduler = _make_inference_scheduler(
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(max_threads=2),
        max_concurrent=2,
        max_inference=2,
    )
    return scheduler, ready_job


def _grant_residency(scheduler: InferenceScheduler, model: str) -> None:
    """Record a live whole-card residency for ``model`` on the worker-wide (single-GPU) card key."""
    scheduler._whole_card_ledger.record_grant(
        None,
        model=model,
        forecast=None,
        cooldown_until=time.time() + 60.0,
        now=time.time(),
        refresh_established=True,
    )


async def test_marked_exclusive_head_without_residency_does_not_suppress_dispatch(
    job_tracker: JobTracker,
) -> None:
    """A deferred whole-card establishment leaves the card free, so other queued work must dispatch."""
    scheduler, ready_job = await _scheduler_with_deferred_exclusive_head(job_tracker)
    assert job_tracker.has_exclusive_job_in_progress() is True

    selected = await scheduler.get_next_job_and_process()

    assert selected is not None
    assert selected.next_job is ready_job


async def test_established_residency_restores_the_suppression(job_tracker: JobTracker) -> None:
    """Once the exclusive head's residency is up, the card is genuinely given away and dispatch is held."""
    scheduler, _ready_job = await _scheduler_with_deferred_exclusive_head(job_tracker)
    _grant_residency(scheduler, _HEAVY_MODEL)

    selected = await scheduler.get_next_job_and_process()

    assert selected is None


async def test_running_exclusive_job_still_suppresses_without_a_residency(job_tracker: JobTracker) -> None:
    """A sampling over-budget job keeps its isolation: its footprint is on the card regardless of the ledger."""
    scheduler, ready_job = await _scheduler_with_deferred_exclusive_head(job_tracker)
    heavy_job = next(job for job in job_tracker.jobs_pending_inference if job is not ready_job)
    await job_tracker.mark_inference_started(heavy_job)

    selected = await scheduler.get_next_job_and_process()

    assert selected is None


async def test_suppressed_dispatch_is_attributed_to_the_exclusive_admit(job_tracker: JobTracker) -> None:
    """The held slot is named as exclusive isolation rather than reported as a gate-less stall."""
    scheduler, ready_job = await _scheduler_with_deferred_exclusive_head(job_tracker)
    _grant_residency(scheduler, _HEAVY_MODEL)
    reference = {
        _RESIDENT_MODEL: make_mock_model_reference_record(_RESIDENT_MODEL),
        _HEAVY_MODEL: make_mock_model_reference_record(_HEAVY_MODEL),
    }
    scheduler._model_metadata = make_test_model_metadata(reference)

    bucket, text = scheduler._classify_dispatch_stall(ready_job, reference)

    assert bucket is SlotDutyBucket.EXCLUSIVE_ISOLATION
    assert "exclusively-admitted" in text
