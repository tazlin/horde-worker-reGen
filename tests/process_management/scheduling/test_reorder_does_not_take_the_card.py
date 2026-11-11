"""A job seated ahead of the head by the placement order does not take the whole card.

Only the head may take the card, since a teardown stops the lanes that serve the head. The placement order
seats a job ahead of the head when a free lane already holds its weights. A job seated that way reaches the
dispatch-time demand as if it were the head. If its forecast then shows a need for the card, the teardown stops
the lane where the head is loading. The head cannot load under the residency. Head protection holds the
promoted job for the head, so neither moves until head protection gives way.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.governance.whole_card import DominanceRung
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_HEAD_MODEL = "cold_head_model"
_RUN_MODEL = "run_model"

_SAMPLES_BESIDE_A_SIBLING = StreamForecast(
    weights_mb=11500.0,
    reserve_mb=3000.0,
    free_now_mb=15000.0,
    free_if_alone_mb=15000.0,
    free_after_model_evict_mb=15000.0,
)
"""A forecast whose weights fit the free room, so the job does not need the card."""

_NEEDS_THE_CARD = StreamForecast(
    weights_mb=11500.0,
    reserve_mb=3000.0,
    free_now_mb=2000.0,
    free_if_alone_mb=15000.0,
    free_after_model_evict_mb=13800.0,
    wants_whole_card=True,
)
"""A forecast that fits alone and not beside a sibling context, so a head with it is given the card."""


async def _run_with_a_cold_head(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rung: DominanceRung,
    forecast: StreamForecast,
) -> tuple[InferenceScheduler, object, object]:
    """Build a card held for the run model, its copy on a free lane, a cold head and one queued run job."""
    retainer = make_mock_process_info(1, model_name=_RUN_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    spare = make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    job_tracker = JobTracker()
    head_job = make_job_pop_response(_HEAD_MODEL)
    run_job = make_job_pop_response(_RUN_MODEL)
    await track_popped_job_async(job_tracker, head_job)
    await track_popped_job_async(job_tracker, run_job)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({1: retainer, 2: spare}), job_tracker=job_tracker)
    scheduler._whole_card_ledger.record_grant(
        None,
        model=_RUN_MODEL,
        forecast=None,
        cooldown_until=scheduler._clock() + 300.0,
        now=scheduler._clock(),
        rung=rung,
    )
    monkeypatch.setattr(scheduler, "_budget_active", lambda: True)
    monkeypatch.setattr(scheduler, "_whole_card_residency_enabled", lambda: True)
    monkeypatch.setattr(scheduler, "_forecast_streaming", lambda *_args, **_kwargs: forecast)
    monkeypatch.setattr(scheduler, "_expected_sampling_seconds", lambda *_args, **_kwargs: 5.0)
    return scheduler, head_job, run_job


async def test_a_run_job_that_samples_beside_a_sibling_is_seated_ahead_of_the_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control. A run job that does not need the card is promoted onto the run's copy."""
    scheduler, head_job, run_job = await _run_with_a_cold_head(
        monkeypatch,
        rung=DominanceRung.CANNOT_CO_SAMPLE,
        forecast=_SAMPLES_BESIDE_A_SIBLING,
    )

    promoted = [job for job, _lane in scheduler._retention_affinity_candidates(head_job)]

    assert promoted == [run_job]


async def test_a_run_job_whose_dispatch_would_take_the_card_is_not_seated_ahead_of_the_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job that would take the card stops the head's lane, so the head stays first in the placement order."""
    scheduler, head_job, _run_job = await _run_with_a_cold_head(
        monkeypatch,
        rung=DominanceRung.CANNOT_CO_SAMPLE,
        forecast=_NEEDS_THE_CARD,
    )
    assert _NEEDS_THE_CARD.needs_exclusive_residency

    assert scheduler._retention_affinity_candidates(head_job) == []
    assert scheduler.pending_inference_in_placement_order()[0] is head_job


async def test_a_job_of_a_held_teardown_is_still_seated_ahead_of_the_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A teardown that the card already holds is not a new one, and the head could not load under it anyway."""
    scheduler, head_job, run_job = await _run_with_a_cold_head(
        monkeypatch,
        rung=DominanceRung.CANNOT_CO_RESIDE,
        forecast=_NEEDS_THE_CARD,
    )

    promoted = [job for job, _lane in scheduler._retention_affinity_candidates(head_job)]

    assert promoted == [run_job]
