"""The overlap cap withholds a fresh sample from a card already owed a large post-processing peak.

With the GPU sampling lease, ``_max_jobs_in_progress_allowed`` pre-stages a second concurrent job up to the
process ceiling whenever free VRAM clears the staging floor. That floor counts only realised
post-processing peaks, not the *imminent* peak of a job that is still sampling, so a second sample can be
staged beside a job whose upscaler is about to claim several gigabytes, over-committing the card mid-flight.
These assert the imminent peak gates the cap: a job still sampling that will post-process withholds the
second concurrent sample, and the gate self-scales away when nothing in flight will post-process.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling import inference_scheduler as inference_scheduler_module
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MODEL = "WAI-NSFW-illustrious-SDXL"

_UPSCALE_PEAK_MB = 8533.0
# Free VRAM that clears the staging floor on its own, but not once the imminent upscale peak is held back.
_FREE_VRAM_MB = 5000.0
_MAX_INFERENCE = 4
_MAX_CONCURRENT = 1


async def _scheduler_with_inflight_sampling_job(
    *,
    monkeypatch: pytest.MonkeyPatch,
    in_flight_peak_mb: float,
) -> InferenceScheduler:
    """A lease-enabled, budget-active scheduler with one in-flight job sampling on process 0.

    ``in_flight_peak_mb`` is the post-processing peak that job reports (0 for a job that does no upscaling),
    so a test can toggle whether an imminent peak is owed.
    """
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        post_processing_budget_reserve_enabled=True,
        gpu_sampling_lease_enabled=True,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
    )
    job_tracker = JobTracker()
    in_flight_job = make_job_pop_response(model=_MODEL)
    await track_popped_job_async(job_tracker, in_flight_job)
    await job_tracker.mark_inference_started(in_flight_job)

    sampling_process = make_mock_process_info(
        process_id=0,
        model_name=_MODEL,
        state=HordeProcessState.INFERENCE_STARTING,
    )
    sampling_process.last_job_referenced = in_flight_job
    process_map = ProcessMap({0: sampling_process})

    scheduler = _make_inference_scheduler(
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        process_map=process_map,
        max_concurrent=_MAX_CONCURRENT,
        max_inference=_MAX_INFERENCE,
    )
    scheduler._process_map.get_free_vram_mb = Mock(return_value=_FREE_VRAM_MB)  # type: ignore[method-assign]

    def _fake_peak(job: object, baseline: str | None) -> float:
        return in_flight_peak_mb if getattr(job, "model", None) == _MODEL else 0.0

    monkeypatch.setattr(inference_scheduler_module, "predict_job_post_processing_vram_mb", _fake_peak)
    return scheduler


async def test_imminent_peak_withholds_the_overlap_prestage(monkeypatch: pytest.MonkeyPatch) -> None:
    """An in-flight sampling job owed a large upscale peak drops the cap to the sampling-slot ceiling."""
    scheduler = await _scheduler_with_inflight_sampling_job(
        monkeypatch=monkeypatch, in_flight_peak_mb=_UPSCALE_PEAK_MB
    )

    # Free (5 GB) clears the staging floor on its own, but free - imminent (5 - 8.5 GB) does not, so no second
    # concurrent sample is pre-staged: the cap stays at the concurrent-sampling ceiling.
    assert scheduler._max_jobs_in_progress_allowed(0) == _MAX_CONCURRENT


async def test_no_imminent_peak_allows_the_overlap_prestage(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing in flight owed a post-processing peak, pre-staging is allowed up to the process ceiling."""
    scheduler = await _scheduler_with_inflight_sampling_job(monkeypatch=monkeypatch, in_flight_peak_mb=0.0)

    assert scheduler._max_jobs_in_progress_allowed(0) == _MAX_INFERENCE
