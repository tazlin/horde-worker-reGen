"""Exclusivity scope for over-budget best-effort admits.

An over-budget best-effort admit exists to un-wedge the head of the queue when the VRAM budget cannot fit
it even after reclaiming every idle resident copy. Running that admit with the device to itself
(``overbudget_exclusive_mode``) protects a genuinely card-dominating model from a concurrent sibling load
pushing its weights into host-RAM streaming. That protection is wasted on a card-light model: a moderate
checkpoint (an SDXL on a high-VRAM card) can reach the force-admit path purely through reserve arithmetic
on a device whose free VRAM is depressed by retained sibling contexts, and it samples correctly alongside
a sibling. Marking such an admit exclusive suppresses every other preload and caps dispatch at one job
from admit through completion, serializing a multi-thread card for the full duration of an ordinary job.

The contract pinned here: exclusivity attaches to an over-budget admit only when the forecast shows the
model's persistent footprint dominates the device (the same ``is_card_demanding`` trust test the
whole-card residency machinery uses). Card-light over-budget admits keep the ``admitted_over_budget``
classification (the widened step grace and the resource-fault retry accounting) but share the device.
An unsized forecast stays exclusive: without a footprint measurement the conservative direction is
isolation.
"""

from __future__ import annotations

from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_CARD_TOTAL_MB = 24074.0

_CARD_LIGHT_MODEL = "AlbedoBase XL (SDXL)"
_CARD_DEMANDING_MODEL = "Flux.1-Schnell fp8 (Compact)"


def _make_forecast(
    *,
    weights_mb: float | None,
    total_vram_mb: float | None = _CARD_TOTAL_MB,
    wants_whole_card: bool = False,
) -> StreamForecast:
    """Build a forecast whose footprint fields drive ``is_card_demanding`` deterministically."""
    return StreamForecast(
        weights_mb=weights_mb,
        reserve_mb=2048.0,
        free_now_mb=10162.0,
        free_if_alone_mb=20801.0,
        free_after_model_evict_mb=13819.0,
        total_vram_mb=total_vram_mb,
        per_process_overhead_mb=3273.0,
        marginal_process_overhead_mb=1746.0,
        wants_whole_card=wants_whole_card,
    )


def _card_light_forecast() -> StreamForecast:
    """A moderate SDXL on a 24GB card: weights + reserve well under the card-demanding fraction."""
    forecast = _make_forecast(weights_mb=4900.0)
    assert forecast.is_card_demanding is False
    return forecast


def _card_demanding_forecast() -> StreamForecast:
    """A combined checkpoint whose footprint dominates the card."""
    forecast = _make_forecast(weights_mb=11900.0, wants_whole_card=True)
    assert forecast.is_card_demanding is True
    return forecast


def _make_scheduler(job_tracker: JobTracker, *, overbudget_exclusive_mode: bool = True):  # noqa: ANN202
    """Build a two-slot scheduler with the exclusive-admit config under test."""
    process_map = ProcessMap({1: make_mock_process_info(1), 2: make_mock_process_info(2)})
    return _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(max_threads=2, overbudget_exclusive_mode=overbudget_exclusive_mode),
        max_concurrent=2,
        max_inference=2,
    )


class TestOverbudgetExclusiveScope:
    """Exclusivity on an over-budget admit follows the footprint, not the config flag alone."""

    async def test_card_light_admit_is_not_exclusive(self, job_tracker: JobTracker) -> None:
        """A card-light model admitted over budget shares the device.

        Suppressing the sibling slot for an ordinary SDXL admit halves a two-thread card's throughput for
        the job's whole pop-to-submit lifetime; the streaming risk exclusivity guards against needs a
        footprint that actually dominates the card.
        """
        scheduler = _make_scheduler(job_tracker)
        job = make_job_pop_response(model=_CARD_LIGHT_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)

        scheduler._mark_overbudget_admit(job, _card_light_forecast())

        assert job_tracker.is_admitted_over_budget(job) is True
        assert job_tracker.is_admitted_exclusive(job) is False
        assert job_tracker.has_exclusive_job_in_progress() is False
        assert scheduler._max_jobs_in_progress_allowed(0) == 2

    async def test_card_light_admit_keeps_overbudget_classification(self, job_tracker: JobTracker) -> None:
        """Sharing the device does not forfeit the over-budget tags (step grace, resource-fault retries)."""
        scheduler = _make_scheduler(job_tracker)
        job = make_job_pop_response(model=_CARD_LIGHT_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)

        scheduler._mark_overbudget_admit(job, _card_light_forecast())

        assert job_tracker.is_admitted_over_budget(job) is True
        assert scheduler.heavy_head_load_grace_active() is True

    async def test_card_demanding_admit_stays_exclusive(self, job_tracker: JobTracker) -> None:
        """CONTROL: a card-dominating model keeps the device to itself.

        This is the case the exclusive mode exists for: a concurrent sibling load would push the heavy
        model's weights into host-RAM streaming and collapse its step rate.
        """
        scheduler = _make_scheduler(job_tracker)
        job = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)

        scheduler._mark_overbudget_admit(job, _card_demanding_forecast())

        assert job_tracker.is_admitted_exclusive(job) is True
        assert scheduler._max_jobs_in_progress_allowed(0) == 1

    async def test_unsized_forecast_stays_exclusive(self, job_tracker: JobTracker) -> None:
        """CONTROL: a forecast that cannot size the footprint keeps the conservative isolation."""
        scheduler = _make_scheduler(job_tracker)
        job = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)

        scheduler._mark_overbudget_admit(job, _make_forecast(weights_mb=None))

        assert job_tracker.is_admitted_exclusive(job) is True

    async def test_missing_forecast_stays_exclusive(self, job_tracker: JobTracker) -> None:
        """CONTROL: an admit path with no forecast at hand keeps the conservative isolation."""
        scheduler = _make_scheduler(job_tracker)
        job = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)

        scheduler._mark_overbudget_admit(job, None)

        assert job_tracker.is_admitted_exclusive(job) is True

    async def test_exclusive_mode_off_marks_nothing_exclusive(self, job_tracker: JobTracker) -> None:
        """CONTROL: with the mode disabled even a card-dominating admit shares (operator override)."""
        scheduler = _make_scheduler(job_tracker, overbudget_exclusive_mode=False)
        job = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)

        scheduler._mark_overbudget_admit(job, _card_demanding_forecast())

        assert job_tracker.is_admitted_exclusive(job) is False
