"""Lifecycle and entry-point contracts for exclusively-admitted over-budget jobs.

An exclusive admit suppresses the whole device: `JobTracker.has_exclusive_job_in_progress` caps
concurrent dispatch at the running job and holds every other model's preload
(`InferenceScheduler._attempt_preload_for_job`). A suppression that broad must provably release, or a
single job wedges the worker harder than the over-commit it was isolating. The release is structural
(the flag is only consulted for jobs in ``PENDING_INFERENCE`` / ``INFERENCE_IN_PROGRESS``), so these
tests pin the stage transitions that carry it:

- a terminal fault or a completed generation releases the suppression immediately;
- a retryable fault keeps it, because the bounded degraded retry of an over-budget job is meant to
  re-run isolated (see ``TrackedJob.admitted_exclusive``), and the job re-enters the covered stages;
- the exclusive job's own preload passes the hold (the exemption in ``_attempt_preload_for_job``), so
  the job that owns the device can always stage its weights onto it.

The direct over-budget tag and log contract is also pinned here: it reports the decided isolation, the signal
triage tooling and operators read, not the configured mode.
"""

from __future__ import annotations

from unittest.mock import Mock

from loguru import logger

from horde_worker_regen.process_management.jobs.job_tracker import (
    InferenceFailureResolution,
    JobTracker,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.governance.preload_admission import AdmissionDecision
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_CARD_LIGHT_MODEL = "AlbedoBase XL (SDXL)"
_CARD_DEMANDING_MODEL = "Flux.1-Schnell fp8 (Compact)"
_SIBLING_MODEL = "WAI-NSFW-illustrious-SDXL"


def _forecast(
    *,
    weights_mb: float,
    wants_whole_card: bool = False,
    total_vram_mb: float = 24074.0,
    free_if_alone_mb: float = 20801.0,
) -> StreamForecast:
    """A sized forecast; the footprint and room fields decide the admit's isolation verdict."""
    return StreamForecast(
        weights_mb=weights_mb,
        reserve_mb=2048.0,
        free_now_mb=10162.0,
        free_if_alone_mb=free_if_alone_mb,
        free_after_model_evict_mb=13819.0,
        total_vram_mb=total_vram_mb,
        per_process_overhead_mb=3273.0,
        marginal_process_overhead_mb=1746.0,
        wants_whole_card=wants_whole_card,
    )


def _tight_card_demanding_forecast() -> StreamForecast:
    """A card-dominating checkpoint on a card too small to host a sibling model beside it."""
    return _forecast(
        weights_mb=11900.0,
        wants_whole_card=True,
        total_vram_mb=16375.0,
        free_if_alone_mb=13500.0,
    )


def _make_scheduler(job_tracker: JobTracker):  # noqa: ANN202
    """A two-slot scheduler with the exclusive over-budget mode enabled (the default)."""
    process_map = ProcessMap({1: make_mock_process_info(1, model_name=None), 2: make_mock_process_info(2)})
    return _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(max_threads=2, overbudget_exclusive_mode=True),
        max_concurrent=2,
        max_inference=2,
    )


async def _exclusive_job_in_progress(job_tracker: JobTracker, model: str = _CARD_DEMANDING_MODEL):  # noqa: ANN202
    """Pop a job, start it, and mark it exclusively admitted (device to itself)."""
    job = make_job_pop_response(model=model)
    await job_tracker.record_popped_job(job)
    await job_tracker.mark_inference_started(job)
    job_tracker.mark_admitted_over_budget(job)
    job_tracker.mark_admitted_exclusive(job)
    return job


class TestExclusivityRelease:
    """The device-wide suppression ends exactly when the exclusive job leaves the covered stages."""

    async def test_terminal_fault_releases_suppression(self, job_tracker: JobTracker) -> None:
        """A terminally-faulted exclusive job stops capping dispatch the moment it faults.

        The suppression outliving its job would serialize the worker on a job that no longer exists,
        a harder wedge than the over-commit exclusivity guards against.
        """
        job_tracker.set_retry_policy(1)
        scheduler = _make_scheduler(job_tracker)
        job = await _exclusive_job_in_progress(job_tracker)
        assert scheduler._max_jobs_in_progress_allowed() == 1

        resolution = job_tracker.handle_job_fault_now(
            faulted_job=job,
            process_info=make_mock_process_info(1, model_name=_CARD_DEMANDING_MODEL),
        )

        assert resolution is InferenceFailureResolution.FAULTED
        assert job_tracker.has_exclusive_job_in_progress() is False
        assert scheduler._max_jobs_in_progress_allowed() == 2

    async def test_retryable_fault_keeps_isolation_for_the_retry(self, job_tracker: JobTracker) -> None:
        """A retryable fault re-queues the job still exclusive, so the bounded retry re-runs isolated.

        The retry classification exists because isolating the job is the remedy being retried; dropping
        exclusivity on requeue would re-run it contended, reproducing the failure it is retrying from.
        """
        job_tracker.set_retry_policy(3)
        job = await _exclusive_job_in_progress(job_tracker)

        resolution = job_tracker.handle_job_fault_now(
            faulted_job=job,
            process_info=make_mock_process_info(1, model_name=_CARD_DEMANDING_MODEL),
        )

        assert resolution is not InferenceFailureResolution.FAULTED
        assert job_tracker.has_exclusive_job_in_progress() is True

    async def test_completed_generation_releases_suppression(self, job_tracker: JobTracker) -> None:
        """A generation handed to safety leaves the inference stages, ending the suppression."""
        job = await _exclusive_job_in_progress(job_tracker)
        job_info = Mock()
        job_info.sdk_api_job_info = job

        await job_tracker.queue_for_safety(job_info)

        assert job_tracker.has_exclusive_job_in_progress() is False


class TestExclusivePreloadGate:
    """The preload hold blocks other models' staging but never the exclusive job's own load."""

    async def test_sibling_model_preload_is_held(self, job_tracker: JobTracker) -> None:
        """Another model's preload is refused while an exclusive job holds the device."""
        scheduler = _make_scheduler(job_tracker)
        await _exclusive_job_in_progress(job_tracker)
        sibling_job = make_job_pop_response(model=_SIBLING_MODEL)
        await job_tracker.record_popped_job(sibling_job)

        scheduler._attempt_preload_for_job(sibling_job, head_job=sibling_job, loaded_models=set())

        assert scheduler._last_preload_admission is not None
        assert scheduler._last_preload_admission.decision is AdmissionDecision.EXCLUSIVE_IN_PROGRESS

    async def test_exclusive_jobs_own_preload_passes_the_hold(self, job_tracker: JobTracker) -> None:
        """The exclusive job's own preload proceeds past the hold; the device is being held for it."""
        scheduler = _make_scheduler(job_tracker)
        exclusive_job = await _exclusive_job_in_progress(job_tracker)

        scheduler._attempt_preload_for_job(exclusive_job, head_job=exclusive_job, loaded_models=set())

        assert scheduler._last_preload_admission is not None
        assert scheduler._last_preload_admission.decision is not AdmissionDecision.EXCLUSIVE_IN_PROGRESS


class TestOverbudgetAdmitLogContract:
    """The admit warning reports the decided isolation, which operators and log triage read."""

    async def _admit_log_line(self, job_tracker: JobTracker, *, forecast: StreamForecast) -> str:
        """Mark an over-budget admit under ``forecast`` and capture the admit warning's message."""
        scheduler = _make_scheduler(job_tracker)
        job = make_job_pop_response(model=_CARD_LIGHT_MODEL)
        await job_tracker.record_popped_job(job)
        await job_tracker.mark_inference_started(job)
        scheduler._mark_overbudget_admit(job, forecast)

        lines: list[str] = []
        sink_id = logger.add(lambda message: lines.append(str(message)), format="{message}", colorize=False)
        try:
            scheduler._log_overbudget_admit(job)
        finally:
            logger.remove(sink_id)
        assert len(lines) == 1
        return lines[0]

    async def test_shared_admit_logs_shared(self, job_tracker: JobTracker) -> None:
        """A non-isolated admit says 'shared': the line must reflect the decision, not the config."""
        line = await self._admit_log_line(job_tracker, forecast=_forecast(weights_mb=4900.0))
        assert "(shared," in line
        assert "(exclusive," not in line

    async def test_exclusive_admit_logs_exclusive(self, job_tracker: JobTracker) -> None:
        """CONTROL: an isolated admit still says 'exclusive'."""
        line = await self._admit_log_line(job_tracker, forecast=_tight_card_demanding_forecast())
        assert "(exclusive," in line
