"""Lifecycle and entry-point contracts for exclusively-admitted over-budget jobs.

An exclusive admit suppresses the whole device: `JobTracker.has_exclusive_job_in_progress` caps
concurrent dispatch at the running job and holds every other model's preload
(`admission.preload.decide_preload_gates`). A suppression that broad must provably release, or a
single job wedges the worker harder than the over-commit it was isolating. The release is structural
(the flag is only consulted for jobs in ``PENDING_INFERENCE`` / ``INFERENCE_IN_PROGRESS``), so these
tests pin the stage transitions that carry it:

- a terminal fault or a completed generation releases the suppression immediately;
- a retryable fault keeps it, because the bounded degraded retry of an over-budget job is meant to
  re-run isolated (see ``TrackedJob.admitted_exclusive``), and the job re-enters the covered stages;
- the exclusive job's own preload passes the hold (the exemption in ``decide_preload_gates``), so
  the job that owns the device can always stage its weights onto it.

The direct over-budget tag and log contract is also pinned here: it reports the decided isolation, the signal
triage tooling and operators read, not the configured mode.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from loguru import logger

from horde_worker_regen.process_management.jobs.job_tracker import (
    InferenceFailureResolution,
    JobTracker,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.admission.preload import decide_preload_gates
from horde_worker_regen.process_management.scheduling.governance.preload_admission import AdmissionDecision
from horde_worker_regen.process_management.scheduling.governance.whole_card import WHOLE_CARD_ESTABLISH_GRACE_SECONDS
from horde_worker_regen.process_management.scheduling.ledgers.dispatch_holds import HoldRemedy
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

        job_id = str(sibling_job.id_)
        plan = decide_preload_gates(scheduler.snapshot(), job_id, head_job_id=job_id, loaded_models=frozenset())

        assert plan.decision is AdmissionDecision.EXCLUSIVE_IN_PROGRESS

    async def test_exclusive_jobs_own_preload_passes_the_hold(self, job_tracker: JobTracker) -> None:
        """The exclusive job's own preload proceeds past the hold; the device is being held for it."""
        scheduler = _make_scheduler(job_tracker)
        exclusive_job = await _exclusive_job_in_progress(job_tracker)

        job_id = str(exclusive_job.id_)
        plan = decide_preload_gates(scheduler.snapshot(), job_id, head_job_id=job_id, loaded_models=frozenset())

        assert plan.decision is not AdmissionDecision.EXCLUSIVE_IN_PROGRESS


class TestNeverConvergeReleaseClearsPendingAdmits:
    """A residency released as unable to converge takes the pending exclusive claims for its model with it."""

    async def test_other_model_head_preload_admitted_after_release(self, job_tracker: JobTracker) -> None:
        """The other-model head may stage once the residency its blocker was waiting on is released.

        The pending admit's claim on the card was for that residency. Left marked, it keeps declining every
        other head's preload with nothing on the card to protect, so the queue can neither drain nor converge.
        """
        scheduler = _make_scheduler(job_tracker)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=2)
        head_job = make_job_pop_response(model=_SIBLING_MODEL)
        await job_tracker.record_popped_job(head_job)
        heavy_job = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(heavy_job)
        job_tracker.mark_admitted_over_budget(heavy_job)
        job_tracker.mark_admitted_exclusive(heavy_job, device_index=None)
        scheduler._whole_card_ledger.record_grant(
            None,
            model=_CARD_DEMANDING_MODEL,
            forecast=None,
            cooldown_until=0.0,
            now=time.time() - 600.0,
        )

        scheduler._restore_siblings_after_whole_card()

        assert scheduler._whole_card_ledger.state_for(None).model is None
        job_id = str(head_job.id_)
        plan = decide_preload_gates(scheduler.snapshot(), job_id, head_job_id=job_id, loaded_models=frozenset())
        assert plan.decision is not AdmissionDecision.EXCLUSIVE_IN_PROGRESS
        assert job_tracker.is_admitted_exclusive(heavy_job) is False

    async def test_in_progress_admit_keeps_its_mark(self, job_tracker: JobTracker) -> None:
        """Only pending admits lose the mark: a sampling job's footprint is on the card whatever the ledger says."""
        running_job = await _exclusive_job_in_progress(job_tracker)
        pending_job = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(pending_job)
        job_tracker.mark_admitted_exclusive(pending_job)

        job_tracker.release_pending_exclusive_admits(_CARD_DEMANDING_MODEL, device_index=None)

        assert job_tracker.is_admitted_exclusive(running_job) is True
        assert job_tracker.is_admitted_exclusive(pending_job) is False


def _establish_grace_active(scheduler) -> bool:  # noqa: ANN001
    """Whether the card's establish grace is open, read apart from the restore window the release may open.

    The restore window covers the respawns and lane restarts reported by the mocked lifecycle.
    """
    return scheduler._whole_card_ledger.grace_active(
        now=scheduler._clock(),
        establish_grace_seconds=WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        restore_grace_seconds=0.0,
    )


class TestHoldWithNoRemedyReleasesResidency:
    """A dispatch hold that cannot seat a residency's head releases that residency and its establish grace.

    The dispatch gate prices the head only once the teardown has done all it can, so a hold with no remedy, or
    with a reroute to another card, means the residency cannot seat its head. Kept, the residency's establish grace
    would defer recovery for up to 120 s after the hold itself stopped deferring it.
    """

    @staticmethod
    async def _scheduler_with_fresh_residency(job_tracker: JobTracker):  # noqa: ANN205
        """A scheduler whose card holds a residency established now for a pending exclusive head."""
        scheduler = _make_scheduler(job_tracker)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=2)
        head_job = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(head_job)
        job_tracker.mark_admitted_over_budget(head_job)
        job_tracker.mark_admitted_exclusive(head_job, device_index=None)
        now = scheduler._clock()
        scheduler._whole_card_ledger.record_grant(
            None,
            model=_CARD_DEMANDING_MODEL,
            forecast=None,
            cooldown_until=now + 600.0,
            now=now,
            establish_grace_seconds=WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        )
        assert scheduler.whole_card_residency_grace_active(), "precondition: the establish grace is open"
        return scheduler, head_job

    async def test_no_remedy_hold_on_the_head_releases_the_residency(self, job_tracker: JobTracker) -> None:
        """A hold with no remedy on the residency's head releases it inside its grace and its cooldown."""
        scheduler, head_job = await self._scheduler_with_fresh_residency(job_tracker)

        scheduler._note_dispatch_hold(head_job, reclaim_applied=False, remedy=HoldRemedy.NONE)
        scheduler._restore_siblings_after_whole_card()

        assert scheduler._whole_card_ledger.state_for(None).model is None
        assert not _establish_grace_active(scheduler)
        assert job_tracker.is_admitted_exclusive(head_job) is False

    async def test_reroute_hold_on_the_head_releases_the_residency(self, job_tracker: JobTracker) -> None:
        """A reroute takes the head to another card, so the residency it leaves is released as well."""
        scheduler, head_job = await self._scheduler_with_fresh_residency(job_tracker)

        scheduler._note_dispatch_hold(head_job, reclaim_applied=False, remedy=HoldRemedy.REROUTE)
        scheduler._restore_siblings_after_whole_card()

        assert scheduler._whole_card_ledger.state_for(None).model is None
        assert not _establish_grace_active(scheduler)

    async def test_hold_with_a_live_remedy_leaves_the_residency(self, job_tracker: JobTracker) -> None:
        """A hold whose reclaim can still seat the head keeps the residency and its grace."""
        scheduler, head_job = await self._scheduler_with_fresh_residency(job_tracker)

        scheduler._note_dispatch_hold(head_job, reclaim_applied=True, remedy=HoldRemedy.RECLAIM)
        scheduler._restore_siblings_after_whole_card()

        assert scheduler._whole_card_ledger.state_for(None).model == _CARD_DEMANDING_MODEL
        assert _establish_grace_active(scheduler)
        assert job_tracker.is_admitted_exclusive(head_job) is True

    async def test_no_remedy_hold_on_another_job_leaves_the_residency(self, job_tracker: JobTracker) -> None:
        """A hold on a job the residency was not granted for leaves the residency and its grace standing."""
        scheduler, _head_job = await self._scheduler_with_fresh_residency(job_tracker)
        same_model_job = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(same_model_job)
        sibling_job = make_job_pop_response(model=_SIBLING_MODEL)
        await job_tracker.record_popped_job(sibling_job)

        scheduler._note_dispatch_hold(same_model_job, reclaim_applied=False, remedy=HoldRemedy.NONE)
        scheduler._note_dispatch_hold(sibling_job, reclaim_applied=False, remedy=HoldRemedy.NONE)
        scheduler._restore_siblings_after_whole_card()

        assert scheduler._whole_card_ledger.state_for(None).model == _CARD_DEMANDING_MODEL
        assert _establish_grace_active(scheduler)


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
