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
  the job that owns the device can always stage its weights onto it;
- the preload pass goes on over a card held for an exclusive admit and reaches that admit;
- a residency that can never converge is released without waiting for the cooldown;
- a pending mark whose card does not hold a residency for the marked model is released.

The direct over-budget tag and log contract is also pinned here: it reports the decided isolation, the signal
triage tooling and operators read, not the configured mode.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from enum import auto
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger
from strenum import StrEnum

from horde_worker_regen.process_management.ipc.messages import HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import (
    InferenceFailureResolution,
    JobTracker,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling import inference_scheduler as inference_scheduler_module
from horde_worker_regen.process_management.scheduling.admission.preload import (
    PreloadGatePlan,
    PreloadPassStep,
    decide_preload_gates,
    pass_control_for,
)
from horde_worker_regen.process_management.scheduling.admission.snapshot import SchedulingSnapshot
from horde_worker_regen.process_management.scheduling.governance.preload_admission import AdmissionDecision
from horde_worker_regen.process_management.scheduling.governance.whole_card import WHOLE_CARD_ESTABLISH_GRACE_SECONDS
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.ledgers.dispatch_holds import HoldRemedy
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
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


def _make_scheduler(  # noqa: ANN202
    job_tracker: JobTracker,
    *,
    clock: Callable[[], float] | None = None,
    cooldown_seconds: int = 0,
):
    """A two-slot scheduler with the exclusive over-budget mode enabled (the default)."""
    process_map = ProcessMap({1: make_mock_process_info(1, model_name=None), 2: make_mock_process_info(2)})
    return _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(
            max_threads=2,
            overbudget_exclusive_mode=True,
            whole_card_residency_cooldown_seconds=cooldown_seconds,
        ),
        max_concurrent=2,
        max_inference=2,
        clock=clock,
    )


class _ResidencyRoute(StrEnum):
    """A healthy route from a held residency to running the model."""

    HOLDER = auto()
    LOADING = auto()
    PRESTAGING = auto()
    SAMPLING = auto()


class _ManualClock:
    """A hand-moved clock, used as the scheduler's time source."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


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


class TestNeverConvergeReleaseSkipsCooldown:
    """A residency that can never converge is released at the establish grace, whatever the cooldown reads.

    While the residency's model is pending, the residency reads as serving and each pass refreshes the cooldown.
    The cooldown then stands a full period past the moment the residency is found unable to converge. Waiting it
    out keeps the card torn down for a model that has lost the route to the card.
    """

    @staticmethod
    async def _scheduler_with_unconvergeable_residency(
        job_tracker: JobTracker,
    ) -> tuple[InferenceScheduler, _ManualClock, ImageGenerateJobPopResponse, float]:
        """A residency granted for a pending exclusive admit while a different-model head waits.

        The model is not loaded, loading or pre-staged on any lane.
        """
        clock = _ManualClock(10_000.0)
        scheduler = _make_scheduler(job_tracker, clock=clock, cooldown_seconds=45)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=2)
        await job_tracker.record_popped_job(make_job_pop_response(model=_SIBLING_MODEL))
        heavy_job = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(heavy_job)
        job_tracker.mark_admitted_over_budget(heavy_job)
        job_tracker.mark_admitted_exclusive(heavy_job, device_index=None)
        granted_at = clock.now
        scheduler._whole_card_ledger.record_grant(
            None,
            model=_CARD_DEMANDING_MODEL,
            forecast=None,
            cooldown_until=granted_at + 45.0,
            now=granted_at,
            establish_grace_seconds=WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        )
        return scheduler, clock, heavy_job, granted_at

    async def test_release_at_the_establish_grace_does_not_wait_for_a_refreshed_cooldown(
        self,
        job_tracker: JobTracker,
    ) -> None:
        """The pass just inside the grace refreshes the cooldown, and the first pass past it releases the card."""
        scheduler, clock, heavy_job, granted_at = await self._scheduler_with_unconvergeable_residency(job_tracker)

        clock.now = granted_at + WHOLE_CARD_ESTABLISH_GRACE_SECONDS - 1.0
        scheduler._restore_siblings_after_whole_card()
        state = scheduler._whole_card_ledger.state_for(None)
        assert state.model == _CARD_DEMANDING_MODEL, "precondition: inside the grace the residency is kept"
        refreshed_cooldown_until = state.cooldown_until
        assert refreshed_cooldown_until > granted_at + WHOLE_CARD_ESTABLISH_GRACE_SECONDS + 30.0, (
            "precondition: the serving pass refreshes the cooldown"
        )

        clock.now = granted_at + WHOLE_CARD_ESTABLISH_GRACE_SECONDS + 1.0
        scheduler._restore_siblings_after_whole_card()

        assert clock.now < refreshed_cooldown_until
        assert scheduler._whole_card_ledger.state_for(None).model is None
        assert job_tracker.is_admitted_exclusive(heavy_job) is False

    async def test_a_drained_residency_still_honours_its_cooldown(self, job_tracker: JobTracker) -> None:
        """A residency whose work has drained is held through the cooldown while a head is not waiting."""
        clock = _ManualClock(10_000.0)
        scheduler = _make_scheduler(job_tracker, clock=clock, cooldown_seconds=45)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=2)
        granted_at = clock.now
        scheduler._whole_card_ledger.record_grant(
            None,
            model=_CARD_DEMANDING_MODEL,
            forecast=None,
            cooldown_until=granted_at + WHOLE_CARD_ESTABLISH_GRACE_SECONDS + 45.0,
            now=granted_at,
            establish_grace_seconds=WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        )

        clock.now = granted_at + WHOLE_CARD_ESTABLISH_GRACE_SECONDS + 1.0
        scheduler._restore_siblings_after_whole_card()

        assert scheduler._whole_card_ledger.state_for(None).model == _CARD_DEMANDING_MODEL

    async def test_a_drained_residency_still_honours_its_minimum_hold(self, job_tracker: JobTracker) -> None:
        """A ready different-model head does not cut a fresh residency's cooldown short inside the minimum hold."""
        clock = _ManualClock(10_000.0)
        scheduler = _make_scheduler(job_tracker, clock=clock, cooldown_seconds=45)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=2)
        ready_head = make_job_pop_response(model="stable_diffusion")
        await job_tracker.record_popped_job(ready_head)
        granted_at = clock.now
        scheduler._whole_card_ledger.record_grant(
            None,
            model=_CARD_DEMANDING_MODEL,
            forecast=None,
            cooldown_until=granted_at + 45.0,
            now=granted_at,
            establish_grace_seconds=WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        )
        assert scheduler._ready_different_model_head_on_device(
            residency_model=_CARD_DEMANDING_MODEL,
            device_index=None,
        ), "precondition: the head is ready on a resident lane"

        clock.now = granted_at + 10.0
        scheduler._restore_siblings_after_whole_card()

        assert scheduler._whole_card_ledger.state_for(None).model == _CARD_DEMANDING_MODEL

    @pytest.mark.parametrize("route", list(_ResidencyRoute))
    async def test_a_residency_with_a_route_to_its_model_is_kept(
        self,
        job_tracker: JobTracker,
        route: _ResidencyRoute,
    ) -> None:
        """A residency whose model is held, loading, pre-staging or sampling is kept past the grace.

        These are the healthy states of a residency before the model runs. The cooldown bypass does not apply to a
        residency in any of them.
        """
        scheduler, clock, heavy_job, granted_at = await self._scheduler_with_unconvergeable_residency(job_tracker)
        lane = scheduler._process_map[1]
        if route is _ResidencyRoute.HOLDER:
            lane.loaded_horde_model_name = _CARD_DEMANDING_MODEL
        elif route is _ResidencyRoute.LOADING:
            scheduler._horde_model_map.update_entry(
                _CARD_DEMANDING_MODEL,
                load_state=ModelLoadState.LOADING,
                process_id=lane.process_id,
            )
        elif route is _ResidencyRoute.PRESTAGING:
            lane.last_process_state = HordeProcessState.PRELOADING_MODEL
            scheduler._whole_card_ledger.state_for(None).prestage_process_id = lane.process_id
        else:
            running_job = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
            await job_tracker.record_popped_job(running_job)
            await job_tracker.mark_inference_started(running_job)
            job_tracker.mark_admitted_exclusive(running_job, device_index=None)

        clock.now = granted_at + WHOLE_CARD_ESTABLISH_GRACE_SECONDS + 1.0
        scheduler._restore_siblings_after_whole_card()

        assert scheduler._whole_card_ledger.state_for(None).model == _CARD_DEMANDING_MODEL
        assert job_tracker.is_admitted_exclusive(heavy_job) is True

    async def test_an_exclusive_job_in_progress_on_another_card_keeps_its_mark(self, job_tracker: JobTracker) -> None:
        """The release clears the pending admits for the released card and leaves a running job's exclusive mark.

        The running job's footprint is on the other card, which the release does not touch.
        """
        clock = _ManualClock(10_000.0)
        scheduler = _make_two_card_scheduler(job_tracker, clock=clock, cooldown_seconds=45)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=2)
        await job_tracker.record_popped_job(make_job_pop_response(model=_SIBLING_MODEL))
        pending_admit = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(pending_admit)
        job_tracker.mark_admitted_exclusive(pending_admit, device_index=0)
        running_admit = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(running_admit)
        await job_tracker.mark_inference_started(running_admit, device_index=1)
        job_tracker.mark_admitted_exclusive(running_admit, device_index=1)
        granted_at = clock.now
        scheduler._whole_card_ledger.record_grant(
            0,
            model=_CARD_DEMANDING_MODEL,
            forecast=None,
            cooldown_until=granted_at + 600.0,
            now=granted_at,
            establish_grace_seconds=WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        )

        clock.now = granted_at + WHOLE_CARD_ESTABLISH_GRACE_SECONDS + 1.0
        scheduler._restore_siblings_after_whole_card()

        assert scheduler._whole_card_ledger.state_for(0).model is None
        assert job_tracker.is_admitted_exclusive(pending_admit) is False
        assert job_tracker.is_admitted_exclusive(running_admit) is True


def _make_two_card_scheduler(  # noqa: ANN202
    job_tracker: JobTracker,
    *,
    clock: Callable[[], float] | None = None,
    cooldown_seconds: int = 0,
):
    """A scheduler driving two cards with two empty lanes each: lanes 1 and 2 on card 0, lanes 3 and 4 on card 1."""
    bridge_data = make_mock_bridge_data(
        max_threads=4,
        overbudget_exclusive_mode=True,
        whole_card_residency_cooldown_seconds=cooldown_seconds,
    )
    return _make_inference_scheduler(
        process_map=ProcessMap(
            {
                process_id: make_mock_process_info(process_id, model_name=None, device_index=device_index)
                for process_id, device_index in ((1, 0), (2, 0), (3, 1), (4, 1))
            },
        ),
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        max_concurrent=4,
        max_inference=4,
        card_runtimes=make_test_card_runtimes(
            target_process_count=2,
            device_indices=(0, 1),
            max_concurrent_inference=2,
            config=bridge_data,
            mask_kind="cuda",
        ),
        clock=clock,
    )


class _ScriptedPreloadWalk:
    """Stand-ins for the gate ladder and the plan executor, so a test states each job's plan and reads the walk.

    The walk itself (the placement order, the stopped cards and the held cards) runs as production runs it.
    """

    def __init__(self, scheduler: InferenceScheduler, plans: dict[str, tuple[AdmissionDecision, int]]) -> None:
        self._scheduler = scheduler
        self._plans = plans
        self.visited: list[str] = []

    def decide(self, _snapshot: object, job_id: str, **_kwargs: object) -> PreloadGatePlan:
        decision, target_process_id = self._plans[job_id]
        return PreloadGatePlan(
            decision=decision, job_id=job_id, reason="scripted", target_process_id=target_process_id
        )

    def run(self, plan: PreloadGatePlan, _snapshot: object) -> PreloadPassStep:
        self.visited.append(plan.job_id)
        assert plan.target_process_id is not None
        process = self._scheduler._process_map.get(plan.target_process_id)
        assert process is not None
        return PreloadPassStep(
            control=pass_control_for(plan.decision),
            decision=plan.decision,
            device_index=self._scheduler._routing_device_index(process),
            is_head_blocker=False,
        )


def _run_scripted_walk(
    scheduler: InferenceScheduler,
    monkeypatch: pytest.MonkeyPatch,
    plans: dict[str, tuple[AdmissionDecision, int]],
    *,
    placement_order: tuple[str, ...] | None = None,
) -> tuple[bool, list[str]]:
    """Run one preload pass over the scripted plans and return whether a preload was sent, and the executed jobs.

    ``placement_order`` replaces the snapshot's placement order, standing in for a reorder that moves a job served
    by a retained or held copy ahead of the queue's first job.
    """
    walk = _ScriptedPreloadWalk(scheduler, plans)
    monkeypatch.setattr(inference_scheduler_module, "decide_preload_gates", walk.decide)
    monkeypatch.setattr(scheduler, "_run_preload_plan", walk.run)
    if placement_order is not None:
        build_snapshot = scheduler.snapshot

        def reordered_snapshot() -> SchedulingSnapshot:
            snapshot = build_snapshot()
            return dataclasses.replace(
                snapshot,
                queue=dataclasses.replace(snapshot.queue, pending_in_placement_order=placement_order),
            )

        monkeypatch.setattr(scheduler, "snapshot", reordered_snapshot)
    return scheduler.preload_models(), walk.visited


def _hold_card_for(
    scheduler: InferenceScheduler,
    job: ImageGenerateJobPopResponse,
    *,
    device_index: int | None,
) -> None:
    """Grant ``job``'s model a fresh residency on ``device_index`` and mark ``job`` as the card's exclusive admit."""
    scheduler._job_tracker.mark_admitted_exclusive(job, device_index=device_index)
    now = scheduler._clock()
    scheduler._whole_card_ledger.record_grant(
        device_index,
        model=job.model,
        forecast=None,
        cooldown_until=now + 45.0,
        now=now,
        establish_grace_seconds=WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
    )


async def _pop_all(job_tracker: JobTracker, jobs: list[ImageGenerateJobPopResponse]) -> list[str]:
    """Track each job in order and return the ids."""
    for job in jobs:
        await job_tracker.record_popped_job(job)
    return [str(job.id_) for job in jobs]


class TestPreloadWalkReachesTheExclusiveBeneficiary:
    """A refusal for an exclusive claim holds the target card for the claiming job, and the pass goes on to that job.

    A card can be granted to a whole-card head queued behind jobs waiting for auxiliary files. Once the files
    land, those jobs come first in placement order and are refused with ``EXCLUSIVE_IN_PROGRESS``. A pass that
    stops there does not reach the card's exclusive admit.
    """

    async def test_one_card_goes_on_to_the_beneficiary_past_jobs_refused_for_the_claim(
        self,
        job_tracker: JobTracker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The pass steps past every refused or unrelated job on the held card and sends the beneficiary's preload."""
        scheduler = _make_scheduler(job_tracker)
        beneficiary = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        refused, unrelated, beneficiary_id = await _pop_all(
            job_tracker,
            [make_job_pop_response(model=_SIBLING_MODEL), make_job_pop_response(model=_CARD_LIGHT_MODEL), beneficiary],
        )
        _hold_card_for(scheduler, beneficiary, device_index=None)

        preloaded, visited = _run_scripted_walk(
            scheduler,
            monkeypatch,
            {
                refused: (AdmissionDecision.EXCLUSIVE_IN_PROGRESS, 1),
                unrelated: (AdmissionDecision.ADMIT, 2),
                beneficiary_id: (AdmissionDecision.ADMIT, 1),
            },
        )

        assert preloaded is True
        assert visited == [refused, beneficiary_id]

    async def test_one_card_without_a_beneficiary_does_not_preload_after_the_refusal(
        self,
        job_tracker: JobTracker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A held card does not admit an unrelated load, so a pass without a beneficiary does not send a preload."""
        scheduler = _make_scheduler(job_tracker)
        refused, unrelated = await _pop_all(
            job_tracker,
            [make_job_pop_response(model=_SIBLING_MODEL), make_job_pop_response(model=_CARD_LIGHT_MODEL)],
        )

        preloaded, visited = _run_scripted_walk(
            scheduler,
            monkeypatch,
            {
                refused: (AdmissionDecision.EXCLUSIVE_IN_PROGRESS, 1),
                unrelated: (AdmissionDecision.ADMIT, 2),
            },
        )

        assert preloaded is False
        assert visited == [refused]

    async def test_other_stops_still_end_the_pass_on_one_card(
        self,
        job_tracker: JobTracker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stop of any other kind still ends a single-card pass before the beneficiary is reached."""
        scheduler = _make_scheduler(job_tracker)
        beneficiary = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        deferred, beneficiary_id = await _pop_all(
            job_tracker, [make_job_pop_response(model=_SIBLING_MODEL), beneficiary]
        )
        _hold_card_for(scheduler, beneficiary, device_index=None)

        preloaded, visited = _run_scripted_walk(
            scheduler,
            monkeypatch,
            {
                deferred: (AdmissionDecision.DEFER_CONCURRENCY, 1),
                beneficiary_id: (AdmissionDecision.ADMIT, 1),
            },
        )

        assert preloaded is False
        assert visited == [deferred]

    async def test_another_cards_work_progresses_beside_a_held_card(
        self,
        job_tracker: JobTracker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A job whose load lands on the other card is preloaded after a refusal on the held card."""
        scheduler = _make_two_card_scheduler(job_tracker)
        refused, other_card = await _pop_all(
            job_tracker,
            [make_job_pop_response(model=_SIBLING_MODEL), make_job_pop_response(model=_CARD_LIGHT_MODEL)],
        )

        preloaded, visited = _run_scripted_walk(
            scheduler,
            monkeypatch,
            {
                refused: (AdmissionDecision.EXCLUSIVE_IN_PROGRESS, 1),
                other_card: (AdmissionDecision.ADMIT, 3),
            },
        )

        assert preloaded is True
        assert visited == [refused, other_card]

    async def test_an_exclusive_admit_for_another_card_does_not_pass_the_held_card(
        self,
        job_tracker: JobTracker,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only the admit planned for the held card is preloaded there.

        An exclusive admit planned for the second card cannot use the hold on the first.
        """
        scheduler = _make_two_card_scheduler(job_tracker)
        other_cards_admit = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        beneficiary = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        refused, other_cards_admit_id, beneficiary_id = await _pop_all(
            job_tracker,
            [make_job_pop_response(model=_SIBLING_MODEL), other_cards_admit, beneficiary],
        )
        _hold_card_for(scheduler, other_cards_admit, device_index=1)
        _hold_card_for(scheduler, beneficiary, device_index=0)

        preloaded, visited = _run_scripted_walk(
            scheduler,
            monkeypatch,
            {
                refused: (AdmissionDecision.EXCLUSIVE_IN_PROGRESS, 1),
                other_cards_admit_id: (AdmissionDecision.ADMIT, 2),
                beneficiary_id: (AdmissionDecision.ADMIT, 2),
            },
        )

        assert preloaded is True
        assert visited == [refused, beneficiary_id]


class TestPreloadWalkReachesTheUndispatchedHeadBeneficiary:
    """The card's exclusive admit is the queue's first job, and a reorder places a refused job ahead of it.

    A residency whose model is the undispatched head never reads as unable to converge, so the pass is the only
    route to running the head. A two-card host stops the card at the refusal for every other stop kind.
    """

    @pytest.mark.parametrize("two_cards", [False, True], ids=["one_card", "two_cards"])
    async def test_the_pass_reaches_the_head_behind_a_refusal_placed_ahead_of_it(
        self,
        job_tracker: JobTracker,
        monkeypatch: pytest.MonkeyPatch,
        two_cards: bool,
    ) -> None:
        """The head is the undispatched head and the pass sends the head's preload after the refusal ahead of it."""
        scheduler = _make_two_card_scheduler(job_tracker) if two_cards else _make_scheduler(job_tracker)
        head = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        head_id, refused = await _pop_all(job_tracker, [head, make_job_pop_response(model=_SIBLING_MODEL)])
        _hold_card_for(scheduler, head, device_index=0 if two_cards else None)
        assert job_tracker.undispatched_head() == head

        preloaded, visited = _run_scripted_walk(
            scheduler,
            monkeypatch,
            {
                refused: (AdmissionDecision.EXCLUSIVE_IN_PROGRESS, 1),
                head_id: (AdmissionDecision.ADMIT, 2),
            },
            placement_order=(refused, head_id),
        )

        assert preloaded is True
        assert visited == [refused, head_id]


class TestExclusiveAdmitWithoutResidency:
    """A pending exclusive mark whose card does not hold a residency for the marked model is released on the next pass.

    A soft reset rebuilds the lanes and leaves the marks and the residency ledger as they were, so a job can stay
    marked for a card whose residency has ended. While the mark stands, every other preload onto that card is refused.
    """

    @staticmethod
    def _release_lines(scheduler: InferenceScheduler) -> list[str]:
        """Run one release pass and return the release lines it logged."""
        lines: list[str] = []
        sink_id = logger.add(lambda message: lines.append(str(message)), format="{message}", colorize=False)
        try:
            scheduler._restore_siblings_after_whole_card()
        finally:
            logger.remove(sink_id)
        return [line for line in lines if line.startswith("Released the pending exclusive admits")]

    async def test_a_mark_without_a_residency_is_released_once_and_the_card_admits_other_preloads(
        self,
        job_tracker: JobTracker,
    ) -> None:
        """The pass clears the mark and logs it once, and another job's preload onto the card is not refused."""
        scheduler = _make_scheduler(job_tracker)
        marked = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        other = make_job_pop_response(model=_SIBLING_MODEL)
        await _pop_all(job_tracker, [marked, other])
        job_tracker.mark_admitted_exclusive(marked, device_index=None)

        first_pass = self._release_lines(scheduler)
        second_pass = self._release_lines(scheduler)

        assert job_tracker.is_admitted_exclusive(marked) is False
        assert len(first_pass) == 1 and _CARD_DEMANDING_MODEL in first_pass[0]
        assert second_pass == []
        other_id = str(other.id_)
        plan = decide_preload_gates(scheduler.snapshot(), other_id, head_job_id=other_id, loaded_models=frozenset())
        assert plan.decision is not AdmissionDecision.EXCLUSIVE_IN_PROGRESS

    async def test_a_mark_with_its_residency_is_kept(self, job_tracker: JobTracker) -> None:
        """A residency held for the marked job's model on the admit's card keeps the mark."""
        scheduler = _make_scheduler(job_tracker)
        marked = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(marked)
        job_tracker.mark_admitted_exclusive(marked, device_index=None)
        now = scheduler._clock()
        scheduler._whole_card_ledger.record_grant(
            None,
            model=_CARD_DEMANDING_MODEL,
            forecast=None,
            cooldown_until=now + 45.0,
            now=now,
            establish_grace_seconds=WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        )

        assert self._release_lines(scheduler) == []
        assert job_tracker.is_admitted_exclusive(marked) is True

    async def test_an_over_budget_mark_is_kept(self, job_tracker: JobTracker) -> None:
        """An over-budget admit isolates the job for the job's whole life, so the pass leaves the mark."""
        scheduler = _make_scheduler(job_tracker)
        marked = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await job_tracker.record_popped_job(marked)
        job_tracker.mark_admitted_over_budget(marked)
        job_tracker.mark_admitted_exclusive(marked, device_index=None)

        assert self._release_lines(scheduler) == []
        assert job_tracker.is_admitted_exclusive(marked) is True

    async def test_another_cards_residency_keeps_only_an_unattributed_mark(self, job_tracker: JobTracker) -> None:
        """A mark without a card is kept while any card holds the model, and a mark for card 0 is released."""
        scheduler = _make_two_card_scheduler(job_tracker)
        unattributed = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        for_card_0 = make_job_pop_response(model=_CARD_DEMANDING_MODEL)
        await _pop_all(job_tracker, [unattributed, for_card_0])
        job_tracker.mark_admitted_exclusive(unattributed, device_index=None)
        job_tracker.mark_admitted_exclusive(for_card_0, device_index=0)
        now = scheduler._clock()
        scheduler._whole_card_ledger.record_grant(
            1,
            model=_CARD_DEMANDING_MODEL,
            forecast=None,
            cooldown_until=now + 45.0,
            now=now,
            establish_grace_seconds=WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        )

        lines = self._release_lines(scheduler)

        assert job_tracker.is_admitted_exclusive(unattributed) is True
        assert job_tracker.is_admitted_exclusive(for_card_0) is False
        assert len(lines) == 1 and "on card 0" in lines[0]


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
