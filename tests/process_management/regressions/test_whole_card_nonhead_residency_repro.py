"""Regression guard for the non-head whole-card residency starvation (the deep-queue head wedge).

The shape of the wedge, on a single 24GB card with ``whole_card_exclusive_residency`` enabled:

  * The head of the queue is an ordinary SDXL job whose model is resident on an idle process (ready to
    dispatch). A heavy Flux fp8 job sits *behind* it, deeper in the queue.
  * Flux's forecast reads ``needs_exclusive_residency``, so the scheduler grants it a whole-card residency
    (pre-staging it, then converging to sole residency), even though it is not the head.

Granting the residency to the deep Flux job reserves the entire card and tears down the sibling processes
serving the heads ahead of it (including the head's own resident process). The real head then has no
resident process and "no preload admitted", so it parks while the card is held for a job whose turn has
not come. The residency stays held because Flux is still queued, so the head starves until the recovery
supervisor soft-resets the pools and faults the backlog.

The fix gates the whole-card residency teardown path on the job being the head (``is_head_blocker``): a
deeper-queue heavy job defers until it becomes the head rather than reserving the card. These tests pin
that a non-head Flux does not claim the card, while a Flux that *is* the head still does (the legitimate
path, including the pre-stage overlap, must not regress). The diagnostic tests pin that a head stalled
behind a non-head residency is attributed to that residency rather than mistaken for a budget defer.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# A 16GB card. Flux fp8 genuinely needs sole residency here: its ~11.5GB weights leave under a full sibling
# model's worth of room (``_CORESIDENT_SIBLING_MODEL_FLOOR_MB``) at sole residency, so it is a true whole-card
# head: the case this file's head-gate must serve while still refusing a *non-head* Flux. (On a roomy 24GB
# card Flux fp8 co-resides instead, which is a different regime covered by the budget-wins-on-a-roomy-card path.)
_DEVICE_TOTAL_VRAM_MB = 16375
_PER_PROCESS_OVERHEAD_MB = 1288
_VRAM_RESERVE_MB = 2048
_RAM_RESERVE_MB = 4096

_FLUX_MODEL = "Flux.1-Schnell fp8 (Compact)"
_FLUX_BASELINE = "flux_schnell"
_FLUX_WEIGHTS_MB = 11500.0
_FLUX_SAMPLING_PEAK_MB = 13500.0  # weights still fit alone on 16GB (the residency is the clean sole-residency case)

_HEAD_SDXL = "Juggernaut XL"  # a light SDXL head, ready to dispatch
_OTHER_SDXL = "CyberRealistic Pony"


def _wedge_bridge_data() -> Mock:
    """Budget-on, whole-card-on, 4-process, 24GB configuration."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        whole_card_exclusive_residency=True,
        whole_card_residency_safety_off_gpu=False,
        safety_on_gpu=False,
        vram_reserve_mb=_VRAM_RESERVE_MB,
        ram_reserve_mb=_RAM_RESERVE_MB,
        vram_per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        overbudget_exclusive_mode=True,
        whole_card_residency_cooldown_seconds=0,
        image_models_to_load=[_HEAD_SDXL, _OTHER_SDXL, _FLUX_MODEL],
        max_threads=1,
    )


def _seed_flux_needs_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin Flux's weight + sampling-peak estimates so its forecast deterministically needs sole residency."""
    monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _FLUX_WEIGHTS_MB)
    monkeypatch.setattr(
        resource_budget,
        "predict_job_sampling_vram_mb",
        lambda job, baseline: _FLUX_SAMPLING_PEAK_MB,
    )


def _make_scheduler(
    process_map: ProcessMap,
    job_tracker: JobTracker,
) -> InferenceScheduler:
    """An InferenceScheduler over ``process_map``/``job_tracker`` with the scale-down + safety levers stubbed.

    Stubbing ``scale_inference_processes`` / ``pause_safety_on_gpu`` lets a test observe the scheduler's
    *decision* (whether a whole-card residency was recorded for a model) without driving real OS processes.
    """
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=_wedge_bridge_data(),
        max_concurrent=1,
        max_inference=4,
    )
    scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=len(list(process_map.values())))
    scheduler._process_lifecycle.pause_safety_on_gpu = Mock(return_value=True)
    scheduler._process_lifecycle.is_safety_gpu_paused = False
    scheduler._measured_available_ram_mb = lambda: 48000.0  # type: ignore[method-assign]
    return scheduler


def _resident(pid: int, model: str | None, state: HordeProcessState) -> HordeProcessInfo:
    """A process pinned to the 24GB device with a low device-free reading (siblings hold the card)."""
    proc = make_mock_process_info(pid, model_name=model, state=state)
    proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
    # Leave only ~2GB free so co-residence is impossible and Flux's forecast needs the whole card.
    proc.vram_usage_mb = _DEVICE_TOTAL_VRAM_MB - 2000
    if model is not None:
        # A fresh committed reservation matching the held card, so the arbiter's measured floor denies a
        # non-head candidate against the resident's footprint rather than relaxing to admit on a cold ledger.
        proc.process_reserved_mb = 16000
        proc.report_sampled_at = time.time()
    return proc


def _flux_residency_recorded(scheduler: InferenceScheduler) -> bool:
    """Whether a whole-card residency is held for Flux (the scheduler reserved the card for it)."""
    found, _device = scheduler._residency_holder_for_model(_FLUX_MODEL)
    return found


# --------------------------------------------------------------------------------------------------------- #
#  A non-head Flux must not claim the whole card.                                                            #
# --------------------------------------------------------------------------------------------------------- #


class TestNonHeadFluxDoesNotClaimCard:
    """A heavy Flux job behind a resident SDXL head must not be granted a whole-card residency."""

    async def test_establish_path_does_not_reserve_card_for_deep_flux(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No live job: a deep Flux must not establish a residency while a different SDXL head is pending.

        The head SDXL is resident on an idle process (ready to dispatch); Flux is the deepest pending job.
        Reserving the card for Flux would tear the head's process down. The teardown lever must stay untouched.
        """
        _seed_flux_needs_exclusive(monkeypatch)
        head_proc = _resident(1, _HEAD_SDXL, HordeProcessState.WAITING_FOR_JOB)
        spare = _resident(2, None, HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({1: head_proc, 2: spare})
        job_tracker = JobTracker()
        scheduler = _make_scheduler(process_map, job_tracker)

        await track_popped_job_async(job_tracker, make_job_pop_response(_HEAD_SDXL))  # the head
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))

        scheduler.preload_models()

        assert not _flux_residency_recorded(scheduler), (
            "a non-head Flux must not be granted a whole-card residency; reserving the card for it starves the "
            "resident SDXL head ahead of it"
        )
        scheduler._process_lifecycle.scale_inference_processes.assert_not_called()

    async def test_prestage_path_does_not_reserve_card_for_deep_flux(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Live job in flight: a deep Flux must not pre-stage a whole-card residency behind a pending SDXL head.

        An SDXL job is sampling, the next-to-serve head is another resident SDXL, and Flux is the deepest job.
        The pre-stage path (which fires while a live job holds the device) must not claim the card for the
        non-head Flux.
        """
        _seed_flux_needs_exclusive(monkeypatch)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 12000.0)

        busy = _resident(1, _HEAD_SDXL, HordeProcessState.INFERENCE_STARTING)
        head_proc = _resident(2, _OTHER_SDXL, HordeProcessState.WAITING_FOR_JOB)
        spare = _resident(3, None, HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({1: busy, 2: head_proc, 3: spare})
        job_tracker = JobTracker()
        scheduler = _make_scheduler(process_map, job_tracker)

        live_job = make_job_pop_response(_HEAD_SDXL)
        await track_popped_job_async(job_tracker, live_job)
        await mark_job_in_progress_async(job_tracker, live_job)
        await track_popped_job_async(job_tracker, make_job_pop_response(_OTHER_SDXL))  # the head (next to serve)
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))

        scheduler.preload_models()

        assert not _flux_residency_recorded(scheduler), (
            "a non-head Flux must not pre-stage a whole-card residency while an SDXL head is the next to serve"
        )

    async def test_deep_flux_behind_multiple_resident_heads_does_not_reserve_card(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Corner case: several resident SDXL heads ahead of Flux; none of them may be torn down for it."""
        _seed_flux_needs_exclusive(monkeypatch)
        head_a = _resident(1, _HEAD_SDXL, HordeProcessState.WAITING_FOR_JOB)
        head_b = _resident(2, _OTHER_SDXL, HordeProcessState.WAITING_FOR_JOB)
        spare = _resident(3, None, HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({1: head_a, 2: head_b, 3: spare})
        job_tracker = JobTracker()
        scheduler = _make_scheduler(process_map, job_tracker)

        await track_popped_job_async(job_tracker, make_job_pop_response(_HEAD_SDXL))
        await track_popped_job_async(job_tracker, make_job_pop_response(_OTHER_SDXL))
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))

        scheduler.preload_models()

        assert not _flux_residency_recorded(scheduler)
        scheduler._process_lifecycle.scale_inference_processes.assert_not_called()


# --------------------------------------------------------------------------------------------------------- #
#  A Flux that IS the head must still claim the whole card (the legitimate path must not regress).           #
# --------------------------------------------------------------------------------------------------------- #


class TestHeadFluxStillClaimsCard:
    """The head-gate must not break the case the whole-card residency exists for: a heavy head."""

    async def test_flux_as_head_reserves_the_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When Flux is the head, the whole-card residency must still be established (control, stays GREEN)."""
        _seed_flux_needs_exclusive(monkeypatch)
        resident_sibling = _resident(1, _OTHER_SDXL, HordeProcessState.WAITING_FOR_JOB)
        spare = _resident(2, None, HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({1: resident_sibling, 2: spare})
        job_tracker = JobTracker()
        scheduler = _make_scheduler(process_map, job_tracker)

        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))

        scheduler.preload_models()

        assert _flux_residency_recorded(scheduler), (
            "a Flux head must still get its whole-card residency; the head-gate must not break the legitimate path"
        )

    async def test_flux_claims_card_once_it_becomes_the_head(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deep Flux must claim the card only after the SDXL job ahead of it drains and Flux becomes the head.

        First cycle (SDXL head pending ahead): no residency for Flux. After the SDXL head is dispatched, Flux is
        the head and the residency must then be established; the deferral is until its turn, not forever.
        """
        _seed_flux_needs_exclusive(monkeypatch)
        head_proc = _resident(1, _HEAD_SDXL, HordeProcessState.WAITING_FOR_JOB)
        spare = _resident(2, None, HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({1: head_proc, 2: spare})
        job_tracker = JobTracker()
        scheduler = _make_scheduler(process_map, job_tracker)

        sdxl_head = make_job_pop_response(_HEAD_SDXL)
        await track_popped_job_async(job_tracker, sdxl_head)
        await track_popped_job_async(job_tracker, make_job_pop_response(_FLUX_MODEL, width=1216, height=1216))

        scheduler.preload_models()
        assert not _flux_residency_recorded(scheduler), "while the SDXL head is pending, Flux must not claim the card"

        # The SDXL head is now dispatched (in progress), so Flux is the head of the queue.
        await mark_job_in_progress_async(job_tracker, sdxl_head)
        scheduler.preload_models()

        assert _flux_residency_recorded(scheduler), "once Flux is the head, it must claim the whole card"


# --------------------------------------------------------------------------------------------------------- #
#  The stall is attributed to the held non-head residency, not mistaken for a budget defer.                  #
# --------------------------------------------------------------------------------------------------------- #


class TestNonHeadResidencyDispatchDiagnostic:
    """``_diagnose_dispatch_stall`` must name a held non-head residency as the reason a head cannot load."""

    def _scheduler_with_no_residents(self) -> tuple[InferenceScheduler, JobTracker]:
        """A worker whose only inference process is idle and model-free (the head's model is not resident)."""
        spare = _resident(2, None, HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({2: spare})
        job_tracker = JobTracker()
        scheduler = _make_scheduler(process_map, job_tracker)
        return scheduler, job_tracker

    async def test_head_stall_attributes_a_held_nonhead_residency(self) -> None:
        """The head's model is not resident while a residency is held for Flux: name Flux as the cause."""
        scheduler, job_tracker = self._scheduler_with_no_residents()
        # A whole-card residency is held for Flux (a different, deeper-queue model).
        scheduler._residency_state(None).model = _FLUX_MODEL
        head = await track_popped_job_async(job_tracker, make_job_pop_response(_OTHER_SDXL))

        reason = scheduler._diagnose_dispatch_stall(head, {})

        assert "whole-card residency is held for non-head model" in reason
        assert _FLUX_MODEL in reason
        assert "budget defer" not in reason

    async def test_head_stall_without_residency_is_not_misattributed(self) -> None:
        """Control: with no residency held, the not-resident head falls back to the generic reason."""
        scheduler, job_tracker = self._scheduler_with_no_residents()
        head = await track_popped_job_async(job_tracker, make_job_pop_response(_OTHER_SDXL))

        reason = scheduler._diagnose_dispatch_stall(head, {})

        assert "whole-card residency is held for non-head model" not in reason
