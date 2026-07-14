"""Reproduction of the unroutable-head wedge: a head whose demand exceeds the card's achievable ceiling.

A head-of-queue job whose VRAM demand exceeds what the card can *ever* offer (its total net of the noise
buffer and the VRAM other processes sustain) used to defer forever: the old impossibility test assumed an
empty card frees everything including foreign allocations, so the demand read as merely-not-fitting-now and
the scheduler tore down every resident, idled the card, and wedged the queue behind head protection with
nothing to reroute to on a single-device worker.

The fix makes the arbiter's achievable ceiling foreign-aware (``total - noise - sustained_foreign_floor``) and
makes a structural-impossibility DENY for an unroutable head terminal: the head is faulted for reissue (as a
scheduling-recovery action, excluded from the consecutive-failure pop pause), its model is placed on a
conditional ceiling hold so pop advertising stops offering it while the ceiling holds, and the queued siblings
dispatch. The hold is not permanent: it lifts on its own once the card's current ceiling recedes past a
hysteresis margin (the foreign floor dropped), so the same box serves the model again with no operator action.

These tests drive the real scheduler and job tracker across scheduling cycles with a hand-advanced clock, so
the foreign floor is established from measured device truth exactly as it is in the running worker.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_popper import _select_models_for_pop
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.foreign_vram_floor import FOREIGN_FLOOR_WINDOW_SECONDS
from horde_worker_regen.process_management.resources.resource_budget import is_model_locally_unservable_for
from horde_worker_regen.process_management.resources.vram_arbiter import VramArbiter
from horde_worker_regen.process_management.scheduling import inference_scheduler as _sched_mod
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl

# Faithful to the reproduced incident: a 16375 MB card whose desktop/other processes sustain ~1912 MB, so the
# achievable ceiling is total - noise(818.75) - foreign(1912) = 13644.25 MB. The head needs 14573 MB and can
# never fit; a sibling needs 5000 MB and fits.
_TOTAL_MB = 16375.0
_FOREIGN_MB = 1912.0
_IMPOSSIBLE_CANDIDATE_MB = 14573.0
_FITTABLE_CANDIDATE_MB = 13000.0
_SIBLING_CANDIDATE_MB = 5000.0

_HEAD_MODEL = "WAI-NSFW-illustrious-SDXL"
_SIBLING_MODEL = "Deliberate"


def _bridge_data() -> Mock:
    """Bridge data with the measured budget on and whole-card residency left off (the arbiter is the gate)."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
        image_models_to_load=[_HEAD_MODEL, _SIBLING_MODEL],
        max_threads=1,
    )


def _candidate_by_model(head_mb: float) -> object:
    """A sampling-VRAM predictor keyed by model: the head gets ``head_mb``, the sibling a fittable figure."""
    return lambda job, baseline: head_mb if job.model == _HEAD_MODEL else _SIBLING_CANDIDATE_MB


class _World:
    """Drives the real scheduler and tracker across cycles with a hand-advanced foreign-floor clock.

    A resident sibling process and an idle slot mirror the incident's idle-pool-with-evicted-head. The
    device-free reading is derived so the instantaneous foreign reading equals ``_FOREIGN_MB`` every cycle,
    so the sustained floor converges deterministically once the observation window is covered.
    """

    def __init__(
        self, *, head_candidate_mb: float, resident_reserved_mb: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.now = 0.0
        self._job_tracker = JobTracker()
        self._model_map = HordeModelMap(root={})

        idle = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        sibling_lane = make_mock_process_info(1, model_name=_SIBLING_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        for proc in (idle, sibling_lane):
            proc.total_vram_mb = int(_TOTAL_MB)
            proc.report_sampled_at = time.time()
        sibling_lane.process_reserved_mb = int(resident_reserved_mb)
        self._process_map = _sched_mod.ProcessMap({0: idle, 1: sibling_lane})
        self._model_map.update_entry(_SIBLING_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)

        reference = {
            _HEAD_MODEL: make_mock_model_reference_record(_HEAD_MODEL, baseline=_SDXL),
            _SIBLING_MODEL: make_mock_model_reference_record(_SIBLING_MODEL, baseline=_SDXL),
        }
        self._scheduler = _make_inference_scheduler(
            process_map=self._process_map,
            horde_model_map=self._model_map,
            job_tracker=self._job_tracker,
            bridge_data=_bridge_data(),
            model_metadata=make_test_model_metadata(reference),
            max_concurrent=1,
            max_inference=2,
            device_free_mb=None,
        )
        self._scheduler.set_vram_arbiter(VramArbiter())
        self._scheduler._foreign_floor_clock = lambda: self.now

        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", _candidate_by_model(head_candidate_mb))
        monkeypatch.setattr(_sched_mod, "predict_job_sampling_vram_mb", _candidate_by_model(head_candidate_mb))
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 1000.0)
        monkeypatch.setattr(self._scheduler, "_measured_available_ram_mb", lambda: 64000.0)

        # Derive the device-free reading so foreign_now = total - device_free - committed == _FOREIGN_MB. The
        # committed footprint is constant (the residents do not change here), so every cycle contributes the
        # same foreign reading and the sustained floor converges to _FOREIGN_MB once the window is covered.
        committed = self._process_map.committed_vram_mb(
            context_constant_mb=self._scheduler.resolved_context_constant_mb(),
        )
        self._device_free_mb = _TOTAL_MB - _FOREIGN_MB - committed
        self._scheduler.set_device_free_mb_provider(lambda _device_index: self._device_free_mb)

    @property
    def scheduler(self) -> InferenceScheduler:
        return self._scheduler

    @property
    def job_tracker(self) -> JobTracker:
        return self._job_tracker

    @property
    def device_free_mb(self) -> float:
        return self._device_free_mb

    async def pop(self, model: str) -> ImageGenerateJobPopResponse:
        job = make_job_pop_response(model)
        await track_popped_job_async(self._job_tracker, job)
        return job

    def _freeze(self) -> None:
        """Refreeze the arbiter on a fresh snapshot at the current clock, recording this cycle's foreign sample."""
        assert self._scheduler._vram_arbiter is not None
        self._scheduler._vram_arbiter.begin_cycle(self._scheduler.build_vram_arbiter_snapshot())

    def preload_cycle(self, *, advance_to: float | None = None) -> None:
        """Advance the clock, refreeze the arbiter, and run one preload pass."""
        if advance_to is not None:
            self.now = advance_to
        self._freeze()
        self._scheduler.preload_models()

    async def dispatch(self) -> bool:
        """Run one dispatch pass against the current frozen cycle."""
        return await self._scheduler.start_inference()


class TestUnroutableHeadFaultsAndHolds:
    """The anchor liveness claim: an unroutable head is faulted, its model held, its sibling dispatched."""

    async def test_impossible_head_faults_holds_and_sibling_dispatches(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Within a bound the head faults (recovery origin, no pause), the model is held, the sibling runs."""
        world = _World(
            head_candidate_mb=_IMPOSSIBLE_CANDIDATE_MB, resident_reserved_mb=2000.0, monkeypatch=monkeypatch
        )
        head = await world.pop(_HEAD_MODEL)
        sibling = await world.pop(_SIBLING_MODEL)
        assert head.id_ is not None
        assert sibling.id_ is not None

        # Warm-up cycle: with fewer than a full window of samples the foreign floor is unknown, so the head is
        # not yet judged impossible and is not faulted (the pre-foreign behaviour is preserved during warm-up).
        world.preload_cycle(advance_to=0.0)
        assert world.job_tracker.get_stage(head.id_) is JobStage.PENDING_INFERENCE
        assert world.job_tracker.is_model_held_by_ceiling(_HEAD_MODEL) is False

        # Once the observation window is covered the sustained foreign floor lands and the head is judged
        # unroutable: it is faulted terminally this cycle.
        world.preload_cycle(advance_to=FOREIGN_FLOOR_WINDOW_SECONDS + 1.0)

        assert world.job_tracker.get_stage(head.id_) is JobStage.PENDING_SUBMIT
        assert world.job_tracker.jobs_lookup[head].state is GENERATION_STATE.faulted
        assert head not in world.job_tracker.jobs_in_progress
        # The fault is a scheduling-recovery action, so it is excluded from the consecutive-failure pop pause.
        assert world.job_tracker.was_faulted_by_scheduling_recovery(head.id_) is True
        # The model is held while the current ceiling holds, so pop advertising and dispatch both stop offering
        # it. The live predicate reads the scheduler's own current-ceiling provider (wired at construction).
        assert world.job_tracker.is_model_held_by_ceiling(_HEAD_MODEL) is True

        # The GPU-feeding outcome: with the head resolved, the fitting sibling reaches dispatch.
        dispatched = await world.dispatch()
        assert dispatched is True
        assert sibling in world.job_tracker.jobs_in_progress
        sibling_lane = world.scheduler._process_map[1]
        assert sibling_lane.last_control_flag is HordeControlFlag.START_INFERENCE


class TestTransientShortfallIsNotFaulted:
    """Regression pin: a head that fits the achievable ceiling but not the tight reading now defers, never faults."""

    async def test_below_ceiling_head_defers_and_is_not_held(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A candidate under the achievable ceiling is a transient shortfall: it defers/reclaims, never faults.

        The residents physically hold the card (a tight device-free reading), so the head does not fit *now*,
        but its demand is under the achievable ceiling, so it is not structurally impossible. It must keep its
        queue position and drive reclaim rather than being terminally faulted.
        """
        # Residents hold a large committed footprint, so the derived device-free reading is tight and the
        # below-ceiling candidate does not fit now.
        world = _World(head_candidate_mb=_FITTABLE_CANDIDATE_MB, resident_reserved_mb=10000.0, monkeypatch=monkeypatch)
        # The tight reading is what makes this a genuine "does not fit now" case rather than an admit.
        assert world.device_free_mb - admission_noise_buffer_mb(_TOTAL_MB) < _FITTABLE_CANDIDATE_MB
        head = await world.pop(_HEAD_MODEL)
        assert head.id_ is not None

        world.preload_cycle(advance_to=0.0)
        world.preload_cycle(advance_to=FOREIGN_FLOOR_WINDOW_SECONDS + 1.0)
        # And a few more cycles to prove the defer is stable, not a delayed fault.
        for extra in range(3):
            world.preload_cycle(advance_to=FOREIGN_FLOOR_WINDOW_SECONDS + 2.0 + extra)

        assert world.job_tracker.get_stage(head.id_) is JobStage.PENDING_INFERENCE
        assert head not in world.job_tracker.jobs_in_progress
        assert world.job_tracker.is_model_held_by_ceiling(_HEAD_MODEL) is False
        assert world.job_tracker.was_faulted_by_scheduling_recovery(head.id_) is False


class TestConditionalCeilingHold:
    """The hold is a live predicate against the current ceiling, not a session-permanent quarantine.

    A mutable ceiling provider stands in for the scheduler's live achievable-ceiling read: the hold applies
    while the candidate sits above ``ceiling - margin`` and lifts once the ceiling recedes past it.
    """

    _MARGIN_MB = JobTracker._CEILING_HOLD_LIFT_MARGIN_MB

    def _tracker_with_ceiling(self, ceiling_box: list[float | None]) -> JobTracker:
        """A tracker whose ceiling provider reads the (mutable) single-element box, keyed by device."""
        job_tracker = JobTracker()
        job_tracker.set_achievable_ceiling_provider(lambda _device_index: ceiling_box[0])
        return job_tracker

    def test_held_model_is_not_advertised_then_reappears_when_ceiling_improves(self) -> None:
        """A held model is dropped from pop selection, and reappears once the current ceiling recedes past margin."""
        bridge_data = _bridge_data()
        # Ceiling below the candidate: the incident's ~13644 MB against a 14573 MB demand.
        ceiling_box: list[float | None] = [13644.0]
        job_tracker = self._tracker_with_ceiling(ceiling_box)
        process_map = _sched_mod.ProcessMap({})

        assert job_tracker.hold_model_by_ceiling(_HEAD_MODEL, candidate_mb=_IMPOSSIBLE_CANDIDATE_MB) is True
        assert is_model_locally_unservable_for(bridge_data, job_tracker, _HEAD_MODEL) is True
        held = _select_models_for_pop(
            bridge_data, process_map, job_tracker, max_inference_processes=2, last_pop_had_no_jobs=False
        )
        assert held is not None and _HEAD_MODEL not in held and _SIBLING_MODEL in held

        # The operator frees other-process VRAM: the current ceiling rises past candidate + margin, so the hold
        # lifts on its own and the model is advertised again with no operator action.
        ceiling_box[0] = _IMPOSSIBLE_CANDIDATE_MB + self._MARGIN_MB + 1.0
        assert job_tracker.is_model_held_by_ceiling(_HEAD_MODEL) is False
        assert is_model_locally_unservable_for(bridge_data, job_tracker, _HEAD_MODEL) is False
        lifted = _select_models_for_pop(
            bridge_data, process_map, job_tracker, max_inference_processes=2, last_pop_had_no_jobs=False
        )
        assert lifted is not None and _HEAD_MODEL in lifted

    def test_hold_does_not_lift_within_the_hysteresis_band(self) -> None:
        """A ceiling that has risen to (but not past) the candidate plus margin keeps the hold, avoiding flap."""
        ceiling_box: list[float | None] = [13644.0]
        job_tracker = self._tracker_with_ceiling(ceiling_box)
        assert job_tracker.hold_model_by_ceiling(_HEAD_MODEL, candidate_mb=_IMPOSSIBLE_CANDIDATE_MB) is True

        # The ceiling recovers to exactly the candidate (no margin to spare): still held, not flapping off.
        ceiling_box[0] = _IMPOSSIBLE_CANDIDATE_MB
        assert job_tracker.is_model_held_by_ceiling(_HEAD_MODEL) is True
        # Even just inside the margin band it stays held.
        ceiling_box[0] = _IMPOSSIBLE_CANDIDATE_MB + self._MARGIN_MB - 1.0
        assert job_tracker.is_model_held_by_ceiling(_HEAD_MODEL) is True

    def test_rearm_after_lift_reports_newly_armed_for_edge_triggered_warning(self) -> None:
        """After a lift a fresh shortfall re-arms the hold (returns True), so the operator warning re-fires once."""
        ceiling_box: list[float | None] = [13644.0]
        job_tracker = self._tracker_with_ceiling(ceiling_box)

        assert job_tracker.hold_model_by_ceiling(_HEAD_MODEL, candidate_mb=_IMPOSSIBLE_CANDIDATE_MB) is True
        # A repeat arm while still held is not newly armed (the warning is not re-emitted every evaluation).
        assert job_tracker.hold_model_by_ceiling(_HEAD_MODEL, candidate_mb=_IMPOSSIBLE_CANDIDATE_MB) is False

        # The ceiling recedes and the hold lifts (and is forgotten).
        ceiling_box[0] = _IMPOSSIBLE_CANDIDATE_MB + self._MARGIN_MB + 1.0
        assert job_tracker.is_model_held_by_ceiling(_HEAD_MODEL) is False

        # A later shortfall re-arms: newly armed again, so the edge-triggered warning fires once more.
        ceiling_box[0] = 13644.0
        assert job_tracker.hold_model_by_ceiling(_HEAD_MODEL, candidate_mb=_IMPOSSIBLE_CANDIDATE_MB) is True
