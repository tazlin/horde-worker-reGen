"""Liveness for the measured-load escape hatch: a starved head at a converged-empty card gets one real attempt.

The measured-truth admission identity refuses a candidate that misses the instantaneous device-free reading on
arithmetic. But a head that has starved past the diagnostic horizon at a card the worker has nothing left to
reclaim on, whose demand is under the achievable ceiling (possible in principle) yet misses available by only a
within-band shortfall, is exactly the regime where the static sampling-VRAM prediction's conservatism or a
transient foreign-VRAM dip explains the gap. Rather than defer an idle card indefinitely on arithmetic that may
be wrong, the arbiter admits one real load so measured reality decides:

- The attempt rides the ordinary dispatch path, so the head reaches the GPU.
- If the child then faults it with the out-of-memory classification, the real load failed: it is the strongest
  possible evidence the demand does not fit this card now, so the job is faulted terminally as a
  scheduling-recovery action (excluded from the consecutive-failure pop pause) and the model is placed on the
  conditional ceiling hold, with the operator warning firing once. A queued sibling then reaches the GPU.
- If the attempt succeeds, nothing else happens: no hold, no fault, and the learned-peak machinery records the
  true figure so future arithmetic improves.

These tests drive the real scheduler and job tracker. The two measured admission inputs (the device-free reading
and the sustained foreign floor) are injected, exactly as the arbiter is designed to consume the parent's
already-reconciled measurement, so the converged-empty/within-band regime is set up deterministically.
"""

from __future__ import annotations

import time

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import (
    InferenceFailureResolution,
    JobStage,
    JobTracker,
)
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.vram_arbiter import _MEASURED_ATTEMPT_BAND_MB, VramArbiter
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

# A 16375 MB card. The foreign floor (900 MB) leaves an achievable ceiling of total - noise(818.75) - 900 =
# 14656.25 MB, above the head's 14573 MB demand, so the head is possible in principle (not structurally denied).
# The device-free reading is set so available = candidate - 400, a 400 MB shortfall well within the band.
_TOTAL_MB = 16375.0
_FOREIGN_FLOOR_MB = 900.0
_HEAD_CANDIDATE_MB = 14573.0
_SIBLING_CANDIDATE_MB = 5000.0
_TARGET_SHORTFALL_MB = 400.0

_HEAD_MODEL = "WAI-NSFW-illustrious-SDXL"
_SIBLING_MODEL = "Deliberate"

_NOISE_MB = admission_noise_buffer_mb(_TOTAL_MB)
# available = device_free - noise (no reservations on a converged-empty card); pick device_free for the shortfall.
_DEVICE_FREE_MB = (_HEAD_CANDIDATE_MB - _TARGET_SHORTFALL_MB) + _NOISE_MB


def _bridge_data() -> object:
    """Bridge data with the measured budget on and whole-card residency left off (the arbiter is the gate)."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
        image_models_to_load=[_HEAD_MODEL, _SIBLING_MODEL],
        max_threads=1,
    )


def _candidate_by_model() -> object:
    """A sampling-VRAM predictor keyed by model: the head's conservative figure, a fittable one for the sibling."""
    return lambda job, baseline: _HEAD_CANDIDATE_MB if job.model == _HEAD_MODEL else _SIBLING_CANDIDATE_MB


class _World:
    """Drives the real scheduler and tracker over a converged-empty card with injected measured inputs.

    One inference process hosts the RAM-staged head (a preload already completed); no other worker model is
    resident, so the card is converged-empty for the head's make-room. The device-free reading and the sustained
    foreign floor are injected so the head misses the instantaneous reading by a within-band shortfall while its
    demand stays under the achievable ceiling.
    """

    def __init__(self, *, staged_model: str, monkeypatch: pytest.MonkeyPatch) -> None:
        self._job_tracker = JobTracker()
        self._job_tracker.set_retry_policy(2)
        self._model_map = HordeModelMap(root={})

        host = make_mock_process_info(0, model_name=staged_model, state=HordeProcessState.PRELOADED_MODEL)
        host.total_vram_mb = int(_TOTAL_MB)
        host.report_sampled_at = time.time()
        host.process_reserved_mb = 0
        self._process_map = _sched_mod.ProcessMap({0: host})
        self._model_map.update_entry(staged_model, load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)

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

        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", _candidate_by_model())
        monkeypatch.setattr(_sched_mod, "predict_job_sampling_vram_mb", _candidate_by_model())
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 1000.0)
        monkeypatch.setattr(self._scheduler, "_measured_available_ram_mb", lambda: 64000.0)
        # Inject the two measured admission inputs directly: a fixed foreign floor (controlling the ceiling) and
        # a fixed device-free reading (controlling available), decoupled so the converged-empty/within-band
        # regime is exact regardless of the harness's committed-footprint arithmetic.
        monkeypatch.setattr(self._scheduler, "_sustained_foreign_floor_mb", lambda *args, **kwargs: _FOREIGN_FLOOR_MB)
        self._scheduler.set_device_free_mb_provider(lambda _device_index: _DEVICE_FREE_MB)
        # The ceiling-hold's live lift predicate reads the card's current achievable ceiling; wire it to the same
        # injected foreign floor the arbiter prices against, so the hold reflects the same measured picture rather
        # than the empty in-test foreign-floor tracker (which would read no floor and lift the hold at once).
        self._job_tracker.set_achievable_ceiling_provider(
            lambda _device_index: _TOTAL_MB - _NOISE_MB - _FOREIGN_FLOOR_MB,
        )

    @property
    def scheduler(self) -> InferenceScheduler:
        return self._scheduler

    @property
    def job_tracker(self) -> JobTracker:
        return self._job_tracker

    async def pop(self, model: str) -> ImageGenerateJobPopResponse:
        job = make_job_pop_response(model)
        await track_popped_job_async(self._job_tracker, job)
        return job

    def starve_head(self, job: ImageGenerateJobPopResponse, *, seconds: float) -> None:
        """Set the head-starvation clock so this job reads as the starved head of an idle device."""
        assert job.id_ is not None
        self._scheduler._head_starvation_job_id = str(job.id_)
        self._scheduler._head_starvation_since = time.time() - seconds

    def freeze(self) -> None:
        """Refreeze the arbiter on a fresh snapshot carrying the injected measured inputs."""
        assert self._scheduler._vram_arbiter is not None
        self._scheduler._vram_arbiter.begin_cycle(self._scheduler.build_vram_arbiter_snapshot())

    async def dispatch(self) -> bool:
        """Run one dispatch pass against the current frozen cycle."""
        return await self._scheduler.start_inference()

    def restage(self, model: str) -> None:
        """Re-stage the host process onto a different RAM-loaded model (the prior job having cleared it)."""
        host = self._process_map[0]
        host.loaded_horde_model_name = model
        host.last_process_state = HordeProcessState.PRELOADED_MODEL
        self._model_map.root.clear()
        self._model_map.update_entry(model, load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)


class TestMeasuredAttemptRegimeIsSetUp:
    """Sanity pins on the injected regime, so a later arithmetic drift cannot silently void the liveness claim."""

    def test_shortfall_is_within_band_and_demand_is_under_ceiling(self) -> None:
        """The head misses available by a within-band shortfall yet stays under the achievable ceiling."""
        available_mb = _DEVICE_FREE_MB - _NOISE_MB
        shortfall_mb = _HEAD_CANDIDATE_MB - available_mb
        ceiling_mb = _TOTAL_MB - _NOISE_MB - _FOREIGN_FLOOR_MB
        assert 0.0 < shortfall_mb <= _MEASURED_ATTEMPT_BAND_MB
        assert ceiling_mb > _HEAD_CANDIDATE_MB


class TestMeasuredAttemptDispatchesThenFaultArmsHold:
    """The failure branch: the head attempts and dispatches, its OOM faults it terminally and arms the hold."""

    async def test_head_attempts_dispatches_then_oom_faults_holds_and_sibling_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Within a bounded pass the head dispatches (attempt); its OOM faults it (recovery), holds it, frees GPU."""
        world = _World(staged_model=_HEAD_MODEL, monkeypatch=monkeypatch)
        head = await world.pop(_HEAD_MODEL)
        assert head.id_ is not None

        world.starve_head(head, seconds=90.0)
        world.freeze()
        dispatched = await world.dispatch()

        # The measured-attempt escape hatch admitted the head onto the GPU despite the tight reading.
        assert dispatched is True
        assert head in world.job_tracker.jobs_in_progress
        assert world.job_tracker.is_measured_attempt(head) is True
        assert world.scheduler._process_map[0].last_control_flag is HordeControlFlag.START_INFERENCE

        # The child faults the real load with the out-of-memory classification: the strongest evidence the demand
        # does not fit this card now. This is the signal the dispatcher computes and hands the tracker.
        resolution = await world.job_tracker.handle_job_fault(head, is_resource_failure=True, retryable=True)

        # Terminal, not a degraded retry: a converged-empty card has nothing left to reclaim for a second try.
        assert resolution is InferenceFailureResolution.FAULTED
        assert world.job_tracker.get_stage(head.id_) is JobStage.PENDING_SUBMIT
        assert world.job_tracker.jobs_lookup[head].state is GENERATION_STATE.faulted
        # The fault is a scheduling-recovery action, so it is excluded from the consecutive-failure pop pause.
        assert world.job_tracker.was_faulted_by_scheduling_recovery(head.id_) is True
        # A real attempt failed, so the model is held (real-attempt evidence) and the fault counter advanced once.
        assert world.job_tracker.is_model_held_by_ceiling(_HEAD_MODEL) is True
        assert world.job_tracker.measured_attempt_faults == 1

        # GPU-feeding outcome: with the head resolved, a queued sibling reaches the GPU (no head-protection wedge).
        world.restage(_SIBLING_MODEL)
        sibling = await world.pop(_SIBLING_MODEL)
        world.starve_head(sibling, seconds=0.0)
        world.freeze()
        dispatched_sibling = await world.dispatch()
        assert dispatched_sibling is True
        assert sibling in world.job_tracker.jobs_in_progress
        assert world.scheduler._process_map[0].last_control_flag is HordeControlFlag.START_INFERENCE


class TestMeasuredAttemptSuccessArmsNothing:
    """The success branch: the head attempts, dispatches, and completes with no hold and no fault."""

    async def test_successful_attempt_arms_no_hold_and_faults_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A measured attempt that does not OOM leaves no ceiling hold and no fault: reality proved the fit."""
        world = _World(staged_model=_HEAD_MODEL, monkeypatch=monkeypatch)
        head = await world.pop(_HEAD_MODEL)
        assert head.id_ is not None

        world.starve_head(head, seconds=90.0)
        world.freeze()
        dispatched = await world.dispatch()

        assert dispatched is True
        assert world.job_tracker.is_measured_attempt(head) is True
        # No OOM was reported, so nothing arms: the tag alone never holds a model or faults a job.
        assert world.job_tracker.is_model_held_by_ceiling(_HEAD_MODEL) is False
        assert world.job_tracker.measured_attempt_faults == 0
        assert world.job_tracker.jobs_lookup[head].state is not GENERATION_STATE.faulted
