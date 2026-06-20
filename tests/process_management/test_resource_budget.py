"""Tests for the worker-owned VRAM budget and its scheduler gating."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import resource_budget
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.resource_budget import BudgetVerdict, RamBudget, VramBudget

from .conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_job,
    make_mock_process_info,
    track_popped_job_async,
)
from .test_inference_scheduling import _make_inference_scheduler


class TestVramBudget:
    """Unit tests for the VramBudget accountant itself (prediction stubbed)."""

    def test_cold_start_admits_when_no_telemetry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no VRAM telemetry yet, the budget admits so a cold worker never wedges."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 9999.0)
        budget = VramBudget(reserve_mb=2048.0)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(job, "stable_diffusion_1", free_vram_mb=None)
        assert verdict.fits is True
        assert verdict.available_mb is None

    def test_admits_when_estimate_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A None estimate means unknown cost; the budget admits rather than blocking blindly."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: None)
        budget = VramBudget(reserve_mb=2048.0)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(job, None, free_vram_mb=500.0)
        assert verdict.fits is True
        assert verdict.predicted_mb is None

    def test_fits_when_free_covers_predicted_plus_reserve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Free VRAM at or above predicted + reserve fits."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 4000.0)
        budget = VramBudget(reserve_mb=2000.0)
        job = make_job_pop_response("stable_diffusion")
        assert budget.check_job(job, "x", free_vram_mb=6000.0).fits is True
        assert budget.check_job(job, "x", free_vram_mb=5999.0).fits is False

    def test_set_reserve_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Updating the reserve changes the verdict immediately (live config reload)."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 4000.0)
        budget = VramBudget(reserve_mb=2000.0)
        job = make_job_pop_response("stable_diffusion")
        assert budget.check_job(job, "x", free_vram_mb=5000.0).fits is False
        budget.set_reserve_mb(1000.0)
        assert budget.check_job(job, "x", free_vram_mb=5000.0).fits is True

    def test_ram_budget_fits_logic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RamBudget admits when available RAM covers predicted RAM plus reserve, else defers."""
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 6000.0)
        budget = RamBudget(reserve_mb=4096.0)
        job = make_job_pop_response("stable_diffusion")
        assert budget.check_job(job, "x", available_ram_mb=11000.0).fits is True
        assert budget.check_job(job, "x", available_ram_mb=9000.0).fits is False
        assert budget.check_job(job, "x", available_ram_mb=None).fits is True

    def test_verdict_reason_strings(self) -> None:
        """The verdict reason renders the relevant branch for logging."""
        assert (
            "cold start" in BudgetVerdict(fits=True, predicted_mb=None, available_mb=None, reserve_mb=2048.0).reason()
        )
        assert (
            "no burden estimate"
            in BudgetVerdict(fits=True, predicted_mb=None, available_mb=1000.0, reserve_mb=2048.0).reason()
        )
        assert (
            "does NOT fit"
            in BudgetVerdict(fits=False, predicted_mb=4000.0, available_mb=1000.0, reserve_mb=2048.0).reason()
        )
        assert "fits" in BudgetVerdict(fits=True, predicted_mb=1000.0, available_mb=8000.0, reserve_mb=2048.0).reason()


def _budget_bridge_data() -> Mock:
    """Mock bridge data with the VRAM budget enabled and real numeric reserves."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2000,
        ram_reserve_mb=4096,
        image_models_to_load=["model_a", "model_b"],
    )


class TestPreloadBudgetGate:
    """Integration tests for the preload-time VRAM budget gate inside the scheduler."""

    async def test_preload_deferred_and_reclaims_when_over_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the next model will not fit, preload is deferred and idle VRAM is reclaimed."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 8000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        # A second, idle process holding a different resident model: the eviction candidate. It reports
        # the (low) device-wide free VRAM the budget reads.
        resident = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.WAITING_FOR_JOB)
        resident.total_vram_mb = 16000
        resident.vram_usage_mb = 15000  # 1000 MB free, well under 8000 + 2000
        process_map = ProcessMap({0: spare, 1: resident})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )

        assert scheduler.preload_models() is False
        # The spare process was NOT told to preload...
        assert spare.last_control_flag != HordeControlFlag.PRELOAD_MODEL
        # ...and the idle resident model was evicted to reclaim VRAM (residency overridden under pressure).
        assert resident.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM

    async def test_starved_head_force_admitted_before_wedge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A head the budget can never satisfy is force-admitted once it has starved past the horizon.

        Regression for an observed save-our-ship catastrophe: after a whole-card teardown/restore left
        idle processes holding allocator-stranded RAM, the head-of-queue model failed the RAM budget and no
        reclaim path could free room, so it was deferred indefinitely until the recovery supervisor soft-reset
        every pool and faulted the backlog. The starvation backstop must force-admit the head onto the idle
        device first. Here ``unload_models`` always reports reclaiming something, so the existing best-effort
        admit never fires (the gate defers every tick); only the time-based backstop can break the wedge.
        """
        import time as _time

        from horde_worker_regen.process_management import inference_scheduler as _sched_mod

        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 1000.0)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 50000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 1000  # ample free VRAM, so only the RAM gate defers
        process_map = ProcessMap({0: spare})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 8000.0)
        # Reclaim always "succeeds", so the RAM branch defers every tick and the existing best-effort
        # admit (which requires reclaim to be exhausted) never triggers -- a perpetual wedge.
        monkeypatch.setattr(scheduler, "unload_models", lambda *a, **k: True)

        # First tick: the head cannot be admitted and the wedge begins; the starvation clock starts.
        assert scheduler.preload_models() is False
        assert spare.last_control_flag != HordeControlFlag.PRELOAD_MODEL
        assert scheduler._head_starvation_job_id == str(job.id_)

        # Backdate the clock past the force-admit horizon to simulate a sustained wedge.
        scheduler._head_starvation_since = _time.time() - (_sched_mod._HEAD_STARVATION_FORCE_ADMIT_SECONDS + 1.0)

        # Next tick: the head is force-admitted best-effort instead of wedging into save-our-ship.
        assert scheduler.preload_models() is True
        assert spare.last_control_flag == HordeControlFlag.PRELOAD_MODEL
        assert job_tracker._tracked_for(job).admitted_over_budget is True

    async def test_starved_head_clock_resets_when_live_job_holds_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The starvation clock must not run while a live job holds the device (head is merely queued).

        Otherwise the backstop would force a second concurrent heavy load and reintroduce the very
        over-commit the budget guards against.
        """
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 1000.0)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 50000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 1000
        busy = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({0: spare, 1: busy})

        job_tracker = JobTracker()
        live = make_job_pop_response("model_b")
        await track_popped_job_async(job_tracker, live)
        await job_tracker.mark_inference_started(live)
        head = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, head)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 8000.0)

        scheduler._update_head_starvation_timer(head)
        # A live job holds the device, so the head's clock must not be running.
        assert scheduler._head_starvation_job_id is None
        assert scheduler._head_starved_seconds(head) == 0.0

    async def test_preload_proceeds_when_within_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ample free VRAM and RAM the budget admits and the preload is sent."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 4000.0)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 1000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 2000  # 14000 MB free, covers 4000 + 2000
        process_map = ProcessMap({0: spare})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )

        assert scheduler.preload_models() is True
        assert spare.last_control_flag == HordeControlFlag.PRELOAD_MODEL

    async def test_preload_deferred_when_over_ram_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """VRAM fits but RAM does not: the preload is deferred and idle RAM is reclaimed."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 1000.0)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 50000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 1000  # ample free VRAM, so the VRAM gate passes
        # A second idle process holding a resident model: the RAM eviction candidate.
        resident = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: spare, 1: resident})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name="model_b",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=1,
        )

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )
        # Force a low available-RAM reading so the RAM budget defers deterministically.
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 8000.0)

        assert scheduler.preload_models() is False
        assert spare.last_control_flag != HordeControlFlag.PRELOAD_MODEL
        assert resident.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM

    async def test_disabled_budget_ignores_low_vram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the budget disabled, a low-VRAM device does not defer the preload."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 8000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 15500  # only 500 MB free
        process_map = ProcessMap({0: spare})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(enable_vram_budget=False, image_models_to_load=["model_a"]),
            max_concurrent=2,
            max_inference=2,
        )

        assert scheduler.preload_models() is True
        assert spare.last_control_flag == HordeControlFlag.PRELOAD_MODEL


class TestCheckJobCommittedReserve:
    """The VRAM budget holds back a committed reserve (e.g. in-flight post-processing) before admitting."""

    def test_committed_reserve_subtracts_from_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job that fits the raw free VRAM is deferred once the committed reserve is held back."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 4000.0)
        budget = VramBudget(reserve_mb=2000.0)
        job = make_job_pop_response("stable_diffusion")
        # 6000 covers 4000 + 2000 with nothing committed...
        assert budget.check_job(job, "x", free_vram_mb=6000.0).fits is True
        # ...but holding back 1500 MB of in-flight post-processing drops effective free to 4500 < 6000.
        verdict = budget.check_job(job, "x", free_vram_mb=6000.0, committed_reserve_mb=1500.0)
        assert verdict.fits is False
        assert verdict.available_mb == 4500.0

    def test_committed_reserve_defaults_to_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Callers (and existing tests) that omit the reserve keep the prior instantaneous behavior."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 4000.0)
        budget = VramBudget(reserve_mb=2000.0)
        job = make_job_pop_response("stable_diffusion")
        assert budget.check_job(job, "x", free_vram_mb=6000.0).fits is True


class TestPredictPostProcessingPeak:
    """The post-processing-phase predictor and its graceful fallback on an older hordelib."""

    def test_reads_phase_split_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The predictor returns the burden's post-processing-phase VRAM figure."""
        burden = Mock(vram_post_processing_mb=1500)
        monkeypatch.setattr(resource_budget, "_estimate_job_burden", lambda job, baseline: burden)
        job = make_job_pop_response("x")
        assert resource_budget.predict_job_post_processing_vram_mb(job, "stable_diffusion_xl") == 1500.0

    def test_none_when_estimate_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No burden estimate means the post-processing peak is unknown (None), not zero."""
        monkeypatch.setattr(resource_budget, "_estimate_job_burden", lambda job, baseline: None)
        job = make_job_pop_response("x")
        assert resource_budget.predict_job_post_processing_vram_mb(job, "x") is None

    def test_none_when_field_absent_on_old_hordelib(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pinned hordelib predating the phase-split lacks the field; the predictor degrades to None."""

        class _OldBurden:
            vram_mb = 5000
            ram_mb = 8000
            # No vram_post_processing_mb attribute (older BurdenEstimate).

        monkeypatch.setattr(resource_budget, "_estimate_job_burden", lambda job, baseline: _OldBurden())
        job = make_job_pop_response("x")
        assert resource_budget.predict_job_post_processing_vram_mb(job, "stable_diffusion_xl") is None


def _post_processing_process(process_id: int, job: object) -> object:
    """A mock inference process in the post-processing phase, holding ``job`` as its referenced job."""
    proc = make_mock_process_info(
        process_id,
        model_name="model_pp",
        state=HordeProcessState.INFERENCE_POST_PROCESSING,
    )
    proc.last_job_referenced = job  # pyrefly: ignore - assigning the tracked job for the reserve lookup
    return proc


class TestUpscaleFactorWiring:
    """The job's upscaler scale factor is resolved and inflates the predicted post-processing peak."""

    def test_factor_resolved_from_job_post_processing(self) -> None:
        """The max upscaler factor is read from the job payload; facefixers and an empty list contribute 1."""
        assert resource_budget._job_upscale_factor(make_mock_job(post_processing=["RealESRGAN_x2plus"])) == 2.0
        assert (
            resource_budget._job_upscale_factor(make_mock_job(post_processing=["RealESRGAN_x4plus", "GFPGAN"])) == 4.0
        )
        assert resource_budget._job_upscale_factor(make_mock_job(post_processing=[])) == 1.0

    def test_post_processing_peak_grows_with_factor(self) -> None:
        """End-to-end through the real hordelib: a 4x upscale reserves more than a 2x, both above zero."""
        job4 = make_mock_job(width=1024, height=1024, post_processing=["RealESRGAN_x4plus"])
        job2 = make_mock_job(width=1024, height=1024, post_processing=["RealESRGAN_x2plus"])
        peak4 = resource_budget.predict_job_post_processing_vram_mb(job4, "stable_diffusion_xl")
        peak2 = resource_budget.predict_job_post_processing_vram_mb(job2, "stable_diffusion_xl")
        assert peak4 is not None and peak2 is not None
        assert peak4 > peak2 > 0


class TestUpscaleDoesNotDriveResidency:
    """Regression: a post-processing upscaler must not flip an ordinary SDXL job into whole-card residency.

    A 4x upscaler's output-scaled activation belongs to the post-processing phase, which runs *after*
    sampling on the already-resident model. Folding it into the weight-residency forecast (as the old
    combined peak did) made a ~4.9GB SDXL job that merely requested an upscaler read as
    weight-dominant/needs-exclusive on a 16GB card; with a single inference process and no idle sibling to
    tear down, the head wedged until a save-our-ship soft reset. The residency forecast and the preload gate
    must key on the sampling-phase peak instead.
    """

    def test_sampling_peak_excludes_post_processing_activation(self) -> None:
        """Adding a 4x upscaler leaves the sampling peak unchanged; only the post-processing peak grows."""
        plain = make_mock_job(width=1024, height=1024, post_processing=[])
        upscaled = make_mock_job(width=1024, height=1024, post_processing=["RealESRGAN_x4plus"])
        sampling_plain = resource_budget.predict_job_sampling_vram_mb(plain, "stable_diffusion_xl")
        sampling_upscaled = resource_budget.predict_job_sampling_vram_mb(upscaled, "stable_diffusion_xl")
        assert sampling_plain is not None and sampling_upscaled is not None
        assert sampling_upscaled == sampling_plain
        # The upscaler's cost is real, it just lands in the post-processing phase rather than the sampling one.
        assert (resource_budget.predict_job_post_processing_vram_mb(upscaled, "stable_diffusion_xl") or 0) > 0

    def test_forecast_uses_sampling_not_combined_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The residency forecast keys on the sampling peak; an inflated combined peak does not flip it."""
        job = make_mock_job(width=1024, height=1024, post_processing=["RealESRGAN_x4plus"])
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda j, b: 4900.0)
        # Sampling-phase peak (weights + a modest sampling activation) that comfortably co-resides.
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda j, b: 6948.0)
        # Tripwire: the bridge.log combined peak (~17GB) would read as whole-card. If the forecast ever reverts
        # to the combined predictor this stub makes the assertions below fail.
        monkeypatch.setattr(resource_budget, "predict_job_vram_mb", lambda j, b: 17023.0)
        forecast = resource_budget.forecast_weight_streaming(
            job,
            "stable_diffusion_xl",
            free_now_mb=15005.0,
            total_vram_mb=16375.0,
            per_process_overhead_mb=1288.0,
            num_inference_processes=1,
            configured_reserve_floor_mb=2048.0,
        )
        assert forecast.fits_coresident is True
        assert forecast.needs_exclusive_residency is False
        assert forecast.requires_sibling_teardown is False


class TestCommittedPostProcessingReserve:
    """The scheduler sums the imminent post-processing peaks of in-flight jobs into a committed reserve."""

    async def test_zero_when_nothing_post_processing(self) -> None:
        """With no process in the post-processing phase, the reserve self-scales to zero."""
        idle = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: idle}),
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )
        assert scheduler._committed_post_processing_reserve_mb() == 0.0

    async def test_sums_in_flight_post_processing_peaks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each process in the post-processing phase contributes its job's predicted peak."""
        from horde_worker_regen.process_management import inference_scheduler as scheduler_module

        monkeypatch.setattr(scheduler_module, "predict_job_post_processing_vram_mb", lambda job, baseline: 1500.0)

        job_tracker = JobTracker()
        pp_job = make_job_pop_response("model_pp")
        await track_popped_job_async(job_tracker, pp_job)
        await job_tracker.mark_inference_started(pp_job)

        pp_proc = _post_processing_process(0, pp_job)
        idle = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)

        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: pp_proc, 1: idle}),
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )
        assert scheduler._committed_post_processing_reserve_mb() == 1500.0

    async def test_zero_when_feature_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reserve is suppressed when post_processing_budget_reserve_enabled is off."""
        from horde_worker_regen.process_management import inference_scheduler as scheduler_module

        monkeypatch.setattr(scheduler_module, "predict_job_post_processing_vram_mb", lambda job, baseline: 1500.0)

        job_tracker = JobTracker()
        pp_job = make_job_pop_response("model_pp")
        await track_popped_job_async(job_tracker, pp_job)
        await job_tracker.mark_inference_started(pp_job)

        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: _post_processing_process(0, pp_job)}),
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(
                enable_vram_budget=True,
                vram_reserve_mb=2000,
                ram_reserve_mb=4096,
                post_processing_budget_reserve_enabled=False,
                image_models_to_load=["model_pp"],
            ),
            max_concurrent=2,
            max_inference=2,
        )
        assert scheduler._committed_post_processing_reserve_mb() == 0.0


class TestPostProcessingOverlapGate:
    """The overlap concurrency bump is withheld when the device lacks headroom for the committed reserve."""

    def _overlap_scheduler(self, free_vram_mb: float) -> InferenceScheduler:
        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 16000 - int(free_vram_mb)
        return _make_inference_scheduler(
            process_map=ProcessMap({0: spare}),
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=4,
        )

    def test_bump_kept_with_ample_headroom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ample effective free VRAM the overlap bump is granted."""
        scheduler = self._overlap_scheduler(free_vram_mb=16000.0)
        monkeypatch.setattr(scheduler, "_committed_post_processing_reserve_mb", lambda: 0.0)
        # base path (lease off): max_concurrent (2) + post_processing_bump (1).
        assert scheduler._max_jobs_in_progress_allowed(1) == 3

    def test_bump_dropped_when_committed_reserve_eats_headroom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the committed reserve drops effective free below the bump floor, the bump is withheld."""
        scheduler = self._overlap_scheduler(free_vram_mb=16000.0)
        # Effective free = 16000 - 14000 = 2000, below the bump floor (max of 3000 and the 2000 reserve).
        monkeypatch.setattr(scheduler, "_committed_post_processing_reserve_mb", lambda: 14000.0)
        assert scheduler._max_jobs_in_progress_allowed(1) == 2

    def test_bump_unaffected_when_feature_disabled(self) -> None:
        """With the feature off, the overlap bump keeps its prior unconditional behavior."""
        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 15000  # only 1000 free, well under any floor
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: spare}),
            bridge_data=make_mock_bridge_data(
                enable_vram_budget=True,
                vram_reserve_mb=2000,
                ram_reserve_mb=4096,
                post_processing_budget_reserve_enabled=False,
                image_models_to_load=["model_a"],
            ),
            max_concurrent=2,
            max_inference=4,
        )
        # Feature off: the bump is retained regardless of free VRAM (prior behavior).
        assert scheduler._max_jobs_in_progress_allowed(1) == 3
