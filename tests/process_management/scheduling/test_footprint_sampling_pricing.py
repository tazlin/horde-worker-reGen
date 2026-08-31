"""The scheduler prices admission sampling peaks from the learned-footprint store, seeded by the static predictor.

These exercise the seam that closes the sampling-peak undershoot: the static per-model predictor is the seed,
and a measured SAMPLE-stage watermark for a job's (baseline, resolution, platform) can only raise the priced
peak. Pricing is proven at the admission surfaces (the disaggregated concurrent-sampling estimate and the
measured-overlay candidate delta), the raised peak is shown to flip a real arbiter verdict, and the
disaggregated observation seam is shown to record the pinned sampler's peak (and to ignore a zero reading).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.process_management.ipc.messages import (
    GENERATION_STATE,
    HordeSampleResultMessage,
    HordeTextEncodeResultMessage,
    SampleSliceResult,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
)
from horde_worker_regen.process_management.resources.vram_footprints import (
    FootprintKey,
    FootprintStage,
    ResolutionBucket,
)
from horde_worker_regen.process_management.scheduling import inference_scheduler as _sched_mod
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_testable_process_manager,
)

_MODEL = "SDXL 1.0"
_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl

_SEED_MB = 6158.0
"""The static whole-job sampling seed the predictor is pinned to return."""
_SAMPLER_ONLY_SEED_MB = 6158.0
"""The static UNet-only sampler seed for the disaggregated pricing branch."""
_WATERMARK_MB = 10500.0
"""A learned SAMPLE-stage watermark well above the seed (the ~11GB-against-6158MB undershoot the store closes)."""


def _reference() -> dict[str, object]:
    return {_MODEL: make_mock_model_reference_record(_MODEL, baseline=_BASELINE)}


def _make_manager() -> HordeWorkerProcessManager:
    """A testable manager with disaggregation on and the SDXL reference loaded into model metadata."""
    ref = _reference()
    pm = make_testable_process_manager(
        enable_pipeline_disaggregation=True,
        post_processing_lane_enabled=False,
        stable_diffusion_reference=ref,  # type: ignore[arg-type]
    )
    pm._model_metadata.set_reference(ref)  # type: ignore[arg-type]
    return pm


def _pin_static_predictors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin both sampling predictors to fixed seeds so a test observes only the store's overlay."""
    monkeypatch.setattr(_sched_mod, "predict_job_sampling_vram_mb", lambda _job, _baseline: _SEED_MB)
    monkeypatch.setattr(_sched_mod, "predict_job_sampler_only_vram_mb", lambda _job, _baseline: _SAMPLER_ONLY_SEED_MB)


def _job_info(width: int, height: int) -> SimpleNamespace:
    return SimpleNamespace(sdk_api_job_info=make_job_pop_response(model=_MODEL, width=width, height=height))


def _sample_key(bucket: ResolutionBucket) -> FootprintKey:
    """The monolithic whole-job SAMPLE key for the SDXL baseline at ``bucket``."""
    return FootprintKey(
        model_baseline=str(_BASELINE),
        resolution_bucket=bucket,
        platform=sys.platform,
        stage=FootprintStage.SAMPLE,
    )


def _isolated_key(bucket: ResolutionBucket) -> FootprintKey:
    """The disaggregated UNet-only SAMPLE_ISOLATED key for the SDXL baseline at ``bucket``."""
    return FootprintKey(
        model_baseline=str(_BASELINE),
        resolution_bucket=bucket,
        platform=sys.platform,
        stage=FootprintStage.SAMPLE_ISOLATED,
    )


class TestDisaggregatedEstimatePricing:
    """The disaggregated concurrent-sampling estimate reads the store, seeded by the static predictor."""

    def test_cold_key_returns_the_static_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With nothing observed, the estimate is the static seed unchanged."""
        _pin_static_predictors(monkeypatch)
        scheduler = _make_manager()._inference_scheduler
        assert scheduler.estimate_disaggregated_sampling_peak_mb(_job_info(1024, 1024)) == _SEED_MB  # type: ignore[arg-type]

    def test_learned_watermark_raises_the_estimate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A SAMPLE_ISOLATED watermark above the seed for the job's bucket raises the priced peak to it."""
        _pin_static_predictors(monkeypatch)
        pm = _make_manager()
        pm._learned_footprint_store.observe_peak(_isolated_key(ResolutionBucket.LE_1024), _WATERMARK_MB)
        assert (
            pm._inference_scheduler.estimate_disaggregated_sampling_peak_mb(_job_info(1024, 1024))  # type: ignore[arg-type]
            == _WATERMARK_MB
        )

    def test_a_different_resolution_bucket_is_unaffected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A watermark learned for the 1024 bucket does not raise a 512-bucket job (small jobs keep concurrency)."""
        _pin_static_predictors(monkeypatch)
        pm = _make_manager()
        pm._learned_footprint_store.observe_peak(_isolated_key(ResolutionBucket.LE_1024), _WATERMARK_MB)
        assert (
            pm._inference_scheduler.estimate_disaggregated_sampling_peak_mb(_job_info(512, 512))  # type: ignore[arg-type]
            == _SEED_MB
        )


class TestMeasuredOverlayDeltaPricing:
    """The measured-overlay candidate delta prices sampling work through the store on both branches."""

    def test_monolithic_branch_raised_by_watermark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole-job (monolithic) candidate delta is the seed cold, the watermark once learned."""
        _pin_static_predictors(monkeypatch)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        baseline = scheduler._model_metadata.get_baseline(_MODEL)

        cold = scheduler._measured_admission_candidate_delta_mb(job, baseline, process_id=None, disaggregated=False)
        assert cold == _SEED_MB

        pm._learned_footprint_store.observe_peak(_sample_key(ResolutionBucket.LE_1024), _WATERMARK_MB)
        raised = scheduler._measured_admission_candidate_delta_mb(job, baseline, process_id=None, disaggregated=False)
        assert raised == _WATERMARK_MB

    def test_disaggregated_branch_raised_by_watermark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sampler-only (disaggregated) candidate delta is likewise seeded then raised by the SAMPLE watermark."""
        _pin_static_predictors(monkeypatch)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        baseline = scheduler._model_metadata.get_baseline(_MODEL)

        cold = scheduler._measured_admission_candidate_delta_mb(job, baseline, process_id=None, disaggregated=True)
        assert cold == _SAMPLER_ONLY_SEED_MB

        pm._learned_footprint_store.observe_peak(_isolated_key(ResolutionBucket.LE_1024), _WATERMARK_MB)
        raised = scheduler._measured_admission_candidate_delta_mb(job, baseline, process_id=None, disaggregated=True)
        assert raised == _WATERMARK_MB


class TestPreloadCandidatePricing:
    """A preload is priced at its full marginal charge without the lease and at the staging charge under it."""

    def test_full_charge_without_the_lease(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the lease the preload is the VRAM moment, so it carries the whole sampling charge."""
        _pin_static_predictors(monkeypatch)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        pm.bridge_data.gpu_sampling_lease_enabled = False
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        baseline = scheduler._model_metadata.get_baseline(_MODEL)

        assert scheduler._preload_candidate_delta_mb(job, baseline, process_id=None) == _SEED_MB

    def test_staging_charge_under_the_lease(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under the lease the preload only stages in RAM, so it carries the staging encode charge."""
        _pin_static_predictors(monkeypatch)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        pm.bridge_data.gpu_sampling_lease_enabled = True
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        baseline = scheduler._model_metadata.get_baseline(_MODEL)

        assert (
            scheduler._preload_candidate_delta_mb(job, baseline, process_id=None) == _sched_mod._STAGING_ENCODE_VRAM_MB
        )


class TestStageIsolation:
    """Monolithic (SAMPLE) and disaggregated (SAMPLE_ISOLATED) peaks never cross-contaminate.

    Mixed operation is designed (a stage fault re-routes a disaggregated job monolithic), so a monolithic
    whole-job peak must not raise the disaggregated sampler-only estimate for the same baseline+bucket: doing so
    would, since watermarks are raise-only, permanently deny the second concurrent sampler.
    """

    def test_monolithic_sample_watermark_does_not_raise_the_disaggregated_estimate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 10500MB monolithic SAMPLE watermark leaves the disaggregated sampler estimate at its seed."""
        _pin_static_predictors(monkeypatch)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        baseline = scheduler._model_metadata.get_baseline(_MODEL)
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        pm._learned_footprint_store.observe_peak(_sample_key(ResolutionBucket.LE_1024), _WATERMARK_MB)

        assert scheduler.estimate_disaggregated_sampling_peak_mb(_job_info(1024, 1024)) == _SEED_MB  # type: ignore[arg-type]
        disagg_delta = scheduler._measured_admission_candidate_delta_mb(
            job,
            baseline,
            process_id=None,
            disaggregated=True,
        )
        assert disagg_delta == _SAMPLER_ONLY_SEED_MB

    def test_isolated_watermark_does_not_raise_the_monolithic_pricing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 10500MB SAMPLE_ISOLATED watermark leaves the monolithic candidate delta at its seed."""
        _pin_static_predictors(monkeypatch)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        baseline = scheduler._model_metadata.get_baseline(_MODEL)
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        pm._learned_footprint_store.observe_peak(_isolated_key(ResolutionBucket.LE_1024), _WATERMARK_MB)

        mono_delta = scheduler._measured_admission_candidate_delta_mb(
            job,
            baseline,
            process_id=None,
            disaggregated=False,
        )
        assert mono_delta == _SEED_MB


class TestGateIntegration:
    """The store-raised peak flips a real arbiter's concurrent-sampling verdict; a small-bucket job still admits."""

    # A 15288MB card: DISAGG_SAMPLE headroom = 15288 - 1288 overhead = 14000. With one 6158MB sampler in flight,
    # a second seed-sized (6158) sampling fits (12316 <= 14000) but a watermark-sized (10500) one does not
    # (16658 > 14000). The ~1700MB / ~2700MB margins clear any admission noise buffer either way.
    _TOTAL_MB = 15288.0
    _OVERHEAD_MB = 1288.0
    _ACTIVE_PEAK_MB = 6158.0

    def _cycle_state(self) -> DeviceVramState:
        return DeviceVramState(
            total_vram_mb=self._TOTAL_MB,
            baseline_mb=0.0,
            committed_vram_mb=0.0,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            num_loaded_inference_processes=1,
            per_process_overhead_mb=self._OVERHEAD_MB,
            marginal_mb=300.0,
            vram_reserve_mb=0.0,
            vae_lane_decode_spike_mb=0.0,
            active_sampling_peaks_total_mb=self._ACTIVE_PEAK_MB,
        )

    def _second_sample_verdict(self, arbiter: VramArbiter, peak_mb: float | None) -> VramDisposition:
        return arbiter.evaluate(
            VramRequest(
                kind=VramRequestKind.DISAGG_SAMPLE,
                job_label="disagg_sample",
                baseline=None,
                device_index=0,
                sampling_peak_mb=peak_mb,
                first_of_kind=False,
                active_sampling_peaks_total_mb=self._ACTIVE_PEAK_MB,
            ),
        ).disposition

    def test_seed_admits_but_learned_watermark_denies_the_second_sampling(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The static seed admits a second 1024 sampling; once the 1024 watermark is learned, it defers."""
        _pin_static_predictors(monkeypatch)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        arbiter = VramArbiter()

        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: self._cycle_state()}))
        seed_peak = scheduler.estimate_disaggregated_sampling_peak_mb(_job_info(1024, 1024))  # type: ignore[arg-type]
        assert self._second_sample_verdict(arbiter, seed_peak) == VramDisposition.FITS

        pm._learned_footprint_store.observe_peak(_isolated_key(ResolutionBucket.LE_1024), _WATERMARK_MB)
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: self._cycle_state()}))
        learned_peak = scheduler.estimate_disaggregated_sampling_peak_mb(_job_info(1024, 1024))  # type: ignore[arg-type]
        assert self._second_sample_verdict(arbiter, learned_peak) == VramDisposition.DEFER

    def test_small_bucket_job_still_admits_after_a_large_watermark_is_learned(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 512-bucket job keeps its small seed peak and admits alongside an in-flight sampler within headroom."""
        _pin_static_predictors(monkeypatch)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        pm._learned_footprint_store.observe_peak(_isolated_key(ResolutionBucket.LE_1024), _WATERMARK_MB)

        arbiter = VramArbiter()
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: self._cycle_state()}))
        small_peak = scheduler.estimate_disaggregated_sampling_peak_mb(_job_info(512, 512))  # type: ignore[arg-type]
        assert self._second_sample_verdict(arbiter, small_peak) == VramDisposition.FITS


class TestDisaggregatedObservation:
    """A disaggregated sample completion records the pinned sampler's peak; a zero reading records nothing."""

    def test_observe_method_records_under_the_isolated_key(self) -> None:
        """The observation helper folds a positive peak into the job's SAMPLE_ISOLATED key; zero is ignored."""
        pm = _make_manager()
        store = pm._learned_footprint_store
        job_info = _job_info(1024, 1024)

        pm._inference_scheduler.observe_disaggregated_sampling_peak(job_info, 0.0)  # type: ignore[arg-type]
        assert len(store) == 0

        pm._inference_scheduler.observe_disaggregated_sampling_peak(job_info, _WATERMARK_MB)  # type: ignore[arg-type]
        observation = store.get_observation(_isolated_key(ResolutionBucket.LE_1024))
        assert observation is not None
        assert observation.watermark_mb == _WATERMARK_MB
        # The monolithic SAMPLE key for the same baseline+bucket is untouched.
        assert store.get_observation(_sample_key(ResolutionBucket.LE_1024)) is None

    @pytest.mark.asyncio
    async def test_sample_completion_through_the_orchestrator_records_the_pinned_peak(self) -> None:
        """A sample result whose pinned sampler reports a peak lands that peak under the job's SAMPLE key."""
        pm = _make_manager()
        inference = make_mock_process_info(0, model_name=_MODEL, process_type=HordeProcessType.INFERENCE)
        pm._process_map[0] = inference
        pm._process_map[1] = make_mock_process_info(1, model_name=None, process_type=HordeProcessType.COMPONENT)
        pm._process_map[2] = make_mock_process_info(2, model_name=None, process_type=HordeProcessType.VAE_LANE)

        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        await pm._job_tracker.record_popped_job(job)
        routed = await pm._inference_scheduler._dispatch_disaggregated(
            job,
            inference,
            dispatched_device_index=None,
            degraded_dispatch=False,
        )
        assert routed is True

        orchestrator = pm._disaggregation_orchestrator
        orchestrator.tick()  # text-encode to the component process
        await orchestrator.handle_stage_result(
            HordeTextEncodeResultMessage(
                process_id=1,
                process_launch_identifier=0,
                info="",
                job_id=job.id_,
                positive_conditioning_bytes=b"pos",
                negative_conditioning_bytes=b"neg",
                state=GENERATION_STATE.ok,
            ),
        )
        orchestrator.tick()  # sample stage to the pinned sampler (process 0)

        # The pinned sampler's latest reported peak is what the observation records at sample completion.
        inference.process_peak_reserved_mb = 9800
        await orchestrator.handle_stage_result(
            HordeSampleResultMessage(
                process_id=0,
                process_launch_identifier=0,
                info="",
                results=[SampleSliceResult(job_id=job.id_, latent_bytes=b"latent", state=GENERATION_STATE.ok)],
            ),
        )

        observation = pm._learned_footprint_store.get_observation(_isolated_key(ResolutionBucket.LE_1024))
        assert observation is not None
        assert observation.watermark_mb == 9800.0

    @pytest.mark.asyncio
    async def test_sim_process_reporting_zero_peak_produces_no_entry(self) -> None:
        """A sample completion whose pinned sampler reports a zero peak records nothing (store semantics)."""
        pm = _make_manager()
        inference = make_mock_process_info(0, model_name=_MODEL, process_type=HordeProcessType.INFERENCE)
        pm._process_map[0] = inference
        pm._process_map[1] = make_mock_process_info(1, model_name=None, process_type=HordeProcessType.COMPONENT)
        pm._process_map[2] = make_mock_process_info(2, model_name=None, process_type=HordeProcessType.VAE_LANE)

        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        await pm._job_tracker.record_popped_job(job)
        await pm._inference_scheduler._dispatch_disaggregated(
            job,
            inference,
            dispatched_device_index=None,
            degraded_dispatch=False,
        )

        orchestrator = pm._disaggregation_orchestrator
        orchestrator.tick()
        await orchestrator.handle_stage_result(
            HordeTextEncodeResultMessage(
                process_id=1,
                process_launch_identifier=0,
                info="",
                job_id=job.id_,
                positive_conditioning_bytes=b"pos",
                negative_conditioning_bytes=b"neg",
                state=GENERATION_STATE.ok,
            ),
        )
        orchestrator.tick()

        inference.process_peak_reserved_mb = 0
        await orchestrator.handle_stage_result(
            HordeSampleResultMessage(
                process_id=0,
                process_launch_identifier=0,
                info="",
                results=[SampleSliceResult(job_id=job.id_, latent_bytes=b"latent", state=GENERATION_STATE.ok)],
            ),
        )

        assert len(pm._learned_footprint_store) == 0


def _make_monolithic_manager() -> HordeWorkerProcessManager:
    """A testable manager with disaggregation off, so a job's forecast prices the whole resident set."""
    ref = _reference()
    pm = make_testable_process_manager(
        enable_pipeline_disaggregation=False,
        post_processing_lane_enabled=False,
        stable_diffusion_reference=ref,  # type: ignore[arg-type]
    )
    pm._model_metadata.set_reference(ref)  # type: ignore[arg-type]
    return pm


def _resident_key(checkpoint: str) -> FootprintKey:
    """The per-checkpoint RESIDENT key (no resolution band: weights do not move with request size)."""
    return FootprintKey(
        model_baseline=str(_BASELINE),
        resolution_bucket=None,
        platform=sys.platform,
        stage=FootprintStage.RESIDENT,
        checkpoint=checkpoint,
    )


class TestResidentFootprintPricing:
    """The scheduler's resident-footprint seam reads the per-checkpoint RESIDENT watermark, context netted out.

    The store records a whole *device* charge (the slot's settled reservation plus its CUDA context), while
    the streaming forecast charges contexts separately from model bytes, so the seam must return the figure
    in context-exclusive terms or the context is paid for twice.
    """

    def test_cold_key_prices_nothing(self) -> None:
        """Nothing observed yields None, which leaves the forecast on its static seeds."""
        scheduler = _make_manager()._inference_scheduler
        assert scheduler._learned_resident_footprint_mb(_MODEL, _BASELINE) is None

    def test_watermark_is_returned_net_of_the_context_constant(self) -> None:
        """The measured device charge is offered back with the context term removed."""
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        pm._learned_footprint_store.observe_peak(_resident_key(_MODEL), 12000.0)
        assert scheduler._learned_resident_footprint_mb(_MODEL, _BASELINE) == pytest.approx(
            12000.0 - scheduler.resolved_context_constant_mb(),
        )

    def test_another_checkpoint_is_unaffected(self) -> None:
        """Resident weights are a property of the file, so one checkpoint's measurement never prices another."""
        pm = _make_manager()
        pm._learned_footprint_store.observe_peak(_resident_key("some other checkpoint"), 12000.0)
        assert pm._inference_scheduler._learned_resident_footprint_mb(_MODEL, _BASELINE) is None

    def test_unkeyable_model_prices_nothing(self) -> None:
        """A model or baseline that cannot be keyed leaves the forecast on its static seeds."""
        scheduler = _make_manager()._inference_scheduler
        assert scheduler._learned_resident_footprint_mb(None, _BASELINE) is None
        assert scheduler._learned_resident_footprint_mb(_MODEL, None) is None

    def test_forecast_footprint_rises_with_the_measurement(self) -> None:
        """End to end: the measured resident figure reaches the streaming forecast's footprint term.

        Read off the monolithic path, where a slot holds the whole resident set the observation records.
        """
        pm = _make_monolithic_manager()
        scheduler = pm._inference_scheduler
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        cold = scheduler._forecast_streaming(job, _BASELINE)
        assert cold.footprint_mb is not None

        pm._learned_footprint_store.observe_peak(
            _resident_key(_MODEL),
            cold.footprint_mb + scheduler.resolved_context_constant_mb() + 4000.0,
        )
        learned = scheduler._forecast_streaming(job, _BASELINE)
        assert learned.footprint_mb == pytest.approx(cold.footprint_mb + 4000.0)
        # The core-weight term is untouched: the measurement carries no weights-versus-support attribution.
        assert learned.weights_mb == cold.weights_mb


class TestMeasuredLoweringPricing:
    """A trusted measured sampling population lowers a static seed the hardware has disproved.

    The whole-job static seed prices every component of a combined checkpoint as co-resident; a backend that
    swaps support weights out before the diffusion weights land never co-hosts them, so the seed can sit
    several GB above any measured peak and hold the job at clearance (or price it past the achievable
    ceiling) for room it does not need. Once the model has produced a result here and the SAMPLE key holds
    enough observations, the margined measured figure governs downward, floored at the core weight figure.
    """

    @staticmethod
    def _pin_inflated_seed(monkeypatch: pytest.MonkeyPatch, *, seed_mb: float, weight_mb: float) -> None:
        """Pin the whole-job seed high and the core weight figure below it, the combined-checkpoint shape."""
        monkeypatch.setattr(_sched_mod, "predict_job_sampling_vram_mb", lambda _job, _baseline: seed_mb)
        monkeypatch.setattr(_sched_mod, "predict_job_weight_mb", lambda _job, _baseline: weight_mb)

    @staticmethod
    def _observe_sample_peaks(pm: HordeWorkerProcessManager, peak_mb: float, *, count: int = 5) -> None:
        """Feed ``count`` identical SAMPLE-stage observations (the measured-estimate minimum is five)."""
        for _ in range(count):
            pm._learned_footprint_store.observe_peak(_sample_key(ResolutionBucket.LE_1024), peak_mb)

    def test_measured_below_seed_lowers_the_monolithic_price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A measured population well under the seed reprices the job at the margined measurement."""
        self._pin_inflated_seed(monkeypatch, seed_mb=17000.0, weight_mb=12600.0)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        baseline = scheduler._model_metadata.get_baseline(_MODEL)
        self._observe_sample_peaks(pm, 13000.0)
        pm._job_tracker._models_with_results.add(_MODEL)

        expected = (
            pm._learned_footprint_store.measured_estimate_mb(
                _sample_key(ResolutionBucket.LE_1024),
            )
            - scheduler.resolved_context_constant_mb()
        )  # type: ignore[operator]
        priced = scheduler._measured_admission_candidate_delta_mb(job, baseline, process_id=None, disaggregated=False)
        assert priced == pytest.approx(expected)
        assert priced is not None and priced < 17000.0

    def test_model_without_a_result_keeps_the_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sampling keys are baseline-wide, so an unproven model never borrows a sibling's measurement."""
        self._pin_inflated_seed(monkeypatch, seed_mb=17000.0, weight_mb=12600.0)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        baseline = scheduler._model_metadata.get_baseline(_MODEL)
        self._observe_sample_peaks(pm, 13000.0)

        assert (
            scheduler._measured_admission_candidate_delta_mb(job, baseline, process_id=None, disaggregated=False)
            == 17000.0
        )

    def test_an_underobserved_key_keeps_the_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Below the measured-estimate observation minimum the conservative seed stays in charge."""
        self._pin_inflated_seed(monkeypatch, seed_mb=17000.0, weight_mb=12600.0)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        baseline = scheduler._model_metadata.get_baseline(_MODEL)
        self._observe_sample_peaks(pm, 13000.0, count=2)
        pm._job_tracker._models_with_results.add(_MODEL)

        assert (
            scheduler._measured_admission_candidate_delta_mb(job, baseline, process_id=None, disaggregated=False)
            == 17000.0
        )

    def test_the_core_weight_figure_floors_the_lowering(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A measured population below the core weights cannot price a job under what its weights occupy."""
        self._pin_inflated_seed(monkeypatch, seed_mb=17000.0, weight_mb=12600.0)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        baseline = scheduler._model_metadata.get_baseline(_MODEL)
        self._observe_sample_peaks(pm, 8000.0)
        pm._job_tracker._models_with_results.add(_MODEL)

        assert (
            scheduler._measured_admission_candidate_delta_mb(job, baseline, process_id=None, disaggregated=False)
            == 12600.0
        )

    def test_a_measurement_above_the_raised_figure_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The measured figure's only authority is downward; above the seed the raise-only watermark governs."""
        self._pin_inflated_seed(monkeypatch, seed_mb=17000.0, weight_mb=12600.0)
        pm = _make_manager()
        scheduler = pm._inference_scheduler
        job = make_job_pop_response(model=_MODEL, width=1024, height=1024)
        baseline = scheduler._model_metadata.get_baseline(_MODEL)
        self._observe_sample_peaks(pm, 18000.0)
        pm._job_tracker._models_with_results.add(_MODEL)

        assert (
            scheduler._measured_admission_candidate_delta_mb(job, baseline, process_id=None, disaggregated=False)
            == 18000.0
        )
