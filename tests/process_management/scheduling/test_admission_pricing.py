"""Each pure pricing function gives the scheduler's own answer over a snapshot of the same worker.

These are differential rows: while the scheduler still prices inline, the snapshot form must agree with it on
every configuration a gate can meet. When a scheduler method is removed, its row here becomes the function's
own specification.
"""

from __future__ import annotations

import sys
from unittest.mock import Mock

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import (
    predict_job_sampler_only_vram_mb,
    predict_job_sampling_vram_mb,
    predict_job_weight_mb,
)
from horde_worker_regen.process_management.resources.vram_footprints import (
    _MIN_OBSERVATIONS_FOR_MEASURED,
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    sampling_footprint_key,
)
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.clearance_lease import CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _slot(
    process_id: int,
    *,
    model: str | None,
    state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB,
    process_type: HordeProcessType = HordeProcessType.INFERENCE,
    reserved_mb: int | None = 3000,
) -> object:
    slot = make_mock_process_info(process_id, model_name=model, state=state, process_type=process_type)
    slot.process_reserved_mb = reserved_mb
    return slot


async def _worker(
    *,
    slots: dict[int, object],
    pending: list[str],
    in_progress: list[str] | None = None,
    learned: dict[tuple[str, FootprintStage], float] | None = None,
) -> tuple[InferenceScheduler, list[object]]:
    """A scheduler over the given slots and queue, with a learned store seeded as requested."""
    tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        process_map=ProcessMap(slots),  # type: ignore[arg-type]
        job_tracker=tracker,
        device_free_mb=6000.0,
        max_inference=2,
    )
    scheduler._process_map.get_free_vram_mb = Mock(return_value=6000.0)  # type: ignore[method-assign]
    scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)  # type: ignore[method-assign]
    jobs = []
    for model in pending:
        job = make_job_pop_response(model=model)
        await track_popped_job_async(tracker, job)
        jobs.append(job)
    for model in in_progress or []:
        job = make_job_pop_response(model=model)
        await track_popped_job_async(tracker, job)
        await mark_job_in_progress_async(tracker, job)
        jobs.append(job)
    if learned:
        store = LearnedFootprintStore()
        for (baseline, stage), peak_mb in learned.items():
            store.observe_peak(
                FootprintKey(model_baseline=baseline, resolution_bucket=None, platform=sys.platform, stage=stage),
                peak_mb,
            )
        scheduler.set_footprint_store(store)
    return scheduler, jobs


class TestForecastAndDeltas:
    """The forecast, the candidate delta and the co-resident maximum agree with the scheduler."""

    async def test_streaming_forecast_matches(self) -> None:
        """The full forecast dataclass is identical, resident credit and overheads included."""
        scheduler, jobs = await _worker(
            slots={0: _slot(0, model="stable_diffusion"), 1: _slot(1, model=None)},
            pending=["stable_diffusion", "other"],
        )
        snapshot = scheduler.snapshot()
        for job in jobs:
            baseline = scheduler._model_metadata.get_baseline(job.model)  # type: ignore[attr-defined]
            assert pricing.forecast_streaming(snapshot, job, baseline) == scheduler._forecast_streaming(  # type: ignore[attr-defined]
                job,
                baseline,
            )

    async def test_candidate_delta_and_learned_peak_match(self) -> None:
        """Resident credit on one slot and a learned raise on the sampling key both carry over."""
        scheduler, jobs = await _worker(
            slots={0: _slot(0, model="stable_diffusion"), 1: _slot(1, model=None)},
            pending=["stable_diffusion"],
            learned={("stable_diffusion_1", FootprintStage.SAMPLE): 9000.0},
        )
        snapshot = scheduler.snapshot()
        job = jobs[0]
        baseline = scheduler._model_metadata.get_baseline(job.model)  # type: ignore[attr-defined]
        for process_id in (0, 1, None):
            for disaggregated in (False, True):
                assert pricing.candidate_delta_mb(
                    snapshot,
                    job,
                    baseline,
                    process_id=process_id,
                    disaggregated=disaggregated,
                ) == scheduler._measured_admission_candidate_delta_mb(  # type: ignore[attr-defined]
                    job,
                    baseline,
                    process_id=process_id,
                    disaggregated=disaggregated,
                )

    async def test_a_job_beside_an_extra_large_sampler_is_priced_sampler_only(self) -> None:
        """Beside a running extra-large job both forms price the candidate at its sampler-only footprint."""
        reference = {
            "sdxl_model": make_mock_model_reference_record(
                "sdxl_model",
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            ),
            "flux_model": make_mock_model_reference_record(
                "flux_model",
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
            ),
        }
        tracker = JobTracker()
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: _slot(0, model="flux_model"), 1: _slot(1, model=None)}),  # type: ignore[arg-type]
            job_tracker=tracker,
            model_metadata=make_test_model_metadata(reference),
            max_inference=2,
        )
        candidate = make_job_pop_response(model="sdxl_model", width=1024, height=1024)
        running = make_job_pop_response(model="flux_model", width=1024, height=1024)
        await track_popped_job_async(tracker, candidate)
        await track_popped_job_async(tracker, running)
        await mark_job_in_progress_async(tracker, running)
        snapshot = scheduler.snapshot()
        baseline = scheduler._model_metadata.get_baseline("sdxl_model")  # type: ignore[attr-defined]
        for process_id in (0, 1, None):
            scheduler_mb = scheduler._measured_admission_candidate_delta_mb(  # type: ignore[attr-defined]
                candidate,
                baseline,
                process_id=process_id,
                disaggregated=False,
            )
            assert (
                pricing.candidate_delta_mb(snapshot, candidate, baseline, process_id=process_id, disaggregated=False)
                == scheduler_mb
            )
            assert scheduler_mb == predict_job_sampler_only_vram_mb(candidate, baseline)
        whole_job_mb = predict_job_sampling_vram_mb(candidate, baseline)
        assert whole_job_mb is not None and scheduler_mb is not None and scheduler_mb < whole_job_mb

    async def test_a_lora_job_priced_from_the_measurement_carries_the_seed_lora_delta(self) -> None:
        """Two jobs share one trusted measurement; the LoRA job prices above it by the seed's LoRA term.

        The sampling key has no LoRA axis, so the measurement cannot tell the two jobs apart. The job without
        LoRAs keeps the measured price unchanged.
        """
        reference = {
            "sdxl_model": make_mock_model_reference_record(
                "sdxl_model",
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            ),
        }
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: _slot(0, model=None)}),  # type: ignore[arg-type]
            job_tracker=JobTracker(),
            model_metadata=make_test_model_metadata(reference),
        )
        plain_job = make_job_pop_response(model="sdxl_model", width=1024, height=1024)
        lora_job = make_job_pop_response(
            model="sdxl_model",
            width=1024,
            height=1024,
            loras=[LorasPayloadEntry(name="detail_lora")],
        )
        baseline = scheduler._model_metadata.get_baseline("sdxl_model")  # type: ignore[attr-defined]
        plain_seed_mb = predict_job_sampling_vram_mb(plain_job, baseline)
        lora_seed_mb = predict_job_sampling_vram_mb(lora_job, baseline)
        assert plain_seed_mb is not None and lora_seed_mb is not None
        seed_lora_delta_mb = lora_seed_mb - plain_seed_mb
        assert seed_lora_delta_mb > 0.0
        key = sampling_footprint_key(plain_job, baseline, stage=FootprintStage.SAMPLE)
        assert key is not None
        assert key == sampling_footprint_key(lora_job, baseline, stage=FootprintStage.SAMPLE)
        floor_mb = predict_job_weight_mb(plain_job, baseline) or 0.0
        store = LearnedFootprintStore()
        for _ in range(_MIN_OBSERVATIONS_FOR_MEASURED):
            store.observe_peak(key, floor_mb + 100.0)
        scheduler.set_footprint_store(store)
        snapshot = scheduler.snapshot()
        measured_estimate_mb = store.measured_estimate_mb(key)
        assert measured_estimate_mb is not None

        plain_mb = pricing.learned_sampling_peak_mb(
            snapshot, plain_job, baseline, static_seed_mb=plain_seed_mb, stage=FootprintStage.SAMPLE
        )
        lora_mb = pricing.learned_sampling_peak_mb(
            snapshot, lora_job, baseline, static_seed_mb=lora_seed_mb, stage=FootprintStage.SAMPLE
        )

        assert floor_mb < plain_mb < plain_seed_mb, "the trusted measurement prices the job without LoRAs"
        assert plain_mb == pytest.approx(measured_estimate_mb - snapshot.context_constant_mb)
        assert lora_mb == pytest.approx(plain_mb + seed_lora_delta_mb)

    async def test_max_coresident_sizes_from_the_card_total_and_overheads(self) -> None:
        """The structural depth is the loader's context plus the marginal contexts the remaining total seats."""
        scheduler, _ = await _worker(slots={0: _slot(0, model=None)}, pending=[])
        scheduler.set_measured_per_process_overhead_mb(1000.0, device_index=0)
        scheduler.set_measured_marginal_overhead_mb(500.0, device_index=0)
        snapshot = scheduler.snapshot()
        # 16000 - 4000 - 1024 = 10976 budget; (10976 - 1000) // 500 = 19 extra contexts beside the loader's.
        assert pricing.max_coresident_for_peak_mb(snapshot, 4000.0, 1024.0) == 20
        # The loader's own context always counts, so a peak past the budget still sizes to one.
        assert pricing.max_coresident_for_peak_mb(snapshot, 15500.0, 1024.0) == 1

    async def test_unpausable_tenancy_matches(self) -> None:
        """A utilities lane on the card is charged the same way, per the lane-reclaim policy."""
        scheduler, _ = await _worker(
            slots={0: _slot(0, model=None), 5: _slot(5, model=None, process_type=HordeProcessType.UTILITIES)},
            pending=[],
        )
        snapshot = scheduler.snapshot()
        assert pricing.unpausable_tenancy_mb(snapshot, None) == scheduler._unpausable_tenancy_mb(None)  # type: ignore[attr-defined]


class TestReclaimPredicates:
    """The reclaim and teardown predicates across the guard combinations."""

    async def _reclaim_worker(self):  # noqa: ANN202
        unloading = _slot(2, model="queued")
        unloading.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM  # type: ignore[attr-defined]
        scheduler, _ = await _worker(
            slots={
                0: _slot(0, model="resident"),
                1: _slot(1, model="busy", state=HordeProcessState.INFERENCE_STARTING),
                2: unloading,
                3: _slot(3, model="queued"),
            },
            pending=["queued", "other"],
            in_progress=["busy"],
        )
        return scheduler.snapshot()

    async def test_the_target_slot_is_spared_unless_it_holds_another_model(self) -> None:
        """Making room on the target itself counts only when it holds a model other than the one being seated."""
        snapshot = await self._reclaim_worker()
        reclaimable = {
            make_room: pricing.has_reclaimable_idle_model(
                snapshot, 0, for_head_of_queue=False, device_index=None, make_room_for_model=make_room
            )
            for make_room in (None, "resident", "other")
        }
        # The busy, unloading and affordable queued-lookahead slots are never candidates for a follower, so
        # the target's own resident model is the only one an "other" load may displace.
        assert reclaimable == {None: False, "resident": False, "other": True}

    async def test_the_head_overrides_the_lookahead_guard(self) -> None:
        """A queued model's idle copy is spared for a follower but reclaimable for the head."""
        snapshot = await self._reclaim_worker()
        assert pricing.has_reclaimable_idle_model(snapshot, 3, for_head_of_queue=False, device_index=None) is True
        assert pricing.has_reclaimable_idle_model(snapshot, 0, for_head_of_queue=True, device_index=None) is True
        assert pricing.has_reclaimable_idle_model(snapshot, 0, for_head_of_queue=False, device_index=None) is False

    async def test_a_lane_that_owns_a_dispatched_job_is_not_reclaimable(self) -> None:
        """A lane idle by state with a resident model is no reclaim target while it owns a dispatched job.

        Reports from before the dispatch can read idle and restore the previous model as resident. The actuator
        skips that lane, so the mirror offering it would hold a rung that frees nothing.
        """
        owner = _slot(1, model="previous")
        scheduler, jobs = await _worker(slots={0: _slot(0, model=None), 1: owner}, pending=[], in_progress=["next"])
        assert pricing.has_reclaimable_idle_model(
            scheduler.snapshot(), 0, for_head_of_queue=True, device_index=None
        ), "without the ownership the idle resident is a reclaim target"

        owner.record_inference_ownership(jobs[0], attempt_ordinal=1)  # type: ignore[attr-defined]

        snapshot = scheduler.snapshot()
        assert pricing.has_reclaimable_idle_model(snapshot, 0, for_head_of_queue=True, device_index=None) is False
        assert (
            scheduler.unload_models_from_vram(scheduler._process_map[0], under_pressure=True, for_head_of_queue=True)
            is False
        )

    async def test_teardown_predicates(self) -> None:
        """The teardownable siblings, what the bare ones return, and the warm tenancy the head could reclaim."""
        components = _slot(2, model=None)
        components.held_components = [Mock()]  # type: ignore[attr-defined]
        scheduler, _jobs = await _worker(
            slots={
                0: _slot(0, model="head"),
                1: _slot(1, model=None, reserved_mb=800),
                2: components,
                3: _slot(3, model="busy", state=HordeProcessState.INFERENCE_STARTING),
            },
            pending=["head"],
            in_progress=["busy"],
        )
        snapshot = scheduler.snapshot()
        assert pricing.has_teardownable_idle_context(snapshot, 0, device_index=None) is True
        assert [slot.process_id for slot in pricing.teardownable_idle_contexts(snapshot, 0, device_index=None)] == [
            1,
            2,
        ]
        # Only the bare idle sibling returns its reservation plus a context; the one holding components does not.
        assert pricing.teardownable_idle_context_returns_mb(snapshot, 0, device_index=None) == [
            800.0 + pricing.marginal_or_seed_mb(snapshot, None),
        ]
        # Slot 2's components live in host RAM (no weights on the device), so they are not card tenancy.
        assert pricing.has_reclaimable_idle_tenancy(snapshot, "head", 0, device_index=None) is False

    async def test_idle_tenancy_counts_only_with_weights_on_the_device(self) -> None:
        """Warm components hold the card only while their weights are in VRAM; host-RAM caches return no VRAM."""
        warm = _slot(2, model="warm")
        warm.held_components = [Mock()]  # type: ignore[attr-defined]
        scheduler, _jobs = await _worker(slots={0: _slot(0, model="head"), 2: warm}, pending=["head"])

        assert pricing.has_reclaimable_idle_tenancy(scheduler.snapshot(), "head", 0, device_index=None) is False

        scheduler._horde_model_map.update_entry("warm", load_state=ModelLoadState.LOADED_IN_VRAM, process_id=2)

        snapshot = scheduler.snapshot()
        assert snapshot.slots[2].resident_weight_models == frozenset({"warm"})
        assert pricing.has_reclaimable_idle_tenancy(snapshot, "head", 0, device_index=None) is True
        assert pricing.has_reclaimable_idle_tenancy(snapshot, "head", 2, device_index=None) is False

    async def test_lookahead_affordability_matches(self) -> None:
        """The static lookahead fit reads the same head and the same reserve."""
        scheduler, _ = await _worker(slots={0: _slot(0, model="resident")}, pending=["head"])
        snapshot = scheduler.snapshot()
        assert pricing.coresident_lookahead_affordable(snapshot, "resident", device_index=None) is (
            scheduler._coresident_lookahead_affordable("resident", device_index=None)  # type: ignore[attr-defined]
        )


class TestDispatchStagingCharge:
    """Under the lease a dispatch charges its staging cost only while clearance is expected before the timeout.

    A staged child that waits out the lease-acquire timeout samples without a grant, so the staging charge is
    safe only when the running samplers will hand the card over well inside it.
    """

    def test_without_the_lease_both_gates_price_the_full_job(self) -> None:
        """Without the lease the dispatch is the VRAM moment, so neither charge applies."""
        assert pricing.dispatch_staging_charge_mb(clearance_lease_active=False) is None
        assert (
            pricing.overlap_staging_charge_mb(
                clearance_lease_active=False,
                running_remaining_sampling_seconds=(1.0,),
                incoming_load_seconds=1.0,
            )
            is None
        )

    def test_the_residency_gate_charges_staging_whatever_the_duration(self) -> None:
        """The residency gate's charge carries no duration term."""
        assert pricing.dispatch_staging_charge_mb(clearance_lease_active=True) == pricing.STAGING_ENCODE_VRAM_MB

    def test_a_short_wait_charges_staging(self) -> None:
        """A sampler finishing well inside the timeout lets the next job stage at the encode charge."""
        assert (
            pricing.overlap_staging_charge_mb(
                clearance_lease_active=True,
                running_remaining_sampling_seconds=(5.0, 3.0),
                incoming_load_seconds=4.0,
            )
            == pricing.STAGING_ENCODE_VRAM_MB
        )

    def test_a_wait_without_its_own_margin_prices_the_full_job(self) -> None:
        """The longest remaining sample plus the load must fit the timeout twice over."""
        half_timeout = CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS / 2.0
        assert (
            pricing.overlap_staging_charge_mb(
                clearance_lease_active=True,
                running_remaining_sampling_seconds=(1.0, half_timeout - 4.0),
                incoming_load_seconds=4.0,
            )
            is None
        )

    def test_an_unknown_remaining_time_prices_the_full_job(self) -> None:
        """A sampler with no trusted rate (staged, loading, under the progress floor) could run past the timeout."""
        assert (
            pricing.overlap_staging_charge_mb(
                clearance_lease_active=True,
                running_remaining_sampling_seconds=(2.0, None),
                incoming_load_seconds=1.0,
            )
            is None
        )

    def test_an_unmeasured_load_prices_the_full_job(self) -> None:
        """Without a measured weight-load time the wait cannot be bounded."""
        assert (
            pricing.overlap_staging_charge_mb(
                clearance_lease_active=True,
                running_remaining_sampling_seconds=(2.0,),
                incoming_load_seconds=None,
            )
            is None
        )


# ---- the partial-load seat's weight share


@pytest.mark.parametrize(
    ("seconds_per_step", "upload_mb_per_second"),
    [
        (None, 4000.0),
        (1.0, None),
        (1.0, 10.0),
        (1.0, 1_000_000.0),
        (0.0, 4000.0),
    ],
    ids=["no_step_time", "no_upload_rate", "implausibly_slow_upload", "implausibly_fast_upload", "zero_step_time"],
)
def test_the_seat_weight_share_falls_back_without_a_usable_measurement(
    seconds_per_step: float | None,
    upload_mb_per_second: float | None,
) -> None:
    """A missing or implausible step time or upload rate prices the fallback share."""
    fraction = pricing.partial_seat_weight_fraction(
        weights_mb=12000.0,
        seconds_per_step=seconds_per_step,
        upload_mb_per_second=upload_mb_per_second,
    )
    assert fraction == pricing.PARTIAL_SEAT_FALLBACK_WEIGHT_FRACTION


@pytest.mark.parametrize(
    ("seconds_per_step", "upload_mb_per_second", "expected"),
    [
        (1.0, 7200.0, 0.4),
        (1.0, 11000.0, pricing.PARTIAL_SEAT_MIN_WEIGHT_FRACTION),
        (1.37, 1100.0, pricing.PARTIAL_SEAT_FALLBACK_WEIGHT_FRACTION),
    ],
    ids=["streams_its_share_in_one_step", "fast_bus_floors_the_share", "slow_bus_caps_at_the_fallback"],
)
def test_the_seat_weight_share_streams_the_offloaded_weights_within_one_step(
    seconds_per_step: float,
    upload_mb_per_second: float,
    expected: float,
) -> None:
    """The share left off the card is what the bus moves in one step's time, clamped to the share's bounds."""
    fraction = pricing.partial_seat_weight_fraction(
        weights_mb=12000.0,
        seconds_per_step=seconds_per_step,
        upload_mb_per_second=upload_mb_per_second,
    )
    assert fraction == pytest.approx(expected)


def test_a_jobs_upload_rate_is_its_peak_weights_over_its_device_load_seconds() -> None:
    """The rate is what the job put on the card over the seconds its device loads took."""
    from hordelib.metrics import JobPhaseMetrics, JobVramFootprint, ModelLoadEvent

    def load(phase: str, seconds: float) -> ModelLoadEvent:
        return ModelLoadEvent(model_name="m", phase=phase, duration_seconds=seconds, timestamp=0.0)  # type: ignore[arg-type]

    metrics = JobPhaseMetrics(
        model_loads=[load("disk_to_ram", 9.0), load("ram_to_vram", 5.0), load("ram_to_vram", 3.0)],
        vram_footprint=JobVramFootprint(peak_resident_weights_mb=8000.0),
    )
    assert pricing.job_upload_mb_per_second(metrics) == pytest.approx(1000.0)

    assert pricing.job_upload_mb_per_second(JobPhaseMetrics(model_loads=[load("ram_to_vram", 5.0)])) is None
    no_upload = JobPhaseMetrics(
        model_loads=[load("disk_to_ram", 9.0)],
        vram_footprint=JobVramFootprint(peak_resident_weights_mb=8000.0),
    )
    assert pricing.job_upload_mb_per_second(no_upload) is None
