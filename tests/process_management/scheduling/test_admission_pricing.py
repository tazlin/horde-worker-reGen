"""Each pure pricing function gives the scheduler's own answer over a snapshot of the same worker.

These are differential rows: while the scheduler still prices inline, the snapshot form must agree with it on
every configuration a gate can meet. When a scheduler method is removed, its row here becomes the function's
own specification.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry
from hordelib.feature_impact import get_baseline_burden

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.device_memory_source import (
    DeviceMemoryReading,
    DeviceMemorySourceKind,
    ReadingProvenance,
)
from horde_worker_regen.process_management.resources.resource_budget import (
    predict_job_sampler_only_vram_mb,
    predict_job_sampling_vram_mb,
    predict_job_weight_mb,
)
from horde_worker_regen.process_management.resources.vram_arbiter import (
    MeasuredVramSnapshot,
    VramArbiter,
    VramRequestKind,
)
from horde_worker_regen.process_management.resources.vram_footprints import (
    _MIN_OBSERVATIONS_FOR_MEASURED,
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    ResolutionBucket,
    sampling_footprint_key,
)
from horde_worker_regen.process_management.scheduling.admission import materialization, preload, pricing
from horde_worker_regen.process_management.scheduling.clearance_lease import CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
from horde_worker_regen.process_management.scheduling.governance.whole_card import DominanceRung
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.liveness._session_replay import SESSION_FOOTPRINTS_PATH, load_footprint_store
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
        """Slot 3 is the follower's ready seat ("queued"). The head ("other") has no lane holding its model."""
        unloading = _slot(2, model="queued")
        unloading.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM  # type: ignore[attr-defined]
        scheduler, _ = await _worker(
            slots={
                0: _slot(0, model="resident"),
                1: _slot(1, model="busy", state=HordeProcessState.INFERENCE_STARTING),
                2: unloading,
                3: _slot(3, model="queued"),
            },
            pending=["other", "queued"],
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
        # The busy, unloading and ready-seat slots are never candidates for a follower, so the target's own
        # resident model is the only one an "other" load may displace.
        assert reclaimable == {None: False, "resident": False, "other": True}

    async def test_the_head_overrides_the_lookahead_guard(self) -> None:
        """A follower's ready seat is spared for a follower's reclaim but reclaimable for the head escalation.

        `pricing.sweep_spares_idle_copy` keeps every pending job's ready seat for an ordinary sweep, and under the
        escalation only the seat the head itself claims. The head here claims none, so the follower's seat yields
        to the starved head, which has priority for the room.
        """
        snapshot = await self._reclaim_worker()
        assert 3 in pricing.ready_seat_process_ids(snapshot)
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

    async def test_lookahead_affordability_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The static lookahead fit, and the predicate and the sweep agree on the copy it decides.

        This row specifies the function. A lookahead copy fits while the card total less the reserve, the head's
        sampling peak and the copy's footprint stays at or above zero. A queued copy with no ready seat (its job
        held by a dispatch gate) is spared exactly while it fits, and `pricing.has_reclaimable_idle_model` offers it
        exactly when `unload_models_from_vram` evicts it, because both ask `pricing.sweep_spares_idle_copy`.
        """
        footprint_mb = 6000.0
        monkeypatch.setattr(pricing, "predict_job_footprint_mb", lambda job, baseline: footprint_mb)
        for head_peak_mb in (2000.0, 9000.0):
            monkeypatch.setattr(pricing, "predict_job_sampling_vram_mb", lambda job, baseline, peak=head_peak_mb: peak)
            lookahead_holder = _slot(1, model="lookahead")
            scheduler, jobs = await _worker(
                slots={0: _slot(0, model=None), 1: lookahead_holder},
                pending=["head", "lookahead"],
            )
            scheduler._dispatch_holds.note_pp_defer(str(jobs[1].id_), deferred=True)  # type: ignore[attr-defined]
            snapshot = scheduler.snapshot()
            total_mb = snapshot.card(None).total_vram_mb
            assert total_mb is not None
            fits = total_mb - snapshot.vram_reserve_mb - head_peak_mb - footprint_mb >= 0
            assert fits is (head_peak_mb == 2000.0), "the two peaks straddle the card's room"
            assert pricing.ready_seat_process_ids(snapshot) == frozenset()

            assert pricing.coresident_lookahead_affordable(snapshot, "lookahead", device_index=None) is fits
            assert pricing.has_reclaimable_idle_model(snapshot, 0, for_head_of_queue=False, device_index=None) is (
                not fits
            )
            assert scheduler.unload_models_from_vram(scheduler._process_map[0], under_pressure=True) is (not fits)
            evicted = lookahead_holder.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM  # type: ignore[attr-defined]
            assert evicted is (not fits)


class TestDominantRunCopy:
    """A rung-1 run's copy with a next job is spared by the sweep and by preload targeting, whoever needs the room."""

    async def _run_worker(self, *, pending: list[str]):  # noqa: ANN202
        """Slot 1 holds the run's model idle; slot 0 is the lane a head for another model would be seated on."""
        run_lane = _slot(1, model="run")
        scheduler, jobs = await _worker(slots={0: _slot(0, model=None), 1: run_lane}, pending=pending)
        scheduler._whole_card_ledger.record_grant(
            None,
            model="run",
            forecast=None,
            cooldown_until=0.0,
            now=scheduler._clock(),
            rung=DominanceRung.CANNOT_CO_SAMPLE,
        )
        return scheduler, jobs, run_lane

    async def test_a_pending_job_or_the_claim_gives_the_run_a_next_job(self) -> None:
        """The copy belongs to the run while a job for its model waits or its claim stands, and not otherwise."""
        scheduler, jobs, _run_lane = await self._run_worker(pending=["other", "run"])
        snapshot = scheduler.snapshot()
        assert snapshot.dominant_runs == {"run": None}
        assert pricing.copy_belongs_to_dominant_run(snapshot, "run", None) is True
        assert pricing.copy_belongs_to_dominant_run(snapshot, "other", None) is False
        assert pricing.copy_belongs_to_dominant_run(snapshot, "run", None, besides_job_id=str(jobs[1].id_)) is (
            snapshot.pop_claim_model == "run"
        ), "the job dispatching now is not its own successor; only the claim is left"

        scheduler._whole_card_ledger.note_terminal_fault("run", now=scheduler._clock())
        released = scheduler.snapshot()
        assert released.pop_claim_model is None
        assert pricing.copy_belongs_to_dominant_run(released, "run", None, besides_job_id=str(jobs[1].id_)) is False

    async def test_a_teardown_residency_is_not_a_run(self) -> None:
        """A rung-2 residency is not a run, and its model is left to the readers that own the teardown."""
        scheduler, _jobs = await _worker(slots={1: _slot(1, model="heavy")}, pending=["heavy"])
        scheduler._whole_card_ledger.record_grant(None, model="heavy", forecast=None, cooldown_until=0.0, now=0.0)
        snapshot = scheduler.snapshot()
        assert snapshot.dominant_runs == {}
        assert pricing.copy_belongs_to_dominant_run(snapshot, "heavy", None) is False

    async def test_the_sweep_spares_the_run_copy_for_the_head_and_takes_it_once_the_run_has_no_next_job(
        self,
    ) -> None:
        """The head escalation does not take a run's copy, which is an idle resident once no job or claim is left.

        The predicate and the actuator agree, since both ask `pricing.sweep_spares_idle_copy`.
        """
        scheduler, _jobs, run_lane = await self._run_worker(pending=["other", "run"])
        snapshot = scheduler.snapshot()
        assert pricing.has_reclaimable_idle_model(snapshot, 0, for_head_of_queue=True, device_index=None) is False
        assert (
            scheduler.unload_models_from_vram(scheduler._process_map[0], under_pressure=True, for_head_of_queue=True)
            is False
        )
        assert run_lane.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM  # type: ignore[attr-defined]

        scheduler, _jobs, run_lane = await self._run_worker(pending=["other"])
        scheduler._whole_card_ledger.note_terminal_fault("run", now=scheduler._clock())
        snapshot = scheduler.snapshot()
        assert pricing.copy_belongs_to_dominant_run(snapshot, "run", None) is False
        assert pricing.has_reclaimable_idle_model(snapshot, 0, for_head_of_queue=True, device_index=None) is True
        assert (
            scheduler.unload_models_from_vram(scheduler._process_map[0], under_pressure=True, for_head_of_queue=True)
            is True
        )
        assert run_lane.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM  # type: ignore[attr-defined]

    async def _stranger_beside_run(self, *, room_closable: bool):  # noqa: ANN202
        """The run's copy idle on slot 1, a different-model job dispatched to slot 0, the run's next job held.

        The next job's dispatch hold carries the room breakdown the reconciliation gate records, which the
        permitted reclaim rungs either cover or do not.
        """
        run_lane = _slot(1, model="run")
        stranger_lane = _slot(0, model="other", state=HordeProcessState.INFERENCE_STARTING)
        scheduler, jobs = await _worker(
            slots={0: stranger_lane, 1: run_lane},
            pending=["run"],
            in_progress=["other"],
        )
        stranger_lane.record_inference_ownership(jobs[1], attempt_ordinal=1)  # type: ignore[attr-defined]
        scheduler._whole_card_ledger.record_grant(
            None,
            model="run",
            forecast=None,
            cooldown_until=0.0,
            now=scheduler._clock(),
            rung=DominanceRung.CANNOT_CO_SAMPLE,
        )
        scheduler._dispatch_holds.note_hold(
            str(jobs[0].id_),
            reclaim_applied=False,
            room_inputs={"room_closable": room_closable},
        )
        return scheduler

    async def test_a_copy_whose_next_job_cannot_be_seated_is_not_spared_for_it(self) -> None:
        """The run's next job held with no reclaim able to release it does not keep the copy from a waiting stranger.

        The stranger was cleared beside the run and waits for room only the copy can return, while the run's job
        is held against the stranger's booking. Spared, the copy holds both until the lease times the stranger
        out. The run's standing claim does not keep it either, since a further pop of the model queues behind
        the same hold.
        """
        scheduler = await self._stranger_beside_run(room_closable=False)
        snapshot = scheduler.snapshot()
        claimed = dataclasses.replace(snapshot, pop_claim_model="run")
        for view in (snapshot, claimed):
            assert pricing.copy_belongs_to_dominant_run(view, "run", None) is False
            assert (
                pricing.sweep_spares_idle_copy(
                    view, view.slots[1], target_process_id=0, for_head_of_queue=True, device_index=None
                )
                is False
            )
            assert pricing.has_reclaimable_idle_model(view, 0, for_head_of_queue=True, device_index=None) is True

    async def test_a_copy_whose_next_job_waits_on_closable_room_stays_with_the_run(self) -> None:
        """A next job held on room the permitted rungs can close keeps the copy, since it starts once they land."""
        scheduler = await self._stranger_beside_run(room_closable=True)
        snapshot = scheduler.snapshot()
        assert pricing.copy_belongs_to_dominant_run(snapshot, "run", None) is True
        assert pricing.has_reclaimable_idle_model(snapshot, 0, for_head_of_queue=True, device_index=None) is False

    async def test_a_run_lane_idle_between_two_run_jobs_keeps_its_copy(self) -> None:
        """The ordinary run: the lane idle between two run jobs, nothing cleared beside it, the next job unheld."""
        scheduler, _jobs, _run_lane = await self._run_worker(pending=["run"])
        snapshot = scheduler.snapshot()
        assert pricing.copy_belongs_to_dominant_run(snapshot, "run", None) is True
        assert (
            pricing.sweep_spares_idle_copy(
                snapshot, snapshot.slots[1], target_process_id=0, for_head_of_queue=True, device_index=None
            )
            is True
        )
        claim_only = dataclasses.replace(
            (await self._run_worker(pending=[]))[0].snapshot(),
            pop_claim_model="run",
        )
        assert pricing.copy_belongs_to_dominant_run(claim_only, "run", None) is True, (
            "with no queued successor the next pop under the claim is the run's next job"
        )

    async def test_a_run_lane_is_no_preload_target_for_another_model(self) -> None:
        """The lane holding a run's copy is committed to the run's next job, so another model's preload has no target.

        Once the run has no next job the lane is an ordinary target again.
        """
        scheduler, jobs = await _worker(slots={1: _slot(1, model="run")}, pending=["other", "run"])
        scheduler._whole_card_ledger.record_grant(
            None, model="run", forecast=None, cooldown_until=0.0, now=0.0, rung=DominanceRung.CANNOT_CO_SAMPLE
        )
        snapshot = scheduler.snapshot()
        assert 1 in preload.lanes_committed_elsewhere(snapshot, "other")
        assert 1 not in preload.lanes_committed_elsewhere(snapshot, "run")
        assert preload.select_preload_target(snapshot, str(jobs[0].id_)) is None

        scheduler._whole_card_ledger.record_run_end(None)
        assert preload.select_preload_target(scheduler.snapshot(), str(jobs[0].id_)) == 1

    async def _warm_run_copy(self, *, pending: list[str]):  # noqa: ANN202
        """The run's copy on slot 1 with its weights on the device and warm components held between jobs."""
        scheduler, jobs, run_lane = await self._run_worker(pending=pending)
        run_lane.held_components = [Mock()]  # type: ignore[attr-defined]
        scheduler._horde_model_map.update_entry("run", load_state=ModelLoadState.LOADED_IN_VRAM, process_id=1)
        return scheduler, jobs

    @staticmethod
    def _head_room(scheduler: InferenceScheduler, head_job_id: str):  # noqa: ANN205
        """The head's preload request on slot 0, and the admission room the arbiter decomposes it into."""
        snapshot = scheduler.snapshot()
        arbiter = VramArbiter()
        request = materialization.build_materialization_request(
            snapshot,
            head_job_id,
            0,
            arbiter=arbiter,
            is_head_of_queue=True,
            head_outstanding_mb=None,
            candidate_delta_override_mb=_HEAD_CANDIDATE_MB,
            kind=VramRequestKind.PRELOAD,
        ).request
        reading = DeviceMemoryReading(
            free_mb=6000.0,
            total_mb=16000.0,
            source=DeviceMemorySourceKind.NVML,
            provenance=ReadingProvenance.DEVICE_WIDE,
        )
        state = scheduler.build_vram_arbiter_device_state(None, device_reading=reading)
        arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
        room = VramArbiter._room(request, state, arbiter._measured(request, state))
        assert room is not None
        return snapshot, request, room

    async def test_a_spared_run_copy_is_not_room_for_the_head(self) -> None:
        """The eviction mirror, the tenancy predicate and the admission room agree about the run's copy.

        The sweep does not take the copy while the run has a next job, so no rung returns its room for the head.
        Counted as room, it made a deficit only its eviction could close read as closable, and the lane rungs
        were spent on it. Counted as reclaimable tenancy, it told the head the card still had something to give.
        """
        scheduler, jobs = await self._warm_run_copy(pending=["other", "run"])
        snapshot, request, room = self._head_room(scheduler, str(jobs[0].id_))

        assert pricing.has_reclaimable_idle_model(snapshot, 0, for_head_of_queue=True, device_index=None) is False
        assert pricing.has_reclaimable_idle_tenancy(snapshot, "other", 0, device_index=None) is False
        assert request.has_reclaimable_idle_tenancy is False
        assert room.reclaimable_mb == 0.0, room.describe()
        assert not room.closable, room.describe()

    async def test_the_copy_is_room_again_once_the_run_has_no_next_job(self) -> None:
        """With the run's work drained the copy is an ordinary idle resident, and all three readers count it."""
        scheduler, jobs = await self._warm_run_copy(pending=["other"])
        scheduler._whole_card_ledger.note_terminal_fault("run", now=scheduler._clock())
        snapshot, request, room = self._head_room(scheduler, str(jobs[0].id_))

        assert pricing.has_reclaimable_idle_model(snapshot, 0, for_head_of_queue=True, device_index=None) is True
        assert pricing.has_reclaimable_idle_tenancy(snapshot, "other", 0, device_index=None) is True
        assert request.has_reclaimable_idle_tenancy is True
        assert room.reclaimable_mb > 0.0, room.describe()


_HEAD_CANDIDATE_MB = 9000.0
"""A head candidate the 6000 MB reading cannot seat, so the room has a deficit for the run's copy to cover."""


_FLUX = KNOWN_IMAGE_GENERATION_BASELINE.flux_1
"""A family whose text encoder is far heavier than SDXL's."""


class TestStagingCharge:
    """A staged job charges its family's support weights plus the encode allowance, floored at SDXL's figure."""

    def test_sdxl_stages_at_2048(self) -> None:
        """SDXL's figure is the one every staging charge was sized against, so its scheduling is unchanged."""
        assert pricing.staging_charge_mb(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl) == 2048.0

    def test_flux_stages_at_its_family_figure(self) -> None:
        """Flux's staging charge is its own support weights plus the allowance, well above SDXL's."""
        flux_burden = get_baseline_burden(_FLUX.value)
        assert flux_burden is not None
        expected_mb = flux_burden.vram_support_weights_mb + pricing.STAGING_ENCODE_ALLOWANCE_MB
        assert expected_mb > 2048.0
        assert pricing.staging_charge_mb(_FLUX) == expected_mb

    @pytest.mark.parametrize("baseline", [None, "not_a_known_family"])
    def test_an_unknown_family_stages_at_the_sdxl_figure(self, baseline: str | None) -> None:
        """A job whose family hordelib has no seed for is charged the SDXL figure."""
        assert pricing.staging_charge_mb(baseline) == pricing.staging_charge_mb(
            KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
        )

    def test_a_lighter_family_is_floored_at_the_sdxl_figure(self) -> None:
        """SD1's support weights are below SDXL's, but its encode still allocates the working set."""
        assert pricing.staging_charge_mb(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1) == 2048.0


class TestDispatchStagingCharge:
    """Under the lease a dispatch charges its staging cost only while clearance is expected before the timeout.

    A staged child that waits out the lease-acquire timeout samples without a grant, so the staging charge is
    safe only when the running samplers will hand the card over well inside it.
    """

    def test_without_the_lease_both_gates_price_the_full_job(self) -> None:
        """Without the lease the dispatch is the VRAM moment, so neither charge applies."""
        assert pricing.dispatch_staging_charge_mb(clearance_lease_active=False, baseline=_FLUX) is None
        assert (
            pricing.overlap_staging_charge_mb(
                clearance_lease_active=False,
                baseline=_FLUX,
                running_remaining_sampling_seconds=(1.0,),
                incoming_load_seconds=1.0,
            )
            is None
        )

    def test_the_residency_gate_charges_staging_whatever_the_duration(self) -> None:
        """The residency gate's charge carries no duration term."""
        assert pricing.dispatch_staging_charge_mb(clearance_lease_active=True, baseline=_FLUX) == (
            pricing.staging_charge_mb(_FLUX)
        )

    def test_a_short_wait_charges_staging(self) -> None:
        """A sampler finishing well inside the timeout lets the next job stage at the encode charge."""
        assert pricing.overlap_staging_charge_mb(
            clearance_lease_active=True,
            baseline=_FLUX,
            running_remaining_sampling_seconds=(5.0, 3.0),
            incoming_load_seconds=4.0,
        ) == pricing.staging_charge_mb(_FLUX)

    def test_a_wait_without_its_own_margin_prices_the_full_job(self) -> None:
        """The longest remaining sample plus the load must fit the timeout twice over."""
        half_timeout = CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS / 2.0
        assert (
            pricing.overlap_staging_charge_mb(
                clearance_lease_active=True,
                baseline=_FLUX,
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
                baseline=_FLUX,
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
                baseline=_FLUX,
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


_REPLAY_CATALOGUE: dict[str, str] = {
    "Flux.1-Schnell fp8 (Compact)": "flux_1",
    "Krea2-Turbo_fp8": "krea2_turbo",
    "Z-Image-Turbo": "z_image_turbo",
    "Qwen-Image": "qwen_image",
    "Stable Cascade 1.0": "stable_cascade",
    "AlbedoBase XL (SDXL)": "stable_diffusion_xl",
    "stable_diffusion": "stable_diffusion_1",
    "Anima-Turbo-v1.1": "anima",
}
"""One offered model per baseline the replay fixture's store measured, keyed to its baseline id."""


def _replay_dominance(store: LearnedFootprintStore, *, foreign_floor_mb: float) -> pricing.CardDominance:
    """Return the verdict over the replay catalogue on the replay fixture's 24 GB card figures."""
    inputs = pricing.DominanceInputs(
        offered_models=frozenset(_REPLAY_CATALOGUE),
        whole_card_pins=frozenset(),
        total_vram_mb=24074.0,
        first_context_mb=462.0,
        marginal_context_mb=488.0,
        admission_noise_mb=602.0,
        foreign_floor_bucket=None,
        sample_observation_counts=(),
    )
    peaks = [pricing.model_sampling_peak(store, model, baseline) for model, baseline in _REPLAY_CATALOGUE.items()]
    return pricing.decide_card_dominance(inputs, foreign_floor_mb=foreign_floor_mb, peaks=peaks)


class TestDominanceVerdict:
    """The two-rung verdict over a stated catalogue, its peak sources and its fallback."""

    @pytest.fixture
    def replay_store(self, tmp_path: Path) -> LearnedFootprintStore:
        """Return the replay fixture's learned store from a 24 GB card, keyed for this platform."""
        return load_footprint_store(SESSION_FOOTPRINTS_PATH, scratch_directory=tmp_path)

    def test_the_measured_tenant_puts_four_models_on_rung_one_and_qwen_on_rung_two(
        self,
        replay_store: LearnedFootprintStore,
    ) -> None:
        """With the 3,300 MB tenant no offered model samples beside Flux, Krea2, Z-Image or Cascade.

        Each partner is priced at its most-observed band, so Cascade's 15,532 MB at le_1024 misses beside SD1.5's
        le_512 figure by about 130 MB. Qwen cannot sample beside even one context.
        """
        dominance = _replay_dominance(replay_store, foreign_floor_mb=3300.0)
        rungs = {model: verdict.rung for model, verdict in dominance.verdicts.items()}

        assert rungs == {
            "Flux.1-Schnell fp8 (Compact)": DominanceRung.CANNOT_CO_SAMPLE,
            "Krea2-Turbo_fp8": DominanceRung.CANNOT_CO_SAMPLE,
            "Z-Image-Turbo": DominanceRung.CANNOT_CO_SAMPLE,
            "Stable Cascade 1.0": DominanceRung.CANNOT_CO_SAMPLE,
            "Qwen-Image": DominanceRung.CANNOT_CO_RESIDE,
            "AlbedoBase XL (SDXL)": DominanceRung.NONE,
            "stable_diffusion": DominanceRung.NONE,
            "Anima-Turbo-v1.1": DominanceRung.NONE,
        }
        flux = dominance.verdicts["Flux.1-Schnell fp8 (Compact)"].peak
        assert flux.band is ResolutionBucket.LE_1024
        assert flux.observation_count > 0

    def test_without_the_tenant_an_sd15_job_fits_beside_flux(self, replay_store: LearnedFootprintStore) -> None:
        """At a zero floor Flux has a partner, and Qwen still cannot sample beside one context."""
        dominance = _replay_dominance(replay_store, foreign_floor_mb=0.0)

        assert dominance.verdicts["Flux.1-Schnell fp8 (Compact)"].rung is DominanceRung.NONE
        assert dominance.verdicts["Qwen-Image"].rung is DominanceRung.CANNOT_CO_RESIDE

    def test_rung_two_needs_the_peak_beside_one_context_to_overflow(self) -> None:
        """Rung 2 is ``total - peak - both contexts - floor - margin < 0``. Rung 1 has no fitting partner."""
        charges_mb = 462.0 + 488.0 + 3300.0 + 602.0
        assert (
            pricing.dominance_rung(19300.0, partner_peaks_mb=[], total_vram_mb=24074.0, charges_mb=charges_mb)
            is DominanceRung.CANNOT_CO_RESIDE
        )
        assert (
            pricing.dominance_rung(19200.0, partner_peaks_mb=[3821.0], total_vram_mb=24074.0, charges_mb=charges_mb)
            is DominanceRung.CANNOT_CO_SAMPLE
        )
        assert (
            pricing.dominance_rung(9475.0, partner_peaks_mb=[3821.0], total_vram_mb=24074.0, charges_mb=charges_mb)
            is DominanceRung.NONE
        )

    def test_an_unpriced_figure_never_withholds_a_co_sample(self) -> None:
        """An unpriced model has no rung, and an unpriced partner reads as fitting beside it."""
        assert (
            pricing.dominance_rung(None, partner_peaks_mb=[], total_vram_mb=24074.0, charges_mb=0.0)
            is DominanceRung.NONE
        )
        assert (
            pricing.dominance_rung(20000.0, partner_peaks_mb=[None], total_vram_mb=24074.0, charges_mb=2000.0)
            is DominanceRung.NONE
        )

    def test_an_unmeasured_model_is_priced_at_the_le_1024_sampling_seed(self) -> None:
        """With no observation the peak is the static sampling seed of a plain 1024x1024 job of the model."""
        peak = pricing.model_sampling_peak(LearnedFootprintStore(), "Flux.1-Schnell fp8 (Compact)", "flux_1")

        assert peak.band is ResolutionBucket.LE_1024
        assert peak.observation_count == 0
        job = make_job_pop_response("Flux.1-Schnell fp8 (Compact)", width=1024, height=1024)
        assert peak.peak_mb == predict_job_sampling_vram_mb(job, "flux_1")

    def test_flux_on_a_16_gb_card_is_rung_two_on_its_static_seed(self) -> None:
        """The 16 GB card the whole-card goldens run on cannot hold Flux's seed beside one context."""
        inputs = pricing.DominanceInputs(
            offered_models=frozenset({"Flux.1-Schnell fp8 (Compact)", "stable_diffusion"}),
            whole_card_pins=frozenset(),
            total_vram_mb=16384.0,
            first_context_mb=1354.0,
            marginal_context_mb=384.0,
            admission_noise_mb=819.2,
            foreign_floor_bucket=None,
            sample_observation_counts=(),
        )
        peaks = [
            pricing.model_sampling_peak(None, "Flux.1-Schnell fp8 (Compact)", "flux_1"),
            pricing.model_sampling_peak(None, "stable_diffusion", "stable_diffusion_1"),
        ]
        dominance = pricing.decide_card_dominance(inputs, foreign_floor_mb=0.0, peaks=peaks)

        assert dominance.verdicts["Flux.1-Schnell fp8 (Compact)"].rung is DominanceRung.CANNOT_CO_RESIDE
        assert dominance.verdicts["stable_diffusion"].rung is DominanceRung.NONE

    def test_a_model_outside_the_offer_is_judged_against_it(self, replay_store: LearnedFootprintStore) -> None:
        """A model the card does not offer gets the rung the offered set gives it, not a cached one."""
        dominance = _replay_dominance(replay_store, foreign_floor_mb=3300.0)
        verdict = pricing.model_dominance(dominance, replay_store, "Another Flux", "flux_1")

        assert verdict is not None
        assert verdict.rung is DominanceRung.CANNOT_CO_SAMPLE
        assert pricing.model_dominance(None, replay_store, "Another Flux", "flux_1") is None


_HYSTERESIS_TOTAL_MB = 24074.0
_HYSTERESIS_CHARGES_MB = 4852.0
_HYSTERESIS_BAND_MB = pricing.DOMINANCE_HYSTERESIS_MB


def _held_rung(
    peak_mb: float,
    partner_mb: float,
    previous: DominanceRung | None,
) -> DominanceRung:
    """Return the rung of ``peak_mb`` beside one partner, on a card with 24,074 MB and 4,852 MB of charges."""
    return pricing.dominance_rung(
        peak_mb,
        partner_peaks_mb=[partner_mb],
        total_vram_mb=_HYSTERESIS_TOTAL_MB,
        charges_mb=_HYSTERESIS_CHARGES_MB,
        previous=previous,
        band_mb=_HYSTERESIS_BAND_MB,
    )


class TestDominanceHysteresis:
    """A held rung moves only when its slack crosses zero by more than the floor bucket, on both edges."""

    def test_the_band_is_the_floor_bucket_the_recompute_key_uses(self) -> None:
        """One width governs both, so a figure moving inside one bucket never moves a rung."""
        from horde_worker_regen.process_management.scheduling.diagnostic_throttle import DIAGNOSTIC_MB_BUCKET

        assert pricing.DOMINANCE_HYSTERESIS_MB == DIAGNOSTIC_MB_BUCKET

    @pytest.mark.parametrize(
        ("slack_mb", "previous", "expected"),
        [
            (100.0, DominanceRung.CANNOT_CO_RESIDE, DominanceRung.CANNOT_CO_RESIDE),
            (300.0, DominanceRung.CANNOT_CO_RESIDE, DominanceRung.CANNOT_CO_SAMPLE),
            (-100.0, DominanceRung.CANNOT_CO_SAMPLE, DominanceRung.CANNOT_CO_SAMPLE),
            (-300.0, DominanceRung.CANNOT_CO_SAMPLE, DominanceRung.CANNOT_CO_RESIDE),
            (-100.0, None, DominanceRung.CANNOT_CO_RESIDE),
        ],
        ids=[
            "held_rung_two_inside_the_band_stays",
            "held_rung_two_past_the_band_leaves",
            "rung_one_inside_the_band_is_not_promoted",
            "rung_one_past_the_band_is_promoted",
            "a_first_verdict_reads_the_bare_sign",
        ],
    )
    def test_rung_two_edges(self, slack_mb: float, previous: DominanceRung | None, expected: DominanceRung) -> None:
        """Rung 2's slack beside one context must cross zero by more than the band to change a held rung."""
        peak_mb = _HYSTERESIS_TOTAL_MB - _HYSTERESIS_CHARGES_MB - slack_mb
        assert _held_rung(peak_mb, 3821.0, previous) is expected

    @pytest.mark.parametrize(
        ("room_mb", "previous", "expected"),
        [
            (-100.0, DominanceRung.NONE, DominanceRung.NONE),
            (-300.0, DominanceRung.NONE, DominanceRung.CANNOT_CO_SAMPLE),
            (100.0, DominanceRung.CANNOT_CO_SAMPLE, DominanceRung.CANNOT_CO_SAMPLE),
            (300.0, DominanceRung.CANNOT_CO_SAMPLE, DominanceRung.NONE),
            (-100.0, None, DominanceRung.CANNOT_CO_SAMPLE),
        ],
        ids=[
            "a_partner_short_inside_the_band_keeps_none",
            "a_partner_short_past_the_band_enters_rung_one",
            "held_rung_one_with_room_inside_the_band_stays",
            "held_rung_one_with_room_past_the_band_leaves",
            "a_first_verdict_reads_the_bare_sign",
        ],
    )
    def test_rung_one_edges(self, room_mb: float, previous: DominanceRung | None, expected: DominanceRung) -> None:
        """The best partner's room must cross zero by more than the band to change a held rung 1."""
        peak_mb = 10000.0
        partner_mb = _HYSTERESIS_TOTAL_MB - _HYSTERESIS_CHARGES_MB - peak_mb - room_mb
        assert _held_rung(peak_mb, partner_mb, previous) is expected

    def test_the_card_verdict_keeps_a_rung_through_drift_inside_the_band(self) -> None:
        """A recompute whose peak moved by less than the band past the threshold keeps the earlier rung."""
        inputs = pricing.DominanceInputs(
            offered_models=frozenset({"big"}),
            whole_card_pins=frozenset(),
            total_vram_mb=_HYSTERESIS_TOTAL_MB,
            first_context_mb=_HYSTERESIS_CHARGES_MB,
            marginal_context_mb=0.0,
            admission_noise_mb=0.0,
            foreign_floor_bucket=None,
            sample_observation_counts=(),
        )
        threshold_mb = _HYSTERESIS_TOTAL_MB - _HYSTERESIS_CHARGES_MB

        def peak(peak_mb: float) -> list[pricing.ModelSamplingPeak]:
            return [pricing.ModelSamplingPeak("big", peak_mb, ResolutionBucket.LE_1024, 10)]

        first = pricing.decide_card_dominance(inputs, foreign_floor_mb=0.0, peaks=peak(threshold_mb - 50.0))
        drifted = pricing.decide_card_dominance(
            inputs,
            foreign_floor_mb=0.0,
            peaks=peak(threshold_mb + 100.0),
            previous=first,
        )
        moved = pricing.decide_card_dominance(
            inputs, foreign_floor_mb=0.0, peaks=peak(threshold_mb + 300.0), previous=first
        )

        assert first.verdicts["big"].rung is DominanceRung.CANNOT_CO_SAMPLE
        assert drifted.verdicts["big"].rung is DominanceRung.CANNOT_CO_SAMPLE
        assert moved.verdicts["big"].rung is DominanceRung.CANNOT_CO_RESIDE


class TestDominancePairing:
    """The co-sampling lever: a dominant side pairs only where the pair leaves room on the card."""

    @pytest.fixture
    def replay_store(self, tmp_path: Path) -> LearnedFootprintStore:
        """Return the replay fixture's learned store from a 24 GB card, keyed for this platform."""
        return load_footprint_store(SESSION_FOOTPRINTS_PATH, scratch_directory=tmp_path)

    def test_nothing_samples_beside_a_rung_one_model(self, replay_store: LearnedFootprintStore) -> None:
        """Behind the 3,300 MB tenant no offered model, not even SD1.5 at 512, may sample beside Flux."""
        dominance = _replay_dominance(replay_store, foreign_floor_mb=3300.0)
        flux = ("Flux.1-Schnell fp8 (Compact)", "flux_1")
        for model, baseline in _REPLAY_CATALOGUE.items():
            assert pricing.pair_co_samples(dominance, replay_store, candidate=(model, baseline), running=flux) is False
            assert pricing.pair_co_samples(dominance, replay_store, candidate=flux, running=(model, baseline)) is False

    def test_a_pair_with_no_dominant_side_is_left_to_the_arbiter(self, replay_store: LearnedFootprintStore) -> None:
        """Two models with partners are not refused by the arithmetic, whatever room their pair leaves."""
        dominance = _replay_dominance(replay_store, foreign_floor_mb=3300.0)
        sdxl = ("AlbedoBase XL (SDXL)", "stable_diffusion_xl")
        assert pricing.pair_co_samples(dominance, replay_store, candidate=sdxl, running=sdxl) is True

    def test_without_the_tenant_flux_takes_a_partner(self, replay_store: LearnedFootprintStore) -> None:
        """At a zero floor Flux is on no rung, so an SD1.5 job may join it."""
        dominance = _replay_dominance(replay_store, foreign_floor_mb=0.0)
        assert (
            pricing.pair_co_samples(
                dominance,
                replay_store,
                candidate=("stable_diffusion", "stable_diffusion_1"),
                running=("Flux.1-Schnell fp8 (Compact)", "flux_1"),
            )
            is True
        )

    def test_an_unsized_card_admits(self) -> None:
        """With no verdict nothing is refused, and the arbiter's memory answer stands alone."""
        assert pricing.pair_co_samples(None, None, candidate=("a", None), running=("b", None)) is True


class TestDominanceCache:
    """The scheduler computes a card's verdict once per change of its inputs and the forecast reads it."""

    _FLUX = "flux_model"
    _SD15 = "sd15_model"

    def _scheduler(self, offered: list[str]) -> InferenceScheduler:
        reference = {
            self._FLUX: make_mock_model_reference_record(self._FLUX, baseline=KNOWN_IMAGE_GENERATION_BASELINE.flux_1),
            self._SD15: make_mock_model_reference_record(
                self._SD15,
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            ),
        }
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: _slot(0, model=None)}),  # type: ignore[arg-type]
            model_metadata=make_test_model_metadata(reference),
            device_free_mb=14000.0,
        )
        scheduler._runtime_config.bridge_data.image_models_to_load = offered  # type: ignore[attr-defined]
        # The first context the whole-card goldens' world charges, so the 16 GB card here is the goldens' card.
        scheduler._runtime_config.bridge_data.vram_per_process_overhead_mb = 1354.0  # type: ignore[attr-defined]
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16384.0)  # type: ignore[method-assign]
        return scheduler

    def test_an_unchanged_input_returns_the_cached_verdict(self) -> None:
        """Two reads with nothing moved return the same record, so nothing is recomputed per cycle."""
        scheduler = self._scheduler([self._FLUX, self._SD15])
        first = scheduler.card_dominance(None)

        assert first is not None
        assert scheduler.card_dominance(None) is first

    def test_the_verdict_follows_the_offered_set(self) -> None:
        """Taking a model out of the offer recomputes the verdict without it."""
        scheduler = self._scheduler([self._FLUX, self._SD15])
        first = scheduler.card_dominance(None)
        scheduler._runtime_config.bridge_data.image_models_to_load = [self._FLUX]  # type: ignore[attr-defined]
        second = scheduler.card_dominance(None)

        assert first is not None
        assert second is not None
        assert second is not first
        assert set(second.verdicts) == {self._FLUX}

    def test_the_verdict_follows_the_foreign_floor(self) -> None:
        """A floor that moves past its bucket recomputes the verdict with the new floor charged."""
        scheduler = self._scheduler([self._FLUX, self._SD15])
        floor_mb: list[float | None] = [None]
        scheduler.current_foreign_floor_mb = lambda _device_index: floor_mb[0]  # type: ignore[method-assign]
        first = scheduler.card_dominance(None)
        floor_mb[0] = 3300.0
        second = scheduler.card_dominance(None)

        assert first is not None
        assert second is not None
        assert second is not first
        assert first.foreign_floor_mb == 0.0
        assert second.foreign_floor_mb == 3300.0

    def test_the_verdict_follows_a_footprint_observation(self) -> None:
        """An observation on an offered baseline's pooled sampling key recomputes the verdict."""
        scheduler = self._scheduler([self._FLUX, self._SD15])
        store = LearnedFootprintStore()
        scheduler.set_footprint_store(store)
        first = scheduler.card_dominance(None)
        store.observe_peak(pricing.pooled_sample_key("flux_1", ResolutionBucket.LE_1024), 15000.0)

        assert scheduler.card_dominance(None) is not first

    async def test_the_snapshot_and_the_live_forecast_read_the_same_verdict(self) -> None:
        """Both forecast forms demand the residency for a rung-2 model, and the snapshot carries the cache."""
        scheduler = self._scheduler([self._FLUX, self._SD15])
        job = make_job_pop_response(model=self._FLUX, width=1024, height=1024)
        await track_popped_job_async(scheduler._job_tracker, job)  # type: ignore[attr-defined]
        snapshot = scheduler.snapshot()
        baseline = KNOWN_IMAGE_GENERATION_BASELINE.flux_1

        assert snapshot.card(None).dominance is scheduler.card_dominance(None)
        live = scheduler._forecast_streaming(job, baseline)  # type: ignore[attr-defined]
        assert pricing.forecast_streaming(snapshot, job, baseline) == live
        assert live.wants_whole_card is True
        assert live.whole_card_tier is True

    async def test_a_rung_one_model_is_not_demanded_a_residency(self) -> None:
        """Flux on a 24 GB card behind a 3,300 MB tenant samples beside a context, so no teardown is asked."""
        scheduler = self._scheduler([self._FLUX, self._SD15])
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=24074.0)  # type: ignore[method-assign]
        scheduler.current_foreign_floor_mb = lambda _device_index: 3300.0  # type: ignore[method-assign]
        dominance = scheduler.card_dominance(None)
        job = make_job_pop_response(model=self._FLUX, width=1024, height=1024)

        assert dominance is not None
        assert dominance.verdicts[self._FLUX].rung is not DominanceRung.CANNOT_CO_RESIDE
        forecast = scheduler._forecast_streaming(job, KNOWN_IMAGE_GENERATION_BASELINE.flux_1)  # type: ignore[attr-defined]
        assert forecast.wants_whole_card is False
        assert forecast.needs_exclusive_residency is False

    def test_a_rung_two_model_carries_its_partial_seat_and_the_ceiling(self) -> None:
        """The offer's seat is the one partial_seat_mb prices a plain job at. A model off rung 2 has none."""
        scheduler = self._scheduler([self._FLUX, self._SD15])
        seat = scheduler.whole_card_seat(self._FLUX, None)
        job = make_job_pop_response(model=self._FLUX, width=1024, height=1024)
        expected_seat_mb = pricing.partial_seat_mb(
            scheduler.snapshot(),
            job,
            "flux_1",
            device_index=None,
            weight_fraction=pricing.PARTIAL_SEAT_FALLBACK_WEIGHT_FRACTION,
        )

        assert seat is not None
        assert seat.seat_mb == expected_seat_mb
        assert seat.ceiling_mb == scheduler.achievable_ceiling_mb(None)
        assert scheduler.whole_card_seat(self._SD15, None) is None

    def test_a_pin_overrides_the_verdict(self) -> None:
        """A pinned model is demanded the residency on a card where it is on no rung."""
        scheduler = self._scheduler([self._FLUX, self._SD15])
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=24074.0)  # type: ignore[method-assign]
        scheduler._runtime_config.bridge_data.whole_card_models = [self._FLUX]  # type: ignore[attr-defined]
        job = make_job_pop_response(model=self._FLUX, width=1024, height=1024)

        forecast = scheduler._forecast_streaming(job, KNOWN_IMAGE_GENERATION_BASELINE.flux_1)  # type: ignore[attr-defined]
        assert forecast.wants_whole_card is True
        assert forecast.whole_card_pinned is True
