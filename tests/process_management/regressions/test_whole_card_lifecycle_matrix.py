"""Lifecycle-style whole-card residency regression matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.reclaim_ladder import VerifiedReclaimLadder
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.governance.whole_card import WholeCardResidencyMachine
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import make_job_pop_response, make_mock_process_info, track_popped_job_async
from tests.process_management.regressions.test_whole_card_deadlock_fixes import (
    _DEVICE_TOTAL_VRAM_MB,
    _FLUX_MODEL,
    _FLUX_WEIGHTS_MB,
    _MARGINAL_OVERHEAD_MB,
    _OTHER_SDXL,
    _PER_PROCESS_OVERHEAD_MB,
    _RESIDENT_SDXL,
    _deadlock_bridge_data,
    _make_real_plm,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


@dataclass(frozen=True)
class WholeCardLifecycleCase:
    """A deterministic whole-card queue lifecycle shape."""

    mode: Literal["initial", "prestaged"]
    target: int
    total_processes: int
    holder_state: HordeProcessState
    holder_load_state: ModelLoadState
    sibling_models: tuple[str | None, ...]
    queue_tail: tuple[str, ...]
    max_threads: int = 1
    safety_on_gpu: bool = False
    safety_pause_required: bool = False


@dataclass(frozen=True)
class LifecycleHarness:
    """Objects shared by a lifecycle matrix case."""

    scheduler: InferenceScheduler
    process_map: ProcessMap
    horde_model_map: HordeModelMap
    job_tracker: JobTracker
    flux_job: ImageGenerateJobPopResponse


@dataclass(frozen=True)
class ResidencyScenarioCase:
    """One invariant-matrix combination for a whole-card residency lifecycle."""

    profile: str
    total_vram_mb: float
    footprint_mb: float
    queue_shape: Literal["single", "burst", "alternating", "rotation"]
    safety_posture: Literal["off_gpu", "other_card", "residency_card"]
    residency_mode: Literal["initial", "prestaged"]


def _forecast_for_target(target: int) -> StreamForecast:
    """Return a whole-card forecast whose process target is deterministic on the 24 GB fixture card."""
    reserve_mb = 6500.0 if target == 1 else 5000.0
    forecast = StreamForecast(
        total_vram_mb=float(_DEVICE_TOTAL_VRAM_MB),
        free_now_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        free_if_alone_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        free_after_model_evict_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        weights_mb=_FLUX_WEIGHTS_MB,
        reserve_mb=reserve_mb,
        per_process_overhead_mb=float(_PER_PROCESS_OVERHEAD_MB),
        marginal_process_overhead_mb=_MARGINAL_OVERHEAD_MB,
        wants_whole_card=True,
    )
    assert forecast.max_resident_processes() == target
    return forecast


def _record_loaded_model(
    horde_model_map: HordeModelMap,
    *,
    model_name: str | None,
    load_state: ModelLoadState,
    process_id: int,
) -> None:
    if model_name is None:
        return
    horde_model_map.update_entry(horde_model_name=model_name, load_state=load_state, process_id=process_id)


def _make_flux_head_harness(case: WholeCardLifecycleCase) -> LifecycleHarness:
    """Build a queue-head Flux lifecycle with a real PLM and mocked process pipes."""
    processes = {
        1: make_mock_process_info(1, model_name=_FLUX_MODEL, state=case.holder_state),
    }
    for offset, model_name in enumerate(case.sibling_models, start=2):
        processes[offset] = make_mock_process_info(
            offset,
            model_name=model_name,
            state=HordeProcessState.WAITING_FOR_JOB,
        )

    for process in processes.values():
        process.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        process.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB

    process_map = ProcessMap(processes)
    horde_model_map = HordeModelMap(root={})
    _record_loaded_model(horde_model_map, model_name=_FLUX_MODEL, load_state=case.holder_load_state, process_id=1)
    for process_id, model_name in enumerate(case.sibling_models, start=2):
        _record_loaded_model(
            horde_model_map,
            model_name=model_name,
            load_state=ModelLoadState.LOADED_IN_VRAM,
            process_id=process_id,
        )

    job_tracker = JobTracker()
    bridge_data = _deadlock_bridge_data(
        max_threads=case.max_threads,
        safety_on_gpu=case.safety_on_gpu,
        whole_card_residency_safety_off_gpu=case.safety_pause_required,
    )
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        horde_model_map=horde_model_map,
        bridge_data=bridge_data,
        max_concurrent=case.max_threads,
        max_inference=case.total_processes,
    )
    scheduler._process_lifecycle = _make_real_plm(
        process_map=process_map,
        job_tracker=job_tracker,
        horde_model_map=horde_model_map,
        bridge_data=bridge_data,
        target_process_count=case.total_processes,
    )
    if case.safety_pause_required:
        scheduler._process_lifecycle._safety_gpu_paused = True

    flux_job = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216)
    return LifecycleHarness(
        scheduler=scheduler,
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        flux_job=flux_job,
    )


async def _queue_flux_head_case(harness: LifecycleHarness, case: WholeCardLifecycleCase) -> None:
    await track_popped_job_async(harness.job_tracker, harness.flux_job)
    for model in case.queue_tail:
        await track_popped_job_async(harness.job_tracker, make_job_pop_response(model))


def _begin_residency_for_case(harness: LifecycleHarness, case: WholeCardLifecycleCase) -> None:
    forecast = _forecast_for_target(case.target)
    if case.mode == "initial":
        harness.scheduler._establish_whole_card_residency(harness.flux_job, forecast, announce=False)
    else:
        harness.scheduler._begin_whole_card_residency(harness.flux_job, forecast, announce=False)


def _complete_child_preload_acks(harness: LifecycleHarness) -> None:
    """Model the child-side PRELOAD_MODEL acknowledgement between scheduler ticks."""
    for process in harness.process_map.values():
        if process.last_control_flag != HordeControlFlag.PRELOAD_MODEL:
            continue
        model_name = process.loaded_horde_model_name
        if model_name is None:
            continue
        process.last_process_state = HordeProcessState.PRELOADED_MODEL
        process.last_control_flag = None
        harness.horde_model_map.update_entry(
            horde_model_name=model_name,
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=process.process_id,
        )


async def _drive_scheduler_lifecycle(
    harness: LifecycleHarness,
    *,
    expected_job: ImageGenerateJobPopResponse,
    max_cycles: int = 6,
) -> ImageGenerateJobPopResponse | None:
    """Run deterministic scheduling ticks until inference dispatches or the lifecycle wedges."""
    for _ in range(max_cycles):
        harness.scheduler._converge_whole_card_residency()
        harness.scheduler.preload_models()
        _complete_child_preload_acks(harness)
        if await harness.scheduler.start_inference():
            assert len(harness.job_tracker.jobs_in_progress) == 1
            dispatched = harness.job_tracker.jobs_in_progress[0]
            assert dispatched is expected_job
            return dispatched
    return None


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            WholeCardLifecycleCase(
                mode="initial",
                target=1,
                total_processes=3,
                holder_state=HordeProcessState.WAITING_FOR_JOB,
                holder_load_state=ModelLoadState.LOADED_IN_VRAM,
                sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL),
                queue_tail=(_RESIDENT_SDXL, _OTHER_SDXL),
            ),
            id="initial-target1-queued-siblings",
        ),
        pytest.param(
            WholeCardLifecycleCase(
                mode="initial",
                target=2,
                total_processes=4,
                holder_state=HordeProcessState.WAITING_FOR_JOB,
                holder_load_state=ModelLoadState.LOADED_IN_VRAM,
                sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL, None),
                queue_tail=(_OTHER_SDXL, _RESIDENT_SDXL),
                max_threads=2,
            ),
            id="initial-target2-high-vram-mixed-siblings",
        ),
        pytest.param(
            WholeCardLifecycleCase(
                mode="prestaged",
                target=1,
                total_processes=3,
                holder_state=HordeProcessState.PRELOADED_MODEL,
                holder_load_state=ModelLoadState.LOADED_IN_RAM,
                sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL),
                queue_tail=(_RESIDENT_SDXL, _OTHER_SDXL),
            ),
            id="prestaged-ram-target1-post-drain",
        ),
        pytest.param(
            WholeCardLifecycleCase(
                mode="prestaged",
                target=2,
                total_processes=4,
                holder_state=HordeProcessState.PRELOADED_MODEL,
                holder_load_state=ModelLoadState.LOADED_IN_RAM,
                sibling_models=(_RESIDENT_SDXL, None, _OTHER_SDXL),
                queue_tail=(_OTHER_SDXL, _RESIDENT_SDXL),
                max_threads=2,
            ),
            id="prestaged-ram-target2-mixed-siblings",
        ),
        pytest.param(
            WholeCardLifecycleCase(
                mode="prestaged",
                target=1,
                total_processes=2,
                holder_state=HordeProcessState.PRELOADED_MODEL,
                holder_load_state=ModelLoadState.LOADED_IN_RAM,
                sibling_models=(_RESIDENT_SDXL,),
                queue_tail=(_RESIDENT_SDXL,),
                safety_on_gpu=True,
                safety_pause_required=True,
            ),
            id="prestaged-target1-safety-already-paused",
        ),
    ],
)
async def test_whole_card_head_lifecycle_matrix_converges(case: WholeCardLifecycleCase) -> None:
    """A whole-card head should converge and dispatch across representative worker lifecycle states."""
    harness = _make_flux_head_harness(case)
    await _queue_flux_head_case(harness, case)
    _begin_residency_for_case(harness, case)

    dispatched = await _drive_scheduler_lifecycle(harness, expected_job=harness.flux_job)

    assert dispatched is harness.flux_job
    assert harness.process_map.num_loaded_inference_processes() <= case.target
    holder = harness.process_map.get(1)
    assert holder is not None
    assert holder.last_control_flag == HordeControlFlag.START_INFERENCE
    assert holder.loaded_horde_model_name == _FLUX_MODEL


async def test_unsized_forecast_leaves_the_live_process_count_alone() -> None:
    """A residency whose forecast cannot size the card converges without cutting the pool.

    An unsized forecast names no depth: the card's total VRAM was never reported, so the number of contexts
    it holds is unknown. Collapsing to sole residency there would tear the pool down on the strength of a
    figure nobody measured, on exactly the hosts where the measurement is missing.
    """
    case = WholeCardLifecycleCase(
        mode="prestaged",
        target=1,
        total_processes=3,
        holder_state=HordeProcessState.PRELOADED_MODEL,
        holder_load_state=ModelLoadState.LOADED_IN_RAM,
        sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL),
        queue_tail=(_RESIDENT_SDXL, _OTHER_SDXL),
    )
    harness = _make_flux_head_harness(case)
    await _queue_flux_head_case(harness, case)

    unsized = StreamForecast(
        total_vram_mb=None,
        free_now_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        free_if_alone_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        free_after_model_evict_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        weights_mb=_FLUX_WEIGHTS_MB,
        reserve_mb=6500.0,
        per_process_overhead_mb=float(_PER_PROCESS_OVERHEAD_MB),
    )
    assert unsized.max_resident_processes() is None

    live_before = harness.process_map.num_loaded_inference_processes()
    harness.scheduler._begin_whole_card_residency(harness.flux_job, unsized, announce=False)
    harness.scheduler._converge_whole_card_residency()

    assert harness.process_map.num_loaded_inference_processes() == live_before
    assert live_before == case.total_processes


async def test_unsized_forecast_establish_leaves_the_live_process_count_alone() -> None:
    """A direct residency establishment with an unsized forecast likewise skips the scale-down.

    The establish path sizes its own target from the forecast rather than reading the ledger, so the
    absent-target contract must hold there independently of the converge path.
    """
    case = WholeCardLifecycleCase(
        mode="prestaged",
        target=1,
        total_processes=3,
        holder_state=HordeProcessState.PRELOADED_MODEL,
        holder_load_state=ModelLoadState.LOADED_IN_RAM,
        sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL),
        queue_tail=(_RESIDENT_SDXL, _OTHER_SDXL),
    )
    harness = _make_flux_head_harness(case)
    await _queue_flux_head_case(harness, case)

    unsized = StreamForecast(
        total_vram_mb=None,
        free_now_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        free_if_alone_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        free_after_model_evict_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        weights_mb=_FLUX_WEIGHTS_MB,
        reserve_mb=6500.0,
        per_process_overhead_mb=float(_PER_PROCESS_OVERHEAD_MB),
    )
    assert unsized.max_resident_processes() is None

    live_before = harness.process_map.num_loaded_inference_processes()
    harness.scheduler._establish_whole_card_residency(harness.flux_job, unsized, announce=False)

    assert harness.process_map.num_loaded_inference_processes() == live_before
    assert live_before == case.total_processes


async def test_governance_reprice_tightens_a_held_residency_without_rewriting_its_grant() -> None:
    """New pricing lowers the live target, books one ladder debt, and preserves grant evidence."""
    case = WholeCardLifecycleCase(
        mode="initial",
        target=2,
        total_processes=4,
        holder_state=HordeProcessState.WAITING_FOR_JOB,
        holder_load_state=ModelLoadState.LOADED_IN_VRAM,
        sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL, None),
        queue_tail=(),
        max_threads=2,
    )
    harness = _make_flux_head_harness(case)
    await _queue_flux_head_case(harness, case)
    grant_forecast = _forecast_for_target(2)
    harness.scheduler._establish_whole_card_residency(harness.flux_job, grant_forecast, announce=True)
    reclaim_ladder = VerifiedReclaimLadder()
    harness.scheduler.set_reclaim_ladder(reclaim_ladder)
    harness.scheduler._forecast_streaming = Mock(return_value=_forecast_for_target(1))

    harness.scheduler._reprice_held_whole_card_residencies()

    state = harness.scheduler._residency_state(None)
    assert state.forecast is grant_forecast
    assert harness.scheduler._whole_card_ledger.effective_target(state) == 1
    assert harness.process_map.num_loaded_inference_processes() == 1
    assert reclaim_ladder.has_context_reduction(None) is True


async def test_governance_reprice_never_grows_a_held_residency() -> None:
    """A later roomier forecast cannot add contexts underneath a model that still owns the card."""
    case = WholeCardLifecycleCase(
        mode="initial",
        target=1,
        total_processes=3,
        holder_state=HordeProcessState.WAITING_FOR_JOB,
        holder_load_state=ModelLoadState.LOADED_IN_VRAM,
        sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL),
        queue_tail=(),
    )
    harness = _make_flux_head_harness(case)
    await _queue_flux_head_case(harness, case)
    grant_forecast = _forecast_for_target(1)
    harness.scheduler._establish_whole_card_residency(harness.flux_job, grant_forecast, announce=True)
    harness.scheduler._forecast_streaming = Mock(return_value=_forecast_for_target(2))

    harness.scheduler._reprice_held_whole_card_residencies()

    state = harness.scheduler._residency_state(None)
    assert state.forecast is grant_forecast
    assert harness.scheduler._whole_card_ledger.effective_target(state) == 1
    assert harness.process_map.num_loaded_inference_processes() == 1


_SCENARIO_PROFILES = (
    ("flux-knife-edge", 16_400.0, 12_600.0),
    ("qwen-knife-edge", 27_700.0, 23_000.0),
    ("z-image-knife-edge", 19_700.0, 15_500.0),
    ("flux-roomy", 24_576.0, 12_600.0),
)
_QUEUE_SHAPES = ("single", "burst", "alternating", "rotation")
_SAFETY_POSTURES = ("off_gpu", "other_card", "residency_card")
_RESIDENCY_MODES = ("initial", "prestaged")


def _residency_scenario_cases() -> list[pytest.ParameterSet]:
    """Build the 4 x 4 x 3 x 2 invariant matrix (96 cases)."""
    return [
        pytest.param(
            ResidencyScenarioCase(
                profile=profile,
                total_vram_mb=total_vram_mb,
                footprint_mb=footprint_mb,
                queue_shape=queue_shape,
                safety_posture=safety_posture,
                residency_mode=residency_mode,
            ),
            id=f"{profile}-{queue_shape}-{safety_posture}-{residency_mode}",
        )
        for profile, total_vram_mb, footprint_mb in _SCENARIO_PROFILES
        for queue_shape in _QUEUE_SHAPES
        for safety_posture in _SAFETY_POSTURES
        for residency_mode in _RESIDENCY_MODES
    ]


def _scenario_forecast(case: ResidencyScenarioCase) -> StreamForecast:
    """Build a sized forecast from a profile without pinning a golden target."""
    overhead_mb = 1_200.0
    marginal_mb = 700.0
    reserve_mb = 2_200.0
    return StreamForecast(
        weights_mb=case.footprint_mb * 0.75,
        footprint_mb=case.footprint_mb,
        reserve_mb=reserve_mb,
        base_reserve_mb=1_000.0,
        free_now_mb=500.0,
        free_if_alone_mb=case.total_vram_mb - overhead_mb,
        free_after_model_evict_mb=case.total_vram_mb - overhead_mb - 3 * marginal_mb,
        total_vram_mb=case.total_vram_mb,
        per_process_overhead_mb=overhead_mb,
        marginal_process_overhead_mb=marginal_mb,
        wants_whole_card=True,
    )


def _queue_for_shape(shape: str) -> tuple[str, ...]:
    """Return model classes for one queue shape; ``light`` needs no residency."""
    return {
        "single": ("heavy-a",),
        "burst": ("heavy-a", "heavy-a", "heavy-a"),
        "alternating": ("heavy-a", "light", "heavy-a", "light"),
        "rotation": ("heavy-a", "heavy-b", "heavy-c", "heavy-a"),
    }[shape]


def _assert_establishment_rate(machine: WholeCardResidencyMachine) -> None:
    """No rolling establishment window contains more grants than policy allows."""
    stamps = tuple(machine.state_for(0).establishments)
    for stamp in stamps:
        in_window = [candidate for candidate in stamps if stamp <= candidate <= stamp + 240.0]
        assert len(in_window) <= 2


@pytest.mark.parametrize("case", _residency_scenario_cases())
def test_residency_scenario_matrix_preserves_lifecycle_invariants(case: ResidencyScenarioCase) -> None:
    """Every card/queue/safety/residency combination preserves the policy invariants."""
    machine = WholeCardResidencyMachine()
    forecast = _scenario_forecast(case)
    target = machine.target_process_count(forecast)
    assert target is not None
    total_processes = 4
    now = 10_000.0
    dispatched = 0

    for model in _queue_for_shape(case.queue_shape):
        if model == "light":
            dispatched += 1
            now += 15.0
            continue

        held = machine.get(0)
        if held is None or held.model != model:
            if held is not None and held.model is not None:
                machine.record_restore(0, now=now, restore_grace_seconds=60.0)
            while machine.establish_rate_exceeded(0, now=now):
                now += 241.0
            machine.record_grant(
                0,
                model=model,
                forecast=forecast,
                cooldown_until=now + 90.0,
                now=now,
                refresh_established=True,
                establish_grace_seconds=120.0,
            )

        effective_target = machine.effective_target(machine.state_for(0))
        assert effective_target is not None
        safety_initially_clear = case.safety_posture != "residency_card"
        initial_count = (
            total_processes if case.residency_mode == "prestaged" else min(total_processes, effective_target)
        )

        assert not machine.teardown_complete(
            forecast,
            loaded_process_count=effective_target + 1,
            process_target=effective_target,
            safety_clear_of_card=True,
            weights_fit_live=True,
            drain_backstop_elapsed=False,
        )
        if not safety_initially_clear:
            assert not machine.teardown_complete(
                forecast,
                loaded_process_count=min(initial_count, effective_target),
                process_target=effective_target,
                safety_clear_of_card=False,
                weights_fit_live=True,
                drain_backstop_elapsed=False,
            )
        assert machine.teardown_complete(
            forecast,
            loaded_process_count=min(initial_count, effective_target),
            process_target=effective_target,
            safety_clear_of_card=True,
            weights_fit_live=True,
            drain_backstop_elapsed=False,
        )
        dispatched += 1
        state = machine.state_for(0)
        window_open = (state.model is not None and (now - state.established_at) < 120.0) or (
            state.restore_at != 0.0 and (now - state.restore_at) < 60.0
        )
        grace_active = machine.grace_active(
            now=now,
            establish_grace_seconds=120.0,
            restore_grace_seconds=60.0,
        )
        assert grace_active is window_open, (
            "the excuse tracks the granted window alone: the rolling grace budget gates a new establishment "
            "at admission and never withdraws the window of one already under way"
        )
        now += 15.0

    if machine.state_for(0).model is not None:
        machine.record_restore(0, now=now, restore_grace_seconds=60.0)
    _assert_establishment_rate(machine)
    assert machine.state_for(0).model is None
    assert dispatched == len(_queue_for_shape(case.queue_shape))


@pytest.mark.slow
def test_db0_twenty_minute_rotation_sim_drains_every_admitted_head() -> None:
    """A virtual 20-minute card-0 rotation remains rate-bounded and drains after each handoff."""
    case = ResidencyScenarioCase(
        profile="flux-knife-edge",
        total_vram_mb=16_400.0,
        footprint_mb=12_600.0,
        queue_shape="rotation",
        safety_posture="residency_card",
        residency_mode="prestaged",
    )
    machine = WholeCardResidencyMachine()
    forecast = _scenario_forecast(case)
    target = machine.target_process_count(forecast)
    assert target is not None
    dispatched = 0

    for now, model in zip(range(20_000, 21_200, 300), ("heavy-a", "heavy-b", "heavy-c", "heavy-a"), strict=True):
        assert not machine.establish_rate_exceeded(0, now=float(now))
        machine.record_grant(
            0,
            model=model,
            forecast=forecast,
            cooldown_until=float(now + 90),
            now=float(now),
            refresh_established=True,
            establish_grace_seconds=120.0,
        )
        assert machine.teardown_complete(
            forecast,
            loaded_process_count=target,
            process_target=target,
            safety_clear_of_card=True,
            weights_fit_live=True,
            drain_backstop_elapsed=False,
        )
        dispatched += 1
        machine.record_restore(0, now=float(now + 60), restore_grace_seconds=60.0)

    _assert_establishment_rate(machine)
    assert dispatched == 4
    assert machine.state_for(0).model is None


@pytest.mark.parametrize(
    "queue_models",
    [
        pytest.param((_RESIDENT_SDXL, _FLUX_MODEL), id="resident-head-then-flux"),
        pytest.param((_RESIDENT_SDXL, _OTHER_SDXL, _FLUX_MODEL), id="resident-head-sdxl-then-flux"),
    ],
)
async def test_whole_card_residency_waits_until_flux_reaches_queue_head(
    queue_models: tuple[str, ...],
) -> None:
    """Flux behind a ready resident head must not reserve the card before its queue turn."""
    head_process = make_mock_process_info(
        1,
        model_name=_RESIDENT_SDXL,
        state=HordeProcessState.PRELOADED_MODEL,
    )
    flux_process = make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    for process in (head_process, flux_process):
        process.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        process.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB

    process_map = ProcessMap({1: head_process, 2: flux_process})
    horde_model_map = HordeModelMap(root={})
    horde_model_map.update_entry(
        horde_model_name=_RESIDENT_SDXL,
        load_state=ModelLoadState.LOADED_IN_RAM,
        process_id=1,
    )
    job_tracker = JobTracker()
    jobs = [make_job_pop_response(model) for model in queue_models]
    for job in jobs:
        await track_popped_job_async(job_tracker, job)

    bridge_data = _deadlock_bridge_data(image_models_to_load=[_RESIDENT_SDXL, _OTHER_SDXL, _FLUX_MODEL])
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        horde_model_map=horde_model_map,
        bridge_data=bridge_data,
        max_concurrent=1,
        max_inference=2,
    )
    scheduler._process_lifecycle = _make_real_plm(
        process_map=process_map,
        job_tracker=job_tracker,
        horde_model_map=horde_model_map,
        bridge_data=bridge_data,
        target_process_count=2,
    )

    scheduler.preload_models()
    assert scheduler.whole_card_residency_state().active is False

    assert await scheduler.start_inference() is True
    assert jobs[0] in job_tracker.jobs_in_progress
    assert jobs[0].model == _RESIDENT_SDXL
    assert head_process.last_control_flag == HordeControlFlag.START_INFERENCE
    assert scheduler.whole_card_residency_state().active is False
    assert flux_process.last_control_flag != HordeControlFlag.START_INFERENCE
    assert all(process.process_type is HordeProcessType.INFERENCE for process in process_map.values())
