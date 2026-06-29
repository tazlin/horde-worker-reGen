"""Regression guards for the whole-card residency grant decision and the dispatch-gate forward-progress invariant.

Whole-card *exclusive* residency does two very different things: it evicts other processes' resident models
(cheap, reversible) and it tears down sibling processes/contexts, moves safety off-GPU, and holds a cooldown
(disruptive, churn-prone). The second is warranted only for a model that genuinely needs the card to itself;
granting it to a model the budget shows can co-reside churns the device and can wedge the queue. These tests pin
the grant decision and the gate that releases a reserved head:

- the grant is decided on the *persistent* weight footprint, never a transient activation spike
  (:class:`TestActivationReserveDoesNotForceResidency`);
- a whole-card-intent (EXTRA_LARGE) model yields to the budget on a card roomy enough for another full model,
  its "never co-sample" contract upheld by the overlap gate instead (:class:`TestIntentYieldsToRoomyBudget`);
- a legitimately reserved head, once the device has drained, always reaches dispatch within a bounded window
  rather than parking forever (:class:`TestReservedHeadNeverWaitsIndefinitely`);
- and making room for a head never evicts an in-flight job's model (:class:`TestMakingRoomNeverStrandsInflight`).

:class:`TestForecastVerdicts` pins the representative forecast verdicts the grant tests reason about, so a change
to the budget arithmetic surfaces here first.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.resource_budget import (
    _CORESIDENT_SIBLING_MODEL_FLOOR_MB,
    StreamForecast,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import _WHOLE_CARD_ESTABLISH_GRACE_SECONDS
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_runtime_config,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# Representative hardware. The 16 GB card is the tight regime where a heavy checkpoint genuinely fills the device;
# the 24 GB card is the roomy regime where the same checkpoint co-resides with a sibling model. The free figures
# are carried on the forecast directly so the verdicts are reproducible.
_CARD16_TOTAL_MB = 16375  # free_if_alone (15021) + per_process_overhead (1354)
_CARD16_PPO_MB = 1354
_CARD16_FREE_IF_ALONE_MB = 15021.0
_CARD16_FREE_AFTER_EVICT_MB = 9605.0  # device-wide free with all four contexts alive but model-free
_BASE_RESERVE_MB = 3100.0  # the bounded inference-reserve floor

_CARD24_TOTAL_MB = 24074
_CARD24_PPO_MB = 3125
_CARD24_MARGINAL_MB = 649.0
_CARD24_FREE_IF_ALONE_MB = float(_CARD24_TOTAL_MB - _CARD24_PPO_MB)
_CARD24_FREE_AFTER_EVICT_MB = _CARD24_FREE_IF_ALONE_MB - 3 * _CARD24_MARGINAL_MB
_CARD24_BASE_RESERVE_MB = 3096.0

_VRAM_RESERVE_MB = 3096.0
_RAM_RESERVE_MB = 8192.0

# A card-filling EXTRA_LARGE checkpoint (whole-card by name) and a moderate-weight SDXL whose activation reserve
# can balloon at a large batch/resolution: the two model classes the grant decision must tell apart.
_FLUX_MODEL = "Flux.1-Schnell fp8 (Compact)"  # EXTRA_LARGE by name (consts.VRAM_HEAVY_MODELS)
_FLUX_WEIGHTS_MB = 11500.0
_FLUX_ESTABLISH_RESERVE_MB = 4646.0
_FLUX16_ESTABLISH_FREE_NOW_MB = 9129.0  # 16 GB device busy: four contexts + resident SDXL models
_FLUX24_ESTABLISH_FREE_NOW_MB = 9677.0  # 24 GB device busy

_MODERATE_SDXL_MODEL = "WAI-NSFW-illustrious-SDXL"
_MODERATE_SDXL_WEIGHTS_MB = 4900.0
_MODERATE_SDXL_BALLOONED_RESERVE_MB = 13785.0  # activation reserve at a large batch/resolution
_MODERATE_SDXL_NORMAL_RESERVE_MB = 3100.0
_MODERATE_SDXL_FREE_NOW_MB = 15005.0

_SDXL_A = "AlbedoBase XL 3.1"
_SDXL_C = "CyberRealistic Pony"
_SDXL_D = "ICBINP XL"
_SDXL_E = "Edge Of Realism"

# The idle-sibling pool. The head model is excluded per-topology so a sibling never duplicates the head (which
# would leave a second holder the convergence cannot collapse).
_SIBLING_MODEL_POOL: list[str] = [_SDXL_A, _SDXL_C, _MODERATE_SDXL_MODEL, _SDXL_D, _SDXL_E]

_CONFIGURED_MODELS: list[str] = [_FLUX_MODEL, _SDXL_A, _SDXL_C, _MODERATE_SDXL_MODEL]


def _bridge_data(**overrides: object) -> Mock:
    """Create a whole-card-on, budget-on bridge-data mock for the scheduler under test."""
    base: dict[str, object] = {
        "enable_vram_budget": True,
        "whole_card_exclusive_residency": True,
        "whole_card_residency_safety_off_gpu": False,
        "safety_on_gpu": True,
        "vram_reserve_mb": _VRAM_RESERVE_MB,
        "ram_reserve_mb": _RAM_RESERVE_MB,
        "vram_per_process_overhead_mb": _CARD16_PPO_MB,
        "overbudget_exclusive_mode": True,
        "whole_card_residency_cooldown_seconds": 45,
        "image_models_to_load": _CONFIGURED_MODELS,
        "max_threads": 1,
    }
    base.update(overrides)
    return make_mock_bridge_data(**base)


def _forecast_16gb(
    *, weights_mb: float, reserve_mb: float, free_now_mb: float, wants_whole_card: bool = False
) -> StreamForecast:
    """Create an establishment forecast on the tight (16 GB) card.

    ``wants_whole_card`` mirrors the scheduler's per-tier flag: an EXTRA_LARGE head carries it, a moderate SDXL
    head does not.
    """
    return StreamForecast(
        weights_mb=weights_mb,
        reserve_mb=reserve_mb,
        base_reserve_mb=_BASE_RESERVE_MB,
        free_now_mb=free_now_mb,
        free_if_alone_mb=_CARD16_FREE_IF_ALONE_MB,
        free_after_model_evict_mb=_CARD16_FREE_AFTER_EVICT_MB,
        total_vram_mb=float(_CARD16_TOTAL_MB),
        per_process_overhead_mb=float(_CARD16_PPO_MB),
        wants_whole_card=wants_whole_card,
    )


def _forecast_24gb(
    *, weights_mb: float, reserve_mb: float, free_now_mb: float, wants_whole_card: bool
) -> StreamForecast:
    """Create an establishment forecast on the roomy (24 GB) card."""
    return StreamForecast(
        weights_mb=weights_mb,
        reserve_mb=reserve_mb,
        base_reserve_mb=_CARD24_BASE_RESERVE_MB,
        free_now_mb=free_now_mb,
        free_if_alone_mb=_CARD24_FREE_IF_ALONE_MB,
        free_after_model_evict_mb=_CARD24_FREE_AFTER_EVICT_MB,
        total_vram_mb=float(_CARD24_TOTAL_MB),
        per_process_overhead_mb=float(_CARD24_PPO_MB),
        marginal_process_overhead_mb=_CARD24_MARGINAL_MB,
        wants_whole_card=wants_whole_card,
    )


class TestForecastVerdicts:
    """Pin the representative forecast verdicts the grant tests reason about.

    These are the fix-independent quantities (servability alone, the budgeted co-resident count). A change to the
    budget arithmetic that altered the grant decision for this hardware class fails here first.
    """

    def test_card_filling_model_needs_sole_residency(self) -> None:
        """A heavy checkpoint that fills a tight card needs sole residency (target one)."""
        forecast = _forecast_16gb(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=_FLUX_ESTABLISH_RESERVE_MB,
            free_now_mb=_FLUX16_ESTABLISH_FREE_NOW_MB,
            wants_whole_card=True,
        )
        assert forecast.needs_exclusive_residency is True
        assert forecast.streams_unavoidably is False
        assert forecast.max_resident_processes() == 1

    def test_moderate_model_is_servable_alone_with_a_ballooned_reserve(self) -> None:
        """A moderate model with a ballooned activation reserve is still servable alone; its peak budgets no context.

        The grant decision for this case is exercised in :class:`TestActivationReserveDoesNotForceResidency`.
        """
        forecast = _forecast_16gb(
            weights_mb=_MODERATE_SDXL_WEIGHTS_MB,
            reserve_mb=_MODERATE_SDXL_BALLOONED_RESERVE_MB,
            free_now_mb=_MODERATE_SDXL_FREE_NOW_MB,
        )
        assert forecast.fits_alone is True
        assert forecast.max_resident_processes() == 1

    def test_card_filling_model_is_budgeted_co_resident_on_a_roomy_card(self) -> None:
        """On a roomy card the budget shows the heavy checkpoint co-resides, so the grant follows the budget."""
        forecast = _forecast_24gb(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=_FLUX_ESTABLISH_RESERVE_MB,
            free_now_mb=_FLUX24_ESTABLISH_FREE_NOW_MB,
            wants_whole_card=True,
        )
        assert forecast.fits_alone is True
        assert forecast.max_resident_processes() == 8
        assert forecast.needs_exclusive_residency is False
        no_intent = dataclasses.replace(forecast, wants_whole_card=False)
        assert no_intent.needs_exclusive_residency is False


class TestActivationReserveDoesNotForceResidency:
    """The grant rests on the persistent weight footprint, never the transient activation reserve."""

    def test_ballooned_reserve_co_resides_rather_than_reserves(self) -> None:
        """A moderate model whose activation reserve has ballooned must co-reside, not claim the whole card."""
        forecast = _forecast_16gb(
            weights_mb=_MODERATE_SDXL_WEIGHTS_MB,
            reserve_mb=_MODERATE_SDXL_BALLOONED_RESERVE_MB,
            free_now_mb=_MODERATE_SDXL_FREE_NOW_MB,
        )
        assert forecast.needs_exclusive_residency is False, (
            "a moderate-weight model on a 16 GB card must not tear down sibling processes for the whole card: its "
            "persistent weights leave ample room, so the large reserve is a transient activation spike to absorb "
            "by evicting a sibling model and sampling under the over-budget step grace"
        )

    @pytest.mark.parametrize("reserve_mb", [3100.0, 6406.0, 9000.0, 11835.0, 13785.0, 15000.0])
    def test_grant_is_invariant_to_the_activation_reserve(self, reserve_mb: float) -> None:
        """Across the range of activation reserves a batch/resolution can drive, a moderate model never reserves.

        A residency that toggles with the transient activation estimate is the over-grant that churns the card.
        """
        forecast = _forecast_16gb(
            weights_mb=_MODERATE_SDXL_WEIGHTS_MB, reserve_mb=reserve_mb, free_now_mb=_MODERATE_SDXL_FREE_NOW_MB
        )
        assert forecast.needs_exclusive_residency is False, (
            f"with activation reserve {reserve_mb:.0f} MB the moderate model must still co-reside"
        )

    def test_card_filling_model_still_reserves(self) -> None:
        """The persistent-weights rule must not over-correct: a model whose weights fill the card still reserves."""
        forecast = _forecast_16gb(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=_FLUX_ESTABLISH_RESERVE_MB,
            free_now_mb=_FLUX16_ESTABLISH_FREE_NOW_MB,
            wants_whole_card=True,
        )
        assert forecast.needs_exclusive_residency is True

    def test_moderate_model_at_a_normal_reserve_co_resides(self) -> None:
        """At a normal activation reserve the moderate model co-resides (the baseline both rules agree on)."""
        forecast = _forecast_16gb(
            weights_mb=_MODERATE_SDXL_WEIGHTS_MB,
            reserve_mb=_MODERATE_SDXL_NORMAL_RESERVE_MB,
            free_now_mb=_MODERATE_SDXL_FREE_NOW_MB,
        )
        assert forecast.needs_exclusive_residency is False


class TestIntentYieldsToRoomyBudget:
    """A whole-card-intent model co-resides where another model fits; the overlap gate stops co-sampling."""

    def test_intent_model_co_resides_when_a_sibling_model_fits(self) -> None:
        """On a card with room for another full model an intent model co-resides rather than reserving the device.

        The criterion is room for a real sibling *model*, not the empty-context count: the budgeted context count
        reads high here, and reads high too for a heavy model on a tight card that cannot host a second model (see
        :meth:`test_intent_holds_when_no_sibling_model_fits`).
        """
        forecast = _forecast_24gb(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=_FLUX_ESTABLISH_RESERVE_MB,
            free_now_mb=_FLUX24_ESTABLISH_FREE_NOW_MB,
            wants_whole_card=True,
        )
        room_for_sibling = _CARD24_FREE_IF_ALONE_MB - _FLUX_WEIGHTS_MB - _CARD24_BASE_RESERVE_MB
        assert room_for_sibling >= _CORESIDENT_SIBLING_MODEL_FLOOR_MB, (
            f"precondition: the card holds a sibling model ({room_for_sibling:.0f} MB free after the weights, vs "
            f"the {_CORESIDENT_SIBLING_MODEL_FLOOR_MB:.0f} MB floor)"
        )
        assert forecast.needs_exclusive_residency is False, (
            "where the budget shows room for another full model, intent must not force a process teardown; "
            "the 'never co-sample' contract is enforced by the overlap gate, not by reserving the device"
        )

    def test_intent_holds_when_no_sibling_model_fits(self) -> None:
        """A heavy intent model on a card too tight for a second model still reserves it.

        The budgeted context count reads high even though no real sibling model fits, which is why the criterion
        must be model room rather than context count; co-residing a second model here would thrash.
        """
        forecast = StreamForecast(
            weights_mb=10000.0,
            reserve_mb=2048.0,
            free_now_mb=13000.0,
            free_if_alone_mb=15000.0,
            free_after_model_evict_mb=14000.0,
            total_vram_mb=16384.0,
            per_process_overhead_mb=1354.0,
            marginal_process_overhead_mb=300.0,
            wants_whole_card=True,
        )
        room_for_sibling = 15000.0 - 10000.0 - 2048.0  # 2952 MB: no SDXL-class sibling fits
        assert room_for_sibling < _CORESIDENT_SIBLING_MODEL_FLOOR_MB
        max_resident = forecast.max_resident_processes()
        assert max_resident is not None and max_resident > 4, (
            "the budgeted context count reads high here even though no real sibling model fits"
        )
        assert forecast.needs_exclusive_residency is True

    def test_card_filling_model_on_a_tight_card_reserves(self) -> None:
        """On a tight card the heavy checkpoint has no room to co-reside, so it reserves."""
        forecast = _forecast_16gb(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=_FLUX_ESTABLISH_RESERVE_MB,
            free_now_mb=_FLUX16_ESTABLISH_FREE_NOW_MB,
            wants_whole_card=True,
        )
        assert forecast.max_resident_processes() == 1
        assert forecast.needs_exclusive_residency is True

    async def test_co_sampling_is_blocked_by_the_overlap_gate(self) -> None:
        """An EXTRA_LARGE candidate never joins a busy card, so co-residing it is safe.

        This is what makes yielding to the budget harmonious: even when the heavy model co-resides, the overlap
        gate still refuses to let it co-sample with another in-flight job, providing the protection the teardown
        otherwise would, without the churn.
        """
        process_map = ProcessMap({
            1: make_mock_process_info(1, model_name=_SDXL_A, state=HordeProcessState.INFERENCE_STARTING),
            2: make_mock_process_info(2, model_name=_FLUX_MODEL, state=HordeProcessState.WAITING_FOR_JOB),
        })
        job_tracker = JobTracker()
        scheduler = _make_inference_scheduler(
            process_map=process_map, job_tracker=job_tracker, bridge_data=_bridge_data(), max_concurrent=2
        )

        running_sdxl = make_job_pop_response(_SDXL_A)
        await mark_job_in_progress_async(job_tracker, running_sdxl)
        flux_candidate = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216)

        assert scheduler._concurrent_overlap_allowed(flux_candidate) is False, (
            "an EXTRA_LARGE candidate must never join a busy card; the overlap gate, not a process teardown, is "
            "what enforces that it never shares a card with a concurrent job"
        )


def _record_loaded_model(
    horde_model_map: HordeModelMap, *, model_name: str, load_state: ModelLoadState, process_id: int
) -> None:
    horde_model_map.update_entry(horde_model_name=model_name, load_state=load_state, process_id=process_id)


def _prestaged_head_topology(*, head_model: str, n_idle_siblings: int = 3) -> tuple[ProcessMap, HordeModelMap]:
    """Build a topology where process one holds the pre-staged head and the rest are idle siblings.

    The siblings hold models queued behind the head; the head model is excluded from the sibling pool so no
    sibling duplicates it (which would leave a second holder the convergence cannot collapse).
    """
    sibling_models = [model for model in _SIBLING_MODEL_POOL if model != head_model][:n_idle_siblings]
    head = make_mock_process_info(1, model_name=head_model, state=HordeProcessState.PRELOADED_MODEL)
    procs = {1: head}
    for offset, sibling_model in enumerate(sibling_models, start=2):
        procs[offset] = make_mock_process_info(
            offset, model_name=sibling_model, state=HordeProcessState.WAITING_FOR_JOB
        )

    busy_used_mb = _CARD16_TOTAL_MB - _FLUX16_ESTABLISH_FREE_NOW_MB  # device-wide used at establishment
    for proc in procs.values():
        proc.total_vram_mb = _CARD16_TOTAL_MB
        proc.vram_usage_mb = busy_used_mb

    process_map = ProcessMap(procs)
    horde_model_map = HordeModelMap(root={})
    _record_loaded_model(horde_model_map, model_name=head_model, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
    for offset, sibling_model in enumerate(sibling_models, start=2):
        _record_loaded_model(
            horde_model_map, model_name=sibling_model, load_state=ModelLoadState.LOADED_IN_VRAM, process_id=offset
        )
    return process_map, horde_model_map


def _drain_device(process_map: ProcessMap, *, free_mb: float) -> None:
    """Mutate the process map so the device-wide free reading reflects the freed VRAM after a teardown."""
    used_mb = max(0, int(_CARD16_TOTAL_MB - free_mb))
    for proc in process_map.values():
        proc.vram_usage_mb = used_mb


def _make_real_plm(
    *, process_map: ProcessMap, job_tracker: JobTracker, horde_model_map: HordeModelMap, bridge_data: Mock
) -> ProcessLifecycleManager:
    """Create a real ProcessLifecycleManager sharing the given map and tracker, with mocked mp pipes."""
    return ProcessLifecycleManager(
        ctx=multiprocessing.get_context("spawn"),  # type: ignore[arg-type]
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        process_message_queue=Mock(),
        card_runtimes=make_test_card_runtimes(target_process_count=4, config=bridge_data),
        disk_lock=Mock(),
        aux_model_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        max_safety_processes=1,
        amd_gpu=False,
        directml=None,
        abort_callback=Mock(),
        state=WorkerState(),
    )


class TestReservedHeadNeverWaitsIndefinitely:
    """Once the device has drained, a legitimately reserved head reaches dispatch within a bounded window.

    This is agnostic about the mechanism (a live free-VRAM reading, a bounded fallback at the dispatch gate, or
    not over-granting); it pins only that a reserved head is never parked forever on residency dynamics. It uses a
    card-filling head on the tight card, where the residency is legitimate, so it isolates the dispatch gate
    rather than the grant decision.
    """

    async def test_reserved_head_dispatches_after_the_device_drains(self) -> None:
        """The pre-staged head dispatches once the device has drained enough to load its weights."""
        process_map, horde_model_map = _prestaged_head_topology(head_model=_FLUX_MODEL)
        job_tracker = JobTracker()
        bridge_data = _bridge_data()
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=bridge_data,
            max_concurrent=1,
            max_inference=4,
        )
        scheduler._process_lifecycle = _make_real_plm(
            process_map=process_map, job_tracker=job_tracker, horde_model_map=horde_model_map, bridge_data=bridge_data
        )

        flux_job = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216, ddim_steps=4)
        await track_popped_job_async(job_tracker, flux_job)
        forecast = _forecast_16gb(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=_FLUX_ESTABLISH_RESERVE_MB,
            free_now_mb=_FLUX16_ESTABLISH_FREE_NOW_MB,
            wants_whole_card=True,
        )
        scheduler._begin_whole_card_residency(flux_job, forecast, announce=True)

        # Back-date establishment past the protective grace: any legitimate setup is long over, so a head still
        # parked from here on is the indefinite wait the invariant forbids.
        scheduler._residency_state(None).established_at = time.time() - (_WHOLE_CARD_ESTABLISH_GRACE_SECONDS + 5.0)

        dispatched = False
        for _ in range(10):
            scheduler._converge_whole_card_residency()
            _drain_device(process_map, free_mb=_CARD16_FREE_IF_ALONE_MB)  # drained to sole residency
            scheduler.preload_models()
            self._complete_child_preload_acks(process_map, horde_model_map)
            if await scheduler.start_inference():
                dispatched = True
                break

        live_free = scheduler._measured_free_vram_mb()
        assert live_free is not None and live_free - _FLUX_WEIGHTS_MB >= _BASE_RESERVE_MB, (
            "precondition: the drained device holds the weights, so dispatching is safe"
        )
        assert dispatched is True, (
            "the device has drained enough to load the head's weights and the protective grace has lapsed, so the "
            "head must dispatch rather than parking indefinitely with no safe fallback"
        )

    @staticmethod
    def _complete_child_preload_acks(process_map: ProcessMap, horde_model_map: HordeModelMap) -> None:
        """Mutate the map to model the child-side PRELOAD_MODEL acknowledgement between scheduler ticks."""
        for process in process_map.values():
            if process.last_control_flag != HordeControlFlag.PRELOAD_MODEL:
                continue
            model_name = process.loaded_horde_model_name
            if model_name is None:
                continue
            process.last_process_state = HordeProcessState.PRELOADED_MODEL
            process.last_control_flag = None
            horde_model_map.update_entry(
                horde_model_name=model_name, load_state=ModelLoadState.LOADED_IN_RAM, process_id=process.process_id
            )


class TestMakingRoomNeverStrandsInflight:
    """Evicting models to make room for a head never unloads a model an in-progress job is using."""

    async def test_head_of_queue_eviction_spares_an_in_progress_model(self) -> None:
        """The last-resort head-of-queue reclaim never evicts a live in-progress model.

        A reclaim that unloaded an in-flight job's model would strand that job too: a second wedge the room-making
        path must never cause.
        """
        # Process one is the head's loader (spared as the target); process two runs an in-progress SDXL job;
        # process three holds an idle resident model that is fair game to reclaim.
        head_loader = make_mock_process_info(1, model_name=_FLUX_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        busy = make_mock_process_info(2, model_name=_SDXL_A, state=HordeProcessState.INFERENCE_STARTING)
        idle_resident = make_mock_process_info(3, model_name=_SDXL_C, state=HordeProcessState.WAITING_FOR_JOB)
        for proc in (head_loader, busy, idle_resident):
            proc.total_vram_mb = _CARD16_TOTAL_MB
            proc.vram_usage_mb = _CARD16_TOTAL_MB - _FLUX16_ESTABLISH_FREE_NOW_MB
        process_map = ProcessMap({1: head_loader, 2: busy, 3: idle_resident})

        horde_model_map = HordeModelMap(root={})
        _record_loaded_model(
            horde_model_map, model_name=_FLUX_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1
        )
        _record_loaded_model(horde_model_map, model_name=_SDXL_A, load_state=ModelLoadState.IN_USE, process_id=2)
        _record_loaded_model(
            horde_model_map, model_name=_SDXL_C, load_state=ModelLoadState.LOADED_IN_VRAM, process_id=3
        )

        job_tracker = JobTracker()
        running_sdxl = make_job_pop_response(_SDXL_A)
        await mark_job_in_progress_async(job_tracker, running_sdxl)

        bridge_data = _bridge_data()
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=horde_model_map,
            bridge_data=bridge_data,
            max_concurrent=1,
            max_inference=3,
        )
        scheduler._process_lifecycle = _make_real_plm(
            process_map=process_map, job_tracker=job_tracker, horde_model_map=horde_model_map, bridge_data=bridge_data
        )

        scheduler.unload_models_from_vram(head_loader, under_pressure=True, for_head_of_queue=True)

        assert busy.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM, (
            "the in-progress job's model must never be evicted to make room for the head, or that job wedges too"
        )
