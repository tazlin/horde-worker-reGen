"""Whether a single-lane worker can swap the model its only lane holds for the head's.

A worker configured with one inference process has exactly one place to put a model. The reclaim surfaces the
admission path offers spare the head's own target slot, because on a multi-lane worker that slot is about to
be written anyway; with one lane it is also the only holder of the memory the head needs, so the sparing has
to distinguish what sits there rather than which slot it is.

These tests hold the topology fixed and vary only what the lane holds and by how much the head is short, so
the answer cannot come from the sizing: the same head, on the same card, is served from an empty lane, and
the occupied-lane arms differ from it only in what already sits on the card.

What they establish is that the sparing keys on model identity, not on the slot. The head's own weights,
wherever they sit, are credit against its load and are never reclaimed; a foreign idle resident on the target
lane is evicted as the ordinary model swap, at any size of shortfall.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.admission_identity import _ADMISSION_NOISE_BUFFER_MB
from horde_worker_regen.process_management.resources.resource_budget import (
    predict_job_sampling_vram_mb,
    predict_job_weight_mb,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    FakeClock,
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_model_metadata,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_RESIDENT_MODEL = "resident_sdxl"
"""The model the single lane already holds, idle, when the head arrives wanting something else."""

_HEAD_MODEL = "head_sdxl"
"""The head of the queue's model: a different checkpoint of the same class as the resident one."""

_CARD_TOTAL_MB = 16375.0
_VRAM_RESERVE_MB = 3096.0
_RAM_RESERVE_MB = 8192.0

_HEAD_JOB_SHAPE = {"width": 512, "height": 512, "ddim_steps": 20}
"""One ordinary SDXL job shape, so the head's cost is the library's own prediction rather than a fixture."""

_WITHIN_BAND_FREE_MB = 7000.0
"""Device-free with the lane's weights in place, leaving the head short by less than the measured-attempt band."""

_BEYOND_BAND_FREE_MB = 5000.0
"""Device-free with the lane's weights in place, leaving the head short by more than that band allows."""

_LANE = 0
"""The worker's only inference process slot."""


def _bridge_data() -> Mock:
    """Budget-on, single-lane bridge data offering both models."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        whole_card_exclusive_residency=False,
        safety_on_gpu=False,
        vram_reserve_mb=_VRAM_RESERVE_MB,
        ram_reserve_mb=_RAM_RESERVE_MB,
        unload_models_from_vram_often=False,
        image_models_to_load=[_RESIDENT_MODEL, _HEAD_MODEL],
        max_threads=1,
        queue_size=2,
    )


class _Device:
    """The card's truthful free-VRAM reading, which the scenario moves only when weights actually leave it."""

    def __init__(self, free_mb: float) -> None:
        """Start the card at ``free_mb`` free."""
        self.free_mb = free_mb

    def __call__(self, _device_index: int | None = None) -> float:
        """Return the current reading, matching the scheduler's device-free provider contract."""
        return self.free_mb


def _head_cost_mb() -> tuple[float, float]:
    """Return the head's predicted weight and sampling-peak cost, from the production predictors.

    Returns:
        The head's weight footprint and its whole-job sampling peak, both in MB.
    """
    job = make_job_pop_response(_HEAD_MODEL, **_HEAD_JOB_SHAPE)  # pyrefly: ignore - shape is a literal kwargs dict
    weight_mb = predict_job_weight_mb(job, "stable_diffusion_xl")
    peak_mb = predict_job_sampling_vram_mb(job, "stable_diffusion_xl")
    assert weight_mb is not None and peak_mb is not None, (
        "the head must be priceable for the scenario to mean anything"
    )
    return weight_mb, peak_mb


async def _build_scenario(
    *,
    lane_holds_resident: bool,
    free_with_resident_mb: float,
    resident_model: str = _RESIDENT_MODEL,
) -> tuple[InferenceScheduler, ProcessMap, _Device, JobTracker]:
    """Build the one-lane worker with a pending head, with the lane occupied or empty.

    The occupied and empty arms differ in exactly two facts: whether the lane holds a resident model, and
    whether that model's weights are on the card. Everything the head is priced against is identical.

    Args:
        lane_holds_resident: Whether the single lane already holds an idle resident model.
        free_with_resident_mb: The card's free reading while those weights are on it; the empty arm reads the
            same figure with the weights returned.
        resident_model: Which model the lane holds. The default is a different checkpoint than the head's;
            naming the head's own model instead puts the lane in the resident-credit case.

    Returns:
        The scheduler, its process map, the card's free-VRAM reading, and the job tracker holding the head.
    """
    resident_weight_mb = predict_job_weight_mb(
        make_job_pop_response(resident_model, **_HEAD_JOB_SHAPE),  # pyrefly: ignore - shape is a literal kwargs dict
        "stable_diffusion_xl",
    )
    assert resident_weight_mb is not None

    process_map = ProcessMap(
        {
            _LANE: make_mock_process_info(
                _LANE,
                model_name=resident_model if lane_holds_resident else None,
                state=HordeProcessState.WAITING_FOR_JOB,
            ),
        },
    )
    horde_model_map = HordeModelMap(root={})
    if lane_holds_resident:
        horde_model_map.update_entry(
            horde_model_name=resident_model,
            load_state=ModelLoadState.LOADED_IN_VRAM,
            process_id=_LANE,
        )

    device = _Device(free_with_resident_mb if lane_holds_resident else free_with_resident_mb + resident_weight_mb)
    bridge_data = _bridge_data()
    job_tracker = JobTracker()
    metadata = make_test_model_metadata(
        {
            name: make_mock_model_reference_record(name, baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl)
            for name in (_RESIDENT_MODEL, _HEAD_MODEL)
        },
    )
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        model_metadata=metadata,
        max_concurrent=1,
        max_inference=1,
        card_runtimes=make_test_card_runtimes(
            config=bridge_data,
            target_process_count=1,
            total_vram_mb=_CARD_TOTAL_MB,
        ),
        device_free_mb=None,
        clock=FakeClock(1000.0),
    )
    scheduler.set_device_free_mb_provider(device)
    lifecycle = Mock()
    lifecycle.get_processes_with_model_for_queued_job = Mock(return_value=[])
    lifecycle.is_model_load_quarantined = Mock(return_value=False)
    lifecycle.is_safety_gpu_paused = False
    lifecycle.post_process_lane_enabled = Mock(return_value=False)
    lifecycle.component_lane_enabled = Mock(return_value=False)
    lifecycle.vae_lane_enabled = Mock(return_value=False)
    scheduler._process_lifecycle = lifecycle

    await track_popped_job_async(
        job_tracker,
        make_job_pop_response(_HEAD_MODEL, **_HEAD_JOB_SHAPE),  # pyrefly: ignore - shape is a literal kwargs dict
    )
    return scheduler, process_map, device, job_tracker


def _apply_child_side_effects(
    process_map: ProcessMap,
    horde_model_map: HordeModelMap,
    device: _Device,
) -> None:
    """Model what the child processes do with the control flags this tick issued.

    An unload returns the resident weights to the card, and a preload lands the head's model in the lane.
    Without this the scenario could not distinguish "no eviction was ever ordered" from "an eviction was
    ordered and the card did not recover", which is the whole question.
    """
    for process in process_map.values():
        if process.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
            unloaded = process.loaded_horde_model_name
            if unloaded is not None:
                weight_mb = predict_job_weight_mb(
                    make_job_pop_response(unloaded, **_HEAD_JOB_SHAPE),  # pyrefly: ignore - literal kwargs dict
                    "stable_diffusion_xl",
                )
                device.free_mb += weight_mb or 0.0
                horde_model_map.expire_entry(unloaded)
            process.loaded_horde_model_name = None
            process.last_control_flag = None
            continue
        if process.last_control_flag == HordeControlFlag.PRELOAD_MODEL:
            model_name = process.loaded_horde_model_name
            process.last_control_flag = None
            if model_name is None:
                continue
            process.last_process_state = HordeProcessState.PRELOADED_MODEL
            horde_model_map.update_entry(
                horde_model_name=model_name,
                load_state=ModelLoadState.LOADED_IN_RAM,
                process_id=process.process_id,
            )


async def _drive_until_served(
    scheduler: InferenceScheduler,
    process_map: ProcessMap,
    device: _Device,
    *,
    ticks: int = 30,
) -> tuple[bool, list[HordeControlFlag]]:
    """Run the scheduling loop for ``ticks`` cycles, reporting service and every control flag issued.

    Returns:
        Whether ``start_inference`` dispatched the head within the window, and the control flags the
        scheduler sent to the lane over the whole run.
    """
    clock = scheduler._clock
    flags: list[HordeControlFlag] = []
    for _ in range(ticks):
        scheduler.preload_models()
        flag = process_map[_LANE].last_control_flag
        if flag is not None:
            flags.append(flag)
        _apply_child_side_effects(process_map, scheduler._horde_model_map, device)
        if await scheduler.start_inference():
            return True, flags
        if isinstance(clock, FakeClock):
            clock.advance(10.0)
    return False, flags


def _resident_weight_mb() -> float:
    """Return the predicted weight footprint of the model the lane holds."""
    weight_mb = predict_job_weight_mb(
        make_job_pop_response(_RESIDENT_MODEL, **_HEAD_JOB_SHAPE),  # pyrefly: ignore - shape is a literal kwargs dict
        "stable_diffusion_xl",
    )
    assert weight_mb is not None
    return weight_mb


def _assert_swap_would_seat_the_head(device: _Device) -> None:
    """Assert the head is blocked by the resident weights alone, and fits once they are returned."""
    _, peak_mb = _head_cost_mb()
    assert device.free_mb - _ADMISSION_NOISE_BUFFER_MB < peak_mb, (
        "precondition: the head cannot fit beside the resident model"
    )
    assert device.free_mb + _resident_weight_mb() - _ADMISSION_NOISE_BUFFER_MB >= peak_mb, (
        "precondition: the head fits once the resident model's weights are returned to the card"
    )


class TestSingleLaneDifferentModelSwap:
    """What a one-lane worker does with a head that wants a different model than the one it holds."""

    async def test_an_empty_lane_seats_the_head(self) -> None:
        """The control arm: with the lane free and the weights off the card, the head is admitted and runs.

        This pins the sizing. The head fits the card once the resident model's weights are not on it, so a
        failure to serve either paired scenario cannot be attributed to a head that was too big all along.
        """
        scheduler, process_map, device, _ = await _build_scenario(
            lane_holds_resident=False,
            free_with_resident_mb=_BEYOND_BAND_FREE_MB,
        )
        _, peak_mb = _head_cost_mb()
        assert device.free_mb - _ADMISSION_NOISE_BUFFER_MB >= peak_mb, (
            "precondition: the card with the lane's weights returned genuinely holds the head's sampling peak"
        )

        served, _flags = await _drive_until_served(scheduler, process_map, device)

        assert served is True, "an empty lane on a card with room must seat the head"
        assert process_map[_LANE].loaded_horde_model_name == _HEAD_MODEL

    async def test_a_within_band_shortfall_is_served_by_the_swap(self) -> None:
        """A head short by less than the uncertainty band is given room rather than gambled through.

        A shortfall this small would also clear the arbiter's measured-attempt hatch, which lets a starved
        head take one real load on a card nothing can free. The card can be freed here, so the hatch is not
        reached: the foreign resident is visible as reclaimable, the card is not converged-empty, and the head
        is served by an ordered eviction. Buying the room outright is the cheaper of the two, and it is the
        one a head at any shortfall can rely on.
        """
        scheduler, process_map, device, job_tracker = await _build_scenario(
            lane_holds_resident=True,
            free_with_resident_mb=_WITHIN_BAND_FREE_MB,
        )
        _assert_swap_would_seat_the_head(device)
        assert process_map[_LANE].is_process_busy() is False, "precondition: the resident lane is idle"
        assert not job_tracker.jobs_in_progress, "precondition: nothing is running, so nothing protects the resident"

        served, flags = await _drive_until_served(scheduler, process_map, device)

        assert served is True
        assert HordeControlFlag.UNLOAD_MODELS_FROM_VRAM in flags, "the head was given room, not gambled through"
        arbiter = scheduler._ensure_preload_arbiter()
        assert arbiter.measured_attempts == 0, "a card with room to reclaim is not converged-empty, so no attempt"

    async def test_a_resident_head_model_is_credited_and_not_evicted(self) -> None:
        """The lane's copy of the head's *own* model is credit against the load, so nothing evicts it.

        The control for the swap: the same one-lane topology, the same card reading, the same shortfall
        against a card priced without the credit. Only the identity of the resident model differs. Its weights
        are already the head's, so the head is priced net of them and seated on them, and the eviction
        surfaces must leave the lane alone. Were the target slot reclaimable on slot identity rather than
        model identity, this head would evict the very weights it is about to reload.
        """
        scheduler, process_map, device, job_tracker = await _build_scenario(
            lane_holds_resident=True,
            free_with_resident_mb=_BEYOND_BAND_FREE_MB,
            resident_model=_HEAD_MODEL,
        )
        assert process_map[_LANE].is_process_busy() is False, "precondition: the resident lane is idle"
        assert not job_tracker.jobs_in_progress, "precondition: nothing is running, so nothing protects the resident"

        served, flags = await _drive_until_served(scheduler, process_map, device)

        assert served is True, "a head whose model is already resident is priced net of those weights"
        assert HordeControlFlag.UNLOAD_MODELS_FROM_VRAM not in flags, (
            "the head's own resident weights are credit against its load, never memory to reclaim for it"
        )
        assert process_map[_LANE].loaded_horde_model_name == _HEAD_MODEL

    async def test_a_beyond_band_shortfall_still_swaps_the_model(self) -> None:
        """The head must be served whatever the size of the shortfall the idle resident is holding.

        The lane is idle and the model it holds is wanted by nothing: no job in progress, no job queued
        behind the head. Its weights are the only memory between the head and the card, and the lane is the
        very process the head would load into, so displacing them costs the worker nothing it was keeping.
        How far short the arithmetic falls decides how much is gained by the swap, never whether the swap is
        available.
        """
        scheduler, process_map, device, job_tracker = await _build_scenario(
            lane_holds_resident=True,
            free_with_resident_mb=_BEYOND_BAND_FREE_MB,
        )
        _assert_swap_would_seat_the_head(device)
        assert process_map[_LANE].is_process_busy() is False, "precondition: the resident lane is idle"
        assert not job_tracker.jobs_in_progress, "precondition: nothing is running, so nothing protects the resident"

        served, flags = await _drive_until_served(scheduler, process_map, device)

        assert HordeControlFlag.UNLOAD_MODELS_FROM_VRAM in flags, (
            "the lane holding the memory the head needs was asked to give it up"
        )
        assert served is True, (
            "the only lane holds an idle model nothing wants and the head needs its memory, so some path must "
            "trade one for the other rather than leaving the worker serving nothing"
        )
