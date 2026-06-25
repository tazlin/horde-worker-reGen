"""Tests for A5.3 per-card VRAM budget, eviction, and whole-card residency.

On a multi-GPU host each card is an independent VRAM domain: the budget reads that card's measured free
VRAM, eviction reclaims only that card's idle residents, and a whole-card residency claims (and later
restores) one card without disturbing another. The single shared safety process is moved off-GPU only for a
residency on the card it is pinned to. A single-GPU host keeps the worker-wide reading throughout, so the
device_index params default to None and behave exactly as before.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_model_metadata,
    make_test_runtime_config,
    mark_job_in_progress_async,
    track_popped_job_async,
)


def _process_with_vram(
    process_id: int,
    *,
    device_index: int,
    total_vram_mb: int,
    vram_usage_mb: int,
    model_name: str | None = None,
    state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB,
) -> HordeProcessInfo:
    """A mock inference process pinned to a card with set VRAM figures."""
    proc = make_mock_process_info(process_id, model_name=model_name, state=state, device_index=device_index)
    proc.total_vram_mb = total_vram_mb
    proc.vram_usage_mb = vram_usage_mb
    return proc


def _two_cards() -> dict[int, CardRuntime]:
    """A 24GB card 0 and an 8GB card 1, both serving stable_diffusion, with two-process pools."""
    rt0 = make_test_card_runtimes(device_indices=(0,), target_process_count=2, total_vram_mb=24576)
    rt1 = make_test_card_runtimes(device_indices=(1,), target_process_count=2, total_vram_mb=8192)
    return {0: rt0[0], 1: rt1[1]}


def _make_scheduler(
    *,
    process_map: ProcessMap,
    card_runtimes: dict[int, CardRuntime] | None,
    job_tracker: JobTracker | None = None,
    process_lifecycle: Mock | None = None,
    safety_on_gpu: bool = False,
) -> InferenceScheduler:
    """Build an InferenceScheduler with a per-card runtime plan, for the budget/residency helpers."""
    bridge_data = make_mock_bridge_data()
    bridge_data.max_threads = 2
    bridge_data.safety_on_gpu = safety_on_gpu
    bridge_data.whole_card_exclusive_residency = True
    bridge_data.whole_card_residency_safety_off_gpu = True
    return InferenceScheduler(
        state=WorkerState(),
        process_map=process_map,
        horde_model_map=HordeModelMap(root={}),
        job_tracker=job_tracker if job_tracker is not None else JobTracker(),
        process_lifecycle=process_lifecycle
        if process_lifecycle is not None
        else Mock(is_model_load_quarantined=Mock(return_value=False), is_safety_gpu_paused=False),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(),
        card_runtimes=card_runtimes,
        max_concurrent_inference_processes=2,
        max_inference_processes=4,
        lru=LRUCache(4),
    )


class TestPerCardVramReads:
    """The process map reports free/total VRAM and live-context counts per card."""

    def test_free_and_total_vram_are_scoped_per_card(self) -> None:
        """Each card's free/total VRAM is read from its own processes; the unfiltered call is the min/max."""
        process_map = ProcessMap(
            {
                0: _process_with_vram(0, device_index=0, total_vram_mb=24000, vram_usage_mb=4000),
                1: _process_with_vram(1, device_index=1, total_vram_mb=8000, vram_usage_mb=6000),
            },
        )
        assert process_map.get_free_vram_mb(device_index=0) == 20000.0
        assert process_map.get_free_vram_mb(device_index=1) == 2000.0
        # The unfiltered (single-GPU / worker-wide) reading is the most conservative across cards.
        assert process_map.get_free_vram_mb() == 2000.0
        assert process_map.get_reported_total_vram_mb(device_index=0) == 24000.0
        assert process_map.get_reported_total_vram_mb(device_index=1) == 8000.0
        assert process_map.get_reported_total_vram_mb() == 24000.0

    def test_live_context_count_is_scoped_per_card(self) -> None:
        """num_loaded_inference_processes counts only the named card's live processes when filtered."""
        process_map = ProcessMap(
            {
                0: _process_with_vram(0, device_index=0, total_vram_mb=24000, vram_usage_mb=0),
                1: _process_with_vram(1, device_index=0, total_vram_mb=24000, vram_usage_mb=0),
                2: _process_with_vram(2, device_index=1, total_vram_mb=8000, vram_usage_mb=0),
            },
        )
        assert process_map.num_loaded_inference_processes(device_index=0) == 2
        assert process_map.num_loaded_inference_processes(device_index=1) == 1
        assert process_map.num_loaded_inference_processes() == 3


class TestPerCardEviction:
    """VRAM reclamation evicts only the idle residents on the target card."""

    def test_unload_targets_only_the_named_card(self) -> None:
        """An eviction scoped to card 0 leaves card 1's idle resident model untouched."""
        loader = _process_with_vram(0, device_index=0, total_vram_mb=24000, vram_usage_mb=0)
        card0_resident = _process_with_vram(
            1,
            device_index=0,
            total_vram_mb=24000,
            vram_usage_mb=12000,
            model_name="model_on_card0",
        )
        card1_resident = _process_with_vram(
            2,
            device_index=1,
            total_vram_mb=8000,
            vram_usage_mb=4000,
            model_name="model_on_card1",
        )
        process_map = ProcessMap({0: loader, 1: card0_resident, 2: card1_resident})
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=_two_cards())

        freed = scheduler.unload_models_from_vram(
            loader,
            under_pressure=True,
            for_head_of_queue=True,
            device_index=0,
        )

        assert freed is True
        assert card0_resident.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        # Card 1's resident model is in a different VRAM domain; reclaiming card 0 must not touch it.
        assert card1_resident.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM


class TestPerCardResidency:
    """Whole-card residencies are held, restored, and safety-gated per card."""

    def _forecast(self, *, total_vram_mb: float, max_resident: int) -> Mock:
        forecast = Mock()
        forecast.max_resident_processes = Mock(return_value=max_resident)
        forecast.total_vram_mb = total_vram_mb
        forecast.fits_weights_now = True
        return forecast

    def test_two_cards_hold_independent_residencies(self) -> None:
        """Establishing a residency on each card records both, keyed by their device index."""
        process_map = ProcessMap(
            {
                0: _process_with_vram(0, device_index=0, total_vram_mb=24000, vram_usage_mb=0),
                1: _process_with_vram(1, device_index=1, total_vram_mb=8000, vram_usage_mb=0),
            },
        )
        lifecycle = Mock(is_safety_gpu_paused=False, pause_safety_on_gpu=Mock(return_value=False))
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=_two_cards(), process_lifecycle=lifecycle)

        scheduler._establish_whole_card_residency(
            make_job_pop_response("big_model"),
            self._forecast(total_vram_mb=24000, max_resident=1),
            announce=True,
            device_index=0,
        )
        scheduler._establish_whole_card_residency(
            make_job_pop_response("other_model"),
            self._forecast(total_vram_mb=8000, max_resident=1),
            announce=True,
            device_index=1,
        )

        held = dict(scheduler._held_residencies())
        assert set(held) == {0, 1}
        assert held[0].model == "big_model"
        assert held[1].model == "other_model"
        assert scheduler._residency_holder_for_model("big_model") == (True, 0)
        assert scheduler._residency_holder_for_model("other_model") == (True, 1)

    async def test_restoring_one_card_leaves_the_other_held(self) -> None:
        """A drained residency on card 0 is restored while card 1's still-active residency is kept."""
        process_map = ProcessMap(
            {
                0: _process_with_vram(0, device_index=0, total_vram_mb=24000, vram_usage_mb=0),
                1: _process_with_vram(1, device_index=1, total_vram_mb=8000, vram_usage_mb=0),
            },
        )
        job_tracker = JobTracker()
        lifecycle = Mock(
            is_safety_gpu_paused=False,
            restore_safety_on_gpu=Mock(return_value=False),
            scale_inference_processes=Mock(return_value=2),
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(),
            job_tracker=job_tracker,
            process_lifecycle=lifecycle,
        )
        # Card 0's residency has drained (its model is queued nowhere); card 1's is still serving a queued job.
        scheduler._residency_state(0).model = "drained_model"
        scheduler._residency_state(1).model = "active_model"
        active_job = make_job_pop_response("active_model")
        await track_popped_job_async(job_tracker, active_job)
        await mark_job_in_progress_async(job_tracker, active_job)

        scheduler._restore_siblings_after_whole_card()

        assert scheduler._residency_state(0).model is None, "drained card-0 residency should be restored"
        assert scheduler._residency_state(1).model == "active_model", "active card-1 residency must be kept"

    def test_safety_is_paused_only_for_the_safety_card(self) -> None:
        """The single safety process (pinned to the lowest-index card) is paused only by that card's residency."""
        process_map = ProcessMap(
            {0: _process_with_vram(0, device_index=0, total_vram_mb=24000, vram_usage_mb=0)},
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(),
            safety_on_gpu=True,
        )
        # Card 0 is the lowest index, so the one safety process sits there.
        assert scheduler._residency_should_pause_safety(0) is True
        assert scheduler._residency_should_pause_safety(1) is False
        # The worker-wide (single-GPU) key always qualifies when safety-off-GPU is configured.
        assert scheduler._residency_should_pause_safety(None) is True
