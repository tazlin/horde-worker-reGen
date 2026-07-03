"""RAM-cache preservation around the VRAM over-budget best-effort admit.

A best-effort VRAM admit historically evicted an idle resident model from system RAM unconditionally,
on the reasoning that a heavy head loads its checkpoint through RAM first. On a host with ample RAM
headroom that eviction buys nothing: the incoming load fits without it, and the only effect is
destroying a sibling's warm RAM cache so the next job for that model reloads its checkpoint from disk.
On a busy multi-model worker the VRAM budget can route ordinary heads through the best-effort admit
many times an hour, so the unconditional purge turns the RAM cache into a disk-reload treadmill (and
the allocator-stuck slots it leaves behind are then recycled, compounding the churn).

The contract pinned here: the terminal admit consults the RAM budget and reclaims idle RAM residents
only when the incoming load does not fit measured available memory. The pressure paths (the RAM
verdict's own reclaim, and the RAM governor's eviction under the danger floor) are unchanged.

A second contract on the reclaim itself: when an idle RAM resident must be sacrificed, the victim is
the cheapest cache to rebuild (the smallest size tier), never a card-dominating checkpoint whose disk
reload costs several times an ordinary model's, unless it is the only candidate.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap, ModelLoadState
from horde_worker_regen.process_management.resources.resource_budget import BudgetVerdict, StreamForecast
from horde_worker_regen.process_management.scheduling.inference_scheduler import VramGateResult
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_SIBLING_MODEL = "WAI-NSFW-illustrious-SDXL"
_HEAD_MODEL = "AlbedoBase XL (SDXL)"
_EXTRA_LARGE_MODEL = "Flux.1-Schnell fp8 (Compact)"


def _resident_in_ram(horde_model_map: HordeModelMap, model: str, process_id: int) -> None:
    """Record ``model`` as an idle RAM-resident copy owned by ``process_id``."""
    horde_model_map.update_entry(
        horde_model_name=model,
        load_state=ModelLoadState.LOADED_IN_RAM,
        process_id=process_id,
    )


async def _overbudget_admit_setup():  # noqa: ANN202
    """Build a scheduler mid over-budget admit: head deferred by VRAM, sibling holding a warm RAM cache.

    The sibling's VRAM copy is already unloaded (``last_control_flag`` reflects the earlier gentle
    reclaim), so both reclaim passes free nothing and the verdict reaches the best-effort admit rung.
    """
    available = make_mock_process_info(1, model_name=None)
    sibling = make_mock_process_info(2, model_name=_SIBLING_MODEL)
    sibling.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
    process_map = ProcessMap({1: available, 2: sibling})
    horde_model_map = HordeModelMap(root={})
    _resident_in_ram(horde_model_map, _SIBLING_MODEL, 2)
    job_tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(max_threads=2),
        max_concurrent=2,
        max_inference=2,
    )
    job = make_job_pop_response(model=_HEAD_MODEL)
    await job_tracker.record_popped_job(job)
    scheduler._vram_budget.check_job = Mock(
        return_value=BudgetVerdict(fits=False, predicted_mb=None, available_mb=5000.0, reserve_mb=2048.0),
    )
    return scheduler, job, sibling


def _apply_verdict(scheduler, job) -> VramGateResult:  # noqa: ANN001
    """Run the VRAM verdict for the head as an idle-device head blocker."""
    available = scheduler._process_map[1]
    forecast = StreamForecast(
        weights_mb=4900.0,
        reserve_mb=2048.0,
        free_now_mb=10162.0,
        free_if_alone_mb=20801.0,
        free_after_model_evict_mb=13819.0,
        total_vram_mb=24074.0,
        per_process_overhead_mb=3273.0,
        marginal_process_overhead_mb=1746.0,
    )
    return scheduler._apply_vram_verdict(
        job,
        available,
        None,
        forecast,
        is_head_blocker=True,
        target_device_index=None,
        no_live_resource_consumer=True,
    )


class TestOverbudgetAdmitRamPurgeGating:
    """The terminal admit's RAM reclaim runs only when the load does not fit available RAM."""

    async def test_admit_with_ram_headroom_preserves_sibling_ram_cache(self) -> None:
        """With the RAM budget fitting the head, the admit must not evict the sibling's RAM copy."""
        scheduler, job, sibling = await _overbudget_admit_setup()
        scheduler._ram_budget.check_job = Mock(
            return_value=BudgetVerdict(fits=True, predicted_mb=8000.0, available_mb=30000.0, reserve_mb=4096.0),
        )

        result = _apply_verdict(scheduler, job)

        assert result is VramGateResult.ADMIT_OVER_BUDGET
        assert sibling.loaded_horde_model_name == _SIBLING_MODEL
        assert sibling.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_RAM

    async def test_admit_without_ram_headroom_still_reclaims(self) -> None:
        """CONTROL: when the head's RAM cost does not fit, the admit reclaims the idle resident copy."""
        scheduler, job, sibling = await _overbudget_admit_setup()
        scheduler._ram_budget.check_job = Mock(
            return_value=BudgetVerdict(fits=False, predicted_mb=8000.0, available_mb=3000.0, reserve_mb=4096.0),
        )

        result = _apply_verdict(scheduler, job)

        assert result is VramGateResult.ADMIT_OVER_BUDGET
        assert sibling.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM


class TestRamUnloadVictimChoice:
    """The idle-RAM reclaim sacrifices the cheapest cache to rebuild, not whichever slot enumerates first."""

    def _two_resident_scheduler(self):  # noqa: ANN202
        """Two idle RAM residents: an EXTRA_LARGE checkpoint on the first slot, a light one on the second."""
        heavy = make_mock_process_info(1, model_name=_EXTRA_LARGE_MODEL)
        light = make_mock_process_info(2, model_name=_SIBLING_MODEL)
        process_map = ProcessMap({1: heavy, 2: light})
        horde_model_map = HordeModelMap(root={})
        _resident_in_ram(horde_model_map, _EXTRA_LARGE_MODEL, 1)
        _resident_in_ram(horde_model_map, _SIBLING_MODEL, 2)
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=JobTracker(),
            bridge_data=make_mock_bridge_data(max_threads=2),
            max_concurrent=2,
            max_inference=2,
        )
        return scheduler, heavy, light

    def test_light_model_evicted_before_extra_large(self) -> None:
        """Under pressure with both tiers idle, the light model's cache is the one sacrificed."""
        scheduler, heavy, light = self._two_resident_scheduler()

        reclaimed = scheduler.unload_models(under_pressure=True)

        assert reclaimed is True
        assert light.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM
        assert heavy.loaded_horde_model_name == _EXTRA_LARGE_MODEL
        assert heavy.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_RAM

    def test_extra_large_still_evicted_when_it_is_the_only_candidate(self) -> None:
        """CONTROL: the tier preference never wedges the reclaim; a lone heavy resident is still evictable."""
        heavy = make_mock_process_info(1, model_name=_EXTRA_LARGE_MODEL)
        process_map = ProcessMap({1: heavy})
        horde_model_map = HordeModelMap(root={})
        _resident_in_ram(horde_model_map, _EXTRA_LARGE_MODEL, 1)
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=JobTracker(),
            bridge_data=make_mock_bridge_data(max_threads=2),
            max_concurrent=2,
            max_inference=2,
        )

        reclaimed = scheduler.unload_models(under_pressure=True)

        assert reclaimed is True
        assert heavy.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM
