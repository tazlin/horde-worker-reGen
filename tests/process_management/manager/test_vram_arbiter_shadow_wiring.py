"""Wiring tests for the VRAM arbiter: per-cycle ticking, shared injection, and release-target honesty."""

from __future__ import annotations

import time

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.resources.vram_arbiter import MeasuredVramSnapshot
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager


async def _noop_sleep(_delay: float) -> None:
    return None


def _tickable_manager() -> HordeWorkerProcessManager:
    process_manager = make_testable_process_manager()
    process_manager._sleep = _noop_sleep
    process_manager._last_status_message_time = time.time()
    return process_manager


class TestShadowCycleWiring:
    """The arbiter is constructed, injected, and ticked once per control-loop iteration."""

    async def test_control_loop_tick_begins_a_cycle(self) -> None:
        """A control-loop tick installs a frozen cycle snapshot on the shared arbiter."""
        process_manager = _tickable_manager()
        assert process_manager._vram_arbiter.has_cycle is False
        assert await process_manager._control_loop_tick() is True
        assert process_manager._vram_arbiter.has_cycle is True

    async def test_the_same_arbiter_is_injected_into_every_collaborator(self) -> None:
        """The scheduler and both orchestrators hold the manager's single arbiter instance."""
        process_manager = _tickable_manager()
        arbiter = process_manager._vram_arbiter
        assert process_manager._inference_scheduler._vram_arbiter is arbiter
        assert process_manager._post_process_orchestrator._vram_arbiter is arbiter
        assert process_manager._disaggregation_orchestrator._vram_arbiter is arbiter

    async def test_scheduler_snapshot_builder_produces_a_device_entry(self) -> None:
        """The scheduler assembles at least one device state for the frozen cycle snapshot."""
        process_manager = _tickable_manager()
        snapshot = process_manager._inference_scheduler.build_vram_arbiter_snapshot()
        assert isinstance(snapshot, MeasuredVramSnapshot)
        assert len(snapshot.devices) >= 1


class TestReleaseTargetHonesty:
    """The RELEASE_CACHE target set qualifies lanes by measured reclaimable cache, not bare reservation.

    A lane whose reservation is its resident footprint (a component lane holding encoders, allocated tracking
    reserved) is never a target: releasing its cache frees nothing, and emitting that rung would keep the
    escalation ladder non-empty forever. A genuinely idle cache-holder (reserved far above allocated) is.
    """

    async def test_resident_weight_lane_is_not_a_release_target_but_a_cache_holder_is(self) -> None:
        """A component lane whose reserved matches its resident encoders is excluded; an idle cache-holder is not."""
        process_manager = _tickable_manager()
        scheduler = process_manager._inference_scheduler

        resident_component = make_mock_process_info(
            process_id=50,
            model_name=None,
            process_type=HordeProcessType.COMPONENT,
        )
        resident_component.process_reserved_mb = 1700
        resident_component.process_allocated_mb = 1690

        cache_holder = make_mock_process_info(
            process_id=51,
            model_name=None,
            process_type=HordeProcessType.INFERENCE,
        )
        cache_holder.process_reserved_mb = 4000
        cache_holder.process_allocated_mb = 500

        scheduler._process_map[50] = resident_component
        scheduler._process_map[51] = cache_holder

        idle, _busy = scheduler._gpu_process_activity_ids(None)
        assert 50 not in idle
        assert 51 in idle

    async def test_unreported_allocation_is_not_assumed_to_hold_cache(self) -> None:
        """Without a measured in-use figure the reclaimable margin is unknown, so the lane is not targeted."""
        process_manager = _tickable_manager()
        scheduler = process_manager._inference_scheduler
        lane = make_mock_process_info(process_id=52, model_name=None, process_type=HordeProcessType.INFERENCE)
        lane.process_reserved_mb = 3000
        lane.process_allocated_mb = None
        scheduler._process_map[52] = lane
        idle, _busy = scheduler._gpu_process_activity_ids(None)
        assert 52 not in idle
