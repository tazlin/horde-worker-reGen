"""The VRAM arbiter as the authority for loading the safety process onto the GPU (SAFETY_LOAD).

The recurring safety-on-GPU seam (bringing the safety process back onto the card after a whole-card residency
freed its context) is gated on the arbiter: an over-committed card keeps safety off-GPU and re-asks, while a
card with room restores it. The initial cold-start safety load is not gated and is out of scope here.
"""

from __future__ import annotations

import uuid
from unittest.mock import Mock

from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
)
from tests.process_management.conftest import make_mock_bridge_data
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _install_cycle(scheduler, *, total_mb: float, committed_mb: float) -> None:  # noqa: ANN001
    """Freeze a crafted arbiter cycle on the scheduler with a device-free reading on card 0.

    The safety load is admitted against the truthful device-free reading net of the noise buffer: a card whose
    worker-committed footprint already consumes it (``device_free = total - committed`` near zero) defers, a
    card with real room admits. The reading is what the parent's NVML sees, so committed footprint maps to it
    directly here.
    """
    arbiter = VramArbiter()
    arbiter.begin_cycle(
        MeasuredVramSnapshot(
            devices={
                0: DeviceVramState(
                    total_vram_mb=total_mb,
                    baseline_mb=0.0,
                    committed_vram_mb=committed_mb,
                    planned_unmaterialized_mb=0.0,
                    committed_is_stale=False,
                    device_free_mb=max(0.0, total_mb - committed_mb),
                ),
            },
        ),
    )
    scheduler._vram_arbiter = arbiter


def _safety_scheduler():  # noqa: ANN202
    """A single-GPU scheduler configured for whole-card safety-off-GPU, with safety currently paused off-GPU."""
    bridge_data = make_mock_bridge_data(safety_on_gpu=True)
    bridge_data.whole_card_residency_safety_off_gpu = True
    scheduler = _make_inference_scheduler(process_map=ProcessMap({}), bridge_data=bridge_data)
    scheduler._process_lifecycle.is_safety_gpu_paused = True
    scheduler._process_lifecycle.restore_safety_on_gpu = Mock(return_value=True)
    return scheduler


async def _queue_safety_backlog(scheduler, *, depth: int) -> None:  # noqa: ANN001
    """Place completed jobs in the pending safety stage."""
    for _ in range(depth):
        job = Mock()
        job.id_ = uuid.uuid4()
        job.model = "stable_diffusion"
        job_info = Mock()
        job_info.sdk_api_job_info = job
        await scheduler._job_tracker.queue_for_safety(job_info)


class TestArbiterAdmitsSafetyGpuLoad:
    """The direct SAFETY_LOAD verdict over the frozen cycle measurement."""

    def test_over_committed_card_defers_the_safety_load(self) -> None:
        """A card committed to capacity does not admit the safety load's charge."""
        scheduler = _safety_scheduler()
        _install_cycle(scheduler, total_mb=16000.0, committed_mb=16000.0)
        assert scheduler._arbiter_admits_safety_gpu_load(None) is False

    def test_card_with_room_admits_the_safety_load(self) -> None:
        """A card with ample capacity admits the safety load's charge."""
        scheduler = _safety_scheduler()
        _install_cycle(scheduler, total_mb=24000.0, committed_mb=1000.0)
        assert scheduler._arbiter_admits_safety_gpu_load(None) is True

    def test_unwired_arbiter_admits(self) -> None:
        """With no arbiter cycle installed the safety load admits (missing-telemetry contract)."""
        scheduler = _safety_scheduler()
        assert scheduler._arbiter_admits_safety_gpu_load(None) is True


class TestDeferredSafetyLoadReconciler:
    """The per-tick reconciler restores a deferred safety load only once the card has room."""

    def test_reconciler_keeps_safety_off_while_the_card_is_over_committed(self) -> None:
        """An over-committed card leaves safety off-GPU and the restore is not issued this cycle."""
        scheduler = _safety_scheduler()
        _install_cycle(scheduler, total_mb=16000.0, committed_mb=16000.0)
        scheduler._restore_deferred_safety_gpu_load()
        scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()

    def test_reconciler_restores_safety_once_the_card_has_room(self) -> None:
        """A card with room re-asks and restores the deferred safety load."""
        scheduler = _safety_scheduler()
        _install_cycle(scheduler, total_mb=24000.0, committed_mb=1000.0)
        scheduler._restore_deferred_safety_gpu_load()
        scheduler._process_lifecycle.restore_safety_on_gpu.assert_called_once_with()

    async def test_reconciler_restores_during_deep_backlog_when_the_card_has_room(self) -> None:
        """A deep safety backlog does not strand a deferred safety load off-GPU."""
        scheduler = _safety_scheduler()
        await _queue_safety_backlog(scheduler, depth=3)
        _install_cycle(scheduler, total_mb=24000.0, committed_mb=1000.0)

        scheduler._restore_deferred_safety_gpu_load()

        scheduler._process_lifecycle.restore_safety_on_gpu.assert_called_once_with()

    async def test_reconciler_avoids_churning_a_shallow_backlog(self) -> None:
        """Shallow safety work still keeps the deferred restore from cycling the safety lane."""
        scheduler = _safety_scheduler()
        await _queue_safety_backlog(scheduler, depth=2)
        _install_cycle(scheduler, total_mb=24000.0, committed_mb=1000.0)

        scheduler._restore_deferred_safety_gpu_load()

        scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()

    def test_reconciler_is_a_noop_when_safety_is_not_paused(self) -> None:
        """With safety already on-GPU the reconciler does nothing (nothing to restore)."""
        scheduler = _safety_scheduler()
        scheduler._process_lifecycle.is_safety_gpu_paused = False
        _install_cycle(scheduler, total_mb=24000.0, committed_mb=1000.0)
        scheduler._restore_deferred_safety_gpu_load()
        scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()
