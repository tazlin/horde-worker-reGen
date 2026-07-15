"""Reproduces a disaggregation service lane stranded off the GPU by a post-processing borrow.

A post-processing drain that cannot fit its chain may borrow one idle disaggregation service lane (the VAE or
component lane) off the GPU to make room, restoring it once accepted PP work drains. The strand this file
reproduces: pausing the VAE lane disables disaggregation, so jobs route monolithic, which raises card pressure
and sustains a post-processing backlog the queue never fully drains. Gated only on a full drain, the loan is
then held indefinitely and disaggregation stays dead with no restore signal.

The specification here:

* a borrowed service lane is returned once no post-processing job has been actively processed for a bounded
  idle window, even while jobs remain queued, so a stalled queue cannot hold a disaggregation lane hostage;
* a reclaim-ladder lane pause that has outlived both responsible restore owners (a saturation episode and a PP
  borrow receipt) is self-healed once the card has been HEALTHY, restoring disaggregation availability; and
* an enabled-but-unavailable disaggregation pipeline emits one edge-latched warning that re-arms on recovery.
"""

from __future__ import annotations

import time

import pytest
from loguru import logger

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.resources.vram_arbiter import ActuatorCommandKind
from horde_worker_regen.process_management.workers import post_process_orchestrator as pp_orchestrator_module
from tests.process_management.regressions.test_post_process_drain_context_reclaim_repro import (
    _live_process,
    _live_shaped_manager,
)
from tests.process_management.workers import test_post_process_orchestration as pp_tests


async def test_borrowed_vae_lane_is_released_when_the_pp_queue_stalls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A borrowed VAE lane must not stay paused forever when the PP job it made room for never dispatches."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    clock = {"t": 1000.0}
    manager._post_process_orchestrator._clock = lambda: clock["t"]

    job_info = pp_tests._make_pp_job_info(["CodeFormers", "RealESRGAN_x4plus"])
    await manager._job_tracker.queue_for_post_processing(job_info)

    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True

    # The card is unchanged, so the chain never fits and never dispatches: nothing is being post-processed while
    # the job stays queued. Advance past the idle-release window but below the aging window, so the job is still
    # pending (not aged out): the borrowed disaggregation lane must be returned to break the wedge.
    clock["t"] += pp_orchestrator_module._BORROW_IDLE_RELEASE_SECONDS + 5.0
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()

    assert job_info in manager._job_tracker.jobs_pending_post_processing
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is False, (
        "VAE lane borrowed for a PP drain stayed paused while the queue stalled without dispatching; "
        "disaggregation is dead until the queue fully drains, which a sustained PP backlog never does"
    )
    # The lane is not re-borrowed for the same stalled episode (no pause/restore thrash every window).
    assert (
        ActuatorCommandKind.PAUSE_VAE_LANE not in manager._post_process_orchestrator._borrowed_service_lane_actuations
    )


async def test_borrowed_lane_re_enabled_after_queue_drains(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the stalled queue empties, a future PP drain may borrow a service lane again (suppression clears)."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    clock = {"t": 1000.0}
    manager._post_process_orchestrator._clock = lambda: clock["t"]

    job_info = pp_tests._make_pp_job_info(["CodeFormers", "RealESRGAN_x4plus"])
    await manager._job_tracker.queue_for_post_processing(job_info)
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True

    clock["t"] += pp_orchestrator_module._BORROW_IDLE_RELEASE_SECONDS + 5.0
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    assert manager._post_process_orchestrator._service_lane_borrow_suppressed is True

    # The stalled job leaves the queue; the suppression that prevents re-borrow must clear so a later episode
    # can borrow a lane again rather than being permanently barred by one stall.
    await manager._job_tracker.abandon_pending_post_processing(job_info)
    await manager.start_post_processing()
    assert manager._post_process_orchestrator._service_lane_borrow_suppressed is False


async def test_stranded_reclaim_ladder_vae_pause_self_heals_on_healthy_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reclaim-ladder VAE pause with no live claimant is restored once the card has been HEALTHY."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    assert manager._disaggregation_roles_live() is True

    # Strand a reclaim-ladder VAE pause and complete its teardown (as the live pause does over control ticks):
    # the lane process is gone and, while the pause is held, the per-tick restart hook is suppressed, so the
    # lane never returns on its own.
    assert manager._process_lifecycle.pause_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    del manager._process_map[3]
    assert manager._disaggregation_roles_live() is False
    assert manager._process_lifecycle.start_vae_lane_processes() is False, (
        "the per-tick restart hook must be wedged by the pause: this is why a stranded lane never returns"
    )

    # Model the lane's respawn (which the harness does not really run) so the liveness check reads restored
    # disaggregation eligibility, not merely a cleared flag.
    def _restart_vae() -> bool:
        manager._process_map[3] = _live_process(3, HordeProcessType.VAE_LANE, reserved_mb=1_330)
        return True

    manager._process_lifecycle.start_vae_lane_processes = _restart_vae  # type: ignore[method-assign]

    # No borrow receipt claims it, no saturation episode owns it, and the card has been HEALTHY well past the
    # backstop debounce: the orphaned pause must be reclaimed.
    assert manager._post_process_orchestrator.is_service_lane_borrowed(ActuatorCommandKind.PAUSE_VAE_LANE) is False
    manager._healthy_since_by_device[0] = time.monotonic() - (manager._STRANDED_LANE_RESTORE_DEBOUNCE_SECONDS + 5.0)

    manager._reclaim_stranded_service_lane_pauses(0)

    assert manager._process_lifecycle.is_vae_lane_gpu_paused is False
    assert manager._process_lifecycle.vae_lane_pause_owner is None
    assert manager._disaggregation_roles_live() is True, "disaggregation routing must be available again"


async def test_backstop_leaves_whole_card_and_borrowed_pauses_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """The self-heal backstop must never lift a whole-card pause or a pause a PP borrow still claims."""
    # A whole-card residency pause has its own restore path and must be left alone.
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    assert manager._process_lifecycle.pause_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    manager._healthy_since_by_device[0] = time.monotonic() - (manager._STRANDED_LANE_RESTORE_DEBOUNCE_SECONDS + 5.0)
    manager._reclaim_stranded_service_lane_pauses(0)
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True
    assert manager._process_lifecycle.vae_lane_pause_owner is PauseOwner.WHOLE_CARD

    # A pause the PP drain still holds a receipt for has a live claimant and must not be reclaimed.
    manager2, _v2, _c2, _s2 = _live_shaped_manager(monkeypatch)
    manager2._post_process_orchestrator._borrowed_service_lane_actuations.add(ActuatorCommandKind.PAUSE_VAE_LANE)
    assert manager2._process_lifecycle.pause_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    manager2._healthy_since_by_device[0] = time.monotonic() - (manager2._STRANDED_LANE_RESTORE_DEBOUNCE_SECONDS + 5.0)
    manager2._reclaim_stranded_service_lane_pauses(0)
    assert manager2._process_lifecycle.is_vae_lane_gpu_paused is True
    assert manager2._process_lifecycle.vae_lane_pause_owner is PauseOwner.RECLAIM_LADDER


async def test_backstop_waits_for_healthy_debounce(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backstop does not lift a pause on a card that has not been HEALTHY long enough."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    assert manager._process_lifecycle.pause_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True

    # Card only just returned HEALTHY: below the debounce, so the backstop holds off.
    manager._healthy_since_by_device[0] = time.monotonic()
    manager._reclaim_stranded_service_lane_pauses(0)
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True

    # Card never reported HEALTHY at all: no restore either.
    manager._healthy_since_by_device.pop(0, None)
    manager._reclaim_stranded_service_lane_pauses(0)
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True


def test_disaggregation_silence_breaker_edge_latches_and_rearms(monkeypatch: pytest.MonkeyPatch) -> None:
    """One warning fires after sustained unavailability and re-arms when routing returns."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    assert manager._disaggregation_roles_live() is True

    # Roles live: nothing to warn about, latch stays clear.
    manager._note_disaggregation_availability()
    assert manager._disaggregation_unavailable_since is None
    assert manager._disaggregation_unavailable_logged is False

    # The VAE lane goes away: the first observation arms the clock but does not warn yet.
    del manager._process_map[3]
    manager._note_disaggregation_availability()
    assert manager._disaggregation_unavailable_since is not None
    assert manager._disaggregation_unavailable_logged is False

    # Push the outage past the warn window: exactly one WARNING naming the down role.
    manager._disaggregation_unavailable_since -= manager._DISAGGREGATION_UNAVAILABLE_WARN_SECONDS + 1.0
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="WARNING")
    try:
        manager._note_disaggregation_availability()
    finally:
        logger.remove(sink_id)
    assert manager._disaggregation_unavailable_logged is True
    assert len(lines) == 1
    assert "VAE lane" in lines[0]
    assert "monolithic" in lines[0]

    # A further tick while still down does not re-log (edge-latched).
    more: list[str] = []
    sink_id = logger.add(lambda message: more.append(message.record["message"]), level="WARNING")
    try:
        manager._note_disaggregation_availability()
    finally:
        logger.remove(sink_id)
    assert more == []

    # Availability returns: the latch re-arms for the next outage.
    manager._process_map[3] = _live_process(3, HordeProcessType.VAE_LANE, reserved_mb=1_330)
    manager._note_disaggregation_availability()
    assert manager._disaggregation_unavailable_since is None
    assert manager._disaggregation_unavailable_logged is False


def _capture_info(action: object) -> list[str]:
    """Run ``action`` (a zero-arg callable) with an INFO loguru sink and return the captured messages."""
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
    try:
        action()  # type: ignore[operator]
    finally:
        logger.remove(sink_id)
    return lines


def test_reclaim_ladder_lane_pause_logs_name_the_reclaim_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reclaim-ladder pause/restore names the reclaim ladder, not a whole-card residency it is not."""
    lifecycle = _live_shaped_manager(monkeypatch)[0]._process_lifecycle

    pause_lines = _capture_info(lambda: lifecycle.pause_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER))
    assert any("Reclaim ladder" in line and "VAE lane" in line for line in pause_lines)
    assert not any("Whole-card residency" in line and "VAE lane" in line for line in pause_lines)

    restore_lines = _capture_info(lambda: lifecycle.restore_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER))
    assert any("Reclaim ladder" in line and "VAE lane" in line for line in restore_lines)


def test_whole_card_lane_pause_logs_name_the_whole_card_residency(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whole-card pause keeps naming the whole-card residency, so the two owners stay distinguishable."""
    lifecycle = _live_shaped_manager(monkeypatch)[0]._process_lifecycle

    pause_lines = _capture_info(lambda: lifecycle.pause_component_off_gpu(owner=PauseOwner.WHOLE_CARD))
    assert any("Whole-card residency" in line and "component lane" in line for line in pause_lines)
    assert not any("Reclaim ladder" in line and "component lane" in line for line in pause_lines)
