"""Reproduces a reclaim-ladder VAE-lane pause discarding a disaggregated job's finished sampling.

Under VRAM pressure the reclaim ladder (a post-processing borrow, or the governor's saturation rung) may pause
the VAE/image lane off the GPU to free a CUDA context. The arbiter guarantees the lane process is idle at the
instant of the pause, but it cannot see the disaggregation orchestrator's queued decode work: a job that has
already finished sampling and sits at ``AWAITING_LATENT_DECODE`` needs that same lane for a ~1-2s decode. Pausing
the lane out from under it reroutes the whole job monolithic, discarding the completed sampling to free room for
a dispatch the decode itself would have cleared within seconds.

The specification here:

* a VAE-lane pause is not executed while a disaggregated decode is queued or in flight on the lane, so the
  reclaim path reports no-op and moves to its next relief option instead of stranding the finished sample;
* a job merely sampling (not yet at the decode stage) does not block the pause, since rerouting it discards no
  finished work and relieving pressure matters more; and
* with no decode pending, the pause proceeds exactly as before.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from loguru import logger

from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.workers.disaggregation_orchestrator import (
    DisaggJobStage,
    _DisaggJobState,
)
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.regressions.test_post_process_drain_context_reclaim_repro import _live_shaped_manager
from tests.process_management.workers import test_post_process_orchestration as pp_tests


def _insert_disagg_job(manager: HordeWorkerProcessManager, *, stage: DisaggJobStage) -> str:
    """Insert one held disaggregated job at ``stage`` into the manager's orchestrator; return its key.

    Builds the held state directly (the same duck-typed job_info the orchestrator's own unit tests use, of which
    only ``sdk_api_job_info`` is read on this path) so the test pins a precise pipeline stage without driving the
    full encode/sample/decode DAG. A decode-stage job carries a sampled latent, matching the real state in which
    the VAE lane is the only thing standing between a finished sample and its images.
    """
    job_info = SimpleNamespace(sdk_api_job_info=make_job_pop_response(model="SDXL 1.0"))
    key = str(job_info.sdk_api_job_info.id_)
    manager._disaggregation_orchestrator._jobs[key] = _DisaggJobState(
        job_info=job_info,  # type: ignore[arg-type]
        stage=stage,
        needs_source_latent=False,
        source_latent_bytes=b"sampled-latent" if stage == DisaggJobStage.AWAITING_LATENT_DECODE else None,
    )
    return key


async def _drive_pp_borrow(manager: HordeWorkerProcessManager) -> None:
    """Run the two-cycle post-processing borrow flow that reaches the reclaim-ladder VAE-lane pause.

    The first cycle spends the softer model/cache reclaim; the second, still non-fitting, reaches the service-lane
    borrow that describes and executes a ``PAUSE_VAE_LANE`` command through the single reclaim owner.
    """
    job_info = pp_tests._make_pp_job_info(["CodeFormers", "RealESRGAN_x4plus"])
    await manager._job_tracker.queue_for_post_processing(job_info)
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()


def test_pending_vae_decode_count_counts_only_the_decode_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """The accessor counts jobs needing the VAE lane for a decode, and excludes every other pipeline stage."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    orchestrator = manager._disaggregation_orchestrator

    assert orchestrator.pending_vae_decode_count() == 0

    _insert_disagg_job(manager, stage=DisaggJobStage.AWAITING_LATENT_DECODE)
    _insert_disagg_job(manager, stage=DisaggJobStage.AWAITING_LATENT_DECODE)
    _insert_disagg_job(manager, stage=DisaggJobStage.SAMPLING)
    _insert_disagg_job(manager, stage=DisaggJobStage.AWAITING_CONDITIONING)
    _insert_disagg_job(manager, stage=DisaggJobStage.AWAITING_SOURCE_LATENT)

    assert orchestrator.pending_vae_decode_count() == 2, (
        "only jobs at AWAITING_LATENT_DECODE need the VAE lane for a decode; sampling and encode stages must "
        "not be counted, or the pause would be withheld for work the lane pause does not strand"
    )


async def test_pp_borrow_does_not_pause_vae_lane_while_a_decode_is_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    """A queued disaggregated decode makes the reclaim-ladder VAE-lane pause a no-op, sparing the finished sample."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    decode_key = _insert_disagg_job(manager, stage=DisaggJobStage.AWAITING_LATENT_DECODE)

    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
    try:
        await _drive_pp_borrow(manager)
    finally:
        logger.remove(sink_id)

    assert manager._process_lifecycle.is_vae_lane_gpu_paused is False, (
        "the reclaim ladder paused the VAE lane while a disaggregated decode was queued on it; the finished "
        "sample will be discarded when the job reroutes monolithic to free room a ~1-2s decode would have"
    )
    assert manager._process_lifecycle.is_component_gpu_paused is False

    # Behavioral end-state: the decode-pending job is neither rerouted nor stranded. A subsequent orchestrator
    # tick sees the lane un-paused, so it does not take the "role lane deliberately paused" reroute that would
    # throw away the completed sampling; the job stays held at its decode stage.
    manager._disaggregation_orchestrator.tick()
    assert decode_key in manager._disaggregation_orchestrator._jobs, (
        "the decode-pending job was rerouted monolithic, discarding its finished sampling"
    )
    assert manager._disaggregation_orchestrator._jobs[decode_key].stage == DisaggJobStage.AWAITING_LATENT_DECODE

    # One edge-latched INFO line names the pending-decode count so live forensics can see the lever working.
    assert any("VAE-lane pause" in line and "disaggregated decode" in line for line in lines), (
        "the skipped VAE-lane pause must emit one INFO line naming the pending-decode count"
    )


async def test_pp_borrow_still_pauses_vae_lane_with_no_pending_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no decode queued, the reclaim-ladder VAE-lane borrow proceeds exactly as before the eligibility gate."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)

    await _drive_pp_borrow(manager)

    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True, (
        "with no disaggregated decode pending the VAE-lane borrow must still pause the idle lane; the decode-"
        "drain gate must not suppress the normal reclaim path"
    )


def test_reclaim_actuator_pause_vae_lane_is_the_shared_decode_drain_choke_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both reclaim paths funnel through this one actuator, so its decode-drain gate covers the governor rung too.

    The governor's saturation rung and the post-processing borrow both execute a ``PAUSE_VAE_LANE`` through this
    single ``pause_vae_lane`` actuator, so a no-op here is what makes each skip the pause. Driving the actuator
    directly proves the choke point without standing up a full saturation episode.
    """
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    scheduler = manager._inference_scheduler

    decode_key = _insert_disagg_job(manager, stage=DisaggJobStage.AWAITING_LATENT_DECODE)
    assert scheduler.pause_vae_lane(None) is False
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is False

    # The decode drains (its result popped the job); the very next pause proceeds, so the gate withholds nothing
    # once no decode is pending.
    del manager._disaggregation_orchestrator._jobs[decode_key]
    assert scheduler.pause_vae_lane(None) is True
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True


async def test_pp_borrow_pauses_vae_lane_despite_a_job_merely_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job still sampling does not block the pause: rerouting it discards no finished work, and relief wins."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    _insert_disagg_job(manager, stage=DisaggJobStage.SAMPLING)

    await _drive_pp_borrow(manager)

    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True, (
        "a job merely sampling (no finished sample to strand) must not withhold the VAE-lane pause; only a "
        "queued or in-flight decode does"
    )
