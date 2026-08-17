"""Reproduces a reclaim-ladder VAE-lane pause discarding a disaggregated job's finished sampling.

Under VRAM pressure the reclaim ladder (a post-processing borrow, or the governor's saturation rung) may pause
the VAE/image lane off the GPU to free a CUDA context. The arbiter guarantees the lane process is idle at the
instant of the pause, but it cannot see the disaggregation orchestrator's queued decode work: a job that has
already finished sampling and sits at ``AWAITING_LATENT_DECODE`` needs that same lane for a ~1-2s decode. Pausing
the lane out from under it reroutes the whole job monolithic, discarding the completed sampling to free room for
a dispatch the decode itself would have cleared within seconds.

The whole-card residency reaches the same lane by its own path. Its establishment pass paused the lane
unconditionally, on the belief that a residency suppresses disaggregation dispatch and therefore leaves the lane
idle; a decode dispatched moments before the residency is still in flight, so the same finished sample was
discarded there.

The specification here:

* a VAE-lane pause is not executed while a disaggregated decode is queued or in flight on the lane, whether the
  reclaim ladder or a whole-card residency asks for it, so the caller moves to its next option instead of
  stranding the finished sample;
* a job merely sampling (not yet at the decode stage) does not block the *reclaim ladder's* pause, since
  rerouting it discards no finished work and relieving device pressure matters more;
* it does block a *whole-card residency's* pause, which buys nothing by taking the lane before the in-flight
  job drains and pays for it with that job's finished sample when the short whole-card decode hold expires;
* with nothing pending, the pause proceeds exactly as before; and
* a whole-card residency that deferred the pause retries it every convergence cycle, so the lane still leaves
  the card once the pipeline drains.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.workers.disaggregation_orchestrator import (
    DisaggJobStage,
    _DisaggJobState,
)
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.regressions.test_post_process_drain_context_reclaim_repro import _live_shaped_manager
from tests.process_management.regressions.test_whole_card_deadlock_fixes import _FLUX_MODEL
from tests.process_management.regressions.test_whole_card_lifecycle_matrix import _forecast_for_target
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler
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


def test_jobs_bound_for_vae_decode_count_also_counts_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wider accessor counts every job that will still need the VAE lane, decoding or still sampling."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    orchestrator = manager._disaggregation_orchestrator

    assert orchestrator.jobs_bound_for_vae_decode_count() == 0

    _insert_disagg_job(manager, stage=DisaggJobStage.AWAITING_LATENT_DECODE)
    _insert_disagg_job(manager, stage=DisaggJobStage.SAMPLING)
    _insert_disagg_job(manager, stage=DisaggJobStage.SAMPLING)
    _insert_disagg_job(manager, stage=DisaggJobStage.AWAITING_CONDITIONING)
    _insert_disagg_job(manager, stage=DisaggJobStage.AWAITING_SOURCE_LATENT)

    assert orchestrator.jobs_bound_for_vae_decode_count() == 3, (
        "a sampling job's only remaining step after its latent lands is the VAE decode, so it is bound for "
        "the lane; jobs still encoding are not, since they may yet reroute without wasting a sample"
    )
    assert orchestrator.pending_vae_decode_count() == 1, (
        "the narrower count must stay at the decode stage alone: the reclaim ladder's policy is unchanged"
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


def _residency_scheduler(
    pending_decodes: list[int],
    bound_for_lane: list[int] | None = None,
) -> InferenceScheduler:
    """A scheduler whose only reachable residency lever is the VAE lane, reading the two disagg counts.

    ``pending_decodes[0]`` is the reclaim ladder's narrow count (jobs at the decode stage) and
    ``bound_for_lane[0]`` the residency's wider one (sampling as well as decoding); it defaults to the same
    list, the case where every job needing the lane is already at its decode.

    The sibling levers (safety, the post-processing lane, the component lane) are mocked out so an assertion
    about the VAE lane observes that lane alone, and the lane's lifecycle pause is a mock so the test reads
    whether the residency asked for it rather than how the lifecycle carried it out.
    """
    scheduler = _make_inference_scheduler()
    lifecycle = scheduler._process_lifecycle
    lifecycle.vae_lane_enabled = Mock(return_value=True)
    lifecycle.pause_vae_lane_off_gpu = Mock(return_value=True)
    lifecycle.is_safety_gpu_paused = False
    lifecycle.scale_inference_processes = Mock(return_value=0)
    scheduler._pause_post_process_for_residency_if_idle = Mock(return_value=False)
    scheduler._residency_should_pause_safety = Mock(return_value=False)
    scheduler._residency_should_pause_post_process = Mock(return_value=False)
    scheduler._residency_should_pause_component_lane = Mock(return_value=False)
    # Convergence spares the residency's staged head; the pause under test does not depend on which slot holds
    # it, so the holder is asserted present rather than built out of a process map.
    scheduler._whole_card_residency_has_holder = Mock(return_value=True)
    scheduler._vae_decode_pending_count = lambda: pending_decodes[0]
    bound = bound_for_lane if bound_for_lane is not None else pending_decodes
    scheduler._vae_lane_bound_job_count = lambda: bound[0]
    return scheduler


def test_establishing_a_residency_defers_the_vae_lane_pause_while_a_decode_is_pending() -> None:
    """A decode in flight when the card is claimed must not be stopped out from under: the pause waits."""
    pending = [1]
    scheduler = _residency_scheduler(pending)

    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
    try:
        scheduler._establish_whole_card_residency(
            make_job_pop_response(_FLUX_MODEL, width=1216, height=1216),
            _forecast_for_target(1),
            announce=False,
        )
    finally:
        logger.remove(sink_id)

    scheduler._process_lifecycle.pause_vae_lane_off_gpu.assert_not_called()
    assert any("VAE-lane pause" in line and "bound for the VAE lane" in line for line in lines), (
        "the deferred whole-card VAE-lane pause must emit one INFO line naming what it is waiting on"
    )

    # The decode lands and the job leaves the pipeline: the next convergence cycle takes the lane off the card,
    # so deferring the pause postpones the residency's commitment rather than abandoning it.
    pending[0] = 0
    scheduler._converge_whole_card_residency()
    scheduler._process_lifecycle.pause_vae_lane_off_gpu.assert_called_once_with(owner=PauseOwner.WHOLE_CARD)


def test_establishing_a_residency_pauses_the_vae_lane_with_no_decode_pending() -> None:
    """With nothing to strand, claiming the card stops the VAE lane at establishment exactly as before."""
    scheduler = _residency_scheduler([0])

    scheduler._establish_whole_card_residency(
        make_job_pop_response(_FLUX_MODEL, width=1216, height=1216),
        _forecast_for_target(1),
        announce=False,
    )

    scheduler._process_lifecycle.pause_vae_lane_off_gpu.assert_called_once_with(owner=PauseOwner.WHOLE_CARD)


def test_residency_convergence_does_not_re_pause_a_lane_it_already_holds() -> None:
    """The per-cycle retry is idempotent: a lane already stopped under this owner is not paused again."""
    scheduler = _residency_scheduler([0])
    scheduler._establish_whole_card_residency(
        make_job_pop_response(_FLUX_MODEL, width=1216, height=1216),
        _forecast_for_target(1),
        announce=False,
    )
    scheduler._process_lifecycle.pause_vae_lane_off_gpu.reset_mock()
    scheduler._process_lifecycle.vae_lane_pause_owner = PauseOwner.WHOLE_CARD

    scheduler._converge_whole_card_residency()
    scheduler._converge_whole_card_residency()

    scheduler._process_lifecycle.pause_vae_lane_off_gpu.assert_not_called()


def test_establishing_a_residency_defers_the_vae_lane_pause_while_a_job_is_sampling() -> None:
    """A job still sampling withholds the residency's pause: the residency cannot use the card before it drains.

    Nothing is queued on the lane yet, so the reclaim ladder's narrow count is zero. The residency reads the
    wider one because taking the lane now gains it a context it cannot spend until the in-flight job finishes,
    while the job's sample is discarded the moment the short whole-card decode hold expires under the pause.
    """
    pending = [0]
    bound = [1]
    scheduler = _residency_scheduler(pending, bound)

    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
    try:
        scheduler._establish_whole_card_residency(
            make_job_pop_response(_FLUX_MODEL, width=1216, height=1216),
            _forecast_for_target(1),
            announce=False,
        )
    finally:
        logger.remove(sink_id)

    scheduler._process_lifecycle.pause_vae_lane_off_gpu.assert_not_called()
    assert any("VAE-lane pause" in line and "bound for the VAE lane" in line for line in lines), (
        "the deferred whole-card VAE-lane pause must emit one INFO line naming the jobs it waits on"
    )

    # The job samples, decodes and leaves the pipeline: the next convergence cycle takes the lane off the card,
    # so waiting on a sampling job postpones the residency's commitment rather than abandoning it.
    bound[0] = 0
    scheduler._converge_whole_card_residency()
    scheduler._process_lifecycle.pause_vae_lane_off_gpu.assert_called_once_with(owner=PauseOwner.WHOLE_CARD)


def test_reclaim_actuator_still_pauses_the_vae_lane_with_only_a_sampling_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control on the split: the reclaim ladder's policy is unchanged, so a sampling job does not stop its pause.

    The ladder pauses to relieve device pressure now. A sampling job has no finished latent to strand, and the
    relief is worth more than an unfinished sample, so only a queued or in-flight decode withholds this pause.
    """
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    scheduler = manager._inference_scheduler
    _insert_disagg_job(manager, stage=DisaggJobStage.SAMPLING)

    assert scheduler.pause_vae_lane(None) is True
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True
