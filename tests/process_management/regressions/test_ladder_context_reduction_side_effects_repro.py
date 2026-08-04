"""Reducing live inference contexts through the reclaim ladder is an actuation, not a residency grant.

The verified reclaim ladder's ``REDUCE_LIVE_CONTEXTS`` rung exists to return retained VRAM to the card by
removing idle inference contexts. Reaching that outcome by routing through
:meth:`InferenceScheduler._establish_whole_card_residency` additionally books a whole-card residency: it
stamps a cooldown, marks the head exclusive worker-wide, moves safety off the GPU, pauses the service lanes,
and opens the establish grace window that tells the recovery supervisor a held queue is intentional. Those are
policy commitments of a whole-card residency, and routing the rung through the grant takes them even when
``whole_card_exclusive_residency`` is disabled, i.e. for an operator who has declined that policy entirely.

The invariant: context reduction removes contexts and nothing else. A residency remains a separate policy that
may *request* a reduction through the ladder, so its side effects belong to the grant that asked for it, never
to the rung.

These pin the absence of the residency side effects, not the absence of the teardown: the reduction itself is
asserted to have happened in each case, and the emergency starvation teardown this rung serves with the flag
off is covered by ``test_flag_off_starvation_teardown_recovery_repro``.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.vram_arbiter import ActuatorCommand, ActuatorCommandKind
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler, _PreloadActuation
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MAX_INFERENCE = 2
_HEAD_MODEL = "AlbedoBase XL (SDXL)"
_WHOLE_CARD_MODEL = "Flux.1-Schnell fp8 (Compact)"  # EXTRA_LARGE by name
_COOLDOWN_SECONDS = 45


class _ScaleRecorder:
    """A ``scale_inference_processes`` stand-in that retires idle contexts from the map toward the target.

    The real lifecycle removes a scaled-down process from the map synchronously, so the live count drops the
    instant the teardown runs; mirroring that keeps the lane-count assertions observing real state.
    """

    def __init__(self, process_map: ProcessMap) -> None:
        self._process_map = process_map

    def __call__(self, target_count: int, *, device_index: int | None = None, **_kwargs: object) -> int:
        loaded = self._process_map.num_loaded_inference_processes()
        while loaded > target_count:
            victim = next(
                pid
                for pid, info in self._process_map.items()
                if info.last_process_state == HordeProcessState.WAITING_FOR_JOB
            )
            del self._process_map[victim]
            loaded -= 1
        return self._process_map.num_loaded_inference_processes()


async def _flag_off_scheduler_with_tracked_head() -> tuple[
    InferenceScheduler, ProcessMap, Mock, ImageGenerateJobPopResponse
]:
    """A residency-disabled scheduler with two idle lanes, a tracked head, and a ladder command ready to fire.

    ``safety_on_gpu`` is granted so the residency's safety-off lever is reachable: a test that asserts safety is
    left alone must be able to observe the pause it forbids.
    """
    process_map = ProcessMap(
        {
            0: make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
            1: make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
        },
    )
    job_tracker = JobTracker()
    job = make_job_pop_response(_HEAD_MODEL)
    await track_popped_job_async(job_tracker, job)
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(
            enable_vram_budget=True,
            whole_card_exclusive_residency=False,
            safety_on_gpu=True,
            whole_card_residency_safety_off_gpu=True,
            whole_card_residency_cooldown_seconds=_COOLDOWN_SECONDS,
        ),
        max_inference=_MAX_INFERENCE,
    )
    lifecycle = Mock()
    lifecycle.scale_inference_processes = _ScaleRecorder(process_map)
    lifecycle.is_safety_gpu_paused = False
    lifecycle.pause_safety_on_gpu = Mock(return_value=True)
    lifecycle.post_process_lane_enabled = Mock(return_value=False)
    lifecycle.component_lane_enabled = Mock(return_value=False)
    lifecycle.vae_lane_enabled = Mock(return_value=False)
    scheduler._process_lifecycle = lifecycle
    scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]

    forecast = Mock()
    forecast.max_resident_processes = Mock(return_value=1)
    forecast.total_vram_mb = 24000.0
    forecast.fits_weights_now = True
    scheduler._preload_actuation = _PreloadActuation(
        job=job,
        available_process=process_map[0],
        forecast=forecast,
        max_resident=1,
    )
    return scheduler, process_map, lifecycle, job


def _issue_reduce_contexts(scheduler: InferenceScheduler) -> None:
    """Deliver one arbiter REDUCE_LIVE_CONTEXTS command through the preload actuation surface."""
    commands = (ActuatorCommand(kind=ActuatorCommandKind.REDUCE_LIVE_CONTEXTS, device_index=None),)
    scheduler._execute_preload_actuations(commands, device_index=None, for_head_of_queue=True)


class TestLadderReductionLeavesSafetyAndExclusivityAlone:
    """The rung reclaims contexts without pausing safety or reserving the worker for its head."""

    async def test_reduction_does_not_pause_safety(self) -> None:
        """A context-reduction command retires an idle context and leaves safety on its card."""
        scheduler, process_map, lifecycle, _job = await _flag_off_scheduler_with_tracked_head()

        _issue_reduce_contexts(scheduler)

        assert process_map.num_loaded_inference_processes() == 1, (
            "precondition: the rung did its job and retired the idle context"
        )
        assert lifecycle.pause_safety_on_gpu.call_count == 0, (
            "moving safety off the GPU is a whole-card residency commitment; reclaiming an idle inference "
            "context must not take it"
        )

    async def test_reduction_does_not_mark_the_head_exclusive_worker_wide(self) -> None:
        """A context-reduction command retires an idle context without reserving the worker for its head."""
        scheduler, process_map, _lifecycle, job = await _flag_off_scheduler_with_tracked_head()

        _issue_reduce_contexts(scheduler)

        assert process_map.num_loaded_inference_processes() == 1, (
            "precondition: the rung did its job and retired the idle context"
        )
        assert scheduler._job_tracker.is_admitted_exclusive(job) is False, (
            "reserving the worker for one job is a whole-card residency commitment; reclaiming an idle "
            "inference context must not take it"
        )
        assert scheduler._job_tracker.has_exclusive_job_in_progress() is False


class TestLadderReductionLeavesTheServiceLanesRunning:
    """Stopping the disaggregation lanes is a residency commitment, not part of reclaiming a spare context."""

    async def test_reduction_does_not_pause_the_vae_or_component_lanes(self) -> None:
        """A context-reduction command retires an idle context and leaves both disaggregation lanes on the card."""
        scheduler, process_map, lifecycle, _job = await _flag_off_scheduler_with_tracked_head()
        # Both lanes enabled and on this card, so the residency's lane-pause levers are reachable: a test that
        # asserts the lanes are left alone must be able to observe the pauses it forbids.
        lifecycle.vae_lane_enabled = Mock(return_value=True)
        lifecycle.component_lane_enabled = Mock(return_value=True)
        lifecycle.pause_vae_lane_off_gpu = Mock(return_value=True)
        lifecycle.pause_component_off_gpu = Mock(return_value=True)

        _issue_reduce_contexts(scheduler)

        assert process_map.num_loaded_inference_processes() == 1, (
            "precondition: the rung did its job and retired the idle context"
        )
        assert lifecycle.pause_vae_lane_off_gpu.call_count == 0, (
            "stopping the VAE lane is a whole-card residency commitment; reclaiming an idle inference context "
            "must not take it"
        )
        assert lifecycle.pause_component_off_gpu.call_count == 0, (
            "stopping the component lane is a whole-card residency commitment; reclaiming an idle inference "
            "context must not take it"
        )


class TestGenuineResidencyEstablishStillClaimsTheCard:
    """Separating the rung from the policy must not weaken the policy: a real grant still claims the card."""

    async def test_establish_reduces_to_target_and_moves_safety_off_gpu(self) -> None:
        """A whole-card head's own residency grant stops the idle siblings and cycles safety off the card."""
        scheduler, process_map, lifecycle, _job = await _flag_off_scheduler_with_tracked_head()
        scheduler._runtime_config.bridge_data.whole_card_exclusive_residency = True
        heavy_job = make_job_pop_response(_WHOLE_CARD_MODEL)
        await track_popped_job_async(scheduler._job_tracker, heavy_job)
        forecast = Mock()
        forecast.max_resident_processes = Mock(return_value=1)
        forecast.total_vram_mb = 16375.0

        scheduler._establish_whole_card_residency(heavy_job, forecast, announce=True)

        assert process_map.num_loaded_inference_processes() == 1
        lifecycle.pause_safety_on_gpu.assert_called_once_with()
        assert scheduler._residency_state(None).model == _WHOLE_CARD_MODEL


class TestManagerWiresTheLadderIntoTheScheduler:
    """The manager hands its ladder engine to the scheduler, so reductions book real restore obligations.

    The scheduler's reduction actuator records its restore obligation on whatever ladder it holds; test
    fixtures inject one explicitly, so only this identity pin guards the production wiring. Left unwired,
    a reduction would shrink the pool and nothing would ever grow it back.
    """

    def test_scheduler_holds_the_managers_ladder_engine(self) -> None:
        """The scheduler's ladder reference is the manager's single ladder engine, not a copy or None."""
        pm = make_testable_process_manager()
        assert pm._inference_scheduler._reclaim_ladder is pm._reclaim_ladder


class TestLadderReductionOpensNoResidencyCooldown:
    """The rung leaves no residency lease behind for unrelated heads to queue behind."""

    async def test_reduction_leaves_no_active_residency_lease(self) -> None:
        """After the reduction no card holds a residency, so unrelated heads are not gated behind one."""
        scheduler, process_map, lifecycle, _job = await _flag_off_scheduler_with_tracked_head()

        _issue_reduce_contexts(scheduler)

        assert process_map.num_loaded_inference_processes() == 1, (
            "precondition: the rung did its job and retired the idle context"
        )
        # is_whole_card_residency_active is what the pop-side large-model re-entry cooldown and the
        # disaggregation dispatch predicate consult before admitting an unrelated job.
        assert scheduler.is_whole_card_residency_active() is False, (
            "a context-reduction rung must not leave a residency lease that gates unrelated heads"
        )
        assert scheduler._residency_state(None).cooldown_until <= time.time(), (
            "a context-reduction rung must not open a residency cooldown"
        )
