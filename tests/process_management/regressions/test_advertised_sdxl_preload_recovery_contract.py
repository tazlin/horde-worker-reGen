"""Contracts for advertised SDXL work that stalls after preload.

An image job inside the advertised ``max_power`` envelope can carry a predicted VRAM demand above the
arbiter's achievable ceiling. The same job can nevertheless be executable through the backend's ordinary
memory management. A static prediction must therefore not create a dispatch hold with no actuator and no
path to a real load; either intake must exclude the work before pop or scheduling must try it within a bound.

A separate ownership contract applies when recovery acts on that hold. Preload records which job requested
the model before inference starts. That scheduling reference is not evidence that the process executed the
job. Rebuilding an idle ``PRELOADED_MODEL`` lane must preserve the pending job without consuming its bounded
inference-attempt budget.

These tests are intentionally RED and cover both contracts at their shared state boundary.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from tests.process_management.conftest import make_job_pop_response, make_mock_process_info, track_popped_job_async
from tests.process_management.lifecycle.test_process_lifecycle import _make_plm
from tests.process_management.liveness.test_bounded_dispatch_matrix import _CARD_8GB, _SDXL, _DispatchWorld

_NOISE_MB = 512.0
_CANDIDATE_MB = 8774.0
_WIDTH = 1216
_HEIGHT = 1216
_STEPS = 25
_MAX_POWER = 48
_TICK_SECONDS = 10.0
_DISPATCH_BOUND_TICKS = 15


def _boundary_job() -> ImageGenerateJobPopResponse:
    """Build an SDXL-family payload near the configured 48-power intake boundary."""
    return make_job_pop_response(
        _SDXL.name,
        width=_WIDTH,
        height=_HEIGHT,
        ddim_steps=_STEPS,
    )


class TestAdvertisedSdxlHeadMustReachARealAttempt:
    """An accepted max-power payload must not park forever after its model reaches ``LOADED_IN_RAM``."""

    async def test_ram_staged_head_reaches_dispatch_within_the_recovery_horizon(self) -> None:
        """A healthy single lane gets a real load attempt instead of a no-actuator residency hold."""
        # The pop API is offered max_pixels=max_power*8*64*64, so this payload is inside the advertised
        # envelope and must not encounter a contradictory, permanently-closed dispatch gate downstream.
        advertised_max_pixels = _MAX_POWER * 8 * 64 * 64
        assert advertised_max_pixels >= _WIDTH * _HEIGHT

        world = _DispatchWorld(
            card=_CARD_8GB,
            lane_count=1,
            max_threads=1,
            queue_depth=0,
            tick_seconds=_TICK_SECONDS,
        )
        # Preload completed into RAM on the only idle lane. Dispatch is the remaining transition under test.
        world.seed_resident(0, _SDXL, in_vram=False)
        job = _boundary_job()

        baseline = world.scheduler._model_metadata.get_baseline(_SDXL.name)
        candidate_mb = world.scheduler._measured_admission_candidate_delta_mb(
            job,
            baseline,
            process_id=None,
            disaggregated=False,
        )
        snapshot = world.scheduler.build_vram_arbiter_snapshot(device_free_mb_by_device={0: world.device_free_mb()})
        state = snapshot.device(None)
        assert state is not None
        assert state.achievable_ceiling_mb() is not None
        assert candidate_mb == _CANDIDATE_MB
        assert state.achievable_ceiling_mb() == _CARD_8GB.total_mb - _NOISE_MB

        await world.pop(job)
        for _ in range(_DISPATCH_BOUND_TICKS):
            await world.step()

        assert world.dispatch_tick(job) is not None, (
            "the advertised SDXL head never received START_INFERENCE after its model reached RAM; "
            f"the static ceiling verdict parked it with no effective actuator: {world.state_dump()}"
        )


async def _prepare_preloaded_pending_job() -> tuple[
    ProcessLifecycleManager,
    ImageGenerateJobPopResponse,
    HordeProcessInfo,
]:
    """Return a lifecycle manager and a never-started job referenced by an idle preloaded lane."""
    process_lifecycle = _make_plm()
    process_lifecycle._end_inference_process = Mock()  # type: ignore[method-assign]
    process_lifecycle._start_inference_process = Mock()  # type: ignore[method-assign]

    process_lifecycle._job_tracker.set_retry_policy(2)
    job = _boundary_job()
    await track_popped_job_async(process_lifecycle._job_tracker, job)
    lane = make_mock_process_info(2, model_name=_SDXL.name, state=HordeProcessState.PRELOADED_MODEL)
    lane.last_job_referenced = job
    process_lifecycle._process_map[2] = lane
    return process_lifecycle, job, lane


class TestSoftResetMustNotInventAnInferenceFailure:
    """A preload reference is scheduling intent, not proof that the child ran or failed the job."""

    async def test_first_soft_reset_does_not_spend_an_attempt_before_start_inference(self) -> None:
        """Cycling ``PRELOADED_MODEL`` leaves the still-pending job's execution budget untouched."""
        process_lifecycle, job, lane = await _prepare_preloaded_pending_job()

        process_lifecycle._replace_inference_process(
            lane,
            intentional_reason="soft reset: preload hold #1",
            recovery_requeue=True,
        )

        assert job.id_ is not None
        tracked = process_lifecycle._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.stage is JobStage.PENDING_INFERENCE
        assert tracked.inference_attempts == 0, (
            "a deliberate reset charged an inference failure to a job that was only preloaded and never ran"
        )

    async def test_two_soft_resets_cannot_terminally_fault_a_job_that_never_started(self) -> None:
        """Repeated recovery cycles preserve the job instead of exhausting both attempts."""
        process_lifecycle, job, first_lane = await _prepare_preloaded_pending_job()

        process_lifecycle._replace_inference_process(
            first_lane,
            intentional_reason="soft reset: preload hold #1",
            recovery_requeue=True,
        )
        second_lane = make_mock_process_info(2, model_name=_SDXL.name, state=HordeProcessState.PRELOADED_MODEL)
        second_lane.last_job_referenced = job
        process_lifecycle._process_map[2] = second_lane
        process_lifecycle._replace_inference_process(
            second_lane,
            intentional_reason="soft reset: preload hold #2",
            recovery_requeue=True,
        )

        assert job.id_ is not None
        tracked = process_lifecycle._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.stage is JobStage.PENDING_INFERENCE, (
            "two administrative rebuilds terminally faulted a job that never entered inference"
        )
        assert tracked.inference_attempts == 0
