"""End-to-end wiring tests for pipeline disaggregation through the process manager.

No GPU and no full scheduling loop: these drive the real process-manager wiring at its seams (the
scheduler's disaggregation router, the orchestrator, the message dispatcher, and the job tracker) with
fake role processes, to prove the integration contracts:

- an eligible job is routed to the disaggregated pipeline, its sampler pinned (booked) so it cannot be
  double-booked, and the pin is released the instant sampling finishes (freeing the slot for the next job);
- a job whose image lane is absent stays on the monolithic path;
- a job parked in the decoding stage is not reaped by the orphaned-in-progress watchdog while the
  orchestrator holds it, and a give-up faults it rather than letting it vanish.
"""

from __future__ import annotations

import base64
import time
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import (
    GENERATION_STATE,
    HordeImageResult,
    HordePostProcessControlMessage,
    HordeProcessMessage,
    HordeProcessState,
    HordeSampleResultMessage,
    HordeTextEncodeResultMessage,
    HordeVaeDecodeControlMessage,
    HordeVaeDecodeResultMessage,
    SampleSliceResult,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.simulation.fake_worker_processes import (
    FakeInferenceProcess,
)
from horde_worker_regen.process_management.workers.component_lane_process import HordeComponentLaneProcess
from horde_worker_regen.process_management.workers.disaggregation_orchestrator import (
    _RESOURCE_DEFER_SECONDS,
    DisaggJobStage,
    DisaggregatedFault,
)
from horde_worker_regen.process_management.workers.vae_lane_process import HordeVaeLaneProcess
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_testable_process_manager,
)

_MODEL = "SDXL 1.0"


def _sdxl_reference() -> dict[str, object]:
    return {
        _MODEL: make_mock_model_reference_record(
            _MODEL,
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
        ),
    }


def _make_manager_with_roles(
    *,
    include_lane: bool = True,
    include_component: bool = True,
) -> tuple[HordeWorkerProcessManager, HordeProcessInfo]:
    """Build a testable manager with disaggregation on and inference/component/lane processes injected."""
    ref = _sdxl_reference()
    pm = make_testable_process_manager(
        enable_pipeline_disaggregation=True,
        post_processing_lane_enabled=False,
        stable_diffusion_reference=ref,  # type: ignore[arg-type]
    )
    pm._model_metadata.set_reference(ref)  # type: ignore[arg-type]

    inference = make_mock_process_info(0, model_name=_MODEL, process_type=HordeProcessType.INFERENCE)
    pm._process_map[0] = inference
    if include_component:
        pm._process_map[1] = make_mock_process_info(1, model_name=None, process_type=HordeProcessType.COMPONENT)
    if include_lane:
        pm._process_map[2] = make_mock_process_info(2, model_name=None, process_type=HordeProcessType.VAE_LANE)
    return pm, inference


_USABLE_SOURCE_IMAGE = base64.b64encode(b"fake-source-image-bytes").decode("ascii")


def _make_source_job(
    *,
    source_processing: str,
    source_image: str | None,
    model: str = _MODEL,
    transparent: bool = False,
) -> ImageGenerateJobPopResponse:
    """Build a real pop response with an explicit source_processing/source_image, per the conftest idiom.

    ``make_job_pop_response`` does not expose the source fields, and the routing decision under test derives
    entirely from them, so these are constructed directly. An inline base64 ``source_image`` is a usable
    source (it decodes); ``None`` is the mislabeled-txt2img case the SDK resolves back to txt2img.
    """
    job_id = uuid.uuid4()
    data: dict[str, Any] = {
        "id": str(job_id),
        "ids": [str(job_id)],
        "model": model,
        "payload": {
            "prompt": "test prompt",
            "width": 512,
            "height": 512,
            "ddim_steps": 30,
            "n_iter": 1,
            "seed": "42",
            "sampler_name": "k_euler",
            "transparent": transparent,
        },
        "skipped": {},
        "source_processing": source_processing,
    }
    if source_image is not None:
        data["source_image"] = source_image
    return ImageGenerateJobPopResponse(**data)


def test_eligibility_derives_from_effective_source_processing() -> None:
    """Class-eligibility keys on the SDK's effective (post-fallback) mode, not the raw pop field.

    This widens the eligible set relative to the raw field: an inpainting/outpainting pop carrying no usable
    source resolves to txt2img and becomes eligible as the txt2img job it actually runs, while the same pop
    with a usable source keeps its inpainting mode and stays monolithic.
    """
    pm, _inference = _make_manager_with_roles()

    unusable_inpaint = _make_source_job(source_processing="inpainting", source_image=None)
    assert pm._disaggregation_class_eligible(unusable_inpaint) is True

    usable_inpaint = _make_source_job(source_processing="inpainting", source_image=_USABLE_SOURCE_IMAGE)
    assert pm._disaggregation_class_eligible(usable_inpaint) is False


def test_transparent_jobs_are_not_disaggregation_eligible() -> None:
    """A transparent (layerdiffuse) job routes monolithic; its staged decode graph is not identity-validated."""
    pm, _inference = _make_manager_with_roles()

    transparent_job = _make_source_job(source_processing="txt2img", source_image=None, transparent=True)
    assert pm._disaggregation_class_eligible(transparent_job) is False

    opaque_job = _make_source_job(source_processing="txt2img", source_image=None, transparent=False)
    assert pm._disaggregation_class_eligible(opaque_job) is True


@pytest.mark.asyncio
async def test_mislabeled_img2img_registers_at_conditioning_stage_without_source_latent() -> None:
    """A pop tagged img2img with no source image routes needs_source_latent=False and enters at conditioning."""
    pm, inference = _make_manager_with_roles()
    job = _make_source_job(source_processing="img2img", source_image=None)
    assert pm._is_disaggregatable_sdk_job(job) is True

    await pm._job_tracker.record_popped_job(job)
    registered = await pm._register_disaggregated_job(job, inference)
    assert registered is True

    state = pm._disaggregation_orchestrator._jobs[str(job.id_)]
    assert state.needs_source_latent is False
    assert state.stage == DisaggJobStage.AWAITING_CONDITIONING


@pytest.mark.asyncio
async def test_genuine_img2img_registers_at_source_latent_stage() -> None:
    """A pop tagged img2img with a usable source image routes needs_source_latent=True (source-latent entry)."""
    pm, inference = _make_manager_with_roles()
    job = _make_source_job(source_processing="img2img", source_image=_USABLE_SOURCE_IMAGE)
    assert pm._is_disaggregatable_sdk_job(job) is True

    await pm._job_tracker.record_popped_job(job)
    registered = await pm._register_disaggregated_job(job, inference)
    assert registered is True

    state = pm._disaggregation_orchestrator._jobs[str(job.id_)]
    assert state.needs_source_latent is True
    assert state.stage == DisaggJobStage.AWAITING_SOURCE_LATENT


@pytest.mark.asyncio
async def test_mislabeled_img2img_flows_txt2img_dag_end_to_end() -> None:
    """A pop mislabeled img2img (no source) flows the txt2img DAG: encode -> sample -> decode -> safety.

    The disaggregation entry stage is conditioning, not source-latent, so no VAE source-encode of a
    placeholder image ever runs; the job completes and hands off to safety with its sampler pin released.
    """
    pm, inference = _make_manager_with_roles()
    job = _make_source_job(source_processing="img2img", source_image=None)
    await pm._job_tracker.record_popped_job(job)

    routed = await pm._inference_scheduler._dispatch_disaggregated(
        job,
        inference,
        dispatched_device_index=None,
        degraded_dispatch=False,
    )
    assert routed is True

    orchestrator = pm._disaggregation_orchestrator
    state = orchestrator._jobs[str(job.id_)]
    assert state.needs_source_latent is False
    assert state.stage == DisaggJobStage.AWAITING_CONDITIONING  # txt2img entry: no source-latent encode

    orchestrator.tick()  # dispatch text-encode to the component process
    await orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job.id_,
            positive_conditioning_bytes=b"pos",
            negative_conditioning_bytes=b"neg",
            state=GENERATION_STATE.ok,
        ),
    )
    orchestrator.tick()  # dispatch the sample stage to the pinned sampler
    await orchestrator.handle_stage_result(
        HordeSampleResultMessage(
            process_id=0,
            process_launch_identifier=0,
            info="",
            results=[SampleSliceResult(job_id=job.id_, latent_bytes=b"latent", state=GENERATION_STATE.ok)],
        ),
    )
    tracked = pm._job_tracker.get_tracked_job(job.id_)
    assert tracked is not None and tracked.stage == JobStage.DISAGGREGATION_DECODING

    orchestrator.tick()  # dispatch the decode stage to the image lane
    await orchestrator.handle_stage_result(
        HordeVaeDecodeResultMessage(
            process_id=2,
            process_launch_identifier=0,
            info="",
            job_id=job.id_,
            job_image_results=[HordeImageResult(image_bytes=b"img")],
            state=GENERATION_STATE.ok,
        ),
    )
    await pm.drive_disaggregation()

    pending_safety_ids = [ji.sdk_api_job_info.id_ for ji in pm._job_tracker.jobs_pending_safety_check]
    assert job.id_ in pending_safety_ids
    assert pm._process_map.is_reserved_for_disaggregation(0) is False  # no reservation leaked


def test_eligibility_requires_flag_family_and_live_roles() -> None:
    """A job is disaggregated only with the flag, an SD1.5/SDXL family, no control, and both roles live."""
    pm, _inference = _make_manager_with_roles()
    job = make_job_pop_response(model=_MODEL)
    assert pm._is_disaggregatable_sdk_job(job) is True

    # Wrong family stays monolithic (baseline resolved from the reference, not the name).
    pm._model_metadata.set_reference(
        {_MODEL: make_mock_model_reference_record(_MODEL, baseline=KNOWN_IMAGE_GENERATION_BASELINE.flux_1)},  # type: ignore[arg-type]
    )
    assert pm._is_disaggregatable_sdk_job(job) is False


def test_eligible_job_stays_monolithic_when_image_lane_absent() -> None:
    """With the image lane missing, an otherwise-eligible job is not disaggregated (monolithic fallback)."""
    pm, _inference = _make_manager_with_roles(include_lane=False)
    job = make_job_pop_response(model=_MODEL)
    assert pm._is_disaggregatable_sdk_job(job) is False


def test_busy_post_process_lane_does_not_delay_decode_and_roles_need_vae_lane() -> None:
    """VAE stages resolve the VAE lane regardless of PP occupancy; roles-live keys on the VAE lane.

    The whole point of splitting VAE off the post-processing lane: a post-processing lane saturated with
    upscale/face-fix work can never sit in front of a critical-path decode, and the disaggregation
    roles-live gate no longer depends on the post-processing lane at all.
    """
    pm, _inference = _make_manager_with_roles()
    # A dedicated post-processing lane, busy on its own work, sits alongside the (idle) VAE lane.
    busy_pp = make_mock_process_info(3, model_name=None, process_type=HordeProcessType.POST_PROCESS)
    busy_pp.last_process_state = HordeProcessState.POST_PROCESSING
    pm._process_map[3] = busy_pp

    # Decode dispatch resolves the idle VAE lane (process 2), never the busy PP lane.
    assert pm._process_map.get_first_available_vae_lane_process() is pm._process_map[2]
    assert pm._disaggregation_roles_live() is True

    # Removing the VAE lane makes roles not live even though the PP lane remains: the gate keys on the VAE lane.
    del pm._process_map[2]
    assert pm._disaggregation_roles_live() is False


def test_whole_card_residency_suppresses_disaggregation_dispatch() -> None:
    """While a card holds a whole-card residency, a class-eligible job routes monolithic (dispatch suppressed).

    The class-eligibility predicate stays True (so residency forecasting still charges the job sampler-only),
    but the dispatch-time predicate returns False, so new jobs do not dispatch encodes into a reserved card.
    """
    pm, _inference = _make_manager_with_roles()
    job = make_job_pop_response(model=_MODEL)
    assert pm._is_disaggregatable_sdk_job(job) is True

    pm._inference_scheduler.is_whole_card_residency_active = lambda: True  # type: ignore[method-assign]
    assert pm._disaggregation_class_eligible(job) is True  # forecast still prices it sampler-only
    assert pm._is_disaggregatable_sdk_job(job) is False  # but no encode dispatched into the reserved card


def test_concurrent_sampling_gate_is_wired_to_the_shared_arbiter() -> None:
    """The orchestrator's concurrent-sampling gate decides through the manager's shared VRAM arbiter.

    The gate arbitrates a second concurrent sampler against static device headroom, now the arbiter's
    authoritative answer. This proves the arbiter is the same instance the manager freezes each cycle, that
    the injected peak estimator runs against the real manager without raising, and that with no cycle snapshot
    frozen the gate admits rather than wedging on missing telemetry.
    """
    pm, _inference = _make_manager_with_roles()
    orchestrator = pm._disaggregation_orchestrator

    assert orchestrator._estimate_sampling_peak_mb == pm._inference_scheduler.estimate_disaggregated_sampling_peak_mb
    assert orchestrator._vram_arbiter is pm._vram_arbiter

    # The peak estimator runs against the real manager without error (returns a float or None).
    job_info = SimpleNamespace(sdk_api_job_info=make_job_pop_response(model=_MODEL))
    _peak = pm._inference_scheduler.estimate_disaggregated_sampling_peak_mb(job_info)  # type: ignore[arg-type]

    # No cycle snapshot has been frozen on the arbiter, so the gate admits on missing telemetry.
    assert orchestrator._admit_concurrent_sampling(8260.0) is True


@pytest.mark.asyncio
async def test_resource_fault_reroutes_job_to_monolithic_and_it_does_not_reenter() -> None:
    """A disaggregated stage failing resource-class past the defer window re-routes the job monolithically.

    The job returns to the pending-inference queue (still owned, its sampler pin freed) with the
    disaggregation-declined latch set, so the eligibility predicate now keeps it monolithic: it cannot bounce
    straight back into the disagg pipeline.
    """
    pm, inference = _make_manager_with_roles()
    job = make_job_pop_response(model=_MODEL)
    await pm._job_tracker.record_popped_job(job)
    await pm._inference_scheduler._dispatch_disaggregated(
        job,
        inference,
        dispatched_device_index=None,
        degraded_dispatch=False,
    )
    assert pm._process_map.is_reserved_for_disaggregation(0) is True

    orchestrator = pm._disaggregation_orchestrator
    virtual_now = [0.0]
    orchestrator._clock = lambda: virtual_now[0]  # type: ignore[method-assign]
    orchestrator.tick()  # dispatch text-encode to the component process

    def _resource_fault() -> HordeTextEncodeResultMessage:
        return HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job.id_,
            positive_conditioning_bytes=None,
            negative_conditioning_bytes=None,
            state=GENERATION_STATE.faulted,
            fault_is_resource_class=True,
        )

    # First resource fault anchors the defer window (the job is deferred, not forfeited or re-routed).
    await orchestrator.handle_stage_result(_resource_fault())
    assert str(job.id_) in orchestrator._jobs
    assert pm._is_disaggregatable_sdk_job(job) is True

    # A recurrence past the window re-routes the job to the monolithic path.
    virtual_now[0] = _RESOURCE_DEFER_SECONDS + 1.0
    await orchestrator.handle_stage_result(_resource_fault())

    assert str(job.id_) not in orchestrator._jobs  # left the pipeline
    assert pm._process_map.is_reserved_for_disaggregation(0) is False  # sampler pin freed
    tracked = pm._job_tracker.get_tracked_job(job.id_)
    assert tracked is not None and tracked.stage == JobStage.PENDING_INFERENCE  # returned to the claim path
    assert job in pm._job_tracker.jobs_pending_inference
    # Latched monolithic: the re-claim cannot route back into disaggregation.
    assert pm._is_disaggregatable_sdk_job(job) is False
    assert pm._disaggregation_class_eligible(job) is False
    # Not faulted: the job runs whole rather than being dropped.
    assert tracked.job_info is not None and tracked.job_info.state != GENERATION_STATE.faulted


@pytest.mark.asyncio
async def test_disaggregated_flow_pins_releases_and_completes() -> None:
    """Route -> encode -> sample (slot released on the sample result) -> decode -> safety, with a freed slot."""
    pm, inference = _make_manager_with_roles()
    job = make_job_pop_response(model=_MODEL)
    await pm._job_tracker.record_popped_job(job)
    assert pm._is_disaggregatable_sdk_job(job) is True

    # The scheduler's dispatch seam routes the job to disaggregation in place of START_INFERENCE.
    routed = await pm._inference_scheduler._dispatch_disaggregated(
        job,
        inference,
        dispatched_device_index=None,
        degraded_dispatch=False,
    )
    assert routed is True
    assert pm._process_map.is_reserved_for_disaggregation(0) is True
    assert job in pm._job_tracker.jobs_in_progress
    # While pinned the sampler is booked: the scheduler cannot hand it another job.
    assert pm._process_map.get_first_available_inference_process() is None
    assert pm._process_map.get_process_by_horde_model_name(_MODEL) is None

    orchestrator = pm._disaggregation_orchestrator
    orchestrator.tick()  # dispatch text-encode to the component process

    await orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job.id_,
            positive_conditioning_bytes=b"pos",
            negative_conditioning_bytes=b"neg",
            state=GENERATION_STATE.ok,
        ),
    )
    orchestrator.tick()  # dispatch the sample stage to the pinned sampler (process 0)
    assert len(inference.pipe_connection.send.call_args_list) >= 1

    await orchestrator.handle_stage_result(
        HordeSampleResultMessage(
            process_id=0,
            process_launch_identifier=0,
            info="",
            results=[SampleSliceResult(job_id=job.id_, latent_bytes=b"latent", state=GENERATION_STATE.ok)],
        ),
    )
    # Early release: the sampler slot is freed and the job left the inference concurrency cap.
    assert pm._process_map.is_reserved_for_disaggregation(0) is False
    assert job not in pm._job_tracker.jobs_in_progress
    tracked = pm._job_tracker.get_tracked_job(job.id_)
    assert tracked is not None and tracked.stage == JobStage.DISAGGREGATION_DECODING
    # The freed slot is schedulable again, so a second job could be admitted onto it while this one decodes.
    assert pm._process_map.get_first_available_inference_process() is inference

    orchestrator.tick()  # dispatch the decode stage to the image lane
    await orchestrator.handle_stage_result(
        HordeVaeDecodeResultMessage(
            process_id=2,
            process_launch_identifier=0,
            info="",
            job_id=job.id_,
            job_image_results=[HordeImageResult(image_bytes=b"img")],
            state=GENERATION_STATE.ok,
        ),
    )
    await pm.drive_disaggregation()  # route the completion into the safety pipeline

    pending_safety_ids = [ji.sdk_api_job_info.id_ for ji in pm._job_tracker.jobs_pending_safety_check]
    assert job.id_ in pending_safety_ids
    assert pm._process_map.is_reserved_for_disaggregation(0) is False  # no reservation leaked


async def _run_disaggregated_job_through_decode(
    pm: HordeWorkerProcessManager,
    inference: HordeProcessInfo,
    job: ImageGenerateJobPopResponse,
) -> None:
    """Route a job through the disaggregated DAG (encode -> sample -> decode) up to its decode result.

    Leaves the decode completion queued on the orchestrator; the caller drives ``drive_disaggregation`` to
    route it into the safety/post-processing pipeline. The decode result carries a single raw image, exactly
    what the VAE lane now returns (no inline post-processing).
    """
    await pm._job_tracker.record_popped_job(job)
    await pm._inference_scheduler._dispatch_disaggregated(
        job,
        inference,
        dispatched_device_index=None,
        degraded_dispatch=False,
    )
    orchestrator = pm._disaggregation_orchestrator
    orchestrator.tick()  # dispatch text-encode to the component process
    await orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job.id_,
            positive_conditioning_bytes=b"pos",
            negative_conditioning_bytes=b"neg",
            state=GENERATION_STATE.ok,
        ),
    )
    orchestrator.tick()  # dispatch the sample stage to the pinned sampler
    await orchestrator.handle_stage_result(
        HordeSampleResultMessage(
            process_id=0,
            process_launch_identifier=0,
            info="",
            results=[SampleSliceResult(job_id=job.id_, latent_bytes=b"latent", state=GENERATION_STATE.ok)],
        ),
    )
    orchestrator.tick()  # dispatch the decode stage to the VAE lane
    await orchestrator.handle_stage_result(
        HordeVaeDecodeResultMessage(
            process_id=2,
            process_launch_identifier=0,
            info="",
            job_id=job.id_,
            job_image_results=[HordeImageResult(image_bytes=b"img")],
            state=GENERATION_STATE.ok,
        ),
    )


@pytest.mark.asyncio
async def test_disaggregated_job_with_post_processing_routes_through_the_dedicated_pp_lane() -> None:
    """A disaggregated job requesting post-processing routes decode -> synthetic result -> the PP lane.

    The VAE lane's decode control message carries no post-processing (that field is gone), so the lane only
    decodes; the synthetic completion then flows the identical post-processing path a monolithic completion
    takes, dispatching a ``HordePostProcessControlMessage`` to a dedicated POST_PROCESS process. This holds
    even though ``post_processing_lane_enabled`` is False in the harness, because disaggregation forces the
    lane on.
    """
    pm, inference = _make_manager_with_roles()
    post_process_lane = make_mock_process_info(3, model_name=None, process_type=HordeProcessType.POST_PROCESS)
    pm._process_map[3] = post_process_lane

    job = make_job_pop_response(model=_MODEL, post_processing=["RealESRGAN_x4plus"])
    await _run_disaggregated_job_through_decode(pm, inference, job)

    # The VAE lane received a decode with no post-processing work: the field no longer exists on the message.
    decode_message = pm._process_map[2].pipe_connection.send.call_args.args[0]
    assert isinstance(decode_message, HordeVaeDecodeControlMessage)
    assert not hasattr(decode_message, "post_processing")

    await pm.drive_disaggregation()  # route the completion; a PP request must go to the PP lane, not safety

    pending_pp_ids = [ji.sdk_api_job_info.id_ for ji in pm._job_tracker.jobs_pending_post_processing]
    assert job.id_ in pending_pp_ids  # queued for the dedicated post-processing lane
    pending_safety_ids = [ji.sdk_api_job_info.id_ for ji in pm._job_tracker.jobs_pending_safety_check]
    assert job.id_ not in pending_safety_ids  # did not skip post-processing to safety

    # The PP orchestrator dispatches the queued job to the POST_PROCESS lane with the raw decoded image.
    pm._post_process_orchestrator._sampling_coresidency_check = lambda reserve_mb: True  # type: ignore[method-assign]
    await pm.start_post_processing()

    pp_control_message = post_process_lane.pipe_connection.send.call_args.args[0]
    assert isinstance(pp_control_message, HordePostProcessControlMessage)
    assert pp_control_message.job_id == job.id_
    assert pp_control_message.images_bytes == [b"img"]  # the VAE lane's raw decode output
    assert pp_control_message.post_processing == ["RealESRGAN_x4plus"]


@pytest.mark.asyncio
async def test_disaggregated_job_without_post_processing_flows_straight_to_safety() -> None:
    """A disaggregated job requesting no post-processing routes decode -> safety, dispatching no PP work."""
    pm, inference = _make_manager_with_roles()
    post_process_lane = make_mock_process_info(3, model_name=None, process_type=HordeProcessType.POST_PROCESS)
    pm._process_map[3] = post_process_lane

    job = make_job_pop_response(model=_MODEL)  # no post_processing
    await _run_disaggregated_job_through_decode(pm, inference, job)
    await pm.drive_disaggregation()

    pending_safety_ids = [ji.sdk_api_job_info.id_ for ji in pm._job_tracker.jobs_pending_safety_check]
    assert job.id_ in pending_safety_ids  # straight to safety
    pending_pp_ids = [ji.sdk_api_job_info.id_ for ji in pm._job_tracker.jobs_pending_post_processing]
    assert job.id_ not in pending_pp_ids  # never queued for post-processing

    pm._post_process_orchestrator._sampling_coresidency_check = lambda reserve_mb: True  # type: ignore[method-assign]
    await pm.start_post_processing()
    post_process_lane.pipe_connection.send.assert_not_called()  # no post-processing dispatched


@pytest.mark.asyncio
async def test_decoding_job_not_reaped_by_orphan_watchdog_and_faults_not_vanishes() -> None:
    """A job in the decoding stage is not punted by the in-progress watchdog; a give-up faults it (not lost)."""
    pm, inference = _make_manager_with_roles()
    job = make_job_pop_response(model=_MODEL)
    await pm._job_tracker.record_popped_job(job)
    await pm._inference_scheduler._dispatch_disaggregated(
        job,
        inference,
        dispatched_device_index=None,
        degraded_dispatch=False,
    )
    orchestrator = pm._disaggregation_orchestrator
    orchestrator.tick()
    await orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job.id_,
            positive_conditioning_bytes=b"pos",
            negative_conditioning_bytes=b"neg",
            state=GENERATION_STATE.ok,
        ),
    )
    orchestrator.tick()
    await orchestrator.handle_stage_result(
        HordeSampleResultMessage(
            process_id=0,
            process_launch_identifier=0,
            info="",
            results=[SampleSliceResult(job_id=job.id_, latent_bytes=b"latent", state=GENERATION_STATE.ok)],
        ),
    )
    tracked = pm._job_tracker.get_tracked_job(job.id_)
    assert tracked is not None and tracked.stage == JobStage.DISAGGREGATION_DECODING

    # The orphaned-in-progress watchdog scans only INFERENCE_IN_PROGRESS jobs, so a decoding job (held by
    # the orchestrator) is never punted, even long past the grace window.
    coordinator = pm._recovery_coordinator
    coordinator._clock = lambda: 10_000_000.0  # far past ORPHAN_IN_PROGRESS_GRACE_SECONDS
    coordinator.reconcile_orphaned_in_progress_jobs()
    still_tracked = pm._job_tracker.get_tracked_job(job.id_)
    assert still_tracked is not None and still_tracked.stage == JobStage.DISAGGREGATION_DECODING

    # If the orchestrator gives up (its per-stage patience faults the job), it terminates as a fault rather
    # than vanishing: exhaust the retries so the give-up is terminal, then feed the faulted completion.
    still_tracked.inference_attempts = 99
    await pm.drive_disaggregation()  # no completions yet; the orchestrator still holds the job
    pm._disaggregated_completions.append(
        (still_tracked.job_info, [], GENERATION_STATE.faulted, DisaggregatedFault(reason="sample faulted")),
    )
    await pm.drive_disaggregation()

    terminal = pm._job_tracker.get_tracked_job(job.id_)
    assert terminal is not None  # not silently dropped
    assert terminal.stage == JobStage.PENDING_SUBMIT
    assert terminal.job_info is not None and terminal.job_info.state == GENERATION_STATE.faulted


@pytest.mark.asyncio
async def test_orphan_punt_releases_the_job_from_the_orchestrator() -> None:
    """When the orphaned-in-progress watchdog punts a held disaggregated job, the orchestrator is told.

    The observed wedge: the pinned sampler hung and was replaced under the same slot id, so the replacement
    (which references no job) left the in-flight disaggregated job unowned. The orphaned-job watchdog punted
    it from the tracker, but nothing informed the orchestrator, which kept holding its pin, reservation, and
    ledger. The punt path now calls the release seam, so the orchestrator drops the job the tracker no longer
    tracks (the invariant: it never holds a job the tracker has released).
    """
    pm, inference = _make_manager_with_roles()
    job = make_job_pop_response(model=_MODEL)
    await pm._job_tracker.record_popped_job(job)
    await pm._inference_scheduler._dispatch_disaggregated(
        job,
        inference,
        dispatched_device_index=None,
        degraded_dispatch=False,
    )
    orchestrator = pm._disaggregation_orchestrator
    assert str(job.id_) in orchestrator._jobs  # held by the orchestrator
    assert pm._process_map.is_reserved_for_disaggregation(0) is True

    # The pinned sampler (process 0) hung and was replaced under the same slot id: the replacement references
    # no job, so the in-flight job now reads as unowned (its reservation leaked onto the cold replacement).
    pm._process_map[0] = make_mock_process_info(0, model_name=None, process_type=HordeProcessType.INFERENCE)

    coordinator = pm._recovery_coordinator
    assert coordinator.inference_slot_owns_job(job.id_) is False
    # Backdate the first-seen time past the grace window so the next reconcile actually punts.
    coordinator.orphan_in_progress_since[job.id_] = time.time() - (coordinator.ORPHAN_IN_PROGRESS_GRACE_SECONDS + 1)
    coordinator.reconcile_orphaned_in_progress_jobs()

    # The job left the tracker (punted back to the claim path) AND the orchestrator released everything it held.
    assert pm._job_tracker.get_stage(job.id_) != JobStage.INFERENCE_IN_PROGRESS
    assert str(job.id_) not in orchestrator._jobs  # no longer held
    assert pm._process_map.is_reserved_for_disaggregation(0) is False  # pin/reservation dropped
    assert orchestrator._active_sampling_peaks == {}  # no ledger entry leaked


@pytest.mark.asyncio
async def test_faulted_disaggregated_completion_carries_reason_and_faulting_process_id() -> None:
    """The synthetic result the parent builds for a faulted disaggregated job is not blank and names the process.

    A genuine (non-resource) text-encode fault carries the child's exception text to the orchestrator, which
    threads it into the completion. The parent formats it into the synthetic ``HordeInferenceResultMessage``
    ``info`` in the ``Model: ... Error: ...`` shape the fault detectors read, and attributes the result to the
    faulting child (process 1), not the blank/zero default the drop points produced before.
    """
    pm, inference = _make_manager_with_roles()
    job = make_job_pop_response(model=_MODEL)
    await pm._job_tracker.record_popped_job(job)
    await pm._inference_scheduler._dispatch_disaggregated(
        job,
        inference,
        dispatched_device_index=None,
        degraded_dispatch=False,
    )
    orchestrator = pm._disaggregation_orchestrator
    orchestrator.tick()  # dispatch text-encode to the component process (process 1)

    await orchestrator.handle_stage_result(
        HordeTextEncodeResultMessage(
            process_id=1,
            process_launch_identifier=0,
            info="",
            job_id=job.id_,
            positive_conditioning_bytes=None,
            negative_conditioning_bytes=None,
            state=GENERATION_STATE.faulted,
            fault_is_resource_class=False,
            fault_reason="RuntimeError: CUDA out of memory",
        ),
    )

    captured = AsyncMock()
    pm._message_dispatcher.handle_synthetic_inference_result = captured  # type: ignore[method-assign]
    await pm.drive_disaggregation()

    captured.assert_awaited_once()
    result = captured.await_args.args[0]
    assert result.state == GENERATION_STATE.faulted
    assert result.process_id == 1  # the faulting child, not 0/safety
    assert f"Model: {_MODEL}. Error: RuntimeError: CUDA out of memory" in result.info  # detector-shaped, not blank


class _RecordingQueue:
    """Captures every message a fake role process emits, in order (a stand-in for the process queue)."""

    def __init__(self) -> None:
        self.messages: list[HordeProcessMessage] = []

    def put(self, message: HordeProcessMessage) -> None:
        self.messages.append(message)


async def _pump_stage_through_fake(
    pm: HordeWorkerProcessManager,
    *,
    process_id: int,
    fake: object,
    queue: _RecordingQueue,
) -> None:
    """Feed the stage control message the orchestrator just dispatched into a real fake, then replay its output.

    This closes the loop the way the live worker does: the orchestrator's dispatch reaches the role process
    over its pipe, the fake produces the same state-change and result traffic a real child would, and that
    traffic is replayed through the real ``MessageDispatcher`` (which routes stage results to the
    orchestrator and puts the stage lanes' reused busy-state changes through the whole-job handler that the
    disaggregation crash guard protects). Nothing is hand-fabricated: the result messages come from the fake.
    """
    control_message = pm._process_map[process_id].pipe_connection.send.call_args.args[0]
    queue.messages.clear()
    fake._receive_and_handle_control_message(control_message)  # type: ignore[attr-defined]
    emitted = list(queue.messages)

    dispatcher = pm._message_dispatcher
    dispatcher._process_message_queue.empty.side_effect = [False] * len(emitted) + [True]  # type: ignore[attr-defined]
    dispatcher._process_message_queue.get.side_effect = emitted  # type: ignore[attr-defined]
    await dispatcher.receive_and_handle_process_messages()


@pytest.mark.asyncio
async def test_disaggregated_flow_runs_through_the_fakes_message_driven() -> None:
    """A txt2img job flows text-encode -> sample -> decode -> done entirely on fake-produced messages.

    Unlike ``test_disaggregated_flow_pins_releases_and_completes`` (which hand-feeds stage results to the
    orchestrator), here the real fake role processes emit the stage results and state traffic, which is
    replayed through the real dispatcher. This exercises the disaggregation state-traffic path live: the
    COMPONENT lane and the pinned sampler both report ``INFERENCE_STARTING`` while holding no whole-job
    model bookkeeping, which must not crash the parent.
    """
    pm, inference = _make_manager_with_roles()
    job = make_job_pop_response(model=_MODEL)
    await pm._job_tracker.record_popped_job(job)

    routed = await pm._inference_scheduler._dispatch_disaggregated(
        job,
        inference,
        dispatched_device_index=None,
        degraded_dispatch=False,
    )
    assert routed is True
    assert pm._process_map.is_reserved_for_disaggregation(0) is True

    # The real fakes stand in for the three role processes, wired to matching ids/launch and their own queues.
    encode_queue = _RecordingQueue()
    encode_service = HordeComponentLaneProcess(
        process_id=1,
        process_message_queue=encode_queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
        dry_run=True,
    )
    sample_queue = _RecordingQueue()
    sampler = FakeInferenceProcess(
        process_id=0,
        process_message_queue=sample_queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        inference_semaphore=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
    )
    decode_queue = _RecordingQueue()
    image_lane = HordeVaeLaneProcess(
        process_id=2,
        process_message_queue=decode_queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
        dry_run=True,
    )

    orchestrator = pm._disaggregation_orchestrator

    # Stage 1: text-encode on the COMPONENT lane. Each stage sends exactly one control message to its
    # target's pipe, so the helper reads that pipe's latest send as the dispatched stage message.
    orchestrator.tick()
    await _pump_stage_through_fake(pm, process_id=1, fake=encode_service, queue=encode_queue)
    # Text-encode advanced the orchestrator to sampling; the job stays in the inference concurrency cap.
    assert job in pm._job_tracker.jobs_in_progress

    # Stage 2: sample on the pinned INFERENCE slot (its INFERENCE_STARTING must not fault the parent).
    orchestrator.tick()
    await _pump_stage_through_fake(pm, process_id=0, fake=sampler, queue=sample_queue)
    # The sample result released the pin and moved the job to decoding, freeing the slot.
    assert pm._process_map.is_reserved_for_disaggregation(0) is False
    tracked = pm._job_tracker.get_tracked_job(job.id_)
    assert tracked is not None and tracked.stage == JobStage.DISAGGREGATION_DECODING

    # Stage 3: decode on the image lane, then route the completion into the safety pipeline.
    orchestrator.tick()
    await _pump_stage_through_fake(pm, process_id=2, fake=image_lane, queue=decode_queue)
    await pm.drive_disaggregation()

    pending_safety_ids = [ji.sdk_api_job_info.id_ for ji in pm._job_tracker.jobs_pending_safety_check]
    assert job.id_ in pending_safety_ids  # reached done, handed off to safety
    assert str(job.id_) not in orchestrator._jobs  # removed from the pipeline
    assert pm._process_map.is_reserved_for_disaggregation(0) is False  # no reservation leaked
