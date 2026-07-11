"""Behavioral liveness tests for the parent-side image-utilities job flow.

The theme is that the worker keeps working no matter how the utilities lane behaves. Two flows are covered:

- **Pre-annotation** (control map derived off-GPU): a servable controlnet job is annotated on the lane;
  its map is injected into inference (or, for a ``return_control_map`` job, delivered straight to submit).
  Every failure mode (lane death, an annotation fault, an age-out, an unservable control type, the lane
  disabled) releases the job to hordelib's in-graph path. A parked job is never lost and never oscillates.
- **Background-strip tail** (the last post-processing transform): a generation job's ``strip_background`` is
  split from the post-processing lane and run on the utilities lane, strictly last. A strip-only job skips
  the post-processing lane. A lane death or age-out mid-strip is a no-image fault (reissue), matching the
  post-processing lane's policy for a transform that could not run.
"""

from __future__ import annotations

import base64
import time
import uuid
from unittest.mock import Mock

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.generation_parameters.alchemy.consts import is_strip_background_form

from horde_worker_regen.process_management.ipc.messages import (
    HordeAnnotationResultMessage,
    HordeAnnotatorAvailabilityMessage,
    HordeControlFlag,
    HordeImageResult,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordePostProcessResultMessage,
    HordeProcessState,
    HordeStartAnnotationControlMessage,
    HordeStartStripControlMessage,
    HordeStripResultMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs import alchemy_popper
from horde_worker_regen.process_management.jobs.alchemy_popper import expand_offered_forms
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_testable_process_manager,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.ipc.test_message_dispatch import _make_dispatcher
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_UTILITIES_PROCESS_ID = 7
_SOURCE_IMAGE_B64 = base64.b64encode(b"fake-source-image-bytes").decode()


def _sent(process_info: HordeProcessInfo) -> Mock:
    """Return the mocked pipe ``send`` for a test process, for asserting dispatched control messages."""
    send = process_info.pipe_connection.send
    assert isinstance(send, Mock)
    return send


def _make_controlnet_job(
    *,
    control_type: str = "canny",
    return_control_map: bool = False,
    image_is_control: bool = False,
) -> ImageGenerateJobPopResponse:
    """Build a real controlnet generation pop response carrying an inline (base64) source image."""
    job_id = str(uuid.uuid4())
    data = {
        "id": job_id,
        "ids": [job_id],
        "model": "stable_diffusion",
        "source_image": _SOURCE_IMAGE_B64,
        "source_processing": "img2img",
        "payload": {
            "prompt": "a controlnet prompt",
            "width": 512,
            "height": 512,
            "ddim_steps": 20,
            "n_iter": 1,
            "seed": "42",
            "sampler_name": "k_euler",
            "control_type": control_type,
            "image_is_control": image_is_control,
            "return_control_map": return_control_map,
        },
    }
    return ImageGenerateJobPopResponse(**data)  # pyrefly: ignore - validated by pydantic


def _add_utilities_process(
    pm: HordeWorkerProcessManager,
    *,
    state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB,
) -> HordeProcessInfo:
    """Register an idle image-utilities lane process on the manager's process map and return it."""
    util = make_mock_process_info(
        _UTILITIES_PROCESS_ID,
        model_name=None,
        state=state,
        process_type=HordeProcessType.UTILITIES,
    )
    pm._process_map[_UTILITIES_PROCESS_ID] = util
    return util


def _mark_servable(pm: HordeWorkerProcessManager, control_types: list[str]) -> None:
    """Feed the manager an annotator-availability snapshot for the utilities lane."""
    pm._on_annotator_availability(
        HordeAnnotatorAvailabilityMessage(
            process_id=_UTILITIES_PROCESS_ID,
            process_launch_identifier=0,
            info="availability",
            servable_control_types=control_types,
        ),
    )


async def _park_one_controlnet_job(
    pm: HordeWorkerProcessManager,
    *,
    control_type: str = "canny",
) -> ImageGenerateJobPopResponse:
    """Enqueue a servable controlnet job, run the scan, and return the (now parked) job."""
    _add_utilities_process(pm)
    _mark_servable(pm, [control_type])
    job = _make_controlnet_job(control_type=control_type)
    await pm._job_tracker.record_popped_job(job, time_popped=time.time())
    pm._advance_pending_annotations()
    return job


class TestPreAnnotationHappyPath:
    """(a) A servable controlnet job is annotated off-GPU and dispatched with the derived map."""

    async def test_scan_parks_job_and_dispatches_annotation(self) -> None:
        """The scan moves a servable controlnet job to PENDING_ANNOTATION and sends START_ANNOTATION."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        _mark_servable(pm, ["canny"])
        job = _make_controlnet_job(control_type="canny")
        assert job.id_ is not None
        await pm._job_tracker.record_popped_job(job, time_popped=time.time())

        pm._advance_pending_annotations()

        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_ANNOTATION
        assert job not in pm._job_tracker.jobs_pending_inference
        sent = _sent(util).call_args.args[0]
        assert isinstance(sent, HordeStartAnnotationControlMessage)
        assert sent.control_flag == HordeControlFlag.START_ANNOTATION
        assert sent.job_id == job.id_
        assert sent.control_type == "canny"
        assert sent.source_image_bytes == b"fake-source-image-bytes"

    async def test_ok_result_releases_job_with_map_bytes(self) -> None:
        """A successful annotation result returns the job to inference carrying the control map bytes."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        job = await _park_one_controlnet_job(pm)
        assert job.id_ is not None

        pm._on_annotation_result(
            HordeAnnotationResultMessage(
                process_id=_UTILITIES_PROCESS_ID,
                process_launch_identifier=0,
                info="ok",
                job_id=job.id_,
                control_map_bytes=b"derived-control-map",
                state=GENERATION_STATE.ok,
            ),
        )

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.stage == JobStage.PENDING_INFERENCE
        assert tracked.premade_control_map_bytes == b"derived-control-map"
        assert job in pm._job_tracker.jobs_pending_inference

    async def test_dispatched_inference_message_carries_premade_map(self) -> None:
        """The scheduler copies a job's derived control map onto its START_INFERENCE message."""
        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        process_map = ProcessMap({0: process_info})
        horde_model_map = HordeModelMap(root={})
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        assert job.id_ is not None
        await track_popped_job_async(job_tracker, job, time_popped=time.time())
        tracked = job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        tracked.premade_control_map_bytes = b"scheduler-carried-map"
        horde_model_map.update_entry(
            horde_model_name="stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
        )
        assert await scheduler.start_inference() is True

        sent = _sent(process_info).call_args.args[0]
        assert isinstance(sent, HordeInferenceControlMessage)
        assert sent.premade_control_map_bytes == b"scheduler-carried-map"


class TestUtilitiesLaneDeath:
    """(b) A job parked when its utilities lane dies is released to in-graph, still schedulable."""

    async def test_lane_death_releases_parked_job_in_graph(self) -> None:
        """A parked job whose lane process is no longer alive re-enters inference with no premade map."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        _mark_servable(pm, ["canny"])
        job = _make_controlnet_job()
        assert job.id_ is not None
        await pm._job_tracker.record_popped_job(job, time_popped=time.time())
        pm._advance_pending_annotations()
        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_ANNOTATION

        util.mp_process.is_alive.return_value = False
        pm._advance_pending_annotations()

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.stage == JobStage.PENDING_INFERENCE
        assert tracked.premade_control_map_bytes is None
        assert job in pm._job_tracker.jobs_pending_inference


class TestAgeOut:
    """(c) A park that outlives the bounded timeout is released to in-graph."""

    async def test_age_out_releases_job_in_graph(self) -> None:
        """A job sitting in PENDING_ANNOTATION past the timeout is released with no premade map."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        job = await _park_one_controlnet_job(pm)
        assert job.id_ is not None

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        tracked.current_stage_since = time.time() - (pm._PENDING_ANNOTATION_TIMEOUT_SECONDS + 5.0)
        pm._advance_pending_annotations()

        assert tracked.stage == JobStage.PENDING_INFERENCE
        assert tracked.premade_control_map_bytes is None
        assert job in pm._job_tracker.jobs_pending_inference


class TestUnservableControlType:
    """(d) A controlnet job whose control type is not servable is never parked."""

    async def test_unservable_control_type_is_not_parked(self) -> None:
        """A job whose control type the lane cannot serve stays in PENDING_INFERENCE, no message sent."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        _mark_servable(pm, ["canny"])
        job = _make_controlnet_job(control_type="depth")
        assert job.id_ is not None
        await pm._job_tracker.record_popped_job(job, time_popped=time.time())

        pm._advance_pending_annotations()

        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
        assert not _sent(util).called
        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert not tracked.annotation_attempted


class TestLaneDisabled:
    """(e) With the lane disabled, the scan is a no-op: zero behavior change versus main."""

    async def test_disabled_lane_never_parks(self) -> None:
        """A default (mock-truthy but not `is True`) config must not activate pre-annotation."""
        pm = make_testable_process_manager()  # enable_image_utilities left at its mock default
        util = _add_utilities_process(pm)
        _mark_servable(pm, ["canny"])
        job = _make_controlnet_job()
        assert job.id_ is not None
        await pm._job_tracker.record_popped_job(job, time_popped=time.time())

        pm._advance_pending_annotations()

        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
        assert not _sent(util).called


class TestStripPopGating:
    """(f) The strip_background offer flips with the utilities lane's health."""

    def test_strip_offered_only_when_lane_healthy(self, monkeypatch) -> None:  # noqa: ANN001
        """strip_background is offered when the lane is up and withheld when it is down."""
        monkeypatch.setattr(alchemy_popper, "strip_background_available", lambda: True)
        bridge_data = make_mock_bridge_data(forms=["post-process"])

        offered_healthy = expand_offered_forms(bridge_data, utilities_lane_healthy=True)
        offered_unhealthy = expand_offered_forms(bridge_data, utilities_lane_healthy=False)

        assert any(is_strip_background_form(form) for form in offered_healthy)
        assert not any(is_strip_background_form(form) for form in offered_unhealthy)


class TestReturnControlMapDelivery:
    """(g) A servable return_control_map job is served by the lane and reaches submit without inference."""

    async def test_return_control_map_delivered_to_safety(self) -> None:
        """The derived control map becomes the job's image and the job goes to safety, never to inference."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        _add_utilities_process(pm)
        _mark_servable(pm, ["canny"])
        job = _make_controlnet_job(control_type="canny", return_control_map=True)
        assert job.id_ is not None
        await pm._job_tracker.record_popped_job(job, time_popped=time.time())

        pm._advance_pending_annotations()
        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_ANNOTATION

        pm._on_annotation_result(
            HordeAnnotationResultMessage(
                process_id=_UTILITIES_PROCESS_ID,
                process_launch_identifier=0,
                info="ok",
                job_id=job.id_,
                control_map_bytes=b"the-control-map",
                state=GENERATION_STATE.ok,
            ),
        )

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.stage == JobStage.PENDING_SAFETY_CHECK
        assert tracked.job_info is not None
        assert tracked.job_info.images_bytes == [b"the-control-map"]
        # It never entered the inference queue: the map is the deliverable, produced entirely on the lane.
        assert job not in pm._job_tracker.jobs_pending_inference

    async def test_return_control_map_fault_falls_through_in_graph(self) -> None:
        """A faulted return_control_map annotation releases the job to inference for the in-graph path."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        _add_utilities_process(pm)
        _mark_servable(pm, ["canny"])
        job = _make_controlnet_job(control_type="canny", return_control_map=True)
        assert job.id_ is not None
        await pm._job_tracker.record_popped_job(job, time_popped=time.time())
        pm._advance_pending_annotations()

        pm._on_annotation_result(
            HordeAnnotationResultMessage(
                process_id=_UTILITIES_PROCESS_ID,
                process_launch_identifier=0,
                info="boom",
                job_id=job.id_,
                control_map_bytes=None,
                state=GENERATION_STATE.faulted,
                fault_reason="RuntimeError: boom",
            ),
        )

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.stage == JobStage.PENDING_INFERENCE
        assert tracked.premade_control_map_bytes is None
        assert job in pm._job_tracker.jobs_pending_inference


class TestAntiPingPong:
    """(h) A released job is never re-parked: it dispatches in-graph exactly once-decided."""

    async def test_aged_out_job_is_not_reparked(self) -> None:
        """After an age-out release, a later scan leaves the job in inference and sends no new annotation."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        _mark_servable(pm, ["canny"])
        job = _make_controlnet_job()
        assert job.id_ is not None
        await pm._job_tracker.record_popped_job(job, time_popped=time.time())
        pm._advance_pending_annotations()
        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_ANNOTATION

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        tracked.current_stage_since = time.time() - (pm._PENDING_ANNOTATION_TIMEOUT_SECONDS + 5.0)
        pm._advance_pending_annotations()
        assert tracked.stage == JobStage.PENDING_INFERENCE
        send_count_after_release = _sent(util).call_count

        # The lane is still healthy and the control type is still servable, yet a second scan must not
        # re-park the already-decided job (the anti-ping-pong latch), and must send no further annotation.
        pm._advance_pending_annotations()

        assert tracked.stage == JobStage.PENDING_INFERENCE
        assert tracked.annotation_attempted is True
        assert _sent(util).call_count == send_count_after_release
        assert job in pm._job_tracker.jobs_pending_inference


async def _park_strip_job(
    pm: HordeWorkerProcessManager,
    *,
    images: tuple[bytes, ...] = (b"raw-image",),
) -> ImageGenerateJobPopResponse:
    """Enqueue a strip-only generation job, carry it to PENDING_STRIP with images, and return it."""
    job = make_job_pop_response(post_processing=["strip_background"])
    job_info = await pm._job_tracker.record_popped_job(job, time_popped=time.time())
    await pm._job_tracker.mark_inference_started(job)
    job_info.job_image_results = [HordeImageResult(image_bytes=image) for image in images]
    await pm._job_tracker.queue_for_strip(job_info, from_post_processing=False)
    return job


class TestStripDispatchHappyPath:
    """A pending strip is dispatched to an idle lane and its result carries the job on to safety."""

    async def test_strip_dispatched_and_result_moves_job_to_safety(self) -> None:
        """The images are sent to the lane and the stripped result becomes the job's images at safety."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        job = await _park_strip_job(pm, images=(b"raw-a", b"raw-b"))
        assert job.id_ is not None

        await pm._advance_pending_strips()

        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_STRIP
        sent = _sent(util).call_args.args[0]
        assert isinstance(sent, HordeStartStripControlMessage)
        assert sent.control_flag == HordeControlFlag.START_BACKGROUND_STRIP
        assert sent.job_id == job.id_
        assert sent.images_bytes == [b"raw-a", b"raw-b"]

        await pm._on_strip_result(
            HordeStripResultMessage(
                process_id=_UTILITIES_PROCESS_ID,
                process_launch_identifier=0,
                info="ok",
                job_id=job.id_,
                images_bytes=[b"stripped-a", b"stripped-b"],
                state=GENERATION_STATE.ok,
            ),
        )

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.stage == JobStage.PENDING_SAFETY_CHECK
        assert tracked.job_info is not None
        assert tracked.job_info.images_bytes == [b"stripped-a", b"stripped-b"]


class TestStripLaneDeath:
    """A strip whose lane dies mid-pass is a no-image fault (reissue), matching the post-processing lane."""

    async def test_lane_death_faults_strip_job_without_images(self) -> None:
        """The job moves to submit as a fault carrying no images; nothing is silently submitted un-stripped."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        job = await _park_strip_job(pm)
        assert job.id_ is not None
        await pm._advance_pending_strips()
        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_STRIP

        util.mp_process.is_alive.return_value = False
        await pm._advance_pending_strips()

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.stage == JobStage.PENDING_SUBMIT
        assert tracked.job_info is not None
        assert tracked.job_info.state == GENERATION_STATE.faulted
        assert tracked.job_info.job_image_results is None


class TestStripAgeOut:
    """A strip that outlives the bounded timeout is a no-image fault, so a job is never parked forever."""

    async def test_age_out_faults_strip_job_without_images(self) -> None:
        """A strip sitting past the timeout is faulted without images."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        _add_utilities_process(pm)
        job = await _park_strip_job(pm)
        assert job.id_ is not None

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        tracked.current_stage_since = time.time() - (pm._PENDING_STRIP_TIMEOUT_SECONDS + 5.0)
        await pm._advance_pending_strips()

        assert tracked.stage == JobStage.PENDING_SUBMIT
        assert tracked.job_info is not None
        assert tracked.job_info.state == GENERATION_STATE.faulted


class TestStripLaneDisabled:
    """With the lane disabled, the strip pass is a no-op: a default (mock-truthy) config never activates it."""

    async def test_disabled_lane_never_dispatches_strip(self) -> None:
        """A default config must not dispatch a strip even with a utilities process present in the map."""
        pm = make_testable_process_manager()  # enable_image_utilities left at its mock default
        util = _add_utilities_process(pm)
        job = await _park_strip_job(pm)
        assert job.id_ is not None

        await pm._advance_pending_strips()

        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_STRIP
        assert not _sent(util).called


class TestStripTailRouting:
    """Strip is split from the post-processing lane and applied strictly last."""

    async def test_strip_only_result_skips_the_post_processing_lane(self) -> None:
        """A strip-only job goes straight to the strip stage, never queuing for the post-processing lane."""
        job_tracker = JobTracker()
        job = make_job_pop_response(post_processing=["strip_background"])
        assert job.id_ is not None
        await job_tracker.record_popped_job(job, time_popped=time.time())
        await mark_job_in_progress_async(job_tracker, job)
        dispatcher = _make_dispatcher(
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(enable_image_utilities=True),
        )

        await dispatcher._handle_inference_result(
            HordeInferenceResultMessage(
                process_id=99,
                process_launch_identifier=0,
                info="",
                state=GENERATION_STATE.ok,
                sdk_api_job_info=job,
                job_image_results=[HordeImageResult(image_bytes=b"raw")],
            ),
        )

        assert job_tracker.get_stage(job.id_) == JobStage.PENDING_STRIP

    async def test_mixed_post_processing_goes_to_the_lane_first(self) -> None:
        """A job with an upscaler and a strip runs the upscaler on the post-processing lane before the strip."""
        job_tracker = JobTracker()
        job = make_job_pop_response(post_processing=["RealESRGAN_x4plus", "strip_background"])
        assert job.id_ is not None
        await job_tracker.record_popped_job(job, time_popped=time.time())
        await mark_job_in_progress_async(job_tracker, job)
        dispatcher = _make_dispatcher(job_tracker=job_tracker)

        await dispatcher._handle_inference_result(
            HordeInferenceResultMessage(
                process_id=99,
                process_launch_identifier=0,
                info="",
                state=GENERATION_STATE.ok,
                sdk_api_job_info=job,
                job_image_results=[HordeImageResult(image_bytes=b"raw")],
            ),
        )

        assert job_tracker.get_stage(job.id_) == JobStage.PENDING_POST_PROCESSING

    async def test_post_processing_result_with_strip_routes_to_strip_stage(self) -> None:
        """After the lane finishes the upscaler, a job that also requested strip advances to the strip stage."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        job = make_job_pop_response(post_processing=["RealESRGAN_x4plus", "strip_background"])
        assert job.id_ is not None
        job_info = await pm._job_tracker.record_popped_job(job, time_popped=time.time())
        await pm._job_tracker.queue_for_post_processing(job_info)
        await pm._job_tracker.begin_post_processing(job_info, process_id=7, process_launch_identifier=0)

        await pm._message_dispatcher._handle_post_process_result(
            HordePostProcessResultMessage(
                process_id=7,
                process_launch_identifier=0,
                info="ok",
                job_id=job.id_,
                job_image_results=[HordeImageResult(image_bytes=b"upscaled")],
                state=GENERATION_STATE.ok,
            ),
        )

        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_STRIP

    async def test_strip_requested_but_lane_disabled_faults_without_parking(self) -> None:
        """A strip job arriving while the lane is disabled is faulted (reissued), never parked in PENDING_STRIP.

        Guards the wedge where the lane flag flips off mid-run with strip work in flight: the strip stage's
        driver never runs while the lane is off, so parking would strand the job. Background removal has no
        in-graph fallback, so the only live option is the same no-image fault the post-processing lane uses.
        """
        job_tracker = JobTracker()
        job = make_job_pop_response(post_processing=["strip_background"])
        assert job.id_ is not None
        await job_tracker.record_popped_job(job, time_popped=time.time())
        await mark_job_in_progress_async(job_tracker, job)
        dispatcher = _make_dispatcher(
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(enable_image_utilities=False),
        )

        await dispatcher._handle_inference_result(
            HordeInferenceResultMessage(
                process_id=99,
                process_launch_identifier=0,
                info="",
                state=GENERATION_STATE.ok,
                sdk_api_job_info=job,
                job_image_results=[HordeImageResult(image_bytes=b"raw")],
            ),
        )

        assert job_tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT
        assert len(job_tracker.jobs_pending_strip) == 0
        tracked = job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.job_info is not None
        assert tracked.job_info.state == GENERATION_STATE.faulted
        assert tracked.job_info.job_image_results is None
