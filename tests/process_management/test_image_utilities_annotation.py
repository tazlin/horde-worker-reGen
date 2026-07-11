"""Behavioral liveness tests for the parent-side image-utilities pre-annotation flow.

The theme is that the worker keeps working no matter how the utilities lane behaves: a controlnet job is
pre-annotated off-GPU when the lane can serve it, and every failure mode (the lane dying, an annotation
faulting, an age-out, an unservable control type, the lane being disabled) releases the job back into the
normal generation flow so hordelib annotates it in-graph. A parked job is never lost and never oscillates.
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
    HordeInferenceControlMessage,
    HordeProcessState,
    HordeStartAnnotationControlMessage,
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
    track_popped_job_async,
)
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


class TestReturnControlMapExcluded:
    """(g) return_control_map jobs are excluded from pre-annotation (item 5 seam is deliberately unwired)."""

    async def test_return_control_map_job_is_not_parked(self) -> None:
        """A return_control_map job falls through to in-graph rather than being parked for the lane."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        _mark_servable(pm, ["canny"])
        job = _make_controlnet_job(control_type="canny", return_control_map=True)
        assert job.id_ is not None
        await pm._job_tracker.record_popped_job(job, time_popped=time.time())

        pm._advance_pending_annotations()

        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
        assert not _sent(util).called


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
