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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.ai_horde_api.fields import GenerationID
from horde_sdk.generation_parameters.alchemy.consts import is_strip_background_form
from loguru import logger

from horde_worker_regen.consts import EXTENDED_CONTROL_TYPES
from horde_worker_regen.process_management.ipc.messages import (
    AlchemyFormSpec,
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeAnnotationResultMessage,
    HordeAnnotatorAvailabilityMessage,
    HordeControlFlag,
    HordeDownloadAvailabilityMessage,
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
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.resources.run_metrics import DecisionKind, DecisionVerdict
from horde_worker_regen.process_management.scheduling.dispatch_affinity import (
    ANTI_STARVATION_TTL_FRACTION,
    fallback_ttl_seconds,
)
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyBucket
from horde_worker_regen.process_management.scheduling.workload_flow import routes_to_utilities_lane
from horde_worker_regen.process_management.simulation._canned_scenarios import CannedAlchemySource
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_testable_process_manager,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.ipc.test_message_dispatch import _make_dispatcher
from tests.process_management.lifecycle.test_utilities_lifecycle import _FakeAdapter
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
    ttl: int | None = None,
) -> ImageGenerateJobPopResponse:
    """Build a real controlnet generation pop response carrying an inline (base64) source image."""
    job_id = str(uuid.uuid4())
    data = {
        "id": job_id,
        "ids": [job_id],
        "ttl": ttl,
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


class TestExtendedControlnetHostileInput:
    """An extended-controlnet job that arrives while the worker withholds extended is faulted for reissue."""

    async def test_unoffered_extended_job_is_faulted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A queued extended-type job the lane cannot serve, with no annotator weights on disk, is reissued."""
        pm = make_testable_process_manager(enable_image_utilities=True, extended_controlnet=True)
        util = _add_utilities_process(pm)
        # Only a classic type is servable and the extended weights are absent, so the lane never will serve it.
        _mark_servable(pm, ["canny"])
        monkeypatch.setattr(pm, "_extended_annotators_on_disk", lambda _types: False)
        job = _make_controlnet_job(control_type="lineart")
        assert job.id_ is not None
        await pm._job_tracker.record_popped_job(job, time_popped=time.time())

        pm._fault_unready_extended_controlnet_jobs()

        assert job not in pm._job_tracker.jobs_pending_inference
        assert not _sent(util).called

    async def test_classic_job_is_never_faulted_by_the_backstop(self) -> None:
        """The extended hostile-input backstop leaves classic controlnet jobs untouched."""
        pm = make_testable_process_manager(enable_image_utilities=True, extended_controlnet=True)
        _add_utilities_process(pm)
        _mark_servable(pm, ["canny"])
        job = _make_controlnet_job(control_type="depth")
        assert job.id_ is not None
        await pm._job_tracker.record_popped_job(job, time_popped=time.time())

        pm._fault_unready_extended_controlnet_jobs()

        assert job in pm._job_tracker.jobs_pending_inference

    async def test_offered_extended_job_is_not_faulted(self) -> None:
        """When the whole extended surface is servable and opted in, an extended job is kept (to be annotated)."""
        pm = make_testable_process_manager(enable_image_utilities=True, extended_controlnet=True)
        _add_utilities_process(pm)
        _mark_servable(pm, sorted(EXTENDED_CONTROL_TYPES))
        job = _make_controlnet_job(control_type="lineart")
        assert job.id_ is not None
        await pm._job_tracker.record_popped_job(job, time_popped=time.time())

        pm._fault_unready_extended_controlnet_jobs()

        assert job in pm._job_tracker.jobs_pending_inference


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


def _alchemist_with_the_lane_down(
    monkeypatch: pytest.MonkeyPatch, **bridge_overrides: object
) -> HordeWorkerProcessManager:
    """Build an alchemist offering post-processing and annotation, with the lane enabled, provisioned and down."""
    monkeypatch.setattr(alchemy_popper, "strip_background_available", lambda: True)
    monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda _form: True)
    overrides: dict[str, object] = {
        "enable_image_utilities": True,
        "alchemist": True,
        "forms": ["post-process", "annotation"],
    }
    overrides.update(bridge_overrides)
    pm = make_testable_process_manager(**overrides)
    lifecycle = pm._process_lifecycle
    lifecycle._entry_points = ProcessEntryPoints(  # type: ignore[arg-type]
        utilities_adapter_factory=lambda *_args, **_kwargs: _FakeAdapter(),
    )
    lifecycle._ctx.Pipe.return_value = (Mock(), Mock())
    pm._download_coordinator.inference_processes_started = True
    return pm


def _lane_forms_offered(pm: HordeWorkerProcessManager) -> set[str]:
    """Return the lane-bound forms the alchemy pop would offer now, read through the coordinator's live gates."""
    coordinator = pm._alchemy_coordinator
    offered = expand_offered_forms(
        coordinator.bridge_data,
        utilities_lane_healthy=coordinator._utilities_lane_healthy(),
        annotation_types=coordinator._annotation_types(),
    )
    return {form for form in offered if routes_to_utilities_lane(form)}


_STRIP_FORM = "strip_background"
_ANNOTATION_FORM = "annotation"


class TestStripPopGating:
    """(f) The lane-bound alchemy offers follow whether the lane serves or will serve."""

    def test_strip_offered_only_when_lane_healthy(self, monkeypatch) -> None:  # noqa: ANN001
        """The pure offer function keeps strip_background exactly when its lane reading allows it.

        The reading the coordinator passes now means "the lane serves or will serve"; the rows below drive it.
        """
        monkeypatch.setattr(alchemy_popper, "strip_background_available", lambda: True)
        bridge_data = make_mock_bridge_data(forms=["post-process"])

        offered_healthy = expand_offered_forms(bridge_data, utilities_lane_healthy=True)
        offered_unhealthy = expand_offered_forms(bridge_data, utilities_lane_healthy=False)

        assert any(is_strip_background_form(form) for form in offered_healthy)
        assert not any(is_strip_background_form(form) for form in offered_unhealthy)

    def test_strip_is_offered_while_the_lane_is_down_by_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A lane that is down because nothing needs it will start for a popped strip, so the offer stands."""
        pm = _alchemist_with_the_lane_down(monkeypatch)
        assert pm._process_map.num_utilities_processes() == 0
        assert pm._process_lifecycle.utilities_lane_will_serve()

        assert _STRIP_FORM in _lane_forms_offered(pm)

    def test_strip_is_offered_while_the_lane_is_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A loaded lane serves the form as before."""
        pm = _alchemist_with_the_lane_down(monkeypatch)
        _add_utilities_process(pm)

        assert _STRIP_FORM in _lane_forms_offered(pm)

    @pytest.mark.parametrize("lane_state", ["paused", "crashed", "start-refused", "disabled"])
    def test_strip_is_withheld_while_the_lane_will_not_serve(
        self,
        monkeypatch: pytest.MonkeyPatch,
        lane_state: str,
    ) -> None:
        """A paused, crashed, refused or disabled lane is not coming up on demand, so the offer is withheld."""
        pm = _alchemist_with_the_lane_down(monkeypatch)
        lifecycle = pm._process_lifecycle
        match lane_state:
            case "paused":
                lifecycle._utilities_gpu_paused = True
                lifecycle._utilities_pause_owner = PauseOwner.RECLAIM_LADDER
            case "crashed":
                lifecycle._initiate_utilities_replacement()
            case "start-refused":
                lifecycle._utilities_demand_start_attempted = True
            case "disabled":
                pm.bridge_data.enable_image_utilities = False
        assert not lifecycle.utilities_lane_will_serve()

        assert _STRIP_FORM not in _lane_forms_offered(pm)

    def test_annotation_is_not_offered_before_the_lane_has_reported_a_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no servable set learned this session the offer cannot carry any types, so it is withheld."""
        pm = _alchemist_with_the_lane_down(monkeypatch)

        assert _ANNOTATION_FORM not in _lane_forms_offered(pm)

    def test_annotation_is_offered_from_the_set_kept_from_the_lanes_last_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A lane that reported its set and then left by policy leaves that set for the offer to carry."""
        pm = _alchemist_with_the_lane_down(monkeypatch)
        _add_utilities_process(pm)
        _mark_servable(pm, ["canny", "depth"])
        del pm._process_map[_UTILITIES_PROCESS_ID]
        assert pm._servable_control_types() == frozenset(), "the live set is pruned with the lane"

        assert _ANNOTATION_FORM in _lane_forms_offered(pm)
        assert pm._alchemy_coordinator._annotation_types() == frozenset({"canny", "depth"})

    def test_a_lane_crash_drops_the_kept_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After a crash replacement the kept set is not offered, even once the lane will serve again."""
        pm = _alchemist_with_the_lane_down(monkeypatch)
        lifecycle = pm._process_lifecycle
        _add_utilities_process(pm)
        _mark_servable(pm, ["canny"])
        del pm._process_map[_UTILITIES_PROCESS_ID]
        lifecycle._initiate_utilities_replacement()
        lifecycle._utilities_processes_should_be_replaced = False
        assert lifecycle.utilities_lane_will_serve()

        assert _ANNOTATION_FORM not in _lane_forms_offered(pm)

    def test_disabling_the_lane_drops_the_kept_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A lane switched off and on again does not offer the set from before it was switched off."""
        pm = _alchemist_with_the_lane_down(monkeypatch)
        _add_utilities_process(pm)
        _mark_servable(pm, ["canny"])
        del pm._process_map[_UTILITIES_PROCESS_ID]
        pm.bridge_data.enable_image_utilities = False
        pm._reconcile_utilities_demand()
        pm.bridge_data.enable_image_utilities = True

        assert _ANNOTATION_FORM not in _lane_forms_offered(pm)


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
        assert pm._job_tracker.total_num_completed_jobs == 1

        # A duplicate child result cannot count the same terminal generation-stage outcome twice.
        pm._on_annotation_result(
            HordeAnnotationResultMessage(
                process_id=_UTILITIES_PROCESS_ID,
                process_launch_identifier=0,
                info="duplicate",
                job_id=job.id_,
                control_map_bytes=b"the-control-map",
                state=GENERATION_STATE.ok,
            ),
        )
        assert pm._job_tracker.total_num_completed_jobs == 1

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
    """A strip outliving its lanes ends as a no-image fault (reissue), matching the post-processing lane."""

    async def test_repeated_lane_death_faults_strip_job_without_images(self) -> None:
        """Once the job has spent every lane it is allowed, it moves to submit as a fault carrying no images.

        The images are re-offered to a replacement lane first, so the fault lands only when the pass has
        genuinely run out of ways to happen; nothing is ever silently submitted un-stripped.
        """
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        job = await _park_strip_job(pm)
        assert job.id_ is not None
        await pm._advance_pending_strips()
        assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_STRIP

        # Each pass kills the lane the strip is on and stands a replacement up in its place.
        for _lane_death in range(pm._MAX_STRIP_DISPATCH_ATTEMPTS):
            util.mp_process.is_alive.return_value = False
            await pm._advance_pending_strips()
            del pm._process_map[_UTILITIES_PROCESS_ID]
            util = _add_utilities_process(pm)
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
        """A strip whose lane never frees is faulted without images once its stay leaves no room for a pass."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        _add_utilities_process(pm, state=HordeProcessState.ALCHEMY_STARTING)
        job = await _park_strip_job(pm)
        assert job.id_ is not None

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        tracked.current_stage_since = time.time() - pm._strip_stage_cap_seconds()
        await pm._advance_pending_strips()

        assert tracked.stage == JobStage.PENDING_SUBMIT
        assert tracked.job_info is not None
        assert tracked.job_info.state == GENERATION_STATE.faulted

    async def test_strip_with_a_free_lane_is_dispatched_whatever_its_stage_age(self) -> None:
        """A strip the lane can take this tick is dispatched, never faulted in the same tick."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        job = await _park_strip_job(pm)
        assert job.id_ is not None

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        tracked.current_stage_since = time.time() - pm._strip_stage_cap_seconds()
        await pm._advance_pending_strips()

        assert tracked.stage == JobStage.PENDING_STRIP
        assert tracked.strip_process_id == util.process_id


class TestStripBudgets:
    """A strip is judged on the time it actually had: its own run, or its wait behind the serial lane."""

    async def test_dispatched_strip_is_budgeted_from_dispatch_not_stage_entry(self) -> None:
        """A strip dispatched moments ago survives a long stay in the stage: the wait is not its run time."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        _add_utilities_process(pm)
        job = await _park_strip_job(pm)
        assert job.id_ is not None
        await pm._advance_pending_strips()

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.strip_dispatched_at is not None
        tracked.current_stage_since = time.time() - (pm._PENDING_STRIP_TIMEOUT_SECONDS * 4)
        tracked.strip_dispatched_at = time.time() - 5.0

        await pm._age_out_pending_strips()

        assert tracked.stage == JobStage.PENDING_STRIP

    async def test_dispatched_strip_ages_out_on_its_own_elapsed_run(self) -> None:
        """Once the dispatched strip itself overruns the budget, the job is faulted without images."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        _add_utilities_process(pm)
        job = await _park_strip_job(pm)
        assert job.id_ is not None
        await pm._advance_pending_strips()

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        tracked.strip_dispatched_at = time.time() - (pm._PENDING_STRIP_TIMEOUT_SECONDS + 5.0)

        await pm._age_out_pending_strips()

        assert tracked.stage == JobStage.PENDING_SUBMIT
        assert tracked.job_info is not None
        assert tracked.job_info.state == GENERATION_STATE.faulted

    async def test_queued_strip_is_not_charged_for_the_strip_ahead_of_it(self) -> None:
        """A job waiting behind a running strip survives any wait short of the stage cap."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        _add_utilities_process(pm)
        first = await _park_strip_job(pm)
        second = await _park_strip_job(pm)
        assert first.id_ is not None
        assert second.id_ is not None
        await pm._advance_pending_strips()

        head = pm._job_tracker.get_tracked_job(first.id_)
        waiter = pm._job_tracker.get_tracked_job(second.id_)
        assert head is not None
        assert waiter is not None
        assert head.strip_dispatched_at is not None
        assert waiter.strip_dispatched_at is None, "the single serial lane takes one strip at a time"

        queue_wait_limit = pm._strip_stage_cap_seconds() - pm._PENDING_STRIP_TIMEOUT_SECONDS
        waiter.current_stage_since = time.time() - (queue_wait_limit - 5.0)
        await pm._age_out_pending_strips()
        assert waiter.stage == JobStage.PENDING_STRIP
        assert waiter.strip_head_since is None

        waiter.current_stage_since = time.time() - (queue_wait_limit + 5.0)
        await pm._age_out_pending_strips()
        assert waiter.stage == JobStage.PENDING_SUBMIT


def _strip_result(job_id: GenerationID, *, ok: bool = True) -> HordeStripResultMessage:
    """Build the utilities lane's result for one background strip, stripped images or a fault."""
    return HordeStripResultMessage(
        process_id=_UTILITIES_PROCESS_ID,
        process_launch_identifier=0,
        info="ok" if ok else "failed",
        job_id=job_id,
        images_bytes=[b"stripped"] if ok else [],
        state=GENERATION_STATE.ok if ok else GENERATION_STATE.faulted,
        fault_reason=None if ok else "background removal failed",
    )


@contextmanager
def _captured_messages() -> Iterator[list[str]]:
    """Collect every loguru message emitted inside the block."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="TRACE")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


class TestStripHeadOfQueueClock:
    """A queued strip is clocked from the moment it reaches the head of the queue, never from stage entry."""

    async def test_waiter_is_dispatched_in_the_tick_its_predecessor_frees_the_lane(self) -> None:
        """A strip that waited longer than one execution budget behind a long pass is dispatched, not faulted.

        The waiter's time behind the running strip is the lane's time, so the tick the lane frees for it
        hands it the lane.
        """
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        first = await _park_strip_job(pm)
        second = await _park_strip_job(pm)
        assert first.id_ is not None
        assert second.id_ is not None
        await pm._advance_pending_strips()
        util.last_process_state = HordeProcessState.ALCHEMY_STARTING

        running = pm._job_tracker.get_tracked_job(first.id_)
        waiter = pm._job_tracker.get_tracked_job(second.id_)
        assert running is not None
        assert waiter is not None
        assert running.strip_dispatched_at is not None
        now = time.time()
        running.current_stage_since = now - 175.0
        running.strip_dispatched_at = now - 170.0
        waiter.current_stage_since = now - (pm._PENDING_STRIP_TIMEOUT_SECONDS + 10.0)
        await pm._advance_pending_strips()
        assert waiter.stage == JobStage.PENDING_STRIP
        assert waiter.strip_dispatched_at is None

        await pm._on_strip_result(_strip_result(first.id_))
        util.last_process_state = HordeProcessState.WAITING_FOR_JOB
        await pm._advance_pending_strips()

        assert waiter.stage == JobStage.PENDING_STRIP
        assert waiter.strip_process_id == util.process_id
        sent = _sent(util).call_args.args[0]
        assert isinstance(sent, HordeStartStripControlMessage)
        assert sent.job_id == second.id_

    async def test_waiter_behind_a_running_strip_is_not_clocked(self) -> None:
        """A strip with a dispatched strip ahead of it has no head clock running, short of the stage cap."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        await _park_strip_job(pm)
        second = await _park_strip_job(pm)
        assert second.id_ is not None
        await pm._advance_pending_strips()
        util.last_process_state = HordeProcessState.ALCHEMY_STARTING

        waiter = pm._job_tracker.get_tracked_job(second.id_)
        assert waiter is not None
        queue_wait_limit = pm._strip_stage_cap_seconds() - pm._PENDING_STRIP_TIMEOUT_SECONDS
        assert queue_wait_limit - 5.0 > pm._PENDING_STRIP_TIMEOUT_SECONDS
        waiter.current_stage_since = time.time() - (queue_wait_limit - 5.0)
        await pm._advance_pending_strips()

        assert waiter.stage == JobStage.PENDING_STRIP
        assert waiter.strip_head_since is None

    async def test_head_waiter_with_no_free_lane_ages_out_after_one_execution_budget(self) -> None:
        """A strip at the head whose lane never frees is faulted once it has waited one execution budget there."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        _add_utilities_process(pm, state=HordeProcessState.ALCHEMY_STARTING)
        job = await _park_strip_job(pm)
        assert job.id_ is not None
        await pm._advance_pending_strips()

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.strip_head_since is not None
        tracked.strip_head_since = time.time() - (pm._PENDING_STRIP_TIMEOUT_SECONDS - 5.0)
        await pm._advance_pending_strips()
        assert tracked.stage == JobStage.PENDING_STRIP

        tracked.strip_head_since = time.time() - (pm._PENDING_STRIP_TIMEOUT_SECONDS + 5.0)
        with _captured_messages() as messages:
            await pm._advance_pending_strips()

        assert tracked.stage == JobStage.PENDING_SUBMIT
        assert tracked.job_info is not None
        assert tracked.job_info.state == GENERATION_STATE.faulted
        assert tracked.job_info.job_image_results is None
        assert any("at the head of the strip queue" in message for message in messages)

    async def test_waiter_past_the_stage_cap_is_faulted_with_its_detail(self) -> None:
        """A strip still queued when no pass could finish inside the stage cap is faulted, naming the cap."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        await _park_strip_job(pm)
        second = await _park_strip_job(pm)
        assert second.id_ is not None
        await pm._advance_pending_strips()
        util.last_process_state = HordeProcessState.ALCHEMY_STARTING

        stage_cap = pm._strip_stage_cap_seconds()
        assert stage_cap == pm._PENDING_STRIP_TIMEOUT_SECONDS * (
            pm.bridge_data.queue_size + pm.bridge_data.max_threads + 1
        )
        waiter = pm._job_tracker.get_tracked_job(second.id_)
        assert waiter is not None
        queue_wait_limit = stage_cap - pm._PENDING_STRIP_TIMEOUT_SECONDS
        waiter.current_stage_since = time.time() - (queue_wait_limit - 5.0)
        await pm._advance_pending_strips()
        assert waiter.stage == JobStage.PENDING_STRIP

        waiter.current_stage_since = time.time() - (queue_wait_limit + 5.0)
        with _captured_messages() as messages:
            await pm._advance_pending_strips()

        assert waiter.stage == JobStage.PENDING_SUBMIT
        assert waiter.job_info is not None
        assert waiter.job_info.state == GENERATION_STATE.faulted
        assert any(
            str(second.id_)[:8] in message and f"{stage_cap:.0f}s strip stage cap" in message for message in messages
        )


class TestSlowStripLaneEnsemble:
    """A strip lane slower than strip arrivals faults only the strips it cannot reach, and nothing else."""

    async def test_slow_strips_keep_the_post_processing_offer_and_breaker_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Seven strip jobs against a lane serving each in most of an execution budget.

        The post-processing lane serves none of these jobs, so the offer stays advertised and the breaker
        stays closed, while the strips that cannot be reached inside the stage cap are faulted on it.
        """
        clock = [100_000.0]
        monkeypatch.setattr(time, "time", lambda: clock[0])
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        service_seconds = pm._PENDING_STRIP_TIMEOUT_SECONDS * 0.9
        arrival_interval_seconds = 34.0
        tick_seconds = 1.0

        jobs = [make_job_pop_response(post_processing=["strip_background"]) for _ in range(7)]
        job_infos = [await pm._job_tracker.record_popped_job(job, time_popped=clock[0]) for job in jobs]
        arrivals = [clock[0] + index * arrival_interval_seconds for index in range(len(jobs))]
        arrived = 0
        in_flight: tuple[GenerationID, float] | None = None
        stripped: set[GenerationID] = set()

        with _captured_messages() as messages:
            while arrived < len(jobs) or pm._job_tracker.jobs_pending_strip:
                while arrived < len(jobs) and clock[0] >= arrivals[arrived]:
                    await pm._job_tracker.mark_inference_started(jobs[arrived])
                    job_infos[arrived].job_image_results = [HordeImageResult(image_bytes=b"raw")]
                    await pm._job_tracker.queue_for_strip(job_infos[arrived], from_post_processing=False)
                    arrived += 1
                if in_flight is not None and clock[0] >= in_flight[1]:
                    await pm._on_strip_result(_strip_result(in_flight[0]))
                    stripped.add(in_flight[0])
                    util.last_process_state = HordeProcessState.WAITING_FOR_JOB
                    in_flight = None
                await pm._advance_pending_strips()
                for tracked in pm._job_tracker.jobs_pending_strip:
                    if tracked.strip_dispatched_at is not None and (
                        in_flight is None or in_flight[0] != tracked.job_id
                    ):
                        in_flight = (tracked.job_id, clock[0] + service_seconds)
                        util.last_process_state = HordeProcessState.ALCHEMY_STARTING

                pm._apply_post_processing_fault_breaker()
                assert pm._state.post_processing_disabled_by_breaker is False
                assert pm._job_popper._should_withhold_post_processing_offer(pm.bridge_data) is False
                assert pm._job_popper._post_processing_offer_withheld_reason() is None
                clock[0] += tick_seconds

        job_ids = [job.id_ for job in jobs if job.id_ is not None]
        assert len(job_ids) == len(jobs)
        faulted = {job_id for job_id in job_ids if job_id not in stripped}
        assert stripped, "the lane serves the strips it reaches"
        assert faulted, "a lane this slow cannot reach every strip inside the stage cap"
        assert pm._job_tracker.count_recent_post_processing_faults(1800) == 0
        stage_cap_faults = [message for message in messages if "strip stage cap" in message]
        assert len(stage_cap_faults) == len(faulted)
        assert not [message for message in messages if "at the head of the strip queue" in message]
        for job_id in faulted:
            tracked = pm._job_tracker.get_tracked_job(job_id)
            assert tracked is not None
            assert tracked.job_info is not None
            assert tracked.job_info.state == GENERATION_STATE.faulted


class TestStripLaneReplacement:
    """A strip whose lane goes away is re-run on the replacement, up to a bounded number of attempts."""

    async def test_lane_death_requeues_the_strip_for_the_replacement_lane(self) -> None:
        """The job keeps its images and its place in the strip stage, re-armed for dispatch."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        util = _add_utilities_process(pm)
        job = await _park_strip_job(pm)
        assert job.id_ is not None
        await pm._advance_pending_strips()

        util.mp_process.is_alive.return_value = False
        requeued_at = time.time()
        await pm._advance_pending_strips()

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.stage == JobStage.PENDING_STRIP
        assert tracked.strip_process_id is None
        assert tracked.strip_dispatched_at is None
        # Back at the head with no lane alive, so its head clock restarts at the requeue.
        assert tracked.strip_head_since is not None
        assert tracked.strip_head_since >= requeued_at
        assert tracked.job_info is not None
        assert tracked.job_info.images_bytes == [b"raw-image"]

        # The replacement lane picks the re-queued strip straight up.
        del pm._process_map[_UTILITIES_PROCESS_ID]
        replacement = _add_utilities_process(pm)
        await pm._advance_pending_strips()

        assert tracked.strip_process_id == replacement.process_id
        sent = _sent(replacement).call_args.args[0]
        assert isinstance(sent, HordeStartStripControlMessage)
        assert sent.job_id == job.id_


class TestLateStripResult:
    """Finished strip work is used wherever it still can be, rather than thrown away behind a write-off."""

    async def test_late_success_reopens_a_written_off_job(self) -> None:
        """A strip that completes after the age-out carries the job on to safety with its stripped images."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        _add_utilities_process(pm)
        job = await _park_strip_job(pm)
        assert job.id_ is not None
        await pm._advance_pending_strips()

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        tracked.strip_dispatched_at = time.time() - (pm._PENDING_STRIP_TIMEOUT_SECONDS + 5.0)
        await pm._age_out_pending_strips()
        assert tracked.stage == JobStage.PENDING_SUBMIT

        await pm._on_strip_result(
            HordeStripResultMessage(
                process_id=_UTILITIES_PROCESS_ID,
                process_launch_identifier=0,
                info="ok",
                job_id=job.id_,
                images_bytes=[b"stripped-late"],
                state=GENERATION_STATE.ok,
            ),
        )

        assert tracked.stage == JobStage.PENDING_SAFETY_CHECK
        assert tracked.job_info is not None
        assert tracked.job_info.state == GENERATION_STATE.ok
        assert tracked.job_info.images_bytes == [b"stripped-late"]
        assert await pm._job_tracker.get_faults_for_job(job.id_) == []

    async def test_late_success_is_discarded_once_the_submit_is_under_way(self) -> None:
        """A job whose outcome is already being reported keeps that outcome; the images cannot be used."""
        pm = make_testable_process_manager(enable_image_utilities=True)
        _add_utilities_process(pm)
        job = await _park_strip_job(pm)
        assert job.id_ is not None
        await pm._advance_pending_strips()

        tracked = pm._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        tracked.strip_dispatched_at = time.time() - (pm._PENDING_STRIP_TIMEOUT_SECONDS + 5.0)
        await pm._age_out_pending_strips()
        assert tracked.job_info is not None
        pm._job_tracker.note_submit_in_flight(tracked.job_info)

        await pm._on_strip_result(
            HordeStripResultMessage(
                process_id=_UTILITIES_PROCESS_ID,
                process_launch_identifier=0,
                info="ok",
                job_id=job.id_,
                images_bytes=[b"stripped-too-late"],
                state=GENERATION_STATE.ok,
            ),
        )

        assert tracked.stage == JobStage.PENDING_SUBMIT
        assert tracked.job_info.state == GENERATION_STATE.faulted
        assert tracked.job_info.job_image_results is None


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


class TestUtilitiesLaneOnDemandEnsemble:
    """The lane comes up for strip work, stays down once released with nothing waiting, and returns for the next."""

    async def test_strip_bursts_start_the_lane_and_a_ladder_restore_without_strips_leaves_it_down(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A strip burst starts the lane once, a ladder restore with no strip waiting leaves it down.

        A later strip job starts it again. Read through lane starts and the decisions recorded, never the demand
        rule.
        """
        pm = make_testable_process_manager(enable_image_utilities=True)
        lifecycle = pm._process_lifecycle
        constructed: list[_FakeAdapter] = []

        def _factory(*_args: object, **_kwargs: object) -> _FakeAdapter:
            adapter = _FakeAdapter()
            constructed.append(adapter)
            return adapter

        lifecycle._entry_points = ProcessEntryPoints(utilities_adapter_factory=_factory)  # type: ignore[arg-type]
        lifecycle._ctx.Pipe.return_value = (Mock(), Mock())
        pm._download_coordinator.inference_processes_started = True
        decisions: list[tuple[DecisionVerdict, str]] = []
        record_decision = pm._run_metrics.record_decision

        def _spy(**kwargs: object) -> object:
            if (
                kwargs["decision_kind"] is DecisionKind.IMAGE_UTILITIES_ROUTING
                and kwargs["subject"] == "utilities_lane"
            ):
                decisions.append((kwargs["verdict"], kwargs["reason"]))  # type: ignore[arg-type]
            return record_decision(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(pm._run_metrics, "record_decision", _spy)

        def _lane() -> HordeProcessInfo | None:
            lanes = [info for info in pm._process_map.values() if info.process_type is HordeProcessType.UTILITIES]
            return lanes[0] if lanes else None

        def _tick() -> None:
            lifecycle._replace_all_utilities_process()
            pm._reconcile_utilities_demand()

        for _ in range(3):
            _tick()
        assert constructed == [], "nothing needs the lane, so nothing starts it"

        jobs = [make_job_pop_response(post_processing=["strip_background"]) for _ in range(3)]
        job_infos = [await pm._job_tracker.record_popped_job(job, time_popped=time.time()) for job in jobs]
        for _ in range(3):
            _tick()
        assert len(constructed) == 1, "a burst of strip jobs starts the lane once"
        lane = _lane()
        assert lane is not None
        lane.last_process_state = HordeProcessState.WAITING_FOR_JOB

        for job, job_info in zip(jobs, job_infos, strict=True):
            await pm._job_tracker.mark_inference_started(job)
            job_info.job_image_results = [HordeImageResult(image_bytes=b"raw")]
            await pm._job_tracker.queue_for_strip(job_info, from_post_processing=False)
        while pm._job_tracker.jobs_pending_strip:
            await pm._advance_pending_strips()
            for tracked in pm._job_tracker.jobs_pending_strip:
                if tracked.strip_dispatched_at is not None:
                    await pm._on_strip_result(_strip_result(tracked.job_id))
                    lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
            _tick()
        assert len(constructed) == 1

        assert pm._inference_scheduler.pause_utilities_lane(None) is True
        for _ in range(4):
            _tick()
        assert _lane() is None
        with _captured_messages() as messages:
            assert pm._inference_scheduler.restore_utilities_lane(None) is True
            for _ in range(3):
                _tick()
        assert _lane() is None, "with no strip waiting the restore leaves the lane down"
        assert len(constructed) == 1
        assert "Reclaim ladder: leaving the image-utilities lane down until a job needs it." in messages

        later = make_job_pop_response(post_processing=["strip_background"])
        await pm._job_tracker.record_popped_job(later, time_popped=time.time())
        _tick()
        assert len(constructed) == 2, "a later strip job starts the lane again"
        assert _lane() is not None

        started = [reason for verdict, reason in decisions if verdict is DecisionVerdict.ADMIT]
        assert started == ["strip_only_job", "strip_only_job"]
        assert (DecisionVerdict.WITHHOLD, "no_demand") in decisions


class _AlchemyLaneRig:
    """An alchemist offering post-processing with the lane down, driven tick by tick through the real coordinator.

    A tick runs the alchemy dispatch, the lane's replacement machine and the demand driver. The pop goes through
    the coordinator's pop policy with a canned source, and only the VRAM and RAM headroom readings are stubbed.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, device_free_mb: float = 24000.0) -> None:
        self.pm = _alchemist_with_the_lane_down(
            monkeypatch,
            forms=["post-process"],
            queue_size=1,
            alchemy_max_concurrency=1,
            alchemy_allow_concurrent=True,
            device_free_mb=device_free_mb,
        )
        self.coordinator = self.pm._alchemy_coordinator
        self.constructed: list[_FakeAdapter] = []

        def _factory(*_args: object, **_kwargs: object) -> _FakeAdapter:
            adapter = _FakeAdapter()
            self.constructed.append(adapter)
            return adapter

        self.pm._process_lifecycle._entry_points = ProcessEntryPoints(  # type: ignore[arg-type]
            utilities_adapter_factory=_factory,
        )
        monkeypatch.setattr(self.coordinator, "_has_vram_headroom", lambda: True)
        monkeypatch.setattr(self.coordinator, "_has_ram_headroom", lambda: True)

    def lane(self) -> HordeProcessInfo | None:
        lanes = [info for info in self.pm._process_map.values() if info.process_type is HordeProcessType.UTILITIES]
        return lanes[0] if lanes else None

    def tick(self) -> None:
        self.coordinator.dispatch_pending_forms()
        self.pm._process_lifecycle._replace_all_utilities_process()
        self.pm._reconcile_utilities_demand()

    def pop_strip_form(self) -> AlchemyFormSpec:
        spec = AlchemyFormSpec(form_id=str(uuid.uuid4()), form=_STRIP_FORM, source_image_bytes=b"source")
        self.coordinator.set_canned_alchemy_source(CannedAlchemySource([spec]))
        self.coordinator._canned_alchemy_pop()
        return spec


class TestAlchemyStripWithTheLaneDown:
    """A strip form offered and popped while the lane is down by policy starts the lane and waits for it."""

    async def test_a_popped_strip_form_starts_the_lane_and_is_served(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The offer stands with the lane down, the pop starts it, the form waits and goes to the loaded lane."""
        rig = _AlchemyLaneRig(monkeypatch)
        for _ in range(3):
            rig.tick()
        assert rig.constructed == [], "an offer alone does not start the lane"
        assert _STRIP_FORM in _lane_forms_offered(rig.pm)

        spec = rig.pop_strip_form()
        assert [pending.form_id for pending in rig.coordinator._pending_forms] == [spec.form_id]

        rig.tick()
        assert len(rig.constructed) == 1, "the popped form starts the lane"
        lane = rig.lane()
        assert lane is not None
        for _ in range(3):
            rig.tick()
        assert [pending.form_id for pending in rig.coordinator._pending_forms] == [spec.form_id], "the form waits"
        assert len(rig.coordinator._pending_submits) == 0
        assert not _sent(lane).called

        lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
        rig.tick()
        assert len(rig.coordinator._pending_forms) == 0
        sent = _sent(lane).call_args.args[0]
        assert isinstance(sent, HordeAlchemyControlMessage)
        assert sent.control_flag is HordeControlFlag.START_ALCHEMY
        assert sent.form.form_id == spec.form_id

        rig.coordinator.on_alchemy_result(
            HordeAlchemyResultMessage(
                process_id=lane.process_id,
                process_launch_identifier=lane.process_launch_identifier,
                info="stripped",
                form_id=spec.form_id,
                form=spec.form,
                state=GENERATION_STATE.ok,
            ),
        )
        assert [submit.result_message.state for submit in rig.coordinator._pending_submits] == [GENERATION_STATE.ok]
        assert len(rig.constructed) == 1

    async def test_a_lane_start_refused_for_vram_faults_the_popped_form(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A lane whose start is deferred for VRAM is not coming up, so the form goes back to the horde."""
        rig = _AlchemyLaneRig(monkeypatch, device_free_mb=100.0)
        assert _STRIP_FORM in _lane_forms_offered(rig.pm)
        spec = rig.pop_strip_form()

        for _ in range(3):
            rig.tick()

        assert rig.constructed == []
        assert rig.pm._process_lifecycle.has_pending_gpu_starts()
        assert len(rig.coordinator._pending_forms) == 0
        faulted = [submit.result_message for submit in rig.coordinator._pending_submits]
        assert [(message.form_id, message.state) for message in faulted] == [(spec.form_id, GENERATION_STATE.faulted)]
        assert _STRIP_FORM not in _lane_forms_offered(rig.pm), "the offer is withheld while the start is deferred"

    def test_an_annotation_alchemist_starts_the_lane_once_to_learn_its_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no set learned the lane starts, and once it has reported, annotation is offered with it down."""
        rig = _AlchemyLaneRig(monkeypatch)
        rig.pm.bridge_data.forms = ["annotation"]
        assert _ANNOTATION_FORM not in _lane_forms_offered(rig.pm)

        rig.tick()
        assert len(rig.constructed) == 1, "the annotation offer needs a set, so the lane starts to report one"
        lane = rig.lane()
        assert lane is not None
        lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
        rig.pm._on_annotator_availability(
            HordeAnnotatorAvailabilityMessage(
                process_id=lane.process_id,
                process_launch_identifier=lane.process_launch_identifier,
                info="availability",
                servable_control_types=["canny"],
            ),
        )
        assert rig.pm._process_lifecycle.utilities_lane_demand() is None, "the rule lapses at the first report"

        assert rig.pm._inference_scheduler.pause_utilities_lane(None) is True
        for _ in range(4):
            rig.tick()
        assert rig.lane() is None
        assert rig.pm._inference_scheduler.restore_utilities_lane(None) is True
        for _ in range(3):
            rig.tick()
        assert rig.lane() is None, "with nothing waiting the restore leaves the lane down"
        assert len(rig.constructed) == 1

        assert _ANNOTATION_FORM in _lane_forms_offered(rig.pm)
        assert rig.coordinator._annotation_types() == frozenset({"canny"})


class _ExtendedLaneRig:
    """A manager with the lane enabled, extended ControlNet opted in and a fake lane adapter, driven tick by tick.

    A tick runs the control loop's lane steps in their loop order: the extended-ControlNet pass, the annotation
    pass, the lane's replacement machine and the demand driver.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, device_free_mb: float = 24000.0) -> None:
        self.pm = make_testable_process_manager(
            enable_image_utilities=True,
            extended_controlnet=True,
            device_free_mb=device_free_mb,
        )
        self.constructed: list[_FakeAdapter] = []
        self.faults: list[str | None] = []
        self.dispatch_decisions: list[tuple[str, DecisionVerdict, str]] = []
        lifecycle = self.pm._process_lifecycle

        def _factory(*_args: object, **_kwargs: object) -> _FakeAdapter:
            adapter = _FakeAdapter()
            self.constructed.append(adapter)
            return adapter

        lifecycle._entry_points = ProcessEntryPoints(utilities_adapter_factory=_factory)  # type: ignore[arg-type]
        lifecycle._ctx.Pipe.return_value = (Mock(), Mock())
        self.pm._download_coordinator.inference_processes_started = True
        # The annotator weights for every extended type are on disk, so a lane that comes up can serve them.
        monkeypatch.setattr(self.pm, "_extended_annotators_on_disk", lambda _types: True)

        fault_now = self.pm._job_tracker.handle_job_fault_now

        def _spy_fault(*args: object, **kwargs: object) -> object:
            self.faults.append(kwargs.get("fault_reason"))  # type: ignore[arg-type]
            return fault_now(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(self.pm._job_tracker, "handle_job_fault_now", _spy_fault)

        sink = self.pm._inference_scheduler._decision_sink
        assert sink is not None

        def _spy_decision(**kwargs: object) -> object:
            if kwargs["decision_kind"] is DecisionKind.INFERENCE_DISPATCH:
                self.dispatch_decisions.append(
                    (str(kwargs["subject"]), kwargs["verdict"], str(kwargs["reason"])),  # type: ignore[arg-type]
                )
            return sink(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(self.pm._inference_scheduler, "_decision_sink", _spy_decision)

    def lane(self) -> HordeProcessInfo | None:
        lanes = [info for info in self.pm._process_map.values() if info.process_type is HordeProcessType.UTILITIES]
        return lanes[0] if lanes else None

    def tick(self) -> None:
        self.pm._fault_unready_extended_controlnet_jobs()
        self.pm._advance_pending_annotations()
        self.pm._process_lifecycle._replace_all_utilities_process()
        self.pm._reconcile_utilities_demand()

    def report_servable(self, control_types: list[str]) -> None:
        lane = self.lane()
        assert lane is not None
        self.pm._on_annotator_availability(
            HordeAnnotatorAvailabilityMessage(
                process_id=lane.process_id,
                process_launch_identifier=0,
                info="availability",
                servable_control_types=control_types,
            ),
        )


class TestExtendedControlnetWaitsForTheLane:
    """An extended-ControlNet job popped with the lane down starts it and waits for its control type."""

    async def test_a_job_popped_with_the_lane_down_is_held_until_its_type_is_servable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The job starts the lane, is held until the lane reports its set, then goes to annotation."""
        rig = _ExtendedLaneRig(monkeypatch)
        for _ in range(3):
            rig.tick()
        assert rig.constructed == [], "extended ControlNet alone does not keep the lane up"

        job = _make_controlnet_job(control_type="lineart", ttl=600)
        assert job.id_ is not None
        await rig.pm._job_tracker.record_popped_job(job, time_popped=time.time())

        rig.tick()
        assert job in rig.pm._job_tracker.jobs_pending_inference, "the job waits for the lane"
        assert len(rig.constructed) == 1, "the pending job starts the lane"
        lane = rig.lane()
        assert lane is not None
        assert rig.pm._inference_scheduler.job_awaits_utilities_lane(job)

        lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
        # The held job takes no slot by design, so the deadlock detector does not read the idle processes as a wedge.
        rig.pm._state.last_job_pop_time = time.time() - 60
        idle_id = max(rig.pm._process_map) + 1
        rig.pm._process_map[idle_id] = make_mock_process_info(
            idle_id,
            model_name="stable_diffusion",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        for _ in range(3):
            rig.pm.detect_deadlock()
        snapshot = rig.pm._message_dispatcher.get_deadlock_snapshot()
        assert snapshot.in_queue_deadlock is False
        assert snapshot.in_deadlock is False
        for _ in range(3):
            rig.tick()
        assert rig.pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
        assert rig.pm._inference_scheduler.job_awaits_utilities_lane(job)
        assert not _sent(lane).called

        rig.report_servable(["canny", "lineart"])
        rig.tick()
        assert not rig.pm._inference_scheduler.job_awaits_utilities_lane(job)
        assert rig.pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_ANNOTATION
        sent = _sent(lane).call_args.args[0]
        assert isinstance(sent, HordeStartAnnotationControlMessage)
        assert sent.control_type == "lineart"
        assert rig.faults == []
        holds = [record for record in rig.dispatch_decisions if record[1] is DecisionVerdict.WITHHOLD]
        assert holds == [(str(job.id_), DecisionVerdict.WITHHOLD, "awaiting_utilities_lane")], "recorded once"

    async def test_a_lane_start_refused_for_vram_faults_the_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A lane whose start is deferred for VRAM is not coming up, so the job is reissued as before."""
        rig = _ExtendedLaneRig(monkeypatch, device_free_mb=100.0)
        job = _make_controlnet_job(control_type="lineart", ttl=600)
        assert job.id_ is not None
        await rig.pm._job_tracker.record_popped_job(job, time_popped=time.time())

        for _ in range(2):
            rig.tick()

        assert rig.constructed == []
        assert rig.pm._process_lifecycle.has_pending_gpu_starts()
        assert job not in rig.pm._job_tracker.jobs_pending_inference
        assert rig.faults == ["extended_controlnet_not_offered"]
        assert not rig.pm._inference_scheduler.job_awaits_utilities_lane(job)

    async def test_a_job_whose_deadline_cannot_cover_the_bring_up_is_faulted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A job already past the anti-starvation share of its ttl is reissued rather than held."""
        rig = _ExtendedLaneRig(monkeypatch)
        job = _make_controlnet_job(control_type="lineart", ttl=100)
        assert job.id_ is not None
        await rig.pm._job_tracker.record_popped_job(job, time_popped=time.time() - 40.0)

        rig.tick()

        assert job not in rig.pm._job_tracker.jobs_pending_inference
        assert rig.faults == ["extended_controlnet_not_offered"]
        assert not rig.pm._inference_scheduler.job_awaits_utilities_lane(job)

    async def test_a_live_lane_whose_reported_set_lacks_the_type_faults_the_job(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Once a live lane reports its servable set, a type missing from it is not waited for."""
        rig = _ExtendedLaneRig(monkeypatch)
        job = _make_controlnet_job(control_type="lineart", ttl=600)
        assert job.id_ is not None
        await rig.pm._job_tracker.record_popped_job(job, time_popped=time.time())
        rig.tick()
        assert rig.pm._inference_scheduler.job_awaits_utilities_lane(job)
        lane = rig.lane()
        assert lane is not None
        lane.last_process_state = HordeProcessState.WAITING_FOR_JOB

        rig.report_servable(["canny"])
        rig.tick()

        assert job not in rig.pm._job_tracker.jobs_pending_inference
        assert rig.faults == ["extended_controlnet_not_offered"]
        assert not rig.pm._inference_scheduler.job_awaits_utilities_lane(job)

    @pytest.mark.parametrize(
        ("age_seconds", "held"),
        [(10.0, True), (50.0, False)],
        ids=["inside-the-fallback-share", "past-the-fallback-share"],
    )
    async def test_a_job_with_no_ttl_is_bounded_by_the_fallback_ttl(
        self,
        monkeypatch: pytest.MonkeyPatch,
        age_seconds: float,
        held: bool,
    ) -> None:
        """With no ttl on the job and none seen, the hold reads the skip budget's default ttl."""
        rig = _ExtendedLaneRig(monkeypatch)
        assert rig.pm._state.recent_job_ttl is None
        # The two ages sit either side of the fallback's anti-starvation share.
        assert 10.0 < ANTI_STARVATION_TTL_FRACTION * fallback_ttl_seconds(None) < 50.0
        job = _make_controlnet_job(control_type="lineart")
        assert job.id_ is not None and job.ttl is None
        await rig.pm._job_tracker.record_popped_job(job, time_popped=time.time() - age_seconds)

        rig.tick()

        assert rig.pm._inference_scheduler.job_awaits_utilities_lane(job) is held
        assert (job in rig.pm._job_tracker.jobs_pending_inference) is held
        assert rig.faults == ([] if held else ["extended_controlnet_not_offered"])

    def test_the_annotator_probe_runs_once_per_type_until_presence_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A second read is served from the cache, and a changed presence report clears it."""
        pm = make_testable_process_manager(enable_image_utilities=True, extended_controlnet=True)
        probed: list[list[str]] = []

        def _probe(control_types: list[str], _root: object) -> bool:
            probed.append(list(control_types))
            return True

        monkeypatch.setattr("horde_model_reference.on_disk_layout.annotators_present_for_control_types", _probe)
        monkeypatch.setattr("horde_model_reference.on_disk_layout.resolve_weights_root", lambda: Path("unused"))

        assert pm._extended_annotators_on_disk(["lineart", "mlsd"]) is True
        assert sorted(probed) == [["lineart"], ["mlsd"]]
        assert pm._extended_annotators_on_disk(["lineart"]) is True
        assert pm._extended_annotators_on_disk(["mlsd", "lineart"]) is True
        assert len(probed) == 2, "a second read does not touch the filesystem"

        def _report(controlnet_present: bool | None) -> None:
            pm._on_download_availability(
                HordeDownloadAvailabilityMessage(
                    process_id=99,
                    process_launch_identifier=0,
                    info="test",
                    available_model_names=[],
                    controlnet_present=controlnet_present,
                ),
            )

        _report(False)
        assert pm._extended_annotators_on_disk(["lineart"]) is True
        assert len(probed) == 3, "the first presence report clears the cache"
        _report(False)
        assert pm._extended_annotators_on_disk(["lineart"]) is True
        assert len(probed) == 3, "an unchanged report keeps it"
        _report(True)
        assert pm._extended_annotators_on_disk(["lineart"]) is True
        assert len(probed) == 4, "a changed report clears it"

    @pytest.mark.parametrize(
        ("expected_sampling_seconds", "held"),
        [(30.0, True), (250.0, False), (None, False)],
        ids=["figure-leaves-room", "figure-leaves-no-room", "no-figure-reads-the-share"],
    )
    async def test_the_hold_reads_the_performance_models_sampling_figure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        expected_sampling_seconds: float | None,
        held: bool,
    ) -> None:
        """Past the anti-starvation share, a job stays held only while its ttl covers its expected sampling."""
        scheduler = _make_inference_scheduler(job_tracker=JobTracker())
        job = _make_controlnet_job(control_type="lineart", ttl=300)
        # 120 s after pop is past the share (90 s) and inside 300 - 30 - 45 = 225 s.
        await scheduler._job_tracker.record_popped_job(job, time_popped=time.time() - 120.0)
        performance_model = Mock()
        performance_model.expected_sampling_seconds.return_value = expected_sampling_seconds
        monkeypatch.setattr(scheduler, "_performance_model", performance_model)
        monkeypatch.setattr(scheduler._model_metadata, "get_baseline", lambda _model: "stable_diffusion_1")

        expired = scheduler.reconcile_utilities_lane_holds([(job, "lineart")])

        assert performance_model.expected_sampling_seconds.called
        assert scheduler.job_awaits_utilities_lane(job) is held
        assert expired == ([] if held else [job])

    async def test_a_held_head_is_stepped_past_and_names_the_lane_hold(self) -> None:
        """Selection skips a lane-held head for the job behind it, and the empty slot names the hold."""
        process_info = make_mock_process_info(
            0, model_name="stable_diffusion", state=HordeProcessState.PRELOADED_MODEL
        )
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name="stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )
        job_tracker = JobTracker()
        held = _make_controlnet_job(control_type="lineart", ttl=600)
        behind = make_job_pop_response("stable_diffusion")
        await job_tracker.record_popped_job(held, time_popped=time.time())
        await job_tracker.record_popped_job(behind, time_popped=time.time())
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: process_info}),
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
        )

        assert scheduler.reconcile_utilities_lane_holds([(held, "lineart")]) == []

        selection = await scheduler.get_next_job_and_process(information_only=True)
        assert selection is not None
        assert selection.next_job is behind
        bucket, text = scheduler._classify_dispatch_stall(held, {})
        assert bucket is SlotDutyBucket.UTILITIES_LANE_BRING_UP
        assert "image-utilities lane" in text

        assert scheduler.reconcile_utilities_lane_holds([]) == []
        assert not scheduler.job_awaits_utilities_lane(held)
        selection = await scheduler.get_next_job_and_process(information_only=True)
        assert selection is not None
        assert selection.next_job is held
