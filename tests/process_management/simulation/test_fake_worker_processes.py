"""Unit tests for the fake worker processes used by the e2e harness.

These drive the fakes' control-message handlers directly (no subprocesses), verifying
they emit the same message sequences the orchestration layer expects from the real
inference and safety processes.
"""

from __future__ import annotations

import uuid
from unittest.mock import Mock

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.fields import GenerationID

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeModelStateChangeMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    HordeSafetyControlMessage,
    HordeSafetyResultMessage,
    HordeSampleControlMessage,
    HordeSampleResultMessage,
    HordeVaeDecodeControlMessage,
    HordeVaeDecodeResultMessage,
    HordeVaeEncodeControlMessage,
    HordeVaeEncodeResultMessage,
    ModelLoadState,
    SampleSliceSpec,
)
from horde_worker_regen.process_management.simulation.fake_worker_processes import (
    FakeInferenceProcess,
    FakePostProcessProcess,
    FakeSafetyProcess,
)
from horde_worker_regen.process_management.workers.vae_lane_process import HordeVaeLaneProcess
from tests.process_management.conftest import make_job_pop_response


class RecordingQueue:
    """Captures every message a fake process emits, in order."""

    def __init__(self) -> None:
        """Start with an empty message log."""
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        """Record a message."""
        self.messages.append(message)

    def of_type[T](self, message_type: type[T]) -> list[T]:
        """Return all recorded messages of the given type, with static type preservation."""
        return [m for m in self.messages if isinstance(m, message_type)]

    def state_changes(self) -> list[HordeProcessState]:
        """Return the sequence of process states reported so far."""
        return [m.process_state for m in self.messages if isinstance(m, HordeProcessStateChangeMessage)]


def make_fake_inference_process(**kwargs: object) -> tuple[FakeInferenceProcess, RecordingQueue]:
    """Construct a FakeInferenceProcess wired to a recording queue and mock primitives."""
    queue = RecordingQueue()
    process = FakeInferenceProcess(
        process_id=1,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        inference_semaphore=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
        **kwargs,  # type: ignore[arg-type]
    )
    return process, queue


def make_fake_post_process_process() -> tuple[FakePostProcessProcess, RecordingQueue]:
    """Construct a FakePostProcessProcess wired to a recording queue and mock primitives."""
    queue = RecordingQueue()
    process = FakePostProcessProcess(
        process_id=3,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
    )
    return process, queue


def make_vae_lane_process() -> tuple[HordeVaeLaneProcess, RecordingQueue]:
    """Construct a dry-run HordeVaeLaneProcess wired to a recording queue and mock primitives.

    The dry-run VAE lane is ML-free (it returns plausible stand-in latent/image bytes), so it stands in
    for the real lane in these protocol tests exactly as it does in the harness.
    """
    queue = RecordingQueue()
    process = HordeVaeLaneProcess(
        process_id=4,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
        dry_run=True,
    )
    return process, queue


def make_fake_safety_process() -> tuple[FakeSafetyProcess, RecordingQueue]:
    """Construct a FakeSafetyProcess wired to a recording queue and mock primitives."""
    queue = RecordingQueue()
    process = FakeSafetyProcess(
        process_id=2,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
    )
    return process, queue


class TestFakeInferenceProcess:
    """The fake inference process must mirror the real process's observable message protocol."""

    def test_init_reports_starting_then_waiting(self) -> None:
        """On startup the process must report PROCESS_STARTING followed by WAITING_FOR_JOB."""
        _, queue = make_fake_inference_process()

        states = queue.state_changes()
        assert states[0] == HordeProcessState.PROCESS_STARTING
        assert HordeProcessState.WAITING_FOR_JOB in states

    def test_preload_emits_model_state_sequence(self) -> None:
        """Preloading must emit PRELOADING_MODEL then PRELOADED_MODEL for the requested model."""
        process, queue = make_fake_inference_process()
        job = make_job_pop_response(model="Deliberate")

        process._receive_and_handle_control_message(
            HordePreloadInferenceModelMessage(
                control_flag=HordeControlFlag.PRELOAD_MODEL,
                horde_model_name="Deliberate",
                will_load_loras=False,
                seamless_tiling_enabled=False,
                sdk_api_job_info=job,
            ),
        )

        model_messages = queue.of_type(HordeModelStateChangeMessage)
        states = [(m.process_state, m.horde_model_state) for m in model_messages]
        assert (HordeProcessState.PRELOADING_MODEL, ModelLoadState.LOADING) in states
        assert (HordeProcessState.PRELOADED_MODEL, ModelLoadState.LOADED_IN_RAM) in states
        assert all(m.horde_model_name == "Deliberate" for m in model_messages)

    def test_preload_different_model_unloads_previous(self) -> None:
        """Preloading a second model must first report the previous model unloaded from RAM."""
        process, queue = make_fake_inference_process()
        job = make_job_pop_response(model="Deliberate")

        process.preload_model("Deliberate")
        queue.messages.clear()
        process.preload_model("AnotherModel")

        model_messages = queue.of_type(HordeModelStateChangeMessage)
        unloads = [
            m
            for m in model_messages
            if m.process_state == HordeProcessState.UNLOADED_MODEL_FROM_RAM and m.horde_model_name == "Deliberate"
        ]
        assert len(unloads) == 1
        assert job.model == "Deliberate"

    def test_start_inference_produces_result_and_returns_to_waiting(self) -> None:
        """A START_INFERENCE message must produce a result with valid PNG images, then WAITING_FOR_JOB."""
        process, queue = make_fake_inference_process()
        job = make_job_pop_response(model="Deliberate", n_iter=2)

        process._receive_and_handle_control_message(
            HordeInferenceControlMessage(
                control_flag=HordeControlFlag.START_INFERENCE,
                horde_model_name="Deliberate",
                sdk_api_job_info=job,
            ),
        )

        result_messages = queue.of_type(HordeInferenceResultMessage)
        assert len(result_messages) == 1
        result = result_messages[0]
        assert result.state == GENERATION_STATE.ok
        assert result.sdk_api_job_info.id_ == job.id_
        assert result.job_image_results is not None
        assert len(result.job_image_results) == 2
        for image_result in result.job_image_results:
            assert image_result.image_bytes.startswith(b"\x89PNG")

        states = queue.state_changes()
        assert HordeProcessState.INFERENCE_STARTING in states
        assert HordeProcessState.INFERENCE_COMPLETE in states
        assert states[-1] == HordeProcessState.WAITING_FOR_JOB

    def test_fail_every_n_reports_faulted_result(self) -> None:
        """With fail_every_n=1, the result message must report a faulted generation with no images."""
        process, queue = make_fake_inference_process(fail_every_n=1)
        job = make_job_pop_response(model="Deliberate")

        process._receive_and_handle_control_message(
            HordeInferenceControlMessage(
                control_flag=HordeControlFlag.START_INFERENCE,
                horde_model_name="Deliberate",
                sdk_api_job_info=job,
            ),
        )

        result_messages = queue.of_type(HordeInferenceResultMessage)
        assert len(result_messages) == 1
        assert result_messages[0].state == GENERATION_STATE.faulted
        assert result_messages[0].job_image_results is None

    def test_unload_from_ram_clears_active_model(self) -> None:
        """UNLOAD_MODELS_FROM_RAM must report the unload and clear the active model."""
        process, queue = make_fake_inference_process()
        process.preload_model("Deliberate")
        queue.messages.clear()

        process._receive_and_handle_control_message(
            HordeInferenceControlMessage(
                control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
                horde_model_name="Deliberate",
                sdk_api_job_info=make_job_pop_response(model="Deliberate"),
            ),
        )

        assert process._active_model_name is None

    def test_sample_stage_emits_starting_result_and_waiting(self) -> None:
        """A START_SAMPLE message must report INFERENCE_STARTING, return a LATENT per slice, then idle.

        This mirrors the real sampler's disaggregated stage: the ``INFERENCE_STARTING`` is a plain
        process-state change (no model-state traffic), so the fake never fakes a model residency the
        parent tracks. Two slices must yield two per-slice results, in order.
        """
        process, queue = make_fake_inference_process()
        job_a = make_job_pop_response(model="SDXL 1.0")
        job_b = make_job_pop_response(model="SDXL 1.0")

        process._receive_and_handle_control_message(
            HordeSampleControlMessage(
                horde_model_name="SDXL 1.0",
                slices=[
                    SampleSliceSpec(
                        job_id=job_a.id_,
                        positive_conditioning_bytes=b"pos",
                        negative_conditioning_bytes=b"neg",
                        sdk_api_job_info=job_a,
                    ),
                    SampleSliceSpec(
                        job_id=job_b.id_,
                        positive_conditioning_bytes=b"pos",
                        negative_conditioning_bytes=b"neg",
                        sdk_api_job_info=job_b,
                    ),
                ],
            ),
        )

        result_messages = queue.of_type(HordeSampleResultMessage)
        assert len(result_messages) == 1
        result = result_messages[0]
        assert [r.job_id for r in result.results] == [job_a.id_, job_b.id_]
        assert all(r.latent_bytes is not None for r in result.results)
        assert all(r.state == GENERATION_STATE.ok for r in result.results)

        # No HordeModelStateChangeMessage: the pinned sampler's residency is the parent's to track.
        assert queue.of_type(HordeModelStateChangeMessage) == []
        states = queue.state_changes()
        assert HordeProcessState.INFERENCE_STARTING in states
        assert states[-1] == HordeProcessState.WAITING_FOR_JOB


class TestVaeLaneProcess:
    """The dry-run VAE lane must mirror the real lane's observable protocol for the VAE stages."""

    def test_vae_encode_emits_post_processing_latent_and_waiting(self) -> None:
        """A VAE-encode message must report POST_PROCESSING, return a source LATENT, then idle."""
        process, queue = make_vae_lane_process()
        job = make_job_pop_response(model="SDXL 1.0")

        process._receive_and_handle_control_message(
            HordeVaeEncodeControlMessage(
                horde_model_name="SDXL 1.0",
                job_id=job.id_,
                sdk_api_job_info=job,
            ),
        )

        result_messages = queue.of_type(HordeVaeEncodeResultMessage)
        assert len(result_messages) == 1
        result = result_messages[0]
        assert result.job_id == job.id_
        assert result.latent_bytes is not None
        assert result.state == GENERATION_STATE.ok

        states = queue.state_changes()
        assert HordeProcessState.POST_PROCESSING in states
        assert states[-1] == HordeProcessState.WAITING_FOR_JOB

    def test_vae_decode_emits_post_processing_images_and_waiting(self) -> None:
        """A VAE-decode message must report POST_PROCESSING, return one PNG per iteration, then idle."""
        process, queue = make_vae_lane_process()
        job = make_job_pop_response(model="SDXL 1.0", n_iter=2)

        process._receive_and_handle_control_message(
            HordeVaeDecodeControlMessage(
                horde_model_name="SDXL 1.0",
                job_id=job.id_,
                sdk_api_job_info=job,
                latent_bytes=b"latent",
            ),
        )

        result_messages = queue.of_type(HordeVaeDecodeResultMessage)
        assert len(result_messages) == 1
        result = result_messages[0]
        assert result.job_id == job.id_
        assert result.state == GENERATION_STATE.ok
        assert result.job_image_results is not None
        assert len(result.job_image_results) == 2
        for image_result in result.job_image_results:
            assert image_result.image_bytes.startswith(b"\x89PNG")

        states = queue.state_changes()
        assert HordeProcessState.POST_PROCESSING in states
        assert states[-1] == HordeProcessState.WAITING_FOR_JOB


class TestFakePostProcessProcess:
    """The fake post-processing lane serves only PP work now; VAE messages must be rejected, not served."""

    def test_vae_decode_message_is_rejected_without_a_result(self) -> None:
        """A VAE stage message reaching the post-processing lane is rejected: no VAE result is emitted.

        The VAE stages moved to the dedicated VAE lane, so the post-processing fake must once again treat a
        VAE control message as an unexpected type rather than silently serving it.
        """
        process, queue = make_fake_post_process_process()
        job = make_job_pop_response(model="SDXL 1.0")

        process._receive_and_handle_control_message(
            HordeVaeDecodeControlMessage(
                horde_model_name="SDXL 1.0",
                job_id=job.id_,
                sdk_api_job_info=job,
                latent_bytes=b"latent",
            ),
        )

        assert queue.of_type(HordeVaeDecodeResultMessage) == []


class TestFakeSafetyProcess:
    """The fake safety process must mirror the real process's observable message protocol."""

    def test_init_reports_starting_then_waiting(self) -> None:
        """On startup the process must report PROCESS_STARTING followed by WAITING_FOR_JOB."""
        _, queue = make_fake_safety_process()

        states = queue.state_changes()
        assert states[0] == HordeProcessState.PROCESS_STARTING
        assert HordeProcessState.WAITING_FOR_JOB in states

    def test_evaluate_safety_approves_every_image(self) -> None:
        """An EVALUATE_SAFETY message must yield one all-clear evaluation per image."""
        process, queue = make_fake_safety_process()
        job_id = GenerationID(root=uuid.uuid4())

        process._receive_and_handle_control_message(
            HordeSafetyControlMessage(
                control_flag=HordeControlFlag.EVALUATE_SAFETY,
                job_id=job_id,
                prompt="a test prompt",
                censor_nsfw=True,
                sfw_worker=True,
                images_bytes=[b"aaa", b"bbb", b"ccc"],
                horde_model_info=None,
            ),
        )

        result_messages = queue.of_type(HordeSafetyResultMessage)
        assert len(result_messages) == 1
        result = result_messages[0]
        assert isinstance(result, HordeSafetyResultMessage)
        assert result.job_id == job_id
        assert len(result.safety_evaluations) == 3
        for evaluation in result.safety_evaluations:
            assert not evaluation.is_nsfw
            assert not evaluation.is_csam
            assert not evaluation.failed
            assert evaluation.replacement_image_bytes is None

        states = queue.state_changes()
        assert HordeProcessState.EVALUATING_SAFETY in states
        assert states[-1] == HordeProcessState.WAITING_FOR_JOB
