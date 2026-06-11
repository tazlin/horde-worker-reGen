"""Fake inference and safety processes for orchestration testing.

These classes speak the exact same pipe/queue message protocol as the real
``HordeInferenceProcess`` and ``HordeSafetyProcess``, but never import hordelib,
torch, or any other ML dependency. They allow the full multiprocessing
orchestration layer (process manager, scheduler, safety orchestrator,
job tracker) to be exercised end-to-end on machines with no GPU and without
the heavy dependency stack loaded into the child processes.

The module-level entry points mirror the signatures of
``worker_entry_points.start_inference_process`` / ``start_safety_process`` so
they can be passed directly as ``multiprocessing.Process`` targets (they must
remain module-level functions to stay picklable under spawn).
"""

from __future__ import annotations

import time

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock, Semaphore
from typing import override

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger

from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management._dummy_images import make_dummy_png_base64
from horde_worker_regen.process_management.debug_attach import maybe_wait_for_process_debugger
from horde_worker_regen.process_management.horde_process import HordeProcess
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeImageResult,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeModelStateChangeMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    HordeSafetyControlMessage,
    HordeSafetyEvaluation,
    HordeSafetyResultMessage,
    ModelLoadState,
)


class FakeInferenceProcess(HordeProcess):
    """A lightweight stand-in for ``HordeInferenceProcess``.

    Reproduces the message sequences the main process expects (preload,
    inference start/complete, unloads) without performing any real work.
    """

    _active_model_name: str | None = None
    _inference_semaphore: Semaphore
    _job_delay_seconds: float
    _fail_every_n: int
    _jobs_started: int = 0

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        inference_semaphore: Semaphore,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        job_delay_seconds: float = 0.0,
        fail_every_n: int = 0,
    ) -> None:
        """Initialise the fake inference process.

        Args:
            process_id (int): The ID of the process. This is not the same as the PID.
            process_message_queue (ProcessQueue): The queue to send messages to the main process.
            pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
            inference_semaphore (Semaphore): The semaphore limiting concurrent inference; acquired and \
                released around each fake job so concurrency control is still exercised.
            disk_lock (Lock): The lock to use for disk access.
            process_launch_identifier (int): The unique identifier for this launch.
            job_delay_seconds (float, optional): How long each fake inference job takes. Defaults to 0.0.
            fail_every_n (int, optional): If > 0, every nth job reports a faulted result instead of \
                images. Defaults to 0 (never fail).
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
        )
        self._inference_semaphore = inference_semaphore
        self._job_delay_seconds = job_delay_seconds
        self._fail_every_n = fail_every_n

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def get_vram_usage_bytes(self) -> int:
        """Return a fixed fake VRAM usage value."""
        return 0

    @override
    def get_vram_total_bytes(self) -> int:
        """Return a fixed fake VRAM total value."""
        return 0

    def on_horde_model_state_change(
        self,
        horde_model_name: str,
        process_state: HordeProcessState,
        horde_model_state: ModelLoadState,
        time_elapsed: float | None = None,
    ) -> None:
        """Send a model state change message followed by a memory report, as the real process does."""
        self.process_message_queue.put(
            HordeModelStateChangeMessage(
                process_state=process_state,
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Model {horde_model_name} {horde_model_state.name}",
                horde_model_name=horde_model_name,
                horde_model_state=horde_model_state,
                time_elapsed=time_elapsed,
            ),
        )
        self.send_memory_report_message(include_vram=True)

    def preload_model(self, horde_model_name: str) -> None:
        """Pretend to preload a model, emitting the same state sequence as the real process."""
        if self._active_model_name == horde_model_name:
            return

        if self._active_model_name is not None:
            self.on_horde_model_state_change(
                process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
                horde_model_name=self._active_model_name,
                horde_model_state=ModelLoadState.ON_DISK,
            )

        self.on_horde_model_state_change(
            process_state=HordeProcessState.PRELOADING_MODEL,
            horde_model_name=horde_model_name,
            horde_model_state=ModelLoadState.LOADING,
        )

        time_start = time.time()
        self._active_model_name = horde_model_name

        self.on_horde_model_state_change(
            process_state=HordeProcessState.PRELOADED_MODEL,
            horde_model_name=horde_model_name,
            horde_model_state=ModelLoadState.LOADED_IN_RAM,
            time_elapsed=time.time() - time_start,
        )

    def _run_fake_inference(self, job_info: ImageGenerateJobPopResponse) -> None:
        """Pretend to run inference and send the result messages for it."""
        self._jobs_started += 1
        should_fail = self._fail_every_n > 0 and self._jobs_started % self._fail_every_n == 0

        self._inference_semaphore.acquire()
        time_start = time.time()
        try:
            if self._job_delay_seconds > 0:
                deadline = time_start + self._job_delay_seconds
                while time.time() < deadline:
                    self.send_heartbeat_message(heartbeat_type=HordeHeartbeatType.INFERENCE_STEP)
                    time.sleep(min(0.05, self._job_delay_seconds))
        finally:
            self._inference_semaphore.release()

        n_iter = job_info.payload.n_iter if job_info.payload.n_iter else 1
        job_image_results = None
        if not should_fail:
            job_image_results = [HordeImageResult(image_base64=make_dummy_png_base64()) for _ in range(n_iter)]

        self.process_message_queue.put(
            HordeInferenceResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info="fake inference",
                state=GENERATION_STATE.ok if not should_fail else GENERATION_STATE.faulted,
                time_elapsed=time.time() - time_start,
                job_image_results=job_image_results,
                sdk_api_job_info=job_info,
            ),
        )

        if self._active_model_name is not None:
            self.on_horde_model_state_change(
                process_state=(
                    HordeProcessState.INFERENCE_COMPLETE if not should_fail else HordeProcessState.INFERENCE_FAILED
                ),
                horde_model_name=self._active_model_name,
                horde_model_state=ModelLoadState.LOADED_IN_VRAM,
            )

        self.send_process_state_change_message(
            HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Handle control messages with the same observable behavior as the real inference process."""
        logger.debug(f"Fake inference process received {type(message).__name__}: {message.control_flag}")

        if isinstance(message, HordePreloadInferenceModelMessage):
            self.preload_model(message.horde_model_name)
        elif isinstance(message, HordeInferenceControlMessage) and (
            message.control_flag == HordeControlFlag.START_INFERENCE
        ):
            if message.horde_model_name != self._active_model_name:
                self.preload_model(message.horde_model_name)

            self.on_horde_model_state_change(
                horde_model_name=message.horde_model_name,
                process_state=HordeProcessState.INFERENCE_STARTING,
                horde_model_state=ModelLoadState.IN_USE,
            )

            self._run_fake_inference(message.sdk_api_job_info)
        elif message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
            if self._active_model_name is not None:
                self.on_horde_model_state_change(
                    process_state=HordeProcessState.UNLOADED_MODEL_FROM_VRAM,
                    horde_model_name=self._active_model_name,
                    horde_model_state=ModelLoadState.LOADED_IN_RAM,
                )
            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="Unloaded models from VRAM",
            )
        elif message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            if self._active_model_name is not None:
                self.on_horde_model_state_change(
                    process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
                    horde_model_name=self._active_model_name,
                    horde_model_state=ModelLoadState.ON_DISK,
                )
            self._active_model_name = None
            self.send_process_state_change_message(
                process_state=HordeProcessState.WAITING_FOR_JOB,
                info="Unloaded models from RAM",
            )

    @override
    def cleanup_for_exit(self) -> None:
        """No resources to release; report the final state like the real process."""
        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDED,
            info="Process ended",
        )


class FakeSafetyProcess(HordeProcess):
    """A lightweight stand-in for ``HordeSafetyProcess`` that approves every image."""

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        evaluation_delay_seconds: float = 0.0,
    ) -> None:
        """Initialise the fake safety process.

        Args:
            process_id (int): The ID of the process. This is not the same as the PID.
            process_message_queue (ProcessQueue): The queue to send messages to the main process.
            pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
            disk_lock (Lock): The lock to use for disk access.
            process_launch_identifier (int): The unique identifier for this launch.
            evaluation_delay_seconds (float, optional): How long each fake evaluation takes. Defaults to 0.0.
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
        )
        self._evaluation_delay_seconds = evaluation_delay_seconds

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def get_vram_usage_bytes(self) -> int:
        """Return a fixed fake VRAM usage value."""
        return 0

    @override
    def get_vram_total_bytes(self) -> int:
        """Return a fixed fake VRAM total value."""
        return 0

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Evaluate any safety request as safe and report back immediately."""
        if not isinstance(message, HordeSafetyControlMessage):
            logger.critical(f"Fake safety process received unexpected message type: {type(message).__name__}")
            return

        self.send_process_state_change_message(
            process_state=HordeProcessState.EVALUATING_SAFETY,
            info="Evaluating safety",
        )

        time_start = time.time()
        if self._evaluation_delay_seconds > 0:
            time.sleep(self._evaluation_delay_seconds)

        self.process_message_queue.put(
            HordeSafetyResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info="fake safety evaluation",
                time_elapsed=time.time() - time_start,
                job_id=message.job_id,
                safety_evaluations=[
                    HordeSafetyEvaluation(
                        is_nsfw=False,
                        is_csam=False,
                        replacement_image_base64=None,
                    )
                    for _ in message.images_base64
                ],
            ),
        )

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def cleanup_for_exit(self) -> None:
        """No resources to release; report the final state like the real process."""
        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDED,
            info="Process ended",
        )


def start_fake_inference_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    inference_semaphore: Semaphore,
    disk_lock: Lock,
    aux_model_lock: Lock,
    vae_decode_semaphore: Semaphore,
    process_launch_identifier: int,
    *,
    low_memory_mode: bool = False,
    high_memory_mode: bool = False,
    very_high_memory_mode: bool = False,
    amd_gpu: bool = False,
    directml: int | None = None,
    vram_heavy_models: bool = False,
    dry_run_skip_inference: bool = False,
    dry_run_inference_delay: float = 1.0,
    fail_every_n: int = 0,
) -> None:
    """Start a fake inference process.

    Signature-compatible with ``worker_entry_points.start_inference_process`` so it can
    be injected as a drop-in multiprocessing target. Memory/GPU related arguments are
    accepted and ignored; ``dry_run_inference_delay`` controls how long fake jobs take.
    ``fail_every_n`` makes every nth job report a faulted result (0 = never), letting
    harnesses exercise the fault path. Inject it with ``functools.partial`` (partials of
    module-level functions stay picklable under spawn).
    """
    logger.remove()
    maybe_wait_for_process_debugger(process_id, "fake inference")
    worker_process = FakeInferenceProcess(
        process_id=process_id,
        process_message_queue=process_message_queue,
        pipe_connection=pipe_connection,
        inference_semaphore=inference_semaphore,
        disk_lock=disk_lock,
        process_launch_identifier=process_launch_identifier,
        job_delay_seconds=dry_run_inference_delay,
        fail_every_n=fail_every_n,
    )
    worker_process.main_loop()


def start_fake_safety_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    process_launch_identifier: int,
    cpu_only: bool = True,
    *,
    high_memory_mode: bool = False,
    amd_gpu: bool = False,
    directml: int | None = None,
    dry_run_skip_safety: bool = False,
) -> None:
    """Start a fake safety process.

    Signature-compatible with ``worker_entry_points.start_safety_process`` so it can
    be injected as a drop-in multiprocessing target. GPU related arguments are
    accepted and ignored.
    """
    logger.remove()
    maybe_wait_for_process_debugger(process_id, "fake safety")
    worker_process = FakeSafetyProcess(
        process_id=process_id,
        process_message_queue=process_message_queue,
        pipe_connection=pipe_connection,
        disk_lock=disk_lock,
        process_launch_identifier=process_launch_identifier,
    )
    worker_process.main_loop()
