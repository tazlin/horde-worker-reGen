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
from hordelib.metrics import JobPhaseMetrics, SamplingStats
from loguru import logger

from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management._dummy_images import make_dummy_png_base64
from horde_worker_regen.process_management.debug_attach import maybe_wait_for_process_debugger
from horde_worker_regen.process_management.horde_process import HordeProcess, HordeProcessType
from horde_worker_regen.process_management.messages import (
    AlchemyFormSpec,
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeDownloadAvailabilityMessage,
    HordeDownloadControlMessage,
    HordeHeartbeatType,
    HordeImageResult,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeJobMetricsMessage,
    HordeModelStateChangeMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    HordeSafetyControlMessage,
    HordeSafetyEvaluation,
    HordeSafetyResultMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadFailure,
    DownloadItem,
    DownloadPhase,
    DownloadStatusSnapshot,
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
    def get_vram_usage_mb(self) -> int:
        """Return a fixed fake VRAM usage value."""
        return 0

    @override
    def get_vram_total_mb(self) -> int:
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

        # The real process snapshots hordelib's metrics collector after each job; emit a
        # synthetic equivalent so the pipe -> dispatcher -> run-metrics chain is exercised
        # without any GPU.
        steps = job_info.payload.ddim_steps if job_info.payload.ddim_steps else 30
        elapsed = max(time.time() - time_start, 0.001)
        self.process_message_queue.put(
            HordeJobMetricsMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Job metrics for {job_info.id_}",
                job_id=str(job_info.id_),
                phase_metrics=JobPhaseMetrics(
                    sampling=SamplingStats(
                        steps_completed=steps,
                        total_steps=steps,
                        duration_seconds=elapsed,
                        iterations_per_second=steps / elapsed,
                    ),
                    vram_used_high_water_mb=1234,
                    ram_used_high_water_mb=2345,
                ),
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

    def _run_fake_alchemy(self, form: AlchemyFormSpec) -> None:
        """Pretend to run an alchemy form, emitting the same message sequence as the real process."""
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_STARTING,
            info=f"Starting alchemy form {form.form} ({form.form_id})",
        )
        time_start = time.time()
        if self._job_delay_seconds > 0:
            time.sleep(self._job_delay_seconds)

        self.process_message_queue.put(
            HordeAlchemyResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Alchemy form {form.form} ({form.form_id})",
                time_elapsed=time.time() - time_start,
                form_id=form.form_id,
                form=form.form,
                state=GENERATION_STATE.ok,
                image_base64=make_dummy_png_base64(),
            ),
        )
        self.process_message_queue.put(
            HordeJobMetricsMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Job metrics for {form.form_id}",
                job_id=form.form_id,
                is_alchemy=True,
                phase_metrics=JobPhaseMetrics(vram_used_high_water_mb=600, ram_used_high_water_mb=1200),
            ),
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_COMPLETE,
            info=f"Finished alchemy form {form.form} ({form.form_id})",
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Handle control messages with the same observable behavior as the real inference process."""
        logger.debug(f"Fake inference process received {type(message).__name__}: {message.control_flag}")

        if isinstance(message, HordeAlchemyControlMessage):
            self._run_fake_alchemy(message.form)
        elif isinstance(message, HordePreloadInferenceModelMessage):
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
    def get_vram_usage_mb(self) -> int:
        """Return a fixed fake VRAM usage value."""
        return 0

    @override
    def get_vram_total_mb(self) -> int:
        """Return a fixed fake VRAM total value."""
        return 0

    def _run_fake_alchemy(self, form: AlchemyFormSpec) -> None:
        """Pretend to run a CLIP-class alchemy form (caption/interrogation/nsfw)."""
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_STARTING,
            info=f"Starting alchemy form {form.form} ({form.form_id})",
        )
        time_start = time.time()
        if self._evaluation_delay_seconds > 0:
            time.sleep(self._evaluation_delay_seconds)

        result_payload: dict = {form.form: "a fake caption"} if form.form == "caption" else {form.form: False}
        self.process_message_queue.put(
            HordeAlchemyResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Alchemy form {form.form} ({form.form_id})",
                time_elapsed=time.time() - time_start,
                form_id=form.form_id,
                form=form.form,
                state=GENERATION_STATE.ok,
                result_payload=result_payload,
            ),
        )
        self.process_message_queue.put(
            HordeJobMetricsMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Job metrics for {form.form_id}",
                job_id=form.form_id,
                is_alchemy=True,
                phase_metrics=JobPhaseMetrics(ram_used_high_water_mb=800),
            ),
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_COMPLETE,
            info=f"Finished alchemy form {form.form} ({form.form_id})",
        )
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Evaluate any safety request as safe and report back immediately."""
        if isinstance(message, HordeAlchemyControlMessage):
            self._run_fake_alchemy(message.form)
            return

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
    gpu_sampling_lease: Semaphore | None = None,
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


class FakeDownloadProcess(HordeProcess):
    """A lightweight stand-in for ``HordeDownloadProcess`` that imports no ML dependencies.

    Starts from a scripted on-disk set and "downloads" any requested model (after an optional
    per-model delay) by adding it to that set, unless it is in ``fail_models``. Emits the same
    ``HordeDownloadAvailabilityMessage`` snapshots the real process does.
    """

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        *,
        scripted_present: list[str] | None = None,
        download_delay_seconds: float = 0.0,
        fail_models: list[str] | None = None,
        rate_limit_kbps: int | None = None,
        paused: bool = False,
    ) -> None:
        """Initialise with a scripted present-set and download behaviour."""
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
        )
        self.process_type = HordeProcessType.DOWNLOAD
        self._present: set[str] = set(scripted_present or [])
        self._download_delay_seconds = download_delay_seconds
        self._fail_models = set(fail_models or [])
        self._pending: list[str] = []
        self._failed: list[str] = []
        self._currently_downloading: str | None = None
        self._paused = paused
        self._rate_limit_kbps = rate_limit_kbps if (rate_limit_kbps or 0) > 0 else None
        self._send_availability()

    def _status_snapshot(self) -> DownloadStatusSnapshot:
        """Project the fake's state into the same rich snapshot the real process emits."""
        if self._currently_downloading is not None:
            phase = DownloadPhase.PAUSED if self._paused else DownloadPhase.DOWNLOADING
            current = CurrentDownloadStatus(
                model_name=self._currently_downloading,
                feature="image model",
                target_dir="models/compvis",
            )
        else:
            phase = DownloadPhase.PAUSED if self._paused and self._pending else DownloadPhase.IDLE
            current = None
        return DownloadStatusSnapshot(
            phase=phase,
            current=current,
            pending=[DownloadItem(model_name=name, feature="image model") for name in self._pending],
            failures=[
                DownloadFailure(model_name=name, feature="image model", reason="failed") for name in self._failed
            ],
            present_model_names=sorted(self._present),
            paused=self._paused,
            rate_limit_kbps=self._rate_limit_kbps,
        )

    def _send_availability(self, info: str = "download availability") -> None:
        self.process_message_queue.put(
            HordeDownloadAvailabilityMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=info,
                available_model_names=sorted(self._present),
                currently_downloading=self._currently_downloading,
                pending_downloads=list(self._pending),
                failed_downloads=list(self._failed),
                status=self._status_snapshot(),
            ),
        )

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        if message.control_flag == HordeControlFlag.RELOAD_MODEL_DATABASE:
            # The fake holds no real model managers; a reference reload is a no-op here.
            return
        if not isinstance(message, HordeDownloadControlMessage):
            logger.warning(f"Fake download process received unexpected message: {type(message).__name__}")
            return
        if message.set_paused is not None:
            self._paused = message.set_paused
        if message.set_rate_limit_kbps is not None:
            self._rate_limit_kbps = message.set_rate_limit_kbps if message.set_rate_limit_kbps > 0 else None
        for model_name in message.model_names:
            if model_name in self._present or model_name in self._pending:
                continue
            self._pending.append(model_name)
        self._send_availability("download request received")

    @override
    def worker_cycle(self) -> None:
        if self._paused or not self._pending:
            return
        model_name = self._pending.pop(0)
        self._currently_downloading = model_name
        self._send_availability(f"downloading {model_name}")
        if self._download_delay_seconds > 0:
            time.sleep(self._download_delay_seconds)
        self._currently_downloading = None
        if model_name in self._fail_models:
            self._failed.append(model_name)
        else:
            self._present.add(model_name)
        self._send_availability(f"finished {model_name}")

    @override
    def cleanup_for_exit(self) -> None:
        return


def start_fake_download_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    process_launch_identifier: int,
    *,
    nsfw: bool = True,
    allow_lora: bool = False,
    allow_controlnet: bool = False,
    allow_sdxl_controlnet: bool = False,
    allow_post_processing: bool = True,
    purge_loras: bool = False,
    amd_gpu: bool = False,
    directml: int | None = None,
    rate_limit_kbps: int | None = None,
    paused: bool = False,
    scripted_present: list[str] | None = None,
    download_delay_seconds: float = 0.0,
    fail_models: list[str] | None = None,
) -> None:
    """Start a fake download process.

    Signature-compatible with ``worker_entry_points.start_download_process``; the worker-config
    arguments are accepted and ignored, except ``rate_limit_kbps``/``paused`` which the fake honors so
    the pause/throttle controls can be exercised. Inject the scripting arguments with
    ``functools.partial`` (partials of module-level functions stay picklable under spawn).
    """
    logger.remove()
    maybe_wait_for_process_debugger(process_id, "fake download")
    worker_process = FakeDownloadProcess(
        process_id=process_id,
        process_message_queue=process_message_queue,
        pipe_connection=pipe_connection,
        disk_lock=disk_lock,
        process_launch_identifier=process_launch_identifier,
        scripted_present=scripted_present,
        download_delay_seconds=download_delay_seconds,
        fail_models=fail_models,
        rate_limit_kbps=rate_limit_kbps,
        paused=paused,
    )
    worker_process.main_loop()
