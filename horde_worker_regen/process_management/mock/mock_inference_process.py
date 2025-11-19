"""Mock inference process for GPU-free testing.

This module provides a mock implementation of the inference process that simulates
realistic worker behavior without requiring GPU hardware or heavy dependencies.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore

from multiprocessing.synchronize import Lock, Semaphore

from loguru import logger
from typing_extensions import override

from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.horde_process import HordeProcess, HordeProcessType
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
    ModelLoadState,
)
from horde_worker_regen.process_management.mock.mock_config import MockConfig
from horde_worker_regen.process_management.mock.mock_data_generator import (
    calculate_mock_inference_time,
    calculate_mock_kudos,
    generate_fake_image,
)

if TYPE_CHECKING:
    from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse


class MockInferenceProcess(HordeProcess):
    """Mock inference process that simulates image generation without GPU.

    This process mimics the behavior of HordeInferenceProcess, sending the same
    message sequences and state transitions, but without actually loading models
    or performing inference. Perfect for testing UI and orchestration logic.
    """

    process_type = HordeProcessType.INFERENCE

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        inference_semaphore: Semaphore,
        vae_decode_semaphore: Semaphore,
        aux_model_lock: Lock,
        disk_lock: Lock,
        process_launch_identifier: int,
        mock_config: MockConfig,
    ) -> None:
        """Initialize the mock inference process.

        Args:
            process_id: The ID of the process.
            process_message_queue: Queue for sending messages to main process.
            pipe_connection: Connection for receiving control messages.
            inference_semaphore: Semaphore for limiting concurrent inference.
            vae_decode_semaphore: Semaphore for limiting VAE decoding.
            aux_model_lock: Lock for auxiliary model operations.
            disk_lock: Lock for disk access.
            process_launch_identifier: Unique identifier for this launch.
            mock_config: Configuration for mock behavior.
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
        )

        self._inference_semaphore = inference_semaphore
        self._vae_decode_semaphore = vae_decode_semaphore
        self._aux_model_lock = aux_model_lock
        self.mock_config = mock_config

        # Track state
        self._loaded_model_name: str | None = None
        self._jobs_completed = 0
        self._pending_download: str | None = None
        self._pending_preload: str | None = None
        self._pending_inference: ImageGenerateJobPopResponse | None = None

        logger.info(f"MockInferenceProcess initialized with config: {mock_config.to_dict()}")

        # Check if we should get stuck after N jobs
        if mock_config.enable_stuck_simulation and mock_config.stuck_after_jobs:
            logger.warning(f"Process will get stuck after {mock_config.stuck_after_jobs} jobs (testing mode)")

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Mock inference process ready",
        )

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Handle control messages from main process.

        Args:
            message: The control message to handle.
        """
        if message.control_flag == HordeControlFlag.DOWNLOAD_MODEL:
            self._handle_download_model(message)
        elif message.control_flag == HordeControlFlag.PRELOAD_MODEL:
            self._handle_preload_model(message)
        elif message.control_flag == HordeControlFlag.START_INFERENCE:
            self._handle_start_inference(message)
        elif message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
            self._handle_unload_vram()
        elif message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            self._handle_unload_ram()
        else:
            logger.warning(f"Unknown control flag: {message.control_flag}")

    def _handle_download_model(self, message: HordeControlMessage) -> None:
        """Simulate model download with progress updates.

        Args:
            message: The download control message.
        """
        model_name = message.horde_model_name
        logger.info(f"Starting mock download of {model_name}")

        # Check for download failure simulation
        if self.mock_config.enable_download_failures:
            if random.random() < self.mock_config.download_failure_rate:
                logger.error(f"Simulated download failure for {model_name}")
                self.send_process_state_change_message(
                    process_state=HordeProcessState.WAITING_FOR_JOB,
                    info=f"Download failed: {model_name}",
                )
                return

        self.send_process_state_change_message(
            process_state=HordeProcessState.DOWNLOADING_MODEL,
            info=f"Downloading {model_name}",
        )

        # Get model size from config
        model_size_mb = self._get_model_size(model_name)

        # Calculate download time based on speed
        download_time = model_size_mb / self.mock_config.download_speed_mbps * 8 / 1000  # Convert to seconds
        download_time /= self.mock_config.speed_multiplier

        # Send progress updates
        steps = 10
        for i in range(steps + 1):
            progress = (i / steps) * 100
            # Send model state change with progress
            self.process_message_queue.put(
                HordeModelStateChangeMessage(
                    process_id=self.process_id,
                    process_launch_identifier=self.process_launch_identifier,
                    info=f"Downloading {model_name}: {progress:.0f}%",
                    horde_model_name=model_name,
                    horde_model_state=ModelLoadState.DOWNLOADING,
                ),
            )
            time.sleep(download_time / steps)

        # Download complete
        self.process_message_queue.put(
            HordeModelStateChangeMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Download complete: {model_name}",
                horde_model_name=model_name,
                horde_model_state=ModelLoadState.ON_DISK,
            ),
        )

        logger.info(f"Mock download complete: {model_name}")
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Download complete",
        )

    def _handle_preload_model(self, message: HordePreloadInferenceModelMessage) -> None:
        """Simulate model preloading into memory.

        Args:
            message: The preload control message.
        """
        model_name = message.horde_model_name
        logger.info(f"Starting mock preload of {model_name}")

        self.send_process_state_change_message(
            process_state=HordeProcessState.PRELOADING_MODEL,
            info=f"Preloading {model_name}",
        )

        # Send model loading state
        self.process_message_queue.put(
            HordeModelStateChangeMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Loading {model_name} to RAM",
                horde_model_name=model_name,
                horde_model_state=ModelLoadState.LOADING,
            ),
        )

        # Calculate load time
        load_time = self._get_model_load_time(model_name)
        load_time /= self.mock_config.speed_multiplier

        # Simulate loading to RAM
        time.sleep(load_time * 0.3)
        self.process_message_queue.put(
            HordeModelStateChangeMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Loaded {model_name} to RAM",
                horde_model_name=model_name,
                horde_model_state=ModelLoadState.LOADED_IN_RAM,
            ),
        )

        # Simulate loading to VRAM
        time.sleep(load_time * 0.7)
        self.process_message_queue.put(
            HordeModelStateChangeMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Loaded {model_name} to VRAM",
                horde_model_name=model_name,
                horde_model_state=ModelLoadState.LOADED_IN_VRAM,
            ),
        )

        self._loaded_model_name = model_name

        logger.info(f"Mock preload complete: {model_name}")
        self.send_process_state_change_message(
            process_state=HordeProcessState.PRELOADED_MODEL,
            info=f"Model preloaded: {model_name}",
        )

    def _handle_start_inference(self, message: HordeInferenceControlMessage) -> None:
        """Simulate inference job execution.

        Args:
            message: The inference control message.
        """
        job = message.job

        logger.info(f"Starting mock inference for job {job.id_}")

        # Check for failure simulation
        if self.mock_config.enable_failures:
            if random.random() < self.mock_config.failure_rate:
                failure_type = random.choice(self.mock_config.failure_types)
                logger.error(f"Simulated inference failure: {failure_type}")

                # Send fault result
                self.process_message_queue.put(
                    HordeInferenceResultMessage(
                        process_id=self.process_id,
                        process_launch_identifier=self.process_launch_identifier,
                        info=f"Inference failed: {failure_type}",
                        job_image_results=[],
                        generation_faults=[failure_type],
                        time_elapsed=1.0,
                        sdk_api_job_info=job,
                    ),
                )
                self.send_process_state_change_message(
                    process_state=HordeProcessState.WAITING_FOR_JOB,
                    info="Inference failed",
                )
                return

        # Determine if this job should be slow
        slowdown = 1.0
        if self.mock_config.enable_slowdowns:
            if random.random() < self.mock_config.slowdown_rate:
                slowdown = self.mock_config.slowdown_multiplier
                logger.info(f"Simulating slow job ({slowdown}x slower)")

        self.send_process_state_change_message(
            process_state=HordeProcessState.INFERENCE_STARTING,
            info="Starting inference",
        )

        # Acquire semaphore (simulated)
        self._inference_semaphore.acquire()

        try:
            # Calculate inference time
            inference_time = calculate_mock_inference_time(
                width=job.payload.width or 512,
                height=job.payload.height or 512,
                steps=job.payload.ddim_steps or 20,
                speed_multiplier=self.mock_config.speed_multiplier,
                slowdown_multiplier=slowdown,
            )

            steps = job.payload.ddim_steps or 20
            time_per_step = inference_time / steps

            # Send heartbeats with progress
            for step in range(steps + 1):
                percent = int((step / steps) * 100)
                self.send_heartbeat_message(
                    heartbeat_type=HordeHeartbeatType.INFERENCE_STEP,
                    percent_complete=percent,
                )
                time.sleep(time_per_step)

            # Post-processing
            self.send_process_state_change_message(
                process_state=HordeProcessState.INFERENCE_POST_PROCESSING,
                info="Post-processing",
            )
            time.sleep(0.5 / self.mock_config.speed_multiplier)

            # Generate fake images
            num_images = job.payload.n or 1
            image_results = []
            for i in range(num_images):
                fake_image = generate_fake_image(
                    width=job.payload.width or 512,
                    height=job.payload.height or 512,
                    job_id=str(job.id_),
                    model_name=job.model,
                    seed=(job.payload.seed or 0) + i,
                    steps=steps,
                )
                image_results.append(
                    HordeImageResult(
                        image_base64=fake_image,
                        seed=(job.payload.seed or 0) + i,
                        generation_faults=[],
                    ),
                )

            # Send result
            self.process_message_queue.put(
                HordeInferenceResultMessage(
                    process_id=self.process_id,
                    process_launch_identifier=self.process_launch_identifier,
                    info="Inference complete",
                    job_image_results=image_results,
                    generation_faults=[],
                    time_elapsed=inference_time,
                    sdk_api_job_info=job,
                ),
            )

            self._jobs_completed += 1
            logger.info(f"Mock inference complete: job {job.id_} ({self._jobs_completed} total)")

            self.send_process_state_change_message(
                process_state=HordeProcessState.INFERENCE_COMPLETE,
                info="Inference complete",
            )

            # Check if we should get stuck
            if self.mock_config.enable_stuck_simulation:
                if self.mock_config.stuck_after_jobs and self._jobs_completed >= self.mock_config.stuck_after_jobs:
                    logger.warning("Simulating stuck process (will stop responding)")
                    # Infinite loop to simulate stuck process
                    while True:
                        time.sleep(1.0)

        finally:
            self._inference_semaphore.release()

        # Return to waiting
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Ready for next job",
        )

    def _handle_unload_vram(self) -> None:
        """Simulate unloading models from VRAM."""
        if self._loaded_model_name:
            logger.info(f"Mock unload from VRAM: {self._loaded_model_name}")
            self.process_message_queue.put(
                HordeModelStateChangeMessage(
                    process_id=self.process_id,
                    process_launch_identifier=self.process_launch_identifier,
                    info=f"Unloaded {self._loaded_model_name} from VRAM",
                    horde_model_name=self._loaded_model_name,
                    horde_model_state=ModelLoadState.LOADED_IN_RAM,
                ),
            )

    def _handle_unload_ram(self) -> None:
        """Simulate unloading models from RAM."""
        if self._loaded_model_name:
            logger.info(f"Mock unload from RAM: {self._loaded_model_name}")
            self.process_message_queue.put(
                HordeModelStateChangeMessage(
                    process_id=self.process_id,
                    process_launch_identifier=self.process_launch_identifier,
                    info=f"Unloaded {self._loaded_model_name} from RAM",
                    horde_model_name=self._loaded_model_name,
                    horde_model_state=ModelLoadState.ON_DISK,
                ),
            )
            self._loaded_model_name = None

    def _get_model_size(self, model_name: str) -> float:
        """Get simulated model size in MB.

        Args:
            model_name: Name of the model.

        Returns:
            Size in megabytes.
        """
        model_lower = model_name.lower()
        for key, size in self.mock_config.model_download_size_mb.items():
            if key in model_lower:
                return size
        return 2000.0  # Default 2GB

    def _get_model_load_time(self, model_name: str) -> float:
        """Get simulated model load time in seconds.

        Args:
            model_name: Name of the model.

        Returns:
            Load time in seconds.
        """
        model_lower = model_name.lower()
        for key, load_time in self.mock_config.model_load_time_seconds.items():
            if key in model_lower:
                return load_time
        return 3.0  # Default 3 seconds

    @override
    def send_memory_report_message(self, include_vram: bool = False) -> bool:
        """Send simulated memory report.

        Args:
            include_vram: Whether to include VRAM usage.

        Returns:
            True if successful.
        """
        from horde_worker_regen.process_management.messages import HordeProcessMemoryMessage

        # Simulate some memory fluctuation
        ram_base = self.mock_config.mock_ram_usage_mb * 1024 * 1024
        vram_base = self.mock_config.mock_vram_usage_mb * 1024 * 1024

        if self.mock_config.simulate_memory_fluctuation:
            ram_usage = int(ram_base * random.uniform(0.8, 1.1))
            vram_usage = int(vram_base * random.uniform(0.85, 1.05))
        else:
            ram_usage = ram_base
            vram_usage = vram_base

        message = HordeProcessMemoryMessage(
            process_id=self.process_id,
            process_launch_identifier=self.process_launch_identifier,
            info="Memory report",
            time_elapsed=None,
            ram_usage_bytes=ram_usage,
            vram_usage_bytes=vram_usage if include_vram else 0,
            vram_total_bytes=self.mock_config.mock_vram_usage_mb * 1024 * 1024 if include_vram else 0,
        )

        self.process_message_queue.put(message)
        return True

    @override
    def cleanup_for_exit(self) -> None:
        """Cleanup before exit."""
        logger.info("Mock inference process cleanup")
        # Nothing to clean up in mock mode

    @override
    def worker_cycle(self) -> None:
        """Worker cycle - called repeatedly in main loop."""
        # Send periodic memory reports
        if not hasattr(self, "_last_memory_report"):
            self._last_memory_report = time.time()

        if time.time() - self._last_memory_report > self._memory_report_interval:
            self.send_memory_report_message(include_vram=True)
            self._last_memory_report = time.time()
