"""Mock safety process for GPU-free testing.

This module provides a mock implementation of the safety process that simulates
safety checking without requiring actual NSFW/CSAM detection models.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore

from multiprocessing.synchronize import Lock

from loguru import logger
from typing_extensions import override

from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.horde_process import HordeProcess, HordeProcessType
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeProcessState,
    HordeSafetyControlMessage,
    HordeSafetyEvaluation,
    HordeSafetyResultMessage,
)
from horde_worker_regen.process_management.mock.mock_config import MockConfig
from horde_worker_regen.process_management.mock.mock_data_generator import (
    generate_fake_csam_score,
    generate_fake_nsfw_score,
)

if TYPE_CHECKING:
    pass


class MockSafetyProcess(HordeProcess):
    """Mock safety process that simulates safety evaluation without real models.

    This process mimics the behavior of HordeSafetyProcess, performing fake
    NSFW and CSAM checks on images. Perfect for testing without loading
    actual safety detection models.
    """

    process_type = HordeProcessType.SAFETY

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        mock_config: MockConfig,
        cpu_only: bool = True,
    ) -> None:
        """Initialize the mock safety process.

        Args:
            process_id: The ID of the process.
            process_message_queue: Queue for sending messages to main process.
            pipe_connection: Connection for receiving control messages.
            disk_lock: Lock for disk access.
            process_launch_identifier: Unique identifier for this launch.
            mock_config: Configuration for mock behavior.
            cpu_only: Whether to use CPU only (ignored in mock, for compatibility).
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
        )

        self.mock_config = mock_config
        self._evaluations_completed = 0

        logger.info(f"MockSafetyProcess initialized (cpu_only={cpu_only})")

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Mock safety process ready",
        )

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Handle control messages from main process.

        Args:
            message: The control message to handle.
        """
        if message.control_flag == HordeControlFlag.EVALUATE_SAFETY:
            self._handle_evaluate_safety(message)
        else:
            logger.warning(f"Unknown control flag: {message.control_flag}")

    def _handle_evaluate_safety(self, message: HordeSafetyControlMessage) -> None:
        """Simulate safety evaluation of images.

        Args:
            message: The safety control message.
        """
        job = message.job
        num_images = len(job.job_image_results)

        logger.info(f"Starting mock safety evaluation for {num_images} images from job {job.sdk_api_job_info.id_}")

        self.send_process_state_change_message(
            process_state=HordeProcessState.EVALUATING_SAFETY,
            info=f"Evaluating {num_images} images",
        )

        # Send heartbeat
        self.send_heartbeat_message(
            heartbeat_type=HordeHeartbeatType.OTHER,
        )

        # Simulate safety check time per image
        check_time = self.mock_config.safety_check_time_seconds
        check_time /= self.mock_config.speed_multiplier

        time.sleep(check_time * num_images)

        # Generate fake safety evaluations
        evaluations = []
        for i, image_result in enumerate(job.job_image_results):
            nsfw_score = generate_fake_nsfw_score()
            csam_score = generate_fake_csam_score()

            # Determine if image should be censored (threshold around 0.7)
            is_nsfw = nsfw_score > 0.7
            is_csam = csam_score > 0.5  # Very conservative threshold

            evaluation = HordeSafetyEvaluation(
                image_index=i,
                nsfw_score=nsfw_score,
                csam_score=csam_score,
                is_nsfw=is_nsfw,
                is_csam=is_csam,
                replacement_image_base64=None,  # In mock mode, don't actually censor
            )
            evaluations.append(evaluation)

            logger.debug(
                f"Mock safety eval: image {i}: nsfw={nsfw_score:.2f}, csam={csam_score:.2f}, "
                f"flagged_nsfw={is_nsfw}, flagged_csam={is_csam}",
            )

        # Send result
        self.process_message_queue.put(
            HordeSafetyResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info="Safety evaluation complete",
                time_elapsed=check_time * num_images,
                safety_evaluations=evaluations,
                job_sdk_api_job_info=job.sdk_api_job_info,
                job_job_image_results=job.job_image_results,
            ),
        )

        self._evaluations_completed += 1
        logger.info(f"Mock safety evaluation complete: {num_images} images ({self._evaluations_completed} total)")

        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Safety evaluation complete",
        )

    @override
    def send_memory_report_message(self, include_vram: bool = False) -> bool:
        """Send simulated memory report.

        Args:
            include_vram: Whether to include VRAM usage (ignored in CPU-only mode).

        Returns:
            True if successful.
        """
        from horde_worker_regen.process_management.messages import HordeProcessMemoryMessage

        # Safety process uses much less memory
        ram_usage = int(self.mock_config.mock_ram_usage_mb * 0.25 * 1024 * 1024)  # 25% of inference RAM

        message = HordeProcessMemoryMessage(
            process_id=self.process_id,
            process_launch_identifier=self.process_launch_identifier,
            info="Memory report",
            time_elapsed=None,
            ram_usage_bytes=ram_usage,
            vram_usage_bytes=0,  # Safety typically runs on CPU
            vram_total_bytes=0,
        )

        self.process_message_queue.put(message)
        return True

    @override
    def cleanup_for_exit(self) -> None:
        """Cleanup before exit."""
        logger.info("Mock safety process cleanup")
        # Nothing to clean up in mock mode

    @override
    def worker_cycle(self) -> None:
        """Worker cycle - called repeatedly in main loop."""
        # Send periodic memory reports
        if not hasattr(self, "_last_memory_report"):
            self._last_memory_report = time.time()

        if time.time() - self._last_memory_report > self._memory_report_interval:
            self.send_memory_report_message(include_vram=False)
            self._last_memory_report = time.time()
