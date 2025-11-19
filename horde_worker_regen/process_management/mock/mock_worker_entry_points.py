"""Entry point functions for starting mock processes.

These functions mirror the real worker entry points but launch mock processes
instead of real GPU-based processes.
"""

from __future__ import annotations

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore

from multiprocessing.synchronize import Lock, Semaphore

from loguru import logger

from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.mock.mock_config import MockConfig
from horde_worker_regen.process_management.mock.mock_inference_process import MockInferenceProcess
from horde_worker_regen.process_management.mock.mock_safety_process import MockSafetyProcess


def start_mock_inference_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    inference_semaphore: Semaphore,
    disk_lock: Lock,
    aux_model_lock: Lock,
    vae_decode_semaphore: Semaphore,
    process_launch_identifier: int,
    mock_config: MockConfig,
) -> None:
    """Start a mock inference process.

    This function starts a mock inference process that simulates image generation
    without requiring GPU hardware or heavy dependencies.

    Args:
        process_id: The ID of the process.
        process_message_queue: Queue for sending messages to main process.
        pipe_connection: Connection for receiving control messages.
        inference_semaphore: Semaphore for limiting concurrent inference.
        disk_lock: Lock for disk access.
        aux_model_lock: Lock for auxiliary model operations.
        vae_decode_semaphore: Semaphore for VAE decoding.
        process_launch_identifier: Unique identifier for this launch.
        mock_config: Configuration for mock behavior.
    """
    logger.info(f"Starting mock inference process {process_id}")
    logger.warning("⚠️  MOCK MODE: Using simulated inference (no GPU required)")
    logger.info(f"Mock config: {mock_config.to_dict()}")

    worker_process = MockInferenceProcess(
        process_id=process_id,
        process_message_queue=process_message_queue,
        pipe_connection=pipe_connection,
        inference_semaphore=inference_semaphore,
        vae_decode_semaphore=vae_decode_semaphore,
        aux_model_lock=aux_model_lock,
        disk_lock=disk_lock,
        process_launch_identifier=process_launch_identifier,
        mock_config=mock_config,
    )

    worker_process.main_loop()


def start_mock_safety_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    process_launch_identifier: int,
    mock_config: MockConfig,
    cpu_only: bool = True,
) -> None:
    """Start a mock safety process.

    This function starts a mock safety process that simulates NSFW/CSAM checking
    without loading actual detection models.

    Args:
        process_id: The ID of the process.
        process_message_queue: Queue for sending messages to main process.
        pipe_connection: Connection for receiving control messages.
        disk_lock: Lock for disk access.
        process_launch_identifier: Unique identifier for this launch.
        mock_config: Configuration for mock behavior.
        cpu_only: Whether to use CPU only (ignored in mock, for compatibility).
    """
    logger.info(f"Starting mock safety process {process_id}")
    logger.warning("⚠️  MOCK MODE: Using simulated safety checking (no models required)")
    logger.info(f"Mock config: {mock_config.to_dict()}")

    worker_process = MockSafetyProcess(
        process_id=process_id,
        process_message_queue=process_message_queue,
        pipe_connection=pipe_connection,
        disk_lock=disk_lock,
        process_launch_identifier=process_launch_identifier,
        mock_config=mock_config,
        cpu_only=cpu_only,
    )

    worker_process.main_loop()
