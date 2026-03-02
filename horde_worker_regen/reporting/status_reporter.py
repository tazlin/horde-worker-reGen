"""Worker status reporting."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from loguru import logger

import horde_worker_regen

if TYPE_CHECKING:
    from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, UserDetailsResponse

    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.device_info import TorchDeviceMap
    from horde_worker_regen.process_management.job_models import APIWorkerMessage


class StatusReporter:
    """Handles periodic status reporting for the worker."""

    def __init__(
        self,
        last_status_message_time: float,
        status_message_frequency: float,
    ) -> None:
        """Initialize the status reporter.

        Args:
            last_status_message_time: The epoch time of the last status message.
            status_message_frequency: The frequency (in seconds) at which to print status messages.
        """
        self.last_status_message_time = last_status_message_time
        self.status_message_frequency = status_message_frequency

    def should_print_status(self, last_pop_maintenance_mode: bool) -> bool:
        """Check if it's time to print status.

        Args:
            last_pop_maintenance_mode: Whether the last job pop was in maintenance mode.

        Returns:
            True if status should be printed, False otherwise.
        """
        if last_pop_maintenance_mode:
            return False

        cur_time = time.time()
        return cur_time - self.last_status_message_time > self.status_message_frequency

    def print_status(
        self,
        bridge_data: reGenBridgeData,
        process_info_strings: list[str],
        api_messages_received: dict[str, APIWorkerMessage],
        jobs_pending_inference: Sequence[ImageGenerateJobPopResponse],
        active_models: set[str],
        pending_megapixelsteps: int,
        num_jobs_total: int,
        total_num_completed_jobs: int,
        num_jobs_faulted: int,
        num_job_slowdowns: int,
        num_process_recoveries: int,
        time_spent_no_jobs_available: float,
        user_info: UserDetailsResponse | None,
        max_concurrent_inference_processes: int,
        device_map: TorchDeviceMap,
        too_many_consecutive_failed_jobs: bool,
        too_many_consecutive_failed_jobs_time: float,
        too_many_consecutive_failed_jobs_wait_time: float,
        session_start_time: float,
        shutting_down: bool,
        jobs_pending_safety_check: int,
        jobs_being_safety_checked: int,
        jobs_in_progress: int,
        total_ram_gigabytes: int,
    ) -> float:
        """Print the status of the worker.

        Args:
            bridge_data: The bridge data configuration.
            process_info_strings: List of process info strings from ProcessMap.
            api_messages_received: Dict of API messages received.
            jobs_pending_inference: List of jobs pending inference.
            active_models: Set of currently loaded model names.
            pending_megapixelsteps: Number of pending megapixelsteps.
            num_jobs_total: Total number of jobs popped.
            total_num_completed_jobs: Total number of jobs submitted.
            num_jobs_faulted: Number of faulted jobs.
            num_job_slowdowns: Number of slow jobs.
            num_process_recoveries: Number of process recoveries.
            time_spent_no_jobs_available: Time spent without jobs available.
            user_info: User information from the API.
            max_concurrent_inference_processes: Maximum concurrent inference processes.
            device_map: Device map.
            too_many_consecutive_failed_jobs: Whether too many consecutive jobs have failed.
            too_many_consecutive_failed_jobs_time: Time of the last consecutive job failure.
            too_many_consecutive_failed_jobs_wait_time: Wait time after consecutive failures.
            session_start_time: Session start time.
            shutting_down: Whether the worker is shutting down.
            jobs_pending_safety_check: Number of jobs pending safety check.
            jobs_being_safety_checked: Number of jobs being safety checked.
            jobs_in_progress: Number of jobs in progress.
            total_ram_gigabytes: Total RAM in gigabytes.

        Returns:
            The updated status message frequency.
        """
        AIWORKER_LIMITED_CONSOLE_MESSAGES = os.getenv("AIWORKER_LIMITED_CONSOLE_MESSAGES", False)

        logging_function = logger.opt(ansi=True).info

        if AIWORKER_LIMITED_CONSOLE_MESSAGES:
            logging_function = logger.opt(ansi=True).success

        # Print header
        logging_function("<fg #dddddd>" + str("^" * 80) + "</>")

        # Print API messages
        self._print_api_messages(logging_function, api_messages_received)

        # Print process info
        logging_function("<b>Process info:</b>")
        for process_info_string in process_info_strings:
            logging_function("  " + process_info_string)

        logging_function("<fg #7b7d7d>" + str("-" * 40) + "</>")

        # Print job info
        self._print_job_info(
            logging_function,
            jobs_pending_inference,
            active_models,
            pending_megapixelsteps,
            num_jobs_total,
            total_num_completed_jobs,
            num_jobs_faulted,
            num_job_slowdowns,
            num_process_recoveries,
            time_spent_no_jobs_available,
        )

        logging_function("<fg #7b7d7d>" + str("-" * 40) + "</>")

        # Print worker info
        self._print_worker_info(
            bridge_data,
            user_info,
            max_concurrent_inference_processes,
            jobs_pending_safety_check,
            jobs_being_safety_checked,
            jobs_in_progress,
        )

        # Print warnings
        self._print_warnings(
            bridge_data,
            device_map,
            too_many_consecutive_failed_jobs,
            too_many_consecutive_failed_jobs_time,
            too_many_consecutive_failed_jobs_wait_time,
            time_spent_no_jobs_available,
            session_start_time,
            total_ram_gigabytes,
        )

        # Print shutdown message
        updated_frequency = self.status_message_frequency
        if shutting_down:
            logger.warning("*" * 80)
            logger.warning("Shutting down after current jobs are finished...")
            updated_frequency = 5.0
            logger.warning("*" * 80)

        self.last_status_message_time = time.time()
        logging_function("<fg #dddddd>" + str("v" * 80) + "</>")

        return updated_frequency

    def _print_api_messages(
        self,
        logging_function: Callable[..., None],
        api_messages_received: dict[str, APIWorkerMessage],
    ) -> None:
        """Print API messages if any."""
        if len(api_messages_received) > 0:
            logging_function("<b>API Messages:</b>")
            for message_id, message in api_messages_received.items():
                try:
                    message_text = message.message_text or ""
                    log_safe_message = message_text.replace("<", "&lt;").replace(">", "&gt;")
                    log_safe_message = log_safe_message.replace("\n", " ")
                    log_safe_message = log_safe_message.replace("\r", " ")
                    log_safe_message = log_safe_message.replace("\t", " ")
                    log_safe_message = log_safe_message.replace("{", "{{").replace("}", "}}")
                    log_safe_message = log_safe_message.replace('"', "'")
                    log_safe_message = log_safe_message.replace("'", "'")

                    logging_function(
                        f"  <fg #000><bg #0ff127>{log_safe_message} "
                        f"(from {message.message_origin}, expires {message.message_expiry}, "
                        f"message_id: {message_id[:8]})</></>",
                        "</></>",
                    )
                except Exception as e:
                    logger.warning(f"Failed to print API message: {e}")

    def _print_job_info(
        self,
        logging_function: Callable[..., None],
        jobs_pending_inference: Sequence[ImageGenerateJobPopResponse],
        active_models: set[str],
        pending_megapixelsteps: int,
        num_jobs_total: int,
        total_num_completed_jobs: int,
        num_jobs_faulted: int,
        num_job_slowdowns: int,
        num_process_recoveries: int,
        time_spent_no_jobs_available: float,
    ) -> None:
        """Print job information."""
        logging_function("<b>Job Info:</b>")
        jobs = []
        for x in jobs_pending_inference:
            shortened_id = str(x.id_.root)[:8] if x.id_ is not None else "None?"
            jobs.append(f"<{shortened_id}: <u>{x.model}></u>")

        logging_function(f"  Jobs: {', '.join(jobs)}")

        logger.debug(f"Active models: {active_models}")

        job_info_message = "  Session job info: " + " | ".join(
            [
                f"pending start: {len(jobs_pending_inference)} (eMPS: {pending_megapixelsteps})",
                f"jobs popped: {num_jobs_total}",
                f"submitted: {total_num_completed_jobs}",
                f"faulted: {num_jobs_faulted}",
                f"slow_jobs: {num_job_slowdowns}",
                f"process_recoveries: {num_process_recoveries}",
                f"{time_spent_no_jobs_available:.2f} seconds without jobs",
            ],
        )

        logging_function(
            f"<fg #7dcea0>{job_info_message}</>",
        )

    def _print_worker_info(
        self,
        bridge_data: reGenBridgeData,
        user_info: UserDetailsResponse | None,
        max_concurrent_inference_processes: int,
        jobs_pending_safety_check: int,
        jobs_being_safety_checked: int,
        jobs_in_progress: int,
    ) -> None:
        """Print worker information."""
        logger.opt(ansi=True).info("<b>Worker Info:</b>")

        max_power_dimension = int(math.sqrt(bridge_data.max_power * 8 * 64 * 64))
        logger.info(
            "  "
            + " | ".join(
                [
                    f"dreamer_name: {bridge_data.dreamer_worker_name}",
                    f"(v{horde_worker_regen.__version__})",
                    f"horde user: {user_info.username if user_info is not None else 'Unknown'}",
                    f"num_models: {len(bridge_data.image_models_to_load)}",
                    f"custom_models: {bool(bridge_data.custom_models)}",
                    f"max_power: {bridge_data.max_power} ({max_power_dimension}x{max_power_dimension})",
                    f"max_threads: {max_concurrent_inference_processes}",
                    f"queue_size: {bridge_data.queue_size}",
                    f"safety_on_gpu: {bridge_data.safety_on_gpu}",
                ],
            ),
        )
        logger.info(
            "  "
            + " | ".join(
                [
                    f"allow_img2img: {bridge_data.allow_img2img}",
                    f"allow_lora: {bridge_data.allow_lora}",
                    f"allow_controlnet: {bridge_data.allow_controlnet}",
                    f"allow_sdxl_controlnet: {bridge_data.allow_sdxl_controlnet}",
                    f"allow_post_processing: {bridge_data.allow_post_processing}",
                    f"post_process_job_overlap: {bridge_data.post_process_job_overlap}",
                ],
            ),
        )

        logger.info(
            "  "
            + " | ".join(
                [
                    f"unload_models_from_vram_often: {bridge_data.unload_models_from_vram_often}",
                    f"high_performance_mode: {bridge_data.high_performance_mode}",
                    f"moderate_performance_mode: {bridge_data.moderate_performance_mode}",
                    f"high_memory_mode: {bridge_data.high_memory_mode}",
                ],
            ),
        )

        logger.debug(
            " | ".join(
                [
                    f"preload_timeout: {bridge_data.preload_timeout}",
                    f"download_timeout: {bridge_data.download_timeout}",
                    f"post_process_timeout: {bridge_data.post_process_timeout}",
                    f"very_high_memory_mode: {bridge_data.very_high_memory_mode}",
                    f"cycle_process_on_model_change: {bridge_data.cycle_process_on_model_change}",
                    f"exit_on_unhandled_faults: {bridge_data.exit_on_unhandled_faults}",
                    f"jobs_pending_safety_check: {jobs_pending_safety_check}",
                    f"jobs_being_safety_checked: {jobs_being_safety_checked}",
                    f"jobs_in_progress: {jobs_in_progress}",
                ],
            ),
        )

    def _print_warnings(
        self,
        bridge_data: reGenBridgeData,
        device_map: TorchDeviceMap,
        too_many_consecutive_failed_jobs: bool,
        too_many_consecutive_failed_jobs_time: float,
        too_many_consecutive_failed_jobs_wait_time: float,
        time_spent_no_jobs_available: float,
        session_start_time: float,
        total_ram_gigabytes: int,
    ) -> None:
        """Print various warnings based on worker state."""
        # Version warnings
        if os.getenv("AIWORKER_NOT_REQUIRED_VERSION"):
            logger.warning(
                "There is a required update available for the AI Worker. `git pull` and `update-runtime` to update.",
            )
        elif os.getenv("AIWORKER_NOT_RECOMMENDED_VERSION"):
            logger.warning(
                "There is a recommended update available for the AI Worker. "
                "`git pull` and `update-runtime` to update.",
            )

        # Extra slow worker warnings
        if bridge_data.extra_slow_worker:
            if not bridge_data.limit_max_steps:
                logger.warning(
                    "Extra slow worker mode is enabled, but limit_max_steps is not enabled. "
                    "Consider enabling limit_max_steps to prevent long running jobs.",
                )
            if bridge_data.max_batch > 1:
                logger.warning(
                    "Extra slow worker mode is enabled, but max_batch is greater than 1. "
                    "Consider setting max_batch to 1 to prevent long running batch jobs.",
                )
            if bridge_data.allow_sdxl_controlnet:
                logger.warning(
                    "Extra slow worker mode is enabled, but allow_sdxl_controlnet is enabled. "
                    "Consider disabling allow_sdxl_controlnet to prevent long running jobs.",
                )

        # Device memory warnings
        for device in device_map.root.values():
            total_memory_mb = device.total_memory / 1024 / 1024
            if total_memory_mb < 10_000 and bridge_data.high_memory_mode:
                logger.warning(
                    f"Device {device.device_name} ({device.device_index}) has less than 10GB of memory. "
                    "This may cause issues with `high_memory_mode` enabled.",
                )
            elif (
                total_memory_mb > 20_000
                and not bridge_data.high_memory_mode
                and bridge_data.max_threads == 1
                and total_ram_gigabytes > 32
            ):
                logger.warning(
                    f"Device {device.device_name} ({device.device_index}) has more than 20GB of memory. "
                    "You should enable `high_memory_mode` in your config to take advantage of this.",
                )
            elif total_memory_mb > 20_000 and bridge_data.extra_slow_worker:
                logger.warning(
                    f"Device {device.device_name} ({device.device_index}) has more than 20GB of memory. "
                    "There are very few GPUs with this much memory that should be running in extra slow worker "
                    "mode. Consider disabling `extra_slow_worker` in your config.",
                )

        # Consecutive failure warning
        if too_many_consecutive_failed_jobs:
            cur_time = time.time()
            time_since_failure = cur_time - too_many_consecutive_failed_jobs_time
            logger.error(
                "Too many consecutive failed jobs. This may be due to a misconfiguration or other issue. "
                "Please check your logs and configuration.",
            )
            logger.error(
                f"Time since last job failure: {time_since_failure:.2f}s. "
                f"{too_many_consecutive_failed_jobs_wait_time} seconds must pass before resuming.",
            )

        # No jobs warning
        minutes_allowed_without_jobs = bridge_data.minutes_allowed_without_jobs
        seconds_allowed_without_jobs = minutes_allowed_without_jobs * 60
        cur_time = time.time()
        cur_session_minutes = (cur_time - session_start_time) / 60
        if time_spent_no_jobs_available > seconds_allowed_without_jobs:
            if not bridge_data.suppress_speed_warnings:
                logger.warning(
                    f"Your worker spent more than {minutes_allowed_without_jobs} minutes combined throughout this "
                    f"session ({time_spent_no_jobs_available / 60:.2f}/{cur_session_minutes:.2f} minutes) "
                    "without jobs. This may be due to low demand. However, offering more models or increasing "
                    "your max_power may help increase the number of jobs you receive and reduce downtime.",
                )
            else:
                logger.debug(
                    f"Suppressed warning about time spent without jobs for {minutes_allowed_without_jobs} minutes",
                )
