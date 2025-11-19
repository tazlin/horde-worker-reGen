"""Data provider for the TUI to access worker process manager data."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from horde_worker_regen.process_management.messages import HordeProcessState, ModelLoadState

if TYPE_CHECKING:
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager


class TUIDataProvider:
    """Provides data from the HordeWorkerProcessManager to the TUI."""

    def __init__(self, process_manager: HordeWorkerProcessManager) -> None:
        """Initialize the data provider.

        Args:
            process_manager: The HordeWorkerProcessManager instance to monitor.
        """
        self.process_manager = process_manager
        self._log_buffer: deque[tuple[datetime, str, str]] = deque(maxlen=1000)  # (timestamp, level, message)
        self._last_update = datetime.now()

    def get_worker_status(self) -> dict:
        """Get overall worker status information."""
        pm = self.process_manager
        uptime = datetime.now() - datetime.fromtimestamp(pm.session_start_time)

        # Calculate uptime percentage
        total_seconds = uptime.total_seconds()
        idle_seconds = pm._time_spent_no_jobs_available
        uptime_percentage = ((total_seconds - idle_seconds) / total_seconds * 100) if total_seconds > 0 else 0

        status = "Running"
        if pm._shutting_down:
            status = "Shutting Down"
        elif pm._process_map.is_any_process_state(HordeProcessState.WAITING_FOR_JOB):
            status = "Idle - Waiting for Jobs"
        elif pm._too_many_consecutive_failed_jobs:
            status = "Paused - Too Many Failures"

        return {
            "status": status,
            "uptime": str(uptime).split(".")[0],  # Remove microseconds
            "uptime_percentage": round(uptime_percentage, 1),
            "session_start": datetime.fromtimestamp(pm.session_start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "shutting_down": pm._shutting_down,
            "maintenance_mode": getattr(pm, "_maintenance_mode", False),
            "worker_name": pm.bridge_data.worker_name if pm.bridge_data else "Unknown",
        }

    def get_process_info(self) -> list[dict]:
        """Get information about all worker processes."""
        processes = []
        pm = self.process_manager

        for process_id, proc_info in pm._process_map.items():
            # Calculate progress percentage
            progress = 0
            if proc_info.last_received_heartbeat:
                progress = proc_info.last_received_heartbeat.progress_percent or 0

            # Format model name
            model_name = "No model loaded"
            if proc_info.loaded_horde_model_name:
                model_name = proc_info.loaded_horde_model_name

            # Format memory usage
            ram_usage = "N/A"
            vram_usage = "N/A"
            if proc_info.last_received_memory_report:
                mem = proc_info.last_received_memory_report
                ram_usage = f"{mem.ram_usage_bytes / 1024**3:.2f} GB"
                if mem.vram_usage_bytes is not None:
                    vram_usage = f"{mem.vram_usage_bytes / 1024**3:.2f} GB"

            processes.append(
                {
                    "id": process_id,
                    "state": proc_info.process_state.name if proc_info.process_state else "UNKNOWN",
                    "model": model_name,
                    "model_state": proc_info.loaded_horde_model_state.name
                    if proc_info.loaded_horde_model_state
                    else "NONE",
                    "progress": progress,
                    "ram_usage": ram_usage,
                    "vram_usage": vram_usage,
                    "is_safety_process": proc_info.is_safety_process,
                    "job_id": str(proc_info.last_job_info.sdk_api_job_info.id_)
                    if proc_info.last_job_info and proc_info.last_job_info.sdk_api_job_info
                    else None,
                },
            )

        return processes

    def get_job_queue_info(self) -> dict:
        """Get information about job queues."""
        pm = self.process_manager

        pending_count = len(pm.jobs_pending_inference)
        in_progress_count = len(pm.jobs_in_progress)
        pending_safety = len(pm.jobs_pending_safety_check)
        being_checked = len(pm.jobs_being_safety_checked)
        pending_submit = len(pm.jobs_pending_submit)

        total_jobs = (
            pending_count + in_progress_count + pending_safety + being_checked + pending_submit
        )

        # Calculate queue utilization
        max_queue_size = pm.bridge_data.queue_size if pm.bridge_data else 1
        utilization = (total_jobs / max_queue_size * 100) if max_queue_size > 0 else 0

        return {
            "pending_inference": pending_count,
            "in_progress": in_progress_count,
            "pending_safety_check": pending_safety,
            "being_safety_checked": being_checked,
            "pending_submit": pending_submit,
            "total": total_jobs,
            "max_queue_size": max_queue_size,
            "utilization": round(utilization, 1),
        }

    def get_kudos_stats(self) -> dict:
        """Get kudos and performance statistics."""
        pm = self.process_manager
        uptime_seconds = (datetime.now() - datetime.fromtimestamp(pm.session_start_time)).total_seconds()

        # Calculate kudos per hour
        kudos_per_hour = 0
        if uptime_seconds > 0:
            kudos_per_hour = (pm.kudos_generated_this_session / uptime_seconds) * 3600

        # Get user info
        user_kudos = 0
        user_name = "Unknown"
        if pm.user_info:
            user_kudos = getattr(pm.user_info, "kudos", 0)
            user_name = getattr(pm.user_info, "username", "Unknown")

        return {
            "session_kudos": round(pm.kudos_generated_this_session, 2),
            "kudos_per_hour": round(kudos_per_hour, 2),
            "user_kudos": round(user_kudos, 2),
            "user_name": user_name,
            "process_recoveries": pm._num_process_recoveries,
            "job_slowdowns": pm._num_job_slowdowns,
            "jobs_faulted": pm._num_jobs_faulted,
            "consecutive_failures": pm._num_consecutive_failed_jobs,
            "idle_time": str(timedelta(seconds=int(pm._time_spent_no_jobs_available))).split(".")[0],
        }

    def get_config_info(self) -> dict:
        """Get worker configuration information."""
        pm = self.process_manager
        bd = pm.bridge_data

        if not bd:
            return {"error": "No bridge data available"}

        # Get model list
        models = []
        if bd.image_models_to_load:
            models = [model.model_name for model in bd.image_models_to_load]

        return {
            "worker_name": bd.worker_name,
            "max_threads": bd.max_threads,
            "queue_size": bd.queue_size,
            "max_power": bd.max_power,
            "allow_unsafe_ip": bd.allow_unsafe_ip,
            "require_upfront_kudos": bd.require_upfront_kudos,
            "models": models,
            "ram_to_leave_free": f"{bd.ram_to_leave_free_mb} MB" if bd.ram_to_leave_free_mb else "Default",
            "vram_to_leave_free": f"{bd.vram_to_leave_free_mb} MB" if bd.vram_to_leave_free_mb else "Default",
            "safety_on_gpu": bd.safety_on_gpu,
            "high_performance_mode": bd.high_performance_mode,
            "moderate_performance_mode": bd.moderate_performance_mode,
            "low_memory_mode": bd.low_memory_mode,
            "very_low_memory_mode": bd.very_low_memory_mode,
        }

    def add_log(self, level: str, message: str) -> None:
        """Add a log message to the buffer.

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: Log message
        """
        self._log_buffer.append((datetime.now(), level, message))

    def get_logs(self, limit: int = 100) -> list[tuple[datetime, str, str]]:
        """Get recent log messages.

        Args:
            limit: Maximum number of log messages to return

        Returns:
            List of (timestamp, level, message) tuples
        """
        return list(self._log_buffer)[-limit:]

    def get_all_data(self) -> dict:
        """Get all data for the dashboard."""
        return {
            "worker_status": self.get_worker_status(),
            "processes": self.get_process_info(),
            "job_queues": self.get_job_queue_info(),
            "kudos_stats": self.get_kudos_stats(),
            "config": self.get_config_info(),
            "last_update": datetime.now(),
        }
