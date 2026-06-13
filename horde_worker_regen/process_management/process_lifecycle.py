"""Manages process start, stop, replace, and hung-process detection."""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable
from multiprocessing.synchronize import Lock as Lock_MultiProcessing
from multiprocessing.synchronize import Semaphore

from loguru import logger

from horde_worker_regen.consts import VRAM_HEAVY_MODELS
from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from horde_worker_regen.process_management.worker_state import WorkerState


class ProcessLifecycleManager:
    """Owns process start/stop/replace logic and related state."""

    _process_map: ProcessMap
    _horde_model_map: HordeModelMap
    _job_tracker: JobTracker
    _process_message_queue: ProcessQueue
    _inference_semaphore: Semaphore
    _disk_lock: Lock_MultiProcessing
    _aux_model_lock: Lock_MultiProcessing
    _vae_decode_semaphore: Semaphore
    _runtime_config: RuntimeConfig
    _max_inference_processes: int
    _max_safety_processes: int
    _amd_gpu: bool
    _directml: int | None
    _abort_callback: Callable[[], None]
    _state: WorkerState
    _entry_points: ProcessEntryPoints

    num_processes_launched: int
    _num_process_recoveries: int
    _safety_processes_should_be_replaced: bool
    _safety_processes_ending: bool
    _recently_recovered: bool
    _hung_processes_detected: bool
    _hung_processes_detected_time: float

    def __init__(
        self,
        *,
        process_map: ProcessMap,
        horde_model_map: HordeModelMap,
        job_tracker: JobTracker,
        process_message_queue: ProcessQueue,
        inference_semaphore: Semaphore,
        disk_lock: Lock_MultiProcessing,
        aux_model_lock: Lock_MultiProcessing,
        vae_decode_semaphore: Semaphore,
        runtime_config: RuntimeConfig,
        max_inference_processes: int,
        max_safety_processes: int,
        amd_gpu: bool,
        directml: int | None,
        abort_callback: Callable[[], None],
        state: WorkerState,
        entry_points: ProcessEntryPoints | None = None,
    ) -> None:
        """Initialize with shared references and callbacks from the parent manager."""
        self._process_map = process_map
        self._horde_model_map = horde_model_map
        self._job_tracker = job_tracker
        self._process_message_queue = process_message_queue
        self._inference_semaphore = inference_semaphore
        self._disk_lock = disk_lock
        self._aux_model_lock = aux_model_lock
        self._vae_decode_semaphore = vae_decode_semaphore
        self._runtime_config = runtime_config
        self._max_inference_processes = max_inference_processes
        self._max_safety_processes = max_safety_processes
        self._amd_gpu = amd_gpu
        self._directml = directml
        self._abort_callback = abort_callback
        self._state = state
        self._entry_points = entry_points if entry_points is not None else ProcessEntryPoints()

        self.num_processes_launched = 0
        self._num_process_recoveries = 0
        self._safety_processes_should_be_replaced = False
        self._safety_processes_ending = False
        self._recently_recovered = False
        self._hung_processes_detected = False
        self._hung_processes_detected_time = 0.0
        self._any_replaced = False
        self._on_process_recovery: Callable[[HordeProcessInfo, str], None] | None = None

    def set_process_recovery_observer(self, observer: Callable[[HordeProcessInfo, str], None]) -> None:
        """Register a callback invoked with the process info and a reason on each recovery.

        Used by the run-metrics aggregator to record crash/hang events.
        """
        self._on_process_recovery = observer

    def _notify_process_recovery(self, process_info: HordeProcessInfo, reason: str) -> None:
        if self._on_process_recovery is None:
            return
        try:
            self._on_process_recovery(process_info, reason)
        except Exception as e:
            logger.warning(f"Process recovery observer failed: {type(e).__name__} {e}")

    @property
    def recently_recovered(self) -> bool:
        """Whether a process was recently recovered (read-only for manager)."""
        return self._recently_recovered

    def start_safety_processes(self) -> None:
        """Start all the safety processes configured to be used."""
        bridge_data = self._runtime_config.bridge_data
        num_processes_to_start = self._max_safety_processes - self._process_map.num_safety_processes()

        if num_processes_to_start < 0:
            logger.critical(
                f"There are already {self._process_map.num_safety_processes()} safety processes running, but "
                f"max_safety_processes is set to {self._max_safety_processes}",
            )
            raise ValueError("num_processes_to_start cannot be less than 0")

        for _ in range(num_processes_to_start):
            pid = self._process_map.num_safety_processes()
            pipe_connection, child_pipe_connection = multiprocessing.Pipe(duplex=True)

            cpu_only = not bridge_data.safety_on_gpu

            process = multiprocessing.Process(
                target=self._entry_points.safety_entry_point,
                args=(
                    pid,
                    self._process_message_queue,
                    child_pipe_connection,
                    self._disk_lock,
                    self.num_processes_launched,
                    cpu_only,
                ),
                kwargs={
                    "high_memory_mode": bridge_data.high_memory_mode,
                    "amd_gpu": self._amd_gpu,
                    "directml": self._directml,
                    "dry_run_skip_safety": bridge_data.dry_run_skip_safety,
                },
            )

            process.start()

            self._process_map[pid] = HordeProcessInfo(
                mp_process=process,
                pipe_connection=pipe_connection,
                process_id=pid,
                process_type=HordeProcessType.SAFETY,
                last_process_state=HordeProcessState.PROCESS_STARTING,
                process_launch_identifier=self.num_processes_launched,
            )

            logger.info(f"Started safety process (id: {pid})")
            self.num_processes_launched += 1

    def start_inference_processes(self) -> None:
        """Start all the inference processes configured to be used."""
        num_processes_to_start = self._max_inference_processes - self._process_map.num_inference_processes()

        if num_processes_to_start < 0:
            logger.critical(
                f"There are already {self._process_map.num_inference_processes()} inference processes running, but "
                f"max_inference_processes is set to {self._max_inference_processes}",
            )
            raise ValueError("num_processes_to_start cannot be less than 0")

        for i in range(num_processes_to_start):
            pid = len(self._process_map)
            self._start_inference_process(pid)

            logger.info(f"Started inference process (id: {pid})")

            if i == 0:
                time.sleep(4)

    def _start_inference_process(self, pid: int) -> HordeProcessInfo:
        """Starts an inference process.

        :param pid: process ID to assign to the process
        :return: The new HordeProcessInfo
        """
        bridge_data = self._runtime_config.bridge_data
        logger.info(f"Starting inference process on PID {pid}")
        vram_heavy_models = any(model in VRAM_HEAVY_MODELS for model in bridge_data.image_models_to_load)

        pipe_connection, child_pipe_connection = multiprocessing.Pipe(duplex=True)
        process = multiprocessing.Process(
            target=self._entry_points.inference_entry_point,
            args=(
                pid,
                self._process_message_queue,
                child_pipe_connection,
                self._inference_semaphore,
                self._disk_lock,
                self._aux_model_lock,
                self._vae_decode_semaphore,
                self.num_processes_launched,
            ),
            kwargs={
                "very_high_memory_mode": bridge_data.very_high_memory_mode,
                "high_memory_mode": bridge_data.high_memory_mode,
                "amd_gpu": self._amd_gpu,
                "directml": self._directml,
                "vram_heavy_models": vram_heavy_models,
                "dry_run_skip_inference": bridge_data.dry_run_skip_inference,
                "dry_run_inference_delay": bridge_data.dry_run_inference_delay,
            },
        )
        process.start()
        process_info = HordeProcessInfo(
            mp_process=process,
            pipe_connection=pipe_connection,
            process_id=pid,
            process_type=HordeProcessType.INFERENCE,
            last_process_state=HordeProcessState.PROCESS_STARTING,
            process_launch_identifier=self.num_processes_launched,
        )
        self._process_map[pid] = process_info
        self.num_processes_launched += 1
        return process_info

    def end_inference_processes(
        self,
        force: bool = False,
    ) -> None:
        """End any inference processes above the configured limit, or all of them if shutting down."""
        if force:
            if not self._state.shutting_down:
                logger.error("Forcing inference processes to end without shutting down")

            for process in self._process_map.get_inference_processes():
                self._end_inference_process(process)

        if len(self._job_tracker.jobs_pending_inference) > 0 and len(
            self._job_tracker.jobs_pending_inference,
        ) != len(self._job_tracker.jobs_in_progress):
            return

        processes_with_model_for_queued_job: list[int] = self.get_processes_with_model_for_queued_job()

        if (
            self._state.shutting_down
            and len(self._job_tracker.jobs_pending_inference) == 0
            and len(self._job_tracker.jobs_in_progress) == 0
        ):
            processes_with_model_for_queued_job = []

        process_info = self._process_map._get_first_inference_process_to_kill(
            disallowed_processes=processes_with_model_for_queued_job,
        )

        if process_info is not None:
            self._end_inference_process(process_info)

    def _end_inference_process(self, process_info: HordeProcessInfo) -> None:
        """Ends an inference process."""
        self._process_map.on_process_ending(process_id=process_info.process_id)
        if process_info.loaded_horde_model_name is not None:
            self._horde_model_map.expire_entry(process_info.loaded_horde_model_name)

        try:
            process_info.safe_send_message(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
        except BrokenPipeError:
            if not self._state.shutting_down:
                logger.debug(f"Process {process_info.process_id} control channel vanished")
        try:
            process_info.mp_process.join(timeout=1)
            process_info.mp_process.kill()
        except Exception as e:
            logger.error(f"Failed to kill process {process_info.process_id}: {e}")

        if not self._state.shutting_down:
            logger.info(f"Ended inference process {process_info.process_id}")

    def end_safety_processes(self) -> None:
        """End any safety processes above the configured limit, or all of them if shutting down."""
        process_info = self._process_map.get_first_available_safety_process()

        if process_info is None:
            return

        # Do not re-target a safety process that is already ending.
        if process_info.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED):
            return

        process_info.safe_send_message(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
        self._process_map.on_process_ending(process_id=process_info.process_id)

        logger.info(f"Ended safety process {process_info.process_id}")

    def _replace_all_safety_process(self) -> None:
        """Replace all of the safety processes."""
        if not self._safety_processes_should_be_replaced:
            return

        if not self._safety_processes_ending and self._process_map.num_loaded_safety_processes() > 0:
            self._safety_processes_ending = True
            self.end_safety_processes()
            return

        if self._process_map.num_loaded_safety_processes() == 0 and self._process_map.num_safety_processes() > 0:
            self._process_map.delete_safety_processes()

        if (
            self._safety_processes_ending
            and self._process_map.num_loaded_safety_processes() == 0
            and self._process_map.num_safety_processes() == 0
        ):
            self.start_safety_processes()
            self._safety_processes_ending = False
            self._safety_processes_should_be_replaced = False
            self._num_process_recoveries += 1

    def _replace_inference_process(self, process_info: HordeProcessInfo) -> None:
        """Replaces an inference process (for whatever reason; probably because it crashed)."""
        bridge_data = self._runtime_config.bridge_data
        logger.debug(f"Replacing {process_info}")
        job_to_remove = None
        for process in self._process_map.values():
            if (
                process.last_job_referenced is not None
                and process.last_job_referenced in self._job_tracker.jobs_lookup
            ):
                job_to_remove = process.last_job_referenced
                break

        if process_info.last_process_state == HordeProcessState.INFERENCE_STARTING:
            try:
                self._inference_semaphore.release()
            except ValueError:
                logger.debug("Inference semaphore already released")
            try:
                self._disk_lock.release()
            except ValueError:
                logger.debug("Disk lock already released")

        elif process_info.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL:
            try:
                self._aux_model_lock.release()
            except ValueError:
                logger.debug("Aux model lock already released")

            if (
                process_info.last_job_referenced is not None
                and process_info.last_job_referenced in self._job_tracker.jobs_lookup
            ):
                job_to_remove = process_info.last_job_referenced
                logger.error(
                    f"Job {job_to_remove.id_ or job_to_remove.ids} was in aux model preload on process "
                    f"{process_info.process_id} but it failed. Removing.",
                )

        if process_info.loaded_horde_model_name is not None:
            self._horde_model_map.expire_entry(process_info.loaded_horde_model_name)

        if job_to_remove is not None:
            self._job_tracker.handle_job_fault_now(
                faulted_job=job_to_remove,
                process_info=process_info,
                process_timeout=bridge_data.process_timeout,
            )

        self._notify_process_recovery(process_info, "inference process replaced (crashed or hung)")

        self._end_inference_process(process_info)
        self._start_inference_process(process_info.process_id)

        self._num_process_recoveries += 1

    def get_processes_with_model_for_queued_job(self) -> list[int]:
        """Get the processes that have the model for any queued job."""
        processes_with_model_for_queued_job: list[int] = []

        queued_models = {
            job.model for job in self._job_tracker.jobs_pending_inference if getattr(job, "model", None) is not None
        }
        in_progress_models = {
            job.model for job in self._job_tracker.jobs_in_progress if getattr(job, "model", None) is not None
        }

        for p in self._process_map.values():
            if (
                p.loaded_horde_model_name in queued_models
                or p.loaded_horde_model_name in in_progress_models
                or p.last_process_state == HordeProcessState.PRELOADED_MODEL
            ):
                processes_with_model_for_queued_job.append(p.process_id)

        return processes_with_model_for_queued_job

    def _hard_kill_processes(
        self,
        inference: bool = True,
        safety: bool = True,
        all_: bool = True,
    ) -> None:
        """Kill all processes immediately."""
        for process_info in self._process_map.values():
            if (
                (inference and process_info.process_type == HordeProcessType.INFERENCE)
                or (safety and process_info.process_type == HordeProcessType.SAFETY)
                or (all_)
            ):
                try:
                    process_info.mp_process.kill()
                    process_info.mp_process.kill()
                    process_info.mp_process.join(1)
                except Exception as e:
                    logger.error(f"Failed to kill process {process_info}: {e}")

        self._process_map.clear()
        self._horde_model_map.root.clear()

    def _check_and_replace_process(
        self,
        process_info: HordeProcessInfo,
        timeout: float,
        state: HordeProcessState,
        error_message: str,
    ) -> bool:
        """Check if a process has been stuck in a state for too long and replace it if it has.

        Returns:
            True if the process was replaced, False otherwise
        """
        now = time.time()
        time_elapsed = now - process_info.last_received_timestamp
        time_elapsed = min(time_elapsed, now - process_info.last_heartbeat_timestamp)

        if time_elapsed > timeout and process_info.last_process_state == state:
            logger.error(f"{process_info} {error_message}, replacing it")
            if process_info.process_type == HordeProcessType.SAFETY:
                self._replace_all_safety_process()
            if process_info.process_type == HordeProcessType.INFERENCE:
                self._replace_inference_process(process_info)
            return True
        return False

    def replace_hung_processes(self) -> bool:
        """Replaces processes that haven't checked in since `process_timeout` seconds in bridgeData."""
        if self._recently_recovered:
            return False

        import threading

        bridge_data = self._runtime_config.bridge_data

        def timed_unset_recently_recovered() -> None:
            time.sleep(bridge_data.inference_step_timeout)
            self._recently_recovered = False

        now = time.time()

        any_replaced = False
        for process_info in self._process_map.values():
            if self._process_map.is_stuck_on_inference(
                process_info.process_id,
                bridge_data.inference_step_timeout,
            ):
                logger.error(f"{process_info} seems to be stuck mid inference, replacing it")
                self._replace_inference_process(process_info)
                any_replaced = True
                self._recently_recovered = True
                threading.Thread(target=timed_unset_recently_recovered).start()
            else:
                conditions: list[tuple[float, HordeProcessState, str]] = [
                    (
                        bridge_data.preload_timeout,
                        HordeProcessState.PRELOADING_MODEL,
                        "seems to be stuck preloading a model",
                    ),
                    (
                        bridge_data.download_timeout,
                        HordeProcessState.DOWNLOADING_AUX_MODEL,
                        "seems to be stuck downloading an auxiliary model (LoRa, etc)",
                    ),
                    (
                        bridge_data.preload_timeout,
                        HordeProcessState.PROCESS_STARTING,
                        "seems to be stuck starting",
                    ),
                    (
                        bridge_data.post_process_timeout + (3 * bridge_data.max_batch),
                        HordeProcessState.INFERENCE_POST_PROCESSING,
                        "seems to be stuck post processing",
                    ),
                ]
                if self._state.last_pop_no_jobs_available:
                    continue

                for timeout, state, error_message in conditions:
                    if self._check_and_replace_process(process_info, timeout, state, error_message):
                        any_replaced = True
                        self._recently_recovered = True

        if self._state.last_pop_no_jobs_available:
            return any_replaced

        all_processes_timed_out = all(
            ((now - process_info.last_received_timestamp) > bridge_data.process_timeout)
            for process_info in self._process_map.values()
        )

        shutdown_timed_out = self._state.shutting_down and (now - self._state.shutting_down_time) > (60 * 5)

        if (all_processes_timed_out and not (self._state.last_pop_no_jobs_available or self._recently_recovered)) or (
            shutdown_timed_out
        ):
            if not self._hung_processes_detected:
                self._hung_processes_detected = True
                self._hung_processes_detected_time = now

            last_detected_delta = now - self._hung_processes_detected_time

            if last_detected_delta < 20:
                return False

            self._job_tracker._purge_jobs()

            if bridge_data.exit_on_unhandled_faults or self._state.shutting_down:
                logger.error("All processes have been unresponsive for too long, exiting.")

                self._abort_callback()
                if bridge_data.exit_on_unhandled_faults:
                    logger.error("Exiting due to exit_on_unhandled_faults being enabled")

                return True

            logger.error("All processes have been unresponsive for too long, attempting to recover.")
            self._recently_recovered = True

            for process_info in self._process_map.values():
                if process_info.process_type == HordeProcessType.INFERENCE:
                    self._replace_inference_process(process_info)
                    self._any_replaced = True

            threading.Thread(target=timed_unset_recently_recovered).start()
        else:
            self._hung_processes_detected = False

        if any_replaced:
            threading.Thread(target=timed_unset_recently_recovered).start()

        return any_replaced

    @property
    def safety_processes_should_be_replaced(self) -> bool:
        """Whether the safety processes should be replaced."""
        return self._safety_processes_should_be_replaced

    @safety_processes_should_be_replaced.setter
    def safety_processes_should_be_replaced(self, value: bool) -> None:
        self._safety_processes_should_be_replaced = value
