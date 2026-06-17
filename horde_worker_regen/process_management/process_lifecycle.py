"""Manages process start, stop, replace, and hung-process detection."""

from __future__ import annotations

import contextlib
import multiprocessing
import time
from collections.abc import Callable
from multiprocessing.synchronize import Lock as Lock_MultiProcessing
from multiprocessing.synchronize import Semaphore

from loguru import logger

from horde_worker_regen.consts import VRAM_HEAVY_MODELS
from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.action_ledger import ActionLedger, LedgerEventType
from horde_worker_regen.process_management.download_process import DOWNLOAD_PROCESS_ID
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.horde_process import HordeProcessType, WorkerCapability
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeDownloadControlMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.owned_process_registry import OwnedProcessRegistry
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from horde_worker_regen.process_management.worker_state import WorkerState

CRASH_LOOP_WINDOW_SECONDS: float = 300.0
"""Sliding window over which an inference slot's replacements are counted for crash-loop detection."""

CRASH_LOOP_MAX_REPLACEMENTS: int = 3
"""Replacements of a single slot within ``CRASH_LOOP_WINDOW_SECONDS`` before it is quarantined.

A slot that dies (or hangs) faster than it can do useful work is not worth respawning indefinitely:
each restart costs a model (re)load and starves the worker. Past this count the slot is quarantined
(left out of the pool) and the lost capacity is surfaced as a severity signal for the higher-level
recovery supervisor rather than papered over by an unbounded respawn loop.
"""

CRASH_LOOP_MAX_START_FAILURES: int = 3
"""Consecutive replacements while still in ``PROCESS_STARTING`` before a slot is quarantined.

This is the rate-independent companion to the sliding-window breaker above. A slot that *never*
advances past ``PROCESS_STARTING`` before dying has not proven it can initialise at all (a broken
dependency, a missing model, an import error). Such a failure is deterministic, so each restart costs
the full, slow cold-start before failing again -- and if that cold-start is slower than
``CRASH_LOOP_WINDOW_SECONDS / CRASH_LOOP_MAX_REPLACEMENTS`` the window breaker can never accumulate
enough replacements *within the window* to trip (the early ones age out), so the slot would respawn
forever. Counting consecutive start-failures regardless of spacing catches exactly that case. The
streak resets the moment the slot reaches any later state (it did initialise, then failed differently).
"""

SAFETY_CRASH_LOOP_MAX: int = 3
"""Safety-pool replacements within ``CRASH_LOOP_WINDOW_SECONDS`` before the pool is reported as failing.

The crash-loop circuit breaker quarantines individual *inference* slots, but the safety pool has no such
per-slot breaker. This count is the equivalent signal for safety: a pool that has been rebuilt more than
this many times in the window is failing (e.g. a safety process that crashes on every start), which the
recovery supervisor escalates rather than rebuilding the pool forever."""

SLOWDOWN_NOTICE_RATIO: float = 2.0
"""Sampling time past this multiple of the job's expected time logs a soft notice (rung 1 of the ladder)."""

SLOWDOWN_WARN_RATIO: float = 4.0
"""Sampling time past this multiple warns, audits, and counts toward the recovery-supervisor severity.

The hard kill remains the ``inference_step_timeout`` in :meth:`replace_hung_processes`; these softer,
evidence-based rungs sit below it so a measurable slowdown is logged before the slot is replaced."""


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
    _gpu_sampling_lease: Semaphore
    _gpu_sampling_lease_enabled: bool
    _runtime_config: RuntimeConfig
    _max_inference_processes: int
    _max_safety_processes: int
    _amd_gpu: bool
    _directml: int | None
    _abort_callback: Callable[[], None]
    _state: WorkerState
    _entry_points: ProcessEntryPoints
    _download_process_info: HordeProcessInfo | None
    _owned_registry: OwnedProcessRegistry | None
    _action_ledger: ActionLedger

    num_processes_launched: int
    _num_process_recoveries: int
    _safety_processes_should_be_replaced: bool
    _safety_processes_ending: bool
    _recently_recovered: bool
    _hung_processes_detected: bool
    _hung_processes_detected_time: float
    _slot_recovery_history: dict[int, list[float]]
    _slot_consecutive_start_failures: dict[int, int]
    _quarantined_inference_slots: set[int]
    _num_slots_quarantined: int
    _safety_recovery_history: list[float]

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
        gpu_sampling_lease: Semaphore,
        gpu_sampling_lease_enabled: bool = False,
        runtime_config: RuntimeConfig,
        max_inference_processes: int,
        max_safety_processes: int,
        amd_gpu: bool,
        directml: int | None,
        abort_callback: Callable[[], None],
        state: WorkerState,
        entry_points: ProcessEntryPoints | None = None,
        owned_registry: OwnedProcessRegistry | None = None,
        action_ledger: ActionLedger | None = None,
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
        self._gpu_sampling_lease = gpu_sampling_lease
        self._gpu_sampling_lease_enabled = gpu_sampling_lease_enabled
        self._runtime_config = runtime_config
        self._max_inference_processes = max_inference_processes
        self._max_safety_processes = max_safety_processes
        self._amd_gpu = amd_gpu
        self._directml = directml
        self._abort_callback = abort_callback
        self._state = state
        self._entry_points = entry_points if entry_points is not None else ProcessEntryPoints()
        self._owned_registry = owned_registry
        # The ledger is always present (an in-memory ring by default) so diagnostics work under test;
        # the parent manager injects a file-backed one in a real run.
        self._action_ledger = action_ledger if action_ledger is not None else ActionLedger()

        self.num_processes_launched = 0
        self._num_process_recoveries = 0
        self._num_slowdown_events = 0
        self._safety_processes_should_be_replaced = False
        self._safety_processes_ending = False
        self._recently_recovered = False
        self._hung_processes_detected = False
        self._hung_processes_detected_time = 0.0
        self._any_replaced = False
        self._on_process_recovery: Callable[[HordeProcessInfo, str], None] | None = None
        self._download_process_info = None
        self._slot_recovery_history = {}
        self._slot_consecutive_start_failures = {}
        self._quarantined_inference_slots = set()
        self._num_slots_quarantined = 0
        self._safety_recovery_history = []

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

    @property
    def num_slots_quarantined(self) -> int:
        """How many inference slots the crash-loop circuit breaker has taken out of the pool."""
        return self._num_slots_quarantined

    @property
    def quarantined_inference_slots(self) -> frozenset[int]:
        """The process ids of inference slots currently quarantined (read-only)."""
        return frozenset(self._quarantined_inference_slots)

    @property
    def download_process_info(self) -> HordeProcessInfo | None:
        """The background download process, or None if one is not running."""
        return self._download_process_info

    @property
    def action_ledger(self) -> ActionLedger:
        """The self-audited record of lifecycle actions taken on child processes (read-only)."""
        return self._action_ledger

    def _register_owned(self, process_info: HordeProcessInfo) -> None:
        """Record a just-started child in the action ledger and the owned-PID registry."""
        self._action_ledger.record(
            LedgerEventType.PROCESS_SPAWNED,
            process_id=process_info.process_id,
            os_pid=process_info.os_pid,
            launch_identifier=process_info.process_launch_identifier,
            detail={"process_type": process_info.process_type.name},
        )
        if self._owned_registry is not None:
            self._owned_registry.record(
                os_pid=process_info.os_pid,
                launch_identifier=process_info.process_launch_identifier,
                process_type=process_info.process_type.name,
            )

    def _log_recovery_diagnostics(self, process_info: HordeProcessInfo, reason: str) -> None:
        """Emit a structured snapshot of why a process is being recovered, so a future hang explains itself.

        Pulls together the OS identity, the last state and heartbeat the parent saw, the child's exit
        code (if it died), and the slot's recent ledger history into one log line. Called at the moment
        of replacement, while ``process_info`` still reflects the faulted state.
        """
        now = time.time()
        exitcode: int | None = None
        with contextlib.suppress(Exception):
            exitcode = process_info.mp_process.exitcode

        recent = self._action_ledger.recent(process_id=process_info.process_id, limit=10)
        recent_summary = "; ".join(
            f"{event.event_type.name}@-{now - event.timestamp:.1f}s" + (f"({event.reason})" if event.reason else "")
            for event in recent
        )
        last_job = process_info.last_job_referenced
        logger.error(
            f"Recovery diagnostics for process {process_info.process_id} (os_pid={process_info.os_pid}, "
            f"launch={process_info.process_launch_identifier}): reason='{reason}'; "
            f"last_state={process_info.last_process_state.name}; exitcode={exitcode}; "
            f"last_heartbeat_type={process_info.last_heartbeat_type.name}; "
            f"since_last_heartbeat={now - process_info.last_heartbeat_timestamp:.1f}s; "
            f"since_last_message={now - process_info.last_received_timestamp:.1f}s; "
            f"last_job={last_job.id_ if last_job is not None else None}; recent_actions=[{recent_summary}]",
        )

    def _forget_owned(self, process_info: HordeProcessInfo) -> None:
        """Drop a cleanly-ended child from the owned-PID registry (no-op if reaping is disabled)."""
        if self._owned_registry is not None:
            self._owned_registry.forget(process_info.os_pid)

    def kill_owned_children(self) -> list[int]:
        """Best-effort kill of every still-owned child by OS pid; for atexit / signal cleanup.

        Identity is re-verified per pid inside the registry, so a reused pid is never killed. Returns
        the pids actually killed. No-op (empty list) when orphan reaping is disabled.
        """
        if self._owned_registry is None:
            return []
        return self._owned_registry.kill_all_owned()

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
            self._register_owned(self._process_map[pid])

            logger.info(f"Started safety process (id: {pid})")
            self.num_processes_launched += 1

    def start_download_process(self) -> None:
        """Start the singleton background download process, if not already running.

        The download process lives outside the process map (it serves no jobs and must not be
        swept up by the hung-process logic); its messages are routed by its reserved process id.
        """
        if self._download_process_info is not None:
            return

        bridge_data = self._runtime_config.bridge_data
        pipe_connection, child_pipe_connection = multiprocessing.Pipe(duplex=True)

        process = multiprocessing.Process(
            target=self._entry_points.download_entry_point,
            args=(
                DOWNLOAD_PROCESS_ID,
                self._process_message_queue,
                child_pipe_connection,
                self._disk_lock,
                self.num_processes_launched,
            ),
            kwargs={
                "nsfw": bridge_data.nsfw,
                "allow_lora": bridge_data.allow_lora,
                "allow_controlnet": bridge_data.allow_controlnet,
                "allow_sdxl_controlnet": bridge_data.allow_sdxl_controlnet,
                "allow_post_processing": bridge_data.allow_post_processing,
                "purge_loras": bridge_data.purge_loras_on_download,
                "amd_gpu": self._amd_gpu,
                "directml": self._directml,
                "rate_limit_kbps": bridge_data.download_rate_limit_kbps,
                "paused": bridge_data.downloads_paused,
            },
        )
        process.start()

        self._download_process_info = HordeProcessInfo(
            mp_process=process,
            pipe_connection=pipe_connection,
            process_id=DOWNLOAD_PROCESS_ID,
            process_type=HordeProcessType.DOWNLOAD,
            last_process_state=HordeProcessState.PROCESS_STARTING,
            process_launch_identifier=self.num_processes_launched,
            capabilities=WorkerCapability(0),
        )
        self._register_owned(self._download_process_info)
        self.num_processes_launched += 1
        logger.info("Started background download process")

    def request_downloads(self, model_names: list[str], *, download_aux: bool = False) -> None:
        """Ask the download process to ensure the given image models are present on disk."""
        if self._download_process_info is None:
            logger.warning("Cannot request downloads: no download process is running")
            return
        if not model_names and not download_aux:
            return
        self._download_process_info.safe_send_message(
            HordeDownloadControlMessage(model_names=list(model_names), download_aux=download_aux),
        )

    def set_download_controls(self, *, paused: bool | None = None, rate_limit_kbps: int | None = None) -> None:
        """Forward live pause/bandwidth controls to the download process (no-op if none is running).

        Used by both the config-reload path and the supervisor pause/resume/rate commands. A ``None``
        argument leaves that control unchanged; ``rate_limit_kbps`` of 0 (or negative) clears the cap.
        """
        if self._download_process_info is None:
            return
        if paused is None and rate_limit_kbps is None:
            return
        self._download_process_info.safe_send_message(
            HordeDownloadControlMessage(
                model_names=[],
                download_aux=False,
                set_paused=paused,
                set_rate_limit_kbps=rate_limit_kbps,
            ),
        )

    def broadcast_reload_model_database(self) -> None:
        """Tell every subprocess to reload its model managers' references from disk (no download).

        Sent after the parent refreshes the on-disk reference, or after the download process reports
        new LoRa/TI availability, so inference and download subprocesses pick up the changes live
        without a restart. Subprocesses never download references; they only re-read the parent's files.
        """
        message = HordeControlMessage(control_flag=HordeControlFlag.RELOAD_MODEL_DATABASE)
        for process_info in self._process_map.get_inference_processes():
            process_info.safe_send_message(message)
        if self._download_process_info is not None:
            self._download_process_info.safe_send_message(message)

    def end_download_process(self) -> None:
        """Stop the background download process, if running."""
        if self._download_process_info is None:
            return
        with contextlib.suppress(BrokenPipeError):
            self._download_process_info.safe_send_message(
                HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS),
            )
        try:
            self._download_process_info.mp_process.join(timeout=1)
            self._download_process_info.mp_process.kill()
        except Exception as e:
            logger.debug(f"Failed to stop download process: {e}")
        self._forget_owned(self._download_process_info)
        self._download_process_info = None

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
                "gpu_sampling_lease": self._gpu_sampling_lease if self._gpu_sampling_lease_enabled else None,
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
        self._register_owned(process_info)
        self.num_processes_launched += 1
        return process_info

    def _allocate_inference_pid(self) -> int:
        """Return the lowest process id not currently in use.

        Slot ids are reused once freed, so this stays stable across scale-down/scale-up cycles
        (a ``len(map)``-based scheme would collide after removing a non-last slot). The download
        process lives outside the map at its own reserved id, so it never participates here.
        """
        used = set(self._process_map.keys())
        pid = 0
        while pid in used:
            pid += 1
        return pid

    def scale_inference_processes(self, target_count: int) -> int:
        """Grow or shrink the running inference processes toward ``target_count``.

        Growth spawns fresh processes (bounded by the launched-process ceiling). Shrink ends idle
        processes, preferring ones not holding a model needed by queued work, and removes them from
        the process map; busy processes are never killed, so the effective count may not reach the
        target in one call. Used by the benchmark to stage processes on demand and as a memory/VRAM
        pressure lever.

        Returns:
            The number of inference processes after scaling.
        """
        target = max(0, min(target_count, self._max_inference_processes))
        current = self._process_map.num_loaded_inference_processes()

        if target > current:
            for _ in range(target - current):
                pid = self._allocate_inference_pid()
                self._start_inference_process(pid)
                logger.info(f"Scaled up: started inference process {pid}")
        elif target < current:
            disallowed = self.get_processes_with_model_for_queued_job()
            for _ in range(current - target):
                victim = self._process_map._get_first_inference_process_to_kill(disallowed_processes=disallowed)
                if victim is None:
                    logger.debug("Scale down: no idle inference process available to stop right now")
                    break
                self._end_inference_process(victim)
                self._process_map.pop(victim.process_id, None)
                logger.info(f"Scaled down: stopped inference process {victim.process_id}")

        return self._process_map.num_loaded_inference_processes()

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

        self._forget_owned(process_info)
        self._action_ledger.record(
            LedgerEventType.PROCESS_ENDED,
            process_id=process_info.process_id,
            os_pid=process_info.os_pid,
            launch_identifier=process_info.process_launch_identifier,
        )

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
        self._forget_owned(process_info)

        logger.info(f"Ended safety process {process_info.process_id}")

    def _initiate_safety_replacement(self) -> None:
        """Flag the safety pool for replacement so the control loop's state machine restarts it.

        Setting this flag is the trigger; `_replace_all_safety_process` (run each control-loop tick)
        then ends, deletes, and restarts the safety process across the next few ticks. The only other
        place this flag is set requires a *running* safety process and an in-flight job, so without
        this a safety process that wedges or dies during startup would never be recovered: it would
        sit pinned at PROCESS_STARTING while the stuck-detection logged a misleading "replacing it"
        forever without doing anything.
        """
        self._safety_processes_should_be_replaced = True

    def _reap_if_crashed(self, process_info: HordeProcessInfo) -> bool:
        """Recover a child that has already exited (crash, sys.exit, segfault) instead of waiting on a timer.

        A dead child sends no further messages, so the state-timeout checks in `replace_hung_processes`
        would otherwise leave it pinned at its last reported state forever. Detecting the exit directly
        lets us restart it promptly and log the exit code so the cause is visible.

        Returns:
            True if the process was found dead and a replacement was initiated.
        """
        if process_info.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED):
            return False
        if process_info.mp_process.is_alive():
            return False

        exit_code = process_info.mp_process.exitcode
        logger.error(
            f"{process_info} exited unexpectedly (exitcode={exit_code}) while "
            f"{process_info.last_process_state.name}; recovering",
        )
        if process_info.process_type == HordeProcessType.SAFETY:
            self._initiate_safety_replacement()
            self._replace_all_safety_process()
        elif process_info.process_type == HordeProcessType.INFERENCE:
            self._replace_inference_process(process_info)
        return True

    def _replace_all_safety_process(self) -> None:
        """Replace all of the safety processes."""
        if not self._safety_processes_should_be_replaced:
            return

        if not self._safety_processes_ending:
            # Enter the ending phase on the first call regardless of whether a process is currently
            # loaded. A safety process that died while still PROCESS_STARTING is never "loaded", so the
            # old ``num_loaded > 0`` guard left ``_safety_processes_ending`` unset; the restart branch
            # below (gated on that flag) then never fired, leaving the worker without safety forever.
            # Setting the flag here covers both the normal end->delete->start flow and a startup crash.
            self._safety_processes_ending = True
            if self._process_map.num_loaded_safety_processes() > 0:
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
            self._record_safety_recovery()

    def _record_safety_recovery(self) -> None:
        """Record that the safety pool was just rebuilt, pruning the history to the crash-loop window."""
        now = time.time()
        self._safety_recovery_history = [
            t for t in self._safety_recovery_history if now - t <= CRASH_LOOP_WINDOW_SECONDS
        ]
        self._safety_recovery_history.append(now)

    @property
    def safety_pool_failing(self) -> bool:
        """Whether the safety pool has been rebuilt too many times recently (its crash-loop signal).

        The equivalent of inference-slot quarantine for the safety pool: True when a safety process has
        had to be rebuilt more than ``SAFETY_CRASH_LOOP_MAX`` times within the crash-loop window (e.g. it
        crashes on every start), which the recovery supervisor escalates instead of rebuilding forever.
        """
        now = time.time()
        recent = [t for t in self._safety_recovery_history if now - t <= CRASH_LOOP_WINDOW_SECONDS]
        return len(recent) > SAFETY_CRASH_LOOP_MAX

    def _release_held_primitives(self, process_info: HordeProcessInfo) -> None:
        """Release every shared primitive a replaced inference child might still be holding.

        A child acquires the inference/VAE/sampling semaphores and the disk/aux locks inside its own
        process, so a child that dies or hangs leaves them held; the parent must release on its behalf
        or that concurrency is lost for the lifetime of the worker (one orphaned inference permit at
        ``max_threads=1`` wedges everything). We deliberately do not infer which primitives are held
        from the last state the parent recorded: a child can crash after acquiring but before the
        parent processes the matching state-change message, and that exact race is what wedged the
        worker before. Releasing unconditionally is safe because every one of these is bounded (the
        semaphores are BoundedSemaphores, a Lock is bound to one), so releasing one the child did not
        hold raises ValueError, which we swallow as a harmless no-op rather than inflating a limit.
        """
        candidates: list[tuple[str, Semaphore | Lock_MultiProcessing]] = [
            ("inference_semaphore", self._inference_semaphore),
            ("disk_lock", self._disk_lock),
            ("aux_model_lock", self._aux_model_lock),
            ("vae_decode_semaphore", self._vae_decode_semaphore),
        ]
        if self._gpu_sampling_lease_enabled:
            candidates.append(("gpu_sampling_lease", self._gpu_sampling_lease))

        released: list[str] = []
        for name, primitive in candidates:
            try:
                primitive.release()
                released.append(name)
            except ValueError:
                # Not held by the dead child; the bounded primitive rejected the spurious release.
                pass

        if released:
            self._action_ledger.record(
                LedgerEventType.SEMAPHORE_RELEASED,
                process_id=process_info.process_id,
                os_pid=process_info.os_pid,
                detail={"released": ", ".join(released)},
            )
            logger.debug(
                f"Released primitives possibly held by replaced process {process_info.process_id}: "
                f"{', '.join(released)}",
            )

    def _record_slot_recovery(self, process_id: int) -> int:
        """Record a replacement of the given slot and return how many happened within the window."""
        now = time.time()
        recent = [t for t in self._slot_recovery_history.get(process_id, []) if now - t <= CRASH_LOOP_WINDOW_SECONDS]
        recent.append(now)
        self._slot_recovery_history[process_id] = recent
        return len(recent)

    def _record_start_failure(self, process_info: HordeProcessInfo) -> int:
        """Track consecutive replacements that never advanced past PROCESS_STARTING; return the streak.

        A replacement while still in ``PROCESS_STARTING`` means the slot died (or was killed) before it
        ever became job-capable, so it never proved it can initialise. These are counted consecutively
        and the streak resets the moment a slot is replaced from any later state (it did initialise, then
        failed differently). Unlike :meth:`_record_slot_recovery`, this is independent of how long each
        failed start took, so a slow but deterministic crash-on-start is still caught.
        """
        process_id = process_info.process_id
        if process_info.last_process_state == HordeProcessState.PROCESS_STARTING:
            streak = self._slot_consecutive_start_failures.get(process_id, 0) + 1
            self._slot_consecutive_start_failures[process_id] = streak
            return streak
        self._slot_consecutive_start_failures.pop(process_id, None)
        return 0

    def _quarantine_inference_slot(self, process_info: HordeProcessInfo, reason: str) -> None:
        """Take a crash-looping (or crash-on-start) inference slot out of the pool instead of respawning it.

        The slot has tripped one of the circuit breakers (too many replacements in the window, or too
        many consecutive failures before reaching readiness), so respawning it would just repeat the
        loop and keep starving the worker. Its OS process was already ended by the caller, and the
        caller already recorded the recovery event; here we only drop it from the process map and
        remember it so it is not silently refilled. The lost capacity is surfaced via
        ``num_slots_quarantined`` so the higher-level recovery supervisor (Phase 5) can escalate when
        too much capacity is lost.
        """
        self._quarantined_inference_slots.add(process_info.process_id)
        self._num_slots_quarantined += 1
        self._process_map.pop(process_info.process_id, None)
        self._action_ledger.record(
            LedgerEventType.PROCESS_QUARANTINED,
            process_id=process_info.process_id,
            os_pid=process_info.os_pid,
            launch_identifier=process_info.process_launch_identifier,
            reason=reason,
        )
        logger.critical(
            f"Inference slot {process_info.process_id} quarantined ({reason}); not respawning it.",
        )

    def rebuild_inference_pool(self, *, reason: str) -> None:
        """Rebuild the inference pool in place: replace live slots and revive quarantined ones.

        The recovery supervisor's soft reset uses this to give a wedged worker (e.g. every slot
        quarantined by the crash-loop breaker) a clean start without restarting the parent process or
        detaching the TUI. The crash-loop history is cleared first: this is a deliberate, supervised
        rebuild, not the unbounded respawn loop the breaker guards against, so prior replacements must
        not immediately re-quarantine the fresh slots.
        """
        logger.error(f"Soft reset: rebuilding inference pool ({reason}).")
        self._slot_recovery_history.clear()
        self._slot_consecutive_start_failures.clear()

        revived = sorted(self._quarantined_inference_slots)
        self._quarantined_inference_slots.clear()

        live = [p for p in self._process_map.values() if p.process_type == HordeProcessType.INFERENCE]
        for process_info in live:
            self._replace_inference_process(process_info)

        for slot_id in revived:
            if slot_id not in self._process_map:
                self._start_inference_process(slot_id)

        self._action_ledger.record(
            LedgerEventType.PROCESS_REPLACED,
            reason=f"soft reset: {reason}",
            detail={"rebuilt_live": len(live), "revived_quarantined": len(revived)},
        )

    def rebuild_safety_pool(self, *, reason: str) -> None:
        """Force the safety pool to be rebuilt (arm + replace), used by the recovery supervisor's soft reset."""
        logger.error(f"Soft reset: rebuilding safety pool ({reason}).")
        self._initiate_safety_replacement()
        self._replace_all_safety_process()

    def _replace_inference_process(self, process_info: HordeProcessInfo) -> None:
        """Replace an inference process (because it crashed, hung, or timed out).

        Frees any shared GPU/disk primitives the dead child may still hold (state-independently; see
        ``_release_held_primitives``), faults its in-flight job, then either respawns the slot or, if
        the slot has been replaced too many times in a short window, quarantines it (crash-loop
        circuit breaker) so a permanently-broken slot cannot spin in an unbounded respawn loop.
        """
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

        self._release_held_primitives(process_info)

        if (
            process_info.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL
            and process_info.last_job_referenced is not None
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
            # A slot crash/hang mid-job is retryable: the job is requeued to a fresh slot (bounded by
            # max_inference_attempts) rather than faulted outright. The crash gives no resource signal,
            # so it takes the ordinary retry, not the degraded path.
            self._job_tracker.handle_job_fault_now(
                faulted_job=job_to_remove,
                process_info=process_info,
                process_timeout=bridge_data.process_timeout,
                retryable=True,
            )

        replacements_in_window = self._record_slot_recovery(process_info.process_id)
        consecutive_start_failures = self._record_start_failure(process_info)
        crash_looped = replacements_in_window > CRASH_LOOP_MAX_REPLACEMENTS
        crash_on_start = consecutive_start_failures >= CRASH_LOOP_MAX_START_FAILURES
        will_quarantine = crash_looped or crash_on_start
        if not will_quarantine:
            quarantine_reason = ""
            recovery_reason = "inference process replaced (crashed or hung)"
        elif crash_on_start:
            quarantine_reason = (
                f"crash on start: {consecutive_start_failures} consecutive failures before reaching readiness"
            )
            recovery_reason = f"inference slot quarantined ({quarantine_reason})"
        else:
            quarantine_reason = (
                f"crash loop: {replacements_in_window} replacements within {CRASH_LOOP_WINDOW_SECONDS:.0f}s"
            )
            recovery_reason = f"inference slot quarantined ({quarantine_reason})"
        # Record the recovery while the process info still reflects the faulted state: ending the
        # process first overwrites last_process_state with PROCESS_ENDING and loses that diagnostic.
        self._log_recovery_diagnostics(process_info, recovery_reason)
        raw_exitcode = getattr(process_info.mp_process, "exitcode", None)
        exitcode = raw_exitcode if isinstance(raw_exitcode, int) else None
        self._action_ledger.record(
            LedgerEventType.PROCESS_REPLACED,
            process_id=process_info.process_id,
            os_pid=process_info.os_pid,
            launch_identifier=process_info.process_launch_identifier,
            job_id=str(job_to_remove.id_) if job_to_remove is not None else None,
            reason=recovery_reason,
            detail={"last_state": process_info.last_process_state.name, "exitcode": exitcode},
        )
        self._notify_process_recovery(process_info, recovery_reason)

        self._end_inference_process(process_info)
        self._num_process_recoveries += 1

        if will_quarantine:
            self._quarantine_inference_slot(process_info, quarantine_reason)
            return

        self._start_inference_process(process_info.process_id)

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
        if self._owned_registry is not None:
            self._owned_registry.clear()

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
            self._action_ledger.record(
                LedgerEventType.TIMEOUT_DETECTED,
                process_id=process_info.process_id,
                os_pid=process_info.os_pid,
                launch_identifier=process_info.process_launch_identifier,
                reason=error_message,
                detail={"state": state.name, "elapsed_s": round(time_elapsed, 1), "timeout_s": timeout},
            )
            if process_info.process_type == HordeProcessType.SAFETY:
                self._log_recovery_diagnostics(process_info, error_message)
                # Arm the replacement before driving it: `_replace_all_safety_process` no-ops unless the
                # flag is set, so omitting this (the historical bug) made the safety branch a silent no-op.
                self._initiate_safety_replacement()
                self._replace_all_safety_process()
            if process_info.process_type == HordeProcessType.INFERENCE:
                self._replace_inference_process(process_info)
            return True
        return False

    def _grade_running_inference(self) -> None:
        """Grade in-flight inference against its expected sampling time, escalating notices for slow jobs.

        The hard kill remains the ``inference_step_timeout`` in :meth:`replace_hung_processes`; this adds
        the softer, evidence-based rungs below it (a job measurably slower than its signature's expected
        time) so a slowdown is logged, audited, and counted toward the recovery-supervisor severity
        before the watchdog resorts to replacing the slot. A job with no expected time (cold start) is
        skipped, so this never fires on an uncalibrated worker.
        """
        now = time.time()
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.last_process_state != HordeProcessState.INFERENCE_STARTING:
                continue
            started = process_info.current_inference_started_at
            expected = process_info.current_job_expected_sampling_seconds
            if started is None or expected is None or expected <= 0:
                continue

            elapsed = now - started
            ratio = elapsed / expected
            level = 2 if ratio >= SLOWDOWN_WARN_RATIO else 1 if ratio >= SLOWDOWN_NOTICE_RATIO else 0
            if level <= process_info.current_job_slowdown_level:
                continue
            process_info.current_job_slowdown_level = level

            job_id = (
                str(process_info.last_job_referenced.id_) if process_info.last_job_referenced is not None else None
            )
            if level >= 2:
                self._num_slowdown_events += 1
                logger.warning(
                    f"Inference on process {process_info.process_id} is {ratio:.1f}x its expected sampling time "
                    f"({elapsed:.0f}s vs ~{expected:.0f}s); watching for a hang.",
                )
                self._action_ledger.record(
                    LedgerEventType.SLOWDOWN_DETECTED,
                    process_id=process_info.process_id,
                    os_pid=process_info.os_pid,
                    launch_identifier=process_info.process_launch_identifier,
                    job_id=job_id,
                    reason=f"{ratio:.1f}x expected sampling time",
                    detail={"elapsed_s": round(elapsed, 1), "expected_s": round(expected, 1)},
                )
            else:
                logger.info(
                    f"Inference on process {process_info.process_id} is running slower than expected "
                    f"({ratio:.1f}x ~{expected:.0f}s); not yet a concern.",
                )

    def replace_hung_processes(self) -> bool:
        """Replaces processes that haven't checked in since `process_timeout` seconds in bridgeData."""
        import threading

        bridge_data = self._runtime_config.bridge_data

        def timed_unset_recently_recovered() -> None:
            time.sleep(bridge_data.inference_step_timeout)
            self._recently_recovered = False

        now = time.time()

        # A live inference slot that has advanced past PROCESS_STARTING has proven it can initialise,
        # so clear any consecutive crash-on-start streak it accrued. Only slots that never get past
        # startup keep accumulating toward the crash-on-start breaker in `_replace_inference_process`.
        for live_process in self._process_map.values():
            if (
                live_process.process_type == HordeProcessType.INFERENCE
                and live_process.last_process_state != HordeProcessState.PROCESS_STARTING
            ):
                self._slot_consecutive_start_failures.pop(live_process.process_id, None)

        # Soft, evidence-based slowdown grading runs every tick (cheap, no side effects beyond logging
        # and audit) regardless of the recovery debounce below, which only gates hard replacement.
        self._grade_running_inference()

        any_replaced = False

        # Definitive crashes (the OS process has exited) are reaped immediately, even right after
        # another recovery. A dead child sends no further messages, so deferring its reap behind the
        # recent-recovery debounce would wedge the worker for the debounce window on every burst of
        # crashes (e.g. a replacement that also dies). Unbounded respawns are prevented by the
        # crash-loop circuit breaker in `_replace_inference_process`, not by this debounce.
        # Snapshot the values: recovering a process can mutate the map (safety end/delete/restart).
        for process_info in list(self._process_map.values()):
            if self._reap_if_crashed(process_info):
                any_replaced = True

        if any_replaced:
            self._recently_recovered = True
            threading.Thread(target=timed_unset_recently_recovered).start()

        # The hang/timeout heuristics below are debounced: a just-replaced process is still spinning
        # up and would otherwise trip the startup / all-processes-timed-out checks before it reports in.
        if self._recently_recovered:
            return any_replaced

        for process_info in list(self._process_map.values()):
            if self._process_map.is_stuck_on_inference(
                process_info.process_id,
                bridge_data.inference_step_timeout,
            ):
                logger.error(f"{process_info} seems to be stuck mid inference, replacing it")
                self._action_ledger.record(
                    LedgerEventType.TIMEOUT_DETECTED,
                    process_id=process_info.process_id,
                    os_pid=process_info.os_pid,
                    launch_identifier=process_info.process_launch_identifier,
                    reason="stuck mid inference (no step progress within inference_step_timeout)",
                )
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
                    (
                        bridge_data.process_timeout,
                        HordeProcessState.WAITING_FOR_JOB,
                        "seems to be stuck idle (silent) while there is work to do",
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
