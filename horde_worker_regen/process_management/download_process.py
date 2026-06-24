"""The dedicated background model-download process.

This process owns a hordelib ``SharedModelManager`` *without* a full ComfyUI init (listing and
downloading checkpoints only needs the model managers, not the inference stack). It reports a rich,
labelled status (phase, the current download with progress/feature/target path, the pending queue, and
failures) so the TUI and console can show exactly when, how, where, and why models download, and it
honours live pause and bandwidth-limit controls.

Behavioural notes grounded in a hordelib source trace:

- ``SharedModelManager.load_model_managers()`` reads the model reference from disk (offline): the
  parent process owns reference downloading. It is reported as the ``INITIALIZING`` phase and retried
  with backoff on failure.
- The first on-disk scan (``available_models``) is an existence check over the configured models; it is
  reported as the ``SCANNING`` phase so it never looks hung. Integrity (checksums) is verified lazily by
  ``validate_model`` after a download, not during this scan.
- ``download_file`` exposes a per-chunk ``callback(downloaded, total)`` but no pause/rate-limit. We
  implement both inside that callback (block while paused; sleep to cap kB/s).

The process lives outside the main process map: it serves no jobs and must not be swept up by the
inference/safety hung-process logic. A dedicated control thread drains its pipe so pause/resume and
rate-limit changes take effect mid-download (the worker loop is blocked inside the download otherwise).
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock, Semaphore

from loguru import logger

from horde_worker_regen.model_download_core import (
    UNKNOWN_DOWNLOAD_HOST,
    ChunkPacer,
    DownloadAborted,
    ModelProgress,
    download_host_for_url,
    download_one_model,
    ensure_aux_model_present,
)
from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.download_scheduler import (
    DownloadKind,
    DownloadTask,
    HostAwareDownloadScheduler,
)
from horde_worker_regen.process_management.horde_process import HordeProcess, HordeProcessType, WorkerCapability
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeDownloadAvailabilityMessage,
    HordeDownloadControlMessage,
)
from horde_worker_regen.process_management.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadFailure,
    DownloadItem,
    DownloadPhase,
    DownloadStatusSnapshot,
)

DOWNLOAD_PROCESS_ID = 9000
"""The reserved process id for the singleton download process (high to avoid inference-slot collisions)."""

_STATUS_EMIT_INTERVAL_SECONDS = 0.5
"""Minimum spacing between progress status messages during a download."""
_LOAD_RETRY_BACKOFF_SECONDS = (5.0, 15.0, 30.0, 60.0)
"""Backoff schedule for retrying the (networked) model-manager load."""
_MAX_DOWNLOAD_ATTEMPTS = 3
"""How many times a per-file fetch (image/aux) is re-attempted after a transient failure before giving up."""
_RETRY_BACKOFF_SECONDS = 10.0
"""Delay before re-queuing a failed per-file fetch (kept short; the scheduler then re-admits it)."""

FEATURE_IMAGE_MODEL = "image model"
FEATURE_LORA = "LoRa (default set)"
FEATURE_CONTROLNET = "ControlNet"
FEATURE_CONTROLNET_ANNOTATORS = "ControlNet annotators"
FEATURE_MISCELLANEOUS = "miscellaneous (SDXL)"
FEATURE_SAFETY = "safety models"


def _post_processing_feature(name: str) -> str:
    return f"post-processing ({name})"


@dataclass
class _TaskRuntime:
    """Live per-download state for one in-flight task: its progress snapshot, pacer, and cancel flag.

    One runtime exists per executing task, so several downloads can report progress and be cancelled
    independently. ``cancelled`` is read by the task's chunk-callback abort predicate (a config removal
    sets it); ``status`` is the snapshot surfaced to the TUI for this task.
    """

    status: CurrentDownloadStatus
    pacer: ChunkPacer
    cancelled: bool = False


class HordeDownloadProcess(HordeProcess):
    """A background process that ensures requested image models (and aux models) are present on disk."""

    capabilities = WorkerCapability(0)

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        download_bandwidth_semaphore: Semaphore,
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
        max_parallel_downloads: int = 4,
        per_host_concurrency: int = 1,
        connections_per_file: int = 4,
    ) -> None:
        """Initialise the download process state (model managers are loaded in the main loop)."""
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
        )
        self.process_type = HordeProcessType.DOWNLOAD

        self._nsfw = nsfw
        self._allow_lora = allow_lora
        self._allow_controlnet = allow_controlnet
        self._allow_sdxl_controlnet = allow_sdxl_controlnet
        self._allow_post_processing = allow_post_processing
        self._purge_loras = purge_loras
        self._amd_gpu = amd_gpu
        self._directml = directml
        self._download_bandwidth_semaphore = download_bandwidth_semaphore
        # The cross-process download slot (a size-1 semaphore) is held once while *any* executor task is
        # active, so internal threading keeps the "only the download process downloads" invariant. A
        # refcount + an "acquired" event coordinate the threads: the first task to begin actually performs
        # the (blocking) acquire and signals the event; concurrent tasks wait on the event so none fetches
        # until the slot is genuinely held. The last task to finish clears the event and releases.
        self._bandwidth_lock = threading.Lock()
        self._bandwidth_count = 0
        self._bandwidth_held = False
        self._bandwidth_ready = threading.Event()

        self._lock = threading.Lock()
        # Per-manager locks serialize the hordelib calls that mutate one manager's shared lists
        # (available_models/tainted_models) and our reads of them, without serializing downloads on
        # *different* managers (which own independent state and may run truly in parallel). Created lazily.
        self._manager_locks: dict[str, threading.Lock] = {}
        # Per-task retry accounting (keyed by the scheduler dedup key) so a transient fetch failure is
        # re-attempted a bounded number of times instead of being abandoned until the next config reload.
        self._attempts: dict[tuple[DownloadKind, str, str], int] = {}
        # Image-model names requested but not yet built into scheduler tasks (host resolution needs the
        # managers, which load after the control thread starts, so requests are staged here first).
        self._pending_image_models: list[str] = []
        self._failures: list[DownloadFailure] = []
        self._present: list[str] = []
        self._phase = DownloadPhase.INITIALIZING
        self._paused = paused
        self._rate_limit_kbps = rate_limit_kbps if (rate_limit_kbps or 0) > 0 else None
        self._error_message: str | None = None

        # The host-aware admission policy and the live per-task runtimes (keyed by the scheduler's
        # dedup key). Several executor threads drain the scheduler; each running task owns a runtime.
        self._scheduler = HostAwareDownloadScheduler(
            max_parallel_downloads=max_parallel_downloads,
            per_host_concurrency=per_host_concurrency,
        )
        # The executor pool is grown lazily to the current global limit (never shrunk): an idle thread just
        # blocks cheaply in ``scheduler.acquire``. This is what makes a *live* raise of
        # ``max_parallel_downloads`` actually take effect, rather than being capped at the boot-time size.
        self._desired_executor_threads = max(1, max_parallel_downloads)
        # Max concurrent connections per single large file (forwarded to the engine, which segments a big
        # file across that many ranged connections to raise its rate). Retuned live via the control message.
        self._connections_per_file = max(1, connections_per_file)
        self._active: dict[tuple[DownloadKind, str, str], _TaskRuntime] = {}
        self._running_count = 0
        self._executor_threads: list[threading.Thread] = []
        self._executor_seq = 0
        """Monotonic counter for unique executor-thread names across self-heal respawns."""

        self._aux_requested = False
        self._aux_enqueued = False
        # The safety models (DeepDanbooru + CLIP) are required for every image job, so they are ensured
        # unconditionally (not gated behind the optional aux pass). ``_safety_present`` is reported to the
        # parent, which defers the safety-process launch until it is True; ``_safety_ensured`` guards the
        # one-shot ensure so a failed attempt is not retried (the parent's grace fallback then starts the
        # safety process, which surfaces the real error). ``_safety_enqueued`` guards the one-shot enqueue.
        self._safety_present = False
        self._safety_ensured = False
        self._safety_enqueued = False
        # On-disk readiness of the gated aux features, recomputed event-driven (after the scan, after each
        # download, and when the aux pass is enqueued) and reported to the parent so it offers a feature to
        # the Horde only once its models/annotators have landed. None means undeterminable (manager not
        # loaded), which the parent reads as "do not gate".
        self._controlnet_present: bool | None = None
        self._sdxl_controlnet_present: bool | None = None
        self._post_processing_present: bool | None = None
        self._reload_requested = False
        # Set after a completed download that changed on-disk references; emitted once on the next
        # status snapshot so the parent can broadcast a reload to the inference subprocesses.
        self._reference_changed_pending = False

        self._last_status_emit = 0.0

    # region status reporting

    def _build_status(self, phase: DownloadPhase) -> DownloadStatusSnapshot:
        scheduled = self._scheduler.pending_snapshot()
        with self._lock:
            staged = [
                DownloadItem(model_name=name, feature=FEATURE_IMAGE_MODEL) for name in self._pending_image_models
            ]
            queued = [DownloadItem(model_name=task.model_name, feature=task.feature) for task in scheduled]
            failures = list(self._failures)
            present = list(self._present)
            active = [runtime.status for runtime in self._active.values()]
            paused = self._paused
            rate = self._rate_limit_kbps
        current = active[0] if active else None
        effective_phase = DownloadPhase.PAUSED if paused and phase == DownloadPhase.DOWNLOADING else phase
        return DownloadStatusSnapshot(
            phase=effective_phase,
            current=current,
            active=active,
            pending=staged + queued,
            failures=failures,
            present_model_names=present,
            paused=paused,
            rate_limit_kbps=rate,
            error_message=self._error_message,
        )

    def _send_status(self, phase: DownloadPhase, *, scan_complete: bool = True, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_status_emit) < _STATUS_EMIT_INTERVAL_SECONDS:
            return
        self._last_status_emit = now
        self._phase = phase
        status = self._build_status(phase)
        with self._lock:
            reference_changed = self._reference_changed_pending
            self._reference_changed_pending = False
        message = HordeDownloadAvailabilityMessage(
            process_id=self.process_id,
            process_launch_identifier=self.process_launch_identifier,
            info=f"download status: {phase.value}",
            available_model_names=status.present_model_names,
            currently_downloading=status.current.model_name if status.current is not None else None,
            pending_downloads=[item.model_name for item in status.pending],
            failed_downloads=[failure.model_name for failure in status.failures],
            scan_complete=scan_complete,
            safety_models_present=self._safety_present,
            safety_models_attempted=self._safety_ensured,
            controlnet_present=self._controlnet_present,
            sdxl_controlnet_present=self._sdxl_controlnet_present,
            post_processing_present=self._post_processing_present,
            status=status,
            reference_changed=reference_changed,
        )
        try:
            self.process_message_queue.put(message)
        except Exception as e:  # noqa: BLE001 - a status emit must never abort an in-flight download
            # If the parent's queue is gone the control thread will see the pipe break and end us; a
            # transient put error here should not propagate up through the chunk callback / orchestration.
            logger.debug(f"Download process: status emit failed: {type(e).__name__}: {e}")

    # endregion

    # region control thread (drains the pipe so pause/rate/requests apply mid-download)

    def _control_loop(self) -> None:
        while not self._end_process:
            try:
                while self.pipe_connection.poll():
                    self._handle_control_message(self.pipe_connection.recv())
            except (EOFError, OSError):
                self._end_process = True
                return
            time.sleep(0.05)

    def _apply_live_gating(self, message: HordeDownloadControlMessage) -> bool:
        """Apply any download-gating flags carried live in a control message; return whether any changed.

        These (nsfw / allow_lora / allow_controlnet / allow_sdxl_controlnet / allow_post_processing / purge)
        were once construction-time only, so a config change to them restarted the process. They are applied
        live instead; the caller re-arms the one-shot aux pass when this returns True, so a newly-enabled
        category (e.g. allow_lora flipped on) is fetched without a restart. Caller holds ``self._lock``.
        """
        changed = False
        if message.set_nsfw is not None and message.set_nsfw != self._nsfw:
            self._nsfw = message.set_nsfw
            changed = True
        if message.set_allow_lora is not None and message.set_allow_lora != self._allow_lora:
            self._allow_lora = message.set_allow_lora
            changed = True
        if message.set_allow_controlnet is not None and message.set_allow_controlnet != self._allow_controlnet:
            self._allow_controlnet = message.set_allow_controlnet
            changed = True
        new_sdxl = message.set_allow_sdxl_controlnet
        if new_sdxl is not None and new_sdxl != self._allow_sdxl_controlnet:
            self._allow_sdxl_controlnet = new_sdxl
            changed = True
        new_pp = message.set_allow_post_processing
        if new_pp is not None and new_pp != self._allow_post_processing:
            self._allow_post_processing = new_pp
            changed = True
        if message.set_purge_loras is not None and message.set_purge_loras != self._purge_loras:
            self._purge_loras = message.set_purge_loras
            changed = True
        return changed

    def _handle_control_message(self, message: object) -> None:
        if isinstance(message, HordeControlMessage) and message.control_flag == HordeControlFlag.END_PROCESS:
            self._end_process = True
            return
        if isinstance(message, HordeControlMessage) and message.control_flag == HordeControlFlag.RELOAD_MODEL_DATABASE:
            # Defer the reload to the main loop so it never races the download thread's manager reads.
            with self._lock:
                self._reload_requested = True
            return
        if not isinstance(message, HordeDownloadControlMessage):
            logger.warning(f"Download process received unexpected control message: {type(message).__name__}")
            return

        # Reconcile against the authoritative configured set (a config edit that removed a model): drop it
        # from the staging buffer + the scheduler queue, and cancel it if it is an in-flight image-model
        # task. Only image-model work is touched; required safety/aux tasks are gated out of the cancel by
        # their kind, so a model removal can never stop them.
        desired = set(message.desired_image_models) if message.desired_image_models is not None else None
        changed = False
        with self._lock:
            if desired is not None:
                kept = [name for name in self._pending_image_models if name in desired]
                if len(kept) != len(self._pending_image_models):
                    self._pending_image_models = kept
                    changed = True
                for (kind, _manager_key, model_name), runtime in self._active.items():
                    if kind is DownloadKind.IMAGE_MODEL:
                        runtime.cancelled = model_name not in desired

            staged_or_active = set(self._pending_image_models) | {
                name for (kind, _mk, name) in self._active if kind is DownloadKind.IMAGE_MODEL
            }
            present = set(self._present)
            for model_name in message.model_names:
                if model_name in present or model_name in staged_or_active:
                    continue
                self._pending_image_models.append(model_name)
                staged_or_active.add(model_name)
                changed = True
            if message.download_aux:
                self._aux_requested = True
            if message.set_paused is not None and message.set_paused != self._paused:
                self._paused = message.set_paused
                changed = True
            if message.set_rate_limit_kbps is not None:
                self._rate_limit_kbps = message.set_rate_limit_kbps if message.set_rate_limit_kbps > 0 else None
                changed = True
            if self._apply_live_gating(message):
                # Re-arm the one-shot aux pass so a newly-enabled category downloads without a process
                # restart; the pass is idempotent (present models are skipped), so replaying a toggle is safe.
                self._aux_requested = True
                self._aux_enqueued = False
                changed = True

        if message.set_max_parallel_downloads is not None or message.set_per_host_concurrency is not None:
            # Retune live; a raised limit wakes blocked executor threads to claim newly-admissible tasks.
            self._scheduler.set_limits(
                max_parallel_downloads=message.set_max_parallel_downloads,
                per_host_concurrency=message.set_per_host_concurrency,
            )
            if message.set_max_parallel_downloads is not None:
                # Grow the pool so a *raised* global limit has threads to use; the scheduler still gates the
                # actual concurrency, so a later lower limit just leaves the surplus threads idle.
                self._ensure_executor_threads(message.set_max_parallel_downloads)
            changed = True

        if message.set_connections_per_file is not None:
            # Applies to the next file fetched; in-flight segmented downloads keep their existing connections.
            self._connections_per_file = max(1, message.set_connections_per_file)
            changed = True

        if desired is not None:
            removed = self._scheduler.prune(
                keep=lambda task: task.kind is not DownloadKind.IMAGE_MODEL or task.model_name in desired,
            )
            if removed:
                changed = True
        if changed:
            self._send_status(self._phase, force=True)

    # endregion

    @override
    def main_loop(self) -> None:
        """Load managers (gracefully), scan disk, then service the download queue until told to end."""
        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)
        threading.Thread(target=self._control_loop, name="download-control", daemon=True).start()

        if self._connections_per_file > 1:
            # Loud, once-per-start: the default (>1) trades resumability for single-file speed, so an operator
            # who restarts mid-download (or has a flaky link) understands why a large file starts over.
            logger.warning(
                "Download process: large files use {} connections each for speed; these downloads CANNOT be "
                "resumed (an interrupted large download restarts from scratch). Set "
                "download_connections_per_file: 1 to keep resumable single-stream downloads.",
                self._connections_per_file,
            )
        else:
            logger.info("Download process: single-stream downloads (download_connections_per_file=1); resumable.")

        self._send_status(DownloadPhase.INITIALIZING, scan_complete=False, force=True)
        if not self._load_managers_with_retry():
            self._send_status(DownloadPhase.ERROR, scan_complete=False, force=True)
            self._idle_until_end()
            return

        self._send_status(DownloadPhase.SCANNING, scan_complete=False, force=True)
        self._refresh_present()
        # Cheap existence probe so a warm worker reports the safety models present in its first
        # authoritative report; the safety process then starts immediately instead of being deferred.
        # A cold worker leaves ``_safety_ensured`` False and the tick loop downloads them (visibly).
        if self._safety_models_present_on_disk():
            self._safety_present = True
            self._safety_ensured = True
        # Probe gated-feature presence too, so a warm worker reports its features ready in the first
        # authoritative report and the parent advertises them without waiting for an aux pass.
        self._refresh_feature_presence()
        self._send_status(DownloadPhase.IDLE, scan_complete=True, force=True)

        logger.info(
            f"Download process ready: parallel={self._desired_executor_threads} "
            f"lora={self._allow_lora} controlnet={self._allow_controlnet} "
            f"post_processing={self._allow_post_processing} nsfw={self._nsfw}",
        )
        self._ensure_executor_threads(self._desired_executor_threads)
        while not self._end_process:
            try:
                progressed = self._orchestrate()
            except Exception as e:  # noqa: BLE001 - orchestration must never crash the (oracle) download process
                logger.exception(f"Download process: orchestration error (continuing): {type(e).__name__}: {e}")
                progressed = False
            # Self-heal: respawn any executor thread that died unexpectedly, so downloads never silently
            # stall while the process still looks alive (the worst case for the availability oracle).
            self._ensure_executor_threads(self._desired_executor_threads)
            if not progressed:
                time.sleep(0.1)

        self._scheduler.close()
        self._bandwidth_ready.set()  # release any task waiting on the slot so its thread can exit
        for thread in list(self._executor_threads):
            thread.join(timeout=2.0)
        self._send_status(DownloadPhase.IDLE, force=True)
        logger.info("Download process ended")
        sys.exit(0)

    def _ensure_executor_threads(self, target_count: int) -> None:
        """Ensure at least *target_count* LIVE executor threads exist (grows / self-heals; never shrinks).

        Called at boot with the configured limit, on a live config change that *raises*
        ``max_parallel_downloads``, and every main-loop tick. Counting only live threads means a thread
        that died unexpectedly is respawned, so downloads never silently stall while the process still
        looks alive. A lowered limit needs no change: surplus threads simply block in ``scheduler.acquire``.
        """
        if self._end_process:
            return
        with self._lock:
            target = max(1, target_count)
            # Drop references to any dead threads, then top up to the target live count.
            self._executor_threads = [thread for thread in self._executor_threads if thread.is_alive()]
            existing = len(self._executor_threads)
            if existing >= target:
                return
            new_threads = [
                threading.Thread(
                    target=self._executor_loop,
                    name=f"download-exec-{self._executor_seq + offset}",
                    daemon=True,
                )
                for offset in range(target - existing)
            ]
            self._executor_seq += len(new_threads)
            self._executor_threads.extend(new_threads)
        for thread in new_threads:
            thread.start()

    def _signal(self, _sig: int, _frame: object) -> None:
        self._end_process = True

    def _idle_until_end(self) -> None:
        while not self._end_process:
            time.sleep(0.2)

    def _load_managers_with_retry(self) -> bool:
        """Load hordelib's model managers, retrying with backoff.

        References are read from disk (offline): the parent process owns reference downloading. This
        process still downloads model *weights*; it just learns *what* to download from the on-disk
        reference the parent wrote.
        """
        from hordelib.api import SharedModelManager

        from horde_worker_regen.reference_helper import ensure_offline_reference_manager

        for attempt in range(len(_LOAD_RETRY_BACKOFF_SECONDS) + 1):
            if self._end_process:
                return False
            try:
                ensure_offline_reference_manager()
                SharedModelManager(do_not_load_model_mangers=True)
                SharedModelManager.load_model_managers(multiprocessing_lock=self.disk_lock)
                if SharedModelManager.manager.compvis is None:
                    raise RuntimeError("compvis model manager failed to load")
                self._error_message = None
                return True
            except Exception as e:  # noqa: BLE001 - report and retry rather than crash the worker
                self._error_message = f"{type(e).__name__}: {e}"
                logger.error(f"Download process: failed to load model managers: {self._error_message}")
                if attempt >= len(_LOAD_RETRY_BACKOFF_SECONDS):
                    return False
                self._send_status(DownloadPhase.ERROR, scan_complete=False, force=True)
                time.sleep(_LOAD_RETRY_BACKOFF_SECONDS[attempt])
        return False

    def _refresh_present(self) -> None:
        from hordelib.api import SharedModelManager

        compvis = SharedModelManager.manager.compvis
        # Snapshot under the same per-manager lock the downloads take, so the read never races a sibling
        # fetch mutating compvis.available_models.
        if compvis is None:
            with self._lock:
                self._present = []
            return
        with self._manager_lock("compvis"):
            present = sorted(compvis.available_models)
        with self._lock:
            self._present = present

    def _reload_model_database(self) -> None:
        """Reload model manager references from disk (offline) after a parent reference refresh."""
        from hordelib.api import SharedModelManager

        try:
            SharedModelManager.manager.reload_database()
            logger.info("Download process reloaded model database from disk")
        except Exception as e:  # noqa: BLE001 - a reload failure must not crash the download process
            logger.error(f"Download process failed to reload model database: {type(e).__name__}: {e}")

    def _orchestrate(self) -> bool:
        """Build pending work into host-tagged scheduler tasks and emit status; executors do the fetching.

        Runs on the main loop. It stages requested image models, the one-shot safety task, and (on
        request) the aux tasks into the scheduler, then reports status. The actual downloads happen on the
        executor-thread pool (:meth:`_executor_loop`), so several hosts download at once. A reference reload
        is deferred until the process is fully idle so it never races an executor reading the managers.
        """
        with self._lock:
            paused = self._paused
            reload_requested = self._reload_requested
            to_build = [] if paused else list(self._pending_image_models)
            if to_build:
                self._pending_image_models = []
            build_safety = not self._safety_enqueued and not self._safety_ensured and not paused
            if build_safety:
                self._safety_enqueued = True
            build_aux = self._aux_requested and not self._aux_enqueued and not paused
            if build_aux:
                self._aux_enqueued = True

        if reload_requested and self._running_count == 0 and not self._scheduler.has_work():
            with self._lock:
                self._reload_requested = False
            self._reload_model_database()
            self._refresh_present()
            self._send_status(DownloadPhase.IDLE, force=True)
            return True

        did = False
        if to_build:
            self._enqueue_image_tasks(to_build)
            did = True
        if build_safety:
            self._enqueue_safety_task()
            did = True
        if build_aux:
            self._enqueue_aux_tasks()
            did = True

        downloading = self._scheduler.active_count > 0
        if paused:
            self._send_status(DownloadPhase.PAUSED)
        elif downloading:
            self._send_status(DownloadPhase.DOWNLOADING)
        else:
            self._send_status(DownloadPhase.IDLE)
        return did or downloading

    # region task building (host-tagged, on the main loop)

    def _enqueue_image_tasks(self, model_names: list[str]) -> None:
        """Turn requested image-model names into host-tagged IMAGE_MODEL tasks (skipping present ones)."""
        from hordelib.api import SharedModelManager

        compvis = SharedModelManager.manager.compvis
        if compvis is None:
            for name in model_names:
                self._record_failure(name, FEATURE_IMAGE_MODEL, "compvis manager unavailable")
            return
        target_dir = str(compvis.model_folder_path)
        tasks = [
            DownloadTask(
                kind=DownloadKind.IMAGE_MODEL,
                model_name=name,
                host=self._host_for(compvis, name),
                feature=FEATURE_IMAGE_MODEL,
                target_dir=target_dir,
            )
            for name in model_names
            if not compvis.is_model_available(name)
        ]
        if tasks:
            self._scheduler.enqueue_many(tasks)

    def _enqueue_safety_task(self) -> None:
        """Enqueue the one-shot required-safety-models task (its source host is opaque, so 'unknown')."""
        if self._safety_present:
            with self._lock:
                self._safety_ensured = True
            return
        self._scheduler.enqueue(
            DownloadTask(
                kind=DownloadKind.SAFETY,
                model_name="safety models",
                host=UNKNOWN_DOWNLOAD_HOST,
                feature=FEATURE_SAFETY,
            ),
        )

    def _enqueue_aux_tasks(self) -> None:
        """Enumerate the enabled aux categories into per-model host-tagged tasks (LoRa/annotators coarse)."""
        from hordelib.api import SharedModelManager

        manager = SharedModelManager.manager
        tasks: list[DownloadTask] = []

        if self._allow_lora and manager.lora is not None:
            tasks.append(
                DownloadTask(
                    kind=DownloadKind.DEFAULT_LORAS,
                    model_name="default LoRas",
                    host="civitai.com",
                    feature=FEATURE_LORA,
                    target_dir=str(manager.lora.model_folder_path),
                ),
            )
        if self._allow_post_processing:
            for manager_key, label in (("gfpgan", "GFPGAN"), ("esrgan", "ESRGAN"), ("codeformer", "CodeFormer")):
                post_processor = getattr(manager, manager_key, None)
                if post_processor is not None:
                    tasks.extend(self._aux_model_tasks(post_processor, manager_key, _post_processing_feature(label)))
        if self._allow_sdxl_controlnet and manager.miscellaneous is not None:
            tasks.extend(self._aux_model_tasks(manager.miscellaneous, "miscellaneous", FEATURE_MISCELLANEOUS))
        if self._allow_controlnet and manager.controlnet is not None:
            tasks.extend(self._controlnet_tasks(manager.controlnet))
            # The annotator task does a full ComfyUI init + preload; only enqueue it when the annotators
            # are not already on disk, so a warm worker neither reports a phantom "annotators" download
            # nor pays for an unnecessary init. Presence-unknown falls through to enqueue (preload is
            # idempotent), matching the prior always-ensure behaviour without the false display.
            if not self._annotators_present():
                tasks.append(
                    DownloadTask(
                        kind=DownloadKind.ANNOTATORS,
                        model_name="annotators",
                        host=UNKNOWN_DOWNLOAD_HOST,
                        feature=FEATURE_CONTROLNET_ANNOTATORS,
                        # Runs a full ComfyUI/torch init that mutates global state, so it must not run
                        # alongside other downloads sharing the model managers.
                        exclusive=True,
                    ),
                )
        if tasks:
            self._scheduler.enqueue_many(tasks)
        # The aux managers are now loaded; probe their presence so a feature whose models are already on
        # disk is reported ready immediately, rather than only after the first completed download.
        self._refresh_feature_presence()

    @staticmethod
    def _annotators_present() -> bool:
        """Whether the ControlNet annotators are already on disk (False when undeterminable).

        Uses hordelib's import-safe on-disk marker check (no ComfyUI init / GPU). Failing to False means
        an unknown is treated as "not present", so the idempotent preload still runs and fetches them if
        actually missing; it just avoids skipping a genuinely-needed fetch.
        """
        try:
            from hordelib.preload import controlnet_annotators_present

            return bool(controlnet_annotators_present())
        except Exception as e:  # noqa: BLE001 - presence is best-effort; a probe failure must not crash
            logger.debug(f"Download process: could not determine annotator presence: {type(e).__name__} {e}")
            return False

    def _refresh_feature_presence(self) -> None:
        """Recompute on-disk readiness for the gated aux features from the loaded managers (cached).

        Cheap and event-driven (run after the disk scan, after each download completes, and when the aux
        pass is enqueued), so the half-second status emit can report presence without re-statting every
        model on every tick. A manager that is not loaded (its feature is not opted in) leaves that
        feature's presence None, so the parent does not gate on an unknown. ControlNet readiness also
        requires the annotators; SDXL-ControlNet additionally requires the miscellaneous models.
        """
        try:
            from hordelib.api import SharedModelManager

            manager = SharedModelManager.manager
        except Exception as e:  # noqa: BLE001 - presence is best-effort; a probe failure must not crash
            logger.debug(f"Download process: feature-presence probe failed: {type(e).__name__}: {e}")
            return
        if manager is None:
            return

        annotators = self._annotators_present()

        controlnet_models = self._manager_all_present(manager, "controlnet", exclude_substring="sdxl")
        controlnet = None if controlnet_models is None else (controlnet_models and annotators)

        sdxl_models = self._manager_all_present(manager, "controlnet", require_substring="sdxl")
        miscellaneous = self._manager_all_present(manager, "miscellaneous")
        if sdxl_models is None or miscellaneous is None:
            sdxl_controlnet = None
        else:
            sdxl_controlnet = sdxl_models and miscellaneous and annotators

        post_results = [self._manager_all_present(manager, key) for key in ("gfpgan", "esrgan", "codeformer")]
        loaded_post = [result for result in post_results if result is not None]
        post_processing = all(loaded_post) if loaded_post else None

        with self._lock:
            self._controlnet_present = controlnet
            self._sdxl_controlnet_present = sdxl_controlnet
            self._post_processing_present = post_processing

    def _manager_all_present(
        self,
        manager: object,
        manager_key: str,
        *,
        require_substring: str | None = None,
        exclude_substring: str | None = None,
    ) -> bool | None:
        """Whether every (optionally name-filtered) model in ``manager.<key>``'s reference is on disk.

        Returns None when that sub-manager is not loaded (so the feature is undeterminable rather than
        falsely "not present"). Reads under the per-manager lock the downloads take, so the on-disk
        check never races a sibling fetch mutating the manager's available set.
        """
        sub_manager = getattr(manager, manager_key, None)
        if sub_manager is None:
            return None
        try:
            with self._manager_lock(manager_key):
                for model_name in sub_manager.model_reference:  # type: ignore[attr-defined]
                    lowered = model_name.lower()
                    if require_substring is not None and require_substring not in lowered:
                        continue
                    if exclude_substring is not None and exclude_substring in lowered:
                        continue
                    if not sub_manager.is_model_available(model_name):  # type: ignore[attr-defined]
                        return False
            return True
        except Exception as e:  # noqa: BLE001 - presence is best-effort; a probe failure must not crash
            logger.debug(f"Download process: presence probe for {manager_key} failed: {type(e).__name__}: {e}")
            return None

    def _aux_model_tasks(self, manager: object, manager_key: str, feature: str) -> list[DownloadTask]:
        """Build host-tagged AUX_MODEL tasks for every not-yet-present model in *manager*'s reference."""
        target_dir = str(manager.model_folder_path)  # type: ignore[attr-defined]
        tasks: list[DownloadTask] = []
        for model_name in manager.model_reference:  # type: ignore[attr-defined]
            if manager.is_model_available(model_name):  # type: ignore[attr-defined]
                continue
            tasks.append(
                DownloadTask(
                    kind=DownloadKind.AUX_MODEL,
                    model_name=model_name,
                    host=self._host_for(manager, model_name),
                    feature=feature,
                    manager_key=manager_key,
                    target_dir=target_dir,
                ),
            )
        return tasks

    def _controlnet_tasks(self, controlnet: object) -> list[DownloadTask]:
        """Build ControlNet model tasks, honouring the SDXL-controlnet opt-in gate."""
        target_dir = str(controlnet.model_folder_path)  # type: ignore[attr-defined]
        tasks: list[DownloadTask] = []
        for cn_model in controlnet.model_reference:  # type: ignore[attr-defined]
            if controlnet.is_model_available(cn_model):  # type: ignore[attr-defined]
                continue
            if "sdxl" in cn_model.lower() and not self._allow_sdxl_controlnet:
                continue
            tasks.append(
                DownloadTask(
                    kind=DownloadKind.AUX_MODEL,
                    model_name=cn_model,
                    host=self._host_for(controlnet, cn_model),
                    feature=FEATURE_CONTROLNET,
                    manager_key="controlnet",
                    target_dir=target_dir,
                ),
            )
        return tasks

    @staticmethod
    def _host_for(manager: object, model_name: str) -> str:
        """Resolve a model's source host from its first download URL (for per-host scheduling)."""
        try:
            downloads = manager.get_model_download(model_name)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - a host lookup must never crash; fall back to the unknown bucket
            return UNKNOWN_DOWNLOAD_HOST
        for entry in downloads:
            host = download_host_for_url(entry.get("file_url"))
            if host != UNKNOWN_DOWNLOAD_HOST:
                return host
        return UNKNOWN_DOWNLOAD_HOST

    # endregion

    # region executor pool (downloads run here, several hosts at once)

    def _executor_loop(self) -> None:
        """One download worker: claim an admissible task from the scheduler and run it, until shutdown.

        The body is fully guarded: an unexpected error in a single task is logged and the loop continues,
        so a thread can never die and silently remove download capacity from the (oracle) process. The
        main loop additionally respawns a thread that does somehow exit (belt and suspenders).
        """
        while not self._end_process:
            try:
                if self._paused:
                    time.sleep(0.1)
                    continue
                task = self._scheduler.acquire(timeout=0.2)
                if task is None:
                    continue
                try:
                    self._run_task(task)
                finally:
                    self._scheduler.release(task)
            except Exception as e:  # noqa: BLE001 - a single task must never kill its executor thread
                logger.exception(f"Download process: executor loop error (continuing): {type(e).__name__}: {e}")
                time.sleep(0.2)

    def _run_task(self, task: DownloadTask) -> None:
        """Execute one task: register its runtime, fetch it (by kind), retry/record, tear the runtime down."""
        runtime = _TaskRuntime(
            status=CurrentDownloadStatus(
                model_name=task.model_name,
                feature=task.feature,
                target_dir=task.target_dir,
                host=task.host,
            ),
            pacer=ChunkPacer(),
        )
        self._begin_task(task, runtime)
        success = False
        aborted = False
        reason: str | None = None
        try:
            success = self._dispatch_task(task, self._make_callback(task, runtime))
        except DownloadAborted:
            # A cancel (config removal) or shutdown: terminal, not a failure, and never retried.
            aborted = True
        except OSError as e:
            reason = "out of disk space" if e.errno == 28 else f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001 - any download error is a recorded failure, not a crash
            reason = f"{type(e).__name__}: {e}"
        finally:
            self._end_task(task)
            self._refresh_present()
            # A completed aux download may have made a gated feature ready; recompute so the next report
            # advertises it (or, on a failure mid-set, keeps withholding it).
            self._refresh_feature_presence()
            with self._lock:
                self._reference_changed_pending = True

        if aborted:
            self._forget_attempts(task)
        elif success:
            self._clear_failure(task.model_name)
            self._forget_attempts(task)
        else:
            self._record_failure(task.model_name, task.feature, reason or "download failed")
            self._maybe_retry(task, reason or "download failed")
        self._send_status(self._phase, force=True)

    def _maybe_retry(self, task: DownloadTask, reason: str) -> None:
        """Re-enqueue a failed per-file fetch a bounded number of times (transient-network resilience).

        Only the per-file fetches (image/aux) retry; the coarse kinds (safety/LoRa/annotators) own their
        own internal retry/wait semantics. A cancelled or shutting-down task is never retried.
        """
        if task.kind not in (DownloadKind.IMAGE_MODEL, DownloadKind.AUX_MODEL):
            return
        if self._end_process:
            return
        with self._lock:
            attempts = self._attempts.get(task.dedup_key, 0) + 1
            self._attempts[task.dedup_key] = attempts
        if attempts > _MAX_DOWNLOAD_ATTEMPTS:
            logger.error(f"Download process: giving up on {task.model_name} after {attempts - 1} retries ({reason})")
            return
        logger.warning(
            f"Download process: {task.model_name} failed ({reason}); retry {attempts}/{_MAX_DOWNLOAD_ATTEMPTS} "
            f"in {_RETRY_BACKOFF_SECONDS:.0f}s",
        )
        # Back off on this task's own executor thread, then re-queue; a config removal that lands meanwhile
        # prunes the re-queued task (it is an IMAGE_MODEL not in the desired set) so the retry self-cancels.
        for _ in range(int(_RETRY_BACKOFF_SECONDS * 10)):
            if self._end_process:
                return
            time.sleep(0.1)
        self._scheduler.enqueue(task)

    def _dispatch_task(self, task: DownloadTask, callback: Callable[[int, int], None]) -> bool:
        """Run *task* against the right manager method for its kind; return whether it succeeded.

        Per-file fetches hold the per-manager lock so the hordelib call that mutates that manager's shared
        model lists never races a sibling download (or our present-set read) on the *same* manager;
        downloads on *different* managers still run truly in parallel.
        """
        from hordelib.api import SharedModelManager

        manager = SharedModelManager.manager
        if task.kind is DownloadKind.IMAGE_MODEL:
            compvis = manager.compvis
            if compvis is None:
                self._record_failure(task.model_name, task.feature, "compvis manager unavailable")
                return False
            with self._manager_lock("compvis"):
                connections = self._connections_per_file
                if download_one_model(compvis, task.model_name, callback=callback, connections=connections):
                    logger.success(f"Download process: downloaded {task.model_name}")
                    return True
            return False
        if task.kind is DownloadKind.AUX_MODEL:
            aux_manager = getattr(manager, task.manager_key, None)
            if aux_manager is None:
                return False
            with self._manager_lock(task.manager_key):
                # Validated fetch (sha256-where-known, else presence) with a re-download on mismatch, so a
                # truncated aux file is repaired here instead of being trusted and faulting a later job.
                return ensure_aux_model_present(
                    aux_manager,
                    task.model_name,
                    callback=callback,
                    connections=self._connections_per_file,
                )
        if task.kind is DownloadKind.SAFETY:
            return self._ensure_safety_models()
        if task.kind is DownloadKind.DEFAULT_LORAS:
            self._download_default_loras(manager)
            return True
        if task.kind is DownloadKind.ANNOTATORS:
            self._download_annotators(manager)
            return True
        return False

    def _manager_lock(self, manager_key: str) -> threading.Lock:
        """Return (creating once) the lock that serializes hordelib calls on *manager_key*'s manager."""
        with self._lock:
            lock = self._manager_locks.get(manager_key)
            if lock is None:
                lock = threading.Lock()
                self._manager_locks[manager_key] = lock
            return lock

    def _forget_attempts(self, task: DownloadTask) -> None:
        with self._lock:
            self._attempts.pop(task.dedup_key, None)

    def _begin_task(self, task: DownloadTask, runtime: _TaskRuntime) -> None:
        """Register the runtime and ensure the cross-process download slot is held before fetching."""
        with self._lock:
            self._active[task.dedup_key] = runtime
            self._running_count += 1
        self._acquire_bandwidth_slot()
        self._send_status(DownloadPhase.DOWNLOADING, force=True)

    def _end_task(self, task: DownloadTask) -> None:
        """Drop the runtime and release the cross-process slot once the last task finishes."""
        with self._lock:
            self._active.pop(task.dedup_key, None)
            self._running_count = max(0, self._running_count - 1)
        self._release_bandwidth_slot()

    def _acquire_bandwidth_slot(self) -> None:
        """Hold the size-1 cross-process download slot while any task runs (first task acquires for all).

        The first concurrent task performs the (blocking) acquire and signals ``_bandwidth_ready``; later
        tasks wait on that event so none fetches before the slot is genuinely held (the prior code let a
        sibling proceed on an intent flag alone). The acquire honours shutdown so a contended slot cannot
        wedge a thread past ``_end_process``.
        """
        with self._bandwidth_lock:
            first = self._bandwidth_count == 0
            self._bandwidth_count += 1
        if not first:
            # Wait in short slices so a waiter still notices shutdown even if the holder never signals.
            while not self._bandwidth_ready.wait(timeout=0.2):
                if self._end_process:
                    return
            return
        while not self._end_process:
            if self._download_bandwidth_semaphore.acquire(timeout=0.2):
                with self._bandwidth_lock:
                    self._bandwidth_held = True
                break
        self._bandwidth_ready.set()

    def _release_bandwidth_slot(self) -> None:
        """Release the cross-process slot (and reset the gate) once the last active task finishes."""
        with self._bandwidth_lock:
            self._bandwidth_count = max(0, self._bandwidth_count - 1)
            if self._bandwidth_count > 0:
                return
            self._bandwidth_ready.clear()
            release = self._bandwidth_held
            self._bandwidth_held = False
        if release:
            self._download_bandwidth_semaphore.release()

    def _make_callback(self, task: DownloadTask, runtime: _TaskRuntime) -> Callable[[int, int], None]:
        """Build this task's per-chunk callback: pace (pause/rate-share/abort) then emit its progress."""

        def callback(downloaded: int, total: int) -> None:
            progress = runtime.pacer.step(
                downloaded,
                total,
                is_paused=lambda: self._paused,
                rate_limit_kbps=self._rate_share,
                should_abort=lambda: self._end_process or runtime.cancelled,
                on_wait=lambda emitted: self._emit_task_progress(runtime, emitted),
            )
            self._emit_task_progress(runtime, progress)

        return callback

    def _rate_share(self) -> int | None:
        """The per-task slice of the global bandwidth cap, so N parallel downloads honour it in aggregate."""
        rate = self._rate_limit_kbps
        if not rate:
            return None
        return max(1, rate // max(1, self._running_count))

    def _emit_task_progress(self, runtime: _TaskRuntime, progress: ModelProgress) -> None:
        """Update one task's progress snapshot and emit a (throttled) status, PAUSED vs DOWNLOADING live."""
        with self._lock:
            runtime.status = CurrentDownloadStatus(
                model_name=runtime.status.model_name,
                feature=runtime.status.feature,
                target_dir=runtime.status.target_dir,
                host=runtime.status.host,
                downloaded_bytes=progress.downloaded_bytes,
                total_bytes=progress.total_bytes,
                speed_bps=progress.speed_bps,
                eta_seconds=progress.eta_seconds,
            )
        if self._paused:
            self._send_status(DownloadPhase.PAUSED, force=True)
        else:
            self._send_status(DownloadPhase.DOWNLOADING)

    def _record_failure(self, model: str, feature: str, reason: str) -> None:
        logger.warning(f"Download failed: {model!r} ({feature}): {reason}")
        with self._lock:
            self._failures = [f for f in self._failures if f.model_name != model]
            self._failures.append(DownloadFailure(model_name=model, feature=feature, reason=reason))

    def _clear_failure(self, model: str) -> None:
        """Drop any recorded failure for *model* (a later attempt succeeded, so it is no longer failed)."""
        with self._lock:
            kept = [f for f in self._failures if f.model_name != model]
            if len(kept) != len(self._failures):
                self._failures = kept

    # endregion

    # region per-kind download helpers

    def _safety_models_present_on_disk(self) -> bool:
        """Existence-only probe for the required safety models, mirroring ``model_download_plan``'s style.

        DeepDanbooru exposes a clean default path; CLIP is fetched by ``open_clip`` into
        ``CACHE_FOLDER_PATH`` (as an ``open_clip_*`` weights file) with no presence API, so it is probed by
        existence. A false negative only costs an idempotent re-ensure (no re-download when the file is
        already there); integrity remains the safety process's responsibility when it loads them.
        """
        try:
            from pathlib import Path

            from horde_safety import CACHE_FOLDER_PATH
            from horde_safety.deep_danbooru_model import default_deep_danbooru_model_path

            deep_danbooru_present = Path(default_deep_danbooru_model_path).exists()
            clip_present = any(Path(CACHE_FOLDER_PATH).rglob("open_clip*"))
            return deep_danbooru_present and clip_present
        except Exception as e:  # noqa: BLE001 - a probe failure must never crash the download process
            logger.warning(f"Download process: could not probe safety-model presence: {type(e).__name__} {e}")
            return False

    def _ensure_safety_models(self) -> bool:
        """Ensure the required safety models (DeepDanbooru + CLIP) are on disk; return whether they are.

        Routing this through the download process (instead of letting the safety process fetch them in its
        constructor) means the TUI/console shows a labelled ``safety models`` download with a phase instead
        of a frozen, hung-looking startup. Set ``_safety_ensured`` even on failure so the attempt is
        one-shot; the parent's grace fallback then starts the safety process, which surfaces the real error.
        """
        if self._safety_present:
            with self._lock:
                self._safety_ensured = True
            return True
        try:
            from horde_safety.deep_danbooru_model import download_deep_danbooru_model
            from horde_safety.interrogate import get_interrogator_no_blip

            download_deep_danbooru_model()
            # No download-only API for CLIP: this downloads it (when absent) and loads it transiently into
            # this process's RAM; the local interrogator is dropped on return, so the RAM is reclaimed.
            # These helpers use their own progress bars (not our chunk callback), so the task runs to
            # completion rather than being interruptible mid-file.
            get_interrogator_no_blip()
            self._safety_present = True
            logger.success("Download process: required safety models are present")
            return True
        except Exception as e:  # noqa: BLE001 - record and let the safety process surface the real failure
            self._record_failure("safety models", FEATURE_SAFETY, f"{type(e).__name__}: {e}")
            logger.error(f"Download process: failed to ensure safety models: {type(e).__name__} {e}")
            return False
        finally:
            with self._lock:
                self._safety_ensured = True

    def _download_default_loras(self, manager: object) -> None:
        """Fetch the curated default-LoRa set via the CivitAI ad-hoc engine (coarse progress only)."""
        lora = manager.lora  # type: ignore[attr-defined]
        if lora is None:
            return
        lora.reset_adhoc_cache()
        lora.download_default_models(nsfw=self._nsfw)
        lora.wait_for_downloads(600)
        lora.wait_for_adhoc_reset(120)
        if self._purge_loras:
            lora.delete_unused_models(30)

    def _download_annotators(self, manager: object) -> None:
        """Initialise ComfyUI and preload the ControlNet annotators (they need a ComfyUI init)."""
        import hordelib
        from hordelib.api import SharedModelManager

        if manager.controlnet is None:  # type: ignore[attr-defined]
            return
        extra_comfyui_args = [f"--directml={self._directml}"] if self._directml is not None else []
        hordelib.initialise(extra_comfyui_args=extra_comfyui_args)
        SharedModelManager.preload_annotators()

    # endregion

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Unused: the dedicated control thread drains the pipe (see ``_control_loop``)."""
        return

    @override
    def cleanup_for_exit(self) -> None:
        """No special cleanup is required for the download process."""
        return
