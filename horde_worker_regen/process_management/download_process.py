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
- The first on-disk scan (``available_models``) can SHA256-hash large files; it is reported as the
  ``SCANNING`` phase so it never looks hung.
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
from typing import override

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock

from loguru import logger

from horde_worker_regen.process_management._aliased_types import ProcessQueue
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

FEATURE_IMAGE_MODEL = "image model"
FEATURE_LORA = "LoRa (default set)"
FEATURE_CONTROLNET = "ControlNet"
FEATURE_CONTROLNET_ANNOTATORS = "ControlNet annotators"
FEATURE_MISCELLANEOUS = "miscellaneous (SDXL)"
FEATURE_SAFETY = "safety models"


def _post_processing_feature(name: str) -> str:
    return f"post-processing ({name})"


class _DownloadInterrupted(Exception):
    """Raised inside the progress callback to abort an in-flight download on shutdown."""


class HordeDownloadProcess(HordeProcess):
    """A background process that ensures requested image models (and aux models) are present on disk."""

    capabilities = WorkerCapability(0)

    def __init__(
        self,
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

        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._failures: list[DownloadFailure] = []
        self._present: list[str] = []
        self._current: CurrentDownloadStatus | None = None
        self._phase = DownloadPhase.INITIALIZING
        self._paused = paused
        self._rate_limit_kbps = rate_limit_kbps if (rate_limit_kbps or 0) > 0 else None
        self._error_message: str | None = None

        self._aux_requested = False
        self._aux_done = False
        self._reload_requested = False
        # Set after a completed download that changed on-disk references; emitted once on the next
        # status snapshot so the parent can broadcast a reload to the inference subprocesses.
        self._reference_changed_pending = False

        # Per-download context the progress callback reads (set before each download).
        self._cb_feature = FEATURE_IMAGE_MODEL
        self._cb_model = ""
        self._cb_target_dir = ""
        self._cb_last_bytes = 0
        self._cb_last_time = 0.0
        self._cb_speed_bps: float | None = None
        self._last_status_emit = 0.0

    # region status reporting

    def _build_status(self, phase: DownloadPhase) -> DownloadStatusSnapshot:
        with self._lock:
            pending = [DownloadItem(model_name=name, feature=FEATURE_IMAGE_MODEL) for name in self._pending]
            failures = list(self._failures)
            present = list(self._present)
            current = self._current
            paused = self._paused
            rate = self._rate_limit_kbps
        effective_phase = DownloadPhase.PAUSED if paused and phase == DownloadPhase.DOWNLOADING else phase
        return DownloadStatusSnapshot(
            phase=effective_phase,
            current=current,
            pending=pending,
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
            status=status,
            reference_changed=reference_changed,
        )
        self.process_message_queue.put(message)

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

        changed = False
        with self._lock:
            present = set(self._present)
            for model_name in message.model_names:
                if model_name in present or model_name in self._pending or model_name == self._cb_model:
                    continue
                self._pending.append(model_name)
                changed = True
            if message.download_aux:
                self._aux_requested = True
            if message.set_paused is not None and message.set_paused != self._paused:
                self._paused = message.set_paused
                changed = True
            if message.set_rate_limit_kbps is not None:
                self._rate_limit_kbps = message.set_rate_limit_kbps if message.set_rate_limit_kbps > 0 else None
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

        self._send_status(DownloadPhase.INITIALIZING, scan_complete=False, force=True)
        if not self._load_managers_with_retry():
            self._send_status(DownloadPhase.ERROR, scan_complete=False, force=True)
            self._idle_until_end()
            return

        self._send_status(DownloadPhase.SCANNING, scan_complete=False, force=True)
        self._refresh_present()
        self._send_status(DownloadPhase.IDLE, scan_complete=True, force=True)

        while not self._end_process:
            if not self._tick():
                time.sleep(0.1)

        self._send_status(DownloadPhase.IDLE, force=True)
        logger.info("Download process ended")
        sys.exit(0)

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
        with self._lock:
            self._present = sorted(compvis.available_models) if compvis is not None else []

    def _reload_model_database(self) -> None:
        """Reload model manager references from disk (offline) after a parent reference refresh."""
        from hordelib.api import SharedModelManager

        try:
            SharedModelManager.manager.reload_database()
            logger.info("Download process reloaded model database from disk")
        except Exception as e:  # noqa: BLE001 - a reload failure must not crash the download process
            logger.error(f"Download process failed to reload model database: {type(e).__name__}: {e}")

    def _tick(self) -> bool:
        """Do one unit of work; return True if it did something, False when idle."""
        with self._lock:
            paused = self._paused
            next_model = self._pending[0] if (self._pending and not paused) else None
            aux_ready = self._aux_requested and not self._aux_done and not paused
            reload_requested = self._reload_requested
            if reload_requested:
                self._reload_requested = False

        if reload_requested:
            self._reload_model_database()
            self._refresh_present()
            self._send_status(DownloadPhase.IDLE, force=True)
            return True

        if paused:
            self._send_status(DownloadPhase.PAUSED)
            return False
        if next_model is not None:
            self._download_image_model(next_model)
            with self._lock:
                self._reference_changed_pending = True
            return True
        if aux_ready:
            self._run_aux_downloads()
            self._aux_done = True
            self._refresh_present()
            with self._lock:
                self._reference_changed_pending = True
            self._send_status(DownloadPhase.IDLE, force=True)
            return True
        self._send_status(DownloadPhase.IDLE)
        return False

    # region downloads

    def _progress_callback(self, downloaded: int, total: int) -> None:
        """Per-chunk hook: update progress, enforce pause/rate-limit, emit throttled status."""
        if self._end_process:
            raise _DownloadInterrupted
        now = time.time()
        delta = max(0, downloaded - self._cb_last_bytes)
        elapsed = now - self._cb_last_time if self._cb_last_time else 0.0

        if self._rate_limit_kbps and elapsed >= 0 and delta > 0:
            allowed = delta / (self._rate_limit_kbps * 1024.0)
            if allowed > elapsed:
                time.sleep(allowed - elapsed)
                now = time.time()
                elapsed = now - self._cb_last_time if self._cb_last_time else 0.0

        if elapsed > 0 and delta > 0:
            instantaneous = delta / elapsed
            self._cb_speed_bps = (
                instantaneous if self._cb_speed_bps is None else (0.7 * self._cb_speed_bps + 0.3 * instantaneous)
            )
        self._cb_last_bytes = downloaded
        self._cb_last_time = now

        eta = None
        if self._cb_speed_bps and total > downloaded:
            eta = (total - downloaded) / self._cb_speed_bps
        with self._lock:
            self._current = CurrentDownloadStatus(
                model_name=self._cb_model,
                feature=self._cb_feature,
                target_dir=self._cb_target_dir,
                downloaded_bytes=downloaded,
                total_bytes=total,
                speed_bps=self._cb_speed_bps,
                eta_seconds=eta,
            )

        # Live pause: block here (holding the chunk loop) until resumed or shutdown.
        while self._paused and not self._end_process:
            self._send_status(DownloadPhase.PAUSED, force=True)
            time.sleep(0.2)
        if self._end_process:
            raise _DownloadInterrupted
        self._send_status(DownloadPhase.DOWNLOADING)

    def _begin_download_context(self, *, model: str, feature: str, target_dir: str) -> None:
        self._cb_model = model
        self._cb_feature = feature
        self._cb_target_dir = target_dir
        self._cb_last_bytes = 0
        self._cb_last_time = 0.0
        self._cb_speed_bps = None
        with self._lock:
            self._current = CurrentDownloadStatus(model_name=model, feature=feature, target_dir=target_dir)
        self._send_status(DownloadPhase.DOWNLOADING, force=True)

    def _end_download_context(self) -> None:
        with self._lock:
            self._current = None

    def _record_failure(self, model: str, feature: str, reason: str) -> None:
        with self._lock:
            self._failures = [f for f in self._failures if f.model_name != model]
            self._failures.append(DownloadFailure(model_name=model, feature=feature, reason=reason))

    def _download_image_model(self, model_name: str) -> None:
        from hordelib.api import SharedModelManager

        compvis = SharedModelManager.manager.compvis
        with self._lock:
            if model_name in self._pending:
                self._pending.remove(model_name)
        if compvis is None:
            self._record_failure(model_name, FEATURE_IMAGE_MODEL, "compvis manager unavailable")
            return

        self._begin_download_context(
            model=model_name,
            feature=FEATURE_IMAGE_MODEL,
            target_dir=str(compvis.model_folder_path),
        )
        started = time.time()
        try:
            succeeded = bool(compvis.download_model(model_name, callback=self._progress_callback))
            if succeeded and compvis.validate_model(model_name) is False:
                succeeded = bool(compvis.download_model(model_name, callback=self._progress_callback))
        except _DownloadInterrupted:
            return
        except OSError as e:
            reason = "out of disk space" if e.errno == 28 else f"{type(e).__name__}: {e}"
            self._record_failure(model_name, FEATURE_IMAGE_MODEL, reason)
            self._end_download_context()
            return
        except Exception as e:  # noqa: BLE001 - any download error is a recorded failure, not a crash
            self._record_failure(model_name, FEATURE_IMAGE_MODEL, f"{type(e).__name__}: {e}")
            self._end_download_context()
            return
        finally:
            self._refresh_present()

        self._end_download_context()
        if succeeded:
            logger.success(f"Download process: downloaded {model_name} in {time.time() - started:.1f}s")
        else:
            self._record_failure(model_name, FEATURE_IMAGE_MODEL, "download failed")
        self._send_status(self._phase, force=True)

    def _run_aux_downloads(self) -> None:
        """Best-effort fetch of the auxiliary/default models permitted by the worker config.

        Mirrors the categories in ``download_models.download_all_models``; any failure is logged but
        never fatal. Where the manager forwards a callback (``download_all_models``) we get live
        per-file progress; the curated LoRa/safety paths report a coarse feature label only.
        """
        try:
            from hordelib.api import SharedModelManager

            manager = SharedModelManager.manager

            self._begin_download_context(model="safety", feature=FEATURE_SAFETY, target_dir="")
            from horde_safety.deep_danbooru_model import download_deep_danbooru_model
            from horde_safety.interrogate import get_interrogator_no_blip

            download_deep_danbooru_model()
            get_interrogator_no_blip()

            if self._allow_lora and manager.lora is not None:
                self._begin_download_context(
                    model="default LoRas",
                    feature=FEATURE_LORA,
                    target_dir=str(manager.lora.model_folder_path),
                )
                manager.lora.reset_adhoc_cache()
                manager.lora.download_default_models(nsfw=self._nsfw)
                manager.lora.wait_for_downloads(600)
                manager.lora.wait_for_adhoc_reset(120)
                if self._purge_loras:
                    manager.lora.delete_unused_models(30)

            if self._allow_post_processing:
                for label, post_processor in (
                    ("GFPGAN", manager.gfpgan),
                    ("ESRGAN", manager.esrgan),
                    ("CodeFormer", manager.codeformer),
                ):
                    if post_processor is not None:
                        self._begin_download_context(
                            model=label,
                            feature=_post_processing_feature(label),
                            target_dir=str(post_processor.model_folder_path),
                        )
                        post_processor.download_all_models(callback=self._progress_callback)

            if self._allow_sdxl_controlnet and manager.miscellaneous is not None:
                self._begin_download_context(
                    model="miscellaneous",
                    feature=FEATURE_MISCELLANEOUS,
                    target_dir=str(manager.miscellaneous.model_folder_path),
                )
                manager.miscellaneous.download_all_models(callback=self._progress_callback)

            if self._allow_controlnet:
                self._download_controlnet_models()
        except _DownloadInterrupted:
            pass
        except Exception as e:  # noqa: BLE001 - aux is best-effort; the worker still serves image jobs
            logger.error(f"Download process: aux downloads failed: {type(e).__name__} {e}")
        finally:
            self._end_download_context()

    def _download_controlnet_models(self) -> None:
        """Fetch ControlNet models and annotators (the annotators need a ComfyUI init)."""
        import hordelib
        from hordelib.api import SharedModelManager

        controlnet = SharedModelManager.manager.controlnet
        if controlnet is None:
            return

        self._begin_download_context(
            model="ControlNet models",
            feature=FEATURE_CONTROLNET,
            target_dir=str(controlnet.model_folder_path),
        )
        for cn_model in controlnet.model_reference:
            if (
                cn_model not in controlnet.available_models
                and "sdxl" in cn_model.lower()
                and not self._allow_sdxl_controlnet
            ):
                continue
            controlnet.download_model(cn_model, callback=self._progress_callback)

        self._begin_download_context(model="annotators", feature=FEATURE_CONTROLNET_ANNOTATORS, target_dir="")
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
