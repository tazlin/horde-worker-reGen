from __future__ import annotations

import queue
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from multiprocessing import Queue

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from loguru import logger

from horde_worker_regen.process_management.action_ledger import ActionLedger, LedgerEventType
from horde_worker_regen.process_management.download_process import DOWNLOAD_PROCESS_ID
from horde_worker_regen.process_management.failure_classification import is_resource_failure
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.job_models import HordeJobInfo
from horde_worker_regen.process_management.job_tracker import InferenceFailureResolution, JobTracker
from horde_worker_regen.process_management.messages import (
    AUX_DOWNLOAD_FAILED_INFO,
    HordeAlchemyResultMessage,
    HordeAuxModelStateChangeMessage,
    HordeDownloadAvailabilityMessage,
    HordeDownloadMetricsMessage,
    HordeInferenceResultMessage,
    HordeJobMetricsMessage,
    HordeModelStateChangeMessage,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    HordeSafetyResultMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.model_metadata import ModelMetadata
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.worker_state import WorkerState
from horde_worker_regen.telemetry_spans import (
    inference_duration_histogram,
    jobs_completed_counter,
    jobs_faulted_counter,
    queue_depth_counter,
)

_excludes_for_job_dump = {"source_image", "source_mask", "extra_source_images", "r2_upload"}

_INFERENCE_ACTIVE_STATES = frozenset(
    {
        HordeProcessState.INFERENCE_STARTING,
        HordeProcessState.INFERENCE_POST_PROCESSING,
        HordeProcessState.INFERENCE_COMPLETE,
        HordeProcessState.INFERENCE_FAILED,
    },
)
"""Slot states from which a return to idle means a job actually ran (so a missing result was lost).

Used by the lost-result reap to exclude the dispatch window: a slot transitioning to idle from a
teardown/preload path is carrying a job the scheduler only just stamped onto it, which it has not run
yet, so there is no result to have lost."""

_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS = 20.0
"""How long a queue deadlock must persist continuously before it counts as a structural wedge.

The recovery supervisor's give-up budget is calibrated for *definitive* signals (a crash-looped pool),
which a slow model load or a normal slot replacement never trips. The instantaneous queue-deadlock flag
does not meet that bar on its own: it also flips during the brief all-idle window between a job finishing
and the scheduler preloading the next model, and the detector deliberately holds it set across that
preload (its anti-flap guard keeps it set while a process is starting). Requiring the deadlock to outlast
any normal model-load / churn window before it drives save-our-ship restores the supervisor's
definitive-signal assumption, so a head whose model is merely loading is not faulted as unrecoverable."""


@dataclass(frozen=True)
class DeadlockSnapshot:
    """Represents the currently observed scheduler deadlock state."""

    in_deadlock: bool
    in_queue_deadlock: bool
    deadlock_started_at: float
    queue_deadlock_started_at: float
    queue_deadlock_model: str | None
    queue_deadlock_process_id: int | None

    def has_active_deadlock(self) -> bool:
        """Return whether any deadlock detector is currently active (diagnostics-grade)."""
        return self.in_deadlock or self.in_queue_deadlock

    def indicates_structural_wedge(self, now: float | None = None) -> bool:
        """Return whether the deadlock state is a genuine, recoverable inference-pool wedge.

        Only a *sustained* queue deadlock qualifies: pending inference work exists, every process is
        idle, and the head's model is not becoming resident, held continuously past any normal
        model-load / churn window (:data:`_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS`). A just-detected queue
        deadlock is the normal all-idle gap between jobs (or a model mid-preload) and must not drive the
        save-our-ship reset, whose give-up budget assumes only definitive signals reach it. The general
        ``in_deadlock`` flag is deliberately excluded too: it also fires for a job legitimately draining
        through the post-inference safety/submit tail during a queue lull.
        """
        if not self.in_queue_deadlock:
            return False
        reference = time.time() if now is None else now
        return (reference - self.queue_deadlock_started_at) >= _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS


class MessageDispatcher:
    """Drains the IPC message queue and updates process/model state."""

    _process_map: ProcessMap
    _horde_model_map: HordeModelMap
    _job_tracker: JobTracker
    _process_message_queue: Queue  # type: ignore[type-arg]

    _runtime_config: RuntimeConfig
    _model_metadata: ModelMetadata
    _action_ledger: ActionLedger
    _on_unload_vram: Callable[[HordeProcessInfo], Awaitable[None]]
    _on_alchemy_result: Callable[[HordeAlchemyResultMessage], None] | None = None
    _on_job_metrics: Callable[[HordeJobMetricsMessage], None] | None = None
    _on_download_metrics: Callable[[HordeDownloadMetricsMessage], None] | None = None
    _on_download_availability: Callable[[HordeDownloadAvailabilityMessage], None] | None = None
    _on_model_load_failure: Callable[[int, str], None] | None = None
    """Invoked as ``(process_id, horde_model_name)`` when a child reports it failed to load a model."""

    _last_deadlock_detected_time: float = 0.0
    _in_deadlock: bool = False
    _in_queue_deadlock: bool = False
    _last_queue_deadlock_detected_time: float = 0.0
    _queue_deadlock_model: str | None = None
    _queue_deadlock_process_id: int | None = None
    _last_deadlock_detail_log_time: float = 0.0

    _DEADLOCK_DETAIL_LOG_INTERVAL_SECONDS = 30.0
    """How often the verbose deadlock dump (process/model maps, per-stage counts) may be emitted while a
    wedge persists. The recurring "still detected" branches run every control-loop tick, so without this
    throttle a sustained wedge floods the log with thousands of identical dumps."""

    def __init__(
        self,
        *,
        process_map: ProcessMap,
        horde_model_map: HordeModelMap,
        job_tracker: JobTracker,
        process_message_queue: Queue,  # type: ignore[type-arg]
        runtime_config: RuntimeConfig,
        model_metadata: ModelMetadata,
        action_ledger: ActionLedger,
        on_unload_vram: Callable[[HordeProcessInfo], Awaitable[None]],
        state: WorkerState,
    ) -> None:
        """Initialize the dispatcher with references to shared state and the message queue.

        Args:
            process_map (ProcessMap): The shared process map to update based on messages from child processes.
            horde_model_map (HordeModelMap): The shared model map to update based on messages from child processes.
            job_tracker (JobTracker): The job tracker to update based on messages from child processes.
            process_message_queue (Queue): The queue from which to receive messages from child processes.
            runtime_config (RuntimeConfig): Holds the current bridge configuration snapshot.
            model_metadata (ModelMetadata): Provides lookups against the stable-diffusion model reference.
            action_ledger (ActionLedger): The shared lifecycle audit ledger; inference retries and terminal
                faults are recorded here so a job's failure history is self-explaining in a post-mortem.
            on_unload_vram (Callable[[HordeProcessInfo], None]): A callback to invoke when a process reports that it
                has unloaded a model from VRAM. This is used to trigger the unloading of the model from any other
                processes that have it loaded, if the current bridge configuration requires aggressive VRAM management.
            state (WorkerState): The shared worker state, which tracks various flags and timestamps related to the
                worker's operation, such as whether it's currently in a deadlock, when the last job was popped, etc.
        """
        self._process_map = process_map
        self._horde_model_map = horde_model_map
        self._job_tracker = job_tracker
        self._process_message_queue = process_message_queue
        self._runtime_config = runtime_config
        self._model_metadata = model_metadata
        self._action_ledger = action_ledger
        self._on_unload_vram = on_unload_vram
        self._state = state

    def set_alchemy_result_handler(self, handler: Callable[[HordeAlchemyResultMessage], None]) -> None:
        """Register the callback invoked when a child process reports an alchemy form result."""
        self._on_alchemy_result = handler

    def set_metrics_handlers(
        self,
        *,
        on_job_metrics: Callable[[HordeJobMetricsMessage], None],
        on_download_metrics: Callable[[HordeDownloadMetricsMessage], None],
    ) -> None:
        """Register the callbacks invoked when a child reports job or download metrics."""
        self._on_job_metrics = on_job_metrics
        self._on_download_metrics = on_download_metrics

    def set_download_availability_handler(
        self,
        handler: Callable[[HordeDownloadAvailabilityMessage], None],
    ) -> None:
        """Register the callback invoked when the download process reports on-disk availability."""
        self._on_download_availability = handler

    def set_model_load_failure_handler(self, handler: Callable[[int, str], None]) -> None:
        """Register the callback invoked when a child reports it failed to load a model.

        Called as ``(process_id, horde_model_name)``. Lets the manager track per-model load failures and
        quarantine a deterministically-unloadable model instead of churning the process pool.
        """
        self._on_model_load_failure = handler

    async def receive_and_handle_process_messages(self) -> None:
        """Receive and handle any messages from the child processes."""
        while not self._process_message_queue.empty():
            try:
                message: HordeProcessMessage = self._process_message_queue.get(block=False)
            except queue.Empty:
                logger.debug("Queue was empty, breaking")
                break

            # The download process lives outside the process map, so its messages must be handled
            # (or ignored) before any of the process-map lookups below, which would otherwise raise.
            if message.process_id == DOWNLOAD_PROCESS_ID:
                if (
                    isinstance(message, HordeDownloadAvailabilityMessage)
                    and self._on_download_availability is not None
                ):
                    self._on_download_availability(message)
                continue

            if not isinstance(message, HordeProcessMessage):
                raise ValueError(f"Received a message that is not a HordeProcessMessage: {message}")

            if self._should_ignore_retired_launch_message(message):
                continue

            if message.process_id not in self._process_map:
                # A late IPC message from a process the map no longer knows. This is expected after an
                # intentional scale-down (whole-card residency teardown, RAM-reclaim cycling): the slot is
                # popped from the map, yet its already-queued terminal messages (PROCESS_ENDING, a final
                # memory report) still arrive. The retired-launch tombstone above normally absorbs these,
                # but it is keyed on the exact launch id and bounded by a TTL, so a message from an older
                # launch, or one arriving after the tombstone is pruned, can slip through. Dropping it is
                # always safe (the process is gone); raising here would crash the control loop and take the
                # whole worker down over a stale status update, which is exactly the fragility that stopping
                # processes mid-session must not introduce.
                logger.warning(
                    f"Ignoring message from process {message.process_id} that is no longer in the process map "
                    f"(launch {message.process_launch_identifier}); it was most likely intentionally stopped: "
                    f"{type(message).__name__}",
                )
                continue

            known_launch_identifier = self._process_map[message.process_id].process_launch_identifier

            if message.process_launch_identifier != known_launch_identifier:
                logger.error(
                    f"Received a message from process {message.process_id} with launch identifier "
                    f"{message.process_launch_identifier}, but expected {known_launch_identifier}",
                )
                logger.error("This is probably due to a process being replaced. Ignoring.")
                logger.error(f"Message: {message}")
                continue

            if isinstance(message, HordeProcessHeartbeatMessage):
                self._handle_heartbeat(message)
            else:
                logger.debug(
                    f"Received {type(message).__name__} from process {message.process_id}: {message.info}",
                )

            if isinstance(message, HordeProcessMemoryMessage):
                self._handle_memory_report(message)
                continue

            if isinstance(message, HordeJobMetricsMessage):
                self._process_map.on_job_metrics(message.process_id, message.phase_metrics)
                if self._on_job_metrics is not None:
                    self._on_job_metrics(message)
                continue

            if isinstance(message, HordeDownloadMetricsMessage):
                self._process_map.on_download_metrics(message.process_id, message.events)
                if self._on_download_metrics is not None:
                    self._on_download_metrics(message)
                continue

            if isinstance(message, HordeProcessStateChangeMessage):
                self._handle_process_state_change(message)

            if isinstance(message, HordeAuxModelStateChangeMessage):
                await self._handle_aux_model_state_change(message)

            if isinstance(message, HordeModelStateChangeMessage):
                self._handle_model_state_change(message)

            if isinstance(message, HordeInferenceResultMessage):
                self._record_completed_job(message.process_id)
                await self._handle_inference_result(message)
            elif isinstance(message, HordeSafetyResultMessage):
                self._record_completed_job(message.process_id)
                await self._handle_safety_result(message)
            elif isinstance(message, HordeAlchemyResultMessage):
                self._record_completed_job(message.process_id)
                if self._on_alchemy_result is not None:
                    self._on_alchemy_result(message)
                else:
                    logger.error(f"Received alchemy result with no handler registered: {message.form_id}")

    def _should_ignore_retired_launch_message(self, message: HordeProcessMessage) -> bool:
        """Return true when a message belongs to an intentionally retired launch."""
        retired_launch = self._process_map.get_retired_launch(
            message.process_id,
            message.process_launch_identifier,
        )
        if retired_launch is None:
            return False

        if isinstance(
            message,
            (HordeInferenceResultMessage, HordeSafetyResultMessage, HordeAlchemyResultMessage),
        ):
            logger.warning(
                f"Ignoring result message from retired {retired_launch.process_type.name.lower()} process "
                f"{message.process_id} launch {message.process_launch_identifier} "
                f"({retired_launch.reason}): {type(message).__name__}",
            )
            return True

        if self._is_late_retired_liveness_message(message):
            logger.debug(
                f"Ignoring late {type(message).__name__} from retired "
                f"{retired_launch.process_type.name.lower()} process {message.process_id} "
                f"launch {message.process_launch_identifier} ({retired_launch.reason})",
            )
            return True

        logger.warning(
            f"Ignoring unexpected {type(message).__name__} from retired "
            f"{retired_launch.process_type.name.lower()} process {message.process_id} "
            f"launch {message.process_launch_identifier} ({retired_launch.reason})",
        )
        return True

    def _is_late_retired_liveness_message(self, message: HordeProcessMessage) -> bool:
        """Return true for stale terminal/liveness messages expected after intentional retirement."""
        if isinstance(message, (HordeProcessHeartbeatMessage, HordeProcessMemoryMessage)):
            return True
        return isinstance(message, HordeProcessStateChangeMessage) and message.process_state in (
            HordeProcessState.PROCESS_ENDING,
            HordeProcessState.PROCESS_ENDED,
        )

    def _record_completed_job(self, process_id: int) -> None:
        """Bump the producing process's completed-work counter (inference, safety check, or alchemy form).

        Surfaced per-process in the live view as running feedback; the safety process is the main
        beneficiary, since its checks are otherwise too fast for any state change to be visible.
        """
        process_info = self._process_map.get(process_id)
        if process_info is not None:
            process_info.num_jobs_completed += 1

    def _handle_heartbeat(self, message: HordeProcessHeartbeatMessage) -> None:
        """Handle a heartbeat message from a child process."""
        self._process_map.on_heartbeat(
            message.process_id,
            heartbeat_type=message.heartbeat_type,
            percent_complete=message.percent_complete,
            current_step=message.current_step,
            total_steps=message.total_steps,
            iterations_per_second=message.iterations_per_second,
        )

        in_progress_job_info = self._process_map[message.process_id].last_job_referenced

        if message.process_warning is not None and (
            in_progress_job_info is not None and in_progress_job_info.payload.n_iter < 4
        ):
            logger.warning(f"Process {message.process_id} warning: {message.process_warning}")

            model_name = self._process_map[message.process_id].loaded_horde_model_name
            model_baseline = self._model_metadata.get_baseline(model_name) if model_name is not None else None

            if model_baseline is not None:
                logger.warning(f"Model baseline triggering warning: {model_baseline}")

            if in_progress_job_info.payload.n_iter != 1:
                logger.warning(f"Batched job triggering warning: {in_progress_job_info.payload.n_iter} images")
                logger.warning("If you think this is in error, please contact the devs on github or discord.")

    def _handle_memory_report(self, message: HordeProcessMemoryMessage) -> None:
        """Handle a memory usage report from a child process."""
        self._process_map.on_memory_report(
            process_id=message.process_id,
            ram_usage_bytes=message.ram_usage_bytes,
            vram_usage_mb=message.vram_usage_mb,
            total_vram_mb=message.vram_total_mb,
        )

    def _handle_process_state_change(self, message: HordeProcessStateChangeMessage) -> None:
        """Handle a process state change message."""
        if self._process_map[message.process_id].last_process_state == message.process_state:
            return

        # Captured before the slot's state is advanced below; the lost-result reap needs the state the
        # slot is transitioning *from* to tell a post-inference idle from the dispatch window.
        previous_state = self._process_map[message.process_id].last_process_state

        self._process_map.on_process_state_change(
            process_id=message.process_id,
            new_state=message.process_state,
        )

        if message.process_state == HordeProcessState.PROCESS_ENDING:
            logger.info(f"Process {message.process_id} is ending")
            self._process_map.on_process_ending(process_id=message.process_id)

        if message.process_state == HordeProcessState.PROCESS_ENDED:
            logger.info(f"Process {message.process_id} has ended with message: {message.info}")
        else:
            logger.debug(f"Process {message.process_id} changed state to {message.process_state}")

        if message.process_state == HordeProcessState.INFERENCE_STARTING:
            loaded_model_name = self._process_map[message.process_id].loaded_horde_model_name
            if loaded_model_name is None:
                raise ValueError(
                    f"Process {message.process_id} has no model loaded, but is starting inference",
                )
            batch_amount = self._process_map[message.process_id].batch_amount
            if batch_amount is None:
                raise ValueError(
                    f"Process {message.process_id} has batch_amount, but is starting inference",
                )
            self._horde_model_map.update_entry(
                horde_model_name=loaded_model_name,
                load_state=ModelLoadState.IN_USE,
                process_id=message.process_id,
            )

        if (
            message.process_state == HordeProcessState.UNLOADED_MODEL_FROM_RAM
            and self._process_map[message.process_id].last_process_state != HordeProcessState.UNLOADED_MODEL_FROM_RAM
        ):
            logger.opt(ansi=True).info(
                f"<fg #7b7d7d>Process {message.process_id} cleared RAM: {message.info}</>",
            )
            self._process_map.on_model_ram_clear(process_id=message.process_id)

        if message.process_state == HordeProcessState.WAITING_FOR_JOB:
            self._reap_lost_inference_result(message.process_id, previous_state=previous_state)

    def _reap_lost_inference_result(self, process_id: int, *, previous_state: HordeProcessState) -> None:
        """Release a job left in progress after its slot returned to idle without a result.

        Inference results and process-state changes share one ordered message stream, and a completing
        job's result is always enqueued before the slot's transition back to ``WAITING_FOR_JOB``. So once
        that transition is processed, a job the slot still references that is *still* marked in progress
        can only mean its result never arrived (e.g. it was dropped by the launch-identifier guard while
        the slot was being replaced). No result will ever move that job on, so it would otherwise sit in
        progress, count against the concurrent-job cap, and wedge dispatch. Releasing it here (retryable,
        so it requeues) recovers the moment the loss is detectable. The periodic orphaned-job watchdog
        remains the backstop for losses where even this transition never arrives (e.g. the slot is
        replaced outright before reporting idle).

        That reasoning only holds for a slot returning to idle *after* running the job. ``last_job_referenced``
        and the in-progress mark are stamped by the scheduler the instant it dispatches a job, before the
        child acknowledges it, so a slot can carry a freshly dispatched job while it is still draining
        state messages from *before* the dispatch: the ``WAITING_FOR_JOB`` it reports after unloading the
        previous model to free VRAM, for example. Reading that stale idle report against the optimistically
        stamped job would fault a job that never ran (a window that widens on slower disks/model swaps). The
        slot's prior state is the discriminator: only a return to idle from an inference-active state can
        mean a result was produced and then lost. A return from a teardown/preload path is the dispatch
        window, so the job is left for the slot to take up.
        """
        if previous_state not in _INFERENCE_ACTIVE_STATES:
            return
        process_info = self._process_map.get(process_id)
        if process_info is None:
            return
        job = process_info.last_job_referenced
        if job is None or job not in self._job_tracker.jobs_in_progress:
            return

        job_id = str(job.id_) if job.id_ is not None else None
        logger.error(
            f"Process {process_id} returned to idle while job {job_id} was still in progress; its inference "
            "result was lost. Releasing the job so it can be retried.",
        )
        self._action_ledger.record(
            LedgerEventType.INFERENCE_FAULTED,
            process_id=process_id,
            os_pid=process_info.os_pid,
            launch_identifier=process_info.process_launch_identifier,
            job_id=job_id,
            reason="inference result lost (slot returned to idle with job still in progress)",
        )
        # Deliberately not passed as a process crash: the slot is alive and idle, only the result was
        # lost, so this takes the ordinary bounded retry (requeue while attempts remain, else fault and
        # report so the horde reissues) rather than the crash path.
        self._job_tracker.handle_job_fault_now(
            faulted_job=job,
            process_timeout=self._runtime_config.bridge_data.process_timeout,
            retryable=True,
        )

    async def _handle_aux_model_state_change(self, message: HordeAuxModelStateChangeMessage) -> None:
        """Handle an auxiliary model state change message (e.g., LoRa downloads)."""
        job_info = message.sdk_api_job_info
        job_context = ""
        if job_info is not None:
            lora_count = len(job_info.payload.loras or [])
            ti_count = len(job_info.payload.tis or [])
            job_context = (
                f" for job {str(job_info.id_)[:8]} (model={job_info.model}, loras={lora_count}, tis={ti_count})"
            )

        if message.process_state == HordeProcessState.DOWNLOADING_AUX_MODEL:
            logger.opt(ansi=True).info(
                f"<fg #7b7d7d>Process {message.process_id} is downloading extra models{job_context}</>",
            )
            self._process_map.on_last_job_reference_change(
                process_id=message.process_id,
                last_job_referenced=message.sdk_api_job_info,
            )

        if message.process_state == HordeProcessState.DOWNLOAD_AUX_COMPLETE:
            logger.opt(ansi=True).info(
                "<fg #7b7d7d>"
                f"Process {message.process_id} finished downloading extra models{job_context} "
                f"in {message.time_elapsed}"
                "</>",
            )
            if message.sdk_api_job_info not in self._job_tracker.jobs_lookup:
                if message.sdk_api_job_info is not None:
                    logger.warning(
                        f"Job {message.sdk_api_job_info.id_} not found in jobs_lookup. (Process {message.process_id})",
                    )
                else:
                    logger.warning(
                        f"Job not found in jobs_lookup. (Process {message.process_id})",
                    )
                logger.debug(f"Jobs lookup: {self._job_tracker.jobs_lookup}")
            else:
                await self._job_tracker.set_job_time_to_download_aux_models(
                    message.sdk_api_job_info,
                    message.time_elapsed,
                )

    def _handle_model_state_change(self, message: HordeModelStateChangeMessage) -> None:
        """Handle a model state change message."""
        if message.horde_model_state == ModelLoadState.FAILED:
            # The model could not be loaded. Do not record it as resident anywhere: clear any LOADING entry
            # this process held for it (left by the PRELOADING_MODEL message moments earlier) so the model is
            # not pinned to a slot that never actually loaded it, then hand the failure to the manager so it
            # can track repeated failures and quarantine the model.
            logger.error(
                f"Process {message.process_id} failed to load model {message.horde_model_name}",
            )
            self._horde_model_map.expire_entry(message.horde_model_name)
            if self._on_model_load_failure is not None:
                self._on_model_load_failure(message.process_id, message.horde_model_name)
            return

        self._horde_model_map.update_entry(
            horde_model_name=message.horde_model_name,
            load_state=message.horde_model_state,
            process_id=message.process_id,
        )

        model_baseline = self._model_metadata.get_baseline(message.horde_model_name)

        if message.horde_model_state != ModelLoadState.ON_DISK:
            self._process_map.on_model_load_state_change(
                process_id=message.process_id,
                horde_model_name=message.horde_model_name,
                horde_model_baseline=model_baseline,
            )

            if message.horde_model_state == ModelLoadState.LOADING:
                logger.debug(f"Process {message.process_id} is loading model {message.horde_model_name}")

            if (
                message.horde_model_state == ModelLoadState.LOADED_IN_VRAM
                or message.horde_model_state == ModelLoadState.LOADED_IN_RAM
            ):
                if message.horde_model_state == ModelLoadState.LOADED_IN_VRAM:
                    loaded_message = (
                        f"Process {message.process_id} just finished inference, and has "
                        f"{message.horde_model_name} in VRAM."
                    )
                    logger.debug(loaded_message)
                elif message.horde_model_state == ModelLoadState.LOADED_IN_RAM:
                    loaded_message = (
                        f"Process {message.process_id} moved model {message.horde_model_name} to system RAM. "
                    )

                    if message.time_elapsed is not None:
                        loaded_message += f"Loading took {message.time_elapsed:.2f} seconds."

                    logger.opt(ansi=True).info(f"<fg #7b7d7d>{loaded_message}</>")

        else:
            # FIXME this message is wrong for download processes
            logger.opt(ansi=True).info(
                f"<fg #7b7d7d>Process {message.process_id} unloaded model {message.horde_model_name}</>",
            )

    async def _handle_inference_result(self, message: HordeInferenceResultMessage) -> None:
        """Handle an inference job result message."""
        # A result (success, fault, or even one for a job we no longer track) means the slot is no longer
        # sampling, so retire its in-flight timestamps first: before the graded-slowdown monitor can read
        # them against a finished job, and before any early-return below. A dropped result (job gone from
        # jobs_lookup) that left ``current_inference_started_at`` set could otherwise leave the slot looking
        # like the dispatch-in-flight owner of a *different* freshly dispatched job.
        if message.process_id in self._process_map:
            self._process_map[message.process_id].current_inference_started_at = None
            self._process_map[message.process_id].current_first_step_at = None
            self._process_map[message.process_id].current_job_expected_sampling_seconds = None

        job_info = await self._job_tracker.get_job_info(message.sdk_api_job_info)
        if job_info is None:
            logger.error(
                f"Job {message.sdk_api_job_info.id_} not found in jobs_lookup. (Process {message.process_id})",
            )
            if message.sdk_api_job_info in self._job_tracker.jobs_in_progress:
                logger.error(
                    f"Job {message.sdk_api_job_info.id_} found in jobs_in_progress. (Process {message.process_id})",
                )
                await self._job_tracker.release_in_progress(message.sdk_api_job_info)
            if message.sdk_api_job_info in self._job_tracker.jobs_pending_inference:
                logger.error(
                    f"Job {message.sdk_api_job_info.id_} found in job_deque. (Process {message.process_id})",
                )
                await self._job_tracker.drop_pending_inference(message.sdk_api_job_info)
            return

        # Faults are resolved before the success bookkeeping: a retryable failure is requeued (and must
        # not be counted as completed), and the retry brain owns the stage move for both outcomes.
        if message.state == GENERATION_STATE.faulted:
            await self._handle_faulted_inference_result(message, job_info)
            return

        if not await self._job_tracker.release_in_progress(message.sdk_api_job_info):
            logger.error(
                f"Job {message.sdk_api_job_info.id_} not found in jobs_in_progress. "
                "Did it fault? "
                f"(Process {message.process_id})",
            )

        if message.sdk_api_job_info.id_ is not None:
            await self._job_tracker.drop_pending_inference_by_id(message.sdk_api_job_info.id_)

        await self._job_tracker.increment_jobs_completed()
        queue_depth_counter.add(-1)
        bridge_data = self._runtime_config.bridge_data
        if bridge_data.unload_models_from_vram_often:
            await self._on_unload_vram(self._process_map[message.process_id])

        if message.time_elapsed is not None:
            inference_duration_histogram.record(message.time_elapsed)
            inference_finished_string = (
                "\0<fg #da9dff>"
                f"Inference finished for job {str(message.sdk_api_job_info.id_)[:8]} "
                f"<u>({message.sdk_api_job_info.model})</u> on process {message.process_id}. "
                f"It took {round(message.time_elapsed, 2)} seconds, finishing at {message.info} "
                f"and reported {message.faults_count} faults."
                "</>"
            )

            logger.opt(ansi=True).info(inference_finished_string)

        else:
            logger.info(f"Inference finished for job {message.sdk_api_job_info.id_}")
            logger.debug(f"Job didn't include time_elapsed: {message.sdk_api_job_info}")

        job_info.state = message.state
        job_info.time_to_generate = message.time_elapsed
        job_info.job_image_results = message.job_image_results

        jobs_completed_counter.add(1)
        await self._job_tracker.queue_for_safety(job_info)

    async def _handle_faulted_inference_result(
        self,
        message: HordeInferenceResultMessage,
        job_info: HordeJobInfo,
    ) -> None:
        """Resolve a faulted inference result: requeue it for another attempt, or fault it terminally.

        Routes through the job tracker's bounded/degraded retry policy. A resource (out-of-memory) failure,
        recognized from the result's diagnostic ``info``, earns one degraded, isolated retry. On exhaustion
        the tracker has already moved the job to ``PENDING_SUBMIT`` and counted it; here we only emit the
        telemetry, audit, and VRAM cleanup the success path would otherwise have done.
        """
        job_id = str(message.sdk_api_job_info.id_) if message.sdk_api_job_info.id_ is not None else None

        # A child that aborted its own stalled aux download (deadline) reports the fault here instead of
        # the parent's watchdog tearing the process down. Mirror the teardown path's backoff handling: it
        # is not a resource/OOM failure, it arms the LoRA-download backoff, and -- once an incident is
        # active -- it is dropped rather than requeued straight back into the same failing download.
        # Retryability is read before this strike is recorded so a lone transient stall keeps its retry.
        is_aux_download_fault = message.info == AUX_DOWNLOAD_FAILED_INFO
        if is_aux_download_fault:
            resource_failure = False
            aux_retryable = not self._state.lora_download_backoff.is_escalation_active(time.time())
            window = self._state.lora_download_backoff.register_timeout(time.time())
            logger.warning(
                f"Job {job_id} aux (LoRA) download was aborted by process {message.process_id} (strike "
                f"{self._state.lora_download_backoff.strikes}); withholding LoRA job pops for {window:.0f}s.",
            )
        else:
            resource_failure = is_resource_failure(message.info)
            aux_retryable = True

        resolution = await self._job_tracker.handle_job_fault(
            message.sdk_api_job_info,
            process_timeout=message.time_elapsed if message.time_elapsed is not None else 0.0,
            is_resource_failure=resource_failure,
            retryable=aux_retryable,
        )

        if resolution is not InferenceFailureResolution.FAULTED:
            degraded = resolution is InferenceFailureResolution.RETRY_DEGRADED
            self._action_ledger.record(
                LedgerEventType.INFERENCE_RETRIED,
                process_id=message.process_id,
                job_id=job_id,
                reason=message.info or "inference failed",
                detail={"degraded": degraded, "resource_failure": resource_failure},
            )
            logger.warning(
                f"Job {job_id} faulted on process {message.process_id} ({message.info}); requeued for "
                f"{'a degraded, isolated' if degraded else 'another'} attempt.",
            )
            return

        # Terminal fault: the tracker has moved the job to PENDING_SUBMIT and counted it as terminal.
        jobs_faulted_counter.add(1)
        queue_depth_counter.add(-1)
        bridge_data = self._runtime_config.bridge_data
        if bridge_data.unload_models_from_vram_often and message.process_id in self._process_map:
            await self._on_unload_vram(self._process_map[message.process_id])

        self._action_ledger.record(
            LedgerEventType.INFERENCE_FAULTED,
            process_id=message.process_id,
            job_id=job_id,
            reason=message.info or "inference failed",
            detail={"resource_failure": resource_failure},
        )
        logger.error(
            f"Job {message.sdk_api_job_info.id_} faulted on process {message.process_id}: {message.info}",
        )
        logger.debug(
            f"Job data: {message.sdk_api_job_info.model_dump(exclude=_excludes_for_job_dump)}",  # type: ignore
        )

    async def _handle_safety_result(self, message: HordeSafetyResultMessage) -> None:
        """Handle a safety check result message."""
        completed_job_info = await self._job_tracker.take_being_safety_checked(message.job_id)

        if completed_job_info is None or completed_job_info.job_image_results is None:
            logger.error(
                f"Expected to find a completed job with ID {message.job_id} but none was found. "
                "This should only happen when certain process crashes occur.",
            )
            return

        num_images_censored = 0
        num_images_csam = 0

        any_safety_failed = False

        job_fault_entries = await self._job_tracker.get_faults_for_job(message.job_id)

        for i in range(len(completed_job_info.job_image_results)):
            if completed_job_info.sdk_api_job_info.id_ is None:
                continue
            completed_job_info.job_image_results[i].generation_faults += job_fault_entries
            replacement_image = message.safety_evaluations[i].replacement_image_base64

            if message.safety_evaluations[i].failed:
                logger.error(
                    f"Job {message.job_id} image #{i} faulted during safety checks. "
                    "Check the safety process logs for more information.",
                )
                any_safety_failed = True
                continue

            if replacement_image is not None:
                completed_job_info.job_image_results[i].image_base64 = replacement_image
                num_images_censored += 1
                if message.safety_evaluations[i].is_csam:
                    num_images_csam += 1
        if completed_job_info.sdk_api_job_info.id_ is None:
            logger.error(
                f"Job {message.job_id} has no id; cannot clear its fault entries. This is unexpected.",
            )
        elif job_fault_entries:
            await self._job_tracker.clear_faults_for_job(completed_job_info.sdk_api_job_info.id_)

        logger.debug(
            f"Job {message.job_id} had {num_images_censored} images censored and took "
            f"{message.time_elapsed:.2f} seconds to check safety",
        )

        if any_safety_failed:
            completed_job_info.state = GENERATION_STATE.faulted
        completed_job_info.censored = False
        for i in range(len(completed_job_info.job_image_results)):
            if message.safety_evaluations[i].is_csam:
                new_meta_entry = GenMetadataEntry(
                    type=METADATA_TYPE.censorship,
                    value=METADATA_VALUE.csam,
                )
                completed_job_info.job_image_results[i].generation_faults.append(new_meta_entry)
                completed_job_info.state = GENERATION_STATE.csam
                completed_job_info.censored = True
            elif message.safety_evaluations[i].is_nsfw:
                if message.safety_evaluations[i].replacement_image_base64 is None:
                    new_meta_entry = GenMetadataEntry(
                        type=METADATA_TYPE.information,
                        value=METADATA_VALUE.nsfw,
                    )
                    completed_job_info.job_image_results[i].generation_faults.append(new_meta_entry)
                else:
                    new_meta_entry = GenMetadataEntry(
                        type=METADATA_TYPE.censorship,
                        value=METADATA_VALUE.nsfw,
                    )
                    completed_job_info.job_image_results[i].generation_faults.append(new_meta_entry)
                    completed_job_info.censored = True
                    if completed_job_info.state != GENERATION_STATE.csam:
                        completed_job_info.state = GENERATION_STATE.censored

        await self._job_tracker.queue_for_submit(completed_job_info)

    def get_deadlock_snapshot(self) -> DeadlockSnapshot:
        """Return a snapshot of the currently detected deadlock state."""
        return DeadlockSnapshot(
            in_deadlock=self._in_deadlock,
            in_queue_deadlock=self._in_queue_deadlock,
            deadlock_started_at=self._last_deadlock_detected_time,
            queue_deadlock_started_at=self._last_queue_deadlock_detected_time,
            queue_deadlock_model=self._queue_deadlock_model,
            queue_deadlock_process_id=self._queue_deadlock_process_id,
        )

    def _print_deadlock_info(self) -> None:
        """Dump the current job/process/model state for a deadlock post-mortem (verbose; throttled)."""
        logger.debug(f"Jobs in queue: {len(self._job_tracker.jobs_pending_inference)}")
        logger.debug(f"Jobs in progress: {len(self._job_tracker.jobs_in_progress)}")
        logger.debug(f"Jobs pending safety check: {len(self._job_tracker.jobs_pending_safety_check)}")
        logger.debug(f"Jobs being safety checked: {len(self._job_tracker.jobs_being_safety_checked)}")
        logger.debug(f"Jobs completed: {len(self._job_tracker.jobs_pending_submit)}")
        logger.debug(f"Jobs faulted: {self._job_tracker.num_jobs_faulted}")
        logger.debug(f"horde_model_map: {self._horde_model_map}")
        logger.debug(f"process_map: {self._process_map}")

    def _should_log_deadlock_detail(self) -> bool:
        """Whether the recurring verbose deadlock dump may be emitted now (throttled per interval).

        The detection-state timestamps mean "first detected" and must stay intact for the snapshot, so a
        separate clock gates the spammy "still detected" dumps to at most once per interval.
        """
        now = time.time()
        if (now - self._last_deadlock_detail_log_time) >= self._DEADLOCK_DETAIL_LOG_INTERVAL_SECONDS:
            self._last_deadlock_detail_log_time = now
            return True
        return False

    def detect_deadlock(self) -> None:
        """Detect if there are jobs in the queue but no processes doing anything."""
        if self._state.last_pop_recently():
            if self._in_deadlock or self._in_queue_deadlock:
                logger.debug("Deadlock cleared after recent job pop.")
            self._in_deadlock = False
            self._in_queue_deadlock = False
            self._queue_deadlock_model = None
            self._queue_deadlock_process_id = None
            return

        queue_deadlock_condition = (
            self._process_map.all_waiting_for_job()
            and len(self._job_tracker.jobs_pending_inference) > 0
            and not any(job in self._job_tracker.jobs_in_progress for job in self._job_tracker.jobs_pending_inference)
        )
        if (
            self._in_queue_deadlock
            and not queue_deadlock_condition
            and (self._process_map.num_starting_processes() == 0 or self._process_map.has_inference_in_progress())
        ):
            # The ``num_starting_processes() == 0`` term is an anti-flap guard so a model-preload window (a
            # re-spawning slot in PROCESS_STARTING) does not prematurely clear a real all-idle deadlock. But a
            # slot genuinely mid-inference disproves the all-idle premise outright, so a slow-to-spawn sibling
            # must not keep the flag latched over a healthy, advancing worker.
            logger.debug("Queue deadlock cleared.")
            self._in_queue_deadlock = False
            self._queue_deadlock_model = None
            self._queue_deadlock_process_id = None

        if not self._in_queue_deadlock and queue_deadlock_condition:
            currently_loaded_models = set()
            model_process_map: dict[str, int] = {}
            for process in self._process_map.values():
                if process.loaded_horde_model_name is not None:
                    currently_loaded_models.add(process.loaded_horde_model_name)
                    model_process_map[process.loaded_horde_model_name] = process.process_id

            for job in self._job_tracker.jobs_pending_inference:
                if job.model is not None and job.model in currently_loaded_models:
                    self._in_queue_deadlock = True
                    self._last_queue_deadlock_detected_time = time.time()
                    self._queue_deadlock_model = job.model
                    self._queue_deadlock_process_id = model_process_map[job.model]
                    break
            else:
                logger.debug("Queue deadlock detected without a model causing it.")
                self._print_deadlock_info()
                self._in_queue_deadlock = True
                self._last_queue_deadlock_detected_time = time.time()
                self._queue_deadlock_model = self._job_tracker.jobs_pending_inference[0].model

        elif self._in_queue_deadlock and (self._last_queue_deadlock_detected_time + 30) < time.time():
            if self._process_map.num_starting_processes() > 0:
                logger.debug("Queue deadlock detected but some processes are starting. Waiting.")
                self._last_queue_deadlock_detected_time = time.time()
                return

            # The detector revisits this branch every tick for the life of the wedge; throttle the
            # verbose dump so a sustained deadlock does not flood the log with identical state.
            if self._should_log_deadlock_detail():
                logger.debug("Queue deadlock still detected after 30 seconds.")
                self._print_deadlock_info()

                if self._queue_deadlock_model is not None:
                    logger.debug(f"Model causing deadlock: {self._queue_deadlock_model}")
                else:
                    logger.warning("Queue deadlock detected but no model causing it.")

            # Keep the flag set so the recovery supervisor can act on a sustained deadlock.

        deadlock_condition = (
            len(self._job_tracker.jobs_pending_inference) > 0
            or len(self._job_tracker.jobs_in_progress) > 0
            or len(self._job_tracker.jobs_lookup) > 0
        ) and self._process_map.num_busy_processes() == 0

        if (not self._in_deadlock) and deadlock_condition:
            self._last_deadlock_detected_time = time.time()
            self._in_deadlock = True
            logger.debug("Deadlock detected")
            self._print_deadlock_info()
        elif self._in_deadlock and (self._last_deadlock_detected_time + 10) < time.time() and deadlock_condition:
            # Recurring every tick while the deadlock persists; share the same throttle as the queue dump.
            if self._should_log_deadlock_detail():
                logger.debug("Deadlock still detected after 10 seconds.")
                self._print_deadlock_info()
        elif self._in_deadlock and not deadlock_condition:
            logger.debug("Deadlock cleared.")
            self._in_deadlock = False
