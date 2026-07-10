from __future__ import annotations

import enum
import queue
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from multiprocessing import Queue

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from horde_sdk.ai_horde_api.fields import GenerationID
from loguru import logger

from horde_worker_regen.consts import AESTHETIC_METADATA_TYPE
from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger, LedgerEventType
from horde_worker_regen.process_management.ipc.messages import (
    AUX_DOWNLOAD_FAILED_INFO,
    HordeAlchemyResultMessage,
    HordeAuxModelStateChangeMessage,
    HordeDownloadAvailabilityMessage,
    HordeDownloadMetricsMessage,
    HordeHeartbeatType,
    HordeInferenceResultMessage,
    HordeJobMetricsMessage,
    HordeModelStateChangeMessage,
    HordePostProcessResultMessage,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    HordeSafetyResultMessage,
    HordeSampleResultMessage,
    HordeTextEncodeResultMessage,
    HordeVaeDecodeResultMessage,
    HordeVaeEncodeResultMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.failure_classification import is_resource_failure
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import InferenceFailureResolution, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap, RetiredProcessLaunch
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from horde_worker_regen.process_management.resources.vram_footprints import (
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    ResolutionBucket,
)
from horde_worker_regen.process_management.scheduling.workload_flow import POST_PROCESS_RESERVE_FLOW
from horde_worker_regen.process_management.workers.download_process import DOWNLOAD_PROCESS_ID
from horde_worker_regen.telemetry_spans import (
    inference_duration_histogram,
    jobs_completed_counter,
    jobs_faulted_counter,
    queue_depth_counter,
)

_excludes_for_job_dump = {"source_image", "source_mask", "extra_source_images", "r2_upload"}

_STAGE_RESULT_MESSAGE_TYPES = (
    HordeTextEncodeResultMessage,
    HordeSampleResultMessage,
    HordeVaeEncodeResultMessage,
    HordeVaeDecodeResultMessage,
)
"""The disaggregated-stage result messages routed to the orchestrator's stage-result handler."""


class _RetiredLaunchMessageAction(enum.Enum):
    """How the dispatcher should treat a message from an intentionally retired launch."""

    NOT_RETIRED = enum.auto()
    IGNORE = enum.auto()
    ACCEPT_POST_PROCESS_RESULT = enum.auto()


_INFERENCE_ACTIVE_STATES = frozenset(
    {
        HordeProcessState.INFERENCE_STARTING,
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
    _reserve_ledger: CommittedReserveLedger
    _on_unload_vram: Callable[[HordeProcessInfo], Awaitable[None]]
    _on_alchemy_result: Callable[[HordeAlchemyResultMessage], None] | None = None
    _on_job_metrics: Callable[[HordeJobMetricsMessage], None] | None = None
    _on_download_metrics: Callable[[HordeDownloadMetricsMessage], None] | None = None
    _on_download_availability: Callable[[HordeDownloadAvailabilityMessage], None] | None = None
    _on_model_load_failure: Callable[[int, str], None] | None = None
    """Invoked as ``(process_id, horde_model_name)`` when a child reports it failed to load a model."""
    _on_inference_step: Callable[[int], None] | None = None
    """Invoked as ``(process_id)`` on each INFERENCE_STEP heartbeat, so the parent's per-step floor can grade
    the slot's sampling pace. Registered by the parent; None leaves the detector idle (standalone tests)."""
    _on_stage_result: Callable[[HordeProcessMessage], Awaitable[None]] | None = None
    """Invoked with a disaggregated-stage result (text-encode/sample/vae-encode/vae-decode), registered
    by the parent's disaggregation orchestrator; it dispatches by concrete message type."""
    _footprint_store: LearnedFootprintStore | None = None
    """Optional learned-footprint store observed on each memory report. Shadow-only: peaks are recorded
    but feed no decision. None (the default) disables observation entirely."""

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
        reserve_ledger: CommittedReserveLedger,
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
            reserve_ledger: Shared committed-resource ledger used to release post-processing holds when a
                result arrives or is known lost.
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
        self._reserve_ledger = reserve_ledger
        self._on_unload_vram = on_unload_vram
        self._state = state
        self._safety_verdicts_known_lost: set[GenerationID] = set()
        """Jobs whose safety verdict was dropped because its producing launch was retired.

        Positive evidence that a verdict will never arrive for these jobs (the message arrived and was
        discarded), as opposed to the orphan watchdog's timeout-based suspicion. Drained by the recovery
        coordinator so a job mid-check when its safety process is replaced is re-queued at once rather than
        only after the watchdog's grace elapses.
        """

        self._post_process_results_known_lost: set[GenerationID] = set()
        """Jobs whose post-processing result was dropped because its producing launch was retired.

        The post-processing analogue of ``_safety_verdicts_known_lost``: positive evidence the result will
        never arrive, drained by the recovery coordinator to skip the orphan watchdog's grace.
        """

    def take_safety_verdicts_known_lost(self) -> set[GenerationID]:
        """Return and clear the jobs whose safety verdict was dropped with their launch retired."""
        lost = self._safety_verdicts_known_lost
        self._safety_verdicts_known_lost = set()
        return lost

    def take_post_process_results_known_lost(self) -> set[GenerationID]:
        """Return and clear the jobs whose post-processing result was dropped with their launch retired."""
        lost = self._post_process_results_known_lost
        self._post_process_results_known_lost = set()
        return lost

    def set_alchemy_result_handler(self, handler: Callable[[HordeAlchemyResultMessage], None]) -> None:
        """Register the callback invoked when a child process reports an alchemy form result."""
        self._on_alchemy_result = handler

    def set_stage_result_handler(self, handler: Callable[[HordeProcessMessage], Awaitable[None]]) -> None:
        """Register the callback invoked when a disaggregated stage reports its result.

        Receives any of the stage result messages (text-encode/sample/vae-encode/vae-decode); the
        registered orchestrator dispatches by concrete type and advances the job's DAG.
        """
        self._on_stage_result = handler

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

    def set_inference_step_observer(self, handler: Callable[[int], None]) -> None:
        """Register the callback invoked with ``(process_id)`` on each INFERENCE_STEP heartbeat.

        Lets the parent's per-step floor grade a slot's sampling pace beat by beat, off the child hot path.
        """
        self._on_inference_step = handler

    def set_footprint_store(self, store: LearnedFootprintStore) -> None:
        """Register the learned-footprint store to observe measured peaks into (shadow-only).

        Once set, each memory report cleanly attributable to a running inference job records its peak
        into the store. The store feeds no decision path; this is measurement for the future arbiter.
        """
        self._footprint_store = store

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

            retired_launch_action = self._classify_retired_launch_message(message)
            if retired_launch_action is _RetiredLaunchMessageAction.IGNORE:
                continue
            if retired_launch_action is _RetiredLaunchMessageAction.ACCEPT_POST_PROCESS_RESULT:
                if isinstance(message, HordePostProcessResultMessage):
                    await self._handle_post_process_result(message)
                else:
                    logger.error(
                        f"Retired-launch classifier accepted an unexpected message type: {type(message).__name__}",
                    )
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
                # An in-flight message from a since-replaced process generation. This is expected and
                # handled (we ignore it), so it is a WARNING, not an ERROR: a single process reload can
                # leave many queued messages behind, and logging each as an error floods the errors-only
                # trace with benign noise that makes a routine replacement (e.g. a maintenance-mode pool
                # reload) look like an error storm in the recovery diagnostics.
                logger.warning(
                    f"Ignoring a stale message from process {message.process_id} (launch identifier "
                    f"{message.process_launch_identifier}, expected {known_launch_identifier}); the process "
                    f"was replaced. Message: {message}",
                )
                continue

            # Adopt the child's self-reported pid (the real interpreter's os.getpid()) as the authoritative
            # os_pid, overriding the parent's spawn-handle guess, so per-PID telemetry addresses the process
            # that actually holds the GPU context even when a launcher stub sits between them. An older child
            # that does not carry the field reports None and keeps the handle-derived value.
            self._process_map.reconcile_reported_os_pid(
                message.process_id,
                getattr(message, "reported_os_pid", None),
            )

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
            elif isinstance(message, HordePostProcessResultMessage):
                self._record_completed_job(message.process_id)
                await self._handle_post_process_result(message)
            elif isinstance(message, HordeSafetyResultMessage):
                self._record_completed_job(message.process_id)
                await self._handle_safety_result(message)
            elif isinstance(message, HordeAlchemyResultMessage):
                self._record_completed_job(message.process_id)
                if self._on_alchemy_result is not None:
                    self._on_alchemy_result(message)
                else:
                    logger.error(f"Received alchemy result with no handler registered: {message.form_id}")
            elif isinstance(message, _STAGE_RESULT_MESSAGE_TYPES):
                self._record_completed_job(message.process_id)
                if self._on_stage_result is not None:
                    await self._on_stage_result(message)
                else:
                    logger.error(f"Received {type(message).__name__} with no stage-result handler registered")

    def _classify_retired_launch_message(self, message: HordeProcessMessage) -> _RetiredLaunchMessageAction:
        """Classify a message that may belong to an intentionally retired launch."""
        retired_launch = self._process_map.get_retired_launch(
            message.process_id,
            message.process_launch_identifier,
        )
        if retired_launch is None:
            return _RetiredLaunchMessageAction.NOT_RETIRED

        if isinstance(
            message,
            (
                HordeInferenceResultMessage,
                HordeSafetyResultMessage,
                HordeAlchemyResultMessage,
                HordePostProcessResultMessage,
                *_STAGE_RESULT_MESSAGE_TYPES,
            ),
        ):
            if isinstance(message, HordePostProcessResultMessage) and self._should_accept_retired_post_process_result(
                message,
                retired_launch,
            ):
                logger.info(
                    f"Accepting post-processing result from retired post_process process {message.process_id} "
                    f"launch {message.process_launch_identifier} for job {message.job_id}.",
                )
                return _RetiredLaunchMessageAction.ACCEPT_POST_PROCESS_RESULT

            logger.warning(
                f"Ignoring result message from retired {retired_launch.process_type.name.lower()} process "
                f"{message.process_id} launch {message.process_launch_identifier} "
                f"({retired_launch.reason}): {type(message).__name__}",
            )
            # A safety verdict dropped here is the only signal that the job it was checking will never get a
            # verdict from that launch; flag it so the recovery coordinator re-checks it at once instead of
            # leaving it stranded in SAFETY_CHECKING until the orphan watchdog's grace elapses. Replacing the
            # safety process is routine (whole-card residency moves it off and back onto the GPU), so without
            # this several such drops can pile up faster than the watchdog clears them and wedge the pipeline.
            if isinstance(message, HordeSafetyResultMessage):
                self._safety_verdicts_known_lost.add(message.job_id)
            if isinstance(message, HordePostProcessResultMessage):
                self._post_process_results_known_lost.add(message.job_id)
                self._release_post_process_reserve(message.job_id)
            return _RetiredLaunchMessageAction.IGNORE

        if self._is_late_retired_liveness_message(message):
            logger.debug(
                f"Ignoring late {type(message).__name__} from retired "
                f"{retired_launch.process_type.name.lower()} process {message.process_id} "
                f"launch {message.process_launch_identifier} ({retired_launch.reason})",
            )
            return _RetiredLaunchMessageAction.IGNORE

        logger.warning(
            f"Ignoring unexpected {type(message).__name__} from retired "
            f"{retired_launch.process_type.name.lower()} process {message.process_id} "
            f"launch {message.process_launch_identifier} ({retired_launch.reason})",
        )
        return _RetiredLaunchMessageAction.IGNORE

    def _should_accept_retired_post_process_result(
        self,
        message: HordePostProcessResultMessage,
        retired_launch: RetiredProcessLaunch,
    ) -> bool:
        """Return whether a retired post-processing launch still owns the job result it produced."""
        if retired_launch.process_type is not HordeProcessType.POST_PROCESS:
            return False
        if message.state == GENERATION_STATE.faulted or message.job_image_results is None:
            return False
        return self._job_tracker.is_current_post_processing_attempt(
            message.job_id,
            process_id=message.process_id,
            process_launch_identifier=message.process_launch_identifier,
        )

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

    def _release_post_process_reserve(self, job_id: GenerationID) -> None:
        """Drop the active post-processing VRAM reserve for ``job_id``."""
        self._reserve_ledger.release(POST_PROCESS_RESERVE_FLOW, str(job_id))

    def _handle_heartbeat(self, message: HordeProcessHeartbeatMessage) -> None:
        """Handle a heartbeat message from a child process."""
        self._process_map.on_heartbeat(
            message.process_id,
            heartbeat_type=message.heartbeat_type,
            percent_complete=message.percent_complete,
            current_step=message.current_step,
            total_steps=message.total_steps,
            iterations_per_second=message.iterations_per_second,
            nonadvancing_step_repeats=message.nonadvancing_step_repeats,
        )

        # Grade the slot's sampling pace beat by beat (the per-step floor). Runs after on_heartbeat so the
        # freshly-updated inter-beat delta and step count are what the detector reads.
        if message.heartbeat_type == HordeHeartbeatType.INFERENCE_STEP and self._on_inference_step is not None:
            self._on_inference_step(message.process_id)

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
            open_fds=message.open_fds,
            fd_soft_limit=message.fd_soft_limit,
            process_reserved_mb=message.process_reserved_mb,
            process_allocated_mb=message.process_allocated_mb,
            process_peak_reserved_mb=message.process_peak_reserved_mb,
            process_aimdo_mb=message.process_aimdo_mb,
            report_sampled_at=message.sampled_at,
        )
        self._observe_footprint_peak(message)

    def _observe_footprint_peak(self, message: HordeProcessMemoryMessage) -> None:
        """Record a reported VRAM peak into the learned-footprint store, if cleanly attributable.

        The measured peaks recorded here raise admission pricing of sampling work (the scheduler reads the
        same store when pricing a job's sampling peak). Only the unambiguous case is wired here: a monolithic
        inference process whose single tracked job is genuinely in progress, whose model baseline is known, and
        whose peak reading is positive. The disaggregated case is observed by the orchestrator, not here.
        Such a report's peak is attributed to the SAMPLE stage (the dominant activation term of a whole
        monolithic job). Reports from the disaggregated lanes (VAE/text-encode/post-process), from idle or
        between-job inference slots, or with an unknown baseline are left unattributed rather than guessed:
        the parent cannot reliably bind those peaks to one stage/job at this seam.
        """
        store = self._footprint_store
        if store is None:
            return

        peak_mb = message.process_peak_reserved_mb
        if peak_mb is None or peak_mb <= 0:
            return

        process_info = self._process_map.get(message.process_id)
        if process_info is None or process_info.process_type is not HordeProcessType.INFERENCE:
            return

        model_name = process_info.loaded_horde_model_name
        job = process_info.last_job_referenced
        if model_name is None or job is None:
            return

        # Bind the peak to a genuinely-running job: the slot's referenced job must be in progress, which
        # ties the peak-since-last-report to that job's sampling rather than to a merely-preloaded slot.
        if job not in self._job_tracker.jobs_in_progress:
            return

        baseline = self._model_metadata.get_baseline(model_name)
        if baseline is None:
            return

        width = job.payload.width
        height = job.payload.height
        if width is None or height is None:
            return

        key = FootprintKey(
            model_baseline=str(baseline),
            resolution_bucket=ResolutionBucket.from_dimensions(width, height, job.payload.n_iter or 1),
            platform=sys.platform,
            stage=FootprintStage.SAMPLE,
        )
        store.observe_peak(key, float(peak_mb))

    def _handle_process_state_change(self, message: HordeProcessStateChangeMessage) -> None:
        """Handle a process state change message."""
        if self._process_map[message.process_id].last_process_state == message.process_state:
            return

        # Captured before the slot's state is advanced below; the lost-result reap needs the state the
        # slot is transitioning *from* to tell a post-inference idle from the dispatch window.
        previous_state = self._process_map[message.process_id].last_process_state
        previous_state_started_at = self._process_map[message.process_id].last_process_state_started_at

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

        if message.process_state == HordeProcessState.TORCH_GPU_INCOMPATIBLE:
            # A torch-bearing inference child found the installed PyTorch has no kernels for this GPU. Latch
            # it here (the parent never imports torch) so the poppers stop popping and the TUI can surface
            # the reason. The child carries the operator-facing detail in `info`; relay it verbatim. The
            # mismatch is a build/hardware fact, so this is sticky for the session.
            if not self._state.gpu_torch_incompatible:
                logger.critical(
                    f"Process {message.process_id} reports the installed PyTorch cannot run this GPU; "
                    f"the worker will stop popping jobs. {message.info}",
                )
            self._state.gpu_torch_incompatible = True
            self._state.gpu_torch_incompatible_reason = message.info

        if message.process_state == HordeProcessState.TORCH_BUILD_CPU_ONLY:
            # A torch-bearing inference child found the installed PyTorch is a CPU-only build. Latch it here
            # (the parent never imports torch) so the image popper stops popping while alchemy keeps running,
            # and the TUI can surface the reason. Build fact, so sticky for the session. This is the runtime
            # equivalent of a 'cpu' install sentinel, for a CPU torch build whose sentinel was never set.
            if not self._state.torch_build_cpu_only:
                logger.warning(
                    f"Process {message.process_id} reports a CPU-only torch build; image generation is "
                    f"disabled (alchemy continues). {message.info}",
                )
            self._state.torch_build_cpu_only = True
            self._state.torch_build_cpu_only_reason = message.info

        # INFERENCE_PRIMED is the dispatch-edge state the whole-job slot now reports (it advances to
        # INFERENCE_STARTING parent-side on the first step), so the model-in-use bookkeeping fires here;
        # stage lanes (disaggregated sample, text-encode) still emit INFERENCE_STARTING directly, so both
        # are handled. The bookkeeping is idempotent for the primed->starting upgrade.
        if message.process_state in (
            HordeProcessState.INFERENCE_PRIMED,
            HordeProcessState.INFERENCE_STARTING,
        ):
            self._handle_inference_starting(message.process_id)

        if (
            message.process_state == HordeProcessState.UNLOADED_MODEL_FROM_RAM
            and self._process_map[message.process_id].last_process_state != HordeProcessState.UNLOADED_MODEL_FROM_RAM
        ):
            logger.opt(ansi=True).info(
                f"<fg #7b7d7d>Process {message.process_id} cleared RAM: {message.info}</>",
            )
            self._process_map.on_model_ram_clear(process_id=message.process_id)

        if message.process_state == HordeProcessState.WAITING_FOR_JOB:
            self._reap_lost_inference_result(
                message.process_id,
                previous_state=previous_state,
                previous_state_started_at=previous_state_started_at,
            )

    def _handle_inference_starting(self, process_id: int) -> None:
        """Apply the whole-job inference bookkeeping when an INFERENCE process reports it is sampling.

        The parent-tracked model residency and batch invariant belong to whole-job inference on an INFERENCE
        process. The disaggregated stage lanes reuse busy states to mark themselves working (the encode
        service reports ``INFERENCE_STARTING``, the image lane ``POST_PROCESSING``) but hold no parent-tracked
        model on that process, so they carry none of this bookkeeping; applying it would fault on their absent
        model. A disaggregated sample stage runs on an INFERENCE process and keeps the loaded-model invariant
        (its UNet is resident), but it batches per-slice and reports no matching model-load transition to flip
        an ``IN_USE`` mark back, so the whole-job batch guard and the model-map ``IN_USE`` update (which assume
        a whole job) are skipped for a pinned sampler. The reservation is that sampler's marker and is still
        held when it reports ``INFERENCE_STARTING`` (it is released only on the sample result).
        """
        process_info = self._process_map[process_id]
        if process_info.process_type != HordeProcessType.INFERENCE:
            return
        loaded_model_name = process_info.loaded_horde_model_name
        if loaded_model_name is None:
            raise ValueError(
                f"Process {process_id} has no model loaded, but is starting inference",
            )
        if self._process_map.is_reserved_for_disaggregation(process_id):
            return
        if process_info.batch_amount is None:
            raise ValueError(
                f"Process {process_id} is starting inference without a batch amount",
            )
        self._horde_model_map.update_entry(
            horde_model_name=loaded_model_name,
            load_state=ModelLoadState.IN_USE,
            process_id=process_id,
        )

    def _reap_lost_inference_result(
        self,
        process_id: int,
        *,
        previous_state: HordeProcessState,
        previous_state_started_at: float,
    ) -> None:
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
        window, so the job is left for the slot to take up. The same ordering check also needs the active
        dispatch timestamp: if a newer dispatch was stamped after the state being closed began, then the
        idle report belongs to older slot work and must not release the newer job.
        """
        if previous_state not in _INFERENCE_ACTIVE_STATES:
            return
        process_info = self._process_map.get(process_id)
        if process_info is None:
            return
        if (
            process_info.current_inference_started_at is not None
            and process_info.current_inference_started_at > previous_state_started_at
        ):
            return
        # Only a whole-job INFERENCE slot can lose an inference result here. A disaggregated stage lane
        # (the encode service reports INFERENCE_STARTING, then WAITING_FOR_JOB) would otherwise trip this
        # inference-active-to-idle check though it holds no whole-job result; the orchestrator owns its
        # stage results, so this reap does not apply to it.
        if process_info.process_type != HordeProcessType.INFERENCE:
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
            # The job's LoRAs are now on disk; record them so a later pending job needing the same LoRAs may
            # line-skip an aux-download-blocked lane without itself triggering a fresh blocking download.
            self._job_tracker.mark_job_loras_cached(message.sdk_api_job_info)
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
                    # Stamp the VRAM-materialization time so the reclaim ladder can rank this idle resident by
                    # recency (LIFO): the driver demotes the least-recently-touched allocator, so the newest
                    # resident is reclaimed first.
                    self._process_map.note_vram_materialized(message.process_id)
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

    async def handle_synthetic_inference_result(self, message: HordeInferenceResultMessage) -> None:
        """Route a parent-synthesized inference result through the same completion path as a child's.

        The disaggregation orchestrator assembles a :class:`HordeInferenceResultMessage` from a job whose
        stages ran across the encode service, samplers, and image lane, then hands it here so its images (or
        its fault) flow into the identical safety/submit path a monolithic child result takes. The images are
        the VAE lane's raw decode output, so a job requesting post-processing routes to the dedicated
        post-processing lane exactly as a monolithic completion does.
        """
        await self._handle_inference_result(message, is_disaggregated_completion=True)

    async def _handle_inference_result(
        self,
        message: HordeInferenceResultMessage,
        *,
        is_disaggregated_completion: bool = False,
    ) -> None:
        """Handle an inference job result message.

        ``is_disaggregated_completion`` is set for a parent-synthesized disaggregated completion: the
        in-progress release tolerates the job's disaggregation-decoding stage (it never sat in
        ``INFERENCE_IN_PROGRESS`` at completion). Post-processing routing is identical to a monolithic
        completion: the synthetic result carries raw decoded images, so a job requesting post-processing is
        queued for the dedicated post-processing lane.
        """
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

        released = await self._job_tracker.release_in_progress(message.sdk_api_job_info)
        # A disaggregated completion never sits in INFERENCE_IN_PROGRESS at this point: the sampler slot was
        # released (and the job moved to the disaggregation-decoding stage) the moment sampling finished, so a
        # failed release is expected there and not an error.
        if not released and not is_disaggregated_completion:
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

        requested_post_processing = job_info.sdk_api_job_info.payload.post_processing
        if requested_post_processing:
            # Disaggregation forces the post-processing lane on regardless of the lane's own config flag, the
            # same way it forces the VAE lane on, so a disaggregated completion routes to the lane here.
            lane_enabled = bridge_data.post_processing_lane_enabled or bridge_data.enable_pipeline_disaggregation
            if lane_enabled:
                await self._job_tracker.queue_for_post_processing(job_info)
                return
            # The lane is the only post-processing path; with it disabled the job should never have been
            # popped with post-processing at all. Report a no-image fault so the horde reissues it instead
            # of silently returning a result that did not receive the requested post-processing.
            reason = "job requested post-processing but the dedicated post-processing lane is disabled"
            logger.error(
                f"Job {message.sdk_api_job_info.id_} requested post-processing "
                f"({requested_post_processing}) but the dedicated post-processing lane is disabled; "
                "reporting the job faulted without images.",
            )
            self._job_tracker.note_post_processing_overcommit_fault()
            await self._job_tracker.fault_post_inference_job(job_info, reason=reason)
            return

        await self._job_tracker.queue_for_safety(job_info)

    async def _handle_post_process_result(self, message: HordePostProcessResultMessage) -> None:
        """Handle a post-processing result: adopt the processed images and move the job on to safety.

        A faulted post-processing pass is reported to the horde as a no-image fault. The worker advertised
        post-processing for this job, so returning raw images would violate the job contract; the horde should
        reissue the job to another worker instead.
        """
        self._release_post_process_reserve(message.job_id)

        if message.time_elapsed is not None:
            logger.info(
                f"Post-processing finished for job {str(message.job_id)[:8]} in "
                f"{round(message.time_elapsed, 2)} seconds on process {message.process_id}.",
            )

        if message.state == GENERATION_STATE.faulted or message.job_image_results is None:
            fault_reason = message.fault_reason or message.info or "post-processing failed"
            tracked = self._job_tracker.get_tracked_job(message.job_id)
            if tracked is None or tracked.job_info is None:
                logger.error(
                    f"Post-processing faulted for job {message.job_id} on process {message.process_id}, "
                    "but the job is no longer tracked.",
                )
                return
            logger.error(
                f"Post-processing faulted for job {message.job_id} on process {message.process_id}; "
                f"reporting the job faulted without images so the horde reissues it: {fault_reason}",
            )
            self._job_tracker.note_post_processing_overcommit_fault()
            self._action_ledger.record(
                LedgerEventType.POST_PROCESS_FAULTED,
                process_id=message.process_id,
                job_id=str(message.job_id),
                reason=fault_reason,
                detail={
                    "fallback": "fault_no_images",
                    "resource_class": message.fault_is_resource_class,
                },
            )
            await self._job_tracker.fault_post_inference_job(
                tracked.job_info,
                reason=fault_reason,
            )
            return

        completed_job_info = await self._job_tracker.take_being_post_processed(message.job_id)

        if completed_job_info is None:
            logger.error(
                f"Expected to find a job being post-processed with ID {message.job_id} but none was found. "
                "This should only happen when certain process crashes occur.",
            )
            return

        completed_job_info.job_image_results = message.job_image_results

        await self._job_tracker.queue_for_safety_post_processed(completed_job_info)

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
        # is not a resource/OOM failure, it arms the LoRA-download backoff, and (once an incident is
        # active) it is dropped rather than requeued straight back into the same failing download.
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
            replacement_image = message.safety_evaluations[i].replacement_image_bytes

            if message.safety_evaluations[i].failed:
                logger.error(
                    f"Job {message.job_id} image #{i} faulted during safety checks. "
                    "Check the safety process logs for more information.",
                )
                any_safety_failed = True
                continue

            # The aesthetic score (when the safety pass produced one) rides along as gen_metadata so a
            # client can rank/curate generations. The float lives in `ref`; `value` is the categorical
            # see_ref sentinel, matching how batch_index/information entries carry a non-enum payload.
            aesthetic_score = message.safety_evaluations[i].aesthetic_score
            if aesthetic_score is not None:
                completed_job_info.job_image_results[i].generation_faults.append(
                    GenMetadataEntry(
                        type=AESTHETIC_METADATA_TYPE,
                        value=METADATA_VALUE.see_ref,
                        ref=str(aesthetic_score),
                    ),
                )

            if replacement_image is not None:
                completed_job_info.job_image_results[i].image_bytes = replacement_image
                num_images_censored += 1
                if message.safety_evaluations[i].is_csam:
                    num_images_csam += 1
        if completed_job_info.sdk_api_job_info.id_ is None:
            logger.error(
                f"Job {message.job_id} has no id; cannot clear its fault entries. This is unexpected.",
            )
        elif job_fault_entries:
            await self._job_tracker.clear_faults_for_job(completed_job_info.sdk_api_job_info.id_)

        # Feed the post-inference backpressure model: the popper throttles new pops when the safety
        # backlog can no longer clear within the job ttl, so a safety stage that is slower than inference
        # cannot grow an unbounded backlog that ages jobs out into horde-forced maintenance.
        safety_elapsed = message.time_elapsed
        if safety_elapsed is not None:
            self._state.record_safety_duration(safety_elapsed)
            safety_elapsed_display = f"{safety_elapsed:.2f}"
        else:
            safety_elapsed_display = "unknown"

        logger.debug(
            f"Job {message.job_id} had {num_images_censored} images censored and took "
            f"{safety_elapsed_display} seconds to check safety",
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
                if message.safety_evaluations[i].replacement_image_bytes is None:
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

        if any_safety_failed:
            # The safety process could not produce a verdict for at least one image, so we cannot vouch
            # that this job's images are safe. Per the no-unchecked-submit invariant we drop every image
            # and report the whole job faulted (the horde reissues it) rather than upload an image the
            # safety classifier never cleared. ``fault_job`` clears ``job_image_results`` so the submitter
            # uploads nothing; ``safety_evaluated`` is deliberately left False to reflect that no clean
            # verdict was obtained.
            logger.error(
                f"Job {message.job_id} had a safety evaluation failure; dropping its images and faulting it "
                "so the horde reissues it (an image the safety check could not clear is never submitted).",
            )
            completed_job_info.fault_job()
        else:
            # The verdict has now been applied to every image, so mark the job safety-evaluated. This is the
            # one and only writer of this flag; the submit boundary refuses to upload any job carrying images
            # without it, so a job whose safety result was lost (and never re-checked) can never be submitted.
            completed_job_info.safety_evaluated = True

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
