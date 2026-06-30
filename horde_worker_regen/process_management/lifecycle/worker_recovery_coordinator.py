"""Coordinate worker-level recovery watchdogs and save-our-ship escalation."""

from __future__ import annotations

import time
from collections.abc import Callable

from horde_sdk.ai_horde_api.fields import GenerationID
from loguru import logger

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger, LedgerEventType
from horde_worker_regen.process_management.ipc.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.ipc.messages import HordeControlFlag
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.lifecycle.recovery_supervisor import RecoveryAction, RecoverySupervisor
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler


class WorkerRecoveryCoordinator:
    """Coordinate worker-level watchdogs, wedge assessment, and recovery actions."""

    ORPHAN_IN_PROGRESS_GRACE_SECONDS = 30.0
    ORPHAN_PUNT_WINDOW_SECONDS = 300.0
    ORPHAN_PUNT_WEDGE_THRESHOLD = 3
    ORPHAN_SAFETY_GRACE_SECONDS = 45.0
    SAFETY_REQUEUE_MAX = 3
    SAFETY_SOFT_PAUSE_SECONDS = 60.0
    RUNAWAY_RECOVERY_WINDOW_SECONDS = 300.0
    RUNAWAY_RECOVERY_CEILING = 20

    def __init__(
        self,
        *,
        state: WorkerState,
        runtime_config: RuntimeConfig,
        job_tracker: JobTracker,
        process_map: ProcessMap,
        process_lifecycle: ProcessLifecycleManager,
        message_dispatcher: MessageDispatcher,
        inference_scheduler: InferenceScheduler,
        action_ledger: ActionLedger,
        bridge_data_provider: Callable[[], reGenBridgeData],
        max_inference_processes_provider: Callable[[], int],
        abort_callback: Callable[[], None],
        recovery_supervisor: RecoverySupervisor | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the recovery coordinator.

        Args:
            state: Mutable worker state shared with other orchestration collaborators.
            runtime_config: Live runtime config used to apply limp-by concurrency.
            job_tracker: Single source of truth for job stages and fault handling.
            process_map: Live process state map.
            process_lifecycle: Process lifecycle facade for pool rebuilds and capacity checks.
            message_dispatcher: Dispatcher that owns queue-deadlock snapshots.
            inference_scheduler: Scheduler exposing bounded grace windows.
            action_ledger: Recovery/action audit sink.
            bridge_data_provider: Return the current live bridge data.
            max_inference_processes_provider: Return the provisioned inference-process count.
            abort_callback: Abort the worker promptly.
            recovery_supervisor: Optional recovery policy object for tests.
            clock: Wall-clock provider for grace windows and rolling recovery counts.
        """
        self._state = state
        self._runtime_config = runtime_config
        self._job_tracker = job_tracker
        self._process_map = process_map
        self._process_lifecycle = process_lifecycle
        self._message_dispatcher = message_dispatcher
        self._inference_scheduler = inference_scheduler
        self._action_ledger = action_ledger
        self._bridge_data_provider = bridge_data_provider
        self._max_inference_processes_provider = max_inference_processes_provider
        self._abort_callback = abort_callback
        self._clock = clock

        self.recovery_supervisor = recovery_supervisor or RecoverySupervisor()
        self.limp_by_active = False
        self.episode_saw_unrecoverable_pool = False
        self.episode_progress_baseline: int | None = None
        self.recovery_event_times: list[float] = []
        self.last_seen_recovery_count = 0
        self.orphan_in_progress_since: dict[GenerationID, float] = {}
        self.orphan_punt_history: list[float] = []
        self.orphan_safety_since: dict[GenerationID, float] = {}
        self.safety_requeue_count: dict[GenerationID, int] = {}

    @property
    def bridge_data(self) -> reGenBridgeData:
        """Return the current live bridge data."""
        return self._bridge_data_provider()

    @property
    def max_inference_processes(self) -> int:
        """Return the provisioned inference-process count."""
        return self._max_inference_processes_provider()

    def is_inference_capacity_available(self) -> bool:
        """Return whether any inference process is alive to serve pending inference work."""
        return any(
            process_info.process_type == HordeProcessType.INFERENCE and process_info.is_process_alive()
            for process_info in self._process_map.values()
        )

    def is_safety_capacity_available(self) -> bool:
        """Return whether any safety process is alive to serve pending safety checks."""
        return any(
            process_info.process_type == HordeProcessType.SAFETY and process_info.is_process_alive()
            for process_info in self._process_map.values()
        )

    def is_safety_pool_ready(self) -> bool:
        """Return whether at least one safety process is alive and able to accept a check."""
        return any(
            process_info.process_type == HordeProcessType.SAFETY and process_info.can_accept_job()
            for process_info in self._process_map.values()
        )

    def is_inference_pool_unrecoverable(self) -> bool:
        """Return whether every inference slot is crash-loop quarantined."""
        return len(self._process_lifecycle.quarantined_inference_slots) >= self.max_inference_processes

    def is_safety_pool_unrecoverable(self) -> bool:
        """Return whether the safety pool is crash-looping and not currently ready."""
        return self._process_lifecycle.safety_pool_failing and not self.is_safety_pool_ready()

    def inference_slot_owns_job(self, job_id: GenerationID) -> bool:
        """Return whether some live inference slot owns the given job."""
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if not process_info.is_process_alive():
                continue
            referenced = process_info.last_job_referenced
            if referenced is None or referenced.id_ != job_id:
                continue
            if not process_info.can_accept_job():
                return True
            if (
                process_info.last_control_flag == HordeControlFlag.START_INFERENCE
                and process_info.current_inference_started_at is not None
            ):
                return True
        return False

    def reconcile_orphaned_in_progress_jobs(self) -> None:
        """Punt jobs stuck in inference with no owning live slot."""
        now = self._clock()
        in_progress = self._job_tracker.jobs_in_progress
        live_ids = {job.id_ for job in in_progress if job.id_ is not None and self.inference_slot_owns_job(job.id_)}

        current_ids = {job.id_ for job in in_progress if job.id_ is not None}
        for job_id in list(self.orphan_in_progress_since):
            if job_id not in current_ids or job_id in live_ids:
                del self.orphan_in_progress_since[job_id]

        for job in in_progress:
            job_id = job.id_
            if job_id is None or job_id in live_ids:
                continue
            first_seen = self.orphan_in_progress_since.setdefault(job_id, now)
            if (now - first_seen) < self.ORPHAN_IN_PROGRESS_GRACE_SECONDS:
                continue

            logger.error(
                f"Job {job_id} has been in progress with no live inference slot for "
                f"{now - first_seen:.0f}s; punting it so the queue can drain (orphaned-job watchdog).",
            )
            self._action_ledger.record(
                LedgerEventType.INFERENCE_FAULTED,
                job_id=str(job_id),
                reason="orphaned in-progress job (no owning live inference slot)",
                detail={"stuck_seconds": round(now - first_seen, 1)},
            )
            self._job_tracker.handle_job_fault_now(
                faulted_job=job,
                process_timeout=self.bridge_data.process_timeout,
                retryable=True,
                scheduling_fault=True,
            )
            del self.orphan_in_progress_since[job_id]
            self.orphan_punt_history.append(now)

    def orphan_wedge_active(self) -> bool:
        """Return whether recurring orphan punts count as a worker-level wedge."""
        now = self._clock()
        self.orphan_punt_history = [
            recovery_time
            for recovery_time in self.orphan_punt_history
            if (now - recovery_time) <= self.ORPHAN_PUNT_WINDOW_SECONDS
        ]
        return len(self.orphan_punt_history) >= self.ORPHAN_PUNT_WEDGE_THRESHOLD

    def engage_safety_soft_pause(self, reason: str) -> None:
        """Soft-pause job popping because safety could not check a result."""
        until = self._clock() + self.SAFETY_SOFT_PAUSE_SECONDS
        if self._state.self_throttle_paused and self._state.self_throttle_paused_until >= until:
            return
        self._state.self_throttle_paused = True
        self._state.self_throttle_paused_until = until
        logger.warning(
            f"Soft-pausing job pops for {self.SAFETY_SOFT_PAUSE_SECONDS:.0f}s: safety could not check a "
            f"result ({reason}). In-flight checked jobs still submit; pops resume automatically once safety "
            "recovers, so the worker does not keep taking on work it cannot safety-check.",
        )

    async def reconcile_orphaned_safety_jobs(self) -> None:
        """Recover jobs stranded in safety checking whose verdict will never return."""
        now = self._clock()
        checking = self._job_tracker.jobs_being_safety_checked
        current_ids = {info.sdk_api_job_info.id_ for info in checking if info.sdk_api_job_info.id_ is not None}

        # Jobs whose verdict was positively dropped (their safety launch was retired mid-check) skip the
        # grace: the verdict is known lost, not merely late. Backdating first-seen routes them through the
        # same requeue/escalation bookkeeping below, so a job is still bounded out if its re-checks keep
        # failing rather than looping forever.
        for job_id in self._message_dispatcher.take_safety_verdicts_known_lost():
            if job_id in current_ids:
                self.orphan_safety_since[job_id] = now - self.ORPHAN_SAFETY_GRACE_SECONDS

        for job_id in list(self.orphan_safety_since):
            if job_id not in current_ids:
                del self.orphan_safety_since[job_id]
        for job_id in list(self.safety_requeue_count):
            if job_id not in current_ids and self._job_tracker.get_stage(job_id) != JobStage.PENDING_SAFETY_CHECK:
                del self.safety_requeue_count[job_id]

        pool_unrecoverable = self.is_safety_pool_unrecoverable()

        for job_info in checking:
            job = job_info.sdk_api_job_info
            job_id = job.id_
            if job_id is None:
                continue
            first_seen = self.orphan_safety_since.setdefault(job_id, now)
            if (now - first_seen) < self.ORPHAN_SAFETY_GRACE_SECONDS:
                continue

            requeues = self.safety_requeue_count.get(job_id, 0)
            if pool_unrecoverable or requeues >= self.SAFETY_REQUEUE_MAX:
                reason = (
                    "safety pool unrecoverable (crash-looping)"
                    if pool_unrecoverable
                    else f"requeued {requeues} times without a verdict"
                )
                logger.critical(
                    f"Job {job_id} could not be safety-checked ({reason}); dropping its images and faulting "
                    "it so the horde reissues it (an image the safety check never cleared is never "
                    "submitted). Soft-pausing pops until safety recovers.",
                )
                job_info.fault_job()
                self._action_ledger.record(
                    LedgerEventType.INFERENCE_FAULTED,
                    job_id=str(job_id),
                    reason=f"safety check unrecoverable ({reason})",
                    detail={"stuck_seconds": round(now - first_seen, 1), "safety_requeues": requeues},
                )
                self._job_tracker.handle_job_fault_now(
                    faulted_job=job,
                    process_timeout=self.bridge_data.process_timeout,
                    retryable=False,
                    scheduling_fault=True,
                )
                self.orphan_safety_since.pop(job_id, None)
                self.safety_requeue_count.pop(job_id, None)
                self.engage_safety_soft_pause(reason)
                continue

            if await self._job_tracker.requeue_one_being_safety_checked(job_id):
                self.safety_requeue_count[job_id] = requeues + 1
                self.orphan_safety_since.pop(job_id, None)
                if not self.is_safety_pool_ready():
                    self._process_lifecycle.safety_processes_should_be_replaced = True
                logger.warning(
                    f"Job {job_id} awaited a safety verdict for {now - first_seen:.0f}s with none returned; "
                    f"requeued it for a fresh safety check (attempt {requeues + 1}/{self.SAFETY_REQUEUE_MAX}). "
                    "Its images are re-checked, never submitted unchecked.",
                )

    def assess_wedge(self) -> bool:
        """Return whether the worker structurally cannot make progress."""
        if self._state.shutting_down:
            return False
        if self._state.downloads_only_hold:
            return False
        structural_queue_wedge = self._message_dispatcher.get_deadlock_snapshot().indicates_structural_wedge()
        if structural_queue_wedge and (
            self._inference_scheduler.whole_card_residency_grace_active()
            or self._inference_scheduler.heavy_head_load_grace_active()
            or self._inference_scheduler.ram_reclaim_cycle_grace_active()
        ):
            structural_queue_wedge = False
        if structural_queue_wedge and self._process_map.has_inference_in_progress():
            structural_queue_wedge = False
        return (
            self.is_inference_pool_unrecoverable()
            or self.is_safety_pool_unrecoverable()
            or structural_queue_wedge
            or self.orphan_wedge_active()
        )

    def made_progress_since_episode(self) -> bool:
        """Return whether a job completed since the current wedge episode opened."""
        if self.episode_progress_baseline is None:
            return False
        return self._job_tracker.total_num_completed_jobs > self.episode_progress_baseline

    def maybe_abort_on_runaway_recoveries(self) -> bool:
        """Abort if process recoveries are flapping faster than the rolling-window ceiling."""
        current = self._process_lifecycle._num_process_recoveries
        if current < self.last_seen_recovery_count:
            self.recovery_event_times.clear()
            self.last_seen_recovery_count = current
            return False
        now = self._clock()
        new_recoveries = current - self.last_seen_recovery_count
        self.last_seen_recovery_count = current
        self.recovery_event_times.extend([now] * new_recoveries)
        cutoff = now - self.RUNAWAY_RECOVERY_WINDOW_SECONDS
        self.recovery_event_times = [
            recovery_time for recovery_time in self.recovery_event_times if recovery_time >= cutoff
        ]
        if len(self.recovery_event_times) < self.RUNAWAY_RECOVERY_CEILING or self._state.shutting_down:
            return False
        logger.critical(
            f"Save-our-ship: {len(self.recovery_event_times)} process recoveries within "
            f"{self.RUNAWAY_RECOVERY_WINDOW_SECONDS:.0f}s (ceiling {self.RUNAWAY_RECOVERY_CEILING}); the worker "
            "is flapping and cannot stabilise. Abandoning ship (the last resort) rather than recovering forever.",
        )
        self._action_ledger.record(
            LedgerEventType.RECOVERY_ABANDONED,
            reason="save-our-ship: runaway process-recovery rate (flapping pool)",
            detail={
                "recoveries_in_window": len(self.recovery_event_times),
                "window_seconds": self.RUNAWAY_RECOVERY_WINDOW_SECONDS,
            },
        )
        self._abort_callback()
        return True

    def run_recovery_supervisor(self) -> None:
        """Drive save-our-ship escalation one tick and perform any returned action."""
        if self._state.shutting_down:
            return
        if self.maybe_abort_on_runaway_recoveries():
            return
        is_wedged = self.assess_wedge()
        if self.is_inference_pool_unrecoverable() or self.is_safety_pool_unrecoverable():
            self.episode_saw_unrecoverable_pool = True
        if self.episode_saw_unrecoverable_pool:
            if self.made_progress_since_episode():
                self.episode_saw_unrecoverable_pool = False
            else:
                is_wedged = True
        action = self.recovery_supervisor.evaluate(is_wedged=is_wedged)
        if self.recovery_supervisor.is_in_episode:
            if self.episode_progress_baseline is None:
                self.episode_progress_baseline = self._job_tracker.total_num_completed_jobs
        else:
            self.episode_progress_baseline = None
            self.episode_saw_unrecoverable_pool = False
        if action is RecoveryAction.SOFT_RESET:
            self.perform_soft_reset()
            self.limp_by_active = True
        elif action is RecoveryAction.GIVE_UP:
            self.give_up_on_wedged_jobs()
        elif self.limp_by_active and not self.recovery_supervisor.is_in_episode:
            self.limp_by_active = False
            self._runtime_config.set_effective_max_threads(self.bridge_data.max_threads)
            logger.info("Save-our-ship: pools recovered; restored configured concurrency (limp-by cleared).")

    def perform_soft_reset(self) -> None:
        """Rebuild the worker's process pools in place and drop one limp-by notch."""
        level = self.recovery_supervisor.limp_by_level
        applied = self._runtime_config.set_effective_max_threads(self._runtime_config.effective_max_threads - 1)
        logger.error(
            f"Save-our-ship soft reset #{level}: rebuilding process pools and limping by "
            f"(effective max_threads -> {applied}).",
        )
        self._action_ledger.record(
            LedgerEventType.SOFT_RESET,
            reason=f"save-our-ship soft reset #{level}",
            detail={"limp_by_level": level, "effective_max_threads": applied},
        )
        self._process_lifecycle.rebuild_inference_pool(reason=f"soft reset #{level}")
        self._process_lifecycle.rebuild_safety_pool(reason=f"soft reset #{level}")

    def give_up_on_wedged_jobs(self) -> None:
        """Fault unservable jobs and abort when no pool can recover."""
        faulted = 0
        structural_queue_wedge = self._message_dispatcher.get_deadlock_snapshot().indicates_structural_wedge()
        if self._inference_scheduler.ram_reclaim_cycle_grace_active():
            structural_queue_wedge = False
        if not self.is_inference_capacity_available() or structural_queue_wedge:
            for job in list(self._job_tracker.jobs_pending_inference):
                if job not in self._job_tracker.jobs_in_progress:
                    self._job_tracker.handle_job_fault_now(job, retryable=False)
                    faulted += 1
        if not self.is_safety_capacity_available():
            stuck_safety = list(self._job_tracker.jobs_pending_safety_check) + list(
                self._job_tracker.jobs_being_safety_checked,
            )
            for job_info in stuck_safety:
                self._job_tracker.handle_job_fault_now(job_info.sdk_api_job_info, retryable=False)
                faulted += 1
        if faulted > 0:
            if structural_queue_wedge and self.is_inference_capacity_available():
                cause = "scheduler wedged with idle processes (queue deadlock) despite a healthy pool"
            else:
                cause = "no inference capacity could be restored"
            logger.critical(
                f"Save-our-ship: gave up on {faulted} unservable job(s) ({cause}) and reported them faulted "
                "so the horde reissues them. Repeated drops like this can trigger horde-forced maintenance.",
            )

        structurally_broken = (
            self.is_inference_pool_unrecoverable()
            or self.is_safety_pool_unrecoverable()
            or self.episode_saw_unrecoverable_pool
            or not self.is_inference_capacity_available()
        )
        self._action_ledger.record(
            LedgerEventType.RECOVERY_ABANDONED,
            reason="save-our-ship: soft resets could not restore a working pool",
            detail={"jobs_faulted": faulted, "structurally_broken": structurally_broken},
        )
        if structurally_broken and not self._state.shutting_down:
            logger.critical(
                "Save-our-ship: the worker cannot restore a working process pool after repeated soft "
                "resets; abandoning ship (the last resort) rather than spinning indefinitely.",
            )
            self._abort_callback()
