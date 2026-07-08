"""Coordinate worker-level recovery watchdogs and save-our-ship escalation."""

from __future__ import annotations

import time
from collections.abc import Callable

from horde_sdk.ai_horde_api.fields import GenerationID
from loguru import logger

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import PopPauseOwner, WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger, LedgerEventType
from horde_worker_regen.process_management.ipc.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.ipc.messages import HordeControlFlag
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.lifecycle.recovery_supervisor import RecoveryAction, RecoverySupervisor
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.workload_flow import POST_PROCESS_RESERVE_FLOW


class WorkerRecoveryCoordinator:
    """Coordinate worker-level watchdogs, wedge assessment, and recovery actions."""

    ORPHAN_IN_PROGRESS_GRACE_SECONDS = 30.0
    ORPHAN_PUNT_WINDOW_SECONDS = 300.0
    ORPHAN_PUNT_WEDGE_THRESHOLD = 3
    ORPHAN_SAFETY_GRACE_SECONDS = 45.0
    SAFETY_REQUEUE_MAX = 3
    SAFETY_SOFT_PAUSE_SECONDS = 60.0
    ORPHAN_POST_PROCESS_GRACE_SECONDS = 90.0
    """Grace before a job stuck in POST_PROCESSING with no result is requeued. Wider than the safety grace
    because a legitimate multi-operation upscale pass on a large batch can run for over a minute."""
    POST_PROCESS_REQUEUE_MAX = 2
    RUNAWAY_RECOVERY_WINDOW_SECONDS = 300.0
    RUNAWAY_RECOVERY_CEILING = 20
    HEALTHY_HOLD_WATCHDOG_GRACE_SECONDS = 120.0
    """How long the soft RAM pop hold may stay engaged on a healthy, idle worker before the watchdog resets
    governance. Comfortably above the pressure-pause window and the deliberate held-queue graces so a normal
    pressure episode clears itself first."""
    HEALTHY_HOLD_ESCALATION_GRACE_SECONDS = 60.0
    """How long the hold may remain re-latched after a governance-baseline reset before the watchdog escalates
    to rebuilding the (idle) inference pool."""

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
        reserve_ledger: CommittedReserveLedger,
        bridge_data_provider: Callable[[], reGenBridgeData],
        max_inference_processes_provider: Callable[[], int],
        abort_callback: Callable[[], None],
        release_disaggregated_job: Callable[[GenerationID], None] = lambda _job_id: None,
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
            reserve_ledger: Shared committed-resource ledger used to release stranded post-processing holds.
            bridge_data_provider: Return the current live bridge data.
            max_inference_processes_provider: Return the provisioned inference-process count.
            abort_callback: Abort the worker promptly.
            release_disaggregated_job: Tell the disaggregation orchestrator to drop any state it holds for a
                job, called whenever a job leaves the tracker by a watchdog/give-up path outside the
                orchestrator's own flow. Idempotent and a no-op for a job the orchestrator does not hold, so it
                is safe to call for every released job.
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
        self._reserve_ledger = reserve_ledger
        self._bridge_data_provider = bridge_data_provider
        self._max_inference_processes_provider = max_inference_processes_provider
        self._abort_callback = abort_callback
        self._release_disaggregated_job = release_disaggregated_job
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
        self.orphan_post_process_since: dict[GenerationID, float] = {}
        self.post_process_requeue_count: dict[GenerationID, int] = {}
        # Healthy-hold watchdog episode timestamps: when the healthy-but-held condition was first observed,
        # and when a governance-baseline reset was applied for it (to time the escalation). Both None when
        # no episode is open.
        self.healthy_hold_since: float | None = None
        self.governance_reset_at: float | None = None

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

    def is_inference_pool_ready(self) -> bool:
        """Return whether the inference pool has reached an accepting state (a lane can take a job).

        The readiness signal the save-our-ship give-up clock gates on. Keyed on ``can_accept_job()`` (a lane
        in WAITING_FOR_JOB / PRELOADED_MODEL / INFERENCE_COMPLETE), the accepting state whose absence the
        deadlock detector's starting-aware guard reads as ``num_starting_processes() > 0`` ("some processes
        are starting. Waiting."). A just-rebuilt pool whose replacement children are still importing torch is
        alive (the processes exist) but not ready (no lane accepts yet), so this is False through the boot
        window and give-up is held off until the pool can actually serve the work it would otherwise fault.
        """
        return self._process_map.num_available_inference_processes() > 0

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
            # A disaggregated job pins its sampler across the encode window (no START_INFERENCE is sent, so
            # the slot reads idle while the encode service produces conditioning). The reservation is the
            # ownership record for that window, so a pinned slot referencing this job owns it and it must not
            # be punted as orphaned.
            if self._process_map.is_reserved_for_disaggregation(process_info.process_id):
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
            # The punt removed the job from the tracker; a disaggregated job the orchestrator is still holding
            # (its pinned sampler was id-reuse-replaced, so it read as unowned) must be released too, or its
            # pin/reservation/ledger leaks and a re-registration under the same id is silently dropped.
            self._release_disaggregated_job(job_id)
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
        pause_reason = f"safety could not check a result ({reason})"
        self._state.self_throttle_paused = True
        self._state.self_throttle_paused_until = until
        self._state.self_throttle_pause_owner = PopPauseOwner.SAFETY
        self._state.self_throttle_pause_reason = pause_reason
        self._action_ledger.record(
            LedgerEventType.POP_PAUSE_ARMED,
            reason=pause_reason,
            detail={
                "owner": PopPauseOwner.SAFETY.value,
                "duration_seconds": round(self.SAFETY_SOFT_PAUSE_SECONDS, 1),
            },
        )
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

    def is_post_process_lane_ready(self) -> bool:
        """Return whether the dedicated post-processing process is alive and able to accept work."""
        return any(
            process_info.process_type == HordeProcessType.POST_PROCESS and process_info.can_accept_job()
            for process_info in self._process_map.values()
        )

    async def reconcile_orphaned_post_process_jobs(self) -> None:
        """Recover jobs stranded in post-processing whose result will never return.

        Unlike a lost safety verdict, a lost post-processing result is recoverable for a bounded number of
        re-attempts because the raw inference images are still held. If those attempts are exhausted the
        worker reports a no-image fault to the horde; returning raw images would violate the post-processing
        contract the worker advertised when it accepted the job.
        """
        now = self._clock()
        being_post_processed = self._job_tracker.jobs_being_post_processed
        current_ids = {
            info.sdk_api_job_info.id_ for info in being_post_processed if info.sdk_api_job_info.id_ is not None
        }

        # Results positively dropped (their post-process launch was retired mid-job, or the lane itself was
        # torn down with the job in flight) skip the grace: the result is known lost, not merely late.
        known_lost = (
            self._message_dispatcher.take_post_process_results_known_lost()
            | self._process_lifecycle.take_post_process_results_known_lost()
        )
        for job_id in known_lost:
            if job_id in current_ids:
                self.orphan_post_process_since[job_id] = now - self.ORPHAN_POST_PROCESS_GRACE_SECONDS

        for job_id in list(self.orphan_post_process_since):
            if job_id not in current_ids:
                del self.orphan_post_process_since[job_id]
        for job_id in list(self.post_process_requeue_count):
            if job_id not in current_ids and self._job_tracker.get_stage(job_id) != JobStage.PENDING_POST_PROCESSING:
                del self.post_process_requeue_count[job_id]

        for job_info in being_post_processed:
            job_id = job_info.sdk_api_job_info.id_
            if job_id is None:
                continue
            first_seen = self.orphan_post_process_since.setdefault(job_id, now)
            if (now - first_seen) < self.ORPHAN_POST_PROCESS_GRACE_SECONDS:
                continue

            requeues = self.post_process_requeue_count.get(job_id, 0)
            if requeues >= self.POST_PROCESS_REQUEUE_MAX:
                reason = f"post-processing result lost after {requeues} requeue attempt(s)"
                logger.error(
                    f"Job {job_id} could not be post-processed (requeued {requeues} times without a "
                    "result); reporting it faulted without images so the horde reissues it.",
                )
                self._action_ledger.record(
                    LedgerEventType.POST_PROCESS_FAULTED,
                    job_id=str(job_id),
                    reason=reason,
                    detail={"stuck_seconds": round(now - first_seen, 1), "post_process_requeues": requeues},
                )
                self._job_tracker.note_post_processing_overcommit_fault()
                self._reserve_ledger.release(POST_PROCESS_RESERVE_FLOW, str(job_id))
                tracked = self._job_tracker.get_tracked_job(job_id)
                if tracked is not None and tracked.job_info is not None:
                    await self._job_tracker.fault_post_inference_job(tracked.job_info, reason=reason)
                self.orphan_post_process_since.pop(job_id, None)
                self.post_process_requeue_count.pop(job_id, None)
                continue

            if await self._job_tracker.requeue_one_being_post_processed(job_id):
                self._reserve_ledger.release(POST_PROCESS_RESERVE_FLOW, str(job_id))
                self.post_process_requeue_count[job_id] = requeues + 1
                self.orphan_post_process_since.pop(job_id, None)
                if not self.is_post_process_lane_ready():
                    self._process_lifecycle.post_process_processes_should_be_replaced = True
                logger.warning(
                    f"Job {job_id} awaited a post-processing result for {now - first_seen:.0f}s with none "
                    f"returned; requeued it for a fresh attempt "
                    f"(attempt {requeues + 1}/{self.POST_PROCESS_REQUEUE_MAX}).",
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
        self.maybe_reset_stuck_governance_hold()
        is_wedged = self.assess_wedge()
        if self.is_inference_pool_unrecoverable() or self.is_safety_pool_unrecoverable():
            self.episode_saw_unrecoverable_pool = True
        if self.episode_saw_unrecoverable_pool:
            if self.made_progress_since_episode():
                self.episode_saw_unrecoverable_pool = False
            else:
                is_wedged = True
        action = self.recovery_supervisor.evaluate(
            is_wedged=is_wedged,
            pool_ready=self.is_inference_pool_ready(),
        )
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
            self.give_up_on_wedged_jobs(terminal=self.recovery_supervisor.give_up_is_terminal)
        elif self.limp_by_active and not self.recovery_supervisor.is_in_episode:
            self.limp_by_active = False
            logger.info("Save-our-ship: pools recovered (soft-reset episode cleared).")

    def perform_soft_reset(self) -> None:
        """Rebuild the worker's process pools in place, preserving the configured concurrency.

        A soft reset rebuilds the pools to clear a transient wedge, but it deliberately does not shed a
        concurrency lane while doing so. Cutting ``effective_max_threads`` on every soft reset let a wedge,
        including one provoked by aggressive co-sampling, ratchet throughput down and outlast its cause. The
        escalation policy still *counts* this reset (a persistent wedge escalates to give-up), so preserving
        concurrency here demotes the lane cut to a warning without weakening the give-up backstop.
        """
        level = self.recovery_supervisor.limp_by_level
        effective = self._runtime_config.effective_max_threads
        logger.warning(
            f"Save-our-ship soft reset #{level}: rebuilding process pools "
            f"(concurrency preserved at effective max_threads {effective}).",
        )
        self._action_ledger.record(
            LedgerEventType.SOFT_RESET,
            reason=f"save-our-ship soft reset #{level}",
            detail={"limp_by_level": level, "effective_max_threads": effective},
        )
        self._process_lifecycle.rebuild_inference_pool(reason=f"soft reset #{level}")
        self._process_lifecycle.rebuild_safety_pool(reason=f"soft reset #{level}")
        # A soft reset rebuilds the pools, but the RAM pop hold and shed/draining bookkeeping live in worker
        # state, not the pool, so a rebuild alone leaves them latched. Return them to baseline too so the
        # reset actually clears a governance hold; the next governance tick re-arms anything still warranted.
        self._inference_scheduler.reset_governance_to_baseline(f"soft reset #{level}")

    def maybe_reset_stuck_governance_hold(self) -> None:
        """Recover a RAM pop hold that stayed engaged after the host became healthy (a governance latch).

        Belt-and-suspenders for the case the per-iteration governance tick fails to clear the soft pop hold
        once RAM recovers: the hold blocks image pops, so the inference queue drains and stays empty. This
        watchdog observes the healthy-but-held condition on an idle worker and, after a grace, resets
        governance to baseline; if that does not stick, it escalates to rebuilding the (all-idle) inference
        pool.

        Deliberately standalone rather than an ``assess_wedge`` trigger: the pool here is healthy and idle, so
        the save-our-ship soft reset's limp-by concurrency notch and unconditional pool churn would be wrong.
        The cheap governance reset is tried first; the pool rebuild is a rare second resort that only fires if
        the hold re-latches despite a healthy host.
        """
        if self._state.shutting_down or self._state.downloads_only_hold:
            self.healthy_hold_since = None
            self.governance_reset_at = None
            return

        held = (
            self._inference_scheduler.governance_healthy_but_held()
            and not self._process_map.has_inference_in_progress()
            and len(self._job_tracker.jobs_pending_inference) == 0
        )
        if not held:
            self.healthy_hold_since = None
            self.governance_reset_at = None
            return

        now = self._clock()
        if self.healthy_hold_since is None:
            self.healthy_hold_since = now
            return
        if (now - self.healthy_hold_since) < self.HEALTHY_HOLD_WATCHDOG_GRACE_SECONDS:
            return

        if self.governance_reset_at is None:
            held_seconds = now - self.healthy_hold_since
            logger.warning(
                f"Healthy-hold watchdog: the RAM pop hold has stayed engaged for {held_seconds:.0f}s on a "
                "healthy, idle worker; resetting governance to baseline.",
            )
            self._action_ledger.record(
                LedgerEventType.GOVERNANCE_RESET,
                reason="healthy-hold watchdog: pop hold latched while host healthy",
                detail={"held_seconds": round(held_seconds, 1)},
            )
            self._inference_scheduler.reset_governance_to_baseline("healthy-hold watchdog")
            self.governance_reset_at = now
            return

        if (now - self.governance_reset_at) < self.HEALTHY_HOLD_ESCALATION_GRACE_SECONDS:
            return

        logger.error(
            "Healthy-hold watchdog: the RAM pop hold re-latched after a governance reset; escalating to an "
            "inference-pool rebuild (all slots idle).",
        )
        self._action_ledger.record(
            LedgerEventType.GOVERNANCE_RESET,
            reason="healthy-hold watchdog escalation: pop hold re-latched after baseline reset",
            detail={"escalated": True},
        )
        self._process_lifecycle.rebuild_inference_pool(reason="healthy-hold watchdog escalation")
        self._inference_scheduler.reset_governance_to_baseline("healthy-hold watchdog escalation")
        self.healthy_hold_since = None
        self.governance_reset_at = None

    def give_up_on_wedged_jobs(self, *, terminal: bool = False) -> None:
        """Fault unservable jobs and abort when no pool can recover.

        Args:
            terminal: Whether the supervisor flagged this give-up as the deliberate abandon-ship escalation
                (a wedge that outlived a fresh soft-reset cycle). A terminal give-up aborts even when the
                pool momentarily looks recoverable, so a persistent wedge over a live-but-idle pool cannot
                spin forever.
        """
        faulted = 0
        structural_queue_wedge = self._message_dispatcher.get_deadlock_snapshot().indicates_structural_wedge()
        if self._inference_scheduler.ram_reclaim_cycle_grace_active():
            structural_queue_wedge = False
        if not self.is_inference_capacity_available() or structural_queue_wedge:
            for job in list(self._job_tracker.jobs_pending_inference):
                if job not in self._job_tracker.jobs_in_progress:
                    self._job_tracker.handle_job_fault_now(job, retryable=False)
                    # Release any disaggregation state so a give-up never strands a held pin/ledger entry.
                    if job.id_ is not None:
                        self._release_disaggregated_job(job.id_)
                    faulted += 1
        if not self.is_safety_capacity_available():
            stuck_safety = list(self._job_tracker.jobs_pending_safety_check) + list(
                self._job_tracker.jobs_being_safety_checked,
            )
            for job_info in stuck_safety:
                self._job_tracker.handle_job_fault_now(job_info.sdk_api_job_info, retryable=False)
                if job_info.sdk_api_job_info.id_ is not None:
                    self._release_disaggregated_job(job_info.sdk_api_job_info.id_)
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
        should_abort = (structurally_broken or terminal) and not self._state.shutting_down
        # Record only when the give-up actually did something: it faulted at least one job, or it is a
        # terminal abort decision. A no-op tick (nothing pending, pool not structurally broken, not terminal)
        # leaves no ledger entry, so a latched give-up cannot spam RECOVERY_ABANDONED with jobs_faulted=0.
        if faulted > 0 or should_abort:
            self._action_ledger.record(
                LedgerEventType.RECOVERY_ABANDONED,
                reason="save-our-ship: soft resets could not restore a working pool",
                detail={"jobs_faulted": faulted, "structurally_broken": structurally_broken, "terminal": terminal},
            )
        if should_abort:
            logger.critical(
                "Save-our-ship: the worker cannot restore a working process pool after repeated soft "
                "resets; abandoning ship (the last resort) rather than spinning indefinitely.",
            )
            self._abort_callback()
