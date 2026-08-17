"""Coordinate worker-level recovery watchdogs and save-our-ship escalation."""

from __future__ import annotations

import enum
import time
from collections.abc import Callable

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.fields import GenerationID
from loguru import logger

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import (
    PopPauseOwner,
    RecoveryParkReason,
    WorkerState,
)
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger, LedgerEventType
from horde_worker_regen.process_management.ipc.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.jobs.job_tracker import JobFaultOrigin, JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.lifecycle.recovery_supervisor import RecoveryAction, RecoverySupervisor
from horde_worker_regen.process_management.resources.reclaim_ladder import (
    LANE_PAUSE_RUNG_KINDS,
    ReclaimRung,
    ReclaimRungKind,
    build_reclaim_ladder,
    execute_reclaim_rung,
    restore_reclaim_rung,
)
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.workload_flow import POST_PROCESS_RESERVE_FLOW


class RecoveryDisposition(enum.Enum):
    """How the worker resolved a request for terminal process recovery.

    The coordinator consumes this result directly instead of inferring whether an untyped abort callback
    changed shared shutdown state. The same value is retained by the process manager as the session's recovery
    outcome, so persistence and the outer entry point agree on whether a failed exit is required.
    """

    CONTINUE_IN_PROCESS = enum.auto()
    """No reliable relaunch contract exists; keep the current process alive and quiesce when remedies expire."""

    RESTART_PROCESS = enum.auto()
    """A relauncher is available; finish bounded cleanup and leave the process with failure status."""


_MIN_CONSTRUCTIVE_RUNG_MB = 1.0
"""Smallest promised free that makes a reclaim rung worth issuing as a constructive remedy.

A rung is priced from its tenant's measured give-back (a resident model's footprint, a process's reclaimable
reservation, a lane's context charge plus reservations), so one that promises essentially nothing has nothing
to return, and issuing it waits out a settling window against an unchanged resource condition. Lane rungs
always carry at least their stopped process's CUDA-context charge, so in practice this skips unmeasured or
empty tenants, never a lane with a context to give back."""


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
    PP_RECLAIM_REMEDY_GRACE_SECONDS = 90.0
    """How long a give-up yields to an in-flight post-processing lane reclaim before faulting the backlog.

    A structural queue wedge over a healthy pool can be a head that simply does not fit beside the
    post-processing lane's resident module weights: the remedy is to unload the idle lane's modules so the
    head's room frees, not to fault a servable backlog. When the give-up finds that remedy reachable it issues
    the unload and yields for this window so the freed VRAM can materialise and the head re-admit. The window
    bounds the yield: a reclaim that never lands (the lane wedged, or the freed room still insufficient) falls
    through to the ordinary give-up so the horde reissues the jobs rather than the worker parking forever.
    Sized to the post-processing admission-patience window."""

    RECLAIM_RUNG_GRACE_SECONDS = 45.0
    """How long one issued reclaim rung is given to clear the wedge before the next rung is consumed.

    A rung's effect is not instant: a lane teardown returns its context only once the OS has reaped the
    process, and the process that could not obtain memory then has to start and drain its backlog. This is the
    settling window the wedge oracle waits out before grading the rung. Sized above an observed child boot so a
    rung that did work is not graded a failure while the capacity it freed is still coming up, and bounded so a
    rung that did nothing does not stall the escalation."""

    RECLAIM_RUNG_ALLOTMENT = 3
    """Reclaim rungs issuable before the escalation proceeds to a pool rebuild; renewed once it has.

    Budgeting the rungs by count rather than by elapsed time is what keeps the rest of the escalation
    reachable: a rung is consumed whether or not it appeared to help, so the allotment cannot be replenished by
    a wedge that simply persists. The allotment is renewed after a pool rebuild so the cheaper remedies below
    the rebuild stay reachable, and the total stays finite because the cursor into the frozen candidate list
    only ever advances."""

    RECLAIM_REMEDY_EPISODE_BUDGET_SECONDS = 300.0
    """Aggregate time the reclaim remedies of one wedge episode may occupy, measured from the first rung.

    An independent bound on the counted allotment: even a run of rungs that each look worth waiting out cannot
    hold accepted work past this, so the give-up that reissues those jobs to the horde stays reachable on a
    condition no rung fixes."""

    RECLAIM_NO_EFFECT_LIMIT = 2
    """Consecutive rungs that may leave the named head blocker untouched before the ladder is set aside.

    Every rung on the ladder frees card memory, which addresses the head only when a shortfall is what holds
    it. When the head's blocker is published and neither it nor the inference-start count moves across a
    rung's full settling window, that rung demonstrably did not address the block. Two such rungs in a row is
    the ladder answering a constraint it has no purchase on, and continuing to issue rungs against it both
    churns resident state and renews the "a remedy remains" excuse that holds the give-up backstop off. One
    is deliberately not enough: a single window can coincide with an unrelated hold lifting."""

    MAX_GIVE_UP_YIELDS_PER_EPISODE = 3
    """Give-ups one wedge episode may refund to an in-flight remedy before it must actually fire.

    A refunded give-up restores the escalation cycle it would have consumed, so an unbounded number of them
    decays the bounded yield into an indefinite park with the safety valve never having fired. Capping the
    refunds means a remedy that keeps looking reachable (a card that keeps regenerating candidates) can defer
    the terminal escalation only a bounded number of times."""

    RECOVERY_PARK_REPROBE_SECONDS = 600.0
    """How long a recovery park stays quiescent before one fresh escalation attempt is permitted.

    A park is entered only when the escalation has nothing left to try, but what dooms a pool is often external
    to the worker (a co-tenant process holding the card's VRAM, an exhausted disk) and can clear with no action
    from the worker at all. Waiting this long between attempts keeps the worker available for that recovery
    while bounding what a permanently doomed pool can cost to one escalation cycle per interval instead of
    continuous process churn and job faulting."""

    POP_GATE_HELD_WEDGE_SECONDS = 900.0
    """How long job pops may be held at one gate with nothing completing before it counts as a wedge.

    The failure-independent backstop. Every other wedge signal names a condition, and a condition nobody
    modelled holds the intake path just as effectively; what no gate-holding failure can produce on its own is
    a completed job, so the absence of one over a long continuous hold is evidence that stands whatever holds
    the gate. Sized above every in-place remedy window the worker gives a condition to clear itself (the
    deferred-start no-progress window and the recovery park's re-probe interval are the widest at 600s), so
    the owning watchdog always acts first and this only sees what they left unresolved. It is far above
    ordinary backpressure: a worker whose local queue is full holds a gate continuously while serving at full
    rate, and completed work excuses that hold whatever its length. It is far below the horizon over which a
    silently gated worker is worth anything to the horde, so the escalation opens in minutes rather than
    hours."""

    HEAD_RECOVERY_BUDGET_FALLBACK_SECONDS = 150.0
    """Give-up deferral budget for head-of-queue model materialisation when no numeric preload timeout is set.

    The deferral normally bounds itself by ``bridge_data.preload_timeout``; this is the backstop bound used
    only if that value is not a real number, so a materialisation that never completes can never defer give-up
    forever. Matches the ``preload_timeout`` default."""

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
        terminal_recovery_callback: Callable[[], RecoveryDisposition],
        release_disaggregated_job: Callable[[GenerationID], None] = lambda _job_id: None,
        unbound_disaggregated_job_ids: Callable[[], set[str]] = set,
        head_aux_prefetch_in_flight: Callable[[], bool] = lambda: False,
        head_block_reason: Callable[[], str | None] = lambda: None,
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
            terminal_recovery_callback: Request a fresh worker process and report whether a relaunch contract
                accepted the request or the current process must continue recovering in place.
            release_disaggregated_job: Tell the disaggregation orchestrator to drop any state it holds for a
                job, called whenever a job leaves the tracker by a watchdog/give-up path outside the
                orchestrator's own flow. Idempotent and a no-op for a job the orchestrator does not hold, so it
                is safe to call for every released job.
            unbound_disaggregated_job_ids: Report the ids of disaggregated jobs staged ahead of a sampler pin,
                which own no inference slot by design until that pin releases. The orphaned-in-progress
                watchdog treats them as owned: they are bounded by the orchestrator's own stage patience, and
                punting them would undo the staging on every streak. Defaults to none for an unwired caller.
            head_aux_prefetch_in_flight: Report whether the head-of-queue job is auxiliary-gated with a
                prefetch still in flight (its per-job deadline exists and has not expired). Save-our-ship
                give-up defers to it exactly as it defers to a head whose model is materialising: a head
                waiting on an in-flight auxiliary download is capacity in flight, not a wedge. It is
                self-bounding by that deadline, so a stalled prefetch stops deferring once the deadline lapses
                (the coordinator faults the job then). Defaults to "never in flight" for an unwired caller.
            head_block_reason: Report the constraint the scheduler currently names as holding the head of
                queue, or None when it names none. The constructive remedy ladder judges its own relevance
                against this: a rung that leaves the named constraint unchanged did not address it. An
                unwired caller reports None, which leaves the ladder's budgets the only bound on it.
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
        self._terminal_recovery_callback = terminal_recovery_callback
        self._release_disaggregated_job = release_disaggregated_job
        self._unbound_disaggregated_job_ids = unbound_disaggregated_job_ids
        self._head_aux_prefetch_in_flight = head_aux_prefetch_in_flight
        self._head_block_reason = head_block_reason
        self._clock = clock

        self.recovery_supervisor = recovery_supervisor or RecoverySupervisor()
        self.limp_by_active = False
        self.episode_saw_unrecoverable_pool = False
        self.episode_progress_baseline: int | None = None
        self.episode_inference_start_baseline: int | None = None
        self.episode_post_processing_progress_baseline: int | None = None
        self.episode_frontier_baseline: tuple[GenerationID, JobStage] | None = None
        self.unrelated_progress_deferral_spent = False
        self.recovery_event_times: list[float] = []
        self.last_seen_recovery_count = 0
        # Edge latch so a runaway-recovery abort the worker declined is recorded once rather than every tick
        # for as long as the rolling window stays over the ceiling.
        self.runaway_abort_refusal_logged = False
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
        # When the give-up last issued a post-processing lane reclaim for a wedged-but-servable head; None when
        # no such remedy is in flight. Cleared when the wedge episode closes so a later episode gets its own
        # remedy attempt.
        self.pp_reclaim_remedy_issued_at: float | None = None
        # The reclaim rungs this wedge episode may issue, frozen at the first look so a flapping card cannot
        # regenerate candidates indefinitely, plus the monotonic cursor into them. None until first frozen.
        self.reclaim_rungs: tuple[ReclaimRung, ...] | None = None
        self.reclaim_cursor = 0
        # Rungs issued against the current allotment, when the most recent one was issued (its settling window),
        # and when the episode's first rung was issued (the aggregate time bound's origin).
        self.reclaim_rungs_issued_in_allotment = 0
        self.reclaim_rung_issued_at: float | None = None
        self.reclaim_remedy_started_at: float | None = None
        # Relevance accounting for the constructive ladder: the named head blocker and inference-start count
        # as they stood when the most recent rung was issued (None once that rung's effect has been judged),
        # how many rungs in a row have left both unchanged, and the episode latch that sets the ladder aside
        # once that run reaches its limit.
        self.reclaim_effect_baseline: tuple[str, int] | None = None
        self.reclaim_no_effect_cycles = 0
        self.reclaim_ladder_ruled_irrelevant = False
        # Give-ups refunded to the constructive rung's settling window. The PP-specific remedy uses its own
        # independently bounded clock and does not consume or inflate this count.
        self.give_up_yields_spent = 0
        # Lane pauses this episode's rungs actually acquired, in issue order, so the episode close restores
        # exactly those and in reverse. A pause that was a no-op (another owner already held it) is not recorded,
        # so this coordinator never lifts a hold it does not own.
        self.reclaim_paused_lanes: list[ReclaimRung] = []
        # When head-of-queue model materialisation (a preload/load for the head over an otherwise idle pool)
        # was first observed in the current continuity; None when the head is not materialising. Bounds the
        # give-up deferral so a stuck load still escalates. Cleared when materialisation stops or the episode
        # closes.
        self.head_recovery_in_flight_since: float | None = None
        # The pop-gate hold currently being judged: the ``last_pop_gate_since`` stamp it is keyed on, and the
        # completed-job count when that hold was first observed. None while no gate holds pops. Keying on the
        # stamp means a new hold gets its own baseline without any explicit reset.
        self.pop_gate_hold_baseline: tuple[float, int] | None = None
        # Edge latch so one held-gate wedge is disclosed once per hold rather than on every assessment tick.
        self.pop_gate_wedge_disclosed_since: float | None = None

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
        if self._inference_starts_backing_off():
            return True
        return any(
            process_info.process_type == HordeProcessType.INFERENCE and process_info.is_process_alive()
            for process_info in self._process_map.values()
        )

    def _inference_starts_backing_off(self) -> bool:
        """Return whether inference starts are intentionally deferred by lifecycle headroom admission."""
        return (
            self._process_lifecycle.has_pending_inference_starts()
            and self._process_lifecycle.pending_gpu_starts_backing_off()
        )

    def _safety_starts_backing_off(self) -> bool:
        """Return whether safety starts are intentionally deferred by lifecycle headroom admission."""
        return (
            self._process_lifecycle.has_pending_safety_starts()
            and self._process_lifecycle.pending_gpu_starts_backing_off()
        )

    def is_safety_capacity_available(self) -> bool:
        """Return whether any safety process is alive to serve pending safety checks."""
        if self._safety_starts_backing_off():
            return True
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
        if self._inference_starts_backing_off():
            return False
        return len(self._process_lifecycle.quarantined_inference_slots) >= self.max_inference_processes

    def is_safety_pool_unrecoverable(self) -> bool:
        """Return whether the safety pool cannot be restored and is not currently ready.

        Two independent signals qualify: the sliding-window crash-loop count, and the consecutive
        start-failure streak. The streak is what covers a deterministic failure whose full cold start spaces
        its rebuilds wider than the window can accumulate, so a pool that never initialises is classified
        however slowly it fails rather than being respawned for the life of the worker.

        Both signals count rebuilds, and a rebuild the parent asked for (a whole-card pause/restore cycle, a
        supervised rebuild) counts the same as one a failing child forced. This verdict drops a job's images
        and faults it, so it additionally requires a safety child that failed on its own account: without one
        the rebuilds are placement churn, and the jobs the outgoing launch was checking are still owed their
        verdicts.
        """
        if self._safety_starts_backing_off():
            return False
        if not self._process_lifecycle.safety_pool_failure_evidence_seen:
            return False
        pool_broken = self._process_lifecycle.safety_pool_failing or self._process_lifecycle.safety_pool_start_failing
        return pool_broken and not self.is_safety_pool_ready()

    def inference_slot_owns_job(self, job_id: GenerationID) -> bool:
        """Return whether some live inference slot owns the given job."""
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if not process_info.is_process_alive():
                continue
            owned_job = process_info.current_inference_job()
            if owned_job is None or owned_job.id_ != job_id:
                continue
            return True
        return False

    def reconcile_orphaned_in_progress_jobs(self) -> None:
        """Punt jobs stuck in inference with no owning live slot."""
        now = self._clock()
        in_progress = self._job_tracker.jobs_in_progress
        # A job staged ahead of a sampler pin is in progress with no slot by design: its encode is running on
        # the component lane and it binds a sampler when the pin releases. The orchestrator's stage patience
        # bounds that wait, so it is owned for this watchdog's purposes rather than orphaned.
        staged_ahead_ids = self._unbound_disaggregated_job_ids()
        live_ids = {
            job.id_
            for job in in_progress
            if job.id_ is not None and (self.inference_slot_owns_job(job.id_) or str(job.id_) in staged_ahead_ids)
        }

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
                    fault_reason=f"safety check unrecoverable ({reason})",
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

    def structural_queue_wedge_active(self) -> bool:
        """Return whether a structural queue deadlock is real rather than a deliberately held queue.

        The dispatcher reports the raw deadlock shape (pending inference work while every process is idle).
        Several scheduler states hold the queue on purpose while capacity lands, and inference actually
        running disproves the all-idle premise outright, so neither is a wedge.

        Every consumer reads this one verdict. The wedge assessment and the give-up that acts on it must not
        diverge: a give-up applying a narrower set of excuses would fault exactly the backlog the scheduler
        is holding for capacity that is about to arrive.
        """
        if not self._message_dispatcher.get_deadlock_snapshot().indicates_structural_wedge():
            return False
        queue_held_for_capacity = (
            self._inference_scheduler.whole_card_residency_grace_active()
            or self._inference_scheduler.whole_card_governor_defer_active()
            or self._inference_scheduler.heavy_head_load_grace_active()
            or self._inference_scheduler.ram_reclaim_cycle_grace_active()
            or self._inference_starts_backing_off()
        )
        if queue_held_for_capacity:
            return False
        return not self._process_map.has_inference_in_progress()

    def pop_gate_wedge_active(self) -> bool:
        """Return whether job pops have been gated far too long with no work completing behind the gate.

        The pop coroutine returns early at any of a dozen preconditions, each owned by a different watchdog,
        and a condition none of them models holds intake exactly as effectively. This keys on the hold itself
        rather than on the condition, so it covers the gates nobody anticipated, and it takes its liveness
        proof from completed jobs, which no gate-holding failure can produce on its own.

        Three facts must hold together: the same gate has held pops for longer than
        ``POP_GATE_HELD_WEDGE_SECONDS``, no pop attempt has reached the horde in that span, and no job has
        completed since the hold was first observed. The last two are what keep ordinary backpressure out: a
        worker whose queue is full holds a gate continuously and attempts no pops while doing so, and its
        completions are the proof that the hold is capacity management rather than a wedge.
        """
        gate = self._state.last_pop_gate
        if gate is None:
            self.pop_gate_hold_baseline = None
            self.pop_gate_wedge_disclosed_since = None
            return False

        held_since = self._state.last_pop_gate_since
        if self.pop_gate_hold_baseline is None or self.pop_gate_hold_baseline[0] != held_since:
            self.pop_gate_hold_baseline = (held_since, self._job_tracker.total_num_completed_jobs)

        now = self._clock()
        if (now - held_since) < self.POP_GATE_HELD_WEDGE_SECONDS:
            return False
        if self._job_tracker.total_num_completed_jobs > self.pop_gate_hold_baseline[1]:
            return False
        if (now - self._state.last_pop_attempt_completed_at) < self.POP_GATE_HELD_WEDGE_SECONDS:
            return False

        if self.pop_gate_wedge_disclosed_since != held_since:
            self.pop_gate_wedge_disclosed_since = held_since
            logger.critical(
                f"Job pops have been held at gate '{gate}' for {now - held_since:.0f}s with no job completed "
                "and no pop attempt reaching the horde in that time; the worker is serving nothing, so "
                "recovery is escalating over the hold regardless of what is holding it.",
            )
        return True

    def assess_wedge(self) -> bool:
        """Return whether the worker structurally cannot make progress."""
        if self._state.shutting_down:
            return False
        if self._state.downloads_only_hold:
            return False
        if self._state.recovery_parked:
            # A park is the state after the remedies ran out: reporting the wedge again would only drive the
            # same exhausted actions. The park's re-probe is what re-opens the assessment.
            return False
        return (
            self.is_inference_pool_unrecoverable()
            or self.is_safety_pool_unrecoverable()
            or self.structural_queue_wedge_active()
            or self.orphan_wedge_active()
            or self.pop_gate_wedge_active()
        )

    def _capture_progress_baseline(self) -> None:
        """Snapshot the progress counters as the reference for :meth:`made_progress_since_episode`.

        Taken when a wedge episode opens and re-taken on every soft reset, so ``made_progress`` measures
        forward motion since the most recent recovery attempt rather than since the episode began.
        """
        self.episode_progress_baseline = self._job_tracker.total_num_completed_jobs
        self.episode_inference_start_baseline = self._job_tracker.total_num_inference_starts
        self.episode_post_processing_progress_baseline = self._job_tracker.total_num_post_processing_progress
        self.episode_frontier_baseline = self._current_recovery_frontier()

    def _current_recovery_frontier(self) -> tuple[GenerationID, JobStage] | None:
        """Return the oldest nonterminal accepted job and its current stage.

        Recovery for accepted work belongs to the oldest unresolved frontier, not to worker-wide counters a
        follower can advance. A successful terminal head leaves this view, and a head entering a later stage
        changes it. Fault and retry transitions are filtered by :meth:`_frontier_made_progress`; a recovery
        action cannot manufacture its own proof by faulting or requeueing the head.
        """
        active = [
            tracked for tracked in self._job_tracker.tracked_jobs() if tracked.stage is not JobStage.PENDING_SUBMIT
        ]
        if not active:
            return None
        head = min(active, key=lambda tracked: tracked.pop_order)
        return head.job_id, head.stage

    def made_progress_since_episode(self) -> bool:
        """Return whether accepted work moved forward since the most recent recovery baseline.

        The baseline is captured when the episode opens and re-captured on each soft reset, so this reports
        progress since the latest soft reset once one has been attempted.

        When accepted work has a frontier, only movement of that same oldest unresolved job counts. Aggregate
        counters remain a fallback for pool-level episodes with no accepted frontier. They are also observed
        separately to permit one bounded delay, but follower throughput never closes or resets the blocked
        head's escalation.
        """
        if self.episode_frontier_baseline is not None:
            return self._frontier_made_progress()

        return self._worker_progress_since_episode()

    def _frontier_made_progress(self) -> bool:
        """Return whether the baseline head advanced rather than faulting or retrying."""
        baseline = self.episode_frontier_baseline
        if baseline is None:
            return False
        baseline_job_id, baseline_stage = baseline
        current = self._current_recovery_frontier()
        if current == baseline:
            return False

        tracked = self._job_tracker.get_tracked_job(baseline_job_id)
        if tracked is None:
            return False
        if tracked.job_info is not None and tracked.job_info.state is GENERATION_STATE.faulted:
            return False
        if current is None or current[0] != baseline_job_id:
            return tracked.stage is JobStage.PENDING_SUBMIT
        return current[1].value > baseline_stage.value

    def _worker_progress_since_episode(self) -> bool:
        """Return whether aggregate completion or stage counters advanced since the episode baseline."""
        if (
            self.episode_progress_baseline is None
            or self.episode_inference_start_baseline is None
            or self.episode_post_processing_progress_baseline is None
        ):
            return False
        if self._job_tracker.total_num_post_processing_progress > self.episode_post_processing_progress_baseline:
            return True
        if self._job_tracker.jobs_pending_post_processing or self._job_tracker.jobs_being_post_processed:
            # A downstream post-processing drain stall is not disproved by starting more upstream inference.
            return False
        if self._job_tracker.total_num_completed_jobs > self.episode_progress_baseline:
            # A completion is end-to-end proof: the job cleared every downstream stage, safety included.
            return True
        if self._job_tracker.jobs_pending_safety_check or self._job_tracker.jobs_being_safety_checked:
            # A safety-stage drain stall is not disproved by more upstream inference starting either. Generated
            # work parks after the sampler while starts keep rising behind it, so crediting starts would let the
            # stalled stage manufacture its own proof of recovery and reset the escalation it should be climbing.
            return False
        return self._job_tracker.total_num_inference_starts > self.episode_inference_start_baseline

    def _unrelated_progress_deferral_available(self) -> bool:
        """Return whether follower throughput earns the episode's one observation delay."""
        return (
            not self.unrelated_progress_deferral_spent
            and self.episode_frontier_baseline is not None
            and self._current_recovery_frontier() == self.episode_frontier_baseline
            and self._worker_progress_since_episode()
        )

    def enter_recovery_park(self, reason: RecoveryParkReason, detail: dict[str, str | int | float | bool]) -> None:
        """Hold escalation quiescent because its remedies are spent and the exit rung was withheld.

        The worker stays up but stops popping work and stops rebuilding pools, so a condition no remaining
        remedy can fix costs a bounded amount instead of unbounded process churn and job faulting. Idempotent:
        a park already engaged is neither re-logged nor re-recorded.

        Args:
            reason: Which exhausted escalation is parking, recorded on worker state and in the ledger.
            detail: Measurements describing the exhausted condition, attached to the ledger record.
        """
        if self._state.recovery_parked:
            return
        self._state.recovery_parked = True
        self._state.recovery_park_reason = reason
        self._state.recovery_park_since = self._clock()
        logger.critical(
            f"Save-our-ship: no remedy remains ({reason.value}) and nothing would bring a fresh process back, "
            "so recovery is parked: job popping and pool rebuilds stop while the worker stays up. The "
            f"escalation is re-attempted in {self.RECOVERY_PARK_REPROBE_SECONDS:.0f}s, so a condition that "
            "clears on its own (a co-tenant process freeing the card, disk space returning) restores service "
            "without an operator. Attach a supervisor or set exit_on_unhandled_faults to have the worker "
            "restart itself instead.",
        )
        self._action_ledger.record(
            LedgerEventType.RECOVERY_PARKED,
            reason=f"save-our-ship: recovery parked ({reason.value})",
            detail={**detail, "reprobe_seconds": self.RECOVERY_PARK_REPROBE_SECONDS},
        )

    def recovery_park_reprobe_due(self) -> bool:
        """Return whether the park has held long enough to permit one fresh escalation attempt."""
        if not self._state.recovery_parked:
            return False
        return (self._clock() - self._state.recovery_park_since) >= self.RECOVERY_PARK_REPROBE_SECONDS

    def leave_recovery_park(self) -> None:
        """Lift the park and re-arm the escalation so the next tick makes one fresh attempt.

        The rolling recovery window, the runaway refusal latch, and the supervisor's episode are all returned
        to baseline: they hold the verdict that the previous attempt was exhausted, and an attempt starting
        from that verdict would neither rebuild nor escalate while the worker resumed accepting work.
        """
        if not self._state.recovery_parked:
            return
        parked_seconds = self._clock() - self._state.recovery_park_since
        reason = self._state.recovery_park_reason
        self._state.recovery_parked = False
        self._state.recovery_park_reason = None
        self._state.recovery_park_since = 0.0
        self.recovery_event_times.clear()
        self.last_seen_recovery_count = self._process_lifecycle._num_process_recoveries
        self.runaway_abort_refusal_logged = False
        self.recovery_supervisor.reset_episode()
        self.limp_by_active = False
        self._clear_recovery_episode_accounting()
        # The healthy-hold watchdog is another bounded recovery episode owned by this coordinator. A stale
        # timestamp would let the first post-park tick skip its grace and rebuild immediately.
        self.healthy_hold_since = None
        self.governance_reset_at = None
        reason_value = reason.value if reason is not None else "unknown"
        logger.warning(
            f"Save-our-ship: lifting the recovery park after {parked_seconds:.0f}s ({reason_value}); resuming "
            "job popping and permitting one fresh escalation attempt. If the condition has not cleared the "
            "worker parks again rather than churning.",
        )

    def _clear_recovery_episode_accounting(self) -> None:
        """Return every SOS episode-owned latch, baseline, lane receipt, and remedy budget to baseline."""
        self.episode_progress_baseline = None
        self.episode_inference_start_baseline = None
        self.episode_post_processing_progress_baseline = None
        self.episode_frontier_baseline = None
        self.unrelated_progress_deferral_spent = False
        self.episode_saw_unrecoverable_pool = False
        self.pp_reclaim_remedy_issued_at = None
        self.head_recovery_in_flight_since = None
        # A park may begin while a constructive rung owns a lane pause. Restore exactly those owned pauses
        # before accepting work again, then discard the frozen ladder so the fresh episode re-assesses reality.
        self.restore_reclaimed_lanes()
        self._reset_constructive_remedy_budget()

    def maybe_abort_on_runaway_recoveries(self) -> bool:
        """Abort if process recoveries are flapping faster than the rolling-window ceiling.

        Returns:
            Whether the abort actually took, so the caller stops driving escalation this tick. An abort the
            worker declined (nothing is watching to relaunch it, and the operator did not opt into exiting)
            returns False and parks recovery instead: rebuilding at this rate has demonstrably not stabilised
            the pool, so continuing to drive the ladder would only churn children with no remedy left to reach.
        """
        if self._state.recovery_parked:
            return False
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
            self.runaway_abort_refusal_logged = False
            return False
        if self.runaway_abort_refusal_logged:
            # A refused abort leaves the ceiling breached every tick. The verdict is already recorded, so
            # report "not aborted" quietly and let the caller keep driving the rungs that remain.
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
        disposition = self._terminal_recovery_callback()
        if disposition is RecoveryDisposition.RESTART_PROCESS:
            return True
        # No relaunch contract exists, and the in-place rung (rebuilding) is what produced this recovery rate
        # in the first place. Park instead of driving it again.
        self.runaway_abort_refusal_logged = True
        self.enter_recovery_park(
            RecoveryParkReason.RUNAWAY_RECOVERIES,
            {
                "recoveries_in_window": len(self.recovery_event_times),
                "window_seconds": self.RUNAWAY_RECOVERY_WINDOW_SECONDS,
            },
        )
        return False

    def _head_recovery_in_flight(self) -> bool:
        """Return whether the head-of-queue job's capacity is in flight, so give-up defers to it.

        True while the head's model is still materialising within its preload budget, or while the head is an
        auxiliary-gated job whose pop-time prefetch is still in flight (bounded by that prefetch deadline).
        Either case is capacity in flight over a ready lane, not a wedge over a healthy pool, so give-up must
        not fault the head. Each input is self-bounding (the preload budget; the prefetch deadline), so a load
        or a download that never lands stops deferring and the wedge escalates exactly as before.
        """
        return self._model_materialization_in_flight() or self._head_aux_prefetch_in_flight()

    def _model_materialization_in_flight(self) -> bool:
        """Return whether the head's model is materialising within its preload budget (give-up defers to it).

        A save-our-ship give-up over a lane that reads ready must not fault the head-of-queue job while the
        pool is in the middle of loading that job's model: a ready lane whose head model is still materialising
        is capacity in flight, not a wedge over a healthy pool. This reports the deferral while the scheduler
        sees the head materialising, bounded by ``bridge_data.preload_timeout`` so a load that never lands (a
        stuck preload) stops deferring and the wedge escalates exactly as before.
        """
        if not self._inference_scheduler.head_model_materializing():
            self.head_recovery_in_flight_since = None
            return False
        now = self._clock()
        if self.head_recovery_in_flight_since is None:
            self.head_recovery_in_flight_since = now
        budget = self.bridge_data.preload_timeout
        budget_seconds = (
            float(budget) if isinstance(budget, int | float) else self.HEAD_RECOVERY_BUDGET_FALLBACK_SECONDS
        )
        return (now - self.head_recovery_in_flight_since) < budget_seconds

    def run_recovery_supervisor(self) -> None:
        """Drive save-our-ship escalation one tick and perform any returned action.

        A parked worker performs no escalation at all until its re-probe interval elapses, at which point the
        park lifts and this tick becomes the fresh attempt.
        """
        if self._state.shutting_down:
            return
        if self._state.recovery_parked:
            if not self.recovery_park_reprobe_due():
                return
            self.leave_recovery_park()
        if self.maybe_abort_on_runaway_recoveries():
            return
        if self._state.recovery_parked:
            # The runaway backstop parked recovery on this very tick (its abort was withheld); there is nothing
            # further to drive.
            return
        self.maybe_reset_stuck_governance_hold()
        is_wedged = self.assess_wedge()
        made_progress = self.made_progress_since_episode()
        head_recovery_in_flight = self._head_recovery_in_flight()
        # A replacement child that is alive and still booting is capacity in flight: give-up must not fault the
        # jobs the finishing boot will serve. The liveness-aware count excludes a child that died mid-boot (it
        # still reports PROCESS_STARTING until reaped), so a dead boot does not hold the give-up backstop.
        boot_in_progress = self._process_map.num_starting_processes_alive() > 0
        if self.is_inference_pool_unrecoverable() or self.is_safety_pool_unrecoverable():
            self.episode_saw_unrecoverable_pool = True
        if self.episode_saw_unrecoverable_pool:
            if made_progress:
                self.episode_saw_unrecoverable_pool = False
            else:
                is_wedged = True
        # Judged before the availability question is asked, so a ladder that has just been measured as having
        # no bearing on the head's blocker is not offered as a reason to defer this tick's escalation.
        self.settle_reclaim_relevance()
        # Consulted only while the worker actually reads as wedged: the candidate snapshot walks the process map,
        # and a healthy worker has no episode for a rung to belong to.
        constructive_remedy_available = is_wedged and self.constructive_remedy_available()
        action = self.recovery_supervisor.evaluate(
            is_wedged=is_wedged,
            pool_ready=self.is_inference_pool_ready(),
            made_progress=made_progress,
            head_recovery_in_flight=head_recovery_in_flight,
            boot_in_progress=boot_in_progress,
            constructive_remedy_available=constructive_remedy_available,
            unrelated_progress_deferral_available=self._unrelated_progress_deferral_available(),
        )
        if self.recovery_supervisor.is_in_episode:
            if self.episode_progress_baseline is None:
                self._capture_progress_baseline()
        else:
            self._clear_recovery_episode_accounting()
        if action is RecoveryAction.RECLAIM:
            self.issue_next_constructive_remedy()
        elif action is RecoveryAction.OBSERVE:
            self.unrelated_progress_deferral_spent = True
            logger.info(
                "Save-our-ship: unrelated work advanced while the recovery frontier remained unchanged; "
                "waiting one bounded action interval before continuing escalation."
            )
        elif action is RecoveryAction.SOFT_RESET:
            self.perform_soft_reset()
            # Re-anchor the progress baseline to the reset: the episode may close only when work moves forward
            # from here, so a rebuild that never serves cannot look like a recovery and reset the escalation.
            self._capture_progress_baseline()
            self.limp_by_active = True
            # The rebuild has happened, so the rungs the earlier allotment did not reach become reachable again.
            self.renew_reclaim_rung_allotment()
        elif action is RecoveryAction.GIVE_UP:
            if self._give_up_yields_to_remedy():
                # The wedge may be curable by a remedy that has not had time to land: defer this give-up (the
                # supervisor refunds the cycle, so the eventual real give-up escalates undiminished).
                self.recovery_supervisor.yield_give_up()
                if self.pp_reclaim_remedy_issued_at is None:
                    self.give_up_yields_spent += 1
            else:
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

    def constructive_remedy_available(self) -> bool:
        """Return whether a reclaim rung remains issuable, or the last one issued is still settling.

        The fact the escalation policy reads to decide that a constructive resource remedy outranks rebuilding
        the pools or faulting accepted work. It is True while either a rung the frozen candidate list still
        holds is within the current allotment, or a rung already issued is inside its settling window (so the
        escalation waits for the wedge oracle instead of climbing past a remedy that may be about to work).

        The bounds that make this eventually False, and so keep the rest of the escalation reachable, are the
        counted allotment (:attr:`RECLAIM_RUNG_ALLOTMENT`), the monotonic cursor into a candidate list frozen
        once per episode, and the aggregate time bound
        (:attr:`RECLAIM_REMEDY_EPISODE_BUDGET_SECONDS`), which overrides a still-settling rung.
        """
        if self._reclaim_remedy_time_budget_spent():
            return False
        if self.reclaim_ladder_ruled_irrelevant:
            return False
        if self._reclaim_rung_settling():
            return True
        if self.reclaim_rungs_issued_in_allotment >= self.RECLAIM_RUNG_ALLOTMENT:
            return False
        return self._next_reclaim_rung() is not None

    def settle_reclaim_relevance(self) -> None:
        """Judge whether the rung that has finished settling changed what the scheduler says blocks the head.

        Called once per escalation tick, and only acts once a rung's settling window has fully elapsed, so
        each issued rung is judged exactly once and on a full window. The judgement needs a published
        blocker: with none named there is nothing to measure a rung against, so the ladder keeps its
        ordinary budgets as its only bound.

        A rung counts as having done something if the named blocker changed or an inference start happened;
        either disproves "this rung addressed nothing" and resets the run. Once
        :attr:`RECLAIM_NO_EFFECT_LIMIT` rungs in a row have done neither, the ladder is set aside for the
        rest of the episode: it stops being issued and stops standing in the way of the give-up backstop.
        The escalation above it is untouched, so recovery still ends in a running worker.
        """
        if self.reclaim_effect_baseline is None or self._reclaim_rung_settling():
            return
        baseline_reason, baseline_starts = self.reclaim_effect_baseline
        self.reclaim_effect_baseline = None
        if (
            self._head_block_reason() != baseline_reason
            or self._job_tracker.total_num_inference_starts > baseline_starts
        ):
            self.reclaim_no_effect_cycles = 0
            return
        self.reclaim_no_effect_cycles += 1
        if self.reclaim_no_effect_cycles < self.RECLAIM_NO_EFFECT_LIMIT:
            return
        self.reclaim_ladder_ruled_irrelevant = True
        logger.warning(
            f"Save-our-ship: {self.reclaim_no_effect_cycles} constructive remedies in a row left the head's "
            f"stated blocker unchanged ({baseline_reason}) with no inference started, so freeing more card "
            "memory is not what this head is waiting on. The reclaim ladder is set aside for this episode and "
            "no longer defers the escalation.",
        )

    def _frozen_reclaim_ladder(self) -> tuple[ReclaimRung, ...]:
        """Return this episode's reclaim rungs, ordering them cheapest-first on the first call.

        The list is snapshotted once and reused for the whole episode. Rebuilding it per tick would let a card
        whose idle tenants come and go regenerate candidates without limit, and an escalation reading a
        never-shrinking candidate list would never reach its pool rebuild or its give-up backstop.
        """
        if self.reclaim_rungs is None:
            protected_models = frozenset(
                job.model for job in self._job_tracker.jobs_pending_inference if isinstance(job.model, str)
            )
            self.reclaim_rungs = build_reclaim_ladder(
                self._inference_scheduler.build_reclaim_ladder_candidates(
                    None,
                    protected_models=protected_models,
                ),
            )
        return self.reclaim_rungs

    def _next_reclaim_rung(self) -> ReclaimRung | None:
        """Return the rung the cursor points at, or None once the frozen list is exhausted."""
        ladder = self._frozen_reclaim_ladder()
        if self.reclaim_cursor >= len(ladder):
            return None
        return ladder[self.reclaim_cursor]

    def _reclaim_rung_settling(self) -> bool:
        """Return whether the most recently issued rung is still inside its settling window."""
        if self.reclaim_rung_issued_at is None:
            return False
        return (self._clock() - self.reclaim_rung_issued_at) < self.RECLAIM_RUNG_GRACE_SECONDS

    def _reclaim_remedy_time_budget_spent(self) -> bool:
        """Return whether this episode's reclaim rungs have occupied their aggregate time bound."""
        if self.reclaim_remedy_started_at is None:
            return False
        return (self._clock() - self.reclaim_remedy_started_at) >= self.RECLAIM_REMEDY_EPISODE_BUDGET_SECONDS

    def issue_next_constructive_remedy(self) -> ReclaimRung | None:
        """Perform the next reclaim rung of this episode, or hold while the last one issued settles.

        Advances the monotonic cursor through the frozen candidate list until a rung actually acts, skipping
        rungs whose target has gone away (those free nothing, so waiting on them would spend the settling window
        for no effect). The rung that acts opens a fresh settling window; the wedge oracle is simply whether the
        next tick still assesses the worker as wedged, so a rung that resolved the condition ends the episode
        and a rung that did not is followed by the next one.

        Returns:
            The rung that was performed, or None when the last rung is still settling, the allotment is spent,
            or nothing in the frozen list acts any more.
        """
        if self._reclaim_rung_settling() or self.reclaim_ladder_ruled_irrelevant:
            return None
        now = self._clock()
        ladder = self._frozen_reclaim_ladder()
        while self.reclaim_cursor < len(ladder):
            if self.reclaim_rungs_issued_in_allotment >= self.RECLAIM_RUNG_ALLOTMENT:
                return None
            rung = ladder[self.reclaim_cursor]
            self.reclaim_cursor += 1
            # A rung that promises nothing frees nothing, so spending a settling window on it is
            # indistinguishable from doing nothing while the wedge stands. Skip to a rung that can actually
            # change the resource condition.
            if rung.promised_freed_mb < _MIN_CONSTRUCTIVE_RUNG_MB:
                continue
            # Snapshot what this rung is supposed to move, taken before it acts so any change it causes is
            # credited to it. Its settling window is then judged on that outcome rather than on the rung
            # merely having been performed.
            head_blocker = self._head_block_reason()
            starts_before_rung = self._job_tracker.total_num_inference_starts
            if not execute_reclaim_rung(rung, self._inference_scheduler):
                continue
            self.reclaim_rungs_issued_in_allotment += 1
            self.reclaim_rung_issued_at = now
            if self.reclaim_remedy_started_at is None:
                self.reclaim_remedy_started_at = now
            self.reclaim_effect_baseline = (head_blocker, starts_before_rung) if head_blocker is not None else None
            if rung.kind in LANE_PAUSE_RUNG_KINDS:
                self.reclaim_paused_lanes.append(rung)
            logger.warning(
                f"Save-our-ship: issued the constructive remedy {rung.kind.value} on {rung.tenant_label} "
                f"(~{rung.promised_freed_mb:.0f}MB promised) and yielding up to "
                f"{self.RECLAIM_RUNG_GRACE_SECONDS:.0f}s for the wedge to clear. Rebuilding the pools against "
                "the same resource condition would not change it, and the accepted work is not faulted while a "
                "remedy remains.",
            )
            self._action_ledger.record(
                LedgerEventType.RECOVERY_RECLAIM_ISSUED,
                reason=f"save-our-ship constructive remedy ({rung.kind.value})",
                detail={
                    "rung_kind": rung.kind.value,
                    "tenant": rung.tenant_label,
                    "promised_freed_mb": round(rung.promised_freed_mb, 1),
                    "rungs_issued_in_allotment": self.reclaim_rungs_issued_in_allotment,
                },
            )
            return rung
        return None

    def renew_reclaim_rung_allotment(self) -> None:
        """Grant a fresh rung allotment because the escalation has moved past the pool rebuild.

        The cursor is untouched, so the remaining rungs are the ones the earlier allotment did not reach and the
        episode's total stays bounded by the frozen list.
        """
        self.reclaim_rungs_issued_in_allotment = 0

    def holds_lane_pause(self, kind: ReclaimRungKind) -> bool:
        """Whether this coordinator's own remedy is currently holding a lane pause of ``kind``.

        The pause is taken through the reclaim-ladder actuator, so the lifecycle records it under the ladder's
        owner. Without this the ladder's stranded-pause backstop reads a live recovery remedy as an orphan and
        restores it, which both defeats the remedy inside its yield window and restarts the lane process each
        time the remedy is re-issued. :meth:`restore_reclaimed_lanes` remains the responsible restore.
        """
        return any(rung.kind is kind for rung in self.reclaim_paused_lanes)

    def restore_reclaimed_lanes(self) -> None:
        """Restart every lane this episode's rungs paused, in reverse issue order.

        The recovery episode owns this restore because the pauses are its own: it holds the receipt of exactly
        which lanes its rungs acquired, so it restores those and no others. No external backstop is relied on,
        because the conditions those backstops require (a card debounced healthy for a sustained window, a
        matching device index) are not reached on a chronically pressured card, and a stranded reclaim-owned
        post-processing pause additionally stops suppressing the lane's admission-patience clock, aging out
        queued work.
        """
        for rung in reversed(self.reclaim_paused_lanes):
            if not restore_reclaim_rung(rung, self._inference_scheduler):
                continue
            self._action_ledger.record(
                LedgerEventType.RECOVERY_RECLAIM_RESTORED,
                reason=f"save-our-ship constructive remedy unwound ({rung.kind.value})",
                detail={"rung_kind": rung.kind.value, "tenant": rung.tenant_label},
            )
        self.reclaim_paused_lanes.clear()

    def _reset_constructive_remedy_budget(self) -> None:
        """Discard the frozen candidate list, the cursor, and every remedy budget for a closed episode."""
        self.reclaim_rungs = None
        self.reclaim_cursor = 0
        self.reclaim_rungs_issued_in_allotment = 0
        self.reclaim_rung_issued_at = None
        self.reclaim_remedy_started_at = None
        self.reclaim_effect_baseline = None
        self.reclaim_no_effect_cycles = 0
        self.reclaim_ladder_ruled_irrelevant = False
        self.give_up_yields_spent = 0

    def _give_up_yields_to_remedy(self) -> bool:
        """Return whether an in-flight remedy still deserves to land, so this give-up defers instead of faulting.

        The reachable-remedy guard: a give-up that arrives while a constructive remedy is still settling, or
        while a post-processing lane reclaim can still be issued for a servable head, would fault work the
        remedy is about to unblock. Constructive-rung refunds are bounded by
        :attr:`MAX_GIVE_UP_YIELDS_PER_EPISODE`; an already-issued post-processing unload instead owns its
        independent wall-clock grace, which is self-bounding and must not be shortened merely because the
        supervisor evaluates several give-up cycles during that window.
        """
        if self.pp_reclaim_remedy_issued_at is not None:
            return self._give_up_yields_to_pp_reclaim()
        if self.give_up_yields_spent >= self.MAX_GIVE_UP_YIELDS_PER_EPISODE:
            return False
        reclaim_rung_still_settling = (
            self._reclaim_rung_settling()
            and not self._reclaim_remedy_time_budget_spent()
            # A ladder ruled irrelevant cannot unblock this head, so its settling window is not a remedy
            # about to land. Deferring to it would hold accepted work behind an action already measured as
            # having no bearing on the block.
            and not self.reclaim_ladder_ruled_irrelevant
        )
        if reclaim_rung_still_settling:
            return True
        return self._give_up_yields_to_pp_reclaim()

    def _give_up_yields_to_pp_reclaim(self) -> bool:
        """Issue or await a post-processing lane reclaim for a wedged head; True while it deserves to land.

        The reachable-remedy guard for the healthy-pool structural queue wedge: a head parked only because
        the post-processing lane's resident module weights hold its room is servable the moment that idle
        lane unloads, so the give-up must drive the unload and yield rather than fault a backlog the card can
        run. Consulted only when a give-up is about to act on a structural queue wedge over available
        capacity; a broken pool's give-up never yields (no lane reclaim restores a dead pool). The first call
        of an episode issues the unload (the scheduler skips busy lanes and lanes already asked, so this never
        tears down in-flight work or re-spams the flag); subsequent calls keep yielding only inside
        :data:`PP_RECLAIM_REMEDY_GRACE_SECONDS`, after which the remedy is judged to have failed and the
        ordinary give-up proceeds. Returns False when no fresh unload could be issued and no issued one is
        still within its window: there is nothing left to yield to.
        """
        if not self.structural_queue_wedge_active() or not self.is_inference_capacity_available():
            return False
        now = self._clock()
        if self.pp_reclaim_remedy_issued_at is not None:
            # One issue per episode. Once its grace expires, reissuing the identical unload would reproduce its
            # own trigger and reset this clock forever, turning a bounded remedy into a permanent park.
            return now - self.pp_reclaim_remedy_issued_at <= self.PP_RECLAIM_REMEDY_GRACE_SECONDS
        if self._inference_scheduler.unload_post_process_models_from_vram():
            self.pp_reclaim_remedy_issued_at = now
            logger.warning(
                "Save-our-ship: the wedged head's blocker may be the post-processing lane's resident VRAM; "
                f"issued the idle lane unload and yielding up to {self.PP_RECLAIM_REMEDY_GRACE_SECONDS:.0f}s "
                "for the freed room to admit the head before faulting the backlog.",
            )
            return True
        return False

    def give_up_on_wedged_jobs(self, *, terminal: bool = False) -> None:
        """Fault unservable jobs and abort when no pool can recover.

        Args:
            terminal: Whether the supervisor flagged this give-up as the deliberate abandon-ship escalation
                (a wedge that outlived a fresh soft-reset cycle). A terminal give-up aborts even when the
                pool momentarily looks recoverable, so a persistent wedge over a live-but-idle pool cannot
                spin forever.
        """
        faulted = 0
        structural_queue_wedge = self.structural_queue_wedge_active()
        in_progress = self._job_tracker.jobs_in_progress
        if not self.is_inference_capacity_available() or structural_queue_wedge:
            for job in list(self._job_tracker.jobs_pending_inference):
                if job in in_progress:
                    continue
                # A retry a same-cycle pool rebuild just granted is left queued for a dispatch opportunity: a
                # non-terminal give-up must not fault the very retry recovery granted before the rebuilt pool
                # has had a chance to run it. The terminal give-up (the wedge outlived the continuation cycle)
                # faults it regardless, so this defers the drop by at most one cycle, it does not prevent it.
                if not terminal and job.id_ is not None and self._job_tracker.retry_granted_by_recovery(job.id_):
                    continue
                self._job_tracker.handle_job_fault_now(
                    job,
                    retryable=False,
                    fault_origin=JobFaultOrigin.SCHEDULING_RECOVERY,
                )
                # Release any disaggregation state so a give-up never strands a held pin/ledger entry.
                if job.id_ is not None:
                    self._release_disaggregated_job(job.id_)
                faulted += 1
        if not self.is_safety_capacity_available():
            stuck_safety = list(self._job_tracker.jobs_pending_safety_check) + list(
                self._job_tracker.jobs_being_safety_checked,
            )
            for job_info in stuck_safety:
                self._job_tracker.handle_job_fault_now(
                    job_info.sdk_api_job_info,
                    retryable=False,
                    fault_origin=JobFaultOrigin.SCHEDULING_RECOVERY,
                )
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

        # Safety capacity counts here for the same reason inference capacity does: it is on every job's path,
        # so a pool that cannot serve safety checks cannot serve work at all. Faulting the safety backlog
        # without recording that as structurally broken lets a worker with no safety process drop its queue on
        # every cycle while the escalation reads the pool as fine and never reaches its last rung.
        structurally_broken = (
            self.is_inference_pool_unrecoverable()
            or self.is_safety_pool_unrecoverable()
            or self.episode_saw_unrecoverable_pool
            or not self.is_inference_capacity_available()
            or not self.is_safety_capacity_available()
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
            disposition = self._terminal_recovery_callback()
            # Park only behind the terminal give-up. A non-terminal one still has its continuation cycle (a
            # fresh soft reset, then the terminal escalation) to try, so a withheld exit there leaves untried
            # remedies. Once the terminal give-up's exit is withheld the ladder is spent, and another cycle
            # would only rebuild and fault against the same pool.
            if terminal and disposition is RecoveryDisposition.CONTINUE_IN_PROCESS:
                self.enter_recovery_park(
                    RecoveryParkReason.UNRECOVERABLE_POOL,
                    {"jobs_faulted": faulted, "terminal": terminal},
                )
