from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from horde_sdk.generation_parameters.alchemy.consts import is_strip_background_form
from loguru import logger

from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordePostProcessControlMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner, ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.process_management.resources.reclaim_ladder import VerifiedReclaimLadder
from horde_worker_regen.process_management.resources.resource_budget import (
    CommittedReserveLedger,
    predict_job_post_processing_vram_mb,
)
from horde_worker_regen.process_management.resources.run_metrics import (
    DecisionKind,
    DecisionSink,
    DecisionVerdict,
)
from horde_worker_regen.process_management.resources.vram_arbiter import (
    ActuatorCommand,
    ActuatorCommandKind,
    VramActuator,
    VramArbiter,
    VramRequest,
    VramRequestKind,
    VramVerdict,
)
from horde_worker_regen.process_management.scheduling.workload_flow import POST_PROCESS_RESERVE_FLOW
from horde_worker_regen.utils.vram_quota import effective_post_process_vram_quota_mb

_ADMISSION_PATIENCE_SECONDS = 90.0
"""How long a job may sit unfittable in the pending post-processing queue before it is faulted.

A chain whose estimated peak plus the VRAM reserve never fits the lane card's free VRAM would otherwise
be deferred forever. Past this window the worker reports the job faulted with no images so the horde reissues
it to another worker; returning raw images would violate the post-processing contract the worker advertised.
Sits below the orphan-recovery grace and the server-side timeout so a structurally unfittable chain terminates
well within patience."""

_DEFER_LOG_INTERVAL_SECONDS = 30.0
"""Minimum spacing between repeated deferral warnings for one job.

A long unfittable wait should leave a few diagnostic lines, not one per scheduling tick; the reclaim
request and the aging decision are governed separately so this throttle only bounds log volume."""

_BORROW_IDLE_RELEASE_SECONDS = 30.0
"""How long a borrowed service lane may stay held with no post-processing job actively using it before it is
restored, even while other post-processing jobs remain queued.

The loan is taken to make room for a post-processing dispatch. Holding it across *adjacent dispatched* jobs
avoids pause/restore churn (a job that fits with the loan dispatches within a tick, keeping the lane in use).
But a borrowed lane is a disaggregation lane, and pausing it disables disaggregation, which routes jobs
monolithic, which raises card pressure and sustains post-processing backlog: the queue then never fully drains
and the loan, gated only on a full drain, would be held indefinitely. This bound releases the lane once no job
has been actively post-processed for this window, so a stalled queue (jobs that defer and age out without ever
dispatching) can no longer hold a disaggregation lane hostage. Sits below the admission-patience window so the
lane is returned well before a stranded job ages out."""


def _round_or_none(value: float | None) -> float | None:
    """Round a measured MB figure to one decimal, passing ``None`` through for absent telemetry."""
    return None if value is None else round(value, 1)


@dataclass
class _DeferralRecord:
    """Bookkeeping for one job the admission gate could not immediately dispatch.

    ``first_deferred_at`` anchors the aging window; ``reclaim_requested`` makes the idle-VRAM reclaim a
    one-shot per starvation episode rather than one request per tick; ``last_logged_at`` throttles the
    deferral warning.
    """

    first_deferred_at: float
    last_logged_at: float
    reclaim_requested: bool = False
    applied_actuations: tuple[ActuatorCommand, ...] = ()
    """Most recent arbiter plan executed for this episode; a newly available plan may run once later."""
    admission_reclaim_attempted: bool = False
    """Whether a non-fitting arbiter verdict already spent the ordinary cache/model reclaim opportunity."""


class PostProcessOrchestrator:
    """Send inference-complete jobs that requested post-processing to the dedicated post-processing lane."""

    _process_map: ProcessMap
    _job_tracker: JobTracker
    _process_lifecycle: ProcessLifecycleManager
    _runtime_config: RuntimeConfig
    _state: WorkerState
    _model_metadata: ModelMetadata
    _reserve_ledger: CommittedReserveLedger
    _request_vram_reclaim: Callable[[HordeProcessInfo, int], bool]
    _vram_actuator: VramActuator | None
    _clock: Callable[[], float]

    def __init__(
        self,
        *,
        process_map: ProcessMap,
        job_tracker: JobTracker,
        process_lifecycle: ProcessLifecycleManager,
        runtime_config: RuntimeConfig,
        state: WorkerState,
        model_metadata: ModelMetadata,
        reserve_ledger: CommittedReserveLedger,
        request_vram_reclaim: Callable[[HordeProcessInfo, int], bool],
        vram_actuator: VramActuator | None = None,
        sampling_coresidency_check: Callable[[float], bool] | None = None,
        whole_card_residency_active: Callable[[], bool] = lambda: False,
        decision_sink: DecisionSink | None = None,
    ) -> None:
        """Initialize the orchestrator with references to its dependencies.

        Args:
            process_map (ProcessMap): The process map to use for finding the post-processing process.
            job_tracker (JobTracker): The job tracker to use for moving jobs between pending and being
                post-processed.
            process_lifecycle (ProcessLifecycleManager): The process lifecycle manager to signal if the
                post-processing lane needs to be replaced.
            runtime_config (RuntimeConfig): Holds the current bridge configuration snapshot.
            state: Shared worker state, including session-latched post-processing suppression.
            model_metadata: Provides the baseline needed for post-processing VRAM estimates.
            reserve_ledger: Shared committed-resource ledger used by every workload flow.
            request_vram_reclaim: Callback that asks the scheduler to evict idle VRAM on the lane's card.
            vram_actuator: Optional shared actuator used to execute the arbiter's complete reclaim plan.
            sampling_coresidency_check: Given a chain's estimated peak (MB), whether the card can run it
                alongside the sampling currently in progress. None (unit tests) allows co-running always.
            whole_card_residency_active: Whether a whole-card residency still owns the lane pause.
            decision_sink: Optional callback the manager injects to record the lane's admission decisions
                (deferrals and their resolution) to the stats export. None in unit tests and until wired.
        """
        self._process_map = process_map
        self._job_tracker = job_tracker
        self._process_lifecycle = process_lifecycle
        self._runtime_config = runtime_config
        self._state = state
        self._model_metadata = model_metadata
        self._reserve_ledger = reserve_ledger
        self._request_vram_reclaim = request_vram_reclaim
        self._vram_actuator = vram_actuator
        self._sampling_coresidency_check = sampling_coresidency_check
        self._whole_card_residency_active = whole_card_residency_active
        # Injected by the manager: records the lane's admission decisions (deferrals and their resolution)
        # to the stats export, coalesced on the receiving side. None in unit tests and until wired.
        self._decision_sink = decision_sink
        # Overridable so the load simulator can drive the aging window off its virtual clock; monotonic
        # keeps the window immune to wall-clock jumps.
        self._clock = time.monotonic
        self._deferrals: dict[str, _DeferralRecord] = {}
        # Service-lane pauses this PP drain actually acquired through the shared reclaim actuator. The receipt
        # from execute_arbiter_commands excludes no-ops, so restoring these can never lift a same-owner pause an
        # independent governor episode already held. Kept across adjacent dispatched jobs (avoiding pause/restore
        # churn), and released when the PP queue fully drains or when no job has been actively post-processed for
        # the idle-release window, so a borrowed disaggregation lane is not held hostage by a stalled queue.
        self._borrowed_service_lane_actuations: set[ActuatorCommandKind] = set()
        # Monotonic time the borrowed service lane(s) last had no post-processing job actively using them, or
        # None when a job is being post-processed (or nothing is borrowed). Anchors the idle-release window that
        # returns a borrowed disaggregation lane a stalled post-processing queue would otherwise hold forever.
        self._borrow_idle_since: float | None = None
        # Latched True once a borrow was released for idleness (the loan never led to a dispatch), so the same
        # stalled episode does not immediately re-borrow the lane it just returned; cleared when the accepted PP
        # queue fully drains and a fresh episode may borrow again.
        self._service_lane_borrow_suppressed: bool = False
        # The single VRAM arbiter, injected by the manager. It is the deciding authority for the lane's memory
        # admission question (replacing the banned free-VRAM read). None until wired (and in tests), where the
        # gate admits on missing telemetry.
        self._vram_arbiter: VramArbiter | None = None

    def set_vram_arbiter(self, arbiter: VramArbiter) -> None:
        """Inject the single VRAM arbiter: the deciding authority for the lane's memory admission question."""
        self._vram_arbiter = arbiter

    def _arbiter_admits_post_processing(
        self,
        *,
        completed_job_info: HordeJobInfo,
        post_process_process: HordeProcessInfo,
        reserve_vram_mb: float,
    ) -> VramVerdict | None:
        """Return the lane's arbiter verdict, or None when admission is intentionally bypassed.

        The chain's estimated peak is charged against the frozen cycle measurement, so a FITS verdict admits
        and a DEFER or DENY carries the reclaim plan the caller must execute. The
        reserve bypass is preserved (a disabled VRAM budget or a zero-peak chain always admits), and a cold or
        unwired arbiter admits, matching the every-gate-admits-on-missing-telemetry contract. The deferral
        bookkeeping (reclaim request, throttled warning, aging) is the caller's, so this stays side-effect free
        and can be evaluated for every candidate in a queue scan.
        """
        bridge_data = self._runtime_config.bridge_data
        if not bridge_data.enable_vram_budget or reserve_vram_mb <= 0:
            return None
        arbiter = self._vram_arbiter
        if arbiter is None or not arbiter.has_cycle:
            return None
        record = self._deferrals.get(str(completed_job_info.sdk_api_job_info.id_))
        return arbiter.evaluate(
            VramRequest(
                kind=VramRequestKind.PP_JOB,
                job_label=f"post_process:{completed_job_info.sdk_api_job_info.id_}",
                baseline=None,
                device_index=post_process_process.device_index,
                target_process_id=post_process_process.process_id,
                candidate_delta_mb=reserve_vram_mb,
                is_head_of_queue=True,
                allow_idle_service_lane_reclaim=(
                    record is not None
                    and record.admission_reclaim_attempted
                    and not self._borrowed_service_lane_actuations
                    and not self._service_lane_borrow_suppressed
                ),
            ),
        )

    def _estimate_post_processing_vram_mb(self, completed_job_info: HordeJobInfo) -> float:
        """Return the committed VRAM reserve for this post-processing unit."""
        sdk_job = completed_job_info.sdk_api_job_info
        baseline = self._model_metadata.get_baseline(sdk_job.model) if sdk_job.model is not None else None
        baseline_name = str(getattr(baseline, "value", baseline)) if baseline is not None else None
        estimate = predict_job_post_processing_vram_mb(sdk_job, baseline_name)
        if estimate is None:
            return 0.0
        return max(0.0, estimate)

    def _effective_lane_cap_mb(self, post_process_process: HordeProcessInfo) -> float | None:
        """Return the lane's allocator-guard cap (MB) for its card, or None when the card total is unknown.

        Derived from the same policy the lane applies to its own allocator, so the gate reasons about the
        exact ceiling the lane enforces rather than only the card's free VRAM.
        """
        total_mb = self._process_map.get_reported_total_vram_mb(device_index=post_process_process.device_index)
        if total_mb is None:
            return None
        return effective_post_process_vram_quota_mb(total_mb)

    async def _fault_if_exceeds_lane_cap(
        self,
        *,
        completed_job_info: HordeJobInfo,
        post_process_process: HordeProcessInfo,
        reserve_vram_mb: float,
    ) -> bool:
        """Fault a chain whose estimated peak exceeds the lane's cap; return whether it was faulted.

        This is structural, not transient: a chain estimated above the lane's own allocator guard cannot
        run in the lane no matter how idle the card becomes, so dispatching it would only burn the patience
        window on a guaranteed in-lane out-of-memory. Faulting it now (no images) lets the horde reissue it
        to a larger worker instead. Only consulted when the VRAM budget is enabled and an estimate exists;
        otherwise the lane's own cap and its reclaim-and-retry remain the backstop.
        """
        if not self._runtime_config.bridge_data.enable_vram_budget or reserve_vram_mb <= 0:
            return False
        lane_cap_mb = self._effective_lane_cap_mb(post_process_process)
        if lane_cap_mb is None or reserve_vram_mb <= lane_cap_mb:
            return False
        await self._fault_without_images(
            completed_job_info,
            reason=(
                f"post-processing chain's estimated peak {reserve_vram_mb:.0f}MB exceeds the lane's "
                f"{lane_cap_mb:.0f}MB VRAM cap on card {post_process_process.device_index}; it cannot be "
                "hosted on this card"
            ),
        )
        return True

    def _note_deferral(
        self,
        *,
        job_id: object,
        post_process_process: HordeProcessInfo,
        reserve_vram_mb: float,
        verdict: VramVerdict,
        now: float,
    ) -> None:
        """Record that a job could not be admitted this tick, requesting reclaim once and logging thriftily.

        The first deferral of a starvation episode issues a single idle-VRAM reclaim request (a freshly
        released slot may not exist yet). Further ticks reuse the record so reclaim is not re-issued every
        cycle, and the warning is throttled so a long unfittable wait leaves a few diagnostic lines rather
        than one per tick.
        """
        key = str(job_id)
        record = self._deferrals.get(key)
        first_seen = record is None
        if record is None:
            record = _DeferralRecord(first_deferred_at=now, last_logged_at=now - _DEFER_LOG_INTERVAL_SECONDS)
            self._deferrals[key] = record

        issued_reclaim = False
        reclaim_action_available = False
        required_actuations = verdict.required_actuations
        if (
            required_actuations
            and self._vram_actuator is not None
            and required_actuations != record.applied_actuations
        ):
            applied_actuations = VerifiedReclaimLadder.execute_arbiter_commands(
                required_actuations,
                self._vram_actuator,
                device_index=post_process_process.device_index,
                for_head_of_queue=True,
            )
            record.applied_actuations = required_actuations
            record.reclaim_requested = True
            for actuation in applied_actuations:
                if actuation.kind in (
                    ActuatorCommandKind.PAUSE_VAE_LANE,
                    ActuatorCommandKind.PAUSE_COMPONENT_LANE,
                ):
                    self._borrowed_service_lane_actuations.add(actuation.kind)
                    # A borrow is only taken for a job that is deferring (not yet dispatched), so the lane is
                    # idle from the instant it is acquired; anchor the idle-release window here.
                    self._borrow_idle_since = now
            reclaim_action_available = bool(applied_actuations)
            issued_reclaim = True
        elif not record.reclaim_requested:
            reclaim_action_available = self._request_vram_reclaim(
                post_process_process,
                post_process_process.device_index,
            )
            record.reclaim_requested = True
            issued_reclaim = True
        record.admission_reclaim_attempted = True

        if first_seen or (now - record.last_logged_at) >= _DEFER_LOG_INTERVAL_SECONDS:
            record.last_logged_at = now
            if issued_reclaim:
                reclaim_note = (
                    "Issued the available VRAM reclaim actions."
                    if reclaim_action_available
                    else "No VRAM reclaim action was available."
                )
            else:
                reclaim_note = "Idle VRAM reclaim already requested this episode."
            measured = verdict.measured
            device_free_mb = measured.device_free_mb
            available_mb = measured.available_mb
            arithmetic = (
                f"{available_mb:.0f} MB available from device-free {device_free_mb:.0f} MB minus "
                f"{measured.outstanding_reservations_mb:.0f} MB outstanding reservations and "
                f"{measured.noise_buffer_mb:.0f} MB noise"
                if available_mb is not None and device_free_mb is not None
                else verdict.reason
            )
            logger.warning(
                f"Deferring post-processing for job {job_id}: candidate {reserve_vram_mb:.0f} MB does not fit "
                f"the measured device room on card {post_process_process.device_index} ({arithmetic}). "
                f"{reclaim_note}",
            )

        # The decision inputs, persisted for offline analysis. Called every deferral tick; the sink coalesces
        # a sustained deferral into one opening record plus bounded heartbeats (independent of the log throttle
        # above, so the two cannot drift into lockstep or mask each other).
        if self._decision_sink is not None:
            measured = verdict.measured
            self._decision_sink(
                decision_kind=DecisionKind.PP_DEFERRAL,
                subject=key,
                verdict=DecisionVerdict.DEFER,
                reason=verdict.reason or "pp_does_not_fit",
                inputs={
                    "device_index": post_process_process.device_index,
                    "candidate_delta_mb": round(reserve_vram_mb, 1),
                    "device_free_mb": _round_or_none(measured.device_free_mb),
                    "available_mb": _round_or_none(measured.available_mb),
                    "outstanding_reservations_mb": round(measured.outstanding_reservations_mb, 1),
                    "noise_buffer_mb": round(measured.noise_buffer_mb, 1),
                },
                timestamp=now,
            )

    def _prune_deferrals(self, pending: tuple[HordeJobInfo, ...]) -> None:
        """Drop deferral bookkeeping for jobs that have left the pending queue (dispatched, aged, faulted)."""
        pending_ids = {str(job.sdk_api_job_info.id_) for job in pending}
        for key in list(self._deferrals):
            if key not in pending_ids:
                del self._deferrals[key]
                # The subject left the pending queue: close any open deferral decision for it with a final
                # resolving record (a resolving verdict for an untracked subject is dropped by the sink).
                if self._decision_sink is not None:
                    self._decision_sink(
                        decision_kind=DecisionKind.PP_DEFERRAL,
                        subject=key,
                        verdict=DecisionVerdict.NO_OP,
                        reason="left_pending_queue",
                    )

    async def start_post_processing(self) -> None:
        """Dispatch pending post-processing work, bypassing an unfittable head and aging out the unservable.

        Three behaviors keep a job whose chain cannot fit the lane's card from wedging the lane:

        - **Queue scan**: the first *fittable* pending job is dispatched, so an unfittable head never
          blocks the fittable jobs queued behind it. A chain that cannot share the card with the current
          sampler is treated the same way: it waits for the sampler while the scan looks for later work that
          can safely co-run.
        - **Aging escape**: a job that has been unfittable for longer than the admission-patience window is
          reported faulted without images, so the horde reissues it rather than the worker parking it forever.
        - **One-shot reclaim**: an unfittable job asks the scheduler to evict idle VRAM once per starvation
          episode, not once per scheduling tick.
        """
        self._release_borrowed_service_lanes_when_unused()
        pending = self._job_tracker.jobs_pending_post_processing
        self._prune_deferrals(pending)
        if not pending:
            return

        if self._state.post_processing_disabled_by_breaker:
            reason = self._state.post_processing_disabled_reason or "post-processing disabled for this session"
            for job_info in list(pending):
                await self._fault_without_images(job_info, reason=reason)
            return

        now = self._clock()

        self._ensure_lane_liveness_for_pending_work()

        post_process_process = self._process_map.get_first_available_post_process_process()
        if post_process_process is not None and await self._try_dispatch_first_fittable(
            pending=pending,
            post_process_process=post_process_process,
            now=now,
        ):
            return

        # Deferral records are otherwise only created against a live lane process, so with the lane
        # absent (crashed, failed to start, or torn down without coming back) no patience clock would
        # ever start and these jobs would wait forever. Arm the clock here so the aging escape below
        # can fault them. Only a whole-card residency pause suppresses the clock: it has a live restore path
        # (the residency completion loop restarts the lane when the card is released) and heavy jobs are
        # expected to wait for it. A reclaim-ladder pause does NOT suppress it: the ladder restores the lane
        # only if the card recovers to HEALTHY, which is not guaranteed within the patience window (a card
        # stuck saturated may never recover), so its pending post-processing must run the liveness countdown
        # and age out to the raw-image fallback rather than wait on a lane that may never return.
        if (
            self._process_map.num_post_process_processes() == 0
            and self._process_lifecycle.post_process_pause_owner is not PauseOwner.WHOLE_CARD
        ):
            for job_info in pending:
                key = str(job_info.sdk_api_job_info.id_)
                if key not in self._deferrals:
                    self._deferrals[key] = _DeferralRecord(
                        first_deferred_at=now,
                        last_logged_at=now,
                        reclaim_requested=True,
                    )
                    logger.warning(
                        f"Post-processing lane has no process; starting the patience window for job "
                        f"{job_info.sdk_api_job_info.id_} (the job is faulted if the lane does not return).",
                    )

        # A no-image fault needs no lane, so age out unservable jobs whether or not the lane is free: a
        # busy or unfittable head must never hold a long-waiting job past its patience window.
        await self._age_out_unfittable(now=now)

    def is_service_lane_borrowed(self, kind: ActuatorCommandKind) -> bool:
        """Whether this PP drain currently holds a receipt for a service-lane pause of ``kind``.

        The self-heal backstop reads this to decide whether a reclaim-ladder lane pause has a live claimant: a
        pause this orchestrator borrowed has a responsible restore owner here and must not be lifted from under
        it, so the backstop only reclaims a ladder-owned pause that no receipt (and no saturation episode)
        claims.
        """
        return kind in self._borrowed_service_lane_actuations

    def _release_borrowed_service_lanes_when_unused(self) -> None:
        """Restore service lanes this PP drain borrowed once no post-processing job is actively using them.

        The loan makes room for a post-processing *dispatch*. While a job is being post-processed the pause
        stays in place (restoring its CUDA context before the chain allocates would recreate the exact non-fit
        the pause resolved), and adjacent dispatched jobs share the one bounded loan. A full drain (nothing
        queued and nothing active) restores immediately, as before.

        The added escape is for a stalled queue: a borrowed lane is a disaggregation lane, so holding it
        disables disaggregation, which routes work monolithic and sustains post-processing backlog the queue
        never fully drains. When no job has been actively post-processed for :data:`_BORROW_IDLE_RELEASE_SECONDS`
        the lane is restored even though jobs remain queued, so a disaggregation lane is never held hostage by
        post-processing work that only defers and ages out without ever dispatching. The owner-guarded actuator
        restore is idempotent, and the receipt-based set ensures this path never lifts a pause it did not
        acquire.
        """
        if not self._job_tracker.jobs_pending_post_processing and not self._job_tracker.jobs_being_post_processed:
            # The starvation episode is over: a future episode may borrow a service lane again.
            self._service_lane_borrow_suppressed = False

        if not self._borrowed_service_lane_actuations:
            self._borrow_idle_since = None
            return
        actuator = self._vram_actuator
        if actuator is None:
            return

        if self._job_tracker.jobs_being_post_processed:
            # A job is actively using the borrowed room; hold the loan and reset the idle clock so the release
            # window measures only uninterrupted idleness after the last dispatched job.
            self._borrow_idle_since = None
            return

        drained = not self._job_tracker.jobs_pending_post_processing
        if not drained:
            now = self._clock()
            if self._borrow_idle_since is None:
                self._borrow_idle_since = now
                return
            if (now - self._borrow_idle_since) < _BORROW_IDLE_RELEASE_SECONDS:
                return
            # The loan never led to a dispatch and the queue is stalled: return the disaggregation lane and do
            # not let this same episode re-borrow it, so the lane is not thrashed pause/restore every window.
            self._service_lane_borrow_suppressed = True

        self._borrow_idle_since = None
        # Reverse the reclaim order. The current drain contract borrows at most one lane, and retaining the
        # canonical unwind order keeps this robust if the eligible lane order grows later.
        for kind in (ActuatorCommandKind.PAUSE_COMPONENT_LANE, ActuatorCommandKind.PAUSE_VAE_LANE):
            if kind not in self._borrowed_service_lane_actuations:
                continue
            if kind is ActuatorCommandKind.PAUSE_COMPONENT_LANE:
                actuator.restore_component_lane(None)
            else:
                actuator.restore_vae_lane(None)
            self._borrowed_service_lane_actuations.discard(kind)

    def _ensure_lane_liveness_for_pending_work(self) -> None:
        """Ask lifecycle for a lane when pending work has no live dispatch target."""
        if self._process_map.num_post_process_processes() > 0:
            return

        pause_owner = self._process_lifecycle.post_process_pause_owner
        if pause_owner is PauseOwner.WHOLE_CARD:
            if (
                not self._whole_card_residency_active()
                and not self._job_tracker.jobs_pending_inference
                and not self._job_tracker.jobs_in_progress
            ):
                self._process_lifecycle.restore_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD)
            return

        if pause_owner is None:
            self._process_lifecycle.start_post_process_processes()

    async def _try_dispatch_first_fittable(
        self,
        *,
        pending: tuple[HordeJobInfo, ...],
        post_process_process: HordeProcessInfo,
        now: float,
    ) -> bool:
        """Dispatch the first fittable pending job, deferring the ones ahead of it. Return whether one went.

        Faults on a job with no deliverable identity or result are terminal (re-running post-processing
        cannot fix them); the scan continues past a faulted job rather than letting it block the queue.
        """
        for completed_job_info in list(pending):
            if not self._is_deliverable(completed_job_info, post_process_process):
                await self._fault_and_abandon(completed_job_info, post_process_process)
                continue

            reserve_vram_mb = self._estimate_post_processing_vram_mb(completed_job_info)
            if await self._fault_if_exceeds_lane_cap(
                completed_job_info=completed_job_info,
                post_process_process=post_process_process,
                reserve_vram_mb=reserve_vram_mb,
            ):
                continue

            if (
                self._sampling_coresidency_check is not None
                and self._process_map.num_busy_with_inference(device_index=post_process_process.device_index) > 0
                and not self._sampling_coresidency_check(reserve_vram_mb)
            ):
                # The card cannot hold this chain alongside the sampling in progress; starting it now
                # would silently demand-page both for the whole overlap. Wait for the sampling to finish
                # (seconds; the dispatch-side gate keeps the next sampling job from jumping in over an
                # already-running chain). The patience record still arms so a pathological never-idle
                # card ages the job out to a no-image fault rather than parking it forever.
                key = str(completed_job_info.sdk_api_job_info.id_)
                if key not in self._deferrals:
                    self._deferrals[key] = _DeferralRecord(
                        first_deferred_at=now,
                        last_logged_at=now,
                        reclaim_requested=True,
                    )
                    logger.debug(
                        f"Deferring post-processing for job {completed_job_info.sdk_api_job_info.id_}: "
                        f"its chain ({reserve_vram_mb:.0f}MB) cannot share the card with the sampling in "
                        "progress; waiting for the card.",
                    )
                continue

            verdict = self._arbiter_admits_post_processing(
                completed_job_info=completed_job_info,
                post_process_process=post_process_process,
                reserve_vram_mb=reserve_vram_mb,
            )
            if verdict is None or verdict.admits:
                self._deferrals.pop(str(completed_job_info.sdk_api_job_info.id_), None)
                return await self._dispatch(
                    completed_job_info=completed_job_info,
                    post_process_process=post_process_process,
                    reserve_vram_mb=reserve_vram_mb,
                )

            self._note_deferral(
                job_id=completed_job_info.sdk_api_job_info.id_,
                post_process_process=post_process_process,
                reserve_vram_mb=reserve_vram_mb,
                verdict=verdict,
                now=now,
            )
        return False

    async def _age_out_unfittable(self, *, now: float) -> None:
        """Fault any job that has been unfittable past the admission-patience window.

        This is an admission decision, not a lane fault, but it still feeds the lane's fault breaker because
        the worker accepted post-processing work it could not host.
        """
        for job_info in list(self._job_tracker.jobs_pending_post_processing):
            job_id = job_info.sdk_api_job_info.id_
            record = self._deferrals.get(str(job_id))
            if record is None or (now - record.first_deferred_at) < _ADMISSION_PATIENCE_SECONDS:
                continue

            waited = now - record.first_deferred_at
            reason = (
                f"post-processing could not be admitted within {waited:.0f}s: no lane process, or its "
                "estimated peak never fit the lane card's free VRAM after commitments"
            )
            logger.warning(
                f"Post-processing for job {job_id} could not be admitted within {waited:.0f}s; faulting it."
            )
            await self._fault_without_images(job_info, reason=reason)
            self._deferrals.pop(str(job_id), None)

    async def _fault_without_images(self, job_info: HordeJobInfo, *, reason: str) -> None:
        """Terminally fault a post-inference job without submitting raw images."""
        logger.error(
            f"Faulting job {job_info.sdk_api_job_info.id_} without images: {reason}. "
            "The horde will reissue it to another worker.",
        )
        self._job_tracker.note_post_processing_overcommit_fault()
        await self._job_tracker.fault_post_inference_job(job_info, reason=reason)
        self._deferrals.pop(str(job_info.sdk_api_job_info.id_), None)

    def _is_deliverable(self, completed_job_info: HordeJobInfo, post_process_process: HordeProcessInfo) -> bool:
        """Return whether the job has the identity, result, and operations post-processing needs."""
        if completed_job_info.job_image_results is None:
            logger.error("completed_job_info.job_image_results is None")
            return False
        if completed_job_info.sdk_api_job_info.id_ is None:
            logger.error("completed_job_info.sdk_api_job_info.id_ is None")
            return False
        if not completed_job_info.sdk_api_job_info.payload.post_processing:
            logger.error("Job queued for post-processing has no post-processing operations")
            return False
        return True

    async def _fault_and_abandon(
        self,
        completed_job_info: HordeJobInfo,
        post_process_process: HordeProcessInfo,
    ) -> None:
        """Terminally fault an undeliverable job and remove it from the pending post-processing queue."""
        reason = "job could not be started on the post-processing lane"
        await self._job_tracker.fault_post_inference_job(
            completed_job_info,
            reason=reason,
        )
        logger.error(f"Failed to start post-processing for job {completed_job_info.sdk_api_job_info.id_}")
        await self._job_tracker.abandon_pending_post_processing(completed_job_info)
        self._deferrals.pop(str(completed_job_info.sdk_api_job_info.id_), None)

    async def _dispatch(
        self,
        *,
        completed_job_info: HordeJobInfo,
        post_process_process: HordeProcessInfo,
        reserve_vram_mb: float,
    ) -> bool:
        """Send the job to the lane, reserving its peak; on send failure signal a lane replacement.

        Returns True when the job was handed to the lane (so the caller stops scanning), False when the
        send failed and the lane could not be marked for replacement (a starting/dead lane the caller
        should leave to recovery).
        """
        # Background removal is not run here: it has no in-graph path (its ``rembg`` stack is not in the
        # main venv) and is applied last on the image-utilities lane after this pass. Only the pure-torch
        # transforms (upscale/face-fix) go to the post-processing child.
        post_processing = completed_job_info.sdk_api_job_info.payload.post_processing or []
        lane_forms = [form for form in post_processing if not is_strip_background_form(form)]
        message_sent_succeeded = post_process_process.safe_send_message(
            HordePostProcessControlMessage(
                control_flag=HordeControlFlag.START_POST_PROCESS,
                job_id=completed_job_info.sdk_api_job_info.id_,
                images_bytes=completed_job_info.images_bytes,
                post_processing=lane_forms,
            ),
        )

        if not message_sent_succeeded:
            live_process = self._process_map.get_post_process_process()
            if live_process is None:
                return False

            if (
                not live_process.is_process_alive()
                or live_process.last_process_state == HordeProcessState.PROCESS_STARTING
            ):
                return False

            logger.error(f"Failed to start post-processing for job {completed_job_info.sdk_api_job_info.id_}")
            self._process_lifecycle.post_process_processes_should_be_replaced = True
            return True

        self._process_map.on_process_state_change(
            post_process_process.process_id,
            HordeProcessState.POST_PROCESSING,
        )
        self._process_map.on_last_job_reference_change(
            post_process_process.process_id,
            completed_job_info.sdk_api_job_info,
        )
        self._reserve_ledger.set(
            POST_PROCESS_RESERVE_FLOW,
            str(completed_job_info.sdk_api_job_info.id_),
            vram_mb=reserve_vram_mb,
        )
        await self._job_tracker.begin_post_processing(
            completed_job_info,
            process_id=post_process_process.process_id,
            process_launch_identifier=post_process_process.process_launch_identifier,
        )
        return True
