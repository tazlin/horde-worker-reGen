from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

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
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.process_management.resources.resource_budget import (
    CommittedReserveLedger,
    predict_job_post_processing_vram_mb,
)
from horde_worker_regen.process_management.scheduling.workload_flow import POST_PROCESS_RESERVE_FLOW

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
        sampling_coresidency_check: Callable[[float], bool] | None = None,
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
            sampling_coresidency_check: Given a chain's estimated peak (MB), whether the card can run it
                alongside the sampling currently in progress. None (unit tests) allows co-running always.
        """
        self._process_map = process_map
        self._job_tracker = job_tracker
        self._process_lifecycle = process_lifecycle
        self._runtime_config = runtime_config
        self._state = state
        self._model_metadata = model_metadata
        self._reserve_ledger = reserve_ledger
        self._request_vram_reclaim = request_vram_reclaim
        self._sampling_coresidency_check = sampling_coresidency_check
        # Overridable so the load simulator can drive the aging window off its virtual clock; monotonic
        # keeps the window immune to wall-clock jumps.
        self._clock = time.monotonic
        self._deferrals: dict[str, _DeferralRecord] = {}

    def _estimate_post_processing_vram_mb(self, completed_job_info: HordeJobInfo) -> float:
        """Return the committed VRAM reserve for this post-processing unit."""
        sdk_job = completed_job_info.sdk_api_job_info
        baseline = self._model_metadata.get_baseline(sdk_job.model) if sdk_job.model is not None else None
        baseline_name = str(getattr(baseline, "value", baseline)) if baseline is not None else None
        estimate = predict_job_post_processing_vram_mb(sdk_job, baseline_name)
        if estimate is None:
            return 0.0
        return max(0.0, estimate)

    def _has_post_processing_headroom(
        self,
        *,
        post_process_process: HordeProcessInfo,
        reserve_vram_mb: float,
    ) -> bool:
        """Return whether the lane can start this chain without over-committing its card's VRAM.

        A pure predicate: the deferral bookkeeping (reclaim request, throttled warning, aging) is the
        caller's, so this can be evaluated for every candidate in a queue scan without side effects.
        """
        bridge_data = self._runtime_config.bridge_data
        if not bridge_data.enable_vram_budget or reserve_vram_mb <= 0:
            return True

        free_vram_mb = self._process_map.get_free_vram_mb(device_index=post_process_process.device_index)
        if free_vram_mb is None:
            return True

        committed_elsewhere_mb = self._reserve_ledger.total_vram_mb()
        required_mb = reserve_vram_mb + bridge_data.vram_reserve_mb
        available_mb = free_vram_mb - committed_elsewhere_mb
        return available_mb >= required_mb

    def _note_deferral(
        self,
        *,
        job_id: object,
        post_process_process: HordeProcessInfo,
        reserve_vram_mb: float,
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
        reclaim_found_idle = False
        if not record.reclaim_requested:
            reclaim_found_idle = self._request_vram_reclaim(post_process_process, post_process_process.device_index)
            record.reclaim_requested = True
            issued_reclaim = True

        if first_seen or (now - record.last_logged_at) >= _DEFER_LOG_INTERVAL_SECONDS:
            record.last_logged_at = now
            bridge_data = self._runtime_config.bridge_data
            free_vram_mb = self._process_map.get_free_vram_mb(device_index=post_process_process.device_index)
            available_mb = (
                free_vram_mb - self._reserve_ledger.total_vram_mb() if free_vram_mb is not None else float("nan")
            )
            if issued_reclaim:
                reclaim_note = (
                    "Issued idle VRAM reclaim first." if reclaim_found_idle else "No idle VRAM reclaim was available."
                )
            else:
                reclaim_note = "Idle VRAM reclaim already requested this episode."
            logger.warning(
                f"Deferring post-processing for job {job_id}: estimated peak {reserve_vram_mb:.0f} MB plus "
                f"reserve {bridge_data.vram_reserve_mb:.0f} MB exceeds free VRAM after commitments "
                f"({available_mb:.0f} MB available on card {post_process_process.device_index}). {reclaim_note}",
            )

    def _prune_deferrals(self, pending: tuple[HordeJobInfo, ...]) -> None:
        """Drop deferral bookkeeping for jobs that have left the pending queue (dispatched, aged, faulted)."""
        pending_ids = {str(job.sdk_api_job_info.id_) for job in pending}
        for key in list(self._deferrals):
            if key not in pending_ids:
                del self._deferrals[key]

    async def start_post_processing(self) -> None:
        """Dispatch pending post-processing work, bypassing an unfittable head and aging out the unservable.

        Three behaviors keep a job whose chain cannot fit the lane's card from wedging the lane:

        - **Queue scan**: the first *fittable* pending job is dispatched, so an unfittable head never
          blocks the fittable jobs queued behind it.
        - **Aging escape**: a job that has been unfittable for longer than the admission-patience window is
          reported faulted without images, so the horde reissues it rather than the worker parking it forever.
        - **One-shot reclaim**: an unfittable job asks the scheduler to evict idle VRAM once per starvation
          episode, not once per scheduling tick.
        """
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
        # can fault them. A deliberate whole-card pause is excluded: the residency lifecycle restarts the
        # lane when the card is released, so those jobs wait for the real lane unless post-processing is
        # session-disabled as structurally unsupported.
        if (
            self._process_map.num_post_process_processes() == 0
            and not self._process_lifecycle.is_post_process_gpu_paused
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
                return False

            if self._has_post_processing_headroom(
                post_process_process=post_process_process,
                reserve_vram_mb=reserve_vram_mb,
            ):
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
        post_processing = completed_job_info.sdk_api_job_info.payload.post_processing
        message_sent_succeeded = post_process_process.safe_send_message(
            HordePostProcessControlMessage(
                control_flag=HordeControlFlag.START_POST_PROCESS,
                job_id=completed_job_info.sdk_api_job_info.id_,
                images_bytes=completed_job_info.images_bytes,
                post_processing=list(post_processing) if post_processing else [],
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
        await self._job_tracker.begin_post_processing(completed_job_info)
        return True
