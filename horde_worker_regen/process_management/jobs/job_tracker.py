"""Unified, ID-keyed job state management.

Every job the worker knows about is exactly one ``TrackedJob`` in a single
``dict[GenerationID, TrackedJob]``, with an explicit :class:`JobStage`. All
stage changes funnel through one transition method that validates legality,
so "a job is in exactly one stage" is structural rather than emergent.

Stage collections that callers previously manipulated directly
(``jobs_pending_inference``, ``jobs_in_progress``, etc.) are now derived,
read-only views. The mutation API is unchanged in name and signature (the
methods remain ``async`` for interface stability) but mutations are plain
synchronous dict operations on the event-loop thread; sequences of awaits
can no longer interleave a partially-applied transition.

Notes on intentional semantics preserved from the previous implementation:

- A job remains visible in ``jobs_pending_inference`` while inference is in
  progress; it leaves that view only when the inference result (or a fault)
  arrives. Queue-size accounting depends on this.
- ``job_faults`` entries are kept independent of job lifetime; they are
  cleared explicitly via :meth:`clear_faults_for_job`, not by finalize/purge.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import auto

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import (
    GenMetadataEntry,
    ImageGenerateJobPopResponse,
)
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from horde_sdk.ai_horde_api.fields import GenerationID
from horde_sdk.worker.chaining import (
    CHAIN_NODE_STATE,
    POST_PROCESS_STAGE_NAME,
    SAFETY_CHECK_STAGE_NAME,
    ChainConsistencyError,
    ChainExecutionContext,
    image_generation_flow,
)
from horde_sdk.worker.consts import GENERATION_PROGRESS
from loguru import logger

from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.utils.job_queue_analyzer import JobQueueAnalyzer


class JobStage(enum.Enum):
    """The lifecycle stage a tracked job is currently in.

    A job is in exactly one stage at any instant. ``DETACHED`` is a transient
    hand-off stage: the job is tracked (still in lookup) but enqueued nowhere,
    e.g. between an inference result arriving and the job being queued for
    safety. Jobs must not remain ``DETACHED`` across loop iterations.
    """

    PENDING_INFERENCE = auto()
    """Popped and waiting for (or undergoing dispatch to) an inference process."""
    INFERENCE_IN_PROGRESS = auto()
    """Sent to an inference process which has not yet returned a result."""
    DISAGGREGATION_DECODING = auto()
    """A disaggregated job whose sampling finished; its latent is being decoded (and post-processed) on the
    image lane while the sampler slot has been released for the next job.

    Deliberately not counted in ``jobs_in_progress`` (the inference concurrency cap), so the freed sampler is
    schedulable again, yet the job is still tracked and in-flight: the disaggregation orchestrator holds its
    decode state and hands off the images (or a fault) on completion. Unlike ``DETACHED`` this is a durable
    stage a job legitimately occupies across many loop iterations while the decode runs."""
    DETACHED = auto()
    """Tracked but not in any queue; a transient hand-off state."""
    PENDING_POST_PROCESSING = auto()
    """Inference finished and the job requested post-processing; waiting for the post-processing lane."""
    POST_PROCESSING = auto()
    """Sent to the post-processing process; awaiting the post-processed images."""
    PENDING_SAFETY_CHECK = auto()
    """Inference (and any post-processing) finished; waiting for a safety process slot."""
    SAFETY_CHECKING = auto()
    """Sent to the safety process; awaiting its verdict."""
    PENDING_SUBMIT = auto()
    """Ready to be submitted to the API (success or fault)."""


class InferenceFailureResolution(enum.Enum):
    """What the tracker decided to do with a job whose inference attempt failed.

    The two ``RETRY*`` outcomes leave the job in :attr:`JobStage.PENDING_INFERENCE` for another
    dispatch; ``FAULTED`` is terminal (the job has been moved to :attr:`JobStage.PENDING_SUBMIT` as a
    fault and counted once). Callers branch on this to log, audit, and (for ``RETRY_DEGRADED``) drive a
    more conservative re-dispatch.
    """

    RETRY = auto()
    """Requeued to ``PENDING_INFERENCE`` for a fresh, normal attempt (a crash/hang/transient failure)."""
    RETRY_DEGRADED = auto()
    """Requeued for one degraded, isolated attempt after a resource (out-of-memory) failure."""
    FAULTED = auto()
    """Attempts exhausted or the failure is non-retryable; moved to ``PENDING_SUBMIT`` as a terminal fault."""


_ALLOWED_TRANSITIONS: dict[JobStage, frozenset[JobStage]] = {
    JobStage.PENDING_INFERENCE: frozenset(
        {
            JobStage.INFERENCE_IN_PROGRESS,
            JobStage.DETACHED,
            JobStage.PENDING_POST_PROCESSING,
            JobStage.PENDING_SAFETY_CHECK,
            JobStage.PENDING_SUBMIT,
        },
    ),
    JobStage.INFERENCE_IN_PROGRESS: frozenset(
        {
            JobStage.PENDING_INFERENCE,
            JobStage.DISAGGREGATION_DECODING,
            JobStage.DETACHED,
            JobStage.PENDING_POST_PROCESSING,
            JobStage.PENDING_SAFETY_CHECK,
            JobStage.PENDING_SUBMIT,
        },
    ),
    JobStage.DISAGGREGATION_DECODING: frozenset(
        {
            JobStage.PENDING_INFERENCE,
            JobStage.DETACHED,
            JobStage.PENDING_POST_PROCESSING,
            JobStage.PENDING_SAFETY_CHECK,
            JobStage.PENDING_SUBMIT,
        },
    ),
    JobStage.DETACHED: frozenset(
        {
            JobStage.PENDING_POST_PROCESSING,
            JobStage.PENDING_SAFETY_CHECK,
            JobStage.PENDING_SUBMIT,
            JobStage.PENDING_INFERENCE,
        },
    ),
    JobStage.PENDING_POST_PROCESSING: frozenset(
        {JobStage.POST_PROCESSING, JobStage.PENDING_SAFETY_CHECK, JobStage.PENDING_SUBMIT, JobStage.DETACHED},
    ),
    JobStage.POST_PROCESSING: frozenset(
        {
            JobStage.DETACHED,
            JobStage.PENDING_POST_PROCESSING,
            JobStage.PENDING_SAFETY_CHECK,
            JobStage.PENDING_SUBMIT,
        },
    ),
    JobStage.PENDING_SAFETY_CHECK: frozenset(
        {JobStage.SAFETY_CHECKING, JobStage.PENDING_SUBMIT, JobStage.DETACHED},
    ),
    JobStage.SAFETY_CHECKING: frozenset(
        {JobStage.DETACHED, JobStage.PENDING_SAFETY_CHECK, JobStage.PENDING_SUBMIT},
    ),
    JobStage.PENDING_SUBMIT: frozenset(),
}
"""Legal stage transitions. ``PENDING_SUBMIT`` is terminal; jobs leave it only by removal."""

_QUEUED_STAGES: tuple[JobStage, ...] = (
    JobStage.PENDING_INFERENCE,
    JobStage.INFERENCE_IN_PROGRESS,
    JobStage.DISAGGREGATION_DECODING,
    JobStage.PENDING_POST_PROCESSING,
    JobStage.POST_PROCESSING,
    JobStage.PENDING_SAFETY_CHECK,
    JobStage.SAFETY_CHECKING,
    JobStage.PENDING_SUBMIT,
)
"""Stages counted by ``num_jobs_total`` (everything except ``DETACHED``)."""


@dataclass
class TrackedJob:
    """A single job known to the worker, with its current lifecycle stage."""

    job_id: GenerationID
    """The generation ID this job is keyed by."""
    sdk_api_job_info: ImageGenerateJobPopResponse
    """The pop response for this job, as last seen."""
    stage: JobStage
    """The current lifecycle stage."""
    job_info: HordeJobInfo | None = None
    """Result-carrying job info. None only for jobs registered outside the normal pop path."""
    time_popped: float | None = None
    """Epoch time the job was popped, or None if it was never formally popped."""
    pop_order: int = 0
    """Monotonic sequence assigned at registration; preserves pop/queue order."""
    stage_sequence: int = 0
    """Monotonic sequence assigned on each stage change; preserves FIFO order within a stage."""
    stage_timestamps: dict[str, float] = field(default_factory=dict)
    """Epoch time of the first entry into each stage, keyed by ``JobStage.name``.

    Together with ``time_popped`` this gives per-job latency breakdowns
    (queue wait, inference, safety, submit)."""
    current_stage_since: float = 0.0
    """Epoch time the job entered its *current* stage (updated on every transition).

    Unlike ``stage_timestamps`` (which records only the *first* entry into each stage), this tracks the
    latest entry, so a job's true time-in-stage is accurate even after it cycles back through a stage (e.g.
    a safety re-check). Drives the status dump's per-stage aging so a stuck stage shows a growing age."""
    inference_attempts: int = 0
    """How many inference attempts have failed for this job; bounds retry against ``max_inference_attempts``."""
    degraded_retry_used: bool = False
    """Whether this job has already spent its one degraded (isolated) retry for a resource failure."""
    needs_degraded_dispatch: bool = False
    """Set when this job's next dispatch should run degraded/isolated (consumed by the scheduler)."""
    admitted_over_budget: bool = False
    """Set when the scheduler admitted this job despite the VRAM budget judging it does not fit.

    Such a job was knowingly over-committed onto a contended device, so a crash/hang of its slot is a
    resource failure even though the dead slot leaves no message to classify. Carrying the signal on the
    job lets the fault path route it to the bounded degraded/isolated retry instead of a plain
    re-dispatch onto another equally-over-committed slot, which would only kill a second process."""
    admitted_exclusive: bool = False
    """Set with ``admitted_over_budget`` when ``overbudget_exclusive_mode`` is on: this heavy job must run
    with the device to itself. While such a job is pending or in progress the scheduler suppresses
    concurrent pre-staging/dispatch of other models so a second resident model cannot push free VRAM to ~0
    and spill this job's weights to system RAM (the live storm's mechanism). It also earns a per-step hang
    grace (``overbudget_step_timeout``) so its legitimately slow steps are not killed as a hang."""
    last_dispatched_device_index: int | None = None
    """The card this job was last dispatched to, or None on a single-GPU host (no card attribution).

    Keys this model's over-budget fault streak per card so a model unservable on a small card can still be
    advertised and run on a larger one. None on single-GPU keeps the streak worker-wide, exactly as before."""
    disaggregation_declined: bool = False
    """Set when this job was pulled back out of the disaggregated pipeline to run monolithically instead.

    Latched when the orchestrator re-routes a job whose disaggregated stage kept failing resource-class (device
    out-of-memory) past the defer window. While set, the disaggregation-eligibility predicate treats the job as
    monolithic (so it is charged its full footprint and never re-claimed into the pipeline). Cleared naturally
    when the job leaves the tracker. See :meth:`requeue_disaggregated_for_monolithic`."""
    chain_context: ChainExecutionContext | None = None
    """The chain-stage state for this job's unit of work, or None for jobs registered outside the pop path.

    Built at registration from the job's requested stages (generate, optional post-processing, safety,
    submit) and advanced in lockstep with :class:`JobStage` transitions, so any observer can read which
    stage of the routing plan the job is in without re-deriving it from queue membership."""


@dataclass(frozen=True)
class JobTrackerSnapshot:
    """Immutable snapshot of the job tracker state."""

    jobs_lookup: dict[ImageGenerateJobPopResponse, HordeJobInfo]
    jobs_in_progress: tuple[ImageGenerateJobPopResponse, ...]
    job_faults: dict[GenerationID, list[GenMetadataEntry]]
    jobs_pending_safety_check: tuple[HordeJobInfo, ...]
    jobs_being_safety_checked: tuple[HordeJobInfo, ...]
    jobs_pending_post_processing: tuple[HordeJobInfo, ...]
    jobs_being_post_processed: tuple[HordeJobInfo, ...]
    jobs_pending_submit: tuple[HordeJobInfo, ...]
    jobs_pending_inference: tuple[ImageGenerateJobPopResponse, ...]
    job_pop_timestamps: dict[ImageGenerateJobPopResponse, float]

    num_jobs_faulted: int
    total_num_completed_jobs: int
    max_pending_megapixelsteps: int
    triggered_max_pending_megapixelsteps: bool
    triggered_max_pending_megapixelsteps_time: float
    last_job_submitted_time: float


class JobTracker:
    """Owner of all job lifecycle state, keyed by ``GenerationID``.

    The mutation methods are ``async def`` so existing call sites (and any
    future cross-loop implementation) keep working, but each mutation is a
    single synchronous operation; there is no internal actor or queue.
    """

    def __init__(self) -> None:
        """Initialize the job tracker."""
        self._jobs: dict[GenerationID, TrackedJob] = {}
        self._job_faults: dict[GenerationID, list[GenMetadataEntry]] = {}

        self._num_jobs_faulted = 0
        self._total_num_completed_jobs = 0
        self._max_pending_megapixelsteps = 25
        self._triggered_max_pending_megapixelsteps_flag = False
        self._triggered_max_pending_megapixelsteps_time_value = 0.0
        self._last_job_submitted_time = time.time()

        self._sequence_counter = 0
        self._finalize_observer: Callable[[TrackedJob, HordeJobInfo], None] | None = None

        # Circuit-breaker / self-throttle bookkeeping (raw counts + timestamps; the scheduler and process
        # manager apply the configured thresholds). A model the device genuinely cannot run faults every
        # attempt no matter how it is isolated; tracked here so the worker can stop popping/admitting it
        # before the horde server forces maintenance for "dropping too many jobs".
        # Keyed by (model, device_index): the over-budget fault streak is per card, so a model unservable on
        # a small card is still advertised/run on a larger one. device_index is None on a single-GPU host, so
        # the keying collapses to one entry per model, behaviourally identical to the prior model-only keys.
        self._model_overbudget_fault_counts: dict[tuple[str, int | None], int] = {}
        self._model_last_overbudget_fault_time: dict[tuple[str, int | None], float] = {}
        self._recent_resource_fault_times: list[float] = []
        # Post-processing over-commit faults (the planner's unhostable-peak faults and watchdog-reaped
        # post-processing-stage stalls), for the feature-level circuit breaker that disables post-processing
        # after a run of them. Feature-level, not per-model, so a flat timestamp list suffices.
        self._recent_post_processing_fault_times: list[float] = []

        # Cumulative completed inference results per card, for the dashboard's per-card jobs/hr trend. Keyed
        # like the fault streaks (device_index, None on a single-GPU host); monotonic across the session.
        self._card_inference_results: dict[int | None, int] = {}

        # Defaults to one attempt (no retry: the pre-resiliency behaviour) so a directly-constructed
        # tracker faults terminally; the worker opts into bounded retry via set_retry_policy().
        self._max_inference_attempts = 1

    def set_retry_policy(self, max_inference_attempts: int) -> None:
        """Set how many inference attempts a job may have before it is reported faulted (clamped to >= 1)."""
        self._max_inference_attempts = max(1, max_inference_attempts)

    # region circuit-breaker / self-throttle accounting

    _RESOURCE_FAULT_RETENTION_SECONDS = 3600.0
    """How long resource-fault timestamps are retained for the self-throttle window query."""

    def _record_resource_fault(self, model: str | None, *, device_index: int | None = None) -> None:
        """Record a terminal resource/OOM fault (per-(model, card) streak + global timestamp for throttle).

        Args:
            model: The faulting job's model, or None (no per-model streak then).
            device_index: The card the job ran on, or None on a single-GPU host (worker-wide streak).
        """
        now = time.time()
        self._recent_resource_fault_times.append(now)
        cutoff = now - self._RESOURCE_FAULT_RETENTION_SECONDS
        self._recent_resource_fault_times = [t for t in self._recent_resource_fault_times if t >= cutoff]
        if model is None:
            return
        key = (model, device_index)
        self._model_overbudget_fault_counts[key] = self._model_overbudget_fault_counts.get(key, 0) + 1
        self._model_last_overbudget_fault_time[key] = now

    def record_model_inference_success(self, model: str | None, *, device_index: int | None = None) -> None:
        """Clear a model's over-budget fault streak after it produces a result on the card it ran on.

        Args:
            model: The succeeding job's model.
            device_index: The card it ran on (clears just that card's streak); None clears every card's
                streak for the model (a single-GPU success, or an unattributed one).
        """
        if model is None:
            return
        if device_index is None:
            keys_to_clear = [key for key in self._model_overbudget_fault_counts if key[0] == model]
            keys_to_clear += [
                key for key in self._model_last_overbudget_fault_time if key[0] == model and key not in keys_to_clear
            ]
        else:
            keys_to_clear = [(model, device_index)]
        cleared = False
        for key in keys_to_clear:
            if self._model_overbudget_fault_counts.pop(key, None) is not None:
                cleared = True
            self._model_last_overbudget_fault_time.pop(key, None)
        if cleared:
            logger.debug(f"Model {model} produced a result; clearing its over-budget fault streak.")

    def get_model_overbudget_fault_count(self, model: str | None, *, device_index: int | None = None) -> int:
        """Return the consecutive terminal over-budget fault count for ``model`` (0 if none).

        With ``device_index`` given, the streak for that card; with None, the worst (max) streak across every
        card (so the single-GPU worker-wide reading, where there is one entry, is unchanged).
        """
        if model is None:
            return 0
        if device_index is not None:
            return self._model_overbudget_fault_counts.get((model, device_index), 0)
        counts = [
            count for (key_model, _device), count in self._model_overbudget_fault_counts.items() if key_model == model
        ]
        return max(counts) if counts else 0

    def model_last_overbudget_fault_time(self, model: str | None, *, device_index: int | None = None) -> float | None:
        """Return the wall-clock time of ``model``'s last terminal over-budget fault, or None.

        With ``device_index`` given, the time for that card; with None, the most recent across every card.
        """
        if model is None:
            return None
        if device_index is not None:
            return self._model_last_overbudget_fault_time.get((model, device_index))
        times = [t for (key_model, _device), t in self._model_last_overbudget_fault_time.items() if key_model == model]
        return max(times) if times else None

    def count_recent_resource_faults(self, window_seconds: float, now: float | None = None) -> int:
        """Return how many terminal resource/OOM faults occurred within the last ``window_seconds``."""
        current = time.time() if now is None else now
        cutoff = current - window_seconds
        return sum(1 for t in self._recent_resource_fault_times if t >= cutoff)

    def note_post_processing_overcommit_fault(self) -> None:
        """Record one post-processing VRAM-over-commit fault for the feature-level circuit breaker.

        Fed from both the scheduler (a peak the planner cannot host, faulted at dispatch) and the process
        lifecycle (a slot reaped silent in post-processing). Feature-level, so no per-model streak is kept.
        """
        now = time.time()
        self._recent_post_processing_fault_times.append(now)
        cutoff = now - self._RESOURCE_FAULT_RETENTION_SECONDS
        self._recent_post_processing_fault_times = [t for t in self._recent_post_processing_fault_times if t >= cutoff]

    def count_recent_post_processing_faults(self, window_seconds: float, now: float | None = None) -> int:
        """Return how many post-processing over-commit faults occurred within the last ``window_seconds``."""
        current = time.time() if now is None else now
        cutoff = current - window_seconds
        return sum(1 for t in self._recent_post_processing_fault_times if t >= cutoff)

    def note_card_inference_result(self, device_index: int | None) -> None:
        """Count one completed inference result against the card it ran on (the per-card jobs/hr source)."""
        self._card_inference_results[device_index] = self._card_inference_results.get(device_index, 0) + 1

    def get_card_inference_results(self, device_index: int | None) -> int:
        """Return cumulative completed inference results on ``device_index`` this session (0 if none)."""
        return self._card_inference_results.get(device_index, 0)

    # endregion

    def set_finalize_observer(self, observer: Callable[[TrackedJob, HordeJobInfo], None]) -> None:
        """Register a callback invoked with each job's final tracked state at finalize time.

        Used by the run-metrics aggregator to fold per-job stage latencies into the
        worker-wide metrics snapshot without wrapping tracker methods.
        """
        self._finalize_observer = observer

    # region internal helpers

    def _next_sequence(self) -> int:
        self._sequence_counter += 1
        return self._sequence_counter

    def _tracked_by_id(self, job_id: GenerationID | None) -> TrackedJob | None:
        if job_id is None:
            return None
        return self._jobs.get(job_id)

    def _tracked_for(self, job: ImageGenerateJobPopResponse) -> TrackedJob | None:
        return self._tracked_by_id(job.id_)

    def _set_stage(self, tracked: TrackedJob, new_stage: JobStage) -> bool:
        """Move a job to a new stage, validating the transition.

        Returns:
            True if the transition was applied, False if it was illegal (and logged).
        """
        if tracked.stage == new_stage:
            return True
        if new_stage not in _ALLOWED_TRANSITIONS[tracked.stage]:
            logger.error(
                f"Illegal job stage transition for job {tracked.job_id}: "
                f"{tracked.stage.name} -> {new_stage.name}. Transition refused.",
            )
            return False
        old_stage = tracked.stage
        tracked.stage = new_stage
        tracked.stage_sequence = self._next_sequence()
        now = time.time()
        tracked.stage_timestamps.setdefault(new_stage.name, now)
        tracked.current_stage_since = now
        self._sync_chain_with_stage(tracked, old_stage, new_stage)
        return True

    def _advance_chain(self, tracked: TrackedJob, progress: GENERATION_PROGRESS) -> None:
        """Advance a job's chain context from a generation-progress observation, tolerating refusals.

        A refusal means the chain's routing plan and the worker's queue bookkeeping disagree (e.g. a
        recovery path re-entered a stage the chain considers finished); the queue bookkeeping remains
        authoritative, so the disagreement is logged rather than raised.
        """
        context = tracked.chain_context
        if context is None:
            return
        try:
            context.advance_for_progress(progress)
        except ChainConsistencyError as e:
            logger.warning(f"Chain state for job {tracked.job_id} refused progress {progress}: {e}")

    def _sync_chain_with_stage(self, tracked: TrackedJob, old_stage: JobStage, new_stage: JobStage) -> None:
        """Mirror a queue-stage transition into the job's chain context.

        The chain is the descriptive routing plan; :class:`JobStage` (queue membership) remains the
        executor of record. Milestone progress values are derived from the transition: entering a working
        stage marks its node executing, and entering the next queue marks the preceding node's completion
        milestone. Fault paths that jump to ``PENDING_SUBMIT`` are recognized by the safety node not being
        mid-execution, leaving the chain to record only the stages that genuinely ran.
        """
        if tracked.chain_context is None:
            return
        snapshot = tracked.chain_context.snapshot()

        if new_stage == JobStage.INFERENCE_IN_PROGRESS:
            self._advance_chain(tracked, GENERATION_PROGRESS.GENERATING)
            return

        if new_stage == JobStage.PENDING_POST_PROCESSING and old_stage != JobStage.POST_PROCESSING:
            self._advance_chain(tracked, GENERATION_PROGRESS.GENERATION_COMPLETE)
            return

        if new_stage == JobStage.POST_PROCESSING:
            self._advance_chain(tracked, GENERATION_PROGRESS.POST_PROCESSING)
            return

        if new_stage == JobStage.PENDING_SAFETY_CHECK:
            self._advance_chain(tracked, GENERATION_PROGRESS.GENERATION_COMPLETE)
            if snapshot.get(POST_PROCESS_STAGE_NAME) == CHAIN_NODE_STATE.EXECUTING:
                self._advance_chain(tracked, GENERATION_PROGRESS.POST_PROCESSING_COMPLETE)
            return

        if new_stage == JobStage.SAFETY_CHECKING:
            self._advance_chain(tracked, GENERATION_PROGRESS.SAFETY_CHECKING)
            return

        if (
            new_stage == JobStage.PENDING_SUBMIT
            and snapshot.get(SAFETY_CHECK_STAGE_NAME) == CHAIN_NODE_STATE.EXECUTING
        ):
            self._advance_chain(tracked, GENERATION_PROGRESS.SAFETY_CHECK_COMPLETE)
            return

    def _register(
        self,
        *,
        job_id: GenerationID,
        sdk_api_job_info: ImageGenerateJobPopResponse,
        stage: JobStage,
        job_info: HordeJobInfo | None,
        time_popped: float | None,
    ) -> TrackedJob:
        """Register a new tracked job, replacing (with a warning) any entry with the same ID."""
        existing = self._jobs.get(job_id)
        if existing is not None:
            logger.warning(
                f"Job {job_id} is already tracked (stage {existing.stage.name}); replacing the old entry. "
                "This can happen when a canned scenario recycles job IDs.",
            )
            del self._jobs[job_id]

        # A chain context is only meaningful when the job enters at the start of its routing plan; a job
        # re-registered mid-flow (orphan recovery) would have to fake the stages it never traversed here.
        chain_context: ChainExecutionContext | None = None
        if stage == JobStage.PENDING_INFERENCE:
            chain_context = ChainExecutionContext(
                image_generation_flow(
                    post_processing=bool(sdk_api_job_info.payload.post_processing),
                    safety_check=True,
                ),
            )

        tracked = TrackedJob(
            job_id=job_id,
            sdk_api_job_info=sdk_api_job_info,
            stage=stage,
            job_info=job_info,
            time_popped=time_popped,
            pop_order=self._next_sequence(),
            stage_sequence=self._next_sequence(),
            chain_context=chain_context,
        )
        now = time.time()
        tracked.stage_timestamps[stage.name] = now
        tracked.current_stage_since = now
        self._jobs[job_id] = tracked
        return tracked

    def _jobs_in_stage(self, *stages: JobStage) -> list[TrackedJob]:
        """Return tracked jobs in the given stage(s), in stage-entry (FIFO) order."""
        return sorted(
            (t for t in self._jobs.values() if t.stage in stages),
            key=lambda t: t.stage_sequence,
        )

    # endregion

    # region read-only views

    def get_tracked_job(self, job_id: GenerationID) -> TrackedJob | None:
        """Return the TrackedJob for an ID, or None if it is not tracked."""
        return self._tracked_by_id(job_id)

    def get_stage(self, job_id: GenerationID) -> JobStage | None:
        """Return the current stage of a job, or None if it is not tracked."""
        tracked = self._tracked_by_id(job_id)
        return tracked.stage if tracked is not None else None

    def tracked_jobs(self) -> tuple[TrackedJob, ...]:
        """Return active tracked jobs in lifecycle order for read-only observers."""
        return tuple(sorted(self._jobs.values(), key=lambda tracked: (tracked.stage_sequence, tracked.pop_order)))

    def stage_age_summary(self, *, now: float | None = None) -> dict[JobStage, tuple[int, float]]:
        """Return per-stage ``(count, oldest_age_seconds)`` for every non-empty stage.

        The age is measured from :attr:`TrackedJob.current_stage_since` (the latest entry into the stage),
        so a job that genuinely sits in a stage shows a growing age while normal throughput stays near
        zero. Surfaced in the periodic status dump so a downstream stall (e.g. jobs aging in
        ``SAFETY_CHECKING`` while inference keeps finishing) is visible at a glance instead of having to be
        reconstructed from raw counts after the fact.
        """
        reference = time.time() if now is None else now
        summary: dict[JobStage, tuple[int, float]] = {}
        for tracked in self._jobs.values():
            if tracked.stage == JobStage.DETACHED:
                continue
            age = max(0.0, reference - tracked.current_stage_since) if tracked.current_stage_since else 0.0
            count, oldest = summary.get(tracked.stage, (0, 0.0))
            summary[tracked.stage] = (count + 1, max(oldest, age))
        return summary

    @property
    def jobs_lookup(self) -> dict[ImageGenerateJobPopResponse, HordeJobInfo]:
        """Return a mapping of pop responses to job info for all tracked jobs that carry job info."""
        return {t.sdk_api_job_info: t.job_info for t in self._jobs.values() if t.job_info is not None}

    @property
    def jobs_in_progress(self) -> tuple[ImageGenerateJobPopResponse, ...]:
        """Return the pop responses for jobs currently being inferred."""
        return tuple(t.sdk_api_job_info for t in self._jobs_in_stage(JobStage.INFERENCE_IN_PROGRESS))

    @property
    def job_faults(self) -> dict[GenerationID, list[GenMetadataEntry]]:
        """Return a copy of the job faults dictionary."""
        return {k: list(v) for k, v in self._job_faults.items()}

    @property
    def jobs_pending_safety_check(self) -> tuple[HordeJobInfo, ...]:
        """Return the `HordeJobInfo` objects for jobs pending safety check."""
        return tuple(t.job_info for t in self._jobs_in_stage(JobStage.PENDING_SAFETY_CHECK) if t.job_info is not None)

    @property
    def jobs_being_safety_checked(self) -> tuple[HordeJobInfo, ...]:
        """Return the `HordeJobInfo` objects for jobs currently being safety checked."""
        return tuple(t.job_info for t in self._jobs_in_stage(JobStage.SAFETY_CHECKING) if t.job_info is not None)

    @property
    def jobs_pending_post_processing(self) -> tuple[HordeJobInfo, ...]:
        """Return the `HordeJobInfo` objects for jobs pending the dedicated post-processing lane."""
        return tuple(
            t.job_info for t in self._jobs_in_stage(JobStage.PENDING_POST_PROCESSING) if t.job_info is not None
        )

    @property
    def jobs_being_post_processed(self) -> tuple[HordeJobInfo, ...]:
        """Return the `HordeJobInfo` objects for jobs currently being post-processed."""
        return tuple(t.job_info for t in self._jobs_in_stage(JobStage.POST_PROCESSING) if t.job_info is not None)

    @property
    def jobs_pending_submit(self) -> tuple[HordeJobInfo, ...]:
        """Return the `HordeJobInfo` objects for jobs pending submit."""
        return tuple(t.job_info for t in self._jobs_in_stage(JobStage.PENDING_SUBMIT) if t.job_info is not None)

    @property
    def jobs_pending_inference(self) -> tuple[ImageGenerateJobPopResponse, ...]:
        """Return the pop responses for queued jobs, in pop order.

        This intentionally includes jobs whose inference is in progress; a job
        leaves this view only when its result (or fault) arrives. Queue-size
        accounting and scheduling look-ahead rely on this.
        """
        queued = [
            t
            for t in self._jobs.values()
            if t.stage in (JobStage.PENDING_INFERENCE, JobStage.INFERENCE_IN_PROGRESS) and t.time_popped is not None
        ]
        queued.sort(key=lambda t: t.pop_order)
        return tuple(t.sdk_api_job_info for t in queued)

    @property
    def job_pop_timestamps(self) -> dict[ImageGenerateJobPopResponse, float]:
        """Return a mapping of pop responses to the time they were popped."""
        return {t.sdk_api_job_info: t.time_popped for t in self._jobs.values() if t.time_popped is not None}

    @property
    def num_jobs_total(self) -> int:
        """Return the total number of jobs across all queued stages."""
        return sum(1 for t in self._jobs.values() if t.stage in _QUEUED_STAGES)

    @property
    def current_queue_size(self) -> int:
        """Return the current size of the inference queue (including jobs being inferred)."""
        return sum(
            1
            for t in self._jobs.values()
            if t.stage in (JobStage.PENDING_INFERENCE, JobStage.INFERENCE_IN_PROGRESS) and t.time_popped is not None
        )

    @property
    def total_num_completed_jobs(self) -> int:
        """Return the total number of completed jobs recorded this session."""
        return self._total_num_completed_jobs

    @property
    def num_jobs_faulted(self) -> int:
        """Return the total number of faulted jobs recorded this session."""
        return self._num_jobs_faulted

    def snapshot(self) -> JobTrackerSnapshot:
        """Return an immutable snapshot view of current job state."""
        return JobTrackerSnapshot(
            jobs_lookup=self.jobs_lookup,
            jobs_in_progress=self.jobs_in_progress,
            job_faults=self.job_faults,
            jobs_pending_safety_check=self.jobs_pending_safety_check,
            jobs_being_safety_checked=self.jobs_being_safety_checked,
            jobs_pending_post_processing=self.jobs_pending_post_processing,
            jobs_being_post_processed=self.jobs_being_post_processed,
            jobs_pending_submit=self.jobs_pending_submit,
            jobs_pending_inference=self.jobs_pending_inference,
            job_pop_timestamps=self.job_pop_timestamps,
            num_jobs_faulted=self._num_jobs_faulted,
            total_num_completed_jobs=self._total_num_completed_jobs,
            max_pending_megapixelsteps=self._max_pending_megapixelsteps,
            triggered_max_pending_megapixelsteps=self._triggered_max_pending_megapixelsteps_flag,
            triggered_max_pending_megapixelsteps_time=self._triggered_max_pending_megapixelsteps_time_value,
            last_job_submitted_time=self._last_job_submitted_time,
        )

    # endregion

    # region megapixelstep throttling state

    def set_performance_mode_thresholds(self, max_pending_megapixelsteps: int) -> None:
        """Set the performance mode thresholds."""
        self._max_pending_megapixelsteps = max_pending_megapixelsteps

    def reset_megapixelstep_trigger(self) -> None:
        """Reset the megapixelstep trigger."""
        self._triggered_max_pending_megapixelsteps_flag = False

    def should_wait_for_pending_megapixelsteps(self) -> bool:
        """Return whether the system should wait based on the currently pending megapixelsteps."""
        pending_megapixelsteps = self.get_pending_megapixelsteps()
        return JobQueueAnalyzer.should_wait_for_pending_megapixelsteps(
            pending_megapixelsteps,
            self._max_pending_megapixelsteps,
        )

    def get_pending_megapixelsteps(self) -> int:
        """Return the number of pending megapixelsteps."""
        return JobQueueAnalyzer.calculate_pending_job_magnitude(
            self.jobs_pending_inference,
            len(self.jobs_pending_submit),
        )

    @property
    def _triggered_max_pending_megapixelsteps(self) -> bool:
        return self._triggered_max_pending_megapixelsteps_flag

    @_triggered_max_pending_megapixelsteps.setter
    def _triggered_max_pending_megapixelsteps(self, value: bool) -> None:
        self._triggered_max_pending_megapixelsteps_flag = value

    @property
    def _triggered_max_pending_megapixelsteps_time(self) -> float:
        return self._triggered_max_pending_megapixelsteps_time_value

    @_triggered_max_pending_megapixelsteps_time.setter
    def _triggered_max_pending_megapixelsteps_time(self, value: float) -> None:
        self._triggered_max_pending_megapixelsteps_time_value = value

    # endregion

    # region lifecycle mutations

    async def record_popped_job(
        self,
        job_pop_response: ImageGenerateJobPopResponse,
        time_popped: float | None = None,
    ) -> HordeJobInfo:
        """Record a popped job and return its corresponding HordeJobInfo.

        Raises:
            ValueError: If the pop response has no generation ID.
        """
        if job_pop_response.id_ is None:
            raise ValueError("Cannot track a popped job without a generation ID")

        stamp = time.time() if time_popped is None else time_popped
        job_info = HordeJobInfo(
            sdk_api_job_info=job_pop_response,
            state=None,
            time_popped=stamp,
        )
        self._register(
            job_id=job_pop_response.id_,
            sdk_api_job_info=job_pop_response,
            stage=JobStage.PENDING_INFERENCE,
            job_info=job_info,
            time_popped=stamp,
        )
        self._job_faults.setdefault(job_pop_response.id_, [])
        return job_info

    async def mark_inference_started(
        self,
        job: ImageGenerateJobPopResponse,
        *,
        device_index: int | None = None,
    ) -> None:
        """Mark a job as started for inference, recording the card it was dispatched to.

        Args:
            job: The job entering inference.
            device_index: The card it was dispatched to (multi-GPU), or None on a single-GPU host. Stored so
                its over-budget fault streak (and the success that clears it) is keyed to that card.
        """
        tracked = self._tracked_for(job)
        if tracked is None:
            if job.id_ is None:
                logger.error("Cannot mark a job without a generation ID as in progress")
                return
            logger.debug(f"Job {job.id_} was not tracked when inference started; registering it now")
            self._register(
                job_id=job.id_,
                sdk_api_job_info=job,
                stage=JobStage.INFERENCE_IN_PROGRESS,
                job_info=None,
                time_popped=None,
            )
            new_tracked = self._tracked_for(job)
            if new_tracked is not None:
                new_tracked.last_dispatched_device_index = device_index
            return
        tracked.last_dispatched_device_index = device_index
        self._set_stage(tracked, JobStage.INFERENCE_IN_PROGRESS)

    async def release_in_progress(self, job: ImageGenerateJobPopResponse) -> bool:
        """Release a job from the in-progress state back to pending inference.

        A job that was marked in progress without ever being formally popped
        (no pop timestamp) has nothing to return to, so releasing it forgets
        it entirely.
        """
        tracked = self._tracked_for(job)
        if tracked is None or tracked.stage != JobStage.INFERENCE_IN_PROGRESS:
            return False
        if tracked.time_popped is None:
            del self._jobs[tracked.job_id]
            return True
        return self._set_stage(tracked, JobStage.PENDING_INFERENCE)

    def mark_disaggregation_decoding(self, job: ImageGenerateJobPopResponse) -> bool:
        """Move a disaggregated job from inference to the decoding stage, freeing its sampler slot.

        Called the instant the sampler returns its latent (``SampleSliceResult``): the job leaves the
        inference concurrency cap (``jobs_in_progress`` counts only ``INFERENCE_IN_PROGRESS``) so the freed
        sampler is schedulable again, while the job stays tracked and in-flight through the image lane's
        decode. Returns False if the job is not currently in ``INFERENCE_IN_PROGRESS``.
        """
        tracked = self._tracked_for(job)
        if tracked is None or tracked.stage != JobStage.INFERENCE_IN_PROGRESS:
            return False
        return self._set_stage(tracked, JobStage.DISAGGREGATION_DECODING)

    def requeue_disaggregated_for_monolithic(self, job: ImageGenerateJobPopResponse) -> bool:
        """Return an in-flight disaggregated job to ``PENDING_INFERENCE`` for a monolithic re-dispatch.

        Called when the disaggregation orchestrator re-routes a job whose stage kept failing resource-class
        (device out-of-memory) past the defer window: the job is still owned/tracked (its sampler pin already
        released) and must run whole instead. Latches :attr:`TrackedJob.disaggregation_declined` so the
        scheduler's eligibility predicate keeps it monolithic on the re-claim, then moves it back to the
        pending-inference queue. Valid only from ``INFERENCE_IN_PROGRESS`` or ``DISAGGREGATION_DECODING``, and
        only for a formally popped job (one with a queue position to return to); returns False otherwise.
        """
        tracked = self._tracked_for(job)
        if tracked is None:
            return False
        if tracked.stage not in (JobStage.INFERENCE_IN_PROGRESS, JobStage.DISAGGREGATION_DECODING):
            return False
        if tracked.time_popped is None:
            return False
        tracked.disaggregation_declined = True
        return self._set_stage(tracked, JobStage.PENDING_INFERENCE)

    def is_disaggregation_declined(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether this job was re-routed out of the disaggregated pipeline to run monolithically (a peek)."""
        tracked = self._tracked_for(job)
        return tracked is not None and tracked.disaggregation_declined

    async def drop_pending_inference(self, job: ImageGenerateJobPopResponse) -> bool:
        """Drop a job from the pending inference queue (it remains tracked, detached)."""
        tracked = self._tracked_for(job)
        if tracked is None or tracked.stage != JobStage.PENDING_INFERENCE:
            return False
        return self._set_stage(tracked, JobStage.DETACHED)

    async def drop_pending_inference_by_id(self, job_id: GenerationID) -> bool:
        """Drop a job from the pending inference queue by its ID (it remains tracked, detached)."""
        tracked = self._tracked_by_id(job_id)
        if tracked is None or tracked.stage != JobStage.PENDING_INFERENCE:
            return False
        return self._set_stage(tracked, JobStage.DETACHED)

    async def queue_for_safety(self, job_info: HordeJobInfo) -> None:
        """Queue a job for safety checking."""
        tracked = self._tracked_for(job_info.sdk_api_job_info)
        if tracked is None:
            if job_info.sdk_api_job_info.id_ is None:
                logger.error("Refusing to queue a job without a generation ID for safety; it cannot be submitted")
                return
            logger.debug(
                f"Job {job_info.sdk_api_job_info.id_} was not tracked when queued for safety; registering it now",
            )
            self._register(
                job_id=job_info.sdk_api_job_info.id_,
                sdk_api_job_info=job_info.sdk_api_job_info,
                stage=JobStage.PENDING_SAFETY_CHECK,
                job_info=job_info,
                time_popped=None,
            )
            return
        # Validate the transition *before* adopting the new result. A stale or duplicate inference
        # result for a job that already reached the terminal PENDING_SUBMIT stage would otherwise
        # overwrite its safety-checked job_info with a fresh, pre-safety one (censored=None) while the
        # refused transition left the job in PENDING_SUBMIT, producing an un-submittable "poison" job
        # that wedges the submit loop. Only adopt the new job_info if the move is actually legal.
        if not self._set_stage(tracked, JobStage.PENDING_SAFETY_CHECK):
            logger.warning(
                f"Refusing to re-queue job {job_info.sdk_api_job_info.id_} for safety from stage "
                f"{tracked.stage.name}; keeping its existing result (likely a stale/duplicate result).",
            )
            return
        tracked.job_info = job_info
        # Inference produced a result on the card it ran on: this model can run there, so clear that card's
        # over-budget fault streak (None on a single-GPU host clears the worker-wide streak, as before).
        self.record_model_inference_success(
            job_info.sdk_api_job_info.model,
            device_index=tracked.last_dispatched_device_index,
        )
        # One inference result landed on this card; feed the per-card jobs/hr trend (keyed like the streak,
        # None on a single-GPU host). Counted on the result message so a per-card rate excludes crash faults.
        self.note_card_inference_result(tracked.last_dispatched_device_index)

    async def queue_for_post_processing(self, job_info: HordeJobInfo) -> None:
        """Queue an inference-complete job (that requested post-processing) for the post-processing lane.

        Inference genuinely produced these (raw) images, so the model-inference success bookkeeping is
        recorded here exactly as :meth:`queue_for_safety` does for the no-post-processing path; the job's
        move to the safety stage after post-processing (:meth:`queue_for_safety_post_processed`) therefore
        does not re-record it.
        """
        tracked = self._tracked_for(job_info.sdk_api_job_info)
        if tracked is None:
            if job_info.sdk_api_job_info.id_ is None:
                logger.error(
                    "Refusing to queue a job without a generation ID for post-processing; it cannot be submitted",
                )
                return
            logger.debug(
                f"Job {job_info.sdk_api_job_info.id_} was not tracked when queued for post-processing; "
                "registering it now",
            )
            self._register(
                job_id=job_info.sdk_api_job_info.id_,
                sdk_api_job_info=job_info.sdk_api_job_info,
                stage=JobStage.PENDING_POST_PROCESSING,
                job_info=job_info,
                time_popped=None,
            )
            return
        if not self._set_stage(tracked, JobStage.PENDING_POST_PROCESSING):
            logger.warning(
                f"Refusing to re-queue job {job_info.sdk_api_job_info.id_} for post-processing from stage "
                f"{tracked.stage.name}; keeping its existing result (likely a stale/duplicate result).",
            )
            return
        tracked.job_info = job_info
        self.record_model_inference_success(
            job_info.sdk_api_job_info.model,
            device_index=tracked.last_dispatched_device_index,
        )
        self.note_card_inference_result(tracked.last_dispatched_device_index)

    async def queue_for_safety_post_processed(self, job_info: HordeJobInfo) -> None:
        """Move a post-processed job on to the safety stage, adopting its post-processed images.

        The inference-success bookkeeping was already recorded when the job entered post-processing, so
        (unlike :meth:`queue_for_safety`) this only validates the transition and adopts the new result.
        """
        tracked = self._tracked_for(job_info.sdk_api_job_info)
        if tracked is None:
            if job_info.sdk_api_job_info.id_ is None:
                logger.error("Refusing to queue a post-processed job without a generation ID for safety")
                return
            self._register(
                job_id=job_info.sdk_api_job_info.id_,
                sdk_api_job_info=job_info.sdk_api_job_info,
                stage=JobStage.PENDING_SAFETY_CHECK,
                job_info=job_info,
                time_popped=None,
            )
            return
        if not self._set_stage(tracked, JobStage.PENDING_SAFETY_CHECK):
            logger.warning(
                f"Refusing to move post-processed job {job_info.sdk_api_job_info.id_} to safety from stage "
                f"{tracked.stage.name}; keeping its existing result.",
            )
            return
        tracked.job_info = job_info

    async def begin_post_processing(self, job_info: HordeJobInfo) -> None:
        """Mark a job as sent to the post-processing process."""
        tracked = self._tracked_for(job_info.sdk_api_job_info)
        if tracked is None:
            logger.error(
                f"Job {job_info.sdk_api_job_info.id_} is not tracked; cannot begin its post-processing",
            )
            return
        self._set_stage(tracked, JobStage.POST_PROCESSING)

    async def abandon_pending_post_processing(self, job_info: HordeJobInfo) -> None:
        """Abandon a job from the pending post-processing state (it remains tracked, detached)."""
        tracked = self._tracked_for(job_info.sdk_api_job_info)
        if tracked is None or tracked.stage != JobStage.PENDING_POST_PROCESSING:
            return
        self._set_stage(tracked, JobStage.DETACHED)

    async def requeue_one_being_post_processed(self, job_id: GenerationID) -> bool:
        """Move a single job from POST_PROCESSING back to PENDING_POST_PROCESSING for a fresh attempt.

        Used by the post-processing-orphan watchdog when a job's post-processing result was lost (the
        process was replaced, or a result message was dropped). Returns True if the job was in
        POST_PROCESSING and was requeued, False otherwise.
        """
        tracked = self._tracked_by_id(job_id)
        if tracked is None or tracked.stage != JobStage.POST_PROCESSING:
            return False
        return self._set_stage(tracked, JobStage.PENDING_POST_PROCESSING)

    async def take_being_post_processed(self, job_id: GenerationID) -> HordeJobInfo | None:
        """Take a job that is currently being post-processed by its ID, detaching it."""
        tracked = self._tracked_by_id(job_id)
        if tracked is None or tracked.stage != JobStage.POST_PROCESSING:
            return None
        self._set_stage(tracked, JobStage.DETACHED)
        return tracked.job_info

    async def queue_for_submit(self, job_info: HordeJobInfo) -> None:
        """Queue a job for submission."""
        tracked = self._tracked_for(job_info.sdk_api_job_info)
        if tracked is None:
            if job_info.sdk_api_job_info.id_ is None:
                logger.error("Refusing to queue a job without a generation ID for submit; it cannot be submitted")
                return
            logger.debug(
                f"Job {job_info.sdk_api_job_info.id_} was not tracked when queued for submit; registering it now",
            )
            self._register(
                job_id=job_info.sdk_api_job_info.id_,
                sdk_api_job_info=job_info.sdk_api_job_info,
                stage=JobStage.PENDING_SUBMIT,
                job_info=job_info,
                time_popped=None,
            )
            return
        # As in queue_for_safety: only adopt the new result if the transition is legal, so a refused
        # move can never leave the job carrying a mismatched job_info.
        if not self._set_stage(tracked, JobStage.PENDING_SUBMIT):
            logger.warning(
                f"Refusing to re-queue job {job_info.sdk_api_job_info.id_} for submit from stage "
                f"{tracked.stage.name}; keeping its existing result.",
            )
            return
        tracked.job_info = job_info

    async def fault_post_inference_job(self, job_info: HordeJobInfo, *, reason: str) -> None:
        """Fault a job after inference produced images, without submitting those images.

        Downstream worker-owned stages must either honor the advertised contract or report a no-image fault so
        the horde reissues the job. Inference completion was already counted when the raw images arrived, so
        this method only clears those images, records a diagnostic, and moves the tracked job to submit.
        """
        tracked = self._tracked_for(job_info.sdk_api_job_info)
        if tracked is None:
            logger.error(f"Job {job_info.sdk_api_job_info.id_} is not tracked; cannot fault it")
            return
        job_info.fault_job()
        tracked.job_info = job_info
        self._job_faults.setdefault(tracked.job_id, []).append(
            GenMetadataEntry(
                type=METADATA_TYPE.information,
                value=METADATA_VALUE.see_ref,
                ref=f"faulted after inference: {reason}"[:255],
            ),
        )
        if not self._set_stage(tracked, JobStage.PENDING_SUBMIT):
            logger.warning(
                f"Refusing to fault post-inference job {job_info.sdk_api_job_info.id_} from stage "
                f"{tracked.stage.name}; keeping its existing state.",
            )

    async def begin_safety_check(self, job_info: HordeJobInfo) -> None:
        """Begin the safety check process for a job."""
        tracked = self._tracked_for(job_info.sdk_api_job_info)
        if tracked is None:
            logger.error(
                f"Job {job_info.sdk_api_job_info.id_} is not tracked; cannot begin its safety check",
            )
            return
        self._set_stage(tracked, JobStage.SAFETY_CHECKING)

    async def abandon_pending_safety(self, job_info: HordeJobInfo) -> None:
        """Abandon a job from the pending safety check state (it remains tracked, detached)."""
        tracked = self._tracked_for(job_info.sdk_api_job_info)
        if tracked is None or tracked.stage != JobStage.PENDING_SAFETY_CHECK:
            return
        self._set_stage(tracked, JobStage.DETACHED)

    async def requeue_being_safety_checked(self) -> None:
        """Requeue all jobs that are currently being safety checked."""
        for tracked in self._jobs_in_stage(JobStage.SAFETY_CHECKING):
            self._set_stage(tracked, JobStage.PENDING_SAFETY_CHECK)

    async def requeue_one_being_safety_checked(self, job_id: GenerationID) -> bool:
        """Move a single job from SAFETY_CHECKING back to PENDING_SAFETY_CHECK, so it is re-evaluated.

        Used by the safety-orphan watchdog when a job's safety result was lost (the safety process was
        replaced, or a result message was dropped): the job is sent back to the front of the safety queue
        for a fresh check rather than being stranded in SAFETY_CHECKING forever. Its images are preserved
        so they are actually re-checked, not silently submitted unchecked. Returns True if the job was in
        SAFETY_CHECKING and was requeued, False otherwise.
        """
        tracked = self._tracked_by_id(job_id)
        if tracked is None or tracked.stage != JobStage.SAFETY_CHECKING:
            return False
        return self._set_stage(tracked, JobStage.PENDING_SAFETY_CHECK)

    async def take_being_safety_checked(self, job_id: GenerationID) -> HordeJobInfo | None:
        """Take a job that is currently being safety checked by its ID, detaching it."""
        tracked = self._tracked_by_id(job_id)
        if tracked is None or tracked.stage != JobStage.SAFETY_CHECKING:
            return None
        self._set_stage(tracked, JobStage.DETACHED)
        return tracked.job_info

    async def record_source_image_fault(self, job_id: GenerationID, entry: GenMetadataEntry) -> None:
        """Record a fault for a job."""
        self._job_faults.setdefault(job_id, []).append(entry)

    async def clear_faults_for_job(self, job_id: GenerationID) -> None:
        """Clear all recorded faults for a job."""
        self._job_faults.pop(job_id, None)

    async def get_faults_for_job(self, job_id: GenerationID) -> list[GenMetadataEntry]:
        """Get all recorded faults for a job."""
        return list(self._job_faults.get(job_id, []))

    async def increment_jobs_faulted(self) -> None:
        """Increment the count of jobs that have encountered faults."""
        self._num_jobs_faulted += 1

    async def increment_jobs_completed(self) -> None:
        """Increment the count of jobs that have been completed."""
        self._total_num_completed_jobs += 1

    async def get_job_info(self, job: ImageGenerateJobPopResponse) -> HordeJobInfo | None:
        """Get the job info for a given job."""
        tracked = self._tracked_for(job)
        return tracked.job_info if tracked is not None else None

    async def set_job_time_to_download_aux_models(
        self,
        job: ImageGenerateJobPopResponse,
        time_elapsed: float | None,
    ) -> bool:
        """Set the time it took to download auxiliary models for a job."""
        tracked = self._tracked_for(job)
        if tracked is None or tracked.job_info is None:
            return False
        tracked.job_info.time_to_download_aux_models = time_elapsed
        return True

    async def get_time_popped(self, job: ImageGenerateJobPopResponse) -> float | None:
        """Get the time a job was popped from the queue."""
        tracked = self._tracked_for(job)
        return tracked.time_popped if tracked is not None else None

    async def ensure_submitted_job_info(self, completed_job_info: HordeJobInfo) -> HordeJobInfo:
        """Ensure the tracker knows about a completed job and mark its submit time."""
        tracked = self._tracked_for(completed_job_info.sdk_api_job_info)
        if tracked is None or tracked.job_info is None:
            logger.warning(
                f"Job {completed_job_info.sdk_api_job_info.id_} was not tracked at submit time; registering it now",
            )
            tracked = self._register(
                job_id=completed_job_info.sdk_api_job_info.id_,  # type: ignore[arg-type]
                sdk_api_job_info=completed_job_info.sdk_api_job_info,
                stage=JobStage.PENDING_SUBMIT,
                job_info=completed_job_info,
                time_popped=None,
            )
        job_info = tracked.job_info
        if job_info is None:
            job_info = completed_job_info
            tracked.job_info = job_info
        job_info.time_submitted = time.time()
        return job_info

    async def finalize_submitted(self, completed_job_info: HordeJobInfo) -> None:
        """Finalize a submitted job, removing it from the tracker entirely."""
        sdk_info = completed_job_info.sdk_api_job_info
        tracked = self._tracked_for(sdk_info)

        if tracked is None:
            logger.warning(f"Job {sdk_info.id_} not found in completed_jobs")
            return

        if tracked.stage != JobStage.PENDING_SUBMIT:
            logger.warning(
                f"Job {sdk_info.id_} was finalized from stage {tracked.stage.name} (expected PENDING_SUBMIT)",
            )

        tracked.stage_timestamps.setdefault("FINALIZED", time.time())

        # Close out the chain: a job that faulted terminally aborts (failing whichever stage was
        # executing); a successful one walks the submit stage to completion.
        if completed_job_info.state == GENERATION_STATE.faulted:
            self._advance_chain(tracked, GENERATION_PROGRESS.ABORTED)
        else:
            self._advance_chain(tracked, GENERATION_PROGRESS.SUBMITTING)
            self._advance_chain(tracked, GENERATION_PROGRESS.SUBMIT_COMPLETE)

        if self._finalize_observer is not None:
            try:
                self._finalize_observer(tracked, completed_job_info)
            except Exception as e:
                logger.warning(f"Job finalize observer failed: {type(e).__name__} {e}")

        del self._jobs[tracked.job_id]
        # Faults are kept independent of job lifetime, but a finalized job will never be read again,
        # so drop its fault list here to keep the fault map from growing for the worker's whole run.
        self._job_faults.pop(tracked.job_id, None)
        self._last_job_submitted_time = time.time()

    def is_degraded_dispatch_pending(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether this job's next dispatch should run degraded/isolated (a peek; does not consume)."""
        tracked = self._tracked_for(job)
        return tracked is not None and tracked.needs_degraded_dispatch

    def clear_degraded_dispatch(self, job: ImageGenerateJobPopResponse) -> None:
        """Consume the degraded-dispatch flag once the scheduler has dispatched the job degraded."""
        tracked = self._tracked_for(job)
        if tracked is not None:
            tracked.needs_degraded_dispatch = False

    def is_admitted_over_budget(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether this job was admitted against the VRAM budget's verdict (a peek; does not consume)."""
        tracked = self._tracked_for(job)
        return tracked is not None and tracked.admitted_over_budget

    def mark_admitted_over_budget(self, job: ImageGenerateJobPopResponse) -> None:
        """Record that the scheduler admitted this job despite the VRAM budget judging it unfit.

        A subsequent slot crash/hang on this job is then treated as a resource failure (see
        :attr:`TrackedJob.admitted_over_budget`), earning the bounded degraded/isolated retry rather
        than a plain re-dispatch onto another over-committed slot.
        """
        tracked = self._tracked_for(job)
        if tracked is not None:
            tracked.admitted_over_budget = True

    def mark_admitted_exclusive(self, job: ImageGenerateJobPopResponse) -> None:
        """Record that this over-budget job must run with the device to itself.

        See :attr:`TrackedJob.admitted_exclusive` for what exclusivity suppresses.
        """
        tracked = self._tracked_for(job)
        if tracked is not None:
            tracked.admitted_exclusive = True

    def is_admitted_exclusive(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether this job was admitted to run exclusively (over-budget, device to itself)."""
        tracked = self._tracked_for(job)
        return tracked is not None and tracked.admitted_exclusive

    def has_exclusive_job_in_progress(self) -> bool:
        """Whether an exclusively-admitted over-budget job is pending or in progress.

        While true, the scheduler must not stage or dispatch another model: the exclusive job needs the
        whole device. Covers ``PENDING_INFERENCE`` and ``INFERENCE_IN_PROGRESS`` (so suppression spans
        from admit through completion, including degraded retries); a terminal fault or success moves the
        job out of those stages, naturally clearing the flag's effect.
        """
        return any(
            tracked.admitted_exclusive
            and tracked.stage in (JobStage.PENDING_INFERENCE, JobStage.INFERENCE_IN_PROGRESS)
            for tracked in self._jobs.values()
        )

    async def handle_job_fault(
        self,
        faulted_job: ImageGenerateJobPopResponse,
        process_info: HordeProcessInfo | None = None,
        process_timeout: float = 0.0,
        *,
        is_resource_failure: bool = False,
        retryable: bool = True,
        scheduling_fault: bool = False,
        fault_reason: str | None = None,
    ) -> InferenceFailureResolution:
        """Resolve a faulted job: requeue it for another (possibly degraded) attempt, or fault it.

        Returns the resolution so the caller can log/audit and, for a degraded retry, drive a more
        conservative re-dispatch. ``retryable=False`` forces a terminal fault (e.g. a post-inference
        safety failure or a shutdown drain, where re-running inference cannot help). ``scheduling_fault``
        marks an ownership/scheduling failure (e.g. an orphan punt) that must not feed the per-card
        "locally unservable" streak, since it is not a verdict on whether the model fits the card.
        """
        return self._resolve_inference_failure_impl(
            faulted_job,
            process_info=process_info,
            process_timeout=process_timeout,
            is_resource_failure=is_resource_failure,
            retryable=retryable,
            scheduling_fault=scheduling_fault,
            fault_reason=fault_reason,
        )

    def handle_job_fault_now(
        self,
        faulted_job: ImageGenerateJobPopResponse,
        process_info: HordeProcessInfo | None = None,
        process_timeout: float = 0.0,
        *,
        is_resource_failure: bool = False,
        retryable: bool = True,
        scheduling_fault: bool = False,
        fault_reason: str | None = None,
    ) -> InferenceFailureResolution:
        """Synchronous fault path for sync callers (e.g. process crash handling). See :meth:`handle_job_fault`."""
        return self._resolve_inference_failure_impl(
            faulted_job,
            process_info=process_info,
            process_timeout=process_timeout,
            is_resource_failure=is_resource_failure,
            retryable=retryable,
            scheduling_fault=scheduling_fault,
            fault_reason=fault_reason,
        )

    def _resolve_inference_failure_impl(
        self,
        faulted_job: ImageGenerateJobPopResponse,
        *,
        process_info: HordeProcessInfo | None,
        process_timeout: float,
        is_resource_failure: bool,
        retryable: bool,
        scheduling_fault: bool = False,
        fault_reason: str | None = None,
    ) -> InferenceFailureResolution:
        tracked = self._tracked_for(faulted_job)

        if tracked is None or tracked.job_info is None:
            logger.error(f"Job {faulted_job.id_} not found in jobs_lookup")
            return InferenceFailureResolution.FAULTED

        if tracked.stage == JobStage.PENDING_SUBMIT:
            logger.warning(f"Job {faulted_job.id_} already in completed_jobs")
            return InferenceFailureResolution.FAULTED

        tracked.inference_attempts += 1

        if process_info is not None:
            logger.error(f"Job {faulted_job.id_} faulted due to process {process_info.process_id} crashing")

        # A slot that crashed/hung while running a job the scheduler knowingly over-committed (admitted
        # against the VRAM budget's verdict) is a resource failure even though the dead slot left no
        # message to classify. Fold that signal in so such a job earns the bounded degraded/isolated
        # retry (which clears the device for it) rather than a plain re-dispatch onto another
        # over-committed slot that would only kill a second process.
        resource_failure = is_resource_failure or tracked.admitted_over_budget

        # A job with no pop timestamp was never formally queued (registered late, e.g. mid-flight), so it
        # has no pending-inference position to return to; such a job is always faulted terminally.
        can_retry = (
            retryable and tracked.time_popped is not None and tracked.inference_attempts < self._max_inference_attempts
        )

        if can_retry and self._set_stage(tracked, JobStage.PENDING_INFERENCE):
            degraded = resource_failure and not tracked.degraded_retry_used
            if degraded:
                tracked.degraded_retry_used = True
                tracked.needs_degraded_dispatch = True
            logger.warning(
                f"Job {faulted_job.id_} inference attempt "
                f"{tracked.inference_attempts}/{self._max_inference_attempts} failed"
                f"{' (resource/OOM)' if resource_failure else ''}; requeuing for "
                f"{'a degraded, isolated' if degraded else 'another'} attempt.",
            )
            return InferenceFailureResolution.RETRY_DEGRADED if degraded else InferenceFailureResolution.RETRY

        # Terminal fault: out of attempts, not retryable, or the requeue transition was refused.
        tracked.job_info.fault_job()
        tracked.job_info.time_to_generate = process_timeout
        self._record_fault_diagnostics(tracked, is_resource_failure=resource_failure, fault_reason=fault_reason)

        # A terminal resource fault feeds the circuit-breaker (per-model "locally unservable" streak) and
        # the self-throttle backstop. A model the device cannot run faults every attempt no matter how it
        # is isolated, so without this the worker keeps popping and dropping it until the horde server
        # forces maintenance; the scheduler/manager read these counters to stop the bleeding first. A
        # scheduling/ownership fault (an orphan punt) is excluded: it is not a card-fit verdict, and keying
        # it to the dispatched card would wrongly de-list a model a capable card can still run.
        if resource_failure and not scheduling_fault:
            # Key the streak to the card the job was dispatched to (recorded at mark_inference_started); None
            # on a single-GPU host keeps it worker-wide. The live process's index is not used for the key so
            # the fault and the success that clears it always agree on the card.
            self._record_resource_fault(faulted_job.model, device_index=tracked.last_dispatched_device_index)

        if self._set_stage(tracked, JobStage.PENDING_SUBMIT):
            # A crash/timeout-faulted job never produces an inference RESULT message, so the
            # dispatcher's per-result completion increment never fires for it. Count it here so the
            # job is not silently dropped from the worker's terminal-job accounting. Without this a
            # caller waiting for every job to reach a terminal state (the e2e harness, and the
            # worker's own queue-drain logic) waits forever on a job that has, in fact, finished
            # (as a fault). The faulted-kudos counter is still incremented once at submit time.
            self._total_num_completed_jobs += 1
        return InferenceFailureResolution.FAULTED

    def _record_fault_diagnostics(
        self,
        tracked: TrackedJob,
        *,
        is_resource_failure: bool,
        fault_reason: str | None = None,
    ) -> None:
        """Attach a fault diagnostic to the job so it rides along on the faulted submit's gen_metadata.

        A faulted job carries no image to hang per-image faults on, so this is the only record of *why*
        it faulted that reaches the horde. The reason and attempt count also aid local post-mortems.
        """
        reason = fault_reason or ("resource/OOM" if is_resource_failure else "inference failure")
        self._job_faults.setdefault(tracked.job_id, []).append(
            GenMetadataEntry(
                type=METADATA_TYPE.information,
                value=METADATA_VALUE.see_ref,
                ref=f"faulted after {tracked.inference_attempts} attempt(s): {reason}"[:255],
            ),
        )

    async def discard_job(self, job_id: GenerationID) -> bool:
        """Forcibly remove a job from the tracker by ID, regardless of its current stage.

        A last-resort drop for the submit loop's backstop: a job the submitter cannot make progress on
        (it keeps raising) is removed here so the queue can drain, rather than the loop spinning on the
        same head-of-queue job forever. Returns True if a job was removed.
        """
        removed = self._jobs.pop(job_id, None)
        self._job_faults.pop(job_id, None)
        if removed is not None:
            self._last_job_submitted_time = time.time()
        return removed is not None

    async def purge_jobs(self) -> None:
        """Clear all jobs from the tracker."""
        self._purge_jobs()

    def _purge_jobs(self) -> None:
        """Synchronous purge for abort/shutdown paths."""
        self._jobs.clear()
        self._last_job_submitted_time = time.time()

    # endregion
