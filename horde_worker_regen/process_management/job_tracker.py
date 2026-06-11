"""Actor-backed job state management.

All lifecycle mutations are serialized through a command queue so callers no
longer coordinate lock ownership across modules.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from horde_sdk.ai_horde_api.apimodels import (
    GenMetadataEntry,
    ImageGenerateJobPopResponse,
)
from horde_sdk.ai_horde_api.fields import GenerationID
from loguru import logger

from horde_worker_regen.process_management.job_models import HordeJobInfo
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.utils.job_queue_analyzer import JobQueueAnalyzer

_T = TypeVar("_T")


@dataclass
class _JobTrackerState:
    jobs_lookup: dict[ImageGenerateJobPopResponse, HordeJobInfo] = field(default_factory=dict)
    jobs_in_progress: list[ImageGenerateJobPopResponse] = field(default_factory=list)
    job_faults: dict[GenerationID, list[GenMetadataEntry]] = field(default_factory=dict)
    jobs_pending_safety_check: list[HordeJobInfo] = field(default_factory=list)
    jobs_being_safety_checked: list[HordeJobInfo] = field(default_factory=list)
    jobs_pending_submit: list[HordeJobInfo] = field(default_factory=list)
    jobs_pending_inference: deque[ImageGenerateJobPopResponse] = field(default_factory=deque)
    job_pop_timestamps: dict[ImageGenerateJobPopResponse, float] = field(default_factory=dict)

    num_jobs_faulted: int = 0
    total_num_completed_jobs: int = 0
    max_pending_megapixelsteps: int = 25
    triggered_max_pending_megapixelsteps: bool = False
    triggered_max_pending_megapixelsteps_time: float = 0.0
    last_job_submitted_time: float = field(default_factory=time.time)


@dataclass(frozen=True)
class JobTrackerSnapshot:
    """Immutable snapshot of the job tracker state."""

    jobs_lookup: dict[ImageGenerateJobPopResponse, HordeJobInfo]
    jobs_in_progress: tuple[ImageGenerateJobPopResponse, ...]
    job_faults: dict[GenerationID, list[GenMetadataEntry]]
    jobs_pending_safety_check: tuple[HordeJobInfo, ...]
    jobs_being_safety_checked: tuple[HordeJobInfo, ...]
    jobs_pending_submit: tuple[HordeJobInfo, ...]
    jobs_pending_inference: tuple[ImageGenerateJobPopResponse, ...]
    job_pop_timestamps: dict[ImageGenerateJobPopResponse, float]

    num_jobs_faulted: int
    total_num_completed_jobs: int
    max_pending_megapixelsteps: int
    triggered_max_pending_megapixelsteps: bool
    triggered_max_pending_megapixelsteps_time: float
    last_job_submitted_time: float


@dataclass
class _JobTrackerCommand:
    handler: Callable[[_JobTrackerState], Any]
    response_future: asyncio.Future[Any]


_StopSentinel = _JobTrackerCommand(handler=lambda _: None, response_future=asyncio.Future())


class JobTracker:
    """Actor-backed owner of all job lifecycle state."""

    def __init__(self) -> None:
        """Initialize the job tracker."""
        self._state = _JobTrackerState()
        self._command_queue: asyncio.Queue[_JobTrackerCommand] = asyncio.Queue()
        self._actor_task: asyncio.Task[None] | None = None

    async def start_actor(self) -> None:
        """Start the internal state actor if it is not already running."""
        if self._actor_task is not None:
            return
        self._actor_task = asyncio.create_task(self._run_actor(), name="JobTrackerActor")

    async def stop_actor(self) -> None:
        """Stop the internal state actor if it is running."""
        if self._actor_task is None:
            return
        await self._command_queue.put(_StopSentinel)
        await self._actor_task
        self._actor_task = None

    async def _run_actor(self) -> None:
        while True:
            command_or_stop = await self._command_queue.get()
            if command_or_stop is _StopSentinel:
                break
            command = command_or_stop
            try:
                result = command.handler(self._state)
            except Exception as exc:
                if not command.response_future.done():
                    command.response_future.set_exception(exc)
            else:
                if not command.response_future.done():
                    command.response_future.set_result(result)

    async def _dispatch(self, handler: Callable[[_JobTrackerState], _T]) -> _T:
        """Dispatch a handler to the actor, returning its result."""
        if self._actor_task is None:
            return handler(self._state)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[_T] = loop.create_future()
        await self._command_queue.put(_JobTrackerCommand(handler=handler, response_future=future))
        return await future

    def snapshot(self) -> JobTrackerSnapshot:
        """Return an immutable snapshot view of current job state."""
        return JobTrackerSnapshot(
            jobs_lookup=dict(self._state.jobs_lookup),
            jobs_in_progress=tuple(self._state.jobs_in_progress),
            job_faults={k: list(v) for k, v in self._state.job_faults.items()},
            jobs_pending_safety_check=tuple(self._state.jobs_pending_safety_check),
            jobs_being_safety_checked=tuple(self._state.jobs_being_safety_checked),
            jobs_pending_submit=tuple(self._state.jobs_pending_submit),
            jobs_pending_inference=tuple(self._state.jobs_pending_inference),
            job_pop_timestamps=dict(self._state.job_pop_timestamps),
            num_jobs_faulted=self._state.num_jobs_faulted,
            total_num_completed_jobs=self._state.total_num_completed_jobs,
            max_pending_megapixelsteps=self._state.max_pending_megapixelsteps,
            triggered_max_pending_megapixelsteps=self._state.triggered_max_pending_megapixelsteps,
            triggered_max_pending_megapixelsteps_time=self._state.triggered_max_pending_megapixelsteps_time,
            last_job_submitted_time=self._state.last_job_submitted_time,
        )

    @property
    def jobs_lookup(self) -> dict[ImageGenerateJobPopResponse, HordeJobInfo]:
        """Return a copy of the lookup map of jobs (`ImageGenerateJobPopResponse`) to job info (`HordeJobInfo`)."""
        return dict(self._state.jobs_lookup)

    @property
    def jobs_in_progress(self) -> tuple[ImageGenerateJobPopResponse, ...]:
        """Return a copy of the jobs currently in progress."""
        return tuple(self._state.jobs_in_progress)

    @property
    def job_faults(self) -> dict[GenerationID, list[GenMetadataEntry]]:
        """Return a copy of the job faults dictionary."""
        return {k: list(v) for k, v in self._state.job_faults.items()}

    @property
    def jobs_pending_safety_check(self) -> tuple[HordeJobInfo, ...]:
        """Return a copy of the `HordeJobInfo` objects for jobs pending safety check."""
        return tuple(self._state.jobs_pending_safety_check)

    @property
    def jobs_being_safety_checked(self) -> tuple[HordeJobInfo, ...]:
        """Return a copy of the `HordeJobInfo` objects for jobs currently being safety checked."""
        return tuple(self._state.jobs_being_safety_checked)

    @property
    def jobs_pending_submit(self) -> tuple[HordeJobInfo, ...]:
        """Return a copy of the `HordeJobInfo` objects for jobs pending submit."""
        return tuple(self._state.jobs_pending_submit)

    @property
    def jobs_pending_inference(self) -> tuple[ImageGenerateJobPopResponse, ...]:
        """Return a copy of the job pop responses (`ImageGenerateJobPopResponse`) for jobs pending inference."""
        return tuple(self._state.jobs_pending_inference)

    @property
    def job_pop_timestamps(self) -> dict[ImageGenerateJobPopResponse, float]:
        """Return a copy of the job pop timestamps dictionary."""
        return dict(self._state.job_pop_timestamps)

    @property
    def num_jobs_total(self) -> int:
        """Return the total number of jobs across all states."""
        return (
            len(self._state.jobs_pending_inference)
            + len(self._state.jobs_in_progress)
            + len(self._state.jobs_pending_safety_check)
            + len(self._state.jobs_being_safety_checked)
            + len(self._state.jobs_pending_submit)
        )

    @property
    def current_queue_size(self) -> int:
        """Return the current size of the inference queue."""
        return len(self._state.jobs_pending_inference)

    @property
    def total_num_completed_jobs(self) -> int:
        """Return the total number of completed jobs recorded this session."""
        return self._state.total_num_completed_jobs

    @property
    def num_jobs_faulted(self) -> int:
        """Return the total number of faulted jobs recorded this session."""
        return self._state.num_jobs_faulted

    def set_performance_mode_thresholds(self, max_pending_megapixelsteps: int) -> None:
        """Set the performance mode thresholds."""
        self._state.max_pending_megapixelsteps = max_pending_megapixelsteps

    def reset_megapixelstep_trigger(self) -> None:
        """Reset the megapixelstep trigger."""
        self._state.triggered_max_pending_megapixelsteps = False

    def should_wait_for_pending_megapixelsteps(self) -> bool:
        """Return whether the system should wait based on the currently pending megapixelsteps."""
        pending_megapixelsteps = self.get_pending_megapixelsteps()
        return JobQueueAnalyzer.should_wait_for_pending_megapixelsteps(
            pending_megapixelsteps,
            self._state.max_pending_megapixelsteps,
        )

    def get_pending_megapixelsteps(self) -> int:
        """Return the number of pending megapixelsteps."""
        return JobQueueAnalyzer.calculate_pending_job_magnitude(
            self._state.jobs_pending_inference,
            len(self._state.jobs_pending_submit),
        )

    @property
    def _triggered_max_pending_megapixelsteps(self) -> bool:
        return self._state.triggered_max_pending_megapixelsteps

    @_triggered_max_pending_megapixelsteps.setter
    def _triggered_max_pending_megapixelsteps(self, value: bool) -> None:
        self._state.triggered_max_pending_megapixelsteps = value

    @property
    def _triggered_max_pending_megapixelsteps_time(self) -> float:
        return self._state.triggered_max_pending_megapixelsteps_time

    @_triggered_max_pending_megapixelsteps_time.setter
    def _triggered_max_pending_megapixelsteps_time(self, value: float) -> None:
        self._state.triggered_max_pending_megapixelsteps_time = value

    @property
    def _max_pending_megapixelsteps(self) -> int:
        return self._state.max_pending_megapixelsteps

    async def record_popped_job(
        self,
        job_pop_response: ImageGenerateJobPopResponse,
        time_popped: float | None = None,
    ) -> HordeJobInfo:
        """Record a popped job and return its corresponding HordeJobInfo."""

        def _apply(state: _JobTrackerState) -> HordeJobInfo:
            stamp = time.time() if time_popped is None else time_popped
            state.jobs_pending_inference.append(job_pop_response)
            state.job_pop_timestamps[job_pop_response] = stamp
            job_info = HordeJobInfo(
                sdk_api_job_info=job_pop_response,
                state=None,
                time_popped=stamp,
            )
            state.jobs_lookup[job_pop_response] = job_info
            if job_pop_response.id_ is not None:
                state.job_faults.setdefault(job_pop_response.id_, [])
            return job_info

        return await self._dispatch(_apply)

    async def mark_inference_started(self, job: ImageGenerateJobPopResponse) -> None:
        """Mark a job as started for inference."""
        await self._dispatch(lambda state: state.jobs_in_progress.append(job))

    async def release_in_progress(self, job: ImageGenerateJobPopResponse) -> bool:
        """Release a job from the in-progress state."""

        def _apply(state: _JobTrackerState) -> bool:
            if job in state.jobs_in_progress:
                state.jobs_in_progress.remove(job)
                return True
            return False

        return await self._dispatch(_apply)

    async def drop_pending_inference(self, job: ImageGenerateJobPopResponse) -> bool:
        """Drop a job from the pending inference state."""

        def _apply(state: _JobTrackerState) -> bool:
            if job in state.jobs_pending_inference:
                state.jobs_pending_inference.remove(job)
                return True
            return False

        return await self._dispatch(_apply)

    async def drop_pending_inference_by_id(self, job_id: GenerationID) -> bool:
        """Drop a job from the pending inference state by its ID."""

        def _apply(state: _JobTrackerState) -> bool:
            for job in state.jobs_pending_inference:
                if job.id_ == job_id:
                    state.jobs_pending_inference.remove(job)
                    return True
            return False

        return await self._dispatch(_apply)

    async def queue_for_safety(self, job_info: HordeJobInfo) -> None:
        """Queue a job for safety checking."""
        await self._dispatch(lambda state: state.jobs_pending_safety_check.append(job_info))

    async def queue_for_submit(self, job_info: HordeJobInfo) -> None:
        """Queue a job for submission."""
        await self._dispatch(lambda state: state.jobs_pending_submit.append(job_info))

    async def begin_safety_check(self, job_info: HordeJobInfo) -> None:
        """Begin the safety check process for a job."""

        def _apply(state: _JobTrackerState) -> None:
            state.jobs_pending_safety_check.remove(job_info)
            state.jobs_being_safety_checked.append(job_info)

        await self._dispatch(_apply)

    async def abandon_pending_safety(self, job_info: HordeJobInfo) -> None:
        """Abandon a job from the pending safety check state."""
        await self._dispatch(lambda state: state.jobs_pending_safety_check.remove(job_info))

    async def requeue_being_safety_checked(self) -> None:
        """Requeue all jobs that are currently being safety checked."""

        def _apply(state: _JobTrackerState) -> None:
            if not state.jobs_being_safety_checked:
                return
            state.jobs_pending_safety_check.extend(state.jobs_being_safety_checked)
            state.jobs_being_safety_checked.clear()

        await self._dispatch(_apply)

    async def take_being_safety_checked(self, job_id: GenerationID) -> HordeJobInfo | None:
        """Take a job that is currently being safety checked by its ID."""

        def _apply(state: _JobTrackerState) -> HordeJobInfo | None:
            for i, job_info in enumerate(state.jobs_being_safety_checked):
                if job_info.sdk_api_job_info.id_ == job_id:
                    return state.jobs_being_safety_checked.pop(i)
            return None

        return await self._dispatch(_apply)

    async def record_source_image_fault(self, job_id: GenerationID, entry: GenMetadataEntry) -> None:
        """Record a fault for a job."""

        def _apply(state: _JobTrackerState) -> None:
            if state.job_faults.get(job_id) is None:
                state.job_faults[job_id] = []
            state.job_faults[job_id].append(entry)

        await self._dispatch(_apply)

    async def clear_faults_for_job(self, job_id: GenerationID) -> None:
        """Clear all recorded faults for a job."""
        await self._dispatch(lambda state: state.job_faults.pop(job_id, None))

    async def get_faults_for_job(self, job_id: GenerationID) -> list[GenMetadataEntry]:
        """Get all recorded faults for a job."""
        return await self._dispatch(lambda state: list(state.job_faults.get(job_id, [])))

    async def increment_jobs_faulted(self) -> None:
        """Increment the count of jobs that have encountered faults."""
        await self._dispatch(lambda state: setattr(state, "num_jobs_faulted", state.num_jobs_faulted + 1))

    async def increment_jobs_completed(self) -> None:
        """Increment the count of jobs that have been completed."""
        await self._dispatch(
            lambda state: setattr(state, "total_num_completed_jobs", state.total_num_completed_jobs + 1),
        )

    async def get_job_info(self, job: ImageGenerateJobPopResponse) -> HordeJobInfo | None:
        """Get the job info for a given job."""
        return await self._dispatch(lambda state: state.jobs_lookup.get(job))

    async def set_job_time_to_download_aux_models(
        self,
        job: ImageGenerateJobPopResponse,
        time_elapsed: float | None,
    ) -> bool:
        """Set the time it took to download auxiliary models for a job."""

        def _apply(state: _JobTrackerState) -> bool:
            job_info = state.jobs_lookup.get(job)
            if job_info is None:
                return False
            job_info.time_to_download_aux_models = time_elapsed
            return True

        return await self._dispatch(_apply)

    async def get_time_popped(self, job: ImageGenerateJobPopResponse) -> float | None:
        """Get the time a job was popped from the queue."""
        return await self._dispatch(lambda state: state.job_pop_timestamps.get(job))

    async def ensure_submitted_job_info(self, completed_job_info: HordeJobInfo) -> HordeJobInfo:
        """Ensure lookup contains an entry for a completed job and mark submit time."""

        def _apply(state: _JobTrackerState) -> HordeJobInfo:
            sdk_info = completed_job_info.sdk_api_job_info
            job_info = state.jobs_lookup.get(sdk_info)
            if job_info is None:
                job_info = HordeJobInfo(
                    sdk_api_job_info=sdk_info,
                    time_popped=-1,
                    job_image_results=completed_job_info.job_image_results,
                    state=completed_job_info.state,
                    censored=completed_job_info.censored,
                    time_to_generate=completed_job_info.time_to_generate,
                    time_to_download_aux_models=completed_job_info.time_to_download_aux_models,
                )
                state.jobs_lookup[sdk_info] = job_info
            job_info.time_submitted = time.time()
            return job_info

        return await self._dispatch(_apply)

    async def finalize_submitted(self, completed_job_info: HordeJobInfo) -> None:
        """Finalize a submitted job, removing it from pending and lookup states."""

        def _apply(state: _JobTrackerState) -> None:
            sdk_info = completed_job_info.sdk_api_job_info

            if completed_job_info in state.jobs_pending_submit:
                state.jobs_pending_submit.remove(completed_job_info)
            else:
                logger.warning(f"Job {sdk_info.id_} not found in completed_jobs")

            if sdk_info in state.jobs_lookup:
                del state.jobs_lookup[sdk_info]
            else:
                logger.warning(f"Job {sdk_info.id_} not found in jobs_lookup")

            if sdk_info in state.job_pop_timestamps:
                state.job_pop_timestamps.pop(sdk_info)

            if sdk_info in state.jobs_in_progress:
                state.jobs_in_progress.remove(sdk_info)

            state.last_job_submitted_time = time.time()

        await self._dispatch(_apply)

    async def handle_job_fault(
        self,
        faulted_job: ImageGenerateJobPopResponse,
        process_info: HordeProcessInfo | None = None,
        process_timeout: float = 0.0,
    ) -> None:
        """Handle a job that has faulted, updating its state and related queues."""

        def _apply(state: _JobTrackerState) -> None:
            job_info = state.jobs_lookup.get(faulted_job)

            if job_info is None:
                logger.error(f"Job {faulted_job.id_} not found in jobs_lookup")
                return

            if faulted_job in state.jobs_pending_inference:
                state.jobs_pending_inference.remove(faulted_job)

            job_info.fault_job()
            job_info.time_to_generate = process_timeout

            if process_info is not None:
                logger.error(f"Job {faulted_job.id_} faulted due to process {process_info.process_id} crashing")

            if faulted_job in state.jobs_in_progress:
                state.jobs_in_progress.remove(faulted_job)

            for horde_job_info in state.jobs_pending_safety_check:
                if horde_job_info.sdk_api_job_info.id_ == faulted_job.id_:
                    state.jobs_pending_safety_check.remove(horde_job_info)
                    break

            if job_info not in state.jobs_pending_submit:
                state.jobs_pending_submit.append(job_info)
            else:
                logger.warning(f"Job {faulted_job.id_} already in completed_jobs")

        await self._dispatch(_apply)

    def handle_job_fault_now(
        self,
        faulted_job: ImageGenerateJobPopResponse,
        process_info: HordeProcessInfo | None = None,
        process_timeout: float = 0.0,
    ) -> None:
        """Synchronous emergency fault path for legacy sync callers."""
        job_info = self._state.jobs_lookup.get(faulted_job)

        if job_info is None:
            logger.error(f"Job {faulted_job.id_} not found in jobs_lookup")
            return

        if faulted_job in self._state.jobs_pending_inference:
            self._state.jobs_pending_inference.remove(faulted_job)

        job_info.fault_job()
        job_info.time_to_generate = process_timeout

        if process_info is not None:
            logger.error(f"Job {faulted_job.id_} faulted due to process {process_info.process_id} crashing")

        if faulted_job in self._state.jobs_in_progress:
            self._state.jobs_in_progress.remove(faulted_job)

        for horde_job_info in self._state.jobs_pending_safety_check:
            if horde_job_info.sdk_api_job_info.id_ == faulted_job.id_:
                self._state.jobs_pending_safety_check.remove(horde_job_info)
                break

        if job_info not in self._state.jobs_pending_submit:
            self._state.jobs_pending_submit.append(job_info)
        else:
            logger.warning(f"Job {faulted_job.id_} already in completed_jobs")

    async def purge_jobs(self) -> None:
        """Clear all jobs from the tracker."""

        def _apply(state: _JobTrackerState) -> None:
            state.jobs_pending_inference.clear()
            state.jobs_being_safety_checked.clear()
            state.jobs_pending_safety_check.clear()
            state.jobs_lookup.clear()
            state.jobs_in_progress.clear()
            state.jobs_pending_submit.clear()
            state.last_job_submitted_time = time.time()

        await self._dispatch(_apply)

    def _purge_jobs(self) -> None:
        """Emergency synchronous purge for abort/shutdown paths."""
        self._state.jobs_pending_inference.clear()
        self._state.jobs_being_safety_checked.clear()
        self._state.jobs_pending_safety_check.clear()
        self._state.jobs_lookup.clear()
        self._state.jobs_in_progress.clear()
        self._state.jobs_pending_submit.clear()
        self._state.last_job_submitted_time = time.time()
