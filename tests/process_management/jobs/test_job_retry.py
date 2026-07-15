"""Tests for the bounded/degraded inference retry policy in JobTracker.

These exercise the retry brain (:meth:`JobTracker.handle_job_fault`) directly: how many attempts a job
gets, when a resource failure earns a degraded/isolated retry, and that a terminal fault is counted and
diagnosed exactly once. The end-to-end recovery behaviour is covered by the chaos suites.
"""

from __future__ import annotations

from horde_worker_regen.process_management.jobs.job_tracker import (
    InferenceFailureResolution,
    JobStage,
    JobTracker,
)
from tests.process_management.conftest import make_job_pop_response


async def _pop_and_start(job_tracker: JobTracker) -> object:
    """Pop a fresh job and mark it in progress, the normal pre-fault state for a dispatched job."""
    job = make_job_pop_response(model="stable_diffusion")
    await job_tracker.record_popped_job(job)
    await job_tracker.mark_inference_started(job)
    return job


async def test_retry_disabled_by_default_faults_immediately(job_tracker: JobTracker) -> None:
    """With the default policy (one attempt) a fault is terminal, matching the pre-resiliency behaviour."""
    job = await _pop_and_start(job_tracker)

    resolution = await job_tracker.handle_job_fault(faulted_job=job)  # pyrefly: ignore

    assert resolution is InferenceFailureResolution.FAULTED
    assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT  # type: ignore[union-attr]
    assert job_tracker.total_num_completed_jobs == 1


async def test_bounded_retry_requeues_then_faults_on_exhaustion(job_tracker: JobTracker) -> None:
    """Two attempts: the first fault requeues for inference, the second (exhausted) faults terminally."""
    job_tracker.set_retry_policy(2)
    job = await _pop_and_start(job_tracker)

    first = await job_tracker.handle_job_fault(faulted_job=job)  # pyrefly: ignore
    assert first is InferenceFailureResolution.RETRY
    assert job_tracker.get_stage(job.id_) is JobStage.PENDING_INFERENCE  # type: ignore[union-attr]
    # A requeued job has not reached a terminal state, so it must not be counted as completed.
    assert job_tracker.total_num_completed_jobs == 0

    await job_tracker.mark_inference_started(job)  # pyrefly: ignore
    second = await job_tracker.handle_job_fault(faulted_job=job)  # type: ignore
    assert second is InferenceFailureResolution.FAULTED
    assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT  # type: ignore[union-attr]
    assert job_tracker.total_num_completed_jobs == 1


async def test_non_retryable_fault_is_terminal_even_with_attempts_left(job_tracker: JobTracker) -> None:
    """A non-retryable fault (safety/shutdown) faults terminally regardless of the attempt budget."""
    job_tracker.set_retry_policy(3)
    job = await _pop_and_start(job_tracker)

    resolution = await job_tracker.handle_job_fault(faulted_job=job, retryable=False)  # pyrefly: ignore

    assert resolution is InferenceFailureResolution.FAULTED
    assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT  # type: ignore[union-attr]


async def test_resource_failure_earns_one_degraded_retry(job_tracker: JobTracker) -> None:
    """A resource (OOM) failure is requeued degraded; a second one retries normally (the budget allows it)."""
    job_tracker.set_retry_policy(3)
    job = await _pop_and_start(job_tracker)

    first = await job_tracker.handle_job_fault(faulted_job=job, is_resource_failure=True)  # pyrefly: ignore
    assert first is InferenceFailureResolution.RETRY_DEGRADED
    assert job_tracker.is_degraded_dispatch_pending(job) is True  # type: ignore[arg-type]

    # The scheduler consumes the degraded flag when it dispatches the isolated retry.
    job_tracker.clear_degraded_dispatch(job)  # type: ignore[arg-type]
    assert job_tracker.is_degraded_dispatch_pending(job) is False  # type: ignore[arg-type]

    await job_tracker.mark_inference_started(job)  # pyrefly: ignore
    second = await job_tracker.handle_job_fault(faulted_job=job, is_resource_failure=True)  # pyrefly: ignore
    # The one degraded retry has been spent, so a further resource failure takes the ordinary retry.
    assert second is InferenceFailureResolution.RETRY
    assert job_tracker.is_degraded_dispatch_pending(job) is False  # type: ignore[arg-type]


async def test_disaggregated_structural_fault_latches_job_monolithic_on_requeue(job_tracker: JobTracker) -> None:
    """A structural disaggregated stage fault requeues the job with the disaggregation-declined latch set.

    The latch is what keeps attempt 2 off the pipeline that structurally failed the job: the scheduler's
    eligibility predicate reads it and dispatches the retry monolithically instead of re-routing disagg.
    """
    job_tracker.set_retry_policy(2)
    job = await _pop_and_start(job_tracker)

    first = await job_tracker.handle_job_fault(  # pyrefly: ignore
        faulted_job=job,
        was_disaggregated_structural_fault=True,
    )

    assert first is InferenceFailureResolution.RETRY
    assert job_tracker.get_stage(job.id_) is JobStage.PENDING_INFERENCE  # type: ignore[union-attr]
    assert job_tracker.is_disaggregation_declined(job) is True  # type: ignore[arg-type]


async def test_ordinary_fault_does_not_latch_disaggregation_declined(job_tracker: JobTracker) -> None:
    """A fault not flagged as a structural disaggregated fault leaves the disaggregation latch untouched.

    Control for the structural-decline behaviour: a resource (OOM) failure keeps its existing degraded-retry
    path and does not force the job monolithic, so the pipeline is free to admit it again.
    """
    job_tracker.set_retry_policy(2)
    job = await _pop_and_start(job_tracker)

    resolution = await job_tracker.handle_job_fault(  # pyrefly: ignore
        faulted_job=job,
        is_resource_failure=True,
    )

    assert resolution is InferenceFailureResolution.RETRY_DEGRADED
    assert job_tracker.is_disaggregation_declined(job) is False  # type: ignore[arg-type]


async def test_disaggregated_structural_fault_still_faults_terminally_on_exhaustion(
    job_tracker: JobTracker,
) -> None:
    """When the attempt budget is spent, a structural disaggregated fault still faults terminally.

    The decline latch only redirects a *retry*; once no retry remains the job faults for the horde to
    reissue, exactly as any other exhausted job.
    """
    job_tracker.set_retry_policy(1)
    job = await _pop_and_start(job_tracker)

    resolution = await job_tracker.handle_job_fault(  # pyrefly: ignore
        faulted_job=job,
        was_disaggregated_structural_fault=True,
    )

    assert resolution is InferenceFailureResolution.FAULTED
    assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT  # type: ignore[union-attr]
    assert job_tracker.total_num_completed_jobs == 1


async def test_terminal_fault_records_one_diagnostic(job_tracker: JobTracker) -> None:
    """A terminal fault attaches exactly one diagnostic entry naming the attempts and reason."""
    job = await _pop_and_start(job_tracker)

    await job_tracker.handle_job_fault(faulted_job=job, is_resource_failure=True)  # pyrefly: ignore

    faults = await job_tracker.get_faults_for_job(job.id_)  # type: ignore[union-attr]
    assert len(faults) == 1
    assert faults[0].ref is not None
    assert "attempt" in faults[0].ref


async def test_job_without_pop_timestamp_is_not_retried(job_tracker: JobTracker) -> None:
    """A job registered without a pop (no queue position to return to) is faulted, never requeued."""
    job_tracker.set_retry_policy(3)
    job = make_job_pop_response(model="stable_diffusion")
    # Marking in progress without a prior pop registers the job with no pop timestamp.
    await job_tracker.mark_inference_started(job)

    resolution = await job_tracker.handle_job_fault(faulted_job=job)

    assert resolution is InferenceFailureResolution.FAULTED
    assert job_tracker.get_stage(job.id_) is not JobStage.PENDING_INFERENCE  # type: ignore[union-attr]


async def test_finalize_drops_fault_entries(job_tracker: JobTracker) -> None:
    """Finalizing a faulted job clears its fault entries so the fault map does not grow unbounded."""
    job = await _pop_and_start(job_tracker)
    await job_tracker.handle_job_fault(faulted_job=job)  # pyrefly: ignore

    tracked = job_tracker.get_tracked_job(job.id_)  # type: ignore[union-attr]
    assert tracked is not None and tracked.job_info is not None
    assert await job_tracker.get_faults_for_job(job.id_) != []  # type: ignore[union-attr]

    await job_tracker.finalize_submitted(tracked.job_info)

    assert await job_tracker.get_faults_for_job(job.id_) == []  # type: ignore[union-attr]
