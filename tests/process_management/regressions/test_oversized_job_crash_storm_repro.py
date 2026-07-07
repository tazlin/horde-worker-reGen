"""Crash classification for jobs that carry the over-budget admission tag.

Failure mode:
    A job carrying ``admitted_over_budget`` can still crash natively or thrash so badly its first step makes
    no progress, and the watchdog kills its slot. Classifying that crash as an ordinary non-resource failure
    means each attempt is a plain re-dispatch onto another equally constrained slot, so a single unserviceable
    job costs several process recoveries before it is finally faulted.

What this module pins:
    The worker-logic half of the cascade, which is deterministic and needs no GPU. The native crash /
    VRAM thrash that *causes* each kill is the GPU-only half and is exercised elsewhere; here a fault is
    injected at the same point the lifecycle injects one when it kills a slot.

The fix:
    A job the scheduler admits against the budget's verdict is tagged ``admitted_over_budget``. A
    crash/hang of its slot is then treated as a resource failure even though the dead slot leaves no
    message to classify, so it earns the bounded degraded/isolated retry (which clears the device and
    runs it alone) instead of a plain re-dispatch onto another over-committed slot. An untagged crash
    (a transient glitch unrelated to capacity) keeps the ordinary retry.
"""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import (
    InferenceFailureResolution,
    JobStage,
    JobTracker,
)
from tests.process_management.conftest import make_job_pop_response, make_mock_process_info
from tests.process_management.regressions.test_budget_starvation_wedge_repro import _HEAD_MODEL


async def _pop_and_start(job_tracker: JobTracker, model: str = _HEAD_MODEL) -> object:
    """Pop a fresh job and mark it in progress: the state a dispatched job is in when its slot dies."""
    job = make_job_pop_response(model=model)
    await job_tracker.record_popped_job(job)
    await job_tracker.mark_inference_started(job)
    return job


class TestSlotCrashRetryClassification:
    """How a slot crash/hang is retried depends on whether the job was knowingly over-committed."""

    async def test_untagged_slot_crash_takes_ordinary_retry(self, job_tracker: JobTracker) -> None:
        """A crash with no capacity signal is a possibly-transient failure: it earns the ordinary retry.

        This is the lifecycle's slot-fault path (``handle_job_fault_now(..., retryable=True)`` with
        ``is_resource_failure`` defaulting False) for a job that was *not* over-committed. A fresh slot
        may succeed where a transiently-broken one failed, so a plain (non-isolated) re-dispatch is
        correct here; only the exhausting attempt is terminal.
        """
        job_tracker.set_retry_policy(2)
        job = await _pop_and_start(job_tracker)
        slot = make_mock_process_info(1, model_name=_HEAD_MODEL, state=HordeProcessState.INFERENCE_STARTING)

        first = job_tracker.handle_job_fault_now(faulted_job=job, process_info=slot)  # pyrefly: ignore
        assert first is InferenceFailureResolution.RETRY
        assert job_tracker.is_degraded_dispatch_pending(job) is False  # type: ignore[arg-type]
        assert job_tracker.get_stage(job.id_) is JobStage.PENDING_INFERENCE  # type: ignore[union-attr]

    async def test_overbudget_admit_crash_takes_isolated_degraded_retry(self, job_tracker: JobTracker) -> None:
        """THE FIX: a crash on a job admitted against the budget is a resource failure -> isolated retry.

        The scheduler tagged this job ``admitted_over_budget`` when it admitted it. Its slot crash
        therefore routes to the bounded degraded/isolated retry (which clears the device and runs the job
        alone) rather than a plain re-dispatch onto another over-committed slot that would only kill a
        second process. This is what breaks the amplification: the over-committed job no longer takes a
        healthy concurrent slot down with it on retry.
        """
        job_tracker.set_retry_policy(2)
        job = await _pop_and_start(job_tracker)
        job_tracker.mark_admitted_over_budget(job)  # pyrefly: ignore
        slot = make_mock_process_info(1, model_name=_HEAD_MODEL, state=HordeProcessState.INFERENCE_STARTING)

        # The lifecycle still passes no explicit resource signal (the dead slot left no message); the tag
        # alone must be enough to earn the degraded path.
        resolution = job_tracker.handle_job_fault_now(faulted_job=job, process_info=slot)  # pyrefly: ignore

        assert resolution is InferenceFailureResolution.RETRY_DEGRADED
        assert job_tracker.is_degraded_dispatch_pending(job) is True  # type: ignore[arg-type]
        assert job_tracker.get_stage(job.id_) is JobStage.PENDING_INFERENCE  # type: ignore[union-attr]

    async def test_bounded_retry_cost_is_attempts_per_fault(self, job_tracker: JobTracker) -> None:
        """The retry budget bounds the per-job cost: a job is dispatched at most ``attempts`` times.

        The storm's escalation was process recoveries climbing at roughly ``max_inference_attempts`` per
        faulted job. The bound itself is correct (a retry policy is desirable); the fix above ensures the
        bounded retries are isolated rather than collateral kills of healthy concurrent slots. This pins
        the arithmetic so a regression that unbounds the retry would be caught.
        """
        attempts = 2
        job_tracker.set_retry_policy(attempts)
        job = await _pop_and_start(job_tracker)
        job_tracker.mark_admitted_over_budget(job)  # pyrefly: ignore

        dispatches = 0
        for _ in range(attempts + 2):  # loop more than the budget to prove it terminates
            slot = make_mock_process_info(1, model_name=_HEAD_MODEL, state=HordeProcessState.INFERENCE_STARTING)
            resolution = job_tracker.handle_job_fault_now(faulted_job=job, process_info=slot)  # pyrefly: ignore
            dispatches += 1
            if resolution is InferenceFailureResolution.FAULTED:
                break
            await job_tracker.mark_inference_started(job)  # pyrefly: ignore

        assert dispatches == attempts
        assert job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT  # type: ignore[union-attr]
