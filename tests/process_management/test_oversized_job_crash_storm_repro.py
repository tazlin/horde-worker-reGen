"""Reproduction and fix for the over-budget "best-effort admit" crash storm.

Failure mode:
    On a device whose free VRAM cannot satisfy the budget's (deliberately conservative) estimate for a
    heavy head-of-queue model, the scheduler force-admits the job best-effort rather than wedge the
    queue. The over-committed slot then crashes natively or thrashes so badly its first step makes no
    progress, and the watchdog kills it. The kill clears the in-progress set, so the next scheduling
    cycle force-admits the *same* job onto another slot and kills it too. Classifying that crash as an
    ordinary (non-resource) failure means each attempt is a plain re-dispatch onto another equally
    over-committed slot, so a single unservable job costs several process recoveries before it is finally
    faulted. Repeated across such jobs this is a sustained recovery/fault storm where process recoveries
    climb at roughly ``max_inference_attempts`` per faulted job.

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

import pytest

from horde_worker_regen.process_management import resource_budget
from horde_worker_regen.process_management.job_tracker import (
    InferenceFailureResolution,
    JobStage,
    JobTracker,
)
from horde_worker_regen.process_management.messages import HordeControlFlag, HordeProcessState

from .conftest import make_job_pop_response, make_mock_process_info
from .test_budget_starvation_wedge_repro import (
    _ACHIEVABLE_FREE_VRAM_MB,
    _HEAD_MODEL,
    _HEAVY_SDXL_PREDICTED_VRAM_MB,
    _VRAM_RESERVE_MB,
    _build_wedged_scheduler,
    _enqueue_head_jobs,
)


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

        The scheduler tagged this job ``admitted_over_budget`` when it force-admitted it. Its slot crash
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


class TestSchedulerTagsOverBudgetAdmit:
    """The scheduler must record the over-budget admission so the crash path can route it correctly."""

    async def test_best_effort_admit_tags_job_over_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the head can only be admitted best-effort, the scheduler tags it ``admitted_over_budget``.

        Drives the real budget gate with a prediction that cannot fit the device's achievable free-VRAM
        floor. The head must still be admitted (a preload is issued) and, crucially, must be tagged so a
        later crash of its over-committed slot takes the isolated degraded retry rather than a plain
        re-dispatch.
        """
        monkeypatch.setattr(
            resource_budget,
            "predict_job_sampling_vram_mb",
            lambda job, baseline: _HEAVY_SDXL_PREDICTED_VRAM_MB,
        )
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 1000.0)

        scheduler, _process_map, job_tracker, proc_sd15, proc_sdxl = _build_wedged_scheduler(
            free_vram_mb=_ACHIEVABLE_FREE_VRAM_MB,
        )
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 64000.0)
        await _enqueue_head_jobs(job_tracker)
        head_job = job_tracker.jobs_pending_inference[0]

        assert _HEAVY_SDXL_PREDICTED_VRAM_MB + _VRAM_RESERVE_MB > _ACHIEVABLE_FREE_VRAM_MB  # genuinely unfittable

        admitted = any(scheduler.preload_models() for _ in range(20))

        assert admitted is True
        assert HordeControlFlag.PRELOAD_MODEL in {proc_sd15.last_control_flag, proc_sdxl.last_control_flag}
        assert job_tracker.is_admitted_over_budget(head_job) is True
