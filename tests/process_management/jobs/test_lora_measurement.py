"""Tests for the harness's LoRA-aware soak measurement surface (no child processes involved).

Covers the JobLifecycleAuditor's completed/faulted split by whether a job carried LoRA references, which
feeds the like-named ``HarnessResult`` fields, and is fed from recorded job-lifecycle events rather than a
GPU.
"""

from __future__ import annotations

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry

from horde_worker_regen.harness import JobLifecycleAuditor
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.simulation._canned_scenarios import make_canned_job
from tests.process_management.conftest import make_mock_job, make_testable_process_manager, track_popped_job_async


def _terminal_job_info(*, with_loras: bool, faulted: bool) -> HordeJobInfo:
    """Build a finalized job info carrying (or not) a LoRA reference in the requested terminal state."""
    loras = [LorasPayloadEntry(name="storm_ref")] if with_loras else None
    response = make_canned_job("Deliberate", loras=loras)
    return HordeJobInfo(
        sdk_api_job_info=response,
        state=GENERATION_STATE.faulted if faulted else GENERATION_STATE.ok,
        time_popped=0.0,
    )


class TestAuditorLoraSplitClassification:
    """`_record_terminal_lora_split` partitions finalized jobs by LoRA presence and terminal state."""

    def test_each_quadrant_is_counted_once(self) -> None:
        """A completed/faulted x with-LoRA/plain job lands in exactly its own bucket."""
        auditor = JobLifecycleAuditor()

        auditor._record_terminal_lora_split(_terminal_job_info(with_loras=True, faulted=False))
        auditor._record_terminal_lora_split(_terminal_job_info(with_loras=False, faulted=False))
        auditor._record_terminal_lora_split(_terminal_job_info(with_loras=True, faulted=True))
        auditor._record_terminal_lora_split(_terminal_job_info(with_loras=False, faulted=True))

        assert auditor.num_jobs_completed_with_loras == 1
        assert auditor.num_jobs_completed_without_loras == 1
        assert auditor.num_jobs_faulted_with_loras == 1
        assert auditor.num_jobs_faulted_without_loras == 1

    def test_split_totals_track_the_stream(self) -> None:
        """Repeated LoRA completions accumulate only in the completed-with-LoRA bucket."""
        auditor = JobLifecycleAuditor()
        for _ in range(4):
            auditor._record_terminal_lora_split(_terminal_job_info(with_loras=True, faulted=False))
        auditor._record_terminal_lora_split(_terminal_job_info(with_loras=False, faulted=False))

        assert auditor.num_jobs_completed_with_loras == 4
        assert auditor.num_jobs_completed_without_loras == 1
        assert auditor.num_jobs_faulted_with_loras == 0
        assert auditor.num_jobs_faulted_without_loras == 0


class TestAuditorLoraSplitThroughTracker:
    """Driving the attached tracker's finalize path records the split from real lifecycle events."""

    async def _finalize(self, manager, *, loras, faulted: bool) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
        tracker = manager._job_tracker
        job = await track_popped_job_async(tracker, make_mock_job(loras=loras))
        job_info = await tracker.get_job_info(job)
        assert job_info is not None
        if faulted:
            job_info.state = GENERATION_STATE.faulted
        await tracker.queue_for_submit(job_info)
        await tracker.finalize_submitted(job_info)

    async def test_finalize_events_feed_the_split(self) -> None:
        """A LoRA job that completes and a plain job that faults are attributed to their own buckets."""
        manager = make_testable_process_manager()
        auditor = JobLifecycleAuditor()
        auditor.attach(manager)

        await self._finalize(manager, loras=[LorasPayloadEntry(name="storm_ref")], faulted=False)
        await self._finalize(manager, loras=None, faulted=True)

        assert auditor.num_jobs_completed_with_loras == 1
        assert auditor.num_jobs_faulted_without_loras == 1
        assert auditor.num_jobs_completed_without_loras == 0
        assert auditor.num_jobs_faulted_with_loras == 0
