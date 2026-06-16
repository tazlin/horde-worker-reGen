"""Tests for the harness JobLifecycleAuditor (no child processes involved)."""

from __future__ import annotations

from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.harness import JobLifecycleAuditor

from .conftest import make_mock_job, make_testable_process_manager, track_popped_job_async


async def _run_clean_lifecycle(manager, job) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Drive one job through the full tracker lifecycle."""
    tracker = manager._job_tracker
    await tracker.mark_inference_started(job)
    await tracker.release_in_progress(job)
    await tracker.drop_pending_inference_by_id(job.id_)
    job_info = await tracker.get_job_info(job)
    await tracker.queue_for_safety(job_info)
    await tracker.begin_safety_check(job_info)
    taken = await tracker.take_being_safety_checked(job.id_)
    await tracker.queue_for_submit(taken)
    await tracker.finalize_submitted(taken)


class TestJobLifecycleAuditor:
    """Tests for attach/verify."""

    async def test_clean_lifecycle_passes(self) -> None:
        """A job that completes the whole pipeline produces no audit failures."""
        manager = make_testable_process_manager()
        auditor = JobLifecycleAuditor()
        auditor.attach(manager)

        job = await track_popped_job_async(manager._job_tracker, make_mock_job())
        await _run_clean_lifecycle(manager, job)

        assert auditor.verify() == []

    async def test_lost_job_is_detected(self) -> None:
        """A popped job that never reaches finalize is reported as lost."""
        manager = make_testable_process_manager()
        auditor = JobLifecycleAuditor()
        auditor.attach(manager)

        await track_popped_job_async(manager._job_tracker, make_mock_job())

        failures = auditor.verify()
        assert any("never finalized" in f for f in failures)
        assert any("did not drain" in f for f in failures)

    async def test_double_finalize_is_detected(self) -> None:
        """Finalizing the same job twice is reported as a double submit."""
        manager = make_testable_process_manager()
        auditor = JobLifecycleAuditor()
        auditor.attach(manager)

        job = await track_popped_job_async(manager._job_tracker, make_mock_job())
        job_info = await manager._job_tracker.get_job_info(job)
        assert job_info is not None
        await manager._job_tracker.queue_for_submit(job_info)
        await manager._job_tracker.finalize_submitted(job_info)
        await manager._job_tracker.finalize_submitted(job_info)

        failures = auditor.verify()
        assert any("double submit" in f for f in failures)

    async def test_faulted_submissions_are_counted(self) -> None:
        """Jobs finalized in a faulted state are tallied separately."""
        manager = make_testable_process_manager()
        auditor = JobLifecycleAuditor()
        auditor.attach(manager)

        job = await track_popped_job_async(manager._job_tracker, make_mock_job())
        job_info = await manager._job_tracker.get_job_info(job)
        job_info.state = GENERATION_STATE.faulted  # pyrefly: ignore
        await manager._job_tracker.queue_for_submit(job_info)  # pyrefly: ignore
        await manager._job_tracker.finalize_submitted(job_info)  # pyrefly: ignore

        assert auditor.num_jobs_submitted_faulted == 1
        assert auditor.verify() == []
