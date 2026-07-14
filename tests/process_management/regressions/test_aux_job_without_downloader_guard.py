"""An auxiliary-bearing job popped with background downloads disabled must fault fast, not wedge.

Without a background download process there is no prefetch coordinator: nothing can ever place a LoRA/TI
job's files on disk, arm a deadline for it, or clear its dispatch gate. Left pending, such a job is
invisible to dispatch forever: it starves the queue head and drives save-our-ship recovery churn while
never progressing. Production cannot reach this state (the downloader is always enabled and LoRA
advertising is suppressed without one), but injected job sources (harness scenarios, canned tests) bypass
pop advertising, so the pop seam guards the invariant that every gated job has an owner: with no owner
possible, the job faults terminally at intake instead.
"""

from __future__ import annotations

from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry

from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from tests.process_management.conftest import make_job_pop_response, make_testable_process_manager


def _lora_job():  # noqa: ANN202
    return make_job_pop_response(
        "some-model",
        loras=[LorasPayloadEntry(name="styleA", model=1.0, clip=1.0, is_version=False)],
    )


class TestAuxJobWithoutDownloaderFaultsFast:
    """With no downloader, an aux-bearing pop is terminally faulted at intake; plain jobs are untouched."""

    async def test_lora_job_popped_without_downloader_faults_terminally(self) -> None:
        """The popped LoRA job leaves PENDING_INFERENCE immediately instead of waiting for a preparer."""
        manager = make_testable_process_manager()
        assert manager._enable_background_downloads is False
        popper = manager._job_popper

        job = _lora_job()
        await popper._enqueue_popped_job(job)

        tracked = manager._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.stage != JobStage.PENDING_INFERENCE, (
            "an aux-bearing job with no possible preparer must not stay pending (it would gate forever)"
        )
        assert job not in manager._job_tracker.jobs_pending_inference

    async def test_plain_job_popped_without_downloader_stays_pending(self) -> None:
        """A job with no auxiliary references flows through the same seam unaffected."""
        manager = make_testable_process_manager()
        popper = manager._job_popper

        job = make_job_pop_response("some-model")
        await popper._enqueue_popped_job(job)

        tracked = manager._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.stage == JobStage.PENDING_INFERENCE
