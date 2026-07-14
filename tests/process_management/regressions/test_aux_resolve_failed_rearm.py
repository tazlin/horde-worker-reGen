"""A child's aux-resolve fault must withdraw the job's prepared state so its files are re-fetched.

The parent marks a job aux-prepared from the download process's prefetch outcomes and, from then on, the
session cache asserts the files are on disk. When an inference child then cannot resolve one of those files
(a raced eviction or disk error), the disk has contradicted that cache. If the retryable fault requeued the
job while it still read as prepared, the reconcile sweep would never re-request the files and the scheduler
would keep re-dispatching the job into the same resolve failure until its attempts were exhausted, with no
download ever issued. These tests pin the withdrawal contract: an ``AUX_RESOLVE_FAILED_INFO`` fault clears
the job's prepared flag and forgets the contradicted cache entries, the reconcile sweep re-arms a fresh
prefetch for the requeued job, and a subsequent successful outcome makes it dispatchable again.
"""

from __future__ import annotations

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry
from horde_sdk.ai_horde_api.consts import GENERATION_STATE

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    AUX_RESOLVE_FAILED_INFO,
    AuxModelKind,
    AuxPrefetchOutcome,
    HordeAuxPrefetchResultMessage,
    HordeInferenceResultMessage,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.models.aux_prefetch_coordinator import AuxPrefetchCoordinator
from tests.process_management.conftest import make_job_pop_response, mark_job_in_progress_async, track_popped_job_async
from tests.process_management.ipc.test_message_dispatch import _make_dispatcher


class _SenderSpy:
    """Records each prefetch request the coordinator would send to the download process."""

    def __init__(self) -> None:
        self.calls: list[tuple[list, list]] = []

    def __call__(self, entries: list, pins: list) -> None:
        self.calls.append((entries, pins))


def _lora(name: str) -> LorasPayloadEntry:
    return LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=False)


def _make_coordinator(tracker: JobTracker) -> tuple[AuxPrefetchCoordinator, _SenderSpy]:
    sender = _SenderSpy()
    coordinator = AuxPrefetchCoordinator(
        job_tracker=tracker,
        state=WorkerState(),
        prefetch_sender=sender,
        download_timeout_provider=lambda: 120.0,
        pin_sender=lambda pins: None,
        clock=lambda: 1_000.0,
    )
    return coordinator, sender


def _ok_outcome(job: ImageGenerateJobPopResponse, name: str) -> HordeAuxPrefetchResultMessage:
    return HordeAuxPrefetchResultMessage(
        process_id=9000,
        process_launch_identifier=1,
        info="prefetch result",
        outcomes=[AuxPrefetchOutcome(kind=AuxModelKind.LORA, name=name, ok=True, requesting_job_ids=[job.id_])],
    )


def _resolve_fault_message(job: ImageGenerateJobPopResponse) -> HordeInferenceResultMessage:
    return HordeInferenceResultMessage(
        process_id=2,
        process_launch_identifier=0,
        info=AUX_RESOLVE_FAILED_INFO,
        time_elapsed=0.0,
        state=GENERATION_STATE.faulted,
        sdk_api_job_info=job,
    )


async def _prepare_job_via_prefetch(
    tracker: JobTracker,
    coordinator: AuxPrefetchCoordinator,
    lora_name: str,
) -> ImageGenerateJobPopResponse:
    """Pop a one-LoRA job and walk it through a successful prefetch so it reads as prepared."""
    job = make_job_pop_response("some-model", loras=[_lora(lora_name)])
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(_ok_outcome(job, lora_name))
    assert tracker.are_job_aux_models_prepared(job) is True
    return job


class TestResolveFaultWithdrawsPreparation:
    """The retryable resolve fault re-gates the job and re-arms its prefetch instead of spinning."""

    async def test_resolve_fault_regates_job_and_reconcile_rearms_prefetch(self) -> None:
        """The incident loop, closed: fault -> withdrawn preparation -> reconcile re-request -> prepared again."""
        tracker = JobTracker()
        tracker.set_retry_policy(3)
        coordinator, sender = _make_coordinator(tracker)
        job = await _prepare_job_via_prefetch(tracker, coordinator, "styleA")
        await mark_job_in_progress_async(tracker, job)
        dispatcher = _make_dispatcher(job_tracker=tracker)

        job_info = await tracker.get_job_info(job)
        assert job_info is not None
        await dispatcher._handle_faulted_inference_result(_resolve_fault_message(job), job_info)

        # The job is requeued for another attempt but no longer reads as prepared, so the dispatch gate
        # holds it and the session cache no longer vouches for the missing file.
        tracked = tracker.get_tracked_job(job.id_)
        assert tracked is not None and tracked.stage == JobStage.PENDING_INFERENCE
        assert tracker.are_job_aux_models_prepared(job) is False
        assert tracker.is_lora_cached(_lora("styleA")) is False
        # Withdrawal must not be undone by the readiness check alone: only a fresh on-disk report may.
        assert tracker.mark_job_aux_prepared_if_ready(job.id_) is False

        requests_before = len(sender.calls)
        coordinator.reconcile_and_refresh_pins()
        assert len(sender.calls) == requests_before + 1
        entries, _pins = sender.calls[-1]
        assert [(entry.kind, entry.name) for entry in entries] == [(AuxModelKind.LORA, "styleA")]

        coordinator.on_prefetch_result(_ok_outcome(job, "styleA"))
        assert tracker.are_job_aux_models_prepared(job) is True

    async def test_terminal_resolve_fault_still_forgets_contradicted_cache(self) -> None:
        """With no attempts left the job faults out, but the lying cache entry must still be dropped.

        A later job referencing the same file must go through a fresh prefetch (whose presence probe
        re-verifies the disk) rather than being insta-prepared from the contradicted session cache.
        """
        tracker = JobTracker()
        tracker.set_retry_policy(1)
        coordinator, sender = _make_coordinator(tracker)
        job = await _prepare_job_via_prefetch(tracker, coordinator, "styleB")
        await mark_job_in_progress_async(tracker, job)
        dispatcher = _make_dispatcher(job_tracker=tracker)

        job_info = await tracker.get_job_info(job)
        assert job_info is not None
        await dispatcher._handle_faulted_inference_result(_resolve_fault_message(job), job_info)

        tracked = tracker.get_tracked_job(job.id_)
        assert tracked is not None and tracked.stage != JobStage.PENDING_INFERENCE
        assert tracker.is_lora_cached(_lora("styleB")) is False

        follow_up = make_job_pop_response("some-model", loras=[_lora("styleB")])
        await track_popped_job_async(tracker, follow_up)
        requests_before = len(sender.calls)
        coordinator.on_job_popped(follow_up)
        assert tracker.are_job_aux_models_prepared(follow_up) is False
        assert len(sender.calls) == requests_before + 1

    async def test_non_aux_fault_leaves_preparation_intact(self) -> None:
        """An ordinary generation fault must not withdraw preparation: the files are still on disk."""
        tracker = JobTracker()
        tracker.set_retry_policy(3)
        coordinator, _sender = _make_coordinator(tracker)
        job = await _prepare_job_via_prefetch(tracker, coordinator, "styleC")
        await mark_job_in_progress_async(tracker, job)
        dispatcher = _make_dispatcher(job_tracker=tracker)

        message = _resolve_fault_message(job).model_copy(update={"info": "RuntimeError: sampler exploded"})
        job_info = await tracker.get_job_info(job)
        assert job_info is not None
        await dispatcher._handle_faulted_inference_result(message, job_info)

        assert tracker.are_job_aux_models_prepared(job) is True
        assert tracker.is_lora_cached(_lora("styleC")) is True
