"""The auxiliary-prefetch fault origin is excluded from the consecutive-failure pop pause.

A fault whose origin is the auxiliary-prefetch pipeline (the worker never ran a generation for it) must not
count toward the consecutive-failure pop pause, or one bad remote file silences a worker that is itself
perfectly healthy. A not-found reference no longer faults a waiting job at all (it is served without the file),
so the aux-origin fault under test here is the deadline backstop: a prefetch that never resolves and is never
skipped, faulted once its per-job deadline expires. These assert that such faults are excluded from the pause
and never trip ``exit_on_unhandled_faults``, that a genuine generation-origin fault still counts (the exclusion
is scoped by origin, not blanket), and that a shared not-found outcome adds no faults because it salvages every
co-waiter. The arrangement is authentic end to end: real jobs are faulted through the prefetch coordinator's
deadline path and reported through a real submitter, all sharing one ``WorkerState``, so the counter the popper
reads is only ever mutated by production code.
"""

from __future__ import annotations

import io
import time
from unittest.mock import AsyncMock, Mock

import PIL.Image
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, TIPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    AuxModelKind,
    AuxPrefetchOutcome,
    HordeAuxPrefetchResultMessage,
    HordeImageResult,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.jobs.job_submitter import JobSubmitter
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.aux_prefetch_coordinator import AuxPrefetchCoordinator
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_test_api_sessions,
    make_test_model_metadata,
    make_test_runtime_config,
    queue_job_for_submit_async,
    track_popped_job_async,
)


class _Clock:
    """A hand-advanceable clock so the prefetch deadline that produces an aux-origin fault is deterministic."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _make_coordinator(
    tracker: JobTracker,
    state: WorkerState,
    *,
    clock: _Clock | None = None,
    download_timeout: float = 120.0,
) -> AuxPrefetchCoordinator:
    """Build a real prefetch coordinator whose control-message senders are inert.

    The senders and downloader-status probe are irrelevant here: the test only drives the coordinator's
    completion/failure/deadline wiring, which mutates the shared tracker and worker state. An injected clock and
    short download timeout let a test expire a job's prefetch deadline to produce a genuine aux-origin fault.
    """
    return AuxPrefetchCoordinator(
        job_tracker=tracker,
        state=state,
        prefetch_sender=lambda _entries, _pins: None,
        download_timeout_provider=lambda: download_timeout,
        pin_sender=lambda _pins: None,
        in_flight_provider=dict,
        clock=(clock if clock is not None else time.time),
    )


def _make_submitter(
    state: WorkerState,
    tracker: JobTracker,
    *,
    horde_client_session: object,
    aiohttp_session: object,
    bridge_data: Mock | None = None,
) -> JobSubmitter:
    """Build a real JobSubmitter over the shared state and tracker with mocked API sessions."""
    return JobSubmitter(
        state=state,
        job_tracker=tracker,
        shutdown_manager=Mock(),
        runtime_config=make_test_runtime_config(
            bridge_data=bridge_data if bridge_data is not None else make_mock_bridge_data()
        ),
        api_sessions=make_test_api_sessions(
            horde_client_session=horde_client_session,
            aiohttp_session=aiohttp_session,
        ),
        model_metadata=make_test_model_metadata(),
    )


def _make_popper(
    state: WorkerState,
    tracker: JobTracker,
    *,
    bridge_data: Mock,
    shutdown_manager: Mock,
) -> JobPopper:
    """Build a real JobPopper over the shared state and tracker."""
    return JobPopper(
        state=state,
        process_map=ProcessMap({}),
        job_tracker=tracker,
        shutdown_manager=shutdown_manager,
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        api_sessions=make_test_api_sessions(horde_client_session=Mock(), aiohttp_session=Mock()),
        max_inference_processes=2,
        max_concurrent_inference_processes=1,
        dry_run_skip_api=False,
    )


def _fault_report_session() -> AsyncMock:
    """A horde session whose submit accepts a fault report (a benign response drives a finished submit)."""
    session = AsyncMock()
    session.submit_request = AsyncMock(return_value=Mock(reward=0.0))
    return session


class _FakeR2Response:
    """A stand-in R2 upload response reporting success."""

    status = 200


class _FakeR2Put:
    """An async-context-manager stand-in for ``aiohttp.ClientSession.put`` that succeeds."""

    async def __aenter__(self) -> _FakeR2Response:
        return _FakeR2Response()

    async def __aexit__(self, *_args: object) -> bool:
        return False


def _r2_ok_session() -> Mock:
    """An aiohttp session whose ``put`` reports a successful (status 200) upload."""
    session = Mock()
    session.put = Mock(return_value=_FakeR2Put())
    return session


def _valid_image_bytes() -> bytes:
    """Encode a tiny real image so the submit path's WebP re-encode succeeds instead of faulting."""
    buffer = io.BytesIO()
    PIL.Image.new("RGB", (8, 8), color=(1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def _ti_job(name: str) -> ImageGenerateJobPopResponse:
    """A real job referencing one textual inversion and a valid R2 upload target."""
    return make_job_pop_response(
        "stable_diffusion",
        tis=[TIPayloadEntry(name=name)],
        r2_upload="https://example.com/upload",
    )


def _ti_terminal_failure_message(name: str, job_ids: list[object]) -> HordeAuxPrefetchResultMessage:
    """A prefetch result reporting a textual inversion as terminally unfetchable for every named job.

    Not a rejection (no ``rejection_reason``): a plain download failure classified terminal. Under the salvage
    contract the coordinator dispatches every named job without the file rather than faulting it.
    """
    return HordeAuxPrefetchResultMessage(
        process_id=9000,
        process_launch_identifier=1,
        info="r",
        outcomes=[
            AuxPrefetchOutcome(
                kind=AuxModelKind.TI,
                name=name,
                ok=False,
                retryable=False,
                detail="permanently unfetchable",
                requesting_job_ids=job_ids,  # pyrefly: ignore - GenerationID list, validated by the model
            ),
        ],
    )


async def _fault_one_ti_job_through_prefetch(
    tracker: JobTracker,
    coordinator: AuxPrefetchCoordinator,
    clock: _Clock,
    ti_name: str,
) -> ImageGenerateJobPopResponse:
    """Pop a TI job and fault it through the prefetch deadline backstop, confirming it reaches PENDING_SUBMIT.

    A not-found reference no longer faults a waiting job (it is served without the file), so the aux-origin fault
    these tests exclude from the pop pause now comes from the deadline backstop: a prefetch that never resolves
    and is never skipped. Popping the job arms its deadline; advancing the injected clock past it and scanning
    faults the job with the aux-prefetch origin, since no download is in flight and the reference is not skipped.
    """
    job = _ti_job(ti_name)
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    clock.now += 121.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT, (
        "arrange failed: prefetch deadline did not fault the job"
    )
    return job


async def _submit_head_pending_job(submitter: JobSubmitter) -> None:
    """Report the head pending-submit job through the real submit path (which drives the failure counter)."""
    await submitter.api_submit_job()


async def _submit_one_successful_generation(
    tracker: JobTracker,
    submitter: JobSubmitter,
) -> None:
    """Run one wholly successful, non-faulted generation through the real submit path."""
    job = make_job_pop_response("stable_diffusion", r2_upload="https://example.com/upload")
    await track_popped_job_async(tracker, job, time_popped=0.0)
    job_info = HordeJobInfo(
        sdk_api_job_info=job,
        state=GENERATION_STATE.ok,
        censored=False,
        safety_evaluated=True,
        time_popped=0.0,
        time_to_generate=1.0,
        job_image_results=[HordeImageResult(image_bytes=_valid_image_bytes())],
    )
    await queue_job_for_submit_async(tracker, job_info)
    await submitter.api_submit_job()


async def _fault_one_generation_job_through_submit(
    tracker: JobTracker,
    submitter: JobSubmitter,
) -> None:
    """Terminally fault one job with generation origin and report it through the real submit path.

    The job is held by the tracker with the default (generation) fault origin and queued for submit in the
    faulted state, so the submitter's failure-counter logic sees an ordinary generation fault, not a
    scheduling-recovery or aux-prefetch one.
    """
    job = make_job_pop_response("stable_diffusion", r2_upload="https://example.com/upload")
    await track_popped_job_async(tracker, job, time_popped=0.0)
    job_info = HordeJobInfo(
        sdk_api_job_info=job,
        state=GENERATION_STATE.faulted,
        censored=None,
        safety_evaluated=False,
        time_popped=0.0,
        time_to_generate=0.0,
        job_image_results=None,
    )
    await queue_job_for_submit_async(tracker, job_info)
    await submitter.api_submit_job()


async def test_aux_prefetch_fault_streak_does_not_pause_pops() -> None:
    """A streak of aux-prefetch faults (a healthy worker, one bad file) must not latch the pop pause.

    Three distinct jobs each reference the same unfetchable textual inversion and are faulted before any
    generation runs. The consecutive-failure pop pause exists to stop a worker whose generations are failing;
    a fault the worker never generated for is outside that contract and must not silence all job intake.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    clock = _Clock()
    coordinator = _make_coordinator(tracker, state, clock=clock)
    submitter = _make_submitter(
        state,
        tracker,
        horde_client_session=_fault_report_session(),
        aiohttp_session=Mock(),
    )

    for index in range(3):
        await _fault_one_ti_job_through_prefetch(tracker, coordinator, clock, f"popular-ti-{index}")
    for _ in range(3):
        await _submit_head_pending_job(submitter)

    popper = _make_popper(state, tracker, bridge_data=make_mock_bridge_data(), shutdown_manager=Mock())
    should_skip = popper._handle_consecutive_failures(make_mock_bridge_data(), time.time())

    assert should_skip is False
    assert state.too_many_consecutive_failed_jobs is False


async def test_aux_prefetch_fault_streak_does_not_kill_worker_under_exit_on_unhandled_faults() -> None:
    """The same aux-prefetch fault streak must not trip ``exit_on_unhandled_faults`` and shut the worker down.

    Killing a healthy worker because one popular remote file cannot be fetched is the worst acceptable
    outcome of this bug, so the shutdown seam must never be invoked for aux-prefetch-origin faults.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    clock = _Clock()
    coordinator = _make_coordinator(tracker, state, clock=clock)
    submitter = _make_submitter(
        state,
        tracker,
        horde_client_session=_fault_report_session(),
        aiohttp_session=Mock(),
    )

    for index in range(3):
        await _fault_one_ti_job_through_prefetch(tracker, coordinator, clock, f"popular-ti-{index}")
    for _ in range(3):
        await _submit_head_pending_job(submitter)

    shutdown_manager = Mock()
    bridge_data = make_mock_bridge_data(exit_on_unhandled_faults=True)
    popper = _make_popper(state, tracker, bridge_data=bridge_data, shutdown_manager=shutdown_manager)

    popper._handle_consecutive_failures(bridge_data, time.time())

    shutdown_manager.shutdown.assert_not_called()


async def test_generation_faults_still_count_and_successes_reset_amid_aux_faults() -> None:
    """Aux faults are excluded from the failure counter while ordinary generation faults still count.

    The exclusion must be scoped by fault origin, not blanket: a fault from the auxiliary-prefetch pipeline
    never reaches the consecutive-failure counter, but a genuine generation-origin fault still increments it
    (guarding against the widened exclusion swallowing real generation failures). A successful generation
    resets the counter, and an aux fault interleaved between the generation fault and the success neither
    increments nor resets it. Across the whole mixed-origin workload the counter never reaches the pause
    threshold, so no pause latches.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    clock = _Clock()
    coordinator = _make_coordinator(tracker, state, clock=clock)
    submitter = _make_submitter(
        state,
        tracker,
        horde_client_session=_fault_report_session(),
        aiohttp_session=_r2_ok_session(),
    )

    # An aux-prefetch-origin fault is reported through the real submit path but never reaches the counter.
    await _fault_one_ti_job_through_prefetch(tracker, coordinator, clock, "popular-ti")
    await _submit_head_pending_job(submitter)
    assert state.consecutive_failed_jobs == 0

    # A genuine generation-origin fault does increment the counter (the over-exclusion guard).
    await _fault_one_generation_job_through_submit(tracker, submitter)
    assert state.consecutive_failed_jobs == 1

    # An aux fault interleaved before the reset neither increments the counter nor clears it.
    await _fault_one_ti_job_through_prefetch(tracker, coordinator, clock, "another-popular-ti")
    await _submit_head_pending_job(submitter)
    assert state.consecutive_failed_jobs == 1

    # A successful generation resets the counter through the production submit path.
    await _submit_one_successful_generation(tracker, submitter)
    assert state.consecutive_failed_jobs == 0

    popper = _make_popper(state, tracker, bridge_data=make_mock_bridge_data(), shutdown_manager=Mock())
    should_skip = popper._handle_consecutive_failures(make_mock_bridge_data(), time.time())

    assert should_skip is False
    assert state.too_many_consecutive_failed_jobs is False


async def test_shared_terminal_outcome_salvages_all_cowaiters_and_adds_no_faults() -> None:
    """A shared terminal outcome dispatches every co-waiter without the file, so it contributes no faults at all.

    The downloader reports a single deduplicated outcome per file, so one unfetchable textual inversion names
    every job waiting on it in one delivery. Under the salvage contract none of those jobs is faulted: each is
    dispatched without the file, exactly as it would be at inference. A shared not-found outcome therefore cannot
    move the consecutive-failure counter or latch the pop pause, because it produces no fault to count.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    coordinator = _make_coordinator(tracker, state)

    shared_ti = "shared-popular-ti"
    jobs = [_ti_job(shared_ti) for _ in range(3)]
    for job in jobs:
        await track_popped_job_async(tracker, job)
        coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(_ti_terminal_failure_message(shared_ti, [job.id_ for job in jobs]))

    faulted = [job for job in jobs if tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT]
    assert faulted == [], "a shared not-found outcome must salvage every co-waiter, not fault any"
    for job in jobs:
        assert tracker.are_job_aux_models_prepared(job) is True
        assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE

    popper = _make_popper(state, tracker, bridge_data=make_mock_bridge_data(), shutdown_manager=Mock())
    should_skip = popper._handle_consecutive_failures(make_mock_bridge_data(), time.time())

    assert should_skip is False
    assert state.too_many_consecutive_failed_jobs is False
