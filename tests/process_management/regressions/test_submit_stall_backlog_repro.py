"""Reproduction tests for a submit-endpoint stall piling finished jobs into the pending-submit queue.

When the horde submit endpoint stops answering while pops keep succeeding, finished generations accumulate at
the head of the pending-submit queue and the submit loop re-attacks the same dead endpoint. These tests encode
the contract the submit path should honor while that endpoint is unavailable:

- Repeated failed attempts of one generation should be spaced by a backoff (an observable, growing wait), not
  hammered back-to-back with no delay, so a stalling endpoint is not amplified by the worker.
- The number of times a single stuck generation re-hits the endpoint should stay bounded, or the job should be
  resolved terminally, rather than diverging.
- Work that was uploaded to R2 once should not be re-uploaded on every submit retry of the same generation.
- A healthy endpoint drains the whole backlog in pop order, and a transient stall that recovers drains
  cleanly without latching any failure state.

The construction is authentic end to end: a real ``JobTracker`` and ``JobSubmitter`` share one ``WorkerState``
and run the real ``api_submit_job`` path. Only the network seams are doubled: the horde client session (made to
time out, error, or recover) and the R2 upload session (a counting success). ``asyncio.sleep`` as imported by the
submitter is replaced with a recorder so any backoff wait is both instant and observable.
"""

from __future__ import annotations

import io
import uuid
from unittest.mock import AsyncMock, Mock

import aiohttp
import PIL.Image
import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from pydantic import JsonValue

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeImageResult
from horde_worker_regen.process_management.jobs import job_submitter as job_submitter_module
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo, PendingJob
from horde_worker_regen.process_management.jobs.job_submitter import JobSubmitter
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_test_api_sessions,
    make_test_model_metadata,
    make_test_runtime_config,
    queue_job_for_submit_async,
    track_popped_job_async,
)

_SUBMIT_RETRY_CAP: int = PendingJob._max_consecutive_failed_job_submits
"""The per-generation submit retry ceiling; attempts on one object stop once this is exceeded."""

_ATTEMPTS_BOUND_PER_IMAGE: int = 30
"""The largest number of times a single image's submit may re-hit the endpoint before we call it unbounded."""


class _SleepSpy:
    """An awaitable stand-in for ``asyncio.sleep`` that records requested delays and returns at once.

    Recording the delays makes any inter-attempt backoff both instantaneous (tests stay fast) and observable
    (a positive recorded delay is the evidence that the retry path waited between attempts).
    """

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float, *args: object, **kwargs: object) -> None:
        self.delays.append(delay)

    @property
    def positive_delays(self) -> list[float]:
        """Return only the strictly-positive recorded delays (the backoff waits)."""
        return [d for d in self.delays if isinstance(d, (int, float)) and d > 0]


class _FakeR2Response:
    """A stand-in R2 upload response reporting success."""

    status = 200


class _FakeR2Put:
    """An async-context-manager stand-in for ``aiohttp.ClientSession.put`` that succeeds."""

    async def __aenter__(self) -> _FakeR2Response:
        return _FakeR2Response()

    async def __aexit__(self, *_args: object) -> bool:
        return False


class _R2PutCounter:
    """A callable stand-in for ``aiohttp.ClientSession.put`` that counts how many uploads were issued."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_args: object, **_kwargs: object) -> _FakeR2Put:
        self.calls += 1
        return _FakeR2Put()


def _r2_counting_session() -> tuple[Mock, _R2PutCounter]:
    """An aiohttp session whose ``put`` reports success and a counter of the uploads it received."""
    counter = _R2PutCounter()
    session = Mock()
    session.put = counter
    return session, counter


def _valid_image_bytes() -> bytes:
    """Encode a tiny real image so the submit path's image handling succeeds instead of faulting early."""
    buffer = io.BytesIO()
    PIL.Image.new("RGB", (8, 8), color=(1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def _timeout_session(exc_type: type[BaseException]) -> AsyncMock:
    """A horde session whose every submit raises ``exc_type`` (a total submit outage of that flavour)."""
    session = AsyncMock()
    session.submit_request = AsyncMock(side_effect=exc_type)
    return session


def _scripted_session(script: list[object]) -> AsyncMock:
    """A horde session whose submits follow ``script`` in order.

    Each entry is either an exception type (raised on that call) or a response object (returned). This models
    an endpoint that fails for a while and then recovers.
    """
    session = AsyncMock()
    session.submit_request = AsyncMock(side_effect=script)
    return session


def _ok_response(reward: float = 1.0) -> Mock:
    """A benign successful submit response carrying a numeric reward."""
    return Mock(reward=reward)


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
            bridge_data=bridge_data if bridge_data is not None else make_mock_bridge_data(),
        ),
        api_sessions=make_test_api_sessions(
            horde_client_session=horde_client_session,
            aiohttp_session=aiohttp_session,
        ),
        model_metadata=make_test_model_metadata(),
    )


def _image_job(r2_upload: str = "https://example.com/upload") -> ImageGenerateJobPopResponse:
    """A single-image job whose pop response carries an R2 upload target (so it is submittable)."""
    return make_job_pop_response("stable_diffusion", r2_upload=r2_upload)


def _batch_job(count: int, r2_upload: str = "https://example.com/upload") -> ImageGenerateJobPopResponse:
    """A batch job whose pop response carries ``count`` generation IDs and matching R2 upload targets."""
    ids = [str(uuid.uuid4()) for _ in range(count)]
    data: dict[str, JsonValue] = {
        "id": ids[0],
        "ids": ids,
        "model": "stable_diffusion",
        "payload": {
            "prompt": "test prompt",
            "width": 512,
            "height": 512,
            "ddim_steps": 30,
            "n_iter": count,
            "seed": "42",
            "sampler_name": "k_euler",
        },
        "skipped": {},
        "source_processing": "txt2img",
        "r2_upload": r2_upload,
        "r2_uploads": [f"{r2_upload}/{index}" for index in range(count)],
    }
    return ImageGenerateJobPopResponse(**data)  # pyrefly: ignore - validated by pydantic


async def _queue_for_submit(
    tracker: JobTracker,
    job: ImageGenerateJobPopResponse,
    *,
    images: int,
    state: GENERATION_STATE = GENERATION_STATE.ok,
) -> None:
    """Track a popped job and queue a completed result for it that is ready to submit.

    ``images`` valid image results are attached so the submit path reaches the API-submit stage; a faulted
    ``state`` clears the images so the fault-report path is taken instead.
    """
    await track_popped_job_async(tracker, job, time_popped=0.0)
    image_results = (
        None
        if state == GENERATION_STATE.faulted
        else [HordeImageResult(image_bytes=_valid_image_bytes()) for _ in range(images)]
    )
    job_info = HordeJobInfo(
        sdk_api_job_info=job,
        state=state,
        censored=False,
        safety_evaluated=True,
        time_popped=0.0,
        time_to_generate=1.0,
        job_image_results=image_results,
    )
    await queue_job_for_submit_async(tracker, job_info)


@pytest.fixture()
def _sleep_spy(monkeypatch: pytest.MonkeyPatch) -> _SleepSpy:
    """Replace ``asyncio.sleep`` as the submitter sees it with a recorder that returns immediately."""
    spy = _SleepSpy()
    monkeypatch.setattr(job_submitter_module.asyncio, "sleep", spy)
    return spy


@pytest.mark.parametrize(
    "exc_type",
    [TimeoutError, aiohttp.ClientError, OSError],
    ids=["timeout", "client_error", "os_error"],
)
async def test_failing_submit_retries_are_spaced_by_backoff(
    exc_type: type[BaseException],
    _sleep_spy: _SleepSpy,
) -> None:
    """Consecutive failed attempts of one stuck generation should be separated by a nonzero backoff wait.

    A submit endpoint that is timing out (or erroring, or refusing the connection) is not helped by being
    re-hit with no delay: the worker should wait, and lengthen the wait, between attempts on the same
    generation. The contract asserted here is that at least one positive backoff sleep occurs between the
    repeated attempts driven by a single submit pass.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    horde = _timeout_session(exc_type)
    aiohttp_session, _counter = _r2_counting_session()
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=aiohttp_session)

    job = _image_job()
    await _queue_for_submit(tracker, job, images=1)
    await submitter.api_submit_job()

    attempts = horde.submit_request.await_count
    assert attempts >= 2, "arrange failed: the failing submit did not re-attempt the same generation"
    assert _sleep_spy.positive_delays, "consecutive failed submit attempts were not spaced by any backoff wait"


async def test_stuck_generation_submit_attempts_stay_bounded_across_passes(_sleep_spy: _SleepSpy) -> None:
    """A persistently timing-out generation must not re-hit the endpoint without bound across submit passes.

    Driving several submit passes against a dead endpoint should either bound how many times one generation
    re-attempts the submit or resolve that generation terminally (removed from the pending-submit queue). An
    ever-growing attempt count for a single stuck job is the amplification this contract forbids.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    horde = _timeout_session(TimeoutError)
    aiohttp_session, _counter = _r2_counting_session()
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=aiohttp_session)

    job = _image_job()
    await _queue_for_submit(tracker, job, images=1)
    for _ in range(5):
        await submitter.api_submit_job()

    attempts = horde.submit_request.await_count
    still_pending = any(pending.sdk_api_job_info.id_ == job.id_ for pending in tracker.jobs_pending_submit)
    assert attempts <= _ATTEMPTS_BOUND_PER_IMAGE or not still_pending, (
        f"one stuck generation re-hit the submit endpoint {attempts} times without terminal resolution"
    )


async def test_r2_upload_not_repeated_on_submit_retry(_sleep_spy: _SleepSpy) -> None:
    """A generation whose image already uploaded to R2 must not re-upload on each submit retry.

    The image bytes are unchanged between submit attempts, so a submit that fails after a successful upload
    should retry only the API submit, not the (bandwidth-heavy) R2 upload. The contract asserted is that the
    R2 upload happens once for the generation even though the API submit is attempted many times.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    horde = _timeout_session(TimeoutError)
    aiohttp_session, counter = _r2_counting_session()
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=aiohttp_session)

    job = _image_job()
    await _queue_for_submit(tracker, job, images=1)
    await submitter.api_submit_job()

    submit_attempts = horde.submit_request.await_count
    assert submit_attempts >= 2, "arrange failed: the submit did not retry, so re-upload cannot be observed"
    assert counter.calls == 1, (
        f"R2 upload was repeated {counter.calls} times across {submit_attempts} submit attempts of one generation"
    )


async def test_batch_submit_outage_lacks_backoff_but_stays_bounded(_sleep_spy: _SleepSpy) -> None:
    """A multi-image batch whose submits all time out retries as a bounded wave, and should back off.

    Each image in the batch should re-attempt a bounded number of times (a retry wave, not a per-image
    divergence), and, as for a single job, consecutive waves should be spaced by a backoff. The bound is
    asserted as a confirmation; the backoff is the contract this reproduction targets.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    horde = _timeout_session(TimeoutError)
    aiohttp_session, _counter = _r2_counting_session()
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=aiohttp_session)

    batch_size = 2
    job = _batch_job(batch_size)
    await _queue_for_submit(tracker, job, images=batch_size)
    await submitter.api_submit_job()

    attempts = horde.submit_request.await_count
    assert attempts >= batch_size, "arrange failed: the batch submit did not attempt each image"
    assert attempts <= batch_size * _ATTEMPTS_BOUND_PER_IMAGE, (
        f"batch submit diverged: {attempts} attempts for a {batch_size}-image batch"
    )
    assert _sleep_spy.positive_delays, "batch submit retry waves were not spaced by any backoff wait"


async def test_faulted_report_submit_outage_stays_bounded(_sleep_spy: _SleepSpy) -> None:
    """A faulted job whose fault-report submit also times out must not spin unboundedly.

    Even the fault report goes to the same dead endpoint; the worker should bound how many times it re-reports
    one faulted job or resolve it terminally, rather than looping on the report forever.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    horde = _timeout_session(TimeoutError)
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=Mock())

    job = _image_job()
    await _queue_for_submit(tracker, job, images=1, state=GENERATION_STATE.faulted)
    for _ in range(5):
        await submitter.api_submit_job()

    attempts = horde.submit_request.await_count
    still_pending = any(pending.sdk_api_job_info.id_ == job.id_ for pending in tracker.jobs_pending_submit)
    assert attempts <= _ATTEMPTS_BOUND_PER_IMAGE or not still_pending, (
        f"a faulted job re-reported itself {attempts} times without terminal resolution"
    )


async def test_healthy_endpoint_drains_backlog_in_pop_order(_sleep_spy: _SleepSpy) -> None:
    """A healthy submit endpoint drains a mixed backlog fully, head first, updating the last-submit time.

    This proves the harness faithfully models the ordinary path: many queued jobs (single and batch) submit
    in pop order, the pending-submit queue empties, and no failure state is left behind.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    horde = AsyncMock()
    horde.submit_request = AsyncMock(return_value=_ok_response())
    aiohttp_session, _counter = _r2_counting_session()
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=aiohttp_session)

    last_submit_before = tracker._last_job_submitted_time
    queued_ids: list[object] = []
    for index in range(12):
        job = _batch_job(2) if index % 3 == 0 else _image_job()
        await _queue_for_submit(tracker, job, images=len(job.ids))
        queued_ids.append(job.id_)

    drain_order: list[object] = []
    for _ in range(len(queued_ids)):
        head = tracker.jobs_pending_submit[0].sdk_api_job_info.id_
        drain_order.append(head)
        await submitter.api_submit_job()

    assert drain_order == queued_ids, "backlog did not drain in pop (FIFO) order"
    assert len(tracker.jobs_pending_submit) == 0, "healthy endpoint left jobs stranded in the submit queue"
    assert state.consecutive_failed_jobs == 0
    assert tracker._last_job_submitted_time > last_submit_before, (
        "a successful drain did not advance the last-submit time"
    )


async def test_transient_stall_recovers_and_drains_without_latched_failure(_sleep_spy: _SleepSpy) -> None:
    """An endpoint that fails a couple of attempts then recovers drains the whole backlog and latches nothing.

    A brief stall that clears should leave the queue empty and the consecutive-failure counter at zero: the
    recovery is complete, so no residual failure state should remain to throttle later work.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    script: list[object] = [TimeoutError, TimeoutError, *[_ok_response() for _ in range(10)]]
    horde = _scripted_session(script)
    aiohttp_session, _counter = _r2_counting_session()
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=aiohttp_session)

    jobs = [_image_job() for _ in range(3)]
    for job in jobs:
        await _queue_for_submit(tracker, job, images=1)
    for _ in jobs:
        await submitter.api_submit_job()

    assert len(tracker.jobs_pending_submit) == 0, "backlog did not drain after the endpoint recovered"
    assert state.consecutive_failed_jobs == 0, "a recovered transient stall left the failure counter latched"


async def test_unexpected_head_failure_backstop_drops_stuck_job(_sleep_spy: _SleepSpy) -> None:
    """The submit loop's backstop drops a head job that repeatedly raises an unforeseen error, so work proceeds.

    An exception that escapes ``api_submit_job`` on the same head job every iteration would otherwise recur
    forever; the bounded backstop discards it after a few identical failures so the queue behind it can drain.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    submitter = _make_submitter(state, tracker, horde_client_session=Mock(), aiohttp_session=Mock())

    head = _image_job()
    behind = _image_job()
    await _queue_for_submit(tracker, head, images=1)
    await _queue_for_submit(tracker, behind, images=1)

    error = RuntimeError("unexpected submit failure")
    for _ in range(submitter._MAX_CONSECUTIVE_HEAD_SUBMIT_FAILURES - 1):
        await submitter._handle_unexpected_submit_failure(error)
        assert len(tracker.jobs_pending_submit) == 2, "head job dropped before the backstop threshold"

    await submitter._handle_unexpected_submit_failure(error)

    remaining_ids = [pending.sdk_api_job_info.id_ for pending in tracker.jobs_pending_submit]
    assert head.id_ not in remaining_ids, "backstop did not drop the stuck head job"
    assert behind.id_ in remaining_ids, "backstop dropped a job that was not the stuck head"


async def test_head_only_stall_does_not_monopolize_the_submit_loop(_sleep_spy: _SleepSpy) -> None:
    """When only the head job's submit fails, the jobs queued behind it should submit within a bounded span.

    A single un-submittable head must not indefinitely block the generations behind it that would submit
    fine. Driving a bounded number of submit passes should see the healthy siblings drain.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)

    head = _image_job()
    siblings = [_image_job() for _ in range(2)]
    stall_count = 0

    def _route(request: object, *_args: object, **_kwargs: object) -> Mock:
        nonlocal stall_count
        if getattr(request, "id_", None) == head.id_:
            stall_count += 1
            raise TimeoutError
        return _ok_response()

    horde = AsyncMock()
    horde.submit_request = AsyncMock(side_effect=_route)
    aiohttp_session, _counter = _r2_counting_session()
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=aiohttp_session)

    await _queue_for_submit(tracker, head, images=1)
    for sibling in siblings:
        await _queue_for_submit(tracker, sibling, images=1)

    for _ in range(len(siblings) + 2):
        await submitter.api_submit_job()

    assert stall_count >= 1, "arrange failed: the head job's submits never stalled"
    remaining_ids = {pending.sdk_api_job_info.id_ for pending in tracker.jobs_pending_submit}
    for sibling in siblings:
        assert sibling.id_ not in remaining_ids, "a healthy sibling was starved behind the stalled head job"
