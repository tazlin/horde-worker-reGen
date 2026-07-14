"""Reproduction tests for the pop side ignoring a stalled submit endpoint, and remote-fault drain semantics.

While the submit endpoint is unavailable, pops on the same session keep succeeding, so the worker keeps
accepting new work at full rate even as finished generations pile up unsubmitted. These tests encode the
contract that the pop side should react to a submit stall (back off, or withhold pops while the pending-submit
backlog is deep and nothing is draining), and they pin down how remote submit rejections behave: a server
"too slow" rejection during an outage should not be treated as the worker's own fault storm, and a duplicate
"already submitted" acknowledgement should not count as a failure.

Construction is authentic: a real ``JobTracker``, ``JobSubmitter``, and ``JobPopper`` share one ``WorkerState``,
so the counters and flags the popper consults are only ever mutated by production code driven through the real
submit path. Only the horde and R2 network seams are doubled.
"""

from __future__ import annotations

import io
import time
from unittest.mock import AsyncMock, Mock

import PIL.Image
import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.generic_api.apimodels import RequestErrorResponse

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeImageResult
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.jobs.job_submitter import JobSubmitter
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_test_api_sessions,
    make_test_model_metadata,
    make_test_runtime_config,
    queue_job_for_submit_async,
    track_popped_job_async,
)


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
    """Encode a tiny real image so the submit path's image handling succeeds instead of faulting early."""
    buffer = io.BytesIO()
    PIL.Image.new("RGB", (8, 8), color=(1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def _timeout_session() -> AsyncMock:
    """A horde session whose every submit times out (a total submit outage)."""
    session = AsyncMock()
    session.submit_request = AsyncMock(side_effect=TimeoutError)
    return session


def _scripted_session(script: list[object]) -> AsyncMock:
    """A horde session whose submits follow ``script`` in order (exception types raised, objects returned)."""
    session = AsyncMock()
    session.submit_request = AsyncMock(side_effect=script)
    return session


def _error_response_session(message: str) -> AsyncMock:
    """A horde session whose every submit returns a horde error carrying ``message``."""
    session = AsyncMock()
    session.submit_request = AsyncMock(return_value=RequestErrorResponse(message=message))
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
) -> JobSubmitter:
    """Build a real JobSubmitter over the shared state and tracker with mocked API sessions."""
    return JobSubmitter(
        state=state,
        job_tracker=tracker,
        shutdown_manager=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=make_mock_bridge_data()),
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


def _image_job() -> ImageGenerateJobPopResponse:
    """A single-image job whose pop response carries an R2 upload target (so it is submittable)."""
    return make_job_pop_response("stable_diffusion", r2_upload="https://example.com/upload")


async def _queue_for_submit(
    tracker: JobTracker,
    job: ImageGenerateJobPopResponse,
) -> None:
    """Track a popped job and queue a completed, submittable single-image result for it."""
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


def _pop_is_withholding(popper: JobPopper, bridge_data: reGenBridgeData, state: WorkerState) -> bool:
    """Read the popper's own suppression surfaces (no side effects) to see if it would hold pops.

    True when any real gate the popper consults is engaged: post-error backoff, the consecutive-failure
    threshold, post-inference (safety) backpressure, or a full local queue. None of these is coupled to the
    pending-submit backlog today, which is exactly what the submit-stall contract expects to change.
    """
    return bool(
        popper.is_in_error_backoff
        or state.consecutive_failed_jobs >= 3
        or popper._is_post_inference_backlogged()
        or popper._is_queue_full(bridge_data),
    )


async def test_repeated_submit_failures_back_off_the_pop_side() -> None:
    """A run of failed submit attempts against a dead endpoint should make the pop side back off.

    A single submit pass against a timing-out endpoint re-attempts one generation many times with zero
    successes. After that many consecutive submit failures the worker should be slowing its pops (rather than
    continuing to accept new work at full rate onto a queue it cannot drain). The pop side's own decision
    surface is read here: an active error backoff or an engaged consecutive-failure skip.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    horde = _timeout_session()
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=_r2_ok_session())

    for _ in range(3):
        await _queue_for_submit(tracker, _image_job())
    await submitter.api_submit_job()

    assert horde.submit_request.await_count >= 3, "arrange failed: fewer than three submit attempts failed"

    bridge_data = make_mock_bridge_data()
    popper = _make_popper(state, tracker, bridge_data=bridge_data, shutdown_manager=Mock())
    should_skip = popper._handle_consecutive_failures(bridge_data, time.time())

    assert should_skip is True or popper.is_in_error_backoff is True, (
        "the pop side kept popping at full rate despite a run of failed submit attempts"
    )


async def test_deep_pending_submit_backlog_withholds_pops() -> None:
    """A deep pending-submit backlog with nothing draining should throttle pops.

    Ten finished generations wait unsubmitted and no submit has run, so admitting still more work only
    deepens a backlog the worker cannot clear. The popper should withhold pops in this state.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)

    queued = [_image_job() for _ in range(10)]
    for job in queued:
        await _queue_for_submit(tracker, job)

    pending_ids = {pending.sdk_api_job_info.id_ for pending in tracker.jobs_pending_submit}
    assert len(tracker.jobs_pending_submit) >= 10, "arrange failed: backlog not deep enough"
    assert pending_ids == {job.id_ for job in queued}, "arrange failed: a job left the queue before any submit ran"

    bridge_data = make_mock_bridge_data()
    popper = _make_popper(state, tracker, bridge_data=bridge_data, shutdown_manager=Mock())

    assert _pop_is_withholding(popper, bridge_data, state), (
        "the popper kept popping while a deep pending-submit backlog was stuck with nothing draining"
    )


async def test_single_transient_submit_stall_does_not_throttle_pops() -> None:
    """One transient submit timeout immediately followed by success must not throttle the pop side.

    A brief hiccup that the very next attempt clears is not a failure worth backing off for: neither the
    consecutive-failure counter nor the pop error backoff should engage. This guards the coupling contract
    against over-triggering on ordinary noise.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    horde = _scripted_session([TimeoutError, _ok_response()])
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=_r2_ok_session())

    await _queue_for_submit(tracker, _image_job())
    await submitter.api_submit_job()

    assert len(tracker.jobs_pending_submit) == 0, "the job did not submit after the transient stall cleared"
    assert state.consecutive_failed_jobs == 0, "a single transient stall incremented the consecutive-failure counter"

    bridge_data = make_mock_bridge_data()
    popper = _make_popper(state, tracker, bridge_data=bridge_data, shutdown_manager=Mock())
    assert popper.is_in_error_backoff is False, "a single transient stall put the pop side into error backoff"
    assert _pop_is_withholding(popper, bridge_data, state) is False


async def test_server_too_slow_rejections_during_outage_do_not_kill_worker() -> None:
    """Server "check your worker speed" rejections during a remote outage should not shut the worker down.

    When the submit endpoint force-faults piled-up jobs with a "took too long" rejection, that is a symptom of
    the remote stall, not evidence the worker's generations are failing. Under ``exit_on_unhandled_faults`` the
    worker must not treat a run of these remote rejections as its own fault storm and shut itself down.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    horde = _error_response_session(
        "Server took too long to respond to this generation. Please check your worker speed.",
    )
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=_r2_ok_session())

    for _ in range(3):
        await _queue_for_submit(tracker, _image_job())
    for _ in range(3):
        await submitter.api_submit_job()

    shutdown_manager = Mock()
    bridge_data = make_mock_bridge_data(exit_on_unhandled_faults=True)
    popper = _make_popper(state, tracker, bridge_data=bridge_data, shutdown_manager=shutdown_manager)
    popper._handle_consecutive_failures(bridge_data, time.time())

    shutdown_manager.shutdown.assert_not_called()


@pytest.mark.parametrize(
    "message",
    ["This generation is already submitted.", "This generation has already been submitted."],
    ids=["substring_match", "phrasing_variant"],
)
async def test_already_submitted_ack_does_not_count_as_failure(message: str) -> None:
    """A duplicate-submit acknowledgement removes the job without counting it as a consecutive failure.

    The duplicate is the horde confirming it already holds the result (typically because a timed-out submit
    actually landed), so the job should leave the queue without incrementing the consecutive-failure counter
    that gates the pop pause. The contract is phrasing-insensitive: it must hold whether or not the server's
    wording contains the exact substring the code keys on.
    """
    state = WorkerState()
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    horde = _error_response_session(message)
    submitter = _make_submitter(state, tracker, horde_client_session=horde, aiohttp_session=_r2_ok_session())

    job = _image_job()
    await _queue_for_submit(tracker, job)
    await submitter.api_submit_job()

    remaining_ids = [pending.sdk_api_job_info.id_ for pending in tracker.jobs_pending_submit]
    assert job.id_ not in remaining_ids, "a duplicate-submit job was left in the pending-submit queue"
    assert state.consecutive_failed_jobs == 0, "a duplicate-submit acknowledgement counted as a job failure"
