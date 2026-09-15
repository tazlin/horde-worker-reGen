"""The text flow's two promises: what it advertises, and that every popped job ends in a submit.

The flow talks to a program the worker does not own, so both halves are worth pinning. What a pop
advertises has to match what the backend will actually accept, or the worker draws jobs it then refuses.
And every failure the backend can hand back has to end in a submit, faulted where it must be: a popped
job the worker drops holds its requester until the server times it out, which the horde charges for.

The tests drive the coordinator's own steps rather than its loop, so each one covers a single decision
with no timing to race.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import Mapping
from typing import override
from unittest.mock import Mock

import pytest
from horde_model_reference.meta_consts import TEXT_BACKENDS
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import (
    JobSubmitResponse,
    ModelPayloadKobold,
    TextGenerateJobPopRequest,
    TextGenerateJobPopResponse,
    TextGenerationJobSubmitRequest,
)
from horde_sdk.ai_horde_api.consts import RC
from loguru import logger

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.jobs import text_generation_coordinator
from horde_worker_regen.process_management.jobs.text_generation_coordinator import (
    FAULTED_GENERATION_TEXT,
    TextGenerationCoordinator,
    TextJobInFlight,
    advertised_model_name,
)
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind
from horde_worker_regen.text_backends import (
    FakeTextBackend,
    TextBackendBusy,
    TextBackendDescription,
    TextBackendRejectedPayload,
    TextBackendUnavailable,
    TextGenerationResult,
)
from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_test_api_sessions,
    make_test_runtime_config,
)

_DESCRIPTION = TextBackendDescription(
    model_name="koboldcpp/Llama-3.2-3B-Instruct-Q4_K_M",
    max_context_length=4096,
    max_length=1024,
)
"""What a live koboldcpp answers: its own name already prefixed, its caps, and no soft prompts.

The coordinator's default backend kind, so this is the description most of these tests run against;
the two that cover another backend's spelling say so.
"""

_SUBMIT_REWARD = 12.5


class _NeverAnsweringTextBackend(FakeTextBackend):
    """A fake whose generations never return, for the flow's own bound on a call that does not come back.

    The driver normally enforces the deadline and reports outliving it as an unavailable backend; this
    stands in for the case where nothing under the flow honours it.
    """

    @override
    async def generate(
        self,
        payload: Mapping[str, object],
        *,
        generation_key: str,
        deadline_seconds: float,
    ) -> TextGenerationResult:
        """Never return."""
        await asyncio.Event().wait()
        raise AssertionError("a generation that never returns cannot produce a result")


class _FakeHordeClientSession:
    """Answers text pops from a script and accepts every submit, recording both.

    Stands in for the SDK client session the same way the alchemy tests' mock does, but typed, because
    the assertions here are about the request the flow built rather than about the call happening.
    """

    def __init__(self, *, pop_responses: list[TextGenerateJobPopResponse] | None = None) -> None:
        self.pop_requests: list[TextGenerateJobPopRequest] = []
        self.submit_requests: list[TextGenerationJobSubmitRequest] = []
        self._pop_responses = deque(pop_responses or [])

    async def submit_request(self, request: object, response_type: object) -> object:
        """Return the next scripted pop answer, or a successful submit answer."""
        if isinstance(request, TextGenerateJobPopRequest):
            self.pop_requests.append(request)
            if self._pop_responses:
                return self._pop_responses.popleft()
            return _empty_pop_response()
        if isinstance(request, TextGenerationJobSubmitRequest):
            self.submit_requests.append(request)
            return JobSubmitResponse(reward=_SUBMIT_REWARD)
        raise AssertionError(f"the text flow made an unexpected request: {request}")


def _empty_pop_response() -> TextGenerateJobPopResponse:
    """Create the answer the horde gives a worker it has no text work for."""
    return TextGenerateJobPopResponse(payload=ModelPayloadKobold(), ids=[])


def _pop_response(*, prompt: str = "a prompt", job_id: str | None = None) -> TextGenerateJobPopResponse:
    """Create a pop answer carrying one generation for this worker."""
    return TextGenerateJobPopResponse(
        payload=ModelPayloadKobold(prompt=prompt),
        id=job_id or str(uuid.uuid4()),
        ids=[],
        model=_DESCRIPTION.model_name,
    )


def _job_id() -> str:
    """Create a job id the horde's own shape, which the submit request validates as a UUID."""
    return str(uuid.uuid4())


def _make_coordinator(
    *,
    backend: FakeTextBackend | None = None,
    session: _FakeHordeClientSession | None = None,
    **bridge_overrides: object,
) -> tuple[TextGenerationCoordinator, FakeTextBackend, _FakeHordeClientSession]:
    """Create a coordinator over a fake backend and a fake horde session, with the scribe role on."""
    resolved_backend = backend if backend is not None else FakeTextBackend(description=_DESCRIPTION)
    resolved_session = session if session is not None else _FakeHordeClientSession()
    # An explicit empty list because the pop request validates the field as a list and the shared mock
    # bridge data leaves it as a Mock attribute.
    bridge_overrides.setdefault("priority_usernames", [])
    bridge_data = make_mock_bridge_data(scribe=True, **bridge_overrides)
    shutdown_manager = Mock()
    shutdown_manager.is_time_for_shutdown.return_value = False

    coordinator = TextGenerationCoordinator(
        state=WorkerState(),
        shutdown_manager=shutdown_manager,
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        api_sessions=make_test_api_sessions(horde_client_session=resolved_session),
        backend=resolved_backend,
        text_backend_kind=bridge_data.text_backend_kind,
    )
    return coordinator, resolved_backend, resolved_session


def _quicken_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the readiness backoff so a cold-start test does not spend real seconds waiting."""
    monkeypatch.setattr(text_generation_coordinator, "READY_BACKOFF_INITIAL_SECONDS", 0.0)
    monkeypatch.setattr(text_generation_coordinator, "READY_BACKOFF_MAX_SECONDS", 0.0)


def test_the_coordinator_names_its_own_workload() -> None:
    """The flow registry keys on this, so it is worth pinning."""
    coordinator, _backend, _session = _make_coordinator()

    assert coordinator.kind is WorkloadKind.TEXT_GENERATION


async def test_nothing_is_popped_before_the_backend_is_ready_and_described() -> None:
    """A pop advertises what the backend can do, so there is nothing truthful to say before it answers."""
    coordinator, _backend, session = _make_coordinator()

    await coordinator.api_text_pop()

    assert coordinator.advertisement is None
    assert session.pop_requests == [], "the flow popped before it knew what the backend could serve"


async def test_the_gate_polls_a_cold_backend_until_it_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend that is still loading answers not-ready for a while, and the gate waits it out."""
    _quicken_readiness(monkeypatch)
    backend = FakeTextBackend(description=_DESCRIPTION, ready_results=(False, False, True))
    coordinator, _backend, session = _make_coordinator(backend=backend)

    assert await coordinator.await_backend_ready() is True

    assert len(backend.ready_calls) == 3
    assert backend.describe_call_count == 1
    assert coordinator.advertisement is not None

    await coordinator.api_text_pop()
    assert len(session.pop_requests) == 1, "the flow pops once the backend has answered"


async def test_the_advertised_model_name_comes_from_the_backend_when_unconfigured() -> None:
    """The backend is the only thing that knows which weights it loaded, so it names the model."""
    coordinator, _backend, _session = _make_coordinator(text_model_name=None)

    await coordinator.await_backend_ready()

    assert coordinator.advertisement is not None
    assert coordinator.advertisement.model_name == "koboldcpp/Llama-3.2-3B-Instruct-Q4_K_M"


async def test_the_advertised_model_name_follows_the_operator_when_configured() -> None:
    """An operator's canonical name is advertised in the backend's spelling, author segment dropped."""
    coordinator, _backend, _session = _make_coordinator(text_model_name="meta-llama/Llama-3.2-3B-Instruct")

    await coordinator.await_backend_ready()

    assert coordinator.advertisement is not None
    assert coordinator.advertisement.model_name == "koboldcpp/Llama-3.2-3B-Instruct"


def test_a_quantisation_suffix_rides_along_into_the_advertised_name() -> None:
    """A quant suffix is part of the model name, so prefixing must not disturb it."""
    assert (
        advertised_model_name(
            configured_name="meta-llama/Llama-3.2-3B-Instruct-Q4_K_M",
            described_name="ignored",
            backend=TEXT_BACKENDS.koboldcpp,
        )
        == "koboldcpp/Llama-3.2-3B-Instruct-Q4_K_M"
    )


def test_the_backend_kind_alone_decides_the_advertised_spelling() -> None:
    """Which backend is attached is the whole of what varies, so another one is one different value."""
    canonical_name = "meta-llama/Llama-3.2-3B-Instruct"

    per_backend = {
        backend: advertised_model_name(
            configured_name=canonical_name,
            described_name=canonical_name,
            backend=backend,
        )
        for backend in TEXT_BACKENDS
    }

    assert per_backend[TEXT_BACKENDS.koboldcpp] == "koboldcpp/Llama-3.2-3B-Instruct"
    assert per_backend[TEXT_BACKENDS.aphrodite] == "aphrodite/meta-llama/Llama-3.2-3B-Instruct"


async def test_a_coordinator_on_another_backend_advertises_that_backends_spelling() -> None:
    """The coordinator carries its backend kind, so the advertisement follows it with no other change."""
    coordinator = TextGenerationCoordinator(
        state=WorkerState(),
        shutdown_manager=Mock(),
        runtime_config=make_test_runtime_config(
            bridge_data=make_mock_bridge_data(scribe=True, priority_usernames=[], text_model_name=None),
        ),
        api_sessions=make_test_api_sessions(horde_client_session=_FakeHordeClientSession()),
        backend=FakeTextBackend(
            description=TextBackendDescription(
                model_name="aphrodite/meta-llama/Llama-3.2-3B-Instruct",
                max_context_length=4096,
                max_length=1024,
            ),
        ),
        text_backend_kind=TEXT_BACKENDS.aphrodite,
    )

    await coordinator.await_backend_ready()

    assert coordinator.advertisement is not None
    assert coordinator.advertisement.model_name == "aphrodite/meta-llama/Llama-3.2-3B-Instruct"


def test_the_backend_factory_is_handed_the_flows_own_backend_kind() -> None:
    """The object built and the kind advertised for are one decision, not two defaults that can drift."""
    requested_kinds: list[TEXT_BACKENDS] = []

    def _build(text_backend_kind: TEXT_BACKENDS) -> FakeTextBackend:
        requested_kinds.append(text_backend_kind)
        return FakeTextBackend(description=_DESCRIPTION)

    coordinator = TextGenerationCoordinator(
        state=WorkerState(),
        shutdown_manager=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=make_mock_bridge_data(scribe=True)),
        api_sessions=make_test_api_sessions(horde_client_session=_FakeHordeClientSession()),
        backend_factory=_build,
        text_backend_kind=TEXT_BACKENDS.aphrodite,
    )

    first_backend = coordinator.require_backend()

    assert requested_kinds == [TEXT_BACKENDS.aphrodite]
    assert coordinator.require_backend() is first_backend, "the factory was called more than once"


def test_a_coordinator_needs_something_to_generate_through() -> None:
    """A flow with neither a backend nor a way to build one would fail later and further from the cause."""
    with pytest.raises(ValueError, match="backend"):
        TextGenerationCoordinator(
            state=WorkerState(),
            shutdown_manager=Mock(),
            runtime_config=make_test_runtime_config(bridge_data=make_mock_bridge_data(scribe=True)),
            api_sessions=make_test_api_sessions(horde_client_session=_FakeHordeClientSession()),
            text_backend_kind=TEXT_BACKENDS.koboldcpp,
        )


def test_the_dry_run_stand_in_describes_itself_as_its_backend_would() -> None:
    """The dry run exercises the real name derivation, so its stand-in is spelled like the live backend."""
    koboldcpp_description = text_generation_coordinator.dry_run_backend_description(TEXT_BACKENDS.koboldcpp)
    aphrodite_description = text_generation_coordinator.dry_run_backend_description(TEXT_BACKENDS.aphrodite)

    assert koboldcpp_description.model_name.startswith("koboldcpp/")
    assert aphrodite_description.model_name.startswith("aphrodite/")


async def test_the_advertised_limits_are_the_lower_of_operator_and_backend() -> None:
    """Advertising past what the backend accepts draws jobs it then refuses."""
    coordinator, _backend, _session = _make_coordinator(max_length=512, max_context_length=8192)

    await coordinator.await_backend_ready()

    assert coordinator.advertisement is not None
    assert coordinator.advertisement.max_length == 512, "the operator's lower generation cap wins"
    assert coordinator.advertisement.max_context_length == 4096, "the backend's lower context cap wins"


async def test_no_soft_prompts_are_advertised_when_the_backend_has_none() -> None:
    """The worker cannot supply a soft prompt the backend does not have, so it claims none."""
    coordinator, _backend, session = _make_coordinator()

    await coordinator.await_backend_ready()
    await coordinator.api_text_pop()

    assert coordinator.advertisement is not None
    assert coordinator.advertisement.soft_prompts == ()
    assert session.pop_requests[0].softprompts == []


async def test_the_described_soft_prompts_are_advertised_exactly() -> None:
    """When the backend exposes soft prompts, the pop offers those and nothing else."""
    described = TextBackendDescription(
        model_name=_DESCRIPTION.model_name,
        max_context_length=_DESCRIPTION.max_context_length,
        max_length=_DESCRIPTION.max_length,
        soft_prompts=("a_soft_prompt", "another_soft_prompt"),
    )
    coordinator, _backend, session = _make_coordinator(backend=FakeTextBackend(description=described))

    await coordinator.await_backend_ready()
    await coordinator.api_text_pop()

    assert session.pop_requests[0].softprompts == ["a_soft_prompt", "another_soft_prompt"]


async def test_the_pop_carries_the_scribe_name_and_thread_count() -> None:
    """The horde sizes this worker's share of text work by what the pop declares."""
    coordinator, _backend, session = _make_coordinator(scribe_name="a-scribe", text_threads=4)

    await coordinator.await_backend_ready()
    await coordinator.api_text_pop()

    pop_request = session.pop_requests[0]
    assert pop_request.name == "a-scribe"
    assert pop_request.threads == 4
    assert pop_request.models == ["koboldcpp/Llama-3.2-3B-Instruct-Q4_K_M"]
    assert pop_request.bridge_agent is not None
    assert pop_request.bridge_agent.startswith("AI Horde Worker reGen:")


async def test_the_in_flight_ceiling_holds_further_pops() -> None:
    """`text_threads` is what the backend can run at once, so the flow holds no more than that."""
    coordinator, _backend, session = _make_coordinator(text_threads=2)
    await coordinator.await_backend_ready()

    for index in range(2):
        job = TextJobInFlight(job_id=f"held-{index}", payload={}, time_popped=0.0)
        coordinator._in_flight[job.job_id] = job

    await coordinator.api_text_pop()

    assert coordinator.num_in_flight == 2
    assert session.pop_requests == [], "the flow popped past its in-flight ceiling"


async def test_an_empty_pop_leaves_nothing_in_flight() -> None:
    """The usual answer when the horde has no text work is no work, which is not a failure."""
    coordinator, _backend, session = _make_coordinator()
    await coordinator.await_backend_ready()

    await coordinator.api_text_pop()

    assert len(session.pop_requests) == 1
    assert coordinator.num_in_flight == 0
    assert session.submit_requests == []


async def test_a_generated_job_is_submitted_as_ok_with_the_backend_text() -> None:
    """The happy path: the text the backend produced is what the horde is given."""
    backend = FakeTextBackend(description=_DESCRIPTION, response_text="a generated answer")
    session = _FakeHordeClientSession(pop_responses=[_pop_response()])
    coordinator, _backend, _session = _make_coordinator(backend=backend, session=session)
    await coordinator.await_backend_ready()

    await coordinator.api_text_pop()
    await _drain_job_tasks(coordinator)

    assert len(session.submit_requests) == 1
    submit_request = session.submit_requests[0]
    assert submit_request.state == GENERATION_STATE.ok
    assert submit_request.generation == "a generated answer"
    assert coordinator.num_jobs_submitted == 1
    assert coordinator.num_jobs_faulted == 0
    assert coordinator.kudos_earned_this_session == pytest.approx(_SUBMIT_REWARD)
    assert coordinator.num_in_flight == 0


async def test_the_payload_reaches_the_backend_as_the_horde_sent_it() -> None:
    """The horde owns the payload's shape and the backend owns its meaning; the worker owns neither."""
    backend = FakeTextBackend(description=_DESCRIPTION, response_text="answered")
    session = _FakeHordeClientSession(pop_responses=[_pop_response(prompt="the exact prompt")])
    coordinator, _backend, _session = _make_coordinator(backend=backend, session=session)
    await coordinator.await_backend_ready()

    await coordinator.api_text_pop()
    await _drain_job_tasks(coordinator)

    assert len(backend.generate_calls) == 1
    assert backend.generate_calls[0].payload == {"prompt": "the exact prompt"}


async def test_the_generation_key_is_the_job_id() -> None:
    """Abandoning a generation later means naming it, and the job id is the name both ends share."""
    job_id = str(uuid.uuid4())
    backend = FakeTextBackend(description=_DESCRIPTION, response_text="answered")
    session = _FakeHordeClientSession(pop_responses=[_pop_response(job_id=job_id)])
    coordinator, _backend, _session = _make_coordinator(backend=backend, session=session)
    await coordinator.await_backend_ready()

    await coordinator.api_text_pop()
    await _drain_job_tasks(coordinator)

    assert backend.generate_calls[0].generation_key == job_id
    assert str(session.submit_requests[0].id_) == job_id


async def test_a_busy_backend_is_re_offered_the_same_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Busy means healthy backend, fine payload, so nothing about the job needs to change."""
    monkeypatch.setattr(text_generation_coordinator, "BUSY_RETRY_WAIT_SECONDS", 0.0)
    backend = FakeTextBackend(
        description=_DESCRIPTION,
        response_text="answered at last",
        generate_failures=(TextBackendBusy("busy"), TextBackendBusy("still busy"), None),
    )
    coordinator, _backend, session = _make_coordinator(backend=backend)
    await coordinator.await_backend_ready()
    job = TextJobInFlight(job_id=_job_id(), payload={"prompt": "a prompt"}, time_popped=0.0)
    coordinator._in_flight[job.job_id] = job

    await coordinator.run_job(job)

    assert len(backend.generate_calls) == 3, "the same job was re-offered until the backend took it"
    assert {call.generation_key for call in backend.generate_calls} == {job.job_id}
    assert session.submit_requests[0].state == GENERATION_STATE.ok
    assert session.submit_requests[0].generation == "answered at last"


async def test_a_persistently_busy_backend_faults_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Holding a job the backend will not take only risks the server timing it out first."""
    monkeypatch.setattr(text_generation_coordinator, "BUSY_RETRY_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(text_generation_coordinator, "BUSY_RETRY_MAX_ATTEMPTS", 2)
    backend = FakeTextBackend(
        description=_DESCRIPTION,
        generate_failures=(TextBackendBusy("busy"), TextBackendBusy("busy"), TextBackendBusy("busy")),
    )
    coordinator, _backend, session = _make_coordinator(backend=backend)
    await coordinator.await_backend_ready()
    job = TextJobInFlight(job_id=_job_id(), payload={}, time_popped=0.0)
    coordinator._in_flight[job.job_id] = job

    await coordinator.run_job(job)

    assert len(backend.generate_calls) == 2
    assert session.submit_requests[0].state == GENERATION_STATE.faulted
    assert coordinator.num_jobs_faulted == 1


async def test_a_rejected_payload_faults_the_job_without_retrying() -> None:
    """A payload the backend refuses will be refused again, so re-offering it only burns time."""
    backend = FakeTextBackend(
        description=_DESCRIPTION,
        generate_failures=(TextBackendRejectedPayload("no"),),
    )
    coordinator, _backend, session = _make_coordinator(backend=backend)
    await coordinator.await_backend_ready()
    job = TextJobInFlight(job_id=_job_id(), payload={}, time_popped=0.0)
    coordinator._in_flight[job.job_id] = job

    await coordinator.run_job(job)

    assert len(backend.generate_calls) == 1, "a refused payload was re-offered"
    assert session.submit_requests[0].state == GENERATION_STATE.faulted
    assert session.submit_requests[0].generation == FAULTED_GENERATION_TEXT
    assert coordinator.advertisement is not None, "a refused payload says nothing about the backend"
    assert coordinator.num_in_flight == 0


async def test_an_unavailable_backend_faults_the_job_and_reopens_the_gate() -> None:
    """Every job fails the same way while the backend is down, so the flow stops popping and re-gates."""
    backend = FakeTextBackend(
        description=_DESCRIPTION,
        generate_failures=(TextBackendUnavailable("gone"),),
    )
    coordinator, _backend, session = _make_coordinator(backend=backend)
    await coordinator.await_backend_ready()
    job = TextJobInFlight(job_id=_job_id(), payload={}, time_popped=0.0)
    coordinator._in_flight[job.job_id] = job

    await coordinator.run_job(job)

    assert session.submit_requests[0].state == GENERATION_STATE.faulted
    assert coordinator.advertisement is None, "the flow kept advertising for a backend that had gone"
    assert [call.generation_key for call in backend.stop_calls] == [job.job_id]

    await coordinator.api_text_pop()
    assert session.pop_requests == [], "the flow popped while the readiness gate was open"


async def test_a_generation_past_its_deadline_is_abandoned_and_faulted() -> None:
    """A generation the worker has given up on must be abandoned, or it holds the backend's slot."""
    backend = FakeTextBackend(description=_DESCRIPTION, latency_seconds=5.0)
    coordinator, _backend, session = _make_coordinator(
        backend=backend,
        text_generation_timeout_seconds=0.05,
    )
    await coordinator.await_backend_ready()
    job = TextJobInFlight(job_id=_job_id(), payload={}, time_popped=0.0)
    coordinator._in_flight[job.job_id] = job

    await coordinator.run_job(job)

    assert backend.generate_calls[0].deadline_seconds == pytest.approx(0.05)
    assert [call.generation_key for call in backend.stop_calls] == [job.job_id]
    assert session.submit_requests[0].state == GENERATION_STATE.faulted


async def test_a_generation_that_never_returns_is_bounded_by_the_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flow keeps its own bound on the call, so a backend that never answers cannot pin a job task."""
    monkeypatch.setattr(text_generation_coordinator, "GENERATE_DEADLINE_GRACE_SECONDS", 0.05)
    backend = _NeverAnsweringTextBackend(description=_DESCRIPTION)
    coordinator, _backend, session = _make_coordinator(
        backend=backend,
        text_generation_timeout_seconds=0.05,
    )
    await coordinator.await_backend_ready()
    job = TextJobInFlight(job_id=_job_id(), payload={}, time_popped=0.0)
    coordinator._in_flight[job.job_id] = job

    await asyncio.wait_for(coordinator.run_job(job), timeout=5.0)

    assert [call.generation_key for call in backend.stop_calls] == [job.job_id]
    assert session.submit_requests[0].state == GENERATION_STATE.faulted


async def test_a_result_for_an_abandoned_generation_is_faulted_not_submitted() -> None:
    """An aborted koboldcpp generation answers with the tokens it had, and a partial answer is not an answer."""
    backend = FakeTextBackend(description=_DESCRIPTION, response_text="half an ans")
    coordinator, _backend, session = _make_coordinator(backend=backend)
    await coordinator.await_backend_ready()
    job = TextJobInFlight(job_id=_job_id(), payload={}, time_popped=0.0, stopped=True)
    coordinator._in_flight[job.job_id] = job

    await coordinator.run_job(job)

    submit_request = session.submit_requests[0]
    assert submit_request.state == GENERATION_STATE.faulted
    assert submit_request.generation == FAULTED_GENERATION_TEXT
    assert "half an ans" not in submit_request.generation
    assert coordinator.num_jobs_faulted == 1


async def test_shutdown_abandons_every_in_flight_generation_and_closes_the_backend() -> None:
    """The worker cannot be held open for a generation in another program, so it abandons and closes."""
    coordinator, backend, _session = _make_coordinator()
    await coordinator.await_backend_ready()
    for index in range(2):
        job = TextJobInFlight(job_id=f"in-flight-{index}", payload={}, time_popped=0.0)
        coordinator._in_flight[job.job_id] = job

    await coordinator.stop_in_flight_and_close()

    assert sorted(call.generation_key for call in backend.stop_calls) == ["in-flight-0", "in-flight-1"]
    assert all(job.stopped for job in coordinator._in_flight.values())
    assert backend.closed is True


async def test_closing_twice_is_harmless() -> None:
    """Shutdown paths can arrive more than once, and a second pass has nothing left to do."""
    coordinator, backend, _session = _make_coordinator()
    await coordinator.await_backend_ready()

    await coordinator.stop_in_flight_and_close()
    await coordinator.stop_in_flight_and_close()

    assert backend.closed is True


async def test_a_stale_job_on_submit_is_not_retried() -> None:
    """A job the server has already written off cannot be submitted, so the flow stops trying."""
    attempts: list[object] = []

    class _StaleSession:
        """Answers every submit with the server's "this generation is gone" error."""

        async def submit_request(self, request: object, response_type: object) -> object:
            attempts.append(request)
            return RequestErrorResponse(message="This generation does not exist")

    coordinator, _backend, _session = _make_coordinator()
    coordinator._api_sessions = make_test_api_sessions(horde_client_session=_StaleSession())
    job = TextJobInFlight(job_id=_job_id(), payload={}, time_popped=0.0)
    coordinator._in_flight[job.job_id] = job

    await coordinator.submit_job(job, generation="", state=GENERATION_STATE.faulted)

    assert len(attempts) == 1, "a stale job was re-submitted"
    assert coordinator.num_in_flight == 0


async def _drain_job_tasks(coordinator: TextGenerationCoordinator) -> None:
    """Wait for the per-job tasks a pop started, so a test can assert on their submits."""
    tasks = set(coordinator._job_tasks)
    if tasks:
        await asyncio.wait(tasks, timeout=10.0)


async def test_a_maintenance_refusal_is_announced_once_and_resumption_once() -> None:
    """A worker in maintenance is refused every pop; that is an operator state, so it is one line each way."""
    coordinator, _backend, session = _make_coordinator()
    assert await coordinator.await_backend_ready() is True
    refusal = RequestErrorResponse(rc=RC.WorkerMaintenance, message="owner-only traffic")
    logged: list[tuple[str, str]] = []
    sink_id = logger.add(lambda m: logged.append((m.record["level"].name, m.record["message"])), level="TRACE")
    try:
        coordinator._handle_pop_error_response(refusal)
        coordinator._handle_pop_error_response(refusal)
        coordinator._handle_pop_error_response(refusal)

        held_lines = [level for level, message in logged if "Text pops are held" in message]
        assert held_lines == ["INFO"]
        assert not any("API Error" in message for _level, message in logged)
        assert sum(1 for level, _message in logged if level == "TRACE") == 2

        coordinator._last_pop_time = 0.0
        await coordinator.api_text_pop()

        assert len(session.pop_requests) == 1
        assert [level for level, message in logged if "Text pops resumed" in message] == ["INFO"]
    finally:
        logger.remove(sink_id)
