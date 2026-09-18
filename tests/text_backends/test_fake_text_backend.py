"""The fake text backend stands in for a real one in dry runs, so its scripting has to be trustworthy.

A stand-in that answered differently from what it was configured with, or that lost track of what it was
asked, would let a flow test pass while the flow was wrong. These tests pin the configured answers, the
readiness script that lets a caller rehearse a cold start, the failure script that lets a caller
rehearse busy-then-succeed without a server, the deadline it honours, and the record it keeps of every
call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from horde_worker_regen.text_backends import (
    FakeTextBackend,
    TextBackend,
    TextBackendBusy,
    TextBackendCredentialRefused,
    TextBackendDescription,
    TextBackendRejectedPayload,
    TextBackendUnavailable,
    TextGenerationProgress,
)

_DESCRIPTION = TextBackendDescription(
    model_name="koboldcpp/a-model.gguf",
    max_context_length=4096,
    max_length=512,
    soft_prompts=("a_soft_prompt",),
)
_GENERATION_KEY = "0123ABCD"


class _ProgressRecorder:
    """Collects every progress report a generation made, which is what the flow would be reading."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.reports: list[TextGenerationProgress] = []

    def __call__(self, progress: TextGenerationProgress) -> None:
        """Record one report."""
        self.reports.append(progress)


def test_fake_satisfies_the_protocol() -> None:
    """The fake is accepted wherever a real backend is, which is the point of it living in the package."""
    assert isinstance(FakeTextBackend(description=_DESCRIPTION), TextBackend)


async def test_ready_is_true_and_records_the_deadline() -> None:
    """A configured fake has its model loaded by construction, and the caller's deadline is observable."""
    backend = FakeTextBackend(description=_DESCRIPTION)

    assert await backend.ready(deadline_seconds=2.5) is True
    assert [call.deadline_seconds for call in backend.ready_calls] == [2.5]


async def test_the_readiness_script_answers_false_until_it_runs_out() -> None:
    """A cold backend answers not-ready for a while, which is the window a readiness gate exists for."""
    backend = FakeTextBackend(description=_DESCRIPTION, ready_results=(False, False, True))

    assert await backend.ready(deadline_seconds=1.0) is False
    assert await backend.ready(deadline_seconds=1.0) is False
    assert await backend.ready(deadline_seconds=1.0) is True
    assert await backend.ready(deadline_seconds=1.0) is True, "past the end of the script the fake is ready"
    assert len(backend.ready_calls) == 4, "a not-ready answer is still a call the flow made"


async def test_a_scripted_credential_refusal_refuses_every_verb() -> None:
    """A password-protected backend refuses the worker everywhere, which is what a flow has to handle."""
    backend = FakeTextBackend(description=_DESCRIPTION, credential_refused=True)

    with pytest.raises(TextBackendCredentialRefused):
        await backend.ready(deadline_seconds=1.0)
    with pytest.raises(TextBackendCredentialRefused):
        await backend.describe()
    with pytest.raises(TextBackendCredentialRefused):
        await backend.capabilities()
    with pytest.raises(TextBackendCredentialRefused):
        await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=1.0)

    assert len(backend.ready_calls) == 1, "a refused call is still a call the flow made"
    assert len(backend.generate_calls) == 1


async def test_a_corrected_credential_serves_again() -> None:
    """An operator fixing the password is the whole remedy, so the stand-in has to be able to rehearse it."""
    backend = FakeTextBackend(description=_DESCRIPTION, response_text="an answer", credential_refused=True)

    with pytest.raises(TextBackendCredentialRefused):
        await backend.ready(deadline_seconds=1.0)

    backend.credential_refused = False

    assert await backend.ready(deadline_seconds=1.0) is True
    result = await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=1.0)
    assert result.text == "an answer"


async def test_stop_and_close_are_unaffected_by_a_refused_credential() -> None:
    """Both run where a failure is already being handled, so neither may raise a second one."""
    backend = FakeTextBackend(description=_DESCRIPTION, credential_refused=True)

    await backend.stop(generation_key=_GENERATION_KEY)
    await backend.close()

    assert [call.generation_key for call in backend.stop_calls] == [_GENERATION_KEY]
    assert backend.closed is True


async def test_describe_returns_the_configured_description() -> None:
    """What the fake claims to be is exactly what it was configured with, so a flow advertises that."""
    backend = FakeTextBackend(description=_DESCRIPTION)

    assert await backend.describe() == _DESCRIPTION
    assert await backend.describe() == _DESCRIPTION
    assert backend.describe_call_count == 2


async def test_generate_returns_the_configured_text() -> None:
    """A fixed response text is the simplest configuration and covers most flow tests."""
    backend = FakeTextBackend(description=_DESCRIPTION, response_text="a fixed answer")

    result = await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=5.0)

    assert result.text == "a fixed answer"


async def test_generate_calls_a_configured_callable_with_the_payload() -> None:
    """A callable response lets a test make the answer depend on what was asked, as a real backend does."""

    def echo_the_prompt(payload: Mapping[str, object]) -> str:
        return f"answering: {payload['prompt']}"

    backend = FakeTextBackend(description=_DESCRIPTION, response_text=echo_the_prompt)

    result = await backend.generate(
        {"prompt": "a prompt"},
        generation_key=_GENERATION_KEY,
        deadline_seconds=5.0,
    )

    assert result.text == "answering: a prompt"


async def test_generate_records_every_payload_and_key_in_order() -> None:
    """A flow test asserts on what the flow forwarded, so the record keeps payloads as they arrived."""
    backend = FakeTextBackend(description=_DESCRIPTION)
    first_payload: dict[str, object] = {"prompt": "first", "nested": {"deeper": [1, 2]}}
    second_payload: dict[str, object] = {"prompt": "second"}

    await backend.generate(first_payload, generation_key="key-one", deadline_seconds=5.0)
    await backend.generate(second_payload, generation_key="key-two", deadline_seconds=7.5)

    assert [call.payload for call in backend.generate_calls] == [first_payload, second_payload]
    assert [call.generation_key for call in backend.generate_calls] == ["key-one", "key-two"]
    assert [call.deadline_seconds for call in backend.generate_calls] == [5.0, 7.5]


async def test_the_failure_script_raises_busy_twice_then_succeeds() -> None:
    """Busy-then-succeed is the sequence a flow's wait-and-retry has to survive, rehearsed with no server."""
    backend = FakeTextBackend(
        description=_DESCRIPTION,
        response_text="eventually",
        generate_failures=(TextBackendBusy("busy"), TextBackendBusy("still busy"), None),
    )

    for _busy_attempt in range(2):
        with pytest.raises(TextBackendBusy):
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=5.0)

    result = await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=5.0)

    assert result.text == "eventually"
    assert len(backend.generate_calls) == 3, "a failed generation is still a call the flow made"


async def test_the_failure_script_can_raise_an_unavailable_backend() -> None:
    """A fake can rehearse the backend going away mid-run."""
    backend = FakeTextBackend(
        description=_DESCRIPTION,
        generate_failures=(TextBackendUnavailable("gone"),),
    )

    with pytest.raises(TextBackendUnavailable):
        await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=5.0)


async def test_the_failure_script_can_raise_a_rejected_payload() -> None:
    """A fake can rehearse the job the backend will never accept, which the flow must fault rather than retry."""
    backend = FakeTextBackend(
        description=_DESCRIPTION,
        generate_failures=(TextBackendRejectedPayload("no"),),
    )

    with pytest.raises(TextBackendRejectedPayload):
        await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=5.0)


async def test_generations_past_the_end_of_the_script_succeed() -> None:
    """The script names the failures, not the successes, so a short script does not starve a long run."""
    backend = FakeTextBackend(
        description=_DESCRIPTION,
        response_text="fine",
        generate_failures=(TextBackendBusy("busy"),),
    )

    with pytest.raises(TextBackendBusy):
        await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=5.0)

    for _later_attempt in range(3):
        assert (await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=5.0)).text == "fine"


async def test_latency_within_the_deadline_still_returns_the_text() -> None:
    """Configured latency is not a failure on its own; only outliving the deadline is."""
    backend = FakeTextBackend(description=_DESCRIPTION, response_text="in time", latency_seconds=0.01)

    result = await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=5.0)

    assert result.text == "in time"


async def test_latency_past_the_deadline_is_unavailable() -> None:
    """A fake too slow for its deadline ends the way the real driver's timeout would, so the flow sees one case."""
    backend = FakeTextBackend(description=_DESCRIPTION, latency_seconds=1.0)

    with pytest.raises(TextBackendUnavailable):
        await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=0.01)


async def test_stop_records_the_keys_it_was_asked_to_abandon() -> None:
    """A flow test checks that shutdown abandoned the generation it started, by key."""
    backend = FakeTextBackend(description=_DESCRIPTION)

    await backend.stop(generation_key="key-one")
    await backend.stop(generation_key="key-two")

    assert [call.generation_key for call in backend.stop_calls] == ["key-one", "key-two"]


async def test_close_is_idempotent() -> None:
    """Shutdown paths may close more than once, so the fake tolerates it exactly as the driver does."""
    backend = FakeTextBackend(description=_DESCRIPTION)

    await backend.close()
    await backend.close()

    assert backend.closed is True


async def test_the_recorded_views_are_copies_the_caller_cannot_edit() -> None:
    """The record is read-only, so a test cannot rewrite the history it is asserting on."""
    backend = FakeTextBackend(description=_DESCRIPTION)
    await backend.ready(deadline_seconds=1.0)
    await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=1.0)
    await backend.stop(generation_key=_GENERATION_KEY)

    assert isinstance(backend.ready_calls, tuple)
    assert isinstance(backend.generate_calls, tuple)
    assert isinstance(backend.stop_calls, tuple)

    with pytest.raises(AttributeError):
        backend.generate_calls[0].generation_key = "rewritten"  # pyrefly: ignore - the refusal is the assertion


async def test_the_stand_in_streams_its_chunks_and_returns_their_concatenation() -> None:
    """A dry run has to exercise the progress path, so the stand-in arrives in pieces the way a backend does."""
    backend = FakeTextBackend(description=_DESCRIPTION, stream_chunks=("one ", "two ", "three"))
    progress = _ProgressRecorder()

    result = await backend.generate(
        {},
        generation_key=_GENERATION_KEY,
        deadline_seconds=5.0,
        on_progress=progress,
    )

    assert result.text == "one two three"
    assert [report.chunks_received for report in progress.reports] == [1, 2, 3]
    assert progress.reports[-1].characters_received == len("one two three")


async def test_a_generation_with_no_chunk_script_arrives_as_one_chunk() -> None:
    """The configured response text is still a generation, so it still reports itself arriving."""
    backend = FakeTextBackend(description=_DESCRIPTION, response_text="all at once")
    progress = _ProgressRecorder()

    result = await backend.generate(
        {},
        generation_key=_GENERATION_KEY,
        deadline_seconds=5.0,
        on_progress=progress,
    )

    assert result.text == "all at once"
    assert [report.chunks_received for report in progress.reports] == [1]


async def test_the_configured_finish_reason_is_reported() -> None:
    """Why a generation stopped is the backend's word, so a stand-in has to be able to say one."""
    backend = FakeTextBackend(description=_DESCRIPTION, response_text="done", finish_reason="stop")

    result = await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=5.0)

    assert result.finish_reason == "stop"


async def test_a_stand_in_without_statistics_reports_no_counts_and_no_rate() -> None:
    """A stock backend counts nothing, and the stand-in for one must not invent counts from its chunks."""
    backend = FakeTextBackend(description=_DESCRIPTION, stream_chunks=("a ", "b"))
    progress = _ProgressRecorder()

    result = await backend.generate(
        {},
        generation_key=_GENERATION_KEY,
        deadline_seconds=5.0,
        on_progress=progress,
    )

    assert (await backend.capabilities()).generation_stats is False
    assert all(report.completion_tokens is None for report in progress.reports)
    assert all(report.tokens_per_second is None for report in progress.reports)
    assert result.generated_tokens is None
    assert result.prompt_tokens is None


async def test_a_stand_in_with_statistics_reports_counts_and_a_rate() -> None:
    """The same switch the real capability probe answers decides whether a text job can show a rate."""
    backend = FakeTextBackend(
        description=_DESCRIPTION,
        stream_chunks=("a ", "b"),
        chunk_interval_seconds=0.01,
        generation_stats=True,
        stats_prompt_tokens=7,
        stats_tokens_per_chunk=5,
    )
    progress = _ProgressRecorder()

    result = await backend.generate(
        {},
        generation_key=_GENERATION_KEY,
        deadline_seconds=5.0,
        on_progress=progress,
    )

    assert (await backend.capabilities()).generation_stats is True
    assert [report.completion_tokens for report in progress.reports] == [5, 10]
    assert all(report.prompt_tokens == 7 for report in progress.reports)
    assert progress.reports[-1].tokens_per_second is not None
    assert result.prompt_tokens == 7
    assert result.generated_tokens == 10


async def test_a_stand_in_that_stalls_never_answers() -> None:
    """A backend that wedged part way through is what a caller's stall bound exists to end."""
    backend = FakeTextBackend(
        description=_DESCRIPTION,
        stream_chunks=("only this much",),
        stalls_after_chunks=True,
    )
    progress = _ProgressRecorder()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            backend.generate(
                {},
                generation_key=_GENERATION_KEY,
                deadline_seconds=60.0,
                on_progress=progress,
            ),
            timeout=0.2,
        )

    assert [report.chunks_received for report in progress.reports] == [1]
