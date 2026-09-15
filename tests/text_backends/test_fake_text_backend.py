"""The fake text backend stands in for a real one in dry runs, so its scripting has to be trustworthy.

A stand-in that answered differently from what it was configured with, or that lost track of what it was
asked, would let a flow test pass while the flow was wrong. These tests pin the configured answers, the
failure script that lets a caller rehearse busy-then-succeed without a server, the deadline it honours,
and the record it keeps of every call.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from horde_worker_regen.text_backends import (
    FakeTextBackend,
    TextBackend,
    TextBackendBusy,
    TextBackendDescription,
    TextBackendRejectedPayload,
    TextBackendUnavailable,
)

_DESCRIPTION = TextBackendDescription(
    model_name="koboldcpp/a-model.gguf",
    max_context_length=4096,
    max_length=512,
    soft_prompts=("a_soft_prompt",),
)
_GENERATION_KEY = "0123ABCD"


def test_fake_satisfies_the_protocol() -> None:
    """The fake is accepted wherever a real backend is, which is the point of it living in the package."""
    assert isinstance(FakeTextBackend(description=_DESCRIPTION), TextBackend)


async def test_ready_is_true_and_records_the_deadline() -> None:
    """A configured fake has its model loaded by construction, and the caller's deadline is observable."""
    backend = FakeTextBackend(description=_DESCRIPTION)

    assert await backend.ready(deadline_seconds=2.5) is True
    assert [call.deadline_seconds for call in backend.ready_calls] == [2.5]


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
