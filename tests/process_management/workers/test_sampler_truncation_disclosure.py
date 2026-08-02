"""Tests for disclosing hordelib's adaptive-sampler coercion on the delivered generation.

hordelib bounds the one sampler that picks its own iteration count and hands back the best-effort
sample instead of running forever. The job succeeds, so the only place the requester can learn the
sample was truncated is the ``gen_metadata`` the worker submits alongside it.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from types import SimpleNamespace

from horde_sdk.ai_horde_api.apimodels import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE

from horde_worker_regen.consts import (
    GEN_METADATA_REF_MAX_LENGTH,
    sampler_truncation_disclosure_ref,
)
from horde_worker_regen.process_management.ipc.messages import (
    HordeImageResult,
    HordeInferenceResultMessage,
)
from horde_worker_regen.process_management.workers.inference_process import (
    read_sampler_truncation,
    sampler_truncation_disclosure,
)


@dataclass
class _StubTruncation:
    """Stands in for hordelib's ``SamplerTruncation`` so these tests need no engine install."""

    sampler: str = "dpm_adaptive"
    nominal_steps: int = 20
    iterations: int = 25
    budget_multiplier: float = 1.25
    capped: bool = True


def test_truncation_produces_exactly_one_information_entry() -> None:
    """A truncated sample is disclosed once, as non-reportable information carrying the numbers."""
    entries = sampler_truncation_disclosure(_StubTruncation())

    assert len(entries) == 1
    assert entries[0].type_ == METADATA_TYPE.information
    assert entries[0].value == METADATA_VALUE.see_ref
    assert entries[0].ref is not None
    assert "25 iterations" in entries[0].ref
    assert "1.25x the 20-step schedule" in entries[0].ref


def test_result_from_an_engine_without_the_bound_reads_as_untruncated() -> None:
    """The record is absent entirely on an engine build that predates the bounded sampler."""
    assert read_sampler_truncation(SimpleNamespace(rawpng=None, faults=[])) is None


def test_result_carrying_a_record_reads_it_back() -> None:
    """A result from an engine with the bound hands the record through unchanged."""
    truncation = _StubTruncation()

    assert read_sampler_truncation(SimpleNamespace(sampler_truncation=truncation)) is truncation


def test_no_truncation_produces_no_entry() -> None:
    """The overwhelmingly common case: a sampler that ran to its own completion says nothing."""
    assert sampler_truncation_disclosure(None) == []


def test_disclosure_quotes_the_multiplier_the_record_carries() -> None:
    """The bound's multiplier travels on the record, so a changed bound cannot desync the text."""
    entries = sampler_truncation_disclosure(_StubTruncation(budget_multiplier=2.0, nominal_steps=30, iterations=60))

    assert entries[0].ref is not None
    assert "2x the 30-step schedule" in entries[0].ref


def test_ref_never_exceeds_the_api_length_limit() -> None:
    """Absurd counts must be truncated worker-side; the API rejects a longer ``ref`` outright."""
    ref = sampler_truncation_disclosure_ref(
        iterations=10**200,
        nominal_steps=10**200,
        multiplier=1.25,
    )

    assert len(ref) == GEN_METADATA_REF_MAX_LENGTH
    assert GenMetadataEntry(type=METADATA_TYPE.information, value=METADATA_VALUE.see_ref, ref=ref).ref == ref


def test_disclosure_survives_the_child_to_parent_ipc_round_trip() -> None:
    """The entry rides the existing per-image faults list, which the process queue pickles."""
    entries = sampler_truncation_disclosure(_StubTruncation())
    result = HordeImageResult(image_bytes=b"png", generation_faults=entries)

    round_tripped = pickle.loads(pickle.dumps(result))

    assert isinstance(round_tripped, HordeImageResult)
    assert len(round_tripped.generation_faults) == 1
    assert round_tripped.generation_faults[0].type_ == METADATA_TYPE.information
    assert round_tripped.generation_faults[0].value == METADATA_VALUE.see_ref
    assert round_tripped.generation_faults[0].ref == entries[0].ref


def test_disclosure_does_not_count_as_a_reported_fault() -> None:
    """A coerced-but-delivered generation is not a faulted one; the count must stay clean."""
    assert METADATA_TYPE.information in HordeInferenceResultMessage.non_reportable_faults

    message = HordeInferenceResultMessage.model_construct(
        job_image_results=[
            HordeImageResult(
                image_bytes=b"png",
                generation_faults=sampler_truncation_disclosure(_StubTruncation()),
            ),
        ],
    )

    assert message.faults_count == 0
