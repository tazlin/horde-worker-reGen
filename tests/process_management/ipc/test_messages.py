"""Tests for IPC message models."""

from __future__ import annotations

import pickle

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE

from horde_worker_regen.consts import AESTHETIC_METADATA_TYPE
from horde_worker_regen.process_management.ipc.messages import (
    HordeImageResult,
    HordeInferenceResultMessage,
    SamplerTruncationReport,
    SampleSliceResult,
)
from tests.process_management.conftest import make_job_pop_response

_JOB_ID = make_job_pop_response(model="stable_diffusion").id_


class TestHordeInferenceResultMessage:
    """Tests for inference result message helpers."""

    def test_faults_count_ignores_non_reportable_metadata(self) -> None:
        """Only reportable generation metadata contributes to the fault count."""
        message = HordeInferenceResultMessage(
            process_id=2,
            process_launch_identifier=9,
            info="4.0 iterations per second",
            state=GENERATION_STATE.ok,
            time_elapsed=1.0,
            sdk_api_job_info=make_job_pop_response(model="stable_diffusion"),
            job_image_results=[
                HordeImageResult(
                    image_bytes=b"image",
                    generation_faults=[
                        GenMetadataEntry(
                            type=METADATA_TYPE.information,
                            value=METADATA_VALUE.see_ref,
                            ref="nsfw",
                        ),
                        GenMetadataEntry(
                            type=AESTHETIC_METADATA_TYPE,
                            value=METADATA_VALUE.see_ref,
                            ref="6.42",
                        ),
                        GenMetadataEntry(
                            type=METADATA_TYPE.censorship,
                            value=METADATA_VALUE.nsfw,
                        ),
                    ],
                ),
            ],
        )

        assert message.faults_count == 1


class TestSampleSliceResult:
    """Tests for the sample stage's per-slice result, which carries the sampler-truncation record."""

    def test_a_slice_without_a_truncation_round_trips_as_none(self) -> None:
        """The field is optional: a slice from an engine without the bounded sampler simply omits it."""
        result = SampleSliceResult(job_id=_JOB_ID, latent_bytes=b"latent", state=GENERATION_STATE.ok)

        round_tripped = pickle.loads(pickle.dumps(result))

        assert round_tripped.latent_bytes == b"latent"
        assert round_tripped.sampler_truncation is None

    def test_a_slice_carrying_a_truncation_round_trips_with_its_numbers(self) -> None:
        """The record has to survive the sampler-to-parent hop; the parent composes the disclosure from it."""
        result = SampleSliceResult(
            job_id=_JOB_ID,
            latent_bytes=b"latent",
            state=GENERATION_STATE.ok,
            sampler_truncation=SamplerTruncationReport(nominal_steps=20, iterations=25, budget_multiplier=1.25),
        )

        round_tripped = pickle.loads(pickle.dumps(result))

        assert round_tripped.sampler_truncation is not None
        assert round_tripped.sampler_truncation.nominal_steps == 20
        assert round_tripped.sampler_truncation.iterations == 25
        assert round_tripped.sampler_truncation.budget_multiplier == 1.25

    def test_a_slice_from_a_sender_predating_the_field_still_validates(self) -> None:
        """The field was added optional-with-default, so an older sender's payload stays acceptable."""
        result = SampleSliceResult.model_validate(
            {"job_id": _JOB_ID, "latent_bytes": b"latent", "state": GENERATION_STATE.ok},
        )

        assert result.sampler_truncation is None
