"""Tests for IPC message models."""

from __future__ import annotations

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE

from horde_worker_regen.consts import AESTHETIC_METADATA_TYPE
from horde_worker_regen.process_management.ipc.messages import HordeImageResult, HordeInferenceResultMessage
from tests.process_management.conftest import make_job_pop_response


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
