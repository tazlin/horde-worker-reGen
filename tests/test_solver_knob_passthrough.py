"""The solver knobs and the schedule survive the trip from a job pop to the backend payload.

A worker does not read these values, it forwards them: the API puts them on the pop response, the SDK
converts the response into generic parameters, and the backend adapter turns those into its own
payload. Nothing in that chain would fail loudly if a field were dropped along the way, because every
one of them is optional and an absent knob renders a perfectly good image, just not the one that was
asked for. This walks a job carrying all of them through the real conversion functions and checks
what comes out the far end.
"""

from __future__ import annotations

import uuid
from typing import cast
from unittest.mock import Mock

import pytest
from horde_model_reference import ModelReferenceManager
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.generation_parameters.image.object_models import BasicImageGenerationParameters
from hordelib.pipeline.payload import ImageGenPayload

# The knobs land in the worker with an unreleased horde_sdk and horde_engine; until the pins are
# bumped to versions carrying them, skip rather than fail (as tests/test_reference_helper.py does).
pytestmark = pytest.mark.skipif(
    "sampler_eta" not in BasicImageGenerationParameters.model_fields,
    reason="installed horde_sdk predates the solver knobs",
)

SOLVER_KNOB_PAYLOAD: dict[str, float | int | str] = {
    "sampler_eta": 0.25,
    "sampler_s_noise": 1.1,
    "sampler_s_churn": 0.5,
    "sampler_s_tmin": 0.2,
    "sampler_s_tmax": 9.0,
    "sampler_solver_type": "heun",
    "sampler_order": 3,
    "flow_shift": 2.5,
}
"""Every knob this wave puts on the wire, at values no default would produce by accident."""

REQUESTED_SCHEDULE = "align_your_steps"
"""A schedule the backend builds itself rather than naming to ComfyUI, so it exercises the new path."""


def _job_pop_response_with_solver_knobs() -> ImageGenerateJobPopResponse:
    """Create a txt2img pop response carrying the schedule and every solver knob."""
    job_id = str(uuid.uuid4())
    payload: dict[str, object] = {
        "prompt": "a test prompt",
        "width": 512,
        "height": 512,
        "ddim_steps": 20,
        "n_iter": 1,
        "seed": "42",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": REQUESTED_SCHEDULE,
        **SOLVER_KNOB_PAYLOAD,
    }
    return ImageGenerateJobPopResponse(
        id=job_id,  # pyrefly: ignore - the alias is what the API sends
        ids=[job_id],
        model="Deliberate",
        payload=payload,  # pyrefly: ignore - validated by pydantic
        skipped={},  # pyrefly: ignore - validated by pydantic
        source_processing="txt2img",  # pyrefly: ignore - validated by pydantic
        # The conversion refuses a job with nowhere to upload to, so the response carries the
        # upload URL a real pop always does.
        r2_upload="https://example.invalid/upload",
    )


@pytest.fixture
def model_reference_manager() -> ModelReferenceManager:
    """Return the one reference lookup the SDK conversion needs, without disk or network state."""
    manager = Mock(spec=ModelReferenceManager)
    model_record = Mock()
    model_record.baseline = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1
    manager.query.return_value.where.return_value.first.return_value = model_record
    return cast(ModelReferenceManager, manager)


@pytest.fixture
def converted_payload(model_reference_manager: ModelReferenceManager) -> ImageGenPayload:
    """The backend payload a pop response carrying every knob converts into."""
    from horde_sdk.worker.dispatch.ai_horde.image.convert import (
        convert_image_job_pop_response_to_parameters,
    )
    from hordelib.pipeline.sdk_adapter import to_image_gen_payload

    conversion_result = convert_image_job_pop_response_to_parameters(
        api_response=_job_pop_response_with_solver_knobs(),
        model_reference_manager=model_reference_manager,
    )
    payload, faults = to_image_gen_payload(conversion_result.generation_parameters)
    assert faults == [], f"conversion recorded faults for a plain txt2img job: {faults}"
    return payload


def test_every_solver_knob_reaches_the_backend_payload(converted_payload: ImageGenPayload) -> None:
    """Each knob arrives with the value the job asked for, under the same name."""
    for field_name, requested_value in SOLVER_KNOB_PAYLOAD.items():
        assert getattr(converted_payload, field_name) == requested_value, (
            f"{field_name} did not survive the conversion chain"
        )


def test_the_requested_schedule_reaches_the_backend_payload(converted_payload: ImageGenPayload) -> None:
    """A schedule the backend generates itself is forwarded by name, not collapsed to a flag."""
    assert converted_payload.scheduler == REQUESTED_SCHEDULE


def test_the_knobs_reach_the_options_the_backend_builds_its_sampler_with(
    converted_payload: ImageGenPayload,
) -> None:
    """The knobs are not merely carried: they resolve to the arguments the sampler is built with."""
    options = converted_payload.solver_options()

    assert options["eta"] == SOLVER_KNOB_PAYLOAD["sampler_eta"]
    assert options["s_noise"] == SOLVER_KNOB_PAYLOAD["sampler_s_noise"]
    assert options["s_churn"] == SOLVER_KNOB_PAYLOAD["sampler_s_churn"]
    assert options["s_tmin"] == SOLVER_KNOB_PAYLOAD["sampler_s_tmin"]
    assert options["s_tmax"] == SOLVER_KNOB_PAYLOAD["sampler_s_tmax"]
    assert options["solver_type"] == SOLVER_KNOB_PAYLOAD["sampler_solver_type"]
    assert options["order"] == SOLVER_KNOB_PAYLOAD["sampler_order"]


def test_a_job_that_asks_for_none_of_them_carries_none_of_them(
    model_reference_manager: ModelReferenceManager,
) -> None:
    """The invariant the whole wave rests on: an ordinary job is unchanged by any of this."""
    from horde_sdk.worker.dispatch.ai_horde.image.convert import (
        convert_image_job_pop_response_to_parameters,
    )
    from hordelib.pipeline.sdk_adapter import to_image_gen_payload

    job_id = str(uuid.uuid4())
    plain_job = ImageGenerateJobPopResponse(
        id=job_id,  # pyrefly: ignore - the alias is what the API sends
        ids=[job_id],
        model="Deliberate",
        payload={  # pyrefly: ignore - validated by pydantic
            "prompt": "a test prompt",
            "width": 512,
            "height": 512,
            "ddim_steps": 20,
            "n_iter": 1,
            "seed": "42",
            "sampler_name": "k_euler",
        },
        skipped={},  # pyrefly: ignore - validated by pydantic
        source_processing="txt2img",  # pyrefly: ignore - validated by pydantic
        r2_upload="https://example.invalid/upload",
    )

    conversion_result = convert_image_job_pop_response_to_parameters(
        api_response=plain_job,
        model_reference_manager=model_reference_manager,
    )
    payload, _faults = to_image_gen_payload(conversion_result.generation_parameters)

    for field_name in SOLVER_KNOB_PAYLOAD:
        assert getattr(payload, field_name) is None, f"{field_name} appeared on a job that never asked for it"
    assert payload.solver_options() == {}
