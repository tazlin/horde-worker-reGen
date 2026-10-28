"""The rules that decide whether the image-utilities lane should be up, one row per rule and one for their order."""

from __future__ import annotations

import dataclasses

import pytest
from horde_sdk.generation_parameters.alchemy.consts import KNOWN_ALCHEMY_FORMS, KNOWN_MISC_POST_PROCESSORS

from horde_worker_regen.consts import EXTENDED_CONTROL_TYPES
from horde_worker_regen.process_management.scheduling.workload_flow import (
    AlchemyFormDemand,
    UtilitiesLaneDemandInputs,
    UtilitiesLaneDemandReason,
    job_wants_pre_annotation,
    post_processing_is_strip_only,
    utilities_lane_demand,
)
from tests.process_management.test_image_utilities_annotation import _make_controlnet_job

_STRIP = KNOWN_MISC_POST_PROCESSORS.strip_background.value
_ANNOTATION = KNOWN_ALCHEMY_FORMS.annotation.value
_CAPTION = KNOWN_ALCHEMY_FORMS.caption.value
_UPSCALER = "RealESRGAN_x4plus"

_IDLE = UtilitiesLaneDemandInputs(
    lane_available=True,
    shutting_down=False,
    alchemy=AlchemyFormDemand(),
    jobs_pending_strip=0,
    strip_only_jobs_before_strip=0,
    pre_annotation_candidates=0,
    jobs_pending_annotation=0,
)
"""An enabled lane with nothing to serve."""


def _with(**changes: object) -> UtilitiesLaneDemandInputs:
    return dataclasses.replace(_IDLE, **changes)  # type: ignore[arg-type]


def test_an_enabled_lane_with_nothing_to_serve_is_not_wanted() -> None:
    """No rule holds, so the lane stays down."""
    assert utilities_lane_demand(_IDLE) is None


@pytest.mark.parametrize(
    "inputs",
    [
        _with(lane_available=False, jobs_pending_strip=1),
        _with(shutting_down=True, jobs_pending_strip=1),
        _with(lane_available=False, alchemy=AlchemyFormDemand(offerable=frozenset({_STRIP}))),
        _with(shutting_down=True, alchemy=AlchemyFormDemand(offerable=frozenset({_STRIP}))),
    ],
    ids=["unavailable-with-a-strip", "shutting-down-with-a-strip", "unavailable-warm", "shutting-down-warm"],
)
def test_a_disabled_lane_or_a_shutdown_wins_over_every_demand(inputs: UtilitiesLaneDemandInputs) -> None:
    """Rule 1 is checked first, so no later rule brings up a lane that cannot or must not run."""
    assert utilities_lane_demand(inputs) is None


def test_extended_controlnet_has_no_warm_rule() -> None:
    """The predicate reads no extended-ControlNet opt-in, so the opt-in alone does not keep the lane up."""
    assert "extended_controlnet_opted_in" not in {
        field.name for field in dataclasses.fields(UtilitiesLaneDemandInputs)
    }
    assert all("extended" not in reason.value for reason in UtilitiesLaneDemandReason)


@pytest.mark.parametrize("control_type", sorted(EXTENDED_CONTROL_TYPES))
def test_a_pending_extended_controlnet_job_is_demand(control_type: str) -> None:
    """An extended job with a source image wants pre-annotation, which is the ControlNet annotation rule."""
    assert job_wants_pre_annotation(_make_controlnet_job(control_type=control_type))
    assert utilities_lane_demand(_with(pre_annotation_candidates=1)) is UtilitiesLaneDemandReason.CONTROLNET_ANNOTATION


@pytest.mark.parametrize("form", [_STRIP, _ANNOTATION], ids=["strip_background", "annotation"])
def test_an_offerable_lane_form_keeps_the_lane_warm(form: str) -> None:
    """Every form the routing sends to the lane counts, read through the routing itself."""
    inputs = _with(alchemy=AlchemyFormDemand(offerable=frozenset({_CAPTION, form})))
    assert utilities_lane_demand(inputs) is UtilitiesLaneDemandReason.ALCHEMY_FORM_WARM


def test_offerable_forms_served_elsewhere_do_not_keep_the_lane_warm() -> None:
    """Caption and upscaler forms run on other lanes, so they do not want this one."""
    inputs = _with(alchemy=AlchemyFormDemand(offerable=frozenset({_CAPTION, _UPSCALER})))
    assert utilities_lane_demand(inputs) is None


def test_a_job_pending_its_strip_is_demand() -> None:
    """A strip waiting for or running on the lane needs it."""
    assert utilities_lane_demand(_with(jobs_pending_strip=1)) is UtilitiesLaneDemandReason.PENDING_STRIP


def test_a_strip_only_job_before_its_strip_is_demand() -> None:
    """A strip-only job counts from pop, since its generation time covers a lane start."""
    assert utilities_lane_demand(_with(strip_only_jobs_before_strip=2)) is UtilitiesLaneDemandReason.STRIP_ONLY_JOB


@pytest.mark.parametrize(
    ("post_processing", "strip_only"),
    [
        ([_STRIP], True),
        ([_UPSCALER, _STRIP], False),
        ([_UPSCALER], False),
        ([], False),
    ],
    ids=["strip-only", "chained", "upscale-only", "none"],
)
def test_only_a_job_whose_post_processing_is_the_strip_alone_counts_before_its_strip(
    post_processing: list[str],
    strip_only: bool,
) -> None:
    """A chained job counts once it reaches ``PENDING_STRIP``, after its post-processing chain."""
    assert post_processing_is_strip_only(post_processing) is strip_only


@pytest.mark.parametrize(
    "inputs",
    [_with(pre_annotation_candidates=1), _with(jobs_pending_annotation=1)],
    ids=["pre-annotation-candidate", "pending-annotation"],
)
def test_controlnet_annotation_is_demand(inputs: UtilitiesLaneDemandInputs) -> None:
    """A pre-annotation candidate and a parked job both want the lane."""
    assert utilities_lane_demand(inputs) is UtilitiesLaneDemandReason.CONTROLNET_ANNOTATION


@pytest.mark.parametrize("form", [_STRIP, _ANNOTATION], ids=["strip_background", "annotation"])
def test_a_lane_form_pending_or_in_flight_is_demand(form: str) -> None:
    """A popped lane form waits for the lane, so it needs it."""
    inputs = _with(alchemy=AlchemyFormDemand(waiting=(_CAPTION, form)))
    assert utilities_lane_demand(inputs) is UtilitiesLaneDemandReason.ALCHEMY_FORM_WAITING


def test_forms_waiting_for_other_lanes_are_not_demand() -> None:
    """Forms routed to the CLIP stack or the post-processing lane do not want this one."""
    inputs = _with(alchemy=AlchemyFormDemand(waiting=(_CAPTION, _UPSCALER)))
    assert utilities_lane_demand(inputs) is None


def test_the_first_rule_that_holds_names_the_reason() -> None:
    """With every rule holding, the reason follows the documented order."""
    every_rule = _with(
        alchemy=AlchemyFormDemand(offerable=frozenset({_STRIP}), waiting=(_STRIP,)),
        jobs_pending_strip=1,
        strip_only_jobs_before_strip=1,
        pre_annotation_candidates=1,
        jobs_pending_annotation=1,
    )
    expected_order = [
        (UtilitiesLaneDemandReason.ALCHEMY_FORM_WARM, {"alchemy": AlchemyFormDemand(waiting=(_STRIP,))}),
        (UtilitiesLaneDemandReason.PENDING_STRIP, {"jobs_pending_strip": 0}),
        (UtilitiesLaneDemandReason.STRIP_ONLY_JOB, {"strip_only_jobs_before_strip": 0}),
        (
            UtilitiesLaneDemandReason.CONTROLNET_ANNOTATION,
            {"pre_annotation_candidates": 0, "jobs_pending_annotation": 0},
        ),
        (UtilitiesLaneDemandReason.ALCHEMY_FORM_WAITING, {"alchemy": AlchemyFormDemand()}),
    ]
    inputs = every_rule
    for reason, lift in expected_order:
        assert utilities_lane_demand(inputs) is reason
        inputs = dataclasses.replace(inputs, **lift)  # type: ignore[arg-type]
    assert utilities_lane_demand(inputs) is None
