"""Coverage contracts for realistic and boundary-shaped popped image jobs.

The generated stateful suites consume a compact payload vocabulary. This module verifies that vocabulary at
the cheaper pure-decision altitude: every pair of payload dimensions is materialized, requirement extraction
matches the SDK object, and every pair of worker capability settings is judged against those requirements.
The cross-product is intentionally kept out of subprocess tests; these decisions are deterministic and gain
nothing from paying process startup for each row.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopRequest,
    ImageGenerateJobPopResponse,
    LorasPayloadEntry,
    TIPayloadEntry,
)

from horde_worker_regen.process_management.gpu.gpu_eligibility import (
    CardProfile,
    JobRequirements,
    card_can_serve,
    describe_job_requirements,
)
from horde_worker_regen.process_management.simulation._canned_scenarios import (
    GeneratingJobSource,
    SoakImageTemplate,
    make_canned_job,
)
from tests.process_management.conftest import make_mock_bridge_data

_SD15 = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1.value
_SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl.value

_PAYLOAD_AXES: dict[str, tuple[object, ...]] = {
    "baseline": (_SD15, _SDXL, "flux", None),
    "source": ("txt2img", "img2img", "remix", "inpainting", "outpainting"),
    "aux": ("none", "lora", "ti", "both"),
    "control": ("none", "canny", "qr_code", "custom_workflow"),
    "post_processing": ("none", "single", "chain"),
    "uncensored": (False, True),
    "geometry": ((64, 64), (64, 128), (128, 128)),
}

_CAPABILITY_AXES: dict[str, tuple[object, ...]] = {
    "allow_controlnet": (False, True),
    "allow_sdxl_controlnet": (False, True),
    "allow_lora": (False, True),
    "allow_post_processing": (False, True),
    "allow_img2img": (False, True),
    "allow_inpainting": (False, True),
    "nsfw": (False, True),
}

type _Row = dict[str, object]
type _Pair = tuple[str, object, str, object]


def _pairs(row: _Row, axes: dict[str, tuple[object, ...]]) -> set[_Pair]:
    """Return every named two-axis projection from one row."""
    return {(first, row[first], second, row[second]) for first, second in itertools.combinations(axes, 2)}


def _covering_array(axes: dict[str, tuple[object, ...]]) -> tuple[_Row, ...]:
    """Build a deterministic greedy pairwise array over the supplied independent axes."""
    candidates = [dict(zip(axes, values, strict=True)) for values in itertools.product(*axes.values())]
    uncovered = set().union(*(_pairs(row, axes) for row in candidates))
    selected: list[_Row] = []
    while uncovered:
        best = max(candidates, key=lambda row: (len(_pairs(row, axes) & uncovered), repr(row)))
        covered = _pairs(best, axes) & uncovered
        if not covered:
            raise AssertionError(f"axis vocabulary contains unreachable pairs: {sorted(uncovered, key=repr)}")
        selected.append(best)
        uncovered -= covered
        candidates.remove(best)
    return tuple(selected)


_PAYLOAD_ROWS = _covering_array(_PAYLOAD_AXES)
_CAPABILITY_ROWS = _covering_array(_CAPABILITY_AXES)
_HIGHER_ORDER_PAYLOAD_ROWS: tuple[_Row, ...] = (
    {
        "baseline": _SDXL,
        "source": "img2img",
        "aux": "both",
        "control": "qr_code",
        "post_processing": "chain",
        "uncensored": True,
        "geometry": (128, 128),
    },
    {
        "baseline": _SD15,
        "source": "inpainting",
        "aux": "both",
        "control": "canny",
        "post_processing": "chain",
        "uncensored": False,
        "geometry": (64, 128),
    },
    {
        "baseline": _SDXL,
        "source": "outpainting",
        "aux": "both",
        "control": "qr_code",
        "post_processing": "chain",
        "uncensored": True,
        "geometry": (128, 128),
    },
    {
        "baseline": "flux",
        "source": "img2img",
        "aux": "lora",
        "control": "none",
        "post_processing": "chain",
        "uncensored": False,
        "geometry": (128, 128),
    },
)


@dataclass(frozen=True)
class _MaterializedPayload:
    """A covering-array row paired with the SDK response built from it."""

    row: _Row
    job: ImageGenerateJobPopResponse


def _materialize_payload(row: _Row) -> _MaterializedPayload:
    """Build the SDK response denoted by one payload row."""
    aux = str(row["aux"])
    control = str(row["control"])
    post_processing = str(row["post_processing"])
    width, height = row["geometry"]  # type: ignore[misc]
    job = make_canned_job(
        "covered-model",
        width=width,
        height=height,
        loras=[LorasPayloadEntry(name="covered-lora")] if aux in {"lora", "both"} else None,
        tis=[TIPayloadEntry(name="covered-ti", inject_ti="prompt")] if aux in {"ti", "both"} else None,
        control_type="canny" if control == "canny" else None,
        workflow=control if control in {"qr_code", "custom_workflow"} else None,
        post_processing={
            "none": None,
            "single": ["GFPGAN"],
            "chain": ["GFPGAN", "RealESRGAN_x4plus"],
        }[post_processing],
        source_processing=str(row["source"]),
    )
    payload_data = job.payload.model_dump()
    payload_data["use_nsfw_censor"] = not bool(row["uncensored"])
    job_data = job.model_dump(by_alias=True)
    job_data["payload"] = payload_data
    return _MaterializedPayload(row=row, job=ImageGenerateJobPopResponse(**job_data))


_MATERIALIZED_PAYLOADS = tuple(_materialize_payload(row) for row in (*_PAYLOAD_ROWS, *_HIGHER_ORDER_PAYLOAD_ROWS))


def _expected_requirements(case: _MaterializedPayload) -> JobRequirements:
    """Return the independently stated eligibility meaning of one payload row."""
    row = case.row
    source = str(row["source"])
    control = str(row["control"])
    aux = str(row["aux"])
    width, height = row["geometry"]  # type: ignore[misc]
    baseline = row["baseline"]
    return JobRequirements(
        model="covered-model",
        baseline=baseline if isinstance(baseline, str) else None,
        weight_mb=2048.0,
        is_sdxl=baseline == _SDXL,
        needs_controlnet=control in {"canny", "qr_code"},
        needs_lora=aux in {"lora", "both"},
        needs_post_processing=row["post_processing"] != "none",
        needs_img2img=source != "txt2img" or control != "none",
        needs_inpainting=source in {"inpainting", "outpainting"},
        needs_nsfw=bool(row["uncensored"]),
        pixels=width * height,
        needs_sdxl_controlnet=control == "qr_code" or (baseline == _SDXL and control in {"canny", "qr_code"}),
        has_ti=aux in {"ti", "both"},
    )


@pytest.mark.parametrize("case", _MATERIALIZED_PAYLOADS, ids=lambda case: "-".join(map(str, case.row.values())))
def test_popped_payload_requirement_extraction_covers_real_job_structures(case: _MaterializedPayload) -> None:
    """SDK payloads map to the capability requirements their fields objectively imply."""
    actual = describe_job_requirements(case.job, case.row["baseline"], weight_mb=2048.0)  # type: ignore[arg-type]
    assert actual == _expected_requirements(case)


def _card(row: _Row) -> CardProfile:
    """Build one card whose effective feature settings are the supplied capability row."""
    bridge = make_mock_bridge_data(**row)
    bridge.max_pixels = 64 * 128
    return CardProfile(
        device_index=0,
        total_vram_mb=8192.0,
        config=bridge,
        served_models=frozenset({"covered-model"}),
    )


def _expected_eligibility(config: _Row, requirements: JobRequirements) -> bool:
    """Evaluate the public conjunction independently of the production helper."""
    if requirements.needs_controlnet and not config["allow_controlnet"]:
        return False
    if requirements.needs_sdxl_controlnet and not config["allow_sdxl_controlnet"]:
        return False
    if requirements.needs_lora and not config["allow_lora"]:
        return False
    if requirements.needs_post_processing and not config["allow_post_processing"]:
        return False
    if requirements.needs_img2img and not config["allow_img2img"]:
        return False
    if requirements.needs_inpainting and not config["allow_inpainting"]:
        return False
    if requirements.needs_nsfw and not config["nsfw"]:
        return False
    return requirements.pixels <= 64 * 128


@pytest.mark.parametrize("config", _CAPABILITY_ROWS, ids=lambda row: "-".join(str(value) for value in row.values()))
def test_capability_setting_pairs_route_the_payload_covering_array(
    config: _Row,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every feature-setting pair is exercised against every representative popped payload."""
    import hordelib.vram_planning as vram_planning

    monkeypatch.setattr(vram_planning, "compute_weight_budget_mb", lambda _total_vram_mb: 4096.0)
    card = _card(config)
    for case in _MATERIALIZED_PAYLOADS:
        requirements = _expected_requirements(case)
        assert card_can_serve(card, requirements) is _expected_eligibility(config, requirements), case.row


def _pop_request(**overrides: object) -> ImageGenerateJobPopRequest:
    """Build a permissive simulated pop request with one or more capability overrides."""
    values: dict[str, object] = {
        "apikey": "0000000000",
        "name": "coverage-worker",
        "models": ["covered-model"],
        "max_pixels": 128 * 128,
        "allow_img2img": True,
        "allow_painting": True,
        "allow_post_processing": True,
        "allow_controlnet": True,
        "allow_extended_controlnet": True,
        "allow_sdxl_controlnet": True,
        "allow_lora": True,
    }
    values.update(overrides)
    return ImageGenerateJobPopRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_mode", "withheld_setting", "needs_image", "needs_mask"),
    [
        ("txt2img", "none", False, False),
        ("img2img", "allow_img2img", True, False),
        ("remix", "allow_img2img", True, False),
        ("inpainting", "allow_painting", True, True),
        ("inpainting", "allow_img2img", True, True),
        ("outpainting", "allow_painting", True, True),
        ("outpainting", "allow_img2img", True, True),
    ],
)
def test_generated_pop_source_respects_source_mode_capabilities(
    source_mode: str,
    withheld_setting: str,
    needs_image: bool,
    needs_mask: bool,
) -> None:
    """Simulation only returns source-shaped jobs when the worker advertised the required capability."""
    template = SoakImageTemplate(model="covered-model", width=64, height=64, source_processing=source_mode)
    accepted = GeneratingJobSource([template], seed=1).next_pop_response(_pop_request())
    assert accepted.id_ is not None
    assert (accepted.source_image is not None) is needs_image
    assert (accepted.source_mask is not None) is needs_mask

    if withheld_setting != "none":
        rejected = GeneratingJobSource([template], seed=1).next_pop_response(
            _pop_request(**{withheld_setting: False}),
        )
        assert rejected.id_ is None


def test_payload_and_capability_arrays_cover_every_declared_pair() -> None:
    """The compact corpora retain every pair if either vocabulary changes."""
    for axes, rows in ((_PAYLOAD_AXES, _PAYLOAD_ROWS), (_CAPABILITY_AXES, _CAPABILITY_ROWS)):
        expected = {
            (first, first_value, second, second_value)
            for first, second in itertools.combinations(axes, 2)
            for first_value in axes[first]
            for second_value in axes[second]
        }
        actual = set().union(*(_pairs(row, axes) for row in rows))
        assert actual == expected


def test_selected_higher_order_interactions_are_explicit_rows() -> None:
    """High-coupling triples remain pinned even though pairwise coverage would not require them."""
    assert any(
        row["baseline"] == _SDXL
        and row["control"] == "qr_code"
        and row["aux"] == "both"
        and row["source"] == "img2img"
        for row in _HIGHER_ORDER_PAYLOAD_ROWS
    )
    assert any(
        row["source"] == "inpainting" and row["control"] == "canny" and row["post_processing"] == "chain"
        for row in _HIGHER_ORDER_PAYLOAD_ROWS
    )
    assert any(
        row["source"] == "outpainting" and row["baseline"] == _SDXL and row["uncensored"] is True
        for row in _HIGHER_ORDER_PAYLOAD_ROWS
    )
