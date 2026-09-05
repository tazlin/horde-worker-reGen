"""The forecast's alone frame retires a whole-card claim only against room the measured admission will find.

``free_if_alone`` assumes every other tenant gone and charges no noise buffer; the dispatch that follows is
priced by the measured admission, which subtracts the noise buffer and sees the foreign floor and any lane
with no off-GPU actuator inside the device-free reading. On a card at its edge the two frames disagreed:
the forecast retired the claim (siblings kept) and the admission then refused the dispatch for the room those
siblings held. These tests pin the alone frame net of both terms, and that a model with ample room (Flux
fp8 on a 24 GB card) still retires its claim, so the anti-churn purpose of the rule is unchanged.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources import resource_budget

_MARGINAL_MB = 487.0


def _forecast(
    *,
    measured_footprint_mb: float,
    admission_noise_mb: float = 0.0,
    unpausable_tenancy_mb: float = 0.0,
    total_vram_mb: float = 24074.0,
) -> resource_budget.StreamForecast:
    """A whole-card-intent forecast on a 24 GB card with one context's overhead, measured over five runs."""
    return resource_budget.StreamForecast(
        weights_mb=17000.0,
        footprint_mb=19500.0,
        reserve_mb=2048.0,
        base_reserve_mb=1519.0,
        free_now_mb=20157.0,
        free_if_alone_mb=total_vram_mb - 488.0,
        free_after_model_evict_mb=20157.0,
        total_vram_mb=total_vram_mb,
        per_process_overhead_mb=488.0,
        marginal_process_overhead_mb=_MARGINAL_MB,
        wants_whole_card=True,
        measured_footprint_mb=measured_footprint_mb,
        measured_observation_count=5,
        admission_noise_mb=admission_noise_mb,
        unpausable_tenancy_mb=unpausable_tenancy_mb,
    )


def test_without_the_measured_terms_the_rule_is_unchanged() -> None:
    """Zero noise and zero unpausable tenancy reproduce the prior arithmetic exactly."""
    forecast = _forecast(measured_footprint_mb=18580.0)

    assert forecast._free_if_alone_measured_mb == forecast.free_if_alone_mb
    assert forecast.measured_retires_whole_card_intent is True


def test_a_measured_figure_inside_the_noise_and_lane_band_keeps_the_claim() -> None:
    """Room the admission will not grant is not room the forecast may retire a claim on."""
    forecast = _forecast(
        measured_footprint_mb=18580.0,
        admission_noise_mb=1203.7,
        unpausable_tenancy_mb=2899.0 + 1250.0,
    )
    alone_measured = forecast.free_if_alone_mb - 1203.7 - (2899.0 + 1250.0)

    assert forecast._free_if_alone_measured_mb == alone_measured
    assert alone_measured - 18580.0 < _MARGINAL_MB
    assert forecast.measured_retires_whole_card_intent is False
    assert forecast.needs_exclusive_residency is True


def test_a_model_with_ample_room_still_retires_the_claim() -> None:
    """Flux fp8 on a 24 GB card keeps its siblings: about nine gigabytes remain after every measured term."""
    forecast = _forecast(
        measured_footprint_mb=11500.0,
        admission_noise_mb=1203.7,
        unpausable_tenancy_mb=1250.0,
    )

    assert forecast.measured_retires_whole_card_intent is True
    assert forecast.needs_exclusive_residency is False


def test_the_builder_carries_the_terms_and_floors_them_at_zero() -> None:
    """The public builder passes both terms through and never records a negative one."""
    from tests.process_management.conftest import make_job_pop_response

    forecast = resource_budget.forecast_weight_streaming(
        make_job_pop_response("Z-Image-Turbo"),
        "z_image_turbo",
        free_now_mb=20157.0,
        total_vram_mb=24074.0,
        per_process_overhead_mb=488.0,
        num_inference_processes=2,
        configured_reserve_floor_mb=0.0,
        admission_noise_mb=1203.7,
        unpausable_tenancy_mb=-5.0,
    )

    assert forecast.admission_noise_mb == 1203.7
    assert forecast.unpausable_tenancy_mb == 0.0
