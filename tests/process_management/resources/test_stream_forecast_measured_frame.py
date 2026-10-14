"""The forecast's alone frame is the room the measured admission will find at sole residency.

``free_if_alone`` assumes every other tenant gone and charges no noise buffer; the dispatch that follows is
priced by the measured admission, which subtracts the noise buffer and sees the foreign floor and any lane
with no off-GPU actuator inside the device-free reading. A job priced against sole residency is compared with
the alone frame net of both terms, so the two frames agree on a card at its edge.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources import resource_budget

_MARGINAL_MB = 487.0


def _forecast(
    *,
    admission_noise_mb: float = 0.0,
    unpausable_tenancy_mb: float = 0.0,
    total_vram_mb: float = 24074.0,
) -> resource_budget.StreamForecast:
    """A whole-card forecast on a 24 GB card with one context's overhead."""
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
        admission_noise_mb=admission_noise_mb,
        unpausable_tenancy_mb=unpausable_tenancy_mb,
    )


def test_without_the_measured_terms_the_frame_is_the_alone_figure() -> None:
    """Zero noise and zero unpausable tenancy leave the alone frame unchanged."""
    forecast = _forecast()

    assert forecast._free_if_alone_measured_mb == forecast.free_if_alone_mb
    assert forecast.sole_residency_room_mb(0.0) == forecast.free_if_alone_mb


def test_the_frame_is_net_of_the_noise_and_the_tenancy_no_teardown_returns() -> None:
    """Room the admission will not grant is not room a sole-residency job is priced against."""
    forecast = _forecast(admission_noise_mb=1203.7, unpausable_tenancy_mb=2899.0 + 1250.0)
    alone_measured = forecast.free_if_alone_mb - 1203.7 - (2899.0 + 1250.0)

    assert forecast._free_if_alone_measured_mb == alone_measured
    assert forecast.sole_residency_room_mb(_MARGINAL_MB) == alone_measured - _MARGINAL_MB


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
