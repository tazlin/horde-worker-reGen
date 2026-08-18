"""Tests for model-level VRAM serviceability arithmetic."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.model_serviceability import (
    ModelFootprintFigures,
    ModelServiceabilityTier,
    assess_model_serviceability,
    max_power_to_pixels,
)

_GB = 1024.0

# Core weights plus per-megapixel sampling activation, matching hordelib's burden seeds.
_SD15 = ModelFootprintFigures(weights_mb=3200.0, activation_per_megapixel_mb=900.0)
_SDXL = ModelFootprintFigures(weights_mb=4900.0, activation_per_megapixel_mb=1200.0)
_FLUX = ModelFootprintFigures(weights_mb=11500.0, activation_per_megapixel_mb=1500.0)


@pytest.mark.parametrize(
    ("card_gb", "figures", "expected"),
    [
        (8, _SD15, True),
        (8, _SDXL, True),
        (8, _FLUX, False),
        (16, _SD15, True),
        (16, _SDXL, True),
        (16, _FLUX, True),
        (24, _SD15, True),
        (24, _SDXL, True),
        (24, _FLUX, True),
    ],
)
def test_serviceability_scales_by_card_capacity(
    card_gb: int,
    figures: ModelFootprintFigures,
    expected: bool,
) -> None:
    """8GB excludes only models whose smallest job exceeds it; 16GB and 24GB keep fitting figures."""
    total_mb = card_gb * _GB

    verdict = assess_model_serviceability(
        total_vram_mb=total_mb,
        baseline_mb=1024.0,
        noise_buffer_mb=admission_noise_buffer_mb(total_mb),
        figures=figures,
    )

    assert verdict.serviceable is expected


def test_unknown_footprint_does_not_exclude_model() -> None:
    """Missing model figures abstain rather than de-listing a model."""
    verdict = assess_model_serviceability(
        total_vram_mb=8 * _GB,
        baseline_mb=1024.0,
        noise_buffer_mb=None,
        figures=None,
    )

    assert verdict.serviceable is True
    assert verdict.tier is ModelServiceabilityTier.SERVICEABLE


def test_sdxl_on_8gb_is_constrained_only_when_max_power_job_does_not_fit() -> None:
    """With ~6.1 GB usable SDXL fits at 512x512 but not at 1024x1024; the verdict names the largest fitting cap."""
    total_mb = 8 * _GB
    baseline_mb = 1536.0
    capacity_mb = total_mb - baseline_mb - admission_noise_buffer_mb(total_mb)

    fits = assess_model_serviceability(
        total_vram_mb=total_mb,
        baseline_mb=baseline_mb,
        noise_buffer_mb=None,
        figures=_SDXL,
        max_pixels=max_power_to_pixels(8),
    )
    assert fits.tier is ModelServiceabilityTier.SERVICEABLE

    constrained = assess_model_serviceability(
        total_vram_mb=total_mb,
        baseline_mb=baseline_mb,
        noise_buffer_mb=None,
        figures=_SDXL,
        max_pixels=max_power_to_pixels(64),
    )
    assert constrained.tier is ModelServiceabilityTier.CONSTRAINED
    assert constrained.serviceable is True
    cap = constrained.largest_fitting_max_power
    assert cap is not None
    assert 8 <= cap < 64
    assert _SDXL.footprint_mb(max_power_to_pixels(cap)) <= capacity_mb
    assert _SDXL.footprint_mb(max_power_to_pixels(cap + 1)) > capacity_mb
    assert "does NOT fit" in constrained.reason()
    assert f"largest fitting max_power is {cap}" in constrained.reason()


def test_unserviceable_when_smallest_job_does_not_fit() -> None:
    """Flux on 8GB cannot host even a 512x512 job, regardless of ``max_pixels``."""
    verdict = assess_model_serviceability(
        total_vram_mb=8 * _GB,
        baseline_mb=1024.0,
        noise_buffer_mb=None,
        figures=_FLUX,
        max_pixels=max_power_to_pixels(8),
    )

    assert verdict.tier is ModelServiceabilityTier.UNSERVICEABLE
    assert verdict.serviceable is False
    assert verdict.largest_fitting_max_power is None
