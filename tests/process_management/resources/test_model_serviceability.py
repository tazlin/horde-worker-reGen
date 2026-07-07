"""Tests for model-level VRAM serviceability arithmetic."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.model_serviceability import (
    ModelFootprintFigures,
    assess_model_serviceability,
)

_GB = 1024.0

_SD15 = ModelFootprintFigures(weights_mb=3200.0, minimum_activation_mb=236.0)
_SDXL = ModelFootprintFigures(weights_mb=4900.0, minimum_activation_mb=2415.0)
_FLUX = ModelFootprintFigures(weights_mb=11500.0, minimum_activation_mb=1893.0)


@pytest.mark.parametrize(
    ("card_gb", "figures", "expected"),
    [
        (8, _SD15, True),
        (8, _SDXL, False),
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
    """8GB excludes only models whose arithmetic exceeds it; 16GB and 24GB keep fitting figures."""
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
