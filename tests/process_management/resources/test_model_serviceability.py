"""Tests for model-level VRAM serviceability arithmetic."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.model_serviceability import (
    ConstrainedLaneState,
    ModelFootprintFigures,
    ModelServiceabilityTier,
    assess_model_serviceability,
    decide_constrained_offer,
    max_power_to_pixels,
    model_footprint_figures_for_baseline,
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


# A streaming baseline: core weights that no consumer card can hold resident, with hordelib's recommended
# minimum card carried alongside (the qwen_image seed shape).
_STREAMING = ModelFootprintFigures(
    weights_mb=19500.0,
    activation_per_megapixel_mb=1500.0,
    min_recommended_card_mb=16000.0,
)


class TestStreamingBaselineServiceability:
    """A model whose core weights cannot be resident is judged by its recommended minimum card."""

    def test_a_card_meeting_the_recommended_minimum_serves_a_streaming_model_at_full_size(self) -> None:
        """24 GB minus a desktop baseline: the weights overflow, the recommended card fits, no size cap."""
        verdict = assess_model_serviceability(
            total_vram_mb=24 * _GB,
            baseline_mb=3600.0,
            noise_buffer_mb=None,
            figures=_STREAMING,
            max_pixels=max_power_to_pixels(32),
        )
        assert verdict.capacity_mb is not None and _STREAMING.streams_on(verdict.capacity_mb)
        assert verdict.tier is ModelServiceabilityTier.SERVICEABLE
        assert "streams" in verdict.reason()

    def test_a_card_below_the_recommended_minimum_cannot_serve_a_streaming_model(self) -> None:
        """16 GB minus a baseline sits under the recommended card: not offered at any size."""
        verdict = assess_model_serviceability(
            total_vram_mb=16 * _GB,
            baseline_mb=1500.0,
            noise_buffer_mb=None,
            figures=_STREAMING,
        )
        assert verdict.tier is ModelServiceabilityTier.UNSERVICEABLE

    def test_an_unknown_recommended_minimum_keeps_the_conservative_answer(self) -> None:
        """Without a recommended card there is nothing to judge a streaming fit by, so it does not fit."""
        figures = ModelFootprintFigures(weights_mb=19500.0, activation_per_megapixel_mb=1500.0)
        verdict = assess_model_serviceability(
            total_vram_mb=24 * _GB,
            baseline_mb=3600.0,
            noise_buffer_mb=None,
            figures=figures,
        )
        assert verdict.tier is ModelServiceabilityTier.UNSERVICEABLE
        assert "unknown" in verdict.reason()

    def test_a_resident_baseline_is_not_judged_by_its_recommendation(self) -> None:
        """SDXL on 8 GB: the weights fit, so the resident inequality governs even with a recommendation set.

        The recommendation may sit above what a small card achieves; a model the card provably seats must not
        be delisted by it.
        """
        sdxl = ModelFootprintFigures(
            weights_mb=4900.0,
            activation_per_megapixel_mb=1200.0,
            min_recommended_card_mb=8000.0,
        )
        verdict = assess_model_serviceability(
            total_vram_mb=8 * _GB,
            baseline_mb=512.0,
            noise_buffer_mb=None,
            figures=sdxl,
        )
        assert verdict.tier is not ModelServiceabilityTier.UNSERVICEABLE


class TestConstrainedOfferLane:
    """A constrained model rides its own capped pop instead of capping every other model's pop."""

    _OFFER = frozenset({"heavy", "light", "other"})

    def test_no_constrained_model_leaves_the_offer_alone(self) -> None:
        """With no constrained model the offer and max_power pass through and the cadence resets."""
        decision = decide_constrained_offer(
            ConstrainedLaneState(full_cycles_taken=2),
            offered_models=self._OFFER,
            model_caps={},
            pop_max_power=32,
        )
        assert decision.advertised_models == self._OFFER
        assert decision.pop_max_power == 32
        assert decision.constrained_pop is False
        assert decision.next_state == ConstrainedLaneState()

    def test_full_size_pops_alternate_with_one_constrained_pop(self) -> None:
        """Three full-size pops of the unconstrained models, then one pop of the constrained model at its cap."""
        state = ConstrainedLaneState()
        seen: list[tuple[frozenset[str], int, bool]] = []
        for _ in range(8):
            decision = decide_constrained_offer(
                state,
                offered_models=self._OFFER,
                model_caps={"heavy": 19},
                pop_max_power=32,
                full_cycles=3,
            )
            seen.append((decision.advertised_models, decision.pop_max_power, decision.constrained_pop))
            state = decision.next_state
        full = (frozenset({"light", "other"}), 32, False)
        constrained = (frozenset({"heavy"}), 19, True)
        assert seen == [full, full, full, constrained, full, full, full, constrained]

    def test_a_wholly_constrained_offer_is_capped_as_one_pop(self) -> None:
        """With nothing to protect, every model goes out capped at the smallest cap."""
        decision = decide_constrained_offer(
            ConstrainedLaneState(),
            offered_models=frozenset({"heavy", "heavier"}),
            model_caps={"heavy": 19, "heavier": 12},
            pop_max_power=32,
        )
        assert decision.advertised_models == frozenset({"heavy", "heavier"})
        assert decision.pop_max_power == 12
        assert decision.constrained_pop is False

    def test_an_idle_fill_pop_never_takes_the_constrained_lane(self) -> None:
        """An idle-fill pop wants the quickest work of any model: capped, not laned, and the cadence holds."""
        state = ConstrainedLaneState(full_cycles_taken=3)
        decision = decide_constrained_offer(
            state,
            offered_models=self._OFFER,
            model_caps={"heavy": 19},
            pop_max_power=32,
            idle_fill=True,
        )
        assert decision.advertised_models == self._OFFER
        assert decision.pop_max_power == 19
        assert decision.constrained_pop is False
        assert decision.next_state == state

    def test_a_cap_never_exceeds_the_configured_max_power(self) -> None:
        """A cap above the configured max_power is clamped to it on the constrained pop."""
        decision = decide_constrained_offer(
            ConstrainedLaneState(full_cycles_taken=3),
            offered_models=self._OFFER,
            model_caps={"heavy": 40},
            pop_max_power=32,
        )
        assert decision.pop_max_power == 32


def test_footprint_figures_forward_the_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The serviceability footprint is looked up by model name so a per-model override applies."""
    import hordelib.feature_impact as feature_impact

    calls: list[tuple[str, str | None]] = []

    class _StubBurden:
        vram_per_megapixel_mb = 1500.0
        min_recommended_vram_mb = 14000.0

        def resident_weight_estimate_mb(self) -> float:
            return 12600.0

    def recorder(baseline: str, model_name: str | None = None) -> object:
        calls.append((baseline, model_name))
        return _StubBurden()

    monkeypatch.setattr(feature_impact, "get_baseline_burden", recorder)

    figures = model_footprint_figures_for_baseline("qwen_image", "Krea2-Turbo_fp8")

    assert figures is not None
    assert figures.weights_mb == 12600.0
    assert calls == [("qwen_image", "Krea2-Turbo_fp8")]
