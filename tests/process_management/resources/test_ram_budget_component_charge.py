"""UNet-only component charging for disaggregation-class jobs in :meth:`RamBudget.check_job`.

A disaggregation-class job's sampler stages only the UNet, so charging its whole checkpoint against system RAM
over-counts by the text encoders and VAE that never enter that process. The RAM budget prices such a job at the
UNet residual read from the checkpoint's component-identity sidecar, floored so a degenerate residual is never
priced near zero, clamped never to exceed the whole-checkpoint charge, and gated on the host RAM danger floor
structurally identically to the marginal reuse credit. This is a bookkeeping-honesty charge (a UNet reload is
cheap), not a strategy that assumes several UNets resident at once.

These are pure ``check_job``/predictor tests: the charge inputs are passed directly so the arithmetic is
asserted without the scheduler's sidecar plumbing.

These tests are authored to run in fake mode with no GPU. Run with ``AI_HORDE_TESTING=True pytest``.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.resources.resource_budget import (
    RamBudget,
    predict_job_unet_only_ram_mb,
)
from tests.process_management.conftest import make_job_pop_response

# The whole-checkpoint charge the disaggregation-class UNet charge is measured against.
_WHOLE_MB = 16000.0
_RESERVE_MB = 4096.0
_MB = 1024 * 1024


@pytest.fixture(autouse=True)
def _pin_whole_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the whole-checkpoint burden estimate so the component arithmetic is asserted against a fixed cost."""
    monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: _WHOLE_MB)


class TestUnetOnlyPredictor:
    """The residual-grounded predictor converts bytes to MB and never prices a UNet swap near zero."""

    def test_residual_bytes_convert_to_mb(self) -> None:
        """A plausible SDXL UNet residual (~5000 MB) is charged its MB value, above the floor."""
        residual_bytes = int(5000.0 * _MB)
        assert predict_job_unet_only_ram_mb(residual_bytes) == pytest.approx(5000.0)

    def test_small_residual_is_floored(self) -> None:
        """A degenerate near-zero residual is charged the component floor, never zero."""
        assert predict_job_unet_only_ram_mb(1) == pytest.approx(resource_budget._COMPONENT_STAGING_CHARGE_FLOOR_MB)

    def test_negative_residual_is_floored(self) -> None:
        """A malformed (negative) residual cannot underflow the charge below the floor."""
        assert predict_job_unet_only_ram_mb(-1) == pytest.approx(resource_budget._COMPONENT_STAGING_CHARGE_FLOOR_MB)


class TestComponentChargeSelection:
    """The disaggregated seam selects the UNet-only charge; missing sidecar and monolithic charge the whole."""

    def test_disaggregated_prices_unet_only(self) -> None:
        """A disaggregation-class job is priced at the passed UNet charge, not the whole checkpoint."""
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=17946.0,
            disaggregated=True,
            component_charge_mb=6000.0,
        )
        assert verdict.fits is True
        assert verdict.predicted_mb == pytest.approx(6000.0)
        assert verdict.uncredited_predicted_mb == pytest.approx(_WHOLE_MB)
        assert verdict.reusable_credit_mb == pytest.approx(_WHOLE_MB - 6000.0)

    def test_missing_sidecar_charges_whole(self) -> None:
        """``disaggregated`` with no component charge (no sidecar) falls back to the whole-checkpoint charge."""
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=17946.0,
            disaggregated=True,
            component_charge_mb=None,
        )
        assert verdict.predicted_mb == pytest.approx(_WHOLE_MB)
        assert verdict.reusable_credit_mb == 0.0
        assert verdict.uncredited_predicted_mb is None

    def test_monolithic_charges_whole(self) -> None:
        """A non-disaggregated job charges the whole checkpoint even if a component charge is passed."""
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=17946.0,
            disaggregated=False,
            component_charge_mb=6000.0,
        )
        assert verdict.predicted_mb == pytest.approx(_WHOLE_MB)
        assert verdict.reusable_credit_mb == 0.0


class TestComponentChargeSafety:
    """The component charge is clamped to the whole charge and gated on the danger floor like the reuse credit."""

    def test_component_charge_clamped_to_whole(self) -> None:
        """A component charge larger than the whole-checkpoint charge never inflates the verdict above it."""
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=60000.0,
            disaggregated=True,
            component_charge_mb=_WHOLE_MB + 5000.0,
        )
        assert verdict.predicted_mb == pytest.approx(_WHOLE_MB)
        assert verdict.reusable_credit_mb == 0.0

    def test_floor_gate_falls_back_to_whole(self) -> None:
        """A component charge that would push available below the danger floor reverts to the whole charge.

        The gate mirrors the reuse-credit path (available - charge - transient headroom >= floor): 17946 - 6000
        - 1024 = 10922 < 12000 floor, so the reduction is withdrawn and the conservative whole charge stands
        (which then defers). This is the never-price-a-stage-into-a-floor-breach contract.
        """
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=17946.0,
            disaggregated=True,
            component_charge_mb=6000.0,
            danger_floor_mb=12000.0,
        )
        assert verdict.fits is False
        assert verdict.predicted_mb == pytest.approx(_WHOLE_MB)
        assert verdict.reusable_credit_mb == 0.0
        assert verdict.uncredited_predicted_mb is None

    def test_resident_zero_charge_admits_stage_free(self) -> None:
        """A 0.0 component charge (checkpoint already staged on the target) admits with nothing to load."""
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=6000.0,
            disaggregated=True,
            component_charge_mb=0.0,
            danger_floor_mb=1024.0,
        )
        assert verdict.fits is True
        assert verdict.predicted_mb == pytest.approx(0.0)
        assert verdict.reusable_credit_mb == pytest.approx(_WHOLE_MB)

    def test_no_whole_estimate_admits_unpriced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no whole-checkpoint estimate the disaggregated branch admits unpriced (never wedges on unknown)."""
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: None)
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=1000.0,
            disaggregated=True,
            component_charge_mb=6000.0,
        )
        assert verdict.fits is True
        assert verdict.predicted_mb is None
