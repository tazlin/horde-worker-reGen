"""Marginal RAM credit for a reusable staging target in :meth:`RamBudget.check_job`.

A preload onto an idle inference slot that retains its unloaded model's pages reuses those pages, so its real
system-RAM growth is a fraction of a cold load. The RAM budget prices this by crediting the raw model cost with
a conservative fraction of the target's retained reusable RSS, floored so a swap is never priced at zero and
gated on the host RAM danger floor so a credit never admits into a floor breach.

These are pure ``check_job`` tests: the credit inputs (retained reusable MB, danger floor MB) are passed
directly so the arithmetic is asserted without the scheduler's measurement plumbing.

These tests are authored but NOT executed here: a live GPU worker occupies the machine and the standing
constraint forbids running pytest beside it. Run with ``AI_HORDE_TESTING=True pytest`` once the box is free.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.resources.resource_budget import RamBudget
from tests.process_management.conftest import make_job_pop_response

# The live-window figure: SDXL ~16000 MB predicted, 4096 MB reserve, ~17946 MB available. The full charge
# (16000 + 4096 = 20096) does not fit 17946 and defers; the credit is what admits the in-place swap.
_PREDICTED_MB = 16000.0
_RESERVE_MB = 4096.0
_AVAILABLE_MB = 17946.0


@pytest.fixture(autouse=True)
def _pin_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the burden estimate so the credit arithmetic is asserted against a fixed model cost."""
    monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: _PREDICTED_MB)


class TestRamBudgetMarginalCredit:
    """The reuse credit discounts the charge, never below the floor, never above the raw cost, floor-gated."""

    def test_full_charge_defers_without_credit(self) -> None:
        """The premise: with no credit the live-window SDXL charge does not fit and defers."""
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(job, "x", available_ram_mb=_AVAILABLE_MB)
        assert verdict.fits is False
        assert verdict.reusable_credit_mb == 0.0
        assert verdict.uncredited_predicted_mb is None

    def test_credited_admit_when_target_retains(self) -> None:
        """A retaining target's credit reduces the charge enough that the same job now fits.

        Retained reusable 6900 MB -> credit 6900 * 0.7 = 4830 MB -> effective max(3500, 16000 - 4830) = 11170
        MB. 17946 available covers 11170 + 4096 reserve, so the swap admits where the cold load deferred.
        """
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=_AVAILABLE_MB,
            reusable_credit_mb=6900.0,
            danger_floor_mb=4800.0,
        )
        assert verdict.fits is True
        assert verdict.predicted_mb == pytest.approx(11170.0)
        assert verdict.reusable_credit_mb == pytest.approx(4830.0)
        assert verdict.uncredited_predicted_mb == pytest.approx(_PREDICTED_MB)

    def test_floor_safety_withdraws_credit(self) -> None:
        """A credit that would push available below the danger floor after the transient spike is denied.

        The credited charge fits the budget, but with a high danger floor the floor-safety gate
        (available - charge - transient headroom >= floor) fails, so the raw charge stands and the job defers.
        This is the never-credit-into-a-floor-breach contract.
        """
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        # 17946 - 11170 (candidate) - 1024 (transient) = 5752 < 6000 floor -> credit withdrawn.
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=_AVAILABLE_MB,
            reusable_credit_mb=6900.0,
            danger_floor_mb=6000.0,
        )
        assert verdict.fits is False
        assert verdict.reusable_credit_mb == 0.0
        assert verdict.predicted_mb == pytest.approx(_PREDICTED_MB)
        assert verdict.uncredited_predicted_mb is None

    def test_zero_credit_leaves_full_charge(self) -> None:
        """A fresh or busy target contributes zero retained RSS, so the verdict is the ordinary full charge."""
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=_AVAILABLE_MB,
            reusable_credit_mb=0.0,
            danger_floor_mb=4800.0,
        )
        assert verdict.fits is False
        assert verdict.reusable_credit_mb == 0.0
        assert verdict.predicted_mb == pytest.approx(_PREDICTED_MB)

    def test_credit_never_prices_below_floor_charge(self) -> None:
        """A target retaining more than a model's worth of pages is still charged the floor, not zero."""
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=60000.0,
            reusable_credit_mb=100000.0,
            danger_floor_mb=1024.0,
        )
        assert verdict.predicted_mb == pytest.approx(resource_budget._MARGINAL_STAGING_CHARGE_FLOOR_MB)
        assert verdict.fits is True

    def test_credit_never_inflates_a_cheap_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job already cheaper than the floor is never raised to the floor by the credit machinery."""
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 2000.0)
        budget = RamBudget(reserve_mb=_RESERVE_MB)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(
            job,
            "x",
            available_ram_mb=60000.0,
            reusable_credit_mb=1000.0,
            danger_floor_mb=1024.0,
        )
        assert verdict.predicted_mb == pytest.approx(2000.0)
        assert verdict.reusable_credit_mb == 0.0
        assert verdict.uncredited_predicted_mb is None
