"""Tests for the worker-owned VRAM budget and its scheduler gating."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.resources.admission_identity import (
    _ADMISSION_NOISE_BUFFER_MB,
    admission_noise_buffer_mb,
    evaluate_admission,
)
from horde_worker_regen.process_management.resources.resource_budget import (
    BudgetVerdict,
    CommittedReserveLedger,
    RamBudget,
    VramBudget,
    assess_ram_pressure,
    ram_pressure_floor_mb,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_job,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


class TestVramBudget:
    """Unit tests for the VramBudget accountant itself (prediction stubbed)."""

    def test_cold_start_admits_when_no_telemetry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no VRAM telemetry yet, the budget admits so a cold worker never wedges."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 9999.0)
        budget = VramBudget(reserve_mb=2048.0)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(job, "stable_diffusion_1", free_vram_mb=None)
        assert verdict.fits is True
        assert verdict.available_mb is None

    def test_admits_when_estimate_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A None estimate means unknown cost; the budget admits rather than blocking blindly."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: None)
        budget = VramBudget(reserve_mb=2048.0)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(job, None, free_vram_mb=500.0)
        assert verdict.fits is True
        assert verdict.predicted_mb is None

    def test_fits_when_free_covers_predicted_plus_reserve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Free VRAM at or above predicted + reserve fits."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 4000.0)
        budget = VramBudget(reserve_mb=2000.0)
        job = make_job_pop_response("stable_diffusion")
        assert budget.check_job(job, "x", free_vram_mb=6000.0).fits is True
        assert budget.check_job(job, "x", free_vram_mb=5999.0).fits is False

    def test_set_reserve_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Updating the reserve changes the verdict immediately (live config reload)."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 4000.0)
        budget = VramBudget(reserve_mb=2000.0)
        job = make_job_pop_response("stable_diffusion")
        assert budget.check_job(job, "x", free_vram_mb=5000.0).fits is False
        budget.set_reserve_mb(1000.0)
        assert budget.check_job(job, "x", free_vram_mb=5000.0).fits is True

    def test_ram_budget_fits_logic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RamBudget admits when available RAM covers predicted RAM plus reserve, else defers."""
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 6000.0)
        budget = RamBudget(reserve_mb=4096.0)
        job = make_job_pop_response("stable_diffusion")
        assert budget.check_job(job, "x", available_ram_mb=11000.0).fits is True
        assert budget.check_job(job, "x", available_ram_mb=9000.0).fits is False
        assert budget.check_job(job, "x", available_ram_mb=None).fits is True

    def test_verdict_reason_strings(self) -> None:
        """The verdict reason renders the relevant branch for logging."""
        assert (
            "cold start" in BudgetVerdict(fits=True, predicted_mb=None, available_mb=None, reserve_mb=2048.0).reason()
        )
        assert (
            "no burden estimate"
            in BudgetVerdict(fits=True, predicted_mb=None, available_mb=1000.0, reserve_mb=2048.0).reason()
        )
        assert (
            "does NOT fit"
            in BudgetVerdict(fits=False, predicted_mb=4000.0, available_mb=1000.0, reserve_mb=2048.0).reason()
        )
        assert "fits" in BudgetVerdict(fits=True, predicted_mb=1000.0, available_mb=8000.0, reserve_mb=2048.0).reason()


class TestCommittedReserveLedger:
    """The shared ledger accounts for every flow's in-flight cost as one combined figure."""

    def test_set_and_total(self) -> None:
        """Entries from different flows sum into one combined VRAM/RAM total."""
        ledger = CommittedReserveLedger()
        ledger.set("image_post_processing", "aggregate", vram_mb=1000.0)
        ledger.set("alchemy", "form-1", vram_mb=400.0, ram_mb=200.0)
        assert ledger.total_vram_mb() == 1400.0
        assert ledger.total_ram_mb() == 200.0

    def test_release_is_idempotent(self) -> None:
        """Releasing a unit drops its reserve; releasing again is harmless."""
        ledger = CommittedReserveLedger()
        ledger.set("alchemy", "form-1", vram_mb=400.0)
        ledger.release("alchemy", "form-1")
        ledger.release("alchemy", "form-1")
        assert ledger.total_vram_mb() == 0.0

    def test_replace_flow_drops_stale_units(self) -> None:
        """Replacing a flow's entries drops units no longer present (self-healing on lost results)."""
        ledger = CommittedReserveLedger()
        ledger.set("image_post_processing", "aggregate", vram_mb=1000.0)
        ledger.replace_flow("alchemy", vram_mb_by_unit={"form-1": 300.0, "form-2": 300.0})
        assert ledger.total_vram_mb() == 1600.0
        # A later reconcile where form-1's process died leaves only form-2 under alchemy.
        ledger.replace_flow("alchemy", vram_mb_by_unit={"form-2": 300.0})
        assert ledger.total_vram_mb() == 1300.0
        # The image flow's entry is untouched by an alchemy replace.
        ledger.replace_flow("alchemy", vram_mb_by_unit={})
        assert ledger.total_vram_mb() == 1000.0

    def test_negative_costs_floored_to_zero(self) -> None:
        """A noisy negative estimate cannot credit headroom back into the total."""
        ledger = CommittedReserveLedger()
        ledger.set("alchemy", "form-1", vram_mb=-500.0)
        assert ledger.total_vram_mb() == 0.0

    def test_planned_counts_full_until_target_reserves(self) -> None:
        """A planned charge counts in full until its target's measured reservation grows past admit time."""
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "job-1", vram_mb=5000.0, target_process_id=3, reserved_at_admit_mb=200.0)
        # Target still at its admit-time reservation: nothing materialised, full planned charge stands.
        assert ledger.effective_planned_vram_mb({3: 200.0}) == 5000.0
        # Target absent from the snapshot is treated as holding zero, so nothing has materialised yet either.
        assert ledger.effective_planned_vram_mb({}) == 5000.0

    def test_planned_decays_as_target_reservation_materialises(self) -> None:
        """The planned charge decays one-for-one as the target's measured reservation fills it in."""
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "job-1", vram_mb=5000.0, target_process_id=3, reserved_at_admit_mb=200.0)
        # Grew 2000 beyond admit: 3000 of the planned charge remains outstanding.
        assert ledger.effective_planned_vram_mb({3: 2200.0}) == 3000.0
        # Fully materialised (grew by the whole planned amount): nothing left to double-charge.
        assert ledger.effective_planned_vram_mb({3: 5200.0}) == 0.0
        assert ledger.effective_planned_vram_mb({3: 9999.0}) == 0.0

    def test_release_drops_planned_charge(self) -> None:
        """Releasing a unit self-heals its planned charge as well as its flat reserve."""
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "job-1", vram_mb=5000.0, target_process_id=3, reserved_at_admit_mb=0.0)
        ledger.release("preload", "job-1")
        assert ledger.effective_planned_vram_mb({}) == 0.0

    def test_planned_is_disjoint_from_flat_total(self) -> None:
        """Planned charges live in their own overlay; the flat committed total is unchanged by them."""
        ledger = CommittedReserveLedger()
        ledger.set("alchemy", "form-1", vram_mb=400.0)
        ledger.set_planned("preload", "job-1", vram_mb=5000.0, target_process_id=3, reserved_at_admit_mb=0.0)
        assert ledger.total_vram_mb() == 400.0
        assert ledger.effective_planned_vram_mb({}) == 5000.0

    def test_reconcile_planned_drops_omitted_units_with_no_release_call(self) -> None:
        """A planned unit absent from the live set is dropped by reconcile alone (omission is the release)."""
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "0", vram_mb=5000.0, target_process_id=0, reserved_at_admit_mb=0.0)
        ledger.set_planned("preload", "1", vram_mb=3000.0, target_process_id=1, reserved_at_admit_mb=0.0)
        # Process 1's admission finished/faulted/died: it simply stops appearing in the live set.
        ledger.reconcile_planned("preload", ["0"])
        assert ledger.effective_planned_vram_mb({}) == 5000.0

    def test_reconcile_planned_preserves_admit_time_decay_baseline(self) -> None:
        """A surviving unit keeps its admit-time reservation baseline, so decay is not reset by the reconcile."""
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "0", vram_mb=5000.0, target_process_id=0, reserved_at_admit_mb=200.0)
        # The target has begun materialising (grew 2000 past admit) by the time the reconcile runs.
        ledger.reconcile_planned("preload", ["0"])
        # Decay is still measured against the admit-time 200, not re-baselined to the current 2200.
        assert ledger.effective_planned_vram_mb({0: 2200.0}) == 3000.0

    def test_reconcile_planned_leaves_other_flows_untouched(self) -> None:
        """Reconciling one flow's planned charges never disturbs another flow's planned entries."""
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "0", vram_mb=5000.0, target_process_id=0, reserved_at_admit_mb=0.0)
        ledger.set_planned("other", "x", vram_mb=1000.0, target_process_id=9, reserved_at_admit_mb=0.0)
        ledger.reconcile_planned("preload", [])
        # The preload flow is emptied; the unrelated flow's planned charge survives.
        assert ledger.effective_planned_vram_mb({}) == 1000.0

    def test_reconcile_planned_is_idempotent(self) -> None:
        """Re-running the reconcile with the same live set is a no-op (safe to drive several times a cycle)."""
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "0", vram_mb=5000.0, target_process_id=0, reserved_at_admit_mb=0.0)
        ledger.reconcile_planned("preload", ["0"])
        ledger.reconcile_planned("preload", ["0"])
        assert ledger.effective_planned_vram_mb({}) == 5000.0

    def test_materialised_then_evicted_anchor_stays_consumed(self) -> None:
        """Once a charge has fully materialised, a later reservation collapse cannot resurrect it."""
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "3", vram_mb=6158.0, target_process_id=3, reserved_at_admit_mb=0.0)
        # Materialise: the target's reservation grows past the whole planned charge.
        assert ledger.effective_planned_vram_mb({3: 6158.0}) == 0.0
        # Evict: the reservation collapses back toward zero. The charge stays consumed (watermark holds).
        assert ledger.effective_planned_vram_mb({3: 68.0}) == 0.0
        assert ledger.effective_planned_vram_mb({}) == 0.0

    def test_same_cycle_admits_with_no_growth_both_count_full(self) -> None:
        """Two anchors admitted the same cycle with no reservation growth both count in full (double-admit guard)."""
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "0", vram_mb=6158.0, target_process_id=0, reserved_at_admit_mb=0.0)
        ledger.set_planned("preload", "1", vram_mb=6134.0, target_process_id=1, reserved_at_admit_mb=0.0)
        assert ledger.effective_planned_vram_mb({0: 0.0, 1: 0.0}) == 6158.0 + 6134.0

    def test_partial_materialisation_watermark_does_not_resurrect(self) -> None:
        """A partly-materialised anchor holds its outstanding share and does not resurrect above it after a drop."""
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "3", vram_mb=6158.0, target_process_id=3, reserved_at_admit_mb=0.0)
        # Grew 3000 of the 6158 charge: 3158 outstanding.
        assert ledger.effective_planned_vram_mb({3: 3000.0}) == pytest.approx(3158.0)
        # Reservation collapses: outstanding stays at the watermarked 3158, it does not climb back to 6158.
        assert ledger.effective_planned_vram_mb({3: 0.0}) == pytest.approx(3158.0)

    def test_re_registering_a_unit_resets_the_watermark(self) -> None:
        """A genuinely new admission on the same unit charges in full again (a fresh entry, fresh watermark)."""
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "3", vram_mb=6158.0, target_process_id=3, reserved_at_admit_mb=0.0)
        assert ledger.effective_planned_vram_mb({3: 6158.0}) == 0.0
        # A new preload is admitted onto the same process; its charge must count in full from its own baseline.
        ledger.set_planned("preload", "3", vram_mb=6158.0, target_process_id=3, reserved_at_admit_mb=6158.0)
        assert ledger.effective_planned_vram_mb({3: 6158.0}) == 6158.0


class TestAdmissionNoiseBuffer:
    """The proportional noise buffer: the greater of a fixed floor and a fraction of the device total."""

    @pytest.mark.parametrize(
        ("total_mb", "expected_mb"),
        [
            (0.0, 512.0),
            (8192.0, 512.0),
            (10240.0, 512.0),
            (16375.0, 818.75),
            (24576.0, 1228.8),
        ],
    )
    def test_floor_below_threshold_and_proportional_above(self, total_mb: float, expected_mb: float) -> None:
        """At or below the floor's break-even total the buffer is 512; above it scales to 5% of the total."""
        assert admission_noise_buffer_mb(total_mb) == pytest.approx(expected_mb)

    def test_unknown_total_takes_the_floor(self) -> None:
        """A cold-start unknown total yields the floor, since no capacity exists to scale against."""
        assert admission_noise_buffer_mb(None) == pytest.approx(_ADMISSION_NOISE_BUFFER_MB)


class TestAdmissionIdentity:
    """The ledger-driven admission inequality: measured floor + planned + candidate vs real capacity."""

    def test_derived_default_buffer_denies_what_the_floor_would_admit_on_a_large_card(self) -> None:
        """With no explicit buffer a 16375MB card derives 818.75, denying a demand that fits under 512."""
        total, baseline = 16375.0, 1700.0
        # Demand lands between the floor ceiling and the derived ceiling, so the derived default flips it.
        floor_capacity = (total - baseline) - _ADMISSION_NOISE_BUFFER_MB
        committed = floor_capacity - 100.0
        derived = evaluate_admission(
            measured_committed_mb=committed,
            planned_unmaterialized_mb=0.0,
            candidate_delta_mb=0.0,
            total_vram_mb=total,
            baseline_mb=baseline,
            committed_is_stale=False,
        )
        assert derived.noise_buffer_mb == pytest.approx(admission_noise_buffer_mb(total))
        assert derived.fits is False
        # The same demand fits when the floor buffer is passed explicitly (the explicit value wins).
        explicit = evaluate_admission(
            measured_committed_mb=committed,
            planned_unmaterialized_mb=0.0,
            candidate_delta_mb=0.0,
            total_vram_mb=total,
            baseline_mb=baseline,
            noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
            committed_is_stale=False,
        )
        assert explicit.noise_buffer_mb == pytest.approx(_ADMISSION_NOISE_BUFFER_MB)
        assert explicit.fits is True

    @pytest.mark.parametrize("total_mb", [8192.0, 16375.0, 24564.0])
    def test_inequality_admits_within_capacity_and_denies_past_it(self, total_mb: float) -> None:
        """Across 8/16/24GB cards, demand within capacity admits and a candidate past it denies."""
        baseline = 1700.0
        capacity = (total_mb - baseline) - _ADMISSION_NOISE_BUFFER_MB
        committed = capacity - 1000.0
        fits = evaluate_admission(
            measured_committed_mb=committed,
            planned_unmaterialized_mb=0.0,
            candidate_delta_mb=500.0,
            total_vram_mb=total_mb,
            baseline_mb=baseline,
            noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
            committed_is_stale=False,
        )
        assert fits.used_measured_floor is True
        assert fits.fits is True
        denies = evaluate_admission(
            measured_committed_mb=committed,
            planned_unmaterialized_mb=0.0,
            candidate_delta_mb=1001.0,
            total_vram_mb=total_mb,
            baseline_mb=baseline,
            noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
            committed_is_stale=False,
        )
        assert denies.fits is False

    def test_resident_credit_admits_what_the_gross_candidate_would_deny(self) -> None:
        """Netting the resident weights out of the candidate delta admits a load the gross figure denies."""
        total, baseline = 16375.0, 1700.0
        capacity = (total - baseline) - _ADMISSION_NOISE_BUFFER_MB
        committed = capacity - 4000.0
        gross_candidate = 6000.0
        resident_credit = 4900.0
        assert (
            evaluate_admission(
                measured_committed_mb=committed,
                planned_unmaterialized_mb=0.0,
                candidate_delta_mb=gross_candidate,
                total_vram_mb=total,
                baseline_mb=baseline,
                noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
                committed_is_stale=False,
            ).fits
            is False
        )
        assert (
            evaluate_admission(
                measured_committed_mb=committed,
                planned_unmaterialized_mb=0.0,
                candidate_delta_mb=gross_candidate - resident_credit,
                total_vram_mb=total,
                baseline_mb=baseline,
                noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
                committed_is_stale=False,
            ).fits
            is True
        )

    def test_planned_charge_counts_toward_demand(self) -> None:
        """A planned (not-yet-materialised) charge can flip an otherwise-fitting admission to a denial."""
        total, baseline = 16375.0, 1700.0
        capacity = (total - baseline) - _ADMISSION_NOISE_BUFFER_MB
        verdict = evaluate_admission(
            measured_committed_mb=capacity - 500.0,
            planned_unmaterialized_mb=1000.0,
            candidate_delta_mb=0.0,
            total_vram_mb=total,
            baseline_mb=baseline,
            noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
            committed_is_stale=False,
        )
        assert verdict.fits is False

    def test_materialised_then_evicted_anchor_does_not_zombie_deny_retention(self) -> None:
        """The observed churn loop: after materialise+evict a retention candidate fits, no zombie planned denial.

        Reproduces a 16GB card whose two preloads materialised (~6158 + ~6134) and were then evicted, collapsing
        the committed floor to ~68MB. Before monotonic consumption the anchors resurrected to ~12248MB and the
        identity denied retention of the just-used ~6158MB model against a ~14436MB capacity. With the watermark
        the anchors stay consumed, so the candidate fits.
        """
        total, baseline = 16375.0, 1700.0
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "0", vram_mb=6158.0, target_process_id=0, reserved_at_admit_mb=0.0)
        ledger.set_planned("preload", "1", vram_mb=6134.0, target_process_id=1, reserved_at_admit_mb=0.0)
        # Both loads materialise into their targets' reservations.
        ledger.effective_planned_vram_mb({0: 6158.0, 1: 6134.0})
        # Eviction collapses both reservations back toward zero.
        planned_after_evict = ledger.effective_planned_vram_mb({0: 68.0, 1: 68.0})
        assert planned_after_evict == 0.0
        verdict = evaluate_admission(
            measured_committed_mb=68.0,
            planned_unmaterialized_mb=planned_after_evict,
            candidate_delta_mb=6158.0,
            total_vram_mb=total,
            baseline_mb=baseline,
            committed_is_stale=False,
        )
        assert verdict.used_measured_floor is True
        assert verdict.fits is True

    def test_staleness_alone_never_denies(self) -> None:
        """Staleness drops the measured floor: with no planned demand a fitting candidate always admits.

        However large the stale committed floor reads, it is dropped, so it can never itself flip the verdict;
        only the parent's planned overlay (empty here) plus the candidate are tested against capacity.
        """
        verdict = evaluate_admission(
            measured_committed_mb=99999.0,
            planned_unmaterialized_mb=0.0,
            candidate_delta_mb=6158.0,
            total_vram_mb=16375.0,
            baseline_mb=1700.0,
            committed_is_stale=True,
        )
        assert verdict.fits is True
        assert verdict.used_measured_floor is False

    def test_stale_planned_overlay_denies_stacked_admissions(self) -> None:
        """The startup storm: stale reports but stacked planned admissions still deny a further over-commit.

        The measured floor is dropped, yet the parent's own planned overlay (12316 MB of already-admitted,
        not-yet-materialised preloads) plus a fresh 6158 MB candidate exceed a ~14781 MB capacity, so the
        degraded identity denies even before the first child memory report lands.
        """
        verdict = evaluate_admission(
            measured_committed_mb=99999.0,
            planned_unmaterialized_mb=12316.0,
            candidate_delta_mb=6158.0,
            total_vram_mb=16375.0,
            baseline_mb=1700.0,
            noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
            committed_is_stale=True,
        )
        assert verdict.used_measured_floor is False
        assert verdict.fits is False
        assert verdict.capacity_mb == pytest.approx((16375.0 - 1700.0) - _ADMISSION_NOISE_BUFFER_MB)
        # The demand drops the dropped measured floor and tests only planned + candidate.
        assert verdict.demand_mb == pytest.approx(12316.0 + 6158.0)

    def test_stale_with_unknown_total_relaxes_regardless(self) -> None:
        """With no known total nothing is knowable, so a stale ledger relaxes fully whatever the planned demand."""
        verdict = evaluate_admission(
            measured_committed_mb=99999.0,
            planned_unmaterialized_mb=12316.0,
            candidate_delta_mb=6158.0,
            total_vram_mb=None,
            baseline_mb=1700.0,
            committed_is_stale=True,
        )
        assert verdict.fits is True
        assert verdict.used_measured_floor is False

    def test_startup_storm_planned_overlay_denies_third_stacked_preload(self) -> None:
        """Cold start, stale child reports: the planned overlay admits the first two preloads and denies the third.

        No child has reported yet (committed stale, baseline still 0), so the measured floor is dropped. Each
        admitted preload registers a planned anchor via the real ledger; against a 16375 MB card the third
        stacked 6158 MB candidate pushes planned + candidate past capacity and the degraded identity denies it,
        the arithmetic the observed three-process over-commit needed and the fixed relax used to skip.
        """
        total, baseline = 16375.0, 0.0
        candidate = 6158.0
        ledger = CommittedReserveLedger()
        outcomes: list[bool] = []
        for index in range(3):
            planned = ledger.effective_planned_vram_mb({})
            verdict = evaluate_admission(
                measured_committed_mb=0.0,
                planned_unmaterialized_mb=planned,
                candidate_delta_mb=candidate,
                total_vram_mb=total,
                baseline_mb=baseline,
                noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
                committed_is_stale=True,
            )
            outcomes.append(verdict.fits)
            if verdict.fits:
                ledger.set_planned(
                    "preload",
                    str(index),
                    vram_mb=candidate,
                    target_process_id=index,
                    reserved_at_admit_mb=0.0,
                )
        assert outcomes == [True, True, False]

    def test_cold_start_relaxes_to_predictive_path(self) -> None:
        """An unknown total (cold start) also relaxes to the predictive path, never wedging."""
        verdict = evaluate_admission(
            measured_committed_mb=1000.0,
            planned_unmaterialized_mb=0.0,
            candidate_delta_mb=1000.0,
            total_vram_mb=None,
            baseline_mb=1700.0,
            committed_is_stale=False,
        )
        assert verdict.fits is True
        assert verdict.used_measured_floor is False

    def test_two_samplers_materialized_admit_on_16gb(self) -> None:
        """Correction 1 worked example: two materialised samplers + contexts + idle VAE lane MUST admit."""
        total, baseline = 16375.0, 1700.0
        committed = 12700.0  # two ~4900MB samplers + their contexts + 6 idle contexts + idle VAE lane
        verdict = evaluate_admission(
            measured_committed_mb=committed,
            planned_unmaterialized_mb=0.0,
            candidate_delta_mb=0.0,
            total_vram_mb=total,
            baseline_mb=baseline,
            committed_is_stale=False,
        )
        assert verdict.used_measured_floor is True
        assert verdict.fits is True
        # And the plan's denial case: a candidate that would push demand past the ceiling.
        assert verdict.capacity_mb is not None
        over = evaluate_admission(
            measured_committed_mb=committed,
            planned_unmaterialized_mb=0.0,
            candidate_delta_mb=verdict.capacity_mb - committed + 1.0,
            total_vram_mb=total,
            baseline_mb=baseline,
            committed_is_stale=False,
        )
        assert over.fits is False

    def test_oversized_candidate_denies_so_caller_defers(self) -> None:
        """A wildly oversized (mis-estimated) candidate denies rather than faulting: the caller defers."""
        verdict = evaluate_admission(
            measured_committed_mb=2000.0,
            planned_unmaterialized_mb=0.0,
            candidate_delta_mb=20000.0,
            total_vram_mb=16375.0,
            baseline_mb=1700.0,
            committed_is_stale=False,
        )
        assert verdict.used_measured_floor is True
        assert verdict.fits is False

    def test_reason_renders_full_identity(self) -> None:
        """The verdict reason renders every term of the identity for a self-explaining log line."""
        verdict = evaluate_admission(
            measured_committed_mb=1000.0,
            planned_unmaterialized_mb=100.0,
            candidate_delta_mb=200.0,
            total_vram_mb=16375.0,
            baseline_mb=1700.0,
            committed_is_stale=False,
        )
        rendered = verdict.reason()
        assert "committed" in rendered
        assert "planned" in rendered
        assert "candidate" in rendered
        assert "capacity" in rendered
        # A cold-start relaxed verdict explains why the measured floor was not applied.
        assert (
            "predictive path"
            in evaluate_admission(
                measured_committed_mb=1000.0,
                planned_unmaterialized_mb=0.0,
                candidate_delta_mb=0.0,
                total_vram_mb=None,
                baseline_mb=1700.0,
                committed_is_stale=False,
            ).reason()
        )

    def test_degraded_stale_reason_renders_dropped_floor_and_verdict(self) -> None:
        """A stale-with-total verdict renders the dropped floor and the planned-only inequality honestly."""
        rendered = evaluate_admission(
            measured_committed_mb=99999.0,
            planned_unmaterialized_mb=12316.0,
            candidate_delta_mb=6158.0,
            total_vram_mb=16375.0,
            baseline_mb=1700.0,
            noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
            committed_is_stale=True,
        ).reason()
        assert "stale" in rendered
        assert "measured floor dropped" in rendered
        # The stale committed figure must not be presented as a live term.
        assert "99999" not in rendered
        assert "planned" in rendered
        assert "candidate" in rendered
        assert "does NOT fit" in rendered


class TestRamBudgetCommittedReserve:
    """RamBudget subtracts the committed reserve symmetrically with VramBudget."""

    def test_committed_reserve_can_flip_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RAM already committed by in-flight work is held back from the admission decision."""
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 6000.0)
        budget = RamBudget(reserve_mb=4096.0)
        job = make_job_pop_response("stable_diffusion")
        # 11000 - 0 >= 6000 + 4096 -> fits; 11000 - 2000 < 10096 -> defers.
        assert budget.check_job(job, "x", available_ram_mb=11000.0, committed_reserve_mb=0.0).fits is True
        assert budget.check_job(job, "x", available_ram_mb=11000.0, committed_reserve_mb=2000.0).fits is False


def _budget_bridge_data() -> Mock:
    """Mock bridge data with the VRAM budget enabled and real numeric reserves."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2000,
        ram_reserve_mb=4096,
        image_models_to_load=["model_a", "model_b"],
    )


class TestPreloadBudgetGate:
    """Integration tests for the preload-time VRAM budget gate inside the scheduler."""

    async def test_preload_deferred_and_reclaims_when_over_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the next model will not fit, preload is deferred and idle VRAM is reclaimed."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 8000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        # A second, idle process holding a different resident model: the eviction candidate. Its committed
        # allocator reservation over-commits the card, so the arbiter's measured floor denies the incoming
        # head and describes an eviction of this idle resident.
        resident = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.WAITING_FOR_JOB)
        resident.total_vram_mb = 16000
        resident.vram_usage_mb = 15000  # 1000 MB free, well under 8000 + 2000
        # A fresh committed report whose allocator reservation over-commits the card: the measured floor
        # applies (not stale) and denies the incoming head, describing an eviction of this idle resident.
        resident.process_reserved_mb = 16000
        resident.report_sampled_at = time.time()
        process_map = ProcessMap({0: spare, 1: resident})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )
        # Prevent the real psutil RAM reading from spuriously tripping the RAM danger floor gate
        # when system available memory is low (common in large combined test runs).
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 8000.0)
        # Pin total RAM too so the absolute danger floor (a percentage of total) is host-independent: 8000 MB
        # available clears 15% of 32000 MB, keeping these marginal-budget cases out of the danger-floor path.
        monkeypatch.setattr(scheduler, "_measured_total_ram_mb", lambda: 32000.0)

        assert scheduler.preload_models() is False
        # The spare process was NOT told to preload...
        assert spare.last_control_flag != HordeControlFlag.PRELOAD_MODEL
        # ...and the idle resident model was evicted to reclaim VRAM (residency overridden under pressure).
        assert resident.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM

    async def test_ram_reclaim_progress_keeps_head_deferred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RAM reclaim progress keeps a VRAM-fitting head deferred until the reclaimed memory is visible."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 1000.0)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 50000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 1000  # ample free VRAM, so only the RAM gate defers
        process_map = ProcessMap({0: spare})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 8000.0)
        # Pin total RAM too so the absolute danger floor (a percentage of total) is host-independent: 8000 MB
        # available clears 15% of 32000 MB, keeping these marginal-budget cases out of the danger-floor path.
        monkeypatch.setattr(scheduler, "_measured_total_ram_mb", lambda: 32000.0)
        # Reclaim reports progress, so the RAM branch waits for the next measured tick even if the idle-device
        # head has been parked for a long time.
        monkeypatch.setattr(scheduler, "unload_models", lambda *a, **k: True)

        assert scheduler.preload_models() is False
        assert spare.last_control_flag != HordeControlFlag.PRELOAD_MODEL
        assert scheduler._head_starvation_job_id == str(job.id_)

        scheduler._head_starvation_since -= 120.0
        assert scheduler.preload_models() is False
        assert spare.last_control_flag != HordeControlFlag.PRELOAD_MODEL
        assert job_tracker._tracked_for(job).admitted_over_budget is False

    async def test_starved_head_clock_resets_when_live_job_holds_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The starvation clock must not run while a live job holds the device (head is merely queued).

        Otherwise the backstop would force a second concurrent heavy load and reintroduce the very
        over-commit the budget guards against.
        """
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 1000.0)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 50000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 1000
        busy = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({0: spare, 1: busy})

        job_tracker = JobTracker()
        live = make_job_pop_response("model_b")
        await track_popped_job_async(job_tracker, live)
        await job_tracker.mark_inference_started(live)
        head = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, head)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 8000.0)
        # Pin total RAM too so the absolute danger floor (a percentage of total) is host-independent: 8000 MB
        # available clears 15% of 32000 MB, keeping these marginal-budget cases out of the danger-floor path.
        monkeypatch.setattr(scheduler, "_measured_total_ram_mb", lambda: 32000.0)

        scheduler._update_head_starvation_timer(head)
        # A live job holds the device, so the head's clock must not be running.
        assert scheduler._head_starvation_job_id is None
        assert scheduler._head_starved_seconds(head) == 0.0

    async def test_preload_proceeds_when_within_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ample free VRAM and RAM the budget admits and the preload is sent."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 4000.0)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 1000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 2000  # 14000 MB free, covers 4000 + 2000
        process_map = ProcessMap({0: spare})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )
        # Prevent the real psutil RAM reading from spuriously tripping the RAM danger floor gate
        # when system available memory is low (common in large combined test runs).
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 8000.0)
        # Pin total RAM too so the absolute danger floor (a percentage of total) is host-independent: 8000 MB
        # available clears 15% of 32000 MB, keeping these marginal-budget cases out of the danger-floor path.
        monkeypatch.setattr(scheduler, "_measured_total_ram_mb", lambda: 32000.0)

        assert scheduler.preload_models() is True
        assert spare.last_control_flag == HordeControlFlag.PRELOAD_MODEL

    async def test_preload_deferred_when_over_ram_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """VRAM fits but RAM does not: the preload is deferred and idle RAM is reclaimed."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 1000.0)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 50000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 1000  # ample free VRAM, so the VRAM gate passes
        # A second idle process holding a resident model: the RAM eviction candidate.
        resident = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: spare, 1: resident})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name="model_b",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=1,
        )

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )
        # Force a low available-RAM reading so the RAM budget defers deterministically.
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 8000.0)
        # Pin total RAM too so the absolute danger floor (a percentage of total) is host-independent: 8000 MB
        # available clears 15% of 32000 MB, keeping these marginal-budget cases out of the danger-floor path.
        monkeypatch.setattr(scheduler, "_measured_total_ram_mb", lambda: 32000.0)

        assert scheduler.preload_models() is False
        assert spare.last_control_flag != HordeControlFlag.PRELOAD_MODEL
        assert resident.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM

    async def test_disabled_budget_ignores_low_vram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the budget disabled, a low-VRAM device does not defer the preload."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 8000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 15500  # only 500 MB free
        process_map = ProcessMap({0: spare})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(enable_vram_budget=False, image_models_to_load=["model_a"]),
            max_concurrent=2,
            max_inference=2,
        )

        assert scheduler.preload_models() is True
        assert spare.last_control_flag == HordeControlFlag.PRELOAD_MODEL


class TestCheckJobCommittedReserve:
    """The VRAM budget holds back a committed reserve (e.g. in-flight post-processing) before admitting."""

    def test_committed_reserve_subtracts_from_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job that fits the raw free VRAM is deferred once the committed reserve is held back."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 4000.0)
        budget = VramBudget(reserve_mb=2000.0)
        job = make_job_pop_response("stable_diffusion")
        # 6000 covers 4000 + 2000 with nothing committed...
        assert budget.check_job(job, "x", free_vram_mb=6000.0).fits is True
        # ...but holding back 1500 MB of in-flight post-processing drops effective free to 4500 < 6000.
        verdict = budget.check_job(job, "x", free_vram_mb=6000.0, committed_reserve_mb=1500.0)
        assert verdict.fits is False
        assert verdict.available_mb == 4500.0

    def test_committed_reserve_defaults_to_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Callers (and existing tests) that omit the reserve keep the prior instantaneous behavior."""
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 4000.0)
        budget = VramBudget(reserve_mb=2000.0)
        job = make_job_pop_response("stable_diffusion")
        assert budget.check_job(job, "x", free_vram_mb=6000.0).fits is True


class TestPredictPostProcessingPeak:
    """The post-processing-phase predictor and its graceful fallback on an older hordelib."""

    def test_reads_phase_split_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The predictor returns the burden's post-processing-phase VRAM figure."""
        burden = Mock(vram_post_processing_mb=1500)
        monkeypatch.setattr(resource_budget, "_estimate_job_burden", lambda job, baseline: burden)
        job = make_job_pop_response("x")
        assert resource_budget.predict_job_post_processing_vram_mb(job, "stable_diffusion_xl") == 1500.0

    def test_none_when_estimate_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No burden estimate means the post-processing peak is unknown (None), not zero."""
        monkeypatch.setattr(resource_budget, "_estimate_job_burden", lambda job, baseline: None)
        job = make_job_pop_response("x")
        assert resource_budget.predict_job_post_processing_vram_mb(job, "x") is None

    def test_none_when_field_absent_on_old_hordelib(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pinned hordelib predating the phase-split lacks the field; the predictor degrades to None."""

        class _OldBurden:
            vram_mb = 5000
            ram_mb = 8000
            # No vram_post_processing_mb attribute (older BurdenEstimate).

        monkeypatch.setattr(resource_budget, "_estimate_job_burden", lambda job, baseline: _OldBurden())
        job = make_job_pop_response("x")
        assert resource_budget.predict_job_post_processing_vram_mb(job, "stable_diffusion_xl") is None


def _post_processing_process(process_id: int, job: object) -> object:
    """A mock inference process in the post-processing phase, holding ``job`` as its referenced job."""
    proc = make_mock_process_info(
        process_id,
        model_name="model_pp",
        state=HordeProcessState.INFERENCE_POST_PROCESSING,
    )
    proc.last_job_referenced = job  # pyrefly: ignore - assigning the tracked job for the reserve lookup
    return proc


class TestUpscaleFactorWiring:
    """The job's upscaler scale factor is resolved and inflates the predicted post-processing peak."""

    def test_factor_resolved_from_job_post_processing(self) -> None:
        """The max upscaler factor is read from the job payload; facefixers and an empty list contribute 1."""
        assert resource_budget._job_upscale_factor(make_mock_job(post_processing=["RealESRGAN_x2plus"])) == 2.0
        assert (
            resource_budget._job_upscale_factor(make_mock_job(post_processing=["RealESRGAN_x4plus", "GFPGAN"])) == 4.0
        )
        assert resource_budget._job_upscale_factor(make_mock_job(post_processing=[])) == 1.0

    def test_post_processing_peak_grows_with_factor(self) -> None:
        """End-to-end through the real hordelib: a 4x upscale reserves more than a 2x, both above zero."""
        job4 = make_mock_job(width=1024, height=1024, post_processing=["RealESRGAN_x4plus"])
        job2 = make_mock_job(width=1024, height=1024, post_processing=["RealESRGAN_x2plus"])
        peak4 = resource_budget.predict_job_post_processing_vram_mb(job4, "stable_diffusion_xl")
        peak2 = resource_budget.predict_job_post_processing_vram_mb(job2, "stable_diffusion_xl")
        assert peak4 is not None and peak2 is not None
        assert peak4 > peak2 > 0


class TestUpscaleDoesNotDriveResidency:
    """Regression: a post-processing upscaler must not flip an ordinary SDXL job into whole-card residency.

    A 4x upscaler's output-scaled activation belongs to the post-processing phase, which runs *after*
    sampling on the already-resident model. Folding it into the weight-residency forecast (as the old
    combined peak did) made a ~4.9GB SDXL job that merely requested an upscaler read as
    weight-dominant/needs-exclusive on a 16GB card; with a single inference process and no idle sibling to
    tear down, the head wedged until a save-our-ship soft reset. The residency forecast and the preload gate
    must key on the sampling-phase peak instead.
    """

    def test_sampling_peak_excludes_post_processing_activation(self) -> None:
        """Adding a 4x upscaler leaves the sampling peak unchanged; only the post-processing peak grows."""
        plain = make_mock_job(width=1024, height=1024, post_processing=[])
        upscaled = make_mock_job(width=1024, height=1024, post_processing=["RealESRGAN_x4plus"])
        sampling_plain = resource_budget.predict_job_sampling_vram_mb(plain, "stable_diffusion_xl")
        sampling_upscaled = resource_budget.predict_job_sampling_vram_mb(upscaled, "stable_diffusion_xl")
        assert sampling_plain is not None and sampling_upscaled is not None
        assert sampling_upscaled == sampling_plain
        # The upscaler's cost is real, it just lands in the post-processing phase rather than the sampling one.
        assert (resource_budget.predict_job_post_processing_vram_mb(upscaled, "stable_diffusion_xl") or 0) > 0

    def test_forecast_uses_sampling_not_combined_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The residency forecast keys on the sampling peak; an inflated combined peak does not flip it."""
        job = make_mock_job(width=1024, height=1024, post_processing=["RealESRGAN_x4plus"])
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda j, b: 4900.0)
        # Sampling-phase peak (weights + a modest sampling activation) that comfortably co-resides.
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda j, b: 6948.0)
        # Tripwire: the bridge.log combined peak (~17GB) would read as whole-card. If the forecast ever reverts
        # to the combined predictor this stub makes the assertions below fail.
        monkeypatch.setattr(resource_budget, "predict_job_vram_mb", lambda j, b: 17023.0)
        forecast = resource_budget.forecast_weight_streaming(
            job,
            "stable_diffusion_xl",
            free_now_mb=15005.0,
            total_vram_mb=16375.0,
            per_process_overhead_mb=1288.0,
            num_inference_processes=1,
            configured_reserve_floor_mb=2048.0,
        )
        assert forecast.fits_coresident is True
        assert forecast.needs_exclusive_residency is False
        assert forecast.requires_sibling_teardown is False


class TestMarginalProcessOverhead:
    """The forecast sizes free_after_model_evict from per_process_overhead + (contexts-1)*marginal.

    On one device the CUDA runtime is loaded once and shared, so a single fresh process measures the whole
    one-time cost (~4.3GB on a 24GB card) while each additional sibling context costs only a few hundred MB.
    Sizing free_after_model_evict as contexts*per_process_overhead multiplies that one-time cost by the
    process count, manufacturing a multi-GB phantom shortfall that wedges high-VRAM workers.
    """

    def _forecast(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        per_process_overhead_mb: float,
        marginal_process_overhead_mb: float | None,
        num_inference_processes: int = 4,
        weights_mb: float = 4900.0,
        sampling_peak_mb: float = 17128.0,
        free_now_mb: float = 18634.0,
    ) -> resource_budget.StreamForecast:
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda j, b: weights_mb)
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda j, b: sampling_peak_mb)
        monkeypatch.setattr(resource_budget, "effective_inference_reserve_mb", lambda *a, **k: 2000.0)
        return resource_budget.forecast_weight_streaming(
            make_mock_job(width=1024, height=1024),
            "stable_diffusion_xl",
            free_now_mb=free_now_mb,
            total_vram_mb=24074.0,
            per_process_overhead_mb=per_process_overhead_mb,
            num_inference_processes=num_inference_processes,
            configured_reserve_floor_mb=0.0,
            marginal_process_overhead_mb=marginal_process_overhead_mb,
        )

    def test_unmeasured_marginal_seeds_conservative_constant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unmeasured marginal prices additional contexts at the seed, NOT the first-context overhead.

        The seed is charged per *additional* context (contexts - 1); the first/sole context still pays the
        full ``per_process_overhead`` (sizing free_if_alone). The one-time runtime cost is never multiplied by
        the process count, so free_after_model_evict is nowhere near the old ``total - N*overhead``.
        """
        seed = resource_budget._SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB
        forecast = self._forecast(monkeypatch, per_process_overhead_mb=4266.0, marginal_process_overhead_mb=None)
        assert forecast.free_after_model_evict_mb == pytest.approx(24074.0 - 4266.0 - seed * 3)
        # free_if_alone keeps the full single-context overhead regardless of the marginal.
        assert forecast.free_if_alone_mb == pytest.approx(24074.0 - 4266.0)

    def test_measured_marginal_frees_after_model_evict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A small measured marginal sizes free_after_model_evict to reality, not the one-time-cost-times-N."""
        # marginal 391 == (5440 idle residency - 4266 probe) / (4 - 1), the measured-hardware numbers.
        forecast = self._forecast(monkeypatch, per_process_overhead_mb=4266.0, marginal_process_overhead_mb=391.0)
        assert forecast.free_after_model_evict_mb == pytest.approx(24074.0 - 4266.0 - 391.0 * 3)  # 18635
        # free_if_alone still pays the full first-context overhead (the surviving process keeps the runtime).
        assert forecast.free_if_alone_mb == pytest.approx(24074.0 - 4266.0)

    def test_marginal_flips_teardown_to_model_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The over-counted overhead raises the (diagnostic) sibling-teardown flag; the marginal lowers it.

        ``requires_sibling_teardown`` is the diagnostic that still tracks the activation-keyed over-count: the
        over-counted overhead raises it, and a measured marginal restores ``fits_after_model_evict`` and lowers
        it. The *grant* decision (``needs_exclusive_residency``) no longer follows that activation-keyed flag --
        it is decided on the persistent weight footprint, so a moderate 4.9 GB model is never granted sole
        residency regardless of the over-count. This is the safer outcome for the probe-overhead wedge: the
        phantom over-count can no longer drive a teardown demand the scheduler acts on (it acts on
        ``needs_exclusive_residency`` / ``needs_process_count_reduction``, both persistent-keyed).
        """
        # The over-count is charging the full first-context overhead against every context; pass it
        # explicitly as the marginal to reproduce that (an unmeasured marginal now seeds a small constant,
        # so None no longer over-counts).
        over_counted = self._forecast(
            monkeypatch,
            per_process_overhead_mb=4266.0,
            marginal_process_overhead_mb=4266.0,
        )
        assert over_counted.requires_sibling_teardown is True
        assert over_counted.needs_exclusive_residency is False

        with_marginal = self._forecast(monkeypatch, per_process_overhead_mb=4266.0, marginal_process_overhead_mb=391.0)
        assert with_marginal.requires_sibling_teardown is False
        assert with_marginal.needs_exclusive_residency is False
        assert with_marginal.fits_after_model_evict is True

    def test_marginal_lifts_max_resident_processes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cheap marginal lets more contexts co-reside than the old per-context overhead allowed."""
        # A moderate model so the budget is comfortably positive and the count is overhead-bound.
        # Charging the full first-context overhead per context (marginal == per_process) is the over-count;
        # a small measured marginal fits more contexts.
        over_counted = self._forecast(
            monkeypatch,
            per_process_overhead_mb=4266.0,
            marginal_process_overhead_mb=4266.0,
            sampling_peak_mb=6948.0,
        )
        with_marginal = self._forecast(
            monkeypatch,
            per_process_overhead_mb=4266.0,
            marginal_process_overhead_mb=391.0,
            sampling_peak_mb=6948.0,
        )
        assert over_counted.max_resident_processes() is not None
        assert with_marginal.max_resident_processes() is not None
        assert with_marginal.max_resident_processes() > over_counted.max_resident_processes()

    def test_seeded_fallback_chain_probe_then_idle_then_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The forecast prices additional contexts by measured marginal when supplied, else the seed.

        The probe/idle-floor resolution is upstream (``ContextOverheadModel``), so by the time a value
        reaches the forecast it is either a measured marginal (used verbatim) or None (seeded). This pins
        both ends: a supplied marginal wins; None seeds the conservative constant, never per_process.
        """
        seed = resource_budget._SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB
        measured = self._forecast(monkeypatch, per_process_overhead_mb=4266.0, marginal_process_overhead_mb=250.0)
        assert measured.free_after_model_evict_mb == pytest.approx(24074.0 - 4266.0 - 250.0 * 3)
        seeded = self._forecast(monkeypatch, per_process_overhead_mb=4266.0, marginal_process_overhead_mb=None)
        assert seeded.free_after_model_evict_mb == pytest.approx(24074.0 - 4266.0 - seed * 3)
        # The first-context overhead is charged exactly once (free_if_alone), never per additional context.
        assert measured.free_if_alone_mb == pytest.approx(24074.0 - 4266.0)
        assert seeded.free_if_alone_mb == pytest.approx(24074.0 - 4266.0)


class TestWholeCardIntent:
    """A baseline declared whole-card (the EXTRA_LARGE tier) claims sole residency even when its weight seed fits.

    Z-Image regressed here: its conservative ~8GB seed read as comfortably co-resident on a 16GB card, so the
    forecast never gave it the card, it co-resided and thrashed. ``wants_whole_card`` biases the residency
    verdict on the tier's intent rather than the knife-edge weight-vs-free fit.
    """

    def _coresident_forecast(self, *, wants_whole_card: bool) -> resource_budget.StreamForecast:
        # A Z-Image-like load on a 16GB card: 10GB weights comfortably fit co-resident and alone.
        return resource_budget.StreamForecast(
            weights_mb=10000.0,
            reserve_mb=2048.0,
            free_now_mb=13000.0,
            free_if_alone_mb=15000.0,
            free_after_model_evict_mb=14000.0,
            total_vram_mb=16384.0,
            per_process_overhead_mb=1354.0,
            marginal_process_overhead_mb=300.0,
            wants_whole_card=wants_whole_card,
        )

    def test_seed_fits_coresident_without_intent(self) -> None:
        """Baseline check: the same load without the intent flag reads co-resident (the regression behavior)."""
        forecast = self._coresident_forecast(wants_whole_card=False)
        assert forecast.fits_coresident is True
        assert forecast.needs_exclusive_residency is False

    def test_intent_forces_exclusive_residency(self) -> None:
        """Whole-card intent forces an exclusive residency forecast even when the seed fits co-resident.

        This is the Z-Image guard: intent wins over the fitting weight estimate, so the model takes the
        exclusive path (evict sibling models, sample alone) rather than co-residing and thrashing. How many
        idle sibling *contexts* survive that teardown is sized separately and budget-relative by
        ``max_resident_processes`` (see :class:`TestWholeCardResidentProcessCount`); intent governs sole
        *sampling*, not a blanket collapse to one process.
        """
        forecast = self._coresident_forecast(wants_whole_card=True)
        assert forecast.needs_exclusive_residency is True

    def test_intent_never_overrides_unservable(self) -> None:
        """Intent must not force exclusive residency on a model that cannot be served alone (fits_alone gate)."""
        forecast = resource_budget.StreamForecast(
            weights_mb=20000.0,  # overflows even the 16GB card alone
            reserve_mb=2048.0,
            free_now_mb=13000.0,
            free_if_alone_mb=15030.0,
            free_after_model_evict_mb=14000.0,
            total_vram_mb=16384.0,
            per_process_overhead_mb=1354.0,
            marginal_process_overhead_mb=300.0,
            wants_whole_card=True,
        )
        assert forecast.fits_alone is False
        assert forecast.needs_exclusive_residency is False
        assert forecast.streams_unavoidably is True


class TestWholeCardResidentProcessCount:
    """A whole-card model's teardown depth is budget-relative, not a blanket collapse to one process.

    ``wants_whole_card`` governs that the model never *co-samples* (the scheduler's concurrency overlap gate),
    not that every sibling *context* must be torn down. An idle, model-free sibling context costs only its
    (cheap) per-context VRAM, so on a card whose VRAM genuinely holds the weights-plus-reserve alongside one or
    more such contexts, keeping them avoids the teardown-and-respawn churn each time the heavy head cycles. The
    same arithmetic returns sole residency on a card with no such room, so the behaviour stays hardware-relative.
    """

    @staticmethod
    def _flux_fp8_forecast(*, total_vram_mb: float) -> resource_budget.StreamForecast:
        # Flux.1-Schnell fp8 (Compact): the hordelib seed (weights ~11.5GB, load peak 14GB -> ~2.5GB activation
        # working set folded into the reserve). A deliberately pessimistic ~3.4GB marginal (far above a real
        # idle context) so the surviving-context result does not rest on an optimistic overhead.
        return resource_budget.StreamForecast(
            weights_mb=11500.0,
            reserve_mb=2500.0,
            base_reserve_mb=2500.0,
            free_now_mb=total_vram_mb - 4266.0,
            free_if_alone_mb=total_vram_mb - 4266.0,
            free_after_model_evict_mb=total_vram_mb - 4266.0,
            total_vram_mb=total_vram_mb,
            per_process_overhead_mb=4266.0,
            marginal_process_overhead_mb=3431.0,
            wants_whole_card=True,
        )

    def test_high_vram_card_keeps_a_sibling_context(self) -> None:
        """On a 24GB card the weights + reserve leave room for a sibling context, so the target is not one."""
        forecast = self._flux_fp8_forecast(total_vram_mb=24074.0)
        assert forecast.max_resident_processes() == 2

    def test_low_vram_card_collapses_to_sole_residency(self) -> None:
        """On a 16GB card the same fp8 weights leave no room, so sole residency is correct and unchanged."""
        forecast = self._flux_fp8_forecast(total_vram_mb=16384.0)
        assert forecast.max_resident_processes() == 1

    def test_unsizable_whole_card_model_still_collapses_to_one(self) -> None:
        """When the footprint cannot be sized (no total VRAM) a whole-card-intent model stays conservative."""
        forecast = resource_budget.StreamForecast(
            weights_mb=11500.0,
            reserve_mb=2500.0,
            free_now_mb=13000.0,
            free_if_alone_mb=None,
            free_after_model_evict_mb=None,
            total_vram_mb=None,
            per_process_overhead_mb=4266.0,
            wants_whole_card=True,
        )
        assert forecast.max_resident_processes() == 1


class TestRamPressureFloor:
    """The absolute system-RAM danger floor: the more conservative of the percentage and the MB floor."""

    def test_percentage_floor_binds_on_a_large_ram_host(self) -> None:
        """On a 32 GB host the 90%-used (10% free) rule binds: floor ~3.2 GB, well above the 1 GB absolute."""
        floor = ram_pressure_floor_mb(32063.0, pause_percent=90.0, min_free_mb=1024.0)
        assert floor == pytest.approx(3206.3, abs=1.0)

    def test_absolute_floor_binds_on_a_small_ram_host(self) -> None:
        """On an 8 GB host 10% free is only ~819 MB, so the 1 GB absolute floor is the more conservative one."""
        floor = ram_pressure_floor_mb(8192.0, pause_percent=90.0, min_free_mb=1024.0)
        assert floor == pytest.approx(1024.0)

    def test_both_floors_are_configurable(self) -> None:
        """A stricter percentage (or absolute floor) raises the danger threshold accordingly."""
        # 80%-used (20% free) on 32 GB -> ~6.4 GB, above a 2 GB absolute.
        assert ram_pressure_floor_mb(32063.0, pause_percent=80.0, min_free_mb=2048.0) == pytest.approx(6412.6, abs=1.0)
        # A large absolute floor wins even against a lenient percentage.
        assert ram_pressure_floor_mb(32063.0, pause_percent=95.0, min_free_mb=4096.0) == pytest.approx(4096.0)

    def test_unknown_total_falls_back_to_absolute_floor(self) -> None:
        """With total RAM unknown (cold start), only the absolute MB floor applies."""
        assert ram_pressure_floor_mb(None, pause_percent=90.0, min_free_mb=1024.0) == pytest.approx(1024.0)


class TestAssessRamPressure:
    """The pressure verdict: under_pressure exactly when measured available falls below the floor."""

    def test_below_floor_is_under_pressure(self) -> None:
        """The fiery 1.2 GB-free moment on a 32 GB host reads as under pressure (floor ~3.2 GB)."""
        verdict = assess_ram_pressure(1200.0, 32063.0, pause_percent=90.0, min_free_mb=1024.0)
        assert verdict.under_pressure is True
        assert verdict.floor_mb == pytest.approx(3206.3, abs=1.0)

    def test_above_floor_is_not_under_pressure(self) -> None:
        """Ample available RAM reads clear, so a healthy worker is never throttled."""
        verdict = assess_ram_pressure(25000.0, 32063.0, pause_percent=90.0, min_free_mb=1024.0)
        assert verdict.under_pressure is False

    def test_missing_available_never_fabricates_pressure(self) -> None:
        """No telemetry (cold start) yields not-under-pressure, so the worker is never wedged on a guess."""
        verdict = assess_ram_pressure(None, 32063.0, pause_percent=90.0, min_free_mb=1024.0)
        assert verdict.under_pressure is False
        assert verdict.available_mb is None


class TestDisaggregatedCharge:
    """The disaggregated (UNet-only) charge flips verdicts a whole-job charge fails on a tight card.

    A disaggregated job's sampler process holds only the core diffusion weights plus sampling activation
    (~6.6GB for SDXL), not the support weights and VAE decode spike the whole-job figure (~16GB) bakes in.
    On a 16GB card the whole-job charge collapses two samplers to one; the sampler-only charge keeps them
    co-resident, which is the entire performance premise of the pipeline.
    """

    _SAMPLER_ONLY_MB = 6600.0
    _WHOLE_JOB_SAMPLING_MB = 16200.0
    _CORE_WEIGHTS_MB = 5000.0
    _FULL_FOOTPRINT_MB = 12000.0

    def _stub_predictors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(resource_budget, "predict_job_sampler_only_vram_mb", lambda j, b: self._SAMPLER_ONLY_MB)
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda j, b: self._WHOLE_JOB_SAMPLING_MB)
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda j, b: self._CORE_WEIGHTS_MB)
        monkeypatch.setattr(resource_budget, "predict_job_footprint_mb", lambda j, b: self._FULL_FOOTPRINT_MB)
        monkeypatch.setattr(resource_budget, "effective_inference_reserve_mb", lambda *a, **k: 1000.0)

    def test_check_job_verdict_flips_with_disaggregated_charge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job the whole-job charge rejects on the measured free VRAM is admitted as disaggregated."""
        self._stub_predictors(monkeypatch)
        budget = VramBudget(reserve_mb=1000.0)
        job = make_job_pop_response("stable_diffusion_xl")
        # 8000 free covers the sampler-only charge (6600 + 1000) but not the whole-job one (16200 + 1000).
        assert budget.check_job(job, "stable_diffusion_xl", free_vram_mb=8000.0).fits is False
        assert budget.check_job(job, "stable_diffusion_xl", free_vram_mb=8000.0, disaggregated=True).fits is True

    def test_forecast_coresident_flips_with_disaggregated_charge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The streaming forecast co-resides two samplers where the whole-job charge would not fit one."""
        self._stub_predictors(monkeypatch)
        job = make_mock_job(width=1024, height=1024)
        common = {
            "free_now_mb": 15000.0,
            "total_vram_mb": 16375.0,
            "per_process_overhead_mb": 1288.0,
            "num_inference_processes": 1,
            "configured_reserve_floor_mb": 0.0,
        }
        whole = resource_budget.forecast_weight_streaming(job, "stable_diffusion_xl", **common)
        disagg = resource_budget.forecast_weight_streaming(job, "stable_diffusion_xl", disaggregated=True, **common)
        assert whole.fits_coresident is False  # ~16GB sampling peak does not co-reside on a 16GB card
        assert disagg.fits_coresident is True  # ~6.6GB sampler-only charge does

    def test_forecast_charges_sampler_only_footprint_and_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fits_alone / is_card_demanding / max_resident_processes read the sampler-only figure, not the whole job."""
        self._stub_predictors(monkeypatch)
        job = make_mock_job(width=1024, height=1024)
        disagg = resource_budget.forecast_weight_streaming(
            job,
            "stable_diffusion_xl",
            free_now_mb=15000.0,
            total_vram_mb=16375.0,
            per_process_overhead_mb=1288.0,
            num_inference_processes=1,
            configured_reserve_floor_mb=0.0,
            disaggregated=True,
        )
        assert disagg.weights_mb == self._SAMPLER_ONLY_MB
        assert disagg.footprint_mb == self._SAMPLER_ONLY_MB

    def test_lane_decode_spike_charged_as_sibling_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The image lane's concurrent decode spike is subtracted from the siblings-present free figure."""
        self._stub_predictors(monkeypatch)
        job = make_mock_job(width=1024, height=1024)
        common = {
            "free_now_mb": 15000.0,
            "total_vram_mb": 16375.0,
            "per_process_overhead_mb": 1288.0,
            "num_inference_processes": 1,
            "configured_reserve_floor_mb": 0.0,
            "disaggregated": True,
        }
        without_spike = resource_budget.forecast_weight_streaming(job, "stable_diffusion_xl", **common)
        with_spike = resource_budget.forecast_weight_streaming(
            job, "stable_diffusion_xl", disaggregation_sibling_charge_mb=4000.0, **common
        )
        assert without_spike.free_after_model_evict_mb is not None
        assert with_spike.free_after_model_evict_mb == pytest.approx(without_spike.free_after_model_evict_mb - 4000.0)

    def test_decode_spike_predictor_reads_field_and_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The decode-spike predictor reads BurdenEstimate.vram_decode_spike_mb, and is None on an older engine."""
        from types import SimpleNamespace

        job = make_mock_job(width=1024, height=1024)
        monkeypatch.setattr(
            resource_budget,
            "_estimate_job_burden",
            lambda j, b: SimpleNamespace(vram_decode_spike_mb=2535.0),
        )
        assert resource_budget.predict_job_decode_spike_mb(job, "stable_diffusion_xl") == 2535.0

        # A pinned hordelib predating the field lacks the attribute: the predictor degrades to None so the
        # caller falls back to the conservative full lane quota rather than faulting.
        monkeypatch.setattr(resource_budget, "_estimate_job_burden", lambda j, b: SimpleNamespace())
        assert resource_budget.predict_job_decode_spike_mb(job, "stable_diffusion_xl") is None


class TestTwoSamplersCoresidentAcceptance:
    """The collapse-proofing acceptance criterion, pinned with honest measured figures.

    Empirical ground truth on this box (16375MB card): two SDXL disaggregated samplers plus the image lane's
    VAE decode were measured co-resident at a 14851MB whole-card peak, i.e. two ~6158MB samplers plus a
    ~2535MB tiled-decode spike. The coresidency verdict must admit the second sampler when the *bounded*
    decode spike is charged, and (the bug this guards) must NOT be forced to deny it by charging the lane's
    full ~8192MB allocator-guard quota, which over-commits the card and collapses the pipeline to one sampler.
    """

    _TOTAL_MB = 16375.0
    _SAMPLER_ONLY_MB = 6158.0
    _DECODE_SPIKE_MB = 2535.0
    _FULL_LANE_QUOTA_MB = 8192.0
    _OVERHEAD_MB = 1288.0
    _MARGINAL_MB = 300.0

    def _second_sampler_forecast(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        sibling_charge_mb: float,
    ) -> resource_budget.StreamForecast:
        """Forecast admitting a SECOND disaggregated sampler while one sampler and the lane are already up."""
        monkeypatch.setattr(resource_budget, "predict_job_sampler_only_vram_mb", lambda j, b: self._SAMPLER_ONLY_MB)
        monkeypatch.setattr(resource_budget, "effective_inference_reserve_mb", lambda *a, **k: 1000.0)
        # free_now reflects the first sampler and its context already resident on the card.
        free_now_mb = self._TOTAL_MB - self._OVERHEAD_MB - self._SAMPLER_ONLY_MB
        return resource_budget.forecast_weight_streaming(
            make_mock_job(width=1024, height=1024),
            "stable_diffusion_xl",
            free_now_mb=free_now_mb,
            total_vram_mb=self._TOTAL_MB,
            per_process_overhead_mb=self._OVERHEAD_MB,
            num_inference_processes=2,  # both sampler contexts
            num_extra_resident_contexts=1,  # the image lane context
            configured_reserve_floor_mb=0.0,
            marginal_process_overhead_mb=self._MARGINAL_MB,
            disaggregated=True,
            disaggregation_sibling_charge_mb=sibling_charge_mb,
        )

    def test_ground_truth_two_samplers_plus_spike_fit_the_card(self) -> None:
        """The measured whole-card peak (2 samplers + bounded decode spike) is within the 16375MB card."""
        peak = 2 * self._SAMPLER_ONLY_MB + self._DECODE_SPIKE_MB
        assert peak == pytest.approx(14851.0)
        assert peak <= self._TOTAL_MB

    def test_second_sampler_admitted_with_bounded_decode_spike(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Charging the bounded ~2535MB decode spike admits the second sampler co-resident (no collapse)."""
        forecast = self._second_sampler_forecast(monkeypatch, sibling_charge_mb=self._DECODE_SPIKE_MB)
        assert forecast.fits_coresident is True

    def test_full_quota_charge_would_collapse_to_one_sampler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression tripwire: charging the full ~8192MB lane quota instead denies the second sampler."""
        forecast = self._second_sampler_forecast(monkeypatch, sibling_charge_mb=self._FULL_LANE_QUOTA_MB)
        assert forecast.fits_coresident is False


class TestMeasuredAdmissionOverlay:
    """The scheduler's ledger-driven measured floor denies over-commits the lying free-VRAM figure would admit."""

    def _scheduler_with_committed(
        self,
        *,
        reserved_mb: int,
        sampled_at: float,
        total_mb: float,
        baseline_mb: float,
    ) -> object:
        process_map = ProcessMap({})
        info = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        info.process_reserved_mb = reserved_mb  # type: ignore[attr-defined]
        info.report_sampled_at = sampled_at  # type: ignore[attr-defined]
        process_map[0] = info  # type: ignore[index]
        scheduler = _make_inference_scheduler(process_map=process_map)
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=total_mb)  # type: ignore[attr-defined]
        scheduler.resolved_context_constant_mb = Mock(return_value=200.0)  # type: ignore[attr-defined]
        scheduler.set_admission_baseline_provider(lambda _device_index: baseline_mb)
        return scheduler

    def test_fresh_floor_denies_oversized_candidate(self) -> None:
        """A fresh committed floor plus an oversized candidate exceeds real capacity and denies (free lies)."""
        scheduler = self._scheduler_with_committed(
            reserved_mb=10000,
            sampled_at=time.time(),
            total_mb=16375.0,
            baseline_mb=1700.0,
        )
        job = make_job_pop_response("stable_diffusion")
        # committed 200+10000 = 10200; capacity = (16375-1700)-512 = 14163; candidate 5000 => 15200 > 14163.
        verdict = scheduler._measured_admission_verdict(
            job,
            "stable_diffusion_1",
            candidate_delta_mb=5000.0,
            device_index=None,
        )
        assert verdict.used_measured_floor is True
        assert verdict.fits is False

    def test_fresh_floor_admits_within_capacity(self) -> None:
        """A fresh committed floor with a candidate that fits real capacity admits."""
        scheduler = self._scheduler_with_committed(
            reserved_mb=5000,
            sampled_at=time.time(),
            total_mb=16375.0,
            baseline_mb=1700.0,
        )
        job = make_job_pop_response("stable_diffusion")
        verdict = scheduler._measured_admission_verdict(job, "x", candidate_delta_mb=2000.0, device_index=None)
        assert verdict.used_measured_floor is True
        assert verdict.fits is True

    def test_stale_floor_falls_back_to_predictive(self) -> None:
        """A stale committed report relaxes the overlay to admit, so the caller relies on its predictive gate."""
        scheduler = self._scheduler_with_committed(
            reserved_mb=10000,
            sampled_at=time.time() - 100.0,
            total_mb=16375.0,
            baseline_mb=1700.0,
        )
        job = make_job_pop_response("stable_diffusion")
        verdict = scheduler._measured_admission_verdict(
            job,
            "stable_diffusion_1",
            candidate_delta_mb=5000.0,
            device_index=None,
        )
        assert verdict.used_measured_floor is False
        assert verdict.fits is True

    def test_denial_increments_counter_and_relaxed_does_not(self) -> None:
        """A measured denial counts toward the per-card denial metric; a relaxed (stale) evaluation does not."""
        scheduler = self._scheduler_with_committed(
            reserved_mb=10000,
            sampled_at=time.time(),
            total_mb=16375.0,
            baseline_mb=1700.0,
        )
        job = make_job_pop_response("stable_diffusion")
        admits = scheduler._measured_admission_admits(
            job,
            "stable_diffusion_1",
            device_index=None,
            process_id=None,
            disaggregated=False,
            log_context="preload",
        )
        assert admits is False
        assert scheduler.latest_admission_denials(device_index=None) == 1

    def test_pressure_reclaim_issues_under_pressure_unload(self) -> None:
        """The no-candidate pressure reclaim issues one under-pressure idle unload via the shared path."""
        process_map = ProcessMap({})
        process_map[0] = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)  # type: ignore[index]
        scheduler = _make_inference_scheduler(process_map=process_map)
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[attr-defined]
        assert scheduler.reclaim_one_idle_model_under_pressure(device_index=None) is True
        _args, kwargs = scheduler.unload_models_from_vram.call_args
        assert kwargs["under_pressure"] is True

    def test_pressure_reclaim_is_noop_without_inference_process(self) -> None:
        """With no inference process to anchor on, the pressure reclaim is a no-op (nothing to reclaim)."""
        scheduler = _make_inference_scheduler(process_map=ProcessMap({}))
        assert scheduler.reclaim_one_idle_model_under_pressure(device_index=None) is False

    def _double_grant_scheduler(self, monkeypatch: pytest.MonkeyPatch, *, sampling_peak_mb: float) -> object:
        """Build a two-process scheduler whose measured floor leaves room for exactly one fresh admission.

        Loader process 0 holds a 5000 MB reservation (the pre-load floor); process 1 is an idle spare with
        no reservation yet. Capacity is ``(16375 - 1700) - 512 = 14163`` MB and the committed floor is
        ``200 + 5000 = 5200`` MB, so a lone 8000 MB candidate fits (13200) but a second concurrent charge of
        the same size does not. ``predict_job_sampling_vram_mb`` is pinned so a granted preload's planned
        charge equals ``sampling_peak_mb``.
        """
        from horde_worker_regen.process_management.scheduling import inference_scheduler as _sched_mod

        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: sampling_peak_mb)
        # The scheduler binds ``predict_job_sampling_vram_mb`` in its own namespace at import; the grant path's
        # candidate-delta pricing reads that binding, so it must be patched there too.
        monkeypatch.setattr(_sched_mod, "predict_job_sampling_vram_mb", lambda job, baseline: sampling_peak_mb)
        loader = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        loader.process_reserved_mb = 5000  # type: ignore[attr-defined]
        loader.report_sampled_at = time.time()  # type: ignore[attr-defined]
        spare = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.report_sampled_at = time.time()  # type: ignore[attr-defined]
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: loader, 1: spare}), max_inference=2)
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16375.0)  # type: ignore[attr-defined]
        scheduler.resolved_context_constant_mb = Mock(return_value=200.0)  # type: ignore[attr-defined]
        scheduler.set_admission_baseline_provider(lambda _device_index: 1700.0)
        return scheduler

    def test_second_same_cycle_grant_sees_first_planned_charge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The race: a preload granted this cycle registers a planned charge the next admission must count.

        Without the first grant an 8000 MB candidate fits the measured floor; once the first preload is
        admitted (its planned charge landing immediately, before any per-cycle reconcile), the same candidate
        is jointly gated and denied. Proves the grant path inserts the charge synchronously so two admissions
        in one window cannot over-commit the same free VRAM.
        """
        scheduler = self._double_grant_scheduler(monkeypatch, sampling_peak_mb=8000.0)
        loader = scheduler._process_map[0]  # type: ignore[index]
        job_b = make_job_pop_response("model_b")

        # Baseline: with nothing admitted yet, the lone candidate fits real capacity.
        before = scheduler._measured_admission_verdict(job_b, "x", candidate_delta_mb=8000.0, device_index=None)
        assert before.used_measured_floor is True
        assert before.fits is True

        # Grant the first preload for real; the grant path registers its planned charge synchronously.
        job_a = make_job_pop_response("model_a")
        assert scheduler._send_preload(job_a, loader) is True

        # The second admission in the same cycle now sees the first's planned charge and is denied.
        after = scheduler._measured_admission_verdict(job_b, "x", candidate_delta_mb=8000.0, device_index=None)
        assert after.used_measured_floor is True
        assert after.planned_unmaterialized_mb == 8000.0
        assert after.fits is False

    def test_reconcile_keeps_planned_while_loading_and_drops_it_when_materialized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The per-cycle seam: a planned charge is kept while the model loads and dropped once it materialises.

        A grant leaves the model in ``LOADING``; the reconcile inside the verdict keeps its planned charge (the
        second candidate stays denied). When the model reaches VRAM its planned charge is dropped by the same
        reconcile (materialisation is now in the measured floor), and the second candidate admits again, all
        without any explicit release call.
        """
        scheduler = self._double_grant_scheduler(monkeypatch, sampling_peak_mb=8000.0)
        loader = scheduler._process_map[0]  # type: ignore[index]
        job_b = make_job_pop_response("model_b")

        job_a = make_job_pop_response("model_a")
        assert scheduler._send_preload(job_a, loader) is True
        # While model_a is LOADING, the planned charge is retained across the reconcile: B stays denied.
        while_loading = scheduler._measured_admission_verdict(job_b, "x", candidate_delta_mb=8000.0, device_index=None)
        assert while_loading.planned_unmaterialized_mb == 8000.0
        assert while_loading.fits is False

        # model_a materialises into VRAM: the reconcile now omits it (no explicit release), so B admits again.
        scheduler._horde_model_map.update_entry(  # type: ignore[attr-defined]
            horde_model_name="model_a",
            load_state=ModelLoadState.LOADED_IN_VRAM,
            process_id=0,
        )
        after_materialized = scheduler._measured_admission_verdict(
            job_b,
            "x",
            candidate_delta_mb=8000.0,
            device_index=None,
        )
        assert after_materialized.planned_unmaterialized_mb == 0.0
        assert after_materialized.fits is True

    def test_reconcile_drops_planned_when_loading_process_dies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dead/idle loader's planned charge disappears via reconcile with no explicit release on the fault path."""
        scheduler = self._double_grant_scheduler(monkeypatch, sampling_peak_mb=8000.0)
        loader = scheduler._process_map[0]  # type: ignore[index]
        job_a = make_job_pop_response("model_a")
        assert scheduler._send_preload(job_a, loader) is True

        # Simulate the loader dying mid-load: its model map entry leaves the loading state (here, cleared).
        scheduler._horde_model_map.root.pop("model_a", None)  # type: ignore[attr-defined]
        job_b = make_job_pop_response("model_b")
        verdict = scheduler._measured_admission_verdict(job_b, "x", candidate_delta_mb=8000.0, device_index=None)
        assert verdict.planned_unmaterialized_mb == 0.0
        assert verdict.fits is True
