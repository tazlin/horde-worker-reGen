"""Tests for the worker-owned VRAM budget and its scheduler gating."""

from __future__ import annotations

import sys
import time
from unittest.mock import Mock

import pytest
from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE

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
from horde_worker_regen.process_management.resources.vram_footprints import (
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_job,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
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
    """The measured-truth admission inequality: candidate vs device-free room net of reservations and noise."""

    def test_derived_default_buffer_denies_what_the_floor_would_admit_on_a_large_card(self) -> None:
        """With no explicit buffer a 16375MB card derives 818.75, denying a demand that fits under 512."""
        total = 16375.0
        device_free = 1000.0
        # The candidate lands between the derived-buffer room and the floor-buffer room, so the derived default
        # (which the caller gets when it passes no explicit buffer) flips it to a denial.
        candidate = 400.0
        derived = evaluate_admission(
            candidate_outstanding_mb=candidate,
            device_free_mb=device_free,
            outstanding_reservations_mb=0.0,
            total_vram_mb=total,
        )
        assert derived.noise_buffer_mb == pytest.approx(admission_noise_buffer_mb(total))
        assert derived.fits is False
        # The same demand fits when the floor buffer is passed explicitly (the explicit value wins).
        explicit = evaluate_admission(
            candidate_outstanding_mb=candidate,
            device_free_mb=device_free,
            outstanding_reservations_mb=0.0,
            total_vram_mb=total,
            noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
        )
        assert explicit.noise_buffer_mb == pytest.approx(_ADMISSION_NOISE_BUFFER_MB)
        assert explicit.fits is True

    @pytest.mark.parametrize("total_mb", [8192.0, 16375.0, 24564.0])
    def test_inequality_admits_within_room_and_denies_past_it(self, total_mb: float) -> None:
        """Across 8/16/24GB cards, a candidate within available room admits and one past it denies.

        The noise buffer is the card's derived proportional buffer, so the available room scales with the card
        the same way the production snapshot sizes it.
        """
        device_free = total_mb - 2000.0
        noise = admission_noise_buffer_mb(total_mb)
        available = device_free - noise
        fits = evaluate_admission(
            candidate_outstanding_mb=available - 500.0,
            device_free_mb=device_free,
            outstanding_reservations_mb=0.0,
            total_vram_mb=total_mb,
        )
        assert fits.available_known is True
        assert fits.fits is True
        denies = evaluate_admission(
            candidate_outstanding_mb=available + 1.0,
            device_free_mb=device_free,
            outstanding_reservations_mb=0.0,
            total_vram_mb=total_mb,
        )
        assert denies.fits is False

    def test_smaller_candidate_admits_what_the_gross_candidate_would_deny(self) -> None:
        """The resident-weight credit the caller nets out of the candidate admits a load the gross figure denies.

        The identity charges only the outstanding cost, so a candidate priced net of weights already physically
        inside the device-free reading fits where the gross figure (double-charging those weights) denies.
        """
        total, device_free = 16375.0, 8000.0
        available = device_free - _ADMISSION_NOISE_BUFFER_MB
        gross_candidate = available + 1000.0
        resident_credit = 2000.0
        assert (
            evaluate_admission(
                candidate_outstanding_mb=gross_candidate,
                device_free_mb=device_free,
                outstanding_reservations_mb=0.0,
                total_vram_mb=total,
                noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
            ).fits
            is False
        )
        assert (
            evaluate_admission(
                candidate_outstanding_mb=gross_candidate - resident_credit,
                device_free_mb=device_free,
                outstanding_reservations_mb=0.0,
                total_vram_mb=total,
                noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
            ).fits
            is True
        )

    def test_reservation_reduces_available_and_can_flip_a_fitting_candidate(self) -> None:
        """An outstanding reservation (admitted but unmaterialised work) can flip an otherwise-fitting candidate."""
        total, device_free = 16375.0, 8000.0
        candidate = 7000.0
        no_reservation = evaluate_admission(
            candidate_outstanding_mb=candidate,
            device_free_mb=device_free,
            outstanding_reservations_mb=0.0,
            total_vram_mb=total,
            noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
        )
        assert no_reservation.fits is True
        with_reservation = evaluate_admission(
            candidate_outstanding_mb=candidate,
            device_free_mb=device_free,
            outstanding_reservations_mb=1000.0,
            total_vram_mb=total,
            noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
        )
        assert with_reservation.fits is False

    def test_materialised_then_evicted_anchor_does_not_zombie_deny_retention(self) -> None:
        """After a materialise-then-evict cycle the reservation is consumed, so a retention candidate fits.

        The reservation ledger consumes each anchor monotonically: two loads materialise into their targets'
        reservations and are then evicted, and the outstanding reservation collapses to zero rather than
        resurrecting to full weight. With no zombie reservation charged, the just-used model's retention
        candidate fits the recovered device-free room.
        """
        total = 16375.0
        ledger = CommittedReserveLedger()
        ledger.set_planned("preload", "0", vram_mb=6158.0, target_process_id=0, reserved_at_admit_mb=0.0)
        ledger.set_planned("preload", "1", vram_mb=6134.0, target_process_id=1, reserved_at_admit_mb=0.0)
        # Both loads materialise into their targets' reservations.
        ledger.effective_planned_vram_mb({0: 6158.0, 1: 6134.0})
        # Eviction collapses both reservations back toward zero.
        reservations_after_evict = ledger.effective_planned_vram_mb({0: 68.0, 1: 68.0})
        assert reservations_after_evict == 0.0
        # Eviction returned the weights to the card, so device-free recovered to hold the retention candidate.
        verdict = evaluate_admission(
            candidate_outstanding_mb=6158.0,
            device_free_mb=14607.0,
            outstanding_reservations_mb=reservations_after_evict,
            total_vram_mb=total,
        )
        assert verdict.available_known is True
        assert verdict.fits is True

    def test_stacked_reservations_deny_a_further_over_commit(self) -> None:
        """Stacked outstanding reservations plus a fresh candidate deny a further over-commit (the double-admit guard).

        Two preloads staged in RAM hold 12316 MB of outstanding reservations the device-free reading does not
        yet reflect; a fresh 6158 MB candidate on top exceeds the available room, so the identity denies it
        before either staged load has materialised.
        """
        verdict = evaluate_admission(
            candidate_outstanding_mb=6158.0,
            device_free_mb=14675.0,
            outstanding_reservations_mb=12316.0,
            total_vram_mb=16375.0,
            noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
        )
        assert verdict.available_known is True
        assert verdict.fits is False
        assert verdict.available_mb == pytest.approx(14675.0 - 12316.0 - _ADMISSION_NOISE_BUFFER_MB)

    def test_startup_storm_reservation_overlay_denies_third_stacked_preload(self) -> None:
        """A cold-start storm admits the first two RAM-staged preloads and denies the third on their reservations.

        Nothing has materialised (device-free stays at the full card), so each admitted preload registers a
        reservation via the real ledger; against a 16375 MB card the third stacked 6158 MB candidate pushes the
        reservations plus the candidate past the available room and the identity denies it.
        """
        total = 16375.0
        candidate = 6158.0
        ledger = CommittedReserveLedger()
        outcomes: list[bool] = []
        for index in range(3):
            reservations = ledger.effective_planned_vram_mb({})
            verdict = evaluate_admission(
                candidate_outstanding_mb=candidate,
                device_free_mb=total,
                outstanding_reservations_mb=reservations,
                total_vram_mb=total,
                noise_buffer_mb=_ADMISSION_NOISE_BUFFER_MB,
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

    def test_unknown_device_free_is_indeterminate_and_never_admits(self) -> None:
        """With no device-free reading the identity is indeterminate: available is unknown and nothing fits.

        The arbiter maps this to a deferral (never a denial, never a fabricated free figure); the identity
        itself reports ``available_known=False`` and ``fits=False`` so no caller reads it as room.
        """
        verdict = evaluate_admission(
            candidate_outstanding_mb=1000.0,
            device_free_mb=None,
            outstanding_reservations_mb=0.0,
            total_vram_mb=16375.0,
        )
        assert verdict.available_known is False
        assert verdict.fits is False
        assert verdict.available_mb is None

    def test_two_samplers_resident_admit_retention_and_deny_a_heavy_candidate(self) -> None:
        """Two materialised samplers are physically inside device-free: a no-cost retention fits, a heavy load denies.

        The samplers' weights are already in the device-free reading, so a retention candidate (no new
        footprint) fits the residual room while a candidate larger than that room denies.
        """
        total = 16375.0
        device_free = 1975.0  # two ~4900MB samplers plus contexts and an idle VAE lane are resident
        verdict = evaluate_admission(
            candidate_outstanding_mb=0.0,
            device_free_mb=device_free,
            outstanding_reservations_mb=0.0,
            total_vram_mb=total,
        )
        assert verdict.available_known is True
        assert verdict.fits is True
        assert verdict.available_mb is not None
        over = evaluate_admission(
            candidate_outstanding_mb=verdict.available_mb + 1.0,
            device_free_mb=device_free,
            outstanding_reservations_mb=0.0,
            total_vram_mb=total,
        )
        assert over.fits is False

    def test_oversized_candidate_does_not_fit_so_caller_defers_or_denies(self) -> None:
        """A wildly oversized (mis-estimated) candidate does not fit: the caller defers rather than faulting."""
        verdict = evaluate_admission(
            candidate_outstanding_mb=20000.0,
            device_free_mb=14675.0,
            outstanding_reservations_mb=0.0,
            total_vram_mb=16375.0,
        )
        assert verdict.available_known is True
        assert verdict.fits is False

    def test_reason_renders_the_full_identity(self) -> None:
        """The verdict reason renders every term of the identity for a self-explaining log line."""
        rendered = evaluate_admission(
            candidate_outstanding_mb=200.0,
            device_free_mb=8000.0,
            outstanding_reservations_mb=100.0,
            total_vram_mb=16375.0,
        ).reason()
        assert "candidate" in rendered
        assert "device-free" in rendered
        assert "reservations" in rendered
        assert "noise" in rendered
        assert "available" in rendered
        # An indeterminate (no device-free reading) verdict explains why admission is deferred.
        deferred = evaluate_admission(
            candidate_outstanding_mb=200.0,
            device_free_mb=None,
            outstanding_reservations_mb=0.0,
            total_vram_mb=16375.0,
        ).reason()
        assert "unavailable" in deferred
        assert "deferred" in deferred


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
        # A second, idle process holding a different resident model: the eviction candidate. Its resident
        # weights physically fill the card (the truthful device-free reading is 1000 MB), so admission denies
        # the incoming head and describes an eviction of this idle resident.
        resident = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.WAITING_FOR_JOB)
        resident.total_vram_mb = 16000
        resident.vram_usage_mb = 15000  # 1000 MB free, well under 8000 + 2000
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
            device_free_mb=1000.0,
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
        assert forecast.needs_process_count_reduction is False


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
        """The over-counted overhead denies model eviction its room; the measured marginal restores it.

        ``fits_after_model_evict`` tracks the activation-keyed over-count: the over-counted overhead
        suppresses it, and a measured marginal restores it. Neither teardown decision follows that
        activation-keyed reading. Both are decided on the persistent weight footprint, so a moderate 4.9 GB
        model is neither granted sole residency (``needs_exclusive_residency``) nor given a process-count cut
        (``needs_process_count_reduction``) regardless of the over-count. That is the safer outcome for the
        probe-overhead wedge: the phantom over-count cannot drive a teardown demand the scheduler acts on.
        """
        # The over-count is charging the full first-context overhead against every context; pass it
        # explicitly as the marginal to reproduce that (an unmeasured marginal now seeds a small constant,
        # so None no longer over-counts).
        over_counted = self._forecast(
            monkeypatch,
            per_process_overhead_mb=4266.0,
            marginal_process_overhead_mb=4266.0,
        )
        assert over_counted.fits_after_model_evict is False
        assert over_counted.needs_exclusive_residency is False
        assert over_counted.needs_process_count_reduction is False

        with_marginal = self._forecast(monkeypatch, per_process_overhead_mb=4266.0, marginal_process_overhead_mb=391.0)
        assert with_marginal.needs_exclusive_residency is False
        assert with_marginal.needs_process_count_reduction is False
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


class TestLearnedResidentFootprint:
    """A measured per-checkpoint resident footprint raises the forecast's static per-baseline seed.

    The static seeds are per-baseline, so every checkpoint of an architecture is priced alike; a heavy
    checkpoint that measurably costs far more than its baseline's seed is granted sibling room that does not
    exist. The measured at-rest figure raises the seed (raise-only, so nothing measured leaves the arithmetic
    exactly as the seeds compute it) and re-keys the sibling-room judgment on what the file actually costs.
    """

    _TOTAL_MB = 24576.0
    _OVERHEAD_MB = 1288.0
    _CORE_WEIGHTS_MB = 4900.0
    _SEED_FOOTPRINT_MB = 6900.0
    _MEASURED_FOOTPRINT_MB = 18000.0

    def _forecast(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        learned_resident_footprint_mb: float | None,
    ) -> resource_budget.StreamForecast:
        """A 24GB card holding one SDXL-class checkpoint, priced with or without a measured resident figure."""
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda j, b: self._CORE_WEIGHTS_MB)
        monkeypatch.setattr(resource_budget, "predict_job_footprint_mb", lambda j, b: self._SEED_FOOTPRINT_MB)
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda j, b: 9000.0)
        monkeypatch.setattr(resource_budget, "effective_inference_reserve_mb", lambda *a, **k: 1000.0)
        return resource_budget.forecast_weight_streaming(
            make_mock_job(width=1024, height=1024),
            "stable_diffusion_xl",
            free_now_mb=20000.0,
            total_vram_mb=self._TOTAL_MB,
            per_process_overhead_mb=self._OVERHEAD_MB,
            num_inference_processes=1,
            configured_reserve_floor_mb=0.0,
            learned_resident_footprint_mb=learned_resident_footprint_mb,
        )

    def test_seeded_card_has_room_for_a_sibling_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Baseline: at the per-baseline seed the 24GB card reads as holding another full model."""
        forecast = self._forecast(monkeypatch, learned_resident_footprint_mb=None)
        assert forecast._has_room_for_coresident_model is True

    def test_measured_footprint_removes_the_sibling_room(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A checkpoint measured at 18GB leaves no room for a sibling model the seed said would fit."""
        forecast = self._forecast(monkeypatch, learned_resident_footprint_mb=self._MEASURED_FOOTPRINT_MB)
        assert forecast._has_room_for_coresident_model is False
        assert forecast.footprint_mb == self._MEASURED_FOOTPRINT_MB

    def test_measured_footprint_leaves_the_core_weights_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The raise lands on the footprint only: the core-weight term drives load feasibility and must not move.

        ``weights_mb`` is deliberately the core diffusion weights without the support components (which
        time-share the card via per-phase swaps), and the at-rest measurement carries no attribution of that
        split. Folding a combined figure into it would make a model whose weights fit the drained card read
        as streaming-unavoidable.
        """
        forecast = self._forecast(monkeypatch, learned_resident_footprint_mb=self._MEASURED_FOOTPRINT_MB)
        assert forecast.weights_mb == self._CORE_WEIGHTS_MB
        assert forecast.streams_unavoidably is False
        assert forecast.fits_alone is True

    def test_measurement_below_the_seed_changes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raise-only: a measurement lighter than the seed leaves the conservative seed standing."""
        light = self._forecast(monkeypatch, learned_resident_footprint_mb=self._SEED_FOOTPRINT_MB - 2000.0)
        assert light.footprint_mb == self._SEED_FOOTPRINT_MB

    def test_cold_store_is_identical_to_the_static_arithmetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The regression keystone: with nothing measured, every forecast field is the static figure."""
        cold = self._forecast(monkeypatch, learned_resident_footprint_mb=None)
        seeded = self._forecast(monkeypatch, learned_resident_footprint_mb=self._SEED_FOOTPRINT_MB)
        assert cold == seeded

    def test_disaggregated_forecast_ignores_the_whole_slot_measurement(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A disaggregated sampler holds the UNet alone, so a whole-slot resident figure must not price it.

        Applying it would re-add the support weights the sampler process never carries and collapse the two
        co-resident samplers the sampler-only charge exists to keep.
        """
        monkeypatch.setattr(resource_budget, "predict_job_sampler_only_vram_mb", lambda j, b: 6600.0)
        monkeypatch.setattr(resource_budget, "effective_inference_reserve_mb", lambda *a, **k: 1000.0)
        common = {
            "free_now_mb": 20000.0,
            "total_vram_mb": self._TOTAL_MB,
            "per_process_overhead_mb": self._OVERHEAD_MB,
            "num_inference_processes": 1,
            "configured_reserve_floor_mb": 0.0,
            "disaggregated": True,
        }
        job = make_mock_job(width=1024, height=1024)
        without = resource_budget.forecast_weight_streaming(job, "stable_diffusion_xl", **common)
        with_measurement = resource_budget.forecast_weight_streaming(
            job,
            "stable_diffusion_xl",
            learned_resident_footprint_mb=self._MEASURED_FOOTPRINT_MB,
            **common,
        )
        assert with_measurement == without


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


class TestMeasuredWholeCardRetirement:
    """A whole-card claim made by a seed is retired once the checkpoint has actually been measured.

    The seed for a Flux fp8 checkpoint charges 16.4 GB of resident footprint (11.5 GB core weights plus
    4.9 GB of support components). On a 16 GB card that leaves no room for a sibling model by the seed's
    own arithmetic, so the intent branch reserves the device. The card measurably holds the same job at
    13.5 GB device-used, and the residency it bought changed nothing about how the job ran.
    """

    _MEASURED_MARGINED_MB = 14850.0
    """The store's margined figure for five 13500 MB observations, with the context charge netted back out
    the way the scheduler hands it to the forecast."""

    def _flux_forecast(self, *, measured_footprint_mb: float | None) -> resource_budget.StreamForecast:
        return resource_budget.StreamForecast(
            weights_mb=11500.0,
            footprint_mb=16400.0,
            reserve_mb=2048.0,
            base_reserve_mb=1519.0,
            free_now_mb=13000.0,
            free_if_alone_mb=16100.0,
            free_after_model_evict_mb=13500.0,
            total_vram_mb=16375.0,
            per_process_overhead_mb=275.0,
            marginal_process_overhead_mb=275.0,
            wants_whole_card=True,
            measured_footprint_mb=measured_footprint_mb,
            measured_observation_count=0 if measured_footprint_mb is None else 5,
        )

    def test_unmeasured_seed_still_claims_the_card(self) -> None:
        """Without measurements the seed governs and the declared intent stands, exactly as before."""
        forecast = self._flux_forecast(measured_footprint_mb=None)
        assert forecast.fits_alone is True
        assert forecast._has_room_for_coresident_model is False
        assert forecast.measured_retires_whole_card_intent is False
        assert forecast.needs_exclusive_residency is True

    def test_measured_footprint_retires_the_claim(self) -> None:
        """Measured, margined, the model leaves the card room for a sibling context, so it stops reserving it."""
        forecast = self._flux_forecast(measured_footprint_mb=self._MEASURED_MARGINED_MB)
        assert forecast.measured_retires_whole_card_intent is True
        assert forecast.needs_exclusive_residency is False
        # The retirement is not a claim that the weights fit everywhere: the ordinary tests still apply.
        assert forecast.fits_coresident is False
        assert forecast._persistent_weights_dominant is False

    def test_a_measured_figure_that_fills_the_card_keeps_the_claim(self) -> None:
        """Measurement retires the claim only when it disproves it; a full card still gets reserved."""
        forecast = self._flux_forecast(measured_footprint_mb=16000.0)
        assert forecast.measured_retires_whole_card_intent is False
        assert forecast.needs_exclusive_residency is True

    def test_retirement_needs_a_sized_card(self) -> None:
        """A card whose capacity is unknown keeps the declared intent (the conservative direction)."""
        forecast = resource_budget.StreamForecast(
            weights_mb=11500.0,
            footprint_mb=16400.0,
            reserve_mb=2048.0,
            free_now_mb=13000.0,
            free_if_alone_mb=None,
            free_after_model_evict_mb=None,
            wants_whole_card=True,
            measured_footprint_mb=self._MEASURED_MARGINED_MB,
        )
        assert forecast.measured_retires_whole_card_intent is False
        assert forecast.needs_exclusive_residency is True


class TestSchedulerMeasuredWholeCardRetirement:
    """End to end through the scheduler: observations in the store retire the whole-card claim in the forecast.

    The unit tests above pin the retirement rule on a hand-built forecast. This one drives the whole seam the
    worker actually uses: a store the scheduler shares with the dispatcher, keyed and margined by the store,
    netted of the context charge by the scheduler, and read by ``_forecast_streaming``. It fails if any link
    in that chain is missing.
    """

    _FLUX_MODEL = "Flux.1-Schnell fp8 (Compact)"
    _TOTAL_MB = 16375.0
    _OVERHEAD_MB = 275.0
    _CORE_WEIGHTS_MB = 11500.0
    _SEED_FOOTPRINT_MB = 16400.0
    _MEASURED_DEVICE_USED_MB = 13500.0

    def _scheduler(self, monkeypatch: pytest.MonkeyPatch) -> tuple[object, LearnedFootprintStore]:
        """A 16GB card serving a Flux checkpoint whose seed reserves the whole device."""
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda j, b: self._CORE_WEIGHTS_MB)
        monkeypatch.setattr(resource_budget, "predict_job_footprint_mb", lambda j, b: self._SEED_FOOTPRINT_MB)
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda j, b: 15218.0)
        monkeypatch.setattr(resource_budget, "effective_inference_reserve_mb", lambda *a, **k: 1519.0)

        process_info = make_mock_process_info(1, model_name=self._FLUX_MODEL, state=HordeProcessState.WAITING_FOR_JOB)
        process_info.total_vram_mb = self._TOTAL_MB
        process_info.vram_usage_mb = self._TOTAL_MB - 13000.0
        reference = {
            self._FLUX_MODEL: make_mock_model_reference_record(
                self._FLUX_MODEL,
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
            ),
        }
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({1: process_info}),
            bridge_data=make_mock_bridge_data(
                enable_vram_budget=True,
                whole_card_exclusive_residency=True,
                vram_per_process_overhead_mb=self._OVERHEAD_MB,
                safety_on_gpu=False,
            ),
            model_metadata=make_test_model_metadata(reference),
            device_free_mb=13000.0,
        )
        store = LearnedFootprintStore()
        scheduler.set_footprint_store(store)
        return scheduler, store

    def _observe(self, store: LearnedFootprintStore, count: int) -> None:
        key = FootprintKey(
            model_baseline=str(KNOWN_IMAGE_GENERATION_BASELINE.flux_1),
            resolution_bucket=None,
            platform=sys.platform,
            stage=FootprintStage.RESIDENT,
            checkpoint=self._FLUX_MODEL,
        )
        for _ in range(count):
            store.observe_peak(key, self._MEASURED_DEVICE_USED_MB)

    def test_unobserved_checkpoint_claims_the_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing measured: the seed's arithmetic stands and the whole-card claim is granted."""
        scheduler, _ = self._scheduler(monkeypatch)
        job = make_job_pop_response(self._FLUX_MODEL, width=1024, height=1024)
        forecast = scheduler._forecast_streaming(job, str(KNOWN_IMAGE_GENERATION_BASELINE.flux_1))

        assert forecast.measured_footprint_mb is None
        assert forecast.needs_exclusive_residency is True

    def test_under_observed_checkpoint_still_claims_the_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One short of the threshold, the measurements do not yet answer and nothing changes."""
        scheduler, store = self._scheduler(monkeypatch)
        self._observe(store, 4)
        job = make_job_pop_response(self._FLUX_MODEL, width=1024, height=1024)
        forecast = scheduler._forecast_streaming(job, str(KNOWN_IMAGE_GENERATION_BASELINE.flux_1))

        assert forecast.measured_footprint_mb is None
        assert forecast.needs_exclusive_residency is True

    def test_observed_checkpoint_retires_the_claim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """At the threshold the measured figure reaches the forecast and the card stops being reserved."""
        scheduler, store = self._scheduler(monkeypatch)
        self._observe(store, 5)
        job = make_job_pop_response(self._FLUX_MODEL, width=1024, height=1024)
        forecast = scheduler._forecast_streaming(job, str(KNOWN_IMAGE_GENERATION_BASELINE.flux_1))

        assert forecast.measured_footprint_mb is not None
        assert forecast.measured_observation_count == 5
        assert forecast.measured_retires_whole_card_intent is True
        assert forecast.needs_exclusive_residency is False
        # The seed still governs every fit decision: only the residency claim is retired.
        assert forecast.footprint_mb == self._SEED_FOOTPRINT_MB
        assert forecast.weights_mb == self._CORE_WEIGHTS_MB


class TestResidencyVerdictWording:
    """The logged verdict names what actually granted the residency.

    An intent-driven residency printed "weight-dominant" whenever the card read full, sending a reader after
    a weight estimate that was never the reason for the teardown.
    """

    def test_intent_driven_residency_is_named_as_intent(self) -> None:
        """A whole-card baseline whose weights are not dominant reports intent, however full the card reads."""
        forecast = resource_budget.StreamForecast(
            weights_mb=11500.0,
            footprint_mb=16400.0,
            reserve_mb=2048.0,
            base_reserve_mb=1519.0,
            free_now_mb=13000.0,
            free_if_alone_mb=16100.0,
            free_after_model_evict_mb=13500.0,
            total_vram_mb=16375.0,
            per_process_overhead_mb=275.0,
            marginal_process_overhead_mb=275.0,
            wants_whole_card=True,
        )
        assert forecast.needs_exclusive_residency is True
        assert forecast.fits_coresident is False
        assert forecast._persistent_weights_dominant is False
        assert "sole residency on intent" in forecast.reason()
        assert "weight-dominant" not in forecast.reason()

    def test_weight_dominant_residency_is_named_as_weight_dominant(self) -> None:
        """A model whose persistent weights genuinely fill the card keeps the weight-dominant verdict."""
        forecast = resource_budget.StreamForecast(
            weights_mb=14500.0,
            reserve_mb=2048.0,
            base_reserve_mb=1519.0,
            free_now_mb=13000.0,
            free_if_alone_mb=16100.0,
            free_after_model_evict_mb=13500.0,
            total_vram_mb=16375.0,
            per_process_overhead_mb=275.0,
            marginal_process_overhead_mb=275.0,
            wants_whole_card=False,
        )
        assert forecast._persistent_weights_dominant is True
        assert forecast.needs_exclusive_residency is True
        assert "weight-dominant" in forecast.reason()


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

    def _disagg_forecast(self) -> resource_budget.StreamForecast:
        """A single disaggregated SDXL sampler forecast on a 16GB card, against already-stubbed predictors."""
        return resource_budget.forecast_weight_streaming(
            make_mock_job(width=1024, height=1024),
            "stable_diffusion_xl",
            free_now_mb=15000.0,
            total_vram_mb=16375.0,
            per_process_overhead_mb=1288.0,
            num_inference_processes=1,
            configured_reserve_floor_mb=0.0,
            disaggregated=True,
        )

    def test_forecast_charges_core_weights_and_the_sampler_only_peak(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The persistent charge is the core diffusion weights; the sampler-only figure is the peak.

        The sampler-only figure is core weights plus a resolution- and batch-scaled activation delta. Pricing
        it as persistent weights makes a transient spike read as permanent residency, which is what turns a
        high-resolution job into a whole-card claim on a card that was holding it comfortably. Split, the
        delta lands in the activation-inclusive reserve where the spike belongs, and no support weights are
        charged because the sampler process holds none.
        """
        self._stub_predictors(monkeypatch)
        disagg = self._disagg_forecast()
        assert disagg.weights_mb == self._CORE_WEIGHTS_MB
        assert disagg.footprint_mb == self._CORE_WEIGHTS_MB
        assert disagg.reserve_mb == pytest.approx(self._SAMPLER_ONLY_MB - self._CORE_WEIGHTS_MB)

    def test_the_split_re_sums_to_the_sampler_only_charge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Weights plus the activation-inclusive reserve still total the sampler-only figure.

        The co-residency tests (``fits_coresident`` / ``_fits_peak``) key on that sum, so the split moves the
        activation between terms without changing what a second sampler is admitted against.
        """
        self._stub_predictors(monkeypatch)
        disagg = self._disagg_forecast()
        assert disagg.weights_mb is not None
        assert disagg.weights_mb + disagg.reserve_mb == pytest.approx(self._SAMPLER_ONLY_MB)

    def test_a_high_activation_sampler_is_not_weight_dominant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A sampler whose activation delta fills the card still reads as co-residable, not card-claiming.

        The residency tests price what stays resident between steps. With the split, an SDXL sampler at a
        large resolution and batch keeps its moderate core weights, so neither sole residency nor a
        sibling-process teardown is demanded on the strength of its activation.
        """
        self._stub_predictors(monkeypatch)
        # Two inference contexts, safety on the card, and the image lane's decode spike: the charges that
        # squeeze the siblings-present figure without any of them being an inference context.
        forecast = resource_budget.forecast_weight_streaming(
            make_mock_job(width=1024, height=1024),
            "stable_diffusion_xl",
            free_now_mb=8000.0,
            total_vram_mb=16375.0,
            per_process_overhead_mb=1288.0,
            num_inference_processes=2,
            num_extra_resident_contexts=1,
            safety_context_charge_mb=3044.0,
            marginal_process_overhead_mb=384.0,
            configured_reserve_floor_mb=0.0,
            disaggregated=True,
            disaggregation_sibling_charge_mb=2500.0,
        )
        assert forecast.needs_exclusive_residency is False
        assert forecast.needs_process_count_reduction is False

    def test_older_engine_fallback_keeps_the_conservative_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hordelib predating the phase split reports the whole-job figure, which stays the peak.

        The predictor falls back to the whole-job sampling charge when ``vram_sampler_only_mb`` is absent, so
        the co-residency sum is that conservative figure and an older engine keeps the behavior it had.
        """
        self._stub_predictors(monkeypatch)
        monkeypatch.setattr(
            resource_budget,
            "predict_job_sampler_only_vram_mb",
            lambda j, b: self._WHOLE_JOB_SAMPLING_MB,
        )
        disagg = self._disagg_forecast()
        assert disagg.weights_mb is not None
        assert disagg.weights_mb + disagg.reserve_mb == pytest.approx(self._WHOLE_JOB_SAMPLING_MB)
        assert disagg.fits_coresident is False

    def test_unestimable_core_weights_fall_back_to_the_sampler_charge(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no core weight seed the sampler charge stands in, so the forecast stays priced and cautious."""
        self._stub_predictors(monkeypatch)
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda j, b: None)
        disagg = self._disagg_forecast()
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
    decode spike is charged, and must NOT be forced to deny it by charging the lane's full ~8192MB
    allocator-guard quota, which over-commits the card and collapses the pipeline to one sampler.

    Which of the two charges is used decides the verdict once the sampler's own activation is large enough
    that the difference between them straddles it. The discrimination is therefore driven at a
    higher-resolution sampler (SDXL at 1216 square, whose sampler-only figure is ~6674MB against ~4900MB of
    core weights) where two samplers plus the bounded spike still fit the card with room to spare. At the
    smallest sizes the card holds a second sampler under either charge, so the choice of charge does not
    matter there and only a larger sampler puts it under test.
    """

    _TOTAL_MB = 16375.0
    _SAMPLER_ONLY_MB = 6158.0
    _HIRES_SAMPLER_ONLY_MB = 6674.0
    """hordelib's sampler-only figure for SDXL at 1216 square: the size at which the lane charge decides
    whether a second sampler is admitted."""
    _DECODE_SPIKE_MB = 2535.0
    _FULL_LANE_QUOTA_MB = 8192.0
    _OVERHEAD_MB = 1288.0
    _MARGINAL_MB = 300.0

    def _second_sampler_forecast(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        sibling_charge_mb: float,
        sampler_only_mb: float,
        edge_mb: int = 1216,
    ) -> resource_budget.StreamForecast:
        """Forecast admitting a SECOND disaggregated sampler while one sampler and the lane are already up."""
        monkeypatch.setattr(resource_budget, "predict_job_sampler_only_vram_mb", lambda j, b: sampler_only_mb)
        monkeypatch.setattr(resource_budget, "effective_inference_reserve_mb", lambda *a, **k: 1000.0)
        # free_now reflects the first sampler and its context already resident on the card.
        free_now_mb = self._TOTAL_MB - self._OVERHEAD_MB - sampler_only_mb
        return resource_budget.forecast_weight_streaming(
            make_mock_job(width=edge_mb, height=edge_mb),
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

    def test_ground_truth_holds_at_the_size_the_charge_is_tested_on(self) -> None:
        """Two of the larger samplers plus the bounded spike also fit, so admitting the second is correct.

        Without this the discriminating pair below could pass by admitting a second sampler the card cannot
        physically hold, which would make the bounded charge right for the wrong reason.
        """
        peak = 2 * self._HIRES_SAMPLER_ONLY_MB + self._DECODE_SPIKE_MB
        assert peak <= self._TOTAL_MB

    def test_second_sampler_admitted_with_bounded_decode_spike(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Charging the bounded ~2535MB decode spike admits the second sampler co-resident (no collapse)."""
        forecast = self._second_sampler_forecast(
            monkeypatch,
            sibling_charge_mb=self._DECODE_SPIKE_MB,
            sampler_only_mb=self._HIRES_SAMPLER_ONLY_MB,
        )
        assert forecast.fits_coresident is True

    def test_full_quota_charge_would_collapse_to_one_sampler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Charging the full ~8192MB lane quota instead denies a second sampler the card physically holds.

        The pair with the test above is the whole point: the same card, the same sampler, and the same
        arithmetic reach opposite verdicts on which lane charge is used, so charging the allocator-guard
        quota is what collapses the pipeline to one sampler.
        """
        forecast = self._second_sampler_forecast(
            monkeypatch,
            sibling_charge_mb=self._FULL_LANE_QUOTA_MB,
            sampler_only_mb=self._HIRES_SAMPLER_ONLY_MB,
        )
        assert forecast.fits_coresident is False


class TestSchedulerReservationOverlay:
    """The scheduler assembles the reservation overlay from grants and reconcile-by-omission.

    A grant registers a reservation synchronously and reconcile-by-omission drops it on materialisation or
    death, so two admissions in one window cannot over-admit the same device-free room and a re-ask is never
    blocked by a dead unit's stale reservation.
    """

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
        """Build a two-process scheduler whose device-free room leaves space for exactly one fresh admission.

        Loader process 0 holds a 5000 MB reservation (its admit-time baseline); process 1 is an idle spare with
        no reservation yet. ``predict_job_sampling_vram_mb`` is pinned so a granted preload's reservation equals
        ``sampling_peak_mb``. The tests drive the scheduler's assembled device state with a device-free reading
        that fits one such candidate but not two once the first grant's reservation is charged.
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

    @staticmethod
    def _candidate_fits(state: object, *, candidate_mb: float) -> bool:
        """Whether ``candidate_mb`` fits the assembled state's device-free room net of its reservations and noise."""
        return evaluate_admission(
            candidate_outstanding_mb=candidate_mb,
            device_free_mb=state.device_free_mb,  # type: ignore[attr-defined]
            outstanding_reservations_mb=state.planned_unmaterialized_mb,  # type: ignore[attr-defined]
            total_vram_mb=state.total_vram_mb,  # type: ignore[attr-defined]
            noise_buffer_mb=state.noise_buffer_mb,  # type: ignore[attr-defined]
        ).fits

    def test_second_same_cycle_grant_registers_a_reservation_the_next_admission_counts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A preload granted this cycle registers a reservation synchronously that the next admission counts.

        Before the grant an 8000 MB candidate fits the device-free room; once the first preload is admitted its
        reservation lands immediately (before any per-cycle reconcile), so the same candidate now sees the
        reservation and is denied. The grant path inserts the reservation synchronously, so two admissions in
        one window cannot over-admit the same device-free room.
        """
        scheduler = self._double_grant_scheduler(monkeypatch, sampling_peak_mb=8000.0)
        loader = scheduler._process_map[0]  # type: ignore[index]
        device_free = 13200.0

        before = scheduler.build_vram_arbiter_device_state(None, device_free_mb=device_free)  # type: ignore[attr-defined]
        assert before.planned_unmaterialized_mb == 0.0
        assert self._candidate_fits(before, candidate_mb=8000.0) is True

        # Grant the first preload for real; the grant path registers its reservation synchronously.
        job_a = make_job_pop_response("model_a")
        assert scheduler._send_preload(job_a, loader) is True

        after = scheduler.build_vram_arbiter_device_state(None, device_free_mb=device_free)  # type: ignore[attr-defined]
        assert after.planned_unmaterialized_mb == 8000.0
        assert self._candidate_fits(after, candidate_mb=8000.0) is False

    def test_reconcile_keeps_reservation_while_loading_and_drops_it_when_materialized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A reservation is kept while the model loads and dropped once it materialises, without a release call.

        A grant leaves the model in ``LOADING``; the per-cycle reconcile keeps its reservation, so a second
        candidate over the same room stays denied. When the model reaches VRAM (physically inside the device-free
        reading now) the same reconcile drops the reservation by omission, and with the device-free reading held
        fixed the room the reservation was protecting is returned, so the second candidate admits again.
        """
        scheduler = self._double_grant_scheduler(monkeypatch, sampling_peak_mb=8000.0)
        loader = scheduler._process_map[0]  # type: ignore[index]
        device_free = 13200.0

        job_a = make_job_pop_response("model_a")
        assert scheduler._send_preload(job_a, loader) is True
        while_loading = scheduler.build_vram_arbiter_device_state(None, device_free_mb=device_free)  # type: ignore[attr-defined]
        assert while_loading.planned_unmaterialized_mb == 8000.0
        assert self._candidate_fits(while_loading, candidate_mb=8000.0) is False

        # model_a materialises into VRAM: the reconcile now omits it (no explicit release), so the reservation drops.
        scheduler._horde_model_map.update_entry(  # type: ignore[attr-defined]
            horde_model_name="model_a",
            load_state=ModelLoadState.LOADED_IN_VRAM,
            process_id=0,
        )
        after_materialized = scheduler.build_vram_arbiter_device_state(None, device_free_mb=device_free)  # type: ignore[attr-defined]
        assert after_materialized.planned_unmaterialized_mb == 0.0
        assert self._candidate_fits(after_materialized, candidate_mb=8000.0) is True

    def test_reconcile_drops_reservation_when_loading_process_dies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dead loader's reservation disappears via reconcile with no explicit release on the fault path."""
        scheduler = self._double_grant_scheduler(monkeypatch, sampling_peak_mb=8000.0)
        loader = scheduler._process_map[0]  # type: ignore[index]
        device_free = 13200.0
        job_a = make_job_pop_response("model_a")
        assert scheduler._send_preload(job_a, loader) is True

        # Simulate the loader dying mid-load: its model map entry leaves the loading state (here, cleared).
        scheduler._horde_model_map.root.pop("model_a", None)  # type: ignore[attr-defined]
        after_death = scheduler.build_vram_arbiter_device_state(None, device_free_mb=device_free)  # type: ignore[attr-defined]
        assert after_death.planned_unmaterialized_mb == 0.0
        assert self._candidate_fits(after_death, candidate_mb=8000.0) is True


class TestRecommendationIsNotAPeakFloor:
    """A baseline's ``min_recommended_vram_mb`` must never raise a job's priced sampling peak.

    That field is a recommendation of how much card to bring, not a measurement of any peak a job reaches.
    Flooring the sampling prediction with it priced every job of a baseline at the recommendation, and on a
    card whose achievable ceiling sits below that figure the unmeasured number alone made the candidate
    structurally impossible: an 8 GB card offers ~7680 MB after the noise buffer and the foreign floor, SDXL
    recommends 8000 MB, and a 512x512 SDXL job whose real sampling estimate is well under the ceiling was
    denied and the model dropped from the offer.
    """

    _EIGHT_GB_ACHIEVABLE_CEILING_MB = 7680.0
    """What an 8 GB card can actually offer one load: 8192 total, less the noise buffer and the foreign floor."""

    def test_recommendation_does_not_raise_a_priced_sampling_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An absurd baseline recommendation leaves the sampling prediction exactly where the estimate put it."""
        job = make_mock_job(width=512, height=512)
        priced_mb = resource_budget.predict_job_sampling_vram_mb(job, "stable_diffusion_xl")
        assert priced_mb is not None

        monkeypatch.setattr(resource_budget, "_baseline_load_peak_mb", lambda _baseline: 99000.0)
        assert resource_budget.predict_job_sampling_vram_mb(job, "stable_diffusion_xl") == priced_mb

    def test_sdxl_at_512_prices_under_an_8gb_card_ceiling(self) -> None:
        """The operator's premise, as arithmetic: a 512x512 SDXL job fits what an 8 GB card can offer."""
        job = make_mock_job(width=512, height=512)
        priced_mb = resource_budget.predict_job_sampling_vram_mb(job, "stable_diffusion_xl")
        assert priced_mb is not None
        assert priced_mb <= self._EIGHT_GB_ACHIEVABLE_CEILING_MB, (
            f"a 512x512 SDXL job prices at {priced_mb:.0f} MB, above the "
            f"{self._EIGHT_GB_ACHIEVABLE_CEILING_MB:.0f} MB an 8 GB card can offer; an 8 GB worker cannot "
            "serve SDXL at this price"
        )

    def test_recommendation_still_seeds_when_no_estimate_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no burden estimate at all the recommendation stands: a seed in place of a price, not a floor."""
        job = make_mock_job(width=512, height=512)
        monkeypatch.setattr(resource_budget, "_estimate_job_burden", lambda _job, _baseline: None)
        monkeypatch.setattr(resource_budget, "_baseline_load_peak_mb", lambda _baseline: 8000.0)
        assert resource_budget.predict_job_sampling_vram_mb(job, "stable_diffusion_xl") == 8000.0

    def test_a_larger_job_still_prices_higher(self) -> None:
        """Dropping the floor does not flatten the prediction: resolution still moves the priced peak."""
        small_mb = resource_budget.predict_job_sampling_vram_mb(
            make_mock_job(width=512, height=512),
            "stable_diffusion_xl",
        )
        large_mb = resource_budget.predict_job_sampling_vram_mb(
            make_mock_job(width=1024, height=1024),
            "stable_diffusion_xl",
        )
        assert small_mb is not None and large_mb is not None
        assert large_mb > small_mb


class TestControlnetWorkflowPricing:
    """A named controlnet workflow is priced as a controlnet job, so the gates carry its extra weights."""

    def test_a_qr_code_workflow_counts_as_a_controlnet_feature(self) -> None:
        """The feature list a qr_code job implies includes controlnet even with no control_type set."""
        from hordelib.feature_impact import FEATURE_KIND

        plain = make_job_pop_response("sdxl", width=1024, height=1024, workflow=None)
        qr = make_job_pop_response("sdxl", width=1024, height=1024, workflow="qr_code")
        assert qr.payload.control_type is None

        assert FEATURE_KIND.controlnet not in resource_budget._job_feature_kinds(plain)
        assert FEATURE_KIND.controlnet in resource_budget._job_feature_kinds(qr)

    def test_a_qr_code_job_is_priced_above_the_same_plain_job(self) -> None:
        """The sampling peak the admission and overlap gates read rises by the controlnet burden."""
        baseline = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl.value
        plain = make_job_pop_response("sdxl", width=1024, height=1024, workflow=None)
        qr = make_job_pop_response("sdxl", width=1024, height=1024, workflow="qr_code")

        plain_mb = resource_budget.predict_job_sampling_vram_mb(plain, baseline)
        qr_mb = resource_budget.predict_job_sampling_vram_mb(qr, baseline)

        assert plain_mb is not None and qr_mb is not None
        assert qr_mb > plain_mb


class TestPerModelBurdenOverrides:
    """The job's model name reaches hordelib's burden lookup, and an older hordelib still answers."""

    _WEIGHTS_MB = 12600.0
    _FOOTPRINT_MB = 17900.0

    class _StubBurden:
        """A burden entry with the two figures the weight and footprint predictions read."""

        vram_per_megapixel_mb = 1500.0
        min_recommended_vram_mb = 14000.0

        def resident_weight_estimate_mb(self) -> float:
            return TestPerModelBurdenOverrides._WEIGHTS_MB

        def resident_footprint_estimate_mb(self) -> float:
            return TestPerModelBurdenOverrides._FOOTPRINT_MB

    def _install_recorder(self, monkeypatch: pytest.MonkeyPatch, *, accepts_model_name: bool) -> list[tuple]:
        """Replace hordelib's lookup with a recorder, optionally one that predates ``model_name``."""
        import hordelib.feature_impact as feature_impact

        calls: list[tuple] = []

        def recorder(baseline: str, model_name: str | None = None) -> object:
            calls.append((baseline, model_name))
            if not accepts_model_name and model_name is not None:
                raise TypeError("get_baseline_burden() got an unexpected keyword argument 'model_name'")
            return TestPerModelBurdenOverrides._StubBurden()

        monkeypatch.setattr(feature_impact, "get_baseline_burden", recorder)
        return calls

    def test_weight_prediction_passes_the_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A family member with its own weight set is charged by name, not by its family's seed."""
        calls = self._install_recorder(monkeypatch, accepts_model_name=True)
        job = make_job_pop_response("Krea2-Turbo_fp8")

        assert resource_budget.predict_job_weight_mb(job, "qwen_image") == self._WEIGHTS_MB
        assert calls == [("qwen_image", "Krea2-Turbo_fp8")]

    def test_footprint_prediction_passes_the_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The full-footprint prediction reads the same per-model entry."""
        calls = self._install_recorder(monkeypatch, accepts_model_name=True)
        job = make_job_pop_response("Krea2-Turbo_fp8")

        assert resource_budget.predict_job_footprint_mb(job, "qwen_image") == self._FOOTPRINT_MB
        assert calls == [("qwen_image", "Krea2-Turbo_fp8")]

    def test_an_older_hordelib_degrades_to_the_baseline_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hordelib predating per-model overrides still yields a figure, from the family seed."""
        calls = self._install_recorder(monkeypatch, accepts_model_name=False)
        job = make_job_pop_response("Krea2-Turbo_fp8")

        assert resource_budget.predict_job_weight_mb(job, "qwen_image") == self._WEIGHTS_MB
        assert calls == [("qwen_image", "Krea2-Turbo_fp8"), ("qwen_image", None)]
