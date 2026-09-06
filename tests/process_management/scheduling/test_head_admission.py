"""The head-of-queue admission ledger, stated directly: every clock is bounded and every step is reproducible.

The scheduler's own tests cover the log lines, ledger events and actuations these steps drive; these rows pin the
state transitions themselves, so a change to a bound or a reset rule is caught where it lives.
"""

from __future__ import annotations

from horde_worker_regen.process_management.scheduling.governance.preload_admission import AdmissionDecision
from horde_worker_regen.process_management.scheduling.ledgers.head_admission import (
    DISPATCH_STALL_LOG_INTERVAL_SECONDS,
    HEAD_RAM_DEFER_BARRIER_CAP_SECONDS,
    HEAD_RAM_DEFER_BARRIER_SECONDS,
    SAFETY_RECOVERY_HOLD_TTL_SECONDS,
    STAGING_DEFER_REPEAT_SECONDS,
    HeadAdmissionLedger,
    HeadRamDeferStep,
    SafetyRecoveryHoldStep,
    StagingDeferReason,
    format_staging_defer_tally,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _ledger() -> tuple[HeadAdmissionLedger, _Clock]:
    clock = _Clock()
    return HeadAdmissionLedger(clock), clock


class TestHeadStarvation:
    """The starvation clock times one head on an idle device and nothing else."""

    def test_times_the_head_only_while_the_device_is_idle(self) -> None:
        """Live work zeroes the clock; an idle device starts it at the head's first observation."""
        ledger, clock = _ledger()
        ledger.track_head_starvation("head", work_in_progress=True)
        assert ledger.starved_seconds("head") == 0.0
        ledger.track_head_starvation("head", work_in_progress=False)
        clock.now += 30.0
        assert ledger.starved_seconds("head") == 30.0
        assert ledger.starved_seconds("other") == 0.0
        assert ledger.starved_seconds(None) == 0.0

    def test_a_new_head_restarts_the_clock(self) -> None:
        """The clock measures this head's wait, not the queue's age."""
        ledger, clock = _ledger()
        ledger.track_head_starvation("first", work_in_progress=False)
        clock.now += 30.0
        ledger.track_head_starvation("second", work_in_progress=False)
        assert ledger.starved_seconds("first") == 0.0
        assert ledger.starved_seconds("second") == 0.0
        clock.now += 5.0
        assert ledger.starved_seconds("second") == 5.0

    def test_clear_and_no_head(self) -> None:
        """A dispatch or an empty queue stops the clock outright."""
        ledger, clock = _ledger()
        ledger.track_head_starvation("head", work_in_progress=False)
        ledger.clear_head_starvation()
        assert ledger.starvation_job_id is None and ledger.starvation_since == 0.0
        ledger.track_head_starvation("head", work_in_progress=False)
        ledger.track_head_starvation(None, work_in_progress=False)
        assert ledger.starvation_job_id is None


class TestRamDeferBarrier:
    """The RAM-defer clock latches the barrier past its bound and declines the head past the cap."""

    def test_first_defer_starts_the_clock_without_a_step(self) -> None:
        """A head seen deferring for the first time is ordinarily queued."""
        ledger, _ = _ledger()
        step = ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        assert step is HeadRamDeferStep.NONE
        assert ledger.ram_defer_job_id == "head"

    def test_engages_past_the_bound_unless_reclaim_grace_holds(self) -> None:
        """Past the bound with nothing freed the barrier engages; a reclaim cycle in its grace suppresses it."""
        ledger, clock = _ledger()
        ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        clock.now += HEAD_RAM_DEFER_BARRIER_SECONDS - 1.0
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
            is HeadRamDeferStep.NONE
        )
        clock.now += 1.0
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=True)
            is HeadRamDeferStep.NONE
        )
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
            is HeadRamDeferStep.ENGAGE_BARRIER
        )

    def test_progress_restarts_the_clock_and_releases_a_held_barrier(self) -> None:
        """Reclaim progress means the head is not starving; a barrier held for it is released."""
        ledger, clock = _ledger()
        ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        clock.now += HEAD_RAM_DEFER_BARRIER_SECONDS
        assert ledger.engage_barrier("head") is True
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=True, reclaim_grace_active=False)
            is HeadRamDeferStep.RELEASE_BARRIER
        )
        assert ledger.ram_defer_since == clock.now
        assert ledger.release_barrier() == "head", "the release itself is the scheduler's actuation"
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=True, reclaim_grace_active=False)
            is HeadRamDeferStep.NONE
        ), "progress with no barrier held asks for nothing"

    def test_a_new_head_restarts_the_clock(self) -> None:
        """A different head at the front is a fresh defer episode."""
        ledger, clock = _ledger()
        ledger.govern_ram_defer("first", made_reclaim_progress=False, reclaim_grace_active=False)
        clock.now += HEAD_RAM_DEFER_BARRIER_SECONDS * 2
        assert (
            ledger.govern_ram_defer("second", made_reclaim_progress=False, reclaim_grace_active=False)
            is HeadRamDeferStep.NONE
        )
        assert ledger.ram_defer_since == clock.now

    def test_declines_past_the_cap(self) -> None:
        """A barrier that has held past its cap without admitting the head declines it."""
        ledger, clock = _ledger()
        ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        ledger.engage_barrier("head")
        clock.now += HEAD_RAM_DEFER_BARRIER_CAP_SECONDS - 1.0
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
            is HeadRamDeferStep.ENGAGE_BARRIER
        )
        clock.now += 1.0
        assert (
            ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
            is HeadRamDeferStep.DECLINE_HEAD
        )

    def test_barrier_withholds_every_job_but_the_head(self) -> None:
        """While held, only the barred head may dispatch; engaging twice is a no-op; release reports the head."""
        ledger, _ = _ledger()
        assert ledger.barrier_withholds("sibling") is False
        assert ledger.engage_barrier("head") is True
        assert ledger.engage_barrier("head") is False
        assert ledger.barrier_withholds("sibling") is True
        assert ledger.barrier_withholds("head") is False
        assert ledger.release_barrier() == "head"
        assert ledger.release_barrier() is None
        assert ledger.barrier_withholds("sibling") is False

    def test_resolve_and_reconcile(self) -> None:
        """An admitted head clears its clock and reports its barrier; a departed head releases its barrier."""
        ledger, _ = _ledger()
        ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        ledger.engage_barrier("head")
        assert ledger.reconcile_to_head("head") is False
        assert ledger.resolve_ram_defer("other") is False
        assert ledger.ram_defer_job_id == "head"
        assert ledger.resolve_ram_defer("head") is True
        assert ledger.ram_defer_job_id is None
        ledger.govern_ram_defer("head", made_reclaim_progress=False, reclaim_grace_active=False)
        assert ledger.reconcile_to_head("next") is True
        assert ledger.ram_defer_job_id is None


class TestSafetyRecoveryHold:
    """The hold engages once, holds to its TTL, expires with a latch, and releases when the pool recovers."""

    def test_engage_hold_expire_and_latch(self) -> None:
        """One query engages, later ones hold, the TTL expires and latches the episode."""
        ledger, clock = _ledger()
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=True) is SafetyRecoveryHoldStep.ENGAGED
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=True) is SafetyRecoveryHoldStep.HOLDING
        clock.now += SAFETY_RECOVERY_HOLD_TTL_SECONDS
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=True) is SafetyRecoveryHoldStep.EXPIRED
        assert ledger.recovery_hold_since == 0.0 and ledger.recovery_hold_expired is True
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=True) is SafetyRecoveryHoldStep.NOT_HOLDING

    def test_recovery_releases_and_clears_the_latch(self) -> None:
        """The pool recovering releases an engaged hold; the next episode may hold again."""
        ledger, clock = _ledger()
        ledger.recovery_hold_step(crash_looping=True, targets_card=True)
        assert ledger.recovery_hold_step(crash_looping=False, targets_card=True) is SafetyRecoveryHoldStep.RELEASED
        assert ledger.recovery_hold_step(crash_looping=False, targets_card=True) is SafetyRecoveryHoldStep.NOT_HOLDING
        ledger.recovery_hold_step(crash_looping=True, targets_card=True)
        clock.now += SAFETY_RECOVERY_HOLD_TTL_SECONDS
        ledger.recovery_hold_step(crash_looping=True, targets_card=True)
        assert ledger.recovery_hold_step(crash_looping=False, targets_card=True) is SafetyRecoveryHoldStep.NOT_HOLDING
        assert ledger.recovery_hold_expired is False
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=True) is SafetyRecoveryHoldStep.ENGAGED

    def test_another_card_neither_engages_nor_releases(self) -> None:
        """A query for a card that is not waiting on the safety start leaves the hold as it is."""
        ledger, _ = _ledger()
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=False) is SafetyRecoveryHoldStep.NOT_HOLDING
        assert ledger.recovery_hold_since == 0.0
        ledger.recovery_hold_step(crash_looping=True, targets_card=True)
        assert ledger.recovery_hold_step(crash_looping=True, targets_card=False) is SafetyRecoveryHoldStep.NOT_HOLDING
        assert ledger.recovery_hold_since != 0.0


class TestRecordsAndThrottles:
    """Admission records, staging deferrals, the missing-model latch, the stall diagnostic and the heavy head."""

    def test_preload_admission_record_and_overlay(self) -> None:
        """The last decision is kept with the clock's stamp; the overlay reads worker-wide under key 0."""
        ledger, clock = _ledger()
        ledger.record_preload_admission(AdmissionDecision.ADMIT, model="m", process_id=3, reason="fits")
        record = ledger.last_preload_admission
        assert record is not None
        assert (record.decision, record.model, record.process_id, record.reason, record.timestamp) == (
            AdmissionDecision.ADMIT,
            "m",
            3,
            "fits",
            clock.now,
        )
        assert ledger.admission_headroom_mb(None) is None
        ledger.note_admission_headroom(None, 512.0)
        assert ledger.admission_headroom_mb(0) == 512.0
        assert ledger.admission_denials(None) == 0

    def test_staging_defers_tally_and_throttle(self) -> None:
        """Every deferral is tallied; an unchanged reason is restated only after the repeat window."""
        ledger, clock = _ledger()
        assert ledger.note_staging_defer(StagingDeferReason.MEASUREMENT_UNREAD) == 0
        assert ledger.note_staging_defer(StagingDeferReason.MEASUREMENT_UNREAD) is None
        assert ledger.note_staging_defer(StagingDeferReason.ENCODE_HEADROOM_SHORT) == 1
        clock.now += STAGING_DEFER_REPEAT_SECONDS
        assert ledger.note_staging_defer(StagingDeferReason.ENCODE_HEADROOM_SHORT) == 0
        assert ledger.staging_defers == {
            StagingDeferReason.MEASUREMENT_UNREAD: 2,
            StagingDeferReason.ENCODE_HEADROOM_SHORT: 2,
        }
        assert format_staging_defer_tally(ledger.staging_defers) == "staging deferred: 4 (headroom 50%, unread 50%)"

    def test_missing_model_latch_expires_on_the_budget(self) -> None:
        """The latch holds inside the preload budget, expires past it, and a dispatch clears it early."""
        ledger, clock = _ledger()
        assert ledger.missing_model_latched(60.0) is False
        ledger.latch_missing_model()
        clock.now += 59.0
        assert ledger.missing_model_latched(60.0) is True
        clock.now += 1.0
        assert ledger.missing_model_latched(60.0) is False
        ledger.latch_missing_model()
        ledger.clear_missing_model()
        assert ledger.missing_model_latched(60.0) is False

    def test_stall_diagnostic_throttles_on_the_injected_clock(self) -> None:
        """A changed reason logs at once; an unchanged one waits out the interval on the ledger's clock."""
        ledger, clock = _ledger()
        assert ledger.note_stall("cap") is True
        assert ledger.note_stall("cap") is False
        assert ledger.stall_reason == "cap"
        assert ledger.note_stall("overlap") is True
        clock.now += DISPATCH_STALL_LOG_INTERVAL_SECONDS - 1.0
        assert ledger.note_stall("overlap") is False
        clock.now += 1.0
        assert ledger.note_stall("overlap") is True
        ledger.clear_stall()
        assert ledger.stall_reason is None

    def test_heavy_head_grace_is_bounded(self) -> None:
        """No admission means no grace; an admission opens it for exactly the window."""
        ledger, clock = _ledger()
        assert ledger.heavy_head_load_grace_active(30.0) is False
        ledger.note_heavy_head_admitted()
        clock.now += 29.0
        assert ledger.heavy_head_load_grace_active(30.0) is True
        clock.now += 1.0
        assert ledger.heavy_head_load_grace_active(30.0) is False


class TestMeasuredFloorDenials:
    """The run-metrics denial figure counts measured refusals of statically admitted candidates, per card."""

    def test_denials_count_per_card_and_read_zero_elsewhere(self) -> None:
        """Each note adds one under the card's key, the worker-wide view under key 0, and an unseen card reads 0."""
        ledger, _clock = _ledger()
        assert ledger.admission_denials(None) == 0
        ledger.note_measured_floor_denial(None)
        ledger.note_measured_floor_denial(1)
        ledger.note_measured_floor_denial(1)
        assert ledger.admission_denials(None) == 1
        assert ledger.admission_denials(0) == 1
        assert ledger.admission_denials(1) == 2
        assert ledger.admission_denials(2) == 0
