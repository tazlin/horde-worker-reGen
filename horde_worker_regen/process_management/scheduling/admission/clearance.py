"""The clearance admit decision: whether a staged child may enter its load-and-sample window.

Under the clearance lease a dispatch only stages the job (checkpoint disk load, prompt encode), so clearance is
the VRAM moment: the child's full materialisation is priced here through the shared MONOLITHIC_DISPATCH
identity. The decision is built from the snapshot and the frozen arbiter; the scheduler executes it (the hold
records, the reservation upgrade, the eviction the verdict describes, the log line).
"""

from __future__ import annotations

from dataclasses import dataclass

from strenum import StrEnum

from horde_worker_regen.process_management.resources.reclaim_ladder import teardown_verification_settle_seconds
from horde_worker_regen.process_management.resources.resource_budget import (
    predict_job_weight_mb,
    predict_model_weight_mb,
)
from horde_worker_regen.process_management.resources.vram_arbiter import (
    ActuatorCommand,
    LaneRungGrade,
    VramArbiter,
    VramVerdict,
)
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.admission.materialization import (
    MaterializationRequest,
    StagedWaiterTerms,
    build_materialization_request,
)
from horde_worker_regen.process_management.scheduling.admission.snapshot import SchedulingSnapshot
from horde_worker_regen.process_management.scheduling.clearance_lease import CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
from horde_worker_regen.process_management.scheduling.workload_flow import DISPATCH_ADMISSION_FLOW


class ClearanceDecision(StrEnum):
    """How the clearance gate answers for one staged child."""

    NO_PROCESS = "no_process"
    """The process id names no slot: withhold, there is nothing to clear."""
    UNPRICED = "unpriced"
    """The slot references no job, or one with no model: grant rather than wedge the child on nothing priceable."""
    HOLD_POST_PROCESSING = "post_processing_coresidency"
    """The co-residency mutex holds the card for an in-flight or pending post-processing chain."""
    BUDGET_INACTIVE = "budget_inactive"
    """The VRAM budget is off: grant without pricing."""
    ADMIT = "admit"
    """The full materialisation or the job's partial-load seat fits: grant and re-book the dispatch reservation at
    what is left to materialise, capped at the granted room for a seat."""
    HOLD = "hold"
    """The materialisation does not fit yet: withhold, run the verdict's evictions, and re-ask next pass."""


@dataclass(frozen=True)
class ClearancePlan:
    """The clearance verdict for one staged child and the figures its execution and diagnostics need."""

    decision: ClearanceDecision
    process_id: int
    job_id: str | None
    model: str | None
    priced: MaterializationRequest | None
    """The priced request when the decision went through the arbiter, else None."""
    verdict: VramVerdict | None

    @property
    def grants(self) -> bool:
        """Whether the child enters its load-and-sample window."""
        return self.decision in (
            ClearanceDecision.UNPRICED,
            ClearanceDecision.BUDGET_INACTIVE,
            ClearanceDecision.ADMIT,
        )

    @property
    def reason(self) -> str:
        """The hold cause the diagnostics coalesce on: the verdict's disposition when one was taken."""
        if self.verdict is not None:
            return self.verdict.disposition.value
        return self.decision.value

    @property
    def actuations(self) -> tuple[ActuatorCommand, ...]:
        """The evictions a held verdict describes, for the reclaim owner to run."""
        if self.decision is not ClearanceDecision.HOLD or self.verdict is None:
            return ()
        return self.verdict.required_actuations

    @property
    def candidate_delta_mb(self) -> float | None:
        """The staged child's remaining materialisation charge, when priced."""
        return self.priced.candidate_delta_mb if self.priced is not None else None

    @property
    def booking_mb(self) -> float | None:
        """The dispatch reservation a grant books: the remaining charge, capped at the room a partial seat grants.

        A child admitted at its partial-load seat loads into the free room its grant carries and no further, so
        a booking above that room would stay outstanding through its whole sample.
        """
        remaining_mb = self.candidate_delta_mb
        if remaining_mb is None or self.verdict is None or not self.verdict.partial_seat:
            return remaining_mb
        available_mb = self.verdict.measured.available_mb
        if available_mb is None:
            return remaining_mb
        return min(remaining_mb, max(0.0, available_mb))


@dataclass(frozen=True)
class StagedWaiterClock:
    """Represents the caller's clearance clock for one staged waiter and the load it would pay once cleared."""

    starved_seconds: float
    """Seconds clearance has held the waiter while waiting could not help: no other grant on its card and no
    reclaim in flight. Zero while either holds."""
    load_seconds: float | None
    """The card's measured RAM-to-VRAM load seconds, or None when none is measured yet."""
    lane_rung_grade: LaneRungGrade = LaneRungGrade.NO_RUNG
    """Where the waiter's starvation episode stands with the service-lane rungs applied for it."""
    seconds_per_step: float | None = None
    """The performance model's expected seconds per sampling step for the waiter's job, or None when unknown."""
    upload_mb_per_second: float | None = None
    """The rate (MB/s) the card has measured jobs putting their weights on it, or None when none is measured."""
    lease_wait_seconds: float | None = None
    """Seconds since clearance first held the waiter. A reclaim leaves it running, since the child's
    lease-acquire timeout counts from the start of its wait. None for a waiter the caller does not time."""


@dataclass
class StarvedLaneRungEpisode:
    """The service-lane rung last applied for one staged waiter, graded as the verified ladder grades a teardown.

    A rung is graded once its lane's processes have exited and the card has been read after that exit. The
    evaluation that first sees the exit priced a reading the snapshot may have frozen before it, so the grade
    lands on the evaluation after. A lane that never exits is graded at the teardown verification bound on
    whatever the card then reports, so a lane that never answers cannot hold the head.
    """

    applied_at: float
    """The scheduler-clock instant the rung was applied."""
    lane_launches: frozenset[tuple[int, int]]
    """The (process id, launch identifier) of every process of the paused lane when the rung was applied."""
    exit_observed: bool = False
    """Whether an evaluation has seen every one of those launches gone."""
    graded: bool = False
    """Whether the rung is graded. It stays graded for the rest of the episode."""
    graded_at_bound: bool = False
    """Whether the grade came from the bound with the lane still present."""

    def grade(self, *, now: float, lane_exited: bool) -> LaneRungGrade:
        """Advance the rung's grade by one evaluation and return it.

        Args:
            now: The scheduler-clock instant of this evaluation.
            lane_exited: Whether none of :attr:`lane_launches` is still active in the process map.
        """
        if self.graded:
            return LaneRungGrade.GRADED
        if self.exit_observed:
            self.graded = True
            return LaneRungGrade.GRADED
        if now - self.applied_at >= teardown_verification_settle_seconds():
            self.graded = True
            self.graded_at_bound = not lane_exited
            return LaneRungGrade.GRADED
        if lane_exited:
            self.exit_observed = True
        return LaneRungGrade.AWAITING_GRADE


_UNTIMED_WAITER = StagedWaiterClock(starved_seconds=0.0, load_seconds=None)
"""The clock of a waiter the caller does not time: no starved seconds, so no measured-load probe."""


def staged_attempt_deadline_seconds(*, probe_after_seconds: float, load_seconds: float | None) -> float:
    """Return the clock reading by which a staged waiter's measured-load probe must be eligible.

    The acquire timeout less the longer of the card's probe delay and its measured load: the grant then reaches
    the child while it still waits, with at least a probe delay or a load's worth of the timeout to spare for
    the gap between the child starting its wait and clearance first holding it.
    """
    margin_seconds = max(probe_after_seconds, load_seconds or 0.0)
    return max(0.0, CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS - margin_seconds)


def _staged_peak_and_holdings_mb(
    snapshot: SchedulingSnapshot,
    job_id: str,
    process_id: int,
) -> tuple[float, float] | None:
    """The staged job's whole peak and what of it its slot reports holding (MB), or None when unpriceable.

    The report is the job's own only net of weights another model keeps on the slot, which stay beside the job.
    Weights of the job's own model, whole or partial, are in the report too, so the peak carries them before
    any resident-weight credit rather than having them credited whole. A slot with no report holds nothing.
    """
    job = snapshot.queue.jobs[job_id]
    if job.model is None:
        return None
    payload = snapshot.queue.payloads[job_id]
    priced_mb = pricing.candidate_delta_mb(
        snapshot,
        payload,
        job.baseline,
        process_id=process_id,
        disaggregated=job.disaggregation_class_eligible,
    )
    if priced_mb is None:
        return None
    reported_mb = snapshot.slots[process_id].reserved_mb
    if reported_mb is None:
        return priced_mb, 0.0
    peak_mb = priced_mb
    if pricing.candidate_weights_resident(snapshot, job.model, process_id):
        peak_mb += predict_job_weight_mb(payload, job.baseline) or 0.0
    own_holdings_mb = max(0.0, float(reported_mb) - _other_models_weights_mb(snapshot, process_id, job.model))
    return peak_mb, min(peak_mb, own_holdings_mb)


def staged_materialization_delta_mb(snapshot: SchedulingSnapshot, job_id: str, process_id: int) -> float | None:
    """The staged child's remaining materialisation (MB): its priced peak net of what it already holds on the card.

    By the clearance moment the child's text encoder, VAE and leftover allocator cache are on the device and
    already missing from the measured device-free reading, while the priced peak is the whole sampling-time
    reservation including them; charged gross, a child reads as further from fitting the more of its own job
    it has staged. Clearance prices this figure and the cleared job's dispatch reservation is booked at it
    against the same report, so the booking has decayed to nothing once the job reaches its peak.

    An unpriceable job (None) passes through, and a slot with no report yet is charged its priced peak.
    """
    peak_and_holdings = _staged_peak_and_holdings_mb(snapshot, job_id, process_id)
    if peak_and_holdings is None:
        return None
    peak_mb, held_mb = peak_and_holdings
    return peak_mb - held_mb


def staged_held_mb(snapshot: SchedulingSnapshot, job_id: str, process_id: int) -> float:
    """What the staged job already holds on the card (MB): the part of its peak its outstanding charge nets out.

    The outstanding charge plus this figure is the job's whole need, which is what an emptied card must seat.
    """
    peak_and_holdings = _staged_peak_and_holdings_mb(snapshot, job_id, process_id)
    return peak_and_holdings[1] if peak_and_holdings is not None else 0.0


def sibling_staging_relief_mb(snapshot: SchedulingSnapshot, job_id: str, process_id: int) -> float:
    """The part (MB) of other staged waiters' staging charges on the card that pricing ``job_id`` nets out.

    A staged waiter not yet cleared on the same card is charged only what its encode can still allocate. One
    that has entered its clearance wait has finished its encode, so everything it holds is already inside the
    device-free reading and its whole outstanding charge is netted. One still encoding stays charged the
    staging charge less what it holds, never more than its outstanding charge. A cleared waiter is about to
    load its weights and keeps its charge, as does a slot past its staging.
    """
    slot = snapshot.slots[process_id]
    device_index = snapshot.routing_device_index(slot)
    reserved_by_pid = dict(snapshot.card(device_index).reserved_by_pid)
    ledger = snapshot.services.reserve_ledger
    granted = snapshot.ledgers.dispatch_holds.clearance_granted_ids
    relief_mb = 0.0
    for sibling in snapshot.slots.values():
        sibling_job_id = sibling.current_job_id
        if (
            sibling.process_id == process_id
            or sibling_job_id is None
            or sibling_job_id == job_id
            or sibling_job_id in granted
            or not sibling.staged_short_of_sampling
            or sibling_job_id not in snapshot.queue.jobs
            or snapshot.routing_device_index(sibling) != device_index
        ):
            continue
        outstanding_mb = ledger.planned_charge_for_unit(DISPATCH_ADMISSION_FLOW, sibling_job_id, reserved_by_pid)
        if sibling.clearance_wait_entered:
            still_to_allocate_mb = 0.0
        else:
            held_mb = staged_held_mb(snapshot, sibling_job_id, sibling.process_id)
            still_to_allocate_mb = max(0.0, pricing.STAGING_ENCODE_VRAM_MB - held_mb)
        relief_mb += max(0.0, outstanding_mb - still_to_allocate_mb)
    return relief_mb


def _other_models_weights_mb(snapshot: SchedulingSnapshot, process_id: int, model: str) -> float:
    """The predicted weights (MB) of every model other than ``model`` whose weights the slot holds on the card."""
    metadata = snapshot.services.model_metadata
    return sum(
        predict_model_weight_mb(resident_model, metadata.get_baseline(resident_model)) or 0.0
        for resident_model in snapshot.slots[process_id].resident_weight_models
        if resident_model != model
    )


def decide_clearance_admit(
    snapshot: SchedulingSnapshot,
    process_id: int,
    *,
    arbiter: VramArbiter,
    post_processing_deferred: bool,
    waiter_clock: StagedWaiterClock = _UNTIMED_WAITER,
) -> ClearancePlan:
    """Decide whether the staged child on ``process_id`` may be cleared into its load-and-sample window.

    ``post_processing_deferred`` is the co-residency mutex's answer for the child's job, read by the caller
    because that predicate still logs and latches on the scheduler; it is consulted only for a priceable job.
    ``waiter_clock`` is the caller's clearance clock for the child, which times its measured-load probe in
    place of the head-starvation clock. A priced decision nets the job's own outstanding staging reservation
    out of the overlay, since the full peak priced here already covers it, nets the part of other staged
    waiters' staging charges they will not allocate (:func:`sibling_staging_relief_mb`), presents what the job
    already holds so the ceiling test sees its whole need, and evaluates against the one frozen measurement the
    snapshot saw.
    """
    slot = snapshot.slots.get(process_id)
    if slot is None:
        return ClearancePlan(ClearanceDecision.NO_PROCESS, process_id, None, None, None, None)
    job_id = slot.current_job_id
    job = snapshot.queue.jobs.get(job_id) if job_id is not None else None
    if job_id is None or job is None or job.model is None:
        return ClearancePlan(ClearanceDecision.UNPRICED, process_id, job_id, None, None, None)
    if post_processing_deferred:
        return ClearancePlan(ClearanceDecision.HOLD_POST_PROCESSING, process_id, job_id, job.model, None, None)
    if not snapshot.budget_active:
        return ClearancePlan(ClearanceDecision.BUDGET_INACTIVE, process_id, job_id, job.model, None, None)
    config = snapshot.config_for(snapshot.routing_device_index(slot))
    priced = build_materialization_request(
        snapshot,
        job_id,
        process_id,
        arbiter=arbiter,
        is_head_of_queue=True,
        head_outstanding_mb=None,
        candidate_delta_override_mb=staged_materialization_delta_mb(snapshot, job_id, process_id),
        staged_waiter=StagedWaiterTerms(
            held_mb=staged_held_mb(snapshot, job_id, process_id),
            starved_seconds=waiter_clock.starved_seconds,
            lease_wait_seconds=waiter_clock.lease_wait_seconds,
            attempt_deadline_seconds=staged_attempt_deadline_seconds(
                probe_after_seconds=float(config.measured_load_probe_seconds),
                load_seconds=waiter_clock.load_seconds,
            ),
            lane_rung_grade=waiter_clock.lane_rung_grade,
            seat_weight_fraction=pricing.partial_seat_weight_fraction(
                weights_mb=predict_job_weight_mb(snapshot.queue.payloads[job_id], job.baseline),
                seconds_per_step=waiter_clock.seconds_per_step,
                upload_mb_per_second=waiter_clock.upload_mb_per_second,
            ),
        ),
        nets_own_dispatch_reservation=True,
        sibling_staging_relief_mb=sibling_staging_relief_mb(snapshot, job_id, process_id),
    )
    verdict = arbiter.evaluate(priced.request)
    decision = ClearanceDecision.ADMIT if verdict.admits else ClearanceDecision.HOLD
    return ClearancePlan(decision, process_id, job_id, job.model, priced, verdict)
