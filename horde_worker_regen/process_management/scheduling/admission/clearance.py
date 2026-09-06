"""The clearance admit decision: whether a staged child may enter its load-and-sample window.

Under the clearance lease a dispatch only stages the job (checkpoint disk load, prompt encode), so clearance is
the VRAM moment: the child's full materialisation is priced here through the shared MONOLITHIC_DISPATCH
identity. The decision is built from the snapshot and the frozen arbiter; the scheduler executes it (the hold
records, the reservation upgrade, the eviction the verdict describes, the log line).
"""

from __future__ import annotations

from dataclasses import dataclass

from strenum import StrEnum

from horde_worker_regen.process_management.resources.vram_arbiter import ActuatorCommand, VramArbiter, VramVerdict
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.admission.materialization import (
    MaterializationRequest,
    build_materialization_request,
)
from horde_worker_regen.process_management.scheduling.admission.snapshot import SchedulingSnapshot


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
    """The full materialisation fits: grant and re-book the dispatch reservation at the full peak."""
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


def staged_materialization_delta_mb(snapshot: SchedulingSnapshot, job_id: str, process_id: int) -> float | None:
    """The staged child's remaining materialisation (MB): its priced peak net of what it already holds on the card.

    By the clearance moment the child's text encoder, VAE and leftover allocator cache are on the device and
    already missing from the measured device-free reading, while the priced peak is the whole sampling-time
    reservation including them; charged gross, a child reads as further from fitting the more of its own job
    it has staged. The reserve ledger decays the staging reservation by the same measured growth. An
    unpriceable job (None) and a resident-weight candidate (already credited) pass through.
    """
    job = snapshot.queue.jobs[job_id]
    if job.model is None:
        return None
    gross_mb = pricing.candidate_delta_mb(
        snapshot,
        snapshot.queue.payloads[job_id],
        job.baseline,
        process_id=process_id,
        disaggregated=job.disaggregation_class_eligible,
    )
    if gross_mb is None:
        return None
    if pricing.candidate_weights_resident(snapshot, job.model, process_id):
        return gross_mb
    staged_mb = snapshot.slots[process_id].reserved_mb
    if staged_mb is None:
        return gross_mb
    return max(0.0, gross_mb - float(staged_mb))


def decide_clearance_admit(
    snapshot: SchedulingSnapshot,
    process_id: int,
    *,
    arbiter: VramArbiter,
    post_processing_deferred: bool,
) -> ClearancePlan:
    """Decide whether the staged child on ``process_id`` may be cleared into its load-and-sample window.

    ``post_processing_deferred`` is the co-residency mutex's answer for the child's job, read by the caller
    because that predicate still logs and latches on the scheduler; it is consulted only for a priceable job.
    A priced decision nets the job's own outstanding staging reservation out of the overlay, since the full
    peak priced here already covers it, and evaluates against the one frozen measurement the snapshot saw.
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
    priced = build_materialization_request(
        snapshot,
        job_id,
        process_id,
        arbiter=arbiter,
        is_head_of_queue=True,
        head_outstanding_mb=None,
        candidate_delta_override_mb=staged_materialization_delta_mb(snapshot, job_id, process_id),
        nets_own_dispatch_reservation=True,
    )
    verdict = arbiter.evaluate(priced.request)
    decision = ClearanceDecision.ADMIT if verdict.admits else ClearanceDecision.HOLD
    return ClearancePlan(decision, process_id, job_id, job.model, priced, verdict)
