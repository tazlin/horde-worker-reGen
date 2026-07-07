"""The ledger-driven VRAM admission identity: measured floor plus planned charges against real capacity.

On Windows/WDDM the driver never fails an allocation at the physical ceiling: an over-commit is silently
demoted to the system-backed shared segment, and both ``mem_get_info`` and core-utilization telemetry keep
reading healthy. The measured *free*-VRAM figure therefore lies precisely when it matters most, so it cannot
be the thing an admission gate reasons against. The only defense that survives the lie is the ledger
arithmetic: the exact device memory the worker has itself committed (per-process CUDA context plus each
process's byte-exact allocator reservation), plus the charges it has admitted but not yet materialised,
compared against the device's real capacity net of the shared baseline it cannot attribute to itself.

This module defines that admission identity once as a pure function so both the residency/preload gate and
its tests reason about the same inequality:

    measured_committed_fresh + planned_unmaterialized + candidate_delta <= (total - baseline) - noise_buffer

- ``measured_committed_fresh``: the worker's committed-VRAM ledger sum (context + allocator-reserved per live
  GPU process, baseline-exclusive), used only while every contributor's report is fresh.
- ``planned_unmaterialized``: VRAM admitted for in-flight work whose allocation the measured floor does not
  yet reflect, decaying per target as that target's measured reservation materialises (the ledger's
  double-count guard).
- ``candidate_delta``: the candidate job's predicted marginal cost, net of any weights already resident in
  the target process (which the measured floor already counts and must not be re-charged).
- ``(total - baseline) - noise_buffer``: the real admission capacity. ``baseline`` is the reconciler's
  measured shared-device baseline (OS/desktop/other apps); ``noise_buffer`` is slack for measurement/rounding
  noise and inter-report activation transients, scaling with device capacity above a floor (see
  :func:`admission_noise_buffer_mb`), NOT the operator's ``vram_reserve_mb``. That configured reserve is the
  *sampling* gate's activation margin and is never a load-feasibility floor (see the reserve-decoupling
  contract in :mod:`~horde_worker_regen.process_management.resources.resource_budget`); folding it in here
  would repeat the wedge that decoupling exists to prevent, where a model whose weights fit the drained card
  reads as unloadable.

Degraded mode is deliberate and safe. Staleness invalidates only the measured committed floor (child
telemetry), never the planned overlay: that overlay is the parent's own admission ledger (its
``CommittedReserveLedger`` anchors) and needs no child report, so stacked unmaterialized admissions remain
knowable arithmetic at admission time. When the committed ledger is stale but a device total is known, the
identity drops the measured floor and tests ``planned + candidate <= capacity`` with
``used_measured_floor=False``: a request can therefore be denied on stacked unmaterialized admissions even
before the first child memory report, while staleness alone (no planned demand, a candidate within capacity)
never denies. Only a cold start with no known total relaxes fully to ``fits=True``, since with no capacity
nothing is knowable; the caller then falls back to its predictive path.
"""

from __future__ import annotations

from dataclasses import dataclass

_ADMISSION_NOISE_BUFFER_MB = 512.0
"""Floor (MB) of the admission noise buffer, the value it takes on small cards and when no total is known.

The admission ceiling is ``(total - baseline) - noise_buffer``: the device capacity net of the shared
baseline, less a buffer that absorbs ordinary measurement and rounding noise between the ledger sum and the
device's true committed figure, plus the inter-report activation transients a child's allocator briefly
holds before the next memory report reflects them. The buffer scales with device capacity (see
:func:`admission_noise_buffer_mb`) so a large card keeps proportional headroom against those transients
while a small card is never starved below this floor. This is intentionally NOT the operator's
``vram_reserve_mb``, which remains the sampling gate's per-step activation margin: the reserve is never a
load-feasibility floor (making it one is exactly the wedge the reserve-decoupling contract prevents). Sized
well below one model's weights so it never denies a load the card physically holds, but above the
sub-hundred-MB slack a rounded device figure introduces."""

_ADMISSION_NOISE_BUFFER_FRACTION = 0.05
"""Fraction of the device total VRAM the noise buffer scales to once that exceeds the floor.

Chosen so the margin tracks device capacity (a 24GB card affords roughly 1.2GB of transient headroom, an
8GB card cannot afford more than the floor) rather than pinning a flat constant tuned to one reference
card, per the fleet-heterogeneity contract."""


def admission_noise_buffer_mb(total_vram_mb: float | None) -> float:
    """Return the admission noise buffer (MB): the floor, or ``5%`` of the device total when that is larger.

    The buffer absorbs measurement noise and the activation transients a child's allocator holds between
    memory reports, and scales with device capacity so large cards keep proportional headroom while small
    cards are never starved below :data:`_ADMISSION_NOISE_BUFFER_MB`. An unknown or non-positive total (cold
    start) yields the floor, since no proportional term can be formed without a capacity to scale against.

    Args:
        total_vram_mb: The device's total VRAM (MB), or None when no total has been reported yet.
    """
    if total_vram_mb is None or total_vram_mb <= 0:
        return _ADMISSION_NOISE_BUFFER_MB
    return max(_ADMISSION_NOISE_BUFFER_MB, _ADMISSION_NOISE_BUFFER_FRACTION * total_vram_mb)


@dataclass(frozen=True)
class AdmissionVerdict:
    """The outcome of one evaluation of the admission identity, carrying every term for logging.

    ``used_measured_floor`` is False whenever the measured floor could not be applied. That splits two ways.
    On a cold start with no known total the verdict fully relaxes to ``fits=True`` and the caller degrades to
    its predictive path. On a stale committed ledger with a known total the measured floor is dropped but the
    planned overlay still counts, so ``fits`` is the authoritative result of ``planned + candidate <=
    capacity`` and may deny (a stacked-admission over-commit before the first child report). When
    ``used_measured_floor`` is True the verdict is the full inequality's result and ``fits`` is authoritative.
    """

    fits: bool
    """Whether the admission identity holds (demand within capacity), or True in the cold-start relaxation."""
    used_measured_floor: bool
    """True when the measured floor was applied (fresh ledger, known total); False when it was dropped."""
    committed_is_stale: bool
    """True when the measured floor was dropped for staleness (the degraded planned-only identity applies)."""
    measured_committed_mb: float
    """The worker's committed-VRAM ledger sum (MB) at evaluation time (context + reserved per live process)."""
    planned_unmaterialized_mb: float
    """VRAM (MB) admitted but not yet reflected in the measured floor, net of what has since materialised."""
    candidate_delta_mb: float
    """The candidate job's marginal predicted cost (MB), net of any weights already resident in its target."""
    total_vram_mb: float | None
    """Device total VRAM (MB), or None at cold start (relaxes the verdict)."""
    baseline_mb: float
    """The reconciler's measured shared-device baseline (MB) subtracted from total to form capacity."""
    noise_buffer_mb: float
    """The fixed noise slack (MB) subtracted on top of the baseline."""

    @property
    def capacity_mb(self) -> float | None:
        """The real admission capacity (MB): ``(total - baseline) - noise_buffer``, or None at cold start."""
        if self.total_vram_mb is None:
            return None
        return (self.total_vram_mb - self.baseline_mb) - self.noise_buffer_mb

    @property
    def demand_mb(self) -> float:
        """The demand (MB) tested against capacity.

        Committed-plus-planned-plus-candidate when the measured floor was applied; planned-plus-candidate on
        the degraded stale path, where the measured floor is dropped but the parent's planned overlay counts.
        """
        if self.used_measured_floor:
            return self.measured_committed_mb + self.planned_unmaterialized_mb + self.candidate_delta_mb
        return self.planned_unmaterialized_mb + self.candidate_delta_mb

    @property
    def headroom_mb(self) -> float | None:
        """Capacity minus demand (MB): positive when the identity holds, or None at cold start."""
        capacity = self.capacity_mb
        if capacity is None:
            return None
        return capacity - self.demand_mb

    def reason(self) -> str:
        """Return the full identity rendered for a log line, so a denial or unload is self-explaining."""
        if not self.used_measured_floor and self.total_vram_mb is None:
            return "no VRAM total yet (cold start); measured floor not applied, admitted on predictive path"
        verb = "fits" if self.fits else "does NOT fit"
        if not self.used_measured_floor:
            return (
                f"committed unavailable (stale, measured floor dropped); planned "
                f"{self.planned_unmaterialized_mb:.0f} + candidate {self.candidate_delta_mb:.0f} = "
                f"{self.demand_mb:.0f} MB vs capacity (total {self.total_vram_mb:.0f} - baseline "
                f"{self.baseline_mb:.0f} - noise {self.noise_buffer_mb:.0f}) = {self.capacity_mb:.0f} MB: {verb}"
            )
        return (
            f"committed {self.measured_committed_mb:.0f} + planned {self.planned_unmaterialized_mb:.0f} + "
            f"candidate {self.candidate_delta_mb:.0f} = {self.demand_mb:.0f} MB vs capacity "
            f"(total {self.total_vram_mb:.0f} - baseline {self.baseline_mb:.0f} - noise "
            f"{self.noise_buffer_mb:.0f}) = {self.capacity_mb:.0f} MB: {verb}"
        )


def evaluate_admission(
    *,
    measured_committed_mb: float,
    planned_unmaterialized_mb: float,
    candidate_delta_mb: float,
    total_vram_mb: float | None,
    baseline_mb: float,
    noise_buffer_mb: float | None = None,
    committed_is_stale: bool,
) -> AdmissionVerdict:
    """Evaluate the ledger-driven admission identity, relaxing to the predictive path when it cannot apply.

    Admits iff ``measured_committed + planned + candidate <= (total - baseline) - noise_buffer`` when the
    measured floor is trustworthy (a fresh committed ledger, a known total). A stale committed ledger (a
    contributor whose report has aged out, making the sum incomparable) drops only that measured floor: with a
    known total the identity still tests the degraded ``planned + candidate <= capacity`` and returns
    ``used_measured_floor=False`` with an authoritative ``fits`` that may deny a stacked-admission over-commit.
    The planned overlay is the parent's own admission ledger and needs no child report, so staleness alone (no
    planned demand, a candidate within capacity) never denies. Only an unknown/non-positive total (cold start)
    relaxes fully to ``fits=True`` so the caller falls back to its predictive gate.

    The noise buffer absorbs measurement noise and the inter-report activation transients a child's allocator
    holds before the next memory report reflects them. In steady state it scales with device capacity so a
    large card keeps proportional headroom against those transients while a small card is not starved below
    the floor. When the caller does not pass an explicit ``noise_buffer_mb`` it is derived from
    ``total_vram_mb`` via :func:`admission_noise_buffer_mb`; an explicitly supplied value always wins.

    Args:
        measured_committed_mb: The worker's committed-VRAM ledger sum (MB), baseline-exclusive.
        planned_unmaterialized_mb: VRAM (MB) admitted but not yet reflected in the measured floor.
        candidate_delta_mb: The candidate job's marginal predicted cost (MB), net of resident credit.
        total_vram_mb: Device total VRAM (MB), or None at cold start.
        baseline_mb: The reconciler's measured shared-device baseline (MB).
        noise_buffer_mb: The noise slack (MB) added to the baseline. None (the default) derives it from
            ``total_vram_mb`` via :func:`admission_noise_buffer_mb`; an explicit value always wins.
        committed_is_stale: True when a committed-ledger contributor's report has aged past the staleness
            bound, making the measured floor incomparable for this evaluation.
    """
    resolved_noise_buffer_mb = (
        noise_buffer_mb if noise_buffer_mb is not None else admission_noise_buffer_mb(total_vram_mb)
    )
    if total_vram_mb is None or total_vram_mb <= 0:
        return AdmissionVerdict(
            fits=True,
            used_measured_floor=False,
            committed_is_stale=committed_is_stale,
            measured_committed_mb=measured_committed_mb,
            planned_unmaterialized_mb=planned_unmaterialized_mb,
            candidate_delta_mb=candidate_delta_mb,
            total_vram_mb=None,
            baseline_mb=baseline_mb,
            noise_buffer_mb=resolved_noise_buffer_mb,
        )

    capacity_mb = (total_vram_mb - baseline_mb) - resolved_noise_buffer_mb
    if committed_is_stale:
        # Drop the untrustworthy measured floor but keep the parent's planned overlay, which needs no child
        # report: stacked unmaterialized admissions are knowable arithmetic and can still deny an over-commit.
        degraded_demand_mb = planned_unmaterialized_mb + candidate_delta_mb
        return AdmissionVerdict(
            fits=degraded_demand_mb <= capacity_mb,
            used_measured_floor=False,
            committed_is_stale=True,
            measured_committed_mb=measured_committed_mb,
            planned_unmaterialized_mb=planned_unmaterialized_mb,
            candidate_delta_mb=candidate_delta_mb,
            total_vram_mb=total_vram_mb,
            baseline_mb=baseline_mb,
            noise_buffer_mb=resolved_noise_buffer_mb,
        )

    demand_mb = measured_committed_mb + planned_unmaterialized_mb + candidate_delta_mb
    return AdmissionVerdict(
        fits=demand_mb <= capacity_mb,
        used_measured_floor=True,
        committed_is_stale=False,
        measured_committed_mb=measured_committed_mb,
        planned_unmaterialized_mb=planned_unmaterialized_mb,
        candidate_delta_mb=candidate_delta_mb,
        total_vram_mb=total_vram_mb,
        baseline_mb=baseline_mb,
        noise_buffer_mb=resolved_noise_buffer_mb,
    )
