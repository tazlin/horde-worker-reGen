"""The measured-truth VRAM admission identity: a candidate fits only within real device-free room.

On Windows/WDDM the driver never fails an allocation at the physical ceiling: an over-commit is silently
demoted to the system-backed shared segment, and both ``mem_get_info`` and core-utilization telemetry keep
reading healthy. That lie was historically answered by reasoning from a book of what the worker believed it
had committed. A book cannot referee several independent allocators and lying per-process telemetry; the one
figure that does not lie under WDDM is the parent-side NVML device-level free reading, which physically
includes every baseline allocation, every foreign allocation, and every materialised worker allocation. This
module makes that reading the primary admission input.

The identity is stated once as a pure function so the arbiter and its tests reason about the same inequality:

    available(d) = device_free_mb(d)
                 - outstanding_reservations_mb(d, excluding the requester's own unit)
                 - noise_buffer_mb(total(d))

    FITS  iff  candidate_outstanding_mb <= available(d)

- ``device_free_mb`` is the frozen per-cycle NVML device-level free reading. It already contains the shared
  baseline (OS/desktop/other apps), foreign allocations, and every materialised worker load, so none of those
  is a separate term: they are physically inside the reading.
- ``outstanding_reservations_mb`` protects work already admitted whose allocation the free reading does not
  yet reflect (a preload staged in RAM about to move to VRAM, a dispatch about to activate). Each reservation
  decays as its target's real reservation materialises, so a load is never counted twice (once physically in
  ``device_free_mb`` once it lands, once as a reservation before it does). The caller nets the requester's own
  outstanding reservation out before passing this figure, so a re-ask never defers on its own admitted plan.
- ``noise_buffer_mb`` is the one margin: it absorbs measurement/rounding noise and the activation transients an
  allocator briefly holds between reports, scaling with device capacity above a floor (see
  :func:`admission_noise_buffer_mb`). It is NOT the operator's ``vram_reserve_mb``: that configured reserve is
  the sampling gate's activation margin and is never a load-feasibility floor (see the reserve-decoupling
  contract in :mod:`~horde_worker_regen.process_management.resources.resource_budget`); folding it in here would
  repeat the wedge that decoupling exists to prevent, where a model whose weights fit the drained card reads as
  unloadable.

When ``device_free_mb`` is None (no NVML reading for this card yet) the identity is indeterminate:
``available_known`` is False and ``fits`` is False, so the arbiter defers with a diagnostic rather than either
denying or fabricating a fictional free figure. The device total is retained only to size the noise buffer.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass

from strenum import StrEnum

_ADMISSION_NOISE_BUFFER_MB = 512.0
"""Floor (MB) of the admission noise buffer, the value it takes on small cards and when no total is known.

The admission margin subtracts this (or a proportional share of the device total, whichever is larger) from
the device-free reading so an admission never lands on the exact measured edge, where ordinary measurement
noise or an inter-report activation transient would tip the card over the paging cliff. It scales with device
capacity (see :func:`admission_noise_buffer_mb`) so a large card keeps proportional headroom while a small
card is never starved below this floor. This is intentionally NOT the operator's ``vram_reserve_mb``, which
remains the sampling gate's per-step activation margin: the reserve is never a load-feasibility floor (making
it one is exactly the wedge the reserve-decoupling contract prevents). Sized well below one model's weights so
it never denies a load the card physically holds, but above the sub-hundred-MB slack a rounded device figure
introduces."""

_ADMISSION_NOISE_BUFFER_FRACTION = 0.05
"""Fraction of the device total VRAM the noise buffer scales to once that exceeds the floor.

Chosen so the margin tracks device capacity (a 24GB card affords roughly 1.2GB of transient headroom, an
8GB card cannot afford more than the floor) rather than pinning a flat constant tuned to one reference
card, per the fleet-heterogeneity contract."""


def admission_noise_buffer_mb(total_vram_mb: float | None) -> float:
    """Return the admission noise buffer (MB): the floor, or ``5%`` of the device total when that is larger.

    The buffer absorbs measurement noise and the activation transients an allocator holds between memory
    reports, and scales with device capacity so large cards keep proportional headroom while small cards are
    never starved below :data:`_ADMISSION_NOISE_BUFFER_MB`. An unknown or non-positive total (cold start)
    yields the floor, since no proportional term can be formed without a capacity to scale against.

    Args:
        total_vram_mb: The device's total VRAM (MB), or None when no total has been reported yet.
    """
    if total_vram_mb is None or total_vram_mb <= 0:
        return _ADMISSION_NOISE_BUFFER_MB
    return max(_ADMISSION_NOISE_BUFFER_MB, _ADMISSION_NOISE_BUFFER_FRACTION * total_vram_mb)


_ADMISSION_MARGIN_FRACTION_WDDM = _ADMISSION_NOISE_BUFFER_FRACTION
"""Fraction of the device total the admission margin scales to on Windows/WDDM, where the free reading
oscillates as the driver demotes over-commits to shared memory; the physics buffer's own fraction."""

_ADMISSION_MARGIN_FRACTION_DEVICE_WIDE = 0.025
"""Fraction of the device total the admission margin scales to where the NVML free reading is device-wide
and stable (Linux): half the WDDM fraction, since there is no paging oscillation to absorb, only the
inter-report activation transients. The device-free governor's floors do not follow this figure; they stay
on :func:`admission_noise_buffer_mb`, because on such a platform an over-commit is a hard OOM rather than
paging and the pressure thresholds are what stand between growth and that failure."""


def admission_margin_mb(
    total_vram_mb: float | None,
    *,
    override_mb: float | None = None,
    platform: str = sys.platform,
) -> float:
    """Return the admission identity's margin (MB): the operator's override, else the platform-scaled buffer.

    The margin the admission sites (the identity, the achievable ceiling, model serviceability, the streaming
    forecast, the retention fit) subtract from device-free room. On WDDM it is the physics buffer; on a
    platform whose free reading is device-wide it scales to :data:`_ADMISSION_MARGIN_FRACTION_DEVICE_WIDE`
    above the same floor. An explicit override wins whole (an operator who knows the card can set it low or
    zero), so the out is direct rather than a scaling knob.

    Args:
        total_vram_mb: The device's total VRAM (MB), or None when no total has been reported yet.
        override_mb: The operator's ``vram_admission_noise_mb`` for this card, or None to derive.
        platform: ``sys.platform`` by default; injectable so a test can pin either shape.
    """
    if override_mb is not None:
        return max(0.0, float(override_mb))
    if total_vram_mb is None or total_vram_mb <= 0:
        return _ADMISSION_NOISE_BUFFER_MB
    fraction = _ADMISSION_MARGIN_FRACTION_WDDM if platform == "win32" else _ADMISSION_MARGIN_FRACTION_DEVICE_WIDE
    return max(_ADMISSION_NOISE_BUFFER_MB, fraction * total_vram_mb)


@dataclass(frozen=True)
class AdmissionVerdict:
    """The outcome of one evaluation of the measured-truth admission identity, carrying every term for logging.

    ``available_known`` is False whenever the device-free reading was absent, so ``available`` could not be
    formed. In that case ``fits`` is False and the caller defers rather than guessing; the arbiter never denies
    on a missing reading and never fabricates a fallback. When ``available_known`` is True ``fits`` is the
    authoritative result of ``candidate_outstanding <= available``.
    """

    fits: bool
    """Whether the candidate fits available room; always False when ``available_known`` is False."""
    available_known: bool
    """True when the device-free reading was present, so ``available`` could be formed; False otherwise."""
    candidate_outstanding_mb: float
    """The candidate's marginal outstanding device cost (MB), net of any weights already resident in its target."""
    device_free_mb: float | None
    """The frozen NVML device-level free VRAM (MB) for this card, or None when no reading was available."""
    outstanding_reservations_mb: float
    """Admitted-but-unmaterialized reservations (MB) the free reading does not yet reflect, net of the
    requester's own outstanding reservation (subtracted by the caller so a re-ask never blocks on itself)."""
    total_vram_mb: float | None
    """Device total VRAM (MB), or None when unknown; retained only to size the noise buffer."""
    noise_buffer_mb: float
    """The one margin (MB) subtracted from device-free room; scales with device capacity above a floor."""

    @property
    def available_mb(self) -> float | None:
        """The admission room (MB): ``device_free - outstanding_reservations - noise``, or None when unknown."""
        if self.device_free_mb is None:
            return None
        return self.device_free_mb - self.outstanding_reservations_mb - self.noise_buffer_mb

    @property
    def headroom_mb(self) -> float | None:
        """Available room minus the candidate (MB): positive when the identity holds, or None when unknown."""
        available = self.available_mb
        if available is None:
            return None
        return available - self.candidate_outstanding_mb

    def stable_reason(self) -> str:
        """Return what the identity decided, with none of the figures it decided from.

        Every measurement in :meth:`reason` moves between cycles: the free reading breathes as foreign VRAM
        does, and the reservations decay as loads materialise. A caller that stores or compares the verdict
        text (a coalesced defer record, a stall attribution held across a settling window, a throttle keyed on
        the objection being unchanged) needs the block, not the arithmetic, or an unchanged refusal reads as a
        new one every cycle and nothing can throttle or judge it. Those callers take this; the lines a person
        reads render it alongside :meth:`reason`.
        """
        if not self.available_known:
            return "no device-free reading for this card, so admission is indeterminate"
        return (
            "the candidate fits the measured available room"
            if self.fits
            else "the candidate does not fit the measured available room"
        )

    def reason(self) -> str:
        """Return the identity rendered for a log line, so a denial or unload is self-explaining.

        Carries the cycle's measurements, so it belongs to a line a person reads. Anything that stores or
        compares a verdict across cycles takes :meth:`stable_reason` instead.
        """
        if not self.available_known:
            return (
                "device-free reading unavailable for this card; admission deferred (no fictional fallback, "
                "no denial on a missing measurement)"
            )
        free = self.device_free_mb if self.device_free_mb is not None else 0.0
        verb = "fits" if self.fits else "does NOT fit"
        available = free - self.outstanding_reservations_mb - self.noise_buffer_mb
        return (
            f"candidate {self.candidate_outstanding_mb:.0f} MB vs available (device-free {free:.0f} - "
            f"reservations {self.outstanding_reservations_mb:.0f} - noise {self.noise_buffer_mb:.0f}) = "
            f"{available:.0f} MB: {verb}"
        )


def evaluate_admission(
    *,
    candidate_outstanding_mb: float,
    device_free_mb: float | None,
    outstanding_reservations_mb: float,
    total_vram_mb: float | None,
    noise_buffer_mb: float | None = None,
) -> AdmissionVerdict:
    """Evaluate the measured-truth admission identity against the frozen device-free reading.

    Admits iff ``candidate_outstanding <= device_free - outstanding_reservations - noise_buffer``. The device
    total is consulted only to size the noise buffer when the caller does not supply one; an explicitly passed
    ``noise_buffer_mb`` always wins. A missing device-free reading yields an indeterminate verdict
    (``available_known=False``, ``fits=False``) so the caller defers rather than denying or fabricating room.

    The reservations figure must already exclude the requester's own outstanding reservation: the identity
    subtracts it whole, so a re-ask that still carries its own admitted-but-unmaterialized plan is not deferred
    on its own footprint. Every other unit's reservation stays fully charged.

    Args:
        candidate_outstanding_mb: The candidate's marginal device cost (MB), net of resident-weight credit.
        device_free_mb: The frozen NVML device-level free VRAM (MB) for this card, or None when unavailable.
        outstanding_reservations_mb: Admitted-but-unmaterialized reservations (MB) the free reading does not
            yet reflect, already net of the requester's own outstanding reservation.
        total_vram_mb: Device total VRAM (MB), or None when unknown; used only to size the noise buffer.
        noise_buffer_mb: The margin (MB). None (the default) derives it from ``total_vram_mb`` via
            :func:`admission_noise_buffer_mb`; an explicit value always wins.
    """
    resolved_noise_buffer_mb = (
        noise_buffer_mb if noise_buffer_mb is not None else admission_noise_buffer_mb(total_vram_mb)
    )
    if device_free_mb is None:
        return AdmissionVerdict(
            fits=False,
            available_known=False,
            candidate_outstanding_mb=candidate_outstanding_mb,
            device_free_mb=None,
            outstanding_reservations_mb=outstanding_reservations_mb,
            total_vram_mb=total_vram_mb,
            noise_buffer_mb=resolved_noise_buffer_mb,
        )
    available_mb = device_free_mb - outstanding_reservations_mb - resolved_noise_buffer_mb
    return AdmissionVerdict(
        fits=candidate_outstanding_mb <= available_mb,
        available_known=True,
        candidate_outstanding_mb=candidate_outstanding_mb,
        device_free_mb=device_free_mb,
        outstanding_reservations_mb=outstanding_reservations_mb,
        total_vram_mb=total_vram_mb,
        noise_buffer_mb=resolved_noise_buffer_mb,
    )


class TenantLane(StrEnum):
    """The class of tenant a live GPU process is, for the room breakdown a refusal is explained with.

    The admission identity prices against one device-free number, which is the right authority but says
    nothing about who holds the card. A refusal a person or a reclaim policy can act on needs the reading
    decomposed by the class of holder, because each class is returned by a different actuator: an idle
    inference sibling by a context teardown, the post-processing and utilities lanes and safety by their
    off-GPU pauses, and the foreign share by nothing the worker can do.
    """

    INFERENCE_TARGET = "inference_target"
    INFERENCE_IDLE = "inference_idle"
    INFERENCE_BUSY = "inference_busy"
    SAFETY = "safety"
    POST_PROCESS = "post_process"
    UTILITIES = "utilities"
    VAE_LANE = "vae_lane"
    COMPONENT = "component"
    FOREIGN = "foreign"


class RoomRungKind(StrEnum):
    """A reclaim rung the room breakdown can price, cheapest first in declaration order."""

    IDLE_SIBLING_CONTEXT = "idle_sibling_context"
    POST_PROCESS_LANE = "post_process_lane"
    SAFETY_OFF_GPU = "safety_off_gpu"
    UTILITIES_LANE = "utilities_lane"


@dataclass(frozen=True)
class RoomRung:
    """One rung's promised return (MB) and whether policy currently permits pulling it."""

    kind: RoomRungKind
    promised_mb: float
    permitted: bool


@dataclass(frozen=True)
class AdmissionRoom:
    """The measured admission identity decomposed by tenant, with the rungs that could change the answer.

    Built from the same frozen per-cycle figures :func:`evaluate_admission` prices from, so the deficit here is
    exactly the verdict's deficit; nothing is re-measured. ``tenancy_mb`` attributes the device-used reading
    (``total - device_free``) to tenant classes from each live process's measured reservation plus its context
    charge; whatever those do not account for is the foreign share (or zero when the sum overshoots the
    reading: the per-process reports and the device reading are not sampled at the same instant).

    ``rungs`` lists what each reclaim rung would return, cheapest first. ``closable`` is whether the permitted
    rungs together cover the deficit, which is what separates a hold worth escalating from one that can only
    wait for the probe or for foreign VRAM to leave.
    """

    candidate_mb: float
    device_free_mb: float
    total_vram_mb: float | None
    outstanding_reservations_mb: float
    noise_buffer_mb: float
    tenancy_mb: Mapping[TenantLane, float]
    rungs: tuple[RoomRung, ...]

    @property
    def available_mb(self) -> float:
        """The admission room (MB): ``device_free - outstanding_reservations - noise``."""
        return self.device_free_mb - self.outstanding_reservations_mb - self.noise_buffer_mb

    @property
    def deficit_mb(self) -> float:
        """How far the candidate misses the room (MB); zero or negative when it fits."""
        return self.candidate_mb - self.available_mb

    @property
    def reclaimable_mb(self) -> float:
        """What the permitted rungs would return together (MB)."""
        return sum(rung.promised_mb for rung in self.rungs if rung.permitted)

    @property
    def unpausable_mb(self) -> float:
        """Tenancy no permitted rung returns (MB): the foreign share, busy siblings, and rungs policy forbids."""
        forbidden = sum(rung.promised_mb for rung in self.rungs if not rung.permitted)
        return (
            self.tenancy_mb.get(TenantLane.FOREIGN, 0.0)
            + self.tenancy_mb.get(TenantLane.INFERENCE_BUSY, 0.0)
            + forbidden
        )

    @property
    def closable(self) -> bool:
        """Whether the permitted rungs together cover the deficit."""
        return self.deficit_mb > 0 and self.reclaimable_mb >= self.deficit_mb

    def rungs_to_close(self) -> tuple[RoomRung, ...]:
        """The cheapest-first permitted rungs whose cumulative return first covers the deficit, or all of them."""
        chosen: list[RoomRung] = []
        covered = 0.0
        for rung in self.rungs:
            if not rung.permitted:
                continue
            chosen.append(rung)
            covered += rung.promised_mb
            if covered >= self.deficit_mb:
                break
        return tuple(chosen)

    def describe(self) -> str:
        """One log line: the deficit, who holds the card, and what the rungs could return."""
        tenancy = ", ".join(f"{lane.value} {mb:.0f}" for lane, mb in self.tenancy_mb.items() if mb > 0.0)
        rungs = ", ".join(
            f"{rung.kind.value} {rung.promised_mb:.0f}{'' if rung.permitted else ' (not permitted)'}"
            for rung in self.rungs
        )
        if self.deficit_mb > 0:
            verdict = f"short {self.deficit_mb:.0f} MB ({'closable by rungs' if self.closable else 'not closable'})"
        else:
            verdict = f"fits by {-self.deficit_mb:.0f} MB"
        return (
            f"{verdict}; tenancy MB: {tenancy or 'none attributed'}; noise {self.noise_buffer_mb:.0f}; "
            f"rungs MB: {rungs or 'none'}; unpausable {self.unpausable_mb:.0f}"
        )

    def as_inputs(self) -> dict[str, str | int | float | bool | None]:
        """Flat scalars for a decision or resource-state record."""
        inputs: dict[str, str | int | float | bool | None] = {
            "room_candidate_mb": round(self.candidate_mb, 1),
            "room_available_mb": round(self.available_mb, 1),
            "room_deficit_mb": round(self.deficit_mb, 1),
            "room_reclaimable_mb": round(self.reclaimable_mb, 1),
            "room_unpausable_mb": round(self.unpausable_mb, 1),
            "room_closable": self.closable,
        }
        for lane, mb in self.tenancy_mb.items():
            inputs[f"tenancy_{lane.value}_mb"] = round(mb, 1)
        for rung in self.rungs:
            inputs[f"rung_{rung.kind.value}_mb"] = round(rung.promised_mb, 1)
            inputs[f"rung_{rung.kind.value}_permitted"] = rung.permitted
        return inputs


def admission_room(
    *,
    candidate_mb: float,
    device_free_mb: float,
    total_vram_mb: float | None,
    outstanding_reservations_mb: float,
    noise_buffer_mb: float,
    per_process_reserved_mb: Mapping[int, float],
    lane_by_process_id: Mapping[int, TenantLane],
    target_process_id: int | None,
    context_mb: float,
    safety_footprint_mb: float,
    post_process_reclaim_permitted: bool,
    safety_reclaim_permitted: bool,
    utilities_reclaim_permitted: bool,
    safety_residency_fixed: bool = False,
) -> AdmissionRoom:
    """Decompose the admission identity for one candidate into tenancy by class and reclaim rungs.

    A live process's tenancy is its measured allocator reservation plus one context charge (``context_mb``,
    the per-context figure the overhead probe measured); safety is priced at its whole-device footprint, the
    figure every other safety consumer uses, when its reservation is not reported. The target slot is a tenant
    too (its context stays whatever happens), but it is never a rung. The foreign share is the device-used
    reading less everything attributed, floored at zero.

    Rungs, cheapest first: each idle inference sibling context (its full tenancy returns when the process
    exits), the post-processing lane, safety off-GPU, the utilities lane. A rung is listed even when policy
    forbids it, so the line a person reads shows what an operator setting would unlock. A fixed safety
    residency is the one exception: no setting unlocks it, so it is charged at no less than its whole-device
    footprint and named as tenancy only, never as a rung a candidate could be admitted against.

    Args:
        candidate_mb: The candidate's outstanding device cost (MB), net of resident credit.
        device_free_mb: The frozen NVML device-level free reading (MB).
        total_vram_mb: The device total (MB), or None when unknown (then no foreign share can be formed).
        outstanding_reservations_mb: Admitted-but-unmaterialised reservations (MB), net of the requester's own.
        noise_buffer_mb: The admission margin (MB).
        per_process_reserved_mb: Each live GPU process's measured reservation (MB) by process id.
        lane_by_process_id: Each live GPU process's tenant class by process id.
        target_process_id: The slot the candidate would materialise on; classed as the target tenant.
        context_mb: The per-context charge (MB) added to each process's reservation.
        safety_footprint_mb: Safety's whole-device footprint (MB), used when its reservation is unreported.
        post_process_reclaim_permitted: Whether policy permits pausing the post-processing lane off-GPU now.
        safety_reclaim_permitted: Whether policy permits moving safety off-GPU now.
        utilities_reclaim_permitted: Whether policy permits pausing the utilities lane off-GPU now.
        safety_residency_fixed: Whether safety's card is fixed, so its charge cannot fall below its
            whole-device footprint and no candidate may be admitted against moving it off.
    """
    tenancy: dict[TenantLane, float] = {}
    rungs: list[RoomRung] = []
    attributed = 0.0
    for process_id, lane in lane_by_process_id.items():
        reserved = per_process_reserved_mb.get(process_id)
        if lane is TenantLane.SAFETY and reserved is None:
            charge = max(0.0, safety_footprint_mb)
        else:
            charge = max(0.0, reserved or 0.0) + max(0.0, context_mb)
        if lane is TenantLane.SAFETY and safety_residency_fixed:
            # A fixed residency is not going anywhere, so the card owes safety its whole-device footprint for
            # the rest of the session. A reservation reported below that figure is an at-rest reading taken
            # between evaluations, not a smaller commitment, and admitting a candidate against the difference
            # over-commits the card the moment the next check runs.
            charge = max(charge, max(0.0, safety_footprint_mb))
        classed = TenantLane.INFERENCE_TARGET if process_id == target_process_id else lane
        tenancy[classed] = tenancy.get(classed, 0.0) + charge
        attributed += charge
        if classed is TenantLane.INFERENCE_IDLE:
            rungs.append(RoomRung(RoomRungKind.IDLE_SIBLING_CONTEXT, charge, True))
    lane_rungs = (
        (TenantLane.POST_PROCESS, RoomRungKind.POST_PROCESS_LANE, post_process_reclaim_permitted),
        (TenantLane.SAFETY, RoomRungKind.SAFETY_OFF_GPU, safety_reclaim_permitted),
        (TenantLane.UTILITIES, RoomRungKind.UTILITIES_LANE, utilities_reclaim_permitted),
    )
    for lane, kind, permitted in lane_rungs:
        if lane is TenantLane.SAFETY and safety_residency_fixed:
            continue
        held = tenancy.get(lane, 0.0)
        if held > 0.0:
            rungs.append(RoomRung(kind, held, permitted))
    if total_vram_mb is not None and total_vram_mb > 0:
        used_mb = max(0.0, total_vram_mb - device_free_mb)
        tenancy[TenantLane.FOREIGN] = max(0.0, used_mb - attributed)
    return AdmissionRoom(
        candidate_mb=candidate_mb,
        device_free_mb=device_free_mb,
        total_vram_mb=total_vram_mb,
        outstanding_reservations_mb=outstanding_reservations_mb,
        noise_buffer_mb=noise_buffer_mb,
        tenancy_mb=tenancy,
        rungs=tuple(rungs),
    )
