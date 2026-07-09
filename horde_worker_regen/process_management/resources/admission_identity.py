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

from dataclasses import dataclass

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

    def reason(self) -> str:
        """Return the identity rendered for a log line, so a denial or unload is self-explaining."""
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
