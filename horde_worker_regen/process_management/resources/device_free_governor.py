"""The device-free governor: a truthful, debounced read of how close each card is to a paging cliff.

On Windows/WDDM the driver never fails an allocation at the physical ceiling; it silently demotes the
least-recently-touched allocator to the system-backed shared segment, and throughput then falls off a hard
cliff the instant NVML device-free reaches roughly zero. Per-process telemetry cannot see this coming (the
demoted process and the crawling process are usually different PIDs, and per-PID shared attribution is
unreliable run to run), but the NVML device-level free figure read from the torch-free parent is truthful
throughout. This module turns that one truthful figure into a small, hysteretic state machine so the
scheduler can hold new VRAM growth before the cliff and reclaim once the card is already over it.

Three states per device, all arithmetic (no card-capacity constants):

- ``HEALTHY``: device free is above the soft floor; nothing to do.
- ``PRESSURE``: device free is below the soft floor. Growth toward the card must stop (no new model brought
  to VRAM on a process that does not already hold it, no safety GPU restore, no paused lane restart); work
  already sampling continues, because on WDDM it is not the active sampler the driver demotes.
- ``SATURATED``: device free is below the hard floor. The card is at or past the cliff; the reclaim ladder
  must run immediately.

The floors scale with the same proportional noise buffer admission uses (see
:func:`~horde_worker_regen.process_management.resources.admission_identity.admission_noise_buffer_mb`) so a
large card keeps proportional headroom and a small card is never starved below an absolute floor. State
changes are debounced over two consecutive samples: NVML is stable, but a two-sample confirm costs one tick
of latency and removes any chance a lone transient (a load spike mid-materialisation) flips the state.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb

_SOFT_FLOOR_FLOOR_MB = 1024.0
"""Absolute lower bound (MB) of the PRESSURE soft floor, used on small cards and when no total is known.

The soft floor is ``max(_SOFT_FLOOR_FLOOR_MB, 2 x noise_buffer)``: twice the admission noise buffer so the
governor begins holding growth a comfortable margin before the card reaches the paging cliff, never below
this absolute so a small card still gets a real warning band rather than a sliver."""

_HARD_FLOOR_FLOOR_MB = 256.0
"""Absolute lower bound (MB) of the SATURATED hard floor, used on small cards and when no total is known.

The hard floor is ``max(_HARD_FLOOR_FLOOR_MB, 0.5 x noise_buffer)``: half the admission noise buffer, the
band just above the physical cliff where reclaim must run now. Kept above zero so the governor reacts a
sample before NVML device-free actually bottoms out and the whole card falls off the throughput cliff."""

_DEBOUNCE_SAMPLES = 2
"""Consecutive samples agreeing on a new raw state before the committed state changes.

NVML device-free is stable, so this is cheap insurance rather than a necessity: it guarantees no single
transient reading (an allocator mid-materialisation, a foreign app's momentary spike) can flip the governor,
at a cost of exactly one tick of extra latency on a genuine transition."""


class GovernorState(enum.StrEnum):
    """How close a device is to the WDDM paging cliff, as a debounced read of NVML device-free VRAM."""

    HEALTHY = "healthy"
    """Device free is above the soft floor; no growth hold and no reclaim."""
    PRESSURE = "pressure"
    """Device free is below the soft floor; hold new VRAM growth, let in-flight sampling continue."""
    SATURATED = "saturated"
    """Device free is below the hard floor; run the reclaim ladder immediately."""


def soft_floor_mb(total_vram_mb: float | None) -> float:
    """Return the PRESSURE soft floor (MB): ``max(_SOFT_FLOOR_FLOOR_MB, 2 x noise_buffer)``.

    Args:
        total_vram_mb: The device total VRAM (MB), or None when no total has been reported yet.
    """
    return max(_SOFT_FLOOR_FLOOR_MB, 2.0 * admission_noise_buffer_mb(total_vram_mb))


def hard_floor_mb(total_vram_mb: float | None) -> float:
    """Return the SATURATED hard floor (MB): ``max(_HARD_FLOOR_FLOOR_MB, 0.5 x noise_buffer)``.

    Args:
        total_vram_mb: The device total VRAM (MB), or None when no total has been reported yet.
    """
    return max(_HARD_FLOOR_FLOOR_MB, 0.5 * admission_noise_buffer_mb(total_vram_mb))


def classify_free(device_free_mb: float, total_vram_mb: float | None) -> GovernorState:
    """Classify a raw device-free reading into a governor state, before debouncing.

    Args:
        device_free_mb: The device's free VRAM (MB) read from NVML in the torch-free parent.
        total_vram_mb: The device total VRAM (MB), or None when no total has been reported yet.
    """
    if device_free_mb < hard_floor_mb(total_vram_mb):
        return GovernorState.SATURATED
    if device_free_mb < soft_floor_mb(total_vram_mb):
        return GovernorState.PRESSURE
    return GovernorState.HEALTHY


@dataclass(frozen=True)
class GovernorSample:
    """The outcome of feeding one NVML device-free reading to a device's governor.

    ``state`` is the committed (debounced) state after this sample; ``previous_state`` is what it was before.
    ``transitioned`` is True exactly on the sample that commits a change, so a caller can act once on entry
    to PRESSURE or SATURATED rather than on every sample while the state persists.
    """

    device_index: int
    device_free_mb: float
    total_vram_mb: float | None
    state: GovernorState
    previous_state: GovernorState
    transitioned: bool
    soft_floor_mb: float
    hard_floor_mb: float


class _DeviceGovernor:
    """The debounced state machine for one device (parent control loop only; no locking)."""

    def __init__(self) -> None:
        self._state = GovernorState.HEALTHY
        self._pending_state = GovernorState.HEALTHY
        self._pending_count = 0
        self.pressure_events = 0
        self.saturation_events = 0

    def observe(self, device_index: int, device_free_mb: float, total_vram_mb: float | None) -> GovernorSample:
        """Fold one device-free reading in, debounce it, and return the resulting sample."""
        raw = classify_free(device_free_mb, total_vram_mb)
        previous = self._state
        transitioned = False

        if raw == self._state:
            # The reading agrees with the committed state; drop any half-formed transition toward another.
            self._pending_state = self._state
            self._pending_count = 0
        else:
            if raw == self._pending_state:
                self._pending_count += 1
            else:
                self._pending_state = raw
                self._pending_count = 1
            if self._pending_count >= _DEBOUNCE_SAMPLES:
                self._state = raw
                self._pending_count = 0
                transitioned = True
                if raw == GovernorState.PRESSURE:
                    self.pressure_events += 1
                elif raw == GovernorState.SATURATED:
                    self.saturation_events += 1

        return GovernorSample(
            device_index=device_index,
            device_free_mb=device_free_mb,
            total_vram_mb=total_vram_mb,
            state=self._state,
            previous_state=previous,
            transitioned=transitioned,
            soft_floor_mb=soft_floor_mb(total_vram_mb),
            hard_floor_mb=hard_floor_mb(total_vram_mb),
        )

    @property
    def state(self) -> GovernorState:
        """The committed (debounced) state."""
        return self._state


class DeviceFreeGovernor:
    """A per-device family of debounced NVML device-free state machines.

    One instance lives on the parent process manager. Each control-loop tick, the parent samples NVML
    device used/free for every driven card and feeds it here; the returned :class:`GovernorSample` tells the
    parent whether to hold VRAM growth (PRESSURE) or run the reclaim ladder (SATURATED) on that card. All
    state is per device so a multi-GPU host governs each card independently.
    """

    def __init__(self) -> None:
        """Initialise with no per-device state; devices are created lazily on first observation."""
        self._governors: dict[int, _DeviceGovernor] = {}

    def observe(self, device_index: int, *, device_free_mb: float, total_vram_mb: float | None) -> GovernorSample:
        """Feed one device-free reading for ``device_index`` and return the debounced sample.

        Args:
            device_index: The device the reading is for.
            device_free_mb: The device's free VRAM (MB) read from NVML in the torch-free parent.
            total_vram_mb: The device total VRAM (MB), or None when no total has been reported yet.
        """
        governor = self._governors.get(device_index)
        if governor is None:
            governor = _DeviceGovernor()
            self._governors[device_index] = governor
        return governor.observe(device_index, device_free_mb, total_vram_mb)

    def state(self, device_index: int) -> GovernorState:
        """Return the committed state for ``device_index`` (HEALTHY when never observed)."""
        governor = self._governors.get(device_index)
        return governor.state if governor is not None else GovernorState.HEALTHY

    def pressure_events(self, device_index: int) -> int:
        """Return how many times ``device_index`` has transitioned into PRESSURE this run."""
        governor = self._governors.get(device_index)
        return governor.pressure_events if governor is not None else 0

    def saturation_events(self, device_index: int) -> int:
        """Return how many times ``device_index`` has transitioned into SATURATED this run."""
        governor = self._governors.get(device_index)
        return governor.saturation_events if governor is not None else 0

    def total_pressure_events(self) -> int:
        """Return the summed PRESSURE transition count across all governed devices."""
        return sum(governor.pressure_events for governor in self._governors.values())

    def total_saturation_events(self) -> int:
        """Return the summed SATURATED transition count across all governed devices."""
        return sum(governor.saturation_events for governor in self._governors.values())
