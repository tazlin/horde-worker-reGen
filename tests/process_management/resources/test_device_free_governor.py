"""Unit tests for the device-free governor: floor arithmetic, debounce, transitions, and counters."""

from __future__ import annotations

from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.device_free_governor import (
    DeviceFreeGovernor,
    GovernorState,
    classify_free,
    hard_floor_mb,
    soft_floor_mb,
)

# A 16GB-class card: the proportional noise buffer (5% of total) dominates the absolute floors, so the soft
# and hard floors scale with the card rather than pinning a constant.
_TOTAL_16GB_MB = 16384.0


class TestFloorArithmetic:
    """The floors are pure functions of the device total via the shared admission noise buffer."""

    def test_soft_floor_is_twice_the_noise_buffer_on_a_large_card(self) -> None:
        """The soft floor is twice the proportional noise buffer once that clears the absolute floor."""
        expected = 2.0 * admission_noise_buffer_mb(_TOTAL_16GB_MB)
        assert soft_floor_mb(_TOTAL_16GB_MB) == expected
        # 5% of 16GB is ~819MB, so twice it (~1638MB) clears the 1024MB absolute floor.
        assert soft_floor_mb(_TOTAL_16GB_MB) > 1024.0

    def test_hard_floor_is_half_the_noise_buffer_but_never_below_the_absolute_floor(self) -> None:
        """The hard floor is half the noise buffer on a large card and the absolute floor on a small one."""
        # Half of ~819MB is ~410MB, above the 256MB absolute floor on a large card.
        assert hard_floor_mb(_TOTAL_16GB_MB) == max(256.0, 0.5 * admission_noise_buffer_mb(_TOTAL_16GB_MB))
        # A tiny card falls back to the absolute floors.
        assert soft_floor_mb(2048.0) == 1024.0
        assert hard_floor_mb(2048.0) == 256.0

    def test_unknown_total_uses_the_absolute_floors(self) -> None:
        """A cold start with no known total yields the absolute floors, not a proportional term."""
        assert soft_floor_mb(None) == 1024.0
        assert hard_floor_mb(None) == 256.0

    def test_soft_floor_is_always_above_the_hard_floor(self) -> None:
        """PRESSURE always engages before SATURATED across every card size."""
        for total in (None, 2048.0, 8192.0, _TOTAL_16GB_MB, 49152.0):
            assert soft_floor_mb(total) > hard_floor_mb(total)


class TestClassify:
    """Raw classification bands free VRAM against the two floors."""

    def test_healthy_above_soft_floor(self) -> None:
        """Free above the soft floor is HEALTHY."""
        assert classify_free(soft_floor_mb(_TOTAL_16GB_MB) + 1.0, _TOTAL_16GB_MB) == GovernorState.HEALTHY

    def test_pressure_between_floors(self) -> None:
        """Free between the hard and soft floors is PRESSURE."""
        free = (soft_floor_mb(_TOTAL_16GB_MB) + hard_floor_mb(_TOTAL_16GB_MB)) / 2.0
        assert classify_free(free, _TOTAL_16GB_MB) == GovernorState.PRESSURE

    def test_saturated_below_hard_floor(self) -> None:
        """Free below the hard floor is SATURATED."""
        assert classify_free(hard_floor_mb(_TOTAL_16GB_MB) - 1.0, _TOTAL_16GB_MB) == GovernorState.SATURATED


def _pressure_free() -> float:
    return (soft_floor_mb(_TOTAL_16GB_MB) + hard_floor_mb(_TOTAL_16GB_MB)) / 2.0


def _saturated_free() -> float:
    return hard_floor_mb(_TOTAL_16GB_MB) - 1.0


def _healthy_free() -> float:
    return soft_floor_mb(_TOTAL_16GB_MB) + 2048.0


class TestDebounceAndTransitions:
    """State changes require two consecutive agreeing samples; a lone transient never flips the state."""

    def test_starts_healthy(self) -> None:
        """A fresh governor reports HEALTHY before any sample."""
        governor = DeviceFreeGovernor()
        assert governor.state(0) == GovernorState.HEALTHY

    def test_single_pressure_sample_does_not_transition(self) -> None:
        """One PRESSURE reading is not yet enough to leave HEALTHY."""
        governor = DeviceFreeGovernor()
        sample = governor.observe(0, device_free_mb=_pressure_free(), total_vram_mb=_TOTAL_16GB_MB)
        assert sample.state == GovernorState.HEALTHY
        assert sample.transitioned is False
        assert governor.state(0) == GovernorState.HEALTHY

    def test_two_consecutive_pressure_samples_transition_once(self) -> None:
        """Two consecutive PRESSURE readings commit the transition exactly once."""
        governor = DeviceFreeGovernor()
        first = governor.observe(0, device_free_mb=_pressure_free(), total_vram_mb=_TOTAL_16GB_MB)
        second = governor.observe(0, device_free_mb=_pressure_free(), total_vram_mb=_TOTAL_16GB_MB)
        assert first.transitioned is False
        assert second.transitioned is True
        assert second.previous_state == GovernorState.HEALTHY
        assert second.state == GovernorState.PRESSURE
        assert governor.state(0) == GovernorState.PRESSURE
        assert governor.pressure_events(0) == 1

    def test_lone_transient_between_healthy_readings_does_not_flip(self) -> None:
        """A single PRESSURE reading bracketed by HEALTHY readings never flips the state."""
        governor = DeviceFreeGovernor()
        governor.observe(0, device_free_mb=_healthy_free(), total_vram_mb=_TOTAL_16GB_MB)
        governor.observe(0, device_free_mb=_pressure_free(), total_vram_mb=_TOTAL_16GB_MB)
        sample = governor.observe(0, device_free_mb=_healthy_free(), total_vram_mb=_TOTAL_16GB_MB)
        assert sample.state == GovernorState.HEALTHY
        assert governor.pressure_events(0) == 0

    def test_two_consecutive_saturated_samples_go_straight_to_saturated(self) -> None:
        """A direct HEALTHY->SATURATED jump counts a saturation event, not a pressure event."""
        governor = DeviceFreeGovernor()
        governor.observe(0, device_free_mb=_saturated_free(), total_vram_mb=_TOTAL_16GB_MB)
        sample = governor.observe(0, device_free_mb=_saturated_free(), total_vram_mb=_TOTAL_16GB_MB)
        assert sample.state == GovernorState.SATURATED
        assert sample.transitioned is True
        assert governor.saturation_events(0) == 1
        assert governor.pressure_events(0) == 0

    def test_recovery_transitions_back_to_healthy(self) -> None:
        """Two consecutive HEALTHY readings recover the card from SATURATED."""
        governor = DeviceFreeGovernor()
        for _ in range(2):
            governor.observe(0, device_free_mb=_saturated_free(), total_vram_mb=_TOTAL_16GB_MB)
        assert governor.state(0) == GovernorState.SATURATED
        governor.observe(0, device_free_mb=_healthy_free(), total_vram_mb=_TOTAL_16GB_MB)
        sample = governor.observe(0, device_free_mb=_healthy_free(), total_vram_mb=_TOTAL_16GB_MB)
        assert sample.state == GovernorState.HEALTHY
        assert sample.transitioned is True

    def test_re_entering_a_state_increments_its_counter_again(self) -> None:
        """Each fresh entry into PRESSURE increments the counter, so re-entry is counted twice."""
        governor = DeviceFreeGovernor()
        for _ in range(2):
            governor.observe(0, device_free_mb=_pressure_free(), total_vram_mb=_TOTAL_16GB_MB)
        for _ in range(2):
            governor.observe(0, device_free_mb=_healthy_free(), total_vram_mb=_TOTAL_16GB_MB)
        for _ in range(2):
            governor.observe(0, device_free_mb=_pressure_free(), total_vram_mb=_TOTAL_16GB_MB)
        assert governor.pressure_events(0) == 2


class TestPerDeviceIsolation:
    """Each device governs independently; counters and totals aggregate across governed cards."""

    def test_devices_are_independent(self) -> None:
        """One card reaching SATURATED does not move another card's state."""
        governor = DeviceFreeGovernor()
        for _ in range(2):
            governor.observe(0, device_free_mb=_saturated_free(), total_vram_mb=_TOTAL_16GB_MB)
        governor.observe(1, device_free_mb=_healthy_free(), total_vram_mb=_TOTAL_16GB_MB)
        assert governor.state(0) == GovernorState.SATURATED
        assert governor.state(1) == GovernorState.HEALTHY

    def test_totals_sum_across_devices(self) -> None:
        """The total counters sum each card's own transition counts."""
        governor = DeviceFreeGovernor()
        for _ in range(2):
            governor.observe(0, device_free_mb=_pressure_free(), total_vram_mb=_TOTAL_16GB_MB)
        for _ in range(2):
            governor.observe(1, device_free_mb=_saturated_free(), total_vram_mb=_TOTAL_16GB_MB)
        assert governor.total_pressure_events() == 1
        assert governor.total_saturation_events() == 1

    def test_unseen_device_reports_healthy_and_zero(self) -> None:
        """A card never observed reports HEALTHY and zero counters rather than raising."""
        governor = DeviceFreeGovernor()
        assert governor.state(7) == GovernorState.HEALTHY
        assert governor.pressure_events(7) == 0
        assert governor.saturation_events(7) == 0
