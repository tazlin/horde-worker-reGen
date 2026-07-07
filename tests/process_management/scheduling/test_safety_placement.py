"""The runtime safety-placement policy: keep safety off-GPU when its charge cannot fit beside sampling.

The policy generalises the whole-card safety-off lever to the ordinary case. Demotion prices a modeled worst
case (device total, largest learned sampling peak, proportional noise buffer, the static safety charge);
re-promotion instead reads the chosen card's measured device-free between allocation peaks, so it stays
satisfiable under sustained load rather than waiting for a sampling-free window that never comes. It only ever
degrades the operator's placement (GPU to CPU) and back, never beyond the operator's grant, and its
pause/restore is hysteresis-gated so a card oscillating around the fit boundary does not flap the safety
process on and off the card. Placement is headroom-aware across cards, not a fixed device 0.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling import inference_scheduler as sched_mod
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _SAFETY_PLACEMENT_PAUSE_STREAK,
    _SAFETY_PLACEMENT_RESTORE_STREAK,
)
from tests.process_management.conftest import make_mock_bridge_data, make_test_card_runtimes
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _placement_scheduler(monkeypatch: pytest.MonkeyPatch, *, safety_on_gpu: bool = True):  # noqa: ANN202
    """A single-GPU scheduler whose safety process is placement-managed, with a mocked lifecycle.

    The CPU-only guard is patched off so the policy is active regardless of the test host: on a real CPU-only
    install safety is always off-GPU already, so the policy would (correctly) be inert.
    """
    monkeypatch.setattr(sched_mod, "is_cpu_only_install", lambda: False)
    bridge_data = make_mock_bridge_data(safety_on_gpu=safety_on_gpu)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({}), bridge_data=bridge_data)
    lifecycle = Mock()
    lifecycle.is_safety_gpu_paused = False
    lifecycle.pause_safety_on_gpu = Mock(return_value=True)
    lifecycle.restore_safety_on_gpu = Mock(return_value=True)
    scheduler._process_lifecycle = lifecycle
    return scheduler


class TestSafetyFitArithmetic:
    """The structural fit is arithmetic over the device total and the largest active sampling peak."""

    def test_charge_fits_on_a_roomy_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A large card holds the safety charge beside a moderate peak, bare and with margin."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=24000.0)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=8192.0)
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=False) is True
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=True) is True

    def test_tight_card_bare_fit_but_no_margin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On a tight card the charge bare-fits but fails the proportional restore margin (hysteresis band)."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=11500.0)
        # 16000 - 11500 - 800 (5% noise) - 3044 (safety) = 656 >= 0 bare; a second 800 margin makes it negative.
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=False) is True
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=True) is False

    def test_nothing_sampling_fits_trivially(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no active sampling peak the charge trivially fits (nothing to fit beside)."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=None)
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=True) is True

    def test_unknown_total_fits_trivially(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unreported device total leaves the charge fitting (missing-telemetry admits)."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=None)
        assert scheduler._safety_fits_beside_largest_sampling_peak(None, require_margin=False) is True


class TestPlacementHysteresis:
    """The pause/restore latch turns on and off only after runs of consecutive non-fitting / fitting cycles."""

    def test_pauses_only_after_consecutive_misses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Safety is not evicted on a single miss; it takes the configured run of consecutive misses."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False

        for _ in range(_SAFETY_PLACEMENT_PAUSE_STREAK - 1):
            scheduler._reconcile_runtime_safety_placement()
            scheduler._process_lifecycle.pause_safety_on_gpu.assert_not_called()

        scheduler._reconcile_runtime_safety_placement()
        scheduler._process_lifecycle.pause_safety_on_gpu.assert_called_once_with()
        assert scheduler._safety_placement_wants_off is True

    def test_restores_only_after_consecutive_measured_headroom_cycles(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A paused-off safety is restored only after the longer run of measured-device-free-headroom cycles.

        The modeled charge stays non-fitting throughout (sustained load), proving the re-promotion is driven by
        the measured device-free signal, not the modeled one that is unsatisfiable while jobs flow.
        """
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False
        scheduler._safety_restore_headroom_fits = lambda device_index: True
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._safety_placement_wants_off = True

        for _ in range(_SAFETY_PLACEMENT_RESTORE_STREAK - 1):
            scheduler._reconcile_runtime_safety_placement()
            scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()

        scheduler._reconcile_runtime_safety_placement()
        scheduler._process_lifecycle.restore_safety_on_gpu.assert_called_once_with()
        assert scheduler._safety_placement_wants_off is False
        assert scheduler._safety_placement_promotions == 1

    def test_config_false_never_promotes_to_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ``safety_on_gpu`` off the policy is inert: it never restores safety to the GPU."""
        scheduler = _placement_scheduler(monkeypatch, safety_on_gpu=False)
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._safety_placement_wants_off = True
        scheduler._safety_restore_headroom_fits = lambda device_index: True

        for _ in range(_SAFETY_PLACEMENT_RESTORE_STREAK + 2):
            scheduler._reconcile_runtime_safety_placement()

        scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()
        scheduler._process_lifecycle.pause_safety_on_gpu.assert_not_called()
        assert scheduler._safety_placement_wants_off is False

    def test_deferred_restore_withheld_while_placement_wants_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The residency-drain safety restore does not fight the placement latch back on-GPU."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._safety_placement_wants_off = True

        scheduler._restore_deferred_safety_gpu_load()

        scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()


class TestDemoteThenMeasuredRepromote:
    """Demotion latches the policy off, and a later run of measured-headroom cycles re-promotes safety."""

    def test_demotion_latches_and_measured_headroom_repromotes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The full timeline: modeled non-fit demotes, measured device-free headroom promotes, counters move."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False
        scheduler._safety_restore_headroom_fits = lambda device_index: False

        for _ in range(_SAFETY_PLACEMENT_PAUSE_STREAK):
            scheduler._reconcile_runtime_safety_placement()

        assert scheduler._safety_placement_wants_off is True
        assert scheduler._safety_placement_demotions == 1
        scheduler._process_lifecycle.pause_safety_on_gpu.assert_called_once_with()

        # The pause has taken effect; the card now reports durable measured free between sampling peaks even
        # though the modeled peak (sustained load) still says it does not fit.
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._safety_restore_headroom_fits = lambda device_index: True

        for _ in range(_SAFETY_PLACEMENT_RESTORE_STREAK - 1):
            scheduler._reconcile_runtime_safety_placement()
            scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()

        scheduler._reconcile_runtime_safety_placement()
        scheduler._process_lifecycle.restore_safety_on_gpu.assert_called_once_with()
        assert scheduler._safety_placement_wants_off is False
        assert scheduler._safety_placement_promotions == 1

    def test_transient_measured_headroom_does_not_flap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A single measured-headroom cycle inside a demoted run does not re-promote (hysteresis)."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._safety_placement_wants_off = True
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: False

        headroom_readings = iter([True, False, True, False, True, False])
        scheduler._safety_restore_headroom_fits = lambda device_index: next(headroom_readings)

        for _ in range(6):
            scheduler._reconcile_runtime_safety_placement()

        scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()
        assert scheduler._safety_placement_wants_off is True


class TestHeadroomAwarePlacement:
    """The placement identity chooses the card with the most verified headroom, not a fixed device 0."""

    def _two_card_scheduler(self, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202
        monkeypatch.setattr(sched_mod, "is_cpu_only_install", lambda: False)
        bridge_data = make_mock_bridge_data(safety_on_gpu=True)
        card_runtimes = make_test_card_runtimes(device_indices=(0, 1), mask_kind="cuda")
        scheduler = _make_inference_scheduler(bridge_data=bridge_data, card_runtimes=card_runtimes)
        scheduler._largest_active_sampling_peak_mb = Mock(return_value=4500.0)
        return scheduler

    def test_chooses_card_with_more_measured_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With measured device-free reported per card, the roomier card wins."""
        scheduler = self._two_card_scheduler(monkeypatch)
        free_by_device = {0: 2000.0, 1: 6000.0}
        scheduler._process_map.get_free_vram_mb = Mock(
            side_effect=lambda *, device_index: free_by_device[device_index]
        )
        assert scheduler._choose_safety_gpu_card() == 1

    def test_falls_back_to_total_less_peak_without_measured_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without measured free, the choice is card total less the modeled sampling peak; the larger card wins."""
        scheduler = self._two_card_scheduler(monkeypatch)
        scheduler._process_map.get_free_vram_mb = Mock(return_value=None)
        total_by_device = {0: 8000.0, 1: 24000.0}
        scheduler._process_map.get_reported_total_vram_mb = Mock(
            side_effect=lambda *, device_index: total_by_device[device_index],
        )
        assert scheduler._choose_safety_gpu_card() == 1

    def test_reconcile_pushes_chosen_card_to_lifecycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each reconcile cycle pushes the chosen card to the lifecycle manager so spawn and restore agree."""
        scheduler = self._two_card_scheduler(monkeypatch)
        scheduler._choose_safety_gpu_card = Mock(return_value=1)
        scheduler._process_lifecycle.safety_gpu_card_index = Mock(return_value=None)

        scheduler._reconcile_runtime_safety_placement()

        scheduler._process_lifecycle.set_desired_safety_card.assert_called_with(1)
