"""The runtime safety-placement policy: keep safety off-GPU when its charge cannot fit beside sampling.

The policy generalises the whole-card safety-off lever to the ordinary case, as arithmetic over (device
total, largest learned sampling peak, proportional noise buffer, the static safety charge). It only ever
degrades the operator's placement (GPU to CPU), never promotes it, and its pause/restore is hysteresis-gated
so a card oscillating around the fit boundary does not flap the safety process on and off the card.
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
from tests.process_management.conftest import make_mock_bridge_data
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

    def test_restores_only_after_consecutive_fits_with_margin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A paused-off safety is restored only after the longer run of fitting-with-margin cycles."""
        scheduler = _placement_scheduler(monkeypatch)
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: True
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._safety_placement_wants_off = True

        for _ in range(_SAFETY_PLACEMENT_RESTORE_STREAK - 1):
            scheduler._reconcile_runtime_safety_placement()
            scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()

        scheduler._reconcile_runtime_safety_placement()
        scheduler._process_lifecycle.restore_safety_on_gpu.assert_called_once_with()
        assert scheduler._safety_placement_wants_off is False

    def test_config_false_never_promotes_to_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ``safety_on_gpu`` off the policy is inert: it never restores safety to the GPU."""
        scheduler = _placement_scheduler(monkeypatch, safety_on_gpu=False)
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        scheduler._safety_placement_wants_off = True
        scheduler._safety_fits_beside_largest_sampling_peak = lambda device_index, *, require_margin: True

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
