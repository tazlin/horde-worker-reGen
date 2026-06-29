"""Unit tests for the per-context VRAM overhead measurement model.

The model is a standalone numeric unit (no process map, no running pool), so its derivation rules are
exercised directly here: the configured-override-else-measured-else-zero per-process rule, the clean
idle-residency capture and its min/effective-floor bookkeeping, and the marginal derivation with its
probe-precedence and fallback-to-None behavior.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.scheduling.context_overhead_model import ContextOverheadModel


class TestPerProcessOverhead:
    """The per-process overhead resolves the configured override, else the measured figure, else zero."""

    def test_zero_until_measured(self) -> None:
        """An unmeasured model reports zero per-process overhead."""
        model = ContextOverheadModel()
        assert model.per_process_mb(config_override_mb=None) == 0.0

    def test_measured_used_when_no_override(self) -> None:
        """With no override, the startup-measured figure is reported."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288)
        assert model.per_process_mb(config_override_mb=None) == pytest.approx(1288.0)

    def test_positive_override_wins_over_measured(self) -> None:
        """A positive operator override supersedes the measured figure."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288)
        assert model.per_process_mb(config_override_mb=2000.0) == pytest.approx(2000.0)

    def test_nonpositive_override_falls_back_to_measured(self) -> None:
        """A zero/negative override is not a real tuning, so the measured figure stands."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288)
        assert model.per_process_mb(config_override_mb=0.0) == pytest.approx(1288.0)
        assert model.per_process_mb(config_override_mb=-5.0) == pytest.approx(1288.0)

    def test_negative_measurement_is_ignored(self) -> None:
        """A negative measured figure is rejected, leaving the default."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(-1.0)
        assert model.per_process_mb(config_override_mb=None) == 0.0

    def test_non_numeric_measurement_is_ignored(self) -> None:
        """Partially-mocked startup figures (a Mock, a bool) coerce to unset and leave the default."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(Mock())  # type: ignore[arg-type]
        model.set_per_process_overhead_mb(True)
        assert model.per_process_mb(config_override_mb=None) == 0.0


class TestMarginalDerivation:
    """The marginal cost prefers the probe delta, else derives it from a clean idle residency, else None."""

    def test_none_without_any_measurement(self) -> None:
        """With nothing measured, no marginal can be derived."""
        model = ContextOverheadModel()
        assert model.marginal_mb(config_override_mb=None) is None

    def test_derived_from_clean_idle_residency(self) -> None:
        """The marginal derives from (idle residency - first-context overhead) / (contexts - 1)."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288)
        model.observe_idle_residency(used_mb=4000.0, idle_inference_process_count=4)
        expected = (4000.0 - 1288.0) / 3
        assert model.marginal_mb(config_override_mb=None) == pytest.approx(expected)

    def test_capture_keeps_minimum_clean_baseline(self) -> None:
        """A later, cache-dirtied higher reading must not replace the clean minimum baseline."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000)
        model.observe_idle_residency(used_mb=4000.0, idle_inference_process_count=3)
        model.observe_idle_residency(used_mb=9000.0, idle_inference_process_count=3)
        assert model._idle_context_residency_mb == pytest.approx(4000.0)

    def test_effective_floor_keeps_worst_reading(self) -> None:
        """The effective floor keeps the worst (highest) reading at a count and supersedes the probe derivation."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000)
        model.observe_idle_residency(used_mb=4000.0, idle_inference_process_count=3)
        model.observe_idle_residency(used_mb=9000.0, idle_inference_process_count=3)
        assert model._effective_idle_used_mb == pytest.approx(9000.0)
        # Once the effective floor exceeds the probe derivation, it supersedes it (allocator retains cache).
        derived_from_floor = (9000.0 - 1000.0) / 2
        assert model.marginal_mb(config_override_mb=None) == pytest.approx(derived_from_floor)

    def test_marginal_is_max_of_probe_and_derivation(self) -> None:
        """The marginal never under-counts: it takes the larger of the probe and the floor derivation.

        A probe above the derivation wins (it is the larger, never-under-count estimate); a zero/unmeasurable
        probe falls back to the derivation.
        """
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288)
        model.observe_idle_residency(used_mb=4000.0, idle_inference_process_count=4)
        derived = (4000.0 - 1288.0) / 3
        # A probe above the derivation supersedes it.
        model.set_marginal_overhead_mb(2000.0)
        assert model.marginal_mb(config_override_mb=None) == pytest.approx(2000.0)
        # A zero/unmeasurable probe falls back to the idle-residency derivation.
        model.set_marginal_overhead_mb(0)
        assert model.marginal_mb(config_override_mb=None) == pytest.approx(derived)

    def test_probe_alone_covers_startup_window(self) -> None:
        """With only the probe delta (no idle baseline yet), the marginal is available at startup."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288)
        model.set_marginal_overhead_mb(455.0)
        assert model.marginal_mb(config_override_mb=None) == pytest.approx(455.0)

    def test_single_process_yields_no_derivation(self) -> None:
        """One context has no additional-context cost to derive (count - 1 == 0)."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288)
        model.observe_idle_residency(used_mb=2000.0, idle_inference_process_count=1)
        assert model.marginal_mb(config_override_mb=None) is None

    def test_residency_at_or_below_overhead_is_inconsistent(self) -> None:
        """A residency at/below the first-context overhead is inconsistent, so no marginal is derived."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(3000)
        model.observe_idle_residency(used_mb=2500.0, idle_inference_process_count=3)
        assert model.marginal_mb(config_override_mb=None) is None
