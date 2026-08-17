"""Unit tests for the per-context VRAM overhead measurement model.

The model is a standalone numeric unit (no process map, no running pool), so its derivation rules are
exercised directly here: the configured-override-else-measured-else-zero per-process rule, the clean
idle-residency capture and its min/effective-floor bookkeeping, and the marginal derivation with its
probe-precedence and fallback-to-None behavior.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.scheduling.context_overhead_model import (
    ContextOverheadModel,
    MarginalOverheadBreakdown,
)


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
    """The marginal takes the larger of the probe delta and the idle-residency derivation, else None.

    A high idle floor is kept trustworthy upstream by the invalidation path, so a floor that survives here
    is one the device has not contradicted and may supersede the probe.
    """

    def test_none_without_any_measurement(self) -> None:
        """With nothing measured, no marginal can be derived."""
        model = ContextOverheadModel()
        assert model.marginal_mb(config_override_mb=None) is None

    def test_derived_from_clean_idle_residency(self) -> None:
        """The marginal derives from (idle residency - first-context overhead) / (contexts - 1)."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288)
        model.observe_idle_residency(context_total_mb=4000.0, context_count=4)
        expected = (4000.0 - 1288.0) / 3
        assert model.marginal_mb(config_override_mb=None) == pytest.approx(expected)

    def test_capture_keeps_minimum_clean_baseline(self) -> None:
        """A later, cache-dirtied higher reading must not replace the clean minimum baseline."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000)
        model.observe_idle_residency(context_total_mb=4000.0, context_count=3)
        model.observe_idle_residency(context_total_mb=9000.0, context_count=3)
        assert model._read_residency_scope(None).clean_total_mb == pytest.approx(4000.0)

    def test_effective_floor_keeps_worst_reading(self) -> None:
        """The effective floor keeps the worst (highest) used-VRAM reading at a process count."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000)
        model.observe_idle_residency(context_total_mb=4000.0, context_count=3)
        model.observe_idle_residency(context_total_mb=9000.0, context_count=3)
        assert model._read_residency_scope(None).effective_total_mb == pytest.approx(9000.0)

    def test_sustained_idle_floor_supersedes_probe(self) -> None:
        """An uncontradicted (sustained) idle floor above the probe supersedes it, never under-counting."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000)
        model.set_marginal_overhead_mb(650.0)
        model.observe_idle_residency(context_total_mb=9000.0, context_count=3)
        # No later reading contradicts the floor, so it stands: (9000 - 1000) / 2 = 4000 > the 650 probe.
        breakdown = model.marginal_breakdown(config_override_mb=None)
        assert breakdown.source == "idle_floor"
        assert model.marginal_mb(config_override_mb=None) == pytest.approx(4000.0)

    def test_probe_wins_when_idle_floor_below_it(self) -> None:
        """A probe above the idle-floor derivation stands; the floor never under-cuts the isolated probe."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288)
        model.observe_idle_residency(context_total_mb=4000.0, context_count=4)  # derives ~904
        model.set_marginal_overhead_mb(2000.0)
        breakdown = model.marginal_breakdown(config_override_mb=None)
        assert breakdown.source == "probe"
        assert model.marginal_mb(config_override_mb=None) == pytest.approx(2000.0)
        # A zero/unmeasurable probe falls back to the (uncapped, below-overhead) idle-residency derivation.
        model.set_marginal_overhead_mb(0)
        assert model.marginal_mb(config_override_mb=None) == pytest.approx((4000.0 - 1288.0) / 3)

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
        model.observe_idle_residency(context_total_mb=2000.0, context_count=1)
        assert model.marginal_mb(config_override_mb=None) is None

    def test_residency_at_or_below_overhead_is_inconsistent(self) -> None:
        """A residency at/below the first-context overhead is inconsistent, so no marginal is derived."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(3000)
        model.observe_idle_residency(context_total_mb=2500.0, context_count=3)
        assert model.marginal_mb(config_override_mb=None) is None


class TestMarginalBreakdown:
    """The breakdown reports which signal produced the chosen marginal, for the forecast diagnostics."""

    def test_unmeasured(self) -> None:
        """Nothing measured: chosen is None and the source is ``seeded`` (the forecast applies the seed)."""
        breakdown = ContextOverheadModel().marginal_breakdown(config_override_mb=None)
        assert breakdown == MarginalOverheadBreakdown(None, None, None, "seeded")

    def test_probe_only(self) -> None:
        """Only a probe (no idle reading yet): the probe is reported as the source and value."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288)
        model.set_marginal_overhead_mb(455.0)
        breakdown = model.marginal_breakdown(config_override_mb=None)
        assert breakdown.source == "probe"
        assert breakdown.probe_mb == pytest.approx(455.0)
        assert breakdown.idle_floor_mb is None
        assert breakdown.chosen_mb == pytest.approx(455.0)

    def test_idle_floor_source(self) -> None:
        """No probe and a usable idle reading: the derivation is reported with source ``idle_floor``."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(4000)
        model.observe_idle_residency(context_total_mb=6000.0, context_count=3)  # derives 1000
        breakdown = model.marginal_breakdown(config_override_mb=None)
        assert breakdown.source == "idle_floor"
        assert breakdown.chosen_mb == pytest.approx(1000.0)


class TestEffectiveFloorInvalidation:
    """A latched effective floor is lowered once a later reading proves it was a transient spike."""

    def test_lower_reading_invalidates_latched_floor(self) -> None:
        """A device-wide used reading below the floor (same context count) lowers it, dropping the marginal."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(4266)
        model.set_marginal_overhead_mb(650.0)
        # Latch a transient spike: 4 idle contexts momentarily read ~16.6 GB used (the db0 phantom).
        model.observe_idle_residency(context_total_mb=16642.0, context_count=4)
        assert model.marginal_breakdown(config_override_mb=None).source == "idle_floor"
        # The device later runs at only ~9 GB used with the same contexts live: the spike was reclaimable.
        model.observe_device_residency(context_total_mb=9000.0, context_count=4)
        assert model._read_residency_scope(None).effective_total_mb == pytest.approx(9000.0)
        # The corrected floor (~1578/ctx) no longer dwarfs the 650 probe by a phantom margin.
        assert model.marginal_mb(config_override_mb=None) == pytest.approx((9000.0 - 4266.0) / 3)

    def test_invalidation_ratchets_only_down(self) -> None:
        """A later higher reading does not raise the floor; only lower readings correct it."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000)
        model.observe_idle_residency(context_total_mb=9000.0, context_count=3)
        model.observe_device_residency(context_total_mb=12000.0, context_count=3)
        assert model._read_residency_scope(None).effective_total_mb == pytest.approx(9000.0)

    def test_invalidation_ignores_fewer_live_contexts(self) -> None:
        """A lower reading with fewer contexts live is not comparable and must not lower the floor."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000)
        model.observe_idle_residency(context_total_mb=9000.0, context_count=4)
        model.observe_device_residency(context_total_mb=5000.0, context_count=2)
        assert model._read_residency_scope(None).effective_total_mb == pytest.approx(9000.0)

    def test_invalidation_noop_without_a_latched_floor(self) -> None:
        """With no effective floor yet, an observation is a harmless no-op."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000)
        model.observe_device_residency(context_total_mb=5000.0, context_count=4)
        assert model._read_residency_scope(None).effective_total_mb is None


class TestPerCardFigures:
    """Each card's measured context cost is kept separately, with the worker-wide maximum as the fallback."""

    def test_each_card_reads_its_own_measurement(self) -> None:
        """A card the probe measured is priced from its own figure, not another card's."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288, device_index=0)
        model.set_per_process_overhead_mb(4266, device_index=1)
        model.set_marginal_overhead_mb(455.0, device_index=0)
        model.set_marginal_overhead_mb(650.0, device_index=1)
        assert model.per_process_mb(config_override_mb=None, device_index=0) == pytest.approx(1288.0)
        assert model.per_process_mb(config_override_mb=None, device_index=1) == pytest.approx(4266.0)
        assert model.marginal_mb(config_override_mb=None, device_index=0) == pytest.approx(455.0)
        assert model.marginal_mb(config_override_mb=None, device_index=1) == pytest.approx(650.0)

    def test_worker_wide_is_the_maximum_across_cards(self) -> None:
        """A caller with no card in hand keeps the conservative reduction: the largest card figure."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288, device_index=0)
        model.set_per_process_overhead_mb(4266, device_index=1)
        model.set_marginal_overhead_mb(455.0, device_index=0)
        model.set_marginal_overhead_mb(650.0, device_index=1)
        assert model.per_process_mb(config_override_mb=None) == pytest.approx(4266.0)
        assert model.marginal_mb(config_override_mb=None) == pytest.approx(650.0)

    def test_unmeasured_card_falls_back_to_worker_wide(self) -> None:
        """A card the probe could not measure prices exactly as it did before per-card figures existed."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288, device_index=0)
        model.set_marginal_overhead_mb(455.0, device_index=0)
        assert model.per_process_mb(config_override_mb=None, device_index=3) == pytest.approx(1288.0)
        assert model.marginal_mb(config_override_mb=None, device_index=3) == pytest.approx(455.0)

    def test_worker_wide_measurement_alone_serves_every_card(self) -> None:
        """A probe payload predating per-card figures leaves every card on the single worker-wide value."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288)
        model.set_marginal_overhead_mb(455.0)
        assert model.per_process_mb(config_override_mb=None, device_index=1) == pytest.approx(1288.0)
        assert model.marginal_mb(config_override_mb=None, device_index=1) == pytest.approx(455.0)

    def test_config_override_still_wins_per_card(self) -> None:
        """The operator override is worker-wide policy and outranks any card's measurement."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1288, device_index=0)
        assert model.per_process_mb(config_override_mb=2000.0, device_index=0) == pytest.approx(2000.0)


class TestPerCardIdleResidency:
    """Idle bare-context readings are card-attributable, so each card derives its marginal from its own."""

    def test_each_card_derives_from_its_own_readings(self) -> None:
        """Two cards at different idle residencies yield two different per-context marginals."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000, device_index=0)
        model.set_per_process_overhead_mb(1000, device_index=1)
        model.observe_idle_residency(context_total_mb=3000.0, context_count=3, device_index=0)
        model.observe_idle_residency(context_total_mb=9000.0, context_count=3, device_index=1)
        assert model.marginal_mb(config_override_mb=None, device_index=0) == pytest.approx(1000.0)
        assert model.marginal_mb(config_override_mb=None, device_index=1) == pytest.approx(4000.0)

    def test_card_readings_still_feed_the_worker_wide_series(self) -> None:
        """A card-less caller keeps the pooled behaviour it had before the readings were keyed per card."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000)
        model.observe_idle_residency(context_total_mb=3000.0, context_count=3, device_index=0)
        model.observe_idle_residency(context_total_mb=9000.0, context_count=3, device_index=1)
        assert model._read_residency_scope(None).clean_total_mb == pytest.approx(3000.0)
        assert model._read_residency_scope(None).effective_total_mb == pytest.approx(9000.0)

    def test_card_without_readings_falls_back_to_the_pooled_series(self) -> None:
        """A card whose readings never carried an index still derives from what the worker observed."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000)
        model.observe_idle_residency(context_total_mb=9000.0, context_count=3)
        assert model.marginal_mb(config_override_mb=None, device_index=2) == pytest.approx(4000.0)

    def test_invalidation_lowers_only_the_cards_it_names(self) -> None:
        """A card proving it runs lower corrects its own floor and the pooled one, never a sibling's."""
        model = ContextOverheadModel()
        model.set_per_process_overhead_mb(1000)
        model.observe_idle_residency(context_total_mb=9000.0, context_count=3, device_index=0)
        model.observe_idle_residency(context_total_mb=9000.0, context_count=3, device_index=1)
        model.observe_device_residency(context_total_mb=4000.0, context_count=3, device_index=0)
        assert model._read_residency_scope(0).effective_total_mb == pytest.approx(4000.0)
        assert model._read_residency_scope(1).effective_total_mb == pytest.approx(9000.0)
        assert model._read_residency_scope(None).effective_total_mb == pytest.approx(4000.0)
