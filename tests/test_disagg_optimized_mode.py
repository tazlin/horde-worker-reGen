"""Tests for ``disaggregation_optimized_mode`` and its bundle application.

Covers the field default, the whole-bundle application when the operator leaves the fields at their default,
each explicitly-set field winning over the bundle, and the warn-once contradiction (mode on while pipeline
disaggregation is explicitly off). The bundle constants are asserted so a future retune is a deliberate,
test-visible change.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger

from horde_worker_regen.bridge_data.data_model import (
    _DISAGGREGATION_OPTIMIZED_BUNDLE,
    apply_disaggregation_optimized_mode,
    reGenBridgeData,
)


@contextmanager
def _captured_logs() -> Iterator[list[str]]:
    """Capture loguru messages emitted within the block into a list of formatted strings."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="DEBUG")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


class TestBundleConstant:
    """The bundle is the measured-good set; its values are pinned so a retune is deliberate."""

    def test_bundle_values(self) -> None:
        """The measured-good bundle is exactly these three values."""
        assert _DISAGGREGATION_OPTIMIZED_BUNDLE == {
            "enable_pipeline_disaggregation": True,
            "pp_overlap_margin_mb_disaggregated": 512.0,
            "component_cache_budget_mb": 12288,
        }


class TestFieldDefault:
    """The mode defaults off, so an untouched config is entirely unchanged."""

    def test_default_is_false(self) -> None:
        """The mode field defaults off."""
        assert reGenBridgeData.model_fields["disaggregation_optimized_mode"].default is False

    def test_off_leaves_bundle_fields_untouched(self) -> None:
        """With the mode off, the three bundled fields keep their own defaults."""
        config = reGenBridgeData.model_validate({})
        assert config.enable_pipeline_disaggregation is False
        assert config.pp_overlap_margin_mb_disaggregated is None
        assert config.component_cache_budget_mb is None


class TestBundleApplication:
    """Mode on applies the bundle only where the operator left the field at its default."""

    def test_all_defaults_get_the_bundle(self) -> None:
        """Mode on with everything at default applies the whole bundle."""
        config = reGenBridgeData.model_validate({"disaggregation_optimized_mode": True})
        assert config.enable_pipeline_disaggregation is True
        assert config.pp_overlap_margin_mb_disaggregated == 512.0
        assert config.component_cache_budget_mb == 12288

    def test_explicit_component_cache_wins(self) -> None:
        """An explicit cache budget is kept while the other fields still get the bundle."""
        config = reGenBridgeData.model_validate(
            {"disaggregation_optimized_mode": True, "component_cache_budget_mb": 4096},
        )
        assert config.component_cache_budget_mb == 4096
        # The untouched fields still receive the bundle.
        assert config.enable_pipeline_disaggregation is True
        assert config.pp_overlap_margin_mb_disaggregated == 512.0

    def test_explicit_pp_margin_wins(self) -> None:
        """An explicit overlap margin is kept while the cache budget still gets the bundle."""
        config = reGenBridgeData.model_validate(
            {"disaggregation_optimized_mode": True, "pp_overlap_margin_mb_disaggregated": 256.0},
        )
        assert config.pp_overlap_margin_mb_disaggregated == 256.0
        assert config.component_cache_budget_mb == 12288

    def test_explicit_enable_true_is_kept_and_others_applied(self) -> None:
        """An explicit enable=true coincides with the bundle and the rest still applies."""
        config = reGenBridgeData.model_validate(
            {"disaggregation_optimized_mode": True, "enable_pipeline_disaggregation": True},
        )
        assert config.enable_pipeline_disaggregation is True
        assert config.component_cache_budget_mb == 12288


class TestContradiction:
    """Mode on with pipeline disaggregation explicitly off keeps the explicit value and warns once."""

    def test_explicit_false_wins_over_mode(self) -> None:
        """An explicit enable=false is kept, and the rest of the bundle still applies."""
        config = reGenBridgeData.model_validate(
            {"disaggregation_optimized_mode": True, "enable_pipeline_disaggregation": False},
        )
        assert config.enable_pipeline_disaggregation is False
        # The rest of the bundle still applies (the operator only pinned the one field).
        assert config.component_cache_budget_mb == 12288

    def test_contradiction_warns_once(self) -> None:
        """The contradiction emits exactly one warning."""
        with _captured_logs() as messages:
            reGenBridgeData.model_validate(
                {"disaggregation_optimized_mode": True, "enable_pipeline_disaggregation": False},
            )
        contradiction_warnings = [
            m for m in messages if "explicitly false" in m and "no jobs will be disaggregated" in m
        ]
        assert len(contradiction_warnings) == 1


class TestApplyHelperDirectly:
    """The pure helper mutates the config per the explicitly-set snapshot it is handed."""

    def test_left_default_field_is_set(self) -> None:
        """A field absent from the snapshot receives the bundle value."""
        config = reGenBridgeData.model_validate({})
        apply_disaggregation_optimized_mode(config, explicitly_set_fields=set(), log=False)
        assert config.component_cache_budget_mb == 12288
        assert config.enable_pipeline_disaggregation is True

    def test_explicitly_set_field_is_left_alone(self) -> None:
        """A field named in the snapshot keeps its value; the others still get the bundle."""
        config = reGenBridgeData.model_validate({"component_cache_budget_mb": 2048})
        apply_disaggregation_optimized_mode(
            config,
            explicitly_set_fields={"component_cache_budget_mb"},
            log=False,
        )
        assert config.component_cache_budget_mb == 2048
        # Fields not in the snapshot still get the bundle.
        assert config.pp_overlap_margin_mb_disaggregated == 512.0
