"""Tests for ``max_throughput_mode`` and its ``model_pool`` sub-field bundle application.

Covers the field default, the whole-bundle application when the operator leaves the pool at its defaults, each
explicitly-set sub-field winning over the bundle, and the warn-once contradiction (mode on while the pool is
explicitly disabled). The bundle constants are asserted so a future retune is a deliberate, test-visible
change.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger

from horde_worker_regen.bridge_data.data_model import (
    _MAX_THROUGHPUT_MODE_BUNDLE,
    apply_max_throughput_mode,
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
    """The bundle is the throughput preset; its values are pinned so a retune is deliberate."""

    def test_bundle_values(self) -> None:
        """The preset bundle is exactly these three model_pool sub-field values."""
        assert _MAX_THROUGHPUT_MODE_BUNDLE == {
            "enabled": True,
            "ranker_enabled": True,
            "download_budget_gb": 50.0,
        }


class TestFieldDefault:
    """The mode defaults off, so an untouched config leaves the pool at its own defaults."""

    def test_default_is_false(self) -> None:
        """The mode field defaults off."""
        assert reGenBridgeData.model_fields["max_throughput_mode"].default is False

    def test_off_leaves_pool_at_defaults(self) -> None:
        """With the mode off, the pool keeps its own defaults (disabled, no download budget)."""
        config = reGenBridgeData.model_validate({})
        assert config.model_pool.enabled is False
        assert config.model_pool.download_budget_gb == 0.0


class TestBundleApplication:
    """Mode on applies the bundle only where the operator left the sub-field at its default."""

    def test_all_defaults_get_the_bundle(self) -> None:
        """Mode on with the pool untouched applies the whole bundle."""
        config = reGenBridgeData.model_validate({"max_throughput_mode": True})
        assert config.model_pool.enabled is True
        assert config.model_pool.ranker_enabled is True
        assert config.model_pool.download_budget_gb == 50.0

    def test_explicit_download_budget_wins(self) -> None:
        """An explicit download budget is kept while the pool is still enabled by the bundle."""
        config = reGenBridgeData.model_validate(
            {"max_throughput_mode": True, "model_pool": {"download_budget_gb": 10.0}},
        )
        assert config.model_pool.download_budget_gb == 10.0
        assert config.model_pool.enabled is True

    def test_explicit_ranker_disabled_wins(self) -> None:
        """An explicit ranker_enabled=false is kept while the pool is still enabled and given a budget."""
        config = reGenBridgeData.model_validate(
            {"max_throughput_mode": True, "model_pool": {"ranker_enabled": False}},
        )
        assert config.model_pool.ranker_enabled is False
        assert config.model_pool.enabled is True
        assert config.model_pool.download_budget_gb == 50.0


class TestContradiction:
    """Mode on with the pool explicitly disabled keeps the explicit value and warns once."""

    def test_explicit_disabled_wins_over_mode(self) -> None:
        """An explicit enabled=false is kept, though the rest of the bundle still applies."""
        config = reGenBridgeData.model_validate(
            {"max_throughput_mode": True, "model_pool": {"enabled": False}},
        )
        assert config.model_pool.enabled is False
        assert config.model_pool.download_budget_gb == 50.0

    def test_contradiction_warns_once(self) -> None:
        """The contradiction emits exactly one warning."""
        with _captured_logs() as messages:
            reGenBridgeData.model_validate(
                {"max_throughput_mode": True, "model_pool": {"enabled": False}},
            )
        contradiction_warnings = [m for m in messages if "explicitly false" in m and "does nothing" in m]
        assert len(contradiction_warnings) == 1


class TestApplyHelperDirectly:
    """The pure helper mutates the pool per the explicitly-set snapshot it is handed."""

    def test_left_default_sub_field_is_set(self) -> None:
        """A sub-field absent from the snapshot receives the bundle value."""
        config = reGenBridgeData.model_validate({})
        apply_max_throughput_mode(config, explicitly_set_pool_fields=set(), log=False)
        assert config.model_pool.enabled is True
        assert config.model_pool.download_budget_gb == 50.0

    def test_explicitly_set_sub_field_is_left_alone(self) -> None:
        """A sub-field named in the snapshot keeps its value; the others still get the bundle."""
        config = reGenBridgeData.model_validate({"model_pool": {"download_budget_gb": 5.0}})
        apply_max_throughput_mode(config, explicitly_set_pool_fields={"download_budget_gb"}, log=False)
        assert config.model_pool.download_budget_gb == 5.0
        assert config.model_pool.enabled is True
