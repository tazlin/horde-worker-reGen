"""Tests for the ``model_stickiness`` -> fixed model pool migration and the inert legacy-key deprecations.

Covers the soft-map bundle constant, the migration firing only when stickiness is positive and the pool is off,
explicit ``model_pool`` values winning over the map, the pool-enabled case ignoring stickiness entirely, the
interaction with ``max_throughput_mode``, the one-time deprecation pointer, and the accepted-and-warned
handling of the inert legacy keys (``dynamic_models``, ``number_of_dynamic_models``, ``max_models_to_download``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from loguru import logger

from horde_worker_regen.bridge_data import data_model as data_model_module
from horde_worker_regen.bridge_data.data_model import (
    _STICKINESS_POOL_MIGRATION_BUNDLE,
    apply_stickiness_pool_migration,
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


@pytest.fixture(autouse=True)
def _reset_deprecation_guard() -> Iterator[None]:
    """Clear the one-time deprecation-warning guard around each test so the warn-once behaviour is observable."""
    data_model_module._warned_deprecated_config_keys.clear()
    try:
        yield
    finally:
        data_model_module._warned_deprecated_config_keys.clear()


class TestBundleConstant:
    """The soft-map bundle is pinned so a change to what a legacy stickiness maps onto is deliberate."""

    def test_bundle_values(self) -> None:
        """The migration bundle is exactly these three model_pool sub-field values."""
        assert _STICKINESS_POOL_MIGRATION_BUNDLE == {
            "enabled": True,
            "ranker_enabled": True,
            "rotation_minutes": 30.0,
        }


class TestMigrationFires:
    """A positive stickiness with the pool off is mapped onto a modest pool."""

    def test_stickiness_maps_pool_on(self) -> None:
        """A positive model_stickiness with no pool config enables a modest pool."""
        config = reGenBridgeData.model_validate({"model_stickiness": 0.4})
        assert config.model_pool.enabled is True
        assert config.model_pool.ranker_enabled is True
        assert config.model_pool.rotation_minutes == 30.0

    def test_zero_stickiness_does_not_migrate(self) -> None:
        """The default (zero) stickiness leaves the pool untouched."""
        config = reGenBridgeData.model_validate({})
        assert config.model_pool.enabled is False


class TestExplicitPoolWins:
    """The operator's explicit model_pool values always win over the map, same as the preset bundles."""

    def test_explicit_pool_disabled_stays_off(self) -> None:
        """An explicit model_pool.enabled=false is honoured, so the map leaves the pool off."""
        config = reGenBridgeData.model_validate(
            {"model_stickiness": 0.4, "model_pool": {"enabled": False}},
        )
        assert config.model_pool.enabled is False

    def test_explicit_rotation_is_kept(self) -> None:
        """An explicit rotation_minutes is kept while the map still enables the pool."""
        config = reGenBridgeData.model_validate(
            {"model_stickiness": 0.4, "model_pool": {"rotation_minutes": 90.0}},
        )
        assert config.model_pool.enabled is True
        assert config.model_pool.rotation_minutes == 90.0


class TestPoolAlreadyEnabled:
    """When the pool is already on, stickiness is ignored entirely and nothing is mapped or warned."""

    def test_pool_enabled_ignores_stickiness(self) -> None:
        """A pool the operator already enabled is not re-touched by the stickiness map."""
        config = reGenBridgeData.model_validate(
            {"model_stickiness": 0.4, "model_pool": {"enabled": True, "rotation_minutes": 45.0}},
        )
        assert config.model_pool.rotation_minutes == 45.0

    def test_pool_enabled_emits_no_deprecation_pointer(self) -> None:
        """With the pool already on, no stickiness deprecation pointer is logged."""
        with _captured_logs() as messages:
            reGenBridgeData.model_validate(
                {"model_stickiness": 0.4, "model_pool": {"enabled": True}},
            )
        assert not [m for m in messages if "model_stickiness" in m and "deprecated" in m]

    def test_max_throughput_mode_takes_precedence(self) -> None:
        """max_throughput_mode enables the pool first, so the stickiness map does not override its rotation."""
        config = reGenBridgeData.model_validate(
            {"model_stickiness": 0.4, "max_throughput_mode": True},
        )
        assert config.model_pool.enabled is True
        # The stickiness map did not run, so rotation stays the pool default rather than the 30.0 map value.
        assert config.model_pool.rotation_minutes == 60.0
        assert config.model_pool.download_budget_gb == 50.0


class TestDeprecationPointerOnce:
    """The stickiness deprecation pointer fires exactly once per process."""

    def test_pointer_logged_once(self) -> None:
        """Two validations of a stickiness config log the pointer only on the first."""
        with _captured_logs() as messages:
            reGenBridgeData.model_validate({"model_stickiness": 0.4})
            reGenBridgeData.model_validate({"model_stickiness": 0.4})
        pointers = [m for m in messages if "`model_stickiness` is deprecated" in m]
        assert len(pointers) == 1


class TestInertLegacyKeys:
    """The inert dynamic-model keys are accepted and draw a one-time pointer at the model pool."""

    @pytest.mark.parametrize(
        ("config_key", "pointer_fragment"),
        [
            ("dynamic_models", "`dynamic_models` is inert"),
            ("number_of_dynamic_models", "`number_of_dynamic_models` is inert"),
            ("max_models_to_download", "`max_models_to_download` is inert"),
        ],
    )
    def test_setting_inert_key_warns_once(self, config_key: str, pointer_fragment: str) -> None:
        """Setting an inert legacy key loads fine and logs its pointer exactly once across two loads."""
        with _captured_logs() as messages:
            first = reGenBridgeData.model_validate({config_key: 1})
            reGenBridgeData.model_validate({config_key: 1})
        assert first is not None
        pointers = [m for m in messages if pointer_fragment in m]
        assert len(pointers) == 1

    def test_unset_inert_keys_are_silent(self) -> None:
        """A config that never sets the legacy keys draws no inert-key pointers."""
        with _captured_logs() as messages:
            reGenBridgeData.model_validate({})
        assert not [m for m in messages if "is inert in reGen" in m]


class TestApplyHelperDirectly:
    """The pure helper mutates the pool per the explicitly-set snapshot it is handed."""

    def test_left_default_sub_field_is_set(self) -> None:
        """A sub-field absent from the snapshot receives the bundle value."""
        config = reGenBridgeData.model_validate({})
        apply_stickiness_pool_migration(config, explicitly_set_pool_fields=set(), log=False)
        assert config.model_pool.enabled is True
        assert config.model_pool.rotation_minutes == 30.0

    def test_explicitly_set_sub_field_is_left_alone(self) -> None:
        """A sub-field named in the snapshot keeps its value; the others still get the bundle."""
        config = reGenBridgeData.model_validate({"model_pool": {"rotation_minutes": 90.0}})
        apply_stickiness_pool_migration(config, explicitly_set_pool_fields={"rotation_minutes"}, log=False)
        assert config.model_pool.rotation_minutes == 90.0
        assert config.model_pool.enabled is True
