"""Config and spawn-env tests for the component-cache budget.

The ``component_cache_budget_mb`` field opts a worker into hordelib's MB-budgeted component cache and is
exported to the GPU children as ``HORDE_COMPONENT_CACHE_MB`` at env-load time. None (the default) leaves the
env unset so children keep the legacy single-slot cache.
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from horde_worker_regen.bridge_data.data_model import reGenBridgeData

_ENV_VAR = "HORDE_COMPONENT_CACHE_MB"


class TestFieldValidation:
    """The budget field defaults to legacy and rejects a negative value."""

    def test_default_is_none_legacy(self) -> None:
        """Unset means legacy single-slot: the field defaults to None."""
        assert reGenBridgeData.model_fields["component_cache_budget_mb"].default is None
        assert reGenBridgeData.model_validate({}).component_cache_budget_mb is None

    def test_accepts_a_positive_budget(self) -> None:
        """An operator opts in with a megabyte budget."""
        assert reGenBridgeData.model_validate({"component_cache_budget_mb": 12288}).component_cache_budget_mb == 12288

    def test_zero_is_allowed_equivalent_to_legacy(self) -> None:
        """Zero is explicitly permitted (hordelib treats it as single-slot)."""
        assert reGenBridgeData.model_validate({"component_cache_budget_mb": 0}).component_cache_budget_mb == 0

    def test_negative_budget_is_rejected(self) -> None:
        """A negative budget is nonsensical and fails validation (ge=0)."""
        with pytest.raises(ValidationError):
            reGenBridgeData.model_validate({"component_cache_budget_mb": -1})


class TestEnvInjection:
    """load_env_vars exports the budget to children only when the operator opted in."""

    def test_set_budget_is_exported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A configured budget is exported as the child-facing env var."""
        monkeypatch.delenv(_ENV_VAR, raising=False)
        bridge_data = reGenBridgeData.model_validate({"component_cache_budget_mb": 12288})

        bridge_data.load_env_vars()

        assert os.environ.get(_ENV_VAR) == "12288"

    def test_none_budget_leaves_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The legacy default never sets the env, so children keep the single-slot cache."""
        monkeypatch.delenv(_ENV_VAR, raising=False)
        bridge_data = reGenBridgeData.model_validate({})

        bridge_data.load_env_vars()

        assert _ENV_VAR not in os.environ

    def test_preexisting_env_value_is_not_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit shell/Docker value wins over the config field."""
        monkeypatch.setenv(_ENV_VAR, "4096")
        bridge_data = reGenBridgeData.model_validate({"component_cache_budget_mb": 12288})

        bridge_data.load_env_vars()

        assert os.environ.get(_ENV_VAR) == "4096"
