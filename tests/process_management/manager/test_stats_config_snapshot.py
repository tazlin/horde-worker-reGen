"""The session_start config snapshot and the workload identity it carries."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources.run_metrics import (
    SCENARIO_ID_ENV_VAR,
    SCENARIO_REVISION_ENV_VAR,
)
from tests.process_management.conftest import make_testable_process_manager


def test_snapshot_carries_scenario_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session driven with a workload identity records it alongside the resolved config."""
    monkeypatch.setenv(SCENARIO_ID_ENV_VAR, "pricing_corpus")
    monkeypatch.setenv(SCENARIO_REVISION_ENV_VAR, "3")

    snapshot = make_testable_process_manager()._stats_config_snapshot()

    assert snapshot["scenario_id"] == "pricing_corpus"
    assert snapshot["scenario_revision"] == "3"


def test_snapshot_omits_identity_for_a_production_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker running no scenario emits the config snapshot it always did."""
    monkeypatch.delenv(SCENARIO_ID_ENV_VAR, raising=False)
    monkeypatch.delenv(SCENARIO_REVISION_ENV_VAR, raising=False)

    snapshot = make_testable_process_manager()._stats_config_snapshot()

    assert "scenario_id" not in snapshot
    assert "scenario_revision" not in snapshot


def test_revision_alone_is_recorded_for_an_unversioned_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unversioned scenario still identifies itself; only the missing revision is omitted."""
    monkeypatch.setenv(SCENARIO_ID_ENV_VAR, "pricing_corpus")
    monkeypatch.delenv(SCENARIO_REVISION_ENV_VAR, raising=False)

    snapshot = make_testable_process_manager()._stats_config_snapshot()

    assert snapshot["scenario_id"] == "pricing_corpus"
    assert "scenario_revision" not in snapshot
