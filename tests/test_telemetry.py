"""Semantic tests for the telemetry opt-in / default-off policy.

These pin the *policy* — tracing is off unless explicitly opted in, and the worker hard-overrides
an ambient enable — rather than how the kill switch is wired.
"""

from __future__ import annotations

import os

import pytest

from horde_worker_regen.telemetry import (
    TELEMETRY_OPT_IN_ENV_VAR,
    enforce_telemetry_default_off,
    telemetry_enabled,
)


@pytest.fixture(autouse=True)
def _clean_telemetry_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate telemetry-related environment variables for each test."""
    for var in (
        TELEMETRY_OPT_IN_ENV_VAR,
        "OTEL_SDK_DISABLED",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "LOGFIRE_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


class TestTelemetryPolicy:
    """Tracing is opt-in; the worker forces the OTel SDK off unless explicitly enabled."""

    def test_default_disables_the_sdk(self) -> None:
        """With no opt-in, the OTel SDK is explicitly disabled."""
        enforce_telemetry_default_off()
        assert os.environ["OTEL_SDK_DISABLED"] == "true"

    def test_opt_in_leaves_sdk_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit opt-in does not disable the SDK."""
        monkeypatch.setenv(TELEMETRY_OPT_IN_ENV_VAR, "1")
        enforce_telemetry_default_off()
        assert "OTEL_SDK_DISABLED" not in os.environ

    def test_hard_overrides_ambient_enable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An ambient OTEL_SDK_DISABLED=false is overridden to true when not opted in."""
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        enforce_telemetry_default_off()
        assert os.environ["OTEL_SDK_DISABLED"] == "true"

    def test_opt_in_respects_ambient_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When opted in, an ambient OTEL_SDK_DISABLED is left untouched (export governs)."""
        monkeypatch.setenv(TELEMETRY_OPT_IN_ENV_VAR, "true")
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        enforce_telemetry_default_off()
        assert os.environ["OTEL_SDK_DISABLED"] == "false"

    def test_ambient_otlp_endpoint_does_not_enable_tracing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ambient OTEL_* settings must not implicitly opt in; tracing stays disabled."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        assert telemetry_enabled() is False
        enforce_telemetry_default_off()
        assert os.environ["OTEL_SDK_DISABLED"] == "true"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("false", False), ("", False)],
    )
    def test_opt_in_flag_parsing(self, monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
        """The opt-in flag accepts common truthy spellings, case-insensitively."""
        monkeypatch.setenv(TELEMETRY_OPT_IN_ENV_VAR, value)
        assert telemetry_enabled() is expected
