"""Tests for the StatusReporter's periodic version/update warnings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from loguru import logger

from horde_worker_regen.reporting.status_reporter import StatusReporter
from horde_worker_regen.update_check import NEWER_RELEASE_ENV_VAR


def _stub_bridge_data() -> SimpleNamespace:
    """A minimal duck-typed bridge_data with only the fields _print_warnings reads."""
    return SimpleNamespace(
        extra_slow_worker=False,
        limit_max_steps=False,
        max_batch=1,
        allow_sdxl_controlnet=False,
        max_threads=1,
        minutes_allowed_without_jobs=10,
        suppress_speed_warnings=False,
    )


def _call_print_warnings() -> list[str]:
    """Invoke _print_warnings with benign inputs, capturing the loguru messages it emits."""
    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(message.record["message"]), level="DEBUG")
    try:
        reporter = StatusReporter(0.0, 0.0)
        reporter._print_warnings(
            _stub_bridge_data(),  # type: ignore[arg-type]
            SimpleNamespace(root={}),  # type: ignore[arg-type]
            too_many_consecutive_failed_jobs=False,
            too_many_consecutive_failed_jobs_time=0.0,
            too_many_consecutive_failed_jobs_wait_time=0.0,
            time_spent_no_jobs_available=0.0,
            session_start_time=0.0,
            total_ram_gigabytes=16,
        )
    finally:
        logger.remove(sink_id)
    return captured


def test_newer_release_env_var_triggers_the_nag(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a newer release was found at startup, the periodic report nags with the version and remedy."""
    monkeypatch.delenv("AIWORKER_NOT_REQUIRED_VERSION", raising=False)
    monkeypatch.setenv(NEWER_RELEASE_ENV_VAR, "99.0.0")
    blob = "\n".join(_call_print_warnings())
    assert "newer AI Worker release (v99.0.0)" in blob
    assert "winget upgrade Haidra.HordeWorker" in blob


def test_required_version_takes_precedence_over_newer_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator-controlled required-update warning wins over the softer newer-release nag."""
    monkeypatch.setenv("AIWORKER_NOT_REQUIRED_VERSION", "1")
    monkeypatch.setenv(NEWER_RELEASE_ENV_VAR, "99.0.0")
    blob = "\n".join(_call_print_warnings())
    assert "required update available" in blob
    assert "newer AI Worker release" not in blob


def test_no_version_warning_when_nothing_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither env var set, no update warning is emitted."""
    monkeypatch.delenv("AIWORKER_NOT_REQUIRED_VERSION", raising=False)
    monkeypatch.delenv(NEWER_RELEASE_ENV_VAR, raising=False)
    blob = "\n".join(_call_print_warnings())
    assert "update available" not in blob
    assert "newer AI Worker release" not in blob
