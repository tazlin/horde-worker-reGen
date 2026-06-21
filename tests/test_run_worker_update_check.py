"""Tests for the headless worker's startup release-update check."""

from __future__ import annotations

import pytest
from loguru import logger

from horde_worker_regen import run_worker, update_check
from horde_worker_regen.update_check import NEWER_RELEASE_ENV_VAR, UpdateInfo


def _capture(func: object) -> list[str]:
    """Run *func* with a temporary loguru sink and return the messages it logged."""
    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(message.record["message"]), level="DEBUG")
    try:
        func()  # type: ignore[operator]
    finally:
        logger.remove(sink_id)
    return captured


def test_update_check_records_and_logs_a_newer_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newer release sets the nag env var and logs the upgrade guidance."""
    monkeypatch.delenv(NEWER_RELEASE_ENV_VAR, raising=False)
    monkeypatch.setattr(
        update_check,
        "check_for_update",
        lambda *a, **k: UpdateInfo(latest_version="99.0.0", html_url="https://example.test"),
    )
    blob = "\n".join(_capture(run_worker._run_release_update_check))
    assert "Update available" in blob
    assert "v99.0.0" in blob
    import os

    assert os.environ[NEWER_RELEASE_ENV_VAR] == "99.0.0"


def test_update_check_logs_when_current(monkeypatch: pytest.MonkeyPatch) -> None:
    """When up to date, the worker logs a reassuring line and sets no nag env var."""
    monkeypatch.delenv(NEWER_RELEASE_ENV_VAR, raising=False)
    monkeypatch.setattr(update_check, "check_for_update", lambda *a, **k: None)
    blob = "\n".join(_capture(run_worker._run_release_update_check))
    assert "up to date" in blob
    import os

    assert NEWER_RELEASE_ENV_VAR not in os.environ


def test_start_release_check_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check never spawns a thread when disabled (e.g. under the test suite)."""
    started: list[bool] = []
    monkeypatch.setattr(update_check, "update_check_disabled", lambda: True)

    import threading

    monkeypatch.setattr(threading, "Thread", lambda *a, **k: started.append(True))
    run_worker._start_release_update_check()
    assert started == []
