"""Tests for the supervisor process's own on-disk log sink."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from loguru import logger

from horde_worker_regen.tui.log_tailer import discover_bridge_logs_grouped
from horde_worker_regen.tui.logging_setup import setup_supervisor_file_logging


def test_writes_discoverable_role_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The sink creates logs/bridge_{role}.log, captures messages, and the Logs tab discovers it."""
    monkeypatch.chdir(tmp_path)
    sink_id = setup_supervisor_file_logging("tui")
    assert sink_id is not None
    try:
        logger.info("supervisor launched the worker")
    finally:
        logger.remove(sink_id)

    log_path = tmp_path / "logs" / "bridge_tui.log"
    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8")
    assert "supervisor launched the worker" in contents
    # Plain format the Logs tab parses for a level token: "<time> | <LEVEL> | ...".
    assert " | INFO     | " in contents

    grouped = discover_bridge_logs_grouped(tmp_path / "logs")
    assert "tui" in grouped
    assert grouped["tui"][0].is_current


def test_quiet_console_removes_default_stderr_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """quiet_console drops the default stderr sink so a full-screen TUI cannot be corrupted by log writes."""
    monkeypatch.chdir(tmp_path)
    # A known default stderr handler stands in for whatever the process started with.
    default_id = logger.add(sys.stderr)
    sink_id = setup_supervisor_file_logging("tui", quiet_console=True)
    try:
        assert sink_id is not None
        # The pre-existing stderr handler was removed by quiet_console; removing it again raises.
        with pytest.raises(ValueError):
            logger.remove(default_id)
    finally:
        logger.remove(sink_id)
        logger.add(sys.stderr)


def test_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A sink that cannot be created returns None rather than blocking the supervisor from starting."""
    monkeypatch.chdir(tmp_path)

    def _boom(*_args: object, **_kwargs: object) -> int:
        raise OSError("disk full")

    monkeypatch.setattr("horde_worker_regen.tui.logging_setup.logger.add", _boom)
    assert setup_supervisor_file_logging("tui") is None
