"""Tests for the spawned-worker startup crash-capture backstops."""

from __future__ import annotations

import faulthandler
from pathlib import Path

import pytest

from horde_worker_regen.process_management.child_crash_capture import (
    enable_child_faulthandler,
    write_startup_crash,
)
from horde_worker_regen.tui.log_tailer import discover_bridge_logs_grouped


def test_write_startup_crash_creates_discoverable_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A startup crash is written to a discoverable logs/bridge_{role}_startup.log with a full traceback."""
    monkeypatch.chdir(tmp_path)
    try:
        raise ImportError("No module named 'diffusers'")
    except ImportError as error:
        write_startup_crash("main", error)

    log_path = tmp_path / "logs" / "bridge_main_startup.log"
    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8")
    # The full exception chain, not just the message, so the operator can see where it failed.
    assert "Traceback (most recent call last):" in contents
    assert "No module named 'diffusers'" in contents
    # The same "| LEVEL |" shape as the other bridge logs so the Logs tab parses a level token.
    assert " | CRITICAL | " in contents

    # The Logs tab globs bridge*.log; the file must show up as its own process entry.
    grouped = discover_bridge_logs_grouped(tmp_path / "logs")
    assert "main_startup" in grouped


def test_write_startup_crash_is_lazy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No file is created unless a crash is actually recorded (so the Logs tab is not cluttered)."""
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "logs" / "bridge_main_startup.log").exists()


def test_write_startup_crash_swallows_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The emergency writer must never raise over the original crash, even if the write itself fails."""
    monkeypatch.chdir(tmp_path)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", _boom)
    # Must not raise.
    write_startup_crash("main", RuntimeError("original"))


def test_enable_child_faulthandler_opens_file_and_enables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The faulthandler backstop opens its per-process file and enables faulthandler without raising."""
    monkeypatch.chdir(tmp_path)
    try:
        enable_child_faulthandler("inference_0")
        assert (tmp_path / "logs" / "bridge_inference_0.faulthandler").exists()
        assert faulthandler.is_enabled()
    finally:
        faulthandler.disable()
