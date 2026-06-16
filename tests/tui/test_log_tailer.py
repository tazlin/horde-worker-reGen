"""Tests for bridge-log discovery: grouping by process and current-vs-historical classification."""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.tui.log_tailer import _classify_bridge_log, discover_bridge_logs_grouped


def test_classify_current_main_and_rotated() -> None:
    """The live main log is current; a loguru-rotated dated file is historical for the same process."""
    current = _classify_bridge_log(Path("bridge.log"))
    assert current.process_key == "main"
    assert current.process_label == "bridge (main)"
    assert current.is_current
    assert current.history_label == "current"

    rotated = _classify_bridge_log(Path("bridge.2026-06-15_00-00-00_000000.log"))
    assert rotated.process_key == "main"
    assert not rotated.is_current
    assert rotated.history_label == "2026-06-15_00-00-00_000000"


def test_classify_console_and_subprocess() -> None:
    """The console redirect and numbered subprocess logs get stable keys and friendly labels."""
    console = _classify_bridge_log(Path("bridge_main_console.log"))
    assert console.process_key == "console"
    assert console.process_label == "console output"

    sub = _classify_bridge_log(Path("bridge_1.log"))
    assert sub.process_key == "1"
    assert sub.process_label == "subprocess 1"


def test_grouped_orders_processes_and_buckets_history(tmp_path: Path) -> None:
    """Discovery groups by process (main, console, numbered) with the live file first, history newest-first."""
    for name in (
        "bridge.log",
        "bridge.2026-06-15_00-00-00_000000.log",
        "bridge.2026-06-14_00-00-00_000000.log",
        "bridge_main_console.log",
        "bridge_1.log",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")

    grouped = discover_bridge_logs_grouped(tmp_path)

    assert list(grouped.keys()) == ["main", "console", "1"]
    main = grouped["main"]
    assert main[0].is_current
    assert [entry.history_label for entry in main] == [
        "current",
        "2026-06-15_00-00-00_000000",
        "2026-06-14_00-00-00_000000",
    ]


def test_missing_directory_is_empty() -> None:
    """A missing logs directory yields no groups rather than raising."""
    assert discover_bridge_logs_grouped(Path("does-not-exist-xyz")) == {}
