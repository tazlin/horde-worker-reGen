"""Tests for bridge-log discovery: grouping by process and current-vs-historical classification."""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.tui.log_tailer import (
    LogFollower,
    _classify_bridge_log,
    discover_bridge_logs_grouped,
)


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


_TAIL = 1024
"""A small tail cap for deterministic bounded-read assertions."""


def _write_lines(path: Path, count: int, *, start: int = 0, prefix: str = "line") -> None:
    """Write ``count`` newline-terminated lines, each comfortably larger than a few bytes."""
    body = "".join(f"{prefix} {i:06d} {'x' * 40}\n" for i in range(start, start + count))
    path.write_text(body, encoding="utf-8")


def _bytes_of(lines: list[str]) -> int:
    """Total bytes the returned lines represent (newlines included)."""
    return sum(len(line) + 1 for line in lines)


def test_first_poll_is_bounded_to_tail(tmp_path: Path) -> None:
    """Opening a file far larger than the cap reads only the trailing window, not the whole file."""
    path = tmp_path / "bridge.log"
    _write_lines(path, 2000)  # ~95KB, well over the 1KB test cap
    follower = LogFollower(path, tail_bytes=_TAIL)

    lines = follower.poll()

    assert _bytes_of(lines) <= _TAIL
    assert lines, "the tail should still yield at least one complete line"
    assert lines[-1].endswith("x" * 40), "the newest line must be present (we read the tail, not the head)"
    assert "line 000000" not in lines[0], "the head of a large file must not be loaded"


def test_catch_up_burst_is_clamped_to_tail(tmp_path: Path) -> None:
    """A large append between polls is clamped: a single poll never buffers more than the cap."""
    path = tmp_path / "bridge.log"
    _write_lines(path, 1)
    follower = LogFollower(path, tail_bytes=_TAIL)
    follower.poll()  # prime at end of the tiny file

    _write_lines(path, 5000, start=1)  # a big burst, far over the cap
    lines = follower.poll()

    assert _bytes_of(lines) <= _TAIL
    assert lines[-1].endswith("x" * 40)


def test_truncation_reprimes_tail_not_whole_file(tmp_path: Path) -> None:
    """When the file shrinks (rotation/truncation), the re-prime stays bounded by the tail cap."""
    path = tmp_path / "bridge.log"
    _write_lines(path, 4000)  # large
    follower = LogFollower(path, tail_bytes=_TAIL)
    follower.poll()  # advances offset to near EOF of the large file

    # Replace with a still-large-but-smaller file so size < offset triggers the rotation branch.
    _write_lines(path, 2000, prefix="rot")
    lines = follower.poll()

    assert _bytes_of(lines) <= _TAIL, "the rotation branch must not read the whole new file"
    assert lines and "rot" in lines[-1]
