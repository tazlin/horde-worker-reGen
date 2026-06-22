"""Follow the worker's on-disk logs (logs/bridge.log and logs/bridge_n.log) for the TUI.

A poll-based follower (driven by a Textual interval) rather than a background thread: it tracks a
byte offset, holds back trailing partial lines, and reseeks on rotation (loguru rotates daily) or
truncation. The worker already writes these files, so the TUI adds no new disk writes.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

_TAIL_BYTES = 256 * 1024
"""Hard ceiling on how many trailing bytes a single read may pull into memory.

Applied to *every* read path (first open, rotation/truncation re-prime, and bursty catch-up), not
just first open, so a multi-GB log can never be buffered whole and freeze/crash the TUI. It is an
absolute byte cap, not a percentage of the file, so the bound holds regardless of file size."""

_MAX_LINES_PER_POLL = 1000
"""Cap lines emitted per poll so a rotation or large catch-up can't flood the UI."""

LOG_DIR = Path("logs")


def discover_bridge_logs(log_dir: Path = LOG_DIR) -> list[Path]:
    """Return the main bridge log followed by any subprocess bridge logs, in stable order."""
    if not log_dir.exists():
        return []
    result: list[Path] = []
    main = log_dir / "bridge.log"
    if main.exists():
        result.append(main)
    result.extend(sorted(log_dir.glob("bridge_*.log"), key=lambda path: path.name))
    return result


_DATE_TOKEN = re.compile(
    r"[._](?P<date>\d{4}-\d{2}-\d{2}(?:[ _T]\d{2}[-:]\d{2}[-:]\d{2})?(?:[_.]\d+)?)$",
)
"""A trailing loguru rotation timestamp on a log stem (dot- or underscore-separated)."""


@dataclasses.dataclass(frozen=True)
class BridgeLog:
    """One discovered bridge log file, classified by which process wrote it and whether it is current."""

    path: Path
    process_key: str
    """A stable key identifying the writing process ("main", "console", "1", "safety", ...)."""
    process_label: str
    """A friendly label for the process selector ("bridge (main)", "console output", ...)."""
    is_current: bool
    """True for the live (un-rotated) file; False for a rotated, dated historical file."""
    history_label: str
    """``"current"`` for the live file, otherwise the rotation date token (for the history selector)."""


def _process_identity(stem: str) -> tuple[str, str]:
    """Map a date-stripped bridge stem to a (stable key, friendly label) for its writing process."""
    if stem == "bridge":
        return "main", "bridge (main)"
    rest = stem[len("bridge_") :] if stem.startswith("bridge_") else stem
    if rest in ("main_console", "console"):
        return "console", "console output"
    if rest.isdigit():
        return rest, f"subprocess {rest}"
    return rest, rest.replace("_", " ")


def _classify_bridge_log(path: Path) -> BridgeLog:
    """Classify one ``bridge*.log`` path by process and current-vs-rotated, stripping any date token."""
    stem = path.name[: -len(".log")] if path.name.endswith(".log") else path.name
    date_match = _DATE_TOKEN.search(stem)
    if date_match is not None:
        history_label = date_match.group("date")
        stem = stem[: date_match.start()]
    else:
        history_label = "current"
    key, label = _process_identity(stem)
    return BridgeLog(path, key, label, date_match is None, history_label)


def _process_sort_key(process_key: str) -> tuple[int, int, str]:
    """Order process keys for the selector: main, console, numbered subprocesses, then the rest."""
    if process_key == "main":
        return (0, 0, "")
    if process_key == "console":
        return (1, 0, "")
    if process_key.isdigit():
        return (2, int(process_key), "")
    return (3, 0, process_key)


def discover_bridge_logs_grouped(log_dir: Path = LOG_DIR) -> dict[str, list[BridgeLog]]:
    """Group all ``bridge*.log`` files by writing process.

    Returns an ordered mapping (main, console, numbered subprocesses, then others) of process key to
    that process's files: the live file first, then rotated historical files newest-first. This lets
    the logs view present one entry per process and tuck the dated rotations behind a history selector
    instead of listing every rotated file as a confusing top-level peer.
    """
    if not log_dir.exists():
        return {}
    grouped: dict[str, list[BridgeLog]] = {}
    for path in log_dir.glob("bridge*.log"):
        entry = _classify_bridge_log(path)
        grouped.setdefault(entry.process_key, []).append(entry)

    for entries in grouped.values():
        entries.sort(key=lambda entry: entry.history_label, reverse=True)  # newest date first
        entries.sort(key=lambda entry: not entry.is_current)  # stable: the live file leads

    return {key: grouped[key] for key in sorted(grouped, key=_process_sort_key)}


class LogFollower:
    """Incrementally yields new complete lines from a single growing/rotating log file."""

    def __init__(self, path: Path, *, tail_bytes: int = _TAIL_BYTES) -> None:
        """Create a follower for ``path`` (not opened until the first :meth:`poll`)."""
        self.path = path
        self._tail_bytes = tail_bytes
        self._offset = 0
        self._signature: tuple[int, int] | None = None
        self._primed = False

    @staticmethod
    def _signature_of(stat: object) -> tuple[int, int]:
        """A best-effort identity for rotation detection (inode + device)."""
        return (getattr(stat, "st_ino", 0), getattr(stat, "st_dev", 0))

    def poll(self) -> list[str]:
        """Return complete lines appended since the previous poll (priming the tail on first call)."""
        try:
            stat = self.path.stat()
        except OSError:
            return []

        signature = self._signature_of(stat)

        if not self._primed:
            self._primed = True
            self._signature = signature
            start = max(0, stat.st_size - self._tail_bytes)
            return self._read_from(start, drop_partial_first=start > 0)

        if signature != self._signature or stat.st_size < self._offset:
            # Rotated or truncated; re-prime the tail of the new file rather than reading it whole.
            # A rotated-in file can already be large (or this can fire on a misdetected huge current
            # file), so reading from offset 0 here was the one unbounded path that could OOM/freeze.
            self._signature = signature
            start = max(0, stat.st_size - self._tail_bytes)
            return self._read_from(start, drop_partial_first=start > 0)

        if stat.st_size == self._offset:
            return []
        return self._read_from(self._offset, drop_partial_first=False)

    def _read_from(self, offset: int, *, drop_partial_first: bool) -> list[str]:
        """Read complete lines from ``offset``, holding back any trailing partial line.

        The read is clamped to the trailing ``_tail_bytes`` of the file: if ``offset`` lags EOF by
        more than the cap (a stalled or bursty catch-up), it is advanced so a single poll can never
        buffer more than the cap into memory. ``drop_partial_first`` is forced on when the clamp
        moves the start forward, since the first line is then almost certainly mid-line.
        """
        try:
            with self.path.open("rb") as handle:
                handle.seek(0, 2)  # SEEK_END
                size = handle.tell()
                if size - offset > self._tail_bytes:
                    offset = size - self._tail_bytes
                    drop_partial_first = True
                handle.seek(offset)
                data = handle.read()
        except OSError:
            return []

        last_newline = data.rfind(b"\n")
        if last_newline == -1:
            # No complete line available yet; wait for more without advancing.
            self._offset = offset
            return []

        complete = data[: last_newline + 1]
        self._offset = offset + len(complete)

        lines = complete.decode("utf-8", errors="replace").splitlines()
        if drop_partial_first and lines:
            lines = lines[1:]
        if len(lines) > _MAX_LINES_PER_POLL:
            lines = lines[-_MAX_LINES_PER_POLL:]
        return lines
