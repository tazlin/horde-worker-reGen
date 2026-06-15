"""Follow the worker's on-disk logs (logs/bridge.log and logs/bridge_n.log) for the TUI.

A poll-based follower (driven by a Textual interval) rather than a background thread: it tracks a
byte offset, holds back trailing partial lines, and reseeks on rotation (loguru rotates daily) or
truncation. The worker already writes these files, so the TUI adds no new disk writes.
"""

from __future__ import annotations

from pathlib import Path

_TAIL_BYTES = 64 * 1024
"""How much of an existing file to show on first open (roughly the last few hundred lines)."""

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
            # Rotated or truncated; start over from the top of the new file.
            self._signature = signature
            return self._read_from(0, drop_partial_first=False)

        if stat.st_size == self._offset:
            return []
        return self._read_from(self._offset, drop_partial_first=False)

    def _read_from(self, offset: int, *, drop_partial_first: bool) -> list[str]:
        """Read complete lines from ``offset``, holding back any trailing partial line."""
        try:
            with self.path.open("rb") as handle:
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
