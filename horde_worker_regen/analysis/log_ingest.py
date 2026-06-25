"""Read worker log files into normalized records, tolerant of how the worker actually writes them.

The worker's loguru sinks all share one line format (see ``hordelib.utils.logger`` and
``tui.logging_setup``)::

    {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}

That grep-friendly format is the contract this module parses. Three real-world wrinkles shape the
reader:

* ``bridge.log`` is written by several processes through an ``enqueue=True`` queue, which under
  contention leaves NUL bytes and the occasional torn line in the file. We strip NUL and treat any
  line without a timestamp prefix as a continuation of the preceding record (which is also how
  multi-line tracebacks arrive), rather than discarding it.
* Sinks rotate at 25 MB with ``compression="zip"``, so a full history is the active ``*.log`` plus a
  spray of ``*.log.zip`` (and occasionally ``*.gz``) archives. :func:`read_records` reads all of them.
* The pre-sink crash backstop (``bridge_<role>_startup.log``) logs a shorter ``{role}:startup`` location
  with no line number, so location parsing degrades instead of failing.

Pure stdlib so it imports without the inference stack (the torch-free orchestrator invariant), and so
the duty-cycle report can share its timestamp primitives.
"""

from __future__ import annotations

import gzip
import re
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# The loguru timestamp prefix, e.g. "2026-06-24 18:30:38.847". Shared with duty_log_report.
TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# A full record head: "<ts> | <LEVEL> | <location> - <message>". The level is space-padded to 8 and the
# location is a single whitespace-free token (a dotted module path plus function and optional line).
_HEAD_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) \| "
    r"(?P<level>\w+)\s+\| "
    r"(?P<loc>\S+) - "
    r"(?P<message>.*)$",
)


def parse_ts(text: str) -> datetime | None:
    """Parse a loguru timestamp prefix from the start of ``text``, or None if it has none."""
    match = TS_RE.match(text)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group("ts"), TS_FORMAT)
    except ValueError:
        return None


@dataclass
class LogRecord:
    """One parsed log line plus any continuation (traceback) lines folded into it."""

    timestamp: datetime | None
    level: str
    name: str
    """The logger name (dotted module path), e.g. ``horde_worker_regen.process_management.process_lifecycle``."""
    function: str
    lineno: int | None
    message: str
    source_path: Path
    raw_lineno: int
    """1-based physical line number within ``source_path`` (its uncompressed member), for evidence refs."""
    continuation: list[str] = field(default_factory=list)
    """Following lines with no timestamp prefix (e.g. a Python traceback) that belong to this record."""

    @property
    def location(self) -> str:
        """The ``name:function:lineno`` location string as originally logged."""
        if self.lineno is None:
            return f"{self.name}:{self.function}"
        return f"{self.name}:{self.function}:{self.lineno}"

    @property
    def full_text(self) -> str:
        """The message plus any continuation lines, joined with newlines."""
        if not self.continuation:
            return self.message
        return "\n".join([self.message, *self.continuation])


def _split_location(loc: str) -> tuple[str, str, int | None]:
    """Split a ``name:function:line`` (or ``name:function``) location into its parts.

    The startup crash backstop logs ``<role>:startup`` with no line number, so a two-part location is
    valid; anything else degrades to (loc, "", None) rather than raising.
    """
    parts = loc.rsplit(":", 2)
    if len(parts) == 3 and parts[2].isdigit():
        return parts[0], parts[1], int(parts[2])
    if len(parts) >= 2:
        return parts[0], parts[1], None
    return loc, "", None


def _read_physical_lines(path: Path) -> Iterator[str]:
    """Yield decoded, NUL-stripped physical lines from a ``.log``, ``.zip``, or ``.gz`` source.

    A ``.zip`` archive may hold several rotated members; each is read in turn. Decoding never raises
    (``errors="replace"``) so a partially-corrupt archive still yields what it can.
    """
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                with archive.open(member) as handle:
                    text = handle.read().decode("utf-8", errors="replace")
                yield from text.replace("\x00", "").splitlines()
        return
    if suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                yield line.replace("\x00", "").rstrip("\n")
        return
    with path.open("rb") as handle:
        text = handle.read().decode("utf-8", errors="replace")
    yield from text.replace("\x00", "").splitlines()


def parse_lines(lines: Iterable[str], source_path: Path) -> list[LogRecord]:
    """Parse physical lines into records, folding timestamp-less continuation lines into the prior record."""
    records: list[LogRecord] = []
    for raw_lineno, line in enumerate(lines, start=1):
        head = _HEAD_RE.match(line)
        if head is None:
            # No record head: a traceback/continuation line, or torn output. Attach it to the record in
            # progress so a crash's stack stays with its log line; drop only truly orphaned leading noise.
            if records:
                records[-1].continuation.append(line)
            continue
        name, function, lineno = _split_location(head.group("loc"))
        records.append(
            LogRecord(
                timestamp=parse_ts(line),
                level=head.group("level"),
                name=name,
                function=function,
                lineno=lineno,
                message=head.group("message"),
                source_path=source_path,
                raw_lineno=raw_lineno,
            ),
        )
    return records


def read_records(*paths: Path) -> list[LogRecord]:
    """Read and parse one or more log sources, returned in stable timestamp order.

    Records that share a timestamp keep their original relative order, and a record with no parseable
    timestamp (rare; a torn head) sorts just after the last good timestamp so it stays near its context.
    """
    all_records: list[LogRecord] = []
    for path in paths:
        all_records.extend(parse_lines(_read_physical_lines(path), path))

    # Stable sort by timestamp; carry the previous good timestamp forward for the rare ts-less head so
    # it does not all collapse to the epoch and scramble ordering.
    def _sort_key(item: tuple[int, LogRecord]) -> tuple[datetime, int]:
        index, record = item
        return (record.timestamp or datetime.min, index)

    indexed = sorted(enumerate(all_records), key=_sort_key)
    return [record for _, record in indexed]
