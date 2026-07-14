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


def _ts_from_prefix(ts: str) -> datetime | None:
    """Build a datetime from a fixed-width ``YYYY-MM-DD HH:MM:SS.ffffff`` prefix by integer slicing.

    Hand-rolled rather than ``datetime.strptime`` because this runs once per log line over multi-MB
    logs, and ``strptime`` (with its format re-parse and locale machinery) dominated the parse. The
    prefix shape is guaranteed by :data:`TS_RE` / :data:`_HEAD_RE`, so the slices are safe; loguru's
    fractional part is milliseconds but may be any width, so it is right-padded to microseconds.
    """
    try:
        return datetime(
            int(ts[0:4]),
            int(ts[5:7]),
            int(ts[8:10]),
            int(ts[11:13]),
            int(ts[14:16]),
            int(ts[17:19]),
            int(ts[20:].ljust(6, "0")[:6]),
        )
    except (ValueError, IndexError):
        return None


def parse_ts(text: str) -> datetime | None:
    """Parse a loguru timestamp prefix from the start of ``text``, or None if it has none."""
    match = TS_RE.match(text)
    if match is None:
        return None
    return _ts_from_prefix(match.group("ts"))


@dataclass
class LogRecord:
    """One parsed log line plus any continuation (traceback) lines folded into it."""

    timestamp: datetime | None
    level: str
    name: str
    """The logger name (dotted module path), e.g. the process-lifecycle module path."""
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


def parse_lines(
    lines: Iterable[str],
    source_path: Path,
    *,
    start_lineno: int = 1,
    carry: LogRecord | None = None,
) -> list[LogRecord]:
    """Parse physical lines into records, folding timestamp-less continuation lines into the prior record.

    ``start_lineno`` numbers the first physical line, so an incremental parse of a file's appended tail keeps
    ``raw_lineno`` continuous with the lines already consumed. ``carry`` is the last record parsed from an
    earlier chunk of the same file: a leading continuation line (this chunk began mid-traceback) folds into
    it, reproducing exactly what a whole-file parse would have done across the chunk boundary.
    """
    records: list[LogRecord] = []
    for raw_lineno, line in enumerate(lines, start=start_lineno):
        head = _HEAD_RE.match(line)
        if head is None:
            # No record head: a traceback/continuation line, or torn output. Attach it to the record in
            # progress so a crash's stack stays with its log line; drop only truly orphaned leading noise.
            if records:
                records[-1].continuation.append(line)
            elif carry is not None:
                carry.continuation.append(line)
            continue
        name, function, lineno = _split_location(head.group("loc"))
        records.append(
            LogRecord(
                # Reuse the head match's ts group instead of re-running TS_RE + strptime on the line.
                timestamp=_ts_from_prefix(head.group("ts")),
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


def _read_region(path: Path, start_offset: int) -> tuple[list[str], int]:
    """Read the whole lines appended to ``path`` at or after ``start_offset``; return them and the new offset.

    Only bytes up to the final newline are consumed, so a trailing partial line (a writer caught mid-flush)
    is left for a later pass rather than parsed as a torn record. NUL bytes are stripped after the
    byte-offset arithmetic so the returned offset stays a true file position that a later ``seek`` can trust.
    """
    with path.open("rb") as handle:
        handle.seek(start_offset)
        data = handle.read()
    last_newline = data.rfind(b"\n")
    if last_newline == -1:
        return [], start_offset
    consumed = data[: last_newline + 1]
    text = consumed.decode("utf-8", errors="replace").replace("\x00", "")
    return text.splitlines(), start_offset + len(consumed)


@dataclass
class FileReadProgress:
    """Per-file incremental parse state: the records settled so far and how far the file has been consumed."""

    consumed_size: int = 0
    """File size (bytes) observed at the last parse; a size below this signals truncation/rotation."""
    consumed_offset: int = 0
    """Byte offset up to which whole lines are parsed and settled; the next pass reads from here."""
    line_count: int = 0
    """Physical lines consumed up to ``consumed_offset`` (keeps ``raw_lineno`` continuous across passes)."""
    records: list[LogRecord] = field(default_factory=list)
    """Records parsed so far, in file order."""


class IncrementalRecordReader:
    """A :func:`read_records`-compatible reader that parses only each file's appended tail across passes.

    Bound to a persistent per-path progress map, so re-reading a growing log re-parses only the bytes added
    since the previous call: an unchanged file is not reopened at all, and one that shrank or is newly seen is
    parsed from zero. The records handed on are identical to a whole-file parse, including continuation lines
    that fold across a chunk boundary (via the carried last record), so detectors see the same input either
    way. The first pass over a large file still parses it in full; only subsequent passes are incremental.
    """

    def __init__(self, progress: dict[Path, FileReadProgress]) -> None:
        """Bind the reader to a caller-owned progress map that persists across passes."""
        self._progress = progress

    def __call__(self, *paths: Path) -> list[LogRecord]:
        """Return the parsed records for ``paths`` (a single active file, or several merged in time order)."""
        if len(paths) == 1:
            return self._records_for(paths[0])
        merged: list[LogRecord] = []
        for path in paths:
            merged.extend(self._records_for(path))
        indexed = sorted(enumerate(merged), key=lambda item: (item[1].timestamp or datetime.min, item[0]))
        return [record for _, record in indexed]

    def _records_for(self, path: Path) -> list[LogRecord]:
        """Parse only the bytes appended to ``path`` since the last pass, returning its full record set."""
        try:
            size = path.stat().st_size
        except OSError:
            progress = self._progress.get(path)
            return progress.records if progress is not None else []

        progress = self._progress.get(path)
        if progress is None or size < progress.consumed_size:
            # Newly seen, or shrank/rotated under us: parse from zero.
            progress = FileReadProgress()
            self._progress[path] = progress
        elif size == progress.consumed_size:
            # Unchanged since the last pass: reuse the cached records without reopening the file.
            return progress.records

        lines, next_offset = _read_region(path, progress.consumed_offset)
        if lines:
            carry = progress.records[-1] if progress.records else None
            new_records = parse_lines(lines, path, start_lineno=progress.line_count + 1, carry=carry)
            progress.records.extend(new_records)
            progress.line_count += len(lines)
            progress.consumed_offset = next_offset
        progress.consumed_size = size
        return progress.records
