"""Discover and group the worker's log files into a queryable bundle.

A worker run scatters its logs across many files in one directory (see
``hordelib.utils.logger`` for the naming): the orchestrator ``bridge.log``, per-slot ``bridge_<N>.log``
loop logs, ``bridge_inference_<N>_startup.log`` pre-sink crash backstops, ``stderr_<N>.log``, plus
zipped rotations of each. This module maps those filenames back to their roles so the rest of the
toolchain can ask "give me the orchestrator records" or "give me slot 3's startup crash" without
re-deriving the naming convention.

Accepts a directory (the usual ``logs/``), a single file (just that log), or a ``.zip`` an operator
sent us (extracted to a temp dir and scanned as a directory). The action ledger, if present, is located
relative to the bundle root via :mod:`ledger_ingest`.
"""

from __future__ import annotations

import re
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from horde_worker_regen.process_management.ipc.action_ledger import LedgerEvent

from . import ledger_ingest
from .log_ingest import LogRecord, read_records, read_time_range

# Role-classifying patterns over a *base* name (rotation timestamp and .zip/.gz suffix already stripped).
_ORCHESTRATOR_RE = re.compile(r"^bridge\.log$")
_CHILD_LOOP_RE = re.compile(r"^bridge_(?P<pid>\d+)\.log$")
_INFERENCE_STARTUP_RE = re.compile(r"^bridge_inference_(?P<pid>\d+)_startup\.log$")
_SAFETY_STARTUP_RE = re.compile(r"^bridge_safety_(?P<pid>\d+)_startup\.log$")
_DOWNLOAD_STARTUP_RE = re.compile(r"^bridge_download_(?P<pid>\d+)_startup\.log$")
_STDERR_RE = re.compile(r"^stderr_(?P<pid>\d+)\.log$")

# A rotated archive carries a timestamp segment before ".log", e.g. "bridge.2026-06-22_00-55-59.log".
_ROTATION_TS_RE = re.compile(r"\.\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?(?=\.log$)")

ROTATION_ABUT_SECONDS = 120.0
"""How large a gap between a rotation's last line and the next file's first may be and still be one run.

A size-triggered rotation hands off within a line, so the true gap is milliseconds; the allowance covers a
slow compression or a quiet worker that logged nothing across the handover. It is deliberately far shorter
than the gap between two worker launches, which is what the chain must not swallow."""


def _base_name(path: Path) -> str:
    """Reduce a possibly-rotated, possibly-compressed filename to its canonical base.

    ``bridge.2026-06-22_00-55-59_013989.log.zip`` -> ``bridge.log`` so a rotation maps to the same role
    as its active file.
    """
    name = path.name
    if name.endswith(".zip"):
        name = name[: -len(".zip")]
    elif name.endswith(".gz"):
        name = name[: -len(".gz")]
    return _ROTATION_TS_RE.sub("", name)


@dataclass
class RotationStitch:
    """Which rotated predecessors of the targeted log were folded into the parse, and which were not.

    A rotation that is read silently widens every span and count the report derives, and one that is
    skipped silently narrows them; either way the numbers stop being attributable to a named set of files.
    This is that set, so a session's span can always be traced back to what was actually read.
    """

    included: list[Path] = field(default_factory=list)
    excluded: list[Path] = field(default_factory=list)

    def describe(self) -> str:
        """A one-line disclosure naming the rotations read and, when any were left out, the first skipped."""
        included = ", ".join(path.name for path in self.included) or "none"
        text = f"stitched {len(self.included)} rotated predecessor(s): {included}"
        if self.excluded:
            first = self.excluded[0].name
            text += (
                f"; excluded {len(self.excluded)} older rotation(s) from {first} back, whose ranges do not "
                "abut this run"
            )
        return text


@dataclass
class LogBundle:
    """A worker run's log files, grouped by role and queryable for records by process slot."""

    root: Path
    orchestrator_paths: list[Path] = field(default_factory=list)
    child_loop_paths: dict[int, list[Path]] = field(default_factory=dict)
    startup_paths: dict[int, list[Path]] = field(default_factory=dict)
    stderr_paths: dict[int, list[Path]] = field(default_factory=dict)
    rotation_stitch: RotationStitch | None = None
    """Which rotated predecessors this bundle folded in, or None when no rotation was in play."""
    record_reader: Callable[..., list[LogRecord]] | None = field(default=None, repr=False, compare=False)
    """Reader used to parse the grouped paths; None uses the default whole-file :func:`read_records`. A live
    watch passes an incremental reader here so re-parses touch only each file's appended tail."""
    _orchestrator_cache: list[LogRecord] | None = field(default=None, init=False, repr=False, compare=False)
    _active_orchestrator_cache: list[LogRecord] | None = field(default=None, init=False, repr=False, compare=False)
    _child_cache: dict[int, list[LogRecord]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _startup_cache: dict[int, list[LogRecord]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _ledger_cache: list[LedgerEvent] | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        active_only: bool = False,
        record_reader: Callable[..., list[LogRecord]] | None = None,
    ) -> LogBundle:
        """Build a bundle from a directory, a single log file, or a ``.zip`` of logs.

        ``active_only`` defaults to False for offline forensics (every rotation is classified). A live watch
        sets it True so only the active ``*.log`` files are classified: rotation archives (``*.zip``/``*.gz``)
        and uncompressed rotations (``bridge.<timestamp>.log``) are skipped, since the current session lives in
        the active files and the rotation history is the bulk of the parse, sort, and scan cost on a
        long-running worker. ``record_reader`` overrides the default whole-file reader (e.g. with an
        incremental one) for every role in the bundle.
        """
        if path.is_file() and path.suffix.lower() == ".zip" and not _looks_like_rotation(path):
            extracted = Path(tempfile.mkdtemp(prefix="horde_log_bundle_"))
            with zipfile.ZipFile(path) as archive:
                archive.extractall(extracted)
            return cls._from_directory(
                extracted,
                ledger_root=path.parent,
                active_only=active_only,
                record_reader=record_reader,
            )
        if path.is_file():
            bundle = cls(root=path.parent, record_reader=record_reader)
            bundle._classify(path, active_only=active_only)
            if not active_only:
                bundle._stitch_rotated_predecessors(path)
            return bundle
        return cls._from_directory(
            path,
            ledger_root=path,
            active_only=active_only,
            record_reader=record_reader,
        )

    @classmethod
    def _from_directory(
        cls,
        directory: Path,
        *,
        ledger_root: Path,
        active_only: bool = False,
        record_reader: Callable[..., list[LogRecord]] | None = None,
    ) -> LogBundle:
        bundle = cls(root=ledger_root, record_reader=record_reader)
        # Offline forensics recurse one level so a capture that preserved the ``logs/`` subdir (as the db0
        # captures do) is still found, without scanning an entire unrelated tree. A live watch stays at the
        # top level: the running worker writes only there, and a nested capture directory would drag its own
        # full-size logs (and duplicate role paths, forcing a per-pass merge sort) into every pass.
        candidates = directory.glob("*") if active_only else [*directory.glob("*"), *directory.glob("*/*")]
        for candidate in candidates:
            if candidate.is_file():
                bundle._classify(candidate, active_only=active_only)
        if not active_only:
            bundle._note_orchestrator_rotations()
        return bundle

    def _note_orchestrator_rotations(self) -> None:
        """Record the orchestrator rotations a directory scan already picked up, so the span is accountable.

        A directory target reads every rotation it finds, which is the right default for forensics but
        leaves the report's span resting on files the reader never sees named. Nothing is excluded here.
        """
        rotations = [path for path in self.orchestrator_paths if _is_rotation(path)]
        if rotations:
            self.rotation_stitch = RotationStitch(included=_sorted_rotations(rotations))

    def _stitch_rotated_predecessors(self, target: Path) -> None:
        """Fold the rotated predecessors of ``target``'s own run into the bundle, newest chain first.

        A parent log rotates by size mid-run, so a capture that names only the active file begins partway
        through the run and undercounts everything measured over it. Walking back from the active file and
        keeping each rotation whose last line abuts the next file's first restores the run; the walk stops
        at the first rotation that does not abut, because that is a different launch and folding it in would
        merge two runs into one. Both the kept and the skipped archives are recorded on
        :attr:`rotation_stitch`.
        """
        if _is_rotation(target):
            # The operator named one archive: give them that archive, not the run it was cut from.
            return
        base = _base_name(target)
        candidates = _sorted_rotations(
            [
                sibling
                for sibling in target.parent.glob("*")
                if sibling.is_file() and sibling != target and _is_rotation(sibling) and _base_name(sibling) == base
            ],
        )
        if not candidates:
            return

        stitch = RotationStitch()
        earliest_start, _ = read_time_range(target)
        for index, candidate in enumerate(reversed(candidates)):
            first, last = read_time_range(candidate)
            if not _abuts(last, earliest_start):
                stitch.excluded = list(reversed(candidates[: len(candidates) - index]))
                break
            stitch.included.insert(0, candidate)
            self._classify(candidate)
            earliest_start = first if first is not None else earliest_start
        self.rotation_stitch = stitch

    def _read(self, *paths: Path) -> list[LogRecord]:
        """Parse ``paths`` through the configured reader (defaulting to the whole-file :func:`read_records`)."""
        reader = self.record_reader if self.record_reader is not None else read_records
        return reader(*paths)

    def _classify(self, path: Path, *, active_only: bool = False) -> None:
        if active_only and (path.suffix.lower() in (".zip", ".gz") or _ROTATION_TS_RE.search(path.name)):
            # Live watch: keep only the active current-session logs, not the rotation history.
            return
        base = _base_name(path)
        if _ORCHESTRATOR_RE.match(base):
            self.orchestrator_paths.append(path)
            return
        for pattern, target in (
            (_CHILD_LOOP_RE, self.child_loop_paths),
            (_INFERENCE_STARTUP_RE, self.startup_paths),
            (_SAFETY_STARTUP_RE, self.startup_paths),
            (_DOWNLOAD_STARTUP_RE, self.startup_paths),
            (_STDERR_RE, self.stderr_paths),
        ):
            match = pattern.match(base)
            if match is not None:
                target.setdefault(int(match.group("pid")), []).append(path)
                return

    def process_ids(self) -> set[int]:
        """All slot ids seen across loop, startup, and stderr logs."""
        return set(self.child_loop_paths) | set(self.startup_paths) | set(self.stderr_paths)

    def orchestrator_records(self) -> list[LogRecord]:
        """All parsed orchestrator (``bridge.log``) records, active plus rotations, in time order."""
        if self._orchestrator_cache is None:
            self._orchestrator_cache = self._read(*self.orchestrator_paths)
        return self._orchestrator_cache

    def active_orchestrator_paths(self) -> list[Path]:
        """Only the live ``bridge.log`` (no zipped/rotated archives): where the most recent sessions live.

        A bounded "recent sessions" pass can read just this and skip decompressing the whole rotation
        history, which is the bulk of the disk I/O and parse cost on a long-running worker.
        """
        return [
            path
            for path in self.orchestrator_paths
            if path.suffix.lower() == ".log" and not _ROTATION_TS_RE.search(path.name)
        ]

    def active_orchestrator_records(self) -> list[LogRecord]:
        """Parsed records from only the live ``bridge.log`` (skips rotations); see ``active_orchestrator_paths``."""
        if self._active_orchestrator_cache is None:
            self._active_orchestrator_cache = self._read(*self.active_orchestrator_paths())
        return self._active_orchestrator_cache

    def child_records(self, process_id: int) -> list[LogRecord]:
        """Parsed loop-log records for one slot, in time order (empty if that slot has no loop log)."""
        if process_id not in self._child_cache:
            self._child_cache[process_id] = self._read(*self.child_loop_paths.get(process_id, []))
        return self._child_cache[process_id]

    def startup_records(self, process_id: int) -> list[LogRecord]:
        """Parsed startup-crash-backstop records for one slot (where pre-sink crashes land)."""
        if process_id not in self._startup_cache:
            self._startup_cache[process_id] = self._read(*self.startup_paths.get(process_id, []))
        return self._startup_cache[process_id]

    def ledger_events(self) -> list[LedgerEvent]:
        """All action-ledger events related to this bundle (empty when no ledger was shipped).

        Cached: a per-session diagnosis queries this once per session, and re-reading/parsing the JSONL
        each time dominated bundle generation.
        """
        if self._ledger_cache is None:
            self._ledger_cache = ledger_ingest.load_ledger_for(self.root)
        return self._ledger_cache


def _uncompressed_name(path: Path) -> str:
    """The filename with any compression suffix removed, so a rotation reads the same zipped or not."""
    name = path.name
    for suffix in (".zip", ".gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _is_rotation(path: Path) -> bool:
    """Whether a path is a rotated archive rather than an active log (``bridge.<ts>.log`` with or without .zip)."""
    return _ROTATION_TS_RE.search(_uncompressed_name(path)) is not None


def _rotation_stamp(path: Path) -> str:
    """The rotation timestamp segment of an archive's name, which sorts chronologically as text."""
    match = _ROTATION_TS_RE.search(_uncompressed_name(path))
    return match.group(0) if match is not None else ""


def _sorted_rotations(paths: list[Path]) -> list[Path]:
    """Rotations ordered oldest first, by the timestamp loguru stamps into the archive name."""
    return sorted(paths, key=lambda path: (_rotation_stamp(path), path.name))


def _abuts(earlier_end: datetime | None, later_start: datetime | None) -> bool:
    """Whether a rotation ending at ``earlier_end`` hands directly over to a file starting at ``later_start``."""
    if earlier_end is None or later_start is None:
        return False
    return abs((later_start - earlier_end).total_seconds()) <= ROTATION_ABUT_SECONDS


def _looks_like_rotation(path: Path) -> bool:
    """Whether a ``.zip`` is a single loguru rotation (one log) rather than an operator's bundle.

    A rotation is named for the file it compressed (``bridge.<ts>.log.zip``); reducing it to a known
    base name tells us to read it in place as that role rather than extracting it as a bundle.
    """
    base = _base_name(path)
    return any(
        pattern.match(base)
        for pattern in (
            _ORCHESTRATOR_RE,
            _CHILD_LOOP_RE,
            _INFERENCE_STARTUP_RE,
            _SAFETY_STARTUP_RE,
            _DOWNLOAD_STARTUP_RE,
            _STDERR_RE,
        )
    )
