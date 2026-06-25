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
from dataclasses import dataclass, field
from pathlib import Path

from horde_worker_regen.process_management.action_ledger import LedgerEvent

from . import ledger_ingest
from .log_ingest import LogRecord, read_records

# Role-classifying patterns over a *base* name (rotation timestamp and .zip/.gz suffix already stripped).
_ORCHESTRATOR_RE = re.compile(r"^bridge\.log$")
_CHILD_LOOP_RE = re.compile(r"^bridge_(?P<pid>\d+)\.log$")
_INFERENCE_STARTUP_RE = re.compile(r"^bridge_inference_(?P<pid>\d+)_startup\.log$")
_SAFETY_STARTUP_RE = re.compile(r"^bridge_safety_(?P<pid>\d+)_startup\.log$")
_DOWNLOAD_STARTUP_RE = re.compile(r"^bridge_download_(?P<pid>\d+)_startup\.log$")
_STDERR_RE = re.compile(r"^stderr_(?P<pid>\d+)\.log$")

# A rotated archive carries a timestamp segment before ".log", e.g. "bridge.2026-06-22_00-55-59.log".
_ROTATION_TS_RE = re.compile(r"\.\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?(?=\.log$)")


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
class LogBundle:
    """A worker run's log files, grouped by role and queryable for records by process slot."""

    root: Path
    orchestrator_paths: list[Path] = field(default_factory=list)
    child_loop_paths: dict[int, list[Path]] = field(default_factory=dict)
    startup_paths: dict[int, list[Path]] = field(default_factory=dict)
    stderr_paths: dict[int, list[Path]] = field(default_factory=dict)
    _ledger_cache: list[LedgerEvent] | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_path(cls, path: Path) -> LogBundle:
        """Build a bundle from a directory, a single log file, or a ``.zip`` of logs."""
        if path.is_file() and path.suffix.lower() == ".zip" and not _looks_like_rotation(path):
            extracted = Path(tempfile.mkdtemp(prefix="horde_log_bundle_"))
            with zipfile.ZipFile(path) as archive:
                archive.extractall(extracted)
            return cls._from_directory(extracted, ledger_root=path.parent)
        if path.is_file():
            bundle = cls(root=path.parent)
            bundle._classify(path)
            return bundle
        return cls._from_directory(path, ledger_root=path)

    @classmethod
    def _from_directory(cls, directory: Path, *, ledger_root: Path) -> LogBundle:
        bundle = cls(root=ledger_root)
        # Recurse one level so a bundle that preserved the ``logs/`` subdir (as the db0 captures do) is
        # still found, without scanning an entire unrelated tree.
        for candidate in [*directory.glob("*"), *directory.glob("*/*")]:
            if candidate.is_file():
                bundle._classify(candidate)
        return bundle

    def _classify(self, path: Path) -> None:
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
        return read_records(*self.orchestrator_paths)

    def child_records(self, process_id: int) -> list[LogRecord]:
        """Parsed loop-log records for one slot, in time order (empty if that slot has no loop log)."""
        return read_records(*self.child_loop_paths.get(process_id, []))

    def startup_records(self, process_id: int) -> list[LogRecord]:
        """Parsed startup-crash-backstop records for one slot (where pre-sink crashes land)."""
        return read_records(*self.startup_paths.get(process_id, []))

    def ledger_events(self) -> list[LedgerEvent]:
        """All action-ledger events related to this bundle (empty when no ledger was shipped).

        Cached: a per-session diagnosis queries this once per session, and re-reading/parsing the JSONL
        each time dominated bundle generation.
        """
        if self._ledger_cache is None:
            self._ledger_cache = ledger_ingest.load_ledger_for(self.root)
        return self._ledger_cache


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
