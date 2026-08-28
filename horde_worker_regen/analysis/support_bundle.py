"""Assemble a single, redacted, self-describing support bundle to hand a maintainer.

One command turns a worker's scattered evidence into one ``.zip`` that is safe to share: it runs the
diagnosis, gathers the logs + ledger + config + host/cache context, and streams **every** text artifact
through the :mod:`redaction` scrubber on the way in, so no API key or personal identifier survives. The
bundle leads with the analysis (``diagnose.txt``) so the maintainer sees the likely root cause before
digging into the raw logs that back it.

Pure-stdlib + the rest of the analysis package; torch-free (the optional GPU probe runs out of process).
"""

from __future__ import annotations

import getpass
import gzip
import json
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from horde_worker_regen.process_management.resources.vram_footprints import FOOTPRINT_STORE_FILENAME

from . import ledger_ingest
from .bundle import ROTATION_ABUT_SECONDS, LogBundle
from .cache_inventory import collect_cache_inventory
from .correlate import build_session_context
from .detectors import run_detectors
from .log_ingest import _read_physical_lines, read_time_range
from .redaction import Redactor, build_redactor
from .sessions import WorkerSession, segment_sessions
from .system_info import (
    collect_system_info,
    config_secret_values,
    config_worker_name,
    resolve_cache_home,
)
from .triage_report import finding_to_dict, render_findings, render_sessions

_DEFAULT_CONFIG_PATH = Path("bridgeData.yaml")
# Per-file cap: keep the tail (the most recent, most relevant lines) of any oversized artifact so one
# verbose file (the console mirror is tens of MB) cannot balloon the bundle or make the interactive build
# crawl. Logs and the JSONL artifacts (ledger, stats) are capped with the same polarity, so a capped
# bundle does not end up with its artifacts covering different windows of the timeline.
# 15 MB keeps a full active bridge.log intact and trims only the largest mirrors. `--full-logs` lifts it.
_MAX_FILE_BYTES = 15 * 1024 * 1024
# A rotated archive: loguru's timestamped roll-over (``bridge.2026-06-22_00-55-59.log``) or its compressed
# form. These are older sessions; the active bridge.log (appended across restarts) already covers history,
# so rotations are excluded by default and included only with --full-logs.
_ROTATION_TS_RE = re.compile(r"\.\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?\.log$")
# A parent log rotates by size mid-run, so the active bridge.log of a long session begins hours in and the
# default bundle loses the session's earlier hours (and, through the stats window, its earlier stats). When
# the earliest bundled session begins mid-run, this many of the active log's abutting predecessors ship
# uncapped, oldest first, so the run is whole back to its launch or to this bound. Three rotations at the
# 15 MB roll size cover most of a day of parent DEBUG output for a few MB compressed.
_MAX_STITCHED_ROTATIONS = 3
_APP_STATE_DIRNAME = ".horde_worker_regen"
_STATS_DIRNAME = "stats"
# The key the JSONL form of the truncation note is emitted under, so a reader that parses every line as
# JSON sees an object it can recognize and skip rather than a bare bracketed line it must tolerate.
_TRUNCATION_NOTE_KEY = "_bundle_truncation"


@dataclass
class BundleResult:
    """Summary of a written support bundle."""

    out_path: Path
    member_count: int
    redaction_count: int
    session_count: int
    size_bytes: int


def _select_sessions(sessions: list[WorkerSession], *, last: bool, index: int | None) -> list[WorkerSession]:
    """Apply last/index selection (default: all sessions)."""
    if index is not None:
        return [s for s in sessions if s.index == index]
    if last and sessions:
        return sessions[-1:]
    return sessions


def _is_rotation(file_path: Path) -> bool:
    """Whether a file is a loguru rotation archive (timestamped or compressed), not an active log."""
    name = file_path.name
    return bool(_ROTATION_TS_RE.search(name)) or name.endswith((".zip", ".gz"))


def _all_log_files(path: Path, *, include_rotations: bool) -> tuple[Path, list[Path]]:
    """Resolve the logs root and the worker's own log files to include.

    Only the top level of the logs directory is bundled: the worker writes all its logs flat in ``logs/``,
    so a nested subdirectory (e.g. an ``external_logs/`` archive of unrelated captures) is not this
    worker's evidence and must not bloat the bundle. Rotation archives are excluded unless requested.
    """
    if path.is_file():
        return path.parent, [path]
    files = sorted(p for p in path.glob("*") if p.is_file() and (include_rotations or not _is_rotation(p)))
    return path, files


def _stitched_predecessors(active_log: Path, *, limit: int = _MAX_STITCHED_ROTATIONS) -> list[Path]:
    """The rotated predecessors of an active parent log that belong to its own run, oldest first.

    Walks back from the active file through same-base rotations, keeping each whose last line abuts the next
    file's first (a size roll-over mid-run) and stopping at the first that does not (an earlier launch) or at
    ``limit``. Reads only each archive's first and last timestamps, so the walk stays cheap.
    """
    base = _base_log_name(active_log)
    candidates = sorted(
        (
            sibling
            for sibling in active_log.parent.glob("*")
            if sibling.is_file()
            and sibling != active_log
            and _is_rotation(sibling)
            and _base_log_name(sibling) == base
        ),
        key=lambda path: path.name,
    )
    stitched: list[Path] = []
    later_start, _ = read_time_range(active_log)
    for candidate in reversed(candidates):
        if len(stitched) >= limit:
            break
        first, last = read_time_range(candidate)
        if later_start is None or last is None or abs((later_start - last).total_seconds()) > ROTATION_ABUT_SECONDS:
            break
        stitched.insert(0, candidate)
        later_start = first if first is not None else later_start
    return stitched


def _base_log_name(file_path: Path) -> str:
    """``bridge.2026-06-22_00-55-59_013989.log.zip`` -> ``bridge.log``: the log a rotation was cut from."""
    name = file_path.name
    for suffix in (".zip", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return _ROTATION_TS_RE.sub(".log", name)


def _member_name(file_path: Path, logs_root: Path) -> str:
    """The in-zip path for a log file: ``logs/<relative>`` with any .zip/.gz rotation suffix removed."""
    try:
        rel = file_path.relative_to(logs_root)
    except ValueError:
        rel = Path(file_path.name)
    name = str(rel)
    for suffix in (".zip", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return f"logs/{name}"


def _truncation_marker() -> str:
    """The one-line note that heads any artifact trimmed to the per-file cap."""
    megabytes = _MAX_FILE_BYTES // (1024 * 1024)
    return f"[... truncated to the most recent {megabytes} MB ...]"


def _drop_partial_first_line(text: str) -> str:
    """Drop the leading partial line left by cutting a file at an arbitrary byte offset."""
    newline = text.find("\n")
    return text[newline + 1 :] if newline != -1 else ""


def _tail_capped(text: str) -> str:
    """Keep ``text``'s most recent whole lines within the cap, headed by the truncation note.

    Every capped artifact keeps its end: an incident is read from what the worker was doing most
    recently, so trimming from the front is what preserves the evidence. Cutting on a line boundary keeps
    what survives parseable rather than leaving a torn leading line.
    """
    if len(text) <= _MAX_FILE_BYTES:
        return text
    return f"{_truncation_marker()}\n{_drop_partial_first_line(text[-_MAX_FILE_BYTES:])}"


def _tail_capped_records(text: str, *, cap: bool) -> str:
    """Keep the most recent whole JSONL records of ``text`` within the cap, headed by a note record.

    Same polarity and record alignment as the log artifacts, so every artifact in one bundle covers the
    same end of the timeline. The note is a JSON object carrying the log wording verbatim: it reads the
    same to a human and still parses for a reader that loads every line as JSON.
    """
    if not cap or len(text) <= _MAX_FILE_BYTES:
        return text
    note = json.dumps({_TRUNCATION_NOTE_KEY: _truncation_marker()})
    return f"{note}\n{_drop_partial_first_line(text[-_MAX_FILE_BYTES:])}"


def _read_log_text(file_path: Path, *, cap: bool) -> str:
    """Read a log file (plain/.zip/.gz, NUL-stripped) as text, keeping only the tail past the size cap."""
    if cap and file_path.suffix.lower() == ".log":
        size = file_path.stat().st_size
        if size > _MAX_FILE_BYTES:
            with file_path.open("rb") as handle:
                handle.seek(-_MAX_FILE_BYTES, os.SEEK_END)
                text = handle.read().decode("utf-8", errors="replace").replace("\x00", "")
            return f"{_truncation_marker()}\n{_drop_partial_first_line(text)}"
    lines = list(_read_physical_lines(file_path))
    text = "\n".join(lines)
    if cap:
        text = _tail_capped(text)
    return text


def _find_stats_files(root: Path, *, modified_since: datetime | None = None) -> list[Path]:
    """Find retained stats JSONL files related to ``root``, keeping those last written at or after ``modified_since``."""
    candidate_dirs = [
        root / _APP_STATE_DIRNAME / _STATS_DIRNAME,
        root.parent / _APP_STATE_DIRNAME / _STATS_DIRNAME,
        root / _STATS_DIRNAME,
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for directory in candidate_dirs:
        for pattern in ("stats-v*.jsonl", "stats-v*.jsonl.gz"):
            for path in sorted(directory.glob(pattern)):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                if not path.is_file():
                    continue
                if modified_since is not None and path.stat().st_mtime < modified_since.timestamp():
                    continue
                found.append(path)
    return sorted(found, key=lambda p: p.name)


def _stats_member_name(file_path: Path) -> str:
    """Return the in-zip stats path, normalizing gzip files back to JSONL text."""
    name = file_path.name
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    return f"stats/{name}"


def _read_stats_text(file_path: Path) -> str:
    """Read a retained stats JSONL file, decompressing gzip sources for redaction."""
    if file_path.name.endswith(".gz"):
        with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return file_path.read_text(encoding="utf-8", errors="replace")


def _make_redactor(config_path: Path, *, redact_identifiers: bool) -> Redactor:
    """Build the scrubber from the config's secrets and the host's personal identifiers."""
    username: str | None
    try:
        username = getpass.getuser()
    except Exception:  # noqa: BLE001 - getuser can fail in odd environments; identifiers are best-effort
        username = os.environ.get("USERNAME") or os.environ.get("USER")
    return build_redactor(
        secrets=config_secret_values(config_path),
        home_path=str(Path.home()),
        username=username,
        worker_name=config_worker_name(config_path),
        redact_identifiers=redact_identifiers,
    )


def build_support_bundle(
    path: Path,
    out: Path,
    *,
    last: bool = False,
    session_index: int | None = None,
    full_logs: bool = False,
    cache_inventory: bool = True,
    probe_gpu: bool = False,
    redact_identifiers: bool = True,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> BundleResult:
    """Write a redacted support bundle for ``path``'s logs to the ``out`` zip and return a summary.

    Args:
        path: A logs directory (usual) or a single log file.
        out: Destination ``.zip`` path.
        last: Restrict the diagnosis to the most recent session.
        session_index: Restrict the diagnosis to a specific session index (default: all sessions).
        full_logs: Include rotation archives and lift the per-file tail cap (a much larger bundle).
        cache_inventory: Include the on-disk model listing.
        probe_gpu: Run the out-of-process GPU probe for the system-info block.
        redact_identifiers: Also scrub home path / username / worker name (not just secrets).
        config_path: The worker config to redact and to source secrets/cache_home from.
    """
    # Classify only the active logs by default: they hold the recent sessions, while the rotation archives
    # are older history that costs minutes to decompress and parse for little incident value. The detectors
    # read child logs through this same bundle, so an archive-inclusive bundle would have every child's
    # whole rotation history parsed the first time a detector joined child records to the session, even
    # though those archives are not shipped. --full-logs opts into the complete history on both counts.
    log_bundle = LogBundle.from_path(path, active_only=not full_logs)
    sessions = segment_sessions(log_bundle.orchestrator_records())
    selected = _select_sessions(sessions, last=last, index=session_index)
    # A session that begins mid-run in the active log continues back through the rotations the size
    # roll-over cut it from. Ship those (bounded, uncapped) so the run's earlier hours are not lost to
    # the default bundle, and let the stats window reach back to where the shipped evidence starts.
    stitched_rotations: list[Path] = []
    if not full_logs and any(s.start_is_lower_bound for s in selected):
        for active_log in log_bundle.active_orchestrator_paths():
            stitched_rotations.extend(_stitched_predecessors(active_log))
    evidence_start = min((s.start_ts for s in sessions if s.start_ts is not None), default=None)
    for rotation in stitched_rotations:
        first, _ = read_time_range(rotation)
        if first is not None and (evidence_start is None or first < evidence_start):
            evidence_start = first
    redactor = _make_redactor(config_path, redact_identifiers=redact_identifiers)
    cache_home = resolve_cache_home(config_path)

    redaction_count = 0
    members: list[str] = []

    def _write(zf: zipfile.ZipFile, name: str, text: str) -> None:
        nonlocal redaction_count
        scrubbed, count = redactor.scrub(text)
        redaction_count += count
        zf.writestr(name, scrubbed)
        members.append(name)

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        # Lead with the analysis the maintainer reads first.
        diagnosis = [(s, run_detectors(build_session_context(s, log_bundle))) for s in selected]
        _write(zf, "diagnose.txt", "\n\n".join(render_findings(s, f) for s, f in diagnosis) or "(no sessions)")
        diagnose_json = [{"session_index": s.index, "findings": [finding_to_dict(x) for x in f]} for s, f in diagnosis]
        _write(zf, "diagnose.json", json.dumps(diagnose_json, indent=2))
        _write(zf, "sessions.txt", render_sessions(sessions, root=log_bundle.root))

        # Host + cache context.
        system_info = collect_system_info(cache_home=cache_home, probe_gpu=probe_gpu)
        _write(zf, "system_info.json", json.dumps(system_info, indent=2))
        if cache_inventory:
            _write(zf, "cache_inventory.json", json.dumps(collect_cache_inventory(cache_home), indent=2))

        # Config (redacted) and the structured ledger.
        if config_path.is_file():
            _write(zf, "config/bridgeData.redacted.yaml", config_path.read_text(encoding="utf-8", errors="replace"))

        # Backend-selection breadcrumbs: the persisted torch build and the audit trail of why it was chosen
        # (driver CUDA ceiling, GPU compute capability, what the arch clamp did). These pin down a
        # wrong-CUDA-build install, which the logs otherwise surface only as a downstream runtime fault.
        install_root = config_path.resolve().parent
        # The learned VRAM footprint store is what priced every admission and residency verdict in the logs
        # (a defer line names the room, the store names the charge), so a bundle without it leaves the
        # arithmetic behind a hold unreadable.
        for source, member in (
            (install_root / "bin" / "backend", "config/backend"),
            (install_root / "bin" / "backend-decision.json", "config/backend-decision.json"),
            (install_root / _APP_STATE_DIRNAME / FOOTPRINT_STORE_FILENAME, f"config/{FOOTPRINT_STORE_FILENAME}"),
        ):
            if source.is_file():
                _write(zf, member, source.read_text(encoding="utf-8", errors="replace"))

        ledger_paths = ledger_ingest.find_ledger_paths(log_bundle.root)
        if ledger_paths:
            ledger_text = "\n".join(p.read_text(encoding="utf-8", errors="replace").rstrip("\n") for p in ledger_paths)
            _write(zf, "action_ledger.jsonl", _tail_capped_records(ledger_text, cap=not full_logs))
        # Retained stats can span months of sessions at several MB each; only the files still being written
        # once the bundled sessions began say anything about them. A modification time is the cheap, stamp-
        # agnostic test for that, and --full-logs ships the whole retention as it does the log rotations.
        stats_since = None if full_logs else evidence_start
        used_stats_names: set[str] = set()
        for stats_path in _find_stats_files(log_bundle.root, modified_since=stats_since):
            name = _stats_member_name(stats_path)
            if name in used_stats_names:
                stem, _, ext = name.rpartition(".")
                index = 1
                while f"{stem}.{index}.{ext}" in used_stats_names:
                    index += 1
                name = f"{stem}.{index}.{ext}"
            used_stats_names.add(name)
            _write(zf, name, _tail_capped_records(_read_stats_text(stats_path), cap=not full_logs))

        # Every log file, redacted. Dedup member names so a rotation that exists both raw and zipped (both
        # reduce to the same `.log` name) does not collide in the archive.
        logs_root, log_files = _all_log_files(path, include_rotations=full_logs)
        used_names: set[str] = set()
        uncapped = set(stitched_rotations)
        for file_path in [*stitched_rotations, *log_files]:
            name = _member_name(file_path, logs_root)
            if name in used_names:
                stem, _, ext = name.rpartition(".")
                index = 1
                while f"{stem}.{index}.{ext}" in used_names:
                    index += 1
                name = f"{stem}.{index}.{ext}"
            used_names.add(name)
            _write(zf, name, _read_log_text(file_path, cap=not full_logs and file_path not in uncapped))

        # Manifest + README last, now that we know the member list and redaction total.
        manifest = {
            "tool": "horde-log bundle",
            "worker_version": collect_system_info()["worker_version"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scope": {
                "sessions": "all" if session_index is None and not last else ("last" if last else session_index),
                "session_count": len(selected),
                "full_logs": full_logs,
                "stitched_rotations": [_member_name(p, logs_root) for p in stitched_rotations],
                "cache_inventory": cache_inventory,
                "gpu_probed": probe_gpu,
                "identifiers_redacted": redact_identifiers,
            },
            "redaction_count": redaction_count,
            "members": members,
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        members.append("manifest.json")
        zf.writestr("README.txt", _readme(redaction_count, redact_identifiers))
        members.append("README.txt")

    return BundleResult(
        out_path=out,
        member_count=len(members),
        redaction_count=redaction_count,
        session_count=len(selected),
        size_bytes=out.stat().st_size,
    )


def _readme(redaction_count: int, redact_identifiers: bool) -> str:
    """The human-facing note shipped at the root of the bundle."""
    scope = "API/CivitAI keys"
    if redact_identifiers:
        scope += " and personal identifiers (home path, username, worker name)"
    return (
        "horde-worker-reGen support bundle\n"
        "=================================\n\n"
        "This archive was generated by `horde-log bundle` to help a maintainer diagnose a worker issue.\n"
        "Start with diagnose.txt (the automated analysis), then sessions.txt and the logs/ directory.\n\n"
        f"Redaction: {redaction_count} occurrence(s) of {scope} were replaced with <REDACTED>/<HOME>/<USER>/\n"
        "<WORKER_NAME> markers before this archive was written. Redaction is best-effort: please skim the\n"
        "contents and confirm nothing sensitive remains before sending this file to anyone.\n"
    )
