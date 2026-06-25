"""Read the worker's action-ledger JSONL into typed events, when it is present.

The worker mirrors every lifecycle action it takes on its children to ``.horde_worker_regen/
action_ledger.jsonl`` (size-rotated to ``.jsonl.1``); see
``horde_worker_regen.process_management.ipc.action_ledger``. That file is the structured spine of an
incident: spawn/replace/quarantine/give-up events with ``os_pid``/``launch_identifier``/``job_id`` and
a free-form ``reason``/``detail``. This module locates and parses it.

The ledger lives next to the app state (cwd ``.horde_worker_regen/``), not in ``logs/``, and is absent
entirely for logs an operator zipped up and sent us. Everything here therefore degrades to an empty
list rather than failing, so the rest of the toolchain can fall back to the human logs.
"""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.process_management.ipc.action_ledger import LedgerEvent

_LEDGER_FILENAME = "action_ledger.jsonl"
_APP_STATE_DIRNAME = ".horde_worker_regen"


def find_ledger_paths(root: Path) -> list[Path]:
    """Find ledger files related to ``root`` (a logs dir, a repo root, or the app-state dir itself).

    Returns existing files oldest-first (the rotated ``.jsonl.1`` before the active ``.jsonl``) so a
    naive concatenation is already in chronological order. Empty if none are found.
    """
    candidate_dirs = [
        root / _APP_STATE_DIRNAME,  # root is a repo/run dir holding .horde_worker_regen/
        root.parent / _APP_STATE_DIRNAME,  # root is logs/, ledger is a sibling
        root,  # root already is .horde_worker_regen/ (or wherever the files sit)
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for directory in candidate_dirs:
        for name in (f"{_LEDGER_FILENAME}.1", _LEDGER_FILENAME):
            path = directory / name
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if path.is_file():
                found.append(path)
    return found


def read_ledger(paths: list[Path]) -> list[LedgerEvent]:
    """Parse ledger JSONL files into events, sorted by timestamp.

    Tolerant of a torn final line (the worker may be mid-write) and of blank lines: anything that does
    not parse as a ``LedgerEvent`` is skipped rather than aborting the read.
    """
    events: list[LedgerEvent] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(LedgerEvent.model_validate_json(line))
            except ValueError:
                continue
    events.sort(key=lambda event: event.timestamp)
    return events


def load_ledger_for(root: Path) -> list[LedgerEvent]:
    """Convenience: find and read all ledger events related to ``root`` (empty if absent)."""
    return read_ledger(find_ledger_paths(root))
