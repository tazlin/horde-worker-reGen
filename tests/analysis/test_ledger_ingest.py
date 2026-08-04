"""Tests for reading the worker's action-ledger JSONL, including the shapes a live file really has.

The ledger is appended while the worker runs and is copied into support bundles, so a reader meets torn
final lines (a write interrupted by a crash), blank lines, and the bundle's own truncation note. None of
those may cost the reader the records around them.
"""

from __future__ import annotations

import json
from pathlib import Path

from horde_worker_regen.analysis.ledger_ingest import find_ledger_paths, read_ledger
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType


def _event_line(event_type: LedgerEventType, *, process_id: int) -> str:
    """One well-formed ledger record as it is written to the JSONL file."""
    return json.dumps(
        {"timestamp": 1_780_000_000.0 + process_id, "event_type": event_type.value, "process_id": process_id},
    )


def test_torn_final_line_is_skipped(tmp_path: Path) -> None:
    """A record cut mid-write costs only itself: every whole record before it is still returned."""
    path = tmp_path / "action_ledger.jsonl"
    path.write_text(
        _event_line(LedgerEventType.PROCESS_SPAWNED, process_id=0)
        + "\n"
        + _event_line(LedgerEventType.INFERENCE_DISPATCHED, process_id=1)
        + "\n"
        + '{"timestamp": 1780000002.0, "event_ty',
        encoding="utf-8",
    )

    events = read_ledger([path])

    assert [event.process_id for event in events] == [0, 1]


def test_bundle_truncation_note_is_skipped(tmp_path: Path) -> None:
    """The note a support bundle heads a trimmed ledger with is not a record and is passed over."""
    path = tmp_path / "action_ledger.jsonl"
    path.write_text(
        json.dumps({"_bundle_truncation": "[... truncated to the most recent 15 MB ...]"})
        + "\n"
        + _event_line(LedgerEventType.PROCESS_REPLACED, process_id=2)
        + "\n",
        encoding="utf-8",
    )

    events = read_ledger([path])

    assert [event.event_type for event in events] == [LedgerEventType.PROCESS_REPLACED]


def test_rotated_and_active_files_are_read_oldest_first(tmp_path: Path) -> None:
    """Both ledger generations are found, so a rotation does not hide the older half of a run."""
    app_state = tmp_path / ".horde_worker_regen"
    app_state.mkdir()
    (app_state / "action_ledger.jsonl.1").write_text(
        _event_line(LedgerEventType.PROCESS_SPAWNED, process_id=0) + "\n",
        encoding="utf-8",
    )
    (app_state / "action_ledger.jsonl").write_text(
        _event_line(LedgerEventType.PROCESS_REPLACED, process_id=1) + "\n",
        encoding="utf-8",
    )

    events = read_ledger(find_ledger_paths(tmp_path))

    assert [event.process_id for event in events] == [0, 1]
