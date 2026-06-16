"""Unit tests for the self-audited action ledger (in-memory ring + optional rotated JSONL file)."""

from __future__ import annotations

import json
from pathlib import Path

from horde_worker_regen.process_management.action_ledger import (
    ActionLedger,
    LedgerEventType,
)


def test_record_and_recent_in_memory_only() -> None:
    """With no path the ledger keeps events in memory and never touches disk."""
    ledger = ActionLedger()
    ledger.record(LedgerEventType.PROCESS_SPAWNED, process_id=0, os_pid=111)
    ledger.record(LedgerEventType.INFERENCE_DISPATCHED, process_id=0, job_id="job-a")
    ledger.record(LedgerEventType.PROCESS_SPAWNED, process_id=1, os_pid=222)

    all_events = ledger.recent(limit=10)
    assert [e.event_type for e in all_events] == [
        LedgerEventType.PROCESS_SPAWNED,
        LedgerEventType.INFERENCE_DISPATCHED,
        LedgerEventType.PROCESS_SPAWNED,
    ]


def test_recent_filters_by_slot() -> None:
    """recent(process_id=...) returns only that slot's events, newest-last."""
    ledger = ActionLedger()
    ledger.record(LedgerEventType.PROCESS_SPAWNED, process_id=0)
    ledger.record(LedgerEventType.PROCESS_SPAWNED, process_id=1)
    ledger.record(LedgerEventType.TIMEOUT_DETECTED, process_id=0, reason="stuck")

    slot0 = ledger.recent(process_id=0, limit=10)
    assert [e.event_type for e in slot0] == [LedgerEventType.PROCESS_SPAWNED, LedgerEventType.TIMEOUT_DETECTED]
    assert slot0[-1].reason == "stuck"


def test_in_memory_ring_is_bounded() -> None:
    """The in-memory ring keeps only the most recent ``max_in_memory`` events."""
    ledger = ActionLedger(max_in_memory=3)
    for i in range(10):
        ledger.record(LedgerEventType.PROCESS_SPAWNED, process_id=i)
    kept = ledger.recent(limit=100)
    assert len(kept) == 3
    assert [e.process_id for e in kept] == [7, 8, 9]


def test_file_mirroring_writes_jsonl(tmp_path: Path) -> None:
    """When a path is set, each event is appended as one JSON object per line."""
    path = tmp_path / "action_ledger.jsonl"
    ledger = ActionLedger(path=path)
    ledger.record(LedgerEventType.PROCESS_SPAWNED, process_id=0, os_pid=111, detail={"process_type": "INFERENCE"})
    ledger.record(LedgerEventType.PROCESS_REPLACED, process_id=0, reason="crashed")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event_type"] == LedgerEventType.PROCESS_SPAWNED.value
    assert first["os_pid"] == 111
    assert first["detail"]["process_type"] == "INFERENCE"
    assert json.loads(lines[1])["reason"] == "crashed"


def test_file_rotation_on_size(tmp_path: Path) -> None:
    """The JSONL file rotates to ``<name>.1`` once it exceeds the size cap."""
    path = tmp_path / "action_ledger.jsonl"
    ledger = ActionLedger(path=path, max_file_bytes=200)
    for _ in range(50):
        ledger.record(LedgerEventType.PROCESS_SPAWNED, process_id=0, reason="x" * 20)

    assert path.with_suffix(path.suffix + ".1").exists(), "expected a rotated backup file"
    assert path.exists()


def test_file_error_degrades_to_memory(tmp_path: Path) -> None:
    """If the file becomes unwritable, the ledger keeps working in memory (never raises)."""
    path = tmp_path / "action_ledger.jsonl"
    ledger = ActionLedger(path=path)
    ledger.record(LedgerEventType.PROCESS_SPAWNED, process_id=0)

    # Point the ledger at a path whose parent is a file, so directory creation/writes fail.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    ledger._path = blocker / "nested" / "ledger.jsonl"

    ledger.record(LedgerEventType.PROCESS_REPLACED, process_id=0)  # must not raise
    assert ledger._file_disabled is True
    assert len(ledger.recent(limit=10)) == 2
