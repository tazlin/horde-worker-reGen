"""Tests for log-bundle discovery and record caching."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_worker_regen.analysis import bundle as bundle_module
from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.log_ingest import LogRecord


def test_record_accessors_cache_parsed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated analysis lookups must not re-read and re-parse the same process logs."""
    calls: list[tuple[Path, ...]] = []

    def _fake_read_records(*paths: Path) -> list[LogRecord]:
        calls.append(paths)
        return []

    monkeypatch.setattr(bundle_module, "read_records", _fake_read_records)
    bundle = LogBundle(
        root=tmp_path,
        orchestrator_paths=[tmp_path / "bridge.log"],
        child_loop_paths={1: [tmp_path / "bridge_1.log"]},
        startup_paths={1: [tmp_path / "bridge_inference_1_startup.log"]},
    )

    bundle.orchestrator_records()
    bundle.orchestrator_records()
    bundle.child_records(1)
    bundle.child_records(1)
    bundle.startup_records(1)
    bundle.startup_records(1)

    assert calls == [
        (tmp_path / "bridge.log",),
        (tmp_path / "bridge_1.log",),
        (tmp_path / "bridge_inference_1_startup.log",),
    ]
