"""Tests for bundling a pricing-corpus run for hand-off.

Synthetic definitions and stats parts on a temporary directory; no worker, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from horde_worker_regen.benchmark.corpus_bundle import (
    BUNDLE_MANIFEST_NAME,
    BundleError,
    bundle_name,
    find_session_parts,
    utc_stamp,
    write_bundle,
)
from horde_worker_regen.benchmark.pricing_corpus import (
    SCENARIO_NAME,
    SCENARIO_REVISION,
    CorpusMachineFacts,
    PricingCorpusDefinition,
    build_pricing_corpus_scenario,
    write_definition_artifact,
)

_CREATED_AT = 1_000_000.0


def _stamped_definition() -> PricingCorpusDefinition:
    _scenario, definition = build_pricing_corpus_scenario("smoke")
    return definition.model_copy(
        update={"created_at": _CREATED_AT, "machine": CorpusMachineFacts(machine_id="test-rig", vram_mb=1)},
    )


def _write_session(
    stats_dir: Path,
    stem: str,
    *,
    started_at: float,
    parts: int = 1,
    scenario_revision: str = SCENARIO_REVISION,
) -> list[Path]:
    """Write a session: the first part carries session_start, later parts only records."""
    written: list[Path] = []
    for index in range(parts):
        path = stats_dir / f"{stem}-{index:03d}.jsonl"
        lines: list[dict[str, object]] = []
        if index == 0:
            lines.append(
                {
                    "event": "session_start",
                    "timestamp": started_at,
                    "config": {"scenario_id": SCENARIO_NAME, "scenario_revision": scenario_revision},
                },
            )
        lines.append({"event": "job_completed", "job": {"time_popped": started_at + 10.0 * (index + 1)}})
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


def test_bundle_name_is_machine_tier_and_utc() -> None:
    """The name sorts across machines and timezones and refuses an unattributed definition."""
    definition = _stamped_definition()
    assert bundle_name(definition) == f"corpus-test-rig-smoke-{utc_stamp(_CREATED_AT)}"
    assert utc_stamp(0.0) == "19700101T000000Z"
    with pytest.raises(BundleError, match="stamped"):
        bundle_name(definition.model_copy(update={"machine": None}))


def test_find_session_parts_picks_the_session_after_the_definition(tmp_path: Path) -> None:
    """An earlier session and a later corpus revision are other runs; the nearest later start wins."""
    definition = _stamped_definition()
    _write_session(tmp_path, "stats-v0.0.0-20000101-000000", started_at=_CREATED_AT - 100.0)
    _write_session(tmp_path, "stats-v0.0.0-20000101-000200", started_at=_CREATED_AT + 20.0, scenario_revision="0")
    expected = _write_session(tmp_path, "stats-v0.0.0-20000101-000100", started_at=_CREATED_AT + 30.0, parts=3)
    _write_session(tmp_path, "stats-v0.0.0-20000101-000300", started_at=_CREATED_AT + 900.0)

    assert find_session_parts(tmp_path, definition) == expected


def test_find_session_parts_refuses_when_nothing_started_after_the_definition(tmp_path: Path) -> None:
    """A session that opened before the definition was written cannot be this run's."""
    definition = _stamped_definition()
    _write_session(tmp_path, "stats-v0.0.0-20000101-000000", started_at=_CREATED_AT - 1.0)
    with pytest.raises(BundleError, match="no pricing-corpus"):
        find_session_parts(tmp_path, definition)


def test_write_bundle_copies_the_pair_and_manifests_it(tmp_path: Path) -> None:
    """The bundle holds byte-identical copies of the definition and every stats part, and hashes them."""
    definition = _stamped_definition()
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    definition_path = stats_dir / "pricing-corpus-smoke-test-rig.json"
    write_definition_artifact(definition, definition_path)
    parts = _write_session(stats_dir, "stats-v0.0.0-20000101-000100", started_at=_CREATED_AT + 5.0, parts=2)

    bundle_dir = write_bundle(definition, definition_path=definition_path, stats_dir=stats_dir, out_root=tmp_path)

    assert bundle_dir.name == bundle_name(definition)
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["machine"]["machine_id"] == "test-rig"
    assert manifest["created_at_utc"] == utc_stamp(_CREATED_AT)
    names = [entry["name"] for entry in manifest["files"]]
    assert names == [definition_path.name, *(part.name for part in parts)]
    assert all((bundle_dir / name).read_bytes() == (stats_dir / name).read_bytes() for name in names)
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
