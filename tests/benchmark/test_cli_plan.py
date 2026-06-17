"""Tests for the `horde-benchmark plan` subcommand (the no-boot resource preview)."""

from __future__ import annotations

import json

import pytest

from horde_worker_regen.benchmark.cli import main


def test_plan_subcommand_emits_one_row_per_level(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`plan --json` exits 0 and prints one verdict row per ladder level without starting a worker."""
    # Avoid the reference-manager network fetch; presence is irrelevant to the fake-mode verdict.
    monkeypatch.setattr("horde_worker_regen.benchmark.controller.model_present_on_disk", lambda _name: True)

    rc = main(
        [
            "plan",
            "--tiers",
            "sd15",
            "--process-mode",
            "fake",
            "--no-concurrency",
            "--no-features",
            "--no-alchemy",
            "--json",
        ],
    )
    assert rc == 0

    rows = json.loads(capsys.readouterr().out)
    assert len(rows) >= 1
    assert {"level_id", "verdict", "will_run", "estimated_vram_mb"} <= set(rows[0])
    assert all(row["will_run"] for row in rows)  # fake mode never gates on resources
