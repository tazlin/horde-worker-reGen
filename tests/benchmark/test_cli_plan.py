"""Tests for the `horde-benchmark plan` subcommand (the no-boot resource preview)."""

from __future__ import annotations

import pytest

from horde_worker_regen.benchmark.cli import main
from horde_worker_regen.benchmark.progress_channel import decode_plan_rows


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

    # The JSON is sentinel-wrapped so it survives log lines/banners sharing stdout; decode_plan_rows
    # extracts it the same way the TUI's plan preview does.
    rows = decode_plan_rows(capsys.readouterr().out)
    assert len(rows) >= 1
    assert all(row.will_run for row in rows)  # fake mode never gates on resources


def test_plan_subcommand_honours_exclude_axis(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--exclude-axis` drops just that axis's levels while leaving the rest of the stage intact."""
    monkeypatch.setattr("horde_worker_regen.benchmark.controller.model_present_on_disk", lambda _name: True)

    rc = main(["plan", "--tiers", "sd15", "--process-mode", "fake", "--exclude-axis", "controlnet", "--json"])
    assert rc == 0

    axes = {row.level_id for row in decode_plan_rows(capsys.readouterr().out)}
    assert not any("controlnet" in level_id for level_id in axes)
    # A sibling feature axis (hires-fix) is untouched by excluding controlnet.
    assert any("hires_fix" in level_id for level_id in axes)
