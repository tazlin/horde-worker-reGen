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
    # Avoid the reference-manager network fetch: skip the sized disk plan and report every model present.
    monkeypatch.setattr("horde_worker_regen.benchmark.requirements.models_disk_plan", lambda _names: None)
    monkeypatch.setattr("horde_worker_regen.benchmark.requirements.model_present_on_disk", lambda _name: True)

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


def test_plan_table_renders_controlnet_column() -> None:
    """The text plan table shows a CN column (size / MISSING) and prompts to download absent annotators."""
    from horde_worker_regen.benchmark.progress_channel import LevelPlanRow
    from horde_worker_regen.benchmark.progress_console import format_plan_table

    rows = [
        LevelPlanRow(
            level_id="C-sd15-controlnet",
            requires_controlnet=True,
            controlnet_installed=True,
            controlnet_annotators_present=False,  # extra installed but the weights are not on disk yet
            controlnet_annotator_bytes=800 * 1024**2,
            will_run=True,
            needs_download=True,  # runnable once the annotators are fetched -> "download first", not "ready"
            download_summary="controlnet annotators",
        ),
        LevelPlanRow(
            level_id="C-sd15-cn-absent",
            requires_controlnet=True,
            controlnet_installed=False,
            will_run=False,
            verdict="controlnet not installed",
        ),
        LevelPlanRow(level_id="A-sd15-baseline", will_run=True),
    ]

    table = format_plan_table(rows)

    assert "CN" in table
    assert "MISSING" in table
    assert "~0.8G" in table
    assert "DOWNLOAD FIRST" in table  # the annotators-missing level reads as a third, actionable state
    assert "controlnet annotators" in table  # named both in its verdict and the trailing download prompt


def test_plan_row_marks_a_fitting_but_incomplete_level_as_download_first() -> None:
    """A level with no skip verdict but missing models/checkpoints/annotators becomes ``needs_download``.

    Names each kind of missing artifact in the summary so the operator sees what to fetch; this is the
    derivation behind the reported bug where such a level wrongly read ``Ready``.
    """
    from horde_worker_regen.benchmark.capabilities.plan_preview import _plan_row
    from horde_worker_regen.benchmark.requirements import LevelRequirements

    req = LevelRequirements(
        level_id="C-sd15",
        stage="controlnet",
        tier="sd15",
        axis="controlnet",
        baseline="stable_diffusion_1",
        models_missing=["AbsentModel"],
        controlnet_checkpoints_missing=["control_canny"],
        controlnet_annotators_present=False,
    )

    row = _plan_row(req, None)  # verdict None: the level fits this machine

    assert row.will_run is True
    assert row.needs_download is True
    assert "1 model" in row.download_summary
    assert "controlnet checkpoints" in row.download_summary
    assert "controlnet annotators" in row.download_summary


def test_plan_row_never_marks_a_hard_skip_as_download_first() -> None:
    """A skipped level (a verdict is present) is never ``download first``, even with missing artifacts."""
    from horde_worker_regen.benchmark.capabilities.plan_preview import _plan_row
    from horde_worker_regen.benchmark.requirements import LevelRequirements

    req = LevelRequirements(
        level_id="F-flux",
        stage="baseline",
        tier="flux",
        axis="baseline",
        baseline="flux_1",
        models_missing=["AbsentModel"],
    )

    row = _plan_row(req, "insufficient VRAM: estimated 16384 MB needed, 12000 MB available")

    assert row.will_run is False
    assert row.needs_download is False
    assert row.download_summary == ""


def test_plan_table_omits_annotator_prompt_when_already_downloaded() -> None:
    """A controlnet level whose annotators are already on disk does NOT nag to download them.

    Reproduces the always-on prompt: the banner used to key off ``controlnet_annotator_bytes > 0`` (a
    static ROM constant) and so fired for every controlnet level even when the weights were present.
    """
    from horde_worker_regen.benchmark.progress_channel import LevelPlanRow
    from horde_worker_regen.benchmark.progress_console import format_plan_table

    rows = [
        LevelPlanRow(
            level_id="C-sd15-controlnet",
            requires_controlnet=True,
            controlnet_installed=True,
            controlnet_annotators_present=True,  # already downloaded (the on-disk marker is present)
            controlnet_annotator_bytes=800 * 1024**2,
            will_run=True,
        ),
    ]

    table = format_plan_table(rows)

    assert "controlnet annotators" not in table


def test_plan_subcommand_honours_exclude_capability(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--exclude-capability` drops just that axis's levels while leaving the rest of the stage intact."""
    # Avoid the reference-manager network fetch: skip the sized disk plan and report every model present.
    monkeypatch.setattr("horde_worker_regen.benchmark.requirements.models_disk_plan", lambda _names: None)
    monkeypatch.setattr("horde_worker_regen.benchmark.requirements.model_present_on_disk", lambda _name: True)

    rc = main(["plan", "--tiers", "sd15", "--process-mode", "fake", "--exclude-capability", "controlnet", "--json"])
    assert rc == 0

    axes = {row.level_id for row in decode_plan_rows(capsys.readouterr().out)}
    assert not any("controlnet" in level_id for level_id in axes)
    # A sibling feature axis (hires-fix) is untouched by excluding controlnet.
    assert any("hires_fix" in level_id for level_id in axes)
