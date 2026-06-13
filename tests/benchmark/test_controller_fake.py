"""Controller tests: fake-mode ramps, crash capture, skip rules, and resume."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from horde_worker_regen.benchmark.controller import BenchmarkController
from horde_worker_regen.benchmark.ladder import LadderOptions, RampLevel, build_default_ladder
from horde_worker_regen.benchmark.scenarios import CannedImageJobSpec, ScenarioSpec


def _mini_ladder(jobs: int = 2) -> list[RampLevel]:
    """A single-level sd15 ladder for fast fake-mode runs."""
    return build_default_ladder(
        LadderOptions(
            tiers=["sd15"],
            jobs_per_level=jobs,
            include_concurrency=False,
            include_features=False,
            include_alchemy=False,
        ),
    )


@pytest.mark.e2e
def test_fake_mode_ramp_end_to_end(tmp_path: Path) -> None:
    """A fake-mode ramp passes its level, records a tier baseline, and writes the reports."""
    ladder = _mini_ladder()
    controller = BenchmarkController(ladder, tmp_path, process_mode="fake")
    report = controller.run()

    assert len(report.levels) == 1
    level_report = report.levels[0]
    assert level_report.outcome == "passed", level_report.reasons
    assert level_report.stats is not None
    # The fake processes emit synthetic sampling stats, so a baseline must be recorded.
    assert "sd15" in report.tier_baselines_its

    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / f"level_{ladder[0].id}.json").exists()
    assert (tmp_path / f"level_{ladder[0].id}.log").exists()
    assert "Deliberate" in report.suggested_bridge_data.models_to_load


@pytest.mark.e2e
def test_validation_soak_runs_after_ramp(tmp_path: Path) -> None:
    """After the ramp, a stage-V soak runs the synthesized config under sustained generated load."""
    ladder = _mini_ladder()
    controller = BenchmarkController(ladder, tmp_path, process_mode="fake", validate=True, soak_seconds=18.0)
    report = controller.run()

    validation = [level for level in report.levels if level.level.stage == "V"]
    assert len(validation) == 1
    soak = validation[0]
    assert soak.level.axis == "validation"
    # The generating source fed the worker for the soak period, so real jobs flowed through.
    assert soak.stats is not None
    assert soak.stats.num_jobs_completed >= 1, soak.reasons
    assert soak.stats.num_jobs_faulted == 0

    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Validation (sustained load)" in report_md
    assert (tmp_path / f"level_{soak.level.id}.json").exists()


@pytest.mark.e2e
def test_no_validate_skips_soak(tmp_path: Path) -> None:
    """`validate=False` produces no stage-V level."""
    ladder = _mini_ladder()
    report = BenchmarkController(ladder, tmp_path, process_mode="fake", validate=False).run()
    assert not any(level.level.stage == "V" for level in report.levels)


@pytest.mark.e2e
def test_failed_axis_skips_higher_rungs(tmp_path: Path) -> None:
    """A failing level stops higher rungs on its axis; the ramp itself continues."""
    scenario = ScenarioSpec(
        name="never-arrives",
        image_jobs=[CannedImageJobSpec(count=2)],
        # Steady arrival at 0.1 jobs/min: the first job is available immediately, but the
        # second only "arrives" after 10 minutes — far beyond the 8s level timeout — so the
        # level can never complete all of its expected jobs and is guaranteed to time out.
        arrival_kind="steady",
        arrival_rate_per_minute=0.1,
    )
    base = _mini_ladder()[0]
    failing = base.model_copy(
        update={
            "id": "B-sd15-slow1",
            "stage": "B",
            "axis": "slow",
            "rung": 1,
            "scenario": scenario,
            "timeout_seconds": 8.0,
            "establishes_tier_baseline": False,
        },
    )
    dependent = failing.model_copy(update={"id": "B-sd15-slow2", "rung": 2})
    ladder = [*_mini_ladder(), failing, dependent]

    controller = BenchmarkController(ladder, tmp_path, process_mode="fake")
    report = controller.run()

    outcomes = {level.level.id: level.outcome for level in report.levels}
    assert outcomes[ladder[0].id] == "passed"
    assert outcomes["B-sd15-slow1"] == "failed"
    assert outcomes["B-sd15-slow2"] == "skipped"
    skip_report = next(level for level in report.levels if level.level.id == "B-sd15-slow2")
    assert any("axis" in reason for reason in skip_report.reasons)


def test_crashed_subprocess_recorded_and_ramp_survives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A level subprocess that dies without a result is a 'crashed' finding, not a ramp abort."""

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=args, returncode=139, stdout="", stderr="segfault")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    ladder = _mini_ladder()
    controller = BenchmarkController(ladder, tmp_path, process_mode="fake")
    report = controller.run()

    assert report.levels[0].outcome == "crashed"
    assert any(finding.kind == "crash" for finding in report.levels[0].findings)
    assert (tmp_path / "report.md").exists()


def test_hung_subprocess_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A level subprocess that exceeds its timeout is killed and recorded as a hang."""

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.TimeoutExpired(cmd="level_runner", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    ladder = _mini_ladder()
    controller = BenchmarkController(ladder, tmp_path, process_mode="fake")
    report = controller.run()

    assert report.levels[0].outcome == "crashed_hang"
    assert any(finding.kind == "hang" for finding in report.levels[0].findings)


@pytest.mark.e2e
def test_resume_reuses_existing_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--resume evaluates stored results instead of re-running subprocesses."""
    ladder = _mini_ladder()
    first_report = BenchmarkController(ladder, tmp_path, process_mode="fake").run()
    assert first_report.levels[0].outcome == "passed"

    def _fail_if_called(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise AssertionError("resume must not re-run level subprocesses")

    monkeypatch.setattr(subprocess, "run", _fail_if_called)

    resumed_report = BenchmarkController(ladder, tmp_path, process_mode="fake", resume=True).run()
    assert resumed_report.levels[0].outcome == "passed"
    assert "sd15" in resumed_report.tier_baselines_its


def test_skip_downloads_skips_networked_levels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--skip-downloads pre-flight-skips network levels without running anything."""

    def _fail_if_called(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise AssertionError("skipped levels must not spawn subprocesses")

    monkeypatch.setattr(subprocess, "run", _fail_if_called)

    ladder = build_default_ladder(
        LadderOptions(
            tiers=["sd15"],
            include_concurrency=False,
            include_features=False,
            include_alchemy=False,
            include_downloads=True,
        ),
    )
    download_levels = [level for level in ladder if level.requires_network]
    assert download_levels

    controller = BenchmarkController(download_levels, tmp_path, process_mode="fake", skip_downloads=True)
    report = controller.run()
    assert all(level.outcome == "skipped" for level in report.levels)
