"""Tests for the run-history data layer: enumerating past runs and diffing two of them."""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityKind, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.result import (
    CapabilityProbeResult,
    CapabilityReport,
    Finding,
    MachineInfo,
    SuggestedBridgeData,
)
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.history import compare_reports, list_runs, load_report

_BASELINE = Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE)


def _write_report(
    run_dir: Path,
    *,
    run_id: str,
    created_at: float,
    verdict: CapabilityVerdict = CapabilityVerdict.PROVEN,
    suggested: SuggestedBridgeData | None = None,
    findings: int = 0,
) -> None:
    """Write a minimal but valid report.json into a run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    finding_list = [Finding(kind="oom", level_id=_BASELINE.slug, evidence="x") for _ in range(findings)]  # type: ignore[arg-type]
    report = CapabilityReport(
        run_id=run_id,
        created_at=created_at,
        machine=MachineInfo(gpu_name="Test GPU", total_vram_mb=16000),
        probes=[CapabilityProbeResult(capability=_BASELINE, verdict=verdict, findings=finding_list)],
        suggested_bridge_data=suggested or SuggestedBridgeData(),
        tier_baselines_its={"sd15": 5.0},
    )
    (run_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")


def test_list_runs_returns_newest_first_and_skips_junk(tmp_path: Path) -> None:
    """Runs are summarized newest-first; a directory without a readable report is ignored."""
    _write_report(tmp_path / "20260101-000000", run_id="old", created_at=100.0)
    _write_report(tmp_path / "20260201-000000", run_id="new", created_at=200.0, findings=2)
    (tmp_path / "not-a-run").mkdir()  # no report.json -> skipped, never raises
    (tmp_path / "20260301-000000").mkdir()
    (tmp_path / "20260301-000000" / "report.json").write_text("{ broken", encoding="utf-8")

    summaries = list_runs(tmp_path)

    assert [summary.run_id for summary in summaries] == ["new", "old"]
    assert summaries[0].num_findings == 2
    assert summaries[0].gpu_name == "Test GPU"


def test_list_runs_on_missing_root_is_empty(tmp_path: Path) -> None:
    """A non-existent results root yields no runs rather than raising."""
    assert list_runs(tmp_path / "does-not-exist") == []


def test_compare_reports_surfaces_outcome_suggested_and_findings_changes(tmp_path: Path) -> None:
    """A diff reports a probe that regressed, a changed suggested field, and the findings delta."""
    older_suggested = SuggestedBridgeData(max_threads=2)
    newer_suggested = SuggestedBridgeData(max_threads=1)
    _write_report(
        tmp_path / "old", run_id="old", created_at=1.0, verdict=CapabilityVerdict.PROVEN, suggested=older_suggested
    )
    _write_report(
        tmp_path / "new",
        run_id="new",
        created_at=2.0,
        verdict=CapabilityVerdict.DISPROVEN,
        suggested=newer_suggested,
        findings=1,
    )

    older = load_report(tmp_path / "old")
    newer = load_report(tmp_path / "new")
    assert older is not None and newer is not None

    comparison = compare_reports(older, newer)

    assert comparison.has_changes
    assert any(change.new == "disproven" for change in comparison.outcome_changes)
    assert any(
        change.field == "max_threads" and change.old == "2" and change.new == "1"
        for change in comparison.suggested_changes
    )
    assert comparison.findings_delta == 1


def test_compare_identical_reports_has_no_changes(tmp_path: Path) -> None:
    """Comparing a run to an identical run reports no differences."""
    _write_report(tmp_path / "a", run_id="a", created_at=1.0)
    _write_report(tmp_path / "b", run_id="b", created_at=2.0)
    older = load_report(tmp_path / "a")
    newer = load_report(tmp_path / "b")
    assert older is not None and newer is not None
    assert compare_reports(older, newer).has_changes is False
