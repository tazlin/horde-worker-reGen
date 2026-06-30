"""Enumerate and compare past benchmark runs from their on-disk reports.

Every ramp leaves a ``report.json`` in its own ``benchmark_results/<timestamp>/`` directory, but the
durable app state remembers only the most recent run. This module turns that directory of reports into
a navigable history: :func:`list_runs` summarizes every run newest-first, and :func:`compare_reports`
diffs two runs so an operator can answer "did this run regress against the last one?".

It is deliberately import-light (only the result models, no executor/hordelib chain) so the TUI can
load it lazily without paying the benchmark's heavy import cost.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.capabilities.capability import CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.result import CapabilityReport

DEFAULT_RESULTS_ROOT = Path("benchmark_results")
"""Where ramps write their timestamped output directories by default."""

_REPORT_FILENAME = "report.json"


class RunSummary(BaseModel):
    """A one-line digest of a past run, for the history list (no per-level detail)."""

    run_dir: str
    run_id: str
    created_at: float = 0.0
    gpu_name: str | None = None
    worker_version: str = ""
    levels_passed: int = 0
    levels_total: int = 0
    num_findings: int = 0


def load_report(run_dir: Path) -> CapabilityReport | None:
    """Load one run's ``report.json``, returning None (never raising) when it is absent or unreadable.

    Loads via the result model directly to keep this module off the heavy benchmark import chain.
    """
    report_path = run_dir / _REPORT_FILENAME
    if not report_path.exists():
        return None
    try:
        return CapabilityReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as load_error:
        logger.debug(f"Could not load benchmark report {report_path}: {load_error}")
        return None


def summarize_report(run_dir: Path, report: CapabilityReport) -> RunSummary:
    """Project a loaded report into its history-list summary."""
    probes_proven = sum(1 for probe in report.probes if probe.verdict is CapabilityVerdict.PROVEN)
    return RunSummary(
        run_dir=str(run_dir),
        run_id=report.run_id or run_dir.name,
        created_at=report.created_at,
        gpu_name=report.machine.gpu_name,
        worker_version=report.worker_version,
        levels_passed=probes_proven,
        levels_total=len(report.probes),
        num_findings=len(report.findings),
    )


def list_runs(results_root: Path = DEFAULT_RESULTS_ROOT) -> list[RunSummary]:
    """Return a summary of every run under ``results_root``, newest first.

    Tolerant by design: a directory without a readable ``report.json`` (an in-progress or corrupted
    run) is skipped rather than raising, so the history view never breaks on one bad run.
    """
    if not results_root.is_dir():
        return []
    summaries: list[RunSummary] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        report = load_report(child)
        if report is not None:
            summaries.append(summarize_report(child, report))
    summaries.sort(key=lambda summary: summary.created_at, reverse=True)
    return summaries


class FieldChange(BaseModel):
    """A single field that differs between two runs, rendered as old -> new strings."""

    field: str
    old: str
    new: str


class OutcomeChange(BaseModel):
    """A level whose outcome differs between two runs (``-`` when absent from one run)."""

    level_id: str
    old: str
    new: str


class ReportComparison(BaseModel):
    """The differences between an older run and a newer run, for the compare view."""

    older_run_id: str
    newer_run_id: str
    outcome_changes: list[OutcomeChange] = Field(default_factory=list)
    capability_changes: list[FieldChange] = Field(default_factory=list)
    suggested_changes: list[FieldChange] = Field(default_factory=list)
    baseline_its_changes: list[FieldChange] = Field(default_factory=list)
    older_findings: int = 0
    newer_findings: int = 0

    @property
    def findings_delta(self) -> int:
        """Change in robustness-finding count (newer minus older); positive means more problems."""
        return self.newer_findings - self.older_findings

    @property
    def has_changes(self) -> bool:
        """Whether the two runs differ in any compared dimension."""
        return bool(
            self.outcome_changes
            or self.capability_changes
            or self.suggested_changes
            or self.baseline_its_changes
            or self.findings_delta,
        )


def _outcomes_by_level(report: CapabilityReport) -> dict[str, str]:
    return {probe.capability.slug: str(probe.verdict) for probe in report.probes}


def _capability_flags(report: CapabilityReport) -> dict[str, bool]:
    capabilities = report.capabilities
    return {
        "hires_fix": capabilities.supports_hires_fix,
        "post_processing": capabilities.supports_post_processing,
        "controlnet": capabilities.supports_controlnet,
        "qr_code": capabilities.supports_qr_code,
        "alchemy_clip": capabilities.supports_alchemy_clip,
        "alchemy_graph": capabilities.supports_alchemy_graph,
        "alchemy_concurrent": capabilities.supports_alchemy_concurrent,
        "lora": capabilities.supports_lora,
    }


def _suggested_fields(report: CapabilityReport) -> dict[str, str]:
    suggested = report.suggested_bridge_data
    return {
        "max_threads": str(suggested.max_threads),
        "queue_size": str(suggested.queue_size),
        "max_batch": str(suggested.max_batch),
        "allow_lora": str(suggested.allow_lora),
        "allow_controlnet": str(suggested.allow_controlnet),
        "allow_sdxl_controlnet": str(suggested.allow_sdxl_controlnet),
        "allow_post_processing": str(suggested.allow_post_processing),
        "models_to_load": ", ".join(suggested.models_to_load) or "(none)",
        "alchemist": str(suggested.alchemist),
        "alchemy_allow_concurrent": str(suggested.alchemy_allow_concurrent),
    }


def _diff_string_maps(older: dict[str, str], newer: dict[str, str]) -> list[FieldChange]:
    """Return the keys whose values differ, preserving ``newer``'s key order then any older-only keys."""
    keys = list(newer) + [key for key in older if key not in newer]
    changes: list[FieldChange] = []
    for key in keys:
        old_value = older.get(key, "-")
        new_value = newer.get(key, "-")
        if old_value != new_value:
            changes.append(FieldChange(field=key, old=old_value, new=new_value))
    return changes


def compare_reports(older: CapabilityReport, newer: CapabilityReport) -> ReportComparison:
    """Diff two runs across level outcomes, capabilities, the recommendation, and baseline throughput.

    Throughput baselines are compared with one decimal of it/s so a trivial sampling jitter does not
    read as a regression; every other dimension is an exact value comparison.
    """
    older_outcomes = _outcomes_by_level(older)
    newer_outcomes = _outcomes_by_level(newer)
    outcome_changes = [
        OutcomeChange(level_id=change.field, old=change.old, new=change.new)
        for change in _diff_string_maps(older_outcomes, newer_outcomes)
    ]

    capability_changes = _diff_string_maps(
        {flag: str(value) for flag, value in _capability_flags(older).items()},
        {flag: str(value) for flag, value in _capability_flags(newer).items()},
    )

    suggested_changes = _diff_string_maps(_suggested_fields(older), _suggested_fields(newer))

    baseline_its_changes = _diff_string_maps(
        {tier: f"{its:.1f}" for tier, its in older.tier_baselines_its.items()},
        {tier: f"{its:.1f}" for tier, its in newer.tier_baselines_its.items()},
    )

    return ReportComparison(
        older_run_id=older.run_id,
        newer_run_id=newer.run_id,
        outcome_changes=outcome_changes,
        capability_changes=capability_changes,
        suggested_changes=suggested_changes,
        baseline_its_changes=baseline_its_changes,
        older_findings=len(older.findings),
        newer_findings=len(newer.findings),
    )


__all__ = [
    "DEFAULT_RESULTS_ROOT",
    "FieldChange",
    "OutcomeChange",
    "ReportComparison",
    "RunSummary",
    "compare_reports",
    "list_runs",
    "load_report",
    "summarize_report",
]
