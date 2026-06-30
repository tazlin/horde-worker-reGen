"""On-disk per-probe run result for the subprocess path, plus re-exports of the report data models.

The benchmark's report *engine* now lives in the capability package: the records in
:mod:`~horde_worker_regen.benchmark.capabilities.result`, the synthesis in
:mod:`~horde_worker_regen.benchmark.capabilities.recommendation`, and the markdown in
:mod:`~horde_worker_regen.benchmark.capabilities.report_render`. What remains here is the lean on-disk
:class:`LevelRunResult` the isolated subprocess runner writes (and its stats adapter), plus re-exports
of the shared leaf models so existing importers resolve them from one place.
"""

from __future__ import annotations

from pydantic import BaseModel

from horde_worker_regen.benchmark.capabilities.result import (
    Finding,
    HarnessSummary,
    MachineInfo,
    SuggestedBridgeData,
    SuggestionBasis,
    SuggestionDecision,
    TierCapability,
    WorkerCapabilities,
)
from horde_worker_regen.benchmark.capabilities.stats import level_stats_from_metrics
from horde_worker_regen.benchmark.criteria import LevelStats
from horde_worker_regen.process_management.resources.run_metrics import RunMetricsSnapshot


class LevelRunResult(BaseModel):
    """What the isolated subprocess level runner writes to disk: raw outcomes, no policy applied."""

    level_id: str
    harness: HarnessSummary
    metrics: RunMetricsSnapshot | None = None
    runner_error: str | None = None
    """Set when the runner itself failed before/while producing results."""


def compute_level_stats(result: LevelRunResult, *, total_vram_mb: int | None = None) -> LevelStats:
    """Distill a subprocess level runner's on-disk result into the criteria-relevant statistics.

    Thin adapter over :func:`~horde_worker_regen.benchmark.capabilities.stats.level_stats_from_metrics`
    for the subprocess path, whose :class:`LevelRunResult` carries its counts on ``harness`` and its
    metrics alongside.
    """
    return level_stats_from_metrics(result.harness, result.metrics, total_vram_mb=total_vram_mb)


__all__ = [
    "Finding",
    "HarnessSummary",
    "LevelRunResult",
    "MachineInfo",
    "SuggestedBridgeData",
    "SuggestionBasis",
    "SuggestionDecision",
    "TierCapability",
    "WorkerCapabilities",
    "compute_level_stats",
    "level_stats_from_metrics",
]
