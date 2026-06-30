"""The capability-probe engine: prove a machine can or can't run a given feature/model/level.

This package is the coherent core of the benchmark subsystem. It is deliberately import-light and
torch-free (every module except the probe runner and executor imports only pydantic plus the pure
``criteria`` / ``scenarios`` / stats helpers), so progress, app-state, and pytest collection never
drag torch into the orchestrator process.

Only the cycle-safe leaf modules are re-exported here. The report layer imports
:mod:`~horde_worker_regen.benchmark.capabilities.stats`, which runs this ``__init__``; re-exporting the
modules that import the report layer back (``result``, ``supervisor``, ``catalog``, ``probe_runner``)
would form an import cycle, so those are imported by their full module path until the report rewire
inverts that dependency.
"""

from __future__ import annotations

from horde_worker_regen.benchmark.capabilities.capability import (
    Capability,
    CapabilityKind,
    CapabilityVerdict,
)
from horde_worker_regen.benchmark.capabilities.plan import CapabilityPlan, build_plan
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.capabilities.stats import (
    level_stats_from_harness_result,
    level_stats_from_metrics,
)
from horde_worker_regen.benchmark.capabilities.timing import ProbeTiming, probe_timing

__all__ = [
    "Capability",
    "CapabilityKind",
    "CapabilityPlan",
    "CapabilityProbe",
    "CapabilityVerdict",
    "ProbeTiming",
    "build_plan",
    "level_stats_from_harness_result",
    "level_stats_from_metrics",
    "probe_timing",
]
