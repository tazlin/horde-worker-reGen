"""End-to-end executor run in fake mode: the catalog runs on one warm worker and synthesizes a report.

The capability-engine replacement for ``test_controller_fake``. It drives :class:`ProbeExecutor`
against the synthetic worker (no GPU), so it exercises the real orchestration: the topological plan,
the supervisor's run/skip decisions, the per-tier baseline reference, and the recommendation synthesis,
all on a single reused worker rather than a cold boot per probe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.catalog import CatalogOptions
from horde_worker_regen.benchmark.capabilities.executor import ProbeExecutor
from horde_worker_regen.benchmark.enums import BenchTier

pytestmark = pytest.mark.e2e


async def test_executor_runs_static_catalog_and_synthesizes(tmp_path: Path) -> None:
    """Every static SD1.5 probe proves out on the warm worker, and the recommendation reflects it."""
    executor = ProbeExecutor(
        catalog_options=CatalogOptions(
            tiers=[BenchTier.SD15],
            jobs_per_level=2,
            include_features=False,
            include_alchemy=False,
        ),
        process_mode="fake",
        run_soak=False,
        out_dir=tmp_path,
    )

    report = await executor.run_async()

    assert report.probes, "expected the catalog to produce probe results"
    not_proven = [(r.capability.slug, r.reasons) for r in report.probes if r.verdict is not CapabilityVerdict.PROVEN]
    assert not not_proven, f"some probes did not prove out: {not_proven}"

    # The baseline established the tier it/s reference the criteria gate compares later probes against.
    assert "sd15" in report.tier_baselines_its

    # Concurrency probes carry their value in the capability magnitude, so the recommendation lifts to it.
    assert report.suggested_bridge_data.max_threads >= 2
    assert report.suggested_bridge_data.queue_size >= 2

    # The report was persisted for the report/monitor subcommands to read back.
    assert (tmp_path / "report.json").exists()


async def test_executor_only_probe_runs_just_that_capability() -> None:
    """``only_probe`` narrows the run to a single capability (the baseline), skipping synthesis soak."""
    executor = ProbeExecutor(
        catalog_options=CatalogOptions(tiers=[BenchTier.SD15], jobs_per_level=1),
        process_mode="fake",
        run_soak=False,
        only_probe="sd15-baseline",
    )

    report = await executor.run_async()

    assert [probe.capability.kind for probe in report.probes] == [CapabilityKind.BASELINE]
    assert report.probes[0].verdict is CapabilityVerdict.PROVEN
