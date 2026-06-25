"""Warm-session end-to-end tests: one worker reused across levels, no per-level respawn."""

from __future__ import annotations

import pytest

from horde_worker_regen.harness import WarmHarnessSession
from horde_worker_regen.process_management.simulation._canned_scenarios import make_alchemy_scenario, make_canned_job

pytestmark = pytest.mark.e2e


async def test_warm_session_reuses_one_worker_across_levels() -> None:
    """Two levels run on the same warm inference processes; only the concurrency cap changes."""
    async with WarmHarnessSession(
        process_mode="fake",
        model_names=["Deliberate", "AlbedoBase XL (SDXL)"],
        max_threads_ceiling=2,
    ) as session:
        manager = session.manager

        result1 = await session.run_level(
            jobs=[make_canned_job("Deliberate") for _ in range(3)],
            threads=1,
            timeout_seconds=60,
        )
        assert result1.num_jobs_completed == 3, result1.failure_summary()
        assert result1.num_jobs_faulted == 0
        launch_ids_after_level1 = {p.process_launch_identifier for p in manager._process_map.get_inference_processes()}
        assert launch_ids_after_level1, "expected at least one warm inference process"

        result2 = await session.run_level(
            jobs=[make_canned_job("AlbedoBase XL (SDXL)") for _ in range(2)],
            threads=2,
            timeout_seconds=60,
        )
        assert result2.num_jobs_completed == 2, result2.failure_summary()
        # Per-level metrics were reset, so level 2 only counts its own jobs.
        assert result2.num_jobs_expected == 2

        launch_ids_after_level2 = {p.process_launch_identifier for p in manager._process_map.get_inference_processes()}
        # The same inference processes served both levels: no per-level respawn = no warm-up cost.
        assert launch_ids_after_level1 == launch_ids_after_level2

        # The effective concurrency cap tracked the second level's thread count.
        assert manager.max_concurrent_inference_processes == 2


async def test_warm_session_runs_an_alchemy_level() -> None:
    """A level mixing an image job and alchemy forms completes on the warm worker."""
    async with WarmHarnessSession(
        process_mode="fake",
        model_names=["Deliberate"],
        max_threads_ceiling=1,
    ) as session:
        result = await session.run_level(
            jobs=[make_canned_job("Deliberate")],
            alchemy_forms=make_alchemy_scenario(["caption"], 2),
            threads=1,
            timeout_seconds=60,
        )
        assert result.num_jobs_completed == 1, result.failure_summary()
        assert result.num_alchemy_forms_completed == 2
