"""End-to-end tests that run the full worker lifecycle through the harness.

These spawn real OS child processes (running the protocol-faithful fakes) and run the
real asyncio orchestration loop with the API faked out; no GPU, no network, no
hordelib/torch in any process.

All tests in this module are async so they use pytest-asyncio's managed event loop
instead of ``asyncio.run()``.  Calling ``asyncio.run()`` from inside a test creates a
nested event loop whose ``ProactorEventLoop`` teardown on Windows can race with the
vscode-pytest named-pipe server, causing the pipe to disappear before the final test
report can be sent.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.harness import HarnessConfig, run_harness_async
from horde_worker_regen.process_management.simulation._canned_scenarios import (
    make_alchemy_scenario,
    make_simple_scenario,
)

# Every scenario spawns real OS child processes through the harness, so the module is opt-in via -m slow.
pytestmark = pytest.mark.slow


@pytest.mark.e2e
async def test_full_lifecycle_fake_processes_no_api() -> None:
    """Every job in a small scenario must complete pop → inference → safety → submit."""
    result = await run_harness_async(
        HarnessConfig(
            num_jobs=3,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=90.0,
        ),
    )

    assert not result.timed_out, f"Harness run timed out before the scenario completed ({result.failure_summary()})"
    assert result.num_jobs_faulted == 0, (
        f"Expected 0 faulted jobs, got {result.num_jobs_faulted} ({result.failure_summary()})"
    )
    assert result.num_jobs_completed == 3, (
        f"Expected 3 completed jobs, got {result.num_jobs_completed} ({result.failure_summary()})"
    )
    assert result.succeeded, f"Harness run did not succeed ({result.failure_summary()})"


@pytest.mark.e2e
async def test_run_metrics_flow_through_fake_processes() -> None:
    """Verify per-job records carry stage latencies and the fakes' synthetic phase metrics.

    Exercises the pipe → dispatcher → run-metrics chain end-to-end.
    """
    result = await run_harness_async(
        HarnessConfig(
            num_jobs=2,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=90.0,
        ),
    )

    assert result.succeeded, f"Harness run did not succeed ({result.failure_summary()})"
    assert result.metrics is not None
    assert len(result.metrics.jobs) == 2

    for record in result.metrics.jobs:
        assert record.e2e_seconds is not None and record.e2e_seconds > 0
        assert record.queue_wait_seconds is not None
        assert record.stage_timestamps.get("FINALIZED") is not None
        assert record.phase_metrics is not None, "fake-process job metrics were not correlated"
        assert record.phase_metrics.sampling is not None
        assert record.phase_metrics.vram_used_high_water_mb == 1234

    assert result.metrics.vram_used_high_water_mb_per_process, "no per-process VRAM high-water recorded"
    assert result.metrics.num_process_recoveries == 0
    assert result.metrics.process_crash_events == []


@pytest.mark.e2e
async def test_mixed_image_and_alchemy_scenario() -> None:
    """Image jobs and canned alchemy forms must both complete in the same fake-mode run."""
    result = await run_harness_async(
        HarnessConfig(
            num_jobs=2,
            alchemy_forms=make_alchemy_scenario(["caption", "RealESRGAN_x4plus"], 2),
            process_mode="fake",
            skip_api=True,
            timeout_seconds=90.0,
            bridge_data_overrides={"alchemy_allow_concurrent": True},
        ),
    )

    assert result.succeeded, f"Harness run did not succeed ({result.failure_summary()})"
    assert result.num_jobs_completed == 2
    assert result.num_alchemy_forms_completed == 2
    assert result.num_alchemy_forms_faulted == 0

    # Alchemy form metrics flow through the same run-metrics chain as image jobs.
    assert result.metrics is not None
    alchemy_records = [record for record in result.metrics.jobs if record.is_alchemy]
    assert len(alchemy_records) == 2


@pytest.mark.e2e
async def test_full_lifecycle_with_simulated_inference_time() -> None:
    """Jobs that take nonzero (fake) inference time must still all complete."""
    scenario = make_simple_scenario(2)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            job_delay_seconds=0.5,
            timeout_seconds=90.0,
        ),
    )

    assert result.succeeded, f"Harness run did not succeed ({result.failure_summary()})"
    assert result.num_jobs_completed == len(scenario), (
        f"Expected {len(scenario)} completed jobs, got {result.num_jobs_completed} ({result.failure_summary()})"
    )
