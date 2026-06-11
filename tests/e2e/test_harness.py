"""End-to-end tests that run the full worker lifecycle through the harness.

These spawn real OS child processes (running the protocol-faithful fakes) and run the
real asyncio orchestration loop with the API faked out — no GPU, no network, no
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
from horde_worker_regen.process_management._canned_scenarios import make_simple_scenario


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
