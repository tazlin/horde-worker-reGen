"""Stress scenarios for the full worker lifecycle.

These contrive more demanding job/worker configurations than the basic harness
tests: model switching, fault injection, batches, and queue pressure. Every run
is checked by the JobLifecycleAuditor: no job lost, no job double-submitted,
and the tracker fully drained.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.harness import HarnessConfig, run_harness_async
from horde_worker_regen.process_management.testing._canned_scenarios import (
    make_batch_scenario,
    make_mixed_model_scenario,
    make_simple_scenario,
    make_varied_size_scenario,
)


@pytest.mark.e2e
async def test_mixed_models_force_model_swaps() -> None:
    """Jobs alternating across two models must all complete despite model switching."""
    scenario = make_mixed_model_scenario(6, ["Deliberate", "Anything Diffusion"])
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=90.0,
            bridge_data_overrides={"queue_size": 2},
        ),
    )

    assert result.succeeded, f"audit: {result.audit_failures}"
    assert result.num_jobs_completed == len(scenario)
    assert result.audit_failures == []


@pytest.mark.e2e
async def test_fault_injection_keeps_pipeline_flowing() -> None:
    """Periodic inference faults must be reported and must not lose or wedge other jobs.

    Retry is disabled so this asserts the report-as-fault path directly (a fault that cannot be
    recovered is still submitted, not lost). The recover-via-retry path is covered separately by
    ``test_chaos_e2e``'s OOM and crash retry probes.
    """
    scenario = make_simple_scenario(6)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=90.0,
            fail_every_n=3,
            bridge_data_overrides={"max_inference_attempts": 1},
        ),
    )

    assert not result.timed_out
    assert result.all_jobs_accounted_for
    assert result.audit_failures == []
    # Jobs 3 and 6 fail inference; they must still be submitted (as faults), not lost.
    assert result.num_jobs_submitted_faulted == 2


@pytest.mark.e2e
async def test_batched_jobs_complete() -> None:
    """Jobs generating multiple images each must flow through safety and submit intact."""
    scenario = make_batch_scenario(2, 3)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=90.0,
        ),
    )

    assert result.succeeded, f"audit: {result.audit_failures}"
    assert result.num_jobs_completed == len(scenario)
    assert result.audit_failures == []


@pytest.mark.e2e
async def test_queue_pressure_with_varied_job_sizes() -> None:
    """A deeper queue with mixed job sizes must drain completely under backpressure rules."""
    scenario = make_varied_size_scenario(8)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            job_delay_seconds=0.1,
            timeout_seconds=120.0,
            bridge_data_overrides={"queue_size": 3},
        ),
    )

    assert result.succeeded, f"audit: {result.audit_failures}"
    assert result.num_jobs_completed == len(scenario)
    assert result.audit_failures == []
