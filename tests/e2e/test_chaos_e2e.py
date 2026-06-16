"""End-to-end chaos scenarios driving the full worker against misbehaving child processes.

Each test spawns real fake child processes (via the harness ``fake`` mode) scripted with a
:class:`FaultProfile`, then asserts the *intended* resilient outcome: every job is accounted for
(completed or faulted, never lost), no audit invariant is violated, and the worker does not wedge.
Probes targeting a known gap are marked ``xfail``. Watchdog timeouts are shrunk via ``bridge_data_overrides``
so a genuinely-wedged run resolves quickly instead of burning the wall clock.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.harness import HarnessConfig, run_harness_async
from horde_worker_regen.process_management._canned_scenarios import make_simple_scenario
from horde_worker_regen.process_management.fault_injection import FaultProfile

# The bridge-data model enforces sane minimums (e.g. inference_step_timeout >= 15), so a wedge probe
# cannot lean on tiny watchdog timeouts. Instead it bounds the whole run with a short timeout_seconds:
# crash detection is immediate (is_alive), and an undetected wedge simply runs out the clock.
_WEDGE_TIMEOUT_SECONDS = 15.0

# Detecting a *hang* (as opposed to a crash, which is caught immediately via is_alive) requires
# waiting out a full inference_step_timeout of silence. That floor is 15s (the bridge-data minimum),
# so a hang probe must allow detection + recovery + the remaining jobs to finish well past it.
_HANG_DETECT_TIMEOUT_SECONDS = 45.0


@pytest.mark.e2e
async def test_oom_fault_is_reported_and_pipeline_continues() -> None:
    """A job that reports an out-of-memory fault must be submitted as faulted, not lost, and the rest flow."""
    scenario = make_simple_scenario(4)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=60.0,
            inference_fault_profile=FaultProfile(oom_on_job_n=2),
        ),
    )

    assert not result.timed_out, result.failure_summary()
    assert result.all_jobs_accounted_for
    assert result.audit_failures == []
    assert result.num_jobs_submitted_faulted >= 1


@pytest.mark.e2e
async def test_midjob_crash_recovers_and_keeps_serving() -> None:
    """A child that crashes mid-inference must be reaped and replaced, faulting only its in-flight job."""
    scenario = make_simple_scenario(4)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=_WEDGE_TIMEOUT_SECONDS,
            job_delay_seconds=0.05,
            inference_fault_profile=FaultProfile(crash_on_job_n=2),
        ),
    )

    assert not result.timed_out, result.failure_summary()
    assert result.all_jobs_accounted_for
    assert result.audit_failures == []


@pytest.mark.e2e
async def test_slow_job_with_heartbeats_is_not_falsely_killed() -> None:
    """A job slower than usual but still emitting step heartbeats must complete, not be killed as hung."""
    scenario = make_simple_scenario(2)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=60.0,
            job_delay_seconds=0.2,
            inference_fault_profile=FaultProfile(slow_factor=5.0),
        ),
    )

    assert result.succeeded, result.failure_summary()


@pytest.mark.e2e
async def test_stale_launch_duplicate_result_is_ignored() -> None:
    """A duplicate result stamped with a stale launch identifier must not double-finalize its job."""
    scenario = make_simple_scenario(3)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=60.0,
            inference_fault_profile=FaultProfile(corrupt_on_job_n=1),
        ),
    )

    assert not result.timed_out, result.failure_summary()
    assert result.all_jobs_accounted_for
    assert result.audit_failures == []


@pytest.mark.e2e
@pytest.mark.xfail(
    reason="A crash-loop breaker now quarantines the perpetually-crashing slot, but "
    "with no inference capacity left the worker has nothing to recover to and the run still times out. "
    "Closing this needs the save-our-ship / limp-by fallback (give up the unservable jobs cleanly).",
    strict=False,
)
async def test_inference_crash_on_start_is_circuit_broken() -> None:
    """A permanently-failing inference start must be quarantined so the worker does not wedge."""
    scenario = make_simple_scenario(2)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=_WEDGE_TIMEOUT_SECONDS,
            inference_fault_profile=FaultProfile(crash_on_start=True),
        ),
    )

    assert not result.timed_out, result.failure_summary()


@pytest.mark.e2e
@pytest.mark.xfail(
    reason="A safety process that fails on every start wedges all image jobs in safety. The "
    "crash-loop breaker covers inference slots, not the safety pool; recovering this needs the "
    "save-our-ship / limp-by fallback.",
    strict=False,
)
async def test_safety_crash_on_start_does_not_wedge_image_jobs() -> None:
    """Image jobs must not be wedged forever by a safety process that crashes on every start."""
    scenario = make_simple_scenario(2)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=_WEDGE_TIMEOUT_SECONDS,
            safety_fault_profile=FaultProfile(crash_on_start=True),
        ),
    )

    assert not result.timed_out, result.failure_summary()


@pytest.mark.e2e
async def test_hang_at_zero_percent_is_recovered() -> None:
    """A child that accepts a job then wedges before its first step must be detected and replaced."""
    scenario = make_simple_scenario(3)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=_HANG_DETECT_TIMEOUT_SECONDS,
            inference_fault_profile=FaultProfile(hang_after_n_jobs=1),
        ),
    )

    assert not result.timed_out, result.failure_summary()
    assert result.all_jobs_accounted_for
