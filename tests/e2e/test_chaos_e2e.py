"""End-to-end chaos scenarios driving the full worker against misbehaving child processes.

Each test spawns real fake child processes (via the harness ``fake`` mode) scripted with a
:class:`FaultProfile`, then asserts the *intended* resilient outcome: every job is accounted for
(completed or faulted, never lost), no audit invariant is violated, and the worker does not wedge.
Probes targeting a known gap are marked ``xfail``. Watchdog timeouts are shrunk via ``bridge_data_overrides``
so a genuinely-wedged run resolves quickly instead of burning the wall clock.
"""

from __future__ import annotations

import sys

import pytest

from horde_worker_regen.harness import HarnessConfig, run_harness_async
from horde_worker_regen.process_management.simulation._canned_scenarios import make_simple_scenario
from horde_worker_regen.process_management.simulation.fault_injection import FaultProfile

# Spawning a fresh child re-imports the whole stack and is several times slower on Windows than on the
# Linux CI runner. A wedge/recovery probe pays that cost once per re-spawn, so any budget sized for CI is
# too tight locally on Windows (the recovery still succeeds, it just runs past the clock). Scale the
# recovery-bounded budgets by this factor on Windows; CI (Linux) keeps the original, tight values so a
# genuine regression still surfaces quickly there.
_SPAWN_SLOWDOWN = 4.0 if sys.platform == "win32" else 1.0

# The bridge-data model enforces sane minimums (e.g. inference_step_timeout >= 15), so a wedge probe
# cannot lean on tiny watchdog timeouts. Instead it bounds the whole run with a short timeout_seconds:
# crash detection is immediate (is_alive), and an undetected wedge simply runs out the clock.
_WEDGE_TIMEOUT_SECONDS = 15.0

# Detecting a *hang* (as opposed to a crash, which is caught immediately via is_alive) requires
# waiting out a full inference_step_timeout of silence. That floor is 15s (the bridge-data minimum), and
# every fresh process hangs after its first job, so the run pays one detect+respawn cycle per job. The
# budget must clear all of them plus the per-respawn spawn cost (the Windows term is what ``_SPAWN_SLOWDOWN``
# covers).
_HANG_DETECT_TIMEOUT_SECONDS = 45.0 * _SPAWN_SLOWDOWN

# A deterministic crash-on-start is only recoverable by the save-our-ship escalation: the crash-loop
# breaker must quarantine the pool (several process re-spawns), the supervisor attempts a soft reset,
# then gives up and abandons ship. Each step is bounded by real process-spawn cost (notably slow on
# Windows), so this allows generous headroom over the observed ~20-30s rather than a tight wedge bound.
_SAVE_OUR_SHIP_TIMEOUT_SECONDS = 60.0 * _SPAWN_SLOWDOWN


# Every scenario spawns real OS child processes through the harness, so the module is opt-in via -m slow.
pytestmark = pytest.mark.slow


@pytest.mark.e2e
async def test_oom_fault_is_retried_and_pipeline_continues() -> None:
    """A one-off out-of-memory fault must be given a degraded retry (not lost) and the pipeline must flow.

    With bounded retry enabled (the default), a transient OOM is requeued for a degraded, isolated retry
    rather than reported faulted on the first failure; the job recovers and every job is accounted for.
    """
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


@pytest.mark.e2e
async def test_oom_fault_is_reported_faulted_when_retry_disabled() -> None:
    """With retry disabled, an out-of-memory fault is reported faulted (not lost) on the first failure."""
    scenario = make_simple_scenario(4)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=60.0,
            inference_fault_profile=FaultProfile(oom_on_job_n=2),
            bridge_data_overrides={"max_inference_attempts": 1},
        ),
    )

    assert not result.timed_out, result.failure_summary()
    assert result.all_jobs_accounted_for
    assert result.audit_failures == []
    assert result.num_jobs_submitted_faulted >= 1


@pytest.mark.e2e
async def test_midjob_crash_recovers_and_keeps_serving() -> None:
    """A child that crashes mid-inference must be reaped and replaced, faulting only its in-flight job.

    Retry is disabled here to isolate what this probe asserts: the *watchdog's* crash detection and slot
    replacement. The crash-then-retry-to-completion path is covered by
    ``test_midjob_crash_is_retried_to_completion``.
    """
    scenario = make_simple_scenario(4)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=_WEDGE_TIMEOUT_SECONDS,
            job_delay_seconds=0.05,
            inference_fault_profile=FaultProfile(crash_on_job_n=2),
            bridge_data_overrides={"max_inference_attempts": 1},
        ),
    )

    assert not result.timed_out, result.failure_summary()
    assert result.all_jobs_accounted_for
    assert result.audit_failures == []


@pytest.mark.e2e
async def test_midjob_crash_is_retried_to_completion() -> None:
    """With bounded retry, a job whose slot crashed mid-inference is requeued and completes, not faulted.

    Each fresh slot the fake scripts crashes on its own second job, so recovering all four jobs takes a
    few replace-and-retry cycles; the generous budget allows for those process re-spawns and model
    re-preloads. The point is that nothing is lost or faulted: the crashed work is retried to success.
    """
    scenario = make_simple_scenario(4)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=50.0,
            job_delay_seconds=0.05,
            inference_fault_profile=FaultProfile(crash_on_job_n=2),
        ),
    )

    assert not result.timed_out, result.failure_summary()
    assert result.all_jobs_accounted_for
    assert result.audit_failures == []
    assert result.num_jobs_submitted_faulted == 0


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
async def test_inference_crash_on_start_is_circuit_broken() -> None:
    """A permanently-failing inference start must be circuit-broken and abandoned, not left to wedge.

    The crash-loop breaker quarantines the slot, the recovery supervisor attempts a soft reset, and when
    that cannot restore a working pool it abandons ship cleanly (the last resort) so the worker stops
    instead of spinning forever.
    """
    scenario = make_simple_scenario(2)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=_SAVE_OUR_SHIP_TIMEOUT_SECONDS,
            inference_fault_profile=FaultProfile(crash_on_start=True),
        ),
    )

    assert not result.timed_out, result.failure_summary()


@pytest.mark.e2e
async def test_safety_crash_on_start_does_not_wedge_image_jobs() -> None:
    """A safety process that crashes on every start must not wedge the worker forever.

    The safety pool is rebuilt on each crash (including a crash during startup, before it ever loads);
    once it has crash-looped past its threshold the recovery supervisor recognizes the pool as failing
    and abandons ship cleanly rather than holding jobs in safety indefinitely.
    """
    scenario = make_simple_scenario(2)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=_SAVE_OUR_SHIP_TIMEOUT_SECONDS,
            safety_fault_profile=FaultProfile(crash_on_start=True),
        ),
    )

    assert not result.timed_out, result.failure_summary()


@pytest.mark.e2e
async def test_hang_at_zero_percent_is_recovered() -> None:
    """A child that accepts a job then wedges before its first step must be detected and replaced.

    Retry is disabled so this probe isolates hang *detection and recovery*: re-feeding a job to a slot
    that wedges on every second job would just incur another full ``inference_step_timeout`` of silence
    per attempt, which is the retry policy's concern, not the watchdog's.
    """
    scenario = make_simple_scenario(3)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=_HANG_DETECT_TIMEOUT_SECONDS,
            inference_fault_profile=FaultProfile(hang_after_n_jobs=1),
            # A hang *at zero percent* emits no first step, so the watchdog reaps it on the first-step grace,
            # not the per-step timeout. The production default for that grace (90s) is sized for a cold
            # combined-checkpoint load and would, on its own, exceed this probe's budget. Pin both timeouts to
            # the bridge-data floor so a wedged-before-first-step slot is detected in the ~15s of silence the
            # budget above assumes (the effective first-step grace is floored at inference_step_timeout, so
            # both must be lowered together).
            bridge_data_overrides={
                "max_inference_attempts": 1,
                "inference_step_timeout": 15,
                "inference_first_step_timeout": 15,
            },
        ),
    )

    assert not result.timed_out, result.failure_summary()
    assert result.all_jobs_accounted_for
