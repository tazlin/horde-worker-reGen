"""End-to-end smoke tests for simulated post-processing VRAM pressure on the dedicated lane.

These drive the full worker against real spawned fake processes: the fake inference process generates the
images and the fake post-processing lane allocates each job's post-processing peak against a shared
simulated-VRAM ledger. Together they prove that finished inference is never forfeited to a post-processing
VRAM shortfall, by exercising both defenses end to end:

- With the VRAM budget on (the default), the admission gate sees the over-committed card and never
  dispatches the unfittable chain: after the admission-patience window it submits a no-image fault
  (:func:`test_post_processing_ages_out_to_fault_on_overcommitted_card`).
- With the budget off, the job is dispatched and stalls the lane; the silence watchdog reaps and replaces
  it, the orphan watchdog requeues the job, and once the re-attempt budget is spent the job submits its raw
  no-image fault (:func:`test_post_processing_stall_recovers_to_fault_without_budget_gate`). This is the
  recovery net for a chain that was admitted (fit the estimate) but stalled in reality.

The same job and fault profile complete cleanly on a roomy card, isolating the peak as the cause. The
deterministic ledger arithmetic is unit-tested in ``tests/process_management/test_sim_vram_harness.py``.

The simulated over-commit is modeled with phantom sibling residency seeded directly on the ledger (VRAM the
lane cannot reclaim), so the peak allocation cannot fit no matter what the lane does.
"""

from __future__ import annotations

import multiprocessing

import pytest

from horde_worker_regen.harness import HarnessConfig, run_harness_async
from horde_worker_regen.process_management.simulation._canned_scenarios import make_post_processing_scenario
from horde_worker_regen.process_management.simulation.fault_injection import FaultProfile
from horde_worker_regen.process_management.simulation.sim_vram import (
    SimProcessVram,
    SimVramLedger,
    SimVramSpec,
)

# Phantom sibling residents the lane cannot reclaim (only the orchestrator could).
_PHANTOM_SIBLINGS = [
    SimProcessVram(process_id=900, weights_mb=6000.0, context_mb=1354.0),
    SimProcessVram(process_id=901, weights_mb=6000.0, context_mb=1354.0),
]

# The post-processing peak the job's upscaler needs.
_PP_PEAK_MB = 6000

# A 16 GB card is over-committed for the peak once the siblings are charged; 48 GB is roomy.
_OVERCOMMITTED_TOTAL_MB = 16375.0
_ROOMY_TOTAL_MB = 49152.0

# The post-processing watchdog fires at post_process_timeout (floored at 15) + 3 * max_batch; pin both low
# so a stalled lane is reaped quickly. The lane teardown marks the in-flight job's result known-lost, so
# the orphan watchdog requeues it without waiting out its grace; after the bounded re-attempts the job
# is reported as a no-image fault.
_BRIDGE_OVERRIDES: dict[str, object] = {
    "max_threads": 1,
    "max_batch": 1,
    "post_process_timeout": 15,
    "allow_post_processing": True,
}


def _seed_ledger(manager: multiprocessing.managers.SyncManager, total_mb: float) -> SimVramLedger:
    """Build a manager-backed ledger seeded with a card total and the phantom sibling residency."""
    ledger = SimVramLedger.from_manager(manager)
    SimVramSpec(device_index=0, total_vram_mb=total_mb, processes=_PHANTOM_SIBLINGS).seed(ledger)
    return ledger


# Every scenario spawns real OS child processes through the harness, so the module is opt-in via -m slow.
pytestmark = pytest.mark.slow


@pytest.mark.e2e
async def test_post_processing_ages_out_to_fault_on_overcommitted_card() -> None:
    """With the budget gate on, the unfittable chain is never dispatched; it is faulted without images.

    The admission gate compares the estimated post-processing peak (plus the reserve) against the card's
    free VRAM, which the over-commit drives below the requirement. Rather than dispatch the chain into a
    stall, the gate defers it and, after the admission-patience window, faults the job without images so the
    horde can reissue it to another worker. The lane never runs the job.
    """
    with multiprocessing.Manager() as manager:
        ledger = _seed_ledger(manager, _OVERCOMMITTED_TOTAL_MB)
        result = await run_harness_async(
            HarnessConfig(
                scenario=make_post_processing_scenario(1),
                process_mode="fake",
                skip_api=True,
                timeout_seconds=180.0,
                job_delay_seconds=0.05,
                post_process_fault_profile=FaultProfile(post_processing_peak_mb=_PP_PEAK_MB),
                sim_vram_ledger=ledger,
                bridge_data_overrides=_BRIDGE_OVERRIDES,
            ),
        )

    assert not result.timed_out, result.failure_summary()
    assert result.all_jobs_accounted_for, result.failure_summary()
    assert result.num_jobs_completed >= 1, result.failure_summary()
    assert result.num_jobs_submitted_faulted >= 1, result.failure_summary()
    assert result.audit_failures == []


@pytest.mark.e2e
async def test_post_processing_stall_recovers_to_fault_without_budget_gate() -> None:
    """With the budget gate off, a dispatched chain stalls the lane and is recovered to a no-image fault.

    Disabling the admission gate is the way to reach the recovery net for a chain that *was* admitted (its
    estimate fit) yet stalled in reality. The stalled lane is detected by the silence watchdog and replaced
    (a real recovery); the orphan watchdog requeues the job onto the fresh lane, which stalls the same way;
    once the re-attempt budget is spent the job is submitted faulted without images.
    Finished inference is never forfeited to a post-processing failure.
    """
    with multiprocessing.Manager() as manager:
        ledger = _seed_ledger(manager, _OVERCOMMITTED_TOTAL_MB)
        result = await run_harness_async(
            HarnessConfig(
                scenario=make_post_processing_scenario(1),
                process_mode="fake",
                skip_api=True,
                timeout_seconds=180.0,
                job_delay_seconds=0.05,
                post_process_fault_profile=FaultProfile(post_processing_peak_mb=_PP_PEAK_MB),
                sim_vram_ledger=ledger,
                bridge_data_overrides={**_BRIDGE_OVERRIDES, "enable_vram_budget": False},
            ),
        )

    assert not result.timed_out, result.failure_summary()
    assert result.all_jobs_accounted_for, result.failure_summary()
    assert result.num_jobs_completed >= 1, result.failure_summary()
    assert result.num_jobs_submitted_faulted >= 1, result.failure_summary()
    assert result.audit_failures == []


@pytest.mark.e2e
async def test_post_processing_completes_on_roomy_card() -> None:
    """The same job and fault profile complete cleanly when the simulated card has room for the peak.

    The only difference from the stall case is the card's total VRAM, isolating the post-processing peak
    as the cause: with headroom the lane's allocation fits and the job finishes first try.
    """
    with multiprocessing.Manager() as manager:
        ledger = _seed_ledger(manager, _ROOMY_TOTAL_MB)
        result = await run_harness_async(
            HarnessConfig(
                scenario=make_post_processing_scenario(1),
                process_mode="fake",
                skip_api=True,
                timeout_seconds=60.0,
                job_delay_seconds=0.05,
                post_process_fault_profile=FaultProfile(post_processing_peak_mb=_PP_PEAK_MB),
                sim_vram_ledger=ledger,
                bridge_data_overrides=_BRIDGE_OVERRIDES,
            ),
        )

    assert result.succeeded, result.failure_summary()
