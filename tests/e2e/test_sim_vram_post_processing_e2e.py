"""End-to-end smoke tests for simulated post-processing VRAM pressure.

These drive the full worker against a real spawned fake inference process whose post-processing phase
allocates against a shared simulated-VRAM ledger. They prove the harness mechanism end to end: the *same*
job and fault profile stalls-and-recovers on an over-committed simulated card but completes on a roomy one.
The deterministic ledger arithmetic is unit-tested in
``tests/process_management/test_sim_vram_harness.py``; here the point is that the stall reaches the real
post-processing watchdog and produces a genuine process recovery.

The simulated over-commit is modeled with phantom sibling residency seeded directly on the ledger (VRAM a
real upscaling process cannot reclaim, exactly the cross-process case), plus the one real fake process's
own weights/context. The VRAM budget is left off so these isolate the post-processing-stall mechanism
rather than the admission budget (that interaction is covered by the planner and overlap-gate tests).
"""

from __future__ import annotations

import multiprocessing

import pytest

from horde_worker_regen.harness import HarnessConfig, run_harness_async
from horde_worker_regen.process_management.simulation._canned_scenarios import make_simple_scenario
from horde_worker_regen.process_management.simulation.fault_injection import FaultProfile
from horde_worker_regen.process_management.simulation.sim_vram import (
    SimProcessVram,
    SimVramLedger,
    SimVramSpec,
)

# The single real fake process's own footprint (an ~SDXL checkpoint plus its CUDA context).
_REAL_WEIGHTS_MB = 4900.0
_REAL_CONTEXT_MB = 1354.0

# Two phantom sibling residents the real process cannot reclaim (only the orchestrator could).
_PHANTOM_SIBLINGS = [
    SimProcessVram(process_id=900, weights_mb=3500.0, context_mb=_REAL_CONTEXT_MB),
    SimProcessVram(process_id=901, weights_mb=3500.0, context_mb=_REAL_CONTEXT_MB),
]

# The post-processing peak the job's upscaler needs after sampling.
_PP_PEAK_MB = 6000

# A 16 GB card is over-committed for the peak once the siblings are charged; 48 GB is roomy. With siblings
# + the real process's own residency at ~16 GB, freeing only the real process's own model still leaves less
# than the peak free (the stall), whereas the large card has ample room.
_OVERCOMMITTED_TOTAL_MB = 16375.0
_ROOMY_TOTAL_MB = 49152.0

# The post-processing watchdog fires at post_process_timeout (floored at 15) + 3 * max_batch; pin both low
# so a stall is reaped quickly, and disable retry so the reaped job faults once instead of re-stalling.
_BRIDGE_OVERRIDES: dict[str, object] = {
    "max_threads": 1,
    "max_batch": 1,
    "post_process_timeout": 15,
    "max_inference_attempts": 1,
}


def _seed_ledger(manager: multiprocessing.managers.SyncManager, total_mb: float) -> SimVramLedger:
    """Build a manager-backed ledger seeded with a card total and the phantom sibling residency."""
    ledger = SimVramLedger.from_manager(manager)
    SimVramSpec(device_index=0, total_vram_mb=total_mb, processes=_PHANTOM_SIBLINGS).seed(ledger)
    return ledger


@pytest.mark.e2e
async def test_post_processing_stall_on_overcommitted_card_recovers_and_faults() -> None:
    """An over-committed simulated card stalls the job's post-processing; the watchdog reaps and faults it.

    With retry disabled, the stalled post-processing process is detected and replaced (a real recovery) and
    its job is reported faulted, never lost and never wedging the run. This is the post-processing
    over-commit dynamic reproduced without a GPU.
    """
    with multiprocessing.Manager() as manager:
        ledger = _seed_ledger(manager, _OVERCOMMITTED_TOTAL_MB)
        result = await run_harness_async(
            HarnessConfig(
                scenario=make_simple_scenario(1),
                process_mode="fake",
                skip_api=True,
                timeout_seconds=90.0,
                job_delay_seconds=0.05,
                inference_fault_profile=FaultProfile(post_processing_peak_mb=_PP_PEAK_MB),
                sim_vram_ledger=ledger,
                sim_inference_weights_mb=_REAL_WEIGHTS_MB,
                sim_inference_context_mb=_REAL_CONTEXT_MB,
                bridge_data_overrides=_BRIDGE_OVERRIDES,
            ),
        )

    assert not result.timed_out, result.failure_summary()
    assert result.all_jobs_accounted_for, result.failure_summary()
    # The fault is asserted via the auditor's submitted-faulted count, not the tracker's num_jobs_faulted:
    # the latter is incremented only on a real horde submit, which skip_api bypasses, so it stays 0 here
    # even though the job genuinely faulted and was reported.
    assert result.num_jobs_submitted_faulted >= 1, result.failure_summary()
    assert result.audit_failures == []


@pytest.mark.e2e
async def test_post_processing_completes_on_roomy_card() -> None:
    """The same job and fault profile complete cleanly when the simulated card has room for the peak.

    The only difference from the stall case is the card's total VRAM, isolating the post-processing peak as
    the cause: with headroom the upscaler allocates and the job finishes, with none it stalls.
    """
    with multiprocessing.Manager() as manager:
        ledger = _seed_ledger(manager, _ROOMY_TOTAL_MB)
        result = await run_harness_async(
            HarnessConfig(
                scenario=make_simple_scenario(1),
                process_mode="fake",
                skip_api=True,
                timeout_seconds=60.0,
                job_delay_seconds=0.05,
                inference_fault_profile=FaultProfile(post_processing_peak_mb=_PP_PEAK_MB),
                sim_vram_ledger=ledger,
                sim_inference_weights_mb=_REAL_WEIGHTS_MB,
                sim_inference_context_mb=_REAL_CONTEXT_MB,
                bridge_data_overrides=_BRIDGE_OVERRIDES,
            ),
        )

    assert result.succeeded, result.failure_summary()
