"""The give-up path must read the same structural-wedge verdict the wedge assessment does.

A structural queue wedge is excused while the scheduler is deliberately holding the queue: a whole-card
model establishing residency, a heavy head loading, a RAM reclaim cycle, backing-off inference starts, or
inference actually in progress. If the give-up path recomputes that verdict with a narrower set of
excuses than ``assess_wedge`` applies, it faults the very backlog the scheduler is holding on purpose.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from tests.process_management.conftest import (
    FakeClock,
    make_job_pop_response,
    make_test_recovery_coordinator,
    track_popped_job_async,
)


class TestGiveUpHonoursTheSameWedgeExcuses:
    """A wedge the assessment excuses must not be actionable by the give-up path."""

    @pytest.mark.parametrize(
        "excusing_grace",
        [
            "whole_card_residency_grace_active",
            "heavy_head_load_grace_active",
        ],
    )
    async def test_excused_structural_wedge_does_not_fault_the_backlog(self, excusing_grace: str) -> None:
        """A held queue is not a wedge, so the give-up leaves its pending work alone.

        The scheduler asserts these graces precisely when it is holding the queue to let capacity land.
        Faulting that backlog discards work the card is about to serve, and the horde reissues it.
        """
        clock = FakeClock()
        coordinator = make_test_recovery_coordinator(
            clock=clock,
            structural_wedge=True,
            **{excusing_grace: True},
        )
        job_tracker = coordinator._job_tracker

        held = await track_popped_job_async(job_tracker, make_job_pop_response())

        assert coordinator.assess_wedge() is False
        coordinator.give_up_on_wedged_jobs()

        assert held in job_tracker.jobs_pending_inference

    async def test_inference_in_progress_also_excuses_the_backlog(self) -> None:
        """Work actively running is the strongest disproof of a wedge, so nothing is faulted."""
        clock = FakeClock()
        coordinator = make_test_recovery_coordinator(
            clock=clock,
            structural_wedge=True,
            lane_state=HordeProcessState.INFERENCE_STARTING,
        )
        job_tracker = coordinator._job_tracker

        queued = await track_popped_job_async(job_tracker, make_job_pop_response())

        assert coordinator.assess_wedge() is False
        coordinator.give_up_on_wedged_jobs()

        assert queued in job_tracker.jobs_pending_inference

    async def test_an_unexcused_structural_wedge_still_faults_the_backlog(self) -> None:
        """The control: with no excuse active, the give-up remains the safety valve it exists to be.

        Narrowing what the give-up acts on must not disarm it, or an genuinely wedged queue would sit
        forever instead of being handed back to the horde.
        """
        clock = FakeClock()
        coordinator = make_test_recovery_coordinator(clock=clock, structural_wedge=True)
        job_tracker = coordinator._job_tracker

        stuck = await track_popped_job_async(job_tracker, make_job_pop_response())

        assert coordinator.assess_wedge() is True
        coordinator.give_up_on_wedged_jobs()

        assert stuck not in job_tracker.jobs_pending_inference
