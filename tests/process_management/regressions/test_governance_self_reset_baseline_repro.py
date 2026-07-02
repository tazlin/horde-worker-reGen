"""Returning RAM governance to baseline must clear the governance latch but nothing another subsystem owns.

A soft reset rebuilds the process pools, but the RAM pop hold and the governor's shed/draining bookkeeping
live in worker state, not the pool, so a rebuild alone leaves them latched. ``reset_governance_to_baseline``
returns exactly the RAM-governance state to baseline (the next tick re-derives whatever the live host
warrants) while leaving alone flags owned by other subsystems or latched for the session.

The contract these tests pin:

* The RAM pop hold, the shed-card / draining / single-GPU shed records, and the RAM-pressure pop-skip
  reason are all cleared.
* Other pop-skip reasons, the shared self-throttle pause, the operator supervisor pause, the downloads-only
  hold, and the session-latched breakers are all preserved.
"""

from __future__ import annotations

from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.scheduling.governance.ram_governor import WorkerProcessShedState
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


class TestResetGovernanceToBaseline:
    """``reset_governance_to_baseline`` clears the governance latch and preserves foreign state."""

    def test_reset_clears_ram_governance_state_only(self) -> None:
        """Governance state returns to baseline; flags other subsystems own are left untouched."""
        scheduler = _make_inference_scheduler(job_tracker=JobTracker())

        # Seed a full pressure-episode state plus flags that belong to other subsystems.
        scheduler._state.ram_pressure_pop_hold = True
        scheduler._state.last_pop_skipped_reasons = {"ram_pressure": 5, "models": 2}
        scheduler._ram_reclaim_cycle_at = 123.0
        scheduler._ram_pressure_notified = True
        scheduler._ram_governor_state.shed_cards = {0, 1}
        scheduler._ram_governor_state.worker_shed = WorkerProcessShedState(
            planned_process_count=4, shed_process_count=3
        )
        scheduler._ram_governor_state.draining_process_ids = {1}

        scheduler._state.self_throttle_paused = True
        scheduler._state.self_throttle_paused_until = 999.0
        scheduler._state.supervisor_paused = True
        scheduler._state.downloads_only_hold = True
        scheduler._state.post_processing_disabled_by_breaker = True
        scheduler._state.gpu_torch_incompatible = True
        scheduler._state.torch_build_cpu_only = True
        scheduler._state.consecutive_failed_jobs = 3

        scheduler.reset_governance_to_baseline("test")

        # RAM-governance state is back at baseline.
        assert scheduler._state.ram_pressure_pop_hold is False
        assert scheduler._state.last_pop_skipped_reasons == {"models": 2}, "only the RAM-pressure reason is dropped"
        assert scheduler._ram_reclaim_cycle_at == 0.0
        assert scheduler._ram_pressure_notified is False
        assert scheduler._ram_governor_state.shed_cards == set()
        assert scheduler._ram_governor_state.worker_shed is None
        assert scheduler._ram_governor_state.draining_process_ids == set()

        # Flags owned by other subsystems (or session-latched) are preserved.
        assert scheduler._state.self_throttle_paused is True
        assert scheduler._state.self_throttle_paused_until == 999.0
        assert scheduler._state.supervisor_paused is True
        assert scheduler._state.downloads_only_hold is True
        assert scheduler._state.post_processing_disabled_by_breaker is True
        assert scheduler._state.gpu_torch_incompatible is True
        assert scheduler._state.torch_build_cpu_only is True
        assert scheduler._state.consecutive_failed_jobs == 3
