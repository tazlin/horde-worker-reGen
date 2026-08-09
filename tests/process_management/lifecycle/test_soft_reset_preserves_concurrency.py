"""A save-our-ship soft reset must rebuild the pools without shedding a concurrency lane.

Cutting ``effective_max_threads`` on every soft reset let a transient wedge, including one provoked by
aggressive co-sampling tripping a sampler watchdog, ratchet worker throughput down and outlast its cause.
The soft reset still rebuilds the pools (recovery is unchanged) and the escalation policy still counts the
reset toward give-up; only the concurrency reduction is demoted to a warning.

``TestSoftResetFaultAttribution`` covers the other half: the job the rebuild interrupts is attributed to the
replacement that took its slot, since the lifecycle layer replaced a healthy process on its own initiative
and there is no crash for an operator to go looking for.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


class TestSoftResetPreservesConcurrency:
    """The soft reset rebuilds the pools but leaves the configured concurrency cap intact."""

    def test_soft_reset_does_not_reduce_effective_max_threads(self) -> None:
        """The rebuild happens, but the concurrency cap is left at its configured value."""
        pm = make_testable_process_manager(max_threads=3)
        coordinator = pm._recovery_coordinator
        coordinator._process_lifecycle = Mock()
        coordinator._inference_scheduler = Mock()
        before = coordinator._runtime_config.effective_max_threads
        # Headroom to reduce, so an "unchanged" assertion is meaningful rather than floored at 1.
        assert before >= 2

        coordinator.perform_soft_reset()

        assert coordinator._runtime_config.effective_max_threads == before
        coordinator._process_lifecycle.rebuild_inference_pool.assert_called_once()
        coordinator._process_lifecycle.rebuild_safety_pool.assert_called_once()

    def test_repeated_soft_resets_never_ratchet_concurrency_down(self) -> None:
        """Several soft resets in a row still leave the concurrency cap untouched."""
        pm = make_testable_process_manager(max_threads=3)
        coordinator = pm._recovery_coordinator
        coordinator._process_lifecycle = Mock()
        coordinator._inference_scheduler = Mock()
        before = coordinator._runtime_config.effective_max_threads

        for _ in range(3):
            coordinator.perform_soft_reset()

        assert coordinator._runtime_config.effective_max_threads == before


_SLOT_ID = 0
"""The inference slot the attribution scenarios replace."""


async def _slot_running_its_last_attempt(pm: object) -> tuple[object, object]:
    """Return a lifecycle whose slot owns a job with no attempts left, so the next failure is terminal."""
    lifecycle = pm._process_lifecycle  # type: ignore[attr-defined]
    lifecycle._end_inference_process = Mock()
    lifecycle._start_inference_process = Mock()
    lifecycle._request_inference_process_start = Mock()
    lifecycle._job_tracker.set_retry_policy(1)

    job = make_job_pop_response()
    await track_popped_job_async(lifecycle._job_tracker, job)
    slot = make_mock_process_info(
        _SLOT_ID,
        model_name="stable_diffusion",
        state=HordeProcessState.INFERENCE_STARTING,
        process_type=HordeProcessType.INFERENCE,
    )
    slot.record_inference_ownership(job, attempt_ordinal=1)
    lifecycle._process_map[_SLOT_ID] = slot
    return lifecycle, (job, slot)


def _fault_reason_for(lifecycle: object, job: object) -> str | None:
    """Return the fault reason recorded against a job."""
    tracked = lifecycle._job_tracker.get_tracked_job(job.id_)  # type: ignore[attr-defined]
    assert tracked is not None
    return tracked.fault_reason  # type: ignore[no-any-return]


class TestSoftResetFaultAttribution:
    """A job faulted by a deliberate replacement is attributed to the replacement, not to a crash."""

    async def test_deliberate_replacement_names_itself_in_the_job_fault(self) -> None:
        """The reason the lifecycle replaced the slot rides along as the job's fault attribution."""
        pm = make_testable_process_manager()
        lifecycle, (job, slot) = await _slot_running_its_last_attempt(pm)

        lifecycle._replace_inference_process(slot, intentional_reason="soft reset: queue deadlock")

        reason = _fault_reason_for(lifecycle, job)
        assert reason is not None
        assert "soft reset: queue deadlock" in reason
        assert "deliberate" in reason

    async def test_reclaim_cycle_names_itself_in_the_job_fault(self) -> None:
        """A RAM-reclaim cycle is deliberate too, and is attributed as such."""
        pm = make_testable_process_manager()
        lifecycle, (job, slot) = await _slot_running_its_last_attempt(pm)

        lifecycle._replace_inference_process(slot, intentional_reclaim=True)

        reason = _fault_reason_for(lifecycle, job)
        assert reason is not None
        assert "reclaim RAM" in reason

    async def test_a_real_crash_is_still_attributed_to_the_process(self) -> None:
        """A replacement with no deliberate reason keeps the ordinary inference-failure attribution."""
        pm = make_testable_process_manager()
        lifecycle, (job, slot) = await _slot_running_its_last_attempt(pm)

        lifecycle._replace_inference_process(slot)

        assert _fault_reason_for(lifecycle, job) == "inference failure"

    async def test_an_explicit_fault_reason_still_wins(self) -> None:
        """A caller-supplied reason (e.g. a resource fault) is not overwritten by the replacement's own."""
        pm = make_testable_process_manager()
        lifecycle, (job, slot) = await _slot_running_its_last_attempt(pm)

        lifecycle._replace_inference_process(
            slot,
            intentional_reason="maintenance-mode pool reload",
            resource_fault_reason="paged sampling watchdog",
        )

        assert _fault_reason_for(lifecycle, job) == "paged sampling watchdog"
