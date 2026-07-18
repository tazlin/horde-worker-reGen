"""The RAM-pressure component-eviction rung: reclaim idle unprotected staged components first.

Before the coarser whole-RAM unload, the scheduler evicts staged checkpoints that no queued or in-flight job
needs, so a queued job's staged model survives. These tests prove that survival (the positive-liveness
contract) and that an all-protected process degrades to the legacy whole-RAM path without wedging.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import (
    HordeEvictComponentsControlMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.component_residency_map import ComponentResidencyMap
from horde_worker_regen.process_management.scheduling.governance import EvictIdleModels
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _checkpoint_snapshots(*identities: str) -> list[object]:
    from horde_worker_regen.process_management.ipc.messages import HeldComponentSnapshot

    return [
        HeldComponentSnapshot(kind="checkpoint", identity=identity, approx_ram_mb=100.0) for identity in identities
    ]


def _sent_evict_message(process_info: object) -> HordeEvictComponentsControlMessage | None:
    """Return the evict message sent to a process, or None if none was sent."""
    send = process_info.pipe_connection.send  # type: ignore[attr-defined]
    for call in send.call_args_list:
        message = call.args[0]
        if isinstance(message, HordeEvictComponentsControlMessage):
            return message
    return None


async def _scheduler_holding(
    *,
    active_model: str,
    staged_identities: tuple[str, ...],
    queued_model: str | None,
) -> tuple[InferenceScheduler, object]:
    """Build a scheduler whose lone idle inference slot holds ``staged_identities`` and, optionally, a queued job."""
    process = make_mock_process_info(0, model_name=active_model, state=HordeProcessState.WAITING_FOR_JOB)
    process_map = ProcessMap({0: process})
    scheduler = _make_inference_scheduler(process_map=process_map)

    residency = ComponentResidencyMap()
    residency.update_from_report(0, 0, _checkpoint_snapshots(*staged_identities))
    scheduler._component_residency_map = residency

    if queued_model is not None:
        await track_popped_job_async(scheduler._job_tracker, make_job_pop_response(queued_model))
    return scheduler, process


class TestQueuedModelSurvives:
    """The rung evicts only the idle, unprotected staged component; a queued job's model survives."""

    async def test_only_unprotected_component_is_evicted(self) -> None:
        """With A active, B queued, C idle-unprotected, the rung evicts exactly C and never B or A."""
        scheduler, process = await _scheduler_holding(
            active_model="A",
            staged_identities=("A", "B", "C"),
            queued_model="B",
        )

        acted = scheduler._evict_unprotected_components_under_pressure()

        assert acted is True
        message = _sent_evict_message(process)
        assert message is not None
        assert message.identities == ["C"]
        # The queued job's model (B) and the tracked active model (A) are never evicted, so B survives to be
        # dispatched: it stays protected and still recorded resident, and its slot is alive and un-faulted.
        assert "B" not in message.identities
        assert "A" not in message.identities
        assert "B" in scheduler._compute_wanted_models()
        assert scheduler._component_residency_map.checkpoint_models_held_on([0]) >= {"A", "B"}
        assert process.is_process_alive() is True

    async def test_governance_rung_short_circuits_the_whole_ram_unload(self) -> None:
        """When the gentle eviction acts, the coarse whole-RAM unload is not run this tick."""
        scheduler, process = await _scheduler_holding(
            active_model="A",
            staged_identities=("A", "B", "C"),
            queued_model="B",
        )
        scheduler.unload_models = Mock(return_value=True)  # type: ignore[method-assign]

        scheduler._execute_governance_actions([EvictIdleModels()])

        assert _sent_evict_message(process) is not None
        scheduler.unload_models.assert_not_called()


class TestAllProtectedDegradesToLegacy:
    """When everything held is protected, the rung no-ops and the legacy whole-RAM path runs without wedging."""

    async def test_no_eviction_when_everything_is_protected(self) -> None:
        """A slot holding only its active model and a queued model has nothing unprotected to evict."""
        scheduler, process = await _scheduler_holding(
            active_model="A",
            staged_identities=("A", "B"),
            queued_model="B",
        )

        acted = scheduler._evict_unprotected_components_under_pressure()

        assert acted is False
        assert _sent_evict_message(process) is None

    async def test_governance_falls_through_to_whole_ram_unload(self) -> None:
        """With nothing evictable, EvictIdleModels degrades to the legacy unload path, no wedge."""
        scheduler, process = await _scheduler_holding(
            active_model="A",
            staged_identities=("A", "B"),
            queued_model="B",
        )
        scheduler.unload_models = Mock(return_value=True)  # type: ignore[method-assign]

        # Must not raise (no wedge); the legacy whole-RAM unload runs because the gentle rung found nothing.
        scheduler._execute_governance_actions([EvictIdleModels()])

        assert _sent_evict_message(process) is None
        scheduler.unload_models.assert_called_once_with(under_pressure=True)


class TestDisabledWhenCacheUntracked:
    """A scheduler without a residency map (budgeted cache off) skips the rung entirely."""

    async def test_no_residency_map_means_no_eviction(self) -> None:
        """The rung is inert when the component cache is untracked, taking the legacy path unchanged."""
        process = make_mock_process_info(0, model_name="A", state=HordeProcessState.WAITING_FOR_JOB)
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: process}))
        assert scheduler._component_residency_map is None

        assert scheduler._evict_unprotected_components_under_pressure() is False
        assert _sent_evict_message(process) is None
