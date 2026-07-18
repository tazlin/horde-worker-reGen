"""The staged-models provider widens the residency-bias floor at the popper seam.

The popper narrows its pop offer toward resident (and RAM-staged) models while a model-swap backlog
persists. The staged-models provider (checkpoints held in a live inference process's RAM cache) feeds the
narrowed floor beside the VRAM-resident set, so a narrowed pop can also offer work the card can start cheaply
from RAM. These exercise that path through the real ``_apply_residency_advertising_bias``.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_api_sessions,
    make_test_runtime_config,
    track_popped_job_async,
)

_RESIDENT = "ResidentModel"
_QUEUED = "QueuedNonResidentModel"
_STAGED = "StagedModel"
_OTHER = "OtherModel"


def _make_popper(
    *,
    process_map: ProcessMap,
    job_tracker: JobTracker,
    staged_models_provider: Callable[[], frozenset[str]] | None = None,
) -> JobPopper:
    """A JobPopper with mostly-mocked dependencies, for exercising the residency-advertising bias."""
    return JobPopper(
        state=WorkerState(),
        process_map=process_map,
        job_tracker=job_tracker,
        shutdown_manager=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=make_mock_bridge_data()),
        api_sessions=make_test_api_sessions(horde_client_session=Mock(), aiohttp_session=Mock()),
        max_inference_processes=2,
        max_concurrent_inference_processes=1,
        staged_models_provider=staged_models_provider,
    )


async def _backlogged_state() -> tuple[ProcessMap, JobTracker]:
    """A resident model plus a queued job for a different model: the swap-backlog that engages narrowing."""
    process_map = ProcessMap(
        {0: make_mock_process_info(0, model_name=_RESIDENT, state=HordeProcessState.WAITING_FOR_JOB)},
    )
    job_tracker = JobTracker()
    await track_popped_job_async(job_tracker, make_job_pop_response(_QUEUED))
    return process_map, job_tracker


class TestStagedModelsFeedTheFloor:
    """Staged checkpoints join the resident set in the narrowed offer."""

    async def test_staged_model_widens_the_narrowed_offer(self) -> None:
        """During narrowing the floor is (resident | staged) & offered, so a staged model is offered."""
        process_map, job_tracker = await _backlogged_state()
        popper = _make_popper(
            process_map=process_map,
            job_tracker=job_tracker,
            staged_models_provider=lambda: frozenset({_STAGED}),
        )

        advertised = popper._apply_residency_advertising_bias({_RESIDENT, _QUEUED, _STAGED, _OTHER})

        # Narrowed to the resident model plus the RAM-staged one; the non-resident/non-staged models drop.
        assert advertised == {_RESIDENT, _STAGED}

    async def test_empty_staged_set_matches_the_default_provider(self) -> None:
        """An empty staged set is parity with no provider: the floor is the resident set alone."""
        map_a, tracker_a = await _backlogged_state()
        map_b, tracker_b = await _backlogged_state()

        popper_default = _make_popper(process_map=map_a, job_tracker=tracker_a)
        popper_empty = _make_popper(
            process_map=map_b,
            job_tracker=tracker_b,
            staged_models_provider=lambda: frozenset(),
        )

        offered = {_RESIDENT, _QUEUED, _STAGED, _OTHER}
        assert popper_default._apply_residency_advertising_bias(set(offered)) == {_RESIDENT}
        assert popper_empty._apply_residency_advertising_bias(set(offered)) == {_RESIDENT}
