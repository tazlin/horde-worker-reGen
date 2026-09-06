"""The snapshot-built materialisation request is the request the scheduler hands the arbiter."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.vram_arbiter import VramRequest
from horde_worker_regen.process_management.scheduling.admission.materialization import (
    build_materialization_request,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


@pytest.mark.parametrize("is_head_of_queue", [False, True])
@pytest.mark.parametrize("override_mb", [None, 1200.0])
@pytest.mark.parametrize("nets_own", [False, True])
async def test_request_matches_the_scheduler(
    is_head_of_queue: bool, override_mb: float | None, nets_own: bool
) -> None:
    """Every field of the arbiter request agrees, with and without the clearance override and the netting."""
    target = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.WAITING_FOR_JOB)
    target.process_reserved_mb = 2600
    idle = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    idle.process_reserved_mb = 700
    busy = make_mock_process_info(2, model_name="other", state=HordeProcessState.INFERENCE_STARTING)
    tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        process_map=ProcessMap({0: target, 1: idle, 2: busy}),
        job_tracker=tracker,
        device_free_mb=5000.0,
        max_inference=3,
        clock=lambda: 5_000.0,
    )
    scheduler._process_map.get_free_vram_mb = Mock(return_value=5000.0)  # type: ignore[method-assign]
    scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)  # type: ignore[method-assign]
    head = make_job_pop_response(model="stable_diffusion")
    running = make_job_pop_response(model="other")
    await track_popped_job_async(tracker, head)
    await track_popped_job_async(tracker, running)
    await mark_job_in_progress_async(tracker, running)
    scheduler.head_admission.track_head_starvation(str(head.id_), work_in_progress=False)

    arbiter = scheduler._ensure_preload_arbiter()  # type: ignore[attr-defined]
    seen: list[VramRequest] = []
    real_evaluate = arbiter.evaluate

    def capture(request: VramRequest):  # noqa: ANN202
        seen.append(request)
        return real_evaluate(request)

    arbiter.evaluate = capture  # type: ignore[method-assign]
    scheduler._execute_preload_actuations = Mock(return_value=())  # type: ignore[method-assign]
    outcome = scheduler._evaluate_materialization_admission(  # type: ignore[attr-defined]
        head,
        target,
        is_head_of_queue=is_head_of_queue,
        head_outstanding_mb=None,
        candidate_delta_override_mb=override_mb,
        nets_own_dispatch_reservation=nets_own,
    )
    assert len(seen) == 1

    built = build_materialization_request(
        scheduler.snapshot(),
        str(head.id_),
        0,
        arbiter=arbiter,
        is_head_of_queue=is_head_of_queue,
        head_outstanding_mb=None,
        candidate_delta_override_mb=override_mb,
        nets_own_dispatch_reservation=nets_own,
    )
    assert built.request == seen[0]
    assert built.candidate_delta_mb == outcome.candidate_delta_mb
    assert built.device_index == outcome.device_index
    assert built.request.starved_seconds == pytest.approx(scheduler._head_starved_seconds(head))  # type: ignore[attr-defined]
