"""The materialisation request built from the scheduling snapshot, and the core that evaluates it."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import predict_job_weight_mb
from horde_worker_regen.process_management.resources.vram_arbiter import VramRequest, VramRequestKind
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.admission.materialization import (
    build_materialization_request,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.workload_flow import DISPATCH_ADMISSION_FLOW
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


async def _worker(
    clock_value: list[float],
) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse, HordeProcessInfo]:
    """A three-slot single-card worker with a tracked head and a running job on another slot."""
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
        clock=lambda: clock_value[0],
    )
    scheduler._process_map.get_free_vram_mb = Mock(return_value=5000.0)  # type: ignore[method-assign]
    scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=16000.0)  # type: ignore[method-assign]
    head = make_job_pop_response(model="stable_diffusion")
    running = make_job_pop_response(model="other")
    await track_popped_job_async(tracker, head)
    await track_popped_job_async(tracker, running)
    await mark_job_in_progress_async(tracker, running)
    scheduler.head_admission.track_head_starvation(str(head.id_), work_in_progress=False)
    clock_value[0] += 30.0
    return scheduler, head, target


@pytest.mark.parametrize("is_head_of_queue", [False, True])
@pytest.mark.parametrize("override_mb", [None, 1200.0])
@pytest.mark.parametrize("nets_own", [False, True])
async def test_request_carries_the_snapshot_pricing(
    is_head_of_queue: bool, override_mb: float | None, nets_own: bool
) -> None:
    """The request names the job and slot, prices through ``pricing``, and nets the own reservation on demand."""
    clock_value = [5_000.0]
    scheduler, head, target = await _worker(clock_value)
    if nets_own:
        # The netting case is the clearance re-price of a dispatched job: its staging reservation is live
        # only while the job is in progress, since the arbiter snapshot reconciles the flow against in-flight jobs.
        await mark_job_in_progress_async(scheduler._job_tracker, head)  # type: ignore[attr-defined]
        scheduler._record_dispatch_reservation(head, target, baseline="stable_diffusion_1", staging_only=True)  # type: ignore[attr-defined]
    arbiter = scheduler._ensure_preload_arbiter()  # type: ignore[attr-defined]
    snapshot = scheduler.snapshot()
    job_id = str(head.id_)

    built = build_materialization_request(
        snapshot,
        job_id,
        0,
        arbiter=arbiter,
        is_head_of_queue=is_head_of_queue,
        head_outstanding_mb=None,
        candidate_delta_override_mb=override_mb,
        nets_own_dispatch_reservation=nets_own,
    )

    request = built.request
    job = snapshot.queue.jobs[job_id]
    assert request.kind is VramRequestKind.MONOLITHIC_DISPATCH
    assert request.job_label == "stable_diffusion"
    assert request.baseline == job.baseline
    assert request.device_index is None
    assert built.device_index is None
    assert request.target_process_id == 0
    assert request.accepted_work is True
    assert request.head_job_id == job_id
    assert request.is_head_of_queue is is_head_of_queue
    assert request.starved_seconds == pytest.approx(30.0)
    assert request.candidate_weights_mb == predict_job_weight_mb(head, job.baseline)
    assert request.can_reduce_live_contexts is False
    expected_delta = (
        override_mb
        if override_mb is not None
        else pricing.candidate_delta_mb(snapshot, head, job.baseline, process_id=0, disaggregated=False)
    )
    assert request.candidate_delta_mb == expected_delta
    assert built.candidate_delta_mb == expected_delta
    own_staging_mb = snapshot.services.reserve_ledger.planned_charge_for_unit(
        DISPATCH_ADMISSION_FLOW, job_id, dict(snapshot.card(None).reserved_by_pid)
    )
    assert (own_staging_mb > 0.0) is nets_own
    assert request.own_dispatch_unmaterialized_mb == own_staging_mb
    if not is_head_of_queue:
        assert request.idle_contexts_teardownable is False


async def test_core_evaluates_the_built_request_against_one_freeze() -> None:
    """The scheduler hands the arbiter the snapshot-built request and freezes a private arbiter once per call."""
    clock_value = [5_000.0]
    scheduler, head, target = await _worker(clock_value)
    arbiter = scheduler._ensure_preload_arbiter()  # type: ignore[attr-defined]
    seen: list[VramRequest] = []
    real_evaluate = arbiter.evaluate
    freezes = 0
    real_begin_cycle = arbiter.begin_cycle

    def capture(request: VramRequest):  # noqa: ANN202
        seen.append(request)
        return real_evaluate(request)

    def count_freeze(*args: object, **kwargs: object) -> None:
        nonlocal freezes
        freezes += 1
        real_begin_cycle(*args, **kwargs)  # type: ignore[arg-type]

    arbiter.evaluate = capture  # type: ignore[method-assign]
    arbiter.begin_cycle = count_freeze  # type: ignore[method-assign]
    scheduler._execute_preload_actuations = Mock(return_value=())  # type: ignore[method-assign]

    outcome = scheduler._evaluate_materialization_admission(  # type: ignore[attr-defined]
        head,
        target,
        is_head_of_queue=True,
        head_outstanding_mb=None,
        candidate_delta_override_mb=1200.0,
        nets_own_dispatch_reservation=True,
    )

    assert freezes == 1
    assert len(seen) == 1
    built = build_materialization_request(
        scheduler.snapshot(),
        str(head.id_),
        0,
        arbiter=arbiter,
        is_head_of_queue=True,
        head_outstanding_mb=None,
        candidate_delta_override_mb=1200.0,
        nets_own_dispatch_reservation=True,
    )
    assert seen[0] == built.request
    assert outcome.candidate_delta_mb == 1200.0
    assert outcome.device_index is None


async def test_core_refuses_an_untracked_job() -> None:
    """A job without an id has no snapshot entry and cannot be priced."""
    clock_value = [5_000.0]
    scheduler, _head, target = await _worker(clock_value)
    untracked = make_job_pop_response(model="stable_diffusion").model_copy(update={"id_": None})

    with pytest.raises(ValueError, match="tracked job"):
        scheduler._evaluate_materialization_admission(  # type: ignore[attr-defined]
            untracked,
            target,
            is_head_of_queue=True,
            head_outstanding_mb=None,
        )
