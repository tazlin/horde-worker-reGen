"""The idle-context teardown flag and depth judged in the measured frame the arbiter refuses on.

A card at its edge: the structural context count (total minus peak minus reserve, contexts at a constant)
says several siblings fit beside a whole-card candidate, while the measured identity (device-free minus
noise, with every lane's context physically inside the reading) refuses by a few hundred MB that one idle
sibling context would return. The seams now widen the teardown flag from the measured deficit, size the depth
to exactly the contexts needed, and leave a deficit no idle context can close alone, which is the churn fence.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.admission_identity import TenantLane
from horde_worker_regen.process_management.resources.resource_budget import _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB
from horde_worker_regen.process_management.resources.vram_arbiter import (
    _FIRST_PARTY_TEARDOWN_GRACE_SECONDS,
    ActuatorCommandKind,
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
)
from horde_worker_regen.process_management.scheduling.admission import pricing
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_TOTAL_MB = 24074.0
_NOISE_MB = 1203.7
_CANDIDATE_MB = 19424.0
_SIBLING_RESERVED_MB = 100.0


def _edge_state(*, device_free_mb: float = 20157.0) -> DeviceVramState:
    """Two inference contexts on a 24 GB card whose device-free reading leaves a whole-card head just short."""
    return DeviceVramState(
        total_vram_mb=_TOTAL_MB,
        baseline_mb=1000.0,
        committed_vram_mb=2000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=device_free_mb,
        noise_buffer_mb=_NOISE_MB,
        per_process_reserved_mb={2: _SIBLING_RESERVED_MB, 3: _SIBLING_RESERVED_MB},
        lane_by_process_id={2: TenantLane.INFERENCE_IDLE, 3: TenantLane.INFERENCE_IDLE},
        marginal_mb=_SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB,
    )


async def _scheduler_with_idle_sibling():  # noqa: ANN202
    """The head staged on slot 2 beside an empty idle sibling on slot 3, both reporting a small reservation."""
    target = make_mock_process_info(2, model_name="model_a", state=HordeProcessState.PRELOADED_MODEL)
    sibling = make_mock_process_info(3, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    target.process_reserved_mb = int(_SIBLING_RESERVED_MB)
    sibling.process_reserved_mb = int(_SIBLING_RESERVED_MB)
    process_map = ProcessMap({2: target, 3: sibling})
    job_tracker = JobTracker()
    job = make_job_pop_response("model_a")
    await track_popped_job_async(job_tracker, job)
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(image_models_to_load=["model_a"]),
        max_concurrent=2,
        max_inference=2,
    )
    return scheduler, job, target, sibling


def _head_request(
    *, candidate_mb: float = _CANDIDATE_MB, starved_seconds: float = 0.0, head: bool = True
) -> VramRequest:
    return VramRequest(
        kind=VramRequestKind.MONOLITHIC_DISPATCH,
        job_label="Z-Image-Turbo",
        baseline="z_image_turbo",
        device_index=0,
        target_process_id=2,
        candidate_delta_mb=candidate_mb,
        is_head_of_queue=head,
        head_job_id="head",
        starved_seconds=starved_seconds,
    )


def _arbiter(state: DeviceVramState) -> VramArbiter:
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
    return arbiter


async def test_a_deficit_one_idle_sibling_context_closes_widens_the_flag_and_sizes_the_depth() -> None:
    """Structural count says five contexts fit; the measured deficit is under one sibling's return: tear one down."""
    scheduler, _job, target, _sibling = await _scheduler_with_idle_sibling()
    arbiter = _arbiter(_edge_state())
    request = _head_request()

    deficit = arbiter.measured_deficit_mb(request)
    assert deficit is not None and 0 < deficit < 600, "the replayed edge: short by a few hundred MB"

    widened, max_resident = scheduler._apply_measured_context_teardown(
        request,
        arbiter,
        target,
        structural_max_resident=5,
        device_index=0,
    )

    assert widened.idle_contexts_teardownable is True
    assert max_resident == 1, "two live contexts, one needed: the depth is exactly the contexts needed"

    past_grace = arbiter.evaluate(replace(widened, starved_seconds=_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 1.0))
    assert past_grace.disposition is VramDisposition.DEFER
    assert [command.kind for command in past_grace.required_actuations] == [ActuatorCommandKind.REDUCE_LIVE_CONTEXTS]


async def test_a_deficit_no_idle_context_can_close_is_left_alone() -> None:
    """A card short by gigabytes with one small idle sibling is not torn down: that teardown produces no fit."""
    scheduler, _job, target, _sibling = await _scheduler_with_idle_sibling()
    arbiter = _arbiter(_edge_state(device_free_mb=12000.0))
    request = _head_request()

    unchanged, max_resident = scheduler._apply_measured_context_teardown(
        request,
        arbiter,
        target,
        structural_max_resident=5,
        device_index=0,
    )

    assert unchanged is request
    assert max_resident == 5
    verdict = arbiter.evaluate(replace(request, starved_seconds=_FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 1.0))
    assert ActuatorCommandKind.REDUCE_LIVE_CONTEXTS not in [command.kind for command in verdict.required_actuations]


async def test_a_fitting_candidate_and_a_non_head_are_never_widened() -> None:
    """The measured widening exists for a starved head that does not fit; nothing else is touched."""
    scheduler, _job, target, _sibling = await _scheduler_with_idle_sibling()
    arbiter = _arbiter(_edge_state())

    fitting = _head_request(candidate_mb=3000.0)
    assert scheduler._apply_measured_context_teardown(
        fitting,
        arbiter,
        target,
        structural_max_resident=None,
        device_index=0,
    ) == (fitting, None)

    follower = _head_request(head=False)
    assert scheduler._apply_measured_context_teardown(
        follower,
        arbiter,
        target,
        structural_max_resident=5,
        device_index=0,
    ) == (follower, 5)


async def test_the_structural_judgement_is_kept_and_the_depth_is_the_deeper() -> None:
    """A request the structural frame already flagged keeps its flag; a shallower structural depth wins."""
    scheduler, _job, target, _sibling = await _scheduler_with_idle_sibling()
    arbiter = _arbiter(_edge_state())

    already = replace(_head_request(), idle_contexts_teardownable=True)
    assert scheduler._apply_measured_context_teardown(
        already,
        arbiter,
        target,
        structural_max_resident=1,
        device_index=0,
    ) == (already, 1)


async def test_the_hold_gate_carries_the_widened_flag_into_the_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the dispatch gate, the head's hold reports the sibling context as a rung that closes the deficit."""
    scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
    arbiter = _arbiter(_edge_state())
    scheduler._vram_arbiter = arbiter
    monkeypatch.setattr(pricing, "candidate_delta_mb", lambda *_args, **_kwargs: _CANDIDATE_MB)
    recorded: list[dict[str, object]] = []
    scheduler._decision_sink = lambda **kwargs: recorded.append(kwargs)

    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

    inputs = recorded[-1]["inputs"]
    assert isinstance(inputs, dict)
    assert inputs["room_closable"] is True
    assert inputs["rung_idle_sibling_context_mb"] > inputs["room_deficit_mb"]
