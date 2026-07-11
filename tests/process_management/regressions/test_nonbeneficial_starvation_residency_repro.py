"""RED reproduction of a non-beneficial whole-card residency and associated wedge.

The live incident had two idle inference processes on a 16 GB card.  An SDXL head was initially deferred
while post-processing and a queued sibling preload held VRAM.  After the short starvation grace the arbiter
received ``idle_contexts_teardownable=True`` merely because an idle sibling existed.  It therefore emitted
``REDUCE_LIVE_CONTEXTS`` even though the rejected-peak calculation returned a target of four processes and
only two were live.  The actuator could not reduce ``2 -> 4``, but still marked the head exclusive, stopped
the disaggregated component/VAE lanes, and recorded whole-card residency.  The card subsequently reported
enough free VRAM for the SDXL head, yet the false residency/grace persisted until SOS rebuilt the pools.

The tests state the following invariant:
* a starvation teardown is actionable only when its computed target is below the live process count;
* a no-op reduction must not acquire exclusivity, stop service lanes, or open recovery-suppression grace;
* the invariant is independent of queue followers, the steady-state residency flag, disaggregation, and card
  size.  Those dimensions model the nearby configurations in which the same false-positive signal can occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import (
    StreamForecast,
)
from horde_worker_regen.process_management.resources.vram_arbiter import (
    ActuatorCommandKind,
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _PreloadActuation,
    _WholeCardDemandOutcome,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import (
    _make_inference_scheduler,
)

_HEAD_MODEL = "Nova Anime XL"
_INCIDENT_TOTAL_MB = 16_375.0
_INCIDENT_WEIGHTS_MB = 9_044.0
_INCIDENT_FREE_AFTER_DRAIN_MB = 13_945.0
_STARVED_SECONDS = 30.0


@dataclass(frozen=True)
class ResidencyCase:
    """One topology/configuration cell for the non-beneficial teardown invariant."""

    live_processes: int
    computed_target: int
    whole_card_enabled: bool
    disaggregation_enabled: bool
    total_vram_mb: float
    follower_models: tuple[str, ...]


_CASES = (
    ResidencyCase(
        2,
        4,
        False,
        True,
        _INCIDENT_TOTAL_MB,
        ("Deliberate 3.0", "Zeipher Female Model"),
    ),
    ResidencyCase(2, 2, True, False, _INCIDENT_TOTAL_MB, ()),
    ResidencyCase(1, 4, False, False, _INCIDENT_TOTAL_MB, ("SDXL 1.0",)),
    ResidencyCase(4, 4, True, True, 24_576.0, ("sd15-a", "sd15-b", "sdxl-b")),
)


def _forecast(total_vram_mb: float) -> StreamForecast:
    """Return the post-drain forecast recorded in the incident: the SDXL head fits co-resident."""
    first_context_mb = 1_354.0
    free_if_alone_mb = total_vram_mb - first_context_mb
    return StreamForecast(
        weights_mb=_INCIDENT_WEIGHTS_MB,
        footprint_mb=_INCIDENT_WEIGHTS_MB,
        reserve_mb=2_048.0,
        base_reserve_mb=2_048.0,
        free_now_mb=min(_INCIDENT_FREE_AFTER_DRAIN_MB, free_if_alone_mb),
        free_if_alone_mb=free_if_alone_mb,
        free_after_model_evict_mb=free_if_alone_mb - 243.0,
        total_vram_mb=total_vram_mb,
        per_process_overhead_mb=first_context_mb,
        marginal_process_overhead_mb=243.0,
    )


async def _scheduler_for(case: ResidencyCase):  # noqa: ANN202
    """Build the incident topology with a tracked SDXL head and optional queued followers."""
    process_map = ProcessMap(
        {
            process_id: make_mock_process_info(
                process_id,
                model_name=None if process_id == 0 else f"resident-{process_id}",
                state=HordeProcessState.WAITING_FOR_JOB,
            )
            for process_id in range(case.live_processes)
        },
    )
    job_tracker = JobTracker()
    head = make_job_pop_response(_HEAD_MODEL)
    await track_popped_job_async(job_tracker, head)
    for model in case.follower_models:
        await track_popped_job_async(job_tracker, make_job_pop_response(model))

    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(
            enable_vram_budget=True,
            whole_card_exclusive_residency=case.whole_card_enabled,
            enable_pipeline_disaggregation=case.disaggregation_enabled,
            image_models_to_load=[_HEAD_MODEL, *case.follower_models],
        ),
        max_inference=max(case.live_processes, 1),
        device_free_mb=_INCIDENT_FREE_AFTER_DRAIN_MB,
    )
    scheduler._is_disaggregation_class_eligible = lambda _job: (
        case.disaggregation_enabled
    )
    return scheduler, head, process_map[0]


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=lambda case: f"live{case.live_processes}-target{case.computed_target}",
)
async def test_starvation_request_does_not_offer_a_non_reducing_context_teardown(
    case: ResidencyCase,
) -> None:
    """The scheduler must not tell the arbiter contexts are teardownable when its target cannot lower the pool."""
    scheduler, head, target = await _scheduler_for(case)
    forecast = _forecast(case.total_vram_mb)
    scheduler._forecast_streaming = Mock(return_value=forecast)  # type: ignore[method-assign]
    scheduler._decide_whole_card_demand = Mock(  # type: ignore[method-assign]
        return_value=_WholeCardDemandOutcome.FALL_THROUGH,
    )
    scheduler._context_reduction_demand = Mock(  # type: ignore[method-assign]
        return_value=(case.computed_target, False),
    )
    scheduler._has_reclaimable_idle_model = Mock(return_value=False)  # type: ignore[method-assign]
    scheduler._head_starved_seconds = Mock(return_value=_STARVED_SECONDS)  # type: ignore[method-assign]
    scheduler._measured_admission_candidate_delta_mb = Mock(  # type: ignore[method-assign]
        return_value=_INCIDENT_WEIGHTS_MB,
    )

    arbiter = VramArbiter()
    arbiter.begin_cycle(
        MeasuredVramSnapshot(
            devices={
                0: DeviceVramState(
                    total_vram_mb=case.total_vram_mb,
                    baseline_mb=0.0,
                    committed_vram_mb=0.0,
                    planned_unmaterialized_mb=3_436.0,
                    committed_is_stale=False,
                    device_free_mb=8_890.0,
                ),
            },
        ),
    )
    scheduler._vram_arbiter = arbiter
    scheduler._execute_preload_actuations = Mock()  # type: ignore[method-assign]

    assert (
        scheduler._admit_preload_under_budget(head, target, is_head_blocker=True)
        is False
    )

    commands = scheduler._execute_preload_actuations.call_args.args[0]
    assert ActuatorCommandKind.REDUCE_LIVE_CONTEXTS not in {
        command.kind for command in commands
    }, (
        f"computed target {case.computed_target} cannot reduce {case.live_processes} live processes"
    )


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=lambda case: f"live{case.live_processes}-target{case.computed_target}",
)
async def test_non_reducing_actuation_does_not_acquire_residency_or_recovery_grace(
    case: ResidencyCase,
) -> None:
    """A stale/repeated REDUCE command that cannot remove a process must be a side-effect-free no-op."""
    scheduler, head, target = await _scheduler_for(case)
    scheduler._preload_actuation = _PreloadActuation(
        job=head,
        available_process=target,
        forecast=_forecast(case.total_vram_mb),
        max_resident=case.computed_target,
    )
    scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]

    reduced = scheduler.reduce_live_contexts(None)

    assert reduced is False
    assert scheduler._job_tracker.is_admitted_exclusive(head) is False
    assert scheduler._residency_state(None).model is None
    assert scheduler.whole_card_residency_grace_active() is False
    scheduler.unload_models_from_vram.assert_not_called()
