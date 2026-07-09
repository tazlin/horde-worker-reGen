"""Scheduler-side wiring for the admission self-count wedge: the two production incidents, end to end.

The arbiter-level liveness contract lives in
``tests/process_management/resources/test_admission_liveness_matrix.py``. These regressions prove the
scheduler feeds that arbiter the inputs the contract needs: it releases a planned charge whose target has died
(so a re-ask is not blocked by its own stale plan), and it presents a resident, idle candidate as a no-op
dispatch (so the gate releases it rather than pricing a materialisation that never happens).

- 24GB flux exclusive head: a preload was admitted and recorded a reservation, then its target was reclaimed
  before the load materialised. A dead target's reservation never grows, so the reservation decayed by neither
  materialisation nor omission; only excluding dead/ended processes from the in-flight set releases it.
- 8GB SD1.5 resident and idle: the model was already resident and idle on the target, so dispatching it moved
  nothing, yet a naive gate would withhold it because the candidate's activation does not fit a card whose
  free room the resident weights already consume. The resident no-op admit is what releases it.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import (
    HordeProcessState,
    ModelInfo,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
)
from horde_worker_regen.process_management.scheduling.workload_flow import PRELOAD_ADMISSION_FLOW
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _install_cycle(scheduler, state: DeviceVramState) -> None:  # noqa: ANN001
    """Freeze a crafted arbiter cycle on the scheduler's arbiter."""
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
    scheduler._vram_arbiter = arbiter


class TestDeadTargetReleasesPlannedCharge:
    """A planned charge whose target process has died is released, so the head's re-ask is not self-blocked."""

    def test_in_flight_set_excludes_a_dead_target_and_reconcile_drops_the_charge(self) -> None:
        """A LOADING map entry on a PROCESS_ENDED slot leaves the in-flight set, releasing its planned charge.

        The stale map entry can outlive the throttled missing-process recovery that expires it, and the dead
        target's reservation never grows to decay the charge, so without this exclusion the charge would pin the
        overlay at full weight and the re-ask would defer forever on its own footprint.
        """
        dead = make_mock_process_info(0, model_name=None, state=HordeProcessState.PROCESS_ENDED)
        dead.process_reserved_mb = 0
        model_map = HordeModelMap(
            root={
                "flux_head": ModelInfo(
                    horde_model_name="flux_head",
                    horde_model_load_state=ModelLoadState.LOADING,
                    process_id=0,
                ),
            },
        )
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: dead}),
            horde_model_map=model_map,
            bridge_data=make_mock_bridge_data(image_models_to_load=["flux_head"]),
        )
        # A charge was recorded when the (since-dead) preload was admitted.
        scheduler._reserve_ledger.set_planned(
            PRELOAD_ADMISSION_FLOW,
            "0",
            vram_mb=8229.0,
            target_process_id=0,
            reserved_at_admit_mb=0.0,
        )

        assert scheduler._in_flight_admitted_planned_units() == set()

        scheduler._reserve_ledger.reconcile_planned(
            PRELOAD_ADMISSION_FLOW,
            scheduler._in_flight_admitted_planned_units(),
        )
        assert scheduler._reserve_ledger.effective_planned_vram_mb({}) == 0.0

    def test_live_loading_target_keeps_its_charge(self) -> None:
        """A LOADING entry on a live slot stays in the in-flight set: a genuinely-in-flight charge is preserved."""
        live = make_mock_process_info(0, model_name="flux_head", state=HordeProcessState.PRELOADING_MODEL)
        live.process_reserved_mb = 0
        model_map = HordeModelMap(
            root={
                "flux_head": ModelInfo(
                    horde_model_name="flux_head",
                    horde_model_load_state=ModelLoadState.LOADING,
                    process_id=0,
                ),
            },
        )
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: live}),
            horde_model_map=model_map,
            bridge_data=make_mock_bridge_data(image_models_to_load=["flux_head"]),
        )
        assert scheduler._in_flight_admitted_planned_units() == {"0"}


class TestResidentIdleDispatchReleasesImmediately:
    """A dispatch to an already-VRAM-resident idle model releases at once, even over a full card (no-op admit)."""

    async def _resident_scheduler(self):  # noqa: ANN202
        """A scheduler whose head model is resident (LOADED_IN_VRAM) and idle on its target process."""
        target = make_mock_process_info(2, model_name="sd15", state=HordeProcessState.WAITING_FOR_JOB)
        target.process_reserved_mb = 7000
        model_map = HordeModelMap(
            root={
                "sd15": ModelInfo(
                    horde_model_name="sd15",
                    horde_model_load_state=ModelLoadState.LOADED_IN_VRAM,
                    process_id=2,
                ),
            },
        )
        job_tracker = JobTracker()
        job = make_job_pop_response("sd15")
        await track_popped_job_async(job_tracker, job)
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({2: target}),
            horde_model_map=model_map,
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(image_models_to_load=["sd15"]),
            max_concurrent=2,
            max_inference=2,
        )
        return scheduler, job, target

    async def test_resident_idle_dispatch_is_not_held_over_a_full_card(self) -> None:
        """The gate releases the dispatch though the 8GB card has no free room for a fresh activation."""
        scheduler, job, target = await self._resident_scheduler()
        total, baseline = 8192.0, 512.0
        over_capacity = DeviceVramState(
            total_vram_mb=total,
            baseline_mb=baseline,
            committed_vram_mb=7714.0,
            planned_unmaterialized_mb=3998.0,
            committed_is_stale=False,
            noise_buffer_mb=admission_noise_buffer_mb(total),
            # The resident weights leave no free room, so a fresh activation would not fit; only the resident
            # no-op admit releases this dispatch.
            device_free_mb=0.0,
        )
        _install_cycle(scheduler, over_capacity)
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]

        held = scheduler._dispatch_residency_reconciliation_holds(job, target)

        assert held is False
        # No eviction was needed: the dispatch materialises nothing, so no reclaim was routed.
        scheduler.unload_models_from_vram.assert_not_called()

    async def test_the_scheduler_marks_the_candidate_resident_on_the_target(self) -> None:
        """The residency helper the request builder reads reports the resident model on its target process."""
        scheduler, _job, _target = await self._resident_scheduler()
        assert scheduler._candidate_weights_resident_on_process("sd15", 2) is True
        # A model that is not the target's resident model is not credited.
        assert scheduler._candidate_weights_resident_on_process("some_other_model", 2) is False
