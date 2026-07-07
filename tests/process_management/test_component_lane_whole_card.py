"""Whole-card residency parity for the component (text-encode) lane.

The component lane holds a permanent CUDA context and resident text encoders that a sibling teardown cannot
reclaim, so a whole-card model must stop it off the card exactly as it stops the VAE lane, the post-processing
lane, and safety. These tests cover the lane's pause/restore machinery, the scheduler wiring that drives it
from a residency, and the interaction that a stopped lane demotes disaggregation to the monolithic path.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
)
from tests.process_management.regressions.test_whole_card_deadlock_fixes import _FLUX_MODEL
from tests.process_management.regressions.test_whole_card_lifecycle_matrix import _forecast_for_target
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


class TestComponentLanePauseRestore:
    """The lifecycle's component-lane off-GPU pause/restore mirrors the VAE lane's."""

    def _disaggregating_lifecycle(self):  # noqa: ANN202
        process_manager = make_testable_process_manager(enable_pipeline_disaggregation=True)
        return process_manager._process_lifecycle

    def test_pause_sets_flag_and_counter_and_is_idempotent(self) -> None:
        """The first pause latches the off-GPU override and counts; a second is a no-op."""
        lifecycle = self._disaggregating_lifecycle()

        assert lifecycle.pause_component_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
        assert lifecycle.is_component_gpu_paused is True
        assert lifecycle.component_gpu_pause_count == 1

        assert lifecycle.pause_component_off_gpu(owner=PauseOwner.WHOLE_CARD) is False
        assert lifecycle.component_gpu_pause_count == 1

    def test_paused_lane_is_not_restarted_by_the_per_tick_start_hook(self) -> None:
        """While paused, the per-tick start hook must not resurrect the lane."""
        lifecycle = self._disaggregating_lifecycle()
        lifecycle.pause_component_off_gpu(owner=PauseOwner.WHOLE_CARD)

        lifecycle.start_component_processes()

        assert lifecycle._process_map.num_component_processes() == 0

    def test_restore_clears_the_pause_and_restarts(self) -> None:
        """Restoring clears the override, counts, and issues a fresh lane start."""
        lifecycle = self._disaggregating_lifecycle()
        lifecycle.pause_component_off_gpu(owner=PauseOwner.WHOLE_CARD)
        lifecycle.start_component_processes = Mock()

        assert lifecycle.restore_component_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
        assert lifecycle.is_component_gpu_paused is False
        assert lifecycle.component_gpu_restore_count == 1
        lifecycle.start_component_processes.assert_called_once_with()

    def test_restore_is_a_noop_when_not_paused(self) -> None:
        """With the lane not paused there is nothing to restore."""
        lifecycle = self._disaggregating_lifecycle()
        assert lifecycle.restore_component_off_gpu(owner=PauseOwner.WHOLE_CARD) is False


class TestComponentLaneResidencyWiring:
    """Establishing and restoring a whole-card residency drives the component lane pause/restore."""

    def _scheduler_with_component_lane(self):  # noqa: ANN202
        scheduler = _make_inference_scheduler(process_map=ProcessMap({}))
        lifecycle = scheduler._process_lifecycle
        lifecycle.component_lane_enabled = Mock(return_value=True)
        lifecycle.pause_component_off_gpu = Mock(return_value=True)
        lifecycle.restore_component_off_gpu = Mock(return_value=True)
        lifecycle.is_safety_gpu_paused = False
        lifecycle.scale_inference_processes = Mock(return_value=0)
        # Isolate the component-lane wiring from the sibling levers.
        scheduler._pause_safety_for_residency_if_idle = Mock(return_value=False)
        scheduler._pause_post_process_for_residency_if_idle = Mock(return_value=False)
        scheduler._residency_should_pause_vae_lane = Mock(return_value=False)
        scheduler._residency_should_pause_safety = Mock(return_value=False)
        scheduler._residency_should_pause_post_process = Mock(return_value=False)
        return scheduler

    def test_membership_in_the_whole_card_pause_predicate(self) -> None:
        """The residency pause set includes the component lane when it is enabled, excludes it otherwise."""
        scheduler = self._scheduler_with_component_lane()
        assert scheduler._residency_should_pause_component_lane(None) is True

        scheduler._process_lifecycle.component_lane_enabled = Mock(return_value=False)
        assert scheduler._residency_should_pause_component_lane(None) is False

    def test_establishing_a_residency_pauses_the_lane(self) -> None:
        """Claiming the card for a whole-card model stops the component lane off-GPU."""
        scheduler = self._scheduler_with_component_lane()
        forecast = _forecast_for_target(1)
        job = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216)

        scheduler._establish_whole_card_residency(job, forecast, announce=False)

        scheduler._process_lifecycle.pause_component_off_gpu.assert_called_once_with(owner=PauseOwner.WHOLE_CARD)

    def test_restoring_after_drain_restarts_the_lane(self) -> None:
        """Once the residency drains, the component lane is restarted along with the other siblings."""
        scheduler = self._scheduler_with_component_lane()
        forecast = _forecast_for_target(1)
        job = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216)
        scheduler._establish_whole_card_residency(job, forecast, announce=False)

        scheduler._restore_siblings_after_whole_card()

        scheduler._process_lifecycle.restore_component_off_gpu.assert_called_once_with(owner=PauseOwner.WHOLE_CARD)


class TestDisaggregationFallsBackWhenLaneDown:
    """A stopped component lane demotes disaggregation dispatch to the monolithic path (never faults)."""

    def _inject_lane(self, process_map: ProcessMap, process_id: int, process_type: HordeProcessType) -> None:
        process_map[process_id] = make_mock_process_info(
            process_id,
            process_type=process_type,
            state=HordeProcessState.WAITING_FOR_JOB,
        )

    def test_roles_live_requires_the_component_lane(self) -> None:
        """The disaggregation liveness predicate reads the component lane, so pausing it falls back for free."""
        process_manager = make_testable_process_manager(enable_pipeline_disaggregation=True)
        process_map = process_manager._process_map

        assert process_manager._disaggregation_roles_live() is False

        self._inject_lane(process_map, 10, HordeProcessType.COMPONENT)
        self._inject_lane(process_map, 11, HordeProcessType.VAE_LANE)
        assert process_manager._disaggregation_roles_live() is True

        # Stopping the component lane (as the whole-card pause does) drops it from the map.
        process_map.retire_process(process_map[10], "component lane paused")
        assert process_manager._disaggregation_roles_live() is False
