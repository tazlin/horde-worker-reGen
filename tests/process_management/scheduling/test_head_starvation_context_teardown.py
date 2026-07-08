"""The scheduler's teardownable-idle-context predicate that feeds the starvation context-teardown escalation.

A head starved past the arbiter's escalation threshold whose remaining deficit is held by idle sibling CUDA
contexts (a bare context that weight eviction cannot reclaim) escalates to a verified context teardown. The
scheduler tells the arbiter whether such a teardown target exists via ``_has_teardownable_idle_context``,
which must exclude the head's own target slot and every busy process. It is independent of
``whole_card_exclusive_residency``: that flag governs steady-state exclusive-residency preference, but the
starvation escalation is an emergency-liveness path that must be reachable regardless of it (the actuation runs
through machinery that does not itself gate on the flag).
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.vram_arbiter import ActuatorCommand, ActuatorCommandKind
from horde_worker_regen.process_management.scheduling.inference_scheduler import _PreloadActuation
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _scheduler(process_map: ProcessMap, *, whole_card: bool = True):  # noqa: ANN202
    """An inference scheduler over ``process_map`` with whole-card residency configured on or off."""
    return _make_inference_scheduler(
        process_map=process_map,
        bridge_data=make_mock_bridge_data(whole_card_exclusive_residency=whole_card),
        max_inference=4,
    )


class TestHasTeardownableIdleContext:
    """Only an idle, non-target, non-busy sibling context on the scoped card is a teardown target."""

    def test_idle_sibling_context_is_teardownable(self) -> None:
        """An idle sibling inference process is a teardown candidate for the head."""
        head = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        idle_sibling = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        scheduler = _scheduler(ProcessMap({0: head, 1: idle_sibling}))
        assert scheduler._has_teardownable_idle_context(head, device_index=None) is True

    def test_head_target_slot_is_never_torn_down(self) -> None:
        """With only the head's own slot present, there is no teardownable sibling."""
        head = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        scheduler = _scheduler(ProcessMap({0: head}))
        assert scheduler._has_teardownable_idle_context(head, device_index=None) is False

    def test_busy_sibling_context_is_never_torn_down(self) -> None:
        """A busy sibling is excluded, so a head beside only busy siblings has no teardown target."""
        head = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        busy_sibling = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.INFERENCE_STARTING)
        scheduler = _scheduler(ProcessMap({0: head, 1: busy_sibling}))
        assert scheduler._has_teardownable_idle_context(head, device_index=None) is False

    def test_flag_does_not_govern_the_emergency_teardown_seam(self) -> None:
        """An idle sibling context is teardownable whether or not steady-state whole-card residency is enabled.

        This is the production wedge cell: on a card with the flag off, a weight-dominant head starved behind
        its own idle sibling contexts must still reach the emergency context teardown. The flag governs the
        steady-state exclusive-residency preference, never this liveness path.
        """
        for whole_card in (True, False):
            head = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
            idle_sibling = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
            scheduler = _scheduler(ProcessMap({0: head, 1: idle_sibling}), whole_card=whole_card)
            assert scheduler._has_teardownable_idle_context(head, device_index=None) is True, (
                f"whole_card={whole_card}: an idle sibling context must be teardownable regardless of the flag"
            )

    def test_no_idle_sibling_is_not_teardownable_regardless_of_flag(self) -> None:
        """With only the head's slot present, there is no teardown target whether or not the flag is set."""
        for whole_card in (True, False):
            head = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
            scheduler = _scheduler(ProcessMap({0: head}), whole_card=whole_card)
            assert scheduler._has_teardownable_idle_context(head, device_index=None) is False

    def test_sibling_on_another_card_is_out_of_scope(self) -> None:
        """A device-scoped query ignores an idle sibling pinned to a different card."""
        head = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0)
        other_card = make_mock_process_info(
            1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=1
        )
        scheduler = _scheduler(ProcessMap({0: head, 1: other_card}))
        assert scheduler._has_teardownable_idle_context(head, device_index=0) is False

    def test_non_inference_sibling_is_not_a_context_teardown_target(self) -> None:
        """A post-processing or other non-inference process is not a teardownable inference context."""
        head = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        pp_sibling = make_mock_process_info(
            1,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
        )
        scheduler = _scheduler(ProcessMap({0: head, 1: pp_sibling}))
        assert scheduler._has_teardownable_idle_context(head, device_index=None) is False


class TestContextTeardownActuationExecutesWithFlagOff:
    """The arbiter's REDUCE_LIVE_CONTEXTS command runs its teardown even when whole-card residency is off.

    The predicate reaching the arbiter is only half the fix: the actuation the arbiter emits must also execute.
    The command routes through :meth:`InferenceScheduler.reduce_live_contexts`, which establishes the residency
    for the head and evicts the idle siblings' VRAM. Neither step gates on ``whole_card_exclusive_residency``, so
    the idle contexts are torn down for the starved head on a card where the flag is off.
    """

    async def test_reduce_contexts_command_drives_the_teardown(self) -> None:
        """With the flag off, the command establishes the head's residency and unloads the idle sibling VRAM."""
        head = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        idle_sibling = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: head, 1: idle_sibling}),
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(
                enable_vram_budget=True,
                whole_card_exclusive_residency=False,
            ),
            max_inference=2,
        )
        scheduler._establish_whole_card_residency = Mock()  # type: ignore[method-assign]
        scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]
        scheduler._preload_actuation = _PreloadActuation(
            job=job,
            available_process=head,
            forecast=Mock(),
            max_resident=1,
        )

        commands = (ActuatorCommand(kind=ActuatorCommandKind.REDUCE_LIVE_CONTEXTS, device_index=None),)
        scheduler._execute_preload_actuations(commands, device_index=None, for_head_of_queue=True)

        scheduler._establish_whole_card_residency.assert_called_once()
        scheduler.unload_models_from_vram.assert_called_once()
        # The head is marked exclusive so the residency established for the emergency teardown is held through
        # the head's dispatch rather than being immediately restored while the flag is off.
        assert scheduler._job_tracker.is_admitted_exclusive(job) is True
