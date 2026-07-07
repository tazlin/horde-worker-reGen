"""The scheduler's teardownable-idle-context predicate that feeds the starvation context-teardown escalation.

A head starved past the arbiter's escalation threshold whose remaining deficit is held by idle sibling CUDA
contexts (a bare context that weight eviction cannot reclaim) escalates to a verified context teardown. The
scheduler tells the arbiter whether such a teardown target exists via ``_has_teardownable_idle_context``,
which must exclude the head's own target slot and every busy process, and must never fire when whole-card
residency is disabled (the teardown runs through the residency machinery).
"""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import make_mock_bridge_data, make_mock_process_info
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

    def test_disabled_whole_card_residency_never_tears_down(self) -> None:
        """With whole-card residency off, no context is teardownable (the teardown needs that machinery)."""
        head = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        idle_sibling = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        scheduler = _scheduler(ProcessMap({0: head, 1: idle_sibling}), whole_card=False)
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
