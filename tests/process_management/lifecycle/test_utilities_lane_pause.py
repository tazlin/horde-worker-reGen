"""The image-utilities lane's off-GPU pause and restore, on the same owner contract as the other lanes.

The lane holds a CUDA context on the card and had no off-GPU actuator, so a starved head short by what it
held had no rung that named it. The pause ends the service through the ordinary replacement state machine,
the per-tick start hook stays a no-op while paused, and only the pausing owner's restore lifts the pause. The
restore starts the lane only when a job needs it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner, ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import make_mock_process_info
from tests.process_management.lifecycle.test_process_lifecycle import _make_plm
from tests.process_management.lifecycle.test_utilities_lifecycle import _FakeAdapter, _make_utilities_plm

_LANE_PROCESS_ID = 4


def _plm_with_lane(
    constructed: list[_FakeAdapter] | None = None,
    *,
    demand: bool = True,
) -> ProcessLifecycleManager:
    """A lifecycle manager with a live utilities lane in the map and a fake adapter factory for restarts."""
    lane = make_mock_process_info(_LANE_PROCESS_ID, model_name=None, process_type=HordeProcessType.UTILITIES)
    return _make_utilities_plm(
        constructed=constructed if constructed is not None else [],
        process_map=ProcessMap({_LANE_PROCESS_ID: lane}),
        demand=demand,
    )


def _paused_with_lane_gone(plm: ProcessLifecycleManager) -> None:
    """Pause the lane for the ladder and finish the teardown the replacement machine performs."""
    assert plm.pause_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    del plm._process_map[_LANE_PROCESS_ID]
    plm._utilities_processes_should_be_replaced = False


@contextmanager
def _captured_messages() -> Iterator[list[str]]:
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def test_pause_arms_the_replacement_and_holds_the_start_hook() -> None:
    """A pause ends the running lane intentionally and the per-tick start stays a no-op while it holds."""
    plm = _plm_with_lane()
    assert plm.is_utilities_gpu_paused is False

    assert plm.pause_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.is_utilities_gpu_paused is True
    assert plm._utilities_processes_should_be_replaced is True
    assert plm._utilities_replacement_intentional is True
    assert plm.start_utilities_processes() is False
    assert plm.pause_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is False, "idempotent"


def test_only_the_pausing_owner_restores() -> None:
    """A foreign owner cannot lift another's hold; the holder's restore clears it and, with demand, starts the lane."""
    plm = _plm_with_lane()
    assert plm.pause_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True

    assert plm.restore_utilities_off_gpu(owner=PauseOwner.WHOLE_CARD) is False
    assert plm.is_utilities_gpu_paused is True

    started: list[bool] = []
    plm.start_utilities_processes = lambda: started.append(True) or True  # type: ignore[method-assign]
    assert plm.restore_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.is_utilities_gpu_paused is False
    assert started == [True]
    assert plm.restore_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is False


def test_pause_is_a_noop_without_a_live_lane_or_with_the_lane_disabled() -> None:
    """Nothing to stop means nothing paused, so no restore obligation can be created for it."""
    disabled = _plm_with_lane()
    disabled._runtime_config.bridge_data.enable_image_utilities = False
    assert disabled.pause_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is False

    empty = _make_plm(process_map=ProcessMap({}))
    empty._runtime_config.bridge_data.enable_image_utilities = True
    assert empty.pause_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is False
    assert empty.is_utilities_gpu_paused is False


def test_restore_without_demand_clears_the_pause_and_leaves_the_lane_down() -> None:
    """The holder's restore discharges the pause, and with no job needing the lane it does not start one."""
    constructed: list[_FakeAdapter] = []
    plm = _plm_with_lane(constructed, demand=False)
    _paused_with_lane_gone(plm)

    with _captured_messages() as messages:
        assert plm.restore_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.is_utilities_gpu_paused is False
    assert constructed == []
    assert plm._process_map.num_utilities_processes() == 0
    assert messages == ["Reclaim ladder: leaving the image-utilities lane down until a job needs it."]


def test_restore_with_demand_starts_the_lane() -> None:
    """With a job needing the lane the restore starts it, as the ladder's restore always did."""
    constructed: list[_FakeAdapter] = []
    plm = _plm_with_lane(constructed, demand=True)
    _paused_with_lane_gone(plm)

    with _captured_messages() as messages:
        assert plm.restore_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.is_utilities_gpu_paused is False
    assert len(constructed) == 1
    assert plm._process_map.num_utilities_processes() == 1
    assert "Reclaim ladder: restarting the image-utilities lane after releasing the card." in messages
