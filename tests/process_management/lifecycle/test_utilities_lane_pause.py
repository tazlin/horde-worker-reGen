"""The image-utilities lane's off-GPU pause and restore, on the same owner contract as the other lanes.

The lane holds a CUDA context on the card and had no off-GPU actuator, so a starved head short by what it
held had no rung that named it. The pause ends the service through the ordinary replacement state machine,
the per-tick start hook stays a no-op while paused, and only the pausing owner's restore brings it back.
"""

from __future__ import annotations

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import make_mock_process_info
from tests.process_management.lifecycle.test_process_lifecycle import _make_plm


def _plm_with_lane():  # noqa: ANN202
    lane = make_mock_process_info(4, model_name=None, process_type=HordeProcessType.UTILITIES)
    plm = _make_plm(process_map=ProcessMap({4: lane}))
    plm._runtime_config.bridge_data.enable_image_utilities = True
    return plm


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
    """A foreign owner cannot lift another's hold; the holder's restore clears it and starts the lane."""
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
