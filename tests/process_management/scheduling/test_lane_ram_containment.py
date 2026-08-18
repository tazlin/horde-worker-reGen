"""Idle service-lane RAM containment: unload an idle lane that ratcheted its resident set.

A disaggregation service lane (the COMPONENT text-encode lane and the VAE decode lane) keeps its encoders
resident across jobs and, alternating between the hot pool models, grows its resident set well past what its
working encoders occupy. These tests prove the scheduler asks such a lane to unload its models from RAM only
when it is idle and above the containment ceiling, spares a busy lane and an inference slot, and throttles the
request so the reload cost stays bounded. They also cover the ``unload_from_ram`` lane-targeting change.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessState,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS,
    _LANE_RAM_CONTAINMENT_RSS_BYTES,
)
from tests.process_management.conftest import make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _unload_ram_message_count(process_info: HordeProcessInfo) -> int:
    """Return how many UNLOAD_MODELS_FROM_RAM control messages were sent to a process."""
    send: Mock = process_info.pipe_connection.send  # type: ignore[assignment]
    count = 0
    for call in send.call_args_list:
        message = call.args[0]
        if getattr(message, "control_flag", None) == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            count += 1
    return count


def _lane(
    process_id: int,
    *,
    process_type: HordeProcessType,
    state: HordeProcessState,
    ram_usage_bytes: int,
) -> HordeProcessInfo:
    """Build a mock service-lane process at a given state and resident-RAM reading."""
    process = make_mock_process_info(process_id, model_name=None, state=state, process_type=process_type)
    process.ram_usage_bytes = ram_usage_bytes
    return process


class TestContainIdleLaneRam:
    """The per-tick containment targets only idle lanes above the RSS ceiling."""

    def test_idle_component_lane_above_ceiling_is_unloaded(self) -> None:
        """An idle COMPONENT lane holding more than the ceiling is asked to unload from RAM."""
        lane = _lane(
            0,
            process_type=HordeProcessType.COMPONENT,
            state=HordeProcessState.WAITING_FOR_JOB,
            ram_usage_bytes=_LANE_RAM_CONTAINMENT_RSS_BYTES + 1,
        )
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: lane}))

        scheduler._contain_idle_lane_ram()

        assert _unload_ram_message_count(lane) == 1

    def test_idle_vae_lane_above_ceiling_is_unloaded(self) -> None:
        """The VAE decode lane participates the same as the component lane."""
        lane = _lane(
            0,
            process_type=HordeProcessType.VAE_LANE,
            state=HordeProcessState.WAITING_FOR_JOB,
            ram_usage_bytes=_LANE_RAM_CONTAINMENT_RSS_BYTES + 1,
        )
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: lane}))

        scheduler._contain_idle_lane_ram()

        assert _unload_ram_message_count(lane) == 1

    def test_busy_lane_is_never_unloaded(self) -> None:
        """A lane mid-encode (not accepting a job) keeps its models even above the ceiling."""
        lane = _lane(
            0,
            process_type=HordeProcessType.COMPONENT,
            state=HordeProcessState.INFERENCE_STARTING,
            ram_usage_bytes=_LANE_RAM_CONTAINMENT_RSS_BYTES + 1,
        )
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: lane}))

        scheduler._contain_idle_lane_ram()

        assert _unload_ram_message_count(lane) == 0

    def test_lane_below_ceiling_is_left_alone(self) -> None:
        """An idle lane holding only its working encoders (under the ceiling) is not disturbed."""
        lane = _lane(
            0,
            process_type=HordeProcessType.COMPONENT,
            state=HordeProcessState.WAITING_FOR_JOB,
            ram_usage_bytes=_LANE_RAM_CONTAINMENT_RSS_BYTES - 1,
        )
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: lane}))

        scheduler._contain_idle_lane_ram()

        assert _unload_ram_message_count(lane) == 0

    def test_inference_process_is_not_a_lane_target(self) -> None:
        """A high-RSS inference slot is left to the creep-containment path, not the lane containment."""
        slot = _lane(
            0,
            process_type=HordeProcessType.INFERENCE,
            state=HordeProcessState.WAITING_FOR_JOB,
            ram_usage_bytes=_LANE_RAM_CONTAINMENT_RSS_BYTES + 1,
        )
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: slot}))

        scheduler._contain_idle_lane_ram()

        assert _unload_ram_message_count(slot) == 0

    def test_containment_is_throttled_then_fires_again_after_the_interval(self) -> None:
        """A second immediate pass sends nothing; once the per-lane interval elapses, it fires again."""
        lane = _lane(
            0,
            process_type=HordeProcessType.COMPONENT,
            state=HordeProcessState.WAITING_FOR_JOB,
            ram_usage_bytes=_LANE_RAM_CONTAINMENT_RSS_BYTES + 1,
        )
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: lane}))

        scheduler._contain_idle_lane_ram()
        assert _unload_ram_message_count(lane) == 1

        # Still inside the interval: the swap storm pays no reload.
        scheduler._contain_idle_lane_ram()
        assert _unload_ram_message_count(lane) == 1

        # Age the recorded stamp past the interval so the lane, still above the ceiling, is contained again.
        scheduler._lane_ram_containment_at[0] -= _LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS + 1.0
        scheduler._contain_idle_lane_ram()
        assert _unload_ram_message_count(lane) == 2


class TestUnloadFromRamLaneTargeting:
    """unload_from_ram now accepts the service lanes instead of warning and returning."""

    def test_idle_component_lane_receives_the_unload_and_clears_model_state(self) -> None:
        """An idle COMPONENT lane is sent UNLOAD_MODELS_FROM_RAM and has its model bookkeeping cleared."""
        lane = _lane(
            0,
            process_type=HordeProcessType.COMPONENT,
            state=HordeProcessState.WAITING_FOR_JOB,
            ram_usage_bytes=_LANE_RAM_CONTAINMENT_RSS_BYTES + 1,
        )
        lane.loaded_horde_model_name = "A"
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: lane}))

        scheduler.unload_from_ram(0)

        assert _unload_ram_message_count(lane) == 1
        assert lane.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM
        assert lane.loaded_horde_model_name is None

    def test_busy_lane_is_spared(self) -> None:
        """A busy lane target is a no-op: no unload is sent."""
        lane = _lane(
            0,
            process_type=HordeProcessType.VAE_LANE,
            state=HordeProcessState.INFERENCE_STARTING,
            ram_usage_bytes=_LANE_RAM_CONTAINMENT_RSS_BYTES + 1,
        )
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: lane}))

        scheduler.unload_from_ram(0)

        assert _unload_ram_message_count(lane) == 0


class TestAlchemyLaneRamPressureReclaim:
    """Host pressure reaches the safety/post-process residents that alchemy grows."""

    def test_idle_alchemy_lanes_receive_ram_unload(self) -> None:
        """Pressure unloads both idle alchemy service processes regardless of the generic lane ceiling."""
        safety = _lane(
            0,
            process_type=HordeProcessType.SAFETY,
            state=HordeProcessState.WAITING_FOR_JOB,
            ram_usage_bytes=6 * 1024 * 1024 * 1024,
        )
        post_process = _lane(
            1,
            process_type=HordeProcessType.POST_PROCESS,
            state=HordeProcessState.WAITING_FOR_JOB,
            ram_usage_bytes=5 * 1024 * 1024 * 1024,
        )
        scheduler = _make_inference_scheduler(process_map=ProcessMap({0: safety, 1: post_process}))

        scheduler._reclaim_idle_alchemy_lanes_under_pressure()

        assert _unload_ram_message_count(safety) == 1
        assert _unload_ram_message_count(post_process) == 1

    def test_busy_alchemy_lane_is_not_interrupted(self) -> None:
        """A form currently executing keeps its models until it returns to an idle state."""
        post_process = _lane(
            1,
            process_type=HordeProcessType.POST_PROCESS,
            state=HordeProcessState.ALCHEMY_STARTING,
            ram_usage_bytes=5 * 1024 * 1024 * 1024,
        )
        scheduler = _make_inference_scheduler(process_map=ProcessMap({1: post_process}))

        scheduler._reclaim_idle_alchemy_lanes_under_pressure()

        assert _unload_ram_message_count(post_process) == 0
