"""Placement tests for the auxiliary lanes (post-processing and image utilities).

These assert only what an operator can observe: the device index a lane actually lands on, and the one
disclosure line about safety co-tenancy. The children are fakes (a mocked multiprocessing context for the
post-processing lane, an injected adapter factory for the utilities lane), so no subprocess is launched
and no GPU is touched.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    PauseOwner,
    ProcessLifecycleManager,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from tests.process_management.conftest import make_test_card_runtimes, make_test_runtime_config

_SHARING_NOTICE_MARKER = "now share device"


class _FakeUtilitiesHandle:
    """A minimal child-process handle over a fake utilities lane."""

    def __init__(self) -> None:
        self.alive = True

    @property
    def pid(self) -> int | None:
        return None

    @property
    def exitcode(self) -> int | None:
        return None

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.alive = False

    def kill(self) -> None:
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        return


class _FakeUtilitiesAdapter:
    """A fake utilities lane adapter that records the device it was asked to run on."""

    def __init__(self, device_index: int) -> None:
        self.device_index = device_index
        self.started = False
        self._handle = _FakeUtilitiesHandle()

    @property
    def handle(self) -> _FakeUtilitiesHandle:
        return self._handle

    def start(self) -> None:
        self.started = True


def _make_placement_plm(
    *,
    device_indices: tuple[int, ...],
    free_mb_by_device: dict[int, float] | None,
    safety_on_gpu: bool,
    utilities_adapters: list[_FakeUtilitiesAdapter],
) -> ProcessLifecycleManager:
    """Build a lifecycle manager with both auxiliary lanes enabled over ``device_indices``.

    ``free_mb_by_device`` of None models a host that has reported no per-card free VRAM yet (the provider
    answers None for every card), which is what drives the fallback to the static placement rule.
    """
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 1
    bridge_data.enable_pipeline_disaggregation = False
    bridge_data.post_processing_lane_enabled = True
    bridge_data.enable_image_utilities = True
    bridge_data.safety_on_gpu = safety_on_gpu
    bridge_data.dry_run_skip_post_processing = False
    bridge_data.dry_run_skip_safety = False
    bridge_data.process_timeout = 120
    bridge_data.inference_step_timeout = 60
    bridge_data.inference_first_step_timeout = 120
    bridge_data.inference_stuck_step_repeat_limit = 20
    bridge_data.preload_timeout = 120
    bridge_data.download_timeout = 120
    bridge_data.post_process_timeout = 60
    bridge_data.max_batch = 1
    bridge_data.exit_on_unhandled_faults = False

    fake_ctx = Mock()
    fake_ctx.get_start_method.return_value = "spawn"
    fake_ctx.Pipe.return_value = (Mock(), Mock())
    fake_ctx.Process.return_value.pid = 12345
    fake_ctx.Process.return_value.exitcode = None

    def _free_mb(device_index: int) -> float | None:
        if free_mb_by_device is None:
            return None
        return free_mb_by_device.get(device_index)

    def _utilities_factory(
        process_id: int,
        process_message_queue: object,
        control_connection: object,
        process_launch_identifier: int,
        *,
        device_index: int,
        python_executable: str,
        child_env: dict[str, str],
        log_path: str | None = None,
    ) -> _FakeUtilitiesAdapter:
        adapter = _FakeUtilitiesAdapter(device_index)
        utilities_adapters.append(adapter)
        return adapter

    return ProcessLifecycleManager(
        ctx=fake_ctx,  # type: ignore[arg-type]
        process_map=ProcessMap({}),
        horde_model_map=Mock(),
        job_tracker=JobTracker(),
        process_message_queue=Mock(),
        card_runtimes=make_test_card_runtimes(
            target_process_count=2,
            device_indices=device_indices,
            config=bridge_data,
        ),
        disk_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        max_safety_processes=1,
        amd_gpu=False,
        directml=None,
        abort_callback=Mock(),
        state=WorkerState(),
        entry_points=ProcessEntryPoints(utilities_adapter_factory=_utilities_factory),
        device_free_mb_provider=_free_mb,
        device_total_vram_mb_provider=lambda _device_index: 24576.0,
    )


@pytest.fixture(autouse=True)
def _accelerated_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model an accelerated (not CPU-only) install, so GPU safety placement is on the table everywhere here."""
    monkeypatch.setattr(
        "horde_worker_regen.process_management.lifecycle.process_lifecycle.is_cpu_only_install",
        lambda: False,
    )


@contextmanager
def _capture_placement_notices() -> Iterator[list[str]]:
    """Capture the safety co-tenancy disclosure lines emitted inside the block."""
    notices: list[str] = []

    def _sink(message: object) -> None:
        text = message.record["message"]  # type: ignore[attr-defined]
        if _SHARING_NOTICE_MARKER in text:
            notices.append(text)

    handler_id = logger.add(_sink, level="TRACE")
    try:
        yield notices
    finally:
        logger.remove(handler_id)


def _lane_processes(plm: ProcessLifecycleManager, process_type: HordeProcessType) -> list[HordeProcessInfo]:
    """Return the processes of one type currently in the map."""
    return [process_info for process_info in plm._process_map.values() if process_info.process_type is process_type]


def _post_process_device(plm: ProcessLifecycleManager) -> int:
    """Return the device index of the one running post-processing lane."""
    lanes = _lane_processes(plm, HordeProcessType.POST_PROCESS)
    assert len(lanes) == 1
    return lanes[0].device_index


def _tear_down_paused_post_process_lane(plm: ProcessLifecycleManager) -> None:
    """Drive the lane's replacement machine to completion while a pause suppresses its restart."""
    for _ in range(4):
        plm._replace_all_post_process_process()
    assert plm._process_map.num_post_process_processes() == 0


def test_single_card_places_both_lanes_on_card_zero_and_says_nothing() -> None:
    """One card means card 0 for both lanes, across a pause and restore, with no placement chatter."""
    adapters: list[_FakeUtilitiesAdapter] = []
    plm = _make_placement_plm(
        device_indices=(0,),
        free_mb_by_device={0: 20000.0},
        safety_on_gpu=True,
        utilities_adapters=adapters,
    )

    with _capture_placement_notices() as notices:
        assert plm.start_safety_processes() is True
        assert plm.start_post_process_processes() is True
        assert plm.start_utilities_processes() is True

        assert plm.post_process_lane_card_index() == 0
        assert plm.utilities_lane_card_index() == 0
        assert _post_process_device(plm) == 0
        assert adapters[0].device_index == 0

        assert plm.pause_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
        _tear_down_paused_post_process_lane(plm)
        assert plm.restore_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
        assert _post_process_device(plm) == 0
        assert plm.post_process_lane_card_index() == 0

    assert notices == []


def test_three_cards_spread_the_two_lanes_across_the_non_safety_cards() -> None:
    """With safety on card 0, post-processing takes the emptiest other card and utilities takes the last."""
    adapters: list[_FakeUtilitiesAdapter] = []
    plm = _make_placement_plm(
        device_indices=(0, 1, 2),
        free_mb_by_device={0: 23000.0, 1: 9000.0, 2: 21000.0},
        safety_on_gpu=True,
        utilities_adapters=adapters,
    )
    assert plm.start_safety_processes() is True
    assert plm.safety_gpu_card_index() == 0

    assert plm.start_post_process_processes() is True
    assert _post_process_device(plm) == 2

    assert plm.start_utilities_processes() is True
    assert adapters[0].device_index == 1

    assert plm.post_process_lane_card_index() == 2
    assert plm.utilities_lane_card_index() == 1


def test_a_lane_paused_while_safety_is_off_gpu_returns_to_its_own_card() -> None:
    """The bundle's flap: a lane restored while safety is off-GPU comes back where it was placed.

    Re-deriving the placement at the restore reads a card map the pause itself distorted (with safety held
    off-GPU, its card looks like the emptiest one), which is how the lane landed on card 0 on top of the
    safety context and a whole-card model.
    """
    adapters: list[_FakeUtilitiesAdapter] = []
    plm = _make_placement_plm(
        device_indices=(0, 1),
        free_mb_by_device={0: 23000.0, 1: 20000.0},
        safety_on_gpu=True,
        utilities_adapters=adapters,
    )
    assert plm.start_safety_processes() is True
    assert plm.start_post_process_processes() is True
    assert _post_process_device(plm) == 1

    # The reclaim ladder takes safety off-GPU, then stops the lane.
    assert plm.pause_safety_on_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.safety_gpu_card_index() is None
    assert plm.pause_post_process_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    _tear_down_paused_post_process_lane(plm)

    with _capture_placement_notices() as notices:
        assert plm.restore_post_process_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
        assert _post_process_device(plm) == 1
        assert plm.post_process_lane_card_index() == 1

    # Safety is still off-GPU, so nothing is shared and nothing is disclosed.
    assert notices == []


def test_safety_landing_on_a_pinned_lane_card_is_disclosed_once_and_moves_nothing() -> None:
    """A lane whose card safety later takes stays put, and one line names both lanes and the card."""
    adapters: list[_FakeUtilitiesAdapter] = []
    plm = _make_placement_plm(
        device_indices=(0, 1),
        free_mb_by_device={0: 23000.0, 1: 20000.0},
        safety_on_gpu=True,
        utilities_adapters=adapters,
    )
    # The lanes are placed before safety spawns, so they take card 1 (card 0 is safety's default).
    assert plm.start_post_process_processes() is True
    assert plm.start_utilities_processes() is True
    assert _post_process_device(plm) == 1
    assert adapters[0].device_index == 1

    with _capture_placement_notices() as notices:
        # The scheduler's headroom-aware placement puts safety on card 1 instead.
        plm.set_desired_safety_card(1)
        assert plm.start_safety_processes() is True
        assert plm.safety_gpu_card_index() == 1

        # No lane moves.
        assert _post_process_device(plm) == 1
        assert plm.post_process_lane_card_index() == 1
        assert plm.utilities_lane_card_index() == 1

        # Edge-triggered: a second safety spawn onto the same card does not repeat the line.
        plm._process_map.delete_safety_processes()
        assert plm.start_safety_processes() is True

    assert len(notices) == 1
    assert "post-processing" in notices[0]
    assert "image-utilities" in notices[0]
    assert "device 1" in notices[0]


def test_a_hot_reload_that_drops_the_pinned_card_re_places_the_lane() -> None:
    """Losing the pinned card is the one thing that re-derives placement; the new card then pins."""
    adapters: list[_FakeUtilitiesAdapter] = []
    plm = _make_placement_plm(
        device_indices=(0, 1, 2),
        free_mb_by_device={0: 23000.0, 1: 20000.0, 2: 21000.0},
        safety_on_gpu=True,
        utilities_adapters=adapters,
    )
    assert plm.start_safety_processes() is True
    assert plm.start_post_process_processes() is True
    assert _post_process_device(plm) == 2

    # The operator drops card 2; card 1 is the only remaining non-safety card.
    del plm._card_runtimes[2]
    assert plm.post_process_lane_card_index() == 1

    plm._process_map.delete_post_process_processes()
    assert plm.start_post_process_processes() is True
    assert _post_process_device(plm) == 1
    assert plm.post_process_lane_card_index() == 1


def test_without_any_free_vram_reading_the_static_rule_stands() -> None:
    """No measured free reading anywhere means the historical rule: the card after the safety card."""
    adapters: list[_FakeUtilitiesAdapter] = []
    plm = _make_placement_plm(
        device_indices=(0, 1, 2),
        free_mb_by_device=None,
        safety_on_gpu=True,
        utilities_adapters=adapters,
    )
    assert plm.start_safety_processes() is True
    assert plm.post_process_lane_card_index() == 1
    assert plm.utilities_lane_card_index() == 1


def test_safety_off_gpu_from_the_start_lets_a_lane_take_card_zero_and_keep_it() -> None:
    """With safety never permitted on a card, the emptiest card wins even when that is card 0."""
    adapters: list[_FakeUtilitiesAdapter] = []
    plm = _make_placement_plm(
        device_indices=(0, 1),
        free_mb_by_device={0: 22000.0, 1: 8000.0},
        safety_on_gpu=False,
        utilities_adapters=adapters,
    )
    assert plm.start_post_process_processes() is True
    assert _post_process_device(plm) == 0

    # A restart does not re-shop the cards, even though card 1 has since become the emptier one.
    plm._process_map.delete_post_process_processes()
    assert plm.start_post_process_processes() is True
    assert _post_process_device(plm) == 0
    assert plm.post_process_lane_card_index() == 0


def test_a_crashed_lane_is_replaced_on_the_card_it_held() -> None:
    """A crash is not an eviction: the replacement lands on the same card the lane was placed on."""
    adapters: list[_FakeUtilitiesAdapter] = []
    plm = _make_placement_plm(
        device_indices=(0, 1, 2),
        free_mb_by_device={0: 23000.0, 1: 9000.0, 2: 21000.0},
        safety_on_gpu=True,
        utilities_adapters=adapters,
    )
    assert plm.start_safety_processes() is True
    assert plm.start_utilities_processes() is True
    assert adapters[0].device_index == 2

    adapters[0].handle.alive = False
    utilities = _lane_processes(plm, HordeProcessType.UTILITIES)[0]
    assert plm._reap_if_crashed(utilities) is True
    for _ in range(4):
        plm._replace_all_utilities_process()

    assert len(adapters) == 2
    assert adapters[1].device_index == 2


def test_a_disabled_post_process_lane_still_answers_its_would_be_card() -> None:
    """The scheduler's per-card holds read a would-be placement for a lane that is not running."""
    adapters: list[_FakeUtilitiesAdapter] = []
    plm = _make_placement_plm(
        device_indices=(0, 1),
        free_mb_by_device={0: 23000.0, 1: 20000.0},
        safety_on_gpu=True,
        utilities_adapters=adapters,
    )
    plm._runtime_config.bridge_data.post_processing_lane_enabled = False
    assert plm.start_safety_processes() is True

    assert plm.post_process_lane_enabled() is False
    assert plm.start_post_process_processes() is False
    assert plm.post_process_lane_card_index() == 1


def test_a_disabled_utilities_lane_does_not_claim_a_card() -> None:
    """A lane that never starts pins nothing, so it never displaces the post-processing lane."""
    adapters: list[_FakeUtilitiesAdapter] = []
    plm = _make_placement_plm(
        device_indices=(0, 1, 2),
        free_mb_by_device={0: 23000.0, 1: 9000.0, 2: 21000.0},
        safety_on_gpu=True,
        utilities_adapters=adapters,
    )
    plm._runtime_config.bridge_data.enable_image_utilities = False
    assert plm.start_safety_processes() is True

    assert plm.start_utilities_processes() is False
    assert adapters == []
    assert plm.start_post_process_processes() is True
    assert _post_process_device(plm) == 2


def test_a_ready_safety_process_does_not_disturb_a_lane_on_another_card() -> None:
    """A lane and safety on different cards stay separate and disclose nothing, however often each restarts."""
    adapters: list[_FakeUtilitiesAdapter] = []
    plm = _make_placement_plm(
        device_indices=(0, 1),
        free_mb_by_device={0: 23000.0, 1: 20000.0},
        safety_on_gpu=True,
        utilities_adapters=adapters,
    )

    with _capture_placement_notices() as notices:
        assert plm.start_safety_processes() is True
        assert plm.start_post_process_processes() is True
        assert _post_process_device(plm) == 1

        safety = _lane_processes(plm, HordeProcessType.SAFETY)[0]
        safety.last_process_state = HordeProcessState.WAITING_FOR_JOB
        plm._observe_safety_pool_readiness()

        plm._process_map.delete_post_process_processes()
        assert plm.start_post_process_processes() is True
        assert _post_process_device(plm) == 1

    assert notices == []
