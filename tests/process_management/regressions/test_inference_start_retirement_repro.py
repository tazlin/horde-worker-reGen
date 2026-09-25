"""Reproduction: a deferred inference start on a card that never drains stays pending for the whole session.

An inference start waits for card headroom (:meth:`ProcessLifecycleManager._defer_gpu_start`) and is retried
only when that headroom appears (:meth:`ProcessLifecycleManager.drain_pending_gpu_starts`). A card held by a
tenant the reclaim ladder cannot evict never reaches the threshold, so the slot never starts, while every count
of planned lanes (the card's ``target_process_count``, the worker-wide ceiling, the pending-start entry itself)
keeps promising it. Safety has a CPU fallback for the same stall; an inference lane has none.

Contracts:

1. Retirement. A deferred inference start that outlives ``PENDING_GPU_START_NO_PROGRESS_SECONDS`` with no
   headroom progress on its card is retired for the session while another inference lane exists: its pending
   entry is removed, its card's target and the worker-wide ceiling drop by one, one WARNING and one ledger
   event record it, and the lane on the other card keeps serving.
2. The last lane stays. With no other inference lane there is nothing to serve on, so the entry stays pending
   and the recovery backstop, released by ``pending_gpu_starts_backing_off`` after the same window, owns it.
3. The pair. Headroom that returns inside the window starts the slot on its own card with no retirement.
4. Restore. A retired slot is planned again on its card, and started there, once that card has held room for a
   start (measured free at or above the requirement, a healthy governor, no structural shortfall) through
   ``RETIRED_INFERENCE_SLOT_RESTORE_DWELL_SECONDS``: its target and the worker-wide ceiling rise by one, and one
   INFO line and one ledger event record it. This holds on a single card, where the lost lane is the second one
   on the only card. Room that comes and goes inside the dwell does not restore, and a card whose shortfall is
   structural never does.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    PENDING_GPU_START_NO_PROGRESS_SECONDS,
    RETIRED_INFERENCE_SLOT_RESTORE_DWELL_SECONDS,
    ProcessLifecycleManager,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from tests.process_management.conftest import make_test_card_runtimes, make_test_runtime_config

_CARD_TOTAL_MB = 10240.0
"""A 10 GB card on each device."""

_PRESSURED_FREE_MB = 600.0
"""Free VRAM on a card another tenant holds: far below any GPU start threshold, and it never moves."""

_ROOMY_FREE_MB = 9000.0
"""Free VRAM on a card with nothing else on it."""

_CYCLE_SECONDS = 10.0
"""World time one driven control cycle spans."""

_RETIREMENT_CYCLE_BOUND = math.ceil(PENDING_GPU_START_NO_PROGRESS_SECONDS / _CYCLE_SECONDS) + 2
"""Cycles within which a stalled deferral must have been resolved."""

_RESTORE_CYCLE_BOUND = math.ceil(RETIRED_INFERENCE_SLOT_RESTORE_DWELL_SECONDS / _CYCLE_SECONDS) + 2
"""Cycles within which a retired slot must return once its card holds room."""

_TRANSIENT_ROOM_CYCLES = max(1, math.floor(RETIRED_INFERENCE_SLOT_RESTORE_DWELL_SECONDS / _CYCLE_SECONDS / 3))
"""How long each spell of room lasts in the transient scenario: well inside the restore dwell."""

_UNREACHABLE_FLOOR_MB = _CARD_TOTAL_MB
"""A standing floor that leaves the card no room a start could ever use: the shortfall is structural."""

_STALLED_CARD = 0
_SERVING_CARD = 1
_STALLED_SLOT = 1
_SERVING_SLOT = 2


class _AdvanceableTime:
    """Stand-in for the ``time`` module whose wall and monotonic clocks the test advances together."""

    def __init__(self) -> None:
        self._offset = 0.0

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - forwards the real module's arbitrary members
        """Return the real ``time`` module's attribute of that name."""
        return getattr(time, name)

    def time(self) -> float:
        """Return the wall clock, shifted by however far the test has advanced it."""
        return time.time() + self._offset

    def monotonic(self) -> float:
        """Return the monotonic clock, shifted by however far the test has advanced it."""
        return time.monotonic() + self._offset

    def advance(self, seconds: float) -> None:
        """Move both clocks forward."""
        self._offset += seconds


class _Cards:
    """The parent's per-card readings, which the test moves between cycles."""

    def __init__(self) -> None:
        self.free_mb: dict[int, float] = {_STALLED_CARD: _PRESSURED_FREE_MB, _SERVING_CARD: _ROOMY_FREE_MB}
        self.governor: dict[int, GovernorState] = {
            _STALLED_CARD: GovernorState.PRESSURE,
            _SERVING_CARD: GovernorState.HEALTHY,
        }

    def give_room(self, device_index: int) -> None:
        """The tenant on ``device_index`` leaves."""
        self.free_mb[device_index] = _ROOMY_FREE_MB
        self.governor[device_index] = GovernorState.HEALTHY

    def take_room(self, device_index: int) -> None:
        """A tenant takes ``device_index`` again."""
        self.free_mb[device_index] = _PRESSURED_FREE_MB
        self.governor[device_index] = GovernorState.PRESSURE


@pytest.fixture
def fake_time(monkeypatch: pytest.MonkeyPatch) -> _AdvanceableTime:
    """Advanceable clocks for the lifecycle module, where deferral ages are taken."""
    advanceable = _AdvanceableTime()
    monkeypatch.setattr("horde_worker_regen.process_management.lifecycle.process_lifecycle.time", advanceable)
    return advanceable


@pytest.fixture
def captured_warnings() -> Iterator[list[str]]:
    """Every WARNING-and-above parent log message emitted during the test."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def captured_infos() -> Iterator[list[str]]:
    """Every INFO-and-above parent log message emitted during the test."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    yield messages
    logger.remove(sink_id)


def _two_card_lifecycle(cards: _Cards) -> ProcessLifecycleManager:
    """Build a lifecycle over two cards that each plan one inference lane, with spawns faked."""
    return _lifecycle(cards, device_indices=(_STALLED_CARD, _SERVING_CARD), target_process_count=1)


def _lifecycle(
    cards: _Cards,
    *,
    device_indices: tuple[int, ...],
    target_process_count: int,
) -> ProcessLifecycleManager:
    """Build a lifecycle over ``device_indices``, each planning ``target_process_count`` lanes, with spawns faked."""
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 1
    bridge_data.enable_pipeline_disaggregation = False
    bridge_data.safety_on_gpu = False
    bridge_data.process_timeout = 120
    bridge_data.inference_step_timeout = 60
    bridge_data.inference_first_step_timeout = 120
    bridge_data.inference_stuck_step_repeat_limit = 20
    bridge_data.preload_timeout = 120
    bridge_data.download_timeout = 120
    bridge_data.post_process_timeout = 60
    bridge_data.max_batch = 1
    bridge_data.exit_on_unhandled_faults = False
    bridge_data.vram_admission_noise_mb = None

    fake_ctx = Mock()
    fake_ctx.get_start_method.return_value = "spawn"
    fake_ctx.Pipe.return_value = (Mock(), Mock())
    fake_ctx.Process.return_value.pid = 12345
    fake_ctx.Process.return_value.exitcode = None

    return ProcessLifecycleManager(
        ctx=fake_ctx,  # type: ignore[arg-type]
        process_map=ProcessMap({}),
        horde_model_map=Mock(),
        job_tracker=JobTracker(),
        process_message_queue=Mock(),
        card_runtimes=make_test_card_runtimes(
            target_process_count=target_process_count,
            device_indices=device_indices,
            config=bridge_data,
            mask_kind="cuda",
            total_vram_mb=_CARD_TOTAL_MB,
        ),
        disk_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        max_safety_processes=1,
        amd_gpu=False,
        directml=None,
        abort_callback=Mock(),
        state=WorkerState(),
        device_free_mb_provider=lambda device_index: cards.free_mb.get(device_index),
        device_total_vram_mb_provider=lambda _device_index: _CARD_TOTAL_MB,
        device_governor_state_provider=lambda device_index: cards.governor[device_index],
        gpu_start_context_mb_provider=lambda: 0.0,
    )


def _bring_up_the_pool(lifecycle: ProcessLifecycleManager) -> None:
    """Start one slot per card: the stalled card's start defers, the other card's lane comes up and serves."""
    assert (
        lifecycle._request_inference_process_start(_STALLED_SLOT, device_index=_STALLED_CARD, reason="pool") is False
    )
    assert lifecycle.has_pending_inference_starts() is True
    assert lifecycle._request_inference_process_start(_SERVING_SLOT, device_index=_SERVING_CARD, reason="pool") is True
    lifecycle._process_map[_SERVING_SLOT].last_process_state = HordeProcessState.WAITING_FOR_JOB


def _control_cycle(lifecycle: ProcessLifecycleManager, fake_time: _AdvanceableTime) -> None:
    """Advance the clock one cycle and run the pending-start drain and retired-slot restore the control tick runs."""
    fake_time.advance(_CYCLE_SECONDS)
    lifecycle.drain_pending_gpu_starts()
    lifecycle.restore_retired_inference_slots()


def _retire_the_stalled_slot(lifecycle: ProcessLifecycleManager, fake_time: _AdvanceableTime) -> None:
    """Drive control cycles until the deferred start on the stalled card is retired."""
    for _ in range(_RETIREMENT_CYCLE_BOUND):
        if not lifecycle.has_pending_inference_starts():
            break
        _control_cycle(lifecycle, fake_time)
    assert lifecycle.has_pending_inference_starts() is False
    assert lifecycle.num_inference_slots_retired == 1


def _retirement_events(lifecycle: ProcessLifecycleManager) -> list[Any]:
    """Return the ledger's inference-start retirement records."""
    return [
        event
        for event in lifecycle.action_ledger.recent(limit=500)
        if event.event_type is LedgerEventType.INFERENCE_START_RETIRED
    ]


def _restore_events(lifecycle: ProcessLifecycleManager) -> list[Any]:
    """Return the ledger's retired-slot restore records."""
    return [
        event
        for event in lifecycle.action_ledger.recent(limit=500)
        if event.event_type is LedgerEventType.INFERENCE_SLOT_RESTORED
    ]


def _inference_lanes_on(lifecycle: ProcessLifecycleManager, device_index: int) -> list[int]:
    """Return the ids of the inference processes on ``device_index``."""
    return [
        process_info.process_id
        for process_info in lifecycle._process_map.values()
        if process_info.process_type is HordeProcessType.INFERENCE and process_info.device_index == device_index
    ]


class TestStalledInferenceStartIsRetired:
    """A deferred inference start on a card that never drains ends in a worker serving on its other lanes."""

    def test_a_card_that_never_drains_retires_its_slot_and_the_other_card_keeps_serving(
        self,
        fake_time: _AdvanceableTime,
        captured_warnings: list[str],
    ) -> None:
        """The incident shape: the slot on the pressured card is resolved within a bounded number of cycles."""
        cards = _Cards()
        lifecycle = _two_card_lifecycle(cards)
        _bring_up_the_pool(lifecycle)

        cycles_run = 0
        while lifecycle.has_pending_inference_starts():
            assert cycles_run < _RETIREMENT_CYCLE_BOUND, (
                f"the deferred inference start was still pending after {_RETIREMENT_CYCLE_BOUND} cycles on a card "
                "that never drains"
            )
            _control_cycle(lifecycle, fake_time)
            cycles_run += 1
        assert cycles_run * _CYCLE_SECONDS >= PENDING_GPU_START_NO_PROGRESS_SECONDS

        # The plan no longer counts the lane: nothing reports it as backing off, and the card plans none.
        assert lifecycle.pending_gpu_starts_backing_off() is False
        assert lifecycle._card_runtimes[_STALLED_CARD].target_process_count == 0
        assert lifecycle._card_runtimes[_SERVING_CARD].target_process_count == 1
        assert lifecycle._max_inference_processes == 1
        assert lifecycle.num_inference_slots_retired == 1
        assert _STALLED_SLOT not in lifecycle._process_map

        events = _retirement_events(lifecycle)
        assert len(events) == 1
        assert events[0].process_id == _STALLED_SLOT
        assert events[0].detail["device_index"] == _STALLED_CARD
        warnings = [message for message in captured_warnings if "Retiring the slot" in message]
        assert len(warnings) == 1
        assert f"device {_STALLED_CARD}" in warnings[0]
        assert f"{_PRESSURED_FREE_MB:.0f}MB free" in warnings[0]

        # The other card's lane is untouched and still takes work, and no scale-up re-plans the retired slot.
        serving = lifecycle._process_map[_SERVING_SLOT]
        assert serving.device_index == _SERVING_CARD
        assert serving.can_accept_job() is True
        assert lifecycle.scale_inference_processes(2) == 1
        assert lifecycle.has_pending_inference_starts() is False
        assert list(lifecycle._process_map) == [_SERVING_SLOT]

        # Later cycles on the same card, which stays pressured, do nothing more.
        for _ in range(_RETIREMENT_CYCLE_BOUND):
            _control_cycle(lifecycle, fake_time)
        assert len(_retirement_events(lifecycle)) == 1
        assert lifecycle._card_runtimes[_STALLED_CARD].target_process_count == 0

    def test_the_last_inference_lane_is_left_pending_for_the_recovery_backstop(
        self, fake_time: _AdvanceableTime
    ) -> None:
        """With no other lane there is nothing to serve on, so the start stays and the backoff releases."""
        cards = _Cards()
        lifecycle = _two_card_lifecycle(cards)
        assert (
            lifecycle._request_inference_process_start(_STALLED_SLOT, device_index=_STALLED_CARD, reason="pool")
            is False
        )

        for _ in range(_RETIREMENT_CYCLE_BOUND):
            _control_cycle(lifecycle, fake_time)

        assert lifecycle.has_pending_inference_starts() is True
        assert lifecycle.pending_gpu_starts_backing_off() is False
        assert lifecycle._card_runtimes[_STALLED_CARD].target_process_count == 1
        assert _retirement_events(lifecycle) == []


class TestHeadroomInsideTheWindowStartsTheSlot:
    """The pair: a card that drains inside the window keeps its ordinary start."""

    def test_headroom_returning_inside_the_window_starts_the_slot_on_its_card(
        self, fake_time: _AdvanceableTime
    ) -> None:
        """Room that appears before the window elapses starts the slot where it was planned."""
        cards = _Cards()
        lifecycle = _two_card_lifecycle(cards)
        _bring_up_the_pool(lifecycle)

        for _ in range(int(PENDING_GPU_START_NO_PROGRESS_SECONDS / _CYCLE_SECONDS / 2)):
            _control_cycle(lifecycle, fake_time)
        assert lifecycle.has_pending_inference_starts() is True

        cards.give_room(_STALLED_CARD)
        _control_cycle(lifecycle, fake_time)

        assert lifecycle.has_pending_inference_starts() is False
        assert lifecycle._process_map[_STALLED_SLOT].device_index == _STALLED_CARD
        assert lifecycle._process_map[_STALLED_SLOT].process_type is HordeProcessType.INFERENCE
        assert lifecycle._card_runtimes[_STALLED_CARD].target_process_count == 1
        assert lifecycle._max_inference_processes == 2
        assert _retirement_events(lifecycle) == []


class TestRetiredSlotIsRestored:
    """A retired slot comes back on its own card once that card has durably had room for a start."""

    def test_a_retired_slot_returns_after_its_card_holds_room_through_the_dwell(
        self,
        fake_time: _AdvanceableTime,
        captured_infos: list[str],
    ) -> None:
        """The tenant leaves; after the dwell the slot is planned and started on the card it was retired from."""
        cards = _Cards()
        lifecycle = _two_card_lifecycle(cards)
        _bring_up_the_pool(lifecycle)
        _retire_the_stalled_slot(lifecycle, fake_time)
        assert lifecycle._max_inference_processes == 1

        cards.give_room(_STALLED_CARD)
        cycles_run = 0
        while lifecycle.num_inference_slots_retired > 0:
            assert cycles_run < _RESTORE_CYCLE_BOUND, (
                f"the retired slot was not restored within {_RESTORE_CYCLE_BOUND} cycles of its card holding room"
            )
            _control_cycle(lifecycle, fake_time)
            cycles_run += 1
        assert cycles_run * _CYCLE_SECONDS >= RETIRED_INFERENCE_SLOT_RESTORE_DWELL_SECONDS, (
            "the slot was restored before its card had held room for the restore dwell"
        )

        assert lifecycle._card_runtimes[_STALLED_CARD].target_process_count == 1
        assert lifecycle._max_inference_processes == 2
        assert _inference_lanes_on(lifecycle, _STALLED_CARD) == [_STALLED_SLOT]
        assert _inference_lanes_on(lifecycle, _SERVING_CARD) == [_SERVING_SLOT]
        events = _restore_events(lifecycle)
        assert len(events) == 1
        assert events[0].detail["device_index"] == _STALLED_CARD
        restore_lines = [message for message in captured_infos if "Restoring a retired INFERENCE slot" in message]
        assert len(restore_lines) == 1
        assert f"device {_STALLED_CARD}" in restore_lines[0]

        # The card at its planned count stays there.
        for _ in range(_RESTORE_CYCLE_BOUND):
            _control_cycle(lifecycle, fake_time)
        assert len(_restore_events(lifecycle)) == 1
        assert lifecycle._card_runtimes[_STALLED_CARD].target_process_count == 1

    def test_the_second_lane_on_a_single_card_comes_back_without_a_restart(self, fake_time: _AdvanceableTime) -> None:
        """One card planning two lanes loses the second to a transient tenant and regains it when the tenant goes."""
        cards = _Cards()
        lifecycle = _lifecycle(cards, device_indices=(_STALLED_CARD,), target_process_count=2)
        cards.give_room(_STALLED_CARD)
        assert lifecycle._request_inference_process_start(1, device_index=_STALLED_CARD, reason="pool") is True
        lifecycle._process_map[1].last_process_state = HordeProcessState.WAITING_FOR_JOB
        cards.take_room(_STALLED_CARD)
        assert lifecycle._request_inference_process_start(2, device_index=_STALLED_CARD, reason="pool") is False

        _retire_the_stalled_slot(lifecycle, fake_time)
        assert lifecycle._card_runtimes[_STALLED_CARD].target_process_count == 1
        assert lifecycle._max_inference_processes == 1

        cards.give_room(_STALLED_CARD)
        for _ in range(_RESTORE_CYCLE_BOUND):
            _control_cycle(lifecycle, fake_time)

        assert lifecycle.num_inference_slots_retired == 0
        assert lifecycle._card_runtimes[_STALLED_CARD].target_process_count == 2
        assert lifecycle._max_inference_processes == 2
        assert sorted(_inference_lanes_on(lifecycle, _STALLED_CARD)) == [1, 2]

    def test_room_that_comes_and_goes_inside_the_dwell_does_not_restore(self, fake_time: _AdvanceableTime) -> None:
        """Each spell of room is shorter than the dwell, so the clock restarts and the slot stays retired."""
        cards = _Cards()
        lifecycle = _two_card_lifecycle(cards)
        _bring_up_the_pool(lifecycle)
        _retire_the_stalled_slot(lifecycle, fake_time)

        spells = 3 * _RESTORE_CYCLE_BOUND // _TRANSIENT_ROOM_CYCLES + 1
        for _ in range(spells):
            cards.give_room(_STALLED_CARD)
            for _ in range(_TRANSIENT_ROOM_CYCLES):
                _control_cycle(lifecycle, fake_time)
            cards.take_room(_STALLED_CARD)
            _control_cycle(lifecycle, fake_time)

        assert lifecycle.num_inference_slots_retired == 1
        assert lifecycle._card_runtimes[_STALLED_CARD].target_process_count == 0
        assert _restore_events(lifecycle) == []
        assert _inference_lanes_on(lifecycle, _STALLED_CARD) == []

    def test_a_structural_shortfall_never_restores(self, fake_time: _AdvanceableTime) -> None:
        """A standing tenant leaves the card less room than a start needs, whatever free reads for a while."""
        cards = _Cards()
        lifecycle = _two_card_lifecycle(cards)
        lifecycle.set_standing_floor_provider(
            lambda device_index: _UNREACHABLE_FLOOR_MB if device_index == _STALLED_CARD else 0.0
        )
        _bring_up_the_pool(lifecycle)
        _retire_the_stalled_slot(lifecycle, fake_time)

        cards.give_room(_STALLED_CARD)
        for _ in range(3 * _RESTORE_CYCLE_BOUND):
            _control_cycle(lifecycle, fake_time)

        assert lifecycle.num_inference_slots_retired == 1
        assert lifecycle._card_runtimes[_STALLED_CARD].target_process_count == 0
        assert _restore_events(lifecycle) == []
