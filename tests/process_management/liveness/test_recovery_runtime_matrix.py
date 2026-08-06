"""Generated runtime-state coverage for queue recovery and process replacement.

These tests compose the manager's shared tracker, process map, scheduler, lifecycle manager, message
dispatcher, recovery coordinator, and recovery supervisor.  Only operating-system process creation and IPC
transport are substituted.  The state transitions on either side of those boundaries remain the production
transitions.

The central contract is that accepted work either reaches a real execution attempt or leaves through a
bounded, explicit terminal path.  A process reference created while arranging a preload is not execution
ownership, a retired process generation cannot complete a newer attempt, and progress elsewhere in the queue
cannot prove that an unchanged blocked head recovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeImageResult,
    HordeInferenceResultMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.recovery_supervisor import RecoverySupervisor
from tests.process_management.conftest import (
    FakeClock,
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)

_MODEL = "stable_diffusion"
_TICK_SECONDS = 1.0


class _Ownership(Enum):
    """How a lane is related to the queue head when a disturbance occurs."""

    UNREFERENCED = "unreferenced"
    PRELOAD_REFERENCE = "preload-reference"
    EXECUTION = "execution"


class _Disturbance(Enum):
    """A lifecycle event that retires the lane generation."""

    CHILD_EXIT = "child-exit"
    SOFT_RESET = "soft-reset"
    RESOURCE_REPLACEMENT = "resource-replacement"


class _TailShape(Enum):
    """Work queued behind the head when its lane is replaced."""

    EMPTY = "empty"
    ONE_PENDING = "one-pending"
    BURST = "burst"
    ALTERNATING_MODELS = "alternating-models"


@dataclass(frozen=True)
class _ReplacementCell:
    """One valid ownership, disturbance, retry-boundary, and queue-shape combination."""

    ownership: _Ownership
    disturbance: _Disturbance
    tail: _TailShape
    attempts_before: int
    lanes: int
    stale_result: bool = False

    @property
    def id(self) -> str:
        """Return a stable pytest row identifier."""
        stale = "-stale" if self.stale_result else ""
        return (
            f"{self.ownership.value}-{self.disturbance.value}-{self.tail.value}-"
            f"a{self.attempts_before}-l{self.lanes}{stale}"
        )


_REPLACEMENT_AXES = ("ownership", "disturbance", "tail", "attempts_before", "lanes")


def _cell_pairs(cell: _ReplacementCell) -> frozenset[tuple[str, object, str, object]]:
    """Return every unordered axis pair covered by a cell."""
    return frozenset(
        (left, getattr(cell, left), right, getattr(cell, right))
        for left_index, left in enumerate(_REPLACEMENT_AXES)
        for right in _REPLACEMENT_AXES[left_index + 1 :]
    )


def _build_replacement_cells() -> tuple[_ReplacementCell, ...]:
    """Greedily select a deterministic pairwise array from all valid replacement states."""
    candidates = tuple(
        _ReplacementCell(ownership, disturbance, tail, attempts, lanes)
        for ownership in _Ownership
        for disturbance in _Disturbance
        for tail in _TailShape
        for attempts in (0, 1)
        for lanes in (1, 2)
        if not (ownership is not _Ownership.EXECUTION and disturbance is _Disturbance.RESOURCE_REPLACEMENT)
    )
    uncovered = set().union(*(_cell_pairs(cell) for cell in candidates))
    selected: list[_ReplacementCell] = []
    remaining = list(candidates)
    while uncovered:
        best = max(remaining, key=lambda cell: (len(_cell_pairs(cell) & uncovered), cell.id))
        selected.append(best)
        uncovered -= _cell_pairs(best)
        remaining.remove(best)
    return tuple(
        _ReplacementCell(
            cell.ownership,
            cell.disturbance,
            cell.tail,
            cell.attempts_before,
            cell.lanes,
            stale_result=cell.ownership is _Ownership.EXECUTION and cell.disturbance is _Disturbance.CHILD_EXIT,
        )
        for cell in selected
    )


_REPLACEMENT_CELLS = _build_replacement_cells()
"""A generated pairwise array spanning every value of the five replacement axes."""


@dataclass
class _ReplacementReceipt:
    """Observable effects proving that a replacement event was applied."""

    old_launch: int
    new_launch: int
    attempts_before: int
    attempts_after: int
    stage_after: JobStage
    stale_messages_before: int
    stale_messages_after: int


class _RecoveryRuntime:
    """A deterministic manager world with real orchestration state and synthetic child boundaries."""

    def __init__(
        self,
        *,
        lanes: int,
        boot_state: HordeProcessState | None = HordeProcessState.WAITING_FOR_JOB,
    ) -> None:
        """Build a manager and replace process creation with generation-stamped process records."""
        self.clock = FakeClock(10_000.0)
        self.manager = make_testable_process_manager(max_threads=lanes, queue_size=max(0, lanes - 1))
        self.tracker = self.manager._job_tracker
        self.tracker._clock = self.clock
        self.tracker.set_retry_policy(2)
        self.process_map = self.manager._process_map
        self.process_map.clear()
        self.lifecycle = self.manager._process_lifecycle
        self.scheduler = self.manager._inference_scheduler
        self.coordinator = self.manager._recovery_coordinator
        self.dispatcher = self.manager._message_dispatcher
        self.boot_state = boot_state
        self.next_launch = 100
        self.starts: list[tuple[int, int]] = []
        self.ends: list[tuple[int, int]] = []
        self.structural_wedge = False

        for process_id in range(lanes):
            lane = make_mock_process_info(
                process_id,
                model_name=None,
                state=HordeProcessState.WAITING_FOR_JOB,
            )
            lane.process_launch_identifier = self._take_launch()
            self.process_map[process_id] = lane

        self.lifecycle._broadcast_inference_end_request = Mock()  # type: ignore[method-assign]
        self.lifecycle._end_inference_process = self._end_process  # type: ignore[method-assign]
        self.lifecycle._request_inference_process_start = self._start_process  # type: ignore[method-assign]
        self.lifecycle.rebuild_safety_pool = Mock()  # type: ignore[method-assign]

        deadlock_snapshot = Mock()
        deadlock_snapshot.indicates_structural_wedge.side_effect = lambda: self.structural_wedge
        self.dispatcher.get_deadlock_snapshot = Mock(return_value=deadlock_snapshot)  # type: ignore[method-assign]

        supervisor = RecoverySupervisor(
            wedge_grace_seconds=1.0,
            reset_interval_seconds=1.0,
            max_soft_resets=1,
            pool_ready_grace_seconds=2.0,
            boot_allowance_seconds=3.0,
            boot_hard_cap_seconds=6.0,
            give_up_cooldown_seconds=2.0,
            max_give_up_cycles=2,
            clean_streak_seconds=2.0,
            clock=self.clock,
        )
        self.coordinator._clock = self.clock
        self.coordinator.recovery_supervisor = supervisor

    def _take_launch(self) -> int:
        """Return a distinct child-generation identifier."""
        launch = self.next_launch
        self.next_launch += 1
        return launch

    def _end_process(self, process_info: HordeProcessInfo, *, join_deadline: float | None = None) -> None:
        """Record the operating-system process boundary crossed by lifecycle replacement."""
        del join_deadline
        self.ends.append((process_info.process_id, process_info.process_launch_identifier))

    def _start_process(self, process_id: int, *, device_index: int, reason: str) -> bool:
        """Install the next process generation without spawning an operating-system child."""
        del reason
        if self.boot_state is None:
            return True
        lane = make_mock_process_info(
            process_id,
            model_name=None,
            state=self.boot_state,
            device_index=device_index,
        )
        lane.process_launch_identifier = self._take_launch()
        self.process_map[process_id] = lane
        self.starts.append((process_id, lane.process_launch_identifier))
        return True

    async def add_head(
        self,
        ownership: _Ownership,
        *,
        attempts_before: int = 0,
    ) -> ImageGenerateJobPopResponse:
        """Queue a head and arrange its requested ownership state."""
        head = await track_popped_job_async(
            self.tracker,
            make_job_pop_response(model=_MODEL),
            time_popped=self.clock(),
        )
        assert head.id_ is not None
        tracked = self.tracker.get_tracked_job(head.id_)
        assert tracked is not None
        tracked.inference_attempts = attempts_before

        if ownership is _Ownership.UNREFERENCED:
            return head

        lane = self.process_map[0]
        lane.loaded_horde_model_name = _MODEL
        if ownership is _Ownership.PRELOAD_REFERENCE:
            lane.last_process_state = HordeProcessState.PRELOADED_MODEL
            lane.record_preload_intent(head)
            return head

        await self.tracker.mark_inference_started(head, device_index=lane.device_index)
        lane.last_process_state = HordeProcessState.INFERENCE_STARTING
        lane.last_control_flag = HordeControlFlag.START_INFERENCE
        lane.current_inference_started_at = self.clock()
        lane.record_inference_ownership(head, attempt_ordinal=attempts_before + 1)
        return head

    async def add_tail(self, shape: _TailShape) -> tuple[ImageGenerateJobPopResponse, ...]:
        """Append the requested pending tail to the queue."""
        models = {
            _TailShape.EMPTY: (),
            _TailShape.ONE_PENDING: (_MODEL,),
            _TailShape.BURST: (_MODEL, _MODEL, _MODEL),
            _TailShape.ALTERNATING_MODELS: (_MODEL, "stable_diffusion_xl", _MODEL),
        }[shape]
        return tuple(
            [
                await track_popped_job_async(
                    self.tracker,
                    make_job_pop_response(model=model),
                    time_popped=self.clock(),
                )
                for model in models
            ],
        )

    def disturb(self, disturbance: _Disturbance) -> None:
        """Apply one lifecycle disturbance through its production entry point."""
        lane = self.process_map[0]
        if disturbance is _Disturbance.SOFT_RESET:
            self.coordinator.perform_soft_reset()
            return
        if disturbance is _Disturbance.RESOURCE_REPLACEMENT:
            self.lifecycle._replace_inference_process(
                lane,
                resource_fault_reason="sampling allocation remained unavailable",
            )
            return
        cast(Mock, lane.mp_process.is_alive).return_value = False
        replaced = self.lifecycle.replace_hung_processes()
        assert replaced is True, "the synthetic child exit must actuate lifecycle replacement"

    async def deliver_stale_success(
        self,
        *,
        process_id: int,
        launch: int,
        job: ImageGenerateJobPopResponse,
    ) -> None:
        """Deliver a successful result stamped by the retired child generation."""
        message = Mock(spec=HordeInferenceResultMessage)
        message.process_id = process_id
        message.process_launch_identifier = launch
        message.reported_os_pid = None
        message.sdk_api_job_info = job
        message.time_elapsed = 1.0
        message.info = "completed by retired generation"
        message.state = GENERATION_STATE.ok
        message.job_image_results = []
        message.faults_count = 0
        cast(Mock, self.dispatcher._process_message_queue.empty).side_effect = [False, True]
        cast(Mock, self.dispatcher._process_message_queue.get).return_value = message
        await self.dispatcher.receive_and_handle_process_messages()

    async def deliver_current_success(
        self,
        *,
        process_id: int,
        job: ImageGenerateJobPopResponse,
    ) -> None:
        """Deliver a successful result from the process generation currently owning the slot."""
        await self.deliver_stale_success(
            process_id=process_id,
            launch=self.process_map[process_id].process_launch_identifier,
            job=job,
        )

    def tick_recovery(self, seconds: float = _TICK_SECONDS) -> None:
        """Run one recovery tick and advance the shared monotonic clock."""
        self.coordinator.run_recovery_supervisor()
        self.clock.advance(seconds)


def _pairwise_values(cells: tuple[_ReplacementCell, ...], left: str, right: str) -> set[tuple[object, object]]:
    """Return the value pairs represented by two replacement-cell fields."""
    return {(getattr(cell, left), getattr(cell, right)) for cell in cells}


def test_replacement_array_covers_every_valid_pair() -> None:
    """Keep later edits from silently collapsing the generated replacement state space."""
    valid_cells = tuple(
        _ReplacementCell(ownership, disturbance, tail, attempts, lanes)
        for ownership in _Ownership
        for disturbance in _Disturbance
        for tail in _TailShape
        for attempts in (0, 1)
        for lanes in (1, 2)
        if not (ownership is not _Ownership.EXECUTION and disturbance is _Disturbance.RESOURCE_REPLACEMENT)
    )
    for left_index, left in enumerate(_REPLACEMENT_AXES):
        for right in _REPLACEMENT_AXES[left_index + 1 :]:
            required = _pairwise_values(valid_cells, left, right)
            represented = _pairwise_values(_REPLACEMENT_CELLS, left, right)
            assert represented == required, f"replacement array misses {left} x {right}: {required - represented}"


@pytest.mark.parametrize("cell", _REPLACEMENT_CELLS, ids=lambda cell: cell.id)
async def test_replacement_preserves_execution_ownership_and_queue(cell: _ReplacementCell) -> None:
    """A replacement spends an attempt only for a job that entered execution on the retired lane."""
    world = _RecoveryRuntime(lanes=cell.lanes)
    head = await world.add_head(cell.ownership, attempts_before=cell.attempts_before)
    tail = await world.add_tail(cell.tail)
    assert head.id_ is not None
    old_lane = world.process_map[0]
    old_launch = old_lane.process_launch_identifier
    stale_before = world.dispatcher.stale_messages_ignored

    world.disturb(cell.disturbance)
    if cell.stale_result:
        await world.deliver_stale_success(process_id=0, launch=old_launch, job=head)

    tracked = world.tracker.get_tracked_job(head.id_)
    assert tracked is not None
    receipt = _ReplacementReceipt(
        old_launch=old_launch,
        new_launch=world.process_map[0].process_launch_identifier,
        attempts_before=cell.attempts_before,
        attempts_after=tracked.inference_attempts,
        stage_after=tracked.stage,
        stale_messages_before=stale_before,
        stale_messages_after=world.dispatcher.stale_messages_ignored,
    )
    expected_delta = 1 if cell.ownership is _Ownership.EXECUTION else 0
    expected_stage = (
        JobStage.PENDING_SUBMIT
        if cell.ownership is _Ownership.EXECUTION and cell.attempts_before + expected_delta >= 2
        else JobStage.PENDING_INFERENCE
    )

    assert world.ends, f"{cell.id}: no old child generation was retired"
    assert world.starts, f"{cell.id}: no replacement child generation was installed"
    assert receipt.new_launch != receipt.old_launch, f"{cell.id}: replacement reused a launch identifier"
    assert receipt.attempts_after - receipt.attempts_before == expected_delta, (
        f"{cell.id}: a replacement may spend an inference attempt only after execution starts; receipt={receipt}"
    )
    assert receipt.stage_after is expected_stage, f"{cell.id}: unexpected bounded-retry outcome; receipt={receipt}"
    for queued in tail:
        assert queued.id_ is not None
        assert world.tracker.get_stage(queued.id_) is JobStage.PENDING_INFERENCE, (
            f"{cell.id}: replacing the head's lane changed unrelated queued work"
        )
    pending_ids = [job.id_ for job in world.tracker.jobs_pending_inference]
    tail_ids = [job.id_ for job in tail]
    assert [job_id for job_id in pending_ids if job_id in tail_ids] == tail_ids, (
        f"{cell.id}: replacement reordered the pending tail"
    )
    if cell.stale_result:
        assert receipt.stale_messages_after == receipt.stale_messages_before + 1, (
            f"{cell.id}: the retired generation's result was not classified as stale"
        )
        assert world.tracker.get_stage(head.id_) is expected_stage, (
            f"{cell.id}: a retired generation changed the replacement attempt's tracker state"
        )


class _ProgressSource(Enum):
    """Motion observed during a transiently clean recovery interval."""

    NONE = "none"
    BLOCKED_HEAD_STARTED = "blocked-head-started"
    FOLLOWER_STARTED = "follower-started"
    FOLLOWER_COMPLETED = "follower-completed"
    FOLLOWER_FAULTED = "follower-faulted"


async def test_follower_throughput_defers_exactly_one_recovery_action() -> None:
    """Unrelated throughput permits one observation window but cannot protect a blocked head indefinitely."""
    world = _RecoveryRuntime(lanes=2)
    await world.add_head(_Ownership.UNREFERENCED)
    await track_popped_job_async(
        world.tracker,
        make_job_pop_response(model=_MODEL),
        time_popped=world.clock(),
    )

    world.structural_wedge = True
    world.tick_recovery()
    await world.tracker.increment_jobs_completed()

    world.tick_recovery()
    assert world.coordinator.unrelated_progress_deferral_spent is True
    assert world.starts == []
    assert world.coordinator.recovery_supervisor.limp_by_level == 0

    world.tick_recovery()
    assert world.starts, "the unchanged frontier must resume escalation after the bounded observation window"
    assert world.coordinator.recovery_supervisor.limp_by_level == 1


@pytest.mark.parametrize("progress_source", _ProgressSource, ids=lambda source: source.value)
async def test_recovery_episode_closes_only_when_the_blocked_head_moves(progress_source: _ProgressSource) -> None:
    """Queue motion unrelated to an unchanged blocked head cannot reset that head's escalation."""
    world = _RecoveryRuntime(lanes=2)
    head = await world.add_head(_Ownership.UNREFERENCED)
    follower = await track_popped_job_async(
        world.tracker,
        make_job_pop_response(model=_MODEL),
        time_popped=world.clock(),
    )
    assert head.id_ is not None
    assert follower.id_ is not None

    world.structural_wedge = True
    world.tick_recovery()
    world.tick_recovery()
    assert world.coordinator.recovery_supervisor.is_in_episode is True
    assert world.starts, "the persistent wedge must reach a soft reset before progress is evaluated"

    world.structural_wedge = False
    if progress_source is _ProgressSource.BLOCKED_HEAD_STARTED:
        lane = world.process_map[0]
        await world.tracker.mark_inference_started(head, device_index=lane.device_index)
        lane.last_process_state = HordeProcessState.INFERENCE_STARTING
        lane.record_inference_ownership(head, attempt_ordinal=1)
    elif progress_source is _ProgressSource.FOLLOWER_STARTED:
        lane = world.process_map[1]
        await world.tracker.mark_inference_started(follower, device_index=lane.device_index)
        lane.last_process_state = HordeProcessState.INFERENCE_STARTING
        lane.record_inference_ownership(follower, attempt_ordinal=1)
    elif progress_source is _ProgressSource.FOLLOWER_COMPLETED:
        await world.tracker.increment_jobs_completed()
    elif progress_source is _ProgressSource.FOLLOWER_FAULTED:
        world.tracker.handle_job_fault_now(follower, retryable=False)

    for _ in range(4):
        world.tick_recovery()

    head_moved = world.tracker.get_stage(head.id_) is not JobStage.PENDING_INFERENCE
    assert head_moved is (progress_source is _ProgressSource.BLOCKED_HEAD_STARTED)
    expected_closed = progress_source is _ProgressSource.BLOCKED_HEAD_STARTED
    assert world.coordinator.recovery_supervisor.is_in_episode is (not expected_closed), (
        f"{progress_source.value}: the recovery episode was reset while the same head remained blocked"
    )


@pytest.mark.parametrize(
    ("boot_state", "give_up_not_before", "give_up_by"),
    (
        pytest.param(HordeProcessState.WAITING_FOR_JOB, 2.0, 5.0, id="ready-persistent-wedge"),
        pytest.param(HordeProcessState.PROCESS_STARTING, 6.0, 10.0, id="live-hung-boot"),
        pytest.param(None, 3.0, 7.0, id="absent-replacement-boot"),
    ),
)
async def test_persistent_wedge_escalation_is_bounded_by_boot_state(
    boot_state: HordeProcessState | None,
    give_up_not_before: float,
    give_up_by: float,
) -> None:
    """A ready replacement and a live hung boot use their distinct bounded give-up horizons."""
    world = _RecoveryRuntime(lanes=1, boot_state=boot_state)
    head = await world.add_head(_Ownership.UNREFERENCED)
    assert head.id_ is not None
    world.structural_wedge = True
    started_at = world.clock()
    faulted_at: float | None = None

    for _ in range(12):
        world.tick_recovery()
        if world.tracker.get_stage(head.id_) is JobStage.PENDING_SUBMIT:
            faulted_at = world.clock() - started_at
            break

    assert world.ends, "the persistent wedge never reached its pool rebuild"
    if boot_state is not None:
        assert world.starts, "the replacement process was not installed"
    assert faulted_at is not None, "the persistent wedge never reached a bounded terminal job path"
    assert faulted_at >= give_up_not_before
    assert faulted_at <= give_up_by


async def test_execution_crash_then_current_generation_success_reaches_safety() -> None:
    """A crash consumes one attempt, and the replacement generation can still complete the retry."""
    world = _RecoveryRuntime(lanes=2)
    head = await world.add_head(_Ownership.EXECUTION)
    followers = await world.add_tail(_TailShape.ALTERNATING_MODELS)
    assert head.id_ is not None

    world.disturb(_Disturbance.CHILD_EXIT)
    replacement = world.process_map[0]
    await world.tracker.mark_inference_started(head, device_index=replacement.device_index)
    replacement.last_process_state = HordeProcessState.INFERENCE_STARTING
    replacement.last_control_flag = HordeControlFlag.START_INFERENCE
    replacement.record_inference_ownership(head, attempt_ordinal=2)
    await world.deliver_current_success(process_id=0, job=head)

    tracked = world.tracker.get_tracked_job(head.id_)
    assert tracked is not None
    assert tracked.inference_attempts == 1
    assert tracked.stage is JobStage.PENDING_SAFETY_CHECK
    assert replacement.current_inference_started_at is None
    for follower in followers:
        assert follower.id_ is not None
        assert world.tracker.get_stage(follower.id_) is JobStage.PENDING_INFERENCE


async def test_crash_on_start_breaker_then_soft_reset_restores_a_ready_lane() -> None:
    """Repeated boot failures quarantine a slot, and the recovery reset revives it without charging queued work."""
    world = _RecoveryRuntime(lanes=1, boot_state=HordeProcessState.PROCESS_STARTING)
    head = await world.add_head(_Ownership.UNREFERENCED)
    assert head.id_ is not None

    for _ in range(3):
        lane = world.process_map[0]
        lane.last_process_state = HordeProcessState.PROCESS_STARTING
        cast(Mock, lane.mp_process.is_alive).return_value = False
        assert world.lifecycle.replace_hung_processes() is True

    assert 0 in world.lifecycle.quarantined_inference_slots
    assert 0 not in world.process_map
    tracked = world.tracker.get_tracked_job(head.id_)
    assert tracked is not None
    assert tracked.stage is JobStage.PENDING_INFERENCE
    assert tracked.inference_attempts == 0

    world.boot_state = HordeProcessState.WAITING_FOR_JOB
    world.coordinator.perform_soft_reset()

    assert world.lifecycle.quarantined_inference_slots == frozenset()
    assert world.process_map[0].can_accept_job() is True
    assert world.tracker.get_stage(head.id_) is JobStage.PENDING_INFERENCE


class _OrphanStage(Enum):
    """Worker-owned stage whose result or owner disappears."""

    INFERENCE = "inference"
    SAFETY = "safety"
    POST_PROCESSING = "post-processing"


class _LossDepth(Enum):
    """Whether ownership loss happens once or through the stage's retry bound."""

    ONE = "one-loss"
    EXHAUSTED = "exhausted"


@pytest.mark.parametrize("stage", _OrphanStage, ids=lambda stage: stage.value)
@pytest.mark.parametrize("loss_depth", _LossDepth, ids=lambda depth: depth.value)
@pytest.mark.parametrize("tail", (_TailShape.EMPTY, _TailShape.BURST), ids=lambda tail: tail.value)
async def test_orphaned_stage_recovery_is_bounded_across_queue_shapes(
    stage: _OrphanStage,
    loss_depth: _LossDepth,
    tail: _TailShape,
) -> None:
    """Lost stage ownership requeues once, then reaches a bounded terminal path without changing its tail."""
    world = _RecoveryRuntime(lanes=2)
    job = await track_popped_job_async(
        world.tracker,
        make_job_pop_response(
            model=_MODEL,
            post_processing=["RealESRGAN_x4plus"] if stage is _OrphanStage.POST_PROCESSING else None,
        ),
        time_popped=world.clock(),
    )
    assert job.id_ is not None
    await world.tracker.mark_inference_started(job, device_index=0)
    job_info = HordeJobInfo(
        sdk_api_job_info=job,
        job_image_results=[HordeImageResult(image_bytes=b"generated")],
        state=GENERATION_STATE.ok,
        censored=False,
        time_popped=world.clock(),
    )
    if stage is _OrphanStage.SAFETY:
        await world.tracker.queue_for_safety(job_info)
        await world.tracker.begin_safety_check(job_info)
    elif stage is _OrphanStage.POST_PROCESSING:
        await world.tracker.queue_for_post_processing(job_info)
        await world.tracker.begin_post_processing(job_info, process_id=40, process_launch_identifier=1)

    followers = await world.add_tail(tail)
    losses = 1
    if loss_depth is _LossDepth.EXHAUSTED:
        losses = {
            _OrphanStage.INFERENCE: 2,
            _OrphanStage.SAFETY: world.coordinator.SAFETY_REQUEUE_MAX + 1,
            _OrphanStage.POST_PROCESSING: world.coordinator.POST_PROCESS_REQUEUE_MAX + 1,
        }[stage]

    for loss_index in range(losses):
        if stage is _OrphanStage.INFERENCE:
            world.coordinator.orphan_in_progress_since[job.id_] = (
                world.clock() - world.coordinator.ORPHAN_IN_PROGRESS_GRACE_SECONDS - 1.0
            )
            world.coordinator.reconcile_orphaned_in_progress_jobs()
        elif stage is _OrphanStage.SAFETY:
            world.coordinator.orphan_safety_since[job.id_] = (
                world.clock() - world.coordinator.ORPHAN_SAFETY_GRACE_SECONDS - 1.0
            )
            await world.coordinator.reconcile_orphaned_safety_jobs()
        else:
            world.coordinator.orphan_post_process_since[job.id_] = (
                world.clock() - world.coordinator.ORPHAN_POST_PROCESS_GRACE_SECONDS - 1.0
            )
            await world.coordinator.reconcile_orphaned_post_process_jobs()

        terminal = world.tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT
        if terminal or loss_index == losses - 1:
            continue
        if stage is _OrphanStage.INFERENCE:
            await world.tracker.mark_inference_started(job, device_index=0)
        elif stage is _OrphanStage.SAFETY:
            await world.tracker.begin_safety_check(job_info)
        else:
            await world.tracker.begin_post_processing(
                job_info,
                process_id=40 + loss_index + 1,
                process_launch_identifier=loss_index + 2,
            )

    expected_terminal = loss_depth is _LossDepth.EXHAUSTED
    expected_requeued_stage = {
        _OrphanStage.INFERENCE: JobStage.PENDING_INFERENCE,
        _OrphanStage.SAFETY: JobStage.PENDING_SAFETY_CHECK,
        _OrphanStage.POST_PROCESSING: JobStage.PENDING_POST_PROCESSING,
    }[stage]
    expected_stage = JobStage.PENDING_SUBMIT if expected_terminal else expected_requeued_stage
    assert world.tracker.get_stage(job.id_) is expected_stage
    for follower in followers:
        assert follower.id_ is not None
        assert world.tracker.get_stage(follower.id_) is JobStage.PENDING_INFERENCE


async def test_repeated_execution_crashes_exhaust_retry_without_changing_followers() -> None:
    """Two genuine execution crashes consume two attempts, retire two generations, and stop retrying."""
    world = _RecoveryRuntime(lanes=2)
    head = await world.add_head(_Ownership.EXECUTION)
    followers = await world.add_tail(_TailShape.BURST)
    assert head.id_ is not None

    first_launch = world.process_map[0].process_launch_identifier
    world.disturb(_Disturbance.CHILD_EXIT)
    after_first = world.tracker.get_tracked_job(head.id_)
    assert after_first is not None
    assert after_first.stage is JobStage.PENDING_INFERENCE
    assert after_first.inference_attempts == 1

    second_lane = world.process_map[0]
    await world.tracker.mark_inference_started(head, device_index=second_lane.device_index)
    second_lane.last_process_state = HordeProcessState.INFERENCE_STARTING
    second_lane.last_control_flag = HordeControlFlag.START_INFERENCE
    second_lane.record_inference_ownership(head, attempt_ordinal=2)
    second_launch = second_lane.process_launch_identifier
    world.disturb(_Disturbance.CHILD_EXIT)

    terminal = world.tracker.get_tracked_job(head.id_)
    assert terminal is not None
    assert terminal.stage is JobStage.PENDING_SUBMIT
    assert terminal.inference_attempts == 2
    assert first_launch != second_launch != world.process_map[0].process_launch_identifier
    for follower in followers:
        assert follower.id_ is not None
        assert world.tracker.get_stage(follower.id_) is JobStage.PENDING_INFERENCE
