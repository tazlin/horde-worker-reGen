"""Liveness contracts for the single control-loop drain of the shared child-to-parent queue.

The orchestrator runs one asyncio control loop on one thread. That thread drains the shared
child-to-parent message queue by calling :meth:`MessageDispatcher.receive_and_handle_process_messages`,
which loops ``while not queue.empty(): queue.get(block=False)``. ``get(block=False)`` is only
non-blocking as far as the readable poll: once the poll reports a frame is present, the framed read of
the body can block unbounded if the body bytes never arrive (a torn length-prefixed frame left by a
writer that died mid-put, or a stalled shared writer lock). Because the drain runs on the loop's only
thread, a block there freezes deadlock detection, the recovery supervisor, and every maintenance task,
so the worker goes unresponsive and even signal-driven shutdown cannot make progress.

The invariant these tests protect: a single drain tick must complete within a bounded time even when the
underlying queue read cannot complete. A read that cannot return must not be able to convert into an
unbounded freeze of the control loop.

Each deliberately-blocked read runs inside a daemon thread with its own event loop and is abandoned
safely, so a wedged read can never stall pytest teardown. The main thread waits on a hard deadline and
records whether the drain returned, rather than blocking on it.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from unittest.mock import Mock

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger
from horde_worker_regen.process_management.ipc.message_dispatcher import MessageDispatcher
from horde_worker_regen.process_management.ipc.messages import (
    HordeProcessState,
    HordeProcessStateChangeMessage,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
)


def _make_dispatcher(
    *,
    process_message_queue: object,
    process_map: ProcessMap | None = None,
    state: WorkerState | None = None,
    job_tracker: JobTracker | None = None,
) -> MessageDispatcher:
    """Build a MessageDispatcher whose message queue is an arbitrary caller-supplied double."""
    if process_map is None:
        process_map = ProcessMap({})
    if state is None:
        state = WorkerState()
    if job_tracker is None:
        job_tracker = JobTracker()

    return MessageDispatcher(
        process_map=process_map,
        horde_model_map=HordeModelMap(root={}),
        job_tracker=job_tracker,
        process_message_queue=process_message_queue,  # pyrefly: ignore - a queue double stands in for the real queue
        runtime_config=make_test_runtime_config(bridge_data=make_mock_bridge_data()),
        model_metadata=make_test_model_metadata(),
        action_ledger=ActionLedger(),
        reserve_ledger=CommittedReserveLedger(),
        on_unload_vram=Mock(),
        state=state,
    )


@dataclass(frozen=True)
class _DeadlineResult:
    """Outcome of running a coroutine against a hard, test-side deadline.

    ``finished`` is whether the coroutine returned before the deadline elapsed. ``error`` carries any
    exception it raised (a coroutine that raised still counts as finished, since it did not wedge).
    """

    finished: bool
    error: BaseException | None


def _run_async_with_deadline(
    make_coro: Callable[[], Coroutine[object, object, object]],
    *,
    deadline_seconds: float,
) -> _DeadlineResult:
    """Run ``make_coro()`` to completion in a daemon thread, bounded by ``deadline_seconds``.

    The coroutine is constructed and awaited inside a private event loop on a daemon thread, so a
    synchronous block inside it freezes only that abandoned thread, never the test runner. The caller
    waits on a completion signal up to the deadline and reports whether the coroutine returned.
    """
    import asyncio

    completed = threading.Event()
    error_box: list[BaseException] = []

    def _runner() -> None:
        try:
            asyncio.run(make_coro())
        except BaseException as exc:  # noqa: BLE001 - relayed to the caller via the result
            error_box.append(exc)
        finally:
            completed.set()

    thread = threading.Thread(target=_runner, daemon=True, name="drain-deadline-probe")
    thread.start()
    finished = completed.wait(timeout=deadline_seconds)
    return _DeadlineResult(finished=finished, error=error_box[0] if error_box else None)


class _NeverReadableQueue:
    """A queue whose readable poll lies: ``empty()`` denies emptiness, yet ``get()`` never returns.

    Models the shared pipe after a torn length-prefixed frame: the poll reports a frame is present, so
    the drain enters the framed read, but the body bytes never arrive and the read blocks forever.
    """

    def __init__(self) -> None:
        self._never = threading.Event()
        self.read_reached = threading.Event()

    def empty(self) -> bool:
        return False

    def get(self, block: bool = False, timeout: float | None = None) -> object:
        self.read_reached.set()
        self._never.wait()
        raise queue.Empty


class _TornAfterSurvivorQueue:
    """Delivers healthy frames first, then presents a torn frame whose read never returns.

    The first ``get()`` yields a queued survivor message (a healthy frame from a still-live writer); the
    next ``get()`` models a torn frame left by a writer killed mid-put and blocks unbounded. Exercises
    the contract that ending a busy writer must neither drop already-enqueued healthy frames nor let the
    resulting torn frame wedge the drain tick.
    """

    def __init__(self, survivors: list[object]) -> None:
        self._items = list(survivors)
        self._never = threading.Event()
        self.block_reached = threading.Event()

    def empty(self) -> bool:
        return False

    def get(self, block: bool = False, timeout: float | None = None) -> object:
        if self._items:
            return self._items.pop(0)
        self.block_reached.set()
        self._never.wait()
        raise queue.Empty


class _ListQueue:
    """A minimal in-memory queue double that drains its backing list and never blocks."""

    def __init__(self, items: list[object]) -> None:
        self._items = list(items)

    def empty(self) -> bool:
        return not self._items

    def get(self, block: bool = False, timeout: float | None = None) -> object:
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)


def _cpu_only_state_change(process_id: int, launch_identifier: int) -> HordeProcessStateChangeMessage:
    """A CPU-only torch build report: a state change whose handling latches an observable worker flag."""
    return HordeProcessStateChangeMessage(
        process_id=process_id,
        process_launch_identifier=launch_identifier,
        process_state=HordeProcessState.TORCH_BUILD_CPU_ONLY,
        info="installed torch is a CPU-only build",
    )


def _gpu_incompatible_state_change(process_id: int, launch_identifier: int) -> HordeProcessStateChangeMessage:
    """A GPU-incompatible torch report: a state change whose handling latches a second observable flag."""
    return HordeProcessStateChangeMessage(
        process_id=process_id,
        process_launch_identifier=launch_identifier,
        process_state=HordeProcessState.TORCH_GPU_INCOMPATIBLE,
        info="installed torch has no kernels for this GPU",
    )


_RED_DEADLINE_SECONDS = 1.5
"""How long the drain is allowed before a non-return is treated as a control-loop wedge. Kept short: a
healthy drain of a handful of frames returns in well under a millisecond, so a full second and a half is
ample headroom while keeping a failing run brief."""

_CONTROL_DEADLINE_SECONDS = 5.0
"""Generous bound for drains that are expected to return promptly, so a slow CI host does not flake."""


class TestDrainTickCannotBeWedgedByBlockingRead:
    """A control-loop drain tick must return within a bound even if a queue read cannot complete."""

    def test_blocking_queue_read_does_not_wedge_control_tick(self) -> None:
        """A drain whose read never returns must still complete the tick within a bounded time.

        Asserts the desired liveness contract at the dispatcher seam: reaching a read that cannot return
        must not convert into an unbounded freeze of the single control-loop thread.
        """
        fake_queue = _NeverReadableQueue()
        dispatcher = _make_dispatcher(process_message_queue=fake_queue)

        result = _run_async_with_deadline(
            dispatcher.receive_and_handle_process_messages,
            deadline_seconds=_RED_DEADLINE_SECONDS,
        )

        # The drain genuinely reached the blocking framed read; the failure below is that read wedging
        # the tick, not some unrelated early return.
        assert fake_queue.read_reached.is_set(), "drain never reached the blocking queue read"
        assert result.finished, (
            "drain tick did not return within the deadline; the blocking read wedged the control loop"
        )

    def test_busy_writer_kill_neither_drops_survivors_nor_wedges_tick(self) -> None:
        """Ending a busy writer must not drop already-enqueued healthy frames nor wedge the drain tick.

        A healthy frame queued before the hazard is delivered and applied (the CPU-only flag latches),
        proving the drain still processes surviving writers' messages; the torn frame the killed writer
        left must then not freeze the tick.
        """
        state = WorkerState()
        process_info = make_mock_process_info(0)
        process_info.process_launch_identifier = 0
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_map = ProcessMap({0: process_info})

        fake_queue = _TornAfterSurvivorQueue([_cpu_only_state_change(0, 0)])
        dispatcher = _make_dispatcher(
            process_message_queue=fake_queue,
            process_map=process_map,
            state=state,
        )

        result = _run_async_with_deadline(
            dispatcher.receive_and_handle_process_messages,
            deadline_seconds=_RED_DEADLINE_SECONDS,
        )

        # The surviving writer's healthy frame was applied (behavioral evidence the drain kept working),
        # and the drain then reached the torn frame.
        assert state.torch_build_cpu_only is True, "the survivor's healthy frame was never applied"
        assert fake_queue.block_reached.is_set(), "drain never reached the torn frame after the survivor"
        assert result.finished, "drain tick did not return within the deadline; the torn frame wedged the control loop"


class TestDrainTickControls:
    """Drains that can complete must complete promptly and apply every healthy frame."""

    def test_empty_queue_tick_returns_promptly(self) -> None:
        """An empty queue drains to completion at once."""
        dispatcher = _make_dispatcher(process_message_queue=_ListQueue([]))

        result = _run_async_with_deadline(
            dispatcher.receive_and_handle_process_messages,
            deadline_seconds=_CONTROL_DEADLINE_SECONDS,
        )

        assert result.finished, "empty-queue drain did not return promptly"
        assert result.error is None

    def test_normal_drain_of_many_messages_from_multiple_writers_completes(self) -> None:
        """A backlog of healthy frames from several writers is fully drained and each is applied."""
        state = WorkerState()
        writer_a = make_mock_process_info(0)
        writer_a.process_launch_identifier = 0
        writer_a.last_process_state = HordeProcessState.WAITING_FOR_JOB
        writer_b = make_mock_process_info(1)
        writer_b.process_launch_identifier = 0
        writer_b.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_map = ProcessMap({0: writer_a, 1: writer_b})

        backlog = _ListQueue(
            [
                _cpu_only_state_change(0, 0),
                _gpu_incompatible_state_change(1, 0),
            ],
        )
        dispatcher = _make_dispatcher(
            process_message_queue=backlog,
            process_map=process_map,
            state=state,
        )

        result = _run_async_with_deadline(
            dispatcher.receive_and_handle_process_messages,
            deadline_seconds=_CONTROL_DEADLINE_SECONDS,
        )

        assert result.finished, "healthy backlog drain did not complete"
        assert backlog.empty(), "not every queued frame was consumed"
        assert state.torch_build_cpu_only is True
        assert state.gpu_torch_incompatible is True

    def test_cooperative_idle_end_leaves_channel_healthy(self) -> None:
        """A cooperatively-ended idle child (PROCESS_ENDING, no kill) leaves the channel draining normally.

        The ending child's own terminal frame is applied, and a following writer's frame is still drained
        and applied, so an orderly end never wedges the shared channel.
        """
        state = WorkerState()
        ending_child = make_mock_process_info(0)
        ending_child.process_launch_identifier = 0
        ending_child.last_process_state = HordeProcessState.WAITING_FOR_JOB
        survivor = make_mock_process_info(1)
        survivor.process_launch_identifier = 0
        survivor.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_map = ProcessMap({0: ending_child, 1: survivor})

        backlog = _ListQueue(
            [
                HordeProcessStateChangeMessage(
                    process_id=0,
                    process_launch_identifier=0,
                    process_state=HordeProcessState.PROCESS_ENDING,
                    info="ending",
                ),
                _cpu_only_state_change(1, 0),
            ],
        )
        dispatcher = _make_dispatcher(
            process_message_queue=backlog,
            process_map=process_map,
            state=state,
        )

        result = _run_async_with_deadline(
            dispatcher.receive_and_handle_process_messages,
            deadline_seconds=_CONTROL_DEADLINE_SECONDS,
        )

        assert result.finished, "drain did not complete across a cooperative end"
        assert process_map[0].last_process_state == HordeProcessState.PROCESS_ENDING
        assert state.torch_build_cpu_only is True, "the following writer's frame was not drained after the end"

    def test_late_message_from_retired_launch_is_skipped_without_wedging(self) -> None:
        """A late frame from a retired launch is dropped, and the channel keeps draining live frames.

        Mirrors a pool rebuild with cooperative ends: the tombstone absorbs the retired launch's late
        frame, and a live writer's following frame is still applied, all within a bounded drain.
        """
        state = WorkerState()
        retired = make_mock_process_info(0)
        retired.process_launch_identifier = 7
        process_map = ProcessMap({0: retired})
        process_map.retire_process(retired, "pool rebuild")

        live = make_mock_process_info(1)
        live.process_launch_identifier = 0
        live.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_map[1] = live

        backlog = _ListQueue(
            [
                HordeProcessStateChangeMessage(
                    process_id=0,
                    process_launch_identifier=7,
                    process_state=HordeProcessState.PROCESS_ENDED,
                    info="late terminal from retired launch",
                ),
                _cpu_only_state_change(1, 0),
            ],
        )
        dispatcher = _make_dispatcher(
            process_message_queue=backlog,
            process_map=process_map,
            state=state,
        )

        result = _run_async_with_deadline(
            dispatcher.receive_and_handle_process_messages,
            deadline_seconds=_CONTROL_DEADLINE_SECONDS,
        )

        assert result.finished, "drain did not complete across a retired-launch tombstone"
        assert backlog.empty()
        assert state.torch_build_cpu_only is True, "the live writer's frame was not applied after the retired frame"
