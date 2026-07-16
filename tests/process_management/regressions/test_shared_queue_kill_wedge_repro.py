"""Real-multiprocessing repro for a shared child-to-parent queue wedged by a killed writer.

Every child (inference, safety, download, and every disaggregation lane) is a writer to one shared
``multiprocessing.Queue``. The parent's single control-loop thread is the only reader, draining it
through :meth:`MessageDispatcher.receive_and_handle_process_messages`. When the parent kills a busy
child (an end, a soft reset, a shutdown sweep), that child can die mid-put and leave a torn
length-prefixed frame in the shared pipe, and can die holding the queue's writer lock.

These tests drive the production drain seam over a real shared queue whose writers put real
``HordeProcessStateChangeMessage`` objects, and split the guarantee into two invariants that
discriminate different production fix architectures:

1. Tick-boundedness: every individual drain tick returns within its bound even after a writer is killed.
   A torn length-prefixed frame violates this by blocking the framed body read on the control-loop
   thread. A bounded-read or off-thread-reader fix can satisfy this invariant without changing the
   transport.

2. Survivor-reachability: after a writer is killed, a message a surviving writer put onto the same queue
   is eventually drained (its handling latches an observable worker flag). A torn frame violates this by
   blocking the reader before the survivor's message; an orphaned writer lock violates it by blocking the
   surviving writer's feeder from ever enqueuing. Only corruption-isolation or kill-avoidance designs
   (not sharing one queue across killed writers, or quiescing a writer before killing it) satisfy this.

Keeping the two invariants as separate tests means a partial fix reports an honest partial result: a
bounded-read fix would flip invariant 1 green while invariant 2 stays red.

Reproduction dynamics: the transport-level corruption is POSIX-specific. There, ``Connection._send_bytes``
prepends a four-byte length header to every frame and the queue guards writes with a writer lock, so a
writer killed mid-frame leaves a length header whose body never follows (a torn frame that blocks the
framed body read) and can orphan the writer lock (blocking every other writer's feeder). The flooding
writer's messages are padded well above the pipe buffer so its feeder is blocked partway through a frame
at kill time, making the wedge reached on essentially every cycle. On platforms whose queue has no writer
lock and whose pipes preserve message boundaries (Windows, where the length-prefix framing does not
exist), neither hazard is reachable through this transport, so these campaigns pass there because the
survivor-reachability and tick-boundedness contracts genuinely hold: the platform-independent risk (a
non-returning read freezing the single control-loop thread, whatever its cause) is covered by the
dispatcher-seam liveness tests in ``ipc/test_dispatcher_drain_liveness.py``. The POSIX injection test at
the end is deterministic: it writes a length header with no body straight into the shared pipe, so its
wedge does not depend on catching a kill at the right instant.

Every drain tick is bounded by a daemon-thread deadline and abandoned safely on a wedge, so a reproduced
freeze can never stall pytest teardown; children are always killed and joined in teardown.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import struct
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.process import BaseProcess

import pytest

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    HordeProcessState,
    HordeProcessStateChangeMessage,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import make_mock_process_info
from tests.process_management.ipc.test_dispatcher_drain_liveness import (
    _make_dispatcher,
    _run_async_with_deadline,
)

_FLOOD_PAD_BYTES = 2 * 1024 * 1024
"""Padding applied to each flooding-writer message's ``info`` field.

Sized well above any OS pipe buffer so the queue's feeder thread cannot complete a single frame's write
before the reader drains: with the parent not yet reading, the feeder blocks partway through the frame
holding the shared writer lock. Killing the writer at that point leaves a torn frame (a length header
with most of its body missing) and orphans the writer lock, so the wedge is reached on essentially every
cycle rather than only when a kill happens to interrupt a small atomic write."""

_TICK_DEADLINE_SECONDS = 1.5
"""How long a single drain tick may take before it counts as unbounded (the control loop wedged)."""

_SURVIVOR_DEADLINE_SECONDS = 5.0
"""How long the parent is allowed to drain the surviving writer's message before it counts unreachable."""

_CAMPAIGN_CYCLES = 4
"""Fresh kill cycles per campaign. Each cycle is independent (its own queue and writers). With the frame
sized above the pipe buffer the wedge is near-deterministic per cycle, so a few cycles suffice."""


def _flood_writer(shared_queue: mp.Queue, process_id: int, pad_bytes: int) -> None:  # type: ignore[type-arg]
    """Flood the shared queue with large, valid state-change messages until killed.

    The messages are real ``HordeProcessStateChangeMessage`` objects (the dispatcher rejects anything
    else), padded so each on-wire frame is large enough to be caught mid-write by a kill.
    """
    padding = "A" * pad_bytes
    while True:
        try:
            shared_queue.put(
                HordeProcessStateChangeMessage(
                    process_id=process_id,
                    process_launch_identifier=0,
                    process_state=HordeProcessState.WAITING_FOR_JOB,
                    info=padding,
                ),
            )
        except Exception:  # noqa: BLE001 - the pipe tearing down on kill is the expected exit
            return


def _survivor_writer(shared_queue: mp.Queue, process_id: int) -> None:  # type: ignore[type-arg]
    """Emit a recognisable message at a steady pace, standing in for a still-live sibling writer.

    The message reports a CPU-only torch build; when the parent drains and handles it, the worker's
    ``torch_build_cpu_only`` flag latches, giving the test an observable signal that the survivor's
    message was genuinely drained (behavioral, not merely enqueued).
    """
    while True:
        try:
            shared_queue.put(
                HordeProcessStateChangeMessage(
                    process_id=process_id,
                    process_launch_identifier=0,
                    process_state=HordeProcessState.TORCH_BUILD_CPU_ONLY,
                    info="installed torch is a CPU-only build",
                ),
            )
        except Exception:  # noqa: BLE001 - expected on teardown
            return
        time.sleep(0.02)


def _kill_all(processes: list[BaseProcess]) -> None:
    """End writers with an immediate kill, the hazard an inference-slot end applies to a busy child."""
    for process in processes:
        with contextlib.suppress(Exception):
            process.kill()


def _terminate_all(processes: list[BaseProcess]) -> None:
    """End writers with the shutdown sweep's ``terminate()`` then a short ``join``."""
    for process in processes:
        with contextlib.suppress(Exception):
            process.terminate()
            process.join(0.2)


def _kill_and_join(process: BaseProcess) -> None:
    """Best-effort terminate-and-reap so a repro cycle never leaks a live child."""
    with contextlib.suppress(Exception):
        if process.is_alive():
            process.kill()
        process.join(timeout=2)


@dataclass(frozen=True)
class _KillCampaignResult:
    """Per-invariant tallies across a kill campaign's fresh cycles.

    ``tick_unbounded_cycles`` counts cycles where at least one drain tick failed to return within its
    bound (invariant 1 breached). ``survivor_unreached_cycles`` counts cycles where the surviving
    writer's flag never latched within the deadline (invariant 2 breached).
    """

    cycles: int
    tick_unbounded_cycles: int
    survivor_unreached_cycles: int


def _run_kill_campaign(
    *,
    num_flooders: int,
    end_flooders: Callable[[list[BaseProcess]], None],
    cycles: int,
    pad_bytes: int,
) -> _KillCampaignResult:
    """Run ``cycles`` fresh kill cycles and tally both invariants' breaches.

    Each cycle spawns ``num_flooders`` flooding writers plus one surviving writer on a fresh shared
    queue, ends the flooders via ``end_flooders`` mid-stream, then drives the real dispatcher's drain
    seam over that queue in bounded ticks until the survivor's flag latches or the deadline expires.
    """
    context = mp.get_context("spawn")
    survivor_pid = 0
    flooder_pids = [10 + index for index in range(num_flooders)]
    tick_unbounded_cycles = 0
    survivor_unreached_cycles = 0

    for _ in range(cycles):
        shared_queue: mp.Queue = context.Queue()  # type: ignore[type-arg]
        flooders: list[BaseProcess] = [
            context.Process(target=_flood_writer, args=(shared_queue, pid, pad_bytes)) for pid in flooder_pids
        ]
        survivor = context.Process(target=_survivor_writer, args=(shared_queue, survivor_pid))
        for process in flooders:
            process.start()
        survivor.start()
        try:
            time.sleep(0.15)
            end_flooders(flooders)

            state = WorkerState()
            process_map = ProcessMap({survivor_pid: make_mock_process_info(survivor_pid)})
            for pid in flooder_pids:
                process_map[pid] = make_mock_process_info(pid)
            dispatcher = _make_dispatcher(
                process_message_queue=shared_queue,
                process_map=process_map,
                state=state,
            )

            observed_unbounded = False
            survivor_reached = False
            deadline = time.monotonic() + _SURVIVOR_DEADLINE_SECONDS
            while time.monotonic() < deadline:
                result = _run_async_with_deadline(
                    dispatcher.receive_and_handle_process_messages,
                    deadline_seconds=_TICK_DEADLINE_SECONDS,
                )
                if not result.finished:
                    # A tick that never returned wedged the reader (a torn frame); further ticks cannot
                    # make progress because the abandoned reader holds the queue's read lock.
                    observed_unbounded = True
                    break
                if state.torch_build_cpu_only:
                    survivor_reached = True
                    break
                time.sleep(0.05)

            if observed_unbounded:
                tick_unbounded_cycles += 1
            if not survivor_reached:
                survivor_unreached_cycles += 1
        finally:
            for process in flooders:
                _kill_and_join(process)
            _kill_and_join(survivor)
            # A wedged reader may still hold the read lock, so closing the queue is best-effort.
            with contextlib.suppress(Exception):
                shared_queue.close()

    return _KillCampaignResult(
        cycles=cycles,
        tick_unbounded_cycles=tick_unbounded_cycles,
        survivor_unreached_cycles=survivor_unreached_cycles,
    )


@pytest.mark.slow
class TestSingleKilledBusyWriterSharesQueue:
    """A single busy writer killed mid-put must not wedge the parent's drain of a surviving writer."""

    @pytest.fixture(scope="class")
    def campaign(self) -> _KillCampaignResult:
        """Run the kill campaign once for both invariant assertions in this class."""
        return _run_kill_campaign(
            num_flooders=1,
            end_flooders=_kill_all,
            cycles=_CAMPAIGN_CYCLES,
            pad_bytes=_FLOOD_PAD_BYTES,
        )

    def test_every_drain_tick_returns_within_bound_after_kill(self, campaign: _KillCampaignResult) -> None:
        """Invariant 1: no drain tick may exceed its bound after the busy writer is killed.

        A torn length-prefixed frame from the killed writer breaks this by blocking the framed body read
        on the single control-loop thread.
        """
        assert campaign.tick_unbounded_cycles == 0, (
            f"{campaign.tick_unbounded_cycles}/{campaign.cycles} cycles had a drain tick that never "
            "returned within its bound after a busy writer was killed (a torn frame wedged the reader)"
        )

    def test_surviving_writer_message_is_eventually_drained_after_kill(self, campaign: _KillCampaignResult) -> None:
        """Invariant 2: a surviving writer's message must still be drained after the busy writer is killed.

        Breached when a torn frame blocks the reader before the survivor's message, or when the killed
        writer orphaned the shared writer lock and the surviving writer could never enqueue.
        """
        assert campaign.survivor_unreached_cycles == 0, (
            f"{campaign.survivor_unreached_cycles}/{campaign.cycles} cycles never drained the surviving "
            "writer's message after a busy writer was killed on the shared queue"
        )


@pytest.mark.slow
class TestTwoWritersKilledSameTickShareQueue:
    """Two busy writers killed together (a pool rebuild or sweep) must not wedge the shared drain."""

    @pytest.fixture(scope="class")
    def campaign(self) -> _KillCampaignResult:
        """Run the two-writer kill campaign once for both invariant assertions in this class."""
        return _run_kill_campaign(
            num_flooders=2,
            end_flooders=_kill_all,
            cycles=_CAMPAIGN_CYCLES,
            pad_bytes=_FLOOD_PAD_BYTES,
        )

    def test_every_drain_tick_returns_within_bound_after_kill(self, campaign: _KillCampaignResult) -> None:
        """Invariant 1: no drain tick may exceed its bound after two writers are killed in the same tick."""
        assert campaign.tick_unbounded_cycles == 0, (
            f"{campaign.tick_unbounded_cycles}/{campaign.cycles} cycles had a drain tick that never "
            "returned after two writers were killed in the same tick"
        )

    def test_surviving_writer_message_is_eventually_drained_after_kill(self, campaign: _KillCampaignResult) -> None:
        """Invariant 2: a surviving writer's message must still be drained after two writers are killed."""
        assert campaign.survivor_unreached_cycles == 0, (
            f"{campaign.survivor_unreached_cycles}/{campaign.cycles} cycles never drained the surviving "
            "writer's message after two writers were killed in the same tick"
        )


@pytest.mark.slow
class TestTerminateSweepBusyWriterSharesQueue:
    """The shutdown blanket ``terminate()`` sweep on a busy writer must not wedge the final drain."""

    @pytest.fixture(scope="class")
    def campaign(self) -> _KillCampaignResult:
        """Run the terminate-sweep campaign once for both invariant assertions in this class."""
        return _run_kill_campaign(
            num_flooders=1,
            end_flooders=_terminate_all,
            cycles=_CAMPAIGN_CYCLES,
            pad_bytes=_FLOOD_PAD_BYTES,
        )

    def test_every_drain_tick_returns_within_bound_after_terminate(self, campaign: _KillCampaignResult) -> None:
        """Invariant 1: no final drain tick may exceed its bound after the shutdown terminate sweep."""
        assert campaign.tick_unbounded_cycles == 0, (
            f"{campaign.tick_unbounded_cycles}/{campaign.cycles} cycles had a drain tick that never "
            "returned after the shutdown terminate sweep on a busy writer"
        )

    def test_surviving_writer_message_is_eventually_drained_after_terminate(
        self,
        campaign: _KillCampaignResult,
    ) -> None:
        """Invariant 2: a surviving writer's message must still be drained after the terminate sweep."""
        assert campaign.survivor_unreached_cycles == 0, (
            f"{campaign.survivor_unreached_cycles}/{campaign.cycles} cycles never drained the surviving "
            "writer's message after the shutdown terminate sweep on a busy writer"
        )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="raw torn-frame injection uses POSIX fileno write semantics; the shared writer handle is not "
    "a plain fd on Windows",
)
def test_torn_frame_injection_wedges_real_dispatcher_drain() -> None:
    """A torn length-prefixed frame in the shared pipe deterministically wedges the real dispatcher drain.

    Writes a length header claiming a body that never follows straight into the shared queue's writer
    pipe, then drives the production drain (:meth:`MessageDispatcher.receive_and_handle_process_messages`)
    over that queue. The poll reports the header as a readable frame, so the drain enters the framed body
    read, which cannot complete. The contract asserted is that the drain tick still returns within a
    bounded time; on a drain that reads the body on the control-loop thread it does not, which is the
    wedge this file exists to catch. Determinism here does not depend on catching a kill at the right
    instant, unlike the probabilistic kill cycles above.
    """
    context = mp.get_context("spawn")
    shared_queue: mp.Queue = context.Queue()  # type: ignore[type-arg]

    # A four-byte big-endian length header declaring a body that is never written: the framed read will
    # consume the header and then block waiting for body bytes that never arrive.
    torn_header = struct.pack("!i", 4096)
    os.write(shared_queue._writer.fileno(), torn_header)  # pyrefly: ignore - reaching into the pipe internals by design

    process_info = make_mock_process_info(0)
    process_info.process_launch_identifier = 0
    process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
    dispatcher = _make_dispatcher(
        process_message_queue=shared_queue,
        process_map=ProcessMap({0: process_info}),
    )

    result = _run_async_with_deadline(
        dispatcher.receive_and_handle_process_messages,
        deadline_seconds=2.0,
    )

    assert result.finished, (
        "dispatcher drain did not return within the deadline: a torn length-prefixed frame in the shared "
        "pipe wedged the control-loop read"
    )
