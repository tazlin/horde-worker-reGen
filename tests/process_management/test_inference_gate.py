"""Bounded permit conservation across capacity changes and child recovery."""

from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import BoundedSemaphore

import pytest

from horde_worker_regen.process_management.ipc.inference_gate import ResizableInferenceGate


def test_resize_requires_every_permit_and_rejects_recovery_overrelease() -> None:
    """Held permits prevent resizing and blind recovery cannot raise capacity."""
    gate = ResizableInferenceGate(1, ctx=multiprocessing.get_context("spawn"))
    assert gate.acquire(block=False)
    assert not gate.acquire(timeout=0.01)
    with pytest.raises(RuntimeError, match="permits are held"):
        gate.resize(3)
    gate.release()
    gate.resize(3)
    assert [gate.acquire(block=False) for _ in range(4)] == [True, True, True, False]
    for _ in range(3):
        gate.release()
    with pytest.raises(ValueError, match="too many"):
        gate.release()
    assert gate.get_value() == 3


def _acquire_in_child(gate: ResizableInferenceGate, connection: Connection) -> None:
    while count := connection.recv():
        acquired = sum(gate.acquire(block=False) for _ in range(count))
        connection.send(acquired)
        if connection.recv():
            for _ in range(acquired):
                gate.release()
            connection.send("released")


def _hold_counter_lock(gate: ResizableInferenceGate, connection: Connection) -> None:
    with gate._locked():
        connection.send("locked")
        connection.recv()


class _InterruptedReleaseGate(ResizableInferenceGate):
    def __init__(self, *, ctx: BaseContext, connection: Connection) -> None:
        super().__init__(1, ctx=ctx)
        self.interrupt = False
        self.connection = connection

    def _notify(self) -> None:
        if self.interrupt:
            self.connection.send("counter_updated")
            self.connection.recv()
        super()._notify()


class _WakeProbe:
    def __init__(self, wake: BoundedSemaphore, connection: Connection) -> None:
        self.wake = wake
        self.connection = connection
        self.observed = False

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        """Report that the waiter saw an empty counter before entering its wake wait."""
        if not self.observed:
            self.connection.send("waiting")
            self.observed = True
        return self.wake.acquire(block, timeout)

    def release(self) -> None:
        """Preserve the native hint's bounded release."""
        self.wake.release()


def _wait_on_interrupted_release(gate: ResizableInferenceGate, connection: Connection) -> None:
    from unittest.mock import patch

    with patch.object(gate, "_wake", _WakeProbe(gate._wake, connection)):
        gate.acquire()
        connection.send("acquired")
        connection.recv()


def _release_until_interrupted(gate: _InterruptedReleaseGate) -> None:
    gate.interrupt = True
    gate.release()


@pytest.mark.slow
def test_returned_permit_survives_death_before_notification() -> None:
    """The original indefinite waiter must progress without any unrelated release or recovery wake."""
    ctx = multiprocessing.get_context("spawn")
    release_parent, release_child = ctx.Pipe()
    wait_parent, wait_child = ctx.Pipe()
    gate = _InterruptedReleaseGate(ctx=ctx, connection=release_child)
    assert gate.acquire(block=False)
    waiter = ctx.Process(target=_wait_on_interrupted_release, args=(gate, wait_child))
    releaser = ctx.Process(target=_release_until_interrupted, args=(gate,))
    waiter.start()
    try:
        assert wait_parent.poll(30) and wait_parent.recv() == "waiting"
        releaser.start()
        assert release_parent.poll(30) and release_parent.recv() == "counter_updated"
        releaser.terminate()
        releaser.join(10)
        assert wait_parent.poll(5) and wait_parent.recv() == "acquired"
        assert gate.get_value() == 0
        wait_parent.send("finish")
        waiter.join(10)
        assert waiter.exitcode == 0
        gate.release()
        assert gate.get_value() == 1
        with pytest.raises(ValueError):
            gate.release()
    finally:
        for process in (waiter, releaser):
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(10)
        for connection in (release_parent, release_child, wait_parent, wait_child):
            connection.close()


@pytest.mark.slow
def test_killed_lock_holder_does_not_strand_recovery() -> None:
    """An abrupt exit inside the counter lock must release the OS lock for the parent."""
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    gate = ResizableInferenceGate(1, ctx=ctx)
    process = ctx.Process(target=_hold_counter_lock, args=(gate, child))
    process.start()
    try:
        assert parent.poll(30) and parent.recv() == "locked"
        process.terminate()
        process.join(10)
        assert gate.acquire(timeout=1)
        gate.release()
        gate.resize(2)
        assert gate.get_value() == 2
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()


@pytest.mark.slow
def test_existing_spawned_child_observes_resize_and_crash_recovery() -> None:
    """One surviving child sees increased capacity and its death leaves recoverable permits."""
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    gate = ResizableInferenceGate(1, ctx=ctx)
    process = ctx.Process(target=_acquire_in_child, args=(gate, child))
    process.start()
    try:
        parent.send(2)
        assert parent.poll(30)
        assert parent.recv() == 1
        parent.send(True)
        assert parent.poll(30)
        assert parent.recv() == "released"
        gate.resize(3)
        parent.send(4)
        assert parent.poll(30)
        assert parent.recv() == 3
        process.terminate()
        process.join(10)
        for _ in range(3):
            gate.release()
        with pytest.raises(ValueError):
            gate.release()
        assert gate.acquire(block=False)
        gate.release()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()
