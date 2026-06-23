"""Regression test: IPC primitives must bind to the passed spawn context, not the global module.

The worker and benchmark start their child processes from an explicit ``spawn`` context. A
``multiprocessing.Queue()`` created from the *global* module instead inherits the global default start
method, which is ``fork`` on Linux. A Queue carries an internal ``SemLock``; pickling a fork-context
SemLock into a spawn child raises::

    RuntimeError: A SemLock created in a fork context is being shared with a process in a spawn context.

The worker only avoided this because ``_prepare_runtime`` forces the global start method to spawn; the
benchmark never does, so it crashed in ``start_safety_processes``. The fix binds the queue (and every
other primitive) to the passed ctx. This is POSIX-only in nature (Windows is spawn-only), but the
introspection below is deterministic on every platform.
"""

from __future__ import annotations

import multiprocessing
import sys

import pytest

from horde_worker_regen.process_management.process_manager import CardConcurrency, MultiprocessingPrimitives


def test_primitives_bind_to_passed_spawn_context() -> None:
    """Every shared primitive a spawn child receives must be spawn-context, not fork-context."""
    spawn_ctx = multiprocessing.get_context("spawn")
    primitives = MultiprocessingPrimitives.create(
        spawn_ctx,
        per_card={
            0: CardConcurrency(
                target_process_count=1,
                max_concurrent_inference=1,
                inference_semaphore_size=1,
                vae_decode_semaphore_size=1,
                gpu_sampling_lease_slots=1,
            ),
        },
    )

    # A Queue's internal lock is a SemLock; _is_fork_ctx is True only when built from a fork context,
    # which is exactly what makes Process.start() under spawn raise. It must be False here.
    assert primitives.process_message_queue._rlock._is_fork_ctx is False  # noqa: SLF001
    assert primitives.disk_lock._is_fork_ctx is False  # noqa: SLF001
    assert primitives.aux_model_lock._is_fork_ctx is False  # noqa: SLF001
    assert primitives.inference_semaphores[0]._is_fork_ctx is False  # noqa: SLF001
    assert primitives.vae_decode_semaphores[0]._is_fork_ctx is False  # noqa: SLF001
    assert primitives.gpu_sampling_leases[0]._is_fork_ctx is False  # noqa: SLF001


@pytest.mark.skipif(sys.platform == "win32", reason="fork start method is POSIX-only; the bug cannot occur on Windows")
def test_fork_context_queue_is_distinguishable() -> None:
    """Prove the _is_fork_ctx introspection above is not vacuous: a fork-context queue reads as fork.

    This is the platform where the regression actually bites (Linux defaults the global module to fork),
    so guard that a fork-built queue is correctly flagged, making the spawn assertions a real check.
    """
    fork_ctx = multiprocessing.get_context("fork")
    fork_queue = fork_ctx.Queue()
    try:
        assert fork_queue._rlock._is_fork_ctx is True  # noqa: SLF001
    finally:
        fork_queue.close()
