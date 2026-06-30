"""Tests for ``ImageGenerationCoordinator``: the image flow's ``FlowCoordinator`` conformance.

The coordinator wraps the existing image popper, submitter, and tracker so image generation presents the
same flow surface as alchemy. These tests pin: it satisfies the runtime-checkable protocol, reports the
right kind and live work count, and preserves the per-loop shutdown supervision the popper and submitter
had as top-level tasks (a subtask ending fires the supervisor callback; a subtask raising does not
cancel its sibling or escape ``run``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import Mock

from horde_worker_regen.process_management.jobs.image_coordinator import ImageGenerationCoordinator
from horde_worker_regen.process_management.scheduling.workload_flow import FlowCoordinator, WorkloadKind

_AsyncRun = Callable[[], Coroutine[Any, Any, None]]


def _coordinator(
    *,
    num_jobs_total: int = 0,
    popper_run: _AsyncRun | None = None,
    submitter_run: _AsyncRun | None = None,
    on_done: Callable[[asyncio.Task[None]], None] | None = None,
) -> ImageGenerationCoordinator:
    popper = Mock()
    submitter = Mock()
    tracker = Mock()
    tracker.num_jobs_total = num_jobs_total
    if popper_run is not None:
        popper.run = popper_run
    if submitter_run is not None:
        submitter.run = submitter_run
    return ImageGenerationCoordinator(
        job_popper=popper,
        job_submitter=submitter,
        job_tracker=tracker,
        subtask_done_callback=on_done,
    )


def test_satisfies_flow_coordinator_protocol() -> None:
    """The coordinator structurally satisfies the runtime-checkable ``FlowCoordinator`` protocol."""
    assert isinstance(_coordinator(), FlowCoordinator)


def test_kind_is_image_generation() -> None:
    """The coordinator identifies as the image-generation workload."""
    assert _coordinator().kind is WorkloadKind.IMAGE_GENERATION


def test_num_in_flight_tracks_job_tracker_total() -> None:
    """Live work count mirrors the tracker's queued-stage total (popped through pending-submit)."""
    assert _coordinator(num_jobs_total=7).num_in_flight == 7


async def test_run_supervises_both_loops_and_fires_callback() -> None:
    """A subtask raising fires the supervisor for both loops, does not escape run, and lets the sibling finish."""
    done_tasks: list[asyncio.Task[None]] = []

    async def popper_run() -> None:
        raise RuntimeError("popper boom")

    submitter_finished = asyncio.Event()

    async def submitter_run() -> None:
        await asyncio.sleep(0)
        submitter_finished.set()

    coordinator = _coordinator(
        popper_run=popper_run,
        submitter_run=submitter_run,
        on_done=done_tasks.append,
    )

    # Must not raise even though the popper loop did.
    await coordinator.run()

    # The supervisor callback fired for both the failed popper and the finished submitter, and the
    # sibling drained rather than being cancelled by the popper's failure.
    assert len(done_tasks) == 2
    assert submitter_finished.is_set()
