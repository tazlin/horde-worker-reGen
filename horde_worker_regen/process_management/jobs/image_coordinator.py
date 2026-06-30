"""The image-generation workload flow, as a :class:`FlowCoordinator`.

Image generation predates the per-workload flow abstraction, so its pop -> dispatch -> submit loop is
split across collaborators (:class:`JobPopper`, :class:`JobSubmitter`, and the shared
:class:`JobTracker`) rather than living in one object the way :class:`AlchemyCoordinator` does. This
coordinator wraps those collaborators so image generation presents the same
:class:`~horde_worker_regen.process_management.scheduling.workload_flow.FlowCoordinator` surface as
every other workload: a :attr:`kind`, a live :attr:`num_in_flight`, and a single :meth:`run` the main
loop launches. That uniformity is what lets the process manager hold one ``WorkloadKind``-keyed flow
registry and lets a future workload plug in without a bespoke launch path.

Dispatch is deliberately not owned here: image-generation dispatch is interwoven with the VRAM budget
and the shared reserve ledger in the process manager's control loop, which is genuinely
image-specific scheduling, so it stays there. This coordinator owns only the pop and submit ends; its
:meth:`run` supervises those two long-lived loops.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind

if TYPE_CHECKING:
    from collections.abc import Callable

    from horde_worker_regen.process_management.jobs.job_popper import JobPopper
    from horde_worker_regen.process_management.jobs.job_submitter import JobSubmitter
    from horde_worker_regen.process_management.jobs.job_tracker import JobTracker


class ImageGenerationCoordinator:
    """Presents image generation as a :class:`FlowCoordinator` over the existing pop/submit loops."""

    def __init__(
        self,
        *,
        job_popper: JobPopper,
        job_submitter: JobSubmitter,
        job_tracker: JobTracker,
        subtask_done_callback: Callable[[asyncio.Task[None]], None] | None = None,
    ) -> None:
        """Wrap the image-generation collaborators behind the flow interface.

        Args:
            job_popper: The image job popper loop (the pop end of the flow).
            job_submitter: The image job submitter loop (the submit end of the flow).
            job_tracker: The shared image job tracker, read for the live in-flight count.
            subtask_done_callback: Attached to each supervised loop's task so a loop ending (by raising
                or returning early) initiates the same graceful shutdown it would have if launched at the
                top level. The process manager passes its main-loop supervisor here; left ``None`` the
                loops run unsupervised (unit tests driving ``run`` directly).
        """
        self._job_popper = job_popper
        self._job_submitter = job_submitter
        self._job_tracker = job_tracker
        self._subtask_done_callback = subtask_done_callback

    @property
    def kind(self) -> WorkloadKind:
        """The workload flow this coordinator runs (satisfies the ``FlowCoordinator`` protocol)."""
        return WorkloadKind.IMAGE_GENERATION

    @property
    def num_in_flight(self) -> int:
        """Image jobs popped, in inference, in safety, or awaiting submission (the flow's live work).

        Mirrors ``JobTracker.num_jobs_total`` (every queued stage from pop through pending-submit), the
        image analogue of ``AlchemyCoordinator.num_in_flight``.
        """
        return self._job_tracker.num_jobs_total

    async def run(self) -> None:
        """Supervise the pop and submit loops for the worker's life.

        Both loops are meant to run until shutdown. Each is launched as its own task carrying the
        supervisor callback, so if one ends the worker shuts down gracefully exactly as it did when the
        two loops were top-level tasks. ``return_exceptions=True`` keeps a failure in one loop from
        cancelling the other mid-flight, so the submitter can keep draining in-flight work after the
        popper has stopped (the shutdown path lets it exit once drained).
        """
        popper_task = asyncio.create_task(self._job_popper.run())
        submitter_task = asyncio.create_task(self._job_submitter.run())
        if self._subtask_done_callback is not None:
            popper_task.add_done_callback(self._subtask_done_callback)
            submitter_task.add_done_callback(self._subtask_done_callback)

        logger.debug("In ImageGenerationCoordinator.run")
        await asyncio.gather(popper_task, submitter_task, return_exceptions=True)
