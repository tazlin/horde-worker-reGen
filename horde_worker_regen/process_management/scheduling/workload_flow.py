"""Names the distinct workloads reGen orchestrates and the capability routing each one needs.

reGen runs several workload "flows" over one shared pool of child processes and one shared resource
budget: image generation and alchemy today, with audio and video generation intended to follow. Each
flow is its own pop -> dispatch -> submit loop; what they share is the process pool (routed by
:class:`~horde_worker_regen.process_management.lifecycle.horde_process.WorkerCapability`), the shared
:class:`~horde_worker_regen.process_management.resources.resource_budget.CommittedReserveLedger`, and this
vocabulary.

This module is deliberately thin scaffolding. It does not own job state (the image pipeline's
``JobTracker`` and the ``AlchemyCoordinator`` still do); it gives the flows a common name
(:class:`WorkloadKind`), a common shape (:class:`FlowCoordinator`), and a single typed source of truth
for which process capability serves which workload, so a future audio/video flow plugs in here rather
than copying a silo.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from horde_sdk.generation_parameters.alchemy.consts import (
    is_facefixer_form,
    is_strip_background_form,
    is_upscaler_form,
)
from strenum import StrEnum

from horde_worker_regen.process_management.lifecycle.horde_process import WorkerCapability


class WorkloadKind(StrEnum):
    """A distinct kind of work reGen pops, runs, and submits as its own flow.

    Audio and video generation are the intended next entries; they are reserved here (commented) rather
    than declared so an unhandled member cannot be routed before its flow exists.
    """

    IMAGE_GENERATION = "image_generation"
    ALCHEMY = "alchemy"
    # AUDIO_GENERATION = "audio_generation"  # reserved: the next flow to add
    # VIDEO_GENERATION = "video_generation"  # reserved


POST_PROCESS_RESERVE_FLOW = "image_post_processing"
"""Committed-reserve ledger flow name for image jobs active on the dedicated post-processing lane."""


_WORKLOAD_CAPABILITIES: dict[WorkloadKind, WorkerCapability] = {
    WorkloadKind.IMAGE_GENERATION: WorkerCapability.IMAGE_GEN,
    WorkloadKind.ALCHEMY: WorkerCapability.ALCHEMY_GRAPH | WorkerCapability.ALCHEMY_CLIP,
}
"""The capability flags that, between them, serve each workload. The single source of truth pairing a
workload with the process capabilities that run it (mirrored per-process by ``DEFAULT_CAPABILITIES``)."""


def capabilities_for_workload(kind: WorkloadKind) -> WorkerCapability:
    """Return the capability flags a process must declare to serve any part of the given workload."""
    return _WORKLOAD_CAPABILITIES[kind]


def capability_for_alchemy_form(form: str) -> WorkerCapability:
    """Return the single capability a process must declare to serve the given alchemy form.

    Graph-backed forms (upscalers, facefixers, strip_background) run on inference processes; every other
    form (caption, interrogation, nsfw) runs on the CLIP stack in the safety process. This is the one
    place the form-to-capability routing fact lives.
    """
    if is_upscaler_form(form) or is_facefixer_form(form) or is_strip_background_form(form):
        return WorkerCapability.ALCHEMY_GRAPH
    return WorkerCapability.ALCHEMY_CLIP


@runtime_checkable
class FlowCoordinator(Protocol):
    """The common shape of a workload flow's main-process loop.

    Every flow pops work, dispatches it to capability-matched processes, and submits results on its own
    asyncio task. This protocol captures that contract plus a minimal observability hook; it intentionally
    does not prescribe how a flow tracks state internally. ``AlchemyCoordinator`` satisfies it directly;
    ``ImageGenerationCoordinator`` satisfies it by wrapping the image pipeline's separate popper, submitter,
    and tracker, so both flows are launched and observed uniformly through the process manager's registry.
    A flow may keep dispatch elsewhere (image generation's is interwoven with the VRAM budget in the
    control loop); the protocol covers the flow's identity, live work count, and lifecycle entry point.
    """

    @property
    def kind(self) -> WorkloadKind:
        """Which workload this coordinator runs."""
        ...

    @property
    def num_in_flight(self) -> int:
        """Units of work currently popped, dispatched, or awaiting submission for this flow."""
        ...

    async def run(self) -> None:
        """Run the flow's pop -> dispatch -> submit loop until shutdown."""
        ...
