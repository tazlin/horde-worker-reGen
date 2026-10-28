"""Names the distinct workloads reGen orchestrates and the capability routing each one needs.

reGen runs several workload "flows" over one shared pool of child processes and one shared resource
budget: image generation, alchemy and text generation today, with audio and video generation intended to
follow. Each flow is its own pop -> dispatch -> submit loop; what they share is the process pool (routed by
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

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from horde_sdk.generation_parameters.alchemy.consts import (
    is_annotation_form,
    is_facefixer_form,
    is_strip_background_form,
    is_upscaler_form,
)

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType, WorkerCapability

# Re-exported from its leaf home so every existing importer of WorkloadKind from this module stays valid.
from horde_worker_regen.process_management.scheduling.workload_kind import WorkloadKind

if TYPE_CHECKING:
    from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

POST_PROCESS_RESERVE_FLOW = "image_post_processing"
"""Committed-reserve ledger flow name for image jobs active on the dedicated post-processing lane."""


PRELOAD_ADMISSION_FLOW = "preload_admission"
"""Reservation-ledger flow name for the admitted-but-not-yet-materialised VRAM of preloads.

Each admitted preload registers its charged candidate delta as a reservation under this flow keyed by the
loading process id, so a second admission in the same window sees the first's reservation before its VRAM
lands in the device-free reading. The entries are pruned each scheduling cycle from live loading state (see
``InferenceScheduler._in_flight_admitted_planned_units``); this flow name is that reservation's namespace."""


DISPATCH_ADMISSION_FLOW = "dispatch_admission"
"""Reservation-ledger flow name for the admitted-but-not-yet-materialised VRAM of monolithic dispatches.

A dispatch is admitted when its already-resident model is sent inference; its activation-inclusive peak (net
of the resident-weight credit) materialises over the sampling window the device-free reading does not yet
reflect. Each admitted dispatch registers that peak as a reservation under this flow keyed by the job id and
targeting the sampling process, so a second admission in the same window sees it and cannot over-admit into
the same physical room. The entries are pruned each scheduling cycle from the in-progress job set (see
``InferenceScheduler._in_flight_dispatch_units``): a finalized, faulted, or process-dead job leaves that set
and its reservation drops by omission; the job-finalize hook tightens that latency."""


GPU_START_FLOW = "gpu_process_start"
"""Reservation-ledger flow name for the CUDA context of an admitted GPU process start.

A child's context forms some time after its start is admitted, and a free reading taken in between still
counts that room as free. Each admitted start on a card (inference, safety on a GPU, and every auxiliary lane)
registers one context as a reservation under this flow keyed by its process id and targeting that process, so
a later start, preload, dispatch or clearance on the card sees it; the charge decays as the child's own reported
reservation grows. The entries are pruned each scheduling cycle from the process map (see
``InferenceScheduler._gpu_start_units``): a child that leaves ``PROCESS_STARTING`` drops its reservation by
omission."""


SAFETY_CHECK_FLOW = "safety_check"
"""Reservation-ledger flow name for the device memory a safety evaluation grows by while it runs.

An on-GPU safety process holds its weights at rest and needs more for the length of a check. Nothing else
reserves that difference, so an admission priced between checks reads it as free. A check books it under
this flow keyed by its job id and targeting the safety process, from the moment the check is found to
need the room until its verdict lands, so no preload, dispatch or clearance is admitted into it. The
entries are pruned each scheduling cycle from the jobs still waiting for or in a check (see
``InferenceScheduler._safety_check_units``)."""


_WORKLOAD_CAPABILITIES: dict[WorkloadKind, WorkerCapability] = {
    WorkloadKind.IMAGE_GENERATION: WorkerCapability.IMAGE_GEN,
    WorkloadKind.ALCHEMY: WorkerCapability.ALCHEMY_GRAPH | WorkerCapability.ALCHEMY_CLIP,
    WorkloadKind.TEXT_GENERATION: WorkerCapability(0),
}
"""The capability flags that, between them, serve each workload. The single source of truth pairing a
workload with the process capabilities that run it (mirrored per-process by ``DEFAULT_CAPABILITIES``).

Text generation declares none: its generations run in a separate program the worker reaches over HTTP
(see :mod:`horde_worker_regen.text_backends`), so no child process of this worker serves any part of it
and there is nothing to route. An empty flag says that explicitly, where an absent entry would read as a
workload whose routing nobody has decided yet."""


def capabilities_for_workload(kind: WorkloadKind) -> WorkerCapability:
    """Return the capability flags a process must declare to serve any part of the given workload.

    An empty flag means the workload runs nowhere in this worker's process pool, which is true of text
    generation; callers looking for a process to dispatch to should find none rather than any.
    """
    return _WORKLOAD_CAPABILITIES[kind]


_IMAGE_ONLY = frozenset({WorkloadKind.IMAGE_GENERATION})
_IMAGE_AND_ALCHEMY = frozenset({WorkloadKind.IMAGE_GENERATION, WorkloadKind.ALCHEMY})

_PROCESS_TYPE_WORKLOADS: dict[HordeProcessType, frozenset[WorkloadKind]] = {
    HordeProcessType.INFERENCE: _IMAGE_ONLY,
    HordeProcessType.COMPONENT: _IMAGE_ONLY,
    HordeProcessType.VAE_LANE: _IMAGE_ONLY,
    HordeProcessType.SAFETY: _IMAGE_AND_ALCHEMY,
    HordeProcessType.POST_PROCESS: _IMAGE_AND_ALCHEMY,
    HordeProcessType.UTILITIES: _IMAGE_AND_ALCHEMY,
    HordeProcessType.DOWNLOAD: _IMAGE_AND_ALCHEMY,
    HordeProcessType.TEXT_BACKEND: frozenset({WorkloadKind.TEXT_GENERATION}),
}
"""The workloads each process type exists for; a worker serving none of them never starts that type.

Wider than the capability maps because a process can be needed without executing a routed unit of work:
safety screens image results, the download process feeds the others their files, and the component and VAE
lanes are stages of a disaggregated image job. No alchemy form routes to an inference process (see
:func:`capability_for_alchemy_form`), so alchemy alone does not start one. A type's own configuration gate
(the post-processing mode, the utilities flag, pipeline disaggregation) still applies on top."""


def process_types_for_workloads(workloads: frozenset[WorkloadKind]) -> frozenset[HordeProcessType]:
    """Return the process types at least one of the given workloads needs."""
    return frozenset(
        process_type
        for process_type, needed_by in _PROCESS_TYPE_WORKLOADS.items()
        if not needed_by.isdisjoint(workloads)
    )


def capability_for_alchemy_form(form: str) -> WorkerCapability:
    """Return the single capability a process must declare to serve the given alchemy form.

    ``strip_background`` runs on the out-of-venv image-utilities lane (its ``rembg`` stack never enters the
    worker's main environment), so it routes to ``IMAGE_UTILITIES``. The other graph-backed forms
    (upscalers, facefixers) run on the post-processing lane; every other form (caption, interrogation,
    nsfw) runs on the CLIP stack in the safety process. This is the one place the form-to-capability
    routing fact lives.
    """
    if is_strip_background_form(form) or is_annotation_form(form):
        return WorkerCapability.IMAGE_UTILITIES
    if is_upscaler_form(form) or is_facefixer_form(form):
        return WorkerCapability.ALCHEMY_GRAPH
    return WorkerCapability.ALCHEMY_CLIP


def routes_to_utilities_lane(form: str) -> bool:
    """Return whether the given alchemy form runs on the image-utilities lane."""
    return capability_for_alchemy_form(form) is WorkerCapability.IMAGE_UTILITIES


def post_processing_is_strip_only(forms: Iterable[object]) -> bool:
    """Return whether a generation job's requested post-processing is background removal and nothing else.

    Such a job skips the post-processing lane and needs only the image-utilities lane after inference. The
    split reads ``is_strip_background_form``, as the post-processing offer depth does, so the two never disagree
    about which job the post-processing lane serves.
    """
    names = [str(form) for form in forms]
    return bool(names) and all(is_strip_background_form(name) for name in names)


def job_wants_pre_annotation(job: ImageGenerateJobPopResponse) -> bool:
    """Return whether a ControlNet job's control map should be derived on the image-utilities lane.

    True when the job requests a control type and its source is a real image (not already a control map)
    that is present to annotate. This covers both a generation job (whose derived map is injected as its
    premade control map) and a ``return_control_map`` job (whose derived map *is* its deliverable); the
    result handler routes the two differently. A job with no source image (nothing to annotate) falls
    through to hordelib's in-graph path.
    """
    payload = job.payload
    if payload.control_type is None:
        return False
    if payload.image_is_control:
        return False
    return job.source_image is not None


class UtilitiesLaneDemandReason(enum.StrEnum):
    """Why the image-utilities lane should be up: the first rule of :func:`utilities_lane_demand` that holds."""

    ALCHEMY_ANNOTATION_UNLEARNED = "alchemy_annotation_unlearned"
    """The alchemist offers annotation and no lane has reported a servable set since the session began or the
    kept set was dropped, so the offer cannot carry any control types until the lane runs once."""
    PENDING_STRIP = "pending_strip"
    """A job waits for or runs its background strip."""
    STRIP_ONLY_JOB = "strip_only_job"
    """A queued or sampling job will need its strip once inference ends, with no post-processing chain first."""
    CONTROLNET_ANNOTATION = "controlnet_annotation"
    """A ControlNet job needs its control map derived on the lane, or is parked for it. An extended-ControlNet job
    counts here like any other, and dispatch holds it until the lane can serve its control type."""
    ALCHEMY_FORM_WAITING = "alchemy_form_waiting"
    """An alchemy form that runs on the lane is pending dispatch or in flight."""


@dataclass(frozen=True)
class AlchemyFormDemand:
    """The alchemy flow's part of the image-utilities lane's demand, as plain form names."""

    annotation_set_unlearned: bool = False
    """The alchemist offers annotation and no servable control-type set is available for the offer."""
    waiting: tuple[str, ...] = ()
    """The forms pending dispatch or in flight, one entry per form."""


class UtilitiesLaneOutlook(enum.StrEnum):
    """Whether a lane-bound alchemy form may wait for the image-utilities lane, read by the alchemy flow."""

    WILL_SERVE = "will_serve"
    """The lane is up, coming up, or down by policy and started on demand."""
    PAUSED = "paused"
    """A reclaim or residency owner holds the lane down and restores it when its episode ends."""
    UNAVAILABLE = "unavailable"
    """The lane crashed, its start was refused, it is disabled, or the worker is shutting down."""


@dataclass(frozen=True)
class UtilitiesLaneDemandInputs:
    """What :func:`utilities_lane_demand` reads, frozen once per evaluation."""

    lane_available: bool
    """The lane is enabled, wanted by the configured workloads and provisioned."""
    shutting_down: bool
    alchemy: AlchemyFormDemand
    jobs_pending_strip: int
    """Jobs in ``PENDING_STRIP``."""
    strip_only_jobs_before_strip: int
    """Jobs pending or in inference whose post-processing is background removal alone."""
    pre_annotation_candidates: int
    """ControlNet jobs pending inference that need pre-annotation and have not been offered to the lane."""
    jobs_pending_annotation: int
    """Jobs in ``PENDING_ANNOTATION``."""


def utilities_lane_demand(inputs: UtilitiesLaneDemandInputs) -> UtilitiesLaneDemandReason | None:
    """Return why the image-utilities lane should be up, or None when no job needs it.

    This is the only place that checks whether the lane is wanted. Every lane start reads it, and so does the
    reclaim ladder's restore, so a lane stopped to free its
    context stays down until work arrives that only the lane can serve.

    The rules, first match wins:

    1. A disabled, unwanted or unprovisioned lane, or a worker shutting down, is never wanted.
    2. An alchemist offering annotation with no servable set reported yet. The alchemy offer stands while the lane is
       down by policy (``strip_background`` always, ``annotation`` with the set kept from the lane's last run),
       so the lane is not kept warm for an offer. The annotation offer must list control types, though, and only
       a lane run reports them. Strips and ControlNet jobs arrive with traffic, and no setting guarantees that one
       will start the lane, so an annotation alchemist could go a whole session without one. This rule starts
       the lane once to read the set and lapses at the lane's first report.
    3. A job in ``PENDING_STRIP``.
    4. A job pending or in inference whose post-processing is background removal alone. Its lead (queue wait
       plus generation) covers a lane start. A job with an upscaler or face fixer counts from rule 3 only, so
       a lane is not started for its own post-processing chain and stopped again.
    5. A ControlNet pre-annotation candidate, or a job in ``PENDING_ANNOTATION``. Extended-ControlNet jobs
       count here, so the lane starts at their pop and dispatch holds them until it can serve their type.
    6. A lane-bound alchemy form pending or in flight. The alchemy offer stands while the lane is down by
       policy, so a popped form starts the lane through this rule and waits for it at dispatch.
    """
    if not inputs.lane_available or inputs.shutting_down:
        return None
    if inputs.alchemy.annotation_set_unlearned:
        return UtilitiesLaneDemandReason.ALCHEMY_ANNOTATION_UNLEARNED
    if inputs.jobs_pending_strip > 0:
        return UtilitiesLaneDemandReason.PENDING_STRIP
    if inputs.strip_only_jobs_before_strip > 0:
        return UtilitiesLaneDemandReason.STRIP_ONLY_JOB
    if inputs.pre_annotation_candidates > 0 or inputs.jobs_pending_annotation > 0:
        return UtilitiesLaneDemandReason.CONTROLNET_ANNOTATION
    if any(routes_to_utilities_lane(form) for form in inputs.alchemy.waiting):
        return UtilitiesLaneDemandReason.ALCHEMY_FORM_WAITING
    return None


@runtime_checkable
class FlowCoordinator(Protocol):
    """The common shape of a workload flow's main-process loop.

    Every flow pops work, dispatches it to capability-matched processes, and submits results on its own
    asyncio task. This protocol captures that contract plus a minimal observability hook; it intentionally
    does not prescribe how a flow tracks state internally. ``AlchemyCoordinator`` satisfies it directly;
    ``ImageGenerationCoordinator`` satisfies it by wrapping the image pipeline's separate popper, submitter,
    and tracker, so both flows are launched and observed uniformly through the process manager's registry.
    A flow may keep dispatch elsewhere (image generation's is interwoven with the VRAM budget in the
    control loop) or have no dispatch at all (text generation's generations run in a separate program
    reached over HTTP); the protocol covers the flow's identity, live work count, last pop time and
    lifecycle entry point.
    """

    @property
    def kind(self) -> WorkloadKind:
        """Which workload this coordinator runs."""
        ...

    @property
    def num_in_flight(self) -> int:
        """Units of work currently popped, dispatched, or awaiting submission for this flow."""
        ...

    @property
    def last_pop_time(self) -> float:
        """When this flow last asked the horde for work, or 0.0 before it has asked at all."""
        ...

    async def run(self) -> None:
        """Run the flow's pop -> dispatch -> submit loop until shutdown."""
        ...
