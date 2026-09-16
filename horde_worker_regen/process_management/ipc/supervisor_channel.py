"""The supervisor channel: structured state + control between a TUI/supervisor and the worker.

A supervising frontend (``horde_worker_regen.tui``) launches the worker as a child process and holds
one end of a duplex pipe. The worker pushes :class:`WorkerStateSnapshot` objects at a steady cadence
and drains :class:`SupervisorControlMessage` commands each loop tick. This mirrors the worker's own
internal IPC (see ``messages.py``) and is the structured upgrade of the ``.abort``-sentinel external
supervision hook already present in the control loop.

Models here are deliberately pure-data and JSON-round-trippable: the default transport is a
``multiprocessing`` pipe (pickle), but the same models serialize cleanly for the localhost-socket
fallback the launcher can swap to without touching any screen code.
"""

from __future__ import annotations

import enum
import threading
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from horde_worker_regen.process_management.models.download_scheduler import DownloadPriorityPolicy
from horde_worker_regen.process_management.models.feature_readiness import FeatureReadiness
from horde_worker_regen.process_management.scheduling.workload_kind import WorkloadKind

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from horde_sdk.ai_horde_api.apimodels.generate.pop import ImageGenerateJobPopPayload

    from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
    from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord
    from horde_worker_regen.process_management.resources.system_memory import SystemMemorySummary

SUPERVISOR_PROTOCOL_VERSION = 28
"""Bumped when the snapshot/command schema changes incompatibly; the TUI checks it on connect.

v2 added per-process ``num_jobs_completed`` and the snapshot's worker-details maintenance/paused and
last-pop no-jobs/skip-reason fields.
v3 added alchemy config/runtime fields, ``pending_jobs``, ``JobFeatureSummary``, and extended
``RecentJobRecord`` with model name and feature data.
v4 added the lightweight :class:`WorkerLivenessFrame`, emitted on its own cadence so the supervisor
can judge worker liveness independently of full-snapshot production.
v5 added the resolved model ``baseline`` to ``JobQueueEntry`` and ``RecentJobRecord`` so the queue and
recent-jobs tables can show a model's baseline alongside its name.
v6 added the snapshot's ``system_memory`` field (:class:`SystemMemorySnapshot`): total/available RAM and
the worker's per-role RSS share.
v7 added ``DownloadStatusSnapshot.active`` (the full set of concurrent in-flight downloads) for
host-parallel downloading; ``current`` is retained as the primary entry for single-download consumers.
v8 added per-card multi-GPU data: ``ProcessSnapshot.device_index`` (which card each slot is pinned to)
and the snapshot's ``per_card`` list of :class:`CardSnapshot` (per-card VRAM, contexts, residency, and
fault/unservable-model health). Additive: a single-GPU host reports exactly one ``CardSnapshot``.
v9 added the snapshot's ``feature_readiness`` (:class:`FeatureReadinessSummary`): the per-feature
deps+on-disk readiness the worker uses to decide which gated features it offers to the Horde.
v10 added ``orchestration_intent`` and ``work_ledger`` so the Overview can show the worker's current
decision and promote per-job state out of the process table.
v11 added worker-owned stats samples/history, model/baseline rollups, and the stats JSONL export control.
v12 added ``pop_governors`` (:class:`PopGovernorsSnapshot`): the live + session-aggregate state of every
pop/scheduling governor holding back or reshaping job pops, for the Overview strip and the stats tab.
v14 added :class:`WorkerFatalConfigError`, a one-shot frame the worker sends before exiting on a fatal,
non-retryable configuration problem (e.g. a worker name taken by another account) so the supervisor can
stop relaunching and the dashboard can show the specific reason and remedy instead of a generic crash.
v16 added post-processing lane counters and per-entry queue order so the TUI can show the image job route
through the dedicated post-processing process.
v17 added the operator-facing reason post-processing was session-disabled.
v19 added the snapshot's ``model_pool`` field (:class:`ModelPoolSnapshot`): the fixed model pool's seats,
bench, current advertising lane, demand-reading age, and download admission budget. Omitted (None) when the pool is
disabled, so a worker with no pool leaves the field unset and older supervisors are unaffected.
v20 added the snapshot's ``disagg_job_stages`` list (:class:`DisaggStageRow`): each in-flight
pipeline-disaggregated job's current stage and the process its stage is dispatched to, so the dashboard
can show mid-pipeline progress (encode / sample / decode) per job. Empty when disaggregation is disabled
or no disaggregated jobs are in flight. v20 also added ``ProcessSnapshot.resident_components``
(:class:`ResidentComponentEntry` entries): the model components a GPU-bearing child holds resident in its
RAM component cache, so the dashboard can show which models/components a lane actually holds. Empty for a
process with no component cache. The model-pool snapshot also gained cumulative fixed/free lane pop and
fulfillment counts for per-lane hit rates.
v21 distinguishes logical model-pool seating from measured residency. Each seat carries a readiness value,
resident process/GPU identities, and whether its last matched pop was already resident. The lane tallies add
resident-hit counts, allowing the TUI to report observed load avoidance rather than infer it from a seat match.
v22 added ``RecentJobRecord.kudos_reward``: what the horde's submit response paid for that job, so the
recent-jobs views can attribute earnings per job rather than only in the session total. None when the reward
is unknown (a faulted job, or one never delivered to the horde).
v23 added :class:`SamplerSummary` plus the ``sampler``/``batch_size`` fields on ``WorkLedgerEntry`` and
``RecentJobRecord``, so a job row can state what its sampler asks the model for per step and how many
images it produces. The worker resolves the sampler's published work profile and measured cost ratio
(it owns the ``horde_sdk`` import); a dashboard only formats what it is handed. v23 also dropped
``WorkLedgerEntry.raw_reason``: it carried the worker-wide pop-block reason on every row, which says
nothing about the job the row is for, and the same text is already on ``orchestration_intent.raw_gate``.
v24 makes text generation visible: ``RecentJobRecord.workload`` replaces its ``is_alchemy`` boolean (a
worker serving three workloads cannot be partitioned by one), the snapshot gains the ``text_*`` counters
and backend identity, and ``WorkerConfigSummary`` gains ``scribe``/``scribe_name``.
v25 makes a supervised external process a row in the process surfaces without entering ``ProcessMap``:
``ProcessSnapshot`` gains ``is_external``, ``os_pid``, ``display_state`` and ``text_backend``
(:class:`TextBackendDetail`), ``device_index`` accepts None for a process pinned to no card, and
:class:`SupervisedProcessSnapshotSource` is the one method such a process implements to produce its row.
A map-derived row leaves every added field at its default, so a v25 snapshot of an image-only worker is
what v24 produced.
v26 puts a text generation in flight on the surfaces that already show work. ``WorkLedgerEntry`` gains
``workload``, ``progress_unit`` (:class:`WorkLedgerProgressUnit`) and ``progress_age_seconds`` so a text
row states what its progress counts and how long since it last moved; :class:`TextBackendActivity` carries
the requests and token rate only the text flow can see onto the backend's row; ``RecentJobRecord`` gains
``generated_tokens``; and the snapshot gains ``workload_totals``
(:class:`WorkloadTotalsSnapshot` per workload), ``text_backend_not_ready_since`` and
``snapshot_interval_seconds``, with ``text_stall_seconds`` and ``text_backend_ready_patience_seconds``
on the config summary so the dashboard's text checks derive their thresholds from the worker rather than
restating them.
v27 makes the worker's identity and its text backend's posture readable from the summary alone. The config
summary gains ``dreamer`` (the third role flag, so every surface can name the enabled roles rather than
guessing from the workload list) and ``text_backend_managed`` (whether the worker launches the backend
itself, which decides the remedy a stuck backend is given); :attr:`WorkerConfigSummary.enabled_worker_names`
and :attr:`WorkerConfigSummary.worker_display_name` are the one place that identity is derived.
``TextBackendDetail`` gains ``launching_since``, the instant the current launch attempt began, which is
what the readiness patience window is measured from for a backend the worker launches, and
``provision_error``, why its program could not be obtained; ``port`` becomes optional because the row now
exists from before the command line is rendered (the supervised backend's life begins at obtaining it, and
``TextBackendState.PROVISIONING`` is the state it reports meanwhile). The snapshot's whole-worker counters
(``num_jobs_popped``, ``num_jobs_submitted``, ``num_jobs_faulted``, ``jobs_in_progress``,
``seconds_since_last_pop``) now include alchemy and text work, so a worker serving either alone reports its
own headline figures rather than zero; the ``alchemy_*`` and ``text_*`` fields and ``workload_totals`` remain
the per-workload split.
v28 adds ``recent_events`` (:class:`WorkerEvent` entries, bounded by :data:`RECENT_EVENTS_IN_SNAPSHOT`): the
worker's own transitions as they happened (pops, preloads, finished and faulted work, backend launches,
process recoveries, finished downloads, maintenance and pop-backoff edges), each with a per-session
``sequence`` so a reconnecting frontend can tell events it has shown from ones it has not. The ring is
recorded where the finished-job records are, so it and ``recent_jobs`` cannot disagree about a job.
"""

RECENT_JOBS_IN_SNAPSHOT = 25
"""How many of the most recent finished-job records to carry in a snapshot (bounds payload size)."""

RECENT_EVENTS_IN_SNAPSHOT = 40
"""How many of the most recent worker events to carry in a snapshot (bounds payload size).

Above the finished-job bound because finishing a job is one of a dozen kinds recorded here: a worker
that finished twenty-five jobs also preloaded models, launched a backend and recovered a process
between them, and a ring the size of the job list would evict the surrounding transitions that explain
the jobs. Small enough that the whole ring rides every snapshot without slicing."""

PENDING_JOBS_IN_SNAPSHOT = 8
"""How many pending-inference jobs to carry in a snapshot (bounds payload size)."""

WORK_LEDGER_ENTRIES_IN_SNAPSHOT = 18
"""How many active/recent job rows to carry for the Overview work ledger (bounds payload size)."""


class SamplerSummary(BaseModel):
    """One job's sampler and what a step of it costs, for compact display.

    The worker fills this in from the SDK's published sampler tables, because this module is imported by
    the TUI and the fake worker and stays free of the heavy ``horde_sdk`` import chain.
    """

    name: str
    """The sampler as the horde asked for it (e.g. ``k_dpmpp_sde``), not canonicalized to a backend name."""
    work_per_step: int | None = None
    """First-order-equivalent model evaluations one schedule step costs: the sampler's order.

    None when the sampler picks its own iteration count (see :attr:`adaptive`) or the SDK publishes no
    profile for it."""
    adaptive: bool = False
    """Whether the sampler decides its own iteration count instead of running the requested steps."""
    cost_ratio: float | None = None
    """Measured per-step wall-clock cost relative to ``k_euler`` on the job's model family.

    One card's evidence, published by the SDK; it is neither a price nor a promise about this machine.
    None where the SDK states no ratio."""


class JobFeatureSummary(BaseModel):
    """The notable features of one image-generation job, for compact display."""

    loras: int = 0
    tis: int = 0
    control_type: str | None = None
    post_processing: list[str] = Field(default_factory=list)
    hires_fix: bool = False
    workflow: str | None = None

    @classmethod
    def from_payload(cls, payload: ImageGenerateJobPopPayload) -> JobFeatureSummary:
        """Summarize the notable features of an image-generation job payload."""
        return cls(
            loras=len(payload.loras) if payload.loras else 0,
            tis=len(payload.tis) if payload.tis else 0,
            control_type=str(payload.control_type) if payload.control_type else None,
            post_processing=[str(post_proc_step) for post_proc_step in payload.post_processing],
            hires_fix=payload.hires_fix,
            workflow=str(payload.workflow) if payload.workflow else None,
        )

    def as_tags(self) -> list[str]:
        """Compact display tags (e.g. ``['2×LoRA', 'canny', 'HiRes']``)."""
        tags: list[str] = []
        if self.loras:
            tags.append(f"{self.loras}×LoRA")
        if self.tis:
            tags.append(f"{self.tis}×TI")
        if self.control_type:
            tags.append(self.control_type)
        for post_proc_step in self.post_processing:
            tags.append(post_proc_step)
        if self.hires_fix:
            tags.append("HiRes")
        if self.workflow:
            tags.append(f"wf:{self.workflow}")
        return tags

    def is_empty(self) -> bool:
        """True when no notable features are present."""
        return not (
            self.loras or self.tis or self.control_type or self.post_processing or self.hires_fix or self.workflow
        )


class JobQueueEntry(BaseModel):
    """A pending-inference job, for the queue-preview on the overview screen."""

    job_id: str
    model: str
    baseline: str | None = None
    """The model's image baseline (e.g. ``stable_diffusion_xl``); None when it could not be resolved."""
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    features: JobFeatureSummary | None = None
    queue_order: int | None = None
    """1-based order among currently tracked image jobs by pop order; None for non-image forms."""


class WorkLedgerStage(enum.StrEnum):
    """A job's operator-facing stage in the Overview work ledger."""

    QUEUED = "queued"
    PREPARING = "preparing"
    INFERENCE = "inference"
    POST_PROCESSING = "post_processing"
    SAFETY = "safety"
    SUBMIT = "submit"
    COMPLETED = "completed"
    FAULTED = "faulted"


class WorkLedgerProgressUnit(enum.StrEnum):
    """What a work-ledger row's ``progress_current``/``progress_total`` count.

    An image job's progress has always been sampler steps and is rendered without a unit word, because
    that is what a reader of the column assumes. A text generation counts something else, and which of
    the two it counts depends on what its backend can report, so the row says so rather than leaving a
    reader to infer it from the model name.
    """

    STEPS = "steps"
    TOKENS = "tokens"
    """Tokens the backend itself counted; only a backend with a statistics route reports them."""
    CHUNKS = "chunks"
    """Stream records that have arrived, which is not a token count and has no total to divide by."""


class WorkLedgerEntry(BaseModel):
    """One active or recently-finished job row for the Overview work ledger."""

    job_id: str
    stage: WorkLedgerStage
    workload: WorkloadKind = WorkloadKind.IMAGE_GENERATION
    """Which workload this row's work belongs to, so one ledger can carry all three."""
    model: str | None = None
    baseline: str | None = None
    process_id: int | None = None
    device_index: int | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    progress_unit: WorkLedgerProgressUnit = WorkLedgerProgressUnit.STEPS
    """What the two progress figures count. A total is absent whenever the unit cannot supply one."""
    progress_age_seconds: float | None = None
    """Seconds since this row's progress last advanced, or None when nothing measures it.

    Filled where the flow observes the work arriving continuously (a streamed text generation), so a row
    that has stopped moving can be told from one that is moving slowly. An image job's staleness is read
    from its process's heartbeat instead, so its rows leave this unset."""
    iterations_per_second: float | None = None
    """The rate the work is arriving at: sampler iterations for an image job, generated tokens for a text
    one, and None wherever the flow has no measured figure."""
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    batch_size: int = 1
    """How many images the job asks for (the request's ``n``); 1 for an ordinary single-image job."""
    sampler: SamplerSummary | None = None
    """The job's sampler and what a step of it costs; None for alchemy rows and jobs with no payload."""
    features: JobFeatureSummary | None = None
    queue_order: int | None = None
    """1-based order among currently tracked image jobs by pop order; None for recent/alchemy rows."""
    age_seconds: float | None = None
    queue_wait_seconds: float | None = None
    safety_seconds: float | None = None
    e2e_seconds: float | None = None
    intent: str | None = None
    faulted: bool = False


class DisaggStageRow(BaseModel):
    """Represents one in-flight pipeline-disaggregated job's current stage and dispatch target.

    Carried on :attr:`WorkerStateSnapshot.disagg_job_stages` so the dashboard can show where each
    disaggregated job is in its pipeline (awaiting conditioning, sampling, awaiting decode, ...). The
    ``process_id``/``process_launch_identifier`` pair is the process the current stage is dispatched to,
    both None while the stage is queued for a role process that is not yet available.
    """

    job_id: str
    stage: str
    """The :class:`~horde_worker_regen.process_management.workers.disaggregation_orchestrator.DisaggJobStage`
    value (e.g. ``sampling``, ``awaiting_latent_decode``)."""
    process_id: int | None = None
    process_launch_identifier: int | None = None


class OrchestrationIntentSnapshot(BaseModel):
    """The scheduler/popper's current plain-English intent for the Overview Now/Next/Why strip."""

    summary: str = "Waiting for worker state."
    next_action: str | None = None
    why: str | None = None
    raw_gate: str | None = None
    target_job_id: str | None = None
    target_model: str | None = None
    target_process_id: int | None = None
    target_device_index: int | None = None
    updated_at: float = Field(default_factory=time.time)


class WorkerConfigSummary(BaseModel):
    """The operationally-relevant bridge-data fields the overview/worker panels display.

    This is a compact projection of ``reGenBridgeData``; the config *editor* reads the full
    ``bridgeData.yaml`` directly, so the snapshot only carries what the dashboards render.
    """

    dreamer_name: str
    dreamer: bool = True
    """Whether the operator selected the image-generation role.

    Defaults on, which is what a worker predating the field meant: the role is on by default and the field
    exists so a surface can tell a dreamer's name from a name the worker never advertises."""
    alchemist_name: str | None = None
    """The worker's alchemist identity, shown in place of the dreamer name on an alchemist-only worker."""
    worker_version: str
    horde_username: str | None = None
    num_models: int = 0
    custom_models: bool = False
    custom_models_ready: int = 0
    custom_models_configured: int = 0
    custom_model_issues: tuple[str, ...] = ()
    max_power: int = 8
    max_threads: int = 1
    queue_size: int = 1
    max_batch: int = 1
    safety_on_gpu: bool = False
    allow_img2img: bool = True
    allow_lora: bool = False
    effective_allow_lora: bool | None = None
    """Whether job pops currently advertise LoRA support; None means same as ``allow_lora``."""
    allow_controlnet: bool = False
    allow_sdxl_controlnet: bool = False
    allow_post_processing: bool = True
    high_performance_mode: bool = False
    moderate_performance_mode: bool = False
    extra_slow_worker: bool = False

    alchemist: bool = False
    alchemy_concurrent: bool = True
    alchemy_max_concurrency: int = 1
    alchemy_vram_headroom_mb: int = 2000
    alchemy_caption_enabled: bool = False
    alchemy_forms: list[str] = Field(default_factory=list)

    scribe: bool = False
    """Whether the operator selected the text-generation role."""
    scribe_name: str | None = None
    """The worker's scribe identity, shown in place of the dreamer name on a scribe-only worker.

    Each role registers as its own separately-named worker on the horde, so this is never the dreamer's
    name. None when the role is off."""
    text_stall_seconds: float = 30.0
    """How long the worker lets a begun generation stay silent before it abandons the job.

    Carried so the dashboard's stall check fires on the worker's own patience rather than a figure of its
    own; the default is the bridge-data default, which is what a worker that predates the field meant."""
    text_backend_ready_patience_seconds: float = 120.0
    """How long the readiness gate waits before it says out loud that the backend has not answered.

    The same figure the gate's own log line uses, so the check and the log agree about when a cold start
    has stopped being a cold start. Not operator-configurable today; it rides the config summary because
    that is where the dashboard's text thresholds come from."""
    text_backend_managed: bool = False
    """Whether the worker launches the text backend itself rather than reaching one the operator runs.

    The two have different remedies when the backend does not answer, and only the worker knows which
    arrangement it is in: a managed backend's failure is in the worker's own log and its supervised row,
    while an attached one is the operator's program at the address they configured."""

    @property
    def enabled_worker_names(self) -> tuple[str, ...]:
        """Return the horde names of the roles this operator enabled, in the order dreamer, alchemist, scribe.

        Each role registers as its own separately-named worker on the horde, so a worker with two roles on
        has two identities and a surface that shows one of them is showing half the worker. The role flags
        decide this rather than the served-workload list: a role whose backend is not ready yet is still
        part of who the worker is. Empty only when every role is off, which the worker warns about at
        start-up.
        """
        by_role = (
            (self.dreamer, self.dreamer_name),
            (self.alchemist, self.alchemist_name),
            (self.scribe, self.scribe_name),
        )
        return tuple(name for enabled, name in by_role if enabled and name)

    @property
    def worker_display_name(self) -> str:
        """Return the enabled roles' names as one phrase, for a surface with room for all of them.

        Falls back to the dreamer name when no role is enabled, so a misconfigured worker still has
        something to be called rather than an empty line.
        """
        return " and ".join(self.enabled_worker_names) or self.dreamer_name


class ResidentComponentEntry(BaseModel):
    """Represents one model component a process holds resident in its RAM component cache, for the dashboard.

    The wire twin of the child-reported ``HeldComponentSnapshot``
    (:mod:`~horde_worker_regen.process_management.ipc.messages`), duplicated here rather than imported so this
    module stays a leaf the message layer can depend on without a cycle (as ``SystemMemorySnapshot`` twins
    ``SystemMemorySummary``). The fields carry the cache's own view, so ``approx_ram_mb`` is only the
    accounted portion of the process's resident-set size.
    """

    kind: str
    """The component kind (``unet``/``clip``/``vae``/``checkpoint``); a ``str`` so an unknown future kind survives."""
    identity: str
    """The content identity of the cached entry (a checkpoint entry's identity is the bare horde model name)."""
    approx_ram_mb: float
    """The entry's approximate resident RAM cost in megabytes, as the component cache's budget estimates it."""


SUPERVISED_PROCESS_ID_BASE = 9100
"""First process id reserved for a supervised external process, which never enters ``ProcessMap``.

The map allocates slot ids from 0 upward, one per pipe-bearing child
(``lifecycle/process_lifecycle.py::_allocate_inference_pid``), and the download process holds the reserved
id 9000 (``workers/download_process.py::DOWNLOAD_PROCESS_ID``). A base of 9100 sits clear of both, leaves
9000-9099 for further reserved single-process ids, and stays a small positive number so the dashboards that
print an id numerically need no special case. Each supervised process takes a fixed offset from this base.
"""

TEXT_BACKEND_PROCESS_ID = SUPERVISED_PROCESS_ID_BASE
"""The reserved row id of the supervised text backend (offset 0); one backend is supervised at a time."""


class TextBackendActivity(BaseModel):
    """Represents what the text flow is currently asking of its backend, for the backend's row.

    The supervisor watches the process and never sees a generation, so these three figures reach the row
    from the coordinator that owns the jobs. Handed to
    [`to_process_snapshot`][horde_worker_regen.process_management.lifecycle.text_backend_supervisor.TextBackendSupervisor.to_process_snapshot]
    rather than held by the supervisor, so no copy of them can go stale between snapshots.
    """

    active_requests: int = 0
    """Generations the backend has begun producing output for."""
    queued_requests: int = 0
    """Jobs the worker has popped and not yet had taken, including one a busy backend turned away."""
    tokens_per_second: float | None = None
    """The most recent rate any in-flight generation reported, or None when none of them can report one.

    A stock backend counts no tokens, so this stays None for its whole life rather than being derived
    from stream records, which carry an arbitrary number of tokens each."""


class TextBackendDetail(BaseModel):
    """Represents what a supervised text backend can say about itself, for the dashboard's row.

    The text backend is an external program the worker launches but does not control, so most of this is
    what it reported rather than what the worker configured: the model name and the caps come from the
    backend's own description, and the footprint is the card's measured free-VRAM drop across its launch
    (it exposes no runtime memory query). A field the current backend cannot answer stays None rather than
    being filled from configuration, because the configured value and the loaded one disagree exactly when
    an operator needs to see it. Nothing here feeds admission or pricing.
    """

    kind: str
    """The ``TEXT_BACKENDS`` value naming the program (``koboldcpp``); a ``str`` so a future kind survives."""
    model_name: str | None = None
    """The model the backend reports having loaded; None before it has been asked or when it did not answer."""
    port: int | None = None
    """The loopback port the backend listens on, or None before its command line has been rendered.

    A managed backend's row exists from the moment the worker sets out to have one, which is before the
    program has been obtained, and the port is part of that rendered command line rather than a fact the
    row can state earlier."""
    context_length: int | None = None
    """The largest prompt-plus-generation token count the backend says it accepts."""
    max_length: int | None = None
    """The largest number of tokens the backend says it generates for one request."""
    launch_count: int = 0
    """How many times the supervisor has started the backend process this session."""
    footprint_mb: int | None = None
    """The card's free-VRAM drop between launch and readiness, or None when it could not be measured.

    A measured whole-process footprint, not a torch allocator reading: it is not comparable with a slot's
    ``vram_usage_mb`` and the dashboards mark it as measured where they show it."""
    launching_since: float | None = None
    """When the most recent launch attempt began, or None before the worker has attempted one.

    The instant a readiness patience window is measured from for a backend the worker launches: obtaining
    the program can take minutes of downloading before a launch is even attempted, and a clock started
    before that reads a legitimate first-run provision as a backend that will not start. Stamped per
    attempt, so a relaunch is given the same patience the first launch had."""
    provision_error: str | None = None
    """Why the backend's program could not be obtained, or None while that has not happened.

    A backend that cannot be obtained is never launched, so without this the row and the headline would sit
    at "obtaining" for the rest of the session over a program that is never coming."""
    ready_since: float | None = None
    """When the backend first answered its readiness probe for the current launch; None when not serving."""
    last_health_ok_at: float | None = None
    """When the backend last answered a readiness probe.

    The supervisor probes only while a launch is starting, so while serving this stays the moment readiness
    was reached: the worker asks a black box nothing it does not need, and the process's continued existence
    is what the supervisor watches instead."""
    relaunch_backoff_seconds: float | None = None
    """Seconds left of the pause before the next launch attempt; None when not backing off."""
    active_requests: int = 0
    """Generations the worker currently has in flight against the backend."""
    queued_requests: int = 0
    """Generations the worker is holding for the backend to take."""
    tokens_per_second: float | None = None
    """The most recent generation rate observed, or None when nothing has been generated yet."""
    model_buffer_mb: int | None = None
    """Weights the backend's loader reported placing in buffers, summed over devices (MiB, as it prints)."""
    kv_buffer_mb: int | None = None
    """The KV cache the backend's loader reported allocating, summed over devices (MiB, as it prints)."""
    compute_buffer_mb: int | None = None
    """The compute buffers the backend's loader reported reserving, summed over devices (MiB, as it prints)."""


class ProcessSnapshot(BaseModel):
    """A serializable projection of one process's live state, for the dashboard's process surfaces.

    Most rows are built from a ``HordeProcessInfo`` in ``ProcessMap`` by :meth:`from_process_info`. A row
    with :attr:`is_external` set instead comes from a supervised external program the worker launched but
    does not speak IPC with (:class:`SupervisedProcessSnapshotSource`); it carries :attr:`display_state` and
    a typed detail block in place of the allocator readings and job progress a child reports, and every
    field it cannot answer keeps its default.
    """

    process_id: int
    process_type: str
    """The ``HordeProcessType`` name (e.g. ``INFERENCE`` / ``SAFETY``)."""
    device_index: int | None = 0
    """The stable index of the GPU this slot is pinned to (0 on a single-GPU host).

    Lets the dashboard group a card's slots together and show a per-process GPU column on multi-GPU hosts.
    None for an external process that runs on no card (a CPU-only text backend)."""
    last_process_state: str
    """The process's own state-vocabulary name (e.g. ``WAITING_FOR_JOB`` / ``INFERENCE_STARTING``).

    A ``HordeProcessState`` name on a map-derived row. An external row carries its supervisor's state name
    instead, which is deliberately disjoint from the image vocabulary so no state-keyed reader matches it;
    what an operator reads is :attr:`display_state`."""
    is_alive: bool
    is_busy: bool

    is_external: bool = False
    """Whether this row is a supervised external program rather than a pipe-bearing child of the worker.

    False for every row derived from ``ProcessMap``. A reader reasoning about inference lanes (sampling,
    serving, per-card work) skips the rows where this is True: they hold no torch allocator, run no horde
    image job, and their state vocabulary is their own."""
    os_pid: int | None = None
    """The operating-system process id, where the row has one worth showing; None for a map-derived row."""
    display_state: str | None = None
    """A free-form operator-facing label for an external row (``ready``, ``relaunching in 8 s``).

    None on a map-derived row, which is labelled from :attr:`last_process_state` instead."""
    text_backend: TextBackendDetail | None = None
    """The text backend's own detail, when this row is one; None for every other row."""

    loaded_horde_model_name: str | None = None
    loaded_horde_model_baseline: str | None = None
    current_job_id: str | None = None

    last_heartbeat_timestamp: float = 0.0
    last_heartbeat_delta: float = 0.0
    last_heartbeat_type: str = "OTHER"
    heartbeats_inference_steps: int = 0
    last_heartbeat_percent_complete: int | None = None

    ram_usage_bytes: int = 0
    vram_usage_mb: int = 0
    total_vram_mb: int = 0
    batch_amount: int = 1

    current_job_width: int | None = None
    current_job_height: int | None = None
    current_job_steps: int | None = None
    """The active job's resolution and step count (None when idle); surfaced as ``W×H`` / steps."""

    last_iterations_per_second: float | None = None
    last_current_step: int | None = None
    last_total_steps: int | None = None

    vram_used_high_water_mb: int = 0
    ram_used_high_water_mb: int = 0

    num_jobs_completed: int = 0
    """Jobs/forms this slot has finished (inference, safety check, or alchemy form); resets on replace."""

    current_job_features: JobFeatureSummary | None = None
    """Notable features of the active job (LoRAs, ControlNet, etc.); None when idle."""

    resident_components: list[ResidentComponentEntry] = Field(default_factory=list)
    """The model components this process last reported holding resident in its RAM component cache.

    Empty for a process that carries no component cache (safety, download, utilities) or has not reported.
    This is the component cache's own listing, so it is approximate: the entries' summed ``approx_ram_mb``
    is only the accounted portion of ``ram_usage_bytes``, and allocator-retained or comfy-held stale pages
    the cache no longer lists are the remainder (RSS minus the listed approximation)."""

    @classmethod
    def from_process_info(cls, info: HordeProcessInfo) -> ProcessSnapshot:
        """Build a snapshot from a live ``HordeProcessInfo`` (read-only; no import coupling)."""
        job = info.last_job_referenced
        current_job_id = str(job.id_.root) if job is not None and job.id_ is not None else None
        baseline = info.loaded_horde_model_baseline

        features: JobFeatureSummary | None = None
        current_job_width: int | None = None
        current_job_height: int | None = None
        current_job_steps: int | None = None
        if info.is_process_busy() and job is not None:
            candidate = JobFeatureSummary.from_payload(job.payload)
            if not candidate.is_empty():
                features = candidate
            current_job_width = job.payload.width
            current_job_height = job.payload.height
            current_job_steps = job.payload.ddim_steps

        return cls(
            process_id=info.process_id,
            process_type=info.process_type.name,
            device_index=info.device_index,
            last_process_state=info.last_process_state.name,
            is_alive=info.is_process_alive(),
            is_busy=info.is_process_busy(),
            loaded_horde_model_name=info.loaded_horde_model_name,
            loaded_horde_model_baseline=str(baseline) if baseline is not None else None,
            current_job_id=current_job_id,
            last_heartbeat_timestamp=info.last_heartbeat_timestamp,
            last_heartbeat_delta=info.last_heartbeat_delta,
            last_heartbeat_type=info.last_heartbeat_type.name,
            heartbeats_inference_steps=info.heartbeats_inference_steps,
            last_heartbeat_percent_complete=info.last_heartbeat_percent_complete,
            ram_usage_bytes=info.ram_usage_bytes,
            vram_usage_mb=info.vram_usage_mb,
            total_vram_mb=info.total_vram_mb,
            batch_amount=info.batch_amount,
            current_job_width=current_job_width,
            current_job_height=current_job_height,
            current_job_steps=current_job_steps,
            last_iterations_per_second=info.last_iterations_per_second,
            last_current_step=info.last_current_step,
            last_total_steps=info.last_total_steps,
            vram_used_high_water_mb=info.vram_used_high_water_mb,
            ram_used_high_water_mb=info.ram_used_high_water_mb,
            num_jobs_completed=info.num_jobs_completed,
            current_job_features=features,
            resident_components=[
                ResidentComponentEntry(kind=held.kind, identity=held.identity, approx_ram_mb=held.approx_ram_mb)
                for held in info.held_components
            ]
            if info.held_components
            else [],
        )


@runtime_checkable
class SupervisedProcessSnapshotSource(Protocol):
    """Something the worker supervises that can project itself as one process-table row.

    An external program the worker launches (the text backend today) is not in ``ProcessMap``: the map's
    entries are pipe-bearing children with torch allocator readings and an image-and-alchemy state
    vocabulary, and every untyped walk of the map would misread an entry that has neither. This protocol is
    the whole of what such a program owes the dashboard, so the snapshot builder can concatenate the
    map-derived rows with the supervised ones and every surface renders one list.

    Runtime-checkable so an assembly can assert a stand-in still produces a row; the check confirms the
    method is present, not that what it returns is honest.

    An implementation may take further optional arguments for facts it cannot see for itself, which the
    snapshot builder then supplies (the text backend's supervisor takes a :class:`TextBackendActivity`:
    it watches the process and never sees a generation). Anything asked for that way is optional, so a
    caller holding only this protocol still gets a complete row.
    """

    def to_process_snapshot(self) -> ProcessSnapshot:
        """Return this process's row, with ``is_external`` set and its own typed detail attached."""
        ...


class RecentJobRecord(BaseModel):
    """A lean projection of one finished job, for the insights/recent-activity views.

    Deliberately a local model (not ``run_metrics.JobMetricsRecord``) so this module, imported by
    the TUI and the fake worker, stays free of the heavy ``horde_sdk``/logfire import chain.
    """

    job_id: str
    workload: WorkloadKind = WorkloadKind.IMAGE_GENERATION
    """Which workload produced this job."""
    faulted: bool = False
    queue_wait_seconds: float | None = None
    e2e_seconds: float | None = None
    safety_seconds: float | None = None
    model_name: str | None = None
    baseline: str | None = None
    """The model's image baseline (e.g. ``stable_diffusion_xl``); None when it could not be resolved."""
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    batch_size: int = 1
    """How many images the job asked for (the request's ``n``); 1 for an ordinary single-image job."""
    sampler: SamplerSummary | None = None
    """The sampler the job ran and what a step of it costs; None for alchemy forms and stage records."""
    features: JobFeatureSummary | None = None
    kudos_reward: float | None = None
    """What the horde paid for this job, or None when no reward is known (faulted, or never delivered)."""
    generated_tokens: int | None = None
    """Tokens a text generation produced, or None for every other workload and for a backend that counts
    none of its own."""

    @classmethod
    def from_metrics_record(
        cls,
        record: JobMetricsRecord,
        baseline: str | None = None,
        sampler: SamplerSummary | None = None,
    ) -> RecentJobRecord:
        """Project a worker-side ``JobMetricsRecord`` into the lean wire form.

        ``baseline`` and ``sampler`` are resolved by the caller (the process manager, which owns the model
        metadata and the SDK's sampler tables) since ``JobMetricsRecord`` carries neither.
        """
        features: JobFeatureSummary | None = None
        has_features = bool(
            record.loras_count
            or record.tis_count
            or record.control_type
            or record.post_processing
            or record.hires_fix,
        )
        if has_features:
            features = JobFeatureSummary(
                loras=record.loras_count,
                tis=record.tis_count,
                control_type=record.control_type,
                post_processing=record.post_processing,
                hires_fix=record.hires_fix,
            )
        return cls(
            job_id=record.job_id,
            workload=record.workload,
            faulted=record.faulted,
            queue_wait_seconds=record.queue_wait_seconds,
            e2e_seconds=record.e2e_seconds,
            safety_seconds=record.safety_seconds,
            model_name=record.model_name,
            baseline=baseline,
            steps=record.steps,
            width=record.width,
            height=record.height,
            batch_size=record.batch_count,
            sampler=sampler,
            features=features,
            kudos_reward=record.kudos_reward,
            generated_tokens=record.generated_tokens,
        )


class WorkerEventKind(enum.StrEnum):
    """One kind of worker transition carried on the event ring.

    Each member names a transition the worker already makes a decision at, not a state it can be asked
    about: an event is recorded where the change happens, so the ring is a history rather than a sampled
    difference between snapshots. The pop, preload and load members are named after the same transitions
    the log-signature registry (``analysis/log_signatures.py``) registers at those sites.
    """

    JOB_POPPED = "job_popped"
    PRELOAD_STARTED = "preload_started"
    PRELOAD_READY = "preload_ready"
    JOB_FINISHED = "job_finished"
    JOB_FAULTED = "job_faulted"
    BACKEND_LAUNCHED = "backend_launched"
    BACKEND_READY = "backend_ready"
    BACKEND_RELAUNCHED = "backend_relaunched"
    PROCESS_RECOVERED = "process_recovered"
    DOWNLOAD_FINISHED = "download_finished"
    MAINTENANCE_ON = "maintenance_on"
    MAINTENANCE_OFF = "maintenance_off"
    POP_BACKOFF_ENTERED = "pop_backoff_entered"
    POP_BACKOFF_LEFT = "pop_backoff_left"


class WorkerEventPayload(BaseModel):
    """Represents the few extra facts an event's own site holds, the same shape for every kind.

    One payload class rather than a subclass per kind: the ring is a list of one model, so a frontend
    reads it without narrowing, and a reader that does not know a kind still gets its fields. A field a
    kind has nothing to say about stays None rather than becoming a zero.
    """

    process_id: int | None = None
    """The worker's own process id for a preload or a recovered slot; None for a site that has none."""
    device_index: int | None = None
    """The card the event happened on; None when the site is not pinned to one."""
    count: int | None = None
    """Whatever the event counts: a backend's launch number, a generation's tokens."""
    detail: str | None = None
    """Why, in the words the site already had (a recovery reason, the model a preload displaces)."""


class WorkerEvent(BaseModel):
    """Represents one worker transition, as the site that made it saw it.

    The ring these sit on is bounded, so a frontend that has been away longer than the bound has missed
    events; :attr:`sequence` is what lets it say so rather than silently re-showing or skipping work.
    """

    model_config = ConfigDict(frozen=True)

    sequence: int
    """Monotonic within one worker session, starting at 1.

    A reconnecting frontend keeps the highest sequence it has shown: an event at or below it has been
    shown, and a ring whose lowest sequence is above it proves events were evicted unseen."""
    kind: WorkerEventKind
    timestamp: float = Field(default_factory=time.time)
    """The worker's wall clock when the transition happened, which is the clock its snapshots use."""
    workload: WorkloadKind | None = None
    """The workload the event belongs to; None for a worker-wide event (a process recovery, a download)."""
    model: str | None = None
    job_id: str | None = None
    duration_seconds: float | None = None
    """How long the thing that just ended took, where the site measured it."""
    kudos: float | None = None
    """What the horde paid, for a finished job the horde paid for; None everywhere else."""
    payload: WorkerEventPayload = Field(default_factory=WorkerEventPayload)


class WorkerEventSink(Protocol):
    """The callable a collaborator with no run-metrics reference is handed to record its transitions.

    Structurally satisfied by ``WorkerRunMetrics.record_event``, which owns the ring and assigns the
    sequence, so a site emits without importing the metrics aggregator: the process manager injects the
    bound method. Declared here beside :class:`WorkerEventKind` so an emitting site needs one import.
    """

    def __call__(
        self,
        kind: WorkerEventKind,
        *,
        workload: WorkloadKind | None = ...,
        model: str | None = ...,
        job_id: str | None = ...,
        duration_seconds: float | None = ...,
        kudos: float | None = ...,
        process_id: int | None = ...,
        device_index: int | None = ...,
        count: int | None = ...,
        detail: str | None = ...,
    ) -> None:
        """Record one worker transition (see ``WorkerRunMetrics.record_event``)."""
        ...


class WorkloadTotalsSnapshot(BaseModel):
    """Represents one workload's session totals, for a per-workload read of a run.

    The wire twin of ``run_metrics.WorkloadTotals``, duplicated here rather than imported so this module
    stays a leaf the metrics layer can depend on without a cycle (as :class:`ResidentComponentEntry`
    twins the child's held-component report). Derived from the finished-job records, so a worker serving
    three workloads has three sets of headline numbers instead of one pair that describes none of them.
    """

    completed: int = 0
    """Records for work that did not fault."""
    faulted: int = 0
    """Records for work that faulted. With :attr:`completed` this partitions the workload's records."""
    kudos: float = 0.0
    """Kudos the horde paid for this workload's completed work; 0.0 when no record carried a reward."""
    mean_generated_tokens: float | None = None
    """Mean tokens per record for the records that reported a token count, or None when none did.

    Only a text workload can carry this, and only against a backend that counts its own tokens; a stock
    one reports nothing a mean could be taken of."""


class StatsSample(BaseModel):
    """One lightweight, worker-owned statistics sample for trend history and JSONL export."""

    timestamp: float = Field(default_factory=time.time)
    jobs_submitted: int = 0
    jobs_faulted: int = 0
    kudos_per_hour: float | None = None
    kudos_this_session: float | None = None
    """Cumulative kudos earned this session at sample time, for windowed kudos/hr deltas."""
    eligible_seconds_total: float = 0.0
    """Cumulative productive (pipeline-occupied) seconds at sample time; the kudos/hr denominator."""
    gpu_duty_percent: float | None = None
    gpu_busy_fraction: float | None = None
    pending_megapixelsteps: int = 0
    jobs_pending_inference: int = 0
    jobs_in_progress: int = 0
    jobs_pending_safety_check: int = 0
    jobs_being_safety_checked: int = 0
    jobs_pending_post_processing: int = 0
    jobs_being_post_processed: int = 0
    jobs_pending_submit: int = 0
    time_spent_no_jobs_available: float = 0.0
    num_process_recoveries: int = 0
    num_job_slowdowns: int = 0
    alchemy_forms_pending: int = 0
    alchemy_forms_in_flight: int = 0
    alchemy_forms_awaiting_submit: int = 0
    alchemy_total_submitted: int = 0
    alchemy_total_faulted: int = 0
    process_state_summary: str = ""
    """Compact per-process state line for offline duty-cycle attribution."""
    orchestration_intent_summary: str = ""
    """The scheduler/popper's current high-level intent at sample time."""
    orchestration_next_action: str | None = None
    """The next planned orchestration action, when known."""
    orchestration_why: str | None = None
    """Human-readable reason for the current orchestration decision."""
    orchestration_raw_gate: str | None = None
    """Raw gate/blocking reason behind the orchestration decision, when available."""
    maintenance_mode: bool = False
    self_throttle_paused: bool = False
    supervisor_paused: bool = False
    last_pop_maintenance_mode: bool = False
    worker_details_maintenance: bool = False
    in_error_backoff: bool = False
    last_pop_no_jobs_available: bool = False
    last_pop_skipped_reasons: dict[str, int] = Field(default_factory=dict)
    last_reduced_pop_skipped_reasons: dict[str, int] = Field(default_factory=dict)
    last_reduced_pop_max_power: int | None = None
    churn_counts: dict[str, int] = Field(default_factory=dict)
    """Cumulative reload/respawn churn counts by kind at sample time."""
    slot_duty_totals: dict[str, float] = Field(default_factory=dict)
    """Cumulative slot-seconds per slot-duty bucket (sampling vs each empty-slot attribution) at sample
    time. Monotonically growing; consumers difference two samples for a window's capacity-normalized
    active/idle/gated breakdown."""
    slot_duty_capacity: int = 0
    """Configured concurrent-inference slot count the slot-duty totals are normalized against."""
    dispatch_hold_bucket: str | None = None
    """The slot-duty bucket currently holding the next dispatch (None when dispatching or no work waits)."""


class StatsRollupRow(BaseModel):
    """Incremental rollup of finalized jobs by model/baseline (image) or by form (alchemy)."""

    model: str | None = None
    baseline: str | None = None
    jobs: int = 0
    megapixelsteps: float = 0.0
    sampling_seconds: float = 0.0
    e2e_seconds: float = 0.0
    batch_gt_one_jobs: int = 0
    faulted_jobs: int = 0
    """How many of the folded jobs/forms faulted (the by-form table surfaces this per form)."""
    vram_high_water_mb: int = 0
    """Peak VRAM high-water observed across the folded jobs/forms, when a child reported it (0 otherwise)."""


class StatsExportState(BaseModel):
    """Current worker-side JSONL export state."""

    enabled: bool = False
    active_file_path: str | None = None
    bytes_in_stats_files: int = 0
    warning_over_50_mib: bool = False
    last_write_error: str | None = None


class StatsHistoryBackfill(BaseModel):
    """Recent exact samples plus a decimated all-session series for reconnecting frontends."""

    recent_samples: list[StatsSample] = Field(default_factory=list)
    all_session_samples: list[StatsSample] = Field(default_factory=list)


class DownloadPhase(enum.StrEnum):
    """What the background download process is doing right now."""

    INITIALIZING = "initializing"
    """Loading model managers / fetching the model reference (a network call on first run)."""
    SCANNING = "scanning"
    """Verifying which configured models are already on disk (may hash large files on first run)."""
    DOWNLOADING = "downloading"
    """Actively downloading one or more models."""
    IDLE = "idle"
    """Nothing queued; all requested models are present (or were skipped)."""
    PAUSED = "paused"
    """Downloads are paused by the operator; queued work is held."""
    ERROR = "error"
    """The download subsystem hit an unrecoverable error (see ``error_message``)."""


class DownloadItem(BaseModel):
    """A queued (not yet started) download, labelled with the feature that needs it."""

    model_name: str
    feature: str
    """Human label for why this downloads (e.g. 'image model', 'LoRa', 'ControlNet annotators')."""
    target_dir: str | None = None
    """Where the file(s) will be written on disk."""
    size_bytes: int | None = None


FEATURE_SAFETY = "safety models"
"""The ``feature`` label (and synthetic model name) of the required-safety-model download.

Declared with the IPC types rather than in the download process because the parent matches on it to read
that download's progress and failure reason back out of a reported status."""

FEATURE_LORA_ADHOC = "LoRa (job)"
"""The ``feature`` label a job-driven ad-hoc LoRA prefetch download carries."""
FEATURE_TI_ADHOC = "textual inversion (job)"
"""The ``feature`` label a job-driven ad-hoc textual-inversion prefetch download carries."""
ADHOC_PREFETCH_FEATURES = frozenset({FEATURE_LORA_ADHOC, FEATURE_TI_ADHOC})
"""Download-feature labels for the job-driven ad-hoc prefetch pipeline (LoRA/TI placed on disk at job pop).

These downloads are how a LoRA job becomes dispatchable, so they are excluded from the worker-wide
LoRA-advertising suppression (which only bulk/default seeding and image/aux fetches should trigger)."""


class CurrentDownloadStatus(BaseModel):
    """The download in progress right now, with live progress."""

    model_name: str
    feature: str
    target_dir: str
    host: str | None = None
    """The source hostname this is downloading from (e.g. ``civitai.com``); None when not tracked."""
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float | None = None
    eta_seconds: float | None = None

    @property
    def percent(self) -> float | None:
        """Completion as a 0-100 percentage, or None when the total size is unknown."""
        if self.total_bytes <= 0:
            return None
        return min(100.0, self.downloaded_bytes / self.total_bytes * 100.0)


class DownloadFailure(BaseModel):
    """A download that was attempted and failed, with a human-readable reason."""

    model_name: str
    feature: str
    reason: str


class DownloadStatusSnapshot(BaseModel):
    """The full state of the download subsystem, for the TUI/console downloads view."""

    phase: DownloadPhase = DownloadPhase.IDLE
    current: CurrentDownloadStatus | None = None
    """The primary in-flight download (``active[0]`` when any), kept for back-compat with single-download
    consumers; prefer :attr:`active` for the full set when downloads run in parallel."""
    active: list[CurrentDownloadStatus] = Field(default_factory=list)
    """Every download in flight right now (one per executor thread); empty when idle. Parallel downloads
    target distinct hosts by default, so this typically lists one entry per active host."""
    pending: list[DownloadItem] = Field(default_factory=list)
    failures: list[DownloadFailure] = Field(default_factory=list)
    present_model_names: list[str] = Field(default_factory=list)
    paused: bool = False
    rate_limit_kbps: int | None = None
    error_message: str | None = None
    priority_policy: DownloadPriorityPolicy = DownloadPriorityPolicy.SERVE_FIRST
    """How the pending queue is ordered; switchable live from the Downloads tab."""
    startup_focus: bool = False
    """True while only the safety models and one image model at a time are admitted, so the worker can start
    serving as early as possible; the rest of the queue waits."""


class DownloadPlanSummary(BaseModel):
    """The disk implications of the current model config (computed once via model_download_plan)."""

    present_bytes: int = 0
    to_download_bytes: int = 0
    total_bytes: int = 0
    free_disk_bytes: int | None = None
    fits: bool = True
    shortfall_bytes: int = 0
    num_present: int = 0
    num_to_download: int = 0
    sizes_complete: bool = True
    """False when some configured models lack size metadata, so the byte totals are a lower bound."""


class FeatureInfoRow(BaseModel):
    """A read-only readiness line for a feature the worker does not gate on disk presence (LoRA, safety).

    These keep their own existing gating (per-job ad-hoc LoRA, the startup safety-model ensure); the row
    only surfaces their state alongside the gated features so the readiness table is a complete picture.
    """

    label: str
    status: str
    """A short human status, e.g. 'present', 'enabled', 'verifying downloads'."""
    ok: bool = True
    """Whether the line reads as healthy/ready (green) versus pending/blocked (muted), for styling."""


class FeatureReadinessSummary(BaseModel):
    """The worker's per-feature readiness: what it offers to the Horde and why anything is withheld.

    ``gated`` carries the features the worker withholds until their models/annotators are on disk
    (ControlNet, SDXL-ControlNet, post-processing); ``informational`` carries read-only lines for
    features with their own gating (LoRA, safety). Built parent-side from the same readiness the pop gate
    enforces, so the table can never disagree with what the worker actually advertises.
    """

    gated: list[FeatureReadiness] = Field(default_factory=list)
    informational: list[FeatureInfoRow] = Field(default_factory=list)


class WorkerFatalConfigError(BaseModel):
    """A one-shot frame the worker sends before exiting on a fatal, non-retryable config problem.

    Some misconfigurations can never succeed on a relaunch: a worker name taken by another account, a
    name still left at its reserved default, or the dreamer/alchemist names colliding. The worker
    detects these at startup (before spawning any children) and sends this frame so the supervisor stops
    burning its restart budget on a config that cannot work, and the dashboard shows the specific reason
    and remedy rather than a generic "crashed". ``detail`` carries the full human explanation, including
    how to fix it.
    """

    protocol_version: int = SUPERVISOR_PROTOCOL_VERSION
    title: str = "Worker configuration problem"
    """A short headline for the dashboard (e.g. "Worker name problem")."""
    detail: str = ""
    """The full operator-facing explanation and remedy."""


class WorkerLivenessFrame(BaseModel):
    """A tiny liveness heartbeat, emitted on its own cadence independent of full-snapshot production.

    The supervisor judges worker responsiveness from :attr:`loop_alive_wall_time` (the wall-clock time
    the worker's control loop last started a tick) rather than from the age of the last full
    :class:`WorkerStateSnapshot`. This decouples "is the loop making progress?" from "what is the rich
    state?": a snapshot that briefly fails to build (or is coalesced away) no longer looks like a stall,
    while a genuinely wedged loop reports an accurate, growing liveness age.
    """

    protocol_version: int = SUPERVISOR_PROTOCOL_VERSION
    loop_alive_wall_time: float = 0.0
    """Worker wall-clock time (``time.time()``) the control loop last began a tick."""


class WholeCardResidencyStatus(BaseModel):
    """Whole-card exclusive-residency posture: whether it can engage, and live detail when it has.

    Whole-card residency stops idle sibling inference processes (and may move the safety process off-GPU)
    to give a very heavy model sole use of the device, so its weights stay resident instead of streaming
    from host RAM. From the dashboard that looks like processes vanishing for no reason; these fields let
    the TUI explain it. ``possible`` is the config/topology heads-up (it *could* engage); the rest is the
    live detail of a residency currently held. MB figures are device VRAM rounded to whole MB.
    """

    possible: bool = False
    """The feature could engage under the current config and process topology (operator heads-up)."""
    enabled: bool = False
    """The ``whole_card_exclusive_residency`` config flag is on."""
    safety_off_gpu_enabled: bool = False
    """A whole-card job would also move the safety process off-GPU (config + safety on-GPU)."""
    cooldown_seconds: int = 0
    """Configured seconds a residency is held after its last heavy job drains, before restoring."""
    per_process_overhead_mb: int = 0
    """Per-process CUDA-context VRAM the forecast assumes (configured override, else startup-measured)."""
    total_vram_mb: int = 0
    """Device total VRAM (MB), 0 before any process has reported."""

    active: bool = False
    """A whole-card residency is currently held."""
    model: str | None = None
    """The model holding sole residency, when active."""
    phase: str = ""
    """``establishing`` (siblings stopping, model loading) or ``holding`` (serving) while active."""
    safety_paused: bool = False
    """The safety process is currently paused off-GPU for this residency."""
    processes_now: int = 0
    """Loaded inference processes right now (after any teardown)."""
    processes_target: int = 0
    """Inference processes the residency targets (the forecast's max-resident count)."""
    processes_max: int = 0
    """The normal inference-process ceiling, so the paused count is ``processes_max - processes_now``."""
    cooldown_remaining_seconds: float | None = None
    """Seconds left before the residency restores once its jobs drained, or None when not active."""

    pop_claim_model: str | None = None
    """The model a standing claim narrows the pop offer to, or None when the pool is unclaimed.

    A residency can be held without claiming the offer (a multi-card host, or a maximum hold of zero), so
    this is not simply ``model`` repeated: it is what the worker is currently asking the horde for."""
    pop_claim_remaining_seconds: float | None = None
    """Seconds until the maximum hold ends the standing claim, or None when no claim stands."""
    pop_claim_release: str | None = None
    """Why the last claim ended (``maximum_hold`` | ``no_further_work`` | ``residency_released``), while that
    is still recent; None otherwise."""

    weights_mb: int | None = None
    """Resident weight footprint of the residency model (detail view)."""
    reserve_mb: int | None = None
    """Free-VRAM headroom the forecast required, the activation working set (detail view)."""
    free_now_mb: int | None = None
    """Measured device-wide free VRAM at establishment (detail view)."""
    free_if_alone_mb: int | None = None
    """Free VRAM achievable with sole residency (detail view)."""
    max_resident_processes: int | None = None
    """The forecast's largest co-resident process count that still avoids streaming."""


class PopGovernorStatus(BaseModel):
    """One pop/scheduling governor's live spell plus session aggregates, for the dashboard and tooling.

    A governor is any condition that holds back or reshapes job pops (whole-card residency, the large-model
    switch throttle and re-entry cooldown, post-inference backpressure, the unservable-model holdback, the
    consecutive-failure pause, pop error-backoff, a LoRA download backoff, model stickiness, the megapixelstep
    wait, the self-throttle). These fields let the TUI show *which* governor is engaged and *for how long*,
    and let the stats tab compare how much of the session each one consumed.
    """

    name: str
    """Stable machine key (snake_case), e.g. ``large_model_switch``."""
    label: str
    """Short human-friendly name for the dashboard."""
    active: bool = False
    """Whether the governor is engaged right now."""
    reason: str | None = None
    """Short human-readable cause for the current engagement, when active."""
    current_spell_seconds: float = 0.0
    """How long the current spell has been engaged (0 when idle)."""
    expected_remaining_seconds: float | None = None
    """Estimated seconds until release, or None when the governor has no fixed timer."""
    triggers: int = 0
    """How many times this governor has engaged this session."""
    total_active_seconds: float = 0.0
    """Aggregate engaged time this session (completed spells plus the live one)."""
    fraction_of_session: float = 0.0
    """``total_active_seconds`` as a fraction of the session length so far (0..1)."""


class PopGovernorsSnapshot(BaseModel):
    """The set of pop/scheduling governors with history or a live spell, plus a roll-up flag."""

    governors: list[PopGovernorStatus] = Field(default_factory=list)
    """Per-governor live + aggregate state, active first (see ``PopGovernorRegistry.views``)."""
    any_active: bool = False
    """Whether any governor is currently engaged (a quick "is the worker being held back" flag)."""


class RamGovernanceSnapshot(BaseModel):
    """The RAM governor's current posture as operator-visible state.

    This is the projection of the scheduler's per-cycle host-memory snapshot after the pure governor
    policy has run: measured RAM vs the danger floor, whether intake is held, and which reclaim remedies
    are currently active. It is intentionally compact so the dashboard can explain why pops or preloads
    are being held without exposing the whole internal process map.
    """

    measured: bool = False
    """Whether the governor has produced at least one RAM verdict this session."""
    under_pressure: bool = False
    """Available RAM is below the absolute danger floor."""
    reason: str = ""
    """Short human explanation, usually the verdict's ``reason()`` string."""
    available_mb: int | None = None
    """Measured available system RAM (MB), or None when telemetry is unavailable."""
    floor_mb: int = 0
    """The active available-RAM danger floor (MB)."""
    total_mb: int | None = None
    """Total system RAM (MB), or None when unknown."""
    pop_hold_active: bool = False
    """The soft intake hold is active because RAM is pressured, near the floor, or reclaim is draining."""
    pop_pause_active: bool = False
    """The hard self-throttle pop pause is active."""
    pop_pause_remaining_seconds: float | None = None
    """Seconds until the hard pop pause lapses, when active."""
    draining_process_ids: list[int] = Field(default_factory=list)
    """Inference process ids currently draining so a RAM-heavy slot can be recycled."""
    shed_card_indices: list[int] = Field(default_factory=list)
    """GPU device indices whose inference context count was reduced for host-RAM pressure."""
    restore_headroom_mb: int = 0
    """Measured RAM headroom above reserves available for restoring a shed context."""
    per_context_ram_estimate_mb: int = 0
    """Estimated resident-RAM cost of restoring one inference context."""
    per_process_ceiling_mb: int | None = None
    """Configured resident-RAM ceiling for a single inference process, when enabled."""


class PreloadAdmissionSnapshot(BaseModel):
    """The most recent preload-admission decision the scheduler made for a queued image job."""

    decision: str = ""
    """Stable decision key from ``AdmissionDecision`` (e.g. ``admit``, ``defer_budget``)."""
    model: str | None = None
    """Model whose queued job was judged, if known."""
    process_id: int | None = None
    """Target process selected by the decision, when one was selected."""
    reason: str = ""
    """Short human explanation for the decision."""
    timestamp: float = 0.0
    """Worker wall-clock time when the decision was recorded; 0 means no decision yet."""


class SchedulingGovernanceSnapshot(BaseModel):
    """Operator-visible scheduler governance state for the Overview and Live views."""

    ram: RamGovernanceSnapshot = Field(default_factory=RamGovernanceSnapshot)
    """The RAM governor's latest measured posture and active remedies."""
    preload: PreloadAdmissionSnapshot = Field(default_factory=PreloadAdmissionSnapshot)
    """The latest preload-admission decision made while scanning the pending queue."""


_CARD_VRAM_PRESSURE_FLOOR_MB = 1024.0
"""Free VRAM below this (or below ~8% of the card) reads as VRAM pressure (a near-OOM heads-up)."""


class CardSnapshot(BaseModel):
    """A serializable per-card view of one GPU this worker drives, for the multi-GPU dashboard.

    One entry per driven card (a single-GPU host reports exactly one). Carries what an operator needs to
    judge a single card at a glance: its VRAM headroom, how many inference contexts it is running against its
    target, the whole-card residency it may be holding, and any models that have gone locally unservable on
    it. Per-card it/s and jobs/hr are derived on the receiving side (from the card's processes and successive
    ``jobs_completed`` deltas) so the math lives with the other throughput trends.
    """

    device_index: int
    """The stable (PCI-bus) index of this card."""
    device_name: str | None = None
    """The human GPU name (e.g. ``NVIDIA GeForce RTX 4090``), or None when the device map has no entry."""
    kind: str = "cuda"
    """The accelerator backend (``cuda``/``rocm``/``xpu``/...)."""

    total_vram_mb: float | None = None
    """This card's total VRAM (MB); None until a process reports and the device map lacks a capacity."""
    free_vram_mb: float | None = None
    """Measured free VRAM (MB) on this card; None until a process on it has reported memory."""

    loaded_contexts: int = 0
    """Inference processes on this card with an allocated device context right now."""
    busy_contexts: int = 0
    """This card's inference processes currently mid-inference (a duty proxy and 'card is active' signal)."""
    target_process_count: int = 0
    """How many inference processes this card aims to run (its queue/concurrency-derived target)."""
    max_concurrent_inference: int = 0
    """This card's concurrent-sampling ceiling."""

    jobs_completed: int = 0
    """Cumulative jobs this card has finished; the receiver derives jobs/hr from successive deltas."""

    residency_model: str | None = None
    """The model holding whole-card residency on this card, or None when none is held."""
    residency_phase: str = ""
    """``establishing`` / ``holding`` while a residency is active on this card, else empty."""

    unservable_models: list[str] = Field(default_factory=list)
    """Models currently treated as locally unservable on this card (its over-budget circuit-breaker tripped)."""
    worst_fault_streak: int = 0
    """The worst per-model consecutive over-budget fault streak on this card (0 when none)."""
    unserviceable_models: list[str] = Field(default_factory=list)
    """Configured models whose smallest legal job cannot fit this card; they are never offered."""
    constrained_models: dict[str, int] = Field(default_factory=dict)
    """Configured models this card can only run below the configured ``max_power``, mapped to the largest
    ``max_power`` that fits. Pops that include one of them ask for jobs no larger than that."""

    @property
    def vram_headroom_fraction(self) -> float | None:
        """Free VRAM as a fraction of this card's total (0.0-1.0), or None when either figure is unknown."""
        if self.free_vram_mb is None or not self.total_vram_mb:
            return None
        return max(0.0, min(self.free_vram_mb / self.total_vram_mb, 1.0))

    @property
    def is_vram_pressured(self) -> bool:
        """Whether this card is low on VRAM (a near-OOM heads-up): under a small floor or a small fraction.

        The fraction guard catches a large card with proportionally little left; the absolute floor catches
        a small card where even a healthy-looking fraction is too few MB to load another model safely.
        """
        if self.free_vram_mb is None:
            return False
        fraction = self.vram_headroom_fraction
        return self.free_vram_mb < _CARD_VRAM_PRESSURE_FLOOR_MB or (fraction is not None and fraction < 0.08)


class SystemMemorySnapshot(BaseModel):
    """The system-RAM picture and the worker's per-role share of it, for the supervisor/TUI.

    The wire projection of :class:`~horde_worker_regen.process_management.resources.system_memory.SystemMemorySummary`.
    Only the raw figures travel; the derived breakdown (used, other, fractions) is recomputed on the
    receiving side via :meth:`to_summary` so the math lives in exactly one place.
    """

    total_bytes: int = 0
    """Total physical RAM on the machine."""
    available_bytes: int = 0
    """RAM the OS reports as available without paging."""
    worker_rss_by_role: dict[str, int] = Field(default_factory=dict)
    """Per-role resident-set sizes (bytes) for the worker's own processes (see the ``ROLE_*`` keys)."""

    @classmethod
    def from_summary(cls, summary: SystemMemorySummary) -> SystemMemorySnapshot:
        """Project a worker-side :class:`SystemMemorySummary` onto the wire model."""
        return cls(
            total_bytes=summary.total_bytes,
            available_bytes=summary.available_bytes,
            worker_rss_by_role=dict(summary.worker_rss_by_role),
        )

    def to_summary(self) -> SystemMemorySummary:
        """Rebuild a :class:`SystemMemorySummary` (with its derived properties) from the wire fields."""
        from horde_worker_regen.process_management.resources.system_memory import build_system_memory_summary

        return build_system_memory_summary(
            total_bytes=self.total_bytes,
            available_bytes=self.available_bytes,
            worker_rss_by_role=self.worker_rss_by_role,
        )


class ModelPoolSeatReadiness(enum.StrEnum):
    """Measured readiness of the model logically assigned to a fixed-pool seat."""

    EMPTY = "EMPTY"
    """The seat has no active model."""
    RESIDENT = "RESIDENT"
    """At least one live inference process currently reports the seated model loaded."""
    COLD = "COLD"
    """The seat remains logically assigned, but no live inference process currently reports its model loaded."""


class ModelPoolSeatRow(BaseModel):
    """Represents one fixed-pool seat's operator-facing state, with monotonic stamps resolved to ages.

    The worker's seat engine keeps its timing in a monotonic clock the supervisor cannot interpret, so the
    stamps are converted to durations at snapshot time: ``dwell_seconds`` is how long the current model has
    held the seat, ``last_fulfilled_age_seconds`` how long since a pop last matched it, and
    ``rescue_expires_in_seconds`` how long a rescue seat has left before release. All are ``None`` when the
    seat is empty or the underlying stamp is unset.
    """

    model: str | None = None
    """The model currently holding the seat, or None when the seat is empty (a pending-only seat included)."""
    source: str | None = None
    """How the seat's model came to hold it (``SeatSource`` value ``MANUAL``/``RANKER``/``RESCUE``); None if empty."""
    state: str
    """Whether the seat is serving its model or mid-download (``SeatState`` value ``ACTIVE``/``PENDING_DOWNLOAD``)."""
    dwell_seconds: float | None = None
    """Seconds the current model has held the seat, or None when the seat is empty."""
    empty_pops: int = 0
    """Consecutive fixed-lane empty pops charged to the seated model since its last successful match."""
    last_fulfilled_age_seconds: float | None = None
    """Seconds since a pop last matched the seated model, or None when none has matched since seating."""
    last_match_was_resident: bool | None = None
    """Whether the seated model was already resident for its last matched pop; None before the first match."""
    readiness: ModelPoolSeatReadiness = ModelPoolSeatReadiness.EMPTY
    """Measured current readiness of the active model, distinct from the logical seat's ``state``."""
    resident_process_ids: list[int] = Field(default_factory=list)
    """Live inference process IDs currently reporting the active seated model loaded."""
    resident_device_indices: list[int] = Field(default_factory=list)
    """GPU device indices carrying those resident processes, deduplicated and sorted."""
    pending_model: str | None = None
    """The model this seat is downloading to swap in, or None when no download is pending."""
    rescue_expires_in_seconds: float | None = None
    """Seconds until a rescue seat's borrowed window closes, or None when the seat is not a rescue."""


class ModelPoolBenchRow(BaseModel):
    """Represents one benched model held out of seating until its cooldown elapses."""

    model: str
    reason: str
    """Why the model was demoted to the bench (``DemotionReason`` value, e.g. ``EMPTY_POPS``)."""
    cooldown_remaining_seconds: float
    """Seconds until the model may seat again (0 once the cooldown has lapsed)."""


class ModelPoolSnapshot(BaseModel):
    """Represents the fixed model pool's operator-facing state for the dashboard, or its disabled marker.

    Carried on :attr:`WorkerStateSnapshot.model_pool` only when the pool is enabled; a worker with the pool
    off leaves that field None so older supervisors and the plain (no-pool) case are unaffected. All
    monotonic timing is resolved to ages/countdowns at snapshot time, so no raw monotonic value travels.
    """

    enabled: bool = False
    """Whether the fixed pool is active (always True when this snapshot is present)."""
    seats: list[ModelPoolSeatRow] = Field(default_factory=list)
    """Every seat in seat order, holding a committed model or empty."""
    bench: list[ModelPoolBenchRow] = Field(default_factory=list)
    """Models cooling down out of seating after a demotion."""
    current_lane: str | None = None
    """The advertising lane the most recent pool-routed pop used (``PopLane`` value ``FIXED``/``FREE``), or
    None when the pool has not routed a pop yet."""
    last_fixed_seat_count: int = 0
    """The seated-model count advertised on the most recent fixed-lane pop."""
    demand_age_seconds: float | None = None
    """Seconds since the pool's demand ranking was last refreshed, or None before the first reading."""
    download_budget_gb: float = 0.0
    """Configured session admission budget (declared-size GB) for pool-initiated downloads (0 = never)."""
    download_bytes_charged: int = 0
    """Reference-declared bytes charged when pool download requests started during this session."""
    fixed_pops: int = 0
    """Session-cumulative pops the pool advertised on the fixed (seated-model) lane."""
    fixed_fulfilled: int = 0
    """How many of ``fixed_pops`` returned a job (the fixed lane's hit rate is this over ``fixed_pops``)."""
    fixed_resident_hits: int = 0
    """Fixed-lane matches whose returned model was already resident at pop time."""
    free_pops: int = 0
    """Session-cumulative pops the pool advertised on the free (unseated-model) lane."""
    free_fulfilled: int = 0
    """How many of ``free_pops`` returned a job (the free lane's hit rate is this over ``free_pops``)."""
    free_resident_hits: int = 0
    """Free-lane matches whose returned model was already resident at pop time."""


class WorkerStateSnapshot(BaseModel):
    """One frame of worker state pushed from the worker to its supervisor over the pipe.

    Carries the same headline information the console ``StatusReporter`` assembles, plus the
    per-process detail the live view renders. Payload size is bounded: ``recent_jobs`` is capped
    at :data:`RECENT_JOBS_IN_SNAPSHOT` and ``recent_events`` at :data:`RECENT_EVENTS_IN_SNAPSHOT`.
    """

    protocol_version: int = SUPERVISOR_PROTOCOL_VERSION
    timestamp: float = Field(default_factory=time.time)
    session_start_time: float = 0.0

    shutting_down: bool = False
    maintenance_mode: bool = False
    """Maintenance: the local pop loop hit maintenance, the operator paused the worker locally, or the
    worker self-throttled (see ``self_throttle_paused``)."""
    self_throttle_paused: bool = False
    """The worker paused popping itself: resource/OOM faults accumulated fast enough that it backed off to
    avoid the horde forcing maintenance for "dropping too many jobs"."""
    supervisor_paused: bool = False
    """The worker is locally paused by operator command (F2 / PAUSE); distinct from any server-side state."""
    last_pop_maintenance_mode: bool = False
    """The most recent pop response returned a maintenance-mode error (cleared on the next successful pop)."""
    worker_details_maintenance: bool = False
    """The horde's worker-details API reports this worker in maintenance (polled, advisory)."""
    worker_details_paused: bool = False
    """The horde's worker-details API reports this worker paused (polled, advisory)."""
    too_many_consecutive_failed_jobs: bool = False

    gpu_torch_incompatible: bool = False
    """The installed PyTorch has no CUDA kernels for this GPU's architecture, so the worker stopped popping.

    Reported by a torch-bearing inference child at startup (the parent and TUI never import torch). A
    build/hardware mismatch: the wheel was compiled for a different set of GPU architectures than the
    installed card. Sticky for the session; fixed by reinstalling the matching backend and restarting."""
    gpu_torch_incompatible_reason: str | None = None
    """Operator-facing detail for ``gpu_torch_incompatible`` (device + remedy); None when not tripped."""

    torch_build_cpu_only: bool = False
    """The installed PyTorch is a CPU-only build, so image generation is disabled (alchemy still runs).

    Reported by a torch-bearing inference child at startup (the parent and TUI never import torch). The
    runtime counterpart of a ``bin/backend`` 'cpu' sentinel: it makes a CPU torch build serve alchemy-only
    even when the sentinel was never set (e.g. a manual CPU install). Sticky for the session; fixed by
    installing a GPU build and restarting."""
    torch_build_cpu_only_reason: str | None = None
    """Operator-facing detail for ``torch_build_cpu_only`` (why image gen is off + remedy); None when not tripped."""

    post_processing_disabled: bool = False
    """Post-processing is session-disabled and no longer advertised to the Horde."""
    post_processing_disabled_reason: str | None = None
    """Operator-facing detail for why post-processing was disabled; None when not tripped."""

    # Connectivity / health signals the worker already tracks (surfaced for the status monitor).
    worker_registered: bool = False
    """Whether the AI Horde API has returned this worker's details at least once (it is known)."""
    user_info_failed: bool = False
    """The most recent user-details API call failed; the clearest API/network-reachability signal."""
    user_info_failed_reason: str | None = None
    """A short description of the last user-info failure, when one occurred."""
    in_error_backoff: bool = False
    """The job-pop throttler is backing off after repeated pop failures (server/network trouble)."""
    consecutive_failed_jobs: int = 0
    """How many jobs have failed back-to-back (resets on a success)."""
    seconds_since_last_pop: float | None = None
    """Seconds since the worker last successfully popped a job (None if it never has)."""
    last_pop_no_jobs_available: bool = False
    """The most recent successful pop returned no job (a short-term 'no work right now' signal)."""
    last_pop_skipped_reasons: dict[str, int] = Field(default_factory=dict)
    """Why the last 'no job available' pop skipped work, per reason (models/nsfw/max_pixels/...)."""
    last_reduced_pop_skipped_reasons: dict[str, int] = Field(default_factory=dict)
    """Why the last reduced-size (constrained lane) 'no job available' pop skipped work, per reason."""
    last_reduced_pop_max_power: int | None = None
    """The ``max_power`` that reduced-size pop asked with, or None when no such pop is outstanding."""
    api_messages: list[str] = Field(default_factory=list)
    """Operator/maintenance messages delivered by the horde in pop responses."""

    config: WorkerConfigSummary
    processes: list[ProcessSnapshot] = Field(default_factory=list)

    # Headline job counters (from the job tracker / submitter / popper).
    num_jobs_popped: int = 0
    num_jobs_submitted: int = 0
    num_jobs_faulted: int = 0
    num_job_slowdowns: int = 0
    num_process_recoveries: int = 0
    pending_megapixelsteps: int = 0
    jobs_pending_inference: int = 0
    jobs_in_progress: int = 0
    jobs_pending_safety_check: int = 0
    jobs_being_safety_checked: int = 0
    jobs_pending_post_processing: int = 0
    jobs_being_post_processed: int = 0
    jobs_pending_submit: int = 0
    """Jobs that have cleared safety and are queued for API submission (the pipeline's tail stage)."""
    time_spent_no_jobs_available: float = 0.0

    kudos_per_hour: float | None = None
    kudos_this_session: float | None = None
    eligible_seconds_total: float = 0.0
    """Cumulative productive (pipeline-occupied) seconds since the first submit; the kudos/hr denominator."""

    active_models: list[str] = Field(default_factory=list)

    gpu_utilization_mean_percent: float | None = None
    """Worker-wide duty cycle: the mean of the per-card means below, so cards count equally."""
    gpu_utilization_busy_fraction: float | None = None
    """Worker-wide busy fraction, reduced across the driven cards the same way."""
    gpu_utilization_samples: int = 0
    """How many GPU-utilization samples back the figures above, summed across cards (0 = unmeasured)."""

    gpu_utilization_mean_percent_per_card: dict[int, float] = Field(default_factory=dict)
    """Per-card duty cycle keyed by device index; cards with no samples in the window are absent."""
    gpu_utilization_busy_fraction_per_card: dict[int, float] = Field(default_factory=dict)
    """Per-card busy fraction keyed by device index; cards with no samples in the window are absent."""
    gpu_utilization_samples_per_card: dict[int, int] = Field(default_factory=dict)
    """Per-card retained sample counts keyed by device index (0 for a card with no telemetry)."""

    vram_high_water_mb_per_process: dict[int, int] = Field(default_factory=dict)
    ram_high_water_mb_per_process: dict[int, int] = Field(default_factory=dict)
    disk_free_bytes: dict[str, int] = Field(default_factory=dict)

    system_memory: SystemMemorySnapshot | None = None
    """Live system-RAM total/available and the worker's per-role RSS share (None before first sample)."""

    downloads: DownloadStatusSnapshot | None = None
    """Live download-subsystem state (None when background downloads are disabled)."""
    download_plan: DownloadPlanSummary | None = None
    """The one-time disk implications of the configured models (None when not computed)."""
    feature_readiness: FeatureReadinessSummary | None = None
    """Per-feature deps+on-disk readiness driving which gated features the worker offers (None until built)."""
    lora_pops_blocked_by_downloads: bool = False
    """Configured LoRA support is temporarily suppressed because background downloads are active."""
    lora_pops_blocked_by_disk: bool = False
    """Configured LoRA support is suppressed because the LoRA cache volume is below its free-space floor.

    Unlike the transient download block, this persists until disk space recovers; the TUI surfaces it
    prominently because, left unaddressed, it stops the worker from serving any LoRA jobs."""

    recent_jobs: list[RecentJobRecord] = Field(default_factory=list)
    """The most recent finished-job records, newest last (capped)."""
    recent_events: list[WorkerEvent] = Field(default_factory=list)
    """The most recent worker transitions, newest last, bounded by :data:`RECENT_EVENTS_IN_SNAPSHOT`."""

    latest_stats_sample: StatsSample | None = None
    """The latest one-second worker-owned stats sample."""
    stats_model_rollups: list[StatsRollupRow] = Field(default_factory=list)
    """Finalized image-job rollups by model."""
    stats_baseline_rollups: list[StatsRollupRow] = Field(default_factory=list)
    """Finalized image-job rollups by baseline."""
    stats_form_rollups: list[StatsRollupRow] = Field(default_factory=list)
    """Finalized alchemy-form rollups by form (``model`` carries the form name); empty for a dreamer worker."""
    stats_export: StatsExportState = Field(default_factory=StatsExportState)
    """Worker-side stats JSONL export state."""
    stats_history_backfill: StatsHistoryBackfill | None = None
    """Bounded stats history for reconnecting frontends."""

    alchemy_forms_pending: int = 0
    """Forms popped from the API but not yet dispatched to a child process."""
    alchemy_forms_in_flight: int = 0
    """Forms dispatched to a child process, awaiting a result message."""
    alchemy_forms_awaiting_submit: int = 0
    """Forms with a completed result, waiting for API submission."""
    alchemy_total_submitted: int = 0
    """Cumulative forms successfully submitted this session."""
    alchemy_total_faulted: int = 0
    """Cumulative forms that faulted (permanently failed) this session."""

    text_jobs_in_flight: int = 0
    """Text jobs popped, generating, or awaiting submission.

    Text generation has no queued or dispatched stage to report beside this: its generations run in a
    separate program the worker reaches over HTTP, so a popped job goes straight to that program."""
    text_total_submitted: int = 0
    """Cumulative text jobs successfully submitted this session."""
    text_total_faulted: int = 0
    """Cumulative text jobs submitted as faulted this session."""
    text_backend_ready: bool = False
    """Whether the text backend has reported a loaded model and described itself.

    False while the readiness gate is open, which a cold start legitimately holds for minutes, and again
    after a backend failure sends the flow back to the gate. Nothing is popped while it is False."""
    text_model_name: str | None = None
    """The text model as the worker advertises it, prefix included; None before the backend has answered."""
    text_context_length: int | None = None
    """The largest prompt-plus-generation token count offered: the lower of the operator's cap and the
    backend's. None before the backend has answered."""
    text_max_length: int | None = None
    """The largest generation length offered, on the same basis as :attr:`text_context_length`."""
    text_backend_not_ready_since: float | None = None
    """When the readiness gate last opened, or None while the backend is ready or never had a flow.

    A gate that has been open for a long time is the difference between a backend still loading weights
    and one the operator never started, and a boolean cannot tell them apart."""

    workload_totals: dict[WorkloadKind, WorkloadTotalsSnapshot] = Field(default_factory=dict)
    """This session's completed/faulted/kudos totals per workload, derived from the finished-job records.

    Workloads that finished nothing are absent rather than present with zeroes, so a dreamer-only worker
    carries one entry."""

    snapshot_interval_seconds: float = 1.0
    """The floor cadence at which the worker publishes snapshots when nothing else changed.

    Carried so a dashboard check about how long something has been true can allow for the worker's own
    reporting delay without restating the cadence."""

    enabled_workloads: list[str] = Field(default_factory=list)
    """The workloads this worker serves, as ``WorkloadKind`` values (e.g. ``image_generation``,
    ``alchemy``), sorted for stable rendering.

    Carried as plain strings so this module stays free of the heavy import chain behind ``WorkloadKind``;
    consumers reconstruct the typed enum. The dashboard uses this to identify the worker's mode (an
    alchemist-only worker reshapes around alchemy) rather than inferring it from model counts. Empty only
    before the first snapshot or for a worker configured to serve nothing."""

    pending_jobs: list[JobQueueEntry] = Field(default_factory=list)
    """Pending-inference jobs (capped at :data:`PENDING_JOBS_IN_SNAPSHOT`), oldest first."""

    orchestration_intent: OrchestrationIntentSnapshot = Field(default_factory=OrchestrationIntentSnapshot)
    """Current plain-English scheduler intent for the Overview Now/Next/Why strip."""

    work_ledger: list[WorkLedgerEntry] = Field(default_factory=list)
    """Active and recent job rows for the Overview work ledger."""

    disagg_job_stages: list[DisaggStageRow] = Field(default_factory=list)
    """Per-job pipeline-disaggregation stage and dispatch target for in-flight disaggregated jobs.

    Empty when disaggregation is off or no disaggregated job is in flight."""

    whole_card_residency: WholeCardResidencyStatus = Field(default_factory=WholeCardResidencyStatus)
    """Whole-card exclusive-residency posture: whether it can engage, and live detail when it has."""

    pop_governors: PopGovernorsSnapshot = Field(default_factory=PopGovernorsSnapshot)
    """The pop/scheduling governors holding back or reshaping job pops, with live + session-aggregate state."""

    scheduling_governance: SchedulingGovernanceSnapshot = Field(default_factory=SchedulingGovernanceSnapshot)
    """RAM-governor posture plus the latest preload-admission decision for operator diagnostics."""

    per_card: list[CardSnapshot] = Field(default_factory=list)
    """Per-card multi-GPU view, one entry per driven card (exactly one on a single-GPU host)."""

    model_pool: ModelPoolSnapshot | None = None
    """Fixed model pool seats/bench/lane/demand-age/budget, or None when the pool is disabled."""


class SupervisorCommand(enum.Enum):
    """A control action the supervisor asks the worker to take, drained each loop tick."""

    PAUSE = enum.auto()
    """Enter maintenance mode: stop popping new jobs (in-flight jobs finish)."""
    RESUME = enum.auto()
    """Leave maintenance mode and resume popping jobs."""
    DRAIN = enum.auto()
    """Stop popping and let in-flight jobs finish without exiting (alias of PAUSE for now)."""
    RESTART_PROCESS = enum.auto()
    """Replace one process by ``process_id``: an inference slot (e.g. a stuck slot) or a service lane.

    Targeting a service lane (COMPONENT/VAE_LANE/POST_PROCESS/UTILITIES) recycles it through its normal
    respawn machine, the sanctioned way to reset a lane whose host commit charge has ballooned past what an
    in-process model unload can return. The safety and download processes are not restartable this way."""
    RELOAD_CONFIG = enum.auto()
    """Re-read ``bridgeData.yaml`` from disk and hot-swap the runtime config."""
    SET_CONCURRENCY = enum.auto()
    """Adjust the live inference concurrency: thread cap and/or running process count."""
    PAUSE_DOWNLOADS = enum.auto()
    """Hold background model downloads (the current chunk loop blocks) until resumed."""
    RESUME_DOWNLOADS = enum.auto()
    """Resume held background model downloads."""
    SET_DOWNLOAD_RATE_LIMIT = enum.auto()
    """Set the background-download bandwidth cap in KB/s (0 or None clears the cap)."""
    SET_DOWNLOAD_PRIORITY_POLICY = enum.auto()
    """Set how the pending download queue is ordered (serve-first tiers or first come, first served)."""
    DOWNLOADS_ONLY_HOLD = enum.auto()
    """Hold the worker in a download-only posture: keep the download process (and reference refresh)
    running but do not start inference/safety or pop jobs. Lets the operator pre-fetch models without
    committing the GPU. Lifted by :attr:`GO_LIVE`."""
    GO_LIVE = enum.auto()
    """Leave the download-only hold and bring the worker fully up (inference/safety start once a model is
    present, popping resumes). In-flight downloads continue; the present-set gate keeps serving safe."""
    DOWNLOAD_MODELS = enum.auto()
    """Fetch a chosen set of models on demand: the selected image models (and optionally the aux pass),
    enqueued into the background download process without changing config. Drives the TUI download picker."""
    SET_SERVER_MAINTENANCE = enum.auto()
    """Set the worker's *server-side* maintenance flag on the horde (``server_maintenance_enabled``).

    Distinct from :attr:`PAUSE`/:attr:`RESUME` (the local pop-pause): this calls the horde API so the
    horde itself stops (or resumes) sending the worker jobs, matching the maintenance the job-pop response
    reports."""
    SHUTDOWN = enum.auto()
    """Begin a graceful, timed shutdown of the worker."""
    SET_STATS_EXPORT = enum.auto()
    """Enable or disable worker-side stats JSONL export for this session."""


class SupervisorControlMessage(BaseModel):
    """A command sent from the supervisor to the worker over the pipe."""

    command: SupervisorCommand
    process_id: int | None = None
    """The target process slot (an inference slot or a service-lane id), required for \
:attr:`SupervisorCommand.RESTART_PROCESS`."""
    target_threads: int | None = None
    """The new concurrent-inference cap, for :attr:`SupervisorCommand.SET_CONCURRENCY` (clamped to \
    the session ceiling)."""
    target_processes: int | None = None
    """The new running inference-process count, for :attr:`SupervisorCommand.SET_CONCURRENCY`."""
    download_rate_limit_kbps: int | None = None
    """The new download bandwidth cap in KB/s, for :attr:`SupervisorCommand.SET_DOWNLOAD_RATE_LIMIT`."""
    download_priority_policy: DownloadPriorityPolicy | None = None
    """The new download queue order, for :attr:`SupervisorCommand.SET_DOWNLOAD_PRIORITY_POLICY`."""
    server_maintenance_enabled: bool | None = None
    """The desired server-side maintenance state, for :attr:`SupervisorCommand.SET_SERVER_MAINTENANCE`."""
    download_model_names: list[str] = Field(default_factory=list)
    """The image models to fetch on demand, for :attr:`SupervisorCommand.DOWNLOAD_MODELS`."""
    download_include_aux: bool = False
    """Whether a :attr:`SupervisorCommand.DOWNLOAD_MODELS` request should also run the aux/default pass."""
    stats_export_enabled: bool | None = None
    """Desired stats JSONL export state for :attr:`SupervisorCommand.SET_STATS_EXPORT`."""


class SupervisorChannel:
    """The worker's end of the supervisor pipe, designed so the consumer can never stall the worker.

    Snapshots are sent on a daemon thread from a single latest-only slot: :meth:`send_snapshot` only
    updates the slot and returns immediately, so the worker's control loop never blocks on a slow or
    hung supervisor (it just sends the freshest state once the consumer catches up). :meth:`drain_commands`
    is non-blocking and skips a tick rather than wait if the sender thread is mid-send. A dead pipe is
    caught and marks the channel closed.

    Send and receive on the one duplex connection are serialized by a lock, so the sender thread and
    the control loop never touch the connection concurrently.
    """

    _LIVENESS_INTERVAL = 1.0
    """How often the sender thread emits a :class:`WorkerLivenessFrame` (seconds)."""

    def __init__(self, connection: Connection) -> None:
        """Wrap the worker-side pipe connection and start the snapshot sender thread."""
        self._connection = connection
        self._lock = threading.Lock()
        self._closed = False
        self._latest: WorkerStateSnapshot | None = None
        self._latest_unsent = False
        """Whether :attr:`_latest` still has to go out. Distinct from :attr:`_pending`, which is also set
        by :meth:`close` to wake the sender, so it alone cannot tell a frame waiting to be sent from one
        already on the wire."""
        self._pending = threading.Event()
        self._stop = threading.Event()
        self._loop_alive_wall_time = time.time()
        """Updated by :meth:`note_alive` from the control loop; read by the sender thread. A single
        float shared one-writer/one-reader (atomic under the GIL), so no lock is needed; a stale read
        only means a slightly older liveness timestamp, which is harmless."""
        self._last_liveness_monotonic = 0.0
        self._sender = threading.Thread(
            target=self._send_loop,
            name="supervisor-snapshot-sender",
            daemon=True,
        )
        self._sender.start()

    def note_alive(self) -> None:
        """Record that the control loop just advanced (call once per tick). Never blocks."""
        self._loop_alive_wall_time = time.time()

    def send_snapshot(self, snapshot: WorkerStateSnapshot) -> bool:
        """Hand the latest snapshot to the sender thread. Never blocks; returns False once closed."""
        if self._closed:
            return False
        self._latest = snapshot
        self._latest_unsent = True
        self._pending.set()
        return True

    def _send_loop(self) -> None:
        """Send the freshest snapshot when one is pending and emit liveness frames on their own cadence.

        Either send may block on a slow consumer; that is fine, because this runs on a daemon thread and
        never touches the worker's control loop.

        A snapshot handed over just before :meth:`close` is flushed on the way out rather than discarded
        with the loop: the frame arriving in that window is the worker's last one, carrying the
        ``shutting_down`` state the supervisor needs to tell a deliberate exit from a worker gone silent.
        The flush is one bounded attempt and shares the ordinary best-effort send, so a pipe that has
        already died costs nothing.
        """
        while not self._stop.is_set():
            got_pending = self._pending.wait(timeout=self._LIVENESS_INTERVAL)

            now = time.monotonic()
            if now - self._last_liveness_monotonic >= self._LIVENESS_INTERVAL:
                self._last_liveness_monotonic = now
                if not self._send_frame(WorkerLivenessFrame(loop_alive_wall_time=self._loop_alive_wall_time)):
                    return

            if not got_pending:
                continue
            if not self._send_pending_snapshot():
                return

        self._send_pending_snapshot()

    def _send_pending_snapshot(self) -> bool:
        """Send the freshest snapshot the sender has accepted, if any. False once the transport is dead.

        The unsent flag is cleared before the frame is read: a producer racing this method then leaves the
        flag set for a frame that just went out, costing one redundant resend, whereas the reverse order
        would mark a frame sent that never was and drop it.
        """
        self._pending.clear()
        if not self._latest_unsent:
            return True
        self._latest_unsent = False
        snapshot = self._latest
        if snapshot is None:
            return True
        return self._send_frame(snapshot)

    def _send_frame(self, frame: WorkerStateSnapshot | WorkerLivenessFrame) -> bool:
        """Send one frame under the connection lock. Returns False once the channel is closed/dead."""
        with self._lock:
            if self._closed:
                return False
            try:
                self._connection.send(frame)
            except Exception:
                self._closed = True
                return False
        return True

    def drain_commands(self) -> list[SupervisorControlMessage]:
        """Return all control messages currently waiting, without blocking (skips if a send is in progress)."""
        if self._closed:
            return []
        commands: list[SupervisorControlMessage] = []
        if not self._lock.acquire(blocking=False):
            return commands
        try:
            while self._connection.poll():
                message = self._connection.recv()
                if isinstance(message, SupervisorControlMessage):
                    commands.append(message)
        except (EOFError, OSError):
            self._closed = True
        finally:
            self._lock.release()
        return commands

    def close(self) -> None:
        """Stop the sender thread (the connection itself is owned by the caller)."""
        self._stop.set()
        self._pending.set()

    @property
    def closed(self) -> bool:
        """Whether the channel has encountered an unrecoverable pipe error."""
        return self._closed
