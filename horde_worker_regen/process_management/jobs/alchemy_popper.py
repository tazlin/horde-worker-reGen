"""Pops, dispatches, and submits alchemy (interrogation) jobs from the AI Horde API.

Admission is two layers. The *fairness* layer decides whether alchemy may take a contended
resource at all: image jobs always take priority for process time, so in concurrent mode
(``alchemy_allow_concurrent``) a graph form is only popped into a process lane no waiting image
job needs, and with concurrency off alchemy reverts to strict backfill (pop only when the image
queue is empty). Beneath it, the *capacity* layer decides whether the device can physically hold
another form: alchemy shares the same :class:`CommittedReserveLedger` as image generation, so a
form is admitted only when *effective* free VRAM/RAM (measured free minus what in-flight image and
alchemy work has already committed) covers the form's predicted cost. The two flows therefore
cannot independently admit against the same free VRAM the way two separate gates once could.

Graph forms (upscalers, facefixers, strip_background) are dispatched to the post-processing lane;
text-output forms (caption, interrogation, nsfw, vectorize) to the safety process. Dispatch is keyed
on :class:`WorkerCapability`, not process type.
"""

from __future__ import annotations

import asyncio
import base64
import statistics
import time
from asyncio import CancelledError
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import aiohttp
import psutil
import yarl
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import (
    AlchemyJobPopResponse,
    AlchemyJobSubmitRequest,
    AlchemyPopRequest,
)
from horde_sdk.ai_horde_api.apimodels.alchemy.submit import AlchemyJobSubmitResponse
from horde_sdk.generation_parameters.alchemy.consts import (
    KNOWN_ANNOTATION_CONTROL_TYPES,
    KNOWN_FACEFIXERS,
    KNOWN_MISC_POST_PROCESSORS,
    KNOWN_UPSCALERS,
    is_annotation_form,
    is_strip_background_form,
)
from loguru import logger

from horde_worker_regen.capabilities import describe_available, strip_background_available, vectorize_available
from horde_worker_regen.consts import (
    AESTHETIC_FORM_NAME,
    DESCRIBE_FORM_NAME,
    PALETTE_FORM_NAME,
    VECTORIZE_FORM_NAME,
    WORKER_KNOWN_BETA_FACEFIXERS,
    WORKER_KNOWN_BETA_UPSCALERS,
)
from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.api_sessions import ApiSessions
from horde_worker_regen.process_management.ipc.messages import (
    AlchemyFormSpec,
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeControlFlag,
)
from horde_worker_regen.process_management.jobs.job_models import PendingAlchemySubmitJob
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import WorkerCapability
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind, capability_for_alchemy_form
from horde_worker_regen.process_management.simulation._canned_scenarios import CannedAlchemySource
from horde_worker_regen.runtime_version import runtime_version
from horde_worker_regen.server_capabilities import server_supports_interrogation_form

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.lifecycle.shutdown_manager import ShutdownManager
    from horde_worker_regen.process_management.resources.run_metrics import WorkerRunMetrics


class _AlchemyPopRequest(AlchemyPopRequest):
    """AlchemyPopRequest with `forms` relaxed to plain strings.

    The worker offers beta post-processors (upscalers, face-fixers) by name ahead of the SDK release
    that lists them in `KNOWN_ALCHEMY_TYPES`, so `forms` must accept plain strings rather than round-trip
    through the published enum. The server validates the names, and unknown ones are withheld until it
    advertises them (see :func:`expand_offered_forms`).
    """

    forms: list[str]  # type: ignore[assignment]


class _AlchemyPopResponse(AlchemyJobPopResponse):
    """AlchemyJobPopResponse opted out of the SDK session's failure-cleanup tracking.

    The base class advertises `AlchemyJobSubmitRequest` as its failure-cleanup request but
    supplies only ``{"id": ...}`` as construction params, so the SDK client session raises a
    pydantic ValidationError (missing `result`/`state`) building the cleanup request, after
    the pop already succeeded server-side, silently dropping the popped forms. The
    coordinator submits faulted states itself, so the SDK's cleanup tracking is unwanted
    here regardless.
    """

    @override
    def ignore_failure(self) -> bool:
        """Opt out of cleanup tracking; the coordinator owns fault submission."""
        return True


class _AlchemySubmitRequest(AlchemyJobSubmitRequest):
    """AlchemyJobSubmitRequest with `result` as a dict.

    The wire format (matching the legacy alchemist and the live API) is a JSON object like
    ``{"caption": "..."}`` or ``{"RealESRGAN_x4plus": "R2"}``, despite the SDK's `str`
    annotation (which carries a FIXME).
    """

    result: dict[str, object]  # type: ignore[assignment]


@dataclass(frozen=True)
class AlchemyFormStatus:
    """One active alchemy form projected for the dashboard's work-ledger and queue tables.

    Alchemy forms are the alchemist worker's unit of work, the analogue of an image job. This carries the
    fields those tables render: the form name (shown where an image job shows its model), the source image
    resolution (shown where an image job shows its W×H size), the lifecycle stage, and the process the form
    was dispatched to (when in flight).
    """

    form_id: str
    form: str
    stage: str
    """One of ``pending`` (popped, not yet dispatched), ``in_flight`` (dispatched, awaiting a result), or
    ``awaiting_submit`` (result in hand, queued for API submission)."""
    width: int | None
    height: int | None
    process_id: int | None


def required_capability(form: str) -> WorkerCapability:
    """Return the capability a process must declare to serve the given form.

    Thin alias over :func:`workload_flow.capability_for_alchemy_form`, which is the single source of truth
    for form-to-capability routing.
    """
    return capability_for_alchemy_form(form)


DEFAULT_ALCHEMY_FORMS: tuple[str, ...] = (
    "caption",
    "nsfw",
    "interrogation",
    "post-process",
    VECTORIZE_FORM_NAME,
    PALETTE_FORM_NAME,
    DESCRIBE_FORM_NAME,
    AESTHETIC_FORM_NAME,
    "annotation",
)
"""Forms an alchemist offers when ``bridge_data.forms`` is left unset (an empty list means "all").

The SDK's ``default_forms`` validator does not fire for the default empty list, so both the dispatch
path and the dashboard projection fall back to this set. The yaml/legacy spelling is ``post-process``;
the SDK enum value is ``post_process``.
"""


def expand_offered_forms(
    bridge_data: reGenBridgeData,
    *,
    utilities_lane_healthy: bool = True,
    annotation_types: frozenset[str] = frozenset(),
) -> list[str]:
    """Expand the bridge-data `forms` config into the individual form names the API expects.

    "post-process" expands to every known upscaler, facefixer, and strip_background,
    mirroring the legacy alchemist. Caption requires the explicit BLIP opt-in.

    ``strip_background`` runs only on the out-of-venv image-utilities lane and has no in-graph fallback,
    so it is offered only when that lane is both provisioned (:func:`strip_background_available`) and
    currently up (``utilities_lane_healthy``). The pure-torch upscalers and face-fixers are unaffected by
    the lane's health. ``utilities_lane_healthy`` defaults True for the config-only callers that cannot see
    runtime process state; the pop path passes the live reading.
    """
    offered: list[str] = []
    configured_forms = bridge_data.forms or list(DEFAULT_ALCHEMY_FORMS)
    configured = {"post-process" if str(form) == "post_process" else str(form) for form in configured_forms}

    if "caption" in configured and bridge_data.alchemy_caption_enabled:
        offered.append("caption")
    if "interrogation" in configured:
        offered.append("interrogation")
    if "nsfw" in configured:
        offered.append("nsfw")
    # vectorize needs both vtracer (the worker-only `vectorize` extra) AND a server that lists the
    # form: a lean install would fault on it, and a server that does not yet support it rejects the
    # whole pop. The server gate is fail-closed until probed, so the worker can ship ahead of the
    # server's go-live and only begins offering the form once the server advertises it.
    if (
        VECTORIZE_FORM_NAME in configured
        and vectorize_available()
        and server_supports_interrogation_form(VECTORIZE_FORM_NAME)
    ):
        offered.append(VECTORIZE_FORM_NAME)
    # palette is pure-Pillow (no optional dep), so it gates only on the server advertising the form;
    # the gate is fail-closed until probed, so the worker ships ahead of the server's go-live.
    if PALETTE_FORM_NAME in configured and server_supports_interrogation_form(PALETTE_FORM_NAME):
        offered.append(PALETTE_FORM_NAME)
    # describe needs the worker-only `describe` extra (blurhash/imagehash) AND a server that lists the
    # form, mirroring vectorize: a lean install faults on it, an unaware server rejects the whole pop.
    if (
        DESCRIBE_FORM_NAME in configured
        and describe_available()
        and server_supports_interrogation_form(DESCRIBE_FORM_NAME)
    ):
        offered.append(DESCRIBE_FORM_NAME)
    # aesthetic is a CLIP-stack form (safety process), always runnable when a safety process is up, so
    # like palette it gates only on the server advertising the form (fail-closed until probed). Its
    # one-time predictor-weight download happens lazily in the safety process on first use.
    if AESTHETIC_FORM_NAME in configured and server_supports_interrogation_form(AESTHETIC_FORM_NAME):
        offered.append(AESTHETIC_FORM_NAME)
    if (
        "annotation" in configured
        and utilities_lane_healthy
        and annotation_types
        and server_supports_interrogation_form("annotation")
    ):
        offered.append("annotation")
    if "post-process" in configured:
        # Newly-added (beta) upscalers are withheld until the server lists them: it rejects the whole
        # pop if offered an unknown post-processor. The gate is fail-closed until probed, so the worker
        # ships ahead of go-live and begins offering them within the probe TTL once the server catches
        # up. The long-standing upscalers are in every server's enum and are never gated. The beta names
        # are also offered straight from the worker-known set so the worker does not have to wait on an
        # SDK release that lists them in KNOWN_UPSCALERS (the pop/submit wire models accept unknown
        # upscaler names as plain strings; hordelib classifies them via its own SDK).
        sdk_upscalers = [m.value for m in KNOWN_UPSCALERS if m != KNOWN_UPSCALERS.BACKEND_DEFAULT]
        extra_beta_upscalers = [u for u in sorted(WORKER_KNOWN_BETA_UPSCALERS) if u not in sdk_upscalers]
        for upscaler_value in (*sdk_upscalers, *extra_beta_upscalers):
            if upscaler_value in WORKER_KNOWN_BETA_UPSCALERS and not server_supports_interrogation_form(
                upscaler_value,
            ):
                continue
            offered.append(upscaler_value)
        # Face-fixers gate identically to the beta upscalers above: a newly-added face restorer is
        # withheld until the server lists it, since one unknown post-processor rejects the whole pop.
        # GFPGAN/CodeFormers are in every server's enum and are never gated. The beta names are also
        # offered from the worker-known set so offering does not wait on an SDK release that lists them.
        sdk_facefixers = [m.value for m in KNOWN_FACEFIXERS if m != KNOWN_FACEFIXERS.BACKEND_DEFAULT]
        extra_beta_facefixers = [f for f in sorted(WORKER_KNOWN_BETA_FACEFIXERS) if f not in sdk_facefixers]
        for facefixer_value in (*sdk_facefixers, *extra_beta_facefixers):
            if facefixer_value in WORKER_KNOWN_BETA_FACEFIXERS and not server_supports_interrogation_form(
                facefixer_value,
            ):
                continue
            offered.append(facefixer_value)
        # strip_background runs only on the image-utilities lane (no in-graph fallback); drop just it when
        # that lane is not provisioned or is momentarily down. Upscalers/face-fixers above are pure torch
        # and stay on offer. Unlike image-generation post-processing, alchemy forms are enumerated per-form,
        # so this granular drop is possible.
        if strip_background_available() and utilities_lane_healthy:
            offered.extend(m.value for m in KNOWN_MISC_POST_PROCESSORS)
        else:
            offered.extend(m.value for m in KNOWN_MISC_POST_PROCESSORS if not is_strip_background_form(m.value))

    return offered


class AlchemyHeadroomEstimator:
    """Predicts whether the worker has the VRAM headroom to take on another alchemy form.

    Tracks a rolling window of the VRAM cost actually observed while recent alchemy forms
    ran (the drop in free VRAM measured between the idle baseline and the low-water mark
    during the run) and predicts the next form's cost as the median of those observations,
    never below a configured floor. The coordinator pops alchemy only when measured free
    VRAM covers the prediction. Observed durations are tracked purely for reporting.

    Cold start (no observations yet) falls back to the configured floor, so behavior is
    bounded by configuration until real measurements accumulate.
    """

    _observed_costs_mb: deque[float]
    _observed_durations_s: deque[float]

    def __init__(self, *, sample_window: int = 16) -> None:
        """Initialize with empty observation windows of the given size."""
        self._observed_costs_mb = deque(maxlen=sample_window)
        self._observed_durations_s = deque(maxlen=sample_window)

    def predicted_cost_mb(self, floor_mb: float) -> float:
        """Return the predicted VRAM cost (MB) of the next alchemy form, never below the floor."""
        if not self._observed_costs_mb:
            return floor_mb
        return max(floor_mb, statistics.median(self._observed_costs_mb))

    def fits(self, *, free_vram_mb: float, floor_mb: float) -> bool:
        """Return True if the measured free VRAM covers the predicted cost."""
        return free_vram_mb >= self.predicted_cost_mb(floor_mb)

    @property
    def median_duration_s(self) -> float | None:
        """Return the median observed alchemy form duration, or None before any observation."""
        if not self._observed_durations_s:
            return None
        return statistics.median(self._observed_durations_s)

    def record_run(self, *, vram_cost_mb: float | None, duration_s: float | None) -> None:
        """Record one completed alchemy run's observed VRAM cost and/or duration.

        Non-positive or missing samples are ignored so a noisy measurement (e.g. VRAM freed
        by other work during the run) cannot drag the prediction below a real cost.
        """
        if vram_cost_mb is not None and vram_cost_mb > 0:
            self._observed_costs_mb.append(vram_cost_mb)
        if duration_s is not None and duration_s > 0:
            self._observed_durations_s.append(duration_s)


class AlchemyCoordinator:
    """Owns the alchemy job lifecycle in the main process: pop -> dispatch -> submit."""

    _state: WorkerState
    _process_map: ProcessMap
    _job_tracker: JobTracker
    _reserve_ledger: CommittedReserveLedger
    _shutdown_manager: ShutdownManager
    _runtime_config: RuntimeConfig
    _api_sessions: ApiSessions

    _pending_forms: deque[AlchemyFormSpec]
    _in_flight: dict[str, AlchemyFormSpec]
    """Forms dispatched to a child process, keyed by form_id, awaiting a result message."""
    _in_flight_owner: dict[str, tuple[int, int]]
    """form_id -> the (process_id, process_launch_identifier) the form was dispatched to.

    A result arrives only from the exact launch the form was sent to, so this lets the reaper tell a
    form whose owning process has died (its result will never come) from one that is merely still running.
    """
    _pending_submits: deque[PendingAlchemySubmitJob]
    _form_time_popped: dict[str, float]

    _estimator: AlchemyHeadroomEstimator
    _free_vram_baseline_mb: float | None
    """Free VRAM (MB) sampled while no alchemy was in flight; the cost baseline."""
    _min_free_vram_mb: float | None
    """Low-water mark of free VRAM (MB) seen while the current alchemy batch was in flight."""

    _last_pop_time: float
    _pop_frequency: float
    _error_pop_frequency: float
    _loop_interval: float

    def __init__(
        self,
        *,
        state: WorkerState,
        process_map: ProcessMap,
        job_tracker: JobTracker,
        shutdown_manager: ShutdownManager,
        runtime_config: RuntimeConfig,
        api_sessions: ApiSessions,
        reserve_ledger: CommittedReserveLedger | None = None,
        canned_alchemy_source: CannedAlchemySource | None = None,
        run_metrics: WorkerRunMetrics | None = None,
        annotation_types_provider: Callable[[], frozenset[str]] | None = None,
    ) -> None:
        """Initialize with the shared main-process collaborators.

        Args:
            state: The shared worker state.
            process_map: The shared process map.
            job_tracker: The image job tracker (consulted for the fairness/contention policy).
            shutdown_manager: The shutdown manager.
            runtime_config: Holds the current bridge configuration snapshot.
            api_sessions: The API session holder.
            reserve_ledger: The shared committed-VRAM/RAM ledger image generation also uses, so the two
                flows account for one another's in-flight cost and cannot over-commit the device. When
                ``None`` (unit tests driving the coordinator alone) a private ledger is created.
            canned_alchemy_source: When set, forms come from this source and submits are
                recorded locally instead of touching the API (harness/benchmark mode).
            run_metrics: The shared run-metrics aggregator. Each finished form is recorded here (its name,
                pop->submit timing, and outcome) so alchemy gets the same recent-jobs and rollup
                observability image generation has. ``None`` (unit tests) disables that recording.
            annotation_types_provider: Return the control types currently servable by utilities lanes.
        """
        self._state = state
        self._process_map = process_map
        self._job_tracker = job_tracker
        self._reserve_ledger = reserve_ledger if reserve_ledger is not None else CommittedReserveLedger()
        self._shutdown_manager = shutdown_manager
        self._runtime_config = runtime_config
        self._api_sessions = api_sessions
        self._canned_alchemy_source = canned_alchemy_source
        self._run_metrics = run_metrics
        self._annotation_types_provider = annotation_types_provider
        self.num_canned_forms_completed = 0
        self.num_canned_forms_faulted = 0
        self.num_forms_submitted = 0
        """Cumulative forms successfully submitted to the API (or recorded in canned mode) this session."""
        self.num_forms_faulted = 0
        """Cumulative forms that permanently faulted (stale/invalid/unrecoverable) this session."""

        self._pending_forms = deque()
        self._in_flight = {}
        self._in_flight_owner = {}
        self._pending_submits = deque()
        self._form_time_popped = {}
        self._form_resolution: dict[str, tuple[int, int] | None] = {}
        """Source-image (width, height) per form_id, decoded once at pop, for the dashboard size column.

        None for a form whose image could not be decoded. Pruned to the live forms each projection."""

        self._estimator = AlchemyHeadroomEstimator()
        self._free_vram_baseline_mb = None
        self._min_free_vram_mb = None

        self._last_pop_time = 0.0
        self._pop_frequency = 4.0
        self._error_pop_frequency = 15.0
        self._loop_interval = 1.0

    @property
    def kind(self) -> WorkloadKind:
        """The workload flow this coordinator runs (satisfies the ``FlowCoordinator`` protocol)."""
        return WorkloadKind.ALCHEMY

    @property
    def num_in_flight(self) -> int:
        """Forms popped, dispatched, or awaiting submission (the flow's total live work units)."""
        return len(self._pending_forms) + len(self._in_flight) + len(self._pending_submits)

    @property
    def num_forms_pending(self) -> int:
        """Forms popped from the API but not yet dispatched to a child process."""
        return len(self._pending_forms)

    @property
    def num_forms_in_flight(self) -> int:
        """Forms dispatched to a child process, awaiting a result message."""
        return len(self._in_flight)

    @property
    def num_graph_forms_waiting_or_running(self) -> int:
        """Graph-backed forms still waiting for or occupying the shared post-processing lane."""
        pending = sum(
            1 for spec in self._pending_forms if required_capability(spec.form) is WorkerCapability.ALCHEMY_GRAPH
        )
        running = sum(
            1 for spec in self._in_flight.values() if required_capability(spec.form) is WorkerCapability.ALCHEMY_GRAPH
        )
        return pending + running

    @property
    def num_forms_awaiting_submit(self) -> int:
        """Forms with a result, waiting for API submission."""
        return len(self._pending_submits)

    @staticmethod
    def _decode_image_resolution(source_image_bytes: bytes) -> tuple[int, int] | None:
        """Return the (width, height) of an encoded source image, or None if it cannot be read.

        Only the image header is parsed (Pillow defers pixel decode), so this is cheap. Any failure
        (malformed data, unknown format) yields None so a form still shows in the tables without a size.
        """
        try:
            import io

            import PIL.Image

            with PIL.Image.open(io.BytesIO(source_image_bytes)) as image:
                return (image.width, image.height)
        except Exception:
            return None

    def active_form_statuses(self) -> list[AlchemyFormStatus]:
        """Project the live alchemy forms for the dashboard work-ledger and queue tables.

        Returns one :class:`AlchemyFormStatus` per form currently pending, in flight, or awaiting submit,
        in pipeline order. Also prunes the resolution cache to the live forms so it cannot grow unbounded
        across a long session.
        """
        statuses: list[AlchemyFormStatus] = []

        for spec in self._pending_forms:
            width, height = self._form_resolution.get(spec.form_id) or (None, None)
            statuses.append(AlchemyFormStatus(spec.form_id, spec.form, "pending", width, height, None))

        for form_id, spec in self._in_flight.items():
            width, height = self._form_resolution.get(form_id) or (None, None)
            owner = self._in_flight_owner.get(form_id)
            statuses.append(
                AlchemyFormStatus(form_id, spec.form, "in_flight", width, height, owner[0] if owner else None),
            )

        for submit in self._pending_submits:
            width, height = self._form_resolution.get(submit.form_id) or (None, None)
            statuses.append(
                AlchemyFormStatus(
                    submit.form_id,
                    str(submit.result_message.form),
                    "awaiting_submit",
                    width,
                    height,
                    None,
                ),
            )

        live_ids = {status.form_id for status in statuses}
        for stale_id in [form_id for form_id in self._form_resolution if form_id not in live_ids]:
            self._form_resolution.pop(stale_id, None)

        return statuses

    def set_canned_alchemy_source(self, source: CannedAlchemySource | None) -> None:
        """Swap the canned alchemy source at runtime and reset its per-level counters."""
        self._canned_alchemy_source = source
        self.num_canned_forms_completed = 0
        self.num_canned_forms_faulted = 0

    @property
    def bridge_data(self) -> reGenBridgeData:
        """Return the current bridge configuration."""
        return self._runtime_config.bridge_data

    # region pop

    def _should_pop(self) -> bool:
        """Return True when an alchemy pop is appropriate this cycle.

        Image work always wins contention. In concurrent mode alchemy may pop while image
        jobs are queued, but only into a spare process lane and only with VRAM headroom for
        a typical alchemy form; otherwise it falls back to backfill (image queue empty).
        """
        bridge_data = self.bridge_data
        if not bridge_data.alchemist:
            return False
        if self._state.shutting_down:
            return False
        if self._state.supervisor_paused or self._state.self_throttle_paused:
            return False
        if self._state.gpu_torch_incompatible:
            # The installed PyTorch cannot run this GPU; alchemy forms would fail the same way. Don't pop.
            return False
        if len(self._pending_forms) + len(self._in_flight) >= max(bridge_data.queue_size, 1):
            return False
        if len(self._in_flight) >= max(bridge_data.alchemy_max_concurrency, 1):
            return False
        if (time.time() - self._last_pop_time) < self._pop_frequency:
            return False

        offered = expand_offered_forms(
            bridge_data,
            utilities_lane_healthy=self._utilities_lane_healthy(),
            annotation_types=self._annotation_types(),
        )
        if not offered:
            return False

        # Don't pop work no process can currently take. strip_background runs on the image-utilities lane,
        # so that lane counts as a place work can land alongside the graph and CLIP lanes.
        graph_available = self._process_map.get_first_available(WorkerCapability.ALCHEMY_GRAPH) is not None
        clip_available = self._process_map.get_first_available(WorkerCapability.ALCHEMY_CLIP) is not None
        utilities_available_lane = self._process_map.get_first_available(WorkerCapability.IMAGE_UTILITIES) is not None
        if not (graph_available or clip_available or utilities_available_lane):
            return False

        # Legacy backfill: only pop when the image queue is fully drained.
        if not bridge_data.alchemy_allow_concurrent:
            return len(self._job_tracker.jobs_pending_inference) == 0

        # Concurrent mode: graph forms share a GPU-bearing lane with image work's post-processing tail, so
        # only pop them while image generation has a spare dispatch lane. CLIP-only forms run on the safety
        # process and don't contend for image lanes, so they skip this check.
        offers_graph = any(required_capability(form) is WorkerCapability.ALCHEMY_GRAPH for form in offered)
        if offers_graph and not self._has_spare_image_lane():
            return False

        return self._has_vram_headroom() and self._has_ram_headroom()

    def _utilities_lane_healthy(self) -> bool:
        """Return True when an image-utilities lane process is up and past startup.

        Gates the ``strip_background`` offer at pop time: the form has no in-graph fallback, so it must not
        be advertised while no lane can serve it (a still-starting, ending, or absent lane).
        """
        return self._process_map.num_loaded_utilities_processes() > 0

    def _annotation_types(self) -> frozenset[str]:
        """Return the servable annotation control types the pop request can carry.

        The provider surfaces whatever the live utilities lanes report as annotatable. That raw set is
        narrowed to ``KNOWN_ANNOTATION_CONTROL_TYPES`` here because the pop request field is typed as a list
        of that enum and is built ahead of the pop's own error handling, so a control type the enum cannot
        name would raise a validation error (crash-looping the pop loop) rather than a handled skip. An empty
        result withholds the form entirely: the server reads an absent or empty ``annotation_types`` as
        matching every type, so offering the form without a servable, nameable type would draw work no lane
        here can produce.
        """
        provider = getattr(self, "_annotation_types_provider", None)
        if provider is None:
            return frozenset()
        nameable = {member.value for member in KNOWN_ANNOTATION_CONTROL_TYPES}
        return frozenset(control_type for control_type in provider() if control_type in nameable)

    def _has_spare_image_lane(self) -> bool:
        """Return True if an idle inference lane exists beyond what queued image jobs need."""
        idle_image_lanes = sum(
            1 for p in self._process_map.get_capable_processes(WorkerCapability.IMAGE_GEN) if p.can_accept_job()
        )
        undispatched_image_jobs = max(
            0,
            len(self._job_tracker.jobs_pending_inference) - len(self._job_tracker.jobs_in_progress),
        )
        return idle_image_lanes > undispatched_image_jobs

    def _has_vram_headroom(self) -> bool:
        """Return True if *effective* free VRAM covers a typical alchemy form (per the estimator).

        Effective free is the measured device-wide free VRAM minus everything the shared ledger records
        as already committed by in-flight image and alchemy work, so a form is not admitted against VRAM
        another flow is about to claim.
        """
        free_vram_mb = self._process_map.get_free_vram_mb()
        if free_vram_mb is None:
            # No VRAM telemetry yet (cold start or CPU-only): fall back to backfill.
            return len(self._job_tracker.jobs_pending_inference) == 0
        effective_free_mb = free_vram_mb - self._reserve_ledger.total_vram_mb()
        return self._estimator.fits(
            free_vram_mb=effective_free_mb,
            floor_mb=float(self.bridge_data.alchemy_vram_headroom_mb),
        )

    def _has_ram_headroom(self) -> bool:
        """Return True if effective available system RAM clears the alchemy RAM floor.

        Graph forms keep weights resident in RAM and could push a memory-resident worker into paging, so
        alchemy is held back when available RAM runs low. Effective available RAM subtracts the shared
        ledger's committed RAM. When the floor is zero or RAM cannot be read, this does not gate.
        """
        floor_mb = float(self.bridge_data.alchemy_ram_headroom_mb)
        if floor_mb <= 0:
            return True
        available_ram_mb = self._measured_available_ram_mb()
        if available_ram_mb is None:
            return True
        return (available_ram_mb - self._reserve_ledger.total_ram_mb()) >= floor_mb

    @staticmethod
    def _measured_available_ram_mb() -> float | None:
        """The measured system-wide available RAM (MB), or None if it cannot be read."""
        try:
            return psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            return None

    def _sample_vram(self) -> None:
        """Track the free-VRAM baseline and in-flight low-water mark for the cost estimator."""
        free_vram_mb = self._process_map.get_free_vram_mb()
        if free_vram_mb is None:
            return
        if self._in_flight:
            self._min_free_vram_mb = (
                free_vram_mb if self._min_free_vram_mb is None else min(self._min_free_vram_mb, free_vram_mb)
            )
        else:
            self._free_vram_baseline_mb = free_vram_mb
            self._min_free_vram_mb = None

    async def _download_source_image(self, source_image: str) -> bytes:
        """Return the form's source image as raw bytes, downloading it if it is a URL.

        The horde API delivers a non-URL source image as a base64 string, so that case is decoded to
        bytes here; a URL yields its bytes directly.
        """
        if not source_image.startswith(("http://", "https://")):
            return base64.b64decode(source_image)

        async with self._api_sessions.require_aiohttp_session().get(
            yarl.URL(source_image, encoded=True),
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            response.raise_for_status()
            return await response.read()

    def _canned_alchemy_pop(self) -> None:
        """Take the next form from the canned source, honoring the same pop policy as the API path."""
        assert self._canned_alchemy_source is not None
        if not self._should_pop():
            return

        self._last_pop_time = time.time()
        spec = self._canned_alchemy_source.next_form()
        if spec is None:
            return

        self._form_time_popped[spec.form_id] = time.time()
        self._form_resolution[spec.form_id] = self._decode_image_resolution(spec.source_image_bytes)
        self._pending_forms.append(spec)
        logger.opt(ansi=True).info(
            f"<fg #34c0eb>Popped canned alchemy form {spec.form_id} ({spec.form})</>",
        )

    def _handle_pop_error_response(self, response: RequestErrorResponse) -> None:
        """Log an alchemy pop error, with actionable guidance for common, recoverable causes."""
        message_lower = response.message.lower()
        if "maintenance mode" in message_lower:
            logger.warning(f"Failed to pop alchemy job (Maintenance Mode): {response}")
        elif "wrong credentials" in message_lower:
            logger.warning(f"Failed to pop alchemy job (Wrong Credentials): {response}")
            logger.error("Did you set a unique `alchemist_name` in bridgeData.yaml?")
            logger.error(
                "Alchemist worker names must be unique horde-wide and cannot reuse your `dreamer_name`. "
                "If you haven't used this name before, try changing it.",
            )
        else:
            logger.error(f"Failed to pop alchemy job (API Error): {response}")

    @logger.catch(reraise=True)
    async def api_alchemy_pop(self) -> None:
        """Pop alchemy forms from the API (or the canned source) when the pop policy allows it."""
        if self._canned_alchemy_source is not None:
            self._canned_alchemy_pop()
            return

        if not self._should_pop():
            return

        self._last_pop_time = time.time()
        bridge_data = self.bridge_data

        annotation_types = self._annotation_types()
        pop_request = _AlchemyPopRequest(
            apikey=bridge_data.api_key,
            name=bridge_data.alchemist_name,
            bridge_agent=f"AI Horde Worker reGen:{runtime_version()}:https://github.com/Haidra-Org/horde-worker-reGen",
            priority_usernames=bridge_data.priority_usernames,
            forms=expand_offered_forms(
                bridge_data,
                utilities_lane_healthy=self._utilities_lane_healthy(),
                annotation_types=annotation_types,
            ),
            annotation_types=sorted(annotation_types) or None,
            amount=max(bridge_data.queue_size, 1),
            threads=1,
            max_tiles=min(max(bridge_data.max_power, 1), 256),
        )

        try:
            pop_response = await self._api_sessions.require_horde_client_session().submit_request(
                pop_request,
                _AlchemyPopResponse,
            )
        except Exception as e:
            logger.warning(f"Failed to pop alchemy job (Unexpected Error): {e}")
            self._last_pop_time = time.time() + (self._error_pop_frequency - self._pop_frequency)
            return

        if isinstance(pop_response, RequestErrorResponse):
            self._handle_pop_error_response(pop_response)
            self._last_pop_time = time.time() + (self._error_pop_frequency - self._pop_frequency)
            return

        if not pop_response.forms:
            skipped = pop_response.skipped.model_dump(exclude_defaults=True) if pop_response.skipped else {}
            logger.debug(f"No alchemy forms available. (Skipped reasons: {skipped})")
            return

        for form in pop_response.forms:
            if form.id_ is None:
                logger.error(f"Popped alchemy form has no id: {form}")
                continue
            if form.source_image is None:
                logger.error(f"Popped alchemy form has no source image: {form}")
                continue
            try:
                source_image_bytes = await self._download_source_image(form.source_image)
            except Exception as e:
                logger.error(f"Failed to download alchemy source image for {form.id_}: {e}")
                self._pending_submits.append(
                    PendingAlchemySubmitJob(
                        result_message=HordeAlchemyResultMessage(
                            process_id=-1,
                            process_launch_identifier=-1,
                            info="Source image download failed",
                            form_id=str(form.id_),
                            form=str(form.form),
                            state=GENERATION_STATE.faulted,
                        ),
                        r2_upload=form.r2_upload,
                        time_popped=time.time(),
                    ),
                )
                continue

            control_type = None
            if is_annotation_form(form.form):
                control_type = form.payload.control_type if form.payload is not None else None
                if control_type is None:
                    logger.error(f"Popped annotation form has no control_type: {form}")
                    continue

            spec = AlchemyFormSpec(
                form_id=str(form.id_),
                form=str(form.form),
                source_image_bytes=source_image_bytes,
                r2_upload=form.r2_upload,
                control_type=str(control_type) if control_type is not None else None,
            )
            self._form_time_popped[spec.form_id] = time.time()
            self._form_resolution[spec.form_id] = self._decode_image_resolution(source_image_bytes)
            self._pending_forms.append(spec)
            logger.opt(ansi=True).info(
                f"<fg #34c0eb>Popped alchemy form {spec.form_id} ({spec.form})</>",
            )

    # endregion

    # region dispatch

    def dispatch_pending_forms(self) -> None:
        """Send queued forms to the first available process declaring the needed capability."""
        still_pending: deque[AlchemyFormSpec] = deque()
        while self._pending_forms:
            spec = self._pending_forms.popleft()
            process_info = self._process_map.get_first_available(required_capability(spec.form))
            if process_info is None:
                still_pending.append(spec)
                continue

            sent = process_info.safe_send_message(
                HordeAlchemyControlMessage(
                    control_flag=HordeControlFlag.START_ALCHEMY,
                    form=spec,
                ),
            )
            if sent:
                self._in_flight[spec.form_id] = spec
                self._in_flight_owner[spec.form_id] = (
                    process_info.process_id,
                    process_info.process_launch_identifier,
                )
                logger.debug(
                    f"Dispatched alchemy form {spec.form_id} ({spec.form}) to process {process_info.process_id}",
                )
            else:
                still_pending.append(spec)

        self._pending_forms = still_pending

    def _reap_lost_in_flight_forms(self) -> None:
        """Drop in-flight forms whose owning process launch is gone, before reconciling the reserve.

        A form's result arrives only from the exact process launch it was dispatched to. A child that
        raises while running a form still reports a faulted result, but one that dies *hard* (segfault,
        OOM kill, watchdog kill) before reporting never does, so the form would otherwise sit in
        ``_in_flight`` forever: it would hold a VRAM reserve in the shared ledger that permanently
        starves image-generation admission, and consume an ``alchemy_max_concurrency`` slot. Recovery
        replaces or removes the dead launch, so :meth:`ProcessMap.is_launch_active` tells us the result
        will never come. Reaping the form here (counting it faulted; the horde reissues it elsewhere) is
        what makes :meth:`_sync_reserve_ledger`'s self-healing real.
        """
        lost = [
            form_id
            for form_id, (process_id, launch) in self._in_flight_owner.items()
            if not self._process_map.is_launch_active(process_id, launch)
        ]
        for form_id in lost:
            spec = self._in_flight.pop(form_id, None)
            self._in_flight_owner.pop(form_id, None)
            self._form_time_popped.pop(form_id, None)
            self.num_forms_faulted += 1
            form_name = spec.form if spec is not None else "unknown"
            logger.warning(
                f"Alchemy form {form_id} ({form_name}) lost: the process it was dispatched to is gone "
                "before a result arrived; dropping it so its resource reserve is released.",
            )

    def _sync_reserve_ledger(self) -> None:
        """Publish each in-flight alchemy form's predicted VRAM/RAM cost into the shared ledger.

        Reconciling the whole ``alchemy`` namespace each cycle (rather than add/release per form) makes
        the reserve self-healing: a form whose owning process died is dropped from ``_in_flight`` by
        :meth:`_reap_lost_in_flight_forms` (run just before this), so its hold simply stops being
        republished and no stale reserve leaks to starve image generation.

        The per-form cost is charged by what the form actually allocates, not a flat figure:

        - **Graph forms** (upscalers, facefixers, strip_background) run on the post-processing process, so
          they reserve the estimator's current VRAM prediction (floored by
          ``alchemy_vram_headroom_mb``). Those weights are also kept resident in system RAM, so the form
          additionally reserves ``alchemy_ram_headroom_mb`` of RAM. That RAM hold is what the image
          scheduler's RAM gate and this coordinator's own :meth:`_has_ram_headroom` subtract, so neither
          flow admits work against RAM a graph form is about to claim.
        - **CLIP forms** (caption, nsfw, interrogation) run on the safety process against an already-resident
          model, so dispatching one adds no not-yet-realised VRAM or RAM and is charged zero. Reserving the
          graph cost for them would needlessly hold image generation back.
        """
        predicted_vram = self._estimator.predicted_cost_mb(float(self.bridge_data.alchemy_vram_headroom_mb))
        graph_ram_mb = float(self.bridge_data.alchemy_ram_headroom_mb)

        vram_mb_by_unit: dict[str, float] = {}
        ram_mb_by_unit: dict[str, float] = {}
        for form_id, spec in self._in_flight.items():
            is_graph = required_capability(spec.form) is WorkerCapability.ALCHEMY_GRAPH
            vram_mb_by_unit[form_id] = predicted_vram if is_graph else 0.0
            ram_mb_by_unit[form_id] = graph_ram_mb if is_graph else 0.0

        self._reserve_ledger.replace_flow(
            "alchemy",
            vram_mb_by_unit=vram_mb_by_unit,
            ram_mb_by_unit=ram_mb_by_unit,
        )

    def on_alchemy_result(self, message: HordeAlchemyResultMessage) -> None:
        """Accept a form result from a child process and queue it for submission."""
        spec = self._in_flight.pop(message.form_id, None)
        self._in_flight_owner.pop(message.form_id, None)
        if spec is None:
            logger.warning(f"Received alchemy result for unknown form {message.form_id} ({message.form})")

        time_popped = self._form_time_popped.pop(message.form_id, time.time())
        self._record_estimator_observation(time_popped)

        self._pending_submits.append(
            PendingAlchemySubmitJob(
                result_message=message,
                r2_upload=spec.r2_upload if spec is not None else None,
                time_popped=time_popped,
            ),
        )

    def _record_estimator_observation(self, time_popped: float) -> None:
        """Feed the headroom estimator the duration and (once a batch drains) VRAM cost."""
        duration_s = time.time() - time_popped
        # Attribute the observed VRAM draw only once the whole in-flight batch has drained,
        # so the baseline-to-low-water delta reflects alchemy's footprint rather than noise.
        if self._in_flight or self._free_vram_baseline_mb is None or self._min_free_vram_mb is None:
            self._estimator.record_run(vram_cost_mb=None, duration_s=duration_s)
            return
        observed_cost_mb = self._free_vram_baseline_mb - self._min_free_vram_mb
        self._estimator.record_run(vram_cost_mb=observed_cost_mb, duration_s=duration_s)
        self._free_vram_baseline_mb = None
        self._min_free_vram_mb = None

    # endregion

    # region submit

    async def _upload_form_image(self, submit: PendingAlchemySubmitJob) -> bool:
        """Upload an image-form result (WebP bytes) to the pop-provided R2 URL."""
        if submit.result_message.image_bytes is None:
            logger.error(f"Alchemy form {submit.form_id} has no image to upload")
            return False
        if not submit.r2_upload:
            logger.error(f"Alchemy form {submit.form_id} has no R2 upload URL")
            return False

        image_bytes = submit.result_message.image_bytes
        async with self._api_sessions.require_aiohttp_session().put(
            yarl.URL(submit.r2_upload, encoded=True),
            data=image_bytes,
            skip_auto_headers=["content-type"],
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            if response.status != 200:
                logger.error(f"Failed to upload alchemy result to R2 ({response.status}) for {submit.form_id}")
                return False
        return True

    def _record_form_metrics(self, submit: PendingAlchemySubmitJob, *, faulted: bool) -> None:
        """Record a finished form's timing and outcome into run metrics (the alchemist analogue of finalize).

        Called once per form at its terminal outcome (submitted or faulted). The pop->submit ``e2e`` and the
        source-image resolution mirror what an image job records, so the form shows up in the recent-jobs
        view and the by-form rollup with a real duration. A no-op when no run-metrics aggregator is wired
        (unit tests).
        """
        if self._run_metrics is None:
            return
        width, height = self._form_resolution.get(submit.form_id) or (None, None)
        self._run_metrics.record_alchemy_form(
            form_id=submit.form_id,
            form=str(submit.result_message.form),
            e2e_seconds=max(0.0, time.time() - submit.time_popped),
            faulted=faulted,
            width=width,
            height=height,
        )

    async def _submit_single_form(self, submit: PendingAlchemySubmitJob) -> None:
        """Upload (if needed) and submit one form result; updates the submit state in place."""
        state = submit.result_message.state

        is_image_form = submit.result_message.result_payload is None
        if state == GENERATION_STATE.ok and is_image_form:
            try:
                if not await self._upload_form_image(submit):
                    submit.retry()
                    return
            except Exception as e:
                logger.error(f"Failed to upload alchemy result for {submit.form_id}: {e}")
                submit.retry()
                return

        submit_request = _AlchemySubmitRequest(
            apikey=self.bridge_data.api_key,
            id=submit.form_id,
            result=submit.submit_result,
            state=state,
        )

        try:
            response = await asyncio.wait_for(
                self._api_sessions.require_horde_client_session().submit_request(
                    submit_request,
                    AlchemyJobSubmitResponse,
                ),
                timeout=15,
            )
        except Exception as e:
            logger.error(f"Failed to submit alchemy form {submit.form_id}: {e}")
            submit.retry()
            return

        if isinstance(response, RequestErrorResponse):
            message_lower = response.message.lower()
            if "does not exist" in message_lower or "already submitted" in message_lower:
                logger.warning(f"Alchemy form {submit.form_id} stale on submit: {response.message}")
                submit.fault()
                self.num_forms_faulted += 1
                self._record_form_metrics(submit, faulted=True)
                return
            logger.error(f"Failed to submit alchemy form (API Error) {submit.retry_attempts_string}: {response}")
            submit.retry()
            return

        time_taken = round(time.time() - submit.time_popped, 2)
        logger.opt(ansi=True).success(
            f"Submitted alchemy form {submit.form_id[:8]} (<u>{submit.result_message.form}</u>) "
            f"for {response.reward:,.2f} kudos. Form popped {time_taken} seconds ago.",
        )
        submit_time = time.time()
        self._state.note_first_kudos_event(submit_time)
        self._state.kudos_generated_this_session += response.reward
        self._state.kudos_events.append((submit_time, response.reward))
        submit.succeed(int(response.reward))
        self.num_forms_submitted += 1
        # A successfully-delivered submit can still carry a faulted generation (e.g. a source-image
        # download failure submitted as faulted), so the recorded outcome follows the form's own state.
        self._record_form_metrics(submit, faulted=submit.result_message.state == GENERATION_STATE.faulted)

    def _canned_submit_alchemy(self) -> None:
        """Record a completed form locally instead of submitting to the API."""
        submit = self._pending_submits.popleft()
        if submit.result_message.state == GENERATION_STATE.ok:
            self.num_canned_forms_completed += 1
            self.num_forms_submitted += 1
            submit.succeed(0)
            time_taken = round(time.time() - submit.time_popped, 2)
            logger.opt(ansi=True).success(
                f"Completed canned alchemy form {submit.form_id[:8]} (<u>{submit.result_message.form}</u>) "
                f"in {time_taken} seconds.",
            )
            self._record_form_metrics(submit, faulted=False)
        else:
            self.num_canned_forms_faulted += 1
            self.num_forms_faulted += 1
            submit.fault()
            logger.error(f"Canned alchemy form {submit.form_id} faulted")
            self._record_form_metrics(submit, faulted=True)

    @logger.catch(reraise=True)
    async def api_submit_alchemy(self) -> None:
        """Submit any completed alchemy forms to the API (or record them locally in canned mode)."""
        if not self._pending_submits:
            return

        if self._canned_alchemy_source is not None:
            self._canned_submit_alchemy()
            return

        submit = self._pending_submits.popleft()
        await self._submit_single_form(submit)
        if not submit.is_finished:
            self._pending_submits.append(submit)

    # endregion

    async def run(self) -> None:
        """Run the alchemy pop/dispatch/submit loop."""
        logger.debug("In AlchemyCoordinator.run")

        while True:
            with logger.catch():
                try:
                    self._sample_vram()
                    await self.api_alchemy_pop()
                    self.dispatch_pending_forms()
                    self._reap_lost_in_flight_forms()
                    self._sync_reserve_ledger()
                    await self.api_submit_alchemy()
                    self._state.alchemy_forms_in_flight = (
                        len(self._pending_forms) + len(self._in_flight) + len(self._pending_submits)
                    )
                except CancelledError as e:
                    self._shutdown_manager.shutdown()
                    logger.debug(f"CancelledError: {e}")

            # Checked outside the catch block so persistent errors cannot prevent shutdown.
            if self._shutdown_manager.is_time_for_shutdown() or self._state.shut_down:
                break

            await asyncio.sleep(self._loop_interval)
