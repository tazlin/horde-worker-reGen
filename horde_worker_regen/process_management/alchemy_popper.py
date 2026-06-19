"""Pops, dispatches, and submits alchemy (interrogation) jobs from the AI Horde API.

Image jobs always take priority for process time and VRAM. By default
(``alchemy_allow_concurrent``) alchemy may still run alongside image generation, but only
when a process lane is spare (no waiting image job needs it) and the headroom estimator
judges there is enough free VRAM for a typical alchemy form. With ``alchemy_allow_concurrent``
off, alchemy reverts to a strict backfill workload that pops only when the image queue is
empty. Graph forms (upscalers, facefixers, strip_background) are dispatched to inference
processes; CLIP forms (caption, interrogation, nsfw) to the safety process. Dispatch is
keyed on :class:`WorkerCapability`, not process type.
"""

from __future__ import annotations

import asyncio
import base64
import statistics
import time
from asyncio import CancelledError
from collections import deque
from typing import TYPE_CHECKING, override

import aiohttp
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
    KNOWN_FACEFIXERS,
    KNOWN_MISC_POST_PROCESSORS,
    KNOWN_UPSCALERS,
    is_facefixer_form,
    is_strip_background_form,
    is_upscaler_form,
)
from loguru import logger

from horde_worker_regen.capabilities import strip_background_available
from horde_worker_regen.process_management._canned_scenarios import CannedAlchemySource
from horde_worker_regen.process_management.api_sessions import ApiSessions
from horde_worker_regen.process_management.horde_process import WorkerCapability
from horde_worker_regen.process_management.job_models import PendingAlchemySubmitJob
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import (
    AlchemyFormSpec,
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeControlFlag,
)
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.worker_state import WorkerState
from horde_worker_regen.runtime_version import runtime_version

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.shutdown_manager import ShutdownManager


class _AlchemyPopRequest(AlchemyPopRequest):
    """AlchemyPopRequest with `forms` relaxed to plain strings.

    The SDK's `KNOWN_ALCHEMY_TYPES` enum has `CodeFormers` aliased to the value "GFPGAN"
    (an upstream copy-paste bug), so "CodeFormers" cannot round-trip through the enum.
    """

    forms: list[str]  # type: ignore[assignment]


class _AlchemyPopResponse(AlchemyJobPopResponse):
    """AlchemyJobPopResponse opted out of the SDK session's failure-cleanup tracking.

    The base class advertises `AlchemyJobSubmitRequest` as its failure-cleanup request but
    supplies only ``{"id": ...}`` as construction params, so the SDK client session raises a
    pydantic ValidationError (missing `result`/`state`) building the cleanup request — after
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


def required_capability(form: str) -> WorkerCapability:
    """Return the capability a process must declare to serve the given form."""
    if is_upscaler_form(form) or is_facefixer_form(form) or is_strip_background_form(form):
        return WorkerCapability.ALCHEMY_GRAPH
    return WorkerCapability.ALCHEMY_CLIP


DEFAULT_ALCHEMY_FORMS: tuple[str, ...] = ("caption", "nsfw", "interrogation", "post-process")
"""Forms an alchemist offers when ``bridge_data.forms`` is left unset (an empty list means "all").

The SDK's ``default_forms`` validator does not fire for the default empty list, so both the dispatch
path and the dashboard projection fall back to this set. The yaml/legacy spelling is ``post-process``;
the SDK enum value is ``post_process``.
"""


def expand_offered_forms(bridge_data: reGenBridgeData) -> list[str]:
    """Expand the bridge-data `forms` config into the individual form names the API expects.

    "post-process" expands to every known upscaler, facefixer, and strip_background,
    mirroring the legacy alchemist. Caption requires the explicit BLIP opt-in.
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
    if "post-process" in configured:
        offered.extend(m.value for m in KNOWN_UPSCALERS if m != KNOWN_UPSCALERS.BACKEND_DEFAULT)
        offered.extend(m.value for m in KNOWN_FACEFIXERS if m != KNOWN_FACEFIXERS.BACKEND_DEFAULT)
        # strip_background needs rembg (no wheels on some backends); drop just it on a lean install.
        # Upscalers/face-fixers above are pure torch and stay on offer. Unlike image-generation
        # post-processing, alchemy forms are enumerated per-form, so this granular drop is possible.
        if strip_background_available():
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
    _shutdown_manager: ShutdownManager
    _runtime_config: RuntimeConfig
    _api_sessions: ApiSessions

    _pending_forms: deque[AlchemyFormSpec]
    _in_flight: dict[str, AlchemyFormSpec]
    """Forms dispatched to a child process, keyed by form_id, awaiting a result message."""
    _pending_submits: deque[PendingAlchemySubmitJob]
    _form_time_popped: dict[str, float]

    _estimator: AlchemyHeadroomEstimator
    _free_vram_baseline_mb: float | None
    """Free VRAM (MB) sampled while no alchemy was in flight — the cost baseline."""
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
        canned_alchemy_source: CannedAlchemySource | None = None,
    ) -> None:
        """Initialize with the shared main-process collaborators.

        Args:
            state: The shared worker state.
            process_map: The shared process map.
            job_tracker: The image job tracker (consulted for contention policy).
            shutdown_manager: The shutdown manager.
            runtime_config: Holds the current bridge configuration snapshot.
            api_sessions: The API session holder.
            canned_alchemy_source: When set, forms come from this source and submits are
                recorded locally instead of touching the API (harness/benchmark mode).
        """
        self._state = state
        self._process_map = process_map
        self._job_tracker = job_tracker
        self._shutdown_manager = shutdown_manager
        self._runtime_config = runtime_config
        self._api_sessions = api_sessions
        self._canned_alchemy_source = canned_alchemy_source
        self.num_canned_forms_completed = 0
        self.num_canned_forms_faulted = 0
        self.num_forms_submitted = 0
        """Cumulative forms successfully submitted to the API (or recorded in canned mode) this session."""
        self.num_forms_faulted = 0
        """Cumulative forms that permanently faulted (stale/invalid/unrecoverable) this session."""

        self._pending_forms = deque()
        self._in_flight = {}
        self._pending_submits = deque()
        self._form_time_popped = {}

        self._estimator = AlchemyHeadroomEstimator()
        self._free_vram_baseline_mb = None
        self._min_free_vram_mb = None

        self._last_pop_time = 0.0
        self._pop_frequency = 4.0
        self._error_pop_frequency = 15.0
        self._loop_interval = 1.0

    @property
    def num_forms_pending(self) -> int:
        """Forms popped from the API but not yet dispatched to a child process."""
        return len(self._pending_forms)

    @property
    def num_forms_in_flight(self) -> int:
        """Forms dispatched to a child process, awaiting a result message."""
        return len(self._in_flight)

    @property
    def num_forms_awaiting_submit(self) -> int:
        """Forms with a result, waiting for API submission."""
        return len(self._pending_submits)

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
        if len(self._pending_forms) + len(self._in_flight) >= max(bridge_data.queue_size, 1):
            return False
        if len(self._in_flight) >= max(bridge_data.alchemy_max_concurrency, 1):
            return False
        if (time.time() - self._last_pop_time) < self._pop_frequency:
            return False

        offered = expand_offered_forms(bridge_data)
        if not offered:
            return False

        # Don't pop work no process can currently take.
        graph_available = self._process_map.get_first_available(WorkerCapability.ALCHEMY_GRAPH) is not None
        clip_available = self._process_map.get_first_available(WorkerCapability.ALCHEMY_CLIP) is not None
        if not (graph_available or clip_available):
            return False

        # Legacy backfill: only pop when the image queue is fully drained.
        if not bridge_data.alchemy_allow_concurrent:
            return len(self._job_tracker.jobs_pending_inference) == 0

        # Concurrent mode: graph forms share inference lanes with image generation, so only
        # take a lane image work does not currently need. (CLIP-only forms run on the safety
        # process and don't contend for image lanes, so they skip this check.)
        offers_graph = any(required_capability(form) is WorkerCapability.ALCHEMY_GRAPH for form in offered)
        if offers_graph and not self._has_spare_image_lane():
            return False

        return self._has_vram_headroom()

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
        """Return True if free VRAM covers a typical alchemy form (per the estimator)."""
        free_vram_mb = self._process_map.get_free_vram_mb()
        if free_vram_mb is None:
            # No VRAM telemetry yet (cold start or CPU-only): fall back to backfill.
            return len(self._job_tracker.jobs_pending_inference) == 0
        return self._estimator.fits(
            free_vram_mb=free_vram_mb,
            floor_mb=float(self.bridge_data.alchemy_vram_headroom_mb),
        )

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

    async def _download_source_image(self, source_image: str) -> str:
        """Return the form's source image as base64, downloading it if it is a URL."""
        if not source_image.startswith(("http://", "https://")):
            return source_image

        async with self._api_sessions.require_aiohttp_session().get(
            yarl.URL(source_image, encoded=True),
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            response.raise_for_status()
            return base64.b64encode(await response.read()).decode("utf-8")

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

        pop_request = _AlchemyPopRequest(
            apikey=bridge_data.api_key,
            name=bridge_data.alchemist_name,
            bridge_agent=f"AI Horde Worker reGen:{runtime_version()}:https://github.com/Haidra-Org/horde-worker-reGen",
            priority_usernames=bridge_data.priority_usernames,
            forms=expand_offered_forms(bridge_data),
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
                source_image_base64 = await self._download_source_image(form.source_image)
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

            spec = AlchemyFormSpec(
                form_id=str(form.id_),
                form=str(form.form),
                source_image_base64=source_image_base64,
                r2_upload=form.r2_upload,
            )
            self._form_time_popped[spec.form_id] = time.time()
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
                logger.debug(
                    f"Dispatched alchemy form {spec.form_id} ({spec.form}) to process {process_info.process_id}",
                )
            else:
                still_pending.append(spec)

        self._pending_forms = still_pending

    def on_alchemy_result(self, message: HordeAlchemyResultMessage) -> None:
        """Accept a form result from a child process and queue it for submission."""
        spec = self._in_flight.pop(message.form_id, None)
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
        if submit.result_message.image_base64 is None:
            logger.error(f"Alchemy form {submit.form_id} has no image to upload")
            return False
        if not submit.r2_upload:
            logger.error(f"Alchemy form {submit.form_id} has no R2 upload URL")
            return False

        image_bytes = base64.b64decode(submit.result_message.image_base64)
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
                return
            logger.error(f"Failed to submit alchemy form (API Error) {submit.retry_attempts_string}: {response}")
            submit.retry()
            return

        time_taken = round(time.time() - submit.time_popped, 2)
        logger.opt(ansi=True).success(
            f"Submitted alchemy form {submit.form_id[:8]} (<u>{submit.result_message.form}</u>) "
            f"for {response.reward:,.2f} kudos. Form popped {time_taken} seconds ago.",
        )
        self._state.kudos_generated_this_session += response.reward
        self._state.kudos_events.append((time.time(), response.reward))
        submit.succeed(int(response.reward))
        self.num_forms_submitted += 1

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
        else:
            self.num_canned_forms_faulted += 1
            self.num_forms_faulted += 1
            submit.fault()
            logger.error(f"Canned alchemy form {submit.form_id} faulted")

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
