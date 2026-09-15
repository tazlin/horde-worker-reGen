"""The text-generation workload flow: pop a text job, ask an external backend for it, submit the text.

Text generation is the one flow with no inference child. Generation happens in a separate program, the
text backend (koboldcpp today; sonar and others planned), which the worker reaches over HTTP through
[`text_backends`][horde_worker_regen.text_backends], so this coordinator is a plain main-process asyncio
loop: torch-free, GPU-free, holding no place in the process pool and no share of the VRAM budget.

Two things the flow owns that the backend deliberately does not:

- **What the worker advertises.** The backend is the only thing that knows which weights it loaded and
  what caps it will honour, so nothing is popped until it has answered, and the pop offers the narrower
  of the operator's configuration and the backend's own answer. Advertising more than the backend
  accepts produces jobs it then refuses.
- **What happens to a job when a call fails.** The drivers never retry and never decide a job's fate;
  only the flow knows the job, the horde and the operator's configuration. Every popped job therefore
  ends in a submit, faulted where it must be. A worker that quietly drops a failed job holds its
  requester until the server times the job out, and the horde charges the worker for that.

Each popped job runs as its own task covering both its generation and its submit, so a slow submit
delays neither the next pop nor another generation.

Which backend is attached is a constructor argument twice over: the object the flow generates through
(the [`TextBackend`][horde_worker_regen.text_backends.protocol.TextBackend] protocol, never a concrete
driver) and the `TEXT_BACKENDS` value naming which one it is, which is all that decides how the
advertised model name is spelled. Adding a backend is therefore a change to those two arguments, and a
worker-owned backend process a change to who builds the first, rather than a change to this loop.
Attaching to a backend the operator started is the whole of the lifecycle today.
"""

from __future__ import annotations

import asyncio
import time
from asyncio import CancelledError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from horde_model_reference.meta_consts import TEXT_BACKENDS
from horde_model_reference.text_backend_names import (
    TEXT_LEGACY_BACKEND_PREFIXES,
    get_model_name_variants,
    strip_backend_prefix,
)
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import (
    JobSubmitResponse,
    TextGenerateJobPopRequest,
    TextGenerateJobPopResponse,
    TextGenerationJobSubmitRequest,
)
from horde_sdk.ai_horde_api.consts import RC
from loguru import logger

from horde_worker_regen.bridge_data.data_model import derive_text_generation_timeout_seconds, reGenBridgeData
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind
from horde_worker_regen.runtime_version import runtime_version
from horde_worker_regen.text_backends import (
    FakeTextBackend,
    KoboldApiTextBackend,
    TextBackend,
    TextBackendBusy,
    TextBackendDescription,
    TextBackendRejectedPayload,
    TextBackendUnavailable,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
    from horde_worker_regen.process_management.config.worker_state import WorkerState
    from horde_worker_regen.process_management.ipc.api_sessions import ApiSessions
    from horde_worker_regen.process_management.lifecycle.shutdown_manager import ShutdownManager


TEXT_POP_REQUEST_TIMEOUT_SECONDS: Final = 30.0
"""Maximum time a text pop may occupy the coordinator loop.

The SDK session inherits aiohttp's much longer default request timeout, and this coordinator is a
gathered main-loop task, so letting that default govern would keep the whole worker alive after every
child had been reaped. Matches the image and alchemy pop bounds and stays well above normal API latency.
"""

TEXT_SUBMIT_REQUEST_TIMEOUT_SECONDS: Final = 30.0
"""Maximum time one text submit attempt may take before it counts as failed and is retried."""

READY_PROBE_DEADLINE_SECONDS: Final = 5.0
"""How long one readiness probe waits for an answer.

Short on purpose: a backend that is up answers the model route immediately, and one that is still
loading refuses the connection, so a long probe deadline buys nothing and delays the next attempt.
"""

READY_BACKOFF_INITIAL_SECONDS: Final = 1.0
"""Wait after the first unready answer, doubled up to :data:`READY_BACKOFF_MAX_SECONDS`."""

READY_BACKOFF_MAX_SECONDS: Final = 15.0
"""Longest wait between readiness probes, so a backend started late is picked up promptly."""

READY_SLOW_START_NOTICE_SECONDS: Final = 120.0
"""How long the readiness gate waits before saying out loud that the backend has not answered.

A cold start is slow in a way that is entirely normal: a packaged backend unpacks itself and then loads
weights from disk, which together run to a minute or more, so the gate's patience is minutes and its
first notice is well past a healthy start. The notice then repeats on this interval, because an operator
who never started the backend, or pointed the worker at the wrong port, would otherwise see a silent
worker. The gate never gives up: a scribe that stopped polling would never serve again once the
operator fixed it.
"""

BUSY_RETRY_MAX_ATTEMPTS: Final = 5
"""How many times one job is re-offered to a busy backend before it is faulted.

A busy answer means the backend is healthy and the payload is fine, so the job is worth re-offering;
but a backend that is busy this long is over-subscribed (or `text_threads` is set above what it will
actually run in parallel), and holding the job any longer only risks the server timing it out.
"""

BUSY_RETRY_WAIT_SECONDS: Final = 2.0
"""Wait between re-offers of a job the backend was too busy to take."""

SUBMIT_MAX_ATTEMPTS: Final = 3
"""How many times a text submit is attempted before the result is given up on."""

SUBMIT_RETRY_WAIT_SECONDS: Final = 2.0
"""Wait between submit attempts for one job."""

GENERATE_DEADLINE_GRACE_SECONDS: Final = 5.0
"""Slack between the deadline handed to the backend and the flow's own bound on the same call.

The driver owns the deadline and reports outliving it as an unavailable backend. This grace makes the
flow's bound the backstop for a driver or stand-in that does not return on time, rather than a second
deadline racing the first.
"""

SHUTDOWN_DRAIN_TIMEOUT_SECONDS: Final = 30.0
"""How long shutdown waits for in-flight generations and submits after asking the backend to stop.

Bounded because the backend is another program: it may ignore the abort, or have no abort route at all,
and the worker cannot be held open on its behalf.
"""

FAULTED_GENERATION_TEXT: Final = "Faulted"
"""What a faulted text submit carries in place of a generation.

The submit model logs an error for an empty generation, and this is the spelling the SDK itself uses
when it faults a text job on the worker's behalf, so a fault reported by the flow and one reported by
the SDK read identically in the log and on the server.
"""

DRY_RUN_CANONICAL_MODEL_NAME: Final = "meta-llama/Llama-3.2-3B-Instruct"
"""The model the dry-run stand-in claims to have loaded, as the text model reference spells it."""

DRY_RUN_MAX_CONTEXT_LENGTH: Final = 4096
"""Context cap the dry-run stand-in claims, in the range a small instruct model runs at."""

DRY_RUN_MAX_LENGTH: Final = 512
"""Generation-length cap the dry-run stand-in claims."""

DRY_RUN_GENERATION_TEXT: Final = "This generation came from a dry-run worker with no text backend attached."
"""What the dry-run stand-in generates, worded so it cannot be mistaken for a real generation."""


@dataclass(frozen=True)
class TextFlowAdvertisement:
    """Represents what the worker offers the horde on behalf of one text backend.

    Built once per readiness gate from the operator's configuration and the backend's own answer, and
    held until the gate runs again, so every pop in between makes the same promise. Frozen because a
    promise that changed under the pops made against it is the disagreement the gate exists to prevent.
    """

    model_name: str
    """The model name as the horde spells it for the backend in use, prefix included."""
    max_length: int
    """Largest generation length offered: the lower of the operator's cap and the backend's."""
    max_context_length: int
    """Largest context offered: the lower of the operator's cap and the backend's."""
    soft_prompts: tuple[str, ...]
    """The soft prompts the backend exposes, exactly as it reported them. Empty when it has none."""


def advertised_model_name(
    *,
    configured_name: str | None,
    described_name: str,
    backend: TEXT_BACKENDS,
) -> str:
    """Return the model name to advertise, spelled the way the horde spells it for this backend.

    The canonical name is the operator's `text_model_name` when they set one, otherwise the backend's
    own reported name with any backend prefix stripped (a backend may report its model already
    prefixed). The advertised name is then the variant of that canonical name belonging to the backend
    in use. Which variant that is, is a convention the model reference owns rather than a rule
    reimplemented here: the koboldcpp variant drops the author segment, so
    `meta-llama/Llama-3.2-3B-Instruct` is advertised as `koboldcpp/Llama-3.2-3B-Instruct`, while the
    aphrodite variant keeps the whole canonical name. Adding a backend therefore changes the
    `backend` argument and nothing here. A quantisation suffix is part of the model name and rides
    along untouched.

    Args:
        configured_name: The operator's `text_model_name`, or `None` to follow the backend.
        described_name: The model name the backend reported.
        backend: Which backend the flow generates through.

    Returns:
        The prefixed model name to offer on a pop.

    Raises:
        RuntimeError: The model reference produced no variant for this backend, which would leave the
            worker with nothing it could truthfully advertise.
    """
    canonical_name = configured_name if configured_name is not None else strip_backend_prefix(described_name)
    backend_prefix = TEXT_LEGACY_BACKEND_PREFIXES[backend]

    variants = get_model_name_variants(canonical_name)
    prefixed_variant = next((variant for variant in variants if variant.startswith(backend_prefix)), None)
    if prefixed_variant is None:
        raise RuntimeError(
            f"No {backend.value} name variant exists for text model {canonical_name!r} (variants: {variants})",
        )
    return prefixed_variant


def build_text_flow_advertisement(
    *,
    bridge_data: reGenBridgeData,
    description: TextBackendDescription,
    backend: TEXT_BACKENDS,
) -> TextFlowAdvertisement:
    """Create the advertisement for a described backend under the operator's configuration.

    Each cap is the lower of the two figures. The operator's is a ceiling they chose; the backend's is
    what it will actually accept, and a job past that is a job it refuses after the worker promised it.
    The soft prompts are the backend's list verbatim: the worker passes a requested soft prompt through
    and has no way to supply one the backend does not have.

    Args:
        bridge_data: The live configuration, read for `text_model_name` and the two caps.
        description: What the backend answered when asked to describe itself.
        backend: Which backend the flow generates through, which decides the advertised spelling.

    Returns:
        The facts every pop until the next readiness gate will carry.
    """
    return TextFlowAdvertisement(
        model_name=advertised_model_name(
            configured_name=bridge_data.text_model_name,
            described_name=description.model_name,
            backend=backend,
        ),
        max_length=min(bridge_data.max_length, description.max_length),
        max_context_length=min(bridge_data.max_context_length, description.max_context_length),
        soft_prompts=tuple(description.soft_prompts),
    )


def dry_run_backend_description(backend: TEXT_BACKENDS) -> TextBackendDescription:
    """Create the description the dry-run stand-in answers with, spelled as that backend would spell it.

    The name is put through the same variant selection a live backend's answer goes through, so a dry
    run exercises the real name derivation rather than a tidier spelling that would hide a mistake in it.

    Args:
        backend: Which backend the stand-in is standing in for.

    Returns:
        A small instruct model with ordinary caps and no soft prompts.
    """
    return TextBackendDescription(
        model_name=advertised_model_name(
            configured_name=DRY_RUN_CANONICAL_MODEL_NAME,
            described_name=DRY_RUN_CANONICAL_MODEL_NAME,
            backend=backend,
        ),
        max_context_length=DRY_RUN_MAX_CONTEXT_LENGTH,
        max_length=DRY_RUN_MAX_LENGTH,
    )


def build_text_backend(
    *,
    bridge_data: reGenBridgeData,
    api_sessions: ApiSessions,
    text_backend_kind: TEXT_BACKENDS,
    base_url: str | None = None,
) -> TextBackend:
    """Create the backend the text flow generates through for this worker's configuration.

    A dry run gets the in-process stand-in, so the flow can be exercised end to end on a machine with no
    inference program installed, the way `fake_worker_processes` lets the image pipeline run with no GPU.
    Otherwise the driver is pointed at the URL the operator says their backend listens on, over the
    aiohttp session the parent already owns. Every backend so far speaks the KoboldAI HTTP API, so they
    share one driver and differ only in what the operator launched at that URL.

    Called when the flow starts rather than when the coordinator is built, because the shared sessions
    are only populated once the event loop is running.

    Args:
        bridge_data: The live configuration, read for the dry-run flag and the backend URL.
        api_sessions: The shared session holder the driver borrows its HTTP session from.
        text_backend_kind: Which backend is attached, which the stand-in mimics. A backend that does not
            speak the KoboldAI API would also select a different driver here.
        base_url: Where the backend listens. None means the operator's `kai_url`; a worker that launches
            the backend itself passes the loopback URL of the process it started.

    Returns:
        A backend satisfying the protocol; which implementation is an implementation detail above here.
    """
    if bridge_data.dry_run_skip_api:
        return FakeTextBackend(
            description=dry_run_backend_description(text_backend_kind),
            response_text=DRY_RUN_GENERATION_TEXT,
            latency_seconds=bridge_data.dry_run_inference_delay,
        )
    return KoboldApiTextBackend(
        base_url if base_url is not None else bridge_data.kai_url,
        api_sessions.require_aiohttp_session(),
    )


@dataclass
class TextJobInFlight:
    """Represents one popped text job for as long as the flow is responsible for it.

    Lives from the pop until the submit resolves, so it also carries the two facts that outlive a single
    call: how many times a busy backend has turned the job away, and whether the flow has already asked
    the backend to abandon the generation.
    """

    job_id: str
    """The horde's id for this generation, which is also the key the backend knows it by."""
    payload: dict[str, object]
    """The payload the horde sent, as it was sent, forwarded to the backend untouched."""
    time_popped: float
    """When the job was popped, for the pop-to-submit timing in the submit log line."""
    busy_attempts: int = 0
    """How many times the backend has answered busy for this job."""
    stopped: bool = False
    """Whether the flow has asked the backend to abandon this generation.

    A stopped generation is faulted whatever comes back for it. A backend may answer an abandoned
    generation with a normal success carrying the tokens it had produced (koboldcpp does), so a result
    arriving after a stop is a partial answer to a job the flow has already given up on, and submitting
    it as a good generation would hand the requester a truncated reply that reads as complete.
    """


class TextGenerationCoordinator:
    """Owns the text-generation job lifecycle in the main process: gate, pop, generate, submit."""

    def __init__(
        self,
        *,
        state: WorkerState,
        shutdown_manager: ShutdownManager,
        runtime_config: RuntimeConfig,
        api_sessions: ApiSessions,
        backend: TextBackend | None = None,
        backend_factory: Callable[[TEXT_BACKENDS], TextBackend] | None = None,
        text_backend_kind: TEXT_BACKENDS,
    ) -> None:
        """Initialize with the shared main-process collaborators and the backend to generate through.

        Args:
            state: The shared worker state, read for shutdown.
            shutdown_manager: The shutdown manager, observed for when to stop and notified on cancellation.
            runtime_config: Holds the current bridge configuration snapshot, re-read every cycle so a
                hot-reloaded `scribe` or `text_threads` takes effect without a restart.
            api_sessions: The API session holder the pops and submits go through.
            backend: The backend to generate through, when the caller already has one (tests, and any
                later assembly that owns the backend's lifecycle itself).
            backend_factory: Builds the backend when the flow starts, for a caller whose backend cannot
                exist before the event loop does (the driver borrows the shared aiohttp session, which
                the main loop populates after construction). Handed `text_backend_kind`, so the object
                built and the kind the flow advertises for cannot disagree. Ignored when `backend` is
                given; one of the two is required.
            text_backend_kind: Which backend the flow generates through, which is the whole of what
                decides how the advertised model name is spelled. Defaults to the first backend the
                worker supported; a further one is a different value here and nothing else.

        Raises:
            ValueError: Neither `backend` nor `backend_factory` was given, leaving nothing to generate
                through.
        """
        if backend is None and backend_factory is None:
            raise ValueError("TextGenerationCoordinator needs either a backend or a backend_factory")

        self._state = state
        self._shutdown_manager = shutdown_manager
        self._runtime_config = runtime_config
        self._api_sessions = api_sessions
        self._backend = backend
        self._backend_factory = backend_factory
        self._text_backend_kind = text_backend_kind

        self._in_flight: dict[str, TextJobInFlight] = {}
        self._job_tasks: set[asyncio.Task[None]] = set()
        self._maintenance_hold_logged = False

        self._advertisement: TextFlowAdvertisement | None = None
        """What the backend's last description lets the worker offer. None until the gate has passed."""

        self.num_jobs_submitted = 0
        """Cumulative text jobs successfully submitted to the API this session."""
        self.num_jobs_faulted = 0
        """Cumulative text jobs submitted as faulted this session."""
        self.kudos_earned_this_session = 0.0
        """Cumulative kudos the horde has rewarded for this flow's submits this session."""

        self._last_pop_time = 0.0
        self._pop_frequency = 2.0
        self._error_pop_frequency = 15.0
        self._loop_interval = 1.0

        # Last skipped-reasons snapshot actually logged, so the "no text jobs available" line is
        # edge-triggered: it fires when the reasons change, never once per empty pop at steady state.
        self._last_logged_no_jobs_skipped: dict[str, object] | None = None
        self._closed = False

    @property
    def kind(self) -> WorkloadKind:
        """The workload flow this coordinator runs (satisfies the ``FlowCoordinator`` protocol)."""
        return WorkloadKind.TEXT_GENERATION

    @property
    def num_in_flight(self) -> int:
        """Text jobs popped, generating, or awaiting submission (the flow's total live work units)."""
        return len(self._in_flight)

    @property
    def advertisement(self) -> TextFlowAdvertisement | None:
        """What the worker currently offers for this backend, or None while the readiness gate is open."""
        return self._advertisement

    @property
    def text_backend_kind(self) -> TEXT_BACKENDS:
        """Which text backend this flow generates through."""
        return self._text_backend_kind

    @property
    def bridge_data(self) -> reGenBridgeData:
        """Return the current bridge configuration."""
        return self._runtime_config.bridge_data

    def require_backend(self) -> TextBackend:
        """Return the backend, building it from the factory on first use.

        Raises:
            RuntimeError: The coordinator was built with neither a backend nor a factory, which its
                constructor refuses, so reaching this means the backend was cleared after construction.
        """
        if self._backend is None:
            if self._backend_factory is None:
                raise RuntimeError("the text flow has no backend and no way to build one")
            self._backend = self._backend_factory(self._text_backend_kind)
        return self._backend

    # region readiness

    async def await_backend_ready(self) -> bool:
        """Poll the backend until it reports a loaded model, then record what it says it can do.

        Nothing is popped before both calls succeed. What a pop advertises is a promise about what the
        backend can do, and only the backend knows which weights it loaded and what caps it honours, so
        a pop made before it has answered is a promise the worker guessed at.

        Returns:
            `True` when the backend answered and described itself, `False` when shutdown interrupted the
            wait.
        """
        self._advertisement = None
        backend = self.require_backend()
        gate_opened_at = time.monotonic()
        backoff_seconds = READY_BACKOFF_INITIAL_SECONDS
        last_notice_at = gate_opened_at

        while not self._state.shutting_down:
            if await backend.ready(deadline_seconds=READY_PROBE_DEADLINE_SECONDS):
                description = await self._describe_backend(backend)
                if description is not None:
                    self._advertisement = build_text_flow_advertisement(
                        bridge_data=self.bridge_data,
                        description=description,
                        backend=self._text_backend_kind,
                    )
                    logger.info(
                        f"Text backend ready after {time.monotonic() - gate_opened_at:.1f}s; advertising "
                        f"{self._advertisement.model_name} (max_length={self._advertisement.max_length}, "
                        f"max_context_length={self._advertisement.max_context_length}, "
                        f"soft_prompts={len(self._advertisement.soft_prompts)})",
                    )
                    return True

            # Edge-triggered on the notice interval rather than emitted per probe: the gate polls every
            # few seconds and a cold start legitimately takes a minute or more.
            waited_seconds = time.monotonic() - gate_opened_at
            if time.monotonic() - last_notice_at >= READY_SLOW_START_NOTICE_SECONDS:
                last_notice_at = time.monotonic()
                logger.warning(
                    f"Text backend at {self.bridge_data.kai_url} has not reported a loaded model after "
                    f"{waited_seconds:.0f}s. Is it running, and is `kai_url` its address? A password-protected "
                    "backend that refuses the worker's credentials reads as a backend that is down.",
                )
            else:
                logger.trace(f"Text backend not ready after {waited_seconds:.1f}s; polling again")

            await asyncio.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, READY_BACKOFF_MAX_SECONDS)

        return False

    async def _describe_backend(self, backend: TextBackend) -> TextBackendDescription | None:
        """Return what the backend says it can do, or None when it could not be asked.

        A backend that answers the readiness probe and then fails to describe itself is a backend that
        went away between the two calls, so the gate stays open rather than advertising a guess.
        """
        try:
            return await backend.describe()
        except TextBackendUnavailable as unavailable:
            logger.warning(f"Text backend reported ready but could not describe itself: {unavailable}")
            return None

    # endregion

    # region pop

    def _should_pop(self) -> bool:
        """Return True when a text pop is appropriate this cycle."""
        bridge_data = self.bridge_data
        if bridge_data.scribe is not True:
            return False
        if self._state.shutting_down:
            return False
        if self._advertisement is None:
            return False
        if len(self._in_flight) >= bridge_data.text_threads:
            return False
        return (time.time() - self._last_pop_time) >= self._pop_frequency

    def _build_pop_request(self, advertisement: TextFlowAdvertisement) -> TextGenerateJobPopRequest:
        """Create the pop request offering exactly what the gate established this worker can serve."""
        bridge_data = self.bridge_data
        return TextGenerateJobPopRequest(
            apikey=bridge_data.api_key,
            name=bridge_data.scribe_name,
            bridge_agent=f"AI Horde Worker reGen:{runtime_version()}:https://github.com/Haidra-Org/horde-worker-reGen",
            priority_usernames=bridge_data.priority_usernames,
            models=[advertisement.model_name],
            max_length=advertisement.max_length,
            max_context_length=advertisement.max_context_length,
            softprompts=list(advertisement.soft_prompts),
            threads=bridge_data.text_threads,
            nsfw=bridge_data.nsfw,
            amount=1,
        )

    def _enter_pop_error_backoff(self) -> None:
        """Hold off the next pop by the error interval after a pop that could not be completed."""
        self._last_pop_time = time.time() + (self._error_pop_frequency - self._pop_frequency)

    def _handle_pop_error_response(self, response: RequestErrorResponse) -> None:
        """Log a text pop error, with actionable guidance for the common, recoverable causes.

        A worker in maintenance is refused every pop with the operator's own maintenance message as the
        text and ``WorkerMaintenance`` as the return code. That is a deliberate operator state, not a
        fault, so it is announced once when it begins (and the resumption once, in the pop path) and
        repeats stay at TRACE; the horde still routes the owner's own requests to such a worker.
        """
        if response.rc == RC.WorkerMaintenance:
            if not self._maintenance_hold_logged:
                logger.info(
                    f"Text pops are held: this worker is in maintenance on the horde ({response.message!r}). "
                    "Only its owner's requests reach it until maintenance is lifted.",
                )
                self._maintenance_hold_logged = True
            else:
                logger.trace(f"Text pop refused while in maintenance: {response.message!r}")
            return
        message_lower = response.message.lower()
        if "wrong credentials" in message_lower:
            logger.warning(f"Failed to pop text job (Wrong Credentials): {response}")
            logger.error("Did you set a unique `scribe_name` in bridgeData.yaml?")
            logger.error(
                "Scribe worker names must be unique horde-wide and cannot reuse your `dreamer_name` or "
                "`alchemist_name`. If you haven't used this name before, try changing it.",
            )
        else:
            logger.error(f"Failed to pop text job (API Error): {response}")

    async def api_text_pop(self) -> None:
        """Pop a text job from the API when the pop policy allows it, and start work on what came back."""
        advertisement = self._advertisement
        if advertisement is None or not self._should_pop():
            return

        self._last_pop_time = time.time()

        try:
            pop_response = await asyncio.wait_for(
                self._api_sessions.require_horde_client_session().submit_request(
                    self._build_pop_request(advertisement),
                    TextGenerateJobPopResponse,
                ),
                timeout=TEXT_POP_REQUEST_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(f"Text pop request timed out after {TEXT_POP_REQUEST_TIMEOUT_SECONDS:.0f} seconds")
            self._enter_pop_error_backoff()
            return
        except Exception as pop_error:
            logger.warning(f"Failed to pop text job (Unexpected Error): {pop_error}")
            self._enter_pop_error_backoff()
            return

        if isinstance(pop_response, RequestErrorResponse):
            self._handle_pop_error_response(pop_response)
            self._enter_pop_error_backoff()
            return

        if self._maintenance_hold_logged:
            logger.info("Text pops resumed: the horde is accepting this worker's pops again.")
            self._maintenance_hold_logged = False
        self._start_popped_jobs(pop_response)

    def _start_popped_jobs(self, pop_response: TextGenerateJobPopResponse) -> None:
        """Take every generation the pop returned into the ledger and start a task for each."""
        job_ids = self._popped_job_ids(pop_response)
        if not job_ids:
            skipped = pop_response.skipped.model_dump(exclude_defaults=True)
            if skipped != self._last_logged_no_jobs_skipped:
                logger.debug(f"No text jobs available. (Skipped reasons: {skipped})")
                self._last_logged_no_jobs_skipped = skipped
            return
        self._last_logged_no_jobs_skipped = None

        payload = dict(pop_response.payload.model_dump(by_alias=True, exclude_unset=True))
        for job_id in job_ids:
            job = TextJobInFlight(job_id=job_id, payload=payload, time_popped=time.time())
            self._in_flight[job_id] = job
            logger.info(
                f"Popped text job {job_id[:8]} for {pop_response.model} "
                f"(softprompt: {pop_response.softprompt}, ttl: {pop_response.ttl})",
            )
            self._launch_job_task(job)

    @staticmethod
    def _popped_job_ids(pop_response: TextGenerateJobPopResponse) -> list[str]:
        """Return the generation ids the pop returned, empty when it returned no work."""
        if pop_response.id_ is not None:
            return [str(pop_response.id_)]
        return [str(job_id) for job_id in pop_response.ids]

    def _launch_job_task(self, job: TextJobInFlight) -> None:
        """Run one job's generation and submit on its own task, so neither blocks the pop loop."""
        task = asyncio.create_task(self.run_job(job))
        self._job_tasks.add(task)
        task.add_done_callback(self._job_tasks.discard)

    # endregion

    # region generate

    async def run_job(self, job: TextJobInFlight) -> None:
        """Generate one popped job and submit the outcome, faulting rather than dropping it.

        Every exit from here submits something: a job popped and then abandoned holds its requester until
        the server times it out, which the horde charges the worker for.
        """
        try:
            result_text = await self._generate_text(job)
            if result_text is None:
                await self.submit_job(job, generation="", state=GENERATION_STATE.faulted)
                return
            await self.submit_job(job, generation=result_text, state=GENERATION_STATE.ok)
        except CancelledError:
            raise
        except Exception as unexpected_error:
            logger.opt(exception=True).error(f"Text job {job.job_id[:8]} failed unexpectedly: {unexpected_error}")
            # The fault report is the last thing owed to the horde for this job, so a failure reporting
            # it is logged here rather than left to surface as an unretrieved task exception.
            try:
                await self.submit_job(job, generation="", state=GENERATION_STATE.faulted)
            except Exception as fault_report_error:
                logger.error(f"Could not report text job {job.job_id[:8]} as faulted: {fault_report_error}")
                self._in_flight.pop(job.job_id, None)

    async def _generate_text(self, job: TextJobInFlight) -> str | None:
        """Return the generated text for one job, or None when the job must be faulted.

        Retries only the case where retrying can work: a busy backend is healthy and the payload is fine,
        so the same job is worth re-offering shortly. A refused payload will be refused again, and an
        unreachable backend makes every job fail the same way, so both end the job here (and the second
        sends the flow back to the readiness gate).
        """
        backend = self.require_backend()
        deadline_seconds = self.generation_deadline_seconds()

        while True:
            try:
                result = await asyncio.wait_for(
                    backend.generate(
                        job.payload,
                        generation_key=job.job_id,
                        deadline_seconds=deadline_seconds,
                    ),
                    timeout=deadline_seconds + GENERATE_DEADLINE_GRACE_SECONDS,
                )
            except TextBackendBusy as busy:
                job.busy_attempts += 1
                if job.busy_attempts >= BUSY_RETRY_MAX_ATTEMPTS:
                    logger.warning(
                        f"Text backend was busy for all {BUSY_RETRY_MAX_ATTEMPTS} attempts at job "
                        f"{job.job_id[:8]}; faulting it rather than holding it longer: {busy}",
                    )
                    return None
                logger.debug(f"Text backend busy for job {job.job_id[:8]}, re-offering it: {busy}")
                await asyncio.sleep(BUSY_RETRY_WAIT_SECONDS)
                continue
            except TextBackendRejectedPayload as rejected:
                logger.error(f"Text backend refused the payload for job {job.job_id[:8]}: {rejected}")
                return None
            except TextBackendUnavailable as unavailable:
                logger.error(f"Text backend failed job {job.job_id[:8]}: {unavailable}")
                await self._stop_generation(job)
                self._advertisement = None
                return None
            except TimeoutError:
                logger.error(
                    f"Text job {job.job_id[:8]} outlived its {deadline_seconds:.0f}s deadline; "
                    "abandoning the generation",
                )
                await self._stop_generation(job)
                return None

            if job.stopped:
                logger.warning(
                    f"Discarding the {len(result.text)} characters that arrived for abandoned text job "
                    f"{job.job_id[:8]}; a stopped generation is incomplete however much of it came back",
                )
                return None
            return result.text

    def generation_deadline_seconds(self) -> float:
        """Return how long one generation may take before the flow abandons it.

        The operator's own figure when they set one, otherwise derived from the generation-length cap.
        """
        configured_timeout = self.bridge_data.text_generation_timeout_seconds
        if configured_timeout is not None:
            return configured_timeout
        return derive_text_generation_timeout_seconds(self.bridge_data.max_length)

    async def _stop_generation(self, job: TextJobInFlight) -> None:
        """Ask the backend to abandon one generation and mark the job as given up on.

        Marked before the call, not after, because the backend may answer the abandoned generation with
        a success carrying whatever it had produced, and that partial answer must not be submitted as a
        good one.
        """
        job.stopped = True
        await self.require_backend().stop(generation_key=job.job_id)

    # endregion

    # region submit

    async def submit_job(self, job: TextJobInFlight, *, generation: str, state: GENERATION_STATE) -> None:
        """Submit one job's outcome to the API and drop it from the ledger.

        A faulted submit carries :data:`FAULTED_GENERATION_TEXT` rather than the empty string the submit
        model logs an error for.
        """
        try:
            await self._submit_with_retries(job, generation=generation, state=state)
        finally:
            self._in_flight.pop(job.job_id, None)

    async def _submit_with_retries(self, job: TextJobInFlight, *, generation: str, state: GENERATION_STATE) -> None:
        """Attempt the submit a bounded number of times, logging the outcome of the last attempt."""
        is_faulted = state == GENERATION_STATE.faulted
        submit_request = TextGenerationJobSubmitRequest(
            apikey=self.bridge_data.api_key,
            id=job.job_id,
            generation=FAULTED_GENERATION_TEXT if is_faulted else generation,
            state=state,
        )

        for attempt in range(1, SUBMIT_MAX_ATTEMPTS + 1):
            try:
                response = await asyncio.wait_for(
                    self._api_sessions.require_horde_client_session().submit_request(
                        submit_request,
                        JobSubmitResponse,
                    ),
                    timeout=TEXT_SUBMIT_REQUEST_TIMEOUT_SECONDS,
                )
            except Exception as submit_error:
                logger.error(f"Failed to submit text job {job.job_id[:8]} (attempt {attempt}): {submit_error}")
                if attempt < SUBMIT_MAX_ATTEMPTS:
                    await asyncio.sleep(SUBMIT_RETRY_WAIT_SECONDS)
                continue

            if isinstance(response, RequestErrorResponse):
                message_lower = response.message.lower()
                if "does not exist" in message_lower or "already submitted" in message_lower:
                    logger.warning(f"Text job {job.job_id[:8]} stale on submit: {response.message}")
                    self.num_jobs_faulted += 1
                    return
                logger.error(f"Failed to submit text job {job.job_id[:8]} (API Error, attempt {attempt}): {response}")
                if attempt < SUBMIT_MAX_ATTEMPTS:
                    await asyncio.sleep(SUBMIT_RETRY_WAIT_SECONDS)
                continue

            self._note_submitted(job, response=response, is_faulted=is_faulted)
            return

        logger.error(f"Gave up submitting text job {job.job_id[:8]} after {SUBMIT_MAX_ATTEMPTS} attempts")

    def _note_submitted(self, job: TextJobInFlight, *, response: JobSubmitResponse, is_faulted: bool) -> None:
        """Record and log one delivered submit. A fault report is delivered work the horde pays nothing for."""
        if is_faulted:
            self.num_jobs_faulted += 1
            logger.info(f"Reported text job {job.job_id[:8]} as faulted to the horde")
            return

        self.num_jobs_submitted += 1
        self.kudos_earned_this_session += response.reward
        time_taken = round(time.time() - job.time_popped, 2)
        logger.success(
            f"Submitted text job {job.job_id[:8]} for {response.reward:,.2f} kudos. "
            f"Job popped {time_taken} seconds ago.",
        )

    # endregion

    async def run(self) -> None:
        """Run the text gate/pop/generate/submit loop until shutdown."""
        logger.debug("In TextGenerationCoordinator.run")

        while True:
            with logger.catch():
                try:
                    if self.bridge_data.scribe is True and self._advertisement is None:
                        await self.await_backend_ready()
                    await self.api_text_pop()
                except CancelledError as cancelled:
                    self._shutdown_manager.shutdown()
                    logger.debug(f"CancelledError: {cancelled}")

            # Checked outside the catch block so persistent errors cannot prevent shutdown.
            if self._shutdown_manager.is_time_for_shutdown() or self._state.shut_down:
                break

            await asyncio.sleep(self._loop_interval)

        await self.stop_in_flight_and_close()

    async def stop_in_flight_and_close(self) -> None:
        """Abandon every in-flight generation, wait a bounded time for the submits, and close the backend.

        The generations are abandoned before the wait, not after it, because a text generation can run
        for minutes and the worker cannot hold the process open for one. Whatever comes back for an
        abandoned generation is faulted by the job's own task, so the horde is told about every popped
        job either way.
        """
        if self._closed:
            return
        self._closed = True

        if self._backend is None:
            # The scribe role was off for this whole run, so the flow never built a backend: there is
            # nothing generating and no driver state to release. Building one here to close it would
            # mean reaching for a session the loop has already finished with.
            return

        backend = self._backend
        for job in list(self._in_flight.values()):
            logger.debug(f"Abandoning in-flight text job {job.job_id[:8]} for shutdown")
            await self._stop_generation(job)

        if self._job_tasks:
            logger.info(f"Waiting up to {SHUTDOWN_DRAIN_TIMEOUT_SECONDS:.0f}s for {len(self._job_tasks)} text job(s)")
            await asyncio.wait(set(self._job_tasks), timeout=SHUTDOWN_DRAIN_TIMEOUT_SECONDS)

        if self._in_flight:
            logger.warning(f"{len(self._in_flight)} text job(s) did not finish before the flow closed")

        await backend.close()
