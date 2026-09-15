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

Generations arrive as a stream of text, which gives the flow the one liveness signal a deadline cannot
provide: a backend that has stopped producing is silent, while a backend that is merely slow is not. So
the deadline stays the outer bound on a whole generation and a silence of `text_stall_seconds` ends one
early, with the first token after a backend starts exempted because a cold backend spends seconds
warming its kernels. The flow spends one short generation of its own on that warm-up where it owns the
backend, so the first popped job does not.

Which backend is attached is a constructor argument twice over: the object the flow generates through
(the [`TextBackend`][horde_worker_regen.text_backends.protocol.TextBackend] protocol, never a concrete
driver) and the `TEXT_BACKENDS` value naming which one it is, which is all that decides how the
advertised model name is spelled. Adding a backend is therefore a change to those two arguments, and a
worker-owned backend process a change to who builds the first, rather than a change to this loop.
Attaching to a backend the operator started is the whole of the lifecycle today.
"""

from __future__ import annotations

import asyncio
import contextlib
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
    TextBackendCapabilities,
    TextBackendDescription,
    TextBackendError,
    TextBackendRejectedPayload,
    TextBackendUnavailable,
    TextGenerationProgress,
    TextGenerationProgressCallback,
    TextGenerationResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
    from horde_worker_regen.process_management.config.worker_state import WorkerState
    from horde_worker_regen.process_management.ipc.api_sessions import ApiSessions
    from horde_worker_regen.process_management.lifecycle.shutdown_manager import ShutdownManager
    from horde_worker_regen.process_management.resources.run_metrics import WorkerRunMetrics


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

WARM_UP_PROMPT: Final = "Hello"
"""The prompt the warm-up generation uses. Short and unremarkable: nothing reads its answer."""

WARM_UP_MAX_LENGTH: Final = 8
"""How many tokens the warm-up generation asks for.

Enough that the backend runs its generation kernels at all, few enough that the wait is the warm-up
itself rather than the generation.
"""

WARM_UP_GENERATION_KEY: Final = "regen-text-warm-up"
"""The key the warm-up generation is issued with, deliberately not the shape of a horde job id.

Job ids are UUIDs, so this can never name a real generation, and an abort or a statistics sample that
went astray onto this key would name nothing the flow is holding.
"""

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


class TextPayloadKeys:
    """The horde payload fields the flow reads for its own records or sets on its own requests.

    The payload is the horde's and is forwarded to the backend untouched; these are the two fields the
    worker looks at, named here rather than spelled at each site.
    """

    PROMPT: Final = "prompt"
    MAX_LENGTH: Final = "max_length"


class _GenerationStalled(Exception):
    """The backend stopped producing tokens for a generation it had begun.

    Private to this module: it never leaves the offer loop that raises it, which turns it into the same
    abandon-and-fault outcome a generation that outlived its deadline gets. A separate type from
    `TimeoutError` because the two are different diagnoses, and the log line has to say which happened.
    """


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


def _payload_int(payload: dict[str, object], key: str) -> int | None:
    """Return one integer field of a popped payload, or None when it is absent or not an integer.

    The payload is the horde's, forwarded to the backend untouched and never validated by the worker, so
    a field read out of it for the worker's own records tolerates a shape the worker did not expect
    rather than faulting a job that is otherwise fine.
    """
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


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
    model_name: str | None = None
    """The model this job was popped against, as the worker advertised it.

    Held on the job rather than read back at submit time: a backend that goes away clears the
    advertisement, and the job's own record would then name no model at all.
    """
    time_generation_started: float | None = None
    """When the backend was first asked to generate this job, or None while it never was.

    The boundary between the job's queue wait and its generation, which is the split an image job's
    record carries and the only one a text job has: the backend is a separate program, so there is no
    dispatch, load or safety stage between the two."""
    time_generation_finished: float | None = None
    """When the backend answered (or was given up on), or None while it has not."""
    progress: TextGenerationProgress | None = None
    """The most recent thing the backend said about this generation while it was running.

    None before the first chunk arrives and for a job the backend never began. Replaced rather than
    accumulated: each report already carries the running totals, so the latest one is the whole state.
    """
    prompt_tokens: int | None = None
    """Prompt tokens the backend reported for this job, or None when nothing it offers reported them."""
    generated_tokens: int | None = None
    """Generated tokens the backend reported for this job, or None when nothing reported them."""
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
        run_metrics: WorkerRunMetrics | None = None,
    ) -> None:
        """Initialize with the shared main-process collaborators and the backend to generate through.

        Args:
            state: The shared worker state, read for shutdown and updated with the session's kudos.
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
            run_metrics: Where each finished job's record lands, so text jobs reach the same aggregates
                image jobs and alchemy forms do. None in tests and in any assembly with no aggregator,
                which records nothing rather than failing.

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
        self._run_metrics = run_metrics

        self._in_flight: dict[str, TextJobInFlight] = {}
        self._job_tasks: set[asyncio.Task[None]] = set()
        self._maintenance_hold_logged = False

        self._advertisement: TextFlowAdvertisement | None = None
        """What the backend's last description lets the worker offer. None until the gate has passed."""

        self._backend_capabilities: TextBackendCapabilities | None = None
        """What the backend's optional routes turned out to be. None until the gate has passed."""

        self._backend_is_cold = True
        """Whether the backend has yet to produce a token since it started.

        A backend that has just launched (or relaunched) spends seconds warming its generation kernels
        before its first token appears, so the stall bound cannot apply to that first token. Set again
        by every readiness gate, which is what a relaunch drives the flow back through, and cleared by
        the first token the backend produces, whether that is the warm-up's or a job's.
        """

        self.num_jobs_submitted = 0
        """Cumulative text jobs successfully submitted to the API this session."""
        self.num_jobs_faulted = 0
        """Cumulative text jobs submitted as faulted this session."""

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
    def backend_ready(self) -> bool:
        """Whether the backend has reported a loaded model and described itself.

        The one readiness fact outside this flow: nothing is popped while it is False, so a dashboard
        showing a scribe with no work has the reason here rather than in the log.
        """
        return self._advertisement is not None

    @property
    def backend_capabilities(self) -> TextBackendCapabilities | None:
        """What the backend's optional routes turned out to be, or None while the gate is open.

        The one thing that decides whether a text job can report token counts and a live rate at all, so
        a reader asking why a job shows none finds the answer here rather than concluding the job is odd.
        """
        return self._backend_capabilities

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
        self._backend_capabilities = None
        self._backend_is_cold = True
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
                    await self._note_backend_capabilities(backend)
                    await self._warm_up_backend()
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

    async def _note_backend_capabilities(self, backend: TextBackend) -> None:
        """Record which optional routes this backend has, and say once what that means for its jobs.

        Asked at the gate rather than on the first generation so the answer is settled before any job
        depends on it, and said out loud because "why does my text worker show no token rate" is
        otherwise a question with no answer anywhere.
        """
        self._backend_capabilities = await backend.capabilities()
        if self._backend_capabilities.generation_stats:
            logger.info("Text backend reports per-request statistics; text jobs will carry token counts and a rate.")
            return
        logger.info(
            "Text backend has no per-request statistics route, so text jobs report how much text has "
            "arrived but no token counts and no token rate. A stock backend build is expected to answer this way.",
        )

    async def _warm_up_backend(self) -> None:
        """Spend one short generation nobody is waiting for, so the first popped job does not pay for it.

        The first generation after a backend starts produces its first tokens seconds later while the
        inference kernels warm: two tokens in five seconds where the same backend runs at a hundred and
        seventy-six a second once warm. A popped job that paid that would look stalled and would be
        slower than the horde was promised. Skipped for a backend the operator runs themselves, which
        may be serving other clients whose generation slot this is not the worker's to spend.

        A warm-up that fails changes nothing: the gate has already passed on the backend's own answers,
        and the next real job will find out soon enough whether generation works.
        """
        if self.bridge_data.text_backend_managed is not True:
            logger.debug("Skipping the text backend warm-up: the operator runs this backend, and it may be serving.")
            return

        warm_up_started_at = time.monotonic()
        try:
            await self.require_backend().generate(
                {TextPayloadKeys.PROMPT: WARM_UP_PROMPT, TextPayloadKeys.MAX_LENGTH: WARM_UP_MAX_LENGTH},
                generation_key=WARM_UP_GENERATION_KEY,
                deadline_seconds=self.generation_deadline_seconds(),
            )
        except TextBackendError as warm_up_failure:
            logger.warning(f"The text backend warm-up generation did not complete: {warm_up_failure}")
            return

        self._backend_is_cold = False
        logger.info(
            f"Text backend warmed up in {time.monotonic() - warm_up_started_at:.1f}s "
            f"({WARM_UP_MAX_LENGTH} tokens, discarded); the first popped job starts on warm kernels.",
        )

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
        self._start_popped_jobs(pop_response, advertisement=advertisement)

    def _start_popped_jobs(
        self,
        pop_response: TextGenerateJobPopResponse,
        *,
        advertisement: TextFlowAdvertisement,
    ) -> None:
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
            job = TextJobInFlight(
                job_id=job_id,
                payload=payload,
                time_popped=time.time(),
                model_name=advertisement.model_name,
            )
            self._in_flight[job_id] = job
            self._state.text_jobs_in_flight = len(self._in_flight)
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
            result = await self._generate_text(job)
            if result is None:
                await self.submit_job(job, generation="", state=GENERATION_STATE.faulted)
                return
            await self.submit_job(job, generation=result.text, state=GENERATION_STATE.ok)
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
                self._state.text_jobs_in_flight = len(self._in_flight)

    async def _generate_text(self, job: TextJobInFlight) -> TextGenerationResult | None:
        """Return the backend's answer for one job, or None when the job must be faulted.

        Retries only the case where retrying can work: a busy backend is healthy and the payload is fine,
        so the same job is worth re-offering shortly. A refused payload will be refused again, and an
        unreachable backend makes every job fail the same way, so both end the job here (and the second
        sends the flow back to the readiness gate).
        """
        try:
            result = await self._offer_until_answered(job)
        finally:
            job.time_generation_finished = time.time()

        if result is not None:
            job.prompt_tokens = result.prompt_tokens
            job.generated_tokens = result.generated_tokens
        return result

    async def _offer_until_answered(self, job: TextJobInFlight) -> TextGenerationResult | None:
        """Return the backend's answer for one job, re-offering it while the backend answers busy.

        Split from :meth:`_generate_text` so every way out of the offer loop stamps the generation's end
        once, rather than each of its exits repeating the stamp.
        """
        backend = self.require_backend()
        deadline_seconds = self.generation_deadline_seconds()

        while True:
            # Re-stamped per attempt, so a job the backend turned away as busy measures its wait to the
            # offer that was actually taken rather than to the first one that was not.
            job.time_generation_started = time.time()
            try:
                result = await self._generate_until_stalled_or_answered(
                    job,
                    backend=backend,
                    deadline_seconds=deadline_seconds,
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
            except _GenerationStalled as stalled:
                logger.error(f"Text job {job.job_id[:8]} stalled: {stalled}. Abandoning the generation.")
                await self._stop_generation(job)
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
            return result

    async def _generate_until_stalled_or_answered(
        self,
        job: TextJobInFlight,
        *,
        backend: TextBackend,
        deadline_seconds: float,
    ) -> TextGenerationResult:
        """Return one offer's answer, giving up early on a backend that stopped producing tokens.

        The deadline bounds a generation that is running slowly; it cannot bound one that is not running
        at all, because a backend that wedged at the first token looks exactly like a backend that will
        answer at the last moment, and the difference is minutes of a requester's time. The text arrives
        continuously, so silence is the signal: a generation that has produced nothing for
        `text_stall_seconds` is over, whatever its deadline still allows.

        Raises:
            _GenerationStalled: The backend produced nothing for longer than its patience allows.
            TextBackendError: Whatever the backend raised for this offer.
            TimeoutError: The offer outlived the deadline without the backend enforcing it.
        """
        chunk_arrived = asyncio.Event()
        generation = asyncio.create_task(
            asyncio.wait_for(
                backend.generate(
                    job.payload,
                    generation_key=job.job_id,
                    deadline_seconds=deadline_seconds,
                    on_progress=self._progress_recorder(job, chunk_arrived),
                ),
                timeout=deadline_seconds + GENERATE_DEADLINE_GRACE_SECONDS,
            ),
        )
        offer_started_at = time.monotonic()

        try:
            while True:
                patience_seconds = self._chunk_patience_seconds(job, deadline_seconds=deadline_seconds)
                last_chunk_at = job.progress.last_chunk_at if job.progress is not None else None
                silent_since = last_chunk_at if last_chunk_at is not None else offer_started_at
                silence_allowed_for = silent_since + patience_seconds - time.monotonic()
                if silence_allowed_for <= 0:
                    raise _GenerationStalled(
                        f"no text arrived for {patience_seconds:.1f}s after "
                        f"{job.progress.chunks_received if job.progress is not None else 0} chunk(s)",
                    )
                if await _generation_finished_first(generation, chunk_arrived, timeout=silence_allowed_for):
                    return generation.result()
                # A chunk landed, so the silence the loop was measuring is over and the next one is
                # measured from this arrival. Every wake re-reads the patience too, because the first
                # chunk of a cold backend is the one token held to a different bound than the rest.
                chunk_arrived.clear()
        finally:
            await _cancel_generation(generation)

    def _progress_recorder(
        self,
        job: TextJobInFlight,
        chunk_arrived: asyncio.Event,
    ) -> TextGenerationProgressCallback:
        """Return the callback that puts one job's live progress where the flow and the snapshot read it."""

        def record_progress(progress: TextGenerationProgress) -> None:
            """Hold the latest report on the job, and wake the stall bound when text actually arrived.

            Statistics samples come through here too and carry no new text, so only a report that
            advanced the chunk count counts as the backend having produced something.
            """
            chunks_before = job.progress.chunks_received if job.progress is not None else 0
            job.progress = progress
            if progress.chunks_received > chunks_before:
                self._backend_is_cold = False
                chunk_arrived.set()

        return record_progress

    def _chunk_patience_seconds(self, job: TextJobInFlight, *, deadline_seconds: float) -> float:
        """Return how long this generation may stay silent before the flow gives up on it.

        A backend that has not produced a token since it started is warming its kernels, which takes
        seconds and is not a fault, so the first token of a cold backend is given the whole deadline.
        Every token after that, and every token of a warm backend, is held to the stall bound.
        """
        if job.progress is None and self._backend_is_cold:
            return deadline_seconds
        return self.bridge_data.text_stall_seconds

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
            self._state.text_jobs_in_flight = len(self._in_flight)

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
        submit_time = time.time()
        if is_faulted:
            self.num_jobs_faulted += 1
            logger.info(f"Reported text job {job.job_id[:8]} as faulted to the horde")
            self._record_job_metrics(job, submit_time=submit_time, faulted=True, kudos_reward=None)
            return

        self.num_jobs_submitted += 1
        # The session kudos pool is the account's, not one flow's: text earnings belong in the same
        # total and on the same kudos/hr clock as image generation's and alchemy's.
        self._state.note_first_kudos_event(submit_time)
        self._state.kudos_generated_this_session += response.reward
        self._state.kudos_events.append((submit_time, response.reward))
        time_taken = round(submit_time - job.time_popped, 2)
        logger.success(
            f"Submitted text job {job.job_id[:8]} for {response.reward:,.2f} kudos. "
            f"Job popped {time_taken} seconds ago.",
        )
        self._record_job_metrics(job, submit_time=submit_time, faulted=False, kudos_reward=response.reward)

    def _record_job_metrics(
        self,
        job: TextJobInFlight,
        *,
        submit_time: float,
        faulted: bool,
        kudos_reward: float | None,
    ) -> None:
        """Record one delivered text job into run metrics. A no-op when no aggregator is wired.

        A job faulted before the backend was ever asked has no generation span, so its timings are left
        unknown rather than reported as zero, and its token counts are unknown for the same reason: only
        a backend that ran the job can say how many tokens it was, and only some backends say at all.
        """
        if self._run_metrics is None:
            return
        generation_started = job.time_generation_started
        generation_finished = job.time_generation_finished
        queue_wait_seconds = max(0.0, generation_started - job.time_popped) if generation_started is not None else None
        generation_seconds = (
            max(0.0, generation_finished - generation_started)
            if generation_started is not None and generation_finished is not None
            else None
        )
        self._run_metrics.record_text_job(
            job_id=job.job_id,
            model_name=job.model_name,
            time_popped=job.time_popped,
            time_submitted=submit_time,
            queue_wait_seconds=queue_wait_seconds,
            generation_seconds=generation_seconds,
            faulted=faulted,
            kudos_reward=kudos_reward,
            prompt_tokens=job.prompt_tokens,
            generated_tokens=job.generated_tokens,
            max_length=_payload_int(job.payload, TextPayloadKeys.MAX_LENGTH),
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


async def _generation_finished_first(
    generation: asyncio.Task[TextGenerationResult],
    chunk_arrived: asyncio.Event,
    *,
    timeout: float,
) -> bool:
    """Return whether the generation finished before a chunk arrived or the wait ran out.

    Three things can end this wait and each means something different: the generation answered, the
    backend produced text (so the silence being measured is over), or neither happened for long enough
    that the backend has stopped. Waiting on the generation alone would measure silence only in whole
    waits, so a stall bound shorter than the deadline would never be reached.
    """
    waiting_for_a_chunk = asyncio.ensure_future(chunk_arrived.wait())
    try:
        finished, _still_waiting = await asyncio.wait(
            {generation, waiting_for_a_chunk},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        waiting_for_a_chunk.cancel()
        # The waiter exists only to wake this loop, so its cancellation is the expected end of it.
        with contextlib.suppress(CancelledError):
            await waiting_for_a_chunk

    return generation in finished


async def _cancel_generation(generation: asyncio.Task[TextGenerationResult]) -> None:
    """Abandon a generation the flow is no longer waiting on, and wait for it to notice.

    Awaited rather than left to finish on its own so that no generation task outlives the offer that
    started it: the offer has already decided the job's outcome, and a task still writing progress into
    a job that has been faulted would be reporting on work nobody is waiting for.
    """
    if generation.done():
        return

    generation.cancel()
    try:
        await generation
    except CancelledError:
        # Expected: this is the cancellation just asked for arriving. The caller's own reason for
        # abandoning the generation is the exception that propagates, not this one.
        pass
    except Exception as abandoned_failure:
        logger.debug(f"An abandoned text generation ended with {abandoned_failure!r}")
