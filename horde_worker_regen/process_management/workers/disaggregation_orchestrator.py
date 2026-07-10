"""The parent-side orchestrator that runs a job as a disaggregated pipeline.

Under pipeline disaggregation a job is split into stages that run in separate processes exchanging
small activations, not weights: a text-encode service produces CONDITIONING, a UNet-only sampler
consumes it and produces a LATENT, an image lane VAE-decodes it to raw images (post-processing runs
on the dedicated post-processing lane afterward), with an optional VAE-encode front-end for img2img.
This orchestrator drives that per-job
DAG from the parent: it dispatches each stage to the appropriate role process, holds the
intermediate blobs so a stage's death loses only the executing stage (the parent re-dispatches from
held state), and advances a job to the next stage as each result arrives.

It is deliberately asynchronous/pipelined: the encode of job B is fired while job A samples, so the
VAE decode of the previous job overlaps the next sample. The role-process finders and the
"images ready" hand-off are injected callables so the flow is exercised against the
``make_testable_process_manager`` fakes without a live GPU.

Stage results arrive through ``MessageDispatcher.set_stage_result_handler`` (registered to
:meth:`handle_stage_result`); the orchestrator is otherwise ticked by the scheduler loop
(:meth:`tick`) to (re)dispatch any job whose next stage has an available role process.

Residency admission (how many samplers may be *resident* at once) is settled by the scheduler before a job
reaches here. This orchestrator additionally gates how many samplers may be *sampling* at the same instant:
two concurrent activation peaks can over-commit a card whose VRAM comfortably holds both samplers at rest, so
:meth:`_dispatch_sample` admits a second-or-later concurrent sampling only when the VRAM arbiter judges its
estimated peak to fit the device's static sampling headroom alongside the in-flight peaks
(:data:`_active_sampling_peaks`). A sole sampling is always admitted (the monolithic status quo), and a gate
deferral is healthy backpressure that never ages a job toward a fault.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto

from loguru import logger

from horde_worker_regen.process_management.ipc.messages import (
    GENERATION_STATE,
    HordeControlMessage,
    HordeImageResult,
    HordeProcessMessage,
    HordeSampleControlMessage,
    HordeSampleResultMessage,
    HordeStageModelMixin,
    HordeTextEncodeControlMessage,
    HordeTextEncodeResultMessage,
    HordeVaeDecodeControlMessage,
    HordeVaeDecodeResultMessage,
    HordeVaeEncodeControlMessage,
    HordeVaeEncodeResultMessage,
    SampleSliceSpec,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.resources.vram_arbiter import (
    VramArbiter,
    VramRequest,
    VramRequestKind,
)

_STAGE_PATIENCE_SECONDS = 90.0
"""How long a job may sit unable to advance (no role process for its next stage) before it is faulted.

Mirrors the post-process lane's admission patience: a job whose next stage has no live role process
(the service/lane/sampler crashed and did not return) is faulted so the horde reissues it, rather than
being parked forever. Sits below the orphan-recovery grace and the server-side timeout."""

_RESOURCE_DEFER_SECONDS = 90.0
"""How long a stage may keep hitting resource-class (device out-of-memory) faults before the job is re-routed.

A resource-class stage fault (the encode lane denied VRAM under device pressure) is not a genuine error, so
the stage is deferred and retried while the pressure clears rather than forfeiting the whole job. When the
pressure does not clear within this window (the fault recurs, or the stage stays undispatchable past it), the
job is re-routed to the monolithic path instead of retrying forever. Sized to the stage patience window."""

_SAMPLING_LIVENESS_GRACE_SECONDS = 5.0
"""How long a ledgered sample whose sampler reports idle on the device is tolerated before it counts as stale.

A sample booked into :data:`_active_sampling_peaks` is normally backed by a running sampler (its process
reports busy from preload through the sampling steps). If that process instead reports idle (finished, or never
started) while its ledger entry lingers, the sample result was lost or never produced, so the entry is holding
device headroom nothing is using. This grace filters the brief window between dispatching the sample and the
child reporting it busy, so a just-dispatched sample is never mistaken for a stalled one; past it, an idle
sampler's ledger entry is provably not live sampling and is cleared by the fast escalation."""

_GATE_DEFER_SANITY_SECONDS = 180.0
"""How long a sample may be held by the concurrent-sampling gate before a lack of system-wide sampling
progress is treated as a stall rather than healthy backpressure, when the fast liveness escalation cannot fire.

A gate deferral is normally healthy: a ready sample waits for an in-flight sampling to free device headroom,
and must never age toward a fault. The primary release is the fast liveness escalation
(:meth:`_handle_gate_deferral`): the moment no sampling is verifiably in flight (every ledger entry's owner is
gone, its sample's dispatch launch is dead, or its sampler reports idle past the liveness grace) the stale
peaks are cleared and the sample re-admits within a tick. This far larger bound is the last resort for the case
the fast path cannot judge stale: the ledger entry's owner is present, its dispatch launch is live, and its
sampler still reports busy, yet no sample result has arrived anywhere in the whole window. That is either a
genuinely long sample or a sampler wedged busy; once this bound elapses the provably-dead entries are cleared
and the job is re-admitted or aged into the normal patience machinery, so no ledger state can hold a job
indefinitely."""


class _DispatchOutcome(StrEnum):
    """Why a tick's attempt to advance a job's stage did or did not send a control message.

    The patience machinery must tell a genuine stall (the job's next stage has no live role process) apart
    from healthy backpressure (a sample the concurrent-sampling gate deliberately deferred): the former ages
    toward a fault, the latter must never fault the job.
    """

    DISPATCHED = auto()
    """A stage control message was sent; the job advanced."""
    NO_ROLE = auto()
    """No live role process (or the send failed) could take the stage; this ages toward the patience fault."""
    GATE_DEFERRED = auto()
    """A ready sample was held back by the concurrent-sampling admission gate; healthy backpressure, never ages."""
    RESOURCE_DEFERRED = auto()
    """An encode or decode stage was withheld by the VRAM arbiter under device pressure; it does not age the
    no-role patience clock but arms the resource-defer window, so it either dispatches when the pressure clears
    or re-routes the job monolithically once the window elapses (the same fallback a resource-class stage fault
    takes)."""


class DisaggJobStage(StrEnum):
    """The stage a disaggregated job is currently waiting on."""

    AWAITING_SOURCE_LATENT = auto()
    """img2img/remix: the image lane is VAE-encoding the source image to a start latent."""
    AWAITING_CONDITIONING = auto()
    """The encode service is producing the positive/negative CONDITIONING."""
    SAMPLING = auto()
    """A sampler is producing the LATENT from injected conditioning."""
    AWAITING_LATENT_DECODE = auto()
    """The image lane is decoding (and post-processing) the LATENT to final images."""
    DONE = auto()
    """The job's images are ready and have been handed off."""


@dataclass
class _DisaggJobState:
    """The parent-held state of one in-flight disaggregated job.

    Every intermediate is held here so a stage process's death loses only the executing stage: the
    parent re-dispatches the stalled stage from this state once a replacement role process appears.
    """

    job_info: HordeJobInfo
    stage: DisaggJobStage
    needs_source_latent: bool
    positive_conditioning_bytes: bytes | None = None
    negative_conditioning_bytes: bytes | None = None
    source_latent_bytes: bytes | None = None
    dispatched_to: tuple[int, int] | None = None
    """The stage's dispatch target as ``(process_id, process_launch_identifier)`` (None when idle awaiting
    dispatch).

    Carries the launch identifier, not the bare process id, so an id-reusing replacement (a watchdog installs
    a new launch under the same slot id) is detected as a retirement rather than mistaken for the still-live
    process the stage was actually sent to. Results only ever arrive from the exact launch the stage was
    dispatched to."""
    pinned_sampler: tuple[int, int] | None = None
    """The inference process pinned as this job's sampler, as ``(process_id, process_launch_identifier)``.

    Chosen by the scheduler at admission and booked (reserved out of the availability pool) from registration
    until sampling finishes, so the scheduler cannot double-book it. Cleared to None when the pinned launch is
    retired/crashed/replaced, so the next dispatch re-resolves to whichever live process now holds the model.
    The launch identifier distinguishes the pinned process from an id-reusing cold replacement that never held
    the preloaded model."""
    first_stalled_at: float | None = None
    """When the job first had no role process for its next stage; anchors the patience window."""
    gate_deferred_since: float | None = None
    """When the job's ready sample was first, and has been continuously, held by the concurrent-sampling gate.

    Set on the first gate deferral and cleared the moment the job dispatches (or its stage changes), so it
    measures an unbroken deferral run. Anchors the :data:`_GATE_DEFER_SANITY_SECONDS` stall bound that keeps a
    leaked or stale sampling ledger from deferring the job forever."""
    sample_dispatched_at: float | None = None
    """When this job's in-flight sample was dispatched (booked into :data:`_active_sampling_peaks`).

    Set when the sample is admitted and cleared when its peak is released, so it spans exactly the interval a
    ledger entry exists for this job. The fast liveness escalation reads it to grace the brief post-dispatch
    window before the sampler reports busy, so a just-dispatched sample is never judged an idle (stale) one."""
    resource_defer_started_at: float | None = None
    """When the current stage first hit a resource-class (device out-of-memory) fault; anchors the defer window.

    Set on the first resource-class fault of a stage and cleared when the stage advances (a non-fault result),
    so it spans only an active device-pressure episode. While set the stage is retried; once the window
    (:data:`_RESOURCE_DEFER_SECONDS`) elapses without the pressure clearing, the job is re-routed monolithically."""


@dataclass(frozen=True)
class DisaggregatedFault:
    """Why a disaggregated job faulted, plus which process the fault is attributed to.

    Threaded from the orchestrator into the completion hand-off so the parent's synthetic result carries a
    real reason (not a blank ``info``) and the faulting process id (not a wrong default). ``reason`` is the
    child's exception summary for a stage fault, or the orchestrator's own text for a parent-side fault (a
    patience age-out, a gate-deferral stall). ``faulted_process_id`` is the child that produced the fault, or
    the pinned sampler for a parent-side fault, and is None only when neither is known.
    """

    reason: str
    faulted_process_id: int | None = None


class DisaggregationOrchestrator:
    """Drive disaggregated jobs stage by stage across the encode service, samplers, and image lane."""

    def __init__(
        self,
        *,
        find_encode_service: Callable[[], HordeProcessInfo | None],
        find_sampler: Callable[[str], HordeProcessInfo | None],
        find_image_lane: Callable[[], HordeProcessInfo | None],
        loader_identity: Callable[[HordeJobInfo], HordeStageModelMixin],
        on_images_ready: Callable[
            [HordeJobInfo, list[HordeImageResult], GENERATION_STATE, DisaggregatedFault | None],
            None,
        ],
        find_process_by_id: Callable[[int], HordeProcessInfo | None] = lambda _pid: None,
        reserve_sampler_process: Callable[[int], None] = lambda _pid: None,
        release_sampler_process: Callable[[int], None] = lambda _pid: None,
        on_sampling_complete: Callable[[HordeJobInfo], None] = lambda _job_info: None,
        reroute_monolithic: Callable[[HordeJobInfo], None] = lambda _job_info: None,
        estimate_sampling_peak_mb: Callable[[HordeJobInfo], float | None] = lambda _job_info: None,
        estimate_decode_spike_mb: Callable[[HordeJobInfo], float | None] = lambda _job_info: None,
        observe_sampling_peak: Callable[[HordeJobInfo, float], None] = lambda _job_info, _peak_mb: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wire the orchestrator to its role-process finders and completion hand-off.

        Args:
            find_encode_service: Returns an available encode-service process, or None.
            find_sampler: Given a model name, returns a sampler holding the model, or None. Used only to
                re-resolve a job's sampler after its pinned process crashes; the pinned process is used for
                the normal dispatch.
            find_image_lane: Returns an available image-lane process, or None.
            loader_identity: Builds the model-identity fields (name/ckpt/flags) a stage loads a subset of.
            on_images_ready: Called with a job's final images, state, and (on a fault) a
                :class:`DisaggregatedFault` carrying the reason and the faulting process id when its decode
                completes or it faults (hands off to the existing safety/submit flow). The fault is None on a
                successful completion.
            find_process_by_id: Resolves a process id to its live process info, or None if it is gone. Used
                to send the sample stage to the pinned sampler.
            reserve_sampler_process: Marks a process id booked as a pinned sampler (skipped by availability).
            release_sampler_process: Releases a pinned-sampler reservation (returns it to the pool).
            on_sampling_complete: Called with a job when its sampling finishes, so the tracker can free the
                inference slot (move the job to the decoding stage) while the job stays in-flight.
            reroute_monolithic: Called with a job whose stage kept hitting resource-class faults past the defer
                window, to return it to the monolithic inference path (the job stays owned/tracked throughout).
            estimate_sampling_peak_mb: Returns a job's estimated sampling-phase activation peak (MB, a whole-
                process weights-plus-activation figure), or None when no estimate is available. Consulted by the
                concurrent-sampling gate; a None estimate never blocks a dispatch.
            estimate_decode_spike_mb: Returns a job's estimated VAE tiled-decode activation spike (MB), or
                None when unavailable. Charged by the arbiter's decode gate; None
                prices the decode as unpriced, so the gate then only withholds a decode onto an already
                over-committed card.
            observe_sampling_peak: Called with a job and the pinned sampler process's measured peak reserved
                (MB) when its sample stage completes successfully, so the learned-footprint store records the
                disaggregated sampling peak the monolithic observation seam cannot attribute. A no-op by default
                (standalone tests) and never called with a non-positive peak.
            clock: Monotonic clock (overridable for the load simulator's virtual time).
        """
        self._find_encode_service = find_encode_service
        self._find_sampler = find_sampler
        self._find_image_lane = find_image_lane
        self._loader_identity = loader_identity
        self._on_images_ready = on_images_ready
        self._find_process_by_id = find_process_by_id
        self._reserve_sampler_process = reserve_sampler_process
        self._release_sampler_process = release_sampler_process
        self._on_sampling_complete = on_sampling_complete
        self._reroute_monolithic = reroute_monolithic
        self._estimate_sampling_peak_mb = estimate_sampling_peak_mb
        self._estimate_decode_spike_mb = estimate_decode_spike_mb
        self._observe_sampling_peak = observe_sampling_peak
        self._clock = clock
        self._jobs: dict[str, _DisaggJobState] = {}
        # The concurrent-sampling admission ledger: each in-flight sample stage's estimated activation peak
        # (MB), keyed by job key. An entry is added when a sample dispatch is admitted and removed on every
        # exit path (sample result, job finish, re-route, pin release, stage-process retirement), so a crashed
        # sampler always frees its headroom. Its emptiness is what makes the gate arbitrate only the SECOND-or-
        # later concurrent sampling: a sole sampling is always admitted (the monolithic status quo).
        self._active_sampling_peaks: dict[str, float] = {}
        # When a sample result last arrived anywhere (success or fault). Seeded to now so a leaked ledger with
        # no sampling ever in flight can still trip the gate-deferral sanity bound. Refreshed on every sample
        # result, so a healthy pipeline (results within the bound) never escalates a gate deferral to a stall.
        self._last_sample_result_at: float = self._clock()
        # The single VRAM arbiter, injected by the manager. It is the deciding authority for the
        # concurrent-sampling gate and for the encode/decode dispatch gates. None until wired (and in tests),
        # where every gate admits on missing telemetry.
        self._vram_arbiter: VramArbiter | None = None

    def set_vram_arbiter(self, arbiter: VramArbiter) -> None:
        """Inject the single VRAM arbiter: the authority for the concurrent-sampling and encode/decode gates."""
        self._vram_arbiter = arbiter

    def active_sampling_peaks_total_mb(self) -> float:
        """Return the summed in-flight sampling peaks (MB), so the cycle snapshot can charge them once."""
        return sum(self._active_sampling_peaks.values())

    def active_sampling_peaks_snapshot(self) -> dict[str, float]:
        """Return a copy of the in-flight sampling peaks (MB) keyed by job id, for stall diagnostics."""
        return dict(self._active_sampling_peaks)

    def pinned_job_id_for_process(self, process_id: int) -> str | None:
        """Return the job id whose sampler is pinned to ``process_id``, or None if no job pins it.

        Lets the dispatch-stall classifier attribute a monolithic head whose model is resident only on a
        pinned sampler lane to the disaggregated job holding that lane.
        """
        for key, state in self._jobs.items():
            if state.pinned_sampler is not None and state.pinned_sampler[0] == process_id:
                return key
        return None

    # -- registration ------------------------------------------------------------------------------

    def register(
        self,
        job_info: HordeJobInfo,
        *,
        needs_source_latent: bool,
        pinned_sampler_process_id: int,
        pinned_sampler_launch_identifier: int = 0,
    ) -> None:
        """Admit a job into the disaggregated pipeline at its first stage, pinning its sampler.

        The scheduler chose ``pinned_sampler_process_id`` (the process it preloaded this job's model onto)
        and routes the job here in place of dispatching monolithic inference to it. The pin records the
        process's ``pinned_sampler_launch_identifier`` too, so an id-reusing replacement of that slot is later
        detected as a retirement rather than being sampled on as a cold process that never held the model. The
        process is reserved out of the availability pool immediately so the scheduler cannot double-book it
        while the encode service produces conditioning; the reservation is released once sampling finishes
        (early release) or the job faults/retires.

        img2img/remix jobs start by VAE-encoding the source (AWAITING_SOURCE_LATENT) while the encode
        service runs in parallel; txt2img jobs start straight at AWAITING_CONDITIONING.
        """
        key = self._key(job_info)
        if key in self._jobs:
            return
        initial_stage = (
            DisaggJobStage.AWAITING_SOURCE_LATENT if needs_source_latent else DisaggJobStage.AWAITING_CONDITIONING
        )
        self._jobs[key] = _DisaggJobState(
            job_info=job_info,
            stage=initial_stage,
            needs_source_latent=needs_source_latent,
            pinned_sampler=(pinned_sampler_process_id, pinned_sampler_launch_identifier),
        )
        self._reserve_sampler_process(pinned_sampler_process_id)
        logger.debug(
            f"Disaggregation: registered job {key} at stage {initial_stage} pinned to sampler "
            f"{pinned_sampler_process_id}",
        )

    def has_job(self, job_info: HordeJobInfo) -> bool:
        """Whether a job is currently in the disaggregated pipeline."""
        return self._key(job_info) in self._jobs

    def _release_pin(self, state: _DisaggJobState) -> None:
        """Release a job's pinned sampler reservation (idempotent), returning the slot to the pool."""
        if state.pinned_sampler is not None:
            self._release_sampler_process(state.pinned_sampler[0])
            state.pinned_sampler = None

    def on_stage_process_retired(self, process_id: int) -> None:
        """Free any job whose current stage or pinned sampler was a now-retired process, for re-dispatch.

        A stage process dying (crash, replacement) never produces its result, so its jobs would sit
        dispatched forever. Clearing their dispatch marker lets the next :meth:`tick` re-dispatch the
        stalled stage from the held intermediates onto a replacement role process. The completed
        earlier stages are untouched: only the executing stage is redone. If the retired process was a job's
        pinned sampler, the pin is released so the next sample dispatch re-resolves to whichever live process
        now holds the model (the reservation is also dropped so the dead pid does not leak as booked, including
        onto an id-reusing replacement).

        Keyed on the bare process id (every launch of a genuinely retired slot is gone), so it is safe for both
        a true crash-and-remove and an id-reusing replacement; :meth:`reconcile_retired_processes` supplies the
        launch-aware detection that decides *when* a still-occupied slot id counts as retired.
        """
        for state in self._jobs.values():
            if state.pinned_sampler is not None and state.pinned_sampler[0] == process_id:
                logger.warning(
                    f"Disaggregation: pinned sampler {process_id} retired for {self._key(state.job_info)}; "
                    "will re-resolve the sampler on the next dispatch",
                )
                self._release_pin(state)
            if state.dispatched_to is not None and state.dispatched_to[0] == process_id:
                logger.warning(
                    f"Disaggregation: stage process {process_id} retired mid-stage for "
                    f"{self._key(state.job_info)} ({state.stage}); will re-dispatch from held state",
                )
                state.dispatched_to = None
                # A sampler retired mid-sampling never returns its result, so free its ledgered peak here
                # (idempotent for non-sampling stages) or its headroom would leak until the job faults.
                self._release_sampling_peak(state)

    def reconcile_retired_processes(self, alive_launches: set[tuple[int, int]]) -> None:
        """Re-dispatch any job whose stage process or pinned sampler is no longer the live launch.

        A stage process that crashed (removed from the map) or was watchdog-replaced (a new launch installed
        under the same slot id) never sends the result the job is waiting on. Comparing the full
        ``(process_id, process_launch_identifier)`` of each job's dispatch target and pin against the live
        launches detects both: a target absent from the live set, and a target whose slot id is live but under
        a *different* launch (the id-reuse case a bare-id check is blind to). Either frees the job so the fault
        story ("stage death loses only the executing stage") holds, and drops a pin/reservation stranded on a
        cold replacement. Called each tick with the currently-live ``(id, launch)`` pairs.
        """
        for state in list(self._jobs.values()):
            if state.dispatched_to is not None and state.dispatched_to not in alive_launches:
                self.on_stage_process_retired(state.dispatched_to[0])
            if state.pinned_sampler is not None and state.pinned_sampler not in alive_launches:
                self.on_stage_process_retired(state.pinned_sampler[0])

    def release_job(self, job_id: object) -> None:
        """Drop all held state for a job that left the system by a path outside this orchestrator's flow.

        A job the tracker no longer tracks (punted by the orphaned-in-progress watchdog, faulted by the
        save-our-ship give-up, or otherwise force-released) must not stay held here: its pin, reservation, and
        sampling-ledger entry would leak, and a later re-registration under the same id would be silently
        dropped, since :meth:`register` is a no-op for an already-held job. This is the single seam every such
        external exit calls so the orchestrator never holds a job the tracker has released. Idempotent, and a
        no-op for a job it does not hold.
        """
        key = str(job_id)
        state = self._jobs.get(key)
        if state is None:
            return
        logger.warning(
            f"Disaggregation: releasing held state for job {key} (left the system via an external path); "
            f"dropping its pin and any sampling-ledger entry.",
        )
        self._release_pin(state)
        self._release_sampling_peak(state)
        self._jobs.pop(key, None)

    # -- scheduler tick ----------------------------------------------------------------------------

    def tick(self) -> None:
        """(Re)dispatch every job whose next stage has an available role process; age out the stalled.

        Dispatch is idempotent: a job already dispatched to a live process is skipped. A job whose role
        process is gone (crashed) is re-dispatched from held state; one that stays undispatchable past
        the patience window is faulted so the horde reissues it.
        """
        now = self._clock()
        for state in list(self._jobs.values()):
            if state.dispatched_to is not None:
                continue
            if (
                state.resource_defer_started_at is not None
                and now - state.resource_defer_started_at > _RESOURCE_DEFER_SECONDS
            ):
                # A stage deferring under device pressure could not clear it within the window (it keeps
                # faulting resource-class, or stays undispatchable): re-route the whole job monolithically
                # rather than retrying forever.
                self._reroute_to_monolithic(state)
                continue
            outcome = self._try_dispatch(state)
            if outcome == _DispatchOutcome.DISPATCHED:
                state.first_stalled_at = None
                state.gate_deferred_since = None
            elif outcome == _DispatchOutcome.GATE_DEFERRED:
                self._handle_gate_deferral(state, now)
            elif outcome == _DispatchOutcome.RESOURCE_DEFERRED:
                # The arbiter withheld an encode/decode under device pressure: this is not a missing role, so it
                # must not age the no-role patience clock. The resource-defer window armed by the dispatch is
                # reconciled at the top of this loop, which re-routes the job monolithically once it elapses.
                state.first_stalled_at = None
                state.gate_deferred_since = None
            else:  # NO_ROLE: the next stage genuinely has no live role process to take it.
                state.gate_deferred_since = None
                if state.first_stalled_at is None:
                    state.first_stalled_at = now
                elif now - state.first_stalled_at > _STAGE_PATIENCE_SECONDS:
                    self._fault_and_finish(state, reason=f"no role process for stage {state.stage}")

    def _handle_gate_deferral(self, state: _DisaggJobState, now: float) -> None:
        """Serialize samplers on genuine backpressure, but escalate a deferral no live sampling justifies.

        The concurrent-sampling gate may serialize samplers, it must never deadlock. A sample it declines is
        healthy backpressure only while a sampling is genuinely in flight to free the headroom it waits on; the
        deferral is then bounded by that job's completion and must not age toward the patience fault. The primary
        escalation is the fast liveness path: the moment no sampling is verifiably in flight (every ledger entry
        is provably not live sampling per :meth:`_sampling_ledger_entry_is_stale`) the stale peaks are cleared so
        the sample re-admits within a tick, because a candidate that fits alone on an idle card must always run.
        Only when a live-looking ledger yields no system-wide progress for the whole :data:`_GATE_DEFER_SANITY_SECONDS`
        window does the far larger sanity bound act as a last resort.
        """
        if state.gate_deferred_since is None:
            state.gate_deferred_since = now
        # Fast liveness escalation: reclaim any headroom no verifiably-live sampling is using. Safe every tick,
        # because a genuinely running sampler (its owner present, its sample dispatched to a live launch, its
        # process reporting busy) is never judged stale, so the gate's protection (never a second non-fitting
        # concurrent sampler) is intact.
        if not self._sampling_verifiably_in_flight(now):
            self._break_gate_deferral_stall(state, now, escalation="no live sampling in flight")
            return
        # A genuine sampling is in flight. The sanity bound is the last resort for a ledger that looks live yet
        # yields no system-wide progress for its whole window; short of it, this is healthy backpressure.
        escalating_on_sanity = (
            now - state.gate_deferred_since > _GATE_DEFER_SANITY_SECONDS
            and now - self._last_sample_result_at > _GATE_DEFER_SANITY_SECONDS
        )
        if not escalating_on_sanity:
            # Healthy backpressure bounded by the in-flight sampling's completion: keep it clear of the no-role
            # patience clock and leave the resource-defer machinery untouched (a gate deferral is not a
            # device-pressure fault, so it must not consume the one-defer budget).
            state.first_stalled_at = None
            return
        self._break_gate_deferral_stall(state, now, escalation="sanity bound with no sampling progress")

    def _break_gate_deferral_stall(self, state: _DisaggJobState, now: float, *, escalation: str) -> None:
        """Clear provably-stale sampling-ledger peaks, then re-admit or age the job that was stuck behind them.

        Every ledger entry that is not verifiably live sampling (its owning job is gone, its sample's dispatch
        launch is dead, or its sampler reports idle past the liveness grace) is a peak reserving headroom nothing
        is using: those are cleared. If that empties the ledger the gate re-admits the job on the next tick. If
        entries remain for live sampling processes the ledger is not the problem, so the job is aged through
        ``first_stalled_at`` and the normal patience fault/reroute machinery applies rather than deferring forever.
        """
        stale_before = dict(self._active_sampling_peaks)
        for ledger_key in list(self._active_sampling_peaks):
            if self._sampling_ledger_entry_is_stale(self._jobs.get(ledger_key), now):
                self._active_sampling_peaks.pop(ledger_key, None)

        if not self._active_sampling_peaks:
            if stale_before:
                logger.warning(
                    f"Disaggregation: sample for {self._key(state.job_info)} was gate-deferred ({escalation}); "
                    f"cleared the stale sampling ledger {stale_before} and will re-admit it.",
                )
            # Reset the deferral anchor so the next tick re-attempts dispatch against the now-clean ledger.
            state.gate_deferred_since = None
            return

        # Live sampling entries remain: this is not a ledger leak. Route the job through patience so it faults
        # or reroutes rather than deferring forever.
        if state.first_stalled_at is None:
            logger.warning(
                f"Disaggregation: sample for {self._key(state.job_info)} was gate-deferred ({escalation}), but "
                f"the sampling ledger {self._active_sampling_peaks} still references live samplers; aging it "
                "into the patience path.",
            )
            state.first_stalled_at = now
        elif now - state.first_stalled_at > _STAGE_PATIENCE_SECONDS:
            self._fault_and_finish(state, reason="gate-deferred with no sampling progress")

    def _sampling_verifiably_in_flight(self, now: float) -> bool:
        """Whether any ledgered sample is backed by a genuinely running sampler.

        True when at least one ledger entry is not stale (its owner is present and sampling, its sample was
        dispatched to a live launch, and its sampler reports busy on the device). When this is False the whole
        ledger is holding headroom no live sampling is using, so a gate deferral behind it can be escalated at
        once rather than waiting on the sanity bound.
        """
        return any(
            not self._sampling_ledger_entry_is_stale(self._jobs.get(ledger_key), now)
            for ledger_key in self._active_sampling_peaks
        )

    def _sampling_ledger_entry_is_stale(self, owner: _DisaggJobState | None, now: float) -> bool:
        """Whether a sampling-ledger entry is not backed by a verifiably-live sampling.

        Stale when the owning job is gone, is no longer in the sampling stage, has no dispatch target, or its
        sample was dispatched to a launch that is no longer live (a crash or an id-reusing replacement). It is
        also stale when the launch is live but its process reports idle (not busy) past the liveness grace: a
        running sampler reports busy from preload through its steps, so an idle sampler whose entry lingers has
        lost or never produced its result. The grace covers only the brief post-dispatch window before the child
        reports busy, so a just-dispatched sample is never mistaken for a stalled one.
        """
        if owner is None or owner.stage != DisaggJobStage.SAMPLING or owner.dispatched_to is None:
            return True
        pid, launch = owner.dispatched_to
        process = self._find_process_by_id(pid)
        if process is None or process.process_launch_identifier != launch:
            return True
        if process.is_process_busy():
            return False
        # Launch is live but the device shows this sampler idle: stale only once past the grace, so a sample
        # dispatched this tick (child not yet reporting busy) is not reclaimed out from under a live sampling.
        dispatched_at = owner.sample_dispatched_at
        return dispatched_at is not None and now - dispatched_at > _SAMPLING_LIVENESS_GRACE_SECONDS

    def _try_dispatch(self, state: _DisaggJobState) -> _DispatchOutcome:
        """Dispatch the job's current stage to a role process; report whether it sent, stalled, or was gated."""
        if state.stage == DisaggJobStage.AWAITING_SOURCE_LATENT:
            return self._dispatch_vae_encode(state)
        if state.stage == DisaggJobStage.AWAITING_CONDITIONING:
            return self._dispatch_text_encode(state)
        if state.stage == DisaggJobStage.SAMPLING:
            return self._dispatch_sample(state)
        if state.stage == DisaggJobStage.AWAITING_LATENT_DECODE:
            return self._dispatch_decode(state)
        return _DispatchOutcome.NO_ROLE

    # -- per-stage dispatch ------------------------------------------------------------------------

    def _dispatch_text_encode(self, state: _DisaggJobState) -> _DispatchOutcome:
        service = self._find_encode_service()
        if service is None:
            return _DispatchOutcome.NO_ROLE
        # Encode is priced as unpriced (None): the gate only withholds a dispatch onto an already over-committed
        # card, never charges a phantom encode cost.
        if not self._admit_stage(VramRequestKind.DISAGG_ENCODE, state, candidate_delta_mb=None):
            return self._defer_stage_for_pressure(state, "text-encode")
        identity = self._loader_identity(state.job_info)
        message = HordeTextEncodeControlMessage(
            **identity.model_dump(),
            job_id=state.job_info.sdk_api_job_info.id_,
            sdk_api_job_info=state.job_info.sdk_api_job_info,
        )
        return _DispatchOutcome.DISPATCHED if self._send(service, message, state) else _DispatchOutcome.NO_ROLE

    def _dispatch_vae_encode(self, state: _DisaggJobState) -> _DispatchOutcome:
        lane = self._find_image_lane()
        if lane is None:
            return _DispatchOutcome.NO_ROLE
        if not self._admit_stage(VramRequestKind.DISAGG_ENCODE, state, candidate_delta_mb=None):
            return self._defer_stage_for_pressure(state, "vae-encode")
        message = HordeVaeEncodeControlMessage(
            **self._loader_identity(state.job_info).model_dump(),
            job_id=state.job_info.sdk_api_job_info.id_,
            sdk_api_job_info=state.job_info.sdk_api_job_info,
        )
        return _DispatchOutcome.DISPATCHED if self._send(lane, message, state) else _DispatchOutcome.NO_ROLE

    def _dispatch_sample(self, state: _DisaggJobState) -> _DispatchOutcome:
        if state.positive_conditioning_bytes is None or state.negative_conditioning_bytes is None:
            return _DispatchOutcome.NO_ROLE
        if state.needs_source_latent and state.source_latent_bytes is None:
            return _DispatchOutcome.NO_ROLE
        identity = self._loader_identity(state.job_info)
        sampler = self._resolve_sampler(state, identity.horde_model_name)
        if sampler is None:
            return _DispatchOutcome.NO_ROLE
        # Gate a second (or later) concurrent sampling against the device's static sampling headroom, so two
        # activation peaks never over-commit the card and drive it into WDDM demand-paging. Checked here, after
        # the inputs are ready and the sampler resolved (a real sampler is waiting), so a deferral is genuine
        # backpressure rather than a missing role.
        peak_mb = self._estimate_sampling_peak_mb(state.job_info)
        if not self._admit_concurrent_sampling(peak_mb):
            logger.debug(
                f"Disaggregation: deferring sample for {self._key(state.job_info)} "
                f"(peak ~{peak_mb:.0f} MB) until an in-flight sampling frees device headroom"
                if peak_mb is not None
                else f"Disaggregation: deferring sample for {self._key(state.job_info)} until headroom frees",
            )
            return _DispatchOutcome.GATE_DEFERRED
        message = HordeSampleControlMessage(
            **identity.model_dump(),
            slices=[
                SampleSliceSpec(
                    job_id=state.job_info.sdk_api_job_info.id_,
                    positive_conditioning_bytes=state.positive_conditioning_bytes,
                    negative_conditioning_bytes=state.negative_conditioning_bytes,
                    source_latent_bytes=state.source_latent_bytes,
                    sdk_api_job_info=state.job_info.sdk_api_job_info,
                ),
            ],
        )
        if not self._send(sampler, message, state):
            return _DispatchOutcome.NO_ROLE
        # Admitted: book its peak into the ledger so the next sampler is gated against the room this one takes.
        # A missing estimate books 0.0 (its presence still makes the ledger non-empty, so the gate arbitrates
        # subsequent samplings, but it reserves no headroom, per "never wedge on a missing estimate").
        self._active_sampling_peaks[self._key(state.job_info)] = peak_mb if peak_mb is not None else 0.0
        state.sample_dispatched_at = self._clock()
        return _DispatchOutcome.DISPATCHED

    def _admit_concurrent_sampling(self, peak_mb: float | None) -> bool:
        """Whether a sample may dispatch now, the VRAM arbiter deciding the concurrent-sampling memory question.

        The gate only ever arbitrates a SECOND-or-later concurrent sampling: a first-of-kind sampling (empty
        ledger) always admits, since a sole over-peak sampling is the monolithic status quo (the driver streams:
        slow but correct) and denying it would wedge a small card whose headroom is below one job's peak. For a
        later sampling the arbiter's :attr:`VramRequestKind.DISAGG_SAMPLE` verdict decides: a FITS admits, a
        DEFER withholds this sample so the caller returns it to the gate-deferral bookkeeping and it
        re-asks next tick. The live in-flight sampling total is passed with the request so a peak booked earlier
        in this same tick is counted before the cycle snapshot is next refrozen. No actuations run here (reclaim
        is single-owner, driven only by the preload path). When no cycle snapshot exists (the arbiter is unwired
        or cold), admit: the gate has always admitted on missing telemetry.
        """
        arbiter = self._vram_arbiter
        if arbiter is None or not arbiter.has_cycle:
            return True
        verdict = arbiter.evaluate(
            VramRequest(
                kind=VramRequestKind.DISAGG_SAMPLE,
                job_label="disagg_sample",
                baseline=None,
                device_index=None,
                sampling_peak_mb=peak_mb,
                first_of_kind=not self._active_sampling_peaks,
                active_sampling_peaks_total_mb=self.active_sampling_peaks_total_mb(),
            ),
        )
        return verdict.admits

    def _admit_stage(
        self,
        kind: VramRequestKind,
        state: _DisaggJobState,
        *,
        candidate_delta_mb: float | None,
    ) -> bool:
        """Whether an encode or decode stage may dispatch now, the VRAM arbiter deciding the memory question.

        A FITS verdict admits; a DEFER or DENY withholds the dispatch this tick so the caller arms the
        resource-defer window and the job either dispatches when the pressure clears or re-routes monolithically
        once the window elapses. A verdict is never turned into a fault. No actuations run here (reclaim is
        single-owner, driven only by the preload path). When no cycle snapshot exists (the arbiter is unwired or
        cold) the stage admits: every gate admits on missing telemetry.
        """
        arbiter = self._vram_arbiter
        if arbiter is None or not arbiter.has_cycle:
            return True
        verdict = arbiter.evaluate(
            VramRequest(
                kind=kind,
                job_label=self._key(state.job_info),
                baseline=None,
                device_index=None,
                candidate_delta_mb=candidate_delta_mb,
            ),
        )
        return verdict.admits

    def _defer_stage_for_pressure(self, state: _DisaggJobState, stage_label: str) -> _DispatchOutcome:
        """Withhold an arbiter-deferred encode/decode this tick, arming the resource-defer reroute window.

        Reuses the resource-class fault window: the first deferral of a stage anchors
        :data:`_RESOURCE_DEFER_SECONDS`, and the top of :meth:`tick` re-routes the job monolithically once that
        window elapses without the pressure clearing. Until then the stage is re-attempted each tick, so a
        transient over-commit is ridden out and a persistent one falls back to the monolithic path rather than
        wedging the pinned-idle sampler.
        """
        if state.resource_defer_started_at is None:
            state.resource_defer_started_at = self._clock()
            logger.debug(
                f"Disaggregation: {stage_label} for {self._key(state.job_info)} withheld by the VRAM arbiter "
                f"under device pressure; deferring up to {_RESOURCE_DEFER_SECONDS:.0f}s before re-routing "
                "monolithically.",
            )
        return _DispatchOutcome.RESOURCE_DEFERRED

    def _release_sampling_peak(self, state: _DisaggJobState) -> None:
        """Drop a job's in-flight sampling peak from the ledger (idempotent), returning its headroom."""
        self._active_sampling_peaks.pop(self._key(state.job_info), None)
        state.sample_dispatched_at = None

    def _resolve_sampler(self, state: _DisaggJobState, horde_model_name: str) -> HordeProcessInfo | None:
        """Resolve the process to sample on: the pin normally, a re-resolution after the pin was cleared.

        The scheduler pinned this job's sampler at admission, so the normal path sends the sample stage to
        that exact process. If the pin is still set but the process (that exact launch) cannot be resolved, the
        pinned sampler vanished or was replaced between ticks: return None (do not silently re-pick, and never
        sample on an id-reusing cold replacement that never held the model), and let the reconcile pass clear
        the pin so the next tick re-resolves cleanly. When the pin has already been cleared (its process
        crashed and was reaped, or was retired), re-resolve to whichever live process now holds the model and
        re-pin to it, so the reservation follows the sampler.
        """
        if state.pinned_sampler is not None:
            pid, launch = state.pinned_sampler
            sampler = self._find_process_by_id(pid)
            if sampler is None or sampler.process_launch_identifier != launch:
                logger.error(
                    f"Disaggregation: pinned sampler {state.pinned_sampler} for "
                    f"{self._key(state.job_info)} is no longer the live launch; deferring until reconcile "
                    "re-resolves it",
                )
                return None
            return sampler
        sampler = self._find_sampler(horde_model_name)
        if sampler is None:
            return None
        state.pinned_sampler = (sampler.process_id, sampler.process_launch_identifier)
        self._reserve_sampler_process(sampler.process_id)
        logger.debug(
            f"Disaggregation: re-resolved sampler for {self._key(state.job_info)} to process "
            f"{sampler.process_id} after its pinned sampler was lost",
        )
        return sampler

    def _dispatch_decode(self, state: _DisaggJobState) -> _DispatchOutcome:
        lane = self._find_image_lane()
        if lane is None:
            return _DispatchOutcome.NO_ROLE
        latent = state.source_latent_bytes  # reused field holds the sampled latent by this stage
        if latent is None:
            return _DispatchOutcome.NO_ROLE
        # Decode is priced with the job's tiled-decode activation spike, so a decode is withheld when it would
        # over-commit the card alongside an in-flight sampling rather than driving both into paging. Any
        # requested post-processing runs on the dedicated post-processing lane after the decode completes.
        if not self._admit_stage(
            VramRequestKind.DISAGG_DECODE,
            state,
            candidate_delta_mb=self._estimate_decode_spike_mb(state.job_info),
        ):
            return self._defer_stage_for_pressure(state, "vae-decode")
        message = HordeVaeDecodeControlMessage(
            **self._loader_identity(state.job_info).model_dump(),
            job_id=state.job_info.sdk_api_job_info.id_,
            sdk_api_job_info=state.job_info.sdk_api_job_info,
            latent_bytes=latent,
        )
        return _DispatchOutcome.DISPATCHED if self._send(lane, message, state) else _DispatchOutcome.NO_ROLE

    def _send(self, process: HordeProcessInfo, message: HordeControlMessage, state: _DisaggJobState) -> bool:
        """Send a stage control message to a role process, marking the job dispatched on success."""
        sent = process.safe_send_message(message)
        if sent:
            state.dispatched_to = (process.process_id, process.process_launch_identifier)
            logger.debug(f"Disaggregation: dispatched {type(message).__name__} for {self._key(state.job_info)}")
        return sent

    # -- result handling ---------------------------------------------------------------------------

    async def handle_stage_result(self, message: HordeProcessMessage) -> None:
        """Advance a job's DAG on a stage result, dispatching the next stage or handing off images."""
        if isinstance(message, HordeTextEncodeResultMessage):
            self._on_text_encode_result(message)
        elif isinstance(message, HordeVaeEncodeResultMessage):
            self._on_vae_encode_result(message)
        elif isinstance(message, HordeSampleResultMessage):
            self._on_sample_result(message)
        elif isinstance(message, HordeVaeDecodeResultMessage):
            self._on_vae_decode_result(message)

    def _lookup(self, job_id: object) -> _DisaggJobState | None:
        return self._jobs.get(str(job_id))

    def _on_text_encode_result(self, message: HordeTextEncodeResultMessage) -> None:
        state = self._lookup(message.job_id)
        if state is None:
            return
        state.dispatched_to = None
        if message.state == GENERATION_STATE.faulted or message.positive_conditioning_bytes is None:
            if self._defer_or_reroute_resource_fault(message.fault_is_resource_class, state):
                return
            self._fault_and_finish(
                state,
                reason=message.fault_reason or "text-encode faulted",
                faulted_process_id=message.process_id,
            )
            return
        state.resource_defer_started_at = None
        state.positive_conditioning_bytes = message.positive_conditioning_bytes
        state.negative_conditioning_bytes = message.negative_conditioning_bytes
        # img2img may still be encoding the source latent; only advance to SAMPLING once both are in.
        if state.needs_source_latent and state.source_latent_bytes is None:
            state.stage = DisaggJobStage.AWAITING_CONDITIONING  # stay pending the source latent
        else:
            state.stage = DisaggJobStage.SAMPLING

    def _on_vae_encode_result(self, message: HordeVaeEncodeResultMessage) -> None:
        state = self._lookup(message.job_id)
        if state is None:
            return
        state.dispatched_to = None
        if message.state == GENERATION_STATE.faulted or message.latent_bytes is None:
            if self._defer_or_reroute_resource_fault(message.fault_is_resource_class, state):
                return
            self._fault_and_finish(
                state,
                reason=message.fault_reason or "vae-encode faulted",
                faulted_process_id=message.process_id,
            )
            return
        state.resource_defer_started_at = None
        state.source_latent_bytes = message.latent_bytes
        if state.positive_conditioning_bytes is not None:
            state.stage = DisaggJobStage.SAMPLING
        else:
            state.stage = DisaggJobStage.AWAITING_CONDITIONING

    def _on_sample_result(self, message: HordeSampleResultMessage) -> None:
        # A sample result (any job, success or fault) is proof sampling is progressing system-wide; refresh the
        # anchor so a healthy pipeline never trips the gate-deferral sanity bound.
        self._last_sample_result_at = self._clock()
        for result in message.results:
            state = self._lookup(result.job_id)
            if state is None:
                continue
            state.dispatched_to = None
            # The sample stage has ended (success or fault): free its ledgered peak so the headroom it held
            # returns for the next sampling, on both the success path below and either fault path.
            self._release_sampling_peak(state)
            if result.state == GENERATION_STATE.faulted or result.latent_bytes is None:
                # A resource-class sample fault is deferred with the pin held (the retry re-samples on the
                # same pinned process) rather than forfeiting the job; a genuine fault releases the pin and
                # forfeits.
                if self._defer_or_reroute_resource_fault(message.fault_is_resource_class, state):
                    continue
                self._release_pin(state)
                self._fault_and_finish(state, reason="sample faulted", faulted_process_id=message.process_id)
                continue
            state.resource_defer_started_at = None
            # Record this sampler's measured peak before the pin is released: the sampler process that produced
            # the result is the pinned sampler, and its latest reported peak reserved is the disaggregated
            # SAMPLE-stage footprint the monolithic observation seam cannot attribute. Read here (pre-release)
            # so the pinned pid is still resolvable; the store ignores a non-positive or absent reading.
            sampler_process = self._find_process_by_id(message.process_id)
            if sampler_process is not None and sampler_process.process_peak_reserved_mb is not None:
                self._observe_sampling_peak(state.job_info, float(sampler_process.process_peak_reserved_mb))
            # Sampling is done: release the sampler slot so the scheduler can admit the next job onto it (the
            # pipeline-overlap win), while this job stays in-flight through decode. Done here, on the result,
            # rather than at sample dispatch (would double-book the running slot) or at completion (would
            # forfeit the overlap).
            self._release_pin(state)
            self._on_sampling_complete(state.job_info)
            # The sampler's retained activation pool is NOT released here: an unconditional post-stage
            # release forces a gc pause plus a full pool rebuild on the next slice for every job, which
            # costs far more than the reservation it returns. Reclaim is on-demand only: the admission
            # arbiter's escalation ladder targets this process's cache when a competing demand needs it.
            # The sampled latent flows into the decode stage; reuse source_latent_bytes as the carrier.
            state.source_latent_bytes = result.latent_bytes
            state.stage = DisaggJobStage.AWAITING_LATENT_DECODE

    def _on_vae_decode_result(self, message: HordeVaeDecodeResultMessage) -> None:
        state = self._lookup(message.job_id)
        if state is None:
            return
        state.dispatched_to = None
        if message.state == GENERATION_STATE.faulted and self._defer_or_reroute_resource_fault(
            message.fault_is_resource_class,
            state,
        ):
            return
        state.resource_defer_started_at = None
        images = message.job_image_results or []
        succeeded = message.state != GENERATION_STATE.faulted
        fault = (
            None
            if succeeded
            else DisaggregatedFault(
                reason=message.fault_reason or "vae-decode faulted",
                faulted_process_id=message.process_id,
            )
        )
        self._finish(state, images, message.state, fault=fault)

    def _defer_or_reroute_resource_fault(self, fault_is_resource_class: bool, state: _DisaggJobState) -> bool:
        """Handle a resource-class stage fault by deferring the stage or re-routing the job. Return if handled.

        A resource-class fault (the stage was denied device VRAM under pressure) is not a genuine error, so
        the job is not forfeited. The first such fault of a stage anchors the defer window and leaves the job
        at its current stage: the caller has already cleared the dispatch marker, so the next ``tick`` re-runs
        the same stage as pressure clears. A recurrence once the window (:data:`_RESOURCE_DEFER_SECONDS`) has
        elapsed re-routes the whole job to the monolithic path. Returns False (not handled) when the fault is
        not resource-class, so the caller takes the genuine-fault forfeit path.
        """
        if not fault_is_resource_class:
            return False
        now = self._clock()
        if state.resource_defer_started_at is None:
            state.resource_defer_started_at = now
            logger.warning(
                f"Disaggregation: stage {state.stage} for {self._key(state.job_info)} hit a resource-class "
                f"fault; deferring up to {_RESOURCE_DEFER_SECONDS}s to retry as device pressure clears",
            )
            return True
        if now - state.resource_defer_started_at > _RESOURCE_DEFER_SECONDS:
            self._reroute_to_monolithic(state)
        return True

    def _reroute_to_monolithic(self, state: _DisaggJobState) -> None:
        """Return a job to the monolithic inference path after its stage kept failing resource-class.

        The job stays owned and tracked throughout: its pin is released and it is popped from the pipeline,
        then the injected ``reroute_monolithic`` returns it to the normal claim/dispatch path (latched so the
        re-claim runs monolithic). No images-faulted report is emitted; the job runs whole instead.
        """
        logger.warning(
            f"Disaggregation: re-routing job {self._key(state.job_info)} to monolithic inference; "
            f"resource-class stage faults persisted past the {_RESOURCE_DEFER_SECONDS}s defer window",
        )
        self._release_pin(state)
        self._release_sampling_peak(state)
        self._jobs.pop(self._key(state.job_info), None)
        self._reroute_monolithic(state.job_info)

    # -- completion / fault ------------------------------------------------------------------------

    def _fault_and_finish(self, state: _DisaggJobState, *, reason: str, faulted_process_id: int | None = None) -> None:
        """Fault a job, attributing it to the faulting child or (for a parent-side fault) the pinned sampler.

        ``faulted_process_id`` is the child that produced a stage fault; when it is None (a parent-side fault
        like a patience age-out or a gate-deferral stall) the job's pinned sampler is used, so the synthetic
        result never reports a wrong default process rather than the one actually implicated.
        """
        attributed_process_id = faulted_process_id
        if attributed_process_id is None and state.pinned_sampler is not None:
            attributed_process_id = state.pinned_sampler[0]
        logger.error(f"Disaggregation: faulting job {self._key(state.job_info)}: {reason}")
        self._finish(
            state,
            [],
            GENERATION_STATE.faulted,
            fault=DisaggregatedFault(reason=reason, faulted_process_id=attributed_process_id),
        )

    def _finish(
        self,
        state: _DisaggJobState,
        images: list[HordeImageResult],
        job_state: GENERATION_STATE,
        *,
        fault: DisaggregatedFault | None = None,
    ) -> None:
        state.stage = DisaggJobStage.DONE
        # Any still-held pin is released so a fault before/during sampling never leaks its sampler as booked.
        self._release_pin(state)
        # And any in-flight sampling peak is freed so a fault mid-sampling never leaks its device headroom.
        self._release_sampling_peak(state)
        self._jobs.pop(self._key(state.job_info), None)
        self._on_images_ready(state.job_info, images, job_state, fault)

    @staticmethod
    def _key(job_info: HordeJobInfo) -> str:
        return str(job_info.sdk_api_job_info.id_)
