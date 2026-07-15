"""Parent-side coordination of the ad-hoc auxiliary (LoRA/TI) prefetch pipeline.

At job pop the parent asks the dedicated download process to place a job's LoRAs and textual inversions on
disk while the job stays pending, so dispatch finds them already cached and the inference path
short-circuits its own download. This coordinator owns the three seams of that flow that live outside the
download process:

- the pop-time trigger, which computes the not-yet-known-cached auxiliary set (plus the eviction-pin set of
  every file any tracked job still references) and sends one prefetch request;
- the completion/failure wiring, which marks each cached file, clears a job's dispatch gate once its full
  auxiliary set is present, and faults a job promptly when a file it needs cannot be fetched; and
- a per-job deadline (derived from the configured download timeout) that faults a job whose prefetch never
  resolves, so a job is never left pending forever; and
- a periodic reconcile-and-pin-refresh step that re-requests any aux-bearing pending job left without an
  in-flight request (a retryable requeue, a lost result message, a restarted downloader) and sends a
  pins-only update whenever the set of still-referenced files changes, coalesced so an unchanged set is never
  re-sent.

It is torch-free and reads only job state and worker state, so it is unit-testable without a live worker.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.ai_horde_api.fields import GenerationID
from loguru import logger

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    AuxModelKind,
    AuxModelRef,
    AuxPrefetchEntry,
    AuxPrefetchOutcome,
    HordeAuxPrefetchResultMessage,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobFaultOrigin, JobStage, JobTracker
from horde_worker_regen.process_management.models.aux_download_backoff import STRIKE_DECAY_SECONDS

_PIN_STAGES = (JobStage.PENDING_INFERENCE, JobStage.INFERENCE_IN_PROGRESS, JobStage.PENDING_ANNOTATION)
"""Stages whose jobs still need their auxiliary files, so those files are pinned against eviction."""

_MAX_DEADLINE_DEFERRALS = 2
"""How many extra download-timeout budgets a job's deadline may be extended by while its files are still in
flight, so a genuinely-progressing (or unmeasurable) download is not faulted at the first deadline yet a
stalled one cannot defer forever. Two extensions cap total patience at three budgets (the original plus two)."""

_AUX_REFETCH_COOLDOWN_SECONDS = 15.0
"""Quiet gap after a failed prefetch of one reference before that same reference may be re-requested.

Contract: a transient failure that requeues a job with retries remaining must not immediately re-request the
same reference into the still-failing download path. Without this gap a sub-second failure is re-requested at
once, fails again while the class incident is still active (so the second failure is classified terminal), and
the two failures burn both of a job's inference attempts inside a second. The effective cooldown is
``min(_AUX_REFETCH_COOLDOWN_SECONDS, remaining class-backoff window)``: capped at this constant so a legitimate
retry is not delayed longer than needed, and never longer than the class's own suppression window (once that
window lapses the reference is fetchable again, so cooling it further would be pointless). While a job's only
uncached references are cooling it holds a live per-job deadline, so it stays bounded exactly as any unresolved
prefetch does and still counts as aux-held against deadlock detection."""


def _no_in_flight() -> dict[str, tuple[int, int]]:
    """Default in-flight provider: report nothing downloading (a coordinator wired without downloader status)."""
    return {}


class AuxPrefetchCoordinator:
    """Drives the pop-time prefetch request and the parent-side completion/failure/deadline wiring."""

    def __init__(
        self,
        *,
        job_tracker: JobTracker,
        state: WorkerState,
        prefetch_sender: Callable[[list[AuxPrefetchEntry], list[AuxModelRef]], None],
        download_timeout_provider: Callable[[], float],
        pin_sender: Callable[[list[AuxModelRef]], None] | None = None,
        in_flight_provider: Callable[[], dict[str, tuple[int, int]]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the coordinator.

        Args:
            job_tracker: The job tracker whose pending jobs drive prefetch and whose caches record readiness.
            state: Worker state, for the per-class auxiliary-download backoffs a failed prefetch feeds.
            prefetch_sender: Sends a prefetch control message ``(entries, pins)`` to the download process.
            download_timeout_provider: Returns the configured download timeout (seconds) for the per-job deadline.
            pin_sender: Sends a pins-only update (empty entries) to the download process, for the edge-triggered
                refresh that keeps the eviction-pin set current as jobs complete between pops. A no-op default
                keeps a coordinator wired without it inert on the pin-refresh path.
            in_flight_provider: Returns the ad-hoc-prefetch downloads the downloader shows in flight right now,
                as ``name -> (downloaded_bytes, total_bytes)``. Read at deadline expiry so a job whose file is
                still being placed on disk defers its fault rather than losing an alive-but-slow download. A
                default reporting nothing keeps a coordinator wired without downloader status faulting on time.
            clock: Wall-clock provider (injectable for tests).
        """
        self._job_tracker = job_tracker
        self._state = state
        self._prefetch_sender = prefetch_sender
        self._download_timeout_provider = download_timeout_provider
        self._pin_sender = pin_sender if pin_sender is not None else (lambda _pins: None)
        self._in_flight_provider = in_flight_provider if in_flight_provider is not None else _no_in_flight
        self._clock = clock
        # Per-job prefetch deadlines: a job whose auxiliary files have not all landed by its deadline is
        # faulted rather than left pending forever. Cleared when the job is prepared, faulted, or gone.
        self._deadlines: dict[GenerationID, float] = {}
        # How many extra budgets each job's deadline has already been extended by while its files stayed in
        # flight, capped at ``_MAX_DEADLINE_DEFERRALS`` so a stalled download cannot defer a fault forever.
        self._deferrals: dict[GenerationID, int] = {}
        # Jobs that have already logged their first deferral, so subsequent deferrals of the same job stay
        # silent (the reason is edge-triggered, not repeated every scan).
        self._deferral_logged: set[GenerationID] = set()
        # Per-reference re-fetch cooldown expiries: a reference whose prefetch just failed is not re-requested
        # until its wall-clock entry here has passed, so a requeued job cannot re-enter the failing download
        # path within a second and burn both attempts. Expired entries are pruned as they are read.
        self._reference_cooldowns: dict[tuple[AuxModelKind, str, bool], float] = {}
        # Jobs whose live deadline is a bounding-only hold (every uncached reference is cooling, nothing is in
        # flight for them). Reconcile must revisit these once their cooldowns lapse, unlike a job holding a
        # genuine in-flight deadline, which it leaves alone.
        self._cooling_deadline_jobs: set[GenerationID] = set()
        # Last observed downloaded-byte count per in-flight file name, so a repeat expiry can tell a download
        # that is advancing (or cannot report bytes at all) from one whose reported bytes have not moved.
        self._last_inflight_bytes: dict[str, int] = {}
        # Canonical identity of the pin set last sent to the download process, so an unchanged set is never
        # re-sent (the eviction-pin update is edge-triggered). Starts empty, which matches a fresh download
        # process's own empty pins, so a worker with no aux-bearing jobs never emits a spurious pin message.
        self._last_sent_pins_key: frozenset[tuple[AuxModelKind, str, bool]] = frozenset()

    def on_job_popped(self, job: ImageGenerateJobPopResponse) -> None:
        """Trigger prefetch for a freshly popped job carrying LoRAs and/or textual inversions.

        A job with none is ignored. A job whose full auxiliary set is already cached this session has its
        dispatch gate cleared immediately (no request). Otherwise the not-yet-cached entries are sent with the
        current eviction-pin set, and a per-job deadline is armed. Never blocks on anything network-bound.
        """
        self._request_prefetch_for_job(job)

    def _request_prefetch_for_job(self, job: ImageGenerateJobPopResponse) -> None:
        """Build and issue one job's prefetch request (or clear its gate when nothing needs fetching).

        Shared by the pop-time trigger and the periodic reconcile sweep so both build the request the same
        way: the not-yet-cached entries plus the current eviction pins, arming a fresh per-job deadline. A
        reference within its post-failure re-fetch cooldown is held back rather than re-requested. A job whose
        whole not-yet-cached set is currently cooling issues no request but keeps a live bounding deadline so it
        stays bounded; a job whose whole set is already cached is marked prepared with no request; a job with no
        auxiliary files or no id is ignored.
        """
        job_id = job.id_
        if job_id is None:
            return
        loras = job.payload.loras or []
        tis = job.payload.tis or []
        if not loras and not tis:
            return

        now = self._clock()
        entries: list[AuxPrefetchEntry] = []
        cooling = False  # at least one still-needed reference is within its post-failure re-fetch cooldown
        for lora in loras:
            if self._job_tracker.is_lora_cached(lora) or self._job_tracker.is_lora_skipped(lora):
                continue
            if self._reference_is_cooling((AuxModelKind.LORA, lora.name, bool(lora.is_version)), now):
                cooling = True
                continue
            entries.append(
                AuxPrefetchEntry(
                    kind=AuxModelKind.LORA,
                    name=lora.name,
                    is_version=bool(lora.is_version),
                    requesting_job_id=job_id,
                ),
            )
        for ti in tis:
            if self._job_tracker.is_ti_cached(ti.name) or self._job_tracker.is_ti_skipped(ti.name):
                continue
            if self._reference_is_cooling((AuxModelKind.TI, ti.name, False), now):
                cooling = True
                continue
            entries.append(AuxPrefetchEntry(kind=AuxModelKind.TI, name=ti.name, requesting_job_id=job_id))

        if entries:
            self._deadlines[job_id] = now + max(0.0, self._download_timeout_provider())
            # A fresh request is a fresh attempt: forget any deferral/cooling bookkeeping a prior attempt left.
            self._forget_deferral_state(job_id)
            pins = self._current_pins()
            self._prefetch_sender(entries, pins)
            self._remember_sent_pins(pins)
            logger.debug(
                f"Requested aux prefetch for job {str(job_id)[:8]}: "
                f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'} not yet cached.",
            )
            return

        if cooling:
            # Every still-needed reference is within its post-failure cooldown: do not re-enter the failing
            # download path yet, but hold a live deadline so the job stays bounded (and counts as aux-held) and
            # scan_deadlines still faults it if the cooldown-plus-refetch cycle never resolves it. The deadline
            # is armed once and not pushed forward on later reconcile passes, so the bound is real.
            self._deadlines.setdefault(job_id, now + max(0.0, self._download_timeout_provider()))
            self._cooling_deadline_jobs.add(job_id)
            return

        # Everything this job needs is already on disk this session; clear its dispatch gate now so it does not
        # fall through to the inference-side preparation path.
        self._cooling_deadline_jobs.discard(job_id)
        self._job_tracker.mark_job_aux_prepared_if_ready(job_id)

    def on_prefetch_result(self, message: HordeAuxPrefetchResultMessage) -> None:
        """Consume per-entry prefetch outcomes: mark cached files, prepare ready jobs, fault failed ones."""
        now = self._clock()
        for outcome in message.outcomes:
            if outcome.ok:
                self._job_tracker.mark_aux_prefetched(
                    outcome.name,
                    is_version=outcome.is_version,
                    is_ti=outcome.kind is AuxModelKind.TI,
                )
                for job_id in outcome.requesting_job_ids:
                    if self._job_tracker.mark_job_aux_prepared_if_ready(job_id):
                        self._deadlines.pop(job_id, None)
                        self._forget_deferral_state(job_id)
                        logger.debug(f"Aux prefetch complete for job {str(job_id)[:8]}; it may now dispatch.")
                continue
            if outcome.rejection_reason is not None:
                self._skip_rejected_aux(outcome)
                continue
            self._fault_failed_outcome(outcome, now=now)

    def scan_deadlines(self, now: float | None = None) -> None:
        """Fault any job whose prefetch has not resolved by its deadline (run from the periodic loop).

        A deadline is a backstop, not the primary failure detector: the downloader reports a genuine failure
        through the result path. So an expiry that finds the job's file still in flight defers the fault by one
        download-timeout budget rather than punishing a slow-but-alive transfer, bounded at
        ``_MAX_DEADLINE_DEFERRALS`` extra budgets and short-circuited earlier if the download reports bytes that
        stop advancing. Only a job with no in-flight file, or one out of deferral budget or observed stalled, is
        faulted, exactly as before the deferral existed.
        """
        reference = self._clock() if now is None else now
        in_flight = self._in_flight_provider()
        self._prune_inflight_memory()
        expired = [job_id for job_id, deadline in self._deadlines.items() if reference >= deadline]
        for job_id in expired:
            tracked = self._job_tracker.get_tracked_job(job_id)
            if tracked is None or tracked.stage != JobStage.PENDING_INFERENCE or tracked.aux_models_prepared:
                self._deadlines.pop(job_id, None)
                self._forget_deferral_state(job_id)
                continue
            if self._defer_deadline_if_in_flight(tracked.sdk_api_job_info, in_flight=in_flight, now=reference):
                continue
            # A deadline is per-job, so each expiry is its own incident: arm the backoff (once) for each
            # auxiliary class the job carries, then fault it. A job carrying both classes arms both. Retryability
            # follows the combined backoff state exactly as a reported failure does; both arms always run (their
            # strike is a side effect) so a class window escalates even when another class already made the fault
            # terminal.
            payload = tracked.sdk_api_job_info.payload
            retryable = True
            if payload.loras:
                retryable = self._arm_lora_backoff(now=reference) and retryable
            if payload.tis:
                retryable = self._arm_ti_backoff(now=reference) and retryable
            self._fault_pending_job(job_id, retryable=retryable, detail="aux prefetch deadline exceeded")

    def _defer_deadline_if_in_flight(
        self,
        job: ImageGenerateJobPopResponse,
        *,
        in_flight: dict[str, tuple[int, int]],
        now: float,
    ) -> bool:
        """Extend a job's deadline when its not-yet-cached files are still downloading; return whether it did.

        Defers only while at least one of the job's outstanding files is present in the downloader's in-flight
        set and (a) reports no byte progress it could stall on, or (b) reports bytes that advanced since the last
        expiry. A file whose reported bytes have not moved since the previous expiry is treated as stalled and no
        longer defers. The deferral is capped per job so a download that neither completes nor reports movement
        cannot postpone the fault indefinitely.
        """
        job_id = job.id_
        if job_id is None:
            return False
        matched = [name for name in self._uncached_entry_names(job) if name in in_flight]
        if not matched:
            return False
        if self._deferrals.get(job_id, 0) >= _MAX_DEADLINE_DEFERRALS:
            return False
        if not any(self._file_still_progressing(name, in_flight) for name in matched):
            return False
        self._deferrals[job_id] = self._deferrals.get(job_id, 0) + 1
        self._deadlines[job_id] = now + max(0.0, self._download_timeout_provider())
        for name in matched:
            self._last_inflight_bytes[name] = in_flight[name][0]
        if job_id not in self._deferral_logged:
            self._deferral_logged.add(job_id)
            logger.info(
                f"Deferring aux prefetch deadline for job {str(job_id)[:8]}: "
                f"{len(matched)} file(s) still downloading.",
            )
        return True

    def _prune_inflight_memory(self) -> None:
        """Drop remembered byte counts for files no job with a live deadline still awaits.

        The stall check compares a file's reported byte count across the consecutive expiries of the job(s)
        waiting on it, so that remembered count must survive any intervening tick on which the in-flight
        provider reports nothing (a provider flicker, or a downloader that briefly stops reporting). Memory is
        therefore retained by which files a still-deadlined job references, never by what happens to be
        downloading on any single tick: an empty or absent in-flight observation between two expiries can no
        longer wipe a file's remembered bytes and reset it to "progressing by default". It cannot grow without
        bound because a prepared, faulted, or departed job drops out of the deadline set, taking its files with
        it once no other deadlined job references them.
        """
        awaited: set[str] = set()
        for job_id in self._deadlines:
            tracked = self._job_tracker.get_tracked_job(job_id)
            if tracked is not None:
                awaited |= self._uncached_entry_names(tracked.sdk_api_job_info)
        self._last_inflight_bytes = {
            name: observed for name, observed in self._last_inflight_bytes.items() if name in awaited
        }

    def on_downloader_reset(self) -> None:
        """Forget every in-flight prefetch deadline so pending jobs are re-requested against a fresh downloader.

        Invoked when the background download process is replaced (its unexpected death, then restart): the
        deadlines and deferral bookkeeping were tracked against a process that no longer exists, and the
        stashed in-flight byte counts describe transfers that can never resume. Clearing them lets the periodic
        reconcile re-request each still-pending aux-bearing job exactly as a fresh pop would, arming one fresh
        download-timeout budget against the new downloader. A job therefore survives the downloader's death and
        is re-fetched within a single budget rather than waiting out the deferral cap against a corpse.
        """
        self._deadlines.clear()
        self._deferrals.clear()
        self._deferral_logged.clear()
        self._last_inflight_bytes.clear()
        # The cooldowns protect against re-entering the *same* failing download path; a replacement downloader
        # is a fresh path, so a pending reference deserves an immediate attempt against it rather than waiting
        # out a cooldown against the corpse.
        self._reference_cooldowns.clear()
        self._cooling_deadline_jobs.clear()

    def _file_still_progressing(self, name: str, in_flight: dict[str, tuple[int, int]]) -> bool:
        """Whether an in-flight file counts as progressing (so its job's deadline may defer again).

        A file reporting zero bytes cannot express progress, so it is given the benefit of the doubt (the
        per-job cap still bounds how long that lasts). A file reporting bytes counts as progressing only while
        those bytes exceed the count observed at the previous expiry; the first byte-carrying observation has no
        prior to compare against and so is progressing by default.
        """
        downloaded, _total = in_flight[name]
        if downloaded <= 0:
            return True
        previous = self._last_inflight_bytes.get(name)
        if previous is None:
            return True
        return downloaded > previous

    def _uncached_entry_names(self, job: ImageGenerateJobPopResponse) -> set[str]:
        """The job's auxiliary reference names not yet known cached (the set a prefetch request would fetch)."""
        names: set[str] = set()
        for lora in job.payload.loras or []:
            if not self._job_tracker.is_lora_cached(lora) and not self._job_tracker.is_lora_skipped(lora):
                names.add(lora.name)
        for ti in job.payload.tis or []:
            if not self._job_tracker.is_ti_cached(ti.name) and not self._job_tracker.is_ti_skipped(ti.name):
                names.add(ti.name)
        return names

    def _forget_deferral_state(self, job_id: GenerationID) -> None:
        """Clear a job's deferral count, one-shot log flag, and cooling-hold marker.

        Called wherever a job's deadline is dropped or superseded (prepare, fault, a fresh in-flight request,
        or departure): the deferral bookkeeping and the bounding-hold marker both belong to the attempt that
        just ended, so a later attempt starts clean.
        """
        self._deferrals.pop(job_id, None)
        self._deferral_logged.discard(job_id)
        self._cooling_deadline_jobs.discard(job_id)

    @staticmethod
    def _outcome_key(outcome: AuxPrefetchOutcome) -> tuple[AuxModelKind, str, bool]:
        """The cooldown/pin identity of one outcome's reference (TIs never carry a version)."""
        is_version = outcome.kind is AuxModelKind.LORA and bool(outcome.is_version)
        return (outcome.kind, outcome.name, is_version)

    def _reference_is_cooling(self, key: tuple[AuxModelKind, str, bool], now: float) -> bool:
        """Whether a reference is still within its post-failure re-fetch cooldown (pruning it once lapsed)."""
        expiry = self._reference_cooldowns.get(key)
        if expiry is None:
            return False
        if now < expiry:
            return True
        del self._reference_cooldowns[key]
        return False

    def _arm_reference_cooldown(self, outcome: AuxPrefetchOutcome, *, now: float) -> None:
        """Hold a just-failed reference against re-request for ``min(cap, remaining class-backoff window)``.

        Capping at the class window means the cooldown never outlives the suppression a strike just armed:
        once that window lapses the reference is fetchable again, so cooling it further would only delay a
        legitimate retry. A failure always arms the class backoff first, so in practice the remaining window
        dominates the cap and the effective cooldown is :data:`_AUX_REFETCH_COOLDOWN_SECONDS`.
        """
        backoff = (
            self._state.lora_download_backoff if outcome.kind is AuxModelKind.LORA else self._state.ti_download_backoff
        )
        cooldown = min(_AUX_REFETCH_COOLDOWN_SECONDS, backoff.remaining_seconds(now))
        if cooldown <= 0.0:
            return
        self._reference_cooldowns[self._outcome_key(outcome)] = now + cooldown

    def reconcile_and_refresh_pins(self) -> None:
        """Heal lost/stale prefetch state and keep the eviction-pin set current (run from the periodic loop).

        Two edge-triggered steps that together make the pipeline self-correcting between pops:

        - Reconcile: any tracked job awaiting inference that carries LoRAs or TIs, is not yet prepared, and has
          no in-flight request (no deadline entry) is re-requested exactly as a fresh pop would. This heals a
          retryable prefetch-failure requeue (the job re-enters the queue with no deadline), a lost result
          message, and a download-process restart that dropped its in-flight map. A job that already holds a
          live deadline is left alone, so its single deadline remains the authoritative one-shot backstop; the
          tracker's retry policy bounds total attempts, so a permanently failing job cannot loop forever.
        - Pin refresh: recompute the pin set and send a pins-only update only when it differs from the last
          set sent, so a job's completed files stop being pinned without waiting for the next pop, and an
          unchanged set never produces a repeated message.
        """
        self._reconcile_pending_jobs()
        self._refresh_pins()

    def _reconcile_pending_jobs(self) -> None:
        """Re-request prefetch for any aux-bearing pending job left without an in-flight request.

        A job holding a genuine in-flight deadline is left alone (its single deadline is the authoritative
        backstop). A job whose deadline is only a bounding hold because its references are cooling is revisited
        so that, once a cooldown lapses, the reference is re-requested; while the cooldown still holds the
        revisit is a no-op that preserves the existing bounding deadline.
        """
        for tracked in self._job_tracker.tracked_jobs():
            if tracked.stage != JobStage.PENDING_INFERENCE or tracked.aux_models_prepared:
                continue
            job = tracked.sdk_api_job_info
            if job.id_ is None:
                continue
            if job.id_ in self._deadlines and job.id_ not in self._cooling_deadline_jobs:
                continue
            if not (job.payload.loras or job.payload.tis):
                continue
            self._request_prefetch_for_job(job)

    def _refresh_pins(self) -> None:
        """Send a pins-only update when the current pin set differs from the last one sent (coalesced)."""
        pins = self._current_pins()
        if self._pins_key(pins) == self._last_sent_pins_key:
            return
        self._pin_sender(pins)
        self._remember_sent_pins(pins)

    def has_live_deadline(self, job_id: GenerationID, now: float | None = None) -> bool:
        """Whether a prefetch request for ``job_id`` is in flight (its deadline exists and has not expired).

        Read by the recovery coordinator so save-our-ship give-up defers to a head-of-queue job whose
        auxiliary prefetch is still progressing, bounded by that deadline: once it expires the entry is gone
        (or scan_deadlines faults the job) and this reports False, so a stalled prefetch cannot defer give-up
        forever.
        """
        deadline = self._deadlines.get(job_id)
        if deadline is None:
            return False
        reference = self._clock() if now is None else now
        return reference < deadline

    def job_ids_with_live_deadlines(self, now: float | None = None) -> set[GenerationID]:
        """Return the ids of jobs whose auxiliary prefetch is still in flight (deadline present, unexpired).

        Read by the deadlock detector so a pending-inference job that is intentionally holding no lane while
        its LoRA/TI prefetch runs in the background download process does not fuel a queue- or general-deadlock
        verdict: "pending plus every process idle" is the aux-prefetch gate working as designed, not a wedge.
        The hold is intrinsically bounded by these deadlines: once a job's deadline expires the entry is gone
        (or ``scan_deadlines`` faults the job) and it drops out of this set, so a stalled prefetch cannot shield
        the queue from deadlock detection forever.
        """
        reference = self._clock() if now is None else now
        return {job_id for job_id, deadline in self._deadlines.items() if reference < deadline}

    @staticmethod
    def _pins_key(pins: list[AuxModelRef]) -> frozenset[tuple[AuxModelKind, str, bool]]:
        """Order-independent identity of a pin set, for coalescing repeated identical updates."""
        return frozenset((pin.kind, pin.name, bool(pin.is_version)) for pin in pins)

    def _remember_sent_pins(self, pins: list[AuxModelRef]) -> None:
        """Record the pin set just sent (on any request), so the next refresh coalesces against it."""
        self._last_sent_pins_key = self._pins_key(pins)

    def _skip_rejected_aux(self, outcome: AuxPrefetchOutcome) -> None:
        """Let every job waiting on a terminally-rejected auxiliary file proceed without it, not faulting them.

        An auxiliary file the fetch API permanently refuses (a LoRA that is invalid, too large, or NSFW on an
        SFW-only worker; a textual inversion the API rejects) will never be on disk, so waiting or retrying is
        futile and faulting the job needlessly drops otherwise-servable work. The file is recorded as skipped
        so a job's aux set counts as ready without it, and each waiting job is re-evaluated for dispatch. No
        download backoff strike is armed: a rejection is a property of that one file, not a sick download path,
        so it must not withhold unrelated jobs of the same class. The skip is recorded even when no job is still
        waiting, so a later job referencing the same file is not re-requested.
        """
        is_ti = outcome.kind is AuxModelKind.TI
        self._job_tracker.mark_aux_skipped(outcome.name, is_version=outcome.is_version, is_ti=is_ti)
        logger.info(
            f"Aux ({'TI' if is_ti else 'LoRA'}) prefetch rejected for {outcome.name!r} "
            f"({outcome.rejection_reason}); dispatching its job(s) without it.",
        )
        for job_id in outcome.requesting_job_ids:
            if self._job_tracker.mark_job_aux_prepared_if_ready(job_id):
                self._deadlines.pop(job_id, None)
                self._forget_deferral_state(job_id)

    def _fault_failed_outcome(self, outcome: AuxPrefetchOutcome, *, now: float) -> None:
        """Fault the still-pending jobs waiting on one failed prefetch outcome, arming its class backoff once.

        The download process reports one deduplicated outcome per file, so a single failed download can name
        several jobs. The escalating backoff for that file's class (LoRA or textual inversion) is armed once for
        that download (not once per waiting job), and every named job gets the same retryability verdict, so a
        shared failure neither over-counts backoff strikes nor treats co-waiting jobs inconsistently. A failure
        feeds the same class backoff the inference-side aux-download fault does; once an incident is active a job
        is faulted terminally rather than requeued into the same failing download path.

        A terminal (non-retryable) failure additionally memoizes the reference as skipped, exactly as a surfaced
        rejection does: retrying it is futile, so already-queued and later jobs referencing the same file
        dispatch without it rather than each faulting in turn. This bounds the damage of one doomed reference
        (a not-found laundered into a plain failure) to a single terminal fault per incident: a subsequent
        failure for a reference already memoized takes the skip path below instead of manufacturing more faults.
        Every fault is stamped with the aux-prefetch origin so a terminal one is excluded from the
        consecutive-failure pop pause.
        """
        live = [job_id for job_id in outcome.requesting_job_ids if self._is_pending(job_id)]
        if not live:
            # Every job that requested this file has already moved on (dispatched, prepared by a sibling
            # entry, or faulted). A late failure for work nobody is waiting on disturbs nothing and must not
            # manufacture a spurious backoff strike.
            return
        is_ti = outcome.kind is AuxModelKind.TI
        if self._job_tracker.is_aux_skipped(outcome.name, is_version=outcome.is_version, is_ti=is_ti):
            # A prior terminal failure already memoized this reference: let its still-pending jobs dispatch
            # without it rather than arming the backoff again or faulting work a skip has already salvaged.
            for job_id in live:
                if self._job_tracker.mark_job_aux_prepared_if_ready(job_id):
                    self._deadlines.pop(job_id, None)
                    self._forget_deferral_state(job_id)
            return
        detail = outcome.detail or "aux prefetch failed"
        if outcome.kind is AuxModelKind.LORA:
            retryable = outcome.retryable and self._arm_lora_backoff(now=now, reference=outcome.name)
        else:
            retryable = outcome.retryable and self._arm_ti_backoff(now=now, reference=outcome.name)
        if not retryable:
            # Terminal: the download will not resolve by retrying, so memoize the reference so co-queued and
            # later jobs skip it instead of each faulting. The job(s) actively waiting on this download are
            # still faulted below (one bounded incident). Unlike a surfaced rejection (a permanent property
            # of the file), a plain failure is an incident verdict: the verdict lapses with the incident's
            # decay window so a fetchable-again reference is retried rather than silently omitted forever.
            self._job_tracker.mark_aux_skipped(
                outcome.name,
                is_version=outcome.is_version,
                is_ti=is_ti,
                expires_at=now + STRIKE_DECAY_SECONDS,
            )
            # The skip supersedes any re-fetch cooldown (a skipped reference is never re-requested anyway).
            self._reference_cooldowns.pop(self._outcome_key(outcome), None)
        else:
            # Retryable: the requeued job will be revisited by the pop/reconcile path, so hold the reference in
            # a cooldown first, or a sub-second failure is re-requested at once and its instant re-failure burns
            # the job's remaining attempt.
            self._arm_reference_cooldown(outcome, now=now)
        for job_id in live:
            self._fault_pending_job(job_id, retryable=retryable, detail=detail)

    def _is_pending(self, job_id: GenerationID) -> bool:
        """Whether the job is still tracked and awaiting inference (so a fault would actually affect it)."""
        tracked = self._job_tracker.get_tracked_job(job_id)
        return tracked is not None and tracked.stage == JobStage.PENDING_INFERENCE

    def _arm_lora_backoff(self, *, now: float, reference: str | None = None) -> bool:
        """Register one LoRA-download timeout strike and return whether a retry is still worthwhile.

        Retryability is read *before* the strike lands: while an incident is already active a fresh failure is
        terminal (requeuing would only re-enter the same failing download path), otherwise it is retryable.
        ``reference`` names the offending file in the warning when the strike traces to one specific reference
        (a reported failure); it is None for the deadline path, whose job may carry several references.
        """
        retryable = not self._state.lora_download_backoff.is_escalation_active(now)
        window = self._state.lora_download_backoff.register_timeout(now)
        named = f" for {reference!r}" if reference is not None else ""
        logger.warning(
            f"Aux (LoRA) prefetch failed{named}; withholding LoRA job pops for {window:.0f}s "
            f"(strike {self._state.lora_download_backoff.strikes}).",
        )
        return retryable

    def _arm_ti_backoff(self, *, now: float, reference: str | None = None) -> bool:
        """Register one textual-inversion download timeout strike and return whether a retry is worthwhile.

        Mirrors :meth:`_arm_lora_backoff` against the textual-inversion backoff: retryability is read *before*
        the strike lands, so while an incident is already active a fresh failure is terminal (requeuing would
        only re-enter the same failing download path), otherwise it is retryable. Unlike the LoRA backoff this
        window cannot suppress pop traffic: the pop request has no per-request textual-inversion capability flag
        (there is no textual-inversion analogue to ``allow_lora``), so the window influences fault classification
        only. ``reference`` names the offending file in the warning when the strike traces to one reference.
        """
        retryable = not self._state.ti_download_backoff.is_escalation_active(now)
        window = self._state.ti_download_backoff.register_timeout(now)
        named = f" for {reference!r}" if reference is not None else ""
        logger.warning(
            f"Aux (TI) prefetch failed{named}; textual-inversion download backoff active for {window:.0f}s "
            f"(strike {self._state.ti_download_backoff.strikes}).",
        )
        return retryable

    def _fault_pending_job(self, job_id: GenerationID, *, retryable: bool, detail: str) -> None:
        """Fault one still-pending prefetch job, dropping its deadline.

        A no-op if the job is gone or no longer pending (already dispatched, prepared by a sibling entry, or
        faulted), so a late or duplicate failure never disturbs a job that has moved on. Any terminal fault is
        stamped with the aux-prefetch origin so it is excluded from the consecutive-failure pop pause: the
        worker never ran a generation for this job, so a fetch it cannot satisfy is not a generation verdict.
        """
        tracked = self._job_tracker.get_tracked_job(job_id)
        if tracked is None or tracked.stage != JobStage.PENDING_INFERENCE:
            self._deadlines.pop(job_id, None)
            self._forget_deferral_state(job_id)
            return
        self._job_tracker.handle_job_fault_now(
            tracked.sdk_api_job_info,
            retryable=retryable,
            fault_reason=detail,
            fault_origin=JobFaultOrigin.AUX_PREFETCH,
        )
        self._deadlines.pop(job_id, None)
        self._forget_deferral_state(job_id)

    def _current_pins(self) -> list[AuxModelRef]:
        """The eviction-pin set: every auxiliary file any not-yet-terminal tracked job still references."""
        seen: set[tuple[AuxModelKind, str, bool]] = set()
        pins: list[AuxModelRef] = []
        for tracked in self._job_tracker.tracked_jobs():
            if tracked.stage not in _PIN_STAGES:
                continue
            payload = tracked.sdk_api_job_info.payload
            for lora in payload.loras or []:
                key = (AuxModelKind.LORA, lora.name, bool(lora.is_version))
                if key not in seen:
                    seen.add(key)
                    pins.append(AuxModelRef(kind=AuxModelKind.LORA, name=lora.name, is_version=bool(lora.is_version)))
            for ti in payload.tis or []:
                key = (AuxModelKind.TI, ti.name, False)
                if key not in seen:
                    seen.add(key)
                    pins.append(AuxModelRef(kind=AuxModelKind.TI, name=ti.name))
        return pins
