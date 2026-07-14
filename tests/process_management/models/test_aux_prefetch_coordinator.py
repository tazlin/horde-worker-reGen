"""Behavioural tests for the parent-side ad-hoc auxiliary (LoRA/TI) prefetch coordinator.

These exercise the three seams the coordinator owns without a live download process: the pop-time request
(correct entries and pin set), the completion path (a job's dispatch gate clears once its whole auxiliary set
is present, and two jobs sharing a file are both prepared), and the failure/deadline paths (a pending job is
faulted, the LoRA-download backoff is armed, and a late outcome for a job that has moved on is a no-op).
"""

from __future__ import annotations

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry, TIPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    AuxModelKind,
    AuxModelRef,
    AuxPrefetchEntry,
    AuxPrefetchOutcome,
    HordeAuxPrefetchResultMessage,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.models.aux_prefetch_coordinator import AuxPrefetchCoordinator
from tests.process_management.conftest import make_job_pop_response, track_popped_job_async


class _Clock:
    """A hand-advanceable clock so deadline behaviour is deterministic."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _SenderSpy:
    """Records the (entries, pins) of each prefetch request the coordinator would send."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[AuxPrefetchEntry], list[AuxModelRef]]] = []

    def __call__(self, entries: list[AuxPrefetchEntry], pins: list[AuxModelRef]) -> None:
        self.calls.append((entries, pins))


class _PinSenderSpy:
    """Records each pins-only update the coordinator would send."""

    def __init__(self) -> None:
        self.calls: list[list[AuxModelRef]] = []

    def __call__(self, pins: list[AuxModelRef]) -> None:
        self.calls.append(pins)


class _InFlightSpy:
    """A scriptable stand-in for the downloader's in-flight ad-hoc prefetch set (name -> (downloaded, total))."""

    def __init__(self) -> None:
        self.map: dict[str, tuple[int, int]] = {}

    def __call__(self) -> dict[str, tuple[int, int]]:
        return dict(self.map)


def _lora(name: str, *, is_version: bool = False) -> LorasPayloadEntry:
    return LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=is_version)


def _job(
    *, loras: list[LorasPayloadEntry] | None = None, tis: list[TIPayloadEntry] | None = None
) -> ImageGenerateJobPopResponse:
    return make_job_pop_response("some-model", loras=loras, tis=tis)


def _make(
    tracker: JobTracker,
    *,
    clock: _Clock | None = None,
    timeout: float = 120.0,
    pin_sender: _PinSenderSpy | None = None,
    in_flight: _InFlightSpy | None = None,
) -> tuple[AuxPrefetchCoordinator, _SenderSpy, WorkerState, _Clock]:
    state = WorkerState()
    sender = _SenderSpy()
    the_clock = clock if clock is not None else _Clock()
    coordinator = AuxPrefetchCoordinator(
        job_tracker=tracker,
        state=state,
        prefetch_sender=sender,
        download_timeout_provider=lambda: timeout,
        pin_sender=pin_sender if pin_sender is not None else _PinSenderSpy(),
        in_flight_provider=in_flight if in_flight is not None else _InFlightSpy(),
        clock=the_clock,
    )
    return coordinator, sender, state, the_clock


def _ok_message(job: ImageGenerateJobPopResponse, name: str) -> HordeAuxPrefetchResultMessage:
    """A success outcome message for one LoRA of one job."""
    assert job.id_ is not None
    return HordeAuxPrefetchResultMessage(
        process_id=9000,
        process_launch_identifier=1,
        info="r",
        outcomes=[AuxPrefetchOutcome(kind=AuxModelKind.LORA, name=name, ok=True, requesting_job_ids=[job.id_])],
    )


async def test_pop_with_loras_sends_one_request_with_entries_and_pins() -> None:
    """Popping a LoRA job issues exactly one request with its uncached entries and the tracked pin set."""
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA"), _lora("12345", is_version=True)])
    await track_popped_job_async(tracker, job)
    coordinator, sender, _state, _clock = _make(tracker)

    coordinator.on_job_popped(job)

    assert len(sender.calls) == 1
    entries, pins = sender.calls[0]
    assert {(e.kind, e.name, e.is_version) for e in entries} == {
        (AuxModelKind.LORA, "styleA", False),
        (AuxModelKind.LORA, "12345", True),
    }
    assert all(e.requesting_job_id == job.id_ for e in entries)
    # The pin set covers every auxiliary file the tracked job still references.
    assert {(p.kind, p.name, p.is_version) for p in pins} == {
        (AuxModelKind.LORA, "styleA", False),
        (AuxModelKind.LORA, "12345", True),
    }


async def test_pop_without_aux_sends_nothing() -> None:
    """A job with neither LoRAs nor TIs triggers no prefetch request."""
    tracker = JobTracker()
    job = _job()
    await track_popped_job_async(tracker, job)
    coordinator, sender, _state, _clock = _make(tracker)

    coordinator.on_job_popped(job)

    assert sender.calls == []


async def test_already_cached_pop_clears_gate_without_request() -> None:
    """When a job's whole auxiliary set is already cached, its gate clears and no request is sent."""
    tracker = JobTracker()
    lora = _lora("cached-style")
    job = _job(loras=[lora])
    await track_popped_job_async(tracker, job)
    tracker.mark_aux_prefetched("cached-style", is_version=False, is_ti=False)
    coordinator, sender, _state, _clock = _make(tracker)

    coordinator.on_job_popped(job)

    assert sender.calls == []
    assert tracker.are_job_aux_models_prepared(job) is True


async def test_completion_marks_prepared_and_clears_deadline() -> None:
    """A success outcome caches the file, prepares the job, and drops its deadline (no later fault)."""
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    coordinator, _sender, _state, clock = _make(tracker, timeout=30.0)
    coordinator.on_job_popped(job)

    coordinator.on_prefetch_result(
        HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[
                AuxPrefetchOutcome(
                    kind=AuxModelKind.LORA,
                    name="styleA",
                    ok=True,
                    requesting_job_ids=[job.id_],
                ),
            ],
        ),
    )

    assert tracker.are_job_aux_models_prepared(job) is True
    # The deadline is cleared, so a later scan past it does not fault the (now prepared) job.
    clock.now += 10_000.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE


async def test_two_jobs_sharing_a_lora_both_prepared_by_one_outcome() -> None:
    """One shared-LoRA outcome carrying both job ids prepares both jobs."""
    tracker = JobTracker()
    job_a = _job(loras=[_lora("shared")])
    job_b = _job(loras=[_lora("shared")])
    await track_popped_job_async(tracker, job_a)
    await track_popped_job_async(tracker, job_b)
    coordinator, _sender, _state, _clock = _make(tracker)
    coordinator.on_job_popped(job_a)
    coordinator.on_job_popped(job_b)

    coordinator.on_prefetch_result(
        HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[
                AuxPrefetchOutcome(
                    kind=AuxModelKind.LORA,
                    name="shared",
                    ok=True,
                    requesting_job_ids=[job_a.id_, job_b.id_],
                ),
            ],
        ),
    )

    assert tracker.are_job_aux_models_prepared(job_a) is True
    assert tracker.are_job_aux_models_prepared(job_b) is True


async def test_partial_completion_does_not_prepare_multi_lora_job() -> None:
    """A job needing two LoRAs is prepared only once the second one lands, not the first."""
    tracker = JobTracker()
    job = _job(loras=[_lora("one"), _lora("two")])
    await track_popped_job_async(tracker, job)
    coordinator, _sender, _state, _clock = _make(tracker)
    coordinator.on_job_popped(job)

    def _ok(name: str) -> HordeAuxPrefetchResultMessage:
        return HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[AuxPrefetchOutcome(kind=AuxModelKind.LORA, name=name, ok=True, requesting_job_ids=[job.id_])],
        )

    coordinator.on_prefetch_result(_ok("one"))
    assert tracker.are_job_aux_models_prepared(job) is False
    coordinator.on_prefetch_result(_ok("two"))
    assert tracker.are_job_aux_models_prepared(job) is True


async def test_failure_faults_job_and_arms_backoff() -> None:
    """A LoRA prefetch failure faults the pending job and registers a LoRA-download backoff strike."""
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    coordinator, _sender, state, _clock = _make(tracker)
    coordinator.on_job_popped(job)

    coordinator.on_prefetch_result(
        HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[
                AuxPrefetchOutcome(
                    kind=AuxModelKind.LORA,
                    name="styleA",
                    ok=False,
                    retryable=True,
                    detail="civitai down",
                    requesting_job_ids=[job.id_],
                ),
            ],
        ),
    )

    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT
    assert state.lora_download_backoff.strikes == 1


async def test_terminal_lora_rejection_dispatches_job_without_it() -> None:
    """A terminally-rejected LoRA lets its job dispatch (prepared, not faulted) and arms no backoff strike."""
    tracker = JobTracker()
    job = _job(loras=[_lora("badlora")])
    await track_popped_job_async(tracker, job)
    coordinator, _sender, state, _clock = _make(tracker)
    coordinator.on_job_popped(job)

    coordinator.on_prefetch_result(
        HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[
                AuxPrefetchOutcome(
                    kind=AuxModelKind.LORA,
                    name="badlora",
                    ok=False,
                    retryable=False,
                    rejection_reason="invalid",
                    requesting_job_ids=[job.id_],
                ),
            ],
        ),
    )

    # The job is prepared and stays pending inference (dispatchable), not faulted to PENDING_SUBMIT.
    assert tracker.are_job_aux_models_prepared(job) is True
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    # A rejected file is a property of that one LoRA, not a sick download path, so no backoff is armed.
    assert state.lora_download_backoff.strikes == 0


async def test_shared_rejected_lora_dispatches_both_jobs_no_strike() -> None:
    """One rejection outcome naming two jobs sharing the LoRA prepares both and arms no backoff strike."""
    tracker = JobTracker()
    job_a = _job(loras=[_lora("badshared")])
    job_b = _job(loras=[_lora("badshared")])
    await track_popped_job_async(tracker, job_a)
    await track_popped_job_async(tracker, job_b)
    coordinator, _sender, state, _clock = _make(tracker)
    coordinator.on_job_popped(job_a)
    coordinator.on_job_popped(job_b)

    coordinator.on_prefetch_result(
        HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[
                AuxPrefetchOutcome(
                    kind=AuxModelKind.LORA,
                    name="badshared",
                    ok=False,
                    retryable=False,
                    rejection_reason="too_large",
                    requesting_job_ids=[job_a.id_, job_b.id_],
                ),
            ],
        ),
    )

    assert tracker.are_job_aux_models_prepared(job_a) is True
    assert tracker.are_job_aux_models_prepared(job_b) is True
    assert tracker.get_stage(job_a.id_) == JobStage.PENDING_INFERENCE
    assert tracker.get_stage(job_b.id_) == JobStage.PENDING_INFERENCE
    assert state.lora_download_backoff.strikes == 0


async def test_rejected_lora_not_re_requested_on_next_job() -> None:
    """After a rejection is recorded, a later job referencing the same LoRA prepares with no new request."""
    tracker = JobTracker()
    job_a = _job(loras=[_lora("badlora")])
    await track_popped_job_async(tracker, job_a)
    coordinator, sender, _state, _clock = _make(tracker)
    coordinator.on_job_popped(job_a)
    # The pop issued one request naming the LoRA.
    assert len(sender.calls) == 1

    coordinator.on_prefetch_result(
        HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[
                AuxPrefetchOutcome(
                    kind=AuxModelKind.LORA,
                    name="badlora",
                    ok=False,
                    retryable=False,
                    rejection_reason="nsfw",
                    requesting_job_ids=[job_a.id_],
                ),
            ],
        ),
    )

    # A new job referencing the same rejected LoRA is prepared immediately, and no new prefetch entry is sent.
    job_b = _job(loras=[_lora("badlora")])
    await track_popped_job_async(tracker, job_b)
    coordinator.on_job_popped(job_b)

    assert len(sender.calls) == 1
    assert tracker.are_job_aux_models_prepared(job_b) is True
    assert tracker.get_stage(job_b.id_) == JobStage.PENDING_INFERENCE


async def test_shared_lora_failure_arms_backoff_once_and_faults_both_jobs() -> None:
    """One failed download shared by two jobs faults both but registers a single backoff strike."""
    tracker = JobTracker()
    job_a = _job(loras=[_lora("shared")])
    job_b = _job(loras=[_lora("shared")])
    await track_popped_job_async(tracker, job_a)
    await track_popped_job_async(tracker, job_b)
    coordinator, _sender, state, _clock = _make(tracker)
    coordinator.on_job_popped(job_a)
    coordinator.on_job_popped(job_b)

    coordinator.on_prefetch_result(
        HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[
                AuxPrefetchOutcome(
                    kind=AuxModelKind.LORA,
                    name="shared",
                    ok=False,
                    requesting_job_ids=[job_a.id_, job_b.id_],
                ),
            ],
        ),
    )

    assert tracker.get_stage(job_a.id_) == JobStage.PENDING_SUBMIT
    assert tracker.get_stage(job_b.id_) == JobStage.PENDING_SUBMIT
    # One download failed once, so exactly one strike is registered despite two waiting jobs.
    assert state.lora_download_backoff.strikes == 1


async def test_deadline_expiry_faults_unresolved_job() -> None:
    """A job whose prefetch never resolves is faulted once its deadline passes."""
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    coordinator, _sender, state, clock = _make(tracker, timeout=60.0)
    coordinator.on_job_popped(job)

    # Before the deadline, the job is untouched.
    clock.now += 30.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE

    # Past the deadline, it is faulted and the backoff is armed.
    clock.now += 40.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT
    assert state.lora_download_backoff.strikes == 1


async def test_ti_job_gates_and_prepares_like_loras() -> None:
    """A textual-inversion job sends a TI entry and is prepared when it lands."""
    tracker = JobTracker()
    job = _job(tis=[TIPayloadEntry(name="emb-1")])
    await track_popped_job_async(tracker, job)
    coordinator, sender, _state, _clock = _make(tracker)
    coordinator.on_job_popped(job)

    assert len(sender.calls) == 1
    entries, _pins = sender.calls[0]
    assert [(e.kind, e.name) for e in entries] == [(AuxModelKind.TI, "emb-1")]

    coordinator.on_prefetch_result(
        HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[AuxPrefetchOutcome(kind=AuxModelKind.TI, name="emb-1", ok=True, requesting_job_ids=[job.id_])],
        ),
    )
    assert tracker.are_job_aux_models_prepared(job) is True


async def test_outcome_for_gone_job_is_a_noop() -> None:
    """A success outcome for a job that already faulted/left the queue neither raises nor mutates it."""
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    coordinator, _sender, _state, _clock = _make(tracker)
    coordinator.on_job_popped(job)

    # The job is faulted out of the pending queue before the outcome arrives.
    tracker.handle_job_fault_now(job, retryable=False, fault_reason="gone")
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT

    coordinator.on_prefetch_result(
        HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[
                AuxPrefetchOutcome(kind=AuxModelKind.LORA, name="styleA", ok=True, requesting_job_ids=[job.id_])
            ],
        ),
    )

    # The already-terminal job is not resurrected into a prepared/pending state.
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT
    assert tracker.are_job_aux_models_prepared(job) is False


async def test_failure_for_gone_job_does_not_double_fault() -> None:
    """A failure outcome for a job no longer pending is a no-op (no spurious extra strike/fault)."""
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    coordinator, _sender, state, _clock = _make(tracker)
    coordinator.on_job_popped(job)
    tracker.handle_job_fault_now(job, retryable=False, fault_reason="gone")

    coordinator.on_prefetch_result(
        HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[
                AuxPrefetchOutcome(
                    kind=AuxModelKind.LORA,
                    name="styleA",
                    ok=False,
                    requesting_job_ids=[job.id_],
                ),
            ],
        ),
    )

    assert state.lora_download_backoff.strikes == 0


async def test_reconcile_re_requests_a_pending_job_with_no_in_flight_request_then_prepares_it() -> None:
    """A pending aux job left without a request (lost message, downloader restart, requeue) is healed.

    The sweep finds a PENDING_INFERENCE LoRA job that is not prepared and has no live deadline, re-requests
    it (arming a fresh deadline), and a later success outcome prepares it so it becomes dispatchable.
    """
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    coordinator, sender, _state, _clock = _make(tracker, timeout=60.0)

    # No pop-time request was recorded for this job (its request message was lost, or the downloader restarted
    # and dropped its in-flight map), so it sits pending with no deadline. The sweep must re-request it.
    assert coordinator.has_live_deadline(job.id_) is False
    coordinator.reconcile_and_refresh_pins()
    assert len(sender.calls) == 1
    assert coordinator.has_live_deadline(job.id_) is True

    # A second immediate sweep must not double-request it (it now has a live deadline).
    coordinator.reconcile_and_refresh_pins()
    assert len(sender.calls) == 1

    # When the download succeeds the job is prepared and becomes dispatchable.
    coordinator.on_prefetch_result(
        HordeAuxPrefetchResultMessage(
            process_id=9000,
            process_launch_identifier=1,
            info="r",
            outcomes=[
                AuxPrefetchOutcome(kind=AuxModelKind.LORA, name="styleA", ok=True, requesting_job_ids=[job.id_]),
            ],
        ),
    )
    assert tracker.are_job_aux_models_prepared(job) is True


async def test_reconcile_does_not_re_arm_a_job_with_a_live_deadline() -> None:
    """A job whose result never arrives is faulted at its deadline exactly once; the sweep never re-arms it."""
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    coordinator, sender, _state, clock = _make(tracker, timeout=60.0)
    coordinator.on_job_popped(job)
    assert len(sender.calls) == 1

    # While the deadline is live, repeated sweeps must not issue a second request for the same job.
    clock.now += 30.0
    coordinator.reconcile_and_refresh_pins()
    coordinator.reconcile_and_refresh_pins()
    assert len(sender.calls) == 1

    # The result never arrives; the deadline faults the job exactly once.
    clock.now += 40.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT
    # A follow-up sweep does not resurrect or re-request the now-faulted job.
    coordinator.reconcile_and_refresh_pins()
    assert len(sender.calls) == 1


async def test_reconcile_marks_prepared_when_everything_already_cached() -> None:
    """A pending aux job whose files are all cached (no deadline) is marked prepared by the sweep, no request."""
    tracker = JobTracker()
    job = _job(loras=[_lora("already")])
    await track_popped_job_async(tracker, job)
    tracker.mark_aux_prefetched("already", is_version=False, is_ti=False)
    coordinator, sender, _state, _clock = _make(tracker)

    coordinator.reconcile_and_refresh_pins()

    assert sender.calls == []
    assert tracker.are_job_aux_models_prepared(job) is True


async def test_pins_only_update_sent_when_pin_set_shrinks_after_completion() -> None:
    """Pins shrink after a job completes: the next periodic step emits one pins-only update, then none."""
    tracker = JobTracker()
    job_a = _job(loras=[_lora("keep")])
    job_b = _job(loras=[_lora("drop")])
    await track_popped_job_async(tracker, job_a)
    await track_popped_job_async(tracker, job_b)
    pins = _PinSenderSpy()
    coordinator, _sender, _state, _clock = _make(tracker, pin_sender=pins)
    coordinator.on_job_popped(job_a)
    coordinator.on_job_popped(job_b)

    # The pin set is current (both jobs' files) after the pops, so a refresh sends nothing.
    coordinator.reconcile_and_refresh_pins()
    assert pins.calls == []

    # job_b completes and leaves the pinned stages, so its file is no longer pinned.
    tracker.handle_job_fault_now(job_b, retryable=False, fault_reason="done")
    coordinator.reconcile_and_refresh_pins()
    assert len(pins.calls) == 1
    assert {(p.kind, p.name) for p in pins.calls[0]} == {(AuxModelKind.LORA, "keep")}

    # An identical consecutive pin set produces no second message (coalesced on change only).
    coordinator.reconcile_and_refresh_pins()
    assert len(pins.calls) == 1


async def test_deadline_defers_while_file_in_flight_then_prepares_on_completion() -> None:
    """A deadline that expires while the file is still downloading defers the fault, then completes normally.

    The job is not faulted at the original deadline because the downloader still shows its file in flight; when
    the download finishes (after that original deadline) the result outcome prepares the job as usual.
    """
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    in_flight = _InFlightSpy()
    coordinator, _sender, state, clock = _make(tracker, timeout=60.0, in_flight=in_flight)
    coordinator.on_job_popped(job)

    # The file is still transferring (bytes present) when the deadline passes, so the fault is deferred.
    in_flight.map = {"styleA": (1_000, 10_000)}
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    assert state.lora_download_backoff.strikes == 0
    assert coordinator.has_live_deadline(job.id_) is True

    # The download completes after the original deadline; the success outcome still prepares the job.
    in_flight.map = {}
    coordinator.on_prefetch_result(_ok_message(job, "styleA"))
    assert tracker.are_job_aux_models_prepared(job) is True


async def test_deadline_defers_to_cap_then_faults_when_bytes_unreported() -> None:
    """A file in flight but reporting no bytes defers up to the cap (three budgets total), then faults."""
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    in_flight = _InFlightSpy()
    coordinator, _sender, state, clock = _make(tracker, timeout=60.0, in_flight=in_flight)
    coordinator.on_job_popped(job)

    # The engine cannot report progress (zero bytes): the first two expiries defer, bounded by the cap.
    in_flight.map = {"styleA": (0, 0)}
    for _ in range(2):
        clock.now += 61.0
        coordinator.scan_deadlines()
        assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
        assert state.lora_download_backoff.strikes == 0

    # The third expiry is out of deferral budget, so the job faults with the usual deadline semantics.
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT
    assert state.lora_download_backoff.strikes == 1


async def test_deadline_defers_once_then_faults_when_reported_bytes_stall() -> None:
    """A file reporting bytes that stop advancing defers exactly once, then faults at the next expiry."""
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    in_flight = _InFlightSpy()
    coordinator, _sender, state, clock = _make(tracker, timeout=60.0, in_flight=in_flight)
    coordinator.on_job_popped(job)

    # First expiry: bytes are reported but there is no prior observation to call it stalled, so it defers.
    in_flight.map = {"styleA": (2_000, 10_000)}
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    assert state.lora_download_backoff.strikes == 0

    # Next expiry: the reported bytes have not advanced, so the stall is detected and the job faults.
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT
    assert state.lora_download_backoff.strikes == 1


async def test_deadline_faults_when_no_matching_file_in_flight() -> None:
    """An expiry with no matching in-flight file faults immediately, exactly as before deferral existed."""
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    in_flight = _InFlightSpy()
    coordinator, _sender, state, clock = _make(tracker, timeout=60.0, in_flight=in_flight)
    coordinator.on_job_popped(job)

    # A different file is downloading, but not the one this job needs, so nothing defers its fault.
    in_flight.map = {"other-file": (500, 1_000)}
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT
    assert state.lora_download_backoff.strikes == 1


async def test_deferral_never_resurrects_a_job_that_moved_on() -> None:
    """A file in flight for a job that already left the pending queue neither defers nor resurrects it."""
    tracker = JobTracker()
    job = _job(loras=[_lora("styleA")])
    await track_popped_job_async(tracker, job)
    in_flight = _InFlightSpy()
    coordinator, _sender, state, clock = _make(tracker, timeout=60.0, in_flight=in_flight)
    coordinator.on_job_popped(job)

    # The job faults out of the pending queue while its file is still in flight (its deadline still tracked).
    tracker.handle_job_fault_now(job, retryable=False, fault_reason="gone")
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT

    in_flight.map = {"styleA": (1_000, 10_000)}
    clock.now += 61.0
    coordinator.scan_deadlines()

    # It is not deferred back to pending, not prepared, and no spurious backoff strike is manufactured.
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT
    assert tracker.are_job_aux_models_prepared(job) is False
    assert state.lora_download_backoff.strikes == 0
    assert coordinator.has_live_deadline(job.id_) is False
