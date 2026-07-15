"""Reproduction tests for the textual-inversion (TI) prefetch failure contract.

These encode the behaviour a failing or terminally-rejected textual inversion should have at the parent-side
prefetch coordinator, mirroring the contract a LoRA already enjoys: a file the download pipeline can never
place on disk must let its jobs dispatch without it (recorded as skipped so it is neither re-requested nor
re-faulted for later jobs) rather than faulting every job that references it and burning each job's inference
budget on the identical doomed fetch. The deadline-deferral tests document how an in-flight TI transfer is
treated at the deadline backstop.
"""

from __future__ import annotations

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry, TIPayloadEntry
from horde_sdk.ai_horde_api.fields import GenerationID

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    AuxModelKind,
    AuxModelRef,
    AuxPrefetchEntry,
    AuxPrefetchOutcome,
    HordeAuxPrefetchResultMessage,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.models.aux_prefetch_coordinator import (
    _AUX_REFETCH_COOLDOWN_SECONDS,
    AuxPrefetchCoordinator,
)
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
        pin_sender=_PinSenderSpy(),
        in_flight_provider=in_flight if in_flight is not None else _InFlightSpy(),
        clock=the_clock,
    )
    return coordinator, sender, state, the_clock


def _result(*outcomes: AuxPrefetchOutcome) -> HordeAuxPrefetchResultMessage:
    """Wrap one or more per-entry outcomes into a result message from the download process."""
    return HordeAuxPrefetchResultMessage(
        process_id=9000,
        process_launch_identifier=1,
        info="r",
        outcomes=list(outcomes),
    )


def _ti_rejection(name: str, *job_ids: GenerationID) -> AuxPrefetchOutcome:
    """A terminal (non-retryable) rejection outcome for a TI the fetch API permanently refuses."""
    return AuxPrefetchOutcome(
        kind=AuxModelKind.TI,
        name=name,
        ok=False,
        retryable=False,
        rejection_reason="rejected",
        requesting_job_ids=list(job_ids),
    )


def _ti_failure(name: str, *job_ids: GenerationID) -> AuxPrefetchOutcome:
    """A plain retryable download failure outcome for a TI (a generic transfer error, not a rejection)."""
    return AuxPrefetchOutcome(
        kind=AuxModelKind.TI,
        name=name,
        ok=False,
        retryable=True,
        detail="download failed",
        requesting_job_ids=list(job_ids),
    )


def _ti_entries_for(sender: _SenderSpy, job_id: GenerationID | None, name: str) -> list[AuxPrefetchEntry]:
    """Every TI prefetch entry naming ``name`` that was requested on behalf of ``job_id``."""
    found: list[AuxPrefetchEntry] = []
    for entries, _pins in sender.calls:
        for entry in entries:
            if entry.kind is AuxModelKind.TI and entry.name == name and entry.requesting_job_id == job_id:
                found.append(entry)
    return found


async def test_terminally_rejected_ti_dispatches_jobs_without_it() -> None:
    """A textual inversion the fetch API permanently rejects lets its job dispatch without the file.

    A rejected TI will never be on disk, so the job that references it must be prepared to dispatch without it
    (staying pending inference), exactly as a terminally-rejected LoRA does, rather than being faulted.
    """
    tracker = JobTracker()
    job = _job(tis=[TIPayloadEntry(name="bad-ti")])
    await track_popped_job_async(tracker, job)
    coordinator, _sender, _state, _clock = _make(tracker)
    coordinator.on_job_popped(job)

    coordinator.on_prefetch_result(_result(_ti_rejection("bad-ti", job.id_)))

    assert tracker.are_job_aux_models_prepared(job) is True
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE


async def test_failed_ti_is_memoized_across_jobs() -> None:
    """A TI terminally rejected for one job is not re-fetched or re-faulted for a later job.

    Once a TI is terminally rejected, it is remembered as skipped: the first job dispatches without it and a
    second job referencing the same TI must have no fresh prefetch request issued for that file and must not be
    faulted on its account. The doomed fetch is remembered, not repeated per job.
    """
    tracker = JobTracker()
    job_a = _job(tis=[TIPayloadEntry(name="X")])
    await track_popped_job_async(tracker, job_a)
    coordinator, sender, _state, _clock = _make(tracker)
    coordinator.on_job_popped(job_a)

    coordinator.on_prefetch_result(_result(_ti_rejection("X", job_a.id_)))
    # The first job dispatches without the rejected file.
    assert tracker.are_job_aux_models_prepared(job_a) is True
    assert tracker.get_stage(job_a.id_) == JobStage.PENDING_INFERENCE

    job_b = _job(tis=[TIPayloadEntry(name="X")])
    await track_popped_job_async(tracker, job_b)
    coordinator.on_job_popped(job_b)

    # No fresh prefetch request for the known-doomed TI is issued on the new job's behalf.
    assert _ti_entries_for(sender, job_b.id_, "X") == []
    # The new job is not faulted by the remembered rejection.
    assert tracker.get_stage(job_b.id_) != JobStage.PENDING_SUBMIT


async def test_repeated_ti_plain_failure_is_terminal_and_salvages_arming_backoff() -> None:
    """A TI whose plain download fails repeatedly is served without it terminally and arms the TI backoff.

    A repeated plain (non-rejection) download failure means the download path is presumed sick, so retrying is
    futile and the reference is classified terminal, exactly as a LoRA does. Under the salvage contract a
    terminal classification memoizes the reference and dispatches the job without the embedding rather than
    faulting it, mirroring the inference path. The second identical failure lands while the first strike still
    holds an active window, so it is terminal and the TI download backoff is armed to withhold further attempts.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(2)
    job = _job(tis=[TIPayloadEntry(name="X")])
    await track_popped_job_async(tracker, job)
    coordinator, _sender, state, _clock = _make(tracker, clock=clock)
    coordinator.on_job_popped(job)

    coordinator.on_prefetch_result(_result(_ti_failure("X", job.id_)))
    coordinator.reconcile_and_refresh_pins()
    coordinator.on_prefetch_result(_result(_ti_failure("X", job.id_)))

    assert tracker.is_ti_skipped("X") is True
    assert tracker.are_job_aux_models_prepared(job) is True
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    assert state.ti_download_backoff.is_escalation_active(clock.now) is True


async def test_ti_transient_failure_recovers_via_reconcile_and_dispatches_with_file() -> None:
    """A transient TI failure is healed by the parent's reconcile loop after a cooldown, then dispatches.

    In-process download retries are not owned here: a single (retryable) plain failure leaves the job pending.
    The reference is held in its post-failure cooldown, so a reconcile within that window does not re-enter the
    failing download path; once the cooldown lapses the periodic reconcile sweep re-issues a fresh prefetch
    request, and a later success outcome prepares the job with the file present. Recovery is driven by the
    parent's reconcile-and-pin-refresh loop, not by the coordinator re-downloading in place.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(2)
    job = _job(tis=[TIPayloadEntry(name="emb")])
    await track_popped_job_async(tracker, job)
    coordinator, sender, _state, _clock = _make(tracker, clock=clock)
    coordinator.on_job_popped(job)

    coordinator.on_prefetch_result(_result(_ti_failure("emb", job.id_)))
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE

    # A reconcile within the reference's cooldown must not re-enter the failing path yet.
    requests_before = len(_ti_entries_for(sender, job.id_, "emb"))
    coordinator.reconcile_and_refresh_pins()
    assert len(_ti_entries_for(sender, job.id_, "emb")) == requests_before

    # Once the cooldown lapses the reconcile loop re-requests the aux-bearing job.
    clock.now += _AUX_REFETCH_COOLDOWN_SECONDS + 1.0
    coordinator.reconcile_and_refresh_pins()
    assert len(_ti_entries_for(sender, job.id_, "emb")) == requests_before + 1

    coordinator.on_prefetch_result(
        _result(AuxPrefetchOutcome(kind=AuxModelKind.TI, name="emb", ok=True, requesting_job_ids=[job.id_])),
    )
    assert tracker.are_job_aux_models_prepared(job) is True
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE


async def test_mixed_aux_job_survives_ti_rejection() -> None:
    """A job whose LoRAs all fetch but whose TI is terminally rejected still dispatches (prepared, not faulted).

    The successful LoRAs plus a skipped TI make the job's auxiliary set ready without the rejected file, so the
    job is prepared and stays pending inference rather than being faulted by the one rejection.
    """
    tracker = JobTracker()
    job = _job(loras=[_lora("one"), _lora("two")], tis=[TIPayloadEntry(name="bad-ti")])
    await track_popped_job_async(tracker, job)
    coordinator, _sender, _state, _clock = _make(tracker)
    coordinator.on_job_popped(job)

    coordinator.on_prefetch_result(
        _result(
            AuxPrefetchOutcome(kind=AuxModelKind.LORA, name="one", ok=True, requesting_job_ids=[job.id_]),
            AuxPrefetchOutcome(kind=AuxModelKind.LORA, name="two", ok=True, requesting_job_ids=[job.id_]),
            _ti_rejection("bad-ti", job.id_),
        ),
    )

    assert tracker.are_job_aux_models_prepared(job) is True
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE


@pytest.mark.parametrize("attempts", [1, 2, 5])
async def test_attempt_policy_does_not_change_ti_rejection_outcome(attempts: int) -> None:
    """A terminally-rejected TI never faults its job, whatever the configured inference-attempt budget.

    A permanent rejection is not a transient failure to retry, so the size of the attempt budget must not
    change the outcome: the job dispatches without the file at every policy.
    """
    tracker = JobTracker()
    tracker.set_retry_policy(attempts)
    job = _job(tis=[TIPayloadEntry(name="bad-ti")])
    await track_popped_job_async(tracker, job)
    coordinator, _sender, _state, _clock = _make(tracker)
    coordinator.on_job_popped(job)

    coordinator.on_prefetch_result(_result(_ti_rejection("bad-ti", job.id_)))

    assert tracker.get_stage(job.id_) != JobStage.PENDING_SUBMIT
    assert tracker.are_job_aux_models_prepared(job) is True


async def test_progressing_ti_download_defers_deadline() -> None:
    """A TI whose transfer keeps advancing is not faulted at the deadline while it makes progress.

    The deadline backstop defers for a file the downloader still shows in flight and advancing, so a slow but
    alive TI download is not punished by the deadline the way a stalled one is.
    """
    tracker = JobTracker()
    job = _job(tis=[TIPayloadEntry(name="emb")])
    await track_popped_job_async(tracker, job)
    in_flight = _InFlightSpy()
    coordinator, _sender, _state, clock = _make(tracker, timeout=60.0, in_flight=in_flight)
    coordinator.on_job_popped(job)

    in_flight.map = {"emb": (1_000, 10_000)}
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE

    in_flight.map = {"emb": (5_000, 10_000)}
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE


async def test_stalled_ti_download_faults_at_deadline() -> None:
    """A TI whose in-flight transfer stops advancing is faulted once the deadline deferral is spent.

    The deadline backstop defers once for an in-flight file, then faults when its reported bytes fail to move,
    so a wedged TI download does not leave the job pending forever.
    """
    tracker = JobTracker()
    job = _job(tis=[TIPayloadEntry(name="emb")])
    await track_popped_job_async(tracker, job)
    in_flight = _InFlightSpy()
    coordinator, _sender, _state, clock = _make(tracker, timeout=60.0, in_flight=in_flight)
    coordinator.on_job_popped(job)

    in_flight.map = {"emb": (2_000, 10_000)}
    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE

    clock.now += 61.0
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT
