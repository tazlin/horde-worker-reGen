"""Deadline handling of an aux prefetch whose download was requested but never entered in-flight.

The parent-side aux-prefetch coordinator arms a per-job deadline at pop and, when it expires, faults a job
whose auxiliary files are not yet on disk. That backstop defers only while the downloader reports a file
actively in flight (:meth:`AuxPrefetchCoordinator._defer_deadline_if_in_flight`). A reference the downloader
never began fetching (its lane starved, so the file was neither placed nor reported in flight) therefore
reaches the deadline unmatched and is faulted.

The standing repo ruling is that the prefetch optimization must never change a job's outcome versus reaching
inference: the coordinator already serves a job without an auxiliary file the fetch API rejects
(:meth:`AuxPrefetchCoordinator._skip_rejected_aux`) and salvages co-waiters on a terminal plain failure. A
job whose reference merely never started downloading is in the same position (inference would run without the
missing file), so faulting it drops otherwise-servable work that the deadline was only meant to bound.

RED contract: a job whose aux entries were requested but never entered an in-flight download by deadline
expiry is salvaged (dispatched to inference without the file), not faulted.

Controls bound the salvage: a download that IS in flight at expiry still defers, an in-flight download whose
reported bytes never advance still faults once the deferral budget is spent, and a reference whose incident
already memoized it skipped is salvaged through that skip (the adjacent salvage path). The reproductions and
controls drive the real coordinator over a real :class:`JobTracker` sharing an injected clock, so no live
worker is needed.
"""

from __future__ import annotations

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry, TIPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    AuxModelKind,
    AuxModelRef,
    AuxPrefetchEntry,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.models.aux_prefetch_coordinator import (
    _MAX_DEADLINE_DEFERRALS,
    AuxPrefetchCoordinator,
)
from tests.process_management.conftest import make_job_pop_response, track_popped_job_async

# The per-job deadline used by these tests; a value the clock is advanced past to force expiry.
_DEADLINE_SECONDS = 30.0


class _Clock:
    """A hand-advanceable clock so deadline and deferral behaviour are deterministic."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _SenderSpy:
    """Records the prefetch requests the coordinator would send to the downloader."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[AuxPrefetchEntry], list[AuxModelRef]]] = []

    def __call__(self, entries: list[AuxPrefetchEntry], pins: list[AuxModelRef]) -> None:
        self.calls.append((entries, pins))


def _make(
    tracker: JobTracker,
    *,
    clock: _Clock,
    timeout: float = _DEADLINE_SECONDS,
    in_flight: dict[str, tuple[int, int]] | None = None,
) -> tuple[AuxPrefetchCoordinator, _SenderSpy, WorkerState]:
    """Build a coordinator over the tracker sharing an injected clock and an optional in-flight view.

    The in-flight provider reads the ``in_flight`` mapping live on each call, so a test can seed it once and
    then mutate it between deadline scans to model a download starting, advancing, or stalling.
    """
    state = WorkerState()
    sender = _SenderSpy()
    coordinator = AuxPrefetchCoordinator(
        job_tracker=tracker,
        state=state,
        prefetch_sender=sender,
        download_timeout_provider=lambda: timeout,
        pin_sender=lambda _pins: None,
        in_flight_provider=(lambda: dict(in_flight)) if in_flight is not None else dict,
        clock=clock,
    )
    return coordinator, sender, state


def _lora(name: str) -> LorasPayloadEntry:
    return LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=False)


async def _job_of(
    tracker: JobTracker,
    coordinator: AuxPrefetchCoordinator,
    *,
    loras: list[LorasPayloadEntry] | None = None,
    tis: list[TIPayloadEntry] | None = None,
    n_iter: int = 1,
) -> ImageGenerateJobPopResponse:
    """Pop-and-track a fresh job carrying the given references and run its pop-time prefetch trigger."""
    job = make_job_pop_response("stable_diffusion", loras=loras, tis=tis, n_iter=n_iter)
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    return job


def _faulted(tracker: JobTracker, job: ImageGenerateJobPopResponse) -> bool:
    """Whether the tracker has moved the job to the fault-submit stage."""
    assert job.id_ is not None
    return tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT


def _salvaged(tracker: JobTracker, job: ImageGenerateJobPopResponse) -> bool:
    """Whether the job is dispatchable without the missing file (aux set counted ready, still pending)."""
    assert job.id_ is not None
    return tracker.are_job_aux_models_prepared(job) and tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE


# Reproduction: a reference requested at pop but never seen in flight is salvaged at the deadline, not faulted.


async def test_never_started_download_at_deadline_is_salvaged_not_faulted() -> None:
    """A job whose LoRA was requested but never entered an in-flight download is served without it.

    The downloader lane never began fetching the reference, so at deadline expiry the coordinator sees nothing
    in flight for the job. Faulting it would drop work inference could still serve without the missing LoRA,
    so the deadline backstop must dispatch the job without the file (recording the reference skipped, exactly
    as the rejection path does) rather than fault it.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(1)
    coordinator, sender, _state = _make(tracker, clock=clock, in_flight={})

    job = await _job_of(tracker, coordinator, loras=[_lora("never-started")])
    # The request was issued at pop, but the downloader never reported it in flight.
    assert sender.calls, "precondition: a prefetch request was issued at pop"
    assert tracker.is_lora_skipped(_lora("never-started")) is False

    clock.now += _DEADLINE_SECONDS + 1.0
    coordinator.scan_deadlines()

    assert not _faulted(tracker, job)
    assert _salvaged(tracker, job)


@pytest.mark.parametrize("kind", [AuxModelKind.LORA, AuxModelKind.TI])
async def test_never_started_download_salvage_holds_for_both_kinds(kind: AuxModelKind) -> None:
    """The never-started salvage lives on the shared kind-agnostic deadline path, for LoRA and TI alike.

    A textual inversion whose download never started is served without it exactly as a LoRA is, so the fix
    belongs on the shared deadline backstop and not one kind's branch.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(1)
    coordinator, _sender, _state = _make(tracker, clock=clock, in_flight={})

    if kind is AuxModelKind.LORA:
        job = await _job_of(tracker, coordinator, loras=[_lora("never-started")])
    else:
        job = await _job_of(tracker, coordinator, tis=[TIPayloadEntry(name="never-started")])

    clock.now += _DEADLINE_SECONDS + 1.0
    coordinator.scan_deadlines()

    assert not _faulted(tracker, job)
    assert _salvaged(tracker, job)


async def test_never_started_download_cowaiters_are_all_salvaged() -> None:
    """Several jobs each awaiting a never-started reference are all salvaged at their deadlines, none faulted.

    A starved downloader lane withholds the same reference from every job that asked for it. The fault count
    must not scale with how many jobs share the doomed-but-never-started download: each is served without the
    file rather than faulted.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(1)
    coordinator, _sender, _state = _make(tracker, clock=clock, in_flight={})

    jobs = [await _job_of(tracker, coordinator, loras=[_lora("never-started")]) for _ in range(3)]

    clock.now += _DEADLINE_SECONDS + 1.0
    coordinator.scan_deadlines()

    for job in jobs:
        assert not _faulted(tracker, job)
        assert _salvaged(tracker, job)


# Controls: the in-flight paths the salvage must not disturb (these pin current, correct behaviour).


async def test_in_flight_progressing_download_defers_at_deadline() -> None:
    """A download reporting advancing bytes at expiry defers rather than faulting (control, passes today).

    The deadline is a backstop, not the primary failure detector: while the downloader shows the file in
    flight and making progress, the job's deadline is extended instead of faulted.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(1)
    in_flight: dict[str, tuple[int, int]] = {"live": (10, 100)}
    coordinator, _sender, _state = _make(tracker, clock=clock, in_flight=in_flight)

    job = await _job_of(tracker, coordinator, loras=[_lora("live")])

    clock.now += _DEADLINE_SECONDS + 1.0
    in_flight["live"] = (40, 100)  # bytes advanced since the request
    coordinator.scan_deadlines()

    assert not _faulted(tracker, job)
    assert not _salvaged(tracker, job)  # still pending its in-flight file, neither faulted nor dispatched


async def test_stuck_in_flight_download_still_faults_after_deferrals_exhaust() -> None:
    """An in-flight download whose bytes never advance still faults once the deferral budget is spent.

    A download that is reported in flight but makes no byte progress is deferred a bounded number of times
    and then faulted, so a genuinely stalled transfer cannot postpone its fault forever. This bounds the
    never-started salvage to references with no in-flight download at all.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(1)
    in_flight: dict[str, tuple[int, int]] = {"stalled": (50, 100)}
    coordinator, _sender, _state = _make(tracker, clock=clock, in_flight=in_flight)

    job = await _job_of(tracker, coordinator, loras=[_lora("stalled")])

    # First expiry: bytes present but unchanged from the request has no prior to compare, so it defers once.
    # Each further expiry finds the same byte count (stalled) and consumes one more deferral until the cap.
    for _ in range(_MAX_DEADLINE_DEFERRALS + 1):
        clock.now += _DEADLINE_SECONDS + 1.0
        coordinator.scan_deadlines()

    assert _faulted(tracker, job)
