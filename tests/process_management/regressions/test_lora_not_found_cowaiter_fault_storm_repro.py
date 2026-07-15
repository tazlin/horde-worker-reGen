"""Concurrent co-waiter behaviour of the parent-side aux-prefetch coordinator on an unfetchable reference.

An auxiliary reference that can never land on disk (a not-found LoRA the fetch API surfaces no rejection
reason for, so the download process laundered it into a plain retryable failure) reaches its terminal verdict
once. The download process reports one deduplicated outcome per file, so that single terminal outcome can name
every job still waiting on the reference. The inference path treats such a LoRA as ignorable and generates
without it; the rejection path (:meth:`AuxPrefetchCoordinator._skip_rejected_aux`) dispatches every co-waiting
job without the file and faults none. The contract these tests hold is that the plain-failure path must not
diverge from that: when the terminal verdict lands, every job co-waiting on the unfetchable reference is served
without it, none faulted, since the same verdict memoizes the reference as skippable for all of them.

The suite is organised as reproductions of the concurrent salvage behaviour, controls that pin the adjacent
paths (a surfaced rejection, a single job, a later job, a genuine transient), and an exploratory matrix that
varies co-waiter count, mixed valid/unfetchable sets, auxiliary kind, retry policy, the deadline backstop, a
downloader reset, and batch shape to map where the salvage holds. Every test drives the real coordinator over a
real :class:`JobTracker` with an injected clock, so no live worker is needed.
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
from horde_worker_regen.process_management.models.aux_prefetch_coordinator import AuxPrefetchCoordinator
from tests.process_management.conftest import make_job_pop_response, track_popped_job_async


class _Clock:
    """A hand-advanceable clock so backoff, cooldown, and deadline behaviour are deterministic."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _SenderSpy:
    """Records the (entries, pins) of each prefetch request the coordinator would send to the downloader."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[AuxPrefetchEntry], list[AuxModelRef]]] = []

    def __call__(self, entries: list[AuxPrefetchEntry], pins: list[AuxModelRef]) -> None:
        self.calls.append((entries, pins))


def _make(
    tracker: JobTracker,
    *,
    clock: _Clock,
    timeout: float = 120.0,
    in_flight: dict[str, tuple[int, int]] | None = None,
) -> tuple[AuxPrefetchCoordinator, _SenderSpy, WorkerState]:
    """Build a coordinator over the given tracker sharing an injected clock and an optional in-flight view."""
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


def _lora(name: str, *, is_version: bool = False) -> LorasPayloadEntry:
    return LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=is_version)


def _job(
    *,
    loras: list[LorasPayloadEntry] | None = None,
    tis: list[TIPayloadEntry] | None = None,
    n_iter: int = 1,
) -> ImageGenerateJobPopResponse:
    return make_job_pop_response("stable_diffusion", loras=loras, tis=tis, n_iter=n_iter)


def _result(*outcomes: AuxPrefetchOutcome) -> HordeAuxPrefetchResultMessage:
    return HordeAuxPrefetchResultMessage(
        process_id=9000,
        process_launch_identifier=1,
        info="r",
        outcomes=list(outcomes),
    )


def _laundered_failure(
    name: str,
    *job_ids: GenerationID,
    kind: AuxModelKind = AuxModelKind.LORA,
) -> AuxPrefetchOutcome:
    """The unfetchable-reference outcome shape: a plain retryable failure carrying no rejection reason.

    This is what the download process emits for a not-found reference the fetch API surfaced no reason for. One
    such outcome can name every job that shared the deduplicated download.
    """
    return AuxPrefetchOutcome(
        kind=kind,
        name=name,
        ok=False,
        retryable=True,
        detail="download failed",
        requesting_job_ids=list(job_ids),
    )


def _rejection(
    name: str,
    reason: str,
    *job_ids: GenerationID,
    kind: AuxModelKind = AuxModelKind.LORA,
) -> AuxPrefetchOutcome:
    """A terminal surfaced-rejection outcome for a reference the fetch API permanently refuses."""
    return AuxPrefetchOutcome(
        kind=kind,
        name=name,
        ok=False,
        retryable=False,
        rejection_reason=reason,
        requesting_job_ids=list(job_ids),
    )


def _success(name: str, *job_ids: GenerationID, kind: AuxModelKind = AuxModelKind.LORA) -> AuxPrefetchOutcome:
    return AuxPrefetchOutcome(kind=kind, name=name, ok=True, requesting_job_ids=list(job_ids))


def _faulted(tracker: JobTracker, jobs: list[ImageGenerateJobPopResponse]) -> list[ImageGenerateJobPopResponse]:
    """Every job of ``jobs`` the tracker has moved to the fault-submit stage."""
    return [job for job in jobs if tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT]


def _prepared(tracker: JobTracker, job: ImageGenerateJobPopResponse) -> bool:
    """Whether ``job`` is dispatchable without the unfetchable file (aux set counted ready, still pending)."""
    return tracker.are_job_aux_models_prepared(job) and tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE


async def _job_of(
    tracker: JobTracker,
    coordinator: AuxPrefetchCoordinator,
    *,
    loras: list[LorasPayloadEntry] | None = None,
    tis: list[TIPayloadEntry] | None = None,
    n_iter: int = 1,
) -> ImageGenerateJobPopResponse:
    """Pop-and-track a fresh job carrying the given auxiliary references and run its pop-time prefetch trigger."""
    job = _job(loras=loras, tis=tis, n_iter=n_iter)
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    return job


async def _prime_lora_incident(tracker: JobTracker, coordinator: AuxPrefetchCoordinator) -> None:
    """Register one prior LoRA download strike so the next same-class failure classifies terminal.

    Uses a throwaway reference distinct from any reference under test, so the class incident is active without
    touching the skip memoization or cooldown of the reference the test then exercises.
    """
    primer = await _job_of(tracker, coordinator, loras=[_lora("incident-primer")])
    coordinator.on_prefetch_result(_result(_laundered_failure("incident-primer", primer.id_)))


# Reproductions: several jobs co-waiting on one unfetchable reference when its terminal verdict lands.


async def test_two_cowaiting_jobs_on_terminal_not_found_are_all_salvaged() -> None:
    """Two jobs co-waiting on one unfetchable LoRA, terminal in a single shared outcome, are both served.

    The first laundered failure requeues the first job and activates the class incident. A second job for the
    same reference then joins the wait. When the deduplicated terminal outcome names both, the verdict that
    memoizes the reference as skippable dispatches every job it names without the file rather than faulting any:
    the prefetch optimization does not change an outcome the inference path would have served.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(2)
    coordinator, _sender, _state = _make(tracker, clock=clock)

    job_a = await _job_of(tracker, coordinator, loras=[_lora("ghost")])
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job_a.id_)))
    assert tracker.get_stage(job_a.id_) == JobStage.PENDING_INFERENCE

    job_b = await _job_of(tracker, coordinator, loras=[_lora("ghost")])
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job_a.id_, job_b.id_)))

    assert _faulted(tracker, [job_a, job_b]) == []
    assert _prepared(tracker, job_a)
    assert _prepared(tracker, job_b)


async def test_primed_cowaiters_on_terminal_not_found_are_all_salvaged() -> None:
    """Every co-waiter on a terminal unfetchable LoRA dispatches without it, faulting none.

    Once the reference is memoized as skippable, a job still waiting on it can dispatch without the file, so the
    terminal verdict salvages the co-waiters it names exactly as a job arriving after the incident is salvaged,
    and exactly as a surfaced rejection does. This is the parity contract that keeps a prefetch fault-free where
    the inference path would serve the job without the missing LoRA.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(2)
    coordinator, _sender, _state = _make(tracker, clock=clock)

    await _prime_lora_incident(tracker, coordinator)
    job_a = await _job_of(tracker, coordinator, loras=[_lora("ghost")])
    job_b = await _job_of(tracker, coordinator, loras=[_lora("ghost")])

    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job_a.id_, job_b.id_)))

    assert _faulted(tracker, [job_a, job_b]) == []
    assert _prepared(tracker, job_a)
    assert _prepared(tracker, job_b)


async def test_fresh_cowaiters_on_shared_terminal_outcome_are_all_salvaged() -> None:
    """Two fresh co-waiters served by one already-terminal shared outcome are both dispatched without the file.

    Neither job failed individually first: a prior unrelated strike made the class incident active, so the very
    first shared outcome for their reference is terminal and names both. No waiter is faulted no matter how many
    jobs happened to share the one download.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, _sender, _state = _make(tracker, clock=clock)

    await _prime_lora_incident(tracker, coordinator)
    job_b = await _job_of(tracker, coordinator, loras=[_lora("ghost")])
    job_c = await _job_of(tracker, coordinator, loras=[_lora("ghost")])

    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job_b.id_, job_c.id_)))

    assert _faulted(tracker, [job_b, job_c]) == []
    assert _prepared(tracker, job_b)
    assert _prepared(tracker, job_c)


# Controls: adjacent paths that already behave, bounding the reproductions to the concurrent plain-failure case.


@pytest.mark.parametrize("count", [2, 3, 5])
async def test_shared_rejection_salvages_all_cowaiters_zero_faults(count: int) -> None:
    """A surfaced terminal rejection naming several co-waiters dispatches them all without the file (control).

    The rejection path is the reference behaviour the plain-failure path is measured against: one deduplicated
    rejection outcome prepares every job it names and faults none, for any number of co-waiters.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    coordinator, _sender, _state = _make(tracker, clock=clock)

    jobs = [await _job_of(tracker, coordinator, loras=[_lora("refused")]) for _ in range(count)]
    coordinator.on_prefetch_result(_result(_rejection("refused", "invalid", *[job.id_ for job in jobs])))

    assert _faulted(tracker, jobs) == []
    for job in jobs:
        assert _prepared(tracker, job)


async def test_single_job_terminal_not_found_is_salvaged_and_memoized() -> None:
    """A single job on an unfetchable LoRA is served without it once the reference goes terminal (control).

    The sequential path: the first laundered failure requeues the job, the second (while the incident is active)
    is terminal, records the reference as skipped, and dispatches the job without it. This anchors that the
    concurrent reproductions differ only in how many jobs share the terminal verdict, not in the verdict itself.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, _sender, _state = _make(tracker, clock=clock)

    job = await _job_of(tracker, coordinator, loras=[_lora("ghost")])
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job.id_)))
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    assert tracker.is_lora_skipped(_lora("ghost")) is False

    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job.id_)))
    assert tracker.is_lora_skipped(_lora("ghost")) is True
    assert _faulted(tracker, [job]) == []
    assert _prepared(tracker, job)


async def test_later_job_after_terminal_incident_dispatches_without_lora() -> None:
    """A job arriving after the reference is memoized skipped dispatches without it, not faulting (control).

    The terminal verdict salvages future jobs through the memoized skip: a later job referencing the same file
    is neither re-requested into the failing path nor faulted. Confirms the memoization works for arrivals after
    the incident, complementing the reproductions that cover the jobs already waiting at the terminal instant.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, sender, _state = _make(tracker, clock=clock)

    first = await _job_of(tracker, coordinator, loras=[_lora("ghost")])
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", first.id_)))
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", first.id_)))
    assert tracker.is_lora_skipped(_lora("ghost")) is True

    sender.calls.clear()
    later = await _job_of(tracker, coordinator, loras=[_lora("ghost")])

    assert _faulted(tracker, [later]) == []
    assert _prepared(tracker, later)
    assert sender.calls == []


async def test_single_transient_lora_failure_requeues_and_arms_backoff_once() -> None:
    """A genuine single transient LoRA failure requeues the job and arms the backoff once (control).

    A retryable failure with attempts remaining leaves the job pending and registers one strike, so a transient
    outage is retried rather than skipped and the reference is not memoized. Distinguishes a recoverable outage
    from the unfetchable-reference case the reproductions target.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, _sender, state = _make(tracker, clock=clock)

    job = await _job_of(tracker, coordinator, loras=[_lora("net")])
    coordinator.on_prefetch_result(_result(_laundered_failure("net", job.id_)))

    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    assert state.lora_download_backoff.strikes == 1
    assert tracker.is_lora_skipped(_lora("net")) is False


# Exploratory matrix: vary queue shape, config, and system condition to map the concurrent-fault dynamic.


@pytest.mark.parametrize("count", [2, 3, 5, 8])
async def test_cowaiter_salvage_does_not_regress_with_queue_depth(count: int) -> None:
    """One terminal outcome naming many co-waiters serves them all without the file, whatever the queue depth.

    Probes whether the fault-free salvage holds independent of how many jobs shared the one unfetchable download.
    If any faulted job appears as the count grows, one bad reference silences a batch of otherwise-servable work.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, _sender, _state = _make(tracker, clock=clock)

    await _prime_lora_incident(tracker, coordinator)
    jobs = [await _job_of(tracker, coordinator, loras=[_lora("ghost")]) for _ in range(count)]

    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", *[job.id_ for job in jobs])))

    assert _faulted(tracker, jobs) == []
    for job in jobs:
        assert _prepared(tracker, job)


async def test_mixed_job_with_valid_and_unfetchable_lora_dispatches_with_the_valid_one() -> None:
    """A job wanting a valid and an unfetchable LoRA dispatches with the valid one rather than being faulted.

    The job carries a LoRA that caches and one that is unfetchable; a co-waiting job wants only the unfetchable
    one. Once the unfetchable reference reaches its terminal verdict and is memoized skipped, the mixed job's
    whole set is resolved (one cached, one skipped), so it dispatches with the valid LoRA, and the ghost-only
    co-waiter is likewise served without the file. Neither is faulted for the doomed reference they share.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, _sender, _state = _make(tracker, clock=clock)

    await _prime_lora_incident(tracker, coordinator)
    mixed = await _job_of(tracker, coordinator, loras=[_lora("good"), _lora("ghost")])
    ghost_only = await _job_of(tracker, coordinator, loras=[_lora("ghost")])

    coordinator.on_prefetch_result(
        _result(
            _success("good", mixed.id_),
            _laundered_failure("ghost", mixed.id_, ghost_only.id_),
        ),
    )

    assert tracker.is_lora_cached(_lora("good")) is True
    assert _faulted(tracker, [mixed, ghost_only]) == []
    assert _prepared(tracker, mixed)
    assert _prepared(tracker, ghost_only)


@pytest.mark.parametrize("kind", [AuxModelKind.LORA, AuxModelKind.TI])
async def test_terminal_unfetchable_reference_cowaiters_are_salvaged_for_both_kinds(kind: AuxModelKind) -> None:
    """Co-waiter salvage on an unfetchable reference holds for both LoRA and textual-inversion kinds.

    The salvage dynamic lives on the shared kind-agnostic failure path, so a not-found textual inversion serves
    its co-waiters without the file exactly as a not-found LoRA does. A divergence between kinds would mean the
    fix belongs on the shared path, not one kind's branch.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, _sender, _state = _make(tracker, clock=clock)

    def refs(name: str) -> dict[str, list[LorasPayloadEntry] | list[TIPayloadEntry]]:
        return {"loras": [_lora(name)]} if kind is AuxModelKind.LORA else {"tis": [TIPayloadEntry(name=name)]}

    primer = await _job_of(tracker, coordinator, **refs("primer"))  # type: ignore[arg-type]
    coordinator.on_prefetch_result(_result(_laundered_failure("primer", primer.id_, kind=kind)))

    job_a = await _job_of(tracker, coordinator, **refs("ghost"))  # type: ignore[arg-type]
    job_b = await _job_of(tracker, coordinator, **refs("ghost"))  # type: ignore[arg-type]
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job_a.id_, job_b.id_, kind=kind)))

    assert _faulted(tracker, [job_a, job_b]) == []
    assert _prepared(tracker, job_a)
    assert _prepared(tracker, job_b)


@pytest.mark.parametrize("attempts", [1, 2, 3])
async def test_retry_policy_does_not_change_cowaiter_salvage(attempts: int) -> None:
    """A terminal outcome serving its co-waiters without the file must not depend on the retry policy.

    A stricter or looser attempt budget changes when a single job's failure turns terminal, but the terminal
    verdict salvages the jobs it names regardless. Varying the policy probes whether attempt bookkeeping
    interacts with the shared-outcome salvage to fault a co-waiter under some budgets.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(attempts)
    coordinator, _sender, _state = _make(tracker, clock=clock)

    await _prime_lora_incident(tracker, coordinator)
    job_a = await _job_of(tracker, coordinator, loras=[_lora("ghost")])
    job_b = await _job_of(tracker, coordinator, loras=[_lora("ghost")])

    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job_a.id_, job_b.id_)))

    assert _faulted(tracker, [job_a, job_b]) == []
    assert _prepared(tracker, job_a)
    assert _prepared(tracker, job_b)


async def test_deadline_backstop_salvages_a_cowaiter_whose_reference_was_skipped() -> None:
    """A co-waiter reaching its deadline after its reference was memoized skipped is salvaged, not faulted.

    A reported terminal failure for a shared reference memoizes it skipped and salvages the co-waiters it names,
    but a co-waiter the failure did not name keeps its own live prefetch deadline. When that deadline expires the
    backstop must honor the memoized skip and dispatch the job without the file rather than fault it for a
    reference the incident already resolved. The backstop is a per-job timeout, so without this it would fault a
    job whose blocking reference is already known skippable.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, _sender, _state = _make(tracker, clock=clock, timeout=30.0, in_flight={})

    await _prime_lora_incident(tracker, coordinator)
    job_a = await _job_of(tracker, coordinator, loras=[_lora("ghost")])
    job_b = await _job_of(tracker, coordinator, loras=[_lora("ghost")])

    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job_a.id_)))
    assert tracker.is_lora_skipped(_lora("ghost")) is True
    assert tracker.get_stage(job_b.id_) == JobStage.PENDING_INFERENCE

    clock.now += 31.0
    coordinator.scan_deadlines()

    assert _faulted(tracker, [job_b]) == []
    assert _prepared(tracker, job_b)


async def test_deadline_backstop_still_faults_a_job_whose_reference_never_resolved() -> None:
    """The backstop still faults a timed-out job whose reference was never skipped (control for the salvage guard).

    The salvage guard must be scoped to references an incident actually memoized skipped: a job whose reference
    never resolved and never got skipped still reaches its deadline unprepared and is faulted, so the backstop
    keeps bounding a genuinely stuck prefetch rather than silently dispatching it without a file no verdict cleared.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(1)
    coordinator, _sender, _state = _make(tracker, clock=clock, timeout=30.0, in_flight={})

    job = await _job_of(tracker, coordinator, loras=[_lora("never-resolves")])
    assert tracker.is_lora_skipped(_lora("never-resolves")) is False

    clock.now += 31.0
    coordinator.scan_deadlines()

    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT


async def test_downloader_reset_before_terminal_still_salvages_cowaiters() -> None:
    """A downloader reset between the requeue and the terminal verdict must not fault the co-waiters.

    The background download process dying and restarting clears in-flight deadlines and cooldowns so pending
    jobs are re-requested against the fresh downloader. If the reference is still unfetchable after the reset,
    the eventual terminal verdict for the co-waiters still serves them without the file rather than faulting each
    job the reset re-armed.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, _sender, _state = _make(tracker, clock=clock)

    job_a = await _job_of(tracker, coordinator, loras=[_lora("ghost")])
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job_a.id_)))
    job_b = await _job_of(tracker, coordinator, loras=[_lora("ghost")])

    coordinator.on_downloader_reset()
    coordinator.reconcile_and_refresh_pins()

    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job_a.id_, job_b.id_)))

    assert _faulted(tracker, [job_a, job_b]) == []
    assert _prepared(tracker, job_a)
    assert _prepared(tracker, job_b)


async def test_batched_cowaiters_on_terminal_not_found_are_all_salvaged() -> None:
    """Multi-image (batched) jobs co-waiting on an unfetchable LoRA are served without it like single-image ones.

    A job requesting several images per pop resolves its auxiliary set the same way a single-image job does, so
    the batch shape must not change the fault-free salvage. Guards against a per-image accounting quirk faulting
    batched work a single-image job would have been served.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, _sender, _state = _make(tracker, clock=clock)

    await _prime_lora_incident(tracker, coordinator)
    job_a = await _job_of(tracker, coordinator, loras=[_lora("ghost")], n_iter=4)
    job_b = await _job_of(tracker, coordinator, loras=[_lora("ghost")], n_iter=4)

    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job_a.id_, job_b.id_)))

    assert _faulted(tracker, [job_a, job_b]) == []
    assert _prepared(tracker, job_a)
    assert _prepared(tracker, job_b)
