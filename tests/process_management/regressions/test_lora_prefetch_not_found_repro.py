"""Reproduction tests for the ad-hoc LoRA not-found fault storm at the parent-side prefetch coordinator.

A LoRA reference that does not exist upstream (its metadata endpoint returns a definitive not-found) can never
be placed on disk. The download process should report that as a terminal rejection so the coordinator skips
the LoRA, dispatches the job without it, and memoizes the verdict so later jobs referencing the same LoRA are
neither re-requested nor re-faulted, exactly as a too-large LoRA already behaves. The production defect is
that the not-found is laundered into a plain retryable failure (``ok=False``, ``rejection_reason=None``,
``retryable=True``): the coordinator then faults every job wanting that LoRA and escalates the LoRA download
backoff once per job, because nothing memoizes the doomed reference.

These encode the desired contract at the coordinator seam (which repo the fix lands in is left open): a
repeated terminal-condition failure for one LoRA must not fault more than one job; a surfaced terminal
rejection dispatches its jobs without the file and is memoized; a genuine transient failure still requeues and
arms the backoff once; and the LoRA backoff withholds LoRA support on the next pop without disturbing
already-queued jobs. The exploratory matrix probes mixed, concurrent, decay-boundary, and cross-kind shapes.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

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
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.aux_download_backoff import STRIKE_DECAY_SECONDS, AuxDownloadBackoff
from horde_worker_regen.process_management.models.aux_prefetch_coordinator import AuxPrefetchCoordinator
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_test_api_sessions,
    make_test_runtime_config,
    track_popped_job_async,
)


class _Clock:
    """A hand-advanceable clock so backoff and deadline behaviour is deterministic."""

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


def _make(
    tracker: JobTracker,
    *,
    state: WorkerState | None = None,
    clock: _Clock | None = None,
    timeout: float = 120.0,
) -> tuple[AuxPrefetchCoordinator, _SenderSpy, WorkerState, _Clock]:
    """Build a coordinator over the given tracker, sharing ``state`` when supplied so a popper can read it."""
    the_state = state if state is not None else WorkerState()
    sender = _SenderSpy()
    the_clock = clock if clock is not None else _Clock()
    coordinator = AuxPrefetchCoordinator(
        job_tracker=tracker,
        state=the_state,
        prefetch_sender=sender,
        download_timeout_provider=lambda: timeout,
        pin_sender=lambda _pins: None,
        in_flight_provider=dict,
        clock=the_clock,
    )
    return coordinator, sender, the_state, the_clock


def _make_popper(state: WorkerState, tracker: JobTracker, *, bridge_data: Mock) -> JobPopper:
    """Build a real JobPopper over the shared state and tracker (background downloads enabled by default)."""
    return JobPopper(
        state=state,
        process_map=ProcessMap({}),
        job_tracker=tracker,
        shutdown_manager=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        api_sessions=make_test_api_sessions(horde_client_session=Mock(), aiohttp_session=Mock()),
        max_inference_processes=2,
        max_concurrent_inference_processes=1,
        dry_run_skip_api=False,
    )


def _lora(name: str, *, is_version: bool = False) -> LorasPayloadEntry:
    return LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=is_version)


def _job(
    *, loras: list[LorasPayloadEntry] | None = None, tis: list[TIPayloadEntry] | None = None
) -> ImageGenerateJobPopResponse:
    return make_job_pop_response("stable_diffusion", loras=loras, tis=tis)


def _result(*outcomes: AuxPrefetchOutcome) -> HordeAuxPrefetchResultMessage:
    return HordeAuxPrefetchResultMessage(
        process_id=9000,
        process_launch_identifier=1,
        info="r",
        outcomes=list(outcomes),
    )


def _lora_rejection(name: str, reason: str, *job_ids: GenerationID) -> AuxPrefetchOutcome:
    """A terminal (surfaced) rejection outcome for a LoRA the fetch API permanently refuses."""
    return AuxPrefetchOutcome(
        kind=AuxModelKind.LORA,
        name=name,
        ok=False,
        retryable=False,
        rejection_reason=reason,
        requesting_job_ids=list(job_ids),
    )


def _lora_laundered_failure(name: str, *job_ids: GenerationID) -> AuxPrefetchOutcome:
    """The production defect's outcome shape for a not-found LoRA: a plain retryable failure, no rejection."""
    return AuxPrefetchOutcome(
        kind=AuxModelKind.LORA,
        name=name,
        ok=False,
        retryable=True,
        detail="download failed",
        requesting_job_ids=list(job_ids),
    )


def _lora_plain_failure(name: str, retryable: bool, *job_ids: GenerationID) -> AuxPrefetchOutcome:
    """A genuine (non-rejection) LoRA download failure with an explicit retryability."""
    return AuxPrefetchOutcome(
        kind=AuxModelKind.LORA,
        name=name,
        ok=False,
        retryable=retryable,
        detail="download failed",
        requesting_job_ids=list(job_ids),
    )


def _lora_entries_for(sender: _SenderSpy, job_id: GenerationID | None, name: str) -> list[AuxPrefetchEntry]:
    """Every LoRA prefetch entry naming ``name`` that was requested on behalf of ``job_id``."""
    found: list[AuxPrefetchEntry] = []
    for entries, _pins in sender.calls:
        for entry in entries:
            if entry.kind is AuxModelKind.LORA and entry.name == name and entry.requesting_job_id == job_id:
                found.append(entry)
    return found


async def test_repeated_laundered_lora_failure_faults_at_most_one_job() -> None:
    """The laundered not-found shape arriving for one LoRA across many jobs must not fault more than one.

    A not-found reference is a terminal condition, so the doomed verdict must be reached once and remembered:
    the same laundered outcome landing for a fresh job each time must not fault every one of them. Absent
    memoization, each job is faulted (the first requeued, then escalation makes the rest terminal), so a single
    bad reference silences work across many jobs.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(2)
    coordinator, _sender, _state, _clock = _make(tracker, clock=clock)

    jobs = [_job(loras=[_lora("ghost")]) for _ in range(4)]
    for job in jobs:
        await track_popped_job_async(tracker, job)
        coordinator.on_job_popped(job)
        coordinator.on_prefetch_result(_result(_lora_laundered_failure("ghost", job.id_)))

    faulted = [job for job in jobs if tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT]
    assert len(faulted) <= 1


async def test_terminally_rejected_lora_dispatches_and_is_memoized_across_jobs() -> None:
    """A LoRA terminally rejected for one job dispatches it without the file and is not re-requested later.

    A surfaced rejection is recorded as skipped: the first job dispatches without the LoRA and a second job
    referencing it issues no fresh prefetch request and is not faulted on its account. The coordinator's
    rejection-skip path is kind-agnostic, so a LoRA rejection behaves like a TI one.
    """
    tracker = JobTracker()
    coordinator, sender, _state, _clock = _make(tracker)

    job_a = _job(loras=[_lora("ghost")])
    await track_popped_job_async(tracker, job_a)
    coordinator.on_job_popped(job_a)
    coordinator.on_prefetch_result(_result(_lora_rejection("ghost", "invalid", job_a.id_)))

    assert tracker.are_job_aux_models_prepared(job_a) is True
    assert tracker.get_stage(job_a.id_) == JobStage.PENDING_INFERENCE

    job_b = _job(loras=[_lora("ghost")])
    await track_popped_job_async(tracker, job_b)
    coordinator.on_job_popped(job_b)

    assert _lora_entries_for(sender, job_b.id_, "ghost") == []
    assert tracker.get_stage(job_b.id_) != JobStage.PENDING_SUBMIT


@pytest.mark.parametrize("reason", ["too_large", "nsfw", "invalid", "mismatch"])
async def test_lora_rejection_skips_dispatches_and_memoizes(reason: str) -> None:
    """Any terminal LoRA rejection skips the file, dispatches the job, and memoizes the verdict (control).

    Whatever concrete reason a LoRA is refused for, the job dispatches without it and the skip is remembered,
    so a later job referencing the same file need not re-request it.
    """
    tracker = JobTracker()
    coordinator, _sender, _state, _clock = _make(tracker)

    job = _job(loras=[_lora("refused")])
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(_result(_lora_rejection("refused", reason, job.id_)))

    assert tracker.are_job_aux_models_prepared(job) is True
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    assert tracker.is_lora_skipped(_lora("refused")) is True


async def test_single_transient_lora_failure_requeues_and_arms_backoff_once() -> None:
    """A single genuine transient LoRA failure requeues the job and arms the LoRA backoff exactly once (control).

    A retryable failure with attempts remaining leaves the job pending inference (the requeue) and registers a
    single backoff strike, so a transient outage is retried rather than skipped and does not over-count the
    backoff.
    """
    tracker = JobTracker()
    tracker.set_retry_policy(3)
    coordinator, _sender, state, clock = _make(tracker)

    job = _job(loras=[_lora("net")])
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(_result(_lora_plain_failure("net", True, job.id_)))

    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    assert state.lora_download_backoff.strikes == 1
    assert tracker.is_lora_skipped(_lora("net")) is False


async def test_terminal_aux_prefetch_fault_excluded_from_consecutive_failure_pause() -> None:
    """A terminally faulted aux-prefetch job carries the aux-prefetch origin, excluded from the failure pause.

    A fault the worker never ran a generation for must not count toward the consecutive-failure pop pause.
    Guards the origin stamping the fault-storm reproduction relies on end to end.
    """
    tracker = JobTracker()
    tracker.set_retry_policy(1)
    coordinator, _sender, _state, _clock = _make(tracker)

    job = _job(loras=[_lora("net")])
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(_result(_lora_plain_failure("net", False, job.id_)))

    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT
    assert tracker.was_faulted_by_non_generation_action(job.id_) is True


async def test_mixed_job_with_laundered_lora_should_dispatch_with_only_valid() -> None:
    """A job wanting a valid, a too-large, and a not-found LoRA dispatches with only the valid one.

    Contract (post-fix shape): the root cause surfaces a not-found as a terminal rejection, exactly like a
    too-large one, so all three of the mixed job's LoRAs resolve to a terminal verdict in one result: the valid
    one caches, and both the too-large and the not-found ones are skipped, leaving the job prepared to dispatch
    without them. The earlier laundered shape (a retryable not-found failure) is exercised separately by
    ``test_repeated_laundered_lora_failure_bounds_damage_to_one_fault``, which asserts bounded damage rather
    than immediate dispatch (a single retryable failure requeues the job, it does not prepare it).
    """
    tracker = JobTracker()
    tracker.set_retry_policy(2)
    coordinator, _sender, _state, _clock = _make(tracker)

    job = _job(loras=[_lora("ghost404"), _lora("good"), _lora("chonk")])
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(
        _result(
            AuxPrefetchOutcome(kind=AuxModelKind.LORA, name="good", ok=True, requesting_job_ids=[job.id_]),
            _lora_rejection("chonk", "too_large", job.id_),
            _lora_rejection("ghost404", "not_found", job.id_),
        ),
    )

    assert tracker.is_lora_cached(_lora("good")) is True
    assert tracker.is_lora_skipped(_lora("ghost404")) is True
    assert tracker.are_job_aux_models_prepared(job) is True


async def test_repeated_laundered_lora_failure_bounds_damage_to_one_fault() -> None:
    """The laundered (retryable) not-found shape bounds its damage: one terminal fault, then later jobs dispatch.

    Contract: the coordinator is the defense in depth for a not-found the root cause still laundered into a
    plain retryable failure. First-failure semantics are preserved: the first laundered failure requeues its
    job (it stays pending, one backoff strike). A second laundered failure for the same reference, arriving
    while the incident is active, is terminal: it faults that job and memoizes the reference as skipped. A
    subsequent job referencing the now-memoized reference then dispatches without it rather than faulting, so a
    single doomed reference costs at most one terminal fault per incident.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, _sender, _state, _clock = _make(tracker, clock=clock)

    job_a = _job(loras=[_lora("ghost")])
    await track_popped_job_async(tracker, job_a)
    coordinator.on_job_popped(job_a)
    coordinator.on_prefetch_result(_result(_lora_laundered_failure("ghost", job_a.id_)))
    assert tracker.get_stage(job_a.id_) == JobStage.PENDING_INFERENCE
    assert tracker.is_lora_skipped(_lora("ghost")) is False

    coordinator.on_prefetch_result(_result(_lora_laundered_failure("ghost", job_a.id_)))
    assert tracker.get_stage(job_a.id_) == JobStage.PENDING_SUBMIT
    assert tracker.is_lora_skipped(_lora("ghost")) is True

    job_b = _job(loras=[_lora("ghost")])
    await track_popped_job_async(tracker, job_b)
    coordinator.on_job_popped(job_b)
    coordinator.on_prefetch_result(_result(_lora_laundered_failure("ghost", job_b.id_)))

    assert tracker.are_job_aux_models_prepared(job_b) is True
    assert tracker.get_stage(job_b.id_) != JobStage.PENDING_SUBMIT


async def test_incident_scoped_skip_lapses_and_reference_is_retried() -> None:
    """A skip memoized from a terminal plain failure lapses with the incident, re-enabling the reference.

    A surfaced rejection is a permanent property of the file, but a terminal plain download failure is an
    incident verdict. Once the incident decay window has passed, a fresh job referencing the same file must
    request it again (the download may now succeed) rather than silently dispatch without it forever.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, sender, _state, _clock = _make(tracker, clock=clock)

    job_a = _job(loras=[_lora("ghost")])
    await track_popped_job_async(tracker, job_a)
    coordinator.on_job_popped(job_a)
    coordinator.on_prefetch_result(_result(_lora_laundered_failure("ghost", job_a.id_)))
    coordinator.on_prefetch_result(_result(_lora_laundered_failure("ghost", job_a.id_)))
    assert tracker.is_lora_skipped(_lora("ghost")) is True

    clock.now += STRIKE_DECAY_SECONDS + 1.0

    assert tracker.is_lora_skipped(_lora("ghost")) is False
    job_b = _job(loras=[_lora("ghost")])
    await track_popped_job_async(tracker, job_b)
    coordinator.on_job_popped(job_b)
    assert _lora_entries_for(sender, job_b.id_, "ghost") != []


async def test_rejection_skip_outlives_the_incident_decay_window() -> None:
    """A surfaced rejection's skip is permanent: it does not lapse with the incident decay window (control).

    The time scoping applies only to plain-failure verdicts; a rejected file (invalid, too large, not found)
    can never become usable, so its skip must survive any amount of elapsed time.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    coordinator, _sender, _state, _clock = _make(tracker, clock=clock)

    job = _job(loras=[_lora("refused")])
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(_result(_lora_rejection("refused", "invalid", job.id_)))
    assert tracker.is_lora_skipped(_lora("refused")) is True

    clock.now += STRIKE_DECAY_SECONDS * 10

    assert tracker.is_lora_skipped(_lora("refused")) is True


async def test_shared_rejection_prepares_all_concurrent_jobs() -> None:
    """One terminal rejection naming several in-flight jobs prepares them all to dispatch without the file.

    The downloader reports one deduplicated outcome per file, so a single rejection can name every job waiting
    on it. Each such job dispatches without the LoRA: the kind-agnostic skip iterates every requesting job id.
    """
    tracker = JobTracker()
    coordinator, _sender, _state, _clock = _make(tracker)

    jobs = [_job(loras=[_lora("shared")]) for _ in range(3)]
    for job in jobs:
        await track_popped_job_async(tracker, job)
        coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(_result(_lora_rejection("shared", "invalid", *[job.id_ for job in jobs])))

    for job in jobs:
        assert tracker.are_job_aux_models_prepared(job) is True
        assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE


async def test_lora_backoff_withholds_next_pop_but_not_queued_jobs() -> None:
    """A LoRA backoff strike withholds LoRA support on the next pop without disturbing already-queued jobs.

    The backoff gates only the ``allow_lora`` capability of future pop requests: a strike flips the next pop's
    LoRA advertising off, yet a LoRA job already queued and prepared is neither dropped nor faulted (the
    withholding cannot reach in-flight work).
    """
    state = WorkerState()
    tracker = JobTracker()
    bridge_data = make_mock_bridge_data()
    popper = _make_popper(state, tracker, bridge_data=bridge_data)

    queued = _job(loras=[_lora("kept")])
    await track_popped_job_async(tracker, queued)
    tracker.mark_aux_prefetched("kept", is_version=False, is_ti=False)
    assert tracker.mark_job_aux_prepared_if_ready(queued.id_) is True

    assert popper._effective_allow_lora(bridge_data) is True

    state.lora_download_backoff.register_timeout(time.time())

    assert popper._effective_allow_lora(bridge_data) is False
    assert tracker.get_stage(queued.id_) == JobStage.PENDING_INFERENCE
    assert tracker.are_job_aux_models_prepared(queued) is True


def test_lora_backoff_strike_decay_boundary_resets_incident() -> None:
    """A LoRA backoff strike after the quiet decay period starts a fresh incident from the base window.

    Consecutive strikes escalate, but a strike arriving more than the decay window after the last one is a new
    incident, not a continuation, so the escalation is no longer in force at that point and the strike count
    resets to one. Documents the injectable-clock decay contract the coordinator relies on.
    """
    backoff = AuxDownloadBackoff()
    t0 = 1_000.0

    backoff.register_timeout(t0)
    assert backoff.strikes == 1
    backoff.register_timeout(t0 + 10.0)
    assert backoff.strikes == 2
    assert backoff.is_escalation_active(t0 + 10.0) is True

    later = t0 + 10.0 + STRIKE_DECAY_SECONDS + 1.0
    assert backoff.is_escalation_active(later) is False
    backoff.register_timeout(later)
    assert backoff.strikes == 1


async def test_second_same_lora_failure_is_terminal_while_escalation_active() -> None:
    """While a LoRA incident is active a second failure for the same reference is classified terminal.

    Once the backoff escalation is in force, requeuing into the same failing download path is futile, so a
    fresh failure faults its job terminally rather than requeuing it. Documents the escalation-active
    classification the laundered-failure storm reproduction builds on.
    """
    tracker = JobTracker()
    tracker.set_retry_policy(5)
    coordinator, _sender, state, clock = _make(tracker)

    job_a = _job(loras=[_lora("net")])
    await track_popped_job_async(tracker, job_a)
    coordinator.on_job_popped(job_a)
    coordinator.on_prefetch_result(_result(_lora_plain_failure("net", True, job_a.id_)))
    assert state.lora_download_backoff.is_escalation_active(clock.now) is True

    job_b = _job(loras=[_lora("net")])
    await track_popped_job_async(tracker, job_b)
    coordinator.on_job_popped(job_b)
    coordinator.on_prefetch_result(_result(_lora_plain_failure("net", True, job_b.id_)))

    assert tracker.get_stage(job_b.id_) == JobStage.PENDING_SUBMIT


@pytest.mark.parametrize("kind", [AuxModelKind.LORA, AuxModelKind.TI])
async def test_terminal_rejection_is_kind_agnostic(kind: AuxModelKind) -> None:
    """A terminal rejection dispatches its job without the file for both LoRA and TI kinds.

    The skip-and-memoize contract must not depend on the auxiliary kind: a rejected LoRA and a rejected TI
    both let their job dispatch without the file, so not-found handling belongs on the shared kind-agnostic
    path.
    """
    tracker = JobTracker()
    coordinator, _sender, _state, _clock = _make(tracker)

    job = _job(loras=[_lora("refused")]) if kind is AuxModelKind.LORA else _job(tis=[TIPayloadEntry(name="refused")])
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)

    outcome = AuxPrefetchOutcome(
        kind=kind,
        name="refused",
        ok=False,
        retryable=False,
        rejection_reason="rejected" if kind is AuxModelKind.TI else "invalid",
        requesting_job_ids=[job.id_],
    )
    coordinator.on_prefetch_result(_result(outcome))

    assert tracker.are_job_aux_models_prepared(job) is True
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
