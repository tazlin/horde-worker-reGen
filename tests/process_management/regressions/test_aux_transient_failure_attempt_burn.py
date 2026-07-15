"""Reproductions for a transient aux-prefetch failure burning both of a job's inference attempts at once.

A retryable aux-prefetch failure faults the job with retries remaining, the tracker requeues it, and the
coordinator's pop/reconcile path immediately re-requests the same reference. With no cooldown a sub-second
downloader failure is re-requested at once; its instant re-failure arrives while the class incident is still
active (so it is classified terminal) and the two failures consume both of the job's inference attempts inside
a second, terminalizing a merely transient outage with 0.00s generated.

These encode the cooldown contract at the coordinator seam:

- incident repro: a transient failure then an immediate reconcile must not re-enter the failing download path
  (no new request) within the cooldown, so the job cannot be terminalized by an instant second failure;
- after the cooldown the reference is re-requested and a success dispatches the job;
- after the cooldown a second genuine failure (incident still active) still faults terminally;
- a job whose only uncached reference is cooling stays bounded: its deadline still faults it if the
  cooldown-plus-refetch cycle never resolves it;
- a cooling job still holds a live deadline, so it stays counted as aux-held and out of deadlock fuel.
"""

from __future__ import annotations

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry
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
    """A hand-advanceable clock so cooldown and deadline behaviour is deterministic."""

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
    clock: _Clock,
    timeout: float = 120.0,
) -> tuple[AuxPrefetchCoordinator, _SenderSpy]:
    sender = _SenderSpy()
    coordinator = AuxPrefetchCoordinator(
        job_tracker=tracker,
        state=WorkerState(),
        prefetch_sender=sender,
        download_timeout_provider=lambda: timeout,
        pin_sender=lambda _pins: None,
        in_flight_provider=dict,
        clock=clock,
    )
    return coordinator, sender


def _lora(name: str) -> LorasPayloadEntry:
    return LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=False)


def _job(name: str) -> ImageGenerateJobPopResponse:
    return make_job_pop_response("stable_diffusion", loras=[_lora(name)])


def _result(*outcomes: AuxPrefetchOutcome) -> HordeAuxPrefetchResultMessage:
    return HordeAuxPrefetchResultMessage(
        process_id=9000, process_launch_identifier=1, info="r", outcomes=list(outcomes)
    )


def _laundered_failure(name: str, *job_ids: GenerationID) -> AuxPrefetchOutcome:
    """A plain retryable LoRA download failure (the laundered/transient shape), no rejection surfaced."""
    return AuxPrefetchOutcome(
        kind=AuxModelKind.LORA,
        name=name,
        ok=False,
        retryable=True,
        detail="download failed",
        requesting_job_ids=list(job_ids),
    )


def _success(name: str, *job_ids: GenerationID) -> AuxPrefetchOutcome:
    return AuxPrefetchOutcome(kind=AuxModelKind.LORA, name=name, ok=True, requesting_job_ids=list(job_ids))


async def test_transient_failure_then_instant_reconcile_does_not_terminalize_within_cooldown() -> None:
    """The incident repro: a transient failure plus an immediate reconcile must not burn the second attempt.

    A transient failure requeues the job (one attempt spent). An immediate reconcile tick (the production loop
    re-requesting a pending job with no deadline) must not re-enter the failing download path within the
    cooldown: no new request goes out, so an instant second downloader failure cannot arrive to terminalize the
    job. The requeued job stays pending and bounded rather than faulting to the horde with 0.00s generated.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(2)
    coordinator, sender = _make(tracker, clock=clock)

    job = _job("ghost")
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    assert len(sender.calls) == 1

    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job.id_)))
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    requests_after_failure = len(sender.calls)

    # The loop tick immediately after the requeue must not re-request the cooling reference, so no instant
    # second failure is possible: the job stays pending and holds a live bounding deadline.
    coordinator.reconcile_and_refresh_pins()
    assert len(sender.calls) == requests_after_failure
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    assert coordinator.has_live_deadline(job.id_) is True


async def test_after_cooldown_reference_is_re_requested_and_success_dispatches() -> None:
    """Once the cooldown lapses the reference is re-requested, and a success outcome dispatches the job."""
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(2)
    coordinator, sender = _make(tracker, clock=clock)

    job = _job("ghost")
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job.id_)))
    requests_after_failure = len(sender.calls)

    clock.now += _AUX_REFETCH_COOLDOWN_SECONDS + 1.0
    coordinator.reconcile_and_refresh_pins()
    assert len(sender.calls) == requests_after_failure + 1

    coordinator.on_prefetch_result(_result(_success("ghost", job.id_)))
    assert tracker.are_job_aux_models_prepared(job) is True
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE


async def test_after_cooldown_second_genuine_failure_faults_terminally() -> None:
    """A second genuine failure after a real cooldown gap still faults terminally (incident still active).

    The cooldown defers the retry; it does not soften the terminal-on-second-failure semantics. Once the
    cooldown lapses the reference is re-requested, and because the class incident is still in force a fresh
    failure is classified terminal and faults the job.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, sender = _make(tracker, clock=clock)

    job = _job("ghost")
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job.id_)))
    assert tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE

    clock.now += _AUX_REFETCH_COOLDOWN_SECONDS + 1.0
    coordinator.reconcile_and_refresh_pins()
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job.id_)))

    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT


async def test_cooling_job_stays_bounded_and_deadline_still_faults_it() -> None:
    """A job cooling on its only reference is never unbounded: its bounding deadline still faults it.

    With the download timeout shorter than the cooldown, the bounding deadline armed while the reference cools
    expires first. The deadline machinery faults the job exactly as it faults any unresolved prefetch, so a
    reference that never becomes fetchable cannot leave the job pending forever.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(3)
    coordinator, _sender = _make(tracker, clock=clock, timeout=5.0)

    job = _job("ghost")
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job.id_)))

    # Reconcile arms the bounding deadline (5s) while the reference is still cooling (15s).
    coordinator.reconcile_and_refresh_pins()
    assert coordinator.has_live_deadline(job.id_) is True

    clock.now += 6.0  # past the bounding deadline, still within the reference cooldown
    coordinator.scan_deadlines()
    assert tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT


async def test_cooling_job_still_counts_as_aux_held_against_deadlock() -> None:
    """A cooling job holds a live deadline, so it stays in the aux-hold set the deadlock detector consults.

    The deadlock detector treats a pending job with a live prefetch deadline as intentionally holding no lane,
    not as wedge fuel. A cooling job's bounding deadline must keep it in that set so a transient failure does
    not turn a legitimately-waiting job into a spurious deadlock verdict.
    """
    clock = _Clock()
    tracker = JobTracker(clock=clock)
    tracker.set_retry_policy(2)
    coordinator, _sender = _make(tracker, clock=clock)

    job = _job("ghost")
    await track_popped_job_async(tracker, job)
    coordinator.on_job_popped(job)
    coordinator.on_prefetch_result(_result(_laundered_failure("ghost", job.id_)))
    coordinator.reconcile_and_refresh_pins()

    assert job.id_ in coordinator.job_ids_with_live_deadlines()
