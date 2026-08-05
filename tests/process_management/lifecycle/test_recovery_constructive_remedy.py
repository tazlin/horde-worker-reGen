"""Tests for the constructive reclaim rungs the save-our-ship escalation reaches before it faults work.

A wedge whose cause is a saturated card is not addressed by rebuilding the same processes against the same
card, and faulting servable jobs addresses it least of all. The escalation therefore issues the ordered reclaim
rungs first. These tests pin both halves of that: the rungs actually get issued (and no accepted work is
dropped while one remains), and the budgets that bound them keep the give-up backstop and the terminal
escalation reachable.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

from horde_worker_regen.process_management.jobs.job_tracker import JobFaultOrigin, JobTracker
from horde_worker_regen.process_management.lifecycle.recovery_supervisor import RecoveryAction, RecoverySupervisor
from horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator import (
    RecoveryDisposition,
    WorkerRecoveryCoordinator,
)
from horde_worker_regen.process_management.resources.reclaim_ladder import (
    CacheReleaseTarget,
    IdleResidentModel,
    LadderCandidates,
    LaneReclaimCandidate,
    ReclaimRungKind,
)
from tests.process_management.conftest import (
    FakeClock,
    make_job_pop_response,
    make_test_recovery_coordinator,
    mark_job_in_progress_async,
    queue_job_for_safety_async,
    track_popped_job_async,
)

_IDLE_RESIDENT_FOOTPRINT_MB = 6000.0
"""A resident worth several GB, so the cheapest rung is plainly worth issuing before dropping jobs."""

_TICK_SECONDS = 1.0
"""Control-loop spacing the tests drive, comfortably finer than every grace under test."""

_ESCALATION_TICK_BUDGET = 600
"""Ticks a test drives before declaring an escalation non-terminating. Far above every timeline here."""


def _job_info_for(job: object) -> Mock:
    """Wrap a popped job in the job-info shape the safety queue expects."""
    job_info = Mock()
    job_info.sdk_api_job_info = job
    job_info.job_image_results = []
    return job_info


def _idle_resident_candidates(*, process_id: int = 2) -> LadderCandidates:
    """Candidates offering one multi-GB idle resident model, the cheapest rung on the ladder."""
    return LadderCandidates(
        device_index=None,
        idle_residents=(
            IdleResidentModel(
                process_id=process_id,
                tenant_label="idle_resident_model",
                materialized_monotonic=100.0,
                footprint_mb=_IDLE_RESIDENT_FOOTPRINT_MB,
            ),
        ),
    )


def _two_rung_candidates() -> LadderCandidates:
    """Candidates offering an idle resident and an idle allocator cache: two rungs, then exhaustion."""
    base = _idle_resident_candidates()
    return LadderCandidates(
        device_index=None,
        idle_residents=base.idle_residents,
        cache_targets=(
            CacheReleaseTarget(
                process_id=3,
                tenant_label="inference#3",
                materialized_monotonic=90.0,
                reclaimable_mb=800.0,
            ),
        ),
    )


_LANE_CANDIDATES = (
    LaneReclaimCandidate(
        kind=ReclaimRungKind.PAUSE_PP_LANE,
        tenant_label="post-processing lane",
        promised_mb=1200.0,
    ),
    LaneReclaimCandidate(
        kind=ReclaimRungKind.PAUSE_VAE_LANE,
        tenant_label="VAE lane",
        promised_mb=900.0,
    ),
)
"""The two service-lane pauses, in the fixed order the ladder escalates through them."""


def _many_rung_candidates() -> LadderCandidates:
    """Candidates offering rungs of every constructive kind the ladder sequences."""
    base = _two_rung_candidates()
    return LadderCandidates(
        device_index=None,
        idle_residents=base.idle_residents,
        cache_targets=base.cache_targets,
        lanes=_LANE_CANDIDATES,
        safety=LaneReclaimCandidate(
            kind=ReclaimRungKind.SAFETY_OFF_GPU,
            tenant_label="safety",
            promised_mb=300.0,
        ),
    )


def _lane_only_candidates() -> LadderCandidates:
    """Candidates offering only the two lane pauses, so both fit inside one rung allotment."""
    return LadderCandidates(device_index=None, lanes=_LANE_CANDIDATES)


def _arm_reclaim_actuator(coordinator: WorkerRecoveryCoordinator, rungs_issued: list[str]) -> None:
    """Make every actuator call on the mocked scheduler report that it acted, recording the rung kind."""
    scheduler = coordinator._inference_scheduler

    def _record(kind: ReclaimRungKind) -> bool:
        rungs_issued.append(kind.value)
        return True

    scheduler.unload_idle_model.side_effect = lambda *_a, **_k: _record(ReclaimRungKind.UNLOAD_IDLE_MODEL)
    scheduler.release_idle_cache.side_effect = lambda *_a, **_k: _record(ReclaimRungKind.RELEASE_IDLE_CACHE)
    scheduler.pause_post_process_lane.side_effect = lambda *_a, **_k: _record(ReclaimRungKind.PAUSE_PP_LANE)
    scheduler.pause_vae_lane.side_effect = lambda *_a, **_k: _record(ReclaimRungKind.PAUSE_VAE_LANE)
    scheduler.pause_component_lane.side_effect = lambda *_a, **_k: _record(ReclaimRungKind.PAUSE_COMPONENT_LANE)
    scheduler.safety_off_gpu.side_effect = lambda *_a, **_k: _record(ReclaimRungKind.SAFETY_OFF_GPU)
    scheduler.restore_post_process_lane.return_value = True
    scheduler.restore_vae_lane.return_value = True
    scheduler.restore_component_lane.return_value = True


def _spy_on_scheduling_recovery_faults(coordinator: WorkerRecoveryCoordinator, transcript: list[str]) -> Mock:
    """Replace the tracker's fault entry point with a spy that appends a marker for recovery-origin faults."""
    spy = Mock(
        side_effect=lambda *_args, **kwargs: (
            transcript.append("fault") if kwargs.get("fault_origin") is JobFaultOrigin.SCHEDULING_RECOVERY else None
        ),
    )
    coordinator._job_tracker.handle_job_fault_now = spy  # type: ignore[method-assign]
    return spy


async def _make_safety_starved_coordinator(
    *,
    candidates: LadderCandidates,
    queued_safety_jobs: int = 5,
) -> tuple[WorkerRecoveryCoordinator, FakeClock, list[str]]:
    """Build the field wedge shape: a safety pool that cannot come up behind a queue of generated jobs.

    The inference pool is healthy and idle, the safety pool is failing with no live process, and generated
    results are queued for a safety check that will never run. This is not a structural queue deadlock, so the
    pre-existing post-processing reclaim guard is inapplicable and any rung observed is the constructive remedy
    cursor's.

    Returns:
        The coordinator, its clock, and the shared transcript that records rung kinds and recovery faults in
        the order they happen.
    """
    clock = FakeClock()
    job_tracker = JobTracker(clock=clock)
    coordinator = make_test_recovery_coordinator(job_tracker=job_tracker, clock=clock, structural_wedge=False)
    coordinator._process_lifecycle.safety_pool_failing = True
    coordinator._inference_scheduler.build_reclaim_ladder_candidates.return_value = candidates

    for _ in range(queued_safety_jobs):
        popped = await track_popped_job_async(job_tracker, make_job_pop_response())
        await queue_job_for_safety_async(job_tracker, _job_info_for(popped))

    transcript: list[str] = []
    _arm_reclaim_actuator(coordinator, transcript)
    return coordinator, clock, transcript


class TestSafetyStarvedWedgeReachesAConstructiveRung:
    """The wedge a saturated card produces: safety cannot obtain its allocation, so it never comes up."""

    async def test_a_rung_is_issued_and_no_job_is_faulted(self) -> None:
        """A multi-GB idle resident is unloaded, and the queued safety backlog is left intact.

        Rebuilding the pools puts the same safety process against the same full card, so the escalation used to
        run straight from "rebuild identically" to "fault the backlog". The resident's weights are the thing
        actually holding the memory safety needs, so unloading them is the cheapest action that can change the
        outcome, and no accepted work is dropped while it is available.
        """
        coordinator, clock, transcript = await _make_safety_starved_coordinator(
            candidates=_idle_resident_candidates(),
        )
        fault_spy = _spy_on_scheduling_recovery_faults(coordinator, transcript)

        for _ in range(10):
            coordinator.run_recovery_supervisor()
            clock.advance(_TICK_SECONDS)

        assert transcript == [ReclaimRungKind.UNLOAD_IDLE_MODEL.value]
        assert fault_spy.call_count == 0
        assert len(coordinator._job_tracker.jobs_pending_safety_check) == 5

    async def test_every_rung_precedes_the_first_recovery_fault(self) -> None:
        """Under this wedge shape the whole rung sequence is issued before any job is faulted.

        Specific to the safety-starved wedge on purpose: the pre-existing post-processing reclaim guard needs
        both inference capacity and a structural queue wedge, and this wedge has no queue deadlock at all, so
        the guard cannot account for the ordering observed here.
        """
        coordinator, clock, transcript = await _make_safety_starved_coordinator(
            candidates=_many_rung_candidates(),
        )
        assert coordinator.structural_queue_wedge_active() is False
        _spy_on_scheduling_recovery_faults(coordinator, transcript)

        for _ in range(_ESCALATION_TICK_BUDGET):
            coordinator.run_recovery_supervisor()
            clock.advance(_TICK_SECONDS)
            if "fault" in transcript:
                break

        assert "fault" in transcript, "the give-up backstop must still be reachable"
        assert transcript[: transcript.index("fault")] == [
            ReclaimRungKind.UNLOAD_IDLE_MODEL.value,
            ReclaimRungKind.RELEASE_IDLE_CACHE.value,
            ReclaimRungKind.PAUSE_PP_LANE.value,
            ReclaimRungKind.PAUSE_VAE_LANE.value,
            ReclaimRungKind.SAFETY_OFF_GPU.value,
        ]

    async def test_the_episode_close_restores_the_lanes_its_rungs_paused(self) -> None:
        """Lanes a recovery rung stopped are restarted by the recovery episode itself, in reverse order.

        No external backstop is relied on: the ones that exist require a card debounced healthy for a sustained
        window, which a chronically pressured card never reaches, so a lane paused here would otherwise stay
        stopped for the rest of the session.
        """
        coordinator, clock, transcript = await _make_safety_starved_coordinator(
            candidates=_lane_only_candidates(),
            queued_safety_jobs=0,
        )
        _spy_on_scheduling_recovery_faults(coordinator, transcript)

        for _ in range(_ESCALATION_TICK_BUDGET):
            coordinator.run_recovery_supervisor()
            clock.advance(_TICK_SECONDS)
            if ReclaimRungKind.PAUSE_VAE_LANE.value in transcript:
                break

        assert [rung.kind for rung in coordinator.reclaim_paused_lanes] == [
            ReclaimRungKind.PAUSE_PP_LANE,
            ReclaimRungKind.PAUSE_VAE_LANE,
        ]

        # The condition clears: safety comes back, accepted work starts moving, and the episode closes on its
        # clean streak.
        coordinator._process_lifecycle.safety_pool_failing = False
        started = await track_popped_job_async(coordinator._job_tracker, make_job_pop_response())
        await mark_job_in_progress_async(coordinator._job_tracker, started)
        for _ in range(60):
            coordinator.run_recovery_supervisor()
            clock.advance(_TICK_SECONDS)

        assert coordinator.recovery_supervisor.is_in_episode is False
        assert coordinator.reclaim_paused_lanes == []
        scheduler = coordinator._inference_scheduler
        assert scheduler.restore_vae_lane.called
        assert scheduler.restore_post_process_lane.called
        assert coordinator.reclaim_rungs is None
        assert coordinator.reclaim_cursor == 0


class TestRemedyBudgetsBoundTheEscalation:
    """The counted allotment and the frozen candidate list are what keep the rest of the ladder reachable."""

    async def test_the_rung_cursor_only_advances_and_the_candidate_list_is_frozen(self) -> None:
        """The cursor is monotonic across many ticks and the candidates are snapshotted exactly once.

        Snapshotting matters because the candidate builder reads live state: a card whose idle tenants come and
        go would regenerate rungs indefinitely, and an escalation reading a never-shrinking candidate list would
        never reach its pool rebuild or its give-up.
        """
        coordinator, clock, transcript = await _make_safety_starved_coordinator(
            candidates=_many_rung_candidates(),
        )
        # A card that keeps offering fresh candidates on every look.
        coordinator._inference_scheduler.build_reclaim_ladder_candidates.side_effect = lambda *_a, **_k: (
            _many_rung_candidates()
        )
        _spy_on_scheduling_recovery_faults(coordinator, transcript)

        frozen_ladder = None
        observed_cursor = 0
        for _ in range(_ESCALATION_TICK_BUDGET):
            coordinator.run_recovery_supervisor()
            clock.advance(_TICK_SECONDS)
            if coordinator.reclaim_rungs is None:
                continue
            if frozen_ladder is None:
                frozen_ladder = coordinator.reclaim_rungs
            assert coordinator.reclaim_rungs is frozen_ladder
            assert coordinator.reclaim_cursor >= observed_cursor
            observed_cursor = coordinator.reclaim_cursor
            if "fault" in transcript:
                break

        assert frozen_ladder is not None
        assert coordinator._inference_scheduler.build_reclaim_ladder_candidates.call_count == 1
        assert observed_cursor == len(frozen_ladder)
        assert transcript.count(ReclaimRungKind.SAFETY_OFF_GPU.value) == 1

    async def test_a_wedge_no_rung_fixes_still_reaches_give_up_then_terminal(self) -> None:
        """The control: rungs must not make the safety valve unreachable.

        A condition none of the rungs resolves has to end in the ordinary give-up (the horde reissues the jobs)
        and then in the terminal escalation, exactly as it did before any rung existed.
        """
        coordinator, clock, transcript = await _make_safety_starved_coordinator(
            candidates=_two_rung_candidates(),
        )
        terminal_recovery = Mock(return_value=RecoveryDisposition.RESTART_PROCESS)
        coordinator._terminal_recovery_callback = terminal_recovery
        fault_spy = _spy_on_scheduling_recovery_faults(coordinator, transcript)

        terminal_give_up_seen = False
        for _ in range(_ESCALATION_TICK_BUDGET):
            coordinator.run_recovery_supervisor()
            if coordinator.recovery_supervisor.give_up_is_terminal:
                terminal_give_up_seen = True
                break
            clock.advance(_TICK_SECONDS)

        assert fault_spy.call_count > 0, "the wedged safety backlog must eventually be reissued to the horde"
        assert terminal_give_up_seen
        assert terminal_recovery.call_count > 0

    async def test_refunded_give_ups_are_capped_so_the_terminal_escalation_arrives(self) -> None:
        """A remedy that always looks reachable can defer terminal give-up only within its declared bound.

        Constructive settling uses the refund-count cap; a post-processing unload uses its independent wall-clock
        grace. Either policy must still deliver the terminal safety valve inside the overall tick budget, and the
        count-governed path may never exceed its cap.
        """
        clock = FakeClock()
        job_tracker = JobTracker(clock=clock)
        coordinator = make_test_recovery_coordinator(
            job_tracker=job_tracker,
            clock=clock,
            structural_wedge=True,
            unload_post_process_models_from_vram=True,
        )
        coordinator._inference_scheduler.build_reclaim_ladder_candidates.side_effect = lambda *_a, **_k: (
            _idle_resident_candidates()
        )
        transcript: list[str] = []
        _arm_reclaim_actuator(coordinator, transcript)

        terminal_give_up_seen = False
        for _ in range(_ESCALATION_TICK_BUDGET):
            coordinator.run_recovery_supervisor()
            if coordinator.recovery_supervisor.give_up_is_terminal:
                terminal_give_up_seen = True
                break
            clock.advance(_TICK_SECONDS)

        assert terminal_give_up_seen
        assert coordinator.give_up_yields_spent <= coordinator.MAX_GIVE_UP_YIELDS_PER_EPISODE

    async def test_the_allotment_is_renewed_only_once_the_pool_rebuild_has_happened(self) -> None:
        """Rungs, then the pool rebuild, then the rungs the first allotment did not reach.

        The allotment is what lets the rebuild happen at all while rungs remain; renewing it afterwards is what
        keeps the remaining (cheaper-than-give-up) rungs reachable, and the monotonic cursor is what keeps the
        total finite.
        """
        coordinator, clock, transcript = await _make_safety_starved_coordinator(
            candidates=_many_rung_candidates(),
        )
        _spy_on_scheduling_recovery_faults(coordinator, transcript)
        soft_reset_calls: list[float] = []
        original_soft_reset = coordinator.perform_soft_reset

        def _record_soft_reset() -> None:
            soft_reset_calls.append(clock.now)
            original_soft_reset()

        coordinator.perform_soft_reset = _record_soft_reset  # type: ignore[method-assign]

        rung_times: list[float] = []
        previous_rung_count = 0
        for _ in range(_ESCALATION_TICK_BUDGET):
            coordinator.run_recovery_supervisor()
            if len(transcript) > previous_rung_count:
                previous_rung_count = len(transcript)
                rung_times.append(clock.now)
            clock.advance(_TICK_SECONDS)
            if "fault" in transcript:
                break

        assert len(soft_reset_calls) == 1
        rungs_before_rebuild = [issued_at for issued_at in rung_times if issued_at < soft_reset_calls[0]]
        rungs_after_rebuild = [issued_at for issued_at in rung_times if issued_at > soft_reset_calls[0]]
        assert len(rungs_before_rebuild) == coordinator.RECLAIM_RUNG_ALLOTMENT
        assert rungs_after_rebuild


async def _make_structural_wedge_coordinator(
    *,
    candidates: LadderCandidates,
    head_block_reason: Callable[[], str | None],
    pending_jobs: int = 3,
) -> tuple[WorkerRecoveryCoordinator, FakeClock, list[str]]:
    """Build a structural queue wedge over a healthy, idle inference lane with a named head blocker.

    The pool is fine and the card has room; the queue simply does not move. This is the shape in which a
    resource-freeing remedy has to justify itself, because nothing about it is a resource shortfall unless
    the named blocker says so.

    Returns:
        The coordinator, its clock, and the shared transcript recording rung kinds and recovery faults.
    """
    clock = FakeClock()
    job_tracker = JobTracker(clock=clock)
    coordinator = make_test_recovery_coordinator(
        job_tracker=job_tracker,
        clock=clock,
        structural_wedge=True,
        head_block_reason_provider=head_block_reason,
    )
    coordinator._inference_scheduler.build_reclaim_ladder_candidates.return_value = candidates

    for _ in range(pending_jobs):
        await track_popped_job_async(job_tracker, make_job_pop_response())

    transcript: list[str] = []
    _arm_reclaim_actuator(coordinator, transcript)
    return coordinator, clock, transcript


class TestRemediesAnswerToTheNamedBlocker:
    """A remedy that provably does not move the named blocker stops being a reason to hold off faulting."""

    async def test_rungs_stop_once_they_do_not_change_the_named_blocker(self) -> None:
        """A blocker no reclaim rung touches bounds the ladder rather than consuming all of it.

        Every rung frees card memory, which addresses nothing when the head is held by an admission decision
        rather than by a shortfall. Left unbounded, the ladder issues its whole sequence against that blocker
        while each issued rung renews the "a remedy remains" excuse that holds the give-up backstop off, so
        the accepted work is neither served nor released.
        """
        coordinator, clock, transcript = await _make_structural_wedge_coordinator(
            candidates=_many_rung_candidates(),
            head_block_reason=lambda: "the head is deferred by an admission decision",
        )
        _spy_on_scheduling_recovery_faults(coordinator, transcript)

        for _ in range(_ESCALATION_TICK_BUDGET):
            coordinator.run_recovery_supervisor()
            clock.advance(_TICK_SECONDS)
            if "fault" in transcript:
                break

        assert "fault" in transcript, "the give-up backstop must still be reachable"
        rungs_before_fault = transcript[: transcript.index("fault")]
        assert len(rungs_before_fault) == coordinator.RECLAIM_NO_EFFECT_LIMIT
        assert coordinator.reclaim_ladder_ruled_irrelevant is True

    async def test_a_remedy_that_moves_the_named_blocker_keeps_the_ladder(self) -> None:
        """A rung that changes what blocks the head is working, so the ladder is not curtailed.

        This is the shortfall case the ladder exists for: each rung frees room and the head's blocker moves
        with it. Curtailing the ladder here would stop the escalation short of the rung that would have
        served the queue.
        """
        transcript: list[str] = []
        coordinator, clock, transcript = await _make_structural_wedge_coordinator(
            candidates=_many_rung_candidates(),
            head_block_reason=lambda: f"the card is short of room for the head ({len(transcript)} freed)",
        )
        _spy_on_scheduling_recovery_faults(coordinator, transcript)

        for _ in range(_ESCALATION_TICK_BUDGET):
            coordinator.run_recovery_supervisor()
            clock.advance(_TICK_SECONDS)
            if "fault" in transcript:
                break

        rungs_before_fault = transcript[: transcript.index("fault")] if "fault" in transcript else transcript
        assert len(rungs_before_fault) > coordinator.RECLAIM_NO_EFFECT_LIMIT
        assert coordinator.reclaim_ladder_ruled_irrelevant is False

    async def test_an_unmoved_blocker_stops_excusing_the_give_up(self) -> None:
        """The settling window of a rung ruled irrelevant is no longer a reason to defer faulting.

        The give-up defers while a remedy may still land. A remedy demonstrated not to address the blocker
        cannot land, so continuing to defer to its window is what turns a bounded escalation into an
        indefinite hold over accepted work.
        """
        coordinator, clock, transcript = await _make_structural_wedge_coordinator(
            candidates=_many_rung_candidates(),
            head_block_reason=lambda: "the head is deferred by an admission decision",
        )
        _spy_on_scheduling_recovery_faults(coordinator, transcript)

        for _ in range(_ESCALATION_TICK_BUDGET):
            coordinator.run_recovery_supervisor()
            if coordinator.reclaim_ladder_ruled_irrelevant:
                break
            clock.advance(_TICK_SECONDS)

        assert coordinator.reclaim_ladder_ruled_irrelevant is True
        # Present the hazard the guard exists for: a rung inside its settling window, which is exactly the
        # state that used to return "a remedy remains".
        coordinator.reclaim_rung_issued_at = clock.now
        assert coordinator._reclaim_rung_settling() is True
        assert coordinator._give_up_yields_to_remedy() is False


class TestReclaimIsSequencedAheadOfTheOtherRungs:
    """The policy object's ordering, independent of any process state."""

    def test_a_reachable_remedy_outranks_both_the_soft_reset_and_the_give_up(self) -> None:
        """RECLAIM is returned while a remedy is reachable, and the pool rebuild follows once it is not."""
        clock = FakeClock()
        supervisor = RecoverySupervisor(clock=clock, wedge_grace_seconds=2, reset_interval_seconds=2)

        assert supervisor.evaluate(is_wedged=True, pool_ready=True, constructive_remedy_available=True) is (
            RecoveryAction.NONE
        )
        clock.advance(2)
        assert supervisor.evaluate(is_wedged=True, pool_ready=True, constructive_remedy_available=True) is (
            RecoveryAction.RECLAIM
        )
        assert supervisor.limp_by_level == 0

        clock.advance(2)
        assert supervisor.evaluate(is_wedged=True, pool_ready=True, constructive_remedy_available=False) is (
            RecoveryAction.SOFT_RESET
        )

    def test_an_unwired_caller_escalates_rather_than_stalling(self) -> None:
        """The default is False, so a caller that reports nothing climbs the ladder exactly as before."""
        clock = FakeClock()
        supervisor = RecoverySupervisor(clock=clock, wedge_grace_seconds=2, reset_interval_seconds=2)

        assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.NONE
        clock.advance(2)
        assert supervisor.evaluate(is_wedged=True, pool_ready=True) is RecoveryAction.SOFT_RESET
