"""Backstops judged while another subsystem is acting, rather than one at a time.

Each backstop is already covered on its own. What is not covered is what happens when a second subsystem is
mid-action underneath it: a recovery remedy respawning children while the deadlock detector times a wedge, a
reclaim ladder still holding candidates while the give-up backstop decides whether to release accepted work,
a whole-card residency legitimately holding the pop offer while the intake sentinel counts silence. Those
compositions are where a backstop is most likely to be defeated, because the other subsystem's activity looks
exactly like the progress the backstop is watching for.

Every scenario here is a pair: one arrangement where the backstop must act and one where it must stay quiet,
differing only in the fact that decides it. The assertions read the subsystems' own state (wedge stamps,
remedy verdicts, escalation disclosures) rather than the absence of a symptom, so a backstop that stopped
running fails the firing half instead of passing the quiet one.
"""

from __future__ import annotations

import time

from loguru import logger

from horde_worker_regen.process_management.config.worker_state import PopGate, WorkerState
from horde_worker_regen.process_management.ipc.message_dispatcher import (
    _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS,
    MessageDispatcher,
)
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator import WorkerRecoveryCoordinator
from horde_worker_regen.process_management.process_manager import (
    POP_LIVENESS_ERROR_SECONDS,
    POP_LIVENESS_FROZEN_QUEUE_SECONDS,
    POP_LIVENESS_NON_EXPLAINING_GOVERNORS,
    HordeWorkerProcessManager,
)
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from tests.process_management.conftest import (
    FakeClock,
    make_job_pop_response,
    make_mock_job,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.integration.test_deadlock_detection import _make_message_dispatcher
from tests.process_management.lifecycle.test_recovery_constructive_remedy import (
    _ESCALATION_TICK_BUDGET,
    _TICK_SECONDS,
    _make_structural_wedge_coordinator,
    _many_rung_candidates,
    _spy_on_scheduling_recovery_faults,
)
from tests.process_management.manager.test_pop_liveness_sentinel import (
    _FROZEN_QUEUE_PHRASE,
    _capture,
    _complete_a_pop_attempt,
    _hold_pops_at_no_inference_process,
    _pooled_manager,
)

_UNCHANGING_BLOCKER = "the head is deferred by an admission decision"
"""A named head blocker no amount of freed card memory can move, so every reclaim rung leaves it standing."""


# region deadlock detector versus recovery respawns


async def _wedged_dispatcher() -> tuple[MessageDispatcher, ProcessMap]:
    """Build a dispatcher latched onto a queue deadlock, with its wedge clock already 35s old.

    The detector reads wall time and exposes no clock seam, so the age is applied by back-dating its own
    detection stamp, which is how the detector's own suite ages it.

    Returns:
        The dispatcher, latched and aged, and the process map its next tick reads.
    """
    process_map = ProcessMap({0: make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)})
    job_tracker = JobTracker()
    await track_popped_job_async(job_tracker, make_mock_job(model="stable_diffusion"))
    dispatcher = _make_message_dispatcher(
        state=WorkerState(last_job_pop_time=time.time() - 60),
        process_map=process_map,
        job_tracker=job_tracker,
    )

    dispatcher.detect_deadlock()
    assert dispatcher._in_queue_deadlock is True, "precondition: the detector latched the wedge"
    dispatcher._last_queue_deadlock_detected_time = time.time() - 35.0
    return dispatcher, process_map


def _add_starting_process(process_map: ProcessMap, *, process_type: HordeProcessType) -> None:
    """Put a freshly respawning child of ``process_type`` into the map, as a recovery remedy would."""
    process_id = max(process_map.keys()) + 1
    process_map[process_id] = make_mock_process_info(
        process_id,
        model_name=None,
        state=HordeProcessState.PROCESS_STARTING,
        process_type=process_type,
    )


class TestWedgeClockVersusRecoveryRespawns:
    """A recovery remedy cycling children must not be mistaken for the pending queue getting capacity.

    The detector restarts its wedge clock whenever an inference slot is booting, because a booting slot is
    capacity arriving for the queue it is timing. A recovery episode that cycles the safety pool or a service
    lane produces the same "a child is starting" appearance on its own cadence, and crediting it would hold
    the clock at zero for as long as the remedy keeps running, which is exactly as long as the wedge lasts.
    """

    async def test_a_safety_respawn_does_not_restart_the_wedge_clock(self) -> None:
        """Safety and post-processing children cycling underneath the wedge leave it timing."""
        dispatcher, process_map = await _wedged_dispatcher()
        wedge_started_at = dispatcher._last_queue_deadlock_detected_time
        _add_starting_process(process_map, process_type=HordeProcessType.SAFETY)
        _add_starting_process(process_map, process_type=HordeProcessType.POST_PROCESS)
        assert process_map.num_starting_processes() > 0, "precondition: children really are respawning"
        assert process_map.num_starting_inference_processes() == 0, "precondition: no inference capacity is arriving"

        dispatcher.detect_deadlock()

        assert dispatcher._in_queue_deadlock is True, "the respawn must not clear the latch"
        assert dispatcher._last_queue_deadlock_detected_time == wedge_started_at, (
            "the wedge clock was restarted by children that bring the pending queue no capacity"
        )
        snapshot = dispatcher.get_deadlock_snapshot()
        assert snapshot.indicates_structural_wedge() is True
        assert (time.time() - snapshot.queue_deadlock_started_at) >= _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS

    async def test_an_inference_respawn_does_restart_the_wedge_clock(self) -> None:
        """A booting inference slot is capacity for the pending queue, so the wedge is not yet structural."""
        dispatcher, process_map = await _wedged_dispatcher()
        wedge_started_at = dispatcher._last_queue_deadlock_detected_time
        _add_starting_process(process_map, process_type=HordeProcessType.INFERENCE)
        assert process_map.num_starting_inference_processes() == 1, "precondition: a lane is booting"

        dispatcher.detect_deadlock()

        assert dispatcher._in_queue_deadlock is True
        assert dispatcher._last_queue_deadlock_detected_time > wedge_started_at, (
            "arriving inference capacity must restart the clock the wedge verdict is taken from"
        )
        assert dispatcher.get_deadlock_snapshot().indicates_structural_wedge() is False


# endregion

# region give-up versus an outstanding reclaim ladder


async def _wedge_with_a_ruled_irrelevant_ladder() -> tuple[WorkerRecoveryCoordinator, FakeClock, list[str]]:
    """Drive a structural wedge until its reclaim ladder is measured as not addressing the head's blocker.

    Returns:
        The coordinator, its clock, and the transcript of rung kinds issued so far.
    """
    coordinator, clock, transcript = await _make_structural_wedge_coordinator(
        candidates=_many_rung_candidates(),
        head_block_reason=lambda: _UNCHANGING_BLOCKER,
    )
    for _ in range(_ESCALATION_TICK_BUDGET):
        if coordinator.reclaim_ladder_ruled_irrelevant:
            break
        coordinator.run_recovery_supervisor()
        clock.advance(_TICK_SECONDS)
    assert coordinator.reclaim_ladder_ruled_irrelevant is True, "precondition: the ladder was measured as irrelevant"
    return coordinator, clock, transcript


class TestGiveUpVersusAnOutstandingReclaimLadder:
    """A give-up defers to a remedy that can still land, and to nothing else.

    The ladder is what the give-up yields to, so a ladder that keeps offering rungs keeps accepted work off
    the escalation for as long as it offers them. Once the rungs have been measured against the head's own
    stated blocker and found not to move it, the remaining offers are no longer a remedy about to land: they
    are only a reason never to reach the backstop.
    """

    async def test_a_ladder_still_proving_itself_defers_the_give_up(self) -> None:
        """The quiet half: rungs are being issued and judged, so the give-up waits rather than faulting."""
        coordinator, clock, transcript = await _make_structural_wedge_coordinator(
            candidates=_many_rung_candidates(),
            head_block_reason=lambda: _UNCHANGING_BLOCKER,
        )
        fault_spy = _spy_on_scheduling_recovery_faults(coordinator, transcript)

        while not transcript:
            coordinator.run_recovery_supervisor()
            clock.advance(_TICK_SECONDS)

        assert coordinator.reclaim_ladder_ruled_irrelevant is False, "precondition: nothing has been ruled out yet"
        assert coordinator._reclaim_rung_settling() is True, "precondition: an issued rung is still settling"
        assert coordinator._give_up_yields_to_remedy() is True
        assert fault_spy.call_count == 0, "no accepted work may be dropped while a remedy can still land"

    async def test_a_ladder_ruled_irrelevant_no_longer_defers_the_give_up(self) -> None:
        """The firing half: rungs remain on offer, and the give-up proceeds anyway.

        The distinguishing fact is only the verdict on the rungs already tried. The candidate list still has
        unissued entries and the episode's allotment is not spent, so a give-up that yielded to availability
        rather than to evidence would still be deferring here.
        """
        coordinator, clock, transcript = await _wedge_with_a_ruled_irrelevant_ladder()
        assert coordinator.reclaim_cursor < len(coordinator._frozen_reclaim_ladder()), (
            "precondition: rungs remain on offer, so availability alone would still defer"
        )
        assert coordinator._give_up_yields_to_remedy() is False

        fault_spy = _spy_on_scheduling_recovery_faults(coordinator, transcript)
        for _ in range(_ESCALATION_TICK_BUDGET):
            coordinator.run_recovery_supervisor()
            clock.advance(_TICK_SECONDS)
            if fault_spy.call_count:
                break

        assert fault_spy.call_count > 0, "the give-up backstop must be reachable past a ladder ruled irrelevant"


# endregion

# region the pop sentinel versus a whole-card residency


def _residency_forecast() -> StreamForecast:
    """A forecast for a head whose weights genuinely fill the card, so the residency is legitimate."""
    return StreamForecast(
        weights_mb=11500.0,
        reserve_mb=4646.0,
        base_reserve_mb=3100.0,
        free_now_mb=9129.0,
        free_if_alone_mb=15021.0,
        free_after_model_evict_mb=9605.0,
        total_vram_mb=16375.0,
        per_process_overhead_mb=1354.0,
        marginal_process_overhead_mb=1354.0,
        wants_whole_card=True,
    )


async def _manager_holding_a_residency(model: str = "whole_card_model") -> tuple[HordeWorkerProcessManager, float]:
    """Build a silent worker whose card is held by a live whole-card residency.

    Returns:
        The manager and the stamp of the last pop attempt that reached the horde.
    """
    manager = _pooled_manager(whole_card_residency_cooldown_seconds=0)
    completed_at = await _complete_a_pop_attempt(manager)
    await _hold_pops_at_no_inference_process(manager)

    manager._inference_scheduler._begin_whole_card_residency(
        make_job_pop_response(model),
        _residency_forecast(),
        announce=False,
    )
    manager._update_pop_governors()

    assert manager._inference_scheduler.is_whole_card_residency_active() is True
    assert manager._pop_governor_registry.any_active(ignore=POP_LIVENESS_NON_EXPLAINING_GOVERNORS) is True
    return manager, completed_at


class TestPopSentinelVersusAWholeCardResidency:
    """Silence a residency is responsible for is explained; the same silence after it lets go is not.

    Establishing a residency stops the sibling processes and holds the pop offer for its own model, so the
    intake path genuinely goes quiet for the length of the window. That is the worker doing something
    deliberate, and the residency logs its own boundaries. What the sentinel must not do is treat the closed
    window as though it were still open: once the card has been released and the queue still is not moving,
    nothing accounts for the silence any more.
    """

    async def test_the_sentinel_is_quiet_while_the_residency_holds_the_card(self) -> None:
        """The quiet half: a live residency explains an intake path that has reached nobody."""
        manager, completed_at = await _manager_holding_a_residency()

        warnings, warn_sink = _capture("WARNING")
        errors, error_sink = _capture("ERROR")
        try:
            manager._check_pop_liveness(completed_at + POP_LIVENESS_ERROR_SECONDS + 5.0)
        finally:
            logger.remove(warn_sink)
            logger.remove(error_sink)

        assert warnings == []
        assert errors == []

    async def test_the_sentinel_escalates_once_the_residency_has_let_go(self) -> None:
        """The firing half: the window closed on a worker that is still serving nothing."""
        manager, completed_at = await _manager_holding_a_residency()

        # Nothing in the queue wants the residency's model and its cooldown is configured away, so the real
        # restore path releases the card on its own reading of that state.
        manager._inference_scheduler._restore_siblings_after_whole_card()
        manager._update_pop_governors()

        assert manager._inference_scheduler.is_whole_card_residency_active() is False
        assert manager._pop_governor_registry.any_active(ignore=POP_LIVENESS_NON_EXPLAINING_GOVERNORS) is False
        assert manager._state.last_pop_gate == PopGate.NO_INFERENCE_PROCESS, "the hold itself has not moved"

        errors, sink_id = _capture("ERROR")
        try:
            manager._check_pop_liveness(completed_at + POP_LIVENESS_ERROR_SECONDS + 5.0)
        finally:
            logger.remove(sink_id)

        assert len(errors) == 1, errors
        assert str(PopGate.NO_INFERENCE_PROCESS) in errors[0]


# endregion

# region the frozen-queue escalation versus a pop-claim withhold


async def _manager_with_a_full_queue_under_a_residency() -> tuple[HordeWorkerProcessManager, float]:
    """Build a worker whose local queue is full, held at the queue-full gate, under a live residency.

    Returns:
        The manager and the wall-clock stamp at which the queue-full gate began holding.
    """
    manager = _pooled_manager(whole_card_residency_cooldown_seconds=0)
    await _complete_a_pop_attempt(manager)

    await track_popped_job_async(manager._job_tracker, make_job_pop_response("head_model"))
    while not manager._job_popper._is_queue_full(manager.bridge_data):
        await track_popped_job_async(manager._job_tracker, make_job_pop_response("trailing_model"))
    await manager._job_popper.api_job_pop(urgent=True)
    assert manager._state.last_pop_gate == PopGate.QUEUE_FULL, "the ladder must stop at the full queue"

    manager._inference_scheduler._begin_whole_card_residency(
        make_job_pop_response("whole_card_model"),
        _residency_forecast(),
        announce=False,
    )
    manager._update_pop_governors()
    assert manager._pop_governor_registry.any_active(ignore=POP_LIVENESS_NON_EXPLAINING_GOVERNORS) is True

    held_since = manager._state.last_pop_gate_since
    assert manager._state.last_pop_attempt_completed_at <= held_since, (
        "the silence being judged has to begin no later than the hold it is attributed to"
    )
    return manager, held_since


class TestFrozenQueueVersusAPopClaimWithhold:
    """A residency withholding the pop offer explains no pops; it never explains a queue that stops moving.

    The sentinel's ordinary silence disclosure asks why no pop attempt is being made, and a residency holding
    the offer answers that. The frozen-queue escalation asks a different question: whether the work the worker
    has already accepted is being served. Nothing about a withhold answers that one, which is why the
    escalation is judged ahead of the governor check rather than behind it.
    """

    async def test_a_draining_queue_under_the_withhold_is_quiet(self) -> None:
        """Work is being completed the whole time, so the full queue is capacity management."""
        manager, held_since = await _manager_with_a_full_queue_under_a_residency()

        errors, error_sink = _capture("ERROR")
        warnings, warn_sink = _capture("WARNING")
        try:
            for tick in range(0, int(POP_LIVENESS_FROZEN_QUEUE_SECONDS) * 2, 10):
                await manager._job_tracker.increment_jobs_completed()
                manager._check_pop_liveness(held_since + tick)
        finally:
            logger.remove(warn_sink)
            logger.remove(error_sink)

        assert [line for line in errors if _FROZEN_QUEUE_PHRASE in line] == []
        assert warnings == [], "the residency also accounts for the absence of pop attempts"

    async def test_a_frozen_queue_under_the_withhold_still_escalates(self) -> None:
        """Nothing moves for the whole span, and the withhold does not excuse that."""
        manager, held_since = await _manager_with_a_full_queue_under_a_residency()

        errors, error_sink = _capture("ERROR")
        warnings, warn_sink = _capture("WARNING")
        try:
            for tick in range(0, int(POP_LIVENESS_FROZEN_QUEUE_SECONDS) * 2, 10):
                manager._check_pop_liveness(held_since + tick)
        finally:
            logger.remove(warn_sink)
            logger.remove(error_sink)

        frozen = [line for line in errors if _FROZEN_QUEUE_PHRASE in line]
        assert len(frozen) == 1, errors
        assert "head_model" in frozen[0]
        assert warnings == [], "the residency still accounts for the absence of pop attempts"


# endregion
