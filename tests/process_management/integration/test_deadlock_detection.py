"""Tests for deadlock detection in MessageDispatcher."""

from __future__ import annotations

import queue
import time
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.action_ledger import ActionLedger
from horde_worker_regen.process_management.ipc.message_dispatcher import (
    _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS,
    MessageDispatcher,
)
from horde_worker_regen.process_management.ipc.messages import HordeProcessMemoryMessage, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_mock_job,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
    track_popped_job_async,
)


def _make_message_dispatcher(
    *,
    state: WorkerState | None = None,
    process_map: ProcessMap | None = None,
    horde_model_map: HordeModelMap | None = None,
    job_tracker: JobTracker | None = None,
    bridge_data: Mock | None = None,
    process_message_queue: queue.Queue[object] | None = None,
) -> MessageDispatcher:
    """Build a MessageDispatcher with mostly-mocked dependencies."""
    if state is None:
        state = WorkerState()
    if process_map is None:
        process_map = ProcessMap({})
    if horde_model_map is None:
        horde_model_map = HordeModelMap(root={})
    if job_tracker is None:
        job_tracker = JobTracker()
    if bridge_data is None:
        bridge_data = make_mock_bridge_data()

    return MessageDispatcher(
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        process_message_queue=process_message_queue or Mock(spec=queue.Queue),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(),
        action_ledger=ActionLedger(),
        reserve_ledger=CommittedReserveLedger(),
        on_unload_vram=Mock(),
        state=state,
    )


@contextmanager
def _capture_levels() -> Iterator[list[tuple[str, str]]]:
    """Capture ``(level name, message)`` pairs for loguru records emitted inside the block."""
    records: list[tuple[str, str]] = []
    handler_id = logger.add(
        lambda m: records.append((m.record["level"].name, m.record["message"])),
        level="TRACE",
    )
    try:
        yield records
    finally:
        logger.remove(handler_id)


def _debug_messages(records: list[tuple[str, str]]) -> list[str]:
    """Messages of the captured records emitted at DEBUG or above."""
    return [message for level, message in records if level != "TRACE"]


async def _make_deadlocked_dispatcher() -> tuple[MessageDispatcher, ProcessMap]:
    """Build a dispatcher whose raw deadlock conditions are both true: one pending job and no busy slot."""
    state = WorkerState(last_job_pop_time=time.time() - 60)
    process_map = ProcessMap({})
    job_tracker = JobTracker()
    await track_popped_job_async(job_tracker, make_mock_job(model="stable_diffusion"))

    return (
        _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker),
        process_map,
    )


async def _start_tracked_inference(message_dispatcher: MessageDispatcher, process_map: ProcessMap) -> None:
    """Move a slot into inference the way dispatch does: the tracker records the start and the slot reports it.

    A queue deadlock clears only on dispatch progress (a job entering inference or completing), never on a slot
    state alone, because the recovery ladder's own remedies change slot states without moving any job.
    """
    job = message_dispatcher._job_tracker.jobs_pending_inference[0]
    await message_dispatcher._job_tracker.mark_inference_started(job)
    process_map[0] = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)


def _age_deadlock_conditions(message_dispatcher: MessageDispatcher) -> None:
    """Backdate the raw-condition clocks so the next tick sees the conditions as sustained."""
    aged = time.time() - (MessageDispatcher._DEADLOCK_PRINT_SUSTAIN_SECONDS + 1)
    message_dispatcher._deadlock_condition_since = aged
    message_dispatcher._queue_deadlock_condition_since = aged


class TestDetectDeadlock:
    """Tests for detect_deadlock."""

    def test_no_jobs_no_deadlock(self) -> None:
        """Deadlocks should never be considered to exist if there are no jobs."""
        message_dispatcher = _make_message_dispatcher()
        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_deadlock is False
        assert message_dispatcher._in_queue_deadlock is False

    async def test_recent_pop_skips_detection(self) -> None:
        """Deadlocks should not be detected if a job was just popped."""
        state = WorkerState(last_job_pop_time=time.time())
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())

        message_dispatcher = _make_message_dispatcher(state=state, job_tracker=job_tracker)
        message_dispatcher._in_deadlock = True
        message_dispatcher._in_queue_deadlock = True
        message_dispatcher._queue_deadlock_model = "stable_diffusion"
        message_dispatcher._queue_deadlock_process_id = 0

        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_deadlock is False
        assert message_dispatcher._in_queue_deadlock is False
        assert message_dispatcher._queue_deadlock_model is None

    async def test_detects_queue_deadlock_when_all_waiting_with_matching_model(self) -> None:
        """When all processes are waiting and one has the needed model, it's a queue deadlock."""
        state = WorkerState(last_job_pop_time=time.time() - 20)
        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_info.loaded_horde_model_name = "stable_diffusion"
        process_map = ProcessMap({0: process_info})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_queue_deadlock is True
        assert message_dispatcher._queue_deadlock_model == "stable_diffusion"
        assert message_dispatcher._queue_deadlock_process_id == 0

    async def test_detects_queue_deadlock_no_model_match_uses_first_job(self) -> None:
        """When all processes are waiting but none has the needed model, still a queue deadlock."""
        state = WorkerState(last_job_pop_time=time.time() - 20)
        process_info = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_info.loaded_horde_model_name = None
        process_map = ProcessMap({0: process_info})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job(model="some_model"))

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_queue_deadlock is True
        assert message_dispatcher._queue_deadlock_model == "some_model"

    async def test_detects_general_deadlock_no_processes(self) -> None:
        """General deadlock: jobs exist but no processes are busy and none waiting."""
        state = WorkerState(last_job_pop_time=time.time() - 20)
        process_map = ProcessMap({})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_deadlock is True

    async def test_deadlock_clears_when_processes_become_busy(self) -> None:
        """If a process starts working, the deadlock should clear."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        process_info = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({0: process_info})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        message_dispatcher._in_deadlock = True
        message_dispatcher._last_deadlock_detected_time = time.time() - 8

        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_deadlock is False

    async def test_queue_deadlock_persists_after_timeout(self) -> None:
        """Queue deadlock should remain active after timeout so the supervisor can recover it."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        process_info = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job(model="stable_diffusion"))

        message_dispatcher = _make_message_dispatcher(
            state=state,
            process_map=process_map,
            job_tracker=job_tracker,
        )
        message_dispatcher._in_queue_deadlock = True
        message_dispatcher._last_queue_deadlock_detected_time = time.time() - 35
        message_dispatcher._queue_deadlock_model = "stable_diffusion"
        message_dispatcher._queue_deadlock_process_id = 0

        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_queue_deadlock is True
        assert message_dispatcher._queue_deadlock_model == "stable_diffusion"

    def test_queue_deadlock_waits_if_processes_starting(self) -> None:
        """Queue deadlock should wait if processes are starting."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        process_info = make_mock_process_info(0, state=HordeProcessState.PROCESS_STARTING)
        process_info.last_process_state = HordeProcessState.PROCESS_STARTING
        process_map = ProcessMap({0: process_info})

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map)
        message_dispatcher._in_queue_deadlock = True
        message_dispatcher._last_queue_deadlock_detected_time = time.time() - 35

        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_queue_deadlock is True

    async def test_deadlock_persists_after_10_seconds(self) -> None:
        """Deadlock should remain active after timeout so the supervisor can recover it."""
        state = WorkerState(last_job_pop_time=time.time() - 60)
        process_map = ProcessMap({})
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())

        message_dispatcher = _make_message_dispatcher(
            state=state,
            process_map=process_map,
            job_tracker=job_tracker,
        )
        message_dispatcher._in_deadlock = True
        message_dispatcher._last_deadlock_detected_time = time.time() - 12

        message_dispatcher.detect_deadlock()
        assert message_dispatcher._in_deadlock is True

    async def test_sustained_queue_deadlock_throttles_verbose_dump(self) -> None:
        """A sustained wedge must not dump the full deadlock diagnostics on every control-loop tick.

        The verbose ``_print_deadlock_info`` dump (process map, model map, per-stage counts) is useful
        once, but the recurring "still detected" branches re-emitted it every ~0.5s tick for the whole
        duration of a wedge, flooding the log with thousands of identical lines. It must be throttled to
        at most once per detail-log interval regardless of how many ticks run.
        """
        state = WorkerState(last_job_pop_time=time.time() - 60)
        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job(model="stable_diffusion"))

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        # Pre-arm both detectors into their *recurring* (already-detected, past-timeout) branches so each
        # tick takes the spammy "still detected" path rather than a one-off initial detection.
        message_dispatcher._in_queue_deadlock = True
        message_dispatcher._last_queue_deadlock_detected_time = time.time() - 35
        message_dispatcher._queue_deadlock_model = "stable_diffusion"
        message_dispatcher._queue_deadlock_process_id = 0
        message_dispatcher._in_deadlock = True
        message_dispatcher._last_deadlock_detected_time = time.time() - 12

        dump_calls = 0
        original_dump = message_dispatcher._print_deadlock_info

        def _counting_dump() -> None:
            nonlocal dump_calls
            dump_calls += 1
            original_dump()

        message_dispatcher._print_deadlock_info = _counting_dump  # type: ignore[method-assign]

        for _ in range(10):
            message_dispatcher.detect_deadlock()

        # Ten ticks of a continuous wedge must collapse to a single verbose dump, not ten (or twenty).
        assert dump_calls == 1
        assert message_dispatcher._in_queue_deadlock is True

    async def test_sub_threshold_condition_prints_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deadlock condition shorter than the sustain threshold must log nothing at DEBUG.

        Every gap between jobs where pending work waits on a deliberate dispatch decision satisfies the raw
        condition for a few seconds, so printing on its rising edge reports a healthy worker as wedged and
        buries the log in verbose dumps.
        """
        message_dispatcher, _ = await _make_deadlocked_dispatcher()

        with _capture_levels() as records:
            for _ in range(5):
                message_dispatcher.detect_deadlock()

        assert _debug_messages(records) == []

        # Load-bearing check: with the sustain threshold removed, the very same tick reports the condition,
        # which is what the printer did before it was gated.
        monkeypatch.setattr(MessageDispatcher, "_DEADLOCK_PRINT_SUSTAIN_SECONDS", 0.0)
        with _capture_levels() as ungated_records:
            message_dispatcher.detect_deadlock()

        assert any(message.startswith("Deadlock detected.") for message in _debug_messages(ungated_records))

    async def test_sustained_condition_prints_summary_once(self) -> None:
        """Once the condition outlives the threshold, one compact summary is logged, not one per tick."""
        message_dispatcher, _ = await _make_deadlocked_dispatcher()
        message_dispatcher.detect_deadlock()
        _age_deadlock_conditions(message_dispatcher)

        with _capture_levels() as records:
            for _ in range(5):
                message_dispatcher.detect_deadlock()

        summaries = [message for message in _debug_messages(records) if message.startswith("Deadlock detected.")]
        assert len(summaries) == 1
        assert "pending=1" in summaries[0]
        assert "in_progress=0" in summaries[0]
        assert "slots=" in summaries[0]

        queue_summaries = [
            message
            for message in _debug_messages(records)
            if message.startswith("Queue deadlock detected without a model causing it.")
        ]
        assert len(queue_summaries) == 1

        # The verbose dump belongs to the investigation trail, not the operator-facing log.
        assert not any(message.startswith("process_map:") for message in _debug_messages(records))
        assert any(level == "TRACE" and message.startswith("process_map:") for level, message in records)

    async def test_cleared_line_requires_a_printed_detected_line(self) -> None:
        """The "cleared" lines are emitted only for an episode whose detection was actually reported."""
        message_dispatcher, process_map = await _make_deadlocked_dispatcher()
        message_dispatcher.detect_deadlock()

        with _capture_levels() as records:
            await _start_tracked_inference(message_dispatcher, process_map)
            message_dispatcher.detect_deadlock()

        assert message_dispatcher._in_deadlock is False
        assert _debug_messages(records) == []

        sustained_dispatcher, sustained_process_map = await _make_deadlocked_dispatcher()
        sustained_dispatcher.detect_deadlock()
        _age_deadlock_conditions(sustained_dispatcher)
        sustained_dispatcher.detect_deadlock()

        with _capture_levels() as sustained_records:
            await _start_tracked_inference(sustained_dispatcher, sustained_process_map)
            sustained_dispatcher.detect_deadlock()

        assert sustained_dispatcher._in_deadlock is False
        assert "Deadlock cleared." in _debug_messages(sustained_records)
        assert "Queue deadlock cleared." in _debug_messages(sustained_records)

    async def test_detection_state_matches_for_sub_threshold_and_sustained(self) -> None:
        """Print gating must not move any detection state the recovery supervisor reads."""
        blip_dispatcher, _ = await _make_deadlocked_dispatcher()
        blip_dispatcher.detect_deadlock()
        blip_snapshot = blip_dispatcher.get_deadlock_snapshot()

        sustained_dispatcher, _ = await _make_deadlocked_dispatcher()
        sustained_dispatcher.detect_deadlock()
        _age_deadlock_conditions(sustained_dispatcher)
        first_detected_at = sustained_dispatcher._last_deadlock_detected_time
        first_queue_detected_at = sustained_dispatcher._last_queue_deadlock_detected_time
        sustained_dispatcher.detect_deadlock()
        sustained_snapshot = sustained_dispatcher.get_deadlock_snapshot()

        assert blip_snapshot.in_deadlock is True
        assert blip_snapshot.in_queue_deadlock is True
        assert sustained_snapshot.in_deadlock is True
        assert sustained_snapshot.in_queue_deadlock is True
        assert sustained_snapshot.queue_deadlock_model == blip_snapshot.queue_deadlock_model
        assert sustained_snapshot.queue_deadlock_process_id == blip_snapshot.queue_deadlock_process_id

        # Printing the summary must not restart the onset clocks the wedge classifier measures against.
        assert sustained_snapshot.deadlock_started_at == first_detected_at
        assert sustained_snapshot.queue_deadlock_started_at == first_queue_detected_at

        for snapshot in (blip_snapshot, sustained_snapshot):
            assert snapshot.indicates_structural_wedge(now=snapshot.queue_deadlock_started_at + 1) is False
            assert (
                snapshot.indicates_structural_wedge(
                    now=snapshot.queue_deadlock_started_at + _MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS,
                )
                is True
            )

    async def test_a_starting_non_inference_child_does_not_restart_the_wedge_clock(self) -> None:
        """A safety/post-process respawn cannot serve pending inference work, so it is not progress.

        The wedge clock the recovery escalation measures against is the queue-deadlock onset stamp. A remedy
        that cycles a non-inference child puts that child into ``PROCESS_STARTING`` on its own cadence, and a
        detector that reads any starting child as a boot worth waiting for restarts the onset stamp every
        time such a remedy runs. The wedge then never reads as sustained however long it stands.
        """
        state = WorkerState(last_job_pop_time=time.time() - 60)
        safety = make_mock_process_info(
            0,
            model_name=None,
            state=HordeProcessState.PROCESS_STARTING,
            process_type=HordeProcessType.SAFETY,
        )
        inference = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: safety, 1: inference})
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job(model="head_model"))

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        detected_at = time.time() - 35
        message_dispatcher._in_queue_deadlock = True
        message_dispatcher._last_queue_deadlock_detected_time = detected_at
        message_dispatcher._queue_deadlock_model = "head_model"

        # The condition this test is about must actually be present: a child really is booting, and it is not
        # one that could take the pending job.
        assert process_map.num_starting_processes() == 1
        assert process_map.has_inference_in_progress() is False

        message_dispatcher.detect_deadlock()

        snapshot = message_dispatcher.get_deadlock_snapshot()
        assert snapshot.in_queue_deadlock is True
        assert snapshot.queue_deadlock_started_at == detected_at
        assert snapshot.indicates_structural_wedge() is True

    async def test_a_starting_inference_slot_is_still_worth_waiting_for(self) -> None:
        """A booting inference slot can take the pending job, so the detector keeps waiting on it.

        This is the anti-flap case the wedge clock exists to tolerate: a slot replacing itself is capacity
        arriving for exactly the work that is waiting, and reporting a sustained wedge over it would drive
        recovery against a worker that is about to serve.
        """
        state = WorkerState(last_job_pop_time=time.time() - 60)
        booting = make_mock_process_info(1, model_name=None, state=HordeProcessState.PROCESS_STARTING)
        process_map = ProcessMap({1: booting})
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job(model="head_model"))

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        detected_at = time.time() - 35
        message_dispatcher._in_queue_deadlock = True
        message_dispatcher._last_queue_deadlock_detected_time = detected_at
        message_dispatcher._queue_deadlock_model = "head_model"

        message_dispatcher.detect_deadlock()

        snapshot = message_dispatcher.get_deadlock_snapshot()
        assert snapshot.in_queue_deadlock is True
        assert snapshot.queue_deadlock_started_at > detected_at
        assert snapshot.indicates_structural_wedge() is False

    async def test_the_named_model_is_the_blocked_head_not_a_later_queue_entry(self) -> None:
        """Attribution names the job the queue is actually stopped on, which is its head.

        The head is what every consumer of this attribution goes on to investigate. Naming whichever later
        queue entry happens to match a resident model points an operator at a job that is merely waiting its
        turn behind the block.
        """
        state = WorkerState(last_job_pop_time=time.time() - 20)
        resident = make_mock_process_info(0, model_name="resident_model", state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: resident})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job(model="head_model"))
        await track_popped_job_async(job_tracker, make_mock_job(model="resident_model"))

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        message_dispatcher.detect_deadlock()

        snapshot = message_dispatcher.get_deadlock_snapshot()
        assert snapshot.in_queue_deadlock is True
        assert snapshot.queue_deadlock_model == "head_model"

    async def test_a_resident_head_still_names_the_slot_holding_its_model(self) -> None:
        """When the head's own model is resident, the slot holding it is still attributed.

        The process attribution is what distinguishes "nothing loaded can serve the head" from "the model is
        right there and the slot is idle anyway", so restricting attribution to the head must not cost it.
        """
        state = WorkerState(last_job_pop_time=time.time() - 20)
        resident = make_mock_process_info(0, model_name="head_model", state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: resident})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job(model="head_model"))
        await track_popped_job_async(job_tracker, make_mock_job(model="other_model"))

        message_dispatcher = _make_message_dispatcher(state=state, process_map=process_map, job_tracker=job_tracker)
        message_dispatcher.detect_deadlock()

        snapshot = message_dispatcher.get_deadlock_snapshot()
        assert snapshot.queue_deadlock_model == "head_model"
        assert snapshot.queue_deadlock_process_id == 0

    async def test_memory_report_does_not_clear_deadlock_signal(self) -> None:
        """Passive child messages should not mask an active deadlock episode."""
        process_info = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: process_info})
        process_message_queue: queue.Queue[object] = queue.Queue()
        process_message_queue.put(
            HordeProcessMemoryMessage(
                process_id=0,
                process_launch_identifier=0,
                info="memory",
                ram_usage_bytes=1024,
            ),
        )

        message_dispatcher = _make_message_dispatcher(
            process_map=process_map,
            process_message_queue=process_message_queue,
        )
        message_dispatcher._in_deadlock = True
        message_dispatcher._in_queue_deadlock = True

        await message_dispatcher.receive_and_handle_process_messages()

        assert message_dispatcher._in_deadlock is True
        assert message_dispatcher._in_queue_deadlock is True
