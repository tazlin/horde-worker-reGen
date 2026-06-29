"""Capacity- and accounting-invariant tests for ``ProcessMap``.

This is the systematic follow-up to the idle-slot ownership audit: several ``ProcessMap`` helpers answer
"how much capacity is there / is the worker stuck / how much VRAM is free", and each keys off a
*list of process states*. A helper whose state list is looser than its name promises is the same class
of latent bug as the original wedge, so these lock the intended semantics down:

* ``num_available_inference_processes`` must mean "can actually accept a job": a dead, ending,
  failed, or just-unloaded slot is not capacity.
* ``num_inference_processes`` counts inference slots only (safety/download are not inference capacity).
* ``all_waiting_for_job`` participates in queue-deadlock detection and so must reflect the safety
  process's state too, not just inference slots.
* the VRAM accounting helpers must ignore a torn-down slot whose sample has been zeroed, so a dead
  slot's stale figure cannot corrupt the budget.
"""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import make_mock_process_info


class TestNumAvailableInferenceProcesses:
    """Available must equal can accept a job, not merely not actively busy."""

    def test_idle_and_preloaded_and_complete_slots_are_available(self) -> None:
        """The three between-jobs states a job can actually be dispatched into all count as available."""
        process_map = ProcessMap(
            {
                1: make_mock_process_info(1, state=HordeProcessState.WAITING_FOR_JOB),
                2: make_mock_process_info(2, state=HordeProcessState.PRELOADED_MODEL),
                3: make_mock_process_info(3, state=HordeProcessState.INFERENCE_COMPLETE),
            },
        )

        assert process_map.num_available_inference_processes() == 3

    def test_actively_busy_slots_are_not_available(self) -> None:
        """A slot mid-inference or mid-preload is not free to take a new job."""
        process_map = ProcessMap(
            {
                1: make_mock_process_info(1, state=HordeProcessState.INFERENCE_STARTING),
                2: make_mock_process_info(2, state=HordeProcessState.PRELOADING_MODEL),
                3: make_mock_process_info(3, state=HordeProcessState.INFERENCE_POST_PROCESSING),
            },
        )

        assert process_map.num_available_inference_processes() == 0

    def test_dead_or_dying_or_failed_slots_are_not_available(self) -> None:
        """A torn-down or failed slot is not capacity, even though it is not 'busy' either.

        These states sit in the gap between ``is_process_busy()`` (false for them) and
        ``can_accept_job()`` (also false): counting them as available over-reports capacity to anything
        that asks "is a slot free", which is exactly the kind of phantom-capacity reasoning that wedged
        the worker before. None of them can take a job until they are replaced or recover.
        """
        process_map = ProcessMap(
            {
                1: make_mock_process_info(1, state=HordeProcessState.PROCESS_ENDED),
                2: make_mock_process_info(2, state=HordeProcessState.PROCESS_ENDING),
                3: make_mock_process_info(3, state=HordeProcessState.INFERENCE_FAILED),
                4: make_mock_process_info(4, state=HordeProcessState.UNLOADED_MODEL_FROM_RAM),
            },
        )

        assert process_map.num_available_inference_processes() == 0

    def test_safety_processes_are_not_counted_as_inference_capacity(self) -> None:
        """An idle safety process is not inference capacity, no matter how idle it looks."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    0,
                    state=HordeProcessState.WAITING_FOR_JOB,
                    process_type=HordeProcessType.SAFETY,
                ),
                1: make_mock_process_info(1, state=HordeProcessState.WAITING_FOR_JOB),
            },
        )

        assert process_map.num_available_inference_processes() == 1


class TestInferenceProcessCounting:
    """Counting helpers must partition by process type as their names promise."""

    def test_num_inference_processes_excludes_safety(self) -> None:
        """``num_inference_processes`` is inference slots only; safety is a different pool."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(0, process_type=HordeProcessType.SAFETY),
                1: make_mock_process_info(1),
                2: make_mock_process_info(2),
            },
        )

        assert process_map.num_inference_processes() == 2

    def test_num_loaded_inference_processes_excludes_ended(self) -> None:
        """A slot reported ended is no longer a live inference process."""
        process_map = ProcessMap(
            {
                1: make_mock_process_info(1, state=HordeProcessState.WAITING_FOR_JOB),
                2: make_mock_process_info(2, state=HordeProcessState.PROCESS_ENDED),
                3: make_mock_process_info(3, state=HordeProcessState.PROCESS_ENDING),
            },
        )

        assert process_map.num_loaded_inference_processes() == 1


class TestAllWaitingForJobReflectsSafety:
    """``all_waiting_for_job`` feeds queue-deadlock detection, so the safety pool's state matters too."""

    def test_idle_inference_and_safety_reads_as_all_waiting(self) -> None:
        """A fully idle worker (inference idle/preloaded, safety idle) reads as all-waiting."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    0,
                    state=HordeProcessState.WAITING_FOR_JOB,
                    process_type=HordeProcessType.SAFETY,
                ),
                1: make_mock_process_info(1, state=HordeProcessState.WAITING_FOR_JOB),
                2: make_mock_process_info(2, state=HordeProcessState.PRELOADED_MODEL),
            },
        )

        assert process_map.all_waiting_for_job() is True

    def test_a_busy_safety_process_means_not_all_waiting(self) -> None:
        """A safety check in flight is real work, so the worker is not idle even if inference is.

        This is why the queue-deadlock detector does not fire while the safety pool is busy: a job is
        still draining through the tail. The behavior is deliberate, so lock it.
        """
        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    0,
                    state=HordeProcessState.EVALUATING_SAFETY,
                    process_type=HordeProcessType.SAFETY,
                ),
                1: make_mock_process_info(1, state=HordeProcessState.WAITING_FOR_JOB),
            },
        )

        assert process_map.all_waiting_for_job() is False


class TestVramAccountingIgnoresTornDownSlots:
    """A slot torn down via ``on_process_ending`` has its VRAM sample zeroed and must drop out."""

    def test_free_and_total_vram_exclude_a_zeroed_dead_slot(self) -> None:
        """A dead slot's zeroed sample must not skew the conservative free-VRAM or device-total figures."""
        live = make_mock_process_info(1, state=HordeProcessState.WAITING_FOR_JOB)
        live.total_vram_mb = 16000
        live.vram_usage_mb = 4000  # 12000 MB free on the live slot
        dead = make_mock_process_info(2, state=HordeProcessState.WAITING_FOR_JOB)
        dead.total_vram_mb = 16000
        dead.vram_usage_mb = 15000  # would drag the conservative min down to 1000 MB if counted
        process_map = ProcessMap({1: live, 2: dead})

        process_map.on_process_ending(2)  # zeroes the dead slot's VRAM sample

        assert process_map.get_free_vram_mb() == 12000.0
        assert process_map.get_reported_total_vram_mb() == 16000.0

    def test_vram_helpers_return_none_before_any_report(self) -> None:
        """Before any slot has reported real VRAM (cold start / CPU-only), the figures are unknown, not 0."""
        process_map = ProcessMap({1: make_mock_process_info(1, state=HordeProcessState.PROCESS_STARTING)})
        process_map[1].total_vram_mb = 0

        assert process_map.get_free_vram_mb() is None
        assert process_map.get_reported_total_vram_mb() is None
