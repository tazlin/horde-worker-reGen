"""The room a safety check is given on its card: the booking, the wait, and the CPU after a wait too long.

An on-GPU safety process needs more device memory for the length of an evaluation than it holds at rest. These
rows pin the parent's side of that: a check's growth is booked before it is sent, a check that does not fit
waits with its booking standing, and one that has waited as long as a CPU check takes is sent there.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.resources.vram_arbiter import VramRequest, VramRequestKind
from horde_worker_regen.process_management.scheduling.ledgers.safety_check_window import (
    SAFETY_CHECK_CPU_SECONDS_SEED,
    SafetyCheckWindow,
    SafetyCheckWindowLedger,
)
from horde_worker_regen.process_management.scheduling.ledgers.safety_placement import SAFETY_GPU_LOAD_CHARGE_MB
from horde_worker_regen.process_management.scheduling.workload_flow import SAFETY_CHECK_FLOW
from tests.process_management.conftest import make_mock_process_info
from tests.process_management.scheduling.test_safety_placement import _placement_harness, _PlacementHarness

_SAFETY_PROCESS_ID = 9
_RESERVED_AT_REST_MB = 924


class _ScriptedArbiter:
    """An arbiter with a live cycle whose verdict a row sets, recording what it was asked."""

    has_cycle = True

    def __init__(self) -> None:
        self.admits = True
        self.requests: list[VramRequest] = []

    def evaluate(self, request: VramRequest) -> Mock:
        self.requests.append(request)
        return Mock(admits=self.admits)


def _window_harness(monkeypatch: pytest.MonkeyPatch) -> tuple[_PlacementHarness, _ScriptedArbiter]:
    """A scheduler with an idle on-GPU safety process reporting its at-rest reservation, and a scripted arbiter."""
    harness = _placement_harness(monkeypatch)
    safety = make_mock_process_info(
        _SAFETY_PROCESS_ID,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    safety.process_reserved_mb = _RESERVED_AT_REST_MB
    harness.scheduler._process_map[_SAFETY_PROCESS_ID] = safety
    harness.scheduler._state.avg_safety_cpu_seconds = 0.0
    arbiter = _ScriptedArbiter()
    harness.scheduler._vram_arbiter = arbiter  # type: ignore[assignment]
    return harness, arbiter


def _booked_mb(harness: _PlacementHarness, job_id: str) -> float:
    """What the reserve ledger holds for ``job_id``'s check."""
    scheduler = harness.scheduler
    return scheduler._reserve_ledger.planned_charge_for_unit(
        SAFETY_CHECK_FLOW,
        job_id,
        scheduler._process_map.reserved_by_pid(None),
    )


class TestSafetyCheckWindow:
    """What the scheduler answers a safety check that asks for room on its card."""

    def test_a_check_that_fits_is_booked_and_opened(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The growth priced is the peak footprint less what the process already holds, and it is booked."""
        harness, arbiter = _window_harness(monkeypatch)
        scheduler = harness.scheduler
        expected_growth_mb = SAFETY_GPU_LOAD_CHARGE_MB - (
            scheduler.resolved_context_constant_mb() + _RESERVED_AT_REST_MB
        )

        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.OPEN

        request = arbiter.requests[0]
        assert request.kind is VramRequestKind.SAFETY_CHECK
        assert request.candidate_delta_mb == pytest.approx(expected_growth_mb)
        assert _booked_mb(harness, "job-a") == pytest.approx(expected_growth_mb)

    def test_a_check_that_does_not_fit_waits_with_its_room_booked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The booking is made once and stands across passes, and the request nets it out of its price."""
        harness, arbiter = _window_harness(monkeypatch)
        arbiter.admits = False
        scheduler = harness.scheduler

        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.WAIT
        booked_mb = _booked_mb(harness, "job-a")
        assert booked_mb > 0
        harness.clock.advance(1.0)
        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.WAIT

        assert _booked_mb(harness, "job-a") == pytest.approx(booked_mb)
        assert arbiter.requests[0].own_planned_unmaterialized_mb == 0.0
        assert arbiter.requests[1].own_planned_unmaterialized_mb == pytest.approx(booked_mb)

    def test_a_waiting_check_opens_once_its_room_appears(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A check that waited and then fits runs on the device with its booking kept for the run."""
        harness, arbiter = _window_harness(monkeypatch)
        arbiter.admits = False
        scheduler = harness.scheduler
        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.WAIT

        harness.clock.advance(1.0)
        arbiter.admits = True

        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.OPEN
        assert _booked_mb(harness, "job-a") > 0
        ledger = scheduler.safety_check_window_ledger
        assert (ledger.opened_after_wait, ledger.sent_to_cpu) == (1, 0)
        assert ledger.waited_seconds == pytest.approx(1.0)

    def test_a_check_that_waits_as_long_as_a_cpu_check_takes_goes_to_the_cpu(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Past the seeded cost of a CPU check the wait ends, the booking is dropped and the CPU takes it."""
        harness, arbiter = _window_harness(monkeypatch)
        arbiter.admits = False
        scheduler = harness.scheduler
        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.WAIT

        harness.clock.advance(SAFETY_CHECK_CPU_SECONDS_SEED - 0.5)
        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.WAIT
        harness.clock.advance(0.5)

        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.CPU
        assert _booked_mb(harness, "job-a") == 0.0
        assert scheduler.safety_check_window_ledger.sent_to_cpu == 1

    def test_the_wait_follows_the_measured_cost_of_a_cpu_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Once a CPU check has been measured on this worker, that figure bounds the wait."""
        harness, arbiter = _window_harness(monkeypatch)
        arbiter.admits = False
        scheduler = harness.scheduler
        scheduler._state.avg_safety_cpu_seconds = 8.0
        assert scheduler.safety_check_patience_seconds() == 8.0
        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.WAIT

        harness.clock.advance(SAFETY_CHECK_CPU_SECONDS_SEED + 1.0)
        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.WAIT
        harness.clock.advance(8.0)

        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.CPU

    def test_safety_off_the_card_is_not_priced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With safety paused off the card the check goes at once and nothing is booked or asked."""
        harness, arbiter = _window_harness(monkeypatch)
        harness.lifecycle.is_safety_gpu_paused = True

        assert harness.scheduler.safety_check_window("job-a") is SafetyCheckWindow.OPEN

        assert arbiter.requests == []
        assert _booked_mb(harness, "job-a") == 0.0

    def test_bookings_drop_when_safety_leaves_the_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A booking cannot outlive the process it was priced on, so it never stands against that process's start."""
        harness, arbiter = _window_harness(monkeypatch)
        arbiter.admits = False
        scheduler = harness.scheduler
        assert scheduler.safety_check_window("job-a") is SafetyCheckWindow.WAIT
        assert _booked_mb(harness, "job-a") > 0

        harness.lifecycle.is_safety_gpu_paused = True
        scheduler._outstanding_planned_vram_mb(None)

        assert _booked_mb(harness, "job-a") == 0.0
        assert scheduler.safety_check_window_ledger.waiting_since == {}


class TestSafetyCheckWindowLedger:
    """The wait records, stated directly."""

    def test_a_job_that_never_waited_resolves_to_nothing(self) -> None:
        """Resolving an unknown job changes no counter."""
        ledger = SafetyCheckWindowLedger(lambda: 100.0)

        assert ledger.resolve("job-a", on_cpu=False) is None
        assert (ledger.opened_after_wait, ledger.sent_to_cpu, ledger.waited_seconds) == (0, 0, 0.0)

    def test_prune_forgets_jobs_that_left_the_queue(self) -> None:
        """A wait for a job no longer queued is dropped without counting as either outcome."""
        ledger = SafetyCheckWindowLedger(lambda: 100.0)
        ledger.waited("gone")
        ledger.waited("kept")

        ledger.prune(["kept"])

        assert set(ledger.waiting_since) == {"kept"}
        assert (ledger.opened_after_wait, ledger.sent_to_cpu) == (0, 0)
