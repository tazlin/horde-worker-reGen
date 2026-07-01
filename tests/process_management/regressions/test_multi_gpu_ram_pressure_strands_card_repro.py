"""Reproductions for a multi-GPU host losing a whole card to the RAM-pressure footprint reduction.

When system RAM crosses its danger floor the scheduler reduces the resident inference-process count to
return each idle context's pinned weights to the OS. Expressed against the *worker-wide* pool, that
reduction targets ``max(1, jobs_in_progress)`` total processes and shrinks with ``device_index=None``, so the
victim selection is free to stop every idle process regardless of which card it is pinned to. On a single-GPU
host collapsing to one context is the intended floor; on a multi-GPU host it can empty an entire card of
contexts (every one was idle and so eligible), and nothing re-establishes them, so that card sits idle for
the rest of the run while the surviving card serializes the work.

The contract these tests pin:

* The RAM-pressure reduction is applied per card and leaves at least one resident context on every driven
  card, recording each card it shrank.
* Once the host clears the danger floor (and the self-throttle pop-pause has lapsed), the shed cards are
  grown back to their planned per-card process count, so both GPUs resume serving.
* A card a whole-card residency is deliberately holding down is left to that path's own restore.
* The single-GPU / worker-wide reduction is unchanged.
"""

from __future__ import annotations

import multiprocessing
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.governance import WorkerProcessShedState
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_runtime_config,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_CARD_TARGET_PROCESSES = 2
_HEALTHY_AVAILABLE_RAM_MB = 40000.0
_TOTAL_RAM_MB = 64000.0
_CRITICAL_AVAILABLE_RAM_MB = 500.0


def _two_card_lifecycle(process_map: ProcessMap) -> ProcessLifecycleManager:
    """A real lifecycle manager driving two cards, each planned for two inference processes (four total).

    Mirrors the worker-host topology where one worker identity drives both cards under one queue: the
    worker-wide ceiling is the summed per-card plan, and each process carries the ``device_index`` of the
    card it is pinned to.
    """
    bridge_data = make_mock_bridge_data()
    return ProcessLifecycleManager(
        ctx=multiprocessing.get_context("spawn"),  # type: ignore[arg-type]
        process_map=process_map,
        horde_model_map=Mock(),
        job_tracker=JobTracker(),
        process_message_queue=Mock(),
        card_runtimes=make_test_card_runtimes(
            device_indices=(0, 1),
            target_process_count=_CARD_TARGET_PROCESSES,
            config=bridge_data,
        ),
        disk_lock=Mock(),
        aux_model_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        max_safety_processes=1,
        amd_gpu=False,
        directml=None,
        abort_callback=Mock(),
        state=WorkerState(),
    )


def _stub_spawn_onto_map(lifecycle: ProcessLifecycleManager) -> None:
    """Replace real process spawning with a stub that adds an idle mock context to the map on its card."""

    def _fake_start(pid: int, *, device_index: int = 0) -> HordeProcessInfo:
        info = make_mock_process_info(
            pid,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            device_index=device_index,
        )
        lifecycle._process_map[pid] = info
        lifecycle.num_processes_launched += 1
        return info

    lifecycle._start_inference_process = _fake_start  # type: ignore[method-assign]


def _two_card_scheduler(process_map: ProcessMap) -> InferenceScheduler:
    """A scheduler driving the two-card pool: real lifecycle, and the same per-card runtime plan it routes on."""
    lifecycle = _two_card_lifecycle(process_map)
    scheduler = _make_inference_scheduler(process_map=process_map, max_inference=2 * _CARD_TARGET_PROCESSES)
    scheduler._process_lifecycle = lifecycle
    scheduler._card_runtimes = make_test_card_runtimes(
        device_indices=(0, 1),
        target_process_count=_CARD_TARGET_PROCESSES,
    )
    return scheduler


def _full_two_card_process_map(*, busy_card1_lead: bool) -> ProcessMap:
    """Four inference contexts: two on card 0 and two on card 1; card 1's lead is optionally mid-flight.

    With ``busy_card1_lead`` process 3 (card 1) reads busy, so a reduction that spares only busy processes
    keeps card 1's context and must take any further victims from the idle contexts on either card.
    """
    card1_lead_state = (
        HordeProcessState.INFERENCE_POST_PROCESSING if busy_card1_lead else HordeProcessState.WAITING_FOR_JOB
    )
    procs = {
        1: make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
        2: make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
        3: make_mock_process_info(3, model_name="WAI-ANI-NSFW-PONYXL", state=card1_lead_state, device_index=1),
        4: make_mock_process_info(4, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=1),
    }
    return ProcessMap(procs)


def _set_available_ram(scheduler: InferenceScheduler, monkeypatch: pytest.MonkeyPatch, available_mb: float) -> None:
    """Pin the scheduler's measured system RAM so the danger-floor verdict is deterministic on any host."""
    monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: available_mb)
    monkeypatch.setattr(scheduler, "_measured_total_ram_mb", lambda: _TOTAL_RAM_MB)


class TestRamPressureReductionIsDeviceAware:
    """The RAM-pressure footprint reduction must not empty a card of every inference context."""

    def test_reduction_keeps_one_context_per_driven_card(self) -> None:
        """With card 1 busy and card 0 idle, the reduction keeps a context on each card instead of stranding card 0."""
        process_map = _full_two_card_process_map(busy_card1_lead=True)
        scheduler = _two_card_scheduler(process_map)

        scheduler._reduce_processes_under_ram_pressure()

        card0 = process_map.num_loaded_inference_processes(device_index=0)
        card1 = process_map.num_loaded_inference_processes(device_index=1)
        assert card1 >= 1, "the busy card must keep its in-flight context"
        assert card0 >= 1, "the idle card must not be stripped of every context (the GPU would go idle)"

    def test_reduction_records_each_shed_card(self) -> None:
        """Every card the reduction shrank is recorded so the recovery path knows what to grow back."""
        process_map = _full_two_card_process_map(busy_card1_lead=True)
        scheduler = _two_card_scheduler(process_map)

        scheduler._reduce_processes_under_ram_pressure()

        assert scheduler._ram_pressure_shed_cards == {0, 1}


class TestRamPressureReExpansion:
    """Once the host recovers, the cards shed by the reduction grow back to their planned process count."""

    def test_recovered_ram_restores_shed_card_to_plan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A card reduced to one context grows back to its per-card target after RAM clears the danger floor."""
        process_map = ProcessMap(
            {
                1: make_mock_process_info(1, model_name="m", state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
                3: make_mock_process_info(3, model_name="n", state=HordeProcessState.WAITING_FOR_JOB, device_index=1),
            },
        )
        scheduler = _two_card_scheduler(process_map)
        _stub_spawn_onto_map(scheduler._process_lifecycle)
        scheduler._ram_pressure_shed_cards = {0, 1}
        _set_available_ram(scheduler, monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)

        scheduler._restore_processes_after_ram_pressure()

        assert process_map.num_loaded_inference_processes(device_index=0) == _CARD_TARGET_PROCESSES
        assert process_map.num_loaded_inference_processes(device_index=1) == _CARD_TARGET_PROCESSES
        assert scheduler._ram_pressure_shed_cards == set(), "the episode is cleared once the cards are restored"

    def test_no_restore_while_still_under_pressure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The shed cards are not grown back while the host is still below its danger floor."""
        process_map = ProcessMap(
            {1: make_mock_process_info(1, model_name="m", state=HordeProcessState.WAITING_FOR_JOB, device_index=0)},
        )
        scheduler = _two_card_scheduler(process_map)
        _stub_spawn_onto_map(scheduler._process_lifecycle)
        scheduler._ram_pressure_shed_cards = {0}
        _set_available_ram(scheduler, monkeypatch, _CRITICAL_AVAILABLE_RAM_MB)

        scheduler._restore_processes_after_ram_pressure()

        assert process_map.num_loaded_inference_processes(device_index=0) == 1, "no growth while RAM is critical"
        assert scheduler._ram_pressure_shed_cards == {0}, "the episode is still pending until RAM recovers"

    def test_restore_defers_when_ram_cannot_hold_another_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A shed card is not grown back while measured RAM has no room for another resident working set.

        The host is above the absolute danger floor (so the reduction is not re-engaged) but a live context
        already retains most of RAM, so adding another would re-pressure the shared pool. The card stays
        pending rather than oscillating the pool back over the floor.
        """
        resident = make_mock_process_info(1, model_name="m", state=HordeProcessState.WAITING_FOR_JOB, device_index=0)
        resident.ram_usage_bytes = 22_000 * 1024 * 1024
        process_map = ProcessMap({1: resident})
        scheduler = _two_card_scheduler(process_map)
        _stub_spawn_onto_map(scheduler._process_lifecycle)
        scheduler._ram_pressure_shed_cards = {0}
        # Above the danger floor (so not under pressure) but only ~6 GB free: far less than the 22 GB a
        # second context would retain.
        _set_available_ram(scheduler, monkeypatch, 6000.0)

        scheduler._restore_processes_after_ram_pressure()

        assert process_map.num_loaded_inference_processes(device_index=0) == 1, "no growth without RAM headroom"
        assert scheduler._ram_pressure_shed_cards == {0}, "the card stays pending until RAM can hold another context"

    def test_restore_skips_card_held_by_whole_card_residency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A card a whole-card residency is holding down is left for that path's own restore, not regrown here."""
        process_map = ProcessMap(
            {1: make_mock_process_info(1, model_name="Flux", state=HordeProcessState.WAITING_FOR_JOB, device_index=0)},
        )
        scheduler = _two_card_scheduler(process_map)
        _stub_spawn_onto_map(scheduler._process_lifecycle)
        scheduler._ram_pressure_shed_cards = {0}
        scheduler._residency_state(0).model = "Flux.1-Schnell fp8 (Compact)"
        _set_available_ram(scheduler, monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)

        scheduler._restore_processes_after_ram_pressure()

        assert process_map.num_loaded_inference_processes(device_index=0) == 1, (
            "a residency-held card must not be regrown by the RAM-pressure restore"
        )


class TestSingleGpuRamPressureRecovery:
    """The worker-wide (single-GPU) reduction is incremental and restorable."""

    def test_single_card_reduces_by_one_and_records_worker_shed(self) -> None:
        """A single-GPU pressure tick sheds one context, not the whole idle pool, and records restore state."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
                1: make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
                2: make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
            },
        )
        scheduler = _make_inference_scheduler(process_map=process_map, max_inference=3)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=2)

        scheduler._reduce_processes_under_ram_pressure()

        scheduler._process_lifecycle.scale_inference_processes.assert_called_once()
        _, kwargs = scheduler._process_lifecycle.scale_inference_processes.call_args
        assert scheduler._process_lifecycle.scale_inference_processes.call_args.args[0] == 2
        assert kwargs["device_index"] is None
        assert kwargs["pressure_shortfall_mb"] is not None
        assert scheduler._ram_pressure_shed_cards == set()
        assert scheduler._ram_governor_state.worker_shed is not None
        assert scheduler._ram_governor_state.worker_shed.planned_process_count == 3
        assert scheduler._ram_governor_state.worker_shed.shed_process_count == 1

    def test_recovered_single_card_restores_one_worker_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A recovered single-GPU worker grows one shed process back toward plan when RAM has headroom."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
                1: make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
            },
        )
        scheduler = _make_inference_scheduler(process_map=process_map, max_inference=4)
        scheduler._ram_governor_state.worker_shed = WorkerProcessShedState(
            planned_process_count=4, shed_process_count=2
        )
        _set_available_ram(scheduler, monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)

        def _scale(target_count: int, *, device_index: int | None = None, **_kwargs: object) -> int:
            assert device_index is None
            process_map[target_count] = make_mock_process_info(
                target_count,
                model_name=None,
                state=HordeProcessState.WAITING_FOR_JOB,
                device_index=0,
            )
            return process_map.num_loaded_inference_processes()

        scheduler._process_lifecycle.scale_inference_processes = Mock(side_effect=_scale)

        scheduler._restore_processes_after_ram_pressure()

        assert process_map.num_loaded_inference_processes() == 3
        assert scheduler._ram_governor_state.worker_shed is not None
        assert scheduler._ram_governor_state.worker_shed.shed_process_count == 1

    def test_single_card_restore_defers_without_headroom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A single-GPU shed episode stays pending while RAM cannot hold another context."""
        resident = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0)
        resident.ram_usage_bytes = 22_000 * 1024 * 1024
        process_map = ProcessMap({0: resident})
        scheduler = _make_inference_scheduler(process_map=process_map, max_inference=2)
        scheduler._ram_governor_state.worker_shed = WorkerProcessShedState(
            planned_process_count=2, shed_process_count=1
        )
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=2)
        _set_available_ram(scheduler, monkeypatch, 6000.0)

        scheduler._restore_processes_after_ram_pressure()

        scheduler._process_lifecycle.scale_inference_processes.assert_not_called()
        assert scheduler._ram_governor_state.worker_shed is not None


class TestRamPressureScaleDownVictimSelection:
    """RAM pressure scale-down should use the smallest idle process that clears the shortfall."""

    def test_small_shortfall_chooses_small_sufficient_idle_process(self) -> None:
        """A ~591 MB RAM shortfall should stop a ~1 GB idle slot before a ~9 GB model holder."""
        large = make_mock_process_info(1, model_name="large", state=HordeProcessState.WAITING_FOR_JOB, device_index=0)
        small = make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0)
        large.ram_usage_bytes = 9_000 * 1024 * 1024
        small.ram_usage_bytes = 1_000 * 1024 * 1024
        process_map = ProcessMap({1: large, 2: small})
        lifecycle = _two_card_lifecycle(process_map)

        victim = lifecycle._select_inference_process_to_scale_down(
            disallowed_processes=[],
            pressure_shortfall_mb=591.0,
        )

        assert victim is small

    def test_falls_back_when_no_idle_process_clears_shortfall(self) -> None:
        """If no reported idle RSS can clear the shortfall, scale-down keeps the old first-eligible fallback."""
        first = make_mock_process_info(1, model_name="first", state=HordeProcessState.WAITING_FOR_JOB, device_index=0)
        second = make_mock_process_info(
            2, model_name="second", state=HordeProcessState.WAITING_FOR_JOB, device_index=0
        )
        first.ram_usage_bytes = 1_000 * 1024 * 1024
        second.ram_usage_bytes = 2_000 * 1024 * 1024
        process_map = ProcessMap({1: first, 2: second})
        lifecycle = _two_card_lifecycle(process_map)

        victim = lifecycle._select_inference_process_to_scale_down(
            disallowed_processes=[],
            pressure_shortfall_mb=20_000.0,
        )

        assert victim is first
