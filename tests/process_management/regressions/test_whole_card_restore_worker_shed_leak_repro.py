"""A RAM-pressure shed owns its restore, whatever the whole-card residency path does meanwhile.

Two independent mechanisms move the single-GPU inference-process count. The RAM governor sheds an idle
context when the host crosses its absolute RAM danger floor, recording the shed in ``worker_shed`` so its
own restore path can grow the pool back once RAM proves headroom. Whole-card residency independently
collapses the pool to the residency holder and, when the residency drains, grows the pool back toward the
launched ceiling through the lifecycle.

The contract these tests pin:

* While a RAM shed stands, a whole-card release does not regrow the pool past the shed count. Regrowing it
  onto a host still under its floor would only be shed again on the next governor tick, a cold reload each
  time.
* The RAM governor restores the shed contexts once the host proves headroom, and resolves its record.
* Under sustained RAM pressure with repeated residency reserve/restore cycles, the recorded shed count
  reflects the contexts actually outstanding and does not accumulate one entry per cycle.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_MAX_INFERENCE = 2
_TOTAL_RAM_MB = 64000.0
_CRITICAL_AVAILABLE_RAM_MB = 500.0
_HEALTHY_AVAILABLE_RAM_MB = 60000.0


def _pin_available_ram(scheduler: InferenceScheduler, monkeypatch: pytest.MonkeyPatch, available_mb: float) -> None:
    """Pin measured system RAM so the danger-floor verdict is deterministic on any host."""
    monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: available_mb)
    monkeypatch.setattr(scheduler, "_measured_total_ram_mb", lambda: _TOTAL_RAM_MB)


def _single_gpu_scheduler_with_scaling(monkeypatch: pytest.MonkeyPatch) -> tuple[InferenceScheduler, ProcessMap]:
    """A single-GPU scheduler whose ``scale_inference_processes`` adds/removes idle contexts on device 0.

    The stub grows or shrinks the process map toward the requested target by inserting or removing idle
    inference contexts, then returns the resulting loaded count, so both the RAM-pressure reduction and the
    whole-card restore observe a real change in the live process count.
    """
    process_map = ProcessMap(
        {
            0: make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
            1: make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, device_index=0),
        },
    )
    scheduler = _make_inference_scheduler(process_map=process_map, max_inference=_MAX_INFERENCE)

    def _scale(target_count: int, *, device_index: int | None = None, **_kwargs: object) -> int:
        loaded = process_map.num_loaded_inference_processes()
        while loaded > target_count:
            victim = next(
                pid
                for pid, info in process_map.items()
                if info.last_process_state == HordeProcessState.WAITING_FOR_JOB
            )
            del process_map[victim]
            loaded -= 1
        while loaded < target_count:
            new_pid = (max(process_map.keys()) + 1) if process_map.keys() else 0
            process_map[new_pid] = make_mock_process_info(
                new_pid,
                model_name=None,
                state=HordeProcessState.WAITING_FOR_JOB,
                device_index=0,
            )
            loaded += 1
        return process_map.num_loaded_inference_processes()

    scheduler._process_lifecycle.scale_inference_processes = _scale  # type: ignore[method-assign]
    scheduler._process_lifecycle.restore_safety_on_gpu = lambda: False  # type: ignore[method-assign]
    return scheduler, process_map


def _arm_drained_residency(scheduler: InferenceScheduler, model: str) -> None:
    """Hold a whole-card residency for ``model`` that has fully drained (no active job, cooldown lapsed).

    This is the state ``_restore_siblings_after_whole_card`` acts on: the residency model is set, nothing
    in flight is still using it, and the cooldown has elapsed, so the pass releases the residency and grows
    the pool back to the launched ceiling this cycle.
    """
    state = scheduler._whole_card_ledger.state_for(None)
    state.model = model
    state.cooldown_until = 0.0
    state.forecast = None


class TestRamShedOwnsItsRestore:
    """A whole-card release leaves a RAM shed standing; the RAM governor restores it on proven headroom."""

    def test_whole_card_release_does_not_regrow_a_pool_the_ram_governor_shed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Under the floor, the residency's release keeps the shed count and the shed record."""
        scheduler, process_map = _single_gpu_scheduler_with_scaling(monkeypatch)
        _pin_available_ram(scheduler, monkeypatch, _CRITICAL_AVAILABLE_RAM_MB)

        scheduler._reduce_processes_under_ram_pressure()
        assert process_map.num_loaded_inference_processes() == 1
        assert scheduler._ram_governor_state.worker_shed is not None

        _arm_drained_residency(scheduler, "CyberRealistic Pony")
        scheduler._restore_siblings_after_whole_card()

        assert process_map.num_loaded_inference_processes() == 1, (
            "a whole-card release regrew a pool the RAM governor shed while the host is still under its floor"
        )
        assert scheduler._ram_governor_state.worker_shed is not None

    def test_ram_governor_restores_the_shed_once_headroom_is_proven(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Healthy RAM after the residency's release returns the pool to plan and resolves the shed record."""
        scheduler, process_map = _single_gpu_scheduler_with_scaling(monkeypatch)
        _pin_available_ram(scheduler, monkeypatch, _CRITICAL_AVAILABLE_RAM_MB)
        scheduler._reduce_processes_under_ram_pressure()
        _arm_drained_residency(scheduler, "CyberRealistic Pony")
        scheduler._restore_siblings_after_whole_card()

        _pin_available_ram(scheduler, monkeypatch, _HEALTHY_AVAILABLE_RAM_MB)
        for _ in range(_MAX_INFERENCE):
            scheduler._restore_processes_after_ram_pressure()

        assert process_map.num_loaded_inference_processes() == _MAX_INFERENCE
        assert scheduler._ram_governor_state.worker_shed is None, (
            "the RAM governor returned the pool to plan, so its shed record is resolved"
        )

    def test_shed_count_does_not_accumulate_across_reserve_restore_cycles(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeat reduce/whole-card-release cycles under pressure keep one shed entry, not one per cycle."""
        scheduler, process_map = _single_gpu_scheduler_with_scaling(monkeypatch)
        _pin_available_ram(scheduler, monkeypatch, _CRITICAL_AVAILABLE_RAM_MB)

        for cycle in range(5):
            scheduler._reduce_processes_under_ram_pressure()
            assert process_map.num_loaded_inference_processes() == 1

            _arm_drained_residency(scheduler, f"model-{cycle}")
            scheduler._restore_siblings_after_whole_card()
            assert process_map.num_loaded_inference_processes() == 1

        worker_shed = scheduler._ram_governor_state.worker_shed
        recorded_shed = worker_shed.shed_process_count if worker_shed is not None else 0
        assert recorded_shed <= 1, (
            f"the recorded shed count grew to {recorded_shed} across cycles with one context outstanding; "
            "the shed bookkeeping accumulates one entry per cycle"
        )
