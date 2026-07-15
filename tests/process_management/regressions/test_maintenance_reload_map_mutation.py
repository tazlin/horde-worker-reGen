"""Reproduces a control-loop crash when the maintenance-mode pool reload mutates the process map mid-iteration.

On a maintenance-mode pop with no jobs in flight, the control loop replaces every inference slot to give the
pool a clean start. The replacement retires the slot from the active process map (a fresh launch spawns
separately), so iterating the live ``self._process_map.values()`` view while replacing raised
``RuntimeError: dictionary changed size during iteration``. That exception escaped the control-loop tick and
took the whole worker down. The reload must iterate a snapshot so every inference slot is replaced regardless
of the map mutation each replacement causes, and the once-per-reload maintenance flag must be set once rather
than re-set inside the per-process loop.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager


async def _noop_sleep(_delay: float) -> None:
    return None


async def test_maintenance_reload_replaces_every_inference_slot_without_map_mutation_crash() -> None:
    """A maintenance reload must replace all inference slots even as each replacement retires its map entry.

    The reload body is driven through the real control-loop tick. Each replacement pops the slot from the
    active map (mirroring ``retire_process``), the mutation that made a live-view iteration raise. The tick
    must complete, replace every inference slot exactly once, leave non-inference slots untouched, and set
    the once-per-reload maintenance flag a single time.
    """
    pm = make_testable_process_manager()
    pm._sleep = _noop_sleep  # type: ignore[method-assign]
    pm._last_status_message_time = time.time()

    # Isolate the maintenance branch: keep the pool-start housekeeping from adding slots mid-tick.
    pm._download_coordinator.maybe_start_inference_processes = Mock()  # type: ignore[method-assign]
    pm._download_coordinator.maybe_start_safety_processes = Mock()  # type: ignore[method-assign]

    inf0 = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB, process_type=HordeProcessType.INFERENCE)
    inf1 = make_mock_process_info(1, state=HordeProcessState.WAITING_FOR_JOB, process_type=HordeProcessType.INFERENCE)
    safety = make_mock_process_info(
        2,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    pm._process_map.update({0: inf0, 1: inf1, 2: safety})

    replaced: list[int] = []

    def _fake_replace(
        process_info: HordeProcessInfo,
        *,
        intentional_reason: str | None = None,
        **_kwargs: object,
    ) -> None:
        replaced.append(process_info.process_id)
        # The real replacement retires the slot from the active map; a fresh launch spawns separately. That
        # net removal is exactly what breaks a live ``.values()`` iteration.
        pm._process_map.pop(process_info.process_id, None)

    pm._process_lifecycle._replace_inference_process = _fake_replace  # type: ignore[method-assign]

    pm._state.last_pop_maintenance_mode = True
    assert pm.num_jobs_total == 0
    assert pm._job_popper._replaced_due_to_maintenance is False

    keep_running = await pm._control_loop_tick()

    assert keep_running is True
    # Every inference slot replaced exactly once; the safety slot left alone.
    assert sorted(replaced) == [0, 1]
    assert pm._job_popper._replaced_due_to_maintenance is True


async def test_maintenance_reload_does_not_fire_while_jobs_are_in_flight() -> None:
    """Guard: the reload's ``num_jobs_total == 0`` gate keeps it from cycling the pool under a live job.

    The reload is a between-jobs pool refresh; it must not tear slots out from under work still in flight.
    A tracked job on an otherwise reload-eligible maintenance tick must leave every slot untouched.
    """
    from tests.process_management.conftest import make_mock_job, track_popped_job_async

    pm = make_testable_process_manager()
    pm._sleep = _noop_sleep  # type: ignore[method-assign]
    pm._last_status_message_time = time.time()
    pm._download_coordinator.maybe_start_inference_processes = Mock()  # type: ignore[method-assign]
    pm._download_coordinator.maybe_start_safety_processes = Mock()  # type: ignore[method-assign]

    inf0 = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB, process_type=HordeProcessType.INFERENCE)
    pm._process_map.update({0: inf0})

    replaced: list[int] = []
    pm._process_lifecycle._replace_inference_process = (  # type: ignore[method-assign]
        lambda process_info, **_kwargs: replaced.append(process_info.process_id)
    )

    await track_popped_job_async(pm._job_tracker, make_mock_job())
    pm._state.last_pop_maintenance_mode = True
    assert pm.num_jobs_total > 0

    await pm._control_loop_tick()

    assert replaced == []
    assert pm._job_popper._replaced_due_to_maintenance is False
