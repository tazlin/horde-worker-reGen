"""Lifecycle tests for the image-utilities lane.

The lane's parent-side adapter is injected as a fake through the ``ProcessEntryPoints`` seam, so these
tests exercise the ``ProcessLifecycleManager`` start / defer / end / replace machinery without launching a
real cross-venv subprocess. The lane starts only when a job needs it, so the shared builder registers a
background-strip alchemy form as waiting, which wants the lane with no image job in hand, unless a row asks for no
demand.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable
from unittest.mock import Mock

from horde_sdk.generation_parameters.alchemy.consts import KNOWN_MISC_POST_PROCESSORS

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType, WorkerCapability
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner, ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.workload_flow import (
    AlchemyFormDemand,
    UtilitiesLaneDemandReason,
)
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from tests.process_management.conftest import (
    device_wide_reading_provider,
    make_job_pop_response,
    make_test_card_runtimes,
    make_test_runtime_config,
)


class _FakeHandle:
    """A minimal :class:`ChildProcessHandle` over a fake utilities lane."""

    def __init__(self) -> None:
        self.alive = True
        self.terminated = False

    @property
    def pid(self) -> int | None:
        return None

    @property
    def exitcode(self) -> int | None:
        return None

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def join(self, timeout: float | None = None) -> None:
        return


class _FakeAdapter:
    """A fake utilities lane adapter that never spawns a subprocess."""

    def __init__(self) -> None:
        self.started = False
        self._handle = _FakeHandle()

    @property
    def handle(self) -> _FakeHandle:
        return self._handle

    def start(self) -> None:
        self.started = True


def _make_utilities_plm(
    *,
    constructed: list[_FakeAdapter],
    device_free_mb_provider: Callable[[int], float | None] | None = None,
    device_total_vram_mb_provider: Callable[[int], float | None] | None = None,
    process_map: ProcessMap | None = None,
    demand: bool = True,
) -> ProcessLifecycleManager:
    """Build a lifecycle manager with the utilities lane enabled and a fake adapter factory injected.

    ``demand`` registers a waiting background-strip alchemy form, which wants the lane with no image job in hand.
    Extended ControlNet is opted in throughout, as it is by default, and does not by itself want the lane.
    """
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 1
    bridge_data.enable_pipeline_disaggregation = False
    bridge_data.post_processing_lane_enabled = False
    bridge_data.safety_on_gpu = False
    bridge_data.enable_image_utilities = True
    bridge_data.extended_controlnet = True
    bridge_data.allow_controlnet = True
    bridge_data.process_timeout = 120
    bridge_data.inference_step_timeout = 60
    bridge_data.inference_first_step_timeout = 120
    bridge_data.inference_stuck_step_repeat_limit = 20
    bridge_data.preload_timeout = 120
    bridge_data.download_timeout = 120
    bridge_data.post_process_timeout = 60
    bridge_data.max_batch = 1
    bridge_data.exit_on_unhandled_faults = False

    def _factory(
        process_id: int,
        process_message_queue: object,
        control_connection: object,
        process_launch_identifier: int,
        *,
        device_index: int,
        python_executable: str,
        child_env: dict[str, str],
        log_path: str | None = None,
    ) -> _FakeAdapter:
        adapter = _FakeAdapter()
        constructed.append(adapter)
        return adapter

    plm = ProcessLifecycleManager(
        ctx=multiprocessing.get_context("spawn"),  # type: ignore[arg-type]
        process_map=process_map if process_map is not None else ProcessMap({}),
        horde_model_map=Mock(),
        job_tracker=JobTracker(),
        process_message_queue=Mock(),
        card_runtimes=make_test_card_runtimes(target_process_count=2),
        disk_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        max_safety_processes=1,
        amd_gpu=False,
        directml=None,
        abort_callback=Mock(),
        state=WorkerState(),
        entry_points=ProcessEntryPoints(utilities_adapter_factory=_factory),
        device_reading_provider=(
            None
            if device_free_mb_provider is None
            else device_wide_reading_provider(device_free_mb_provider, device_total_vram_mb_provider)
        ),
    )
    _set_lane_form_waiting(plm, waiting=demand)
    return plm


_STRIP_FORM = KNOWN_MISC_POST_PROCESSORS.strip_background.value


def _set_lane_form_waiting(plm: ProcessLifecycleManager, *, waiting: bool) -> None:
    """Report one background-strip alchemy form waiting for the lane, or none."""
    demand = AlchemyFormDemand(waiting=(_STRIP_FORM,)) if waiting else AlchemyFormDemand()
    plm.set_utilities_alchemy_form_demand_provider(lambda: demand)


def test_utilities_lane_disabled_is_a_noop() -> None:
    """With the flag off (its default), the lane never starts."""
    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(constructed=constructed)
    plm._runtime_config.bridge_data.enable_image_utilities = False

    assert plm.utilities_process_enabled() is False
    assert plm.start_utilities_processes() is False
    assert plm._process_map.num_utilities_processes() == 0
    assert constructed == []


def test_utilities_lane_starts_and_is_capability_discoverable() -> None:
    """When enabled and admitted, the lane starts, appears in the map, and declares IMAGE_UTILITIES."""
    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(constructed=constructed)

    assert plm.utilities_process_enabled() is True
    assert plm.start_utilities_processes() is True
    assert plm._process_map.num_utilities_processes() == 1
    assert len(constructed) == 1 and constructed[0].started is True

    # Discoverable by capability rather than a per-type finder.
    capable = plm._process_map.get_capable_processes(WorkerCapability.IMAGE_UTILITIES)
    assert len(capable) == 1
    assert capable[0].process_type is HordeProcessType.UTILITIES

    # Idempotent: a second start does not create a second lane.
    assert plm.start_utilities_processes() is True
    assert plm._process_map.num_utilities_processes() == 1


def test_utilities_lane_is_admitted_through_the_vram_gate() -> None:
    """With ample free VRAM the lane is admitted (not deferred)."""
    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(
        constructed=constructed,
        device_free_mb_provider=lambda _index: 24000.0,
        device_total_vram_mb_provider=lambda _index: 24576.0,
    )
    assert plm.start_utilities_processes() is True
    assert plm.has_pending_gpu_starts() is False
    assert plm._process_map.num_utilities_processes() == 1


def test_full_device_defers_then_retries_without_wedging() -> None:
    """A pressured card defers the lane; a permanently-full device must not wedge, and headroom retries it.

    Hostile self-infliction: repeatedly draining while the device stays full must neither raise nor start
    the lane, and the deferral must remain pending so the next headroom sample re-evaluates it.
    """
    constructed: list[_FakeAdapter] = []
    free_holder = {"mb": 100.0}
    plm = _make_utilities_plm(
        constructed=constructed,
        device_free_mb_provider=lambda _index: free_holder["mb"],
        device_total_vram_mb_provider=lambda _index: 24576.0,
    )

    # A near-full card defers the start rather than launching onto pressure.
    assert plm.start_utilities_processes() is False
    assert plm.has_pending_gpu_starts() is True
    assert plm._process_map.num_utilities_processes() == 0

    # The rest of the manager loop keeps ticking: draining a still-full device neither raises nor starts.
    for _ in range(5):
        assert plm.drain_pending_gpu_starts() == 0
    assert plm.has_pending_gpu_starts() is True
    assert plm._process_map.num_utilities_processes() == 0

    # Once the device drains, the deferred start is honoured on the next drain.
    free_holder["mb"] = 24000.0
    assert plm.drain_pending_gpu_starts() == 1
    assert plm.has_pending_gpu_starts() is False
    assert plm._process_map.num_utilities_processes() == 1


def test_end_utilities_processes_tears_down_cleanly() -> None:
    """Ending the lane marks the intent, sends the end command, and moves it to PROCESS_ENDING."""
    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(constructed=constructed)
    assert plm.start_utilities_processes() is True
    utilities = plm._process_map.get_capable_processes(WorkerCapability.IMAGE_UTILITIES)[0]

    plm.end_utilities_processes()

    assert utilities.end_intended is True
    assert utilities.last_process_state == HordeProcessState.PROCESS_ENDING


def test_crashed_lane_is_recovered_by_the_reaper() -> None:
    """A dead lane (handle reports not-alive) is reaped and replaced through the utilities state machine."""
    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(constructed=constructed)
    assert plm.start_utilities_processes() is True
    utilities = plm._process_map.get_capable_processes(WorkerCapability.IMAGE_UTILITIES)[0]

    # Model the subprocess dying: the handle now reports not-alive.
    constructed[0].handle.alive = False

    assert plm._reap_if_crashed(utilities) is True
    assert plm.utilities_processes_should_be_replaced is True

    # Drive the replacement machine to completion: end -> delete -> start a fresh lane.
    for _ in range(4):
        plm._replace_all_utilities_process()
    assert plm._process_map.num_utilities_processes() == 1
    assert len(constructed) == 2


def test_operator_recycle_respawns_utilities_lane_without_counting_a_recovery() -> None:
    """An operator recycle of the utilities lane respawns it but, being intentional, is not a crash recovery."""
    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(constructed=constructed)
    assert plm.start_utilities_processes() is True
    utilities = plm._process_map.get_capable_processes(WorkerCapability.IMAGE_UTILITIES)[0]
    recoveries_before = plm._num_process_recoveries

    assert plm.recycle_lane_process(utilities) is True
    assert plm._utilities_replacement_intentional is True

    for _ in range(4):
        plm._replace_all_utilities_process()

    assert plm._process_map.num_utilities_processes() == 1  # a fresh lane came up
    assert len(constructed) == 2
    assert plm._utilities_replacement_intentional is False  # intentional flag consumed
    assert plm._num_process_recoveries == recoveries_before  # not counted as a crash recovery


async def _queue_strip_only_job(plm: ProcessLifecycleManager) -> None:
    """Pop a generation job whose only post-processing is background removal."""
    job = make_job_pop_response(post_processing=["strip_background"])
    await plm._job_tracker.record_popped_job(job, time_popped=time.time())


async def test_start_refuses_without_demand_and_starts_with_it() -> None:
    """Every start path goes through the demand check, so a lane nothing needs is not started."""
    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(constructed=constructed, demand=False)

    assert plm.utilities_lane_demand() is None
    assert plm.start_utilities_processes() is False
    assert constructed == []

    await _queue_strip_only_job(plm)
    assert plm.utilities_lane_demand() is UtilitiesLaneDemandReason.STRIP_ONLY_JOB
    assert plm.start_utilities_processes() is True
    assert len(constructed) == 1


def test_deferred_start_is_dropped_once_its_demand_lapses() -> None:
    """A start deferred for VRAM is retried only while the lane is still wanted."""
    constructed: list[_FakeAdapter] = []
    free_holder = {"mb": 100.0}
    plm = _make_utilities_plm(
        constructed=constructed,
        device_free_mb_provider=lambda _index: free_holder["mb"],
        device_total_vram_mb_provider=lambda _index: 24576.0,
    )
    assert plm.start_utilities_processes() is False
    assert plm.has_pending_gpu_starts() is True

    _set_lane_form_waiting(plm, waiting=False)
    free_holder["mb"] = 24000.0
    assert plm.drain_pending_gpu_starts() == 0
    assert plm.has_pending_gpu_starts() is False
    assert constructed == []


async def test_driver_starts_once_per_demand_episode() -> None:
    """The per-tick driver starts the lane on the first tick with demand and not again while it runs."""
    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(constructed=constructed, demand=False)

    idle = plm.reconcile_utilities_demand(pause_stranded=False)
    assert idle.started_for is None
    assert idle.held_down is True

    await _queue_strip_only_job(plm)
    await _queue_strip_only_job(plm)
    first = plm.reconcile_utilities_demand(pause_stranded=False)
    assert first.started_for is UtilitiesLaneDemandReason.STRIP_ONLY_JOB
    for _ in range(5):
        assert plm.reconcile_utilities_demand(pause_stranded=False).started_for is None
    assert len(constructed) == 1


async def test_driver_does_not_start_beside_a_replacement_or_a_deferred_start() -> None:
    """A lane the replacement machine or a VRAM deferral is already bringing up is left to that path."""
    constructed: list[_FakeAdapter] = []
    free_holder = {"mb": 24000.0}
    plm = _make_utilities_plm(
        constructed=constructed,
        device_free_mb_provider=lambda _index: free_holder["mb"],
        device_total_vram_mb_provider=lambda _index: 24576.0,
        demand=False,
    )
    await _queue_strip_only_job(plm)

    plm._utilities_processes_should_be_replaced = True
    assert plm.reconcile_utilities_demand(pause_stranded=False).started_for is None
    assert constructed == []
    plm._utilities_processes_should_be_replaced = False

    free_holder["mb"] = 100.0
    assert plm.start_utilities_processes() is False
    assert plm.has_pending_gpu_starts() is True
    assert plm.reconcile_utilities_demand(pause_stranded=False).started_for is None
    assert constructed == []

    free_holder["mb"] = 24000.0
    assert plm.drain_pending_gpu_starts() == 1
    assert len(constructed) == 1


async def test_driver_restores_a_stranded_ladder_pause_only_with_demand() -> None:
    """A ladder pause no owner can restore is lifted by the driver once a job needs the lane."""
    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(constructed=constructed, demand=True)
    assert plm.start_utilities_processes() is True
    assert plm.pause_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    for _ in range(4):
        plm._replace_all_utilities_process()
    assert plm._process_map.num_utilities_processes() == 0
    assert plm.is_utilities_gpu_paused is True

    assert plm.reconcile_utilities_demand(pause_stranded=False).started_for is None
    assert plm.is_utilities_gpu_paused is True

    _set_lane_form_waiting(plm, waiting=False)
    assert plm.reconcile_utilities_demand(pause_stranded=True).started_for is None
    assert plm.is_utilities_gpu_paused is True, "a stranded pause with nothing to serve is left alone"

    _set_lane_form_waiting(plm, waiting=True)
    restored = plm.reconcile_utilities_demand(pause_stranded=True)
    assert restored.started_for is UtilitiesLaneDemandReason.ALCHEMY_FORM_WAITING
    assert plm.is_utilities_gpu_paused is False
    assert len(constructed) == 2


def test_crash_replacement_without_demand_leaves_the_lane_down_and_counts_the_recovery() -> None:
    """The replacement's final start obeys the demand check, and the crash is still a recovery."""
    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(constructed=constructed)
    assert plm.start_utilities_processes() is True
    utilities = plm._process_map.get_capable_processes(WorkerCapability.IMAGE_UTILITIES)[0]
    recoveries_before = plm._num_process_recoveries

    _set_lane_form_waiting(plm, waiting=False)
    constructed[0].handle.alive = False
    assert plm._reap_if_crashed(utilities) is True
    for _ in range(4):
        plm._replace_all_utilities_process()

    assert plm._process_map.num_utilities_processes() == 0
    assert len(constructed) == 1
    assert plm.utilities_processes_should_be_replaced is False
    assert plm._num_process_recoveries == recoveries_before + 1


async def test_an_extended_controlnet_job_is_demand_and_the_opt_in_alone_is_not() -> None:
    """With extended ControlNet opted in the lane is not wanted until an extended job with a source image arrives."""
    from tests.process_management.test_image_utilities_annotation import _make_controlnet_job

    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(constructed=constructed, demand=False)
    assert plm.utilities_lane_demand() is None

    job = _make_controlnet_job(control_type="lineart")
    await plm._job_tracker.record_popped_job(job, time_popped=time.time())

    assert plm.utilities_lane_demand() is UtilitiesLaneDemandReason.CONTROLNET_ANNOTATION


def test_a_lane_down_by_policy_or_running_will_serve() -> None:
    """A job may wait for a lane that is down by policy, starting or running."""
    constructed: list[_FakeAdapter] = []
    plm = _make_utilities_plm(constructed=constructed, demand=False)
    assert plm.utilities_lane_will_serve() is True, "down by policy, startable on demand"

    _set_lane_form_waiting(plm, waiting=True)
    assert plm.start_utilities_processes() is True
    assert plm.utilities_lane_will_serve() is True, "running"


def test_a_crashed_paused_or_refused_lane_will_not_serve() -> None:
    """A lane that crashed, was paused by an owner or had its start refused for VRAM is not waited for."""
    constructed: list[_FakeAdapter] = []
    free_holder = {"mb": 24000.0}
    plm = _make_utilities_plm(
        constructed=constructed,
        device_free_mb_provider=lambda _index: free_holder["mb"],
        device_total_vram_mb_provider=lambda _index: 24576.0,
    )
    assert plm.start_utilities_processes() is True
    utilities = plm._process_map.get_capable_processes(WorkerCapability.IMAGE_UTILITIES)[0]
    constructed[0].handle.alive = False
    assert plm.utilities_lane_will_serve() is False, "crashed"
    assert plm._reap_if_crashed(utilities) is True
    assert plm.utilities_lane_will_serve() is False, "crash replacement under way"
    for _ in range(4):
        plm._replace_all_utilities_process()
    assert plm.utilities_lane_will_serve() is True, "replaced"

    assert plm.pause_utilities_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    assert plm.utilities_lane_will_serve() is False, "paused"
    for _ in range(4):
        plm._replace_all_utilities_process()
    assert plm.utilities_lane_will_serve() is False, "paused and down"

    plm._utilities_gpu_paused = False
    free_holder["mb"] = 100.0
    assert plm.start_utilities_processes() is False
    assert plm.has_pending_gpu_starts() is True
    assert plm.utilities_lane_will_serve() is False, "start refused for VRAM"
