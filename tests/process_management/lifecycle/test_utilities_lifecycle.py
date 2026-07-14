"""Lifecycle tests for the image-utilities lane.

The lane's parent-side adapter is injected as a fake through the ``ProcessEntryPoints`` seam, so these
tests exercise the ``ProcessLifecycleManager`` start / defer / end / replace machinery without launching a
real cross-venv subprocess.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from unittest.mock import Mock

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType, WorkerCapability
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from tests.process_management.conftest import make_test_card_runtimes, make_test_runtime_config


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
) -> ProcessLifecycleManager:
    """Build a lifecycle manager with the utilities lane enabled and a fake adapter factory injected."""
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 1
    bridge_data.enable_pipeline_disaggregation = False
    bridge_data.post_processing_lane_enabled = False
    bridge_data.safety_on_gpu = False
    bridge_data.enable_image_utilities = True
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
    ) -> _FakeAdapter:
        adapter = _FakeAdapter()
        constructed.append(adapter)
        return adapter

    return ProcessLifecycleManager(
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
        device_free_mb_provider=device_free_mb_provider,
        device_total_vram_mb_provider=device_total_vram_mb_provider,
    )


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
