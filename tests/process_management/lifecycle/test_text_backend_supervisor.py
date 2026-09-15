"""The text backend supervisor's launch, readiness gate, footprint, relaunch ladder and teardown.

Intervals here stay above Windows' 15.6 ms monotonic clock resolution: asyncio treats a sleep shorter than
the clock resolution as already due, so a sub-resolution poll interval spins without yielding wall time and
a patience window never elapses.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.owned_process_registry import OwnedProcessRegistry
from horde_worker_regen.process_management.lifecycle.text_backend_supervisor import (
    KoboldcppArguments,
    LaunchedProcess,
    TextBackendLaunchSpec,
    TextBackendState,
    TextBackendSupervisor,
    TextBackendSupervisorTimings,
    koboldcpp_launch_spec,
    port_is_free,
)
from horde_worker_regen.text_backends import TextBackendDescription, TextGenerationResult

FAST = TextBackendSupervisorTimings(
    ready_poll_interval=0.03,
    ready_probe_deadline=0.03,
    ready_patience=0.3,
    liveness_poll_interval=0.03,
    stop_grace=0.5,
    relaunch_backoff=(0.03, 0.06),
)


class FakeLaunched:
    """A process handle whose exit is scripted by the test."""

    def __init__(self, pid: int) -> None:
        """Create a running handle with the given pid."""
        self._pid = pid
        self._exit_code: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0

    @property
    def pid(self) -> int:
        """Return the scripted pid."""
        return self._pid

    def poll(self) -> int | None:
        """Return the scripted exit code, None while running."""
        return self._exit_code

    def exit(self, code: int) -> None:
        """Make the process appear to have exited with ``code``."""
        self._exit_code = code

    def terminate(self) -> None:
        """Record the terminate and exit as a terminated process would."""
        self.terminate_calls += 1
        self._exit_code = -15

    def kill(self) -> None:
        """Record the kill and exit as a killed process would."""
        self.kill_calls += 1
        self._exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        """Return the exit code, or raise as `Popen.wait` does while the process still runs."""
        if self._exit_code is None:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0.0)
        return self._exit_code


class StubBackend:
    """A `TextBackend` whose `ready()` answers follow a script, then a default; nothing else is exercised."""

    def __init__(self, ready_results: list[bool], *, default_ready: bool = True) -> None:
        """Script the first answers of `ready()`; later calls answer ``default_ready``."""
        self._ready_results = list(ready_results)
        self.default_ready = default_ready
        self.ready_calls = 0

    async def ready(self, *, deadline_seconds: float) -> bool:
        """Return the next scripted answer, or the default once the script is spent."""
        self.ready_calls += 1
        if self._ready_results:
            return self._ready_results.pop(0)
        return self.default_ready

    async def describe(self) -> TextBackendDescription:
        """Return a placeholder description."""
        return TextBackendDescription(model_name="stub", max_context_length=1, max_length=1)

    async def generate(
        self,
        payload: Mapping[str, object],
        *,
        generation_key: str,
        deadline_seconds: float,
    ) -> TextGenerationResult:
        """Return an empty generation."""
        return TextGenerationResult(text="")

    async def stop(self, *, generation_key: str) -> None:
        """Do nothing."""
        return

    async def close(self) -> None:
        """Do nothing."""
        return


class Launcher:
    """Records every launch and hands out scripted handles in order."""

    def __init__(self, handles: list[FakeLaunched]) -> None:
        """Queue the handles successive launches receive."""
        self._handles = list(handles)
        self.specs: list[TextBackendLaunchSpec] = []

    def __call__(self, spec: TextBackendLaunchSpec) -> LaunchedProcess:
        """Record the spec and return the next handle."""
        self.specs.append(spec)
        return self._handles.pop(0)


def free_port() -> int:
    """Return a loopback port nothing is listening on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def spec_on(port: int, tmp_path: Path, *, device_index: int | None = 0) -> TextBackendLaunchSpec:
    """Return a launch spec for a fake executable listening on ``port``."""
    return TextBackendLaunchSpec(
        executable=tmp_path / "backend.exe",
        arguments=("--port", str(port)),
        port=port,
        log_path=tmp_path / "logs" / "text_backend.log",
        device_index=device_index,
    )


async def wait_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    """Yield to the loop until ``predicate`` holds, failing the test after ``timeout`` seconds."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.02)


def test_koboldcpp_launch_spec_renders_the_expected_argv(tmp_path: Path) -> None:
    """The koboldcpp argv carries the model, loopback port and host, CUDA ordinal, layers, context, and quiet flags."""
    spec = koboldcpp_launch_spec(
        executable=tmp_path / "koboldcpp.exe",
        model_path=tmp_path / "model.gguf",
        port=5001,
        device_index=1,
        gpu_layers=99,
        context_length=4096,
        log_path=tmp_path / "text_backend.log",
    )

    assert spec.command == [
        str(tmp_path / "koboldcpp.exe"),
        KoboldcppArguments.MODEL,
        str(tmp_path / "model.gguf"),
        KoboldcppArguments.PORT,
        "5001",
        KoboldcppArguments.HOST,
        "127.0.0.1",
        KoboldcppArguments.USE_CUDA,
        "1",
        KoboldcppArguments.GPU_LAYERS,
        "99",
        KoboldcppArguments.CONTEXT_SIZE,
        "4096",
        KoboldcppArguments.SKIP_LAUNCHER,
        KoboldcppArguments.QUIET,
    ]
    assert spec.base_url == "http://127.0.0.1:5001"
    assert spec.device_index == 1


def test_koboldcpp_launch_spec_omits_the_cuda_flag_off_gpu(tmp_path: Path) -> None:
    """Without a device index the CUDA flag is absent and the footprint device is None."""
    spec = koboldcpp_launch_spec(
        executable=tmp_path / "koboldcpp",
        model_path=tmp_path / "model.gguf",
        port=5001,
        device_index=None,
        gpu_layers=0,
        context_length=2048,
        log_path=tmp_path / "text_backend.log",
    )

    assert KoboldcppArguments.USE_CUDA not in spec.arguments
    assert spec.device_index is None


def test_port_is_free_reports_a_held_port() -> None:
    """A bound port reads as held while the holder lives and free once it is closed."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        held_port = int(holder.getsockname()[1])
        assert port_is_free(held_port) is False
    assert port_is_free(held_port) is True


async def test_ready_after_polls_makes_the_backend_serving_and_measures_the_footprint(tmp_path: Path) -> None:
    """Readiness after several polls yields SERVING, a free-VRAM delta footprint, and a recorded PID."""
    handle = FakeLaunched(pid=424242)
    launcher = Launcher([handle])
    backend = StubBackend(ready_results=[False, False, True])
    registry = OwnedProcessRegistry(tmp_path / "owned.json")
    readings = iter([(10000.0, 16000.0), (7300.0, 16000.0)])

    supervisor = TextBackendSupervisor(
        launch_spec=spec_on(free_port(), tmp_path),
        backend=backend,
        owned_registry=registry,
        launch_process=launcher,
        read_device_free_total_mb=lambda device_index: next(readings),
        timings=FAST,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: supervisor.is_serving)
    assert backend.ready_calls == 3
    assert supervisor.launch_count == 1
    assert supervisor.footprint_mb == pytest.approx(2700.0)
    assert supervisor.pids == (424242,)
    recorded = registry._load()
    assert [record.os_pid for record in recorded] == [424242]
    assert recorded[0].process_type == HordeProcessType.TEXT_BACKEND.name

    await supervisor.stop()
    await asyncio.wait_for(task, timeout=5.0)
    assert handle.terminate_calls == 1
    assert supervisor.state is TextBackendState.STOPPED
    assert supervisor.pids == ()
    assert registry._load() == []


async def test_footprint_is_unmeasured_without_a_reader(tmp_path: Path) -> None:
    """No free-VRAM reader means no footprint, not a zero."""
    handle = FakeLaunched(pid=1)
    supervisor = TextBackendSupervisor(
        launch_spec=spec_on(free_port(), tmp_path),
        backend=StubBackend(ready_results=[True]),
        launch_process=Launcher([handle]),
        timings=FAST,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: supervisor.is_serving)
    assert supervisor.footprint_mb is None

    await supervisor.stop()
    await asyncio.wait_for(task, timeout=5.0)


async def test_an_exit_before_ready_is_relaunched_after_backoff(tmp_path: Path) -> None:
    """A process that dies during start-up is relaunched and the second launch can serve."""
    first = FakeLaunched(pid=11)
    first.exit(1)
    second = FakeLaunched(pid=12)
    launcher = Launcher([first, second])
    supervisor = TextBackendSupervisor(
        launch_spec=spec_on(free_port(), tmp_path),
        backend=StubBackend(ready_results=[True]),
        launch_process=launcher,
        timings=FAST,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: supervisor.is_serving)
    assert supervisor.launch_count == 2
    assert supervisor.pids == (12,)

    await supervisor.stop()
    await asyncio.wait_for(task, timeout=5.0)


async def test_missing_the_ready_patience_stops_the_process_and_relaunches(tmp_path: Path) -> None:
    """A launch that never answers ready within the patience is terminated and tried again."""
    first = FakeLaunched(pid=21)
    second = FakeLaunched(pid=22)
    launcher = Launcher([first, second])
    never_ready = StubBackend(ready_results=[], default_ready=False)
    supervisor = TextBackendSupervisor(
        launch_spec=spec_on(free_port(), tmp_path),
        backend=never_ready,
        launch_process=launcher,
        timings=FAST,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: first.terminate_calls == 1)
    assert supervisor.launch_count == 1
    never_ready.default_ready = True
    await wait_until(lambda: supervisor.is_serving)
    assert supervisor.launch_count == 2
    assert supervisor.pids == (22,)

    await supervisor.stop()
    await asyncio.wait_for(task, timeout=5.0)


async def test_an_exit_while_serving_is_relaunched(tmp_path: Path) -> None:
    """A serving process that exits is relaunched and the registry follows the new PID."""
    first = FakeLaunched(pid=31)
    second = FakeLaunched(pid=32)
    registry = OwnedProcessRegistry(tmp_path / "owned.json")
    supervisor = TextBackendSupervisor(
        launch_spec=spec_on(free_port(), tmp_path),
        backend=StubBackend(ready_results=[True, True]),
        owned_registry=registry,
        launch_process=Launcher([first, second]),
        timings=FAST,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: supervisor.is_serving)
    first.exit(3)
    await wait_until(lambda: supervisor.launch_count == 2 and supervisor.is_serving)
    assert [record.os_pid for record in registry._load()] == [32]

    await supervisor.stop()
    await asyncio.wait_for(task, timeout=5.0)


async def test_a_held_port_defers_the_launch_without_starting_anything(tmp_path: Path) -> None:
    """A port another process holds defers the launch into backoff; no process is started."""
    launcher = Launcher([FakeLaunched(pid=41)])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        held_port = int(holder.getsockname()[1])
        supervisor = TextBackendSupervisor(
            launch_spec=spec_on(held_port, tmp_path),
            backend=StubBackend(ready_results=[True]),
            launch_process=launcher,
            timings=FAST,
        )
        task = asyncio.create_task(supervisor.run())

        await wait_until(lambda: supervisor.state is TextBackendState.BACKING_OFF)
        assert launcher.specs == []
        assert supervisor.launch_count == 0

        await supervisor.stop()
        await asyncio.wait_for(task, timeout=5.0)


async def test_stop_during_backoff_ends_the_run_without_another_launch(tmp_path: Path) -> None:
    """Stopping while backing off leaves the run ended and the second handle unused."""
    exited = FakeLaunched(pid=51)
    exited.exit(2)
    slow_backoff = FAST.model_copy(update={"relaunch_backoff": (30.0,)})
    launcher = Launcher([exited, FakeLaunched(pid=52)])
    supervisor = TextBackendSupervisor(
        launch_spec=spec_on(free_port(), tmp_path),
        backend=StubBackend(ready_results=[True]),
        launch_process=launcher,
        timings=slow_backoff,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: supervisor.state is TextBackendState.BACKING_OFF)
    await supervisor.stop()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert supervisor.state is TextBackendState.STOPPED
    assert supervisor.launch_count == 1
    assert len(launcher.specs) == 1
