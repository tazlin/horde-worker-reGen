"""The text backend supervisor's launch, readiness gate, footprint, relaunch ladder and teardown.

Intervals here stay above Windows' 15.6 ms monotonic clock resolution: asyncio treats a sleep shorter than
the clock resolution as already due, so a sub-resolution poll interval spins without yielding wall time and
a patience window never elapses.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import aiohttp
import psutil
import pytest
from horde_model_reference.text_backend_names import TEXT_BACKENDS

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    TEXT_BACKEND_PROCESS_ID,
    TextBackendActivity,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.owned_process_registry import OwnedProcessRegistry
from horde_worker_regen.process_management.lifecycle.text_backend_supervisor import (
    LaunchedProcess,
    LaunchSpecFactory,
    TextBackendState,
    TextBackendSupervisor,
    TextBackendSupervisorTimings,
    default_launch_process,
    parse_backend_buffer_sizes,
    port_is_free,
)
from horde_worker_regen.text_backends import (
    KoboldApiTextBackend,
    TextBackendDescription,
    TextBackendError,
    TextBackendLaunchSettings,
    TextBackendLaunchSpec,
    TextBackendUnavailable,
    TextGenerationResult,
    UnsupportedTextBackendError,
    build_launch_spec,
)
from horde_worker_regen.text_backends.provision import TextBackendProvisionError
from worker_bootstrap.koboldcpp_bin import koboldcpp_executable

REAL_MODEL_ENV_VAR = "HORDE_TEXT_SMOKE_GGUF"
"""Path of a GGUF the real-binary row loads; the row skips when it is unset or the file is missing."""

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

    def __init__(
        self,
        ready_results: list[bool],
        *,
        default_ready: bool = True,
        description: TextBackendDescription | None = None,
        describe_error: TextBackendError | None = None,
    ) -> None:
        """Script the first answers of `ready()`; later calls answer ``default_ready``."""
        self._ready_results = list(ready_results)
        self.default_ready = default_ready
        self.ready_calls = 0
        self.describe_calls = 0
        self._description = description or TextBackendDescription(
            model_name="stub",
            max_context_length=1,
            max_length=1,
        )
        self._describe_error = describe_error

    async def ready(self, *, deadline_seconds: float) -> bool:
        """Return the next scripted answer, or the default once the script is spent."""
        self.ready_calls += 1
        if self._ready_results:
            return self._ready_results.pop(0)
        return self.default_ready

    async def describe(self) -> TextBackendDescription:
        """Return the scripted description, or raise the scripted failure."""
        self.describe_calls += 1
        if self._describe_error is not None:
            raise self._describe_error
        return self._description

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


def factory_for(spec: TextBackendLaunchSpec) -> LaunchSpecFactory:
    """Return a factory handing the supervisor a spec that is already rendered.

    The supervisor obtains its backend through a factory, and these rows are about what it does once it
    has one, so the spec is built up front (a free port is probed while the test knows it is free) and the
    factory only hands it over. The rows about obtaining supply their own factory.
    """
    return lambda: spec


def free_port() -> int:
    """Return a loopback port nothing is listening on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def spec_on(port: int, tmp_path: Path, *, device_index: int | None = 0) -> TextBackendLaunchSpec:
    """Return a launch spec for a fake executable listening on ``port``; no backend is implied."""
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
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
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
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
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
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
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
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
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
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
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
            launch_spec_factory=factory_for(spec_on(held_port, tmp_path)),
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
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
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


BACKEND_LOAD_LOG = """load_tensors: offloaded 29/29 layers to GPU
load_tensors:        CUDA0 model buffer size =  1918.35 MiB
load_tensors:    CUDA_Host model buffer size =   308.23 MiB
llama_context:  CUDA_Host  output buffer size =     1.47 MiB
llama_kv_cache:      CUDA0 KV buffer size =   476.00 MiB
sched_reserve:      CUDA0 compute buffer size =   278.79 MiB
sched_reserve:  CUDA_Host compute buffer size =    16.30 MiB
Load Text Model OK: True
"""
"""The load-time lines koboldcpp prints, copied from a real backend log (one line per device, plus the
``output buffer size`` line that no buffer pattern may claim)."""


class LoggingLauncher:
    """Hands out scripted handles and appends each launch's own backend output to the spec's log."""

    def __init__(self, handles: list[FakeLaunched], outputs: list[str]) -> None:
        """Queue the handles and the text each successive launch writes before it becomes ready."""
        self._handles = list(handles)
        self._outputs = list(outputs)
        self.specs: list[TextBackendLaunchSpec] = []

    def __call__(self, spec: TextBackendLaunchSpec) -> LaunchedProcess:
        """Append this launch's output as the real backend would, then return the next handle."""
        self.specs.append(spec)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        with spec.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(self._outputs.pop(0))
        return self._handles.pop(0)


def test_buffer_sizes_are_summed_per_kind_across_device_lines() -> None:
    """Each buffer kind sums its device lines, and the output-buffer line belongs to none of them."""
    sizes = parse_backend_buffer_sizes(BACKEND_LOAD_LOG)

    assert sizes.model_mebibytes == 2227
    assert sizes.kv_mebibytes == 476
    assert sizes.compute_mebibytes == 295


def test_buffer_sizes_are_none_when_the_backend_printed_none() -> None:
    """A backend that prints no buffer lines leaves every size unset rather than reporting zero."""
    sizes = parse_backend_buffer_sizes("Load Text Model OK: True\n")

    assert sizes.model_mebibytes is None
    assert sizes.kv_mebibytes is None
    assert sizes.compute_mebibytes is None


async def test_the_stopped_row_names_the_backend_without_claiming_a_process(tmp_path: Path) -> None:
    """Before anything is launched the row exists, is external, is not alive, and carries no pid."""
    supervisor = TextBackendSupervisor(
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
        backend=StubBackend(ready_results=[True]),
        backend_kind=TEXT_BACKENDS.koboldcpp,
        launch_process=Launcher([FakeLaunched(pid=61)]),
        timings=FAST,
    )

    row = supervisor.to_process_snapshot()

    assert row.is_external is True
    assert row.process_id == TEXT_BACKEND_PROCESS_ID
    assert row.process_type == HordeProcessType.TEXT_BACKEND.name
    assert row.is_alive is False
    assert row.os_pid is None
    assert row.display_state == "stopped"
    assert row.text_backend is not None
    assert row.text_backend.kind == str(TEXT_BACKENDS.koboldcpp)
    assert row.text_backend.launch_count == 0
    assert row.text_backend.model_name is None


async def test_the_row_reports_the_requests_the_flow_has_in_flight(tmp_path: Path) -> None:
    """The supervisor watches the process and never sees a generation, so the flow hands it the work.

    Without that the row would read idle through every generation it served, which is the opposite of
    what an operator watching the process table is looking for.
    """
    supervisor = TextBackendSupervisor(
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
        backend=StubBackend(ready_results=[True]),
        backend_kind=TEXT_BACKENDS.koboldcpp,
        launch_process=Launcher([FakeLaunched(pid=62)]),
        timings=FAST,
    )

    row = supervisor.to_process_snapshot(
        activity=TextBackendActivity(active_requests=2, queued_requests=1, tokens_per_second=31.5),
    )

    assert row.is_busy is True
    assert row.text_backend is not None
    assert row.text_backend.active_requests == 2
    assert row.text_backend.queued_requests == 1
    assert row.text_backend.tokens_per_second == 31.5


async def test_a_row_with_nothing_in_flight_is_idle(tmp_path: Path) -> None:
    """A backend the flow is asking nothing of is idle; a busy row would read as a lane holding work."""
    supervisor = TextBackendSupervisor(
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
        backend=StubBackend(ready_results=[True]),
        backend_kind=TEXT_BACKENDS.koboldcpp,
        launch_process=Launcher([FakeLaunched(pid=63)]),
        timings=FAST,
    )

    with_nothing = supervisor.to_process_snapshot(activity=TextBackendActivity())
    without_activity = supervisor.to_process_snapshot()

    for row in (with_nothing, without_activity):
        assert row.is_busy is False
        assert row.text_backend is not None
        assert row.text_backend.active_requests == 0
        assert row.text_backend.queued_requests == 0
        assert row.text_backend.tokens_per_second is None


async def test_the_ready_row_carries_the_description_footprint_pid_and_buffer_sizes(tmp_path: Path) -> None:
    """A serving backend's row states what it loaded, what it cost, and what its loader reported."""
    handle = FakeLaunched(pid=71)
    backend = StubBackend(
        ready_results=[True],
        description=TextBackendDescription(
            model_name="koboldcpp/Llama-3.2-3B",
            max_context_length=4352,
            max_length=512,
        ),
    )
    readings = iter([(10000.0, 16000.0), (7300.0, 16000.0)])
    port = free_port()
    supervisor = TextBackendSupervisor(
        launch_spec_factory=factory_for(spec_on(port, tmp_path)),
        backend=backend,
        backend_kind=TEXT_BACKENDS.koboldcpp,
        launch_process=LoggingLauncher([handle], [BACKEND_LOAD_LOG]),
        read_device_free_total_mb=lambda device_index: next(readings),
        timings=FAST,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: supervisor.is_serving)
    row = supervisor.to_process_snapshot()

    assert row.display_state == "ready"
    assert row.is_alive is True
    assert row.os_pid == 71
    assert row.device_index == 0
    assert row.loaded_horde_model_name == "koboldcpp/Llama-3.2-3B"
    detail = row.text_backend
    assert detail is not None
    assert detail.model_name == "koboldcpp/Llama-3.2-3B"
    assert detail.context_length == 4352
    assert detail.max_length == 512
    assert detail.port == port
    assert detail.launch_count == 1
    assert detail.footprint_mb == 2700
    assert detail.ready_since is not None
    assert detail.last_health_ok_at == detail.ready_since
    assert detail.relaunch_backoff_seconds is None
    assert (detail.model_buffer_mb, detail.kv_buffer_mb, detail.compute_buffer_mb) == (2227, 476, 295)
    assert backend.describe_calls == 1

    await supervisor.stop()
    await asyncio.wait_for(task, timeout=5.0)


async def test_the_launching_row_reports_launching_before_readiness(tmp_path: Path) -> None:
    """A backend still loading its model reads as launching, with no description and no buffer sizes."""
    starting = StubBackend(ready_results=[], default_ready=False)
    supervisor = TextBackendSupervisor(
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
        backend=starting,
        launch_process=LoggingLauncher([FakeLaunched(pid=81)], [BACKEND_LOAD_LOG]),
        timings=FAST,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: supervisor.state is TextBackendState.LAUNCHING and supervisor.launch_count == 1)
    row = supervisor.to_process_snapshot()

    assert row.display_state == "launching"
    assert row.is_alive is True
    assert row.text_backend is not None
    assert row.text_backend.model_name is None
    assert row.text_backend.model_buffer_mb is None
    assert row.text_backend.launching_since is not None, "the readiness patience is measured from here"
    assert starting.describe_calls == 0

    await supervisor.stop()
    await asyncio.wait_for(task, timeout=5.0)


async def test_the_backing_off_row_counts_down_to_the_relaunch(tmp_path: Path) -> None:
    """A backend waiting to be relaunched says how long is left, and holds no detail from the dead launch."""
    exited = FakeLaunched(pid=91)
    exited.exit(1)
    slow_backoff = FAST.model_copy(update={"relaunch_backoff": (30.0,)})
    supervisor = TextBackendSupervisor(
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
        backend=StubBackend(ready_results=[True]),
        launch_process=LoggingLauncher([exited, FakeLaunched(pid=92)], [BACKEND_LOAD_LOG, BACKEND_LOAD_LOG]),
        timings=slow_backoff,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: supervisor.state is TextBackendState.BACKING_OFF)
    row = supervisor.to_process_snapshot()

    assert row.is_alive is False
    assert row.display_state is not None
    assert row.display_state.startswith("relaunching in ")
    assert row.text_backend is not None
    assert row.text_backend.relaunch_backoff_seconds is not None
    assert 0.0 < row.text_backend.relaunch_backoff_seconds <= 30.0
    assert row.text_backend.model_name is None
    assert row.text_backend.ready_since is None
    # The attempt outlives the launch it belongs to: a reader asking how long the backend has failed to
    # answer needs when the worker last tried, which a row cleared between attempts cannot say.
    assert row.text_backend.launching_since is not None

    await supervisor.stop()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_each_launch_describes_once_and_reads_only_its_own_output(tmp_path: Path) -> None:
    """A relaunch re-asks the backend and parses the second launch's lines, not the log's whole history."""
    first = FakeLaunched(pid=101)
    second = FakeLaunched(pid=102)
    relaunched_log = BACKEND_LOAD_LOG.replace("1918.35", "918.35").replace("476.00", "376.00")
    backend = StubBackend(ready_results=[True, True])
    supervisor = TextBackendSupervisor(
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
        backend=backend,
        launch_process=LoggingLauncher([first, second], [BACKEND_LOAD_LOG, relaunched_log]),
        timings=FAST,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: supervisor.is_serving)
    assert backend.describe_calls == 1
    first.exit(3)
    await wait_until(lambda: supervisor.launch_count == 2 and supervisor.is_serving)
    row = supervisor.to_process_snapshot()

    assert backend.describe_calls == 2
    assert row.os_pid == 102
    detail = row.text_backend
    assert detail is not None
    assert detail.launch_count == 2
    assert (detail.model_buffer_mb, detail.kv_buffer_mb) == (1227, 376)

    await supervisor.stop()
    await asyncio.wait_for(task, timeout=5.0)


async def test_a_backend_that_will_not_describe_itself_still_serves(tmp_path: Path) -> None:
    """A refused description leaves the row's model fields unset without holding up the readiness gate."""
    supervisor = TextBackendSupervisor(
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
        backend=StubBackend(ready_results=[True], describe_error=TextBackendUnavailable("no answer")),
        launch_process=Launcher([FakeLaunched(pid=111)]),
        timings=FAST,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: supervisor.is_serving)
    row = supervisor.to_process_snapshot()

    assert row.display_state == "ready"
    assert row.loaded_horde_model_name is None
    assert row.text_backend is not None
    assert row.text_backend.model_name is None
    assert row.text_backend.context_length is None

    await supervisor.stop()
    await asyncio.wait_for(task, timeout=5.0)


async def test_a_backend_on_no_card_reports_no_device_index(tmp_path: Path) -> None:
    """A CPU-only backend's row leaves the card unset instead of being charged to card 0."""
    supervisor = TextBackendSupervisor(
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path, device_index=None)),
        backend=StubBackend(ready_results=[True]),
        launch_process=Launcher([FakeLaunched(pid=121)]),
        timings=FAST,
    )
    task = asyncio.create_task(supervisor.run())

    await wait_until(lambda: supervisor.is_serving)
    row = supervisor.to_process_snapshot()

    assert row.device_index is None
    assert row.text_backend is not None
    assert row.text_backend.footprint_mb is None

    await supervisor.stop()
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.slow
async def test_real_koboldcpp_serves_and_leaves_no_process_behind(tmp_path: Path) -> None:
    """The real koboldcpp binary reaches ready through the supervisor, and stop() ends its whole process tree.

    koboldcpp's launched PID is a PyInstaller bootloader whose child is the real server; this row proves
    the descendant is resolved, recorded, and gone after stop, and that the port is released. Any backend
    that spawns its server as a child gets the same handling; koboldcpp is the one provisioned here.
    """
    executable = koboldcpp_executable()
    model_env = os.environ.get(REAL_MODEL_ENV_VAR)
    if executable is None or model_env is None or not Path(model_env).is_file():
        pytest.skip(f"needs bin/koboldcpp and {REAL_MODEL_ENV_VAR} pointing at a GGUF")

    port = free_port()
    spec = build_launch_spec(
        TEXT_BACKENDS.koboldcpp,
        TextBackendLaunchSettings(
            executable=executable,
            model_path=Path(model_env),
            port=port,
            device_index=0,
            gpu_layers=99,
            context_length=2048,
            log_path=tmp_path / "text_backend.log",
        ),
    )
    registry = OwnedProcessRegistry(tmp_path / "owned.json")
    async with aiohttp.ClientSession() as session:
        backend = KoboldApiTextBackend(spec.base_url, session)
        supervisor = TextBackendSupervisor(
            launch_spec_factory=factory_for(spec),
            backend=backend,
            owned_registry=registry,
            launch_process=default_launch_process,
        )
        task = asyncio.create_task(supervisor.run())
        try:
            await wait_until(lambda: supervisor.is_serving, timeout=300.0)
            tree = supervisor.pids
            assert len(tree) >= 2, f"expected the launched process and its server child, saw {tree}"
            assert sorted(record.os_pid for record in registry._load()) == sorted(tree)
            description = await backend.describe()
            assert description.max_context_length == 2048
        finally:
            await supervisor.stop()
            await asyncio.wait_for(task, timeout=60.0)
            await backend.close()

    assert registry._load() == []
    assert [pid for pid in tree if psutil.pid_exists(pid)] == []
    assert port_is_free(port) is True


async def test_a_row_before_any_launch_has_no_launch_instant(tmp_path: Path) -> None:
    """A supervisor that has not run has attempted nothing, and a clock started here would time a download.

    The instant is what the dashboard's readiness patience is measured from, so its absence is how the
    dashboard knows the worker is still obtaining the backend rather than failing to start it.
    """
    supervisor = TextBackendSupervisor(
        launch_spec_factory=factory_for(spec_on(free_port(), tmp_path)),
        backend=StubBackend(ready_results=[True]),
        launch_process=Launcher([FakeLaunched(pid=1)]),
        timings=FAST,
    )

    row = supervisor.to_process_snapshot()

    assert row.display_state == "stopped"
    assert row.text_backend is not None
    assert row.text_backend.launching_since is None


async def test_the_row_reads_provisioning_while_the_program_is_being_obtained(tmp_path: Path) -> None:
    """A first run downloads the program, and that minute is the backend's life rather than a gap in it.

    The row exists from construction, so the dashboard can say what the worker is doing; before the spec
    is rendered there is no port to name and no card the backend was told to use.
    """
    obtaining = asyncio.Event()
    spec = spec_on(free_port(), tmp_path)

    def obtain_slowly() -> TextBackendLaunchSpec:
        obtaining.set()
        while not release_spec.is_set():
            time.sleep(0.01)
        return spec

    release_spec = asyncio.Event()
    supervisor = TextBackendSupervisor(
        launch_spec_factory=obtain_slowly,
        backend=StubBackend(ready_results=[True]),
        launch_process=Launcher([FakeLaunched(pid=61)]),
        timings=FAST,
    )
    task = asyncio.create_task(supervisor.run())
    try:
        await asyncio.wait_for(obtaining.wait(), timeout=5.0)
        row = supervisor.to_process_snapshot()

        assert supervisor.state is TextBackendState.PROVISIONING
        assert row.display_state == "provisioning"
        assert row.last_process_state == "PROVISIONING"
        assert row.is_alive is False, "no process exists yet, and the row must not claim one"
        assert row.text_backend is not None
        assert row.text_backend.port is None
        assert row.text_backend.launching_since is None
        assert row.device_index is None
    finally:
        release_spec.set()

    await wait_until(lambda: supervisor.is_serving)
    ready_row = supervisor.to_process_snapshot()
    assert ready_row.text_backend is not None
    assert ready_row.text_backend.port == spec.port

    await supervisor.stop()
    await asyncio.wait_for(task, timeout=5.0)


async def test_a_program_that_cannot_be_obtained_is_reported_on_the_row_and_stops(tmp_path: Path) -> None:
    """Obtaining is not retried, so the reason has to reach the row: no clock would ever run out."""

    def cannot_obtain() -> TextBackendLaunchSpec:
        raise TextBackendProvisionError("the release download failed")

    supervisor = TextBackendSupervisor(
        launch_spec_factory=cannot_obtain,
        backend=StubBackend(ready_results=[True]),
        launch_process=Launcher([]),
        timings=FAST,
    )

    await asyncio.wait_for(supervisor.run(), timeout=5.0)

    row = supervisor.to_process_snapshot()
    assert supervisor.state is TextBackendState.STOPPED
    assert supervisor.launch_count == 0, "nothing is launched for a backend there is no program for"
    assert row.display_state == "not obtained"
    assert row.text_backend is not None
    assert row.text_backend.provision_error is not None
    assert "the release download failed" in row.text_backend.provision_error


async def test_an_unrenderable_backend_is_reported_the_same_way(tmp_path: Path) -> None:
    """The other failure the factory can raise is a backend the worker has no command line for."""

    def cannot_render() -> TextBackendLaunchSpec:
        raise UnsupportedTextBackendError("sonar")

    supervisor = TextBackendSupervisor(
        launch_spec_factory=cannot_render,
        backend=StubBackend(ready_results=[True]),
        launch_process=Launcher([]),
        timings=FAST,
    )

    await asyncio.wait_for(supervisor.run(), timeout=5.0)

    row = supervisor.to_process_snapshot()
    assert row.text_backend is not None
    assert row.text_backend.provision_error is not None
    assert "sonar" in row.text_backend.provision_error
