"""The streamed driver against a real koboldcpp, which is the only place its framing is really proved.

Everything else in this directory drives the driver against an application built from a reading of
koboldcpp's source. That reading can be wrong in ways no amount of fixture agreement would expose: a
header, the exact server-sent-events framing, how the abort ends a stream in flight. This row launches
the provisioned binary on a real model and generates through it both ways, so the two answers can be
compared and the progress the worker reports can be seen to have arrived.

Opt-in twice over: it is `slow`, and it skips unless `bin/koboldcpp` is provisioned and
`HORDE_TEXT_SMOKE_GGUF` points at a model file.
"""

from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path
from typing import Final

import aiohttp
import pytest
from horde_model_reference.text_backend_names import TEXT_BACKENDS

from horde_worker_regen.process_management.lifecycle.owned_process_registry import OwnedProcessRegistry
from horde_worker_regen.process_management.lifecycle.text_backend_supervisor import (
    TextBackendSupervisor,
    default_launch_process,
)
from horde_worker_regen.text_backends import (
    KoboldApiTextBackend,
    TextBackendLaunchSettings,
    TextGenerationProgress,
    build_launch_spec,
)
from worker_bootstrap.koboldcpp_bin import koboldcpp_executable

REAL_MODEL_ENV_VAR: Final = "HORDE_TEXT_SMOKE_GGUF"
"""Where the operator says their GGUF is. Without it there is no model to load and the row skips."""

_SMOKE_PROMPT: Final = "Write one short sentence about the sea."
_SMOKE_MAX_LENGTH: Final = 32
_SMOKE_CONTEXT_LENGTH: Final = 2048
_READY_TIMEOUT_SECONDS: Final = 300.0
"""A cold koboldcpp unpacks itself and then loads weights, which runs to minutes on a first start."""
_GENERATION_DEADLINE_SECONDS: Final = 120.0


def _free_port() -> int:
    """Return a loopback port nothing is listening on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def _wait_until_serving(supervisor: TextBackendSupervisor) -> None:
    """Yield to the loop until the supervisor says the backend is serving, or fail the row."""
    deadline = asyncio.get_running_loop().time() + _READY_TIMEOUT_SECONDS
    while not supervisor.is_serving:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"the backend did not start serving within {_READY_TIMEOUT_SECONDS:.0f}s")
        await asyncio.sleep(0.2)


@pytest.mark.slow
async def test_real_koboldcpp_streams_a_generation_and_reports_it_arriving(tmp_path: Path) -> None:
    """A real streamed generation arrives in pieces and answers the same shape the blocking route does.

    The two routes are the same request to the same model, so their answers cannot be compared word for
    word (sampling is not deterministic across them), but both must produce text, and the streamed one
    must have reported itself arriving before it finished, which is the whole reason for driving it.
    """
    executable = koboldcpp_executable()
    model_path = os.environ.get(REAL_MODEL_ENV_VAR)
    if executable is None or model_path is None or not Path(model_path).is_file():
        pytest.skip(f"needs bin/koboldcpp and {REAL_MODEL_ENV_VAR} pointing at a GGUF")

    launch_spec = build_launch_spec(
        TEXT_BACKENDS.koboldcpp,
        TextBackendLaunchSettings(
            executable=executable,
            model_path=Path(model_path),
            port=_free_port(),
            device_index=0,
            gpu_layers=99,
            context_length=_SMOKE_CONTEXT_LENGTH,
            log_path=tmp_path / "text_backend.log",
        ),
    )
    payload: dict[str, object] = {"prompt": _SMOKE_PROMPT, "max_length": _SMOKE_MAX_LENGTH}
    reports: list[TextGenerationProgress] = []

    async with aiohttp.ClientSession() as session:
        backend = KoboldApiTextBackend(launch_spec.base_url, session)
        supervisor = TextBackendSupervisor(
            launch_spec=launch_spec,
            backend=backend,
            owned_registry=OwnedProcessRegistry(tmp_path / "owned.json"),
            launch_process=default_launch_process,
        )
        supervising = asyncio.create_task(supervisor.run())
        try:
            await _wait_until_serving(supervisor)
            capabilities = await backend.capabilities()

            streamed = await backend.generate(
                payload,
                generation_key="smoke-streamed",
                deadline_seconds=_GENERATION_DEADLINE_SECONDS,
                on_progress=reports.append,
            )

            blocking_backend = KoboldApiTextBackend(launch_spec.base_url, session)
            # No public switch chooses the route: a driver takes the stream whenever the backend has
            # one. Latching the flag is how this row asks the same backend the same question the other
            # way, which is what makes the two answers comparable.
            blocking_backend._stream_route_missing = True
            blocking = await blocking_backend.generate(
                payload,
                generation_key="smoke-blocking",
                deadline_seconds=_GENERATION_DEADLINE_SECONDS,
            )
            await blocking_backend.close()
        finally:
            await supervisor.stop()
            await asyncio.wait_for(supervising, timeout=60.0)
            await backend.close()

    assert streamed.text.strip() != "", "a streamed generation produced no text at all"
    assert blocking.text.strip() != "", "the blocking route produced no text at all"
    assert streamed.finish_reason is not None, "koboldcpp names a finish reason on its last stream record"

    assert reports, "the generation finished without ever reporting that it was arriving"
    assert [report.chunks_received for report in reports] == sorted(report.chunks_received for report in reports)
    assert reports[-1].chunks_received >= 1
    assert reports[-1].characters_received == len(streamed.text)

    if capabilities.generation_stats:
        assert streamed.generated_tokens is not None, "a backend that counts tokens must report them"
        assert any(report.tokens_per_second is not None for report in reports)
    else:
        assert all(report.completion_tokens is None for report in reports)
