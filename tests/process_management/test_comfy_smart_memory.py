"""Tests for the ComfyUI memory-mode flag seeded onto ComfyUI-running children.

``--disable-smart-memory`` unloads every model after each prompt below anything the worker can suppress, so
a child carrying it can never honour a retention grant. Inference children launch without it: eviction is
hordelib's explicit end-of-job free, which the scheduler's per-dispatch grant suppresses. Lane and safety
children take no grants and always carry the flag. ``legacy_comfy_vram_unload`` restores the flag on
inference children; ``comfy_smart_memory`` is deprecated and inert.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen.bridge_data import data_model as data_model_module
from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.worker_entry_points import (
    _seed_extra_comfyui_args,
    start_component_process,
    start_post_process_process,
    start_safety_process,
    start_vae_lane_process,
)

_DISABLE_SMART_MEMORY = "--disable-smart-memory"


@contextmanager
def _captured_logs() -> Iterator[list[str]]:
    """Capture loguru messages emitted within the block into a list of formatted strings."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="DEBUG")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


@pytest.fixture
def _reset_deprecation_guard() -> Iterator[None]:
    """Clear the one-time deprecation-warning guard around a test so the warn-once behaviour is observable."""
    data_model_module._warned_deprecated_config_keys.clear()
    try:
        yield
    finally:
        data_model_module._warned_deprecated_config_keys.clear()


def test_bridge_data_defaults_to_retention_regime() -> None:
    """The default config leaves inference children in the retention regime (no legacy unload flag)."""
    assert reGenBridgeData.model_fields["legacy_comfy_vram_unload"].default is False


def test_seed_omits_disable_flag_by_default() -> None:
    """Without the legacy regime the child carries no memory-mode flag, so a retention grant can hold."""
    assert _seed_extra_comfyui_args(disable_smart_memory=False) == []


def test_seed_emits_disable_flag_when_requested() -> None:
    """The legacy regime is exactly the ComfyUI flag."""
    assert _seed_extra_comfyui_args(disable_smart_memory=True) == [_DISABLE_SMART_MEMORY]


def _plm_with_legacy_unload(value: bool) -> Mock:
    """Spawn one inference child with ``legacy_comfy_vram_unload`` pinned to ``value``; return the fake context."""
    from tests.process_management.lifecycle.test_process_lifecycle import _make_plm

    fake_ctx = Mock()
    fake_ctx.get_start_method.return_value = "spawn"
    fake_ctx.Pipe.return_value = (Mock(), Mock())
    fake_ctx.Process.return_value.pid = 4321

    plm = _make_plm(ctx=fake_ctx)
    plm._runtime_config.bridge_data.legacy_comfy_vram_unload = value
    plm._start_inference_process(0)
    return fake_ctx


def test_inference_spawn_threads_legacy_unload_off() -> None:
    """The inference spawn forwards the default (retention regime) to the child entry point."""
    fake_ctx = _plm_with_legacy_unload(False)
    spawn_kwargs = fake_ctx.Process.call_args.kwargs["kwargs"]
    assert spawn_kwargs["legacy_comfy_vram_unload"] is False


def test_inference_spawn_threads_legacy_unload_on() -> None:
    """The escape hatch is honored end to end into the child entry point kwargs."""
    fake_ctx = _plm_with_legacy_unload(True)
    spawn_kwargs = fake_ctx.Process.call_args.kwargs["kwargs"]
    assert spawn_kwargs["legacy_comfy_vram_unload"] is True


@pytest.mark.parametrize(
    "entry_point",
    [start_safety_process, start_post_process_process, start_vae_lane_process, start_component_process],
)
def test_lane_and_safety_entry_points_take_no_memory_mode_knob(entry_point: object) -> None:
    """No config value can lift the flag off a lane or safety child: the entry points accept no such argument."""
    parameters = inspect.signature(entry_point).parameters  # type: ignore[arg-type]
    assert "legacy_comfy_vram_unload" not in parameters
    assert "comfy_smart_memory" not in parameters


def test_safety_spawn_passes_no_memory_mode_knob() -> None:
    """The safety spawn hands its child nothing that could suppress the flag."""
    from tests.process_management.lifecycle.test_process_lifecycle import _make_plm

    fake_ctx = Mock()
    fake_ctx.get_start_method.return_value = "spawn"
    fake_ctx.Pipe.return_value = (Mock(), Mock())
    fake_ctx.Process.return_value.pid = 4322

    plm = _make_plm(ctx=fake_ctx)
    plm.start_safety_processes()

    spawn_kwargs = fake_ctx.Process.call_args.kwargs["kwargs"]
    assert "legacy_comfy_vram_unload" not in spawn_kwargs
    assert "comfy_smart_memory" not in spawn_kwargs


@pytest.mark.usefixtures("_reset_deprecation_guard")
@pytest.mark.parametrize("configured_value", [True, False])
def test_deprecated_comfy_smart_memory_warns_once_and_is_inert(configured_value: bool) -> None:
    """Setting the retired key loads fine, warns exactly once across two loads, and changes no behaviour."""
    with _captured_logs() as messages:
        first = reGenBridgeData.model_validate({"comfy_smart_memory": configured_value})
        reGenBridgeData.model_validate({"comfy_smart_memory": configured_value})

    pointers = [message for message in messages if "`comfy_smart_memory` is deprecated" in message]
    assert len(pointers) == 1
    assert first.legacy_comfy_vram_unload is False


@pytest.mark.usefixtures("_reset_deprecation_guard")
def test_unset_comfy_smart_memory_is_silent() -> None:
    """A config that never mentions the retired key draws no pointer."""
    with _captured_logs() as messages:
        reGenBridgeData.model_validate({})

    assert not [message for message in messages if "`comfy_smart_memory` is deprecated" in message]
