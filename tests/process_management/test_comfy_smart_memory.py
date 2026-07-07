"""Tests for the ComfyUI smart-memory policy plumbed to inference-serving children.

Per-job offloading (``--disable-smart-memory``) is the default: cross-process residency is not
reconciled at dispatch time, so on tight cards a sampling peak beside an idle sibling's resident weights
overcommits the device faster than the reclaim ladder can evict. The ``comfy_smart_memory`` bridge field
opts in to cross-job VRAM residency for cards with headroom to spare.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.worker_entry_points import _seed_extra_comfyui_args


def test_bridge_data_defaults_smart_memory_off() -> None:
    """The default keeps per-job offloading until dispatch-time residency reconciliation exists."""
    assert reGenBridgeData.model_fields["comfy_smart_memory"].default is False


def test_seed_omits_disable_flag_when_smart_memory_on() -> None:
    """Smart memory on must not carry ``--disable-smart-memory`` (which offloads after every job)."""
    assert _seed_extra_comfyui_args(comfy_smart_memory=True) == []


def test_seed_restores_disable_flag_when_smart_memory_off() -> None:
    """The escape hatch (off) restores the aggressive-offload flag."""
    assert _seed_extra_comfyui_args(comfy_smart_memory=False) == ["--disable-smart-memory"]


def _plm_with_smart_memory(value: bool) -> Mock:
    """Spawn one inference child with ``comfy_smart_memory`` pinned to ``value``; return the fake context."""
    from tests.process_management.lifecycle.test_process_lifecycle import _make_plm

    fake_ctx = Mock()
    fake_ctx.get_start_method.return_value = "spawn"
    fake_ctx.Pipe.return_value = (Mock(), Mock())
    fake_ctx.Process.return_value.pid = 4321

    plm = _make_plm(ctx=fake_ctx)
    plm._runtime_config.bridge_data.comfy_smart_memory = value
    plm._start_inference_process(0)
    return fake_ctx


def test_inference_spawn_threads_smart_memory_true() -> None:
    """The inference spawn forwards the bridge field (on) to the child entry point."""
    fake_ctx = _plm_with_smart_memory(True)
    spawn_kwargs = fake_ctx.Process.call_args.kwargs["kwargs"]
    assert spawn_kwargs["comfy_smart_memory"] is True


def test_inference_spawn_threads_smart_memory_false() -> None:
    """The escape hatch (off) is honored end to end into the child entry point kwargs."""
    fake_ctx = _plm_with_smart_memory(False)
    spawn_kwargs = fake_ctx.Process.call_args.kwargs["kwargs"]
    assert spawn_kwargs["comfy_smart_memory"] is False
