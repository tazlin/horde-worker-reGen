"""Routing/unit tests for the image-utilities job flow: control-map injection, form routing, prefetch paths."""

from __future__ import annotations

from types import SimpleNamespace

from horde_sdk.generation_parameters.image.object_models import (
    ControlnetGenerationParameters,
    ImageGenerationComponentContainer,
)

from horde_worker_regen.process_management.lifecycle.horde_process import WorkerCapability
from horde_worker_regen.process_management.scheduling.workload_flow import capability_for_alchemy_form
from horde_worker_regen.process_management.workers import rembg_prefetch
from horde_worker_regen.process_management.workers.inference_process import inject_premade_control_map


def _params_with_controlnet() -> SimpleNamespace:
    """Build a stand-in generation-parameters object carrying a real controlnet component."""
    container = ImageGenerationComponentContainer(
        components=[
            ControlnetGenerationParameters(
                controlnet_type="canny",
                source_image=None,
                control_map=None,
            ),
        ],
    )
    return SimpleNamespace(additional_params=container)


def _params_without_controlnet() -> SimpleNamespace:
    return SimpleNamespace(additional_params=ImageGenerationComponentContainer(components=[]))


def test_inject_premade_control_map_sets_control_map_on_controlnet_component() -> None:
    """A pre-annotated control map is assigned onto the job's controlnet parameters."""
    params = _params_with_controlnet()

    applied = inject_premade_control_map(params, b"pre-made-control-map")  # type: ignore[arg-type]

    assert applied is True
    assert params.additional_params.controlnet_params is not None
    assert params.additional_params.controlnet_params.control_map == b"pre-made-control-map"


def test_inject_premade_control_map_no_controlnet_component_is_a_noop() -> None:
    """A job without controlnet parameters is left untouched and the helper reports it did not apply."""
    params = _params_without_controlnet()

    applied = inject_premade_control_map(params, b"pre-made-control-map")  # type: ignore[arg-type]

    assert applied is False
    assert params.additional_params.controlnet_params is None


def test_strip_background_routes_to_image_utilities_capability() -> None:
    """strip_background is served by the image-utilities lane, not the post-processing (graph) lane."""
    assert capability_for_alchemy_form("strip_background") == WorkerCapability.IMAGE_UTILITIES


def test_rembg_cache_dir_derivation(monkeypatch: object) -> None:
    """The rembg cache dir mirrors AIWORKER_CACHE_HOME/horde/image-utilities/rembg, or None when unset."""
    import os
    from pathlib import Path

    monkeypatch.setenv("AIWORKER_CACHE_HOME", os.path.join("X:", "cache"))  # type: ignore[attr-defined]
    cache_dir = rembg_prefetch.rembg_cache_dir()
    assert cache_dir == Path("X:", "cache", "horde", "image-utilities", "rembg")

    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)  # type: ignore[attr-defined]
    assert rembg_prefetch.rembg_cache_dir() is None


def test_u2net_present_false_without_cache_home(monkeypatch: object) -> None:
    """The presence probe is False when there is no cache home, and does not raise."""
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)  # type: ignore[attr-defined]
    assert rembg_prefetch.u2net_present() is False


def test_ensure_u2net_present_returns_none_without_cache_home(monkeypatch: object) -> None:
    """The pre-place is a no-op returning None when there is no isolated cache to populate."""
    monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)  # type: ignore[attr-defined]
    assert rembg_prefetch.ensure_u2net_present() is None
