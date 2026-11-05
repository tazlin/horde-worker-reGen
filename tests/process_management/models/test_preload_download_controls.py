"""Exercise the standalone preload entry point without booting inference or fetching remote models."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from horde_worker_regen import model_download_core, reference_helper
from horde_worker_regen.bridge_data import beta_source
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader
from horde_worker_regen.download_models import download_all_models


class _Manager:
    """Presence can be true for corrupt bytes, as in hordelib's existence-only presence probe."""

    def __init__(self, name: str, *, present: bool = False, valid: bool = False) -> None:
        self.model_reference: dict[str, object] = {name: object()}
        self.name = name
        self.present = present
        self.valid = valid
        self.available_models = [name] if present else []
        self.downloads: list[tuple[str, int]] = []
        self.tainted: list[str] = []
        self.connections: int | None = None
        self.reset_adhoc_cache = Mock()
        self.download_default_models = Mock()
        self.wait_for_downloads = Mock()
        self.wait_for_adhoc_reset = Mock()

    def configure_downloads(self, *, connections: int) -> None:
        self.connections = connections

    def is_model_available(self, _name: str) -> bool:
        return self.present

    def validate_model(self, _name: str, skip_checksum: bool = False) -> bool:
        return self.valid

    def taint_models(self, names: list[str]) -> None:
        self.tainted.extend(names)
        self.present = False

    def download_model(
        self,
        name: str,
        *,
        callback: Callable[[int, int], None] | None = None,
        connections: int = 1,
    ) -> bool:
        # A manager short-circuits an existing file unless the caller taints it first.
        if self.present:
            return True
        self.downloads.append((name, connections))
        self.present = self.valid = True
        assert callback is not None
        callback(1024, 1024)
        return True


def test_preload_applies_config_and_repairs_present_corrupt_aux(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI applies one connection count and cap to every ordinary category and repairs a corrupt aux file."""
    managers = {
        key: _Manager(key, present=key == "gfpgan")
        for key in ("compvis", "controlnet", "miscellaneous", "gfpgan", "esrgan", "codeformer", "lora", "ti")
    }
    config = SimpleNamespace(
        prepare_custom_models=Mock(),
        download_connections_per_file=2,
        download_rate_limit_kbps=8192,
        download_per_host_concurrency=1,
        download_max_parallel_downloads=4,
        allow_lora=True,
        nsfw=True,
        purge_loras_on_download=False,
        allow_controlnet=True,
        allow_sdxl_controlnet=True,
        allow_post_processing=True,
        image_models_to_load=["compvis"],
    )
    monkeypatch.setattr(reference_helper, "ensure_model_reference_manager_initialized", lambda: object())
    monkeypatch.setattr(beta_source, "beta_aware_image_records", lambda _manager: {})
    monkeypatch.setattr(BridgeDataLoader, "load_from_env_vars", lambda **_kwargs: config)
    fake_hordelib = types.ModuleType("hordelib")
    fake_hordelib.initialise = Mock()  # type: ignore[attr-defined]
    fake_api = types.ModuleType("hordelib.api")
    fake_api.SharedModelManager = SimpleNamespace(  # type: ignore[attr-defined]
        manager=SimpleNamespace(**managers),
        load_model_managers=Mock(),
        preload_annotators=lambda: True,
    )
    fake_hordelib.api = fake_api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hordelib", fake_hordelib)
    monkeypatch.setitem(sys.modules, "hordelib.api", fake_api)
    for module_name, function in (
        ("horde_safety.deep_danbooru_model", "download_deep_danbooru_model"),
        ("horde_safety.interrogate", "get_interrogator_no_blip"),
    ):
        module = types.ModuleType(module_name)
        setattr(module, function, lambda: None)
        monkeypatch.setitem(sys.modules, module_name, module)

    observed_rates: list[int | None] = []
    original_step = model_download_core.ChunkPacer.step

    def step(
        self: model_download_core.ChunkPacer,
        *args: object,
        **kwargs: object,
    ) -> model_download_core.ModelProgress:
        rate = kwargs["rate_limit_kbps"]
        assert callable(rate)
        value = rate()
        assert value is None or isinstance(value, int)
        observed_rates.append(value)
        return original_step(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(model_download_core.ChunkPacer, "step", step)
    download_all_models(load_config_from_env_vars=True)

    for key in ("compvis", "controlnet", "miscellaneous", "gfpgan", "esrgan", "codeformer"):
        assert managers[key].downloads == [(key, 2)]
    assert managers["gfpgan"].tainted == ["gfpgan"]
    assert managers["gfpgan"].valid
    assert managers["lora"].connections == managers["ti"].connections == 2
    managers["lora"].wait_for_downloads.assert_called_once_with(600)
    assert observed_rates == [8192] * 6
