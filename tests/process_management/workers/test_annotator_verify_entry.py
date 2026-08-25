"""The annotator verify helper is a plain module entry point whose exit status is the verdict."""

from __future__ import annotations

import sys
import types

import pytest

from horde_worker_regen.process_management.workers import annotator_verify


def _install_fake_hordelib(monkeypatch: pytest.MonkeyPatch, *, verdict: bool) -> list[list[str] | None]:
    """Stand in for hordelib and its SharedModelManager; record the extra ComfyUI args initialise saw."""
    seen: list[list[str] | None] = []
    fake_hordelib = types.ModuleType("hordelib")

    def _initialise(*, setup_logging: bool, extra_comfyui_args: list[str] | None) -> None:
        assert setup_logging is False
        seen.append(extra_comfyui_args)

    fake_hordelib.initialise = _initialise  # type: ignore[attr-defined]
    fake_api = types.ModuleType("hordelib.api")
    fake_api.SharedModelManager = types.SimpleNamespace(preload_annotators=lambda: verdict)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hordelib", fake_hordelib)
    monkeypatch.setitem(sys.modules, "hordelib.api", fake_api)
    return seen


def test_exit_status_follows_the_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passing preload exits 0 and a failing one exits 1."""
    _install_fake_hordelib(monkeypatch, verdict=True)
    assert annotator_verify.main([]) == 0
    _install_fake_hordelib(monkeypatch, verdict=False)
    assert annotator_verify.main([]) == 1


def test_directml_device_is_forwarded_as_a_comfyui_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DirectML index becomes the ComfyUI flag the inference processes also pass."""
    seen = _install_fake_hordelib(monkeypatch, verdict=True)

    assert annotator_verify.main(["--directml", "1"]) == 0

    assert seen == [["--directml=1"]]
