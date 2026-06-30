"""The benchmark ``download`` subcommand fetches checkpoints for real and reports structured progress.

Proves the benchmark download path (now on the shared core) actually downloads: it drives
``_download_compvis_models`` against a real loopback server through a fake, real-downloading compvis, and
checks the emitted event sequence. Also covers the stdin control channel and the launcher flag.
"""

from __future__ import annotations

import io
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)
from horde_model_reference.on_disk_layout import is_present

from horde_worker_regen.benchmark import cli
from horde_worker_regen.benchmark.download_progress import (
    DownloadControl,
    DownloadEvent,
    decode_download_control,
    encode_download_control,
)
from tests.download_test_helpers import FakeModelServer, RealDownloadCompVis, deterministic_bytes

_MODEL = "Z-Image-Turbo"
_FILES: tuple[tuple[str, str, int], ...] = (
    ("z_image_turbo_bf16.safetensors", "unet", 4096),
    ("ae.safetensors", "vae", 1024),
    ("qwen_3_4b.safetensors", "text_encoders", 2048),
)


def _record(base_url: str) -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        name=_MODEL,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo,
        nsfw=True,
        description="benchmark download record",
        config=GenericModelRecordConfig(
            download=[
                DownloadRecord(file_name=name, file_url=f"{base_url}/{name}", file_purpose=purpose)
                for name, purpose, _size in _FILES
            ],
        ),
    )


def _inject_fake_hordelib(monkeypatch: pytest.MonkeyPatch, compvis: RealDownloadCompVis) -> None:
    """Make ``from hordelib.api import SharedModelManager`` resolve to a fake backed by *compvis*."""
    fake_api = types.ModuleType("hordelib.api")
    fake_api.SharedModelManager = SimpleNamespace(  # type: ignore[attr-defined]
        load_model_managers=lambda *args, **kwargs: None,
        manager=SimpleNamespace(compvis=compvis),
    )
    hordelib_stub = sys.modules.get("hordelib") or types.ModuleType("hordelib")
    hordelib_stub.api = fake_api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hordelib", hordelib_stub)
    monkeypatch.setitem(sys.modules, "hordelib.api", fake_api)


def test_download_subcommand_fetches_and_emits_event_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_download_compvis_models downloads the missing checkpoint and emits started -> progress -> finished."""
    server = FakeModelServer()
    for name, _purpose, size in _FILES:
        server.add(name, deterministic_bytes(name, size))
    server.start()
    try:
        compvis = RealDownloadCompVis(tmp_path, {_MODEL: _record(server.base_url)})
        _inject_fake_hordelib(monkeypatch, compvis)

        events: list[DownloadEvent] = []
        failed = cli._download_compvis_models([_MODEL], emit=events.append, json_progress=True)

        assert failed == 0
        assert is_present(_record(server.base_url), tmp_path) is True

        kinds = [event.kind for event in events]
        assert kinds[0] == "model_started"
        assert "model_progress" in kinds
        assert kinds[-1] == "model_finished"
        assert events[-1].ok is True
        # Progress events carry live byte counts for the model.
        progress = [event for event in events if event.kind == "model_progress"]
        assert progress and progress[-1].total_bytes > 0
    finally:
        server.stop()


def test_download_control_codec_round_trips() -> None:
    """Control commands survive the stdin JSON-line encode/decode used by the modal <-> subprocess link."""
    controls = (DownloadControl(cmd="pause"), DownloadControl(cmd="resume"), DownloadControl(cmd="rate", kbps=256))
    for control in controls:
        assert decode_download_control(encode_download_control(control)) == control
    assert decode_download_control("") is None
    assert decode_download_control("not json") is None


def test_stdin_control_thread_applies_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stdin control reader folds pause/rate lines into the DownloadControls the core reads."""
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"cmd":"pause"}\n{"cmd":"rate","kbps":256}\n'))
    controls = cli._start_stdin_control_thread()

    deadline = time.time() + 2.0
    while time.time() < deadline and controls.rate_limit_kbps() is None:
        time.sleep(0.01)

    assert controls.is_paused() is True
    assert controls.rate_limit_kbps() == 256


def test_build_download_command_carries_control_stdin_flag() -> None:
    """The launcher only asks the subprocess to read stdin control when explicitly requested."""
    from horde_worker_regen.tui.benchmark_launcher import BenchmarkOptions

    options = BenchmarkOptions(tiers=["sd15"])
    assert "--control-stdin" not in options.build_download_command()
    assert "--control-stdin" in options.build_download_command(control_stdin=True)


def test_download_path_skips_the_gpu_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The download subcommand must build its ladder without probing the GPU.

    The out-of-process torch/CUDA probe is a cold, multi-minute startup that the download path never uses
    (it discards the machine info), and on a cold .exe install it was the dominant cost behind the
    "Could not work out the download plan: timed out after 240 seconds" preview failure.
    """
    captured: dict[str, object] = {}

    def fake_prepare(args: object, tiers: object, *, probe_devices: bool = True) -> tuple[list, object, object]:
        captured["probe_devices"] = probe_devices
        return [], SimpleNamespace(total_vram_mb=None), SimpleNamespace()

    monkeypatch.setattr(cli, "_prepare_catalog", fake_prepare)
    args = SimpleNamespace(
        tiers="sd15",
        dry_run=True,
        json_progress=True,
        control_stdin=False,
        process_mode="real",
    )

    rc = cli._run_download(args)  # type: ignore[arg-type]

    assert rc == 0
    assert captured["probe_devices"] is False
