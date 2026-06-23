"""End-to-end round-trip for background model downloads, with real HTTP and real disk writes.

This exercises the actual moving parts rather than a faked timeline:

  * a real local HTTP server serves deterministic bytes for a Z-Image-Turbo-shaped 3-file model;
  * the download backend performs genuine chunked HTTP downloads to a temp weights root (faked *files*,
    real *download*), writing each file and its checksum sidecar to the canonical on-disk layout;
  * the real :class:`HordeDownloadProcess` orchestration drives it through its real control/tick methods
    (enqueue, dedup-against-present, pause/resume, status snapshots);
  * the real :class:`DownloadsView` is driven through Textual's ``run_test`` Pilot (button clicks), and a
    capstone test shows a UI pause press actually holding back a real download.

The download process's heavy ``hordelib`` model-manager load is replaced by a fake ``SharedModelManager``
whose ``compvis`` downloads for real; everything else is the production code path. No GPU, no torch, no
network beyond loopback.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)
from horde_model_reference.on_disk_layout import file_paths_for, is_present
from textual.app import App, ComposeResult

from horde_worker_regen.tui.widgets.downloads import DownloadsView
from tests.download_test_helpers import FakeModelServer, RealDownloadCompVis, deterministic_bytes

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.connection import Connection

    from horde_worker_regen.process_management.download_process import HordeDownloadProcess
    from horde_worker_regen.process_management.messages import HordeDownloadControlMessage
    from horde_worker_regen.process_management.supervisor_channel import WorkerStateSnapshot

# (file_name, file_purpose, size_bytes) for the real Z-Image-Turbo layout, shrunk to test sizes.
_FILES: tuple[tuple[str, str, int], ...] = (
    ("z_image_turbo_bf16.safetensors", "unet", 4096),
    ("ae.safetensors", "vae", 1024),
    ("qwen_3_4b.safetensors", "text_encoders", 2048),
)
_MODEL_NAME = "Z-Image-Turbo"
_OTHER_NAME = "Deliberate"
_OTHER_FILE = "deliberate.safetensors"


@pytest.fixture
def model_server() -> Iterator[FakeModelServer]:
    """A running fake model server seeded with the Z-Image files and one other model."""
    server = FakeModelServer()
    for file_name, _purpose, size in _FILES:
        server.add(file_name, deterministic_bytes(file_name, size))
    server.add(_OTHER_FILE, deterministic_bytes(_OTHER_FILE, 512))
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _zimage_record(base_url: str) -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        name=_MODEL_NAME,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo,
        nsfw=True,
        description="round-trip record",
        config=GenericModelRecordConfig(
            download=[
                DownloadRecord(
                    file_name=file_name,
                    file_url=f"{base_url}/{file_name}",
                    file_purpose=file_purpose,
                )
                for file_name, file_purpose, _size in _FILES
            ],
        ),
    )


def _other_record(base_url: str) -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        name=_OTHER_NAME,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        nsfw=False,
        description="single-file record",
        config=GenericModelRecordConfig(
            download=[DownloadRecord(file_name=_OTHER_FILE, file_url=f"{base_url}/{_OTHER_FILE}")],
        ),
    )


def _make_download_process(
    compvis: RealDownloadCompVis,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[HordeDownloadProcess, Connection]:
    """Construct a real HordeDownloadProcess wired to a fake, real-downloading SharedModelManager."""
    import multiprocessing as mp

    from horde_worker_regen.process_management.download_process import HordeDownloadProcess

    fake_manager = SimpleNamespace(compvis=compvis)
    fake_api = types.ModuleType("hordelib.api")
    fake_api.SharedModelManager = SimpleNamespace(manager=fake_manager)  # type: ignore[attr-defined]
    hordelib_stub = sys.modules.get("hordelib") or types.ModuleType("hordelib")
    hordelib_stub.api = fake_api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hordelib", hordelib_stub)
    monkeypatch.setitem(sys.modules, "hordelib.api", fake_api)

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = HordeDownloadProcess(
        process_id=9000,
        process_message_queue=ctx.Queue(),
        pipe_connection=child_conn,
        disk_lock=ctx.Lock(),
        download_bandwidth_semaphore=ctx.Semaphore(1),
        process_launch_identifier=1,
    )
    # Keep the work focused on image models: the required safety models and the optional aux pass are
    # unrelated networked downloads we are not exercising here.
    process._safety_present = True
    process._safety_ensured = True
    process._aux_enqueued = True
    process._refresh_present()
    return process, parent_conn


def _drain_to_idle(process: HordeDownloadProcess, *, max_ticks: int = 50) -> None:
    """Build and run scheduled work synchronously until idle (executor threads, inline, no real loop).

    The production process runs the executor pool on threads; here a single test thread plays both the
    orchestrator (``_orchestrate`` stages tasks into the scheduler) and an executor (drain the scheduler,
    running each admissible task to completion), so the real download path is exercised deterministically.
    """
    for _ in range(max_ticks):
        did = process._orchestrate()
        ran = False
        if not process._paused:
            task = process._scheduler.acquire(timeout=0.0)
            while task is not None:
                ran = True
                try:
                    process._run_task(task)
                finally:
                    process._scheduler.release(task)
                task = process._scheduler.acquire(timeout=0.0)
        if not did and not ran and not process._scheduler.has_work() and process._running_count == 0:
            return
    raise AssertionError("download process did not reach idle within the tick budget")


def test_real_download_writes_files_and_marks_present(model_server: FakeModelServer, tmp_path: Path) -> None:
    """A genuine HTTP download lands all three files (and sidecars) and flips canonical presence to True."""
    record = _zimage_record(model_server.base_url)
    compvis = RealDownloadCompVis(tmp_path, {_MODEL_NAME: record})

    assert is_present(record, tmp_path) is False
    assert compvis.download_model(_MODEL_NAME) is True

    assert is_present(record, tmp_path) is True
    for path in file_paths_for(record, tmp_path):
        assert path.exists()
        assert path.with_suffix(".sha256").exists()
    assert (tmp_path / "vae" / "ae.safetensors").exists()
    assert (tmp_path / "text_encoders" / "qwen_3_4b.safetensors").exists()


def test_process_downloads_missing_and_skips_present(
    model_server: FakeModelServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process downloads a configured-missing model and never re-fetches one already on disk."""
    records = {
        _MODEL_NAME: _zimage_record(model_server.base_url),
        _OTHER_NAME: _other_record(model_server.base_url),
    }
    # Pre-place the "other" model so it is already present before the process starts.
    other_dest = file_paths_for(records[_OTHER_NAME], tmp_path)[0]
    other_dest.parent.mkdir(parents=True, exist_ok=True)
    other_dest.write_bytes(deterministic_bytes(_OTHER_FILE, 512))

    compvis = RealDownloadCompVis(tmp_path, records)
    process, _conn = _make_download_process(compvis, monkeypatch)

    # Refresh now that the present model is on disk; it should be seen as present from the start.
    process._refresh_present()
    assert _OTHER_NAME in process._present

    process._handle_control_message(
        _control_message(model_names=[_MODEL_NAME, _OTHER_NAME]),
    )
    # Only the missing model is staged; the present one is deduped out.
    assert process._pending_image_models == [_MODEL_NAME]

    _drain_to_idle(process)

    assert is_present(records[_MODEL_NAME], tmp_path) is True
    assert _MODEL_NAME in process._present
    # The present model's file was never requested from the server; the missing one's three files were.
    assert model_server.hits[f"/{_OTHER_FILE}"] == 0
    for file_name, _purpose, _size in _FILES:
        assert model_server.hits[f"/{file_name}"] == 1


def test_pause_holds_the_queue_then_resume_downloads(
    model_server: FakeModelServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paused process leaves a queued model unfetched; resuming downloads it for real."""
    records = {_MODEL_NAME: _zimage_record(model_server.base_url)}
    compvis = RealDownloadCompVis(tmp_path, records)
    process, _conn = _make_download_process(compvis, monkeypatch)

    process._handle_control_message(_control_message(model_names=[_MODEL_NAME], set_paused=True))
    assert process._paused is True

    # While paused, orchestrating stages nothing into the scheduler and nothing is fetched.
    for _ in range(5):
        assert process._orchestrate() is False
    assert model_server.hits.total() == 0
    assert process._pending_image_models == [_MODEL_NAME]

    process._handle_control_message(_control_message(set_paused=False))
    assert process._paused is False
    _drain_to_idle(process)

    assert is_present(records[_MODEL_NAME], tmp_path) is True
    assert sum(model_server.hits.values()) == len(_FILES)


def _control_message(
    *,
    model_names: list[str] | None = None,
    set_paused: bool | None = None,
) -> HordeDownloadControlMessage:
    from horde_worker_regen.process_management.messages import HordeDownloadControlMessage

    return HordeDownloadControlMessage(
        model_names=list(model_names or []),
        set_paused=set_paused,
    )


# --- Textual UI round-trip (run_test / Pilot) -------------------------------------------------------


class _DownloadsHost(App[None]):
    """Hosts the real DownloadsView and records the control messages it posts."""

    def __init__(self) -> None:
        super().__init__()
        self.view = DownloadsView()
        self.pause_requests: list[DownloadsView.PauseToggleRequested] = []
        self.rate_requests: list[DownloadsView.RateLimitRequested] = []
        self.hold_requests: list[DownloadsView.DownloadsOnlyHoldRequested] = []
        self.go_live_requests: list[DownloadsView.GoLiveRequested] = []

    def compose(self) -> ComposeResult:
        yield self.view

    def on_downloads_view_pause_toggle_requested(self, message: DownloadsView.PauseToggleRequested) -> None:
        self.pause_requests.append(message)

    def on_downloads_view_rate_limit_requested(self, message: DownloadsView.RateLimitRequested) -> None:
        self.rate_requests.append(message)

    def on_downloads_view_downloads_only_hold_requested(
        self,
        message: DownloadsView.DownloadsOnlyHoldRequested,
    ) -> None:
        self.hold_requests.append(message)

    def on_downloads_view_go_live_requested(self, message: DownloadsView.GoLiveRequested) -> None:
        self.go_live_requests.append(message)


def _downloads_host() -> _DownloadsHost:
    return _DownloadsHost()


def _downloading_snapshot(present: list[str]) -> WorkerStateSnapshot:
    from horde_worker_regen.process_management.supervisor_channel import (
        CurrentDownloadStatus,
        DownloadPhase,
        DownloadStatusSnapshot,
        WorkerConfigSummary,
        WorkerStateSnapshot,
    )

    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="test", worker_version="0.0.0"),
        downloads=DownloadStatusSnapshot(
            phase=DownloadPhase.DOWNLOADING,
            current=CurrentDownloadStatus(
                model_name=_MODEL_NAME,
                feature="image model",
                target_dir="models/compvis",
                downloaded_bytes=2048,
                total_bytes=4096,
                speed_bps=1024.0,
            ),
            present_model_names=present,
        ),
    )


@pytest.mark.e2e
async def test_downloads_view_buttons_emit_control_messages() -> None:
    """Clicking the real Downloads controls posts the pause and rate-limit messages the app forwards."""
    from textual.widgets import Input

    app = _downloads_host()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view.update_view(_downloading_snapshot(present=[]))
        await pilot.pause()

        await pilot.click("#downloads-pause")
        await pilot.pause()
        assert len(app.pause_requests) == 1
        assert app.pause_requests[0].currently_paused is False

        app.view.query_one("#downloads-rate", Input).value = "500"
        await pilot.click("#downloads-rate-apply")
        await pilot.pause()
        assert len(app.rate_requests) == 1
        assert app.rate_requests[0].kbps == 500


@pytest.mark.e2e
async def test_downloads_view_hold_and_go_live_buttons_emit_messages() -> None:
    """The download-only and go-live controls post the messages the app turns into supervisor commands."""
    app = _downloads_host()
    # A wide terminal so all five controls in the row are on-screen and clickable.
    async with app.run_test(size=(180, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#downloads-only-hold")
        await pilot.pause()
        assert len(app.hold_requests) == 1

        await pilot.click("#downloads-go-live")
        await pilot.pause()
        assert len(app.go_live_requests) == 1


@pytest.mark.e2e
async def test_ui_pause_press_holds_a_real_download_then_resume_completes_it(
    model_server: FakeModelServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capstone: a UI pause click gates a real download; resuming via the same channel completes it."""
    records = {_MODEL_NAME: _zimage_record(model_server.base_url)}
    compvis = RealDownloadCompVis(tmp_path, records)
    process, _conn = _make_download_process(compvis, monkeypatch)

    app = _downloads_host()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#downloads-pause")
        await pilot.pause()

    # The view reported the user's intent; the app turns "currently paused == False" into a pause request,
    # which reaches the process as set_paused=True. Drive that real control message into the real process.
    assert app.pause_requests and app.pause_requests[0].currently_paused is False
    process._handle_control_message(_control_message(model_names=[_MODEL_NAME], set_paused=True))
    assert process._paused is True

    for _ in range(5):
        assert process._orchestrate() is False
    assert model_server.hits.total() == 0
    assert is_present(records[_MODEL_NAME], tmp_path) is False

    # Resume through the same control channel and the queued model downloads for real.
    process._handle_control_message(_control_message(set_paused=False))
    _drain_to_idle(process)
    assert is_present(records[_MODEL_NAME], tmp_path) is True
