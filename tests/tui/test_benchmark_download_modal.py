"""Pilot tests for the benchmark download modal's live controls and progress rendering.

Drives the real :class:`BenchmarkDownloadModal` under Textual's ``run_test``: the plan subprocess is
faked so no real benchmark runs, and the modal's Pause/Resume + rate-limit buttons are asserted to write
the right control lines to the (captured) subprocess stdin, while a ``model_progress`` event is shown as a
live progress bar.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input

from horde_worker_regen.benchmark.download_progress import (
    DownloadControl,
    DownloadEvent,
    DownloadModelRow,
    decode_download_control,
    encode_download_event,
)
from horde_worker_regen.tui.benchmark_launcher import BenchmarkOptions
from horde_worker_regen.tui.widgets.benchmark_download import BenchmarkDownloadModal


class _FakeStdin:
    """Captures the control lines the modal writes to the subprocess."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, data: str) -> None:
        self.lines.append(data)

    def flush(self) -> None:
        """No-op; the modal flushes after each control write."""


class _FakePopen:
    """A stand-in for the download subprocess that only needs a capturing stdin."""

    def __init__(self) -> None:
        self.stdin = _FakeStdin()


class _ModalHost(App[None]):
    """Pushes the download modal as a screen so Pilot can drive its buttons."""

    def __init__(self, options: BenchmarkOptions) -> None:
        super().__init__()
        self.modal = BenchmarkDownloadModal(options)

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self.modal)


@pytest.fixture(autouse=True)
def _fake_plan_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the modal's on-mount dry-run from spawning a real benchmark; return a one-model plan."""
    planned = encode_download_event(
        DownloadEvent(
            kind="planned",
            models=[DownloadModelRow(name="Deliberate", size_bytes=100, on_disk=False, target_path="x")],
            to_download_bytes=100,
            free_disk_bytes=10_000_000,
            fits=True,
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=planned, stderr="", returncode=0),
    )


@pytest.mark.e2e
async def test_modal_controls_write_control_lines_to_stdin() -> None:
    """Pause/Resume and Apply-limit write the matching control commands to the subprocess stdin."""
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        # Simulate an in-flight download: a live process with capturing stdin and the controls shown.
        fake = _FakePopen()
        modal._process = fake  # type: ignore[assignment]
        modal.query_one("#download-controls").display = True
        await pilot.pause()

        await pilot.click("#download-pause")
        await pilot.pause()

        modal.query_one("#download-rate", Input).value = "500"
        await pilot.click("#download-rate-apply")
        await pilot.pause()

        # Pause (a no-arg command) and Apply-limit (a parameterized one) both reach the subprocess stdin.
        commands = [decode_download_control(line) for line in fake.stdin.lines]
        assert DownloadControl(cmd="pause") in commands
        assert DownloadControl(cmd="rate", kbps=500) in commands

        # Resume maps correctly off the toggled state (same _write_control path as Pause, flag flipped).
        fake.stdin.lines.clear()
        modal._toggle_pause()
        await pilot.pause()
        assert decode_download_control(fake.stdin.lines[-1]) == DownloadControl(cmd="resume")


@pytest.mark.e2e
async def test_modal_renders_live_progress_bar() -> None:
    """A model_progress event is folded into a live progress line for the current model."""
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal

        modal._apply_event(DownloadEvent(kind="model_started", name="Deliberate", index=1, total=1))
        modal._apply_event(
            DownloadEvent(
                kind="model_progress",
                name="Deliberate",
                index=1,
                total=1,
                downloaded_bytes=512,
                total_bytes=1024,
                speed_bps=1024.0,
                eta_seconds=0.5,
            ),
        )
        await pilot.pause()

        assert modal._current_progress is not None
        line = modal._progress_line(modal._current_progress)
        assert "Deliberate" in line.plain
        assert "50.0%" in line.plain  # 512 / 1024
