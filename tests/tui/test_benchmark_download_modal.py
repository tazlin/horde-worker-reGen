"""Pilot tests for the benchmark download modal's live controls and progress rendering.

Drives the real :class:`BenchmarkDownloadModal` under Textual's ``run_test``: the plan subprocess is
faked so no real benchmark runs, and the modal's Pause/Resume + rate-limit buttons are asserted to write
the right control lines to the (captured) subprocess stdin, while a ``model_progress`` event is shown as a
live progress bar.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input

from horde_worker_regen.benchmark.download_progress import (
    DownloadControl,
    DownloadEvent,
    DownloadModelRow,
    decode_download_control,
    encode_download_event,
)
from horde_worker_regen.tui.benchmark_launcher import BenchmarkOptions
from horde_worker_regen.tui.widgets.benchmark_download import BenchmarkDownloadModal, DownloadLiveState


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

    def __init__(
        self,
        options: BenchmarkOptions,
        *,
        delegate: Callable[[list[str]], bool] | None = None,
        live_state: Callable[[], DownloadLiveState | None] | None = None,
    ) -> None:
        super().__init__()
        self.modal = BenchmarkDownloadModal(options, delegate=delegate, live_state=live_state)

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


@pytest.mark.e2e
async def test_modal_delegates_missing_models_to_the_worker_when_a_delegate_is_supplied() -> None:
    """With a delegate (a live worker), confirming hands the missing models off instead of self-downloading.

    The benchmark's download phase folds into the running worker's single download surface: the model
    names reach the delegate and no out-of-process download subprocess is spawned.
    """
    requested: list[list[str]] = []
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]), delegate=lambda names: requested.append(names) or True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        # Seed the computed plan directly (the on-mount dry-run is faked elsewhere), with one missing model.
        modal._plan = DownloadEvent(
            kind="planned",
            models=[DownloadModelRow(name="Deliberate", size_bytes=100, on_disk=False, target_path="x")],
            to_download_bytes=100,
            free_disk_bytes=10_000_000,
            fits=True,
        )
        modal.query_one("#download-start", Button).disabled = False
        await pilot.pause()

        await pilot.click("#download-start")
        await pilot.pause()

        assert requested == [["Deliberate"]]  # the missing model reached the worker delegate
        assert modal._process is None  # no self-download subprocess was spawned
        assert modal.query_one("#download-start", Button).disabled is True


def _plan(*models: DownloadModelRow, fits: bool = True) -> DownloadEvent:
    """A planned event over *models* with a benign disk budget, for driving the modal's render."""
    to_download = sum(model.size_bytes or 0 for model in models if not model.on_disk)
    return DownloadEvent(
        kind="planned",
        models=list(models),
        to_download_bytes=to_download,
        free_disk_bytes=10_000_000_000,
        fits=fits,
    )


def _row(name: str, *, on_disk: bool = False, size_bytes: int = 100) -> DownloadModelRow:
    """A single plan row; ``on_disk`` is the offline scan's verdict, which a live worker may override."""
    return DownloadModelRow(name=name, size_bytes=size_bytes, on_disk=on_disk, target_path="x")


async def test_live_present_overrides_a_scanned_missing_model() -> None:
    """A model the offline scan called missing but the live worker reports present needs no download.

    The worker is authoritative: the row reads on-disk and drops out of the missing set, so the operator is
    not offered a redundant fetch for something they already have.
    """
    live = DownloadLiveState(present=frozenset({"Deliberate"}))
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]), live_state=lambda: live)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        modal._render_plan(_plan(_row("Deliberate", on_disk=False)))
        await pilot.pause()

        assert modal._missing_model_names() == []
        start = modal.query_one("#download-start", Button)
        assert start.disabled is True
        assert "Nothing to download" in str(start.label)


async def test_live_in_flight_shows_downloading_and_is_not_offered() -> None:
    """A model the worker is fetching reads as downloading, not ready, and is excluded from the missing set.

    This is the download-only corner case: with the worker mid-fetch, the benchmark must not claim the model
    is ready, nor offer to fetch it again.
    """
    live = DownloadLiveState(in_flight=frozenset({"Deliberate"}))
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]), live_state=lambda: live)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        modal._render_plan(_plan(_row("Deliberate", on_disk=False)))
        await pilot.pause()

        assert modal._row_status(_row("Deliberate"), live) == "downloading"
        assert modal._missing_model_names() == []
        start = modal.query_one("#download-start", Button)
        assert start.disabled is True
        assert "Worker is fetching these" in str(start.label)  # distinct from "you have everything"


async def test_mixed_live_state_offers_only_the_genuinely_missing() -> None:
    """With one model present, one in-flight and one truly absent, only the absent one is offered."""
    live = DownloadLiveState(present=frozenset({"A"}), in_flight=frozenset({"B"}))
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]), delegate=lambda names: True, live_state=lambda: live)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        modal._render_plan(_plan(_row("A"), _row("B"), _row("C")))
        await pilot.pause()

        assert modal._missing_model_names() == ["C"]
        start = modal.query_one("#download-start", Button)
        assert start.disabled is False
        assert "Request 1 model(s)" in str(start.label)  # a live worker phrases it as a request, not a download


async def test_delegate_requests_only_models_not_already_in_flight() -> None:
    """Contrived: the plan lists two missing models but the worker is already fetching one; only the other ships.

    Guards against re-requesting an in-flight model (the scheduler would dedupe it, but the UI must not even
    ask), which would otherwise look like a redundant action to the operator.
    """
    requested: list[list[str]] = []
    live = DownloadLiveState(in_flight=frozenset({"A"}))
    app = _ModalHost(
        BenchmarkOptions(tiers=["sd15"]),
        delegate=lambda names: requested.append(names) or True,
        live_state=lambda: live,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        modal._render_plan(_plan(_row("A"), _row("B")))
        await pilot.pause()

        await pilot.click("#download-start")
        await pilot.pause()
        assert requested == [["B"]]  # only the model not already in flight is requested


async def test_no_live_state_falls_back_to_the_offline_scan() -> None:
    """With no worker (no live_state), the row's own scanned on_disk decides, and the verb is 'Download'."""
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        modal._render_plan(_plan(_row("A", on_disk=False), _row("B", on_disk=True)))
        await pilot.pause()

        assert modal._missing_model_names() == ["A"]
        start = modal.query_one("#download-start", Button)
        assert start.disabled is False
        assert "Download 1 model(s)" in str(start.label)  # no live worker -> a self-download, not a request


async def test_a_raising_live_state_reader_falls_back_to_the_scan() -> None:
    """Contrived: a flaky live-state read must never break the plan; it degrades to the offline scan."""

    def _boom() -> DownloadLiveState | None:
        raise RuntimeError("snapshot read blew up")

    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]), live_state=_boom)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        modal._render_plan(_plan(_row("A", on_disk=False)))
        await pilot.pause()

        assert modal._missing_model_names() == ["A"]  # unbroken: the scan still drives the plan
        assert modal.query_one("#download-start", Button).disabled is False
