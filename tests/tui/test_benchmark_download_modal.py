"""Pilot tests for the benchmark download modal: plan rendering, live-worker overlay, and delegation.

Drives the real :class:`BenchmarkDownloadModal` under Textual's ``run_test``: the plan subprocess is faked so
no real benchmark runs. The modal is confirmation-only (it never downloads in-process), so the tests assert
that the plan reflects a live worker's authoritative state and that confirming hands the missing models to the
download delegate (the worker's own orchestration).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from horde_worker_regen.benchmark.download_progress import (
    DownloadEvent,
    DownloadModelRow,
    encode_download_event,
)
from horde_worker_regen.tui.benchmark_launcher import BenchmarkOptions
from horde_worker_regen.tui.widgets.benchmark_download import BenchmarkDownloadModal, DownloadLiveState


def _accept_all(model_names: list[str]) -> bool:
    """Return True for any request: the default delegate for tests that do not inspect what was requested."""
    return True


class _ModalHost(App[None]):
    """Pushes the download modal as a screen so Pilot can drive its buttons."""

    def __init__(
        self,
        options: BenchmarkOptions,
        *,
        delegate: Callable[[list[str]], bool] = _accept_all,
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


def _plan(*models: DownloadModelRow, fits: bool = True) -> DownloadEvent:
    """Return a planned event over *models* with a benign disk budget, for driving the modal's render."""
    to_download = sum(model.size_bytes or 0 for model in models if not model.on_disk)
    return DownloadEvent(
        kind="planned",
        models=list(models),
        to_download_bytes=to_download,
        free_disk_bytes=10_000_000_000,
        fits=fits,
    )


def _row(name: str, *, on_disk: bool = False, size_bytes: int = 100) -> DownloadModelRow:
    """Return a single plan row; ``on_disk`` is the offline scan's verdict, which a live worker may override."""
    return DownloadModelRow(name=name, size_bytes=size_bytes, on_disk=on_disk, target_path="x")


async def test_confirming_requests_the_missing_models_via_the_delegate() -> None:
    """Confirming hands the missing models to the download delegate and never downloads in-process.

    The benchmark's download phase folds into the worker's single download surface: the model names reach the
    delegate, the modal records that a request was made, and the Request button disables to prevent re-asking.
    """
    requested: list[list[str]] = []
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]), delegate=lambda names: requested.append(names) or True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        modal._render_plan(_plan(_row("Deliberate", on_disk=False)))
        await pilot.pause()

        await pilot.click("#download-start")
        await pilot.pause()

        assert requested == [["Deliberate"]]  # the missing model reached the download orchestration
        assert modal._requested_download is True
        assert modal.query_one("#download-start", Button).disabled is True


async def test_feature_models_are_not_requested_as_image_models() -> None:
    """Controlnet/post-proc checkpoint rows (``is_aux``) are fetched via the aux pass, never requested by name.

    Requesting an aux model by name routes it to the image manager, which has no record of it and fails (the
    live-worker "Cannot download model without a reference record" bug). The fix keeps aux rows out of the
    by-name image request while the delegate's ``include_aux`` still fetches them through their own managers.
    """
    requested: list[list[str]] = []
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]), delegate=lambda names: requested.append(names) or True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        image_row = _row("SD15Checkpoint", on_disk=False)
        aux_row = DownloadModelRow(name="control_canny", on_disk=False, target_path="x", is_aux=True)
        modal._render_plan(_plan(image_row, aux_row))
        await pilot.pause()

        # The plan still shows both, so the operator sees the full picture of what will be fetched.
        assert modal._missing_model_names() == ["SD15Checkpoint", "control_canny"]

        await pilot.click("#download-start")
        await pilot.pause()
        # ...but only the image model is requested by name; the controlnet checkpoint is left to the aux pass.
        assert requested == [["SD15Checkpoint"]]


async def test_a_rejected_request_surfaces_an_error_and_does_not_mark_requested() -> None:
    """If the download subsystem cannot be reached, the modal says so and stays re-tryable (not 'requested')."""
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]), delegate=lambda names: False)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        modal._render_plan(_plan(_row("Deliberate", on_disk=False)))
        await pilot.pause()

        await pilot.click("#download-start")
        await pilot.pause()

        assert modal._requested_download is False  # nothing was accepted, so the run can still be retried


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
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]), live_state=lambda: live)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        modal._render_plan(_plan(_row("A"), _row("B"), _row("C")))
        await pilot.pause()

        assert modal._missing_model_names() == ["C"]
        start = modal.query_one("#download-start", Button)
        assert start.disabled is False
        assert "Request 1 model(s)" in str(start.label)


async def test_delegate_requests_only_models_not_already_in_flight() -> None:
    """Contrived: the plan lists two missing models but the worker is fetching one; only the other ships.

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
    """With no worker (no live_state), the row's own scanned on_disk decides which models are missing."""
    app = _ModalHost(BenchmarkOptions(tiers=["sd15"]))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = app.modal
        modal._render_plan(_plan(_row("A", on_disk=False), _row("B", on_disk=True)))
        await pilot.pause()

        assert modal._missing_model_names() == ["A"]
        start = modal.query_one("#download-start", Button)
        assert start.disabled is False
        assert "Request 1 model(s)" in str(start.label)


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
