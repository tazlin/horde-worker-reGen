"""End-to-end TUI workflow tests for the download subsystem, driven through real clicks.

These run the actual ``HordeWorkerTUI`` under Textual's ``run_test`` harness against an in-process
[`FakeSupervisor`][tests.tui._fake_supervisor.FakeSupervisor]. Each test follows a genuine operator
goal: pause a download, cap bandwidth, pre-fetch models without committing the GPU, or push a config
reload. They assert user-visible outcomes (a button flips its label, the chosen models reach the worker,
the worker comes up before any fetch is requested) rather than internal call wiring.

The supervisor is faked, not the UI: the real Downloads widgets, the real app message handlers, the real
``_tick`` flush, and the real download-picker modal are all exercised. Only the worker process behind the
supervisor boundary is stood in for, which is what makes these deterministic without a GPU or a spawned
child.
"""

from __future__ import annotations

from pathlib import Path

from textual.pilot import Pilot
from textual.widgets import Button, Input, TabbedContent

from horde_worker_regen.app_state import AppStateStore, OnboardingChoice
from horde_worker_regen.process_management.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadPhase,
    DownloadPlanSummary,
    DownloadStatusSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.app import HordeWorkerTUI
from horde_worker_regen.tui.widgets.download_picker import DownloadPickerModal, DownloadPickerRow
from horde_worker_regen.tui.widgets.onboarding import WorkerStartModal
from tests.tui._fake_supervisor import FakeSupervisor


def _make_app(
    tmp_path: Path,
    *,
    auto_start: bool,
) -> tuple[FakeSupervisor, HordeWorkerTUI]:
    """Build a TUI wired to a fake worker, with app state primed to avoid first-run prompts.

    Onboarding is pre-declined so no benchmark modal appears. With ``auto_start`` on, ``on_mount`` starts
    the (fake) worker and shows no start prompt, giving a running worker and a clear screen; with it off,
    the real first-run :class:`WorkerStartModal` appears, which the download-only scenarios drive.
    """
    config_path = tmp_path / "bridgeData.yaml"
    config_path.write_text("api_key: test\ndreamer_name: TestWorker\n", encoding="utf-8")
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    store.record_onboarding_choice(OnboardingChoice.DECLINED)
    if auto_start:
        store.set_auto_start_worker(True)
    fake = FakeSupervisor()
    app = HordeWorkerTUI(fake, config_path=config_path, app_state_store=store)
    return fake, app


async def _open_downloads_tab(app: HordeWorkerTUI, pilot: Pilot[None]) -> None:
    """Activate the Downloads tab and let the layout settle so its controls are clickable."""
    app.query_one("#main-tabs", TabbedContent).active = "tab-downloads"
    await pilot.pause()


def _downloading_snapshot(*, paused: bool = False) -> WorkerStateSnapshot:
    """A snapshot the worker would report for one in-flight image-model download (optionally paused)."""
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="TestWorker", worker_version="0.0.0"),
        download_plan=DownloadPlanSummary(num_present=1, num_to_download=2),
        downloads=DownloadStatusSnapshot(
            phase=DownloadPhase.PAUSED if paused else DownloadPhase.DOWNLOADING,
            current=CurrentDownloadStatus(
                model_name="BigModel",
                feature="image model",
                target_dir="/models",
                downloaded_bytes=512,
                total_bytes=1024,
                speed_bps=4096.0,
            ),
            paused=paused,
        ),
    )


async def test_operator_pauses_then_resumes_background_downloads(tmp_path: Path) -> None:
    """The Downloads pause control sends pause, flips to Resume when the worker reports it, then resumes.

    Exercises both directions of the contract: the click reaches the worker (recorded request), and the
    worker's reported paused state is reflected back in the live button label the operator reads.
    """
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await _open_downloads_tab(app, pilot)

        # A download is actively transferring; the control offers to pause it.
        fake.latest_snapshot = _downloading_snapshot(paused=False)
        app._tick()
        await pilot.pause()
        pause_button = app.query_one("#downloads-pause", Button)
        assert "Pause downloads" in str(pause_button.label)

        await pilot.click("#downloads-pause")
        await pilot.pause()
        assert fake.pause_downloads_calls == 1
        assert fake.resume_downloads_calls == 0

        # The worker now reports the download as paused; the control flips to Resume on its own.
        fake.latest_snapshot = _downloading_snapshot(paused=True)
        app._tick()
        await pilot.pause()
        assert "Resume downloads" in str(pause_button.label)

        await pilot.click("#downloads-pause")
        await pilot.pause()
        assert fake.resume_downloads_calls == 1


async def test_operator_caps_then_clears_download_bandwidth(tmp_path: Path) -> None:
    """Typing a KB/s cap and pressing Apply sends that value to the worker; a 0 clears the cap."""
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await _open_downloads_tab(app, pilot)

        rate_input = app.query_one("#downloads-rate", Input)
        rate_input.value = "500"
        await pilot.click("#downloads-rate-apply")
        await pilot.pause()
        assert fake.rate_limits_kbps == [500]

        rate_input.value = "0"
        await pilot.click("#downloads-rate-apply")
        await pilot.pause()
        assert fake.rate_limits_kbps == [500, 0]


async def test_first_run_download_only_then_go_live(tmp_path: Path) -> None:
    """The first-run 'Download models only' choice holds the worker; Go live then releases it to serve.

    The hold is deferred until the freshly-started worker's pipe is up, then flushed by the refresh tick;
    Go live is only sent afterwards, so the worker is held in download-only before it is released to serve.
    """
    fake, app = _make_app(tmp_path, auto_start=False)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, WorkerStartModal)

        await pilot.click("#worker-start-download-only")
        await pilot.pause()
        # The choice starts the worker and focuses the Downloads tab.
        assert fake.start_calls == 1
        assert app.query_one("#main-tabs", TabbedContent).active == "tab-downloads"

        app._tick()  # flush the deferred download-only hold (idempotent once it has been sent).
        assert fake.downloads_only_hold_calls == 1

        await pilot.click("#downloads-go-live")
        await pilot.pause()
        assert fake.go_live_calls == 1
        assert fake.requests == ["downloads_only_hold", "go_live"]


async def test_picker_selection_requests_models_in_download_only_hold(tmp_path: Path) -> None:
    """Confirming a picker selection on a running worker fetches exactly those models while holding inference."""
    fake, app = _make_app(tmp_path, auto_start=True)
    rows = [
        DownloadPickerRow(name="Present", baseline="stable_diffusion_xl", size_bytes=6_000_000_000, on_disk=True),
        DownloadPickerRow(name="Missing A", baseline="stable_diffusion_1", size_bytes=2_000_000_000, on_disk=False),
        DownloadPickerRow(name="Missing B", baseline="flux_1", size_bytes=None, on_disk=False),
    ]
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.push_screen(DownloadPickerModal(rows), app._on_download_selection)
        await pilot.pause()

        # The picker defaults to the not-on-disk models; confirm fetches them.
        await pilot.click("#download-picker-confirm")
        await pilot.pause()

        assert fake.downloads_only_hold_calls == 1
        assert len(fake.download_requests) == 1
        request = fake.download_requests[0]
        assert request.model_names == ["Missing A", "Missing B"]
        assert request.include_aux is False


async def test_picker_selection_with_stopped_worker_holds_before_fetching(tmp_path: Path) -> None:
    """With the worker stopped, a picker selection starts it, then holds *before* fetching on the next tick.

    The ordering matters: the worker must enter download-only hold before the model fetch is requested, so
    a cold install never starts inference while it pre-downloads the chosen models.
    """
    fake, app = _make_app(tmp_path, auto_start=False)
    rows = [
        DownloadPickerRow(name="Missing A", baseline="stable_diffusion_1", size_bytes=2_000_000_000, on_disk=False),
    ]
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        # Decline to start the worker at the first-run prompt, then choose to pre-download models.
        assert isinstance(app.screen, WorkerStartModal)
        await pilot.click("#worker-start-stay-stopped")
        await pilot.pause()
        assert fake.start_calls == 0

        app.push_screen(DownloadPickerModal(rows), app._on_download_selection)
        await pilot.pause()
        await pilot.click("#download-picker-confirm")
        await pilot.pause()
        # Confirming a selection on a stopped worker starts it so the request has somewhere to land.
        assert fake.start_calls == 1

        app._tick()  # flush the deferred hold and model request (idempotent once they have been sent).
        # The hold is requested before the fetch, so a cold install never starts inference mid-download.
        assert fake.requests == ["downloads_only_hold", "download_models"]
        assert fake.download_requests[0].model_names == ["Missing A"]


async def test_reload_config_key_reaches_the_worker(tmp_path: Path) -> None:
    """Pressing F5 forwards a bridgeData reload request to the running worker."""
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await pilot.press("f5")
        await pilot.pause()
        assert fake.reload_config_calls == 1
