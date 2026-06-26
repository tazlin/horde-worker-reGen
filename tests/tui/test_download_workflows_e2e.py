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

import asyncio
from pathlib import Path

import pytest
from textual.pilot import Pilot
from textual.widgets import Button, Input, Static, TabbedContent

import horde_worker_regen.tui.app as app_module
from horde_worker_regen.app_state import AppStateStore, OnboardingChoice
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadPhase,
    DownloadPlanSummary,
    DownloadStatusSnapshot,
    FeatureReadinessSummary,
    ProcessSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.models.feature_readiness import (
    FeatureReadiness,
    FeatureReadinessState,
    GatedFeature,
)
from horde_worker_regen.tui.app import BenchmarkActionConfirmModal, BenchmarkOverWorkerModal, HordeWorkerTUI
from horde_worker_regen.tui.benchmark_launcher import BenchmarkOptions
from horde_worker_regen.tui.widgets.benchmark import BenchmarkView
from horde_worker_regen.tui.widgets.download_picker import DownloadPickerModal, DownloadPickerRow
from horde_worker_regen.tui.widgets.downloads import DownloadsView
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


async def test_reload_config_key_is_not_global_binding(tmp_path: Path) -> None:
    """F5 no longer forwards a config reload; config changes flow through the Config tab."""
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await pilot.press("f5")
        await pilot.pause()
        assert fake.reload_config_calls == 0


async def test_feature_readiness_panel_appears_when_the_worker_reports_it(tmp_path: Path) -> None:
    """The Downloads feature-readiness panel stays hidden until the worker reports readiness, then shows.

    Workflow (c) at the UI seam: a gated feature whose models are still downloading is surfaced to the
    operator (rather than silently advertised), and the panel only appears once the worker has something
    to say about feature readiness.
    """
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await _open_downloads_tab(app, pilot)

        readiness_panel = app.query_one("#downloads-readiness", Static)
        assert readiness_panel.display is False  # nothing reported yet

        snapshot = WorkerStateSnapshot(config=WorkerConfigSummary(dreamer_name="T", worker_version="0.0.0"))
        snapshot.feature_readiness = FeatureReadinessSummary(
            gated=[
                FeatureReadiness(
                    feature=GatedFeature.CONTROLNET,
                    label="ControlNet",
                    state=FeatureReadinessState.WAITING,
                    detail="models still downloading",
                ),
            ],
        )
        fake.latest_snapshot = snapshot
        app._tick()
        await pilot.pause()
        assert readiness_panel.display is True


async def test_benchmark_over_a_serving_worker_asks_before_stopping_it(tmp_path: Path) -> None:
    """Requesting a benchmark while the worker serves pops a confirm modal instead of silently stopping it."""
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        launched: list[BenchmarkOptions] = []
        app._launch_benchmark = launched.append  # type: ignore[method-assign]  # observe without spawning a run

        app.on_benchmark_view_run_requested(BenchmarkView.RunRequested(BenchmarkOptions(tiers=["sd15"])))
        await pilot.pause()

        assert isinstance(app.screen, BenchmarkOverWorkerModal)  # the operator is asked first
        assert launched == []  # nothing launched yet
        assert app._supervisor.is_alive()  # the worker is still serving


async def test_cancelling_the_benchmark_prompt_keeps_the_worker_serving(tmp_path: Path) -> None:
    """Cancelling the confirm leaves the worker running and launches no benchmark."""
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        launched: list[BenchmarkOptions] = []
        app._launch_benchmark = launched.append  # type: ignore[method-assign]

        app.on_benchmark_view_run_requested(BenchmarkView.RunRequested(BenchmarkOptions(tiers=["sd15"])))
        await pilot.pause()
        await pilot.click("#bench-over-worker-cancel")
        await pilot.pause()

        assert launched == []  # cancel never starts the benchmark
        assert app._supervisor.is_alive()  # and never stops the worker
        assert not isinstance(app.screen, BenchmarkOverWorkerModal)  # the modal is dismissed


async def test_confirming_the_benchmark_prompt_launches_it(tmp_path: Path) -> None:
    """Confirming the prompt proceeds to launch the benchmark (which stops the worker)."""
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        launched: list[BenchmarkOptions] = []
        app._launch_benchmark = launched.append  # type: ignore[method-assign]  # avoid spawning a real run

        app.on_benchmark_view_run_requested(BenchmarkView.RunRequested(BenchmarkOptions(tiers=["sd15"])))
        await pilot.pause()
        await pilot.click("#bench-over-worker-confirm")
        await pilot.pause()

        assert [options.tiers for options in launched] == [["sd15"]]  # the confirmed options reach the launch


async def test_benchmark_with_no_live_worker_skips_the_prompt(tmp_path: Path) -> None:
    """With the worker stopped there is nothing to tear down, so the benchmark launches without a prompt."""
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        fake._alive = False  # the worker is not running; a benchmark takes the GPU with nothing to stop
        launched: list[BenchmarkOptions] = []
        app._launch_benchmark = launched.append  # type: ignore[method-assign]

        app.on_benchmark_view_run_requested(BenchmarkView.RunRequested(BenchmarkOptions(tiers=["sd15"])))
        await pilot.pause()

        assert not isinstance(app.screen, BenchmarkOverWorkerModal)  # no teardown -> no prompt
        assert [options.tiers for options in launched] == [["sd15"]]  # launched straight away


def test_benchmark_download_delegate_routes_to_a_live_worker(tmp_path: Path) -> None:
    """A live worker's delegate requests the missing models (with aux) straight through its download process."""
    fake, app = _make_app(tmp_path, auto_start=True)
    fake.start()  # the worker is running

    delegate = app._benchmark_download_delegate()
    assert delegate(["ModelA", "ModelB"]) is True
    assert fake.download_requests[-1].model_names == ["ModelA", "ModelB"]
    assert fake.download_requests[-1].include_aux is True


def test_benchmark_download_delegate_starts_a_stopped_worker_into_a_download_only_hold(tmp_path: Path) -> None:
    """With the worker stopped, the delegate starts it GPU-idle and queues the hold + request for the pipe.

    The benchmark never self-downloads: a stopped worker is brought up into a download-only hold (so the GPU
    stays uncommitted) and the chosen models -- with aux -- are queued to be sent once its control pipe is up.
    """
    fake, app = _make_app(tmp_path, auto_start=True)
    # The app is not mounted here, so on_mount's auto-start has not run: the worker is not live yet.
    assert not fake.is_alive()

    delegate = app._benchmark_download_delegate()
    assert delegate(["ModelA", "ModelB"]) is True

    assert fake.start_calls == 1  # the worker was started to download, GPU idle
    assert app._pending_downloads_only_hold is True  # held until the operator goes live
    assert app._pending_download_models is not None
    assert app._pending_download_models.model_names == ["ModelA", "ModelB"]
    assert app._pending_download_models.include_aux is True


_EXPECTED_GRACEFUL_HANDOFF_REQUESTS = ["drain", "downloads_only_hold", "set_concurrency:processes=0:threads=None"]
"""The exact control sequence a successful graceful handoff sends: stop popping, hold, then scale to zero."""


def _inference_process(*, busy: bool) -> ProcessSnapshot:
    """Return a live inference process snapshot, either mid-job (busy) or idle (awaiting work)."""
    return ProcessSnapshot(
        process_id=0,
        process_type="INFERENCE",
        last_process_state="INFERENCE_STARTING" if busy else "WAITING_FOR_JOB",
        is_alive=True,
        is_busy=busy,
    )


def _serving_snapshot() -> WorkerStateSnapshot:
    """Return a snapshot of a worker mid-job: one inference in flight on a live inference process."""
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="TestWorker", worker_version="0.0.0"),
        jobs_in_progress=1,
        jobs_pending_inference=0,
        processes=[_inference_process(busy=True)],
    )


def _drained_but_inference_up_snapshot() -> WorkerStateSnapshot:
    """Return a snapshot drained of jobs whose inference process is still up, holding the GPU."""
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="TestWorker", worker_version="0.0.0"),
        jobs_in_progress=0,
        jobs_pending_inference=0,
        processes=[_inference_process(busy=False)],
    )


def _idle_drained_snapshot() -> WorkerStateSnapshot:
    """Return a snapshot fully drained and scaled down: no job in flight and no inference process alive."""
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="TestWorker", worker_version="0.0.0"),
        jobs_in_progress=0,
        jobs_pending_inference=0,
        processes=[],
    )


async def test_benchmark_drains_a_serving_worker_step_by_step_then_frees_the_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker mid-job is drained, held, and scaled to zero in order, the handoff waiting on each transition.

    The fake worker responds the way a real one would: it reports its in-flight job finished only after DRAIN,
    and its inference process gone only after SET_CONCURRENCY(0). The handoff must observe each transition
    before taking the next step, must keep the worker alive, and must never hard-stop it.
    """
    monkeypatch.setattr(app_module, "_BENCHMARK_DRAIN_POLL_SECONDS", 0.02)
    monkeypatch.setattr(app_module, "_BENCHMARK_DRAIN_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(app_module, "_BENCHMARK_SCALE_TIMEOUT_SECONDS", 5.0)
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        started: list[BenchmarkOptions] = []
        app._benchmark_supervisor.start = lambda options: started.append(options)  # type: ignore[method-assign]
        app._pending_benchmark_options = BenchmarkOptions(tiers=["sd15"])
        fake.latest_snapshot = _serving_snapshot()
        fake.requests.clear()

        async def _worker_responds_to_commands() -> None:
            """Advance the worker's reported state in response to the handoff's commands, like a real worker."""
            while "drain" not in fake.requests:
                await asyncio.sleep(0.01)
            fake.latest_snapshot = _drained_but_inference_up_snapshot()  # the job finished
            while not fake.set_concurrency_calls:
                await asyncio.sleep(0.01)
            fake.latest_snapshot = _idle_drained_snapshot()  # the inference process exited

        responder = asyncio.create_task(_worker_responds_to_commands())
        await asyncio.to_thread(app._start_benchmark_flow)
        await responder
        await pilot.pause()

        assert app._supervisor.is_alive()  # kept alive, not torn down
        assert fake.stop_calls == 0  # no hard stop
        assert app._benchmark_drained_worker is True
        assert fake.requests == _EXPECTED_GRACEFUL_HANDOFF_REQUESTS
        assert fake.set_concurrency_calls == [(0, None)]
        assert [options.tiers for options in started] == [["sd15"]]  # launched only after the GPU freed


async def test_benchmark_hard_stops_when_an_in_flight_job_will_not_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job that never finishes makes the drain wait time out before any hold/scale, so the handoff hard-stops.

    Proves the drain step genuinely gates on the worker's reported state: with the job stuck in flight, the
    worker is never held or scaled (that would strand the operator), and the backstop stop frees the GPU.
    """
    monkeypatch.setattr(app_module, "_BENCHMARK_DRAIN_POLL_SECONDS", 0.02)
    monkeypatch.setattr(app_module, "_BENCHMARK_DRAIN_TIMEOUT_SECONDS", 0.1)
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        started: list[BenchmarkOptions] = []
        app._benchmark_supervisor.start = lambda options: started.append(options)  # type: ignore[method-assign]
        app._pending_benchmark_options = BenchmarkOptions(tiers=["sd15"])
        fake.latest_snapshot = _serving_snapshot()  # the job stays in flight forever
        fake.requests.clear()

        await asyncio.to_thread(app._start_benchmark_flow)
        await pilot.pause()

        assert app._benchmark_drained_worker is False
        assert fake.stop_calls == 1  # backstop fired
        assert fake.requests == ["drain"]  # only the drain was attempted; never hold/scale an undrained worker
        assert fake.set_concurrency_calls == []
        assert [options.tiers for options in started] == [["sd15"]]  # the benchmark still launched


async def test_benchmark_hard_stops_when_inference_will_not_scale_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inference process that never exits makes the scale-down wait time out, so the handoff hard-stops.

    Proves the scale step gates on the GPU actually freeing: the job drains and the worker is held and asked to
    scale to zero, but its inference process lingers, so the backstop stop frees the GPU for the run.
    """
    monkeypatch.setattr(app_module, "_BENCHMARK_DRAIN_POLL_SECONDS", 0.02)
    monkeypatch.setattr(app_module, "_BENCHMARK_SCALE_TIMEOUT_SECONDS", 0.1)
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        started: list[BenchmarkOptions] = []
        app._benchmark_supervisor.start = lambda options: started.append(options)  # type: ignore[method-assign]
        app._pending_benchmark_options = BenchmarkOptions(tiers=["sd15"])
        fake.latest_snapshot = _drained_but_inference_up_snapshot()  # drained, but inference never sheds
        fake.requests.clear()

        await asyncio.to_thread(app._start_benchmark_flow)
        await pilot.pause()

        assert app._benchmark_drained_worker is False
        assert fake.stop_calls == 1  # backstop fired
        assert fake.requests == _EXPECTED_GRACEFUL_HANDOFF_REQUESTS  # drain + hold + scale were all attempted
        assert [options.tiers for options in started] == [["sd15"]]  # the benchmark still launched


def _downloads_snapshot(*, phase: DownloadPhase, present: tuple[str, ...] = ()) -> WorkerStateSnapshot:
    """A worker snapshot whose download subsystem is in *phase* with *present* models already on disk."""
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="TestWorker", worker_version="0.0.0"),
        downloads=DownloadStatusSnapshot(phase=phase, present_model_names=list(present)),
    )


def _enter_benchmark_download_wait(app: HordeWorkerTUI, fake: FakeSupervisor, names: list[str]) -> None:
    """Drive the benchmark download delegate on a live worker so the app enters its waiting-for-models mode."""
    fake.start()
    delegate = app._benchmark_download_delegate()
    assert delegate(names) is True
    assert app._benchmark_waiting_models == set(names)


async def test_benchmark_waiting_banner_tracks_progress_then_clears_when_downloads_settle(tmp_path: Path) -> None:
    """The waiting banner reflects download progress and clears once the subsystem finishes, re-enabling Run.

    The completion is judged by the subsystem returning to idle (not per-name), so a requested set that
    includes un-named feature models still resolves rather than waiting forever.
    """
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        _enter_benchmark_download_wait(app, fake, ["ModelA", "ModelB"])
        view = app.query_one(BenchmarkView)

        # The worker reports an active download with one model already present: the banner shows 1 of 2.
        fake.latest_snapshot = _downloads_snapshot(phase=DownloadPhase.DOWNLOADING, present=("ModelA",))
        app._tick()
        await pilot.pause()
        assert view._waiting is not None
        assert (view._waiting.ready, view._waiting.total) == (1, 2)
        assert app.query_one("#benchmark-waiting-banner", Static).display is True
        assert app.query_one("#benchmark-download", Button).disabled is True  # no double-request while waiting

        # The subsystem settles to idle: the wait completes, the banner clears, and Run is no longer gated.
        fake.latest_snapshot = _downloads_snapshot(phase=DownloadPhase.IDLE, present=("ModelA", "ModelB"))
        app._tick()
        await pilot.pause()
        assert app._benchmark_waiting_models == set()
        assert view._waiting is None
        assert app.query_one("#benchmark-waiting-banner", Static).display is False


async def test_benchmark_waiting_does_not_complete_on_an_idle_snapshot_before_downloads_start(
    tmp_path: Path,
) -> None:
    """A worker still idle right after the request must not be read as "done" before the fetch has begun."""
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        _enter_benchmark_download_wait(app, fake, ["ModelA"])

        # The download has not started yet (still idle, nothing present): the wait must persist, not complete.
        fake.latest_snapshot = _downloads_snapshot(phase=DownloadPhase.IDLE, present=())
        app._tick()
        await pilot.pause()
        assert app._benchmark_waiting_models == {"ModelA"}  # still waiting, not a false completion


async def test_running_while_benchmark_models_download_warns_first(tmp_path: Path) -> None:
    """Pressing Run while the models still download pops a confirm; confirming abandons the wait and proceeds."""
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        _enter_benchmark_download_wait(app, fake, ["ModelA"])
        proceeded: list[BenchmarkOptions] = []
        app._proceed_with_run_request = proceeded.append  # type: ignore[method-assign]

        app.on_benchmark_view_run_requested(BenchmarkView.RunRequested(BenchmarkOptions(tiers=["sd15"])))
        await pilot.pause()
        assert isinstance(app.screen, BenchmarkActionConfirmModal)  # gated: the operator is asked first
        assert proceeded == []  # nothing launched while the dialog is up

        await pilot.click("#bench-confirm-confirm")
        await pilot.pause()
        assert app._benchmark_waiting_models == set()  # the wait was abandoned to run now
        assert [options.tiers for options in proceeded] == [["sd15"]]  # and the run proceeded


async def test_cancelling_the_run_while_waiting_keeps_waiting(tmp_path: Path) -> None:
    """Cancelling the run-while-downloading confirm leaves the wait intact and launches nothing."""
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        _enter_benchmark_download_wait(app, fake, ["ModelA"])
        proceeded: list[BenchmarkOptions] = []
        app._proceed_with_run_request = proceeded.append  # type: ignore[method-assign]

        app.on_benchmark_view_run_requested(BenchmarkView.RunRequested(BenchmarkOptions(tiers=["sd15"])))
        await pilot.pause()
        await pilot.click("#bench-confirm-cancel")
        await pilot.pause()
        assert app._benchmark_waiting_models == {"ModelA"}  # still waiting
        assert proceeded == []  # the run did not proceed


async def test_go_live_while_benchmark_only_downloads_run_warns_then_clears_on_confirm(tmp_path: Path) -> None:
    """Going live while benchmark-only models download warns; confirming abandons the wait and goes live."""
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        _enter_benchmark_download_wait(app, fake, ["ModelA"])
        app._benchmark_waiting_outside_config = lambda: True  # type: ignore[method-assign]  # not in config
        fake.requests.clear()

        app.on_downloads_view_go_live_requested(DownloadsView.GoLiveRequested())
        await pilot.pause()
        assert isinstance(app.screen, BenchmarkActionConfirmModal)  # warned before stranding the fetch
        assert "go_live" not in fake.requests  # not sent until the operator agrees

        await pilot.click("#bench-confirm-confirm")
        await pilot.pause()
        assert app._benchmark_waiting_models == set()  # the wait is abandoned (serving may stop the fetch)
        assert "go_live" in fake.requests


async def test_go_live_does_not_warn_when_benchmark_models_are_in_config(tmp_path: Path) -> None:
    """When every waited-for model is in the worker config, going live re-fetches them anyway: no warning.

    The wait is left intact (the config-driven download continues), so the banner keeps tracking them.
    """
    fake, app = _make_app(tmp_path, auto_start=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        _enter_benchmark_download_wait(app, fake, ["ModelA"])
        app._benchmark_waiting_outside_config = lambda: False  # type: ignore[method-assign]  # all in config
        fake.requests.clear()

        app.on_downloads_view_go_live_requested(DownloadsView.GoLiveRequested())
        await pilot.pause()
        assert not isinstance(app.screen, BenchmarkActionConfirmModal)  # no warning needed
        assert "go_live" in fake.requests  # went live straight away
        assert app._benchmark_waiting_models == {"ModelA"}  # the config-driven fetch is still tracked
