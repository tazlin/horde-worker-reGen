"""Behavioural cover for the progressive experience levels.

These assert the properties the design rests on: navigation never changes shape, the level default is
announced rather than applied silently, withheld settings survive a save, and the liveness indicator
cannot be satisfied by the failure it is meant to detect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from textual.widgets import TabbedContent

from horde_worker_regen.app_state import (
    APP_STATE_SCHEMA_VERSION,
    AppStateStore,
    DisplayDensity,
    ExperienceLevel,
    OnboardingChoice,
    WorkerAppState,
)
from horde_worker_regen.tui.app import HordeWorkerTUI
from horde_worker_regen.tui.widgets.config_editor import ConfigEditorView
from horde_worker_regen.tui.widgets.experience import ExperienceIntroductionModal
from horde_worker_regen.tui.widgets.simple import (
    LivenessIndicator,
    SimpleActivityView,
    SimpleHomeView,
    SimpleModelStatusView,
    job_progress_fraction,
)
from tests.tui._fake_supervisor import FakeSupervisor

ALL_TABS = [
    "tab-overview",
    "tab-stats",
    "tab-control",
    "tab-gpus",
    "tab-live",
    "tab-downloads",
    "tab-logs",
    "tab-config",
    "tab-insights",
    "tab-diagnostics",
    "tab-benchmark",
]


def _make_app(tmp_path: Path, *, level: ExperienceLevel | None = None) -> tuple[FakeSupervisor, HordeWorkerTUI]:
    """Build a TUI over a fake worker, optionally pinned to an experience level."""
    config_path = tmp_path / "bridgeData.yaml"
    config_path.write_text("api_key: test\ndreamer_name: TestWorker\n", encoding="utf-8")
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    store.record_onboarding_choice(OnboardingChoice.DECLINED)
    store.set_auto_start_worker(True)
    if level is not None:
        store.set_experience_level(level)
    fake = FakeSupervisor()
    return fake, HordeWorkerTUI(fake, config_path=config_path, app_state_store=store)


def _rendered_row(app: HordeWorkerTUI, *, row: int) -> str:
    """Return one row of the composited screen as plain text, as the terminal would show it."""
    strips = app.screen._compositor.render_strips()
    return "".join(segment.text for segment in strips[row]).rstrip()


class _Process:
    """A stand-in carrying only the progress fields the Simple helpers read."""

    def __init__(
        self, steps: int = 0, percent: int | None = None, current: int | None = None, total: int | None = None
    ) -> None:
        """Store the reported progress counters."""
        self.heartbeats_inference_steps = steps
        self.last_heartbeat_percent_complete = percent
        self.last_current_step = current
        self.last_total_steps = total


class _Snapshot:
    """A stand-in carrying only the fields the liveness indicator reads."""

    def __init__(self, *, steps: int, timestamp: float, jobs_in_progress: int) -> None:
        """Store one frame of worker state."""
        self.processes = [_Process(steps)]
        self.timestamp = timestamp
        self.jobs_in_progress = jobs_in_progress


@pytest.mark.slow
@pytest.mark.parametrize("level", list(ExperienceLevel))
async def test_every_destination_exists_at_every_level(tmp_path: Path, level: ExperienceLevel) -> None:
    """No level hides, merges, or reorders a destination; the level changes depth in place."""
    _fake, app = _make_app(tmp_path, level=level)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._commit_experience_level(level)
        await pilot.pause()
        tabs = app.query_one("#main-tabs", TabbedContent)

        present = [pane.id for pane in tabs.query("TabPane") if (pane.id or "").startswith("tab-")]
        assert present == ALL_TABS

        # Present is not enough: a hidden tab is unreachable even though its pane still exists.
        for tab_id in ALL_TABS:
            assert tabs.get_tab(tab_id).display, f"{tab_id} is not reachable at {level}"

        # Every destination can actually be activated.
        for tab_id in ALL_TABS:
            tabs.active = tab_id
            await pilot.pause()
            assert tabs.active == tab_id


@pytest.mark.slow
async def test_simple_and_operator_views_swap_without_both_showing(tmp_path: Path) -> None:
    """Each shared destination shows exactly one of its two presentations, never both or neither."""
    from horde_worker_regen.tui.widgets.downloads import DownloadsView
    from horde_worker_regen.tui.widgets.live_view import LiveView
    from horde_worker_regen.tui.widgets.overview import OverviewView

    _fake, app = _make_app(tmp_path)
    pairs = [(SimpleHomeView, OverviewView), (SimpleActivityView, LiveView), (SimpleModelStatusView, DownloadsView)]
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for level in ExperienceLevel:
            app._commit_experience_level(level)
            await pilot.pause()
            simple = level is ExperienceLevel.SIMPLE
            for simple_view, operator_view in pairs:
                assert app.query_one(simple_view).display is simple
                assert app.query_one(operator_view).display is not simple


@pytest.mark.slow
async def test_pre_level_state_is_offered_the_choice_exactly_once(tmp_path: Path) -> None:
    """An installation predating the levels is told the default changed, and only the first time."""
    state_path = tmp_path / ".horde_worker_regen" / "state.json"
    state_path.parent.mkdir(parents=True)
    # A genuine v1 file: no experience keys, and accumulated state worth preserving.
    state_path.write_text(
        json.dumps({"schema_version": 1, "auto_start_worker": True, "setup_complete": True}),
        encoding="utf-8",
    )
    config_path = tmp_path / "bridgeData.yaml"
    config_path.write_text("api_key: test\ndreamer_name: TestWorker\n", encoding="utf-8")

    store = AppStateStore(state_path)
    assert store.load().needs_experience_introduction is True

    app = HordeWorkerTUI(FakeSupervisor(), config_path=config_path, app_state_store=store)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ExperienceIntroductionModal)
        app.screen.dismiss(ExperienceLevel.ADVANCED)
        await pilot.pause()

    reloaded = store.load()
    assert reloaded.experience_level is ExperienceLevel.ADVANCED
    assert reloaded.needs_experience_introduction is False
    assert reloaded.auto_start_worker is True, "answering the notice must not discard accumulated state"

    # A second launch must not ask again.
    app2 = HordeWorkerTUI(FakeSupervisor(), config_path=config_path, app_state_store=store)
    async with app2.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert not isinstance(app2.screen, ExperienceIntroductionModal)


def test_fresh_install_is_not_offered_the_upgrade_notice(tmp_path: Path) -> None:
    """A new contributor has no prior experience to be surprised by, so nothing is announced."""
    store = AppStateStore(tmp_path / "state.json")
    assert store.load().needs_experience_introduction is False
    assert store.load().experience_level is ExperienceLevel.SIMPLE


def test_loading_an_old_state_preserves_it_and_stamps_the_version(tmp_path: Path) -> None:
    """Bumping the schema must not discard accumulated state."""
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "auto_start_worker": True,
                "overview_trend_window": "60m",
                "overview_hidden_elements": ["a", "b"],
                "worker_version_last_ran": "17.0.0",
                "detailed_info": True,
            },
        ),
        encoding="utf-8",
    )
    state = AppStateStore(path).load()
    assert state.auto_start_worker is True
    assert state.overview_trend_window.value == "60m"
    assert state.overview_hidden_elements == ["a", "b"]
    assert state.worker_version_last_ran == "17.0.0"
    assert state.overview_view_mode.value == "details", "the pre-existing detailed_info migration still applies"
    assert state.schema_version == APP_STATE_SCHEMA_VERSION


def test_unknown_theme_falls_back_rather_than_failing_the_load(tmp_path: Path) -> None:
    """A theme this build cannot restore must not block startup."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema_version": 2, "theme_name": "not-a-real-theme"}), encoding="utf-8")
    assert AppStateStore(path).load().theme_name == "horde-dark"

    store = AppStateStore(tmp_path / "other.json")
    store.set_theme_name("not-a-real-theme")
    assert store.load().theme_name == "horde-dark"
    store.set_theme_name("horde-light")
    assert store.load().theme_name == "horde-light"


@pytest.mark.slow
async def test_settings_withheld_in_simple_survive_a_save(tmp_path: Path) -> None:
    """A field the current level does not show must still be written back, not dropped."""
    _fake, app = _make_app(tmp_path, level=ExperienceLevel.SIMPLE)
    config_path = tmp_path / "bridgeData.yaml"
    data = yaml.safe_load(config_path.read_text()) or {}
    data["max_threads"] = 2
    data["queue_size"] = 3
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#main-tabs", TabbedContent).active = "tab-config"
        await pilot.pause()
        editor = app.query_one(ConfigEditorView)
        assert editor._save() is True
        await pilot.pause()

    after = yaml.safe_load(config_path.read_text()) or {}
    assert after["max_threads"] == 2
    assert after["queue_size"] == 3
    assert after["api_key"] == "test"


@pytest.mark.slow
async def test_simple_config_always_offers_the_way_back(tmp_path: Path) -> None:
    """The Dashboard page is never withheld, so the level can always be changed from the editor."""
    _fake, app = _make_app(tmp_path, level=ExperienceLevel.SIMPLE)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#main-tabs", TabbedContent).active = "tab-config"
        await pilot.pause()
        subtabs = app.query_one(ConfigEditorView).query_one("#config-subtabs", TabbedContent)
        assert subtabs.get_tab("cfgtab-dashboard").display

        offered = [p.id for p in subtabs.query("TabPane") if subtabs.get_tab(p.id).display]
        assert "cfgtab-advanced" not in offered
        assert "cfgtab-per-gpu" not in offered

        app._commit_experience_level(ExperienceLevel.ADVANCED)
        await pilot.pause()
        offered_advanced = [p.id for p in subtabs.query("TabPane") if subtabs.get_tab(p.id).display]
        assert "cfgtab-advanced" in offered_advanced
        assert "cfgtab-per-gpu" in offered_advanced


@pytest.mark.slow
async def test_every_shortcut_is_discoverable_though_the_footer_truncates(tmp_path: Path) -> None:
    """The footer shows only what fits, so the palette and help must carry the complete set."""
    _fake, app = _make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        actions = app.keyboard_actions()
        assert actions, "there are bindings to discover"

        titles = [command.title for command in app.get_system_commands(app.screen)]
        for _keys, description, _action in actions:
            assert any(title.startswith(description) for title in titles), f"{description} missing from the palette"
        for label, _tab_id in app._DESTINATIONS:
            assert f"Go to {label}" in titles

        # The shortcut is shown alongside the command, so the palette teaches the key.
        assert any("(f3)" in title for title in titles)

        # Exactly one command-palette key: Textual pins its own, so declaring another duplicates it.
        assert sum(1 for _k, _d, action in actions if action == "command_palette") == 0


def test_liveness_freezes_when_the_worker_wedges() -> None:
    """A stalled worker must not be able to satisfy the indicator meant to detect it.

    The snapshot timestamp keeps advancing while a wedged worker's loop still runs, so an indicator
    driven by it (or by the render loop) would animate over a dead worker.
    """
    indicator = LivenessIndicator()
    markers = []
    for tick in range(60):
        snapshot = _Snapshot(steps=100, timestamp=1000.0 + tick, jobs_in_progress=1)
        indicator.update(snapshot, is_alive=True)
        markers.append(str(indicator.marker(snapshot, is_alive=True)))

    assert indicator.stalled is True
    assert len(set(markers[:5])) == 1, "the indicator must not animate while no progress is reported"


def test_liveness_advances_only_on_real_inference_progress() -> None:
    """The indicator advances when the worker's own step counter advances."""
    indicator = LivenessIndicator()
    markers = []
    for tick in range(6):
        snapshot = _Snapshot(steps=100 + tick * 7, timestamp=1000.0 + tick, jobs_in_progress=1)
        indicator.update(snapshot, is_alive=True)
        markers.append(str(indicator.marker(snapshot, is_alive=True)))

    assert len(set(markers)) > 1
    assert indicator.stalled is False


def test_liveness_shows_an_idle_worker_as_alive() -> None:
    """With no work in hand the step counter is legitimately static, which is not a stall."""
    indicator = LivenessIndicator()
    markers = []
    for tick in range(6):
        snapshot = _Snapshot(steps=100, timestamp=2000.0 + tick, jobs_in_progress=0)
        indicator.update(snapshot, is_alive=True)
        markers.append(str(indicator.marker(snapshot, is_alive=True)))

    assert indicator.stalled is False
    assert len(set(markers)) > 1, "an idle but responsive worker still reads as alive"


def test_liveness_is_inert_when_the_worker_is_not_running() -> None:
    """A stopped worker is not a stalled one."""
    indicator = LivenessIndicator()
    for _ in range(50):
        indicator.update(None, is_alive=False)
    assert indicator.stalled is False


def test_progress_is_reported_not_invented() -> None:
    """Progress comes from the worker; an unreported job is indeterminate rather than guessed."""
    assert job_progress_fraction(_Process(percent=50)) == pytest.approx(0.5)
    assert job_progress_fraction(_Process(current=5, total=20)) == pytest.approx(0.25)
    assert job_progress_fraction(_Process(percent=90, current=1, total=20)) == pytest.approx(0.9)
    assert job_progress_fraction(_Process()) is None
    assert job_progress_fraction(_Process(current=5, total=0)) is None


@pytest.mark.slow
async def test_simple_renders_at_the_eighty_by_twentyfour_floor(tmp_path: Path) -> None:
    """The smallest supported terminal must render Simple without the body scrolling sideways."""
    fake, app = _make_app(tmp_path, level=ExperienceLevel.SIMPLE)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._tick()
        await pilot.pause()
        home = app.query_one(SimpleHomeView)
        assert home.display is True
        assert home.region.width <= 80
        # The level stays legible in the footer rather than costing a content row, and both gateways
        # into the complete shortcut lists survive the footer's truncation at this width.
        footer_row = _rendered_row(app, row=23)
        assert "Simple" in footer_row
        assert "? Help" in footer_row
        assert "palette" in footer_row


@pytest.mark.slow
async def test_density_and_theme_round_trip(tmp_path: Path) -> None:
    """Density and theme are applied and persisted."""
    _fake, app = _make_app(tmp_path, level=ExperienceLevel.ADVANCED)
    store = app._app_state_store
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.set_display_density(DisplayDensity.COMPACT)
        await pilot.pause()
        assert app.screen.has_class("density-compact")
        app.set_theme_name("horde-light")
        await pilot.pause()
        assert app.theme == "horde-light"

    assert store.load().display_density is DisplayDensity.COMPACT
    assert store.load().theme_name == "horde-light"


def test_state_defaults_keep_simple_for_new_installs() -> None:
    """The shipped default is Simple, with no notice owed."""
    state = WorkerAppState()
    assert state.experience_level is ExperienceLevel.SIMPLE
    assert state.needs_experience_introduction is False
    assert state.display_density is DisplayDensity.COMFORTABLE
