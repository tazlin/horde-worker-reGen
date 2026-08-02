"""Behavioural cover for the progressive experience levels.

These assert the properties the design rests on: navigation keeps its shape at every level, a changed
default is announced rather than applied silently, settings a level does not show survive a save, and
the liveness indicator draws only on signals a wedged worker cannot produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from textual.widgets import Collapsible, TabbedContent, TabPane

from horde_worker_regen.app_state import (
    APP_STATE_SCHEMA_VERSION,
    AppStateStore,
    DisplayDensity,
    ExperienceLevel,
    OnboardingChoice,
    OverviewViewMode,
    WorkerAppState,
)
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.app import HordeWorkerTUI
from horde_worker_regen.tui.config_form import CONFIG_FIELDS
from horde_worker_regen.tui.widgets.config_editor import ConfigEditorView, _subtab_id
from horde_worker_regen.tui.widgets.experience import ExperienceIntroductionModal
from horde_worker_regen.tui.widgets.simple import (
    LivenessIndicator,
    SimpleActivityView,
    SimpleHomeView,
    SimpleModelStatusView,
    TabPrimer,
    job_progress_fraction,
)
from horde_worker_regen.tui.widgets.stats import StatsView
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


def _screen_text(app: HordeWorkerTUI) -> str:
    """Return the whole composited screen as plain text, as the terminal would show it."""
    strips = app.screen._compositor.render_strips()
    return "\n".join("".join(segment.text for segment in strip) for strip in strips)


def _rendered_row(app: HordeWorkerTUI, *, row: int) -> str:
    """Return one row of the composited screen as plain text, as the terminal would show it."""
    strips = app.screen._compositor.render_strips()
    return "".join(segment.text for segment in strips[row]).rstrip()


class _Process:
    """A stand-in carrying only the progress fields the Simple helpers read."""

    def __init__(
        self,
        steps: int = 0,
        percent: int | None = None,
        current: int | None = None,
        total: int | None = None,
        heartbeat: float = 0.0,
    ) -> None:
        """Store the reported progress counters."""
        self.heartbeats_inference_steps = steps
        self.last_heartbeat_percent_complete = percent
        self.last_current_step = current
        self.last_total_steps = total
        self.last_heartbeat_timestamp = heartbeat


class _Snapshot:
    """A stand-in carrying only the fields the liveness indicator reads.

    ``heartbeat`` is the child's last heartbeat of any kind and ``steps`` its sampling counter. The
    worker advances the two independently, and :class:`LivenessIndicator` reads both.
    """

    def __init__(
        self,
        *,
        steps: int,
        timestamp: float,
        jobs_in_progress: int,
        heartbeat: float = 0.0,
    ) -> None:
        """Store one frame of worker state."""
        self.processes = [_Process(steps, heartbeat=heartbeat)]
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

        # Both directions are asserted: the palette offers every destination and invents none. A
        # one-way check would pass while a newly composed tab went unreachable by name.
        assert [tab_id for _label, tab_id in app.destinations()] == ALL_TABS


@pytest.mark.slow
async def test_density_mode_does_not_reveal_what_the_level_withholds(tmp_path: Path) -> None:
    """The density mode and the experience level constrain the same sub-tabs and must not fight.

    Written independently, the two settings fight: whichever runs last wins, so cycling the view mode in
    Simple re-shows every tuning page that level withholds, and choosing a level while thin undoes thin.
    A single arbiter reading both inputs makes the order irrelevant.
    """
    _fake, app = _make_app(tmp_path, level=ExperienceLevel.SIMPLE)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(ConfigEditorView)
        subtabs = editor.query_one("#config-subtabs", TabbedContent)
        withheld = _subtab_id("Throughput")

        assert not subtabs.get_tab(withheld).display

        for _ in range(len(OverviewViewMode)):
            app.action_cycle_view_mode()
            await pilot.pause()
            assert not subtabs.get_tab(withheld).display, "cycling density must not reveal a withheld page"

        app._commit_experience_level(ExperienceLevel.ADVANCED)
        await pilot.pause()
        assert subtabs.get_tab(withheld).display


@pytest.mark.slow
async def test_simple_withholds_only_shortcuts_whose_subject_is_hidden(tmp_path: Path) -> None:
    """Simple must not offer a key that acts on an operator view it does not display.

    Customising the Overview, revealing elements hidden from it, and cycling its density all address a
    widget Simple replaces. Advertising them promises a control that would appear to do nothing. Every
    other shortcut, and every destination, stays available.
    """
    _fake, app = _make_app(tmp_path, level=ExperienceLevel.SIMPLE)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        simple_actions = {action for _keys, _description, action in app.keyboard_actions()}
        assert not (simple_actions & app._OPERATOR_VIEW_ACTIONS)
        assert "start_stop_worker" in simple_actions, "the primary action must survive"

        app._commit_experience_level(ExperienceLevel.ADVANCED)
        await pilot.pause()
        advanced_actions = {action for _keys, _description, action in app.keyboard_actions()}
        assert advanced_actions >= app._OPERATOR_VIEW_ACTIONS


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
        for label, _tab_id in app.destinations():
            assert f"Go to {label}" in titles

        # The shortcut is shown alongside the command, so the palette teaches the key.
        assert any("(f3)" in title for title in titles)

        # Exactly one command-palette key: Textual pins its own, so declaring another duplicates it.
        assert sum(1 for _k, _d, action in actions if action == "command_palette") == 0


def test_liveness_freezes_when_the_worker_wedges() -> None:
    """A wedged worker must not be able to satisfy the indicator meant to expose it.

    Both signals a healthy supervisor produces are supplied here: the snapshot timestamp advances every
    frame, and the dashboard's render loop is running. Only the child is dead. An indicator driven by
    either signal would keep animating over it.
    """
    indicator = LivenessIndicator()
    markers = []
    for tick in range(60):
        snapshot = _Snapshot(steps=100, timestamp=1000.0 + tick, jobs_in_progress=1, heartbeat=900.0)
        indicator.update(snapshot, is_alive=True)
        markers.append(str(indicator.marker(snapshot, is_alive=True)))

    assert len(set(markers)) == 1, "the indicator must not animate while the child reports nothing"


def test_liveness_does_not_read_a_model_load_as_a_wedge() -> None:
    """A busy non-sampling stage is not a stall, and must keep animating.

    The worker resets ``heartbeats_inference_steps`` to zero on every heartbeat that is not a sampling
    step, so it is pinned at zero for the whole of a model load or a post-processing pass, and those
    routinely run for tens of seconds. An indicator watching only that counter freezes during ordinary
    healthy operation, and any stall verdict built on it reports a fault to a contributor whose worker
    is working.
    """
    indicator = LivenessIndicator()
    markers = []
    for tick in range(30):
        snapshot = _Snapshot(steps=0, timestamp=1000.0 + tick, jobs_in_progress=1, heartbeat=1000.0 + tick)
        indicator.update(snapshot, is_alive=True)
        markers.append(str(indicator.marker(snapshot, is_alive=True)))

    assert len(set(markers)) > 1, "a heartbeating child mid-load must still read as working"


def test_liveness_advances_only_on_real_inference_progress() -> None:
    """The indicator advances when the worker's own step counter advances."""
    indicator = LivenessIndicator()
    markers = []
    for tick in range(6):
        snapshot = _Snapshot(steps=100 + tick * 7, timestamp=1000.0 + tick, jobs_in_progress=1)
        indicator.update(snapshot, is_alive=True)
        markers.append(str(indicator.marker(snapshot, is_alive=True)))

    assert len(set(markers)) > 1


def test_liveness_defers_to_the_health_report_for_the_alarm() -> None:
    """The glyph escalates on the health report's verdict, which this class takes rather than forms."""
    indicator = LivenessIndicator()
    snapshot = _Snapshot(steps=100, timestamp=1000.0, jobs_in_progress=1, heartbeat=1000.0)
    indicator.update(snapshot, is_alive=True)

    assert str(indicator.marker(snapshot, is_alive=True, concerning=False)) != "!"
    assert str(indicator.marker(snapshot, is_alive=True, concerning=True)) == "!"


def test_liveness_shows_an_idle_worker_as_alive() -> None:
    """With no work in hand the step counter is legitimately static, which is not a stall."""
    indicator = LivenessIndicator()
    markers = []
    for tick in range(6):
        snapshot = _Snapshot(steps=100, timestamp=2000.0 + tick, jobs_in_progress=0)
        indicator.update(snapshot, is_alive=True)
        markers.append(str(indicator.marker(snapshot, is_alive=True)))

    assert len(set(markers)) > 1, "a responsive idle worker must still breathe"
    assert len(set(markers)) > 1, "an idle but responsive worker still reads as alive"


def test_liveness_is_inert_when_the_worker_is_not_running() -> None:
    """A stopped worker reads as stopped rather than as a fault."""
    indicator = LivenessIndicator()
    for _ in range(50):
        indicator.update(None, is_alive=False)
    assert str(indicator.marker(None, is_alive=False)) == "○"


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


@pytest.mark.slow
@pytest.mark.parametrize(
    ("level", "expect_visible"),
    [(ExperienceLevel.ADVANCED, False), (ExperienceLevel.DEVELOPER, True)],
)
async def test_developer_shows_the_safety_levers_advanced_does_not(
    tmp_path: Path,
    level: ExperienceLevel,
    expect_visible: bool,
) -> None:
    """Developer earns its warning by showing the internal safety levers Advanced holds back.

    The tier covers the fuses, floors and breakers that protect the worker from itself: hung-process
    timeouts, the VRAM and RAM budget, the fault breakers. Set carelessly they either remove a protection
    silently or fire on healthy work, which is a different hazard from a setting that trades throughput.
    Without a populated tier the two levels render identically and the Developer warning promises
    nothing.
    """
    dangerous_keys = {field.key for field in CONFIG_FIELDS if field.risk_level == "dangerous"}
    assert dangerous_keys, "the tier has to be populated for the level to mean anything"

    _fake, app = _make_app(tmp_path, level=level)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._commit_experience_level(level)
        await pilot.pause()
        editor = app.query_one(ConfigEditorView)
        tagged = editor.query(".field-dangerous")
        assert len(tagged) >= len(dangerous_keys), "every dangerous field carries the gating class"
        for widget in tagged:
            assert widget.display is expect_visible, f"{widget} at {level}"


@pytest.mark.slow
async def test_a_fully_withheld_section_takes_its_heading_with_it(tmp_path: Path) -> None:
    """A section whose fields are all withheld must not leave a titled, empty block behind.

    The heading, rule and guidance are composed as siblings of the fields, so gating only the fields
    renders a labelled section with nothing under it.
    """
    from textual.widgets import Label

    _fake, app = _make_app(tmp_path, level=ExperienceLevel.ADVANCED)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(ConfigEditorView)
        gated_headings = editor.query("Label.config-section.field-dangerous")
        assert gated_headings, "a section of entirely Developer-only fields tags its heading"
        assert not any(heading.display for heading in gated_headings)

        # An ungated heading is unaffected, so this gates by tier rather than hiding headings wholesale.
        plain_headings = [
            heading
            for heading in editor.query(Label)
            if heading.has_class("config-section") and not heading.has_class("field-dangerous")
        ]
        assert any(heading.display for heading in plain_headings)

        app._commit_experience_level(ExperienceLevel.DEVELOPER)
        await pilot.pause()
        assert all(heading.display for heading in gated_headings)


@pytest.mark.slow
async def test_every_technical_destination_is_explained(tmp_path: Path) -> None:
    """No destination that keeps its operator widget is left without a Simple framing.

    Overview, Live and Downloads swap in a bespoke Simple view instead, so they are the only tabs that
    need no primer. Anything else reaching Simple unframed is a page of operator vocabulary with nothing
    to orient a first-time contributor.
    """
    bespoke = {"tab-overview", "tab-live", "tab-downloads", "tab-config"}
    _fake, app = _make_app(tmp_path, level=ExperienceLevel.SIMPLE)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tabs = app.query_one("#main-tabs", TabbedContent)
        for pane in tabs.query(TabPane):
            if not pane.id or not pane.id.startswith("tab-") or pane.id in bespoke:
                continue
            assert pane.query(TabPrimer), f"{pane.id} reaches Simple with no framing"


@pytest.mark.slow
async def test_the_primer_reports_live_figures_rather_than_illustrations(tmp_path: Path) -> None:
    """The worked example reads the worker's own numbers, and says so when it has none.

    An example carrying invented figures drifts from the table beneath it the moment either changes, and
    teaches a contributor to read a number that is not theirs. A figure the worker does not report has to
    be named as absent for the same reason.
    """
    fake, app = _make_app(tmp_path, level=ExperienceLevel.SIMPLE)
    fake.latest_snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="TestWorker", worker_version="0.0.0"),
        num_jobs_submitted=1204,
        num_jobs_faulted=7,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        app.query_one("#main-tabs", TabbedContent).active = "tab-stats"
        app._tick()
        for _ in range(4):
            await pilot.pause()
        # Read the composited screen: what a contributor sees, not what the widget was handed.
        rendered = _screen_text(app)
        assert "1,204" in rendered, "the example quotes the worker's own submitted count"
        # Kudos/hr is absent from this snapshot, so it must say so rather than show a fabricated zero.
        # Reporting and inventing are the two directions the example has to get right.
        assert "not yet" in rendered


@pytest.mark.slow
async def test_folding_the_example_gives_the_screen_back(tmp_path: Path) -> None:
    """The example must be dismissible, because it is longer than the 80x24 floor can spare.

    Expanded it fills the screen, which is acceptable for something meant to be read once and is why it
    opens that way. Collapsing it has to return the operator widget the framing describes.
    """
    _fake, app = _make_app(tmp_path, level=ExperienceLevel.SIMPLE)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one("#main-tabs", TabbedContent).active = "tab-stats"
        await pilot.pause()
        stats = app.query_one(StatsView)
        collapsible = app.query_one("#tab-stats", TabPane).query_one(TabPrimer).query_one(Collapsible)

        assert not collapsible.collapsed, "the example opens expanded so it is read before dismissed"
        collapsible.collapsed = True
        for _ in range(4):
            await pilot.pause()
        assert stats.region.height > 0, "collapsing must return the widget the framing describes"
