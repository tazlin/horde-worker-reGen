"""The Stats tab's startup measurement section shows each card's reading in plain words at Simple and in full above."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console, RenderableType
from textual.app import App, ComposeResult
from textual.widgets import Static

import horde_worker_regen.tui
from horde_worker_regen.app_state import DEFAULT_THEME_NAME
from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.ipc.supervisor_channel import StartupContextReading, StartupContextRun
from horde_worker_regen.tui.design import register_horde_themes
from horde_worker_regen.tui.widgets.startup_measurement import (
    OVERRIDE_FIELD,
    StartupMeasurementView,
    render_measurement_detail,
    render_measurement_simple,
)


def _run(first_mb: int, marginal_mb: int, *, baseline_mb: int | None = 3600) -> StartupContextRun:
    after_context_mb = (baseline_mb or 0) + first_mb
    return StartupContextRun(
        first_context_mb=first_mb,
        marginal_mb=marginal_mb,
        baseline_mb=baseline_mb,
        after_context_mb=after_context_mb,
        after_holder_mb=after_context_mb + marginal_mb if marginal_mb > 0 else None,
    )


def _consistent(device_index: int = 0, *, override_mb: int | None = None) -> StartupContextReading:
    return StartupContextReading(
        device_index=device_index,
        chosen=_run(487, 488),
        verdict="consistent",
        probe_runs=1,
        chosen_run=1,
        first_context_override_mb=override_mb,
    )


def _inconsistent(device_index: int = 0, *, override_mb: int | None = None) -> StartupContextReading:
    runs = [_run(1161, 426), _run(900, 300), _run(1000, 200)]
    return StartupContextReading(
        device_index=device_index,
        chosen=runs[2],
        verdict="inconsistent",
        probe_runs=3,
        chosen_run=3,
        first_context_override_mb=override_mb,
        runs=runs,
    )


def _unchecked(device_index: int = 0) -> StartupContextReading:
    return StartupContextReading(
        device_index=device_index,
        chosen=_run(1161, 0, baseline_mb=None),
        verdict="unchecked",
        probe_runs=1,
        chosen_run=1,
    )


def _text(renderable: RenderableType, *, width: int) -> str:
    console_file = StringIO()
    Console(file=console_file, width=width, force_terminal=False, color_system=None).print(renderable)
    rendered = console_file.getvalue()
    print(rendered)
    return rendered


def _flat(rendered: str) -> str:
    """Return the rendered text with panel borders and wrapping removed, so a sentence reads on one line."""
    stripped = [line.strip().strip("│").strip() for line in rendered.splitlines()]
    return " ".join(part for part in stripped if part)


def test_the_override_field_is_a_config_key() -> None:
    """The dashboard names the setting by a real config field, so a rename cannot leave it naming nothing."""
    assert OVERRIDE_FIELD in reGenBridgeData.model_fields


def test_the_plain_form_words_a_consistent_card() -> None:
    """One card that agreed reads as agreed, with the figure in use, and the card is not numbered."""
    rendered = _flat(_text(render_measurement_simple([_consistent()]), width=80))

    assert "Before it loads a model, each GPU process uses some of the card's memory." in rendered
    assert "The two figures agreed. The worker uses 487 MB." in rendered
    assert "Card 0" not in rendered
    assert "Insights" not in rendered


def test_the_plain_form_words_an_inconsistent_card() -> None:
    """A card that never agreed reads with its try count and the middle figure in use, and points at Insights."""
    rendered = _flat(_text(render_measurement_simple([_inconsistent()]), width=80))

    assert (
        "The two figures did not agree on any of the 3 tries. The worker uses 1000 MB, the middle figure of the tries."
    ) in rendered
    assert "The Insights tab says what to do about figures that did not agree." in rendered


def test_the_plain_form_words_an_unchecked_card() -> None:
    """A card whose figures could not be compared says so and gives the first figure."""
    rendered = _flat(_text(render_measurement_simple([_unchecked()]), width=80))

    assert "The worker could not compare the two figures, so the reading was not checked." in rendered
    assert "The worker uses 1161 MB." in rendered


def test_the_plain_form_names_the_override() -> None:
    """With the setting above 0 the figure in use is the setting's, named as the config file writes it."""
    rendered = _flat(_text(render_measurement_simple([_inconsistent(override_mb=600)]), width=100))

    assert "The worker uses 600 MB, set in vram_per_process_overhead_mb." in rendered
    assert "middle figure" not in rendered


def test_the_detail_form_shows_figures_readings_and_tries() -> None:
    """Each card's figures, raw readings, check and try, then every try of the card that disagreed."""
    rendered = _text(render_measurement_detail([_consistent(0), _inconsistent(1)]), width=140)

    card_zero = next(line for line in rendered.splitlines() if line.strip("│ ").startswith("0 "))
    for cell in ("487", "488", "3,600", "4,087", "4,575", "agreed", "1 of 1"):
        assert cell in card_zero
    card_one = next(line for line in rendered.splitlines() if line.strip("│ ").startswith("1 "))
    for cell in ("1,000", "200", "4,600", "4,800", "did not agree", "3 of 3"):
        assert cell in card_one
    tries = rendered.split("Every try on card 1")[1]
    assert "1,161" in tries
    assert "900" in tries
    assert "yes" in next(line for line in tries.splitlines() if "1,000" in line)
    assert "Every try on card 0" not in rendered


def test_the_detail_form_shows_an_unread_reading_and_the_override() -> None:
    """A reading the probe did not take reads as not read, and a set override is stated once."""
    unchecked = _text(render_measurement_detail([_unchecked()]), width=140)
    overridden = _flat(_text(render_measurement_detail([_consistent(override_mb=600)]), width=140))

    unchecked_row = next(line for line in unchecked.splitlines() if line.strip("│ ").startswith("0 "))
    assert "not measured" in unchecked_row
    assert "not read" in unchecked_row
    assert "not checked" in unchecked_row
    assert "vram_per_process_overhead_mb is 600, so the worker uses 600 MB as the first figure on every card." in (
        overridden
    )


def test_two_cards_are_numbered_in_the_plain_form() -> None:
    """A host with two cards names each card on its line."""
    rendered = _flat(_text(render_measurement_simple([_consistent(0), _inconsistent(1)]), width=120))

    assert "Card 0: The two figures agreed. The worker uses 487 MB." in rendered
    assert "Card 1: The two figures did not agree on any of the 3 tries." in rendered


def test_an_unknown_verdict_from_a_newer_worker_is_shown_as_it_arrives() -> None:
    """A verdict string this dashboard does not know is printed unchanged and does not raise."""
    reading = _consistent().model_copy(update={"verdict": "suspicious"})

    simple = _flat(_text(render_measurement_simple([reading]), width=100))
    detail = _text(render_measurement_detail([reading]), width=140)

    assert 'Check result: "suspicious". The worker uses 487 MB.' in simple
    assert "suspicious" in detail


class _Host(App[None]):
    """Loads the dashboard's stylesheet and themes, so the level rules are the ones the dashboard uses."""

    CSS_PATH = str(Path(horde_worker_regen.tui.__file__).parent / "horde.tcss")

    def __init__(self) -> None:
        super().__init__()
        register_horde_themes(self)
        self.theme = DEFAULT_THEME_NAME

    def compose(self) -> ComposeResult:
        yield StartupMeasurementView()


@pytest.mark.asyncio
async def test_unchanged_readings_do_not_rebuild_and_the_level_picks_the_form() -> None:
    """The section is rebuilt only when the readings change, and the screen's level class shows one form."""
    app = _Host()
    async with app.run_test(size=(80, 40)) as pilot:
        view = app.query_one(StartupMeasurementView)
        simple = view.query_one("#startup-measurement-simple", Static)
        detail = view.query_one("#startup-measurement-detail", Static)
        updates: list[str] = []
        original_update = simple.update

        def counting_update(*args: object, **kwargs: object) -> None:
            updates.append("simple")
            original_update(*args, **kwargs)  # type: ignore[arg-type]

        simple.update = counting_update  # type: ignore[method-assign]

        view.update_readings([_consistent()])
        view.update_readings([_consistent()])
        assert updates == ["simple"]
        view.update_readings([_consistent(override_mb=600)])
        assert updates == ["simple", "simple"]

        app.screen.set_classes(["level-simple"])
        await pilot.pause()
        assert simple.display is True
        assert detail.display is False

        app.screen.set_classes(["level-advanced"])
        await pilot.pause()
        assert simple.display is False
        assert detail.display is True
