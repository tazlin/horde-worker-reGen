"""The Stats tab's card earning section shows the worker's figures in plain words at Simple and in full above it."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console, RenderableType
from textual.app import App, ComposeResult
from textual.widgets import Static

import horde_worker_regen.tui
from horde_worker_regen.app_state import DEFAULT_THEME_NAME
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CardEarningCardRow,
    CardEarningRow,
    CardEarningSnapshot,
    CardEarningWindowRestart,
    WholeCardResidencyStatus,
)
from horde_worker_regen.tui.design import register_horde_themes
from horde_worker_regen.tui.widgets.earning import CardEarningView, render_earning_detail, render_earning_simple
from horde_worker_regen.tui.widgets.overview import OverviewView

_NOW = 1_800_000_000.0
_WALL = 9 * 3600.0
_MIN_JOBS = 17
"""A floor that differs from the worker's, so the text is shown to come from the snapshot."""


def _row(
    model: str,
    *,
    seconds: float,
    rate: float | None,
    rest: float | None,
    verdict: str,
    jobs: int,
    device_index: int = 0,
    claim_seconds: float = 0.0,
) -> CardEarningRow:
    return CardEarningRow(
        device_index=device_index,
        model=model,
        model_kudos_per_second=rate,
        model_band_low=None if rate is None else rate - 0.1,
        model_band_high=None if rate is None else rate + 0.1,
        rest_kudos_per_second=rest,
        rest_band_low=None if rest is None else rest - 0.05,
        rest_band_high=None if rest is None else rest + 0.05,
        jobs=jobs,
        model_buckets=36,
        rest_buckets=35,
        attributed_card_seconds=seconds,
        claim_seconds=claim_seconds,
        verdict=verdict,
    )


def _worked_rows(device_index: int = 0) -> list[CardEarningRow]:
    """Return the rows in the snapshot's order, the most attributed seconds first."""
    return [
        _row(
            "Qwen-Image",
            seconds=9000.0,
            rate=0.40,
            rest=0.95,
            verdict="below_the_rest",
            jobs=120,
            claim_seconds=3240.0,
            device_index=device_index,
        ),
        _row(
            "Flux.1-Schnell", seconds=8000.0, rate=1.20, rest=0.70, verdict="pays", jobs=400, device_index=device_index
        ),
        _row("Krea2", seconds=6000.0, rate=0.80, rest=0.85, verdict="pays", jobs=90, device_index=device_index),
        _row(
            "Z-Image-Turbo", seconds=4000.0, rate=1.00, rest=0.84, verdict="pays", jobs=150, device_index=device_index
        ),
        _row(
            "AlbedoBase XL",
            seconds=1000.0,
            rate=0.90,
            rest=0.86,
            verdict="unmeasured",
            jobs=6,
            device_index=device_index,
        ),
    ]


def _worked_snapshot(
    *,
    restart: CardEarningWindowRestart | None = None,
    rows_omitted: int = 0,
    attributed_seconds: float = 28000.0,
) -> CardEarningSnapshot:
    return CardEarningSnapshot(
        rows=_worked_rows(),
        rows_omitted=rows_omitted,
        cards=[
            CardEarningCardRow(
                device_index=0,
                wall_seconds=_WALL,
                attributed_seconds=attributed_seconds,
                unattributed_seconds=_WALL - attributed_seconds,
            ),
        ],
        window_started_at=_NOW - _WALL,
        last_restart=restart,
        verdict_min_jobs=_MIN_JOBS,
    )


def _text(renderable: RenderableType, *, width: int) -> str:
    console_file = StringIO()
    Console(file=console_file, width=width, force_terminal=False, color_system=None).print(renderable)
    rendered = console_file.getvalue()
    print(rendered)
    return rendered


def _line_with(rendered: str, needle: str) -> str:
    return next(line for line in rendered.splitlines() if needle in line)


def _earning_lines(rendered: str) -> str:
    """Return the part of the plain form above the time shares."""
    return rendered.split("Where the card's time went")[0]


def test_the_plain_form_orders_models_by_rate_and_words_each_verdict() -> None:
    """Measured models come first, best earner on top, and each verdict phrase follows from the row's figures."""
    rendered = _text(render_earning_simple(_worked_snapshot(), now=_NOW), width=80)
    earning = _earning_lines(rendered)

    order = [earning.index(model) for model in ("Flux.1-Schnell", "Z-Image-Turbo", "Krea2", "Qwen-Image", "Albedo")]
    assert order == sorted(order)
    assert "What each model earns on this card" in rendered
    assert "earning well" in _line_with(earning, "Flux.1-Schnell")
    assert "earning well" in _line_with(earning, "Z-Image-Turbo")
    assert "about the same as the rest" in _line_with(earning, "Krea2")
    assert "below the rest" in _line_with(earning, "Qwen-Image")
    assert f"not measured yet (6 of {_MIN_JOBS} jobs)" in _line_with(earning, "AlbedoBase XL")


def test_the_bars_are_scaled_to_the_best_earner_on_the_card() -> None:
    """The best earner fills the bar and every other bar is its rate's share of that, without colour."""
    earning = _earning_lines(_text(render_earning_simple(_worked_snapshot(), now=_NOW), width=80))

    lengths = {
        model: _line_with(earning, model).count("█")
        for model in ("Flux.1-Schnell", "Z-Image-Turbo", "Krea2", "Qwen-Image", "AlbedoBase XL")
    }
    assert lengths == {"Flux.1-Schnell": 12, "Z-Image-Turbo": 10, "Krea2": 8, "Qwen-Image": 4, "AlbedoBase XL": 0}
    assert "·" in _line_with(earning, "AlbedoBase XL")


def test_the_plain_form_fits_the_narrowest_terminal() -> None:
    """An 80-column terminal shows every line whole, bar and phrase included."""
    rendered = _text(render_earning_simple(_worked_snapshot(), now=_NOW), width=80)

    assert all(len(line) <= 80 for line in rendered.splitlines())
    assert "about the same as the rest" in rendered


def test_the_footer_names_the_span_and_the_settings_that_restarted_it() -> None:
    """The footer says how long the window covers and which settings changed at the last restart."""
    restart = CardEarningWindowRestart(at=_NOW - _WALL, changed_settings=["max_threads", "queue_size"])
    rendered = _text(render_earning_simple(_worked_snapshot(restart=restart), now=_NOW), width=120)

    assert "Measured over the last 9 hours." in rendered
    assert "Restarted when max_threads and queue_size changed." in rendered


def test_the_footer_without_a_known_reason_or_restart() -> None:
    """A restart found at load has no settings named, and a window that never restarted says nothing of one."""
    found_at_load = CardEarningWindowRestart(at=_NOW - _WALL, changed_settings=None)
    with_unknown = _text(render_earning_simple(_worked_snapshot(restart=found_at_load), now=_NOW), width=120)
    without = _text(render_earning_simple(_worked_snapshot(), now=_NOW), width=120)

    assert "Restarted because the settings differed from the last run's." in with_unknown
    assert "Restarted" not in without
    assert "Measured over the last 9 hours." in without


def test_the_detailed_form_shows_every_figure() -> None:
    """Advanced lists the model's rate and band, the rest's rate and band, jobs, readings and the verdict."""
    rendered = _text(render_earning_detail(_worked_snapshot(), now=_NOW), width=160)
    flux = _line_with(rendered, "Flux.1-Schnell")

    for figure in ("1.20", "1.10 to 1.30", "0.70", "0.65 to 0.75", "400", "36/35", "earning well"):
        assert figure in flux
    assert f"not measured yet (6 of {_MIN_JOBS} jobs)" in _line_with(rendered, "AlbedoBase XL")
    assert "95% confidence" in rendered


def test_the_time_shares_include_idle_at_both_levels() -> None:
    """Each model's share of the card's wall is shown, then the idle share, and Advanced adds the seconds."""
    simple = _text(render_earning_simple(_worked_snapshot(), now=_NOW), width=80)
    shares = simple.split("Where the card's time went")[1]

    assert "28%" in _line_with(shares, "Qwen-Image")
    assert "25%" in _line_with(shares, "Flux.1-Schnell")
    assert "14%" in _line_with(shares, "idle")
    assert _line_with(shares, "Qwen-Image").count("█") == 7
    assert _line_with(shares, "AlbedoBase XL").count("█") == 1

    detail = _text(render_earning_detail(_worked_snapshot(), now=_NOW), width=160)
    time_table = detail.split("Where the card's time went")[1]
    assert "2h 30m 00s" in _line_with(time_table, "Qwen-Image")
    assert "1h 13m 20s" in _line_with(time_table, "idle")


def test_the_claim_share_is_shown_beside_the_verdict() -> None:
    """The time a whole-card claim stood for a model is its share of the card's wall, next to its verdict."""
    simple = _text(render_earning_simple(_worked_snapshot(), now=_NOW), width=200)
    assert "Asked the horde only for Qwen-Image for 10% of the card's time. It is below the rest." in simple

    detail = _text(render_earning_detail(_worked_snapshot(), now=_NOW), width=160)
    claims = detail.split("Time spent offering one model only")[1]
    claim_line = _line_with(claims, "Qwen-Image")
    assert "10%" in claim_line
    assert "54m 00s" in claim_line
    assert "below the rest" in claim_line


def test_an_empty_window_says_what_it_waits_for() -> None:
    """With no figures the section says nothing is measured yet and that finished jobs are what it needs."""
    for render in (render_earning_simple, render_earning_detail):
        rendered = _text(render(CardEarningSnapshot(), now=_NOW), width=80)
        assert "Not measured yet." in rendered
        assert "finished job" in rendered


def test_omitted_rows_are_stated_and_their_time_is_shown() -> None:
    """Rows cut by the snapshot's bound are counted in the footer and their card time is not lost."""
    rendered = _text(
        render_earning_simple(_worked_snapshot(rows_omitted=3, attributed_seconds=29000.0), now=_NOW),
        width=120,
    )

    assert "And 3 more models with less card time." in rendered
    assert "3%" in _line_with(rendered, "other models")


def test_two_cards_get_a_block_each() -> None:
    """A multi-card host shows one block per card, named by its index."""
    earning = CardEarningSnapshot(
        rows=[*_worked_rows(0), *_worked_rows(1)[:2]],
        cards=[
            CardEarningCardRow(
                device_index=0, wall_seconds=_WALL, attributed_seconds=28000.0, unattributed_seconds=4400.0
            ),
            CardEarningCardRow(
                device_index=1, wall_seconds=_WALL, attributed_seconds=17000.0, unattributed_seconds=15400.0
            ),
        ],
        window_started_at=_NOW - _WALL,
        verdict_min_jobs=_MIN_JOBS,
    )

    for render in (render_earning_simple, render_earning_detail):
        rendered = _text(render(earning, now=_NOW), width=160)
        assert "What each model earns on card 0" in rendered
        assert "What each model earns on card 1" in rendered
        second_card = rendered.split("What each model earns on card 1")[1]
        assert "Krea2" not in second_card
        assert "Flux.1-Schnell" in second_card


def test_an_unknown_verdict_from_a_newer_worker_is_shown_as_it_arrives() -> None:
    """A verdict string this dashboard does not know is printed unchanged and does not raise."""
    earning = _worked_snapshot()
    earning.rows[1] = earning.rows[1].model_copy(update={"verdict": "pays_handsomely"})

    for render in (render_earning_simple, render_earning_detail):
        assert "pays_handsomely" in _line_with(_text(render(earning, now=_NOW), width=160), "Flux.1-Schnell")


def test_the_residency_panel_uses_the_same_words_for_the_claim() -> None:
    """The Overview's claim value reads as the Stats tab reads the same model."""
    residency = WholeCardResidencyStatus(
        pop_claim_model="Flux.1-Schnell",
        pop_claim_card_index=0,
        pop_claim_verdict="pays",
        pop_claim_remaining_seconds=60.0,
    )

    with_figures = _text(OverviewView._render_residency_panel(residency, earning=_worked_snapshot()), width=120)
    without_figures = _text(OverviewView._render_residency_panel(residency), width=120)

    assert "earning well" in _line_with(with_figures, "Claim value")
    assert "not below the rest" in _line_with(without_figures, "Claim value")


class _Host(App[None]):
    """Loads the dashboard's stylesheet and themes, so the level rules are the ones the dashboard uses."""

    CSS_PATH = str(Path(horde_worker_regen.tui.__file__).parent / "horde.tcss")

    def __init__(self) -> None:
        super().__init__()
        register_horde_themes(self)
        self.theme = DEFAULT_THEME_NAME

    def compose(self) -> ComposeResult:
        yield CardEarningView()


@pytest.mark.asyncio
async def test_an_unchanged_snapshot_does_not_rebuild_and_the_level_picks_the_form() -> None:
    """The figures are rebuilt only when they change, and the screen's level class shows one form."""
    app = _Host()
    async with app.run_test(size=(80, 40)) as pilot:
        view = app.query_one(CardEarningView)
        simple = view.query_one("#card-earning-simple", Static)
        detail = view.query_one("#card-earning-detail", Static)
        updates: list[str] = []
        original_update = simple.update

        def counting_update(*args: object, **kwargs: object) -> None:
            updates.append("simple")
            original_update(*args, **kwargs)  # type: ignore[arg-type]

        simple.update = counting_update  # type: ignore[method-assign]

        view.update_earning(_worked_snapshot(), now=_NOW)
        view.update_earning(_worked_snapshot(), now=_NOW + 0.5)
        assert updates == ["simple"]
        view.update_earning(_worked_snapshot(rows_omitted=1), now=_NOW + 1.0)
        assert updates == ["simple", "simple"]

        app.screen.set_classes(["level-simple"])
        await pilot.pause()
        assert simple.display is True
        assert detail.display is False

        app.screen.set_classes(["level-advanced"])
        await pilot.pause()
        assert simple.display is False
        assert detail.display is True
