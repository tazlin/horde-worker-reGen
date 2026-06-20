"""Tests for the shared presentation helpers: job-id colouring and baseline abbreviation."""

from __future__ import annotations

from horde_worker_regen.tui.formatters import (
    _JOB_ID_PALETTE,
    job_id_color,
    job_id_text,
    short_baseline,
    short_job_id,
)


def test_short_job_id_takes_the_first_uuid_group() -> None:
    """The displayed prefix is the first hyphen-delimited group of the id."""
    assert short_job_id("7f3a1c9e-4b2c-4d6e-8a1f-0c2b07d49abc") == "7f3a1c9e"
    assert short_job_id(None) == "-"
    assert short_job_id("") == "-"


def test_job_id_colour_is_deterministic_and_in_palette() -> None:
    """The same id always maps to the same palette colour."""
    job_id = "9c2b07d4-1111-2222-3333-444455556666"
    colour = job_id_color(job_id)
    assert colour in _JOB_ID_PALETTE
    assert job_id_color(job_id) == colour


def test_job_id_colour_keys_on_the_first_group_only() -> None:
    """Two ids sharing a first group share a colour; a different first group can differ."""
    same_a = job_id_color("aabbccdd-0000-1111-2222-333344445555")
    same_b = job_id_color("aabbccdd-9999-8888-7777-666655554444")
    assert same_a == same_b


def test_job_id_colour_handles_non_uuid_ids() -> None:
    """A non-hex id still colours (via a character-sum fallback) rather than raising."""
    colour = job_id_color("mock-queued-3")
    assert colour in _JOB_ID_PALETTE


def test_job_id_colour_none_is_grey() -> None:
    """An absent id is a neutral grey, not a palette colour."""
    assert job_id_color(None) == "grey50"


def test_job_id_text_carries_the_palette_colour() -> None:
    """The rich renderable styles the prefix with the deterministic colour."""
    job_id = "7f3a1c9e-4b2c-4d6e-8a1f-0c2b07d49abc"
    text = job_id_text(job_id)
    assert text.plain == "7f3a1c9e"
    assert str(text.style) == job_id_color(job_id)


def test_short_baseline_abbreviates_known_baselines() -> None:
    """Known baselines collapse to compact labels for a narrow column."""
    assert short_baseline("stable_diffusion_xl") == "SDXL"
    assert short_baseline("stable_diffusion_1") == "SD1.5"
    assert short_baseline("flux_1") == "Flux"
    assert short_baseline(None) == "-"


def test_short_baseline_falls_back_for_unknown_baselines() -> None:
    """An unknown baseline reads as a cleaned-up name rather than being dropped."""
    assert short_baseline("some_new_baseline") == "some new baseline"
