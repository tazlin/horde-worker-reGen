"""Tests for the StatusReporter's worker-info lines."""

from __future__ import annotations

from unittest.mock import MagicMock

from loguru import logger

from horde_worker_regen.reporting.status_reporter import StatusReporter


def _feature_line(
    *,
    allow_post_processing: bool,
    post_processing_offered: bool | None,
    post_processing_offer_withheld_reason: str = "",
) -> str:
    """Run _print_worker_info and return the line carrying the feature flags."""
    bridge_data = MagicMock()
    bridge_data.max_power = 8
    bridge_data.allow_post_processing = allow_post_processing
    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(message.record["message"]), level="DEBUG")
    try:
        StatusReporter(0.0, 0.0)._print_worker_info(
            bridge_data,
            None,
            1,
            0,
            0,
            0,
            post_processing_offered=post_processing_offered,
            post_processing_offer_withheld_reason=post_processing_offer_withheld_reason,
        )
    finally:
        logger.remove(sink_id)
    return next(line for line in captured if "allow_post_processing:" in line)


def test_agreeing_offer_leaves_the_configured_echo_unchanged() -> None:
    """An offer matching the configured flag adds nothing to the echo."""
    line = _feature_line(allow_post_processing=True, post_processing_offered=True)

    assert "allow_post_processing: True | " in line
    assert "offered:" not in line


def test_no_pop_built_yet_leaves_the_configured_echo_unchanged() -> None:
    """Before the first pop there is no wire value to report."""
    line = _feature_line(allow_post_processing=True, post_processing_offered=None)

    assert "offered:" not in line


def test_withheld_offer_is_appended_with_its_reason() -> None:
    """The configured flag says True while the pop carried False, so the line says both and why."""
    line = _feature_line(
        allow_post_processing=True,
        post_processing_offered=False,
        post_processing_offer_withheld_reason="the post-processing lane is paused off the GPU",
    )

    assert "allow_post_processing: True (offered: False, the post-processing lane is paused off the GPU) | " in line
