"""e2e tests for the logs view's scroll behavior: don't yank the viewport when the user has scrolled up."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static

from horde_worker_regen.tui.widgets.logs import LogsView


class _FakeFollower:
    """A LogFollower stand-in that yields pre-canned batches of lines, then nothing."""

    def __init__(self, batches: list[list[str]]) -> None:
        self._batches = iter(batches)

    def poll(self) -> list[str]:
        return next(self._batches, [])


class _LogsHost(App[None]):
    """Hosts the logs view in isolation."""

    def compose(self) -> ComposeResult:
        yield LogsView()


def _batch(prefix: str, count: int) -> list[str]:
    """A batch of well-formed loguru-style lines (so level parsing keeps them)."""
    return [f"2026-06-20 00:00:00 | INFO | {prefix} {i}" for i in range(count)]


@pytest.mark.e2e
async def test_logs_pause_autoscroll_when_scrolled_up_and_resume_on_jump(monkeypatch: pytest.MonkeyPatch) -> None:
    """While scrolled up, new lines neither yank the view nor are lost; End jumps back and resumes."""
    # Keep the test off any real log files the discovery scan might find in the working tree.
    monkeypatch.setattr("horde_worker_regen.tui.widgets.logs.discover_bridge_logs_grouped", dict)

    app = _LogsHost()
    async with app.run_test(size=(80, 20)) as pilot:
        view = app.query_one(LogsView)
        log = view.query_one("#log-output", RichLog)
        hint = view.query_one("#log-scroll-hint", Static)
        view._current_path = Path("fake.log")

        async def settle() -> None:
            for _ in range(5):
                await pilot.pause()

        # First batch arrives while pinned to the bottom: the view follows the tail. (Whether this batch
        # is drained by the manual call or the periodic interval, the at-bottom outcome is the same.)
        view._follower = _FakeFollower([_batch("line", 100)])  # type: ignore[assignment]
        view._poll()
        await settle()
        assert log.auto_scroll is True
        assert hint.display is False

        # The operator scrolls up to read history.
        log.scroll_to(y=0, animate=False)
        await settle()
        assert log.is_vertical_scroll_end is False, "precondition: the view must be scrolled away from the bottom"

        # A new batch arrives while scrolled up: it must not move the viewport, and the unseen lines are
        # tallied and surfaced. The batch is drained exactly once (by whichever poll wins), so the count
        # is deterministic regardless of the periodic interval.
        view._follower = _FakeFollower([_batch("more", 50)])  # type: ignore[assignment]
        view._poll()
        await settle()
        assert log.auto_scroll is False
        assert view._unseen_below == 50
        assert hint.display is True

        # Jumping to latest resumes following and clears the unseen tally.
        view.action_jump_to_latest()
        await settle()
        assert log.auto_scroll is True
        assert view._unseen_below == 0
        assert hint.display is False
