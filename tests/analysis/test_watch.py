"""Unit tests for the live-watch change detection (pure passes; no polling loop)."""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.watch import WatchState, watch_pass

_STARTUP = "2026-06-24 18:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process"
_OOM = "2026-06-24 18:00:10.000 | ERROR | x:y:1 - CUDA out of memory. Tried to allocate 2.00 GiB"


def _bundle(tmp_path: Path, text: str) -> LogBundle:
    (tmp_path / "bridge.log").write_text(text, encoding="utf-8")
    return LogBundle.from_path(tmp_path)


def test_alerts_once_then_stays_quiet(tmp_path: Path) -> None:
    """A finding alerts on the pass it first appears, and not again while unchanged."""
    bundle = _bundle(tmp_path, _STARTUP + "\n" + _OOM + "\n")
    alerts, state = watch_pass(bundle, WatchState())
    assert any("out-of-memory" in a.lower() or "oom" in a.lower() for a in alerts)
    alerts_again, _ = watch_pass(bundle, state)
    assert alerts_again == []


def test_new_session_resets_and_announces(tmp_path: Path) -> None:
    """When a newer session appears, the watcher announces it and re-evaluates from a clean baseline."""
    bundle = _bundle(tmp_path, _STARTUP + "\n" + _OOM + "\n")
    _, state = watch_pass(bundle, WatchState())
    bundle = _bundle(
        tmp_path,
        _STARTUP + "\n" + _OOM + "\n2026-06-24 18:05:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process\n",
    )
    alerts, _ = watch_pass(bundle, state)
    assert any("session #1 started" in a for a in alerts)
