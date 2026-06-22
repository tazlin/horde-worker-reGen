"""Tests for the Windows worker-host tray icon (``tui/tray.py``).

The real ``pystray`` is never required here: the unsupported path must be an inert no-op, and the
supported path is exercised with a fake pystray so the wiring is verified without a display.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.tui import tray


class _FakeIcon:
    """A stand-in for ``pystray.Icon`` that records run/stop without touching a real notification area."""

    def __init__(self, name: str, image: object, title: str | None = None, menu: object = None) -> None:
        self.name = name
        self.title = title
        self.menu = menu
        self.ran = False
        self.stopped = False

    def run(self) -> None:
        self.ran = True

    def stop(self) -> None:
        self.stopped = True


class _FakeMenu:
    SEPARATOR = object()

    def __init__(self, *items: object) -> None:
        self.items = items


class _FakeMenuItem:
    def __init__(self, text: object, action: object, **kwargs: object) -> None:
        self.text = text
        self.action = action
        self.kwargs = kwargs


class _FakePystray:
    Icon = _FakeIcon
    Menu = _FakeMenu
    MenuItem = _FakeMenuItem


def _enable_fake_tray(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make :func:`tray.tray_supported` true and back the module with the fake pystray."""
    monkeypatch.setattr(tray, "_TRAY_IMPORT_OK", True)
    monkeypatch.setattr(tray.sys, "platform", "win32")
    monkeypatch.setattr(tray, "pystray", _FakePystray)
    monkeypatch.setattr(tray, "_create_icon_image", lambda: object())


def test_start_is_a_noop_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the tray unsupported, start() does nothing and never raises (the host runs without an icon)."""
    monkeypatch.setattr(tray, "tray_supported", lambda: False)
    worker_tray = tray.WorkerTray(
        on_open_dashboard=lambda: pytest.fail("must not open"),
        on_stop=lambda: pytest.fail("must not stop"),
        status_provider=lambda: "Worker stopped (fake)",
    )
    worker_tray.start()
    worker_tray.stop()  # also a safe no-op
    assert worker_tray._icon is None


def test_start_creates_an_icon_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """When supported, start() builds an icon with the live status line and runs it on a thread."""
    _enable_fake_tray(monkeypatch)
    worker_tray = tray.WorkerTray(
        on_open_dashboard=lambda: None,
        on_stop=lambda: None,
        status_provider=lambda: "Worker running (real)",
    )
    worker_tray.start()
    try:
        icon = worker_tray._icon
        assert isinstance(icon, _FakeIcon)
        # The first menu item is the dynamic, non-clickable status line.
        status_item = icon.menu.items[0]
        assert status_item.action is None
        assert status_item.text(status_item) == "Worker running (real)"
    finally:
        worker_tray.stop()
    assert worker_tray._icon is None
    assert icon.stopped is True


def test_menu_callbacks_dispatch_to_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Open/Stop menu actions invoke the provided callbacks; Stop also removes the icon."""
    _enable_fake_tray(monkeypatch)
    calls: list[str] = []
    worker_tray = tray.WorkerTray(
        on_open_dashboard=lambda: calls.append("open"),
        on_stop=lambda: calls.append("stop"),
        status_provider=lambda: "status",
    )
    worker_tray.start()

    worker_tray._handle_open()
    assert calls == ["open"]

    worker_tray._handle_stop()
    assert calls == ["open", "stop"]
    assert worker_tray._icon is None  # stop handler removed the icon


def test_open_handler_swallows_callback_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing Open callback is logged, not propagated, so the tray stays responsive."""
    _enable_fake_tray(monkeypatch)

    def _boom() -> None:
        raise RuntimeError("dashboard launch failed")

    worker_tray = tray.WorkerTray(on_open_dashboard=_boom, on_stop=lambda: None, status_provider=lambda: "s")
    worker_tray.start()
    try:
        worker_tray._handle_open()  # must not raise
    finally:
        worker_tray.stop()


def test_open_dashboard_reuses_a_running_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """open_dashboard points a browser at an existing web server rather than spawning a second one."""
    monkeypatch.setattr(tray, "_web_server_running", lambda port, host="127.0.0.1": True)
    opened: list[str] = []
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(tray.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn a launcher"))

    tray.open_dashboard(8000)
    assert opened == ["http://127.0.0.1:8000"]


def test_open_dashboard_spawns_launcher_when_none_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no web server up, open_dashboard spawns the web launcher (which attaches to the live host)."""
    monkeypatch.setattr(tray, "_web_server_running", lambda port, host="127.0.0.1": False)
    spawned: list[list[str]] = []
    monkeypatch.setattr(tray.subprocess, "Popen", lambda command, *a, **k: spawned.append(command))

    tray.open_dashboard(8000)
    assert spawned and spawned[0][1:] == ["-m", "horde_worker_regen.tui.web"]
