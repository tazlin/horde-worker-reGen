"""Tests for the web launcher's wiring: the served command must attach sessions to the worker host."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest
import textual_serve.server

from horde_worker_regen.tui import socket_protocol as sp
from horde_worker_regen.tui import web
from horde_worker_regen.tui.web import _rewrite_page_origin


@pytest.fixture(autouse=True)
def _isolated_dashboard_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each launcher test from a directory with no bridgeData.yaml.

    ``main`` resolves the bind address from the config file in the working directory, so a checkout whose
    own bridgeData.yaml binds the network would otherwise send these tests down the serve path (spawning
    a worker host) instead of the branch under test. Tests that want configured values pass an explicit
    path, which this does not affect.
    """
    monkeypatch.chdir(tmp_path)


def test_served_command_attaches_to_host() -> None:
    """The per-session dashboard command points at the host socket and carries the worker mode."""
    args = web._parse_args(["--process-mode", "fake"])
    command = web._build_served_command(args, 7717, remote_exposed=False)
    # Invoked via ``python -m`` so cmd.exe cannot shadow it with the repo's horde-worker.cmd launcher.
    assert command.startswith(f'"{sys.executable}" -m horde_worker_regen.tui.app ')
    assert "--attach 127.0.0.1:7717" in command
    assert "--process-mode fake" in command


def test_served_command_forwards_config() -> None:
    """A configured bridgeData path is forwarded (quoted) to the dashboard sessions."""
    args = web._parse_args(["--config", "my config.yaml"])
    command = web._build_served_command(args, 9000, remote_exposed=False)
    assert '--config "my config.yaml"' in command


def test_served_command_marks_sessions_exposed_only_when_bound_off_loopback() -> None:
    """The credential-withholding flag reaches the dashboard exactly when the bind is not loopback."""
    args = web._parse_args([])
    assert "--remote-exposed" not in web._build_served_command(args, 7717, remote_exposed=False)
    assert "--remote-exposed" in web._build_served_command(args, 7717, remote_exposed=True)


def test_host_port_resolution_prefers_flag_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host-port resolution is flag, then environment, then the protocol default."""
    monkeypatch.delenv("HORDE_WORKER_HOST_PORT", raising=False)
    assert web._resolve_host_port(None) == sp.DEFAULT_HOST_PORT
    monkeypatch.setenv("HORDE_WORKER_HOST_PORT", "9999")
    assert web._resolve_host_port(None) == 9999
    assert web._resolve_host_port(1234) == 1234


def test_web_host_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The web server binds loopback by default and honours an explicit override."""
    monkeypatch.delenv("HORDE_WORKER_WEB_HOST", raising=False)
    assert web._resolve_host(None, {}) == "127.0.0.1"
    assert web._resolve_host("0.0.0.0", {}) == "0.0.0.0"


def test_bind_resolution_prefers_flag_then_env_then_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host and port each fall back flag, environment, bridge data, default, in that order."""
    monkeypatch.delenv("HORDE_WORKER_WEB_HOST", raising=False)
    monkeypatch.delenv("HORDE_WORKER_WEB_PORT", raising=False)
    config = {web.CONFIG_HOST_KEY: "10.0.0.5", web.CONFIG_PORT_KEY: 9001}

    assert web._resolve_host(None, config) == "10.0.0.5"
    assert web._resolve_port(None, config) == 9001

    monkeypatch.setenv("HORDE_WORKER_WEB_HOST", "192.168.1.2")
    monkeypatch.setenv("HORDE_WORKER_WEB_PORT", "9500")
    assert web._resolve_host(None, config) == "192.168.1.2"
    assert web._resolve_port(None, config) == 9500

    assert web._resolve_host("0.0.0.0", config) == "0.0.0.0"
    assert web._resolve_port(8123, config) == 8123


def test_unusable_configured_port_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher reads the YAML without a schema, so a nonsense port is reported and ignored."""
    monkeypatch.delenv("HORDE_WORKER_WEB_PORT", raising=False)
    assert web._resolve_port(None, {web.CONFIG_PORT_KEY: "not a port"}) == web.DEFAULT_PORT


def test_dashboard_config_is_empty_when_the_file_is_absent(tmp_path: Path) -> None:
    """A missing bridgeData.yaml leaves every dashboard setting at its default rather than failing."""
    assert web._load_dashboard_config(str(tmp_path / "absent.yaml")) == {}


def test_dashboard_config_reads_the_bind_keys(tmp_path: Path) -> None:
    """The bind keys are read straight from the YAML, without going through reGenBridgeData."""
    config_path = tmp_path / "bridgeData.yaml"
    config_path.write_text(f"{web.CONFIG_HOST_KEY}: 0.0.0.0\n{web.CONFIG_PORT_KEY}: 8080\n", encoding="utf-8")

    loaded = web._load_dashboard_config(str(config_path))

    assert web._resolve_host(None, loaded) == "0.0.0.0"
    assert web._resolve_port(None, loaded) == 8080


def test_mobile_fit_injects_viewport_and_font_sizing_into_the_served_page() -> None:
    """The served page gains phone sizing, scrolling, and opt-in keyboard handling."""
    template = Path(textual_serve.server.__file__).parent / "templates" / "app_index.html"
    rewritten = web._inject_mobile_fit(template.read_text(encoding="utf-8"))

    assert 'name="viewport"' in rewritten
    assert rewritten.index('name="viewport"') < rewritten.index("</head>")
    # The script must land before the page's own window.onload handler builds the terminal from it.
    assert "dataset.fontSize" in rewritten
    assert rewritten.index("dataset.fontSize") < rewritten.index("</body>")
    assert "touchmove" in rewritten
    assert 'new WheelEvent("wheel"' in rewritten
    assert "touch-action: none" in rewritten
    assert "100dvh" in rewritten
    assert "window.visualViewport" in rewritten
    assert 'terminal.style.height = height + "px"' in rewritten
    assert 'keyboardButton.id = "mobile-keyboard-toggle"' in rewritten
    assert "viewportTop + viewportHeight - 88" in rewritten
    assert 'textarea.inputMode = keyboardEnabled ? "text" : "none"' in rewritten
    assert "textarea.readOnly = !keyboardEnabled" in rewritten
    assert 'touchAxis = totalX > totalY * 1.2 ? "horizontal" : "vertical"' in rewritten
    assert 'ctrlKey: touchAxis === "horizontal"' in rewritten
    assert 'shiftKey: touchAxis === "horizontal"' not in rewritten


def test_wildcard_bind_page_addresses_the_client_own_origin() -> None:
    """A page served from 0.0.0.0 must not tell the browser to open a websocket back to 0.0.0.0.

    That address is not routable from anywhere, so the session never opens and the browser sits on the
    splash screen. The rewrite points every URL at the address the request actually arrived on.
    """
    served = '<a href="http://0.0.0.0:8009/x">y</a> ws="ws://0.0.0.0:8009/ws"'
    rewritten = _rewrite_page_origin(
        served,
        served_origin="http://0.0.0.0:8009",
        client_origin="http://192.168.1.7:8009",
    )

    assert "0.0.0.0" not in rewritten
    assert 'ws="ws://192.168.1.7:8009/ws"' in rewritten
    assert 'href="http://192.168.1.7:8009/x"' in rewritten


def test_page_origin_is_left_alone_when_it_already_matches_the_client() -> None:
    """A loopback or explicitly-addressed bind is already routable, so nothing is rewritten."""
    served = 'ws="ws://127.0.0.1:8000/ws"'
    assert (
        _rewrite_page_origin(served, served_origin="http://127.0.0.1:8000", client_origin="http://127.0.0.1:8000")
        == served
    )


def test_mobile_fit_leaves_unrecognised_markup_alone() -> None:
    """A template without the insertion points is passed through, so upstream drift degrades quietly."""
    assert web._inject_mobile_fit("<p>not the page</p>") == "<p>not the page</p>"


def test_app_window_is_the_default_browser_is_opt_in() -> None:
    """The borderless app window is the default; --browser opts back into a normal tab."""
    assert web._parse_args([]).browser is False
    assert web._parse_args(["--browser"]).browser is True


def test_chromium_app_command_none_when_no_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no Chromium browser anywhere, app-mode is unavailable (caller falls back to a tab)."""
    monkeypatch.setattr(web.shutil, "which", lambda name: None)
    monkeypatch.setattr(web.os.path, "isfile", lambda path: False)
    assert web._chromium_app_command("http://127.0.0.1:8000") is None


def test_chromium_app_command_builds_app_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """A discovered browser is launched with --app=URL for a borderless window."""
    sentinel = "/opt/chromium/chromium"
    monkeypatch.setattr(web.shutil, "which", lambda name: sentinel if name == "chromium" else None)
    monkeypatch.setattr(web.os.path, "isfile", lambda path: path == sentinel)
    command = web._chromium_app_command("http://127.0.0.1:8000")
    assert command == [sentinel, "--app=http://127.0.0.1:8000"]


def test_open_app_window_false_without_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """_open_app_window reports failure (no launch) when no browser command is found."""
    monkeypatch.setattr(web, "_chromium_app_command", lambda url: None)
    assert web._open_app_window("http://127.0.0.1:8000") is False


def test_open_app_window_launches_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """A found browser command is launched and reported as success."""
    launched: list[list[str]] = []
    monkeypatch.setattr(web, "_chromium_app_command", lambda url: ["browser", f"--app={url}"])
    monkeypatch.setattr(web.subprocess, "Popen", lambda command: launched.append(command))
    assert web._open_app_window("http://127.0.0.1:8000") is True
    assert launched == [["browser", "--app=http://127.0.0.1:8000"]]


def test_open_dashboard_prefers_app_window_then_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dashboard opens as an app window when possible, else in the default browser."""
    opened: list[str] = []
    monkeypatch.setattr(web.webbrowser, "open", lambda url: opened.append(url))

    monkeypatch.setattr(web, "_open_app_window", lambda url: True)
    web._open_dashboard("http://x", app_window=True)
    assert opened == []  # app window handled it; no browser tab

    monkeypatch.setattr(web, "_open_app_window", lambda url: False)
    web._open_dashboard("http://x", app_window=True)
    assert opened == ["http://x"]  # fell back to a tab

    web._open_dashboard("http://y", app_window=False)
    assert opened == ["http://x", "http://y"]  # --browser always uses a tab


def test_graphical_environment_true_on_windows_and_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows and macOS always have a window server, so a browser can be opened."""
    monkeypatch.setattr(web.sys, "platform", "win32")
    assert web._is_graphical_environment() is True
    monkeypatch.setattr(web.sys, "platform", "darwin")
    assert web._is_graphical_environment() is True


def test_graphical_environment_linux_needs_a_display(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Linux a browser needs an X11 or Wayland display; neither set is headless."""
    monkeypatch.setattr(web.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert web._is_graphical_environment() is False
    monkeypatch.setenv("DISPLAY", ":0")
    assert web._is_graphical_environment() is True


def test_main_falls_back_to_terminal_when_headless_with_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A no-display box with a real terminal runs the in-terminal TUI instead of serving a browser."""
    monkeypatch.setattr(web, "_is_graphical_environment", lambda: False)
    monkeypatch.setattr(web.sys.stdout, "isatty", lambda: True)
    captured: list[list[str]] = []
    monkeypatch.setattr(web, "_run_terminal_fallback", lambda args: captured.append([args.process_mode]))

    web.main(["--process-mode", "fake"])

    assert captured == [["fake"]]


def test_main_refuses_when_headless_with_no_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """No display and no terminal means the dashboard cannot be shown, so exit with guidance."""
    monkeypatch.setattr(web, "_is_graphical_environment", lambda: False)
    monkeypatch.setattr(web.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(web, "_run_terminal_fallback", lambda args: pytest.fail("must not run the terminal TUI"))

    with pytest.raises(SystemExit) as exc_info:
        web.main([])

    assert exc_info.value.code == 1


def test_main_serves_anyway_when_lan_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binding the LAN (--host) is explicit serve intent, so the headless fallback is skipped."""
    monkeypatch.setattr(web, "_is_graphical_environment", lambda: False)
    monkeypatch.setattr(web.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(web, "_run_terminal_fallback", lambda args: pytest.fail("must not fall back when LAN-bound"))
    monkeypatch.setattr(web, "_host_running", lambda address: True)
    monkeypatch.setattr(web, "_announce_dashboard_port", lambda address, port: None)
    monkeypatch.setattr(web, "_schedule_dashboard_open", lambda *a, **k: None)

    served: list[str] = []

    class _FakeServer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def serve(self) -> None:
            served.append("served")

    import textual_serve.server

    monkeypatch.setattr(textual_serve.server, "Server", _FakeServer)

    web.main(["--host", "0.0.0.0", "--no-browser"])

    assert served == ["served"]


def _host_watcher_alive() -> bool:
    """Whether a host-liveness watcher thread is currently running."""
    return any(thread.name == "host-liveness-watch" and thread.is_alive() for thread in threading.enumerate())


def test_main_stops_host_watcher_so_its_leash_cannot_kill_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``serve()`` returns, ``main`` must stop the liveness watcher so its hard-exit leash never fires late.

    The watcher is a daemon whose on-host-gone callback hard-exits the whole process. Left running after the
    launcher unwinds (a non-blocking ``serve()``, as here and under any test exercising ``main``), it would
    later conclude the absent host is gone and ``os._exit`` an unrelated, still-running process. Binding it to
    the launcher's lifetime must both stop the thread and suppress the leash on a deliberate unwind.
    """
    monkeypatch.setattr(web, "_is_graphical_environment", lambda: True)
    # No host is spawned and nothing listens on the port, so without the lifetime binding the watcher would
    # exhaust its grace and fire the leash after main() has already returned.
    monkeypatch.setattr(web, "_host_running", lambda address: True)
    monkeypatch.setattr(web, "_announce_dashboard_port", lambda address, port: None)
    monkeypatch.setattr(web, "_schedule_dashboard_open", lambda *a, **k: None)

    wound_down = threading.Event()
    monkeypatch.setattr(web, "_wind_down_launcher", wound_down.set)

    class _FakeServer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def serve(self) -> None:
            pass

    import textual_serve.server

    monkeypatch.setattr(textual_serve.server, "Server", _FakeServer)

    web.main(["--no-browser"])

    # main() joins the watcher on unwind, so it should already be gone; allow a brief margin regardless.
    deadline = time.time() + 5.0
    while time.time() < deadline and _host_watcher_alive():
        time.sleep(0.05)

    assert not _host_watcher_alive(), (
        "the host-liveness watcher outlived the launcher; it could later kill the process"
    )
    assert not wound_down.is_set(), "the launcher's hard-exit leash fired during a deliberate unwind"
