"""Serve the worker dashboard in a web browser, backed by a persistent worker host.

This is the default launch path for non-technical users. It does two things:

1. Ensures a [`WorkerHost`][horde_worker_regen.tui.worker_host.WorkerHost] is running (spawning one if the
   host port is free), so a single worker is owned independently of any browser session.
2. Serves the TUI with ``textual-serve``, instructing each per-session TUI subprocess to *attach* to that
   host rather than own a worker. Closing a browser tab therefore leaves the worker running; closing this
   launcher stops the worker cleanly.

Network exposure is conservative: the web server binds ``127.0.0.1`` by default. Binding the network is
a deliberate power-user action, taken via ``--host``, ``HORDE_WORKER_WEB_HOST``, or ``dashboard_web_host``
in the bridge data (in that order of precedence), and it exposes an unauthenticated dashboard. Doing so
tells each dashboard session to withhold the credential fields from its config editor and prints what
that does not cover. (The worker host always binds loopback.)

The served page is also fitted for a phone browser: see :func:`_inject_mobile_fit`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Protocol

import psutil
from ruamel.yaml.error import YAMLError

from horde_worker_regen.process_management.lifecycle.owned_process_registry import kill_process_tree
from horde_worker_regen.tui import config_form
from horde_worker_regen.tui import socket_protocol as sp
from horde_worker_regen.tui.job_object import WorkerJobObject

if TYPE_CHECKING:
    from aiohttp.web import Application


class _DashboardServer(Protocol):
    """The shutdown surface added to the textual-serve server used by this launcher."""

    def serve(self, debug: bool = False) -> None: ...

    def request_threadsafe_exit(self) -> None: ...

    def wait_stopped(self, timeout: float) -> bool: ...

    async def _make_app(self) -> Application: ...


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
HOST_ENV_VAR = "HORDE_WORKER_WEB_HOST"
PORT_ENV_VAR = "HORDE_WORKER_WEB_PORT"
HOST_PORT_ENV_VAR = "HORDE_WORKER_HOST_PORT"

CONFIG_HOST_KEY = "dashboard_web_host"
CONFIG_PORT_KEY = "dashboard_web_port"
"""bridgeData.yaml keys holding the bind address, the lowest-priority source after the flag and env var."""

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_HOST_SHUTDOWN_TIMEOUT_SECONDS = 120.0

_HOST_WATCH_CONNECT_TIMEOUT_SECONDS = 1.0
"""Per-attempt connect timeout while the watcher waits for the host's socket to come up."""

_HOST_WATCH_STARTUP_GRACE_SECONDS = 60.0
"""How long the watcher keeps trying to reach the host before treating it as failed to start.

The host binds its socket early in startup (before any worker/torch work), so a couple of seconds is the
normal case; this is generous headroom for a loaded box. Exhausting it means the host never came up, which
leaves nothing to serve, so the launcher winds down just as it would for a host that came up and then died.
"""

_HOST_WATCH_STOP_JOIN_SECONDS = 5.0
"""How long :func:`main` waits for the liveness watcher to unwind after signalling it to stop.

Generous over the watcher's own connect timeout and stop-poll interval, so a deliberate launcher unwind
joins the watcher rather than leaving it as a daemon that could later fire its process-killing leash.
"""

_DASHBOARD_SHUTDOWN_GRACE_SECONDS = 10.0
"""How long host-driven shutdown gives textual-serve to close its sessions before the precise reap fallback."""

_DASHBOARD_SESSION_KILL_GRACE_SECONDS = 2.0
"""Per-session terminate grace before the precise reap fallback escalates to a kill."""

_PROCESS_CREATE_TIME_TOLERANCE_SECONDS = 1.0
"""PID-reuse guard tolerance for dashboard session process identities."""


class _DashboardSessionProcesses:
    """Track only the Textual session shells this launcher owns.

    Browser processes are deliberately never registered. On Windows each session shell is also assigned to
    a kill-on-close Job Object, while the identity-checked in-memory set provides the bounded normal-shutdown
    fallback on every platform.
    """

    def __init__(self) -> None:
        self._identities: dict[int, float] = {}
        self._lock = threading.Lock()
        self._job = WorkerJobObject()

    def register(self, pid: int | None) -> None:
        """Record a freshly spawned session shell and bind it to the Windows launcher-lifetime job."""
        if pid is None:
            return
        try:
            create_time = psutil.Process(pid).create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        with self._lock:
            self._identities[pid] = create_time
        self._job.assign(pid)

    def forget(self, pid: int | None) -> None:
        """Forget a session shell after textual-serve has observed its clean exit."""
        if pid is None:
            return
        with self._lock:
            self._identities.pop(pid, None)

    def terminate_all(self) -> list[int]:
        """Terminate every still-matching session tree, never any other launcher descendant."""
        with self._lock:
            identities = dict(self._identities)
        terminated: list[int] = []
        for pid, create_time in identities.items():
            try:
                process = psutil.Process(pid)
                if abs(process.create_time() - create_time) > _PROCESS_CREATE_TIME_TOLERANCE_SECONDS:
                    continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            kill_process_tree(pid, grace_seconds=_DASHBOARD_SESSION_KILL_GRACE_SECONDS)
            terminated.append(pid)
        return terminated


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the web-server command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="horde-worker-web",
        description="Serve the AI Horde worker dashboard in a web browser.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Address to bind the web server (default {DEFAULT_HOST}; ${HOST_ENV_VAR} then "
        f"{CONFIG_HOST_KEY} in bridgeData.yaml are the fallbacks). "
        "Use 0.0.0.0 to expose on the LAN (unauthenticated; opt-in).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Web server port (default {DEFAULT_PORT}; ${PORT_ENV_VAR} then {CONFIG_PORT_KEY} in "
        "bridgeData.yaml are the fallbacks).",
    )
    parser.add_argument(
        "--host-port",
        type=int,
        default=None,
        help=f"Worker-host socket port (default {sp.DEFAULT_HOST_PORT}; ${HOST_PORT_ENV_VAR} overrides).",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open any window automatically.")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report whether a worker host is already running (and its status), then exit.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Ask a running worker host to stop the worker and exit cleanly, then exit.",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open the dashboard as a normal browser tab instead of a borderless app window.",
    )
    parser.add_argument(
        "--process-mode",
        choices=("real", "fake"),
        default="real",
        help="Worker mode: 'fake' serves a synthetic worker for demos.",
    )
    parser.add_argument("--config", type=str, default=None, help="Forwarded to the dashboard: bridgeData.yaml path.")
    parser.add_argument("-e", "--load-config-from-env-vars", action="store_true", help="Worker reads AIWORKER_* env.")
    parser.add_argument("--amd", "--amd-gpu", action="store_true", help="Enable AMD GPU optimisations on the worker.")
    parser.add_argument("-n", "--worker-name", type=str, default=None, help="Override the worker name.")
    parser.add_argument("--directml", type=int, default=None, help="Enable directml on the given device index.")
    return parser.parse_args(argv)


def _resolve_host(arg_host: str | None, config: Mapping[str, object]) -> str:
    """Resolve the web bind host from the flag, then the environment, then the config, then the default."""
    configured_host = config.get(CONFIG_HOST_KEY)
    return arg_host or os.getenv(HOST_ENV_VAR) or (str(configured_host) if configured_host else DEFAULT_HOST)


def _resolve_port(arg_port: int | None, config: Mapping[str, object]) -> int:
    """Resolve the web port from the flag, then the environment, then the config, then the default."""
    if arg_port is not None:
        return arg_port
    env_port = os.getenv(PORT_ENV_VAR)
    if env_port:
        return int(env_port)
    configured_port = config.get(CONFIG_PORT_KEY)
    if configured_port is None:
        return DEFAULT_PORT
    try:
        return int(str(configured_port))
    except ValueError:
        # The launcher reads the YAML directly rather than through reGenBridgeData, so a bad value here
        # has had no schema to bounce off. Saying so and carrying on beats refusing to open a dashboard.
        print(
            f"Ignoring {CONFIG_PORT_KEY}={configured_port!r} in the config: not a port number. Using {DEFAULT_PORT}.",
            file=sys.stderr,
        )
        return DEFAULT_PORT


def _load_dashboard_config(config_path: str | None) -> Mapping[str, object]:
    """Read the bridge data YAML for the dashboard's own settings, returning empty on any read failure.

    Uses the config editor's light ruamel loader rather than ``reGenBridgeData`` so the launcher stays
    free of the SDK import chain. Only the dashboard keys are consulted; the worker validates the rest.
    """
    path = Path(config_path) if config_path else config_form.DEFAULT_CONFIG_PATH
    try:
        loaded = config_form.load_config(path)
    except (OSError, YAMLError) as read_error:
        print(f"Could not read {path} for dashboard settings ({read_error}); using defaults.", file=sys.stderr)
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def _resolve_host_port(arg_host_port: int | None) -> int:
    """Resolve the worker-host socket port from the flag, then the environment, then the default."""
    if arg_host_port is not None:
        return arg_host_port
    env_port = os.getenv(HOST_PORT_ENV_VAR)
    return int(env_port) if env_port else sp.DEFAULT_HOST_PORT


def _is_loopback(host: str) -> bool:
    """Whether binding *host* keeps the dashboard reachable only from this machine."""
    return host in _LOOPBACK_HOSTS


def _build_served_command(args: argparse.Namespace, host_port: int, *, remote_exposed: bool) -> str:
    """Compose the per-session dashboard command, which attaches to the worker host.

    The TUI is invoked via ``python -m`` rather than the ``horde-worker`` console script because
    textual-serve launches this through the shell (``cmd.exe /c`` on Windows), and cmd.exe resolves
    bare names against the current directory before PATH. The web server runs from the repo root, whose
    ``horde-worker.cmd`` launcher would otherwise shadow the console script and re-invoke the *web*
    server. Module invocation cannot be shadowed and mirrors how the worker host is spawned.

    Args:
        args: The parsed launcher arguments, for the worker options the dashboard needs.
        host_port: Port of the worker host this dashboard session attaches to.
        remote_exposed: Whether the web server binds an address other than loopback, which the
            dashboard needs to know so it can withhold the credential fields from the config editor.

    Returns:
        The shell command textual-serve runs for each browser session.
    """
    parts = [
        f'"{sys.executable}"',
        "-m",
        "horde_worker_regen.tui.app",
        f"--attach 127.0.0.1:{host_port}",
        f"--process-mode {args.process_mode}",
    ]
    if args.config:
        parts.append(f'--config "{args.config}"')
    if remote_exposed:
        parts.append("--remote-exposed")
    return " ".join(parts)


MOBILE_TARGET_COLUMNS = 52
"""Columns the served terminal aims for on a narrow viewport.

This is a readability-first compromise: compact tab labels may leave a small overflow that teaches the
sideways gesture, while tables shed to their identity columns rather than shrinking the text further. The
minimum font-size floor takes precedence on very narrow phones.
"""

_MONOSPACE_CELL_WIDTH_RATIO = 0.6
"""Cell width as a fraction of font size for the served page's monospace font, used to size the font."""

_MOBILE_MIN_FONT_SIZE_PX = 12
_MOBILE_MAX_FONT_SIZE_PX = 16
"""Bounds on the fitted font size. The maximum is textual-serve's own default, which makes the fitting
script a no-op on a desktop viewport: only a viewport too narrow to reach the target column count at
that size gets anything smaller. Twelve pixels is the legibility floor; very narrow phones get
fewer columns and rely on the phone layout's shedding rather than shrinking the controls to 8px."""

_NARROW_VIEWPORT_PX = 700
"""Viewport width below which the served page's fixed-width session dialog is allowed to reflow."""

_HEAD_CLOSE_TAG = "</head>"
_BODY_CLOSE_TAG = "</body>"

_MOBILE_HEAD_TEMPLATE = Template(
    """<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  @media (max-width: ${narrow_viewport_px}px) {
    html, body { width: 100%; height: 100%; overflow: hidden; }
    .intro { width: 100%; height: auto; padding: 1rem; box-sizing: border-box; }
    .textual-terminal {
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100dvh;
      box-sizing: border-box;
      touch-action: pinch-zoom;
      overscroll-behavior: none;
    }
    .xterm .xterm-viewport { scrollbar-width: none; }
    .xterm .xterm-viewport::-webkit-scrollbar { width: 0; height: 0; }
    #mobile-controls-dock {
      position: absolute;
      z-index: 10000;
      display: none;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      height: 52px;
      padding: 2px 8px;
      box-sizing: border-box;
      background: #0c181f;
    }
    #mobile-controls-dock button {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 48px;
      height: 48px;
      margin: 0;
      padding: 0;
      border: 2px solid #c792ea;
      border-radius: 12px;
      background: #1e1e2e;
      color: white;
      font: 24px sans-serif;
      opacity: 0.9;
    }
    #mobile-tabs-toggle,
    #mobile-palette-button {
      width: 48px;
      padding: 0;
      font-size: 24px;
    }
    #mobile-tabs-toggle.-enabled { background: #6c3baa; }
    #mobile-keyboard-toggle.-enabled { background: #6c3baa; }
    @media (pointer: coarse) {
      #mobile-controls-dock { display: flex; }
    }
  }
</style>
""",
)
"""Injected into the served page's head.

Without the viewport tag a phone lays the page out at a notional desktop width and scales the result
down, which renders the terminal too small to read. The dialog rule follows from the tag: the page's
session dialog is a fixed 640px, which overflows a phone once the layout viewport is the real one.
"""

_MOBILE_FIT_SCRIPT_TEMPLATE = Template(
    """<script>
  (function () {
    var terminal = document.querySelector(".textual-terminal");
    if (!terminal) {
      return;
    }
    if (!new URLSearchParams(window.location.search).has("fontsize")) {
      var fitted = Math.round(window.innerWidth / (${target_columns} * ${cell_width_ratio}));
      var fontSize = Math.max(${min_font_size}, Math.min(${max_font_size}, fitted));
      terminal.dataset.fontSize = String(fontSize);
    }

    // 100vh includes browser chrome on several mobile browsers. Size the terminal to the visual
    // viewport instead, refitting xterm whenever the address bar or software keyboard changes it.
    var visualViewport = window.visualViewport;
    var coarsePointer = window.matchMedia("(pointer: coarse)").matches || navigator.maxTouchPoints > 0;
    var mobileControlsEnabled = coarsePointer && window.innerWidth <= ${narrow_viewport_px};
    var keyboardRailHeight = mobileControlsEnabled ? 52 : 0;
    var mobileDock = null;
    var viewportFrame = null;
    function fitVisibleViewport() {
      if (viewportFrame !== null) {
        cancelAnimationFrame(viewportFrame);
      }
      viewportFrame = requestAnimationFrame(function () {
        viewportFrame = null;
        var width = Math.floor(visualViewport ? visualViewport.width : window.innerWidth);
        var height = Math.floor(visualViewport ? visualViewport.height : window.innerHeight);
        terminal.style.left = Math.floor(visualViewport ? visualViewport.offsetLeft : 0) + "px";
        terminal.style.top = Math.floor(visualViewport ? visualViewport.offsetTop : 0) + "px";
        terminal.style.width = width + "px";
        terminal.style.height = Math.max(1, height - keyboardRailHeight) + "px";
        positionMobileDock();
        if (typeof window.onresize === "function") {
          window.onresize();
        }
      });
    }
    fitVisibleViewport();
    window.addEventListener("load", fitVisibleViewport);
    window.addEventListener("orientationchange", fitVisibleViewport);
    if (visualViewport) {
      visualViewport.addEventListener("resize", fitVisibleViewport);
      visualViewport.addEventListener("scroll", fitVisibleViewport);
    }

    // Every Textual widget is painted into xterm's canvas. xterm focuses one hidden textarea for all
    // terminal keyboard input, so a phone cannot distinguish a painted tab from a painted form field
    // and normally opens its keyboard for both. Keep that transport textarea non-editable until the
    // operator explicitly asks for typing with this real, browser-level button.
    var keyboardEnabled = false;
    var keyboardButton = null;
    var paletteButton = null;
    var tabsButton = null;
    var mainTabsVisible = true;
    function positionMobileDock() {
      if (!mobileDock) {
        return;
      }
      var viewportWidth = visualViewport ? visualViewport.width : window.innerWidth;
      var viewportHeight = visualViewport ? visualViewport.height : window.innerHeight;
      var viewportLeft = visualViewport ? visualViewport.offsetLeft : 0;
      var viewportTop = visualViewport ? visualViewport.offsetTop : 0;
      mobileDock.style.left = Math.floor(viewportLeft) + "px";
      mobileDock.style.top = Math.max(0, Math.floor(viewportTop + viewportHeight - keyboardRailHeight)) + "px";
      mobileDock.style.width = Math.floor(viewportWidth) + "px";
    }
    function dispatchTerminalKey(key, code, keyCode, modifiers) {
      var textarea = terminal.querySelector(".xterm-helper-textarea");
      if (!textarea) {
        return;
      }
      ["keydown", "keyup"].forEach(function (type) {
        textarea.dispatchEvent(new KeyboardEvent(type, {
          bubbles: true,
          cancelable: true,
          key: key,
          code: code,
          keyCode: keyCode,
          which: keyCode,
          ctrlKey: Boolean(modifiers.ctrl),
          altKey: Boolean(modifiers.alt),
          shiftKey: Boolean(modifiers.shift)
        }));
      });
    }
    function configureKeyboardTransport() {
      if (!mobileControlsEnabled) {
        return;
      }
      var textarea = terminal.querySelector(".xterm-helper-textarea");
      if (!textarea) {
        return;
      }
      textarea.readOnly = !keyboardEnabled;
      textarea.inputMode = keyboardEnabled ? "text" : "none";
      textarea.setAttribute("aria-hidden", keyboardEnabled ? "false" : "true");
      if (keyboardEnabled) {
        textarea.focus({ preventScroll: true });
      } else {
        textarea.blur();
      }
    }
    if (mobileControlsEnabled) {
      mobileDock = document.createElement("div");
      mobileDock.id = "mobile-controls-dock";
      paletteButton = document.createElement("button");
      paletteButton.id = "mobile-palette-button";
      paletteButton.type = "button";
      paletteButton.textContent = "\u2630";
      paletteButton.title = "Open dashboard command palette";
      paletteButton.setAttribute("aria-label", "Open dashboard command palette");
      paletteButton.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        dispatchTerminalKey("ArrowDown", "ArrowDown", 40, { ctrl: true, alt: true, shift: true });
      });
      tabsButton = document.createElement("button");
      tabsButton.id = "mobile-tabs-toggle";
      tabsButton.type = "button";
      tabsButton.textContent = "\u25b4";
      tabsButton.classList.add("-enabled");
      tabsButton.title = "Hide main dashboard tabs";
      tabsButton.setAttribute("aria-label", "Hide main dashboard tabs");
      tabsButton.setAttribute("aria-pressed", "true");
      tabsButton.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        mainTabsVisible = !mainTabsVisible;
        tabsButton.textContent = mainTabsVisible ? "\u25b4" : "\u25be";
        tabsButton.classList.toggle("-enabled", mainTabsVisible);
        tabsButton.setAttribute("aria-pressed", String(mainTabsVisible));
        tabsButton.setAttribute(
          "aria-label",
          mainTabsVisible ? "Hide main dashboard tabs" : "Show main dashboard tabs"
        );
        tabsButton.title = mainTabsVisible ? "Hide main dashboard tabs" : "Show main dashboard tabs";
        dispatchTerminalKey("ArrowUp", "ArrowUp", 38, { ctrl: true, alt: true, shift: true });
      });
      keyboardButton = document.createElement("button");
      keyboardButton.id = "mobile-keyboard-toggle";
      keyboardButton.type = "button";
      keyboardButton.textContent = "\u2328";
      keyboardButton.setAttribute("aria-label", "Enable dashboard keyboard");
      keyboardButton.setAttribute("aria-pressed", "false");
      keyboardButton.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        keyboardEnabled = !keyboardEnabled;
        keyboardButton.classList.toggle("-enabled", keyboardEnabled);
        keyboardButton.setAttribute("aria-pressed", String(keyboardEnabled));
        keyboardButton.setAttribute(
          "aria-label",
          keyboardEnabled ? "Disable dashboard keyboard" : "Enable dashboard keyboard"
        );
        configureKeyboardTransport();
      });
      mobileDock.appendChild(paletteButton);
      mobileDock.appendChild(tabsButton);
      mobileDock.appendChild(keyboardButton);
      document.body.appendChild(mobileDock);
      positionMobileDock();
      window.addEventListener("load", configureKeyboardTransport);
      var terminalObserver = new MutationObserver(configureKeyboardTransport);
      terminalObserver.observe(terminal, { childList: true, subtree: true });
      if (visualViewport) {
        visualViewport.addEventListener("resize", positionMobileDock);
        visualViewport.addEventListener("scroll", positionMobileDock);
      }
      window.addEventListener("orientationchange", positionMobileDock);
    }

    // xterm.js disables its own touch scrolling while an application (Textual) has mouse reporting
    // enabled. Convert a one-finger drag anywhere on the terminal into the same wheel events Textual
    // already understands, so a phone user never has to catch the one-cell in-app scrollbar.
    var lastTouchX = null;
    var lastTouchY = null;
    var touchStartX = null;
    var touchStartY = null;
    var touchAxis = null;
    var touchWheelRemainder = 0;
    var horizontalSwipeSent = false;
    function moveTouchedTab(direction) {
      // Use an application-only modified-arrow binding, carried by xterm's normal input route. This does
      // not synthesize a click, depend on current focus, or activate the tab under the finger before moving.
      // The main strip always lives in the first six terminal rows; a swipe lower down is the Config strip.
      var screen = terminal.querySelector(".xterm-screen");
      var measure = terminal.querySelector(".xterm-char-measure-element");
      var screenTop = screen ? screen.getBoundingClientRect().top : terminal.getBoundingClientRect().top;
      var cellHeight = measure ? measure.getBoundingClientRect().height : 0;
      if (cellHeight <= 0) {
        cellHeight = parseFloat(terminal.dataset.fontSize || "16");
      }
      var terminalRow = Math.floor((touchStartY - screenTop) / cellHeight);
      var configStrip = terminalRow >= 6;
      var key = direction > 0 ? "ArrowRight" : "ArrowLeft";
      var keyCode = direction > 0 ? 39 : 37;
      dispatchTerminalKey(key, key, keyCode, { ctrl: true, alt: true, shift: configStrip });
    }
    terminal.addEventListener("touchstart", function (event) {
      if (event.touches.length !== 1) {
        lastTouchX = lastTouchY = touchStartX = touchStartY = touchAxis = null;
        touchWheelRemainder = 0;
        horizontalSwipeSent = false;
        return;
      }
      var touch = event.touches[0];
      lastTouchX = touchStartX = touch.clientX;
      lastTouchY = touchStartY = touch.clientY;
      touchAxis = null;
    }, { passive: true });
    terminal.addEventListener("touchmove", function (event) {
      if (lastTouchX === null || lastTouchY === null || event.touches.length !== 1) {
        lastTouchX = lastTouchY = touchStartX = touchStartY = touchAxis = null;
        return;
      }
      var touch = event.touches[0];
      var deltaX = lastTouchX - touch.clientX;
      var deltaY = lastTouchY - touch.clientY;
      if (touchAxis === null && touchStartX !== null && touchStartY !== null) {
        var totalX = Math.abs(touch.clientX - touchStartX);
        var totalY = Math.abs(touch.clientY - touchStartY);
        if (Math.max(totalX, totalY) < 8) {
          return;
        }
        touchAxis = totalX > totalY * 1.2 ? "horizontal" : "vertical";
      }
      lastTouchX = touch.clientX;
      lastTouchY = touch.clientY;
      var wheelDelta = touchAxis === "horizontal" ? deltaX : deltaY;
      if (touchAxis === "horizontal" && horizontalSwipeSent) {
        event.preventDefault();
        return;
      }
      touchWheelRemainder += wheelDelta;
      // xterm consumes wheel input in terminal-row units. Sending every 1-2 pixel touch movement made
      // a drag look active to the browser while often producing no mouse packet for Textual at all.
      var wheelSteps = Math.trunc(touchWheelRemainder / 12);
      if (wheelSteps === 0) {
        return;
      }
      touchWheelRemainder -= wheelSteps * 12;
      horizontalSwipeSent = touchAxis === "horizontal";
      if (horizontalSwipeSent) {
        moveTouchedTab(wheelSteps);
        event.preventDefault();
        return;
      }
      var target = document.elementFromPoint(touch.clientX, touch.clientY) || terminal;
      target.dispatchEvent(new WheelEvent("wheel", {
        bubbles: true,
        cancelable: true,
        clientX: touch.clientX,
        clientY: touch.clientY,
        deltaMode: WheelEvent.DOM_DELTA_LINE,
        deltaY: wheelSteps,
        ctrlKey: false
      }));
      event.preventDefault();
    }, { passive: false });
    function clearTouchGesture() {
      lastTouchX = lastTouchY = touchStartX = touchStartY = touchAxis = null;
      touchWheelRemainder = 0;
      horizontalSwipeSent = false;
    }
    terminal.addEventListener("touchend", clearTouchGesture, { passive: true });
    terminal.addEventListener("touchcancel", clearTouchGesture, { passive: true });
  })();
</script>
""",
)
"""Injected at the end of the served page's body.

The viewport tag alone would leave a phone with roughly 40 columns at the default font size, so the two
injections only work as a pair. This sizes the font to reach ``MOBILE_TARGET_COLUMNS`` instead, writing
the attribute the page's own script reads when it constructs the terminal under ``window.onload``. An
explicit ``?fontsize=`` in the URL always wins.
"""


def _inject_mobile_fit(html: str) -> str:
    """Return *html* with the mobile viewport handling added, or unchanged if its markers are absent.

    Args:
        html: The served page's markup.

    Returns:
        The markup with the head and body injections applied. Markup missing either closing tag is
        returned as-is, so a change to textual-serve's template degrades to the desktop behaviour
        rather than breaking the dashboard.
    """
    if _HEAD_CLOSE_TAG not in html or _BODY_CLOSE_TAG not in html:
        return html
    head_addition = _MOBILE_HEAD_TEMPLATE.substitute(narrow_viewport_px=_NARROW_VIEWPORT_PX)
    body_addition = _MOBILE_FIT_SCRIPT_TEMPLATE.substitute(
        target_columns=MOBILE_TARGET_COLUMNS,
        cell_width_ratio=_MONOSPACE_CELL_WIDTH_RATIO,
        min_font_size=_MOBILE_MIN_FONT_SIZE_PX,
        max_font_size=_MOBILE_MAX_FONT_SIZE_PX,
        narrow_viewport_px=_NARROW_VIEWPORT_PX,
    )
    html = html.replace(_HEAD_CLOSE_TAG, f"{head_addition}{_HEAD_CLOSE_TAG}", 1)
    return html.replace(_BODY_CLOSE_TAG, f"{body_addition}{_BODY_CLOSE_TAG}", 1)


WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "[::]"})  # noqa: S104 - matched, not bound
"""Bind addresses that mean "every interface" and so are not an address any client can connect back to."""


def _rewrite_page_origin(html: str, *, served_origin: str, client_origin: str) -> str:
    """Return *html* with textual-serve's own origin swapped for the one the browser connected to.

    Every URL on the served page (the session websocket above all) is built from the server's
    ``public_url``, which defaults to the bind address. Bind every interface and that becomes
    ``http://0.0.0.0:<port>``, which no client can route to: the page loads, its websocket never opens,
    and the browser sits on the splash screen forever. The browser already knows the address that
    reached this server, so the page is rewritten per request to use it.

    Args:
        html: The served page's markup.
        served_origin: The server's own ``public_url``, e.g. ``http://0.0.0.0:8000``.
        client_origin: Scheme and host the request arrived on, e.g. ``http://192.168.1.7:8000``.

    Returns:
        The markup with both the HTTP and the websocket form of ``served_origin`` replaced.
    """
    if served_origin == client_origin:
        return html
    websocket_scheme = "wss" if served_origin.startswith("https") else "ws"
    client_websocket_scheme = "wss" if client_origin.startswith("https") else "ws"
    served_websocket_origin = f"{websocket_scheme}:{served_origin.split(':', 1)[1]}"
    client_websocket_origin = f"{client_websocket_scheme}:{client_origin.split(':', 1)[1]}"
    return html.replace(served_websocket_origin, client_websocket_origin).replace(served_origin, client_origin)


def _build_server(
    command: str,
    *,
    host: str,
    port: int,
    title: str,
    host_address: tuple[str, int],
    session_processes: _DashboardSessionProcesses,
) -> _DashboardServer:
    """Create the textual-serve server, extended so the served page reaches back and fits a phone.

    ``textual_serve`` pulls in aiohttp and jinja2, which the headless terminal fallback never needs, so
    the import stays deferred to here. Subclassing is how textual-serve expects a server to be extended:
    ``_make_app`` is its application factory.

    Args:
        command: The per-session dashboard command textual-serve runs.
        host: Address to bind.
        port: Port to bind.
        title: Name shown for the served application.
        host_address: Loopback address of the persistent worker host used by the native companion page.
        session_processes: Exact process ownership tracker for per-browser Textual sessions.

    Returns:
        An unstarted server; call ``serve()`` on it.
    """
    from aiohttp import web as aiohttp_web
    from aiohttp.typedefs import Handler
    from textual_serve.app_service import AppService
    from textual_serve.server import Server as TextualServeServer
    from textual_serve.server import to_int

    from horde_worker_regen.tui.native_dashboard import NativeDashboardWeb

    native_dashboard = NativeDashboardWeb(host_address)

    class TrackedAppService(AppService):
        """A Textual session whose shell is recorded for precise bounded teardown."""

        async def _open_app_process(self, width: int = 80, height: int = 24) -> asyncio.subprocess.Process:
            """Start the session shell and register only that owned process, never the browser."""
            process = await super()._open_app_process(width, height)
            session_processes.register(process.pid)
            return process

        async def stop(self) -> None:
            """Stop the session and forget it only after the process has actually exited."""
            try:
                await super().stop()
            finally:
                process = self._process
                if process is not None and process.returncode is not None:
                    session_processes.forget(process.pid)

    class MobileFriendlyServer(TextualServeServer):
        """A textual-serve server whose page addresses the client's own origin and fits a phone screen."""

        def __init__(self, command: str, *, host: str, port: int, title: str) -> None:
            super().__init__(command, host=host, port=port, title=title)
            self._serve_loop: asyncio.AbstractEventLoop | None = None
            self._exit_requested = threading.Event()
            self._serve_stopped = threading.Event()

        def serve(self, debug: bool = False) -> None:
            """Serve until stopped, publishing the loop so the host watcher can request a safe exit."""
            if self._exit_requested.is_set():
                self._serve_stopped.set()
                return
            try:
                self._serve_loop = asyncio.get_event_loop()
            except RuntimeError:
                self._serve_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._serve_loop)
            if self._exit_requested.is_set():
                self._serve_stopped.set()
                return
            try:
                super().serve(debug)
            finally:
                self._serve_stopped.set()

        def request_threadsafe_exit(self) -> None:
            """Ask aiohttp to unwind on its owning loop; safe before or during :meth:`serve`."""
            self._exit_requested.set()
            loop = self._serve_loop
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(self.request_exit)

        def wait_stopped(self, timeout: float) -> bool:
            """Wait up to ``timeout`` seconds for :meth:`serve` to return."""
            return self._serve_stopped.wait(timeout)

        async def handle_websocket(self, request: aiohttp_web.Request) -> aiohttp_web.WebSocketResponse:
            """Handle one browser session using the ownership-tracked Textual app service."""
            websocket = aiohttp_web.WebSocketResponse(heartbeat=15)
            width = to_int(request.query.get("width", "80"), 80)
            height = to_int(request.query.get("height", "24"), 24)
            app_service: TrackedAppService | None = None
            try:
                await websocket.prepare(request)

                async def close_websocket() -> None:
                    await websocket.close()

                app_service = TrackedAppService(
                    self.command,
                    write_bytes=websocket.send_bytes,
                    write_str=websocket.send_str,
                    close=close_websocket,
                    download_manager=self.download_manager,
                    debug=self.debug,
                )
                await app_service.start(width, height)
                await self._process_messages(websocket, app_service)
            except asyncio.CancelledError:
                await websocket.close()
            finally:
                if app_service is not None:
                    await app_service.stop()
            return websocket

        async def _make_app(self) -> aiohttp_web.Application:
            """Build the aiohttp application, adding the page rewrite to it."""
            app = await super()._make_app()
            native_dashboard.register(app)
            app.middlewares.append(self._rewrite_page_middleware)
            return app

        @aiohttp_web.middleware
        async def _rewrite_page_middleware(
            self,
            request: aiohttp_web.Request,
            handler: Handler,
        ) -> aiohttp_web.StreamResponse:
            """Point the served page at the client's own origin, and fit it to a phone viewport."""
            response = await handler(request)
            if request.path != "/":
                return response
            if not isinstance(response, aiohttp_web.Response) or response.content_type != "text/html":
                return response
            if not response.text:
                return response
            html = response.text
            # Only a wildcard bind produces an origin the client cannot route to. An explicit address is
            # left alone so a deliberate public_url (say, behind a reverse proxy) is never second-guessed.
            if host in WILDCARD_HOSTS:
                html = _rewrite_page_origin(
                    html,
                    served_origin=self.public_url,
                    client_origin=f"{request.scheme}://{request.host}",
                )
            response.text = _inject_mobile_fit(html)
            return response

    return MobileFriendlyServer(command, host=host, port=port, title=title)


def _print_remote_exposure_warning(host: str, port: int) -> None:
    """Print what binding a non-loopback address does, and what withholding the secrets does not cover."""
    print(
        f"\nServing the dashboard on {host}:{port}, reachable from other machines.\n"
        "  There is no authentication and no encryption. Anyone who can reach this port can start and\n"
        "  stop the worker, change every setting, and read the worker's logs.\n"
        "  The API key and Civitai token fields are withheld from the config editor, so they cannot be\n"
        "  read or replaced from the dashboard. The keys themselves stay in bridgeData.yaml on this\n"
        "  machine, and anything logged remains visible on the Logs tab.\n"
        "  Only do this on a network you trust.\n",
        file=sys.stderr,
    )


def _is_graphical_environment() -> bool:
    """Whether a browser can plausibly be opened on this machine.

    Windows and macOS always have a window server. On Linux a browser needs an X11 or Wayland
    display, so a server/SSH session with neither set is treated as headless. This is the signal the
    web launcher uses to avoid serving a dashboard nobody can open and to fall back to the terminal UI.
    """
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _run_terminal_fallback(args: argparse.Namespace) -> None:
    """Run the in-terminal TUI instead of serving a browser dashboard, mapping across the worker options."""
    from horde_worker_regen.tui import app as tui_app

    tui_argv = ["--process-mode", args.process_mode]
    if args.config:
        tui_argv += ["--config", args.config]
    if args.load_config_from_env_vars:
        tui_argv.append("-e")
    if args.amd:
        tui_argv.append("--amd")
    if args.worker_name:
        tui_argv += ["-n", args.worker_name]
    if args.directml is not None:
        tui_argv += ["--directml", str(args.directml)]
    tui_app.main(tui_argv)


def _host_running(address: tuple[str, int]) -> bool:
    """Whether a worker host already accepts connections at ``address``."""
    try:
        with socket.create_connection(address, timeout=0.5):
            return True
    except OSError:
        return False


def _announce_dashboard_port(address: tuple[str, int], web_port: int) -> None:
    """Update an already-running host's tray URL to this launcher's resolved web port."""
    try:
        with socket.create_connection(address, timeout=2.0) as sock:
            sp.send_frame(sock, sp.dashboard_port_message(web_port))
    except OSError as announce_error:
        print(f"Could not update the worker host's dashboard port ({announce_error}).", file=sys.stderr)


def _spawn_host(host_port: int, web_port: int, args: argparse.Namespace) -> subprocess.Popen[bytes]:
    """Launch the worker host as a child process, forwarding the worker options."""
    command = [
        sys.executable,
        "-m",
        "horde_worker_regen.tui.worker_host",
        "--port",
        str(host_port),
        "--process-mode",
        args.process_mode,
        "--dashboard-port",
        str(web_port),
    ]
    if args.load_config_from_env_vars:
        command.append("-e")
    if args.amd:
        command.append("--amd")
    if args.worker_name:
        command += ["-n", args.worker_name]
    if args.directml is not None:
        command += ["--directml", str(args.directml)]
    return subprocess.Popen(command)


def _shutdown_host(process: subprocess.Popen[bytes], address: tuple[str, int]) -> None:
    """Ask the host to stop the worker and exit cleanly, falling back to termination if it overruns."""
    try:
        with socket.create_connection(address, timeout=2.0) as sock:
            sp.send_frame(sock, sp.lifecycle_message(sp.LIFECYCLE_SHUTDOWN))
    except OSError:
        pass
    try:
        process.wait(timeout=_HOST_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.terminate()


def _await_host_socket(
    address: tuple[str, int],
    *,
    grace_seconds: float,
    stop_event: threading.Event | None = None,
) -> socket.socket | None:
    """Connect to the host, retrying until it is reachable or ``grace_seconds`` elapses (then None).

    The launcher may have just spawned the host, which needs a moment to bind; the attached case
    (a host already running) connects on the first try. A set ``stop_event`` abandons the wait at once and
    returns None, so the launcher can unwind the watcher without blocking out the full grace.
    """
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            return None
        try:
            return socket.create_connection(address, timeout=_HOST_WATCH_CONNECT_TIMEOUT_SECONDS)
        except OSError:
            # Interruptible wait: a set stop_event ends the retry immediately, otherwise pause and retry.
            if stop_event is not None:
                if stop_event.wait(0.5):
                    return None
            else:
                time.sleep(0.5)
    return None


def _watch_host_liveness(
    address: tuple[str, int],
    on_host_gone: Callable[[], None],
    *,
    grace_seconds: float = _HOST_WATCH_STARTUP_GRACE_SECONDS,
    stop_event: threading.Event | None = None,
) -> None:
    """Hold a connection to the host and call ``on_host_gone`` once it goes away (the launcher's leash).

    The host outlives any one browser session and carries the tray whose "Stop worker && exit" ends the
    host process directly, which the launcher's ``serve()`` (a separate process) cannot otherwise notice:
    it would keep serving a dead host as an invisible orphaned console. Watching the host's own control
    socket is the reliable, pid-reuse-immune way to learn it is gone, and it covers both the spawned and
    the attached case. A clean socket close is the authoritative signal; an explicit ``host_shutdown``
    frame, when present, just lets the caller log the host's exit with intent.

    ``on_host_gone`` is the launcher's shutdown leash, so it must fire only for a host that genuinely went
    away, never for a launcher that is unwinding on purpose. A set ``stop_event`` means the latter: the watcher
    then returns without firing the leash. Binding the watcher to that event is what keeps it from outliving
    its launcher as a daemon that later stops an unrelated lifecycle (e.g. once ``serve()`` returns, including
    in tests that exercise :func:`main` with a non-blocking server).
    """
    sock = _await_host_socket(address, grace_seconds=grace_seconds, stop_event=stop_event)
    if sock is not None:
        try:
            with sock:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        break
                    message = sp.recv_frame(sock)
                    if message is None:
                        break  # the host closed the connection: it is gone
                    if message.get("type") == sp.MSG_HOST_SHUTDOWN:
                        break  # the host announced it is tearing down
        except (OSError, ValueError):
            pass
    if stop_event is not None and stop_event.is_set():
        return  # the launcher is unwinding deliberately; the host-gone leash must not fire
    on_host_gone()


def _wind_down_launcher(
    server: _DashboardServer,
    session_processes: _DashboardSessionProcesses,
    *,
    graceful_timeout: float = _DASHBOARD_SHUTDOWN_GRACE_SECONDS,
) -> None:
    """Close the launcher without touching browser processes, with a bounded session-only reap fallback.

    The normal path asks aiohttp/textual-serve to close its WebSockets. Textual-serve then sends each TUI
    session its quit message and waits for it. If that cooperative path wedges, only the registered session
    shell trees are terminated; browsers were never registered and cannot be selected by this fallback.
    """
    print("Worker host has exited; closing the dashboard launcher.", file=sys.stderr)
    server.request_threadsafe_exit()
    if server.wait_stopped(graceful_timeout):
        return
    terminated = session_processes.terminate_all()
    if terminated:
        print(
            f"Dashboard sessions did not exit within {graceful_timeout:.1f}s; terminated session roots {terminated}.",
            file=sys.stderr,
        )
    if server.wait_stopped(_DASHBOARD_SESSION_KILL_GRACE_SECONDS):
        return
    # The exact owned session trees have already been reaped. A final hard exit cannot affect a browser,
    # because browser processes are outside the session registry and Windows session Job Object.
    os._exit(0)


def _query_host_status(address: tuple[str, int], *, timeout: float = 2.0) -> dict[str, object] | None:
    """Connect to a worker host and return its first status frame, or None if none is reachable.

    The host greets a new client with ``hello`` and then broadcasts a status frame within one control
    interval, so a short read loop is enough to capture the current worker state.
    """
    try:
        with socket.create_connection(address, timeout=timeout) as sock:
            sock.settimeout(timeout)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                message = sp.recv_frame(sock)
                if message is None:
                    return None
                if message.get("type") == sp.MSG_STATUS:
                    return message
            return None
    except OSError:
        return None


def _print_host_status(address: tuple[str, int], *, timeout: float = 2.0) -> int:
    """Print whether a worker host runs at ``address`` and its status; return a process exit code."""
    status = _query_host_status(address, timeout=timeout)
    if status is None:
        print(f"No worker host is running on {address[0]}:{address[1]}.")
        return 1
    running = "running" if status.get("worker_running") else "stopped"
    print(
        f"Worker host on {address[0]}:{address[1]}: worker {running} "
        f"(status={status.get('status')}, mode={status.get('mode')}).",
    )
    return 0


def _request_host_stop(address: tuple[str, int], *, timeout: float = 5.0) -> int:
    """Ask a running worker host to stop the worker and exit; return a process exit code.

    After sending the request the write side is half-closed and the socket drained to EOF. This makes
    the host consume the frame before we fully close: a bare close can race the host's reader and let an
    RST discard the still-buffered request (the launcher's own exit path tolerates that race via a
    process-terminate backstop this command has no handle for).
    """
    try:
        with socket.create_connection(address, timeout=min(timeout, 2.0)) as sock:
            sp.send_frame(sock, sp.lifecycle_message(sp.LIFECYCLE_SHUTDOWN))
            sock.shutdown(socket.SHUT_WR)
            sock.settimeout(timeout)
            with contextlib.suppress(OSError):
                while sock.recv(4096):
                    pass
    except OSError:
        print(f"No worker host is running on {address[0]}:{address[1]}.")
        return 1
    print(f"Asked the worker host on {address[0]}:{address[1]} to stop; it drains in-flight jobs first.")
    return 0


def _chromium_app_command(url: str) -> list[str] | None:
    """A command to open *url* as a borderless app window in an installed Chromium browser, or None.

    App mode (``--app=URL``) yields a tab-less, address-bar-less window that does not read as a web page,
    which is the point: it looks like the worker's own window without the confusion of a browser tab. We
    never install a browser; we only use one already present (Edge ships with Windows; Chrome/Chromium/
    Edge are common on Linux). When none is found the caller falls back to a normal browser tab.
    """
    candidates: list[str] = []
    if sys.platform == "win32":
        for base in (
            os.environ.get("PROGRAMFILES", ""),
            os.environ.get("PROGRAMFILES(X86)", ""),
            os.environ.get("LOCALAPPDATA", ""),
        ):
            if not base:
                continue
            candidates.append(os.path.join(base, r"Microsoft\Edge\Application\msedge.exe"))
            candidates.append(os.path.join(base, r"Google\Chrome\Application\chrome.exe"))
    for name in (
        "msedge",
        "microsoft-edge",
        "microsoft-edge-stable",
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "chrome",
        "brave-browser",
    ):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for exe in candidates:
        if exe and os.path.isfile(exe):
            return [exe, f"--app={url}"]
    return None


def _open_app_window(url: str) -> bool:
    """Launch the dashboard as a Chromium app window; return False when no suitable browser is found."""
    command = _chromium_app_command(url)
    if command is None:
        return False
    try:
        subprocess.Popen(command)
    except OSError:
        return False
    return True


def _open_dashboard(url: str, *, app_window: bool) -> None:
    """Open *url* as an app window when requested and possible, otherwise in the default browser."""
    if app_window and _open_app_window(url):
        return
    webbrowser.open(url)


def _schedule_dashboard_open(host: str, port: int, *, app_window: bool) -> None:
    """Open the dashboard shortly after the server starts (loopback only)."""
    if not _is_loopback(host):
        return
    url = f"http://{host}:{port}"
    threading.Timer(1.5, lambda: _open_dashboard(url, app_window=app_window)).start()


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point (``horde-worker-web``): ensure a host, serve the dashboard, open a browser."""
    args = _parse_args(argv)

    # Control commands act on an already-running host and exit; they never start a server or a worker.
    if args.status or args.stop:
        control_address = ("127.0.0.1", _resolve_host_port(args.host_port))
        raise SystemExit(_print_host_status(control_address) if args.status else _request_host_stop(control_address))

    dashboard_config = _load_dashboard_config(args.config)
    web_host = _resolve_host(args.host, dashboard_config)
    web_port = _resolve_port(args.port, dashboard_config)
    remote_exposed = not _is_loopback(web_host)

    # A loopback-only web dashboard is useless on a machine with no browser, so on a headless box fall
    # back to the in-terminal TUI (or, with no terminal either, point the user at the right mode).
    # Binding the LAN (--host) or suppressing the auto-open (--no-browser) is explicit "serve anyway"
    # intent (e.g. for a remote browser), so it skips the fallback.
    forced_serve = args.no_browser or remote_exposed
    if not _is_graphical_environment() and not forced_serve:
        if sys.stdout.isatty():
            print("No graphical display detected; opening the in-terminal dashboard instead of a browser.")
            _run_terminal_fallback(args)
            return
        print(
            "No graphical display and no interactive terminal were detected, so the web dashboard cannot "
            "be shown here. Use '--headless' to run the worker with no UI, or '--host 0.0.0.0' to serve "
            "the dashboard for a browser on another machine (unauthenticated; opt-in).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if remote_exposed:
        _print_remote_exposure_warning(web_host, web_port)

    host_port = _resolve_host_port(args.host_port)
    host_address = ("127.0.0.1", host_port)

    host_process = None if _host_running(host_address) else _spawn_host(host_port, web_port, args)
    if host_process is None:
        _announce_dashboard_port(host_address, web_port)

    if not args.no_browser:
        _schedule_dashboard_open(web_host, web_port, app_window=not args.browser)

    session_processes = _DashboardSessionProcesses()
    server = _build_server(
        _build_served_command(args, host_port, remote_exposed=remote_exposed),
        host=web_host,
        port=web_port,
        title="AI Horde Worker",
        host_address=host_address,
        session_processes=session_processes,
    )

    # Follow the host to the grave: if it exits on its own (notably the tray's "Stop worker && exit"),
    # this launcher must not linger as an orphaned console serving a dead host. The watcher requests a normal
    # server unwind and owns a bounded, session-only reap fallback, so it is bound to this launcher's lifetime:
    # once serve() returns the launcher is unwinding on its own terms and the watcher must not fire later.
    watch_stop = threading.Event()
    watcher = threading.Thread(
        target=_watch_host_liveness,
        args=(host_address, lambda: _wind_down_launcher(server, session_processes)),
        kwargs={"stop_event": watch_stop},
        name="host-liveness-watch",
        daemon=True,
    )
    watcher.start()

    try:
        server.serve()
    finally:
        watch_stop.set()
        if host_process is not None:
            _shutdown_host(host_process, host_address)
        watcher.join(timeout=_HOST_WATCH_STOP_JOIN_SECONDS)


if __name__ == "__main__":
    main()
