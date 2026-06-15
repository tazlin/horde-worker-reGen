"""Serve the worker dashboard in a web browser, backed by a persistent worker host.

This is the default launch path for non-technical users. It does two things:

1. Ensures a [`WorkerHost`][horde_worker_regen.tui.worker_host.WorkerHost] is running (spawning one if the
   host port is free), so a single worker is owned independently of any browser session.
2. Serves the TUI with ``textual-serve``, instructing each per-session TUI subprocess to *attach* to that
   host rather than own a worker. Closing a browser tab therefore leaves the worker running; closing this
   launcher stops the worker cleanly.

Network exposure is conservative: the web server binds ``127.0.0.1`` by default. Binding the LAN is a
deliberate power-user action via ``--host`` / ``HORDE_WORKER_WEB_HOST`` and exposes an unauthenticated
dashboard, so it must be opted into. (The worker host always binds loopback.)
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import webbrowser

from horde_worker_regen.tui import socket_protocol as sp

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
HOST_ENV_VAR = "HORDE_WORKER_WEB_HOST"
PORT_ENV_VAR = "HORDE_WORKER_WEB_PORT"
HOST_PORT_ENV_VAR = "HORDE_WORKER_HOST_PORT"

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_HOST_SHUTDOWN_TIMEOUT_SECONDS = 120.0


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
        help=f"Address to bind the web server (default {DEFAULT_HOST}; ${HOST_ENV_VAR} overrides). "
        "Use 0.0.0.0 to expose on the LAN (unauthenticated; opt-in).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Web server port (default {DEFAULT_PORT}; ${PORT_ENV_VAR} overrides).",
    )
    parser.add_argument(
        "--host-port",
        type=int,
        default=None,
        help=f"Worker-host socket port (default {sp.DEFAULT_HOST_PORT}; ${HOST_PORT_ENV_VAR} overrides).",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open any window automatically.")
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


def _resolve_host(arg_host: str | None) -> str:
    """Resolve the web bind host from the flag, then the environment, then the safe default."""
    return arg_host or os.getenv(HOST_ENV_VAR) or DEFAULT_HOST


def _resolve_port(arg_port: int | None) -> int:
    """Resolve the web port from the flag, then the environment, then the default."""
    if arg_port is not None:
        return arg_port
    env_port = os.getenv(PORT_ENV_VAR)
    return int(env_port) if env_port else DEFAULT_PORT


def _resolve_host_port(arg_host_port: int | None) -> int:
    """Resolve the worker-host socket port from the flag, then the environment, then the default."""
    if arg_host_port is not None:
        return arg_host_port
    env_port = os.getenv(HOST_PORT_ENV_VAR)
    return int(env_port) if env_port else sp.DEFAULT_HOST_PORT


def _build_served_command(args: argparse.Namespace, host_port: int) -> str:
    """Compose the per-session dashboard command, which attaches to the worker host."""
    parts = ["horde-worker", f"--attach 127.0.0.1:{host_port}", f"--process-mode {args.process_mode}"]
    if args.config:
        parts.append(f'--config "{args.config}"')
    return " ".join(parts)


def _host_running(address: tuple[str, int]) -> bool:
    """Whether a worker host already accepts connections at ``address``."""
    try:
        with socket.create_connection(address, timeout=0.5):
            return True
    except OSError:
        return False


def _spawn_host(host_port: int, args: argparse.Namespace) -> subprocess.Popen[bytes]:
    """Launch the worker host as a child process, forwarding the worker options."""
    command = [
        sys.executable,
        "-m",
        "horde_worker_regen.tui.worker_host",
        "--port",
        str(host_port),
        "--process-mode",
        args.process_mode,
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
    if host not in _LOOPBACK_HOSTS:
        return
    url = f"http://{host}:{port}"
    threading.Timer(1.5, lambda: _open_dashboard(url, app_window=app_window)).start()


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point (``horde-worker-web``): ensure a host, serve the dashboard, open a browser."""
    from textual_serve.server import Server

    args = _parse_args(argv)
    web_host = _resolve_host(args.host)
    web_port = _resolve_port(args.port)
    host_port = _resolve_host_port(args.host_port)
    host_address = ("127.0.0.1", host_port)

    host_process = None if _host_running(host_address) else _spawn_host(host_port, args)

    if not args.no_browser:
        _schedule_dashboard_open(web_host, web_port, app_window=not args.browser)

    server = Server(_build_served_command(args, host_port), host=web_host, port=web_port, title="AI Horde Worker")
    try:
        server.serve()
    finally:
        if host_process is not None:
            _shutdown_host(host_process, host_address)


if __name__ == "__main__":
    main()
