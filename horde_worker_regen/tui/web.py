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
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
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
    """Compose the per-session dashboard command, which attaches to the worker host.

    The TUI is invoked via ``python -m`` rather than the ``horde-worker`` console script because
    textual-serve launches this through the shell (``cmd.exe /c`` on Windows), and cmd.exe resolves
    bare names against the current directory before PATH. The web server runs from the repo root, whose
    ``horde-worker.cmd`` launcher would otherwise shadow the console script and re-invoke the *web*
    server. Module invocation cannot be shadowed and mirrors how the worker host is spawned.
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
    return " ".join(parts)


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


def _print_host_status(address: tuple[str, int]) -> int:
    """Print whether a worker host runs at ``address`` and its status; return a process exit code."""
    status = _query_host_status(address)
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
        with socket.create_connection(address, timeout=2.0) as sock:
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
    if host not in _LOOPBACK_HOSTS:
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

    web_host = _resolve_host(args.host)
    web_port = _resolve_port(args.port)

    # A loopback-only web dashboard is useless on a machine with no browser, so on a headless box fall
    # back to the in-terminal TUI (or, with no terminal either, point the user at the right mode).
    # Binding the LAN (--host) or suppressing the auto-open (--no-browser) is explicit "serve anyway"
    # intent (e.g. for a remote browser), so it skips the fallback.
    forced_serve = args.no_browser or web_host not in _LOOPBACK_HOSTS
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

    from textual_serve.server import Server

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
