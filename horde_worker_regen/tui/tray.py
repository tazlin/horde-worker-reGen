"""A Windows system-tray icon for a running worker host, so a detached worker stays visible.

The host ([`WorkerHost`][horde_worker_regen.tui.worker_host.WorkerHost]) is the process that outlives every
browser tab and the launcher console. On Windows a hard-closed launcher (the window's close button or
``taskkill``) skips the launcher's clean shutdown and can leave that host running with no console, no
browser, and nothing in the notification area: an invisible worker. This module gives the persistent host a
tray icon whose menu can reopen the dashboard or stop the worker cleanly, so a running worker is always
visible and stoppable without resorting to Task Manager.

The tray is optional and best-effort. ``pystray``/``Pillow`` are Windows-only extras and may be absent, and
the icon is only meaningful on Windows, so when either condition fails :class:`WorkerTray` is an inert
no-op. Nothing here raises into the host's control path: a tray failure must never take the worker down.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import pystray
    from PIL import Image, ImageDraw

    _TRAY_IMPORT_OK = True
else:
    try:
        import pystray
        from PIL import Image, ImageDraw

        _TRAY_IMPORT_OK = True
    except ImportError:
        # The tray extras are not installed; the host still runs, just without an icon.
        pystray = None
        Image = None
        ImageDraw = None
        _TRAY_IMPORT_OK = False


def tray_supported() -> bool:
    """Whether a tray icon can be shown here: Windows with the ``pystray``/``Pillow`` extras installed."""
    return _TRAY_IMPORT_OK and sys.platform == "win32"


def _web_server_running(port: int, host: str = "127.0.0.1") -> bool:
    """Whether a web dashboard server already accepts connections at ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def open_dashboard(web_port: int = 8000) -> None:
    """Open the dashboard for the already-running host: reuse a live web server, otherwise start one.

    The host does not run the web server itself (the launcher does), so this either points a browser at an
    existing server or spawns a fresh launcher. A spawned launcher detects this host on its socket and
    serves a session *attached* to it rather than spawning a second worker.
    """
    url = f"http://127.0.0.1:{web_port}"
    if _web_server_running(web_port):
        import webbrowser

        webbrowser.open(url)
        return
    try:
        subprocess.Popen([sys.executable, "-m", "horde_worker_regen.tui.web"])
    except OSError:
        logger.opt(exception=True).warning("Could not launch the web dashboard from the tray.")


def _create_icon_image() -> Image.Image:
    """Draw the tray glyph in code so no packaged image asset is needed: a blue tile with a white 'H'."""
    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, size - 3, size - 3), radius=12, fill=(46, 134, 222, 255))
    white = (255, 255, 255, 255)
    draw.rectangle((20, 16, 27, 48), fill=white)
    draw.rectangle((37, 16, 44, 48), fill=white)
    draw.rectangle((20, 29, 44, 35), fill=white)
    return image


class WorkerTray:
    """A best-effort Windows tray icon for a running worker host.

    Construct it with the callbacks its menu should invoke and a provider for the live status line, then
    call :meth:`start`. When the tray is unsupported every method is a safe no-op, so the host needs no
    platform guard of its own.
    """

    def __init__(
        self,
        *,
        on_open_dashboard: Callable[[], None],
        on_stop: Callable[[], None],
        status_provider: Callable[[], str],
        title: str = "AI Horde Worker",
    ) -> None:
        """Store the menu callbacks and status provider; the icon is not created until :meth:`start`."""
        self._on_open_dashboard = on_open_dashboard
        self._on_stop = on_stop
        self._status_provider = status_provider
        self._title = title
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Show the tray icon on a background thread; a no-op when the tray is unsupported or already up."""
        if not tray_supported() or self._icon is not None:
            return
        try:
            menu = pystray.Menu(
                # A non-clickable live status line; pystray re-reads the callable each time the menu opens.
                pystray.MenuItem(lambda item: self._status_provider(), None, enabled=False),
                pystray.Menu.SEPARATOR,
                # "&&" renders a literal "&": a single "&" is a Windows menu mnemonic marker.
                pystray.MenuItem("Open dashboard", self._handle_open, default=True),
                pystray.MenuItem(
                    "Exit Now (Stop the worker before you do this)", self._handle_stop
                ),  # This instantly kills the process, which is not ideal
            )
            self._icon = pystray.Icon("horde-worker", _create_icon_image(), title=self._title, menu=menu)
            self._thread = threading.Thread(target=self._icon.run, name="worker-tray", daemon=True)
            self._thread.start()
            logger.debug("Worker tray icon started.")
        except Exception:
            logger.opt(exception=True).warning("Failed to start the worker tray icon; continuing without it.")
            self._icon = None

    def _handle_open(self, icon: object = None, item: object = None) -> None:
        """Tray menu action: open the dashboard, swallowing any failure so the icon stays responsive."""
        try:
            self._on_open_dashboard()
        except Exception:
            logger.opt(exception=True).warning("Tray 'Open dashboard' action failed.")

    def _handle_stop(self, icon: object = None, item: object = None) -> None:
        """Tray menu action: stop the worker, then remove the icon (the host is exiting)."""
        try:
            self._on_stop()
        finally:
            self.stop()

    def stop(self) -> None:
        """Remove the tray icon; safe to call when it was never started or is already stopped."""
        icon = self._icon
        if icon is None:
            return
        self._icon = None
        try:
            icon.stop()
        except Exception:
            logger.opt(exception=True).debug("Error stopping the tray icon (ignored).")
