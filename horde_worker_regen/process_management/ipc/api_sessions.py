"""Set-once HTTP session handles shared across process-management components."""

from __future__ import annotations

import aiohttp
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIAsyncClientSession


class ApiSessions:
    """HTTP session handles that the main loop initializes once after startup.

    The main process-manager loop sets both sessions once the event loop is running;
    before that, the ``require_*`` accessors raise ``RuntimeError``. Components that
    only need HTTP handles should depend on this class rather than the full context.
    """

    _horde_client_session: AIHordeAPIAsyncClientSession | None
    _aiohttp_session: aiohttp.ClientSession | None

    def __init__(self) -> None:
        """Create empty session slots. Sessions are populated later by the main loop."""
        self._horde_client_session = None
        self._aiohttp_session = None

    def set_horde_client_session(self, session: AIHordeAPIAsyncClientSession) -> None:
        """Store the horde-sdk client session. Called once by the main loop."""
        self._horde_client_session = session

    def set_aiohttp_session(self, session: aiohttp.ClientSession) -> None:
        """Store the aiohttp client session. Called once by the main loop."""
        self._aiohttp_session = session

    def require_horde_client_session(self) -> AIHordeAPIAsyncClientSession:
        """Return the horde client session, raising if it was not set yet.

        Raises:
            RuntimeError: If called before the main loop has set the session.
        """
        if self._horde_client_session is None:
            raise RuntimeError(
                "horde_client_session accessed before main loop initialization",
            )
        return self._horde_client_session

    def require_aiohttp_session(self) -> aiohttp.ClientSession:
        """Return the aiohttp session, raising if it was not set yet.

        Raises:
            RuntimeError: If called before the main loop has set the session.
        """
        if self._aiohttp_session is None:
            raise RuntimeError(
                "aiohttp_session accessed before main loop initialization",
            )
        return self._aiohttp_session
