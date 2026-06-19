"""Best-effort, advisory checks against the AI Horde API for the setup wizard.

These never block setup. They exist to turn a cryptic failure twenty minutes into a run (a "wrong
credentials" rejection, a worker-name clash) into a hint at the moment the user types the value.

Every public check is blocking and meant to run off the UI thread, imports ``horde_sdk`` lazily so the
TUI parent process stays light until it is actually needed, and degrades to ``UNKNOWN`` on any
transport or SDK error so we stay silent rather than warn falsely when the user is simply offline.

The network calls are isolated in the ``_submit_*`` / ``_fetch_*`` helpers so tests can stub them
without a live horde.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class AdvisoryStatus(enum.StrEnum):
    """The outcome of an advisory check."""

    OK = "ok"
    """The value is good (key resolves to a user, name is free)."""
    PROBLEM = "problem"
    """The horde positively rejected the value (bad key, name already taken)."""
    UNKNOWN = "unknown"
    """Could not check (offline, SDK error); the caller should stay silent."""


@dataclass(frozen=True)
class AdvisoryResult:
    """An advisory status plus a short human detail (username, error message, worker id)."""

    status: AdvisoryStatus
    detail: str = ""


def _submit_find_user(api_key: str) -> Any:  # noqa: ANN401 - SDK response is a union we duck-type
    """Resolve *api_key* to a user via the horde ``find_user`` endpoint (blocking, off-thread)."""
    from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIManualClient
    from horde_sdk.ai_horde_api.apimodels import FindUserRequest, UserDetailsResponse

    return AIHordeAPIManualClient().submit_request(
        FindUserRequest(apikey=api_key),
        UserDetailsResponse,
    )


def verify_api_key(api_key: str) -> AdvisoryResult:
    """Resolve *api_key* to a horde user.

    Returns OK with the username when the key is valid, PROBLEM when the horde rejects it, and UNKNOWN
    on any error (so an offline user is never told their key is bad).
    """
    try:
        from horde_sdk.generic_api.apimodels import RequestErrorResponse

        response = _submit_find_user(api_key)
    except Exception:  # noqa: BLE001 - any failure means "could not check", not "invalid"
        return AdvisoryResult(AdvisoryStatus.UNKNOWN)
    if isinstance(response, RequestErrorResponse):
        return AdvisoryResult(AdvisoryStatus.PROBLEM, response.message)
    username = str(response.username or "")
    return AdvisoryResult(AdvisoryStatus.OK, username)


def _fetch_worker_details(worker_name: str) -> Any:  # noqa: ANN401 - SDK response, duck-typed
    """Look up an existing worker by name (blocking, off-thread)."""
    from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPISimpleClient

    return AIHordeAPISimpleClient().worker_details_by_name(worker_name=worker_name)


def check_worker_name_available(worker_name: str) -> AdvisoryResult:
    """Whether *worker_name* is free on the horde.

    Returns PROBLEM only on a positive collision (a worker with that name exists); a missing name or
    any lookup error degrades to UNKNOWN, so we only ever warn when we are sure.
    """
    try:
        details = _fetch_worker_details(worker_name)
    except Exception:  # noqa: BLE001 - a 404 / transport error is "could not confirm", not "free"
        return AdvisoryResult(AdvisoryStatus.UNKNOWN)
    if details is None:
        return AdvisoryResult(AdvisoryStatus.OK)
    return AdvisoryResult(AdvisoryStatus.PROBLEM, str(getattr(details, "id_", "") or ""))


__all__ = [
    "AdvisoryResult",
    "AdvisoryStatus",
    "check_worker_name_available",
    "verify_api_key",
]
