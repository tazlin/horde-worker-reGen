"""Startup fail-fast checks for the worker's configured names.

Worker names are unique horde-wide and are tied to the API key that first registers them. The
config template ships with reserved placeholder names, and each worker *type* (the image "dreamer"
and the alchemy "alchemist") registers as a separate, uniquely-named worker. Getting this wrong
otherwise surfaces only as a late, cryptic "Wrong credentials to submit as this worker" at pop time.

This module verifies the configuration *before* any processes spawn:

1. A local check (no network): names must not be the reserved defaults, and the alchemist name must
   differ from the dreamer name when alchemy is enabled.
2. A network check: each enabled name must be either unregistered (a brand-new worker, the normal
   first-run case) or already owned by the configured API key. The name is resolved through the
   single-worker-by-name endpoint, *not* the all-workers list: the list only returns workers that are
   currently active, so an idle (offline) worker registered under the name is invisible there and a
   collision would slip past this check, only to fail later at pop time. The by-name endpoint finds
   the worker regardless of activity, and its ``WorkerNotFound`` response is the genuine "name is free"
   signal. Ownership is then accepted on *either* the worker id appearing in the account's worker_ids
   OR the worker's owner matching the authenticated username, because the user-details worker_ids list
   can lag or omit a worker the account genuinely owns (right after a fresh registration, or once an
   idle worker is pruned from the list while still findable by name). Per the project's chosen policy
   this hard-fails on *any* failure, including the API being unreachable (after a small bounded retry),
   so the worker never silently runs under a name the horde will reject.
"""

from __future__ import annotations

import time

from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIClientSession, AIHordeAPISimpleClient
from horde_sdk.ai_horde_api.apimodels import (
    FindUserRequest,
    SingleWorkerDetailsResponse,
    SingleWorkerNameDetailsRequest,
    UserDetailsResponse,
    WorkerDetailItem,
)
from horde_sdk.ai_horde_api.consts import RC
from horde_sdk.generic_api.apimodels import RequestErrorResponse
from loguru import logger

from horde_worker_regen.bridge_data.data_model import reGenBridgeData

_OWNERSHIP_CHECK_ATTEMPTS = 3
"""How many times the network ownership check is attempted before hard-failing on a transient error."""

_OWNERSHIP_CHECK_RETRY_DELAY_SECONDS = 2.0
"""Delay between ownership-check attempts after a transient (e.g. network) failure."""


class WorkerNameConfigError(Exception):
    """Raised when a worker name is a reserved default, duplicated, or owned by another account."""


def verify_worker_identity(bridge_data: reGenBridgeData) -> None:
    """Fail fast on a worker-name misconfiguration before any work begins.

    Raises:
        WorkerNameConfigError: If a name is a reserved default, the dreamer/alchemist names collide,
            a name is owned by a different account, or ownership cannot be verified.
    """
    _validate_worker_names_local(bridge_data)

    if bridge_data.dry_run_skip_api:
        logger.warning("dry_run_skip_api is set; skipping the worker-name ownership check.")
        return

    _verify_worker_names_owned(bridge_data)


def _validate_worker_names_local(bridge_data: reGenBridgeData) -> None:
    """Reject reserved-default or colliding worker names without touching the network."""
    fields = type(bridge_data).model_fields
    dreamer_default = fields["dreamer_worker_name"].default
    alchemist_default = fields["alchemist_name"].default

    if bridge_data.dreamer_worker_name == dreamer_default:
        raise WorkerNameConfigError(
            f"Your worker name is still the default ({dreamer_default!r}). Set a unique `dreamer_name` "
            "in bridgeData.yaml; the default is reserved and the horde will reject it.",
        )

    if bridge_data.alchemist:
        if bridge_data.alchemist_name == alchemist_default:
            raise WorkerNameConfigError(
                f"Alchemy is enabled but `alchemist_name` is still the default ({alchemist_default!r}). "
                "Set a unique `alchemist_name` in bridgeData.yaml; the default is reserved.",
            )
        if bridge_data.alchemist_name == bridge_data.dreamer_worker_name:
            raise WorkerNameConfigError(
                "`alchemist_name` must differ from `dreamer_name`: each worker type registers as a "
                "separate, uniquely-named worker on the horde.",
            )


def _verify_worker_names_owned(bridge_data: reGenBridgeData) -> None:
    """Verify each enabled worker name is unregistered or owned by this API key (network, hard-fail)."""
    names = [bridge_data.dreamer_worker_name]
    if bridge_data.alchemist:
        names.append(bridge_data.alchemist_name)

    last_error: Exception | None = None
    for attempt in range(_OWNERSHIP_CHECK_ATTEMPTS):
        try:
            owned_worker_ids, account_username = _fetch_account_identity(bridge_data.api_key)
            for name in names:
                worker = _lookup_registered_worker(name, bridge_data.api_key)
                if worker is None:
                    logger.info(f"Worker name {name!r} is not yet registered; it will be created on first pop.")
                    continue
                if not _worker_is_owned_by_account(worker, owned_worker_ids, account_username):
                    raise WorkerNameConfigError(
                        f"Worker name {name!r} is already registered to another account "
                        f"(owner: {worker.owner or 'unknown'}). Worker names are unique horde-wide; "
                        "choose a different name in bridgeData.yaml.",
                    )
                logger.debug(f"Worker name {name!r} is owned by this account ({worker.id_}).")
            return
        except WorkerNameConfigError:
            raise  # A definitive verdict, not a transient error; do not retry.
        except Exception as network_error:  # noqa: BLE001 - any other failure is retried then hard-fails
            last_error = network_error
            logger.warning(
                f"Worker-name ownership check attempt {attempt + 1}/{_OWNERSHIP_CHECK_ATTEMPTS} failed: "
                f"{network_error}",
            )
            if attempt < _OWNERSHIP_CHECK_ATTEMPTS - 1:
                time.sleep(_OWNERSHIP_CHECK_RETRY_DELAY_SECONDS)

    raise WorkerNameConfigError(
        f"Could not verify worker-name ownership with the AI Horde API after {_OWNERSHIP_CHECK_ATTEMPTS} "
        f"attempts: {last_error}. The worker will not start until the API is reachable and the "
        "configuration is valid.",
    )


def _fetch_account_identity(api_key: str) -> tuple[set[str], str | None]:
    """Return the worker IDs and username for the account behind ``api_key``.

    The username is returned alongside the ids so the ownership check can fall back to an owner-name
    match when ``worker_ids`` does not yet (or no longer) list a worker the account actually owns.
    """
    with AIHordeAPIClientSession() as session:
        response = session.submit_request(FindUserRequest(apikey=api_key), UserDetailsResponse)
    if isinstance(response, RequestErrorResponse):
        raise RuntimeError(f"find_user returned an error: {response.message}")
    worker_ids = {str(worker_id) for worker_id in (response.worker_ids or [])}
    return worker_ids, response.username


def _lookup_registered_worker(name: str, api_key: str) -> WorkerDetailItem | None:
    """Return the worker registered under ``name``, or None only when the name is genuinely free.

    Uses the single-worker-by-name endpoint rather than the all-workers list: the list only returns
    *active* workers, so an idle worker registered under the name would be invisible there and a name
    collision would slip past the preflight, surfacing only as a cryptic credentials error at pop time.

    The error is handled by meaning, not swallowed: a ``WorkerNotFound`` response is the one signal that
    the name is unregistered (mapped to None, the normal first-run case). Any *other* error response is
    raised so the caller's retry/hard-fail path treats it as a transient or definitive failure to verify,
    never as "the name is free".
    """
    with AIHordeAPIClientSession() as session:
        response = session.submit_request(
            SingleWorkerNameDetailsRequest(worker_name=name, apikey=api_key),
            SingleWorkerDetailsResponse,
        )
    if isinstance(response, RequestErrorResponse):
        if response.rc == RC.WorkerNotFound:
            return None
        raise RuntimeError(f"worker-details-by-name returned an error for {name!r}: {response.message}")
    return response


def _worker_is_owned_by_account(
    worker: WorkerDetailItem,
    owned_worker_ids: set[str],
    account_username: str | None,
) -> bool:
    """Whether a registered ``worker`` belongs to the authenticated account.

    Ownership is accepted on *either* signal: the worker id appears in the account's ``worker_ids``,
    or the worker's ``owner`` matches the authenticated ``username``. The owner-name match is the
    robust fallback: ``worker_ids`` can lag or omit a worker the account genuinely owns (a freshly
    registered worker, or an idle one pruned from the list while still findable by name), and relying
    on it alone falsely rejected an owned worker as "another account", refusing to start. Usernames
    are unique horde-wide (they carry a discriminator), so an owner/username match is a safe
    same-account signal.
    """
    if str(worker.id_) in owned_worker_ids:
        return True
    owner = (worker.owner or "").strip().lower()
    username = (account_username or "").strip().lower()
    return bool(owner) and owner == username


def lookup_worker_by_name(
    simple_client: AIHordeAPISimpleClient,
    name: str,
    *,
    api_key: str | None = None,
) -> WorkerDetailItem | None:
    """Return the active worker registered under ``name`` (case-insensitive), or None if none is found.

    Uses the *list* endpoint with a name filter so a missing worker is an empty result rather than an
    exception, and re-checks the name client-side so the result is correct regardless of how the
    server interprets the filter. The list endpoint only returns *active* workers, which suits its
    callers (e.g. toggling maintenance on the worker you are currently running). The startup ownership
    preflight deliberately does not use this: it must also see idle workers, so it goes through the
    single-worker-by-name endpoint instead (see ``_lookup_registered_worker``).
    """
    response = simple_client.workers_all_details(worker_name=name, api_key=api_key)
    for worker in response:
        if worker.name is not None and worker.name.lower() == name.lower():
            return worker
    return None
