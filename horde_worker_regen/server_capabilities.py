"""Runtime detection of which features the connected AI Horde *server* actually supports.

Distinct from :mod:`horde_worker_regen.capabilities`, which probes what *this host* can run
(optional native packages). Here the question is what the remote server accepts, so the worker never
advertises something the server will reject.

The motivating case is the ``vectorize`` interrogation form. The server validates a worker's offered
pop ``forms`` against a fixed enum; offering a form the server does not list makes it reject the
*entire* pop request, which would silently break an alchemist for all of its forms (not just the new
one). A worker carrying support for a form can therefore be published ahead of the server's go-live,
as long as it only offers the form once the server advertises it.

The server publishes that enum in its OpenAPI/Swagger document
(``definitions.ModelInterrogationFormStable.properties.name.enum``), so this module reads it and
exposes a fail-closed lookup. "Fail-closed" means: until a probe has succeeded, every form is treated
as unsupported, so a pre-go-live deployment never breaks pops. The probe is refreshed on a TTL, so a
long-running worker begins offering a newly-enabled form within the TTL of the server going live,
without a restart. Probe failures back off on a shorter interval and never clobber a prior good
result, so a transient outage does not drop a form already known to be supported.
"""

from __future__ import annotations

import time

import aiohttp
from horde_sdk.ai_horde_api.endpoints import get_ai_horde_swagger_url
from loguru import logger

SUCCESS_TTL_SECONDS = 1800.0
"""How long a successful probe is trusted before the next refresh (server feature sets change rarely)."""

FAILURE_RETRY_SECONDS = 60.0
"""How soon to retry after a failed probe, short enough to pick up go-live promptly without hammering."""

_SWAGGER_FETCH_TIMEOUT_SECONDS = 10.0

# Path into the Swagger 2.0 document to the interrogation form-name enum. Flask-RESTX emits Swagger
# 2.0 (`definitions`); `components.schemas` is the OpenAPI 3 fallback should the server ever migrate.
_FORM_SCHEMA_NAME = "ModelInterrogationFormStable"
_FORM_ENUM_KEYS = ("properties", "name", "enum")

_supported_interrogation_forms: frozenset[str] | None = None
_next_refresh_monotonic: float = 0.0


def _parse_interrogation_form_enum(spec: dict[str, object]) -> frozenset[str]:
    """Extract the interrogation form-name enum from a parsed Swagger/OpenAPI document.

    Raises a ``KeyError``/``TypeError`` if the expected path is absent; callers treat any failure as
    "unknown" (fail-closed).
    """
    schemas = spec.get("definitions")
    if not isinstance(schemas, dict):
        components = spec.get("components")
        schemas = components.get("schemas") if isinstance(components, dict) else None
    if not isinstance(schemas, dict):
        raise KeyError("no definitions/components.schemas in spec")

    node: object = schemas[_FORM_SCHEMA_NAME]
    for key in _FORM_ENUM_KEYS:
        if not isinstance(node, dict):
            raise TypeError(f"unexpected swagger shape at {key!r}")
        node = node[key]
    if not isinstance(node, list):
        raise TypeError("form enum is not a list")
    return frozenset(str(form) for form in node)


async def _fetch_swagger_spec(url: str) -> dict[str, object]:
    """Fetch and parse the server's Swagger/OpenAPI document. Separated out as a test seam."""
    async with (
        aiohttp.ClientSession() as session,
        session.get(url, timeout=aiohttp.ClientTimeout(total=_SWAGGER_FETCH_TIMEOUT_SECONDS)) as response,
    ):
        response.raise_for_status()
        # The server may not label the swagger doc as application/json; don't enforce it.
        return await response.json(content_type=None)


async def refresh_server_capabilities(*, force: bool = False) -> None:
    """Refresh the cached set of server-supported interrogation forms, honouring the TTL.

    A no-op when the cache is still fresh (unless ``force``). Designed to be called once per
    alchemy loop iteration: it self-throttles, fetches off no hot path, and never raises (a probe
    failure only logs and schedules an earlier retry).
    """
    global _supported_interrogation_forms, _next_refresh_monotonic

    now = time.monotonic()
    if not force and now < _next_refresh_monotonic:
        return

    url = get_ai_horde_swagger_url()
    try:
        spec = await _fetch_swagger_spec(url)
        forms = _parse_interrogation_form_enum(spec)
    except Exception as exc:
        # Keep any prior good result; just retry sooner. Fail-closed only matters before the first
        # success, when the cache is still None.
        _next_refresh_monotonic = now + FAILURE_RETRY_SECONDS
        logger.warning(f"Could not probe server interrogation-form support from {url}: {type(exc).__name__} {exc}")
        return

    if forms != _supported_interrogation_forms:
        logger.info(f"Server-supported interrogation forms: {sorted(forms)}")
    _supported_interrogation_forms = forms
    _next_refresh_monotonic = now + SUCCESS_TTL_SECONDS


def server_supports_interrogation_form(form: str) -> bool:
    """Return whether the server is known to support *form* (fail-closed before the first probe)."""
    return _supported_interrogation_forms is not None and form in _supported_interrogation_forms


def reset_server_capabilities_cache() -> None:
    """Clear the cached probe result. For tests and forced re-probing."""
    global _supported_interrogation_forms, _next_refresh_monotonic
    _supported_interrogation_forms = None
    _next_refresh_monotonic = 0.0
