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

The same reasoning applies to the *generation-metadata type* enum
(``definitions.GenerationMetadataStable.properties.type.enum``). The worker attaches an aesthetic
score to every image generation as a ``gen_metadata`` entry, and the server validates each entry's
``type`` against that enum, rejecting the *whole* submit if it sees an unknown type. So the aesthetic
score is only produced once the server advertises the ``aesthetic_score`` type, letting a worker that
carries the feature ship ahead of the server's go-live. Both enums are read from the one Swagger fetch.
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

# Paths into the Swagger 2.0 document to the enums we probe. Flask-RESTX emits Swagger 2.0
# (`definitions`); `components.schemas` is the OpenAPI 3 fallback should the server ever migrate.
_FORM_SCHEMA_NAME = "ModelInterrogationFormStable"
_FORM_ENUM_KEYS = ("properties", "name", "enum")

_METADATA_TYPE_SCHEMA_NAME = "GenerationMetadataStable"
_METADATA_TYPE_ENUM_KEYS = ("properties", "type", "enum")

_supported_interrogation_forms: frozenset[str] | None = None
_supported_generation_metadata_types: frozenset[str] | None = None
_next_refresh_monotonic: float = 0.0


def _extract_schemas(spec: dict[str, object]) -> dict[str, object]:
    """Return the schema table from a parsed Swagger 2.0 (`definitions`) or OpenAPI 3 document.

    Raises ``KeyError`` if neither is present; callers treat any failure as "unknown" (fail-closed).
    """
    schemas = spec.get("definitions")
    if not isinstance(schemas, dict):
        components = spec.get("components")
        schemas = components.get("schemas") if isinstance(components, dict) else None
    if not isinstance(schemas, dict):
        raise KeyError("no definitions/components.schemas in spec")
    return schemas


def _parse_enum(schemas: dict[str, object], schema_name: str, enum_keys: tuple[str, ...]) -> frozenset[str]:
    """Extract a string enum at ``schema_name`` + ``enum_keys`` from a schema table.

    Raises a ``KeyError``/``TypeError`` if the expected path is absent; callers treat any failure as
    "unknown" (fail-closed).
    """
    node: object = schemas[schema_name]
    for key in enum_keys:
        if not isinstance(node, dict):
            raise TypeError(f"unexpected swagger shape at {key!r}")
        node = node[key]
    if not isinstance(node, list):
        raise TypeError(f"enum at {schema_name} is not a list")
    return frozenset(str(value) for value in node)


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

    A no-op when the cache is still fresh (unless ``force``). Designed to be called once per pop-loop
    iteration (alchemy and image generation both): it self-throttles, fetches off no hot path, and
    never raises (a probe failure only logs and schedules an earlier retry).
    """
    global _supported_interrogation_forms, _supported_generation_metadata_types, _next_refresh_monotonic

    now = time.monotonic()
    if not force and now < _next_refresh_monotonic:
        return

    url = get_ai_horde_swagger_url()
    try:
        spec = await _fetch_swagger_spec(url)
        schemas = _extract_schemas(spec)
        forms = _parse_enum(schemas, _FORM_SCHEMA_NAME, _FORM_ENUM_KEYS)
        metadata_types = _parse_enum(schemas, _METADATA_TYPE_SCHEMA_NAME, _METADATA_TYPE_ENUM_KEYS)
    except Exception as exc:
        # Keep any prior good result; just retry sooner. Fail-closed only matters before the first
        # success, when the caches are still None.
        _next_refresh_monotonic = now + FAILURE_RETRY_SECONDS
        logger.warning(f"Could not probe server capabilities from {url}: {type(exc).__name__} {exc}")
        return

    if forms != _supported_interrogation_forms:
        logger.info(f"Server-supported interrogation forms: {sorted(forms)}")
    if metadata_types != _supported_generation_metadata_types:
        logger.info(f"Server-supported generation-metadata types: {sorted(metadata_types)}")
    _supported_interrogation_forms = forms
    _supported_generation_metadata_types = metadata_types
    _next_refresh_monotonic = now + SUCCESS_TTL_SECONDS


def server_supports_interrogation_form(form: str) -> bool:
    """Return whether the server is known to support *form* (fail-closed before the first probe)."""
    return _supported_interrogation_forms is not None and form in _supported_interrogation_forms


def server_supports_generation_metadata_type(metadata_type: str) -> bool:
    """Return whether the server accepts *metadata_type* on a generation (fail-closed before probe).

    The server rejects an entire generation submit if it carries a ``gen_metadata`` entry whose
    ``type`` it does not recognise, so an optional metadata attachment must be withheld until this
    returns ``True``.
    """
    return _supported_generation_metadata_types is not None and metadata_type in _supported_generation_metadata_types


def reset_server_capabilities_cache() -> None:
    """Clear the cached probe result. For tests and forced re-probing."""
    global _supported_interrogation_forms, _supported_generation_metadata_types, _next_refresh_monotonic
    _supported_interrogation_forms = None
    _supported_generation_metadata_types = None
    _next_refresh_monotonic = 0.0
