"""Tests for the server-capability probe that gates features on what the server actually supports."""

from __future__ import annotations

import pytest

from horde_worker_regen import server_capabilities
from horde_worker_regen.server_capabilities import (
    _FORM_ENUM_KEYS,
    _FORM_SCHEMA_NAME,
    _METADATA_TYPE_ENUM_KEYS,
    _METADATA_TYPE_SCHEMA_NAME,
    _parse_enum,
    refresh_server_capabilities,
    reset_server_capabilities_cache,
    server_supports_generation_metadata_type,
    server_supports_interrogation_form,
)


def _swagger_2_doc(
    forms: list[str],
    metadata_types: list[str] | None = None,
) -> dict[str, object]:
    """A minimal Swagger 2.0 document carrying both probed enums."""
    if metadata_types is None:
        metadata_types = ["lora", "censorship"]
    return {
        "definitions": {
            _FORM_SCHEMA_NAME: {"properties": {"name": {"enum": forms}}},
            _METADATA_TYPE_SCHEMA_NAME: {"properties": {"type": {"enum": metadata_types}}},
        },
    }


def _openapi_3_doc(
    forms: list[str],
    metadata_types: list[str] | None = None,
) -> dict[str, object]:
    """The OpenAPI 3 (`components.schemas`) shape, the fallback if the server ever migrates."""
    if metadata_types is None:
        metadata_types = ["lora", "censorship"]
    return {
        "components": {
            "schemas": {
                _FORM_SCHEMA_NAME: {"properties": {"name": {"enum": forms}}},
                _METADATA_TYPE_SCHEMA_NAME: {"properties": {"type": {"enum": metadata_types}}},
            },
        },
    }


def _schemas(doc: dict[str, object]) -> dict[str, object]:
    """Pull the schema table out of either swagger shape for direct `_parse_enum` calls."""
    definitions = doc.get("definitions")
    if isinstance(definitions, dict):
        return definitions
    components = doc["components"]
    assert isinstance(components, dict)
    schemas = components["schemas"]
    assert isinstance(schemas, dict)
    return schemas


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    reset_server_capabilities_cache()


class TestParseEnum:
    """Each enum is read from either swagger shape, and absence is an error."""

    def test_parses_swagger_2(self) -> None:
        """The Swagger 2.0 `definitions` shape yields the form enum."""
        schemas = _schemas(_swagger_2_doc(["caption", "vectorize"]))
        assert _parse_enum(schemas, _FORM_SCHEMA_NAME, _FORM_ENUM_KEYS) == frozenset({"caption", "vectorize"})

    def test_parses_openapi_3_fallback(self) -> None:
        """The OpenAPI 3 `components.schemas` shape yields the form enum."""
        schemas = _schemas(_openapi_3_doc(["nsfw"]))
        assert _parse_enum(schemas, _FORM_SCHEMA_NAME, _FORM_ENUM_KEYS) == frozenset({"nsfw"})

    def test_parses_metadata_type_enum(self) -> None:
        """The generation-metadata type enum is read from its own schema."""
        schemas = _schemas(_swagger_2_doc(["caption"], ["lora", "aesthetic_score"]))
        assert _parse_enum(schemas, _METADATA_TYPE_SCHEMA_NAME, _METADATA_TYPE_ENUM_KEYS) == frozenset(
            {"lora", "aesthetic_score"},
        )

    def test_raises_on_missing_path(self) -> None:
        """A schema table missing the expected schema raises (so the caller can fail-closed)."""
        with pytest.raises((KeyError, TypeError)):
            _parse_enum({}, _FORM_SCHEMA_NAME, _FORM_ENUM_KEYS)


class TestServerSupportGate:
    """The cached gate is fail-closed, refreshes on success, and survives transient failures."""

    def test_fail_closed_before_any_probe(self) -> None:
        """With no successful probe yet, every form reads as unsupported."""
        assert server_supports_interrogation_form("vectorize") is False
        assert server_supports_interrogation_form("caption") is False

    @pytest.mark.asyncio
    async def test_refresh_populates_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A successful probe makes listed forms read as supported and others not."""

        async def _fake_fetch(url: str) -> dict[str, object]:
            return _swagger_2_doc(["caption", "interrogation", "nsfw", "vectorize"])

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _fake_fetch)
        await refresh_server_capabilities(force=True)

        assert server_supports_interrogation_form("vectorize") is True
        assert server_supports_interrogation_form("not_a_form") is False

    @pytest.mark.asyncio
    async def test_server_without_form_is_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A successful probe whose enum lacks the form reports it unsupported."""

        async def _fake_fetch(url: str) -> dict[str, object]:
            return _swagger_2_doc(["caption", "interrogation", "nsfw"])

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _fake_fetch)
        await refresh_server_capabilities(force=True)

        assert server_supports_interrogation_form("vectorize") is False

    @pytest.mark.asyncio
    async def test_failure_keeps_prior_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A transient probe failure must not drop a form already known to be supported."""

        async def _good_fetch(url: str) -> dict[str, object]:
            return _swagger_2_doc(["vectorize"])

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _good_fetch)
        await refresh_server_capabilities(force=True)
        assert server_supports_interrogation_form("vectorize") is True

        async def _failing_fetch(url: str) -> dict[str, object]:
            raise ConnectionError("boom")

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _failing_fetch)
        await refresh_server_capabilities(force=True)
        assert server_supports_interrogation_form("vectorize") is True

    @pytest.mark.asyncio
    async def test_ttl_skips_redundant_refresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Within the TTL, a non-forced refresh does not re-fetch."""
        calls = 0

        async def _counting_fetch(url: str) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return _swagger_2_doc(["vectorize"])

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _counting_fetch)
        await refresh_server_capabilities(force=True)
        await refresh_server_capabilities()  # within TTL, should be a no-op
        assert calls == 1


class TestGenerationMetadataTypeGate:
    """The aesthetic-score gate is fail-closed and follows the server's metadata-type enum."""

    def test_fail_closed_before_any_probe(self) -> None:
        """With no successful probe yet, the metadata type reads as unsupported."""
        assert server_supports_generation_metadata_type("aesthetic_score") is False

    @pytest.mark.asyncio
    async def test_supported_when_server_lists_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Once the server advertises the type, the gate opens; unknown types stay closed."""

        async def _fake_fetch(url: str) -> dict[str, object]:
            return _swagger_2_doc(["caption"], ["lora", "censorship", "aesthetic_score"])

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _fake_fetch)
        await refresh_server_capabilities(force=True)

        assert server_supports_generation_metadata_type("aesthetic_score") is True
        assert server_supports_generation_metadata_type("not_a_type") is False

    @pytest.mark.asyncio
    async def test_unsupported_when_type_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A probe whose metadata enum lacks the type reports it unsupported (pre-go-live)."""

        async def _fake_fetch(url: str) -> dict[str, object]:
            return _swagger_2_doc(["caption"], ["lora", "censorship"])

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _fake_fetch)
        await refresh_server_capabilities(force=True)

        assert server_supports_generation_metadata_type("aesthetic_score") is False

    @pytest.mark.asyncio
    async def test_missing_metadata_schema_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the metadata schema is absent entirely, the whole probe fails-closed (both gates)."""

        async def _fake_fetch(url: str) -> dict[str, object]:
            return {"definitions": {_FORM_SCHEMA_NAME: {"properties": {"name": {"enum": ["caption"]}}}}}

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _fake_fetch)
        await refresh_server_capabilities(force=True)

        assert server_supports_generation_metadata_type("aesthetic_score") is False
        assert server_supports_interrogation_form("caption") is False
