"""Tests for the server-capability probe that gates features on what the server actually supports."""

from __future__ import annotations

import pytest

from horde_worker_regen import server_capabilities
from horde_worker_regen.server_capabilities import (
    _EXTENDED_CONTROLNET_PROPERTY,
    _FORM_ENUM_KEYS,
    _FORM_SCHEMA_NAME,
    _METADATA_TYPE_ENUM_KEYS,
    _METADATA_TYPE_SCHEMA_NAME,
    _POP_INPUT_SCHEMA_NAME,
    _parse_enum,
    _parse_property_present,
    refresh_server_capabilities,
    reset_server_capabilities_cache,
    server_supports_extended_controlnet,
    server_supports_generation_metadata_type,
    server_supports_interrogation_form,
)


_PRODUCTION_SWAGGER_URL = "https://aihorde.net/api/swagger.json"


def _pop_input_schema(extended_controlnet: bool) -> dict[str, object]:
    """The `PopInputStable` schema, with the extended-controlnet property present only on new servers."""
    properties: dict[str, object] = {"allow_controlnet": {"type": "boolean"}}
    if extended_controlnet:
        properties[_EXTENDED_CONTROLNET_PROPERTY] = {"type": "boolean"}
    return {"properties": properties}


def _swagger_2_doc(
    forms: list[str],
    metadata_types: list[str] | None = None,
    *,
    extended_controlnet: bool = True,
) -> dict[str, object]:
    """A minimal Swagger 2.0 document carrying every probed schema."""
    if metadata_types is None:
        metadata_types = ["lora", "censorship"]
    return {
        "definitions": {
            _FORM_SCHEMA_NAME: {"properties": {"name": {"enum": forms}}},
            _METADATA_TYPE_SCHEMA_NAME: {"properties": {"type": {"enum": metadata_types}}},
            _POP_INPUT_SCHEMA_NAME: _pop_input_schema(extended_controlnet),
        },
    }


_cached_production_swagger_doc: dict[str, object] | None = None


def _production_swagger_doc() -> dict[str, object]:
    """The production swagger.json shape, for a live probe."""
    global _cached_production_swagger_doc
    if _cached_production_swagger_doc is not None:
        return _cached_production_swagger_doc

    import requests

    response = requests.get(_PRODUCTION_SWAGGER_URL)
    response.raise_for_status()
    _cached_production_swagger_doc = response.json()

    return _cached_production_swagger_doc


def _openapi_3_doc(
    forms: list[str],
    metadata_types: list[str] | None = None,
    *,
    extended_controlnet: bool = True,
) -> dict[str, object]:
    """The OpenAPI 3 (`components.schemas`) shape, the fallback if the server ever migrates."""
    if metadata_types is None:
        metadata_types = ["lora", "censorship"]
    return {
        "components": {
            "schemas": {
                _FORM_SCHEMA_NAME: {"properties": {"name": {"enum": forms}}},
                _METADATA_TYPE_SCHEMA_NAME: {"properties": {"type": {"enum": metadata_types}}},
                _POP_INPUT_SCHEMA_NAME: _pop_input_schema(extended_controlnet),
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

    @pytest.mark.network
    def test_production_swagger_doc_parses(self) -> None:
        """The production swagger.json shape is parseable and yields a non-empty form enum."""
        doc = _production_swagger_doc()
        schemas = _schemas(doc)
        forms = _parse_enum(schemas, _FORM_SCHEMA_NAME, _FORM_ENUM_KEYS)
        assert isinstance(forms, frozenset)
        assert len(forms) > 0

    @pytest.mark.network
    def test_production_swagger_doc_has_metadata_type_enum(self) -> None:
        """The production swagger.json shape is parseable and yields a non-empty generation-metadata type enum."""
        doc = _production_swagger_doc()
        schemas = _schemas(doc)
        metadata_types = _parse_enum(schemas, _METADATA_TYPE_SCHEMA_NAME, _METADATA_TYPE_ENUM_KEYS)
        assert isinstance(metadata_types, frozenset)
        assert len(metadata_types) > 0

    @pytest.mark.network
    def test_production_swagger_doc_has_pop_input_schema(self) -> None:
        """The production swagger.json shape is parseable and yields a `PopInputStable` schema with properties.

        ``PopInputStable`` uses ``allOf`` composition; we exercise ``_parse_property_present`` here
        rather than manually unwinding the schema.
        """
        doc = _production_swagger_doc()
        schemas = _schemas(doc)
        assert _parse_property_present(schemas, _POP_INPUT_SCHEMA_NAME, "allow_controlnet") is True


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
    async def test_missing_metadata_schema_keeps_other_gates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing metadata schema fails-closed for that gate only; the forms gate still succeeds."""

        async def _fake_fetch(url: str) -> dict[str, object]:
            return {"definitions": {_FORM_SCHEMA_NAME: {"properties": {"name": {"enum": ["caption"]}}}}}

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _fake_fetch)
        await refresh_server_capabilities(force=True)

        assert server_supports_generation_metadata_type("aesthetic_score") is False
        assert server_supports_interrogation_form("caption") is True


class TestParsePropertyPresent:
    """The property-presence helper reports declared properties and raises on a malformed schema."""

    def test_present_property(self) -> None:
        """A declared property reads as present."""
        schemas = _schemas(_swagger_2_doc(["caption"], extended_controlnet=True))
        assert _parse_property_present(schemas, _POP_INPUT_SCHEMA_NAME, _EXTENDED_CONTROLNET_PROPERTY) is True

    def test_absent_property_is_false_not_error(self) -> None:
        """An undeclared property is a definitive False (an older server), not a raise."""
        schemas = _schemas(_swagger_2_doc(["caption"], extended_controlnet=False))
        assert _parse_property_present(schemas, _POP_INPUT_SCHEMA_NAME, _EXTENDED_CONTROLNET_PROPERTY) is False

    def test_missing_schema_raises(self) -> None:
        """A schema table missing the named schema raises (so the caller can fail-closed)."""
        with pytest.raises((KeyError, TypeError)):
            _parse_property_present({}, _POP_INPUT_SCHEMA_NAME, _EXTENDED_CONTROLNET_PROPERTY)

    def test_malformed_properties_raises(self) -> None:
        """A schema whose `properties` is not an object raises (unexpected shape, treated as unknown)."""
        with pytest.raises(TypeError):
            _parse_property_present(
                {_POP_INPUT_SCHEMA_NAME: {"properties": ["not", "an", "object"]}},
                _POP_INPUT_SCHEMA_NAME,
                _EXTENDED_CONTROLNET_PROPERTY,
            )

    def test_all_of_present_property(self) -> None:
        """A property declared inside an ``allOf`` branch (possibly via ``$ref``) reads as present."""
        schemas = {
            _POP_INPUT_SCHEMA_NAME: {
                "allOf": [
                    {"$ref": "#/definitions/PopInput"},
                    {"properties": {"allow_controlnet": {"type": "boolean"}}},
                ],
            },
            "PopInput": {"properties": {"allow_img2img": {"type": "boolean"}}},
        }
        assert _parse_property_present(schemas, _POP_INPUT_SCHEMA_NAME, "allow_controlnet") is True
        assert _parse_property_present(schemas, _POP_INPUT_SCHEMA_NAME, "allow_img2img") is True

    def test_all_of_absent_property(self) -> None:
        """A property absent from all ``allOf`` branches returns False (not an error)."""
        schemas = {
            _POP_INPUT_SCHEMA_NAME: {
                "allOf": [
                    {"properties": {"allow_lora": {"type": "boolean"}}},
                ],
            },
        }
        assert _parse_property_present(schemas, _POP_INPUT_SCHEMA_NAME, _EXTENDED_CONTROLNET_PROPERTY) is False

    def test_all_of_unresolved_ref_raises(self) -> None:
        """An unresolvable ``$ref`` inside ``allOf`` raises (unknown schema shape, treated as fail-closed)."""
        schemas = {
            _POP_INPUT_SCHEMA_NAME: {
                "allOf": [
                    {"$ref": "#/definitions/NonexistentSchema"},
                ],
            },
        }
        with pytest.raises(KeyError):
            _parse_property_present(schemas, _POP_INPUT_SCHEMA_NAME, "allow_controlnet")

    def test_all_of_non_dict_entry_raises(self) -> None:
        """A non-dict ``allOf`` entry raises (unexpected shape)."""
        schemas = {
            _POP_INPUT_SCHEMA_NAME: {
                "allOf": ["not_an_object"],
            },
        }
        with pytest.raises(TypeError):
            _parse_property_present(schemas, _POP_INPUT_SCHEMA_NAME, "allow_controlnet")

    def test_neither_properties_nor_all_of_raises(self) -> None:
        """A schema with neither ``properties`` nor ``allOf`` raises (unknown shape)."""
        with pytest.raises(KeyError):
            _parse_property_present(
                {_POP_INPUT_SCHEMA_NAME: {"type": "object"}},
                _POP_INPUT_SCHEMA_NAME,
                _EXTENDED_CONTROLNET_PROPERTY,
            )


class TestExtendedControlNetGate:
    """The extended-controlnet gate is fail-closed and follows property presence on `PopInputStable`."""

    def test_fail_closed_before_any_probe(self) -> None:
        """With no successful probe yet, extended controlnet reads as unsupported."""
        assert server_supports_extended_controlnet() is False

    @pytest.mark.asyncio
    async def test_supported_when_property_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A probe whose `PopInputStable` declares the property opens the gate."""

        async def _fake_fetch(url: str) -> dict[str, object]:
            return _swagger_2_doc(["caption"], extended_controlnet=True)

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _fake_fetch)
        await refresh_server_capabilities(force=True)

        assert server_supports_extended_controlnet() is True

    @pytest.mark.asyncio
    async def test_unsupported_when_property_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A probe against an older server (no such property) keeps the gate closed."""

        async def _fake_fetch(url: str) -> dict[str, object]:
            return _swagger_2_doc(["caption"], extended_controlnet=False)

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _fake_fetch)
        await refresh_server_capabilities(force=True)

        assert server_supports_extended_controlnet() is False

    @pytest.mark.asyncio
    async def test_failure_keeps_prior_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A transient probe failure must not drop an already-proven capability."""

        async def _good_fetch(url: str) -> dict[str, object]:
            return _swagger_2_doc(["caption"], extended_controlnet=True)

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _good_fetch)
        await refresh_server_capabilities(force=True)
        assert server_supports_extended_controlnet() is True

        async def _failing_fetch(url: str) -> dict[str, object]:
            raise ConnectionError("boom")

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _failing_fetch)
        await refresh_server_capabilities(force=True)
        assert server_supports_extended_controlnet() is True

    @pytest.mark.asyncio
    async def test_missing_pop_input_schema_keeps_other_gates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing ``PopInputStable`` fails-closed for the extended-controlnet gate only."""

        async def _fake_fetch(url: str) -> dict[str, object]:
            return {
                "definitions": {
                    _FORM_SCHEMA_NAME: {"properties": {"name": {"enum": ["caption"]}}},
                    _METADATA_TYPE_SCHEMA_NAME: {"properties": {"type": {"enum": ["lora"]}}},
                },
            }

        monkeypatch.setattr(server_capabilities, "_fetch_swagger_spec", _fake_fetch)
        await refresh_server_capabilities(force=True)

        assert server_supports_extended_controlnet() is False
        assert server_supports_interrogation_form("caption") is True
