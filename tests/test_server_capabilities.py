"""Tests for the server-capability probe that gates forms on what the server actually supports."""

from __future__ import annotations

import pytest

from horde_worker_regen import server_capabilities
from horde_worker_regen.server_capabilities import (
    _parse_interrogation_form_enum,
    refresh_server_capabilities,
    reset_server_capabilities_cache,
    server_supports_interrogation_form,
)


def _swagger_2_doc(forms: list[str]) -> dict[str, object]:
    """A minimal Swagger 2.0 document carrying the interrogation form-name enum."""
    return {"definitions": {"ModelInterrogationFormStable": {"properties": {"name": {"enum": forms}}}}}


def _openapi_3_doc(forms: list[str]) -> dict[str, object]:
    """The OpenAPI 3 (`components.schemas`) shape, the fallback if the server ever migrates."""
    return {"components": {"schemas": {"ModelInterrogationFormStable": {"properties": {"name": {"enum": forms}}}}}}


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    reset_server_capabilities_cache()


class TestParseEnum:
    """The form-name enum is read from either swagger shape, and absence is an error."""

    def test_parses_swagger_2(self) -> None:
        """The Swagger 2.0 `definitions` shape yields the enum."""
        assert _parse_interrogation_form_enum(_swagger_2_doc(["caption", "vectorize"])) == frozenset(
            {"caption", "vectorize"},
        )

    def test_parses_openapi_3_fallback(self) -> None:
        """The OpenAPI 3 `components.schemas` shape yields the enum."""
        assert _parse_interrogation_form_enum(_openapi_3_doc(["nsfw"])) == frozenset({"nsfw"})

    def test_raises_on_missing_path(self) -> None:
        """A document missing the expected schema raises (so the caller can fail-closed)."""
        with pytest.raises((KeyError, TypeError)):
            _parse_interrogation_form_enum({"definitions": {}})


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
