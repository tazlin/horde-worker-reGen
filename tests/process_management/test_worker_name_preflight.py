"""Tests for the startup worker-name fail-fast checks (``worker_identity``)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management.config import worker_identity
from horde_worker_regen.process_management.config.worker_identity import WorkerNameConfigError, verify_worker_identity

_DREAMER_DEFAULT = reGenBridgeData.model_fields["dreamer_worker_name"].default
_ALCHEMIST_DEFAULT = reGenBridgeData.model_fields["alchemist_name"].default


def _bridge_data(
    *,
    dreamer: str | None = "Unique Dreamer",
    alchemist: bool = False,
    alchemist_name: str | None = None,
    dry_run_skip_api: bool = False,
) -> reGenBridgeData:
    """Build a real bridge config, overriding only the name-related fields under test.

    ``dreamer=None`` leaves the reserved default in place; ``alchemist_name=None`` leaves whatever
    the model default is.
    """
    bridge_data = reGenBridgeData(api_key="0000000000")
    if dreamer is not None:
        bridge_data.dreamer_worker_name = dreamer
    if alchemist_name is not None:
        bridge_data.alchemist_name = alchemist_name
    bridge_data.alchemist = alchemist
    bridge_data.dry_run_skip_api = dry_run_skip_api
    return bridge_data


class TestLocalNameValidation:
    """Default/duplicate names fail fast without any network access."""

    def test_default_dreamer_name_fails(self) -> None:
        """The reserved default dreamer name is rejected (image generation is always enabled)."""
        with pytest.raises(WorkerNameConfigError):
            verify_worker_identity(_bridge_data(dreamer=None, dry_run_skip_api=True))

    def test_alchemist_enabled_with_default_alchemist_name_fails(self) -> None:
        """With alchemy enabled, the reserved default alchemist name is rejected."""
        with pytest.raises(WorkerNameConfigError):
            verify_worker_identity(
                _bridge_data(alchemist=True, alchemist_name=_ALCHEMIST_DEFAULT, dry_run_skip_api=True),
            )

    def test_alchemist_name_equal_to_dreamer_fails(self) -> None:
        """The dreamer and alchemist names must differ when alchemy is enabled."""
        with pytest.raises(WorkerNameConfigError):
            verify_worker_identity(
                _bridge_data(
                    dreamer="Same Name",
                    alchemist=True,
                    alchemist_name="Same Name",
                    dry_run_skip_api=True,
                ),
            )

    def test_unique_names_pass(self) -> None:
        """Unique dreamer and alchemist names pass (dry_run_skip_api short-circuits the network check)."""
        verify_worker_identity(
            _bridge_data(
                dreamer="Unique Dreamer",
                alchemist=True,
                alchemist_name="Unique Alchemist",
                dry_run_skip_api=True,
            ),
        )

    def test_alchemist_disabled_ignores_default_alchemist_name(self) -> None:
        """With alchemy disabled, the default alchemist name is not enforced."""
        verify_worker_identity(
            _bridge_data(dreamer="Unique Dreamer", alchemist_name=_ALCHEMIST_DEFAULT, dry_run_skip_api=True),
        )


class TestOwnershipCheck:
    """The network check passes only when a name is unregistered or owned by this API key."""

    def test_owned_worker_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A registered worker whose id is owned by this API key passes."""
        worker = Mock(id_="worker-123", name="Unique Dreamer", owner="Me#1")
        monkeypatch.setattr(worker_identity, "_fetch_account_identity", lambda api_key: ({"worker-123"}, "Me#1"))
        monkeypatch.setattr(worker_identity, "_lookup_registered_worker", lambda name, api_key: worker)

        verify_worker_identity(_bridge_data(dreamer="Unique Dreamer"))

    def test_owned_by_owner_name_passes_when_id_missing_from_worker_ids(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A worker owned by this account must pass even when worker_ids does not list its id.

        Regression for the false "another account" rejection: the user-details worker_ids list lagged
        behind a freshly-registered/idle worker, so an owned worker (owner == authenticated username)
        was rejected and the worker refused to start. The owner-name match is the robust fallback.
        """
        worker = Mock(id_="alch-9", name="My Alchemist", owner="Tazlin#6572")
        monkeypatch.setattr(
            worker_identity,
            "_fetch_account_identity",
            lambda api_key: ({"some-other-id"}, "Tazlin#6572"),
        )
        monkeypatch.setattr(worker_identity, "_lookup_registered_worker", lambda name, api_key: worker)

        verify_worker_identity(_bridge_data(dreamer="My Alchemist"))

    def test_foreign_idle_worker_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A registered worker owned by a different account hard-fails (id absent and owner differs).

        The by-name lookup finds the worker even when it is idle/offline, so a collision with another
        account's worker is caught at preflight rather than slipping through (the all-workers list would
        have omitted an inactive worker and waved the name through as "free").
        """
        worker = Mock(id_="someone-else", name="Unique Dreamer", owner="Someone#999")
        monkeypatch.setattr(worker_identity, "_fetch_account_identity", lambda api_key: ({"mine-1"}, "Me#1"))
        monkeypatch.setattr(worker_identity, "_lookup_registered_worker", lambda name, api_key: worker)

        with pytest.raises(WorkerNameConfigError, match="another account"):
            verify_worker_identity(_bridge_data(dreamer="Unique Dreamer"))

    def test_unregistered_worker_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A name not yet registered is the normal first-run case and passes."""
        monkeypatch.setattr(worker_identity, "_fetch_account_identity", lambda api_key: (set(), "Me#1"))
        monkeypatch.setattr(worker_identity, "_lookup_registered_worker", lambda name, api_key: None)

        verify_worker_identity(_bridge_data(dreamer="Brand New Worker"))

    def test_non_not_found_lookup_error_hard_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A lookup error that is not WorkerNotFound must not be read as "name is free"; it hard-fails."""

        def _boom(name: str, api_key: str) -> object:
            raise RuntimeError("worker-details-by-name returned an error")

        monkeypatch.setattr(worker_identity, "_fetch_account_identity", lambda api_key: (set(), "Me#1"))
        monkeypatch.setattr(worker_identity, "_lookup_registered_worker", _boom)
        monkeypatch.setattr(worker_identity, "_OWNERSHIP_CHECK_RETRY_DELAY_SECONDS", 0.0)

        with pytest.raises(WorkerNameConfigError, match="Could not verify"):
            verify_worker_identity(_bridge_data(dreamer="Unique Dreamer"))

    def test_network_failure_hard_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unreachable API hard-fails the check after exhausting retries."""

        def _boom(api_key: str) -> tuple[set[str], str | None]:
            raise RuntimeError("network down")

        monkeypatch.setattr(worker_identity, "_fetch_account_identity", _boom)
        monkeypatch.setattr(worker_identity, "_OWNERSHIP_CHECK_RETRY_DELAY_SECONDS", 0.0)

        with pytest.raises(WorkerNameConfigError, match="Could not verify"):
            verify_worker_identity(_bridge_data(dreamer="Unique Dreamer"))

    def test_dry_run_skips_network(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With dry_run_skip_api set, the network ownership check is skipped entirely."""
        calls = {"count": 0}

        def _boom(api_key: str) -> tuple[set[str], str | None]:
            calls["count"] += 1
            raise RuntimeError("should not be called when API is skipped")

        monkeypatch.setattr(worker_identity, "_fetch_account_identity", _boom)

        verify_worker_identity(_bridge_data(dreamer="Unique Dreamer", dry_run_skip_api=True))
        assert calls["count"] == 0


class _FakeSession:
    """A stand-in for ``AIHordeAPIClientSession`` that returns a canned response from submit_request."""

    def __init__(self, response: object) -> None:
        self._response = response

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def submit_request(self, request: object, response_type: object) -> object:
        return self._response


class TestLookupRegisteredWorker:
    """The by-name lookup interprets the error response by meaning, not by swallowing it."""

    def test_worker_not_found_maps_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A WorkerNotFound error response is the "name is free" signal and yields None."""
        from horde_sdk.ai_horde_api.consts import RC
        from horde_sdk.generic_api.apimodels import RequestErrorResponse

        error = RequestErrorResponse(message="Worker not found", rc=RC.WorkerNotFound)
        monkeypatch.setattr(worker_identity, "AIHordeAPIClientSession", lambda: _FakeSession(error))

        assert worker_identity._lookup_registered_worker("Idle Worker", "0" * 22) is None

    def test_other_error_response_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Any non-WorkerNotFound error response is raised, never read as an absent worker."""
        from horde_sdk.ai_horde_api.consts import RC
        from horde_sdk.generic_api.apimodels import RequestErrorResponse

        error = RequestErrorResponse(message="nope", rc=RC.UserNotFound)
        monkeypatch.setattr(worker_identity, "AIHordeAPIClientSession", lambda: _FakeSession(error))

        with pytest.raises(RuntimeError, match="returned an error"):
            worker_identity._lookup_registered_worker("Some Worker", "0" * 22)

    def test_found_worker_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A successful response (the worker exists, active or idle) is returned for ownership checks."""
        worker = Mock(id_="w-1", name="Some Worker", owner="Me#1")
        monkeypatch.setattr(worker_identity, "AIHordeAPIClientSession", lambda: _FakeSession(worker))

        assert worker_identity._lookup_registered_worker("Some Worker", "0" * 22) is worker
