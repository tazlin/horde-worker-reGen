"""Contract checks for ordinary worker traffic against the loopback Horde subset."""

from __future__ import annotations

import aiohttp
import pytest
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIAsyncClientSession
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopRequest, ImageGenerateJobPopResponse
from horde_sdk.ai_horde_api.apimodels.alchemy.pop import AlchemyJobPopResponse
from horde_sdk.ai_horde_api.apimodels.users import UserDetailsResponse
from horde_sdk.ai_horde_api.apimodels.workers.workers import (
    SingleWorkerDetailsResponse,
    SingleWorkerNameDetailsRequest,
)

from horde_worker_regen.benchmark.local_horde import LocalHorde
from horde_worker_regen.benchmark.scenarios import CannedAlchemyFormSpec, CannedImageJobSpec, Scenario


async def test_image_batch_requires_upload_and_every_generation_submit() -> None:
    """Count every batch generation once and reject success before upload."""
    scenario = Scenario(name="batch", image_jobs=[CannedImageJobSpec(n_iter=2)])
    async with LocalHorde() as horde, aiohttp.ClientSession() as client:
        identifiers = horde.enqueue(scenario.expand_image_jobs(), [])
        async with client.post(
            f"{horde.url}/api/v2/generate/pop",
            json={
                "name": "test",
                "models": ["Deliberate"],
                "max_pixels": 1024**2,
            },
        ) as response:
            assert response.status == 200
            job = ImageGenerateJobPopResponse.model_validate(await response.json())
        work = horde.work[identifiers[0]]
        assert len(work.generation_ids) == 2
        for index, identifier in enumerate(job.ids):
            submit = {"id": str(identifier), "state": "ok", "generation": "R2", "seed": 1}
            if index == 0:
                async with client.post(f"{horde.url}/api/v2/generate/submit", json=submit) as response:
                    assert response.status == 409
            async with client.put(f"{horde.url}/upload/{identifier}", data=b"encoded-image") as response:
                assert response.status == 200
            async with client.post(f"{horde.url}/api/v2/generate/submit", json=submit) as response:
                assert response.status == 200
            assert work.finalized is (index == 1)
            async with client.post(f"{horde.url}/api/v2/generate/submit", json=submit) as response:
                assert response.status == 200
        assert len(work.submitted) == 2


async def test_pop_honors_model_and_feature_offer_and_sources_are_local() -> None:
    """Withheld features stay queued and accepted control images are fetched locally."""
    scenario = Scenario(name="control", image_jobs=[CannedImageJobSpec(control_type="canny")])
    async with LocalHorde() as horde, aiohttp.ClientSession() as client:
        horde.enqueue(scenario.expand_image_jobs(), [])
        request = {"name": "test", "models": ["Deliberate"], "max_pixels": 512**2, "allow_controlnet": False}
        async with client.post(f"{horde.url}/api/v2/generate/pop", json=request) as response:
            assert not ImageGenerateJobPopResponse.model_validate(await response.json()).ids
        request["allow_controlnet"] = True
        async with client.post(f"{horde.url}/api/v2/generate/pop", json=request) as response:
            job = ImageGenerateJobPopResponse.model_validate(await response.json())
        assert job.source_image is not None and job.source_image.startswith(horde.url)
        async with client.get(job.source_image) as response:
            assert response.status == 200
            assert (await response.read()).startswith(b"\x89PNG")


async def test_alchemy_forms_and_late_results_keep_their_original_identity() -> None:
    """A delayed form result cannot finalize a newly queued form."""
    scenario = Scenario(name="caption", alchemy_forms=[CannedAlchemyFormSpec(form="caption")])
    async with LocalHorde() as horde, aiohttp.ClientSession() as client:
        ids = horde.enqueue([], scenario.expand_alchemy_forms())
        async with client.post(
            f"{horde.url}/api/v2/interrogate/pop",
            json={
                "name": "test",
                "forms": ["caption"],
                "amount": 1,
                "priority_usernames": [],
            },
        ) as response:
            assert response.status == 200
            popped = AlchemyJobPopResponse.model_validate(await response.json())
        assert popped.forms and str(popped.forms[0].id_) == ids[0]
        later = horde.enqueue([], scenario.expand_alchemy_forms())
        async with client.post(
            f"{horde.url}/api/v2/interrogate/submit",
            json={
                "id": ids[0],
                "state": "ok",
                "result": {"caption": "example"},
            },
        ) as response:
            assert response.status == 200
        assert horde.work[ids[0]].finalized
        assert not horde.work[later[0]].finalized


async def test_control_requires_token_and_user_response_matches_sdk() -> None:
    """Only the controller can access commands while ordinary SDK user requests succeed."""
    async with LocalHorde() as horde, aiohttp.ClientSession() as client:
        async with client.get(f"{horde.url}/worker-commands") as response:
            assert response.status == 401
        async with client.get(f"{horde.url}/api/v2/find_user") as response:
            user = UserDetailsResponse.model_validate(await response.json())
        assert user.username == "benchmark#1"


async def test_ordinary_sdk_worker_lookup_and_pop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the actual SDK transport and response parsing required by normal startup and intake."""
    async with LocalHorde() as horde, aiohttp.ClientSession() as client:
        monkeypatch.setattr("horde_sdk.ai_horde_api.apimodels.base.AI_HORDE_BASE_URL", f"{horde.url}/api/")
        ids = horde.enqueue(Scenario(name="sdk", image_jobs=[CannedImageJobSpec()]).expand_image_jobs(), [])
        async with AIHordeAPIAsyncClientSession(client) as sdk:
            worker = await sdk.submit_request(
                SingleWorkerNameDetailsRequest(worker_name="local-test", apikey="b" * 22),
                SingleWorkerDetailsResponse,
            )
            assert isinstance(worker, SingleWorkerDetailsResponse)
            assert worker.owner == "benchmark#1" and not worker.maintenance_mode
            job = await sdk.submit_request(
                ImageGenerateJobPopRequest(name="local-test", models=["Deliberate"], max_pixels=512**2),
                ImageGenerateJobPopResponse,
            )
            assert isinstance(job, ImageGenerateJobPopResponse)
            assert str(job.ids[0]) == ids[0]


async def test_normal_server_capability_probe_and_sdk_alchemy_pop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local server satisfies the ordinary capability probe before serving a selected caption form."""
    from horde_sdk.ai_horde_api.apimodels.alchemy.pop import AlchemyJobPopResponse, AlchemyPopRequest

    from horde_worker_regen import server_capabilities

    for name in (
        "_supported_interrogation_forms",
        "_supported_generation_metadata_types",
        "_supports_extended_controlnet",
    ):
        monkeypatch.setattr(server_capabilities, name, None)
    monkeypatch.setattr(server_capabilities, "_next_refresh_monotonic", 0)
    async with LocalHorde() as horde, aiohttp.ClientSession() as client:
        monkeypatch.setattr(server_capabilities, "get_ai_horde_swagger_url", lambda: f"{horde.url}/api/swagger.json")
        await server_capabilities.refresh_server_capabilities(force=True)
        assert server_capabilities.server_supports_interrogation_form("caption")
        monkeypatch.setattr("horde_sdk.ai_horde_api.apimodels.base.AI_HORDE_BASE_URL", f"{horde.url}/api/")
        scenario = Scenario(name="caption", alchemy_forms=[CannedAlchemyFormSpec(form="caption")])
        ids = horde.enqueue([], scenario.expand_alchemy_forms())
        async with AIHordeAPIAsyncClientSession(client) as sdk:
            response = await sdk.submit_request(
                AlchemyPopRequest(name="test", forms=["caption"], priority_usernames=[], amount=1),
                AlchemyJobPopResponse,
            )
            assert isinstance(response, AlchemyJobPopResponse)
            assert str(response.forms[0].id_) == ids[0]
