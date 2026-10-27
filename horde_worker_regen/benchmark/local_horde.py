"""Serve finite benchmark workloads through the worker's ordinary Horde HTTP protocol."""

from __future__ import annotations

import base64
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Self

from aiohttp import web
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopRequest, ImageGenerateJobPopResponse
from horde_sdk.ai_horde_api.apimodels.alchemy.pop import (
    AlchemyJobPopResponse,
    AlchemyPopFormPayload,
    AlchemyPopRequest,
)
from horde_sdk.ai_horde_api.apimodels.alchemy.submit import AlchemyJobSubmitRequest
from horde_sdk.ai_horde_api.apimodels.generate.submit import ImageGenerationJobSubmitRequest
from horde_sdk.generation_parameters.alchemy.consts import (
    KNOWN_ALCHEMY_FORMS,
    KNOWN_CLIP_BLIP_TYPES,
    KNOWN_FACEFIXERS,
    KNOWN_UPSCALERS,
)
from pydantic import ValidationError

from horde_worker_regen.process_management.ipc.messages import AlchemyFormSpec
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    SupervisorControlMessage,
    WorkerLivenessFrame,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.simulation._canned_scenarios import make_empty_pop_response


@dataclass
class IssuedWork:
    """Represents one dispatched job and its generation-level terminal receipts."""

    job_id: str
    generation_ids: set[str]
    submitted: dict[str, str] = field(default_factory=dict)
    uploaded: set[str] = field(default_factory=set)
    issued: bool = False
    requires_upload: bool = True

    @property
    def finalized(self) -> bool:
        """Whether every generation has a terminal submit receipt."""
        return self.generation_ids == self.submitted.keys()

    @property
    def faulted(self) -> bool:
        """Whether any generation was reported faulted."""
        return any(state == "faulted" for state in self.submitted.values())


class LocalHorde:
    """Dispatch queued work and acknowledge uploads/submits without resetting worker state.

    Only the controller installs work. Generation ids remain registered for the entire run,
    so late results and retried submissions cannot be attributed to a subsequent check.
    """

    def __init__(self) -> None:
        """Create an idle service bound to loopback when entered."""
        self.token = secrets.token_urlsafe(32)
        self.url = ""
        self.snapshot: WorkerStateSnapshot | None = None
        self.liveness: WorkerLivenessFrame | None = None
        self.last_contact: float | None = None
        self.phase = "Starting the worker"
        self.work: dict[str, IssuedWork] = {}
        self._generation_jobs: dict[str, str] = {}
        self._images: deque[ImageGenerateJobPopResponse] = deque()
        self._forms: deque[AlchemyFormSpec] = deque()
        self._sources: dict[str, bytes] = {}
        self._commands: deque[SupervisorControlMessage] = deque()
        self._runner: web.AppRunner | None = None
        self.api_requests: set[str] = set()
        self.offered_forms: set[str] | None = None
        self.forms_offer_count: int = 0

    async def __aenter__(self) -> Self:
        """Start the service on an ephemeral loopback port."""
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/api/v2/generate/pop", self._image_pop)
        app.router.add_post("/api/v2/generate/submit", self._image_submit)
        app.router.add_post("/api/v2/interrogate/pop", self._alchemy_pop)
        app.router.add_post("/api/v2/interrogate/submit", self._alchemy_submit)
        app.router.add_put("/upload/{generation_id}", self._upload)
        app.router.add_get("/source/{source_id}", self._source)
        app.router.add_post("/worker-state", self._worker_state)
        app.router.add_post("/worker-liveness", self._worker_liveness)
        app.router.add_get("/worker-commands", self._worker_commands)
        app.router.add_get("/api/swagger.json", self._swagger)
        app.router.add_route("*", "/api/v2/{path:.*}", self._ancillary)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        address = self._runner.addresses[0]
        self.url = f"http://127.0.0.1:{address[1]}"
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close HTTP connections after the worker has stopped."""
        if self._runner is not None:
            await self._runner.cleanup()

    def enqueue(self, jobs: list[ImageGenerateJobPopResponse], forms: list[AlchemyFormSpec]) -> list[str]:
        """Install fresh work and return the ids whose terminal receipts must be awaited."""
        identifiers: list[str] = []
        for job in jobs:
            job_id = str(job.ids[0])
            ids = {str(identifier) for identifier in job.ids}
            self._register(job_id, ids)
            update: dict[str, object] = {
                "r2_uploads": [f"{self.url}/upload/{identifier}" for identifier in job.ids],
                "r2_upload": f"{self.url}/upload/{job.ids[0]}",
            }
            if job.source_image:
                self._sources[job_id] = base64.b64decode(job.source_image)
                update["source_image"] = f"{self.url}/source/{job_id}"
            self._images.append(job.model_copy(update=update))
            identifiers.append(job_id)
        for form in forms:
            self._register(
                form.form_id,
                {form.form_id},
                requires_upload=form.form not in {item.value for item in KNOWN_CLIP_BLIP_TYPES},
            )
            self._sources[form.form_id] = form.source_image_bytes
            self._forms.append(form)
            identifiers.append(form.form_id)
        return identifiers

    def _register(self, job_id: str, ids: set[str], *, requires_upload: bool = True) -> None:
        if job_id in self.work or any(identifier in self._generation_jobs for identifier in ids):
            raise ValueError("benchmark work must have fresh generation ids")
        self.work[job_id] = IssuedWork(job_id, ids, requires_upload=requires_upload)
        self._generation_jobs.update(dict.fromkeys(ids, job_id))

    def command(self, message: SupervisorControlMessage) -> None:
        """Queue a normal supervisor command for the worker's pipe relay."""
        self._commands.append(message)

    def withdraw_unissued(self) -> None:
        """Stop offering queued work without discarding receipts for previously issued work."""
        self._images.clear()
        self._forms.clear()

    def _authorize(self, request: web.Request) -> None:
        if not secrets.compare_digest(request.headers.get("Authorization", ""), f"Bearer {self.token}"):
            raise web.HTTPUnauthorized()

    async def _worker_state(self, request: web.Request) -> web.Response:
        self._authorize(request)
        self.snapshot = WorkerStateSnapshot.model_validate(await request.json())
        self.last_contact = time.monotonic()
        return web.json_response({"ok": True})

    async def _worker_liveness(self, request: web.Request) -> web.Response:
        self._authorize(request)
        self.liveness = WorkerLivenessFrame.model_validate(await request.json())
        self.last_contact = time.monotonic()
        return web.json_response({"ok": True})

    async def _worker_commands(self, request: web.Request) -> web.Response:
        self._authorize(request)
        commands = [item.model_dump(mode="json") for item in self._commands]
        self._commands.clear()
        return web.json_response(commands)

    async def _image_pop(self, request: web.Request) -> web.Response:
        try:
            offered = ImageGenerateJobPopRequest.model_validate(await request.json())
        except ValidationError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        for job in self._images:
            if self._image_eligible(job, offered):
                self._images.remove(job)
                self.work[str(job.ids[0])].issued = True
                return web.json_response(job.model_dump(mode="json", by_alias=True))
        return web.json_response(make_empty_pop_response().model_dump(mode="json", by_alias=True))

    @staticmethod
    def _image_eligible(job: ImageGenerateJobPopResponse, offered: ImageGenerateJobPopRequest) -> bool:
        payload = job.payload
        return (
            job.model in offered.models
            and payload.width * payload.height <= offered.max_pixels
            and (not payload.post_processing or offered.allow_post_processing)
            and (not payload.control_type or offered.allow_controlnet)
            and (payload.workflow != "qr_code" or offered.allow_controlnet or offered.allow_sdxl_controlnet)
            and (not payload.loras or offered.allow_lora)
        )

    async def _alchemy_pop(self, request: web.Request) -> web.Response:
        offered = AlchemyPopRequest.model_validate(await request.json())
        self.offered_forms = set(offered.forms)
        self.forms_offer_count += 1
        forms: list[AlchemyPopFormPayload] = []
        for form in list(self._forms):
            if form.form not in offered.forms:
                continue
            self._forms.remove(form)
            self.work[form.form_id].issued = True
            forms.append(
                AlchemyPopFormPayload(
                    id=form.form_id,
                    form=form.form,
                    source_image=f"{self.url}/source/{form.form_id}",
                    r2_upload=f"{self.url}/upload/{form.form_id}",
                )
            )
            if len(forms) >= offered.amount:
                break
        return web.json_response(AlchemyJobPopResponse(forms=forms).model_dump(mode="json", by_alias=True))

    async def _swagger(self, request: web.Request) -> web.Response:
        # The ordinary worker probes these schema fragments before advertising server-gated features.
        # A local dispatch service must describe its supported forms instead of contacting the live Horde.
        forms = sorted(
            {str(item.value) for values in (KNOWN_ALCHEMY_FORMS, KNOWN_UPSCALERS, KNOWN_FACEFIXERS) for item in values}
        )
        return web.json_response(
            {
                "swagger": "2.0",
                "definitions": {
                    "ModelInterrogationFormStable": {"properties": {"name": {"enum": forms}}},
                    "GenerationMetadataStable": {"properties": {"type": {"enum": []}}},
                    "PopInputStable": {"properties": {"allow_extended_controlnet": {"type": "boolean"}}},
                },
            }
        )

    async def _image_submit(self, request: web.Request) -> web.Response:
        result = ImageGenerationJobSubmitRequest.model_validate(await request.json())
        return self._submit(str(result.id_), result.state.value)

    async def _alchemy_submit(self, request: web.Request) -> web.Response:
        result = AlchemyJobSubmitRequest.model_validate(await request.json())
        return self._submit(str(result.id_), result.state.value)

    def _submit(self, identifier: str, state: str) -> web.Response:
        job_id = self._generation_jobs.get(identifier)
        if job_id is None:
            raise web.HTTPNotFound(text="unknown generation")
        work = self.work[job_id]
        if not work.issued:
            raise web.HTTPConflict(text="generation has not been dispatched")
        if state != "faulted" and work.requires_upload and identifier not in work.uploaded:
            raise web.HTTPConflict(text="image upload must precede a successful submit")
        previous = work.submitted.get(identifier)
        if previous is not None and previous != state:
            raise web.HTTPConflict(text="generation already submitted with a different result")
        work.submitted[identifier] = state
        return web.json_response({"reward": 0.0})

    async def _upload(self, request: web.Request) -> web.Response:
        identifier = request.match_info["generation_id"]
        job_id = self._generation_jobs.get(identifier)
        if job_id is None:
            raise web.HTTPNotFound()
        if not self.work[job_id].issued:
            raise web.HTTPConflict(text="generation has not been dispatched")
        if not await request.read():
            raise web.HTTPBadRequest(text="empty image upload")
        self.work[job_id].uploaded.add(identifier)
        return web.Response(status=200)

    async def _source(self, request: web.Request) -> web.Response:
        content = self._sources.get(request.match_info["source_id"])
        if content is None:
            raise web.HTTPNotFound()
        return web.Response(body=content, content_type="image/png")

    async def _ancillary(self, request: web.Request) -> web.Response:
        path = request.match_info["path"]
        self.api_requests.add(path)
        if path == "find_user" or path.startswith("users/"):
            return web.json_response({"id": 1, "username": "benchmark#1", "kudos": 0, "trusted": True})
        if path.startswith("status/"):
            return web.json_response([] if path.endswith(("models", "workers")) else {})
        if path.startswith("workers"):
            if path == "workers":
                return web.json_response([])
            return web.json_response(
                {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "name": path.rsplit("/", 1)[-1],
                    "type": "image",
                    "owner": "benchmark#1",
                    "maintenance_mode": False,
                    "bridge_agent": "local-benchmark:1:offline",
                    "paused": False,
                }
            )
        raise web.HTTPNotFound(text=f"unsupported benchmark API request: {path}")
