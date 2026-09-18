"""Tests for the alchemy job models, form routing, and wire-format workarounds."""

import time
import uuid
from collections import deque
from io import BytesIO
from typing import cast

import PIL.Image
import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels.alchemy.pop import AlchemyPopFormPayload
from horde_sdk.generation_parameters.alchemy.consts import KNOWN_ALCHEMY_FORMS, KNOWN_ALCHEMY_TYPES
from loguru import logger
from pydantic import ValidationError

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.ipc.messages import (
    AlchemyFormSpec,
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeControlFlag,
)
from horde_worker_regen.process_management.jobs import alchemy_popper
from horde_worker_regen.process_management.jobs.alchemy_popper import (
    AlchemyCoordinator,
    AlchemyHeadroomEstimator,
    _AlchemyPopRequest,
    _AlchemySubmitRequest,
    expand_offered_forms,
    lane_bound_post_processor_candidates,
    required_capability,
)
from horde_worker_regen.process_management.jobs.job_models import PendingAlchemySubmitJob
from horde_worker_regen.process_management.lifecycle.horde_process import WorkerCapability
from horde_worker_regen.process_management.models.model_availability import PostProcessorPresence
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger


class _RecordingQueue:
    """Minimal process-message queue for child-process unit tests."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        self.messages.append(message)


def _png_bytes() -> bytes:
    image = PIL.Image.new("RGB", (8, 8), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _result_message(
    form: str,
    *,
    result_payload: dict | None = None,
    image_bytes: bytes | None = None,
) -> HordeAlchemyResultMessage:
    return HordeAlchemyResultMessage(
        process_id=0,
        process_launch_identifier=0,
        info="test",
        form_id="00000000-0000-0000-0000-000000000000",
        form=form,
        state=GENERATION_STATE.ok,
        result_payload=result_payload,
        image_bytes=image_bytes,
    )


class TestRequiredCapability:
    """Forms route to the capability that can serve them."""

    def test_graph_forms_route_to_the_post_processing_lane(self) -> None:
        """Graph-backed forms (upscalers/facefixers) require ALCHEMY_GRAPH (the post-processing lane)."""
        for form in ("RealESRGAN_x4plus", "4x_AnimeSharp", "NMKD_Siax", "GFPGAN", "CodeFormers"):
            assert required_capability(form) == WorkerCapability.ALCHEMY_GRAPH, form

    def test_strip_background_routes_to_image_utilities(self) -> None:
        """strip_background runs on the out-of-venv image-utilities lane, not the post-processing lane."""
        assert required_capability("strip_background") == WorkerCapability.IMAGE_UTILITIES

    def test_annotation_routes_to_image_utilities(self) -> None:
        """Annotation runs on the isolated image-utilities lane."""
        assert required_capability("annotation") == WorkerCapability.IMAGE_UTILITIES

    def test_clip_forms_route_to_safety_process(self) -> None:
        """Text-output forms require ALCHEMY_CLIP (the safety process)."""
        for form in ("caption", "interrogation", "nsfw", "vectorize", "palette", "describe"):
            assert required_capability(form) == WorkerCapability.ALCHEMY_CLIP, form

    def test_the_umbrella_name_parses_and_routes_to_the_clip_lane(self) -> None:
        """A popped ``post_process`` would reach dispatch and route to ALCHEMY_CLIP, which cannot serve it.

        The pop payload's ``form`` is typed ``KNOWN_ALCHEMY_TYPES | str`` and the umbrella name is not in that
        enum, so nothing in parsing rejects it and the coordinator would queue it like any other form. The
        routing table has no entry for it and falls through to the CLIP lane. Only the offer keeps this off
        the worker: the expansion never emits the name, so the horde has nothing to hand back.
        """
        assert KNOWN_ALCHEMY_FORMS.post_process.value not in [member.value for member in KNOWN_ALCHEMY_TYPES]
        parsed = AlchemyPopFormPayload(
            id=str(uuid.uuid4()),
            form=KNOWN_ALCHEMY_FORMS.post_process.value,
            source_image="",
        )
        assert str(parsed.form) == KNOWN_ALCHEMY_FORMS.post_process.value
        assert required_capability(str(parsed.form)) == WorkerCapability.ALCHEMY_CLIP


class TestExpandOfferedForms:
    """Bridge-data forms expand to the individual form names the API expects."""

    def _bridge_data(self, **kwargs: object) -> reGenBridgeData:
        return reGenBridgeData(api_key="0000000000", **kwargs)  # pyrefly: ignore

    def test_default_forms_without_caption_opt_in(self) -> None:
        """The default forms offer everything except caption (BLIP opt-in)."""
        bridge_data = self._bridge_data()
        offered = expand_offered_forms(bridge_data)

        assert "caption" not in offered, "caption requires the explicit BLIP opt-in"
        assert "interrogation" in offered
        assert "nsfw" in offered
        assert "RealESRGAN_x4plus" in offered
        assert "4x_AnimeSharp" in offered
        assert "GFPGAN" in offered
        assert "CodeFormers" in offered
        assert "BACKEND_DEFAULT" not in offered

    def test_lane_bound_forms_withheld_while_the_post_processing_lane_is_down(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Upscalers and face-fixers are dropped while no post-processing lane is up; other forms stay."""
        monkeypatch.setattr(alchemy_popper, "strip_background_available", lambda: True)
        bridge_data = self._bridge_data()
        offered = expand_offered_forms(bridge_data, post_process_lane_healthy=False)

        assert "RealESRGAN_x4plus" not in offered
        assert "GFPGAN" not in offered
        assert "CodeFormers" not in offered
        assert "interrogation" in offered, "CLIP-stack forms do not touch the post-processing lane"
        assert "strip_background" in offered, "strip_background runs on the image-utilities lane"

    def test_clip_forms_withheld_while_no_clip_lane_is_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLIP-stack forms are dropped while the safety process is down; the other families are untouched."""
        monkeypatch.setattr(alchemy_popper, "strip_background_available", lambda: True)
        bridge_data = self._bridge_data(alchemy_caption_enabled=True)
        offered = expand_offered_forms(bridge_data, clip_lane_healthy=False)

        assert "interrogation" not in offered
        assert "nsfw" not in offered
        assert "caption" not in offered
        assert "RealESRGAN_x4plus" in offered, "upscalers run on the post-processing lane"
        assert "GFPGAN" in offered
        assert "strip_background" in offered, "background removal runs on the image-utilities lane"

    def test_clip_forms_return_when_the_clip_lane_comes_back(self) -> None:
        """The withholding is a live reading, not a latch: the forms are offered again once safety is up."""
        bridge_data = self._bridge_data(alchemy_caption_enabled=True)

        assert "interrogation" not in expand_offered_forms(bridge_data, clip_lane_healthy=False)
        offered = expand_offered_forms(bridge_data, clip_lane_healthy=True)
        assert "interrogation" in offered
        assert "nsfw" in offered
        assert "caption" in offered

    def test_lane_families_are_withheld_independently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each family follows only its own lane, so one lane going down never costs another's forms."""
        monkeypatch.setattr(alchemy_popper, "strip_background_available", lambda: True)
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)
        bridge_data = self._bridge_data()
        types = frozenset({"canny"})

        graph_only = expand_offered_forms(
            bridge_data,
            utilities_lane_healthy=False,
            clip_lane_healthy=False,
            annotation_types=types,
        )
        assert "RealESRGAN_x4plus" in graph_only
        assert "strip_background" not in graph_only
        assert "annotation" not in graph_only
        assert "nsfw" not in graph_only

        utilities_only = expand_offered_forms(
            bridge_data,
            post_process_lane_healthy=False,
            clip_lane_healthy=False,
            annotation_types=types,
        )
        assert "strip_background" in utilities_only
        assert "annotation" in utilities_only
        assert "RealESRGAN_x4plus" not in utilities_only
        assert "nsfw" not in utilities_only

    def test_the_post_process_umbrella_name_is_never_offered(self) -> None:
        """The umbrella config name expands away, so the worker never invites the unexpanded form.

        It routes to ALCHEMY_CLIP by falling off the end of the routing table rather than by belonging
        there, and the safety process cannot run a post-processor, so a popped ``post_process`` would fault.
        The worker's protection against that is simply never offering the name.
        """
        offered = expand_offered_forms(self._bridge_data(forms=["post-process"]))

        assert KNOWN_ALCHEMY_FORMS.post_process.value not in offered
        assert "post-process" not in offered
        assert "RealESRGAN_x4plus" in offered

    def test_caption_offered_with_opt_in(self) -> None:
        """Caption is offered once alchemy_caption_enabled is set."""
        bridge_data = self._bridge_data(alchemy_caption_enabled=True)
        assert "caption" in expand_offered_forms(bridge_data)

    def test_annotation_offered_with_a_servable_type(self, monkeypatch: object) -> None:
        """Annotation is offered only when the server and utilities lane can both serve it."""
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)  # type: ignore[attr-defined]
        bridge_data = self._bridge_data(forms=["annotation"])

        assert expand_offered_forms(bridge_data, annotation_types=frozenset({"canny"})) == ["annotation"]
        assert expand_offered_forms(bridge_data, annotation_types=frozenset()) == []

    def test_annotation_withheld_without_lane_or_server(self, monkeypatch: object) -> None:
        """Annotation is fail-closed: a down lane or an unaware server withholds it even with servable types.

        The form has no in-graph fallback, so advertising it while no lane can serve it (or while the server
        does not yet list it, which would make the server reject the entire pop) must not happen.
        """
        bridge_data = self._bridge_data(forms=["annotation"])
        types = frozenset({"canny"})

        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)  # type: ignore[attr-defined]
        assert expand_offered_forms(bridge_data, utilities_lane_healthy=False, annotation_types=types) == []

        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: False)  # type: ignore[attr-defined]
        assert expand_offered_forms(bridge_data, utilities_lane_healthy=True, annotation_types=types) == []

    def test_restricted_forms(self) -> None:
        """An explicit forms list restricts what is offered."""
        bridge_data = self._bridge_data(forms=["nsfw"])
        offered = expand_offered_forms(bridge_data)
        assert offered == ["nsfw"]

    def test_vectorize_offered_when_available_and_server_supports(self, monkeypatch: object) -> None:
        """Vectorize is offered when vtracer is installed and the server advertises the form."""
        monkeypatch.setattr(alchemy_popper, "vectorize_available", lambda: True)  # type: ignore[attr-defined]
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)  # type: ignore[attr-defined]
        bridge_data = self._bridge_data(forms=["vectorize"])
        assert expand_offered_forms(bridge_data) == ["vectorize"]

    def test_vectorize_dropped_when_unavailable(self, monkeypatch: object) -> None:
        """Vectorize is dropped on a lean install lacking vtracer, so we never advertise a faulting form."""
        monkeypatch.setattr(alchemy_popper, "vectorize_available", lambda: False)  # type: ignore[attr-defined]
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)  # type: ignore[attr-defined]
        bridge_data = self._bridge_data(forms=["vectorize"])
        assert expand_offered_forms(bridge_data) == []

    def test_vectorize_dropped_when_server_unsupported(self, monkeypatch: object) -> None:
        """Vectorize is withheld until the server advertises it, so we can ship ahead of go-live.

        Offering a form the server's enum rejects would make it reject the entire pop, so the gate is
        fail-closed: vtracer present but server support unknown/absent means the form is not offered.
        """
        monkeypatch.setattr(alchemy_popper, "vectorize_available", lambda: True)  # type: ignore[attr-defined]
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: False)  # type: ignore[attr-defined]
        bridge_data = self._bridge_data(forms=["vectorize"])
        assert expand_offered_forms(bridge_data) == []

    def test_palette_offered_when_server_supports(self, monkeypatch: object) -> None:
        """Palette (pure-Pillow, no optional dep) is offered once the server advertises the form."""
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)  # type: ignore[attr-defined]
        bridge_data = self._bridge_data(forms=["palette"])
        assert expand_offered_forms(bridge_data) == ["palette"]

    def test_palette_withheld_when_server_unsupported(self, monkeypatch: object) -> None:
        """Palette is fail-closed until the server advertises it, so the worker can ship ahead of go-live."""
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: False)  # type: ignore[attr-defined]
        bridge_data = self._bridge_data(forms=["palette"])
        assert expand_offered_forms(bridge_data) == []

    def test_describe_offered_when_available_and_server_supports(self, monkeypatch: object) -> None:
        """Describe is offered when its deps import and the server advertises the form."""
        monkeypatch.setattr(alchemy_popper, "describe_available", lambda: True)  # type: ignore[attr-defined]
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)  # type: ignore[attr-defined]
        bridge_data = self._bridge_data(forms=["describe"])
        assert expand_offered_forms(bridge_data) == ["describe"]

    def test_describe_dropped_when_unavailable(self, monkeypatch: object) -> None:
        """Describe is dropped on a lean install lacking its deps, so we never advertise a faulting form."""
        monkeypatch.setattr(alchemy_popper, "describe_available", lambda: False)  # type: ignore[attr-defined]
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)  # type: ignore[attr-defined]
        bridge_data = self._bridge_data(forms=["describe"])
        assert expand_offered_forms(bridge_data) == []

    def test_describe_withheld_when_server_unsupported(self, monkeypatch: object) -> None:
        """Describe is withheld until the server advertises it (fail-closed gate)."""
        monkeypatch.setattr(alchemy_popper, "describe_available", lambda: True)  # type: ignore[attr-defined]
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: False)  # type: ignore[attr-defined]
        bridge_data = self._bridge_data(forms=["describe"])
        assert expand_offered_forms(bridge_data) == []

    def test_aesthetic_offered_when_server_supports(self, monkeypatch: object) -> None:
        """Aesthetic (a CLIP-stack form, always runnable when the safety process is up) gates on the server."""
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)  # type: ignore[attr-defined]
        bridge_data = self._bridge_data(forms=["aesthetic"])
        assert expand_offered_forms(bridge_data) == ["aesthetic"]

    def test_aesthetic_withheld_when_server_unsupported(self, monkeypatch: object) -> None:
        """Aesthetic is fail-closed until the server advertises it, so the worker can ship ahead of go-live."""
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: False)  # type: ignore[attr-defined]
        bridge_data = self._bridge_data(forms=["aesthetic"])
        assert expand_offered_forms(bridge_data) == []

    def test_beta_upscalers_offered_when_server_supports(self, monkeypatch: object) -> None:
        """Beta upscalers join the post-process offer once the server advertises them."""
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)  # type: ignore[attr-defined]
        offered = expand_offered_forms(self._bridge_data(forms=["post-process"]))
        assert "4xNomos8kSC" in offered
        assert "2xModernSpanimationV1" in offered
        # The long-standing upscalers are always offered regardless of the gate.
        assert "RealESRGAN_x4plus" in offered

    def test_beta_upscalers_withheld_when_server_unsupported(self, monkeypatch: object) -> None:
        """Beta upscalers are withheld pre-go-live, but the long-standing upscalers stay on offer.

        Offering an upscaler the server's enum rejects would make it reject the whole pop, so the new
        names are fail-closed until the server advertises them; the established names are never gated.
        """
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: False)  # type: ignore[attr-defined]
        offered = expand_offered_forms(self._bridge_data(forms=["post-process"]))
        assert "4xNomos8kSC" not in offered
        assert "2xModernSpanimationV1" not in offered
        assert "RealESRGAN_x4plus" in offered
        assert "GFPGAN" in offered

    def test_beta_facefixers_offered_when_server_supports(self, monkeypatch: object) -> None:
        """Beta face restorers join the post-process offer once the server advertises them."""
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)  # type: ignore[attr-defined]
        offered = expand_offered_forms(self._bridge_data(forms=["post-process"]))
        assert "RestoreFormer" in offered
        assert "GFPGANv1.3" in offered
        # The long-standing face-fixers are always offered regardless of the gate.
        assert "GFPGAN" in offered
        assert "CodeFormers" in offered

    def test_beta_facefixers_withheld_when_server_unsupported(self, monkeypatch: object) -> None:
        """Beta face restorers are withheld pre-go-live; the established face-fixers stay on offer.

        An unknown post-processor makes the server reject the whole pop, so the new names are fail-closed
        until the server advertises them, while GFPGAN/CodeFormers (in every server's enum) are never gated.
        """
        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: False)  # type: ignore[attr-defined]
        offered = expand_offered_forms(self._bridge_data(forms=["post-process"]))
        assert "RestoreFormer" not in offered
        assert "GFPGANv1.3" not in offered
        assert "GFPGAN" in offered
        assert "CodeFormers" in offered

    def test_post_processor_missing_from_a_known_set_is_withheld(self) -> None:
        """A post-processor whose model is not on disk is dropped; its present peers and CLIP forms stay.

        The gate is per model rather than per feature, so one missing upscaler costs the worker that
        upscaler alone instead of the whole post-process expansion.
        """
        presence = PostProcessorPresence(lane_bound=frozenset({"RealESRGAN_x4plus", "CodeFormers"}))
        offered = expand_offered_forms(self._bridge_data(), presence=presence)

        assert "RealESRGAN_x4plus" in offered
        assert "CodeFormers" in offered
        assert "GFPGAN" not in offered
        assert "4x_AnimeSharp" not in offered
        assert "interrogation" in offered, "CLIP-stack forms need no post-processor weight"
        assert "nsfw" in offered

    def test_unknown_presence_withholds_nothing(self) -> None:
        """An unreported presence gates nothing, so a worker with no download process is unaffected."""
        offered = expand_offered_forms(self._bridge_data(), presence=PostProcessorPresence())

        for candidate in lane_bound_post_processor_candidates():
            assert candidate in offered, candidate

    def test_empty_known_set_withholds_every_upscaler_and_face_fixer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A report that nothing is on disk drops every upscaler and face fixer, and nothing else."""
        monkeypatch.setattr(alchemy_popper, "strip_background_available", lambda: True)
        offered = expand_offered_forms(self._bridge_data(), presence=PostProcessorPresence(lane_bound=frozenset()))

        for candidate in lane_bound_post_processor_candidates():
            assert candidate not in offered, candidate
        assert "strip_background" in offered, "background removal has its own weight and its own gate"
        assert "interrogation" in offered
        assert "nsfw" in offered

    def test_beta_name_needs_both_server_support_and_the_model_on_disk(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A beta post-processor is offered only when the server lists it *and* its weight is on disk."""
        bridge_data = self._bridge_data(forms=["post-process"])
        without_beta = PostProcessorPresence(lane_bound=frozenset({"RealESRGAN_x4plus"}))
        with_beta = PostProcessorPresence(lane_bound=frozenset({"RealESRGAN_x4plus", "4xNomos8kSC"}))

        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: True)
        assert "4xNomos8kSC" not in expand_offered_forms(bridge_data, presence=without_beta)
        assert "4xNomos8kSC" in expand_offered_forms(bridge_data, presence=with_beta)

        monkeypatch.setattr(alchemy_popper, "server_supports_interrogation_form", lambda form: False)
        assert "4xNomos8kSC" not in expand_offered_forms(bridge_data, presence=with_beta)

    def test_strip_background_gates_on_its_own_weight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Background removal is withheld when its weight is known missing, and offered otherwise."""
        monkeypatch.setattr(alchemy_popper, "strip_background_available", lambda: True)
        bridge_data = self._bridge_data(forms=["post-process"])

        withheld = expand_offered_forms(bridge_data, presence=PostProcessorPresence(strip_background=False))
        assert "strip_background" not in withheld
        assert "RealESRGAN_x4plus" in withheld, "the torch post-processors do not need the rembg weight"

        present = expand_offered_forms(bridge_data, presence=PostProcessorPresence(strip_background=True))
        assert "strip_background" in present
        assert "strip_background" in expand_offered_forms(bridge_data, presence=PostProcessorPresence())

    def test_caption_gates_on_its_own_model(self) -> None:
        """Caption is withheld while its model is known missing, opt-in or not."""
        bridge_data = self._bridge_data(alchemy_caption_enabled=True)

        assert "caption" not in expand_offered_forms(bridge_data, presence=PostProcessorPresence(caption=False))
        assert "caption" in expand_offered_forms(bridge_data, presence=PostProcessorPresence(caption=True))
        assert "caption" in expand_offered_forms(bridge_data, presence=PostProcessorPresence())

    def test_config_accepts_vectorize_but_rejects_typos(self) -> None:
        """The worker's forms validator accepts the worker-known vectorize form yet still rejects typos.

        The worker overrides the SDK's stricter forms validator so it can offer vectorize against a
        published SDK that does not yet list it, without weakening typo protection.
        """
        assert self._bridge_data(forms=["vectorize"]).forms == ["vectorize"]
        with pytest.raises(ValidationError):
            self._bridge_data(forms=["vectorrize"])

    def test_config_accepts_palette_and_describe(self) -> None:
        """The worker's forms validator accepts the worker-known palette/describe forms.

        These are worker-known extras (see ``WORKER_KNOWN_EXTRA_ALCHEMY_FORMS``) so a config can list
        them against a published SDK that predates the form, without weakening typo protection.
        """
        assert self._bridge_data(forms=["palette", "describe"]).forms == ["palette", "describe"]
        with pytest.raises(ValidationError):
            self._bridge_data(forms=["palettte"])

    def test_config_accepts_aesthetic(self) -> None:
        """The worker's forms validator accepts the worker-known aesthetic form."""
        assert self._bridge_data(forms=["aesthetic"]).forms == ["aesthetic"]


class TestAlchemySubmitShapes:
    """The submit wire format matches the legacy alchemist protocol."""

    def test_text_form_submits_inline_result(self) -> None:
        """Text forms submit their result dict inline."""
        submit = PendingAlchemySubmitJob(
            result_message=_result_message("caption", result_payload={"caption": "a test image"}),
            time_popped=0.0,
        )
        assert submit.submit_result == {"caption": "a test image"}

    def test_vectorize_submits_inline_svg(self) -> None:
        """Vectorize is a text form: its SVG result submits inline with no R2 upload."""
        submit = PendingAlchemySubmitJob(
            result_message=_result_message("vectorize", result_payload={"vectorize": "<svg></svg>"}),
            time_popped=0.0,
        )
        assert submit.submit_result == {"vectorize": "<svg></svg>"}

    def test_image_form_submits_r2_sentinel(self) -> None:
        """Image forms submit the R2 sentinel after upload."""
        submit = PendingAlchemySubmitJob(
            result_message=_result_message("RealESRGAN_x4plus", image_bytes=b"hi"),
            r2_upload="https://example.com/upload",
            time_popped=0.0,
        )
        assert submit.submit_result == {"RealESRGAN_x4plus": "R2"}

    def test_submit_request_accepts_dict_result(self) -> None:
        """The submit request subclass allows the dict wire format."""
        request = _AlchemySubmitRequest(
            apikey="0000000000",
            id="00000000-0000-0000-0000-000000000000",
            result={"nsfw": False},
            state=GENERATION_STATE.ok,
        )
        assert request.result == {"nsfw": False}

    def test_aesthetic_form_submits_inline_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A successful aesthetic form reports its numeric score without terminating the child process."""
        from horde_worker_regen.process_management.workers.safety_process import HordeSafetyProcess

        process = HordeSafetyProcess.__new__(HordeSafetyProcess)
        process_queue = _RecordingQueue()
        process.process_id = 0
        process.process_launch_identifier = 1
        process.device_index = 0
        process.process_message_queue = cast(ProcessQueue, process_queue)
        monkeypatch.setattr(process, "_score_aesthetic", lambda _image: 4.9564)
        monkeypatch.setattr(process, "_send_alchemy_job_metrics", lambda _form: None)

        process.start_alchemy(
            AlchemyFormSpec(
                form_id="00000000-0000-0000-0000-000000000000",
                form="aesthetic",
                source_image_bytes=_png_bytes(),
            ),
        )

        result_messages = [
            message for message in process_queue.messages if isinstance(message, HordeAlchemyResultMessage)
        ]
        assert len(result_messages) == 1
        assert result_messages[0].state == GENERATION_STATE.ok
        assert result_messages[0].result_payload == {"aesthetic": 4.9564}
        assert PendingAlchemySubmitJob(result_message=result_messages[0], time_popped=0.0).submit_result == {
            "aesthetic": 4.9564,
        }

    def test_pop_request_accepts_plain_string_forms(self) -> None:
        """The pop request subclass offers post-processors by name without round-tripping the SDK enum."""
        # The worker offers beta post-processors (e.g. RestoreFormer) ahead of the SDK release that lists
        # them, so the pop request must accept plain strings rather than enum members.
        request = _AlchemyPopRequest(
            apikey="0000000000",
            name="test alchemist",
            priority_usernames=[],
            forms=["CodeFormers", "RestoreFormer", "caption"],
            amount=1,
        )
        assert "CodeFormers" in request.forms
        assert "RestoreFormer" in request.forms


class TestVectorizeOp:
    """The vectorizer turns a raster image into an SVG string."""

    def test_vectorize_image_returns_svg(self) -> None:
        """_vectorize_image traces a raster into a non-empty SVG string."""
        from horde_worker_regen.process_management.workers.safety_process import HordeSafetyProcess

        # The op only touches PIL + vtracer, so a bare instance (no safety models) is enough.
        process = HordeSafetyProcess.__new__(HordeSafetyProcess)

        image = PIL.Image.new("RGB", (32, 32))
        for x in range(32):
            for y in range(32):
                image.putpixel((x, y), (255, 0, 0) if (x // 8 + y // 8) % 2 == 0 else (0, 0, 255))

        svg = process._vectorize_image(image)

        assert isinstance(svg, str)
        assert "<svg" in svg
        assert len(svg) > 0


def _checkerboard(size: int = 32) -> PIL.Image.Image:
    image = PIL.Image.new("RGB", (size, size))
    block = size // 4
    for x in range(size):
        for y in range(size):
            image.putpixel((x, y), (255, 0, 0) if (x // block + y // block) % 2 == 0 else (0, 0, 255))
    return image


class TestPaletteOp:
    """The palette op extracts an ordered dominant-colour list with no model or optional dep."""

    def test_extract_palette_returns_ordered_hex_colors(self) -> None:
        """_extract_palette returns hex colours with proportions, most-prominent first."""
        from horde_worker_regen.process_management.workers.safety_process import HordeSafetyProcess

        process = HordeSafetyProcess.__new__(HordeSafetyProcess)
        result = process._extract_palette(_checkerboard(), color_count=4)

        colors = result["colors"]
        assert isinstance(colors, list)
        assert len(colors) >= 1
        for entry in colors:
            assert entry["hex"].startswith("#") and len(entry["hex"]) == 7
            assert 0.0 <= entry["proportion"] <= 1.0
        # Proportions are sorted descending (most-prominent first).
        proportions = [entry["proportion"] for entry in colors]
        assert proportions == sorted(proportions, reverse=True)


class TestDescribeOp:
    """The describe op bundles blurhash, perceptual hashes, and geometry facts."""

    def test_describe_image_returns_metadata_bundle(self) -> None:
        """_describe_image returns a blurhash, perceptual hashes, and dimensions."""
        from horde_worker_regen.process_management.workers.safety_process import HordeSafetyProcess

        process = HordeSafetyProcess.__new__(HordeSafetyProcess)
        result = process._describe_image(_checkerboard(size=64))

        assert isinstance(result["blurhash"], str) and result["blurhash"]
        assert isinstance(result["phash"], str) and len(result["phash"]) == 16
        assert isinstance(result["dhash"], str)
        assert result["width"] == 64
        assert result["height"] == 64
        assert result["aspect_ratio"] == 1.0
        assert result["has_alpha"] is False


class TestAestheticPredictor:
    """The aesthetic head scores a CLIP ViT-L/14 embedding into a float (no network in this test)."""

    def test_scorer_loads_state_dict_and_scores(self, tmp_path: object) -> None:
        """AestheticScorer loads a predictor state_dict from disk and scores an embedding to a float."""
        import torch

        from horde_worker_regen.process_management.workers.aesthetic_predictor import (
            AESTHETIC_EMBEDDING_DIM,
            AestheticPredictor,
            AestheticScorer,
        )

        # Build the predictor and persist its (randomly-initialised) weights, bypassing the network
        # download so the scorer's load+score path is exercised hermetically.
        weight_path = tmp_path / "predictor.pth"  # type: ignore[attr-defined]
        torch.save(AestheticPredictor().state_dict(), weight_path)

        scorer = AestheticScorer(weight_path, device="cpu")
        embedding = torch.randn(1, AESTHETIC_EMBEDDING_DIM)
        score = scorer.score(embedding)

        assert isinstance(score, float)
        # The scorer rounds to 4 decimals; an untrained head still yields a finite scalar.
        assert score == round(score, 4)

    def test_predictor_forward_shape(self) -> None:
        """The predictor maps a batch of embeddings to one score each."""
        import torch

        from horde_worker_regen.process_management.workers.aesthetic_predictor import (
            AESTHETIC_EMBEDDING_DIM,
            AestheticPredictor,
        )

        out = AestheticPredictor()(torch.randn(3, AESTHETIC_EMBEDDING_DIM))
        assert out.shape == (3, 1)


class _StubProcessInfo:
    """Stands in for HordeProcessInfo at the dispatch seam."""

    def __init__(self, process_id: int, process_launch_identifier: int = 0, device_index: int = 0) -> None:
        self.process_id = process_id
        self.process_launch_identifier = process_launch_identifier
        self.device_index = device_index
        self.sent_messages: list[HordeAlchemyControlMessage] = []

    def safe_send_message(self, message: HordeAlchemyControlMessage) -> bool:
        self.sent_messages.append(message)
        return True


class _StubProcessMap:
    """Stands in for ProcessMap; records which capability each dispatch asked for."""

    def __init__(
        self,
        available: dict[WorkerCapability, _StubProcessInfo],
        *,
        free_vram_mb: float | None = 8000.0,
    ) -> None:
        self.available = available
        self.free_vram_mb = free_vram_mb
        self.requested_capabilities: list[WorkerCapability] = []

    def get_first_available(
        self,
        capability: WorkerCapability,
        disallowed_processes: list[int] | None = None,
        *,
        device_index: int | None = None,
    ) -> _StubProcessInfo | None:
        self.requested_capabilities.append(capability)
        process_info = self.available.get(capability)
        if process_info is None or (device_index is not None and process_info.device_index != device_index):
            return None
        return process_info

    def get_capable_processes(self, capability: WorkerCapability) -> list[_StubProcessInfo]:
        process_info = self.available.get(capability)
        return [process_info] if process_info is not None else []

    def get_free_vram_mb(self, *, device_index: int | None = None) -> float | None:
        return self.free_vram_mb


def _make_coordinator(process_map: _StubProcessMap) -> AlchemyCoordinator:
    coordinator = AlchemyCoordinator.__new__(AlchemyCoordinator)
    coordinator._process_map = process_map  # type: ignore[assignment]
    coordinator._runtime_config = _StubRuntimeConfig(  # type: ignore[assignment]
        reGenBridgeData(api_key="0" * 22, alchemist=True, alchemy_vram_headroom_mb=2000),  # type: ignore[arg-type]
    )
    coordinator._job_tracker = _StubJobTracker()  # type: ignore[assignment]
    coordinator._reserve_ledger = CommittedReserveLedger()
    coordinator._pending_forms = deque()
    coordinator._pending_form_deadlines = {}
    coordinator._in_flight = {}
    coordinator._in_flight_card = {}
    coordinator._in_flight_owner = {}
    coordinator._pending_submits = deque()
    coordinator._form_time_popped = {}
    coordinator._estimator = AlchemyHeadroomEstimator()
    coordinator._free_vram_baseline_mb = {}
    coordinator._min_free_vram_mb = {}
    coordinator.num_forms_faulted = 0
    return coordinator


class TestAlchemyDispatch:
    """The coordinator dispatches queued forms by required capability."""

    def test_graph_and_clip_forms_dispatch_by_capability(self) -> None:
        """Graph and CLIP forms go to their respective capable processes."""
        graph_process = _StubProcessInfo(process_id=1)
        clip_process = _StubProcessInfo(process_id=2)
        process_map = _StubProcessMap(
            {
                WorkerCapability.ALCHEMY_GRAPH: graph_process,
                WorkerCapability.ALCHEMY_CLIP: clip_process,
            },
        )
        coordinator = _make_coordinator(process_map)

        upscale_form = AlchemyFormSpec(form_id="form-1", form="RealESRGAN_x4plus", source_image_bytes=b"hi")
        caption_form = AlchemyFormSpec(form_id="form-2", form="caption", source_image_bytes=b"hi")
        coordinator._pending_forms.extend([upscale_form, caption_form])

        coordinator.dispatch_pending_forms()

        assert len(coordinator._pending_forms) == 0
        assert set(coordinator._in_flight) == {"form-1", "form-2"}
        assert [m.form.form_id for m in graph_process.sent_messages] == ["form-1"]
        assert [m.form.form_id for m in clip_process.sent_messages] == ["form-2"]
        assert all(m.control_flag == HordeControlFlag.START_ALCHEMY for m in graph_process.sent_messages)

    def test_form_stays_queued_when_no_capable_process(self) -> None:
        """A form stays queued until a capable process is available."""
        process_map = _StubProcessMap({})
        coordinator = _make_coordinator(process_map)
        form = AlchemyFormSpec(form_id="form-1", form="nsfw", source_image_bytes=b"hi")
        coordinator._pending_forms.append(form)

        coordinator.dispatch_pending_forms()

        assert list(coordinator._pending_forms) == [form]
        assert coordinator._in_flight == {}

    def test_a_queued_form_carries_a_dispatch_deadline(self) -> None:
        """Queueing a popped form starts its monotonic dispatch deadline."""
        coordinator = _make_coordinator(_StubProcessMap({}))
        form = AlchemyFormSpec(form_id="form-1", form="nsfw", source_image_bytes=b"hi")

        before = time.monotonic()
        coordinator._queue_pending_form(form)

        deadline = coordinator._pending_form_deadlines["form-1"]
        assert deadline >= before + alchemy_popper.PENDING_FORM_DISPATCH_DEADLINE_SECONDS
        assert list(coordinator._pending_forms) == [form]

    def test_an_undispatchable_form_waits_until_its_deadline(self) -> None:
        """A form no process can take is held, not faulted, while its deadline is still ahead."""
        coordinator = _make_coordinator(_StubProcessMap({}))
        coordinator._queue_pending_form(AlchemyFormSpec(form_id="form-1", form="nsfw", source_image_bytes=b"hi"))

        coordinator.dispatch_pending_forms()

        assert [spec.form_id for spec in coordinator._pending_forms] == ["form-1"]
        assert len(coordinator._pending_submits) == 0
        assert "form-1" in coordinator._pending_form_deadlines

    def test_an_undispatchable_form_is_faulted_once_at_its_deadline(self) -> None:
        """A form whose lane never returned is handed back faulted exactly once, and logged once."""
        coordinator = _make_coordinator(_StubProcessMap({}))
        form = AlchemyFormSpec(form_id="form-1", form="nsfw", source_image_bytes=b"hi", r2_upload="https://r2/put")
        coordinator._queue_pending_form(form)
        coordinator._form_time_popped["form-1"] = 123.0
        coordinator._pending_form_deadlines["form-1"] = time.monotonic() - 1.0

        lines: list[str] = []
        sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="WARNING")
        try:
            coordinator.dispatch_pending_forms()
            coordinator.dispatch_pending_forms()
        finally:
            logger.remove(sink_id)

        assert len(coordinator._pending_forms) == 0
        assert coordinator._pending_form_deadlines == {}
        assert len(coordinator._pending_submits) == 1
        submit = coordinator._pending_submits[0]
        assert submit.result_message.state == GENERATION_STATE.faulted
        assert submit.form_id == "form-1"
        assert submit.r2_upload == "https://r2/put"
        # The fault is reported to the horde from the form's own pop instant, so its recorded e2e is real.
        assert submit.time_popped == 123.0
        assert len([line for line in lines if "form-1" in line]) == 1, lines

    def test_a_form_dispatched_before_its_deadline_is_not_faulted(self) -> None:
        """A lane that comes back in time serves the form; nothing is handed back."""
        clip_process = _StubProcessInfo(process_id=2)
        process_map = _StubProcessMap({})
        coordinator = _make_coordinator(process_map)
        coordinator._queue_pending_form(AlchemyFormSpec(form_id="form-1", form="nsfw", source_image_bytes=b"hi"))

        coordinator.dispatch_pending_forms()
        assert [spec.form_id for spec in coordinator._pending_forms] == ["form-1"]

        process_map.available[WorkerCapability.ALCHEMY_CLIP] = clip_process
        coordinator.dispatch_pending_forms()

        assert set(coordinator._in_flight) == {"form-1"}
        assert coordinator._pending_form_deadlines == {}
        assert len(coordinator._pending_submits) == 0
        assert [m.form.form_id for m in clip_process.sent_messages] == ["form-1"]

    def test_a_servable_form_is_dispatched_rather_than_faulted_at_its_deadline(self) -> None:
        """The deadline only ever decides a form nothing can take; a free lane is served either way."""
        clip_process = _StubProcessInfo(process_id=2)
        coordinator = _make_coordinator(_StubProcessMap({WorkerCapability.ALCHEMY_CLIP: clip_process}))
        coordinator._queue_pending_form(AlchemyFormSpec(form_id="form-1", form="nsfw", source_image_bytes=b"hi"))
        coordinator._pending_form_deadlines["form-1"] = time.monotonic() - 1.0

        coordinator.dispatch_pending_forms()

        assert set(coordinator._in_flight) == {"form-1"}
        assert len(coordinator._pending_submits) == 0

    def test_a_form_queued_without_a_deadline_never_expires(self) -> None:
        """Only a deadline this coordinator set can fault a form, so an externally-queued form is safe."""
        coordinator = _make_coordinator(_StubProcessMap({}))
        coordinator._pending_forms.append(AlchemyFormSpec(form_id="form-1", form="nsfw", source_image_bytes=b"hi"))

        coordinator.dispatch_pending_forms()

        assert [spec.form_id for spec in coordinator._pending_forms] == ["form-1"]
        assert len(coordinator._pending_submits) == 0

    def test_result_moves_form_to_pending_submit(self) -> None:
        """A result message moves the form from in-flight to pending submit."""
        process_map = _StubProcessMap({WorkerCapability.ALCHEMY_CLIP: _StubProcessInfo(process_id=2)})
        coordinator = _make_coordinator(process_map)
        form = AlchemyFormSpec(form_id="form-1", form="nsfw", source_image_bytes=b"hi", r2_upload=None)
        coordinator._pending_forms.append(form)
        coordinator._form_time_popped["form-1"] = 123.0
        coordinator.dispatch_pending_forms()

        coordinator.on_alchemy_result(
            HordeAlchemyResultMessage(
                process_id=2,
                process_launch_identifier=0,
                info="test",
                form_id="form-1",
                form="nsfw",
                state=GENERATION_STATE.ok,
                result_payload={"nsfw": False},
            ),
        )

        assert coordinator._in_flight == {}
        assert len(coordinator._pending_submits) == 1
        submit = coordinator._pending_submits[0]
        assert submit.submit_result == {"nsfw": False}
        assert submit.time_popped == 123.0


class TestAnnotationTypesProvider:
    """The coordinator narrows the lane's servable set to control types the pop enum can name."""

    def test_intersects_with_known_control_types(self) -> None:
        """A servable type the pop enum cannot name is dropped, so the enum-typed field never rejects it."""
        coordinator = AlchemyCoordinator.__new__(AlchemyCoordinator)
        coordinator._annotation_types_provider = lambda: frozenset({"canny", "not_a_real_control_type"})
        assert coordinator._annotation_types() == frozenset({"canny"})

    def test_empty_without_a_provider(self) -> None:
        """With no provider wired the coordinator reports no servable types, so the form is withheld."""
        coordinator = AlchemyCoordinator.__new__(AlchemyCoordinator)
        coordinator._annotation_types_provider = None
        assert coordinator._annotation_types() == frozenset()


class TestAlchemyHeadroomEstimator:
    """The estimator predicts VRAM cost from observed runs, bounded by the configured floor."""

    def test_cold_start_returns_floor(self) -> None:
        """With no observations the prediction is exactly the floor."""
        estimator = AlchemyHeadroomEstimator()
        assert estimator.predicted_cost_mb(2000.0) == 2000.0
        assert estimator.median_duration_s is None

    def test_floor_is_a_lower_bound(self) -> None:
        """Observed costs below the floor never lower the prediction beneath it."""
        estimator = AlchemyHeadroomEstimator()
        estimator.record_run(vram_cost_mb=100.0, duration_s=1.0)
        estimator.record_run(vram_cost_mb=150.0, duration_s=1.0)
        assert estimator.predicted_cost_mb(2000.0) == 2000.0

    def test_prediction_rises_to_observed_median(self) -> None:
        """Above-floor observations raise the prediction to their median."""
        estimator = AlchemyHeadroomEstimator()
        for cost in (2500.0, 3500.0, 4500.0):
            estimator.record_run(vram_cost_mb=cost, duration_s=2.0)
        assert estimator.predicted_cost_mb(2000.0) == 3500.0
        assert estimator.median_duration_s == 2.0

    def test_non_positive_samples_ignored(self) -> None:
        """Zero/negative VRAM deltas (noise) are discarded."""
        estimator = AlchemyHeadroomEstimator()
        estimator.record_run(vram_cost_mb=0.0, duration_s=None)
        estimator.record_run(vram_cost_mb=-500.0, duration_s=0.0)
        assert estimator.predicted_cost_mb(2000.0) == 2000.0
        assert estimator.median_duration_s is None

    def test_fits_compares_against_prediction(self) -> None:
        """`fits` is True only when free VRAM covers the prediction."""
        estimator = AlchemyHeadroomEstimator()
        assert estimator.fits(free_vram_mb=2500.0, floor_mb=2000.0)
        assert not estimator.fits(free_vram_mb=1500.0, floor_mb=2000.0)


class _StubState:
    def __init__(
        self,
        *,
        shutting_down: bool = False,
        supervisor_paused: bool = False,
        self_throttle_paused: bool = False,
        gpu_torch_incompatible: bool = False,
        recovery_parked: bool = False,
        downloads_only_hold: bool = False,
    ) -> None:
        self.shutting_down = shutting_down
        self.supervisor_paused = supervisor_paused
        self.self_throttle_paused = self_throttle_paused
        self.gpu_torch_incompatible = gpu_torch_incompatible
        self.recovery_parked = recovery_parked
        self.downloads_only_hold = downloads_only_hold

    @property
    def workload_intake_paused(self) -> bool:
        """Mirror the shared WorkerState intake boundary for the policy-only test double."""
        return (
            self.shutting_down
            or self.supervisor_paused
            or self.self_throttle_paused
            or self.recovery_parked
            or self.downloads_only_hold
        )


class _StubRuntimeConfig:
    def __init__(self, bridge_data: reGenBridgeData) -> None:
        self.bridge_data = bridge_data


class _StubJobTracker:
    def __init__(self, *, pending: int = 0, in_progress: int = 0) -> None:
        self._pending = pending
        self._in_progress = in_progress

    @property
    def jobs_pending_inference(self) -> tuple[int, ...]:
        return tuple(range(self._pending))

    @property
    def jobs_in_progress(self) -> tuple[int, ...]:
        return tuple(range(self._in_progress))


class _IdleProc:
    def can_accept_job(self) -> bool:
        return True


class _PolicyProcessMap:
    def __init__(
        self,
        *,
        graph: object = None,
        clip: object = None,
        idle_image_lanes: int = 0,
        free_vram_mb: float | None = None,
        utilities_lanes: int = 0,
    ) -> None:
        self._graph = graph
        self._clip = clip
        self._image_lanes = [_IdleProc() for _ in range(idle_image_lanes)]
        self._free_vram_mb = free_vram_mb
        self._utilities_lanes = utilities_lanes

    def get_first_available(
        self,
        capability: WorkerCapability,
        disallowed_processes: list[int] | None = None,
        *,
        device_index: int | None = None,
    ) -> object:
        if capability is WorkerCapability.ALCHEMY_GRAPH:
            return self._graph
        if capability is WorkerCapability.ALCHEMY_CLIP:
            return self._clip
        if capability is WorkerCapability.IMAGE_UTILITIES and self._utilities_lanes > 0:
            return object()
        return None

    def get_capable_processes(self, capability: WorkerCapability) -> list[_IdleProc]:
        if capability is WorkerCapability.IMAGE_GEN:
            return self._image_lanes
        return []

    def get_free_vram_mb(self, *, device_index: int | None = None) -> float | None:
        return self._free_vram_mb

    def num_loaded_utilities_processes(self) -> int:
        return self._utilities_lanes

    def num_loaded_post_process_processes(self) -> int:
        # The graph capability is served only by the post-processing lane, so the two readings agree.
        return 1 if self._graph is not None else 0

    def num_loaded_safety_processes(self) -> int:
        # The CLIP capability is served only by the safety process, so the two readings agree.
        return 1 if self._clip is not None else 0


def _make_policy_coordinator(
    *,
    bridge_data: reGenBridgeData,
    process_map: _PolicyProcessMap,
    job_tracker: _StubJobTracker,
    in_flight: int = 0,
) -> AlchemyCoordinator:
    coordinator = AlchemyCoordinator.__new__(AlchemyCoordinator)
    coordinator._runtime_config = _StubRuntimeConfig(bridge_data)  # type: ignore[assignment]
    coordinator._state = _StubState()  # type: ignore[assignment]
    coordinator._process_map = process_map  # type: ignore[assignment]
    coordinator._job_tracker = job_tracker  # type: ignore[assignment]
    coordinator._reserve_ledger = CommittedReserveLedger()
    coordinator._pending_forms = deque()
    coordinator._pending_form_deadlines = {}
    coordinator._in_flight = {f"form-{i}": None for i in range(in_flight)}  # type: ignore[misc]
    coordinator._in_flight_card = {}
    coordinator._in_flight_owner = {}
    coordinator._estimator = AlchemyHeadroomEstimator()
    coordinator._last_pop_time = 0.0
    # The pop gate reads the hold as well as the instant, and this double skips __init__.
    coordinator._pop_hold_until = 0.0
    coordinator._pop_frequency = 4.0
    return coordinator


class TestShouldPopPolicy:
    """`_should_pop` enforces image priority, spare-lane, VRAM headroom, and the in-flight cap."""

    def _bridge_data(self, **kwargs: object) -> reGenBridgeData:
        defaults: dict[str, object] = {"api_key": "0000000000", "alchemist": True, "queue_size": 3}
        defaults.update(kwargs)
        return reGenBridgeData(**defaults)  # type: ignore[arg-type]

    def test_backfill_mode_blocks_while_image_queue_busy(self) -> None:
        """With concurrency off, alchemy waits for the image queue to drain."""
        bridge_data = self._bridge_data(alchemy_allow_concurrent=False)
        process_map = _PolicyProcessMap(clip=object(), graph=object(), idle_image_lanes=2, free_vram_mb=8000.0)
        coordinator = _make_policy_coordinator(
            bridge_data=bridge_data,
            process_map=process_map,
            job_tracker=_StubJobTracker(pending=1),
        )
        assert coordinator._should_pop() is False

    def test_backfill_mode_allows_when_image_queue_empty(self) -> None:
        """With concurrency off, alchemy pops once no image jobs are queued."""
        bridge_data = self._bridge_data(alchemy_allow_concurrent=False)
        process_map = _PolicyProcessMap(clip=object(), graph=object(), idle_image_lanes=2, free_vram_mb=8000.0)
        coordinator = _make_policy_coordinator(
            bridge_data=bridge_data,
            process_map=process_map,
            job_tracker=_StubJobTracker(pending=0),
        )
        assert coordinator._should_pop() is True

    def test_concurrent_mode_allows_with_spare_lane_and_vram(self) -> None:
        """Concurrent mode pops alongside image work when a lane is spare and VRAM fits."""
        bridge_data = self._bridge_data()
        process_map = _PolicyProcessMap(graph=object(), clip=object(), idle_image_lanes=2, free_vram_mb=8000.0)
        coordinator = _make_policy_coordinator(
            bridge_data=bridge_data,
            process_map=process_map,
            job_tracker=_StubJobTracker(pending=1, in_progress=0),
        )
        assert coordinator._should_pop() is True

    def test_concurrent_mode_blocks_without_spare_lane(self) -> None:
        """No spare lane (every idle lane is needed by a queued image job) blocks the pop."""
        bridge_data = self._bridge_data()
        process_map = _PolicyProcessMap(graph=object(), clip=object(), idle_image_lanes=1, free_vram_mb=8000.0)
        coordinator = _make_policy_coordinator(
            bridge_data=bridge_data,
            process_map=process_map,
            job_tracker=_StubJobTracker(pending=1, in_progress=0),
        )
        assert coordinator._should_pop() is False

    def test_concurrent_mode_blocks_without_vram_headroom(self) -> None:
        """Insufficient free VRAM blocks the pop even with a spare lane."""
        bridge_data = self._bridge_data(alchemy_vram_headroom_mb=2000)
        process_map = _PolicyProcessMap(graph=object(), clip=object(), idle_image_lanes=2, free_vram_mb=500.0)
        coordinator = _make_policy_coordinator(
            bridge_data=bridge_data,
            process_map=process_map,
            job_tracker=_StubJobTracker(pending=0),
        )
        assert coordinator._should_pop() is False

    def test_in_flight_cap_blocks_pop(self) -> None:
        """The alchemy_max_concurrency cap bounds how many forms are in flight at once."""
        bridge_data = self._bridge_data(alchemy_max_concurrency=1)
        process_map = _PolicyProcessMap(graph=object(), clip=object(), idle_image_lanes=2, free_vram_mb=8000.0)
        coordinator = _make_policy_coordinator(
            bridge_data=bridge_data,
            process_map=process_map,
            job_tracker=_StubJobTracker(pending=0),
            in_flight=1,
        )
        assert coordinator._should_pop() is False

    @pytest.mark.parametrize(
        ("hold", "should_pop"),
        [("recovery", False), ("downloads-only", False), ("none", True)],
    )
    @pytest.mark.parametrize(
        ("lane", "form"),
        [
            ("graph", "RealESRGAN_x4plus"),
            ("clip", "interrogation"),
            ("utilities", "strip_background"),
        ],
        ids=["post-processing", "safety", "utilities"],
    )
    def test_worker_wide_holds_block_alchemy_across_servable_lanes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        hold: str,
        should_pop: bool,
        lane: str,
        form: str,
    ) -> None:
        """Worker-wide holds reject alchemy work even when a surviving lane could accept it.

        The active arm for each lane proves that the arranged process map, form offer, queue, timing, and
        headroom otherwise pass the admission policy. This keeps a held result attributable to the worker
        posture rather than to an unavailable lane or malformed offer. Download-only and terminal-recovery
        holds share this boundary because both promise that no workload is accepted while GPU pools are held.
        """
        monkeypatch.setattr(alchemy_popper, "expand_offered_forms", lambda *_args, **_kwargs: [form])
        process_map = _PolicyProcessMap(
            graph=object() if lane == "graph" else None,
            clip=object() if lane == "clip" else None,
            utilities_lanes=1 if lane == "utilities" else 0,
            idle_image_lanes=1,
            free_vram_mb=8_000.0,
        )
        coordinator = _make_policy_coordinator(
            bridge_data=self._bridge_data(alchemy_allow_concurrent=False),
            process_map=process_map,
            job_tracker=_StubJobTracker(pending=0),
        )
        coordinator._state.recovery_parked = hold == "recovery"
        coordinator._state.downloads_only_hold = hold == "downloads-only"

        assert coordinator._should_pop() is should_pop

    def test_a_down_clip_lane_narrows_the_offer_without_closing_the_pop_gate(self) -> None:
        """With safety down the CLIP forms leave the offer, and the graph lane still pops its own work."""
        bridge_data = self._bridge_data(alchemy_allow_concurrent=False)
        process_map = _PolicyProcessMap(graph=object(), clip=None, idle_image_lanes=2, free_vram_mb=8000.0)
        coordinator = _make_policy_coordinator(
            bridge_data=bridge_data,
            process_map=process_map,
            job_tracker=_StubJobTracker(pending=0),
        )

        offered = expand_offered_forms(bridge_data, clip_lane_healthy=coordinator._clip_lane_healthy())
        assert "nsfw" not in offered
        assert "RealESRGAN_x4plus" in offered
        assert coordinator._should_pop() is True

    def test_a_clip_only_worker_stops_popping_while_safety_is_down(self) -> None:
        """Nothing is left to offer when the only configured family's lane is down, so no pop goes out."""
        bridge_data = self._bridge_data(forms=["nsfw"], alchemy_allow_concurrent=False)
        process_map = _PolicyProcessMap(graph=object(), clip=None, idle_image_lanes=2, free_vram_mb=8000.0)
        coordinator = _make_policy_coordinator(
            bridge_data=bridge_data,
            process_map=process_map,
            job_tracker=_StubJobTracker(pending=0),
        )

        assert coordinator._should_pop() is False

    def test_unknown_vram_falls_back_to_backfill(self) -> None:
        """When VRAM telemetry is unavailable, alchemy only pops with an empty image queue."""
        bridge_data = self._bridge_data()
        process_map = _PolicyProcessMap(graph=object(), clip=object(), idle_image_lanes=2, free_vram_mb=None)
        busy = _make_policy_coordinator(
            bridge_data=bridge_data,
            process_map=process_map,
            job_tracker=_StubJobTracker(pending=1, in_progress=0),
        )
        assert busy._should_pop() is False

        idle = _make_policy_coordinator(
            bridge_data=bridge_data,
            process_map=process_map,
            job_tracker=_StubJobTracker(pending=0),
        )
        assert idle._should_pop() is True
