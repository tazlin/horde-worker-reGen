"""Contains the classes to form a safety process, which is responsible for evaluating the safety of images.

The safety process also serves the CLIP-stack alchemy forms (caption, interrogation, nsfw):
it already owns the CLIP interrogator and NSFW checker, so those forms run here rather than
in the (comfy-loaded) inference processes.
"""

import base64
import enum
import time
from enum import auto
from io import BytesIO
from typing import TYPE_CHECKING

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock
from typing import override

import PIL
import PIL.Image
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.generation_parameters.alchemy.consts import (
    is_caption_form,
    is_interrogator_form,
    is_nsfw_detector_form,
)
from loguru import logger

from horde_worker_regen import ASSETS_FOLDER_PATH
from horde_worker_regen.consts import is_aesthetic_form, is_describe_form, is_palette_form, is_vectorize_form
from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.ipc.messages import (
    AlchemyFormSpec,
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessState,
    HordeSafetyControlMessage,
    HordeSafetyEvaluation,
    HordeSafetyResultMessage,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcess

if TYPE_CHECKING:
    from horde_safety.deep_danbooru_model import DeepDanbooruModel
    from horde_safety.interrogate import Interrogator
    from horde_safety.nsfw_checker_class import NSFWChecker, NSFWResult

    from horde_worker_regen.process_management.workers.aesthetic_predictor import AestheticScorer
else:

    class Interrogator:
        """Dummy class to prevent type errors."""

    class NSFWChecker:
        """Dummy class to prevent type errors."""

    class NSFWResult:
        """Dummy class to prevent type errors."""

    class DeepDanbooruModel:
        """Dummy class to prevent type errors."""


def resolve_safety_device(*, cpu_only: bool, cuda_available: bool) -> str:
    """Resolve the device horde_safety models load onto, falling back to CPU when CUDA is truly absent.

    The parent decides ``cpu_only`` from config plus the torch-free install sentinel, which can report
    "GPU" while the actual torch build is CPU-only: a manually installed CPU torch whose ``bin/backend``
    sentinel was never written is the motivating case. Loading a model on ``"cuda"`` then raises during
    deserialization ("Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is
    False"). The safety child is torch-bearing, so it has the final say: when CUDA is not actually
    available it uses CPU regardless of the flag. It never forces CPU when CUDA is present, so a normal
    GPU worker is unaffected.
    """
    if cpu_only or not cuda_available:
        return "cpu"
    return "cuda"


class CensorReason(enum.Enum):
    """The reason for censoring an image."""

    CSAM = auto()
    CENSORLIST = auto()
    SFW_REQUEST = auto()
    SFW_WORKER = auto()


class HordeSafetyProcess(HordeProcess):
    """The safety process, which is responsible for evaluating the safety of images."""

    _interrogator: Interrogator
    _deep_danbooru_model: DeepDanbooruModel

    _nsfw_checker: NSFWChecker

    censor_csam_image_base64: str
    censor_censorlist_image_base64: str
    censor_sfw_request_image_base64: str
    censor_sfw_worker_image_base64: str

    _dry_run_skip_safety: bool

    _safety_device: str = "cpu"
    """The torch device the safety models (and the aesthetic head) run on."""
    _aesthetic_scorer: "AestheticScorer | None" = None
    """Lazily-loaded LAION aesthetic head; built on first use and shared by the form and gen-metadata paths."""
    _aesthetic_unavailable: bool = False
    """Latched once the aesthetic weight cannot be obtained, so the cheap per-image path stops retrying."""

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        cpu_only: bool = True,
        *,
        dry_run_skip_safety: bool = False,
    ) -> None:
        """Initialise the safety process.

        Args:
            process_id (int): The ID of the process.
            process_message_queue (ProcessQueue): The process message queue.
            pipe_connection (Connection): The connection to the parent process.
            disk_lock (Lock): The lock to use when accessing the disk.
            process_launch_identifier (int): The unique identifier for this launch.
            cpu_only (bool, optional): Whether to only use the CPU. Defaults to True.
            dry_run_skip_safety (bool, optional): Skip real safety evaluation. Defaults to False.
        """
        super().__init__(
            process_id=process_id,
            process_message_queue=process_message_queue,
            pipe_connection=pipe_connection,
            disk_lock=disk_lock,
            process_launch_identifier=process_launch_identifier,
        )

        self._dry_run_skip_safety = dry_run_skip_safety
        self._label_tables = {}

        if not dry_run_skip_safety:
            try:
                from horde_safety.deep_danbooru_model import get_deep_danbooru_model
                from horde_safety.interrogate import get_interrogator_no_blip
            except Exception as e:
                logger.error(f"Failed to import horde_safety: {type(e).__name__} {e}")
                raise

            try:
                import torch

                device = resolve_safety_device(cpu_only=cpu_only, cuda_available=torch.cuda.is_available())
                self._safety_device = device
                if device == "cpu" and not cpu_only:
                    logger.warning(
                        "Safety process was configured for GPU but torch reports no CUDA device available; "
                        "loading safety models on CPU instead. This is expected with a CPU-only torch build "
                        "whose install sentinel was not set (e.g. a manually installed CPU torch).",
                    )
                logger.debug(f"Initialising horde_safety on device={device} (cpu_only={cpu_only})")
                self._deep_danbooru_model = get_deep_danbooru_model(device=device)
                self._interrogator = get_interrogator_no_blip(device=device)
            except Exception as e:
                logger.error(f"Failed to initialise horde_safety: {type(e).__name__} {e}")
                raise

            try:
                from horde_safety.nsfw_checker_class import NSFWChecker

                self._nsfw_checker = NSFWChecker(
                    self._interrogator,
                    self._deep_danbooru_model,
                )
            except Exception as e:
                logger.error(f"Failed to initialise NSFWChecker: {type(e).__name__} {e}")
                raise

            try:
                self.load_censor_files()
            except Exception as e:
                logger.error(f"Failed to load censor files: {type(e).__name__} {e}")
                raise
        else:
            logger.info("Dry-run mode: skipping safety model initialisation")

        info_message = "Horde safety process started."

        logger.info(info_message)
        self.send_process_state_change_message(
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info=info_message,
        )

        if not dry_run_skip_safety:
            logger.info(
                "The first job will always take several seconds longer when on CPU. Subsequent jobs will be faster.",
            )

    def _set_censor_image(self, reason: CensorReason, image_base64: str) -> None:
        if reason == CensorReason.CSAM:
            self.censor_csam_image_base64 = image_base64
        elif reason == CensorReason.CENSORLIST:
            self.censor_censorlist_image_base64 = image_base64
        elif reason == CensorReason.SFW_REQUEST:
            self.censor_sfw_request_image_base64 = image_base64
        elif reason == CensorReason.SFW_WORKER:
            self.censor_sfw_worker_image_base64 = image_base64
        else:
            raise ValueError(f"Unknown censor reason: {reason}")

    def load_censor_files(self) -> None:
        """Load the censor images from disk."""
        file_lookup = {
            CensorReason.CSAM: "nsfw_censor_csam.png",
            CensorReason.CENSORLIST: "nsfw_censor_censorlist.png",
            CensorReason.SFW_REQUEST: "nsfw_censor_sfw_request.png",
            CensorReason.SFW_WORKER: "nsfw_censor_sfw_worker.png",
        }

        for reason in CensorReason:
            with open(ASSETS_FOLDER_PATH / file_lookup[reason], "rb") as f:
                self._set_censor_image(reason, base64.b64encode(f.read()).decode("utf-8"))

    _caption_model_loaded: bool = False
    """Whether BLIP has been (lazily) loaded for caption forms."""
    _ranking_lists: dict[str, list[str]] | None = None
    """The interrogation ranking word lists, loaded on first interrogation form."""
    _label_tables: dict[str, object]
    """Per-category CLIP text embedding tables, built lazily from the ranking lists."""

    def _ensure_caption_model(self) -> None:
        """Load BLIP into the interrogator on first caption form (significant RAM/VRAM cost)."""
        if self._caption_model_loaded:
            return
        logger.info("Loading caption (BLIP) model for the first caption alchemy form...")
        self._interrogator.load_caption_model()  # type: ignore[attr-defined]
        self._caption_model_loaded = True
        self.send_memory_report_message(include_vram=False)

    def _get_ranking_lists(self) -> dict[str, list[str]]:
        """Load the legacy interrogation ranking lists (vendored from clipfree) once."""
        if self._ranking_lists is None:
            lists: dict[str, list[str]] = {}
            for file in sorted((ASSETS_FOLDER_PATH / "ranking_lists").glob("*.txt")):
                lines = file.read_text(encoding="utf-8").splitlines()
                lists[file.stem] = [line.strip() for line in lines if line.strip()]
            self._ranking_lists = lists
        return self._ranking_lists

    def _interrogate_image(
        self,
        image: PIL.Image.Image,
        top_count: int = 5,
    ) -> dict[str, list[dict[str, str | float]]]:
        """Rank the legacy interrogation word lists against the image.

        Reproduces the legacy alchemist result shape: ``{category: [{"text", "confidence"}, ...]}``
        using the same softmax-of-scaled-similarities ranking the old clipfree interrogator used.
        """
        import torch
        from clip_interrogator import LabelTable

        image_features = self._interrogator.image_to_features(image)  # type: ignore[attr-defined]

        results: dict[str, list[dict[str, str | float]]] = {}
        for category, labels in self._get_ranking_lists().items():
            table = self._label_tables.get(category)
            if table is None:
                table = LabelTable(labels, f"alchemy_{category}", self._interrogator)
                self._label_tables[category] = table

            # LabelTable.embeds entries are tensors on a fresh build but numpy arrays when
            # clip_interrogator loads them from its on-disk cache.
            embeds = [
                e if isinstance(e, torch.Tensor) else torch.from_numpy(e)
                for e in table.embeds  # type: ignore[attr-defined]
            ]
            text_features = torch.stack(embeds).to(image_features.device)
            if text_features.dim() == 3:
                text_features = text_features.squeeze(1)

            with torch.no_grad():
                similarity = (100.0 * image_features.float() @ text_features.float().T).softmax(dim=-1)

            count = min(top_count, len(labels))
            top_probs, top_idx = similarity.cpu().topk(count, dim=-1)
            results[category] = [
                {
                    "text": table.labels[int(top_idx[0][i])],  # type: ignore[attr-defined]
                    "confidence": float(top_probs[0][i]) * 100,
                }
                for i in range(count)
            ]
        return results

    def _vectorize_image(self, image: PIL.Image.Image) -> str:
        """Trace a raster image into an SVG string with vtracer.

        vtracer is a worker-local optional dependency (the ``vectorize`` extra); the form is only
        offered when it imports, so an absent install surfaces as a faulted form rather than a
        silently dropped one. PNG is used as the intermediate because it is lossless and vtracer
        accepts the raw bytes directly.
        """
        import vtracer

        png_buffer = BytesIO()
        image.save(png_buffer, format="PNG")
        return vtracer.convert_raw_image_to_svg(png_buffer.getvalue(), img_format="png")

    def _extract_palette(self, image: PIL.Image.Image, color_count: int = 6) -> dict[str, object]:
        """Extract a dominant-colour palette from an image, ordered most-prominent first.

        Pillow's median-cut quantizer does the clustering, so this needs no model and no optional
        dependency. The image is thumbnailed first because palette fidelity does not benefit from full
        resolution and the quantizer cost scales with pixel count. Each entry carries the colour as a
        ``#rrggbb`` hex string and its ``proportion`` of sampled pixels (the quantizer's per-bucket
        pixel counts), so a caller can weight by prominence.
        """
        sample = image.convert("RGB")
        sample.thumbnail((256, 256))
        quantized = sample.quantize(colors=color_count, method=PIL.Image.Quantize.MEDIANCUT)

        flat_palette = quantized.getpalette() or []
        bucket_counts = quantized.getcolors() or []  # list of (pixel_count, palette_index)
        total_pixels = sum(count for count, _ in bucket_counts) or 1

        colors: list[dict[str, object]] = []
        for count, index in sorted(bucket_counts, reverse=True):
            r, g, b = flat_palette[index * 3 : index * 3 + 3]
            colors.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "proportion": round(count / total_pixels, 4)})

        return {"colors": colors}

    def _describe_image(self, image: PIL.Image.Image) -> dict[str, object]:
        """Produce a cheap technical-metadata bundle for an image.

        Bundles a BlurHash placeholder string (for progressive-load UIs), perceptual hashes (for
        dedup/similarity), and basic geometry/format facts. BlurHash and the perceptual hashes come
        from the worker-only ``describe`` dependencies; the geometry facts use Pillow directly. The form
        is only offered when those dependencies import (see ``capabilities.describe_available``), so a
        missing install surfaces as a faulted form rather than a partial result.
        """
        import blurhash
        import imagehash
        import numpy as np

        rgb_image = image.convert("RGB")

        # halcy's pure-python blurhash takes an (H, W, 3) pixel array rather than a file buffer, which
        # also lets us skip a PNG re-encode of an image we already hold decoded.
        blurhash_string = blurhash.encode(np.array(rgb_image), components_x=4, components_y=3)

        width, height = image.size
        return {
            "blurhash": blurhash_string,
            "phash": str(imagehash.phash(rgb_image)),
            "dhash": str(imagehash.dhash(rgb_image)),
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 4) if height else None,
            "has_alpha": image.mode in ("RGBA", "LA") or "transparency" in image.info,
            "format": image.format,
        }

    def _score_aesthetic(self, image: PIL.Image.Image) -> float | None:
        """Return the LAION 0-10 aesthetic score for an image, or ``None`` if the head is unavailable.

        Reuses the interrogator's CLIP ViT-L/14 embedding (``image_to_features``) rather than embedding
        the image again, then runs the small MLP head. The head is loaded lazily on first use; a failure
        to obtain its weight is latched so the cheap per-generation path does not retry the download for
        every image.
        """
        if self._aesthetic_unavailable:
            return None
        if self._aesthetic_scorer is None:
            try:
                from horde_worker_regen.process_management.workers.aesthetic_predictor import load_aesthetic_scorer

                self._aesthetic_scorer = load_aesthetic_scorer(device=self._safety_device)
            except Exception as e:
                logger.warning(f"Aesthetic scoring unavailable (could not load predictor): {type(e).__name__} {e}")
                self._aesthetic_unavailable = True
                return None

        image_features = self._interrogator.image_to_features(image)  # type: ignore[attr-defined]
        return self._aesthetic_scorer.score(image_features)

    def _send_alchemy_job_metrics(self, form: AlchemyFormSpec) -> None:
        """Snapshot hordelib's metrics collector for this form and forward it (best effort)."""
        try:
            from hordelib.api import get_metrics_collector

            from horde_worker_regen.process_management.ipc.messages import HordeJobMetricsMessage

            self.process_message_queue.put(
                HordeJobMetricsMessage(
                    process_id=self.process_id,
                    process_launch_identifier=self.process_launch_identifier,
                    info=f"Job metrics for {form.form_id}",
                    job_id=form.form_id,
                    is_alchemy=True,
                    phase_metrics=get_metrics_collector().snapshot_and_reset_job(),
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to send alchemy job metrics: {type(e).__name__} {e}")

    def start_alchemy(self, form: AlchemyFormSpec) -> None:
        """Run a non-graph alchemy form (caption/interrogation/nsfw/vectorize/palette/describe) and report it.

        These forms produce a text/JSON result rather than an image, so they run on the safety
        process (which already owns the CLIP stack) instead of an inference process, and their
        result is delivered in the submit payload without an R2 image upload.
        """
        self.send_process_state_change_message(
            process_state=HordeProcessState.ALCHEMY_STARTING,
            info=f"Starting alchemy form {form.form} ({form.form_id})",
        )

        time_start = time.time()
        state = GENERATION_STATE.faulted
        result_payload: dict | None = None

        try:
            image = PIL.Image.open(BytesIO(base64.b64decode(form.source_image_base64)))

            if is_caption_form(form.form):
                self._ensure_caption_model()
                caption = self._interrogator.generate_caption(image)  # type: ignore[attr-defined]
                result_payload = {"caption": caption}
            elif is_interrogator_form(form.form):
                result_payload = {"interrogation": self._interrogate_image(image)}
            elif is_nsfw_detector_form(form.form):
                nsfw_result = self._nsfw_checker.check_for_nsfw(image=image)  # type: ignore[attr-defined]
                if nsfw_result is None:
                    raise RuntimeError("NSFW check returned no result")
                result_payload = {"nsfw": nsfw_result.is_nsfw}
            elif is_vectorize_form(form.form):
                result_payload = {"vectorize": self._vectorize_image(image)}
            elif is_palette_form(form.form):
                result_payload = {"palette": self._extract_palette(image)}
            elif is_describe_form(form.form):
                result_payload = {"describe": self._describe_image(image)}
            elif is_aesthetic_form(form.form):
                aesthetic_score = self._score_aesthetic(image)
                if aesthetic_score is None:
                    raise RuntimeError("Aesthetic predictor is unavailable")
                result_payload = {"aesthetic": aesthetic_score}
            else:
                raise ValueError(f"Unknown alchemy form for safety process: {form.form}")

            state = GENERATION_STATE.ok
        except Exception as e:
            logger.error(f"Alchemy form {form.form} ({form.form_id}) failed: {type(e).__name__} {e}")

        self.process_message_queue.put(
            HordeAlchemyResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=f"Alchemy form {form.form} ({form.form_id})",
                time_elapsed=time.time() - time_start,
                form_id=form.form_id,
                form=form.form,
                state=state,
                result_payload=result_payload,
            ),
        )
        self._send_alchemy_job_metrics(form)

        process_state = (
            HordeProcessState.ALCHEMY_COMPLETE if state == GENERATION_STATE.ok else HordeProcessState.ALCHEMY_FAILED
        )
        self.send_process_state_change_message(
            process_state=process_state,
            info=f"Finished alchemy form {form.form} ({form.form_id})",
        )
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        if isinstance(message, HordeAlchemyControlMessage):
            if message.control_flag != HordeControlFlag.START_ALCHEMY:
                raise ValueError(f"Expected {HordeControlFlag.START_ALCHEMY}, got {message.control_flag}")
            if self._dry_run_skip_safety:
                logger.info(f"Dry-run: skipping alchemy form {message.form.form} ({message.form.form_id})")
                self.process_message_queue.put(
                    HordeAlchemyResultMessage(
                        process_id=self.process_id,
                        process_launch_identifier=self.process_launch_identifier,
                        info="Dry-run alchemy form",
                        time_elapsed=0.0,
                        form_id=message.form.form_id,
                        form=message.form.form,
                        state=GENERATION_STATE.ok,
                        result_payload={message.form.form: "dry-run"},
                    ),
                )
                self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")
                return
            self.start_alchemy(message.form)
            return

        if not isinstance(message, HordeSafetyControlMessage):
            raise TypeError(f"Expected {HordeSafetyControlMessage}, got {type(message)}")

        if message.control_flag != HordeControlFlag.EVALUATE_SAFETY:
            raise ValueError(f"Expected {HordeControlFlag.EVALUATE_SAFETY}, got {message.control_flag}")

        if self._dry_run_skip_safety:
            logger.info(f"Dry-run: skipping safety evaluation for job {message.job_id}")
            self.process_message_queue.put(
                HordeSafetyResultMessage(
                    process_id=self.process_id,
                    process_launch_identifier=self.process_launch_identifier,
                    info=f"Dry-run safety evaluation for job {message.job_id}",
                    time_elapsed=0.0,
                    job_id=message.job_id,
                    safety_evaluations=[
                        HordeSafetyEvaluation(is_nsfw=False, is_csam=False, replacement_image_base64=None)
                        for _ in message.images_base64
                    ],
                ),
            )
            self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")
            return

        self.send_memory_report_message(include_vram=False)

        time_start = time.time()

        logger.info(
            f"Horde safety process received job {message.job_id}. Number of images: {len(message.images_base64)}",
        )

        safety_evaluations: list[HordeSafetyEvaluation] = []

        for image_base64 in message.images_base64:
            # Decode the image from base64
            image_bytes = BytesIO(base64.b64decode(image_base64))
            try:
                image_as_pil = PIL.Image.open(image_bytes)
            except Exception as e:
                logger.error(f"Failed to open image: {type(e).__name__} {e}")
                safety_evaluations.append(
                    HordeSafetyEvaluation(
                        is_nsfw=True,
                        is_csam=True,
                        replacement_image_base64=None,
                        failed=True,
                    ),
                )

                continue

            nsfw_result: NSFWResult | None = self._nsfw_checker.check_for_nsfw(
                image=image_as_pil,
                prompt=message.prompt,
                model_info=message.horde_model_info.model_dump() if message.horde_model_info is not None else None,
            )

            if nsfw_result is None:
                raise RuntimeError("NSFW result is None")

            replacement_image_base64: str | None = None

            if nsfw_result.is_csam:
                replacement_image_base64 = self.censor_csam_image_base64
                logger.debug(f"CSAM detected in image {message.job_id}. Image is deleted.")
            elif message.sfw_worker and nsfw_result.is_nsfw:
                replacement_image_base64 = self.censor_sfw_worker_image_base64
                logger.info(f"SFW worker detected NSFW in image {message.job_id}.")
            elif message.censor_nsfw and nsfw_result.is_nsfw:
                replacement_image_base64 = self.censor_sfw_request_image_base64
                logger.info(f"Censor list detected NSFW in image {message.job_id}.")

            # The aesthetic score rides on the same CLIP embedding the NSFW check just used, so it is
            # cheap to attach here. It is scored on the original image (not any censored replacement)
            # so the metadata describes what was actually generated.
            aesthetic_score = self._score_aesthetic(image_as_pil) if message.include_aesthetic_score else None

            safety_evaluations.append(
                HordeSafetyEvaluation(
                    is_nsfw=nsfw_result.is_nsfw,
                    is_csam=nsfw_result.is_csam,
                    replacement_image_base64=replacement_image_base64,
                    aesthetic_score=aesthetic_score,
                ),
            )

        time_elapsed = time.time() - time_start

        info_message = f"Finished evaluating safety for job {message.job_id}"
        logger.info(info_message)

        self.process_message_queue.put(
            HordeSafetyResultMessage(
                process_id=self.process_id,
                process_launch_identifier=self.process_launch_identifier,
                info=info_message,
                time_elapsed=time_elapsed,
                job_id=message.job_id,
                safety_evaluations=safety_evaluations,
            ),
        )
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    @override
    def cleanup_for_exit(self) -> None:
        return
