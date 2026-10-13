"""Contains the classes to form a safety process, which is responsible for evaluating the safety of images.

The safety process also serves the CLIP-stack alchemy forms (caption, interrogation, nsfw):
it already owns the CLIP interrogator and NSFW checker, so those forms run here rather than
in the (comfy-loaded) inference processes.

An on-GPU safety process shares its card with samplers it does not schedule, so an evaluation can find the
device full. That evaluation is rerun on the CPU in the same process (see ``_evaluating_on_cpu``) and the
process stays an on-GPU safety process for the next one.
"""

import enum
import gc
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
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
from pydantic import BaseModel, ConfigDict

from horde_worker_regen import ASSETS_FOLDER_PATH
from horde_worker_regen.consts import is_aesthetic_form, is_describe_form, is_palette_form, is_vectorize_form
from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.ipc.messages import (
    AlchemyFormSpec,
    HordeAlchemyControlMessage,
    HordeAlchemyResultMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeProcessState,
    HordeSafetyControlMessage,
    HordeSafetyEvaluation,
    HordeSafetyResultMessage,
    UnsupportedControlMessageError,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcess
from horde_worker_regen.utils.oom_signature import is_resource_class_exception

if TYPE_CHECKING:
    from horde_safety.deep_danbooru_model import DeepDanbooruModel
    from horde_safety.interrogate import Interrogator
    from horde_safety.nsfw_checker_class import NSFWChecker, NSFWResult
    from torch import Tensor

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


_UNREADABLE_CHECKPOINT_MARKERS = (
    "PytorchStreamReader",
    "central directory",
    "invalid load key",
    "does not match expected hash",
)
"""Fragments raised when a checkpoint file on disk is unusable and only a re-fetch can help.

The first three are torch deserialization failures on a file that is not a readable archive (truncated
or corrupt); the last is horde_safety's hash-verification failure, which likewise leaves the offending
file on disk to fail identically on every relaunch until it is removed."""


def _discard_unreadable_deep_danbooru_weight(error: BaseException) -> None:
    """Delete a DeepDanbooru weight torch could not open, so the next start re-fetches it.

    A weight left half-written by an interrupted transfer fails deserialization identically on every
    restart, and the parent can only see a child that exits nonzero: the pool is respawned until it is
    declared unrecoverable, with the unreadable file still on disk to fail the same way afterwards.
    Removing it (and the download process's verified marker) converts a permanent crash loop into a
    re-download. Nothing here may raise: this runs while another exception is propagating.
    """
    if not any(marker in str(error) for marker in _UNREADABLE_CHECKPOINT_MARKERS):
        return
    try:
        from horde_worker_regen.process_management.workers.safety_model_prefetch import (
            DEEP_DANBOORU_MARKER_SUFFIX,
            deep_danbooru_model_path,
        )

        weight_path = deep_danbooru_model_path()
        marker_path = weight_path.with_name(weight_path.name + DEEP_DANBOORU_MARKER_SUFFIX)
        marker_path.unlink(missing_ok=True)
        weight_path.unlink(missing_ok=True)
        logger.error(
            f"The DeepDanbooru safety weight at {weight_path} is not a readable checkpoint (an interrupted "
            "or corrupt download). It has been deleted so the worker downloads it again on the next start; "
            "the safety check cannot run until it is present.",
        )
    except Exception as cleanup_error:  # noqa: BLE001 - cleanup must never mask the original failure
        logger.warning(f"Could not discard the unreadable DeepDanbooru weight: {type(cleanup_error).__name__}")


def _clear_safety_accelerator_cache() -> None:
    """Return cached safety allocations to the device without importing torch in the parent."""
    from hordelib.api import clear_accelerator_cache

    clear_accelerator_cache()


_DEVICE_FREE_FIGURE_RE = re.compile(r"of which (?P<free>[\d.]+ \w+) is free")
"""The device-free figure the CUDA allocator states in its out-of-memory text."""


class SafetyEvaluationStage(enum.StrEnum):
    """The steps of a safety evaluation that allocate on the device."""

    CLIP_STAGING = "CLIP staging"
    NSFW_CHECK = "NSFW check"
    AESTHETIC_SCORE = "aesthetic score"


class _DeviceOutOfMemory(BaseModel):
    """A device out-of-memory failure of one evaluation, reduced to what the CPU rerun reports."""

    model_config = ConfigDict(frozen=True)

    stage: SafetyEvaluationStage
    free_figure: str | None
    """The allocator's device-free figure, or None when the error text carries none."""


type _TextFeatureCaches = tuple[list[tuple[Tensor, Tensor]], dict[tuple[str, ...], Tensor], dict[str, Tensor]]


def _safety_text_feature_caches() -> _TextFeatureCaches:
    """Return horde_safety's process-wide CLIP text caches: encoded tokens, per-list and per-text embeddings.

    horde_safety keys these by text alone and exposes no accessor, so its module globals are read directly.
    """
    from horde_safety import interrogate

    return (
        interrogate._cached_tokens,  # noqa: SLF001
        interrogate._cached_text_array_features,  # noqa: SLF001
        interrogate._cached_text_features,  # noqa: SLF001
    )


class _TextFeatureCacheSides:
    """Keep horde_safety's CLIP text caches apart for the device and for the CPU.

    The caches are keyed by text alone, so an embedding computed on the device in half precision would be
    handed to a CPU evaluation, which then fails on the device mismatch (and would otherwise score against
    embeddings a CPU process never computes). One side is live in horde_safety and the other is parked
    here, and ``swap`` exchanges them. The CPU side is kept between fallbacks so only the first one pays for
    encoding the predicate texts.
    """

    def __init__(self) -> None:
        self._parked: _TextFeatureCaches = ([], {}, {})

    def swap(self) -> None:
        """Mutate horde_safety's caches in place so the parked side is live and the live side is parked."""
        tokens, text_array_features, text_features = _safety_text_feature_caches()
        parked_tokens, parked_text_array_features, parked_text_features = self._parked
        self._parked = (list(tokens), dict(text_array_features), dict(text_features))
        tokens[:] = parked_tokens
        text_array_features.clear()
        text_array_features.update(parked_text_array_features)
        text_features.clear()
        text_features.update(parked_text_features)


def _tensor_dtypes(model: object) -> dict[str, object]:
    """Return the dtype of every parameter and buffer of a torch module, by name."""
    return {
        name: tensor.dtype
        for name, tensor in (*model.named_parameters(), *model.named_buffers())  # type: ignore[attr-defined]
    }


def _restore_tensor_dtypes(model: object, dtypes: dict[str, object]) -> None:
    """Mutate a torch module so each parameter and buffer has the dtype recorded for it.

    CLIP on CUDA is a mixed-precision module: open_clip halves the convolution, linear, attention and
    projection weights and leaves the rest in single precision. A blanket ``half()`` would not rebuild that
    layout, so each tensor goes back to its recorded dtype.
    """
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):  # type: ignore[attr-defined]
        recorded = dtypes.get(name)
        if recorded is not None and tensor.dtype != recorded:
            tensor.data = tensor.data.to(dtype=recorded)


class _OnDemandDeepDanbooruModel:
    """Keep DeepDanbooru off-GPU between anime checks while preserving its public surface.

    CLIP is the safety lane's hot resident model. DeepDanbooru is consulted only after CLIP classifies an
    image as anime, so pinning its roughly 600 MB model to the card between jobs needlessly reduces inference
    headroom. This proxy stages the existing horde-safety model for the one call that needs it and returns it
    to CPU immediately afterwards. Unknown attributes delegate to the wrapped model because ``NSFWChecker``
    also reads its tag table.
    """

    def __init__(self, model: DeepDanbooruModel, *, execution_device: str) -> None:
        self._model = model
        self._execution_device = execution_device
        self._resident_device = "cpu"

    def __getattr__(self, name: str) -> object:
        return getattr(self._model, name)

    def evaluate_tensor(self, tensor_to_evaluate: object) -> object:
        """Stage on first use in a safety batch and evaluate on the configured device."""
        if self._resident_device != self._execution_device:
            self._model.to(self._execution_device)  # type: ignore[attr-defined]
            self._model._initial_device = self._execution_device  # type: ignore[attr-defined]  # noqa: SLF001
            self._resident_device = self._execution_device
        return self._model.evaluate_tensor(tensor_to_evaluate)  # type: ignore[arg-type]

    def offload(self) -> None:
        """Return staged weights to CPU after the complete safety job or alchemy form."""
        if self._resident_device == "cpu":
            return
        self._model.to("cpu")  # type: ignore[attr-defined]
        self._model._initial_device = "cpu"  # type: ignore[attr-defined]  # noqa: SLF001
        self._resident_device = "cpu"

    def run_on(self, execution_device: str) -> None:
        """Mutate where later evaluations run, returning any staged weights to CPU first."""
        self.offload()
        self._execution_device = execution_device


class _ClipResidency:
    """Where the interrogator's CLIP weights live between evaluations, switchable by the parent in place.

    CLIP is the safety lane's hot resident model. While *resident* it stays on the execution device. While
    *demoted* it lives in host RAM between calls and is staged onto the device for each evaluation, which
    gives the card most of the safety footprint back within a second without ending the process or its
    CUDA context (the alternative the parent otherwise has). Drives the interrogator's own
    ``clip_model``/``clip_offloaded`` pair so its internal caption path stays consistent.
    """

    def __init__(self, interrogator: object, *, execution_device: str) -> None:
        self._interrogator = interrogator
        self._execution_device = execution_device
        self.resident = True

    def stage(self) -> None:
        """Put CLIP on the execution device for an evaluation (a no-op when it is already there)."""
        if self._execution_device == "cpu" or not getattr(self._interrogator, "clip_offloaded", False):
            return
        self._interrogator.clip_model = self._interrogator.clip_model.to(self._execution_device)  # type: ignore[attr-defined]
        self._interrogator.clip_offloaded = False  # type: ignore[attr-defined]

    def release(self) -> None:
        """After an evaluation, return CLIP to host RAM if it is demoted; keep it on the device if resident."""
        if self.resident or self._execution_device == "cpu":
            return
        if getattr(self._interrogator, "clip_offloaded", False):
            return
        self._interrogator.clip_model = self._interrogator.clip_model.to("cpu")  # type: ignore[attr-defined]
        self._interrogator.clip_offloaded = True  # type: ignore[attr-defined]


class CensorReason(enum.Enum):
    """The reason for censoring an image."""

    CSAM = auto()
    CENSORLIST = auto()
    SFW_REQUEST = auto()
    SFW_WORKER = auto()


class HordeSafetyProcess(HordeProcess):
    """The safety process, which is responsible for evaluating the safety of images."""

    _periodic_report_includes_vram: bool = True
    """Sample VRAM in the interval report so a GPU-resident safety model appears in the attribution snapshot.

    When ``safety_on_gpu`` the safety models (DeepDanbooru, CLIP, and the aesthetic head) load on the GPU and
    hold device memory the parent's committed-VRAM ledger must otherwise account for as an unexplained drift.
    Its allocator stats then flow through the same per-process fields as every other GPU process, with no
    special-casing. When the safety process runs on CPU it never initialises CUDA, so the reporter thread's
    device-init guard keeps the VRAM read withheld (the fields stay None) and only RAM/FDs are reported."""

    _interrogator: Interrogator
    _deep_danbooru_model: DeepDanbooruModel | _OnDemandDeepDanbooruModel

    _nsfw_checker: NSFWChecker

    censor_csam_image_bytes: bytes
    censor_censorlist_image_bytes: bytes
    censor_sfw_request_image_bytes: bytes
    censor_sfw_worker_image_bytes: bytes

    _dry_run_skip_safety: bool

    _safety_device: str = "cpu"
    """The torch device the safety models (and the aesthetic head) run on."""
    _aesthetic_scorer: "AestheticScorer | None" = None
    """Lazily-loaded LAION aesthetic head; built on first use and shared by the form and gen-metadata paths."""
    _aesthetic_unavailable: bool = False
    """Latched once the aesthetic weight cannot be obtained, so the cheap per-image path stops retrying."""
    _clip_residency: "_ClipResidency | None" = None
    """Where CLIP lives between evaluations; None until the interrogator is built (or in dry-run)."""
    _evaluating_on_cpu_now: bool = False
    """Whether an evaluation that ran out of device memory is being rerun on the CPU."""
    _evaluation_stage: SafetyEvaluationStage = SafetyEvaluationStage.CLIP_STAGING
    """The step the evaluation in hand has reached, so a device failure can be attributed to it."""
    _text_feature_cache_sides: "_TextFeatureCacheSides | None" = None
    """The parked side of horde_safety's text caches, built on the first CPU rerun."""

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
                # CLIP is the only safety model held on GPU while idle. DeepDanbooru is used only for images
                # CLIP has already identified as anime, so keep its weights in host RAM and stage them through
                # the small on-demand proxy for that conditional evaluation.
                deep_danbooru = get_deep_danbooru_model(device="cpu" if device != "cpu" else device)
                self._deep_danbooru_model = _OnDemandDeepDanbooruModel(
                    deep_danbooru,
                    execution_device=device,
                )
                self._interrogator = get_interrogator_no_blip(device=device)
                self._clip_residency = _ClipResidency(self._interrogator, execution_device=device)
            except Exception as e:
                logger.error(f"Failed to initialise horde_safety: {type(e).__name__} {e}")
                _discard_unreadable_deep_danbooru_weight(e)
                raise

            try:
                from horde_safety.nsfw_checker_class import NSFWChecker

                self._nsfw_checker = NSFWChecker(
                    self._interrogator,
                    self._deep_danbooru_model,  # pyrefly: ignore [bad-argument-type]
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

    def _set_censor_image(self, reason: CensorReason, image_bytes: bytes) -> None:
        if reason == CensorReason.CSAM:
            self.censor_csam_image_bytes = image_bytes
        elif reason == CensorReason.CENSORLIST:
            self.censor_censorlist_image_bytes = image_bytes
        elif reason == CensorReason.SFW_REQUEST:
            self.censor_sfw_request_image_bytes = image_bytes
        elif reason == CensorReason.SFW_WORKER:
            self.censor_sfw_worker_image_bytes = image_bytes
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
                self._set_censor_image(reason, f.read())

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

    def _offload_transient_models(self) -> None:
        """Return every lazily-loaded safety companion to CPU, retaining only CLIP on the GPU."""
        if self._safety_device == "cpu":
            return

        deep_danbooru = getattr(self, "_deep_danbooru_model", None)
        if isinstance(deep_danbooru, _OnDemandDeepDanbooruModel):
            deep_danbooru.offload()
        if self._caption_model_loaded and self._interrogator.caption_model is not None:  # type: ignore[attr-defined]
            self._interrogator.caption_model = self._interrogator.caption_model.to("cpu")  # type: ignore[attr-defined]
            self._interrogator.caption_offloaded = True  # type: ignore[attr-defined]
        if self._aesthetic_scorer is not None:
            self._aesthetic_scorer.to("cpu")
        if self._clip_residency is not None:
            self._clip_residency.release()

    def _stage_clip(self) -> None:
        """Put demoted CLIP weights back on the device for the evaluation about to run."""
        if self._clip_residency is not None:
            self._clip_residency.stage()

    def _park_clip_in_host_ram(self) -> None:
        """Mutate the interrogator so every CLIP weight is in host RAM and it is marked offloaded.

        ``Module.to`` moves parameters one at a time, so a device move that runs out of memory part-way
        leaves the model split across devices with the interrogator's offload flag never updated.
        """
        self._interrogator.clip_model = self._interrogator.clip_model.to("cpu")  # type: ignore[attr-defined]
        self._interrogator.clip_offloaded = True  # type: ignore[attr-defined]

    def demote_safety_weights(self) -> None:
        """Hold CLIP in host RAM between evaluations, returning its device memory now. The process stays up.

        A release that itself runs out of device memory still ends demoted with a memory report sent.
        """
        if self._clip_residency is None or self._safety_device == "cpu":
            self.send_memory_report_message(include_vram=self._safety_device != "cpu")
            return
        self._clip_residency.resident = False
        logger.info("Safety weights demoted to host RAM; CLIP is staged onto the device per evaluation.")
        try:
            self.release_allocator_cache()
        except Exception as e:
            if not is_resource_class_exception(e):
                raise
        else:
            return
        logger.warning(
            "Releasing the demoted safety weights ran out of device memory. CLIP is held in host RAM and the "
            "process stays up.",
        )
        self._park_clip_in_host_ram()
        self.send_memory_report_message(include_vram=True)

    def promote_safety_weights(self) -> None:
        """Keep CLIP resident on the device again after a demotion.

        A promotion that runs out of device memory leaves the weights demoted, which needs no device
        memory at rest, and reports memory the way a demotion does.
        """
        if self._clip_residency is None or self._safety_device == "cpu":
            return
        self._clip_residency.resident = True
        try:
            self._clip_residency.stage()
        except Exception as e:
            if not is_resource_class_exception(e):
                raise
        else:
            logger.info("Safety weights promoted; CLIP is resident on the device again.")
            self.send_memory_report_message(include_vram=True)
            return
        self._clip_residency.resident = False
        self._park_clip_in_host_ram()
        logger.warning(
            "Promoting the safety weights ran out of device memory. They stay demoted in host RAM and the "
            "process stays up.",
        )
        self.release_allocator_cache()

    @property
    def _evaluation_device(self) -> str:
        """The device the evaluation in hand computes on: the CPU during a rerun, else the safety device."""
        return "cpu" if self._evaluating_on_cpu_now else self._safety_device

    def _device_out_of_memory(self, error: Exception) -> _DeviceOutOfMemory | None:
        """Return a description of ``error`` when it is a device out-of-memory a CPU rerun can answer.

        None means the failure keeps its terminal meaning: it is not an out-of-memory, the process has no
        device to fall back from, or it already is the CPU rerun.
        """
        if self._safety_device == "cpu" or self._clip_residency is None or self._evaluating_on_cpu_now:
            return None
        if not is_resource_class_exception(error):
            return None
        free_figure = _DEVICE_FREE_FIGURE_RE.search(str(error))
        return _DeviceOutOfMemory(
            stage=self._evaluation_stage,
            free_figure=free_figure.group("free") if free_figure is not None else None,
        )

    @contextmanager
    def _evaluating_on_cpu(self) -> Iterator[None]:
        """Context manager that runs its body's evaluation on the CPU in a process initialised on CUDA.

        On entry everything the process holds on the device is returned to it, so the card has the memory
        back before the rerun starts, and each piece of device or precision state an evaluation reads is
        pointed at the CPU:

        - CLIP moves to host RAM in single precision. Its half-precision weights convert exactly, and
          single precision is what a CPU process computes in (half-precision convolutions are not what
          it runs).
        - The interrogator's ``device`` routes the preprocessed image and the tokens.
        - horde_safety's text caches swap to a CPU side, see ``_TextFeatureCacheSides``.
        - DeepDanbooru and the aesthetic head, always single precision, run where they already rest.

        On exit CLIP returns to its recorded per-tensor precision and stays in host RAM marked offloaded,
        whatever the residency flag says, because the device was full a moment ago. The next evaluation's
        staging puts it back, and a resident CLIP then stays on the device as before.
        """
        interrogator = self._interrogator
        deep_danbooru = self._deep_danbooru_model
        execution_device = interrogator.device  # type: ignore[attr-defined]
        if self._text_feature_cache_sides is None:
            self._text_feature_cache_sides = _TextFeatureCacheSides()
        text_feature_cache_sides = self._text_feature_cache_sides

        self._offload_transient_models()
        self._park_clip_in_host_ram()
        clip_dtypes = _tensor_dtypes(interrogator.clip_model)  # type: ignore[attr-defined]
        interrogator.clip_model = interrogator.clip_model.float()  # type: ignore[attr-defined]
        interrogator.clip_offloaded = False  # type: ignore[attr-defined]
        interrogator.device = "cpu"  # type: ignore[attr-defined]
        if isinstance(deep_danbooru, _OnDemandDeepDanbooruModel):
            deep_danbooru.run_on("cpu")
        text_feature_cache_sides.swap()
        self._evaluating_on_cpu_now = True
        gc.collect()
        _clear_safety_accelerator_cache()
        self.send_memory_report_message(include_vram=True)
        try:
            yield
        finally:
            self._evaluating_on_cpu_now = False
            text_feature_cache_sides.swap()
            if isinstance(deep_danbooru, _OnDemandDeepDanbooruModel):
                deep_danbooru.run_on(self._safety_device)
            interrogator.device = execution_device  # type: ignore[attr-defined]
            _restore_tensor_dtypes(interrogator.clip_model, clip_dtypes)  # type: ignore[attr-defined]
            interrogator.clip_offloaded = True  # type: ignore[attr-defined]

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

                self._aesthetic_scorer = load_aesthetic_scorer(device=self._evaluation_device)
            except Exception as e:
                logger.warning(f"Aesthetic scoring unavailable (could not load predictor): {type(e).__name__} {e}")
                self._aesthetic_unavailable = True
                return None
        else:
            self._aesthetic_scorer.to(self._evaluation_device)

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
            with self.periodic_heartbeat(heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE):
                image = PIL.Image.open(BytesIO(form.source_image_bytes))
                self._stage_clip()

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

        # Alchemy may have staged BLIP, DeepDanbooru, the aesthetic head, or large temporary embedding
        # tensors. Return those allocations before advertising the shared safety lane as idle.
        self.release_allocator_cache()

        process_state = (
            HordeProcessState.ALCHEMY_COMPLETE if state == GENERATION_STATE.ok else HordeProcessState.ALCHEMY_FAILED
        )
        self.send_process_state_change_message(
            process_state=process_state,
            info=f"Finished alchemy form {form.form} ({form.form_id})",
        )
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    @logger.catch(reraise=True)
    def release_allocator_cache(self) -> None:
        """Restore the safety lane's CLIP-only idle residency, then report memory.

        CLIP stays hot because every safety check uses it. Conditional/lazy companions (DeepDanbooru, BLIP,
        and the aesthetic head) live on CPU between calls, and cached activation blocks are returned to the
        driver. This makes the process's idle GPU footprint a real fixed safety cost rather than the high-water
        mark of its most recent form. When safety runs on CPU there is no device allocator to trim.
        """
        if self._safety_device == "cpu":
            self.send_memory_report_message(include_vram=False)
            return
        self._offload_transient_models()
        gc.collect()
        _clear_safety_accelerator_cache()
        self.send_memory_report_message(include_vram=True)

    @logger.catch(reraise=True)
    def unload_transient_models_from_ram(self) -> None:
        """Discard optional alchemy companions while retaining the safety lane's base models."""
        if not self._dry_run_skip_safety:
            # ``get_interrogator_no_blip`` builds the interrogator without ever assigning ``caption_model``.
            # The attribute exists only after ``_ensure_caption_model`` has run for a caption form.
            if self._caption_model_loaded:
                caption_model = self._interrogator.caption_model  # type: ignore[attr-defined]
                if caption_model is not None:
                    self._interrogator.caption_model = None  # type: ignore[attr-defined]
                    del caption_model
                self._interrogator.caption_offloaded = False  # type: ignore[attr-defined]
                self._caption_model_loaded = False
            self._label_tables.clear()
            self._ranking_lists = None
            self._aesthetic_scorer = None
            gc.collect()
            if self._safety_device != "cpu":
                _clear_safety_accelerator_cache()

        self.send_process_state_change_message(
            process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
            info="Unloaded transient safety/alchemy models from RAM",
        )
        self.send_memory_report_message(include_vram=self._safety_device != "cpu")
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    def _stage_clip_for_evaluation(self) -> _DeviceOutOfMemory | None:
        """Stage CLIP for an evaluation, returning the failure when the device has no room for it."""
        self._evaluation_stage = SafetyEvaluationStage.CLIP_STAGING
        try:
            self._stage_clip()
        except Exception as e:
            device_failure = self._device_out_of_memory(e)
            if device_failure is None:
                raise
            return device_failure
        return None

    def _evaluate_images(self, message: HordeSafetyControlMessage) -> list[HordeSafetyEvaluation]:
        """Return one evaluation per image of the job, computed on whichever device the models are on."""
        safety_evaluations: list[HordeSafetyEvaluation] = []

        for image_bytes in message.images_bytes:
            try:
                image_as_pil = PIL.Image.open(BytesIO(image_bytes))
            except Exception as e:
                logger.error(f"Failed to open image: {type(e).__name__} {e}")
                safety_evaluations.append(
                    HordeSafetyEvaluation(
                        is_nsfw=True,
                        is_csam=True,
                        replacement_image_bytes=None,
                        failed=True,
                    ),
                )

                continue

            self._evaluation_stage = SafetyEvaluationStage.NSFW_CHECK
            nsfw_result: NSFWResult | None = self._nsfw_checker.check_for_nsfw(
                image=image_as_pil,
                prompt=message.prompt,
                model_info=(message.horde_model_info.model_dump() if message.horde_model_info is not None else None),
            )

            if nsfw_result is None:
                raise RuntimeError("NSFW result is None")

            replacement_image_bytes: bytes | None = None

            if nsfw_result.is_csam:
                replacement_image_bytes = self.censor_csam_image_bytes
                logger.debug(f"CSAM detected in image {message.job_id}. Image is deleted.")
            elif message.sfw_worker and nsfw_result.is_nsfw:
                replacement_image_bytes = self.censor_sfw_worker_image_bytes
                logger.info(f"SFW worker detected NSFW in image {message.job_id}.")
            elif message.censor_nsfw and nsfw_result.is_nsfw:
                replacement_image_bytes = self.censor_sfw_request_image_bytes
                logger.info(f"Censor list detected NSFW in image {message.job_id}.")

            # The aesthetic score rides on the same CLIP embedding the NSFW check just used, so it is
            # cheap to attach here. It is scored on the original image (not any censored replacement)
            # so the metadata describes what was actually generated.
            self._evaluation_stage = SafetyEvaluationStage.AESTHETIC_SCORE
            aesthetic_score = self._score_aesthetic(image_as_pil) if message.include_aesthetic_score else None

            safety_evaluations.append(
                HordeSafetyEvaluation(
                    is_nsfw=nsfw_result.is_nsfw,
                    is_csam=nsfw_result.is_csam,
                    replacement_image_bytes=replacement_image_bytes,
                    aesthetic_score=aesthetic_score,
                ),
            )

        return safety_evaluations

    def _evaluate_images_on_cpu(
        self,
        message: HordeSafetyControlMessage,
        device_failure: _DeviceOutOfMemory,
    ) -> list[HordeSafetyEvaluation]:
        """Return the job's evaluations computed on the CPU after the device ran out of memory for them.

        The whole job is rerun, so no verdict mixes devices. A failure here is not caught: the process
        then ends, as any failed evaluation does.
        """
        rerun_start = time.time()
        with self._evaluating_on_cpu():
            safety_evaluations = self._evaluate_images(message)
        free_figure = f"{device_failure.free_figure} free" if device_failure.free_figure else "free figure not stated"
        logger.warning(
            f"Safety job {message.job_id} ran out of device memory during {device_failure.stage} "
            f"({free_figure}). The check was rerun on the CPU in {time.time() - rerun_start:.2f} seconds and "
            "the process stays up.",
        )
        return safety_evaluations

    def _evaluate_images_with_cpu_fallback(
        self,
        message: HordeSafetyControlMessage,
        device_failure: _DeviceOutOfMemory | None,
    ) -> list[HordeSafetyEvaluation]:
        """Return the job's evaluations, rerunning them on the CPU when the device runs out of memory.

        ``device_failure`` is a staging failure that already ruled the device out. The rerun starts after
        the failed attempt's exception is released: the traceback keeps that attempt's device tensors
        alive, and they must be collectable before the device memory is handed back.
        """
        if device_failure is None:
            try:
                return self._evaluate_images(message)
            except Exception as e:
                device_failure = self._device_out_of_memory(e)
                if device_failure is None:
                    raise
        return self._evaluate_images_on_cpu(message, device_failure)

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        if message.control_flag == HordeControlFlag.RELEASE_ALLOCATOR_CACHE:
            self.release_allocator_cache()
            return

        if message.control_flag == HordeControlFlag.DEMOTE_SAFETY_WEIGHTS:
            self.demote_safety_weights()
            return

        if message.control_flag == HordeControlFlag.PROMOTE_SAFETY_WEIGHTS:
            self.promote_safety_weights()
            return

        if message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            self.unload_transient_models_from_ram()
            return

        if isinstance(message, HordeAlchemyControlMessage):
            if message.control_flag != HordeControlFlag.START_ALCHEMY:
                raise UnsupportedControlMessageError(
                    f"Expected {HordeControlFlag.START_ALCHEMY}, got {message.control_flag}",
                )
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
            raise UnsupportedControlMessageError(f"Expected {HordeSafetyControlMessage}, got {type(message)}")

        if message.control_flag != HordeControlFlag.EVALUATE_SAFETY:
            raise UnsupportedControlMessageError(
                f"Expected {HordeControlFlag.EVALUATE_SAFETY}, got {message.control_flag}",
            )

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
                        HordeSafetyEvaluation(is_nsfw=False, is_csam=False, replacement_image_bytes=None)
                        for _ in message.images_bytes
                    ],
                ),
            )
            self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")
            return

        # The lane serves one evaluation at a time, so its busy state is what tells the parent the job was
        # actually received rather than still sitting in the pipe. The parent marks the same state at the
        # send; this is the child's own confirmation of it, and the terminal WAITING_FOR_JOB releases it.
        self.send_process_state_change_message(
            process_state=HordeProcessState.JOB_RECEIVED,
            info=f"Received safety job {message.job_id}",
        )

        self.send_memory_report_message(include_vram=False)

        # An off-GPU check costs seconds per image and the whole evaluation is one blocking call sequence,
        # during which the process loop publishes nothing. Without a beat the watchdog cannot tell a slow
        # check from a wedged one, so it reads a legitimately long CPU check as a hang.
        with self.periodic_heartbeat(heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE):
            staging_failure = self._stage_clip_for_evaluation()

            time_start = time.time()

            logger.info(
                f"Horde safety process received job {message.job_id}. Number of images: {len(message.images_bytes)}",
            )

            # A CPU rerun is timed with the device attempt it replaces, so the parent's backpressure and
            # grace models read what the job really cost.
            safety_evaluations = self._evaluate_images_with_cpu_fallback(message, staging_failure)

            # Measured inside the heartbeat window: leaving the context joins the reporter thread, and that
            # join must not land in the duration the parent's backpressure and grace models read.
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
        # The evaluation's activation cache is reclaimable, not fixed safety residency. Trim before WAITING
        # so the parent never sees an idle safety lane retaining the previous image's high-water allocation.
        self.release_allocator_cache()
        self.send_process_state_change_message(HordeProcessState.WAITING_FOR_JOB, "Waiting for job")

    @override
    def cleanup_for_exit(self) -> None:
        return
