"""Constants for the reGen bridge."""

from typing import Protocol

from horde_sdk.ai_horde_api.apimodels import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from horde_sdk.worker.dispatch.ai_horde.image.convert import (
    AI_HORDE_EXTENDED_IMAGE_CONTROL_TYPES,
    AI_HORDE_LEGACY_IMAGE_CONTROL_TYPES,
)

BRIDGE_CONFIG_FILENAME = "bridgeData.yaml"

CLASSIC_CONTROL_TYPES = frozenset(control_type.value for control_type in AI_HORDE_LEGACY_IMAGE_CONTROL_TYPES)
"""The classic controlnet control types every controlnet-capable worker serves.

The SDK's AI Horde adapter owns this protocol vocabulary, including the distinction between the legacy
``hough`` spelling and the extended ``mlsd`` spelling.
"""

EXTENDED_CONTROL_TYPES = frozenset(control_type.value for control_type in AI_HORDE_EXTENDED_IMAGE_CONTROL_TYPES)
"""The extended controlnet control types (everything the annotators can produce beyond the classic set).

A worker only advertises ``allow_extended_controlnet`` once its annotators cover this whole set, because the
per-pop offer is a single boolean and the server may then dispatch any extended control type to it.
"""

VERSION_META_REMOTE_URL = (
    "https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/horde_worker_regen/_version_meta.json"
)


KNOWN_SLOW_MODELS_DIFFICULTIES = {"Stable Cascade 1.0": 6.0, "Flux.1-Schnell fp8 (Compact)": 6.0}

VRAM_HEAVY_MODELS = ["Stable Cascade 1.0", "Flux.1-Schnell fp16 (Compact)", "Flux.1-Schnell fp8 (Compact)"]
"""Checkpoints classified as "very large" by name rather than baseline.

The named-checkpoint escape hatch folded into
:func:`~horde_worker_regen.process_management.models.model_sizing.model_size_tier`, which is the single
authority for "very large" (whole-card) classification across the worker. Prefer the tier predicates
(:func:`~horde_worker_regen.process_management.models.model_size_tier`,
:func:`~horde_worker_regen.process_management.models.model_sizing.is_extra_large_model`) over reading this
list directly, so a model's size is asked the same way everywhere.
"""
KNOWN_SLOW_WORKFLOWS = {"qr_code": 2.0}
KNOWN_CONTROLNET_WORKFLOWS = {"qr_code": 2.0}

BASE_LORA_DOWNLOAD_TIMEOUT = 60
EXTRA_LORA_DOWNLOAD_TIMEOUT = 30
MAX_LORAS = 5

TOTAL_LORA_DOWNLOAD_TIMEOUT = BASE_LORA_DOWNLOAD_TIMEOUT + (EXTRA_LORA_DOWNLOAD_TIMEOUT * MAX_LORAS)

MAX_SOURCE_IMAGE_RETRIES = 5

VECTORIZE_FORM_NAME = "vectorize"
"""The on-wire alchemy form name for the image vectorizer (raster -> SVG).

Defined worker-side (rather than taken only from horde_sdk's ``KNOWN_ALCHEMY_FORMS``) so the worker
can serve the form against the currently published SDK. The pop/async wire models already accept
unknown form names as plain strings (warn-only), but the bridge-data ``forms`` config validator in
the SDK hard-rejects unknown forms, so the worker re-validates ``forms`` against the SDK enum *plus*
this worker-known set (see ``reGenBridgeData.validate_alchemy_forms``). The matching SDK enum member
is added in parallel so the form is first-class once the SDK ships.
"""

PALETTE_FORM_NAME = "palette"
"""The on-wire alchemy form name for dominant-colour palette extraction (raster -> colour list).

A text-output, model-free form in the same family as :data:`VECTORIZE_FORM_NAME`: it runs on the
safety process and returns its result inline (no R2 upload). It has no optional dependency (the
palette is computed with Pillow, always present), so unlike vectorize it needs no availability probe.
"""

DESCRIBE_FORM_NAME = "describe"
"""The on-wire alchemy form name for the cheap technical-metadata bundle (blurhash, perceptual hash,
dimensions, dominant colour, alpha).

A text-output form like :data:`VECTORIZE_FORM_NAME`. Its blurhash/perceptual-hash pieces need the
worker-only ``describe`` extra, so it is gated on an availability probe (see
``capabilities.describe_available``).
"""

AESTHETIC_FORM_NAME = "aesthetic"
"""The on-wire alchemy form name for the LAION aesthetic score (raster -> 0-10 quality float).

A text-output form like :data:`VECTORIZE_FORM_NAME`, but model-backed: it runs on the safety process,
reusing the CLIP ViT-L/14 embedding that process already computes plus a small MLP head (see
``process_management.workers.aesthetic_predictor``). It needs no worker-only optional dependency (the
safety process always has torch and CLIP), so it is gated only on server support, like
:data:`PALETTE_FORM_NAME`. The same scorer also feeds the per-generation aesthetic ``gen_metadata``.
"""

WORKER_KNOWN_EXTRA_ALCHEMY_FORMS = frozenset(
    {VECTORIZE_FORM_NAME, PALETTE_FORM_NAME, DESCRIBE_FORM_NAME, AESTHETIC_FORM_NAME},
)
"""Alchemy forms this worker serves that the currently published SDK enum does not yet list.

The bridge-data ``forms`` config validator in the SDK hard-rejects unknown forms; the worker
re-validates against the SDK enum *plus* this set (see ``reGenBridgeData.validate_alchemy_forms``) so
a config can list these forms before an SDK release that adds them.
"""


def is_vectorize_form(form: str) -> bool:
    """Return whether *form* is the image vectorizer form name."""
    return form == VECTORIZE_FORM_NAME


def is_palette_form(form: str) -> bool:
    """Return whether *form* is the colour-palette extraction form name."""
    return form == PALETTE_FORM_NAME


def is_describe_form(form: str) -> bool:
    """Return whether *form* is the technical-metadata (describe) form name."""
    return form == DESCRIBE_FORM_NAME


def is_aesthetic_form(form: str) -> bool:
    """Return whether *form* is the aesthetic-score form name."""
    return form == AESTHETIC_FORM_NAME


AESTHETIC_METADATA_TYPE = "aesthetic_score"
"""The ``gen_metadata`` ``type`` under which the per-generation aesthetic score is reported.

Defined worker-side so the worker can emit it against a published horde_sdk that predates the matching
``METADATA_TYPE.aesthetic_score`` enum member: ``GenMetadataEntry.type_`` is ``METADATA_TYPE | str`` with
a warn-only validator, so the string round-trips. The SDK member is added in parallel and silences the
validator's warning once released.
"""

GEN_METADATA_REF_MAX_LENGTH = 255
"""The ``GenMetadataEntry.ref`` length the AI-Horde API accepts.

Mirrored from the SDK's field constraint so worker-composed ``ref`` text can be truncated to fit
rather than failing validation at submit time, when the generation is already paid for.
"""


def sampler_truncation_disclosure_ref(*, iterations: int, nominal_steps: int, multiplier: float) -> str:
    """Compose the ``gen_metadata`` ref disclosing that a sampler was stopped at its bound.

    The requester is owed the fact that the sample it received is the solver's best effort rather
    than its converged output, along with the numbers that make the coercion checkable.

    Args:
        iterations: The solver iterations run before the bound stopped the loop.
        nominal_steps: The step count the requested schedule advertised.
        multiplier: hordelib's iteration-budget multiplier, quoted so the disclosure cannot drift
            from the bound that produced it.

    Returns:
        str: The disclosure text, truncated to :data:`GEN_METADATA_REF_MAX_LENGTH`.
    """
    ref = (
        f"adaptive sampler iteration cap: solver truncated at {iterations} iterations "
        f"({multiplier:g}x the {nominal_steps}-step schedule); best-effort converged sample delivered"
    )
    return ref[:GEN_METADATA_REF_MAX_LENGTH]


class SamplerTruncationRecord(Protocol):
    """The shape a sampler-truncation record must present to be disclosed.

    Structural rather than imported so the worker keeps building and running against an engine build
    that predates the bounded sampler, and so the parent can disclose the record a child forwarded over
    IPC using the same code path the child uses for hordelib's own record.
    """

    nominal_steps: int
    iterations: int
    budget_multiplier: float


def sampler_truncation_disclosure(truncation: SamplerTruncationRecord | None) -> list[GenMetadataEntry]:
    """Turn a sampler-truncation record into the disclosure ``gen_metadata`` entry.

    hordelib bounds the one sampler that chooses its own iteration count and delivers the best-effort
    sample rather than letting the job burn indefinitely (see
    ``hordelib.execution.adaptive_sampler_bound``). That coercion changes what the requester receives,
    so it is disclosed on the successful submission. ``METADATA_TYPE.information`` is non-reportable
    (see :attr:`HordeInferenceResultMessage.non_reportable_faults`), so the entry describes the
    generation without counting against the job's fault total.

    Composed here, rather than on either inference path, so the monolithic and the disaggregated
    submissions disclose the same coercion in exactly the same shape.

    Args:
        truncation: The record the engine attached to the result (directly on the monolithic path, or
            forwarded from the sample stage on the disaggregated one), or None if the sampler ran to
            its own completion.

    Returns:
        list[GenMetadataEntry]: One disclosure entry, or an empty list when nothing was truncated.
    """
    if truncation is None:
        return []

    return [
        GenMetadataEntry(
            type=METADATA_TYPE.information,
            value=METADATA_VALUE.see_ref,
            ref=sampler_truncation_disclosure_ref(
                iterations=truncation.iterations,
                nominal_steps=truncation.nominal_steps,
                multiplier=truncation.budget_multiplier,
            ),
        ),
    ]


WORKER_KNOWN_BETA_UPSCALERS = frozenset(
    {
        "4xNomos8kSC",
        "4xLSDIRplus",
        "4xNomosWebPhoto_RealPLKSR",
        "4xNomos2_realplksr_dysample",
        "4xNomos2_hq_dat2",
        "2xModernSpanimationV1",
    },
)
"""Upscaler models this worker can run but whose acceptance depends on the AI-Horde server.

These are distributed as beta via the model-reference pending queue and added to the AI-Horde server's
``KNOWN_POST_PROCESSORS`` only at go-live. The server rejects an entire interrogation pop if it offers
a post-processor the server does not list, so the worker must withhold these names until the server
advertises them (checked via :func:`server_supports_interrogation_form`). The long-standing upscalers
are in every server's enum and are never gated this way. Membership here gates only *offering*; the
weights are resolved separately through hordelib's esrgan beta source.
"""

WORKER_KNOWN_BETA_FACEFIXERS = frozenset(
    {
        "GFPGANv1.3",
        "RestoreFormer",
    },
)
"""Face-restoration models this worker can run but whose acceptance depends on the AI-Horde server.

The face-fixer analogue of :data:`WORKER_KNOWN_BETA_UPSCALERS`: distributed as beta via the
model-reference pending queue (the ``gfpgan`` category) and added to the server's ``KNOWN_POST_PROCESSORS``
only at go-live, so the worker withholds these names until the server advertises them (checked via
:func:`server_supports_interrogation_form`). The long-standing ``GFPGAN``/``CodeFormers`` are in every
server's enum and are never gated. ``RestoreFormer`` loads through hordelib's spandrel core; both weights
are resolved through hordelib's gfpgan beta source.
"""
