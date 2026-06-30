"""Constants for the reGen bridge."""

BRIDGE_CONFIG_FILENAME = "bridgeData.yaml"

VERSION_META_REMOTE_URL = (
    "https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/horde_worker_regen/_version_meta.json"
)


KNOWN_SLOW_MODELS_DIFFICULTIES = {"Stable Cascade 1.0": 6.0, "Flux.1-Schnell fp8 (Compact)": 6.0}
VRAM_HEAVY_MODELS = ["Stable Cascade 1.0", "Flux.1-Schnell fp16 (Compact)", "Flux.1-Schnell fp8 (Compact)"]
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

WORKER_KNOWN_EXTRA_ALCHEMY_FORMS = frozenset({VECTORIZE_FORM_NAME})
"""Alchemy forms this worker serves that the currently published SDK enum does not yet list."""


def is_vectorize_form(form: str) -> bool:
    """Return whether *form* is the image vectorizer form name."""
    return form == VECTORIZE_FORM_NAME


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
